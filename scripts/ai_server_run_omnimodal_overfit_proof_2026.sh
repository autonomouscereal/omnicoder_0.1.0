#!/usr/bin/env bash
set -euo pipefail

# Scratch-only overfit proof runner for the 2026 omnimodal ledger probe.
# This script intentionally refuses external resume/checkpoint inputs. Eval
# only reads checkpoints produced inside this run directory.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

REPO="${OMNICODER_REPO:-$DEFAULT_REPO}"
WEIGHTS_ROOT="${OMNICODER_WEIGHTS_ROOT:-/home/cereal/omnicoder_2026_work/weights}"
IMAGE="${OMNICODER_DOCKER_IMAGE:-omnicoder:cuda-posttrain-2026}"
PYTHON_BIN="${OMNICODER_OVERFIT_PYTHON:-python}"
HOST_PYTHON_BIN="${OMNICODER_OVERFIT_HOST_PYTHON:-python3}"
BACKEND="${OMNICODER_OVERFIT_BACKEND:-docker}"
DEVICE_MODE="${OMNICODER_OVERFIT_DEVICE_MODE:-p40}"
GPU_DEVICES="${OMNICODER_OVERFIT_GPU_DEVICES:-1}"

RUN_TAG_RAW="${OMNICODER_OVERFIT_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_TAG="${RUN_TAG_RAW//[^A-Za-z0-9_.-]/_}"
OUT_DIR="${OMNICODER_OVERFIT_OUT_DIR:-weights/overfit_proof_2026/omnimodal_overfit_${RUN_TAG}}"

EXAMPLES_PER_MODALITY="${OMNICODER_OVERFIT_EXAMPLES_PER_MODALITY:-10}"
GROUPS_RAW="${OMNICODER_OVERFIT_GROUPS:-text,code_tool,image_ocr,video,audio_tts_music,ledger_all}"
RUN_TRAIN="${OMNICODER_OVERFIT_RUN_TRAIN:-1}"
RUN_EVAL="${OMNICODER_OVERFIT_RUN_EVAL:-1}"
RUN_PREDICT="${OMNICODER_OVERFIT_RUN_PREDICT:-0}"
RUN_SUMMARY="${OMNICODER_OVERFIT_RUN_SUMMARY:-1}"
ALLOW_FALLBACK_MATERIALIZER="${OMNICODER_OVERFIT_ALLOW_FALLBACK_MATERIALIZER:-1}"

NPROC_PER_NODE="${OMNICODER_OVERFIT_NPROC_PER_NODE:-1}"
RANK_DEVICE_MAP="${OMNICODER_OVERFIT_RANK_DEVICE_MAP:-0}"
PLACEMENT_LAYER_COUNTS="${OMNICODER_OVERFIT_PLACEMENT_LAYER_COUNTS:-4}"
PRESET="${OMNICODER_OVERFIT_PRESET:-ledger_probe}"
SEQ_LEN="${OMNICODER_OVERFIT_SEQ_LEN:-128}"
BATCH_SIZE="${OMNICODER_OVERFIT_BATCH_SIZE:-1}"
STEPS="${OMNICODER_OVERFIT_STEPS:-600}"
LEARNING_RATE="${OMNICODER_OVERFIT_LR:-0.0008}"
PRECISION="${OMNICODER_OVERFIT_PRECISION:-fp32}"
INIT_DTYPE="${OMNICODER_OVERFIT_INIT_DTYPE:-fp32}"
LM_LOSS_CHUNK_TOKENS="${OMNICODER_OVERFIT_LM_LOSS_CHUNK_TOKENS:-64}"
DIST_TIMEOUT_SECONDS="${OMNICODER_OVERFIT_DIST_TIMEOUT_SECONDS:-7200}"
CUDA_ALLOC_CONF="${OMNICODER_OVERFIT_CUDA_ALLOC_CONF:-max_split_size_mb:128,expandable_segments:True}"
TOP_K="${OMNICODER_OVERFIT_TOP_K:-8}"
MAX_POSITIONS="${OMNICODER_OVERFIT_MAX_POSITIONS:-12}"
MAX_OUTPUT_TOKENS="${OMNICODER_OVERFIT_MAX_OUTPUT_TOKENS:-64}"
MAX_PROMPT_TOKENS="${OMNICODER_OVERFIT_MAX_PROMPT_TOKENS:-512}"

truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

host_path() {
  local value="$1"
  if [[ "$value" == /workspace/weights/* ]]; then
    printf '%s\n' "$WEIGHTS_ROOT/${value#/workspace/weights/}"
  elif [[ "$value" == /workspace/* ]]; then
    printf '%s\n' "$REPO/${value#/workspace/}"
  elif [[ "$value" == weights/* ]]; then
    printf '%s\n' "$WEIGHTS_ROOT/${value#weights/}"
  elif [[ "$value" == /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$REPO/$value"
  fi
}

container_path() {
  local value="$1"
  if [[ "$value" == "$WEIGHTS_ROOT"/* ]]; then
    printf '%s\n' "/workspace/weights/${value#"$WEIGHTS_ROOT"/}"
  elif [[ "$value" == "$REPO"/* ]]; then
    printf '%s\n' "/workspace/${value#"$REPO"/}"
  elif [[ "$value" == weights/* ]]; then
    printf '%s\n' "/workspace/$value"
  elif [[ "$value" == /workspace/* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$value"
  fi
}

csv_count() {
  local raw="$1"
  local count=0
  local item
  IFS=',' read -r -a items <<< "$raw"
  for item in "${items[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    [[ -n "$item" ]] && count=$((count + 1))
  done
  printf '%s\n' "$count"
}

require_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$name must be a non-negative integer, got: $value" >&2
    exit 2
  fi
}

require_no_external_checkpoint_inputs() {
  local name
  for name in \
    OMNICODER_OVERFIT_CHECKPOINT \
    OMNICODER_OVERFIT_RESUME \
    OMNICODER_OVERFIT_RESUME_CHECKPOINT \
    OMNICODER_RESUME_CHECKPOINT \
    OMNICODER_EVAL_CHECKPOINT \
    OMNICODER_BATCH_PRED_CHECKPOINT
  do
    if [[ -n "${!name:-}" ]]; then
      echo "Refusing external checkpoint input via $name=${!name}" >&2
      echo "This proof must train from scratch and eval only checkpoints created under $HOST_OUT_DIR/ckpt." >&2
      exit 3
    fi
  done
}

HOST_OUT_DIR="$(host_path "$OUT_DIR")"
CONTAINER_OUT_DIR="$(container_path "$HOST_OUT_DIR")"
LOG_DIR="$HOST_OUT_DIR/logs"

mkdir -p "$HOST_OUT_DIR" "$LOG_DIR"
require_no_external_checkpoint_inputs
require_integer OMNICODER_OVERFIT_EXAMPLES_PER_MODALITY "$EXAMPLES_PER_MODALITY"
require_integer OMNICODER_OVERFIT_NPROC_PER_NODE "$NPROC_PER_NODE"
require_integer OMNICODER_OVERFIT_SEQ_LEN "$SEQ_LEN"
require_integer OMNICODER_OVERFIT_BATCH_SIZE "$BATCH_SIZE"
require_integer OMNICODER_OVERFIT_STEPS "$STEPS"
require_integer OMNICODER_OVERFIT_LM_LOSS_CHUNK_TOKENS "$LM_LOSS_CHUNK_TOKENS"

if (( EXAMPLES_PER_MODALITY != 10 )); then
  log "examples_per_modality=$EXAMPLES_PER_MODALITY; default proof contract is 10 rows per group"
fi
if (( NPROC_PER_NODE < 1 )); then
  echo "OMNICODER_OVERFIT_NPROC_PER_NODE must be >= 1" >&2
  exit 2
fi
if [[ "$(csv_count "$PLACEMENT_LAYER_COUNTS")" -ne "$NPROC_PER_NODE" ]]; then
  echo "OMNICODER_OVERFIT_PLACEMENT_LAYER_COUNTS must have one entry per rank: nproc=$NPROC_PER_NODE placement=$PLACEMENT_LAYER_COUNTS" >&2
  exit 2
fi
if [[ "$(csv_count "$RANK_DEVICE_MAP")" -ne "$NPROC_PER_NODE" ]]; then
  echo "OMNICODER_OVERFIT_RANK_DEVICE_MAP must have one entry per rank: nproc=$NPROC_PER_NODE rank_map=$RANK_DEVICE_MAP" >&2
  exit 2
fi

USE_GPU=0
case "${DEVICE_MODE,,}" in
  p40|cuda|gpu) USE_GPU=1 ;;
  cpu) USE_GPU=0 ;;
  auto)
    if command -v nvidia-smi >/dev/null 2>&1; then
      USE_GPU=1
    else
      USE_GPU=0
    fi
    ;;
  *)
    echo "OMNICODER_OVERFIT_DEVICE_MODE must be p40, cuda, gpu, cpu, or auto; got: $DEVICE_MODE" >&2
    exit 2
    ;;
esac

case "${BACKEND,,}" in
  docker|local) ;;
  *)
    echo "OMNICODER_OVERFIT_BACKEND must be docker or local; got: $BACKEND" >&2
    exit 2
    ;;
esac

if [[ "${BACKEND,,}" == "docker" ]]; then
  if [[ "$HOST_OUT_DIR" != "$WEIGHTS_ROOT"/* && "$HOST_OUT_DIR" != "$REPO"/* ]]; then
    echo "Docker backend requires OMNICODER_OVERFIT_OUT_DIR under repo or weights root, got: $HOST_OUT_DIR" >&2
    exit 2
  fi
fi

IFS=',' read -r -a PROOF_GROUP_ARRAY <<< "$GROUPS_RAW"
GROUPS_CLEAN=()
for group in "${PROOF_GROUP_ARRAY[@]}"; do
  group="${group#"${group%%[![:space:]]*}"}"
  group="${group%"${group##*[![:space:]]}"}"
  [[ -n "$group" ]] && GROUPS_CLEAN+=("$group")
done
if (( ${#GROUPS_CLEAN[@]} == 0 )); then
  echo "No overfit proof groups selected." >&2
  exit 2
fi

write_launch_manifest() {
  local status="$1"
  "$HOST_PYTHON_BIN" - "$HOST_OUT_DIR/omnimodal_overfit_launch.json" "$status" <<PY
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
status = sys.argv[2]
payload = {
    "schema": "omnicoder.omnimodal_overfit_launcher_2026.v1",
    "status": status,
    "repo": ${REPO@Q},
    "weights_root": ${WEIGHTS_ROOT@Q},
    "out_dir": ${HOST_OUT_DIR@Q},
    "container_out_dir": ${CONTAINER_OUT_DIR@Q},
    "backend": ${BACKEND@Q},
    "device_mode": ${DEVICE_MODE@Q},
    "gpu_devices": ${GPU_DEVICES@Q},
    "groups": ${GROUPS_RAW@Q}.split(","),
    "examples_per_modality": int(${EXAMPLES_PER_MODALITY@Q}),
    "nproc_per_node": int(${NPROC_PER_NODE@Q}),
    "rank_device_map": ${RANK_DEVICE_MAP@Q},
    "placement_layer_counts": ${PLACEMENT_LAYER_COUNTS@Q},
    "preset": ${PRESET@Q},
    "seq_len": int(${SEQ_LEN@Q}),
    "steps": int(${STEPS@Q}),
    "scratch_only": True,
    "external_checkpoints_allowed": False,
}
out.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_cmd() {
  local label="$1"
  local log_file="$2"
  shift 2
  local -a cmd=("$@")
  local quoted
  printf -v quoted '%q ' "${cmd[@]}"
  mkdir -p "$(dirname "$log_file")"
  {
    printf 'label=%s\n' "$label"
    printf 'command=%s\n' "$quoted"
  } > "${log_file}.cmd"

  if [[ "${BACKEND,,}" == "local" ]]; then
    log "run local: $label"
    (
      cd "$REPO"
      export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
      export TOKENIZERS_PARALLELISM=false
      if (( USE_GPU == 0 )); then
        export CUDA_VISIBLE_DEVICES=""
      elif [[ -n "$GPU_DEVICES" ]]; then
        export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
      fi
      "${cmd[@]}"
    ) 2>&1 | tee "$log_file"
    return "${PIPESTATUS[0]}"
  fi

  local container="omnicoder_overfit_${RUN_TAG}_${label//[^A-Za-z0-9_.-]/_}_$$"
  local -a docker_args=(
    run --rm
    --name "$container"
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    -e PYTHONPATH=/workspace/src
    -e TOKENIZERS_PARALLELISM=false
    -e NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    -e NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    -e NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
    -e TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
    -e PYTORCH_CUDA_ALLOC_CONF="$CUDA_ALLOC_CONF"
    -e OMNICODER2026_DIST_TIMEOUT_SECONDS="$DIST_TIMEOUT_SECONDS"
    -e OMNICODER2026_LM_LOSS_CHUNK_TOKENS="$LM_LOSS_CHUNK_TOKENS"
    -e OMNICODER_REQUIRE_HF_TOKENIZER="${OMNICODER_REQUIRE_HF_TOKENIZER:-1}"
    -v "$REPO:/workspace"
    -v "$WEIGHTS_ROOT:/workspace/weights"
    -v /home/cereal:/home/cereal:ro
    -w /workspace
  )
  if (( USE_GPU == 1 )); then
    docker_args+=(--gpus "device=${GPU_DEVICES}" -e CUDA_DEVICE_ORDER=PCI_BUS_ID)
  else
    docker_args+=(-e CUDA_VISIBLE_DEVICES="")
  fi
  docker_args+=("$IMAGE" bash -lc "set -euo pipefail; $quoted")

  log "run docker: $label container=$container"
  docker "${docker_args[@]}" 2>&1 | tee "$log_file"
  return "${PIPESTATUS[0]}"
}

write_fallback_materializer() {
  local target="$HOST_OUT_DIR/fallback_materialize_omnimodal_overfit_2026.py"
  cat > "$target" <<'PY'
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from omnicoder.tokenization.omni_ledger_2026 import DEFAULT_LEDGER
from omnicoder.training.simple_tokenizer import get_text_tokenizer

SCHEMA = "omnicoder.omnimodal_overfit_proof_2026.v1"
GROUPS = ("text", "code_tool", "image_ocr", "video", "audio_tts_music", "ledger_all")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")


def encode(tokenizer: Any, text: str) -> list[int]:
    ids = [int(token_id) if 0 <= int(token_id) < 128000 else 1 for token_id in tokenizer.encode(text)]
    return ids or [1]


def span(family: str, seed: int, count: int = 8) -> list[int]:
    lo, hi = DEFAULT_LEDGER.as_config_ranges()[family]
    width = int(hi) - int(lo)
    return [int(lo) + ((int(seed) * 37 + index * 11) % width) for index in range(count)]


def row(group: str, modality: str, prompt: str, target_ids: list[int], family: str, target: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "group": group,
        "modality": modality,
        "prompt": prompt,
        "target": target,
        "prompt_token_ids": encode(get_text_tokenizer(prefer_hf=True), prompt),
        "target_token_ids": [int(token_id) for token_id in target_ids],
        "source_id": f"omnimodal_overfit_{group}_{family}",
        "source_uri": "local://omnicoder/overfit_proof_2026",
        "source_date": "2026-05-28",
        "license": "internal diagnostic proof rows",
        "split": "train",
        "quality_score": 1.0,
        "contamination_status": "clean",
        "task_type": "overfit_trainability_proof",
        "diagnostic_only": True,
        "target_ledger_family": family,
        "valid_target_tokens": len(target_ids),
    }


def main() -> int:
    out = Path(sys.argv[1])
    examples = int(sys.argv[2])
    tokenizer = get_text_tokenizer(prefer_hf=True)
    data_dir = out / "data"
    tasks_dir = out / "tasks"
    ranges = DEFAULT_LEDGER.as_config_ranges()
    groups: dict[str, list[dict[str, Any]]] = {}
    groups["text"] = [
        row("text", "text", f"user: remember text proof {i}\nassistant:", encode(tokenizer, f" text_proof_answer_{i} stable"), "text", f"text_proof_answer_{i} stable")
        for i in range(examples)
    ]
    groups["code_tool"] = [
        row("code_tool", "code" if i % 2 == 0 else "tool", f"user: code tool proof {i}\nassistant:", encode(tokenizer, f" def proof_{i}(): return {i}") if i % 2 == 0 else span("tool_agent", i), "text" if i % 2 == 0 else "tool_agent", f"proof_{i}")
        for i in range(examples)
    ]
    groups["image_ocr"] = [
        row("image_ocr", "image" if i % 2 == 0 else "ocr", f"user: image ocr proof {i}\nassistant:", span("vision_semantic", i, 6) + span("vision_residual", i, 4) if i % 2 == 0 else encode(tokenizer, f" OCR_TEXT_PROOF_{i}"), "vision_semantic" if i % 2 == 0 else "text", f"image_or_ocr_proof_{i}")
        for i in range(examples)
    ]
    groups["video"] = [
        row("video", "video", f"user: video temporal proof {i}\nassistant:", span("vision_semantic", i, 5) + span("time_space", i, 5), "time_space", f"video_proof_{i}")
        for i in range(examples)
    ]
    audio_families = ("speech_tts", "audio_music", "music_control")
    groups["audio_tts_music"] = [
        row("audio_tts_music", "tts" if i % 3 == 0 else "audio" if i % 3 == 1 else "music", f"user: audio tts music proof {i}\nassistant:", span(audio_families[i % 3], i, 8), audio_families[i % 3], f"audio_music_proof_{i}")
        for i in range(examples)
    ]
    families = list(ranges)
    groups["ledger_all"] = [
        row("ledger_all", "tool" if family == "tool_agent" else "image" if family.startswith("vision") else "text", f"user: ledger family {family} proof\nassistant:", span(family, i, 8), family, f"{family}_proof")
        for i, family in enumerate(families[:examples])
    ]
    manifest_groups = []
    for group, rows in groups.items():
        write_jsonl(data_dir / f"{group}.jsonl", rows)
        write_jsonl(
            tasks_dir / f"{group}.jsonl",
            [
                {
                    "benchmark_id": f"local_{group}_overfit",
                    "task_id": f"{group}_{index:02d}",
                    "reportable": False,
                    "prompt": item["prompt"],
                    "source": "local_overfit_2026",
                    "task_format": "local_overfit",
                    "output_modality": item["modality"],
                    "target_modality": item["modality"],
                    "output_field": "prediction" if item["modality"] in {"text", "code", "tool", "ocr"} else "generated_artifact",
                }
                for index, item in enumerate(rows)
            ],
        )
        manifest_groups.append(
            {
                "group": group,
                "rows": len(rows),
                "data": str(data_dir / f"{group}.jsonl"),
                "tasks": str(tasks_dir / f"{group}.jsonl"),
                "target_families": dict(sorted(Counter(str(item["target_ledger_family"]) for item in rows).items())),
            }
        )
    write_json(
        out / "omnimodal_overfit_manifest.json",
        {
            "schema": SCHEMA,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "examples_per_modality": examples,
            "groups": manifest_groups,
            "ledger": DEFAULT_LEDGER.as_metadata(),
            "fallback_materializer": True,
        },
    )
    print(json.dumps({"status": "ok", "manifest": str(out / "omnimodal_overfit_manifest.json"), "fallback_materializer": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
  printf '%s\n' "$target"
}

write_fallback_summary() {
  local target="$HOST_OUT_DIR/omnimodal_overfit_summary.json"
  "$HOST_PYTHON_BIN" - "$HOST_OUT_DIR" "$target" "${GROUPS_CLEAN[@]}" <<'PY'
import json
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
groups = sys.argv[3:]
summary = {"schema": "omnicoder.omnimodal_overfit_proof_2026.summary.v1", "run": str(run), "status": "passed", "groups": {}}
for group in groups:
    data = run / "data" / f"{group}.jsonl"
    ckpt = run / "ckpt" / group
    loss = run / "eval" / f"{group}.loss.json"
    targets = run / "eval" / f"{group}.targets.json"
    item = {
        "data": str(data),
        "checkpoint": str(ckpt),
        "loss": str(loss),
        "targets": str(targets),
        "data_exists": data.exists() and data.stat().st_size > 0,
        "checkpoint_complete": (ckpt / ".complete.json").exists(),
        "loss_exists": loss.exists() and loss.stat().st_size > 0,
        "targets_exists": targets.exists() and targets.stat().st_size > 0,
    }
    if data.exists():
        item["rows"] = sum(1 for line in data.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())
    if loss.exists():
        try:
            item["loss_json"] = json.loads(loss.read_text(encoding="utf-8"))
        except Exception as exc:
            item["loss_error"] = repr(exc)
    if targets.exists():
        try:
            item["target_json"] = json.loads(targets.read_text(encoding="utf-8"))
        except Exception as exc:
            item["target_error"] = repr(exc)
    if not (item["data_exists"] and item["checkpoint_complete"] and item["loss_exists"] and item["targets_exists"]):
        summary["status"] = "failed"
    summary["groups"][group] = item
out.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"status": summary["status"], "out": str(out)}, sort_keys=True))
PY
}

checkpoint_is_owned_and_complete() {
  local checkpoint="$1"
  local expected_world_size="$2"
  [[ "$checkpoint" == "$HOST_OUT_DIR/ckpt/"* ]] || return 1
  [[ -d "$checkpoint" ]] || return 1
  [[ -s "$checkpoint/manifest.json" ]] || return 1
  [[ -s "$checkpoint/.complete.json" ]] || return 1
  grep -Eq "\"world_size\"[[:space:]]*:[[:space:]]*$expected_world_size([,[:space:]}]|$)" "$checkpoint/manifest.json" || return 1
  local rank rank_file marker count
  for ((rank=0; rank<expected_world_size; rank++)); do
    printf -v rank_file 'rank%05d.pt' "$rank"
    marker="$checkpoint/${rank_file}.complete.json"
    [[ -s "$checkpoint/$rank_file" ]] || return 1
    [[ -s "$marker" ]] || return 1
  done
  count="$(find "$checkpoint" -maxdepth 1 -type f -name 'rank*.pt' | wc -l)"
  [[ "$count" -eq "$expected_world_size" ]] || return 1
}

materialize_proof_data() {
  local log_file="$LOG_DIR/materialize.log"
  local -a cmd=(
    "$PYTHON_BIN" -m omnicoder.eval.omnimodal_overfit_proof_2026
    materialize
    --out "$CONTAINER_OUT_DIR"
    --examples-per-modality "$EXAMPLES_PER_MODALITY"
  )
  if run_cmd materialize "$log_file" "${cmd[@]}"; then
    return 0
  fi
  if ! truthy "$ALLOW_FALLBACK_MATERIALIZER"; then
    return 1
  fi
  log "module materializer failed; using script-local fallback materializer"
  local fallback_host
  fallback_host="$(write_fallback_materializer)"
  run_cmd materialize_fallback "$LOG_DIR/materialize_fallback.log" "$PYTHON_BIN" "$(container_path "$fallback_host")" "$CONTAINER_OUT_DIR" "$EXAMPLES_PER_MODALITY"
}

train_group() {
  local group="$1"
  local data="$CONTAINER_OUT_DIR/data/${group}.jsonl"
  local ckpt="$CONTAINER_OUT_DIR/ckpt/${group}"
  local -a cmd=(
    "$PYTHON_BIN" -m torch.distributed.run
    --standalone
    --nproc_per_node="$NPROC_PER_NODE"
    -m omnicoder.training.pipeline_pretrain_2026_dense
    --data "$data"
    --out "$ckpt"
    --log_file "$CONTAINER_OUT_DIR/logs/${group}.train.jsonl"
    --train_diagnostics_file "$CONTAINER_OUT_DIR/logs/${group}.diag.rank{rank}.jsonl"
    --preset "$PRESET"
    --allow_probe
    --placement_layer_counts "$PLACEMENT_LAYER_COUNTS"
    --rank_device_map "$RANK_DEVICE_MAP"
    --pipeline_schedule gpipe
    --pipeline_microbatches 1
    --batch_size "$BATCH_SIZE"
    --seq_len "$SEQ_LEN"
    --steps "$STEPS"
    --lr "$LEARNING_RATE"
    --max_records "$EXAMPLES_PER_MODALITY"
    --precision "$PRECISION"
    --init_dtype "$INIT_DTYPE"
    --optimizer_in_backward_update lowmem_adafactor
    --lm_loss_chunk_tokens "$LM_LOSS_CHUNK_TOKENS"
    --target_boundary_weight 2
    --target_prefix_weight 2
    --target_prefix_tokens 2
    --require_target_contract
    --allow_p40_target_contract_eval
    --no_shuffle
    --checkpoint_sync_backend filesystem
    --dist_timeout_seconds "$DIST_TIMEOUT_SECONDS"
  )
  run_cmd "train_${group}" "$LOG_DIR/${group}.train.console.log" "${cmd[@]}"
  if ! checkpoint_is_owned_and_complete "$HOST_OUT_DIR/ckpt/$group" "$NPROC_PER_NODE"; then
    echo "Scratch checkpoint did not pass completeness/ownership checks: $HOST_OUT_DIR/ckpt/$group" >&2
    exit 5
  fi
}

eval_group() {
  local group="$1"
  local data="$CONTAINER_OUT_DIR/data/${group}.jsonl"
  local ckpt="$CONTAINER_OUT_DIR/ckpt/${group}"
  local -a loss_cmd=(
    "$PYTHON_BIN" -m torch.distributed.run
    --standalone
    --nproc_per_node="$NPROC_PER_NODE"
    -m omnicoder.eval.pipeline_sample_loss_2026
    --checkpoint "$ckpt"
    --data "$data"
    --out "$CONTAINER_OUT_DIR/eval/${group}.loss.json"
    --preset "$PRESET"
    --rank_device_map "$RANK_DEVICE_MAP"
    --placement_layer_counts "$PLACEMENT_LAYER_COUNTS"
    --precision "$PRECISION"
    --init-dtype "$INIT_DTYPE"
    --seq-len "$SEQ_LEN"
    --max-records-per-file "$EXAMPLES_PER_MODALITY"
    --lm-loss-chunk-tokens "$LM_LOSS_CHUNK_TOKENS"
  )
  run_cmd "loss_${group}" "$LOG_DIR/${group}.loss.console.log" "${loss_cmd[@]}"

  local -a target_cmd=(
    "$PYTHON_BIN" -m torch.distributed.run
    --standalone
    --nproc_per_node="$NPROC_PER_NODE"
    -m omnicoder.eval.pipeline_target_token_diagnostics_2026
    --checkpoint "$ckpt"
    --data "$data"
    --out "$CONTAINER_OUT_DIR/eval/${group}.targets.json"
    --preset "$PRESET"
    --rank-device-map "$RANK_DEVICE_MAP"
    --placement-layer-counts "$PLACEMENT_LAYER_COUNTS"
    --precision "$PRECISION"
    --init-dtype "$INIT_DTYPE"
    --seq-len "$SEQ_LEN"
    --max-records-per-file "$EXAMPLES_PER_MODALITY"
    --top-k "$TOP_K"
    --max-positions "$MAX_POSITIONS"
  )
  run_cmd "targets_${group}" "$LOG_DIR/${group}.targets.console.log" "${target_cmd[@]}"
}

predict_group() {
  local group="$1"
  local tasks="$CONTAINER_OUT_DIR/tasks/${group}.jsonl"
  local ckpt="$CONTAINER_OUT_DIR/ckpt/${group}"
  local -a cmd=(
    "$PYTHON_BIN" -m omnicoder.eval.pipeline_checkpoint_batch_predict_2026
    --checkpoint "$ckpt"
    --tasks "$tasks"
    --out "$CONTAINER_OUT_DIR/eval/${group}.predictions.jsonl"
    --summary "$CONTAINER_OUT_DIR/eval/${group}.prediction_summary.json"
    --model "$ckpt"
    --preset "$PRESET"
    --nproc-per-node "$NPROC_PER_NODE"
    --rank-device-map "$RANK_DEVICE_MAP"
    --placement-layer-counts "$PLACEMENT_LAYER_COUNTS"
    --precision "$PRECISION"
    --init-dtype "$INIT_DTYPE"
    --max-output-tokens "$MAX_OUTPUT_TOKENS"
    --max-prompt-tokens "$MAX_PROMPT_TOKENS"
    --dist-timeout-seconds "$DIST_TIMEOUT_SECONDS"
    --allow-local-dev-tasks
    --force
  )
  run_cmd "predict_${group}" "$LOG_DIR/${group}.predict.console.log" "${cmd[@]}"
}

summarize_run() {
  local -a cmd=(
    "$PYTHON_BIN" -m omnicoder.eval.omnimodal_overfit_proof_2026
    summary
    --run "$CONTAINER_OUT_DIR"
    --out "$CONTAINER_OUT_DIR/omnimodal_overfit_summary.json"
  )
  if run_cmd summary "$LOG_DIR/summary.log" "${cmd[@]}"; then
    return 0
  fi
  log "module summary failed; writing fallback summary"
  write_fallback_summary | tee "$LOG_DIR/summary_fallback.log"
}

write_launch_manifest running
log "scratch omnimodal overfit proof out=$HOST_OUT_DIR backend=$BACKEND device_mode=$DEVICE_MODE gpu_devices=$GPU_DEVICES"
materialize_proof_data

if truthy "$RUN_TRAIN"; then
  for group in "${GROUPS_CLEAN[@]}"; do
    train_group "$group"
  done
else
  log "training skipped by OMNICODER_OVERFIT_RUN_TRAIN=$RUN_TRAIN"
fi

if truthy "$RUN_EVAL"; then
  for group in "${GROUPS_CLEAN[@]}"; do
    if ! checkpoint_is_owned_and_complete "$HOST_OUT_DIR/ckpt/$group" "$NPROC_PER_NODE"; then
      echo "Eval refused missing or unowned scratch checkpoint: $HOST_OUT_DIR/ckpt/$group" >&2
      exit 5
    fi
    eval_group "$group"
    if truthy "$RUN_PREDICT"; then
      predict_group "$group"
    fi
  done
else
  log "eval skipped by OMNICODER_OVERFIT_RUN_EVAL=$RUN_EVAL"
fi

if truthy "$RUN_SUMMARY"; then
  summarize_run
fi

write_launch_manifest complete
log "done"
echo "omnimodal_overfit_out=$HOST_OUT_DIR"
echo "omnimodal_overfit_manifest=$HOST_OUT_DIR/omnimodal_overfit_manifest.json"
echo "omnimodal_overfit_summary=$HOST_OUT_DIR/omnimodal_overfit_summary.json"
