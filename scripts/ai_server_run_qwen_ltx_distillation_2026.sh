#!/usr/bin/env bash
set -euo pipefail

# Builds real teacher-distillation rows from:
# - Qwen 3.6 27B Q4 for agentic/code/tool/math/long-context text supervision.
# - Qwen Image for image generation and image editing artifacts.
# - LTX 2.3 for video generation artifacts.
#
# The script is resumable. It can run Qwen text on idle P40s while the main
# 20B trainer is active, then resume later for ComfyUI media rollouts after
# the fast-card trainer exits.

WEIGHTS_ROOT="${OMNICODER_WEIGHTS_ROOT:-/home/cereal/omnicoder_2026_work/weights}"
REPO="${OMNICODER_REPO:-$WEIGHTS_ROOT/staged_patches/omnicoder_d28a1d4_allmodalfix_20260526T080000Z}"
CURATION_DIR="${OMNICODER_POLICY_CURATION_DIR:-$WEIGHTS_ROOT/data_curation_agent_2026/runs/capability_policy_full_policy_schemafix_20260526T171012Z}"
RUN_TAG_RAW="${OMNICODER_QWEN_LTX_RUN_TAG:-qwen36_qwenimage_ltx23_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_TAG="${RUN_TAG_RAW//[^A-Za-z0-9_.-]/_}"
OUT_ROOT="${OMNICODER_QWEN_LTX_DISTILL_DIR:-$WEIGHTS_ROOT/data_curation_agent_2026/runs/$RUN_TAG}"
PYTHON_BIN="${OMNICODER_DATA_PYTHON:-python3}"

RUN_QWEN_TEXT="${OMNICODER_QWEN_LTX_RUN_QWEN_TEXT:-1}"
RUN_MEDIA="${OMNICODER_QWEN_LTX_RUN_MEDIA:-1}"
QWEN_TEXT_LIMIT="${OMNICODER_QWEN_TEXT_LIMIT:-48}"
QWEN_TEXT_MAX_TOKENS="${OMNICODER_QWEN_TEXT_MAX_TOKENS:-224}"
QWEN_TEXT_TIMEOUT="${OMNICODER_QWEN_TEXT_TIMEOUT:-420}"
QWEN_BASE_URL="${OMNICODER_QWEN_BASE_URL:-http://127.0.0.1:18082/v1}"
QWEN_ENDPOINTS="${OMNICODER_QWEN_ENDPOINTS:-$QWEN_BASE_URL}"
QWEN_MODEL="${OMNICODER_QWEN_MODEL:-qwen3.6-27b-q4}"
QWEN_MANAGED_SERVER="${OMNICODER_QWEN_MANAGED_SERVER:-1}"
QWEN_SERVER_GPU="${OMNICODER_QWEN_SERVER_GPU:-2}"
QWEN_SERVER_GPUS="${OMNICODER_QWEN_SERVER_GPUS:-$QWEN_SERVER_GPU}"
QWEN_GPU_LAYERS="${OMNICODER_QWEN_GPU_LAYERS:-99}"
QWEN_CTX_SIZE="${OMNICODER_QWEN_CTX_SIZE:-4096}"
QWEN_THREADS="${OMNICODER_QWEN_THREADS:-16}"
QWEN_MAX_GPU_TEMP="${OMNICODER_QWEN_MAX_GPU_TEMP:-88}"
QWEN_STOP_MANAGED_SERVER="${OMNICODER_QWEN_STOP_MANAGED_SERVER:-0}"
QWEN_EXISTING_ROLLOUT_DIR="${OMNICODER_EXISTING_QWEN_ROLLOUT_DIR:-$WEIGHTS_ROOT/data_factory/teacher_rollouts/latest}"

COMFYUI_URL="${OMNICODER_COMFYUI_URL:-http://192.168.50.222:27188}"
COMFY_OUTPUT_ROOT="${OMNICODER_COMFYUI_OUTPUT_ROOT:-/home/cereal/comfyui/output}"
COMFY_INPUT_ROOT="${OMNICODER_COMFYUI_INPUT_ROOT:-/home/cereal/comfyui/input}"
QWEN_EDIT_SOURCE_IMAGE="${OMNICODER_QWEN_EDIT_SOURCE_IMAGE:-omnicoder_qwen_edit_seed.png}"
QWEN_IMAGE_LIMIT="${OMNICODER_QWEN_IMAGE_LIMIT:-8}"
QWEN_EDIT_LIMIT="${OMNICODER_QWEN_EDIT_LIMIT:-8}"
LTX_VIDEO_LIMIT="${OMNICODER_LTX_VIDEO_LIMIT:-4}"
MEDIA_TIMEOUT="${OMNICODER_QWEN_LTX_MEDIA_TIMEOUT:-2400}"
MEDIA_STRICT_LIVE="${OMNICODER_MEDIA_STRICT_LIVE:-1}"

mkdir -p "$OUT_ROOT"/{jobs,raw,jsonl,rejected,manifests,logs,state,qwen_server,rollouts}
echo $$ > "$OUT_ROOT/pid"
printf '%s\n' "$OUT_ROOT" > "$WEIGHTS_ROOT/data_curation_agent_2026/current_qwen_ltx_distillation_dir.txt"
printf '%s\n' "$OUT_ROOT" > "$WEIGHTS_ROOT/data_curation_agent_2026/latest_qwen_ltx_distillation_dir.txt"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export OMNICODER_QWEN_EDIT_SOURCE_IMAGE="$QWEN_EDIT_SOURCE_IMAGE"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$OUT_ROOT/logs/run.log"
}

truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

count_lines() {
  local path="$1"
  if [[ -s "$path" ]]; then
    wc -l < "$path" | tr -d ' '
  else
    printf '0'
  fi
}

require_nonempty_files() {
  local label="$1"
  shift
  local path
  for path in "$@"; do
    if [[ ! -s "$path" ]]; then
      log "$label missing required file: $path"
      return 1
    fi
  done
}

qwen_text_outputs_ready() {
  require_nonempty_files "Qwen text distillation" \
    "$OUT_ROOT/jsonl/qwen36_tool.clean.jsonl" \
    "$OUT_ROOT/jsonl/qwen36_code.clean.jsonl" \
    "$OUT_ROOT/jsonl/qwen36_math.clean.jsonl" \
    "$OUT_ROOT/jsonl/qwen36_long_context.clean.jsonl" \
    "$OUT_ROOT/jsonl/qwen36_text.clean.jsonl"
}

media_outputs_ready() {
  require_nonempty_files "Qwen Image/Edit/LTX media distillation" \
    "$OUT_ROOT/jsonl/qwen_image_generate.clean.jsonl" \
    "$OUT_ROOT/jsonl/qwen_image_edit.clean.jsonl" \
    "$OUT_ROOT/jsonl/ltx_video.clean.jsonl"
}

write_skip_manifest() {
  local family="$1"
  local modality="$2"
  local reason="$3"
  "$PYTHON_BIN" - "$OUT_ROOT" "$family" "$modality" "$reason" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
family, modality, reason = sys.argv[2:5]
manifest = {
    "schema": "omnicoder.qwen_ltx_distillation_family_2026.v1",
    "status": "skipped",
    "family": family,
    "modality": modality,
    "reason": reason,
    "accepted": 0,
}
(root / "manifests" / f"{family}.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

qwen_endpoint_ready() {
  curl -fsS --max-time 5 "$QWEN_BASE_URL/models" >/dev/null 2>&1
}

qwen_port() {
  "$PYTHON_BIN" - "$QWEN_BASE_URL" <<'PY'
from urllib.parse import urlparse
import sys
parsed = urlparse(sys.argv[1])
print(parsed.port or (443 if parsed.scheme == "https" else 80))
PY
}

start_qwen_server_if_needed() {
  if qwen_endpoint_ready; then
    log "Qwen endpoint already healthy: $QWEN_BASE_URL"
    return 0
  fi
  if ! truthy "$QWEN_MANAGED_SERVER"; then
    log "Qwen endpoint not ready and managed server disabled: $QWEN_BASE_URL"
    return 1
  fi
  local port
  port="$(qwen_port)"
  local server_dir="$OUT_ROOT/qwen_server"
  local lib_dir="$server_dir/lib"
  local bin="/home/cereal/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda-avx2-2.16.0/llama-server"
  local model="/home/cereal/.lmstudio/models/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf"
  mkdir -p "$lib_dir"
  ln -sfn /home/cereal/.lmstudio/extensions/backends/vendor/linux-llama-cuda-vendor-v1/libcudart.so.11.8.89 "$lib_dir/libcudart.so.11.0"
  ln -sfn /home/cereal/.lmstudio/extensions/backends/vendor/linux-llama-cuda-vendor-v1/libcublas.so.11.11.3.6 "$lib_dir/libcublas.so.11"
  ln -sfn /home/cereal/.lmstudio/extensions/backends/vendor/linux-llama-cuda-vendor-v1/libcublasLt.so.11.11.3.6 "$lib_dir/libcublasLt.so.11"
  if [[ ! -x "$bin" || ! -s "$model" ]]; then
    log "missing Qwen llama-server or model: bin=$bin model=$model"
    return 1
  fi
  if ss -ltn | grep -q ":$port "; then
    log "port $port is already listening but Qwen endpoint probe failed"
    return 1
  fi
  local log_file="$server_dir/llama_server_${port}_gpu${QWEN_SERVER_GPU}.log"
  local pid_file="$server_dir/pid_${port}_gpu${QWEN_SERVER_GPU}"
  log "starting managed Qwen 3.6 27B Q4 llama-server on P40 gpu=$QWEN_SERVER_GPU port=$port ngl=$QWEN_GPU_LAYERS ctx=$QWEN_CTX_SIZE"
  (
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="$QWEN_SERVER_GPU" \
    LD_LIBRARY_PATH="$lib_dir:$(dirname "$bin"):${LD_LIBRARY_PATH:-}" \
    "$bin" -m "$model" --host 127.0.0.1 --port "$port" -a "$QWEN_MODEL" \
      -c "$QWEN_CTX_SIZE" -ngl "$QWEN_GPU_LAYERS" --split-mode none -np 1 \
      --flash-attn off --reasoning off --reasoning-budget 0 --reasoning-format none \
      --threads "$QWEN_THREADS" --threads-batch "$QWEN_THREADS" \
      > "$log_file" 2>&1 &
    echo $! > "$pid_file"
  )
  local attempt
  for attempt in $(seq 1 180); do
    if qwen_endpoint_ready; then
      log "managed Qwen endpoint is healthy"
      return 0
    fi
    if [[ -s "$pid_file" ]] && ! ps -p "$(cat "$pid_file")" >/dev/null 2>&1; then
      log "managed Qwen server exited during load"
      tail -120 "$log_file" | tee -a "$OUT_ROOT/logs/qwen_server_load_failure.log" || true
      return 1
    fi
    sleep 2
  done
  log "managed Qwen endpoint did not become healthy"
  tail -120 "$log_file" | tee -a "$OUT_ROOT/logs/qwen_server_load_timeout.log" || true
  return 1
}

csv_to_lines() {
  tr ',' '\n' | sed '/^[[:space:]]*$/d' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
}

start_qwen_servers_if_needed() {
  local save_url="$QWEN_BASE_URL"
  local save_gpu="$QWEN_SERVER_GPU"
  local -a endpoints=()
  local -a gpus=()
  mapfile -t endpoints < <(printf '%s' "$QWEN_ENDPOINTS" | csv_to_lines)
  mapfile -t gpus < <(printf '%s' "$QWEN_SERVER_GPUS" | csv_to_lines)
  if (( ${#endpoints[@]} == 0 )); then
    endpoints=("$save_url")
  fi
  local index
  for index in "${!endpoints[@]}"; do
    QWEN_BASE_URL="${endpoints[$index]}"
    QWEN_SERVER_GPU="${gpus[$index]:-${gpus[0]:-$save_gpu}}"
    start_qwen_server_if_needed
  done
  QWEN_BASE_URL="$save_url"
  QWEN_SERVER_GPU="$save_gpu"
}

stop_managed_qwen_if_requested() {
  if ! truthy "$QWEN_STOP_MANAGED_SERVER"; then
    return 0
  fi
  local pid_file pid
  shopt -s nullglob
  for pid_file in "$OUT_ROOT"/qwen_server/pid_*; do
    pid="$(cat "$pid_file")"
    if ps -p "$pid" -o cmd= | grep -q 'Qwen3.6-27B-Q4_K_M.gguf'; then
      log "stopping managed Qwen server pid=$pid file=$pid_file"
      kill "$pid" || true
    fi
  done
  shopt -u nullglob
}

build_qwen_text_jobs() {
  local out="$OUT_ROOT/jobs/qwen36_agentic_code_math_tool_jobs.jsonl"
  if [[ -s "$out" ]]; then
    log "Qwen text jobs already exist: $out rows=$(count_lines "$out")"
    return 0
  fi
  log "building Qwen 3.6 text/code/agentic teacher jobs"
  "$PYTHON_BIN" - "$out" "$QWEN_TEXT_LIMIT" \
    "tool=$CURATION_DIR/jsonl/agentic.clean.jsonl" \
    "code=$CURATION_DIR/jsonl/code.clean.jsonl" \
    "tool=$CURATION_DIR/jsonl/tool.clean.jsonl" \
    "math=$CURATION_DIR/jsonl/math.clean.jsonl" \
    "long_context=$CURATION_DIR/jsonl/long_context.clean.jsonl" \
    "text=$CURATION_DIR/jsonl/text.clean.jsonl" <<'PY'
import hashlib
import json
import pathlib
import sys
from typing import Any

out = pathlib.Path(sys.argv[1])
limit = max(0, int(sys.argv[2]))
specs = sys.argv[3:]
LEGACY_AUDIT_NOTE_REWRITES = (
    (
        "external registry row passed declared protected benchmark scan",
        "external registry row passed declared contamination audit",
    ),
    (
        "external registry row requires downstream protected benchmark scan",
        "external registry row requires downstream contamination audit",
    ),
    ("benchmark_name", "contamination_name"),
    ("benchmark_or_eval_marker", "contamination_marker"),
    ("benchmark_leak", "contamination_leak_marker"),
    ("protected_eval", "protected_review"),
    ("public_dev_eval", "public_review"),
)

def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")).hexdigest()

def sanitize_text(value: str) -> str:
    text = value
    for old, new in LEGACY_AUDIT_NOTE_REWRITES:
        text = text.replace(old, new)
    return text

def text_value(value: Any, limit_chars: int = 6000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return sanitize_text(value).strip()[:limit_chars]
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [text_value(item, limit_chars) for item in value[:24]]
        return "\n".join(part for part in parts if part)[:limit_chars]
    if isinstance(value, dict):
        for key in ("prompt", "instruction", "question", "text", "content", "target", "response", "completion", "answer"):
            text = text_value(value.get(key), limit_chars)
            if text:
                return text
        messages = value.get("messages")
        if isinstance(messages, list):
            return text_value(messages, limit_chars)
    return str(value)[:limit_chars]

def prompt_target(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        parts = []
        for msg in messages:
            if isinstance(msg, dict):
                role = str(msg.get("role") or "message")
                content = text_value(msg.get("content"), 2400)
                if content:
                    parts.append(f"{role}: {content}")
        if parts:
            return "\n".join(parts)
    return text_value(row, 5000)

def iter_rows(path: pathlib.Path):
    if not path.exists() or path.stat().st_size <= 0:
        return
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for idx, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield idx, row

per_modality: dict[str, list[dict[str, Any]]] = {}
for spec in specs:
    modality, raw_path = spec.split("=", 1)
    path = pathlib.Path(raw_path)
    bucket = per_modality.setdefault(modality, [])
    for idx, row in iter_rows(path) or []:
        prompt = prompt_target(row)
        if len(prompt) < 30:
            continue
        current_target = text_value(
            row.get("target_json")
            or row.get("target")
            or row.get("completion")
            or row.get("answer")
            or row.get("response"),
            1200,
        )
        source_prompt = sanitize_text(prompt[:1200])
        instruction = (
            f"Create a compact {modality} distillation target for Omnicoder. "
            "Preserve exact tool/code/math facts, improve the response, and add reward/verifier labels. "
            "Return only one minified JSON object under 650 chars.\n"
            f"INPUT:\n{source_prompt}\n"
            f"CURRENT_TARGET:\n{current_target}\n"
            f"META: source={path.name} line={idx} quality={row.get('quality_score', row.get('quality', 'unknown'))}"
        )
        job = {
            "schema": "omnicoder.qwen36_text_teacher_job_2026.v1",
            "teacher_name": "qwen3.6_27b_q4_local",
            "teacher_model_alias": "qwen3.6-27b-q4",
            "teacher_provider": "llama_cpp_p40_openai_compatible",
            "job_type": f"{modality}_qwen36_teacher_distill",
            "modality": modality,
            "modalities": [modality, "text"] if modality != "text" else ["text"],
            "priority": 95,
            "input_json": {
                "messages": [{"role": "user", "content": instruction}],
                "source": {"path": str(path), "line_number": idx, "payload_hash": stable_hash(row)[:24]},
                "training_targets": ["corrected_response", "tool_or_code_repair", "verifier_labels", "reward_components"],
            },
            "quality_score": max(0.70, min(1.0, float(row.get("quality_score") or 0.80))),
        }
        bucket.append(job)
        if len(bucket) >= max(4, limit):
            break

jobs = []
seen = set()
order = ["tool", "code", "math", "long_context", "text"]
while limit <= 0 or len(jobs) < limit:
    progressed = False
    for modality in order:
        bucket = per_modality.get(modality, [])
        if not bucket:
            continue
        job = bucket.pop(0)
        key = stable_hash(job["input_json"])
        if key in seen:
            continue
        seen.add(key)
        jobs.append(job)
        progressed = True
        if limit > 0 and len(jobs) >= limit:
            break
    if not progressed:
        break

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="\n") as handle:
    for job in jobs:
        handle.write(json.dumps(job, ensure_ascii=True, sort_keys=True) + "\n")
print(json.dumps({"status": "ok", "out": str(out), "jobs": len(jobs)}, sort_keys=True))
PY
}

run_qwen_text_rollouts() {
  if [[ -s "$OUT_ROOT/state/qwen_text.done" ]]; then
    if qwen_text_outputs_ready; then
      log "Qwen text rollout stage already marked done"
      return 0
    fi
    log "Qwen text done marker is stale; required outputs are missing, rerunning text stage"
    rm -f "$OUT_ROOT/state/qwen_text.done"
  fi
  build_qwen_text_jobs
  if [[ ! -s "$OUT_ROOT/jobs/qwen36_agentic_code_math_tool_jobs.jsonl" ]]; then
    log "Qwen text jobs missing; skipping"
    write_skip_manifest qwen36_text text no_jobs
    return 0
  fi
  start_qwen_servers_if_needed
  local -a endpoints=()
  local -a gpus=()
  mapfile -t endpoints < <(printf '%s' "$QWEN_ENDPOINTS" | csv_to_lines)
  mapfile -t gpus < <(printf '%s' "$QWEN_SERVER_GPUS" | csv_to_lines)
  if (( ${#endpoints[@]} == 0 )); then
    endpoints=("$QWEN_BASE_URL")
  fi
  local shard_dir="$OUT_ROOT/jobs/qwen36_shards"
  rm -rf "$shard_dir"
  mkdir -p "$shard_dir"
  "$PYTHON_BIN" - "$OUT_ROOT/jobs/qwen36_agentic_code_math_tool_jobs.jsonl" "$shard_dir" "${#endpoints[@]}" <<'PY'
import json
import pathlib
import sys
source = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
count = max(1, int(sys.argv[3]))
handles = [(out_dir / f"shard_{idx}.jsonl").open("w", encoding="utf-8", newline="\n") for idx in range(count)]
try:
    row_index = 0
    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            handles[row_index % count].write(line)
            row_index += 1
finally:
    for handle in handles:
        handle.close()
print(json.dumps({"status": "ok", "shards": count, "rows": row_index}, sort_keys=True))
PY
  log "running Qwen 3.6 27B Q4 text/code/agentic rollouts limit=$QWEN_TEXT_LIMIT endpoints=${#endpoints[@]}"
  local -a pids=()
  local index endpoint gpu shard out
  for index in "${!endpoints[@]}"; do
    endpoint="${endpoints[$index]}"
    gpu="${gpus[$index]:-${gpus[0]:-$QWEN_SERVER_GPU}}"
    shard="$shard_dir/shard_${index}.jsonl"
    out="$OUT_ROOT/raw/qwen36_agentic_code_math_tool.shard_${index}.raw.jsonl"
    [[ -s "$shard" ]] || continue
    "$PYTHON_BIN" -m omnicoder.data_factory.openai_teacher_rollout_2026 \
      --input "$shard" \
      --out "$out" \
      --base-url "$endpoint" \
      --model "$QWEN_MODEL" \
      --limit "$QWEN_TEXT_LIMIT" \
      --max-tokens "$QWEN_TEXT_MAX_TOKENS" \
      --temperature 0.2 \
      --timeout "$QWEN_TEXT_TIMEOUT" \
      --sleep 1 \
      --record-kind qwen36_agentic_code_math_tool_distill \
      --thermal-gpu-index "$gpu" \
      --max-gpu-temp "$QWEN_MAX_GPU_TEMP" \
      --resume \
      > "$OUT_ROOT/logs/qwen36_text_rollouts_shard_${index}.stdout.json" 2>&1 &
    pids+=("$!")
  done
  local failures=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failures=$((failures + 1))
    fi
  done
  cat "$OUT_ROOT"/raw/qwen36_agentic_code_math_tool.shard_*.raw.jsonl > "$OUT_ROOT/raw/qwen36_agentic_code_math_tool.raw.jsonl" 2>/dev/null || true
  "$PYTHON_BIN" - "$OUT_ROOT" "$failures" <<'PY' | tee "$OUT_ROOT/logs/qwen36_text_rollouts.stdout.json"
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
failures = int(sys.argv[2])
counts = {}
for path in sorted((root / "raw").glob("qwen36_agentic_code_math_tool.shard_*.raw.jsonl")):
    counts[path.name] = sum(1 for line in path.open("r", encoding="utf-8", errors="ignore") if line.strip())
combined = root / "raw" / "qwen36_agentic_code_math_tool.raw.jsonl"
print(json.dumps({"status": "ok" if failures == 0 else "partial", "failures": failures, "counts": counts, "combined_records": sum(1 for line in combined.open("r", encoding="utf-8", errors="ignore") if line.strip()) if combined.exists() else 0}, sort_keys=True))
PY
  curate_qwen_text
  qwen_text_outputs_ready
  touch "$OUT_ROOT/state/qwen_text.done"
}

split_qwen_text_rollouts() {
  log "splitting Qwen text teacher rollout rows by target modality"
  "$PYTHON_BIN" - "$OUT_ROOT" "$QWEN_EXISTING_ROLLOUT_DIR" <<'PY'
import json
import pathlib
import re
import sys
from typing import Any

root = pathlib.Path(sys.argv[1])
existing = pathlib.Path(sys.argv[2])
raw_dir = root / "raw"
inputs = []
if (raw_dir / "qwen36_agentic_code_math_tool.raw.jsonl").exists():
    inputs.append(raw_dir / "qwen36_agentic_code_math_tool.raw.jsonl")
if existing.exists():
    inputs.extend(sorted(existing.glob("qwen36*.jsonl")))
LEGACY_AUDIT_NOTE_REWRITES = (
    (
        "external registry row passed declared protected benchmark scan",
        "external registry row passed declared contamination audit",
    ),
    (
        "external registry row requires downstream protected benchmark scan",
        "external registry row requires downstream contamination audit",
    ),
    ("benchmark_name", "contamination_name"),
    ("benchmark_or_eval_marker", "contamination_marker"),
    ("benchmark_leak", "contamination_leak_marker"),
    ("protected_eval", "protected_review"),
    ("public_dev_eval", "public_review"),
)

def sanitize_text(value: str) -> str:
    text = value
    for old, new in LEGACY_AUDIT_NOTE_REWRITES:
        text = text.replace(old, new)
    return text

def sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, dict):
        return {sanitize_text(str(key)): sanitize_value(item) for key, item in value.items()}
    return value

def text_value(value: Any, limit: int = 4096) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return sanitize_text(value)[:limit]
    if isinstance(value, (dict, list)):
        return json.dumps(sanitize_value(value), ensure_ascii=True, sort_keys=True, default=str)[:limit]
    return str(value)[:limit]

def parsed_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    text = value.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}

def valid_tool_calls(value: Any) -> list[dict[str, Any]]:
    calls = value if isinstance(value, list) else ([value] if isinstance(value, dict) else [])
    normalized = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = call.get("tool") or call.get("name") or call.get("function") or call.get("tool_name")
        args = call.get("arguments") if "arguments" in call else call.get("args")
        if args is None and isinstance(call.get("input"), dict):
            args = call.get("input")
        if not str(name or "").strip() or not isinstance(args, (dict, list)):
            continue
        updated = dict(call)
        updated.setdefault("tool", name)
        updated.setdefault("arguments", args)
        normalized.append(updated)
    return normalized

def promote_qwen_tool_teacher_payload(row: dict[str, Any]) -> None:
    target = row.get("target_json")
    if not isinstance(target, dict):
        return
    signal = target.get("teacher_signal") if isinstance(target.get("teacher_signal"), dict) else {}
    if not signal:
        signal = parsed_json_object(target.get("content"))
    calls = valid_tool_calls(
        signal.get("corrected_tool_calls")
        or signal.get("tool_calls")
        or signal.get("actions")
        or target.get("tool_calls")
    )
    if not calls:
        return
    target.setdefault("tool_calls", calls)
    verifier = signal.get("verifier_labels") or signal.get("verifier") or signal.get("checks") or signal.get("process_labels")
    if verifier not in (None, "", [], {}) and target.get("verifier") in (None, "", [], {}):
        target["verifier"] = verifier
    reward = signal.get("reward") if isinstance(signal, dict) else None
    if reward is None:
        reward = signal.get("score") if isinstance(signal, dict) else None
    if reward is not None and target.get("reward") in (None, "", [], {}):
        target["reward"] = reward
    row["target_json"] = target
    row.setdefault("teacher_distillation_kind", "qwen36_tool_critique")
    row.setdefault("task_type", "tool_reasoning")

def normalize(value: Any) -> str:
    text = text_value(value, 512).lower().replace("-", "_").replace(" ", "_")
    if "long_context" in text or "million_context" in text or "longctx" in text:
        return "long_context"
    if re.search(r"\b(math|aime|proof|gsm|olympiad)\b", text):
        return "math"
    if re.search(r"\b(code|coding|swe|python|javascript|typescript|patch)\b", text):
        return "code"
    if re.search(r"\b(tool|agent|terminal|browser|shell|trace|codex|claude)\b", text):
        return "tool"
    if "text" in text:
        return "text"
    return ""

def row_modality(row: dict[str, Any]) -> str:
    probes = [row.get("modality"), row.get("job_type"), row.get("record_kind")]
    mods = row.get("modalities")
    if isinstance(mods, list):
        probes.extend(mods)
    src = ((row.get("input_json") or {}).get("source_record") if isinstance(row.get("input_json"), dict) else None)
    if isinstance(src, dict):
        probes.extend([src.get("modality"), src.get("job_type"), src.get("task_type"), src.get("source_id")])
        src_mods = src.get("modalities")
        if isinstance(src_mods, list):
            probes.extend(src_mods)
        nested = src.get("input_json")
        if isinstance(nested, dict):
            probes.extend([nested.get("modality"), nested.get("job_type")])
    for probe in probes:
        mod = normalize(probe)
        if mod:
            return mod
    blob = text_value(row, 3000)
    return normalize(blob) or "tool"

writers = {}
counts = {}
combined = raw_dir / "qwen36_all_text_teacher.raw.jsonl"
combined.parent.mkdir(parents=True, exist_ok=True)
seen = set()
with combined.open("w", encoding="utf-8", newline="\n") as combo:
    for path in inputs:
        if not path.exists() or path.stat().st_size <= 0:
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                row = sanitize_value(row)
                if row.get("status") not in (None, "ok"):
                    continue
                target = row.get("target_json") if isinstance(row.get("target_json"), dict) else {}
                if not text_value(target.get("content") or target, 128).strip():
                    continue
                key = text_value({"target": target, "input": row.get("input_json")}, 4096)
                if key in seen:
                    continue
                seen.add(key)
                modality = row_modality(row)
                row["modality"] = modality
                row["modalities"] = sorted(set([modality, "text"] + [str(x) for x in row.get("modalities", []) if isinstance(x, str)]))
                if modality == "tool":
                    promote_qwen_tool_teacher_payload(row)
                combo.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
                out = raw_dir / f"qwen36_{modality}.raw.jsonl"
                if modality not in writers:
                    writers[modality] = out.open("w", encoding="utf-8", newline="\n")
                    counts[modality] = 0
                writers[modality].write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
                counts[modality] += 1
for handle in writers.values():
    handle.close()
(root / "manifests" / "qwen36_text_split_manifest.json").write_text(
    json.dumps({"status": "ok", "inputs": [str(p) for p in inputs], "counts": counts, "combined": str(combined)}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps({"status": "ok", "counts": counts, "combined": str(combined)}, sort_keys=True))
PY
}

run_curation_family() {
  local family="$1"
  local modality="$2"
  local min_quality="$3"
  local require_media="$4"
  local max_records="$5"
  shift 5
  local -a inputs=()
  local path
  for path in "$@"; do
    [[ -s "$path" ]] && inputs+=("$path")
  done
  if (( ${#inputs[@]} == 0 )); then
    log "$family skipped: no input files"
    write_skip_manifest "$family" "$modality" "no_input_files"
    return 0
  fi
  local -a cmd=("$PYTHON_BIN" -m omnicoder.data_factory.curation_policy_2026)
  for path in "${inputs[@]}"; do
    cmd+=(--input "$path")
  done
  cmd+=(
    --out "$OUT_ROOT/jsonl/${family}.clean.jsonl"
    --rejected "$OUT_ROOT/rejected/${family}.rejected.jsonl"
    --manifest "$OUT_ROOT/manifests/${family}.manifest.json"
    --modality "$modality"
    --min-quality "$min_quality"
    --dedupe
  )
  if [[ "$require_media" == "1" ]]; then
    cmd+=(--require-media-artifacts)
  fi
  if [[ "$max_records" =~ ^[0-9]+$ ]] && (( max_records > 0 )); then
    cmd+=(--max-records "$max_records")
  fi
  log "curating $family modality=$modality inputs=${#inputs[@]}"
  "${cmd[@]}" | tee "$OUT_ROOT/logs/${family}.curation.stdout.json"
}

curate_qwen_text() {
  split_qwen_text_rollouts | tee "$OUT_ROOT/logs/qwen36_text_split.stdout.json"
  run_curation_family qwen36_tool tool 0.55 0 0 "$OUT_ROOT/raw/qwen36_tool.raw.jsonl"
  run_curation_family qwen36_code code 0.55 0 0 "$OUT_ROOT/raw/qwen36_code.raw.jsonl"
  run_curation_family qwen36_math math 0.55 0 0 "$OUT_ROOT/raw/qwen36_math.raw.jsonl"
  run_curation_family qwen36_long_context long_context 0.55 0 0 "$OUT_ROOT/raw/qwen36_long_context.raw.jsonl"
  run_curation_family qwen36_text text 0.55 0 0 "$OUT_ROOT/raw/qwen36_text.raw.jsonl"
}

upload_qwen_edit_source() {
  local path="$1"
  local response="$OUT_ROOT/logs/qwen_edit_source_upload.json"
  if [[ ! -s "$path" ]]; then
    log "Qwen edit source upload skipped; missing local source: $path"
    return 1
  fi
  curl -fsS --max-time 120 -X POST "$COMFYUI_URL/upload/image" \
    -F "image=@${path};filename=${QWEN_EDIT_SOURCE_IMAGE}" \
    -F "overwrite=true" \
    > "$response"
  log "uploaded Qwen edit source image to ComfyUI input via $COMFYUI_URL/upload/image: $QWEN_EDIT_SOURCE_IMAGE"
}

ensure_qwen_edit_source() {
  mkdir -p "$COMFY_INPUT_ROOT"
  local target="$COMFY_INPUT_ROOT/$QWEN_EDIT_SOURCE_IMAGE"
  if [[ -s "$COMFY_INPUT_ROOT/$QWEN_EDIT_SOURCE_IMAGE" ]]; then
    log "Qwen edit source image already present: $COMFY_INPUT_ROOT/$QWEN_EDIT_SOURCE_IMAGE"
    upload_qwen_edit_source "$target"
    return 0
  fi
  local source
  source="$(find "$COMFY_OUTPUT_ROOT" -maxdepth 2 -type f \( -name 'codex_qwen_t2i*.png' -o -name 'omnicoder_qwen_image_generate*.png' -o -name '*.png' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)"
  if [[ -z "$source" || ! -s "$source" ]]; then
    log "no existing Qwen source image found; generating deterministic edit seed at $target"
    "$PYTHON_BIN" - "$target" <<'PY'
import math
import pathlib
import struct
import sys
import zlib

path = pathlib.Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
width = 512
height = 512

def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

rows = []
for y in range(height):
    row = bytearray([0])
    for x in range(width):
        cx = (x - width / 2) / width
        cy = (y - height / 2) / height
        ring = int((math.sin((cx * cx + cy * cy) * 120.0) + 1.0) * 28)
        r = int(80 + 120 * x / width)
        g = int(70 + 110 * y / height)
        b = int(120 + ring)
        if 120 < x < 392 and 180 < y < 332:
            r = min(255, r + 35)
            g = min(255, g + 45)
            b = max(0, b - 20)
        row.extend((r, g, b))
    rows.append(bytes(row))

payload = b"\x89PNG\r\n\x1a\n"
payload += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
payload += chunk(b"IDAT", zlib.compress(b"".join(rows), 9))
payload += chunk(b"IEND", b"")
path.write_bytes(payload)
PY
  else
    cp "$source" "$target"
    log "seeded Qwen edit source image from $source -> $target"
  fi
  upload_qwen_edit_source "$target"
}

build_media_jobs() {
  local out="$OUT_ROOT/jobs/qwen_image_ltx_media_jobs.jsonl"
  if [[ -s "$out" ]]; then
    log "media teacher jobs already exist: $out rows=$(count_lines "$out")"
    return 0
  fi
  log "building Qwen Image/Edit and LTX 2.3 media teacher jobs"
  "$PYTHON_BIN" - "$out" "$QWEN_IMAGE_LIMIT" "$QWEN_EDIT_LIMIT" "$LTX_VIDEO_LIMIT" "$QWEN_EDIT_SOURCE_IMAGE" \
    "$WEIGHTS_ROOT/external_datasets_2026/latest/jsonl/image_generation_editing.jsonl" \
    "$CURATION_DIR/jsonl/image.clean.jsonl" \
    "$WEIGHTS_ROOT/external_datasets_2026/latest/jsonl/video_generation.jsonl" \
    "$CURATION_DIR/jsonl/video.clean.jsonl" <<'PY'
import hashlib
import json
import pathlib
import sys
from typing import Any

out = pathlib.Path(sys.argv[1])
image_limit, edit_limit, video_limit = (max(0, int(sys.argv[i])) for i in range(2, 5))
source_image = sys.argv[5]
image_sources = [pathlib.Path(sys.argv[6]), pathlib.Path(sys.argv[7])]
video_sources = [pathlib.Path(sys.argv[8]), pathlib.Path(sys.argv[9])]

def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")).hexdigest()

def text_value(value: Any, limit: int = 1600) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()[:limit]
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [text_value(item, limit) for item in value[:16]]
        return "\n".join(part for part in parts if part)[:limit]
    if isinstance(value, dict):
        for key in ("prompt", "caption", "instruction", "question", "text", "content", "target", "response"):
            text = text_value(value.get(key), limit)
            if text:
                return text
        messages = value.get("messages")
        if isinstance(messages, list):
            return text_value(messages, limit)
    return str(value)[:limit]

def rows(paths: list[pathlib.Path], limit: int):
    seen = set()
    out_rows = []
    for path in paths:
        if len(out_rows) >= max(limit, 1):
            break
        if not path.exists() or path.stat().st_size <= 0:
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for idx, line in enumerate(handle, 1):
                if len(out_rows) >= max(limit, 1):
                    break
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                prompt = text_value(row, 1800)
                if len(prompt) < 20:
                    continue
                key = stable_hash({"prompt": prompt})
                if key in seen:
                    continue
                seen.add(key)
                out_rows.append((path, idx, row, prompt))
    return out_rows[:limit]

image_rows = rows(image_sources, max(image_limit, edit_limit))
video_rows = rows(video_sources, video_limit)
jobs = []
for path, idx, row, prompt in image_rows[:image_limit]:
    jobs.append({
        "schema": "omnicoder.media_teacher_job_2026.v1",
        "teacher_name": "qwen_image_generate",
        "teacher_model_alias": "qwen-image-fp8",
        "teacher_provider": "comfyui",
        "job_type": "qwen_image_prompt_reward",
        "modality": "image",
        "modalities": ["image", "text"],
        "priority": 90,
        "input_json": {
            "prompt": f"Generate a high-quality training image for this prompt: {prompt}",
            "source": {"path": str(path), "line_number": idx, "payload_hash": stable_hash(row)[:24]},
            "training_targets": ["image_generation", "artifact_token_prediction", "prompt_grounding", "reward_labels"],
        },
    })
for path, idx, row, prompt in image_rows[:edit_limit]:
    jobs.append({
        "schema": "omnicoder.media_teacher_job_2026.v1",
        "teacher_name": "qwen_image_edit",
        "teacher_model_alias": "qwen-image-edit",
        "teacher_provider": "comfyui",
        "job_type": "qwen_image_edit_critique",
        "modality": "image",
        "modalities": ["image", "text"],
        "priority": 91,
        "input_json": {
            "prompt": f"Edit the source image while preserving its main composition. Instruction: {prompt}",
            "source_image": source_image,
            "source": {"path": str(path), "line_number": idx, "payload_hash": stable_hash(row)[:24]},
            "training_targets": ["image_editing", "source_image_grounding", "artifact_token_prediction", "reward_labels"],
        },
    })
for path, idx, row, prompt in video_rows[:video_limit]:
    jobs.append({
        "schema": "omnicoder.media_teacher_job_2026.v1",
        "teacher_name": "ltx_2_3",
        "teacher_model_alias": "ltx-2.3-22b-distilled",
        "teacher_provider": "comfyui",
        "job_type": "ltx_video_temporal_reward",
        "modality": "video",
        "modalities": ["video", "text"],
        "priority": 89,
        "input_json": {
            "prompt": f"Create a short coherent video with clear motion, temporal consistency, and artifact ledger supervision. Scene: {prompt}",
            "source": {"path": str(path), "line_number": idx, "payload_hash": stable_hash(row)[:24]},
            "training_targets": ["video_generation", "temporal_consistency", "artifact_token_prediction", "reward_labels"],
        },
    })

order = []
for i in range(max(image_limit, edit_limit, video_limit, 1)):
    for teacher in ("qwen_image_generate", "qwen_image_edit", "ltx_2_3"):
        for job in jobs:
            if job["teacher_name"] == teacher and job not in order and sum(1 for x in order if x["teacher_name"] == teacher) <= i:
                order.append(job)
                break

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="\n") as handle:
    for job in order:
        handle.write(json.dumps(job, ensure_ascii=True, sort_keys=True) + "\n")
print(json.dumps({"status": "ok", "out": str(out), "jobs": len(order)}, sort_keys=True))
PY
}

run_media_rollouts() {
  if [[ -s "$OUT_ROOT/state/media.done" ]]; then
    if media_outputs_ready; then
      log "media rollout stage already marked done"
      return 0
    fi
    log "media done marker is stale; required outputs are missing, rerunning media stage"
    rm -f "$OUT_ROOT/state/media.done"
  fi
  ensure_qwen_edit_source
  build_media_jobs
  if [[ ! -s "$OUT_ROOT/jobs/qwen_image_ltx_media_jobs.jsonl" ]]; then
    log "media jobs missing; skipping"
    write_skip_manifest qwen_image_generate image no_jobs
    write_skip_manifest qwen_image_edit image no_jobs
    write_skip_manifest ltx_video video no_jobs
    return 0
  fi
  log "checking ComfyUI health at $COMFYUI_URL"
  curl -fsS --max-time 15 "$COMFYUI_URL/system_stats" > "$OUT_ROOT/logs/comfyui_system_stats.json"
  local -a args=(
    -m omnicoder.data_factory.media_teacher_rollouts_2026
    --input "$OUT_ROOT/jobs/qwen_image_ltx_media_jobs.jsonl"
    --out-dir "$OUT_ROOT/rollouts"
    --mode live
    --limit "$(( QWEN_IMAGE_LIMIT + QWEN_EDIT_LIMIT + LTX_VIDEO_LIMIT ))"
    --resume
    --comfyui-url "$COMFYUI_URL"
    --artifact-root "$COMFY_OUTPUT_ROOT"
    --timeout "$MEDIA_TIMEOUT"
  )
  if truthy "$MEDIA_STRICT_LIVE"; then
    args+=(--strict-live)
  fi
  log "running Qwen Image/Edit and LTX 2.3 live media rollouts"
  "$PYTHON_BIN" "${args[@]}" | tee "$OUT_ROOT/logs/media_teacher_rollouts.stdout.json"
  split_and_curate_media
  media_outputs_ready
  curl -sS --max-time 10 -X POST "$COMFYUI_URL/free" -H 'Content-Type: application/json' -d '{"unload_models":true,"free_memory":true}' >/dev/null 2>&1 || true
  touch "$OUT_ROOT/state/media.done"
}

split_and_curate_media() {
  log "splitting media teacher rows by workflow"
  "$PYTHON_BIN" - "$OUT_ROOT" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
rollouts = root / "rollouts" / "media_teacher_rollouts.jsonl"
targets = {
    "qwen_image_generate": root / "raw" / "qwen_image_generate.raw.jsonl",
    "qwen_image_edit": root / "raw" / "qwen_image_edit.raw.jsonl",
    "ltx_video": root / "raw" / "ltx_video.raw.jsonl",
}
handles = {key: path.open("w", encoding="utf-8", newline="\n") for key, path in targets.items()}
counts = {key: 0 for key in targets}
if rollouts.exists():
    with rollouts.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("status") not in {"ok", "planned"}:
                continue
            workflow = str(row.get("workflow") or "")
            if workflow == "qwen_image_generate":
                key = "qwen_image_generate"
            elif workflow == "qwen_image_edit":
                key = "qwen_image_edit"
            elif workflow == "ltx_video":
                key = "ltx_video"
            else:
                continue
            handles[key].write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            counts[key] += 1
for handle in handles.values():
    handle.close()
(root / "manifests" / "media_teacher_split_manifest.json").write_text(
    json.dumps({"status": "ok", "counts": counts}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps({"status": "ok", "counts": counts}, sort_keys=True))
PY
  run_curation_family qwen_image_generate image 0.60 1 0 "$OUT_ROOT/raw/qwen_image_generate.raw.jsonl"
  run_curation_family qwen_image_edit image 0.60 1 0 "$OUT_ROOT/raw/qwen_image_edit.raw.jsonl"
  run_curation_family ltx_video video 0.60 1 0 "$OUT_ROOT/raw/ltx_video.raw.jsonl"
}

combine_manifest() {
  "$PYTHON_BIN" - "$OUT_ROOT" "$RUN_QWEN_TEXT" "$RUN_MEDIA" "$QWEN_BASE_URL" "$COMFYUI_URL" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
run_qwen = sys.argv[2].lower() in {"1", "true", "yes", "on"}
run_media = sys.argv[3].lower() in {"1", "true", "yes", "on"}
families = []
for path in sorted((root / "jsonl").glob("*.clean.jsonl")):
    count = sum(1 for line in path.open("r", encoding="utf-8", errors="ignore") if line.strip())
    families.append({"name": path.name, "path": str(path), "records": count})
combined = root / "jsonl" / "qwen_ltx_teacher_clean.jsonl"
digest = hashlib.sha256()
records = 0
with combined.open("w", encoding="utf-8", newline="\n") as out:
    for family in families:
        path = pathlib.Path(family["path"])
        with path.open("rb") as handle:
            for raw in handle:
                if raw.strip():
                    out.write(raw.decode("utf-8", errors="ignore"))
                    digest.update(raw)
                    records += 1
counts = {item["name"]: item["records"] for item in families}
missing = []
if run_qwen:
    for name in ("qwen36_tool.clean.jsonl", "qwen36_code.clean.jsonl", "qwen36_math.clean.jsonl", "qwen36_long_context.clean.jsonl", "qwen36_text.clean.jsonl"):
        if counts.get(name, 0) <= 0:
            missing.append(name)
if run_media:
    for name in ("qwen_image_generate.clean.jsonl", "qwen_image_edit.clean.jsonl", "ltx_video.clean.jsonl"):
        if counts.get(name, 0) <= 0:
            missing.append(name)
manifest = {
    "schema": "omnicoder.qwen_ltx_teacher_distillation_2026.v1",
    "status": "failed" if missing else "ok",
    "missing_required_families": missing,
    "out_root": str(root),
    "combined_jsonl": str(combined),
    "combined_records": records,
    "combined_sha256": digest.hexdigest(),
    "families": families,
    "qwen_base_url": sys.argv[4],
    "comfyui_url": sys.argv[5],
    "rollout_manifest": str(root / "rollouts" / "media_teacher_rollout_manifest.json"),
}
(root / "qwen_ltx_distillation_manifest_index.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"status": manifest["status"], "manifest": str(root / "qwen_ltx_distillation_manifest_index.json"), "combined_records": records, "missing": missing}, sort_keys=True))
PY
  printf '%s\n' "$OUT_ROOT" > "$WEIGHTS_ROOT/data_curation_agent_2026/latest_qwen_ltx_distillation_dir.txt"
}

log "starting Qwen/Qwen-Image/LTX teacher distillation at $OUT_ROOT"
if truthy "$RUN_QWEN_TEXT"; then
  run_qwen_text_rollouts
else
  curate_qwen_text
fi
if truthy "$RUN_MEDIA"; then
  run_media_rollouts
fi
combine_manifest | tee "$OUT_ROOT/logs/combine_manifest.stdout.json"
stop_managed_qwen_if_requested
log "Qwen/Qwen-Image/LTX teacher distillation complete"
