#!/usr/bin/env bash
set -euo pipefail

ROOT="${OMNICODER_REPO:-/home/cereal/omnicoder_2026_work}"
RUN_ID="${OMNICODER_RUN_ID:-dataset_sidecars_$(date -u +%Y%m%dT%H%M%SZ)}"
PROFILE="${OMNICODER_DATASET_PROFILE:-profiles/dataset_curation_2026.json}"
TEACHER_MODEL="${OMNICODER_TEACHER_MODEL:-qwen/qwen3.6-27b}"
TEACHER_MODEL_GPU1="${OMNICODER_TEACHER_MODEL_GPU1:-qwen/qwen3.6-27b}"
TEACHER_MODEL_GPU2="${OMNICODER_TEACHER_MODEL_GPU2:-qwen/qwen3.6-27b2}"
TEACHER_MODEL_GPU3="${OMNICODER_TEACHER_MODEL_GPU3:-qwen/qwen3.6-27b3}"
TEACHER_BASE_URL="${OMNICODER_TEACHER_BASE_URL:-http://127.0.0.1:1234/v1}"
MAX_RECORDS_PER_DATASET="${OMNICODER_MAX_RECORDS_PER_DATASET:-1024}"
TEACHER_LIMIT="${OMNICODER_TEACHER_LIMIT:-256}"
MAX_GPU_TEMP="${OMNICODER_MAX_GPU_TEMP:-78}"
TEACHER_JOB_ROOT="${OMNICODER_TEACHER_JOB_ROOT:-weights/data_factory/runs/teacher_jobs}"
PROMOTE_LATEST="${OMNICODER_PROMOTE_LATEST:-1}"
PROMOTE_SHARED_ARTIFACTS="${OMNICODER_PROMOTE_SHARED_ARTIFACTS:-0}"
CURATED_DATASET_SOURCE="${OMNICODER_CURATED_DATASET_SOURCE:-weights/curated_datasets_2026/latest}"
EXTERNAL_DATASET_SOURCE="${OMNICODER_EXTERNAL_DATASET_SOURCE:-weights/external_datasets_2026/latest}"
TEACHER_JOB_SOURCE="${OMNICODER_TEACHER_JOB_SOURCE:-${TEACHER_JOB_ROOT}/latest}"
PYTHON_BIN="${OMNICODER_DATA_PYTHON:-python3}"
ENFORCE_DATASET_MINIMA="${OMNICODER_ENFORCE_DATASET_MINIMA:-1}"
DATASET_INCLUDE_WAVES="${OMNICODER_DATASET_INCLUDE_WAVES:-}"
DATASET_INCLUDE_FAMILIES="${OMNICODER_DATASET_INCLUDE_FAMILIES:-}"
DATASET_INCLUDE_NAMES="${OMNICODER_DATASET_INCLUDE_NAMES:-}"
HF_STEP_TIMEOUT_SECONDS="${OMNICODER_HF_STEP_TIMEOUT_SECONDS:-90}"
MATERIALIZE_DEFERRED_SOURCES="${OMNICODER_MATERIALIZE_DEFERRED_SOURCES:-0}"
MATERIALIZE_HF_SOURCES="${OMNICODER_MATERIALIZE_HF_SOURCES:-0}"
TRACE_LIMIT="${OMNICODER_TRACE_LIMIT:-0}"
LMSTUDIO_TRACE_LIMIT="${OMNICODER_LMSTUDIO_TRACE_LIMIT:-100000}"
LOCAL_TRACE_SOURCE="${OMNICODER_LOCAL_TRACE_SOURCE:-weights/curated_datasets_2026/runs/${RUN_ID}_local_traces}"
COVERAGE_STRICT="${OMNICODER_COVERAGE_STRICT:-0}"
REQUIRE_MEDIA_TEACHER_ROLLOUTS="${OMNICODER_REQUIRE_MEDIA_TEACHER_ROLLOUTS:-1}"
REQUIRE_REPORTABLE_TASKS="${OMNICODER_REQUIRE_REPORTABLE_TASKS:-0}"
REQUIRE_OFFICIAL_REPORTABLE_TASKS="${OMNICODER_REQUIRE_OFFICIAL_REPORTABLE_TASKS:-0}"
MATERIALIZE_BENCHMARK_TASKS="${OMNICODER_MATERIALIZE_BENCHMARK_TASKS:-0}"
BENCHMARK_PROFILE="${OMNICODER_BENCHMARK_PROFILE:-profiles/benchmark_suite_2026.json}"
BENCHMARK_MATERIALIZATION_ROOT="${OMNICODER_BENCHMARK_MATERIALIZATION_ROOT:-weights/data_factory/runs/benchmark_materialization/${RUN_ID}}"
BENCHMARK_MATERIALIZATION_LIMIT="${OMNICODER_BENCHMARK_MATERIALIZATION_LIMIT:-128}"
BENCHMARK_MATERIALIZATION_SUITE="${OMNICODER_BENCHMARK_MATERIALIZATION_SUITE:-known}"
BENCHMARK_MATERIALIZATION_MODE="${OMNICODER_BENCHMARK_MATERIALIZATION_MODE:-public-dev}"
BENCHMARK_MATERIALIZE_DOWNLOAD="${OMNICODER_BENCHMARK_MATERIALIZE_DOWNLOAD:-1}"
BENCHMARK_MATERIALIZE_STRICT="${OMNICODER_BENCHMARK_MATERIALIZE_STRICT:-0}"
BENCHMARK_MATERIALIZE_PROFILE_ROOTS="${OMNICODER_BENCHMARK_MATERIALIZE_PROFILE_ROOTS:-0}"
REPORTABLE_ROOT="${OMNICODER_REPORTABLE_ROOT:-}"
MEDIA_TEACHER_ROLLOUT_MODE="${OMNICODER_MEDIA_TEACHER_ROLLOUT_MODE:-live}"
MEDIA_TEACHER_LIMIT="${OMNICODER_MEDIA_TEACHER_LIMIT:-$TEACHER_LIMIT}"
LOCAL_HF_PROFILE="${OMNICODER_LOCAL_HF_PROFILE:-profiles/training_harness_2026.json}"
LOCAL_HF_BACKEND="${OMNICODER_LOCAL_HF_BACKEND:-unsloth}"
LOCAL_HF_MODEL="${OMNICODER_LOCAL_HF_MODEL:-}"
LOCAL_HF_TRAIN_JSONL="${OMNICODER_LOCAL_HF_TRAIN_JSONL:-}"
LOCAL_HF_MAX_STEPS="${OMNICODER_LOCAL_HF_MAX_STEPS:-1000}"
LOCAL_HF_MAX_SEQ_LEN="${OMNICODER_LOCAL_HF_MAX_SEQ_LEN:-4096}"
LOCAL_HF_HOST_GPU_IDS="${OMNICODER_LOCAL_HF_HOST_GPU_IDS:-}"
LOCAL_HF_PROTECTED_GPUS="${OMNICODER_LOCAL_HF_PROTECTED_GPUS:-0,4,6}"
LOCAL_HF_DRY_RUN="${OMNICODER_LOCAL_HF_DRY_RUN:-1}"
ACTION="${1:-all}"

cd "$ROOT"
export RUN_ID
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export OMNICODER_TRACE_WORK_DIR="${OMNICODER_TRACE_WORK_DIR:-weights/data_factory/runs/trace_orchestrator/${RUN_ID}}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

require_exact_teacher_models() {
  local item
  for item in "$TEACHER_MODEL" "$TEACHER_MODEL_GPU1" "$TEACHER_MODEL_GPU2" "$TEACHER_MODEL_GPU3"; do
    case "$item" in
      qwen/qwen3.6-27b|qwen/qwen3.6-27b2|qwen/qwen3.6-27b3) ;;
      *) echo "bad Qwen teacher model id: $item (must be one of the exact LM Studio pinned ids)" >&2; exit 23 ;;
    esac
  done
}

require_non_qwen_fast_hf_model() {
  if [[ -z "$LOCAL_HF_MODEL" ]]; then
    echo "LOCAL_HF_MODEL is required for local HF fast-card lanes and must not be Qwen" >&2
    exit 24
  fi
  if [[ "$LOCAL_HF_MODEL" == Qwen/* || "$LOCAL_HF_MODEL" == qwen/* ]]; then
    echo "LOCAL_HF_MODEL must not be Qwen on fast-card/non-Qwen lanes: $LOCAL_HF_MODEL" >&2
    exit 24
  fi
}

require_exact_teacher_models

require_nonempty_jsonl() {
  local path="$1"
  local label="$2"
  if [[ ! -s "$path" ]]; then
    echo "required trace artifact is missing or empty: $label -> $path" >&2
    exit 11
  fi
  local rows
  rows=$(wc -l < "$path" | tr -d ' ')
  if [[ "${rows:-0}" -le 0 ]]; then
    echo "required trace artifact has zero rows: $label -> $path" >&2
    exit 12
  fi
  log "trace gate passed: $label rows=$rows path=$path"
}

preflight() {
  log "target containers"
  docker ps -a --filter "name=omnicoder_target20b" --format "table {{.Names}}\t{{.Status}}\t{{.RunningFor}}" || true
  log "gpu snapshot"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,pstate --format=csv,noheader,nounits || true
  log "disk"
  df -h "$ROOT" /home/cereal || true
}

collect_curate() {
  local out="weights/data_factory/runs/${RUN_ID}"
  local codex_out="data/raw/codex_traces_2026/codex_ai_${RUN_ID}.jsonl"
  local claude_out="data/raw/claude_traces_2026/claude_ai_${RUN_ID}.jsonl"
  local hermes_out="data/raw/hermes_traces_2026/hermes_ai_${RUN_ID}.jsonl"
  local lmstudio_out="data/raw/lmstudio_traces_2026/lmstudio_ai_${RUN_ID}.jsonl"
  local comfy_out="data/raw/comfyui_outputs_2026/comfyui_${RUN_ID}.jsonl"
  local run_profile="$out/trace_profile.current_run.json"
  mkdir -p "$out/logs" data/raw/codex_traces_2026 data/raw/claude_traces_2026 data/raw/hermes_traces_2026 data/raw/lmstudio_traces_2026 data/raw/comfyui_outputs_2026 data/raw
  log "export agent-memory PostgreSQL audit"
  local am_export_ok=0
  if "$PYTHON_BIN" -m omnicoder.data_factory.curated_dataset_builder_2026 \
      --profile "$PROFILE" \
      --out-dir "weights/curated_datasets_2026/runs/${RUN_ID}" \
      export-agent-memory | tee "$out/logs/agent_memory_export.json"; then
    am_export_ok=1
  else
    log "agent-memory PostgreSQL/CLI export failed"
  fi
  if [[ "$am_export_ok" != "1" && "${OMNICODER_ALLOW_AGENT_MEMORY_FALLBACK:-1}" != "1" ]]; then
    echo "agent-memory export failed and OMNICODER_ALLOW_AGENT_MEMORY_FALLBACK is not enabled" >&2
    exit 13
  fi
  if [[ "$am_export_ok" != "1" ]]; then
    log "using pre-exported data/raw/agent_memory_events_2026.jsonl fallback"
  fi
  require_nonempty_jsonl "data/raw/agent_memory_events_2026.jsonl" "agent_memory_events"
  log "collect Codex traces"
  "$PYTHON_BIN" -m omnicoder.data_factory.memory_trace_collectors_2026 collect-codex \
    --input /home/cereal/.codex \
    --out "$codex_out" \
    --source-date 2026-05-24 --min-year 2025 --max-year 2026 --limit "$TRACE_LIMIT"
  require_nonempty_jsonl "$codex_out" "codex_traces"
  log "collect Claude traces"
  "$PYTHON_BIN" -m omnicoder.data_factory.memory_trace_collectors_2026 collect-claude \
    --input /home/cereal/.claude \
    --out "$claude_out" \
    --source-date 2026-05-24 --min-year 2025 --max-year 2026 --limit "$TRACE_LIMIT" \
    --source-kind auto
  require_nonempty_jsonl "$claude_out" "claude_traces"
  if [[ -d /home/cereal/.hermes ]]; then
    log "collect Hermes traces"
    "$PYTHON_BIN" -m omnicoder.data_factory.memory_trace_collectors_2026 collect-generic \
      --input /home/cereal/.hermes \
      --out "$hermes_out" \
      --source-date 2026-05-24 --min-year 2025 --max-year 2026 --limit "$TRACE_LIMIT" \
      --collector hermes_trace
    require_nonempty_jsonl "$hermes_out" "hermes_traces"
  fi
  if [[ -d /home/cereal/.lmstudio ]]; then
    log "collect LM Studio conversations"
    "$PYTHON_BIN" -m omnicoder.data_factory.memory_trace_collectors_2026 collect-generic \
      --input /home/cereal/.lmstudio \
      --out "$lmstudio_out" \
      --source-date 2026-05-24 --min-year 2025 --max-year 2026 --limit "$LMSTUDIO_TRACE_LIMIT" \
      --collector lmstudio_conversation
    require_nonempty_jsonl "$lmstudio_out" "lmstudio_traces"
  fi
  log "collect ComfyUI media manifests"
  "$PYTHON_BIN" -m omnicoder.data_factory.ingest_comfyui_outputs \
    --input /home/cereal/comfyui/output \
    --out "$comfy_out" \
    --dataset_name comfyui_outputs_2026 \
    --namespace train --bucket multimodal_media --split train \
    --source_date 2026-05-24 --license internal --limit 0
  require_nonempty_jsonl "$comfy_out" "comfyui_media_manifests"
  log "write current-run trace profile"
  "$PYTHON_BIN" - <<PY
import json
import os
from pathlib import Path

profile_path = Path("$PROFILE")
if not profile_path.is_absolute():
    profile_path = Path.cwd() / profile_path
profile = json.loads(profile_path.read_text(encoding="utf-8"))

sources = [
    {"harness": "agent_memory", "path": "data/raw/agent_memory_events_2026.jsonl"},
    {"harness": "codex", "path": "$codex_out"},
    {"harness": "claude", "path": "$claude_out"},
]
optional = [
    ("hermes", "$hermes_out"),
    ("lmstudio", "$lmstudio_out"),
    ("comfyui", "$comfy_out"),
]
for harness, raw in optional:
    path = Path(raw)
    if path.exists() and path.stat().st_size > 0:
        sources.append({"harness": harness, "path": raw})

trace_inputs = dict(profile.get("trace_inputs") if isinstance(profile.get("trace_inputs"), dict) else {})
trace_inputs["sources"] = sources
trace_inputs.setdefault("patterns", ["*.jsonl"])
profile["trace_inputs"] = trace_inputs

data = dict(profile.get("data") if isinstance(profile.get("data"), dict) else {})
data["limit"] = int(os.environ.get("OMNICODER_TRACE_ORCHESTRATOR_LIMIT", data.get("limit") or 250000) or 0)
data["per_file_limit"] = int(os.environ.get("OMNICODER_TRACE_ORCHESTRATOR_PER_FILE_LIMIT", data.get("per_file_limit") or 50000) or 0)
profile["data"] = data

builder = dict(profile.get("builder_2026") if isinstance(profile.get("builder_2026"), dict) else {})
builder["trace_limit"] = int(os.environ.get("OMNICODER_BUILDER_TRACE_LIMIT", builder.get("trace_limit") or 250000) or 0)
builder["per_trace_source_limit"] = int(os.environ.get("OMNICODER_BUILDER_PER_TRACE_SOURCE_LIMIT", builder.get("per_trace_source_limit") or 50000) or 0)
supplemental = dict(builder.get("supplemental_sources") if isinstance(builder.get("supplemental_sources"), dict) else {})
for key in ("long_context_roots", "text_roots"):
    values = supplemental.get(key)
    if isinstance(values, list):
        supplemental[key] = [
            value
            for value in values
            if "/weights/training_orchestration_2026" not in str(value)
            and "/weights/benchmarks_2026" not in str(value)
            and "/weights/data_factory" not in str(value)
        ]
builder["supplemental_sources"] = supplemental
profile["builder_2026"] = builder

out = Path("$run_profile")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(profile, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"status": "ok", "profile": str(out), "sources": sources}, sort_keys=True))
PY
  log "run trace orchestrator"
  "$PYTHON_BIN" -m omnicoder.data_factory.trace_orchestrator_2026 --profile "$run_profile" > "$out/logs/trace_orchestrator.log" 2>&1
  log "build run-scoped curated dataset"
  "$PYTHON_BIN" -m omnicoder.data_factory.curated_dataset_builder_2026 \
    --profile "$run_profile" \
    --out-dir "weights/curated_datasets_2026/runs/${RUN_ID}" \
    build | tee "$out/logs/curated_dataset_builder.json"
  if truthy "$PROMOTE_LATEST"; then
    ln -sfn "$ROOT/weights/curated_datasets_2026/runs/${RUN_ID}" weights/curated_datasets_2026/latest
    log "promoted curated dataset symlink to weights/curated_datasets_2026/runs/${RUN_ID}"
  else
    log "kept curated dataset run-scoped at weights/curated_datasets_2026/runs/${RUN_ID}"
  fi
}

external_expansion() {
  local out="weights/external_datasets_2026/runs/${RUN_ID}"
  local requirement_args=()
  local selection_args=()
  local materializer_args=()
  if [[ "$ENFORCE_DATASET_MINIMA" == "1" || "$ENFORCE_DATASET_MINIMA" == "true" ]]; then
    requirement_args+=(--enforce-requirements)
  fi
  if truthy "$MATERIALIZE_DEFERRED_SOURCES"; then
    materializer_args+=(--materialize-deferred-sources)
    log "external dataset expansion will materialize deferred live-download sources by override"
  fi
  if truthy "$MATERIALIZE_HF_SOURCES"; then
    materializer_args+=(--materialize-hf-sources)
    log "external dataset expansion will allow live Hugging Face materialization by override"
  fi
  materializer_args+=(--hf-step-timeout-seconds "$HF_STEP_TIMEOUT_SECONDS")
  if [[ -n "$DATASET_INCLUDE_WAVES" ]]; then
    local old_ifs="$IFS"
    IFS=","
    read -ra waves <<< "$DATASET_INCLUDE_WAVES"
    IFS="$old_ifs"
    for wave in "${waves[@]}"; do
      wave="${wave//[[:space:]]/}"
      [[ -n "$wave" ]] && selection_args+=(--include-wave "$wave")
    done
    log "external dataset expansion is filtered to registry wave(s): $DATASET_INCLUDE_WAVES"
  fi
  if [[ -n "$DATASET_INCLUDE_FAMILIES" ]]; then
    local old_ifs="$IFS"
    IFS=","
    read -ra families <<< "$DATASET_INCLUDE_FAMILIES"
    IFS="$old_ifs"
    for family in "${families[@]}"; do
      family="${family//[[:space:]]/}"
      [[ -n "$family" ]] && selection_args+=(--include-family "$family")
    done
    log "external dataset expansion is filtered to registry family/families: $DATASET_INCLUDE_FAMILIES"
  fi
  if [[ -n "$DATASET_INCLUDE_NAMES" ]]; then
    local old_ifs="$IFS"
    IFS=","
    read -ra names <<< "$DATASET_INCLUDE_NAMES"
    IFS="$old_ifs"
    for name in "${names[@]}"; do
      name="${name#"${name%%[![:space:]]*}"}"
      name="${name%"${name##*[![:space:]]}"}"
      [[ -n "$name" ]] && selection_args+=(--include-name "$name")
    done
    log "external dataset expansion is filtered to registry dataset name(s): $DATASET_INCLUDE_NAMES"
  fi
  mkdir -p "$out"
  log "build external dataset expansion"
  "$PYTHON_BIN" -m omnicoder.data_factory.dataset_expansion_2026 \
    --profile "$PROFILE" \
    --out-dir "$out" \
    --download \
    --max-records-per-dataset "$MAX_RECORDS_PER_DATASET" \
    "${materializer_args[@]}" \
    "${selection_args[@]}" \
    "${requirement_args[@]}" \
    build | tee "$out/external_dataset_manifest.stdout.json"
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
p = Path("weights/external_datasets_2026/runs") / Path(__import__("os").environ["RUN_ID"]) / "manifests" / "external_dataset_manifest.json"
data = json.loads(p.read_text())
records = data.get("records", {})
if int(records.get("train") or 0) <= 0:
    raise SystemExit("external expansion produced no train rows; refusing latest promotion")
if data.get("status") != "passed":
    raise SystemExit(f"external expansion requirements failed: {json.dumps(data.get('requirement_report', {}), sort_keys=True)[:4000]}")
train_path = Path(data.get("training_paths", {}).get("train_all_external", ""))
if not train_path.is_absolute():
    train_path = Path.cwd() / train_path
synthetic_train = 0
with train_path.open("r", encoding="utf-8", errors="ignore") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("synthetic_seed_only"):
            synthetic_train += 1
if synthetic_train:
    raise SystemExit(f"external expansion attempted to promote {synthetic_train} synthetic seed rows into train")
PY
  EXTERNAL_DATASET_SOURCE="$out"
  log "kept external dataset run-scoped until integrity and index gates pass: $out"
  log "strict integrity scan for external train bucket"
  mkdir -p "$out/integrity"
  "$PYTHON_BIN" -m omnicoder.data_factory.dataset_integrity_2026 \
    --input "$out/jsonl/train_all_external.jsonl" \
    --out-dir "$out/integrity" \
    --write-accepted \
    | tee "$out/integrity/dataset_integrity.stdout.json"
  "$PYTHON_BIN" - <<'PY'
import json
import os
import hashlib
from pathlib import Path
run_id = os.environ["RUN_ID"]
run_dir = Path("weights/external_datasets_2026/runs") / run_id
manifest = run_dir / "integrity" / "dataset_integrity_manifest.json"
data = json.loads(manifest.read_text(encoding="utf-8"))
rejected = int(data.get("rejected") or 0)
accepted = int(data.get("accepted") or 0)
accepted_path = Path(data.get("accepted_jsonl") or "")
rejected_path = Path(data.get("rejected_jsonl") or "")
if accepted <= 0 or not accepted_path.exists():
    raise SystemExit(f"external train bucket integrity scan accepted no rows; refusing train promotion: {json.dumps(data.get('counts', {}), sort_keys=True)}")
rejected_audit_path = run_dir / "integrity" / "dataset_integrity_rejected_audit.jsonl"
rejected_payload_deleted = False
if rejected > 0 and rejected_path.exists():
    with rejected_path.open("r", encoding="utf-8", errors="ignore") as src, rejected_audit_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            payload = json.dumps(row, ensure_ascii=True, sort_keys=True, default=str, separators=(",", ":"))
            integrity = row.get("dataset_integrity_2026")
            integrity_reasons = integrity.get("reasons") if isinstance(integrity, dict) else None
            audit = {
                "record_id": row.get("record_id") or row.get("id") or row.get("source_id"),
                "source_id": row.get("source_id"),
                "split": row.get("split"),
                "modality": row.get("modality"),
                "payload_sha256": row.get("payload_sha256") or row.get("sha256") or hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest(),
                "rejection_reasons": row.get("rejection_reasons") or row.get("reasons") or row.get("integrity_reasons") or integrity_reasons,
            }
            dst.write(json.dumps(audit, ensure_ascii=True, sort_keys=True) + "\n")
    rejected_path.unlink()
    rejected_payload_deleted = True
    data["rejected_jsonl_payload_removed"] = True
    data["rejected_jsonl_deleted"] = str(rejected_path)
    data["rejected_audit_jsonl"] = str(rejected_audit_path)
    data["rejected_jsonl"] = ""
    manifest.write_text(json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({
    "status": "integrity_checked",
    "accepted_rows": accepted,
    "rejected_rows": rejected,
    "rejected_payload_deleted": rejected_payload_deleted,
    "rejected_audit_jsonl": str(rejected_audit_path) if rejected > 0 else "",
    "counts": data.get("counts", {}),
}, sort_keys=True))
PY
  "$PYTHON_BIN" -m omnicoder.data_factory.external_train_rewrite_2026 \
    --accepted "$out/integrity/dataset_integrity_accepted.jsonl" \
    --jsonl-dir "$out/jsonl" \
    --source-manifest "$out/manifests/external_dataset_manifest.json" \
    --out "$out/integrity/train_bucket_integrity_rewrite.json" \
    | tee "$out/integrity/train_bucket_integrity_rewrite.stdout.json"
  log "index external train bucket"
  "$PYTHON_BIN" -m omnicoder.data_factory.dataset_index_2026 \
    --input "$out/jsonl/train_all_external.jsonl" \
    --out "$out/integrity/train_all_external.index.json" \
    --expected-split train \
    | tee "$out/integrity/train_all_external.index.stdout.json"
  "$PYTHON_BIN" - "$out/manifests/external_dataset_manifest.json" "$out/integrity/train_all_external.index.json" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
index_path = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
index = json.loads(index_path.read_text(encoding="utf-8"))
if index.get("status") != "passed":
    raise SystemExit(
        "external train index did not pass: "
        + json.dumps(index.get("fail_reasons", []), ensure_ascii=True, sort_keys=True)
    )
manifest["promotion_allowed"] = True
manifest["promotion_status"] = "integrity_rewrite_and_index_passed"
manifest["promotion_index"] = {
    "counts": index.get("counts", {}),
    "path": str(index_path),
    "rows": index.get("rows"),
    "status": index.get("status"),
}
manifest_path.write_text(
    json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
  if truthy "$PROMOTE_LATEST"; then
    ln -sfn "$ROOT/$out" weights/external_datasets_2026/latest
    EXTERNAL_DATASET_SOURCE="weights/external_datasets_2026/latest"
    log "promoted external dataset symlink to $out after integrity and index gates"
  else
    log "kept external dataset run-scoped at $out"
  fi
}

agentic_tool_training() {
  local run_out="weights/agentic_tool_training_2026/runs/${RUN_ID}"
  local out="$run_out"
  local source="${OMNICODER_TRACE_WORK_DIR}/jsonl/contamination_scanned.jsonl"
  mkdir -p "$out"
  require_nonempty_jsonl "$source" "trace_orchestrator_contamination_scanned"
  log "build agentic tool SFT/reward/preference/RLVR exports"
  "$PYTHON_BIN" -m omnicoder.training.agentic_tool_training_2026 \
    --profile profiles/agentic_tool_training_2026.json \
    build \
    --input "$source" \
    --out-dir "$out" \
    --limit 0 | tee "$run_out/agentic_tool_training_manifest.stdout.json"
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
p = Path("weights/agentic_tool_training_2026/runs") / Path(os.environ["RUN_ID"]) / "agentic_tool_training_manifest.json"
data = json.loads(p.read_text())
counts = data.get("counts", {})
required = ["sft", "reward", "preference", "rlvr", "tool_rlvr"]
missing = {name: int(counts.get(name) or 0) for name in required if int(counts.get(name) or 0) <= 0}
if missing:
    raise SystemExit(f"agentic tool training produced empty required exports: {missing}")
PY
  if truthy "$PROMOTE_LATEST"; then
    ln -sfn "$ROOT/$run_out" weights/agentic_tool_training_2026/latest_run
  fi
  if truthy "$PROMOTE_SHARED_ARTIFACTS"; then
    cp "$run_out"/*.jsonl weights/agentic_tool_training_2026/ 2>/dev/null || true
    cp "$run_out"/agentic_tool_training_manifest.json weights/agentic_tool_training_2026/ 2>/dev/null || true
    log "promoted agentic tool exports to shared weights/agentic_tool_training_2026"
  fi
  log "agentic tool training exports refreshed in $out"
}

local_trace_bundle() {
  local out="weights/curated_datasets_2026/runs/${RUN_ID}_local_traces"
  mkdir -p "weights/data_factory/runs/${RUN_ID}"
  log "build strict-date local Codex/Claude/agent-memory trace bundle"
  "$PYTHON_BIN" -m omnicoder.data_factory.curated_dataset_builder_2026 \
    --profile "$PROFILE" \
    --out-dir "$out" \
    export-local-traces | tee "weights/data_factory/runs/${RUN_ID}/local_trace_export.stdout.json"
  require_nonempty_jsonl "$out/raw/normalized_traces.jsonl" "strict_local_normalized_traces"
  if [[ -s "$out/jsonl/train_agentic_focus.jsonl" ]]; then
    LOCAL_TRACE_SOURCE="$out"
  fi
  if truthy "$PROMOTE_LATEST"; then
    ln -sfn "$ROOT/$out" weights/curated_datasets_2026/latest_local_traces
  fi
  log "local trace bundle dir: $out"
}

build_jobs_if_present() {
  local records="$1"
  local job_type="$2"
  local out="$3"
  if [[ -s "$records" ]]; then
    "$PYTHON_BIN" -m omnicoder.data_factory.teacher_jobs_2026 build \
      --records "$records" \
      --teacher qwen3.6_27b_q4_local \
      --job_type "$job_type" \
      --limit "$TEACHER_LIMIT" \
      --out "$out"
  else
    log "skip teacher jobs; missing or empty records: $records"
    : > "$out"
  fi
}

build_teacher_jobs_if_present() {
  local records="$1"
  local teacher="$2"
  local job_type="$3"
  local out="$4"
  if [[ -s "$records" ]]; then
    "$PYTHON_BIN" -m omnicoder.data_factory.teacher_jobs_2026 build \
      --records "$records" \
      --teacher "$teacher" \
      --job_type "$job_type" \
      --limit "$TEACHER_LIMIT" \
      --out "$out"
  else
    log "skip $teacher teacher jobs; missing or empty records: $records"
    : > "$out"
  fi
}

teacher_jobs() {
  local job_dir="${TEACHER_JOB_ROOT}/${RUN_ID}"
  mkdir -p "$job_dir"
  log "build agentic/math/code/tool teacher jobs"
  build_jobs_if_present "${CURATED_DATASET_SOURCE}/jsonl/train_agentic_focus.jsonl" agentic_math_code_tool_critique "$job_dir/agentic_jobs.jsonl"
  build_jobs_if_present "${LOCAL_TRACE_SOURCE}/jsonl/train_agentic_focus.jsonl" strict_local_trace_replay_critique "$job_dir/local_trace_jobs.jsonl"
  build_jobs_if_present "${CURATED_DATASET_SOURCE}/jsonl/train_code.jsonl" code_repair_reasoning_critique "$job_dir/code_jobs.jsonl"
  build_jobs_if_present "${CURATED_DATASET_SOURCE}/jsonl/train_tool.jsonl" tool_call_replay_reward_critique "$job_dir/tool_jobs.jsonl"
  build_jobs_if_present "${EXTERNAL_DATASET_SOURCE}/jsonl/math_reasoning.jsonl" math_rlvr_answer_critique "$job_dir/math_jobs.jsonl"
  build_jobs_if_present "${EXTERNAL_DATASET_SOURCE}/jsonl/coding_agentic.jsonl" coding_agent_trajectory_critique "$job_dir/external_code_jobs.jsonl"
  build_jobs_if_present "${EXTERNAL_DATASET_SOURCE}/jsonl/agentic_tool_reasoning.jsonl" agentic_tool_reasoning_critique "$job_dir/external_tool_jobs.jsonl"
  build_jobs_if_present "${EXTERNAL_DATASET_SOURCE}/jsonl/terminal_browser_agents.jsonl" terminal_browser_agent_critique "$job_dir/external_terminal_jobs.jsonl"
  build_jobs_if_present "${EXTERNAL_DATASET_SOURCE}/jsonl/research_internal_all_external.jsonl" research_internal_distillation_review "$job_dir/research_internal_jobs.jsonl"
  local job_files=()
  mapfile -d '' job_files < <(find "$job_dir" -maxdepth 1 -name '*_jobs.jsonl' -type f -size +0c -print0 | sort -z)
  if [[ "${#job_files[@]}" -eq 0 ]]; then
    echo "no teacher jobs produced" >&2
    exit 3
  fi
  cat "${job_files[@]}" > "$job_dir/all_jobs.jsonl"
  awk 'NR % 3 == 1' "$job_dir/all_jobs.jsonl" > "$job_dir/shard_gpu1.jsonl"
  awk 'NR % 3 == 2' "$job_dir/all_jobs.jsonl" > "$job_dir/shard_gpu2.jsonl"
  awk 'NR % 3 == 0' "$job_dir/all_jobs.jsonl" > "$job_dir/shard_gpu3.jsonl"
  if truthy "$PROMOTE_LATEST"; then
    ln -sfn "$ROOT/$job_dir" "${TEACHER_JOB_ROOT}/latest"
  fi
  wc -l "$job_dir"/*.jsonl
  log "teacher job dir: $job_dir"
}

modality_teacher_jobs() {
  local job_dir="${TEACHER_JOB_ROOT}/${RUN_ID}/modality"
  mkdir -p "$job_dir"
  log "build image/video/audio/music teacher jobs for multimodal distillation"
  build_teacher_jobs_if_present "${CURATED_DATASET_SOURCE}/jsonl/train_media_focus.jsonl" qwen3_omni_optional multimodal_alignment "$job_dir/curated_media_omni_jobs.jsonl"
  build_teacher_jobs_if_present "${EXTERNAL_DATASET_SOURCE}/jsonl/image_generation_editing.jsonl" qwen_image_generate image_reward_label "$job_dir/image_reward_jobs.jsonl"
  build_teacher_jobs_if_present "${EXTERNAL_DATASET_SOURCE}/jsonl/image_generation_editing.jsonl" qwen_image_edit image_edit_critique "$job_dir/image_edit_jobs.jsonl"
  build_teacher_jobs_if_present "${EXTERNAL_DATASET_SOURCE}/jsonl/video_generation.jsonl" ltx_2_3 temporal_reward_label "$job_dir/video_temporal_jobs.jsonl"
  build_teacher_jobs_if_present "${EXTERNAL_DATASET_SOURCE}/jsonl/video_generation.jsonl" ltx_2_3 image_to_video_plan "$job_dir/image_to_video_jobs.jsonl"
  build_teacher_jobs_if_present "${EXTERNAL_DATASET_SOURCE}/jsonl/audio_music_speech.jsonl" qwen3_omni_optional audio_video_understanding "$job_dir/audio_omni_jobs.jsonl"
  build_teacher_jobs_if_present "${EXTERNAL_DATASET_SOURCE}/jsonl/speech_audio.jsonl" qwen3_omni_optional speech_caption "$job_dir/speech_caption_jobs.jsonl"
  build_teacher_jobs_if_present "${EXTERNAL_DATASET_SOURCE}/jsonl/music_generation.jsonl" ace_step_1_5 music_reward_label "$job_dir/music_reward_jobs.jsonl"
  build_teacher_jobs_if_present "${EXTERNAL_DATASET_SOURCE}/jsonl/music_generation.jsonl" ace_step_1_5 music_plan "$job_dir/music_plan_jobs.jsonl"
  build_teacher_jobs_if_present "${EXTERNAL_DATASET_SOURCE}/jsonl/omnimodal_understanding.jsonl" gemini_omni_optional cross_modal_reward "$job_dir/omni_cross_modal_jobs.jsonl"
  local job_files=()
  mapfile -d '' job_files < <(find "$job_dir" -maxdepth 1 -name '*_jobs.jsonl' -type f -size +0c -print0 | sort -z)
  if [[ "${#job_files[@]}" -gt 0 ]]; then
    cat "${job_files[@]}" > "$job_dir/all_modality_teacher_jobs.jsonl"
  else
    : > "$job_dir/all_modality_teacher_jobs.jsonl"
  fi
  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
job_dir = Path("$job_dir")
counts = {}
for path in sorted(job_dir.glob("*_jobs.jsonl")):
    counts[path.name] = sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())
(job_dir / "modality_teacher_jobs_manifest.json").write_text(
    json.dumps({"status": "ok", "run_id": "$RUN_ID", "counts": counts}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
  wc -l "$job_dir"/*.jsonl
  log "modality teacher job dir: $job_dir"
}

mix_plan() {
  local out_dir="weights/training_orchestration_2026/runs/${RUN_ID}"
  local curation_manifest="weights/curated_datasets_2026/runs/${RUN_ID}/manifests/curation_manifest.json"
  local external_manifest="${EXTERNAL_DATASET_SOURCE}/manifests/external_dataset_manifest.json"
  local agentic_manifest="weights/agentic_tool_training_2026/runs/${RUN_ID}/agentic_tool_training_manifest.json"
  local teacher_manifest="${TEACHER_JOB_ROOT}/${RUN_ID}/modality/modality_teacher_jobs_manifest.json"
  mkdir -p "$out_dir/manifests"
  log "build adaptive mixture plan"
  "$PYTHON_BIN" -m omnicoder.training.training_orchestration_2026 \
    --profile profiles/training_orchestration_2026.json \
    --out-dir "$out_dir" \
    mix-plan \
    --curation-manifest "$curation_manifest" \
    --external-manifest "$external_manifest" \
    --agentic-manifest "$agentic_manifest" \
    --teacher-manifest "$teacher_manifest" \
    --output "$out_dir/manifests/mixture_plan.json" \
    | tee "$out_dir/manifests/mixture_plan.stdout.json"
  if [[ ! -s "$out_dir/manifests/mixture_plan.json" ]]; then
    echo "adaptive mixture plan was not written" >&2
    exit 14
  fi
  if truthy "$PROMOTE_LATEST"; then
    mkdir -p weights/training_orchestration_2026/manifests
    cp "$out_dir/manifests/mixture_plan.json" weights/training_orchestration_2026/manifests/mixture_plan.json
  fi
  log "mixture plan: $out_dir/manifests/mixture_plan.json"
}

p40_teacher_rollouts() {
  local job_dir="${TEACHER_JOB_SOURCE}"
  local out_dir="weights/data_factory/teacher_rollouts/${RUN_ID}"
  mkdir -p "$out_dir/logs"
  log "launch P40 teacher rollouts"
  local pids=()
  if [[ -s "$job_dir/shard_gpu1.jsonl" ]]; then
    CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -m omnicoder.data_factory.openai_teacher_rollout_2026 \
      --input "$job_dir/shard_gpu1.jsonl" \
      --out "$out_dir/qwen36_gpu1.jsonl" \
      --base-url "$TEACHER_BASE_URL" \
      --model "$TEACHER_MODEL_GPU1" \
      --limit "$TEACHER_LIMIT" --max-tokens 1024 --temperature 0.2 --timeout 180 --sleep 2 \
      --record-kind qwen36_p40_agentic_math_code_tool \
      --thermal-gpu-index 1 --max-gpu-temp "$MAX_GPU_TEMP" \
      --resume \
      > "$out_dir/logs/gpu1.log" 2>&1 &
    pids+=("$!")
  fi
  if [[ -s "$job_dir/shard_gpu2.jsonl" ]]; then
    CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -m omnicoder.data_factory.openai_teacher_rollout_2026 \
      --input "$job_dir/shard_gpu2.jsonl" \
      --out "$out_dir/qwen36_gpu2.jsonl" \
      --base-url "$TEACHER_BASE_URL" \
      --model "$TEACHER_MODEL_GPU2" \
      --limit "$TEACHER_LIMIT" --max-tokens 1024 --temperature 0.2 --timeout 180 --sleep 2 \
      --record-kind qwen36_p40_agentic_math_code_tool \
      --thermal-gpu-index 2 --max-gpu-temp "$MAX_GPU_TEMP" \
      --resume \
      > "$out_dir/logs/gpu2.log" 2>&1 &
    pids+=("$!")
  fi
  if [[ -s "$job_dir/shard_gpu3.jsonl" ]]; then
    CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -m omnicoder.data_factory.openai_teacher_rollout_2026 \
      --input "$job_dir/shard_gpu3.jsonl" \
      --out "$out_dir/qwen36_gpu3.jsonl" \
      --base-url "$TEACHER_BASE_URL" \
      --model "$TEACHER_MODEL_GPU3" \
      --limit "$TEACHER_LIMIT" --max-tokens 1024 --temperature 0.2 --timeout 180 --sleep 2 \
      --record-kind qwen36_p40_agentic_math_code_tool \
      --thermal-gpu-index 3 --max-gpu-temp "$MAX_GPU_TEMP" \
      --resume \
      > "$out_dir/logs/gpu3.log" 2>&1 &
    pids+=("$!")
  fi
  if [[ "${#pids[@]}" -eq 0 ]]; then
    echo "no nonempty teacher shards found in $job_dir" >&2
    exit 4
  fi
  local failures=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failures=$((failures + 1))
    fi
  done
  cat "$out_dir"/qwen36_gpu*.jsonl > "$out_dir/qwen36_agentic_math_code_tool.jsonl" 2>/dev/null || true
  if [[ ! -s "$out_dir/qwen36_agentic_math_code_tool.jsonl" ]]; then
    echo "teacher rollouts produced no combined rows" >&2
    exit 5
  fi
  local refreshed_source="weights/agentic_tool_training_2026/runs/${RUN_ID}/trace_plus_teacher_rollouts.jsonl"
  local refreshed_out="weights/agentic_tool_training_2026/runs/${RUN_ID}/after_teacher"
  mkdir -p "$(dirname "$refreshed_source")"
  if [[ -s "${OMNICODER_TRACE_WORK_DIR}/jsonl/contamination_scanned.jsonl" ]]; then
    cat "${OMNICODER_TRACE_WORK_DIR}/jsonl/contamination_scanned.jsonl" "$out_dir/qwen36_agentic_math_code_tool.jsonl" > "$refreshed_source"
  else
    cp "$out_dir/qwen36_agentic_math_code_tool.jsonl" "$refreshed_source"
  fi
  log "refresh agentic RLVR exports with teacher rollout rows"
  "$PYTHON_BIN" -m omnicoder.training.agentic_tool_training_2026 \
    --profile profiles/agentic_tool_training_2026.json \
    build \
    --input "$refreshed_source" \
    --out-dir "$refreshed_out" \
    --limit 0 > "weights/agentic_tool_training_2026/runs/${RUN_ID}/agentic_tool_training_after_teacher.stdout.json"
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
p = Path("weights/agentic_tool_training_2026/runs") / Path(os.environ["RUN_ID"]) / "after_teacher" / "agentic_tool_training_manifest.json"
data = json.loads(p.read_text())
counts = data.get("counts", {})
required = ["sft", "reward", "preference", "rlvr", "tool_rlvr"]
missing = {name: int(counts.get(name) or 0) for name in required if int(counts.get(name) or 0) <= 0}
if missing:
    raise SystemExit(f"teacher-refreshed agentic exports are empty: {missing}")
PY
  if truthy "$PROMOTE_SHARED_ARTIFACTS"; then
    cp "$refreshed_out"/*.jsonl weights/agentic_tool_training_2026/ 2>/dev/null || true
    cp "$refreshed_out"/agentic_tool_training_manifest.json weights/agentic_tool_training_2026/ 2>/dev/null || true
    log "promoted teacher-refreshed agentic exports to shared weights/agentic_tool_training_2026"
  fi
  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
out = Path("$out_dir")
counts = {}
for path in sorted(out.glob("qwen36*.jsonl")):
    counts[path.name] = sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())
(out / "teacher_rollout_manifest.json").write_text(json.dumps({"status": "ok", "run_id": "$RUN_ID", "failures": $failures, "counts": counts}, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
  if truthy "$PROMOTE_LATEST"; then
    ln -sfn "$ROOT/$out_dir" weights/data_factory/teacher_rollouts/latest
  fi
  if [[ "$failures" -gt 0 ]]; then
    log "teacher rollout dir: $out_dir with $failures failed worker(s); combined nonempty output promoted"
  else
    log "teacher rollout dir: $out_dir"
  fi
}

media_teacher_rollouts() {
  local job_dir="${TEACHER_JOB_ROOT}/${RUN_ID}/modality"
  local jobs="$job_dir/all_modality_teacher_jobs.jsonl"
  local out_dir="weights/data_factory/teacher_rollouts/${RUN_ID}"
  mkdir -p "$out_dir/logs"
  if [[ ! -s "$jobs" ]]; then
    echo "no queued modality teacher jobs found: $jobs" >&2
    exit 7
  fi
  log "run media teacher rollouts mode=${MEDIA_TEACHER_ROLLOUT_MODE}"
  local args=(
    -m omnicoder.data_factory.media_teacher_rollouts_2026
    --input "$jobs"
    --out-dir "$out_dir"
    --mode "$MEDIA_TEACHER_ROLLOUT_MODE"
    --limit "$MEDIA_TEACHER_LIMIT"
    --resume
  )
  if [[ "$MEDIA_TEACHER_ROLLOUT_MODE" == "live" ]]; then
    args+=(--strict-live)
  fi
  "$PYTHON_BIN" "${args[@]}" | tee "$out_dir/logs/media_teacher_rollouts.stdout.json"
  if [[ ! -s "$out_dir/media_teacher_rollouts.jsonl" ]]; then
    echo "media teacher rollouts produced no rows" >&2
    exit 8
  fi
  if truthy "$PROMOTE_LATEST"; then
    ln -sfn "$ROOT/$out_dir" weights/data_factory/teacher_rollouts/latest
  fi
  log "media teacher rollout dir: $out_dir"
}

benchmark_materialize() {
  local out_dir="$BENCHMARK_MATERIALIZATION_ROOT"
  local manifest="$out_dir/manifests/benchmark_materialization_manifest.json"
  mkdir -p "$out_dir/logs"
  local args=(
    -m omnicoder.data_factory.benchmark_materializer_2026
    --profile "$BENCHMARK_PROFILE"
    --run-id "$RUN_ID"
    --out-root "$out_dir"
    --manifest-out "$manifest"
    --suite "$BENCHMARK_MATERIALIZATION_SUITE"
    --mode "$BENCHMARK_MATERIALIZATION_MODE"
    --limit "$BENCHMARK_MATERIALIZATION_LIMIT"
  )
  if truthy "$BENCHMARK_MATERIALIZE_DOWNLOAD"; then
    args+=(--download)
  fi
  if truthy "$BENCHMARK_MATERIALIZE_STRICT"; then
    args+=(--strict)
  fi
  if truthy "$BENCHMARK_MATERIALIZE_PROFILE_ROOTS"; then
    args+=(--write-profile-reportable-roots)
  fi
  args+=(materialize)
  log "materialize official/public benchmark task JSONLs"
  CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "${args[@]}" | tee "$out_dir/logs/benchmark_materializer.stdout.json"
  log "benchmark materialization manifest: $manifest"
}

local_hf_trainer() {
  require_non_qwen_fast_hf_model
  local source="$LOCAL_HF_TRAIN_JSONL"
  if [[ -z "$source" ]]; then
    if [[ -s "weights/agentic_tool_training_2026/latest_run/tool_sft.jsonl" ]]; then
      source="weights/agentic_tool_training_2026/latest_run/tool_sft.jsonl"
    elif [[ -s "weights/external_datasets_2026/latest/jsonl/train_all_external.jsonl" ]]; then
      source="weights/external_datasets_2026/latest/jsonl/train_all_external.jsonl"
    elif [[ -s "weights/curated_datasets_2026/latest/jsonl/train_all.jsonl" ]]; then
      source="weights/curated_datasets_2026/latest/jsonl/train_all.jsonl"
    fi
  fi
  if [[ -z "$source" || ! -s "$source" ]]; then
    echo "local HF trainer source is missing; set OMNICODER_LOCAL_HF_TRAIN_JSONL" >&2
    exit 9
  fi
  local out_dir="weights/local_hf_trainer_2026/runs/${RUN_ID}"
  local manifest="$out_dir/local_hf_trainer_manifest.json"
  local args=(
    -m omnicoder.training.local_hf_trainer_bridge_2026
    sft
    --backend "$LOCAL_HF_BACKEND"
    --model "$LOCAL_HF_MODEL"
    --train-jsonl "$source"
    --out-dir "$out_dir"
    --manifest "$manifest"
    --max-seq-len "$LOCAL_HF_MAX_SEQ_LEN"
    --max-steps "$LOCAL_HF_MAX_STEPS"
    --load-in-4bit
    --packing
    --assistant-only-loss
    --protected-gpus "$LOCAL_HF_PROTECTED_GPUS"
  )
  if [[ -n "$LOCAL_HF_HOST_GPU_IDS" ]]; then
    args+=(--host-gpu-ids "$LOCAL_HF_HOST_GPU_IDS")
  fi
  if truthy "$LOCAL_HF_DRY_RUN"; then
    args+=(--dry-run)
  fi
  mkdir -p "$out_dir/logs"
  log "validate/run optional local HF trainer backend=${LOCAL_HF_BACKEND}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" "$PYTHON_BIN" "${args[@]}" | tee "$out_dir/logs/local_hf_trainer.stdout.json"
  log "local HF trainer manifest: $manifest"
}

status() {
  preflight
  log "latest external manifest"
  if [[ -f weights/external_datasets_2026/latest/manifests/external_dataset_manifest.json ]]; then
    "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
p = Path("weights/external_datasets_2026/latest/manifests/external_dataset_manifest.json")
print(json.dumps(json.loads(p.read_text()).get("records", {}), indent=2))
PY
  fi
  log "teacher rollout counts"
  find weights/data_factory/teacher_rollouts/latest -maxdepth 1 -name '*.jsonl' -print -exec wc -l {} \; 2>/dev/null || true
  log "agentic tool training counts"
  find weights/agentic_tool_training_2026 -maxdepth 1 -name 'tool_*.jsonl' -print -exec wc -l {} \; 2>/dev/null || true
  log "modality teacher job counts"
  find "${TEACHER_JOB_ROOT}/latest/modality" -maxdepth 1 -name '*.jsonl' -print -exec wc -l {} \; 2>/dev/null || true
  log "benchmark materialization"
  if [[ -f "$BENCHMARK_MATERIALIZATION_ROOT/manifests/benchmark_materialization_manifest.json" ]]; then
    "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
p = Path("$BENCHMARK_MATERIALIZATION_ROOT/manifests/benchmark_materialization_manifest.json")
data = json.loads(p.read_text(encoding="utf-8"))
print(json.dumps({k: data.get(k) for k in ("run_id", "mode", "materialized", "needs_data", "rows")}, indent=2))
PY
  else
    echo "missing: $BENCHMARK_MATERIALIZATION_ROOT/manifests/benchmark_materialization_manifest.json"
  fi
}

coverage_report() {
  local out_dir="weights/data_factory/runs/${RUN_ID}"
  local report="$out_dir/coverage_report.json"
  local benchmark_manifest="$BENCHMARK_MATERIALIZATION_ROOT/manifests/benchmark_materialization_manifest.json"
  local args=(--root "$ROOT" --run-id "$RUN_ID" --out "$report")
  if [[ -n "$REPORTABLE_ROOT" ]]; then
    args+=(--reportable-root "$REPORTABLE_ROOT")
  elif [[ -d "$BENCHMARK_MATERIALIZATION_ROOT/reportable_2026" ]]; then
    args+=(--reportable-root "$BENCHMARK_MATERIALIZATION_ROOT/reportable_2026")
  fi
  if [[ -f "$benchmark_manifest" ]]; then
    args+=(--benchmark-materialization-manifest "$benchmark_manifest")
  fi
  if truthy "$COVERAGE_STRICT"; then
    args+=(--strict)
  fi
  if truthy "$REQUIRE_MEDIA_TEACHER_ROLLOUTS"; then
    args+=(--require-media-teacher-rollouts)
  fi
  if truthy "$REQUIRE_REPORTABLE_TASKS"; then
    args+=(--require-reportable-tasks)
  fi
  if truthy "$REQUIRE_OFFICIAL_REPORTABLE_TASKS"; then
    args+=(--require-official-reportable-tasks)
  fi
  log "validate run-scoped materialized coverage"
  "$PYTHON_BIN" -m omnicoder.data_factory.coverage_validator_2026 "${args[@]}" | tee "$out_dir/coverage_report.stdout.json"
}

case "$ACTION" in
  preflight) preflight ;;
  collect-curate) collect_curate ;;
  external-expansion) external_expansion ;;
  agentic-tool-training) agentic_tool_training ;;
  local-traces) local_trace_bundle ;;
  teacher-jobs) teacher_jobs ;;
  modality-teacher-jobs) modality_teacher_jobs ;;
  mix-plan) mix_plan ;;
  p40-teacher) p40_teacher_rollouts ;;
  media-teacher-rollouts) media_teacher_rollouts ;;
  benchmark-materialize) benchmark_materialize ;;
  local-hf-trainer) local_hf_trainer ;;
  coverage-report) coverage_report ;;
  status) status ;;
  all)
    preflight
    collect_curate
    local_trace_bundle
    agentic_tool_training
    external_expansion
    teacher_jobs
    modality_teacher_jobs
    if truthy "$MATERIALIZE_BENCHMARK_TASKS"; then
      benchmark_materialize
    fi
    mix_plan
    p40_teacher_rollouts
    media_teacher_rollouts
    coverage_report
    status
    ;;
  *)
    echo "unknown action: $ACTION" >&2
    exit 2
    ;;
esac
