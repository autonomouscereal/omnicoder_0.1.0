#!/usr/bin/env bash
set -euo pipefail

ROOT="${OMNICODER_REPO:-/home/cereal/omnicoder_2026_work}"
RUN_ID="${OMNICODER_RUN_ID:-dataset_sidecars_$(date -u +%Y%m%dT%H%M%SZ)}"
PROFILE="${OMNICODER_DATASET_PROFILE:-profiles/dataset_curation_2026.json}"
TEACHER_MODEL="${OMNICODER_TEACHER_MODEL:-qwen3.6-27b-q4}"
MAX_RECORDS_PER_DATASET="${OMNICODER_MAX_RECORDS_PER_DATASET:-1024}"
TEACHER_LIMIT="${OMNICODER_TEACHER_LIMIT:-256}"
MAX_GPU_TEMP="${OMNICODER_MAX_GPU_TEMP:-78}"
TEACHER_JOB_ROOT="${OMNICODER_TEACHER_JOB_ROOT:-weights/data_factory/runs/teacher_jobs}"
PYTHON_BIN="${OMNICODER_DATA_PYTHON:-python3}"
ENFORCE_DATASET_MINIMA="${OMNICODER_ENFORCE_DATASET_MINIMA:-1}"
TRACE_LIMIT="${OMNICODER_TRACE_LIMIT:-0}"
LMSTUDIO_TRACE_LIMIT="${OMNICODER_LMSTUDIO_TRACE_LIMIT:-100000}"
ACTION="${1:-all}"

cd "$ROOT"
export RUN_ID
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export OMNICODER_TRACE_WORK_DIR="${OMNICODER_TRACE_WORK_DIR:-weights/data_factory/runs/trace_orchestrator/${RUN_ID}}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

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
  log "run trace orchestrator"
  "$PYTHON_BIN" -m omnicoder.data_factory.trace_orchestrator_2026 --profile "$PROFILE" > "$out/logs/trace_orchestrator.log" 2>&1
  log "build run-scoped curated dataset"
  "$PYTHON_BIN" -m omnicoder.data_factory.curated_dataset_builder_2026 \
    --profile "$PROFILE" \
    --out-dir "weights/curated_datasets_2026/runs/${RUN_ID}" \
    build | tee "$out/logs/curated_dataset_builder.json"
  ln -sfn "$ROOT/weights/curated_datasets_2026/runs/${RUN_ID}" weights/curated_datasets_2026/latest
  log "promoted curated dataset symlink to weights/curated_datasets_2026/runs/${RUN_ID}"
}

external_expansion() {
  local out="weights/external_datasets_2026/runs/${RUN_ID}"
  local requirement_args=()
  if [[ "$ENFORCE_DATASET_MINIMA" == "1" || "$ENFORCE_DATASET_MINIMA" == "true" ]]; then
    requirement_args+=(--enforce-requirements)
  fi
  mkdir -p "$out"
  log "build external dataset expansion"
  "$PYTHON_BIN" -m omnicoder.data_factory.dataset_expansion_2026 \
    --profile "$PROFILE" \
    --out-dir "$out" \
    --download \
    --max-records-per-dataset "$MAX_RECORDS_PER_DATASET" \
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
  ln -sfn "$ROOT/$out" weights/external_datasets_2026/latest
  log "promoted external dataset symlink to $out"
}

agentic_tool_training() {
  local out="weights/agentic_tool_training_2026"
  local run_out="weights/agentic_tool_training_2026/runs/${RUN_ID}"
  local source="${OMNICODER_TRACE_WORK_DIR}/jsonl/contamination_scanned.jsonl"
  mkdir -p "$out" "$run_out"
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
from pathlib import Path
p = Path("weights/agentic_tool_training_2026/agentic_tool_training_manifest.json")
data = json.loads(p.read_text())
counts = data.get("counts", {})
required = ["sft", "reward", "preference", "rlvr", "tool_rlvr"]
missing = {name: int(counts.get(name) or 0) for name in required if int(counts.get(name) or 0) <= 0}
if missing:
    raise SystemExit(f"agentic tool training produced empty required exports: {missing}")
PY
  cp "$out"/tool_*.jsonl "$run_out"/ 2>/dev/null || true
  cp "$out"/agentic_tool_training_manifest.json "$run_out"/ 2>/dev/null || true
  ln -sfn "$ROOT/$run_out" weights/agentic_tool_training_2026/latest_run
  log "agentic tool training exports refreshed in $out"
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

teacher_jobs() {
  local job_dir="${TEACHER_JOB_ROOT}/${RUN_ID}"
  mkdir -p "$job_dir"
  log "build agentic/math/code/tool teacher jobs"
  build_jobs_if_present weights/curated_datasets_2026/latest/jsonl/train_agentic_focus.jsonl agentic_math_code_tool_critique "$job_dir/agentic_jobs.jsonl"
  build_jobs_if_present weights/curated_datasets_2026/latest/jsonl/train_code.jsonl code_repair_reasoning_critique "$job_dir/code_jobs.jsonl"
  build_jobs_if_present weights/curated_datasets_2026/latest/jsonl/train_tool.jsonl tool_call_replay_reward_critique "$job_dir/tool_jobs.jsonl"
  build_jobs_if_present weights/external_datasets_2026/latest/jsonl/math_reasoning.jsonl math_rlvr_answer_critique "$job_dir/math_jobs.jsonl"
  build_jobs_if_present weights/external_datasets_2026/latest/jsonl/coding_agentic.jsonl coding_agent_trajectory_critique "$job_dir/external_code_jobs.jsonl"
  build_jobs_if_present weights/external_datasets_2026/latest/jsonl/agentic_tool_reasoning.jsonl agentic_tool_reasoning_critique "$job_dir/external_tool_jobs.jsonl"
  build_jobs_if_present weights/external_datasets_2026/latest/jsonl/terminal_browser_agents.jsonl terminal_browser_agent_critique "$job_dir/external_terminal_jobs.jsonl"
  build_jobs_if_present weights/external_datasets_2026/latest/jsonl/research_internal_all_external.jsonl research_internal_distillation_review "$job_dir/research_internal_jobs.jsonl"
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
  ln -sfn "$ROOT/$job_dir" "${TEACHER_JOB_ROOT}/latest"
  wc -l "$job_dir"/*.jsonl
  log "teacher job dir: $job_dir"
}

p40_teacher_rollouts() {
  local job_dir="${TEACHER_JOB_ROOT}/latest"
  local out_dir="weights/data_factory/teacher_rollouts/${RUN_ID}"
  mkdir -p "$out_dir/logs"
  log "launch P40 teacher rollouts"
  local pids=()
  if [[ -s "$job_dir/shard_gpu1.jsonl" ]]; then
    "$PYTHON_BIN" -m omnicoder.data_factory.openai_teacher_rollout_2026 \
      --input "$job_dir/shard_gpu1.jsonl" \
      --out "$out_dir/qwen36_gpu1.jsonl" \
      --base-url http://127.0.0.1:18084/v1 \
      --model "$TEACHER_MODEL" \
      --limit "$TEACHER_LIMIT" --max-tokens 1024 --temperature 0.2 --timeout 180 --sleep 2 \
      --record-kind qwen36_p40_agentic_math_code_tool \
      --thermal-gpu-index 1 --max-gpu-temp "$MAX_GPU_TEMP" \
      > "$out_dir/logs/gpu1.log" 2>&1 &
    pids+=("$!")
  fi
  if [[ -s "$job_dir/shard_gpu2.jsonl" ]]; then
    "$PYTHON_BIN" -m omnicoder.data_factory.openai_teacher_rollout_2026 \
      --input "$job_dir/shard_gpu2.jsonl" \
      --out "$out_dir/qwen36_gpu2.jsonl" \
      --base-url http://127.0.0.1:18082/v1 \
      --model "$TEACHER_MODEL" \
      --limit "$TEACHER_LIMIT" --max-tokens 1024 --temperature 0.2 --timeout 180 --sleep 2 \
      --record-kind qwen36_p40_agentic_math_code_tool \
      --thermal-gpu-index 2 --max-gpu-temp "$MAX_GPU_TEMP" \
      > "$out_dir/logs/gpu2.log" 2>&1 &
    pids+=("$!")
  fi
  if [[ -s "$job_dir/shard_gpu3.jsonl" ]]; then
    "$PYTHON_BIN" -m omnicoder.data_factory.openai_teacher_rollout_2026 \
      --input "$job_dir/shard_gpu3.jsonl" \
      --out "$out_dir/qwen36_gpu3.jsonl" \
      --base-url http://127.0.0.1:18085/v1 \
      --model "$TEACHER_MODEL" \
      --limit "$TEACHER_LIMIT" --max-tokens 1024 --temperature 0.2 --timeout 180 --sleep 2 \
      --record-kind qwen36_p40_agentic_math_code_tool \
      --thermal-gpu-index 3 --max-gpu-temp "$MAX_GPU_TEMP" \
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
    --out-dir weights/agentic_tool_training_2026 \
    --limit 0 > "weights/agentic_tool_training_2026/runs/${RUN_ID}/agentic_tool_training_after_teacher.stdout.json"
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
p = Path("weights/agentic_tool_training_2026/agentic_tool_training_manifest.json")
data = json.loads(p.read_text())
counts = data.get("counts", {})
required = ["sft", "reward", "preference", "rlvr", "tool_rlvr"]
missing = {name: int(counts.get(name) or 0) for name in required if int(counts.get(name) or 0) <= 0}
if missing:
    raise SystemExit(f"teacher-refreshed agentic exports are empty: {missing}")
PY
  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
out = Path("$out_dir")
counts = {}
for path in sorted(out.glob("qwen36*.jsonl")):
    counts[path.name] = sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())
(out / "teacher_rollout_manifest.json").write_text(json.dumps({"status": "ok", "run_id": "$RUN_ID", "failures": $failures, "counts": counts}, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
  ln -sfn "$ROOT/$out_dir" weights/data_factory/teacher_rollouts/latest
  if [[ "$failures" -gt 0 ]]; then
    log "teacher rollout dir: $out_dir with $failures failed worker(s); combined nonempty output promoted"
  else
    log "teacher rollout dir: $out_dir"
  fi
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
}

case "$ACTION" in
  preflight) preflight ;;
  collect-curate) collect_curate ;;
  external-expansion) external_expansion ;;
  agentic-tool-training) agentic_tool_training ;;
  teacher-jobs) teacher_jobs ;;
  p40-teacher) p40_teacher_rollouts ;;
  status) status ;;
  all)
    preflight
    collect_curate
    agentic_tool_training
    external_expansion
    teacher_jobs
    p40_teacher_rollouts
    status
    ;;
  *)
    echo "unknown action: $ACTION" >&2
    exit 2
    ;;
esac
