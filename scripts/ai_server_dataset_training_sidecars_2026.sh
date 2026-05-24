#!/usr/bin/env bash
set -euo pipefail

ROOT="${OMNICODER_REPO:-/home/cereal/omnicoder_2026_work}"
RUN_ID="${OMNICODER_RUN_ID:-dataset_sidecars_$(date -u +%Y%m%dT%H%M%SZ)}"
PROFILE="${OMNICODER_DATASET_PROFILE:-profiles/dataset_curation_2026.json}"
TEACHER_MODEL="${OMNICODER_TEACHER_MODEL:-qwen3.6-27b-q4}"
MAX_RECORDS_PER_DATASET="${OMNICODER_MAX_RECORDS_PER_DATASET:-1024}"
TEACHER_LIMIT="${OMNICODER_TEACHER_LIMIT:-256}"
MAX_GPU_TEMP="${OMNICODER_MAX_GPU_TEMP:-78}"
ACTION="${1:-all}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
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
  mkdir -p "$out/logs" data/raw/codex_traces_2026 data/raw/claude_traces_2026 data/raw/comfyui_outputs_2026
  log "collect Codex traces"
  python3 -m omnicoder.data_factory.memory_trace_collectors_2026 collect-codex \
    --input /home/cereal/.codex \
    --out "data/raw/codex_traces_2026/codex_ai_${RUN_ID}.jsonl" \
    --source-date 2026-05-24 --min-year 2025 --max-year 2026 --limit 0 || true
  log "collect Claude traces"
  python3 -m omnicoder.data_factory.memory_trace_collectors_2026 collect-claude \
    --input /home/cereal/.claude \
    --out "data/raw/claude_traces_2026/claude_ai_${RUN_ID}.jsonl" \
    --source-date 2026-05-24 --min-year 2025 --max-year 2026 --limit 0 \
    --source-kind auto || true
  log "collect ComfyUI media manifests"
  python3 -m omnicoder.data_factory.ingest_comfyui_outputs \
    --input /home/cereal/comfyui/output \
    --out "data/raw/comfyui_outputs_2026/comfyui_${RUN_ID}.jsonl" \
    --dataset_name comfyui_outputs_2026 \
    --namespace train --bucket multimodal_media --split train \
    --source_date 2026-05-24 --license internal --limit 0 || true
  log "run trace orchestrator"
  python3 -m omnicoder.data_factory.trace_orchestrator_2026 --profile "$PROFILE" > "$out/logs/trace_orchestrator.log" 2>&1 || true
  log "build run-scoped curated dataset"
  python3 -m omnicoder.data_factory.curated_dataset_builder_2026 \
    --profile "$PROFILE" \
    --out-dir "weights/curated_datasets_2026/runs/${RUN_ID}" \
    build | tee "$out/logs/curated_dataset_builder.json"
}

external_expansion() {
  local out="weights/external_datasets_2026/runs/${RUN_ID}"
  mkdir -p "$out"
  log "build external dataset expansion"
  python3 -m omnicoder.data_factory.dataset_expansion_2026 \
    --profile "$PROFILE" \
    --out-dir "$out" \
    --download \
    --max-records-per-dataset "$MAX_RECORDS_PER_DATASET" \
    build | tee "$out/external_dataset_manifest.stdout.json"
  ln -sfn "$ROOT/$out" weights/external_datasets_2026/latest
  log "promoted external dataset symlink to $out"
}

teacher_jobs() {
  local job_dir="weights/data_factory/trace_orchestrator_2026/teacher_jobs/${RUN_ID}"
  mkdir -p "$job_dir"
  log "build agentic/math/code/tool teacher jobs"
  python3 -m omnicoder.data_factory.teacher_jobs_2026 build \
    --records weights/curated_datasets_2026/latest/jsonl/train_agentic_focus.jsonl \
    --teacher qwen3.6_27b_q4_local \
    --job_type agentic_math_code_tool_critique \
    --limit "$TEACHER_LIMIT" \
    --out "$job_dir/agentic_jobs.jsonl"
  python3 -m omnicoder.data_factory.teacher_jobs_2026 build \
    --records weights/curated_datasets_2026/latest/jsonl/train_code.jsonl \
    --teacher qwen3.6_27b_q4_local \
    --job_type code_repair_reasoning_critique \
    --limit "$TEACHER_LIMIT" \
    --out "$job_dir/code_jobs.jsonl"
  python3 -m omnicoder.data_factory.teacher_jobs_2026 build \
    --records weights/curated_datasets_2026/latest/jsonl/train_tool.jsonl \
    --teacher qwen3.6_27b_q4_local \
    --job_type tool_call_replay_reward_critique \
    --limit "$TEACHER_LIMIT" \
    --out "$job_dir/tool_jobs.jsonl"
  if [[ -s weights/external_datasets_2026/latest/jsonl/math_reasoning.jsonl ]]; then
    python3 -m omnicoder.data_factory.teacher_jobs_2026 build \
      --records weights/external_datasets_2026/latest/jsonl/math_reasoning.jsonl \
      --teacher qwen3.6_27b_q4_local \
      --job_type math_rlvr_answer_critique \
      --limit "$TEACHER_LIMIT" \
      --out "$job_dir/math_jobs.jsonl"
  fi
  cat "$job_dir"/*_jobs.jsonl > "$job_dir/all_jobs.jsonl"
  awk 'NR % 3 == 1' "$job_dir/all_jobs.jsonl" > "$job_dir/shard_gpu1.jsonl"
  awk 'NR % 3 == 2' "$job_dir/all_jobs.jsonl" > "$job_dir/shard_gpu2.jsonl"
  awk 'NR % 3 == 0' "$job_dir/all_jobs.jsonl" > "$job_dir/shard_gpu3.jsonl"
  ln -sfn "$ROOT/$job_dir" weights/data_factory/trace_orchestrator_2026/teacher_jobs/latest
  wc -l "$job_dir"/*.jsonl
  log "teacher job dir: $job_dir"
}

p40_teacher_rollouts() {
  local job_dir="weights/data_factory/trace_orchestrator_2026/teacher_jobs/latest"
  local out_dir="weights/data_factory/teacher_rollouts/${RUN_ID}"
  mkdir -p "$out_dir/logs"
  log "launch P40 teacher rollouts"
  nohup python3 -m omnicoder.data_factory.openai_teacher_rollout_2026 \
    --input "$job_dir/shard_gpu1.jsonl" \
    --out "$out_dir/qwen36_gpu1.jsonl" \
    --base-url http://127.0.0.1:18084/v1 \
    --model "$TEACHER_MODEL" \
    --limit "$TEACHER_LIMIT" --max-tokens 1024 --temperature 0.2 --timeout 180 --sleep 2 \
    --record-kind qwen36_p40_agentic_math_code_tool \
    --thermal-gpu-index 1 --max-gpu-temp "$MAX_GPU_TEMP" \
    > "$out_dir/logs/gpu1.log" 2>&1 &
  nohup python3 -m omnicoder.data_factory.openai_teacher_rollout_2026 \
    --input "$job_dir/shard_gpu2.jsonl" \
    --out "$out_dir/qwen36_gpu2.jsonl" \
    --base-url http://127.0.0.1:18082/v1 \
    --model "$TEACHER_MODEL" \
    --limit "$TEACHER_LIMIT" --max-tokens 1024 --temperature 0.2 --timeout 180 --sleep 2 \
    --record-kind qwen36_p40_agentic_math_code_tool \
    --thermal-gpu-index 2 --max-gpu-temp "$MAX_GPU_TEMP" \
    > "$out_dir/logs/gpu2.log" 2>&1 &
  nohup python3 -m omnicoder.data_factory.openai_teacher_rollout_2026 \
    --input "$job_dir/shard_gpu3.jsonl" \
    --out "$out_dir/qwen36_gpu3.jsonl" \
    --base-url http://127.0.0.1:18085/v1 \
    --model "$TEACHER_MODEL" \
    --limit "$TEACHER_LIMIT" --max-tokens 1024 --temperature 0.2 --timeout 180 --sleep 2 \
    --record-kind qwen36_p40_agentic_math_code_tool \
    --thermal-gpu-index 3 --max-gpu-temp "$MAX_GPU_TEMP" \
    > "$out_dir/logs/gpu3.log" 2>&1 &
  ln -sfn "$ROOT/$out_dir" weights/data_factory/teacher_rollouts/latest
  log "teacher rollout dir: $out_dir"
}

status() {
  preflight
  log "latest external manifest"
  if [[ -f weights/external_datasets_2026/latest/manifests/external_dataset_manifest.json ]]; then
    python3 - <<'PY'
import json
from pathlib import Path
p = Path("weights/external_datasets_2026/latest/manifests/external_dataset_manifest.json")
print(json.dumps(json.loads(p.read_text()).get("records", {}), indent=2))
PY
  fi
  log "teacher rollout counts"
  find weights/data_factory/teacher_rollouts/latest -maxdepth 1 -name '*.jsonl' -print -exec wc -l {} \; 2>/dev/null || true
}

case "$ACTION" in
  preflight) preflight ;;
  collect-curate) collect_curate ;;
  external-expansion) external_expansion ;;
  teacher-jobs) teacher_jobs ;;
  p40-teacher) p40_teacher_rollouts ;;
  status) status ;;
  all)
    preflight
    collect_curate
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
