#!/usr/bin/env bash
set -euo pipefail

WEIGHTS_ROOT="${OMNICODER_WEIGHTS_ROOT:-/home/cereal/omnicoder_2026_work/weights}"
REPO="${OMNICODER_REPO:-$WEIGHTS_ROOT/staged_patches/omnicoder_d28a1d4_allmodalfix_20260526T080000Z}"
RUN_TAG="${OMNICODER_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
RECOVERY_RUN="${OMNICODER_RECOVERY_RUN:-$WEIGHTS_ROOT/training_orchestration_2026/posttrain_recovergrpo_d28a1d4_20260526T051052Z}"
BALANCED_RUN="${OMNICODER_BALANCED_RUN:-$WEIGHTS_ROOT/training_orchestration_2026/balanced_allmodal_posttrain_20260526T082100Z}"
CONTAINER_NAME="${OMNICODER_CONTAINER_NAME:-omnicoder_pipeline_posttrain_balanced_allmodal_${RUN_TAG}}"
OUT_DIR="${OMNICODER_OUT_DIR:-weights/training_orchestration_2026/posttrain_balanced_allmodal_${RUN_TAG}}"
ACTIVE_CONTAINER_GLOB="${OMNICODER_ACTIVE_CONTAINER_GLOB:-omnicoder_pipeline_posttrain_recovergrpo}"
MIN_FREE_GB="${OMNICODER_MIN_FREE_GB:-60}"

if [[ "${OMNICODER_ALLOW_ACTIVE_TRAINING:-0}" != "1" ]]; then
  if docker ps --format '{{.Names}}' | grep -q "$ACTIVE_CONTAINER_GLOB"; then
    echo "Refusing to launch: an active recovery training container matches $ACTIVE_CONTAINER_GLOB." >&2
    echo "Set OMNICODER_ALLOW_ACTIVE_TRAINING=1 only if you intentionally want concurrent fast-GPU training." >&2
    exit 3
  fi
fi

free_gb="$(df -BG "$WEIGHTS_ROOT" | awk 'NR==2 {gsub(/G/, "", $4); print $4}')"
if [[ -n "$free_gb" && "$free_gb" -lt "$MIN_FREE_GB" ]]; then
  echo "Refusing to launch: only ${free_gb}G free under $WEIGHTS_ROOT; need at least ${MIN_FREE_GB}G." >&2
  exit 4
fi

find_latest_checkpoint() {
  python3 - "$RECOVERY_RUN" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1]) / "checkpoints" / "posttrain"
candidates = []
for path in sorted(run.glob("01_grpo_rlvr_replay_pipeline*")):
    if not path.is_dir():
        continue
    complete_path = path / ".complete.json"
    manifest_path = path / "manifest.json"
    if not complete_path.exists() or not manifest_path.exists():
        continue
    try:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    world_size = int(manifest.get("world_size") or complete.get("world_size") or 0)
    if world_size <= 0:
        continue
    if len(list(path.glob("rank*.pt"))) != world_size:
        continue
    if len(list(path.glob("rank*.pt.complete.json"))) != world_size:
        continue
    step = int(complete.get("global_step") or manifest.get("global_step") or -1)
    candidates.append((step, path))
if not candidates:
    raise SystemExit("no complete checkpoint found")
print(max(candidates, key=lambda item: item[0])[1])
PY
}

RESUME_CHECKPOINT="${OMNICODER_RESUME_CHECKPOINT:-$(find_latest_checkpoint)}"
SFT_JSONL="${OMNICODER_BALANCED_SFT_JSONL:-weights/training_orchestration_2026/balanced_allmodal_posttrain_20260526T082100Z/balanced_allmodal_sft.jsonl}"
RLVR_JSONL="${OMNICODER_BALANCED_RLVR_JSONL:-weights/training_orchestration_2026/balanced_allmodal_posttrain_20260526T082100Z/balanced_allmodal_rlvr.jsonl}"
REWARD_JSONL="${OMNICODER_BALANCED_REWARD_JSONL:-weights/training_orchestration_2026/balanced_allmodal_posttrain_20260526T082100Z/balanced_allmodal_reward.jsonl}"
MANIFEST="${OMNICODER_BALANCED_MANIFEST:-weights/training_orchestration_2026/balanced_allmodal_posttrain_20260526T082100Z/balanced_allmodal_manifest.json}"

for required in "$REPO" "$BALANCED_RUN" "$WEIGHTS_ROOT/${SFT_JSONL#weights/}" "$WEIGHTS_ROOT/${RLVR_JSONL#weights/}" "$WEIGHTS_ROOT/${REWARD_JSONL#weights/}"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 5
  fi
done

export OMNICODER_REPO="$REPO"
export OMNICODER_WEIGHTS_ROOT="$WEIGHTS_ROOT"
export OMNICODER_CONTAINER_NAME="$CONTAINER_NAME"
export OMNICODER_OUT_DIR="$OUT_DIR"
export OMNICODER_MODE=run-posttraining
export OMNICODER_RESUME_CHECKPOINT="$RESUME_CHECKPOINT"
export OMNICODER_CURATION_MANIFEST="$MANIFEST"
export OMNICODER_POSTTRAIN_ALGORITHM_ORDER="${OMNICODER_POSTTRAIN_ALGORITHM_ORDER:-reward_weighted_sft_replay}"
export OMNICODER_POSTTRAIN_INPUT_JSONL="${OMNICODER_POSTTRAIN_INPUT_JSONL:-reward_weighted_sft_replay=$SFT_JSONL,grpo_rlvr_replay=$RLVR_JSONL,process_reward_replay=$REWARD_JSONL}"
export OMNICODER_POSTTRAIN_EXPLICIT_INPUTS_ONLY="${OMNICODER_POSTTRAIN_EXPLICIT_INPUTS_ONLY:-1}"
export OMNICODER_POSTTRAIN_STEPS="${OMNICODER_POSTTRAIN_STEPS:-32}"
export OMNICODER_SAVE_INTERVAL="${OMNICODER_SAVE_INTERVAL:-0}"
export OMNICODER_SEQ_LEN="${OMNICODER_SEQ_LEN:-1024}"
export OMNICODER_BATCH_SIZE="${OMNICODER_BATCH_SIZE:-1}"
export OMNICODER_POSTTRAIN_LR="${OMNICODER_POSTTRAIN_LR:-0.000001}"
export OMNICODER_HELDOUT_MAX_RECORDS_PER_FILE="${OMNICODER_HELDOUT_MAX_RECORDS_PER_FILE:-16}"
export OMNICODER_BENCHMARK_MAX_RECORDS_PER_FILE="${OMNICODER_BENCHMARK_MAX_RECORDS_PER_FILE:-16}"
export OMNICODER_HELDOUT_SAMPLE_LOSS_TIMEOUT_SECONDS="${OMNICODER_HELDOUT_SAMPLE_LOSS_TIMEOUT_SECONDS:-3600}"
export OMNICODER_BENCHMARK_SAMPLE_LOSS_TIMEOUT_SECONDS="${OMNICODER_BENCHMARK_SAMPLE_LOSS_TIMEOUT_SECONDS:-3600}"
export OMNICODER_REPORTABLE_TASK_ROOTS="${OMNICODER_REPORTABLE_TASK_ROOTS:-/home/cereal/omnicoder_2026_work/weights/official_benchmarks_2026/runs/bench_reportable_fix_eaa2463_20260525T181734Z/local_2026}"
export OMNICODER_ALLOW_LOCAL_BENCHMARK_TASK_ROOTS="${OMNICODER_ALLOW_LOCAL_BENCHMARK_TASK_ROOTS:-1}"
export OMNICODER_DETACH="${OMNICODER_DETACH:-1}"

exec "$REPO/scripts/ai_server_fast_pipeline_20b.sh"
