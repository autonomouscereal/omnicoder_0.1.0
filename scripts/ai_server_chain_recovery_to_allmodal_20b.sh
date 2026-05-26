#!/usr/bin/env bash
set -euo pipefail

# Wait for the active 20B recovery run to finish, verify a complete sharded
# checkpoint, then launch the patched balanced all-modal posttraining stage.
# This script never stops or restarts the recovery container.

WEIGHTS_ROOT="${OMNICODER_WEIGHTS_ROOT:-/home/cereal/omnicoder_2026_work/weights}"
REPO="${OMNICODER_REPO:-$WEIGHTS_ROOT/staged_patches/omnicoder_d28a1d4_allmodalfix_20260526T080000Z}"
RECOVERY_CONTAINER="${OMNICODER_RECOVERY_CONTAINER:-omnicoder_pipeline_posttrain_recovergrpo_d28a1d4_20260526T051052Z}"
RECOVERY_RUN="${OMNICODER_RECOVERY_RUN:-$WEIGHTS_ROOT/training_orchestration_2026/posttrain_recovergrpo_d28a1d4_20260526T051052Z}"
POLL_SECONDS="${OMNICODER_CHAIN_POLL_SECONDS:-60}"
MAX_WAIT_SECONDS="${OMNICODER_CHAIN_MAX_WAIT_SECONDS:-0}"
MIN_RESUME_STEP="${OMNICODER_CHAIN_MIN_RESUME_STEP:-432}"
CLEANUP_OLDER="${OMNICODER_CHAIN_CLEANUP_OLDER_CHECKPOINTS:-1}"
RUN_TAG="${OMNICODER_CHAIN_RUN_TAG:-balanced_allmodal_sft_$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${OMNICODER_CHAIN_LOG_DIR:-$WEIGHTS_ROOT/training_orchestration_2026/chain_logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${OMNICODER_CHAIN_LOG_FILE:-$LOG_DIR/chain_recovery_to_allmodal_${RUN_TAG}.log}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_FILE"
}

container_running() {
  docker inspect -f '{{.State.Running}}' "$RECOVERY_CONTAINER" 2>/dev/null | grep -qx true
}

container_exists() {
  docker inspect "$RECOVERY_CONTAINER" >/dev/null 2>&1
}

latest_complete_checkpoint() {
  python3 - "$RECOVERY_RUN" "$MIN_RESUME_STEP" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1]) / "checkpoints" / "posttrain"
min_step = int(sys.argv[2])
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
    rank_files = sorted(path.glob("rank*.pt"))
    markers = sorted(path.glob("rank*.pt.complete.json"))
    if len(rank_files) != world_size or len(markers) != world_size:
        continue
    step = int(complete.get("global_step") or manifest.get("global_step") or -1)
    if step < min_step:
        continue
    loss = complete.get("last_loss", manifest.get("last_loss"))
    candidates.append((step, str(loss), path))
if not candidates:
    raise SystemExit(2)
step, loss, path = max(candidates, key=lambda item: item[0])
print(f"{step}\t{loss}\t{path}")
PY
}

cleanup_older_checkpoints() {
  local keep="$1"
  if [[ "$CLEANUP_OLDER" != "1" ]]; then
    return 0
  fi
  local old="$RECOVERY_RUN/checkpoints/posttrain/01_grpo_rlvr_replay_pipeline.step432"
  if [[ -d "$old" && "$old" != "$keep" ]]; then
    log "Removing older duplicate safe checkpoint after newer checkpoint was verified: $old"
    rm -rf --one-file-system "$old"
  fi
}

launch_allmodal() {
  local resume_checkpoint="$1"
  cd "$REPO"
  export OMNICODER_REPO="$REPO"
  export OMNICODER_WEIGHTS_ROOT="$WEIGHTS_ROOT"
  export OMNICODER_RECOVERY_RUN="$RECOVERY_RUN"
  export OMNICODER_RESUME_CHECKPOINT="$resume_checkpoint"
  export OMNICODER_RUN_TAG="$RUN_TAG"
  export OMNICODER_CONTAINER_NAME="${OMNICODER_CONTAINER_NAME:-omnicoder_pipeline_posttrain_balanced_allmodal_${RUN_TAG}}"
  export OMNICODER_OUT_DIR="${OMNICODER_OUT_DIR:-weights/training_orchestration_2026/posttrain_balanced_allmodal_${RUN_TAG}}"
  export OMNICODER_POSTTRAIN_ALGORITHM_ORDER="${OMNICODER_POSTTRAIN_ALGORITHM_ORDER:-reward_weighted_sft_replay}"
  export OMNICODER_POSTTRAIN_STEPS="${OMNICODER_POSTTRAIN_STEPS:-32}"
  export OMNICODER_SAVE_INTERVAL="${OMNICODER_SAVE_INTERVAL:-0}"
  export OMNICODER_SEQ_LEN="${OMNICODER_SEQ_LEN:-1024}"
  export OMNICODER_BATCH_SIZE="${OMNICODER_BATCH_SIZE:-1}"
  export OMNICODER_DETACH="${OMNICODER_DETACH:-1}"
  export OMNICODER_MIN_FREE_GB="${OMNICODER_MIN_FREE_GB:-60}"
  log "Launching balanced all-modal posttraining from $resume_checkpoint"
  bash scripts/ai_server_launch_balanced_allmodal_posttrain_20b.sh 2>&1 | tee -a "$LOG_FILE"
}

main() {
  log "Chain runner started"
  log "recovery_container=$RECOVERY_CONTAINER"
  log "recovery_run=$RECOVERY_RUN"
  log "repo=$REPO"
  local started
  started="$(date +%s)"
  if ! container_exists; then
    log "Recovery container does not exist; selecting latest complete checkpoint immediately."
  fi
  while container_running; do
    if [[ "$MAX_WAIT_SECONDS" != "0" ]]; then
      local now elapsed
      now="$(date +%s)"
      elapsed=$((now - started))
      if [[ "$elapsed" -ge "$MAX_WAIT_SECONDS" ]]; then
        log "Timed out waiting for recovery container after ${elapsed}s"
        exit 10
      fi
    fi
    local status_line=""
    if [[ -f "$RECOVERY_RUN/logs/posttrain_01_grpo_rlvr_replay_pipeline_reward_replay.jsonl" ]]; then
      status_line="$(grep -h '"step"' "$RECOVERY_RUN/logs/posttrain_01_grpo_rlvr_replay_pipeline_reward_replay.jsonl" | tail -n 1 || true)"
    fi
    log "Recovery still running; latest_loss_line=${status_line:-none}"
    sleep "$POLL_SECONDS"
  done

  log "Recovery container is not running; verifying latest complete checkpoint."
  local found step loss ckpt
  found="$(latest_complete_checkpoint)"
  step="$(printf '%s' "$found" | cut -f1)"
  loss="$(printf '%s' "$found" | cut -f2)"
  ckpt="$(printf '%s' "$found" | cut -f3-)"
  log "Latest complete checkpoint: step=$step loss=$loss path=$ckpt"
  cleanup_older_checkpoints "$ckpt"
  launch_allmodal "$ckpt"
  log "Chain runner done"
}

main "$@"
