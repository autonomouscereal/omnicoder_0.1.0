#!/usr/bin/env bash
set -euo pipefail

# Waits for the current 20B posttraining run and the capability curation pass,
# then builds a new balanced all-modal manifest from the cleaned data and
# launches the next posttraining chunk from the newest complete checkpoint.

WEIGHTS_ROOT="${OMNICODER_WEIGHTS_ROOT:-/home/cereal/omnicoder_2026_work/weights}"
REPO="${OMNICODER_REPO:-$WEIGHTS_ROOT/staged_patches/omnicoder_d28a1d4_allmodalfix_20260526T080000Z}"
ACTIVE_CONTAINER="${OMNICODER_ACTIVE_CONTAINER:-omnicoder_pipeline_posttrain_capability_no_refusal_capability_no_refusal_step480_20260526T155702Z}"
ACTIVE_RUN_DIR="${OMNICODER_ACTIVE_RUN_DIR:-$WEIGHTS_ROOT/training_orchestration_2026/posttrain_capability_no_refusal_capability_no_refusal_step480_20260526T155702Z}"
CURATION_DIR="${OMNICODER_POLICY_CURATION_DIR:-$WEIGHTS_ROOT/data_curation_agent_2026/runs/capability_policy_full_policy_schemafix_20260526T171012Z}"
RUN_TAG_RAW="${OMNICODER_QUEUE_RUN_TAG:-policy_schemafix_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_TAG="${RUN_TAG_RAW//[^A-Za-z0-9_.-]/_}"
QUEUE_DIR="${OMNICODER_QUEUE_DIR:-$WEIGHTS_ROOT/training_orchestration_2026/queued_policy_posttrain_${RUN_TAG}}"
BALANCED_REL="weights/training_orchestration_2026/balanced_allmodal_policy_${RUN_TAG}"
BALANCED_ABS="$WEIGHTS_ROOT/${BALANCED_REL#weights/}"
POLL_SECONDS="${OMNICODER_QUEUE_POLL_SECONDS:-300}"
MAX_RECORDS_PER_MODALITY="${OMNICODER_POLICY_BALANCED_MAX_RECORDS_PER_MODALITY:-4096}"
MIN_RECORDS_PER_REQUIRED="${OMNICODER_POLICY_BALANCED_MIN_RECORDS_PER_REQUIRED:-16}"

mkdir -p "$QUEUE_DIR"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

wait_for_pid_file_exit() {
  local pid_file="$1"
  local label="$2"
  if [[ ! -s "$pid_file" ]]; then
    log "$label pid file not present yet: $pid_file"
    while [[ ! -s "$pid_file" ]]; do sleep "$POLL_SECONDS"; done
  fi
  local pid
  pid="$(cat "$pid_file")"
  while ps -p "$pid" >/dev/null 2>&1; do
    log "$label still running pid=$pid"
    sleep "$POLL_SECONDS"
  done
  log "$label exited pid=$pid"
}

wait_for_container_exit() {
  local name="$1"
  while true; do
    local state
    state="$(docker inspect -f '{{.State.Status}} oom={{.State.OOMKilled}} exit={{.State.ExitCode}}' "$name" 2>/dev/null || true)"
    if [[ -z "$state" ]]; then
      log "container missing: $name"
      return 1
    fi
    log "container_state $name $state"
    if [[ "$state" != running* ]]; then
      if [[ "$state" == *"oom=true"* || "$state" != *"exit=0"* ]]; then
        log "active training did not exit cleanly; refusing queued launch"
        return 2
      fi
      return 0
    fi
    sleep "$POLL_SECONDS"
  done
}

latest_complete_checkpoint() {
  python3 - "$ACTIVE_RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "checkpoints" / "posttrain"
candidates = []
for path in sorted(root.glob("*")):
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
    rank_files = list(path.glob("rank*.pt"))
    rank_markers = list(path.glob("rank*.pt.complete.json"))
    if len(rank_files) != world_size or len(rank_markers) != world_size:
        continue
    step = int(complete.get("global_step") or manifest.get("global_step") or -1)
    mtime = complete_path.stat().st_mtime
    candidates.append((step, mtime, path))
if not candidates:
    raise SystemExit("no complete checkpoint found")
print(max(candidates, key=lambda item: (item[0], item[1]))[2])
PY
}

build_balanced_manifest() {
  mkdir -p "$BALANCED_ABS"
  local -a sources=()
  add_source() {
    local modality="$1"
    local path="$2"
    if [[ -s "$path" ]]; then
      sources+=(--source "$modality=$path")
    fi
  }
  add_source text "$CURATION_DIR/jsonl/text.clean.jsonl"
  add_source long_context "$CURATION_DIR/jsonl/long_context.clean.jsonl"
  add_source code "$CURATION_DIR/jsonl/code.clean.jsonl"
  add_source math "$CURATION_DIR/jsonl/math.clean.jsonl"
  add_source tool "$CURATION_DIR/jsonl/tool.clean.jsonl"
  add_source tool "$CURATION_DIR/jsonl/agentic.clean.jsonl"
  add_source image "$CURATION_DIR/jsonl/image.clean.jsonl"
  add_source video "$CURATION_DIR/jsonl/video.clean.jsonl"
  add_source audio "$CURATION_DIR/jsonl/audio.clean.jsonl"
  add_source music "$CURATION_DIR/jsonl/music.clean.jsonl"
  add_source ocr "$CURATION_DIR/jsonl/ocr.clean.jsonl"

  local require="text,code,tool,image,video,audio,music,long_context,math"
  if [[ -s "$CURATION_DIR/jsonl/ocr.clean.jsonl" ]]; then
    require="$require,ocr"
  fi

  log "building balanced manifest at $BALANCED_ABS"
  python3 -m omnicoder.data_factory.balanced_allmodal_posttrain_2026 \
    --no-profile-sources \
    --out-dir "$BALANCED_ABS" \
    --manifest "$BALANCED_ABS/balanced_allmodal_manifest.json" \
    "${sources[@]}" \
    --require-modalities "$require" \
    --min-records-per-required-modality "$MIN_RECORDS_PER_REQUIRED" \
    --max-records-per-modality "$MAX_RECORDS_PER_MODALITY" \
    --reject-refusal-boilerplate \
    --reject-eval-holdout \
    --min-quality-score 0.60 \
    --require-media-artifacts \
    --strip-token-ids
}

wait_for_pid_file_exit "$CURATION_DIR/pid" "capability_curation"
if [[ ! -s "$CURATION_DIR/curation_manifest_index.json" ]]; then
  log "curation manifest index missing; refusing queued launch"
  exit 10
fi
build_balanced_manifest
wait_for_container_exit "$ACTIVE_CONTAINER"
RESUME_CHECKPOINT="$(latest_complete_checkpoint)"
log "launching next posttraining from $RESUME_CHECKPOINT"

OMNICODER_RESUME_CHECKPOINT="$RESUME_CHECKPOINT" \
OMNICODER_BALANCED_RUN="$BALANCED_ABS" \
OMNICODER_BALANCED_SFT_JSONL="$BALANCED_REL/balanced_allmodal_sft.jsonl" \
OMNICODER_BALANCED_RLVR_JSONL="$BALANCED_REL/balanced_allmodal_rlvr.jsonl" \
OMNICODER_BALANCED_REWARD_JSONL="$BALANCED_REL/balanced_allmodal_reward.jsonl" \
OMNICODER_BALANCED_MANIFEST="$BALANCED_REL/balanced_allmodal_manifest.json" \
OMNICODER_CONTAINER_NAME="omnicoder_pipeline_posttrain_policy_schemafix_${RUN_TAG}" \
OMNICODER_RUN_TAG="policy_schemafix_${RUN_TAG}" \
OMNICODER_OUT_DIR="weights/training_orchestration_2026/posttrain_policy_schemafix_${RUN_TAG}" \
OMNICODER_ACTIVE_CONTAINER_GLOB="omnicoder_pipeline_posttrain_" \
OMNICODER_POSTTRAIN_STEPS="${OMNICODER_QUEUED_POSTTRAIN_STEPS:-64}" \
OMNICODER_SEQ_LEN="${OMNICODER_QUEUED_SEQ_LEN:-1024}" \
OMNICODER_DETACH=1 \
bash "$REPO/scripts/ai_server_launch_balanced_allmodal_posttrain_20b.sh"

log "queued launch submitted"
