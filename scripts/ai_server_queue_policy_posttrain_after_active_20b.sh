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
MUSIC_TTS_ACE_DIR="${OMNICODER_MUSIC_TTS_ACE_CURATION_DIR:-}"
if [[ -z "$MUSIC_TTS_ACE_DIR" && -s "$WEIGHTS_ROOT/data_curation_agent_2026/latest_music_tts_ace_curation_dir.txt" ]]; then
  MUSIC_TTS_ACE_DIR="$(cat "$WEIGHTS_ROOT/data_curation_agent_2026/latest_music_tts_ace_curation_dir.txt")"
fi
QWEN_LTX_DISTILL_DIR="${OMNICODER_QWEN_LTX_DISTILL_DIR:-}"
if [[ -z "$QWEN_LTX_DISTILL_DIR" && -s "$WEIGHTS_ROOT/data_curation_agent_2026/current_qwen_ltx_distillation_dir.txt" ]]; then
  QWEN_LTX_DISTILL_DIR="$(cat "$WEIGHTS_ROOT/data_curation_agent_2026/current_qwen_ltx_distillation_dir.txt")"
fi
if [[ -z "$QWEN_LTX_DISTILL_DIR" && -s "$WEIGHTS_ROOT/data_curation_agent_2026/latest_qwen_ltx_distillation_dir.txt" ]]; then
  QWEN_LTX_DISTILL_DIR="$(cat "$WEIGHTS_ROOT/data_curation_agent_2026/latest_qwen_ltx_distillation_dir.txt")"
fi
RUN_QWEN_LTX_DISTILL="${OMNICODER_RUN_QWEN_LTX_DISTILL:-1}"
RUN_TAG_RAW="${OMNICODER_QUEUE_RUN_TAG:-policy_schemafix_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_TAG="${RUN_TAG_RAW//[^A-Za-z0-9_.-]/_}"
QUEUE_DIR="${OMNICODER_QUEUE_DIR:-$WEIGHTS_ROOT/training_orchestration_2026/queued_policy_posttrain_${RUN_TAG}}"
BALANCED_REL="weights/training_orchestration_2026/balanced_allmodal_policy_${RUN_TAG}"
BALANCED_ABS="$WEIGHTS_ROOT/${BALANCED_REL#weights/}"
POLL_SECONDS="${OMNICODER_QUEUE_POLL_SECONDS:-300}"
MAX_RECORDS_PER_MODALITY="${OMNICODER_POLICY_BALANCED_MAX_RECORDS_PER_MODALITY:-4096}"
MIN_RECORDS_PER_REQUIRED="${OMNICODER_POLICY_BALANCED_MIN_RECORDS_PER_REQUIRED:-16}"
QWEN_LTX_DISTILL_SCRIPT="${OMNICODER_QWEN_LTX_DISTILL_SCRIPT:-$REPO/scripts/ai_server_run_qwen_ltx_distillation_2026.sh}"
PYTHON_BIN="${OMNICODER_DATA_PYTHON:-python3}"

mkdir -p "$QUEUE_DIR"
echo $$ > "$QUEUE_DIR/pid"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

wait_for_pid_file_exit() {
  local pid_file="$1"
  local label="$2"
  if [[ ! -s "$pid_file" ]]; then
    log "$label pid file not present: $pid_file"
    return 3
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

require_music_tts_family_files() {
  local -a required=(
    "$MUSIC_TTS_ACE_DIR/jsonl/tts.clean.jsonl"
    "$MUSIC_TTS_ACE_DIR/jsonl/music.clean.jsonl"
    "$MUSIC_TTS_ACE_DIR/jsonl/musicbench.clean.jsonl"
    "$MUSIC_TTS_ACE_DIR/jsonl/ace_rollouts.clean.jsonl"
  )
  local path
  for path in "${required[@]}"; do
    if [[ ! -s "$path" ]]; then
      log "required music/TTS/ACE family missing or empty: $path"
      return 1
    fi
  done
}

refresh_qwen_ltx_distill_dir() {
  if [[ -s "$WEIGHTS_ROOT/data_curation_agent_2026/current_qwen_ltx_distillation_dir.txt" ]]; then
    local current_dir
    current_dir="$(cat "$WEIGHTS_ROOT/data_curation_agent_2026/current_qwen_ltx_distillation_dir.txt")"
    if [[ -s "$current_dir/pid" ]] && ps -p "$(cat "$current_dir/pid")" >/dev/null 2>&1; then
      QWEN_LTX_DISTILL_DIR="$current_dir"
      return 0
    fi
    if [[ -z "$QWEN_LTX_DISTILL_DIR" ]]; then
      QWEN_LTX_DISTILL_DIR="$current_dir"
      return 0
    fi
  fi
  if [[ -z "$QWEN_LTX_DISTILL_DIR" && -s "$WEIGHTS_ROOT/data_curation_agent_2026/latest_qwen_ltx_distillation_dir.txt" ]]; then
    QWEN_LTX_DISTILL_DIR="$(cat "$WEIGHTS_ROOT/data_curation_agent_2026/latest_qwen_ltx_distillation_dir.txt")"
  fi
}

qwen_ltx_manifest_status_ok() {
  refresh_qwen_ltx_distill_dir
  [[ -n "$QWEN_LTX_DISTILL_DIR" ]] || return 1
  "$PYTHON_BIN" - "$QWEN_LTX_DISTILL_DIR/qwen_ltx_distillation_manifest_index.json" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists() or p.stat().st_size <= 0:
    raise SystemExit(1)
data = json.loads(p.read_text(encoding="utf-8"))
raise SystemExit(0 if data.get("status") == "ok" else 1)
PY
}

require_qwen_ltx_family_files() {
  refresh_qwen_ltx_distill_dir
  if [[ -z "$QWEN_LTX_DISTILL_DIR" ]]; then
    log "Qwen/LTX distillation dir is empty"
    return 1
  fi
  local -a required=(
    "$QWEN_LTX_DISTILL_DIR/jsonl/qwen36_tool.clean.jsonl"
    "$QWEN_LTX_DISTILL_DIR/jsonl/qwen36_code.clean.jsonl"
    "$QWEN_LTX_DISTILL_DIR/jsonl/qwen36_math.clean.jsonl"
    "$QWEN_LTX_DISTILL_DIR/jsonl/qwen36_long_context.clean.jsonl"
    "$QWEN_LTX_DISTILL_DIR/jsonl/qwen36_text.clean.jsonl"
    "$QWEN_LTX_DISTILL_DIR/jsonl/qwen_image_generate.clean.jsonl"
    "$QWEN_LTX_DISTILL_DIR/jsonl/qwen_image_edit.clean.jsonl"
    "$QWEN_LTX_DISTILL_DIR/jsonl/ltx_video.clean.jsonl"
  )
  local path
  for path in "${required[@]}"; do
    if [[ ! -s "$path" ]]; then
      log "required Qwen/Qwen-Image/LTX family missing or empty: $path"
      return 1
    fi
  done
}

run_qwen_ltx_distillation_if_needed() {
  if [[ "$RUN_QWEN_LTX_DISTILL" != "1" && "$RUN_QWEN_LTX_DISTILL" != "true" ]]; then
    log "Qwen/LTX distillation disabled by OMNICODER_RUN_QWEN_LTX_DISTILL=$RUN_QWEN_LTX_DISTILL"
    return 0
  fi
  refresh_qwen_ltx_distill_dir
  if [[ -n "$QWEN_LTX_DISTILL_DIR" && -s "$QWEN_LTX_DISTILL_DIR/pid" ]]; then
    local pid
    pid="$(cat "$QWEN_LTX_DISTILL_DIR/pid")"
    if ps -p "$pid" >/dev/null 2>&1; then
      log "waiting for in-flight Qwen/LTX distillation pid=$pid dir=$QWEN_LTX_DISTILL_DIR"
      wait_for_pid_file_exit "$QWEN_LTX_DISTILL_DIR/pid" "qwen_ltx_distillation"
    fi
  fi
  if qwen_ltx_manifest_status_ok && require_qwen_ltx_family_files; then
    log "Qwen/LTX distillation already complete: $QWEN_LTX_DISTILL_DIR"
    return 0
  fi
  if [[ -z "$QWEN_LTX_DISTILL_DIR" ]]; then
    QWEN_LTX_DISTILL_DIR="$WEIGHTS_ROOT/data_curation_agent_2026/runs/qwen36_qwenimage_ltx23_${RUN_TAG}"
  fi
  if [[ ! -x "$QWEN_LTX_DISTILL_SCRIPT" ]]; then
    chmod +x "$QWEN_LTX_DISTILL_SCRIPT" 2>/dev/null || true
  fi
  log "running Qwen 3.6/Qwen Image/LTX 2.3 teacher distillation before queued posttraining"
  OMNICODER_QWEN_LTX_DISTILL_DIR="$QWEN_LTX_DISTILL_DIR" \
  OMNICODER_QWEN_LTX_RUN_QWEN_TEXT=1 \
  OMNICODER_QWEN_LTX_RUN_MEDIA=1 \
  OMNICODER_QWEN_STOP_MANAGED_SERVER="${OMNICODER_QWEN_STOP_MANAGED_SERVER:-0}" \
  bash "$QWEN_LTX_DISTILL_SCRIPT" 2>&1 | tee -a "$QUEUE_DIR/qwen_ltx_distillation.log"
  require_qwen_ltx_family_files
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
  if [[ -n "$QWEN_LTX_DISTILL_DIR" ]]; then
    add_source tool "$QWEN_LTX_DISTILL_DIR/jsonl/qwen36_tool.clean.jsonl"
    add_source code "$QWEN_LTX_DISTILL_DIR/jsonl/qwen36_code.clean.jsonl"
    add_source math "$QWEN_LTX_DISTILL_DIR/jsonl/qwen36_math.clean.jsonl"
    add_source long_context "$QWEN_LTX_DISTILL_DIR/jsonl/qwen36_long_context.clean.jsonl"
    add_source text "$QWEN_LTX_DISTILL_DIR/jsonl/qwen36_text.clean.jsonl"
    add_source image "$QWEN_LTX_DISTILL_DIR/jsonl/qwen_image_generate.clean.jsonl"
    add_source image "$QWEN_LTX_DISTILL_DIR/jsonl/qwen_image_edit.clean.jsonl"
    add_source video "$QWEN_LTX_DISTILL_DIR/jsonl/ltx_video.clean.jsonl"
  fi
  if [[ -n "$MUSIC_TTS_ACE_DIR" ]]; then
    add_source audio "$MUSIC_TTS_ACE_DIR/jsonl/tts.clean.jsonl"
    add_source audio "$MUSIC_TTS_ACE_DIR/jsonl/ace_tts.clean.jsonl"
    add_source music "$MUSIC_TTS_ACE_DIR/jsonl/music.clean.jsonl"
    add_source music "$MUSIC_TTS_ACE_DIR/jsonl/musicbench.clean.jsonl"
    add_source music "$MUSIC_TTS_ACE_DIR/jsonl/ace_rollouts.clean.jsonl"
  fi
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
    --cap text=2048 \
    --cap code=3072 \
    --cap tool=3072 \
    --cap math=4096 \
    --cap long_context=2048 \
    --cap image=1024 \
    --cap video=1024 \
    --cap audio=1536 \
    --cap music=1024 \
    --source-floor qwen36_tool.clean.jsonl=16 \
    --source-floor qwen36_code.clean.jsonl=16 \
    --source-floor qwen36_math.clean.jsonl=16 \
    --source-floor qwen_image_generate.clean.jsonl=8 \
    --source-floor qwen_image_edit.clean.jsonl=8 \
    --source-floor ltx_video.clean.jsonl=4 \
    --source-floor tts.clean.jsonl=16 \
    --source-floor music.clean.jsonl=16 \
    --source-floor ace_rollouts.clean.jsonl=8 \
    --reject-refusal-boilerplate \
    --reject-eval-holdout \
    --min-quality-score 0.60 \
    --require-media-artifacts \
    --strip-token-ids
}

if [[ -s "$CURATION_DIR/pid" ]]; then
  wait_for_pid_file_exit "$CURATION_DIR/pid" "capability_curation" || true
elif [[ -s "$CURATION_DIR/curation_manifest_index.json" ]]; then
  log "capability_curation manifest exists and pid file is absent; treating curation as complete"
else
  log "capability_curation pid and manifest are both missing; refusing queued launch"
  exit 10
fi
if [[ ! -s "$CURATION_DIR/curation_manifest_index.json" ]]; then
  log "curation manifest index missing; refusing queued launch"
  exit 10
fi
if [[ -n "$MUSIC_TTS_ACE_DIR" ]]; then
  if [[ -s "$MUSIC_TTS_ACE_DIR/pid" ]]; then
    wait_for_pid_file_exit "$MUSIC_TTS_ACE_DIR/pid" "music_tts_ace_curation"
  fi
  if [[ ! -s "$MUSIC_TTS_ACE_DIR/music_tts_ace_manifest_index.json" ]]; then
    log "music/TTS/ACE manifest missing for $MUSIC_TTS_ACE_DIR; refusing queued launch"
    exit 11
  fi
  require_music_tts_family_files || exit 12
fi
wait_for_container_exit "$ACTIVE_CONTAINER"
run_qwen_ltx_distillation_if_needed
build_balanced_manifest
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
