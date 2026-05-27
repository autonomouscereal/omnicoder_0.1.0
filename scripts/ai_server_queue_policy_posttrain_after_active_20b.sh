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
EXTRA_BALANCED_SOURCES="${OMNICODER_EXTRA_BALANCED_SOURCES:-}"
EXTRA_BALANCED_SOURCE_FLOORS="${OMNICODER_EXTRA_BALANCED_SOURCE_FLOORS:-}"
BALANCED_TEXT_CAP="${OMNICODER_POLICY_BALANCED_TEXT_CAP:-8192}"
BALANCED_CODE_CAP="${OMNICODER_POLICY_BALANCED_CODE_CAP:-8192}"
BALANCED_TOOL_CAP="${OMNICODER_POLICY_BALANCED_TOOL_CAP:-8192}"
BALANCED_MATH_CAP="${OMNICODER_POLICY_BALANCED_MATH_CAP:-8192}"
BALANCED_LONG_CONTEXT_CAP="${OMNICODER_POLICY_BALANCED_LONG_CONTEXT_CAP:-4096}"
BALANCED_IMAGE_CAP="${OMNICODER_POLICY_BALANCED_IMAGE_CAP:-2048}"
BALANCED_VIDEO_CAP="${OMNICODER_POLICY_BALANCED_VIDEO_CAP:-2048}"
BALANCED_AUDIO_CAP="${OMNICODER_POLICY_BALANCED_AUDIO_CAP:-2048}"
BALANCED_MUSIC_CAP="${OMNICODER_POLICY_BALANCED_MUSIC_CAP:-2048}"
BALANCED_AGENTIC_SOURCE_FLOOR="${OMNICODER_BALANCED_AGENTIC_SOURCE_FLOOR:-512}"
BALANCED_BASE_LONG_CONTEXT_SOURCE_FLOOR="${OMNICODER_BALANCED_BASE_LONG_CONTEXT_SOURCE_FLOOR:-$MIN_RECORDS_PER_REQUIRED}"
BALANCED_QWEN_TEXT_SOURCE_FLOOR="${OMNICODER_BALANCED_QWEN_TEXT_SOURCE_FLOOR:-16}"
BALANCED_QWEN_LONG_CONTEXT_SOURCE_FLOOR="${OMNICODER_BALANCED_QWEN_LONG_CONTEXT_SOURCE_FLOOR:-16}"
MEDIA_TEACHER_SOURCE_FLOOR_SCALE="${OMNICODER_MEDIA_TEACHER_SOURCE_FLOOR_SCALE:-1}"
MEDIA_TEACHER_IMAGE_SOURCE_FLOOR_SCALE="${OMNICODER_MEDIA_TEACHER_IMAGE_SOURCE_FLOOR_SCALE:-$MEDIA_TEACHER_SOURCE_FLOOR_SCALE}"
MEDIA_TEACHER_VIDEO_SOURCE_FLOOR_SCALE="${OMNICODER_MEDIA_TEACHER_VIDEO_SOURCE_FLOOR_SCALE:-$MEDIA_TEACHER_SOURCE_FLOOR_SCALE}"
MEDIA_TEACHER_AUDIO_SOURCE_FLOOR_SCALE="${OMNICODER_MEDIA_TEACHER_AUDIO_SOURCE_FLOOR_SCALE:-$MEDIA_TEACHER_SOURCE_FLOOR_SCALE}"
MEDIA_TEACHER_MUSIC_SOURCE_FLOOR_SCALE="${OMNICODER_MEDIA_TEACHER_MUSIC_SOURCE_FLOOR_SCALE:-$MEDIA_TEACHER_SOURCE_FLOOR_SCALE}"
REQUIRE_AGENTIC_BALANCED_SOURCE="${OMNICODER_REQUIRE_AGENTIC_BALANCED_SOURCE:-1}"
REQUIRE_QWEN_TEXT_LONG_CONTEXT_BALANCED_SOURCES="${OMNICODER_REQUIRE_QWEN_TEXT_LONG_CONTEXT_BALANCED_SOURCES:-1}"
REQUIRE_MEDIA_TEACHER_BALANCED_FLOORS="${OMNICODER_REQUIRE_MEDIA_TEACHER_BALANCED_FLOORS:-1}"

mkdir -p "$QUEUE_DIR"
echo $$ > "$QUEUE_DIR/pid"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
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
  add_extra_sources() {
    local raw="$1"
    [[ -n "$raw" ]] || return 0
    local old_ifs="$IFS"
    IFS=$',\n'
    local item
    for item in $raw; do
      item="${item#"${item%%[![:space:]]*}"}"
      item="${item%"${item##*[![:space:]]}"}"
      [[ -n "$item" ]] || continue
      if [[ "$item" != *=* && "$item" != *::* ]]; then
        log "skipping malformed extra balanced source: $item"
        continue
      fi
      local modality path
      if [[ "$item" == *::* ]]; then
        modality="${item%%::*}"
        path="${item#*::}"
      else
        modality="${item%%=*}"
        path="${item#*=}"
      fi
      add_source "$modality" "$path"
    done
    IFS="$old_ifs"
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
  add_extra_sources "$EXTRA_BALANCED_SOURCES"

  scale_floor() {
    local base="$1"
    local scale="$2"
    "$PYTHON_BIN" - "$base" "$scale" <<'PY'
import math
import sys

base = max(0, int(float(sys.argv[1])))
scale = max(0.0, float(sys.argv[2]))
print(int(math.ceil(base * scale)))
PY
  }
  add_source_floor() {
    local source_name="$1"
    local floor="$2"
    if [[ "$floor" =~ ^[0-9]+$ ]] && (( floor > 0 )); then
      source_floor_args+=(--source-floor "$source_name=$floor")
    fi
  }
  local -a source_floor_args=()
  add_source_floor qwen36_tool.clean.jsonl 16
  add_source_floor qwen36_code.clean.jsonl 16
  add_source_floor qwen36_math.clean.jsonl 16
  add_source_floor qwen36_long_context.clean.jsonl "$BALANCED_QWEN_LONG_CONTEXT_SOURCE_FLOOR"
  add_source_floor qwen36_text.clean.jsonl "$BALANCED_QWEN_TEXT_SOURCE_FLOOR"
  add_source_floor agentic.clean.jsonl "$BALANCED_AGENTIC_SOURCE_FLOOR"
  add_source_floor long_context.clean.jsonl "$BALANCED_BASE_LONG_CONTEXT_SOURCE_FLOOR"
  add_source_floor qwen_image_generate.clean.jsonl "$(scale_floor "${OMNICODER_BALANCED_QWEN_IMAGE_GENERATE_SOURCE_FLOOR:-8}" "$MEDIA_TEACHER_IMAGE_SOURCE_FLOOR_SCALE")"
  add_source_floor qwen_image_edit.clean.jsonl "$(scale_floor "${OMNICODER_BALANCED_QWEN_IMAGE_EDIT_SOURCE_FLOOR:-8}" "$MEDIA_TEACHER_IMAGE_SOURCE_FLOOR_SCALE")"
  add_source_floor ltx_video.clean.jsonl "$(scale_floor "${OMNICODER_BALANCED_LTX_VIDEO_SOURCE_FLOOR:-4}" "$MEDIA_TEACHER_VIDEO_SOURCE_FLOOR_SCALE")"
  add_source_floor tts.clean.jsonl "$(scale_floor "${OMNICODER_BALANCED_TTS_SOURCE_FLOOR:-16}" "$MEDIA_TEACHER_AUDIO_SOURCE_FLOOR_SCALE")"
  add_source_floor ace_tts.clean.jsonl "$(scale_floor "${OMNICODER_BALANCED_ACE_TTS_SOURCE_FLOOR:-8}" "$MEDIA_TEACHER_AUDIO_SOURCE_FLOOR_SCALE")"
  add_source_floor music.clean.jsonl "$(scale_floor "${OMNICODER_BALANCED_MUSIC_SOURCE_FLOOR:-16}" "$MEDIA_TEACHER_MUSIC_SOURCE_FLOOR_SCALE")"
  add_source_floor musicbench.clean.jsonl "$(scale_floor "${OMNICODER_BALANCED_MUSICBENCH_SOURCE_FLOOR:-8}" "$MEDIA_TEACHER_MUSIC_SOURCE_FLOOR_SCALE")"
  add_source_floor ace_rollouts.clean.jsonl "$(scale_floor "${OMNICODER_BALANCED_ACE_ROLLOUTS_SOURCE_FLOOR:-8}" "$MEDIA_TEACHER_MUSIC_SOURCE_FLOOR_SCALE")"
  add_extra_source_floors() {
    local raw="$1"
    [[ -n "$raw" ]] || return 0
    local old_ifs="$IFS"
    IFS=$',\n'
    local item
    for item in $raw; do
      item="${item#"${item%%[![:space:]]*}"}"
      item="${item%"${item##*[![:space:]]}"}"
      [[ -n "$item" ]] || continue
      if [[ "$item" != *=* ]]; then
        log "skipping malformed extra source floor: $item"
        continue
      fi
      source_floor_args+=(--source-floor "$item")
    done
    IFS="$old_ifs"
  }
  add_extra_source_floors "$EXTRA_BALANCED_SOURCE_FLOORS"

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
    --cap "text=$BALANCED_TEXT_CAP" \
    --cap "code=$BALANCED_CODE_CAP" \
    --cap "tool=$BALANCED_TOOL_CAP" \
    --cap "math=$BALANCED_MATH_CAP" \
    --cap "long_context=$BALANCED_LONG_CONTEXT_CAP" \
    --cap "image=$BALANCED_IMAGE_CAP" \
    --cap "video=$BALANCED_VIDEO_CAP" \
    --cap "audio=$BALANCED_AUDIO_CAP" \
    --cap "music=$BALANCED_MUSIC_CAP" \
    "${source_floor_args[@]}" \
    --reject-refusal-boilerplate \
    --reject-eval-holdout \
    --min-quality-score 0.60 \
    --require-media-artifacts \
    --strip-token-ids
}

verify_balanced_source_presence() {
  local manifest="$BALANCED_ABS/balanced_allmodal_manifest.json"
  local -a required_sources=()
  if truthy "$REQUIRE_AGENTIC_BALANCED_SOURCE"; then
    required_sources+=(agentic.clean.jsonl)
  fi
  if truthy "$REQUIRE_QWEN_TEXT_LONG_CONTEXT_BALANCED_SOURCES"; then
    required_sources+=(qwen36_text.clean.jsonl qwen36_long_context.clean.jsonl)
  fi
  if (( ${#required_sources[@]} == 0 )); then
    log "balanced source presence gate disabled"
    return 0
  fi
  log "verifying protected balanced sources survived policy filters: ${required_sources[*]}"
  "$PYTHON_BIN" - "$manifest" "${required_sources[@]}" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
required = sys.argv[2:]
payload = json.loads(manifest.read_text(encoding="utf-8"))
kept_by_name: dict[str, int] = {}
for report in payload.get("source_reports", []):
    name = Path(str(report.get("path") or "")).name
    if not name:
        continue
    kept_by_name[name] = kept_by_name.get(name, 0) + int(report.get("records_kept") or 0)
missing = {name: kept_by_name.get(name, 0) for name in required if kept_by_name.get(name, 0) <= 0}
if missing:
    raise SystemExit(json.dumps({
        "status": "failed",
        "reason": "protected_balanced_sources_missing_after_filters",
        "manifest": str(manifest),
        "missing": missing,
        "kept_by_name": {name: kept_by_name.get(name, 0) for name in required},
    }, ensure_ascii=True, sort_keys=True))
print(json.dumps({
    "status": "passed",
    "manifest": str(manifest),
    "kept_by_name": {name: kept_by_name.get(name, 0) for name in required},
}, ensure_ascii=True, sort_keys=True))
PY
}

verify_balanced_media_source_floors() {
  if ! truthy "$REQUIRE_MEDIA_TEACHER_BALANCED_FLOORS"; then
    log "media teacher source-floor gate disabled"
    return 0
  fi
  local manifest="$BALANCED_ABS/balanced_allmodal_manifest.json"
  local -a required_paths=()
  if [[ -n "$QWEN_LTX_DISTILL_DIR" ]]; then
    required_paths+=(
      "$QWEN_LTX_DISTILL_DIR/jsonl/qwen_image_generate.clean.jsonl"
      "$QWEN_LTX_DISTILL_DIR/jsonl/qwen_image_edit.clean.jsonl"
      "$QWEN_LTX_DISTILL_DIR/jsonl/ltx_video.clean.jsonl"
    )
  fi
  if [[ -n "$MUSIC_TTS_ACE_DIR" ]]; then
    required_paths+=(
      "$MUSIC_TTS_ACE_DIR/jsonl/tts.clean.jsonl"
      "$MUSIC_TTS_ACE_DIR/jsonl/music.clean.jsonl"
      "$MUSIC_TTS_ACE_DIR/jsonl/musicbench.clean.jsonl"
      "$MUSIC_TTS_ACE_DIR/jsonl/ace_rollouts.clean.jsonl"
    )
    if [[ -s "$MUSIC_TTS_ACE_DIR/jsonl/ace_tts.clean.jsonl" ]]; then
      required_paths+=("$MUSIC_TTS_ACE_DIR/jsonl/ace_tts.clean.jsonl")
    fi
  fi
  if (( ${#required_paths[@]} == 0 )); then
    log "no media teacher floor paths configured"
    return 0
  fi
  log "verifying media teacher source floors survived policy filters"
  "$PYTHON_BIN" - "$manifest" "${required_paths[@]}" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
required_paths = [Path(item) for item in sys.argv[2:]]
payload = json.loads(manifest.read_text(encoding="utf-8"))
source_floors = payload.get("source_floors") if isinstance(payload.get("source_floors"), dict) else {}
source_floor_counts = payload.get("source_floor_counts") if isinstance(payload.get("source_floor_counts"), dict) else {}
kept_by_path: dict[str, int] = {}
for report in payload.get("source_reports", []):
    path = str(report.get("path") or "")
    if path:
        kept_by_path[path] = kept_by_path.get(path, 0) + int(report.get("records_kept") or 0)
missing: dict[str, dict[str, int]] = {}
for path in required_paths:
    name = path.name
    expected = int(source_floors.get(str(path)) or source_floors.get(name) or 0)
    if expected <= 0:
        continue
    kept = int(source_floor_counts.get(str(path)) or kept_by_path.get(str(path)) or 0)
    if kept < expected:
        missing[str(path)] = {"expected_floor": expected, "kept": kept}
if missing:
    raise SystemExit(json.dumps({
        "status": "failed",
        "reason": "media_teacher_source_floors_below_target_after_filters",
        "manifest": str(manifest),
        "missing": missing,
    }, ensure_ascii=True, sort_keys=True))
print(json.dumps({"status": "passed", "manifest": str(manifest), "checked": len(required_paths)}, ensure_ascii=True, sort_keys=True))
PY
}

verify_balanced_integrity_preflight() {
  local integrity_preflight_dir="$BALANCED_ABS/integrity_preflight"
  mkdir -p "$integrity_preflight_dir"
  log "running dataset-integrity preflight for balanced posttraining JSONL"
  local -a integrity_args=(
    --input "$BALANCED_ABS/balanced_allmodal_sft.jsonl" \
    --input "$BALANCED_ABS/balanced_allmodal_rlvr.jsonl" \
    --input "$BALANCED_ABS/balanced_allmodal_reward.jsonl" \
    --out-dir "$integrity_preflight_dir" \
    --manifest "$integrity_preflight_dir/dataset_integrity_manifest.json" \
    --max-artifact-bytes "${OMNICODER_QUEUE_INTEGRITY_MAX_ARTIFACT_BYTES:-4194304}"
  )
  if [[ "${OMNICODER_QUEUE_INTEGRITY_NO_ARTIFACT_SCAN:-0}" == "1" || "${OMNICODER_QUEUE_INTEGRITY_NO_ARTIFACT_SCAN:-0}" == "true" ]]; then
    integrity_args+=(--no-artifact-scan)
  fi
  "$PYTHON_BIN" -m omnicoder.data_factory.dataset_integrity_2026 "${integrity_args[@]}" \
    | tee -a "$QUEUE_DIR/dataset_integrity_preflight.log"
  "$PYTHON_BIN" - "$integrity_preflight_dir/dataset_integrity_manifest.json" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if int(payload.get("rejected") or 0) > 0:
    raise SystemExit(json.dumps({
        "status": "failed",
        "reason": "dataset_integrity_preflight_rejected_rows",
        "manifest": str(path),
        "rejected": payload.get("rejected"),
        "counts": payload.get("counts"),
    }, ensure_ascii=True, sort_keys=True))
print(json.dumps({"status": "passed", "manifest": str(path), "records": payload.get("accepted", 0)}, ensure_ascii=True, sort_keys=True))
PY
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
verify_balanced_source_presence
verify_balanced_media_source_floors
verify_balanced_integrity_preflight
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
