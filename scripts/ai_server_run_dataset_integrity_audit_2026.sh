#!/usr/bin/env bash
set -euo pipefail

# Read-only integrity audit for trainable Omnicoder JSONL sources.
# It does not rewrite source datasets or touch active optimizer containers.

WEIGHTS_ROOT="${OMNICODER_WEIGHTS_ROOT:-/home/cereal/omnicoder_2026_work/weights}"
REPO="${OMNICODER_REPO:-$WEIGHTS_ROOT/staged_patches/omnicoder_d28a1d4_allmodalfix_20260526T080000Z}"
PYTHON_BIN="${OMNICODER_DATA_PYTHON:-python3}"
RUN_TAG_RAW="${OMNICODER_INTEGRITY_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_TAG="${RUN_TAG_RAW//[^A-Za-z0-9_.-]/_}"
OUT_ROOT="${OMNICODER_INTEGRITY_OUT_ROOT:-$WEIGHTS_ROOT/data_curation_agent_2026/integrity_audits/dataset_integrity_${RUN_TAG}}"

CAPABILITY_DIR="${OMNICODER_POLICY_CURATION_DIR:-$WEIGHTS_ROOT/data_curation_agent_2026/runs/capability_policy_full_policy_schemafix_20260526T171012Z}"
ACTIVE_RUN_DIR="${OMNICODER_ACTIVE_RUN_DIR:-$WEIGHTS_ROOT/training_orchestration_2026/posttrain_capability_no_refusal_capability_no_refusal_step480_20260526T155702Z}"
QUEUE_DIR="${OMNICODER_QUEUE_DIR:-$WEIGHTS_ROOT/training_orchestration_2026/queued_policy_posttrain_policy_schemafix_qwen4_ltx_strict_20260526T232538Z}"
ACTIVE_BALANCED_DIR="${OMNICODER_ACTIVE_BALANCED_DIR:-$WEIGHTS_ROOT/training_orchestration_2026/balanced_allmodal_capability_no_refusal_metafilter_musicfallback_20260526T153925Z}"
POLICY_BALANCED_DIR="${OMNICODER_POLICY_BALANCED_DIR:-$WEIGHTS_ROOT/training_orchestration_2026/balanced_allmodal_policy_policy_schemafix_musictts_expanded_20260526T202058Z}"
QWEN_LTX_DISTILL_DIR="${OMNICODER_QWEN_LTX_DISTILL_DIR:-}"
MUSIC_TTS_ACE_DIR="${OMNICODER_MUSIC_TTS_ACE_CURATION_DIR:-}"
MAX_RECORDS="${OMNICODER_INTEGRITY_MAX_RECORDS:-0}"
MAX_RECORDS_PER_INPUT="${OMNICODER_INTEGRITY_MAX_RECORDS_PER_INPUT:-0}"
MAX_ARTIFACT_BYTES="${OMNICODER_INTEGRITY_MAX_ARTIFACT_BYTES:-67108864}"
NO_ARTIFACT_SCAN="${OMNICODER_INTEGRITY_NO_ARTIFACT_SCAN:-0}"
WRITE_ACCEPTED="${OMNICODER_INTEGRITY_WRITE_ACCEPTED:-0}"
INCLUDE_HISTORICAL_BALANCED="${OMNICODER_INTEGRITY_INCLUDE_HISTORICAL_BALANCED:-0}"

if [[ -z "$QWEN_LTX_DISTILL_DIR" && -s "$WEIGHTS_ROOT/data_curation_agent_2026/current_qwen_ltx_distillation_dir.txt" ]]; then
  QWEN_LTX_DISTILL_DIR="$(cat "$WEIGHTS_ROOT/data_curation_agent_2026/current_qwen_ltx_distillation_dir.txt")"
fi
if [[ -z "$QWEN_LTX_DISTILL_DIR" && -s "$WEIGHTS_ROOT/data_curation_agent_2026/latest_qwen_ltx_distillation_dir.txt" ]]; then
  QWEN_LTX_DISTILL_DIR="$(cat "$WEIGHTS_ROOT/data_curation_agent_2026/latest_qwen_ltx_distillation_dir.txt")"
fi
if [[ -z "$MUSIC_TTS_ACE_DIR" && -s "$WEIGHTS_ROOT/data_curation_agent_2026/latest_music_tts_ace_curation_dir.txt" ]]; then
  MUSIC_TTS_ACE_DIR="$(cat "$WEIGHTS_ROOT/data_curation_agent_2026/latest_music_tts_ace_curation_dir.txt")"
fi

mkdir -p "$OUT_ROOT/logs"
printf '%s\n' "$OUT_ROOT" > "$WEIGHTS_ROOT/data_curation_agent_2026/latest_dataset_integrity_audit_dir.txt"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
shopt -s nullglob globstar

declare -A SEEN_INPUTS=()
INPUTS=()

add_input() {
  local path="$1"
  if [[ -s "$path" && -z "${SEEN_INPUTS[$path]:-}" ]]; then
    SEEN_INPUTS[$path]=1
    INPUTS+=(--input "$path")
  fi
}

add_glob() {
  local pattern="$1"
  local path
  for path in $pattern; do
    add_input "$path"
  done
}

add_balanced_dir() {
  local dir="$1"
  if [[ -d "$dir" ]]; then
    add_input "$dir/balanced_allmodal_sft.jsonl"
    add_input "$dir/balanced_allmodal_rlvr.jsonl"
    add_input "$dir/balanced_allmodal_reward.jsonl"
  fi
}

add_glob "$CAPABILITY_DIR/jsonl/*.clean.jsonl"

if [[ -n "$QWEN_LTX_DISTILL_DIR" ]]; then
  add_glob "$QWEN_LTX_DISTILL_DIR/jsonl/*.clean.jsonl"
fi

if [[ -n "$MUSIC_TTS_ACE_DIR" ]]; then
  add_glob "$MUSIC_TTS_ACE_DIR/jsonl/*.clean.jsonl"
fi

add_balanced_dir "$ACTIVE_BALANCED_DIR"
add_balanced_dir "$POLICY_BALANCED_DIR"
if [[ "$INCLUDE_HISTORICAL_BALANCED" == "1" || "$INCLUDE_HISTORICAL_BALANCED" == "true" ]]; then
  add_glob "$WEIGHTS_ROOT/training_orchestration_2026/balanced_allmodal_*/*.jsonl"
  add_glob "$WEIGHTS_ROOT/training_orchestration_2026/balanced_allmodal_policy_*/*.jsonl"
fi
add_glob "$ACTIVE_RUN_DIR/inputs/*.jsonl"
add_glob "$ACTIVE_RUN_DIR/data/*.jsonl"
add_glob "$QUEUE_DIR/**/*.jsonl"

if (( ${#INPUTS[@]} == 0 )); then
  "$PYTHON_BIN" - "$OUT_ROOT/dataset_integrity_manifest.json" <<'PY'
import datetime
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "schema": "omnicoder.dataset_integrity_audit_2026.v1",
    "status": "skipped",
    "reason": "no_trainable_jsonl_inputs_found",
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  echo "{\"status\":\"skipped\",\"reason\":\"no_trainable_jsonl_inputs_found\",\"manifest\":\"$OUT_ROOT/dataset_integrity_manifest.json\"}"
  exit 0
fi

printf '%s\n' "${INPUTS[@]}" > "$OUT_ROOT/input_args.txt"
"$PYTHON_BIN" - "$OUT_ROOT/input_args.txt" "$OUT_ROOT/input_files.json" <<'PY'
import json
import pathlib
import sys
items = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
paths = [items[index + 1] for index, item in enumerate(items[:-1]) if item == "--input"]
pathlib.Path(sys.argv[2]).write_text(json.dumps({"inputs": paths}, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

ARGS=("${INPUTS[@]}" --out-dir "$OUT_ROOT" --manifest "$OUT_ROOT/dataset_integrity_manifest.json" --max-artifact-bytes "$MAX_ARTIFACT_BYTES")
if [[ "$MAX_RECORDS" =~ ^[0-9]+$ ]] && (( MAX_RECORDS > 0 )); then
  ARGS+=(--max-records "$MAX_RECORDS")
fi
if [[ "$MAX_RECORDS_PER_INPUT" =~ ^[0-9]+$ ]] && (( MAX_RECORDS_PER_INPUT > 0 )); then
  ARGS+=(--max-records-per-input "$MAX_RECORDS_PER_INPUT")
fi
if [[ "$NO_ARTIFACT_SCAN" == "1" || "$NO_ARTIFACT_SCAN" == "true" ]]; then
  ARGS+=(--no-artifact-scan)
fi
if [[ "$WRITE_ACCEPTED" == "1" || "$WRITE_ACCEPTED" == "true" ]]; then
  ARGS+=(--write-accepted)
fi

"$PYTHON_BIN" -m omnicoder.data_factory.dataset_integrity_2026 "${ARGS[@]}" | tee "$OUT_ROOT/logs/dataset_integrity_audit.log"
