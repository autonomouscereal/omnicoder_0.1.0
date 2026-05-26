#!/usr/bin/env bash
set -euo pipefail

# Full capability-data curation pass for Omnicoder 2026.
# This is intended to prepare the next real training stage without touching any
# active optimizer container.

WEIGHTS_ROOT="${OMNICODER_WEIGHTS_ROOT:-/home/cereal/omnicoder_2026_work/weights}"
WORK_ROOT="${OMNICODER_WORK_ROOT:-/home/cereal/omnicoder_2026_work}"
REPO="${OMNICODER_REPO:-$WEIGHTS_ROOT/staged_patches/omnicoder_d28a1d4_allmodalfix_20260526T080000Z}"
PYTHON_BIN="${OMNICODER_CURATION_PYTHON:-python3}"
RUN_TAG_RAW="${OMNICODER_CURATION_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_TAG="${RUN_TAG_RAW//[^A-Za-z0-9_.-]/_}"
OUT_ROOT="${OMNICODER_CURATION_OUT_ROOT:-$WEIGHTS_ROOT/data_curation_agent_2026/runs/capability_policy_${RUN_TAG}}"
MAX_RECORDS="${OMNICODER_CURATION_MAX_RECORDS:-0}"

mkdir -p "$OUT_ROOT/jsonl" "$OUT_ROOT/rejected" "$OUT_ROOT/manifests" "$OUT_ROOT/logs"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

shopt -s nullglob globstar

run_family() {
  local family="$1"
  local modality="$2"
  local min_quality="$3"
  local require_media="$4"
  shift 4
  local -a inputs=()
  local pattern path
  for pattern in "$@"; do
    for path in $pattern; do
      if [[ -s "$path" ]]; then
        inputs+=("$path")
      fi
    done
  done
  local manifest="$OUT_ROOT/manifests/${family}.manifest.json"
  if (( ${#inputs[@]} == 0 )); then
    "$PYTHON_BIN" - <<PY
import json, pathlib, datetime
path = pathlib.Path("$manifest")
path.write_text(json.dumps({
    "schema": "omnicoder.dataset_curation_agent_2026.family.v1",
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "family": "$family",
    "modality": "$modality",
    "status": "skipped",
    "reason": "no_input_files",
    "input_pattern_count": $#,
}, ensure_ascii=True, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
    echo "{\"family\":\"$family\",\"status\":\"skipped\",\"reason\":\"no_input_files\"}" | tee "$OUT_ROOT/logs/${family}.log"
    return 0
  fi

  local -a cmd=("$PYTHON_BIN" -m omnicoder.data_factory.curation_policy_2026)
  for path in "${inputs[@]}"; do
    cmd+=(--input "$path")
  done
  cmd+=(
    --out "$OUT_ROOT/jsonl/${family}.clean.jsonl"
    --rejected "$OUT_ROOT/rejected/${family}.rejected.jsonl"
    --manifest "$manifest"
    --modality "$modality"
    --min-quality "$min_quality"
    --dedupe
  )
  if [[ "$MAX_RECORDS" =~ ^[0-9]+$ ]] && (( MAX_RECORDS > 0 )); then
    cmd+=(--max-records "$MAX_RECORDS")
  fi
  if [[ "$require_media" == "1" ]]; then
    cmd+=(--require-media-artifacts)
  fi
  "${cmd[@]}" | tee "$OUT_ROOT/logs/${family}.log"
}

run_family text text 0.65 0 \
  "$WEIGHTS_ROOT/external_datasets_2026/runs/curation_twentyfirst_fix_dee8580_20260525T170500Z/jsonl/text_pretraining_all.jsonl" \
  "$WEIGHTS_ROOT/curated_datasets_2026/latest/jsonl/train_text.jsonl"

run_family long_context long_context 0.65 0 \
  "$WEIGHTS_ROOT/external_datasets_2026/runs/curation_twentyfirst_fix_dee8580_20260525T170500Z/jsonl/long_context_all.jsonl" \
  "$WEIGHTS_ROOT/curated_datasets_2026/latest/jsonl/train_long_context.jsonl"

run_family code code 0.70 0 \
  "$WEIGHTS_ROOT/external_datasets_2026/runs/curation_twentyfirst_fix_dee8580_20260525T170500Z/jsonl/code_all.jsonl" \
  "$WEIGHTS_ROOT/curated_datasets_2026/latest/jsonl/train_code.jsonl"

run_family math math 0.70 0 \
  "$WEIGHTS_ROOT/external_datasets_2026/latest/jsonl/math_reasoning.jsonl" \
  "$WEIGHTS_ROOT/curated_datasets_2026/latest/jsonl/train_math.jsonl"

run_family tool tool 0.70 0 \
  "$WEIGHTS_ROOT/external_datasets_2026/runs/curation_twentyfirst_fix_dee8580_20260525T170500Z/jsonl/tool_all.jsonl" \
  "$WEIGHTS_ROOT/curated_datasets_2026/latest/jsonl/train_tool.jsonl" \
  "$WEIGHTS_ROOT/agentic_tool_training_2026/moretrain6b_enriched_fullmodal_rl_20260523T083822Z/tool_sft.jsonl" \
  "$WEIGHTS_ROOT/agentic_tool_training_2026/moretrain6b_enriched_fullmodal_rl_20260523T083822Z/tool_reward.jsonl" \
  "$WEIGHTS_ROOT/agentic_tool_training_2026/moretrain6b_enriched_fullmodal_rl_20260523T083822Z/tool_rlvr.jsonl" \
  "$WEIGHTS_ROOT/agentic_tool_training_2026/moretrain6b_enriched_fullmodal_rl_20260523T083822Z/tool_preference.jsonl"

run_family agentic agentic 0.70 0 \
  "$WEIGHTS_ROOT/data_factory/trace_orchestrator_2026/jsonl/*.jsonl" \
  "$WEIGHTS_ROOT/data_factory/trace_orchestrator_2026/exports/*.jsonl" \
  "$WORK_ROOT/data/raw/agent_memory_events_2026.jsonl" \
  "$WORK_ROOT/data/raw/codex_traces_2026/*.jsonl" \
  "$WORK_ROOT/data/raw/claude_traces_2026/*.jsonl" \
  "$WEIGHTS_ROOT/staged_trace_uploads/*/data/raw/*.jsonl" \
  "$WEIGHTS_ROOT/staged_trace_uploads/*/data/raw/**/*.jsonl" \
  "$WEIGHTS_ROOT/curated_datasets_2026/latest/jsonl/train_agentic_focus.jsonl"

run_family image image 0.60 1 \
  "$WEIGHTS_ROOT/external_datasets_2026/runs/curation_twentyfirst_fix_dee8580_20260525T170500Z/jsonl/image_all.jsonl" \
  "$WEIGHTS_ROOT/curated_datasets_2026/latest/jsonl/train_image.jsonl"

run_family video video 0.60 1 \
  "$WEIGHTS_ROOT/external_datasets_2026/runs/curation_twentyfirst_fix_dee8580_20260525T170500Z/jsonl/video_all.jsonl" \
  "$WEIGHTS_ROOT/curated_datasets_2026/latest/jsonl/train_video.jsonl"

run_family audio audio 0.60 1 \
  "$WEIGHTS_ROOT/external_datasets_2026/runs/omnicoder_sidecar_external_clean_reviewed_574f615_20260525T130729Z/jsonl/audio_all.jsonl" \
  "$WEIGHTS_ROOT/external_datasets_2026/runs/curation_28_0aeace4_20260525T203044Z/jsonl/audio_all.jsonl" \
  "$WEIGHTS_ROOT/curated_datasets_2026/latest/jsonl/train_audio.jsonl"

run_family music music 0.60 1 \
  "$WEIGHTS_ROOT/external_datasets_2026/runs/omnicoder_sidecar_external_clean_reviewed_574f615_20260525T130729Z/jsonl/music_all.jsonl" \
  "$WEIGHTS_ROOT/external_datasets_2026/runs/curation_28_0aeace4_20260525T203044Z/jsonl/music_all.jsonl" \
  "$WEIGHTS_ROOT/curated_datasets_2026/latest/jsonl/train_music.jsonl"

run_family ocr ocr 0.70 1 \
  "$WEIGHTS_ROOT/curated_datasets_2026/latest/jsonl/train_ocr.jsonl" \
  "$WEIGHTS_ROOT/external_datasets_2026/runs/*/jsonl/ocr*.jsonl"

"$PYTHON_BIN" - <<PY
import hashlib
import json
from pathlib import Path

root = Path("$OUT_ROOT")
combined = root / "jsonl" / "all_modalities_capability_clean.jsonl"
families = []
with combined.open("w", encoding="utf-8", newline="\\n") as out:
    for manifest_path in sorted((root / "manifests").glob("*.manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        families.append(manifest)
        clean = root / "jsonl" / f"{manifest_path.name.removesuffix('.manifest.json')}.clean.jsonl"
        if clean.exists():
            with clean.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.strip():
                        out.write(line)

def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)

digest = hashlib.sha256()
if combined.exists():
    with combined.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
index = {
    "schema": "omnicoder.capability_curation_run_2026.v1",
    "run_tag": "$RUN_TAG",
    "out_root": str(root),
    "combined_jsonl": str(combined),
    "combined_records": count_lines(combined),
    "combined_sha256": digest.hexdigest(),
    "families": families,
}
(root / "curation_manifest_index.json").write_text(json.dumps(index, ensure_ascii=True, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps({"status": "ok", "manifest": str(root / "curation_manifest_index.json"), "combined_records": index["combined_records"], "combined_sha256": index["combined_sha256"]}, ensure_ascii=True, sort_keys=True))
PY
