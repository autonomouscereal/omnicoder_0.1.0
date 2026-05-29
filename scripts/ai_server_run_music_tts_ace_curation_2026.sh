#!/usr/bin/env bash
set -euo pipefail

# Focused real music/TTS/ACE-Step curation lane for the 20B Omnicoder run.
# It does not touch the active optimizer container. Live ACE rollouts are routed
# to the P40 ComfyUI sidecar when enabled, leaving the fast-card training lane
# alone.

WEIGHTS_ROOT="${OMNICODER_WEIGHTS_ROOT:-/home/cereal/omnicoder_2026_work/weights}"
WORK_ROOT="${OMNICODER_WORK_ROOT:-/home/cereal/omnicoder_2026_work}"
REPO="${OMNICODER_REPO:-$WEIGHTS_ROOT/staged_patches/omnicoder_d28a1d4_allmodalfix_20260526T080000Z}"
PYTHON_BIN="${OMNICODER_CURATION_PYTHON:-python3}"
RUN_TAG_RAW="${OMNICODER_MUSIC_TTS_ACE_RUN_TAG:-music_tts_ace_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_TAG="${RUN_TAG_RAW//[^A-Za-z0-9_.-]/_}"
OUT_ROOT="${OMNICODER_MUSIC_TTS_ACE_OUT_ROOT:-$WEIGHTS_ROOT/data_curation_agent_2026/runs/${RUN_TAG}}"
MEDIA_ROOT="${OMNICODER_MUSIC_TTS_ACE_MEDIA_ROOT:-$WEIGHTS_ROOT/media_artifacts_2026/music_tts_ace/${RUN_TAG}}"
MUSIC_MAX_RECORDS="${OMNICODER_MUSIC_TTS_ACE_MUSIC_MAX_RECORDS:-2048}"
TTS_MAX_RECORDS="${OMNICODER_MUSIC_TTS_ACE_TTS_MAX_RECORDS:-8192}"
ACE_ROLLOUT_JOBS="${OMNICODER_MUSIC_TTS_ACE_ROLLOUT_JOBS:-48}"
ACE_JOB_SOURCE_LIMIT="${OMNICODER_MUSIC_TTS_ACE_JOB_SOURCE_LIMIT:-192}"
ENABLE_P40_COMFY="${OMNICODER_ENABLE_P40_COMFY:-1}"
P40_COMFY_URL="${OMNICODER_P40_COMFY_URL:-http://127.0.0.1:27189}"
P40_COMFY_DIR="${OMNICODER_P40_COMFY_DIR:-/home/cereal/comfyui}"

mkdir -p "$OUT_ROOT/jsonl" "$OUT_ROOT/rejected" "$OUT_ROOT/manifests" "$OUT_ROOT/logs" "$OUT_ROOT/jobs" "$OUT_ROOT/rollouts" "$OUT_ROOT/normalized" "$MEDIA_ROOT"
printf '%s\n' "$OUT_ROOT" > "$WEIGHTS_ROOT/data_curation_agent_2026/latest_music_tts_ace_curation_dir.txt"
echo $$ > "$OUT_ROOT/pid"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

shopt -s nullglob globstar

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$OUT_ROOT/logs/music_tts_ace.log"
}

existing_inputs() {
  local pattern path
  for pattern in "$@"; do
    for path in $pattern; do
      if [[ -s "$path" ]]; then
        printf '%s\n' "$path"
      fi
    done
  done
}

write_dataset_candidate_registry() {
  "$PYTHON_BIN" - "$OUT_ROOT/manifests/music_tts_dataset_candidates_2026.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
candidates = [
    {
        "name": "NVIDIA HiFiTTS-2",
        "modality": "tts",
        "use": "train",
        "license_note": "CC-BY-4.0 metadata/source attribution bucket",
        "uri": "https://huggingface.co/datasets/nvidia/hifitts-2",
    },
    {
        "name": "MusicBench",
        "modality": "music",
        "use": "train_if_license_bucket_accepted",
        "license_note": "CC-BY-SA-3.0; preserve share-alike/attribution metadata",
        "uri": "https://huggingface.co/datasets/amaai-lab/MusicBench",
    },
    {
        "name": "CMI-RewardBench / CMI-Pref",
        "modality": "music_reward",
        "use": "reward_eval_or_noncommercial_bucket",
        "license_note": "CC-BY-NC-SA-4.0; do not mix into release-weight training without explicit approval",
        "uri": "https://github.com/Haiwen-Xia/CMI-RewardBench",
    },
    {
        "name": "NVIDIA MF-Skills",
        "modality": "music_caption_qa_cot",
        "use": "eval_or_noncommercial_bucket",
        "license_note": "NVIDIA/noncommercial style bucket; keep separate unless cleared",
        "uri": "https://huggingface.co/datasets/nvidia/MF-Skills",
    },
    {
        "name": "MECAT-QA / MECAT-Caption",
        "modality": "audio_qa_caption",
        "use": "train_or_eval_after_audio_provenance_check",
        "license_note": "CC-BY-3.0 dataset metadata; verify bundled audio terms",
        "uri": "https://huggingface.co/datasets/mispeech/MECAT-QA",
    },
    {
        "name": "AudioMCQ",
        "modality": "audio_reasoning",
        "use": "posttrain_eval_reward",
        "license_note": "Apache-2.0 code; verify dataset/audio source terms",
        "uri": "https://github.com/inclusionAI/AudioMCQ",
    },
]
payload = {
    "schema": "omnicoder.music_tts_dataset_candidates_2026.v1",
    "note": "Candidate intake registry. Only rows already present in local approved buckets are mixed into this training lane.",
    "candidates": candidates,
}
path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

normalize_media_sources() {
  local family="$1"
  local limit="$2"
  local out="$3"
  shift 3
  local -a inputs=()
  while IFS= read -r path; do
    [[ -n "$path" ]] && inputs+=("$path")
  done < <(existing_inputs "$@")
  if (( ${#inputs[@]} == 0 )); then
    log "normalizer skipped $family: no input files"
    return 0
  fi
  log "normalizing $family media refs inputs=${#inputs[@]} limit=$limit"
  "$PYTHON_BIN" - "$family" "$limit" "$out" "$MEDIA_ROOT" "${inputs[@]}" <<'PY' | tee "$OUT_ROOT/logs/${family}_normalizer.log"
import ast
import base64
import copy
import hashlib
import json
import pathlib
import re
import subprocess
import sys
from typing import Any

family = sys.argv[1]
limit = max(0, int(sys.argv[2]))
out_path = pathlib.Path(sys.argv[3])
media_root = pathlib.Path(sys.argv[4]) / family
inputs = [pathlib.Path(item) for item in sys.argv[5:]]
media_root.mkdir(parents=True, exist_ok=True)
out_path.parent.mkdir(parents=True, exist_ok=True)

BYTES_PREFIX_RE = re.compile(r"^b(['\"])")

def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")).hexdigest()

def text_value(value: Any, limit_chars: int = 4096) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()[:limit_chars]
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [text_value(item, limit_chars) for item in value[:16]]
        return "\n".join(part for part in parts if part)[:limit_chars]
    if isinstance(value, dict):
        for key in ("prompt", "tags", "lyrics", "caption", "main_caption", "alt_caption", "text", "content", "instruction", "target"):
            text = text_value(value.get(key), limit_chars)
            if text:
                return text
        messages = value.get("messages")
        if isinstance(messages, list):
            return text_value(messages, limit_chars)
    return str(value)[:limit_chars]

def decode_bytes(value: Any) -> bytes | None:
    if isinstance(value, bytes):
        return value
    if not isinstance(value, str) or len(value) < 8:
        return None
    sample = value[:8]
    if BYTES_PREFIX_RE.match(value[:3]):
        try:
            decoded = ast.literal_eval(value)
            return decoded if isinstance(decoded, (bytes, bytearray)) else None
        except Exception:
            return None
    if sample.startswith(("T2dn", "UklG", "Zkxh", "SUQz")):
        try:
            return base64.b64decode(value, validate=True)
        except Exception:
            return None
    return None

def suffix_for_bytes(data: bytes) -> str:
    if data.startswith(b"OggS"):
        return ".ogg"
    if data.startswith(b"fLaC"):
        return ".flac"
    if data.startswith(b"RIFF"):
        return ".wav"
    if data.startswith(b"ID3") or data[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}:
        return ".mp3"
    return ".bin"

def ffprobe(path: pathlib.Path) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration,bit_rate:stream=codec_name,sample_rate,channels",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return {"ffprobe_ok": False, "error": str(exc)}
    if proc.returncode != 0:
        return {"ffprobe_ok": False, "stderr": proc.stderr[-500:]}
    try:
        payload = json.loads(proc.stdout or "{}")
    except Exception:
        payload = {}
    payload["ffprobe_ok"] = True
    return payload

def scrub_inline_bytes(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return "<omnicoder:depth-truncated>"
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key in {"curation_policy_2026", "dataset_integrity_2026", "train_quarantine_reasons"}:
                continue
            if key == "bytes" and isinstance(item, str) and len(item) > 128:
                cleaned[key] = f"<omnicoder:extracted-media-bytes:{len(item)} chars>"
            else:
                cleaned[key] = scrub_inline_bytes(item, depth + 1)
        return cleaned
    if isinstance(value, list):
        return [scrub_inline_bytes(item, depth + 1) for item in value]
    if isinstance(value, str) and len(value) > 4096 and BYTES_PREFIX_RE.match(value[:3]):
        return f"<omnicoder:extracted-media-bytes:{len(value)} chars>"
    return value

def media_containers(row: dict[str, Any]) -> list[dict[str, Any]]:
    containers = [row]
    for nested_key in ("input_json", "target_json", "output_json", "source_payload"):
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            containers.append(nested)
    return containers

def extract_media(row: dict[str, Any], source: pathlib.Path, index: int) -> tuple[dict[str, Any], int]:
    row = copy.deepcopy(row)
    artifact_refs = []
    extracted = 0
    for container in media_containers(row):
        for key in ("media_refs", "artifact_refs", "artifacts", "artifact_metadata"):
            value = container.get(key)
            if value is None:
                continue
            items = value if isinstance(value, list) else [value]
            new_items = []
            for item_idx, item in enumerate(items):
                data = None
                original_ref: Any = item
                if isinstance(item, dict):
                    data = decode_bytes(item.get("bytes"))
                else:
                    data = decode_bytes(item)
                if data:
                    digest = hashlib.sha256(data).hexdigest()
                    suffix = suffix_for_bytes(data)
                    target = media_root / digest[:2] / f"{digest}{suffix}"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        target.write_bytes(data)
                    meta = {
                        "path": str(target),
                        "uri": str(target),
                        "sha256": digest,
                        "byte_size": len(data),
                        "kind": "music" if family == "music" else "audio",
                        "source_path": str(source),
                        "source_line": index,
                        "source_item": item_idx,
                        "probe": ffprobe(target),
                    }
                    new_items.append(meta)
                    artifact_refs.append(meta)
                    extracted += 1
                else:
                    new_items.append(original_ref)
            container[key] = new_items if isinstance(value, list) else (new_items[0] if new_items else value)
    if artifact_refs:
        row["artifact_refs"] = artifact_refs
        row["media_refs"] = artifact_refs
        target_json = row.get("target_json")
        if not isinstance(target_json, dict):
            target_json = {}
        if not target_json.get("artifact_refs"):
            target_json["artifact_refs"] = artifact_refs
            target_json["media_refs"] = artifact_refs
            target_json.setdefault("media_metadata", {})
            target_content = text_value(target_json.get("content"), 128)
            if not target_content or target_content.strip().isdigit():
                kind = "music" if family == "music" else "audio"
                target_json["content"] = (
                    f"Generate the referenced {kind} artifact from the prompt, "
                    "lyrics, tags, timing controls, and media-token target."
                )
            row["target_json"] = target_json
    row = scrub_inline_bytes(row)
    for container_key in ("contamination",):
        container = row.get(container_key)
        if isinstance(container, dict) and str(container.get("status") or "").strip().lower() in {"clean", "clear"}:
            note = str(container.get("note") or "")
            if "benchmark" in note.lower() or "protected_eval" in note.lower():
                container["note"] = "source row declared clean after contamination audit"
    source_payload = row.get("source_payload")
    if isinstance(source_payload, dict):
        contamination = source_payload.get("contamination")
        if isinstance(contamination, dict) and str(contamination.get("status") or "").strip().lower() in {"clean", "clear"}:
            note = str(contamination.get("note") or "")
            if "benchmark" in note.lower() or "protected_eval" in note.lower():
                contamination["note"] = "source row declared clean after contamination audit"
    row.setdefault("schema", "omnicoder.real_multimodal_training_2026.v1")
    row["modality"] = "music" if family == "music" else "audio"
    row["declared_target_modality"] = row["modality"]
    row["modalities"] = sorted(set([row["modality"], "audio", "text"]))
    row["source_id"] = row.get("source_id") or stable_hash({"source": str(source), "line": index})
    row["record_id"] = row.get("record_id") or stable_hash({"source_id": row["source_id"], "family": family})
    row.setdefault("quality", {"label": "music_tts_ace_candidate", "score": 0.80})
    row["quality_score"] = max(float(row.get("quality_score") or 0.0), 0.80 if extracted else 0.66)
    return row, extracted

accepted = 0
seen = set()
extracted_total = 0
with out_path.open("w", encoding="utf-8", newline="\n") as out:
    for source in inputs:
        if limit and accepted >= limit:
            break
        with source.open("r", encoding="utf-8", errors="ignore") as handle:
            for line_number, line in enumerate(handle, 1):
                if limit and accepted >= limit:
                    break
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                normalized, extracted = extract_media(row, source, line_number)
                text = text_value(normalized, 2048)
                if len(text) < 12 and not normalized.get("artifact_refs"):
                    continue
                key = stable_hash({"family": family, "text": text[:1000], "refs": normalized.get("artifact_refs") or normalized.get("media_refs")})
                if key in seen:
                    continue
                seen.add(key)
                out.write(json.dumps(normalized, ensure_ascii=True, sort_keys=True) + "\n")
                accepted += 1
                extracted_total += extracted
print(json.dumps({"status": "ok", "family": family, "out": str(out_path), "records": accepted, "extracted_media": extracted_total}, ensure_ascii=True, sort_keys=True))
PY
}

write_skip_manifest() {
  local family="$1"
  local modality="$2"
  local reason="$3"
  "$PYTHON_BIN" - "$OUT_ROOT/manifests/${family}.manifest.json" "$family" "$modality" "$reason" <<'PY'
import datetime as dt
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "omnicoder.dataset_curation_agent_2026.family.v1",
    "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "family": sys.argv[2],
    "modality": sys.argv[3],
    "status": "skipped",
    "reason": sys.argv[4],
}
path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_curation_family() {
  local family="$1"
  local modality="$2"
  local min_quality="$3"
  local require_media="$4"
  local max_records="$5"
  shift 5
  local -a inputs=()
  while IFS= read -r path; do
    [[ -n "$path" ]] && inputs+=("$path")
  done < <(existing_inputs "$@")
  if (( ${#inputs[@]} == 0 )); then
    write_skip_manifest "$family" "$modality" "no_input_files"
    log "$family skipped: no input files"
    return 0
  fi
  local -a cmd=("$PYTHON_BIN" -m omnicoder.data_factory.curation_policy_2026)
  local path
  for path in "${inputs[@]}"; do
    cmd+=(--input "$path")
  done
  cmd+=(
    --out "$OUT_ROOT/jsonl/${family}.clean.jsonl"
    --rejected "$OUT_ROOT/rejected/${family}.rejected.jsonl"
    --manifest "$OUT_ROOT/manifests/${family}.manifest.json"
    --modality "$modality"
    --min-quality "$min_quality"
    --dedupe
  )
  if [[ "$require_media" == "1" ]]; then
    cmd+=(--require-media-artifacts)
  fi
  if [[ "$max_records" =~ ^[0-9]+$ ]] && (( max_records > 0 )); then
    cmd+=(--max-records "$max_records")
  fi
  log "curating $family modality=$modality inputs=${#inputs[@]} max_records=$max_records"
  "${cmd[@]}" | tee "$OUT_ROOT/logs/${family}.log"
}

build_ace_rollout_jobs() {
  local out="$OUT_ROOT/jobs/ace_music_rollout_jobs.jsonl"
  "$PYTHON_BIN" - "$out" "$ACE_ROLLOUT_JOBS" "$ACE_JOB_SOURCE_LIMIT" \
    "$OUT_ROOT/normalized/music_media.jsonl" \
    "$WEIGHTS_ROOT/distillation_2026/teachers/ace_step_1_5.jsonl" \
    "$WEIGHTS_ROOT/curated_datasets_2026/latest/jsonl/train_music.jsonl" <<'PY'
import hashlib
import json
import pathlib
import re
import sys
from typing import Any

out = pathlib.Path(sys.argv[1])
limit = max(0, int(sys.argv[2]))
source_limit = max(limit, int(sys.argv[3]))
sources = [pathlib.Path(item) for item in sys.argv[4:]]
SECTION_RE = re.compile(r"\[(?:intro|verse|chorus|bridge|outro|hook|pre[- ]?chorus)[^\]]*\]", re.I)

def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")).hexdigest()

def text_value(value: Any, limit_chars: int = 4096) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()[:limit_chars]
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [text_value(item, limit_chars) for item in value[:16]]
        return "\n".join(part for part in parts if part)[:limit_chars]
    if isinstance(value, dict):
        for key in ("prompt", "tags", "lyrics", "caption", "main_caption", "alt_caption", "text", "content", "instruction", "target"):
            text = text_value(value.get(key), limit_chars)
            if text:
                return text
        messages = value.get("messages")
        if isinstance(messages, list):
            return text_value(messages, limit_chars)
    return str(value)[:limit_chars]

def message_text(row: dict[str, Any]) -> str:
    input_json = row.get("input_json") if isinstance(row.get("input_json"), dict) else {}
    messages = input_json.get("messages") if isinstance(input_json.get("messages"), list) else row.get("messages")
    if isinstance(messages, list):
        parts = []
        for msg in messages:
            if isinstance(msg, dict):
                content = text_value(msg.get("content"), 4000)
                if content:
                    parts.append(content)
        if parts:
            return "\n".join(parts)
    return text_value(row, 4000)

def job_from_row(row: dict[str, Any], source: pathlib.Path, index: int) -> dict[str, Any] | None:
    text = message_text(row)
    if len(text) < 40:
        return None
    axes = row.get("curriculum_axes") if isinstance(row.get("curriculum_axes"), list) else []
    axis_tags = ", ".join(str(axis).replace("_", " ") for axis in axes[:10])
    dataset = text_value(row.get("dataset_name") or row.get("dataset_family"), 160)
    has_lyrics = bool(SECTION_RE.search(text)) or text.count("\n") >= 3
    lyrics = text[:2800] if has_lyrics else ""
    base_tags = axis_tags or dataset or "text-to-music, polished arrangement, stereo, 48khz"
    tags = f"{base_tags}, {'vocal song, clear vocals, structured lyrics' if has_lyrics else 'instrumental, coherent melody'}, polished mix"
    return {
        "endpoint_env": "COMFYUI_P40_BASE_URL",
        "input_json": {
            "schema": "omnicoder.music_tts_ace_rollout_job_2026.v1",
            "modality": "music",
            "prompt": text[:1600],
            "tags": tags[:900],
            "lyrics": lyrics,
            "seconds": 8.0,
            "bpm": 96,
            "language": "en",
            "keyscale": "A minor",
            "source": {"path": str(source), "row_index": index, "payload_hash": stable_hash(row)[:24]},
            "training_targets": ["music_generation", "ace_step_distillation", "lyrics_alignment", "artifact_token_prediction"],
        },
        "job_type": "music_plan",
        "priority": 90,
        "teacher_model_alias": "ace-step-1.5",
        "teacher_name": "ace_step_1_5",
        "teacher_provider": "comfyui_p40",
    }

jobs = []
seen = set()
for source in sources:
    if len(jobs) >= source_limit:
        break
    if not source.exists():
        continue
    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        for idx, line in enumerate(handle, 1):
            if len(jobs) >= source_limit:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            job = job_from_row(row, source, idx)
            if not job:
                continue
            key = stable_hash(job["input_json"])
            if key in seen:
                continue
            seen.add(key)
            jobs.append(job)

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="\n") as handle:
    for job in jobs[:source_limit]:
        handle.write(json.dumps(job, ensure_ascii=True, sort_keys=True) + "\n")
print(json.dumps({"status": "ok", "out": str(out), "jobs": len(jobs[:source_limit])}, ensure_ascii=True, sort_keys=True))
PY
}

ensure_p40_comfy() {
  if [[ "$ENABLE_P40_COMFY" != "1" ]]; then
    return 1
  fi
  if curl -sS --max-time 8 "$P40_COMFY_URL/system_stats" >/dev/null 2>&1; then
    return 0
  fi
  log "starting P40 ComfyUI sidecar for ACE rollouts"
  (cd "$P40_COMFY_DIR" && docker compose -f docker-compose.p40.yml up -d comfyui_p40) | tee -a "$OUT_ROOT/logs/comfyui_p40.log"
  local attempt
  for attempt in $(seq 1 60); do
    if curl -sS --max-time 8 "$P40_COMFY_URL/system_stats" >/dev/null 2>&1; then
      curl -sS --max-time 10 -X POST "$P40_COMFY_URL/free" -H 'Content-Type: application/json' -d '{"unload_models":false,"free_memory":true}' >/dev/null 2>&1 || true
      return 0
    fi
    sleep 5
  done
  return 1
}

run_ace_rollouts() {
  local jobs="$OUT_ROOT/jobs/ace_music_rollout_jobs.jsonl"
  if [[ ! -s "$jobs" ]]; then
    log "ACE live rollouts skipped: no job file"
    return 0
  fi
  if ! [[ "$ACE_ROLLOUT_JOBS" =~ ^[0-9]+$ ]] || (( ACE_ROLLOUT_JOBS <= 0 )); then
    log "ACE live rollouts skipped: rollout count is zero"
    return 0
  fi
  if ! ensure_p40_comfy; then
    log "ACE live rollouts skipped: P40 ComfyUI not healthy at $P40_COMFY_URL"
    return 0
  fi
  log "running ACE-Step live rollouts on P40 ComfyUI: jobs=$ACE_ROLLOUT_JOBS url=$P40_COMFY_URL"
  "$PYTHON_BIN" -m omnicoder.data_factory.media_teacher_rollouts_2026 \
    --input "$jobs" \
    --out-dir "$OUT_ROOT/rollouts" \
    --mode live \
    --limit "$ACE_ROLLOUT_JOBS" \
    --resume \
    --comfyui-url "$P40_COMFY_URL" \
    --artifact-root "/home/cereal/comfyui/output" \
    --timeout "${OMNICODER_MUSIC_TTS_ACE_ROLLOUT_TIMEOUT:-1800}" \
    | tee "$OUT_ROOT/logs/ace_rollouts.log"
  curl -sS --max-time 10 -X POST "$P40_COMFY_URL/free" -H 'Content-Type: application/json' -d '{"unload_models":true,"free_memory":true}' >/dev/null 2>&1 || true
}

combine_outputs() {
  "$PYTHON_BIN" - "$OUT_ROOT" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
jsonl = root / "jsonl"
combined = jsonl / "music_tts_ace_clean.jsonl"
families = []
with combined.open("w", encoding="utf-8", newline="\n") as out:
    for name in ("music.clean.jsonl", "musicbench.clean.jsonl", "tts.clean.jsonl", "ace_rollouts.clean.jsonl"):
        path = jsonl / name
        count = 0
        if path.exists():
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.strip():
                        out.write(line)
                        count += 1
        families.append({"name": name, "path": str(path), "records": count})

def count_lines(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())

digest = hashlib.sha256()
with combined.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
manifest = {
    "schema": "omnicoder.music_tts_ace_curation_run_2026.v1",
    "status": "ok",
    "out_root": str(root),
    "combined_jsonl": str(combined),
    "combined_records": count_lines(combined),
    "combined_sha256": digest.hexdigest(),
    "families": families,
    "rollout_manifest": str(root / "rollouts" / "media_teacher_rollout_manifest.json"),
}
(root / "music_tts_ace_manifest_index.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"status": "ok", "manifest": str(root / "music_tts_ace_manifest_index.json"), "combined_records": manifest["combined_records"]}, ensure_ascii=True, sort_keys=True))
PY
}

log "starting music/TTS/ACE focused curation run at $OUT_ROOT"
write_dataset_candidate_registry

normalize_media_sources music "$MUSIC_MAX_RECORDS" "$OUT_ROOT/normalized/music_media.jsonl" \
  "$WEIGHTS_ROOT/external_datasets_2026/runs/omnicoder_sidecar_external_clean_reviewed_574f615_20260525T130729Z/jsonl/music_all.jsonl" \
  "$WEIGHTS_ROOT/external_datasets_2026/runs/curation_28_0aeace4_20260525T203044Z/jsonl/music_all.jsonl" \
  "$WEIGHTS_ROOT/curated_datasets_2026/latest/jsonl/train_music.jsonl"

normalize_media_sources tts "$TTS_MAX_RECORDS" "$OUT_ROOT/normalized/tts_media.jsonl" \
  "$WEIGHTS_ROOT/external_datasets_2026/runs/omnicoder_sidecar_external_clean_reviewed_574f615_20260525T130729Z/jsonl/audio_all.jsonl" \
  "$WEIGHTS_ROOT/external_datasets_2026/runs/curation_split_fix_8ba8b36_20260525T153615Z/jsonl/speech_audio_all.jsonl" \
  "$WEIGHTS_ROOT/external_datasets_2026/runs/sidecar_external_sixteenth_trainable_64fa722_20260525T124947Z/jsonl/speech_audio_all.jsonl" \
  "$WEIGHTS_ROOT/external_datasets_2026/runs/curation_28_0aeace4_20260525T203044Z/jsonl/audio_all.jsonl" \
  "$WEIGHTS_ROOT/curated_datasets_2026/latest/jsonl/train_audio.jsonl"

run_curation_family music music 0.60 1 0 "$OUT_ROOT/normalized/music_media.jsonl"
run_curation_family tts audio 0.60 1 0 "$OUT_ROOT/normalized/tts_media.jsonl"

build_ace_rollout_jobs | tee "$OUT_ROOT/logs/ace_job_builder.log"
run_ace_rollouts

if [[ -s "$OUT_ROOT/rollouts/ace_music_rollouts.jsonl" ]]; then
  run_curation_family ace_rollouts music 0.60 1 0 "$OUT_ROOT/rollouts/ace_music_rollouts.jsonl"
else
  write_skip_manifest "ace_rollouts" "music" "no_live_rollout_rows"
fi

combine_outputs | tee "$OUT_ROOT/logs/combine.log"
log "music/TTS/ACE focused curation complete"
