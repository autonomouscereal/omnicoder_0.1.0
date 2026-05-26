#!/usr/bin/env bash
set -euo pipefail

# Supplemental real music/TTS intake for Omnicoder 20B.
# This script is intentionally a sidecar: it does not touch the active optimizer
# container. It expands the completed music_tts_ace curation directory with:
# - LAION Orpheus-style expressive TTS parquet rows with extracted WAV artifacts
# - JamendoMaxCaps CC-BY / CC-BY-SA music rows with downloaded MP3 artifacts
# - optional additional ACE-Step live teacher rollouts on the P40 ComfyUI sidecar

WEIGHTS_ROOT="${OMNICODER_WEIGHTS_ROOT:-/home/cereal/omnicoder_2026_work/weights}"
REPO="${OMNICODER_REPO:-$WEIGHTS_ROOT/staged_patches/omnicoder_d28a1d4_allmodalfix_20260526T080000Z}"
BASE_DIR="${OMNICODER_MUSIC_TTS_ACE_BASE_DIR:-}"
if [[ -z "$BASE_DIR" && -s "$WEIGHTS_ROOT/data_curation_agent_2026/latest_music_tts_ace_curation_dir.txt" ]]; then
  BASE_DIR="$(cat "$WEIGHTS_ROOT/data_curation_agent_2026/latest_music_tts_ace_curation_dir.txt")"
fi
if [[ -z "$BASE_DIR" || ! -s "$BASE_DIR/music_tts_ace_manifest_index.json" ]]; then
  echo "Base music/TTS/ACE curation dir is missing or incomplete: $BASE_DIR" >&2
  exit 2
fi

RUN_TAG_RAW="${OMNICODER_MUSIC_TTS_EXPANSION_RUN_TAG:-music_tts_expansion_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_TAG="${RUN_TAG_RAW//[^A-Za-z0-9_.-]/_}"
OUT_ROOT="${OMNICODER_MUSIC_TTS_EXPANSION_OUT_ROOT:-$WEIGHTS_ROOT/data_curation_agent_2026/runs/${RUN_TAG}}"
MEDIA_ROOT="${OMNICODER_MUSIC_TTS_EXPANSION_MEDIA_ROOT:-$WEIGHTS_ROOT/media_artifacts_2026/music_tts_ace/${RUN_TAG}}"
PYTHON_BIN="${OMNICODER_CURATION_PYTHON:-python3}"
HF_PYTHON="${OMNICODER_HF_AUDIO_PYTHON:-$WEIGHTS_ROOT/tools/hf_audio_intake_venv/bin/python}"

LAION_FILES="${OMNICODER_LAION_ORPHEUS_PARQUET_FILES:-6}"
LAION_MAX_RECORDS="${OMNICODER_LAION_ORPHEUS_MAX_RECORDS:-2048}"
JAMENDO_JSONL_FILES="${OMNICODER_JAMENDO_JSONL_FILES:-24}"
JAMENDO_MAX_RECORDS="${OMNICODER_JAMENDO_MAX_RECORDS:-192}"
ACE_EXTRA_JOBS="${OMNICODER_MUSIC_TTS_EXTRA_ACE_JOBS:-64}"
P40_COMFY_URL="${OMNICODER_P40_COMFY_URL:-http://127.0.0.1:27189}"
P40_COMFY_DIR="${OMNICODER_P40_COMFY_DIR:-/home/cereal/comfyui}"

mkdir -p "$OUT_ROOT/jsonl" "$OUT_ROOT/rejected" "$OUT_ROOT/manifests" "$OUT_ROOT/logs" "$OUT_ROOT/raw" "$OUT_ROOT/jobs" "$OUT_ROOT/rollouts" "$MEDIA_ROOT"
printf '%s\n' "$OUT_ROOT" > "$WEIGHTS_ROOT/data_curation_agent_2026/latest_music_tts_ace_curation_dir.txt"
echo $$ > "$OUT_ROOT/pid"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$OUT_ROOT/logs/music_tts_expansion.log"
}

ensure_hf_python() {
  if [[ -x "$HF_PYTHON" ]]; then
    return 0
  fi
  log "creating HF audio intake venv at $(dirname "$(dirname "$HF_PYTHON")")"
  "$PYTHON_BIN" -m venv "$(dirname "$(dirname "$HF_PYTHON")")"
  "$HF_PYTHON" -m pip install --upgrade pip
  "$HF_PYTHON" -m pip install pyarrow requests tqdm
}

run_curation_family() {
  local family="$1"
  local modality="$2"
  local min_quality="$3"
  local require_media="$4"
  local max_records="$5"
  local input="$6"
  if [[ ! -s "$input" ]]; then
    log "$family skipped: no input file $input"
    "$PYTHON_BIN" - "$OUT_ROOT/manifests/${family}.manifest.json" "$family" "$modality" <<'PY'
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
    "reason": "no_input_file",
}
path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    return 0
  fi
  local -a cmd=("$PYTHON_BIN" -m omnicoder.data_factory.curation_policy_2026
    --input "$input"
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
  log "curating $family modality=$modality max_records=$max_records"
  "${cmd[@]}" | tee "$OUT_ROOT/logs/${family}.log"
}

extract_laion_orpheus_tts() {
  log "extracting LAION Orpheus-style expressive TTS files=$LAION_FILES max=$LAION_MAX_RECORDS"
  "$HF_PYTHON" - "$OUT_ROOT/raw/laion_orpheus_tts.raw.jsonl" "$MEDIA_ROOT/laion_orpheus_tts" "$LAION_FILES" "$LAION_MAX_RECORDS" <<'PY' | tee "$OUT_ROOT/logs/laion_orpheus_tts_intake.log"
import datetime as dt
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import time

import pyarrow.parquet as pq
import requests

out_path = pathlib.Path(sys.argv[1])
media_root = pathlib.Path(sys.argv[2])
file_count = max(0, int(sys.argv[3]))
max_records = max(0, int(sys.argv[4]))
raw_root = out_path.parent / "laion_orpheus_parquet"
raw_root.mkdir(parents=True, exist_ok=True)
media_root.mkdir(parents=True, exist_ok=True)

base = "https://huggingface.co/datasets/laion/laions_got_talent_with_voice_emotion_speed_tags_for_orpheus_tuning/resolve/refs%2Fconvert%2Fparquet/default/partial-train/{idx:04d}.parquet"
VOICE_TEXT_RE = re.compile(r'text="(.*)"\\s*>?$', re.S)

def stable_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")).hexdigest()

def ffprobe(path):
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration,bit_rate:stream=codec_name,sample_rate,channels",
                "-of", "json", str(path),
            ],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
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

def download(url, path):
    if path.exists() and path.stat().st_size > 0:
        return
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)

def clean_prompt(text, transcript):
    text = text or ""
    match = VOICE_TEXT_RE.search(text)
    if match:
        return match.group(1).replace('\\"', '"').strip()
    return (transcript or text).strip()

seen = set()
written = 0
downloaded_files = 0
with out_path.open("w", encoding="utf-8", newline="\n") as out:
    for idx in range(file_count):
        if max_records and written >= max_records:
            break
        parquet_path = raw_root / f"{idx:04d}.parquet"
        try:
            download(base.format(idx=idx), parquet_path)
        except Exception as exc:
            print(json.dumps({"event": "download_failed", "idx": idx, "error": str(exc)}), flush=True)
            continue
        downloaded_files += 1
        table = pq.read_table(parquet_path)
        for row_index in range(table.num_rows):
            if max_records and written >= max_records:
                break
            row = {name: table[name][row_index].as_py() for name in table.column_names}
            audio = row.get("audio") if isinstance(row.get("audio"), dict) else {}
            data = audio.get("bytes")
            if not data:
                continue
            digest = hashlib.sha256(data).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            target = media_root / digest[:2] / f"{digest}.wav"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(data)
            prompt_text = clean_prompt(row.get("text"), row.get("whisper_transcription"))
            if len(prompt_text) < 20:
                continue
            emotion_tags = row.get("emotion_tags") or []
            if not isinstance(emotion_tags, list):
                emotion_tags = []
            artifact = {
                "path": str(target),
                "uri": str(target),
                "sha256": digest,
                "byte_size": len(data),
                "kind": "audio",
                "dataset": "laion_orpheus_tts",
                "source_parquet": str(parquet_path),
                "source_row": row_index,
                "probe": ffprobe(target),
            }
            record = {
                "schema": "omnicoder.real_multimodal_training_2026.v1",
                "source_dataset": "laion/laions_got_talent_with_voice_emotion_speed_tags_for_orpheus_tuning",
                "dataset_name": "LAION Orpheus TTS voice emotion speed tags",
                "license": "apache-2.0",
                "license_tier": "permissive",
                "modality": "audio",
                "declared_target_modality": "audio",
                "modalities": ["audio", "text"],
                "task_type": "tts_expressive_generation",
                "source_id": stable_hash({"dataset": "laion_orpheus", "sha256": digest}),
                "record_id": stable_hash({"dataset": "laion_orpheus", "sha256": digest, "row": row_index}),
                "prompt": (
                    "Generate expressive TTS audio with "
                    f"voice={row.get('voice') or 'unknown'}, speed={row.get('characters_per_second') or ''}, "
                    f"language={row.get('language_code') or ''}, emotion_tags={','.join(str(x) for x in emotion_tags[:8])}. "
                    f"Text: {prompt_text}"
                ),
                "target": prompt_text,
                "transcript": row.get("whisper_transcription") or prompt_text,
                "voice": row.get("voice"),
                "language_code": row.get("language_code"),
                "emotion": row.get("emotion"),
                "emotion_tags": emotion_tags,
                "characters_per_second": row.get("characters_per_second"),
                "duration": row.get("duration"),
                "artifact_refs": [artifact],
                "media_refs": [artifact],
                "quality": {"label": "expressive_tts_real_audio", "score": 0.84},
                "quality_score": 0.84,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            out.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
            written += 1
print(json.dumps({"status": "ok", "dataset": "laion_orpheus_tts", "downloaded_parquet_files": downloaded_files, "records": written, "out": str(out_path)}, ensure_ascii=True, sort_keys=True))
PY
}

extract_jamendo_music() {
  log "extracting JamendoMaxCaps CC music files=$JAMENDO_JSONL_FILES max=$JAMENDO_MAX_RECORDS"
  "$HF_PYTHON" - "$OUT_ROOT/raw/jamendo_music.raw.jsonl" "$MEDIA_ROOT/jamendo_music" "$JAMENDO_JSONL_FILES" "$JAMENDO_MAX_RECORDS" <<'PY' | tee "$OUT_ROOT/logs/jamendo_music_intake.log"
import datetime as dt
import hashlib
import json
import pathlib
import subprocess
import sys
import time

import requests

out_path = pathlib.Path(sys.argv[1])
media_root = pathlib.Path(sys.argv[2])
jsonl_file_count = max(0, int(sys.argv[3]))
max_records = max(0, int(sys.argv[4]))
raw_root = out_path.parent / "jamendo_jsonl"
raw_root.mkdir(parents=True, exist_ok=True)
media_root.mkdir(parents=True, exist_ok=True)

API = "https://huggingface.co/api/datasets/amaai-lab/JamendoMaxCaps/tree/main?recursive=1"
RESOLVE = "https://huggingface.co/datasets/amaai-lab/JamendoMaxCaps/resolve/main/{path}"
ALLOWED_LICENSE_MARKERS = (
    "creativecommons.org/licenses/by/",
    "creativecommons.org/licenses/by-sa/",
)

def stable_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")).hexdigest()

def ffprobe(path):
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration,bit_rate:stream=codec_name,sample_rate,channels",
                "-of", "json", str(path),
            ],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
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

def download(url, path, max_bytes=24 * 1024 * 1024):
    if path.exists() and path.stat().st_size > 0:
        return True
    with requests.get(url, stream=True, timeout=90, allow_redirects=True) as response:
        if response.status_code >= 400:
            return False
        total = 0
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(1024 * 256):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    tmp.unlink(missing_ok=True)
                    return False
                handle.write(chunk)
        if total <= 0:
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(path)
    return True

def license_allowed(value):
    value = (value or "").lower()
    if "nc" in value or "nd" in value:
        return False
    return any(marker in value for marker in ALLOWED_LICENSE_MARKERS)

def music_prompt(row):
    info = row.get("musicinfo") if isinstance(row.get("musicinfo"), dict) else {}
    tags = info.get("tags") if isinstance(info.get("tags"), dict) else {}
    tag_values = []
    for key in ("genres", "instruments", "vartags"):
        vals = tags.get(key)
        if isinstance(vals, list):
            tag_values.extend(str(item) for item in vals[:12] if item)
    descriptors = []
    for key in ("vocalinstrumental", "speed", "acousticelectric", "lang", "gender"):
        value = info.get(key)
        if value:
            descriptors.append(str(value))
    title = row.get("name") or "untitled"
    artist = row.get("artist_name") or "unknown artist"
    album = row.get("album_name") or ""
    tag_text = ", ".join(dict.fromkeys(descriptors + tag_values))
    if not tag_text:
        tag_text = "instrumental, music, licensed jamendo track"
    return (
        f"Generate or understand a {tag_text} music track titled '{title}' by {artist}"
        + (f" from album '{album}'." if album else ".")
        + f" Duration target: {row.get('duration') or 'unknown'} seconds."
    )

response = requests.get(API, timeout=60)
response.raise_for_status()
jsonl_paths = [item["path"] for item in response.json() if str(item.get("path", "")).endswith(".jsonl")]
jsonl_paths = sorted(jsonl_paths)[:jsonl_file_count]

written = 0
seen = set()
downloaded_jsonls = 0
with out_path.open("w", encoding="utf-8", newline="\n") as out:
    for rel in jsonl_paths:
        if max_records and written >= max_records:
            break
        local_jsonl = raw_root / rel
        local_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if not local_jsonl.exists():
            if not download(RESOLVE.format(path=rel), local_jsonl, max_bytes=16 * 1024 * 1024):
                continue
        downloaded_jsonls += 1
        with local_jsonl.open("r", encoding="utf-8", errors="ignore") as handle:
            for line_number, line in enumerate(handle, 1):
                if max_records and written >= max_records:
                    break
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not license_allowed(row.get("license_ccurl")):
                    continue
                audio_url = row.get("audio") or row.get("audiodownload")
                if not audio_url:
                    continue
                track_id = str(row.get("id") or stable_hash(row)[:16])
                digest_hint = stable_hash({"track_id": track_id, "audio_url": audio_url})
                target = media_root / digest_hint[:2] / f"jamendo_{track_id}_{digest_hint[:12]}.mp3"
                target.parent.mkdir(parents=True, exist_ok=True)
                if not download(audio_url, target):
                    continue
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                final_target = target.with_name(f"{digest}.mp3")
                if target != final_target:
                    if not final_target.exists():
                        target.replace(final_target)
                    else:
                        target.unlink(missing_ok=True)
                    target = final_target
                probe = ffprobe(target)
                artifact = {
                    "path": str(target),
                    "uri": str(target),
                    "sha256": digest,
                    "byte_size": target.stat().st_size,
                    "kind": "music",
                    "dataset": "JamendoMaxCaps",
                    "source_jsonl": rel,
                    "source_line": line_number,
                    "probe": probe,
                }
                record = {
                    "schema": "omnicoder.real_multimodal_training_2026.v1",
                    "source_dataset": "amaai-lab/JamendoMaxCaps",
                    "dataset_name": "JamendoMaxCaps",
                    "license": row.get("license_ccurl"),
                    "license_tier": "attribution_sharealike" if "by-sa" in (row.get("license_ccurl") or "").lower() else "attribution",
                    "modality": "music",
                    "declared_target_modality": "music",
                    "modalities": ["audio", "music", "text"],
                    "task_type": "music_generation_and_understanding",
                    "source_id": stable_hash({"dataset": "jamendo", "id": row.get("id")}),
                    "record_id": stable_hash({"dataset": "jamendo", "id": row.get("id"), "sha256": digest}),
                    "prompt": music_prompt(row),
                    "target": json.dumps({
                        "title": row.get("name"),
                        "artist": row.get("artist_name"),
                        "album": row.get("album_name"),
                        "duration": row.get("duration"),
                        "musicinfo": row.get("musicinfo"),
                        "license": row.get("license_ccurl"),
                    }, ensure_ascii=True, sort_keys=True),
                    "metadata": {
                        key: row.get(key)
                        for key in ("id", "name", "duration", "artist_name", "album_name", "releasedate", "shareurl", "shorturl", "license_ccurl", "musicinfo")
                    },
                    "artifact_refs": [artifact],
                    "media_refs": [artifact],
                    "quality": {"label": "licensed_music_real_audio", "score": 0.82},
                    "quality_score": 0.82,
                    "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
                out.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
                written += 1
print(json.dumps({"status": "ok", "dataset": "jamendo_maxcaps", "downloaded_jsonls": downloaded_jsonls, "records": written, "out": str(out_path)}, ensure_ascii=True, sort_keys=True))
PY
}

ensure_p40_comfy() {
  if curl -sS --max-time 8 "$P40_COMFY_URL/system_stats" >/dev/null 2>&1; then
    return 0
  fi
  log "starting P40 ComfyUI sidecar for additional ACE rollouts"
  (cd "$P40_COMFY_DIR" && docker compose -f docker-compose.p40.yml up -d comfyui_p40) | tee -a "$OUT_ROOT/logs/comfyui_p40.log"
  local attempt
  for attempt in $(seq 1 60); do
    if curl -sS --max-time 8 "$P40_COMFY_URL/system_stats" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

build_extra_ace_jobs() {
  log "building extra ACE-Step rollout jobs count=$ACE_EXTRA_JOBS"
  "$PYTHON_BIN" - "$OUT_ROOT/jobs/ace_music_extra_jobs.jsonl" "$ACE_EXTRA_JOBS" "$OUT_ROOT/jsonl/jamendo.clean.jsonl" "$BASE_DIR/jsonl/musicbench.clean.jsonl" <<'PY' | tee "$OUT_ROOT/logs/ace_extra_job_builder.log"
import hashlib
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
limit = max(0, int(sys.argv[2]))
sources = [pathlib.Path(item) for item in sys.argv[3:]]

def stable_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")).hexdigest()

def text_value(value, limit_chars=3000):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()[:limit_chars]
    if isinstance(value, dict):
        for key in ("prompt", "target", "caption", "main_caption", "alt_caption", "text"):
            text = text_value(value.get(key), limit_chars)
            if text:
                return text
    return str(value)[:limit_chars]

jobs = []
seen = set()
for source in sources:
    if len(jobs) >= limit:
        break
    if not source.exists():
        continue
    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        for idx, line in enumerate(handle, 1):
            if len(jobs) >= limit:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            prompt = text_value(row.get("prompt") or row, 1600)
            if len(prompt) < 30:
                continue
            tags = "ace step, licensed music, text-to-music, coherent arrangement, stereo, 48khz"
            if row.get("dataset_name"):
                tags += f", {row.get('dataset_name')}"
            job = {
                "endpoint_env": "COMFYUI_P40_BASE_URL",
                "input_json": {
                    "schema": "omnicoder.music_tts_ace_rollout_job_2026.v1",
                    "modality": "music",
                    "prompt": prompt,
                    "tags": tags[:900],
                    "lyrics": "",
                    "seconds": 8.0,
                    "bpm": 96,
                    "language": "en",
                    "keyscale": "A minor",
                    "source": {"path": str(source), "row_index": idx, "payload_hash": stable_hash(row)[:24]},
                    "training_targets": ["music_generation", "ace_step_distillation", "licensed_music_style_grounding", "artifact_token_prediction"],
                },
                "job_type": "music_plan",
                "priority": 92,
                "teacher_model_alias": "ace-step-1.5",
                "teacher_name": "ace_step_1_5",
                "teacher_provider": "comfyui_p40",
            }
            key = stable_hash(job["input_json"])
            if key in seen:
                continue
            seen.add(key)
            jobs.append(job)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="\n") as handle:
    for job in jobs:
        handle.write(json.dumps(job, ensure_ascii=True, sort_keys=True) + "\n")
print(json.dumps({"status": "ok", "jobs": len(jobs), "out": str(out)}, ensure_ascii=True, sort_keys=True))
PY
}

run_extra_ace_rollouts() {
  local jobs="$OUT_ROOT/jobs/ace_music_extra_jobs.jsonl"
  if [[ "$ACE_EXTRA_JOBS" == "0" || ! -s "$jobs" ]]; then
    log "extra ACE rollouts skipped"
    return 0
  fi
  if ! ensure_p40_comfy; then
    log "extra ACE rollouts skipped: P40 ComfyUI not healthy"
    return 0
  fi
  log "running extra ACE-Step live rollouts on P40: $ACE_EXTRA_JOBS"
  "$PYTHON_BIN" -m omnicoder.data_factory.media_teacher_rollouts_2026 \
    --input "$jobs" \
    --out-dir "$OUT_ROOT/rollouts" \
    --mode live \
    --limit "$ACE_EXTRA_JOBS" \
    --resume \
    --comfyui-url "$P40_COMFY_URL" \
    --artifact-root "/home/cereal/comfyui/output" \
    --timeout "${OMNICODER_MUSIC_TTS_ACE_ROLLOUT_TIMEOUT:-1800}" \
    | tee "$OUT_ROOT/logs/ace_extra_rollouts.log"
  curl -sS --max-time 10 -X POST "$P40_COMFY_URL/free" -H 'Content-Type: application/json' -d '{"unload_models":true,"free_memory":true}' >/dev/null 2>&1 || true
}

combine_outputs() {
  "$PYTHON_BIN" - "$OUT_ROOT" "$BASE_DIR" <<'PY'
import hashlib
import json
import pathlib
import shutil
import sys

root = pathlib.Path(sys.argv[1])
base = pathlib.Path(sys.argv[2])
jsonl = root / "jsonl"

families = []

def copy_lines(out_name, inputs):
    out_path = jsonl / out_name
    count = 0
    seen = set()
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for path in inputs:
            path = pathlib.Path(path)
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    key = hashlib.sha256(line.encode("utf-8", errors="ignore")).hexdigest()
                    if key in seen:
                        continue
                    seen.add(key)
                    out.write(line)
                    count += 1
    families.append({"name": out_name, "path": str(out_path), "records": count})
    return out_path, count

copy_lines("music.clean.jsonl", [base / "jsonl/music.clean.jsonl", jsonl / "jamendo.clean.jsonl"])
copy_lines("tts.clean.jsonl", [base / "jsonl/tts.clean.jsonl", jsonl / "laion_orpheus_tts.clean.jsonl"])
copy_lines("musicbench.clean.jsonl", [base / "jsonl/musicbench.clean.jsonl"])
copy_lines("ace_rollouts.clean.jsonl", [base / "jsonl/ace_rollouts.clean.jsonl", jsonl / "ace_extra_rollouts.clean.jsonl"])

combined = jsonl / "music_tts_ace_clean.jsonl"
with combined.open("w", encoding="utf-8", newline="\n") as out:
    for name in ("music.clean.jsonl", "musicbench.clean.jsonl", "tts.clean.jsonl", "ace_rollouts.clean.jsonl"):
        path = jsonl / name
        if path.exists():
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.strip():
                        out.write(line)

def count_lines(path):
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
    "base_dir": str(base),
    "out_root": str(root),
    "combined_jsonl": str(combined),
    "combined_records": count_lines(combined),
    "combined_sha256": digest.hexdigest(),
    "families": families,
    "supplemental_sources": [
        "laion/laions_got_talent_with_voice_emotion_speed_tags_for_orpheus_tuning",
        "amaai-lab/JamendoMaxCaps",
        "ACE-Step 1.5 live P40 rollouts",
    ],
}
(root / "music_tts_ace_manifest_index.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"status": "ok", "manifest": str(root / "music_tts_ace_manifest_index.json"), "combined_records": manifest["combined_records"]}, ensure_ascii=True, sort_keys=True))
PY
}

log "starting supplemental music/TTS expansion at $OUT_ROOT base=$BASE_DIR"
ensure_hf_python
extract_laion_orpheus_tts
run_curation_family laion_orpheus_tts audio 0.60 1 "$LAION_MAX_RECORDS" "$OUT_ROOT/raw/laion_orpheus_tts.raw.jsonl"
extract_jamendo_music
run_curation_family jamendo music 0.60 1 "$JAMENDO_MAX_RECORDS" "$OUT_ROOT/raw/jamendo_music.raw.jsonl"
build_extra_ace_jobs
run_extra_ace_rollouts
if [[ -s "$OUT_ROOT/rollouts/ace_music_rollouts.jsonl" ]]; then
  run_curation_family ace_extra_rollouts music 0.60 1 0 "$OUT_ROOT/rollouts/ace_music_rollouts.jsonl"
else
  log "extra ACE rollout curation skipped: no rollout rows"
fi
combine_outputs | tee "$OUT_ROOT/logs/combine.log"
log "supplemental music/TTS expansion complete"
