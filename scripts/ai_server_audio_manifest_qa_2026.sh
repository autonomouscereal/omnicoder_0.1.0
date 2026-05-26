#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-}"
OUT="${OUT:-}"
MAX="${MAX:-200}"
if [[ -z "$ROOT" ]]; then
  echo "ROOT is required" >&2
  exit 2
fi
if [[ -z "$OUT" ]]; then
  OUT="/tmp/omnicoder_audio_qa_$(date -u +%Y%m%dT%H%M%SZ).jsonl"
fi
mkdir -p "$(dirname "$OUT")"

LIST_FILE="$(mktemp)"
trap 'rm -f "$LIST_FILE"' EXIT
find "$ROOT" -type f \( -iname '*.wav' -o -iname '*.flac' -o -iname '*.mp3' -o -iname '*.ogg' -o -iname '*.m4a' \) | sort > "$LIST_FILE.all"
if [[ "$MAX" == "0" ]]; then
  mv "$LIST_FILE.all" "$LIST_FILE"
else
  head -n "$MAX" "$LIST_FILE.all" > "$LIST_FILE"
  rm -f "$LIST_FILE.all"
fi

while IFS= read -r file_path; do
      kind="audio"
      case "$file_path" in
        */music/*) kind="music" ;;
        */tts/*|*/laion_orpheus_tts/*) kind="tts" ;;
        */jamendo_music/*) kind="music" ;;
      esac

      probe="$(ffprobe -v error -select_streams a:0 \
        -show_entries stream=codec_name,sample_rate,channels,bits_per_sample,bit_rate \
        -show_entries format=duration,size,bit_rate \
        -of json "$file_path" < /dev/null || true)"

      vol="$(nice -n 19 ionice -c3 ffmpeg -hide_banner -nostats -threads 1 -i "$file_path" \
        -af volumedetect -f null - < /dev/null 2>&1 || true)"

      sil="$(nice -n 19 ionice -c3 ffmpeg -hide_banner -nostats -threads 1 -i "$file_path" \
        -af silencedetect=n=-50dB:d=0.30 -f null - < /dev/null 2>&1 || true)"

      python3 - "$file_path" "$kind" "$probe" "$vol" "$sil" <<'PY'
import json
import re
import sys

path, kind, probe_s, vol_s, sil_s = sys.argv[1:6]
try:
    probe = json.loads(probe_s)
except Exception:
    probe = {}
stream = (probe.get("streams") or [{}])[0]
fmt = probe.get("format") or {}
duration = float(fmt.get("duration") or 0)
size = int(float(fmt.get("size") or 0))
sample_rate = int(stream.get("sample_rate") or 0)
channels = int(stream.get("channels") or 0)

def grab(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None

mean_db = grab(r"mean_volume:\s*(-?\d+(?:\.\d+)?) dB", vol_s)
max_db = grab(r"max_volume:\s*(-?\d+(?:\.\d+)?) dB", vol_s)
silence_total = sum(float(value) for value in re.findall(r"silence_duration:\s*(\d+(?:\.\d+)?)", sil_s))
silence_ratio = silence_total / duration if duration > 0 else 1.0

flags: list[str] = []
if duration <= 0 or size < 4096:
    flags.append("decode_or_size_fail")
if kind == "music" and not (5 <= duration <= 900):
    flags.append("music_duration_outlier")
if kind == "tts" and not (0.25 <= duration <= 120):
    flags.append("tts_duration_outlier")
if kind == "music" and sample_rate not in (44100, 48000):
    flags.append("music_sample_rate_unexpected")
if kind == "tts" and sample_rate not in (16000, 22050, 24000, 44100, 48000):
    flags.append("tts_sample_rate_unexpected")
if channels < 1:
    flags.append("no_audio_channels")
if silence_ratio > 0.25:
    flags.append("too_much_silence")
if max_db is not None and max_db >= -0.1:
    flags.append("clip_risk")
if mean_db is not None and (mean_db < -45 or mean_db > -6):
    flags.append("mean_loudness_outlier")

score = max(0.0, 1.0 - 0.15 * len(flags))
print(json.dumps({
    "path": path,
    "kind": kind,
    "duration": duration,
    "size": size,
    "sample_rate": sample_rate,
    "channels": channels,
    "codec": stream.get("codec_name"),
    "mean_db": mean_db,
    "max_db": max_db,
    "silence_total": silence_total,
    "silence_ratio": round(silence_ratio, 6),
    "flags": flags,
    "qa_score": round(score, 3),
}, ensure_ascii=True, sort_keys=True))
PY
done < "$LIST_FILE" > "$OUT"

python3 - "$OUT" <<'PY'
import collections
import json
import pathlib
import statistics
import sys

path = pathlib.Path(sys.argv[1])
rows = [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]
flags = collections.Counter(flag for row in rows for flag in row.get("flags", []))
by_kind = collections.Counter(row.get("kind") for row in rows)
scores = [float(row.get("qa_score") or 0.0) for row in rows]
summary = {
    "schema": "omnicoder.audio_manifest_qa_2026.v1",
    "rows": len(rows),
    "by_kind": dict(sorted(by_kind.items())),
    "flags": dict(sorted(flags.items())),
    "score_mean": round(statistics.mean(scores), 4) if scores else None,
    "score_min": min(scores) if scores else None,
    "out": str(path),
}
summary_path = path.with_suffix(path.suffix + ".summary.json")
summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
PY
