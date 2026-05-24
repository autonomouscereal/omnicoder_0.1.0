#!/usr/bin/env bash
set -euo pipefail

REPO="${OMNICODER_REPO:-/home/cereal/omnicoder_2026_work}"
NAME_FILTER="${OMNICODER_NAME_FILTER:-omnicoder_target20b_fast_}"
if [[ -n "${OMNICODER_CONTAINER_NAME:-}" ]]; then
  NAME_FILTER="$OMNICODER_CONTAINER_NAME"
fi

cd "$REPO"

container="$(docker ps -a --filter "name=${NAME_FILTER}" --format '{{.Names}}' | head -1 || true)"
echo "=== target container ==="
if [[ -z "$container" ]]; then
  echo "no container matched name filter: $NAME_FILTER"
else
  docker inspect -f '{{.Name}} status={{.State.Status}} running={{.State.Running}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} ipc={{.HostConfig.IpcMode}} started={{.State.StartedAt}} finished={{.State.FinishedAt}}' "$container"
  echo "--- logs tail ---"
  docker logs --tail "${OMNICODER_LOG_TAIL:-120}" "$container" 2>&1 || true
fi

echo "=== latest fast run artifacts ==="
python3 - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

root_env = os.environ.get("OMNICODER_OUT_DIR", "").strip()
if root_env:
    roots = [Path(root_env)]
else:
    roots = sorted(
        Path("weights/training_orchestration_2026").glob("target20b_fast_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not roots:
        roots = sorted(
            Path("weights/training_orchestration_2026").glob("target20b_pipeline_*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
if not roots:
    print("no fast target run roots found")
    raise SystemExit(0)

root = roots[0]
print("run_root", root)
for stage in ["01_text", "02_code", "03_tool", "04_image", "05_video", "06_audio", "07_music", "08_long_context"]:
    loss_path = root / "logs" / f"{stage}_loss.jsonl"
    rows = []
    if loss_path.exists():
        for line in loss_path.read_text(errors="ignore").splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        continue
    vals = [float(row["loss"]) for row in rows if row.get("loss") is not None]
    if vals:
        print(stage, "points", len(vals), "first", vals[0], "last", vals[-1], "best", min(vals), "last_step", rows[-1].get("step"), "local_step", rows[-1].get("local_step"))
        print(stage, "last5", json.dumps(rows[-5:], ensure_ascii=True)[:2500])

for path in sorted((root / "checkpoints").glob("*")) if (root / "checkpoints").exists() else []:
    marker = path / ".complete.json" if path.is_dir() else Path(str(path) + ".complete.json")
    print("checkpoint", path.name, "complete", marker.exists())

summary = root / "real_training_summary.json"
if summary.exists():
    try:
        payload = json.loads(summary.read_text(errors="ignore"))
        print("summary", json.dumps({"status": payload.get("status"), "training": payload.get("training", {}).get("status"), "final_checkpoint": payload.get("training", {}).get("final_checkpoint")}, indent=2))
    except Exception as exc:
        print("summary_json_error", exc)
PY

echo "=== gpu ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,pstate --format=csv,noheader,nounits || true
echo "=== gpu processes ==="
nvidia-smi pmon -c 1 || true
