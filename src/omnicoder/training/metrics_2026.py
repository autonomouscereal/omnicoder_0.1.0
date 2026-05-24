from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


JSON_RE = re.compile(r"\{.*\}")


def iter_json_events(path: str | Path) -> Iterable[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text:
            continue
        candidates = [text]
        match = JSON_RE.search(text)
        if match and match.group(0) != text:
            candidates.append(match.group(0))
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except Exception:
                continue
            if isinstance(payload, dict):
                yield payload
                break


def summarize_training_log(path: str | Path) -> dict[str, Any]:
    losses: list[float] = []
    steps: list[int] = []
    events = 0
    last: dict[str, Any] | None = None
    for payload in iter_json_events(path):
        events += 1
        last = payload
        if payload.get("loss") is not None:
            losses.append(float(payload["loss"]))
        if payload.get("step") is not None:
            steps.append(int(payload["step"]))
    summary = {
        "path": str(path),
        "json_events": events,
        "steps": max(steps) if steps else 0,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
        "last_event": last,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse Omnicoder 2026 training JSON logs")
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    summary = summarize_training_log(args.log)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
