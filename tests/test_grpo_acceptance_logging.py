from __future__ import annotations

import json
from pathlib import Path


def test_grpo_logs_acceptance(tmp_path: Path):
    # Create a tiny prompts jsonl
    data = tmp_path / "prompts.jsonl"
    data.write_text('{"prompt": "Hello", "targets": ["Hello"]}\n')
    out_log = tmp_path / "grpo_log.jsonl"

    # Run a few GRPO steps on CPU
    import sys
    from omnicoder.training.rl_grpo import main as grpo_main
    argv = sys.argv
    try:
        sys.argv = [
            "grpo",
            "--prompts", str(data),
            "--device", "cpu",
            "--steps", "2",
            "--group_size", "2",
            "--max_new_tokens", "8",
            "--lr", "1e-6",
            "--log_file", str(out_log),
        ]
        grpo_main()
    finally:
        sys.argv = argv

    assert out_log.exists()
    lines = [l for l in out_log.read_text().splitlines() if l.strip()]
    assert len(lines) >= 2
    rec = json.loads(lines[-1])
    assert "accept_rate" in rec and "avg_len" in rec and "loss" in rec


