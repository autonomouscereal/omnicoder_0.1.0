from __future__ import annotations

import json
from pathlib import Path

from omnicoder.eval import omnimodal_overfit_proof_2026 as proof


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_materialize_overfit_proof_covers_all_token_families(tmp_path: Path) -> None:
    out = tmp_path / "proof"

    code = proof.main(["materialize", "--out", str(out), "--examples-per-modality", "10"])

    assert code == 0
    manifest = json.loads((out / "omnimodal_overfit_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == proof.SCHEMA
    assert {item["group"] for item in manifest["groups"]} == set(proof.PROOF_GROUPS)
    for group in proof.PROOF_GROUPS:
        rows = _read_jsonl(out / "data" / f"{group}.jsonl")
        assert len(rows) == 10
        assert all(row["target_token_ids"] for row in rows)
        assert all(row["contamination_status"] == "clean" for row in rows)
        assert all(row["quality_score"] == 1.0 for row in rows)
    ledger_rows = _read_jsonl(out / "data" / "ledger_all.jsonl")
    assert {"vision_semantic", "vision_residual", "speech_tts", "audio_music", "music_control", "time_space", "tool_agent"}.issubset(
        {row["target_ledger_family"] for row in ledger_rows}
    )


def test_summary_fails_without_eval_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "proof"
    proof.main(["materialize", "--out", str(out), "--examples-per-modality", "2"])

    result = proof.summary(type("Args", (), {"run": str(out), "out": ""})())

    assert result["status"] == "failed"
    summary = json.loads((out / "omnimodal_overfit_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
