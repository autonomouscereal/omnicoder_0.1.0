from __future__ import annotations

import json
from pathlib import Path

from omnicoder.eval import omnimodal_overfit_proof_2026 as proof


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_eval_artifacts(out: Path, *, high_group: str | None = None, high_modality: str = "tool") -> None:
    eval_dir = out / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    for group in proof.PROOF_GROUPS:
        loss = 0.2 if group == high_group and not high_modality else 0.01
        modalities = {"text": {"tokens": 4, "loss": 0.01, "loss_sum": 0.04}}
        if group == high_group and high_modality:
            modalities[high_modality] = {"tokens": 4, "loss": 0.2, "loss_sum": 0.8}
        payload = {
            "schema": "omnicoder.pipeline_sample_loss_2026.v1",
            "overall": {"tokens": 8, "loss": loss, "loss_sum": loss * 8},
            "modalities": modalities,
        }
        (eval_dir / f"{group}.loss.json").write_text(json.dumps(payload), encoding="utf-8")
        (eval_dir / f"{group}.targets.json").write_text(json.dumps({"target_tokens": 8}), encoding="utf-8")


def _write_eval_artifacts_with_pipeline_target_schema(out: Path) -> None:
    _write_eval_artifacts(out)
    eval_dir = out / "eval"
    for group in proof.PROOF_GROUPS:
        (eval_dir / f"{group}.targets.json").write_text(
            json.dumps(
                {
                    "schema": "omnicoder.pipeline_target_token_diagnostics_2026.v1",
                    "status": "ok",
                    "overall": {"records": 2, "target_tokens": 8},
                }
            ),
            encoding="utf-8",
        )


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


def test_summary_passes_with_low_reload_sample_loss(tmp_path: Path) -> None:
    out = tmp_path / "proof"
    proof.main(["materialize", "--out", str(out), "--examples-per-modality", "2"])
    _write_eval_artifacts(out)

    result = proof.summary(type("Args", (), {"run": str(out), "out": "", "max_reload_sample_loss": 0.05})())

    assert result["status"] == "passed"
    summary = json.loads((out / "omnimodal_overfit_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"


def test_summary_accepts_pipeline_target_diagnostics_schema(tmp_path: Path) -> None:
    out = tmp_path / "proof"
    proof.main(["materialize", "--out", str(out), "--examples-per-modality", "2"])
    _write_eval_artifacts_with_pipeline_target_schema(out)

    result = proof.summary(type("Args", (), {"run": str(out), "out": "", "max_reload_sample_loss": 0.05})())

    assert result["status"] == "passed"
    summary = json.loads((out / "omnimodal_overfit_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"


def test_summary_fails_on_high_reload_sample_loss(tmp_path: Path) -> None:
    out = tmp_path / "proof"
    proof.main(["materialize", "--out", str(out), "--examples-per-modality", "2"])
    _write_eval_artifacts(out, high_group="code_tool", high_modality="tool")

    result = proof.summary(type("Args", (), {"run": str(out), "out": "", "max_reload_sample_loss": 0.05})())

    assert result["status"] == "failed"
    summary = json.loads((out / "omnimodal_overfit_summary.json").read_text(encoding="utf-8"))
    group = summary["groups"]["code_tool"]
    assert "high_sample_loss" in group["failures"]
    assert group["sample_loss_failures"][0]["bucket"] == "modality:tool"
