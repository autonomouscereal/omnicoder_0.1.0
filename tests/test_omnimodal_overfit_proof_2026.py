from __future__ import annotations

import json
from pathlib import Path

from omnicoder.eval import omnimodal_overfit_proof_2026 as proof


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _group_modalities(out: Path, group: str) -> list[str]:
    rows = _read_jsonl(out / "data" / f"{group}.jsonl")
    return sorted({str(row.get("modality") or "unknown") for row in rows})


def _write_eval_artifacts(
    out: Path,
    *,
    groups: tuple[str, ...] = proof.PROOF_GROUPS,
    high_group: str | None = None,
    high_modality: str = "tool",
) -> None:
    eval_dir = out / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    for group in groups:
        modalities = {name: {"tokens": 4, "loss": 0.01, "loss_sum": 0.04} for name in _group_modalities(out, group)}
        if group == high_group and high_modality:
            modalities[high_modality] = {"tokens": 4, "loss": 0.2, "loss_sum": 0.8}
        total_tokens = sum(int(bucket["tokens"]) for bucket in modalities.values())
        loss_sum = sum(float(bucket["loss_sum"]) for bucket in modalities.values())
        loss = loss_sum / max(1, total_tokens)
        payload = {
            "schema": "omnicoder.pipeline_sample_loss_2026.v1",
            "overall": {"tokens": total_tokens, "loss": loss, "loss_sum": loss_sum},
            "modalities": modalities,
        }
        target_tokens = {name: {"records": 1, "target_tokens": 4} for name in modalities}
        (eval_dir / f"{group}.loss.json").write_text(json.dumps(payload), encoding="utf-8")
        (eval_dir / f"{group}.targets.json").write_text(
            json.dumps({"target_tokens": sum(bucket["target_tokens"] for bucket in target_tokens.values()), "modalities": target_tokens}),
            encoding="utf-8",
        )


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
                    "modalities": {name: {"records": 1, "target_tokens": 4} for name in _group_modalities(out, group)},
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
    assert manifest["shared_checkpoint_group"] == "omni_all"
    assert {item["group"] for item in manifest["groups"]} == set(proof.PROOF_GROUPS)
    for group in proof.PROOF_GROUPS:
        rows = _read_jsonl(out / "data" / f"{group}.jsonl")
        assert len(rows) == (60 if group == "omni_all" else 10)
        assert all(row["target_token_ids"] for row in rows)
        assert all(row["contamination_status"] == "clean" for row in rows)
        assert all(row["quality_score"] == 1.0 for row in rows)
        for row in rows:
            if row.get("artifact_token_ids"):
                assert row["artifact_token_ids"] == row["target_token_ids"]
                assert row["valid_target_tokens"] == len(row["target_token_ids"])
    ledger_rows = _read_jsonl(out / "data" / "ledger_all.jsonl")
    assert {"vision_semantic", "vision_residual", "speech_tts", "audio_music", "music_control", "time_space", "tool_agent"}.issubset(
        {row["target_ledger_family"] for row in ledger_rows}
    )
    by_family = {row["target_ledger_family"]: row["modality"] for row in ledger_rows}
    assert by_family["speech_tts"] == "tts"
    assert by_family["audio_music"] == "audio"
    assert by_family["music_control"] == "music"
    assert by_family["time_space"] == "video"
    assert by_family["tool_agent"] == "tool"
    assert by_family["flow"] == "flow"
    omni_rows = _read_jsonl(out / "data" / "omni_all.jsonl")
    assert {"text", "code_tool", "image_ocr", "video", "audio_tts_music", "ledger_all"} == {row["origin_group"] for row in omni_rows}


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
    assert any(failure["bucket"] == "modality:tool" for failure in group["sample_loss_failures"])


def test_summary_can_scope_to_selected_modality_group(tmp_path: Path) -> None:
    out = tmp_path / "proof"
    proof.main(["materialize", "--out", str(out), "--examples-per-modality", "2"])
    _write_eval_artifacts(out, groups=("video",))

    result = proof.summary(type("Args", (), {"run": str(out), "out": "", "groups": "video", "max_reload_sample_loss": 0.05})())

    assert result["status"] == "passed"
    summary = json.loads((out / "omnimodal_overfit_summary.json").read_text(encoding="utf-8"))
    assert list(summary["groups"]) == ["video"]


def test_summary_fails_when_eval_omits_expected_modality(tmp_path: Path) -> None:
    out = tmp_path / "proof"
    proof.main(["materialize", "--out", str(out), "--examples-per-modality", "2"])
    _write_eval_artifacts(out, groups=("code_tool",))
    eval_dir = out / "eval"
    (eval_dir / "code_tool.loss.json").write_text(
        json.dumps(
            {
                "schema": "omnicoder.pipeline_sample_loss_2026.v1",
                "overall": {"tokens": 4, "loss": 0.01, "loss_sum": 0.04},
                "modalities": {"code": {"tokens": 4, "loss": 0.01, "loss_sum": 0.04}},
            }
        ),
        encoding="utf-8",
    )

    result = proof.summary(type("Args", (), {"run": str(out), "out": "", "groups": "code_tool", "max_reload_sample_loss": 0.05})())

    assert result["status"] == "failed"
    summary = json.loads((out / "omnimodal_overfit_summary.json").read_text(encoding="utf-8"))
    assert summary["groups"]["code_tool"]["missing_sample_loss_modalities"] == ["tool"]


def test_train_plan_uses_manifest_row_counts_for_shared_all_modality_group(tmp_path: Path) -> None:
    out = tmp_path / "proof"
    proof.main(["materialize", "--out", str(out), "--examples-per-modality", "10"])

    plan = proof.train_plan(type("Args", (), {"run": str(out), "out": "", "groups": ""})())

    assert "--max_records 10" in plan["commands"][0]
    assert "--max_records 60" in plan["commands"][-1]
