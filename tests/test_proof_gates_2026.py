from __future__ import annotations

import json
from pathlib import Path

from omnicoder.eval.proof_gates_2026 import DEFAULT_REQUIRED_MODALITIES, build_proof_gate, main


def _target_payload(tokens: int = 8) -> dict:
    counts = {modality: tokens for modality in DEFAULT_REQUIRED_MODALITIES}
    return {
        "schema": "omnicoder.lm_loss_diagnostics_2026.v1",
        "status": "ok",
        "target_counts_by_modality": counts,
        "optimized_target_counts_by_modality": counts,
    }


def _heldout_payload(loss: float = 2.0, tokens: int = 16) -> dict:
    return {
        "schema": "omnicoder.pipeline_sample_loss_2026.v1",
        "status": "passed",
        "modalities": {
            modality: {"avg_loss": loss, "tokens": tokens}
            for modality in DEFAULT_REQUIRED_MODALITIES
        },
    }


def _release_gate_payload() -> dict:
    return {
        "schema": "omnicoder.omnimodal_release_gate_2026.v1",
        "status": "passed",
        "accepted_modalities": ["text", "code", "tool", "ocr", "image", "video", "audio", "music", "tts"],
    }


def _profile_payload() -> dict:
    names = ["fakequant_chunk2048_loss64", "reasoning_effort2_q4_chunk2048_loss64", "reasoning_efforthigh_q4_chunk2048_loss64"]
    q4_env = {
        "OMNICODER_FAKE_QUANT": "1",
        "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "8192",
        "OMNICODER_FFN_CHUNK_TOKENS": "1024",
        "OMNICODER_PLACEMENT_LAYER_COUNTS": "16,16,32",
        "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
        "OMNICODER_PIPELINE_MICROBATCHES": "1",
        "OMNICODER_BATCH_SIZE": "1",
    }
    q4_variants = [
        {
            "variant": "ffn_chunk1024_headroom_q4_chunk8192_loss64",
            "status": "passed",
            "last_seq_len": seq_len,
            "last_target_token_coverage": 1.0,
            "sequence_tokens_per_sec": 6.1,
            "no_checkpoint_requested": True,
            "no_checkpoint_written": True,
            "container_state": {"exit_code": 0, "oom_killed": False},
            "requested_env": q4_env,
        }
        for seq_len in (1024, 2048)
    ]
    return {
        "schema": "omnicoder.profile_matrix_20b_2026.v1",
        "status": "passed",
        "variants": q4_variants
        + [
            {
                "variant": name,
                "status": "passed",
                "last_seq_len": 1024,
                "last_target_token_coverage": 1.0,
                "sequence_tokens_per_sec": 6.1,
                "no_checkpoint_requested": True,
                "no_checkpoint_written": True,
                "container_state": {"exit_code": 0, "oom_killed": False},
            }
            for name in names
        ],
    }


def _coverage_payload() -> dict:
    return {"status": "passed", "modalities": {modality: {"records": 16} for modality in DEFAULT_REQUIRED_MODALITIES}}


def _contract_payload() -> dict:
    return {
        "status": "passed",
        "checks": {
            "n_layers": {"actual": 64},
            "d_model": {"actual": 4096},
            "mlp_dim": {"actual": 15360},
            "mtp_heads": {"actual": 2},
            "vocab_size": {"actual": 330000},
        },
        "target_context_length": 1048576,
        "residual_mode": "block_attnres",
    }


def _context_ladder_payload() -> dict:
    return {
        "status": "passed",
        "rungs": [
            {"seq_len": rung, "loss": 2.0, "recall_passed": True, "status": "passed"}
            for rung in (8192, 32768, 131072, 262144, 524288, 1048576)
        ],
    }


def _ready_report() -> dict:
    return build_proof_gate(
        target_payloads=[_target_payload()],
        heldout_payloads=[_heldout_payload()],
        release_gate_payloads=[_release_gate_payload()],
        q4_profile_payloads=[_profile_payload()],
        reasoning_profile_payloads=[_profile_payload()],
        coverage_payloads=[_coverage_payload()],
        reportable_payloads=[{"status": "passed", "official": 1}],
        gguf_payloads=[{"status": "passed", "manifest_only": False, "artifact": "model.gguf", "tokens_per_second": 5.0, "peak_vram_gib": 22.0}],
        contract_payloads=[_contract_payload()],
        context_ladder_payloads=[_context_ladder_payload()],
    )


def test_pre_full_training_proof_gate_passes_only_with_all_contract_evidence() -> None:
    report = _ready_report()

    assert report["status"] == "ready"
    assert report["ready_for_full_training"] is True
    assert report["blockers"] == []
    assert report["checks"]["q4_profile"]["status"] == "passed"
    assert report["checks"]["reasoning_profile"]["status"] == "passed"


def test_pre_full_training_proof_gate_blocks_missing_media_and_manifest_only_gguf() -> None:
    release = _release_gate_payload()
    release["accepted_modalities"] = ["text", "code", "tool", "ocr", "image"]
    report = build_proof_gate(
        target_payloads=[_target_payload()],
        heldout_payloads=[_heldout_payload()],
        release_gate_payloads=[release],
        q4_profile_payloads=[_profile_payload()],
        reasoning_profile_payloads=[_profile_payload()],
        coverage_payloads=[_coverage_payload()],
        reportable_payloads=[{"status": "passed", "official": 1}],
        gguf_payloads=[{"status": "passed", "manifest_only": True}],
        contract_payloads=[_contract_payload()],
        context_ladder_payloads=[_context_ladder_payload()],
    )

    assert report["status"] == "blocked"
    assert any("decode_and_media_release_gate:release_gate_missing_video" == item for item in report["blockers"])
    assert "gguf_runtime:gguf_runtime_manifest_only" in report["blockers"]


def test_pre_full_training_proof_gate_cli_writes_fail_closed_report(tmp_path: Path, capsys) -> None:
    target = tmp_path / "target.json"
    target.write_text(json.dumps(_target_payload()) + "\n", encoding="utf-8")
    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps(_heldout_payload()) + "\n", encoding="utf-8")
    release = tmp_path / "release.json"
    release.write_text(json.dumps(_release_gate_payload()) + "\n", encoding="utf-8")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(_profile_payload()) + "\n", encoding="utf-8")
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(_contract_payload()) + "\n", encoding="utf-8")
    context = tmp_path / "context.json"
    context.write_text(json.dumps(_context_ladder_payload()) + "\n", encoding="utf-8")
    out = tmp_path / "pre_full_training_proof_gate.json"

    code = main(
        [
            "--target-diagnostics",
            str(target),
            "--heldout-sample-loss",
            str(heldout),
            "--decode-release-gate",
            str(release),
            "--q4-profile-summary",
            str(profile),
            "--reasoning-profile-summary",
            str(profile),
            "--contract-report",
            str(contract),
            "--context-ladder-proof",
            str(context),
            "--coverage-report",
            str(tmp_path / "missing_coverage.json"),
            "--reportable-summary",
            str(tmp_path / "missing_reportable.json"),
            "--gguf-runtime-proof",
            str(tmp_path / "missing_gguf.json"),
            "--out",
            str(out),
        ]
    )

    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert out.exists()


def test_pre_full_training_proof_gate_rejects_hollow_status_reports() -> None:
    report = build_proof_gate(
        target_payloads=[_target_payload()],
        heldout_payloads=[_heldout_payload()],
        release_gate_payloads=[_release_gate_payload()],
        q4_profile_payloads=[_profile_payload()],
        reasoning_profile_payloads=[_profile_payload()],
        coverage_payloads=[{"status": "passed"}],
        reportable_payloads=[{"status": "passed"}],
        gguf_payloads=[{"status": "passed", "artifact": "model.gguf"}],
        contract_payloads=[_contract_payload()],
        context_ladder_payloads=[_context_ladder_payload()],
    )

    assert report["status"] == "blocked"
    assert "data_coverage:coverage_missing_coverage_evidence" in report["blockers"]
    assert "reportable_scores:reportable_scores_missing_official_scores" in report["blockers"]
    assert "gguf_runtime:gguf_runtime_missing_runtime_evidence" in report["blockers"]


def test_pre_full_training_proof_gate_accepts_coverage_validator_counts() -> None:
    coverage_report = {
        "schema": "omnicoder.dataset_coverage_validator_2026.v1",
        "status": "passed",
        "counts": {
            "curated_train_files": {"train_text.jsonl": 8, "train_image.jsonl": 8},
            "qwen36_agentic_math_code_tool_rollouts": 16,
        },
    }
    report = build_proof_gate(
        target_payloads=[_target_payload()],
        heldout_payloads=[_heldout_payload()],
        release_gate_payloads=[_release_gate_payload()],
        q4_profile_payloads=[_profile_payload()],
        reasoning_profile_payloads=[_profile_payload()],
        coverage_payloads=[coverage_report],
        reportable_payloads=[{"status": "passed", "official": 1}],
        gguf_payloads=[{"status": "passed", "manifest_only": False, "artifact": "model.gguf", "tokens_per_second": 5.0, "peak_vram_gib": 22.0}],
        contract_payloads=[_contract_payload()],
        context_ladder_payloads=[_context_ladder_payload()],
    )

    assert report["checks"]["data_coverage"]["status"] == "passed"
