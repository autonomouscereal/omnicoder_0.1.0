from __future__ import annotations

from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ai_server_run_post_checkpoint_eval_20b.sh"


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_post_checkpoint_eval_wires_target_token_diagnostics_contract() -> None:
    text = _script_text()
    start = text.index("target_diagnostics_common=(")
    end = text.index("docker_eval heldout_sample_loss")
    block = text[start:end]

    assert "-m omnicoder.eval.pipeline_target_token_diagnostics_2026" in block
    assert "--out \"/workspace/$OUT_DIR/heldout_target_token_diagnostics.local_regression.json\"" in text
    assert "--require-target-contract" in block
    assert "--allow-p40-target-contract-eval" in block


def test_post_checkpoint_eval_wires_decode_sanity_artifacts_and_release_gate() -> None:
    text = _script_text()
    prediction_start = text.index("docker_eval decode_sanity_predictions")
    gate_start = text.index("docker_eval decode_sanity_release_gate")
    public_dev_start = text.index("public_dev_roots_present=()")
    prediction_block = text[prediction_start:gate_start]
    gate_block = text[gate_start:public_dev_start]

    assert "decode_sanity_tasks.local_regression.jsonl" in text
    assert "decode_sanity_predictions.local_regression.jsonl" in prediction_block
    assert "decode_sanity_prediction_summary.local_regression.json" in prediction_block
    for modality in ("text", "code", "math", "tool", "image", "video", "audio", "music", "tts", "ocr"):
        assert f'"modality":"{modality}"' in text

    assert "-m omnicoder.eval.pipeline_checkpoint_batch_predict_2026" in prediction_block
    assert "--require-target-contract" in prediction_block
    assert "--allow-p40-target-contract-eval" in prediction_block

    assert "-m omnicoder.eval.omnimodal_release_gate_2026" in gate_block
    assert "--predictions \"/workspace/$OUT_DIR/decode_sanity_predictions.local_regression.jsonl\"" in gate_block
    assert "--out \"/workspace/$OUT_DIR/decode_sanity_release_gate.local_regression.json\"" in gate_block
    assert "--require-modalities \"$DECODE_SANITY_RELEASE_GATE_MODALITIES\"" in gate_block
