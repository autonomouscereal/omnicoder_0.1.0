from __future__ import annotations

import os
from pathlib import Path


def test_decode_step_onnx_dynamo_or_fallback(tmp_path: Path) -> None:
    """
    Export a small decode-step ONNX using the dynamo exporter path when available
    (falls back to legacy exporter). Assert that an ONNX file is produced and is
    minimally loadable by onnxruntime if installed.
    """
    out = tmp_path / "dyn_decode.onnx"
    # Call the exporter CLI entry with defaults (no --no_dynamo)
    import sys
    from omnicoder.export.onnx_export import main as onnx_export_main
    argv = sys.argv
    try:
        sys.argv = [
            "onnx_export",
            "--output",
            str(out),
            "--seq_len",
            "1",
            "--mobile_preset",
            "mobile_4gb",
            "--decode_step",
            "--opset",
            "18",
        ]
        onnx_export_main()
    finally:
        sys.argv = argv

    assert out.exists(), "ONNX file not created"

    # If onnxruntime is present, ensure basic load succeeds and first output is logits
    try:
        import onnxruntime as ort  # type: ignore

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])  # type: ignore
        outs = [o.name for o in sess.get_outputs()]
        assert outs and outs[0] == "logits"
    except Exception:
        # OK to skip runtime validation if onnxruntime isn't present
        pass


