import tempfile
from pathlib import Path

import numpy as np

from omnicoder.export.onnx_export import main as onnx_export_main


def test_decode_step_export_has_no_verifier_outputs(tmp_path: Path) -> None:
    # Export a small decode-step ONNX graph
    out_dir = tmp_path / "text"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(out_dir / "decode_step.onnx")

    import sys
    import time
    argv = sys.argv
    try:
        sys.argv = [
            "onnx_export",
            "--output",
            model_path,
            "--seq_len",
            "1",
            "--mobile_preset",
            "mobile_4gb",
            "--decode_step",
            "--no_dynamo",
        ]
        t0 = time.perf_counter()
        onnx_export_main()
        t1 = time.perf_counter()
        print({'event': 'timing', 'name': 'decode_step_export', 'dt': float(t1 - t0)})
    finally:
        sys.argv = argv

    # Load ONNX and ensure output names do not contain a verifier head
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])  # type: ignore
    outputs = [o.name for o in sess.get_outputs()]
    # Expected outputs: logits, nk_lat_i..., nv_lat_i..., (optional) mtp_logits_* only
    for name in outputs:
        assert "verifier" not in name


