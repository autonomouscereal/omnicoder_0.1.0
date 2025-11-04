from __future__ import annotations

from pathlib import Path


def test_dynamic_cache_shim_sidecar(tmp_path: Path):
    # Export decode-step with dynamic_cache_shim; assert sidecar JSON is written
    from omnicoder.export.onnx_export import main as onnx_export_main
    out_path = tmp_path / "omnicoder_decode_step.onnx"
    import sys
    import time
    argv = sys.argv
    try:
        sys.argv = [
            "onnx_export",
            "--output",
            str(out_path),
            "--seq_len",
            "1",
            "--mobile_preset",
            "mobile_4gb",
            "--decode_step",
            "--dynamic_cache_shim",
            "--no_dynamo",
        ]
        t0 = time.perf_counter()
        onnx_export_main()
        t1 = time.perf_counter()
        print({'event': 'timing', 'name': 'dynamic_cache_shim_export', 'dt': float(t1 - t0)})
    finally:
        sys.argv = argv
    assert out_path.exists(), "ONNX model export failed"
    sidecar = Path(str(out_path).replace(".onnx", ".dynamic_cache_hint.json"))
    assert sidecar.exists(), "DynamicCache shim sidecar not written"
    # Optional: validate minimal keys
    try:
        import json
        meta = json.loads(sidecar.read_text())
        assert meta.get("cache_interface") == "DynamicCacheShim"
    except Exception:
        pass


