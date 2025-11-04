import os
import sys
import tempfile


def test_dynamic_cache_shim_export_creates_sidecar_and_model():
    # Export decode-step with shim and ensure model + sidecar exist
    from omnicoder.export.onnx_export import main as onnx_main
    argv = sys.argv
    try:
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'decode.onnx')
            sys.argv = [
                'onnx_export',
                '--output', out,
                '--seq_len', '1',
                '--mobile_preset', 'mobile_4gb',
                '--decode_step',
                '--emit_longctx_variants',
                '--dynamic_cache_shim',
            ]
            onnx_main()
            assert os.path.exists(out)
            sidecar = out.replace('.onnx', '.dynamic_cache_hint.json')
            assert os.path.exists(sidecar)
    finally:
        sys.argv = argv


def test_dynamic_cache_flag_allows_export_without_error():
    # When --dynamic_cache is set, exporter should attempt dynamo and fall back gracefully
    from omnicoder.export.onnx_export import main as onnx_main
    argv = sys.argv
    try:
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'decode.onnx')
            sys.argv = [
                'onnx_export',
                '--output', out,
                '--seq_len', '1',
                '--mobile_preset', 'mobile_4gb',
                '--decode_step',
                '--dynamic_cache',
            ]
            import time
            t0 = time.perf_counter()
            onnx_main()
            t1 = time.perf_counter()
            print({'event': 'timing', 'name': 'dynamic_cache_export', 'dt': float(t1 - t0)})
            assert os.path.exists(out)
    finally:
        sys.argv = argv


