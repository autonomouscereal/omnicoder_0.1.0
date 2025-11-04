import os
from pathlib import Path

from omnicoder.export.onnx_export import main as onnx_export_main


def test_decode_step_export_smoke(tmp_path: Path, monkeypatch):
    # Keep export tiny to minimize memory/time in CI
    monkeypatch.setenv('OMNICODER_EXPORT_TINY', '1')
    args = [
        'prog',
        '--output', str(tmp_path / 'decode.onnx'),
        '--mobile_preset', 'mobile_4gb',
        '--decode_step',
        '--opset', '18',
        '--kv_paged',
        '--kv_page_len', '128',
        '--emit_longctx_variants',
    ]
    monkeypatch.setenv('PYTEST_CURRENT_TEST', 'test_export_decode_step_smoke.py::test_decode_step_export_smoke')
    with monkeypatch.context() as m:
        m.setenv('OMNICODER_USE_DYNAMO', '1')
        # Patch argv for argparse
        import sys
        old_argv = sys.argv
        sys.argv = args
        try:
            onnx_export_main()
        finally:
            sys.argv = old_argv
    assert (tmp_path / 'decode.onnx').exists()
    # Sidecars
    assert (tmp_path / 'decode.kv_paging.json').exists()

