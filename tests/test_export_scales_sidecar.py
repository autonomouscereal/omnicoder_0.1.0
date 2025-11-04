import os
import sys
import json
import tempfile


def test_kv_paging_sidecar_written():
    from omnicoder.export.onnx_export import main as onnx_main
    argv = sys.argv
    try:
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'decode.onnx')
            sys.argv = ['onnx_export', '--output', out, '--seq_len', '1', '--mobile_preset', 'mobile_4gb', '--decode_step', '--kv_paged']
            onnx_main()
            side = out.replace('.onnx', '.kv_paging.json')
            assert os.path.exists(side)
            meta = json.loads(open(side, 'r').read())
            assert meta.get('paged') is True
            assert 'page_len' in meta
    finally:
        sys.argv = argv


