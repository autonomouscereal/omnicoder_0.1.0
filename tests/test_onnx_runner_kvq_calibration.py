import os
import sys
import tempfile
import json


def test_onnx_runner_accepts_kvq_calibration():
    # Export decode-step
    from omnicoder.export.onnx_export import main as onnx_main
    from omnicoder.inference.runtimes.onnx_decode_generate import main as run_main
    argv = sys.argv
    try:
        with tempfile.TemporaryDirectory() as td:
            onx = os.path.join(td, 'decode.onnx')
            sys.argv = ['onnx_export', '--output', onx, '--seq_len', '1', '--mobile_preset', 'mobile_4gb', '--decode_step']
            onnx_main()
            # Minimal calibration JSON
            cal = os.path.join(td, 'kvq.json')
            json.dump({'scheme': 'u8', 'group_size': 64, 'layers': {}}, open(cal, 'w'))
            # Runner CLI
            sys.argv = ['onnx_run', '--model', onx, '--prompt', 'hi', '--max_new_tokens', '1', '--kvq', 'u8', '--kvq_calibration', cal]
            run_main()
    finally:
        sys.argv = argv


