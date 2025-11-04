import os
import sys
from pathlib import Path
import subprocess


def test_emit_longctx_variants(tmp_path: Path):
    out = tmp_path / "omnicoder_decode_step.onnx"
    cmd = [
        sys.executable, "-m", "omnicoder.export.onnx_export",
        "--output", str(out),
        "--seq_len", "1",
        "--mobile_preset", "mobile_4gb",
        "--decode_step",
        "--emit_longctx_default",
        "--no_dynamo", "--opset", "17",
    ]
    env = os.environ.copy()
    # Require both 32k and 128k for this test
    env['OMNICODER_EXPORT_ALL_LONGCTX'] = '1'
    rc = subprocess.run(cmd, check=False, env=env)
    assert rc.returncode == 0
    ctx32k = out.with_name("omnicoder_decode_step_ctx32k.onnx")
    ctx128k = out.with_name("omnicoder_decode_step_ctx128k.onnx")
    # Best-effort: ensure files exist
    assert ctx32k.exists(), "32k variant not emitted"
    assert ctx128k.exists(), "128k variant not emitted"

import os
import sys
import tempfile


def test_emit_longctx_variants_exist():
    from omnicoder.export.onnx_export import main as onnx_main
    argv = sys.argv
    try:
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'decode.onnx')
            sys.argv = [
                'onnx_export', '--output', out, '--seq_len', '1', '--mobile_preset', 'mobile_4gb', '--decode_step', '--emit_longctx_variants', '--no_dynamo', '--opset', '17'
            ]
            os.environ['OMNICODER_EXPORT_ALL_LONGCTX'] = '1'
            onnx_main()
            assert os.path.exists(out)
            ctx32 = out.replace('.onnx', '_ctx32k.onnx')
            ctx128 = out.replace('.onnx', '_ctx128k.onnx')
            assert os.path.exists(ctx32) or os.path.exists(ctx128)
    finally:
        os.environ.pop('OMNICODER_EXPORT_ALL_LONGCTX', None)
        sys.argv = argv


