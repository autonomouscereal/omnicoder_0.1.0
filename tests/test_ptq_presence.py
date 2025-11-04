import os
from pathlib import Path
import subprocess
import sys


def test_per_op_ptq_inserts_qdq(tmp_path: Path):
    # Export a tiny decode-step ONNX first
    onnx_path = tmp_path / 'decode.onnx'
    env = dict(os.environ)
    # Prefer legacy exporter path in constrained containers to avoid sporadic SIGKILL
    env['OMNICODER_USE_DYNAMO'] = '0'
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    env['TORCH_NUM_THREADS'] = '1'
    # Add guards to reduce memory use and avoid SIGKILL in constrained containers
    env.setdefault('OMNICODER_USE_DYNAMO', '0')
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')
    env.setdefault('TORCH_NUM_THREADS', '1')
    env['OMNICODER_EXPORT_TINY'] = '1'
    subprocess.run([
        sys.executable, '-m', 'omnicoder.export.onnx_export',
        '--output', str(onnx_path), '--seq_len', '1', '--mobile_preset', 'mobile_4gb', '--decode_step', '--opset', '18', '--no_dynamo'
    ], check=True, env=env)
    # Run per-op PTQ (generic preset)
    q_path = tmp_path / 'decode_int8.onnx'
    subprocess.run([
        sys.executable, '-m', 'omnicoder.export.onnx_quantize_per_op',
        '--model', str(onnx_path), '--out', str(q_path), '--preset', 'generic', '--auto_exclude', '--per_channel'
    ], check=True)
    # Inspect graph for QDQ/QLinear nodes
    import onnx
    m = onnx.load(str(q_path))
    assert any(n.op_type in ('QuantizeLinear','DequantizeLinear','QLinearMatMul') for n in m.graph.node)


