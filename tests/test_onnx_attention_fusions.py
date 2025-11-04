import os
import sys
import tempfile
import onnx


def test_attention_fusion_presence():
	# Skip if onnxruntime-tools unavailable; we just check exporter integration produces a file
	from omnicoder.export.onnx_export import main as onnx_main
	argv = sys.argv
	try:
		with tempfile.TemporaryDirectory() as td:
			out = os.path.join(td, 'decode.onnx')
			sys.argv = ['onnx_export', '--output', out, '--seq_len', '1', '--mobile_preset', 'mobile_4gb', '--decode_step']
			onnx_main()
			assert os.path.exists(out)
			m = onnx.load(out)
			# Ensure basic attention or fused path exists: MatMul+Softmax, com.microsoft::Attention, or QLinearMatMul nodes from PTQ
			node_types = [n.op_type for n in m.graph.node]
			fused_domains = [n.domain for n in m.graph.node]
			has_basic = ('MatMul' in node_types and 'Softmax' in node_types)
			has_ms_attn = any((d == 'com.microsoft' and t == 'Attention') for t, d in zip(node_types, fused_domains))
			has_qlinear = ('QLinearMatMul' in node_types)
			assert (has_basic or has_ms_attn or has_qlinear)
	finally:
		sys.argv = argv


