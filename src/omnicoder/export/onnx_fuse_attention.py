from __future__ import annotations

"""ONNX attention fusion and QDQ insertion pass.

Attempts to fuse MatMul-Softmax-MatMul patterns into com.microsoft::Attention
and inserts Q/DQ pairs to enable QLinearMatMul conversion in quantization.

Uses onnxruntime-tools optimizer when available; otherwise no-ops.
"""

from pathlib import Path
from typing import Tuple


def fuse_and_pack(in_path: str | Path, out_path: str | Path, provider_hint: str | None = None) -> Tuple[int, int]:
    # Prefer modern onnxruntime.transformers optimizer; fallback to legacy tools; else copy
    used = ""
    try:
        # ORT >= 1.20
        from onnxruntime.transformers.optimizer import optimize_model  # type: ignore
        used = "ort"
        optimized_model = optimize_model(
            str(in_path),
            model_type='bert',
            num_heads=8,
            hidden_size=1024,
            optimization_options={
                'enable_embed_layer_norm': False,
                'enable_attention': True,
            },
        )
        optimized_model.save_model_to_file(str(out_path))
    except Exception:
        try:
            from onnxruntime_tools import optimizer  # type: ignore
            from onnxruntime_tools.transformers.onnx_model_bert import BertOptimizationOptions  # type: ignore
            used = "tools"
            opt_options = BertOptimizationOptions("bert")
            opt_options.enable_embed_layer_norm = False
            opt_options.enable_attention = True
            optimized_model = optimizer.optimize_model(
                str(in_path),
                model_type='bert',
                num_heads=8,
                hidden_size=1024,
                optimization_options=opt_options,
            )
            optimized_model.save_model_to_file(str(out_path))
        except Exception:
            try:
                import shutil
                shutil.copyfile(in_path, out_path)
            except Exception:
                return (0, 0)
    # Count fused Attention and QLinearMatMul nodes
    try:
        import onnx  # type: ignore
        m = onnx.load(str(out_path))
        attn = 0
        qlin = 0
        for n in m.graph.node:
            if n.domain == 'com.microsoft' and n.op_type == 'Attention':
                attn += 1
            if n.op_type == 'QLinearMatMul':
                qlin += 1
        return (attn, qlin)
    except Exception:
        return (0, 0)


# Optional utility: insert conservative QDQ wrappers around MatMul inputs to improve PTQ fusions
# (not wired by default to avoid changing existing flows)
def insert_qdq_inputs(onnx_path: str | Path, out_path: str | Path) -> bool:
    try:
        import onnx  # type: ignore
        from onnx import helper, TensorProto  # type: ignore
        model = onnx.load(str(onnx_path))
        graph = model.graph
        new_nodes = []
        for node in list(graph.node):
            if node.op_type == 'MatMul':
                # Add QuantizeLinear/DequantizeLinear around first input
                x_name = node.input[0]
                scale_name = x_name + '_scale'
                zero_name = x_name + '_zero'
                # Create initializer placeholders if not present
                def _make_init(name, tensor):
                    for init in graph.initializer:
                        if init.name == name:
                            return
                    graph.initializer.extend([tensor])
                scale_init = helper.make_tensor(scale_name, TensorProto.FLOAT, dims=[], vals=[1.0])
                zero_init = helper.make_tensor(zero_name, TensorProto.UINT8, dims=[], vals=[0])
                _make_init(scale_name, scale_init)
                _make_init(zero_name, zero_init)
                q_node = helper.make_node('QuantizeLinear', [x_name, scale_name, zero_name], [x_name + ':q'])
                dq_node = helper.make_node('DequantizeLinear', [q_node.output[0], scale_name, zero_name], [x_name + ':dq'])
                node.input[0] = dq_node.output[0]
                new_nodes.extend([q_node, dq_node])
        # Prepend new nodes
        graph.node[:0] = new_nodes + list(graph.node)
        onnx.save(model, str(out_path))
        return True
    except Exception:
        return False



