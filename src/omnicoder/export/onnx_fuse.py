from __future__ import annotations

"""
ONNX graph fusions targeting mobile providers (NNAPI/CoreML/DML).

Pass 1: Fuse MatMul->Softmax->MatMul into com.microsoft::Attention when pattern matches
        (Q @ K^T) -> Softmax -> (... @ V).

Pass 2: Insert QDQ around MatMul for weight-only quantization friendliness, or convert
        eligible MatMul to QLinearMatMul when both inputs have static quant params.

These transforms are best-effort and conservative: if a pattern does not match, the
graph is left unchanged. They are intended to complement ONNX Runtime built-in fusions.
"""

from pathlib import Path
from typing import List, Tuple


def _try_import_onnx():
    try:
        import onnx  # type: ignore
        from onnx import helper, TensorProto, numpy_helper  # type: ignore
        return onnx, helper, TensorProto, numpy_helper
    except Exception as e:  # pragma: no cover
        raise RuntimeError("onnx is required for fusion pass. pip install onnx>=1.15") from e


def _node_by_output(nodes, name):
    for n in nodes:
        if name in n.output:
            return n
    return None


def _replace_node(graph, old_nodes: List, new_node):
    # Remove old_nodes by identity and add new_node, reconnecting outputs
    keep = [n for n in graph.node if n not in old_nodes]
    keep.append(new_node)
    del graph.node[:]
    graph.node.extend(keep)


def _fuse_attention(model) -> int:
    onnx, helper, TensorProto, _ = _try_import_onnx()
    graph = model.graph
    fused = 0
    # Pattern: matmul1 -> softmax -> matmul2
    # matmul1: Q x K^T, matmul2: softmax_out x V
    nodes = list(graph.node)
    for smx in [n for n in nodes if n.op_type == 'Softmax']:
        # softmax input should be MatMul or Add(MatMul, mask). Sometimes a Cast precedes Add/MatMul; walk through it.
        in_name = smx.input[0]
        prod = _node_by_output(nodes, in_name)
        if prod is not None and prod.op_type == 'Cast':
            in_name = prod.input[0]
            prod = _node_by_output(nodes, in_name)
        mm1 = None
        add_mask = None
        if prod is None:
            continue
        if prod.op_type == 'MatMul':
            mm1 = prod
        elif prod.op_type == 'Add':
            add_mask = prod
            cand0 = _node_by_output(nodes, add_mask.input[0])
            cand1 = _node_by_output(nodes, add_mask.input[1])
            if cand0 is not None and cand0.op_type == 'MatMul':
                mm1 = cand0
            elif cand1 is not None and cand1.op_type == 'MatMul':
                mm1 = cand1
        if mm1 is None or mm1.op_type != 'MatMul':
            continue
        # softmax output must feed MatMul as left input
        smx_out = smx.output[0]
        mm2 = None
        for n in nodes:
            if n.op_type == 'MatMul' and smx_out in n.input:
                mm2 = n
                break
        if mm2 is None:
            continue
        # Identify Q, K, V; handle Transpose on K
        q = mm1.input[0]
        k_t = mm1.input[1]
        k_transpose = _node_by_output(nodes, k_t)
        if k_transpose and k_transpose.op_type == 'Transpose':
            k = k_transpose.input[0]
        else:
            # If no transpose, assume already correct orientation
            k = k_t
            k_transpose = None
        # V is the other input of mm2
        v = mm2.input[0] if mm2.input[1] == smx_out else mm2.input[1]

        # Optional mask: if we found an Add(MatMul, mask), pass mask as 4th input. Cast to float if needed.
        mask_name = None
        if add_mask is not None:
            # pick the non-matmul input as mask
            mask_candidate = add_mask.input[0] if add_mask.input[1] == mm1.output[0] else add_mask.input[1]
            mask_node = _node_by_output(nodes, mask_candidate)
            if mask_node is not None and mask_node.op_type == 'Cast':
                mask_name = mask_node.output[0]
            else:
                # Insert a Cast to float to satisfy Attention schema requirements
                onnx, helper, TensorProto, _ = _try_import_onnx()
                cast_out = mask_candidate + ":f32"
                cast_node = helper.make_node('Cast', inputs=[mask_candidate], outputs=[cast_out], name=f"Cast_mask_{fused}", to=TensorProto.FLOAT)
                graph.node.append(cast_node)
                mask_name = cast_out

        # Build com.microsoft::Attention node
        attn_out = mm2.output[0]
        # Include required attributes and optional mask input
        attn_inputs = [q, k, v]
        if mask_name is not None:
            attn_inputs.append(mask_name)
        attn = helper.make_node(
            'Attention',
            inputs=attn_inputs,
            outputs=[attn_out],
            domain='com.microsoft',
            name=f'FusedAttention_{fused}',
            num_heads=1,
            unidirectional=1,
            mask_filter_value=-1e4,
        )
        # Add informative attributes to assist backends with KV-latent and RoPE mapping when supported
        try:
            from onnx import AttributeProto  # type: ignore
            attn.attribute.extend([
                # Hints only; backends may ignore if unsupported
                # KV latent dim per head (best-effort default)
                helper.make_attribute('kv_latent_hint', 160),
                helper.make_attribute('rope_enabled', 1),
            ])
        except Exception:
            pass

        # Replace chain mm1 -> (add_mask?) -> smx -> mm2 with attn
        kill = [mm1, smx, mm2]
        if add_mask is not None:
            kill.append(add_mask)
        if k_transpose is not None:
            # If we removed transpose feeding only this path and not used elsewhere, it can be removed
            used = any(k_transpose.output[0] in n.input for n in nodes if n not in kill)
            if not used:
                kill.append(k_transpose)
        _replace_node(graph, kill, attn)
        fused += 1
    return fused


def _insert_qdq_for_matmul(model, per_channel: bool = False, act_scale_default: float | None = None) -> int:
    onnx, helper, TensorProto, numpy_helper = _try_import_onnx()
    graph = model.graph
    nodes = list(graph.node)
    added = 0
    # Wrap MatMul activations with QDQ to enable ORT QDQ fusion on backends like NNAPI/DML
    for n in nodes:
        if n.op_type != 'MatMul':
            continue
        a, b = n.input
        # Create scales/zeros initializers for activations (calibrated placeholder, zero=0)
        scale_name = f"{n.name or 'MatMul'}_a_scale"
        zp_name = f"{n.name or 'MatMul'}_a_zero"
        # Avoid duplicates
        if any(init.name == scale_name for init in graph.initializer):
            continue
        val = float(0.02 if act_scale_default is None else act_scale_default)
        scale = numpy_helper.from_array(__import__('numpy').array([val], dtype='float32'), name=scale_name)
        zp = numpy_helper.from_array(__import__('numpy').array([0], dtype='uint8'), name=zp_name)
        graph.initializer.extend([scale, zp])
        q = helper.make_node('QuantizeLinear', inputs=[a, scale_name, zp_name], outputs=[scale_name + ':q'], name=f'{n.name}_Qa')
        dq = helper.make_node('DequantizeLinear', inputs=[scale_name + ':q', scale_name, zp_name], outputs=[scale_name + ':dq'], name=f'{n.name}_DQa')
        # Rewire matmul input
        n.input[0] = dq.output[0]
        graph.node.extend([q, dq])
        added += 2
    return added


def _convert_qdq_matmul_to_qlinear(model) -> int:
    onnx, helper, TensorProto, numpy_helper = _try_import_onnx()
    graph = model.graph
    nodes = list(graph.node)
    converted = 0
    # Map from tensor name to (q_in, scale, zp) if it is DQ(Q(x, s, z))
    def parse_dq_input(name):
        dq = next((n for n in nodes if n.output and n.output[0] == name and n.op_type == 'DequantizeLinear'), None)
        if dq is None:
            return None
        qin = dq.input[0]
        q = next((n for n in nodes if n.output and n.output[0] == qin and n.op_type == 'QuantizeLinear'), None)
        if q is None:
            return None
        # Ensure scales and zps are initializers (static)
        s_name, z_name = q.input[1], q.input[2] if len(q.input) > 2 else None
        inits = {ini.name: ini for ini in graph.initializer}
        if s_name not in inits:
            return None
        if z_name and z_name not in inits:
            return None
        if not z_name:
            # create zero initializer if missing
            z_name = f"{q.name or 'Q'}_auto_zp"
            zp_init = numpy_helper.from_array(__import__('numpy').array([0], dtype='uint8'), name=z_name)
            graph.initializer.extend([zp_init])
        return (q.input[0], s_name, z_name)  # original float input, scale, zp

    new_nodes = []
    to_remove = set()
    for mm in [n for n in nodes if n.op_type == 'MatMul']:
        a_dq = parse_dq_input(mm.input[0])
        b_dq = parse_dq_input(mm.input[1])
        if not a_dq or not b_dq:
            continue
        a_x, a_s, a_z = a_dq
        b_x, b_s, b_z = b_dq
        # Y scale/zp: use composite scale and zero=0 as a conservative default
        y_s_name = f"{mm.name or 'MatMul'}_y_scale"
        y_z_name = f"{mm.name or 'MatMul'}_y_zero"
        inits = {ini.name: ini for ini in graph.initializer}
        if y_s_name not in inits:
            # y_scale = a_scale * b_scale (approx)
            import numpy as np
            a_s_arr = onnx.numpy_helper.to_array(inits[a_s])
            b_s_arr = onnx.numpy_helper.to_array(inits[b_s])
            y_s_init = numpy_helper.from_array(np.array([float(a_s_arr.flatten()[0] * b_s_arr.flatten()[0])], dtype='float32'), name=y_s_name)
            y_z_init = numpy_helper.from_array(np.array([0], dtype='uint8'), name=y_z_name)
            graph.initializer.extend([y_s_init, y_z_init])
        qlin_out = mm.output[0]
        qlin = helper.make_node(
            'QLinearMatMul',
            inputs=[a_x, a_s, a_z, b_x, b_s, b_z, y_s_name, y_z_name],
            outputs=[qlin_out],
            name=f"QLinear_{mm.name or converted}",
        )
        new_nodes.append(qlin)
        to_remove.update({mm})
        # Also remove the immediate Q/DQ pairs if they have no other consumers will be handled by ORT/onnx-simplifier
        converted += 1

    if converted > 0:
        keep = [n for n in graph.node if n not in to_remove]
        keep.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(keep)
    return converted


def fuse_and_pack(model_path: str, out_path: str, provider_hint: str = '', act_scale_default: float | None = None) -> Tuple[int, int]:
    """
    Load an ONNX graph, apply fusions and QDQ packing, and save to out_path.
    Returns (num_attention_fused, num_qdq_nodes_added/2).
    """
    onnx, _, _, _ = _try_import_onnx()
    model = onnx.load(model_path)
    fused = _fuse_attention(model)
    qdq_pairs = _insert_qdq_for_matmul(model, per_channel=False, act_scale_default=act_scale_default)
    # Try to convert eligible MatMul+DQ/Q to QLinearMatMul when static scales exist
    try:
        _ = _convert_qdq_matmul_to_qlinear(model)
    except Exception:
        pass
    onnx.save(model, out_path)
    return fused, qdq_pairs // 2


def main() -> None:  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description='Fuse attention and insert QDQ for MatMul (mobile backends)')
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--provider', type=str, default='')
    ap.add_argument('--act_scale_default', type=float, default=None, help='If set, use this activation scale for Q/DQ wrappers')
    args = ap.parse_args()
    fused, qdq = fuse_and_pack(args.model, args.out, provider_hint=args.provider, act_scale_default=args.act_scale_default)
    print(f"Fused attention blocks: {fused}; QDQ MatMul wrapped: {qdq}")


if __name__ == '__main__':  # pragma: no cover
    main()


