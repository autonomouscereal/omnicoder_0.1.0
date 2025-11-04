from __future__ import annotations

import os
from pathlib import Path


def test_decode_step_dynamic_cache_roundtrip(tmp_path: Path):
    # Soft skip if onnxruntime is not installed
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return

    # Export a small decode-step ONNX for mobile_4gb preset
    from omnicoder.export.onnx_export import main as onnx_export_main
    from omnicoder.modeling.transformer_moe import OmniTransformer
    from omnicoder.config import MobilePreset
    import numpy as np

    out_dir = tmp_path / "text"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(out_dir / "decode_step_test.onnx")

    import sys
    argv = sys.argv
    try:
        # Force tiny export in pytest to keep runtime under 10s while preserving IO names
        os.environ['OMNICODER_EXPORT_TINY_FORCE_ALL'] = '1'
        sys.argv = [
            "onnx_export",
            "--output",
            model_path,
            "--seq_len",
            "1",
            "--mobile_preset",
            "mobile_4gb",
            "--decode_step",
            "--no_dynamo",
        ]
        onnx_export_main()
    finally:
        sys.argv = argv

    # Derive L, H, DL from the exported ONNX IO to avoid env-specific shrink mismatches
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])  # type: ignore
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    in_names = [i.name for i in inputs]
    out_names = [o.name for o in outputs]
    assert in_names[0] == "input_ids"

    # Step 1: feed input id and zero-length caches (T_past=0)
    # Discover K/V inputs and static dims
    k_inputs = [i for i in inputs if i.name.startswith('k_lat_')]
    v_inputs = [i for i in inputs if i.name.startswith('v_lat_')]
    k_inputs.sort(key=lambda x: int(x.name.split('_')[-1]))
    v_inputs.sort(key=lambda x: int(x.name.split('_')[-1]))
    num_layers = min(len(k_inputs), len(v_inputs))
    heads_per_layer, dl_per_layer = [], []
    for i in range(num_layers):
        ks = k_inputs[i].shape
        try:
            heads = int(ks[1]); dl = int(ks[3])
        except Exception:
            heads, dl = 8, 160
        heads_per_layer.append(heads)
        dl_per_layer.append(dl)
    ids = np.random.randint(0, 32000, size=(1, 1), dtype=np.int64)
    feeds = {"input_ids": ids}
    for i in range(num_layers):
        feeds[k_inputs[i].name] = np.zeros((1, heads_per_layer[i], 0, dl_per_layer[i]), dtype=np.float32)
    for i in range(num_layers):
        feeds[v_inputs[i].name] = np.zeros((1, heads_per_layer[i], 0, dl_per_layer[i]), dtype=np.float32)

    out_vals = sess.run(out_names, feeds)
    # Expect logits + 2*num_layers cache tensors (+ optional mtp heads we ignore)
    assert len(out_vals) >= 1 + 2 * num_layers
    # Extract new caches and check T_total == 1
    nk = out_vals[1 : 1 + num_layers]
    nv = out_vals[1 + num_layers : 1 + 2 * num_layers]
    for t in nk + nv:
        assert t.shape[2] == 1

    # Step 2: feed again with past caches from step 1; expect T_total == 2
    # Use the canonical vocab size for test graphs (matches exporter default)
    ids2 = np.random.randint(0, 32000, size=(1, 1), dtype=np.int64)
    feeds2 = {"input_ids": ids2}
    for i in range(num_layers):
        feeds2[k_inputs[i].name] = nk[i]
    for i in range(num_layers):
        feeds2[v_inputs[i].name] = nv[i]
    out_vals2 = sess.run(out_names, feeds2)
    nk2 = out_vals2[1 : 1 + num_layers]
    nv2 = out_vals2[1 + num_layers : 1 + 2 * num_layers]
    for t in nk2 + nv2:
        assert t.shape[2] == 2


