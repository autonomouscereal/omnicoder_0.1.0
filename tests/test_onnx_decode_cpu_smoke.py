import os
from pathlib import Path


def test_onnx_decode_step_cpu_smoke():
    # Locate a decode-step ONNX model if available
    candidates = [
        Path('weights/release/text/omnicoder_decode_step.onnx'),
        Path('weights/text/omnicoder_decode_step.onnx'),
    ]
    onnx_path = next((p for p in candidates if p.exists()), None)
    if onnx_path is None:
        # Nothing to run in this environment; consider this a no-op smoke
        return
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        # ORT not available in this environment
        return
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])  # type: ignore
    inputs = sess.get_inputs()
    input_name = inputs[0].name
    outputs = sess.get_outputs()
    output_names = [o.name for o in outputs]
    # Find K/V input tensors
    k_inputs = [i for i in inputs if i.name.startswith('k_lat_')]
    v_inputs = [i for i in inputs if i.name.startswith('v_lat_')]
    k_inputs.sort(key=lambda x: int(x.name.split('_')[-1]))
    v_inputs.sort(key=lambda x: int(x.name.split('_')[-1]))
    L = min(len(k_inputs), len(v_inputs))
    import numpy as np  # type: ignore
    # One prompt token
    ids = np.random.randint(0, 32000, size=(1, 1), dtype=np.int64)
    feeds = {input_name: ids}
    # Zero-length past K/V per layer
    for i in range(L):
        ks = k_inputs[i].shape
        try:
            heads = int(ks[1]); dl = int(ks[3])
        except Exception:
            heads, dl = 8, 160
        feeds[k_inputs[i].name] = np.zeros((1, heads, 0, dl), dtype=np.float32)
        feeds[v_inputs[i].name] = np.zeros((1, heads, 0, dl), dtype=np.float32)
    _ = sess.run(output_names, feeds)
    # If we reached here without exceptions, the smoke passes


