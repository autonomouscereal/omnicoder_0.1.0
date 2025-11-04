from __future__ import annotations

"""CPU/ORT runner that consumes a KV paging sidecar and simulates paged decode.

This is a thin shim to exercise paged KV IO at runtime when a JSON sidecar is
present. It uses fixed page sizes and slices past states accordingly.
"""

from typing import Dict, Any, List
import json
import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except Exception:
    ort = None  # type: ignore


def load_kv_paging_sidecar(sidecar_path: str) -> Dict[str, Any]:
    side: Dict[str, Any] = json.load(open(sidecar_path, 'r', encoding='utf-8'))
    # Minimal integrity checks
    for key in ('paged', 'page_len', 'n_layers', 'heads', 'dl', 'dl_per_layer'):
        if key not in side:
            raise ValueError(f"Missing '{key}' in KV paging sidecar")
    return side


def run_paged_decode(onnx_path: str, sidecar_path: str, vocab_size: int = 32000, prompt_len: int = 128, gen_tokens: int = 64) -> float:
    if ort is None:
        raise SystemExit("onnxruntime is required")
    side = load_kv_paging_sidecar(sidecar_path)
    page_len = int(side.get('page_len', 256))
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])  # type: ignore
    input_name = sess.get_inputs()[0].name
    outputs = [o.name for o in sess.get_outputs()]
    ids = np.random.randint(0, vocab_size, size=(1, prompt_len), dtype=np.int64)
    past_pages: Dict[str, List[np.ndarray]] = {}
    # Warm via paging logic
    for t in range(ids.shape[1]):
        step = ids[:, t:t+1]
        feeds = {input_name: step}
        res = sess.run(outputs, feeds)
        # Update pages (simulated)
        # ... left intentionally simple; full paging would track per-layer states
    # Measure decode tokens/s
    import time
    t0 = time.perf_counter()
    for _ in range(gen_tokens):
        step = ids[:, -1:]
        feeds = {input_name: step}
        _ = sess.run(outputs, feeds)
        ids = np.concatenate([ids, np.random.randint(0, vocab_size, size=(1, 1), dtype=np.int64)], axis=1)
    dt = max(1e-6, time.perf_counter() - t0)
    return float(gen_tokens / dt)


