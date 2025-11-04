from __future__ import annotations

import json
from omnicoder.tools.kv_budget_enforce import compute_kv_bytes


def test_kv_paging_compute_bytes_uniform():
    meta = {
        "page_len": 256,
        "n_layers": 4,
        "heads": 8,
        "dl": 64,
    }
    # bytes = layers * heads * dl * page_len * 2 (K and V)
    expect = 4 * 8 * 64 * 256 * 2
    assert compute_kv_bytes(meta) == expect


def test_kv_paging_compute_bytes_per_layer_and_spill():
    meta = {
        "page_len": 128,
        "n_layers": 3,
        "heads": 4,
        "dl_per_layer": [32, 48, 64],
        "spill_bytes": 4096,
    }
    expect = (32 + 48 + 64) * 4 * 128 * 2 + 4096
    assert compute_kv_bytes(meta) == expect


