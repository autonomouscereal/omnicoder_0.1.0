from __future__ import annotations

import json
from pathlib import Path


def test_provider_bench_json_schema() -> None:
    """Sanity-check that provider bench JSONs have expected keys when present.

    This does not run benches; it validates schema if files exist (e.g., from CI artifacts).
    """
    # Vision
    vjson = Path('weights/release/vision/provider_bench.json')
    if vjson.exists():
        data = json.loads(vjson.read_text(encoding='utf-8'))
        assert 'results' in data and isinstance(data['results'], list)
        for rec in data['results']:
            assert 'provider' in rec and 'tps' in rec
    # VQDec
    qjson = Path('weights/release/vqdec/provider_bench.json')
    if qjson.exists():
        data = json.loads(qjson.read_text(encoding='utf-8'))
        assert 'results' in data and isinstance(data['results'], list)
        for rec in data['results']:
            assert 'provider' in rec and 'tps' in rec


