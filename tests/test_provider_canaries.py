import json
from pathlib import Path


def test_text_provider_bench_canary(tmp_path):
    # Build a tiny ONNX decode-step model and run provider bench on CPU EP (in-process to avoid shell kill signals)
    import sys, os, json as _json, pathlib
    from omnicoder.export.onnx_export import main as onnx_export_main
    from omnicoder.inference.runtimes.provider_bench import main as bench_main
    onnx_path = tmp_path / 'omni.onnx'
    argv_save = sys.argv
    try:
        # Export ONNX decode-step deterministically
        sys.argv = [
            'onnx_export', '--output', str(onnx_path), '--seq_len', '1', '--mobile_preset', 'mobile_4gb', '--decode_step', '--no_dynamo', '--opset', '17'
        ]
        onnx_export_main()
    finally:
        sys.argv = argv_save
    assert onnx_path.exists()
    # Bench provider in-process and write JSON
    out_json = tmp_path / 'bench.json'
    argv_save = sys.argv
    try:
        sys.argv = [
            'provider_bench', '--model', str(onnx_path), '--providers', 'CPUExecutionProvider', '--prompt_len', '16', '--gen_tokens', '16', '--out_json', str(out_json)
        ]
        bench_main()
    finally:
        sys.argv = argv_save
    assert out_json.exists()
    # Diagnostics dump
    try:
        logs_dir = pathlib.Path('tests_logs')
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / 'provider_canary.json').write_text(out_json.read_text(encoding='utf-8'), encoding='utf-8')
    except Exception:
        pass
    data = _json.loads(out_json.read_text(encoding='utf-8'))
    # Accept either flat mapping or structured results list
    if isinstance(data, dict) and 'results' in data:
        providers = [r.get('provider') for r in data.get('results', [])]
        assert 'CPUExecutionProvider' in providers
    else:
        assert 'CPUExecutionProvider' in data


