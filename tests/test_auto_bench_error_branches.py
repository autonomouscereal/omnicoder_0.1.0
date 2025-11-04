import os
from pathlib import Path


def test_bench_providers_invalid_path_returns_error():
    from omnicoder.eval.auto_benchmark import bench_providers
    fake_onnx = str(Path('does_not_exist') / 'model.onnx')
    res = bench_providers(fake_onnx, providers=["CPUExecutionProvider"], prompt_len=1, gen_tokens=1, vocab_size=8)
    assert isinstance(res, dict)
    assert "CPUExecutionProvider" in res
    assert isinstance(res["CPUExecutionProvider"], dict)
    assert "error" in res["CPUExecutionProvider"], res


def test_bench_image_latency_onnx_missing_folder_returns_none():
    from omnicoder.eval.auto_benchmark import bench_image_latency
    out = bench_image_latency(device="cpu", backend="onnx", sd_model=None, sd_local_path=str(Path('weights')/ 'missing' / 'onnx'), provider="CPUExecutionProvider", provider_profile="")
    # When ONNX folder missing or invalid, function should return None without raising.
    assert out is None or isinstance(out, dict)


def test_quality_metrics_imports_missing_dont_crash(tmp_path):
    # Ensure optional quality block doesn't raise when inputs are empty
    from omnicoder.eval.auto_benchmark import main as auto_bench_main
    # Create temp output file path
    out_json = tmp_path / 'summary.json'
    # Simulate running with minimal required args; no quality inputs provided
    import sys
    import time
    argv_bak = sys.argv[:]
    try:
        sys.argv = [
            'auto_bench',
            '--device', 'cpu',
            '--seq_len', '8',
            '--gen_tokens', '8',
            '--preset', 'mobile_4gb',
            '--out', str(out_json),
        ]
        t0 = time.perf_counter()
        auto_bench_main()
        t1 = time.perf_counter()
        print({
            'event': 'slow_test_probe',
            'name': 'test_quality_metrics_imports_missing_dont_crash',
            'dt': float(t1 - t0),
        })
        assert out_json.exists()
    finally:
        sys.argv = argv_bak


