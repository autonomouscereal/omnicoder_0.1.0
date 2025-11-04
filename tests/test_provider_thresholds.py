def test_default_thresholds_present():
    # Ensure provider_bench default threshold string includes DML and CoreML
    import importlib
    m = importlib.import_module('omnicoder.inference.runtimes.provider_bench')
    import inspect
    src = inspect.getsource(m)
    assert 'DmlExecutionProvider=10.0' in src
    assert 'CoreMLExecutionProvider=6.0' in src



def test_cpu_throughput_canary_tokens_per_second():
	from omnicoder.inference.benchmark import bench_tokens_per_second
	from omnicoder.modeling.transformer_moe import OmniTransformer
	# Use a tiny-but-representative model to ensure stable TPS without exceeding 10s
	m = OmniTransformer(
		vocab_size=32000,
		n_layers=2,
		d_model=256,
		n_heads=4,
		mlp_dim=768,
		n_experts=1,
		top_k=1,
		max_seq_len=2048,
		use_rope=True,
		kv_latent_dim=64,
		multi_query=True,
		multi_token=1,
	)
	tps = bench_tokens_per_second(m, seq_len=64, gen_tokens=64, device='cpu')
	print({"provider": "CPUExecutionProvider", "tps": float(tps), "min_tps": 200.0, "pass": bool(tps > 200.0)})
	assert tps > 200.0

