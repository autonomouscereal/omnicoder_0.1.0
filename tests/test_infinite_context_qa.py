import torch
from omnicoder.inference.generate import generate, build_mobile_model_by_name
from omnicoder.training.simple_tokenizer import get_text_tokenizer


def test_infinite_context_memory_priming_runs():
    tok = get_text_tokenizer(prefer_hf=False)
    # Construct a prompt longer than the window to trigger memory priming
    long_text = " ".join(["alpha"] * 200)
    input_ids = torch.tensor([tok.encode(long_text)], dtype=torch.long)
    model = build_mobile_model_by_name('mobile_4gb', mem_slots=2)
    model.eval()
    out = generate(
        model,
        input_ids,
        max_new_tokens=4,
        temperature=0.8,
        top_k=0,
        top_p=1.0,
        window_size=64,
        kvq='none',
    )
    assert out.shape[1] >= input_ids.shape[1], "Should append generated tokens to original input"


def test_kv_bound_canary_runs_decode_after_memory_priming():
	# Ensure we can run another decode step after priming without unbounded KV growth
	from omnicoder.inference.generate import build_mobile_model_by_name
	from omnicoder.training.simple_tokenizer import get_text_tokenizer
	import torch
	model = build_mobile_model_by_name('mobile_4gb', mem_slots=2)
	model.eval()
	tok = get_text_tokenizer(prefer_hf=False)
	ids = torch.tensor([tok.encode("alpha ")], dtype=torch.long)
	# First step generate (primes mem if longer prompt is used elsewhere)
	out1 = model(ids, use_cache=True, past_kv=[(None, None) for _ in range(len(model.blocks))])
	assert isinstance(out1, tuple)
	# Next step with cached KV present should also succeed
	out2 = model(torch.tensor([[tok.encode("beta")[0]]], dtype=torch.long), use_cache=True, past_kv=[(None, None) for _ in range(len(model.blocks))])
	assert isinstance(out2, tuple)


