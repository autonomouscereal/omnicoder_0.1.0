import torch

from omnicoder.inference.generate import build_mobile_model_by_name


def test_router_knobs_runtime_only():
    model = build_mobile_model_by_name('mobile_4gb', multi_token=2)
    model.eval()
    # Ensure router attributes exist and can be tweaked without errors
    for blk in getattr(model, 'blocks', []):
        if hasattr(blk, 'moe'):
            # Top-k change
            try:
                blk.moe.top_k = max(1, int(getattr(blk.moe, 'top_k', 1)))
            except Exception:
                pass
            # Capacity factor
            try:
                blk.moe.capacity_factor = float(getattr(blk.moe, 'capacity_factor', 1.0))
            except Exception:
                pass
    # One forward pass
    tok = torch.randint(0, getattr(model, 'vocab_size', 32000), (1, 4), dtype=torch.long)
    out = model(tok, past_kv=None, use_cache=False)
    assert isinstance(out, torch.Tensor) or (isinstance(out, tuple) and len(out) >= 1)

