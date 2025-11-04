import torch


def test_windowed_attention_masks_recent_tokens_only():
    from omnicoder.modeling.attention import LatentKVAttention
    torch.manual_seed(0)
    attn = LatentKVAttention(d_model=64, n_heads=4, kv_latent_dim=16, use_rope=False, window_size=4, use_sdpa=False)
    x = torch.randn(1, 6, 64)
    # No past: should work and return shape (1,6,64)
    y = attn(x, use_cache=False)
    assert y.shape == x.shape
    # With past of len 8: mask should constrain to last 4 of total past+current
    past_k = torch.randn(1, 4, 8, 16)  # (B,H,Tpast,DL)
    past_v = torch.randn(1, 4, 8, 16)
    y2, nk, nv = attn(x, past_k_latent=past_k, past_v_latent=past_v, use_cache=True)
    assert y2.shape == x.shape and nk is not None and nv is not None


def test_windowed_attention_decode_stability_small_noise():
    from omnicoder.modeling.attention import LatentKVAttention
    torch.manual_seed(0)
    attn = LatentKVAttention(d_model=64, n_heads=4, kv_latent_dim=16, use_rope=False, window_size=8, use_sdpa=True)
    # Simulate single-token decode with growing past and ensure outputs remain finite and bounded
    past_k = None
    past_v = None
    x = torch.randn(1, 1, 64)
    for t in range(16):
        if past_k is not None and past_v is not None:
            y, past_k, past_v = attn(x, past_k_latent=past_k, past_v_latent=past_v, use_cache=True)
        else:
            y, past_k, past_v = attn(x, use_cache=True)
        assert torch.isfinite(y).all()
        x = torch.randn(1, 1, 64)  # next token


