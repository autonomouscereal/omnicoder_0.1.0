import torch

from omnicoder.modeling.transformer_moe import OmniTransformer


def test_grin_router_tiny_convergence():
    # Fixed seed for determinism
    torch.manual_seed(1234)
    # Tiny model for fast convergence
    model = OmniTransformer(n_layers=1, d_model=64, n_heads=4, mlp_dim=128, n_experts=6, top_k=2, max_seq_len=32)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Enable GRIN router via blended router switch if present
    # and stochastic sampling for broader exploration
    if hasattr(model.blocks[0].moe, '_router_grin'):
        # Shift blend toward GRIN during training
        try:
            model.blocks[0].moe._blend.data = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        except Exception:
            pass
    if hasattr(model.blocks[0].moe, '_router_topk'):
        try:
            r = model.blocks[0].moe._router_topk
            r.sample_gumbel_topk = True  # type: ignore[attr-defined]
            r.use_gumbel = True
            r.jitter_noise = 0.3
        except Exception:
            pass

    # Synthetic tokens and labels
    x = torch.randint(0, 100, (8, 16), dtype=torch.long)
    y = torch.randint(0, model.vocab_size, (8, 16), dtype=torch.long)

    losses = []
    for _ in range(8):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    # Expect non-increasing trend; allow small noise by checking last <= first * 1.05
    assert losses[-1] <= losses[0] * 1.05


