import torch


def test_grin_gate_basic_and_aux_stats():
    from omnicoder.modeling.routing import GRINGate
    gate = GRINGate(d_model=32, n_experts=6, k=2, temperature=1.0, jitter_noise=0.1, st_tau=1.0, mask_drop=0.1)
    x = torch.randn(2, 4, 32)
    idx, scores, probs = gate(x)
    assert idx.shape == (2, 4, 2)
    assert scores.shape == (2, 4, 2)
    assert probs.shape == (2, 4, 6)
    aux = getattr(gate, 'last_aux', None)
    assert aux is not None and 'importance' in aux and 'load' in aux
    assert aux['importance'].shape[0] == 6 and aux['load'].shape[0] == 6

    # Export-guard: ensure calling under torch.jit.trace runs without error
    gate.eval()
    traced = torch.jit.trace(gate, (torch.randn(2, 4, 32),), strict=False)
    out = traced(torch.randn(2, 4, 32))


def test_grin_gate_converges_on_sloped_tokens():
    # Toy convergence measured via cross-entropy decrease rather than hard accuracy
    from omnicoder.modeling.routing import GRINGate
    torch.manual_seed(0)
    gate = GRINGate(d_model=16, n_experts=4, k=1)
    gate.train()
    opt = torch.optim.Adam(gate.parameters(), lr=1e-2)

    # Baseline loss on a fixed batch
    x0 = torch.randn(8, 8, 16)
    target0 = (x0.abs().mean(dim=-1) > 0.8).long() % 4
    _, _, probs0 = gate(x0)
    one_hot0 = torch.zeros_like(probs0)
    one_hot0.scatter_(-1, target0.unsqueeze(-1), 1.0)
    base_loss = -(one_hot0 * (probs0.clamp_min(1e-8).log())).mean().item()

    # Train for a modest number of steps on fresh batches
    for _ in range(60):
        x = torch.randn(8, 8, 16)
        target_expert = (x.abs().mean(dim=-1) > 0.8).long() % 4
        _, _, probs = gate(x)
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(-1, target_expert.unsqueeze(-1), 1.0)
        loss = -(one_hot * (probs.clamp_min(1e-8).log())).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # Re-evaluate on original batch and ensure loss decreased by a small margin
    _, _, probs1 = gate(x0)
    one_hot1 = torch.zeros_like(probs1)
    one_hot1.scatter_(-1, target0.unsqueeze(-1), 1.0)
    final_loss = -(one_hot1 * (probs1.clamp_min(1e-8).log())).mean().item()
    assert final_loss <= base_loss * 0.98


