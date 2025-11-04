import torch


def test_mtp_verifier_shapes_and_acceptance_path():
    from omnicoder.modeling.transformer_moe import OmniTransformer
    from omnicoder.inference.generate import generate
    model = OmniTransformer(vocab_size=128, n_layers=2, d_model=64, n_heads=4, mlp_dim=128, multi_token=2)
    tok = torch.randint(0, 128, (1, 4), dtype=torch.long)
    out = generate(model, tok, max_new_tokens=4, temperature=0.8, top_k=10, verify_threshold=0.0, speculative_draft_len=1)
    assert out.shape[0] == 1 and out.shape[1] >= tok.shape[1]


