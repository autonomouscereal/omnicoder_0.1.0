import torch


def test_verifier_runtime_gating_and_write_policy_runs():
    from omnicoder.modeling.transformer_moe import OmniTransformer
    from omnicoder.inference.generate import generate
    from omnicoder.inference.knn_cache import KNNCache

    vocab = 128
    model = OmniTransformer(vocab_size=vocab, n_layers=2, d_model=64, n_heads=4, mlp_dim=128, multi_token=2)
    # Ensure write_head exists and is callable
    assert hasattr(model, 'write_head')

    tok = torch.randint(0, vocab, (1, 4), dtype=torch.long)
    cache = KNNCache(dim=model.lm_head.in_features, use_faiss=False)

    out = generate(
        model,
        tok,
        max_new_tokens=4,
        temperature=0.8,
        top_k=10,
        verify_threshold=0.0,
        speculative_draft_len=1,
        knn_cache=cache,
        knn_k=4,
        knn_lambda=0.3,
    )
    assert out.shape[0] == 1 and out.shape[1] >= tok.shape[1]
