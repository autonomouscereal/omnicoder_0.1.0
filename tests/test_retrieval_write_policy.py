import torch


def test_retrieval_write_policy_executes_with_budget():
    from omnicoder.modeling.transformer_moe import OmniTransformer
    from omnicoder.inference.generate import generate
    from omnicoder.inference.knn_cache import KNNCache

    model = OmniTransformer(vocab_size=96, n_layers=2, d_model=48, n_heads=4, mlp_dim=96, multi_token=1)
    tok = torch.randint(0, 96, (1, 3), dtype=torch.long)
    cache = KNNCache(dim=model.lm_head.in_features, use_faiss=False)

    # Run with windowed decode disabled; ensure no error and cache may receive writes
    out = generate(
        model,
        tok,
        max_new_tokens=8,
        temperature=1.0,
        top_k=5,
        top_p=0.9,
        knn_cache=cache,
        knn_k=2,
        knn_lambda=0.5,
    )
    assert out.shape[0] == 1 and out.shape[1] >= tok.shape[1]
    # KNN cache may remain empty due to randomness, but the path executed without exceptions
