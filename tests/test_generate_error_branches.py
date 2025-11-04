import torch


def test_generate_knn_blend_handles_missing_write_head():
    from omnicoder.inference.generate import build_mobile_model_by_name, generate
    from omnicoder.inference.knn_cache import KNNCache
    model = build_mobile_model_by_name('mobile_4gb', mem_slots=0)
    model.eval()
    tokenizer_dim = getattr(model.ln_f, 'normalized_shape', [model.lm_head.in_features])[0]
    knn = KNNCache(dim=tokenizer_dim)
    ids = torch.randint(0, getattr(model, 'vocab_size', 32000), (1, 4), dtype=torch.long)
    out = generate(model, ids, max_new_tokens=1, knn_cache=knn, kvq='none')
    assert out.shape[1] >= ids.shape[1]


