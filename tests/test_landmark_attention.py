import os, torch
from omnicoder.modeling.transformer_moe import OmniTransformer


def test_landmark_attention_fullseq_integration_cpu():
    os.environ['OMNICODER_USE_LANDMARKS'] = '1'
    os.environ['OMNICODER_NUM_LANDMARKS'] = '4'
    m = OmniTransformer(n_layers=2, d_model=128, n_heads=4, mlp_dim=256, n_experts=2, top_k=1, max_seq_len=128)
    m.eval()
    x = torch.randint(0, 100, (1, 16))
    with torch.no_grad():
        y = m(x)
    # Forward in full-seq can return (logits, [img_lat?, aud_lat?, hidden?])
    if isinstance(y, tuple):
        logits = y[0]
    else:
        logits = y
    assert isinstance(logits, torch.Tensor)
    assert logits.shape[:2] == (1, 16)


def test_landmark_attention_decode_step_safe():
    os.environ['OMNICODER_USE_LANDMARKS'] = '1'
    os.environ['OMNICODER_NUM_LANDMARKS'] = '4'
    m = OmniTransformer(n_layers=2, d_model=128, n_heads=4, mlp_dim=256, n_experts=2, top_k=1, max_seq_len=128)
    m.eval()
    x = torch.randint(0, 100, (1, 1))
    with torch.no_grad():
        logits, new_kv, *_ = m(x, past_kv=None, use_cache=True)
    assert isinstance(logits, torch.Tensor)
    assert isinstance(new_kv, list)
    assert len(new_kv) == len(m.blocks)


