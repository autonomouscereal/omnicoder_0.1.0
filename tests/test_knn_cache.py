import numpy as np
import torch

from omnicoder.inference.knn_cache import KNNCache
from omnicoder.modeling.transformer_moe import OmniTransformer


def test_knn_cache_query_and_blend():
    dim = 32
    vocab = 100
    cache = KNNCache(dim=dim, use_cosine=True, use_faiss=False)
    # Populate with two simple directions mapping to token ids 1 and 2
    h1 = np.zeros((dim,), dtype=np.float32)
    h1[0] = 1.0
    h2 = np.zeros((dim,), dtype=np.float32)
    h2[1] = 1.0
    cache.add(h1, token_id=1)
    cache.add(h2, token_id=2)

    # Query halfway between should give non-zero for both tokens
    q = np.zeros((dim,), dtype=np.float32)
    q[0] = 0.7
    q[1] = 0.7
    probs = cache.query(q, k=2, vocab_size=vocab)
    assert probs.shape == (vocab,)
    assert probs[1] > 0.0 and probs[2] > 0.0
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)


def test_model_return_hidden_decode_step():
    # Tiny model
    model = OmniTransformer(vocab_size=128, n_layers=2, d_model=64, n_heads=4, mlp_dim=128, max_seq_len=32)
    model.eval()
    ids = torch.randint(0, 128, (1, 1), dtype=torch.long)
    # Ask for hidden state during decode-step
    out = model(ids, past_kv=None, use_cache=True, return_hidden=True)
    assert isinstance(out, tuple)
    logits = out[0]
    new_kv = out[1]
    hidden = out[-1]
    assert logits.shape[-1] == 128
    assert isinstance(new_kv, list)
    assert hidden.ndim == 3 and hidden.shape[0] == 1 and hidden.shape[1] == 1 and hidden.shape[2] == 64


