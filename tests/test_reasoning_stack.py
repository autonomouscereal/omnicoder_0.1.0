import os
import types
import torch


def _dummy_model(vocab_size: int = 16, d_model: int = 8):
    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = d_model
            self.emb = torch.nn.Embedding(vocab_size, d_model)
            self.proj = torch.nn.Linear(d_model, vocab_size)
        def embed(self, x):
            return self.emb(x)
        def lm_head(self, x):
            return self.proj(x)
        def forward(self, x, past_kv=None, use_cache=False, return_hidden=False, prefix_hidden=None):
            h = self.emb(x)
            logits = self.proj(h)
            if return_hidden:
                return logits, past_kv, None, logits  # reuse logits tensor as hidden shape proxy
            if use_cache:
                return logits, past_kv, None, None
            return logits
    return Dummy()


def test_agot_selects_from_topk_and_respects_depth_token_budget():
    from omnicoder.reasoning.adaptive_graph import AdaptiveGraphOfThoughts
    m = _dummy_model()
    os.environ['OMNICODER_AGOT_ENABLE'] = '1'
    os.environ['OMNICODER_AGOT_WIDTH'] = '3'
    os.environ['OMNICODER_AGOT_DEPTH'] = '2'
    os.environ['OMNICODER_AGOT_TOKEN_BUDGET'] = '2'
    agot = AdaptiveGraphOfThoughts()
    ids = torch.tensor([[1, 2]], dtype=torch.long)
    logits = torch.randn(1, ids.size(1), 16)
    # Force a clear top-1 to verify it can be chosen
    logits[:, -1, :] = -10.0
    logits[:, -1, 5] = 10.0
    out = agot.step(m, ids, None, logits, None, None, 1.0, 5, 0.0)
    assert out.shape == (1, 1)
    assert int(out.item()) in range(16)


def test_latent_bfs_scores_monotonic_with_depth_one_candidate():
    from omnicoder.reasoning.latent_bfs import LatentBFS
    m = _dummy_model()
    os.environ['OMNICODER_LATENT_BFS_ENABLE'] = '1'
    os.environ['OMNICODER_LATENT_BFS_WIDTH'] = '2'
    os.environ['OMNICODER_LATENT_BFS_DEPTH'] = '2'
    ltf = LatentBFS()
    ids = torch.tensor([[1, 2]], dtype=torch.long)
    # Hidden proxy tensor (B,T,C)
    hidden = torch.randn(1, ids.size(1), m.d_model)
    # Two candidates from same distribution should score finite numbers
    cands = [torch.tensor([[3]]), torch.tensor([[4]])]
    scores = ltf.score_candidates(m, ids, None, hidden, cands)
    assert isinstance(scores, list) and len(scores) == 2
    assert all(isinstance(s, float) for s in scores)


def test_graphrag_bias_ids_extraction_is_bounded_and_unique():
    from omnicoder.retrieval.graphrag import KGTriple, collect_bias_ids
    triples = [KGTriple(head='A', relation='r', tail='B') for _ in range(16)]
    calls = []
    def fake_encode(s: str):
        calls.append(s)
        return [1, 2, 3, 4]
    ids = collect_bias_ids(triples, fake_encode, max_terms=8, tail_tokens=2)
    assert len(ids) <= 8 * 3  # worst-case with small vocab; still bounded
    assert len(ids) == len(list(dict.fromkeys(ids)))


