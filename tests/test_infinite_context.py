import torch

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.inference.generate import generate


def test_memory_priming_with_window():
    # Tiny model for smoke
    model = OmniTransformer(vocab_size=128, n_layers=2, d_model=64, n_heads=4, mlp_dim=128, mem_slots=4)
    model.eval()
    # Construct a long prompt where prefix > window
    window = 8
    total = 20
    input_ids = torch.randint(0, 127, (1, total), dtype=torch.long)
    # Should run without error and produce new tokens
    out = generate(
        model,
        input_ids,
        max_new_tokens=2,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        verify_threshold=0.0,
        verifier_steps=1,
        speculative_draft_len=0,
        kvq='none',
        kvq_group=64,
        knn_cache=None,
        knn_k=0,
        knn_lambda=0.0,
        window_size=window,
    )
    assert out.shape[1] == total + 2


