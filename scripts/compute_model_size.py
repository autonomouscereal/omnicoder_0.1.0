import os
import json
import torch

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.config import MobilePreset


def bytes_gib(n: int) -> float:
    return float(n) / (1024.0 ** 3)


def main() -> None:
    os.environ.setdefault('PYTHONWARNINGS', 'ignore')
    # Match the benchmark's default mobile_4gb preset
    p = MobilePreset()
    model = OmniTransformer(
        vocab_size=p.vocab_size,
        n_layers=p.n_layers,
        d_model=p.d_model,
        n_heads=p.n_heads,
        mlp_dim=p.mlp_dim,
        n_experts=p.moe_experts,
        top_k=p.moe_top_k,
        max_seq_len=p.max_seq_len,
        use_rope=True,
        kv_latent_dim=p.kv_latent_dim,
        multi_query=p.multi_query,
        multi_token=1,
    )

    params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_bytes = params_bytes + buffers_bytes

    out = {
        'params_bytes': int(params_bytes),
        'buffers_bytes': int(buffers_bytes),
        'total_bytes': int(total_bytes),
        'total_GiB': bytes_gib(total_bytes),
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        model = model.to('cuda')
        after = torch.cuda.memory_allocated()
        out.update({
            'cuda_alloc_bytes': int(after),
            'cuda_alloc_GiB': bytes_gib(after),
            'delta_bytes': int(after - before),
            'delta_GiB': bytes_gib(after - before),
        })
    else:
        out.update({'cuda': False})

    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()


