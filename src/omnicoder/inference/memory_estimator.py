import argparse

from omnicoder.config import MobilePreset


def estimate_model_memory_bytes(preset: MobilePreset) -> int:
    """
    Rough estimate of weight memory with int4 packing + overhead.
    Count only Transformer parameters of OmniTransformer given preset.
    This is a back-of-the-envelope calculator for device budgeting.
    """
    n_layers = preset.n_layers
    d_model = preset.d_model
    n_heads = preset.n_heads
    head_dim = d_model // n_heads
    mlp_dim = preset.mlp_dim
    n_experts = preset.moe_experts

    # Embedding + LM head
    vocab = preset.vocab_size
    params_embed = vocab * d_model
    params_lm_head = d_model * vocab

    # Attention projections per layer: q, k, v, o
    params_attn = (d_model * d_model) * 2  # q and o roughly dxd; k/v are shared in MQA ~ d * head_dim * 2
    params_kv = d_model * head_dim * 2
    # Latent projections per head dim
    kv_lat = preset.kv_latent_dim
    params_latent = (head_dim * kv_lat) * 3 + (kv_lat * head_dim)
    params_attn_total = n_layers * (params_attn + params_kv + params_latent)

    # MoE MLP per layer: experts are not all active, but we store all weights
    params_moe_ff = (d_model * mlp_dim) + (mlp_dim * d_model)
    params_moe_total = n_layers * n_experts * params_moe_ff

    total_params = params_embed + params_lm_head + params_attn_total + params_moe_total

    # int4 weights ~ 0.5 bytes/param; add 15% overhead for scales/metadata
    bytes_weights = int(total_params * 0.5 * 1.15)
    return bytes_weights


def estimate_kv_cache_bytes(preset: MobilePreset, seq_len: int, batch_size: int = 1, kvq: str = 'fp16', group_size: int = 64) -> int:
    """
    Estimate KV cache memory with latent-KV of size kv_latent_dim per head.
    Uses float16 for cache on device accelerators by default (2 bytes), but many
    mobile runtimes keep activations in fp16/fp32. We'll assume fp16.
    """
    n_layers = preset.n_layers
    n_heads = preset.n_heads
    kv_lat = preset.kv_latent_dim
    # fp16(default)=2 bytes, u8=1 byte, nf4=0.5 byte per value (packed as 4 bits)
    if kvq == 'u8':
        bytes_per = 1
    elif kvq == 'nf4':
        bytes_per = 0.5
    else:
        bytes_per = 2
    # store K and V
    per_layer = batch_size * n_heads * seq_len * kv_lat * bytes_per * 2
    return int(n_layers * per_layer)


def human_bytes(num: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq_len', type=int, default=4096)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--preset', type=str, default='mobile_4gb', choices=['mobile_4gb', 'mobile_2gb'])
    ap.add_argument('--kvq', type=str, default='fp16', choices=['fp16','u8','nf4'])
    ap.add_argument('--kvq_group', type=int, default=64)
    args = ap.parse_args()

    if args.preset == 'mobile_4gb':
        preset = MobilePreset()
    else:
        from omnicoder.config import MobilePreset2GB
        preset = MobilePreset2GB()
    w_bytes = estimate_model_memory_bytes(preset)
    kv_bytes = estimate_kv_cache_bytes(preset, args.seq_len, args.batch_size, kvq=args.kvq, group_size=args.kvq_group)
    total = w_bytes + kv_bytes
    print(f"Preset: {preset.name}")
    print(f"Weights (int4 est): {human_bytes(w_bytes)}")
    print(f"KV cache ({args.kvq}, seq={args.seq_len}, B={args.batch_size}): {human_bytes(kv_bytes)}")
    print(f"Total (est): {human_bytes(total)}")


if __name__ == '__main__':
    main()


