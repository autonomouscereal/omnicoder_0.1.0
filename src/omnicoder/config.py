from dataclasses import dataclass, field


def get_mobile_preset(name: str):
    """Return a mobile preset dataclass instance by name.

    Supported names: "mobile_4gb", "mobile_2gb".
    """
    name_l = name.lower().strip()
    if name_l == "mobile_4gb":
        return MobilePreset()
    if name_l == "mobile_2gb":
        return MobilePreset2GB()
    # High-capacity sparse MoE variants (≥16 experts/layer) for DS‑MoE curriculum
    if name_l == "mobile_4gb_moe16":
        p = MobilePreset()
        p.name = "mobile_4gb_moe16"
        p.moe_experts = 16
        p.moe_group_sizes = [8, 8]
        p.moe_sub_experts_per = 2
        p.moe_shared_general = 2
        return p
    if name_l == "mobile_2gb_moe16":
        p = MobilePreset2GB()
        p.name = "mobile_2gb_moe16"
        p.moe_experts = 16
        p.moe_group_sizes = [8, 8]
        p.moe_sub_experts_per = 2
        p.moe_shared_general = 2
        return p
    # Larger draft presets for speculative decoding training (2B/3B target parameter scales)
    if name_l == "draft_2b":
        p = MobilePreset()
        p.name = "draft_2b"
        p.n_layers = 24
        p.d_model = 2048
        p.n_heads = 16
        p.mlp_dim = 5504
        p.moe_experts = 8
        p.kv_latent_dim = 256
        p.default_window_size = 4096
        # Disable hierarchical groups for draft presets; flat router over 8 experts
        p.moe_group_sizes = []
        return p
    if name_l == "draft_3b":
        p = MobilePreset()
        p.name = "draft_3b"
        p.n_layers = 28
        p.d_model = 2304
        p.n_heads = 18
        p.mlp_dim = 6144
        p.moe_experts = 8
        p.kv_latent_dim = 256
        p.default_window_size = 4096
        # Disable hierarchical groups for draft presets; flat router over 8 experts
        p.moe_group_sizes = []
        return p
    raise ValueError(f"Unknown mobile preset: {name}")


def get_rope_scale_for_target_ctx(base_max_seq_len: int, target_ctx: int) -> float:
    """Compute a simple RoPE scale factor to extrapolate to a longer target context.

    Our attention precomputes RoPE as sin/cos(freqs) with time index t / rope_scale.
    To extend to a target_ctx > base_max_seq_len, set rope_scale > 1 so the effective
    frequencies are lower and the model can generalize further in time.

    Returns 1.0 if target_ctx <= base_max_seq_len.
    """
    if target_ctx <= 0:
        return 1.0
    if target_ctx <= base_max_seq_len:
        return 1.0
    return float(target_ctx) / float(base_max_seq_len)

def get_rope_interp_base(base: float, scale: float) -> float:
    """Compute an adjusted RoPE base for position interpolation (optional).

    If env OMNICODER_USE_YARN=1 or OMNICODER_USE_PI=1 is set and scale>1,
    adjust base modestly to preserve frequency coverage at longer contexts.
    This is a conservative helper; defaults to returning the original base.
    """
    import os
    try:
        use = (os.getenv('OMNICODER_USE_YARN','0')=='1') or (os.getenv('OMNICODER_USE_PI','0')=='1')
        if not use:
            return float(base)
        s = float(scale)
        if s <= 1.0:
            return float(base)
        # Simple heuristic: compress base by s**0.5 to counteract scaling
        return float(base) / (s ** 0.5)
    except Exception:
        return float(base)

@dataclass
class MoEConfig:
    num_layers: int = 16
    model_dim: int = 2048
    num_heads: int = 16
    mlp_dim: int = 5504
    moe_experts: int = 16
    moe_top_k: int = 2
    dropout: float = 0.0
    max_seq_len: int = 8192
    kv_latent_dim: int = 256  # MLA latent size
    multi_query: bool = True
    multi_token: int = 2      # tokens per step (training head)
    vocab_size: int = 32000


@dataclass
class MobilePreset:
    """Configuration presets for constrained mobile deployment.

    The "mobile_4gb" preset is a conservative default intended to fit in
    ~2–4GB RAM with 4-bit weights and small KV cache when paired with
    int4/gguf or ExecuTorch/CoreML backends.
    """
    name: str = "mobile_4gb"
    # Text core
    n_layers: int = 12
    d_model: int = 1024
    n_heads: int = 8
    mlp_dim: int = 2816
    moe_experts: int = 16
    moe_top_k: int = 2
    # MoE router regularization & dispatch
    router_temp: float = 1.2
    router_jitter: float = 0.2
    router_use_gumbel: bool = True
    router_expert_dropout_p: float = 0.0
    router_sinkhorn_iters: int = 0
    router_sinkhorn_tau: float = 1.0
    moe_capacity_factor: float = 1.2
    moe_static_capacity: int | None = None  # if set, fixes per-expert cap per step
    max_seq_len: int = 8192
    kv_latent_dim: int = 160
    multi_query: bool = True
    vocab_size: int = 32000
    # Quantization/export hints
    weight_bits: int = 4
    activation_bits: int = 8
    group_size: int = 128
    # Attention performance knobs
    use_sdpa: bool = True
    # Hierarchical router grouping (contiguous expert ranges); empty => flat router
    moe_group_sizes: list[int] = field(default_factory=lambda: [8, 8])
    # DeepSeek-style specialization knobs
    moe_sub_experts_per: int = 2
    moe_shared_general: int = 1
    # Default sliding window for infinite-context decode on mobile (0 disables)
    default_window_size: int = 2048
    # Cross-modal verifier defaults
    use_cm_verifier_default: bool = False
    cm_verifier_threshold: float = 0.6


@dataclass
class MobilePreset2GB:
    """More aggressive preset targeting ~2 GB RAM with 4-bit weights.

    Intended for lower-memory devices; reduce depth/width and KV latent size.
    """
    name: str = "mobile_2gb"
    n_layers: int = 8
    d_model: int = 768
    n_heads: int = 8
    mlp_dim: int = 2048
    moe_experts: int = 8
    moe_top_k: int = 2
    router_temp: float = 1.2
    router_jitter: float = 0.2
    router_use_gumbel: bool = True
    router_expert_dropout_p: float = 0.0
    router_sinkhorn_iters: int = 0
    router_sinkhorn_tau: float = 1.0
    moe_capacity_factor: float = 1.2
    moe_static_capacity: int | None = None
    max_seq_len: int = 4096
    kv_latent_dim: int = 128
    multi_query: bool = True
    vocab_size: int = 32000
    weight_bits: int = 4
    activation_bits: int = 8
    group_size: int = 128
    use_sdpa: bool = True
    moe_group_sizes: list[int] = field(default_factory=lambda: [4, 4])
    moe_sub_experts_per: int = 2
    moe_shared_general: int = 1
    default_window_size: int = 1024
    use_cm_verifier_default: bool = False
    cm_verifier_threshold: float = 0.6

@dataclass
class TrainConfig:
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    gradient_checkpointing: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    qlora_4bit: bool = True

@dataclass
class MultiModalConfig:
    use_vision: bool = True
    use_video: bool = True
    use_audio: bool = True
    image_codebook_size: int = 8192
    video_codebook_size: int = 8192
    audio_codebook_size: int = 2048
    # Reserved vocab ranges (contiguous) within the unified tokenizer space
    # Start indices are examples; real systems should ensure no overlap with base text vocab.
    image_vocab_start: int = 32000
    video_vocab_start: int = 32000 + 8192
    audio_vocab_start: int = 32000 + 8192 + 8192
