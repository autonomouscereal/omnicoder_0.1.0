from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from omnicoder.config_2026 import get_omnicoder2026_preset


@dataclass(frozen=True)
class Budget:
    profile: str
    params_b: float
    trunk_params_b: float
    auxiliary_params_b: float
    weight_gib_q4: float
    dense_full_kv_gib_fp16: float
    dense_full_kv_gib_q4: float
    omnicoder_state_gib_estimate: float
    runtime_headroom_gib: float
    total_native_estimate_gib: float
    fits_24gb_native_estimate: bool
    notes: tuple[str, ...]


def estimate_param_breakdown_for_preset(p) -> dict[str, float]:
    d_model = p.d_model
    inner = p.n_heads * p.head_dim
    q_rank = min(int(p.q_lora_rank), d_model)
    o_rank = int(p.o_lora_rank)
    o_groups = max(1, int(p.o_groups))
    pattern = list(p.layer_pattern)
    expanded = [pattern[i % len(pattern)] for i in range(p.n_layers)]
    sparse_layers = sum(1 for x in expanded if x in {"csa", "hca", "csa_hca"})
    local_layers = sum(1 for x in expanded if x == "local")
    kda_layers = sum(1 for x in expanded if x in {"kda", "delta"})

    sparse_attn = sparse_layers * (
        d_model * q_rank
        + q_rank * inner
        + d_model * p.head_dim
        + d_model * p.head_dim
        + d_model
        + max(1, inner * o_rank // o_groups)
        + o_rank * d_model
        + d_model * d_model
    )
    local_attn = local_layers * (4 * d_model * d_model)
    kda = kda_layers * (7 * d_model * d_model + int(p.kda_kernel_size) * d_model)
    mlp = p.n_layers * (3 * d_model * p.mlp_dim)
    embed = p.vocab_size * d_model
    head = 0 if p.tie_embeddings else p.vocab_size * d_model
    norms = p.n_layers * 6 * d_model

    residual_mode = str(getattr(p, "residual_mode", "")).lower()
    if residual_mode in {"block_attnres", "attnres", "attention_residual"}:
        residual_one = 2 * d_model * int(p.block_attnres_rank) + d_model + 1
        residual = p.n_layers * 2 * residual_one
    else:
        residual = 0

    reasoning_slots = max(0, int(getattr(p, "reasoning_slots", 0) or 0))
    reasoning_rank = max(0, min(int(getattr(p, "reasoning_cell_rank", 0) or 0), d_model))
    latent_reasoner = (
        reasoning_slots * d_model
        + 4 * d_model * reasoning_rank
        + 2 * d_model
        + d_model * 5
        + 5
        + 1
    ) if reasoning_slots > 0 and reasoning_rank > 0 else 0

    mtp = int(getattr(p, "mtp_heads", 0)) * p.vocab_size * d_model
    flow = d_model + d_model * int(getattr(p, "flow_latent_dim", 0))
    native_media_feature_dim = int(getattr(p, "native_media_feature_dim", 0))
    native_media_position_dim = int(getattr(p, "native_media_position_dim", 0))
    native_media_type_vocab = int(getattr(p, "native_media_type_vocab", 0))
    native_media = (
        native_media_feature_dim * d_model
        + native_media_position_dim * d_model
        + native_media_type_vocab * d_model
        + d_model
        + d_model
        + d_model * native_media_feature_dim
    )
    grounding_sync = d_model * 8 + 8 + d_model + 1
    final_norm = d_model

    return {
        "sparse_attention": float(sparse_attn),
        "local_attention": float(local_attn),
        "kda": float(kda),
        "mlp": float(mlp),
        "embedding": float(embed),
        "untied_lm_head": float(head),
        "block_norms": float(norms),
        "final_norm": float(final_norm),
        "block_attention_residual": float(residual),
        "adaptive_latent_reasoner": float(latent_reasoner),
        "mtp_heads": float(mtp),
        "flow_head": float(flow),
        "native_media_bridge": float(native_media),
        "grounding_sync_heads": float(grounding_sync),
    }


def estimate_params_for_preset(p) -> float:
    return float(sum(estimate_param_breakdown_for_preset(p).values()))


def estimate_budget(profile: str, context: int = 1_048_576) -> Budget:
    p = get_omnicoder2026_preset(profile)
    breakdown = estimate_param_breakdown_for_preset(p)
    params = float(sum(breakdown.values()))
    auxiliary_keys = {
        "final_norm",
        "block_attention_residual",
        "adaptive_latent_reasoner",
        "mtp_heads",
        "flow_head",
        "native_media_bridge",
        "grounding_sync_heads",
    }
    auxiliary_params = float(sum(value for key, value in breakdown.items() if key in auxiliary_keys))
    trunk_params = params - auxiliary_params
    params_b = params / 1e9
    weight_gib_q4 = params * 0.5 / (1024**3)

    # Full dense KV is the rejected baseline: K and V for every token/layer/head.
    kv_values = int(context) * p.n_layers * p.n_heads * p.head_dim * 2
    dense_full_kv_gib_fp16 = kv_values * 2 / (1024**3)
    dense_full_kv_gib_q4 = kv_values * 0.5 / (1024**3)

    pattern = list(p.layer_pattern)
    expanded = [pattern[i % len(pattern)] for i in range(p.n_layers)]
    csa_layers = sum(1 for x in expanded if x in {"csa", "csa_hca"})
    hca_layers = sum(1 for x in expanded if x == "hca")
    kda_layers = sum(1 for x in expanded if x in {"kda", "delta"})
    csa_blocks = max(1, (int(context) + p.csa_compress_rate - 1) // p.csa_compress_rate)
    hca_blocks = max(1, (int(context) + p.hca_compress_rate - 1) // p.hca_compress_rate)
    resident_slots = csa_blocks * csa_layers + hca_blocks * hca_layers + p.sink_tokens * (csa_layers + hca_layers)
    latent_bytes_q8 = resident_slots * p.head_dim * 2
    kda_state_bytes = kda_layers * p.d_model * 4
    local_window_bytes = (csa_layers + hca_layers) * p.local_window * p.num_key_value_heads * p.head_dim * 2
    omnicoder_state_gib_estimate = (latent_bytes_q8 + kda_state_bytes + local_window_bytes) / (1024**3)
    runtime_headroom_gib = max(2.0, weight_gib_q4 * 0.20)
    total_native = weight_gib_q4 + omnicoder_state_gib_estimate + runtime_headroom_gib
    return Budget(
        profile=p.name,
        params_b=params_b,
        trunk_params_b=trunk_params / 1e9,
        auxiliary_params_b=auxiliary_params / 1e9,
        weight_gib_q4=weight_gib_q4,
        dense_full_kv_gib_fp16=dense_full_kv_gib_fp16,
        dense_full_kv_gib_q4=dense_full_kv_gib_q4,
        omnicoder_state_gib_estimate=omnicoder_state_gib_estimate,
        runtime_headroom_gib=runtime_headroom_gib,
        total_native_estimate_gib=total_native,
        fits_24gb_native_estimate=total_native <= 24.0,
        notes=(
            "Dense full-KV numbers are the rejected baseline.",
            "Parameter estimate includes the shared trunk plus enabled MTP, residual-attention, flow, native-media, grounding, and sync heads.",
            "Adaptive latent reasoning is parameter-shared compute depth; it adds slots/control heads, not another vocab projection.",
            "Native estimate assumes KDA recurrent state plus resident q8-like CSA/HCA shared K=V latent state.",
            "CSA attention computes only a selected prefix/recent sparse gather per query chunk; it does not materialize a 1M x 262k score matrix.",
            "Real 1M serving still needs chunked prefill, paged state, and offload validation.",
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate Omnicoder 2026 native-1M memory budget")
    ap.add_argument("--profile", default="omnicoder2026_20b_1m")
    ap.add_argument("--context", type=int, default=1_048_576)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compact", action="store_true")
    args = ap.parse_args()
    payload = asdict(estimate_budget(args.profile, args.context))
    if args.out:
        from pathlib import Path

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True) if args.compact else json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
