from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextTier:
    name: str
    block_size: int
    retention: str
    cache_dtype: str


def default_context_hierarchy() -> tuple[ContextTier, ...]:
    return (
        ContextTier("gdn2_kda_recurrent_state", 0, "fp32_correctness_then_fp16_or_q8_runtime_state", "fp32_or_q8_state"),
        ContextTier("local_exact", 128, "sliding_window_inside_csa_hca", "q8_or_fp16"),
        ContextTier("compressed_sparse_attention", 4, "shared_k_equals_v_topk_prefix_causal_slots", "fp8_q8_then_q4_latent"),
        ContextTier("heavily_compressed_attention", 128, "shared_k_equals_v_dense_coarse_causal_slots", "q4_or_int2_history"),
        ContextTier("star_attention_anchors", 8192, "repo_file_symbol_tool_state_anchor_cache", "q8_or_fp16"),
        ContextTier("turboquant_cache_branch", 0, "optional_training_free_kv_state_quantization", "3p5_bits_channel_experimental"),
        ContextTier("mhc_depth_residual", 8, "depth_residual_hook", "fp32_gates"),
    )


def hierarchy_manifest() -> dict:
    return {
        "schema": "omnicoder2026_context_hierarchy_v1",
        "tiers": [tier.__dict__ for tier in default_context_hierarchy()],
        "native_context": 1_048_576,
        "full_kv_all_layers": False,
        "stock_gguf_full_1m": False,
    }
