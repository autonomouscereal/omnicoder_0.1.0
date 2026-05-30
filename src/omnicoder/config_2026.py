from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BlockKind = Literal["kda", "csa", "hca", "local", "delta", "csa_hca"]
DEFAULT_LAYER_PATTERN: tuple[BlockKind, ...] = ("kda", "kda", "kda", "csa", "kda", "kda", "kda", "hca")


@dataclass
class Omnicoder2026Preset:
    name: str
    architecture: str = "omnicoder2026_dense_kda_csa_hca_attnres_one_trunk"
    vocab_size: int = 330_000
    n_layers: int = 48
    d_model: int = 4096
    n_heads: int = 32
    head_dim: int = 128
    num_key_value_heads: int = 1
    mlp_dim: int = 15360
    max_seq_len: int = 1_048_576
    train_seq_len: int = 4096
    local_window: int = 128
    csa_block_size: int = 4_096
    csa_top_k_blocks: int = 512
    hca_block_size: int = 131_072
    latent_dim: int = 512
    rope_dim: int = 64
    sink_tokens: int = 4
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    csa_compress_rate: int = 4
    hca_compress_rate: int = 128
    index_head_dim: int = 128
    kda_kernel_size: int = 4
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    residual_mode: str = "block_attnres"
    block_attnres_block_size: int = 128
    block_attnres_max_blocks: int = 1024
    block_attnres_rank: int = 256
    block_attnres_chunk_tokens: int = 2048
    layer_pattern: tuple[BlockKind, ...] = DEFAULT_LAYER_PATTERN
    tie_embeddings: bool = True
    mtp_heads: int = 2
    reasoning_slots: int = 8
    reasoning_max_steps: int = 8
    reasoning_default_steps: int = 0
    reasoning_cell_rank: int = 512
    reasoning_pool_tokens: int = 1024
    reasoning_output_scale: float = 0.05
    flow_latent_dim: int = 1024
    native_media_feature_dim: int = 3072
    native_media_position_dim: int = 4
    native_media_type_vocab: int = 16
    fake_quant: bool = False
    fake_quant_group_size: int = 128
    gguf_bridge_architecture: str = "qwen3-compatible-short-context-bridge"
    native_runtime: str = "omnicoder2026"
    notes: tuple[str, ...] = field(default_factory=tuple)


def get_omnicoder2026_preset(name: str) -> Omnicoder2026Preset:
    key = name.strip().lower().replace("-", "_")
    if key in ("probe", "native1m_probe", "omnicoder2026_native1m_probe"):
        return Omnicoder2026Preset(
            name="omnicoder2026_native1m_probe",
            vocab_size=270_592,
            n_layers=4,
            d_model=512,
            n_heads=8,
            head_dim=64,
            num_key_value_heads=1,
            mlp_dim=1408,
            train_seq_len=1024,
            local_window=1024,
            csa_block_size=512,
            csa_top_k_blocks=16,
            hca_block_size=2048,
            latent_dim=64,
            rope_dim=32,
            sink_tokens=2,
            q_lora_rank=128,
            o_lora_rank=128,
            o_groups=2,
            index_head_dim=32,
            flow_latent_dim=256,
            reasoning_slots=4,
            reasoning_max_steps=4,
            reasoning_default_steps=0,
            reasoning_cell_rank=64,
            reasoning_pool_tokens=128,
            block_attnres_rank=32,
            block_attnres_chunk_tokens=512,
            fake_quant_group_size=64,
            layer_pattern=("kda", "kda", "csa", "hca"),
            notes=("Small native-1M construction/training probe; not a capability target.",),
        )
    if key in ("ledger_probe", "full_ledger_probe", "omnicoder2026_full_ledger_probe"):
        return Omnicoder2026Preset(
            name="omnicoder2026_full_ledger_probe",
            vocab_size=330_000,
            n_layers=4,
            d_model=512,
            n_heads=8,
            head_dim=64,
            num_key_value_heads=1,
            mlp_dim=1408,
            train_seq_len=1024,
            local_window=1024,
            csa_block_size=512,
            csa_top_k_blocks=16,
            hca_block_size=2048,
            latent_dim=64,
            rope_dim=32,
            sink_tokens=2,
            q_lora_rank=128,
            o_lora_rank=128,
            o_groups=2,
            index_head_dim=32,
            flow_latent_dim=256,
            reasoning_slots=4,
            reasoning_max_steps=4,
            reasoning_default_steps=0,
            reasoning_cell_rank=64,
            reasoning_pool_tokens=128,
            block_attnres_rank=32,
            block_attnres_chunk_tokens=512,
            fake_quant_group_size=64,
            layer_pattern=("kda", "kda", "csa", "hca"),
            notes=(
                "Small full-ledger training verifier covering text/control/vision/speech/audio/music/tool/flow IDs.",
                "Use for orchestration learning gates; it is not a capability target.",
            ),
        )
    if key in (
        "target",
        "target_20b",
        "omnicoder2026_20b_1m",
        "omnicoder_20b_1m_dense",
        "dense_omni_24gb",
    ):
        return Omnicoder2026Preset(
            name="omnicoder2026_20b_1m",
            n_layers=64,
            d_model=4096,
            n_heads=32,
            head_dim=128,
            num_key_value_heads=1,
            mlp_dim=15360,
            residual_mode="block_attnres",
            mtp_heads=2,
            q_lora_rank=1024,
            o_lora_rank=1024,
            o_groups=8,
            csa_block_size=4096,
            csa_top_k_blocks=512,
            hca_block_size=131072,
            latent_dim=512,
            flow_latent_dim=1024,
            reasoning_slots=8,
            reasoning_max_steps=8,
            reasoning_default_steps=0,
            reasoning_cell_rank=512,
            reasoning_pool_tokens=1024,
            block_attnres_rank=256,
            block_attnres_chunk_tokens=2048,
            notes=(
                "Primary single-24GB Q4 target: dense 20B-class KDA plus CSA/HCA sparse latent global layers.",
                "Native 1M depends on recurrent state plus compressed sparse/heavily-compressed latent KV, not full GQA KV.",
                "TurboQuant applies to q4 weights plus compressed native state; full training uses sharded/QAT recipes, not a 24GB single-card fp16 pass.",
                "Stock GGUF bridge remains shorter-context; native 1M uses the Omnicoder2026 runtime path.",
            ),
        )
    if key in ("target_7b", "omnicoder2026_7b_1m", "omnicoder_7b_1m_dense"):
        return Omnicoder2026Preset(
            name="omnicoder2026_7b_1m_legacy",
            n_layers=40,
            d_model=3072,
            n_heads=24,
            head_dim=128,
            num_key_value_heads=1,
            mlp_dim=8192,
            csa_block_size=4096,
            csa_top_k_blocks=512,
            hca_block_size=131072,
            latent_dim=384,
            flow_latent_dim=768,
            notes=(
                "Legacy practical profile kept for compatibility and fast experiments.",
                "It is not the current 24GB omnimodal contract target.",
            ),
        )
    if key in ("target_12b", "stretch_12b", "target_16b", "stretch_16b", "omnicoder2026_12b_1m", "omnicoder2026_16b_1m", "omnicoder_12b_1m_dense"):
        return Omnicoder2026Preset(
            name="omnicoder2026_16b_1m",
            notes=(
                "Legacy/intermediate 16B-class native-1M profile: 6 x [kda, kda, kda, csa, kda, kda, kda, hca].",
                "It inherits the 15360-MLP headroom rule and is not the current contract target.",
                "Single-24GB Q4 inference depends on compressed state/offload; prefer RTX 8000 or multi-GPU for validation.",
                "Stock GGUF bridge is a compatibility milestone, not the full native-1M runtime.",
            ),
        )
    if key in ("pilot_3b", "omnicoder2026_3b_pilot"):
        return Omnicoder2026Preset(
            name="omnicoder2026_3b_pilot",
            vocab_size=270_592,
            n_layers=32,
            d_model=2560,
            n_heads=20,
            head_dim=128,
            num_key_value_heads=1,
            mlp_dim=6912,
            train_seq_len=4096,
            local_window=128,
            csa_block_size=2048,
            csa_top_k_blocks=512,
            hca_block_size=65536,
            latent_dim=256,
            rope_dim=64,
            sink_tokens=8,
            flow_latent_dim=512,
            notes=("Pilot scale for data/training/QAT before the 20B contract target.",),
        )
    raise ValueError(f"Unknown Omnicoder2026 preset: {name}")


def preset_to_model_kwargs(preset: Omnicoder2026Preset) -> dict:
    return {
        "vocab_size": preset.vocab_size,
        "n_layers": preset.n_layers,
        "d_model": preset.d_model,
        "n_heads": preset.n_heads,
        "head_dim": preset.head_dim,
        "num_key_value_heads": preset.num_key_value_heads,
        "mlp_dim": preset.mlp_dim,
        "max_seq_len": preset.max_seq_len,
        "local_window": preset.local_window,
        "csa_block_size": preset.csa_block_size,
        "csa_top_k_blocks": preset.csa_top_k_blocks,
        "hca_block_size": preset.hca_block_size,
        "latent_dim": preset.latent_dim,
        "rope_dim": preset.rope_dim,
        "sink_tokens": preset.sink_tokens,
        "q_lora_rank": preset.q_lora_rank,
        "o_lora_rank": preset.o_lora_rank,
        "o_groups": preset.o_groups,
        "csa_compress_rate": preset.csa_compress_rate,
        "hca_compress_rate": preset.hca_compress_rate,
        "index_head_dim": preset.index_head_dim,
        "kda_kernel_size": preset.kda_kernel_size,
        "hc_mult": preset.hc_mult,
        "hc_sinkhorn_iters": preset.hc_sinkhorn_iters,
        "residual_mode": preset.residual_mode,
        "block_attnres_block_size": preset.block_attnres_block_size,
        "block_attnres_max_blocks": preset.block_attnres_max_blocks,
        "block_attnres_rank": preset.block_attnres_rank,
        "block_attnres_chunk_tokens": preset.block_attnres_chunk_tokens,
        "layer_pattern": preset.layer_pattern,
        "tie_embeddings": preset.tie_embeddings,
        "mtp_heads": preset.mtp_heads,
        "reasoning_slots": preset.reasoning_slots,
        "reasoning_max_steps": preset.reasoning_max_steps,
        "reasoning_default_steps": preset.reasoning_default_steps,
        "reasoning_cell_rank": preset.reasoning_cell_rank,
        "reasoning_pool_tokens": preset.reasoning_pool_tokens,
        "reasoning_output_scale": preset.reasoning_output_scale,
        "flow_latent_dim": preset.flow_latent_dim,
        "native_media_feature_dim": preset.native_media_feature_dim,
        "native_media_position_dim": preset.native_media_position_dim,
        "native_media_type_vocab": preset.native_media_type_vocab,
        "fake_quant": preset.fake_quant,
        "fake_quant_group_size": preset.fake_quant_group_size,
    }
