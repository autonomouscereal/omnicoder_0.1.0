from __future__ import annotations

import math
import os
import contextlib
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from omnicoder.modeling.kda_2026 import GatedDeltaNet2
from omnicoder.tokenization.omni_ledger_2026 import DEFAULT_LEDGER

BlockKind = Literal["kda", "csa", "hca", "local", "delta", "csa_hca"]
DEFAULT_LAYER_PATTERN: tuple[BlockKind, ...] = ("kda", "kda", "kda", "csa", "kda", "kda", "kda", "hca")


@contextlib.contextmanager
def _default_device_scope(device: torch.device | None):
    if device is None:
        yield
        return
    previous = torch.get_default_device()
    torch.set_default_device(device)
    try:
        yield
    finally:
        torch.set_default_device(previous)


@dataclass
class OmniCoder2026Config:
    """Dense native-1M Omnicoder trunk.

    The 2026 rebuild is a dense one-trunk architecture. It borrows verified
    primitives from DeepSeek V4, Kimi Linear, Attention Residuals/MoDA, NVIDIA
    long-context runtime work, and current omnimodal systems without adopting a
    MoE router as the core model.
    """

    vocab_size: int = 330_000
    n_layers: int = 48
    d_model: int = 4096
    n_heads: int = 32
    head_dim: int = 128
    num_key_value_heads: int = 1
    mlp_dim: int = 15_360
    max_seq_len: int = 1_048_576

    # DeepSeek V4 verified shape: short local window + compressed sparse slots.
    local_window: int = 128
    csa_block_size: int = 4096
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

    # Kimi Linear/KDA-style recurrent-linear layers.
    kda_kernel_size: int = 4
    kda_state_dtype: str = "fp16"

    # Residual attention hooks. block_attnres is the active native-1M path:
    # it attends each residual update to compressed causal block summaries of
    # the residual stream instead of retaining full per-layer hidden history.
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    residual_mode: str = "block_attnres"
    block_attnres_block_size: int = 128
    block_attnres_max_blocks: int = 1024
    block_attnres_rank: int = 256
    block_attnres_chunk_tokens: int = 2048

    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    initializer_std: float = 0.02
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
    token_ranges: dict[str, tuple[int, int]] = field(default_factory=DEFAULT_LEDGER.as_config_ranges)

    # Deployment metadata. These are surfaced in manifests and profile exports.
    weight_quant_target: str = "q4_groupwise"
    kv_quant_target: str = "fp8_or_int8_mla_latents; oscar_int2_history_experimental"
    gguf_bridge_architecture: str = "qwen3_compatible_short_context_bridge"

    @property
    def global_block_size(self) -> int:
        return self.csa_block_size

    @property
    def global_top_k_blocks(self) -> int:
        return self.csa_top_k_blocks

    @property
    def mla_latent_dim(self) -> int:
        return self.latent_dim

    @classmethod
    def probe(cls) -> "OmniCoder2026Config":
        return cls(
            vocab_size=270_592,
            n_layers=4,
            d_model=512,
            n_heads=8,
            head_dim=64,
            num_key_value_heads=1,
            mlp_dim=1408,
            local_window=128,
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
        )

    @classmethod
    def pilot_3b(cls) -> "OmniCoder2026Config":
        return cls(
            vocab_size=270_592,
            n_layers=32,
            d_model=2560,
            n_heads=20,
            head_dim=128,
            num_key_value_heads=1,
            mlp_dim=6912,
            local_window=128,
            csa_block_size=2048,
            csa_top_k_blocks=512,
            hca_block_size=65_536,
            latent_dim=256,
            flow_latent_dim=512,
        )

    @classmethod
    def target_7b(cls) -> "OmniCoder2026Config":
        return cls(
            n_layers=40,
            d_model=3072,
            n_heads=24,
            head_dim=128,
            num_key_value_heads=1,
            mlp_dim=8192,
            local_window=128,
            csa_block_size=4096,
            csa_top_k_blocks=512,
            hca_block_size=131_072,
            latent_dim=384,
            flow_latent_dim=768,
        )

    @classmethod
    def target_20b(cls) -> "OmniCoder2026Config":
        return cls(
            n_layers=64,
            d_model=4096,
            n_heads=32,
            head_dim=128,
            num_key_value_heads=1,
            mlp_dim=15_360,
            local_window=128,
            csa_block_size=4096,
            csa_top_k_blocks=512,
            hca_block_size=131_072,
            latent_dim=512,
            q_lora_rank=1024,
            o_lora_rank=1024,
            o_groups=8,
            flow_latent_dim=1024,
            reasoning_slots=8,
            reasoning_max_steps=8,
            reasoning_default_steps=0,
            reasoning_cell_rank=512,
            reasoning_pool_tokens=1024,
            residual_mode="block_attnres",
            block_attnres_rank=256,
            block_attnres_chunk_tokens=2048,
            mtp_heads=2,
        )

    @classmethod
    def target_16b(cls) -> "OmniCoder2026Config":
        return cls(
            n_layers=48,
            d_model=4096,
            n_heads=32,
            head_dim=128,
            num_key_value_heads=1,
            mlp_dim=15_360,
            local_window=128,
            csa_block_size=4096,
            csa_top_k_blocks=512,
            hca_block_size=131_072,
            latent_dim=512,
            flow_latent_dim=1024,
            reasoning_slots=8,
            reasoning_max_steps=8,
            reasoning_default_steps=0,
            reasoning_cell_rank=512,
            reasoning_pool_tokens=1024,
        )


class _FakeQuantWeightSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, w: torch.Tensor, group_size: int) -> torch.Tensor:
        return _fake_quant_weight_value(w, int(group_size))

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


def _fake_quant_weight_value(w: torch.Tensor, group_size: int) -> torch.Tensor:
        group_size = int(group_size)
        if group_size <= 0 or w.numel() == 0:
            return w
        orig_shape = w.shape
        flat = w.reshape(w.shape[0], -1)
        groups = math.ceil(flat.shape[1] / group_size)
        pad = groups * group_size - flat.shape[1]
        if pad:
            flat = F.pad(flat, (0, pad))
        grouped = flat.reshape(flat.shape[0], groups, group_size)
        scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
        dq = grouped.div(scale)
        dq.round_()
        dq.clamp_(-7, 7)
        dq.mul_(scale)
        dq = dq.reshape(flat.shape[0], groups * group_size)
        if pad:
            dq = dq[:, :-pad]
        return dq.reshape(orig_shape)


def _fake_quant_weight(w: torch.Tensor, group_size: int) -> torch.Tensor:
    return _FakeQuantWeightSTE.apply(w, int(group_size))


class _ChunkedFakeQuantLinearSTE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        group_size: int,
        chunk_rows: int,
    ) -> torch.Tensor:
        ctx.group_size = int(group_size)
        ctx.chunk_rows = max(1, int(chunk_rows))
        ctx.has_bias = bias is not None
        ctx.save_for_backward(x, weight)
        x_flat = x.reshape(-1, x.shape[-1])
        out_features = int(weight.shape[0])
        output = x_flat.new_empty((x_flat.shape[0], out_features))
        for start in range(0, out_features, ctx.chunk_rows):
            end = min(out_features, start + ctx.chunk_rows)
            q_weight = _fake_quant_weight_value(weight[start:end], ctx.group_size)
            q_bias = bias[start:end] if bias is not None else None
            output[:, start:end] = F.linear(x_flat, q_weight, q_bias)
        return output.reshape((*x.shape[:-1], out_features))

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None, None]:
        saved = ctx.saved_tensors
        x = saved[0]
        weight = saved[1]
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
        x_flat = x.reshape(-1, x.shape[-1])
        chunk_rows = max(1, int(ctx.chunk_rows))
        group_size = int(ctx.group_size)
        needs_x, needs_weight, needs_bias = ctx.needs_input_grad[:3]

        grad_x: torch.Tensor | None = None
        if needs_x:
            grad_x_flat = torch.zeros_like(x_flat)
            for start in range(0, weight.shape[0], chunk_rows):
                end = min(int(weight.shape[0]), start + chunk_rows)
                q_weight = _fake_quant_weight_value(weight[start:end], group_size)
                grad_x_flat.add_(grad_output_flat[:, start:end].matmul(q_weight))
            grad_x = grad_x_flat.reshape_as(x)

        grad_weight: torch.Tensor | None = None
        if needs_weight:
            grad_weight = torch.empty_like(weight)
            for start in range(0, weight.shape[0], chunk_rows):
                end = min(int(weight.shape[0]), start + chunk_rows)
                grad_weight[start:end] = grad_output_flat[:, start:end].transpose(0, 1).matmul(x_flat)

        grad_bias: torch.Tensor | None = None
        if needs_bias and ctx.has_bias:
            grad_bias = grad_output_flat.sum(dim=0)
        return grad_x, grad_weight, grad_bias, None, None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


class QuantAwareLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, *, fake_quant: bool = False, group_size: int = 128):
        super().__init__(in_features, out_features, bias=bias)
        self.fake_quant = bool(fake_quant)
        self.group_size = int(group_size)
        self.fake_quant_chunk_rows = max(0, _env_int("OMNICODER2026_FAKE_QUANT_CHUNK_ROWS", 0))
        self.fake_quant_max_full_elements = max(0, _env_int("OMNICODER2026_FAKE_QUANT_MAX_FULL_ELEMENTS", 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fake_quant and self.fake_quant_chunk_rows > 0:
            max_full = int(self.fake_quant_max_full_elements)
            if max_full <= 0 or self.weight.numel() > max_full:
                return self._chunked_fake_quant_linear(x)
        weight = _fake_quant_weight(self.weight, self.group_size) if self.fake_quant else self.weight
        return F.linear(x, weight, self.bias)

    def _chunked_fake_quant_linear(self, x: torch.Tensor) -> torch.Tensor:
        rows = max(1, int(self.fake_quant_chunk_rows))
        return _ChunkedFakeQuantLinearSTE.apply(x, self.weight, self.bias, self.group_size, rows)


def reset_omnicoder2026_parameters(module: nn.Module, cfg: OmniCoder2026Config) -> None:
    std = float(getattr(cfg, "initializer_std", 0.02) or 0.02)
    for child in module.modules():
        if isinstance(child, nn.Embedding):
            nn.init.normal_(child.weight, mean=0.0, std=std)
        elif isinstance(child, QuantAwareLinear):
            nn.init.normal_(child.weight, mean=0.0, std=std)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, AdaptiveLatentReasoner):
            nn.init.normal_(child.slot_embeddings, mean=0.0, std=std)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 1_000_000.0):
        super().__init__()
        self.dim = int(dim)
        self.base = float(base)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / max(1, self.dim)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        t = x.shape[-2]
        if positions is None:
            positions = torch.arange(t, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions.to(self.inv_freq.dtype), self.inv_freq.to(x.device))
        emb = torch.cat((freqs, freqs), dim=-1).to(dtype=x.dtype)
        return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = cos.shape[-1]
    x_rot, x_pass = x[..., :d], x[..., d:]
    x1 = x_rot[..., ::2]
    x2 = x_rot[..., 1::2]
    rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
    out = (x_rot * cos) + (rot * sin)
    return torch.cat((out, x_pass), dim=-1) if x_pass.numel() else out


def apply_rope_tail(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = cos.shape[-1]
    if d <= 0:
        return x
    x_pass, x_rot = x[..., :-d], x[..., -d:]
    x1 = x_rot[..., ::2]
    x2 = x_rot[..., 1::2]
    rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
    out = (x_rot * cos) + (rot * sin)
    return torch.cat((x_pass, out), dim=-1) if x_pass.numel() else out


class SwiGLU(nn.Module):
    def __init__(self, cfg: OmniCoder2026Config):
        super().__init__()
        lin = lambda i, o: QuantAwareLinear(i, o, bias=False, fake_quant=cfg.fake_quant, group_size=cfg.fake_quant_group_size)
        self.gate = lin(cfg.d_model, cfg.mlp_dim)
        self.up = lin(cfg.d_model, cfg.mlp_dim)
        self.down = lin(cfg.mlp_dim, cfg.d_model)
        self.chunk_tokens = max(0, _env_int("OMNICODER2026_FFN_CHUNK_TOKENS", 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chunk_tokens > 0 and x.shape[-2] > self.chunk_tokens:
            output: torch.Tensor | None = None
            for start in range(0, x.shape[-2], self.chunk_tokens):
                end = min(x.shape[-2], start + self.chunk_tokens)
                x_chunk = x[..., start:end, :]
                piece = self.down(F.silu(self.gate(x_chunk)) * self.up(x_chunk))
                if output is None:
                    output = piece.new_empty((*piece.shape[:-2], x.shape[-2], piece.shape[-1]))
                output[..., start:end, :] = piece
            if output is not None:
                return output
        return self.down(F.silu(self.gate(x)) * self.up(x))


class GatedDeltaLayer(nn.Module):
    """KDA/Gated-DeltaNet-2 dense recurrent-linear mixer.

    This path keeps the correctness oracle in plain PyTorch and fp32 recurrent
    state. A production path can replace it with chunkwise KDA/GDN2 kernels
    without changing the surrounding block contract.
    """

    def __init__(self, cfg: OmniCoder2026Config):
        super().__init__()
        kernel_size = int(cfg.kda_kernel_size)
        lin = lambda i, o: QuantAwareLinear(i, o, bias=False, fake_quant=cfg.fake_quant, group_size=cfg.fake_quant_group_size)
        self.pre_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.depthwise = nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size, padding=kernel_size - 1, groups=cfg.d_model)
        self.gdn2 = GatedDeltaNet2(d_model=cfg.d_model, n_heads=cfg.n_heads, head_dim=cfg.head_dim)
        self.gate_proj = QuantAwareLinear(cfg.d_model, cfg.d_model, bias=True, fake_quant=cfg.fake_quant, group_size=cfg.fake_quant_group_size)
        self.out_proj = lin(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.depthwise(x.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        recurrent = self.gdn2(self.pre_norm(torch.tanh(conv)))
        return self.out_proj(recurrent * torch.sigmoid(self.gate_proj(x)))


class LocalCausalAttention(nn.Module):
    def __init__(self, cfg: OmniCoder2026Config):
        super().__init__()
        self.cfg = cfg
        inner = cfg.n_heads * cfg.head_dim
        lin = lambda i, o: QuantAwareLinear(i, o, bias=False, fake_quant=cfg.fake_quant, group_size=cfg.fake_quant_group_size)
        self.q_proj = lin(cfg.d_model, inner)
        self.k_proj = lin(cfg.d_model, inner)
        self.v_proj = lin(cfg.d_model, inner)
        self.o_proj = lin(inner, cfg.d_model)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.rope = RotaryEmbedding(min(cfg.rope_dim, cfg.head_dim))

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)

    def _local_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int) -> torch.Tensor:
        t = q.shape[2]
        if t <= window:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        parts: list[torch.Tensor] = []
        for start in range(0, t, window):
            end = min(t, start + window)
            left = max(0, start - window)
            qi = q[:, :, start:end, :]
            ki = k[:, :, left:end, :]
            vi = v[:, :, left:end, :]
            mask = torch.ones((end - start, end - left), dtype=torch.bool, device=q.device).tril(diagonal=start - left)
            parts.append(F.scaled_dot_product_attention(qi, ki, vi, attn_mask=mask, dropout_p=0.0))
        return torch.cat(parts, dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_norm(self._shape(self.q_proj(x)))
        k = self.k_norm(self._shape(self.k_proj(x)))
        v = self._shape(self.v_proj(x))
        cos, sin = self.rope(q)
        q = apply_rope(q, cos.view(1, 1, t, -1), sin.view(1, 1, t, -1))
        k = apply_rope(k, cos.view(1, 1, t, -1), sin.view(1, 1, t, -1))
        y = self._local_attention(q, k, v, int(self.cfg.local_window))
        return self.o_proj(y.transpose(1, 2).contiguous().view(b, t, self.cfg.n_heads * self.cfg.head_dim))


class SparseLatentAttention(nn.Module):
    """DeepSeek-V4-style CSA or HCA layer with exact local tokens.

    CSA layers keep medium-grain compressed sparse summaries and select a
    top-k/prefix causal set. HCA layers keep heavily compressed low-cost
    summaries. Both preserve a short exact local path.
    """

    def __init__(self, cfg: OmniCoder2026Config, mode: Literal["csa", "hca"]):
        super().__init__()
        if mode not in {"csa", "hca"}:
            raise ValueError(f"unknown sparse latent mode: {mode}")
        self.cfg = cfg
        self.mode = mode
        inner = cfg.n_heads * cfg.head_dim
        lin = lambda i, o: QuantAwareLinear(i, o, bias=False, fake_quant=cfg.fake_quant, group_size=cfg.fake_quant_group_size)
        q_rank = max(1, min(int(cfg.q_lora_rank), cfg.d_model))
        o_rank = max(1, int(cfg.o_lora_rank))
        self.q_a_proj = lin(cfg.d_model, q_rank)
        self.q_a_norm = RMSNorm(q_rank, cfg.rms_norm_eps)
        self.q_b_proj = lin(q_rank, inner)
        self.kv_proj = lin(cfg.d_model, cfg.head_dim)
        self.compress_kv_proj = lin(cfg.d_model, cfg.head_dim)
        self.compress_gate_proj = QuantAwareLinear(cfg.d_model, 1, bias=True, fake_quant=cfg.fake_quant, group_size=cfg.fake_quant_group_size)
        self.o_groups = max(1, min(int(cfg.o_groups), int(cfg.n_heads)))
        if inner % self.o_groups == 0 and o_rank % self.o_groups == 0 and self.o_groups > 1:
            self.o_inner_per_group = inner // self.o_groups
            self.o_rank_per_group = o_rank // self.o_groups
            self.o_a_groups = nn.ModuleList(
                [lin(self.o_inner_per_group, self.o_rank_per_group) for _ in range(self.o_groups)]
            )
            self.o_a_proj = None
        else:
            self.o_inner_per_group = inner
            self.o_rank_per_group = o_rank
            self.o_a_groups = None
            self.o_a_proj = lin(inner, o_rank)
        self.o_b_proj = lin(o_rank, cfg.d_model)
        self.global_gate = lin(cfg.d_model, cfg.d_model)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.sink_logits = nn.Parameter(torch.zeros(cfg.n_heads, max(1, cfg.sink_tokens)))
        self.rope = RotaryEmbedding(min(cfg.rope_dim, cfg.head_dim))

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)

    def _shape_mqa(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, 1, t, self.cfg.head_dim)

    def _summarize(self, x: torch.Tensor, block_size: int) -> torch.Tensor:
        c = self.compress_kv_proj(x)
        gate = self.compress_gate_proj(x)
        b, t, d = c.shape
        block_size = max(1, int(block_size))
        blocks = max(1, math.ceil(t / block_size))
        pad = blocks * block_size - t
        if pad:
            c = F.pad(c, (0, 0, 0, pad))
            gate = F.pad(gate, (0, 0, 0, pad), value=-1e4)
        c = c.view(b, blocks, block_size, d)
        gate = gate.view(b, blocks, block_size, 1).float()
        weights = torch.softmax(gate, dim=2).to(dtype=c.dtype)
        return (c * weights).sum(dim=2)

    def _local_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if k.shape[1] == 1 and q.shape[1] > 1:
            k = k.expand(-1, q.shape[1], -1, -1)
            v = v.expand(-1, q.shape[1], -1, -1)
        t = q.shape[2]
        window = max(1, int(self.cfg.local_window))
        if t <= window:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        parts: list[torch.Tensor] = []
        for start in range(0, t, window):
            end = min(t, start + window)
            left = max(0, start - window)
            qi = q[:, :, start:end, :]
            ki = k[:, :, left:end, :]
            vi = v[:, :, left:end, :]
            mask = torch.ones((end - start, end - left), dtype=torch.bool, device=q.device).tril(diagonal=start - left)
            parts.append(F.scaled_dot_product_attention(qi, ki, vi, attn_mask=mask, dropout_p=0.0))
        return torch.cat(parts, dim=2)

    def _global_mask(self, t: int, n_blocks: int, block_size: int, device: torch.device) -> torch.Tensor:
        q_block = torch.div(torch.arange(t, device=device), int(block_size), rounding_mode="floor")
        block_idx = torch.arange(n_blocks, device=device)
        # Compressed summaries are block-level aggregates. The current block can
        # contain future tokens, so only completed previous blocks are legal in
        # the global path; the current block is covered by local causal attention.
        causal = block_idx.view(1, -1) < q_block.view(-1, 1)
        if self.mode == "csa":
            top_k = min(max(1, int(self.cfg.csa_top_k_blocks)), n_blocks)
            if top_k < n_blocks:
                distance = q_block.view(-1, 1) - block_idx.view(1, -1)
                scores = (-distance).masked_fill(~causal, -(t + n_blocks))
                recent = torch.topk(scores, k=top_k, dim=-1).indices
                sparse = torch.zeros_like(causal)
                sparse.scatter_(1, recent, True)
                prefix = block_idx.view(1, -1) < min(4, n_blocks)
                causal = causal & (sparse | prefix)
        return causal

    def _selected_blocks(self, start: int, end: int, n_blocks: int, block_size: int, device: torch.device) -> torch.Tensor:
        first_block = start // max(1, block_size)
        last_block = max(0, (end - 1) // max(1, block_size))
        if self.mode == "csa":
            top_k = min(max(1, int(self.cfg.csa_top_k_blocks)), n_blocks)
            left = max(0, first_block - top_k + 1)
            recent = torch.arange(left, min(n_blocks, last_block + 1), device=device)
            prefix = torch.arange(0, min(4, n_blocks), device=device)
            if recent.numel() == 0:
                return prefix
            return torch.unique(torch.cat((prefix, recent)), sorted=True)
        return torch.arange(0, min(n_blocks, last_block + 1), device=device)

    def _global_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int) -> torch.Tensor:
        t = q.shape[2]
        n_blocks = k.shape[2]
        chunk = max(1, int(self.cfg.local_window))
        parts: list[torch.Tensor] = []
        for start in range(0, t, chunk):
            end = min(t, start + chunk)
            selected = self._selected_blocks(start, end, n_blocks, block_size, q.device)
            k_sel = k.index_select(2, selected)
            v_sel = v.index_select(2, selected)
            positions = torch.arange(start, end, device=q.device)
            q_block = torch.div(positions, int(block_size), rounding_mode="floor")
            # Strictly exclude the query token's current compressed block. That
            # summary may include future tokens inside the block and would let
            # teacher-forced loss cheat while autoregressive decode fails.
            mask = selected.view(1, -1) < q_block.view(-1, 1)
            parts.append(self._sink_attention(q[:, :, start:end, :], k_sel, v_sel, mask))
        return torch.cat(parts, dim=2)

    def _o_a(self, y: torch.Tensor) -> torch.Tensor:
        if self.o_a_groups is None:
            assert self.o_a_proj is not None
            return self.o_a_proj(y)
        grouped = y.view(*y.shape[:-1], self.o_groups, self.o_inner_per_group)
        return torch.cat([proj(grouped[..., i, :]) for i, proj in enumerate(self.o_a_groups)], dim=-1)

    def _sink_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(float(self.cfg.head_dim))
        scores = torch.einsum("bhtd,bksd->bhts", q, k) * scale
        scores = scores.masked_fill(~mask.view(1, 1, mask.shape[0], mask.shape[1]), torch.finfo(scores.dtype).min)
        max_scores = scores.amax(dim=-1, keepdim=True)
        sink = self.sink_logits.to(dtype=scores.dtype, device=scores.device).view(1, self.cfg.n_heads, 1, -1)
        sink_max = sink.amax(dim=-1, keepdim=True)
        denom_max = torch.maximum(max_scores, sink_max)
        exp_scores = torch.exp(scores - denom_max)
        exp_sinks = torch.exp(sink - denom_max).sum(dim=-1, keepdim=True)
        probs = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + exp_sinks).clamp_min(1e-9)
        return torch.einsum("bhts,bksd->bhtd", probs, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_norm(self._shape(self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))))
        local_kv = self.k_norm(self._shape_mqa(self.kv_proj(x)))
        cos, sin = self.rope(q)
        q = apply_rope_tail(q, cos.view(1, 1, t, -1), sin.view(1, 1, t, -1))
        local_kv = apply_rope_tail(local_kv, cos.view(1, 1, t, -1), sin.view(1, 1, t, -1))
        local_y = self._local_attention(q, local_kv, local_kv)

        block_size = int(self.cfg.csa_compress_rate if self.mode == "csa" else self.cfg.hca_compress_rate)
        summary = self._summarize(x, block_size)
        k = self.k_norm(self._shape_mqa(summary))
        ccos, csin = self.rope(k)
        k = apply_rope_tail(k, ccos.view(1, 1, summary.shape[1], -1), csin.view(1, 1, summary.shape[1], -1))
        v = k
        global_y = self._global_attention(q, k, v, block_size)

        local_y = local_y.transpose(1, 2).contiguous().view(b, t, self.cfg.n_heads * self.cfg.head_dim)
        global_y = global_y.transpose(1, 2).contiguous().view(b, t, self.cfg.n_heads * self.cfg.head_dim)
        gate = torch.sigmoid(self.global_gate(x))
        return self.o_b_proj(self._o_a(local_y * (1.0 - gate) + global_y * gate))


class MHCResidual(nn.Module):
    """Lightweight mHC/depth-residual hook.

    The parameterized gate gives the block a trainable residual path now, while
    leaving room for a production Sinkhorn/hyper-connection kernel later.
    """

    def __init__(self, cfg: OmniCoder2026Config):
        super().__init__()
        self.enabled = int(cfg.hc_mult) > 1 and cfg.residual_mode != "plain"
        self.gate = QuantAwareLinear(cfg.d_model, cfg.d_model, bias=True, fake_quant=cfg.fake_quant, group_size=cfg.fake_quant_group_size)
        self.scale = nn.Parameter(torch.tensor([1.0 / math.sqrt(max(1, int(cfg.hc_mult)))]))

    def forward(self, x: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        if update.dtype != x.dtype:
            update = update.to(dtype=x.dtype)
        if not self.enabled:
            return x + update
        scale = self.scale.to(device=x.device, dtype=x.dtype)
        gate = torch.sigmoid(self.gate(x)).to(dtype=x.dtype)
        return x + scale * gate * update


class BlockAttentionResidual(nn.Module):
    """Memory-bounded residual-attention update.

    Full residual attention over every previous layer/token state is too
    expensive for the native-1M target. This module keeps the intelligence
    signal of learned residual selection by attending each update to compressed
    causal block summaries from the incoming residual stream. It adds no
    per-token KV cache and its context is bounded by
    ``block_attnres_max_blocks``.
    """

    def __init__(self, cfg: OmniCoder2026Config):
        super().__init__()
        self.enabled = str(cfg.residual_mode).lower() in {"block_attnres", "attnres", "attention_residual"}
        self.block_size = max(1, int(cfg.block_attnres_block_size))
        self.max_blocks = max(1, int(cfg.block_attnres_max_blocks))
        self.rank = max(1, int(cfg.block_attnres_rank))
        self.chunk_tokens = max(1, int(cfg.block_attnres_chunk_tokens))
        self.q = QuantAwareLinear(cfg.d_model, self.rank, bias=False, fake_quant=cfg.fake_quant, group_size=cfg.fake_quant_group_size)
        self.k = QuantAwareLinear(cfg.d_model, self.rank, bias=False, fake_quant=cfg.fake_quant, group_size=cfg.fake_quant_group_size)
        self.update_gate = QuantAwareLinear(cfg.d_model, 1, bias=True, fake_quant=cfg.fake_quant, group_size=cfg.fake_quant_group_size)
        self.scale = nn.Parameter(torch.tensor([0.0]))

    def _block_summaries(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        block = self.block_size
        pad = (block - (t % block)) % block
        if pad:
            x_pad = F.pad(x, (0, 0, 0, pad))
        else:
            x_pad = x
        summaries = x_pad.view(b, -1, block, d).mean(dim=2)
        positions = torch.arange(summaries.shape[1], device=x.device)
        if summaries.shape[1] > self.max_blocks:
            prefix = summaries[:, :1, :]
            tail = summaries[:, -(self.max_blocks - 1):, :] if self.max_blocks > 1 else summaries[:, :0, :]
            summaries = torch.cat((prefix, tail), dim=1)
            tail_positions = positions[-(self.max_blocks - 1):] if self.max_blocks > 1 else positions[:0]
            positions = torch.cat((positions[:1], tail_positions), dim=0)
        return summaries, positions

    def forward(self, x: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        if update.dtype != x.dtype:
            update = update.to(dtype=x.dtype)
        if not self.enabled:
            return x + update
        summaries, summary_positions = self._block_summaries(x)
        q = self.q(update)
        k = self.k(summaries)
        block_positions = summary_positions.view(1, 1, -1)
        residual_chunks: list[torch.Tensor] = []
        chunk_tokens = max(1, _env_int("OMNICODER2026_BLOCK_ATTNRES_CHUNK_TOKENS", self.chunk_tokens))
        for start in range(0, x.shape[1], chunk_tokens):
            end = min(x.shape[1], start + chunk_tokens)
            q_chunk = q[:, start:end, :]
            scores = torch.einsum("btd,bsd->bts", q_chunk, k) / math.sqrt(max(1, q.shape[-1]))
            token_blocks = torch.div(
                torch.arange(start, end, device=x.device),
                self.block_size,
                rounding_mode="floor",
            ).view(1, -1, 1)
            scores = scores.masked_fill(block_positions > token_blocks, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
            residual_chunks.append(torch.einsum("bts,bsd->btd", weights, summaries))
        residual_context = torch.cat(residual_chunks, dim=1) if residual_chunks else update.new_zeros(update.shape)
        gate = torch.sigmoid(self.update_gate(x)).to(dtype=x.dtype)
        scale = self.scale.to(device=x.device, dtype=x.dtype)
        return x + update + torch.tanh(scale) * gate * residual_context


class NativeContinuousMediaBridge(nn.Module):
    """Shared SenseNova-style continuous media bridge.

    Edge preprocessing may patchify pixels, stack video frame patches, window
    waveform/spectrogram samples, or pack OCR document crops. The trunk sees
    all of them through this one shared feature projection plus type/time/grid
    metadata. There are no modality-specific learned encoders in this path.
    """

    def __init__(self, cfg: OmniCoder2026Config):
        super().__init__()
        self.feature_dim = int(cfg.native_media_feature_dim)
        self.position_dim = int(cfg.native_media_position_dim)
        self.feature_proj = QuantAwareLinear(self.feature_dim, cfg.d_model, bias=False, fake_quant=cfg.fake_quant, group_size=cfg.fake_quant_group_size)
        self.position_proj = QuantAwareLinear(self.position_dim, cfg.d_model, bias=False, fake_quant=cfg.fake_quant, group_size=cfg.fake_quant_group_size)
        self.type_embed = nn.Embedding(int(cfg.native_media_type_vocab), cfg.d_model)
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.reconstruction_head = nn.Sequential(
            RMSNorm(cfg.d_model, cfg.rms_norm_eps),
            QuantAwareLinear(cfg.d_model, self.feature_dim, bias=False, fake_quant=cfg.fake_quant, group_size=cfg.fake_quant_group_size),
        )

    def _fit_last_dim(self, value: torch.Tensor, size: int) -> torch.Tensor:
        if value.shape[-1] == size:
            return value
        if value.shape[-1] > size:
            return value[..., :size]
        return F.pad(value, (0, size - value.shape[-1]))

    def embed(
        self,
        features: torch.Tensor,
        type_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.dim() != 3:
            raise ValueError(f"native media features must be [batch, tokens, channels], got {tuple(features.shape)}")
        features = self._fit_last_dim(features, self.feature_dim)
        x = self.feature_proj(features)
        if type_ids is None:
            type_ids = torch.zeros(features.shape[:2], dtype=torch.long, device=features.device)
        x = x + self.type_embed(type_ids.to(features.device).long().remainder(self.type_embed.num_embeddings))
        if positions is None:
            positions = torch.zeros((*features.shape[:2], self.position_dim), dtype=features.dtype, device=features.device)
        positions = self._fit_last_dim(positions.to(features.device, dtype=features.dtype), self.position_dim)
        x = x + self.position_proj(positions)
        return self.norm(x)

    def reconstruct(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.reconstruction_head(hidden)


class AdaptiveLatentReasoner(nn.Module):
    """Shared latent deliberation cell for effort-controlled reasoning.

    The cell adds compute-depth without adding vocab heads. It pools long
    sequences into a bounded hidden workspace, refines a small set of continuous
    slots for a caller-selected number of steps, then broadcasts a low-rank
    update back into the trunk stream. The slots are hidden states, not public
    text chain-of-thought tokens.
    """

    def __init__(self, cfg: OmniCoder2026Config):
        super().__init__()
        self.cfg = cfg
        self.slots = max(0, int(cfg.reasoning_slots or 0))
        self.max_steps = max(0, int(cfg.reasoning_max_steps or 0))
        self.default_steps = max(0, min(self.max_steps, int(cfg.reasoning_default_steps or 0)))
        self.pool_tokens = max(1, int(cfg.reasoning_pool_tokens or 1))
        rank = max(1, min(int(cfg.reasoning_cell_rank or 1), int(cfg.d_model)))
        self.enabled = self.slots > 0 and self.max_steps > 0 and rank > 0
        self.slot_embeddings = nn.Parameter(torch.empty(max(1, self.slots), cfg.d_model))
        self.input_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.slot_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.cell_down = QuantAwareLinear(cfg.d_model, rank, bias=False, fake_quant=cfg.fake_quant, group_size=cfg.fake_quant_group_size)
        self.cell_up = QuantAwareLinear(rank, cfg.d_model, bias=False, fake_quant=cfg.fake_quant, group_size=cfg.fake_quant_group_size)
        self.output_down = QuantAwareLinear(cfg.d_model, rank, bias=False, fake_quant=cfg.fake_quant, group_size=cfg.fake_quant_group_size)
        self.output_up = QuantAwareLinear(rank, cfg.d_model, bias=False, fake_quant=cfg.fake_quant, group_size=cfg.fake_quant_group_size)
        self.control_head = QuantAwareLinear(cfg.d_model, 5, bias=True, fake_quant=cfg.fake_quant, group_size=cfg.fake_quant_group_size)
        self.output_scale = nn.Parameter(torch.tensor(float(cfg.reasoning_output_scale), dtype=torch.float32))
        self.last_diagnostics: dict[str, Any] = {}

    def _steps_from_effort(self, effort: int | str | None) -> int:
        if not self.enabled:
            return 0
        if effort is None:
            return self.default_steps
        if isinstance(effort, str):
            key = effort.strip().lower()
            if key in {"off", "none", "0"}:
                return 0
            if key in {"low", "1"}:
                return min(self.max_steps, 1)
            if key in {"medium", "med"}:
                return min(self.max_steps, max(2, self.max_steps // 2))
            if key in {"high", "hard"}:
                return min(self.max_steps, max(4, self.max_steps))
            try:
                effort = int(key)
            except ValueError:
                return self.default_steps
        return max(0, min(self.max_steps, int(effort)))

    def _pooled_context(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] <= self.pool_tokens:
            return self.input_norm(x)
        chunk_count = min(self.pool_tokens, int(x.shape[1]))
        chunk_size = int(math.ceil(float(x.shape[1]) / float(chunk_count)))
        pad = chunk_count * chunk_size - int(x.shape[1])
        pooled = F.pad(x, (0, 0, 0, pad)) if pad > 0 else x
        pooled = pooled.reshape(x.shape[0], chunk_count, chunk_size, x.shape[-1]).mean(dim=2)
        return self.input_norm(pooled)

    def forward(
        self,
        x: torch.Tensor,
        *,
        effort: int | str | None = None,
        return_controls: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        steps = self._steps_from_effort(effort)
        if steps <= 0:
            self.last_diagnostics = {
                "schema": "omnicoder.latent_reasoner_diagnostics_2026.v1",
                "enabled": bool(self.enabled),
                "steps": 0,
                "slots": int(self.slots),
                "pool_tokens": int(min(self.pool_tokens, int(x.shape[1]))),
            }
            return x, None
        context = self._pooled_context(x)
        slots = self.slot_embeddings[: self.slots].to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        scale = 1.0 / math.sqrt(float(x.shape[-1]))
        for _ in range(steps):
            slot_q = self.slot_norm(slots)
            attn = torch.matmul(slot_q, context.transpose(-1, -2)) * scale
            attn = torch.softmax(attn, dim=-1)
            slot_context = torch.matmul(attn, context)
            slots = slots + self.cell_up(F.silu(self.cell_down(slot_context)))
        slot_summary = slots.mean(dim=1)
        update = self.output_up(F.silu(self.output_down(slot_summary))).unsqueeze(1)
        scale_param = self.output_scale.to(device=x.device, dtype=x.dtype)
        x = x + update.to(dtype=x.dtype) * scale_param
        controls_tensor = self.control_head(slot_summary)
        names = ("difficulty", "halt_continue", "answer_readiness", "verifier_margin", "tool_readiness")
        controls = {name: controls_tensor[:, index] for index, name in enumerate(names)}
        self.last_diagnostics = {
            "schema": "omnicoder.latent_reasoner_diagnostics_2026.v1",
            "enabled": True,
            "steps": int(steps),
            "slots": int(self.slots),
            "pool_tokens": int(context.shape[1]),
            "control_names": list(names),
        }
        return x, controls if return_controls else None


class OmniCoder2026Block(nn.Module):
    def __init__(self, cfg: OmniCoder2026Config, kind: BlockKind):
        super().__init__()
        canonical = "kda" if kind == "delta" else "csa" if kind == "csa_hca" else kind
        self.kind = canonical
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        if canonical == "kda":
            self.mixer = GatedDeltaLayer(cfg)
        elif canonical == "local":
            self.mixer = LocalCausalAttention(cfg)
        elif canonical in {"csa", "hca"}:
            self.mixer = SparseLatentAttention(cfg, canonical)  # type: ignore[arg-type]
        else:
            raise ValueError(f"Unknown block kind: {kind}")
        self.ffn = SwiGLU(cfg)
        residual_cls = BlockAttentionResidual if str(cfg.residual_mode).lower() in {"block_attnres", "attnres", "attention_residual"} else MHCResidual
        self.attn_residual = residual_cls(cfg)
        self.ffn_residual = residual_cls(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn_residual(x, self.mixer(self.attn_norm(x)))
        x = self.ffn_residual(x, self.ffn(self.ffn_norm(x)))
        return x


class OmniCoder2026(nn.Module):
    def __init__(
        self,
        cfg: OmniCoder2026Config,
        *,
        init_layer_devices: list[torch.device] | None = None,
        init_embed_device: torch.device | None = None,
        init_head_device: torch.device | None = None,
        checkpoint_blocks: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = int(cfg.vocab_size)
        self.max_seq_len = int(cfg.max_seq_len)
        pattern = list(cfg.layer_pattern)
        layer_devices = list(init_layer_devices or [])
        if layer_devices and len(layer_devices) != int(cfg.n_layers):
            raise ValueError(f"expected {cfg.n_layers} init layer devices, got {len(layer_devices)}")
        if layer_devices and init_head_device is None:
            init_head_device = layer_devices[-1]
        if init_embed_device is None:
            init_embed_device = init_head_device
        with _default_device_scope(init_embed_device):
            self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        blocks: list[OmniCoder2026Block] = []
        for i in range(cfg.n_layers):
            block_device = layer_devices[i] if layer_devices else None
            with _default_device_scope(block_device):
                blocks.append(OmniCoder2026Block(cfg, pattern[i % len(pattern)]))
        self.blocks = nn.ModuleList(blocks)
        with _default_device_scope(init_head_device):
            self.latent_reasoner = AdaptiveLatentReasoner(cfg)
            self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        # The tied 330k-token output matrix is too large to fake-quantize on
        # every forward pass at the 20B target scale; keep it trainable in
        # precision and quantize it during the export/runtime pack step.
        with _default_device_scope(init_head_device):
            self.lm_head = QuantAwareLinear(cfg.d_model, cfg.vocab_size, bias=False, fake_quant=False, group_size=cfg.fake_quant_group_size)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        with _default_device_scope(init_head_device):
            self.mtp_heads = nn.ModuleList([
                QuantAwareLinear(cfg.d_model, cfg.vocab_size, bias=False, fake_quant=cfg.fake_quant, group_size=cfg.fake_quant_group_size)
                for _ in range(int(cfg.mtp_heads))
            ])
            self.flow_head = nn.Sequential(
                RMSNorm(cfg.d_model, cfg.rms_norm_eps),
                QuantAwareLinear(cfg.d_model, cfg.flow_latent_dim, bias=False, fake_quant=cfg.fake_quant, group_size=cfg.fake_quant_group_size),
            )
            self.native_media_bridge = NativeContinuousMediaBridge(cfg)
            self.grounding_head = QuantAwareLinear(cfg.d_model, 8, bias=True, fake_quant=cfg.fake_quant, group_size=cfg.fake_quant_group_size)
            self.sync_head = QuantAwareLinear(cfg.d_model, 1, bias=True, fake_quant=cfg.fake_quant, group_size=cfg.fake_quant_group_size)
        reset_omnicoder2026_parameters(self, cfg)
        self._weighted_device_map: dict[str, object] | None = None
        self._weighted_pipeline_stages: list[tuple[torch.device, int, int]] = []
        self._checkpoint_blocks = bool(checkpoint_blocks)
        self.last_reasoning_diagnostics: dict[str, Any] = {}

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        for parameter in module.parameters(recurse=True):
            return parameter.device
        for buffer in module.buffers(recurse=True):
            return buffer.device
        return torch.device("cpu")

    def _chunked_lm_loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute next-token loss without materializing full-sequence vocab logits."""
        loss_sum, total_tokens = self._chunked_lm_loss_sum(hidden, labels)
        return loss_sum / float(max(1, total_tokens))

    def _chunked_lm_loss_sum(self, hidden: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Return summed next-token loss and token count for microbatch aggregation."""
        if labels.device != hidden.device:
            labels = labels.to(hidden.device, non_blocking=True)
        chunk_tokens = max(1, _env_int("OMNICODER2026_LM_LOSS_CHUNK_TOKENS", 128))
        shifted_hidden = hidden[:, :-1, :]
        shifted_labels = labels[:, 1:]
        total_tokens = int(shifted_labels.numel())
        loss_sum = hidden.new_zeros(())
        if total_tokens <= 0:
            return loss_sum, 0
        for start in range(0, shifted_hidden.shape[1], chunk_tokens):
            end = min(shifted_hidden.shape[1], start + chunk_tokens)
            logits = self.lm_head(shifted_hidden[:, start:end, :])
            loss_sum = loss_sum + F.cross_entropy(
                logits.transpose(1, 2),
                shifted_labels[:, start:end],
                reduction="sum",
            )
        return loss_sum, total_tokens

    @staticmethod
    def _contiguous_pipeline_stages(layer_devices: list[torch.device]) -> list[tuple[torch.device, int, int]]:
        if not layer_devices:
            return []
        stages: list[tuple[torch.device, int, int]] = []
        start = 0
        current = layer_devices[0]
        for index, device in enumerate(layer_devices[1:], start=1):
            if device != current:
                stages.append((current, start, index))
                start = index
                current = device
        stages.append((current, start, len(layer_devices)))
        return stages

    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self._checkpoint_blocks and self.training and torch.is_grad_enabled():
            return activation_checkpoint(block, x, use_reentrant=False)
        return block(x)

    def _run_stage(self, x: torch.Tensor, start: int, end: int, device: torch.device, *, non_blocking: bool = True) -> torch.Tensor:
        if x.device != device:
            x = x.to(device, non_blocking=bool(non_blocking))
        for block in self.blocks[start:end]:
            x = self._run_block(block, x)
        return x

    def _weighted_reasoning_effort(self) -> int | str | None:
        raw = os.environ.get("OMNICODER2026_WEIGHTED_REASONING_EFFORT", "")
        if not raw:
            return int(self.cfg.reasoning_default_steps)
        try:
            return int(raw)
        except ValueError:
            return raw

    def _apply_latent_reasoning(
        self,
        x: torch.Tensor,
        *,
        effort: int | str | None = None,
        return_controls: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if isinstance(self.latent_reasoner, AdaptiveLatentReasoner):
            x, controls = self.latent_reasoner(x, effort=effort, return_controls=return_controls)
            self.last_reasoning_diagnostics = dict(self.latent_reasoner.last_diagnostics)
            return x, controls
        return x, None

    def _forward_weighted_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed_device = self._module_device(self.embed)
        if input_ids.device != embed_device:
            input_ids = input_ids.to(embed_device, non_blocking=True)
        x = self.embed(input_ids)
        if self._weighted_pipeline_stages:
            for device, start, end in self._weighted_pipeline_stages:
                x = self._run_stage(x, start, end, device)
            return x
        for index, block in enumerate(self.blocks):
            block_device = self._module_device(block)
            x = self._run_stage(x, index, index + 1, block_device)
        return x

    def forward_weighted_pipeline_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        microbatches: int = 1,
        async_streams: bool = True,
    ) -> torch.Tensor:
        """Compute weighted-placement LM loss with optional microbatch pipelining.

        This method is intentionally outside ``forward`` so inference/eval and
        checkpoint formats stay unchanged. It only targets the dense trainer's
        next-token loss path.
        """
        requested_microbatches = max(1, int(microbatches))
        batch_size = int(input_ids.shape[0])
        if self._weighted_device_map is None or requested_microbatches <= 1 or batch_size <= 1:
            out = self(
                input_ids,
                labels=labels,
                return_aux=False,
                return_logits=False,
                return_hidden=False,
            )
            loss = out["loss"]
            if loss is None:
                raise RuntimeError("weighted pipeline loss path did not produce a loss")
            return loss
        chunks = min(requested_microbatches, batch_size)
        input_chunks = list(torch.chunk(input_ids, chunks, dim=0))
        label_chunks = list(torch.chunk(labels, chunks, dim=0))
        if async_streams and self._can_stream_weighted_pipeline():
            return self._forward_weighted_pipeline_loss_streamed(input_chunks, label_chunks)
        return self._forward_weighted_pipeline_loss_serial(input_chunks, label_chunks)

    def _can_stream_weighted_pipeline(self) -> bool:
        if os.environ.get("OMNICODER2026_ENABLE_ASYNC_PIPELINE", "0") not in {"1", "true", "True", "yes", "YES"}:
            return False
        if not torch.cuda.is_available() or len(self._weighted_pipeline_stages) <= 1:
            return False
        devices = [device for device, _, _ in self._weighted_pipeline_stages]
        head_device = self._module_device(self.norm)
        return head_device.type == "cuda" and all(device.type == "cuda" for device in devices)

    def _forward_weighted_pipeline_loss_serial(
        self,
        input_chunks: list[torch.Tensor],
        label_chunks: list[torch.Tensor],
    ) -> torch.Tensor:
        head_device = self._module_device(self.norm)
        loss_sums: list[torch.Tensor] = []
        total_tokens = 0
        for input_chunk, label_chunk in zip(input_chunks, label_chunks, strict=True):
            x = self._forward_weighted_hidden(input_chunk)
            if x.device != head_device:
                x = x.to(head_device, non_blocking=True)
            x, _controls = self._apply_latent_reasoning(x, effort=self._weighted_reasoning_effort())
            hidden = self.norm(x)
            loss_sum, tokens = self._chunked_lm_loss_sum(hidden, label_chunk)
            loss_sums.append(loss_sum)
            total_tokens += int(tokens)
        return torch.stack(loss_sums).sum() / float(max(1, total_tokens))

    def _forward_weighted_pipeline_loss_streamed(
        self,
        input_chunks: list[torch.Tensor],
        label_chunks: list[torch.Tensor],
    ) -> torch.Tensor:
        stages = list(self._weighted_pipeline_stages)
        head_device = self._module_device(self.norm)
        stream_cache: dict[str, torch.cuda.Stream] = {}

        def stage_stream(device: torch.device) -> torch.cuda.Stream:
            key = str(device)
            stream = stream_cache.get(key)
            if stream is None:
                with torch.cuda.device(device):
                    stream = torch.cuda.Stream(device=device)
                stream_cache[key] = stream
            return stream

        stage_streams = [stage_stream(device) for device, _, _ in stages]
        head_stream = stage_stream(head_device)
        stage_outputs: dict[tuple[int, int], torch.Tensor] = {}
        stage_events: dict[tuple[int, int], torch.cuda.Event] = {}
        loss_sums: list[torch.Tensor | None] = [None] * len(input_chunks)
        loss_events: list[torch.cuda.Event | None] = [None] * len(input_chunks)
        token_counts = [max(0, int(labels[:, 1:].numel())) for labels in label_chunks]
        embed_device = self._module_device(self.embed)
        embed_stream = stage_stream(embed_device)
        embedded_chunks: list[torch.Tensor] = []
        embedded_events: list[torch.cuda.Event] = []
        with torch.cuda.device(embed_device), torch.cuda.stream(embed_stream):
            for input_chunk in input_chunks:
                ids = input_chunk
                if ids.device != embed_device:
                    ids = ids.to(embed_device, non_blocking=True)
                embedded = self.embed(ids)
                event = torch.cuda.Event()
                event.record(embed_stream)
                embedded_chunks.append(embedded)
                embedded_events.append(event)

        for clock in range(len(input_chunks) + len(stages) - 1):
            for stage_index in reversed(range(len(stages))):
                microbatch_index = clock - stage_index
                if microbatch_index < 0 or microbatch_index >= len(input_chunks):
                    continue
                device, start, end = stages[stage_index]
                stream = stage_streams[stage_index]
                with torch.cuda.device(device), torch.cuda.stream(stream):
                    if stage_index == 0:
                        stream.wait_event(embedded_events[microbatch_index])
                        x = embedded_chunks[microbatch_index]
                    else:
                        previous_key = (microbatch_index, stage_index - 1)
                        previous_event = stage_events.pop(previous_key)
                        stream.wait_event(previous_event)
                        x = stage_outputs.pop(previous_key)
                    if x.is_cuda:
                        x.record_stream(stream)
                    x = self._run_stage(x, start, end, device, non_blocking=False)
                    ready_event = torch.cuda.Event()
                    ready_event.record(stream)
                    if stage_index == len(stages) - 1:
                        with torch.cuda.device(head_device), torch.cuda.stream(head_stream):
                            head_stream.wait_event(ready_event)
                            if x.device != head_device:
                                x = x.to(head_device, non_blocking=True)
                            x, _controls = self._apply_latent_reasoning(x, effort=self._weighted_reasoning_effort())
                            hidden = self.norm(x)
                            loss_sum, _ = self._chunked_lm_loss_sum(hidden, label_chunks[microbatch_index])
                            loss_sums[microbatch_index] = loss_sum
                            done = torch.cuda.Event()
                            done.record(head_stream)
                            loss_events[microbatch_index] = done
                    else:
                        stage_outputs[(microbatch_index, stage_index)] = x
                        stage_events[(microbatch_index, stage_index)] = ready_event

        current = torch.cuda.current_stream(head_device)
        for event in loss_events:
            if event is not None:
                current.wait_event(event)
        concrete_losses = [loss_sum for loss_sum in loss_sums if loss_sum is not None]
        if not concrete_losses:
            return self.norm.weight.sum() * 0.0
        return torch.stack(concrete_losses).sum() / float(max(1, sum(token_counts)))

    def apply_weighted_device_map(
        self,
        layer_devices: list[torch.device],
        *,
        embed_device: torch.device,
        head_device: torch.device,
        checkpoint_blocks: bool = False,
    ) -> dict[str, object]:
        if len(layer_devices) != len(self.blocks):
            raise ValueError(f"expected {len(self.blocks)} layer devices, got {len(layer_devices)}")
        self.embed.to(embed_device)
        self.lm_head.to(head_device)
        if self.cfg.tie_embeddings:
            if embed_device != head_device:
                raise ValueError("tied embeddings require embed_device == head_device for weighted placement")
            self.lm_head.weight = self.embed.weight
        for block, device in zip(self.blocks, layer_devices, strict=True):
            block.to(device)
        self.norm.to(head_device)
        self.latent_reasoner.to(head_device)
        self.mtp_heads.to(head_device)
        self.flow_head.to(head_device)
        self.native_media_bridge.to(embed_device)
        self.grounding_head.to(head_device)
        self.sync_head.to(head_device)
        self._checkpoint_blocks = bool(checkpoint_blocks)
        layer_counts: dict[str, int] = {}
        for device in layer_devices:
            key = str(device)
            layer_counts[key] = layer_counts.get(key, 0) + 1
        self._weighted_device_map = {
            "mode": "weighted_layers",
            "embed_device": str(embed_device),
            "head_device": str(head_device),
            "layer_devices": [str(device) for device in layer_devices],
            "layer_counts": layer_counts,
            "checkpoint_blocks": bool(checkpoint_blocks),
            "pipeline_stages": [
                {"device": str(device), "start": int(start), "end": int(end)}
                for device, start, end in self._contiguous_pipeline_stages(layer_devices)
            ],
        }
        self._weighted_pipeline_stages = self._contiguous_pipeline_stages(layer_devices)
        return dict(self._weighted_device_map)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        flow_targets: torch.Tensor | None = None,
        flow_mask: torch.Tensor | None = None,
        native_media_features: torch.Tensor | None = None,
        native_media_type_ids: torch.Tensor | None = None,
        native_media_positions: torch.Tensor | None = None,
        native_media_targets: torch.Tensor | None = None,
        native_media_mask: torch.Tensor | None = None,
        return_mtp: bool = False,
        return_aux: bool = False,
        return_logits: bool = True,
        return_hidden: bool = True,
        reasoning_effort: int | str | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        if input_ids.shape[1] > self.max_seq_len:
            raise ValueError(f"sequence length {input_ids.shape[1]} exceeds native context {self.max_seq_len}")
        embed_device = self._module_device(self.embed)
        if input_ids.device != embed_device:
            input_ids = input_ids.to(embed_device, non_blocking=True)
        x = self.embed(input_ids)
        native_media_token_count = 0
        if native_media_features is not None:
            media_device = self._module_device(self.native_media_bridge)
            if native_media_features.device != media_device:
                native_media_features = native_media_features.to(media_device, non_blocking=True)
            if native_media_type_ids is not None and native_media_type_ids.device != media_device:
                native_media_type_ids = native_media_type_ids.to(media_device, non_blocking=True)
            if native_media_positions is not None and native_media_positions.device != media_device:
                native_media_positions = native_media_positions.to(media_device, non_blocking=True)
            media_x = self.native_media_bridge.embed(native_media_features, native_media_type_ids, native_media_positions)
            native_media_token_count = int(media_x.shape[1])
            if native_media_token_count > x.shape[1]:
                raise ValueError("native media feature tokens cannot exceed input_ids length in aligned mode")
            x = x.clone()
            x[:, :native_media_token_count, :] = x[:, :native_media_token_count, :] + media_x.to(device=x.device, dtype=x.dtype)
        for block in self.blocks:
            block_device = self._module_device(block)
            if x.device != block_device:
                x = x.to(block_device, non_blocking=True)
            x = self._run_block(block, x)
        head_device = self._module_device(self.norm)
        if x.device != head_device:
            x = x.to(head_device, non_blocking=True)
        reasoner_controls: dict[str, torch.Tensor] | None = None
        x, reasoner_controls = self._apply_latent_reasoning(x, effort=reasoning_effort, return_controls=bool(return_aux))
        loss = None
        hidden = self.norm(x)
        logits = None
        if labels is not None and not return_logits:
            loss = self._chunked_lm_loss(hidden, labels)
        else:
            logits = self.lm_head(hidden)
            if labels is not None:
                if labels.device != logits.device:
                    labels = labels.to(logits.device, non_blocking=True)
                loss = F.cross_entropy(logits[:, :-1, :].transpose(1, 2), labels[:, 1:])
        result: dict[str, torch.Tensor | list[torch.Tensor] | None] = {"loss": loss}
        if return_logits:
            result["logits"] = logits
        if return_hidden:
            result["hidden_states"] = hidden
        if return_mtp:
            result["mtp_logits"] = [head(hidden) for head in self.mtp_heads]
        if flow_targets is not None or return_aux:
            flow = self.flow_head(hidden)
            result["flow"] = flow
            if flow_targets is not None:
                if flow_targets.device != flow.device:
                    flow_targets = flow_targets.to(flow.device, non_blocking=True)
                flow_loss_raw = F.mse_loss(flow, flow_targets, reduction="none").mean(dim=-1)
                if flow_mask is not None:
                    if flow_mask.device != flow.device:
                        flow_mask = flow_mask.to(flow.device, non_blocking=True)
                    active = flow_mask.to(dtype=flow_loss_raw.dtype)
                    flow_loss = (flow_loss_raw * active).sum() / active.sum().clamp_min(1.0)
                else:
                    flow_loss = flow_loss_raw.mean()
                result["flow_loss"] = flow_loss
                result["loss"] = flow_loss if loss is None else loss + flow_loss
        if native_media_features is not None and (native_media_targets is not None or return_aux):
            media_device = self._module_device(self.native_media_bridge)
            media_hidden = hidden[:, :native_media_token_count, :]
            if media_hidden.device != media_device:
                media_hidden = media_hidden.to(media_device, non_blocking=True)
            native_media_recon = self.native_media_bridge.reconstruct(media_hidden)
            result["native_media_reconstruction"] = native_media_recon
            if native_media_targets is not None:
                if native_media_targets.device != native_media_recon.device:
                    native_media_targets = native_media_targets.to(native_media_recon.device, non_blocking=True)
                if native_media_targets.shape[1] < native_media_token_count:
                    raise ValueError(
                        f"native media targets must cover {native_media_token_count} aligned tokens, "
                        f"got {native_media_targets.shape[1]}"
                    )
                native_media_targets = native_media_targets[:, :native_media_token_count, :]
                native_media_targets = self.native_media_bridge._fit_last_dim(
                    native_media_targets.to(dtype=native_media_recon.dtype),
                    native_media_recon.shape[-1],
                )
                media_loss_raw = F.mse_loss(native_media_recon, native_media_targets, reduction="none").mean(dim=-1)
                if native_media_mask is not None:
                    if native_media_mask.device != native_media_recon.device:
                        native_media_mask = native_media_mask.to(native_media_recon.device, non_blocking=True)
                    active = native_media_mask[:, :native_media_token_count].to(dtype=media_loss_raw.dtype)
                    native_media_loss = (media_loss_raw * active).sum() / active.sum().clamp_min(1.0)
                else:
                    native_media_loss = media_loss_raw.mean()
                result["native_media_loss"] = native_media_loss
                base_loss = result.get("loss")
                if base_loss is not None and native_media_loss.device != base_loss.device:
                    native_media_loss = native_media_loss.to(base_loss.device, non_blocking=True)
                result["loss"] = native_media_loss if base_loss is None else base_loss + native_media_loss
        if return_aux:
            result["grounding"] = self.grounding_head(hidden)
            result["sync"] = self.sync_head(hidden).squeeze(-1)
            if reasoner_controls is not None:
                for name, tensor in reasoner_controls.items():
                    result[f"reasoning_{name}"] = tensor
        return result

    def architecture_manifest(self) -> dict:
        cfg = self.cfg
        pattern = [("kda" if k == "delta" else "csa" if k == "csa_hca" else k) for k in cfg.layer_pattern]
        expanded = [pattern[i % len(pattern)] for i in range(cfg.n_layers)]
        return {
            "architecture": "omnicoder2026_dense_kda_csa_hca_attnres_one_trunk",
            "native_context": int(cfg.max_seq_len),
            "layers": int(cfg.n_layers),
            "pattern": pattern,
            "expanded_counts": {kind: expanded.count(kind) for kind in sorted(set(expanded))},
            "dense": True,
            "moe": False,
            "kda": {
                "variant": "gated_deltanet2_pytorch_correctness_path",
                "kernel_size": int(cfg.kda_kernel_size),
                "state_dtype": cfg.kda_state_dtype,
                "role": "dominant recurrent-linear memory path with no per-token KV cache",
            },
            "v4_attention": {
                "shared_kv_mqa": int(cfg.num_key_value_heads) == 1,
                "kv_rule": "K=V for sparse CSA/HCA branches",
                "q_lora_rank": int(cfg.q_lora_rank),
                "o_lora_rank": int(cfg.o_lora_rank),
                "o_groups": int(cfg.o_groups),
                "rope_applied_to": "trailing partial head dimensions",
                "sink_type": "per-head sink logits in the denominator, not value tokens",
                "local_window": int(cfg.local_window),
            },
            "csa": {
                "compress_rate": int(cfg.csa_compress_rate),
                "top_k_blocks": int(cfg.csa_top_k_blocks),
                "chunked_sparse_gather": True,
                "legacy_block_size": int(cfg.csa_block_size),
                "role": "DeepSeek-V4-style compressed sparse recall with deterministic recency/prefix indexer placeholder",
            },
            "hca": {
                "compress_rate": int(cfg.hca_compress_rate),
                "legacy_block_size": int(cfg.hca_block_size),
                "role": "heavily compressed long-range causal global trail",
            },
            "local_window": int(cfg.local_window),
            "m_hyper_connections": {
                "mode": cfg.residual_mode,
                "hc_mult": int(cfg.hc_mult),
                "sinkhorn_iters_for_full_kernel": int(cfg.hc_sinkhorn_iters),
            },
            "attention_residuals": {
                "mode": cfg.residual_mode,
                "block_size": int(cfg.block_attnres_block_size),
                "max_blocks": int(cfg.block_attnres_max_blocks),
                "rank": int(cfg.block_attnres_rank),
                "chunk_tokens": int(cfg.block_attnres_chunk_tokens),
                "memory_rule": "compressed causal residual block summaries; no full depth-token history",
            },
            "adaptive_latent_reasoning": {
                "mode": "shared_low_rank_hidden_deliberation_slots",
                "slots": int(cfg.reasoning_slots),
                "max_steps": int(cfg.reasoning_max_steps),
                "default_steps": int(cfg.reasoning_default_steps),
                "cell_rank": int(cfg.reasoning_cell_rank),
                "pool_tokens": int(cfg.reasoning_pool_tokens),
                "controls": ["difficulty", "halt_continue", "answer_readiness", "verifier_margin", "tool_readiness"],
                "public_cot": False,
                "rule": "adds compute-depth by reusing one latent cell; does not add full-vocab verifier heads",
            },
            "native_continuous_media": {
                "mode": "shared_sensenova_style_patch_segment_flow",
                "feature_dim": int(cfg.native_media_feature_dim),
                "position_dim": int(cfg.native_media_position_dim),
                "type_vocab": int(cfg.native_media_type_vocab),
                "trunk_rule": "all image/video/audio/music/TTS/OCR patches enter through one shared projection plus type/time metadata",
                "no_in_trunk_modality_adapters": True,
            },
            "generation_modes": ["autoregressive_tokens", "continuous_latent_flow", "native_continuous_patch_segment_flow", "codec_bridge_tokens"],
            "omni_heads": ["shared_lm_head", "mtp_heads", "latent_reasoner_controls", "flow_head", "native_media_reconstruction_head", "grounding_head", "sync_head"],
            "token_ranges": {k: [int(v[0]), int(v[1])] for k, v in cfg.token_ranges.items()},
            "quantization": {
                "weights": cfg.weight_quant_target,
                "kv_state": cfg.kv_quant_target,
                "fake_quant": bool(cfg.fake_quant),
            },
            "training_placement": self._weighted_device_map or {"mode": "single_device_or_fsdp"},
            "gguf_bridge": {
                "stock_llama_cpp": cfg.gguf_bridge_architecture,
                "truth": "stock GGUF is the adoption bridge; true 1M needs the native KDA/CSA/HCA runtime",
            },
        }


def build_omnicoder2026(profile: str = "target_20b", **overrides) -> OmniCoder2026:
    key = profile.strip().lower().replace("-", "_")
    if key in ("probe", "dense_native1m_probe", "omnicoder2026_native1m_probe"):
        cfg = OmniCoder2026Config.probe()
    elif key in ("ledger_probe", "full_ledger_probe", "omnicoder2026_full_ledger_probe"):
        cfg = OmniCoder2026Config.probe()
        cfg.vocab_size = 330_000
    elif key in ("pilot_3b", "omnicoder2026_3b_pilot"):
        cfg = OmniCoder2026Config.pilot_3b()
    elif key in ("target", "target_20b", "dense_omni_24gb", "omnicoder_20b_1m", "omnicoder2026_20b_1m"):
        cfg = OmniCoder2026Config.target_20b()
    elif key in ("target_7b", "omnicoder_7b_1m", "omnicoder2026_7b_1m"):
        cfg = OmniCoder2026Config.target_7b()
    elif key in ("target_12b", "target_16b", "omnicoder_12b_1m", "omnicoder2026_12b_1m", "omnicoder2026_16b_1m"):
        cfg = OmniCoder2026Config.target_16b()
    else:
        raise ValueError(f"Unknown Omnicoder2026 profile: {profile}")
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        elif key == "global_block_size":
            cfg.csa_block_size = int(value)
        elif key == "global_top_k_blocks":
            cfg.csa_top_k_blocks = int(value)
        elif key == "mla_latent_dim":
            cfg.latent_dim = int(value)
        else:
            raise ValueError(f"Unknown Omnicoder2026 config override: {key}")
    return OmniCoder2026(cfg)
