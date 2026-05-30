"""Pure PyTorch 2026 recurrent-linear attention reference paths.

Tensor convention for the functional paths:
    q, k:  (batch, time, heads, key_dim)
    v:     (batch, time, heads, value_dim)
    state: (batch, heads, key_dim, value_dim), always float32

These implementations are intentionally small and kernel-free. They are meant
for correctness checks, CPU/CUDA smoke tests, and as stable fallbacks when
Triton or fused chunkwise kernels are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


State = torch.Tensor


def _check_bthd(name: str, x: torch.Tensor) -> None:
    if x.ndim != 4:
        raise ValueError(f"{name} must have shape (batch, time, heads, dim); got {tuple(x.shape)}")


def _check_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    _check_bthd("q", q)
    _check_bthd("k", k)
    _check_bthd("v", v)
    if q.shape[:3] != k.shape[:3] or q.shape[:3] != v.shape[:3]:
        raise ValueError(
            "q, k, and v must share (batch, time, heads); "
            f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q and k key dimensions must match; got {q.shape[-1]} and {k.shape[-1]}")


def _initial_state(
    q: torch.Tensor,
    value_dim: int,
    initial_state: Optional[torch.Tensor],
) -> torch.Tensor:
    batch, _, heads, key_dim = q.shape
    expected = (batch, heads, key_dim, value_dim)
    if initial_state is None:
        return torch.zeros(expected, device=q.device, dtype=torch.float32)
    if tuple(initial_state.shape) != expected:
        raise ValueError(f"initial_state must have shape {expected}; got {tuple(initial_state.shape)}")
    return initial_state.to(device=q.device, dtype=torch.float32)


def _optional_gate(
    gate: Optional[torch.Tensor],
    step: int,
    like: torch.Tensor,
    *,
    sigmoid: bool,
) -> torch.Tensor:
    if gate is None:
        return torch.ones((*like.shape[:-1], 1), device=like.device, dtype=torch.float32)
    if gate.ndim == 4:
        current = gate[:, step]
    elif gate.ndim == 3:
        current = gate[:, step].unsqueeze(-1)
    else:
        raise ValueError(f"gate must have shape (batch, time, heads) or (batch, time, heads, 1); got {tuple(gate.shape)}")
    current = current.to(device=like.device, dtype=torch.float32)
    return torch.sigmoid(current) if sigmoid else current


def _prepare_gate(gate: Optional[torch.Tensor], like: torch.Tensor, *, sigmoid: bool) -> torch.Tensor | float:
    if gate is None:
        return 1.0
    if gate.ndim == 3:
        gate = gate.unsqueeze(-1)
    elif gate.ndim != 4:
        raise ValueError(f"gate must have shape (batch, time, heads) or (batch, time, heads, 1); got {tuple(gate.shape)}")
    gate = gate.to(device=like.device, dtype=torch.float32)
    return torch.sigmoid(gate) if sigmoid else gate


def _read(q_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    return torch.matmul(q_t.unsqueeze(-2), state).squeeze(-2)


def _outer(k_t: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    return k_t.unsqueeze(-1) * residual.unsqueeze(-2)


def kda_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    beta: Optional[torch.Tensor] = None,
    forget: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Kernelized Delta Attention reference recurrence.

    Args:
        q, k: (batch, time, heads, key_dim)
        v: (batch, time, heads, value_dim)
        beta: optional write gate, shape (batch, time, heads[, 1]). Values are
            used directly, so callers may pass probabilities or learned scales.
        forget: optional retention gate, shape (batch, time, heads[, 1]). Values
            are used directly; ``1`` preserves the previous state.
        initial_state: optional fp32 recurrent state with shape
            (batch, heads, key_dim, value_dim).

    Returns:
        output with shape (batch, time, heads, value_dim) and dtype ``v.dtype``,
        plus the final fp32 recurrent state.
    """

    _check_qkv(q, k, v)
    q_f = q.float()
    k_f = F.normalize(k.float(), dim=-1)
    v_f = v.float()
    state = _initial_state(q, v.shape[-1], initial_state)
    outputs = torch.empty_like(v_f)
    beta_f = _prepare_gate(beta, v_f, sigmoid=False)
    forget_f = _prepare_gate(forget, v_f, sigmoid=False)

    for step in range(q.shape[1]):
        q_t = q_f[:, step]
        k_t = k_f[:, step]
        v_t = v_f[:, step]
        write = beta_f[:, step] if isinstance(beta_f, torch.Tensor) else beta_f
        retain = forget_f[:, step] if isinstance(forget_f, torch.Tensor) else forget_f
        prediction = _read(k_t, state)
        update = _outer(k_t, write * (v_t - prediction))
        state = (retain.unsqueeze(-2) * state if isinstance(retain, torch.Tensor) else state * float(retain)) + update
        outputs[:, step] = _read(q_t, state)

    return outputs.to(dtype=v.dtype), state


def gated_deltanet2_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    write_gate: Optional[torch.Tensor] = None,
    forget_gate: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gated DeltaNet-2 reference recurrence.

    The gate tensors are treated as logits and passed through sigmoid. This
    mirrors the usual module path where projections emit unconstrained gates.
    Shapes match :func:`kda_pytorch`.
    """

    _check_qkv(q, k, v)
    q_f = q.float()
    k_f = F.normalize(k.float(), dim=-1)
    v_f = v.float()
    state = _initial_state(q, v.shape[-1], initial_state)
    outputs = torch.empty_like(v_f)
    write_f = _prepare_gate(write_gate, v_f, sigmoid=True)
    forget_f = _prepare_gate(forget_gate, v_f, sigmoid=True)

    for step in range(q.shape[1]):
        q_t = q_f[:, step]
        k_t = k_f[:, step]
        v_t = v_f[:, step]
        write = write_f[:, step] if isinstance(write_f, torch.Tensor) else write_f
        retain = forget_f[:, step] if isinstance(forget_f, torch.Tensor) else forget_f
        prediction = _read(k_t, state)
        update = _outer(k_t, write * (v_t - prediction))
        state = (retain.unsqueeze(-2) * state if isinstance(retain, torch.Tensor) else state * float(retain)) + update
        outputs[:, step] = _read(q_t, state)

    return outputs.to(dtype=v.dtype), state


def kaczmarz_linear_attention_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    relaxation: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Online Kaczmarz linear attention reference path.

    Each token performs the Kaczmarz projection update:
        state += eta * outer(k, v - k @ state) / (||k||^2 + eps)

    Args and return shapes match :func:`kda_pytorch`. ``relaxation`` is an
    optional per-token scale with shape (batch, time, heads[, 1]).
    """

    _check_qkv(q, k, v)
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    state = _initial_state(q, v.shape[-1], initial_state)
    outputs = torch.empty_like(v_f)
    relaxation_f = _prepare_gate(relaxation, v_f, sigmoid=False)

    for step in range(q.shape[1]):
        q_t = q_f[:, step]
        k_t = k_f[:, step]
        v_t = v_f[:, step]
        eta = relaxation_f[:, step] if isinstance(relaxation_f, torch.Tensor) else relaxation_f
        denom = k_t.square().sum(dim=-1, keepdim=True).clamp_min(float(eps))
        residual = v_t - _read(k_t, state)
        state = state + _outer(k_t, eta * residual / denom)
        outputs[:, step] = _read(q_t, state)

    return outputs.to(dtype=v.dtype), state


@dataclass(frozen=True)
class RecurrentLinearAttentionConfig:
    d_model: int
    n_heads: int
    head_dim: Optional[int] = None

    @property
    def key_dim(self) -> int:
        if self.head_dim is not None:
            return int(self.head_dim)
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads when head_dim is not set")
        return self.d_model // self.n_heads


class _BaseRecurrentLinearAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: Optional[int] = None) -> None:
        super().__init__()
        self.cfg = RecurrentLinearAttentionConfig(d_model=d_model, n_heads=n_heads, head_dim=head_dim)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.cfg.key_dim
        inner = self.n_heads * self.head_dim
        self.q_proj = nn.Linear(self.d_model, inner, bias=False)
        self.k_proj = nn.Linear(self.d_model, inner, bias=False)
        self.v_proj = nn.Linear(self.d_model, inner, bias=False)
        self.o_proj = nn.Linear(inner, self.d_model, bias=False)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, _ = x.shape
        return x.view(batch, time, self.n_heads, self.head_dim)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, _, _ = x.shape
        return x.reshape(batch, time, self.n_heads * self.head_dim)


class KDA(_BaseRecurrentLinearAttention):
    """Module wrapper for :func:`kda_pytorch` over input shape (batch, time, d_model)."""

    def __init__(self, d_model: int, n_heads: int, head_dim: Optional[int] = None) -> None:
        super().__init__(d_model, n_heads, head_dim)
        self.beta_proj = nn.Linear(self.d_model, self.n_heads, bias=True)
        self.forget_proj = nn.Linear(self.d_model, self.n_heads, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        beta = torch.sigmoid(self.beta_proj(x))
        forget = torch.sigmoid(self.forget_proj(x))
        y, state = kda_pytorch(
            self._shape(self.q_proj(x)),
            self._shape(self.k_proj(x)),
            self._shape(self.v_proj(x)),
            beta=beta,
            forget=forget,
            initial_state=initial_state,
        )
        out = self.o_proj(self._merge(y))
        return (out, state) if return_state else out


class GatedDeltaNet2(_BaseRecurrentLinearAttention):
    """Module wrapper for :func:`gated_deltanet2_pytorch` over (batch, time, d_model)."""

    def __init__(self, d_model: int, n_heads: int, head_dim: Optional[int] = None) -> None:
        super().__init__(d_model, n_heads, head_dim)
        self.write_gate_proj = nn.Linear(self.d_model, self.n_heads, bias=True)
        self.forget_gate_proj = nn.Linear(self.d_model, self.n_heads, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        y, state = gated_deltanet2_pytorch(
            self._shape(self.q_proj(x)),
            self._shape(self.k_proj(x)),
            self._shape(self.v_proj(x)),
            write_gate=self.write_gate_proj(x),
            forget_gate=self.forget_gate_proj(x),
            initial_state=initial_state,
        )
        out = self.o_proj(self._merge(y))
        return (out, state) if return_state else out


class KaczmarzLinearAttention(_BaseRecurrentLinearAttention):
    """Module wrapper for :func:`kaczmarz_linear_attention_pytorch` over (batch, time, d_model)."""

    def __init__(self, d_model: int, n_heads: int, head_dim: Optional[int] = None) -> None:
        super().__init__(d_model, n_heads, head_dim)
        self.relaxation_proj = nn.Linear(self.d_model, self.n_heads, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        relaxation = 2.0 * torch.sigmoid(self.relaxation_proj(x))
        y, state = kaczmarz_linear_attention_pytorch(
            self._shape(self.q_proj(x)),
            self._shape(self.k_proj(x)),
            self._shape(self.v_proj(x)),
            relaxation=relaxation,
            initial_state=initial_state,
        )
        out = self.o_proj(self._merge(y))
        return (out, state) if return_state else out


__all__ = [
    "GatedDeltaNet2",
    "KDA",
    "KaczmarzLinearAttention",
    "RecurrentLinearAttentionConfig",
    "gated_deltanet2_pytorch",
    "kaczmarz_linear_attention_pytorch",
    "kda_pytorch",
]
