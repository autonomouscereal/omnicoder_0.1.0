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
import os
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


def _truthy_env(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


def _gdn2_tensor_scan(
    q_f: torch.Tensor,
    k_f: torch.Tensor,
    v_f: torch.Tensor,
    write_f: torch.Tensor,
    forget_f: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Branch-free tensor recurrence used by the compiled fast path.

    Inputs are already fp32, keys are normalized, and gates are sigmoid outputs.
    The math intentionally mirrors ``gated_deltanet2_pytorch`` exactly.
    """

    outputs = torch.empty_like(v_f)
    for step in range(q_f.shape[1]):
        q_t = q_f[:, step]
        k_t = k_f[:, step]
        v_t = v_f[:, step]
        write = write_f[:, step]
        retain = forget_f[:, step]
        prediction = _read(k_t, state)
        update = _outer(k_t, write * (v_t - prediction))
        state = retain.unsqueeze(-2) * state + update
        outputs[:, step] = _read(q_t, state)
    return outputs, state


@torch.jit.script
def _gdn2_tensor_scan_jit(
    q_f: torch.Tensor,
    k_f: torch.Tensor,
    v_f: torch.Tensor,
    write_f: torch.Tensor,
    forget_f: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = torch.empty_like(v_f)
    for step in range(q_f.size(1)):
        q_t = q_f[:, step]
        k_t = k_f[:, step]
        v_t = v_f[:, step]
        write = write_f[:, step]
        retain = forget_f[:, step]
        prediction = torch.matmul(k_t.unsqueeze(-2), state).squeeze(-2)
        update = k_t.unsqueeze(-1) * (write * (v_t - prediction)).unsqueeze(-2)
        state = retain.unsqueeze(-2) * state + update
        outputs[:, step] = torch.matmul(q_t.unsqueeze(-2), state).squeeze(-2)
    return outputs, state


_COMPILED_GDN2_SCAN: object | None = None
_GDN2_COMPILE_DISABLED = False


def _compiled_gdn2_scan(
    q_f: torch.Tensor,
    k_f: torch.Tensor,
    v_f: torch.Tensor,
    write_f: torch.Tensor,
    forget_f: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    global _COMPILED_GDN2_SCAN
    if _COMPILED_GDN2_SCAN is None:
        _COMPILED_GDN2_SCAN = torch.compile(_gdn2_tensor_scan, mode="reduce-overhead", fullgraph=True)  # type: ignore[attr-defined]
    compiled = _COMPILED_GDN2_SCAN
    return compiled(q_f, k_f, v_f, write_f, forget_f, state)  # type: ignore[operator]


def _gdn2_compile_available(ref: torch.Tensor) -> bool:
    global _GDN2_COMPILE_DISABLED
    if _GDN2_COMPILE_DISABLED:
        return False
    if not _truthy_env("OMNICODER2026_GDN2_COMPILED_CHUNKS", "0"):
        return False
    if not ref.is_cuda:
        return False
    if not hasattr(torch, "compile"):
        return False
    try:
        major, minor = torch.cuda.get_device_capability(ref.device)
    except Exception:
        return False
    return (major, minor) >= (7, 5)


def _gdn2_jit_available(ref: torch.Tensor) -> bool:
    if not _truthy_env("OMNICODER2026_GDN2_JIT_SCAN", "0"):
        return False
    if not ref.is_cuda and not _truthy_env("OMNICODER2026_GDN2_JIT_SCAN_CPU", "0"):
        return False
    return True


def _gated_deltanet2_compiled_chunks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    write_gate: torch.Tensor,
    forget_gate: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optional compiled-chunk GDN2 path for fast CUDA cards.

    This is not an approximation. It uses the same fp32 recurrence as the
    reference path, split into fixed chunks so Dynamo/Inductor can remove Python
    overhead while the eager function remains the correctness fallback.
    """

    global _GDN2_COMPILE_DISABLED
    _check_qkv(q, k, v)
    if write_gate is None or forget_gate is None:
        raise ValueError("compiled GDN2 path requires tensor write and forget gates")
    q_f = q.float()
    k_f = F.normalize(k.float(), dim=-1)
    v_f = v.float()
    write_f = _prepare_gate(write_gate, v_f, sigmoid=True)
    forget_f = _prepare_gate(forget_gate, v_f, sigmoid=True)
    if not isinstance(write_f, torch.Tensor) or not isinstance(forget_f, torch.Tensor):
        raise ValueError("compiled GDN2 path requires tensor gates")
    state = _initial_state(q, v.shape[-1], initial_state)
    chunk = max(1, int(chunk_size if chunk_size is not None else _env_int("OMNICODER2026_GDN2_COMPILED_CHUNK_TOKENS", 32)))
    mode = str(os.environ.get("OMNICODER2026_GDN2_COMPILED_MODE", "chunked")).strip().lower()
    if mode in {"full", "fullscan", "scan"}:
        max_full_tokens = max(1, _env_int("OMNICODER2026_GDN2_COMPILED_FULL_MAX_TOKENS", 128))
        if q_f.shape[1] <= max_full_tokens:
            try:
                y, state = _compiled_gdn2_scan(q_f, k_f, v_f, write_f, forget_f, state)
                return y.to(dtype=v.dtype), state
            except Exception:
                _GDN2_COMPILE_DISABLED = True
                return gated_deltanet2_pytorch(
                    q,
                    k,
                    v,
                    write_gate=write_gate,
                    forget_gate=forget_gate,
                    initial_state=initial_state,
                )
    output = torch.empty_like(v_f)
    for start in range(0, q_f.shape[1], chunk):
        end = min(q_f.shape[1], start + chunk)
        q_chunk = q_f[:, start:end]
        k_chunk = k_f[:, start:end]
        v_chunk = v_f[:, start:end]
        write_chunk = write_f[:, start:end]
        forget_chunk = forget_f[:, start:end]
        if end - start != chunk:
            y_chunk, state = _gdn2_tensor_scan(q_chunk, k_chunk, v_chunk, write_chunk, forget_chunk, state)
            output[:, start:end] = y_chunk
            continue
        try:
            y_chunk, state = _compiled_gdn2_scan(q_chunk, k_chunk, v_chunk, write_chunk, forget_chunk, state)
        except Exception:
            _GDN2_COMPILE_DISABLED = True
            return gated_deltanet2_pytorch(
                q,
                k,
                v,
                write_gate=write_gate,
                forget_gate=forget_gate,
                initial_state=initial_state,
            )
        output[:, start:end] = y_chunk
    return output.to(dtype=v.dtype), state


def _gated_deltanet2_jit_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    write_gate: torch.Tensor,
    forget_gate: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_qkv(q, k, v)
    q_f = q.float()
    k_f = F.normalize(k.float(), dim=-1)
    v_f = v.float()
    write_f = _prepare_gate(write_gate, v_f, sigmoid=True)
    forget_f = _prepare_gate(forget_gate, v_f, sigmoid=True)
    if not isinstance(write_f, torch.Tensor) or not isinstance(forget_f, torch.Tensor):
        raise ValueError("JIT GDN2 path requires tensor gates")
    state = _initial_state(q, v.shape[-1], initial_state)
    y, state = _gdn2_tensor_scan_jit(q_f, k_f, v_f, write_f, forget_f, state)
    return y.to(dtype=v.dtype), state


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
    if isinstance(beta_f, torch.Tensor) and isinstance(forget_f, torch.Tensor):
        outputs, state = _gdn2_tensor_scan(q_f, k_f, v_f, beta_f, forget_f, state)
        return outputs.to(dtype=v.dtype), state

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
    if isinstance(write_f, torch.Tensor) and isinstance(forget_f, torch.Tensor):
        outputs, state = _gdn2_tensor_scan(q_f, k_f, v_f, write_f, forget_f, state)
        return outputs.to(dtype=v.dtype), state

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
    def __init__(self, d_model: int, n_heads: int, head_dim: Optional[int] = None, *, create_qkv: bool = True) -> None:
        super().__init__()
        self.cfg = RecurrentLinearAttentionConfig(d_model=d_model, n_heads=n_heads, head_dim=head_dim)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.cfg.key_dim
        inner = self.n_heads * self.head_dim
        if create_qkv:
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
        super().__init__(d_model, n_heads, head_dim, create_qkv=False)
        inner = self.n_heads * self.head_dim
        self.in_proj = nn.Linear(self.d_model, 3 * inner + 2 * self.n_heads, bias=True)
        with torch.no_grad():
            self.in_proj.bias[: 3 * inner].zero_()

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        q_key = prefix + "q_proj.weight"
        k_key = prefix + "k_proj.weight"
        v_key = prefix + "v_proj.weight"
        write_key = prefix + "write_gate_proj.weight"
        write_bias_key = prefix + "write_gate_proj.bias"
        forget_key = prefix + "forget_gate_proj.weight"
        forget_bias_key = prefix + "forget_gate_proj.bias"
        in_weight_key = prefix + "in_proj.weight"
        in_bias_key = prefix + "in_proj.bias"
        qkv_weight_key = prefix + "qkv_proj.weight"
        gate_weight_key = prefix + "gate_proj.weight"
        gate_bias_key = prefix + "gate_proj.bias"
        inner = self.n_heads * self.head_dim
        if in_weight_key not in state_dict:
            if all(key in state_dict for key in (q_key, k_key, v_key, write_key, forget_key)):
                state_dict[in_weight_key] = torch.cat(
                    (
                        state_dict[q_key],
                        state_dict[k_key],
                        state_dict[v_key],
                        state_dict[write_key],
                        state_dict[forget_key],
                    ),
                    dim=0,
                )
            elif qkv_weight_key in state_dict and gate_weight_key in state_dict:
                state_dict[in_weight_key] = torch.cat((state_dict[qkv_weight_key], state_dict[gate_weight_key]), dim=0)
        if in_bias_key not in state_dict:
            if all(key in state_dict for key in (write_bias_key, forget_bias_key)):
                state_dict[in_bias_key] = torch.cat(
                    (
                        torch.zeros(3 * inner, dtype=state_dict[write_bias_key].dtype, device=state_dict[write_bias_key].device),
                        state_dict[write_bias_key],
                        state_dict[forget_bias_key],
                    ),
                    dim=0,
                )
            elif gate_bias_key in state_dict:
                state_dict[in_bias_key] = torch.cat(
                    (
                        torch.zeros(3 * inner, dtype=state_dict[gate_bias_key].dtype, device=state_dict[gate_bias_key].device),
                        state_dict[gate_bias_key],
                    ),
                    dim=0,
                )
        for legacy_key in (
            q_key,
            k_key,
            v_key,
            write_key,
            write_bias_key,
            forget_key,
            forget_bias_key,
            qkv_weight_key,
            gate_weight_key,
            gate_bias_key,
        ):
            if legacy_key in state_dict:
                state_dict.pop(legacy_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        inner = self.n_heads * self.head_dim
        q_raw, k_raw, v_raw, write_gate, forget_gate = self.in_proj(x).split(
            (inner, inner, inner, self.n_heads, self.n_heads),
            dim=-1,
        )
        q = self._shape(q_raw)
        k = self._shape(k_raw)
        v = self._shape(v_raw)
        if _gdn2_compile_available(q):
            y, state = _gated_deltanet2_compiled_chunks(
                q,
                k,
                v,
                write_gate=write_gate,
                forget_gate=forget_gate,
                initial_state=initial_state,
            )
        elif _gdn2_jit_available(q):
            y, state = _gated_deltanet2_jit_scan(
                q,
                k,
                v,
                write_gate=write_gate,
                forget_gate=forget_gate,
                initial_state=initial_state,
            )
        else:
            y, state = gated_deltanet2_pytorch(
                q,
                k,
                v,
                write_gate=write_gate,
                forget_gate=forget_gate,
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
    "_gated_deltanet2_compiled_chunks",
    "_gated_deltanet2_jit_scan",
    "gated_deltanet2_pytorch",
    "kaczmarz_linear_attention_pytorch",
    "kda_pytorch",
]
