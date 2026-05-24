import torch, torch.nn as nn, torch.nn.functional as F
from omnicoder.utils.logger import get_logger

import torch
import torch.nn as nn

from omnicoder.utils.torchutils import safe_concat2 as _safe_cat  # type: ignore

import math as _math

class TopKRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        k: int = 2,
        temperature: float = 1.0,
        jitter_noise: float = 0.0,
        use_gumbel: bool = False,
        expert_dropout_p: float = 0.0,
        sinkhorn_iters: int = 0,
        sinkhorn_tau: float = 1.0,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.k = k

        dummy = torch.ops.aten.new_zeros.default(torch.tensor(0.0), (1,))
        self.register_buffer("inv_temp", torch.ops.aten.div.Scalar(torch.ops.aten.new_ones.default(dummy, (1,)), temperature))
        self.register_buffer("jitter_mult", torch.ops.aten.add.Scalar(dummy, jitter_noise))

        self.gate_weight = nn.Parameter(torch.empty((n_experts, d_model)))
        self.gate_bias = nn.Parameter(torch.zeros((n_experts,)))
        self.cond_proj_weight = nn.Parameter(torch.empty((n_experts, d_model)))
        self.cond_proj_bias = nn.Parameter(torch.zeros((n_experts,)))

        nn.init.normal_(self.gate_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cond_proj_weight, mean=0.0, std=0.02)

    @staticmethod
    def _gumbel_like(t: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        u = torch.ops.aten.rand_like.default(t)
        u = torch.ops.aten.add.Scalar(u, eps)
        l1 = torch.ops.aten.log.default(u)
        nl1 = torch.ops.aten.neg.default(l1)
        nl1 = torch.ops.aten.add.Scalar(nl1, eps)
        l2 = torch.ops.aten.log.default(nl1)
        g = torch.ops.aten.neg.default(l2)
        return g

    def forward(self, x: torch.Tensor, cond: dict | None = None):
        # ALL COPIES AT THE VERY TOP — 100% device agnostic
        inv_temp = torch.ops.aten._to_copy.default(self.inv_temp, dtype=x.dtype, device=x.device)
        jitter_mult = torch.ops.aten._to_copy.default(self.jitter_mult, dtype=x.dtype, device=x.device)

        w = torch.ops.aten._to_copy.default(self.gate_weight, dtype=x.dtype, device=x.device)
        b = torch.ops.aten._to_copy.default(self.gate_bias, dtype=x.dtype, device=x.device)
        logits = torch.ops.aten.linear.default(x, w, b)

        # FIXED: cond_term ALWAYS uses exact logits shape (B, 1, E) — this was the source of the broadcast error
        B = torch.ops.aten.sym_size.int(logits, 0)
        E = torch.ops.aten.sym_size.int(logits, 2)
        cond_term = torch.ops.aten.new_zeros.default(logits, [B, 1, E])
        logits = torch.ops.aten.add.Tensor(logits, cond_term)

        jitter = torch.ops.aten.mul.Tensor(torch.ops.aten.randn_like.default(logits), jitter_mult)
        logits = torch.ops.aten.add.Tensor(logits, jitter)

        logits = torch.ops.aten.mul.Tensor(logits, inv_temp)
        probs_full = torch.ops.aten.softmax.int(logits, -1)

        _tk = torch.ops.aten.topk.default(logits, self.k, -1, True, True)
        topk_vals = _tk[0]
        idx = _tk[1]
        scores = torch.ops.aten.softmax.int(topk_vals, -1)

        importance = torch.ops.aten.mean.dim(probs_full, [0, 1], False)
        idx_flat = torch.ops.aten.reshape.default(idx, (torch.ops.aten.sym_size.int(idx, 0) * torch.ops.aten.sym_size.int(idx, 1) * torch.ops.aten.sym_size.int(idx, 2),))
        onehot = torch.ops.aten.one_hot(torch.ops.aten.to.dtype(idx_flat, torch.long, False, False), self.n_experts)
        counts = torch.ops.aten.sum.dim_IntList(onehot.to(dtype=probs_full.dtype), [0], False)
        den_t = torch.ops.aten.to.dtype(torch.ops.aten.sum.dim_IntList(counts, [0], False), probs_full.dtype, False, False)
        den_t = torch.ops.aten.clamp_min.default(den_t, 1.0)
        load = torch.ops.aten.div.Tensor(counts, den_t)

        return idx, scores, probs_full, {"importance": importance, "load": load}


class HierarchicalRouter(nn.Module):
    def __init__(self, d_model: int, n_experts: int, group_sizes: list[int] | None = None, k: int = 2, temperature: float = 1.0, jitter_noise: float = 0.0):
        super().__init__()
        self.n_experts = n_experts
        self.k = k

        dummy = torch.ops.aten.new_zeros.default(torch.tensor(0.0), (1,))
        self.register_buffer("inv_temp", torch.ops.aten.div.Scalar(torch.ops.aten.new_ones.default(dummy, (1,)), temperature))
        self.register_buffer("jitter_mult", torch.ops.aten.add.Scalar(dummy, jitter_noise))

        self.gate_expert_weight = nn.Parameter(torch.empty((n_experts, d_model)))
        self.gate_expert_bias = nn.Parameter(torch.zeros((n_experts,)))
        self.gate_group_weight = nn.Parameter(torch.empty((n_experts, d_model)))
        self.gate_group_bias = nn.Parameter(torch.zeros((n_experts,)))

        nn.init.normal_(self.gate_expert_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.gate_group_weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, cond: dict | None = None):
        # ALL COPIES AT THE VERY TOP
        inv_temp = torch.ops.aten._to_copy.default(self.inv_temp, dtype=x.dtype, device=x.device)
        jitter_mult = torch.ops.aten._to_copy.default(self.jitter_mult, dtype=x.dtype, device=x.device)

        w = torch.ops.aten._to_copy.default(self.gate_expert_weight, dtype=x.dtype, device=x.device)
        b = torch.ops.aten._to_copy.default(self.gate_expert_bias, dtype=x.dtype, device=x.device)
        expert_logits = torch.ops.aten.linear.default(x, w, b)

        gw = torch.ops.aten._to_copy.default(self.gate_group_weight, dtype=x.dtype, device=x.device)
        gb = torch.ops.aten._to_copy.default(self.gate_group_bias, dtype=x.dtype, device=x.device)
        group_bias = torch.ops.aten.linear.default(x, gw, gb)
        expert_logits = torch.ops.aten.add.Tensor(expert_logits, group_bias)

        jitter = torch.ops.aten.mul.Tensor(torch.ops.aten.randn_like.default(expert_logits), jitter_mult)
        expert_logits = torch.ops.aten.add.Tensor(expert_logits, jitter)

        expert_logits = torch.ops.aten.mul.Tensor(expert_logits, inv_temp)
        probs_full = torch.ops.aten.softmax.int(expert_logits, -1)

        _tk = torch.ops.aten.topk.default(expert_logits, self.k, -1, True, True)
        topk_vals = _tk[0]
        idx = _tk[1]
        scores = torch.ops.aten.softmax.int(topk_vals, -1)

        importance = torch.ops.aten.mean.dim(probs_full, [0, 1], False)
        idx_flat = torch.ops.aten.reshape.default(idx, (torch.ops.aten.sym_size.int(idx, 0) * torch.ops.aten.sym_size.int(idx, 1) * torch.ops.aten.sym_size.int(idx, 2),))
        onehot = torch.ops.aten.one_hot(torch.ops.aten.to.dtype(idx_flat, torch.long, False, False), self.n_experts)
        counts = torch.ops.aten.sum.dim_IntList(onehot.to(dtype=probs_full.dtype), [0], False)
        den_t = torch.ops.aten.to.dtype(torch.ops.aten.sum.dim_IntList(counts, [0], False), probs_full.dtype, False, False)
        den_t = torch.ops.aten.clamp_min.default(den_t, 1.0)
        load = torch.ops.aten.div.Tensor(counts, den_t)

        return idx, scores, probs_full, {"importance": importance, "load": load}

class MultiHeadRouter(nn.Module):
    """
    Multi-head gating: multiple independent linear gates vote on experts.

    We compute per-head logits over experts, softmax to probabilities, then
    average probabilities across heads to obtain a consensus distribution.
    Top-k selection and aux stats follow the same pattern as TopKRouter.

    This approximates multi-head gating variants and serves as a drop-in
    alternative to classic TopK gating.
    """

    def __init__(self, d_model: int, n_experts: int, k: int = 2, num_gates: int = 4, temperature: float = 1.0, jitter_noise: float = 0.0):
        super().__init__()
        self.n_experts = int(n_experts)
        self.k = int(k)
        self.num_gates = max(1, int(num_gates))
        self.temperature = max(1e-6, float(temperature))
        self.jitter_noise = float(jitter_noise)
        self.gates = nn.ModuleList([nn.Linear(d_model, n_experts, bias=False) for _ in range(self.num_gates)])
        self.last_aux: dict | None = None

    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
        # Logging disabled in hot path per no-IO rule
        probs_accum = None
        for gate in self.gates:
            # Copy weight/bias to exact input device/dtype (no .data mutation!)
            w = torch.ops.aten._to_copy.default(gate.weight, dtype=x.dtype, device=x.device)
            b = torch.ops.aten._to_copy.default(gate.bias, dtype=x.dtype, device=x.device) if gate.bias is not None else None
            logits = torch.ops.aten.linear.default(x, w, b)

            if self.training and self.jitter_noise > 0.0:
                logits = torch.ops.aten.add.Tensor(
                    logits,
                    torch.ops.aten.mul.Scalar(torch.ops.aten.randn_like.default(logits), float(self.jitter_noise))
                )
            logits = torch.ops.aten.mul.Scalar(logits, float(1.0 / self.temperature))
            p = torch.ops.aten.softmax.int(logits, -1)
            probs_accum = p if probs_accum is None else torch.ops.aten.add.Tensor(probs_accum, p)

        assert probs_accum is not None
        probs_full = torch.ops.aten.mul.Scalar(probs_accum, float(1.0 / float(self.num_gates)))

        # Top-k selection
        _tk = torch.ops.aten.topk.default(probs_full, self.k, -1, True, True)
        topk_vals, idx = _tk[0], _tk[1]

        # Normalize selected scores per token
        den = torch.ops.aten.sum.dim_IntList(topk_vals, [-1], True)
        scores = torch.ops.aten.div.Tensor(topk_vals, torch.ops.aten.clamp_min.default(den, 1e-9))

        # Aux
        with torch.no_grad():
            importance = torch.ops.aten.mean.dim(probs_full, [0, 1], False)
            load = torch.ops.aten.mean.dim(
                torch.ops.aten.to.dtype(torch.ops.aten.gt.Scalar(probs_full, 0.0), probs_full.dtype, False, False),
                [0, 1], False
            )
        _aux = {"importance": importance, "load": load}
        return idx, scores, probs_full, _aux


class GRINGate(nn.Module):
    """
    Gradient-Informed Routing (GRIN) gate (training-ready approximation).

    - Computes base expert logits from inputs
    - Predicts per-token difficulty and modulates logits
    - Applies masked-softmax sampling with straight-through (ST) estimator
      to obtain top-k one-hot selections while keeping gradients via soft
      probabilities
    - Exposes auxiliary stats: importance, load
    """

    def __init__(self, d_model: int, n_experts: int, k: int = 2, temperature: float = 1.0, jitter_noise: float = 0.0, st_tau: float = 1.0, mask_drop: float = 0.0):
        super().__init__()
        self.n_experts = int(n_experts)
        self.k = int(k)
        self.temperature = max(1e-6, float(temperature))
        self.jitter_noise = float(jitter_noise)
        self.st_tau = max(1e-6, float(st_tau))
        self.mask_drop = max(0.0, min(1.0, float(mask_drop)))
        self.base_gate = nn.Linear(d_model, n_experts, bias=False)
        # Difficulty predictor (token-level): produces a scalar in [0,1]
        self.diff_pred = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.GELU(),
            nn.Linear(d_model // 2, 1, bias=False),
        )
        self.last_aux: dict | None = None

    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
        # Logging disabled in hot path per no-IO rule

        # === FIX: use aten.linear + copied weight (no nn.Linear) ===
        w = torch.ops.aten._to_copy.default(self.base_gate.weight, dtype=x.dtype, device=x.device)
        b = torch.ops.aten._to_copy.default(self.base_gate.bias, dtype=x.dtype, device=x.device) if self.base_gate.bias is not None else None
        logits = torch.ops.aten.linear.default(x, w, b)   # (B,T,E)
        # === end fix ===

        # === FIX: replace self.diff_pred(x) with explicit aten ops (no nn.Sequential) ===
        ln = self.diff_pred[0]          # LayerNorm
        lin1 = self.diff_pred[1]        # Linear(d_model -> d_model//2)
        lin2 = self.diff_pred[3]        # Linear(d_model//2 -> 1)

        # LayerNorm (copy weight + bias)
        ln_w = torch.ops.aten._to_copy.default(ln.weight, dtype=x.dtype, device=x.device)
        ln_b = torch.ops.aten._to_copy.default(ln.bias, dtype=x.dtype, device=x.device)
        h = torch.ops.aten.layer_norm.default(x, (x.shape[-1],), ln_w, ln_b, 1e-5)

        # Linear 1 (no bias)
        w1 = torch.ops.aten._to_copy.default(lin1.weight, dtype=x.dtype, device=x.device)
        h = torch.ops.aten.linear.default(h, w1, None)

        # GELU
        h = torch.ops.aten.gelu.default(h)

        # Linear 2 (no bias)
        w2 = torch.ops.aten._to_copy.default(lin2.weight, dtype=x.dtype, device=x.device)
        diff = torch.ops.aten.linear.default(h, w2, None)
        diff = torch.ops.aten.sigmoid.default(diff)   # (B,T,1)
        # === end fix ===

        # Modulate logits per token: higher difficulty -> softer distribution
        logits = logits / (1.0 + diff)

        if self.training and self.jitter_noise > 0.0:
            logits = torch.ops.aten.add.Tensor(
                logits,
                torch.ops.aten.mul.Scalar(torch.ops.aten.randn_like.default(logits), float(self.jitter_noise))
            )
        logits = torch.ops.aten.mul.Scalar(logits, float(1.0 / self.temperature))

        # Masked-softmax sampling with optional random drop to encourage exploration
        if self.training and self.mask_drop > 0.0:
            mask = (torch.ops.aten.lt.Scalar(torch.ops.aten.rand_like.default(logits), float(self.mask_drop)))
            neg = logits.new_tensor(-1e9)
            logits = torch.ops.aten.add.Tensor(
                torch.ops.aten.mul.Tensor(torch.ops.aten.to.dtype(mask, logits.dtype, False, False), neg),
                torch.ops.aten.mul.Tensor(torch.ops.aten.to.dtype(torch.ops.aten.logical_not.default(mask), logits.dtype, False, False), logits)
            )

        # Soft probabilities used for gradients
        probs_soft = torch.ops.aten.softmax.int(logits, -1)  # (B,T,E)

        # Hard top-k indices from logits
        topk_vals, idx = torch.ops.aten.topk.default(logits, int(self.k), -1, True, True)

        # Straight-through estimator...
        if idx.dim() == 2:
            idx3 = torch.ops.aten.reshape.default(idx, (idx.shape[0], idx.shape[1], 1))
        else:
            idx3 = idx

        hard_full = torch.ops.aten.one_hot.default(idx3, int(probs_soft.shape[-1]))
        hard_full = torch.ops.aten.to.dtype(hard_full, probs_soft.dtype, False, False)
        hard_any = torch.ops.aten.amax.default(hard_full, 2)

        probs_st = torch.ops.aten.add.Tensor(
            torch.ops.aten.sub.Tensor(hard_any, probs_soft),
            probs_soft
        )

        scores = torch.ops.aten.gather.default(probs_st, -1, idx)
        scores = torch.ops.aten.div.Tensor(
            scores,
            torch.ops.aten.clamp_min.default(torch.ops.aten.sum.dim_IntList(scores, [-1], True), 1e-9)
        )

        with torch.no_grad():
            importance = torch.ops.aten.mean.dim(probs_soft, [0, 1], False)
            load = torch.ops.aten.mean.dim(hard_any, [0, 1], False)

        _aux = {"importance": importance, "load": load}
        return idx, scores, probs_soft, _aux


class LLMRouter(nn.Module):
    """
    Context-aware router inspired by LLM-based routing ideas.

    This module augments token representations with a lightweight
    contextual encoder (1-layer Transformer-style self-attention)
    before producing expert logits. It remains export-friendly and
    efficient, and can be enabled via env or training flags.

    Interface matches other routers: returns (idx, scores, probs_full).
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        k: int = 2,
        temperature: float = 1.0,
        jitter_noise: float = 0.0,
        num_heads: int = 4,
    ):
        super().__init__()

        self.n_experts = int(n_experts)
        self.k = int(k)
        self.temperature = max(1e-6, float(temperature))
        self.jitter_noise = float(jitter_noise)
        # Minimal self-attention encoder (single block) for context-aware routing
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=max(1, int(num_heads)), batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2, bias=False),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model, bias=False),
        )
        # Expert gate after contextualization
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.last_aux: dict | None = None

    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
        # Logging disabled in hot path per no-IO rule
        # Lightweight context encoder
        h = self.ln(x)
        # Causal mask to preserve autoregressive nature
        b, t, c = h.shape
        try:
            mask = torch.ops.aten.triu.default(torch.ops.aten.new_ones.default(h, (t, t), dtype=torch.bool), 1)
        except Exception:
            mask = None
        try:
            # MultiheadAttention expects (B,T,C) with batch_first=True
            attn_out, _ = self.attn(h, h, h, attn_mask=mask)
        except Exception:
            attn_out = h
        h2 = torch.ops.aten.add.Tensor(h, attn_out)
        try:
            ff_out = self.ff(h2)
        except Exception:
            ff_out = h2
        ctxt = torch.ops.aten.add.Tensor(h2, ff_out)
        logits = self.gate(ctxt)
        if self.training and self.jitter_noise > 0.0:
            logits = torch.ops.aten.add.Tensor(
                logits,
                torch.ops.aten.mul.Scalar(torch.ops.aten.randn_like.default(logits), float(self.jitter_noise))
            )
        logits = torch.ops.aten.mul.Scalar(logits, float(1.0 / self.temperature))
        probs_full = torch.ops.aten.softmax.int(logits, -1)
        # Top-k
        topk_vals, idx = torch.ops.aten.topk.default(logits, self.k, -1, True, True)
        scores = torch.ops.aten.softmax.int(topk_vals, -1)
        # Aux
        with torch.no_grad():
            importance = torch.ops.aten.mean.dim(probs_full, [0, 1], False)
            # Count selections via one_hot and sum; normalize by total selections
            idx_flat = torch.ops.aten.reshape.default(idx, (idx.shape[0] * idx.shape[1] * idx.shape[2],))  # (B*T*K,)
            oh = torch.ops.aten.one_hot(idx_flat.to(dtype=torch.long), self.n_experts)  # (B*T*K,E)
            counts = torch.ops.aten.sum.dim_IntList(oh.to(dtype=probs_full.dtype), [0], False)  # (E,)
            den_t = torch.ops.aten.clamp_min.default(torch.ops.aten.sum.dim_IntList(counts, [0], False), 1.0)
            load = torch.ops.aten.div.Tensor(counts, den_t)
        # Avoid persisting tensors on module during forward; return aux if needed by caller
        _aux = {"importance": importance, "load": load}
        # Logging disabled in hot path per no-IO rule
        return idx, scores, probs_full, _aux


class InteractionRouter(nn.Module):
    """
    Skeleton for an interaction-aware router (I2MoE-like) that can route
    based on the presence of paired modality embeddings (e.g., text+image).

    Inputs:
      - x: token features (B,T,C)
      - cond: optional dict of normalized modality embeddings (e.g., from PreAligner)

    It biases expert logits using a small MLP over concatenated [x || cond_proj]
    when conditioning is provided. Falls back to TopK-like behavior otherwise.
    """

    def __init__(self, d_model: int, n_experts: int, k: int = 2, temperature: float = 1.0):
        super().__init__()
        self.n_experts = int(n_experts)
        self.k = int(k)
        self.temperature = max(1e-6, float(temperature))
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        # Simple conditioning projector
        self.cond_proj = nn.Linear(d_model * 2, d_model, bias=False)
        self.last_aux: dict | None = None

    def forward(self, x: torch.Tensor, cond: dict | None = None):
        # x: (B,T,C)
        # Logging disabled in hot path per no-IO rule
        h = x
        if cond is not None:
            # Use the first available conditioning vector (e.g., image or text)
            c = None
            for key in ("image", "text", "audio", "video"):
                if isinstance(cond, dict) and key in cond and isinstance(cond[key], torch.Tensor):
                    c = cond[key]
                    break
            if c is not None and c.dim() == 2 and (c.shape[0] == x.shape[0]):
                # Broadcast conditioning across time and combine
                c_expand = torch.ops.aten.repeat_interleave.self_int(torch.ops.aten.unsqueeze.default(c, 1), x.shape[1], 1)

                hc = _safe_cat(x, c_expand, -1)
                try:
                    h = self.cond_proj(hc)
                except Exception:
                    h = x
        logits = torch.ops.aten.mul.Scalar(self.gate(h), float(1.0 / self.temperature))
        probs = torch.ops.aten.softmax.int(logits, -1)
        # Replace torch.sort with aten.topk to avoid ONNX/export incompatibilities and Python call targets
        _tk = torch.ops.aten.topk.default(logits, self.k, -1, True, True)
        topk_vals, idx = _tk[0], _tk[1]
        scores = torch.ops.aten.softmax.int(topk_vals, -1)
        with torch.no_grad():
            importance = torch.ops.aten.mean.dim(probs, [0, 1], False)
            # Flatten idx without -1 to avoid unbacked SymInt creation under compile
            _Bsym = torch.ops.aten.sym_size.int(idx, 0)
            _Tsym = torch.ops.aten.sym_size.int(idx, 1)
            _Nsym = _Bsym * _Tsym
            idx_flat = torch.ops.aten.reshape.default(idx, (_Nsym,))
            oh = torch.ops.aten.one_hot(torch.ops.aten.to.dtype(idx_flat, torch.long, False, False), self.n_experts)
            counts = torch.ops.aten.sum.dim_IntList(torch.ops.aten.to.dtype(oh, probs.dtype, False, False), [0], False)
            den = torch.ops.aten.clamp_min.default(torch.ops.aten.sum.dim_IntList(counts, [0], False), 1.0)
            load = torch.ops.aten.div.Tensor(counts, den)
        # Avoid persisting tensors on module during forward; return aux if needed by caller
        _aux = {"importance": importance, "load": load}
        # Logging disabled in hot path per no-IO rule
        return idx, scores, probs, _aux
