import torch, torch.nn as nn, torch.nn.functional as F
from omnicoder.utils.logger import get_logger


class TopKRouter(nn.Module):
    """Top-k token router for MoE with optional regularization and noise.

    Exposes `last_aux` after forward with:
      - z_loss: Switch-Transformer style logit-norm regularizer
      - importance: mean gate probs per expert
      - load: fraction of tokens whose top-k includes the expert

    NOTE (extremely verbose rule adherence):
    - No torch.function calls in compiled/graph regions. Use torch.ops.aten.* overloads explicitly.
    - No method-style ops (e.g., t.unsqueeze, masked_fill, where) in hot paths; compose with aten ops.
    - Avoid Python arithmetic on tensors (e.g., logits + noise); use aten.add/mul.
    - Avoid Python int/float over SymInt where possible; prefer aten.size/shape propagation.
    - No io/env/device/dtype branching in forward.
    - All random draws via aten.rand_like/randn_like anchored to existing tensors.

    Implementation notes (why this looks the way it does):
    - We removed forward-time module state mutations (e.g., last_aux writes) because they
      trigger Dynamo higher-order side-effects under checkpointing and compiled graphs.
      Instead, aux is returned as an optional 4th tuple element by routers that compute it.
    - All logic is aten-only to keep CUDA Graphs and export stable across runs.
    - Fixed shapes and removal of .item() or locals() usage prevents trace breaks.
    """

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
        # Clamp scalars once; used only as Python constants and fed into aten.Scalar overloads below
        self.temperature = max(1e-6, float(temperature))
        self.jitter_noise = float(jitter_noise)
        self.use_gumbel = bool(use_gumbel)
        self.expert_dropout_p = max(0.0, min(1.0, float(expert_dropout_p)))
        self.sinkhorn_iters = max(0, int(sinkhorn_iters))
        self.sinkhorn_tau = max(1e-6, float(sinkhorn_tau))
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        # Optional conditioning projection to bias expert logits with pooled modality embedding
        self.cond_proj = nn.Linear(d_model, n_experts, bias=False)
        # Populated after forward
        self.last_aux: dict | None = None
        # When True, stores per-token probs tensor for KL computation (costly);
        # training loop toggles this only when needed.
        self.store_probs_for_kl: bool = False
        # Optional stochastic top-k via Gumbel noise (masked-softmax sampling proxy)
        self.sample_gumbel_topk: bool = False

    @staticmethod
    def _gumbel_like(t: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """Draw Gumbel(0,1) noise via aten ops only.
        Construction: g = -log(-log(u + eps) + eps), where u ~ Uniform(0,1).
        - RNG: aten.rand_like.default keeps lineage anchored to input tensor (device/dtype-safe).
        - Math: aten.add/log/neg chain avoids method targets; eps prevents log(0).
        """
        u = torch.ops.aten.rand_like.default(t)
        u = torch.ops.aten.add.Scalar(u, float(eps))
        l1 = torch.ops.aten.log.default(u)
        nl1 = torch.ops.aten.neg.default(l1)
        nl1 = torch.ops.aten.add.Scalar(nl1, float(eps))
        l2 = torch.ops.aten.log.default(nl1)
        g = torch.ops.aten.neg.default(l2)
        return g

    def forward(self, x: torch.Tensor, cond: dict | None = None):
        # x: (B, T, C)
        # Logging disabled in hot path per no-IO rule
        # Gating logits via linear; stays as call_module to integrate with FakeTensor safely
        logits = self.gate(x)  # (B, T, E)
        # Add conditioning bias when provided (uses first available modality vector)
        try:
            v0 = None
            if cond is not None:
                v0 = cond.get("image")
                if v0 is None:
                    v0 = cond.get("text")
                    if v0 is None:
                        v0 = cond.get("audio")
                        if v0 is None:
                            v0 = cond.get("video")
            if v0 is not None:
                try:
                    b = self.cond_proj(v0)
                    logits = torch.ops.aten.add.Tensor(logits, torch.ops.aten.unsqueeze.default(b, 1))
                except Exception:
                    pass
        except Exception:
            pass
        # Temperature and optional noise for exploration during training
        if self.training:
            if self.jitter_noise > 0.0:
                if self.use_gumbel:
                    # logits += jitter * gumbel_like(logits) via aten add/mul
                    logits = torch.ops.aten.add.Tensor(
                        logits,
                        torch.ops.aten.mul.Scalar(self._gumbel_like(logits), float(self.jitter_noise))
                    )
                else:
                    # logits += jitter * randn_like(logits) via aten only
                    logits = torch.ops.aten.add.Tensor(
                        logits,
                        torch.ops.aten.mul.Scalar(torch.ops.aten.randn_like.default(logits), float(self.jitter_noise))
                    )
            # Optional expert dropout to improve robustness (aten-only composition)
            if self.expert_dropout_p > 0.0:
                # Draw Bernoulli mask by thresholding uniform(0,1): rand_like < p
                drop_mask = torch.ops.aten.lt.Scalar(torch.ops.aten.rand_like.default(logits), float(self.expert_dropout_p))
                # Prevent dropping all experts per token: keep argmax expert via equality mask over expert indices
                max_idx = torch.ops.aten.argmax.default(logits, -1, True)  # (B,T,1)
                E = torch.ops.aten.sym_size.int(logits, 2)
                rng = torch.ops.aten.cumsum.default(
                    torch.ops.aten.new_ones.default(logits, (E,), dtype=torch.long), 0
                )
                rng = torch.ops.aten.sub.Tensor(rng, 1)                   # 0..E-1
                rng = torch.ops.aten.reshape.default(rng, (1, 1, E)) # (1,1,E)
                mx = torch.ops.aten.reshape.default(max_idx, (torch.ops.aten.sym_size.int(max_idx, 0), torch.ops.aten.sym_size.int(max_idx, 1), 1))
                keep_mask = torch.ops.aten.eq.Tensor(rng, mx)             # (B,T,E)
                # Force keep by zeroing drop_mask at keep positions: (~keep_mask) & drop_mask
                drop_mask = torch.ops.aten.logical_and.default(torch.ops.aten.logical_not.default(keep_mask), drop_mask)
                # Compose masked logits arithmetically (no masked_fill/where)
                neg = torch.finfo(logits.dtype).min
                logits = torch.ops.aten.add.Tensor(
                    torch.ops.aten.mul.Tensor(torch.ops.aten.to.dtype(drop_mask, logits.dtype, False, False), neg),
                    torch.ops.aten.mul.Tensor(
                        torch.ops.aten.to.dtype(torch.ops.aten.logical_not.default(drop_mask), logits.dtype, False, False),
                        logits,
                    ),
                )
        # Scale by inverse temperature via aten.mul.Scalar to avoid aten.div ambiguity
        logits = torch.ops.aten.mul.Scalar(logits, float(1.0 / self.temperature))

        # Compute probabilities (aten softmax to avoid F.* targets)
        probs_full = torch.ops.aten.softmax.int(logits, -1)  # (B, T, E)

        # Unified path: no export/tracing gating; always prefer topk for performance (aten.topk)
        if self.training and self.sample_gumbel_topk:
            g = self._gumbel_like(logits)
            noisy = torch.ops.aten.add.Tensor(logits, g)
            _tk = torch.ops.aten.topk.default(noisy, self.k, -1, True, True)
            topk_vals, idx = _tk[0], _tk[1]
        else:
            _tk = torch.ops.aten.topk.default(logits, self.k, -1, True, True)
            topk_vals, idx = _tk[0], _tk[1]

        # Normalize selected expert scores across top-k (aten softmax)
        scores = torch.ops.aten.softmax.int(topk_vals, -1)  # (B, T, k)

        # z-loss over logits to keep them bounded — all aten math
        z = torch.ops.aten.logsumexp.default(logits, [-1])  # (B, T)
        z2 = torch.ops.aten.mul.Tensor(z, z)
        z_loss = torch.ops.aten.mean.default(z2)

        # Router auxiliary stats (no grads needed except for z-loss)
        with torch.no_grad():
            # Importance: mean over (B,T)
            importance = torch.ops.aten.mean.dim(probs_full, [0, 1], False)  # (E,)
            # Load: fraction of tokens selecting each expert in top-k
            # Avoid reshape(-1,) to keep shapes explicit for Inductor
            idx_flat = torch.ops.aten.reshape.default(idx, (torch.ops.aten.sym_size.int(idx, 0) * torch.ops.aten.sym_size.int(idx, 1) * torch.ops.aten.sym_size.int(idx, 2),))  # (B*T*k,)
            # Build counts without index_add to avoid ONNX scatter duplicate-index warning
            onehot = torch.ops.aten.one_hot(idx_flat.to(dtype=torch.long), self.n_experts)  # (B*T*k,E)
            counts = torch.ops.aten.sum.dim_IntList(onehot.to(dtype=probs_full.dtype), [0], False)  # (E,)
            # Denominator as tensor: total selections = sum(counts) == B*T*k; clamp to avoid div-by-zero
            den_t = torch.ops.aten.to.dtype(torch.ops.aten.sum.dim_IntList(counts, [0], False), probs_full.dtype, False, False)
            den_t = torch.ops.aten.clamp_min.default(den_t, 1.0)
            load = torch.ops.aten.div.Tensor(counts, den_t)
        # Cache aux (z_loss retains grad). Do NOT mutate module attributes during forward.
        # Return aux as part of outputs for callers that need it; default callers can ignore.
        _aux = None
        if self.store_probs_for_kl:
            _aux = {"importance": importance, "load": load, "z_loss": z_loss, "probs": probs_full}

        # Logging disabled in hot path per no-IO rule
        return idx, scores, probs_full, _aux


class HierarchicalRouter(nn.Module):
    """
    Hierarchical/multimodal-aware router.

    Two-stage gating:
      1) Group gate picks a modality/task-specific subgroup (softly or argmax)
      2) Expert gate scores experts; experts outside selected groups are masked

    Groups are defined as contiguous expert ranges via `group_sizes`.
    Falls back to flat routing when no groups are provided.
    """

    def __init__(self, d_model: int, n_experts: int, group_sizes: list[int] | None = None, k: int = 2, temperature: float = 1.0, jitter_noise: float = 0.0):
        super().__init__()
        self.n_experts = int(n_experts)
        self.k = int(k)
        self.temperature = max(1e-6, float(temperature))
        self.jitter_noise = float(jitter_noise)
        self.group_sizes = list(group_sizes) if group_sizes is not None else None
        self.num_groups = sum(1 for _ in (self.group_sizes or [])) if self.group_sizes is not None else 0
        # Group gate and expert gate
        self.gate_group = nn.Linear(d_model, self.num_groups if self.num_groups > 0 else 1, bias=False)
        self.gate_expert = nn.Linear(d_model, n_experts, bias=False)
        # Optional conditioning projection (text/image/audio/video) → d_model
        # Used when a pre-aligned embedding is provided to bias routing.
        self.cond_proj = nn.Linear(d_model * 2, d_model, bias=False)
        self.last_aux: dict | None = None

        # Precompute expert index ranges per group
        self._ranges: list[tuple[int, int]] = []
        if self.group_sizes:
            start = 0
            for sz in self.group_sizes:
                self._ranges.append((start, start + int(sz)))
                start += int(sz)
            # Clip to n_experts
            self._ranges = [(max(0, a), min(self.n_experts, b)) for a, b in self._ranges]

    def forward(self, x: torch.Tensor, cond: dict | None = None):
        # x: (B, T, C)
        # Logging removed from hot path to keep compile/cudagraph compatibility
        bt = x.shape[0] * x.shape[1]
        h = x
        # If conditioning is provided (e.g., from PreAligner), combine with tokens
        if isinstance(cond, dict):
            c = None
            for key in ("text", "image", "audio", "video"):
                v = cond.get(key, None)
                if isinstance(v, torch.Tensor) and (v.dim() == 2) and (v.shape[0] == x.shape[0]):
                    c = v
                    break
            if c is not None:
                # Replace method-style expand with aten.repeat_interleave
                c_expand = torch.ops.aten.repeat_interleave.self_int(torch.ops.aten.unsqueeze.default(c, 1), x.shape[1], 1)
                try:
                    from omnicoder.utils.torchutils import safe_concat2 as _safe_cat  # type: ignore
                    h = self.cond_proj(_safe_cat(x, c_expand, -1))
                except Exception:
                    h = x
        # Expert logits
        expert_logits = self.gate_expert(h)  # (B,T,E)
        # Optional noise for exploration
        if self.training and self.jitter_noise > 0.0:
            expert_logits = torch.ops.aten.add.Tensor(
                expert_logits,
                torch.ops.aten.mul.Scalar(torch.ops.aten.randn_like.default(expert_logits), float(self.jitter_noise))
            )
        # Temperature scaling via aten.mul.Scalar
        expert_logits = torch.ops.aten.mul.Scalar(expert_logits, float(1.0 / self.temperature))

        if self.num_groups > 0:
            group_logits = self.gate_group(h)  # (B,T,G)
            group_probs = torch.ops.aten.softmax.int(group_logits, -1)  # (B,T,G)
            # Select top-1 group per token (argmax) for masking
            g_idx = torch.ops.aten.argmax.default(group_probs, -1, False)  # (B,T)
            # Build mask over experts outside the selected group per token (aten ops only)
            mask = torch.ops.aten.new_zeros.default(expert_logits, expert_logits.shape, dtype=torch.bool)
            for gid, (a, b) in enumerate(self._ranges):
                # Tokens where selected group == gid
                sel = torch.ops.aten.unsqueeze.default(torch.ops.aten.eq.Scalar(g_idx, int(gid)), -1)
                # Mask all experts outside [a,b)
                if a > 0:
                    rng = torch.ops.aten.cumsum.default(torch.ops.aten.new_ones.default(mask, (self.n_experts,), dtype=torch.long), 0)
                    rng = torch.ops.aten.sub.Tensor(rng, 1)
                    rng = torch.ops.aten.reshape.default(rng, (1, 1, self.n_experts))
                    mask = torch.ops.aten.logical_or.default(mask, torch.ops.aten.logical_and.default(sel, torch.ops.aten.lt.Scalar(rng, a)))
                if b < self.n_experts:
                    rng = torch.ops.aten.cumsum.default(torch.ops.aten.new_ones.default(mask, (self.n_experts,), dtype=torch.long), 0)
                    rng = torch.ops.aten.sub.Tensor(rng, 1)
                    rng = torch.ops.aten.reshape.default(rng, (1, 1, self.n_experts))
                    mask = torch.ops.aten.logical_or.default(mask, torch.ops.aten.logical_and.default(sel, torch.ops.aten.ge.Scalar(rng, b)))
            # Compose masked logits arithmetically
            neg = torch.finfo(expert_logits.dtype).min
            expert_logits = torch.ops.aten.add.Tensor(
                torch.ops.aten.mul.Tensor(torch.ops.aten.to.dtype(mask, expert_logits.dtype, False, False), neg),
                torch.ops.aten.mul.Tensor(
                    torch.ops.aten.to.dtype(torch.ops.aten.logical_not.default(mask), expert_logits.dtype, False, False),
                    expert_logits,
                ),
            )
        probs_full = torch.ops.aten.softmax.int(expert_logits, -1)
        # Top-k selection among experts
        _tk = torch.ops.aten.topk.default(expert_logits, self.k, -1, True, True)
        topk_vals, idx = _tk[0], _tk[1]
        scores = torch.ops.aten.softmax.int(topk_vals, -1)

        # Aux statistics
        with torch.no_grad():
            importance = torch.ops.aten.mean.dim(probs_full, [0, 1], False)  # (E,)
            # Count selections via one_hot and sum across all tokens (B*T*K)
            idx_flat = torch.ops.aten.reshape.default(idx, (idx.shape[0] * idx.shape[1] * idx.shape[2],))  # (B*T*K,)
            oh = torch.ops.aten.one_hot(idx_flat.to(dtype=torch.long), self.n_experts)  # (B*T*K,E)
            counts = torch.ops.aten.sum.dim_IntList(oh.to(dtype=probs_full.dtype), [0], False)  # (E,)
            den_t = torch.ops.aten.clamp_min.default(torch.ops.aten.sum.dim_IntList(counts, [0], False), 1.0)
            load = torch.ops.aten.div.Tensor(counts, den_t)
        # Avoid persisting tensors on module during forward; return aux if needed by caller
        _aux = {"importance": importance, "load": load}
        # Logging disabled in hot path per no-IO rule
        return idx, scores, probs_full, _aux


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
            logits = gate(x)  # (B,T,E)
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
            load = torch.ops.aten.mean.dim(torch.ops.aten.to.dtype(torch.ops.aten.gt.Scalar(probs_full, 0.0), probs_full.dtype, False, False), [0, 1], False)
        # Avoid persisting tensors on module during forward; return aux if needed by caller
        _aux = {"importance": importance, "load": load}
        # Logging disabled in hot path per no-IO rule
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
        logits = self.base_gate(x)  # (B,T,E)
        diff = torch.ops.aten.sigmoid.default(self.diff_pred(x))  # (B,T,1)
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
            # Replace masked_fill with arithmetic composition to avoid masked_subblock issues
            neg = logits.new_tensor(-1e9)
            logits = torch.ops.aten.add.Tensor(
                torch.ops.aten.mul.Tensor(torch.ops.aten.to.dtype(mask, logits.dtype, False, False), neg),
                torch.ops.aten.mul.Tensor(torch.ops.aten.to.dtype(torch.ops.aten.logical_not.default(mask), logits.dtype, False, False), logits)
            )
        # Soft probabilities used for gradients
        probs_soft = torch.ops.aten.softmax.int(logits, -1)  # (B,T,E)
        # Hard top-k indices from logits (export-safe sort when tracing/onnx)
        topk_vals, idx = torch.ops.aten.topk.default(logits, int(self.k), -1, True, True)
        # Straight-through estimator: build one-hot hard assignment, add soft gradient
        # Normalize idx to (B,T,K) so the K-reduction is well-defined
        if idx.dim() == 2:
            idx3 = torch.ops.aten.reshape.default(idx, (idx.shape[0], idx.shape[1], 1))
        else:
            idx3 = idx
        # Build hard one-hot without in-place scatter_: (B,T,K,E)
        # aten.one_hot does not accept dtype arg; cast after creation via aten.to.dtype
        hard_full = torch.ops.aten.one_hot.default(idx3, int(probs_soft.shape[-1]))
        hard_full = torch.ops.aten.to.dtype(hard_full, probs_soft.dtype, False, False)
        # Collapse K to get (B,T,E)
        hard_any = torch.ops.aten.amax.default(hard_full, 2)
        # Compose straight-through probabilities in (B,T,E)
        probs_st = torch.ops.aten.add.Tensor(
            torch.ops.aten.sub.Tensor(hard_any, probs_soft),
            probs_soft
        )
        # Scores normalized over selected experts per token from composed probs
        scores = torch.ops.aten.gather.default(probs_st, -1, idx)
        scores = torch.ops.aten.div.Tensor(scores, torch.ops.aten.clamp_min.default(torch.ops.aten.sum.dim_IntList(scores, [-1], True), 1e-9))
        with torch.no_grad():
            importance = torch.ops.aten.mean.dim(probs_soft, [0, 1], False)
            load = torch.ops.aten.mean.dim(hard_any, [0, 1], False)
        # Avoid persisting tensors on module during forward; return aux if needed by caller
        _aux = {"importance": importance, "load": load}
        # Logging disabled in hot path per no-IO rule
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
        import math as _math
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
                from omnicoder.utils.torchutils import safe_concat2 as _safe_cat  # type: ignore
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
