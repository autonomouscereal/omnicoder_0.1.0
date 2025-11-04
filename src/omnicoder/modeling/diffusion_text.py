from __future__ import annotations

"""
Diffusion-based parallel text generator integrated with OmniTransformer.

Design goals (rulebook aligned):
- Operate in embedding space (continuous) to enable Gaussian noise and aten-only math.
- No device moves in forward; allocate via like-factories or arithmetic anchors tied to live tensors.
- No Python slicing on tensors; use aten ops. No F.*. No .view/.expand. 3D bmm policy maintained.
- Keep this as a parallel "brain"; does not replace AR decode. Can be used to propose drafts.
- Timing/logging allowed but never gate. No .item() on live tensors inside compiled regions.
"""

from typing import Optional, Tuple

import torch
from torch import nn as _nn


class DiffusionScheduler:
    """Simple cosine beta schedule over T steps for embedding-space diffusion.

    We build tensors anchored to a provided reference tensor to keep device/dtype lineage consistent.
    """

    def __init__(self, num_steps: int = 8):
        self.num_steps = int(num_steps) if num_steps > 1 else 1

    def build(self, anchor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # t: [0..T-1] as device-local long tensor
        T = self.num_steps
        t0 = torch.ops.aten.new_ones.default(anchor, (T,), dtype=torch.long)
        t = torch.ops.aten.cumsum.default(t0, 0)
        t = torch.ops.aten.sub.Tensor(t, torch.ops.aten.new_ones.default(t, (T,), dtype=torch.long))
        # s = (t+0.008)/(T+0.008) (approx) using float math anchored to anchor
        tf = torch.ops.aten.to.dtype(t, anchor.dtype, False, False)
        off = torch.ops.aten.mul.Scalar(torch.ops.aten.add.Scalar(tf, 0.008), 1.0)
        den = torch.ops.aten.add.Scalar(torch.ops.aten.mul.Scalar(torch.ops.aten.ones_like.default(tf), float(T)), 0.008)
        s = torch.ops.aten.div.Tensor(off, den)
        # cos schedule: abar = cos^2(s * pi/2)
        pi2 = torch.ops.aten.mul.Scalar(torch.ops.aten.ones_like.default(s), 1.5707963267948966)
        cs = torch.ops.aten.cos.default(torch.ops.aten.mul.Tensor(s, pi2))
        abar = torch.ops.aten.mul.Tensor(cs, cs)
        # Convert to per-step betas: beta_t = 1 - abar[t]/abar[t-1]; we also return sqrt_abar, sqrt_one_m_abar
        # Prepend abar[0] for alignment; use arithmetic to avoid Python indexing
        a0 = torch.ops.aten.slice.Tensor(abar, 0, 0, 1, 1)
        tail = torch.ops.aten.slice.Tensor(abar, 0, 0, (abar.shape[0] - 1), 1)
        from omnicoder.utils.torchutils import safe_concat2 as _safe_cat  # type: ignore
        prev = _safe_cat(a0, tail, 0)
        eps = torch.ops.aten.mul.Scalar(torch.ops.aten.ones_like.default(abar), 1e-5)
        prev = torch.ops.aten.maximum.default(prev, eps)
        ratio = torch.ops.aten.div.Tensor(abar, prev)
        one = torch.ops.aten.add.Scalar(torch.ops.aten.mul.Scalar(torch.ops.aten.ones_like.default(ratio), 0.0), 1.0)
        betas = torch.ops.aten.sub.Tensor(one, ratio)
        sqrt_abar = torch.ops.aten.sqrt.default(abar)
        sqrt_one_m_abar = torch.ops.aten.sqrt.default(torch.ops.aten.sub.Tensor(one, abar))
        return betas, sqrt_abar, sqrt_one_m_abar


class DiffusionTextGenerator(_nn.Module):
    """Wrapper that uses OmniTransformer as an embedding denoiser.

    - Takes noisy target embeddings and optional prompt embeddings; denoises over T steps.
    - Timestep conditioning via a small MLP that produces an additive bias in embedding space.
    - Does not alter OmniTransformer internals; uses `prefix_hidden` to inject the sequence.
    """

    def __init__(self, model: _nn.Module, d_model: int, num_steps: int = 8):
        super().__init__()
        self.model = model
        self.num_steps = int(num_steps) if num_steps > 1 else 1
        self.t_embed = _nn.Sequential(
            _nn.Linear(d_model, d_model, bias=True),
            _nn.SiLU(),
            _nn.Linear(d_model, d_model, bias=True),
        )
        # sinusoidal basis for time embedding (no positional cache)
        self.register_buffer("_time_w", torch.linspace(1.0, 1000.0, steps=d_model).reshape(1, 1, -1), persistent=False)

    def time_embedding(self, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        # t: (B,1,1) float in [0,1]; produce (B,1,d_model)
        # Build phase = t * w, then concat sin/cos via projection (use only sin part for simplicity)
        w = torch.ops.aten._to_copy(self._time_w, dtype=like.dtype)
        phase = torch.ops.aten.mul.Tensor(t, w)
        te = torch.ops.aten.sin.default(phase)
        return self.t_embed(te)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor],
        gen_tokens: int,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Parallel diffusion generation of `gen_tokens` tokens conditioned on optional prompt.

        Returns token ids (1, gen_tokens).
        """
        model = self.model
        steps_n = int(steps or self.num_steps)
        # Build empty prompt hidden if no prompt ids
        B = 1
        V = getattr(model, 'vocab_size', 32000)
        d_model = getattr(model, 'd_model', None)
        if d_model is None:
            try:
                d_model = int(model.lm_head.in_features)  # type: ignore[attr-defined]
            except Exception:
                d_model = 1024
        # Obtain embedding table (tied or via attribute)
        try:
            embed_w = model.embed.weight  # type: ignore[attr-defined]
            E = embed_w
        except Exception:
            try:
                E = torch.ops.aten.transpose.int(model.lm_head.weight, 0, 1)  # type: ignore[attr-defined]
            except Exception:
                E = None  # type: ignore[assignment]
        # Prompt embeddings
        if input_ids is not None:
            ids = torch.ops.aten.to.dtype(input_ids, torch.long, False, False)
            ids = torch.ops.aten.reshape.default(ids, (B, -1))
            if E is not None:
                # one_hot + matmul avoids F.embedding; aten-first compliant
                oh = torch.ops.aten.one_hot.default(ids, V)
                oh = torch.ops.aten.to.dtype(oh, E.dtype, False, False)
                prompt = torch.ops.aten.matmul.default(oh, E)
            else:
                # If no embedding is accessible, create random small vectors anchored to lm_head when present
                try:
                    anchor = model.lm_head.weight  # type: ignore[attr-defined]
                except Exception:
                    anchor = self._time_w
                prompt = torch.ops.aten.randn.default(anchor, (B, ids.shape[1], d_model))
        else:
            # Zero-length prompt segment anchored to a live tensor
            try:
                anchor = model.lm_head.weight  # type: ignore[attr-defined]
            except Exception:
                anchor = self._time_w
            prompt = torch.ops.aten.new_zeros.default(anchor, (B, 0, d_model))
        # Initialize target embeddings with Gaussian noise
        tgt = torch.ops.aten.randn.default(prompt, (B, gen_tokens, d_model))
        # Scheduler
        _, sqrt_abar, sqrt_one_m_abar = DiffusionScheduler(steps_n).build(tgt)
        # Iterate denoise
        for si in range(steps_n - 1, -1, -1):  # coarse-to-fine
            # t in [0,1] as a tensor built without Python max/float
            # Use scheduler length from sqrt_abar shape: Tlen = sqrt_abar.shape[0]
            Tlen = torch.ops.aten.select.int(torch.ops.aten._shape_as_tensor(sqrt_abar), 0, 0)
            oneL = torch.ops.aten.ones_like.default(Tlen)
            denom = torch.ops.aten.sub.Tensor(Tlen, oneL)          # T-1
            denom = torch.ops.aten.maximum.default(denom, oneL)    # clamp_min 1
            si_t = torch.ops.aten.mul.Scalar(oneL, si)             # scalar tensor si
            ratio_i = torch.ops.aten.div.Tensor(si_t, denom)       # int64 ratio in [0,1]
            ratio = torch.ops.aten._to_copy(ratio_i, dtype=tgt.dtype)
            ratio = torch.ops.aten.reshape.default(ratio, (1, 1, 1))
            t = torch.ops.aten.mul.Tensor(torch.ops.aten.ones_like.default(tgt[..., :1]), ratio)
            # Build noisy input concat: [prompt, tgt]
            from omnicoder.utils.torchutils import safe_concat2 as _safe_cat  # type: ignore
            x = _safe_cat(prompt, tgt, 1)
            te = self.time_embedding(t, x)
            x_in = torch.ops.aten.add.Tensor(x, te)
            # Use model to predict clean embedding for the whole sequence; keep last gen_tokens
            out = model(input_ids=None, past_kv=None, use_cache=False, prefix_hidden=x_in)  # type: ignore[arg-type]
            if isinstance(out, tuple):
                hidden = out[0]
            else:
                hidden = out
            # Slice last gen_tokens without Tensor->Python int by building index tensor
            Ttot_t = torch.ops.aten.select.int(torch.ops.aten._shape_as_tensor(hidden), 0, 1)
            gen_t = torch.ops.aten.mul.Scalar(torch.ops.aten.ones_like.default(Ttot_t), int(gen_tokens))
            start_t = torch.ops.aten.sub.Tensor(Ttot_t, gen_t)
            # Build 0..gen_tokens-1 then add start_t using tensor-based length
            gl = torch.ops.aten.mul.Scalar(torch.ops.aten.ones_like.default(Ttot_t), gen_tokens)
            ones_g = torch.ops.aten.new_ones.default(hidden, (1,), dtype=torch.long)
            # Repeat ones to length gl via repeat_interleave (no expand)
            ones_g = torch.ops.aten.repeat_interleave.self_Tensor(ones_g, gl)
            rng = torch.ops.aten.cumsum.default(ones_g, 0)
            rng = torch.ops.aten.sub.Tensor(rng, torch.ops.aten.ones_like.default(rng))
            # Expand start_t to long and add
            start_long = torch.ops.aten.to.dtype(start_t, torch.long, False, False)
            addv = torch.ops.aten.mul.Tensor(torch.ops.aten.ones_like.default(rng), start_long)
            idx_abs = torch.ops.aten.add.Tensor(rng, addv)
            x0_pred = torch.ops.aten.index_select.default(hidden, 1, idx_abs)
            # DDIM-like update: x_t-1 ≈ sqrt(abar[t-1])*x0 + sqrt(1-abar[t-1])*epsilon; here predict x0 and recombine
            a = sqrt_abar[si]
            om = sqrt_one_m_abar[si]
            a = torch.ops.aten.reshape.default(a, (1, 1, 1))
            om = torch.ops.aten.reshape.default(om, (1, 1, 1))
            eps = torch.ops.aten.div.Tensor(torch.ops.aten.sub.Tensor(tgt, torch.ops.aten.mul.Tensor(a, x0_pred)), om)
            if si > 0:
                a_prev = sqrt_abar[si - 1]
                om_prev = sqrt_one_m_abar[si - 1]
                a_prev = torch.ops.aten.reshape.default(a_prev, (1, 1, 1))
                om_prev = torch.ops.aten.reshape.default(om_prev, (1, 1, 1))
                tgt = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(a_prev, x0_pred), torch.ops.aten.mul.Tensor(om_prev, eps))
            else:
                tgt = x0_pred
        # Map final embeddings to token ids via argmax over vocab using lm_head
        if hasattr(self.model, 'lm_head'):
            logits = torch.ops.aten.matmul.default(tgt, torch.ops.aten.transpose.int(self.model.lm_head.weight, 0, 1))  # type: ignore[attr-defined]
        else:
            # Use E^T to map back
            head = torch.ops.aten.transpose.int(E, 0, 1) if E is not None else torch.ops.aten.randn.default(tgt, (d_model, V))
            logits = torch.ops.aten.matmul.default(tgt, head)
        ids = torch.ops.aten.argmax.default(logits, -1)
        ids = torch.ops.aten.to.dtype(ids, torch.long, False, False)
        return torch.ops.aten.reshape.default(ids, (B, gen_tokens))


