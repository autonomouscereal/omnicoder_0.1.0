import torch, torch.nn as nn, torch.nn.functional as F


class RecurrentMemory(nn.Module):
    """
    Lightweight recurrent memory compressor for infinite-context-style usage.

    Compresses a sequence of hidden states (B, T, C) into M memory slots (B, M, C)
    using gated pooling and a small projection MLP. Intended to be fed into the
    attention layers as prefix memory (converted to latent K/V inside attention).

    The compressor is cheap and export-friendly; it avoids recurrence in favor of
    per-window aggregation, suitable for decode loops with sliding windows.
    
    HISTORY (why earlier versions hurt CG/TPS):
    - Forward wrote `last_mem` inside the graph to expose activations for aux losses. Under
      torch.compile + CUDA Graphs, this introduced side effects that changed the weakref set
      across warmup and replay, triggering cudagraph AssertionError and blocking CG engagement.
    
    CURRENT (why this is better):
    - We keep the attribute for API compatibility but do not assign to it during forward.
      Callers can recompute auxiliary losses from returned tensors outside compiled regions.
    """

    def __init__(self, d_model: int, num_slots: int = 8, reduce: str = "mean"):
        super().__init__()
        self.num_slots = max(1, int(num_slots))
        self.reduce = reduce
        # Slot queries to attend over tokens for weighted pooling
        self.slot_query = nn.Parameter(torch.randn(self.num_slots, d_model) * 0.02)
        # Precompute scale for attention logits to avoid math ops in forward
        self._attn_scale = float(1.0 / (d_model ** 0.5))
        # Simple projection after pooling to re-normalize
        self.post = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # NOTE (history and rules): We used to update `last_mem` during forward for aux losses.
        # Under torch.compile + CUDA Graphs this creates side effects that break weakref
        # tracking across warmup/replay. We keep the attribute for compatibility but do not
        # write to it during forward. Training code should recompute from returned tensors.
        # This adheres to the no-mutation rule and improves cudagraph stability and TPS.
        self.last_mem: torch.Tensor | None = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        hidden: (B, T, C)
        returns memory: (B, M, C)
        """
        b, t, c = hidden.shape
        # Repeat queries per batch: (B,M,C)
        q0 = torch.ops.aten.unsqueeze.default(self.slot_query, 0)            # (1,M,C)
        q = torch.ops.aten.repeat_interleave.self_int(q0, int(b), 0)         # (B,M,C)
        # Compute logits with batched matmul: q @ hidden^T => (B,M,T)
        ht = torch.ops.aten.transpose.int(hidden, 1, 2)                      # (B,C,T)
        attn_logits = torch.ops.aten.bmm.default(q, ht)                       # (B,M,T)
        attn_logits = torch.ops.aten.mul.Scalar(attn_logits, self._attn_scale)
        attn = torch.ops.aten.softmax.int(attn_logits, -1)
        # Weighted sum over time via bmm after flattening M into batch
        attn_flat = torch.ops.aten.reshape.default(attn, (int(b * self.num_slots), 1, int(t)))
        hid_rep = torch.ops.aten.repeat_interleave.self_int(hidden, int(self.num_slots), 0)  # (B*M,T,C)
        mem_flat = torch.ops.aten.bmm.default(attn_flat, hid_rep)             # (B*M,1,C)
        mem = torch.ops.aten.reshape.default(mem_flat, (int(b), int(self.num_slots), int(c)))
        # Optional simple reduce fusion with global mean for stability
        if self.reduce == "mean":
            mean_pool = torch.ops.aten.mean.dim(hidden, [1], True)            # (B,1,C)
            mp_rep = torch.ops.aten.repeat_interleave.self_int(mean_pool, int(self.num_slots), 1)
            mem = torch.ops.aten.mul.Scalar(mem, 0.5)
            mem = torch.ops.aten.add.Tensor(mem, torch.ops.aten.mul.Scalar(mp_rep, 0.5))
        mem_out = self.post(mem)
        # Do not assign to self.last_mem here (no forward-time state writes)
        return mem_out


class CompressiveKV(nn.Module):
    """
    Compressive memory for latent K/V streams (InfiniAttention-style proxy).

    Given latent K/V with time dimension, compresses a long prefix into a fixed
    number of slots via lightweight pooling + projection, returning compact
    (k_mem, v_mem) to concatenate before the recent window.

    Export-friendly and stateless; no recurrence, uses simple segment pooling.
    """

    def __init__(self, latent_dim: int, slots: int = 8):
        super().__init__()
        self.slots = max(1, int(slots))
        # Small projection after pooling for each of K and V
        self.post_k = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim, bias=False),
        )
        self.post_v = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim, bias=False),
        )
        # Learned write gate: modulates how much of each segment to keep
        self.write_gate = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 1, bias=True),
        )
        # Do not mutate module state during forward; stats can be computed externally.
        self.last_gate_mean: torch.Tensor | None = None
        # Learned retention head for keep/compress/drop classification per segment
        self.retention_head = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, 3)
        )
        # Last retention logits placeholder; not updated in forward.
        self.last_retention_logits: torch.Tensor | None = None

    @torch.no_grad()
    def _segment_bounds(self, T: int) -> list[tuple[int, int]]:
        # Evenly divide range [0, T) into self.slots segments
        seg = []
        if T <= 0:
            return [(0, 0)] * self.slots
        base = T // self.slots
        rem = T % self.slots
        start = 0
        for i in range(self.slots):
            end = start + base + (1 if i < rem else 0)
            seg.append((start, end))
            start = end
        return seg

    def forward(self, k_lat: torch.Tensor, v_lat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        k_lat, v_lat: (B, H, T, DL)
        returns: k_mem, v_mem with shape (B, H, M, DL)
        """
        assert k_lat.ndim == 4 and v_lat.ndim == 4
        b, h, t, dl = k_lat.shape
        segs = self._segment_bounds(t)
        # Avoid zero-length segments
        k_mem = []
        v_mem = []
        for (a, bnd) in segs:
            if bnd <= a:
                # Pad with zeros using aten.new_zeros anchored to inputs (device/dtype-safe).
                # Avoids .to/.new_tensor, keeps hot path aten-only and graph-friendly.
                k_mem.append(torch.ops.aten.new_zeros.default(k_lat, k_lat[:, :, :1, :].shape))
                v_mem.append(torch.ops.aten.new_zeros.default(v_lat, v_lat[:, :, :1, :].shape))
            else:
                k_seg = k_lat[:, :, a:bnd, :].mean(dim=2, keepdim=True)
                v_seg = v_lat[:, :, a:bnd, :].mean(dim=2, keepdim=True)
                # Write gate from K segment
                try:
                    g = torch.sigmoid(self.write_gate(k_seg))  # (B,H,1,1)
                    k_seg = k_seg * g
                    v_seg = v_seg * g
                    # Skip caching stats in forward to avoid graph side effects.
                except Exception:
                    pass
                # Retention decision: keep/compress/drop
                try:
                    logits = self.retention_head(k_seg)
                    # Soft decisions (export-friendly):
                    #   keep: 0 -> pass through
                    #   compress: 1 -> scale by 0.5
                    #   drop: 2 -> scale by 0.0
                    probs = torch.softmax(logits, dim=-1)  # (B,H,1,3)
                    keep_w = probs[..., 0:1]
                    comp_w = probs[..., 1:2]
                    drop_w = probs[..., 2:3]
                    scale = keep_w + 0.5 * comp_w + 0.0 * drop_w
                    k_seg = k_seg * scale
                    v_seg = v_seg * scale
                except Exception:
                    pass
                k_mem.append(k_seg)
                v_mem.append(v_seg)
        from omnicoder.utils.torchutils import safe_concat as _safe_concat  # type: ignore
        k_mem_t = _safe_concat(k_mem, 2)
        v_mem_t = _safe_concat(v_mem, 2)
        return self.post_k(k_mem_t), self.post_v(v_mem_t)


def compressive_aux_loss(mem: torch.Tensor, hidden: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Auxiliary loss encouraging memory slots to summarize hidden states.

    Computes cosine distance between each slot and the global mean of hidden states:
    hidden: (B,T,C), mem: (B,M,C)
    """
    import torch.nn.functional as F
    h_mean = hidden.mean(dim=1, keepdim=True)  # (B,1,C)
    mem_n = F.normalize(mem, dim=-1)
    h_n = F.normalize(h_mean, dim=-1)
    cos = (mem_n * h_n).sum(dim=-1)  # (B,M)
    loss = (1.0 - cos).mean(dim=-1)  # (B,)
    return loss.mean() if reduce else loss


class LandmarkIndexer(nn.Module):
    """
    Landmark tokens for random-access style context: summarize sequence into M tokens.

    Splits the time axis into M segments and produces one landmark per segment via
    gated pooling + projection. Intended for full-seq passes (not decode-step).
    """

    def __init__(self, d_model: int, num_landmarks: int = 8):
        super().__init__()
        self.num_landmarks = max(1, int(num_landmarks))
        self.query = nn.Parameter(torch.randn(self.num_landmarks, d_model) * 0.02)
        self._attn_scale = float(1.0 / (d_model ** 0.5))
        self.post = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.last_landmarks: torch.Tensor | None = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        b, t, c = hidden.shape
        # Build (B,M,C) queries without expand
        q0 = torch.ops.aten.unsqueeze.default(self.query, 0)                  # (1,M,C)
        q = torch.ops.aten.repeat_interleave.self_int(q0, int(b), 0)          # (B,M,C)
        # Logits via bmm
        ht = torch.ops.aten.transpose.int(hidden, 1, 2)                        # (B,C,T)
        logits = torch.ops.aten.bmm.default(q, ht)                              # (B,M,T)
        logits = torch.ops.aten.mul.Scalar(logits, self._attn_scale)
        w = torch.ops.aten.softmax.int(logits, -1)                              # (B,M,T)
        # Pool via bmm on flattened batches
        w_flat = torch.ops.aten.reshape.default(w, (int(b * self.num_landmarks), 1, int(t)))
        hid_rep = torch.ops.aten.repeat_interleave.self_int(hidden, int(self.num_landmarks), 0)  # (B*M,T,C)
        lm_flat = torch.ops.aten.bmm.default(w_flat, hid_rep)                   # (B*M,1,C)
        lm = torch.ops.aten.reshape.default(lm_flat, (int(b), int(self.num_landmarks), int(c)))  # (B,M,C)
        lm = self.post(lm)
        # Do not assign last_landmarks during forward; return output only
        return lm

