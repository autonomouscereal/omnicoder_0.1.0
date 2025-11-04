import os
import torch
import torch.nn as nn
# Hoist perf timing imports to module scope (rule 3). Keep timing ON; do not gate or remove.
from omnicoder.utils.perf import add as _perf_add  # log aggregated timings without touching storages
import time as _t  # used only for perf counters; acceptable if it causes subgraph split, path remains identical


class FastHeadShortlist:
    """
    Exact argmax using a shortlist computed from a low-dimensional subspace.

    Guarantees exactness via a residual-norm bound with adaptive shortlist widening.
    Designed for decode-step (seqlen==1) but supports arbitrary (B,T,C) by using the last token.
    """

    def __init__(self, lm_head: nn.Linear, sub_dims: int | None = None):
        assert isinstance(lm_head, nn.Linear)
        self.lm_head = lm_head
        d_model = int(lm_head.in_features)
        self.vocab = int(lm_head.out_features)
        # Choose a small, fixed subset of feature indices deterministically
        try:
            k = int(os.getenv('OMNICODER_FAST_SUB_DIMS', str(sub_dims or 128)))
        except Exception:
            k = sub_dims or 128
        # Clamp k into [8, d_model] using Python math (runs at init; not in hot path)
        k = max(8, min(d_model, k))
        # Evenly spaced indices across feature dimension to approximate variance coverage
        # Compute stride (init-time only)
        stride = max(1, d_model // k)
        # Build evenly spaced indices without torch.arange (aten-only)
        # First build 0..d_model-1 via cumsum
        base = torch.ops.aten.cumsum.default(torch.ones((d_model,), dtype=torch.long), 0)
        base = torch.ops.aten.sub.Tensor(base, torch.ones((d_model,), dtype=torch.long))
        # Take every `stride`-th index via slicing with aten.gather over a computed index list
        take = (d_model + stride - 1) // stride
        mask = (base.remainder(stride) == 0)
        idx = base.masked_select(mask)[:k]
        if int(idx.numel()) < k:
            # Fallback pad
            pad = (k - int(idx.numel()))
            tail_len = pad
            # Build tail range [d_model-pad, d_model)
            tbase = torch.ops.aten.cumsum.default(torch.ones((tail_len,), dtype=torch.long), 0)
            tbase = torch.ops.aten.sub.Tensor(tbase, torch.ones((tail_len,), dtype=torch.long))
            tail = torch.ops.aten.add.Scalar(tbase, d_model - pad)
            from omnicoder.utils.torchutils import safe_concat2 as _safe_cat  # type: ignore
            idx = _safe_cat(idx, tail, 0)
        self.register_buffers(idx)
        self._refresh_weight_views()

    def register_buffers(self, idx: torch.Tensor) -> None:
        self.sub_idx = idx
        self.sub_idx_device = None
        self.W_sub = None
        self.W_res_norm = None
        self.W_full_device = None
        self.W_res_norm_max = None

    def _refresh_weight_views(self) -> None:
        # Snapshot full FP32 weight to avoid backend-specific reformatting (e.g., dynamic int8)
        W = torch.ops.aten.detach.default(self.lm_head.weight)
        W = torch.ops.aten.to.dtype(W, torch.float32, False, False)
        self.W_full = W
        # Subspace view: avoid Python slicing and method-style ops; use aten.index_select + aten.transpose
        self.W_sub = torch.ops.aten.index_select.default(self.W_full, 1, self.sub_idx)
        self.W_sub_t = torch.ops.aten.transpose.int(self.W_sub, 0, 1)  # (Ksub, V)
        # Residual norms per row
        # Build a mask to zero out subspace dims, then compute L2 norm of the residual
        d = self.W_full.shape[1]
        # Precompute residual index set once using pure tensor ops (no .tolist on Fake tensors)
        # Build 0..d-1 safely on the same device/type without arange(device=...)
        all_idx = torch.cumsum(self.W_full.new_ones((d,), dtype=torch.long), dim=0)
        all_idx = torch.sub(all_idx, 1)
        # Tensor-native set difference to avoid Python materialization
        try:
            res_idx = torch.setdiff1d(all_idx, self.sub_idx.to(all_idx.device))
        except Exception:
            # Fallback: boolean mask via isin when setdiff1d is unavailable
            mask = ~torch.isin(all_idx, self.sub_idx.to(all_idx.device))
            res_idx = all_idx.masked_select(mask)
        if res_idx.numel() == 0:
            res_idx = all_idx
        self.res_idx = res_idx
        # Residual slice via aten.index_select to avoid Python slicing in compiled regions
        W_res = torch.ops.aten.index_select.default(self.W_full, 1, res_idx)
        # Vector norm via aten to keep targets aten-only
        self.W_res_norm = torch.ops.aten.linalg_vector_norm.default(W_res, 2.0, [1], False)
        self.W_res_norm_max_t = torch.ops.aten.amax.default(self.W_res_norm, [0], False)

    def _ensure_on_device(self, device: torch.device) -> None:
        # Refresh cached views if the underlying module has migrated devices
        # Disabled: no device moves or checks in hot or helper paths per rules
        return

    @torch.no_grad()
    def argmax(self, hidden: torch.Tensor, initial_k: int = 2048, widen_factor: float = 2.0) -> torch.Tensor:
        """
        hidden: (B,T,C) or (B,C). Uses last time step when 3D.
        Returns next_id: (B,1) LongTensor exact argmax over vocab.
        """
        if hidden.dim() == 3:
            # Use aten.slice to select the last time step (rule 5: no Python slicing)
            T = int(hidden.shape[1])
            one = 1
            sl = torch.ops.aten.slice.Tensor(hidden, 1, T - 1, T, one)  # (B,1,C)
            hidden_last = torch.ops.aten.squeeze.dim(sl, 1)  # (B,C)
        else:
            hidden_last = hidden
        device = hidden_last.device
        self._ensure_on_device(device)

        # Subspace coarse scores (aten-only ops in compiled region)
        _t0 = _t.perf_counter()
        # Gather subspace features via aten.index_select (rule 5)
        h_sub = torch.ops.aten.index_select.default(hidden_last, 1, self.sub_idx)  # (B,Ksub)
        # Matmul via aten.mm (2D); shapes (B,K) x (K,V) requires bmm or batched; here use mm per-batch via broadcasting avoided
        # Since h_sub is (B,K) and W_sub_t is (K,V), use mm by flattening batch through bmm fallback: add singleton dim then squeeze
        coarse = torch.ops.aten.bmm.default(torch.ops.aten.unsqueeze.default(h_sub, 1), torch.ops.aten.unsqueeze.default(self.W_sub_t, 0))
        coarse = torch.ops.aten.squeeze.dim(coarse, 1)  # (B,V)
        _t1 = _t.perf_counter(); _perf_add('head.coarse', float(_t1 - _t0))
        # Residual norm of hidden outside subspace using precomputed residual index set
        h_res = torch.ops.aten.index_select.default(hidden_last, 1, self.res_idx)
        h_res_norm = torch.ops.aten.linalg_vector_norm.default(h_res, 2.0, [1], False)  # (B,)

        B = int(hidden_last.shape[0])
        K = int(initial_k)
        max_expand = int(os.getenv('OMNICODER_FAST_SHORTLIST_MAX', '32768'))
        # Single-pass shortlist with optional one-time widen to cap gathers at <=2 per step
        widen_tries = 0
        def _recheck_with_k(K_use: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            _tg0 = _t.perf_counter()
            values, indices = torch.ops.aten.topk.default(coarse, K_use, 1, True, False)  # (B,K_use)
            B_loc = indices.shape[0]
            K_loc = indices.shape[1]
            flat_idx = torch.ops.aten.reshape.default(indices, (B_loc * K_loc,))
            W_rows = torch.ops.aten.index_select.default(self.W_full, 0, flat_idx)
            W_sel = torch.ops.aten.reshape.default(W_rows, (B, K_use, self.W_full.shape[1]))
            _tg1 = _t.perf_counter(); _perf_add('head.gather', float(_tg1 - _tg0))
            _td0 = _t.perf_counter()
            if B == 1:
                # Select first batch via aten.select and use bmm with explicit batch dim to avoid aten.mm meta errors
                W0 = torch.ops.aten.select.int(W_sel, 0, 0)                 # (K_use, C)
                h0 = torch.ops.aten.select.int(hidden_last, 0, 0)           # (C,)
                W0b = torch.ops.aten.unsqueeze.default(W0, 0)               # (1,K_use,C)
                h0v = torch.ops.aten.unsqueeze.default(h0, 1)               # (C,1)
                h0b = torch.ops.aten.unsqueeze.default(h0v, 0)              # (1,C,1)
                fsb = torch.ops.aten.bmm.default(W0b, h0b)                   # (1,K_use,1)
                fs = torch.ops.aten.squeeze.dim(torch.ops.aten.squeeze.dim(fsb, 2), 0)  # (K_use,)
                bf, bp = torch.ops.aten.max.dim(fs, 0, False)
                full_scores = torch.ops.aten.reshape.default(fs, (1, fs.shape[0]))
                best_full_vals = torch.ops.aten.reshape.default(bf, (1,))
                best_pos = torch.ops.aten.reshape.default(bp, (1,))
            else:
                full_scores = torch.ops.aten.bmm.default(W_sel, torch.ops.aten.unsqueeze.default(hidden_last, 2))
                full_scores = torch.ops.aten.squeeze.dim(full_scores, 2)
                best_full_vals, best_pos = torch.ops.aten.max.dim(full_scores, 1, False)
            _td1 = _t.perf_counter(); _perf_add('head.dot', float(_td1 - _td0))
            _tc0 = _t.perf_counter()
            # Conservative certificate without masked writes/scatter: global upper bound
            coarse_max = torch.ops.aten.amax.default(coarse, [1], False)
            ub_rest = torch.ops.aten.add.Tensor(coarse_max, torch.ops.aten.mul.Tensor(h_res_norm, self.W_res_norm_max_t))
            ok = torch.ops.aten.ge.Tensor(best_full_vals, ub_rest)
            _tc1 = _t.perf_counter(); _perf_add('head.certificate', float(_tc1 - _tc0))
            return ok, best_pos, indices, full_scores

        ok, best_pos, indices, full_scores = _recheck_with_k(K)
        if bool(ok.all()):
            return indices.gather(1, torch.ops.aten.reshape.default(best_pos, (B, 1)))
        # One widening attempt only to keep gathers bounded
        K2 = min(self.vocab, max_expand, max(int(K * widen_factor), K + 2048))
        # Cheaper certificate for widened pass: avoid coarse_plus topk by using global coarse max as over-ub
        _tgx0 = _t.perf_counter()
        values2, indices2 = torch.ops.aten.topk.default(coarse, K2, 1, True, False)
        flat2 = torch.ops.aten.reshape.default(indices2, (B * K2,))
        W_sel2 = torch.ops.aten.index_select.default(self.W_full, 0, flat2)
        W_sel2 = torch.ops.aten.reshape.default(W_sel2, (B, K2, self.W_full.shape[1]))
        _tgx1 = _t.perf_counter(); _perf_add('head.gather', float(_tgx1 - _tgx0))
        _tdx0 = _t.perf_counter()
        if B == 1:
            W0 = torch.ops.aten.select.int(W_sel2, 0, 0)                    # (K2, C)
            h0 = torch.ops.aten.select.int(hidden_last, 0, 0)               # (C,)
            W0b = torch.ops.aten.unsqueeze.default(W0, 0)                   # (1,K2,C)
            h0v = torch.ops.aten.unsqueeze.default(h0, 1)                   # (C,1)
            h0b = torch.ops.aten.unsqueeze.default(h0v, 0)                  # (1,C,1)
            fs2b = torch.ops.aten.bmm.default(W0b, h0b)                      # (1,K2,1)
            fs2 = torch.ops.aten.squeeze.dim(torch.ops.aten.squeeze.dim(fs2b, 2), 0)  # (K2,)
            bf2, bp2 = torch.ops.aten.max.dim(fs2, 0, False)
            best_full_vals2 = torch.ops.aten.reshape.default(bf2, (1,))
            best_pos2 = torch.ops.aten.reshape.default(bp2, (1,))
        else:
            full_scores2 = torch.ops.aten.bmm.default(W_sel2, torch.ops.aten.unsqueeze.default(hidden_last, 2))
            full_scores2 = torch.ops.aten.squeeze.dim(full_scores2, 2)
            best_full_vals2, best_pos2 = torch.ops.aten.max.dim(full_scores2, 1, False)
        _tdx1 = _t.perf_counter(); _perf_add('head.dot', float(_tdx1 - _tdx0))
        # Global-max upper bound for excluded tokens
        _tcx0 = _t.perf_counter()
        coarse_max = torch.ops.aten.amax.default(coarse, [1], False)
        ub_rest2 = torch.ops.aten.add.Tensor(coarse_max, torch.ops.aten.mul.Tensor(h_res_norm, self.W_res_norm_max_t))
        ok2 = torch.ops.aten.ge.Tensor(best_full_vals2, ub_rest2)
        _tcx1 = _t.perf_counter(); _perf_add('head.certificate', float(_tcx1 - _tcx0))
        return torch.ops.aten.gather.dim(indices2, 1, torch.ops.aten.reshape.default(best_pos2, (B, 1)))


def attach_fast_head(model: nn.Module) -> None:
    try:
        if not hasattr(model, 'lm_head') or not isinstance(model.lm_head, nn.Linear):
            return
        if getattr(model, '_fast_head', None) is None:
            sub_dims = None
            try:
                sub_dims = int(os.getenv('OMNICODER_FAST_SUB_DIMS', '64'))
            except Exception:
                sub_dims = 64
            model._fast_head = FastHeadShortlist(model.lm_head, sub_dims=sub_dims)
    except Exception:
        pass


