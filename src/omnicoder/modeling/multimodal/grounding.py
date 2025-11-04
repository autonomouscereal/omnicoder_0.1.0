from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroundingHead(nn.Module):
    """
    Open-vocab grounding head (YOLO-E inspired) that predicts boxes and class scores
    conditioned on a text embedding from PreAligner.

    Inputs:
      - vision_tokens: (B, T, C) from vision backbone
      - text_emb: (B, D) normalized embedding

    Outputs:
      - boxes: (B, N, 4) in normalized xywh
      - scores: (B, N)
      - text_scores: (B, N) alignment scores w.r.t. text
    """

    def __init__(self, d_model: int = 768, num_anchors: int = 100) -> None:
        super().__init__()
        self.num_anchors = int(num_anchors)
        self.box_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 4)
        )
        self.obj_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        self.fuse = nn.Sequential(
            nn.Linear(d_model + d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(self, vision_tokens: torch.Tensor, text_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Broadcast text embedding
        # Normalize text embedding via aten ops only
        _eps = 1e-12
        _sq = torch.ops.aten.mul.Tensor(text_emb, text_emb)
        _sum = torch.ops.aten.sum.dim_IntList(_sq, [-1], True)
        _norm = torch.ops.aten.sqrt.default(torch.ops.aten.add.Scalar(_sum, _eps))
        txt = torch.ops.aten.div.Tensor(text_emb, _norm)
        # Build a (B,T,1) ones tensor from aten.slice, then materialize (B,T,D) via multiply
        _bt1 = torch.ops.aten.slice.Tensor(vision_tokens, -1, 0, 1, 1)
        ones_bt1 = torch.ops.aten.add.Scalar(torch.ops.aten.mul.Scalar(_bt1, 0.0), 1.0)
        txt_b = torch.ops.aten.unsqueeze.default(txt, 1)
        txt_b = torch.ops.aten.mul.Tensor(txt_b, ones_bt1)
        from omnicoder.utils.torchutils import safe_concat as _safe_concat  # type: ignore
        h = self.fuse(_safe_concat([vision_tokens, txt_b], -1))
        # Predict per-token boxes and objectness
        boxes_lin = self.box_head(h)
        boxes = torch.ops.aten.sigmoid.default(boxes_lin)  # (B,T,4) normalized
        obj_lin = self.obj_head(h)
        obj_sig = torch.ops.aten.sigmoid.default(obj_lin)
        obj = torch.ops.aten.squeeze.dim(obj_sig, -1)  # (B,T)
        # Select top anchors with constant-K topk on a padded tensor (export-safe, no Tensor→Python ints)
        # Pad with -inf to ensure length >= K, then clamp gathered indices to T-1 for safe box/text gathers.
        # Build a (B,1) column and repeat to (B,K)
        _col = torch.ops.aten.slice.Tensor(obj, -1, 0, 1, 1)  # (B,1)
        _ones_bk = torch.ops.aten.repeat_interleave.self_int(_col, int(self.num_anchors), -1)  # (B,K)
        # Construct -inf pad anchored to obj's dtype
        _neg = float(torch.finfo(obj.dtype).min)
        _pad = torch.ops.aten.add.Scalar(torch.ops.aten.mul.Scalar(_ones_bk, 0.0), _neg)
        from omnicoder.utils.torchutils import safe_concat2 as _safe_cat  # type: ignore
        obj_pad = _safe_cat(obj, _pad, 1)  # (B, T+K)
        _tk = torch.ops.aten.topk.default(obj_pad, int(self.num_anchors), -1, True, True)
        scores, idx_raw = _tk[0], _tk[1]  # scores: (B,K) from padded array
        # Clamp indices to [0, T-1] to safely gather from original tensors
        _sh = torch.ops.aten._shape_as_tensor(obj)
        _T = torch.ops.aten.select.int(_sh, 0, 1)  # scalar tensor T
        _T = torch.ops.aten.to.dtype(_T, idx_raw.dtype, False, False)
        _Tm1 = torch.ops.aten.sub.Scalar(_T, 1)
        idx = torch.ops.aten.minimum.default(idx_raw, _Tm1)  # (B,K)
        # Gather boxes using clamped indices (repeat index along last dim to 4)
        _idx_u = torch.ops.aten.unsqueeze.default(idx, -1)  # (B,K,1)
        _gidx = torch.ops.aten.repeat_interleave.self_int(_idx_u, 4, -1)  # (B,K,4)
        top_boxes = torch.ops.aten.gather.default(boxes, 1, _gidx)
        # Text alignment via cosine similarity
        # Normalize fused tokens via aten ops only
        _sqh = torch.ops.aten.mul.Tensor(h, h)
        _sumh = torch.ops.aten.sum.dim_IntList(_sqh, [-1], True)
        _normh = torch.ops.aten.sqrt.default(torch.ops.aten.add.Scalar(_sumh, _eps))
        tok_n = torch.ops.aten.div.Tensor(h, _normh)
        txt_n = torch.ops.aten.unsqueeze.default(txt, 1)
        text_scores_full = torch.ops.aten.sum.dim_IntList(torch.ops.aten.mul.Tensor(tok_n, txt_n), [-1], False)
        top_text_scores = torch.ops.aten.gather.default(text_scores_full, 1, idx)
        return top_boxes, scores, top_text_scores


class SegmentationHead(nn.Module):
    """
    Segment-Anything-style mask head conditioned on points/boxes.

    Minimal export-friendly version: predicts a low-res mask per query token and
    upsamples to image size downstream.
    """

    def __init__(self, d_model: int = 768, hidden: int = 256) -> None:
        super().__init__()
        self.mask_mlp = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.to_mask = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, tokens: torch.Tensor, num_patches: int) -> torch.Tensor:
        # tokens: (B, T, C)
        h = self.mask_mlp(tokens)  # (B, T, H)
        m = self.to_mask(h.transpose(1, 2))  # (B,1,T)
        # Return low-res vectorized mask; caller reshapes/upsamples
        return torch.sigmoid(m)


