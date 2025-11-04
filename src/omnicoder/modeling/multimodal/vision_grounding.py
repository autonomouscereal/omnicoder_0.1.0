from __future__ import annotations

import torch
import torch.nn as nn


class SimpleGroundingHead(nn.Module):
    """
    A lightweight open-vocab grounding head that takes visual tokens and a text
    query embedding, produces coarse bounding boxes and a confidence logit per
    proposal. This is a placeholder for a YOLO‑E–style module, but kept tiny and
    export‑friendly for smoke tests.
    """

    def __init__(self, d_model: int = 384, num_props: int = 10) -> None:
        super().__init__()
        self.num_props = int(num_props)
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        # Predict (cx, cy, w, h) normalized and a confidence logit
        self.bbox = nn.Linear(d_model, 4)
        self.conf = nn.Linear(d_model, 1)

    def forward(self, vision_tokens: torch.Tensor, text_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # vision_tokens: (B, T, C), text_embed: (B, C) normalized
        B, T, C = vision_tokens.shape
        # VisionBackbone may return inference tensors under @torch.inference_mode.
        # Clone to obtain a normal tensor usable by autograd ops (e.g., LayerNorm)
        vt = vision_tokens.clone()
        h = self.fc(vt)
        # Attend to tokens using the text query via dot product and pick top-K
        q = text_embed.unsqueeze(1)  # (B,1,C)
        attn = torch.softmax((q * h).sum(dim=-1), dim=-1)  # (B,T)
        # Weighted sum of token features to produce proposals (simple tiling)
        # For smoke test, tile the same weighted sum as multiple proposals
        feat = torch.einsum('bt,btc->bc', attn, h)
        feat = feat.unsqueeze(1).expand(B, self.num_props, C)
        boxes = torch.sigmoid(self.bbox(feat))  # (B,P,4) normalized
        conf = self.conf(feat).squeeze(-1)      # (B,P)
        return boxes, conf



class RepRTAHead(nn.Module):
    """
    YOLO‑E–inspired lightweight head with a RepRTA-like text–region alignment step.

    - Projects tokens and a text embedding to a shared space
    - Computes per-token relevance via scaled dot product with adapted text query
    - Forms proposals by weighted pooling of token features
    - Predicts (cx, cy, w, h) in normalized [0,1] and a confidence per proposal
    """

    def __init__(self, d_model: int = 384, num_props: int = 10) -> None:
        super().__init__()
        self.num_props = int(num_props)
        self.token_proj = nn.Linear(d_model, d_model, bias=False)
        self.text_proj = nn.Linear(d_model, d_model, bias=False)
        self.adapter = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.bbox = nn.Linear(d_model, 4)
        self.conf = nn.Linear(d_model, 1)

    def forward(self, vision_tokens: torch.Tensor, text_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # vision_tokens: (B,T,C), text_embed: (B,C)
        B, T, C = vision_tokens.shape
        # Clone to avoid "Inference tensors cannot be saved for backward" when backbone used inference_mode
        vt_in = vision_tokens.clone()
        tq_in = text_embed.clone()
        vt = self.token_proj(vt_in)
        tq = self.text_proj(tq_in)
        # RepRTA-like adaptation: combine global pooled tokens with text to refine query
        pooled = vt_in.mean(dim=1)
        from omnicoder.utils.torchutils import safe_concat2 as _safe_cat  # type: ignore
        rep = self.adapter(_safe_cat(tq, pooled, -1))  # (B,C)
        rep = torch.nn.functional.normalize(rep, dim=-1)
        # Token relevance via dot product
        rel = torch.softmax((vt * rep.unsqueeze(1)).sum(dim=-1) / max(C ** 0.5, 1e-6), dim=-1)  # (B,T)
        feat = torch.einsum('bt,btc->bc', rel, vt)  # (B,C)
        feat = self.norm(feat)
        # Tile proposals
        props = feat.unsqueeze(1).expand(B, self.num_props, C)
        boxes = torch.sigmoid(self.bbox(props))
        conf = self.conf(props).squeeze(-1)
        return boxes, conf


class SimpleSegHead(nn.Module):
    """
    Lightweight segmentation head that projects vision tokens to a coarse
    spatial mask, optionally guided by a text embedding via a relevance map.
    Produces a (B, 1, H, W) mask in [0,1] with a fixed grid inferred from T.
    """

    def __init__(self, d_model: int = 384, grid_hw: tuple[int, int] | None = None) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, 1)
        self.grid_hw = grid_hw  # (H,W) if provided; else inferred

    def forward(self, vision_tokens: torch.Tensor, text_embed: torch.Tensor | None = None) -> torch.Tensor:
        # vision_tokens: (B,T,C)
        B, T, C = vision_tokens.shape
        # Clone to avoid inference-mode autograd restriction in Linear/LayerNorm
        vt = vision_tokens.clone()
        logits = self.proj(vt).squeeze(-1)  # (B,T)
        # If text provided, weight tokens by a simple relevance score
        if text_embed is not None and text_embed.dim() == 2 and text_embed.size(0) == B and text_embed.size(1) == C:
            rel = torch.softmax((vt * text_embed.unsqueeze(1)).sum(dim=-1) / max(C ** 0.5, 1e-6), dim=-1)
            logits = logits + rel
        # Infer grid size
        if self.grid_hw is None:
            h = w = int(T ** 0.5)
            if h * w != T:
                # pad or trim to nearest square for a simple layout
                w = h
                h = T // max(1, w)
                if h * w <= 0:
                    h = 1; w = T
        else:
            h, w = int(self.grid_hw[0]), int(self.grid_hw[1])
        # Truncate or pad to h*w tokens
        if h * w <= T:
            logits = logits[:, : h * w]
        else:
            pad = h * w - T
            logits = torch.nn.functional.pad(logits, (0, pad), value=float('-inf'))
        mask = torch.sigmoid(torch.ops.aten.reshape.default(logits, (B, 1, h, w)))
        return mask

