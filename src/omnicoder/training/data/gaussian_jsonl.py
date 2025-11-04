from __future__ import annotations

"""
Gaussian JSONL dataset loader for 3D/2D splats.

JSONL schema (each line may include either or both blocks; optional text/qa fields):

{
  "text": str?, "answer": str?, "question": str?,
  "gs3d": {
    "pos": [[x,y,z], ...],                # (N,3)
    "cov": [[[...3],[...3],[...3]], ...]  # (N,3,3) OR
    "cov_diag": [[sx2,sy2,sz2], ...],     # (N,3)
    "rgb": [[r,g,b], ...],                # (N,3) in [0,1]
    "opa": [[a], ...],                    # (N,1) in [0,1]
    "K": [[fx,0,cx], [0,fy,cy], [0,0,1]], # (3,3)
    "R": [[...3],[...3],[...3]],          # (3,3)
    "t": [tx,ty,tz],                      # (3,)
    "H": 224, "W": 224                   # optional image size, defaults 224
  },
  "gs2d": {
    "mean": [[u,v], ...],                 # (N,2)
    "cov_diag": [[sx2,sy2], ...],         # (N,2)
    "rgb": [[r,g,b], ...],                # (N,3)
    "opa": [[a], ...],                    # (N,1)
    "H": 224, "W": 224                   # optional
  }
}

This module performs minimal validation and converts lists to tensors suitable for
the MultimodalComposer.fuse_all Gaussian arguments.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset


def _to_tensor(x: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, dtype=dtype)


class GaussianJSONL(Dataset[Dict[str, Any]]):
    def __init__(self, jsonl_path: str) -> None:
        self.path = Path(jsonl_path)
        self.rows: List[Dict[str, Any]] = []
        if not self.path.exists():
            raise FileNotFoundError(f"GaussianJSONL not found: {jsonl_path}")
        for ln in self.path.read_text(encoding='utf-8', errors='ignore').splitlines():
            if not ln.strip():
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            self.rows.append(obj)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = dict(self.rows[idx])
        out: Dict[str, Any] = {}
        # Pass through optional text fields
        for k in ("text", "question", "answer"):
            if k in row:
                out[k] = row[k]
        # 3D splats
        g3 = row.get("gs3d") or {}
        if isinstance(g3, dict) and g3:
            pos = _to_tensor(g3.get("pos", []))  # (N,3)
            cov = g3.get("cov")
            cov_diag = g3.get("cov_diag")
            if cov is not None:
                cov_t = _to_tensor(cov)
                cov_diag_t = None
            elif cov_diag is not None:
                cov_t = None
                cov_diag_t = _to_tensor(cov_diag)
            else:
                cov_t = None
                cov_diag_t = _to_tensor([[1.0, 1.0, 1.0] for _ in range(max(1, pos.shape[0]))])
            out.update({
                "gs3d_pos_bnh3": pos.unsqueeze(0),
                "gs3d_cov_bnh33": None if cov_t is None else cov_t.unsqueeze(0),
                "gs3d_cov_diag_bnh3": None if cov_diag_t is None else cov_diag_t.unsqueeze(0),
                "gs3d_rgb_bnh3": _to_tensor(g3.get("rgb", [])).unsqueeze(0),
                "gs3d_opa_bnh1": _to_tensor(g3.get("opa", [])).unsqueeze(0),
                "gs3d_K_b33": _to_tensor(g3.get("K", [[224.0,0.0,112.0],[0.0,224.0,112.0],[0.0,0.0,1.0]])).unsqueeze(0),
                "gs3d_R_b33": _to_tensor(g3.get("R", [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])).unsqueeze(0),
                "gs3d_t_b3": _to_tensor(g3.get("t", [0.0,0.0,0.0])).unsqueeze(0),
                "gs3d_hw": (int(g3.get("H", 224)), int(g3.get("W", 224))),
            })
        # 2D splats
        g2 = row.get("gs2d") or {}
        if isinstance(g2, dict) and g2:
            out.update({
                "gs2d_mean_bng2": _to_tensor(g2.get("mean", [])).unsqueeze(0),
                "gs2d_cov_diag_bng2": _to_tensor(g2.get("cov_diag", [])).unsqueeze(0),
                "gs2d_rgb_bng3": _to_tensor(g2.get("rgb", [])).unsqueeze(0),
                "gs2d_opa_bng1": _to_tensor(g2.get("opa", [])).unsqueeze(0),
                "gs2d_hw": (int(g2.get("H", 224)), int(g2.get("W", 224))),
            })
        return out


