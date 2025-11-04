from __future__ import annotations

"""
Train a tiny linear autoencoder on saved KV tensors to learn a compact latent
representation. Produces a .pt file with encoder/decoder weights and writes a
JSON sidecar describing the latent size and layer mapping.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn
import torch.optim as optim


class KVAutoEncoder(nn.Module):
    def __init__(self, dim: int, latent: int) -> None:
        super().__init__()
        self.enc = nn.Linear(dim, latent, bias=False)
        self.dec = nn.Linear(latent, dim, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.enc(x)
        y = self.dec(z)
        return y, z


def _load_kv_tensors(root: str, max_files: int = 64) -> List[torch.Tensor]:
    feats: List[torch.Tensor] = []
    for p in sorted(Path(root).glob('kv_*.pt'))[:max_files]:
        try:
            dump = torch.load(p, map_location='cpu')
            # Each entry: {'k': tensor(B,H,T,DL), 'v': tensor(B,H,T,DL), 'meta': KvQuantMeta}
            for entry in dump:
                k = entry.get('k', None)
                v = entry.get('v', None)
                if isinstance(k, torch.Tensor):
                    feats.append(k.flatten(start_dim=0, end_dim=2))  # (B*H*T, DL)
                if isinstance(v, torch.Tensor):
                    feats.append(v.flatten(start_dim=0, end_dim=2))
        except Exception:
            continue
    return feats


def main() -> None:
    ap = argparse.ArgumentParser(description='Train a tiny KV autoencoder on dumped KV tensors')
    ap.add_argument('--kv_dir', type=str, default='weights/kv_dump')
    ap.add_argument('--latent', type=int, default=64)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--out_pt', type=str, default='weights/kv_autoencoder.pt')
    ap.add_argument('--out_json', type=str, default='weights/kv_compress_sidecar.json')
    args = ap.parse_args()

    feats = _load_kv_tensors(args.kv_dir)
    if not feats:
        print('[kv_ae] no KV dumps found; exiting successfully')
        return
    # Infer dim from first tensor
    dim = int(feats[0].size(-1))
    model = KVAutoEncoder(dim=dim, latent=int(args.latent)).to(args.device)
    opt = optim.AdamW(model.parameters(), lr=float(args.lr))
    loss_fn = nn.MSELoss()

    # Simple streaming training
    step = 0
    idx = 0
    batch = 8192
    while step < int(args.steps):
        x = feats[idx % len(feats)]
        idx += 1
        if x.numel() == 0:
            continue
        # Sample a batch of rows
        sel = torch.randint(0, x.size(0), (min(batch, x.size(0)),))
        xb = x[sel].to(args.device)
        y, _ = model(xb)
        loss = loss_fn(y, xb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            print({'step': step, 'mse': float(loss.item())})

    Path(args.out_pt).parent.mkdir(parents=True, exist_ok=True)
    _safe_save({'dim': dim, 'latent': int(args.latent), 'state_dict': model.state_dict()}, args.out_pt)
    sidecar = {
        'kv_autoencoder': {
            'dim': dim,
            'latent': int(args.latent),
            'weights': str(Path(args.out_pt).resolve()),
        }
    }
    Path(args.out_json).write_text(json.dumps(sidecar, indent=2), encoding='utf-8')
    print({'out_pt': args.out_pt, 'out_json': args.out_json})


if __name__ == '__main__':
    main()


