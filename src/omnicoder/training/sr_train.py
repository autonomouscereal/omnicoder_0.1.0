from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image  # type: ignore


class SRDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, pred_dir: str, ref_dir: str, scale: int = 2, image_size: Tuple[int, int] = (256, 256)) -> None:
        self.pred_dir = Path(pred_dir)
        self.ref_dir = Path(ref_dir)
        self.files = [f for f in os.listdir(ref_dir) if f.lower().endswith((".png",".jpg",".jpeg"))]
        self.scale = scale
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self.files[idx]
        hr = Image.open(self.ref_dir / name).convert('RGB').resize(self.image_size)
        import numpy as np
        hr_t = torch.from_numpy(np.array(hr).astype('float32') / 255.0).permute(2, 0, 1)
        # Create LR by downscale+upsample to simulate bicubic baseline
        lr = hr.resize((self.image_size[0]//self.scale, self.image_size[1]//self.scale), Image.BICUBIC)
        lr_up = lr.resize(self.image_size, Image.BICUBIC)
        lr_t = torch.from_numpy(np.array(lr_up).astype('float32') / 255.0).permute(2, 0, 1)
        return lr_t, hr_t


class TinySRNet(nn.Module):
    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.net(x), 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Simple SR trainer (baseline model) for SR metrics benchmarking")
    ap.add_argument("--pred_dir", type=str, required=True)
    ap.add_argument("--ref_dir", type=str, required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="weights/sr_baseline.pt")
    args = ap.parse_args()

    ds = SRDataset(args.pred_dir, args.ref_dir)
    if len(ds) == 0:
        raise SystemExit("sr_train: dataset empty")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)

    model = TinySRNet().to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    step = 0
    last_loss = None  # track last loss for best-checkpoint metric
    for lr_img, hr_img in dl:
        lr_img = lr_img.to(args.device)
        hr_img = hr_img.to(args.device)
        pred = model(lr_img)
        loss = loss_fn(pred, hr_img)
        last_loss = float(loss.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % 50 == 0 or step == 1:
            print(f"step {step}/{args.steps} loss={loss.item():.4f}")
        if step >= args.steps:
            break
    try:
        from omnicoder.utils.checkpoint import save_with_sidecar  # type: ignore
    except Exception:
        save_with_sidecar = None  # type: ignore
    meta = {'train_args': {'steps': int(args.steps)}}
    if callable(save_with_sidecar):
        final = save_with_sidecar(args.out, model.state_dict(), meta=meta)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        _safe_save({k: (v.detach().to('cpu') if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}, args.out)
        final = args.out
    # Best-checkpoint via last training loss (lower is better)
    try:
        if last_loss is not None:
            from omnicoder.utils.checkpoint import maybe_save_best  # type: ignore
            maybe_save_best(args.out, model, 'sr_l1', float(last_loss), higher_is_better=False)
    except Exception:
        pass
    print(f"Saved SR baseline checkpoint to {final}")


if __name__ == "__main__":
    main()


