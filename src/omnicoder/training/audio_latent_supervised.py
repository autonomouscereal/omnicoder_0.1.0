from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import soundfile as sf  # type: ignore
import warnings

# Suppress librosa's pkg_resources deprecation warning (setuptools>=81)
warnings.filterwarnings("ignore", category=UserWarning, module="librosa.core.intervals")


class AudioPairDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, ref_dir: str, pred_dir: str | None = None, sample_rate: int = 16000) -> None:
        self.ref_dir = Path(ref_dir)
        self.pred_dir = Path(pred_dir) if pred_dir else None
        self.files = [f for f in self.ref_dir.rglob("*.wav")]
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ref_path = self.files[idx]
        ref, sr = sf.read(ref_path)
        if sr != self.sample_rate:
            import librosa  # type: ignore
            ref = librosa.resample(ref, orig_sr=sr, target_sr=self.sample_rate)
        ref_t = torch.tensor(ref, dtype=torch.float32).unsqueeze(0)  # (1,T)
        # For simplicity, target equals ref; model is trained to reconstruct latent; extend with pred_dir later
        return ref_t, ref_t


class TinyAudioLatent(nn.Module):
    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, embed_dim, 5, stride=2, padding=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 1, 4, stride=2, padding=1), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        y = self.decoder(z)
        return z, y


def main() -> None:
    ap = argparse.ArgumentParser(description="Audio latent supervised training (reconstruction proxy)")
    ap.add_argument("--ref_dir", type=str, required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="weights/audio_latent.pt")
    args = ap.parse_args()

    ds = AudioPairDataset(args.ref_dir)
    if len(ds) == 0:
        raise SystemExit("audio_latent_supervised: dataset empty")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)

    model = TinyAudioLatent().to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    step = 0
    last_l1 = None
    for x, y in dl:
        x = x.to(args.device)
        y = y.to(args.device)
        z, yhat = model(x)
        loss = loss_fn(yhat, y)
        last_l1 = float(loss.item())
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
    # Best-checkpoint via last training L1 (lower is better)
    try:
        if last_l1 is not None:
            from omnicoder.utils.checkpoint import maybe_save_best  # type: ignore
            maybe_save_best(args.out, model, 'audio_l1', float(last_l1), higher_is_better=False)
    except Exception:
        pass
    print(f"Saved audio latent checkpoint to {final}")


if __name__ == "__main__":
    main()


