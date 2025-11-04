import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from omnicoder.utils.resources import recommend_num_workers
from PIL import Image
import os

from omnicoder.modeling.multimodal.vqvae import ImageVQVAE


class ImageFolderDataset(Dataset):
    def __init__(self, root: str, size: int = 224):
        self.root = root
        self.size = int(size)
        self.files = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.files.append(os.path.join(dirpath, fn))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        img = Image.open(path).convert('RGB').resize((self.size, self.size))
        x = torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).reshape(self.size, self.size, 3).numpy().astype('float32') / 255.0)).permute(2, 0, 1)
        return x


def main() -> None:
    ap = argparse.ArgumentParser(description='Train a tiny Image VQ-VAE codebook')
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--image_size', type=int, default=224)
    ap.add_argument('--codebook_size', type=int, default=8192)
    ap.add_argument('--code_dim', type=int, default=192)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', type=str, default='weights/image_vq_codebook.pt')
    args = ap.parse_args()

    ds = ImageFolderDataset(args.data, size=args.image_size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=recommend_num_workers(), drop_last=True)
    model = ImageVQVAE(codebook_size=args.codebook_size, code_dim=args.code_dim).to(args.device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    use_amp = args.device.startswith('cuda') and torch.cuda.is_available()
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)  # type: ignore[attr-defined]
        _ac = torch.amp.autocast('cuda') if use_amp else torch.amp.autocast('cpu')  # type: ignore[attr-defined]
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        _ac = torch.cuda.amp.autocast(enabled=use_amp)

    it = iter(dl)
    for step in range(1, args.steps + 1):
        try:
            x = next(it)
        except StopIteration:
            it = iter(dl)
            x = next(it)
        x = x.to(args.device)
        with _ac:
            # ImageVQVAE returns (x_rec, codes) or (rec, com, ppx, x_rec, idx) depending on implementation
            out = model(x)
            if isinstance(out, tuple) and len(out) == 2:
                xr, _ = out
                rec = F.l1_loss(xr, x)
            elif isinstance(out, tuple) and len(out) >= 4:
                # rec_loss, commit_loss, perplexity, x_rec, idx
                rec = out[0]
            else:
                xr = out  # type: ignore[assignment]
                rec = F.l1_loss(xr, x)
            # commitment/embedding losses are implicit via EMA; rec only
            loss = rec
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step % 100 == 0:
            print(f"step={step} rec={float(rec):.4f}")
        if step % 1000 == 0:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            model.quant.embed.detach().cpu()
            model.save_codebook(args.out)
            print(f"[save] codebook -> {args.out}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save_codebook(args.out)
    print(f"[done] codebook -> {args.out}")


if __name__ == '__main__':
    main()

# Note: This file accidentally contained two modules. The second duplicate block has been removed.


