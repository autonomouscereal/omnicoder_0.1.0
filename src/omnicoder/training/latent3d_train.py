from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn

from omnicoder.modeling.multimodal.latent3d import VoxelLatentHead, SimpleOrthoRenderer
from omnicoder.modeling.transformer_moe import OmniTransformer


def main() -> None:
    ap = argparse.ArgumentParser(description="Tiny 3D latent head training (toy fit)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="weights/latent3d_head.pt")
    args = ap.parse_args()

    device = torch.device(args.device)
    # Tiny text model to produce hidden states
    model = OmniTransformer(vocab_size=32000, n_layers=2, d_model=256, n_heads=4, mlp_dim=768, n_experts=2, top_k=1, kv_latent_dim=64, multi_token=1).to(device)
    head = VoxelLatentHead(d_model=256, depth=8, height=16, width=16, hidden=256).to(device)
    rend = SimpleOrthoRenderer(depth=8, out_h=64, out_w=64).to(device)

    mse = nn.MSELoss()
    opt = torch.optim.AdamW(list(head.parameters()) + list(model.parameters()), lr=1e-3)

    # Synthetic targets: simple geometric blobs
    def _make_target(b: int) -> torch.Tensor:
        x = torch.zeros((b, 3, 64, 64), device=device)
        for i in range(b):
            c = (i % 3)
            x[i, c, 16:48, 16:48] = 1.0
        return x

    tok = torch.randint(low=0, high=32000, size=(4, 32), device=device)
    step = 0
    while step < int(args.steps):
        step += 1
        logits = model(tok)
        if isinstance(logits, tuple):
            logits = logits[0]
        hidden = model.ln_f(model.blocks[-1].ln2.weight.new_zeros((tok.size(0), 1, model.ln_f.normalized_shape[0])))  # placeholder hidden proxy
        vox = head(hidden)
        img = rend(vox)
        tgt = _make_target(img.size(0))
        loss = mse(img, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(head.parameters()) + list(model.parameters()), 1.0)
        opt.step()
        if step % 20 == 0:
            print(f"step {step}/{args.steps} loss={loss.item():.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    try:
        from omnicoder.utils.checkpoint import save_with_sidecar, maybe_save_best  # type: ignore
    except Exception:
        save_with_sidecar = None  # type: ignore
        maybe_save_best = None  # type: ignore
    payload = {"voxel_head": head.state_dict()}
    if callable(save_with_sidecar):
        final = save_with_sidecar(args.out, payload, meta={'train_args': {'steps': int(args.steps)}})
    else:
        _safe_save(payload, args.out)
        final = args.out
    try:
        if callable(maybe_save_best) and 'loss' in locals():
            maybe_save_best(args.out, head, 'latent3d_mse', float(loss.item()), higher_is_better=False)
    except Exception:
        pass
    print(f"Saved 3D head to {final}")


if __name__ == "__main__":
    main()


