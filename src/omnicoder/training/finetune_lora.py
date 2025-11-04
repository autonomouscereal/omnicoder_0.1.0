import argparse
from pathlib import Path
from typing import Iterable
import time
import json
import os
import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn
from torch.utils.data import DataLoader

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.training.data.datamodule import DataModule


class LoRALinear(nn.Module):
    """
    Minimal LoRA wrapper for nn.Linear supporting weight-only adapters.
    W_out = W + scale * A @ B, with rank r. Bias is passed through unchanged.
    """

    def __init__(self, linear: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.0):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bias = linear.bias
        self.weight = linear.weight
        self.r = r
        self.scale = alpha / max(1, r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if r > 0:
            self.A = nn.Parameter(torch.zeros(self.in_features, r))
            self.B = nn.Parameter(torch.zeros(r, self.out_features))
            nn.init.kaiming_uniform_(self.A, a=5**0.5)
            nn.init.zeros_(self.B)
        else:
            self.register_parameter('A', None)
            self.register_parameter('B', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.r and self.A is not None and self.B is not None:
            x_d = self.dropout(x)
            update = x_d @ self.A @ self.B
            return base + self.scale * update
        return base


def _modules_of_type(model: nn.Module, types: tuple[type, ...]) -> Iterable[tuple[str, nn.Module, nn.Module]]:
    for name, module in model.named_modules():
        if isinstance(module, types):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            parent = model.get_submodule(parent_name) if parent_name else model
            yield name.split('.')[-1], module, parent


def inject_lora(model: nn.Module, r: int, alpha: int, dropout: float) -> int:
    """Replace nn.Linear with LoRALinear in-place; returns number of layers replaced."""
    replaced = 0
    for leaf_name, leaf, parent in _modules_of_type(model, (nn.Linear,)):
        lora = LoRALinear(leaf, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, leaf_name, lora)
        replaced += 1
    return replaced


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='.')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--seq_len', type=int, default=256)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--lora_r', type=int, default=16)
    ap.add_argument('--lora_alpha', type=int, default=32)
    ap.add_argument('--lora_dropout', type=float, default=0.05)
    ap.add_argument('--out', type=str, default='weights/omnicoder_lora.pt')
    ap.add_argument('--log_interval', type=int, default=20)
    ap.add_argument('--log_file', type=str, default='weights/finetune_lora_log.jsonl')
    ap.add_argument('--save_interval', type=int, default=0)
    args = ap.parse_args()

    dm = DataModule(train_folder=args.data, seq_len=args.seq_len, batch_size=args.batch_size)
    dl: DataLoader = dm.train_loader()

    model = OmniTransformer()
    num = inject_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    print(f"Injected LoRA into {num} Linear layers")
    # Resume LoRA-only best if exists
    try:
        from omnicoder.utils.checkpoint import load_best_or_latest  # type: ignore
        _loaded = load_best_or_latest(model, args.out)
        if _loaded is not None:
            print(f"[resume] loaded {_loaded}")
    except Exception:
        pass

    # Freeze base weights; train only LoRA params
    for p in model.parameters():
        p.requires_grad_(False)
    lora_params = [p for n, p in model.named_parameters() if ('A' in n or 'B' in n)]
    for p in lora_params:
        p.requires_grad_(True)

    model.to(args.device)
    opt = torch.optim.AdamW(lora_params, lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    step = 0
    tokens_seen = 0
    ema_step_time = None
    is_cuda = (str(args.device).startswith('cuda') and torch.cuda.is_available())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    for input_ids, labels in dl:
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        t0 = time.perf_counter()
        outputs = model(input_ids)
        if isinstance(outputs, tuple):
            logits = outputs[0]  # ignore mtp heads here
        else:
            logits = outputs
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        dt = max(1e-9, time.perf_counter() - t0)
        ema_step_time = dt if ema_step_time is None else (0.9 * ema_step_time + 0.1 * dt)
        toks = int(input_ids.numel())
        tokens_seen += toks
        tok_s = toks / dt
        tok_s_ema = toks / max(ema_step_time, 1e-9)
        mem_alloc = mem_rsvd = None
        if is_cuda:
            try:
                mem_alloc = int(torch.cuda.memory_allocated() // (1024*1024))
                mem_rsvd = int(torch.cuda.memory_reserved() // (1024*1024))
            except Exception:
                pass
        if step % max(1, args.log_interval) == 0 or step == 1:
            steps_left = max(0, args.steps - step)
            eta_s = steps_left * (ema_step_time or dt)
            eta_min = int(eta_s // 60)
            eta_sec = int(eta_s % 60)
            msg = f"step {step}/{args.steps} loss {loss.item():.4f} | tok/s {tok_s_ema:,.0f} | eta {eta_min:02d}:{eta_sec:02d}"
            if mem_alloc is not None:
                msg += f" | cuda_mem(MiB) alloc={mem_alloc} rsvd={mem_rsvd}"
            print(msg, flush=True)
            rec = {
                'step': step,
                'max_steps': args.steps,
                'loss': float(loss.item()),
                'tokens_seen': int(tokens_seen),
                'tokens_per_s': float(tok_s),
                'tokens_per_s_ema': float(tok_s_ema),
                'eta_seconds': float(eta_s),
                'cuda_mem_alloc_mb': mem_alloc,
                'cuda_mem_reserved_mb': mem_rsvd,
            }
            try:
                with open(args.log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(rec) + '\n')
            except Exception:
                pass
        if args.save_interval and args.save_interval > 0 and (step % args.save_interval == 0):
            ckpt_path = Path(args.out).with_name(Path(args.out).stem + f"_step{step}" + Path(args.out).suffix)
            try:
                # LoRA-only save
                _safe_save({k: v.cpu() for k, v in model.state_dict().items() if ('A' in k or 'B' in k)}, ckpt_path)
                print(f"[ckpt] saved {ckpt_path}")
            except Exception:
                pass
            # Best LoRA policy: track best (lowest loss)
            try:
                from omnicoder.utils.checkpoint import maybe_save_best  # type: ignore
                maybe_save_best(args.out, model, 'lora_loss', float(loss.item()), higher_is_better=False)
            except Exception:
                pass
        if step >= args.steps:
            break

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Save only LoRA parameters for portability
    state = {k: v.cpu() for k, v in model.state_dict().items() if ('A' in k or 'B' in k)}
    # Also write *_last and update *_best if this final loss is better
    _safe_save(state, args.out)
    # Write manifest next to final LoRA state
    try:
        from omnicoder.utils.model_manifest import build_manifest, save_manifest_for_checkpoint  # type: ignore
        man = build_manifest(model=None, modality='text', preset=os.getenv('OMNICODER_PRESET','mobile_4gb'), extra={'lora_only': True})
        save_manifest_for_checkpoint(args.out, man)
    except Exception:
        pass
    # Update best from final loss as well
    try:
        from omnicoder.utils.checkpoint import maybe_save_best  # type: ignore
        if 'loss' in locals():
            maybe_save_best(args.out, model, 'lora_loss', float(loss.item()), higher_is_better=False)
    except Exception:
        pass
    print(f"Saved LoRA state to {args.out}")


if __name__ == "__main__":
    main()
