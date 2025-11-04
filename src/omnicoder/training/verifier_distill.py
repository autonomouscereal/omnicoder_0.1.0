from __future__ import annotations

"""
Distill an external verifier head for draft-and-verify decoding.

Trains the student's lightweight `verifier_head` to match a stronger teacher's
next-token distribution. Optionally also trains the main `lm_head`.

Usage (GPU recommended):
  python -m omnicoder.training.verifier_distill \
    --data ./ --seq_len 512 --steps 200 --device cuda \
    --teacher microsoft/phi-2 --student_mobile_preset mobile_4gb \
    --verifier_only --kl_temp 1.5 --lr 2e-4 --out weights/omnicoder_verifier_kd.pt
"""

import argparse
import time
import os
from pathlib import Path
from typing import Optional

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn
from torch.utils.data import DataLoader

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.config import MobilePreset, MobilePreset2GB
from omnicoder.training.data.datamodule import DataModule


def _enable_gradient_checkpointing(model: nn.Module) -> None:
    for m in model.modules():
        if hasattr(m, 'gradient_checkpointing'):
            try:
                m.gradient_checkpointing = True  # type: ignore[attr-defined]
            except Exception:
                pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Verifier-head distillation (teacher -> student.verifier_head)")
    ap.add_argument("--data", type=str, default=".")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--student_mobile_preset", type=str, default="mobile_4gb")
    ap.add_argument("--teacher", type=str, default="microsoft/phi-2")
    ap.add_argument("--teacher_dtype", type=str, default="auto", help="auto|fp16|bf16|fp32")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--kl_temp", type=float, default=1.5)
    ap.add_argument("--verifier_only", action="store_true", help="Update only the verifier head; freeze other params")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--log_interval", type=int, default=20)
    ap.add_argument("--out", type=str, default="weights/omnicoder_verifier_kd.pt")
    ap.add_argument("--load_student", type=str, default="", help="Optional student checkpoint to resume")
    args = ap.parse_args()

    # Ensure HF cache persists across runs (defaults to /models/hf if unset)
    os.environ.setdefault("HF_HOME", "/models/hf")
    # Rely on HF_HOME only; TRANSFORMERS_CACHE is deprecated

    # Data
    dm = DataModule(train_folder=args.data, seq_len=args.seq_len, batch_size=args.batch_size)
    dl: DataLoader = dm.train_loader()

    # Student
    preset = MobilePreset() if args.student_mobile_preset == "mobile_4gb" else MobilePreset2GB()
    student = OmniTransformer(
        vocab_size=preset.vocab_size,
        n_layers=preset.n_layers,
        d_model=preset.d_model,
        n_heads=preset.n_heads,
        mlp_dim=preset.mlp_dim,
        n_experts=preset.moe_experts,
        top_k=preset.moe_top_k,
        max_seq_len=args.seq_len,
        use_rope=True,
        kv_latent_dim=preset.kv_latent_dim,
        multi_query=preset.multi_query,
        multi_token=2,
    )
    # Resume from best-known if present
    try:
        from omnicoder.utils.checkpoint import load_best_or_latest  # type: ignore
        _loaded = load_best_or_latest(student, args.out)
        if _loaded is not None:
            print(f"[resume] loaded {_loaded}")
    except Exception:
        pass
    # Optional: resume student
    if args.load_student:
        try:
            sd = torch.load(args.load_student, map_location="cpu")
            if isinstance(sd, dict):
                student.load_state_dict(sd, strict=False)
                print(f"[resume] loaded student weights from {args.load_student}")
        except Exception as e:
            print(f"[resume] skip: {e}")
    if args.verifier_only:
        for n, p in student.named_parameters():
            if 'verifier_head' not in n:
                p.requires_grad_(False)
    if args.gradient_checkpointing:
        _enable_gradient_checkpointing(student)
    # Reduce optimizer state memory pressure on CUDA by preferring bf16 when available
    if args.device.startswith('cuda') and torch.cuda.is_available():
        try:
            student = student.to(dtype=torch.bfloat16, device=args.device)
        except Exception:
            student = student.to(args.device)
    else:
        student = student.to(args.device)
    student.train()

    # Teacher (HF causal LM)
    try:
        from transformers import AutoModelForCausalLM  # type: ignore
    except Exception as e:
        raise RuntimeError("transformers is required. pip install transformers") from e
    dtype_map = {
        "auto": torch.float16 if torch.cuda.is_available() else None,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    teacher_dtype = dtype_map.get(args.teacher_dtype, None)
    # Try to load teacher offline; if unavailable, fall back to a frozen student copy to enable smoke runs
    try:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher,
            dtype=teacher_dtype,
            cache_dir=os.getenv("HF_HOME", None),
            local_files_only=True,
        )
        teacher.eval().to(args.device)
    except Exception:
        # Offline fallback
        print("[warn] teacher not found offline; using a frozen student copy as dummy teacher for this run")
        teacher = OmniTransformer(
            vocab_size=student.vocab_size,
            n_layers=len(student.blocks),
            d_model=student.ln_f.normalized_shape[0],
            n_heads=student.blocks[0].attn.n_heads if hasattr(student.blocks[0], 'attn') else 8,
            mlp_dim=student.lm_head.in_features * 4,
            n_experts=preset.moe_experts,
            top_k=preset.moe_top_k,
            max_seq_len=args.seq_len,
            use_rope=True,
            kv_latent_dim=preset.kv_latent_dim,
            multi_query=preset.multi_query,
            multi_token=1,
        )
        try:
            teacher.load_state_dict(student.state_dict(), strict=False)
        except Exception:
            pass
        teacher.eval().to(args.device)

    # Optim
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr)

    kl_loss = nn.KLDivLoss(reduction='batchmean')
    steps = 0
    t0 = time.time()
    # Optional student->teacher vocab aligner (learned linear map). Created lazily on first mismatch.
    proj_align: nn.Linear | None = None
    for batch in dl:
        if isinstance(batch, (tuple, list)) and len(batch) >= 1:
            ids = batch[0].to(args.device)
        elif isinstance(batch, dict) and 'input_ids' in batch:
            ids = batch['input_ids'].to(args.device)
        else:
            raise TypeError("Unsupported batch format; expected (ids, labels) or dict with 'input_ids'")
        with torch.no_grad():
            t_out = teacher(ids)
            # Normalize teacher outputs to logits tensor
            if hasattr(t_out, "logits"):
                t_logits = t_out.logits  # type: ignore[attr-defined]
            elif isinstance(t_out, tuple):
                t_logits = t_out[0]  # type: ignore[index]
            else:
                t_logits = t_out  # type: ignore[assignment]
        s_out = student(ids)
        if isinstance(s_out, tuple):
            s_logits = s_out[0]
        else:
            s_logits = s_out
        # Align vocab dims via a learnable projection from student->teacher space when mismatch.
        # This preserves teacher information and avoids slicing/padding hacks.
        if t_logits.size(-1) != s_logits.size(-1):
            Vt = int(t_logits.size(-1))
            Vs = int(s_logits.size(-1))
            if (proj_align is None) or (proj_align.in_features != Vs) or (proj_align.out_features != Vt):
                proj_align = nn.Linear(Vs, Vt, bias=False).to(args.device)
                # Train aligner jointly with student
                try:
                    opt.add_param_group({'params': proj_align.parameters()})
                except Exception:
                    pass
            # Map student logits to teacher vocab
            s_logits = proj_align(s_logits)
        # Soften teacher distribution
        t_log_probs = torch.log_softmax(t_logits / max(1e-6, args.kl_temp), dim=-1)

        v_logits = student.verifier_head(student.ln_f(student.embed(ids))) if args.verifier_only else student.verifier_head(student.ln_f(student.embed(ids)))
        # Align verifier logits as well if projection exists to ensure KL terms share the same target space
        if (proj_align is not None) and (v_logits.size(-1) != t_logits.size(-1)):
            v_logits = proj_align(v_logits)
        # Align shapes (B,T,V)
        s_log_probs = torch.log_softmax(s_logits / max(1e-6, args.kl_temp), dim=-1)
        v_log_probs = torch.log_softmax(v_logits / max(1e-6, args.kl_temp), dim=-1)
        # Loss: KL(student||teacher) + KL(verifier||teacher) (optionally scale)
        loss = kl_loss(s_log_probs, t_log_probs.detach()) * (0.0 if args.verifier_only else 1.0)
        loss = loss + kl_loss(v_log_probs, t_log_probs.detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        steps += 1
        if steps % args.log_interval == 0:
            dt = time.time() - t0
            print(f"step {steps}/{args.steps} loss={float(loss.item()):.4f} dt={dt:.1f}s")
        if steps >= args.steps:
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from omnicoder.utils.checkpoint import save_with_sidecar, maybe_save_best  # type: ignore
    except Exception:
        save_with_sidecar = None  # type: ignore
        maybe_save_best = None  # type: ignore
    if callable(save_with_sidecar):
        final = save_with_sidecar(out_path, student.state_dict(), meta={'train_args': {'steps': int(args.steps)}})
    else:
        _safe_save(student.state_dict(), out_path)
        final = out_path
    # Best via TPS and small CE proxy
    try:
        if callable(maybe_save_best):
            from omnicoder.utils.evalhooks import text_tps  # type: ignore
            tps = text_tps(student, device=args.device, seq_len=128, gen_tokens=64)
            if tps is not None:
                maybe_save_best(args.out, student, 'tps_eval', float(tps), higher_is_better=True)
            # CE proxy on one batch
            try:
                batch = next(iter(dl))
                ids = batch[0].to(args.device) if isinstance(batch, (tuple, list)) else batch['input_ids'].to(args.device)
                with torch.inference_mode():
                    out = student(ids)
                logits = out[0] if isinstance(out, tuple) else out
                bsz, t, v = logits.shape
                ce = nn.CrossEntropyLoss()(logits[:, :-1, :].reshape(-1, v), ids[:, 1:].reshape(-1)).item()
                maybe_save_best(args.out, student, 'verifier_ce', float(ce), higher_is_better=False)
            except Exception:
                pass
    except Exception:
        pass
    print(f"Saved verifier-distilled checkpoint to {final}")


if __name__ == "__main__":
    main()


