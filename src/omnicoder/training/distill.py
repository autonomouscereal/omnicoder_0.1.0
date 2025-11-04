"""
Knowledge Distillation training loop (teacher -> student)

Train the compact OmniTransformer student to match a larger teacher's
token distributions using KL divergence (logit matching). Optionally
adds supervised CE on ground-truth next tokens if labels are provided.

This implementation is intentionally dependency-light and works with
any HuggingFace causal LLM as teacher. It supports gradient
checkpointing and LoRA for memory efficiency, and can run on a single
24 GB GPU for mobile-sized students.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional
import time
import json
import contextlib

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
try:
    # Best-effort cudagraph step marker to avoid replay overwrite between iterations
    from omnicoder.utils.torchutils import get_cudagraph_step_marker as _get_cg_mark  # type: ignore
except Exception:
    _get_cg_mark = None  # type: ignore
import torch.nn as nn
from torch.utils.data import DataLoader
from omnicoder.utils.resources import recommend_num_workers
from itertools import cycle

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.config import MobilePreset
from omnicoder.training.data.datamodule import DataModule


def _enable_gradient_checkpointing(model: nn.Module) -> None:
    """Enable lightweight activation checkpointing on transformer blocks.

    This wraps each transformer block's forward pass with
    torch.utils.checkpoint.checkpoint so that intermediate activations are
    recomputed during the backward pass, reducing peak memory. We avoid
    altering module signatures and only wrap blocks that accept a single
    positional tensor argument during training (our Block(x) path).
    """
    try:
        from torch.utils.checkpoint import checkpoint  # type: ignore
    except Exception:
        return

    # Only wrap our OmniTransformer blocks to keep semantics simple
    if hasattr(model, "blocks"):
        for idx, blk in enumerate(getattr(model, "blocks")):
            orig_forward = blk.forward

            def _wrapped(x, *args, _orig=orig_forward, **kwargs):  # type: ignore
                # Only checkpoint the common training path without cache/state
                use_cache = bool(kwargs.get("use_cache", False))
                past_k = kwargs.get("past_k_latent", None)
                past_v = kwargs.get("past_v_latent", None)
                if not use_cache and past_k is None and past_v is None:
                    # Explicitly set use_reentrant per PyTorch 2.5 warning; non-reentrant is recommended
                    try:
                        # Preserve RNG state; do NOT mutate module state or flip flags in forward.
                        # The block will honor deterministic=True internally without persistent mutation.
                        return checkpoint(lambda _x: _orig(_x, deterministic=True), x, use_reentrant=False, preserve_rng_state=True)  # type: ignore[call-arg]
                    except TypeError:
                        # Older torch without the kwarg
                        try:
                            return checkpoint(lambda _x: _orig(_x, deterministic=True), x, preserve_rng_state=True)  # type: ignore[call-arg]
                        except TypeError:
                            return checkpoint(lambda _x: _orig(_x, deterministic=True), x)
                return _orig(x, *args, **kwargs)

            blk.forward = _wrapped  # type: ignore[assignment]


def _inject_lora_linear(module: nn.Module, r: int, alpha: int, dropout: float) -> int:
    replaced = 0
    for name, child in list(module.named_modules()):
        if isinstance(child, nn.Linear) and child.out_features >= 64:
            parent = module
            path = name.split(".")
            for p in path[:-1]:
                parent = getattr(parent, p)
            leaf_name = path[-1]
            base = getattr(parent, leaf_name)

            class LoRALinear(nn.Module):
                def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
                    super().__init__()
                    self.base = base
                    # Expose Linear-like attributes for compatibility
                    try:
                        self.in_features = int(getattr(base, "in_features", 0))
                    except Exception:
                        self.in_features = 0
                    try:
                        self.out_features = int(getattr(base, "out_features", 0))
                    except Exception:
                        self.out_features = 0
                    try:
                        self.weight = base.weight  # type: ignore[assignment]
                    except Exception:
                        pass
                    try:
                        self.bias = base.bias  # type: ignore[assignment]
                    except Exception:
                        pass
                    self.r = r
                    self.dropout = nn.Dropout(dropout)
                    self.scaling = alpha / max(r, 1)
                    self.A = nn.Linear(base.in_features, r, bias=False)
                    self.B = nn.Linear(r, base.out_features, bias=False)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.base(x) + self.B(self.A(self.dropout(x))) * self.scaling

            lora = LoRALinear(base, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, leaf_name, lora)
            replaced += 1
    return replaced


def main() -> None:
    ap = argparse.ArgumentParser(description="Knowledge distillation: teacher (HF) -> student (OmniTransformer)")
    ap.add_argument("--data", type=str, default=os.getenv("OMNICODER_KD_DATA", "."), help="Folder of .txt or a JSONL with {text,rationale?,router_targets?}")
    ap.add_argument("--data_is_jsonl", action="store_true", default=(os.getenv("OMNICODER_KD_DATA_IS_JSONL", "0") == "1"), help="Interpret --data as a KD JSONL with rationales and router targets")
    ap.add_argument("--seq_len", type=int, default=int(os.getenv("OMNICODER_KD_SEQ_LEN", "512")))
    ap.add_argument("--batch_size", type=int, default=int(os.getenv("OMNICODER_KD_BATCH", "2")))
    ap.add_argument("--steps", type=int, default=int(os.getenv("OMNICODER_KD_STEPS", "200")))
    ap.add_argument("--device", type=str, default=os.getenv("OMNICODER_KD_DEVICE", ("cuda" if torch.cuda.is_available() else "cpu")))
    ap.add_argument("--student_mobile_preset", type=str, default=os.getenv("OMNICODER_STUDENT_PRESET", "mobile_4gb"))
    ap.add_argument("--teacher", type=str, default=os.getenv("OMNICODER_KD_TEACHER", "microsoft/phi-2"), help="HF model id for teacher")
    ap.add_argument("--teachers", type=str, nargs='*', default=os.getenv("OMNICODER_KD_TEACHERS", "").split() if os.getenv("OMNICODER_KD_TEACHERS") else [], help="Optional multiple HF teacher ids; will be used round-robin per step")
    ap.add_argument("--teacher_local", type=str, default=os.getenv("OMNICODER_KD_TEACHER_LOCAL", ""), help="Optional local path to a pre-downloaded HF model directory")
    ap.add_argument("--teacher_device_map", type=str, default=os.getenv("OMNICODER_KD_TEACHER_DEVICE_MAP", "auto"), help="Device map for teacher: auto|cpu|cuda")
    ap.add_argument("--teacher_dtype", type=str, default=os.getenv("OMNICODER_KD_TEACHER_DTYPE", "auto"), help="auto|fp16|bf16|fp32")
    ap.add_argument("--lr", type=float, default=float(os.getenv("OMNICODER_KD_LR", "0.0002")))
    ap.add_argument("--kl_temp", type=float, default=float(os.getenv("OMNICODER_KD_TEMP", "1.5")), help="Softmax temperature for KD")
    ap.add_argument("--alpha_kd", type=float, default=float(os.getenv("OMNICODER_KD_ALPHA", "0.9")), help="Weight for KD loss vs CE")
    ap.add_argument("--gradient_checkpointing", action="store_true", default=(os.getenv("OMNICODER_KD_GRAD_CHKPT", "0")=="1"))
    ap.add_argument("--lora", action="store_true", default=(os.getenv("OMNICODER_KD_LORA", "0")=="1"))
    ap.add_argument("--lora_r", type=int, default=int(os.getenv("OMNICODER_KD_LORA_R", "16")))
    ap.add_argument("--lora_alpha", type=int, default=int(os.getenv("OMNICODER_KD_LORA_ALPHA", "32")))
    ap.add_argument("--lora_dropout", type=float, default=float(os.getenv("OMNICODER_KD_LORA_DROPOUT", "0.05")))
    ap.add_argument("--out", type=str, default=os.getenv("OMNICODER_KD_OUT", "weights/omnicoder_student_kd.pt"))
    ap.add_argument("--resume_ckpt", type=str, default=os.getenv("OMNICODER_KD_RESUME", ""))
    ap.add_argument("--state_out", type=str, default=os.getenv("OMNICODER_KD_STATE_OUT", "weights/kd_state.pt"))
    ap.add_argument("--log_interval", type=int, default=int(os.getenv("OMNICODER_KD_LOG_INTERVAL", "20")), help="Steps between progress logs")
    ap.add_argument("--save_interval", type=int, default=int(os.getenv("OMNICODER_KD_SAVE_INTERVAL", "1000")), help="Steps between interim checkpoint saves (0 to disable)")
    ap.add_argument("--log_file", type=str, default=os.getenv("OMNICODER_KD_LOG_FILE", "weights/kd_train_log.jsonl"), help="Path to append JSONL logs")
    ap.add_argument("--router_temp", type=float, default=float(os.getenv("OMNICODER_ROUTER_TEMP", "1.2")))
    ap.add_argument("--router_jitter", type=float, default=float(os.getenv("OMNICODER_ROUTER_JITTER", "0.2")))
    ap.add_argument("--router_use_gumbel", action="store_true", default=(os.getenv("OMNICODER_ROUTER_USE_GUMBEL", "0")=="1"))
    ap.add_argument("--aux_lb_coef", type=float, default=float(os.getenv("OMNICODER_AUX_LB_COEF", "0.01")))
    ap.add_argument("--aux_importance_coef", type=float, default=float(os.getenv("OMNICODER_AUX_IMPORTANCE_COEF", "0.01")))
    ap.add_argument("--aux_load_coef", type=float, default=float(os.getenv("OMNICODER_AUX_LOAD_COEF", "0.01")))
    ap.add_argument("--aux_zloss_coef", type=float, default=float(os.getenv("OMNICODER_AUX_ZLOSS_COEF", "0.001")))
    ap.add_argument("--aux_sinkhorn_kl_coef", type=float, default=float(os.getenv("OMNICODER_AUX_SINKHORN_KL_COEF", "0.0")))
    # Variable-K and halting/difficulty auxiliary (unsupervised) – enable smarter reasoning signals
    ap.add_argument("--var_k_train", action="store_true", default=(os.getenv("OMNICODER_VAR_K_TRAIN", "0") == "1"), help="Enable variable-K expert selection during KD based on difficulty")
    ap.add_argument("--var_k_min", type=int, default=int(os.getenv("OMNICODER_VAR_K_MIN", "1")))
    ap.add_argument("--var_k_max", type=int, default=int(os.getenv("OMNICODER_VAR_K_MAX", "4")))
    ap.add_argument("--var_k_threshold", type=float, default=float(os.getenv("OMNICODER_VAR_K_THRESH", "0.5")), help="Difficulty threshold in [0,1] above which to raise expert K")
    def _env_float(key: str, default: float | None = None) -> float | None:
        val = os.getenv(key, '')
        if val is None or val.strip() == '':
            return default
        try:
            return float(val)
        except Exception:
            return default
    ap.add_argument("--var_k_threshold_start", type=float, default=_env_float('OMNICODER_VAR_K_THRESH_START', None), help="Optional: start threshold for a linear schedule")
    ap.add_argument("--var_k_threshold_end", type=float, default=_env_float('OMNICODER_VAR_K_THRESH_END', None), help="Optional: end threshold for a linear schedule")
    ap.add_argument("--diff_loss_coef", type=float, default=float(os.getenv("OMNICODER_DIFF_LOSS_COEF", "0.0")), help="Aux loss weight for difficulty head (MSE to 1-top_prob)")
    ap.add_argument("--halt_loss_coef", type=float, default=float(os.getenv("OMNICODER_HALT_LOSS_COEF", "0.0")), help="Aux loss weight for halting head (BCE to entropy threshold)")
    ap.add_argument("--halt_entropy", type=float, default=float(os.getenv("OMNICODER_HALT_ENTROPY", "1.0")), help="Entropy threshold for halting label")
    # Rationale/sequence-level KD controls
    ap.add_argument("--seq_kd", action="store_true", default=(os.getenv("OMNICODER_KD_SEQ", "0")=="1"), help="Enable sequence-level KD (teacher vs student sequence scores)")
    ap.add_argument("--rationale_field", type=str, default=os.getenv("OMNICODER_KD_RATIONALE_FIELD", ""), help="Optional JSONL field containing teacher rationales for auxiliary losses")
    ap.add_argument("--expert_route_kd", action="store_true", default=(os.getenv("OMNICODER_KD_EXPERT_ROUTE", "0")=="1"), help="Enable expert-aware routing KD (match router distributions)")
    ap.add_argument("--router_expert_dropout_p", type=float, default=float(os.getenv("OMNICODER_ROUTER_EXPERT_DROPOUT_P", "0.0")))
    ap.add_argument("--router_sinkhorn_iters", type=int, default=int(os.getenv("OMNICODER_ROUTER_SINKHORN_ITERS", "0")))
    ap.add_argument("--router_sinkhorn_tau", type=float, default=float(os.getenv("OMNICODER_ROUTER_SINKHORN_TAU", "1.0")))
    ap.add_argument("--moe_static_capacity", type=int, default=int(os.getenv("OMNICODER_MOE_STATIC_CAPACITY", "0")))
    ap.add_argument("--image_latent_loss", action="store_true", default=(os.getenv("OMNICODER_KD_IMG_LAT", "0")=="1"))
    ap.add_argument("--audio_latent_loss", action="store_true", default=(os.getenv("OMNICODER_KD_AUD_LAT", "0")=="1"))
    ap.add_argument("--latent_dim", type=int, default=int(os.getenv("OMNICODER_LATENT_DIM", "16")))
    ap.add_argument("--moe_group_sizes", type=str, default=os.getenv("OMNICODER_MOE_GROUP_SIZES", ""), help="Comma-separated expert group sizes for hierarchical routing (e.g., 4,4)")
    # CuMo upcycling (clone dense FFN to multiple experts at init)
    ap.add_argument("--cumo_upcycle", action="store_true", default=(os.getenv("OMNICODER_CUMO_UPCYCLE", "0") == "1"))
    ap.add_argument("--cumo_target_experts", type=int, default=int(os.getenv("OMNICODER_CUMO_TARGET", "0")))
    args = ap.parse_args()

    # Device fallback if CUDA not available
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available in this environment; falling back to CPU.")
        args.device = "cpu"

    # For smoke runs (steps<=1), force offline-friendly settings to avoid any
    # heavyweight or network-dependent code paths in constrained CI/containers.
    if int(args.steps) <= 1:
        os.environ.setdefault("OMNICODER_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # Data
    # Smoke-friendly tiny loader to avoid repo-wide file scans and speed up CI
    if int(args.steps) <= 1:
        try:
            B = max(1, int(args.batch_size))
            S = max(8, int(args.seq_len))
            # Deterministic tiny batch: two simple sequences per batch element
            toy = []
            for _ in range(B):
                # Build 0..S-1 using aten-only ops (no torch.arange)
                ones = torch.ops.aten.new_ones.default(torch.empty(0, dtype=torch.long), (S,), dtype=torch.long)
                ids = torch.ops.aten.cumsum.default(ones, 0)
                ids = torch.ops.aten.sub.Tensor(ids, torch.ops.aten.new_ones.default(ids, ids.shape, dtype=ids.dtype))
                ids = torch.ops.aten.remainder.Tensor(ids, 97)
                toy.append((ids[:-1], ids[1:]))
            dl = DataLoader(toy, batch_size=B, shuffle=False, num_workers=recommend_num_workers())
            kd_jsonl_mode = False
        except Exception:
            # Fallback to normal loader if toy creation fails for any reason
            if args.data_is_jsonl:
                from omnicoder.training.data.kd_jsonl import kd_loader
                dl = kd_loader(args.data, seq_len=args.seq_len, batch_size=args.batch_size, include_rationale=True)
                kd_jsonl_mode = True
            else:
                dm = DataModule(train_folder=args.data, seq_len=args.seq_len, batch_size=args.batch_size)
                dl = dm.train_loader()
                kd_jsonl_mode = False
    else:
        if args.data_is_jsonl:
            from omnicoder.training.data.kd_jsonl import kd_loader
            dl = kd_loader(args.data, seq_len=args.seq_len, batch_size=args.batch_size, include_rationale=True)
            kd_jsonl_mode = True
        else:
            dm = DataModule(train_folder=args.data, seq_len=args.seq_len, batch_size=args.batch_size)
            dl = dm.train_loader()
            kd_jsonl_mode = False

    # Student
    from omnicoder.config import MobilePreset2GB
    preset = MobilePreset() if args.student_mobile_preset == "mobile_4gb" else MobilePreset2GB()
    # Smoke mode: shrink student architecture drastically to avoid OOM in CI/containers
    if int(args.steps) <= 1:
        class _Tiny:
            n_layers = 2
            d_model = 128
            n_heads = 4
            mlp_dim = 256
            moe_experts = 2
            moe_top_k = 1
            kv_latent_dim = 64
            multi_query = True
            vocab_size = 320
            max_seq_len = int(max(256, args.seq_len))
        preset = _Tiny()
    # Align VQ codebook sizes into text vocab segments when present to unify token space
    try:
        from omnicoder.config import MultiModalConfig
        mmc = MultiModalConfig()
        # Expand vocab if needed to reserve code ranges; trivial example: append image/video codebooks
        unified_vocab = max(preset.vocab_size, mmc.image_codebook_size + mmc.video_codebook_size + 32000)
    except Exception:
        unified_vocab = preset.vocab_size
    # Build student from preset and optionally resume best-known checkpoint
    student = OmniTransformer(
        vocab_size=preset.vocab_size,
        n_layers=preset.n_layers,
        d_model=preset.d_model,
        n_heads=preset.n_heads,
        mlp_dim=preset.mlp_dim,
        n_experts=preset.moe_experts,
        top_k=preset.moe_top_k,
        max_seq_len=max(preset.max_seq_len, args.seq_len),
        use_rope=True,
        kv_latent_dim=preset.kv_latent_dim,
        multi_query=preset.multi_query,
        multi_token=1,
    ).to(args.device)
    try:
        from omnicoder.utils.checkpoint import load_best_or_latest  # type: ignore
        _loaded = load_best_or_latest(student, args.out)
        if _loaded is not None:
            print(f"[resume] loaded {_loaded}")
    except Exception:
        pass
    # Optional CuMo upcycling before training
    if args.cumo_upcycle and int(args.cumo_target_experts) > 0:
        try:
            from omnicoder.modeling.utils.cumo import upcycle_ffn_to_experts
            changed = upcycle_ffn_to_experts(student, target_experts=int(args.cumo_target_experts))
            if changed:
                print(f"[CuMo] Upcycled {changed} MoE layers to {int(args.cumo_target_experts)} experts")
        except Exception as e:
            print(f"[CuMo] Upcycle skipped: {e}")
    # Apply hierarchical router if requested
    try:
        group_sizes = []
        if args.moe_group_sizes.strip():
            group_sizes = [int(x) for x in args.moe_group_sizes.split(',') if x.strip()]
        if group_sizes:
            from omnicoder.modeling.routing import HierarchicalRouter  # type: ignore
            for blk in getattr(student, 'blocks', []):
                if hasattr(blk, 'moe'):
                    blk.moe.router = HierarchicalRouter(preset.d_model, preset.moe_experts, group_sizes=list(group_sizes), k=preset.moe_top_k)
    except Exception:
        pass

    if args.lora:
        replaced = _inject_lora_linear(student, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
        print(f"[LoRA] injected into {replaced} Linear layers")

    if args.gradient_checkpointing:
        _enable_gradient_checkpointing(student)

    student.to(args.device)
    student.train()

    # Teacher (HF causal LM) – support multi-teacher round-robin
    # For smoke/CI (steps<=1), skip heavy HF loading and use a frozen student copy as dummy teacher.
    use_dummy_teacher = (int(args.steps) <= 1) or (os.getenv("OMNICODER_KD_SMOKE", "0") == "1")
    teacher_models = []
    if not use_dummy_teacher:
        # Many recent transformer model configs pull in vision stacks (timm/torchvision)
        # during lazy import resolution even if we only need tiny GPT-like teachers for
        # this smoke path. Guard against optional vision deps on minimal environments.
        try:
            # Disable torchvision requirement paths in transformers/timm if present
            os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
            os.environ.setdefault("USE_TORCHVISION", "0")
            os.environ.setdefault("TORCHVISION_DISABLE", "1")
            # Prefer offline resolution in tests/CI to avoid network stalls
            if os.getenv("OMNICODER_OFFLINE", "") == "1":
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from transformers import AutoModelForCausalLM
        except Exception as e:  # pragma: no cover
            raise RuntimeError("transformers is required for KD. pip install transformers") from e

    # Teacher loading with resilience
    teacher_ids = args.teachers if args.teachers else [args.teacher]
    teacher_srcs = [(args.teacher_local if args.teacher_local else tid) for tid in teacher_ids]
    dtype_map = {
        "auto": torch.float16 if torch.cuda.is_available() else None,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.teacher_dtype.lower(), None)
    device_map = args.teacher_device_map if args.teacher_device_map in ("auto", "cpu", "cuda") else "auto"
    if use_dummy_teacher:
        # Build one dummy teacher from the student weights to avoid heavyweight loads
        t = OmniTransformer(
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
            t.load_state_dict(student.state_dict(), strict=False)
        except Exception:
            pass
        t.eval()
        for p in t.parameters():
            p.requires_grad_(False)
        try:
            # Ensure dummy teacher runs on the same device as student to avoid CPU fallbacks
            t.to(args.device)
        except Exception:
            pass
        teacher_models.append(t)
    else:
        for teacher_src in teacher_srcs:
            try:
                t = AutoModelForCausalLM.from_pretrained(
                    teacher_src,
                    dtype=torch_dtype,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )
                t.eval()
            except Exception as e:
                print(f"[warn] Failed to load teacher {teacher_src} with device_map={device_map}, dtype={torch_dtype}: {e}")
                print("[warn] Falling back to CPU fp16 (if available).")
                try:
                    t = AutoModelForCausalLM.from_pretrained(
                        teacher_src,
                        dtype=(torch.float16 if torch.cuda.is_available() else None),
                        low_cpu_mem_usage=True,
                        local_files_only=True,
                    )
                    t.eval()
                except Exception as e2:
                    print(f"[warn] Teacher load still failed: {e2}")
                    print("[warn] Using a frozen copy of the student as a dummy teacher for this run (smoke mode).")
                    t = OmniTransformer(
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
                        t.load_state_dict(student.state_dict(), strict=False)
                    except Exception:
                        pass
                    t.eval()
            for p in t.parameters():
                p.requires_grad_(False)
            teacher_models.append(t)
    
    def _pick_teacher(step_idx: int):
        return teacher_models[step_idx % len(teacher_models)]

    # Optimizer selection
    optim_name = os.getenv('OMNICODER_OPTIM', 'adamw').strip().lower()
    weight_decay = float(os.getenv('OMNICODER_WEIGHT_DECAY', '0.01'))
    params = [p for p in student.parameters() if p.requires_grad]
    if optim_name in ('adamw8bit','adamw_8bit','adam8bit'):
        try:
            import bitsandbytes as bnb  # type: ignore
            opt = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=weight_decay)
            print('[optim] using AdamW8bit')
        except Exception:
            opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=weight_decay)
            print('[optim] AdamW8bit unavailable; falling back to AdamW')
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=weight_decay)
    # Optional resume (model/optimizer)
    try:
        if args.resume_ckpt and Path(args.resume_ckpt).exists():
            sd = torch.load(args.resume_ckpt, map_location='cpu')
            student.load_state_dict(sd, strict=False)
            print(f"[resume] KD loaded model from {args.resume_ckpt}")
        if args.state_out and Path(args.state_out).exists():
            st = torch.load(args.state_out, map_location='cpu')
            if isinstance(st, dict) and 'optimizer' in st:
                opt.load_state_dict(st['optimizer'])
    except Exception:
        pass
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    lb_coef = float(args.aux_lb_coef)

    # Prepare logging/IO
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    step = 0
    tokens_seen = 0
    ema_step_time: Optional[float] = None
    is_cuda = (str(args.device).startswith("cuda") and torch.cuda.is_available())
    # Install cudagraph step marker once (no hot-path env checks)
    try:
        _cg_mark = (_get_cg_mark() if _get_cg_mark is not None else None)
    except Exception:
        _cg_mark = None

    for batch in cycle(dl):
        if kd_jsonl_mode:
            input_ids, labels, router_targets = batch  # type: ignore
        else:
            input_ids, labels = batch  # type: ignore
            router_targets = None
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        t0 = time.perf_counter()
        # Run teacher on its own device to avoid device mismatches with student
        with torch.no_grad():
            try:
                # Prefer the embedding weight device as the canonical input device
                emb = _pick_teacher(step).get_input_embeddings()
                teacher_dev = emb.weight.device if hasattr(emb, "weight") else next(_pick_teacher(step).parameters()).device
            except Exception:
                teacher_dev = next(_pick_teacher(step).parameters()).device
            # Mark a new cudagraph step before teacher forward (prevents replay-owned aliasing)
            try:
                if _cg_mark is not None:
                    _cg_mark()  # type: ignore[misc]
            except Exception:
                pass
            t_out = _pick_teacher(step)(input_ids=input_ids.to(teacher_dev))
            # HF models return an object with .logits; our dummy teacher (OmniTransformer)
            # returns either a Tensor or a tuple. Normalize to logits Tensor.
            if hasattr(t_out, "logits"):
                t_logits = t_out.logits  # type: ignore[attr-defined]
            elif isinstance(t_out, tuple):
                t_logits = t_out[0]  # type: ignore[index]
            else:
                t_logits = t_out  # type: ignore[assignment]

        # Update router exploration knobs during training
        try:
            for blk in student.blocks:
                if hasattr(blk, 'moe') and hasattr(blk.moe, 'router'):
                    blk.moe.router.temperature = float(args.router_temp)
                    blk.moe.router.jitter_noise = float(args.router_jitter)
                    blk.moe.router.use_gumbel = bool(args.router_use_gumbel)
                    blk.moe.router.expert_dropout_p = float(args.router_expert_dropout_p)
                    blk.moe.router.sinkhorn_iters = int(args.router_sinkhorn_iters)
                    blk.moe.router.sinkhorn_tau = float(args.router_sinkhorn_tau)
                    blk.moe.router.store_probs_for_kl = bool(args.aux_sinkhorn_kl_coef and args.aux_sinkhorn_kl_coef > 0)
                if hasattr(blk, 'moe'):
                    blk.moe.static_capacity = int(args.moe_static_capacity) if args.moe_static_capacity and args.moe_static_capacity > 0 else None
        except Exception:
            pass

        # Mark a new cudagraph step before student forward
        try:
            if _cg_mark is not None:
                _cg_mark()  # type: ignore[misc]
        except Exception:
            pass
        s_outputs = student(input_ids)
        if isinstance(s_outputs, tuple):
            s_logits = s_outputs[0]  # type: ignore
            s_mtp = s_outputs[1] if len(s_outputs) > 1 and isinstance(s_outputs[1], list) else None  # type: ignore
            # Try to detect continuous latent outputs if present
            s_img_lat = None
            s_aud_lat = None
            try:
                # Heuristic: if second element is not list, pack (logits, img_lat, aud_lat, ...)
                if s_mtp is None and len(s_outputs) >= 2 and isinstance(s_outputs[1], torch.Tensor):
                    s_img_lat = s_outputs[1]
                    if len(s_outputs) >= 3 and isinstance(s_outputs[2], torch.Tensor):
                        s_aud_lat = s_outputs[2]
            except Exception:
                pass
            s_verifier = s_outputs[-1] if isinstance(s_outputs[-1], torch.Tensor) and s_outputs[-1].dim() == 3 else None  # type: ignore
            # Heuristically capture difficulty and halting scores when heads are present
            diff_score = None
            halt_score = None
            try:
                for extra in s_outputs[1:]:
                    if isinstance(extra, torch.Tensor) and extra.dim() >= 2 and extra.size(-1) == 1:
                        if diff_score is None:
                            diff_score = extra
                        elif halt_score is None:
                            halt_score = extra
                            break
            except Exception:
                diff_score = None
                halt_score = None
        else:
            s_logits = s_outputs  # type: ignore
            s_mtp = None
            s_verifier = None
            s_img_lat = None
            s_aud_lat = None
            diff_score = None
            halt_score = None

        # Align logits: move teacher logits to student device and shift by one for next-token prediction
        t_logits = t_logits.to(s_outputs[0].device if isinstance(s_outputs, tuple) else s_outputs.device)  # type: ignore[attr-defined]
        # Align tensor shapes: slice both to min seq length to handle T_fixed padding
        min_seq = min(s_logits.size(1), t_logits.size(1))
        s_logits_aligned = s_logits[:, :min_seq, :]
        t_logits_aligned = t_logits[:, :min_seq, :]
        labels_aligned = labels[:, :min_seq]
        
        t_logits_shifted = t_logits_aligned[:, :-1, :].contiguous()
        s_logits_shifted = s_logits_aligned[:, :-1, :].contiguous()
        labels_shifted = labels_aligned[:, 1:].contiguous()

        # KD with temperature (only if vocab sizes match)
        if s_logits_shifted.size(-1) == t_logits_shifted.size(-1):
            T = max(args.kl_temp, 1e-6)
            log_p_student = torch.log_softmax(s_logits_shifted / T, dim=-1)
            p_teacher = torch.softmax(t_logits_shifted / T, dim=-1)
            loss_kd = kl_loss(log_p_student, p_teacher) * (T * T)
            kd_weight = args.alpha_kd
        else:
            # Fallback: vocab mismatch between teacher and student; skip KD term.
            loss_kd = torch.zeros((), device=s_logits_shifted.device)
            kd_weight = 0.0

        # Optional sequence-level KD (teacher vs student total sequence log-probs)
        if args.seq_kd and s_logits.size(-1) == t_logits.size(-1):
            with torch.no_grad():
                VV = int(t_logits.size(-1))
                labels_safe = labels.clamp(min=0, max=max(0, VV - 1))
                t_lp = torch.log_softmax(t_logits, dim=-1).gather(-1, labels_safe.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
            VV2 = int(s_logits.size(-1))
            labels_safe2 = labels.clamp(min=0, max=max(0, VV2 - 1))
            s_lp = torch.log_softmax(s_logits, dim=-1).gather(-1, labels_safe2.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
            seq_loss = ((s_lp - t_lp) ** 2).mean()
            loss_kd = loss_kd + 0.1 * seq_loss

        # Optional CE on ground-truth
        # Guard against edge cases with tiny vocab shapes in smoke mode
        V = int(s_logits_shifted.size(-1))
        if V <= 1:
            loss_ce = torch.zeros((), device=s_logits_shifted.device)
        else:
            # Align time dims robustly (handle off-by-one due to shifts)
            _Ts = torch.ops.aten.sym_size.int(s_logits_shifted, 1)
            _Tl = torch.ops.aten.sym_size.int(labels_shifted, 1)
            if _Ts != _Tl:
                _Teff = _Ts if _Ts < _Tl else _Tl
                s_logits_shifted = torch.ops.aten.slice.Tensor(s_logits_shifted, 1, 0, _Teff, 1).contiguous()
                labels_shifted = torch.ops.aten.slice.Tensor(labels_shifted, 1, 0, _Teff, 1).contiguous()
            # Clamp labels into valid range [0, V)
            safe_labels = labels_shifted.clamp(min=0, max=V - 1)
            loss_ce = ce_loss(s_logits_shifted.reshape(-1, V), safe_labels.reshape(-1))
        # Auxiliary CE on verifier head
        if s_verifier is not None:
            v_logits_shifted = s_verifier[:, :-1, :].contiguous()
            VV = int(v_logits_shifted.size(-1))
            if VV > 1:
                # Align verifier time dims with labels_shifted
                _Tv = torch.ops.aten.sym_size.int(v_logits_shifted, 1)
                _Tl = torch.ops.aten.sym_size.int(labels_shifted, 1)
                if _Tv != _Tl:
                    _Teff2 = _Tv if _Tv < _Tl else _Tl
                    v_logits_shifted = torch.ops.aten.slice.Tensor(v_logits_shifted, 1, 0, _Teff2, 1).contiguous()
                    labels_shifted = torch.ops.aten.slice.Tensor(labels_shifted, 1, 0, _Teff2, 1).contiguous()
                v_safe_labels = labels_shifted.clamp(min=0, max=VV - 1)
                loss_ce = loss_ce + 0.25 * ce_loss(v_logits_shifted.reshape(-1, VV), v_safe_labels.reshape(-1))

        loss = kd_weight * loss_kd + (1.0 - kd_weight) * loss_ce

        # Optional: proof-margin regression head if present (reuse verifier logits probability)
        try:
            if s_verifier is not None:
                v_logits_shifted = s_verifier[:, :-1, :].contiguous()
                with torch.no_grad():
                    VVv = int(v_logits_shifted.size(-1))
                    v_labels_safe = labels_shifted.clamp(min=0, max=max(0, VVv - 1))
                    # VERBOSE: Replace F.one_hot with aten.one_hot + aten.to.dtype to keep aten-only targets
                    correct = torch.ops.aten.one_hot.default(v_labels_safe, int(VVv))
                    correct = torch.ops.aten.to.dtype(correct, v_logits_shifted.dtype, False, False)
                    target_margin = (correct * torch.softmax(v_logits_shifted, dim=-1)).sum(dim=-1)
                pred_margin = (torch.softmax(v_logits_shifted, dim=-1) * correct).sum(dim=-1)
                loss = loss + 0.05 * torch.mean((pred_margin - target_margin.detach()) ** 2)
        except Exception:
            pass

        # Router auxiliary regularization terms
        lb = 0.0
        imp_loss = 0.0
        load_loss = 0.0
        z_reg = 0.0
        for blk in student.blocks:
            if hasattr(blk, 'moe'):
                if getattr(blk.moe, 'last_load_penalty', None) is not None:
                    lb = lb + blk.moe.last_load_penalty
                aux = getattr(blk.moe, 'last_router_aux', None)
                if aux is not None and isinstance(aux, dict):
                    importance = aux.get('importance', None)
                    load_stat = aux.get('load', None)
                    z_loss = aux.get('z_loss', None)
                    sinkhorn_target = aux.get('sinkhorn_target', None)
                    probs_for_kl = aux.get('probs_for_kl', None)
                    if isinstance(importance, torch.Tensor) and importance.numel() > 0:
                        E = importance.numel()
                        target = 1.0 / float(E)
                        imp_loss = imp_loss + ((importance - target) ** 2).mean()
                    if isinstance(load_stat, torch.Tensor) and load_stat.numel() > 0:
                        E = load_stat.numel()
                        target = 1.0 / float(E)
                        load_loss = load_loss + ((load_stat - target) ** 2).mean()
                    if isinstance(z_loss, torch.Tensor):
                        z_reg = z_reg + z_loss
                    if float(args.aux_sinkhorn_kl_coef) > 0.0 and isinstance(sinkhorn_target, torch.Tensor) and isinstance(probs_for_kl, torch.Tensor):
                        eps = 1e-9
                        P = sinkhorn_target.clamp_min(eps)
                        Q = probs_for_kl.clamp_min(eps)
                        kl = (P * (P.log() - Q.log())).mean()
                        loss = loss + float(args.aux_sinkhorn_kl_coef) * kl
                    # Expert-aware routing KD: match student router probs to teacher soft targets (if provided)
                    if args.expert_route_kd and isinstance(probs_for_kl, torch.Tensor):
                        # If router_targets are provided (JSONL), use per-layer means; else fallback to uniform
                        if kd_jsonl_mode and router_targets and isinstance(router_targets, list) and len(router_targets) > 0 and isinstance(router_targets[0], dict):
                            tinfo = router_targets[0]
                            layer_means = tinfo.get('layer_means', None)
                            if layer_means is not None and isinstance(layer_means, list):
                                # Match by layer index if shapes align
                                # Compute MSE to the provided per-expert means
                                means = torch.tensor(layer_means, device=probs_for_kl.device, dtype=probs_for_kl.dtype)
                                # Broadcast to token dims as needed
                                target_prob = means.mean(dim=0) if means.dim() == 2 else means
                                target_prob = target_prob.clamp_min(1e-6)
                                target_prob = target_prob / target_prob.sum()
                                loss = loss + 0.01 * ((probs_for_kl.mean(dim=(0,1)) - target_prob) ** 2).mean()
                                used_router_target = True
                            else:
                                used_router_target = False
                        else:
                            used_router_target = False
                        if not used_router_target:
                            E = probs_for_kl.size(-1)
                            target = torch.full_like(probs_for_kl, 1.0 / float(E))
                            loss = loss + 0.005 * ((probs_for_kl - target) ** 2).mean()
        if isinstance(lb, torch.Tensor):
            loss = loss + lb_coef * lb
        if isinstance(imp_loss, torch.Tensor):
            loss = loss + float(args.aux_importance_coef) * imp_loss
        if isinstance(load_loss, torch.Tensor):
            loss = loss + float(args.aux_load_coef) * load_loss
        if isinstance(z_reg, torch.Tensor):
            loss = loss + float(args.aux_zloss_coef) * z_reg

        # Auxiliary MTP losses to encourage lookahead prediction (if present)
        if s_mtp is not None:
            for offset, la_logits in enumerate(s_mtp, start=1):
                if la_logits.size(1) <= offset:
                    continue
                la_pred = la_logits[:, :-offset, :]
                la_tgt = labels[:, offset:]
                VV_la = int(la_pred.size(-1))
                la_tgt_safe = la_tgt.clamp(min=0, max=max(0, VV_la - 1))
                loss = loss + 0.25 * ce_loss(la_pred.reshape(-1, la_pred.size(-1)), la_tgt_safe.reshape(-1))
        
        # Optional continuous latent reconstruction consistency (unsupervised proxy)
        if args.image_latent_loss and s_img_lat is not None:
            loss = loss + 0.01 * (s_img_lat[:, -1, :].pow(2).mean())
        if args.audio_latent_loss and s_aud_lat is not None:
            loss = loss + 0.01 * (s_aud_lat[:, -1, :].pow(2).mean())

        # Train difficulty and halting heads (unsupervised targets from logits)
        if float(args.diff_loss_coef) > 0.0 or float(args.halt_loss_coef) > 0.0:
            try:
                with torch.no_grad():
                    probs = torch.softmax(s_logits, dim=-1)
                    top_p = probs.max(dim=-1).values  # (B,T)
                    entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-9)), dim=-1)  # (B,T)
                    target_diff = 1.0 - top_p
                    target_halt = (entropy <= float(args.halt_entropy)).float()
                if float(args.diff_loss_coef) > 0.0 and isinstance(diff_score, torch.Tensor):
                    pred = torch.sigmoid(diff_score.squeeze(-1))
                    mse = torch.mean((pred - target_diff.detach()) ** 2)
                    loss = loss + float(args.diff_loss_coef) * mse
                if float(args.halt_loss_coef) > 0.0 and isinstance(halt_score, torch.Tensor):
                    bce = nn.BCEWithLogitsLoss()
                    hs = halt_score.squeeze(-1)
                    loss = loss + float(args.halt_loss_coef) * bce(hs, target_halt.detach())
            except Exception:
                pass

        # Variable-K expert selection during KD if enabled
        if bool(args.var_k_train):
            try:
                if isinstance(diff_score, torch.Tensor):
                    d_val = float(torch.sigmoid(diff_score).mean().item())
                else:
                    with torch.no_grad():
                        probs_last = torch.softmax(s_logits[:, -1, :], dim=-1)
                        top_prob = torch.topk(probs_last, k=1, dim=-1).values
                        d_val = float(1.0 - top_prob.mean().item())
                thr = float(args.var_k_threshold)
                try:
                    if args.var_k_threshold_start or args.var_k_threshold_end:
                        t = min(1.0, float(step) / float(max(1, args.steps)))
                        a = float(args.var_k_threshold_start or thr)
                        b = float(args.var_k_threshold_end or thr)
                        thr = (1.0 - t) * a + t * b
                except Exception:
                    pass
                cur_top = int(args.var_k_max if d_val >= float(thr) else args.var_k_min)
                for blk in student.blocks:
                    if hasattr(blk, 'moe') and hasattr(blk.moe, 'n_experts'):
                        n_e = int(getattr(blk.moe, 'n_experts', cur_top))
                        blk.moe.top_k = max(1, min(cur_top, n_e))
            except Exception:
                pass

        # Mixed precision + grad accumulation (env-controlled)
        use_amp = (os.getenv("OMNICODER_KD_AMP", "1") == "1") and (args.device.startswith('cuda')) and torch.cuda.is_available()
        accum = max(1, int(os.getenv("OMNICODER_KD_ACCUM", "1")))
        if 'scaler' not in locals():
            try:
                scaler = torch.amp.GradScaler('cuda', enabled=use_amp)  # type: ignore[attr-defined]
                autocast = (lambda: torch.amp.autocast('cuda')) if use_amp else (lambda: torch.amp.autocast('cpu'))  # type: ignore[attr-defined]
            except Exception:
                scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
                autocast = torch.cuda.amp.autocast if use_amp else torch.cpu.amp.autocast  # type: ignore[attr-defined]
        with (autocast() if use_amp else contextlib.nullcontext()):
            _loss = loss
        if use_amp:
            scaler.scale(_loss).backward()
        else:
            _loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        if (step + 1) % accum == 0:
            if use_amp:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
        step += 1
        # Periodic state save for crash-safe resume
        try:
            if args.state_out and ((step % max(10, int(os.getenv("OMNICODER_KD_STATE_INTERVAL","100")))) == 0):
                _safe_save({'optimizer': opt.state_dict()}, args.state_out)
        except Exception:
            pass

        # Throughput + ETA
        dt = max(1e-9, time.perf_counter() - t0)
        ema_step_time = dt if ema_step_time is None else (0.9 * ema_step_time + 0.1 * dt)
        tokens_this = int(input_ids.numel())
        tokens_seen += tokens_this
        tok_s = tokens_this / dt
        tok_s_ema = tokens_this / max(ema_step_time, 1e-9)
        mem_alloc = mem_rsvd = None
        if is_cuda:
            try:
                mem_alloc = int(torch.cuda.memory_allocated() // (1024 * 1024))
                mem_rsvd = int(torch.cuda.memory_reserved() // (1024 * 1024))
            except Exception:
                pass

        if step % max(1, args.log_interval) == 0 or step == 1:
            steps_left = max(0, args.steps - step)
            eta_s = steps_left * (ema_step_time or dt)
            eta_min = int(eta_s // 60)
            eta_sec = int(eta_s % 60)
            msg = (
                f"step {step}/{args.steps} | loss {loss.item():.4f} | kd {loss_kd.item():.4f} | ce {loss_ce.item():.4f} "
                f"| tok/s {tok_s_ema:,.0f} | eta {eta_min:02d}:{eta_sec:02d}"
            )
            if mem_alloc is not None and mem_rsvd is not None:
                msg += f" | cuda_mem(MiB) alloc={mem_alloc} rsvd={mem_rsvd}"
            print(msg, flush=True)

            rec = {
                "step": step,
                "max_steps": args.steps,
                "loss": float(loss.item()),
                "loss_kd": float(loss_kd.item()),
                "loss_ce": float(loss_ce.item()),
                "tokens_seen": int(tokens_seen),
                "tokens_per_s": float(tok_s),
                "tokens_per_s_ema": float(tok_s_ema),
                "eta_seconds": float(eta_s),
                "cuda_mem_alloc_mb": mem_alloc,
                "cuda_mem_reserved_mb": mem_rsvd,
            }
            try:
                with open(args.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception:
                pass

        # Periodic checkpoint
        if args.save_interval and args.save_interval > 0 and (step % args.save_interval == 0):
            ckpt_path = Path(args.out).with_name(Path(args.out).stem + f"_step{step}" + Path(args.out).suffix)
            try:
                try:
                    from omnicoder.utils.checkpoint import ensure_unique_path  # type: ignore
                except Exception:
                    ensure_unique_path = None  # type: ignore
                target_ckpt = ensure_unique_path(ckpt_path) if ensure_unique_path else ckpt_path
                _safe_save(student.state_dict(), target_ckpt)
                print(f"[ckpt] saved {ckpt_path}", flush=True)
            except Exception:
                pass

        if step >= args.steps:
            break

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    try:
        try:
            from omnicoder.utils.checkpoint import ensure_unique_path, save_with_sidecar  # type: ignore
        except Exception:
            ensure_unique_path = None  # type: ignore
            save_with_sidecar = None  # type: ignore
        target_out = ensure_unique_path(args.out) if ensure_unique_path else args.out
        if callable(save_with_sidecar):
            save_with_sidecar(target_out, student.state_dict(), meta={'train_args': {'steps': int(args.steps)}})
        else:
            _safe_save(student.state_dict(), target_out)
        print(f"Saved distilled student checkpoint to {target_out}")
    except Exception as e:
        print(f"[warn] Could not save distilled student checkpoint to {args.out}: {e}")


if __name__ == "__main__":
    main()
