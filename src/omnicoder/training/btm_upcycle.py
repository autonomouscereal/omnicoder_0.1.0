from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore


@dataclass
class DomainModel:
    name: str
    state_dict: Dict[str, torch.Tensor]


def _load_state(path: str) -> Dict[str, torch.Tensor]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    obj = torch.load(str(p), map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict):
        # flat sd
        return obj
    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def _find_moe_expert_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
    """Group FFN expert parameter keys per expert index.

    Heuristic: look for patterns like 'moe_layers.XX.experts.N.ffn.{w|b}' or 'blocks.XX.moe.experts.N...'
    Returns mapping expert_id -> list of parameter keys (names) in student model.
    """
    experts: Dict[str, List[str]] = {}
    for k in state_dict.keys():
        parts = k.split(".")
        # find an 'experts' segment followed by an index
        for i, seg in enumerate(parts):
            if seg == "experts" and i + 1 < len(parts):
                idx = parts[i + 1]
                # ensure idx looks like an integer
                if idx.isdigit():
                    experts.setdefault(idx, []).append(k)
                break
    # Keep deterministic order per expert
    for idx in list(experts.keys()):
        experts[idx] = sorted(experts[idx])
    return experts


def _copy_ffn_from_domain(
    student_sd: Dict[str, torch.Tensor],
    domain_sd: Dict[str, torch.Tensor],
    mapping: List[Tuple[str, str]],
) -> None:
    """Copy matching tensors from domain checkpoint into student according to a name mapping list.

    mapping: list of (student_key, domain_key) pairs that should be shape-compatible.
    """
    for sk, dk in mapping:
        if dk not in domain_sd or sk not in student_sd:
            continue
        if student_sd[sk].shape != domain_sd[dk].shape:
            continue
        student_sd[sk] = domain_sd[dk].clone()


def _build_round_robin_mapping(
    student_expert_keys: Dict[str, List[str]],
    domain_expert_keys: Dict[str, List[str]],
) -> List[Tuple[str, str]]:
    mapping: List[Tuple[str, str]] = []
    # Flatten per expert
    s_experts = sorted(student_expert_keys.keys(), key=lambda x: int(x))
    d_experts = sorted(domain_expert_keys.keys(), key=lambda x: int(x))
    if not d_experts:
        return mapping
    for i, s_idx in enumerate(s_experts):
        d_idx = d_experts[i % len(d_experts)]
        s_keys = student_expert_keys[s_idx]
        d_keys = domain_expert_keys[d_idx]
        # Pair by suffix match when possible, else fallback by position
        d_by_suffix = {k.split("experts."+d_idx+".", 1)[-1]: k for k in d_keys}
        for sk in s_keys:
            suffix = sk.split("experts."+s_idx+".", 1)[-1]
            dk = d_by_suffix.get(suffix, d_keys[len(mapping) % len(d_keys)])
            mapping.append((sk, dk))
    return mapping


def _finetune_router(
    model: torch.nn.Module,
    steps: int,
    device: str,
    lr: float,
) -> None:
    # Freeze everything except router parameters (look for names containing 'router' or 'gate')
    for n, p in model.named_parameters():
        if ("router" in n) or ("gate" in n and "weight" in n):
            p.requires_grad = True
        else:
            p.requires_grad = False
    model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    # Minimal synthetic loop to adjust router logits distribution
    vocab = getattr(model, "vocab_size", 32000)
    seq = 64
    for _ in range(max(1, steps)):
        x = torch.randint(0, vocab, (2, seq), device=device)
        try:
            out = model(x)
            loss = 0.0
            # Prefer a router aux loss if present on the module
            if isinstance(out, dict) and "router_aux_loss" in out:
                loss = out["router_aux_loss"]
            else:
                # Fallback tiny l2 on any router logits parameters to regularize
                reg = 0.0
                for n, p in model.named_parameters():
                    if p.requires_grad and p.ndim > 1:
                        reg = reg + (p ** 2).mean()
                loss = reg * 1e-4
        except Exception:
            # Robust fallback: sum of router params
            reg = 0.0
            for n, p in model.named_parameters():
                if p.requires_grad and p.ndim > 1:
                    reg = reg + (p ** 2).mean()
            loss = reg * 1e-4
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def main() -> None:
    ap = argparse.ArgumentParser(description="Branch-Train-Merge (BTM) upcycling: merge domain experts into MoE and fine-tune router")
    ap.add_argument("--student_ckpt", type=str, required=True, help="Path to base student checkpoint (MoE)")
    ap.add_argument("--domains", type=str, nargs="+", required=True, help="List of domain checkpoints to merge (code/math/VL/ASR/TTS etc.)")
    ap.add_argument("--assign", type=str, default="round_robin", choices=["round_robin"], help="Expert assignment strategy")
    ap.add_argument("--finetune_router_steps", type=int, default=int(os.getenv("OMNICODER_BTM_ROUTER_STEPS", "200")))
    ap.add_argument("--finetune_router_lr", type=float, default=float(os.getenv("OMNICODER_BTM_ROUTER_LR", "5e-5")))
    ap.add_argument("--device", type=str, default=os.getenv("OMNICODER_TRAIN_DEVICE", "cuda"))
    ap.add_argument("--out", type=str, required=True, help="Output merged student checkpoint path")
    args = ap.parse_args()

    student_sd = _load_state(args.student_ckpt)
    student_experts = _find_moe_expert_keys(student_sd)
    if not student_experts:
        raise RuntimeError("Student checkpoint does not appear to contain MoE experts (experts.* parameter groups not found)")

    domains: List[DomainModel] = []
    for d in args.domains:
        domains.append(DomainModel(name=Path(d).stem, state_dict=_load_state(d)))

    # Build and apply mappings expert-by-expert per domain, cycling through experts
    applied: List[Tuple[str, str, str]] = []  # (student_key, domain_name, domain_key)
    for i, domain in enumerate(domains):
        domain_experts = _find_moe_expert_keys(domain.state_dict)
        mapping = _build_round_robin_mapping(student_experts, domain_experts)
        _copy_ffn_from_domain(student_sd, domain.state_dict, mapping)
        for sk, dk in mapping:
            applied.append((sk, domain.name, dk))

    # Write a merged checkpoint to disk
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _safe_save({"state_dict": student_sd, "btm_merge": {"applied": applied, "domains": [d.name for d in domains]}}, str(out_path))

    # Optional router fine-tune in-memory (best-effort; robust to missing imports)
    if int(args.finetune_router_steps) > 0:
        try:
            from omnicoder.modeling.transformer_moe import OmniTransformer  # type: ignore

            # Instantiate a student model matching the checkpoint sizes when possible
            # Minimal config instantiation; fall back to loading strictly by state_dict
            model = OmniTransformer()
            missing, unexpected = model.load_state_dict(student_sd, strict=False)
            if len(unexpected) > 0:
                # Retry with strict=False (already), continue
                pass
            _finetune_router(model, steps=int(args.finetune_router_steps), device=args.device, lr=float(args.finetune_router_lr))
            student_sd = model.state_dict()
            _safe_save({"state_dict": student_sd, "btm_merge": {"applied": applied, "domains": [d.name for d in domains], "router_ft": int(args.finetune_router_steps)}}, str(out_path))
        except Exception:
            # Keep merged weights without router fine-tune when model import/instantiate fails
            pass

    # Emit a small summary next to the output ckpt
    try:
        (out_path.with_suffix(".json")).write_text(json.dumps({
            "student": str(args.student_ckpt),
            "domains": [d.name for d in domains],
            "assign": args.assign,
            "applied": len(applied),
            "router_steps": int(args.finetune_router_steps),
        }, indent=2), encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    main()


