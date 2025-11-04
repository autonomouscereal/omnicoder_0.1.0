from __future__ import annotations

"""ExecuTorch decode-step exporter wired for NNAPI delegate (best-effort).

This tool exports the PyTorch decode-step graph using torch.export and, when
ExecuTorch is installed, lowers it to Edge dialect and serializes a .pte file.

Usage:
  python -m omnicoder.export.executorch_export --out weights/text/decode_step.pte --mobile_preset mobile_4gb

Notes:
  - Requires ExecuTorch (`pip install executorch`) for full .pte export.
  - Falls back to saving a torch.export ExportedProgram graph as a debug .ep.pt file if ExecuTorch is missing.
  - NNAPI delegate wiring is backend/tooling dependent; this exporter emits a standard decode-step graph suitable for delegate partitioning.
"""

import argparse
import os
from pathlib import Path
import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore

from .onnx_export import DecodeStepWrapper  # reuse wrapper and model presets
from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.config import MobilePreset, MobilePreset2GB


def _build_model(preset: str, seq_len: int, vocab_size: int, multi_token: int) -> OmniTransformer:
    if preset in ("mobile_4gb", "mobile_2gb"):
        p = MobilePreset() if preset == "mobile_4gb" else MobilePreset2GB()
        model = OmniTransformer(
            vocab_size=p.vocab_size,
            n_layers=p.n_layers,
            d_model=p.d_model,
            n_heads=p.n_heads,
            mlp_dim=p.mlp_dim,
            n_experts=p.moe_experts,
            top_k=p.moe_top_k,
            max_seq_len=seq_len,
            use_rope=True,
            kv_latent_dim=p.kv_latent_dim,
            multi_query=p.multi_query,
            multi_token=multi_token,
        )
    else:
        model = OmniTransformer(vocab_size=vocab_size, multi_token=multi_token)
    # Disable HRM/SSM blocks for a stable decode-step graph
    model.eval()
    try:
        if hasattr(model, 'use_hrm'):
            model.use_hrm = False  # type: ignore[attr-defined]
        for blk in getattr(model, 'blocks', []):
            if hasattr(blk, 'use_ssm'):
                blk.use_ssm = False
                if hasattr(blk, 'ssm'):
                    blk.ssm = None
    except Exception:
        pass
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, required=True, help='Output .pte path (ExecuTorch). If ExecuTorch missing, writes .ep.pt debug graph.')
    ap.add_argument('--mobile_preset', type=str, default='mobile_4gb')
    ap.add_argument('--seq_len', type=int, default=1)
    ap.add_argument('--vocab_size', type=int, default=32000)
    ap.add_argument('--multi_token', type=int, default=1)
    args = ap.parse_args()

    model = _build_model(args.mobile_preset, args.seq_len, args.vocab_size, args.multi_token)
    wrapper = DecodeStepWrapper(model)

    # Dummy inputs for export
    B = 1
    T_past = 0
    H = model.blocks[0].attn.n_heads
    DL = model.blocks[0].attn.kv_latent_dim
    input_ids = torch.randint(0, model.vocab_size, (B, 1), dtype=torch.long)
    past = []
    for _ in model.blocks:
        past.append(torch.zeros(B, H, T_past, DL))  # k
    for _ in model.blocks:
        past.append(torch.zeros(B, H, T_past, DL))  # v

    # Export with torch.export (produces ExportedProgram)
    try:
        exported = torch.export.export(wrapper, (input_ids, *past))  # type: ignore[attr-defined]
    except Exception as e:
        raise SystemExit(f"torch.export failed: {e}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Try ExecuTorch lowering and .pte serialization if available
    try:
        import executorch  # type: ignore
        # The API surface can change; best-effort usage guarded by try/except
        try:
            # Edge dialect lowering
            from executorch.exir import to_edge  # type: ignore
            edge = to_edge(exported)
            # Serialize to .pte (API varies across versions)
            save_fn = getattr(edge, 'serialize', None) or getattr(edge, 'save', None)
            if callable(save_fn):
                save_fn(str(out_path))
                print(f"Wrote ExecuTorch program: {out_path}")
                return
        except Exception:
            pass
        # Fallback: save exported program for downstream tools
        ep_path = out_path.with_suffix('.ep.pt')
        _safe_save(exported, ep_path)
        print(f"ExecuTorch not available or API mismatch. Wrote ExportedProgram: {ep_path}")
    except Exception:
        # ExecuTorch missing
        ep_path = out_path.with_suffix('.ep.pt')
        _safe_save(exported, ep_path)
        print(f"ExecuTorch not installed. Wrote ExportedProgram: {ep_path}")


if __name__ == '__main__':
    main()

import argparse
import torch

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.config import MobilePreset, MobilePreset2GB


class DecodeStepWrapper(torch.nn.Module):
    """
    Export a single decoding step with KV-cache IO for ExecuTorch.
    Inputs
      - input_ids: (B, 1) int64
      - past caches per layer: k_lat_{i}, v_lat_{i} with shape (B, H, T_past, DL)
    Outputs
      - logits: (B, 1, V)
      - new caches per layer: nk_lat_{i}, nv_lat_{i} each (B, H, T_past+1, DL)
    """

    def __init__(self, model: OmniTransformer):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, *past: torch.Tensor):
        past_kv = None
        if past:
            assert len(past) % 2 == 0
            num_layers = len(past) // 2
            past_kv = []
            for i in range(num_layers):
                past_k = past[i]
                past_v = past[i + num_layers]
                past_kv.append((past_k, past_v))
        outputs = self.model(input_ids, past_kv=past_kv, use_cache=True)
        # Model may return (logits, new_kv, mtp_logits?, verifier_logits?)
        if isinstance(outputs, tuple):
            logits = outputs[0]  # type: ignore[index]
            new_kv = outputs[1]  # type: ignore[index]
            mtp_logits = outputs[2] if len(outputs) > 2 else None  # type: ignore[index]
        else:
            logits = outputs  # type: ignore[assignment]
            new_kv = []  # type: ignore[assignment]
            mtp_logits = None
        flat_new = []
        for k, v in new_kv:
            flat_new.append(k)
        for k, v in new_kv:
            flat_new.append(v)
        if mtp_logits is not None:
            return (logits,) + tuple(flat_new) + tuple(mtp_logits)
        return (logits,) + tuple(flat_new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, default='weights/omnicoder_decode_step.pte')
    ap.add_argument('--seq_len', type=int, default=1)
    ap.add_argument('--vocab_size', type=int, default=32000)
    ap.add_argument('--mobile_preset', type=str, default='mobile_4gb')
    ap.add_argument('--two_expert_split', action='store_true', help='Emit a duplicate program with 2-expert suffix for tooling that expects split variants')
    args = ap.parse_args()

    if args.mobile_preset in ('mobile_4gb','mobile_2gb'):
        preset = MobilePreset() if args.mobile_preset == 'mobile_4gb' else MobilePreset2GB()
        model = OmniTransformer(
            vocab_size=preset.vocab_size,
            n_layers=preset.n_layers,
            d_model=preset.d_model,
            n_heads=preset.n_heads,
            mlp_dim=preset.mlp_dim,
            n_experts=preset.moe_experts,
            top_k=preset.moe_top_k,
            max_seq_len=max(64, args.seq_len),
            use_rope=True,
            kv_latent_dim=preset.kv_latent_dim,
            multi_query=preset.multi_query,
            multi_token=1,
        )
    else:
        model = OmniTransformer(vocab_size=args.vocab_size, multi_token=1)
    model.eval()

    wrapper = DecodeStepWrapper(model)

    # Dummy inputs: (B=1,1), plus per-layer caches with T_past=0
    B = 1
    T_past = 0
    H = model.blocks[0].attn.n_heads
    DL = model.blocks[0].attn.kv_latent_dim
    input_ids = torch.randint(0, model.vocab_size, (B, 1), dtype=torch.long)
    past = []
    for _ in model.blocks:
        past.append(torch.zeros(B, H, T_past, DL))  # k
    for _ in model.blocks:
        past.append(torch.zeros(B, H, T_past, DL))  # v

    example_input = (input_ids, *past)

    try:
        from torch.export import export
        exp_prog = export(wrapper, example_input)
        with open(args.out, 'wb') as f:
            f.write(exp_prog.to_pte())
        print(f"Saved ExecuTorch decode-step program to {args.out}")
        if args.two_expert_split:
            alt = args.out.replace('.pte', '_2expert_hint.pte')
            try:
                with open(alt, 'wb') as f:
                    f.write(exp_prog.to_pte())
                print(f"Saved auxiliary 2-expert-hint program to {alt}")
            except Exception:
                pass
    except Exception:
        print("ExecuTorch export requires PyTorch 2.3+ and executorch tooling. Falling back to TorchScript .pt (not stateful):")
        ts = torch.jit.trace(wrapper, example_input, check_trace=False)
        fallback = args.out.replace('.pte', '.pt')
        ts.save(fallback)
        print(f"Saved TorchScript module to {fallback}")


if __name__ == "__main__":
    main()
