from __future__ import annotations

"""
Core ML MLProgram decode-step exporter (best-effort).

Exports a single decoding step with recurrent KV inputs/outputs using
coremltools MLProgram. Embeds lightweight metadata (RoPE and KV-latent
settings) into the model's user-defined metadata to guide on-device
attention mapping.

Notes
- Requires coremltools>=7.
- Graph coverage depends on PyTorch/coremltools versions. This exporter
  chooses static example shapes (B=1, past_len=1) for stability.
- For Apple native attention mapping and RoPE, a dedicated MIL pipeline
  may be needed; this exporter focuses on producing a functional MLProgram
  with KV streaming IO and metadata tags.
"""

import argparse
from pathlib import Path

import torch

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.config import MobilePreset, MobilePreset2GB, get_rope_scale_for_target_ctx


def _build_model(preset_name: str, seq_len: int, rope_base: float, target_ctx: int, yarn: bool, multi_token: int, window_size: int) -> OmniTransformer:
    if preset_name == "mobile_4gb":
        preset = MobilePreset()
    elif preset_name == "mobile_2gb":
        preset = MobilePreset2GB()
    else:
        preset = MobilePreset()
    rope_scale = 1.0
    if target_ctx and target_ctx > 0:
        rope_scale = get_rope_scale_for_target_ctx(preset.max_seq_len, target_ctx)
        if yarn:
            rope_scale = float(rope_scale) * 0.9
    model = OmniTransformer(
        vocab_size=preset.vocab_size,
        n_layers=preset.n_layers,
        d_model=preset.d_model,
        n_heads=preset.n_heads,
        mlp_dim=preset.mlp_dim,
        n_experts=preset.moe_experts,
        top_k=preset.moe_top_k,
        max_seq_len=max(64, seq_len),
        use_rope=True,
        kv_latent_dim=preset.kv_latent_dim,
        multi_query=preset.multi_query,
        multi_token=max(1, int(multi_token)),
        rope_scale=rope_scale,
        rope_base=float(rope_base),
    )
    # Disable HRM/SSM for export to simplify graphs unless explicitly requested
    try:
        import os as _os
        _export_hrm = (_os.getenv("OMNICODER_EXPORT_HRM", "0") == "1")
        if hasattr(model, "use_hrm"):
            if not _export_hrm:
                model.use_hrm = False  # type: ignore[attr-defined]
                if hasattr(model, "hrm"):
                    model.hrm = None  # type: ignore[attr-defined]
        for blk in getattr(model, "blocks", []):
            if hasattr(blk, "use_ssm") and blk.use_ssm:
                blk.use_ssm = False
                if hasattr(blk, "ssm"):
                    blk.ssm = None
            if window_size and hasattr(blk, "attn"):
                try:
                    blk.attn.window_size = int(window_size)
                except Exception:
                    pass
    except Exception:
        pass
    model.eval()
    return model


class DecodeStepWrapper(torch.nn.Module):
    def __init__(self, model: OmniTransformer):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, *past: torch.Tensor):
        past_kv = None
        if past:
            assert len(past) % 2 == 0
            L = len(past) // 2
            past_kv = []
            for i in range(L):
                past_kv.append((past[i], past[i + L]))
        outputs = self.model(input_ids, past_kv=past_kv, use_cache=True)
        if isinstance(outputs, tuple):
            logits = outputs[0]
            new_kv = outputs[1]
            mtp_logits = outputs[2] if len(outputs) > 2 else None
        else:
            logits = outputs
            new_kv = []
            mtp_logits = None
        flat_new = []
        for k, v in new_kv:
            flat_new.append(k)
        for k, v in new_kv:
            flat_new.append(v)
        if mtp_logits is not None:
            return (logits,) + tuple(flat_new) + tuple(mtp_logits)
        return (logits,) + tuple(flat_new)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Core ML MLProgram decode-step with KV streaming")
    ap.add_argument("--out", type=str, default="weights/text/omnicoder_decode_step.mlmodel")
    ap.add_argument("--preset", type=str, default="mobile_4gb", choices=["mobile_4gb","mobile_2gb"]) 
    ap.add_argument("--seq_len", type=int, default=1)
    ap.add_argument("--rope_base", type=float, default=10000.0)
    ap.add_argument("--target_ctx", type=int, default=0)
    ap.add_argument("--yarn", action="store_true")
    ap.add_argument("--multi_token", type=int, default=1)
    ap.add_argument("--window_size", type=int, default=0)
    ap.add_argument("--prefer_qlinear", action="store_true", help="Attempt to quantize weights post-conversion to prefer QLinearMatMul paths where supported (8-bit linear)")
    args = ap.parse_args()

    try:
        import coremltools as ct  # type: ignore
    except Exception as e:
        print("coremltools>=7 is required for Core ML export. Error:", e)
        return

    model = _build_model(args.preset, seq_len=int(args.seq_len), rope_base=float(args.rope_base), target_ctx=int(args.target_ctx), yarn=bool(args.yarn), multi_token=int(args.multi_token), window_size=int(args.window_size))
    wrapper = DecodeStepWrapper(model)

    # Example inputs (static for export stability)
    B = 1
    T_past = 1  # avoid zero-length dims for converters
    L = len(model.blocks)
    H = model.blocks[0].attn.n_heads
    DL = model.blocks[0].attn.kv_latent_dim
    input_ids = torch.randint(0, model.vocab_size, (B, 1), dtype=torch.long)
    past = []
    for _ in range(L):
        past.append(torch.zeros(B, H, T_past, DL))
    for _ in range(L):
        past.append(torch.zeros(B, H, T_past, DL))

    # Trace with TorchScript for coremltools conversion
    traced = torch.jit.trace(wrapper, (input_ids, *past), check_trace=False)
    # Convert to MLProgram
    try:
        # Prefer QLinearMatMul routing where coremltools supports it. Newer coremltools
        # expose compute_precision or MIL passes; we hint INT8 weight use for matmul
        # and leave activations in fp16/fp32 (device decides). Best-effort only.
        convert_kwargs = {
            "convert_to": "mlprogram",
            "minimum_deployment_target": getattr(ct.target, "iOS17", None) or None,
        }
        # Some coremltools versions accept compute_units/compute_precision
        try:
            from coremltools.converters.mil import Builder as _B  # type: ignore
            # Presence indicates MIL pipeline; we can pass through kwargs safely
            convert_kwargs["compute_units"] = getattr(ct.ComputeUnit, "ALL", None) or None
        except Exception:
            pass
        mlmodel = ct.convert(traced, **{k: v for k, v in convert_kwargs.items() if v is not None})
    except Exception:
        mlmodel = ct.convert(traced, convert_to="mlprogram")

    # Attach metadata for on-device mapping
    try:
        md = mlmodel.user_defined_metadata
        md["rope_base"] = str(float(args.rope_base))
        md["rope_scale"] = str(float(get_rope_scale_for_target_ctx(model.max_seq_len, int(args.target_ctx)) if int(args.target_ctx) > 0 else 1.0))
        md["kv_latent_dim"] = str(int(DL))
        md["heads"] = str(int(H))
        md["layers"] = str(int(L))
        if int(args.window_size) > 0:
            md["window_size"] = str(int(args.window_size))
    except Exception:
        pass

    # Optional post-conversion weight quantization to prefer QLinearMatMul
    if bool(args.prefer_qlinear) or os.getenv("OMNICODER_COREML_PREFER_QLINEAR", "1") == "1":
        try:
            # Newer API path
            try:
                from coremltools.optimize.coreml import quantization_utils as _q  # type: ignore
            except Exception:
                # Older API path
                from coremltools.models.neural_network import quantization_utils as _q  # type: ignore
            mlmodel = _q.quantize_weights(mlmodel, nbits=8, quantization_mode="linear")
            try:
                md = mlmodel.user_defined_metadata
                md["qlinear_preferred"] = "1"
            except Exception:
                pass
            print("[coreml] applied 8-bit linear weight quantization to prefer QLinearMatMul")
        except Exception as e:
            print(f"[warn] Core ML weight quantization skipped: {e}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    print(f"Saved Core ML MLProgram decode-step to {out_path}")


if __name__ == "__main__":
    main()


