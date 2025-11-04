import argparse
import json
from pathlib import Path

import torch

from omnicoder.config import MobilePreset, MobilePreset2GB, get_mobile_preset
from omnicoder.inference.memory_estimator import (
    estimate_kv_cache_bytes,
    estimate_model_memory_bytes,
    human_bytes,
)
from omnicoder.modeling.transformer_moe import OmniTransformer


def _build_model(preset_name: str, seq_len: int, multi_token: int = 1) -> OmniTransformer:
    try:
        preset = get_mobile_preset(preset_name)
    except Exception:
        if preset_name == "mobile_4gb":
            preset = MobilePreset()
        elif preset_name == "mobile_2gb":
            preset = MobilePreset2GB()
        else:
            raise
    return OmniTransformer(
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
        multi_token=multi_token,
    )


def _count_fusions(model_path: Path) -> dict:
    counts = {"Attention": 0, "QLinearMatMul": 0, "QuantizeLinear": 0, "DequantizeLinear": 0}
    try:
        import onnx  # type: ignore
    except Exception:
        return counts
    try:
        m = onnx.load(str(model_path))
    except Exception:
        return counts
    for n in m.graph.node:
        if n.domain == 'com.microsoft' and n.op_type == 'Attention':
            counts["Attention"] += 1
        if n.op_type == 'QLinearMatMul':
            counts["QLinearMatMul"] += 1
        if n.op_type == 'QuantizeLinear':
            counts["QuantizeLinear"] += 1
        if n.op_type == 'DequantizeLinear':
            counts["DequantizeLinear"] += 1
    return counts


def _export_onnx_decode_step(model: OmniTransformer, out_path: Path, opset: int = 17) -> None:
    from .onnx_export import DecodeStepWrapper  # reuse wrapper definition

    model.eval()
    wrapper = DecodeStepWrapper(model)
    # Dummy inputs with empty past caches
    B = 1
    T_past = 0
    H = model.blocks[0].attn.n_heads
    DL = model.blocks[0].attn.kv_latent_dim
    input_ids = torch.randint(0, model.vocab_size, (B, 1), dtype=torch.long)
    # Build past K/V tensors using comprehensions to minimize Python overhead
    past_k = [torch.zeros(B, H, T_past, DL) for _ in model.blocks]
    past_v = [torch.zeros(B, H, T_past, DL) for _ in model.blocks]
    past = past_k + past_v

    nb = len(model.blocks)
    input_names = ["input_ids", *[f"k_lat_{i}" for i in range(nb)], *[f"v_lat_{i}" for i in range(nb)]]
    # Probe wrapper once to determine actual number of outputs
    try:
        with torch.inference_mode():
            trial = wrapper(input_ids, *past)
        n_actual = (len(trial) if isinstance(trial, tuple) else 1)
    except Exception:
        n_actual = 1 + (2 * nb)
    # Build output names to exactly match n_actual (prefer per-layer mapping)
    preferred = ["logits"] + [f"nk_lat_{i}" for i in range(nb)] + [f"nv_lat_{i}" for i in range(nb)]
    output_names: list[str] = []
    i = 0
    while i < n_actual and i < len(preferred):
        output_names.append(preferred[i])
        i += 1
    while i < n_actual:
        output_names.append(f"out_{i}")
        i += 1
    # Dynamic axes: inputs only
    dynamic_axes = {"input_ids": {1: "t_step"}}
    for name in input_names[1:]:
        dynamic_axes[name] = {2: "t_past"}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (input_ids, *past),
        str(out_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )


def _quantize_onnx_dynamic(model_path: Path, out_path: Path) -> bool:
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except Exception:
        return False
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(model_input=str(model_path), model_output=str(out_path), weight_type=QuantType.QInt8)
    return True


def _export_executorch_decode_step(model: OmniTransformer, out_path: Path) -> bool:
    try:
        from torch.export import export as torch_export
    except Exception:
        return False

    from .executorch_export import DecodeStepWrapper  # reuse wrapper definition

    model.eval()
    wrapper = DecodeStepWrapper(model)
    B = 1
    T_past = 0
    H = model.blocks[0].attn.n_heads
    DL = model.blocks[0].attn.kv_latent_dim
    input_ids = torch.randint(0, model.vocab_size, (B, 1), dtype=torch.long)
    L = len(model.blocks)
    past = [torch.zeros(B, H, T_past, DL) for _ in range(L)]  # k
    past += [torch.zeros(B, H, T_past, DL) for _ in range(L)]  # v
    example_input = (input_ids, *past)

    try:
        exp_prog = torch_export(wrapper, example_input)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(exp_prog.to_pte())
        return True
    except Exception:
        # Fallback TorchScript
        ts = torch.jit.trace(wrapper, example_input, check_trace=False)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fallback = str(out_path).replace(".pte", ".pt")
        ts.save(fallback)
        return False


def _write_summary(summary_path: Path, preset_name: str, seq_len_for_budget: int) -> None:
    if preset_name == "mobile_4gb":
        preset = MobilePreset()
    else:
        preset = MobilePreset2GB()
    w_bytes = estimate_model_memory_bytes(preset)
    kv_bytes = estimate_kv_cache_bytes(preset, seq_len_for_budget, batch_size=1)
    total = w_bytes + kv_bytes
    summary = {
        "preset": preset.name,
        "weights_bytes": w_bytes,
        "kv_cache_bytes_seq_len": seq_len_for_budget,
        "kv_cache_bytes": kv_bytes,
        "total_bytes": total,
        "human": {
            "weights": human_bytes(w_bytes),
            "kv_cache": human_bytes(kv_bytes),
            "total": human_bytes(total),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="One-command mobile packager for OmniCoder text path (optionally bundles multimodal decoders)")
    ap.add_argument("--preset", default="mobile_4gb", choices=["mobile_4gb", "mobile_2gb"], help="Mobile preset")
    ap.add_argument("--out_dir", default="weights/text", help="Output directory for exported artifacts")
    ap.add_argument("--seq_len_budget", type=int, default=4096, help="Seq len used for KV cache memory estimate")
    ap.add_argument("--enforce_budget_bytes", type=int, default=0, help="If >0, fail packaging when weights+KV exceed this budget")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--quantize_onnx", action="store_true", help="Apply ONNX dynamic int8 quantization")
    ap.add_argument("--quantize_onnx_per_op", action="store_true", help="Run per-op ONNX PTQ helper with preset")
    ap.add_argument("--insert_qdq", action="store_true", help="Insert Q/DQ wrappers around MatMul inputs to improve QLinear fusions (guarded)")
    ap.add_argument("--onnx_preset", type=str, default="generic", choices=["generic","nnapi","coreml","dml"], help="Per-op PTQ preset")
    ap.add_argument("--provider_profile", type=str, default="", help="Optional provider profile JSON; maps provider to PTQ preset and session options")
    ap.add_argument("--export_executorch", action="store_true", help="Export ExecuTorch stateful decode-step program")
    ap.add_argument("--export_hrm", action="store_true", help="If set, keep HRM active during export (defaults to disabled)")
    ap.add_argument("--pt_int4_weights", action="store_true", help="Before ExecuTorch export, replace Linear with Int4Linear (weight-only int4)")
    ap.add_argument("--multi_token", type=int, default=1, help="Number of MTP heads (>1 adds lookahead heads; decode-step uses 1)")
    ap.add_argument("--onnx_provider_hint", type=str, default="CPUExecutionProvider", help="Provider hint for target device (e.g., NNAPIExecutionProvider, CoreMLExecutionProvider)")
    ap.add_argument("--nnapi_maps", action="store_true", help="Write NNAPI quantization maps and per-node maps sidecars")
    ap.add_argument("--parity_check", action="store_true", help="Run ONNX parity check vs PyTorch decode-step and write a JSON report next to ONNX")
    # Optional multimodal decoder bundling (reuse exporters from autofetch_backbones)
    ap.add_argument("--with_vision", action="store_true", help="Export a compact vision backbone and include provider maps")
    ap.add_argument("--vision_backend", type=str, default="", help="timm backbone (e.g., mobilevit_xs, efficientvit_lite0, vit_tiny_patch16_224)")
    ap.add_argument("--vision_export_coreml", action="store_true")
    ap.add_argument("--vision_export_executorch", action="store_true")
    ap.add_argument("--vision_export_grounding", action="store_true", help="Also export lightweight grounding heads (simple, rep_rta) to ONNX (and optionally Core ML/ExecuTorch)")
    ap.add_argument("--vision_tokens", type=int, default=196, help="Dummy token count (T) for grounding head export")
    ap.add_argument("--vision_d_model", type=int, default=384, help="d_model for grounding head export")
    ap.add_argument("--vision_num_props", type=int, default=10, help="Number of proposals for grounding head export")
    ap.add_argument("--with_sd", action="store_true", help="Export Stable Diffusion components (ONNX/Core ML/ExecuTorch) and provider maps")
    ap.add_argument("--sd_model", type=str, default="", help="HF id for SD, e.g., runwayml/stable-diffusion-v1-5")
    ap.add_argument("--sd_local_path", type=str, default="", help="Local SD pipeline path")
    ap.add_argument("--sd_export_coreml", action="store_true")
    ap.add_argument("--sd_export_executorch", action="store_true")
    ap.add_argument("--with_vqdec", action="store_true", help="Export image VQ decoder (indices->image) and provider maps")
    ap.add_argument("--image_vq_codebook", type=str, default="", help="Path to Image VQ-VAE codebook blob (.pt)")
    ap.add_argument("--vqdec_hq", type=int, default=14)
    ap.add_argument("--vqdec_wq", type=int, default=14)
    ap.add_argument("--vqdec_export_coreml", action="store_true")
    ap.add_argument("--vqdec_export_executorch", action="store_true")
    # Optional: export continuous latent heads (standalone ONNX)
    ap.add_argument("--with_latent_heads", action="store_true", help="Export standalone ONNX for continuous latent heads (image/audio)")
    ap.add_argument("--with_piper", action="store_true", help="Download a Piper TTS .onnx model into weights/audio")
    ap.add_argument("--piper_url", type=str, default="", help="URL to a Piper .onnx model to download (optional)")
    # Optional provider profiles for non-text bundles
    ap.add_argument("--vision_provider_profile", type=str, default="", help="Provider profile JSON to copy next to vision exports")
    ap.add_argument("--image_provider_profile", type=str, default="", help="Provider profile JSON to copy next to SD exports")
    ap.add_argument("--vqdec_provider_profile", type=str, default="", help="Provider profile JSON to copy next to VQ decoder exports")
    # Video reference only
    ap.add_argument("--video_model", type=str, default="", help="HF id for text-to-video or image-to-video pipeline (reference only)")
    ap.add_argument("--video_local_path", type=str, default="", help="Local path to a video pipeline (reference only)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    onnx_path = out_dir / "omnicoder_decode_step.onnx"
    onnx_q_path = out_dir / "omnicoder_decode_step_int8.onnx"
    onnx_q_perop_path = out_dir / "omnicoder_decode_step_int8_perop.onnx"
    onnx_fused_path = out_dir / "omnicoder_decode_step_fused.onnx"
    et_path = out_dir / "omnicoder_decode_step.pte"
    summary_path = out_dir / "mobile_packager_summary.json"
    nnapi_quant_maps_path = out_dir / "nnapi_quant_maps.json"
    nnapi_nodes_path = out_dir / "nnapi_nodes.json"
    coreml_quant_maps_path = out_dir / "coreml_quant_maps.json"
    dml_quant_maps_path = out_dir / "dml_quant_maps.json"

    print(f"[1/4] Building model for preset={args.preset} ...")
    model = _build_model(args.preset, seq_len=1, multi_token=1)  # decode-step graph uses single-token steps

    print(f"[2/4] Exporting ONNX decode-step to {onnx_path} ...")
    # When HRM is enabled for export, we rely on the exporter arg to keep it active
    if args.export_hrm and hasattr(model, 'use_hrm'):
        try:
            model.use_hrm = True  # type: ignore[attr-defined]
        except Exception:
            pass
    _export_onnx_decode_step(model, onnx_path, opset=args.opset)
    # Apply ONNX fusion pass for Attention and QDQ MatMul
    try:
        from .onnx_fuse_attention import fuse_and_pack, insert_qdq_inputs  # type: ignore
        fused, qdq = fuse_and_pack(str(onnx_path), str(onnx_fused_path), provider_hint=args.onnx_provider_hint)
        print(f"[2a] Fused attention blocks={fused}, QDQ MatMul pairs={qdq} -> {onnx_fused_path}")
        # Default-enable QDQ insertion for mobile provider hints unless user opts out by not passing the flag in non-mobile contexts
        want_qdq = bool(args.insert_qdq)
        if not want_qdq and args.onnx_provider_hint in ("NNAPIExecutionProvider", "CoreMLExecutionProvider", "DmlExecutionProvider"):
            want_qdq = True
        if want_qdq:
            qdq_src = str(onnx_fused_path if onnx_fused_path.exists() else onnx_path)
            qdq_ok = insert_qdq_inputs(qdq_src, str(onnx_fused_path))
            print(f"  [2a+] Insert QDQ: {'ok' if qdq_ok else 'skipped'}")
    except Exception as e:
        print(f"  [warn] fusion pass skipped: {e}")

    # Optional: map provider profile to PTQ preset and op list
    ptq_preset = args.onnx_preset
    ptq_op_types: list[str] | None = None
    if args.provider_profile:
        try:
            prof = json.loads(Path(args.provider_profile).read_text(encoding='utf-8'))
            prov = str(prof.get('provider', ''))
            mapping = {
                'NNAPIExecutionProvider': 'nnapi',
                'CoreMLExecutionProvider': 'coreml',
                'DmlExecutionProvider': 'dml',
            }
            if prov in mapping:
                ptq_preset = mapping[prov]
            # Custom op coverage override
            if 'ptq_op_types' in prof and isinstance(prof['ptq_op_types'], list):
                ptq_op_types = [str(x) for x in prof['ptq_op_types']]
        except Exception as e:
            print(f"  [warn] failed to parse provider profile: {e}")

    if args.quantize_onnx:
        print(f"[2b] Applying ONNX dynamic int8 quantization -> {onnx_q_path} ...")
        ok = _quantize_onnx_dynamic(onnx_path, onnx_q_path)
        if not ok:
            print("  Skipped: onnxruntime quantization not available.")
    if args.quantize_onnx_per_op:
        try:
            import sys as _sys
            print(f"[2c] Per-op PTQ preset={ptq_preset} -> {onnx_q_perop_path} ...")
            cmd = [
                __import__("sys").executable, "-m", "omnicoder.export.onnx_quantize_per_op",
                "--model", str(onnx_path),
                "--out", str(onnx_q_perop_path),
                "--preset", ptq_preset,
                "--auto_exclude",
                "--per_channel",
            ]
            if ptq_op_types:
                cmd.extend(["--op_types", ",".join(ptq_op_types)])
            _run_per_op = __import__("subprocess").run(cmd, check=False)
            # Quick coverage check: warn if no QDQ/QLinear nodes present
            try:
                import onnx  # type: ignore
                m = onnx.load(str(onnx_q_perop_path))
                has_quant = any(n.op_type in ("QuantizeLinear","DequantizeLinear","QLinearMatMul") for n in m.graph.node)
                if not has_quant:
                    print("  [warn] PTQ produced no QDQ/QLinear nodes; verify preset/provider profile and graph patterns.")
            except Exception:
                pass
        except Exception as e:
            print(f"  Skipped per-op PTQ: {e}")

    # Optional parity check (PyTorch vs ONNX decode-step)
    if args.parity_check:
        try:
            from omnicoder.tools.onnx_parity_check import main as _parity_main  # type: ignore
            import sys as _sys
            _argv = _sys.argv
            _sys.argv = [
                _argv[0],
                "--onnx", str(onnx_path),
                "--preset", str(args.preset),
                "--seq_len", "1",
            ]
            try:
                _parity_main()
            finally:
                _sys.argv = _argv
        except Exception as e:
            print(f"  [warn] parity check skipped: {e}")

    if args.export_executorch:
        print(f"[3/4] Exporting ExecuTorch decode-step to {et_path} ...")
        if args.pt_int4_weights:
            try:
                from omnicoder.modeling.quant.int4_linear import quantize_module_int4_linear
                replaced = quantize_module_int4_linear(model)
                print(f"  [int4] Replaced {replaced} Linear layers with Int4Linear")
            except Exception as e:
                print(f"  [int4] Skipped int4 weight replacement: {e}")
        ok = _export_executorch_decode_step(model, et_path)
        if not ok:
            print("  Saved TorchScript fallback (.pt). For .pte export, install PyTorch 2.3+ and executorch.")

    # Optionally write NNAPI maps sidecars (quantization prefs and per-node maps)
    if args.nnapi_maps or args.onnx_provider_hint in ("NNAPIExecutionProvider","CoreMLExecutionProvider","DmlExecutionProvider") or args.export_executorch:
        try:
            from .executorch_quant_maps import write_nnapi_maps, write_nnapi_node_maps, write_coreml_maps, write_dml_maps  # type: ignore
            write_nnapi_maps(nnapi_quant_maps_path)
            write_nnapi_node_maps(onnx_fused_path if onnx_fused_path.exists() else onnx_path, nnapi_nodes_path)
            write_coreml_maps(coreml_quant_maps_path)
            write_dml_maps(dml_quant_maps_path)
            print(f"[3a] Wrote maps: NNAPI={nnapi_quant_maps_path}, nodes={nnapi_nodes_path}; CoreML={coreml_quant_maps_path}; DML={dml_quant_maps_path}")
        except Exception as e:
            print(f"  [warn] failed to write NNAPI maps: {e}")

    # Minimal sidecar hints for Core ML attention mapping and ExecuTorch delegate
    try:
        hints = {}
        n_layers = len(getattr(model, 'blocks', []))
        heads = getattr(getattr(model.blocks[0], 'attn', object()), 'n_heads', 0) if n_layers else 0
        dl = getattr(getattr(model.blocks[0], 'attn', object()), 'kv_latent_dim', 0) if n_layers else 0
        # Core ML attention mapping hints
        coreml_hints = {
            "use_apple_attention": True,
            "prefer_fp16": True,
            "rope": {"enabled": True, "base": 10000.0},
            "kv_latent": {"enabled": bool(dl), "dl_per_layer": dl},
            "heads": heads,
            "layers": n_layers,
        }
        (out_dir / "coreml_attention_hints.json").write_text(__import__('json').dumps(coreml_hints, indent=2))
        # ExecuTorch delegate hints
        execu_hints = {
            "delegate": "nnapi",
            "preferences": {"int8_preferred": True, "attention_fused": True},
            "ops": {"Attention": {"quantize": True}, "MatMul": {"quantize": True}},
        }
        (out_dir / "executorch_delegate_hints.json").write_text(__import__('json').dumps(execu_hints, indent=2))
    except Exception:
        pass

    # Enforce fused Attention/QLinear presence for Core ML / NNAPI when hinted
    try:
        if args.onnx_provider_hint in ("NNAPIExecutionProvider", "CoreMLExecutionProvider"):
            fused_src = onnx_fused_path if onnx_fused_path.exists() else onnx_path
            counts = _count_fusions(fused_src)
            attn_ok = counts.get("Attention", 0) > 0
            q_ok = (counts.get("QLinearMatMul", 0) > 0) or (counts.get("QuantizeLinear", 0) > 0 and counts.get("DequantizeLinear", 0) > 0)
            if not attn_ok:
                print(f"[fail] {args.onnx_provider_hint} requires fused com.microsoft::Attention; found {counts}")
                raise SystemExit(3)
            # Require QLinear path for NNAPI; Core ML tolerates QDQ but prefer QLinear
            if args.onnx_provider_hint == "NNAPIExecutionProvider" and not q_ok:
                print(f"[fail] NNAPI requires QLinearMatMul or Q/DQ coverage; found {counts}")
                raise SystemExit(3)
        # For DML, prefer QLinear present; warn when missing
        if args.onnx_provider_hint in ("DmlExecutionProvider", "DMLExecutionProvider"):
            fused_src = onnx_fused_path if onnx_fused_path.exists() else onnx_path
            counts = _count_fusions(fused_src)
            if counts.get("QLinearMatMul", 0) <= 0:
                print(f"[warn] DML prefers QLinearMatMul; found {counts}")
    except SystemExit:
        raise
    except Exception as e:
        print(f"  [warn] fusion enforcement skipped: {e}")

    print(f"[4/4] Writing memory budget summary -> {summary_path} ...")
    _write_summary(summary_path, args.preset, args.seq_len_budget)
    if int(args.enforce_budget_bytes) > 0:
        try:
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            total = int(summary.get('total_bytes', 0))
            if total > int(args.enforce_budget_bytes):
                print(f"[budget] FAIL: total_bytes={total} exceeds budget={int(args.enforce_budget_bytes)}")
                raise SystemExit(2)
            else:
                print(f"[budget] OK: total_bytes={total} <= budget={int(args.enforce_budget_bytes)}")
        except Exception as e:
            print(f"[budget] check skipped: {e}")

    # Integrate KVQ calibration if present under weights/; copy sidecar next to text ONNX
    try:
        kvq_src = Path('weights') / 'kvq_calibration.json'
        if kvq_src.exists():
            kvq_dst = onnx_path.with_suffix('.kvq.json')
            kvq_dst.write_text(kvq_src.read_text(encoding='utf-8'), encoding='utf-8')
            print(f"[kvq] Copied calibration sidecar to {kvq_dst}")
    except Exception as e:
        print(f"  [kvq] skip copy: {e}")

    # Write KV retention sidecar next to ONNX (compressive slots + window policy)
    try:
        import os as _os
        import json as _json
        # Derive defaults: slots from env or 4; window from preset default
        slots = int(_os.getenv('OMNICODER_COMPRESSIVE_SLOTS', '4'))
        try:
            # Prefer preset default window if available
            if hasattr(model, 'default_window_size'):
                window = int(getattr(model, 'default_window_size'))  # type: ignore[attr-defined]
            else:
                window = 2048 if args.preset == 'mobile_4gb' else 1024
        except Exception:
            window = 2048
        ret_path = onnx_path.with_suffix('.kv_retention.json')
        blob = {"compressive_slots": int(max(0, slots)), "window_size": int(max(1, window)), "schema": 1}
        ret_path.write_text(_json.dumps(blob, indent=2), encoding='utf-8')
        print(f"[kv] Wrote retention sidecar: {ret_path}")
    except Exception as e:
        print(f"  [kv] skip retention sidecar: {e}")

    # Optional multimodal bundling under the root weights dir
    out_root = out_dir.parent
    bundle_manifest = {
        "text": {
            "onnx": str(onnx_path),
            "onnx_int8": str(onnx_q_path) if onnx_q_path.exists() else None,
            "onnx_int8_perop": str(onnx_q_perop_path) if onnx_q_perop_path.exists() else None,
            "executorch": str(et_path) if et_path.exists() else None,
            "summary": str(summary_path),
        },
        "vision": {},
        "image": {},
        "vqdec": {},
        "audio": {},
        "latents": {},
    }

    # Vision backbone (optional)
    if args.with_vision:
        try:
            from .autofetch_backbones import _export_vision_backbone  # type: ignore
            print("[Vision] Exporting compact vision backbone ...")
            bundle_manifest["vision"] = _export_vision_backbone(
                out_root,
                backend=(args.vision_backend or None),
                onnx_opset=int(args.opset),
                do_coreml=bool(args.vision_export_coreml),
                do_executorch=bool(args.vision_export_executorch),
            )
            if bool(args.vision_export_grounding):
                try:
                    # Export grounding heads into the same vision folder
                    print("[Vision] Exporting grounding heads ...")
                    import sys as _sys, subprocess as _sp
                    vis_dir = out_root / "vision"
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    cmd = [
                        _sys.executable, "-m", "omnicoder.export.onnx_export_grounding",
                        "--out", str(vis_dir),
                        "--d_model", str(int(args.vision_d_model)),
                        "--num_props", str(int(args.vision_num_props)),
                        "--tokens", str(int(args.vision_tokens)),
                    ]
                    if bool(args.vision_export_coreml):
                        cmd.append("--coreml")
                    if bool(args.vision_export_executorch):
                        cmd.append("--executorch")
                    rc = _sp.run(cmd, check=False).returncode
                    if rc == 0:
                        # Attach to manifest
                        import json as _json
                        gman = (vis_dir / "grounding_export.json")
                        if gman.exists():
                            bundle_manifest["vision"]["grounding"] = _json.loads(gman.read_text(encoding='utf-8'))
                except Exception as e:
                    print(f"  [warn] grounding export skipped: {e}")
            # Copy provider profile if provided
            try:
                if args.vision_provider_profile:
                    prof = Path(args.vision_provider_profile)
                    if prof.exists():
                        dst = out_root / "vision" / "provider_profile.json"
                        dst.write_text(prof.read_text(encoding='utf-8'), encoding='utf-8')
            except Exception:
                pass
        except Exception as e:
            print(f"  [warn] vision export skipped: {e}")

    # Stable Diffusion components (optional)
    if args.with_sd:
        try:
            from .autofetch_backbones import _export_sd  # type: ignore
            print("[Image] Exporting Stable Diffusion components ...")
            sd_hf = args.sd_model or None
            sd_local = args.sd_local_path or None
            bundle_manifest["image"] = _export_sd(
                out_root,
                hf_id=sd_hf,
                local_path=sd_local,
                do_onnx=True,
                do_coreml=bool(args.sd_export_coreml),
                do_executorch=bool(args.sd_export_executorch),
                onnx_opset=int(args.opset),
            )
            # Copy provider profile if provided
            try:
                if args.image_provider_profile:
                    prof = Path(args.image_provider_profile)
                    if prof.exists():
                        dst = out_root / "sd_export" / "onnx" / "provider_profile.json"
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_text(prof.read_text(encoding='utf-8'), encoding='utf-8')
            except Exception:
                pass
        except Exception as e:
            print(f"  [warn] SD export skipped: {e}")

    # Image VQ decoder (optional)
    # Auto-enable VQ decoder export when a codebook path is provided
    if (args.image_vq_codebook and not args.with_vqdec):
        args.with_vqdec = True  # type: ignore[attr-defined]
    if args.with_vqdec and args.image_vq_codebook:
        try:
            from .autofetch_backbones import _export_vqdec  # type: ignore
            print("[VQDec] Exporting image VQ decoder ...")
            bundle_manifest["vqdec"] = _export_vqdec(
                out_root,
                codebook_path=str(args.image_vq_codebook),
                do_onnx=True,
                do_coreml=bool(args.vqdec_export_coreml),
                do_executorch=bool(args.vqdec_export_executorch),
                hq=int(args.vqdec_hq),
                wq=int(args.vqdec_wq),
                onnx_opset=int(args.opset),
            )
            # Copy provider profile if provided
            try:
                if args.vqdec_provider_profile:
                    prof = Path(args.vqdec_provider_profile)
                    if prof.exists():
                        dst = out_root / "vqdec" / "provider_profile.json"
                        dst.write_text(prof.read_text(encoding='utf-8'), encoding='utf-8')
            except Exception:
                pass
        except Exception as e:
            print(f"  [warn] VQ decoder export skipped: {e}")

    # Continuous latent heads (optional)
    if args.with_latent_heads:
        try:
            from .onnx_export_latent_heads import main as _export_latent_heads  # type: ignore
            print("[Latents] Exporting continuous latent heads ...")
            lat_out = out_dir / "latent_heads.onnx"
            # Call the main entrypoint in-process
            import sys as _sys
            _argv = _sys.argv
            _sys.argv = [
                _argv[0],
                "--out", str(lat_out),
                "--mobile_preset", str(args.preset),
                "--opset", str(int(args.opset)),
            ]
            try:
                _export_latent_heads()
                bundle_manifest["latents"] = {"onnx": str(lat_out)}
            finally:
                _sys.argv = _argv
        except Exception as e:
            print(f"  [warn] Latent heads export skipped: {e}")

    # Piper TTS (optional download)
    if args.with_piper and args.piper_url:
        try:
            from .autofetch_backbones import _download_piper_model  # type: ignore
            print("[Piper] Downloading ONNX TTS model ...")
            p = _download_piper_model(out_root, args.piper_url)
            bundle_manifest["audio"] = {"piper_onnx": p}
        except Exception as e:
            print(f"  [warn] Piper download skipped: {e}")

    # Video reference (record-only)
    if args.video_model or args.video_local_path:
        try:
            vdir = out_root / "video"
            vdir.mkdir(parents=True, exist_ok=True)
            ref = {"hf_id": (args.video_model or None), "local_path": (args.video_local_path or None)}
            (vdir / "reference.json").write_text(json.dumps(ref, indent=2), encoding='utf-8')
            bundle_manifest.setdefault("video", ref)
        except Exception:
            pass

    # Write top-level manifest next to text/image/vision/vqdec folders
    try:
        top_manifest = out_root / "mobile_packager_manifest.json"
        top_manifest.write_text(json.dumps(bundle_manifest, indent=2))
        print(f"[manifest] {top_manifest}")
    except Exception as e:
        print(f"  [warn] failed to write top-level manifest: {e}")

    print("Done. Artifacts:")
    print(f" - ONNX decode-step: {onnx_path}")
    if onnx_q_path.exists():
        print(f" - ONNX int8 (dynamic): {onnx_q_path}")
    if onnx_q_perop_path.exists():
        print(f" - ONNX int8 (per-op preset): {onnx_q_perop_path}")
    if et_path.exists():
        print(f" - ExecuTorch program: {et_path}")
    if 'nnapi_quant_maps_path' in locals() and nnapi_quant_maps_path.exists():
        print(f" - NNAPI quant maps: {nnapi_quant_maps_path}")
    if 'nnapi_nodes_path' in locals() and nnapi_nodes_path.exists():
        print(f" - NNAPI per-node maps: {nnapi_nodes_path}")
    if 'coreml_quant_maps_path' in locals() and coreml_quant_maps_path.exists():
        print(f" - Core ML quant maps: {coreml_quant_maps_path}")
    if 'dml_quant_maps_path' in locals() and dml_quant_maps_path.exists():
        print(f" - DML quant maps: {dml_quant_maps_path}")
    print(f" - Summary: {summary_path}")
    print(" - Vision/Image/VQDec/Audio bundles (if requested) are summarized in mobile_packager_manifest.json at the weights root.")
    print("Next steps:")
    print(" - Android: load ONNX with ORT-mobile or ExecuTorch with NNAPI delegate; stream tokens with the KV-cache I/O.")
    print(" - iOS: convert decode-step to Core ML (ANE/GPU) or run with MLC-LLM; keep caches on-device between steps.")


if __name__ == "__main__":
    main()


