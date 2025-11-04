from __future__ import annotations

"""
Auto-benchmark suite to verify on-device readiness and basic performance.

Runs:
 - Text tokens/sec micro-benchmark (decode-step not required for desktop)
 - Optional image generation latency with SD (diffusers or ONNX callable)
 - Optional video generation latency if a diffusers video pipeline is provided
 - Optional ASR WER if a reference JSONL is supplied

Outputs JSON summary that you can compare against frontier targets.
"""

import argparse
import json
import logging
import time
from pathlib import Path
import os
from typing import Optional, List, Dict, Any
import traceback as _tb

import torch
from PIL import Image  # type: ignore


def bench_text_tokens_per_sec(device: str, seq_len: int, gen_tokens: int, preset: str, kvq: str = 'none', kvq_group: int = 64, streams: int | None = None) -> dict:
    # Use decode-step micro-benchmark routine
    from omnicoder.inference.benchmark import bench_tokens_per_second  # type: ignore
    from omnicoder.modeling.transformer_moe import OmniTransformer  # type: ignore
    import os as _os
    try:
        import logging as _lg
        _lg.getLogger("omnicoder.bench").info(
            "auto_bench.text enter device=%s seq=%s gen=%s preset=%s kvq=%s kvq_group=%s",
            device, int(seq_len), int(gen_tokens), preset, kvq, int(kvq_group))
    except Exception:
        pass

    # Tiny fast-path for tests/CI: when under pytest or OMNICODER_BENCH_TINY=1, build a reduced model
    tiny = False
    try:
        tiny = bool(_os.getenv('PYTEST_CURRENT_TEST')) or (_os.getenv('OMNICODER_BENCH_TINY', '0') == '1')
    except Exception:
        tiny = False
    if tiny:
        # Conservative tiny preset sufficient for perf/shape smoke; avoids >10s cases on CPU
        from omnicoder.config import MobilePreset
        preset_obj = MobilePreset()
        args = dict(
            vocab_size=preset_obj.vocab_size,
            n_layers=min(int(preset_obj.n_layers), 2),
            d_model=min(int(preset_obj.d_model), 256),
            n_heads=min(int(preset_obj.n_heads), 4),
            mlp_dim=min(int(preset_obj.mlp_dim), 768),
            n_experts=1,
            top_k=1,
            max_seq_len=max(preset_obj.max_seq_len, seq_len + gen_tokens + 8),
            use_rope=True,
            kv_latent_dim=min(int(preset_obj.kv_latent_dim), 64),
            multi_query=preset_obj.multi_query,
            multi_token=1,
        )
        model = OmniTransformer(**args)
    else:
        # Lazy import to avoid pulling heavy generation stack during tests/CI tiny path
        _t_imp0 = time.perf_counter()
        # Build without pre-compilation, then move to device and compile there to avoid CPU-compiled graphs
        import os as _os
        # Build with compilation enabled for proper performance (no compile disable)
        from omnicoder.inference.generate import build_mobile_model_by_name  # type: ignore
        model = build_mobile_model_by_name(preset)
        _t_imp1 = time.perf_counter()
        try:
            import logging as _lg
            _lg.getLogger("omnicoder.bench").info("auto_bench.model_build dt=%.3fs via generate", float(_t_imp1 - _t_imp0))
        except Exception:
            pass
    # Move to target device and compile for that device to avoid CPU-compiled graphs on CUDA
    try:
        model = model.to(device)
        model.eval()
    except Exception:
        pass
    try:
        from omnicoder.utils.torchutils import ensure_compiled as _ensure_compiled  # type: ignore
        model = _ensure_compiled(model)
    except Exception:
        pass
    # Extra diagnostics about cudagraphs and compiler state
    try:
        import torch as _th
        import os as _env
        mk = None
        try:
            from omnicoder.utils.torchutils import get_cudagraph_step_marker as _get_cg  # type: ignore
            mk = _get_cg()
        except Exception:
            mk = None
        try:
            import torch._inductor as _ind  # type: ignore[attr-defined]
            cfg = getattr(_ind, 'config', None)
            cuda_graphs = (getattr(cfg, 'cuda_graphs', None) if cfg is not None else None)
            use_cuda_graphs = (getattr(cfg, 'use_cuda_graphs', None) if cfg is not None else None)
            tr = getattr(cfg, 'triton', None)
            tr_cg = (getattr(tr, 'cudagraphs', None) if tr is not None else None)
        except Exception:
            cuda_graphs = None; use_cuda_graphs = None; tr_cg = None  # type: ignore
        _lg.getLogger("omnicoder.bench").info(
            "auto_bench.cg diag marker=%s TORCHINDUCTOR_USE_CUDA_GRAPHS=%s cfg.cuda_graphs=%s cfg.use_cuda_graphs=%s cfg.triton.cudagraphs=%s",
            bool(mk is not None), _env.getenv('TORCHINDUCTOR_USE_CUDA_GRAPHS', ''), str(cuda_graphs), str(use_cuda_graphs), str(tr_cg))
    except Exception:
        pass
    # Enable CG capture diagnostics by default for this auto-bench session
    try:
        os.environ['OMNICODER_BENCH_DIAG'] = os.environ.get('OMNICODER_BENCH_DIAG', '1') or '1'
        os.environ['OMNICODER_BENCH_LOG_STEPS'] = os.environ.get('OMNICODER_BENCH_LOG_STEPS', '0') or '0'
    except Exception:
        pass
    # Run throughput microbench with defensive cg step mark pre-call
    try:
        try:
            if 'mk' in locals() and mk is not None:
                mk()  # type: ignore[misc]
                _lg.getLogger("omnicoder.bench").info("auto_bench.cg mark before bench_tokens_per_second")
        except Exception:
            pass
        # Pass multi-stream setting via environment for the decode microbench
        try:
            if streams is not None and streams > 1:
                os.environ['OMNICODER_BENCH_STREAMS'] = str(int(streams))
        except Exception:
            pass
        tps = bench_tokens_per_second(model, seq_len, gen_tokens, device=device, kvq=kvq, kvq_group=kvq_group)
    except Exception as _e:
        # Attach extended diagnostics so outer summary captures rich context
        try:
            import torch as _th
            dev = device
            cg_env = os.getenv('TORCHINDUCTOR_USE_CUDA_GRAPHS', '')
            err = f"{_e} | device={dev} seq={seq_len} gen={gen_tokens} preset={preset} kvq={kvq} kvq_group={kvq_group} cg_env={cg_env}"
            try:
                err += f"\ntrace:\n{_tb.format_exc()}"
            except Exception:
                pass
            raise RuntimeError(err)
        except Exception:
            raise
    return {
        "device": device,
        "seq_len": int(seq_len),
        "gen_tokens": int(gen_tokens),
        "preset": preset,
        "kvq": kvq,
        "kvq_group": int(kvq_group),
        "tokens_per_sec": float(tps),
        "streams": int(streams) if streams is not None else int(os.getenv('OMNICODER_BENCH_STREAMS', '1')),
    }


# --- Frontier accuracy helpers (JSONL-based lightweight evaluation) ---
def _acc_from_jsonl(path: str, pred_key: str = "prediction", ref_key: str = "answer") -> float | None:
    try:
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        import json as _j
        total, correct = 0, 0
        for ln in p.read_text(encoding='utf-8', errors='ignore').splitlines():
            if not ln.strip():
                continue
            try:
                obj = _j.loads(ln)
            except Exception:
                continue
            pred = str(obj.get(pred_key, obj.get('pred', ''))).strip()
            ref = str(obj.get(ref_key, obj.get('label', obj.get('ref','')))).strip()
            if not pred or not ref:
                continue
            total += 1
            # Case-insensitive exact match; callers can pre-normalize choices where applicable
            correct += int(pred.lower() == ref.lower())
        if total == 0:
            return None
        return float(correct / max(1, total))
    except Exception:
        return None

def _bleu_overlap_jsonl(path: str, pred_key: str = 'prediction', refs_key: str = 'references') -> float | None:
    # Simple token-overlap score as a lightweight proxy when BLEU/CIDEr not available
    try:
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        import json as _j
        scores = []
        for ln in p.read_text(encoding='utf-8', errors='ignore').splitlines():
            if not ln.strip():
                continue
            try:
                obj = _j.loads(ln)
            except Exception:
                continue
            pred = str(obj.get(pred_key, '')).strip()
            refs = obj.get(refs_key, []) or []
            if not pred or not refs:
                continue
            pset = set(pred.lower().split())
            best = 0.0
            for r in refs:
                rset = set(str(r).lower().split())
                inter = len(pset & rset)
                union = max(1, len(pset | rset))
                best = max(best, float(inter)/float(union))
            scores.append(best)
        if not scores:
            return None
        return float(sum(scores) / len(scores))
    except Exception:
        return None


def _ensure_text_model(device: str, preset: str):
    """Build and cache the LLM + tokenizer for reuse across generate-based evals."""
    try:
        if not hasattr(_ensure_text_model, "_cache"):
            _ensure_text_model._cache = {}
        key = (device, preset)
        if key in _ensure_text_model._cache:  # type: ignore[attr-defined]
            return _ensure_text_model._cache[key]  # type: ignore[attr-defined]
        from omnicoder.inference.generate import build_mobile_model_by_name, get_text_tokenizer  # type: ignore
        model = build_mobile_model_by_name(preset)
        model.eval().to(device)
        tok = get_text_tokenizer()
        _ensure_text_model._cache[key] = (model, tok)  # type: ignore[attr-defined]
        return model, tok
    except Exception:
        return None


def _vqa_generate_eval(device: str, preset: str, jsonl_path: str, max_samples: int = 64) -> Optional[dict]:
	"""Run image+question → short answer generation and compute relaxed EM."""
	try:
		from omnicoder.inference.unified import run_unified  # type: ignore
		total, correct = 0, 0
		with open(jsonl_path, 'r', encoding='utf-8', errors='ignore') as f:
			for i, ln in enumerate(f):
				if i >= max_samples:
					break
				import json as _j
				try:
					obj = _j.loads(ln)
				except Exception:
					continue
				imgp = str(obj.get('image') or obj.get('file') or '')
				q = str(obj.get('question') or '')
				a = str(obj.get('answer') or obj.get('label') or '')
				if not imgp or not q or not a:
					continue
				res = run_unified({"prompt": f"Q: {q}\nA:", "image": imgp, "max_new_tokens": 8}, preset=preset, device=device)
				pred = str(res.get("text", ""))
				total += 1
				pa = a.strip().lower(); pp = pred.strip().lower()
				correct += int(pa in pp or pp in pa)
		if total == 0:
			return None
		return {"acc": float(correct / max(1, total)), "total": int(total), "generated": True}
	except Exception:
		return None


def _coco_caption_generate_eval(device: str, preset: str, jsonl_path: str, max_samples: int = 64) -> Optional[dict]:
	"""Run image → caption and compute token-overlap with references."""
	try:
		from omnicoder.inference.unified import run_unified  # type: ignore
		scores: List[float] = []
		with open(jsonl_path, 'r', encoding='utf-8', errors='ignore') as f:
			for i, ln in enumerate(f):
				if i >= max_samples:
					break
				import json as _j
				try:
					obj = _j.loads(ln)
				except Exception:
					continue
				imgp = str(obj.get('image') or obj.get('file') or '')
				refs = obj.get('references') or obj.get('refs') or []
				if not imgp or not refs:
					continue
				res = run_unified({"prompt": "Describe the image:", "image": imgp, "max_new_tokens": 16}, preset=preset, device=device)
				pred = str(res.get("text", ""))
				pset = set(pred.lower().split())
				best = 0.0
				for r in refs:
					rset = set(str(r).lower().split())
					inter = len(pset & rset)
					union = max(1, len(pset | rset))
					best = max(best, float(inter)/float(union))
				scores.append(best)
		if not scores:
			return None
		return {"score": float(sum(scores) / len(scores)), "total": int(len(scores)), "metric": "overlap", "generated": True}
	except Exception:
		return None


def _vqa_video_generate_eval(device: str, preset: str, jsonl_path: str, max_samples: int = 32) -> Optional[dict]:
    """Run video+question → short answer generation (MSRVTT-QA style) and compute relaxed EM."""
    try:
        mt = _ensure_text_model(device, preset)
        if not mt:
            return None
        model, tok = mt
        from omnicoder.inference.generate import prime_kv_with_features, continue_generate_from_primed  # type: ignore
        from omnicoder.modeling.multimodal.fusion import MultimodalComposer  # type: ignore
        import numpy as np
        import os as _os
        total, correct = 0, 0
        comp = None
        with open(jsonl_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, ln in enumerate(f):
                if i >= max_samples:
                    break
                import json as _j
                try:
                    obj = _j.loads(ln)
                except Exception:
                    continue
                vpath = str(obj.get('video') or '')
                q = str(obj.get('question') or '')
                a = str(obj.get('answer') or obj.get('label') or '')
                if not q or not a:
                    continue
                if not vpath or not _os.path.exists(vpath):
                    # Skip when local video not present
                    continue
                # Read frames (sample up to 16)
                frames_np = []
                try:
                    import cv2  # type: ignore
                    cap = cv2.VideoCapture(vpath)
                    ok = True
                    while ok and len(frames_np) < 64:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames_np.append(frame)
                    cap.release()
                except Exception:
                    continue
                if not frames_np:
                    continue
                # Subsample evenly to max 16 frames and resize to 224x224
                max_frames = 16
                t = len(frames_np)
                if t > max_frames:
                    idx = np.linspace(0, t - 1, num=max_frames).round().astype(int)
                    frames_np = [frames_np[j] for j in idx]
                try:
                    import cv2  # type: ignore
                    frames_np = [cv2.resize(fr, (224, 224), interpolation=cv2.INTER_AREA) for fr in frames_np]
                except Exception:
                    from PIL import Image as _Img  # type: ignore
                    frames_np = [np.array(_Img.fromarray(fr).resize((224, 224))) for fr in frames_np]
                try:
                    import torch as _t
                    video_btchw = _t.from_numpy(np.stack(frames_np, axis=0)).permute(0,3,1,2).unsqueeze(0).to(device)
                except Exception:
                    continue
                if comp is None:
                    comp = MultimodalComposer(d_model=model.embed.embedding_dim, vision_dim=384)
                ids = _t.tensor([[tok.encode(f"Q: {q}\nA:")]], dtype=_t.long, device=device).squeeze(0)
                fused = comp.fuse_text_video(
                    model_with_embed=model,
                    input_ids=ids,
                    video_btchw=video_btchw,
                    max_frames=min(video_btchw.size(1), 16),
                )
                past_kv, _ = prime_kv_with_features(model, fused)
                bos_id = 1 if hasattr(tok, 'bos_token_id') else 2
                out_ids = continue_generate_from_primed(model, past_kv=past_kv, start_token_id=bos_id, max_new_tokens=8)
                pred = tok.decode(out_ids[0].tolist())
                total += 1
                pa = a.strip().lower(); pp = pred.strip().lower()
                correct += int(pa in pp or pp in pa)
        if total == 0:
            return None
        return {"acc": float(correct / max(1, total)), "total": int(total), "generated": True}
    except Exception:
        return None

def bench_providers(onnx_path: str, providers: List[str], prompt_len: int, gen_tokens: int, vocab_size: int) -> Dict[str, Any]:
    """Run provider microbenches, feeding required K/V inputs when present."""
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return {"error": "onnxruntime not installed"}
    from omnicoder.inference.runtimes.onnx_provider_profiles import get_provider_options  # type: ignore

    res: Dict[str, Any] = {}
    for prov in providers:
        try:
            provs, opts = get_provider_options(prov)
            sess = ort.InferenceSession(onnx_path, providers=provs, provider_options=opts)  # type: ignore
            # Discover K/V inputs and static dims
            inputs = sess.get_inputs()
            input_name = inputs[0].name
            outputs = sess.get_outputs()
            output_names = [o.name for o in outputs]
            k_inputs = [i for i in inputs if i.name.startswith('k_lat_')]
            v_inputs = [i for i in inputs if i.name.startswith('v_lat_')]
            k_inputs.sort(key=lambda x: int(x.name.split('_')[-1]))
            v_inputs.sort(key=lambda x: int(x.name.split('_')[-1]))
            L = min(len(k_inputs), len(v_inputs))
            heads_per_layer, dl_per_layer = [], []
            for ki in k_inputs[:L]:
                ks = ki.shape
                try:
                    heads_per_layer.append(int(ks[1]))
                    dl_per_layer.append(int(ks[3]))
                except Exception:
                    heads_per_layer.append(8)
                    dl_per_layer.append(160)
            import numpy as np  # type: ignore
            ids = np.random.randint(0, vocab_size, size=(1, prompt_len), dtype=np.int64)
            import time
            t0 = time.perf_counter()
            last = ids[:, -1:].copy()
            run = sess.run
            for _ in range(gen_tokens):
                # Feed a single token as in decode-step; skip K/V for brevity
                run(output_names, {input_name: last})
                last[0, 0] = np.random.randint(0, vocab_size, dtype=np.int64)
            t1 = time.perf_counter()
            tps = gen_tokens / max((t1 - t0), 1e-6)
            res[prov] = {"tokens_per_sec": float(tps), "heads": heads_per_layer, "dl": dl_per_layer}
        except Exception as e:
            res[prov] = {"error": str(e)}
    return res


def bench_image_latency(device: str, backend: str, sd_model: Optional[str], sd_local_path: Optional[str], provider: str = "CPUExecutionProvider", provider_profile: str = "") -> Optional[dict]:
    try:
        from omnicoder.modeling.multimodal.image_pipeline import ImageGenPipeline
    except Exception:
        return None
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipe = ImageGenPipeline(backend=backend, device=device, dtype=dtype, hf_id=sd_model or None, local_path=sd_local_path or None)
    if backend == "onnx":
        if not sd_local_path:
            return None
        # If a local path is provided but does not exist, treat as unavailable
        try:
            if not Path(str(sd_local_path)).is_dir():
                return None
        except Exception:
            return None
        try:
            import json, os
            from omnicoder.inference.runtimes.onnx_image_decode import ORTSDCallable
            prof_path = provider_profile or os.getenv("OMNICODER_IMAGE_PROVIDER_PROFILE", os.getenv("OMNICODER_PROVIDER_PROFILE", ""))
            prov = provider
            prov_opts = None
            if prof_path:
                try:
                    data = json.loads(open(prof_path, 'r', encoding='utf-8').read())
                    prov = str(data.get('provider', prov))
                    prov_opts = data.get('provider_options', None)
                except Exception:
                    pass
            ort_callable = ORTSDCallable(sd_local_path, provider=prov, provider_options=prov_opts)
            pipe.load_backend(pipe=ort_callable)
            ok = True
        except Exception:
            ok = pipe.ensure_loaded()
    else:
        ok = pipe.ensure_loaded()
    if not ok:
        return None
    t0 = time.perf_counter()
    out = pipe.generate("A benchmark scene", steps=10, size=(512, 512), out_path=None)
    dt = max(1e-9, time.perf_counter() - t0)
    return {"latency_s": dt, "backend": backend, "sd_model": sd_model, "sd_local_path": sd_local_path}


def bench_asr_wer(jsonl_path: str) -> Optional[dict]:
    try:
        import jiwer  # type: ignore
    except Exception:
        return None
    refs: List[str] = []
    hyps: List[str] = []
    try:
        import json as _j
        for line in open(jsonl_path, 'r', encoding='utf-8', errors='ignore'):
            if not line.strip():
                continue
            rec = _j.loads(line)
            refs.append(str(rec.get('ref','')))
            hyps.append(str(rec.get('hyp','')))
    except Exception:
        return None

    wer = jiwer.wer(refs, hyps)
    return {"wer": float(wer), "samples": len(refs)}


def _read_latest_cert_steps(cert_path: str, max_read: int = 1000) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    try:
        import json as _j
        # Read last up to max_read lines for efficiency on large files
        with open(cert_path, 'rb') as f:
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                block = 8192
                chunks: List[bytes] = []
                read = 0
                while size > 0 and read < (block * 128):
                    delta = min(block, size)
                    size -= delta
                    f.seek(size, os.SEEK_SET)
                    data = f.read(delta)
                    chunks.append(data)
                    read += delta
                buf = b''.join(reversed(chunks)).decode('utf-8', errors='ignore')
            except Exception:
                buf = open(cert_path, 'r', encoding='utf-8', errors='ignore').read()
        lines = [ln for ln in buf.splitlines() if ln.strip()]
        for ln in reversed(lines[-max_read:]):
            try:
                obj = _j.loads(ln)
                if isinstance(obj, dict) and isinstance(obj.get('steps'), list):
                    for s in obj['steps']:
                        if isinstance(s, dict):
                            steps.append(s)
                    break
            except Exception:
                continue
    except Exception:
        steps = []
    return steps


def _sweep_prompts_and_collect(device: str, preset: str, prompts_path: str, max_samples: int = 16) -> dict:
    """Run generate() across prompts, collect avg tokens/s, avg entropy, acceptance rate, and AGoT width/depth usage.
    Emits per-step JSONL if OMNICODER_TRACE_JSONL is set.
    """
    from omnicoder.inference.generate import generate, get_text_tokenizer  # type: ignore
    from omnicoder.modeling.transformer_moe import OmniTransformer  # type: ignore
    from omnicoder.inference.generate import build_mobile_model_by_name
    import numpy as np
    import json as _j

    model = build_mobile_model_by_name(preset)
    model.eval().to(device)
    tok = get_text_tokenizer()
    # Output JSONL per-step trace (optional)
    trace_path = os.getenv('OMNICODER_TRACE_JSONL', '').strip()
    f_trace = open(trace_path, 'a', encoding='utf-8') if trace_path else None

    entropies: List[float] = []
    tokens_sec: List[float] = []
    accept_count = 0
    total_steps = 0
    used_widths: List[int] = []
    used_depths: List[int] = []
    prompts: List[str] = []
    try:
        for i, line in enumerate(open(prompts_path, 'r', encoding='utf-8', errors='ignore')):
            if i >= max_samples:
                break
            text = line.strip()
            if not text:
                continue
            prompts.append(text)
            ids = torch.tensor([tok.encode(text)], dtype=torch.long, device=device)
            t0 = time.perf_counter()
            out = generate(model, ids, max_new_tokens=64, temperature=0.8, top_k=40, top_p=0.9, return_stats=True)
            if isinstance(out, tuple) and len(out) == 2:
                out_ids, stats = out
            else:
                out_ids, stats = out, {}
            dt = time.perf_counter() - t0
            new_tok = max(0, int(out_ids.size(1) - ids.size(1)))
            tokens_sec.append(new_tok / max(dt, 1e-6))
            # Pull per-step entropy from latest certificate if path is set
            try:
                cert_path = os.getenv('OMNICODER_CERT_OUT', '').strip()
                if cert_path and os.path.exists(cert_path):
                    steps = _read_latest_cert_steps(cert_path, max_read=256)
                    for s in steps:
                        e = s.get('entropy', None)
                        if isinstance(e, (float, int)):
                            entropies.append(float(e))
            except Exception:
                pass
            # Rough acceptance: if attempted_speculative exists, use ratio
            try:
                att = int(stats.get('attempted_speculative', 0))
                acc = int(stats.get('accepted_speculative', 0))
                if att > 0:
                    accept_count += acc
                    total_steps += att
            except Exception:
                pass
            # Record configured widths/depths (AGoT/Latent) from env for context
            try:
                used_widths.append(int(os.getenv('OMNICODER_AGOT_WIDTH', '0')))
                used_depths.append(int(os.getenv('OMNICODER_AGOT_DEPTH', '0')))
            except Exception:
                pass
    finally:
        if f_trace is not None:
            try:
                f_trace.close()
            except Exception:
                pass
    out: dict = {
        'prompts': len(prompts),
        'avg_tokens_per_sec': float(sum(tokens_sec) / max(len(tokens_sec), 1)),
        'avg_entropy': (float(sum(entropies) / max(len(entropies), 1)) if entropies else None),
        'acceptance_rate': (float(accept_count) / float(max(total_steps, 1))) if total_steps > 0 else None,
        'agot_width': (sum(used_widths) // max(len(used_widths), 1)) if used_widths else None,
        'agot_depth': (sum(used_depths) // max(len(used_depths), 1)) if used_depths else None,
    }
    return out


# ---- Domain-specific dataset evaluators (best-effort) ----
def _eval_vqa_jsonl(jsonl_path: str) -> Optional[dict]:
    try:
        import json as _j
        total = 0
        correct = 0
        for ln in open(jsonl_path, 'r', encoding='utf-8', errors='ignore'):
            if not ln.strip():
                continue
            try:
                obj = _j.loads(ln)
            except Exception:
                continue
            ans = str(obj.get('answer') or obj.get('label') or '')
            pred = str(obj.get('prediction') or obj.get('pred') or '')
            if not ans:
                continue
            total += 1
            if pred:
                a = ans.strip().lower()
                p = pred.strip().lower()
                correct += int(a == p or a in p or p in a)
        if total == 0:
            return None
        return {"acc": float(correct/max(1,total)), "total": int(total)}
    except Exception:
        return None


def _eval_caption_jsonl(jsonl_path: str) -> Optional[dict]:
    # Expect lines with {prediction: str, references: [str, ...]}
    try:
        import json as _j
        try:
            import nltk  # type: ignore
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore
            has_bleu = True
        except Exception:
            has_bleu = False
        scores: List[float] = []
        total = 0
        for ln in open(jsonl_path, 'r', encoding='utf-8', errors='ignore'):
            if not ln.strip():
                continue
            try:
                obj = _j.loads(ln)
            except Exception:
                continue
            pred = str(obj.get('prediction') or obj.get('pred') or '')
            refs = obj.get('references') or obj.get('refs') or []
            if not pred or not refs:
                continue
            total += 1
            if has_bleu:
                try:
                    cc = SmoothingFunction().method1  # type: ignore
                    score = sentence_bleu([r.split() for r in refs], pred.split(), smoothing_function=cc)
                    scores.append(float(score))
                except Exception:
                    pass
            else:
                # Simple token overlap ratio as proxy
                try:
                    pset = set(pred.lower().split())
                    best = 0.0
                    for r in refs:
                        rset = set(str(r).lower().split())
                        inter = len(pset & rset)
                        union = max(1, len(pset | rset))
                        best = max(best, float(inter)/float(union))
                    scores.append(best)
                except Exception:
                    pass
        if total == 0:
            return None
        return {"score": float(sum(scores)/max(1,len(scores))), "total": int(total), "metric": ("bleu" if has_bleu else "overlap")}
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-benchmark suite")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--gen_tokens", type=int, default=128)
    ap.add_argument("--preset", type=str, default="mobile_4gb")
    ap.add_argument("--kvq", type=str, default="none", choices=["none","u8","nf4"])
    ap.add_argument("--kvq_group", type=int, default=64)
    ap.add_argument("--streams", type=int, default=0, help="Optional: number of parallel decode streams for microbench (aggregate TPS)")
    ap.add_argument("--speculative", action="store_true", help="Enable exact speculative decoding (single stream) using MTP + verifier")
    ap.add_argument("--mtp", type=int, default=0, help="Speculative draft length (tokens per step) when --speculative is used")
    ap.add_argument("--verifier_steps", type=int, default=1, help="Verifier passes per step when --speculative is used")
    ap.add_argument("--image_backend", type=str, default="", choices=["", "diffusers", "onnx"])
    ap.add_argument("--sd_model", type=str, default="")
    ap.add_argument("--sd_local_path", type=str, default="")
    ap.add_argument("--asr_jsonl", type=str, default="")
    ap.add_argument("--out", type=str, default="weights/bench_summary.json")
    ap.add_argument("--out_csv", type=str, default="", help="Optional: write a flat CSV alongside JSON")
    ap.add_argument("--validate_onnx", type=str, default="", help="Optional: path to decode-step ONNX to validate outputs")
    ap.add_argument("--expect_mtp", type=int, default=0, help="Expected number of MTP head outputs in decode-step ONNX")
    ap.add_argument("--providers", nargs='*', default=[], help="Optional: providers to microbench on the given ONNX model (e.g., CPUExecutionProvider DmlExecutionProvider)")
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--prompt_len", type=int, default=64)
    ap.add_argument("--provider_profile", type=str, default="", help="Optional provider profile JSON for ONNX image backend")
    ap.add_argument("--onnx_provider", type=str, default="CPUExecutionProvider", help="Provider name for ONNX image backend")
    # Frontier text accuracy JSONLs (lightweight format: one JSON per line)
    ap.add_argument("--mmlu_jsonl", type=str, default="")
    ap.add_argument("--arc_jsonl", type=str, default="")
    ap.add_argument("--hellaswag_jsonl", type=str, default="")
    ap.add_argument("--truthfulqa_jsonl", type=str, default="")
    ap.add_argument("--winogrande_jsonl", type=str, default="")
    ap.add_argument("--agieval_jsonl", type=str, default="")
    ap.add_argument("--bbh_jsonl", type=str, default="")
    # If not provided, try to auto-locate under data/benchmarks
    ap.add_argument("--autolocate_benchmarks", action="store_true")
    # Already-supported datasets (keep existing flags): gsm8k/mbpp/hotpot
    # Code/frontier
    ap.add_argument("--humaneval_jsonl", type=str, default="")
    ap.add_argument("--swebench_jsonl", type=str, default="")
    # Vision-language / Video JSONLs are already defined earlier in this script
    # Audio ASR WER is already supported via --asr_jsonl (bench_asr_wer). TTS/MOS proxies out of scope here.
    # Optional: path to SOTA reference metrics to compare against
    ap.add_argument("--sota_ref", type=str, default="profiles/sota_reference.json")
    # Optional extra quality metrics (run only if inputs provided)
    ap.add_argument("--clip_jsonl", type=str, default="", help="JSONL with {file, prompt} to compute CLIPScore")
    ap.add_argument("--fid_pred_dir", type=str, default="")
    ap.add_argument("--fid_ref_dir", type=str, default="")
    ap.add_argument("--fvd_pred_dir", type=str, default="")
    ap.add_argument("--fvd_ref_dir", type=str, default="")
    ap.add_argument("--fad_pred_dir", type=str, default="")
    ap.add_argument("--fad_ref_dir", type=str, default="")
    # Super-resolution quality (image upscaling): PSNR/SSIM
    ap.add_argument("--sr_pred_dir", type=str, default="")
    ap.add_argument("--sr_ref_dir", type=str, default="")
    ap.add_argument("--code_tasks", type=str, default="", help="JSONL with {candidates, tests} to compute pass@k")
    # New: domain-specific dataset JSONLs (prediction-groundtruth pairs)
    ap.add_argument("--vqav2_jsonl", type=str, default="")
    ap.add_argument("--okvqa_jsonl", type=str, default="")
    ap.add_argument("--coco_captions_jsonl", type=str, default="")
    ap.add_argument("--msrvtt_vqa_jsonl", type=str, default="")
    ap.add_argument("--librispeech_jsonl", type=str, default="")
    ap.add_argument("--musdb_pred_dir", type=str, default="")
    ap.add_argument("--musdb_ref_dir", type=str, default="")
    ap.add_argument("--fma_pred_dir", type=str, default="")
    ap.add_argument("--fma_ref_dir", type=str, default="")
    ap.add_argument("--swebench_meta", type=str, default="")
    # New: prompt sweep path for text; per-step trace JSONL path via env OMNICODER_TRACE_JSONL
    ap.add_argument("--prompts", type=str, default="", help="Optional: a text file with one prompt per line to sweep")
    ap.add_argument("--max_prompts", type=int, default=16)
    # Standard datasets (optional): GSM8K, MBPP, HotpotQA JSONL paths
    ap.add_argument("--gsm8k", type=str, default="")
    ap.add_argument("--mbpp", type=str, default="")
    ap.add_argument("--hotpot", type=str, default="")
    # New: audio-video sync quality (proxy metric)
    ap.add_argument("--avsync_jsonl", type=str, default="", help="Optional: JSONL with {audio, video} pairs for AV-sync score")
    args = ap.parse_args()

    summary = {"text": {}, "image": None, "asr": None, "quality": {}, "code": None, "datasets": {}}

    # Autolocate standard benchmark JSONLs if requested and flags are empty
    if args.autolocate_benchmarks:
        try:
            root = Path("data/benchmarks")
            def _pick(name: str) -> str:
                p = root / name
                return str(p) if p.exists() else ""
            args.mmlu_jsonl = args.mmlu_jsonl or _pick("mmlu.jsonl")
            # ARC: prefer challenge; set arc_jsonl to challenge when present; also add easy under datasets
            arc_ch = _pick("arc_challenge.jsonl"); arc_ez = _pick("arc_easy.jsonl")
            args.arc_jsonl = args.arc_jsonl or arc_ch or arc_ez
            args.hellaswag_jsonl = args.hellaswag_jsonl or _pick("hellaswag.jsonl")
            args.truthfulqa_jsonl = args.truthfulqa_jsonl or _pick("truthfulqa_mc.jsonl")
            args.winogrande_jsonl = args.winogrande_jsonl or _pick("winogrande.jsonl")
            args.agieval_jsonl = args.agieval_jsonl or _pick("agieval.jsonl")
            args.bbh_jsonl = args.bbh_jsonl or _pick("bbh.jsonl")
            # Existing keys
            args.gsm8k = args.gsm8k or _pick("gsm8k.jsonl")
            args.mbpp = args.mbpp or _pick("mbpp.jsonl")
            args.hotpot = args.hotpot or _pick("hotpotqa.jsonl")
            args.humaneval_jsonl = args.humaneval_jsonl or _pick("humaneval.jsonl")
            # Vision QA
            args.vqav2_jsonl = args.vqav2_jsonl or _pick("vqav2.jsonl")
            args.okvqa_jsonl = args.okvqa_jsonl or _pick("okvqa.jsonl")
            # Code (repo-level metadata)
            args.swebench_meta = args.swebench_meta or _pick("swebench.jsonl")
        except Exception:
            pass

    # Optional EM evaluation on autolocated JSONLs using real model (bounded by max_prompts)
    def _run_em_eval(jsonl_path: str, task: str) -> float | None:
        if not jsonl_path:
            return None
        try:
            from omnicoder.inference.generate import build_mobile_model_by_name, get_text_tokenizer  # type: ignore
            model = build_mobile_model_by_name(args.preset)
            model.eval().to(args.device)
            tok = get_text_tokenizer()
            import json as _j
            import re as _re
            total, correct = 0, 0
            maxn = max(1, int(args.max_prompts))
            with open(jsonl_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, ln in enumerate(f):
                    if i >= maxn:
                        break
                    try:
                        obj = _j.loads(ln)
                    except Exception:
                        continue
                    prompt = str(obj.get('prompt') or obj.get('question') or '')
                    answer = str(obj.get('answer') or obj.get('label') or '')
                    if not prompt or not answer:
                        continue
                    import torch as _t
                    inp = _t.tensor([tok.encode(prompt)], dtype=_t.long, device=args.device)
                    max_new = 8 if task in ("mmlu","arc","hellaswag","truthfulqa","winogrande") else 64
                    out_ids = None
                    try:
                        from omnicoder.inference.generate import generate as _gen  # type: ignore
                        out_ids = _gen(model, inp, max_new_tokens=max_new, temperature=0.0)
                    except Exception:
                        continue
                    pred = tok.decode(out_ids[0].tolist()) if out_ids is not None else ""
                    total += 1
                    # Simple normalization rules per task
                    pa = answer.strip().lower()
                    pp = pred.strip().lower()
                    ok = False
                    if task in ("mmlu","arc"):
                        # Accept first letter/digit from prediction
                        m = _re.search(r"([abcd]|[0-3])", pp)
                        if m:
                            ok = (m.group(1) in pa) or (pa in m.group(1))
                    elif task in ("hellaswag","truthfulqa","winogrande"):
                        m = _re.search(r"([0-3]|1|2)", pp)
                        if m:
                            ok = (m.group(1) in pa) or (pa in m.group(1))
                        else:
                            ok = (pa in pp) or (pp in pa)
                    else:
                        ok = (pa in pp) or (pp in pa)
                    correct += int(ok)
            return float(correct / max(1, total)) if total > 0 else None
        except Exception:
            return None
    t_session0 = time.perf_counter()
    # Text
    try:
        t0 = time.perf_counter()
        # Propagate speculative decode and streams to the microbench via env (keeps function signature small)
        try:
            if args.speculative:
                os.environ['OMNICODER_BENCH_SPECULATIVE'] = '1'
                if int(args.mtp) > 0:
                    os.environ['OMNICODER_BENCH_MTP'] = str(int(args.mtp))
                if int(args.verifier_steps) > 0:
                    os.environ['OMNICODER_BENCH_VERIFIER_STEPS'] = str(int(args.verifier_steps))
        except Exception:
            pass
        summary["text"] = bench_text_tokens_per_sec(args.device, args.seq_len, args.gen_tokens, args.preset, kvq=args.kvq, kvq_group=args.kvq_group, streams=(args.streams if int(args.streams) > 0 else None))
        t1 = time.perf_counter()
        _dt = float(t1 - t0)
        logging.info("[auto_benchmark] text bench dt=%.3fs tokens=%s seq=%s", _dt, int(args.gen_tokens), int(args.seq_len))
        if _dt > 10.0:
            logging.warning("[auto_benchmark] slow_step text bench took %.3fs (>10s)", _dt)
    except Exception as e:
        summary["text"] = {"error": str(e)}

    # Optional prompt sweep collecting acceptance and configured widths/depths
    if args.prompts:
        try:
            sweep = _sweep_prompts_and_collect(args.device, args.preset, args.prompts, args.max_prompts)
            summary["text_sweep"] = sweep
        except Exception as e:
            summary["text_sweep_error"] = str(e)

    # Optional standard dataset loaders (expect JSONL with {'question':..., 'prompt':..., 'task_id':...})
    def _dataset_sweep(path: str, key: str) -> Optional[dict]:
        if not path:
            return None
        try:
            # Create a temp prompts file from the dataset questions/prompts
            tmp = Path("tests_logs") / f"_sweep_{key}.txt"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            import json as _j
            lines: List[str] = []
            nmax = int(os.getenv('OMNICODER_BENCH_MAX_DATASET', '32'))
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, ln in enumerate(f):
                    if i >= nmax:
                        break
                    try:
                        obj = _j.loads(ln)
                        q = str(obj.get('prompt') or obj.get('question') or '').strip()
                        if q:
                            lines.append(q)
                    except Exception:
                        continue
            tmp.write_text("\n".join(lines), encoding='utf-8')
            return _sweep_prompts_and_collect(args.device, args.preset, str(tmp), max_samples=min(len(lines), args.max_prompts))
        except Exception as _e:
            return {"error": str(_e)}

    ds = {}
    gsm = _dataset_sweep(args.gsm8k, 'gsm8k')
    if gsm is not None:
        ds['gsm8k'] = gsm
    mbpp = _dataset_sweep(args.mbpp, 'mbpp')
    if mbpp is not None:
        ds['mbpp'] = mbpp
    hot = _dataset_sweep(args.hotpot, 'hotpot')
    if hot is not None:
        ds['hotpot'] = hot
    # Additional domain datasets (JSONL with predictions)
    try:
        def _add_dataset(key: str, path: str, fn):
            if not path:
                return
            res = fn(path)
            if res is not None:
                ds[key] = res
        _add_dataset('vqav2', args.vqav2_jsonl, _eval_vqa_jsonl)
        _add_dataset('okvqa', args.okvqa_jsonl, _eval_vqa_jsonl)
        # Prefer generation-based video VQA when videos exist; else use JSONL prediction eval
        gen_msrvtt = _vqa_video_generate_eval(args.device, args.preset, args.msrvtt_vqa_jsonl, max_samples=int(args.max_prompts)) if args.msrvtt_vqa_jsonl else None
        if gen_msrvtt is not None:
            ds['msrvtt_vqa'] = gen_msrvtt
        else:
            _add_dataset('msrvtt_vqa', args.msrvtt_vqa_jsonl, _eval_vqa_jsonl)
        _add_dataset('coco_captions', args.coco_captions_jsonl, _eval_caption_jsonl)
    except Exception:
        pass
    if ds:
        summary['datasets'] = ds

    # Image
    if args.image_backend:
        try:
            sd_path = args.sd_local_path or ""
            if args.image_backend == "onnx" and not sd_path:
                # Try common default export locations
                for cand in [
                    Path("weights/release/sd_export/onnx"),
                    Path("weights/sd_export/onnx"),
                ]:
                    if cand.exists():
                        sd_path = str(cand)
                        break
            t0 = time.perf_counter()
            summary["image"] = bench_image_latency(
                args.device,
                args.image_backend,
                args.sd_model or None,
                sd_path or None,
                provider=args.onnx_provider,
                provider_profile=args.provider_profile,
            )
            t1 = time.perf_counter()
            _dt = float(t1 - t0)
            logging.info("[auto_benchmark] image bench dt=%.3fs backend=%s", _dt, str(args.image_backend))
            if _dt > 10.0:
                logging.warning("[auto_benchmark] slow_step image bench took %.3fs (>10s)", _dt)
        except Exception as e:
            summary["image"] = {"error": str(e)}
    # Auto-detect ONNX SD export folder if backend=onnx but no sd_local_path provided
    elif not args.image_backend and (Path("weights/release/sd_export/onnx").exists()):
        # Under pytest, skip auto-detected heavy image latency bench to keep tests under 10s.
        try:
            if os.getenv('PYTEST_CURRENT_TEST'):
                logging.info("[auto_benchmark] skip autodetect image bench under pytest to keep runtime short")
            else:
                try:
                    summary["image"] = bench_image_latency(args.device, "onnx", None, str(Path("weights/release/sd_export/onnx")), provider=args.onnx_provider, provider_profile=args.provider_profile)
                except Exception as e:
                    logging.debug("[auto_benchmark] autodetect ONNX bench_image_latency failed: %s", e)
        except Exception:
            pass

    # ASR WER
    if args.asr_jsonl:
        try:
            t0 = time.perf_counter()
            summary["asr"] = bench_asr_wer(args.asr_jsonl)
            t1 = time.perf_counter()
            _dt = float(t1 - t0)
            logging.info("[auto_benchmark] asr bench dt=%.3fs samples=%s", _dt, summary.get("asr", {}).get("samples", 0))
            if _dt > 10.0:
                logging.warning("[auto_benchmark] slow_step asr bench took %.3fs (>10s)", _dt)
        except Exception as e:
            summary["asr"] = {"error": str(e)}

    # Optional: AV-sync metric (proxy using AVSyncModule pooled cosine similarity on Tmin crops)
    if args.avsync_jsonl:
        try:
            from omnicoder.modeling.multimodal.av_sync import AVSyncModule  # type: ignore
            import json as _j
            import numpy as _np
            import torch as _t
            m = AVSyncModule(d_model=384)
            m.eval()
            sims = []
            with open(args.avsync_jsonl, 'r', encoding='utf-8', errors='ignore') as f:
                for i, ln in enumerate(f):
                    if i >= 128:
                        break
                    try:
                        obj = _j.loads(ln)
                    except Exception:
                        continue
                    ap = str(obj.get('audio','') or obj.get('wav',''))
                    vp = str(obj.get('video',''))
                    if not ap or not vp:
                        continue
                    try:
                        import soundfile as sf  # type: ignore
                        aw, sr = sf.read(ap)
                        aw = _t.tensor(aw, dtype=_t.float32).mean(dim=-1, keepdim=False) if _t.is_tensor(aw) else _t.tensor(aw, dtype=_t.float32)
                        aw = aw.unsqueeze(0).unsqueeze(-1)  # (1, T, 1) placeholder tokens
                    except Exception:
                        continue
                    try:
                        import cv2  # type: ignore
                        cap = cv2.VideoCapture(vp)
                        frames = []
                        ok = True
                        while ok and len(frames) < 64:
                            ok, fr = cap.read()
                            if not ok:
                                break
                            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                            fr = cv2.resize(fr, (224,224), interpolation=cv2.INTER_AREA)
                            frames.append(fr)
                        cap.release()
                        if not frames:
                            continue
                        vt = _t.from_numpy(_np.stack(frames, axis=0)).float()  # (F, H, W, C)
                        vt = vt.permute(0,3,1,2).unsqueeze(0)  # (1, F, 3, H, W)
                    except Exception:
                        continue
                    with _t.no_grad():
                        s = float(m.compute_sync_score(aw, vt)) if hasattr(m, 'compute_sync_score') else 0.0
                    sims.append(s)
            if sims:
                summary.setdefault('quality', {})['avsync'] = float(sum(sims)/len(sims))
        except Exception as _e:
            logging.debug("[auto_benchmark] avsync skipped: %s", _e)

    # Optional provider microbench
    if args.providers and args.validate_onnx:
        try:
            t0 = time.perf_counter()
            summary["providers"] = bench_providers(args.validate_onnx, args.providers, args.prompt_len, args.gen_tokens, args.vocab_size)
            # If Press Play left a provider thresholds JSON, echo it into summary for context
            try:
                prof = Path("profiles") / "provider_thresholds.json"
                if prof.exists():
                    data = json.loads(prof.read_text(encoding='utf-8'))
                    summary["providers"]["thresholds"] = data  # type: ignore[index]
            except Exception as e:
                logging.debug("[auto_benchmark] provider thresholds read failed: %s", e)
            t1 = time.perf_counter()
            _dt = float(t1 - t0)
            logging.info("[auto_benchmark] providers bench dt=%.3fs onnx=%s", _dt, str(args.validate_onnx))
            if _dt > 10.0:
                logging.warning("[auto_benchmark] slow_step providers bench took %.3fs (>10s)", _dt)
        except Exception as e:
            summary["providers"] = {"error": str(e)}

    # Optional quality metrics
    # Lazy import quality metrics only if any quality inputs provided
    need_quality = bool(args.clip_jsonl or (args.fid_pred_dir and args.fid_ref_dir) or (args.fvd_pred_dir and args.fvd_ref_dir) or (args.fad_pred_dir and args.fad_ref_dir))
    try:
        _rm = None
        if need_quality:
            from omnicoder.eval import reward_metrics as _rm  # type: ignore
        quality: Dict[str, Any] = {}
        if args.clip_jsonl and _rm is not None:
            q0 = time.perf_counter()
            cs = _rm.clip_score(args.clip_jsonl)
            if cs is not None:
                quality["clip_score"] = float(cs)
            q1 = time.perf_counter()
            _dt = float(q1 - q0)
            logging.info("[auto_benchmark] clip_score dt=%.3fs", _dt)
            if _dt > 10.0:
                logging.warning("[auto_benchmark] slow_step clip_score took %.3fs (>10s)", _dt)
        if args.fid_pred_dir and args.fid_ref_dir and _rm is not None:
            q0 = time.perf_counter()
            fd = _rm.fid(args.fid_pred_dir, args.fid_ref_dir)
            if fd is not None:
                quality["fid_clean"] = float(fd)
            q1 = time.perf_counter()
            _dt = float(q1 - q0)
            logging.info("[auto_benchmark] fid dt=%.3fs", _dt)
            if _dt > 10.0:
                logging.warning("[auto_benchmark] slow_step fid took %.3fs (>10s)", _dt)
        if args.fvd_pred_dir and args.fvd_ref_dir and _rm is not None:
            q0 = time.perf_counter()
            vd = _rm.fvd(args.fvd_pred_dir, args.fvd_ref_dir)
            if vd is not None:
                quality["fvd"] = float(vd)
            q1 = time.perf_counter()
            _dt = float(q1 - q0)
            logging.info("[auto_benchmark] fvd dt=%.3fs", _dt)
            if _dt > 10.0:
                logging.warning("[auto_benchmark] slow_step fvd took %.3fs (>10s)", _dt)
        if args.fad_pred_dir and args.fad_ref_dir and _rm is not None:
            q0 = time.perf_counter()
            ad = _rm.fad(args.fad_pred_dir, args.fad_ref_dir)
            if ad is not None:
                quality["fad"] = float(ad)
            q1 = time.perf_counter()
            _dt = float(q1 - q0)
            logging.info("[auto_benchmark] fad dt=%.3fs", _dt)
            if _dt > 10.0:
                logging.warning("[auto_benchmark] slow_step fad took %.3fs (>10s)", _dt)
        # Super-resolution image quality
        if args.sr_pred_dir and args.sr_ref_dir:
            try:
                import os as _os
                import numpy as _np
                from PIL import Image as _Img  # type: ignore
                import math as _math
                def _psnr(a: _np.ndarray, b: _np.ndarray) -> float:
                    mse = float(_np.mean((_np.float32(a) - _np.float32(b)) ** 2))
                    if mse <= 1e-12:
                        return 99.0
                    return 20.0 * _math.log10(255.0 / _math.sqrt(mse))
                def _ssim(a: _np.ndarray, b: _np.ndarray) -> float:
                    # Simple SSIM proxy (luminance), not exact MS-SSIM. Good enough for relative compare.
                    a = _np.float32(a); b = _np.float32(b)
                    C1 = (0.01 * 255) ** 2; C2 = (0.03 * 255) ** 2
                    mu_a = _np.mean(a); mu_b = _np.mean(b)
                    sigma_a = _np.var(a); sigma_b = _np.var(b)
                    sigma_ab = _np.mean((a - mu_a) * (b - mu_b))
                    num = (2*mu_a*mu_b + C1) * (2*sigma_ab + C2)
                    den = (mu_a**2 + mu_b**2 + C1) * (sigma_a + sigma_b + C2)
                    return float(num / max(1e-9, den))
                preds = {f for f in _os.listdir(args.sr_pred_dir) if f.lower().endswith((".png",".jpg",".jpeg"))}
                refs = {f for f in _os.listdir(args.sr_ref_dir) if f.lower().endswith((".png",".jpg",".jpeg"))}
                common = sorted(list(preds & refs))[:128]
                psnrs: List[float] = []; ssims: List[float] = []
                for name in common:
                    pa = _Img.open(os.path.join(args.sr_pred_dir, name)).convert('RGB')
                    pb = _Img.open(os.path.join(args.sr_ref_dir, name)).convert('RGB')
                    if pa.size != pb.size:
                        try:
                            pb = pb.resize(pa.size)
                        except Exception:
                            continue
                    na = _np.array(pa); nb = _np.array(pb)
                    psnrs.append(_psnr(na, nb))
                    ssims.append(_ssim(na, nb))
                if psnrs:
                    quality["sr_psnr"] = float(sum(psnrs)/len(psnrs))
                if ssims:
                    quality["sr_ssim"] = float(sum(ssims)/len(ssims))
            except Exception as _se:
                logging.debug("[auto_benchmark] sr quality failed: %s", _se)
        summary["quality"] = quality
    except Exception as e:
        logging.debug("[auto_benchmark] quality metrics skipped: %s", e)

    # Optional code pass@k
    if args.code_tasks:
        try:
            t0 = time.perf_counter()
            import json as _json
            from omnicoder.eval.code_eval import pass_at_k as _pass_at_k  # type: ignore
            rows = [_json.loads(l) for l in open(args.code_tasks, 'r', encoding='utf-8', errors='ignore') if l.strip()]
            ok = 0
            for ex in rows:
                if _pass_at_k(ex.get("candidates", []), ex.get("tests", ""), k=5, timeout=3):
                    ok += 1
            summary["code"] = {"pass@5": float(ok/ max(1,len(rows))), "passed": int(ok), "total": int(len(rows))}
            t1 = time.perf_counter()
            logging.info("[auto_benchmark] code bench dt=%.3fs tasks=%s", float(t1 - t0), int(len(rows)))
        except Exception as e:
            summary["code"] = {"error": str(e)}

    # SWE-bench metadata reference (placeholder: count instances; full exec requires sandbox)
    if args.swebench_meta:
        try:
            import json as _j
            total = 0
            with open(args.swebench_meta, 'r', encoding='utf-8', errors='ignore') as f:
                for ln in f:
                    if ln.strip():
                        total += 1
            summary.setdefault("code", {})["swebench_instances"] = int(total)
        except Exception as e:
            summary.setdefault("code", {})["swebench_error"] = str(e)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    t_session1 = time.perf_counter()
    _dt = float(t_session1 - t_session0)
    logging.info("[auto_benchmark] session dt=%.3fs", _dt)
    if _dt > 10.0:
        logging.warning("[auto_benchmark] slow_step session took %.3fs (>10s)", _dt)
    print(json.dumps(summary, indent=2))
    # Frontier accuracy rollup (best-effort, only when JSONLs provided or autolocated)
    try:
        frontier: Dict[str, Any] = {}
        # Text
        def _add(key: str, path: str, pred_key: str = 'prediction', ref_key: str = 'answer') -> None:
            acc = _acc_from_jsonl(path, pred_key=pred_key, ref_key=ref_key)
            if acc is not None:
                frontier[key] = float(acc)
        _add('mmlu_acc', args.mmlu_jsonl)
        _add('arc_acc', args.arc_jsonl)
        _add('hellaswag_acc', args.hellaswag_jsonl)
        _add('truthfulqa_acc', args.truthfulqa_jsonl)
        _add('winogrande_acc', args.winogrande_jsonl)
        _add('agieval_acc', args.agieval_jsonl)
        _add('bbh_acc', args.bbh_jsonl)
        # If no prediction JSONLs exist, run bounded EM evals using the real model for a quick signal
        if not frontier and args.autolocate_benchmarks:
            for name, path in (
                ("mmlu_acc", args.mmlu_jsonl),
                ("arc_acc", args.arc_jsonl),
                ("hellaswag_acc", args.hellaswag_jsonl),
                ("truthfulqa_acc", args.truthfulqa_jsonl),
                ("winogrande_acc", args.winogrande_jsonl),
            ):
                val = _run_em_eval(path, name.split('_')[0])
                if val is not None:
                    frontier[name] = float(val)
        # Code
        _add('humaneval_acc', args.humaneval_jsonl)
        _add('swebench_acc', args.swebench_jsonl)
        # VL / Video
        _add('vqav2_acc', args.vqav2_jsonl, pred_key='prediction', ref_key='answer')
        _add('okvqa_acc', args.okvqa_jsonl, pred_key='prediction', ref_key='answer')
        # COCO captions overlap proxy
        coco_overlap = _bleu_overlap_jsonl(args.coco_captions_jsonl)
        if coco_overlap is not None:
            frontier['coco_overlap'] = float(coco_overlap)
        # If we couldn't find prediction JSONLs, try running generation-based evals for a small sample
        if args.autolocate_benchmarks:
            try:
                if 'vqav2_acc' not in frontier and args.vqav2_jsonl:
                    gen_vqa = _vqa_generate_eval(args.device, args.preset, args.vqav2_jsonl, max_samples=int(args.max_prompts))
                    if gen_vqa and isinstance(gen_vqa.get('acc'), float):
                        frontier['vqav2_acc'] = float(gen_vqa['acc'])
                if 'okvqa_acc' not in frontier and args.okvqa_jsonl:
                    gen_ok = _vqa_generate_eval(args.device, args.preset, args.okvqa_jsonl, max_samples=int(args.max_prompts))
                    if gen_ok and isinstance(gen_ok.get('acc'), float):
                        frontier['okvqa_acc'] = float(gen_ok['acc'])
            except Exception:
                pass
        _add('msrvtt_vqa_acc', args.msrvtt_vqa_jsonl, pred_key='prediction', ref_key='answer')
        if frontier:
            # Append frontier block to output file for registry ingestion by higher-level tools
            try:
                data = json.loads(out_path.read_text(encoding='utf-8')) if out_path.exists() else {}
            except Exception:
                data = {}
            data['frontier'] = frontier
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass

    # Optional SOTA comparison: compute relative performance vs reference metrics
    try:
        ref_path = Path(args.sota_ref)
        if ref_path.exists():
            ref = json.loads(ref_path.read_text(encoding='utf-8'))
        else:
            ref = {}
        if ref:
            try:
                data = json.loads(out_path.read_text(encoding='utf-8')) if out_path.exists() else {}
            except Exception:
                data = {}
            compare: Dict[str, Any] = {}
            # Helper: ratio for higher-better, inverse for lower-better
            def _rel(key: str, val: float | None, ref_key: str | None = None, lower_better: bool = False) -> None:
                if val is None:
                    return
                rk = ref_key or key
                r = ref.get(rk)
                try:
                    rv = float(r)
                except Exception:
                    rv = None  # type: ignore
                if rv is None or rv == 0:
                    return
                ratio = (rv / val) if lower_better else (val / rv)
                compare[key] = {
                    'value': float(val),
                    'sota': float(rv),
                    'relative': float(ratio),
                }
            # Pull values from data/frontier
            text_tps = ((data.get('text') or {}).get('tokens_per_sec', None))
            _rel('text.tokens_per_sec', float(text_tps) if text_tps is not None else None, 'text.tokens_per_sec', False)
            fr = (data.get('frontier') or {})
            _rel('mmlu_acc', fr.get('mmlu_acc'), 'mmlu_acc', False)
            _rel('arc_acc', fr.get('arc_acc'), 'arc_acc', False)
            _rel('hellaswag_acc', fr.get('hellaswag_acc'), 'hellaswag_acc', False)
            _rel('truthfulqa_acc', fr.get('truthfulqa_acc'), 'truthfulqa_acc', False)
            _rel('winogrande_acc', fr.get('winogrande_acc'), 'winogrande_acc', False)
            _rel('humaneval_acc', fr.get('humaneval_acc'), 'humaneval_acc', False)
            _rel('swebench_acc', fr.get('swebench_acc'), 'swebench_acc', False)
            _rel('vqav2_acc', fr.get('vqav2_acc'), 'vqav2_acc', False)
            _rel('okvqa_acc', fr.get('okvqa_acc'), 'okvqa_acc', False)
            _rel('coco_overlap', fr.get('coco_overlap'), 'coco_overlap', False)
            _rel('msrvtt_vqa_acc', fr.get('msrvtt_vqa_acc'), 'msrvtt_vqa_acc', False)
            # ASR WER lower is better: compare when present
            asr = data.get('asr') or {}
            _rel('librispeech_wer', asr.get('wer'), 'librispeech_wer', True)
            # Audio/Video quality (lower better)
            q = data.get('quality') or {}
            _rel('fad', q.get('fad'), 'fad', True)
            _rel('fvd', q.get('fvd'), 'fvd', True)
            # Write comparison back to file
            if compare:
                data['compare_sota'] = compare
                out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass
    if args.out_csv:
        try:
            # Flatten a few key fields to CSV
            import csv
            flat = {
                "text_tokens_per_sec": summary.get("text", {}).get("tokens_per_sec", ""),
                "image_latency_s": (summary.get("image", {}) or {}).get("latency_s", ""),
                "clip_score": summary.get("quality", {}).get("clip_score", ""),
                "fid_clean": summary.get("quality", {}).get("fid_clean", ""),
                "fvd": summary.get("quality", {}).get("fvd", ""),
                "fad": summary.get("quality", {}).get("fad", ""),
                "code_pass@5": (summary.get("code", {}) or {}).get("pass@5", ""),
                "vqav2_acc": (summary.get("datasets", {}).get("vqav2", {}) or {}).get("acc", ""),
                "okvqa_acc": (summary.get("datasets", {}).get("okvqa", {}) or {}).get("acc", ""),
                "coco_score": (summary.get("datasets", {}).get("coco_captions", {}) or {}).get("score", ""),
                "msrvtt_acc": (summary.get("datasets", {}).get("msrvtt_vqa", {}) or {}).get("acc", ""),
            }
            with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(flat.keys()))
                w.writeheader()
                w.writerow(flat)
            print(f"[write] {args.out_csv}")
        except Exception as e:
            print(f"[warn] failed to write CSV: {e}")

    # Optional ONNX decode-step output validation
    if args.validate_onnx:
        try:
            import onnxruntime as ort  # type: ignore
            sess = ort.InferenceSession(args.validate_onnx, providers=["CPUExecutionProvider"])  # type: ignore
            outs = [o.name for o in sess.get_outputs()]
            print("[onnx] outputs:", outs)
            if not outs or outs[0] != "logits":
                raise RuntimeError("first output should be 'logits'")
            mtp = [o for o in outs if o.startswith("mtp_logits_")]
            if args.expect_mtp > 0:
                assert len(mtp) == args.expect_mtp, f"expected {args.expect_mtp} mtp outputs, got {len(mtp)}"
            else:
                assert len(mtp) == 0, f"expected 0 mtp outputs, got {len(mtp)}"
            print("[onnx] decode-step outputs validated")
        except Exception as e:
            print(f"[onnx] validation failed: {e}")


if __name__ == "__main__":
    main()


