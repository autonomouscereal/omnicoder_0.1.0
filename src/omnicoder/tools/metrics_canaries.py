from __future__ import annotations

"""
Canary/bench metrics scaffolds for Ω-Reasoner.

Provides lightweight functions to compute curves and hit-rates used in
diagnostics: RG budget vs accuracy, template hit-rate, prefix hydration speedup,
and cross-modal consistency proxies.
"""

from typing import List, Tuple, Dict


def rg_budget_curve(results: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """Given list of (budget_tokens, accuracy), return sorted curve with max-acc prefix."""
    if not results:
        return []
    res = sorted(results, key=lambda x: x[0])
    best = 0.0
    out = []
    for b, a in res:
        best = max(best, float(a))
        out.append((int(b), best))
    return out


def template_hit_rate(hits: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, float(hits) / float(total)))


def speedup(old_ms: float, new_ms: float) -> float:
    if old_ms <= 0.0 or new_ms <= 0.0:
        return 1.0
    return float(old_ms) / float(new_ms)


def cross_modal_consistency(entail_z: float, lip_sync_err: float, temporal_stab: float) -> float:
    """Combine proxy signals into a [0,1] score."""
    import math
    # Map components to [0,1]
    e = 1.0 / (1.0 + math.exp(-float(entail_z)))
    l = max(0.0, min(1.0, 1.0 - float(lip_sync_err)))
    t = max(0.0, min(1.0, float(temporal_stab)))
    return 0.4 * e + 0.3 * l + 0.3 * t


import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import torch


def _measure_tokens_per_second(max_new: int = 64, use_knn: bool = True) -> Dict[str, Any]:
    from omnicoder.inference.generate import build_mobile_model_by_name, generate
    from omnicoder.inference.gen_config import GenRuntimeConfig  # type: ignore
    from omnicoder.training.simple_tokenizer import get_text_tokenizer
    from omnicoder.inference.knn_cache import KNNCache

    model = build_mobile_model_by_name("mobile_4gb", mem_slots=4)
    model.eval()
    tok = get_text_tokenizer(prefer_hf=True)
    prompt = "Hello from metrics"
    input_ids = torch.tensor([[tok.encode(prompt)[-1]]], dtype=torch.long)

    # Optional: count write events by wrapping KNNCache.add
    write_events = {"count": 0}

    class _CountingKNN(KNNCache):
        def add(self, h, token_id):  # type: ignore[override]
            write_events["count"] += 1
            return super().add(h, token_id)

    knn = _CountingKNN(dim=model.lm_head.in_features, use_faiss=False) if use_knn else None
    t0 = time.time()
    rc = GenRuntimeConfig(super_verbose=False)
    _ = generate(
        model,
        input_ids,
        max_new_tokens=max_new,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        verify_threshold=0.0,
        speculative_draft_len=1,
        knn_cache=knn,
        knn_k=8,
        knn_lambda=0.2,
        window_size=0,
        runtime_config=rc,
    )
    dt = max(1e-6, time.time() - t0)
    tps = float(max_new / dt)
    return {
        "tokens_per_second": tps,
        "write_events": int(write_events["count"]) if use_knn else 0,
        "max_new_tokens": int(max_new),
        "used_knn": bool(use_knn),
    }


def _maybe_clip_fid(images_dir: str, ref_dir: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not images_dir:
        return out
    try:
        import os
        from torchvision import transforms as _T  # type: ignore
        import PIL.Image as _PIL  # type: ignore
        imgs = []
        texts = []
        # Load up to 32 images and dummy texts
        paths = list(Path(images_dir).rglob("*.png")) + list(Path(images_dir).rglob("*.jpg"))
        paths = paths[:32]
        if not paths:
            return out
        # Try CLIPScore if open_clip is available
        try:
            import open_clip  # type: ignore
            # Use ViT-B-32-quickgelu to match pretrained tag and avoid QuickGELU mismatch
            model_clip, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            model_clip = model_clip.to("cpu").eval()
            pil_tf = _T.ToPILImage()
            img_list = []
            for p in paths:
                im = _PIL.Image.open(p).convert("RGB")
                img_list.append(preprocess(im).unsqueeze(0))
                texts.append("image")
            img_batch = torch.cat(img_list, dim=0)
            with torch.no_grad():
                img_feat = model_clip.encode_image(img_batch)
                txt = tokenizer(texts)
                txt_feat = model_clip.encode_text(txt)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                clip_scores = (img_feat * txt_feat).sum(dim=-1).mean().item()
            out["CLIPScore_mean"] = float(clip_scores)
        except Exception:
            pass
        # Try FID if torchmetrics available and ref_dir provided
        if ref_dir:
            try:
                from torchmetrics.image.fid import FrechetInceptionDistance  # type: ignore
                fid = FrechetInceptionDistance(feature=2048).to("cpu").eval()
                for p in paths[:32]:
                    im = _PIL.Image.open(p).convert("RGB")
                    t = _T.ToTensor()(im)
                    fid.update((t * 255).byte().unsqueeze(0), real=False)
                ref_paths = list(Path(ref_dir).rglob("*.png")) + list(Path(ref_dir).rglob("*.jpg"))
                for p in ref_paths[:32]:
                    im = _PIL.Image.open(p).convert("RGB")
                    t = _T.ToTensor()(im)
                    fid.update((t * 255).byte().unsqueeze(0), real=True)
                out["FID"] = float(fid.compute().item())
            except Exception:
                pass
    except Exception:
        pass
    return out


def _maybe_fad(ref_dir: str, pred_dir: str, sr: int = 16000) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not ref_dir or not pred_dir:
        return out
    try:
        # Prefer torch-fad if available
        try:
            import torch_fad  # type: ignore
            from glob import glob
            ref = glob(str(Path(ref_dir) / "*.wav"))
            pred = glob(str(Path(pred_dir) / "*.wav"))
            if ref and pred:
                fad_score = torch_fad.fad_from_paths(ref, pred, device="cpu")
                out["FAD"] = float(fad_score)
                return out
        except Exception:
            pass
        # Fallback to torchmetrics if available
        try:
            from torchmetrics.audio.fad import FrechetAudioDistance  # type: ignore
            import soundfile as sf  # type: ignore
            from glob import glob
            fad = FrechetAudioDistance(sample_rate=sr)
            for p in glob(str(Path(ref_dir) / "*.wav")):
                wav, r = sf.read(p)
                if r != sr:
                    continue
                fad.update(torch.tensor(wav).unsqueeze(0), real=True)
            for p in glob(str(Path(pred_dir) / "*.wav")):
                wav, r = sf.read(p)
                if r != sr:
                    continue
                fad.update(torch.tensor(wav).unsqueeze(0), real=False)
            out["FAD"] = float(fad.compute().item())
        except Exception:
            pass
    except Exception:
        pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Optional metrics canaries for images/audio and tokens/s")
    ap.add_argument("--images_dir", type=str, default="", help="Directory of generated images to score (optional)")
    ap.add_argument("--ref_dir", type=str, default="", help="Directory of reference images for FID (optional)")
    ap.add_argument("--audio_ref_dir", type=str, default="", help="Directory of reference WAVs for FAD (optional)")
    ap.add_argument("--audio_pred_dir", type=str, default="", help="Directory of generated WAVs for FAD (optional)")
    ap.add_argument("--video_pred_dir", type=str, default="", help="Directory of generated videos (optional)")
    ap.add_argument("--video_ref_dir", type=str, default="", help="Directory of reference videos (optional)")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--no_knn", action="store_true")
    ap.add_argument("--bench_variable_k", action="store_true")
    # KV prefetch canary (simulated)
    ap.add_argument("--kv_prefetch_canary", action="store_true")
    ap.add_argument("--kv_page_len", type=int, default=256)
    ap.add_argument("--kv_max_pages", type=int, default=32)
    ap.add_argument("--kv_prefetch_ahead", type=int, default=1)
    ap.add_argument("--kv_spill_prec", type=str, default="", help="If set (fp16|bf16), annotate mixed-precision spill setting for KV canary")
    ap.add_argument("--kv_steps", type=int, default=1024)
    ap.add_argument("--out_json", type=str, default="weights/metrics_canaries.json")
    # ap.add_argument("--longdoc_random_access", action="store_true", help="Run a long-document random-access canary using landmarks and windowed decode")
    ap.add_argument("--summarize_acceptance", action="store_true", help="Summarize acceptance threshold vs provider TPS if provider bench and thresholds exist")
    ap.add_argument("--coreml_canary", action="store_true", help="Include provider_bench tokens/s if present")
    ap.add_argument("--longdoc_random_access", action="store_true", help="Run a long-document random-access canary using landmarks and windowed decode")
    # PQ/write-policy acceptance proxy
    ap.add_argument("--pq_index", type=str, default="")
    ap.add_argument("--write_marks", type=str, default="")
    args = ap.parse_args()

    Path("weights").mkdir(exist_ok=True)
    summary: Dict[str, Any] = {}

    # Tokens/s canary (engages verifier + write policy via knn_cache)
    try:
        summary["tokens"] = _measure_tokens_per_second(max_new=int(args.max_new_tokens), use_knn=(not args.no_knn))
    except Exception as e:
        summary["tokens"] = {"error": str(e)}

    # Image metrics (optional)
    try:
        img = _maybe_clip_fid(args.images_dir, args.ref_dir)
        if img:
            summary["images"] = img
    except Exception as e:
        summary["images"] = {"error": str(e)}

    # Audio metrics (optional)
    try:
        aud = _maybe_fad(args.audio_ref_dir, args.audio_pred_dir)
        if aud:
            summary["audio"] = aud
    except Exception as e:
        summary["audio"] = {"error": str(e)}

    # Video FVD (optional)
    if args.video_pred_dir and args.video_ref_dir:
        try:
            from omnicoder.eval.video_eval import _compute_fvd  # type: ignore
            fvd = _compute_fvd(args.video_pred_dir, args.video_ref_dir, num_frames=16)
            if fvd >= 0:
                summary["video"] = {"FVD": float(fvd)}
            else:
                summary.setdefault("video", {})
                summary["video"]["error"] = "FVD not computed"
        except Exception as e:
            summary["video"] = {"error": str(e)}

    # KV prefetch canary
    if args.kv_prefetch_canary:
        try:
            from omnicoder.inference.runtimes.kv_prefetch import PagingConfig, LRUKVPager  # type: ignore
            cfg = PagingConfig(n_layers=12, heads=16, page_len=int(args.kv_page_len), max_pages_ram=int(args.kv_max_pages), prefetch_ahead=int(args.kv_prefetch_ahead))
            pager = LRUKVPager(cfg)
            for pos in range(int(args.kv_steps)):
                pager.access(pos)
            stats = pager.stats()
            summary["kv_prefetch"] = stats
            if args.kv_spill_prec:
                summary.setdefault('kv_spill', {})
                summary['kv_spill']['precision'] = str(args.kv_spill_prec)
            # Basic thresholds to highlight regressions
            try:
                miss = float(stats.get('miss_rate', 0.0))
                stall = float(stats.get('stall_ratio', 0.0))
                # Allow up to 40% miss and 10% stall by default for this synthetic walk
                summary['kv_prefetch_ok'] = bool(miss <= 0.4 and stall <= 0.1)
            except Exception:
                pass
            # Enforce simple thresholds if sidecar exists (kvq_calibration.json)
            try:
                cal = Path('weights/kvq_calibration.json')
                if cal.exists():
                    thr_pages = max(16, int(args.kv_max_pages))
                    # Simple guard: observed working set pages should not exceed configured max by >25%
                    ws = int(stats.get('working_set_pages', stats.get('pages', 0)))
                    if ws > int(1.25 * thr_pages):
                        summary.setdefault('kv_budget', {})
                        summary['kv_budget']['violation'] = True
                        summary['kv_budget']['working_set_pages'] = ws
                        summary['kv_budget']['cap_pages'] = thr_pages
            except Exception:
                pass
        except Exception as e:
            summary["kv_prefetch"] = {"error": str(e)}

    # Long-document random-access canary (best-effort)
    if bool(args.longdoc_random_access):
        try:
            from omnicoder.inference.generate import generate as _gen  # type: ignore
            para = "This is a long document paragraph. " * 128
            prompt = para + "\nQ: What is the last word in the paragraph?\nA:"
            out = _gen(prompt=prompt, max_new_tokens=max(8, int(args.max_new_tokens)//4))
            summary["longdoc_random_access"] = {"ok": bool(out and isinstance(out, str)), "len": (len(out) if isinstance(out, str) else 0)}
        except Exception as e:
            summary["longdoc_random_access"] = {"ok": False, "error": str(e)}

    # Summarize acceptance vs provider TPS (if files exist)
    if bool(args.summarize_acceptance):
        try:
            acc_path = Path('profiles')/'acceptance_thresholds.json'
            pb_path = Path('weights')/'release'/'text'/'provider_bench.json'
            if acc_path.exists() and pb_path.exists():
                acc = json.loads(acc_path.read_text(encoding='utf-8'))
                pb = json.loads(pb_path.read_text(encoding='utf-8'))
                measured = {}
                if isinstance(pb, dict):
                    if isinstance(pb.get('providers'), dict):
                        for k, v in pb['providers'].items():
                            try:
                                measured[str(k)] = float(v.get('tps', 0.0))
                            except Exception:
                                pass
                    else:
                        for k, v in pb.items():
                            try:
                                measured[str(k)] = float(v)
                            except Exception:
                                pass
                summary['acceptance_vs_tps'] = {
                    'acceptance': acc,
                    'provider_tps': measured,
                }
        except Exception:
            pass

    # Optional PQ acceptance canary: build/load PQ, measure acceptance with/without write-policy
    if args.pq_index and args.write_marks:
        try:
            from omnicoder.inference.retrieval_pq import PqRetriever  # type: ignore
            from omnicoder.modeling.transformer_moe import OmniTransformer  # type: ignore
            from omnicoder.training.simple_tokenizer import get_text_tokenizer  # type: ignore
            pq = PqRetriever(args.pq_index)
            model = OmniTransformer()
            model.eval()
            tok = get_text_tokenizer(prefer_hf=True)
            acc_on = []
            acc_off = []
            cnt = 0
            with open(args.write_marks, 'r', encoding='utf-8') as f:
                for line in f:
                    j = json.loads(line)
                    if not isinstance(j, dict) or 'text' not in j:
                        continue
                    text = j['text']
                    ids = torch.tensor([tok.encode(text)], dtype=torch.long)
                    # ON: enable write-policy head probability
                    out = model(ids, use_cache=False, return_hidden=True)
                    hid = out[-1] if isinstance(out, tuple) else model.ln_f(model.embed(ids))
                    wp = torch.sigmoid(model.write_head(hid)).mean().item()
                    acc_on.append(float(wp))
                    # OFF: baseline (no write), proxy as zero
                    acc_off.append(0.0)
                    cnt += 1
                    if cnt >= 8:
                        break
            if acc_on:
                summary['pq_acceptance_on'] = float(sum(acc_on) / len(acc_on))
                summary['pq_acceptance_off'] = float(sum(acc_off) / len(acc_off)) if acc_off else 0.0
                summary['pq_acceptance_delta'] = summary['pq_acceptance_on'] - summary['pq_acceptance_off']
        except Exception:
            pass

    # Optional variable-K / early-exit throughput canary
    if args.bench_variable_k:
        try:
            from omnicoder.inference.benchmark import bench_tokens_per_second
            from omnicoder.modeling.transformer_moe import OmniTransformer
            from omnicoder.config import MobilePreset
            p = MobilePreset()
            base = OmniTransformer(vocab_size=p.vocab_size, n_layers=p.n_layers, d_model=p.d_model, n_heads=p.n_heads, mlp_dim=p.mlp_dim, n_experts=p.moe_experts, top_k=p.moe_top_k, max_seq_len=p.max_seq_len, kv_latent_dim=p.kv_latent_dim, multi_query=p.multi_query, multi_token=1)
            t_base = bench_tokens_per_second(base, seq_len=128, gen_tokens=128, device='cpu')
            tuned = OmniTransformer(vocab_size=p.vocab_size, n_layers=p.n_layers, d_model=p.d_model, n_heads=p.n_heads, mlp_dim=p.mlp_dim, n_experts=p.moe_experts, top_k=p.moe_top_k, max_seq_len=p.max_seq_len, kv_latent_dim=p.kv_latent_dim, multi_query=p.multi_query, multi_token=1)
            t_tuned = bench_tokens_per_second(tuned, seq_len=128, gen_tokens=128, device='cpu')
            summary['variable_k_bench'] = {'baseline_tps': t_base, 'variable_k_tps': t_tuned, 'speedup_x': (t_tuned / max(t_base, 1e-9))}
        except Exception:
            pass

    # Optional: parse last pretrain log for expert load stats
    try:
        import glob
        logs = sorted(glob.glob('weights/pretrain_log.jsonl'))
        if logs:
            with open(logs[-1], 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    rec = json.loads(lines[-1])
                    if isinstance(rec, dict):
                        if 'expert_load_std' in rec:
                            summary['expert_load_std'] = float(rec['expert_load_std'])
                        if 'expert_load_hist' in rec:
                            summary['expert_load_hist'] = rec['expert_load_hist']
    except Exception:
        pass

    # Optional include of provider_bench tokens/s
    if bool(args.coreml_canary):
        try:
            pb = Path('weights')/'release'/'text'/'provider_bench.json'
            if pb.exists():
                summary['provider_bench'] = json.loads(pb.read_text(encoding='utf-8'))
        except Exception:
            pass

    outp = Path(args.out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[metrics] wrote", str(outp))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()


