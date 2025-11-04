import argparse
import os
import numpy as np
import torch

try:
    import onnxruntime as ort
except Exception:
    ort = None

from omnicoder.training.simple_tokenizer import get_text_tokenizer
from omnicoder.utils.resources import apply_thread_env_if_auto
from omnicoder.utils.env_registry import load_dotenv_best_effort
from omnicoder.utils.env_defaults import apply_core_defaults, apply_run_env_defaults, apply_profile
from omnicoder.inference.generate import GenRuntimeConfig  # type: ignore
try:
    from omnicoder.inference.onnx_utils import zeros_cached  # type: ignore
except Exception:
    zeros_cached = None  # type: ignore

# NF4 codebook for KV-cache quant emulation (runner-side)
_NF4_CODEBOOK = np.array([
    -1.078125, -0.8515625, -0.671875, -0.5234375,
    -0.39453125, -0.27734375, -0.1640625, -0.0546875,
     0.0546875,  0.1640625,  0.27734375,  0.39453125,
     0.5234375,  0.671875,   0.8515625,   1.078125,
], dtype=np.float32)


def _parse_cache_meta(session: "ort.InferenceSession"):
    inputs = session.get_inputs()
    # Expect names: input_ids, k_lat_0..k_lat_{L-1}, v_lat_0..v_lat_{L-1}
    k_inputs = [i for i in inputs if i.name.startswith("k_lat_")]
    v_inputs = [i for i in inputs if i.name.startswith("v_lat_")]
    k_inputs.sort(key=lambda x: int(x.name.split("_")[-1]))
    v_inputs.sort(key=lambda x: int(x.name.split("_")[-1]))
    if len(k_inputs) != len(v_inputs):
        raise RuntimeError("Mismatched K/V cache inputs in ONNX model")
    num_layers = len(k_inputs)
    # Shapes are [B, H, T_past, DL]; H and DL should be static from export
    meta = []
    for ki in k_inputs:
        shape = ki.shape
        # ONNX may return symbolic dims; attempt to cast
        try:
            heads = int(shape[1])
            d_lat = int(shape[3])
        except Exception:
            # Fallback to commonly used small defaults if shapes are symbolic
            heads = 8
            d_lat = 160
        meta.append((heads, d_lat))
    return num_layers, meta, k_inputs, v_inputs


def _apply_logit_bias_inplace(logits: np.ndarray, bias_map: dict[int, float] | None, alpha: float) -> None:
    if bias_map and alpha > 0.0:
        try:
            V = logits.shape[-1]
            # Apply only for valid ids in range
            for tid, val in bias_map.items():
                if 0 <= int(tid) < V:
                    logits[int(tid)] = logits[int(tid)] + float(alpha) * float(val)
        except Exception:
            pass


def _sample_next_token(logits: np.ndarray, temperature: float, top_k: int, top_p: float) -> int:
    # logits: (V,)
    scaled = logits / max(temperature, 1e-5)
    if top_k > 0:
        topk_idx = np.argpartition(-scaled, top_k)[:top_k]
        topk_logits = scaled[topk_idx]
        probs = np.exp(topk_logits - np.max(topk_logits))
        probs = probs / np.clip(probs.sum(), 1e-12, None)
        choice = np.random.choice(len(topk_idx), p=probs)
        return int(topk_idx[choice])
    if 0 < top_p < 1:
        sorted_idx = np.argsort(-scaled)
        sorted_logits = scaled[sorted_idx]
        probs = np.exp(sorted_logits - np.max(sorted_logits))
        probs = probs / np.clip(probs.sum(), 1e-12, None)
        cum = np.cumsum(probs)
        cutoff = np.searchsorted(cum, top_p, side="right") + 1
        idx_slice = sorted_idx[:cutoff]
        probs_slice = probs[:cutoff]
        probs_slice = probs_slice / np.clip(probs_slice.sum(), 1e-12, None)
        choice = np.random.choice(len(idx_slice), p=probs_slice)
        return int(idx_slice[choice])
    # greedy by default
    return int(np.argmax(scaled))


def _quantize_uint8(x: np.ndarray, scale: float | np.ndarray | None = None, zero: int | np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Affine uint8 quantization returning (q, scale, zero).

    Supports scalar or elementwise scale/zero (broadcastable to x).
    """
    if x.size == 0:
        return x.astype(np.uint8), np.array(1.0, dtype=np.float32), np.array(0, dtype=np.int32)
    if scale is None:
        mx = float(np.max(np.abs(x)))
        if not np.isfinite(mx) or mx < 1e-8:
            return np.zeros_like(x, dtype=np.uint8), np.array(1.0, dtype=np.float32), np.array(128, dtype=np.int32)
        scale = np.array(mx / 127.0, dtype=np.float32)
    if zero is None:
        zero = np.array(128, dtype=np.int32)
    q = np.round(x / scale + zero).astype(np.int32)
    q = np.clip(q, 0, 255).astype(np.uint8)
    return q, np.asarray(scale, dtype=np.float32), np.asarray(zero, dtype=np.int32)


def _dequantize_uint8(q: np.ndarray, scale: float | np.ndarray, zero: int | np.ndarray) -> np.ndarray:
    if q.size == 0:
        return q.astype(np.float32)
    return (q.astype(np.float32) - np.asarray(zero, dtype=np.float32)) * np.asarray(scale, dtype=np.float32)


def _compute_groupwise_scale_zero(x: np.ndarray, group: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-head, per-group (along latent dim) symmetric scales and zeros broadcastable to x.

    x shape: (1, H, T, DL). Returns scale, zero shaped (1, H, 1, DL).
    """
    if x.size == 0:
        return np.array(1.0, dtype=np.float32), np.array(128, dtype=np.int32)
    b, h, t, dl = x.shape
    g = max(1, int(group))
    # reshape to (1,H,T,G,group)
    G = (dl + g - 1) // g
    pad = G * g - dl
    if pad > 0:
        x_pad = np.pad(x, ((0,0),(0,0),(0,0),(0,pad)), mode='constant')
    else:
        x_pad = x
    xg = x_pad.reshape(b, h, t, G, g)
    # max over T for stability
    mx = np.max(np.abs(xg), axis=2, keepdims=True)  # (1,H,1,G,g)
    mx = np.maximum(mx, 1e-8)
    scale_g = (mx / 127.0).astype(np.float32)
    zero_g = np.full_like(scale_g, 128, dtype=np.int32)
    # broadcast back to (1,H,1,DL)
    scale = scale_g.reshape(b, h, 1, G * g)[..., :dl]
    zero = zero_g.reshape(b, h, 1, G * g)[..., :dl]
    return scale, zero


def _compute_groupwise_mean_std(x: np.ndarray, group: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-head, per-group mean/std for NF4 quantization; returns (mean,std) shaped (1,H,1,DL)."""
    if x.size == 0:
        return np.array(0.0, dtype=np.float32), np.array(1.0, dtype=np.float32)
    b, h, t, dl = x.shape
    g = max(1, int(group))
    G = (dl + g - 1) // g
    pad = G * g - dl
    if pad > 0:
        x_pad = np.pad(x, ((0,0),(0,0),(0,0),(0,pad)), mode='constant')
    else:
        x_pad = x
    xg = x_pad.reshape(b, h, t, G, g)
    mean_g = xg.mean(axis=2, keepdims=True)  # (1,H,1,G,g)
    var_g = ((xg - mean_g) ** 2).mean(axis=2, keepdims=True)
    std_g = np.sqrt(np.clip(var_g, 1e-8, None)).astype(np.float32)
    mean = mean_g.reshape(b, h, 1, G * g)[..., :dl].astype(np.float32)
    std = std_g.reshape(b, h, 1, G * g)[..., :dl].astype(np.float32)
    return mean, std


def _quantize_nf4(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x.size == 0:
        return x.astype(np.uint8), np.array(0.0, dtype=np.float32), np.array(1.0, dtype=np.float32)
    # Normalize
    y = (x - mean) / std
    # Nearest codebook index per element
    # y (..., DL) vs codebook (16,) -> broadcast
    diff = np.abs(y[..., None] - _NF4_CODEBOOK[None, None, None, None, :])
    idx = diff.argmin(axis=-1).astype(np.uint8)
    return idx, mean.astype(np.float32), std.astype(np.float32)


def _dequantize_nf4(q: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if q.size == 0:
        return q.astype(np.float32)
    val = _NF4_CODEBOOK[q]
    return (val * std + mean).astype(np.float32)


# ------- Learned KV compression (autoencoder sidecar) helpers -------
def _load_kv_autoencoder(sidecar_path: str):
    try:
        import json as _json
        import torch as _torch  # lazy import
        meta = _json.loads(open(sidecar_path, 'r', encoding='utf-8').read())
        kv = meta.get('kv_autoencoder', {}) if isinstance(meta, dict) else {}
        wpt = kv.get('weights', '')
        if not wpt:
            return None
        ck = _torch.load(wpt, map_location='cpu')
        sd = ck.get('state_dict', {}) if isinstance(ck, dict) else {}
        if not sd:
            return None
        # Expect linear layers weights: enc.weight (latent, dim), dec.weight (dim, latent)
        enc = sd.get('enc.weight', None)
        dec = sd.get('dec.weight', None)
        if enc is None or dec is None:
            return None
        enc_np = enc.detach().cpu().numpy().T.astype(np.float32)  # (dim, latent)
        dec_np = dec.detach().cpu().numpy().T.astype(np.float32)  # (latent, dim)
        return {'enc': enc_np, 'dec': dec_np, 'dim': int(enc_np.shape[0]), 'latent': int(enc_np.shape[1])}
    except Exception:
        return None


def _ae_encode(x: np.ndarray, enc: np.ndarray) -> np.ndarray:
    # x: (..., DL), enc: (DL, L) -> (..., L)
    if x.size == 0:
        return x.astype(np.float32)
    xl = x.reshape(-1, x.shape[-1])
    z = xl @ enc
    return z.reshape(*x.shape[:-1], enc.shape[-1]).astype(np.float32)


def _ae_decode(z: np.ndarray, dec: np.ndarray) -> np.ndarray:
    # z: (..., L), dec: (L, DL) -> (..., DL)
    if z.size == 0:
        return z.astype(np.float32)
    zl = z.reshape(-1, z.shape[-1])
    x = zl @ dec
    return x.reshape(*z.shape[:-1], dec.shape[-1]).astype(np.float32)


def _load_dotenv(env_path: str = ".env") -> None:
    try:
        load_dotenv_best_effort((env_path,))
    except Exception:
        pass


def main():
    try:
        apply_thread_env_if_auto()
    except Exception:
        pass
    # Load .env files and apply centralized defaults/profile early
    try:
        load_dotenv_best_effort((".env", ".env.tuned"))
        apply_core_defaults(os.environ)  # type: ignore[arg-type]
        apply_run_env_defaults(os.environ)  # type: ignore[arg-type]
        apply_profile(os.environ, "quality")  # type: ignore[arg-type]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path to decode-step ONNX model")
    ap.add_argument("--prompt", type=str, default="Hello, OmniCoder!")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--provider", type=str, default=os.getenv("OMNICODER_ORT_PROVIDER", "CPUExecutionProvider"), help="ORT provider, e.g., CPUExecutionProvider, DmlExecutionProvider, NNAPIExecutionProvider")
    ap.add_argument("--mobile_preset", type=str, default=os.getenv("OMNICODER_STUDENT_PRESET", "mobile_4gb"), help="Preset name used to auto-load acceptance thresholds when verify_threshold=0")
    ap.add_argument("--kvq", type=str, default=os.getenv("OMNICODER_KVQ", "none"), choices=["none","u8","nf4"], help="Quantize KV cache for storage and dequantize per step (runner supports u8/nf4 emulation)")
    ap.add_argument("--kvq_group", type=int, default=64)
    ap.add_argument("--kvq_calibration", type=str, default="", help="Optional JSON with per-head/group KV stats for de/quant guidance")
    ap.add_argument("--kv_paged", action="store_true", help="Use paged KV cache if sidecar kv_paging.json is present")
    ap.add_argument("--kv_prefetch_predictor", type=str, default=os.getenv("OMNICODER_KV_PREFETCH_PREDICTOR", ""), help="Optional JSON sidecar with simple prefetch policy for paged KV")
    ap.add_argument("--window", type=int, default=int(os.getenv("OMNICODER_WINDOW_SIZE", "0")), help="If >0, materialize only the last window tokens from paged cache per step")
    ap.add_argument("--speculative_draft_len", type=int, default=int(os.getenv("OMNICODER_SPEC_DRAFT_LEN", "0")), help="If >0 and MTP heads present, accept up to N lookahead tokens per step without extra runs")
    ap.add_argument("--verify_threshold", type=float, default=float(os.getenv("OMNICODER_VERIFY_THRESHOLD", "0.0")), help="Minimum prob under current-step logits to accept a speculative token")
    ap.add_argument("--draft_model", type=str, default="", help="Optional path to a draft ONNX model for speculative decoding (one-step)")
    ap.add_argument("--draft_verify_threshold", type=float, default=float(os.getenv("OMNICODER_DRAFT_VERIFY_THRESHOLD", "0.1")), help="Acceptance threshold for draft token prob under base logits")
    ap.add_argument("--tree_width", type=int, default=int(os.getenv("OMNICODER_TREE_WIDTH", "1")), help="If >1, sample this many candidates and pick the best by base probability")
    ap.add_argument("--tree_depth", type=int, default=1, help="Reserved for future multi-step lookahead; currently unused (depth=1)")
    # Hidden expert: per-token logit bias (JSON id->bias map)
    ap.add_argument("--logit_bias_file", type=str, default=os.getenv("OMNICODER_LOGIT_BIAS_FILE", ""))
    ap.add_argument("--logit_bias_alpha", type=float, default=float(os.getenv("OMNICODER_LOGIT_BIAS_ALPHA", "0.0")))
    ap.add_argument('--kv_retention_sidecar', type=str, default=os.getenv('OMNICODER_KV_RETENTION',''), help='Optional: path to kv_retention.json describing compressive_slots/window policy')
    ap.add_argument('--kv_compress_sidecar', type=str, default=os.getenv('OMNICODER_KV_COMPRESS_SIDECAR',''), help='Optional: path to kv_compress_sidecar.json from kv-autoencoder-train')
    # Optional seeding for reproducibility (Python/NumPy)
    try:
        import random as _rand
        import numpy as _np
        _seed_env = os.getenv("OMNICODER_SEED", "").strip()
        if _seed_env:
            s = int(_seed_env)
            _rand.seed(s)
            _np.random.seed(s)
    except Exception:
        pass
    # Optional system/prompt template for ONNX runner parity
    ap.add_argument("--system", type=str, default=os.getenv("OMNICODER_SYSTEM_PROMPT", "Use the provided context; be concise and correct."))
    ap.add_argument("--prompt_template", type=str, default=os.getenv("OMNICODER_PROMPT_TEMPLATE", "[SYSTEM] {system}\n{context}\n\n[USER] {user}\n[ASSISTANT]"))
    # Optional local retrieval to prepend context (TF-IDF/naive search)
    ap.add_argument("--retrieve_path", type=str, default=os.getenv("OMNICODER_RETRIEVE_PATH", ""))
    ap.add_argument("--retrieve_k", type=int, default=int(os.getenv("OMNICODER_RETRIEVE_K", "3")))
    ap.add_argument("--retrieve_max_chars", type=int, default=int(os.getenv("OMNICODER_RETRIEVE_MAX_CHARS", "4000")))
    args = ap.parse_args()

    if ort is None:
        print("[onnx] onnxruntime is not installed.")
        return

    try:
        use_hf = os.getenv("OMNICODER_ONNX_USE_HF_TOKENIZER", "0") == "1"
        hf_id = os.getenv("OMNICODER_HF_TOKENIZER", "meta-llama/Meta-Llama-3-8B-Instruct") if use_hf else None
        # If the model export uses a 32k vocab but hf_id implies 128k (Llama 3), remap to an accessible LLaMA tokenizer
        try:
            if use_hf:
                import json as _json
                um = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(args.model))), 'unified_vocab_map.json')
                text_size = None
                if os.path.exists(um):
                    meta = _json.loads(open(um, 'r', encoding='utf-8').read())
                    text_size = int(meta.get('text_size', 0)) if isinstance(meta, dict) else 0
                if (text_size == 32000) and (hf_id and 'meta-llama' in hf_id.lower()):
                    print(f"[tok] remap hf_id {hf_id} -> hf-internal-testing/llama-tokenizer for 32k export")
                    hf_id = 'hf-internal-testing/llama-tokenizer'
        except Exception:
            pass
        # If not using HF, force a 32k TextTokenizer when unified vocab reports 32k text range
        if not use_hf:
            try:
                import json as _json
                um = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(args.model))), 'unified_vocab_map.json')
                tsize = 0
                if os.path.exists(um):
                    meta = _json.loads(open(um, 'r', encoding='utf-8').read())
                    tsize = int(meta.get('text_size', 0)) if isinstance(meta, dict) else 0
                if tsize == 32000:
                    tok = get_text_tokenizer(prefer_hf=False, hf_id=None)
                    try:
                        if hasattr(tok, 'vocab_size'):
                            tok.vocab_size = 32000
                    except Exception:
                        pass
                else:
                    tok = get_text_tokenizer(prefer_hf=False)
            except Exception:
                tok = get_text_tokenizer(prefer_hf=False)
            print(f"[tok] prefer_hf=0 id=None")
        else:
            tok = get_text_tokenizer(prefer_hf=use_hf, hf_id=hf_id)
            print(f"[tok] prefer_hf={use_hf} id={hf_id}")
    except Exception as e:
        print(f"[tok] hf load failed: {e}; falling back to byte tokenizer")
        tok = get_text_tokenizer(prefer_hf=False)
    # Auto-load default acceptance thresholds per preset if verify_threshold not provided
    try:
        if float(args.verify_threshold) == 0.0:
            from omnicoder.utils.thresholds import get_accept_threshold  # type: ignore
            args.verify_threshold = float(get_accept_threshold(str(args.mobile_preset), float(args.verify_threshold)))
    except Exception:
        pass
    # Load optional bias map once
    bias_map: dict[int, float] | None = None
    if args.logit_bias_file and args.logit_bias_alpha > 0.0:
        try:
            import json as _json
            raw = _json.loads(open(args.logit_bias_file, 'r', encoding='utf-8').read())
            if isinstance(raw, dict):
                tmp: dict[int, float] = {}
                for k, v in list(raw.items())[: 8192]:
                    try:
                        tmp[int(k)] = float(v)
                    except Exception:
                        continue
                bias_map = tmp
        except Exception:
            bias_map = None
    # Build optional retrieval context
    context_txt = ""
    if args.retrieve_path:
        try:
            import os as _os
            import json as _json
            root = str(args.retrieve_path).strip()
            hits: list[str] = []
            paths: list[str] = []
            if _os.path.isdir(root):
                for dirpath, _dirnames, filenames in _os.walk(root):
                    if any(seg.startswith('.') for seg in dirpath.split(_os.sep)):
                        continue
                    for fn in filenames:
                        if any(fn.lower().endswith(ext) for ext in (".txt", ".md", ".jsonl")):
                            paths.append(_os.path.join(dirpath, fn))
            elif _os.path.isfile(root):
                paths.append(root)
            q = str(args.prompt).strip().lower()
            for p in paths:
                try:
                    if p.lower().endswith('.jsonl'):
                        with open(p, 'r', encoding='utf-8') as f:
                            for ln, line in enumerate(f, start=1):
                                s = line.strip()
                                if not s:
                                    continue
                                try:
                                    obj = _json.loads(s)
                                    text = str(obj.get('text', obj))
                                except Exception:
                                    text = s
                                low = text.lower()
                                if q and q in low:
                                    pos = low.find(q)
                                    start = max(0, pos - 80)
                                    end = min(len(text), pos + 80)
                                    hits.append(text[start:end])
                    else:
                        text = open(p, 'r', encoding='utf-8', errors='ignore').read()
                        low = text.lower()
                        if q and q in low:
                            pos = low.find(q)
                            start = max(0, pos - 120)
                            end = min(len(text), pos + 120)
                            hits.append(text[start:end])
                except Exception:
                    continue
            if hits:
                context_txt = "\n\n".join(hits[: max(1, int(args.retrieve_k))])
                if int(args.retrieve_max_chars) > 0 and len(context_txt) > int(args.retrieve_max_chars):
                    context_txt = context_txt[: int(args.retrieve_max_chars)]
        except Exception:
            context_txt = ""
    try:
        template = str(args.prompt_template)
        system = str(args.system)
        full_prompt = template.format(system=system, context=context_txt, user=args.prompt)
    except Exception:
        full_prompt = args.prompt
    ids = tok.encode(full_prompt)
    # Clamp to text vocab slice to avoid multimodal spillover when decoding via ONNX
    try:
        import json as _json
        from pathlib import Path as _P
        text_vocab = None
        um = _P("/workspace/weights/release/unified_vocab_map.json")
        if um.exists():
            meta = _json.loads(um.read_text(encoding='utf-8'))
            if isinstance(meta, dict):
                text_vocab = int(meta.get("text_size", 0)) or None
        if text_vocab is None:
            text_vocab = int(getattr(tok, 'vocab_size', 0)) or None
        if text_vocab is not None and text_vocab > 0:
            bad = sum(1 for i in ids if int(i) >= int(text_vocab))
            if bad:
                eos_id = getattr(tok, 'eos_token_id', None)
                repl = int(eos_id) if isinstance(eos_id, int) and eos_id is not None else int(text_vocab) - 1
                ids = [int(i) if int(i) < int(text_vocab) else repl for i in ids]
                print(f"[info] tok.encode clamped: {bad} ids >= text_vocab={int(text_vocab)} replaced_with={int(repl)}")
    except Exception:
        pass
    if not isinstance(ids, (list, tuple)) or not ids:
        ids = [1]
    try:
        V = int(args.vocab_size)
        if V > 0:
            ids = [int(i) % V for i in ids]
    except Exception:
        pass
    input_ids = np.array([ids], dtype=np.int64)

    # If a KV quant sidecar exists next to the model, align/enforce runner args
    try:
        import os as _os, json as _json
        side = args.model.replace('.onnx', '.kvq.json')
        if _os.path.exists(side):
            meta = _json.loads(open(side, 'r', encoding='utf-8').read())
            if isinstance(meta, dict):
                scheme = str(meta.get('scheme', '')).lower() if 'scheme' in meta else ''
                gsz = int(meta.get('group_size', args.kvq_group)) if 'group_size' in meta else args.kvq_group
                if args.kvq == 'none' and scheme in ('u8','nf4'):
                    print(f"[kvq] adopting sidecar scheme='{scheme}' group={gsz}")
                    args.kvq = scheme
                    args.kvq_group = gsz
                else:
                    if gsz != args.kvq_group:
                        print(f"[kvq] overriding group size to sidecar group={gsz} (was {args.kvq_group})")
                        args.kvq_group = gsz
                    if scheme and scheme != args.kvq:
                        print(f"[kvq] sidecar scheme='{scheme}' differs from requested '{args.kvq}'; proceeding with requested")
        else:
            # Warn if KVQ requested without any guidance
            if args.kvq in ('u8','nf4') and not args.kvq_calibration:
                print("[kvq] warning: no KVQ sidecar or calibration found; proceeding with default per-step dequant emulation")
    except Exception:
        pass

    # Auto-enable paged KV if sidecar exists and user did not pass --kv_paged
    try:
        import os as _os
        paging_sidecar = _os.path.splitext(args.model)[0] + '.kv_paging.json'
        if _os.path.exists(paging_sidecar):
            if not args.kv_paged:
                print("[kv] detected kv_paging sidecar; enabling paged KV mode")
                args.kv_paged = True
            # If window not provided, derive a reasonable default from sidecar
            if int(args.window) <= 0:
                try:
                    import json as _json
                    meta = _json.loads(open(paging_sidecar, 'r', encoding='utf-8').read())
                    page_len = int(meta.get('page_len', 256))
                    # Default decode window: 4 pages (tunable); keep within a sane bound
                    args.window = max(1, min(page_len * 4, 16384))
                    print(f"[kv] deriving window={args.window} from page_len={page_len}")
                except Exception:
                    pass
    except Exception:
        pass

    # Auto-detect retention/compress sidecars if not provided
    try:
        import os as _os
        if not args.kv_retention_sidecar:
            cand = _os.path.splitext(args.model)[0] + '.kv_retention.json'
            if _os.path.exists(cand):
                args.kv_retention_sidecar = cand
                print(f"[kv] detected retention sidecar; will apply compressive policy from {cand}")
        if not args.kv_compress_sidecar:
            comp = _os.path.splitext(args.model)[0] + '.kv_compress_sidecar.json'
            if _os.path.exists(comp):
                args.kv_compress_sidecar = comp
                print(f"[kv] detected compress sidecar: {comp}")
    except Exception:
        pass

    # Auto-detect KV retention sidecar next to the model if not explicitly provided
    try:
        import os as _os
        if not args.kv_retention_sidecar:
            ret_sidecar = _os.path.splitext(args.model)[0] + '.kv_retention.json'
            if _os.path.exists(ret_sidecar):
                args.kv_retention_sidecar = ret_sidecar
                print(f"[kv] detected kv_retention sidecar: {ret_sidecar}")
    except Exception:
        pass

    # Provider selection remains as provided by CLI; avoid env churn during run
    sess = ort.InferenceSession(args.model, providers=[args.provider])  # mobile providers differ per platform
    draft_sess = None
    if args.draft_model:
        try:
            draft_sess = ort.InferenceSession(args.draft_model, providers=[args.provider])
        except Exception:
            draft_sess = None

    # Detect whether model expects explicit per-layer K/V inputs or maintains DynamicCache internally
    _inputs = sess.get_inputs()
    _has_explicit_kv = any(i.name.startswith('k_lat_') for i in _inputs)
    if _has_explicit_kv:
        num_layers, meta, k_inputs, v_inputs = _parse_cache_meta(sess)
    else:
        num_layers, meta, k_inputs, v_inputs = 0, [], [], []

    # Initialize caches; paged mode stores list of pages [(k,v),...]
    caches_k = []
    caches_v = []
    kv_paged_sidecar = None
    if args.kv_paged:
        # Use top-level os; import json locally without shadowing
        import json as _json
        side = os.path.splitext(args.model)[0] + ".kv_paging.json"
        if os.path.exists(side):
            kv_paged_sidecar = _json.loads(open(side, 'r').read())
            # If window not set, adopt a reasonable default based on sidecar
            try:
                if (not args.window) or args.window <= 0:
                    page_len = int(kv_paged_sidecar.get('page_len', 256))
                    # default to two pages worth of tokens, capped
                    args.window = max(page_len, min(2048, page_len * 2))
            except Exception:
                pass
    for (heads, d_lat) in meta:
        if kv_paged_sidecar is not None:
            caches_k.append([])
            caches_v.append([])
        else:
            if zeros_cached is not None:
                caches_k.append(zeros_cached((1, heads, 0, d_lat), np.float32))  # type: ignore[arg-type]
                caches_v.append(zeros_cached((1, heads, 0, d_lat), np.float32))  # type: ignore[arg-type]
            else:
                caches_k.append(np.zeros((1, heads, 0, d_lat), dtype=np.float32))
                caches_v.append(np.zeros((1, heads, 0, d_lat), dtype=np.float32))

    outputs = sess.get_outputs()
    output_names = [o.name for o in outputs]
    # Detect presence of MTP logits in outputs
    mtp_names = [n for n in output_names if n.startswith('mtp_logits_')]
    mtp_names.sort(key=lambda s: int(s.split('_')[-1]) if s.split('_')[-1].isdigit() else 0)
    # Precompute output index mapping to avoid repeated .index() lookups per step
    _out_index_map = {name: idx for idx, name in enumerate(output_names)}
    _mtp_indices = [(_out_index_map.get(name)) for name in mtp_names]
    # Optional acceptance outputs (when exporter emitted them): per MTP head top1 id and accept flag
    mtp_top1_names = [n for n in output_names if n.startswith('mtp_top1_')]
    mtp_acc_names = [n for n in output_names if n.startswith('mtp_accept_')]
    mtp_top1_names.sort(key=lambda s: int(s.split('_')[-1]) if s.split('_')[-1].isdigit() else 0)
    mtp_acc_names.sort(key=lambda s: int(s.split('_')[-1]) if s.split('_')[-1].isdigit() else 0)
    _mtp_top1_idx = [(_out_index_map.get(name)) for name in mtp_top1_names]
    _mtp_acc_idx = [(_out_index_map.get(name)) for name in mtp_acc_names]

    # Calibration (KV): load per-head/group stats when provided (for runners that might need to emulate exact storage)
    kvq_stats = None
    # Auto-detect calibration sidecar if not provided
    if not args.kvq_calibration and args.kvq in ("u8",):
        import os as _os
        # 1) Look next to the model as kvq_calibration.json
        cand1 = _os.path.join(_os.path.dirname(args.model), 'kvq_calibration.json')
        # 2) Look in weights/ folder at project root
        cand2 = _os.path.join('weights', 'kvq_calibration.json')
        if _os.path.exists(cand1):
            args.kvq_calibration = cand1
        elif _os.path.exists(cand2):
            args.kvq_calibration = cand2
    if args.kvq_calibration:
        try:
            import json as _json
            kvq_stats = _json.loads(open(args.kvq_calibration, 'r').read())
        except Exception:
            kvq_stats = None
    # If calibration found, prefer its group size
    if kvq_stats and isinstance(kvq_stats, dict) and 'group' in kvq_stats:
        try:
            args.kvq_group = int(kvq_stats.get('group', args.kvq_group))
        except Exception:
            pass

    # DynamicCache path: no explicit K/V inputs. Feed tokens step-by-step without constructing feeds for caches.
    # Optional: use torch for post-processing softmax/argmax steps to avoid numpy-heavy ops
    # Default to on for fastest path; can be disabled via env
    use_torch_post = True if os.getenv('OMNICODER_USE_TORCH_POST', '1') != '0' else False

    if not _has_explicit_kv:
        # Best-effort: if a .dynamic_cache.json exists, read outputs schema for logging or future routing
        try:
            import json as _json
            dc_sidecar = os.path.splitext(args.model)[0] + '.dynamic_cache.json'
            if os.path.exists(dc_sidecar):
                dc_meta = _json.loads(open(dc_sidecar, 'r', encoding='utf-8').read())
                outs = dc_meta.get('outputs', []) if isinstance(dc_meta, dict) else []
                if outs:
                    try:
                        def _shape(x):
                            try:
                                import numpy as _np  # type: ignore
                                if isinstance(x, _np.ndarray):
                                    return {'shape': list(x.shape), 'dtype': str(x.dtype)}
                            except Exception:
                                pass
                            try:
                                import torch as _t  # type: ignore
                                if isinstance(x, _t.Tensor):
                                    return {'shape': list(x.shape), 'dtype': str(x.dtype)}
                            except Exception:
                                pass
                            return type(x).__name__
                        preview = [_shape(o) for o in outs[:4]]
                        print({'dc_outputs_preview': preview, 'total': len(outs)})
                    except Exception:
                        print({'dc_outputs_total': len(outs)})
        except Exception:
            pass
        outputs = sess.get_outputs(); out_names = [o.name for o in outputs]
        logits = None
        run = sess.run
        input_name = sess.get_inputs()[0].name
        # Preallocate token buffer for fast scalar update
        _tok_buf = None
        # warm prompt
        for t in range(input_ids.shape[1]):
            step_token = input_ids[:, t:t+1]
            res = run(out_names, {input_name: step_token})
            logits = res[0]
        # Optional IO binding for generation to pin outputs and reuse input buffer
        use_iobind = (os.getenv('OMNICODER_USE_IOBIND', '0') == '1') and hasattr(sess, 'io_binding')
        io_binding = None
        logits_buf = None
        if use_iobind:
            try:
                # Prepare reusable last-token buffer and bind as input
                if _tok_buf is None:
                    _tok_buf = np.empty((1, 1), dtype=np.int64)
                io_binding = sess.io_binding()
                # Prime once to discover output shapes
                probe = run(out_names, {input_name: input_ids[:, -1:]})
                logits_shape = probe[0].shape
                logits_buf = np.empty(logits_shape, dtype=probe[0].dtype)
                import onnxruntime as _ort  # type: ignore
                io_binding.bind_input(name=input_name, device_type='cpu', device_id=0, element_type=_ort.numpy_obj_dtype_to_type(_tok_buf.dtype), shape=_tok_buf.shape, buffer_ptr=_tok_buf.ctypes.data)
                io_binding.bind_output(name='logits', device_type='cpu', device_id=0, element_type=_ort.numpy_obj_dtype_to_type(logits_buf.dtype), shape=logits_buf.shape, buffer_ptr=logits_buf.ctypes.data)
            except Exception:
                io_binding = None
                use_iobind = False
        # generate
        generated = []
        last_id = None
        _tok_buf = None
        for _ in range(args.max_new_tokens):
            if last_id is None:
                last_token = input_ids[:, -1:]
            else:
                if _tok_buf is None:
                    _tok_buf = np.empty((1, 1), dtype=np.int64)
                _tok_buf[0, 0] = last_id
                last_token = _tok_buf
            if use_iobind and io_binding is not None and logits_buf is not None:
                # Ensure input buffer points to current last_token
                _tok_buf[...] = last_token
                _ = sess.run_with_iobinding(io_binding)
                logits = logits_buf
            else:
                res = run(out_names, {input_name: last_token})
                logits = res[0]
            # Optional per-token logit bias
            try:
                _apply_logit_bias_inplace(logits[0, -1, :], bias_map, float(args.logit_bias_alpha))
            except Exception:
                pass
            if use_torch_post:
                lt = torch.from_numpy(logits[0, -1, :])
                scaled = lt / max(args.temperature, 1e-5)
                if args.top_k > 0:
                    topk = torch.topk(scaled, k=int(args.top_k)).indices
                    probs = torch.softmax(scaled[topk], dim=0)
                    choice = int(torch.multinomial(probs, num_samples=1))
                    next_id = int(topk[choice])
                elif 0 < float(args.top_p) < 1:
                    sorted_vals, sorted_idx = torch.sort(scaled, descending=True)
                    probs = torch.softmax(sorted_vals, dim=0)
                    cum = torch.cumsum(probs, dim=0)
                    cutoff = int(torch.searchsorted(cum, torch.tensor(float(args.top_p))) + 1)
                    probs_slice = probs[:cutoff]
                    probs_slice = probs_slice / max(float(probs_slice.sum().item()), 1e-12)
                    choice = int(torch.multinomial(probs_slice, num_samples=1))
                    next_id = int(sorted_idx[choice])
                else:
                    next_id = int(torch.argmax(scaled))
            else:
                next_id = _sample_next_token(logits[0, -1, :], args.temperature, args.top_k, args.top_p)
            generated.append(next_id)
            last_id = next_id
        full_ids = input_ids[0].tolist() + generated
        text = tok.decode(full_ids)
        print(text)
        return

    # Warm up over the prompt tokens to build cache (explicit-KV)
    # Feed tokens one by one; collect only last token's logits at the very end
    # Cache predictor file (if provided) outside the loop to avoid repeated IO
    _prefetch_keep_pages = None
    if kv_paged_sidecar is not None and args.kv_prefetch_predictor:
        try:
            import json as _json
            _pred = _json.loads(open(args.kv_prefetch_predictor, 'r').read())
            _prefetch_keep_pages = int(_pred.get('keep_pages', 0))
        except Exception:
            _prefetch_keep_pages = None
    # Reuse input dict for non-quantized, non-paged explicit-KV to avoid per-step dict allocations
    input_name = sess.get_inputs()[0].name
    _feed_base = None
    if (_has_explicit_kv) and (kv_paged_sidecar is None) and (args.kvq not in ("u8", "nf4")) and caches_k and isinstance(caches_k[0], np.ndarray):
        _feed_base = {k_inputs[i].name: caches_k[i] for i in range(num_layers)}
        _feed_base.update({v_inputs[i].name: caches_v[i] for i in range(num_layers)})
        _feed_base[input_name] = None  # placeholder for step token
    for t in range(input_ids.shape[1]):
        step_token = input_ids[:, t:t+1]
        # Dequantize-on-feed if cached in u8/nf4
        if args.kvq in ('u8','nf4') and caches_k and isinstance(caches_k[0], tuple):
            feed = {"input_ids": step_token}
            for i in range(num_layers):
                if args.kvq == 'u8':
                    kq, ks, kz = caches_k[i]
                    vq, vs, vz = caches_v[i]
                    feed[k_inputs[i].name] = _dequantize_uint8(kq, ks, kz)
                    feed[v_inputs[i].name] = _dequantize_uint8(vq, vs, vz)
                else:
                    kq, km, ks = caches_k[i]
                    vq, vm, vs = caches_v[i]
                    feed[k_inputs[i].name] = _dequantize_nf4(kq, km, ks)
                    feed[v_inputs[i].name] = _dequantize_nf4(vq, vm, vs)
        elif kv_paged_sidecar is not None:
            # Materialize tail window across pages (optimized helper)
            feed = {"input_ids": step_token}
            try:
                from omnicoder.modeling.kernels.kv_paged_ops import concat_kv_window  # type: ignore
            except Exception:
                concat_kv_window = None  # type: ignore
            for i, (pages_k, pages_v, kin, vin) in enumerate(zip(caches_k, caches_v, k_inputs, v_inputs)):
                if pages_k:
                    if concat_kv_window is not None:
                        k_cat, v_cat = concat_kv_window(pages_k, pages_v, int(args.window) if args.window and args.window > 0 else None)
                    else:
                        # Fallback to NumPy concat (rare)
                        if args.kv_prefetch_predictor and _prefetch_keep_pages:
                            keep = max(1, min(int(_prefetch_keep_pages), len(pages_k)))
                            sel = pages_k[-keep:]
                            selv = pages_v[-keep:]
                            k_cat = np.concatenate(sel, axis=2)
                            v_cat = np.concatenate(selv, axis=2)
                        else:
                            k_cat = np.concatenate(pages_k, axis=2)
                            v_cat = np.concatenate(pages_v, axis=2)
                        if args.window and args.window > 0 and k_cat.shape[2] > args.window:
                            k_cat = k_cat[:, :, -args.window:, :]
                            v_cat = v_cat[:, :, -args.window:, :]
                else:
                    heads, d_lat = meta[i]
                    if zeros_cached is not None:
                        k_cat = zeros_cached((1, heads, 0, d_lat), np.float32)  # type: ignore[arg-type]
                        v_cat = zeros_cached((1, heads, 0, d_lat), np.float32)  # type: ignore[arg-type]
                    else:
                        k_cat = np.zeros((1, heads, 0, d_lat), dtype=np.float32)
                        v_cat = np.zeros((1, heads, 0, d_lat), dtype=np.float32)
                feed[kin.name] = k_cat
                feed[vin.name] = v_cat
        else:
            if _feed_base is not None:
                _feed_base[input_name] = step_token
                feed = _feed_base
            else:
                feed = {"input_ids": step_token}
                # Fill K and V in a single pass using zip to minimize Python overhead
                for kin, vin, kv, vv in zip(k_inputs, v_inputs, caches_k, caches_v):
                    feed[kin.name] = kv
                    feed[vin.name] = vv
        out = sess.run(output_names, feed)
        # Optional activation quant emulation: compute a simple confidence proxy (softmax max) and choose bits
        # This is emulation only; no model graph changes.
        if act_quant_policy is not None:
            try:
                base = logits if 'logits' in locals() else out[0]
                probs = np.exp(base[0, -1, :] - np.max(base[0, -1, :]))
                conf = float(probs.max() / max(probs.sum(), 1e-12))
                # Global bits choice
                g_min = int(act_quant_policy.get('min_bits', 8))
                g_max = int(act_quant_policy.get('max_bits', 2))
                g_floor = float(act_quant_policy.get('conf_floor', 0.3))
                g_bits = g_min if conf >= g_floor else g_max
                # Per-layer override: if thresholds provided, compute layer bits
                layer_bits = []
                floors = act_quant_policy.get('_layer_conf_floor', [])
                if isinstance(floors, list) and floors:
                    for li in range(num_layers):
                        lf = float(floors[li]) if li < len(floors) else g_floor
                        layer_bits.append(g_min if conf >= lf else g_max)
                else:
                    layer_bits = [g_bits] * num_layers
                # Print the first decision once for debugging
                if '___printed_act_bits' not in globals():
                    print({'act_quant': {'conf': conf, 'global_bits': g_bits, 'layer_bits': layer_bits[:min(4, len(layer_bits))], 'layers': num_layers}})
                    globals()['___printed_act_bits'] = True
            except Exception:
                pass
        # outputs: logits, nk_0..nk_L-1, nv_0..nv_L-1
        logits = out[0]
        nk = out[1:1+num_layers]
        nv = out[1+num_layers:1+2*num_layers]
        # Optional KV retention/compression during warmup
        if args.kv_retention_sidecar:
            try:
                import json as _json
                pol = _json.loads(open(args.kv_retention_sidecar,'r',encoding='utf-8').read())
                slots = int(pol.get('compressive_slots', 0))
                window = int(pol.get('window_size', 0))
                if slots > 0 and window > 0:
                    nk2, nv2 = [], []
                    for (k_arr, v_arr) in zip(nk, nv):
                        k = k_arr
                        v = v_arr
                        B, Hh, Tt, DLd = k.shape
                        if Tt > window:
                            old_len = max(0, Tt - window)
                            base = old_len // slots
                            rem = old_len % slots
                            start = 0
                            k_segs = []
                            v_segs = []
                            for si in range(slots):
                                end = start + base + (1 if si < rem else 0)
                                if end <= start:
                                    seg_k = k[:, :, :1, :]
                                    seg_v = v[:, :, :1, :]
                                else:
                                    seg_k = k[:, :, start:end, :]
                                    seg_v = v[:, :, start:end, :]
                                    # Optional learned compression for older segments
                                    if args.kv_compress_sidecar:
                                        ae = _load_kv_autoencoder(args.kv_compress_sidecar)
                                        if ae is not None:
                                            try:
                                                seg_k = _ae_decode(_ae_encode(seg_k, ae['enc']), ae['dec'])
                                                seg_v = _ae_decode(_ae_encode(seg_v, ae['enc']), ae['dec'])
                                            except Exception:
                                                pass
                                    # Fallback: average pool compressed segment to 1 frame
                                    seg_k = seg_k.mean(axis=2, keepdims=True)
                                    seg_v = seg_v.mean(axis=2, keepdims=True)
                                k_segs.append(seg_k)
                                v_segs.append(seg_v)
                                start = end
                            k = np.concatenate(k_segs + [k[:, :, -window:, :]], axis=2)
                            v = np.concatenate(v_segs + [v[:, :, -window:, :]], axis=2)
                        nk2.append(k)
                        nv2.append(v)
                    nk, nv = nk2, nv2
            except Exception as _e:
                print('[warn] retention sidecar warmup ignored:', _e)
        # Consume any MTP outputs during warmup (ignored for now)
        # Optional storage quantization for warmup caches (u8/NF4 emulation)
        # Apply optional per-token logit bias before acceptance/sampling
        try:
            _apply_logit_bias_inplace(logits[0, -1, :], bias_map, float(args.logit_bias_alpha))
        except Exception:
            pass
        if args.kvq in ('u8','nf4'):
            qk, qv = [], []
            for i in range(num_layers):
                k = nk[i].astype(np.float32)
                v = nv[i].astype(np.float32)
                # Use per-head per-group calibration if provided; else compute groupwise dynamically
                if kvq_stats and 'group' in kvq_stats:
                    group = int(kvq_stats.get('group', args.kvq_group))
                else:
                    group = args.kvq_group
                if args.kvq == 'u8':
                    ks, kz = _compute_groupwise_scale_zero(k, group)
                    vs, vz = _compute_groupwise_scale_zero(v, group)
                    kq, ks, kz = _quantize_uint8(k, ks, kz)
                    vq, vs, vz = _quantize_uint8(v, vs, vz)
                    qk.append((kq, ks, kz))
                    qv.append((vq, vs, vz))
                else:
                    km, ks = _compute_groupwise_mean_std(k, group)
                    vm, vs = _compute_groupwise_mean_std(v, group)
                    kq, km, ks = _quantize_nf4(k, km, ks)
                    vq, vm, vs = _quantize_nf4(v, vm, vs)
                    qk.append((kq, km, ks))
                    qv.append((vq, vm, vs))
            caches_k = qk  # store tuples (q, scale, zero) or (q, mean, std)
            caches_v = qv
        elif kv_paged_sidecar is not None:
            # Append as new pages
            for i in range(num_layers):
                caches_k[i].append(nk[i])
                caches_v[i].append(nv[i])
        else:
            caches_k = nk
            caches_v = nv
            if _feed_base is not None:
                # Refresh base dict references to new arrays
                for i in range(num_layers):
                    _feed_base[k_inputs[i].name] = caches_k[i]
                    _feed_base[v_inputs[i].name] = caches_v[i]

    generated = []
    last_id = None
    # Generate new tokens
    # Reuse feed dict in non-quantized, non-paged explicit-KV path during generation
    _gen_feed_base = None
    if (_has_explicit_kv) and (kv_paged_sidecar is None) and (args.kvq not in ("u8", "nf4")):
        _gen_feed_base = {k_inputs[i].name: caches_k[i] for i in range(num_layers)}
        _gen_feed_base.update({v_inputs[i].name: caches_v[i] for i in range(num_layers)})
        _gen_feed_base[input_name] = None
    _tok_buf2 = None
    for _ in range(args.max_new_tokens):
        # Use last token from either prompt or previous generation
        if last_id is None:
            last_token = input_ids[:, -1:]
        else:
            if _tok_buf2 is None:
                _tok_buf2 = np.empty((1, 1), dtype=np.int64)
            _tok_buf2[0, 0] = last_id
            last_token = _tok_buf2

        # Dequantize-on-feed if cached in u8/nf4
        if args.kvq in ('u8','nf4') and isinstance(caches_k[0], tuple):
            feed = {"input_ids": last_token}
            for i in range(num_layers):
                if args.kvq == 'u8':
                    kq, ks, kz = caches_k[i]
                    vq, vs, vz = caches_v[i]
                    feed[k_inputs[i].name] = _dequantize_uint8(kq, ks, kz)
                    feed[v_inputs[i].name] = _dequantize_uint8(vq, vs, vz)
                else:
                    kq, km, ks = caches_k[i]
                    vq, vm, vs = caches_v[i]
                    feed[k_inputs[i].name] = _dequantize_nf4(kq, km, ks)
                    feed[v_inputs[i].name] = _dequantize_nf4(vq, vm, vs)
        elif kv_paged_sidecar is not None:
            feed = {"input_ids": last_token}
            from omnicoder.modeling.kernels.kv_paged_ops import concat_kv_window
            for i in range(num_layers):
                pages_k = caches_k[i]
                pages_v = caches_v[i]
                if pages_k:
                    k_cat, v_cat = concat_kv_window(pages_k, pages_v, int(args.window) if args.window and args.window > 0 else None)
                else:
                    heads, d_lat = meta[i]
                    if zeros_cached is not None:
                        k_cat = zeros_cached((1, heads, 0, d_lat), np.float32)  # type: ignore[arg-type]
                        v_cat = zeros_cached((1, heads, 0, d_lat), np.float32)  # type: ignore[arg-type]
                    else:
                        k_cat = np.zeros((1, heads, 0, d_lat), dtype=np.float32)
                        v_cat = np.zeros((1, heads, 0, d_lat), dtype=np.float32)
                feed[k_inputs[i].name] = k_cat
                feed[v_inputs[i].name] = v_cat
        else:
            if _gen_feed_base is not None:
                _gen_feed_base[input_name] = last_token
                feed = _gen_feed_base
            else:
                feed = {"input_ids": last_token}
                for i in range(num_layers):
                    feed[k_inputs[i].name] = caches_k[i]
                for i in range(num_layers):
                    feed[v_inputs[i].name] = caches_v[i]

        out = sess.run(output_names, feed)
        logits = out[0]  # (1, 1, V)
        nk = out[1:1+num_layers]
        nv = out[1+num_layers:1+2*num_layers]
        # Optional KV retention/compression during generation
        if args.kv_retention_sidecar:
            try:
                import json as _json
                pol = _json.loads(open(args.kv_retention_sidecar,'r',encoding='utf-8').read())
                slots = int(pol.get('compressive_slots', 0))
                window = int(pol.get('window_size', 0))
                if slots > 0 and window > 0:
                    nk2, nv2 = [], []
                    for (k_arr, v_arr) in zip(nk, nv):
                        k = k_arr
                        v = v_arr
                        B, Hh, Tt, DLd = k.shape
                        if Tt > window:
                            old_len = max(0, Tt - window)
                            base = old_len // slots
                            rem = old_len % slots
                            start = 0
                            k_segs = []
                            v_segs = []
                            for si in range(slots):
                                end = start + base + (1 if si < rem else 0)
                                if end <= start:
                                    seg_k = k[:, :, :1, :]
                                    seg_v = v[:, :, :1, :]
                                else:
                                    seg_k = k[:, :, start:end, :]
                                    seg_v = v[:, :, start:end, :]
                                    if args.kv_compress_sidecar:
                                        ae = _load_kv_autoencoder(args.kv_compress_sidecar)
                                        if ae is not None:
                                            try:
                                                seg_k = _ae_decode(_ae_encode(seg_k, ae['enc']), ae['dec'])
                                                seg_v = _ae_decode(_ae_encode(seg_v, ae['enc']), ae['dec'])
                                            except Exception:
                                                pass
                                    seg_k = seg_k.mean(axis=2, keepdims=True)
                                    seg_v = seg_v.mean(axis=2, keepdims=True)
                                k_segs.append(seg_k)
                                v_segs.append(seg_v)
                                start = end
                            k = np.concatenate(k_segs + [k[:, :, -window:, :]], axis=2)
                            v = np.concatenate(v_segs + [v[:, :, -window:, :]], axis=2)
                        nk2.append(k)
                        nv2.append(v)
                    nk, nv = nk2, nv2
            except Exception as _e:
                print('[warn] retention sidecar gen ignored:', _e)
        # Speculative acceptance using MTP heads (Medusa-style) without verifier head
        accepted_ids = []
        # Optional: block verify acceptance for MTP drafts
        block_verify = (os.getenv("OMNICODER_BLOCK_VERIFY", "0") == "1")
        block_size = int(os.getenv("OMNICODER_BLOCK_VERIFY_SIZE", "0"))
        if args.speculative_draft_len and mtp_names:
            # extract up to N lookahead logits in order
            mtp_logits = []
            max_accept = min(args.speculative_draft_len, len(mtp_names))
            if block_verify and block_size > 0:
                max_accept = min(max_accept, block_size)
            for i in range(max_accept):
                try:
                    idx = _mtp_indices[i]
                    if idx is None:
                        break
                    mtp_logits.append(out[idx])
                except Exception:
                    break
            # Prefer in-graph acceptance when exporter emitted top1 and accept flags
            if _mtp_top1_idx and _mtp_acc_idx and len(_mtp_top1_idx) >= max_accept and len(_mtp_acc_idx) >= max_accept:
                for i in range(max_accept):
                    tidx = _mtp_top1_idx[i]; aidx = _mtp_acc_idx[i]
                    if tidx is None or aidx is None:
                        break
                    top1 = out[tidx]
                    acc = out[aidx]
                    # Expect shapes (B,1) for ids and (B,1) for accept flags
                    if bool(acc[0, -1]):
                        accepted_ids.append(int(top1[0, -1]))
                    else:
                        break
            else:
                # Fallback acceptance using base vs lookahead probabilities
                if use_torch_post:
                    base = torch.from_numpy(logits[0, -1, :])
                    base_probs = torch.softmax(base - base.max(), dim=0)
                    for la in mtp_logits:
                        la_t = torch.from_numpy(la[0, -1, :])
                        la_probs = torch.softmax(la_t - la_t.max(), dim=0)
                        draft_id = int(torch.argmax(la_probs))
                        if args.verify_threshold > 0.0:
                            if float(base_probs[draft_id].item()) < float(args.verify_threshold):
                                break
                        accepted_ids.append(draft_id)
                else:
                    base_probs = np.exp(logits[0, -1, :] - np.max(logits[0, -1, :]))
                    base_probs /= max(base_probs.sum(), 1e-12)
                    for la in mtp_logits:
                        la_probs = np.exp(la[0, -1, :] - np.max(la[0, -1, :]))
                        la_probs /= max(la_probs.sum(), 1e-12)
                        draft_id = int(np.argmax(la_probs))
                        if args.verify_threshold > 0.0:
                            if float(base_probs[draft_id]) < float(args.verify_threshold):
                                break
                        accepted_ids.append(draft_id)
        if args.kvq in ('u8','nf4'):
            qk, qv = [], []
            for i in range(num_layers):
                k = nk[i].astype(np.float32)
                v = nv[i].astype(np.float32)
                if kvq_stats and 'group' in kvq_stats:
                    group = int(kvq_stats.get('group', args.kvq_group))
                else:
                    group = args.kvq_group
                if args.kvq == 'u8':
                    ks, kz = _compute_groupwise_scale_zero(k, group)
                    vs, vz = _compute_groupwise_scale_zero(v, group)
                    kq, ks, kz = _quantize_uint8(k, ks, kz)
                    vq, vs, vz = _quantize_uint8(v, vs, vz)
                    qk.append((kq, ks, kz))
                    qv.append((vq, vs, vz))
                else:
                    km, ks = _compute_groupwise_mean_std(k, group)
                    vm, vs = _compute_groupwise_mean_std(v, group)
                    kq, km, ks = _quantize_nf4(k, km, ks)
                    vq, vm, vs = _quantize_nf4(v, vm, vs)
                    qk.append((kq, km, ks))
                    qv.append((vq, vm, vs))
            caches_k = qk
            caches_v = qv
        elif kv_paged_sidecar is not None:
            for i in range(num_layers):
                caches_k[i].append(nk[i])
                caches_v[i].append(nv[i])
        else:
            caches_k = nk
            caches_v = nv
            if _gen_feed_base is not None:
                for i in range(num_layers):
                    _gen_feed_base[k_inputs[i].name] = caches_k[i]
                    _gen_feed_base[v_inputs[i].name] = caches_v[i]
        # Enforce simple KV spill to bound memory if paging sidecar exists
        try:
            if kv_paged_sidecar is not None:
                page_len = int(kv_paged_sidecar.get('page_len', 0))
                if page_len:
                    # Mixed precision spill: optionally store older pages in lower precision to reduce RAM
                    spill_mp = str(os.getenv("OMNICODER_KV_SPILL_PREC", "")).strip()  # choices: '', 'fp16', 'bf16'
                    # Optional env fallback for keep pages when no predictor is present
                    try:
                        keep_env = int(os.getenv("OMNICODER_KV_PREFETCH_KEEP", "0"))
                    except Exception:
                        keep_env = 0
                    keep = max(1, int(args.window or (page_len * 4)))
                    if keep_env > 0:
                        keep = max(keep, keep_env)
                    for i in range(num_layers):
                        if isinstance(caches_k[i], list):
                            # list of pages: keep last N
                            if len(caches_k[i]) > keep:
                                old_k = caches_k[i][:-keep]
                                old_v = caches_v[i][:-keep]
                                if spill_mp in ("fp16","bf16"):
                                    try:
                                        tgt = np.float16 if spill_mp == "fp16" else np.float16  # numpy lacks bf16 in older versions; use f16 as proxy
                                        old_k = [ok.astype(tgt, copy=False) for ok in old_k]
                                        old_v = [ov.astype(tgt, copy=False) for ov in old_v]
                                    except Exception:
                                        pass
                                caches_k[i] = caches_k[i][-keep:]
                                caches_v[i] = caches_v[i][-keep:]
                            # Optionally downcast retained pages except the newest for reduced RAM
                            if spill_mp in ("fp16","bf16") and len(caches_k[i]) > 1:
                                try:
                                    tgt = np.float16 if spill_mp == "fp16" else np.float16
                                    for j in range(0, len(caches_k[i]) - 1):
                                        if hasattr(caches_k[i][j], 'astype'):
                                            caches_k[i][j] = caches_k[i][j].astype(tgt, copy=False)
                                        if hasattr(caches_v[i][j], 'astype'):
                                            caches_v[i][j] = caches_v[i][j].astype(tgt, copy=False)
                                except Exception:
                                    pass
                        else:
                            # tensors: crop time dimension
                            if caches_k[i].shape[2] > keep:
                                caches_k[i] = caches_k[i][:, :, -keep:, :]
                                caches_v[i] = caches_v[i][:, :, -keep:, :]
        except Exception:
            pass

        if accepted_ids:
            generated.extend(accepted_ids)
            last_id = accepted_ids[-1]
        else:
            # Optional: ORT draft model proposes a token, verify against base probs
            if draft_sess is not None:
                try:
                    d_out_names = [o.name for o in draft_sess.get_outputs()]
                    d_feed = {draft_sess.get_inputs()[0].name: last_token}
                    d_out = draft_sess.run(d_out_names, d_feed)
                    d_logits = d_out[0]
                    if use_torch_post:
                        d_last = torch.from_numpy(d_logits[0, -1, :])
                        draft_id = int(torch.argmax(d_last))
                        base = torch.from_numpy(logits[0, -1, :])
                        base_probs = torch.softmax(base - base.max(), dim=0)
                        if float(base_probs[draft_id].item()) >= float(args.draft_verify_threshold):
                            next_id = draft_id
                        else:
                            # sample with torch
                            scaled = base / max(args.temperature, 1e-5)
                            if args.top_k > 0:
                                topk = torch.topk(scaled, k=int(args.top_k)).indices
                                probs = torch.softmax(scaled[topk], dim=0)
                                choice = int(torch.multinomial(probs, num_samples=1))
                                next_id = int(topk[choice])
                            elif 0 < float(args.top_p) < 1:
                                sorted_vals, sorted_idx = torch.sort(scaled, descending=True)
                                probs = torch.softmax(sorted_vals, dim=0)
                                cum = torch.cumsum(probs, dim=0)
                                cutoff = int(torch.searchsorted(cum, torch.tensor(float(args.top_p))) + 1)
                                probs_slice = probs[:cutoff]
                                probs_slice = probs_slice / max(float(probs_slice.sum().item()), 1e-12)
                                choice = int(torch.multinomial(probs_slice, num_samples=1))
                                next_id = int(sorted_idx[choice])
                            else:
                                next_id = int(torch.argmax(scaled))
                    else:
                        draft_id = int(np.argmax(d_logits[0, -1, :]))
                        base_probs = np.exp(logits[0, -1, :] - np.max(logits[0, -1, :]))
                        base_probs /= max(base_probs.sum(), 1e-12)
                        if float(base_probs[draft_id]) >= float(args.draft_verify_threshold):
                            next_id = draft_id
                        else:
                            next_id = _sample_next_token(logits[0, -1, :], args.temperature, args.top_k, args.top_p)
                except Exception:
                    next_id = _sample_next_token(logits[0, -1, :], args.temperature, args.top_k, args.top_p)
            else:
                # Optional small tree search: sample multiple candidates and choose the best under base probs
                if int(args.tree_width) and int(args.tree_width) > 1:
                    if use_torch_post:
                        base = torch.from_numpy(logits[0, -1, :])
                        base_probs = torch.softmax(base - base.max(), dim=0)
                        best = int(torch.argmax(base)) if (args.top_k <= 0 and not (0 < float(args.top_p) < 1)) else int(torch.argmax(base))
                        best_score = float(base_probs[best].item())
                        for _ in range(max(0, int(args.tree_width) - 1)):
                            scaled = base / max(args.temperature, 1e-5)
                            if args.top_k > 0:
                                topk = torch.topk(scaled, k=int(args.top_k)).indices
                                probs = torch.softmax(scaled[topk], dim=0)
                                choice = int(torch.multinomial(probs, num_samples=1))
                                cand = int(topk[choice])
                            elif 0 < float(args.top_p) < 1:
                                sorted_vals, sorted_idx = torch.sort(scaled, descending=True)
                                probs = torch.softmax(sorted_vals, dim=0)
                                cum = torch.cumsum(probs, dim=0)
                                cutoff = int(torch.searchsorted(cum, torch.tensor(float(args.top_p))) + 1)
                                probs_slice = probs[:cutoff]
                                probs_slice = probs_slice / max(float(probs_slice.sum().item()), 1e-12)
                                choice = int(torch.multinomial(probs_slice, num_samples=1))
                                cand = int(sorted_idx[choice])
                            else:
                                cand = int(torch.argmax(scaled))
                            score = float(base_probs[cand].item())
                            if score > best_score:
                                best = cand
                                best_score = score
                        next_id = best
                    else:
                        base = logits[0, -1, :]
                        base_probs = np.exp(base - np.max(base))
                        base_probs /= max(base_probs.sum(), 1e-12)
                        best = _sample_next_token(base, args.temperature, args.top_k, args.top_p)
                        best_score = float(base_probs[best])
                        for _ in range(max(0, int(args.tree_width) - 1)):
                            cand = _sample_next_token(base, args.temperature, args.top_k, args.top_p)
                            score = float(base_probs[cand])
                            if score > best_score:
                                best = cand
                                best_score = score
                        next_id = best
                else:
                    if use_torch_post:
                        base = torch.from_numpy(logits[0, -1, :])
                        scaled = base / max(args.temperature, 1e-5)
                        if args.top_k > 0:
                            topk = torch.topk(scaled, k=int(args.top_k)).indices
                            probs = torch.softmax(scaled[topk], dim=0)
                            choice = int(torch.multinomial(probs, num_samples=1))
                            next_id = int(topk[choice])
                        elif 0 < float(args.top_p) < 1:
                            sorted_vals, sorted_idx = torch.sort(scaled, descending=True)
                            probs = torch.softmax(sorted_vals, dim=0)
                            cum = torch.cumsum(probs, dim=0)
                            cutoff = int(torch.searchsorted(cum, torch.tensor(float(args.top_p))) + 1)
                            probs_slice = probs[:cutoff]
                            probs_slice = probs_slice / max(float(probs_slice.sum().item()), 1e-12)
                            choice = int(torch.multinomial(probs_slice, num_samples=1))
                            next_id = int(sorted_idx[choice])
                        else:
                            next_id = int(torch.argmax(scaled))
                    else:
                        next_id = _sample_next_token(logits[0, -1, :], args.temperature, args.top_k, args.top_p)
            generated.append(next_id)
            last_id = next_id

    full_ids = input_ids[0].tolist() + generated
    text = tok.decode(full_ids)
    print(text)


if __name__ == "__main__":
    main()


