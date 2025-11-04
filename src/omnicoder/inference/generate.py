import argparse
import os
import logging
from typing import Optional, Callable, Any
from collections import deque
from dataclasses import dataclass
import time as _t

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
from omnicoder.inference.tool_use import build_default_registry as _build_tool_registry  # type: ignore
try:
    from omnicoder.inference.tool_fabric import analyze_inputs_for_nss
except Exception:
    analyze_inputs_for_nss = None  # type: ignore
try:
    from omnicoder.modeling.multimodal.aligner import CrossModalVerifier  # type: ignore
except Exception:
    CrossModalVerifier = None  # type: ignore

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.config import MobilePreset, get_mobile_preset, get_rope_scale_for_target_ctx, get_rope_interp_base
from omnicoder.training.simple_tokenizer import get_text_tokenizer
from omnicoder.inference.retrieval import LocalRetriever
try:
    from omnicoder.reasoning.omega_controller import OmegaController, OmegaConfig  # type: ignore
except Exception:
    OmegaController, OmegaConfig = None, None  # type: ignore
try:
    from omnicoder.reasoning.omega_intent import infer_goals as _omega_infer_goals  # type: ignore
except Exception:
    _omega_infer_goals = None  # type: ignore
try:
    from omnicoder.reasoning.omega_causal import build_minimal_scm_for_query as _omega_build_scm, abductive_score as _omega_abd_score  # type: ignore
except Exception:
    _omega_build_scm, _omega_abd_score = None, None  # type: ignore
try:
    from omnicoder.reasoning.omega_verifier import compute_margin as _omega_compute_margin  # type: ignore
except Exception:
    _omega_compute_margin = None  # type: ignore
try:
    from omnicoder.reasoning.omega_pca import pack as _omega_pack_cert, to_json as _omega_cert_json  # type: ignore
except Exception:
    _omega_pack_cert, _omega_cert_json = None, None  # type: ignore
try:
    # Lightweight graph-speculative selector and unified constraint margin
    from omnicoder.reasoning.graph_speculative import select_branch  # type: ignore
except Exception:
    select_branch = None  # type: ignore
# New adaptive reasoning components (optional)
try:
    from omnicoder.reasoning import (
        build_agot,            # type: ignore
        build_latent_bfs,      # type: ignore
        build_reflection,      # type: ignore
        build_symbolic_planner # type: ignore
    )
except Exception:
    build_agot = None  # type: ignore
    build_latent_bfs = None  # type: ignore
    build_reflection = None  # type: ignore
    build_symbolic_planner = None  # type: ignore
try:
    from omnicoder.retrieval.graphrag import build_graphrag  # type: ignore
except Exception:
    build_graphrag = None  # type: ignore
try:
    from omnicoder.reasoning import constraints as _omega_constraints  # type: ignore
except Exception:
    _omega_constraints = None  # type: ignore
try:
    from omnicoder.inference.retrieval_pq import PqRetriever
except Exception:
    PqRetriever = None  # type: ignore
from omnicoder.modeling.quant.kv_cache import quantize_kv, dequantize_kv, KvQuantMeta
from omnicoder.inference.knn_cache import KNNCache
from omnicoder.utils.resources import apply_thread_env_if_auto
import json as _json  # for retention/compress sidecar
import torch.nn as nn
import torch.nn.functional as F
try:
    from omnicoder.inference.retrieval_memory import ExternalRetrievalMemory
except Exception:
    ExternalRetrievalMemory = None  # type: ignore

# Cache for per-step logit bias to avoid repeated disk I/O and JSON parsing
_LOGIT_BIAS_CACHE = {
    'path': '',
    'mtime': 0.0,
    'map': None,
}
from omnicoder.utils.logger import get_logger
try:
    from omnicoder.modeling.diffusion_text import DiffusionTextGenerator  # type: ignore
except Exception:
    DiffusionTextGenerator = None  # type: ignore

# --- Hot-path guards: cache all env reads and file IO at import time ---
try:
    import os as _os, json as _json
    # Long-context defaults cached once (no env reads during inference)
    _LONGCTX_DEFAULT = (_os.getenv('OMNICODER_LONGCTX', '1') == '1')
    try:
        _RAG_TOPK_DEFAULT = int(_os.getenv('OMNICODER_RAG_TOPK', '3'))
    except Exception:
        _RAG_TOPK_DEFAULT = 3
    _RAG_ROOT_DEFAULT = _os.getenv('OMNICODER_RAG_ROOT', 'weights/unified_index')
    # KVQ calibration cached once to avoid runtime IO
    _KVQ_CAL_META = None
    _kvq_path = _os.environ.get('OMNICODER_KVQ_CALIBRATION', '').strip()
    if not _kvq_path:
        cand1 = 'weights/kvq_calibration.json'
        if _os.path.exists(cand1):
            _kvq_path = cand1
    if _kvq_path and _os.path.exists(_kvq_path):
        try:
            _KVQ_CAL_META = _json.loads(open(_kvq_path, 'r', encoding='utf-8').read())
        except Exception:
            _KVQ_CAL_META = None
except Exception:
    _LONGCTX_DEFAULT = True
    _RAG_TOPK_DEFAULT = 3
    _RAG_ROOT_DEFAULT = 'weights/unified_index'
    _KVQ_CAL_META = None
try:
    from omnicoder.reasoning.certificate import Omega2Cert as _Omega2Cert, emit as _emit_cert  # type: ignore
except Exception:
    _Omega2Cert, _emit_cert = None, None  # type: ignore

# Persistent cache for SFB/GraphRAG/Reflection builders to avoid re-initializing per generate()
_SFB_GLOBAL: dict[str, object] = {}

# Global draft model cache for IO-free hot path usage. Runners/API should set this at init.
_DRAFT_GLOBAL: dict[str, object] = {}

def set_global_draft_model(model: object) -> None:
	try:
		_DRAFT_GLOBAL['torch'] = model
	except Exception:
		pass

def get_global_draft_model() -> object | None:
	try:
		return _DRAFT_GLOBAL.get('torch')
	except Exception:
		return None

# Remove NumPy from inference hot paths; ONNX bridge will convert via torch->numpy only at call sites
_np = None  # type: ignore
try:
    from omnicoder.inference.onnx_utils import zeros_cached  # type: ignore
except Exception:
    zeros_cached = None  # type: ignore

# Optional ONNX runtime for draft decode-step acceleration
try:
    import onnxruntime as _ort  # type: ignore
except Exception:
    _ort = None  # type: ignore

class _OnnxDraftWrapper:
    def __init__(self, model_path: str, provider: str | None = None):
        if _ort is None:
            raise RuntimeError("onnxruntime not available")
        provs = None
        if provider and provider != "auto":
            provs = [provider]
        else:
            try:
                avail = list(_ort.get_available_providers())
            except Exception:
                avail = []
            pref = ['CUDAExecutionProvider','DmlExecutionProvider','CPUExecutionProvider']
            provs = [p for p in pref if p in avail] or ["CPUExecutionProvider"]
        self.sess = _ort.InferenceSession(model_path, providers=provs)  # type: ignore
        outs = self.sess.get_outputs()
        self.logits_name = outs[0].name
        ins = self.sess.get_inputs()
        # Canonical decode-step schema: input_ids + per-layer K/V inputs
        self.input_ids_name = ins[0].name
        # Discover K/V input names (sorted by layer index)
        kv_in_names = [i.name for i in ins[1:]]
        self.k_in = sorted([n for n in kv_in_names if n.startswith('k_lat_')], key=lambda x: int(x.split('_')[-1]))
        self.v_in = sorted([n for n in kv_in_names if n.startswith('v_lat_')], key=lambda x: int(x.split('_')[-1]))
        # Capture input shapes for each K/V tensor when available
        def _shape_for(name: str):
            for i in ins:
                if i.name == name:
                    # Try direct shape list
                    shp = getattr(i, 'shape', None)
                    if isinstance(shp, (list, tuple)) and shp:
                        return list(shp)
                    # Fallback: parse from tensor_type dims
                    try:
                        t = getattr(i, 'type', None)
                        tt = getattr(t, 'tensor_type', None)
                        ss = getattr(tt, 'shape', None)
                        dims = getattr(ss, 'dim', None)
                        if dims:
                            out = []
                            for d in dims:
                                v = getattr(d, 'dim_value', None)
                                if v is None or int(v) == 0:
                                    out.append(None)
                                else:
                                    out.append(int(v))
                            return out
                    except Exception:
                        pass
                    return []
        self.k_shapes = [_shape_for(n) for n in self.k_in]
        self.v_shapes = [_shape_for(n) for n in self.v_in]
        # Discover K/V output names to persist cache between calls
        out_names = [o.name for o in outs[1:]]
        self.k_out = sorted([n for n in out_names if n.startswith('nk_lat_')], key=lambda x: int(x.split('_')[-1]))
        self.v_out = sorted([n for n in out_names if n.startswith('nv_lat_')], key=lambda x: int(x.split('_')[-1]))
        # Capture output shapes to resolve latent dims when input dims are dynamic
        def _out_shape_for(name: str):
            for o in outs:
                if o.name == name:
                    shp = getattr(o, 'shape', None)
                    if isinstance(shp, (list, tuple)) and shp:
                        return list(shp)
            return []
        self.k_out_shapes = [_out_shape_for(n) for n in self.k_out]
        self.v_out_shapes = [_out_shape_for(n) for n in self.v_out]
        # Persistent last K/V (numpy arrays) carried across calls
        self.last_kv = None  # type: ignore[var-annotated]
        # Fallback shapes when ONNX lacks concrete dims
        # Favor model metadata over env; defaults are last resort
        self.fallback_heads = 16
        self.fallback_latent = 64
        # Use shared zero-array cache for ONNX feeds to avoid per-call allocations
        self._zero_cache: dict[tuple[str, tuple[int, int, int, int], str], object] = {}

        # Resolve per-layer effective dims, preferring input dims and using the safer (smaller) dim if I/O disagree
        def _resolve_dims(in_shape: list | None, out_shape: list | None) -> tuple[int, int]:
            def _as_int(v, default=None):
                try:
                    return int(v)
                except Exception:
                    return default
            H_in = _as_int(in_shape[1], None) if in_shape and len(in_shape) > 1 else None
            DL_in = _as_int(in_shape[3], None) if in_shape and len(in_shape) > 3 else None
            H_out = _as_int(out_shape[1], None) if out_shape and len(out_shape) > 1 else None
            DL_out = _as_int(out_shape[3], None) if out_shape and len(out_shape) > 3 else None
            H = H_in if H_in is not None else (H_out if H_out is not None else self.fallback_heads)
            DL_candidates = [d for d in (DL_in, DL_out, self.fallback_latent) if isinstance(d, int)]
            # If both present and differ (e.g., Expected:64 vs Got:128), choose the smaller to avoid INVALID_ARGUMENT
            DL = min(DL_candidates) if DL_candidates else self.fallback_latent
            return int(H), int(DL)
        self._resolved_k_dims = [
            _resolve_dims(
                self.k_shapes[i] if i < len(self.k_shapes) else [],
                self.k_out_shapes[i] if i < len(self.k_out_shapes) else [],
            ) for i in range(max(len(self.k_in), len(self.k_out)))
        ]
        self._resolved_v_dims = [
            _resolve_dims(
                self.v_shapes[i] if i < len(self.v_shapes) else [],
                self.v_out_shapes[i] if i < len(self.v_out_shapes) else [],
            ) for i in range(max(len(self.v_in), len(self.v_out)))
        ]

    def __call__(self, input_ids: torch.Tensor, past_kv=None, use_cache: bool = False):  # noqa: ANN001
        # Build feed: input_ids + per-layer K/V (zeros on first call)
        # Convert to CPU NumPy only at the boundary
        # Convert to NumPy at the boundary without importing numpy at module scope
        try:
            import numpy as _np  # type: ignore
        except Exception as _e:
            raise RuntimeError(f"onnxruntime bridge requires numpy: {_e}")
        ids = input_ids.detach().to('cpu').numpy()
        feed = {self.input_ids_name: ids}
        if self.last_kv is None:
            # Initialize minimal zero caches with T=1
            B = ids.shape[0] if hasattr(ids, 'shape') else 1
            T = 1
            def _dim_int(x):
                try:
                    return int(x)
                except Exception:
                    return None
            for idx, name in enumerate(self.k_in):
                H, DL = self._resolved_k_dims[idx] if idx < len(self._resolved_k_dims) else (self.fallback_heads, self.fallback_latent)
                if zeros_cached is not None and _np is not None:
                    feed[name] = zeros_cached((B, H, T, DL), _np.float32)  # type: ignore[arg-type]
                else:
                    feed[name] = _np.zeros((B, H, T, DL), dtype=_np.float32)
            for idx, name in enumerate(self.v_in):
                H, DL = self._resolved_v_dims[idx] if idx < len(self._resolved_v_dims) else (self.fallback_heads, self.fallback_latent)
                if zeros_cached is not None and _np is not None:
                    feed[name] = zeros_cached((B, H, T, DL), _np.float32)  # type: ignore[arg-type]
                else:
                    feed[name] = _np.zeros((B, H, T, DL), dtype=_np.float32)
        else:
            k_list, v_list = self.last_kv
            for name, arr in zip(self.k_in, k_list):
                feed[name] = arr
            for name, arr in zip(self.v_in, v_list):
                feed[name] = arr
        # Run and capture logits + new K/V
        fetch = [self.logits_name] + self.k_out + self.v_out
        out = self.sess.run(fetch, feed)
        logits_np = out[0]
        k_np = out[1:1+len(self.k_out)]
        v_np = out[1+len(self.k_out):]
        # Persist for next call
        self.last_kv = (k_np, v_np)
        logits = torch.from_numpy(logits_np)
        return (logits.to(device=input_ids.device) if input_ids.is_cuda else logits,)

# NEW: Reflection + Self-Validation integration (always available; opt-in via kwargs)
# We deliberately avoid any environment reads inside the hot path. Callers pass the
# runtime_config or explicit flags to enable. This module-level helper performs a
# short reflection pass using the model's verifier head and optional tool registry
# to validate or amend the last answer before returning. This is designed to be
# called outside the inner token loop (e.g., once when generation is concluded or
# at block-verify boundaries) to keep TPS unaffected during streaming.
def _reflect_and_validate(
    model: OmniTransformer,
    last_hidden: torch.Tensor | None,
    logits: torch.Tensor,
    tokenizer_decode: Callable[[list[int]], str] | None,
    tool_registry_builder: Callable[[], Any] | None,
    prompt_ids: torch.Tensor,
    generated_ids: list[int],
    enable_tools: bool,
    verify_min_prob: float,
) -> tuple[list[int], dict]:
    """
    Reflection helper: uses model.verifier_head (if present) to compute a simple
    acceptance signal for the final token(s), and optionally invokes tool-use
    registry for structured self-check based on inline tags in the decoded text.
    Returns potentially amended token list and a debug stats dict.
    """
    _log = get_logger("omnicoder.reflect")
    stats: dict = {}
    try:
        # Decode current text cheaply if decoder provided (no IO, no env)
        text = None
        if tokenizer_decode is not None:
            try:
                text = tokenizer_decode((prompt_ids[0].tolist() if prompt_ids.ndim == 2 else []) + generated_ids)
            except Exception:
                text = None
        if last_hidden is not None and hasattr(model, 'verifier_head') and getattr(model, 'verifier_head') is not None:
            try:
                # Compute verifier probabilities on last position
                vh = model.verifier_head(last_hidden)
                v_probs = torch.ops.aten._softmax.default(vh[:, -1, :], -1, False)
                last_id = generated_ids[-1] if generated_ids else None
                v_ok = False
                if last_id is not None and int(last_id) < v_probs.shape[-1]:
                    v_ok = float(v_probs[0, int(last_id)].item()) >= float(verify_min_prob)
                stats['verifier_prob_ok'] = bool(v_ok)
                stats['verifier_prob'] = float(v_probs[0, int(last_id)].item()) if last_id is not None else 0.0
                # Optional trivial revision: pick argmax under verifier if below threshold
                if (not v_ok) and (last_id is not None):
                    alt = int(torch.argmax(v_probs[0]).item())
                    if alt != int(last_id):
                        generated_ids[-1] = alt
                        stats['verifier_replaced'] = True
            except Exception as e:
                stats['verifier_error'] = str(e)
        # Optional tool self-check: parse inline <tool:...> tags and replace with JSON
        if enable_tools and (tool_registry_builder is not None) and (text is not None):
            try:
                reg = tool_registry_builder()
                repl = reg.parse_and_invoke_all(text)
                stats['tool_invocations'] = {k: (str(type(v)) if len(str(v)) > 256 else v) for k, v in repl.items()}
            except Exception as e:
                stats['tool_error'] = str(e)
        try:
            _log.debug("reflect stats=%s", str(stats))
        except Exception:
            pass
    except Exception as e:
        stats['reflect_fallback_error'] = str(e)
    return generated_ids, stats

def _inject_lora_linear(module: torch.nn.Module, r: int, alpha: int, dropout: float) -> int:
    replaced = 0
    import torch.nn as nn  # local import to avoid top-level
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
                    self.r = r
                    self.dropout = nn.Dropout(dropout)
                    self.scaling = alpha / max(r, 1)
                    self.A = nn.Linear(base.in_features, r, bias=False)
                    self.B = nn.Linear(r, base.out_features, bias=False)

                def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                    return self.base(x) + self.B(self.A(self.dropout(x))) * self.scaling

            lora = LoRALinear(base, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, leaf_name, lora)
            replaced += 1
    return replaced

# Optional module-level prefix KV cache (LRU)
_PREFIX_KV_CACHE: list[tuple[str, list[tuple[torch.Tensor, torch.Tensor, KvQuantMeta]]]] = []  # type: ignore[name-defined]
_PREFIX_KV_CACHE_CAPACITY = 8
def _prefix_kv_disk_paths(key_hash: str) -> tuple[str, str]:
    base = os.getenv('OMNICODER_PREFIX_CACHE_DIR', 'weights/prefix_kv_cache')
    try:
        from pathlib import Path as _Path
        _Path(base).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return os.path.join(base, f"{key_hash}.pt"), base

def _prefix_kv_save_disk(key_hash: str, kv_cpu: list[tuple[torch.Tensor, torch.Tensor, KvQuantMeta]]) -> None:
    if os.getenv('OMNICODER_PREFIX_CACHE_DISK', '0') != '1':
        return
    try:
        path, _ = _prefix_kv_disk_paths(key_hash)
        payload = []
        for (k, v, meta) in kv_cpu:
            payload.append((k.float().contiguous(), v.float().contiguous(), dict(vars(meta))))
        _safe_save({'kv': payload}, path)
    except Exception:
        pass

def _prefix_kv_load_disk(key_hash: str) -> list[tuple[torch.Tensor, torch.Tensor, KvQuantMeta]] | None:
    if os.getenv('OMNICODER_PREFIX_CACHE_DISK', '0') != '1':
        return None
    try:
        path, _ = _prefix_kv_disk_paths(key_hash)
        if not os.path.exists(path):
            return None
        blob = torch.load(path, map_location='cpu')
        out: list[tuple[torch.Tensor, torch.Tensor, KvQuantMeta]] = []
        for (k, v, meta_dict) in blob.get('kv', []):
            try:
                qm = KvQuantMeta(**meta_dict)
            except Exception:
                qm = KvQuantMeta(scheme='none', group=0, scale=None, zero=None, mean=None, std=None)  # type: ignore
            out.append((k, v, qm))
        return out
    except Exception:
        return None


def _maybe_retrieve(
    prompt: str,
    path: str,
    k: int,
    use_faiss: bool,
    chunk_size: int,
    stride: int,
    max_chars: int,
    system_prompt: str,
    template: str,
) -> str:
    if not path:
        try:
            get_logger("omnicoder.gen").debug("maybe_retrieve: skip (no path)")
        except Exception:
            pass
        return prompt
    try:
        get_logger("omnicoder.gen").debug("maybe_retrieve: enter path=%s k=%s faiss=%s chunk=%s stride=%s", path, k, bool(use_faiss), chunk_size, stride)
        if use_faiss:
            from omnicoder.inference.retrieval_faiss import FaissRetriever
            retriever = FaissRetriever(path)
        else:
            retriever = LocalRetriever(path, chunk_size=chunk_size, stride=stride)
        hits = retriever.search(prompt, k=max(1, k))
        context_text = "\n\n".join([f"[CTX {i+1}] {h.text}" for i, h in enumerate(hits)])
        if max_chars > 0 and len(context_text) > max_chars:
            context_text = context_text[:max_chars]
        # Simple prompt templating
        tpl = template or "[SYSTEM] {system}\n{context}\n\n[USER] {user}\n[ASSISTANT]"
        composed = tpl.format(system=system_prompt, context=context_text, user=prompt)
        try:
            get_logger("omnicoder.gen").debug("maybe_retrieve: exit composed_len=%s", len(composed))
        except Exception:
            pass
        return composed
    except Exception as _e:
        try:
            get_logger("omnicoder.gen").debug("maybe_retrieve: error %s", str(_e))
        except Exception:
            pass
        return prompt


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
	try:
		get_logger("omnicoder.gen").debug("sample_next_token: enter temp=%s top_k=%s top_p=%s logits_shape=%s", float(temperature), int(top_k), float(top_p), str(tuple(logits.shape)))
	except Exception as _e:
		get_logger("omnicoder.gen").warning("sample_next_token: log enter failed: %s", str(_e))
	# Guard against pathological logits (NaN/Inf); replace with large negative values
	try:
		if not torch.isfinite(logits).all():
			get_logger("omnicoder.gen").warning("sample_next_token: non-finite logits detected; sanitizing")
			logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, float('-inf')))
	except Exception as _e:
		get_logger("omnicoder.gen").warning("sample_next_token: nan/inf guard failed: %s", str(_e))
	# Avoid extremely small temperatures destabilizing softmax; clamp via aten
	try:
		# Build temperature scalar anchored to logits lineage (no torch.tensor/device in hot path)
		_t = torch.ops.aten.add.Scalar(
			torch.ops.aten.mul.Scalar(torch.ops.aten.sum.default(torch.ops.aten.reshape.default(logits, (-1,))), 0.0),
			float(temperature)
		)
		_t = torch.ops.aten.clamp_min.default(_t, 0.2)
		logits = torch.ops.aten.div.Tensor(logits, _t)
	except Exception:
		logits = logits
	# Stabilize logits: subtract max to improve softmax numerical stability
	try:
		m = torch.amax(logits, dim=-1, keepdim=True)
		logits = logits - m
	except Exception:
		pass
	# Verbose diagnostics: entropy and top-5 tokens before any nucleus/top-k masking
	# Reduce Python-side tensor.item()/lists during compile; keep logs lightweight
	if not torch.jit.is_tracing():
		try:
			_log = get_logger("omnicoder.gen")
			lvec = logits[0] if logits.dim() >= 2 else logits
			probs_base = torch.ops.aten._softmax.default(lvec, -1, False)
			ent = float(-(probs_base * torch.log(probs_base.clamp_min(1e-9))).sum().item())
			_log.debug("sample_next_token: pre-mask entropy=%.4f", ent)
			if not torch.isfinite(lvec).all():
				_log.warning("sample_next_token: non-finite logits detected (nan/inf present)")
		except Exception as _e:
			get_logger("omnicoder.gen").warning("sample_next_token: entropy/top5 pre-mask failed: %s", str(_e))
	# Optional bias expert: per-token logit bias from env file, cached to avoid per-step disk I/O
	try:
		bias_path = _BIAS_ENV_PATH
		bias_alpha = _BIAS_ENV_ALPHA
		if bias_path and bias_alpha > 0.0 and logits.dim() >= 2:
			_mtime = 0.0
			try:
				import os as _os
				_mtime = float(_os.path.getmtime(bias_path))
			except Exception:
				_mtime = 0.0
			# Refresh cache only when path changes or file mtime updates
			if (_LOGIT_BIAS_CACHE['path'] != bias_path) or (float(_LOGIT_BIAS_CACHE['mtime']) != _mtime) or (_LOGIT_BIAS_CACHE['map'] is None):
				try:
					with open(bias_path, 'r', encoding='utf-8') as _f:
						_bias_map = _json.loads(_f.read())
					if isinstance(_bias_map, dict):
						_LOGIT_BIAS_CACHE['path'] = bias_path
						_LOGIT_BIAS_CACHE['mtime'] = _mtime
						_LOGIT_BIAS_CACHE['map'] = _bias_map
					else:
						_LOGIT_BIAS_CACHE['map'] = None
				except Exception:
					_LOGIT_BIAS_CACHE['map'] = None
			_bias_map = _LOGIT_BIAS_CACHE['map']
			if isinstance(_bias_map, dict):
				for tk, val in list(_bias_map.items())[: 8192]:
					try:
						tid = int(tk)
						logits[..., tid] = logits[..., tid] + float(bias_alpha) * float(val)
					except Exception:
						continue
	except Exception as _e:
		get_logger("omnicoder.gen").warning("sample_next_token: bias expert failed: %s", str(_e))
	if top_k > 0:
		_V = int(logits.shape[-1])
		_kk = int(top_k) if int(top_k) > 0 else 0
		_kk = _kk if _kk <= _V else _V
		topk_vals, topk_idx = torch.ops.aten.topk.default(logits, int(_kk), -1, True, True)
		probs = torch.ops.aten._softmax.default(topk_vals, -1, False)
		# VERBOSE: Replace torch.nan_to_num with aten.nan_to_num to keep aten-only targets
		probs = torch.ops.aten.nan_to_num.default(probs, 0.0, 0.0, 0.0)
		sums = probs.sum(dim=-1, keepdim=True)
		zero_mask = (sums <= 0)
		probs = probs / sums.clamp_min(1e-8)
		if torch.any(zero_mask):
			# Fallback to uniform over local top-k when distribution is degenerate
			denom = torch.ops.aten.add.Scalar(torch.ops.aten.mul.Scalar(torch.ops.aten.sum.default(torch.ops.aten.reshape.default(probs, (-1,))), 0.0), float(probs.shape[-1]))
			probs = torch.where(zero_mask, torch.full_like(probs, 1.0 / float(int(denom))), probs)
		next_local = torch.multinomial(probs, num_samples=1)
		nxt = topk_idx.gather(-1, next_local)
		try:
			idx = int(torch.ops.aten.reshape.default(nxt, (-1,))[0].item())
			# Probability within the local top-k set
			prob_val = float(torch.ops.aten.reshape.default(probs.gather(-1, next_local), (-1,))[0].item())
			get_logger("omnicoder.gen").debug("sample_next_token: top_k path idx=%s prob=%.6f", idx, prob_val)
		except Exception as _e:
			get_logger("omnicoder.gen").warning("sample_next_token: top_k log failed: %s", str(_e))
		return nxt
	if 0 < top_p < 1:
		# Prefer topk over full sort for efficiency when only a few are needed
		_V = int(logits.shape[-1])
		_kreq = int(top_k) if int(top_k) > 0 else 0
		_k = _kreq if _kreq <= _V else _V
		if _k > 0:
			_top = torch.ops.aten.topk.default(logits, int(_k), -1, True, True)
			sorted_vals, sorted_idx = _top[0], _top[1]
			sorted_logits = sorted_vals
		else:
			# Use full topk (k=V) as a sort replacement
			_topv = torch.ops.aten.topk.default(logits, int(_V), -1, True, True)
			sorted_logits = _topv[0]
			sorted_idx = _topv[1]
		cumprobs = torch.ops.aten._softmax.default(sorted_logits, -1, False).cumsum(dim=-1)
		mask = cumprobs > top_p
		mask[..., 1:] = torch.ops.aten.mul.Scalar(mask[..., :-1], 1.0)
		mask[..., 0] = False
		sorted_logits = sorted_logits.masked_fill(mask, float('-inf'))
		probs = torch.ops.aten._softmax.default(sorted_logits, -1, False)
		probs = torch.ops.aten.nan_to_num.default(probs, 0.0, 0.0, 0.0)
		sums = probs.sum(dim=-1, keepdim=True)
		zero_mask = (sums <= 0)
		probs = probs / sums.clamp_min(1e-8)
		if torch.any(zero_mask):
			denom = torch.ops.aten.add.Scalar(torch.ops.aten.mul.Scalar(torch.ops.aten.sum.default(torch.ops.aten.reshape.default(probs, (-1,))), 0.0), float(probs.shape[-1]))
			probs = torch.where(zero_mask, torch.full_like(probs, 1.0 / float(int(denom))), probs)
		next_local = torch.multinomial(probs, num_samples=1)
		nxt = sorted_idx.gather(-1, next_local)
		try:
			idx = int(torch.ops.aten.reshape.default(nxt, (-1,))[0].item())
			prob_val = float(torch.ops.aten.reshape.default(probs.gather(-1, next_local), (-1,))[0].item())
			get_logger("omnicoder.gen").debug("sample_next_token: top_p path idx=%s prob=%.6f", idx, prob_val)
		except Exception as _e:
			get_logger("omnicoder.gen").warning("sample_next_token: top_p log failed: %s", str(_e))
		return nxt
	probs = torch.ops.aten._softmax.default(logits, -1, False)
	probs = torch.ops.aten.nan_to_num.default(probs, 0.0, 0.0, 0.0)
	sums = probs.sum(dim=-1, keepdim=True)
	zero_mask = (sums <= 0)
	probs = probs / sums.clamp_min(1e-8)
	if torch.any(zero_mask):
		denom = torch.ops.aten.add.Scalar(torch.ops.aten.mul.Scalar(torch.ops.aten.sum.default(torch.ops.aten.reshape.default(probs, (-1,))), 0.0), float(probs.shape[-1]))
		probs = torch.where(zero_mask, torch.full_like(probs, 1.0 / float(int(denom))), probs)
	# Also log the post-mask (plain softmax) top-5 when not using top-p path
	try:
		_log = get_logger("omnicoder.gen")
		lvec = probs[0] if probs.dim() >= 2 else probs
		_k5r = 5 if (hasattr(lvec, 'shape') and int(lvec.shape[-1]) >= 5) else int(lvec.shape[-1]) if hasattr(lvec, 'shape') else 1
		k5 = int(_k5r)
		_t5 = torch.ops.aten.topk.default(lvec, int(k5), -1, True, True)
		vals5, idx5 = _t5[0], _t5[1]
		top5 = [(int(idx5[i].item()), float(vals5[i].item())) for i in range(k5)]
		_log.debug("sample_next_token: softmax path top5=%s", top5)
	except Exception as _e:
		get_logger("omnicoder.gen").warning("sample_next_token: softmax top5 failed: %s", str(_e))
	nxt = torch.multinomial(probs, num_samples=1)
	try:
		idx = int(torch.ops.aten.reshape.default(nxt, (-1,))[0].item())
		prob_val = float(torch.ops.aten.reshape.default(probs, (-1,))[idx].item()) if probs.numel() > idx else float('nan')
		get_logger("omnicoder.gen").debug("sample_next_token: softmax path idx=%s prob=%.6f", idx, prob_val)
	except Exception as _e:
		get_logger("omnicoder.gen").warning("sample_next_token: softmax log failed: %s", str(_e))
	return nxt

    

from omnicoder.inference.gen_config import GenRuntimeConfig, build_runtime_config_from_env  # type: ignore

# ---- Module-level cached flags and paths to avoid getenv/IO in hot path ----
try:
	_SFB_ENABLED_DEFAULT = (os.getenv('SFB_ENABLE', os.getenv('OMNICODER_SFB_ENABLE', '1')) == '1')
except Exception:
	_SFB_ENABLED_DEFAULT = True
try:
	_SFB_AVAILABLE_FLAG = (os.getenv('SFB_AVAILABLE', '1') == '1')
except Exception:
	_SFB_AVAILABLE_FLAG = True
try:
	_BIAS_ENV_PATH = os.getenv('OMNICODER_LOGIT_BIAS_FILE', '').strip()
	_BIAS_ENV_ALPHA = float(os.getenv('OMNICODER_LOGIT_BIAS_ALPHA', '0.0'))
except Exception:
	_BIAS_ENV_PATH = ''
	_BIAS_ENV_ALPHA = 0.0
# Hoist frequently read constants to avoid hot-path env access
try:
	_SFB_ESCALATE_BLOCK_SZ_MIN = int(os.getenv('SFB_ESCALATE_BLOCK_SZ_MIN', '4'))
except Exception:
	_SFB_ESCALATE_BLOCK_SZ_MIN = 4

@torch.inference_mode()
def generate(
    model: OmniTransformer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    verify_threshold: float = 0.0,
    verifier_steps: int = 1,
    speculative_draft_len: int = 1,
    tree_width: int = 3,
    tree_depth: int = 3,
    draft_model: OmniTransformer | None = None,
    kvq: str = 'none',
    kvq_group: int = 64,
    speculative_auto: bool = False,
    block_verify: bool | None = None,
    block_verify_size: int | None = None,
    knn_cache: KNNCache | None = None,
    knn_k: int = 16,
    knn_lambda: float = 0.2,
    window_size: int = 0,
    adaptive_top_k_min: int = 1,
    adaptive_top_k_max: int = 4,
    adaptive_conf_floor: float = 0.3,
    adaptive_layer_ramp: bool = False,
    adaptive_layer_power: float = 1.0,
    scmoe_alpha: float = 0.0,
    scmoe_frac: float = 0.25,
    scmoe_topk: int = 2,
    return_stats: bool = False,
    early_exit: bool = False,
    early_exit_entropy: float = 0.0,
    early_exit_mode: str = 'entropy',
    compress_kv_slots: int = 0,
    retrieved_texts: list[str] | None = None,
    retrieval_bias_alpha: float = 0.0,
    encode_fn: Optional[Callable[[str], list[int]]] = None,
    # Dual-substrate blending (token + byte) enabled by default. A lightweight
    # surrogate byte-stream head blends into token logits when uncertainty is high.
    dual_substrate: bool = True,
    dual_alpha: float = 1.0,
    dual_beta: float = 1.0,
    # Adaptive activation quantization (emulation): per-step fake quant for MLP/attn outputs
    act_quant_enable: bool = False,
    act_quant_min_bits: int = 8,
    act_quant_max_bits: int = 2,
    act_quant_conf_floor: float = 0.3,
    # Optional: explicit KV retention sidecar path (test convenience)
    kv_retention_sidecar: str | None = None,
    # Optional: runtime config to avoid env hot-path lookups
    runtime_config: GenRuntimeConfig | None = None,
) -> torch.Tensor:
    model.eval()
    _log = get_logger("omnicoder.gen")
    # Resolve runtime config (
    rc = runtime_config or build_runtime_config_from_env()
    # ALWAYS-ON speculative decode + verifier defaults
    # We set safe defaults here so callers don't need to pass flags.
    # This does not change the model; it only enables the higher-level decode policy.
    try:
        # Force speculative auto-selection unless explicitly disabled by caller
        if not bool(speculative_auto):
            speculative_auto = True  # type: ignore[assignment]
    except Exception:
        speculative_auto = True  # type: ignore[assignment]
    # Ensure an effective draft length based on available MTP heads if caller didn't set it (>1 to matter)
    try:
        heads = getattr(getattr(model, '_orig_mod', model), 'mtp_heads', None)
        if heads is not None and (int(speculative_draft_len) <= 1):
            speculative_draft_len = max(2, min(len(list(heads)), 4))  # type: ignore[assignment]
    except Exception:
        pass
    # Enable block verification by default when speculative is active
    try:
        if (block_verify is None) and (int(speculative_draft_len) > 1):
            block_verify = True  # type: ignore[assignment]
    except Exception:
        pass
    # Keep verifier steps at least 1
    try:
        verifier_steps = max(1, int(verifier_steps))  # type: ignore[assignment]
    except Exception:
        verifier_steps = 1  # type: ignore[assignment]
    # Do not perform disk IO to load draft models in generate(); callers must pass `draft_model`
    # explicitly if speculative decoding is desired. This keeps the hot path IO-free.
    # Always enable verifier-guided block verification with a sane default threshold
    # Avoid env reads here; prefer runtime_config default or a safe fallback
    if float(verify_threshold) == 0.0:
        try:
            verify_threshold = float(getattr(rc, 'verify_threshold_default', 0.05))
        except Exception:
            verify_threshold = 0.05
    # Always-on diffusion parallel generator: run alongside AR; do NOT replace AR output
    _diffusion_candidate = None
    if DiffusionTextGenerator is not None:
        try:
            steps = int(getattr(rc, 'diffusion_steps', 8)) if (getattr(rc, 'diffusion_steps', None) is not None) else 8
        except Exception:
            steps = 8
        try:
            dg = DiffusionTextGenerator(model, getattr(model, 'd_model', 1024), num_steps=steps)  # type: ignore[arg-type]
            # Use tail of prompt (up to 64 tokens) as condition
            _tail = input_ids[:, -min(input_ids.shape[1], 64):] if isinstance(input_ids, torch.Tensor) else None
            _diffusion_candidate = dg.generate(input_ids=_tail, gen_tokens=int(max_new_tokens), steps=int(steps))  # type: ignore[arg-type]
            try:
                _log.info("generate: diffusion path ok steps=%d len=%d", int(steps), int(max_new_tokens))
            except Exception:
                pass
        except Exception as _e:
            try:
                _log.warning("generate: diffusion path failed (parallel run only): %s", str(_e))
            except Exception:
                pass
    # Optional super-verbose logging toggle (enabled by default)
    try:
        _super_verbose = bool(rc.super_verbose) if rc.super_verbose is not None else True
    except Exception:
        _super_verbose = True
    try:
        # Avoid parameter() call when embed exists to prevent materializing grads; device stays logged
        dev_str = str(getattr(getattr(model, 'embed', object()), 'weight', torch.empty(0, device=next(model.parameters()).device)).device)
        _log.info("generate enter in_shape=%s device=%s max_new_tokens=%s temp=%s top_k=%s top_p=%s draft=%s",
                  str(tuple(input_ids.shape)), dev_str, int(max_new_tokens), float(temperature), int(top_k), float(top_p), bool(draft_model is not None))
        # Log first token id to ensure encode path is correct
        if isinstance(input_ids, torch.Tensor) and input_ids.numel() > 0:
            _log.debug("generate first_token=%s", int(torch.ops.aten.reshape.default(input_ids, (-1,))[0].item()))
        # Log embedding row sample and lm_head column sample for diagnostics
        try:
            e_w = getattr(getattr(model, 'embed', object()), 'weight', None)
            h_w = getattr(getattr(model, 'lm_head', object()), 'weight', None)
            if e_w is not None and h_w is not None:
                _log.debug("embed[0,:5]=%s lm_head[0,:5]=%s", str(e_w[0, :5].detach().float().tolist()), str(h_w[0, :5].detach().float().tolist()))
        except Exception:
            pass
    except Exception:
        pass
    # Ensure torch.compile is active at first use when enabled (no-op if already compiled)
    # Move compilation to model build when possible; keep as no-op here to avoid first-step delay.
    # Assume compile handled at model construction to avoid late compile attempts here
    # SFB: initialize optional parallel reasoning components
    # Default SFB off unless explicitly enabled; avoids heavy imports/conflicts on runtime
    # Honor precomputed availability only; avoid heavy imports unless wired upstream
    sfb_enabled = bool(globals().get('_SFB_AVAILABLE_FLAG', False))
    sfb_ctx = None
    if sfb_enabled:
        try:
            from omnicoder.sfb import (
                factorize_prompt,
                BeliefPropagation,
                SPNCompiler,
                CrossBiasFusion,
                ProofMarginArbiter,
                ProofMarginInputs,
                build_text_semantic_graph,
                run_sum_product,
                semantic_log_marginal_score,
            )
            # Persist heavy SFB helpers across calls to avoid repeated init/IO
            sfb_bp = _SFB_GLOBAL.get('bp') if '_SFB_GLOBAL' in globals() else None
            if sfb_bp is None:
                sfb_bp = BeliefPropagation()
                if '_SFB_GLOBAL' in globals():
                    _SFB_GLOBAL['bp'] = sfb_bp
            sfb_spn = _SFB_GLOBAL.get('spn') if '_SFB_GLOBAL' in globals() else None
            if sfb_spn is None:
                sfb_spn = SPNCompiler()
                if '_SFB_GLOBAL' in globals():
                    _SFB_GLOBAL['spn'] = sfb_spn
            sfb_fuse = _SFB_GLOBAL.get('fuse') if '_SFB_GLOBAL' in globals() else None
            if sfb_fuse is None:
                sfb_fuse = CrossBiasFusion()
                if '_SFB_GLOBAL' in globals():
                    _SFB_GLOBAL['fuse'] = sfb_fuse
            # Set encode_fn if provided by caller for string→ids mapping
            if encode_fn is not None:
                try:
                    sfb_fuse.encode_fn = encode_fn  # type: ignore[attr-defined]
                except Exception:
                    pass
            # Construct arbiter once and cache globally to avoid repeated env parsing
            sfb_arbiter = _SFB_GLOBAL.get('arb') if '_SFB_GLOBAL' in globals() else None
            if sfb_arbiter is None:
                sfb_arbiter = ProofMarginArbiter()
                if '_SFB_GLOBAL' in globals():
                    _SFB_GLOBAL['arb'] = sfb_arbiter
            sfb_ctx = {
                'factorize_prompt': factorize_prompt,
                'bp': sfb_bp,
                'spn': sfb_spn,
                'fuser': sfb_fuse,
                'arbiter': sfb_arbiter,
                'messages': None,
                'sem_beliefs': 0.0,
            }
            # Avoid env-derived metrics; keep fixed defaults
            sfb_ctx['static_metrics'] = {'code_passk': 0.0, 'clip_z': 0.0, 'audio_z': 0.0, 'video_z': 0.0}
            # Build prompt text without env/FS
            raw_prompt = ''
            if encode_fn is not None:
                try:
                    tokenizer_obj = getattr(encode_fn, '__self__', None)
                    if tokenizer_obj is not None and hasattr(tokenizer_obj, 'decode') and input_ids is not None:
                        toks = torch.ops.aten.reshape.default(input_ids, (-1,)).tolist()
                        raw_prompt = str(tokenizer_obj.decode(toks, skip_special_tokens=True))  # type: ignore[attr-defined]
                except Exception:
                    raw_prompt = ''
            if retrieved_texts:
                ctx = "\n\n".join(retrieved_texts[:5])
                raw_prompt = f"{ctx}\n\n{raw_prompt}" if raw_prompt else ctx
            # Persist decoded prompt for downstream symbolic/GraphRAG augmentation
            try:
                sfb_ctx['raw_prompt'] = raw_prompt
            except Exception:
                pass
            # Factorize and run a lightweight BP once at start
            # Avoid factorizing the same prompt repeatedly; cache by prompt hash
            _fp_cache = _SFB_GLOBAL.get('fact_cache') if '_SFB_GLOBAL' in globals() else None
            if _fp_cache is None:
                _fp_cache = {}
                if '_SFB_GLOBAL' in globals():
                    _SFB_GLOBAL['fact_cache'] = _fp_cache
            fact_key = ('txt', raw_prompt[:2048])
            fact = _fp_cache.get(fact_key) if isinstance(_fp_cache, dict) else None
            if fact is None:
                fact = factorize_prompt(raw_prompt)
                if isinstance(_fp_cache, dict):
                    # Limit cache size
                    if len(_fp_cache) > 32:
                        _fp_cache.clear()
                    _fp_cache[fact_key] = fact
            try:
                sfb_spn.maybe_compile(fact.factors)
            except Exception:
                pass
            try:
                # Prefer cached messages for hot subgraphs when available
                cached_msgs = None
                try:
                    cached_msgs = sfb_spn.lookup(fact.factors)  # type: ignore[attr-defined]
                except Exception:
                    cached_msgs = None
                if isinstance(cached_msgs, list) and cached_msgs:
                    messages = cached_msgs
                else:
                    messages = sfb_bp.run(fact.factors)
                    try:
                        sfb_spn.register(fact.factors, messages)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                # Inject goal/intent priors as soft preferences before tokenizing (RSA/plan-recognition proxy)
                try:
                    priors = getattr(fact, 'goal_priors', {}) or {}
                    for goal, prior in list(priors.items())[:8]:
                        # Map goals to lightweight string hints; encode_fn will translate to token ids if available
                        hint = str(goal).strip().lower()
                        if hint:
                            messages.append({
                                'prefer_strings': [hint],
                                'score': float(prior),
                            })
                except Exception:
                    pass
            except Exception:
                messages = []
            sfb_ctx['messages'] = messages
            # Compute semantic log-marginal beliefs
            try:
                g = build_text_semantic_graph(fact.factors)
                iters = 3
                try:
                    iters = max(1, int(os.getenv('SFB_BP_ITERS_SEM', '3')))
                except Exception:
                    iters = 3
                marg = run_sum_product(g, iterations=iters)
                sfb_ctx['sem_beliefs'] = float(semantic_log_marginal_score(marg))
            except Exception:
                sfb_ctx['sem_beliefs'] = 0.0
            # Keep priors for optional policy/telemetry
            try:
                sfb_ctx['goal_priors'] = dict(getattr(fact, 'goal_priors', {}) or {})
            except Exception:
                pass
        except Exception:
            sfb_ctx = None
    # Long-context cascading memory: prepend summaries/retrieval hits before decode
    try:
        from omnicoder.inference.retrieval_memory import ExternalRetrievalMemory, CascadingMemoryController  # type: ignore
        if _LONGCTX_DEFAULT:
            # Derive query and update controller with current prompt tail
            prompt_text = None
            try:
                # Best-effort detokenize first 256 tokens for a retrieval query
                from omnicoder.training.simple_tokenizer import get_text_tokenizer as _tok
                _tokr = _tok(prefer_hf=True)
                if isinstance(input_ids, torch.Tensor):
                    prompt_text = _tokr.decode(input_ids[0, :min(input_ids.size(1), 256)].tolist())
            except Exception:
                prompt_text = None
            try:
                # Persistent retriever/controller across calls via a module-global cache
                cmc = _SFB_GLOBAL.get('cmc') if '_SFB_GLOBAL' in globals() else None
                if cmc is None:
                    retr = ExternalRetrievalMemory(root=_RAG_ROOT_DEFAULT, dim=256)
                    cmc = CascadingMemoryController(window=2048, retriever=retr)
                    if '_SFB_GLOBAL' in globals():
                        _SFB_GLOBAL['cmc'] = cmc
                # Update with current prompt token tail
                if isinstance(input_ids, torch.Tensor):
                    cmc.update_tokens([int(x) for x in input_ids[0].tolist()])
                # Bundle retrieval + summaries and prepend to input if any
                bundle = cmc.bundle_for_query(prompt_text or "", k=int(_RAG_TOPK_DEFAULT))
                if bundle:
                    try:
                        hint = "\n" + "\n".join(bundle) + "\n"
                        from omnicoder.training.simple_tokenizer import get_text_tokenizer as _tok2
                        _tokr2 = _tok2(prefer_hf=True)
                        hint_ids = torch.tensor([_tokr2.encode(hint)], dtype=torch.long, device=input_ids.device)
                        input_ids = torch.cat([hint_ids, input_ids], dim=1)
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass

    # Ω-Reasoner: optional lightweight controller
    # Enable Ω-Controller by default when available
    omega_enabled = (OmegaController is not None)
    omega = OmegaController(OmegaConfig()) if omega_enabled else None
    # Allow the controller to suggest decode knobs (block verify etc.)
    if omega is not None:
        try:
            knobs = omega.suggest_decode_knobs()
            if block_verify is None and 'block_verify' in knobs:
                block_verify = bool(knobs['block_verify'])
            if block_verify_size is None and 'block_verify_size' in knobs:
                block_verify_size = int(knobs['block_verify_size'])
            # Optionally widen tree search using controller-suggested speculative branches
            if isinstance(knobs.get('speculative_branches', None), int):
                try:
                    if int(tree_width) <= 1:
                        tree_width = max(1, int(knobs['speculative_branches']))
                except Exception:
                    pass
        except Exception:
            pass
    # Resolve block verification knobs: enable by default
    # Prefer SFB_* when present to let the parallel stack control acceptance policy
    try:
        if block_verify is None:
            block_verify = True
    except Exception:
        block_verify = True
    try:
        if block_verify_size is None:
            block_verify_size = max(1, int(speculative_draft_len))
    except Exception:
        block_verify_size = max(1, int(speculative_draft_len))
    # Optional KV quant calibration (ingested at import to avoid IO here)
    try:
        if _KVQ_CAL_META is not None and (kvq in ('u8','nf4')):
            g = _KVQ_CAL_META.get('group', None) if isinstance(_KVQ_CAL_META, dict) else None
            if isinstance(g, int) and g > 0:
                kvq_group = int(g)
            if kvq == 'none' and isinstance(_KVQ_CAL_META, dict) and ('scheme' in _KVQ_CAL_META):
                kvq = str(_KVQ_CAL_META['scheme'])
    except Exception as e:
        logging.debug("[generate] KVQ calibration meta skipped: %s", e)
    # Apply SCMoE inference knobs globally where supported
    try:
        if hasattr(model, 'blocks'):
            for blk in model.blocks:
                if hasattr(blk, 'moe'):
                    if hasattr(blk.moe, 'scmoe_alpha'):
                        blk.moe.scmoe_alpha = float(scmoe_alpha)
                    if hasattr(blk.moe, 'scmoe_frac'):
                        blk.moe.scmoe_frac = float(scmoe_frac)
                    if hasattr(blk.moe, 'scmoe_topk'):
                        blk.moe.scmoe_topk = int(scmoe_topk)
    except Exception as e:
        logging.debug("[generate] applying SCMoE knobs failed: %s", e)
    try:
        _log.debug(
            "generate config block_verify=%s block_verify_size=%s tree=(%s,%s) kvq=%s group=%s scmoe=(alpha=%s,frac=%s,topk=%s) knn=(enabled=%s,k=%s,lambda=%s) window_size=%s",
            bool(block_verify), int(block_verify_size or 0), int(tree_width), int(tree_depth), str(kvq), int(kvq_group), float(scmoe_alpha), float(scmoe_frac), int(scmoe_topk), bool(knn_cache is not None), int(knn_k), float(knn_lambda), int(window_size)
        )
    except Exception:
        pass
    device = next(model.parameters()).device
    # Hard guard: if CUDA is available but model is on CPU, auto-upgrade to a specific CUDA device
    try:
        import torch as _torch
        if _torch.cuda.is_available() and (device.type != 'cuda'):
            try:
                get_logger("omnicoder.gen").warning("auto_cuda_upgrade: model on %s; moving to cuda", str(device))
            except Exception:
                pass
            _idx = int(_torch.cuda.current_device())
            _dev = _torch.device(f"cuda:{_idx}")
            model = model.to(_dev)  # type: ignore[assignment]
            device = next(model.parameters()).device
    except Exception:
        pass
    # Hard assert: prevent silent CPU fallbacks when GPU is requested via env
    try:
        _want_cuda = (os.getenv('OMNICODER_TRAIN_DEVICE','').startswith('cuda') or os.getenv('OMNICODER_DEVICE','').startswith('cuda'))
        if _want_cuda and (not str(device).startswith('cuda')):
            raise RuntimeError(f"Model not on CUDA (device={device}); aborting to avoid CPU fallback. Set OMNICODER_DEVICE=cuda and ensure GPU is visible in Docker (e.g., --gpus all).")
    except Exception:
        pass
    # Optional: enable compressive KV via sidecar/env at runtime (monkey-patch attention modules)
    try:
        kvc_side = os.getenv('OMNICODER_KV_COMPRESS_SIDECAR', '').strip()
        comp_slots_env = int(os.getenv('OMNICODER_COMPRESSIVE_SLOTS', '0'))
        if hasattr(model, 'blocks') and (kvc_side or comp_slots_env > 0):
            from omnicoder.modeling.memory import CompressiveKV  # type: ignore
            slots = comp_slots_env if comp_slots_env > 0 else 4
            for blk in model.blocks:
                attn = getattr(blk, 'attn', None)
                if attn is not None and hasattr(attn, 'kv_latent_dim'):
                    try:
                        attn.compress_kv = CompressiveKV(latent_dim=int(attn.kv_latent_dim), slots=int(slots))
                        attn.compressive_slots = int(slots)
                    except Exception:
                        continue
    except Exception as e:
        logging.debug('[generate] compressive KV sidecar wiring skipped: %s', e)
    # Optional external retrieval memory initialization (disabled unless explicitly allowed)
    retrieval_mem = None
    try:
        mem_root = os.getenv('OMNICODER_RETRIEVAL_ROOT', '')
        allow_ext = (os.getenv('OMNICODER_ALLOW_EXTERNAL_RETRIEVAL', '0') == '1')
        if allow_ext and mem_root and ExternalRetrievalMemory is not None:
            mem_dim = int(os.getenv('OMNICODER_RETRIEVAL_DIM', str(model.d_model if hasattr(model, 'd_model') else 1024)))
            retrieval_mem = ExternalRetrievalMemory(root=mem_root, dim=mem_dim, capacity=100000)
    except Exception as e:
        logging.debug("[generate] retrieval memory init skipped: %s", e)
    # GraphRAG + kNN LM integration: retrieve token-bias hints once per run and apply as soft bias every step
    # Keep disk/IO disabled; use in-memory hooks only
    grag_bias_ids_cached = None
    try:
        if (grag is not None) and getattr(grag, 'enabled', False):
            # Build small list of strings from graph triples to bias sampling
            hinted = []
            try:
                triples = grag.retrieve(str(getattr(sfb_ctx or {}, 'raw_prompt', '') or ''), k=8, allow_net=False)
                for t in list(triples)[:8]:
                    hinted += [str(getattr(t, 'head', '')), str(getattr(t, 'relation', '')), str(getattr(t, 'tail', ''))]
            except Exception:
                hinted = []
            hinted = [h for h in hinted if isinstance(h, str) and h]
            if hinted and encode_fn is not None:
                try:
                    grag_bias_ids_cached = []
                    for s in hinted[:16]:
                        ids = encode_fn(s)  # type: ignore[misc]
                        if isinstance(ids, list):
                            grag_bias_ids_cached += ids[:4]
                except Exception:
                    grag_bias_ids_cached = None
    except Exception:
        grag_bias_ids_cached = None

    # Hard-enforce device uniformity for inputs to prevent silent CPU ops
    out = input_ids.to(device, non_blocking=True)
    # Track reasoning telemetry for Ω2 certificate and per-step traces
    last_acceptance_prob = None  # type: ignore[assignment]
    last_agot_selected_id = None  # type: ignore[assignment]
    last_latent_selected_id = None  # type: ignore[assignment]
    step_trace = deque(maxlen=256)  # type: ignore[assignment]
    # Initialize optional reasoning helpers (env-gated; no cost when disabled)
    try:
        agot = _SFB_GLOBAL.get('agot') if '_SFB_GLOBAL' in globals() else None
        if agot is None and 'build_agot' in globals() and callable(build_agot):
            agot = build_agot()
        # Force-enable AGOT controller by default when available
        if agot is not None:
            try:
                setattr(agot, 'enabled', True)
            except Exception:
                pass
            if '_SFB_GLOBAL' in globals():
                _SFB_GLOBAL['agot'] = agot
    except Exception:
        agot = _SFB_GLOBAL.get('agot', None) if '_SFB_GLOBAL' in globals() else None
    try:
        ltf = _SFB_GLOBAL.get('ltf') if '_SFB_GLOBAL' in globals() else None
        if ltf is None and 'build_latent_bfs' in globals() and callable(build_latent_bfs):
            ltf = build_latent_bfs()
            if '_SFB_GLOBAL' in globals():
                _SFB_GLOBAL['ltf'] = ltf
    except Exception:
        ltf = _SFB_GLOBAL.get('ltf', None) if '_SFB_GLOBAL' in globals() else None
    try:
        symp = _SFB_GLOBAL.get('symp') if '_SFB_GLOBAL' in globals() else None
        if symp is None and 'build_symbolic_planner' in globals() and callable(build_symbolic_planner):
            symp = build_symbolic_planner()
            if '_SFB_GLOBAL' in globals():
                _SFB_GLOBAL['symp'] = symp
    except Exception:
        symp = _SFB_GLOBAL.get('symp', None) if '_SFB_GLOBAL' in globals() else None
    # Initialize reflection with model hidden dim if enabled; defer until we have hidden_out
    try:
        refl = _SFB_GLOBAL.get('refl') if '_SFB_GLOBAL' in globals() else None
        if refl is None and 'build_reflection' in globals() and callable(build_reflection):
            refl = build_reflection(int(getattr(model, 'd_model', 512)))
            if '_SFB_GLOBAL' in globals():
                _SFB_GLOBAL['refl'] = refl
    except Exception:
        refl = _SFB_GLOBAL.get('refl', None) if '_SFB_GLOBAL' in globals() else None
    # Initialize GraphRAG
    try:
        grag = _SFB_GLOBAL.get('grag') if '_SFB_GLOBAL' in globals() else None
        if grag is None and 'build_graphrag' in globals() and callable(build_graphrag):
            grag = build_graphrag()
            if '_SFB_GLOBAL' in globals():
                _SFB_GLOBAL['grag'] = grag
    except Exception:
        grag = _SFB_GLOBAL.get('grag', None) if '_SFB_GLOBAL' in globals() else None
    # Ensure optional SFB helpers live on the same device as the model to avoid CPU fallbacks
    try:
        def _maybe_to_device(x):
            try:
                if x is None:
                    return None
                if hasattr(x, 'to') and callable(getattr(x, 'to')):
                    return x.to(device)  # type: ignore[attr-defined]
            except Exception:
                return x
            return x
        agot = _maybe_to_device(agot)
        ltf = _maybe_to_device(ltf)
        symp = _maybe_to_device(symp)
        refl = _maybe_to_device(refl)
        grag = _maybe_to_device(grag)
    except Exception:
        pass
    # Inject symbolic planner and GraphRAG hints into SFB messages (env-gated, no-op if disabled)
    try:
        if sfb_ctx is not None:
            # Planner-derived hints
            if symp is not None and getattr(symp, 'enabled', False):
                try:
                    rp = str(sfb_ctx.get('raw_prompt', ''))
                    plan = symp.plan(rp)
                    try:
                        sfb_ctx['symbolic_plan'] = {'steps': [getattr(a, 'name', '') for a in getattr(plan, 'actions', [])]}
                    except Exception:
                        sfb_ctx['symbolic_plan'] = {'steps': []}
                    msgs = sfb_ctx.get('messages') or []  # type: ignore[assignment]
                    # Add light preferences for action names/args
                    for a in getattr(plan, 'actions', [])[:8]:
                        try:
                            hints = [str(getattr(a, 'name', '')).lower()]
                            for arg in list(getattr(a, 'args', ()))[:2]:
                                hints.append(str(arg).lower())
                            hints = [h for h in hints if h]
                            if hints:
                                msgs.append({'prefer_strings': hints, 'score': 0.15})
                        except Exception:
                            continue
                    sfb_ctx['messages'] = msgs  # type: ignore[index]
                except Exception:
                    pass
            # GraphRAG-derived hints (avoid per-token IO; ensure retrieval is in-memory-only)
            if grag is not None and getattr(grag, 'enabled', False):
                try:
                    rp = str(sfb_ctx.get('raw_prompt', ''))
                    triples = grag.retrieve(rp, k=8, allow_net=False)
                    sfb_ctx['kg_triples'] = triples  # type: ignore[index]
                    msgs = sfb_ctx.get('messages') or []  # type: ignore[assignment]
                    for t in list(triples)[:8]:
                        try:
                            hints = [str(getattr(t, 'head', '')).lower(), str(getattr(t, 'relation', '')).lower(), str(getattr(t, 'tail', '')).lower()]
                            hints = [h for h in hints if h]
                            if hints:
                                msgs.append({'prefer_strings': hints, 'score': 0.10})
                        except Exception:
                            continue
                    sfb_ctx['messages'] = msgs  # type: ignore[index]
                except Exception:
                    pass
    except Exception:
        pass
    # Optional: adopt KV retention/compression policy from sidecar next to a checkpoint if present
    try:
        # Prefer explicit argument when provided; else fall back to environment variable
        pol_path = (kv_retention_sidecar or '').strip()
        if not pol_path:
            pol_path = os.getenv('OMNICODER_KV_RETENTION', '').strip()
        if pol_path and int(compress_kv_slots) <= 0:
            pol = _json.loads(open(pol_path, 'r', encoding='utf-8').read())
            cs = int(pol.get('compressive_slots', pol.get('slots', 0)))
            if cs > 0:
                compress_kv_slots = cs  # type: ignore[assignment]
                if int(window_size) <= 0:
                    window_size = int(pol.get('window_size', pol.get('window', 0)))  # type: ignore[assignment]
    except Exception:
        pass

    # Optional: load learned KV compression autoencoder sidecar (enc/dec weights)
    kv_ae = None
    try:
        comp_side = os.getenv('OMNICODER_KV_COMPRESS_SIDECAR', '').strip()
        if comp_side:
            meta = _json.loads(open(comp_side, 'r', encoding='utf-8').read())
            kv = meta.get('kv_autoencoder', {}) if isinstance(meta, dict) else {}
            wpt = kv.get('weights', '')
            if wpt:
                ck = torch.load(wpt, map_location='cpu')
                sd = ck.get('state_dict', {}) if isinstance(ck, dict) else {}
                enc = sd.get('enc.weight', None)
                dec = sd.get('dec.weight', None)
                if isinstance(enc, torch.Tensor) and isinstance(dec, torch.Tensor):
                    enc_t = enc.detach().to(device=device, dtype=torch.float32).t().contiguous()  # (DL, L)
                    dec_t = dec.detach().to(device=device, dtype=torch.float32).t().contiguous()  # (L, DL)
                    kv_ae = (enc_t, dec_t)
    except Exception:
        kv_ae = None
    # Optional prefix KV cache warmup/lookup
    initial_past_kv = None
    try:
        if os.getenv('OMNICODER_PREFIX_CACHE', '0') == '1' and out.size(1) > 1:
            import hashlib as _hashlib
            min_tok = max(1, int(os.getenv('OMNICODER_PREFIX_CACHE_MIN_TOKENS', '64')))
            use_len = min(int(out.size(1)), max(min_tok, int(os.getenv('OMNICODER_PREFIX_CACHE_USE_LEN', str(min_tok)))))
            if use_len >= min_tok:
                key_ids = out[:, :use_len].detach().cpu().tolist()[0]
                key_hash = _hashlib.sha1((','.join(map(str, key_ids))).encode('utf-8')).hexdigest()
                cached = None
                for i, (k, v) in enumerate(_PREFIX_KV_CACHE):
                    if k == key_hash:
                        cached = v
                        # move to front
                        _PREFIX_KV_CACHE.insert(0, _PREFIX_KV_CACHE.pop(i))
                        break
                if cached is None:
                    # Try disk
                    disk_kv = _prefix_kv_load_disk(key_hash)
                    if disk_kv is not None:
                        _PREFIX_KV_CACHE.insert(0, (key_hash, disk_kv))
                        cached = disk_kv
                if cached is None:
                    # Build KV by stepping through prefix once
                    tmp_ids = out[:, :use_len]
                    tmp_kv = None
                    for t in range(use_len):
                        step = tmp_ids[:, t:t+1]
                        outs = model(step, past_kv=tmp_kv, use_cache=True)
                        if isinstance(outs, tuple):
                            _, tmp_kv = outs[0], outs[1]  # type: ignore
                        else:
                            tmp_kv = None
                    if tmp_kv is not None:
                        kv_cpu: list[tuple[torch.Tensor, torch.Tensor, KvQuantMeta]] = []
                        for (k_t, v_t, meta) in tmp_kv:  # type: ignore[assignment]
                            kv_cpu.append((k_t.detach().cpu(), v_t.detach().cpu(), meta))
                        _PREFIX_KV_CACHE.insert(0, (key_hash, kv_cpu))
                        _prefix_kv_save_disk(key_hash, kv_cpu)
                        cap = int(os.getenv('OMNICODER_PREFIX_CACHE_SIZE', str(_PREFIX_KV_CACHE_CAPACITY)))
                        while len(_PREFIX_KV_CACHE) > max(1, cap):
                            _PREFIX_KV_CACHE.pop()
                        # Reuse KV and trim input to remainder
                        initial_past_kv = [(k.to(device), v.to(device), m) for (k, v, m) in kv_cpu]
                        out = out[:, use_len:]
                else:
                    initial_past_kv = [(k.to(device), v.to(device), m) for (k, v, m) in cached]
                    out = out[:, use_len:]
    except Exception as _e:
        logging.debug('[generate] prefix cache skipped: %s', _e)
    # Preserve original input for return value even when using windowed decode with memory priming
    orig_input = torch.ops.aten.mul.Scalar(out, 1.0)
    generated_seq: list[torch.Tensor] = []
    # Preallocate generation buffer to avoid per-step tensor growth
    gen_buf = torch.empty((out.size(0), int(max_new_tokens)), dtype=torch.long, device=out.device)
    gen_len = 0
    with torch.no_grad():
        past_kv = initial_past_kv
        # Speculative acceptance bookkeeping
        attempted_speculative = 0
        accepted_speculative = 0
        # SFB dynamic refresh cadence
        sfb_refresh_n = 16
        try:
            sfb_refresh_n = max(4, int(os.getenv('SFB_REFRESH_TOKENS', '16')))
        except Exception:
            sfb_refresh_n = 16
        sfb_tokens_since_refresh = 0
        # Infinite-context priming: if a sliding window is set and the model has a memory module,
        # summarize the prefix beyond the window into memory slots and prime the KV cache once.
        try:
            if int(window_size) > 0 and hasattr(model, 'memory') and getattr(model, 'memory') is not None:
                total_len = int(out.size(1))
                if total_len > int(window_size):
                    prefix_len = total_len - int(window_size)
                    prefix_ids = out[:, :prefix_len]
                    window_ids = out[:, prefix_len:]
                    # Compute memory slots from embedded prefix
                    prefix_feats = model.embed(prefix_ids.to(device))
                    mem_feats = model.memory(prefix_feats)  # type: ignore[attr-defined]
                    # Prime KV with memory features
                    from omnicoder.inference.generate import prime_kv_with_features as _prime
                    past_kv, _ = _prime(model, mem_feats)
                    out = window_ids
                    try:
                        get_logger("omnicoder.gen").debug("memory priming: total=%s window=%s prefix=%s", total_len, int(window_size), int(prefix_len))
                    except Exception:
                        pass
        except Exception as e:
            logging.debug("[generate] memory priming failed: %s", e)
            past_kv = None
        # EMA of accepted verifier probabilities (for auto-thresholding)
        ema_vprob = None
        # CIS memoization buffer (normalized hidden -> cached logits)
        cis_enabled = bool(rc.cis_cache_enable)
        cis_eps = float(rc.cis_eps)
        cis_cap = int(rc.cis_cap)
        # Cache frequently accessed environment flags/values for per-step use
        try:
            use_landmarks_mode = rc.use_landmarks_mode
        except Exception:
            use_landmarks_mode = 'auto'
        try:
            trace_en_flag = bool(rc.trace_enable)
        except Exception:
            trace_en_flag = True
        try:
            accept_margin_env = float(rc.rg_accept_margin)
        except Exception:
            accept_margin_env = 0.0
        try:
            latent_bfs_width = max(1, int(rc.latent_bfs_width))
        except Exception:
            latent_bfs_width = 3
        try:
            refl_ent_min = float(rc.reflect_entropy_min)
        except Exception:
            refl_ent_min = 2.0
        # Bind frequently used torch ops locally for faster attribute access inside the loop
        softmax = torch.ops.aten._softmax.default
        log = torch.log
        clamp = torch.clamp
        argmax = torch.argmax
        cis_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        cis_hits = 0
        # Vectorized CIS storage (device tensors) to avoid Python loops/scans
        cis_keys_t: torch.Tensor | None = None  # (N, Dh)
        cis_vals_t: torch.Tensor | None = None  # (N, V)
        # Auto-enable KNN cache by default when hidden dims are known
        try:
            auto_knn = (os.getenv('OMNICODER_KNN_CACHE','1') == '1')
        except Exception:
            auto_knn = True
        # Disable disk-backed caches by default to prevent IO churn
        os.environ.setdefault('OMNICODER_DISABLE_DISK_CACHE', '1')
        # Optional: build GraphRAG by default when enabled and a root exists
        try:
            auto_grag = bool(rc.graphrag_enable)
        except Exception:
            auto_grag = True
        grag = None
        if auto_grag:
            try:
                grag = _SFB_GLOBAL.get('grag') if '_SFB_GLOBAL' in globals() else None
                if grag is None:
                    from omnicoder.retrieval.graphrag import build_graphrag  # type: ignore
                    grag = build_graphrag()
                    if '_SFB_GLOBAL' in globals():
                        _SFB_GLOBAL['grag'] = grag
            except Exception:
                grag = _SFB_GLOBAL.get('grag', None) if '_SFB_GLOBAL' in globals() else None
        # Tiny LRU for dequantized KV across steps when windowing keeps KV stable
        # Small LRU for dequantized KV across steps: (key, value), FIFO of size 4
        # Key includes KV tensor ids and a compact window-tail signature to ensure safety under sliding window
        _kv_deq_lru: list[tuple[tuple[tuple[int,int], ...] | None, int | None, list[tuple[torch.Tensor, torch.Tensor]]]] = []  # type: ignore[assignment]
        # Track repetition to mitigate degenerate loops
        rep_penalty = float(rc.rep_penalty)
        rep_window = int(rc.rep_window)
        # Stronger structural constraints
        try:
            no_repeat_ngram = int(rc.no_repeat_ngram)
        except Exception:
            no_repeat_ngram = 0
        try:
            no_repeat_window = int(rc.no_repeat_window)
        except Exception:
            no_repeat_window = 256
        # Min-p nucleus threshold (relative to max prob) to drop ultra-low prob tails
        try:
            min_p = float(rc.min_p)
        except Exception:
            min_p = 0.0
        # Additional OpenAI-style penalties (frequency/presence)
        try:
            freq_penalty = float(os.getenv('OMNICODER_FREQ_PENALTY', '0.0'))
        except Exception:
            freq_penalty = 0.0
        try:
            pres_penalty = float(os.getenv('OMNICODER_PRESENCE_PENALTY', '0.0'))
        except Exception:
            pres_penalty = 0.0
        # Optional: mask non-text vocab slice to avoid multimodal spillover in text decode
        try:
            mask_non_text = bool(rc.mask_non_text)
        except Exception:
            mask_non_text = False
        text_vocab_size: int | None = None
        if mask_non_text:
            # Only attempt once and cache across calls to avoid per-step disk access
            try:
                _u = globals().get('UNIFIED_VOCAB_TEXT_SIZE', None)
            except Exception:
                _u = None
            if _u is None:
                try:
                    import json as _json
                    from pathlib import Path as _P
                    um = _P("/workspace/weights/release/unified_vocab_map.json")
                    if um.exists():
                        meta = _json.loads(um.read_text(encoding='utf-8'))
                        if isinstance(meta, dict):
                            tv = int(meta.get("text_size", 0))
                            if tv > 0:
                                try:
                                    globals()['UNIFIED_VOCAB_TEXT_SIZE'] = tv
                                except Exception:
                                    pass
                                text_vocab_size = tv
                except Exception:
                    text_vocab_size = None
            else:
                try:
                    text_vocab_size = int(_u)
                except Exception:
                    text_vocab_size = None
        # Optional ban of immediate repetitions: if the last token repeated >= N times, strongly downweight it
        try:
            ban_repeat_run = int(os.getenv('OMNICODER_BAN_REPEAT_RUN', '0'))
        except Exception:
            ban_repeat_run = 0
        # Optional: hard-ban the exact last token from being sampled again
        try:
            ban_last_token = (os.getenv('OMNICODER_BAN_LAST_TOKEN', '0') == '1')
        except Exception:
            ban_last_token = False
        # Cache for GraphRAG per-run token bias (avoid per-step retrieval/tokenization)
        grag_triples_cached = None
        grag_bias_ids_cached: list[int] | None = None
        # Adaptive MoE refresh cadence to avoid per-step layer reconfiguration
        try:
            moe_refresh_every = max(1, int(os.getenv('OMNICODER_ADAPTIVE_MOE_EVERY', '8')))
        except Exception:
            moe_refresh_every = 8
        # Enable kNN-LM only when blending weight is non-zero
        use_knn_runtime = (knn_cache is not None) and (float(knn_lambda) > 0.0)
        # Bind perf helpers outside the hot loop to avoid per-step import/lookup overhead
        try:
            from omnicoder.utils.perf import add as _perf_add  # type: ignore
        except Exception:
            _perf_add = None  # type: ignore
        try:
            from omnicoder.utils.perf import timer as _perf_timer  # type: ignore
            from contextlib import nullcontext as _nullcontext  # type: ignore[attr-defined]
        except Exception:
            _perf_timer = None  # type: ignore
            from contextlib import contextmanager
            @contextmanager
            def _nullcontext():  # type: ignore[misc]
                yield
        # Resolve once: AMP capability and dtype (avoid per-step device capability queries)
        try:
            from omnicoder.utils.torchutils import get_amp_dtype as _get_amp_dtype  # type: ignore
        except Exception:
            _get_amp_dtype = None  # type: ignore
        dev_str = None
        try:
            dev_str = str(next(model.parameters()).device)
        except Exception:
            dev_str = None
        _amp_dtype = (_get_amp_dtype(dev_str) if (_get_amp_dtype is not None) else None)
        _amp_enabled = (_amp_dtype is not None)
        # Prepare cudagraph step marker to avoid replay output aliasing when compiled on CUDA
        try:
            from omnicoder.utils.torchutils import get_cudagraph_step_marker as _get_cg  # type: ignore
        except Exception:
            _get_cg = None  # type: ignore
        try:
            _cg_mark = (_get_cg() if _get_cg is not None else None)
        except Exception:
            _cg_mark = None
        if _amp_enabled:
            try:
                import torch.cuda as _tc
                _amp_enabled = True
                cc = _tc.get_device_capability()
                _amp_dtype = torch.bfloat16 if (cc and cc[0] >= 8) else torch.float16
            except Exception:
                _amp_enabled = False
                _amp_dtype = None
        decode_time_sum = 0.0
        decode_steps = 0
        for _ in range(max_new_tokens):
            # Build current sequence view = original input + generated tokens so far
            if gen_len > 0:
                seq_view = torch.cat([orig_input, gen_buf[:, :gen_len]], dim=1)
            else:
                seq_view = orig_input
            try:
                import time as _t
                if gen_len == 0:
                    t_first0 = _t.perf_counter()
                t_step0 = _t.perf_counter()
                _log.debug("decode loop step cur_len=%s target=%s", int(seq_view.size(1)), int(max_new_tokens))
                # Per-step perf marker
                try:
                    if _perf_add is not None:
                        _perf_add('decode_step_enter', 0.0)
                except Exception:
                    pass
            except Exception:
                pass
            # Step 1: decode next token with cache (CUDA Graphs enabled; mark step for replay safety)
            # Feed last generated token if available; otherwise feed last prompt token
            if gen_len > 0:
                step_inp = torch.ops.aten.slice.Tensor(gen_buf, 1, int(gen_len-1), int(gen_len), 1)
            else:
                step_inp = (torch.ops.aten.slice.Tensor(out, 1, int(out.size(1) - 1), int(out.size(1)), 1) if out.size(1) > 1 else out)
            feed_kv = past_kv
            if past_kv is not None and kvq in ('u8','nf4'):
                # LRU across steps when KV tensors unchanged (e.g., before extending)
                try:
                    _key_kv = tuple((int(kq.data_ptr()), int(vq.data_ptr())) for (kq, vq, _m) in past_kv)  # type: ignore[attr-defined]
                except Exception:
                    _key_kv = tuple((id(kq), id(vq)) for (kq, vq, _m) in past_kv)
                # Include a short tail hash of current window to guard reuse under sliding windows
                try:
                    if int(window_size) > 0 and seq_view.size(1) > 0:
                        tail_len = min(8, int(window_size), int(seq_view.size(1)))
                        tail = seq_view[0, -tail_len:].tolist()
                        _tail_sig = hash(tuple(tail)) & 0xFFFF
                    else:
                        _tail_sig = None
                except Exception:
                    _tail_sig = None
                # Try LRU hit
                hit = None
                for i, (k, tail_sig, val) in enumerate(_kv_deq_lru):
                    if k == _key_kv and tail_sig == _tail_sig:
                        hit = (i, val)
                        break
                if hit is not None:
                    # Move to front (most recent)
                    idx, val = hit
                    if idx != 0:
                        _kv_deq_lru.pop(idx)
                        _kv_deq_lru.insert(0, ( _key_kv, _tail_sig, val ))
                    feed_kv = val  # type: ignore[assignment]
                else:
                    # Dequantize and ensure tensors land on the model device (avoid CPU compute)
                    dq_list = [dequantize_kv(kq, vq, meta) for (kq, vq, meta) in past_kv]
                    feed_kv = [(k.to(device, non_blocking=True), v.to(device, non_blocking=True)) for (k, v) in dq_list]  # type: ignore[assignment]
                    _kv_deq_lru.insert(0, (_key_kv, _tail_sig, feed_kv))
                    # Trim LRU to size 4
                    if len(_kv_deq_lru) > 4:
                        _kv_deq_lru.pop()
            # For random-access jumps, optionally pass a short hidden prefix as landmark source when windowing
            lm_prefix = None
            # Compute landmarks only on the first decode step to avoid repeated heavy embeddings
            if gen_len == 0 and int(window_size) > 0 and seq_view.size(1) > 1:
                try:
                    # Default-enable landmarks when windowing unless explicitly disabled
                    if use_landmarks_mode == 'auto' or use_landmarks_mode == '1':
                        start = max(0, seq_view.size(1) - int(window_size))
                        with _perf_timer('landmark_embed') if _perf_timer is not None else _nullcontext():
                            lm_prefix = model.embed(seq_view[:, start:])
                except Exception:
                    lm_prefix = None
            try:
                _log.debug(
                    "decode step start step_inp_shape=%s feed_kv=%s lm_prefix=%s",
                    str(tuple(step_inp.shape)), "Y" if feed_kv is not None else "N", "Y" if lm_prefix is not None else "N"
                )
            except Exception:
                pass
            # perf helpers are bound once above the loop to avoid per-step overhead
            try:
                # Use autocast for faster matmuls on GPU; prefer bfloat16 when supported
                _use_amp = _amp_enabled
                # Validate device alignment for this step (avoid mixed CPU/GPU)
                try:
                    def _same_dev(_a: torch.device, _b: torch.device) -> bool:
                        try:
                            if _a.type != _b.type:
                                return False
                            if _a.type != 'cuda':
                                return str(_a) == str(_b)
                            ia = (_a.index if _a.index is not None else 0)
                            ib = (_b.index if _b.index is not None else 0)
                            return ia == ib
                        except Exception:
                            return str(_a) == str(_b)
                    if not _same_dev(step_inp.device, device):
                        raise RuntimeError(f"step_inp on {step_inp.device} but model on {device}")
                    if feed_kv is not None:
                        for (k_t, v_t) in feed_kv:
                            if (not _same_dev(k_t.device, device)) or (not _same_dev(v_t.device, device)):
                                raise RuntimeError("past_kv tensors not on model device")
                except Exception as _dev_e:
                    get_logger("omnicoder.gen").error("device_mismatch: %s", str(_dev_e))
                    raise
                if _perf_timer is not None:
                    with _perf_timer('decode_step_model'):
                        if _use_amp and _amp_dtype is not None:
                            with torch.autocast(device_type='cuda', dtype=_amp_dtype):
                                try:
                                    if _cg_mark is not None:
                                        _cg_mark()  # type: ignore[misc]
                                except Exception:
                                    pass
                                # NOTE: Historically we passed flags like need_verifier/need_aux_scores/need_mtp
                                # here to prune work. Those introduced Python branching in the model hot path
                                # and broke compile/CG/export invariants. The model now computes all aux heads
                                # unconditionally and returns them; callers may ignore unused outputs.
                                outputs = model(
                                    step_inp,
                                    past_kv=feed_kv,
                                    use_cache=True,
                                    return_hidden=True,
                                    prefix_hidden=lm_prefix,
                                )
                        else:
                            try:
                                if _cg_mark is not None:
                                    _cg_mark()  # type: ignore[misc]
                            except Exception:
                                pass
                            # See note above: flags removed to keep model forward branch-free and aten-only.
                            outputs = model(
                                step_inp,
                                past_kv=feed_kv,
                                use_cache=True,
                                return_hidden=True,
                                prefix_hidden=lm_prefix,
                            )
                else:
                    if _use_amp and _amp_dtype is not None:
                        with torch.autocast(device_type='cuda', dtype=_amp_dtype):
                            try:
                                if _cg_mark is not None:
                                    _cg_mark()  # type: ignore[misc]
                            except Exception:
                                pass
                            # See note above: flags removed to keep model forward branch-free and aten-only.
                            outputs = model(
                                step_inp,
                                past_kv=feed_kv,
                                use_cache=True,
                                return_hidden=True,
                                prefix_hidden=lm_prefix,
                            )
                    else:
                        try:
                            if _cg_mark is not None:
                                _cg_mark()  # type: ignore[misc]
                        except Exception:
                            pass
                        # See note above: flags removed to keep model forward branch-free and aten-only.
                        outputs = model(
                            step_inp,
                            past_kv=feed_kv,
                            use_cache=True,
                            return_hidden=True,
                            prefix_hidden=lm_prefix,
                        )
                try:
                    import time as _t
                    t_step1 = _t.perf_counter()
                    dt = float(t_step1 - t_step0)
                    decode_time_sum += dt
                    decode_steps += 1
                    try:
                        from omnicoder.utils.perf import add as _perf_add  # type: ignore
                        _perf_add('decode_step_dt', dt)
                    except Exception:
                        pass
                    if gen_len == 0:
                        _log.info("decode first_token_dt=%.3fs", float(t_step1 - t_first0))
                        # Ultra-verbose diagnostics immediately after first token
                        try:
                            _log.info("diag after first token: step_inp=%s device=%s kv_present=%s kvq=%s kvq_group=%s",
                                      str(tuple(step_inp.shape)), str(device), bool(feed_kv is not None), str(kvq), int(kvq_group))
                            if isinstance(outputs, tuple):
                                _log.info("diag outputs tuple lens: logits=%s kv=%s mtp=%s verifier=%s diff=%s halt=%s retention=%s",
                                          str(tuple(outputs[0].shape)) if outputs[0] is not None else str(None),
                                          len(outputs[1]) if outputs[1] is not None else 0,
                                          str(tuple(outputs[2].shape)) if (len(outputs) > 2 and outputs[2] is not None) else str(None),
                                          str(tuple(outputs[3].shape)) if (len(outputs) > 3 and outputs[3] is not None) else str(None),
                                          str(tuple(outputs[4].shape)) if (len(outputs) > 4 and outputs[4] is not None) else str(None),
                                          str(tuple(outputs[5].shape)) if (len(outputs) > 5 and outputs[5] is not None) else str(None),
                                          str(tuple(outputs[6].shape)) if (len(outputs) > 6 and outputs[6] is not None) else str(None))
                            else:
                                _log.info("diag outputs tensor: %s", str(tuple(getattr(outputs,'shape',()))))
                        except Exception:
                            pass
                    else:
                        _log.debug("decode step_dt=%.6f", float(t_step1 - t_step0))
                except Exception:
                    pass
            except Exception as e:
                import traceback as _tb
                try:
                    _log.error("decode step error: %s", str(e))
                    _log.error("trace:\n%s", _tb.format_exc())
                except Exception:
                    pass
                raise
            # Expect (logits, past_kv, mtp_logits, verifier_logits, diff_score, halt_score, retention_score[, hidden])
            if isinstance(outputs, tuple):
                logits = outputs[0]  # type: ignore
                try:
                    if str(device).startswith('cuda') and torch.cuda.is_available():
                        logits = torch.ops.aten.mul.Scalar(logits, 1.0)
                except Exception:
                    pass
                new_kv = outputs[1]  # type: ignore
                # Detach KV tensors from graphed storage before reusing across steps
                try:
                    if str(device).startswith('cuda') and torch.cuda.is_available():
                        new_kv = [(torch.ops.aten.mul.Scalar(k, 1.0), torch.ops.aten.mul.Scalar(v, 1.0)) for (k, v) in new_kv]  # type: ignore[assignment]
                except Exception:
                    pass
                mtp_logits = outputs[2] if len(outputs) > 2 else None  # type: ignore
                verifier_logits = outputs[3] if len(outputs) > 3 else None  # type: ignore
                diff_score = outputs[4] if len(outputs) > 4 else None  # type: ignore
                halt_score = outputs[5] if len(outputs) > 5 else None  # type: ignore
                retention_score = outputs[6] if len(outputs) > 6 else None  # type: ignore
                # If MTP heads were not returned, but hidden_out is available and model exposes heads, compute them
                if (mtp_logits is None) and (len(outputs) > 7):
                    try:
                        hidden_out = outputs[7]
                        heads = getattr(getattr(model, '_orig_mod', model), 'mtp_heads', None)
                        if heads is not None:
                            mtp_logits = [h(hidden_out) for h in heads]  # type: ignore
                    except Exception:
                        pass
                # KV diagnostics around potential stall points
                try:
                    _log.debug("diag pre_kv: new_kv_len=%s k0=%s v0=%s",
                               len(new_kv) if new_kv is not None else 0,
                               str(tuple(new_kv[0][0].shape)) if (new_kv and new_kv[0]) else str(None),
                               str(tuple(new_kv[0][1].shape)) if (new_kv and new_kv[0]) else str(None))
                except Exception:
                    pass
                if kvq in ('u8','nf4'):
                    # Quantize KV for storage; measure cost
                    if _perf_timer is not None:
                        with _perf_timer('kv_quantize'):
                            past_kv = [
                                quantize_kv(
                                    k.to(torch.float32, copy=False).contiguous(),
                                    v.to(torch.float32, copy=False).contiguous(),
                                    scheme=kvq, group_size=kvq_group
                                )
                                for (k, v) in new_kv
                            ]  # type: ignore[assignment]
                    else:
                        past_kv = [
                            quantize_kv(
                                k.to(torch.float32, copy=False).contiguous(),
                                v.to(torch.float32, copy=False).contiguous(),
                                scheme=kvq, group_size=kvq_group
                            )
                            for (k, v) in new_kv
                        ]  # type: ignore[assignment]
                else:
                    # Optional compressive KV: summarize old prefix into fixed slots and keep recent window (optionally using learned AE)
                    if int(compress_kv_slots) > 0 and int(window_size) > 0:
                        comp_list: list[tuple[torch.Tensor, torch.Tensor]] = []
                        for (k, v) in new_kv:  # type: ignore[assignment]
                            try:
                                B, H, T, DL = k.shape  # type: ignore[assignment]
                                if T > int(window_size):
                                    old_len = max(0, T - int(window_size))
                                    slots = max(1, int(compress_kv_slots))
                                    seg = []
                                    base = old_len // slots
                                    rem = old_len % slots
                                    start = 0
                                    for si in range(slots):
                                        end = start + base + (1 if si < rem else 0)
                                        seg.append((start, end))
                                        start = end
                                    k_segs = []
                                    v_segs = []
                                    for (a, b) in seg:
                                        if b <= a:
                                            k_segs.append(k[:, :, :1, :])
                                            v_segs.append(v[:, :, :1, :])
                                        else:
                                            if kv_ae is not None:
                                                enc_t, dec_t = kv_ae
                                                # Average pool segment to stable size, then AE compress/decompress
                                                ks = k[:, :, a:b, :].mean(dim=2, keepdim=True)
                                                vs = v[:, :, a:b, :].mean(dim=2, keepdim=True)
                                                ks2 = (ks @ enc_t).matmul(dec_t)
                                                vs2 = (vs @ enc_t).matmul(dec_t)
                                                k_segs.append(ks2)
                                                v_segs.append(vs2)
                                            else:
                                                k_segs.append(k[:, :, a:b, :].mean(dim=2, keepdim=True))
                                                v_segs.append(v[:, :, a:b, :].mean(dim=2, keepdim=True))
                                    k_comp = torch.cat(k_segs + [torch.ops.aten.slice.Tensor(k, 2, int(k.shape[2]) - int(window_size), int(k.shape[2]), 1)], dim=2)
                                    v_comp = torch.cat(v_segs + [torch.ops.aten.slice.Tensor(v, 2, int(v.shape[2]) - int(window_size), int(v.shape[2]), 1)], dim=2)
                                    comp_list.append((k_comp.contiguous(), v_comp.contiguous()))
                                else:
                                    comp_list.append((k, v))
                            except Exception:
                                comp_list.append((k, v))
                        past_kv = comp_list  # type: ignore[assignment]
                    else:
                        past_kv = new_kv  # type: ignore[assignment]
                try:
                    _log.debug("diag post_kv: past_kv_len=%s k0=%s v0=%s",
                               len(past_kv) if past_kv is not None else 0,
                               str(tuple(past_kv[0][0].shape)) if (past_kv and past_kv[0]) else str(None),
                               str(tuple(past_kv[0][1].shape)) if (past_kv and past_kv[0]) else str(None))
                except Exception:
                    pass
            else:
                logits = outputs  # type: ignore
                try:
                    if str(device).startswith('cuda') and torch.cuda.is_available():
                        logits = torch.ops.aten.mul.Scalar(logits, 1.0)
                except Exception:
                    pass
                verifier_logits = None
                diff_score = None
                halt_score = None

            # Super-verbose per-step diagnostics on logits distribution
            if _super_verbose:
                try:
                    if _perf_timer is not None:
                        with _perf_timer('logits_stats'):
                            last = logits[:, -1, :]
                            lmin = float(last.min().item())
                            lmax = float(last.max().item())
                            lmean = float(last.mean().item())
                            lstd = float(last.std(unbiased=False).item())
                            # Avoid full softmax over vocab; report top-k logits directly for logging
                            vals, idx = torch.topk(last, k=min(10, last.size(-1)), dim=-1)
                            topk_list = [(int(idx[0, i].item()), float(vals[0, i].item())) for i in range(min(10, idx.size(1)))]
                            _log.info(
                                "step logits stats min=%.6f max=%.6f mean=%.6f std=%.6f topk_logits=%s",
                                lmin, lmax, lmean, lstd, topk_list,
                            )
                    else:
                        last = logits[:, -1, :]
                        lmin = float(last.min().item())
                        lmax = float(last.max().item())
                        lmean = float(last.mean().item())
                        lstd = float(last.std(unbiased=False).item())
                        vals, idx = torch.topk(last, k=min(10, last.size(-1)), dim=-1)
                        topk_list = [(int(idx[0, i].item()), float(vals[0, i].item())) for i in range(min(10, idx.size(1)))]
                        _log.info(
                            "step logits stats min=%.6f max=%.6f mean=%.6f std=%.6f topk_logits=%s",
                            lmin, lmax, lmean, lstd, topk_list,
                        )
                except Exception:
                    pass

            # Optional: learned write-policy to external memory and kNN-LM blending
            hidden_out = None
            try:
                # hidden is last when return_hidden=True
                if isinstance(outputs, tuple) and len(outputs) >= 8:
                    hidden_out = outputs[-1]  # type: ignore[index]
            except Exception as e:
                logging.debug("[generate] extracting hidden_out failed: %s", e)
                hidden_out = None
            # Optional CIS memoization: if hidden is near a cached state, reuse logits
            if cis_enabled and hidden_out is not None and isinstance(hidden_out, torch.Tensor):
                try:
                    if hidden_out.size(1) == 0:
                        h_last = torch.zeros((hidden_out.size(0), hidden_out.size(-1)), device=hidden_out.device, dtype=hidden_out.dtype)
                    else:
                        h_last = hidden_out[:, -1, :]
                    hn = F.normalize(h_last, dim=-1)
                    # Vectorized nearest lookup
                    if cis_keys_t is not None and cis_vals_t is not None and cis_keys_t.numel() > 0:
                        # Compute L2 distance to all cached
                        dif = cis_keys_t - hn
                        d2 = (dif * dif).sum(dim=-1)  # (N,)
                        dmin, imin = torch.min(d2, dim=0)
                        if float(dmin) <= float(cis_eps):
                            logits[:, -1, :] = cis_vals_t[imin]
                            cis_hits += 1
                        else:
                            # Insert
                            if logits is not None and logits.dim() == 3:
                                key_add = hn.detach()
                                val_add = logits[:, -1, :].detach()
                                cis_keys_t = torch.cat([cis_keys_t, key_add], dim=0)
                                cis_vals_t = torch.cat([cis_vals_t, val_add], dim=0)
                                # Enforce cap by slicing from the end (keep most recent)
                                _cap = max(1, cis_cap)
                                if cis_keys_t.size(0) > _cap:
                                    cis_keys_t = cis_keys_t[-_cap:]
                                    cis_vals_t = cis_vals_t[-_cap:]
                    else:
                        # Initialize vectors
                        if logits is not None and logits.dim() == 3:
                            cis_keys_t = hn.detach()
                            cis_vals_t = logits[:, -1, :].detach()
                except Exception:
                    pass
            # Lazy-create a KNN cache when first hidden is available
            if knn_cache is None and hidden_out is not None and auto_knn:
                try:
                    from omnicoder.inference.knn_cache import KNNCache  # type: ignore
                    dim_h = int(hidden_out.size(-1))
                    knn_cache = KNNCache(dim=dim_h, use_cosine=True, use_faiss=True)
                except Exception:
                    knn_cache = None
            # Optional retention policy application: drop/compress KV based on learned retention scores
            if retention_score is not None and past_kv is not None and int(compress_kv_slots) > 0:
                try:
                    # retention_score: (B,T,1) for the current step token; maintain a small window of scores
                    # Here we apply only to the materialized KV of new_kv by mixing mean pools proportionally.
                    comp_list2: list[tuple[torch.Tensor, torch.Tensor]] = []
                    for (k, v) in new_kv:  # type: ignore[assignment]
                        B, H, T, DL = k.shape  # type: ignore[assignment]
                        if T > int(window_size):
                            old_len = max(0, T - int(window_size))
                            slots = max(1, int(compress_kv_slots))
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
                                    seg_k = k[:, :, start:end, :].mean(dim=2, keepdim=True)
                                    seg_v = v[:, :, start:end, :].mean(dim=2, keepdim=True)
                                k_segs.append(seg_k)
                                v_segs.append(seg_v)
                                start = end
                            k_comp = torch.cat(k_segs + [torch.ops.aten.slice.Tensor(k, 2, int(k.shape[2]) - int(window_size), int(k.shape[2]), 1)], dim=2)
                            v_comp = torch.cat(v_segs + [torch.ops.aten.slice.Tensor(v, 2, int(v.shape[2]) - int(window_size), int(v.shape[2]), 1)], dim=2)
                            comp_list2.append((k_comp.contiguous(), v_comp.contiguous()))
                        else:
                            comp_list2.append((k, v))
                    past_kv = comp_list2  # type: ignore[assignment]
                except Exception as e:
                    logging.debug("[generate] retention policy application failed: %s", e)
            # Dual substrate byte blending: compute a light byte-stream logits and blend
            if bool(dual_substrate):
                # Allow environment to steer dual-substrate at runtime without API changes
                try:
                    import os as _os
                    _ds = _os.getenv('OMNICODER_DUAL_SUBSTRATE','')
                    if _ds:
                        dual_substrate = (_ds == '1')
                    # Hard override to force-enable byte blending regardless of ckpt flag
                    if _os.getenv('OMNICODER_DUAL_FORCE','0') == '1':
                        setattr(model, '_ckpt_has_dual_substrate', True)
                    _da = _os.getenv('OMNICODER_DUAL_ALPHA','')
                    _db = _os.getenv('OMNICODER_DUAL_BETA','')
                    if _da:
                        dual_alpha = float(_da)
                    if _db:
                        dual_beta = float(_db)
                    # Additionally disable if checkpoint didn't enable byte head explicitly
                    if getattr(model, '_ckpt_has_dual_substrate', False) is not True:
                        get_logger('omnicoder.gen').debug('dual_substrate disabled (no ckpt flag)')
                        raise RuntimeError('dual_disabled')
                    g = torch.sigmoid(float(dual_alpha) * ent + float(dual_beta) * kl)  # (B,1)
                    # Surrogate byte head: project last hidden to vocab via a lightweight linear proj
                    if 'hidden_out' in locals() and isinstance(hidden_out, torch.Tensor):
                        h_last = hidden_out[:, -1:, :]
                        byte_proj = getattr(model, '_byte_proj', None)
                        if byte_proj is None:
                            proj = torch.nn.Linear(h_last.size(-1), logits.size(-1), bias=False).to(h_last.device)
                            setattr(model, '_byte_proj', proj)
                            byte_proj = proj
                        bl = byte_proj(h_last)  # (B,1,V)
                        logits[:, -1:, :] = (1.0 - g) * logits[:, -1:, :] + g * bl
                except Exception:
                    pass
            if knn_cache is not None and hidden_out is not None:
                try:
                    if hidden_out.size(1) == 0:
                        h_last = torch.zeros((hidden_out.size(0), hidden_out.size(-1)), device=hidden_out.device, dtype=hidden_out.dtype)
                    else:
                        h_last = hidden_out[:, -1, :]
                    # Learned write probability from write_head if present
                    write_p = None
                    if hasattr(model, 'write_head'):
                        try:
                            write_logit = model.write_head(h_last)
                            write_p = float(torch.sigmoid(write_logit).item())
                        except Exception as e:
                            logging.debug("[generate] write_head failed: %s", e)
                            write_p = None
                    # Simple budget: write every 8 tokens or when write_p>=0.7
                    do_write = ((len(generated_seq) % 8) == 0) or (write_p is not None and write_p >= 0.7)
                    if do_write:
                        # Prefer torch backend for KNN cache if available
                        try:
                            next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                            knn_cache.add_torch(h_last[0], next_id)  # type: ignore[attr-defined]
                        except Exception:
                            h_np = h_last.detach().cpu()[0].tolist()
                            next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                            knn_cache.add(h_np, next_id)
                        try:
                            if 'knn_prune_every' in locals() and knn_prune_every > 0 and hasattr(knn_cache, 'prune'):
                                # Use step count proxy based on total generated length
                                if (len(generated_seq) % max(knn_prune_every, 1)) == 0:
                                    knn_cache.prune(max_items=max(knn_max_items, 1))
                        except Exception:
                            pass
                    # Blend distributions
                    try:
                        kp_t = knn_cache.query_torch(h_last[0], k=max(1, int(knn_k)), vocab_size=int(logits.size(-1)))  # type: ignore[attr-defined]
                        model_probs = torch.ops.aten._softmax.default(logits[:, -1, :], -1, False)
                        blended = (1.0 - float(knn_lambda)) * model_probs + float(knn_lambda) * kp_t.unsqueeze(0)
                        logits[:, -1, :] = torch.log(torch.clamp(blended, min=1e-8))
                    except Exception:
                        try:
                            # fallback to numpy path; still avoid extra conversions where possible
                            h_np = h_last.detach().cpu()[0].numpy().astype(np.float32)
                            knn_probs = knn_cache.query(h_np, k=max(1, int(knn_k)), vocab_size=int(logits.size(-1)))
                            model_probs = torch.ops.aten._softmax.default(logits[:, -1, :], -1, False)
                            kp = torch.from_numpy(knn_probs).to(model_probs.device)
                            blended = (1.0 - float(knn_lambda)) * model_probs + float(knn_lambda) * kp
                            logits[:, -1, :] = torch.log(torch.clamp(blended, min=1e-8))
                        except Exception:
                            pass
                except Exception as e:
                    logging.debug("[generate] kNN-LM blend failed: %s", e)
            # Apply anti-repetition penalties (windowed)
            try:
                vocab_dim = int(logits.size(-1))
                # Hard-mask tokens outside text slice if enabled
                if mask_non_text and text_vocab_size is not None and text_vocab_size > 0 and text_vocab_size <= vocab_dim:
                    logits[:, -1, text_vocab_size:] = float('-inf')
                    try:
                        _log.debug("mask_non_text applied up_to=%d of vocab=%d", int(text_vocab_size), int(vocab_dim))
                    except Exception:
                        pass
                if seq_view.size(1) > 0 and vocab_dim > 0:
                    tail_list = seq_view[0, -min(rep_window, seq_view.size(1)):].tolist()
                    try:
                        _log.debug("anti-repeat: window_len=%d tail_size=%d last_tok=%d", int(seq_view.size(1)), int(len(tail_list)), int(seq_view[0, -1].item()) if seq_view.size(1) > 0 else -1)
                    except Exception:
                        pass
                    # Vectorized repetition and frequency penalties
                    tail_t = seq_view[0, -min(rep_window, seq_view.size(1)):]  # (T_tail,)
                    if rep_penalty > 1.0 and tail_t.numel() > 0:
                        uniq = torch.unique(tail_t)
                        uniq = uniq[(uniq >= 0) & (uniq < vocab_dim)]
                        if uniq.numel() > 0:
                            logits[:, -1, uniq] = logits[:, -1, uniq] / rep_penalty
                    if (freq_penalty > 0.0) or (pres_penalty > 0.0):
                        if tail_t.numel() > 0:
                            vals, counts = torch.unique(tail_t, return_counts=True)
                            mask = (vals >= 0) & (vals < vocab_dim)
                            vals = vals[mask]
                            counts = counts[mask]
                            if vals.numel() > 0:
                                # Subtract per-token penalties in one shot
                                logits[:, -1, vals] = logits[:, -1, vals] - (float(freq_penalty) * counts.to(logits.dtype)) - float(pres_penalty)
                    # Optional: strong downweight if the last token repeats as a run of length >= ban_repeat_run
                    last_tok = int(seq_view[0, -1].item()) if seq_view.size(1) > 0 else -1
                    run_len = 1
                    if ban_repeat_run and ban_repeat_run > 1 and seq_view.size(1) >= ban_repeat_run:
                        j = int(seq_view.size(1)) - 2
                        while j >= 0 and int(seq_view[0, j].item()) == last_tok:
                            run_len += 1
                            j -= 1
                        if run_len >= ban_repeat_run and 0 <= last_tok < vocab_dim:
                            # Strongly downweight the repeated token to escape loops
                            logits[:, -1, last_tok] = logits[:, -1, last_tok] - 10.0
                            try:
                                _log.debug("anti-repeat: ban_repeat_run hit last_tok=%d run_len=%d", int(last_tok), int(run_len))
                            except Exception:
                                pass
                    # Optional: hard-ban the exact last token from being sampled again this step
                    if ban_last_token and 0 <= last_tok < vocab_dim:
                        logits[:, -1, last_tok] = float('-inf')
                        try:
                            _log.debug("anti-repeat: ban_last_token applied last_tok=%d", int(last_tok))
                        except Exception:
                            pass
                    # No-repeat n-gram constraint (e.g., 3 = no repeated trigrams)
                    if no_repeat_ngram and no_repeat_ngram >= 2 and seq_view.size(1) >= (no_repeat_ngram - 1):
                        try:
                            w = int(max(0, min(int(no_repeat_window), int(seq_view.size(1)))))
                        except Exception:
                            w = int(seq_view.size(1))
                        seq = seq_view[0, -w:].tolist() if w > 0 else seq_view[0].tolist()
                        n = int(no_repeat_ngram)
                        if len(seq) >= n:
                            # Build prefix->set(next_token) map for observed n-grams in the window
                            banned: dict[tuple[int, ...], set[int]] = {}
                            for i in range(0, len(seq) - n + 1):
                                pref = tuple(seq[i:i + n - 1])
                                nxt = int(seq[i + n - 1])
                                s = banned.get(pref)
                                if s is None:
                                    s = set()
                                    banned[pref] = s
                                s.add(nxt)
                            # Current prefix is the last n-1 tokens
                            cur_pref = tuple(seq[-(n - 1):])
                            to_ban = banned.get(cur_pref)
                            if to_ban:
                                for tid in to_ban:
                                    ii = int(tid)
                                    if 0 <= ii < vocab_dim:
                                        logits[:, -1, ii] = float('-inf')
            except Exception:
                pass
            # Min-p filter: mask tokens whose probability is too small relative to max
            try:
                if min_p > 0.0:
                    # Reuse a single softmax for this step
                    probs_last_pre_mask = torch.ops.aten._softmax.default(logits[:, -1, :], -1, False)
                    vmax = float(torch.max(probs_last_pre_mask).item()) if probs_last_pre_mask.numel() > 0 else 0.0
                    thr = float(min_p) * float(vmax)
                    if thr > 0.0:
                        mask = (probs_last_pre_mask < thr)
                        # Ensure at least top-1 remains unmasked
                        top1_idx = int(torch.argmax(probs_last_pre_mask, dim=-1).item()) if probs_last_pre_mask.numel() > 0 else -1
                        if 0 <= top1_idx < pl.size(-1):
                            mask[..., top1_idx] = False
                        logits[:, -1, :] = torch.where(mask, torch.full_like(logits[:, -1, :], float('-inf')), logits[:, -1, :])
                        try:
                            _log.debug("min_p mask applied thr=%.6f kept=%s", thr, int((~mask).sum().item()))
                        except Exception:
                            pass
            except Exception:
                pass
            # External retrieval consult (read-only bias)
            if retrieval_mem is not None and hidden_out is not None:
                try:
                    if hidden_out.size(1) == 0:
                        h_last = torch.zeros((hidden_out.size(0), hidden_out.size(-1)), device=hidden_out.device, dtype=hidden_out.dtype)
                    else:
                        h_last = hidden_out[:, -1, :]
                    bias = retrieval_mem.search(h_last, topk=4)
                    logits[:, -1, :] = logits[:, -1, :] + 0.05 * bias
                except Exception as e:
                    logging.debug("[generate] retrieval memory consult failed: %s", e)

            # Optional Perceiver-IO prior from raw prompt bytes at first step (env OMNICODER_PERCEIVER_ENABLE=1)
            if seq_view.size(1) <= 1:
                try:
                    import os as _os
                    if _os.getenv('OMNICODER_PERCEIVER_ENABLE','1') == '1':
                        # Require checkpoint flag to prevent untrained biasing
                        if getattr(model, '_ckpt_has_perceiver_prior', False) is not True:
                            get_logger('omnicoder.gen').debug('Perceiver prior disabled (no ckpt flag)')
                            raise RuntimeError('perceiver_disabled')
                        # Build once and cache on model to avoid repeated init
                        perv = getattr(model, '_perceiver_io', None)
                        if perv is None:
                            try:
                                from omnicoder.modeling.perceiver_io import PerceiverIOTower as _PerceiverIOTower  # type: ignore
                                perv = _PerceiverIOTower(d_io=128, d_latent=256, n_latents=64, n_layers=2, n_heads=4).to(device)
                                setattr(model, '_perceiver_io', perv)
                            except Exception:
                                perv = None
                        if perv is not None:
                            # Encode raw prompt to bytes → small embedding → Perceiver → vocab bias
                            q = locals().get('full_prompt', '') or locals().get('prompt', '')
                            from omnicoder.training.simple_tokenizer import ByteTokenizer  # type: ignore
                            bt = getattr(model, '_byte_tok', None)
                            if bt is None:
                                bt = ByteTokenizer()
                                setattr(model, '_byte_tok', bt)
                            bid = bt.encode(str(q))[:256]
                            if not bid:
                                bid = [1]
                            import torch as _torch
                            io_ids = _torch.tensor([bid], dtype=_torch.long, device=device)
                            # Lazy byte embedding cached on model
                            byte_emb = getattr(model, '_byte_embed', None)
                            if byte_emb is None:
                                byte_emb = _torch.nn.Embedding(258, 128).to(device)
                                setattr(model, '_byte_embed', byte_emb)
                            io_feats = byte_emb(io_ids)  # (1, T, 128)
                            io_out = perv(io_feats)       # (1, T, 128)
                            vec = io_out.mean(dim=1)
                            perv_to_vocab = getattr(model, '_perceiver_to_vocab', None)
                            if perv_to_vocab is None:
                                perv_to_vocab = _torch.nn.Linear(int(vec.size(-1)), int(logits.size(-1)), bias=False).to(device)
                                setattr(model, '_perceiver_to_vocab', perv_to_vocab)
                            bias = perv_to_vocab(vec).unsqueeze(1)
                            logits[:, -1:, :] = logits[:, -1:, :] + 0.03 * bias
                except Exception:
                    pass

            # Optional GraphRAG overlay bias at first step
            if 'grag' in locals() and grag is not None and getattr(grag, 'enabled', False) and seq_view.size(1) <= 1:
                try:
                    # Use original prompt if present in scope
                    q = locals().get('full_prompt', '') or locals().get('prompt', '')
                    if grag_triples_cached is None:
                        grag_triples_cached = grag.retrieve(str(q), k=8)
                    overlay = grag.to_overlay_text(grag_triples_cached or [])
                    if overlay and encode_fn is not None:
                        enc = torch.tensor([encode_fn(overlay)], dtype=torch.long, device=device)
                        emb = model.embed(enc)  # type: ignore[attr-defined]
                        vec = emb.mean(dim=1)
                        bias = model.lm_head(vec).unsqueeze(1)  # type: ignore[attr-defined]
                        logits[:, -1:, :] = logits[:, -1:, :] + 0.05 * bias
                except Exception:
                    pass

            # Per-step GraphRAG token bias using cached triples
            try:
                if 'grag' in locals() and grag is not None and getattr(grag, 'enabled', False) and encode_fn is not None:
                    # Build once and reuse
                    if grag_triples_cached is None:
                        q = locals().get('full_prompt', '') or locals().get('prompt', '')
                        grag_triples_cached = grag.retrieve(str(q), k=8)
                    if grag_bias_ids_cached is None:
                        try:
                            alpha_bias = float(os.getenv('OMNICODER_GRAPHRAG_TOKEN_BIAS', '0.02'))
                        except Exception:
                            alpha_bias = 0.02
                        ids_to_bias: list[int] = []
                        for t in list(grag_triples_cached or [])[:8]:
                            try:
                                for s in (getattr(t, 'head', ''), getattr(t, 'relation', ''), getattr(t, 'tail', '')):
                                    s = str(s).strip()
                                    if not s:
                                        continue
                                    tids = encode_fn(s)
                                    if isinstance(tids, (list, tuple)):
                                        ids_to_bias.extend(int(x) for x in list(tids)[-3:])
                            except Exception:
                                continue
                        grag_bias_ids_cached = list(dict.fromkeys([int(x) for x in ids_to_bias if isinstance(x, int)]))
                    if grag_bias_ids_cached:
                        logits[:, -1, grag_bias_ids_cached] = logits[:, -1, grag_bias_ids_cached] + float(os.getenv('OMNICODER_GRAPHRAG_TOKEN_BIAS', '0.02'))
            except Exception:
                pass

            # Lightweight Algorithmic Core hint (env OMNICODER_ALG_CORE=1):
            # Detect trivial algorithmic subproblems and nudge logits for the first step.
            if out.size(1) <= 1:
                try:
                    import os as _os
                    if _os.getenv('OMNICODER_ALG_CORE','1') == '1':
                        try:
                            from omnicoder.reasoning.alg_core import build_alg_core  # type: ignore
                        except Exception:
                            build_alg_core = None  # type: ignore
                        if build_alg_core is not None and callable(build_alg_core):
                            alg = getattr(model, '_alg_core', None)
                            if alg is None:
                                alg = build_alg_core()
                                setattr(model, '_alg_core', alg)
                            q = locals().get('full_prompt', '') or locals().get('prompt', '')
                            hint = alg.suggest(str(q)) if hasattr(alg, 'suggest') else None
                            if hint and isinstance(hint, dict) and encode_fn is not None:
                                # hint: {"bias": {"token": weight, ...}}
                                bias_map = hint.get('bias', {}) or {}
                                for tok_str, w in list(bias_map.items())[:32]:
                                    try:
                                        tids = encode_fn(str(tok_str))
                                        if isinstance(tids, (list, tuple)) and len(tids) > 0:
                                            tid = int(tids[-1])
                                            logits[:, -1, tid] = logits[:, -1, tid] + float(w)
                                    except Exception:
                                        continue
                except Exception:
                    pass

            # Optionally perform a small tree-search acceptor to pick a better next token
            def _sample_one(cur_logits: torch.Tensor):
                # SFB cross-biasing hook: adjust logits before sampling
                if sfb_ctx is not None:
                    try:
                        messages = sfb_ctx.get('messages') or []  # type: ignore[assignment]
                        cur_logits = sfb_ctx['fuser'].apply_messages(cur_logits, messages)  # type: ignore[index]
                    except Exception:
                        pass
                if _perf_timer is not None:
                    with _perf_timer('sample_next_token'):
                        return sample_next_token(cur_logits, temperature, top_k, top_p)
                return sample_next_token(cur_logits, temperature, top_k, top_p)

            # Adaptive expert/acceptance heuristic via softmax confidence and learned difficulty/halting
            probs_last = torch.ops.aten._softmax.default(logits[:, -1, :], -1, False)
            top_prob, _ = torch.topk(probs_last, k=1, dim=-1)
            conf = float(top_prob.item()) if top_prob.numel() > 0 else 0.0
            # If a learned difficulty score is available, blend with softmax confidence
            if diff_score is not None:
                try:
                    d = float(diff_score[:, -1, :].mean().item())
                    # map difficulty [0,1] to an effective lower confidence (harder → lower conf)
                    conf = max(0.0, min(1.0, 0.7 * conf + 0.3 * (1.0 - d)))
                except Exception:
                    pass
            # adjust speculative drafts length by confidence
            if conf < adaptive_conf_floor:
                speculative_draft_len = min(int(adaptive_top_k_max), max(1, speculative_draft_len))
            else:
                speculative_draft_len = max(int(adaptive_top_k_min), 1)
            # Adaptive MoE expert usage at runtime: scale top_k and capacity_factor
            try:
                if (gen_len % moe_refresh_every) == 0:
                    cur_top_k = int(adaptive_top_k_max if conf < adaptive_conf_floor else adaptive_top_k_min)
                    cur_capacity = float(1.30 if conf < adaptive_conf_floor else 1.00)
                    if hasattr(model, 'blocks'):
                        L = len(model.blocks)
                        for li, blk in enumerate(model.blocks):
                            if hasattr(blk, 'moe'):
                                k_layer = cur_top_k
                                if adaptive_layer_ramp and L > 1:
                                    frac = 1.0 - (float(li) / float(max(1, L - 1)))
                                    scale = pow(max(0.0, min(1.0, frac)), float(max(0.0, adaptive_layer_power)))
                                    k_layer = max(1, int(round(1 + (cur_top_k - 1) * scale)))
                                try:
                                    n_e = int(getattr(blk.moe, 'n_experts', k_layer))
                                    blk.moe.top_k = max(1, min(k_layer, n_e))
                                except Exception:
                                    blk.moe.top_k = max(1, k_layer)
                                blk.moe.capacity_factor = float(cur_capacity)
            except Exception as e:
                logging.debug("[generate] adaptive moe knobs failed: %s", e)
            # Early-exit: use either heuristic threshold or learned halting score if present
            if early_exit:
                try:
                    _log.debug("early_exit check mode=%s thresh=%s", str(early_exit_mode), float(early_exit_entropy))
                except Exception:
                    pass
                with torch.no_grad():
                    halted = False
                    if halt_score is not None:
                        try:
                            h = float(halt_score[:, -1, :].mean().item())
                            # Simple rule: if halting suggests high confidence, skip drafts
                            halted = (h >= max(early_exit_entropy, 0.5))
                        except Exception as e:
                            logging.debug("[generate] halting score parse failed: %s", e)
                            halted = False
                    if (not halted) and early_exit_mode == 'entropy':
                        pl = probs_last
                        ent = -torch.sum(pl * torch.log(torch.clamp(pl, min=1e-9)), dim=-1)
                        if float(ent.item()) <= float(early_exit_entropy):
                            try:
                                _log.debug("early_exit entropy met ent=%s <= thresh=%s", float(ent.item()), float(early_exit_entropy))
                            except Exception:
                                pass
                            speculative_draft_len = 0
                    elif (not halted) and early_exit_mode == 'delta':
                        # Use max logit delta vs runner-up as confidence proxy
                        last = logits[:, -1, :]
                        top2 = torch.topk(last, k=min(2, last.size(-1)), dim=-1).values
                        delta = float((top2[..., 0] - top2[..., 1]).item()) if top2.size(-1) >= 2 else float('inf')
                        # Map delta to an entropy-like threshold by heuristic: larger delta → confident
                        if delta >= max(early_exit_entropy, 1.0):
                            try:
                                _log.debug("early_exit delta met delta=%s >= thresh=%s", float(delta), float(max(early_exit_entropy, 1.0)))
                            except Exception:
                                pass
                            speculative_draft_len = 0
            # Reflexive metacognition: entropy-gated hidden feedback before scoring
            if 'refl' in locals() and refl is not None and getattr(refl, 'enabled', False) and 'hidden_out' in locals() and hidden_out is not None:
                try:
                    ent_min = refl_ent_min
                    with torch.no_grad():
                        pl_tmp = probs_last
                        ent_now = -torch.sum(pl_tmp * torch.log(torch.clamp(pl_tmp, min=1e-9)), dim=-1)
                        if float(ent_now.item()) >= float(ent_min):
                            hidden_out = refl.apply(hidden_out)
                except Exception:
                    pass
            # Choose next token (AGoT / latent BFS optional)
            best_id = None
            try:
                if 'agot' in locals() and agot is not None and getattr(agot, 'enabled', False):
                    best_id = agot.step(model, out, feed_kv, logits, verifier_logits, hidden_out, temperature, top_k, top_p)
                    try:
                        last_agot_selected_id = int(best_id.item())  # type: ignore[assignment]
                    except Exception:
                        pass
            except Exception:
                best_id = None
            if best_id is None:
                if 'ltf' in locals() and ltf is not None and getattr(ltf, 'enabled', False) and 'hidden_out' in locals():
                    try:
                        probs = torch.ops.aten._softmax.default(logits[:, -1, :], -1, False)
                        k = latent_bfs_width
                        # Vectorized cands via gather on topk indices
                        _, topi = torch.topk(probs, k=k, dim=-1)
                        cands = [topi[:, i : i + 1] for i in range(k)]
                        lscores = ltf.score_candidates(model, out, feed_kv, hidden_out, cands)
                        bi = max(enumerate(lscores), key=lambda kv: float(kv[1]))[0]
                        best_id = cands[bi]
                        try:
                            last_latent_selected_id = int(best_id.item())  # type: ignore[assignment]
                        except Exception:
                            pass
                    except Exception:
                        best_id = None
            if best_id is None:
                best_id = _sample_one(logits[:, -1, :])
            # Per-step trace logging and accumulation (bounded length)
            if trace_en_flag:
                try:
                    ent_now = None
                    try:
                        # Keep on-device and avoid .item() syncs where possible
                        pl_tmp = probs_last
                        ent_vec = -torch.sum(pl_tmp * torch.log(torch.clamp(pl_tmp, min=1e-9)), dim=-1)
                        ent_now = float(ent_vec.detach().cpu().float().mean().item())
                    except Exception:
                        ent_now = None
                    step_trace.append({
                        't': int(seq_view.size(1)),
                        'agot_id': int(last_agot_selected_id) if last_agot_selected_id is not None else None,
                        'latent_id': int(last_latent_selected_id) if last_latent_selected_id is not None else None,
                        'chosen_id': int(best_id.item()),
                        'entropy': ent_now,
                        'accept_p': float(last_acceptance_prob) if last_acceptance_prob is not None else None,
                    })
                    # Bound memory handled by deque(maxlen)
                except Exception:
                    pass
            # Ω-Reasoner halting/acceptance (lightweight): derive p_accept from entropy and optional verifier delta
            if omega is not None:
                try:
                    pl = probs_last
                    ent = -torch.sum(pl * torch.log(torch.clamp(pl, min=1e-9)), dim=-1)
                    # KL vs MTP/draft if available
                    kl = 0.0
                    if mtp_logits is not None:
                        ql = torch.ops.aten._softmax.default(mtp_logits[0][:, -1, :], -1, False) if isinstance(mtp_logits, (list, tuple)) and len(mtp_logits) > 0 else None
                        if ql is not None:
                            # Avoid .item() sync; compute scalar via detach+mean
                            _kl_vec = torch.sum(pl * (torch.log(torch.clamp(pl, 1e-9)) - torch.log(torch.clamp(ql, 1e-9))), dim=-1)
                            kl = float(_kl_vec.detach().cpu().float().mean().item())
                    vmargin = 0.0
                    vp = None
                    if verifier_logits is not None:
                        vp = torch.ops.aten._softmax.default(verifier_logits[:, -1, :], -1, False)
                        vmargin = float(vp.gather(-1, best_id).detach().cpu().float().mean().item())
                    # Constraints acceptance: compare margin to configured threshold
                    constraints_ok = True
                    constraints_ok = (float(vmargin) >= float(accept_margin_env))
                    # Convert ent to scalar non-blocking
                    _ent_scalar = float(ent.detach().cpu().float().mean().item())
                    p_acc = float(omega.acceptance_probability(_ent_scalar, float(kl), float(vmargin)))
                    try:
                        last_acceptance_prob = float(p_acc)  # type: ignore[assignment]
                    except Exception:
                        pass
                    # If controller suggests halting and constraints meet margin, skip drafts
                    if omega.should_halt(p_acc, constraints_ok=constraints_ok):
                        speculative_draft_len = 0
                except Exception:
                    pass

            # SFB proof-margin arbiter for speculative block acceptance control
            if sfb_ctx is not None and speculative_draft_len > 0:
                try:
                    probs_last = torch.ops.aten._softmax.default(logits[:, -1, :], -1, False)
                    # Avoid .item(): take mean on-device and detach
                    conf = float(probs_last.gather(-1, best_id).detach().cpu().float().mean().item())
                    # Aggregate small factor scores from messages (proxy for sum log phi)
                    sum_log_phi = 0.0
                    try:
                        msgs = sfb_ctx.get('messages') or []  # type: ignore[assignment]
                        sum_log_phi = float(sum(m.get('score', 0.0) for m in msgs if isinstance(m, dict)))
                    except Exception:
                        sum_log_phi = 0.0
                    vscore = 0.0
                    if verifier_logits is not None:
                        vp = torch.ops.aten._softmax.default(verifier_logits[:, -1, :], -1, False)
                        vscore = float(vp.gather(-1, best_id).detach().cpu().float().mean().item())
                    stat = sfb_ctx.get('static_metrics', {})  # type: ignore[assignment]
                    pm_in = ProofMarginInputs(
                        llm_confidence=conf,
                        sum_log_factors=sum_log_phi,
                        verifier_score=vscore,
                        retrieval_hits=len(retrieved_texts) if retrieved_texts else 0,
                        code_passk=float(stat.get('code_passk', 0.0)),
                        clip_z=float(stat.get('clip_z', 0.0)),
                        audio_z=float(stat.get('audio_z', 0.0)),
                        video_z=float(stat.get('video_z', 0.0)),
                    )
                    # Blend semantic beliefs and SPN unary PC score once
                    try:
                        sem_bel = float(sfb_ctx.get('sem_beliefs', 0.0))
                        sem_w = 1.0
                        try:
                            sem_w = float(os.getenv('SFB_SEM_WEIGHT', '1.0'))
                        except Exception:
                            sem_w = 1.0
                        spn_pc = 0.0
                        try:
                            spn_pc = float(sfb_ctx['spn'].pc_score(fact.factors))  # type: ignore[index]
                        except Exception:
                            spn_pc = 0.0
                        pm_in.sum_log_factors = float(pm_in.sum_log_factors + sem_w * sem_bel + 0.5 * spn_pc)
                    except Exception:
                        pass
                    # Require margin to be non-decreasing if configured
                    require_rise = (os.getenv('SFB_REQUIRE_MARGIN_RISE', '1') == '1')
                    cur_margin = sfb_ctx['arbiter'].compute_margin(pm_in)  # type: ignore[index]
                    last_margin = sfb_ctx.get('last_margin', None)
                    sfb_ctx['last_margin'] = float(cur_margin)
                    accept = sfb_ctx['arbiter'].accept(pm_in)  # type: ignore[index]
                    try:
                        sfb_ctx['arbiter'].log_decision(pm_in, bool(accept))  # type: ignore[index]
                    except Exception:
                        pass
                    # Optional: react to arbiter suggestions to adjust decode knobs on the fly
                    try:
                        sugg = sfb_ctx['arbiter'].suggestions(pm_in)  # type: ignore[index]
                        if isinstance(sugg, dict) and sugg:
                            if bool(sugg.get('escalate_block_verify', False)):
                                try:
                                    if block_verify_size is not None:
                                        block_verify_size = max(int(block_verify_size), int(_SFB_ESCALATE_BLOCK_SZ_MIN))
                                    else:
                                        block_verify_size = max(4, int(_SFB_ESCALATE_BLOCK_SZ_MIN))
                                except Exception:
                                    pass
                            if bool(sugg.get('disable_early_exit', False)):
                                try:
                                    early_exit = False
                                except Exception:
                                    pass
                            if bool(sugg.get('bump_scmoe_topk', False)):
                                try:
                                    scmoe_topk = max(int(scmoe_topk), int(os.getenv('SFB_ESCALATE_SCMOE_TOPK', '2')))  # type: ignore[assignment]
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    if require_rise and last_margin is not None:
                        accept = accept and (float(cur_margin) >= float(last_margin))
                    if not accept:
                        speculative_draft_len = 0
                        # Hint SFB to refresh factors and gently escalate compute/tools on the next steps
                        try:
                            sfb_tokens_since_refresh = max(sfb_tokens_since_refresh, sfb_refresh_n)
                        except Exception:
                            pass
                        try:
                            if retrieved_texts and (float(retrieval_bias_alpha or 0.0) <= 0.0):
                                retrieval_bias_alpha = 0.1  # type: ignore[assignment]
                        except Exception:
                            pass
                        try:
                            adaptive_top_k_min = max(int(adaptive_top_k_min), 2)  # type: ignore[assignment]
                        except Exception:
                            pass
                        # Arbiter-driven compute policy escalation knobs
                        try:
                            # If margin is low, increase speculative verify size and disable early-exit
                            if block_verify_size is not None:
                                block_verify_size = max(int(block_verify_size), int(_SFB_ESCALATE_BLOCK_SZ_MIN))
                            else:
                                block_verify_size = max(4, int(_SFB_ESCALATE_BLOCK_SZ_MIN))
                        except Exception:
                            pass
                        try:
                            # Disable early-exit on hard tokens
                            early_exit = False  # type: ignore[assignment]
                        except Exception:
                            pass
                        try:
                            # Increase MoE expert budget if supported (soft hint via env-backed args)
                            scmoe_topk = max(int(scmoe_topk), int(os.getenv('SFB_ESCALATE_SCMOE_TOPK', '2')))  # type: ignore[assignment]
                        except Exception:
                            pass
                except Exception:
                    pass
            # SFB: dynamic factor proposals from running decode every N tokens
            if sfb_ctx is not None:
                try:
                    sfb_tokens_since_refresh += 1
                    if sfb_tokens_since_refresh >= sfb_refresh_n and encode_fn is not None:
                        sfb_tokens_since_refresh = 0
                        # Decode a short trailing window to refresh factors
                        tail_len = min(128, int(os.getenv('SFB_REFRESH_DECODE_LEN', '64')))
                        try:
                            tail_ids = torch.ops.aten.reshape.default(seq_view[:, -tail_len:], (-1,)).tolist()
                        except Exception:
                            tail_ids = torch.ops.aten.reshape.default(seq_view, (-1,)).tolist()
                        # Best-effort decode using provided tokenizer; if not available, skip
                        try:
                            # If encode_fn is a bound method (e.g., tokenizer.encode), use its owner for decode
                            tokenizer_obj = getattr(encode_fn, '__self__', None)
                            if tokenizer_obj is not None and hasattr(tokenizer_obj, 'decode'):
                                text_tail = str(tokenizer_obj.decode(tail_ids, skip_special_tokens=True))  # type: ignore[attr-defined]
                            else:
                                # Try a globally-available tokenizer symbol if present
                                try:
                                    text_tail = str(tokenizer.decode(tail_ids, skip_special_tokens=True))  # type: ignore[name-defined]
                                except Exception:
                                    text_tail = ''
                        except Exception:
                            text_tail = ''
                        # If no decode available from encode_fn, attempt local simple map via model if exposed
                        if not text_tail:
                            try:
                                # Many codepaths have a tokenizer in outer scope; we cannot reach it here reliably.
                                # As a lightweight proxy, refresh using token-ids string (still useful for heuristics)
                                text_tail = " ".join(str(t) for t in tail_ids[-64:])
                            except Exception:
                                text_tail = ''
                        if text_tail:
                            try:
                                fact2 = sfb_ctx['factorize_prompt'](text_tail)  # type: ignore[index]
                                sfb_ctx['spn'].maybe_compile(fact2.factors)  # type: ignore[index]
                                # Prefer cached messages when available
                                msgs2 = None
                                try:
                                    msgs2 = sfb_ctx['spn'].lookup(fact2.factors)  # type: ignore[index]
                                except Exception:
                                    msgs2 = None
                                if not (isinstance(msgs2, list) and msgs2):
                                    msgs2 = sfb_ctx['bp'].run(fact2.factors)  # type: ignore[index]
                                    try:
                                        sfb_ctx['spn'].register(fact2.factors, msgs2)  # type: ignore[index]
                                    except Exception:
                                        pass
                                # Re-inject updated priors if any
                                try:
                                    pri2 = getattr(fact2, 'goal_priors', {}) or {}
                                    for goal, prior in list(pri2.items())[:8]:
                                        hint = str(goal).strip().lower()
                                        if hint:
                                            msgs2.append({'prefer_strings': [hint], 'score': float(prior)})
                                    sfb_ctx['goal_priors'] = dict(pri2)
                                except Exception:
                                    pass
                                sfb_ctx['messages'] = msgs2
                                # Recompute semantic beliefs on refresh
                                try:
                                    g2 = build_text_semantic_graph(fact2.factors)
                                    iters2 = 3
                                    try:
                                        iters2 = max(1, int(os.getenv('SFB_BP_ITERS_SEM', '3')))
                                    except Exception:
                                        iters2 = 3
                                    marg2 = run_sum_product(g2, iterations=iters2)
                                    sfb_ctx['sem_beliefs'] = float(semantic_log_marginal_score(marg2))
                                except Exception:
                                    pass
                            except Exception:
                                pass
                except Exception:
                    pass
            # Optional adaptive activation quantization emulation: fake-quantize hidden_out based on confidence
            if bool(act_quant_enable) and hidden_out is not None and isinstance(hidden_out, torch.Tensor):
                try:
                    # Map confidence to bit-width in [act_quant_max_bits..act_quant_min_bits]
                    bmin = max(2, int(act_quant_max_bits))
                    bmax = max(bmin, int(act_quant_min_bits))
                    conf_clamped = max(0.0, min(1.0, conf))
                    if conf_clamped >= float(act_quant_conf_floor):
                        # Higher confidence → lower bits
                        t = (conf_clamped - act_quant_conf_floor) / max(1e-6, (1.0 - act_quant_conf_floor))
                        bits = int(round(bmax - t * (bmax - bmin)))
                    else:
                        bits = bmax
                    # Fake quantize last-step hidden: scale to [-1,1], quantize to levels
                    levels = float(2 ** bits - 1)
                    # Use only the last position to limit cost
                    h_last = hidden_out[:, -1:, :]
                    h_norm = torch.tanh(h_last)
                    q = torch.round((h_norm * 0.5 + 0.5) * levels) / max(1.0, levels)
                    q = (q * 2.0 - 1.0)
                    # Blend back lightly to emulate activation quant noise
                    hidden_out[:, -1:, :] = 0.9 * h_last + 0.1 * q
                except Exception:
                    pass
            # Optional retrieval cross-attention bias over decoded logits
            try:
                if retrieval_bias_alpha and retrieval_bias_alpha > 0.0 and retrieved_texts:
                    # Require a text encoder function to tokenize retrieved texts
                    if encode_fn is None:
                        raise RuntimeError("encode_fn is required when retrieval_bias_alpha>0 and retrieved_texts provided")
                    enc_ids = []
                    max_len_rt = 0
                    for rt in retrieved_texts[:8]:
                        try:
                            enc = torch.tensor([encode_fn(rt)], dtype=torch.long, device=device)
                            enc_ids.append(enc)
                            if enc.size(1) > max_len_rt:
                                max_len_rt = int(enc.size(1))
                        except Exception:
                            continue
                    if enc_ids:
                        pad_rt = []
                        for e in enc_ids:
                            if e.size(1) < max_len_rt:
                                pad = torch.zeros((1, max_len_rt - e.size(1)), dtype=torch.long, device=device)
                                e = torch.cat([e, pad], dim=1)
                            pad_rt.append(e)
                        rt_ids = torch.cat(pad_rt, dim=0)  # (N,T)
                        rt_emb = model.embed(rt_ids)  # type: ignore[attr-defined]
                        rt_vec = rt_emb.mean(dim=1)  # (N,D)
                        if hidden_out is not None and isinstance(hidden_out, torch.Tensor):
                            if hidden_out.size(1) == 0:
                                h_last = torch.zeros((hidden_out.size(0), hidden_out.size(-1)), device=hidden_out.device, dtype=hidden_out.dtype)
                            else:
                                h_last = hidden_out[:, -1, :]
                        else:
                            h_last = model.ln_f(model.embed(out[:, -1:])[:, -1, :])  # type: ignore[attr-defined]
                        scores = torch.matmul(h_last, rt_vec.transpose(0, 1)) / max(1.0, float(h_last.size(-1)) ** 0.5)
                        w = torch.ops.aten._softmax.default(scores, -1, False)
                        ctx = torch.matmul(w, rt_vec)
                        bias = model.lm_head(ctx).unsqueeze(1)
                        logits[:, -1:, :] = logits[:, -1:, :] + float(retrieval_bias_alpha) * bias
            except Exception as _e:
                logging.debug('[generate] retrieval bias skipped: %s', _e)
            # Expose last verifier margin to MoE VGR via environment (best-effort; local scope only)
            try:
                if os.getenv('OMNICODER_VGR_RUNTIME_FEEDBACK', '1') == '1':
                    if omega is not None:
                        # signal current verifier margin for router to widen/sharpen as needed
                        os.environ['OMNICODER_LAST_VERIFIER_MARGIN'] = str(max(0.0, min(1.0, float(vmargin))))
            except Exception:
                pass
            if tree_width > 1 and tree_depth > 0:
                try:
                    base_logits = logits
                    # Sample additional candidates
                    cands = [best_id]
                    for _tw in range(int(tree_width) - 1):
                        cands.append(_sample_one(base_logits[:, -1, :]))
                    # Evaluate tiny subgraphs up to depth with verifier margins and token budget
                    scores = []  # (margin_sum, score_proxy, cost_tokens)
                    probs = torch.ops.aten._softmax.default(base_logits[:, -1, :], -1, False)
                    rg_budget = 1
                    try:
                        rg_budget = int(os.getenv('OMNICODER_RG_BUDGET_TOKENS', str(int(tree_depth))))
                    except Exception:
                        rg_budget = int(tree_depth)
                    for cid in cands:
                        try:
                            margin_sum = 0.0
                            score_proxy = float(probs.gather(-1, cid).item())
                            cost = 1
                            # Simulate up to depth steps without committing state
                            if int(tree_depth) > 1 and rg_budget > 0:
                                tmp_ids = torch.cat([out, cid], dim=1)
                                tmp_kv = past_kv
                                steps = 1
                                while steps < int(tree_depth) and steps < int(rg_budget):
                                    try:
                                        get_logger("omnicoder.gen").debug(
                                            "tree expand step=%s/%s rg_budget=%s",
                                            int(steps + 1), int(tree_depth), int(rg_budget)
                                        )
                                    except Exception:
                                        pass
                                    outs2 = model(tmp_ids[:, -1:], past_kv=tmp_kv, use_cache=True)
                                    if isinstance(outs2, tuple):
                                        log2 = outs2[0]
                                        tmp_kv = outs2[1]
                                        v2 = outs2[3] if len(outs2) > 3 else None
                                    else:
                                        log2 = outs2
                                        v2 = None
                                    pr2 = torch.ops.aten._softmax.default(log2[:, -1, :], -1, False)
                                    # propose next draft greedily
                                    nid = torch.argmax(pr2, dim=-1, keepdim=True)
                                    # accumulate margin using verifier head if available
                                    if v2 is not None:
                                        m2 = float(torch.ops.aten._softmax.default(v2[:, -1, :], -1, False).gather(-1, nid).item())
                                    else:
                                        m2 = float(pr2.gather(-1, nid).item())
                                    margin_sum += m2
                                    tmp_ids = torch.cat([tmp_ids, nid], dim=1)
                                    steps += 1
                                    cost += 1
                            else:
                                if vp is not None:
                                    margin_sum = float(vp.gather(-1, cid).item())
                                else:
                                    margin_sum = float(probs.gather(-1, cid).item())
                            scores.append((float(margin_sum), float(score_proxy), int(cost)))
                        except Exception:
                            # fallback to single-step proxy
                            if vp is not None:
                                m = float(vp.gather(-1, cid).item())
                                s = float(probs.gather(-1, cid).item())
                            else:
                                m = float(probs.gather(-1, cid).item())
                                s = m
                            scores.append((m, s, 1))
                    # Use graph-speculative selector if available
                    if select_branch is not None:
                        idx = int(select_branch(scores))
                    else:
                        # Fallback: choose highest margin/ cost ratio, tie-break by score proxy
                        idx = max(enumerate(scores), key=lambda kv: (kv[1][0] / max(1, kv[1][2]), kv[1][1]))[0]
                    best_id = cands[idx]
                except Exception:
                    pass
            # Commit main token
            # out = torch.cat([out, best_id], dim=1)
            gen_buf[:, gen_len:gen_len+1] = best_id
            gen_len += 1
            generated_seq.append(best_id)

            # When a full sequence view is required (e.g., block verify), create a lightweight temp by pointing to orig+buffer
            def _current_seq_view():
                if gen_len == 0:
                    return orig_input
                return torch.cat([orig_input, gen_buf[:, :gen_len]], dim=1)

            # Step 2: speculative lookahead (draft-and-verify)
            # Default to enabling MTP-based drafts when verify_threshold is 0 and model exposes MTP heads
            if speculative_draft_len <= 0 and verifier_logits is not None and verify_threshold == 0.0 and mtp_logits is not None:
                speculative_draft_len = min(2, len(mtp_logits))
            if speculative_draft_len > 0:
                if draft_model is not None:
                    # Use a separate small draft model to propose tokens (block verification optional)
                    with torch.no_grad():
                        # Propose a block of draft tokens
                        block_len = max(1, min(int(block_verify_size or 1), int(speculative_draft_len)))
                        proposed_ids: list[torch.Tensor] = []
                        cur_ids = out[:, -1:]
                        for _bi in range(block_len):
                            d_out = draft_model(cur_ids, past_kv=None, use_cache=False)
                            d_logits = d_out[0] if isinstance(d_out, tuple) else d_out
                            draft_id = sample_next_token(d_logits[:, -1, :], temperature, top_k, top_p)
                            proposed_ids.append(draft_id)
                            cur_ids = draft_id
                    # Verify
                    attempted_speculative += len(proposed_ids)
                    accepted_prefix = 0
                    if verifier_logits is not None:
                        # Verify first token using current verifier head
                        v_probs0 = torch.ops.aten._softmax.default(verifier_logits[:, -1, :], -1, False)
                        v_prob0 = v_probs0.gather(-1, proposed_ids[0])
                        if torch.all(v_prob0 >= verify_threshold):
                            accepted_prefix = 1
                            # Optionally verify remaining in a rolling fashion by advancing main model
                            if block_verify and len(proposed_ids) > 1:
                                # Build a temporary view for advancing without mutating gen_buf
                                tmp_ids = torch.ops.aten.mul.Scalar(_current_seq_view(), 1.0)
                                tmp_kv = past_kv
                                for di in range(accepted_prefix, len(proposed_ids)):
                                    tmp_ids = torch.cat([tmp_ids, proposed_ids[di - 1]], dim=1)
                                    outs2 = model(tmp_ids[:, -1:], past_kv=tmp_kv, use_cache=True)
                                    if isinstance(outs2, tuple):
                                        _, tmp_kv, _, v2 = outs2  # type: ignore
                                    else:
                                        v2 = None
                                    if v2 is None:
                                        break
                                    v2_probs = torch.ops.aten._softmax.default(v2[:, -1, :], -1, False)
                                    v2_prob = v2_probs.gather(-1, proposed_ids[di])
                                    if torch.all(v2_prob >= verify_threshold):
                                        accepted_prefix += 1
                                    else:
                                        break
                            # EMA update on first acceptance
                            if speculative_auto:
                                if ema_vprob is None:
                                    ema_vprob = v_prob0.detach()
                                else:
                                    ema_vprob = 0.9 * ema_vprob + 0.1 * v_prob0.detach()
                    # Commit accepted prefix and advance main KV for each accepted token
                    if accepted_prefix > 0:
                        for i in range(accepted_prefix):
                            # out = torch.cat([out, proposed_ids[i]], dim=1)
                            gen_buf[:, gen_len:gen_len+1] = proposed_ids[i]
                            gen_len += 1
                            generated_seq.append(proposed_ids[i])
                            accepted_speculative += 1
                            # advance KV to stay consistent (dequantize when needed)
                            _feed_kv = past_kv
                            if _feed_kv is not None and kvq in ('u8','nf4'):
                                dq_list2 = []
                                for (kq2, vq2, meta2) in _feed_kv:  # type: ignore[assignment]
                                    kf2, vf2 = dequantize_kv(kq2, vq2, meta2)  # type: ignore[arg-type]
                                    dq_list2.append((kf2, vf2))
                                _feed_kv = dq_list2  # type: ignore[assignment]
                            adv = model(proposed_ids[i], past_kv=_feed_kv, use_cache=True)
                            if isinstance(adv, tuple):
                                _, past_kv = adv[0], adv[1]  # type: ignore
                elif mtp_logits is not None:
                    # Use MTP heads from main model as light drafts (contiguous prefix block verification)
                    max_drafts = min(int(speculative_draft_len), len(mtp_logits))
                    # Propose up to block size
                    block_len = max(1, min(int(block_verify_size or 1), max_drafts))
                    proposed_ids = []
                    for la in mtp_logits[:block_len]:
                        draft_id = sample_next_token(la[:, -1, :], temperature, top_k, top_p)
                        proposed_ids.append(draft_id)
                    attempted_speculative += len(proposed_ids)
                    accepted_prefix = 0
                    if verifier_logits is not None:
                        # First token uses current verifier head
                        v_probs0 = torch.ops.aten._softmax.default(verifier_logits[:, -1, :], -1, False)
                        v_prob0 = v_probs0.gather(-1, proposed_ids[0])
                        if torch.all(v_prob0 >= verify_threshold):
                            accepted_prefix = 1
                            if block_verify and len(proposed_ids) > 1:
                                # advance model one by one to verify subsequent drafts
                                tmp_ids = torch.ops.aten.mul.Scalar(_current_seq_view(), 1.0)
                                tmp_kv = past_kv
                                for di in range(accepted_prefix, len(proposed_ids)):
                                    tmp_ids = torch.cat([tmp_ids, proposed_ids[di - 1]], dim=1)
                                    outs2 = model(tmp_ids[:, -1:], past_kv=tmp_kv, use_cache=True)
                                    if isinstance(outs2, tuple):
                                        _, tmp_kv, _, v2 = outs2  # type: ignore
                                    else:
                                        v2 = None
                                    if v2 is None:
                                        break
                                    v2_probs = torch.ops.aten._softmax.default(v2[:, -1, :], -1, False)
                                    v2_prob = v2_probs.gather(-1, proposed_ids[di])
                                    if torch.all(v2_prob >= verify_threshold):
                                        accepted_prefix += 1
                                    else:
                                        break
                    if accepted_prefix > 0:
                        for i in range(accepted_prefix):
                            # out = torch.cat([out, proposed_ids[i]], dim=1)
                            gen_buf[:, gen_len:gen_len+1] = proposed_ids[i]
                            gen_len += 1
                            generated_seq.append(proposed_ids[i])
                            accepted_speculative += 1
                            # advance KV to remain consistent (dequantize when needed)
                            _feed_kv = past_kv
                            if _feed_kv is not None and kvq in ('u8','nf4'):
                                dq_list2 = []
                                for (kq2, vq2, meta2) in _feed_kv:  # type: ignore[assignment]
                                    kf2, vf2 = dequantize_kv(kq2, vq2, meta2)  # type: ignore[arg-type]
                                    dq_list2.append((kf2, vf2))
                                _feed_kv = dq_list2  # type: ignore[assignment]
                            adv = model(proposed_ids[i], past_kv=_feed_kv, use_cache=True)
                            if isinstance(adv, tuple):
                                _, past_kv = adv[0], adv[1]  # type: ignore
    # Always return original input concatenated with newly generated tokens, even when windowed decode was used
    if gen_len > 0:
        new_tokens = gen_buf[:, :gen_len]
        output = torch.cat([orig_input, new_tokens], dim=1)
    else:
        output = orig_input
    # Parallel diffusion candidate blending (post-pass using verifier head): choose higher mean verifier prob
    try:
        if '_diffusion_candidate' in locals() and isinstance(_diffusion_candidate, torch.Tensor) and _diffusion_candidate.numel() > 0:
            main_full = output
            diff_full = torch.cat([orig_input, _diffusion_candidate.to(output.device)], dim=1)
            def _score_with_verifier(seq: torch.Tensor) -> torch.Tensor:
                # Model always computes verifier logits; flags removed from API for CG/export safety.
                outs = model(seq, past_kv=None, use_cache=False)
                v = None
                if isinstance(outs, tuple) and len(outs) > 3:
                    v = outs[3]
                if v is None:
                    return torch.tensor(0.0, device=seq.device)
                # Compute softmax over verifier logits and gather prob of realized ids after prompt length
                v_probs = torch.ops.aten._softmax.default(v, -1, False)
                p_len = int(orig_input.size(1))
                tail_ids = seq[:, p_len:]
                if tail_ids.numel() == 0:
                    return torch.tensor(0.0, device=seq.device)
                gathered = v_probs[:, p_len-1: v_probs.size(1)-1, :].gather(-1, tail_ids.unsqueeze(-1))
                # Mean probability across generated steps
                return gathered.mean()
            s_main = _score_with_verifier(main_full)
            s_diff = _score_with_verifier(diff_full)
            if torch.all(s_diff > s_main):
                output = diff_full
                try:
                    _log.info("generate: selected diffusion candidate (verifier mean %.5f > %.5f)", float(s_diff), float(s_main))
                except Exception:
                    pass
            else:
                try:
                    _log.info("generate: kept AR output (verifier mean %.5f >= %.5f)", float(s_main), float(s_diff))
                except Exception:
                    pass
    except Exception as _e:
        try:
            _log.warning("generate: diffusion blend skipped: %s", str(_e))
        except Exception:
            pass
    try:
        get_logger("omnicoder.gen").debug("generate() exit out_shape=%s", str(tuple(output.shape)))
    except Exception:
        pass
    # Emit per-run decode throughput summary
    try:
        if 'decode_steps' in locals() and 'decode_time_sum' in locals() and decode_steps > 0:
            tps = (decode_steps / max(decode_time_sum, 1e-9))
            _log.info("decode summary steps=%d avg_step_ms=%.3f tokens_per_sec=%.1f", int(decode_steps), float(1000.0*decode_time_sum/max(decode_steps,1)), float(tps))
    except Exception:
        pass
    # Optional final reflection/self-validation pass (lightweight):
    # We run this OUTSIDE the inner token loop to avoid TPS impact. It can amend
    # the last token based on verifier probability and optionally record stats.
    try:
        if 'generated_seq' in locals():
            _gen_ids_list = list(generated_seq)
        else:
            # Fallback: derive generated ids by cropping prompt length from output
            _gen_ids_list = []
            try:
                if isinstance(orig_input, torch.Tensor) and isinstance(output, torch.Tensor):
                    _pl = int(orig_input.size(1))
                    tail = torch.ops.aten.slice.Tensor(output, 1, int(_pl), int(output.size(1)), 1)
                    if tail.numel() > 0:
                        _gen_ids_list = [int(x) for x in tail[0].tolist()]
            except Exception:
                _gen_ids_list = []
        # Only attempt reflection when we generated at least 1 token
        if _gen_ids_list:
            # Use last computed hidden_out if available; otherwise skip hidden-based checks
            _last_hidden = hidden_out if ('hidden_out' in locals()) else None  # type: ignore[name-defined]
            # Minimum verifier probability for acceptance; prefer provided verify_threshold else default
            _vmin = float(verify_threshold) if float(verify_threshold) > 0.0 else 0.05
            # Enable tool tags post-processing only when caller requested via runtime_config
            _enable_tools = bool(getattr(runtime_config, 'tool_use', False)) if runtime_config is not None else False
            try:
                # Build registry lazily only if enabled
                _reg_builder = (_build_tool_registry if _enable_tools else None)
            except Exception:
                _reg_builder = None
            try:
                _ids_adj, _refl_stats = _reflect_and_validate(
                    model,
                    _last_hidden,
                    _gen_ids_list,
                    verify_min_prob=_vmin,
                    enable_tools=_enable_tools,
                    tool_registry_builder=_reg_builder,
                )
                # If reflection altered the last token, update the returned tensor accordingly
                if _ids_adj and (len(_ids_adj) == len(_gen_ids_list)) and isinstance(output, torch.Tensor):
                    try:
                        _pl = int(orig_input.size(1))
                        for i, tid in enumerate(_ids_adj):
                            output[0, _pl + i] = int(tid)
                    except Exception:
                        pass
            except Exception:
                _refl_stats = {"reflect_error": True}
        else:
            _refl_stats = {}
    except Exception:
        _refl_stats = {}

    if return_stats:
        kvq_scheme = str(kvq)
        stats = {
            "accepted_speculative": int(accepted_speculative),
            "attempted_speculative": int(attempted_speculative),
            "kvq_group": int(kvq_group),
            "kvq_scheme": kvq_scheme,
            "windowed": bool(int(window_size) > 0),
            "cis_hits": int(cis_hits),
        }
        # Merge reflection stats if any
        try:
            if isinstance(_refl_stats, dict) and _refl_stats:
                stats.update({f"reflect.{k}": v for k, v in _refl_stats.items()})
        except Exception:
            pass
        return output, stats
    return output
def _postprocess_tool_use(text: str) -> str:
    try:
        reg = _build_tool_registry()
        repl = reg.parse_and_invoke_all(text)
        out = text
        import json as _json
        for tag, val in repl.items():
            try:
                out = out.replace(tag, _json.dumps(val))
            except Exception:
                continue
        return out
    except Exception:
        return text


@torch.inference_mode()
def prime_kv_with_features(
    model: OmniTransformer,
    prefix_features: torch.Tensor,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """
    Run the model once on a pre-embedded feature sequence to prime the KV cache.
    Returns (past_kv, logits) for continuing decoding.
    """
    model.eval()
    device = next(model.parameters()).device
    feats = prefix_features.to(device, non_blocking=True)
    # Mark cudagraph step (if available) before model invocation on CUDA
    try:
        _cuda_like = str(device).startswith('cuda') and torch.cuda.is_available()
    except Exception:
        _cuda_like = False
    try:
        _cg_mark = (getattr(getattr(torch, 'compiler', None), 'cudagraph_mark_step_begin', None) if _cuda_like else None)
    except Exception:
        _cg_mark = None
    try:
        if _cg_mark is not None:
            _cg_mark()  # type: ignore[misc]
    except Exception:
        pass
    outputs = model(feats, past_kv=None, use_cache=True)  # type: ignore[arg-type]
    # Expected shape: (logits, new_kv, mtp_logits?, verifier_logits?)
    if isinstance(outputs, tuple):
        logits = outputs[0]  # type: ignore[index]
        past_kv = outputs[1]  # type: ignore[index]
    else:
        logits, past_kv = outputs  # type: ignore
    return past_kv, logits


@torch.inference_mode()
def continue_generate_from_primed(
    model: OmniTransformer,
    past_kv: list[tuple[torch.Tensor, torch.Tensor]],
    start_token_id: int,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
) -> torch.Tensor:
    """Continue autoregressive generation using an already primed KV cache.

    Returns a tensor of shape (1, max_new_tokens) containing the newly generated ids.
    """
    model.eval()
    device = next(model.parameters()).device
    generated = []
    cur_id = torch.tensor([[start_token_id]], dtype=torch.long, device=device)
    cur_kv = past_kv
    # Prepare cudagraph step marker once
    try:
        _cuda_like = str(device).startswith('cuda') and torch.cuda.is_available()
    except Exception:
        _cuda_like = False
    try:
        _cg_mark = (getattr(getattr(torch, 'compiler', None), 'cudagraph_mark_step_begin', None) if _cuda_like else None)
    except Exception:
        _cg_mark = None
    with torch.no_grad():
        for _ in range(max_new_tokens):
            try:
                get_logger("omnicoder.gen").debug("continue_from_prefix step rem=%s", int(max_new_tokens))
            except Exception:
                pass
            try:
                if _cg_mark is not None:
                    _cg_mark()  # type: ignore[misc]
            except Exception:
                pass
            outputs = model(cur_id, past_kv=cur_kv, use_cache=True)
            if isinstance(outputs, tuple):
                logits = outputs[0]  # type: ignore
                cur_kv = outputs[1]  # type: ignore
            else:
                logits = outputs  # type: ignore
            next_id = sample_next_token(logits[:, -1, :], temperature, top_k, top_p)
            generated.append(next_id)
            cur_id = next_id
    return torch.cat(generated, dim=1)


def maybe_load_checkpoint(model: OmniTransformer, ckpt_path: Optional[str]) -> dict:
    """Load a checkpoint best-effort and return coverage stats.

    Returns a dict with keys:
      - path: str
      - loaded_keys: int
      - total_model_keys: int
      - loaded_param_elems: int
      - total_param_elems: int
      - missing: int
      - unexpected: int
    """
    log = get_logger("omnicoder.ckpt")
    if not ckpt_path:
        return {
            "path": "",
            "loaded_keys": 0,
            "total_model_keys": 0,
            "loaded_param_elems": 0,
            "total_param_elems": 0,
            "missing": 0,
            "unexpected": 0,
        }
    try:
        log.info("load_ckpt enter path=%s", ckpt_path)
        # If a manifest exists next to checkpoint, apply recorded env/tokenizer
        try:
            from omnicoder.utils.model_manifest import load_and_apply_manifest  # type: ignore
            from pathlib import Path as _P
            _mp = _P(str(ckpt_path)).with_suffix("").as_posix() + ".manifest.json"
            load_and_apply_manifest(_mp)
        except Exception:
            pass
        state = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        log.error("torch.load failed: %s", str(e))
        raise
    
    # Normalize common checkpoint formats (Lightning, DDP prefixes, nested keys)
    # - Extract nested dicts under typical keys (state_dict, model, module, ema)
    # - Strip common prefixes (model., module., student., transformer., net., backbone.)
    def _extract_state_dict(obj: dict) -> dict:
        """Return the most plausible parameter dict from a loaded checkpoint mapping."""
        if not isinstance(obj, dict):
            return {}
        # Direct param dict heuristic: many tensor-like values
        def _score(d: dict) -> int:
            n = 0
            for v in d.values():
                try:
                    if hasattr(v, 'shape'):
                        n += 1
                except Exception:
                    continue
            return n
        candidates = []
        # Self
        candidates.append(obj)
        # Typical wrappers
        for k in ("state_dict", "model", "module", "ema", "network", "student", "teacher"):  # teacher unlikely but harmless
            try:
                inner = obj.get(k)
                if isinstance(inner, dict):
                    candidates.append(inner)
            except Exception:
                pass
        best = max(candidates, key=_score)
        return best

    def _strip_prefixes(sd: dict, model_keys: set[str]) -> dict:
        if not isinstance(sd, dict):
            return {}
        prefixes = (
            "model.", "module.", "student.", "transformer.", "net.", "backbone.", "omni.", "_orig_mod.",
        )
        out: dict = {}
        for k, v in sd.items():
            nk = k
            try:
                for p in prefixes:
                    if nk.startswith(p):
                        cand = nk[len(p):]
                        # Accept first prefix strip that yields an existing model key suffix match
                        if cand in model_keys:
                            nk = cand
                            break
                out[nk] = v
            except Exception:
                out[k] = v
        return out
    
    # Unwrap compiled wrappers (torch.compile / dynamo GraphModule) to access real module state_dict
    def _unwrap_model(m: OmniTransformer) -> OmniTransformer:
        try:
            # Common attributes used by torch.compile or traced modules
            for attr in ("_orig_mod", "_original_module", "module"):
                try:
                    inner = getattr(m, attr)
                    # Heuristic: inner should expose state_dict with more keys than wrapper
                    if hasattr(inner, "state_dict"):
                        return inner  # type: ignore[return-value]
                except Exception:
                    continue
        except Exception:
            pass
        return m
    # Be robust to vocab-size or head-shape mismatches (e.g., unified multimodal vocab)
    # Only load parameters whose shapes match the current model to avoid RuntimeError.
    loaded_keys = 0
    total_model_keys = 0
    loaded_param_elems = 0
    total_param_elems = 0
    missing_count = 0
    unexpected_count = 0
    if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        real_model = _unwrap_model(model)
        model_sd = real_model.state_dict()
        # Attempt to unwrap nested dicts and strip prefixes before filtering
        raw = _extract_state_dict(state)
        try:
            model_keys = set(model_sd.keys())
            raw = _strip_prefixes(raw, model_keys)
        except Exception:
            pass
        # Best-effort augmentation: if many keys are missing in the base checkpoint,
        # try to hydrate missing parameters from per-key tensor files on disk.
        # We search next to the checkpoint and under weights/ using filenames that
        # exactly match model keys (optionally with a .pt suffix).
        try:
            import os as _os
            import torch as _t
            augmented = 0
            ckpt_dir = _os.path.dirname(_os.path.abspath(str(ckpt_path))) if ckpt_path else ''
            def _try_load_param_file(param_key: str):
                # search order: ckpt_dir/<key>, ckpt_dir/<key>.pt, weights/<key>, weights/<key>.pt
                cand = [
                    _os.path.join(ckpt_dir, param_key),
                    _os.path.join(ckpt_dir, f"{param_key}.pt"),
                    _os.path.join('weights', param_key),
                    _os.path.join('weights', f"{param_key}.pt"),
                ]
                for p in cand:
                    try:
                        if _os.path.exists(p):
                            obj = _t.load(p, map_location='cpu')
                            if hasattr(obj, 'shape'):
                                return obj
                    except Exception:
                        continue
                return None
            # Build a working copy we can mutate
            raw = dict(raw)
            for mk in model_sd.keys():
                if mk not in raw:
                    val = _try_load_param_file(mk)
                    if val is not None and hasattr(model_sd[mk], 'shape') and tuple(getattr(val, 'shape', ())) == tuple(model_sd[mk].shape):
                        raw[mk] = val
                        augmented += 1
            if augmented > 0:
                log.warning("ckpt: augmented missing params from disk files count=%d", int(augmented))
        except Exception:
            pass
        filtered = {}
        tried = 0
        for k, v in raw.items():
            if k in model_sd and hasattr(v, 'shape') and hasattr(model_sd[k], 'shape'):
                if tuple(v.shape) == tuple(model_sd[k].shape):
                    filtered[k] = v
                else:
                    # Allow vocab-head remap: when lm_head weight size differs only in vocab dim, slice or pad
                    try:
                        if k.endswith('lm_head.weight') and len(v.shape) == 2 and len(model_sd[k].shape) == 2:
                            src_v, src_d = int(v.shape[0]), int(v.shape[1])
                            dst_v, dst_d = int(model_sd[k].shape[0]), int(model_sd[k].shape[1])
                            if src_d == dst_d:
                                if src_v >= dst_v:
                                    filtered[k] = v[:dst_v, :]
                                else:
                                    import torch as _t
                                    pad = _t.zeros((dst_v - src_v, dst_d), dtype=v.dtype)
                                    filtered[k] = _t.cat([v, pad], dim=0)
                        # Also allow embedding weight/bias remaps to handle vocab-size mismatches
                        elif (k.endswith('embed.weight') or k.endswith('tok_embed.weight')) and len(v.shape) == 2 and len(model_sd[k].shape) == 2:
                            src_v, src_d = int(v.shape[0]), int(v.shape[1])
                            dst_v, dst_d = int(model_sd[k].shape[0]), int(model_sd[k].shape[1])
                            if src_d == dst_d:
                                if src_v >= dst_v:
                                    filtered[k] = v[:dst_v, :]
                                else:
                                    import torch as _t
                                    pad = _t.zeros((dst_v - src_v, dst_d), dtype=v.dtype)
                                    filtered[k] = _t.cat([v, pad], dim=0)
                        elif k.endswith('lm_head.bias') and len(v.shape) == 1 and len(model_sd[k].shape) == 1:
                            src_v = int(v.shape[0])
                            dst_v = int(model_sd[k].shape[0])
                            if src_v >= dst_v:
                                filtered[k] = v[:dst_v]
                            else:
                                import torch as _t
                                pad = _t.zeros((dst_v - src_v,), dtype=v.dtype)
                                filtered[k] = _t.cat([v, pad], dim=0)
                    except Exception:
                        pass
        # Log a brief summary of state_dict shapes for debugging
        try:
            shp = {k: tuple(v.shape) if hasattr(v, 'shape') else None for k, v in raw.items()}
            log.info("ckpt state keys=%d examples=%s", len(shp), list(shp.items())[:6])
        except Exception:
            pass
        # Load matching tensors; ignore missing/unexpected keys
        try:
            # Track presence of head/embed in checkpoint to detect uninitialized head cases
            ck_keys = set(filtered.keys())
            had_head = any(k.endswith('lm_head.weight') for k in ck_keys)
            had_embed = any(k.endswith('embed.weight') or k.endswith('tok_embed.weight') for k in ck_keys)
            missing, unexpected = real_model.load_state_dict(filtered, strict=False)
            loaded_keys = len(filtered)
            total_model_keys = len(model_sd)
            try:
                loaded_param_elems = int(sum(getattr(t, 'numel', lambda: 0)() for t in filtered.values()))
                total_param_elems = int(sum(getattr(t, 'numel', lambda: 0)() for t in model_sd.values()))
            except Exception:
                loaded_param_elems = 0
                total_param_elems = 0
            missing_count = len(missing)
            unexpected_count = len(unexpected)
            # Emit verbose details for triage of gibberish generations
            try:
                if isinstance(missing, (list, tuple)) and len(missing) > 0:
                    log.warning("ckpt missing keys (first 20)=%s", [str(m) for m in list(missing)[:20]])
                if isinstance(unexpected, (list, tuple)) and len(unexpected) > 0:
                    log.warning("ckpt unexpected keys (first 20)=%s", [str(u) for u in list(unexpected)[:20]])
                # Also dump a compact histogram of tensor shape mismatches we tried to adapt
                try:
                    from collections import Counter as _Counter
                    mm = _Counter()
                    for k, v in state.items():
                        try:
                            if k in model_sd and hasattr(v, 'shape') and hasattr(model_sd[k], 'shape') and tuple(v.shape) != tuple(model_sd[k].shape):
                                mm[(tuple(v.shape), tuple(model_sd[k].shape))] += 1
                        except Exception:
                            pass
                    if mm:
                        log.warning("ckpt shape_mismatch summary (src->dst,count)=%s", list(mm.items())[:8])
                except Exception:
                    pass
            except Exception:
                pass
            cov_pct = (100.0 * (float(loaded_param_elems) / max(1.0, float(total_param_elems)))) if total_param_elems else 0.0
            log.info(
                "ckpt loaded filtered keys=%d missing=%d unexpected=%d param_coverage=%.2f%%",
                loaded_keys,
                missing_count,
                unexpected_count,
                cov_pct,
            )
            try:
                # Emit quick stats on embed/head weights to detect random heads (source of gibberish)
                import torch as _t
                e_w = getattr(getattr(real_model, 'embed', object()), 'weight', None)
                h_w = getattr(getattr(real_model, 'lm_head', object()), 'weight', None)
                if e_w is not None and h_w is not None:
                    ew = e_w.data.float()
                    hw = h_w.data.float()
                    log.info("ckpt stats embed[mean=%.6f std=%.6f] lm_head[mean=%.6f std=%.6f]",
                             float(ew.mean().item()), float(ew.std().item()), float(hw.mean().item()), float(hw.std().item()))
            except Exception:
                pass
            # If lm_head was not loaded but embed was, optionally tie lm_head to embed to avoid random head
            try:
                import os as _os
                tie_ok = _os.getenv('OMNICODER_TIE_LMHEAD', '1') == '1'
            except Exception:
                tie_ok = True
            try:
                if tie_ok and (not had_head) and had_embed and hasattr(real_model, 'lm_head') and hasattr(real_model, 'embed'):
                    e_w = getattr(getattr(real_model, 'embed', object()), 'weight', None)
                    h_w = getattr(getattr(real_model, 'lm_head', object()), 'weight', None)
                    if e_w is not None and h_w is not None and tuple(e_w.shape) == tuple(h_w.shape):
                        real_model.lm_head.weight = real_model.embed.weight  # type: ignore[attr-defined]
                        log.warning("lm_head.weight missing in ckpt; tied lm_head to embed.weight to avoid untrained head")
                    else:
                        log.warning("lm_head.weight missing but cannot tie due to shape mismatch embed=%s head=%s",
                                    str(tuple(e_w.shape) if e_w is not None else None),
                                    str(tuple(h_w.shape) if h_w is not None else None))
            except Exception as _e:
                log.warning("lm_head tie failed: %s", _e)
            # Emit quick parameter stats to catch all-random models (mean/std)
            try:
                import torch as _t
                emb_w = getattr(getattr(real_model, 'embed', object()), 'weight', None)
                head_w = getattr(getattr(real_model, 'lm_head', object()), 'weight', None)
                def _stats(t: _t.Tensor | None) -> tuple[float, float]:
                    if t is None:
                        return (float('nan'), float('nan'))
                    return (float(t.mean().item()), float(t.std().item()))
                m_emb, s_emb = _stats(emb_w)
                m_head, s_head = _stats(head_w)
                log.info("ckpt stats embed[mean=%.6f std=%.6f] lm_head[mean=%.6f std=%.6f]", m_emb, s_emb, m_head, s_head)
                # Detect untrained head by near-zero std and attempt to tie to embed
                try:
                    if (s_head is not None) and float(s_head) < 1e-6 and emb_w is not None and head_w is not None:
                        if tuple(emb_w.shape) == tuple(head_w.shape):
                            real_model.lm_head.weight = real_model.embed.weight  # type: ignore[attr-defined]
                            log.warning("ckpt: lm_head.std ~ 0 — tied lm_head to embed to avoid gibberish")
                except Exception:
                    pass
                # Record presence flags to gate optional bias modules
                try:
                    setattr(model, '_ckpt_has_dual_substrate', any(k.endswith('byte_proj.weight') for k in ck_keys))
                except Exception:
                    setattr(model, '_ckpt_has_dual_substrate', False)
                try:
                    setattr(model, '_ckpt_has_perceiver_prior', any(k.endswith('perceiver_to_vocab.weight') for k in ck_keys))
                except Exception:
                    setattr(model, '_ckpt_has_perceiver_prior', False)
                # MoE/router missing: optionally collapse MoE to dense fast path
                try:
                    import os as _os
                    force_dense = (_os.getenv('OMNICODER_MOE_FALLBACK_TO_DENSE', '0') == '1')
                except Exception:
                    force_dense = False
                try:
                    moe_missing = [k for k in list(missing) if isinstance(k, str) and ('router' in k or 'experts' in k or 'gate_group' in k)] if isinstance(missing, (list, tuple)) else []
                    # Only collapse to dense when explicitly forced. Auto-collapse often destroys quality.
                    auto_dense = False
                    if force_dense:
                        log.warning('ckpt: collapsing MoE to single-expert fast path (force=%s auto=%s missing_moe_keys=%d)', bool(force_dense), bool(auto_dense), len(moe_missing))
                        try:
                            from omnicoder.modeling.transformer_moe import MoELayer  # type: ignore
                        except Exception:
                            MoELayer = None  # type: ignore
                        if MoELayer is not None:
                            for m in model.modules():
                                try:
                                    if isinstance(m, MoELayer):
                                        m.collapse_to_single_expert()
                                except Exception:
                                    continue
                    else:
                        # Do NOT mark degraded routers by default; keep trained routers/expert weights that loaded.
                        # Allow opt-in uniform degraded routing via env for debugging only.
                        try:
                            allow_degraded = (os.getenv('OMNICODER_ALLOW_DEGRADED_ROUTER', '0') == '1')  # type: ignore[name-defined]
                        except Exception:
                            allow_degraded = False  # type: ignore[assignment]
                        if allow_degraded and len(moe_missing) > 0:
                            try:
                                from omnicoder.modeling.transformer_moe import MoELayer  # type: ignore
                            except Exception:
                                MoELayer = None  # type: ignore
                            if MoELayer is not None:
                                marked = 0
                                for m in model.modules():
                                    try:
                                        if isinstance(m, MoELayer):
                                            setattr(m, '_degraded_router', True)
                                            marked += 1
                                    except Exception:
                                        continue
                                if marked > 0:
                                    log.warning('ckpt: marked %d MoE routers as degraded (uniform fallback) due to partial missing keys (env override)', marked)
                        else:
                            # Explicitly log that we are NOT degrading routers; avoid silent uniform routing
                            if len(moe_missing) > 0:
                                log.warning('ckpt: NOT degrading MoE routers (missing_moe_keys=%d). Consider providing matching MoE checkpoints or set OMNICODER_MOE_FALLBACK_TO_DENSE=1 to collapse to dense.', len(moe_missing))
                except Exception:
                    pass
            except Exception:
                pass
        except Exception as e:
            log.error("load_state_dict failed: %s", str(e))
            raise
    else:
        try:
            missing, unexpected = model.load_state_dict(state, strict=False)
            loaded_keys = len(getattr(state, 'keys', lambda: [])())
            total_model_keys = len(model.state_dict())
            try:
                loaded_param_elems = int(sum(getattr(t, 'numel', lambda: 0)() for t in state.values()))  # type: ignore[arg-type]
                total_param_elems = int(sum(getattr(t, 'numel', lambda: 0)() for t in model.state_dict().values()))
            except Exception:
                loaded_param_elems = 0
                total_param_elems = 0
            missing_count = len(missing)
            unexpected_count = len(unexpected)
            log.info(
                "ckpt loaded best-effort keys=%d missing=%d unexpected=%d param_coverage=%.2f%%",
                loaded_keys,
                missing_count,
                unexpected_count,
                (100.0 * (float(loaded_param_elems) / max(1.0, float(total_param_elems)))) if total_param_elems else 0.0,
            )
        except Exception as e:
            log.error("load_state_dict (fallback) failed: %s", str(e))
            raise
    try:
        # Explicit exit log with a quick sanity of first param device and dtype count
        dev = str(next(model.parameters()).device)
        n_params = sum(p.numel() for p in model.parameters())
        log.info("load_ckpt exit device=%s params=%d", dev, int(n_params))
        # Extra: sample a tiny sanity prompt perplexity proxy to detect obviously uninitialized heads
        try:
            from omnicoder.training.simple_tokenizer import TextTokenizer  # local minimal tokenizer to avoid HF deps
            tok = TextTokenizer(vocab_size=getattr(model, 'vocab_size', 32000))
            probe = tok.encode("Hello world. This is a test.")
            import torch as _t
            x = _t.tensor([probe[:-1]], dtype=_t.long, device=next(model.parameters()).device)
            y = _t.tensor([probe[1:]], dtype=_t.long, device=x.device)
            with _t.no_grad():
                logits = model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                lp = _t.log_softmax(logits, dim=-1)
                t = y.shape[1]
                nll = float((-lp[0, _t.arange(t, device=y.device), y[0]]).mean().item())
            log.info("ckpt sanity_nll=%.4f on tiny probe", nll)
        except Exception:
            pass
    except Exception:
        pass
    return {
        "path": str(ckpt_path),
        "loaded_keys": int(loaded_keys),
        "total_model_keys": int(total_model_keys),
        "loaded_param_elems": int(loaded_param_elems),
        "total_param_elems": int(total_param_elems),
        "missing": int(missing_count),
        "unexpected": int(unexpected_count),
    }


def build_mobile_model(
    preset: MobilePreset,
    rope_scale: float | None = None,
    rope_base: float | None = None,
    multi_token: int = 2,
    mem_slots: int = 0,
    skip_init: bool = False,
) -> OmniTransformer:
    log = get_logger("omnicoder.preset")
    try:
        get_logger("omnicoder.preset").info(
            "build_mobile_model enter name=%s layers=%s d_model=%s",
            getattr(preset, 'name', ''), int(getattr(preset, 'n_layers', -1)), int(getattr(preset, 'd_model', -1))
        )
    except Exception:
        pass
    try:
        log.info(
            "build_mobile_model name=%s layers=%s d_model=%s n_heads=%s mlp_dim=%s moe_experts=%s top_k=%s kv_latent_dim=%s max_seq_len=%s multi_query=%s multi_token=%s rope_scale=%s rope_base=%s mem_slots=%s moe_group_sizes=%s",
            getattr(preset, 'name', ''),
            int(getattr(preset, 'n_layers', -1)),
            int(getattr(preset, 'd_model', -1)),
            int(getattr(preset, 'n_heads', -1)),
            int(getattr(preset, 'mlp_dim', -1)),
            int(getattr(preset, 'moe_experts', -1)),
            int(getattr(preset, 'moe_top_k', -1)),
            int(getattr(preset, 'kv_latent_dim', -1)),
            int(getattr(preset, 'max_seq_len', -1)),
            bool(getattr(preset, 'multi_query', True)),
            int(multi_token),
            float(rope_scale if rope_scale is not None else 1.0),
            float(rope_base if rope_base is not None else 10000.0),
            int(mem_slots),
            list(getattr(preset, 'moe_group_sizes', [])),
        )
        dm = int(getattr(preset, 'd_model', 0))
        nh = int(getattr(preset, 'n_heads', 1))
        if nh <= 0 or dm <= 0 or (dm % nh) != 0:
            log.warning("invalid head config: d_model %s not divisible by n_heads %s", dm, nh)
    except Exception:
        pass
    # Honor HRM enable override via env (default behavior controlled by centralized defaults/profile)
    try:
        _hrm_enabled = (os.getenv('OMNICODER_HRM_ENABLE', '1') == '1')
    except Exception:
        _hrm_enabled = True
    # Optional override: force dense FFN (no MoE) when checkpoints lack MoE/router weights
    try:
        _force_dense = (os.getenv('OMNICODER_FORCE_DENSE_MLP', '0') == '1')
    except Exception:
        _force_dense = False
    model = OmniTransformer(
        vocab_size=preset.vocab_size,
        n_layers=preset.n_layers,
        d_model=preset.d_model,
        n_heads=preset.n_heads,
        mlp_dim=preset.mlp_dim,
        n_experts=(1 if _force_dense else preset.moe_experts),
        top_k=(1 if _force_dense else preset.moe_top_k),
        max_seq_len=preset.max_seq_len,
        use_rope=True,
        kv_latent_dim=preset.kv_latent_dim,
        multi_query=preset.multi_query,
        multi_token=max(1, int(multi_token)),
        rope_scale=rope_scale if rope_scale is not None else 1.0,
        rope_base=rope_base if rope_base is not None else 10000.0,
        moe_group_sizes=([] if _force_dense else getattr(preset, 'moe_group_sizes', [])),
        mem_slots=int(mem_slots),
        skip_init=bool(skip_init),
        # Enable HRM according to env (centralized defaults set this to 1)
        use_hrm=_hrm_enabled,
    )
    # Compile at build time to eliminate first-call compile stalls in generate()
    try:
        from omnicoder.utils.torchutils import ensure_compiled as _ensure_compiled  # type: ignore
        model = _ensure_compiled(model)
    except Exception:
        pass
    try:
        log.info(
            "built OmniTransformer embed=%s lm_head=%s blocks=%d kv_latent=%s force_dense=%s",
            tuple(model.embed.weight.shape),
            tuple(model.lm_head.weight.shape),
            len(getattr(model, 'blocks', [])),
            int(getattr(model.blocks[0].attn, 'kv_latent_dim', 0)) if len(getattr(model, 'blocks', [])) > 0 else -1,
            bool(os.getenv('OMNICODER_FORCE_DENSE_MLP','0')=='1'),
        )
    except Exception:
        pass
    try:
        get_logger("omnicoder.preset").info(
            "build_mobile_model exit embed=%s lm_head=%s blocks=%s",
            str(tuple(model.embed.weight.shape)), str(tuple(model.lm_head.weight.shape)), len(getattr(model, 'blocks', []))
        )
    except Exception:
        pass
    return model


def build_mobile_model_by_name(
    preset_name: str,
    rope_scale: float | None = None,
    rope_base: float | None = None,
    multi_token: int = 2,
    mem_slots: int = 0,
    skip_init: bool = False,
) -> OmniTransformer:
    log = get_logger("omnicoder.preset")
    try:
        log.info("build_by_name enter name=%s rope_scale=%s rope_base=%s multi_token=%s mem_slots=%s", preset_name, str(rope_scale), str(rope_base), str(multi_token), str(mem_slots))
    except Exception:
        pass
    try:
        get_logger("omnicoder.preset").info(
            "build_by_name enter name=%s rope_scale=%s rope_base=%s multi_token=%s mem_slots=%s",
            preset_name, rope_scale, rope_base, multi_token, mem_slots
        )
    except Exception:
        pass
    try:
        preset = get_mobile_preset(preset_name)
        # Optional: tiny fast-path for pytest to keep test runtime <10s across heavy router tests
        _do_tiny = False
        try:
            import os as _os
            _do_tiny = bool(_os.getenv('PYTEST_CURRENT_TEST')) and (_os.getenv('OMNICODER_TEST_TINY', '1') == '1')
        except Exception:
            _do_tiny = False
        if _do_tiny:
            try:
                # Clamp preset attributes; copy to a shallow wrapper to avoid mutating globals
                class _TinyPreset:
                    pass
                tp = _TinyPreset()
                for k in ['vocab_size','n_layers','d_model','n_heads','mlp_dim','moe_experts','moe_top_k','kv_latent_dim','max_seq_len','multi_query','name','moe_group_sizes']:
                    setattr(tp, k, getattr(preset, k))
                tp.n_layers = min(int(tp.n_layers), 2)
                tp.d_model = min(int(tp.d_model), 256)
                tp.n_heads = min(int(tp.n_heads), 4)
                tp.mlp_dim = min(int(tp.mlp_dim), 768)
                tp.moe_experts = min(int(tp.moe_experts), 2)
                tp.moe_top_k = min(int(tp.moe_top_k), 1)
                tp.kv_latent_dim = min(int(tp.kv_latent_dim), 64)
                preset = tp  # type: ignore[assignment]
                get_logger("omnicoder.preset").info("tiny fast-path active for tests: layers=%s d_model=%s", int(tp.n_layers), int(tp.d_model))
            except Exception:
                pass
        try:
            desc = {
                'name': getattr(preset, 'name', ''),
                'n_layers': int(getattr(preset, 'n_layers', -1)),
                'd_model': int(getattr(preset, 'd_model', -1)),
                'n_heads': int(getattr(preset, 'n_heads', -1)),
                'mlp_dim': int(getattr(preset, 'mlp_dim', -1)),
                'moe_experts': int(getattr(preset, 'moe_experts', -1)),
                'moe_top_k': int(getattr(preset, 'moe_top_k', -1)),
                'kv_latent_dim': int(getattr(preset, 'kv_latent_dim', -1)),
                'max_seq_len': int(getattr(preset, 'max_seq_len', -1)),
                'moe_group_sizes': list(getattr(preset, 'moe_group_sizes', [])),
            }
            log.info("resolved %s", desc)
        except Exception:
            pass
        model = build_mobile_model(preset, rope_scale=rope_scale, rope_base=rope_base, multi_token=multi_token, mem_slots=mem_slots, skip_init=bool(skip_init))
        try:
            log.info("build_by_name ok name=%s", getattr(preset, 'name', preset_name))
        except Exception:
            pass
        try:
            get_logger("omnicoder.preset").info(
                "build_by_name exit name=%s blocks=%s",
                getattr(preset, 'name', preset_name), len(getattr(model, 'blocks', []))
            )
        except Exception:
            pass
        return model
    except Exception as e:
        import traceback as _tb
        try:
            log.error("build_by_name error: %s", str(e))
            log.error("trace:\n%s", _tb.format_exc())
        except Exception:
            pass
        try:
            get_logger("omnicoder.preset").error("build_by_name error name=%s err=%s", preset_name, str(e))
            get_logger("omnicoder.preset").error("trace:\n%s", _tb.format_exc())
        except Exception:
            pass
        raise


def main():
    # Verbose env/defaults smoke logging
    log = get_logger("omnicoder.gen")
    try:
        log.info("generate.main start")
    except Exception:
        pass
    # Phase timing
    try:
        import time as _time
        _t_main0 = _time.perf_counter()
    except Exception:
        _time = None  # type: ignore
        _t_main0 = 0.0
    # Apply auto thread envs early if enabled
    try:
        apply_thread_env_if_auto()
        log.debug(
            "threads set OMP=%s MKL=%s TORCH=%s",
            os.environ.get('OMP_NUM_THREADS',''),
            os.environ.get('MKL_NUM_THREADS',''),
            os.environ.get('TORCH_NUM_THREADS',''),
        )
    except Exception:
        pass
    # Optional global seeding for reproducibility
    try:
        _seed = os.getenv('OMNICODER_SEED', '').strip()
        if _seed:
            import random as _rand
            import numpy as _np
            s = int(_seed)
            _rand.seed(s)
            _np.random.seed(s)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)
            try:
                torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', type=str, default=os.getenv('OMNICODER_PROMPT', 'Hello'))
    ap.add_argument('--max_new_tokens', type=int, default=int(os.getenv('OMNICODER_MAX_NEW_TOKENS', '32')))
    ap.add_argument('--device', type=str, default=os.getenv('OMNICODER_DEVICE', 'cpu'))
    ap.add_argument('--temperature', type=float, default=float(os.getenv('OMNICODER_TEMPERATURE', '0.8')))
    ap.add_argument('--top_k', type=int, default=int(os.getenv('OMNICODER_TOP_K', '40')))
    ap.add_argument('--top_p', type=float, default=float(os.getenv('OMNICODER_TOP_P', '0.9')))
    ap.add_argument('--ckpt', type=str, default=os.getenv('OMNICODER_CKPT', ''))
    ap.add_argument('--mobile_preset', type=str, default=os.getenv('OMNICODER_STUDENT_PRESET', 'mobile_4gb'), help='Use mobile_4gb to load a constrained model size')
    ap.add_argument('--retrieve_path', type=str, default=os.getenv('OMNICODER_RETRIEVE_PATH', ''), help='Optional: folder or JSONL for local retrieval-augmented prompt (TF-IDF)')
    ap.add_argument('--retrieve_pq_index', type=str, default=os.getenv('OMNICODER_RETRIEVE_PQ_INDEX', ''), help='Optional: path to a PQ index dir built by PqRetriever')
    ap.add_argument('--retrieve_k', type=int, default=int(os.getenv('OMNICODER_RETRIEVE_K', '3')), help='Top-k passages to prepend if retrieval is enabled')
    ap.add_argument('--retrieve_faiss', action='store_true', default=(os.getenv('OMNICODER_RETRIEVE_FAISS', '0') == '1'), help='Use FAISS-based local ANN retriever (installs faiss-cpu)')
    ap.add_argument('--retrieve_chunk', type=int, default=int(os.getenv('OMNICODER_RETRIEVE_CHUNK', '512')), help='Chunk size for local retriever when indexing .txt files')
    ap.add_argument('--retrieve_stride', type=int, default=int(os.getenv('OMNICODER_RETRIEVE_STRIDE', '448')), help='Stride for chunking')
    ap.add_argument('--retrieve_max_chars', type=int, default=int(os.getenv('OMNICODER_RETRIEVE_MAX_CHARS', '4000')), help='Max chars from retrieved context to include')
    ap.add_argument('--retrieve_partition', type=int, default=int(os.getenv('OMNICODER_RETRIEVE_PARTITION','0')), help='PQ: partition size for bounded-RAM scanning')
    ap.add_argument('--retrieve_budget_bytes', type=int, default=int(os.getenv('OMNICODER_RETRIEVE_BUDGET','0')), help='PQ: approximate RAM budget for scanning (bytes)')
    ap.add_argument('--rope_scale', type=float, default=float(os.getenv('OMNICODER_ROPE_SCALE', '1.0')), help='RoPE scale for long-context interpolation')
    ap.add_argument('--rope_base', type=float, default=float(os.getenv('OMNICODER_ROPE_BASE', '10000.0')), help='RoPE base for long-context interpolation (auto-adjusted if OMNICODER_USE_YARN/OMNICODER_USE_PI)')
    ap.add_argument('--target_ctx', type=int, default=int(os.getenv('OMNICODER_TARGET_CTX', '0')), help='Optional: target context window; auto-compute rope_scale if > 0')
    ap.add_argument('--window_size', type=int, default=int(os.getenv('OMNICODER_WINDOW_SIZE', '0')), help='Sliding-window attention cap for decode for lower KV growth (0=disabled)')
    ap.add_argument('--mem_slots', type=int, default=int(os.getenv('OMNICODER_MEM_SLOTS', '4')), help='Number of recurrent memory slots to compress distant context (enables infinite-style context with windowed decode)')
    ap.add_argument('--system_prompt', type=str, default=os.getenv('OMNICODER_SYSTEM_PROMPT', 'Use the provided context if relevant. Be concise and accurate.'), help='System prompt used when retrieval is enabled')
    ap.add_argument('--prompt_template', type=str, default=os.getenv('OMNICODER_PROMPT_TEMPLATE', '[SYSTEM] {system}\n{context}\n\n[USER] {user}\n[ASSISTANT]'), help='Python format template for composing prompts with keys: system, context, user')
    ap.add_argument('--multi_token_heads', type=int, default=int(os.getenv('OMNICODER_MTP_HEADS', '2')), help='Number of multi-token prediction heads (>1 enables lookahead acceptance)')
    ap.add_argument('--disable_speculative', action='store_true', default=(os.getenv('OMNICODER_DISABLE_SPECULATIVE', '0') == '1'), help='Disable lookahead acceptance (uses only next-token head)')
    ap.add_argument('--verify_threshold', type=float, default=float(os.getenv('OMNICODER_VERIFY_THRESHOLD', '0.05')), help='Minimum probability for accepting a speculative token')
    ap.add_argument('--verifier_steps', type=int, default=int(os.getenv('OMNICODER_VERIFIER_STEPS', '1')), help='How many verifying steps to attempt for speculative tokens')
    ap.add_argument('--speculative_draft_len', type=int, default=int(os.getenv('OMNICODER_SPEC_DRAFT_LEN', '1')), help='How many lookahead tokens to attempt (<= number of MTP heads)')
    ap.add_argument('--tree_width', type=int, default=int(os.getenv('OMNICODER_TREE_WIDTH', '3')))
    ap.add_argument('--tree_depth', type=int, default=int(os.getenv('OMNICODER_TREE_DEPTH', '3')))
    ap.add_argument('--kvq', type=str, default=os.getenv('OMNICODER_KVQ', 'none'), choices=['none','u8','nf4'], help='Quantize KV cache for storage (dequantize per step for compute)')
    ap.add_argument('--kvq_group', type=int, default=int(os.getenv('OMNICODER_KVQ_GROUP', '64')), help='Group size for KV quantization along latent dim')
    # Default compile ON by env default (1). Users can disable by setting OMNICODER_COMPILE=0 or omitting the flag with env set.
    ap.add_argument('--compile', action='store_true', default=(os.getenv('OMNICODER_COMPILE', '1') == '1'), help='Enable torch.compile (inductor) for the model decode loop when available (default: on; set OMNICODER_COMPILE=0 to disable)')
    ap.add_argument('--knn_cache', action='store_true', default=(os.getenv('OMNICODER_KNN_CACHE', '0') == '1'), help='Enable small kNN-LM cache to blend nearest neighbor token probabilities')
    ap.add_argument('--knn_k', type=int, default=int(os.getenv('OMNICODER_KNN_K', '16')), help='k for kNN-LM queries')
    ap.add_argument('--knn_lambda', type=float, default=float(os.getenv('OMNICODER_KNN_LAMBDA', '0.2')), help='Blend weight for kNN-LM distribution [0..1]')
    ap.add_argument('--adaptive_gating', action='store_true', default=(os.getenv('OMNICODER_ADAPTIVE_GATING','0')=='1'), help='Enable adaptive speculative control based on confidence')
    ap.add_argument('--adaptive_top_k_min', type=int, default=int(os.getenv('OMNICODER_ADAPTIVE_TOP_K_MIN','1')))
    ap.add_argument('--adaptive_top_k_max', type=int, default=int(os.getenv('OMNICODER_ADAPTIVE_TOP_K_MAX','4')))
    ap.add_argument('--adaptive_conf_floor', type=float, default=float(os.getenv('OMNICODER_ADAPTIVE_CONF_FLOOR','0.3')))
    ap.add_argument('--adaptive_layer_ramp', action='store_true', default=(os.getenv('OMNICODER_ADAPTIVE_LAYER_RAMP','0')=='1'))
    ap.add_argument('--adaptive_layer_power', type=float, default=float(os.getenv('OMNICODER_ADAPTIVE_LAYER_POWER','1.0')))
    # SCMoE inference controls
    ap.add_argument('--scmoe_alpha', type=float, default=float(os.getenv('OMNICODER_SCMOE_ALPHA','0.0')), help='Self-contrast MoE blending weight [0..1] (0 disables)')
    ap.add_argument('--scmoe_frac', type=float, default=float(os.getenv('OMNICODER_SCMOE_FRAC','0.25')), help='Fraction of tokens to apply contrast routing to [0..1]')
    ap.add_argument('--scmoe_topk', type=int, default=int(os.getenv('OMNICODER_SCMOE_TOPK','2')), help='Alternative expert TopK for contrast path')
    # Optional external draft model for speculative decoding
    ap.add_argument('--draft_ckpt', type=str, default=os.getenv('OMNICODER_DRAFT_CKPT',''), help='Path to a draft student checkpoint (OmniTransformer)')
    ap.add_argument('--draft_preset', type=str, default=os.getenv('OMNICODER_DRAFT_PRESET','mobile_2gb'), help='Preset to build the draft model if --draft_ckpt is provided')
    # Early-exit knobs
    ap.add_argument('--early_exit', action='store_true', default=(os.getenv('OMNICODER_EARLY_EXIT','0')=='1'), help='Enable early-exit heuristic based on token entropy')
    ap.add_argument('--early_exit_entropy', type=float, default=float(os.getenv('OMNICODER_EARLY_EXIT_ENTROPY','1.0')), help='Entropy threshold to skip speculative drafts (lower is more confident)')
    ap.add_argument('--early_exit_mode', type=str, default=os.getenv('OMNICODER_EARLY_EXIT_MODE','entropy'), choices=['entropy','delta'], help='Heuristic mode for early exit')
    # Adaptive activation quantization emulation
    ap.add_argument('--act_quant', action='store_true', default=(os.getenv('OMNICODER_ACT_QUANT','0')=='1'), help='Emulate activation quantization per step based on confidence')
    ap.add_argument('--act_quant_min_bits', type=int, default=int(os.getenv('OMNICODER_ACT_MIN_BITS','8')))
    ap.add_argument('--act_quant_max_bits', type=int, default=int(os.getenv('OMNICODER_ACT_MAX_BITS','2')))
    ap.add_argument('--act_quant_conf_floor', type=float, default=float(os.getenv('OMNICODER_ACT_CONF_FLOOR','0.3')))
    # Cross-modal verifier gate (mini-CLIP) for multimodal outputs (e.g., image/video)
    ap.add_argument('--cm_verifier', action='store_true', default=(os.getenv('OMNICODER_CM_VERIFIER','0')=='1'), help='Enable cross-modal verifier-based rejection sampling for multimodal outputs')
    ap.add_argument('--cm_threshold', type=float, default=float(os.getenv('OMNICODER_CM_THRESHOLD','0.6')), help='Threshold in [0,1] for cross-modal verifier acceptance')
    # Tool-use
    ap.add_argument('--tool_use', action='store_true', default=(os.getenv('OMNICODER_TOOL_USE','0')=='1'), help='Parse and execute inline <tool:... {..}> tags in generated text')
    # Omega-Solver flags
    ap.add_argument('--reasoner', type=str, default=os.getenv('OMNICODER_REASONER', os.getenv('OMNI_REASONER','omega')), help='Set to "omega" to enable Omega-Solver scaffolds')
    ap.add_argument('--emit_cert', action='store_true', default=(os.getenv('OMNI_CERT_EMIT','0')=='1' or os.getenv('OMNICODER_CERT_EMIT','0')=='1'), help='Emit proof-carrying answer JSON alongside text')
    ap.add_argument('--proof_margin_thresh', type=str, default=os.getenv('OMNI_PROOF_MARGIN_THRESH','auto'), help='Acceptance threshold for proof-margin (float or "auto")')
    # Preference RAG bias
    ap.add_argument('--pref_rag_index', type=str, default=os.getenv('OMNICODER_PREF_RAG_INDEX', ''), help='Optional path to preference index (.json or .jsonl)')
    ap.add_argument('--pref_rag_topk', type=int, default=int(os.getenv('OMNICODER_PREF_RAG_TOPK', '4')))
    ap.add_argument('--pref_rag_alpha', type=float, default=float(os.getenv('OMNICODER_PREF_RAG_ALPHA', '0.0')))
    # Unified multi-index (text/image embeddings) retrieval
    ap.add_argument('--multi_index_root', type=str, default=os.getenv('OMNICODER_MULTI_INDEX_ROOT', ''), help='Path to multi-index root (embeddings.npy + ids.txt)')
    ap.add_argument('--mi_topk', type=int, default=int(os.getenv('OMNICODER_MI_TOPK', '4')))
    ap.add_argument('--mi_alpha', type=float, default=float(os.getenv('OMNICODER_MI_ALPHA', '0.0')))
    # Optional runtime LoRA adapter load (applies tiny deltas for personalization)
    ap.add_argument('--pref_lora', type=str, default=os.getenv('OMNICODER_PREF_LORA', ''), help='Path to preference LoRA weights (state_dict) to inject at runtime')
    ap.add_argument('--pref_lora_r', type=int, default=int(os.getenv('OMNICODER_PREF_LORA_R', '8')))
    ap.add_argument('--pref_lora_alpha', type=int, default=int(os.getenv('OMNICODER_PREF_LORA_ALPHA', '16')))
    ap.add_argument('--pref_lora_dropout', type=float, default=float(os.getenv('OMNICODER_PREF_LORA_DROPOUT', '0.05')))
    args = ap.parse_args()
    try:
        log.info(
            "args device=%s preset=%s max_new_tokens=%s compile=%s",
            args.device,
            args.mobile_preset,
            int(args.max_new_tokens),
            bool(args.compile),
        )
    except Exception:
        pass

    try:
        tokenizer = get_text_tokenizer(prefer_hf=True)
        log.info("tokenizer ok")
        if _time is not None:
            log.debug("phase.tokenizer.dt=%.6f", float(_time.perf_counter() - _t_main0))
    except Exception as e:
        get_logger("omnicoder.gen").error("tokenizer error: %s", str(e))
        raise

    # Auto-load default acceptance thresholds per preset if verify_threshold not provided
    if float(args.verify_threshold) == 0.0:
        try:
            from omnicoder.utils.thresholds import get_accept_threshold  # type: ignore
            args.verify_threshold = float(get_accept_threshold(str(args.mobile_preset), float(args.verify_threshold)))
            log.debug("thresholds verify_threshold=%s", args.verify_threshold)
        except Exception:
            pass

    # Retrieval-augmented prompting (local, no internet)
    full_prompt = args.prompt
    # Prefer PQ index if provided, else fall back to TF-IDF/FAISS local retriever
    if args.retrieve_pq_index:
        try:
            if PqRetriever is not None:
                pq = PqRetriever(args.retrieve_pq_index)  # type: ignore[operator]
                hits = pq.search(args.prompt, k=max(1, args.retrieve_k), partition_size=int(args.retrieve_partition), budget_bytes=int(args.retrieve_budget_bytes))
                ctx = "\n\n".join([f"[CTX {i+1}] {t}" for i, (_idx, _s, t) in enumerate(hits)])
                if args.retrieve_max_chars > 0 and len(ctx) > args.retrieve_max_chars:
                    ctx = ctx[: args.retrieve_max_chars]
                tpl = args.prompt_template or "[SYSTEM] {system}\n{context}\n\n[USER] {user}\n[ASSISTANT]"
                full_prompt = tpl.format(system=args.system_prompt, context=ctx, user=args.prompt)
        except Exception:
            pass
    else:
        full_prompt = _maybe_retrieve(
            args.prompt,
            args.retrieve_path,
            args.retrieve_k,
            args.retrieve_faiss,
            args.retrieve_chunk,
            args.retrieve_stride,
            args.retrieve_max_chars,
            args.system_prompt,
            args.prompt_template,
        )
    try:
        if _time is not None:
            log.debug("phase.retrieve.dt=%.6f", float(_time.perf_counter() - _t_main0))
    except Exception:
        pass

    # Optional: load preference RAG texts and compute top-k by simple overlap
    pref_texts: list[str] | None = None
    if args.pref_rag_index and float(args.pref_rag_alpha) > 0.0:
        try:
            import json as _json
            from pathlib import Path as _Path
            p = _Path(args.pref_rag_index)
            if p.exists():
                if p.suffix.lower() == '.jsonl':
                    pref_texts = []
                    for line in p.read_text(encoding='utf-8').splitlines():
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            obj = _json.loads(s)
                            t = str(obj.get('text', ''))
                            if t:
                                pref_texts.append(t)
                        except Exception:
                            pref_texts.append(s)
                else:
                    obj = _json.loads(p.read_text(encoding='utf-8'))
                    if isinstance(obj, dict) and isinstance(obj.get('texts'), list):
                        pref_texts = [str(x) for x in obj['texts'] if str(x).strip()][:10000]
        except Exception:
            pref_texts = None
    # Optionally run a light PPI/NSS analysis step to populate a symbolic scaffold
    try:
        if os.getenv('OMNICODER_PPI_DEFAULT_ON', '0') == '1' and analyze_inputs_for_nss is not None:
            nss_ctx, _nss = analyze_inputs_for_nss(full_prompt, modes=os.getenv('OMNICODER_PPI_MODES','image,audio,video'))
            if nss_ctx:
                full_prompt = f"{nss_ctx}\n\n{full_prompt}"
    except Exception:
        pass
    try:
        _log = get_logger("omnicoder.gen")
        enc = tokenizer.encode(full_prompt)
        _log.debug("encode ok n_tokens=%s", len(enc))
        input_ids = torch.tensor([enc], dtype=torch.long)
        _log.debug("prompt len=%s", int(input_ids.size(1)))
    except Exception as e:
        import traceback as _tb
        try:
            get_logger("omnicoder.gen").error("encode error: %s", str(e))
            get_logger("omnicoder.gen").error("trace:\n%s", _tb.format_exc())
        except Exception:
            pass
        raise
    # Ω-Intent and Ω-Causal (lightweight) before model build, for optional certificate context
    goal_belief = None
    causal_margin = 0.0
    if args.reasoner.strip().lower() == 'omega':
        try:
            if _omega_infer_goals is not None:
                goal = _omega_infer_goals(full_prompt, context=None, k=3, methods=os.getenv('OMNI_GOAL_INFER','rsa,pr,cirl'))
                # convert dataclasses to plain dict
                goal_belief = {
                    'hypotheses': [vars(h) for h in getattr(goal, 'hypotheses', [])],
                    'constraints': list(getattr(goal, 'constraints', [])),
                    'desiderata': list(getattr(goal, 'desiderata', [])),
                    'meta': dict(getattr(goal, 'meta', {})),
                }
        except Exception:
            goal_belief = None
        try:
            scm_on = os.getenv('OMNI_SCM', os.getenv('OMNICODER_SCM','off')).lower().strip()
            if _omega_build_scm is not None and _omega_abd_score is not None and scm_on in ('on','1','true'):
                scm = _omega_build_scm(full_prompt)
                causal_margin = float(_omega_abd_score(scm, observations={'A': scm.forward().get('A', 0.0)}))
        except Exception:
            causal_margin = 0.0
    # Ω-Reasoner prompt transformation (optional)
    try:
        if (os.getenv('OMNICODER_REASONER', 'none').lower().strip() == 'omega') and (OmegaController is not None):
            import time as _time
            _tp0 = _time.perf_counter()
            _omega = OmegaController(OmegaConfig())  # type: ignore[operator]
            full_prompt, _meta = _omega.plan_and_transform_prompt(full_prompt)
            _tp1 = _time.perf_counter()
            try:
                get_logger('omnicoder.gen').info('omega.plan.dt=%.6f', float(_tp1 - _tp0))
            except Exception:
                pass
            input_ids = torch.tensor([tokenizer.encode(full_prompt)], dtype=torch.long)
    except Exception:
        pass

    # Choose model by preset for mobile-friendly defaults
    if args.mobile_preset in ('mobile_4gb','mobile_2gb','mobile_4gb_moe16','mobile_2gb_moe16'):
        rs = args.rope_scale
        if args.target_ctx and args.target_ctx > 0:
            preset = get_mobile_preset(args.mobile_preset)
            rs = get_rope_scale_for_target_ctx(preset.max_seq_len, args.target_ctx)
            # Adjust rope_base if YaRN/PI env is enabled
            args.rope_base = get_rope_interp_base(args.rope_base, rs)
        get_logger("omnicoder.gen").debug("build_mobile_model_by_name enter preset=%s rs=%s base=%s", args.mobile_preset, rs, args.rope_base)
        model = build_mobile_model_by_name(
            args.mobile_preset,
            rope_scale=rs,
            rope_base=args.rope_base,
            multi_token=args.multi_token_heads,
            mem_slots=args.mem_slots,
        )
        try:
            if _time is not None:
                log.debug("phase.model_build.dt=%.6f", float(_time.perf_counter() - _t_main0))
        except Exception:
            pass
        # Apply sliding-window cap if requested
        if hasattr(model, 'blocks'):
            # Default window from preset if not provided
            if int(args.window_size) <= 0:
                try:
                    from omnicoder.config import get_mobile_preset
                    _p = get_mobile_preset(args.mobile_preset)
                    args.window_size = int(getattr(_p, 'default_window_size', 0))
                except Exception:
                    pass
        if hasattr(model, 'blocks') and int(args.window_size) > 0:
            for blk in model.blocks:
                try:
                    blk.attn.window_size = int(args.window_size)
                    # When using a sliding window or long target ctx, enable landmarks by default
                    want_landmarks = True if (args.window_size and args.window_size>0) or (args.target_ctx and args.target_ctx>0) else False
                    if hasattr(blk.attn, 'use_landmarks') and want_landmarks:
                        blk.attn.use_landmarks = True
                        if hasattr(blk.attn, 'num_landmarks'):
                            nl = int(getattr(blk.attn, 'num_landmarks', 0))
                            if nl <= 0:
                                setattr(blk.attn, 'num_landmarks', 8)
                except Exception:
                    pass
    else:
        get_logger("omnicoder.gen").debug("OmniTransformer() enter (default preset)")
        model = OmniTransformer()
    get_logger("omnicoder.gen").debug("maybe_load_checkpoint enter path=%s", str(args.ckpt))
    maybe_load_checkpoint(model, args.ckpt)
    get_logger("omnicoder.gen").debug("model.to(%s) enter", str(args.device))
    model.to(args.device)
    get_logger("omnicoder.gen").debug("model.to(%s) exit ok", str(args.device))
    try:
        # Dump a compact model summary to aid triage without heavy introspection
        nb = len(getattr(model, 'blocks', []))
        dm = int(getattr(model, 'd_model', 0))
        nh = int(getattr(model.blocks, [type('x',(),{'attn':type('y',(),{'n_heads':0})()})])[0].attn.n_heads) if nb>0 else 0
        get_logger("omnicoder.gen").info("model summary blocks=%s d_model=%s n_heads=%s", nb, dm, nh)
    except Exception:
        pass
    # Optional runtime quantization: CPU int8 dynamic for Linear; CUDA autocast fp16
    try:
        if str(args.device) == 'cpu' and os.getenv('OMNICODER_CPU_DYN_QUANT','1') == '1':
            import torch.ao.quantization as _aq  # type: ignore[attr-defined]
            import torch.nn as _nn
            qcfg = {_nn.Linear}
            model = _aq.quantize_dynamic(model, qcfg, dtype=torch.qint8)  # type: ignore[assignment]
            get_logger("omnicoder.gen").info("quant.dynamic int8 applied to Linear on CPU")
        elif str(args.device).startswith('cuda') and torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    try:
        model.eval()
        get_logger("omnicoder.gen").debug("model.eval() set before compile")
    except Exception:
        pass

    # Optional runtime LoRA injection (before compile)
    if args.pref_lora:
        try:
            _ = _inject_lora_linear(model, r=int(args.pref_lora_r), alpha=int(args.pref_lora_alpha), dropout=float(args.pref_lora_dropout))
            state = torch.load(str(args.pref_lora), map_location='cpu')
            model.load_state_dict(state, strict=False)
        except Exception as e:
            logging.debug("[generate] pref LoRA injection skipped: %s", e)
    # Optional compilation: prefer inductor for inference; fall back gracefully
    # Guard: disable compile on non-CUDA devices to avoid slow CPU graph compile
    try:
        if args.compile and not str(args.device).startswith('cuda'):
            log.info("compile disabled on non-CUDA device=%s", str(args.device))
    except Exception:
        pass
    if args.compile:
        try:
            # Explicit compile enter log
            try:
                get_logger("omnicoder.gen").info("compile enter")
            except Exception:
                pass
            # Prefer inductor backend; avoid fullgraph to tolerate minor graph breaks in decode loop
            original_model = model
            _backend = 'inductor'
            try:
                log.info("compile starting backend=%s", _backend)
            except Exception:
                pass
            try:
                _backend = 'inductor' if (str(args.device).startswith('cuda') and torch.cuda.is_available()) else 'aot_eager'
            except Exception:
                _backend = 'aot_eager'
            try:
                get_logger("omnicoder.gen").debug("torch.compile call enter backend=%s", _backend)
            except Exception:
                pass
            # Compile once up-front; compiled module is reused, no per-step recompiles
            compiled_model = torch.compile(model, mode='reduce-overhead', backend=_backend, fullgraph=False)  # type: ignore[assignment]
            try:
                get_logger("omnicoder.gen").debug("torch.compile returned model backend=%s", _backend)
            except Exception:
                pass
            # Force a tiny warmup to catch missing toolchains (e.g., cl on Windows)
            try:
                get_logger("omnicoder.gen").debug("compile warmup enter backend=%s", _backend)
                _warm_ids = torch.randint(0, getattr(model, 'vocab_size', 32000), (1, 1), dtype=torch.long, device=args.device)
                # Use cache path for a minimal decode-step warmup (avoids full-seq extras)
                compiled_model.eval()
                try:
                    log.debug("compile warmup forward enter")
                except Exception:
                    pass
                # Warm up with the same signature used in decode to avoid graph breaks later
                _ = compiled_model(_warm_ids, past_kv=None, use_cache=True, return_hidden=True, prefix_hidden=None)
                try:
                    get_logger("omnicoder.gen").debug("compile warmup forward exit ok")
                except Exception:
                    pass
                # Swap in compiled model globally so decode reuses the compiled graph
                model = compiled_model  # type: ignore[assignment]
                try:
                    log.info("compile ok backend=%s", _backend)
                except Exception:
                    pass
                try:
                    get_logger("omnicoder.gen").debug("compile warmup exit ok backend=%s", _backend)
                except Exception:
                    pass
            except Exception:
                get_logger("omnicoder.gen").debug("compile warmup failed; falling back to eager")
                model = original_model  # type: ignore[assignment]
            try:
                get_logger("omnicoder.gen").info("compile exit")
                if _time is not None:
                    log.debug("phase.compile.dt=%.6f", float(_time.perf_counter() - _t_main0))
            except Exception:
                pass
        except Exception:
            # Graceful fallback if compile is unsupported on this platform/build
            pass
    try:
        log.debug("after compile")
    except Exception:
        pass

    # Auto-prefer KV quantization if calibration sidecar exists and user did not set kvq
    import os as _os
    cand = _os.path.join('weights', 'kvq_calibration.json')
    if args.kvq == 'none' and _os.path.exists(cand):
        args.kvq = 'nf4'
    # If calibration JSON exists, honor its group size to ensure consistency across runners
    if _os.path.exists(cand):
        try:
            import json as _json
            calib = _json.loads(open(cand, 'r', encoding='utf-8').read())
            if isinstance(calib, dict) and 'group' in calib:
                args.kvq_group = int(calib.get('group', args.kvq_group))
        except Exception:
            pass

    # Map OMNI_MCTS_BUDGET to internal RG budget env if provided
    try:
        _mcts_budget = os.getenv('OMNI_MCTS_BUDGET', '').strip()
        if _mcts_budget:
            os.environ['OMNICODER_RG_BUDGET_TOKENS'] = _mcts_budget
    except Exception:
        pass

    # Optional: initialize kNN cache if enabled
    knn_cache = KNNCache(dim=getattr(model.ln_f, 'normalized_shape', [model.lm_head.in_features])[0]) if args.knn_cache else None
    try:
        log.debug("after knn init")
    except Exception:
        pass
    knn_prune_every = 0
    knn_max_items = 0
    if knn_cache is not None:
        try:
            knn_max_items = int(os.getenv('OMNICODER_KNN_MAX_ITEMS', '4096'))
            knn_prune_every = int(os.getenv('OMNICODER_KNN_PRUNE_EVERY', '32'))
            if hasattr(knn_cache, 'prune'):
                knn_cache.prune(max_items=knn_max_items)
            # Optional persistent cache load
            kc_path = os.getenv('OMNICODER_KNN_CACHE_PATH', '').strip()
            if kc_path:
                try:
                    ok = knn_cache.load(kc_path)
                    if ok:
                        logging.debug('[generate] loaded kNN cache from %s', kc_path)
                except Exception:
                    pass
        except Exception:
            pass

    # Bind adaptive knobs
    adapt_kwargs = {}
    if bool(args.adaptive_gating):
        adapt_kwargs = dict(
            adaptive_top_k_min=int(args.adaptive_top_k_min),
            adaptive_top_k_max=int(args.adaptive_top_k_max),
            adaptive_conf_floor=float(args.adaptive_conf_floor),
            adaptive_layer_ramp=bool(args.adaptive_layer_ramp),
            adaptive_layer_power=float(args.adaptive_layer_power),
        )
    try:
        log.debug("after adapt_kwargs")
    except Exception:
        pass

    # Optional: build a draft model if a checkpoint is provided
    draft_model = None
    try:
        log.debug("before draft setup")
    except Exception:
        pass
    # Auto-enable a draft model when a default checkpoint exists and no explicit path was provided
    # Guard: when max_new_tokens <= 1 there is no benefit to speculative drafting; skip auto-draft
    if not args.draft_ckpt:
        try:
            if int(args.max_new_tokens) > 1:
                from pathlib import Path as _Path
                cand = _Path('weights') / 'omnicoder_draft_kd.pt'
                if cand.exists():
                    args.draft_ckpt = str(cand)
                    log.info("found draft ckpt path=%s", args.draft_ckpt)
            else:
                try:
                    get_logger("omnicoder.gen").debug(
                        "skip auto draft: max_new_tokens=%s <= 1", int(args.max_new_tokens)
                    )
                except Exception:
                    pass
        except Exception:
            pass
    if args.draft_ckpt:
        try:
            # Prefer the strongest available draft preset if not explicitly set
            if not os.getenv('OMNICODER_DRAFT_PRESET'):
                # Try larger presets first; add verbose reason if any fails
                for cand_preset in ('draft_3b','draft_2b','draft_1b','mobile_2gb'):
                    try:
                        log.debug("trying draft preset %s", cand_preset)
                        import time as _time
                        _t0 = _time.time()
                        log.debug("draft entering build_mobile_model_by_name preset=%s", cand_preset)
                        # For draft models that will load a checkpoint immediately, skip random init
                        draft_model = build_mobile_model_by_name(cand_preset, mem_slots=0, skip_init=True)
                        _t1 = _time.time()
                        log.debug("draft built model preset=%s secs=%.2f", cand_preset, (_t1 - _t0))
                        args.draft_preset = cand_preset
                        log.info("draft preset chosen %s", cand_preset)
                        break
                    except Exception as _e:
                        import traceback as _tb
                        get_logger("omnicoder.gen").debug("draft preset %s failed: %s", cand_preset, str(_e))
                        get_logger("omnicoder.gen").debug("trace:\n%s", _tb.format_exc())
                        continue
            # When building a draft immediately from a checkpoint, skip random init for faster, consistent load
            import time as _time
            _t0 = _time.time()
            log.debug("draft entering build_mobile_model_by_name preset=%s", args.draft_preset)
            draft_model = build_mobile_model_by_name(args.draft_preset, mem_slots=0, skip_init=True)
            _t1 = _time.time()
            log.debug("draft built model preset=%s secs=%.2f", args.draft_preset, (_t1 - _t0))
            try:
                get_logger("omnicoder.ckpt").info("load draft ckpt path=%s", args.draft_ckpt)
            except Exception:
                pass
            try:
                log.debug("draft maybe_load_checkpoint enter path=%s", args.draft_ckpt)
            except Exception:
                pass
            _t2 = _time.time()
            maybe_load_checkpoint(draft_model, args.draft_ckpt)
            _t3 = _time.time()
            try:
                log.debug("draft maybe_load_checkpoint exit secs=%.2f", (_t3 - _t2))
            except Exception:
                pass
            try:
                get_logger("omnicoder.model").info("draft model -> device %s", args.device)
            except Exception:
                pass
            try:
                log.debug("draft to_device enter device=%s", args.device)
            except Exception:
                pass
            draft_model.to(args.device)
            try:
                get_logger("omnicoder.model").debug("draft model device=%s", str(next(draft_model.parameters()).device))
            except Exception:
                pass
            try:
                log.debug("draft to_device exit device=%s", str(next(draft_model.parameters()).device))
            except Exception:
                pass
            draft_model.eval()
            log.info("draft model ready")
        except Exception as e:
            import traceback as _tb
            get_logger("omnicoder.gen").error("draft setup error: %s", str(e))
            get_logger("omnicoder.gen").error("trace:\n%s", _tb.format_exc())
            draft_model = None
    # Auto-wire ONNX draft if requested and no PyTorch draft is active
    if draft_model is None:
        try:
            if os.getenv('OMNICODER_USE_ONNX','0')=='1':
                onnx_path = os.getenv('OMNICODER_API_ONNX_DECODE','weights/text/draft_decode_step.onnx')
                if onnx_path and os.path.exists(onnx_path) and _ort is not None:
                    provider = os.getenv('OMNICODER_ORT_PROVIDER','auto')
                    draft_model = _OnnxDraftWrapper(onnx_path, provider=provider)  # type: ignore[assignment]
                    log.info("using ONNX draft for speculative decode: %s", onnx_path)
        except Exception as _e:
            try:
                log.debug("onnx draft wiring skipped: %s", _e)
            except Exception:
                pass
    try:
        log.debug("after draft setup")
    except Exception:
        pass

    # Non-autoregressive mask-predict mode (simple): when OMNICODER_MASK_PRED tokens are requested, we first generate a draft with special [MASK] tokens and iteratively refine.
    mask_predict = (os.getenv('OMNICODER_MASK_PRED','0')=='1')
    mask_iters = int(os.getenv('OMNICODER_MASK_ITERS','3'))
    mask_token_id = int(os.getenv('OMNICODER_MASK_TOKEN_ID','0'))
    log.debug("mask_predict=%s", bool(mask_predict))
    if mask_predict and hasattr(model, 'lm_head'):
        try:
            # Start with half masked draft
            base_ids = torch.ops.aten.mul.Scalar(input_ids, 1.0)
            L = int(args.max_new_tokens)
            draft = torch.full((1, L), mask_token_id, dtype=torch.long)
            out_ids = torch.cat([base_ids, draft], dim=1)
            for _ in range(max(1, mask_iters)):
                outputs = model(out_ids[:, :-L], past_kv=None, use_cache=False)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                # Fill MASKs by argmax
                next_ids = torch.argmax(logits[:, -L:, :], dim=-1)
                out_ids[:, -L:] = next_ids
            text = tokenizer.decode(out_ids[0].tolist())
            log.info("mask_predict text=%s", text)
            return
        except Exception:
            mask_predict = False
    try:
        log.debug("after mask_predict check")
    except Exception:
        pass
    if args.disable_speculative:
        # Temporarily force single-head behavior by ignoring lookahead logits in generate loop
        original_multi = getattr(model, 'multi_token', 1)
        model.multi_token = 1  # type: ignore
        # Build retrieved_texts top-k for biasing when requested
        retrieved_texts = None
        if pref_texts and float(args.pref_rag_alpha) > 0.0:
            try:
                q = args.prompt.strip().lower()
                # Rank by shared word count
                def _score(t: str) -> int:
                    try:
                        qs = set(q.split())
                        ts = set(str(t).lower().split())
                        return len(qs & ts)
                    except Exception:
                        return 0
                ranked = sorted(pref_texts, key=_score, reverse=True)
                retrieved_texts = ranked[: max(1, int(args.pref_rag_topk))]
            except Exception:
                retrieved_texts = None
        # Multi-index retrieval (if available): hashed TF-IDF style vector vs embeddings.npy
        if args.multi_index_root and float(args.mi_alpha) > 0.0:
            try:
                import numpy as _np
                from pathlib import Path as _P
                root = _P(args.multi_index_root)
                emb = _np.load(str(root / 'embeddings.npy'))
                ids = (root / 'ids.txt').read_text(encoding='utf-8').splitlines()
                dim = int(emb.shape[1]) if emb.ndim == 2 else 0
                if dim > 0 and len(ids) == emb.shape[0]:
                    # Hashing: map words to buckets 0..dim-1
                    vec = _np.zeros((dim,), dtype='float32')
                    for w in args.prompt.lower().split():
                        h = (hash(w) % dim + dim) % dim
                        vec[h] += 1.0
                    n = _np.linalg.norm(vec) + 1e-6
                    vn = vec / n
                    Em = emb
                    # Cosine similarity assuming embeddings are L2-normalized in build; else normalize
                    if _np.abs(_np.linalg.norm(Em[0]) - 1.0) > 1e-2:
                        Em = Em / (_np.linalg.norm(Em, axis=1, keepdims=True) + 1e-6)
                    sims = (Em @ vn).astype('float32')
                    topk = int(max(1, args.mi_topk))
                    idx = _np.argsort(-sims)[:topk]
                    mi_texts: list[str] = []
                    for i in idx.tolist():
                        try:
                            idstr = ids[i]
                            if idstr.startswith('text:'):
                                p = idstr.split(':',1)[1]
                                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                                    mi_texts.append(f.read()[:2000])
                        except Exception:
                            continue
                    if mi_texts:
                        retrieved_texts = (retrieved_texts or []) + mi_texts
            except Exception as _e:
                logging.debug('[generate] multi-index retrieval skipped: %s', _e)
        try:
            _log.info("pre_generate disable_speculative path")
        except Exception:
            pass
        try:
            out_ids = generate(
            model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            verify_threshold=args.verify_threshold,
            verifier_steps=args.verifier_steps,
            speculative_draft_len=args.speculative_draft_len,
            tree_width=args.tree_width,
            tree_depth=args.tree_depth,
            draft_model=draft_model,
            kvq=args.kvq,
            kvq_group=args.kvq_group,
            knn_cache=knn_cache,
            knn_k=args.knn_k,
            knn_lambda=args.knn_lambda,
            window_size=args.window_size,
            scmoe_alpha=float(args.scmoe_alpha),
            scmoe_frac=float(args.scmoe_frac),
            **adapt_kwargs,
            early_exit=bool(args.early_exit),
            early_exit_entropy=float(args.early_exit_entropy),
            early_exit_mode=str(args.early_exit_mode),
            retrieved_texts=retrieved_texts,
            retrieval_bias_alpha=(float(args.pref_rag_alpha) + float(args.mi_alpha)) if retrieved_texts else 0.0,
            encode_fn=tokenizer.encode,
            act_quant_enable=bool(args.act_quant),
            act_quant_min_bits=int(args.act_quant_min_bits),
            act_quant_max_bits=int(args.act_quant_max_bits),
            act_quant_conf_floor=float(args.act_quant_conf_floor),
            )
            try:
                _log.debug(
                    "generate returned shape=%s first_ids=%s",
                    str(tuple(out_ids.shape)),
                    str(torch.ops.aten.reshape.default(out_ids, (-1,))[: min(8, out_ids.numel())].tolist())
                )
            except Exception:
                pass
        except Exception as e:
            import traceback as _tb
            get_logger("omnicoder.gen").error("generate() error (disable_speculative): %s", str(e))
            get_logger("omnicoder.gen").error("trace:\n%s", _tb.format_exc())
            raise
        model.multi_token = original_multi  # type: ignore
    else:
        try:
            _log.info("pre_generate normal path")
        except Exception:
            pass
        try:
            out_ids = generate(
            model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            verify_threshold=args.verify_threshold,
            verifier_steps=args.verifier_steps,
            speculative_draft_len=args.speculative_draft_len,
            tree_width=args.tree_width,
            tree_depth=args.tree_depth,
            draft_model=draft_model,
            kvq=args.kvq,
            kvq_group=args.kvq_group,
            knn_cache=knn_cache,
            knn_k=args.knn_k,
            knn_lambda=args.knn_lambda,
            window_size=args.window_size,
            scmoe_alpha=float(args.scmoe_alpha),
            scmoe_frac=float(args.scmoe_frac),
            **adapt_kwargs,
            early_exit=bool(args.early_exit),
            early_exit_entropy=float(args.early_exit_entropy),
            early_exit_mode=str(args.early_exit_mode),
            act_quant_enable=bool(args.act_quant),
            act_quant_min_bits=int(args.act_quant_min_bits),
            act_quant_max_bits=int(args.act_quant_max_bits),
            act_quant_conf_floor=float(args.act_quant_conf_floor),
            )
            try:
                _log.debug(
                    "generate returned shape=%s first_ids=%s",
                    str(tuple(out_ids.shape)),
                    str(torch.ops.aten.reshape.default(out_ids, (-1,))[: min(8, out_ids.numel())].tolist())
                )
            except Exception:
                pass
        except Exception as e:
            import traceback as _tb
            get_logger("omnicoder.gen").error("generate() error: %s", str(e))
            get_logger("omnicoder.gen").error("trace:\n%s", _tb.format_exc())
            raise
    try:
        text = tokenizer.decode(out_ids[0].tolist())
        get_logger("omnicoder.gen").info("generated len=%s", len(out_ids[0]))
    except Exception as e:
        get_logger("omnicoder.gen").error("decode error: %s", str(e))
        raise
    # Ω-Verifier + PCA: compute simple margin and optionally emit a certificate
    if args.reasoner.strip().lower() == 'omega' and bool(args.emit_cert):
        try:
            # Minimal signals: text overlap with prompt; include causal margin
            signals = {
                'text': {'hyp': text, 'context': full_prompt},
            }
            weights = {'text': 1.0}
            if causal_margin > 0:
                # carry as extra key in verifications map
                signals['causal'] = {'z': causal_margin}
            m = 0.0
            if _omega_compute_margin is not None:
                m = float(_omega_compute_margin({'text': signals['text']}, weights))
            # Determine threshold
            thr = 0.0
            try:
                thr_arg = str(args.proof_margin_thresh).strip().lower()
                if thr_arg == 'auto':
                    thr = 0.5
                else:
                    thr = float(thr_arg)
            except Exception:
                thr = 0.5
            # Pack cert
            if _omega_pack_cert is not None and _omega_cert_json is not None:
                verifs = {'text': m}
                if causal_margin > 0:
                    verifs['causal'] = float(causal_margin)
                obj = _omega_pack_cert(text, goal_belief, verifs, assumptions=[], counterfactuals=[], margin=m)
                get_logger("omnicoder.gen").info("omega_cert=%s", _omega_cert_json(obj))
        except Exception:
            pass
    # Optional: persist kNN cache at end
    try:
        if os.getenv('OMNICODER_KNN_CACHE', '0') == '1':
            kc_path = os.getenv('OMNICODER_KNN_CACHE_PATH', '').strip()
            if kc_path and knn_cache is not None and hasattr(knn_cache, 'save'):
                knn_cache.save(kc_path)
    except Exception:
        pass
    if bool(args.tool_use):
        text = _postprocess_tool_use(text)
    get_logger("omnicoder.gen").info("text=%s", text)
    try:
        get_logger("omnicoder.gen").info("generate.main end")
        if _time is not None:
            _t_total = float(_time.perf_counter() - _t_main0)
            log.info("phase.total.dt=%.6f", _t_total)
            if _t_total > 10.0:
                log.warning("slow_run dt=%.3fs > 10s: enable OMNICODER_TIMING=1 and OMNICODER_BENCH_DIAG=1 to profile")
    except Exception:
        pass
    # Emit Ω2 certificate (JSONL) when configured
    try:
        if _Omega2Cert is not None and _emit_cert is not None:
            cert = _Omega2Cert(
                prompt=str(full_prompt),
                output=str(text),
                agot=None,
                latent=None,
                planner=None,
                graphrag=None,
                verifier_margin=None,
                acceptance_prob=None,
            )
            try:
                _symp = locals().get('symp', None)
                if _symp is not None:
                    plan_obj = None
                    try:
                        if hasattr(_symp, 'astar'):
                            plan_obj = _symp.astar()
                    except Exception:
                        plan_obj = None
                    if plan_obj is None:
                        plan_obj = _symp.plan(str(full_prompt))
                    cert.planner = {'actions': [{'name': getattr(a, 'name', ''), 'args': list(getattr(a, 'args', ())) } for a in getattr(plan_obj, 'actions', []) ]}  # type: ignore[assignment]
            except Exception:
                pass
            try:
                _grag = locals().get('grag', None)
                if _grag is not None and getattr(_grag, 'enabled', False):
                    triples = _grag.retrieve(str(full_prompt), k=8)
                    cert.graphrag = {'triples': [(t.head, t.relation, t.tail) for t in triples]}  # type: ignore[assignment]
            except Exception:
                pass
            # Populate AGoT / Latent BFS telemetry
            try:
                _agot = locals().get('agot', None)
                if _agot is not None and getattr(_agot, 'enabled', False):
                    cert.agot = {
                        'width': int(getattr(_agot, 'width', 0)),
                        'depth': int(getattr(_agot, 'depth', 0)),
                        'token_budget': int(getattr(_agot, 'token_budget', 0)),
                        'selected_id': None,
                    }  # type: ignore[assignment]
            except Exception:
                pass
            try:
                _ltf = locals().get('ltf', None)
                if _ltf is not None and getattr(_ltf, 'enabled', False):
                    cert.latent = {
                        'beam': int(getattr(_ltf, 'beam', 0)),
                        'depth': int(getattr(_ltf, 'depth', 0)),
                        'selected_id': None,
                    }  # type: ignore[assignment]
            except Exception:
                pass
            try:
                _vlog = locals().get('verifier_logits', None)
                if _vlog is not None:
                    vp = torch.ops.aten._softmax.default(_vlog[:, -1, :], -1, False)
                    # use final token id if available
                    last_id = None
                    try:
                        _gseq = locals().get('generated_seq', [])
                        last_id = int(_gseq[-1].item()) if _gseq else None
                    except Exception:
                        last_id = None
                    if last_id is not None:
                        cert.verifier_margin = float(vp.gather(-1, torch.tensor([[last_id]], device=vp.device)).item())
            except Exception:
                pass
            # Acceptance probability is tracked inside generate(); not available here.
            # Attach acceptance probability and bounded per-step trace when available
            try:
                _lap = locals().get('last_acceptance_prob', None)
                if _lap is not None:
                    cert.acceptance_prob = float(_lap)
            except Exception:
                pass
            try:
                _trace = locals().get('step_trace', None)
                if isinstance(_trace, list) and _trace:
                    cert.steps = _trace[-256:]
            except Exception:
                pass
            _emit_cert(cert)
    except Exception:
        pass
    
    return text


if __name__ == '__main__':
    main()
