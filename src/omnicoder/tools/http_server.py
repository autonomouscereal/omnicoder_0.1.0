from __future__ import annotations

import base64
import io
import os
import os as _os
from pathlib import Path
from typing import Optional, List, Tuple

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from omnicoder.utils.logger import get_logger  # type: ignore
from omnicoder.utils.env_registry import load_dotenv_best_effort
from omnicoder.utils.env_defaults import apply_core_defaults, apply_run_env_defaults, apply_profile


# Text generation helpers (student PT / ORT / teacher / vLLM)

def _generate_text(prompt: str, max_new_tokens: int, preset: str, device: str, ckpt: str | None) -> str:
	try:
		from omnicoder.inference.generate import build_mobile_model_by_name, generate, get_text_tokenizer, maybe_load_checkpoint  # type: ignore
		import torch
		model = build_mobile_model_by_name(preset)
		if ckpt:
			try:
				maybe_load_checkpoint(model, ckpt)
			except Exception:
				pass
		model.eval().to(device)
		tok = get_text_tokenizer(prefer_hf=True)
		ids = torch.tensor([[x for x in tok.encode(prompt)[:512]]], dtype=torch.long, device=device)
		out_ids = generate(model, ids, max_new_tokens=max_new_tokens)
		return tok.decode(out_ids[0].tolist())
	except Exception as e:
		try:
			get_logger("omnicoder.api").error("_generate_text failed: %s", str(e))
		except Exception:
			pass
		return ""


def _generate_text_onnx(prompt: str, max_new_tokens: int, onnx_path: str) -> str | None:
	try:
		from omnicoder.inference.runtimes.onnx_decode_generate import generate_with_onnx  # type: ignore
		return generate_with_onnx(onnx_path, prompt, max_new_tokens)
	except Exception:
		return None


def _generate_image(prompt: str, width: int, height: int, backend: str | None, device: str, sd_model: str | None, sd_local_path: str | None, out_path):
	try:
		from omnicoder.modeling.multimodal.image_pipeline import ImageGenPipeline  # type: ignore
		import torch
		pipe = ImageGenPipeline(backend=(backend or "diffusers"), device=device, dtype=(torch.float16 if device.startswith("cuda") else torch.float32), hf_id=sd_model or None, local_path=sd_local_path or None)
		ok = pipe.ensure_loaded()
		if not ok:
			return str(out_path)
		img = pipe.generate(prompt, steps=20, size=(int(width), int(height)), out_path=out_path)
		return str(img)
	except Exception:
		return str(out_path)


def _generate_video(prompt: str, frames: int, steps: int, device: str, backend: str | None, video_model: str | None, video_local_path: str | None, out_path):
	try:
		from omnicoder.modeling.multimodal.video_pipeline import VideoGenPipeline  # type: ignore
		pipe = VideoGenPipeline(backend=(backend or "diffusers"), device=device, hf_id=video_model or None, local_path=video_local_path or None)
		ok = pipe.ensure_loaded()
		if not ok:
			return None
		vp = pipe.generate(prompt, frames=int(frames), steps=int(steps), out_path=out_path)
		return str(vp)
	except Exception:
		return None


# Text generation helpers
_MODEL_CACHE: dict[tuple[str, str], object] = {}
_TOKENIZER_CACHE: Optional[object] = None
_LOADED_CKPT: Optional[str] = None


def _build_text_model_and_tokenizer(preset: str, device: str):
	from omnicoder.inference.generate import build_mobile_model_by_name  # type: ignore
	from omnicoder.training.simple_tokenizer import get_dual_tokenizers  # type: ignore
	import torch
	from omnicoder.utils.best_weights import find_best_for  # type: ignore

	global _MODEL_CACHE, _TOKENIZER_CACHE

	# Resolve device dynamically to avoid CUDA hangs if GPU isn't actually available
	resolved_device = device
	try:
		if device.startswith("cuda") and not torch.cuda.is_available():
			resolved_device = "cpu"
	except Exception:
		resolved_device = device

	key = (preset, resolved_device)
	model = _MODEL_CACHE.get(key)
	if model is None:
		# Prefer a best checkpoint if available for text modality
		best_ckpt = find_best_for('text', preset_hint=preset)
		model = build_mobile_model_by_name(preset)
		# If a best checkpoint exists, load it
		try:
			if best_ckpt:
				from omnicoder.inference.generate import maybe_load_checkpoint as _maybe_load  # type: ignore
				_ = _maybe_load(model, best_ckpt)
		except Exception:
			pass
		# Prefer fp16 on CUDA for speed; otherwise fp32
		try:
			if resolved_device.startswith("cuda"):
				torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
				torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
				model = model.half()
		except Exception:
			pass
		model.eval().to(resolved_device)
		_MODEL_CACHE[key] = model

	if _TOKENIZER_CACHE is None:
		try:
			tok_tok, tok_byte = get_dual_tokenizers(prefer_hf=True)
			# Enable dual substrate globally for inference
			# Centralized defaults/profile control DUAL_SUBSTRATE; avoid overriding here
			_TOKENIZER_CACHE = tok_tok if os.getenv('OMNICODER_FORBID_SIMPLE','1')=='0' else tok_tok
		except Exception:
			# Last resort fall back to token tokenizer only
			from omnicoder.training.simple_tokenizer import get_text_tokenizer  # type: ignore
			_TOKENIZER_CACHE = get_text_tokenizer(prefer_hf=False)
	return model, _TOKENIZER_CACHE


# Apply centralized defaults and quality profile at import time for the server process
try:
	load_dotenv_best_effort((".env", ".env.tuned"))
	apply_core_defaults(os.environ)  # type: ignore[arg-type]
	apply_run_env_defaults(os.environ)  # type: ignore[arg-type]
	apply_profile(os.environ, "quality")  # type: ignore[arg-type]
except Exception:
	pass


app = FastAPI(title="OmniCoder Inference Server")
_LOG = get_logger("omnicoder.api")
try:
	_LOG.info("api: FastAPI app created")
except Exception:
	pass

# ---- Minimal UI ----
_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>OmniCoder UI</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 20px; }
      h1 { font-size: 20px; }
      .row { display: flex; gap: 16px; flex-wrap: wrap; }
      .card { border: 1px solid #ddd; border-radius: 6px; padding: 12px; flex: 1 1 320px; }
      label { display: block; margin: 6px 0 2px; font-size: 12px; color: #444; }
      input, textarea, select, button { width: 100%; padding: 8px; box-sizing: border-box; }
      button { cursor: pointer; margin-top: 8px; }
      #imgout { max-width: 100%; border: 1px solid #eee; margin-top: 8px; }
      code { background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }
    </style>
  </head>
  <body>
    <h1>OmniCoder — Minimal UI</h1>
    <div class="row">
      <div class="card">
        <h3>Unified (single prompt)</h3>
        <label>Prompt</label>
        <textarea id="uni_prompt" rows="4" placeholder="Describe or request anything… (the model will decide text/image/video/audio)"></textarea>
        <label>Max new tokens (text fallback)</label>
        <input id="uni_tokens" type="number" value="64" />
        <button onclick="genUnified()">Run</button>
        <pre id="uni_text" style="white-space:pre-wrap;margin-top:8px;"></pre>
        <div id="uni_path" style="font-size:12px;color:#666;"></div>
        <img id="uni_img" style="max-width:100%;display:none;margin-top:8px;border:1px solid #eee;" />
      </div>

      <div class="card">
        <h3>Text</h3>
        <label>Prompt</label>
        <textarea id="txt_prompt" rows="4" placeholder="Write a short poem about the ocean"></textarea>
        <label>Backend</label>
        <select id="txt_backend">
          <option value="auto" selected>auto</option>
          <option value="teacher">teacher</option>
          <option value="student">student</option>
        </select>
        <label>Max new tokens</label>
        <input id="txt_tokens" type="number" value="64" />
        <button onclick="genText()">Generate Text</button>
        <pre id="txt_out" style="white-space:pre-wrap;margin-top:8px;"></pre>
      </div>

      <div class="card">
        <h3>Image</h3>
        <label>Prompt</label>
        <textarea id="img_prompt" rows="4" placeholder="A watercolor painting of snowy mountains"></textarea>
        <div class="row">
          <div style="flex:1">
            <label>Width</label>
            <input id="img_w" type="number" value="512" />
          </div>
          <div style="flex:1">
            <label>Height</label>
            <input id="img_h" type="number" value="512" />
          </div>
        </div>
        <button onclick="genImage()">Generate Image</button>
        <img id="imgout" />
        <div id="imgpath" style="font-size:12px;color:#666;"></div>
      </div>

      <div class="card">
        <h3>Video</h3>
        <label>Prompt</label>
        <textarea id="vid_prompt" rows="4" placeholder="A drone flyover of a beach at sunset"></textarea>
        <div class="row">
          <div style="flex:1">
            <label>Frames</label>
            <input id="vid_frames" type="number" value="24" />
          </div>
          <div style="flex:1">
            <label>Steps</label>
            <input id="vid_steps" type="number" value="25" />
          </div>
        </div>
        <button onclick="genVideo()">Generate Video</button>
        <div id="vidpath" style="font-size:12px;color:#666;margin-top:8px;"></div>
      </div>
    </div>

    <script>
      async function genUnified() {
        const prompt = document.getElementById('uni_prompt').value.trim();
        const max_new_tokens = parseInt(document.getElementById('uni_tokens').value || '64');
        const img = document.getElementById('uni_img');
        const uniText = document.getElementById('uni_text');
        const uniPath = document.getElementById('uni_path');
        img.style.display = 'none'; img.src=''; uniText.textContent = '…'; uniPath.textContent='';
        const res = await fetch('/generate/unified', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ prompt, max_new_tokens }) });
        const js = await res.json();
        // Display based on returned fields
        if (js.text) {
          uniText.textContent = js.text;
        } else {
          uniText.textContent = '';
        }
        if (js.image_path) {
          uniPath.textContent = js.image_path;
          // Attempt to load when accessible by server path mapping
          img.src = js.image_path; img.style.display = 'block';
        }
        if (js.video_path) {
          uniPath.textContent = js.video_path;
        }
        if (js.audio_path) {
          uniPath.textContent = js.audio_path;
        }
        if (!js.text && !js.image_path && !js.video_path && !js.audio_path && js.error) {
          uniText.textContent = js.error;
        }
      }
      async function genText() {
        const prompt = document.getElementById('txt_prompt').value.trim();
        const backend = document.getElementById('txt_backend').value;
        const max_new_tokens = parseInt(document.getElementById('txt_tokens').value || '64');
        document.getElementById('txt_out').textContent = '…';
        const res = await fetch('/generate/text', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ prompt, max_new_tokens, backend }) });
        const js = await res.json();
        document.getElementById('txt_out').textContent = js.text || js.error || JSON.stringify(js);
      }
      async function genImage() {
        const prompt = document.getElementById('img_prompt').value.trim();
        const width = parseInt(document.getElementById('img_w').value || '512');
        const height = parseInt(document.getElementById('img_h').value || '512');
        document.getElementById('imgout').src = '';
        document.getElementById('imgpath').textContent = '…';
        const res = await fetch('/generate/image', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ prompt, width, height, return_base64: true }) });
        const js = await res.json();
        if (js.image_base64) {
          document.getElementById('imgout').src = 'data:image/png;base64,' + js.image_base64;
          document.getElementById('imgpath').textContent = '';
        } else {
          document.getElementById('imgpath').textContent = js.image_path || js.error || '';
        }
      }
      async function genVideo() {
        const prompt = document.getElementById('vid_prompt').value.trim();
        const frames = parseInt(document.getElementById('vid_frames').value || '24');
        const steps = parseInt(document.getElementById('vid_steps').value || '25');
        document.getElementById('vidpath').textContent = '…';
        const res = await fetch('/generate/video', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ prompt, frames, steps }) });
        const js = await res.json();
        document.getElementById('vidpath').textContent = js.video_path || js.error || JSON.stringify(js);
      }
    </script>
  </body>
  </html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
	return HTMLResponse(_HTML)


@app.get("/ui", response_class=HTMLResponse)
def ui() -> HTMLResponse:
	return HTMLResponse(_HTML)


@app.get("/health")
def health() -> dict:
	try:
		_LOG.info("api: /health")
	except Exception:
		pass
	return {"status": "ok"}


# ---- Request models ----
class UnifiedReq(BaseModel):
	prompt: str
	max_new_tokens: int = 64
	image: str | None = None
	video: str | None = None
	audio: str | None = None
	target: str | None = None

class TextReq(BaseModel):
	prompt: str
	max_new_tokens: int = 64
	backend: str | None = None
	preset: str = "mobile_4gb"
	device: str = "cuda"
	onnx_decode: str | None = None
	ckpt: str | None = None

class ImageReq(BaseModel):
	prompt: str
	width: int = 512
	height: int = 512
	backend: str | None = None
	device: str = "cuda"
	sd_model: str | None = None
	sd_local_path: str | None = None
	return_base64: bool | None = False

class VideoReq(BaseModel):
	prompt: str
	frames: int = 24
	steps: int = 25
	device: str = "cuda"
	backend: str | None = None
	video_model: str | None = None
	video_local_path: str | None = None


@app.on_event("startup")
def _warm_start() -> None:
	# Warm default model/tokenizer once at startup to avoid per-request init
	try:
		_LOG.info("api: startup begin")
	except Exception:
		pass
	try:
		global _TOKENIZER_CACHE
		preset = os.environ.get("OMNICODER_API_PRESET", os.environ.get("OMNICODER_PRESET", "mobile_4gb"))
		device = os.environ.get("OMNICODER_TRAIN_DEVICE", "cuda")
		# HRM and verbosity are controlled by centralized defaults/profile
		# Harden against accidental simple-tokenizer override (produces '?' gibberish)
		try:
			if os.environ.get("OMNICODER_FORCE_SIMPLE_TOKENIZER", "0") == "1":
				_LOG.warning("startup: OMNICODER_FORCE_SIMPLE_TOKENIZER=1 detected — disabling to honor HF tokenizer/sweep")
				os.environ["OMNICODER_FORCE_SIMPLE_TOKENIZER"] = "0"
		except Exception:
			pass
		# Snapshot key tokenizer-related env for diagnostics (exclude secrets)
		try:
			env_keys = [
				"OMNICODER_API_PREFER_HF_TOKENIZER",
				"OMNICODER_API_FORCE_HF",
				"OMNICODER_HF_TOKENIZER",
				"OMNICODER_DISABLE_TOKENIZER_REMAP",
				"OMNICODER_FORBID_GPT2",
				"OMNICODER_FORCE_SIMPLE_TOKENIZER",
				"OMNICODER_TOKENIZER_DIAG",
				"OMNICODER_AUTO_SELECT_TOKENIZER",
				"OMNICODER_TOKENIZER_SWEEP_ENABLE",
				"OMNICODER_TOKENIZER_CANDIDATES",
				"OMNICODER_LLAMA3_SUBSET",
				"OMNICODER_NO_REPEAT_NGRAM",
				"OMNICODER_NO_REPEAT_WINDOW",
				"OMNICODER_MIN_P",
				"OMNICODER_REP_PENALTY",
				"OMNICODER_REP_WINDOW",
				"OMNICODER_FREQ_PENALTY",
				"OMNICODER_PRESENCE_PENALTY",
				"OMNICODER_BAN_REPEAT_RUN",
				"OMNICODER_BAN_LAST_TOKEN",
				"OMNICODER_MASK_NON_TEXT",
			]
			snap = {k: os.environ.get(k, "") for k in env_keys}
			hf_tok_present = bool(os.environ.get("HF_TOKEN"))
			_LOG.info("env tokenizer snapshot: %s (HF_TOKEN=%s)", snap, "set" if hf_tok_present else "unset")
		except Exception:
			pass
		# Probe SFB heavy deps once (but do not enable heavy SFB by default; avoids slowdown and conflicts)
		try:
			sfb_amr_ok = False
			sfb_allennlp_ok = False
			# If an isolated venv is provided for SFB heavy deps, expose its site-packages to sys.path
			try:
				import sys as _sys
				venv_path = _os.getenv("SFB_VENV", "").strip()
				if venv_path and _os.path.isdir(venv_path):
					pyver = f"python{_sys.version_info.major}.{_sys.version_info.minor}"
					added = []
					for p in (f"{venv_path}/lib/{pyver}/site-packages", f"{venv_path}/lib/site-packages"):
						if _os.path.isdir(p) and p not in _sys.path:
							_sys.path.insert(0, p)
							added.append(p)
					if added:
						_LOG.info("sfb probe: extended sys.path with %s", added)
			except Exception as _e:
				try:
					_LOG.info("sfb probe: venv sys.path extension failed: %s", str(_e))
				except Exception:
					pass
			try:
				import amrlib  # type: ignore
				_LOG.info("sfb probe: amrlib ok")
				sfb_amr_ok = True
			except Exception as e:
				_LOG.info("sfb probe: amrlib missing: %s", str(e))
			# Skip allennlp heavy probe to avoid import cost/conflicts with Torch 2.5
			try:
				import importlib
				spec = importlib.util.find_spec('allennlp')  # type: ignore[attr-defined]
				if spec is not None:
					sfb_allennlp_ok = True
					_LOG.info("sfb probe: allennlp present (skipping import)")
				else:
					_LOG.info("sfb probe: allennlp missing")
			except Exception:
				_LOG.info("sfb probe: allennlp missing")
			# If heavy deps are missing, mark SFB as unavailable to prevent per-request attempts
			try:
				_os.environ.setdefault("SFB_AVAILABLE", "1" if (sfb_amr_ok or sfb_allennlp_ok) else "0")
				_LOG.info("sfb probe: available=%s", _os.environ.get("SFB_AVAILABLE", "0"))
			except Exception:
				pass
		except Exception:
			pass
		# Move heavy SFB init/failure marking to startup to avoid repeated attempts during generation
		try:
			if _os.getenv("SFB_ENABLE", _os.getenv("OMNICODER_SFB_ENABLE", "0")) == "1":
				from omnicoder.sfb.factorize import factorize_prompt  # type: ignore
				_LOG.info("sfb warmup: starting")
				try:
					# This will set internal TRIED flags and cache small components
					factorize_prompt("warmup")
				except Exception as _e:
					_LOG.info("sfb warmup: completed with errors (expected if heavy deps missing): %s", str(_e))
		except Exception:
			pass
		# Logging level/file defaults come from centralized defaults; do not override here
		# Prefer ONNX decode-step if available; skip heavy PT warmup to reduce startup latency
		onnx_default = os.environ.get("OMNICODER_API_ONNX_DECODE", "/workspace/weights/release/text/omnicoder_decode_step.onnx")
		if onnx_default and os.path.exists(onnx_default):
			# Quick ORT session probe for diagnostics
			try:
				import onnxruntime as ort  # type: ignore
				prov = os.environ.get("OMNICODER_ORT_PROVIDER", "auto")
				provs = [prov] if prov != "auto" else None
				if provs is None:
					avail = []
					try:
						avail = list(ort.get_available_providers())
					except Exception:
						avail = []
					pref = ['CUDAExecutionProvider','DmlExecutionProvider','CPUExecutionProvider']
					provs = [p for p in pref if p in avail] or ["CPUExecutionProvider"]
				_LOG.info("onnx probe: model=%s providers=%s", onnx_default, provs)
				# Guard older/namespace-collided installs that miss InferenceSession
				if not hasattr(ort, 'InferenceSession'):
					raise RuntimeError("module 'onnxruntime' has no attribute 'InferenceSession'")
				sess = ort.InferenceSession(onnx_default, providers=provs)  # type: ignore
				outs = [o.name for o in sess.get_outputs()]
				_LOG.info("onnx probe: outputs=%s", outs[:4])
				# If only CPUExecutionProvider is available and CUDA is present, prefer PyTorch CUDA path
				try:
					avail = list(ort.get_available_providers())
				except Exception:
					avail = []
				try:
					import torch as _t
					has_cuda = bool(_t.cuda.is_available())
				except Exception:
					has_cuda = False
				if (avail == ["CPUExecutionProvider"] or ("CUDAExecutionProvider" not in avail)) and has_cuda:
					os.environ["OMNICODER_USE_ONNX"] = "0"
					os.environ["OMNICODER_ONNX_OK"] = "0"
					_LOG.info("onnx probe: disabling OMNICODER_USE_ONNX due to missing GPU ORT provider (CUDA available)")
				else:
					os.environ["OMNICODER_ONNX_OK"] = "1"
			except Exception as e:
				_LOG.error("onnx probe failed: %s", str(e))
				# Force PT fallback to avoid hangs when ORT import is corrupt
				try:
					os.environ["OMNICODER_USE_ONNX"] = "0"
					os.environ["OMNICODER_ONNX_OK"] = "0"
				except Exception:
					pass
			# Log CUDA visibility for PT fallback speed diagnostics
			try:
				import torch as _t
				_LOG.info("cuda_available=%s cuda_device_count=%s", str(_t.cuda.is_available()), str(_t.cuda.device_count()))
			except Exception as _onnx_e:
				try:
					_LOG.error("onnx probe failed: %s", str(_onnx_e))
				except Exception:
					pass
				# Disable ONNX path for this session
				try:
					os.environ["OMNICODER_USE_ONNX"] = "0"
					os.environ["OMNICODER_ONNX_OK"] = "0"
				except Exception:
					pass
			# Set a reasonable default tokenizer based on released vocab size
			try:
				import json as _json
				um = Path("/workspace/weights/release/unified_vocab_map.json")
				meta = _json.loads(um.read_text(encoding='utf-8')) if um.exists() else {}
				tsize = int(meta.get("text_size", 0)) if isinstance(meta, dict) else 0
				env_tok = os.environ.get("OMNICODER_HF_TOKENIZER", "").strip()
				if os.getenv("OMNICODER_DISABLE_TOKENIZER_REMAP", "0") != "1":
					if tsize == 32000 and env_tok and ("meta-llama" in env_tok.lower()):
						_LOG.info("startup remap: text_size=32000 and env OMNICODER_HF_TOKENIZER=%s -> hf-internal-testing/llama-tokenizer", env_tok)
						os.environ["OMNICODER_HF_TOKENIZER"] = "hf-internal-testing/llama-tokenizer"
					elif tsize == 32000 and not env_tok:
						os.environ["OMNICODER_HF_TOKENIZER"] = "hf-internal-testing/llama-tokenizer"
			except Exception:
				pass
		# Do not return early — still warm the PyTorch model/checkpoint so generation never does heavy init per request
		# Select checkpoint(s) to load at startup. Prefer explicit chain, else a
		# single explicit CKPT, else discover the best final stage (PPO→GRPO→ALIGN → PRETRAIN/ DRAFT as fallbacks).
		ckpt = os.environ.get("OMNICODER_API_CKPT", None)
		env_chain = os.environ.get("OMNICODER_API_CKPT_CHAIN", "").strip()
		chain: list[str] = []
		if env_chain:
			for t in env_chain.replace(",", " ").split():
				try:
					if Path(t).exists() and t not in chain:
						chain.append(t)
				except Exception:
					pass
		if not chain:
			if ckpt and Path(ckpt).exists():
				chain.append(str(ckpt))
			else:
				discover = [
					"weights/run_full/omnicoder_ppo.pt",
					"weights/omnicoder_ppo.pt",
					"weights/run_full/omnicoder_grpo.pt",
					"weights/omnicoder_grpo.pt",
					"weights/run_full/omnicoder_align.pt",
					"weights/omnicoder_align.pt",
					# Add earlier-stage fallbacks to hydrate missing heads/routers if needed
					"weights/omnicoder_pretrain_dsm.pt",
					"weights/omnicoder_draft_kd.pt",
				]
				for c in discover:
					try:
						if Path(c).exists():
							chain.append(c)
							os.environ["OMNICODER_API_CKPT"] = c
							break
					except Exception:
						pass
		# Respect explicit HF force; do not override tokenizer selection based on ckpt name
		try:
			base = Path(ckpt).name if ckpt else ""
			if base:
				_LOG.info("startup: ckpt base=%s (OMNICODER_API_FORCE_HF=%s)", base, os.environ.get("OMNICODER_API_FORCE_HF",""))
		except Exception:
			pass
		model, tok = _build_text_model_and_tokenizer(preset, device)
		# Build and set a small global draft model for speculative decoding (IO-free in hot path)
		try:
			from omnicoder.inference.generate import build_mobile_model_by_name, set_global_draft_model  # type: ignore
			draft_preset = os.environ.get("OMNICODER_API_DRAFT_PRESET", "draft_2b")
			draft_model = build_mobile_model_by_name(draft_preset, mem_slots=0, skip_init=True)
			set_global_draft_model(draft_model)
			_LOG.info("api: global draft model set preset=%s", draft_preset)
		except Exception as _e:
			try:
				_LOG.warning("api: draft preload failed/skipped: %s", str(_e))
			except Exception:
				pass
		# Assert model and all experts reside on the intended device to avoid accidental CPU execution
		try:
			mdev = str(next(model.parameters()).device)
			if device.startswith('cuda') and (not mdev.startswith('cuda')):
				_LOG.warning("startup: model on %s but requested device=%s; moving to device", mdev, device)
				model = model.to(device)
				# refresh cache
				_MODEL_CACHE[(preset, device)] = model
		except Exception:
			pass
		# If we later collapse MoE to dense (due to missing router/expert weights), log it at startup
		try:
			from omnicoder.modeling.transformer_moe import MoELayer  # type: ignore
			n_moe = sum(1 for b in getattr(model, 'blocks', []) if hasattr(b, 'moe'))
			single_expert = 0
			degraded = 0
			for b in getattr(model, 'blocks', []):
				if hasattr(b, 'moe'):
					m = b.moe
					try:
						if int(getattr(m, 'n_experts', 0)) == 1 and int(getattr(m, 'top_k', 0)) == 1:
							single_expert += 1
						if bool(getattr(m, '_degraded_router', False)):
							degraded += 1
					except Exception:
						pass
			_LOG.info("startup: moe_layers=%d single_expert=%d degraded=%d", int(n_moe), int(single_expert), int(degraded))
		except Exception:
			pass
		# If configured, run tokenizer sweep at startup and optionally auto-select best
		try:
			if os.getenv("OMNICODER_TOKENIZER_DIAG", "0") == "1":
				cand_env = os.environ.get("OMNICODER_TOKENIZER_CANDIDATES", "").strip()
				candidates: List[str] = []
				if cand_env:
					for c in cand_env.replace(",", " ").split():
						if c:
							candidates.append(c)
				env_id = os.environ.get("OMNICODER_HF_TOKENIZER", "").strip()
				if env_id:
					candidates.append(env_id)
				tlocal = os.environ.get("OMNICODER_API_TEACHER_PATH", "").strip()
				if tlocal and os.path.isdir(tlocal):
					candidates.insert(0, tlocal)
				candidates += [
					"hf-internal-testing/llama-tokenizer",
					"openlm-research/open_llama_7b",
				]
				_LOG.info("warm_start: tokenizer_diag candidates=%s", candidates)
				diag = _diagnose_tokenizers(model, candidates)
				_LOG.info("warm_start: tokenizer_diag results(head)=%s", diag[:4])
				if os.getenv("OMNICODER_AUTO_SELECT_TOKENIZER", "0") == "1":
					try:
						model_vocab_size = int(getattr(model, 'vocab_size', 0) or 0)
					except Exception:
						model_vocab_size = 0
					best_tok_for_model = None
					best_nll_for_model = float('inf')
					for r in diag:
						if "avg_nll" not in r:
							continue
						name = r.get("tokenizer")
						if name in (None, "simple32k", "__simple32k__"):
							continue
						try:
							vs = int(r.get("vocab_size") or 0)
						except Exception:
							vs = 0
						# Allow equal or superset vocab; if superset, we'll wrap down to model_vocab_size
						if model_vocab_size and vs != model_vocab_size:
							if not (vs > model_vocab_size and model_vocab_size in (32000, 65536)):
								continue
						try:
							nll = float(r["avg_nll"])
						except Exception:
							continue
						if nll < best_nll_for_model:
							best_nll_for_model = nll
							best_tok_for_model = str(name)
					if best_tok_for_model:
						try:
							_LOG.warning(
								"warm_start: auto-select tokenizer=%s avg_nll=%.4f (model_vocab=%s)",
								best_tok_for_model, best_nll_for_model, str(model_vocab_size),
							)
							from omnicoder.training.simple_tokenizer import AutoTokenizerWrapper  # type: ignore
							tok_obj = AutoTokenizerWrapper(best_tok_for_model)
							_TOKENIZER_CACHE = tok_obj
							# Wrap superset to subset of model vocab
							try:
								mv = int(getattr(model, 'vocab_size', 0) or 0)
								tv = int(getattr(tok_obj, 'vocab_size', 0) or 0)
								if mv and tv and tv != mv:
									from omnicoder.training.simple_tokenizer import SubsetTokenizerWrapper  # type: ignore
									_TOKENIZER_CACHE = SubsetTokenizerWrapper(tok_obj, allowed_vocab=mv)
									_LOG.warning("warm_start: wrapped selected tokenizer to subset vocab=%s", mv)
							except Exception:
								pass
						except Exception as e:
							_LOG.error("warm_start: tokenizer auto-select failed: %s", str(e))
		except Exception:
			pass
		# Load checkpoints; if a chain was specified, load in order. Otherwise, load only the single best ckpt.
		from omnicoder.inference.generate import maybe_load_checkpoint  # type: ignore
		try:
			if chain:
				_LOG.info("warm_start: loading ckpt chain=%s", chain)
				# Pre-check coverage before actually loading to avoid corrupting the model with incompatible checkpoints
				def _estimate_ckpt_coverage(_model, _path: str) -> float:
					try:
						import torch as _t
						state = _t.load(_path, map_location='cpu')
					except Exception as _e:
						try:
							_LOG.error("warm_start: coverage precheck load failed for %s: %s", _path, str(_e))
						except Exception:
							pass
						return 0.0
					try:
						msd = _model.state_dict()
					except Exception:
						return 0.0
					loaded_param_elems = 0
					total_param_elems = 0
					try:
						for k, dst in msd.items():
							total_param_elems += int(getattr(dst, 'numel', lambda: 0)())
						for k, src in state.items():  # type: ignore[attr-defined]
							if k in msd and hasattr(src, 'shape') and hasattr(msd[k], 'shape'):
								s = tuple(src.shape)
								d = tuple(msd[k].shape)
								if s == d:
									loaded_param_elems += int(getattr(src, 'numel', lambda: 0)())
								else:
									# Consider adaptable vocab-only mismatches as covered (slice/pad path)
									try:
										if k.endswith('lm_head.weight') and len(s) == 2 and len(d) == 2 and s[1] == d[1]:
											loaded_param_elems += int(d[0] * d[1])
										elif (k.endswith('embed.weight') or k.endswith('tok_embed.weight')) and len(s) == 2 and len(d) == 2 and s[1] == d[1]:
											loaded_param_elems += int(d[0] * d[1])
										elif k.endswith('lm_head.bias') and len(s) == 1 and len(d) == 1:
											loaded_param_elems += int(d[0])
									except Exception:
										pass
					except Exception:
						pass
					if total_param_elems <= 0:
						return 0.0
					return float(100.0 * (float(loaded_param_elems) / float(total_param_elems)))

				try:
					min_cov = float(os.environ.get("OMNICODER_MIN_CKPT_COVERAGE", "85"))
				except Exception:
					min_cov = 85.0
				last_applied: str | None = None
				import glob as _glob
				try:
					_moe_param_files = _glob.glob('weights/blocks.*.moe.experts.*.weight')
				except Exception:
					_moe_param_files = []
				for p in chain:
					# Pre-check
					cov_est = _estimate_ckpt_coverage(model, p)
					_LOG.info("warm_start: precheck coverage path=%s est=%.2f%% min=%.2f%%", p, cov_est, min_cov)
					allow_low = os.environ.get("OMNICODER_ALLOW_LOW_COVERAGE", "0") == "1"
					if cov_est < min_cov and not allow_low:
						_LOG.warning(
							"warm_start: skipping checkpoint due to low coverage (%.2f%% < %.2f%%): %s (set OMNICODER_ALLOW_LOW_COVERAGE=1 to force)",
							cov_est, min_cov, p
						)
						continue
					if cov_est < min_cov and allow_low:
						_LOG.warning(
							"warm_start: coverage %.2f%% < %.2f%% but override enabled; attempting load",
							cov_est, min_cov
						)
					_LOG.info("warm_start: maybe_load_checkpoint path=%s", p)
					stats = maybe_load_checkpoint(model, p)
					try:
						cov = 100.0 * (float(stats.get('loaded_param_elems', 0)) / max(1.0, float(stats.get('total_param_elems', 0)))) if stats else 0.0
					except Exception:
						cov = 0.0
					_LOG.info("warm_start: ckpt coverage path=%s keys=%s/%s params=%.2f%% missing=%s unexpected=%s",
							  p,
							  stats.get('loaded_keys', 0) if stats else 0,
							  stats.get('total_model_keys', 0) if stats else 0,
							  cov,
							  stats.get('missing', 0) if stats else 0,
							  stats.get('unexpected', 0) if stats else 0)
					last_applied = p
				# If final head appears untrained (very low std), try to hydrate lm_head/embed specifically from draft KD
				try:
					import torch as _t
					e_w = getattr(getattr(model, 'embed', object()), 'weight', None)
					h_w = getattr(getattr(model, 'lm_head', object()), 'weight', None)
					if e_w is not None and h_w is not None:
						s_head = float(h_w.detach().float().std().item())
						if s_head < 1e-6:
							cand = Path('weights') / 'omnicoder_draft_kd.pt'
							if cand.exists():
								try:
									_LOG.warning("warm_start: lm_head std ~ 0 — hydrating head from %s", str(cand))
									sd = _t.load(str(cand), map_location='cpu')
									if isinstance(sd, dict):
										# Accept common wrappers
										for k in ('state_dict','model','module','student','network'):
											obj = sd.get(k)
											if isinstance(obj, dict):
												sd = obj
												break
										for k in list(sd.keys()):
											if k.endswith('lm_head.weight') and tuple(getattr(model.lm_head, 'weight').shape) == tuple(getattr(sd[k], 'shape', ())):  # type: ignore[attr-defined]
												model.lm_head.weight.data.copy_(sd[k])  # type: ignore[index]
												_LOG.warning("warm_start: copied lm_head.weight from draft KD")
												break
								except Exception as _e:
									_LOG.error("warm_start: head hydration failed: %s", str(_e))
				except Exception:
					pass
				# Log final lm_head/embed shapes after chain load
				try:
					e_shape = tuple(getattr(getattr(model, 'embed', object()), 'weight', None).shape) if getattr(getattr(model, 'embed', object()), 'weight', None) is not None else None
					h_shape = tuple(getattr(getattr(model, 'lm_head', object()), 'weight', None).shape) if getattr(getattr(model, 'lm_head', object()), 'weight', None) is not None else None
					_LOG.info("warm_start: post-load embed=%s lm_head=%s", str(e_shape), str(h_shape))
				except Exception:
					pass
				global _LOADED_CKPT
				_LOADED_CKPT = last_applied or (chain[-1] if chain else None)
			# Now that weights (if any) have been loaded into the real model, compile it for speed
			try:
				import torch as _t
				if _t.cuda.is_available() and os.getenv('OMNICODER_COMPILE','1') == '1':
					# Prefer safe aot_eager; allow Inductor opt-in and suppress Dynamo errors
					try:
						import torch._dynamo as _dyn  # type: ignore[attr-defined]
						_dyn.config.suppress_errors = True  # type: ignore[attr-defined]
					except Exception:
						pass
					# Enable TF32 for speed on Ampere+
					try:
						_t.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
						_t.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
					except Exception:
						pass
					allow_inductor = (os.getenv('OMNICODER_ALLOW_INDUCTOR','0') == '1')
					backends = ['aot_eager']
					if allow_inductor:
						try:
							import importlib as _il  # type: ignore
							_il.import_module('torch._inductor.compile_worker.watchdog')
							backends.append('inductor')
						except Exception:
							_LOG.warning("startup: inductor watchdog missing; skipping inductor backend")
					compiled = None
					last_err = None
					for backend in backends:
						try:
							_LOG.info("startup: compiling model with torch.compile backend=%s", backend)
							if backend == 'aot_eager':
								compiled = _t.compile(model, backend=backend)  # type: ignore[attr-defined]
							else:
								# Guard to keep tolerance to minor graph breaks
								compiled = _t.compile(model, mode='reduce-overhead', backend=backend, fullgraph=False)  # type: ignore[attr-defined]
							compiled.eval()
							warm = _t.randint(0, getattr(model, 'vocab_size', 32000), (1,1), dtype=_t.long, device=next(model.parameters()).device)
							_ = compiled(warm, past_kv=None, use_cache=True)
							model = compiled  # type: ignore[assignment]
							_LOG.info("startup: compile ok backend=%s", backend)
							# Update global cache so request path reuses compiled model
							try:
								resolved_device = device
								try:
									if str(device).startswith("cuda") and not _t.cuda.is_available():
										resolved_device = "cpu"
								except Exception:
									pass
								from omnicoder.tools.http_server import _MODEL_CACHE  # type: ignore
								_MODEL_CACHE[(preset, resolved_device)] = model
							except Exception:
								pass
							break
						except Exception as _ce:
							last_err = _ce
							_LOG.warning("startup: compile failed backend=%s: %s", backend, str(_ce))
							compiled = None
					if compiled is None:
						if last_err is not None:
							_LOG.warning("startup: compile skipped after all backends failed: %s", str(last_err))
						# Ensure eager path stays stable and suppresses dynamo errors when present
						try:
							import torch._dynamo as _dyn  # type: ignore[attr-defined]
							_dyn.config.suppress_errors = True  # type: ignore[attr-defined]
						except Exception:
							pass
			except Exception as _e:
				try:
					_LOG.warning("startup: compile skipped: %s", str(_e))
				except Exception:
					pass
			# Optional: diagnose tokenizer alignment and auto-select a better match
			try:
				if os.getenv("OMNICODER_TOKENIZER_DIAG", "0") == "1":
					cand_env = os.environ.get("OMNICODER_TOKENIZER_CANDIDATES", "").strip()
					candidates: List[str] = []
					if cand_env:
						for c in cand_env.replace(",", " ").split():
							if c:
								candidates.append(c)
					# Include current env id, teacher local path, common llama ids
					env_id = os.environ.get("OMNICODER_HF_TOKENIZER", "").strip()
					if env_id:
						candidates.append(env_id)
					tlocal = os.environ.get("OMNICODER_API_TEACHER_PATH", "").strip()
					if tlocal and os.path.isdir(tlocal):
						candidates.insert(0, tlocal)
					candidates += [
						"hf-internal-testing/llama-tokenizer",
						"openlm-research/open_llama_7b",
					]
					_LOG.info("tokenizer_diag: candidates=%s", candidates)
					diag = _diagnose_tokenizers(model, candidates)
					_LOG.info("tokenizer_diag: results=%s", diag[:4])
					# Auto-select best if allowed — but only if vocab_size matches the model
					if os.getenv("OMNICODER_AUTO_SELECT_TOKENIZER", "0") == "1":
						try:
							model_vocab_size = int(getattr(model, 'vocab_size', 0) or 0)
						except Exception:
							model_vocab_size = 0
						best_tok_for_model = None
						best_nll_for_model = float('inf')
						for r in diag:
							if "avg_nll" not in r:
								continue
							name = r.get("tokenizer")
							if name in (None, "simple32k"):
								continue
							try:
								vs = int(r.get("vocab_size") or 0)
							except Exception:
								vs = 0
							if model_vocab_size and vs != model_vocab_size:
								# Incompatible tokenizer — skip
								continue
							try:
								nll = float(r["avg_nll"])
							except Exception:
								continue
							if nll < best_nll_for_model:
								best_nll_for_model = nll
								best_tok_for_model = str(name)
						if best_tok_for_model:
							try:
								_LOG.warning(
									"tokenizer_diag: auto-selecting compatible tokenizer=%s avg_nll=%.4f (model_vocab=%s)",
									best_tok_for_model, best_nll_for_model, str(model_vocab_size),
								)
								from omnicoder.training.simple_tokenizer import AutoTokenizerWrapper  # type: ignore
								tok_obj = AutoTokenizerWrapper(best_tok_for_model)
								_TOKENIZER_CACHE = tok_obj
							except Exception as e:
								_LOG.error("tokenizer_diag: auto-select failed: %s", str(e))
								# Force re-evaluation via default path on first request
								_TOKENIZER_CACHE = None
						else:
							_LOG.warning(
								"tokenizer_diag: no compatible tokenizer found (model_vocab=%s); keeping default selection",
								str(model_vocab_size),
							)
			except Exception as e:
				try:
					_LOG.error("tokenizer_diag failed: %s", str(e))
				except Exception:
					pass
		except Exception:
			pass
	except Exception as e:
		# Keep server alive even if warm fails; it will lazy-init on first request
		pass

	try:
		_LOG.info("api: startup end")
	except Exception:
		pass

############################
# HF teacher text backend  #
############################
_TEACHER_CACHE: dict[str, object] = {}


def _ensure_hf_teacher(hf_id: str, device: str):
	import torch
	from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

	# Prefer local path if provided to avoid network
	local_path = os.environ.get("OMNICODER_API_TEACHER_PATH", "").strip()
	src = local_path if (local_path and os.path.isdir(local_path)) else hf_id
	key = f"{src}|{device}"
	if key in _TEACHER_CACHE:
		return _TEACHER_CACHE[key]

	tok = AutoTokenizer.from_pretrained(
		src,
		token=os.environ.get("HF_TOKEN"),
		use_auth_token=None,
		local_files_only=bool(local_path),
		trust_remote_code=True,
	)
	if tok.pad_token is None:
		tok.pad_token = tok.eos_token
	dtype = torch.float16 if device.startswith("cuda") else torch.float32
	model = AutoModelForCausalLM.from_pretrained(
		src,
		dtype=dtype,
		device_map="auto" if device.startswith("cuda") else None,
		low_cpu_mem_usage=True,
		token=os.environ.get("HF_TOKEN"),
		use_auth_token=None,
		local_files_only=bool(local_path),
		trust_remote_code=True,
	)
	if not device.startswith("cuda"):
		model = model.to(device)
	model.eval()
	_TEACHER_CACHE[key] = (model, tok)
	return _TEACHER_CACHE[key]


def _generate_text_hf_teacher(prompt: str, max_new_tokens: int, device: str) -> str:
	import torch
	try:
		hf_id = os.environ.get("OMNICODER_API_TEACHER_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
		model, tok = _ensure_hf_teacher(hf_id, device)
		inputs = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
		with torch.no_grad():
			out_ids = model.generate(
				**inputs,
				do_sample=True,
				temperature=0.9,
				top_k=50,
				top_p=0.95,
				max_new_tokens=int(max_new_tokens),
				pad_token_id=tok.eos_token_id,
			)
		return tok.decode(out_ids[0], skip_special_tokens=True)
	except Exception as e:
		try:
			_LOG.error("teacher backend failed: %s", str(e))
		except Exception:
			pass
		return ""


@app.post("/generate/text")
def generate_text(req: TextReq) -> dict:
	# Backend selection (prefer student unless explicitly asked for teacher)
	try:
		_LOG.info(
			"api: /generate/text req prompt_len=%d max_new_tokens=%d backend_req=%s device_req=%s",
			len(getattr(req, 'prompt', '') or ''),
			int(getattr(req, 'max_new_tokens', 0) or 0),
			str(getattr(req, 'backend', None)),
			str(getattr(req, 'device', None)),
		)
	except Exception:
		pass
	use_onnx = os.environ.get("OMNICODER_USE_ONNX", "0") == "1" and os.environ.get("OMNICODER_ONNX_OK", "0") == "1"
	onnx_default = os.environ.get("OMNICODER_API_ONNX_DECODE", "/workspace/weights/release/text/omnicoder_decode_step.onnx")
	onnx_path = (req.onnx_decode or onnx_default) if use_onnx and os.path.exists(onnx_default) else None
	if use_onnx and onnx_path and os.path.exists(onnx_path):
		backend = 'onnx'
		out = _generate_text_onnx(req.prompt, req.max_new_tokens, onnx_path)
		if out is None:
			# Silent fallback to teacher; never expose error string
			try:
				# Strictly student-only when requested
				backend = (req.backend or os.environ.get("OMNICODER_API_TEXT_BACKEND", "student")).lower()
				if backend == 'teacher':
					out = _generate_text_hf_teacher(req.prompt, req.max_new_tokens, req.device)
				else:
					ckpt = req.ckpt or os.environ.get("OMNICODER_API_CKPT", None)
					out = _generate_text(req.prompt, req.max_new_tokens, req.preset, req.device, ckpt)
			except Exception:
				out = ""
	else:
		backend = (req.backend or os.environ.get("OMNICODER_API_TEXT_BACKEND", "student")).lower()
		if backend == "teacher":
			out = _generate_text_hf_teacher(req.prompt, req.max_new_tokens, req.device)
		elif backend == "vllm":
			try:
				from omnicoder.inference.runtimes.vllm_adapter import generate_text_vllm  # type: ignore
				out = generate_text_vllm(req.prompt, req.max_new_tokens, model_id=os.environ.get("OMNICODER_VLLM_MODEL"))
			except Exception as _e:
				_LOG.warning("vllm backend failed; falling back to student: %s", str(_e))
				ckpt = req.ckpt or os.environ.get("OMNICODER_API_CKPT", None)
				out = _generate_text(req.prompt, req.max_new_tokens, req.preset, req.device, ckpt)
		else:
			ckpt = req.ckpt or os.environ.get("OMNICODER_API_CKPT", None)
			out = _generate_text(req.prompt, req.max_new_tokens, req.preset, req.device, ckpt)
	try:
		_LOG.info("gen_text backend=%s use_onnx=%s onnx=%s", backend if 'backend' in locals() else ('onnx' if (use_onnx and onnx_path and os.path.exists(onnx_path)) else 'student'), str(use_onnx), str(bool(onnx_path)))
	except Exception:
		pass
	try:
		_LOG.info("api: /generate/text response len=%d", len(out or ""))
	except Exception:
		pass
	return {"text": out}


@app.get("/debug/info")
def debug_info() -> dict:
	try:
		try:
			_LOG.info("api: /debug/info")
		except Exception:
			pass
		preset = os.environ.get("OMNICODER_API_PRESET", os.environ.get("OMNICODER_PRESET", "mobile_4gb"))
		device = os.environ.get("OMNICODER_TRAIN_DEVICE", "cuda")
		model, tok = _build_text_model_and_tokenizer(preset, device)
		mv = getattr(model, 'vocab_size', None)
		e_shape = tuple(getattr(getattr(model, 'embed', object()), 'weight', None).shape) if getattr(getattr(model, 'embed', object()), 'weight', None) is not None else None
		h_shape = tuple(getattr(getattr(model, 'lm_head', object()), 'weight', None).shape) if getattr(getattr(model, 'lm_head', object()), 'weight', None) is not None else None
		tv = int(getattr(tok, 'vocab_size', 0) or 0)
		tok_path = getattr(getattr(tok, "_tok", object()), "name_or_path", None)
		info = {
			"model_vocab": mv,
			"embed_shape": e_shape,
			"lm_head_shape": h_shape,
			"tokenizer_vocab": tv,
			"tokenizer_path": tok_path or os.environ.get("OMNICODER_HF_TOKENIZER", ""),
			"ckpt_loaded": _LOADED_CKPT,
		}
		try:
			# Provide last recorded ckpt coverage from logs by recomputing a tiny stat: count of equal-shaped params
			from collections import Counter as _Counter  # noqa: F401
			msd = getattr(model, 'state_dict', lambda: {})()
			info["model_keys"] = len(msd)
		except Exception:
			pass
		try:
			_LOG.info(
				"api: /debug/info ok model_vocab=%s lm_head=%s tokenizer_vocab=%s tokenizer_path=%s ckpt_loaded=%s",
				str(info.get("model_vocab")),
				str(info.get("lm_head_shape")),
				str(info.get("tokenizer_vocab")),
				str(info.get("tokenizer_path")),
				str(info.get("ckpt_loaded")),
			)
		except Exception:
			pass
		return info
	except Exception as e:
		try:
			_LOG.error("api: /debug/info error: %s", str(e))
		except Exception:
			pass
		return {"error": str(e)}


@app.get("/debug/tokenizer_sweep")
def debug_tokenizer_sweep() -> dict:
	try:
		preset = os.environ.get("OMNICODER_API_PRESET", os.environ.get("OMNICODER_PRESET", "mobile_4gb"))
		device = os.environ.get("OMNICODER_TRAIN_DEVICE", "cuda")
		model, _tok = _build_text_model_and_tokenizer(preset, device)
		cand_env = os.environ.get("OMNICODER_TOKENIZER_CANDIDATES", "").strip()
		candidates: List[str] = []
		if cand_env:
			for c in cand_env.replace(",", " ").split():
				if c:
					candidates.append(c)
		env_id = os.environ.get("OMNICODER_HF_TOKENIZER", "").strip()
		if env_id:
			candidates.append(env_id)
		tlocal = os.environ.get("OMNICODER_API_TEACHER_PATH", "").strip()
		if tlocal and os.path.isdir(tlocal):
			candidates.insert(0, tlocal)
		# Expand built-in sweep set with ~15 LLaMA-family/public tokenizers
		candidates += [
			"hf-internal-testing/llama-tokenizer",
			"openlm-research/open_llama_3b",
			"openlm-research/open_llama_7b",
			"openlm-research/open_llama_13b",
			"openlm-research/open_llama_7b_v2",
			"openlm-research/open_llama_13b_v2",
			"NousResearch/Llama-2-7b-hf",
			"meta-llama/Llama-2-7b-hf",
			"mistralai/Mistral-7B-v0.1",
			"tiiuae/falcon-7b",
			"TheBloke/OpenLlama-7B-GGUF",
			"decapoda-research/llama-7b-hf",
			"openaccess-ai-collective/open_llama_7b_v2",
			"Weyaxi/LLaMA-7B-Tokenizer",
			"bigscience/bloom-560m",
		]
		res = _diagnose_tokenizers(model, candidates)
		return {"results": res}
	except Exception as e:
		return {"error": str(e)}


@app.get("/debug/tokenizer_probe")
def debug_tokenizer_probe() -> dict:
	"""Generate a short preview for a fixed prompt with several tokenizer candidates without changing global selection.

	Returns a list of {tokenizer, vocab_size, preview}.
	"""
	try:
		prompt = os.environ.get("OMNICODER_PROBE_PROMPT", "Write one sentence about pizza.")
		preset = os.environ.get("OMNICODER_API_PRESET", os.environ.get("OMNICODER_PRESET", "mobile_4gb"))
		device = os.environ.get("OMNICODER_TRAIN_DEVICE", "cuda")
		model, _tok_main = _build_text_model_and_tokenizer(preset, device)
		from omnicoder.training.simple_tokenizer import AutoTokenizerWrapper, TextTokenizer  # type: ignore
		cand_env = os.environ.get("OMNICODER_TOKENIZER_CANDIDATES", "").strip()
		cands: list[str] = []
		if cand_env:
			for c in cand_env.replace(",", " ").split():
				if c:
					cands.append(c)
		cands = list(dict.fromkeys(cands + [
			os.environ.get("OMNICODER_HF_TOKENIZER", "").strip(),
			"hf-internal-testing/llama-tokenizer",
			"openlm-research/open_llama_7b",
		]))
		previews: list[dict] = []
		for cid in [c for c in cands if c]:
			try:
				tok = AutoTokenizerWrapper(cid)
			except Exception:
				try:
					tok = TextTokenizer(vocab_size=32000)
				except Exception:
					continue
			try:
				ids = tok.encode(prompt)
				from torch import no_grad
				import torch
				with no_grad():
					inp = torch.tensor([ids[:64]], dtype=torch.long, device=next(model.parameters()).device)  # type: ignore[attr-defined]
					logits = model(inp)
					if isinstance(logits, tuple):
						logits = logits[0]
					# Sample 16 tokens greedily for a fast preview
					gen = inp.clone()
					for _ in range(16):
						l = logits[:, -1, :]
						nxt = int(torch.argmax(l, dim=-1).item())
						gen = torch.cat([gen, torch.tensor([[nxt]], dtype=torch.long, device=gen.device)], dim=1)
						logits = model(gen)
						if isinstance(logits, tuple):
							logits = logits[0]
				txt = tok.decode(gen[0].tolist())
				try:
					vs = int(getattr(tok, 'vocab_size', 0) or 0)
				except Exception:
					vs = 0
				previews.append({"tokenizer": cid, "vocab_size": vs, "preview": txt[:160]})
			except Exception as _e:
				previews.append({"tokenizer": cid, "error": str(_e)})
		return {"prompt": prompt, "previews": previews}
	except Exception as e:
		return {"error": str(e)}


@app.post("/generate/image")
def generate_image(req: ImageReq) -> dict:
	out_dir = Path(os.environ.get("OMNICODER_OUT_ROOT", "weights")) / "api_out" / "images"
	out_path = out_dir / "image_out.png"
	img_path = _generate_image(req.prompt, req.width, req.height, req.backend, req.device, req.sd_model, req.sd_local_path, out_path)
	if bool(req.return_base64):
		with open(img_path, "rb") as f:
			b64 = base64.b64encode(f.read()).decode("ascii")
		return {"image_base64": b64}
	return {"image_path": str(img_path)}


@app.post("/generate/video")
def generate_video(req: VideoReq) -> dict:
	out_dir = Path(os.environ.get("OMNICODER_OUT_ROOT", "weights")) / "api_out" / "video"
	out_path = out_dir / "video_out.mp4"
	vp = _generate_video(req.prompt, req.frames, req.steps, req.device, req.backend, req.video_model, req.video_local_path, out_path)
	if vp is None:
		return {"error": "Video backend not available. Install video pipeline or provide local model."}
	return {"video_path": str(vp)}


# ---- Unified multimodal endpoint ----
@app.post("/generate/unified")
def generate_unified(req: UnifiedReq) -> dict:
	try:
		from omnicoder.inference.unified import run_unified  # type: ignore
		preset = os.environ.get("OMNICODER_API_PRESET", os.environ.get("OMNICODER_PRESET", "mobile_4gb"))
		device = os.environ.get("OMNICODER_TRAIN_DEVICE", "cuda")
		inputs = {"prompt": req.prompt, "max_new_tokens": int(req.max_new_tokens)}
		if req.image:
			inputs["image"] = req.image
		if req.video:
			inputs["video"] = req.video
		if req.audio:
			inputs["audio"] = req.audio
		if req.target:
			inputs["target"] = req.target
		res = run_unified(inputs, preset=preset, device=device, ckpt=os.environ.get("OMNICODER_API_CKPT", ""))
		return res
	except Exception as e:
		try:
			_LOG.error("/generate/unified failed: %s", str(e))
		except Exception:
			pass
		return {"error": str(e)}


def main() -> None:
	import uvicorn  # type: ignore
	host = os.environ.get("SERVER_HOST", "0.0.0.0")
	port = int(os.environ.get("SERVER_PORT", "8000"))
	try:
		_LOG.info("api: starting uvicorn host=%s port=%d", host, port)
	except Exception:
		pass
	uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
	main()


