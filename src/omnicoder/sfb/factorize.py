from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import re
from omnicoder.utils.logger import get_logger
try:
	from omnicoder.utils.perf import add as _perf_add  # type: ignore
except Exception:  # pragma: no cover
	def _perf_add(name: str, dt: float) -> None:  # type: ignore
		return
_AMR_STOG: Any | None = None
_SRL_PRED: Any | None = None
_GOAL_HEAD: Any | None = None
_AMR_TRIED: bool = False
_SRL_TRIED: bool = False


def _load_amr_stog() -> Optional[Any]:
	global _AMR_STOG
	global _AMR_TRIED
	if _AMR_STOG is not None:
		return _AMR_STOG
	try:
		# Avoid retrying heavy imports repeatedly during generation unless explicitly requested
		if _AMR_TRIED and os.getenv("SFB_HEAVY_RETRY", os.getenv("SFB_RETRY", "0")) != "1":
			return None
		# Default to offline to avoid any network IO during generation
		allow_net = os.getenv("SFB_ALLOW_NET", "0") == "1"
		model_path = os.getenv("SFB_AMR_MODEL_PATH", "").strip()
		# If SFB_VENV is set, extend sys.path with its site-packages to find amrlib
		try:
			import sys as _sys
			venv_path = os.getenv("SFB_VENV", "").strip()
			if venv_path and os.path.isdir(venv_path):
				pyver = f"python{_sys.version_info.major}.{_sys.version_info.minor}"
				for p in (f"{venv_path}/lib/{pyver}/site-packages", f"{venv_path}/lib/site-packages"):
					if os.path.isdir(p) and p not in _sys.path:
						_sys.path.insert(0, p)
		except Exception:
			pass
		# Some amrlib builds for recent Python may not expose download_stog; handle gracefully
		import amrlib  # type: ignore
		if model_path:
			_AMR_STOG = amrlib.load_stog_model(model_path)
		else:
			# Only allow default download when explicitly permitted
			if not allow_net:
				return None
			try:
				_AMR_STOG = amrlib.load_stog_model()
			except Exception:
				# Try to download model weights into the venv and load again, but tolerate version differences
				try:
					try:
						from amrlib.downloads import download_stog  # type: ignore
						download_stog()
					except Exception:
						if hasattr(amrlib, 'download_stog'):
							amrlib.download_stog()  # type: ignore[attr-defined]
					_AMR_STOG = amrlib.load_stog_model()
				except Exception:
					_AMR_STOG = None
		get_logger("omnicoder.sfb").info("AMR stog load path=%s allow_net=%s", ("custom" if model_path else "default"), bool(allow_net))
		return _AMR_STOG
	except Exception as e:
		if not _AMR_TRIED:
			get_logger("omnicoder.sfb").warning("AMR stog load failed (will not retry this session unless SFB_HEAVY_RETRY=1): %s", e)
		else:
			get_logger("omnicoder.sfb").debug("AMR stog load failed: %s", e)
		_AMR_STOG = None
		_AMR_TRIED = True
		return None


def _load_srl_predictor() -> Optional[Any]:
	global _SRL_PRED
	global _SRL_TRIED
	if _SRL_PRED is not None:
		return _SRL_PRED
	try:
		# Avoid retrying heavy imports repeatedly during generation unless explicitly requested
		if _SRL_TRIED and os.getenv("SFB_HEAVY_RETRY", os.getenv("SFB_RETRY", "0")) != "1":
			return None
		# 1) Try AllenNLP SRL when available (attempt from isolated venv when provided)
		try:
			allow_net = os.getenv("SFB_ALLOW_NET", "0") == "1"
			# If SFB_VENV is set, try to import from its site-packages
			venv_path = os.getenv("SFB_VENV", "").strip()
			if venv_path and os.path.isdir(venv_path):
				import sys as _sys
				pyver = f"python{_sys.version_info.major}.{_sys.version_info.minor}"
				cands = [
					f"{venv_path}/lib/{pyver}/site-packages",
					f"{venv_path}/lib/site-packages",
				]
				for p in cands:
					if os.path.isdir(p) and p not in _sys.path:
						_sys.path.insert(0, p)
			model_path = os.getenv("SFB_SRL_MODEL_PATH", "").strip()
			if not model_path:
				try:
					bundled = os.path.join("weights", "sfb", "srl", "model.tar.gz")
					if os.path.exists(bundled):
						model_path = bundled
				except Exception:
					model_path = ""
			from allennlp.predictors.predictor import Predictor  # type: ignore
			import allennlp_models.tagging  # type: ignore  # noqa: F401
			if model_path:
				_SRL_PRED = Predictor.from_path(model_path)
			else:
				if not allow_net:
					raise RuntimeError("allennlp srl requires network for default model")
				_SRL_PRED = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-2020.12.15.tar.gz")
			get_logger("omnicoder.sfb").info("SRL predictor load path=%s allow_net=%s", ("bundled" if model_path else "remote"), bool(allow_net))
			return _SRL_PRED
		except Exception as _aln_e:
			get_logger("omnicoder.sfb").info("SRL allennlp backend unavailable: %s", _aln_e)
		# 2) Fallback: SpaCy-based SRL-lite (dependency parse → predicate-argument spans)
		try:
			import spacy  # type: ignore
			try:
				nlp = spacy.load(os.getenv("SFB_SPACY_MODEL", "en_core_web_sm"))
			except Exception:
				nlp = spacy.blank("en")
				try:
					nlp.add_pipe("tok2vec")
				except Exception:
					pass
			class _SpaCySRLPredictor:
				def __init__(self, _nlp):
					self._nlp = _nlp
				def predict(self, sentence: str) -> Dict[str, Any]:
					doc = self._nlp(sentence)
					words = [t.text for t in doc]
					verbs: List[Dict[str, Any]] = []
					for t in doc:
						if t.pos_ == "VERB":
							# Collect simple roles via dependency labels
							tags = ["O"] * len(doc)
							# Mark verb itself as B-V
							try:
								v_i = t.i
								tags[v_i] = "B-V"
							except Exception:
								pass
							for child in t.children:
								label = child.dep_.upper()
								role = None
								if label in ("NSUBJ", "CSUBJ"):
									role = "ARG0"
								elif label in ("DOBJ", "OBJ"):  # direct object
									role = "ARG1"
								elif label in ("IOBJ",):
									role = "ARG2"
								elif label in ("OBL", "ADVCL", "ADVMOD", "AUX"):
									role = "ARGM"
								if role is not None:
									idx = child.i
									tags[idx] = f"B-{role}"
							verbs.append({"verb": t.lemma_, "tags": tags})
					return {"words": words, "verbs": verbs}
			_SRL_PRED = _SpaCySRLPredictor(nlp)
			get_logger("omnicoder.sfb").info("SRL predictor load path=spacy-lite allow_net=%s", False)
			return _SRL_PRED
		except Exception as _sp_e:
			get_logger("omnicoder.sfb").warning("SpaCy SRL-lite unavailable: %s", _sp_e)
	except Exception as e:
		if not _SRL_TRIED:
			get_logger("omnicoder.sfb").warning("SRL predictor load failed (will not retry this session unless SFB_HEAVY_RETRY=1): %s", e)
		else:
			get_logger("omnicoder.sfb").debug("SRL predictor load failed: %s", e)
		_SRL_PRED = None
		_SRL_TRIED = True
		return None


def _load_goal_head(goals: List[str]) -> Optional[Any]:
	global _GOAL_HEAD
	if _GOAL_HEAD is not None:
		return _GOAL_HEAD
	try:
		from .goal_prior import GoalPriorHead  # local, light
		path = os.getenv('SFB_GOAL_HEAD_PATH', '').strip()
		if path and os.path.exists(path):
			_GOAL_HEAD = GoalPriorHead.load(path)
			return _GOAL_HEAD
		# Initialize fresh head with default goals
		_GOAL_HEAD = GoalPriorHead(goals=goals)
		# Optional training sidecar
		train_path = os.getenv('SFB_GOAL_HEAD_TRAIN', '').strip()
		if train_path and os.path.exists(train_path):
			_GOAL_HEAD.train_from_jsonl(train_path, epochs=int(os.getenv('SFB_GOAL_HEAD_EPOCHS', '3')))
			save_path = os.getenv('SFB_GOAL_HEAD_SAVE', '').strip()
			if save_path:
				_GOAL_HEAD.save(save_path)
		return _GOAL_HEAD
	except Exception as e:
		get_logger("omnicoder.sfb").debug("goal head setup failed: %s", e)
		_GOAL_HEAD = None
		return None


@dataclass
class Factor:
	name: str
	scope: Tuple[str, ...]
	meta: Dict[str, Any]


@dataclass
class FactorizationResult:
	text: str
	factors: List[Factor]
	goal_priors: Dict[str, float]


def _goal_priors_heuristic(prompt: str) -> Dict[str, float]:
	# Very lightweight heuristic placeholder; a trained RSA head can replace this
	priors: Dict[str, float] = {}
	lower = prompt.lower()
	if any(k in lower for k in ("sql", "database", "join")):
		priors["sql"] = 0.7
	if any(k in lower for k in ("image", "photo", "picture", "detect")):
		priors["vqa"] = max(priors.get("vqa", 0.0), 0.6)
	if any(k in lower for k in ("audio", "transcribe", "wer")):
		priors["asr"] = 0.6
	if any(k in lower for k in ("code", "python", "compile", "unit test")):
		priors["code"] = 0.7
	# Optional override to seed goal priors from env string like "code:0.8,vqa:0.3"
	try:
		gp = os.getenv("SFB_GOAL_PRIOR", "").strip()
		if gp:
			for kv in gp.split(','):
				if not kv.strip():
					continue
				if ':' in kv:
					k, v = kv.split(':', 1)
					k = k.strip().lower()
					try:
						priors[k] = float(v)
					except Exception:
						continue
	except Exception:
		pass
	return priors


def factorize_prompt(prompt: str) -> FactorizationResult:
	modes = os.getenv("SFB_FACTORIZER", "amr,srl").split(",")
	modes = [m.strip().lower() for m in modes if m.strip()]
	factors: List[Factor] = []
	text = prompt
	try:
		get_logger("omnicoder.sfb").debug("factorize enter modes=%s len=%s", modes, len(prompt))
	except Exception:
		pass
	# AMR backend (optional via amrlib/penman). Normalize roles/predicates robustly when available.
	if "amr" in modes:
		try:
			import time as _t
			_t0 = _t.perf_counter()
			triples: List[Tuple[str,str,str]] = []
			try:
				stog = _load_amr_stog()
				if stog is None:
					raise RuntimeError("amr stog unavailable")
				g = stog.parse_sents([prompt])[0]
				import penman  # type: ignore
				g_obj = penman.decode(g)
				# Normalize: collapse :instance to 'instance', lowercase roles, stringify target
				triples = []
				for t in g_obj.triples():  # type: ignore[attr-defined]
					role = str(getattr(t, 'role', '')).lower()
					if role == ':instance':
						role = 'instance'
					triples.append((str(getattr(t, 'source', '')), role, str(getattr(t, 'target', ''))))
			except Exception as e:
				# Fallback: shallow predicate-argument mining via regex on verbs and prepositions
				# This is a weak proxy to keep factors typed without heavy deps
				get_logger("omnicoder.sfb").debug("AMR fallback regex path: %s", e)
				lower_p = prompt.lower()
				# Extract main verbs (very coarse)
				verbs = re.findall(r"\b([a-z]{3,})\b", lower_p)[:16]
				for v in verbs:
					triples.append(("event", "predicate", v))
				# Extract copula and possession patterns: X is Y, X are Y, X has Y, X have Y
				for m in re.finditer(r"\b([a-z][a-z0-9_]{1,24})\s+(is|are|has|have)\s+([a-z][a-z0-9_]{1,24})\b", lower_p):
					triples.append((m.group(1), m.group(2), m.group(3)))
				# Prepositional relations: X of Y, X for Y, X to Y
				for m in re.finditer(r"\b([a-z][a-z0-9_]{1,24})\s+(of|for|to|in|on|with|from)\s+([a-z][a-z0-9_]{1,24})\b", lower_p):
					triples.append((m.group(1), m.group(2), m.group(3)))
			factors.append(Factor(name="predicate_structure", scope=("text",), meta={"mode": "amr", "type": "semantic", "triples": triples}))
			try:
				dt = float(_t.perf_counter() - _t0)
				_perf_add('sfb.amr', dt)
				if dt > 1.0:
					get_logger("omnicoder.sfb").info("factorize.amr.dt=%.6f", dt)
			except Exception:
				pass
		except Exception as e:
			get_logger("omnicoder.sfb").debug("AMR factor construction failed: %s", e)
			factors.append(Factor(name="predicate_structure", scope=("text",), meta={"mode": "amr", "type": "semantic"}))
	# SRL backend (optional via allennlp). Normalize predicate-arguments when available.
	if "srl" in modes:
		try:
			import time as _t
			_t0 = _t.perf_counter()
			roles: List[Dict[str, Any]] = []
			try:
				_srl = _load_srl_predictor()
				if _srl is None:
					raise RuntimeError("srl predictor unavailable")
				pred = _srl.predict(sentence=prompt)
				words = pred.get("words", []) or []
				for verb in pred.get("verbs", [])[:4]:
					v = str(verb.get("verb", "")).strip()
					tags = list(verb.get("tags", []))
					# Hardened BIO→spans conversion aligned to words when available
					args: Dict[str, Any] = {}
					cur = None; start = -1
					for i, tag in enumerate(tags):
						if tag.startswith("B-"):
							if cur is not None and start >= 0:
								span_tokens = words[start:i] if words and len(words) == len(tags) else prompt.split()[start:i]
								if span_tokens:
									args[cur] = " ".join(span_tokens)
							cur = tag[2:]
							start = i
						elif tag.startswith("I-"):
							# continue current span
							pass
						else:
							if cur is not None and start >= 0:
								span_tokens = words[start:i] if words and len(words) == len(tags) else prompt.split()[start:i]
								if span_tokens:
									args[cur] = " ".join(span_tokens)
							cur, start = None, -1
					if cur is not None and start >= 0:
						span_tokens = words[start:len(tags)] if words and len(words) == len(tags) else prompt.split()[start:len(tags)]
						if span_tokens:
							args[cur] = " ".join(span_tokens)
					roles.append({"verb": v, "args": args, "tags": tags})
			except Exception as e:
				# Fallback: tiny on-device SRL heuristic (distilled proxy): POS-like regexes
				get_logger("omnicoder.sfb").debug("SRL fallback heuristic path: %s", e)
				lower_p = prompt.lower()
				roles = []
				# Subject-verb-object heuristic
				s_vo = re.findall(r"\b(we|you|they|he|she|it|i|user|system|model|agent|\w{3,})\s+([a-z]{3,})\s+(?:a|an|the)?\s*(\w{3,})\b", lower_p)
				for (subj, v, obj) in s_vo[:4]:
					roles.append({"verb": v, "args": {"ARG0": subj, "ARG1": obj}})
				# Quantities
				nums = re.findall(r"[-+]?\d+(?:\.\d+)?", prompt)
				if nums:
					roles.append({"verb": "have", "args": {"ARG0": "user", "ARG2": nums[:4]}})
			factors.append(Factor(name="semantic_roles", scope=("text",), meta={"mode": "srl", "type": "semantic", "roles": roles}))
			try:
				dt = float(_t.perf_counter() - _t0)
				_perf_add('sfb.srl', dt)
				if dt > 1.0:
					get_logger("omnicoder.sfb").info("factorize.srl.dt=%.6f", dt)
			except Exception:
				pass
		except Exception as e:
			get_logger("omnicoder.sfb").debug("SRL factor construction failed: %s", e)
			factors.append(Factor(name="semantic_roles", scope=("text",), meta={"mode": "srl", "type": "semantic"}))
	lower = prompt.lower()
	if any(k in lower for k in ("sum", "add", "multiply", "div", "equals", "=", "+", "-", "*", "/")):
		# Try to surface a candidate expression for downstream exact solver
		expr = None
		try:
			m = re.search(r"([\d\s\+\-\*\/\(\)\.]+)=?", prompt)
			expr = m.group(1).strip() if m else None
		except Exception:
			expr = None
		meta = {"type": "numeric"}
		if expr:
			meta["expr"] = expr
		factors.append(Factor(name="numeric_reasoning", scope=("text",), meta=meta))
	# Lightweight logic factor: detect boolean constraints or implications
	if any(k in lower for k in (" if ", " then ", "=>", " and ", " or ", "not ", ">", "<", "==", "!=", ">=", "<=")):
		logic_expr = None
		try:
			# Heuristic: extract content inside backticks or after keywords
			m = re.search(r"`([^`]+)`", prompt)
			if m:
				logic_expr = m.group(1).strip()
			else:
				# Fallback: take a short window containing operators
				m2 = re.search(r"([\w\s\(\)\!\=\<\>\&\|\:]+)", prompt)
				logic_expr = m2.group(1).strip() if m2 else None
		except Exception:
			logic_expr = None
		meta = {"type": "logic"}
		if logic_expr:
			meta["expr"] = logic_expr
		factors.append(Factor(name="logic_constraints", scope=("text",), meta=meta))
	if any(k in lower for k in ("code", "python", "function", "class", "compile", "unit test")):
		factors.append(Factor(name="code_generation", scope=("text",), meta={"type": "code"}))
	if any(k in lower for k in ("image", "photo", "picture", "detect", "segment")):
		factors.append(Factor(name="vision_grounding", scope=("image","text"), meta={"type": "vision", "desc": prompt}))
	if any(k in lower for k in ("audio", "speech", "transcribe", "wer")):
		factors.append(Factor(name="audio_transcription", scope=("audio","text"), meta={"type": "audio", "desc": prompt}))
	if any(k in lower for k in ("video", "clip", "frames", "temporal")):
		factors.append(Factor(name="video_temporal", scope=("video","text"), meta={"type": "video", "desc": prompt}))
	# Retrieval context factor
	try:
		enable_ret = (os.getenv('SFB_INFER_RETRIEVAL', '1') == '1')
	except Exception:
		enable_ret = True
	if enable_ret:
		try:
			import time as _t
			_t0 = _t.perf_counter()
			head = lower[:1024]
			markers = 0
			for kw in ("retrieved", "context:", "knowledge:", "documents:", "sources:"):
				if kw in head:
					markers += 1
			import re as _re
			urls = _re.findall(r"https?://\S+", prompt[:2048])
			if markers >= 1 or len(urls) >= 2:
				factors.append(Factor(name="retrieval_context", scope=("text",), meta={"type": "retrieval", "markers": int(markers), "urls": int(len(urls))}))
			try:
				dt = float(_t.perf_counter() - _t0)
				_perf_add('sfb.heuristic', dt)
			except Exception:
				pass
		except Exception as e:
			get_logger("omnicoder.sfb").debug("retrieval factor detection failed: %s", e)
	priors = _goal_priors_heuristic(prompt)
	# Blend trainable goal head if available and enabled via mode
	try:
		mode = os.getenv('SFB_GOAL_PRIOR', '').strip().lower()
		use_head = True
		# Interpret common modes: 'rsa' (default head), 'none' (disable), numeric map handled above
		if mode in ('none', 'off', '0'):
			use_head = False
		# If the string looks like manual priors (contains ':'), keep head enabled (may still help)
		if use_head:
			head = _load_goal_head(goals=["code","vqa","asr","sql","video"])
			if head is not None:
				pred = head.predict(prompt)
				for k, v in pred.items():
					# Max blend: learned overrides heuristic when confident
					priors[k] = max(priors.get(k, 0.0), float(v))
	except Exception as e:
		get_logger("omnicoder.sfb").debug("goal head blend skipped: %s", e)
	return FactorizationResult(text=text, factors=factors, goal_priors=priors)


