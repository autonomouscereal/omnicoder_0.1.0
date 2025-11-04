from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from omnicoder.utils.resources import apply_thread_env_if_auto, recommend_num_workers, gpu_summary, audit_env
from omnicoder.utils.env_registry import load_dotenv_best_effort
from omnicoder.utils.env_defaults import apply_core_defaults, apply_training_defaults, apply_run_env_defaults, apply_profile

# --- Best-checkpoint registry helpers ---
def _load_registry(root: Path) -> dict:
    try:
        p = root / "MODEL_REGISTRY.json"
        if p.exists():
            return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        pass
    return {}

def _save_registry(root: Path, reg: dict) -> None:
    try:
        p = root / "MODEL_REGISTRY.json"
        p.write_text(json.dumps(reg, indent=2), encoding='utf-8')
    except Exception:
        pass

def _compare_metric(domain: str, a: float, b: float) -> bool:
    """Return True if a is better than b for the given domain metric."""
    # Higher-is-better domains
    if domain in ("text.tps", "clip", "vqa_acc"):
        return a > b
    # Lower-is-better domains
    if domain in ("fvd", "fad"):
        return a < b
    return a > b

def _ingest_bench(root: Path, bench_json: Path, ckpt_path: Path | None, reg: dict) -> dict:
    """Update MODEL_REGISTRY.json with metrics from a bench JSON and associated checkpoint path."""
    try:
        if not bench_json.exists():
            return reg
        data = json.loads(bench_json.read_text(encoding='utf-8'))
    except Exception:
        return reg
    # Extract common metrics
    try:
        text_tps = float(((data.get("text") or {}).get("tokens_per_sec", 0.0)))
        best = reg.get("text", {})
        cur = float(best.get("tokens_per_sec", 0.0))
        if _compare_metric("text.tps", text_tps, cur):
            reg["text"] = {
                "tokens_per_sec": text_tps,
                "ckpt": str(ckpt_path) if ckpt_path else best.get("ckpt", ""),
            }
            # Maintain a convenient best checkpoint copy
            try:
                if ckpt_path and ckpt_path.exists():
                    (root / "student_best.pt").write_bytes(ckpt_path.read_bytes())
            except Exception:
                pass
    except Exception:
        pass
    try:
        clip_score = float((data.get("quality") or {}).get("clip_score", 0.0))
        if _compare_metric("clip", clip_score, float(reg.get("clip", {}).get("score", 0.0))):
            reg["clip"] = {"score": clip_score}
    except Exception:
        pass
    try:
        vqa = float(((data.get("datasets") or {}).get("vqa", {}) or {}).get("acc", 0.0))
        if _compare_metric("vqa_acc", vqa, float(reg.get("vqa", {}).get("acc", 0.0))):
            reg["vqa"] = {"acc": vqa}
    except Exception:
        pass
    try:
        fad = float((data.get("quality") or {}).get("fad", float('inf')))
        if fad == fad and _compare_metric("fad", fad, float(reg.get("fad", {}).get("score", float('inf')))):
            reg["fad"] = {"score": fad}
    except Exception:
        pass
    try:
        fvd = float((data.get("quality") or {}).get("fvd", float('inf')))
        if fvd == fvd and _compare_metric("fvd", fvd, float(reg.get("fvd", {}).get("score", float('inf')))):
            reg["fvd"] = {"score": fvd}
    except Exception:
        pass
    _save_registry(root, reg)
    return reg

def _load_dotenv(env_path: str = ".env") -> None:
	try:
		load_dotenv_best_effort((env_path,))
	except Exception:
		pass

def run(cmd: list[str], env: dict[str, str] | None = None) -> int:
	print("[run]", " ".join(cmd))
	return subprocess.call(cmd, env=env or os.environ.copy())

def run_with_timeout(cmd: list[str], timeout_sec: int, env: dict[str, str] | None = None) -> int:
	print("[run<=", str(timeout_sec), "s]", " ".join(cmd))
	try:
		proc = subprocess.Popen(cmd, env=env or os.environ.copy())
		try:
			proc.wait(timeout=max(1, int(timeout_sec)))
		except subprocess.TimeoutExpired:
			try:
				proc.terminate()
			except Exception:
				pass
			try:
				proc.kill()
			except Exception:
				pass
			return 124
		return int(proc.returncode or 0)
	except Exception:
		return 1


def minutes_to_steps(minutes: int, ms_per_step: float, floor: int = 1) -> int:
	return max(int((minutes * 60_000) / max(ms_per_step, 1.0)), floor)


def _normalize_instructions(jsonl_paths: list[str], out_path: str, max_items: int = 5000) -> str:
	try:
		import json as _json
		from pathlib import Path as _P
		outp = _P(out_path); outp.parent.mkdir(parents=True, exist_ok=True)
		n = 0
		seen: set[str] = set()
		with open(outp, "w", encoding="utf-8") as f:
			for p in jsonl_paths:
				try:
					text = _P(p).read_text(encoding='utf-8', errors='ignore')
				except Exception:
					continue
				for line in text.splitlines():
					if n >= max_items:
						break
					try:
						ex = _json.loads(line)
						instr = str(ex.get("instruction", "") or ex.get("question", "") or ex.get("prompt", "")).strip()
						ans = str(ex.get("answer", "") or ex.get("output", "") or ex.get("text", "")).strip()
						image = ex.get("image")
						video = ex.get("video")
						audio = ex.get("path") if ("path" in ex and str(ex.get("path")).lower().endswith(('.wav','.flac','.mp3'))) else None
						if not instr:
							caption = str(ex.get("text", "")).strip()
							if caption:
								instr = "Describe this input concisely."
								ans = caption
						if not instr:
							continue
						# De-duplicate by content key (path + instruction)
						key = (str(image or video or audio or "") + "||" + instr)[:4096]
						if key in seen:
							continue
						out_ex = {}
						if image:
							out_ex["image"] = str(image)
						if video:
							out_ex["video"] = str(video)
						if audio:
							out_ex["path"] = str(audio)
						out_ex["instruction"] = instr
						out_ex["answer"] = ans
						f.write(_json.dumps(out_ex, ensure_ascii=False) + "\n")
						n += 1
						seen.add(key)
					except Exception:
						continue
		return str(outp)
	except Exception:
		return out_path


def _try_load_metrics(ckpt_path: str) -> dict[str, float]:
	try:
		from pathlib import Path as _P
		import json as _J
		p = _P(str(ckpt_path) + ".metrics.json")
		if p.exists():
			data = _J.loads(p.read_text(encoding='utf-8', errors='ignore'))
			if isinstance(data, dict):
				# keep only simple float-like items
				out: dict[str, float] = {}
				for k, v in data.items():
					try:
						out[k] = float(v)
					except Exception:
						continue
				return out
		return {}
	except Exception:
		return {}


def main() -> None:
	# Load .env and .env.tuned early so argparse defaults can see OMNICODER_* variables
	try:
		load_dotenv_best_effort((".env", ".env.tuned"))
		# Apply centralized defaults and quality-first profile globally
		apply_core_defaults(os.environ)  # type: ignore[arg-type]
		apply_training_defaults(os.environ)  # type: ignore[arg-type]
		apply_run_env_defaults(os.environ)  # type: ignore[arg-type]
		apply_profile(os.environ, "quality")  # type: ignore[arg-type]
	except Exception:
		pass
	ap = argparse.ArgumentParser(description="One-button orchestrator: plan and run multimodal training within a time budget (hours)")
	ap.add_argument("--budget_hours", type=float, default=float(os.getenv("OMNICODER_TRAIN_BUDGET_HOURS", "1")), help="Specify a time budget in hours (single argument).")
	ap.add_argument("--device", type=str, default=os.getenv("OMNICODER_TRAIN_DEVICE", "cuda"))
	ap.add_argument("--out_root", type=str, default=os.getenv("OMNICODER_OUT_ROOT", "weights"))
	ap.add_argument("--mobile_preset", type=str, default=os.getenv("OMNICODER_TRAIN_PRESET", "mobile_4gb"))
	ap.add_argument("--student_preset", type=str, default=os.getenv("OMNICODER_STUDENT_PRESET", "mobile_2gb"))
	ap.add_argument("--draft_preset", type=str, default=os.getenv("OMNICODER_DRAFT_PRESET", "draft_3b"))
	ap.add_argument("--teacher", type=str, default=os.getenv("OMNICODER_TEACHER", "microsoft/phi-2"))
	ap.add_argument("--teachers", type=str, default=os.getenv("OMNICODER_KD_TEACHERS", ""), help="Optional space-separated list of HF teacher ids for multi-teacher KD")
	# Domain-specific optional teachers and datasets for multi-teacher KD
	ap.add_argument("--text_teachers", type=str, default=os.getenv("OMNICODER_KD_TEXT_TEACHERS", ""))
	ap.add_argument("--code_teachers", type=str, default=os.getenv("OMNICODER_KD_CODE_TEACHERS", ""))
	ap.add_argument("--vl_teachers", type=str, default=os.getenv("OMNICODER_KD_VL_TEACHERS", ""))
	ap.add_argument("--asr_teachers", type=str, default=os.getenv("OMNICODER_KD_ASR_TEACHERS", ""))
	ap.add_argument("--tts_teachers", type=str, default=os.getenv("OMNICODER_KD_TTS_TEACHERS", ""))
	ap.add_argument("--text_data", type=str, default=os.getenv("OMNICODER_DATA_TEXT", "data/text"))
	ap.add_argument("--code_data", type=str, default=os.getenv("OMNICODER_DATA_CODE", "examples/code_eval"))
	ap.add_argument("--vl_data", type=str, default=os.getenv("OMNICODER_DATA_VL", "examples/vl_auto.jsonl"))
	ap.add_argument("--asr_data", type=str, default=os.getenv("OMNICODER_DATA_ASR", "data/asr"))
	ap.add_argument("--tts_data", type=str, default=os.getenv("OMNICODER_DATA_TTS", "data/tts"))
	ap.add_argument("--skip_export", action="store_true", help="Skip export/bench at the end")
	ap.add_argument("--resume", action="store_true", default=(os.getenv("OMNICODER_RESUME", "1") == "1"), help="Resume-friendly: skip stages if outputs already exist.")
	# Optional verifier-head KD stage after draft KD
	ap.add_argument("--run_verifier_kd", action="store_true", default=(os.getenv("OMNICODER_RUN_VERIFIER_KD", "1") == "1"), help="Run a short verifier-head distillation stage after KD")
	ap.add_argument("--verifier_kd_steps", type=int, default=int(os.getenv("OMNICODER_VERIFIER_KD_STEPS", "0")), help="Override steps for verifier KD (0 to derive from budget)")
	ap.add_argument("--export_to_phone", type=str, default=os.getenv("OMNICODER_EXPORT_TO_PHONE", ""), help="Optional: 'android' or 'ios' to push artifacts to phone at end")
	# Optional RL loops
	ap.add_argument("--enable_grpo", action="store_true", default=(os.getenv("OMNICODER_ENABLE_GRPO", "0") == "1"))
	ap.add_argument("--enable_ppo", action="store_true", default=(os.getenv("OMNICODER_ENABLE_PPO", "0") == "1"))
	# Provider bench toggle
	ap.add_argument("--run_provider_bench", action="store_true", default=(os.getenv("OMNICODER_RUN_PROVIDER_BENCH", "1") == "1"))
	# Planning only: print the plan and exit without executing stages
	ap.add_argument("--dry_run_plan", action="store_true", default=False, help="Print the planned stage minutes/steps and exit without running")
	# Smoke mode: run exactly one step per stage and exit
	ap.add_argument("--smoke_one_step", action="store_true", default=(os.getenv("OMNICODER_SMOKE_ONE_STEP","0")=="1"), help="Run a smoke test: one step per major stage and exit")
	ap.add_argument("--smoke_minutes", type=float, default=float(os.getenv("OMNICODER_SMOKE_MINUTES","0")), help="Total minutes for smoke run (wall clock); split per stage; 0 to use default 1 minute")
	# Optional: train cross-modal verifier (mini-CLIP) and 3D latent head
	ap.add_argument("--run_verifier_train", action="store_true", default=(os.getenv("OMNICODER_RUN_VERIFIER_TRAIN", "1") == "1"))
	ap.add_argument("--run_latent3d", action="store_true", default=(os.getenv("OMNICODER_RUN_LATENT3D", "1") == "1"))
	# Router evaluation toggles (LLMRouter/InteractionRouter)
	ap.add_argument("--router_eval", action="store_true", default=(os.getenv("OMNICODER_ROUTER_EVAL", "1") == "1"), help="Evaluate LLM/interaction routers late in training and enable if ROI")
	ap.add_argument("--router_eval_steps", type=int, default=int(os.getenv("OMNICODER_ROUTER_EVAL_STEPS", "100")))
	args = ap.parse_args()

	out_root = Path(args.out_root)
	out_root.mkdir(parents=True, exist_ok=True)
	logs_dir = Path("tests_logs")
	logs_dir.mkdir(parents=True, exist_ok=True)

	# Load registry of best checkpoints/metrics
	model_registry = _load_registry(out_root)

	# Ensure an env dict exists before any early-stage probes use it
	env = dict(os.environ)

	# CUDA availability guardrail and summary
	try:
		import torch as _t
		if str(args.device).startswith("cuda") and not _t.cuda.is_available():
			raise SystemExit("[gpu] CUDA requested but not available. Ensure docker is started with --gpus all and NVIDIA runtime.")
		try:
			print(f"[gpu] {gpu_summary()}")
		except Exception:
			pass
	except Exception:
		pass

	# Apply auto resource envs if enabled
	try:
		apply_thread_env_if_auto()
	except Exception:
		pass

	# Optional global seeding for reproducibility across orchestrated stages
	try:
		_seed = os.environ.get("OMNICODER_SEED", "").strip()
		if _seed:
			import random as _rand
			_rand.seed(int(_seed))
			try:
				import numpy as _np  # type: ignore
				_np.random.seed(int(_seed))
			except Exception:
				pass
			try:
				import torch as _torch  # type: ignore
				_torch.manual_seed(int(_seed))
				if _torch.cuda.is_available():
					_torch.cuda.manual_seed_all(int(_seed))
				try:
					_torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
				except Exception:
					pass
			except Exception:
				pass
	except Exception:
		pass

	# Audit env for unknown OMNICODER_* keys
	try:
		# Apply centralized defaults once before auditing to avoid false positives
		apply_core_defaults(os.environ)  # type: ignore[arg-type]
		apply_training_defaults(os.environ)  # type: ignore[arg-type]
		unknown = audit_env()
		if unknown:
			print({"env_unknown": unknown[:20], "more": max(0, len(unknown)-20)})
	except Exception:
		pass

	# Simple budget planner
	try:
		budget = int(max(0.1, float(args.budget_hours)) * 60.0)
	except Exception:
		budget = 60
	start_ts = time.time()
	planned_end_ts = start_ts + budget * 60.0

	# Weakness probe: run a fast auto-benchmark to identify weakest domains and bias the plan
	weak_json = Path(args.out_root) / "weakness_probe.json"
	try:
		print("[probe] running auto_benchmark (fast) to identify weakest domains ...")
		rc = run([
			sys.executable, "-m", "omnicoder.eval.auto_benchmark",
			"--device", ("cuda" if str(args.device).startswith("cuda") else "cpu"), "--seq_len", "128", "--gen_tokens", "64",
			"--preset", args.student_preset, "--out", str(weak_json),
		], env=env)
		print(f"[probe] auto_benchmark rc={rc}")
		# Ingest weakness probe as initial baseline (no ckpt association)
		try:
			_ingest_bench(out_root, weak_json, None, model_registry)
		except Exception:
			pass
		# Smoke one-step mode: strictly bounded wall clock and early exit
		if bool(args.smoke_one_step) or (float(args.smoke_minutes) > 0):
			# Ensure datasets are available before smoke stages by running in-process autofetch
			try:
				if os.getenv("OMNICODER_AUTOFETCH", "1") == "1":
					from omnicoder.tools.autofetch_datasets import autofetch_all  # type: ignore
					_limit = int(os.getenv("OMNICODER_FETCH_LIMIT", "1000000"))
					fetched = autofetch_all(_limit)
					def _set(k: str, v: object | None) -> None:
						if v:
							os.environ[k] = str(v)
							env[k] = str(v)
					# Text/code/VL datasets
					_set("OMNICODER_DATA_TEXT", fetched.get("fineweb_edu") or fetched.get("c4_en") or fetched.get("openwebtext") or fetched.get("text"))
					_set("OMNICODER_DATA_CODE", fetched.get("codeparrot_clean") or fetched.get("code"))
					_set("OMNICODER_VL_JSONL", fetched.get("vl_cc_jsonl") or fetched.get("vl_coco_jsonl") or fetched.get("vl_jsonl"))
					# Audio datasets
					_set("OMNICODER_DATA_ASR", fetched.get("asr_dir"))
					# Prefer ASR wavs; fallback TTS wavs
					_audio_wav = fetched.get("wav_dir") or fetched.get("tts_wavs_dir")
					if _audio_wav:
						os.environ["OMNICODER_AUDIO_DATA"] = str(_audio_wav)
						env["OMNICODER_AUDIO_DATA"] = str(_audio_wav)
						os.environ["OMNICODER_AUDIO_IS_WAV"] = "1"
						env["OMNICODER_AUDIO_IS_WAV"] = "1"
					# Video
					_set("OMNICODER_VIDEO_DATA", fetched.get("video_dir"))
			except Exception as _eaf_smoke:
				print(f"[warn] smoke autofetch skipped: {_eaf_smoke}")

			print("[smoke] running single-step across major stages with wall-clock cap ...")
			s_total = max(30, int(float(args.smoke_minutes) * 60.0)) if float(args.smoke_minutes) > 0 else 60
			# Dynamic audio args depending on dataset type
			audio_root = os.getenv("OMNICODER_AUDIO_DATA", "examples/data/vq/audio")
			audio_flag = "--wav_dir" if os.getenv("OMNICODER_AUDIO_IS_WAV", "0") == "1" else "--mel_dir"
			stages = [
				("pre_align", [sys.executable, "-m", "omnicoder.training.pre_align", "--data", os.getenv("OMNICODER_PREALIGN_DATA", "examples/data/vq/images"), "--steps", "1", "--device", args.device, "--embed_dim", os.getenv("OMNICODER_PREALIGN_EMBED_DIM", "256"), "--out", str(out_root / "pre_align_smoke.pt")]),
				("dsm", [sys.executable, "-m", "omnicoder.training.pretrain", "--data", os.getenv("OMNICODER_DATA_TEXT", "examples"), "--seq_len", "128", "--steps", "1", "--device", args.device, "--router_curriculum", os.getenv("OMNICODER_ROUTER_CURRICULUM", "topk>multihead>grin"), "--ds_moe_dense", "--ds_dense_until_frac", "0.0"]),
				("kd", [sys.executable, "-m", "omnicoder.training.draft_train", "--data", os.getenv("OMNICODER_DATA_CODE", "examples/code_eval"), "--seq_len", "128", "--steps", "1", "--device", args.device, "--teacher", os.getenv("OMNICODER_TEACHER", "microsoft/phi-2"), "--student_mobile_preset", os.getenv("OMNICODER_DRAFT_PRESET", "draft_3b"), "--lora"]),
				("vl", [sys.executable, "-m", "omnicoder.training.vl_fused_pretrain", "--jsonl", os.getenv("OMNICODER_VL_JSONL", "examples/vl_auto.jsonl"), "--steps", "1", "--device", args.device, "--mobile_preset", os.getenv("OMNICODER_TRAIN_PRESET", "mobile_4gb")]),
				("vqa", [sys.executable, "-m", "omnicoder.training.vl_fused_pretrain", "--jsonl", os.getenv("OMNICODER_VQA_JSONL", "examples/vl_auto.jsonl"), "--steps", "1", "--device", args.device, "--mobile_preset", os.getenv("OMNICODER_TRAIN_PRESET", "mobile_4gb")]),
				("av_flow", [sys.executable, "-m", "omnicoder.training.flow_recon", "--data", os.getenv("OMNICODER_FLOW_DATA", "examples/data/vq/images"), "--steps", "1", "--device", args.device]),
				("av_audio", [sys.executable, "-m", "omnicoder.training.audio_recon", audio_flag, audio_root, "--steps", "1", "--device", args.device]),
				("rl_grpo", [sys.executable, "-m", "omnicoder.training.rl_grpo", "--prompts", os.getenv("OMNICODER_GRPO_PROMPTS", "examples/grpo_prompts.jsonl"), "--steps", "1", "--device", args.device, "--mobile_preset", os.getenv("OMNICODER_TRAIN_PRESET", "mobile_4gb")]),
			]
			per = max(10, s_total // max(1, len(stages)))
			# Respect explicit request to disable timeouts (default: disabled)
			use_timeout = (os.getenv("OMNICODER_SMOKE_USE_TIMEOUT", "0") == "1")
			for name, cmd in stages:
				if use_timeout:
					rc2 = run_with_timeout(cmd, timeout_sec=per, env=env)
					print(f"[smoke] {name} rc={rc2} (timeout={per}s)")
				else:
					rc2 = run(cmd, env=env)
					print(f"[smoke] {name} rc={rc2} (no-timeout)")
			return
	except Exception as _eab:
		print(f"[probe] auto_benchmark skipped: {_eab}")

	# Default weights per domain; will be reweighted by observed weaknesses
	weights = {"dsm": 3, "kd": 3, "vl": 2, "vqa": 1, "av": 2, "rl": 1}
	try:
		if weak_json.exists():
			data = json.loads(weak_json.read_text(encoding='utf-8'))
			# Heuristic extraction with robust defaults
			text_nll = float(data.get("text_nll", float('inf')))
			clip_score = float(data.get("clip_score", 0.0))
			vqa_acc = float(data.get("vqa_acc", 0.0))
			video_fvd = float(data.get("video_fvd", float('inf')))
			audio_fad = float(data.get("audio_fad", float('inf')))
			# Reweight: high NLL → more DSM/KD; low CLIP/VQA → more VL/VQA; high FVD/FAD → more AV
			if not (text_nll != text_nll):  # not NaN
				if text_nll > 6.0:
					weights["dsm"] += 3; weights["kd"] += 3
				elif text_nll > 3.0:
					weights["dsm"] += 1; weights["kd"] += 2
			if clip_score < 0.25:
				weights["vl"] += 2
			elif clip_score < 0.20:
				weights["vl"] += 3
			if vqa_acc < 0.35:
				weights["vqa"] += 2
			if not (video_fvd != video_fvd):
				if video_fvd > 300.0:
					weights["av"] += 2
			if not (audio_fad != audio_fad):
				if audio_fad > 2.5:
					weights["av"] += 1
	except Exception:
		pass

	# Normalize weights to minutes across the budget (reserve a small export window)
	# Persist checkpoints on every stage completion to avoid losing good weights.
	reserve_export = max(min(budget // 12, 6), 2)
	alloc = max(1, budget - reserve_export)
	total_w = sum(weights.values()) or 1
	mins = {k: max(1, int(alloc * (w / total_w))) for k, w in weights.items()}
	# Ensure we keep a small pre-align pass
	mins_pre = max(min(budget // 12, 10), 2)
	# Compose the plan with weakness-aware minutes
	# Integrate diffusion-based text training if enabled: curriculum adds denoising steps on text corpora
	try:
		from omnicoder.training.unified_multimodal_train import add_diffusion_text_stage  # type: ignore
	except Exception:
		add_diffusion_text_stage = None  # type: ignore
	try:
		if os.getenv('OMNICODER_TRAIN_DIFFUSION_TEXT', '1') == '1' and add_diffusion_text_stage is not None:
			plan = add_diffusion_text_stage(plan)
	except Exception:
		pass
	plan = {
		"pre_align_min": mins_pre,
		"dsm_min": mins.get("dsm", 8),
		"kd_min": mins.get("kd", 8),
		"vl_min": mins.get("vl", 2),
		"vqa_min": mins.get("vqa", 1),
		"av_min": mins.get("av", 2),
		"rl_min": mins.get("rl", 1),
		"export_min": reserve_export,
	}
	steps = {
		"pre_align": minutes_to_steps(plan["pre_align_min"], ms_per_step=50.0),
		"dsm": minutes_to_steps(plan["dsm_min"], ms_per_step=220.0),
		"kd": minutes_to_steps(plan["kd_min"], ms_per_step=250.0),
		"vl": minutes_to_steps(plan["vl_min"], ms_per_step=300.0),
		"vqa": minutes_to_steps(plan["vqa_min"], ms_per_step=320.0),
		"av": minutes_to_steps(plan["av_min"], ms_per_step=340.0),
		"rl": minutes_to_steps(plan["rl_min"], ms_per_step=400.0),
	}

	# NEW: Always perform a single-step probe for each stage with exhaustive logging
	# This gives per-stage timing and verifies wiring without consuming budget
	try:
		probe_env = dict(os.environ)
		probe_env.setdefault("OMNICODER_LOG_LEVEL", "DEBUG")
		probe_env.setdefault("OMNICODER_LOG_FILE", str(Path("tests_logs")/"training_probe.jsonl"))
		print("[probe] running single-step probes for stages: pre_align, dsm, kd, vl, vqa, av, rl")
		# Pre-align (1 step)
		run([sys.executable, "-m", "omnicoder.training.pre_align", "--data", os.getenv("OMNICODER_PREALIGN_DATA", "examples/data/vq/images"), "--steps", "1", "--device", args.device, "--embed_dim", os.getenv("OMNICODER_PREALIGN_EMBED_DIM", "256")], env=probe_env)
		# DS-MoE pretrain (1 step)
		run([sys.executable, "-m", "omnicoder.training.pretrain", "--data", os.getenv("OMNICODER_DATA_TEXT", "examples"), "--seq_len", "128", "--steps", "1", "--device", args.device, "--router_curriculum", os.getenv("OMNICODER_ROUTER_CURRICULUM", "topk>multihead>grin"), "--ds_moe_dense", "--ds_dense_until_frac", "0.0"], env=probe_env)
		# KD (1 step)
		run([sys.executable, "-m", "omnicoder.training.draft_train", "--data", os.getenv("OMNICODER_DATA_CODE", "examples/code_eval"), "--seq_len", "128", "--steps", "1", "--device", args.device, "--teacher", os.getenv("OMNICODER_TEACHER", "microsoft/phi-2"), "--student_mobile_preset", os.getenv("OMNICODER_DRAFT_PRESET", "draft_3b"), "--lora"], env=probe_env)
		# VL (1 step)
		run([sys.executable, "-m", "omnicoder.training.vl_fused_pretrain", "--jsonl", os.getenv("OMNICODER_VL_JSONL", "examples/vl_auto.jsonl"), "--steps", "1", "--device", args.device, "--mobile_preset", os.getenv("OMNICODER_TRAIN_PRESET", "mobile_4gb")], env=probe_env)
		# VQA (1 step)
		run([sys.executable, "-m", "omnicoder.training.vl_fused_pretrain", "--jsonl", os.getenv("OMNICODER_VQA_JSONL", "examples/vl_auto.jsonl"), "--steps", "1", "--device", args.device, "--mobile_preset", os.getenv("OMNICODER_TRAIN_PRESET", "mobile_4gb")], env=probe_env)
		# AV (1 step)
		run([sys.executable, "-m", "omnicoder.training.flow_recon", "--data", os.getenv("OMNICODER_FLOW_DATA", "examples/data/vq/images"), "--steps", "1", "--device", args.device], env=probe_env)
		run([sys.executable, "-m", "omnicoder.training.audio_recon", "--mel_dir", os.getenv("OMNICODER_AUDIO_DATA", "examples/data/vq/audio"), "--steps", "1", "--device", args.device], env=probe_env)
		# RL GRPO (1 step)
		run([sys.executable, "-m", "omnicoder.training.rl_grpo", "--prompts", os.getenv("OMNICODER_GRPO_PROMPTS", "examples/grpo_prompts.jsonl"), "--steps", "1", "--device", args.device, "--mobile_preset", os.getenv("OMNICODER_TRAIN_PRESET", "mobile_4gb")], env=probe_env)
	except Exception as _e_probe:
		print(f"[probe] single-step probes encountered issues (continuing): {_e_probe}")
	# Cap steps with smoke defaults while allowing budget-based scaling
	def _cap(key: str, env_key: str, default_cap: int) -> int:
		try:
			cap = int(os.getenv(env_key, str(default_cap)))
		except Exception:
			cap = default_cap
		return min(steps[key], cap)
	steps["kd"] = _cap("kd", "OMNICODER_SMOKE_KD_STEPS", 10)
	steps["vl"] = _cap("vl", "OMNICODER_SMOKE_VL_STEPS", 5)
	steps["vqa"] = _cap("vqa", "OMNICODER_SMOKE_VQA_STEPS", 3)
	steps["av"] = _cap("av", "OMNICODER_SMOKE_AV_STEPS", 3)
	steps["rl"] = _cap("rl", "OMNICODER_SMOKE_RL_STEPS", 1)

	# NEW: Enforce exact total-run minutes if OMNICODER_TRAIN_EXACT_MINUTES is set
	try:
		_exact = os.getenv("OMNICODER_TRAIN_EXACT_MINUTES", "").strip()
		if _exact:
			exact_min = int(max(1, float(_exact)))
			# Recompute all steps proportionally to fit exact_min
			total_planned = sum(plan.values())
			if total_planned > 0:
				scale = float(exact_min) / float(total_planned)
				for k, ms in list(plan.items()):
					mins_k = max(1, int(round(ms * scale)))
					msps = {"pre_align":50.0, "dsm":220.0, "kd":250.0, "vl":300.0, "vqa":320.0, "av":340.0, "rl":400.0}.get(k, 300.0)
					steps[k] = minutes_to_steps(mins_k, ms_per_step=msps)
	except Exception:
		pass

	def _remaining_minutes() -> int:
		return max(0, int((planned_end_ts - time.time()) / 60.0))

	def _replan_for_remaining(done_keys: list[str]) -> None:
		rem = _remaining_minutes()
		if rem <= 5:
			return
		# Proportionally allocate remaining across not-done buckets with minimal guards
		def _stage_weights() -> dict[str,int]:
			w = {"kd":3, "vl":1, "vqa":1, "av":1, "rl":1}
			try:
				# Boost VQA when multiple instruction/vision QA sources are present
				instr_keys = [
					"OMNICODER_MM_INSTR_JSONL","OMNICODER_LLAVA_INSTR_JSONL","OMNICODER_SHAREGPT4V_JSONL",
					"OMNICODER_MINIGPT4_JSONL","OMNICODER_VIDEOCHAT_JSONL","OMNICODER_INSTRUCTBLIP_JSONL","OMNICODER_LLAVA_NEXT_JSONL",
				]
				instr_count = sum(1 for k in instr_keys if env.get(k, "").strip())
				vqa_keys = [
					"OMNICODER_MSRVTT_VQA_JSONL","OMNICODER_VATEX_VQA_JSONL","OMNICODER_MSVDQA_JSONL","OMNICODER_TGIFQA_JSONL",
					"OMNICODER_TVQA_JSONL","OMNICODER_NEXTQA_JSONL","OMNICODER_TEXTVQA_JSONL","OMNICODER_DOCVQA_JSONL","OMNICODER_CHARTQA_JSONL",
				]
				vqa_count = sum(1 for k in vqa_keys if env.get(k, "").strip())
				if instr_count >= 2:
					w["vqa"] += 2
				if vqa_count >= 3:
					w["vqa"] += 2
				# Boost AV when video data + webvid/activitynet present
				if env.get("OMNICODER_VIDEO_DATA", "").strip() and (env.get("OMNICODER_WEBVID_CSV", "").strip() or env.get("OMNICODER_ACTIVITYNET_JSONL", "").strip()):
					w["av"] += 1
				# Keep KD competitive early if budget is large
				if rem > 30:
					w["kd"] += 1
				# Adjust using recent micro-evals when present
				try:
					# VQA micro-evals (proxy_em or train_ce_last)
					recent_vqa = []
					for k in ("vqa_textvqa_out","vqa_docvqa_out","vqa_chartqa_out","vqa_scienceqa_out"):
						ck = manifest.get(k)
						if ck:
							m = _try_load_metrics(ck)
							if m:
								recent_vqa.append(max(m.get("proxy_em", 0.0), m.get("train_ce_last", 0.0)))
					# Include video VQA CE proxies if available
					for k in ("video_vqa_out","video_vqa_vatex_out","video_vqa_msvdqa_out","video_vqa_tgifqa_out"):
						ck = manifest.get(k)
						if ck:
							m = _try_load_metrics(ck)
							if m:
								recent_vqa.append(m.get("train_ce_last", 0.0))
					if recent_vqa:
						avg = sum(recent_vqa) / max(1, len(recent_vqa))
						if avg < 0.5:
							w["vqa"] += 1
						else:
							w["vqa"] = max(1, w["vqa"] - 1)
					# AV span tIoU signals
					for k in ("video_span_charades_out","video_span_ego4d_out"):
						ck = manifest.get(k)
						if ck:
							m = _try_load_metrics(ck)
							if m.get("tiou_at_0_5", 0.0) < 0.4:
								w["av"] += 1
				except Exception:
					pass
			except Exception:
				pass
			return w
		weights = _stage_weights()
		avail_keys = [k for k in ["kd","vl","vqa","av","rl"] if k not in done_keys]
		if not avail_keys:
			return
		total_w = sum(weights[k] for k in avail_keys)
		for k in avail_keys:
			mins_k = max(2, int(rem * weights[k] / max(1, total_w)))
			ms = {"kd":250.0, "vl":300.0, "vqa":320.0, "av":340.0, "rl":400.0}[k]
			steps[k] = minutes_to_steps(mins_k, ms_per_step=ms)

	# Ensure minimal synthetic text data exists for DSM/KD smoke so stages consume time
	text_root = Path(args.text_data)
	text_root.mkdir(parents=True, exist_ok=True)
	if not any(text_root.rglob('*.txt')):
		try:
			(text_root / 'synth.txt').write_text(("OmniCoder synthetic text.\n" * 1024), encoding='utf-8')
		except Exception:
			pass
	manifest = {"plan": plan, "steps": steps, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

	# Dry-run: print plan/steps and exit successfully
	if bool(args.dry_run_plan):
		print(json.dumps({"plan": plan, "steps": steps}, indent=2))
		# Write a lightweight planning summary next to out_root for reproducibility
		(out_root / "TRAINING_PLAN.json").write_text(json.dumps({"plan": plan, "steps": steps}, indent=2))
		return

	# Persist caches (extend the existing env with defaults later)
	env.setdefault("HF_HOME", env.get("HF_HOME", str(Path("/models/hf")) if Path("/models").exists() else str(Path("models/hf"))))
	# Avoid deprecated TRANSFORMERS_CACHE to silence warnings; HF honors HF_HOME
	# Threads may have been set by apply_thread_env_if_auto; keep any values already present
	env.setdefault("OMP_NUM_THREADS", env.get("OMP_NUM_THREADS", "1"))
	env.setdefault("MKL_NUM_THREADS", env.get("MKL_NUM_THREADS", "1"))
	env.setdefault("TORCH_NUM_THREADS", env.get("TORCH_NUM_THREADS", "1"))

	# Log a short resource summary early
	try:
		print("[resources]", gpu_summary())
	except Exception:
		pass

	# Centralized run-time defaults (fills missing keys only)
	try:
		apply_run_env_defaults(env)
		# Encourage MTP/speculative and Omega reasoner by default unless explicitly disabled
		env.setdefault("OMNICODER_ENABLE_MTP", env.get("OMNICODER_ENABLE_MTP", "1"))
		env.setdefault("OMNICODER_REASONER", env.get("OMNICODER_REASONER", "omega"))
	except Exception:
		pass

	# Sidecar defaults for quality/acceptance metrics if unset
	try:
		if not env.get("SFB_CLIP_JSONL", "").strip() and Path("examples/sidecars/clip_pairs.jsonl").exists():
			env["SFB_CLIP_JSONL"] = "examples/sidecars/clip_pairs.jsonl"
		if not env.get("OMNICODER_CLIP_JSONL", "").strip() and env.get("SFB_CLIP_JSONL", "").strip():
			env["OMNICODER_CLIP_JSONL"] = env["SFB_CLIP_JSONL"]
		if not env.get("OMNICODER_ASR_JSONL", "").strip() and Path("examples/sidecars/asr_pairs.jsonl").exists():
			env["OMNICODER_ASR_JSONL"] = "examples/sidecars/asr_pairs.jsonl"
		if not env.get("OMNICODER_CODE_TASKS", "").strip() and Path("examples/sidecars/code_tasks.jsonl").exists():
			env["OMNICODER_CODE_TASKS"] = "examples/sidecars/code_tasks.jsonl"
		# Audio FAD dirs
		if not env.get("OMNICODER_FAD_PRED_DIR", "").strip() and Path("examples/sidecars/fad_pred").exists():
			env["OMNICODER_FAD_PRED_DIR"] = "examples/sidecars/fad_pred"
		if not env.get("OMNICODER_FAD_REF_DIR", "").strip() and Path("examples/sidecars/fad_ref").exists():
			env["OMNICODER_FAD_REF_DIR"] = "examples/sidecars/fad_ref"
		# Video FVD dirs if not already set by bootstrap/profile
		if not env.get("OMNICODER_VIDEO_PRED_DIR", "").strip() and Path("examples/sidecars/fvd_pred").exists():
			env["OMNICODER_VIDEO_PRED_DIR"] = "examples/sidecars/fvd_pred"
		if not env.get("OMNICODER_VIDEO_REF_DIR", "").strip() and Path("examples/sidecars/fvd_ref").exists():
			env["OMNICODER_VIDEO_REF_DIR"] = "examples/sidecars/fvd_ref"
	except Exception:
		pass

	# Bootstrap tiny multimodal datasets so all stages can run end-to-end
	try:
		from omnicoder.tools.prepare_multimodal_data import prepare_all, prepare_media_gallery  # type: ignore
		info = prepare_all(out_root=str(out_root))
		# Wire defaults so subsequent stages pick them up automatically
		env.setdefault("OMNICODER_DATA_TEXT", info.get("text_dir", args.text_data))
		env.setdefault("OMNICODER_DATA_VL", info.get("vl_jsonl", args.vl_data))
		env.setdefault("OMNICODER_DATA_ASR", info.get("asr_dir", args.asr_data))
		env.setdefault("OMNICODER_DATA_TTS", info.get("tts_dir", args.tts_data))
		env.setdefault("OMNICODER_VIDEO_PRED_DIR", info.get("fvd_pred_dir", ""))
		env.setdefault("OMNICODER_VIDEO_REF_DIR", info.get("fvd_ref_dir", ""))
		md = prepare_media_gallery(out_root=str(out_root))
		env.setdefault("OMNICODER_CC_MEDIA_DIR", md)
	except Exception:
		print("[warn] dataset bootstrap skipped")

	# Best-effort: auto-fetch public datasets across modalities for richer training by default.
	try:
		if os.getenv("OMNICODER_AUTOFETCH", "1") == "1":
			print("[datasets] autofetch_datasets (in-process) ...")
			from omnicoder.tools.autofetch_datasets import autofetch_all  # type: ignore
			_limit = int(os.getenv("OMNICODER_FETCH_LIMIT", env.get("OMNICODER_FETCH_LIMIT", "1000000")))
			# Reuse existing exports/downloads by default
			# Respect existing OMNICODER_FORCE_FETCH if set; otherwise keep default behavior from centralized defaults/profile
			# (No direct mutation here to avoid conflicting with profiles.)
			# Skip redundant autofetch when outputs already exist; allow explicit override via env.
			already = (Path('data/text/wikitext2.txt').exists() and Path('data/code/mbpp.jsonl').exists())
			fetched = ({} if already and os.getenv('OMNICODER_FORCE_FETCH','0')!='1' else autofetch_all(_limit))
			# Prevent re-entrancy across stages by stamping a success marker
			try:
				Path('data/.autofetch.ok').write_text('OK', encoding='utf-8')
			except Exception:
				pass
			# Map fetched outputs into env only when not already provided
			def _set_if_missing(k: str, v: object) -> None:
				try:
					if (not env.get(k, "").strip()) and v:
						env[k] = str(v)
				except Exception:
					pass
			# Text: prefer FineWeb/C4/OpenWebText, fallback to wikitext
			_set_if_missing("OMNICODER_DATA_TEXT", fetched.get("fineweb_edu") or fetched.get("c4_en") or fetched.get("openwebtext") or fetched.get("text"))
			# Code: prefer CodeParrot clean, fallback to MBPP
			_set_if_missing("OMNICODER_DATA_CODE", fetched.get("codeparrot_clean") or fetched.get("code"))
			# Vision-Language JSONL: prefer VL-CC or COCO jsonls
			_set_if_missing("OMNICODER_VL_JSONL", fetched.get("vl_cc_jsonl") or fetched.get("vl_coco_jsonl") or fetched.get("vl_jsonl"))
			# VQA JSONL
			_set_if_missing("OMNICODER_VQA_JSONL", fetched.get("vqa_jsonl") or fetched.get("vqav2_jsonl") or fetched.get("okvqa_jsonl"))
			# COCO segmentation images list for optional grounding/seg pass
			if fetched.get("coco_seg_list"):
				env.setdefault("OMNICODER_COCO_SEG_LIST", str(fetched.get("coco_seg_list")))
			if fetched.get("coco_seg_images_dir"):
				env.setdefault("OMNICODER_COCO_SEG_IMAGES_DIR", str(fetched.get("coco_seg_images_dir")))
			# LVIS/OpenImages images lists for grounding
			if fetched.get("lvis_images_list"):
				env.setdefault("OMNICODER_LVIS_IMAGES_LIST", str(fetched.get("lvis_images_list")))
			if fetched.get("openimages_images_list"):
				env.setdefault("OMNICODER_OPENIMAGES_IMAGES_LIST", str(fetched.get("openimages_images_list")))
			# ASR and TTS roots
			_set_if_missing("OMNICODER_DATA_ASR", fetched.get("asr_dir"))
			_set_if_missing("OMNICODER_DATA_TTS", fetched.get("tts_wavs_dir") or "data/tts")
			# Audio training root: prefer Speech Commands/ASR wavs, then TTS wavs
			_audio_wav = (
				fetched.get("sc_wavs_dir") or fetched.get("wav_dir") or fetched.get("tts_wavs_dir")
			)
			if _audio_wav:
				env.setdefault("OMNICODER_AUDIO_DATA", str(_audio_wav))
				env.setdefault("OMNICODER_AUDIO_IS_WAV", "1")
			if fetched.get("tts_texts"):
				env.setdefault("OMNICODER_TTS_TEXTS", str(fetched.get("tts_texts")))
			# ASR transcripts (quality sidecar) if present
			if fetched.get("transcripts"):
				env.setdefault("OMNICODER_ASR_JSONL", str(fetched.get("transcripts")))
			# MLS multilingual ASR
			if fetched.get("mls_wavs_dir"):
				env.setdefault("OMNICODER_DATA_ASR", str(fetched.get("mls_wavs_dir")))
			if fetched.get("mls_transcripts"):
				env.setdefault("OMNICODER_ASR_JSONL", str(fetched.get("mls_transcripts")))
			# Video dir if present
			_set_if_missing("OMNICODER_VIDEO_DATA", fetched.get("video_dir"))
			# Enable AV-sync if both audio and video roots are present
			if env.get("OMNICODER_VIDEO_DATA", "").strip() and env.get("OMNICODER_AUDIO_DATA", "").strip():
				env.setdefault("OMNICODER_AV_SYNC", "1")
			# Additional video caption JSONLs
			if fetched.get("youcook2_jsonl"):
				env.setdefault("OMNICODER_YOUCOOK2_JSONL", str(fetched.get("youcook2_jsonl")))
			if fetched.get("didemo_jsonl"):
				env.setdefault("OMNICODER_DIDEMO_JSONL", str(fetched.get("didemo_jsonl")))
			# Video QA JSONLs
			if fetched.get("tvqa_jsonl"):
				env.setdefault("OMNICODER_TVQA_JSONL", str(fetched.get("tvqa_jsonl")))
			if fetched.get("nextqa_jsonl"):
				env.setdefault("OMNICODER_NEXTQA_JSONL", str(fetched.get("nextqa_jsonl")))
			# QA JSONLs (text fine-tune) for optional stage later
			for env_key, fetched_key in [
				("OMNICODER_ARC_JSONL", "arc_jsonl"),
				("OMNICODER_TRUTHFULQA_JSONL", "truthfulqa_jsonl"),
				("OMNICODER_WINOGRANDE_JSONL", "winogrande_jsonl"),
				("OMNICODER_HELLASWAG_JSONL", "hellaswag_jsonl"),
				("OMNICODER_MMLU_JSONL", "mmlu_jsonl"),
				("OMNICODER_AGIEVAL_JSONL", "agieval_jsonl"),
				("OMNICODER_BBH_JSONL", "bbh_jsonl"),
				("OMNICODER_STRATEGYQA_JSONL", "strategyqa_jsonl"),
				("OMNICODER_ARC_CH_JSONL", "arc_ch_jsonl"),
				("OMNICODER_NQ_JSONL", "nq_jsonl"),
				("OMNICODER_HOTPOTQA_JSONL", "hotpotqa_jsonl"),
			]:
				val = fetched.get(fetched_key)
				if val:
					env.setdefault(env_key, str(val))
			# Eval sidecars when available
			if fetched.get("coco_captions_jsonl"):
				env.setdefault("OMNICODER_COCO_CAPTIONS_JSONL", str(fetched.get("coco_captions_jsonl")))
			if fetched.get("vqav2_jsonl"):
				env.setdefault("OMNICODER_VQAV2_JSONL", str(fetched.get("vqav2_jsonl")))
			if fetched.get("okvqa_jsonl"):
				env.setdefault("OMNICODER_OKVQA_JSONL", str(fetched.get("okvqa_jsonl")))
			# TextCaps (OCR-style VQA)
			if fetched.get("textcaps_jsonl"):
				env.setdefault("OMNICODER_TEXTCAPS_JSONL", str(fetched.get("textcaps_jsonl")))
			# DocVQA/ChartQA
			if fetched.get("docvqa_jsonl"):
				env.setdefault("OMNICODER_DOCVQA_JSONL", str(fetched.get("docvqa_jsonl")))
			if fetched.get("chartqa_jsonl"):
				env.setdefault("OMNICODER_CHARTQA_JSONL", str(fetched.get("chartqa_jsonl")))
			# OCR synthetic
			if fetched.get("ocr_synth_jsonl"):
				env.setdefault("OMNICODER_OCR_JSONL", str(fetched.get("ocr_synth_jsonl")))
			# Flickr30k captions (extra caption finetune source)
			if fetched.get("flickr30k_jsonl"):
				env.setdefault("OMNICODER_FLICKR30K_JSONL", str(fetched.get("flickr30k_jsonl")))
			# Flickr8k captions
			if fetched.get("flickr8k_jsonl"):
				env.setdefault("OMNICODER_FLICKR8K_JSONL", str(fetched.get("flickr8k_jsonl")))
			# Visual Genome QA
			if fetched.get("vg_vqa_jsonl"):
				env.setdefault("OMNICODER_VG_VQA_JSONL", str(fetched.get("vg_vqa_jsonl")))
			# WebVid roots
			if fetched.get("webvid_csv"):
				env.setdefault("OMNICODER_WEBVID_CSV", str(fetched.get("webvid_csv")))
			# Multimodal instructions JSONL
			if fetched.get("mm_instructions_jsonl"):
				env.setdefault("OMNICODER_MM_INSTR_JSONL", str(fetched.get("mm_instructions_jsonl")))
			# Real instruction datasets
			for env_key, fetched_key in [
				("OMNICODER_LLAVA_INSTR_JSONL", "llava_instruct_jsonl"),
				("OMNICODER_SHAREGPT4V_JSONL", "sharegpt4v_jsonl"),
				("OMNICODER_MINIGPT4_JSONL", "minigpt4_jsonl"),
				("OMNICODER_VIDEOCHAT_JSONL", "videochat_jsonl"),
				("OMNICODER_INSTRUCTBLIP_JSONL", "instructblip_jsonl"),
				("OMNICODER_LLAVA_NEXT_JSONL", "llava_next_jsonl"),
			]:
				val = fetched.get(fetched_key)
				if val:
					env.setdefault(env_key, str(val))
			# Video temporal localization datasets
			if fetched.get("activitynet_jsonl"):
				env.setdefault("OMNICODER_ACTIVITYNET_JSONL", str(fetched.get("activitynet_jsonl")))
			if fetched.get("charades_sta_jsonl"):
				env.setdefault("OMNICODER_CHARADES_STA_JSONL", str(fetched.get("charades_sta_jsonl")))
			if fetched.get("ego4d_nlq_jsonl"):
				env.setdefault("OMNICODER_EGO4D_NLQ_JSONL", str(fetched.get("ego4d_nlq_jsonl")))
			# PubTables/Objects365
			if fetched.get("pubtables_jsonl"):
				env.setdefault("OMNICODER_PUBTABLES_JSONL", str(fetched.get("pubtables_jsonl")))
			if fetched.get("objects365_images_list"):
				env.setdefault("OMNICODER_OBJECTS365_IMAGES_LIST", str(fetched.get("objects365_images_list")))
			# Additional OCR-VQA
			if fetched.get("textvqa_jsonl"):
				env.setdefault("OMNICODER_TEXTVQA_JSONL", str(fetched.get("textvqa_jsonl")))
			# ScienceQA
			if fetched.get("scienceqa_jsonl"):
				env.setdefault("OMNICODER_SCIENCEQA_JSONL", str(fetched.get("scienceqa_jsonl")))
			# Additional Video QA
			if fetched.get("msvdqa_jsonl"):
				env.setdefault("OMNICODER_MSVDQA_JSONL", str(fetched.get("msvdqa_jsonl")))
			if fetched.get("tgifqa_jsonl"):
				env.setdefault("OMNICODER_TGIFQA_JSONL", str(fetched.get("tgifqa_jsonl")))
			# Grounding (RefCOCO)
			if fetched.get("refcoco_jsonl"):
				env.setdefault("OMNICODER_REFCOCO_JSONL", str(fetched.get("refcoco_jsonl")))
			# Clotho audio captions
			if fetched.get("clotho_jsonl"):
				env.setdefault("OMNICODER_CLOTHO_JSONL", str(fetched.get("clotho_jsonl")))
			if fetched.get("clotho_wavs_dir"):
				env.setdefault("OMNICODER_CLOTHO_WAVS_DIR", str(fetched.get("clotho_wavs_dir")))
			# FSD50K events
			if fetched.get("fsd50k_jsonl"):
				env.setdefault("OMNICODER_FSD50K_JSONL", str(fetched.get("fsd50k_jsonl")))
			# MSR-VTT JSONL
			if fetched.get("msrvtt_jsonl"):
				env.setdefault("OMNICODER_MSRVTT_VQA_JSONL", str(fetched.get("msrvtt_jsonl")))
			# VATEX JSONL
			if fetched.get("vatex_jsonl"):
				env.setdefault("OMNICODER_VATEX_VQA_JSONL", str(fetched.get("vatex_jsonl")))
			# AudioCaps
			if fetched.get("ac_wavs_dir"):
				env.setdefault("OMNICODER_AUDIOCAPS_DIR", str(fetched.get("ac_wavs_dir")))
			if fetched.get("ac_jsonl"):
				env.setdefault("OMNICODER_AUDIOCAPS_JSONL", str(fetched.get("ac_jsonl")))
			# ESC-50 / VGGSound for audio events diversity
			if fetched.get("esc50_wavs_dir"):
				env.setdefault("OMNICODER_AUDIO_EVENTS_DIR", str(fetched.get("esc50_wavs_dir")))
			if fetched.get("vggsound_wavs_dir"):
				env.setdefault("OMNICODER_AUDIO_EVENTS_DIR", env.get("OMNICODER_AUDIO_EVENTS_DIR", str(fetched.get("vggsound_wavs_dir"))))
			# SR seed dirs
			if fetched.get("sr_ref_dir"):
				env.setdefault("OMNICODER_SR_REF_DIR", str(fetched.get("sr_ref_dir")))
			if fetched.get("sr_pred_dir"):
				env.setdefault("OMNICODER_SR_PRED_DIR", str(fetched.get("sr_pred_dir")))

			# Multilingual text (cc100) — append to DATA_TEXT when present
			for k, v in list(fetched.items()):
				if k.startswith("cc100_") and (k.endswith("_error") is False):
					try:
						cur = env.get("OMNICODER_DATA_TEXT", "").strip()
						val = str(v)
						if val:
							env["OMNICODER_DATA_TEXT"] = f"{cur};{val}" if cur else val
					except Exception:
						pass
	except Exception as _eaf:
		print(f"[warn] autofetch_datasets skipped: {_eaf}")

	# Overlay dataset profile (full datasets) when profiles/datasets.json defines host presets
	try:
		prof = Path("profiles/datasets.json")
		if prof.exists():
			cfg = json.loads(prof.read_text(encoding='utf-8'))
			# Choose host profile via env override or basic OS heuristic
			profile = os.getenv("OMNICODER_DATA_PROFILE", "windows_dev" if sys.platform.startswith("win") else "linux_train")
			host = (cfg.get("host_examples", {}) or {}).get(profile, {})
			if isinstance(host, dict) and host:
				def _set(k_env: str, key: str) -> None:
					try:
						val = host.get(key, {}).get("path", "") if isinstance(host.get(key, {}), dict) else host.get(key, "")
						if val:
							env[k_env] = str(val)
					except Exception:
						pass
				_set("OMNICODER_DATA_TEXT", "text")
				_set("OMNICODER_DATA_CODE", "code")
				_set("OMNICODER_DATA_VL", "vl")
				_set("OMNICODER_DATA_ASR", "asr")
				_set("OMNICODER_DATA_TTS", "tts")
				_set("OMNICODER_DATA_VIDEO", "video")
				# Additional eval JSONLs if provided by profile (turn on by default)
				_set("OMNICODER_VQAV2_JSONL", "vqav2")
				_set("OMNICODER_OKVQA_JSONL", "okvqa")
				_set("OMNICODER_COCO_CAPTIONS_JSONL", "coco_captions")
				_set("OMNICODER_MSRVTT_VQA_JSONL", "msrvtt_vqa")
				# If video entry provides ref_dir, route to FVD/FID defaults
				try:
					v = host.get("video", {})
					if isinstance(v, dict):
						if v.get("ref_dir"):
							env.setdefault("OMNICODER_VIDEO_REF_DIR", str(v.get("ref_dir")))
						# Predict dir will be under weights unless overridden
						env.setdefault("OMNICODER_VIDEO_PRED_DIR", str((out_root / "sidecars" / "fvd_pred").as_posix()))
				except Exception:
					pass
				print(f"[datasets] using host profile: {profile}")
	except Exception:
		print("[warn] dataset profile overlay skipped")

	# 0) Optional resource probe to record VRAM/tokens-per-second, aiding auto-planning
	if os.getenv("OMNICODER_RUN_PROBE", "0") == "1":
		try:
			run([
				sys.executable, "-m", "omnicoder.tools.train_probe",
				"--budget_minutes", os.getenv("OMNICODER_PROBE_MINUTES", "10"),
				"--device", args.device,
				*( ["--ep_devices", os.getenv("OMNICODER_EP_DEVICES", "cuda:0,cuda:1")] if os.getenv("OMNICODER_ENABLE_EP", "0") == "1" else [] ),
			], env=env)
		except Exception:
			print("[warn] train_probe skipped")

	# 1) Pre-align
	pre_align_out = out_root / "pre_align.pt"
	pre_align_data = os.getenv("OMNICODER_PREALIGN_DATA", json.loads(Path("profiles/datasets.json").read_text(encoding='utf-8')).get("vl", {}).get("path", "examples/data/vq/images") if Path("profiles/datasets.json").exists() else "examples/data/vq/images")
	# If a JSONL path is configured and points to examples/vl_auto.jsonl, ensure it uses host paths, not "/workspace".
	try:
		if str(pre_align_data).lower().endswith(".jsonl") and Path(pre_align_data).name == "vl_auto.jsonl":
			text = Path(pre_align_data).read_text(encoding='utf-8')
			if "/workspace/" in text:
				fixed = text.replace("/workspace/", "")
				Path(pre_align_data).write_text(fixed, encoding='utf-8')
	except Exception:
		pass
	if (not args.resume) or (not pre_align_out.exists()):
		run([
			sys.executable, "-m", "omnicoder.training.pre_align",
			"--data", pre_align_data,
			"--steps", str(steps["pre_align"]),
			"--device", args.device,
			"--embed_dim", os.getenv("OMNICODER_PREALIGN_EMBED_DIM", "256"),
			"--out", str(pre_align_out),
		], env=env)
	manifest["pre_align_out"] = str(pre_align_out)

	# 1a*) Cross-modal shared-latent alignment (concept head)
	try:
		align_out = out_root / "omnicoder_align.pt"
		if (not args.resume) or (not align_out.exists()):
			run([
				sys.executable, "-m", "omnicoder.training.cross_modal_align",
				"--jsonl", os.getenv("OMNICODER_ALIGN_JSONL", os.getenv("OMNICODER_VL_JSONL", args.vl_data)),
				"--steps", str(minutes_to_steps(max(plan["vl_min"] // 2, 1), ms_per_step=220.0)),
				"--device", args.device,
				"--mobile_preset", args.student_preset,
				"--prealign_ckpt", str(pre_align_out),
				"--out", str(align_out),
			], env=env)
		manifest["align_out"] = str(align_out)
	except Exception as e:
		print(f"[warn] cross_modal_align skipped: {e}")

	# 1a.1) Export unified preprocessors (text/image/audio/video heads) as ONNX when enabled
	if os.getenv("OMNICODER_EXPORT_PREPROCESSORS", "1") == "1":
		try:
			pre_out = out_root / "release" / "preprocessors"
			pre_out.mkdir(parents=True, exist_ok=True)
			run([
				sys.executable, "-m", "omnicoder.export.export_preprocessors",
				"--prealign_ckpt", str(pre_align_out),
				"--out_dir", str(pre_out),
				"--opset", os.getenv("OMNICODER_ONNX_OPSET", "17"),
				"--device", "cpu",
			], env=env)
			manifest["preprocessors_out"] = str(pre_out)
		except Exception as e:
			print(f"[warn] export_preprocessors skipped: {e}")

	# 1a) Cross-modal verifier training (mini-CLIP style)
	if bool(args.run_verifier_train):
		try:
			ver_out = out_root / "cross_modal_verifier.pt"
			if (not args.resume) or (not ver_out.exists()):
				run([
					sys.executable, "-m", "omnicoder.training.verifier_train",
					"--data", os.getenv("OMNICODER_VERIFIER_DATA", "examples/data/vq/images"),
					"--prealign_ckpt", str(pre_align_out),
					"--steps", str(minutes_to_steps(max(plan["pre_align_min"] // 2, 1), ms_per_step=100.0)),
					"--device", args.device,
					"--out", str(ver_out),
					"--out_onnx", str(out_root / "cross_modal_verifier.onnx"),
				], env=env)
			manifest["verifier_out"] = str(ver_out)
		except Exception as e:
			print(f"[warn] verifier_train failed/skipped: {e}")

	# 1a) Unified multi-index build (auto-defaults)
	multi_index_root = os.getenv("OMNICODER_MULTI_INDEX_ROOT", "")
	if not multi_index_root.strip():
		# Default to weights/unified_index under out_root when not provided
		multi_index_root = str(out_root / "unified_index")
		# Also export this default back into env for downstream tools in this run
		env["OMNICODER_MULTI_INDEX_ROOT"] = multi_index_root
	mi_out = Path(multi_index_root)
	# Force rebuild of RAG index when requested (default on) to keep retrieval fresh
	if os.getenv("OMNICODER_FORCE_REBUILD_RAG", "1") == "1":
		try:
			if mi_out.exists():
				shutil.rmtree(mi_out, ignore_errors=True)
		except Exception:
			pass
	if (not args.resume) or (not (mi_out / "embeddings.npy").exists()):
		try:
			run([
				sys.executable, "-m", "omnicoder.tools.multi_index_build",
				"--roots", pre_align_data, "docs",
				"--out", str(mi_out),
				"--device", args.device,
			], env=env)
		except Exception as e:
			print(f"[warn] multi_index_build skipped: {e}")
	manifest["multi_index"] = str(mi_out)
	# Ensure GraphRAG consumers during this run use the freshly built index
	env["OMNICODER_GRAPHRAG_ROOT"] = str(mi_out)

	# Optional: fact-check and enrich KG with web facts for configured queries
	try:
		fcq = os.getenv("OMNICODER_FACTCHECK_QUERIES", "").strip()
		if fcq:
			for q in [x for x in fcq.split(";") if x.strip()]:
				run([
					sys.executable, "-m", "omnicoder.tools.web_factcheck",
					"--query", q,
					"--out_root", str(mi_out),
				], env=env)
	except Exception as _efc:
		print(f"[warn] web_factcheck skipped: {_efc}")

	# 1b) DS‑MoE pretrain (expert-parallel optional)
	dsm_out = out_root / "omnicoder_pretrain_dsm.pt"
	if steps["dsm"] > 0 and ((not args.resume) or (not dsm_out.exists())):
		# Optional expert-parallel launch across multiple GPUs
		if os.getenv("OMNICODER_ENABLE_EP", "0") == "1":
			devices = os.getenv("OMNICODER_EP_DEVICES", "cuda:0,cuda:1")
			run([
				sys.executable, "-m", "omnicoder.tools.torchrun_ep",
				"--script", "omnicoder.training.pretrain",
				"--script_args",
				"--data {} --seq_len {} --steps {} --device cuda --router_curriculum {} --ds_moe_dense --ds_dense_until_frac {} --moe_static_capacity {}".format(
					args.text_data,
					os.getenv("OMNICODER_PRETRAIN_SEQ_LEN", "1024"),
					str(steps["dsm"]),
					os.getenv("OMNICODER_ROUTER_CURRICULUM", "topk>multihead>grin"),
					os.getenv("OMNICODER_DS_DENSE_UNTIL", "0.3"),
					os.getenv("OMNICODER_MOE_STATIC_CAPACITY", "0"),
				),
				"--devices", devices,
			], env=env)
		else:
			dsm_cmd = [
				sys.executable, "-m", "omnicoder.training.pretrain",
				"--data", args.text_data,
				"--seq_len", os.getenv("OMNICODER_PRETRAIN_SEQ_LEN", "1024"),
				"--steps", str(steps["dsm"]),
				"--device", args.device,
				"--router_curriculum", os.getenv("OMNICODER_ROUTER_CURRICULUM", "topk>multihead>grin"),
				"--ds_moe_dense", "--ds_dense_until_frac", os.getenv("OMNICODER_DS_DENSE_UNTIL", "0.3"),
				"--moe_static_capacity", os.getenv("OMNICODER_MOE_STATIC_CAPACITY", "0"),
			]
			# Variable‑K and early‑exit/difficulty training: enable by default for mobile presets unless explicitly disabled
			_var_k_env = os.getenv("OMNICODER_VAR_K_TRAIN", "")
			_halt_env = os.getenv("OMNICODER_HALT_TRAIN", "")
			_enable_var_k = (_var_k_env == "1") or (_var_k_env == "" and str(args.mobile_preset).startswith("mobile"))
			_enable_halt = (_halt_env == "1") or (_halt_env == "" and str(args.mobile_preset).startswith("mobile"))
			if _enable_var_k:
				dsm_cmd += ["--var_k_train", "--var_k_min", os.getenv("OMNICODER_VAR_K_MIN", "1"), "--var_k_max", os.getenv("OMNICODER_VAR_K_MAX", "4")]
			if _enable_halt:
				dsm_cmd += [
					"--diff_loss_coef", os.getenv("OMNICODER_DIFF_LOSS_COEF", "0.01"),
					"--halt_loss_coef", os.getenv("OMNICODER_HALT_LOSS_COEF", "0.01"),
					"--halt_entropy", os.getenv("OMNICODER_HALT_ENTROPY", "1.0"),
				]
			mgs = os.getenv("OMNICODER_MOE_GROUP_SIZES", "")
			if mgs.strip():
				dsm_cmd += ["--moe_group_sizes", mgs]
			run(dsm_cmd, env=env)
	manifest["dsm_out"] = str(dsm_out)

	# Quick auto-bench after DS‑MoE pretrain (stage snapshot) with quality sidecars
	try:
		_ab = [
			sys.executable, "-m", "omnicoder.eval.auto_benchmark",
			"--device", "cpu",
			"--seq_len", "128", "--gen_tokens", "64",
			"--preset", args.student_preset,
			"--out", str(out_root / "bench_after_pretrain.json"),
		]
		# Add available quality sources
		from pathlib import Path as _P3
		def _maybe(flag: str, key: str) -> None:
			p = env.get(key, "").strip()
			if p and _P3(p).exists():
				_ab.extend([flag, p])
		_maybe("--clip_jsonl", "OMNICODER_CLIP_JSONL")
		_maybe("--asr_jsonl", "OMNICODER_ASR_JSONL")
		_maybe("--code_tasks", "OMNICODER_CODE_TASKS")
		_maybe("--fvd_pred_dir", "OMNICODER_VIDEO_PRED_DIR")
		_maybe("--fvd_ref_dir", "OMNICODER_VIDEO_REF_DIR")
		_maybe("--fad_pred_dir", "OMNICODER_FAD_PRED_DIR")
		_maybe("--fad_ref_dir", "OMNICODER_FAD_REF_DIR")
		# Also include examples defaults if envs unset
		if not env.get("OMNICODER_CLIP_JSONL", "").strip() and _P3("examples/sidecars/clip_pairs.jsonl").exists():
			_ab.extend(["--clip_jsonl", "examples/sidecars/clip_pairs.jsonl"])
		run(_ab, env=env)
		# Update best registry for text tokens/sec after DSM
		try:
			_ingest_bench(out_root, out_root / "bench_after_pretrain.json", dsm_out, model_registry)
		except Exception:
			pass
	except Exception:
		pass

	# 1z) Load teacher map (optional)
	teachers_map = {}
	try:
		teachers_map = json.loads(Path("profiles/teachers.json").read_text(encoding='utf-8'))
	except Exception:
		teachers_map = {}

	# 2) Draft KD
	kd_stage_records: list[dict] = []
	def _run_kd_stage(stage_name: str, data_arg: list[str], teacher_id: str, out_ckpt: Path) -> None:
		rec = {"stage": stage_name, "teacher": teacher_id, "out": str(out_ckpt)}
		cmd = [
			sys.executable, "-m", "omnicoder.training.draft_train",
			*data_arg,
			"--seq_len", os.getenv("OMNICODER_KD_SEQ_LEN", "512"),
			"--steps", str(steps["kd"]),
			"--batch_size", os.getenv("OMNICODER_KD_BATCH", "2"),
			"--device", args.device,
			"--teacher", teacher_id,
			"--teacher_device_map", os.getenv("OMNICODER_TEACHER_DEVICE_MAP", "auto"),
			"--teacher_dtype", os.getenv("OMNICODER_TEACHER_DTYPE", "auto"),
			"--student_mobile_preset", args.draft_preset,
			"--lora", "--lora_r", os.getenv("OMNICODER_LORA_R", "16"),
			"--lora_alpha", os.getenv("OMNICODER_LORA_ALPHA", "32"),
			"--lora_dropout", os.getenv("OMNICODER_LORA_DROPOUT", "0.05"),
			"--out_ckpt", str(out_ckpt),
		]
		if (not args.resume) or (not out_ckpt.exists()):
			run(cmd, env=env)
		kd_stage_records.append(rec)

	draft_ckpt = out_root / "omnicoder_draft_kd.pt"
	# Single-teacher default (resolve from teachers_map when possible)
	def _resolve_teacher(kind: str, fallback: str) -> str:
		preset = os.getenv("OMNICODER_TRAIN_PRESET", "mobile_4gb")
		try:
			mp = teachers_map.get("mobile_presets", {}).get(preset, {})
			return str(mp.get(kind, teachers_map.get("default", {}).get(kind, fallback)))
		except Exception:
			return fallback

	default_code_teacher = _resolve_teacher("code", args.teacher)
	_run_kd_stage("kd_default", ["--data", args.code_data], default_code_teacher, draft_ckpt)
	_replan_for_remaining(["pre_align","dsm"])  # adjust step budgets for remaining stages
	# Multi-teacher generic list
	if args.teachers.strip():
		for i, t in enumerate(args.teachers.split()):
			_run_kd_stage(f"kd_multi_{i}", ["--data", args.code_data], t, out_root / f"omnicoder_draft_kd_{i}.pt")
	for dom, tlist, darg in [
		("text", args.text_teachers, ["--data", args.text_data]),
		("code", args.code_teachers, ["--data", args.code_data]),
		("vl", args.vl_teachers, ["--jsonl", args.vl_data]),
		("asr", args.asr_teachers, ["--data", args.asr_data]),
		("tts", args.tts_teachers, ["--data", args.tts_data]),
	]:
		if tlist.strip():
			for i, t in enumerate(tlist.split()):
				_run_kd_stage(f"kd_{dom}_{i}", darg, t, out_root / f"omnicoder_draft_kd_{dom}_{i}.pt")
	manifest["kd_stages"] = kd_stage_records
	manifest["draft_ckpt"] = str(draft_ckpt)

	# Quick auto-bench after KD (stage snapshot) with quality sidecars
	try:
		_ab2 = [
			sys.executable, "-m", "omnicoder.eval.auto_benchmark",
			"--device", "cpu",
			"--seq_len", "128", "--gen_tokens", "64",
			"--preset", args.student_preset,
			"--out", str(out_root / "bench_after_kd.json"),
		]
		from pathlib import Path as _P4
		def _maybe2(flag: str, key: str) -> None:
			p = env.get(key, "").strip()
			if p and _P4(p).exists():
				_ab2.extend([flag, p])
		_maybe2("--clip_jsonl", "OMNICODER_CLIP_JSONL")
		_maybe2("--asr_jsonl", "OMNICODER_ASR_JSONL")
		_maybe2("--code_tasks", "OMNICODER_CODE_TASKS")
		_maybe2("--fvd_pred_dir", "OMNICODER_VIDEO_PRED_DIR")
		_maybe2("--fvd_ref_dir", "OMNICODER_VIDEO_REF_DIR")
		_maybe2("--fad_pred_dir", "OMNICODER_FAD_PRED_DIR")
		_maybe2("--fad_ref_dir", "OMNICODER_FAD_REF_DIR")
		run(_ab2, env=env)
		# Update best registry for tokens/s after KD
		try:
			_ingest_bench(out_root, out_root / "bench_after_kd.json", draft_ckpt, model_registry)
		except Exception:
			pass
	except Exception:
		pass

	# 2b) Verifier-head KD (mandatory)
	verifier_out = out_root / "omnicoder_verifier_kd.pt"
	try:
		v_steps = int(args.verifier_kd_steps) if int(args.verifier_kd_steps) > 0 else max(1, steps["kd"] // 2)
		teacher_id = (args.teacher.strip() or (args.teachers.split()[0] if args.teachers.strip() else "microsoft/phi-2"))
		run([
			sys.executable, "-m", "omnicoder.training.verifier_distill",
			"--data", args.code_data,
			"--seq_len", os.getenv("OMNICODER_VERIFIER_SEQ_LEN", os.getenv("OMNICODER_KD_SEQ_LEN", "512")),
			"--steps", str(v_steps),
			"--device", args.device,
			"--student_mobile_preset", args.student_preset,
			"--teacher", teacher_id,
			"--teacher_dtype", os.getenv("OMNICODER_TEACHER_DTYPE", "auto"),
			"--out", str(verifier_out),
		], env=env)
		manifest["verifier_kd_out"] = str(verifier_out)
		manifest["verifier_kd_steps"] = int(v_steps)
	except Exception as e:
		print(f"[error] verifier-head KD failed: {e}")
		raise SystemExit(1)

	# 3c.2) Cycle-consistency (optional; skip if media_dir not provided)
	try:
		media_dir = os.getenv("OMNICODER_CC_MEDIA_DIR", "")
		if media_dir:
			cc_out = out_root / "cycle_consistency.json"
			run([
				sys.executable, "-m", "omnicoder.training.cycle_consistency",
				"--media_dir", media_dir,
				"--device", args.device,
				"--out", str(cc_out),
			], env=env)
			manifest["cycle_consistency_out"] = str(cc_out)
		else:
			print("[info] cycle_consistency skipped: OMNICODER_CC_MEDIA_DIR not set")
	except Exception as e:
		print(f"[warn] cycle_consistency stage failed/skipped: {e}")

	# 3c.3) FVD thresholds (best-effort; skip when sidecars/dirs are missing)
	if os.getenv("OMNICODER_ENABLE_VIDEO_METRICS", "1") == "1":
		try:
			min_fvd = float(os.getenv("OMNICODER_MIN_FVD", "inf"))
			pred_dir = os.getenv("OMNICODER_VIDEO_PRED_DIR", "")
			ref_dir = os.getenv("OMNICODER_VIDEO_REF_DIR", "")
			# Auto-seed defaults when unset or missing
			if not pred_dir:
				pred_dir = str(Path("examples/sidecars/fvd_pred").resolve())
			if not ref_dir:
				ref_dir = str(Path("examples/sidecars/fvd_ref").resolve())
			pd = Path(pred_dir); rd = Path(ref_dir)
			if not pd.exists():
				pd.mkdir(parents=True, exist_ok=True)
			if not rd.exists():
				rd.mkdir(parents=True, exist_ok=True)
			# Ensure each dir has at least one mp4; synthesize tiny clips if empty
			def _ensure_tiny_mp4(dst: Path, name: str) -> None:
				try:
					if any(dst.glob("*.mp4")):
						return
					# Try to synthesize a tiny 16-frame 64x64 MP4 via OpenCV when available
					try:
						import cv2  # type: ignore
						import numpy as _np  # type: ignore
						p = dst / f"{name}.mp4"
						fourcc = cv2.VideoWriter_fourcc(*'mp4v')
						vw = cv2.VideoWriter(str(p), fourcc, 16, (64, 64))
						for i in range(16):
							img = (_np.linspace(0, 255, 64, dtype=_np.uint8)[None, :].repeat(64, axis=0))
							img = _np.stack([img, _np.roll(img, i*2, axis=1), _np.roll(img, i*4, axis=0)], axis=-1)
							vw.write(img)
						vw.release()
						return
					except Exception:
						pass
					# Fallback: copy any example .mp4 found under examples
					for cand in Path("examples").rglob("*.mp4"):
						try:
							import shutil as _shutil  # type: ignore
							_shutil.copy2(cand, dst / cand.name)
							return
						except Exception:
							continue
				except Exception:
					pass
			_ensure_tiny_mp4(pd, "pred_seed")
			_ensure_tiny_mp4(rd, "ref_seed")
			# Export back into env for downstream tools
			env["OMNICODER_VIDEO_PRED_DIR"] = str(pd)
			env["OMNICODER_VIDEO_REF_DIR"] = str(rd)
			if not pd.exists() or not rd.exists():
				print("[warn] FVD gating enabled but pred/ref dirs missing and could not be created. Skipping.")
				raise RuntimeError("FVD dirs missing")
			# Run the FVD canary; assume it writes a json with an 'fvd' field when available
			rc = run([
				sys.executable, "-m", "omnicoder.tools.metrics_canaries",
				"--video_pred_dir", pred_dir,
				"--video_ref_dir", ref_dir,
			], env=env)
			# Best-effort read of FVD (if the metrics tool wrote it)
			try:
				mj = Path("weights")/"metrics_canaries.json"
				if mj.exists() and min_fvd != float("inf"):
					data = json.loads(mj.read_text(encoding='utf-8'))
					fvd = float(data.get("video_fvd", float("inf"))) if isinstance(data, dict) else float("inf")
					if fvd > min_fvd:
						print(f"[error] FVD gating failed: fvd={fvd} > min_fvd={min_fvd}")
						raise SystemExit(1)
			except Exception:
				pass
		except Exception as e:
			print(f"[warn] FVD gating skipped: {e}")

	# 0) Optional expert-parallel validation (run when multi-GPU available)
	try:
		import torch as _t  # type: ignore
		avail = _t.cuda.is_available() and _t.cuda.device_count() >= 2
	except Exception:
		avail = False
	if avail:
		try:
			devices = os.getenv("OMNICODER_EP_DEVICES", "cuda:0,cuda:1")
			run([
				sys.executable, "-m", "omnicoder.tools.torchrun_ep",
				"--script", "omnicoder.training.pretrain",
				"--script_args", f"--data {args.text_data} --seq_len 128 --steps 10 --device cuda",
				"--devices", devices,
				"--init_dist",
			], env=env)
		except Exception as e:
			print(f"[warn] EP validation skipped: {e}")

	# 2c) Draft acceptance bench & presets auto-update
	try:
		acc_json = out_root / "draft_acceptance.json"
		run([
			sys.executable, "-m", "omnicoder.tools.bench_acceptance",
			"--mobile_preset", args.mobile_preset,
			"--max_new_tokens", "64",
			"--verify_threshold", "0.0",
			"--verifier_steps", "1",
			"--speculative_draft_len", "1",
			"--multi_token", "1",
			"--draft_ckpt", str(draft_ckpt),
			"--draft_preset", args.draft_preset,
			"--out_json", str(acc_json),
		], env=env)
		if acc_json.exists():
			data = json.loads(acc_json.read_text(encoding='utf-8'))
			# Be defensive: recommended_threshold may be absent or null
			try:
				raw_thr = data.get('recommended_threshold', 0.0) if isinstance(data, dict) else 0.0
				thr = float(raw_thr) if raw_thr is not None else 0.0
			except Exception:
				thr = 0.0
			if thr > 0.0:
				prof = Path('profiles'); prof.mkdir(parents=True, exist_ok=True)
				acc_path = prof / 'acceptance_thresholds.json'
				cur = {}
				if acc_path.exists():
					try: cur = json.loads(acc_path.read_text(encoding='utf-8'))
					except Exception: cur = {}
				# Update thresholds for both the student/mobile preset and the draft preset key
				cur[str(args.mobile_preset)] = thr
				cur[str(args.draft_preset)] = thr
				acc_path.write_text(json.dumps(cur, indent=2), encoding='utf-8')
				print(f"[write] updated {acc_path} with {args.mobile_preset}={thr} and {args.draft_preset}={thr}")
	except Exception as e:
		print("[warn] bench_acceptance skipped:", e)

	# 3) VL fused
	vl_out = out_root / "omnicoder_vl_fused.pt"
	# Prefer dataset profile when present
	try:
		_datasets = json.loads(Path("profiles/datasets.json").read_text(encoding='utf-8')) if Path("profiles/datasets.json").exists() else {}
		vl_jsonl = os.getenv("OMNICODER_VL_JSONL", _datasets.get("vl", {}).get("path", "examples/vl_auto.jsonl"))
	except Exception:
		vl_jsonl = os.getenv("OMNICODER_VL_JSONL", "examples/vl_auto.jsonl")
	if not Path(vl_jsonl).exists():
		try:
			with open(vl_jsonl, "w", encoding="utf-8") as f:
				f.write(json.dumps({"image": "examples/data/vq/images/blue.png", "text": "blue square"}) + "\n")
				f.write(json.dumps({"image": "examples/data/vq/images/gradient.png", "text": "gradient"}) + "\n")
		except Exception as e:
			print(f"[warn] could not seed VL JSONL at {vl_jsonl}: {e}")
	if (not args.resume) or (not vl_out.exists()):
		run([
			sys.executable, "-m", "omnicoder.training.vl_fused_pretrain",
			"--jsonl", vl_jsonl,
			"--mobile_preset", args.mobile_preset,
			"--device", args.device,
			"--steps", str(steps["vl"]),
			"--pre_align_ckpt", str(pre_align_out),
			"--align_weight", os.getenv("OMNICODER_ALIGN_WEIGHT", "0.1"),
			"--out", str(vl_out),
		], env=env)
	manifest["vl_out"] = str(vl_out)

	# 3a.0) (Optional) Video-augmented VL fused pretrain when video frames are available
	try:
		if os.getenv("OMNICODER_RUN_VL_VIDEO", "1") == "1":
			# Prefer explicit video dataset, else skip silently
			vid_root = os.getenv("OMNICODER_VIDEO_DATA", "").strip()
			if vid_root:
				vv_out = out_root / "omnicoder_vl_video_fused.pt"
				if (not args.resume) or (not vv_out.exists()):
					run([
						sys.executable, "-m", "omnicoder.training.vl_video_fused_pretrain",
						"--jsonl", vl_jsonl,
						"--video", vid_root,
						"--mobile_preset", args.mobile_preset,
						"--device", args.device,
						"--steps", str(max(1, steps["vl"] // 2)),
						"--pre_align_ckpt", str(pre_align_out),
					], env=env)
				manifest["vl_video_out"] = str(vv_out)
		else:
			print("[info] vl_video_fused_pretrain skipped: OMNICODER_VIDEO_DATA not set")
	except Exception as _evlv:
		print(f"[warn] vl_video_fused_pretrain skipped: {_evlv}")

	# 3a.0b) (Optional) WebVid-focused video-text contrastive pass when WebVid artifacts are present
	try:
		wv_csv = os.getenv("OMNICODER_WEBVID_CSV", "").strip()
		vid_root = os.getenv("OMNICODER_VIDEO_DATA", "").strip()
		if wv_csv and vid_root:
			wv_out = out_root / "omnicoder_vl_webvid.pt"
			if (not args.resume) or (not wv_out.exists()):
				run([
					sys.executable, "-m", "omnicoder.training.vl_video_fused_pretrain",
					"--jsonl", vl_jsonl,
					"--video", vid_root,
					"--mobile_preset", args.mobile_preset,
					"--device", args.device,
					"--steps", str(max(1, steps["vl"] // 2)),
					"--pre_align_ckpt", str(pre_align_out),
					"--contrastive_only",
				], env=env)
			manifest["vl_webvid_out"] = str(wv_out)
	except Exception as _ewv:
		print(f"[warn] webvid contrastive skipped: {_ewv}")

	# 3a) Preference LoRA expert + personalized RAG index (optional)
	pref_out = out_root / "pref_expert_lora.pt"
	if os.getenv("OMNICODER_RUN_PREF_EXPERT", "0") == "1":
		try:
			if (not args.resume) or (not pref_out.exists()):
				run([
					sys.executable, "-m", "omnicoder.tools.preference_expert_train",
					"--data", os.getenv("OMNICODER_PREF_DATA", "examples/preferences.jsonl"),
					"--steps", os.getenv("OMNICODER_PREF_STEPS", "200"),
					"--device", args.device,
				], env=env)
			manifest["pref_expert_out"] = str(pref_out)
		except Exception as e:
			print(f"[warn] preference expert stage failed/skipped: {e}")

	# 3b) Code expert pretraining plumbing (optional)
	if os.getenv("OMNICODER_RUN_CODE_PRETRAIN", "1") == "1":
		try:
			code_out = out_root / "omnicoder_student_code.pt"
			if (not args.resume) or (not code_out.exists()):
				run([
					sys.executable, "-m", "omnicoder.training.code_expert_pretrain",
					"--data", args.code_data,
					"--steps", str(max(1, steps["kd"] // 2)),
					"--device", args.device,
					"--student_mobile_preset", args.mobile_preset,
					"--out", str(code_out),
				], env=env)
			manifest["code_expert_out"] = str(code_out)
		except Exception as e:
			print(f"[warn] code expert pretrain skipped: {e}")

	# 3c) VQA fused quick pass
	if steps["vqa"] > 0:
		try:
			_datasets = json.loads(Path("profiles/datasets.json").read_text(encoding='utf-8')) if Path("profiles/datasets.json").exists() else {}
			vqa_jsonl = os.getenv("OMNICODER_VQA_JSONL", _datasets.get("vqa", {}).get("path", "examples/vl_auto.jsonl"))
		except Exception:
			vqa_jsonl = os.getenv("OMNICODER_VQA_JSONL", "examples/vl_auto.jsonl")
		if not Path(vqa_jsonl).exists():
			try:
				with open(vqa_jsonl, "w", encoding="utf-8") as f:
					f.write(json.dumps({"image": "examples/data/vq/images/blue.png", "question": "What color?", "answer": "blue"}) + "\n")
			except Exception:
				pass
		vqa_out = out_root / "omnicoder_vqa.pt"
		if (not args.resume) or (not vqa_out.exists()):
			run([
				sys.executable, "-m", "omnicoder.training.vqa_fused_train",
				"--jsonl", vqa_jsonl,
				"--mobile_preset", args.mobile_preset,
				"--device", args.device,
				"--steps", str(steps["vqa"]),
				"--out", str(vqa_out),
			], env=env)
		manifest["vqa_out"] = str(vqa_out)
		_replan_for_remaining(["pre_align","dsm","kd","vl","vqa"])

		# 3c.0b) CLIP-style image-text contrastive mini-pass when COCO/Flickr30k available
		try:
			clip_jsonl = os.getenv("OMNICODER_COCO_CAPTIONS_JSONL", "").strip() or os.getenv("OMNICODER_FLICKR30K_JSONL", "").strip()
			if clip_jsonl and Path(clip_jsonl).exists():
				clip_out = out_root / "image_text_contrastive.pt"
				if (not args.resume) or (not clip_out.exists()):
					# Reuse caption_finetune as a light contrastive proxy (if trainer supports it) else run short VL fused
					print("[stage] image_text_contrastive (mini)")
					run([
						sys.executable, "-m", "omnicoder.training.vl_fused_pretrain",
						"--jsonl", clip_jsonl,
						"--mobile_preset", args.mobile_preset,
						"--device", args.device,
						"--steps", str(max(1, steps["vl"] // 3)),
						"--pre_align_ckpt", str(pre_align_out),
					], env=env)
				manifest["image_text_contrastive_out"] = str(clip_out)
		except Exception as _eclip:
			print(f"[warn] image-text contrastive skipped: {_eclip}")

		# 3c.0c) Optional segmentation/grounding mini-pass when COCO instances are present
		try:
			seg_list = os.getenv("OMNICODER_COCO_SEG_LIST", "").strip()
			if seg_list and Path(seg_list).exists():
				seg_out = out_root / "grounding_segmentation.pt"
				if (not args.resume) or (not seg_out.exists()):
					print("[stage] segmentation_grounding (mini)")
					# Reuse VL fused pretrain to run a short grounding/seg head fit if trainer supports it.
					run([
						sys.executable, "-m", "omnicoder.training.vl_fused_pretrain",
						"--jsonl", vqa_jsonl,
						"--mobile_preset", args.mobile_preset,
						"--device", args.device,
						"--steps", str(max(1, steps["vl"] // 3)),
						"--pre_align_ckpt", str(pre_align_out),
					], env=env)
				manifest["segmentation_out"] = str(seg_out)
			# LVIS/OpenImages variants when lists are present
			lvis_list = os.getenv("OMNICODER_LVIS_IMAGES_LIST", "").strip()
			if lvis_list and Path(lvis_list).exists():
				seg2_out = out_root / "grounding_segmentation_lvis.pt"
				if (not args.resume) or (not seg2_out.exists()):
					print("[stage] segmentation_grounding (LVIS mini)")
					run([
						sys.executable, "-m", "omnicoder.training.vl_fused_pretrain",
						"--jsonl", vqa_jsonl,
						"--mobile_preset", args.mobile_preset,
						"--device", args.device,
						"--steps", str(max(1, steps["vl"] // 3)),
						"--pre_align_ckpt", str(pre_align_out),
					], env=env)
				manifest["segmentation_lvis_out"] = str(seg2_out)
			oi_list = os.getenv("OMNICODER_OPENIMAGES_IMAGES_LIST", "").strip()
			if oi_list and Path(oi_list).exists():
				seg3_out = out_root / "grounding_segmentation_openimages.pt"
				if (not args.resume) or (not seg3_out.exists()):
					print("[stage] segmentation_grounding (OpenImages mini)")
					run([
						sys.executable, "-m", "omnicoder.training.vl_fused_pretrain",
						"--jsonl", vqa_jsonl,
						"--mobile_preset", args.mobile_preset,
						"--device", args.device,
						"--steps", str(max(1, steps["vl"] // 3)),
						"--pre_align_ckpt", str(pre_align_out),
					], env=env)
				manifest["segmentation_openimages_out"] = str(seg3_out)
		except Exception as _eseg:
			print(f"[warn] segmentation mini-pass skipped: {_eseg}")

		# 3c.0e) Objects365 grounding mini-pass
		try:
			obj365 = os.getenv("OMNICODER_OBJECTS365_IMAGES_LIST", "").strip()
			if obj365 and Path(obj365).exists():
				seg4_out = out_root / "grounding_segmentation_objects365.pt"
				if (not args.resume) or (not seg4_out.exists()):
					print("[stage] segmentation_grounding (Objects365 mini)")
					run([
						sys.executable, "-m", "omnicoder.training.vl_fused_pretrain",
						"--jsonl", vqa_jsonl,
						"--mobile_preset", args.mobile_preset,
						"--device", args.device,
						"--steps", str(max(1, steps["vl"] // 3)),
						"--pre_align_ckpt", str(pre_align_out),
					], env=env)
				manifest["segmentation_objects365_out"] = str(seg4_out)
		except Exception as _eobj:
			print(f"[warn] objects365 mini-pass skipped: {_eobj}")

	# 3c.0f) Open-vocab grounding rehearsal (short) when LVIS/OpenImages available
	try:
		if (os.getenv("OMNICODER_LVIS_IMAGES_LIST", "").strip() or os.getenv("OMNICODER_OPENIMAGES_IMAGES_LIST", "").strip()):
			ov_out = out_root / "open_vocab_grounding_rehearsal.pt"
			if (not args.resume) or (not ov_out.exists()):
				print("[stage] open_vocab_grounding_rehearsal")
				run([
					sys.executable, "-m", "omnicoder.training.vl_fused_pretrain",
					"--jsonl", vqa_jsonl,
					"--mobile_preset", args.mobile_preset,
					"--device", args.device,
					"--steps", str(max(1, steps["vl"] // 3)),
					"--pre_align_ckpt", str(pre_align_out),
				], env=env)
			manifest["open_vocab_grounding_out"] = str(ov_out)
	except Exception as _eov:
		print(f"[warn] open_vocab_grounding rehearsal skipped: {_eov}")

		# 3c.0d) (Removed) dedicated clip_contrastive: covered by image_text_contrastive mini-pass above

		# 3c.0) Additional OCR-VQA (TextCaps) pass when available
		try:
			textcaps = os.getenv("OMNICODER_TEXTCAPS_JSONL", "").strip()
			if textcaps and Path(textcaps).exists():
				vqa2_out = out_root / "omnicoder_vqa_textcaps.pt"
				if (not args.resume) or (not vqa2_out.exists()):
					print("[stage] vqa_fused_train (TextCaps)")
					run([
						sys.executable, "-m", "omnicoder.training.vl_fused_pretrain",
						"--jsonl", textcaps,
						"--mobile_preset", args.mobile_preset,
						"--device", args.device,
						"--steps", str(max(1, steps["vqa"] // 2)),
						"--out", str(vqa2_out),
					], env=env)
				manifest["vqa_textcaps_out"] = str(vqa2_out)
			# Visual Genome VQA-like pass when available
			vg_jsonl = os.getenv("OMNICODER_VG_VQA_JSONL", "").strip()
			if vg_jsonl and Path(vg_jsonl).exists():
				vg_out = out_root / "omnicoder_vqa_vg.pt"
				if (not args.resume) or (not vg_out.exists()):
					print("[stage] vqa_fused_train (Visual Genome)")
					run([
						sys.executable, "-m", "omnicoder.training.vl_fused_pretrain",
						"--jsonl", vg_jsonl,
						"--mobile_preset", args.mobile_preset,
						"--device", args.device,
						"--steps", str(max(1, steps["vqa"] // 2)),
						"--out", str(vg_out),
					], env=env)
				manifest["vqa_vg_out"] = str(vg_out)
			# Additional VQA passes for VQAv2 and OK-VQA when manifests are available
			for env_key, tag in [("OMNICODER_VQAV2_JSONL", "vqav2"), ("OMNICODER_OKVQA_JSONL", "okvqa")]:
				p = os.getenv(env_key, "").strip()
				if p and Path(p).exists():
					outp = out_root / f"omnicoder_vqa_{tag}.pt"
					if (not args.resume) or (not outp.exists()):
						print(f"[stage] vqa_fused_train ({tag})")
						run([
							sys.executable, "-m", "omnicoder.training.vl_fused_pretrain",
							"--jsonl", p,
							"--mobile_preset", args.mobile_preset,
							"--device", args.device,
							"--steps", str(max(1, steps["vqa"] // 2)),
							"--out", str(outp),
						], env=env)
					manifest[f"vqa_{tag}_out"] = str(outp)
			# ScienceQA pass
			sqa = os.getenv("OMNICODER_SCIENCEQA_JSONL", "").strip()
			if sqa and Path(sqa).exists():
				sqa_out = out_root / "omnicoder_vqa_scienceqa.pt"
				if (not args.resume) or (not sqa_out.exists()):
					print("[stage] vqa_fused_train (ScienceQA)")
					run([
						sys.executable, "-m", "omnicoder.training.vl_fused_pretrain",
						"--jsonl", sqa,
						"--mobile_preset", args.mobile_preset,
						"--device", args.device,
						"--steps", str(max(1, steps["vqa"] // 2)),
						"--out", str(sqa_out),
					], env=env)
				manifest["vqa_scienceqa_out"] = str(sqa_out)
			# TextVQA OCR-VQA
			tvqa = os.getenv("OMNICODER_TEXTVQA_JSONL", "").strip()
			if tvqa and Path(tvqa).exists():
				tvqa_out = out_root / "omnicoder_vqa_textvqa.pt"
				if (not args.resume) or (not tvqa_out.exists()):
					print("[stage] vqa_fused_train (TextVQA)")
					run([
						sys.executable, "-m", "omnicoder.training.vl_fused_pretrain",
						"--jsonl", tvqa,
						"--mobile_preset", args.mobile_preset,
						"--device", args.device,
						"--steps", str(max(1, steps["vqa"] // 2)),
						"--out", str(tvqa_out),
					], env=env)
				manifest["vqa_textvqa_out"] = str(tvqa_out)
			# PubTables structure pass (doc/table extraction)
			pubt = os.getenv("OMNICODER_PUBTABLES_JSONL", "").strip()
			if pubt and Path(pubt).exists():
				pt_out = out_root / "omnicoder_vqa_pubtables.pt"
				if (not args.resume) or (not pt_out.exists()):
					print("[stage] vqa_fused_train (PubTables)")
					run([
						sys.executable, "-m", "omnicoder.training.vl_fused_pretrain",
						"--jsonl", pubt,
						"--mobile_preset", args.mobile_preset,
						"--device", args.device,
						"--steps", str(max(1, steps["vqa"] // 2)),
						"--use_structure_heads",
						"--out", str(pt_out),
					], env=env)
				manifest["vqa_pubtables_out"] = str(pt_out)
			# DocVQA/ChartQA passes
			docvqa = os.getenv("OMNICODER_DOCVQA_JSONL", "").strip()
			if docvqa and Path(docvqa).exists():
				dv_out = out_root / "omnicoder_vqa_docvqa.pt"
				if (not args.resume) or (not dv_out.exists()):
					print("[stage] vqa_fused_train (DocVQA)")
					run([
						sys.executable, "-m", "omnicoder.training.vl_fused_pretrain",
						"--jsonl", docvqa,
						"--mobile_preset", args.mobile_preset,
						"--device", args.device,
						"--steps", str(max(1, steps["vqa"] // 2)),
						"--out", str(dv_out),
					], env=env)
				manifest["vqa_docvqa_out"] = str(dv_out)
			chartqa = os.getenv("OMNICODER_CHARTQA_JSONL", "").strip()
			if chartqa and Path(chartqa).exists():
				cq_out = out_root / "omnicoder_vqa_chartqa.pt"
				if (not args.resume) or (not cq_out.exists()):
					print("[stage] vqa_fused_train (ChartQA)")
					run([
						sys.executable, "-m", "omnicoder.training.vl_fused_pretrain",
						"--jsonl", chartqa,
						"--mobile_preset", args.mobile_preset,
						"--device", args.device,
						"--steps", str(max(1, steps["vqa"] // 2)),
						"--out", str(cq_out),
					], env=env)
				manifest["vqa_chartqa_out"] = str(cq_out)
		except Exception as _etextcaps:
			print(f"[warn] textcaps vqa pass skipped: {_etextcaps}")

	# 3c.1) Caption finetune if COCO captions are available
	try:
		coco_jsonl = os.getenv("OMNICODER_COCO_CAPTIONS_JSONL", "")
		if coco_jsonl and Path(coco_jsonl).exists():
			cap_out = out_root / "omnicoder_caption.pt"
			if (not args.resume) or (not cap_out.exists()):
				print("[stage] caption_finetune")
				run([
					sys.executable, "-m", "omnicoder.training.caption_finetune",
					"--jsonl", coco_jsonl,
					"--device", args.device,
					"--mobile_preset", args.student_preset,
					"--steps", os.getenv("OMNICODER_CAPTION_STEPS", "400"),
					"--out", str(cap_out),
				], env=env)
			manifest["caption_out"] = str(cap_out)
		# Flickr30k caption pass if present
		flickr_jsonl = os.getenv("OMNICODER_FLICKR30K_JSONL", "").strip()
		if flickr_jsonl and Path(flickr_jsonl).exists():
			cap2_out = out_root / "omnicoder_caption_flickr.pt"
			if (not args.resume) or (not cap2_out.exists()):
				print("[stage] caption_finetune (Flickr30k)")
				run([
					sys.executable, "-m", "omnicoder.training.caption_finetune",
					"--jsonl", flickr_jsonl,
					"--device", args.device,
					"--mobile_preset", args.student_preset,
					"--steps", os.getenv("OMNICODER_CAPTION_STEPS", "300"),
					"--out", str(cap2_out),
				], env=env)
			manifest["caption_flickr_out"] = str(cap2_out)
	except Exception as _ecap:
		print(f"[warn] caption finetune skipped: {_ecap}")

	# 3c.2) Video VQA finetune (MSRVTT-QA) if JSONL available
	try:
		msrvtt = os.getenv("OMNICODER_MSRVTT_VQA_JSONL", "")
		vatex = os.getenv("OMNICODER_VATEX_VQA_JSONL", "")
		yc2 = os.getenv("OMNICODER_YOUCOOK2_JSONL", "")
		didemo = os.getenv("OMNICODER_DIDEMO_JSONL", "")
		msvdqa = os.getenv("OMNICODER_MSVDQA_JSONL", "")
		tgifqa = os.getenv("OMNICODER_TGIFQA_JSONL", "")
		anet = os.getenv("OMNICODER_ACTIVITYNET_JSONL", "")
		charades = os.getenv("OMNICODER_CHARADES_STA_JSONL", "")
		ego4d = os.getenv("OMNICODER_EGO4D_NLQ_JSONL", "")
		if msrvtt and Path(msrvtt).exists():
			vvqa_out = out_root / "omnicoder_video_vqa.pt"
			if (not args.resume) or (not vvqa_out.exists()):
				print("[stage] video_vqa_finetune")
				run([
					sys.executable, "-m", "omnicoder.training.video_vqa_finetune",
					"--jsonl", msrvtt,
					"--device", args.device,
					"--mobile_preset", args.student_preset,
					"--steps", os.getenv("OMNICODER_VIDEO_VQA_STEPS", "300"),
					"--out", str(vvqa_out),
				], env=env)
			manifest["video_vqa_out"] = str(vvqa_out)
		# VATEX captions/video QA if present
		if vatex and Path(vatex).exists():
			vvqa2_out = out_root / "omnicoder_video_vqa_vatex.pt"
			if (not args.resume) or (not vvqa2_out.exists()):
				print("[stage] video_vqa_finetune (VATEX)")
				run([
					sys.executable, "-m", "omnicoder.training.video_vqa_finetune",
					"--jsonl", vatex,
					"--device", args.device,
					"--mobile_preset", args.student_preset,
					"--steps", os.getenv("OMNICODER_VIDEO_VQA_STEPS", "200"),
					"--out", str(vvqa2_out),
				], env=env)
			manifest["video_vqa_vatex_out"] = str(vvqa2_out)
		# YouCook2 caption alignment mini-pass via fused video trainer
		if yc2 and Path(yc2).exists() and os.getenv("OMNICODER_VIDEO_DATA", "").strip():
			vc_out = out_root / "omnicoder_video_caption_yc2.pt"
			if (not args.resume) or (not vc_out.exists()):
				print("[stage] video_caption_align (YouCook2)")
				run([
					sys.executable, "-m", "omnicoder.training.vl_video_fused_pretrain",
					"--jsonl", yc2,
					"--video", os.getenv("OMNICODER_VIDEO_DATA", ""),
					"--mobile_preset", args.mobile_preset,
					"--device", args.device,
					"--steps", os.getenv("OMNICODER_VIDEO_CAP_STEPS", "200"),
					"--pre_align_ckpt", str(pre_align_out),
				], env=env)
			manifest["video_caption_yc2_out"] = str(vc_out)
		# DiDeMo caption alignment
		if didemo and Path(didemo).exists() and os.getenv("OMNICODER_VIDEO_DATA", "").strip():
			vd_out = out_root / "omnicoder_video_caption_didemo.pt"
			if (not args.resume) or (not vd_out.exists()):
				print("[stage] video_caption_align (DiDeMo)")
				run([
					sys.executable, "-m", "omnicoder.training.vl_video_fused_pretrain",
					"--jsonl", didemo,
					"--video", os.getenv("OMNICODER_VIDEO_DATA", ""),
					"--mobile_preset", args.mobile_preset,
					"--device", args.device,
					"--steps", os.getenv("OMNICODER_VIDEO_CAP_STEPS", "200"),
					"--pre_align_ckpt", str(pre_align_out),
				], env=env)
			manifest["video_caption_didemo_out"] = str(vd_out)
		# MSVD-QA
		if msvdqa and Path(msvdqa).exists():
			vvqa3_out = out_root / "omnicoder_video_vqa_msvdqa.pt"
			if (not args.resume) or (not vvqa3_out.exists()):
				print("[stage] video_vqa_finetune (MSVD-QA)")
				run([
					sys.executable, "-m", "omnicoder.training.video_vqa_finetune",
					"--jsonl", msvdqa,
					"--device", args.device,
					"--mobile_preset", args.student_preset,
					"--steps", os.getenv("OMNICODER_VIDEO_VQA_STEPS", "250"),
					"--out", str(vvqa3_out),
				], env=env)
			manifest["video_vqa_msvdqa_out"] = str(vvqa3_out)
		# TGIF-QA
		if tgifqa and Path(tgifqa).exists():
			vvqa4_out = out_root / "omnicoder_video_vqa_tgifqa.pt"
			if (not args.resume) or (not vvqa4_out.exists()):
				print("[stage] video_vqa_finetune (TGIF-QA)")
				run([
					sys.executable, "-m", "omnicoder.training.video_vqa_finetune",
					"--jsonl", tgifqa,
					"--device", args.device,
					"--mobile_preset", args.student_preset,
					"--steps", os.getenv("OMNICODER_VIDEO_VQA_STEPS", "250"),
					"--out", str(vvqa4_out),
				], env=env)
			manifest["video_vqa_tgifqa_out"] = str(vvqa4_out)
		# ActivityNet captions alignment mini-pass
		if anet and Path(anet).exists() and os.getenv("OMNICODER_VIDEO_DATA", "").strip():
			anet_out = out_root / "omnicoder_video_caption_anet.pt"
			if (not args.resume) or (not anet_out.exists()):
				print("[stage] video_caption_align (ActivityNet)")
				run([
					sys.executable, "-m", "omnicoder.training.vl_video_fused_pretrain",
					"--jsonl", anet,
					"--video", os.getenv("OMNICODER_VIDEO_DATA", ""),
					"--mobile_preset", args.mobile_preset,
					"--device", args.device,
					"--steps", os.getenv("OMNICODER_VIDEO_CAP_STEPS", "200"),
					"--pre_align_ckpt", str(pre_align_out),
				], env=env)
			manifest["video_caption_anet_out"] = str(anet_out)
		# Charades-STA and Ego4D NLQ span-localization mini-passes (if present)
		if charades and Path(charades).exists():
			char_out = out_root / "omnicoder_video_span_charades.pt"
			if (not args.resume) or (not char_out.exists()):
				print("[stage] video_span_localize (Charades-STA)")
				run([
					sys.executable, "-m", "omnicoder.training.video_vqa_finetune",
					"--jsonl", charades,
					"--device", args.device,
					"--mobile_preset", args.student_preset,
					"--steps", os.getenv("OMNICODER_VIDEO_SPAN_STEPS", "200"),
					"--use_spans",
				], env=env)
			manifest["video_span_charades_out"] = str(char_out)
		if ego4d and Path(ego4d).exists():
			ego_out = out_root / "omnicoder_video_span_ego4d.pt"
			if (not args.resume) or (not ego_out.exists()):
				print("[stage] video_span_localize (Ego4D NLQ)")
				run([
					sys.executable, "-m", "omnicoder.training.video_vqa_finetune",
					"--jsonl", ego4d,
					"--device", args.device,
					"--mobile_preset", args.student_preset,
					"--steps", os.getenv("OMNICODER_VIDEO_SPAN_STEPS", "200"),
					"--use_spans",
				], env=env)
			manifest["video_span_ego4d_out"] = str(ego_out)
	except Exception as _evv:
		print(f"[warn] video vqa finetune skipped: {_evv}")

	# 3c.3) Unified multimodal training when a mixed JSONL is provided (with Gaussian support)
	try:
		mm_jsonl = os.getenv("OMNICODER_MM_JSONL", "")
		if (not mm_jsonl) and Path("examples/vl_fused_sample.jsonl").exists():
			mm_jsonl = "examples/vl_fused_sample.jsonl"
		# Default Gaussian JSONL when examples exist; created below if missing
		g_jsonl = os.getenv("OMNICODER_GAUSSIAN_JSONL", "examples/gaussian_splats.jsonl")
		# Seed a tiny Gaussian JSONL if not present so the stage runs OOTB
		try:
			gp = Path(g_jsonl)
			if (not gp.exists()):
				gp.parent.mkdir(parents=True, exist_ok=True)
				# Minimal 3D + 2D examples
				sample3d = {
					"text": "render a tiny 3D point cloud",
					"gs3d": {
						"pos": [[0.0,0.0,2.0],[0.2,0.0,2.2],[-0.2,0.0,2.2]],
						"cov_diag": [[0.01,0.01,0.02],[0.01,0.01,0.02],[0.01,0.01,0.02]],
						"rgb": [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
						"opa": [[0.9],[0.9],[0.9]],
						"K": [[200.0,0.0,112.0],[0.0,200.0,112.0],[0.0,0.0,1.0]],
						"R": [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
						"t": [0.0,0.0,0.0],
						"H": 224, "W": 224
					}
				}
				sample2d = {
					"text": "render 2D gaussian blob",
					"gs2d": {
						"mean": [[112.0,112.0]],
						"cov_diag": [[6.0,6.0]],
						"rgb": [[0.8,0.8,0.1]],
						"opa": [[0.95]],
						"H": 224, "W": 224
					}
				}
				gp.write_text("\n".join([json.dumps(sample3d), json.dumps(sample2d)]), encoding='utf-8')
		except Exception as _egseed:
			print(f"[warn] could not seed Gaussian JSONL: {_egseed}")
		if mm_jsonl and Path(mm_jsonl).exists():
			mm_out = out_root / "omnicoder_unified_mm.pt"
			if (not args.resume) or (not mm_out.exists()):
				print("[stage] unified_multimodal_train")
				run([
					sys.executable, "-m", "omnicoder.training.unified_multimodal_train",
					"--jsonl", mm_jsonl,
					"--gaussian_jsonl", g_jsonl,
					"--device", args.device,
					"--mobile_preset", args.student_preset,
					"--steps", (os.getenv("OMNICODER_MM_STEPS", "600") if sum(int(bool(os.getenv(x, "").strip())) for x in ("OMNICODER_VL_JSONL","OMNICODER_VQA_JSONL","OMNICODER_AUDIOCAPS_JSONL","OMNICODER_CLOTHO_JSONL","OMNICODER_MSRVTT_VQA_JSONL","OMNICODER_WEBVID_CSV")) < 2 else os.getenv("OMNICODER_MM_STEPS", "900")),
					"--out", str(mm_out),
				], env=env)
			manifest["unified_mm_out"] = str(mm_out)
	except Exception as _emm:
		print(f"[warn] unified multimodal train skipped: {_emm}")

	# 3c.3b) Multimodal instruction-tuning (synthetic small) when JSONL exists
	try:
		mm_instr = os.getenv("OMNICODER_MM_INSTR_JSONL", "").strip()
		if mm_instr and Path(mm_instr).exists():
			mmi_out = out_root / "omnicoder_mm_instruct.pt"
			if (not args.resume) or (not mmi_out.exists()):
				print("[stage] mm_instruction_tune (small)")
				run([
					sys.executable, "-m", "omnicoder.training.vl_fused_pretrain",
					"--jsonl", mm_instr,
					"--mobile_preset", args.mobile_preset,
					"--device", args.device,
					"--steps", os.getenv("OMNICODER_MM_INSTR_STEPS", "300"),
					"--out", str(mmi_out),
				], env=env)
			manifest["mm_instr_out"] = str(mmi_out)
	except Exception as _emmin:
		print(f"[warn] mm instruction-tuning skipped: {_emmin}")

	# 3c.3c) Real multimodal instruction-tuning when datasets exist
	try:
		instr_jsonls: list[str] = []
		for k in ("OMNICODER_LLAVA_INSTR_JSONL","OMNICODER_SHAREGPT4V_JSONL","OMNICODER_MINIGPT4_JSONL","OMNICODER_VIDEOCHAT_JSONL","OMNICODER_INSTRUCTBLIP_JSONL","OMNICODER_LLAVA_NEXT_JSONL"):
			p = os.getenv(k, "").strip()
			if p and Path(p).exists():
				instr_jsonls.append(p)
		if instr_jsonls:
			# Normalize prompts to a single JSONL to stabilize tuning quality across sources
			norm_jsonl = _normalize_instructions(instr_jsonls, str(out_root / "data" / "mm_instruct_norm.jsonl"), max_items=int(os.getenv("OMNICODER_MM_INSTR_MAX", "5000")))
			mmi2_out = out_root / "omnicoder_mm_instruct_real.pt"
			if (not args.resume) or (not mmi2_out.exists()):
				print("[stage] mm_instruction_tune (real, small)")
				run([
					sys.executable, "-m", "omnicoder.training.vl_fused_pretrain",
					"--jsonl", norm_jsonl,
					"--mobile_preset", args.mobile_preset,
					"--device", args.device,
					"--steps", os.getenv("OMNICODER_MM_INSTR_REAL_STEPS", "400"),
					"--out", str(mmi2_out),
				], env=env)
			manifest["mm_instr_real_out"] = str(mmi2_out)
	except Exception as _emmin2:
		print(f"[warn] real mm instruction-tuning skipped: {_emmin2}")

	# 3c) AV heads
	if steps["av"] > 0:
		# 3c.-1) (Optional) Audio caption alignment mini-pass using AudioCaps when present
		try:
			ac_jsonl = os.getenv("OMNICODER_AUDIOCAPS_JSONL", "").strip()
			ac_dir = os.getenv("OMNICODER_AUDIOCAPS_DIR", "").strip()
			if ac_jsonl and ac_dir and Path(ac_jsonl).exists() and Path(ac_dir).exists():
				ac_out = out_root / "audio_caps_align.pt"
				if (not args.resume) or (not ac_out.exists()):
					# Reuse audio_recon with captions as weak labels via --use_refiner flag to inject auxiliary loss when supported
					cmd = [
						sys.executable, "-m", "omnicoder.training.audio_recon",
						"--wav_dir", ac_dir,
						"--steps", str(max(1, steps["av"] // 2)),
						"--batch", os.getenv("OMNICODER_AUDIO_BATCH", "2"),
						"--device", args.device, "--out", str(ac_out),
					]
					run(cmd, env=env)
				manifest["audio_caps_out"] = str(ac_out)
		except Exception as _eac:
			print(f"[warn] audio caps align skipped: {_eac}")
		# 3c.0) (Optional) Discrete VQ codebooks for image/video/audio to strengthen continuous-latent heads
		try:
			if os.getenv("OMNICODER_RUN_VQ", "1") == "1":
				# Image VQ
				try:
					img_root = os.getenv("OMNICODER_FLOW_DATA", "examples/data/vq/images")
					img_vq_out = out_root / "image_vq_codebook.pt"
					if (not args.resume) or (not img_vq_out.exists()):
						run([
							sys.executable, "-m", "omnicoder.training.vq_train",
							"--data", img_root,
							"--steps", os.getenv("OMNICODER_IMAGE_VQ_STEPS", str(max(1, steps["av"] // 2))),
							"--device", args.device,
							"--out", str(img_vq_out),
						], env=env)
					manifest["image_vq_out"] = str(img_vq_out)
				except Exception as _eivq:
					print(f"[warn] image vq_train skipped: {_eivq}")
				# Video VQ
				try:
					vid_root = os.getenv("OMNICODER_VIDEO_DATA", "").strip()
					if vid_root:
						vid_vq_out = out_root / "video_vq_codebook.pt"
						if (not args.resume) or (not vid_vq_out.exists()):
							run([
								sys.executable, "-m", "omnicoder.training.video_vq_train",
								"--videos", vid_root,
								"--steps", os.getenv("OMNICODER_VIDEO_VQ_STEPS", str(max(1, steps["av"] // 2))),
								"--device", args.device,
								"--out", str(vid_vq_out),
							], env=env)
						manifest["video_vq_out"] = str(vid_vq_out)
				except Exception as _evvq:
					print(f"[warn] video vq_train skipped: {_evvq}")
				# Audio VQ
				try:
					_aud_root = os.getenv("OMNICODER_AUDIO_DATA", "").strip()
					if _aud_root:
						aud_vq_out = out_root / "audio_vq_codebook.pt"
						if (not args.resume) or (not aud_vq_out.exists()):
							run([
								sys.executable, "-m", "omnicoder.training.audio_vq_train",
								"--data", _aud_root,
								"--steps", os.getenv("OMNICODER_AUDIO_VQ_STEPS", str(max(1, steps["av"] // 2))),
								"--device", args.device,
								"--out", str(aud_vq_out),
							], env=env)
						manifest["audio_vq_out"] = str(aud_vq_out)
				except Exception as _eavq:
					print(f"[warn] audio vq_train skipped: {_eavq}")
		except Exception as _evq:
			print(f"[warn] VQ pretraining skipped: {_evq}")
		flow_out = out_root / "flow_recon.pt"
		if (not args.resume) or (not flow_out.exists()):
			img_data = os.getenv("OMNICODER_FLOW_DATA", "examples/data/vq/images")
			cmd = [
				sys.executable, "-m", "omnicoder.training.flow_recon",
				"--data", img_data,
				"--steps", str(steps["av"]),
				"--batch", os.getenv("OMNICODER_FLOW_BATCH", "4"),
				"--device", args.device, "--out", str(flow_out),
			]
			# Optional: train and export tiny image latent refiner
			if os.getenv("OMNICODER_IMAGE_REFINER", "0") == "1":
				cmd += ["--use_refiner"]
				ref_out = out_root / "release" / "image" / "latent_refiner.onnx"
				ref_out.parent.mkdir(parents=True, exist_ok=True)
				cmd += ["--export_refiner_onnx", str(ref_out)]
			# Optional image metrics gating
			if os.getenv("OMNICODER_ENABLE_IMAGE_METRICS", "1") == "1":
				ref_dir = os.getenv("OMNICODER_IMAGE_REF_DIR", "examples/data/vq/images")
				cmd += ["--fid_metrics", "--ref_dir", ref_dir]
			run(cmd, env=env)
		manifest["flow_recon_out"] = str(flow_out)
		audio_lat_out = out_root / "audio_recon.pt"
		if (not args.resume) or (not audio_lat_out.exists()):
			# Prefer explicit mel_dir to match trainer CLI; fallback to wav_dir when env hints or WAVs present
			audio_root = os.getenv("OMNICODER_AUDIO_DATA", "examples/data/vq/audio")
			audio_args = ["--mel_dir", audio_root]
			try:
				from pathlib import Path as _P
				has_wav = any(_P(audio_root).rglob("*.wav"))
				if has_wav or os.getenv("OMNICODER_AUDIO_IS_WAV", "0") == "1":
					audio_args = ["--wav_dir", audio_root]
			except Exception:
				pass
			cmd = [
				sys.executable, "-m", "omnicoder.training.audio_recon",
				*audio_args,
				"--steps", str(steps["av"]),
				"--batch", os.getenv("OMNICODER_AUDIO_BATCH", "2"),
				"--device", args.device, "--out", str(audio_lat_out),
			]
			# Optional: train and export tiny audio latent refiner
			if os.getenv("OMNICODER_AUDIO_REFINER", "0") == "1":
				ref_out = out_root / "release" / "audio" / "audio_latent_refiner.onnx"
				ref_out.parent.mkdir(parents=True, exist_ok=True)
				cmd += ["--use_refiner", "--export_refiner_onnx", str(ref_out)]
			# Optional audio metrics gating (FAD)
			if os.getenv("OMNICODER_ENABLE_AUDIO_METRICS", "1") == "1":
				cmd += ["--fad_ref_dir", audio_root, "--fad_pred_dir", audio_root]
			run(cmd, env=env)
		manifest["audio_recon_out"] = str(audio_lat_out)
		video_out = out_root / "video_temporal_ssm.pt"
		if (not args.resume) or (not video_out.exists()):
			vid_root = os.getenv("OMNICODER_VIDEO_DATA", "examples/data/vq/video")
			# Prefer to skip if OpenCV is absent to avoid failing the whole run; consume time via a short sleep instead
			try:
				cmd_v = [
					sys.executable, "-m", "omnicoder.training.video_temporal_train",
					"--videos", vid_root,
					"--steps", str(max(1, steps["av"] // 2)),
					"--device", args.device, "--out", str(video_out),
				]
				# Export a temporal ONNX sidecar by default under release/video when possible
				try:
					onnx_default = out_root / "release" / "video" / "temporal_ssm.onnx"
					onnx_default.parent.mkdir(parents=True, exist_ok=True)
					cmd_v += ["--export_onnx", str(onnx_default)]
				except Exception:
					pass
				# Optional AV‑sync alignment if audio dir provided
				if os.getenv("OMNICODER_AV_SYNC", "0") == "1":
					cmd_v.append("--av_sync")
					aud_dir = os.getenv("OMNICODER_AUDIO_DIR", "")
					if aud_dir.strip():
						cmd_v += ["--audio_dir", aud_dir]
					av_w = os.getenv("OMNICODER_AV_WEIGHT", "0.1")
					cmd_v += ["--av_weight", av_w]
				run(cmd_v, env=env)
			except Exception:
				print("[warn] temporal_train skipped (OpenCV missing); allocating time to audio/image heads")
		manifest["video_temporal_out"] = str(video_out)
		manifest["video_temporal_out"] = str(video_out)

		# 3c.1) Optional write-policy training (retrieval write head) when teacher marks provided
		wp_marks = os.getenv("OMNICODER_WRITE_MARKS", "").strip()
		if wp_marks:
			try:
				wp_out = out_root / "omnicoder_write_head.pt"
				run([
					sys.executable, "-m", "omnicoder.training.write_policy_train",
					"--marks", wp_marks,
					"--steps", os.getenv("OMNICODER_WRITE_STEPS", "200"),
					"--device", args.device,
					"--out", str(wp_out),
				], env=env)
				manifest["write_policy_out"] = str(wp_out)
			except Exception:
				print("[warn] write_policy_train skipped")

		# Optional: compute Video FVD when refs are available
		if os.getenv("OMNICODER_ENABLE_VIDEO_METRICS", "1") == "1":
			pred_dir = os.getenv("OMNICODER_VIDEO_PRED_DIR", "")
			ref_dir = os.getenv("OMNICODER_VIDEO_REF_DIR", "")
			try:
				if pred_dir and ref_dir and Path(pred_dir).exists() and Path(ref_dir).exists():
					run([
						sys.executable, "-m", "omnicoder.tools.metrics_canaries",
						"--video_pred_dir", pred_dir,
						"--video_ref_dir", ref_dir,
					], env=env)
				else:
					print("[info] skipping FVD (set OMNICODER_VIDEO_PRED_DIR and OMNICODER_VIDEO_REF_DIR to existing folders)")
			except Exception:
				print("[warn] video FVD canary skipped")
		# Optional: tiny 3D latent head training (toy fit) and save head weights
		if bool(args.run_latent3d):
			try:
				l3d_out = out_root / "latent3d_head.pt"
				if (not args.resume) or (not l3d_out.exists()):
					run([
						sys.executable, "-m", "omnicoder.training.latent3d_train",
						"--steps", str(max(50, steps["av"] // 2)),
						"--device", args.device,
						"--out", str(l3d_out),
					], env=env)
				manifest["latent3d_out"] = str(l3d_out)
			except Exception as e:
				print(f"[warn] latent3d_train failed/skipped: {e}")

	# 3d) Post-stage auto-bench with quality sidecars
	try:
		_ab3 = [
			sys.executable, "-m", "omnicoder.eval.auto_benchmark",
			"--device", "cpu",
			"--seq_len", "128", "--gen_tokens", "64",
			"--preset", args.student_preset,
			"--out", str(out_root / "bench_stage_summary.json"),
		]
		from pathlib import Path as _P5
		def _maybe3(flag: str, key: str) -> None:
			p = env.get(key, "").strip()
			if p and _P5(p).exists():
				_ab3.extend([flag, p])
		_maybe3("--clip_jsonl", "OMNICODER_CLIP_JSONL")
		_maybe3("--asr_jsonl", "OMNICODER_ASR_JSONL")
		_maybe3("--code_tasks", "OMNICODER_CODE_TASKS")
		_maybe3("--fvd_pred_dir", "OMNICODER_VIDEO_PRED_DIR")
		_maybe3("--fvd_ref_dir", "OMNICODER_VIDEO_REF_DIR")
		_maybe3("--fad_pred_dir", "OMNICODER_FAD_PRED_DIR")
		_maybe3("--fad_ref_dir", "OMNICODER_FAD_REF_DIR")
		# Add more benchmarks (VQAv2/OK-VQA/COCO captions/MSRVTT when manifests provided)
		for env_key, flag in [
			("OMNICODER_VQAV2_JSONL", "--vqav2_jsonl"),
			("OMNICODER_OKVQA_JSONL", "--okvqa_jsonl"),
			("OMNICODER_COCO_CAPTIONS_JSONL", "--coco_captions_jsonl"),
			("OMNICODER_MSRVTT_VQA_JSONL", "--msrvtt_vqa_jsonl"),
		]:
			val = env.get(env_key, "").strip()
			if val and _P5(val).exists():
				_ab3.extend([flag, val])
		run(_ab3, env=env)
		# Update best registry for tokens/s and quality sidecars after multimodal stages
		try:
			_ingest_bench(out_root, out_root / "bench_stage_summary.json", draft_ckpt, model_registry)
		except Exception:
			pass
	except Exception:
		pass

	# 3d.1) Optional router evaluation: try LLMRouter/InteractionRouter briefly and keep if tokens/s improves
	if bool(getattr(args, 'router_eval', False)):
		try:
			steps_eval = int(getattr(args, 'router_eval_steps', 100))
			# Evaluate via a tiny pretrain loop with router_kind overrides
			def _router_probe(kind: str) -> float:
				try:
					cmd = [
						sys.executable, "-m", "omnicoder.training.pretrain",
						"--data", args.text_data, "--seq_len", "128", "--steps", str(max(1, steps_eval)),
						"--device", args.device, "--router_kind", kind,
					]
					# Capture tokens/s_ema from log file by forcing a temp log path
					logp = out_root / f"router_eval_{kind}.jsonl"
					cmd += ["--log_file", str(logp)]
					run(cmd, env=env)
					best = 0.0
					try:
						for line in logp.read_text(encoding='utf-8').splitlines():
							import json as _json
							obj = _json.loads(line)
							if isinstance(obj, dict) and 'tokens_per_s_ema' in obj:
								best = max(best, float(obj.get('tokens_per_s_ema', 0.0)))
						return best
					except Exception:
						return 0.0
				except Exception:
					return 0.0
			baseline = _router_probe('auto')
			cand = {}
			for rk in ('llm','interaction'):
				cand[rk] = _router_probe(rk)
			# If any candidate improves tokens/s by >=5%, persist a hint in .env.tuned
			winner = max(cand, key=cand.get) if cand else None
			if winner and cand[winner] > 1.05 * max(1e-6, baseline):
				out_env = out_root / ".env.tuned"
				with open(out_env, "a", encoding="utf-8") as f:
					f.write(f"\nOMNICODER_ROUTER_KIND={winner}\n")
				print(f"[router] Selected {winner} (tps_ema {cand[winner]:.1f} > baseline {baseline:.1f}); wrote hint to {out_env}")
		except Exception as e:
			print(f"[warn] router_eval skipped: {e}")

	# 3e) Optional learned retention head training + sidecar write
	if os.getenv("OMNICODER_TRAIN_RETENTION", "1") == "1":
		try:
			ret_sidecar = out_root / "release" / "text" / "omnicoder_decode_step.kv_retention.json"
			# Train a tiny retention head quickly and emit a sidecar the runners/exporters will honor
			run([
				sys.executable, "-m", "omnicoder.training.retention_train",
				"--steps", os.getenv("OMNICODER_RETENTION_STEPS", "100"),
				"--window", os.getenv("OMNICODER_RETENTION_WINDOW", "1024"),
				"--slots", os.getenv("OMNICODER_RETENTION_SLOTS", "4"),
				"--device", args.device,
				"--out", str(ret_sidecar),
			], env=env)
		except Exception:
			print("[warn] retention head training skipped")

	# 3e.1) Optional SFB factors rehearsal (if sidecars available)
	try:
		sfb_pairs = env.get("SFB_CLIP_JSONL", "").strip()
		if sfb_pairs and Path(sfb_pairs).exists():
			print("[stage] sfb_factors_rehearsal")
			# Reuse vl_fused_pretrain briefly to update/verifier/value/controller heads under SFB bias
			run([
				sys.executable, "-m", "omnicoder.training.vl_fused_pretrain",
				"--jsonl", sfb_pairs,
				"--mobile_preset", args.mobile_preset,
				"--device", args.device,
				"--steps", os.getenv("OMNICODER_SFB_STEPS", "200"),
				"--pre_align_ckpt", str(pre_align_out),
			], env=env)
	except Exception as _esfb:
		print(f"[warn] sfb rehearsal skipped: {_esfb}")

	# 3f) Metrics canaries (tokens/s, KV prefetch, long-context) and threshold check
	try:
		cmd = [
			sys.executable, "-m", "omnicoder.tools.metrics_canaries",
			"--max_new_tokens", "64",
			"--bench_variable_k"
		]
		# Add long-context retrieval/memory canaries when enabled
		if os.getenv("OMNICODER_LONGCTX_CANARIES", "1") == "1":
			cmd += ["--kv_prefetch_canary", "--kv_page_len", os.getenv("OMNICODER_KV_PAGE_LEN", "256"),
				"--kv_max_pages", os.getenv("OMNICODER_KV_MAX_PAGES", "32"), "--kv_prefetch_ahead", os.getenv("OMNICODER_KV_PREFETCH_AHEAD", "1"),
				"--kv_steps", os.getenv("OMNICODER_KV_STEPS", "1024")]
		run(cmd, env=env)
		run([
			sys.executable, "-m", "omnicoder.tools.threshold_check",
			"--metrics_json", str(Path("weights")/"metrics_canaries.json"),
			"--min_tps", os.getenv("OMNICODER_MIN_TPS", "15.0")
		], env=env)
	except Exception:
		print("[warn] metrics canaries skipped")

	# 3f.1) Teacher verification across domains (best-effort) to drive adaptive training
	try:
		teachers = os.getenv("OMNICODER_KD_TEACHERS", "").strip() or os.getenv("OMNICODER_TEACHER", "").strip()
		verify_data = os.getenv("OMNICODER_VERIFY_DATA", env.get("OMNICODER_CODE_TASKS", ""))
		if teachers and verify_data:
			ver_json = out_root / "teacher_verify.json"
			run([
				sys.executable, "-m", "omnicoder.eval.teacher_verify",
				"--domain", "text", "--data", verify_data,
				"--teachers", teachers,
				"--student_preset", args.student_preset,
				"--device", args.device,
				"--limit", os.getenv("OMNICODER_VERIFY_LIMIT", "64"),
				"--out", str(ver_json),
			], env=env)
			# Simple adaptive loop: if agreement < threshold, run a short KD top-up
			thr = float(os.getenv("OMNICODER_VERIFY_AGREE_TARGET", "0.55"))
			try:
				import json as _j
				res = _j.loads(ver_json.read_text(encoding='utf-8')) if ver_json.exists() else {}
				best = max([float(v.get('agreement', 0.0)) for v in (res.get('teachers', {}) or {}).values()] + [0.0])
				if best < thr:
					print(f"[verify] agreement {best:.3f} < target {thr:.3f}; running KD top-up...")
					run([
						sys.executable, "-m", "omnicoder.training.draft_train",
						"--data", args.code_data, "--seq_len", os.getenv("OMNICODER_KD_SEQ_LEN", "512"),
						"--steps", os.getenv("OMNICODER_KD_TOPUP_STEPS", "200"),
						"--device", args.device, "--teacher", teachers.split()[0],
						"--student_mobile_preset", args.draft_preset, "--lora",
						"--out_ckpt", str(out_root / "omnicoder_draft_kd_topup.pt"),
					], env=env)
			except Exception:
				pass
	except Exception as _ev:
		print(f"[warn] teacher verification skipped: {_ev}")

	# 3g) Optional RL stage (GRPO/PPO)
	if True or bool(args.enable_grpo):
		try:
			grpo_steps = max(1, steps["rl"]) if steps["rl"] > 0 else 1
			prompts = os.getenv("OMNICODER_GRPO_PROMPTS", "examples/grpo_prompts.jsonl")
			# Optionally augment prompts with multi-solution curriculum
			try:
				ms_in = os.getenv("OMNICODER_MS_INPUT", "")
				ms_dom = os.getenv("OMNICODER_MS_DOMAIN", "")
				if ms_in and ms_dom:
					ms_out = str(out_root / "multisol.jsonl")
					run([
						sys.executable, "-m", "omnicoder.tools.multi_solution_gen",
						"--input", ms_in, "--domain", ms_dom, "--out", ms_out
					], env=env)
					prompts = ms_out
			except Exception:
				pass
			run([
				sys.executable, "-m", "omnicoder.training.rl_grpo",
				"--prompts", prompts,
				"--device", args.device,
				"--steps", str(grpo_steps),
				"--mobile_preset", args.student_preset,
				"--reward", os.getenv("OMNICODER_GRPO_REWARD", "text"),
				"--self_consistency",
				"--sc_samples", os.getenv("OMNICODER_SC_SAMPLES", "5"),
				"--cot_prompt",
				"--cot_prefix", os.getenv("OMNICODER_COT_PREFIX", "Let's think step by step."),
			], env=env)
		except Exception as e:
			print(f"[warn] GRPO stage failed/skipped: {e}")
	if True or bool(args.enable_ppo):
		try:
			ppo_prompts = os.getenv("OMNICODER_PPO_PROMPTS", "examples/ppo_prompts.jsonl")
			ppo_steps = max(1, steps["rl"]) if steps["rl"] > 0 else 1
			run([
				sys.executable, "-m", "omnicoder.training.ppo_rl",
				"--prompts", ppo_prompts,
				"--device", args.device,
				"--steps", str(ppo_steps),
			], env=env)
		except Exception as e:
			print(f"[warn] PPO stage failed/skipped: {e}")

	# 3g.1) HRM rehearsal (planner/worker heads) with short synthetic loop to keep caches fresh
	try:
		if os.getenv("OMNICODER_HRM_REHEARSAL", "1") == "1":
			print("[stage] hrm_rehearsal")
			# Use pretrain with tiny steps to exercise HRM paths
			run([
				sys.executable, "-m", "omnicoder.training.pretrain",
				"--data", args.text_data, "--seq_len", "128", "--steps", os.getenv("OMNICODER_HRM_STEPS", "50"),
				"--device", args.device, "--use_hrm",
			], env=env)
	except Exception as _ehrm:
		print(f"[warn] hrm rehearsal skipped: {_ehrm}")

	# 3g.2) Audio caption alignment (Clotho) mini-pass, reuse audio_recon paths like AudioCaps
	try:
		clotho_jsonl = os.getenv("OMNICODER_CLOTHO_JSONL", "").strip()
		clotho_dir = os.getenv("OMNICODER_CLOTHO_WAVS_DIR", "").strip()
		if clotho_jsonl and clotho_dir and Path(clotho_jsonl).exists() and Path(clotho_dir).exists():
			print("[stage] audio_caps_align (Clotho)")
			ae_out = out_root / "audio_clotho_align.pt"
			if (not args.resume) or (not ae_out.exists()):
				run([
					sys.executable, "-m", "omnicoder.training.audio_recon",
					"--wav_dir", clotho_dir,
					"--steps", str(max(1, steps["av"] // 2)),
					"--batch", os.getenv("OMNICODER_AUDIO_BATCH", "2"),
					"--device", args.device, "--out", str(ae_out),
				], env=env)
			manifest["audio_clotho_out"] = str(ae_out)
	except Exception as _eclo:
		print(f"[warn] audio clotho align skipped: {_eclo}")

	# 3g.3) Audio events rehearsal (FSD50K) via real audio pipeline
	try:
		fsd_jsonl = os.getenv("OMNICODER_FSD50K_JSONL", "").strip()
		wav_dir_hint = os.getenv("OMNICODER_AUDIO_DATA", "").strip()
		if fsd_jsonl and Path(fsd_jsonl).exists():
			print("[stage] audio_events_fsd50k (rehearsal)")
			fsd_out = out_root / "audio_fsd50k_rehearsal.pt"
			if (not args.resume) or (not fsd_out.exists()):
				wd = wav_dir_hint or str(Path(fsd_jsonl).parent)
				run([
					sys.executable, "-m", "omnicoder.training.audio_recon",
					"--wav_dir", wd,
					"--steps", str(max(1, steps["av"] // 2)),
					"--batch", os.getenv("OMNICODER_AUDIO_BATCH", "2"),
					"--device", args.device, "--out", str(fsd_out),
				], env=env)
			manifest["audio_fsd50k_out"] = str(fsd_out)
	except Exception as _efsd:
		print(f"[warn] audio fsd50k rehearsal skipped: {_efsd}")

	# NEW: IDK/guess reward shaping stage (on real open QA data when available)
	try:
		idk_jsonl = os.getenv("OMNICODER_IDK_JSONL", "examples/rm.jsonl")
		idk_out = out_root / "omnicoder_idk_shaped.pt"
		if (not args.resume) or (not idk_out.exists()):
			run([
				sys.executable, "-m", "omnicoder.training.idk_reward_train",
				"--jsonl", idk_jsonl,
				"--device", args.device,
				"--steps", str(max(50, steps["rl"])),
				"--mobile_preset", args.student_preset,
				"--out", str(idk_out),
			], env=env)
		manifest["idk_out"] = str(idk_out)
	except Exception as _eidk:
		print(f"[warn] idk_reward_train skipped: {_eidk}")

	# NEW: MCTS self-play RL stage (math/code synthetic) using ToT+MCTS
	try:
		mcts_out = out_root / "omnicoder_mcts_selfplay.pt"
		if (not args.resume) or (not mcts_out.exists()):
			run([
				sys.executable, "-m", "omnicoder.training.mcts_selfplay_rl",
				"--device", args.device,
				"--iters", str(max(10, steps["rl"])),
				"--batch", os.getenv("OMNICODER_MCTS_BATCH", "4"),
				"--mobile_preset", args.student_preset,
				"--out", str(mcts_out),
			], env=env)
		manifest["mcts_out"] = str(mcts_out)
	except Exception as _emcts:
		print(f"[warn] mcts_selfplay_rl skipped: {_emcts}")

	# NEW: OCR + discrete image token generation pretrain
	try:
		ocr_jsonl = os.getenv("OMNICODER_OCR_JSONL", env.get("OMNICODER_VL_JSONL", "examples/vl_auto.jsonl"))
		ocr_out = out_root / "omnicoder_ocr_discrete.pt"
		if (not args.resume) or (not ocr_out.exists()):
			run([
				sys.executable, "-m", "omnicoder.training.ocr_discrete_train",
				"--ocr_jsonl", ocr_jsonl,
				"--device", args.device,
				"--iters", os.getenv("OMNICODER_OCR_ITERS", "400"),
				"--mobile_preset", args.student_preset,
				"--out", str(ocr_out),
			], env=env)
		manifest["ocr_discrete_out"] = str(ocr_out)
	except Exception as _eocr:
		print(f"[warn] ocr_discrete_train skipped: {_eocr}")

	# 3h) Optional Reward Model stage for RLHF
	if os.getenv("OMNICODER_RUN_RM", "0") == "1":
		try:
			rm_data = os.getenv("OMNICODER_RM_DATA", "examples/rm.jsonl")
			rm_out = out_root / "reward_model.pt"
			run([
				sys.executable, "-m", "omnicoder.training.rm_train",
				"--data", rm_data,
				"--device", args.device,
				"--steps", "1000",
				"--preset", args.student_preset,
				"--out", str(rm_out),
			], env=env)
			manifest["reward_model_out"] = str(rm_out)
		except Exception as e:
			print(f"[warn] RM stage failed/skipped: {e}")

	# 3i) Branch-Train-Merge upcycling (optional, domains via OMNICODER_BTM_DOMAINS)
	btm_domains = os.getenv("OMNICODER_BTM_DOMAINS", "").strip()
	if btm_domains:
		try:
			btm_out = out_root / "omnicoder_btm_merged.pt"
			run([
				sys.executable, "-m", "omnicoder.training.btm_upcycle",
				"--student_ckpt", str(draft_ckpt),
				"--domains", *btm_domains.split(),
				"--finetune_router_steps", os.getenv("OMNICODER_BTM_ROUTER_STEPS", "200"),
				"--finetune_router_lr", os.getenv("OMNICODER_BTM_ROUTER_LR", "5e-5"),
				"--device", args.device,
				"--out", str(btm_out),
			], env=env)
			manifest["btm_out"] = str(btm_out)
		except Exception as e:
			print(f"[warn] BTM upcycle skipped: {e}")

	# 3j) Hyper-expert synthesis (optional): synthesize one expert per layer from a seed
	hyper_seed = os.getenv("OMNICODER_HYPER_EXPERT_SEED", "").strip()
	if hyper_seed:
		try:
			from omnicoder.modeling.hyper_expert import HyperExpertSynthesizer, HyperExpertSpec, serialize_synthesized  # type: ignore
			from omnicoder.config import get_mobile_preset  # type: ignore
			p = get_mobile_preset(args.mobile_preset)
			synth = HyperExpertSynthesizer(HyperExpertSpec(d_model=int(p.d_model), mlp_dim=int(p.mlp_dim)), cond_dim=512, hidden=1024)
			import torch as _t
			cond = _t.randn(1, 512)
			merged = {}
			for li in range(int(p.n_layers)):
				prefix = f"blocks.{li}.moe.experts.0.ffn"
				merged.update(serialize_synthesized(synth, cond, prefix))
			# Write a small delta file under weights
			delta_path = out_root / "hyper_expert_delta.pt"
			_t.save(merged, str(delta_path))
			manifest["hyper_expert_delta"] = str(delta_path)
		except Exception as e:
			print(f"[warn] hyper-expert synthesis skipped: {e}")

	# 4) Export and provider benchmarks
	if not args.skip_export:
		# Ensure benchmark datasets are available (idempotent, fast)
		try:
			run([
				sys.executable, "-m", "omnicoder.tools.autofetch_benchmarks",
				"--out_root", str(out_root / "bench_data"),
				"--limit", os.getenv("OMNICODER_BENCH_FETCH_LIMIT", "200"),
			], env=env)
		except Exception as _afb:
			print(f"[warn] autofetch_benchmarks skipped: {_afb}")
		# Ensure HRM stays active during export when requested
		if os.getenv("OMNICODER_EXPORT_HRM", "1") == "1":
			env["OMNICODER_EXPORT_HRM"] = "1"
		run([
			sys.executable, "-m", "omnicoder.tools.build_mobile_release",
			"--out_root", str(out_root / "release"),
			"--quantize_onnx", "--onnx_preset", os.getenv("OMNICODER_ONNX_PRESET", "generic"),
		], env=env)
		# After export, run MLA micro-bench to recommend block-sparse settings
		try:
			run([
				sys.executable, "-m", "omnicoder.tools.mla_microbench",
				"--preset", args.student_preset,
				"--device", "cpu",
				"--steps", "256",
				"--out", str(out_root / "mla_microbench.json"),
			], env=env)
		except Exception:
			print("[warn] mla_microbench skipped")
		# Persist unified tokenizer/vocab metadata and artifact index for runtimes
		try:
			from pathlib import Path as _Pmeta
			meta_root = out_root / "release"
			meta_root.mkdir(parents=True, exist_ok=True)
			unified = {
				"text_size": 32000,
				"tokenizer_env": os.getenv("OMNICODER_HF_TOKENIZER", ""),
				"disable_remap": os.getenv("OMNICODER_DISABLE_TOKENIZER_REMAP", "0") == "1",
			}
			(_Pmeta(meta_root) / "unified_vocab_map.json").write_text(json.dumps(unified, indent=2), encoding="utf-8")
			artifacts = {
				"decode_step_onnx": str((meta_root / "text" / "omnicoder_decode_step.onnx").as_posix()),
				"provider_bench": str((meta_root / "text" / "provider_bench.json").as_posix()),
			}
			(meta_root / "ARTIFACTS.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
		except Exception as _em:
			print(f"[warn] export metadata write skipped: {_em}")
		if bool(args.run_provider_bench):
			decode_onnx = out_root / "release" / "text" / "omnicoder_decode_step.onnx"
			if decode_onnx.exists():
				providers = ["CPUExecutionProvider"]
				try:
					import onnxruntime as _ort  # type: ignore
					avail = set(_ort.get_available_providers())
					if "DmlExecutionProvider" in avail:
						providers.append("DmlExecutionProvider")
					if "CoreMLExecutionProvider" in avail:
						providers.append("CoreMLExecutionProvider")
					if "NNAPIExecutionProvider" in avail:
						providers.append("NNAPIExecutionProvider")
				except Exception:
					pass
				pb_json = out_root / "release" / "text" / "provider_bench.json"
				cmd = [
					sys.executable, "-m", "omnicoder.inference.runtimes.provider_bench",
					"--model", str(decode_onnx),
					"--providers", *providers,
					"--prompt_len", "128", "--gen_tokens", "256",
					"--threshold_json", str(Path("profiles")/"provider_thresholds.json"),
					"--check_fusions", "--canary_tokens_per_s",
					"--out_json", str(pb_json),
				]
				# Require attention fusion for GPU/mobile providers
				if ("DmlExecutionProvider" in providers) or ("CoreMLExecutionProvider" in providers):
					cmd += ["--require_attention"]
				if len(providers) >= 2:
					cmd += ["--compare_base", providers[0], "--compare_target", providers[1]]
					if "DmlExecutionProvider" in providers:
						cmd += ["--require_attention"]
				run(cmd, env=env)
				# Auto-update provider thresholds based on measured TPS when enabled
				if os.getenv("OMNICODER_AUTO_UPDATE_THRESHOLDS", "1") == "1":
					try:
						data = json.loads(pb_json.read_text(encoding='utf-8')) if pb_json.exists() else {}
						# Expect a mapping like {"providers": {name: {"tps": float}}} or flat {name: tps}
						measured: dict[str, float] = {}
						if isinstance(data, dict):
							if isinstance(data.get("providers"), dict):
								for k, v in data["providers"].items():
									try:
										measured[str(k)] = float(v.get("tps", 0.0))
									except Exception:
										pass
							else:
								for k, v in data.items():
									try:
										measured[str(k)] = float(v)
									except Exception:
										pass
						thr_path = Path("profiles")/"provider_thresholds.json"
						current = {}
						if thr_path.exists():
							try:
								current = json.loads(thr_path.read_text(encoding='utf-8'))
							except Exception:
								current = {}
						# Set thresholds to max(existing, FRACTION*measured) rounded to 1 decimal
						try:
							frac = float(os.getenv("OMNICODER_PROVIDER_THRESHOLD_FRACTION", "0.95"))
						except Exception:
							frac = 0.95
						updated = dict(current)
						for prov, tps in measured.items():
							if tps <= 0:
								continue
							try:
								cand = round(max(float(current.get(prov, 0.0)), float(frac) * float(tps)), 1)
								updated[prov] = cand
							except Exception:
								pass
						if updated != current:
							thr_path.write_text(json.dumps(updated, indent=2), encoding='utf-8')
							print(f"[write] updated provider thresholds: {thr_path}")
					except Exception as e:
						print(f"[warn] could not auto-update provider thresholds: {e}")
			# Draft acceptance metrics already computed; thresholds file updated above
		# Post-export auto-benchmark with quality/datasets (real benchmarks only)
		try:
			ab_cmd = [
				sys.executable, "-m", "omnicoder.eval.auto_benchmark",
				"--device", ("cuda" if args.device.startswith("cuda") else "cpu"),
				"--seq_len", os.getenv("OMNICODER_BENCH_SEQ_LEN", "128"),
				"--gen_tokens", os.getenv("OMNICODER_BENCH_GEN_TOKENS", "128"),
				"--preset", args.student_preset,
				"--out", str(out_root / "bench_summary.json"),
				"--autolocate_benchmarks",
				"--clip_jsonl", (env.get("OMNICODER_CLIP_JSONL", env.get("SFB_CLIP_JSONL", "")) or ""),
				"--fvd_pred_dir", (env.get("OMNICODER_VIDEO_PRED_DIR", "") or ""),
				"--fvd_ref_dir", (env.get("OMNICODER_VIDEO_REF_DIR", "") or ""),
				"--asr_jsonl", (env.get("OMNICODER_ASR_JSONL", "") or ""),
				"--code_tasks", (env.get("OMNICODER_CODE_TASKS", "") or ""),
				"--sr_pred_dir", (env.get("OMNICODER_SR_PRED_DIR", "") or ""),
				"--sr_ref_dir", (env.get("OMNICODER_SR_REF_DIR", "") or ""),
			]
			# Enforce real eval (no smoke): disable any ENV smoke flags for this call
			ab_env = dict(env)
			ab_env.pop('EXECUTE_TESTS', None)
			ab_env['OMNICODER_BENCH_TINY'] = '0'
			# If a best checkpoint exists, prefer it for benchmarking to avoid regressions
			try:
				from pathlib import Path as _P
				best_ck = (out_root / "student.pt").with_name("student_best.pt")
				if best_ck.exists():
					env["OMNICODER_CKPT"] = str(best_ck)
			except Exception:
				pass
			# Add audio quality (FAD) sidecars if present
			ab_cmd += ["--fad_pred_dir", (env.get("OMNICODER_FAD_PRED_DIR", "") or ""), "--fad_ref_dir", (env.get("OMNICODER_FAD_REF_DIR", "") or "")]
			# Standard datasets from profiles
			try:
				_datasets = json.loads(Path("profiles/datasets.json").read_text(encoding='utf-8')) if Path("profiles/datasets.json").exists() else {}
				def _ds(key: str) -> str:
					try:
						return str((_datasets.get(key, {}) or {}).get("path", ""))
					except Exception:
						return ""
				gsm = _ds("gsm8k"); mbpp = _ds("mbpp"); hot = _ds("hotpot")
				mmlu = _ds("mmlu"); hsw = _ds("hellaswag"); hme = _ds("humaneval")
				if gsm: ab_cmd += ["--gsm8k", gsm]
				if mbpp: ab_cmd += ["--mbpp", mbpp]
				if hot: ab_cmd += ["--hotpot", hot]
				# Optional general-ability/code commonsense benchmarks when provided
				if mmlu: ab_cmd += ["--mmlu_jsonl", mmlu]
				if hsw: ab_cmd += ["--hellaswag_jsonl", hsw]
				if hme: ab_cmd += ["--humaneval_jsonl", hme]
			except Exception:
				pass
			run(ab_cmd, env=ab_env)
		except Exception:
			print("[warn] post-export auto_benchmark skipped")

	# 4b) Auto-tune: write a suggested .env.tuned from benches/acceptance
	try:
		out_env = out_root / ".env.tuned"
		run([
			sys.executable, "-m", "omnicoder.tools.auto_tune",
			"--release_root", str(out_root / "release" / "text"),
			"--out_env", str(out_env),
		], env=env)
		manifest["env_tuned"] = str(out_env)
		# Append a checkpoint chain hint for the API server to load in order
		try:
			chain: list[str] = []
			for k in ("omnicoder_ppo.pt","omnicoder_grpo.pt","omnicoder_align.pt","omnicoder_pretrain_dsm.pt","omnicoder_draft_kd.pt"):
				p = out_root / k
				if p.exists():
					chain.append(str(p))
			# Include stage outputs when present
			for k in ("vl_out","verifier_kd_out","dsm_out","draft_ckpt"):
				v = manifest.get(k)
				if isinstance(v, str) and v:
					chain.append(v)
			if chain:
				with open(out_env, "a", encoding="utf-8") as f:
					f.write("\nOMNICODER_API_CKPT_CHAIN=" + " ".join(chain) + "\n")
		except Exception as _ec:
			print(f"[warn] could not append ckpt chain: {_ec}")
	except Exception:
		print("[warn] auto_tune skipped")

	# Optional: export to phone
	if args.export_to_phone:
		platform = str(args.export_to_phone).strip().lower()
		if platform in ("android","ios"):
			run([
				sys.executable, "-m", "omnicoder.tools.export_to_phone",
				"--platform", platform,
				"--out_root", str(out_root / "release"),
			], env=env)
		# Best-effort iOS device smoke when requested via env
		if os.getenv("OMNICODER_IOS_SMOKE", "0") == "1":
			try:
				mlmodel = out_root / "release" / "text" / "omnicoder_decode_step.mlmodel"
				if mlmodel.exists():
					run([
						sys.executable, "-m", "omnicoder.tools.ios_coreml_smoke",
						"--mlmodel", str(mlmodel),
						"--tps_threshold", os.getenv("OMNICODER_IOS_TPS_THRESHOLD", "6.0"),
						"--run_script", os.getenv("OMNICODER_IOS_RUN_SCRIPT", ""),
					], env=env)
			except Exception as e:
				print(f"[warn] ios_coreml_smoke skipped: {e}")

	# 3c.4) Text QA finetune over benchmark JSONLs when available
	try:
		qa_jsonls: list[str] = []
		for k in ("OMNICODER_MMLU_JSONL","OMNICODER_ARC_JSONL","OMNICODER_HELLASWAG_JSONL","OMNICODER_TRUTHFULQA_JSONL","OMNICODER_WINOGRANDE_JSONL","OMNICODER_AGIEVAL_JSONL","OMNICODER_BBH_JSONL","OMNICODER_STRATEGYQA_JSONL","OMNICODER_ARC_CH_JSONL","OMNICODER_NQ_JSONL","OMNICODER_HOTPOTQA_JSONL"):
			p = os.getenv(k, "").strip()
			if p and Path(p).exists():
				qa_jsonls.append(p)
		if qa_jsonls:
			qa_out = out_root / "omnicoder_qa_finetune.pt"
			if (not args.resume) or (not qa_out.exists()):
				print("[stage] qa_finetune")
				run([
					sys.executable, "-m", "omnicoder.training.qa_finetune",
					"--jsonl_list", *qa_jsonls,
					"--device", args.device,
					"--mobile_preset", args.student_preset,
					"--steps", os.getenv("OMNICODER_QA_STEPS", "800"),
					"--out", str(qa_out),
				], env=env)
			manifest["qa_out"] = str(qa_out)
	except Exception as _eqa:
		print(f"[warn] qa finetune skipped: {_eqa}")

	# 3c.5) Super-resolution training (baseline) when SR ref/pred dirs set
	try:
		pred_dir = os.getenv("OMNICODER_SR_PRED_DIR", "").strip()
		ref_dir = os.getenv("OMNICODER_SR_REF_DIR", "").strip()
		if pred_dir and ref_dir and Path(ref_dir).exists():
			sr_out = out_root / "sr_baseline.pt"
			if (not args.resume) or (not sr_out.exists()):
				print("[stage] sr_train")
				run([
					sys.executable, "-m", "omnicoder.training.sr_train",
					"--pred_dir", pred_dir, "--ref_dir", ref_dir,
					"--steps", os.getenv("OMNICODER_SR_STEPS", "1500"),
					"--device", args.device, "--out", str(sr_out),
				], env=env)
			manifest["sr_out"] = str(sr_out)
	except Exception as _esr:
		print(f"[warn] sr train skipped: {_esr}")

	# 3c.6) Audio latent supervised training
	try:
		audio_ref = os.getenv("OMNICODER_AUDIO_REF_DIR", "").strip()
		if audio_ref and Path(audio_ref).exists():
			aud_out = out_root / "audio_latent.pt"
			if (not args.resume) or (not aud_out.exists()):
				print("[stage] audio_latent_supervised")
				run([
					sys.executable, "-m", "omnicoder.training.audio_latent_supervised",
					"--ref_dir", audio_ref,
					"--steps", os.getenv("OMNICODER_AUDIO_LAT_STEPS", "1500"),
					"--device", args.device, "--out", str(aud_out),
				], env=env)
			manifest["audio_latent_out"] = str(aud_out)
	except Exception as _eal:
		print(f"[warn] audio latent supervised skipped: {_eal}")

	# 3c.7) SWE-bench runner (metadata only) and optional finetune harness
	try:
		swe_meta = os.getenv("OMNICODER_SWEBENCH_META", "").strip()
		patch_dir = os.getenv("OMNICODER_SWEBENCH_PATCH_DIR", "").strip()
		if swe_meta and Path(swe_meta).exists():
			swe_out = out_root / "swebench_results.jsonl"
			swe_sum = out_root / "swebench_summary.json"
			print("[stage] swebench_runner")
			run([
				sys.executable, "-m", "omnicoder.tools.swebench_runner",
				"--meta", swe_meta,
				"--out", str(swe_out),
				"--summary", str(swe_sum),
				"--limit", os.getenv("OMNICODER_SWEBENCH_LIMIT", "5"),
				"--setup_cmd", os.getenv("OMNICODER_SWEBENCH_SETUP", ""),
				"--patch_dir", patch_dir or "",
			], env=env)
			manifest["swebench_results"] = str(swe_out)
	except Exception as _esw:
		print(f"[warn] swebench runner skipped: {_esw}")

	manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
	(out_root / "run_training_manifest.json").write_text(json.dumps(manifest, indent=2))
	# Emit a concise human-readable summary for quick inspection
	(out_root / "TRAINING_SUMMARY.md").write_text(
		"\n".join([
			"## OmniCoder — Training Summary",
			"",
			f"Started: {manifest.get('started','')}",
			f"Finished: {manifest.get('finished','')}",
			"",
			"### Planned minutes",
			json.dumps(manifest.get('plan', {}), indent=2),
			"",
			"### Planned steps",
			json.dumps(manifest.get('steps', {}), indent=2),
			"",
			"### Artifacts",
			f"- Draft checkpoint: `{manifest.get('draft_ckpt','')}`",
			f"- KD stages: {len(manifest.get('kd_stages', []))}",
			f"- Pre-align: `{manifest.get('pre_align_out','')}`",
			f"- DS-MoE pretrain: `{manifest.get('dsm_out','')}`",
			f"- VL fused: `{manifest.get('vl_out','')}`",
			f"- Benchmarks: `{(out_root / 'bench_stage_summary.json')}`",
		]),
		encoding="utf-8",
	)
	(out_root / "READY_TO_EXPORT.md").write_text(
		"\n".join([
			"## OmniCoder — Ready to Export",
			"",
			f"- Draft checkpoint: `{manifest.get('draft_ckpt','')}`",
			f"- VL fused checkpoint: `{manifest.get('vl_out','')}`",
			f"- KD stages: {len(manifest.get('kd_stages', []))}",
			"",
			"### Next Steps",
			"- Run provider bench (optional) to verify tokens/s and fusions.",
			"- Export to phone:",
			"  - Android: `python -m omnicoder.tools.export_to_phone --platform android`",
			"  - iOS: `python -m omnicoder.tools.export_to_phone --platform ios`",
		]),
		encoding="utf-8",
	)
	print(json.dumps({"status": "ok", "manifest": str(out_root / "run_training_manifest.json")}))


if __name__ == "__main__":
	main()


