from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


SCHEMA = "omnicoder.media_teacher_rollout_2026.v1"
MANIFEST_SCHEMA = "omnicoder.media_teacher_rollouts_manifest_2026.v1"
DEFAULT_COMFYUI_URL = "http://192.168.50.222:27188"
NEGATIVE_IMAGE = "low quality, blurry, distorted text, watermark, deformed"
NEGATIVE_VIDEO = "low quality, blurry, jitter, text, watermark"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = Path(path)
    if not source.exists():
        return rows
    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
                if limit and len(rows) >= limit:
                    break
    return rows


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]], append: bool = True) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with target.open(mode, encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, default=str, separators=(",", ":")) + "\n")


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def count_ok(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") in {"ok", "planned"}:
                total += 1
    return total


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [first_text(item) for item in value[:24]]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in (
            "prompt",
            "instruction",
            "question",
            "caption",
            "text",
            "content",
            "target_text",
            "response",
            "completion",
        ):
            text = first_text(value.get(key))
            if text:
                return text
        messages = value.get("messages")
        if isinstance(messages, list):
            text = first_text(messages)
            if text:
                return text
    return ""


def prompt_from_job(job: dict[str, Any]) -> str:
    payload = job.get("input_json") if isinstance(job.get("input_json"), dict) else {}
    text = first_text(payload) or first_text(job)
    return (text or "Create a high quality multimodal teacher artifact for this training job.")[:2400]


def classify_job(job: dict[str, Any]) -> dict[str, str]:
    teacher = str(job.get("teacher_name") or "").lower()
    job_type = str(job.get("job_type") or "").lower()
    text = f"{teacher} {job_type}"
    if "qwen_image_edit" in text or "image_edit" in text:
        return {"media_family": "qwen_image", "test": "qwen_edit", "workflow": "qwen_image_edit", "modality": "image"}
    if "qwen_image" in text or "image_reward" in text or "image_prompt" in text:
        return {"media_family": "qwen_image", "test": "qwen_t2i", "workflow": "qwen_image_generate", "modality": "image"}
    if "ltx" in text or "video" in text or "temporal" in text:
        return {"media_family": "ltx_video", "test": "ltx_t2v", "workflow": "ltx_video", "modality": "video"}
    if "ace" in text or "music" in text:
        return {"media_family": "ace_music", "test": "ace_music", "workflow": "ace_music", "modality": "music"}
    if "speech" in text or "audio" in text:
        return {"media_family": "ace_music", "test": "ace_music", "workflow": "ace_audio", "modality": "audio"}
    if "omni" in text or "multimodal" in text:
        return {"media_family": "omni_media", "test": "qwen_t2i", "workflow": "omni_media", "modality": "image"}
    return {"media_family": "unsupported", "test": "unsupported", "workflow": "unsupported", "modality": "unknown"}


def modality_list(media_family: str, modality: str) -> list[str]:
    if media_family == "qwen_image":
        return ["image", "text"]
    if media_family == "ltx_video":
        return ["video", "audio", "text"]
    if media_family == "ace_music":
        return ["music" if modality == "music" else "audio", "audio", "text"]
    if media_family == "omni_media":
        return ["image", "video", "audio", "text"]
    return ["text"]


def request_json(method: str, url: str, payload: Any | None = None, timeout: int = 60) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=int(timeout)) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8", errors="replace")) if raw else {}


def qwen_image_workflow(prompt: str, prefix: str, seed: int) -> dict[str, Any]:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_fp8_e4m3fn.safetensors", "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE_IMAGE, "clip": ["2", 0]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": seed, "steps": 8, "cfg": 4.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0, "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["8", 0]}},
    }


def ace_music_workflow(prompt: str, prefix: str, seed: int) -> dict[str, Any]:
    tags = prompt[:700] or "cinematic electronic, instrumental, polished, 96 bpm"
    return {
        "1": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "qwen_0.6b_ace15.safetensors", "clip_name2": "qwen_1.7b_ace15.safetensors", "type": "ace", "device": "default"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": "acestep_v1.5_turbo.safetensors", "weight_dtype": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ace_1.5_vae.safetensors"}},
        "4": {"class_type": "EmptyAceStep1.5LatentAudio", "inputs": {"seconds": 8.0, "batch_size": 1}},
        "5": {"class_type": "TextEncodeAceStepAudio1.5", "inputs": {"clip": ["1", 0], "tags": tags, "lyrics": "", "seed": seed, "bpm": 96, "duration": 8.0, "timesignature": "4", "language": "en", "keyscale": "A minor", "generate_audio_codes": True, "cfg_scale": 2.0, "temperature": 0.85, "top_p": 0.9, "top_k": 0, "min_p": 0.0}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["2", 0], "shift": 3.0}},
        "8": {"class_type": "KSampler", "inputs": {"model": ["7", 0], "seed": seed, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["4", 0]}},
        "9": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveAudioMP3", "inputs": {"audio": ["9", 0], "filename_prefix": prefix, "quality": "128k"}},
        "11": {"class_type": "SaveAudio", "inputs": {"audio": ["9", 0], "filename_prefix": prefix}},
    }


def ltx_encoder(comfyui_url: str) -> str:
    try:
        info = request_json("GET", comfyui_url.rstrip("/") + "/object_info", timeout=90)
        field = info.get("LTXAVTextEncoderLoader", {}).get("input", {}).get("required", {}).get("text_encoder")
        opts: list[str] = []
        if isinstance(field, list) and field:
            if isinstance(field[0], list):
                opts = [str(item) for item in field[0]]
            elif len(field) > 1 and isinstance(field[1], dict) and isinstance(field[1].get("options"), list):
                opts = [str(item) for item in field[1]["options"]]
        if "gemma_3_12B_it_fp4_mixed.safetensors" in opts:
            return "gemma_3_12B_it_fp4_mixed.safetensors"
        if opts:
            return opts[0]
    except Exception:
        pass
    return "gemma_3_12B_it_fp4_mixed.safetensors"


def ltx_video_workflow(comfyui_url: str, prompt: str, prefix: str, seed: int) -> dict[str, Any]:
    encoder = ltx_encoder(comfyui_url)
    return {
        "1": {"class_type": "LowVRAMCheckpointLoader", "inputs": {"ckpt_name": "ltx-2.3-22b-distilled.safetensors", "dependencies": ["7", 0]}},
        "2": {"class_type": "ModelSamplingLTXV", "inputs": {"model": ["1", 0], "max_shift": 2.05, "base_shift": 0.95}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0], "lora_name": "ltx-2.3-22b-distilled-lora-384.safetensors", "strength_model": 0.35}},
        "4": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": encoder, "ckpt_name": "ltx-2.3-22b-distilled.safetensors", "device": "default"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE_VIDEO, "clip": ["4", 0]}},
        "7": {"class_type": "LTXVConditioning", "inputs": {"positive": ["5", 0], "negative": ["6", 0], "frame_rate": 12.0}},
        "8": {"class_type": "CFGGuider", "inputs": {"model": ["3", 0], "positive": ["7", 0], "negative": ["7", 1], "cfg": 1.0}},
        "9": {"class_type": "LTXVScheduler", "inputs": {"steps": 6, "max_shift": 2.05, "base_shift": 0.95, "stretch": True, "terminal": 0.1}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral_cfg_pp"}},
        "12": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": 384, "height": 256, "length": 9, "batch_size": 1}},
        "17": {"class_type": "LTXVBaseSampler", "inputs": {"model": ["3", 0], "vae": ["1", 2], "width": 384, "height": 256, "num_frames": 9, "guider": ["8", 0], "sampler": ["11", 0], "sigmas": ["9", 0], "noise": ["10", 0], "strength": 1.0, "crop": "disabled", "crf": 35, "blur": 1}},
        "19": {"class_type": "LTXVTiledVAEDecode", "inputs": {"vae": ["1", 2], "latents": ["17", 0], "horizontal_tiles": 1, "vertical_tiles": 1, "overlap": 1, "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}},
        "20": {"class_type": "SaveAnimatedWEBP", "inputs": {"filename_prefix": prefix, "images": ["19", 0], "fps": 12.0, "lossless": False, "quality": 80, "method": "default"}},
    }


def workflow_for_job(job: dict[str, Any], family: dict[str, str], index: int, comfyui_url: str) -> dict[str, Any]:
    input_json = job.get("input_json") if isinstance(job.get("input_json"), dict) else {}
    embedded = input_json.get("workflow") if isinstance(input_json.get("workflow"), dict) else None
    if embedded:
        return embedded
    prompt = prompt_from_job(job)
    prefix = f"omnicoder_{family['workflow']}_{stable_hash({'job': job, 'index': index})[:12]}"
    seed = int(stable_hash({"job": job, "index": index})[:8], 16)
    if family["media_family"] == "qwen_image":
        return qwen_image_workflow(prompt, prefix, seed)
    if family["media_family"] == "ace_music":
        return ace_music_workflow(prompt, prefix, seed)
    if family["media_family"] == "ltx_video":
        return ltx_video_workflow(comfyui_url, prompt, prefix, seed)
    return qwen_image_workflow(prompt, prefix, seed)


def history_artifacts(history_item: dict[str, Any]) -> list[str]:
    outputs = history_item.get("outputs") if isinstance(history_item.get("outputs"), dict) else {}
    artifacts: list[str] = []
    for node in outputs.values():
        if not isinstance(node, dict):
            continue
        for values in node.values():
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, dict) and value.get("filename"):
                    artifacts.append(str(value["filename"]))
    return artifacts


def artifact_metadata(path: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    p = Path(path)
    if root and not p.is_absolute():
        p = Path(root) / p
    meta: dict[str, Any] = {"path": str(p), "exists": p.exists()}
    if p.exists() and p.is_file():
        meta.update({"byte_size": p.stat().st_size, "sha256": file_sha256(p), "suffix": p.suffix.lower()})
    return meta


def parse_runner_stdout(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for line in reversed(text.splitlines()):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
    return {"stdout": text}


def run_subprocess_job(
    job: dict[str, Any],
    family: dict[str, str],
    index: int,
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any], list[dict[str, Any]], str]:
    jobs_dir = out_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_json = jobs_dir / f"media_teacher_job_{index:08d}.json"
    write_json(job_json, job)
    if args.runner_command:
        command = str(args.runner_command).format(
            job_json=job_json,
            out_dir=out_dir,
            test=family["test"],
            media_family=family["media_family"],
        )
    else:
        script = Path(args.modal_script)
        if not script.exists():
            return "failed", {"error": f"missing_runner_script:{script}"}, [], f"missing_runner_script:{script}"
        tests = family["test"]
        command = f"{sys.executable} {script} --tests {tests}"
    env = dict(os.environ)
    env.update(
        {
            "OMNICODER_MEDIA_TEACHER_JOB": str(job_json),
            "OMNICODER_MEDIA_TEACHER_OUT_DIR": str(out_dir),
            "OMNICODER_MEDIA_TEACHER_TEST": family["test"],
            "OMNICODER_MEDIA_TEACHER_FAMILY": family["media_family"],
        }
    )
    proc = subprocess.run(
        command,
        shell=True,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=args.timeout,
        env=env,
    )
    payload = parse_runner_stdout(proc.stdout)
    files = payload.get("files") if isinstance(payload.get("files"), list) else []
    if not files and isinstance(payload.get("summary"), dict):
        files = payload["summary"].get("files") if isinstance(payload["summary"].get("files"), list) else []
    artifacts = [artifact_metadata(str(path), args.artifact_root) for path in files]
    ok = proc.returncode == 0 and bool(payload.get("ok", bool(artifacts)))
    result = {
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "payload": payload,
    }
    return ("ok" if ok else "failed"), result, artifacts, "" if ok else "runner_failed"


def run_comfyui_job(
    job: dict[str, Any],
    family: dict[str, str],
    index: int,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any], list[dict[str, Any]], str]:
    comfyui_url = args.comfyui_url.rstrip("/")
    workflow = workflow_for_job(job, family, index, comfyui_url)
    queued = request_json("POST", comfyui_url + "/prompt", {"prompt": workflow}, timeout=args.request_timeout)
    if queued.get("node_errors"):
        return "failed", {"queued": queued, "error": "node_errors"}, [], "node_errors"
    prompt_id = str(queued.get("prompt_id") or "")
    if not prompt_id:
        return "failed", {"queued": queued, "error": "missing_prompt_id"}, [], "missing_prompt_id"
    started = time.time()
    history_item: dict[str, Any] = {}
    while time.time() - started < args.timeout:
        history = request_json("GET", comfyui_url + f"/history/{prompt_id}", timeout=args.request_timeout)
        if prompt_id in history:
            history_item = history[prompt_id]
            status = history_item.get("status", {}) if isinstance(history_item.get("status"), dict) else {}
            if status.get("status_str") == "error" or status.get("completed") is True:
                break
        time.sleep(max(1.0, float(args.poll_seconds)))
    status_obj = history_item.get("status", {}) if isinstance(history_item.get("status"), dict) else {}
    artifacts = [artifact_metadata(name, args.artifact_root) for name in history_artifacts(history_item)]
    ok = status_obj.get("completed") is True and bool(artifacts or history_item.get("outputs"))
    result = {"prompt_id": prompt_id, "queued": queued, "history_status": status_obj}
    return ("ok" if ok else "failed"), result, artifacts, "" if ok else "comfyui_failed"


def row_for_result(
    job: dict[str, Any],
    family: dict[str, str],
    index: int,
    status: str,
    result: dict[str, Any],
    artifacts: list[dict[str, Any]],
    error: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "run_id": args.run_id,
        "index": index,
        "status": status,
        "error": error,
        "teacher": job.get("teacher_name"),
        "job_type": job.get("job_type"),
        "media_family": family["media_family"],
        "workflow": family["workflow"],
        "test": family["test"],
        "modality": family["modality"],
        "modalities": modality_list(family["media_family"], family["modality"]),
        "record_kind": "comfyui_modality_teacher_rollout",
        "input_json": {"source_job": job, "prompt": prompt_from_job(job)},
        "target_json": {
            "teacher_status": status,
            "rollout_result": result,
            "artifact_metadata": artifacts,
            "distillation_target": "Predict artifact ledger tokens, multimodal critique, reward labels, and repair actions.",
        },
        "artifact_metadata": artifacts,
        "artifact_count": len(artifacts),
        "split": "train",
        "quality_score": 0.85 if status == "ok" else 0.0,
    }


def output_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "all": out_dir / "media_teacher_rollouts.jsonl",
        "qwen_image": out_dir / "qwen_image_rollouts.jsonl",
        "ltx_video": out_dir / "ltx_video_rollouts.jsonl",
        "ace_music": out_dir / "ace_music_rollouts.jsonl",
        "omni_media": out_dir / "omni_media_rollouts.jsonl",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(out_dir)
    if not args.resume:
        for path in set(paths.values()):
            if path.exists():
                path.unlink()
    jobs = read_jsonl(args.input, args.limit)
    existing = count_jsonl(paths["all"]) if args.resume else 0
    rows_written = 0
    failures = 0
    for index, job in enumerate(jobs, 1):
        if index <= existing:
            continue
        family = classify_job(job)
        if family["media_family"] == "unsupported":
            status, result, artifacts, error = "failed" if args.strict_live else "skipped", {"error": "unsupported_media_teacher_job"}, [], "unsupported_media_teacher_job"
        elif args.mode == "dry-run":
            status, result, artifacts, error = "planned", {"planned": True}, [], ""
        elif args.mode == "report":
            status, result, artifacts, error = "planned", {"report_only": True}, [], ""
        elif args.runner_command or args.modal_script:
            status, result, artifacts, error = run_subprocess_job(job, family, index, out_dir, args)
        else:
            status, result, artifacts, error = run_comfyui_job(job, family, index, args)
        row = row_for_result(job, family, index, status, result, artifacts, error, args)
        write_jsonl(paths["all"], [row])
        if family["media_family"] in paths:
            write_jsonl(paths[family["media_family"]], [row])
        rows_written += 1
        if status == "failed":
            failures += 1
    counts = {path.name: count_ok(path) for path in sorted(set(paths.values())) if path.exists()}
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "ok" if failures == 0 else "failed" if args.strict_live else "partial",
        "run_id": args.run_id,
        "mode": args.mode,
        "jobs": len(jobs),
        "written": rows_written,
        "failures": failures,
        "counts": counts,
        "paths": {key: str(path) for key, path in paths.items() if path.exists()},
    }
    write_json(out_dir / "media_teacher_rollout_manifest.json", manifest)
    rollout_manifest = out_dir / "teacher_rollout_manifest.json"
    merged: dict[str, Any] = {}
    if rollout_manifest.exists():
        try:
            merged = json.loads(rollout_manifest.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            merged = {}
    merged.update({"status": manifest["status"], "run_id": args.run_id})
    existing_counts = merged.get("counts") if isinstance(merged.get("counts"), dict) else {}
    existing_counts.update(counts)
    merged["counts"] = existing_counts
    merged["media_teacher_rollouts"] = manifest
    write_json(rollout_manifest, merged)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen Image, LTX, and ACE media teacher rollouts from queued modality jobs")
    parser.add_argument("--input", "--jobs", dest="input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=["dry-run", "report", "live"], default="live")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-live", action="store_true")
    parser.add_argument("--runner-command", default=os.getenv("OMNICODER_MEDIA_TEACHER_RUNNER_COMMAND", ""))
    parser.add_argument("--modal-script", default=os.getenv("OMNICODER_MEDIA_TEACHER_MODAL_SCRIPT", ""))
    parser.add_argument("--comfyui-url", "--comfy-url", dest="comfyui_url", default=os.getenv("OMNICODER_COMFYUI_URL", DEFAULT_COMFYUI_URL))
    parser.add_argument("--artifact-root", default=os.getenv("OMNICODER_COMFYUI_OUTPUT_ROOT", "/opt/ComfyUI/output"))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("OMNICODER_MEDIA_TEACHER_TIMEOUT", "2400")))
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.run_id:
        args.run_id = Path(args.out_dir).name
    try:
        manifest = run(args)
        print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
        if args.mode == "live" and args.strict_live and manifest["status"] != "ok":
            return 6
        return 0
    except (OSError, subprocess.TimeoutExpired, TimeoutError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True, sort_keys=True), file=sys.stderr)
        return 6 if args.mode == "live" and args.strict_live else 2


if __name__ == "__main__":
    raise SystemExit(main())
