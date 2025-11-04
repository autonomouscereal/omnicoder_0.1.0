from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import torch

from omnicoder.inference.generate import (
    generate,
    build_mobile_model_by_name,
    maybe_load_checkpoint,
)
from omnicoder.training.simple_tokenizer import get_text_tokenizer
from omnicoder.modeling.multimodal.fusion import MultimodalComposer
from omnicoder.inference.tool_use import build_default_registry


def _plan_target_via_model(model: Any, tok: Any, prompt: str, device: str, default: str = "text") -> str:
	try:
		plan_q = (
			(prompt or "").rstrip()
			+ "\n\nYou are a modality planner. Choose exactly one modality for the best answer to the user.\n"
			+ "Options: text, image, video, audio.\n"
			+ "Answer with just one word from the options.\n"
		)
		inp = torch.tensor([tok.encode(plan_q)], dtype=torch.long, device=device)
		out_ids = generate(model, inp, max_new_tokens=4, temperature=0.0)
		out = tok.decode(out_ids[0].tolist())
		ans = out[len(plan_q):].strip().lower()
		if "video" in ans:
			return "video"
		if "image" in ans or "picture" in ans or "photo" in ans:
			return "image"
		if "audio" in ans or "speech" in ans or "sound" in ans or "voice" in ans:
			return "audio"
		if "text" in ans:
			return "text"
	except Exception:
		pass
	return default


@torch.inference_mode()
def run_unified(inputs: Dict[str, Any], preset: str = "mobile_4gb", device: str = "cuda", ckpt: str = "") -> Dict[str, Any]:
	"""Single entrypoint for unified multimodal inference.

	inputs may contain: prompt/text, image (path), video (path), audio (path), target (text|image|video|audio) or auto.
	Returns a dict with keys depending on the target: {text} or {image_path} or {video_path} or {audio_path}.
	"""
	tok = get_text_tokenizer(prefer_hf=True)
	model = build_mobile_model_by_name(preset)
	if ckpt:
		maybe_load_checkpoint(model, ckpt)
	model.eval().to(device)

	composer = MultimodalComposer(d_model=model.embed.embedding_dim, vision_dim=384)

	# Prepare optional fused features
	prompt = str(inputs.get("prompt") or inputs.get("text") or "").strip()
	input_ids = torch.tensor([tok.encode(prompt or "")], dtype=torch.long, device=device)
	image_path = str(inputs.get("image") or "").strip()
	video_path = str(inputs.get("video") or "").strip()

	fused = input_ids
	# NEW: pre-generation tool pass (outside hot decode loop). If prompt contains
	# inline tool tags, resolve them once and append a small JSON blob to context.
	try:
		if '<tool:' in prompt:
			reg = build_default_registry()
			repl = reg.parse_and_invoke_all(prompt)
			if repl:
				ctx = "\n" + str(repl)
				ctx_ids = torch.tensor([tok.encode(ctx)], dtype=torch.long, device=device)
				fused = torch.cat([fused, ctx_ids], dim=1)
	except Exception:
		pass
	if video_path:
		try:
			import numpy as np
			import cv2  # type: ignore
			frames = []
			cap = cv2.VideoCapture(video_path)
			ok = True
			while ok and len(frames) < 64:
				ok, frame = cap.read()
				if not ok:
					break
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
				frames.append(frame)
			cap.release()
			if frames:
				arr = torch.from_numpy((np.stack(frames, axis=0).astype("float32") / 255.0)).permute(0,3,1,2).unsqueeze(0).to(device)
				fused = composer.fuse_text_video(model_with_embed=model, input_ids=input_ids, video_btchw=arr, max_frames=min(arr.size(1), 16))
		except Exception:
			pass
	elif image_path:
		try:
			from PIL import Image  # type: ignore
			import numpy as np
			img = Image.open(image_path).convert("RGB").resize((224, 224))
			arr = torch.from_numpy(np.array(img).astype("float32") / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
			fused = composer.fuse_text_image(model_with_embed=model, input_ids=input_ids, image_bchw=arr)
		except Exception:
			pass

	need_hidden = True
	outputs = model(fused, return_hidden=need_hidden)
	hidden = None
	if isinstance(outputs, tuple):
		if len(outputs) == 2:
			_, hidden = outputs
		elif len(outputs) == 3:
			hidden = outputs[2]

	# Model-based modality planner (no keyword heuristics)
	explicit = str(inputs.get("target") or inputs.get("modality") or "").strip().lower()
	target = explicit if explicit in ("text","image","video","audio") else _plan_target_via_model(model, tok, prompt, device, default="text")
	out: Dict[str, Any] = {"target": target}

	if target == "image":
		# Use image generator with hidden/text conditioning
		from omnicoder.modeling.multimodal.image_pipeline import ImageGenPipeline
		dtype = torch.float16 if device.startswith("cuda") else torch.float32
		pipe = ImageGenPipeline(backend=os.getenv("OMNICODER_IMAGE_BACKEND", "diffusers"), device=device, dtype=dtype, hf_id=os.getenv("OMNICODER_SD_MODEL", ""), local_path=os.getenv("OMNICODER_SD_LOCAL_PATH", "") or None)
		if not pipe.ensure_loaded():
			# Fallback to text generation
			out_ids = generate(model, input_ids, max_new_tokens=int(inputs.get("max_new_tokens", 64)))
			out["text"] = tok.decode(out_ids[0].tolist())
			return out
		img_out = Path(inputs.get("image_out", "weights/image_unified.png"))
		img_out.parent.mkdir(parents=True, exist_ok=True)
		pipe.generate(prompt or "A detailed scene", conditioning=hidden, steps=int(inputs.get("image_steps", 20)), size=(int(inputs.get("image_width", 512)), int(inputs.get("image_height", 512))), out_path=str(img_out))
		out["image_path"] = str(img_out)
		return out

	if target == "video":
		from omnicoder.modeling.multimodal.video_pipeline import VideoGenPipeline
		dtype = torch.float16 if device.startswith("cuda") else torch.float32
		vpipe = VideoGenPipeline(backend=os.getenv("OMNICODER_VIDEO_BACKEND", "diffusers"), device=device, dtype=dtype, hf_id=os.getenv("OMNICODER_T2V_MODEL", ""), local_path=os.getenv("OMNICODER_T2V_LOCAL_PATH", "") or None)
		if not vpipe.ensure_loaded():
			out_ids = generate(model, input_ids, max_new_tokens=int(inputs.get("max_new_tokens", 64)))
			out["text"] = tok.decode(out_ids[0].tolist())
			return out
		vid_out = Path(inputs.get("video_out", "weights/video_unified.mp4"))
		vid_out.parent.mkdir(parents=True, exist_ok=True)
		vpipe.generate(prompt or "A dynamic scene", steps=int(inputs.get("video_steps", 25)), size=(int(inputs.get("video_width", 512)), int(inputs.get("video_height", 320))), num_frames=int(inputs.get("video_frames", 24)), out_path=str(vid_out))
		out["video_path"] = str(vid_out)
		return out

	if target == "audio":
		# Simple TTS path
		try:
			from omnicoder.modeling.multimodal.tts import TTSAdapter
			tts = TTSAdapter(model_name=os.getenv("OMNICODER_TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC"))
			wav_out = Path(inputs.get("audio_out", "weights/tts_unified.wav"))
			wav_out.parent.mkdir(parents=True, exist_ok=True)
			text = prompt or inputs.get("tts_text", "Hello from OmniCoder unified brain.")
			tts.synthesize(text, str(wav_out))
			out["audio_path"] = str(wav_out)
			return out
		except Exception:
			pass

	# Default: text generation
	out_ids = generate(model, input_ids, max_new_tokens=int(inputs.get("max_new_tokens", 64)))
	out["text"] = tok.decode(out_ids[0].tolist())
	return out


