from __future__ import annotations

from typing import Optional


def text_tps(model, device: str = 'cuda', seq_len: int = 128, gen_tokens: int = 64) -> Optional[float]:
	"""Run a lightweight text throughput benchmark (tokens/sec) using existing bench util.

	Returns TPS (higher is better) or None on failure. Keeps model unmodified.
	"""
	try:
		from omnicoder.inference.benchmark import bench_tokens_per_second  # type: ignore
		return float(bench_tokens_per_second(model, seq_len=seq_len, gen_tokens=gen_tokens, device=device))
	except Exception:
		return None


def clipscore_mean(images_bchw, texts: list[str]) -> Optional[float]:
	"""Compute CLIPScore mean using open_clip when available; returns None if unavailable."""
	try:
		import open_clip  # type: ignore
		import torch
		from torchvision import transforms as _T  # type: ignore
		model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai')
		tok = open_clip.get_tokenizer('ViT-B-32')
		model = model.to('cpu').eval()
		pil_tf = _T.ToPILImage()
		img_list = [preprocess(pil_tf(images_bchw[i].cpu())).unsqueeze(0) for i in range(min(len(images_bchw), len(texts)))]
		if not img_list:
			return None
		img_batch = torch.cat(img_list, dim=0)
		with torch.inference_mode():
			img_feat = model.encode_image(img_batch)
			txt = tok(texts[:img_batch.size(0)])
			txt_feat = model.encode_text(txt)
			img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
			txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
			cs = (img_feat * txt_feat).sum(dim=-1).mean().item()
		return float(cs)
	except Exception:
		return None


def fid_score(ref_dir: str, gen_images_bchw) -> Optional[float]:
	"""Compute a quick FID proxy if torchmetrics is available; returns None on failure."""
	try:
		from torchmetrics.image.fid import FrechetInceptionDistance  # type: ignore
		import torch
		from pathlib import Path
		from torchvision import transforms as _T  # type: ignore
		fid = FrechetInceptionDistance(feature=2048).to('cpu').eval()
		imgs = (gen_images_bchw.clamp(0, 1) * 255).byte()
		for i in range(min(imgs.size(0), 32)):
			fid.update(imgs[i].unsqueeze(0), real=True)
		ref_paths = list(Path(ref_dir).glob('*.png'))[:32] + list(Path(ref_dir).glob('*.jpg'))[:32]
		to_u8 = _T.ToTensor()
		for p in ref_paths:
			from PIL import Image as _PIL
			im = _PIL.Image.open(p).convert('RGB')
			imt = to_u8(im)
			fid.update((imt * 255).byte().unsqueeze(0), real=False)
		return float(fid.compute().item())
	except Exception:
		return None


def fad_score(ref_dir: str, pred_dir: str, sr: int = 16000) -> Optional[float]:
	"""Compute FAD when torch-fad is available; returns None otherwise."""
	try:
		import torch_fad  # type: ignore
		from glob import glob
		ref = glob(f"{ref_dir}/*.wav")
		pred = glob(f"{pred_dir}/*.wav")
		if not ref or not pred:
			return None
		val = float(torch_fad.fad_from_paths(ref, pred, device='cuda' if __import__('torch').cuda.is_available() else 'cpu'))
		return val
	except Exception:
		return None

