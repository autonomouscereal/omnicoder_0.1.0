from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn
from torch.utils.data import DataLoader
from omnicoder.utils.resources import recommend_num_workers

from omnicoder.modeling.multimodal.aligner import PreAligner, CrossModalVerifier
from omnicoder.modeling.multimodal.vision_encoder import VisionBackbone
from omnicoder.training.data.vl_image_pairs import ImagePairsDataset
from omnicoder.training.simple_tokenizer import get_text_tokenizer


class TextEmbedder(nn.Module):
	def __init__(self, vocab_size: int, embed_dim: int) -> None:
		super().__init__()
		self.emb = nn.Embedding(int(vocab_size), int(embed_dim))

	def forward(self, ids: torch.Tensor) -> torch.Tensor:
		e = self.emb(ids)
		return e.mean(dim=1)


def _images_loader(path: str, batch_size: int, image_size: Tuple[int, int]) -> DataLoader:
	def collate(batch: list[tuple[torch.Tensor, str]]) -> tuple[torch.Tensor, list[str]]:
		imgs, texts = zip(*batch)
		return torch.stack(imgs, dim=0), list(texts)

	ds = ImagePairsDataset(path, image_size=tuple(image_size))
	# On Windows, avoid multiprocessing with a local collate function (cannot be pickled)
	import sys as _sys
	nw = 0 if _sys.platform.startswith("win") else recommend_num_workers()
	return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=nw, collate_fn=collate)


def main() -> None:
	ap = argparse.ArgumentParser(description="Train mini-CLIP style CrossModalVerifier using PreAligner embeddings")
	ap.add_argument("--data", type=str, required=True, help="Image folder with filenames containing text captions or paired list")
	ap.add_argument("--prealign_ckpt", type=str, required=True)
	ap.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
	ap.add_argument("--embed_dim", type=int, default=256)
	ap.add_argument("--batch_size", type=int, default=16)
	ap.add_argument("--steps", type=int, default=1000)
	ap.add_argument("--lr", type=float, default=1e-4)
	ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	ap.add_argument("--out", type=str, default="weights/cross_modal_verifier.pt")
	ap.add_argument("--out_onnx", type=str, default="weights/cross_modal_verifier.onnx")
	ap.add_argument("--opset", type=int, default=17)
	args = ap.parse_args()

	device = torch.device(args.device)
	# Data
	dl = _images_loader(args.data, batch_size=int(args.batch_size), image_size=tuple(args.image_size))

	# Models: frozen PreAligner heads + vision and tiny text embedder → verifier
	ck = torch.load(args.prealign_ckpt, map_location=device)
	ed = int(ck.get("embed_dim", int(args.embed_dim)))
	aligner = PreAligner(embed_dim=ed, text_dim=ed, image_dim=768).to(device).eval()
	aligner.load_state_dict(ck["aligner"])  # type: ignore[index]
	for p in aligner.parameters():
		p.requires_grad_(False)

	import os as _os
	vision = VisionBackbone(backend=_os.getenv("OMNICODER_VISION_BACKEND", "dinov3"), d_model=768, return_pooled=True).to(device).eval()
	tok = get_text_tokenizer(prefer_hf=True)
	try:
		vocab_size = int(getattr(tok, "vocab_size", 32000))
	except Exception:
		vocab_size = 32000
	text_emb = TextEmbedder(vocab_size=vocab_size, embed_dim=ed).to(device)

	verifier = CrossModalVerifier().to(device)
	opt = torch.optim.AdamW(list(text_emb.parameters()) + list(verifier.parameters()), lr=float(args.lr))

	step = 0
	bce = nn.BCELoss()
	for images, texts in dl:
		step += 1
		images = images.to(device)
		with torch.no_grad():
			_, pooled = vision(images)
			if pooled is None:
				pooled = vision(images)[0].mean(dim=1)
			pooled = pooled.clone()
		# Tokenize texts
		ids = [torch.tensor(tok.encode(str(t) or ""), dtype=torch.long) for t in texts]
		max_t = max(1, max(int(x.numel()) for x in ids))
		ids_pad = torch.zeros((len(ids), max_t), dtype=torch.long)
		for i, row in enumerate(ids):
			n = min(max_t, int(row.numel()))
			if n > 0:
				ids_pad[i, :n] = row[:n]
		ids_pad = ids_pad.to(device)
		txt_vec = text_emb(ids_pad)

		# Important: allow gradient flow into the text path so `text_emb` trains,
		# while `aligner` remains frozen (its params have requires_grad=False).
		em = aligner(text=txt_vec, image=pooled)
		t_emb = em["text"]
		i_emb = em["image"]

		# Positive labels for matching pairs; negatives via batch roll
		pos = verifier(t_emb, i_emb)
		i_emb_neg = torch.roll(i_emb, shifts=1, dims=0)
		neg = verifier(t_emb, i_emb_neg)
		# BCE with targets 1 for pos, 0 for neg
		loss = bce(pos, torch.ones_like(pos)) + bce(neg, torch.zeros_like(neg))
		loss.backward()
		torch.nn.utils.clip_grad_norm_(list(text_emb.parameters()) + list(verifier.parameters()), 1.0)
		opt.step(); opt.zero_grad(set_to_none=True)
		if step % 50 == 0:
			print(f"step {step}/{args.steps} loss={loss.item():.4f}")
		if step >= int(args.steps):
			break

	Path(args.out).parent.mkdir(parents=True, exist_ok=True)
	try:
		from omnicoder.utils.checkpoint import save_with_sidecar, maybe_save_best  # type: ignore
	except Exception:
		save_with_sidecar = None  # type: ignore
		maybe_save_best = None  # type: ignore
	payload = {"embed_dim": ed, "text_emb": text_emb.state_dict(), "verifier": verifier.state_dict()}
	if callable(save_with_sidecar):
		final = save_with_sidecar(args.out, payload, meta={'train_args': {'steps': int(args.steps)}})
	else:
	_safe_save(payload, args.out)
		final = args.out
	try:
		if callable(maybe_save_best) and 'loss' in locals():
			maybe_save_best(args.out, verifier, 'verifier_bce', float(loss.item()), higher_is_better=False)
	except Exception:
		pass
	print(f"Saved verifier to {final}")

	# ONNX export wrapper: takes two normalized embeddings and outputs score
	class VerifierONNX(nn.Module):
		def __init__(self, v: CrossModalVerifier):
			super().__init__(); self.v = v
		def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
			return self.v(a, b)

	onnx_path = Path(args.out_onnx)
	onnx_path.parent.mkdir(parents=True, exist_ok=True)
	dummy_a = torch.randn(1, ed, device=device)
	dummy_b = torch.randn(1, ed, device=device)
	torch.onnx.export(VerifierONNX(verifier).eval().to(device), (dummy_a, dummy_b), onnx_path.as_posix(),
					  input_names=["a", "b"], output_names=["score"], opset_version=int(args.opset),
					  dynamic_axes={"a": {0: "batch"}, "b": {0: "batch"}, "score": {0: "batch"}})
	print(f"Exported verifier ONNX to {onnx_path}")


if __name__ == "__main__":
	main()


