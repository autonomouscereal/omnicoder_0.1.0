from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from omnicoder.modeling.omnicoder2026 import build_omnicoder2026
from omnicoder.training.simple_tokenizer import get_text_tokenizer
from omnicoder.tokenization.text_range_2026 import effective_text_token_range


@torch.inference_mode()
def generate_text(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    text_range: tuple[int, int] | None = None,
) -> torch.Tensor:
    ids = input_ids
    for _ in range(int(max_new_tokens)):
        out = model(ids)
        logits = out["logits"][:, -1, :]
        if text_range is not None:
            lo, hi = text_range
            masked = logits.float().clone()
            masked[..., : int(lo)] = float("-inf")
            masked[..., int(hi) :] = float("-inf")
            logits = masked
        if temperature <= 0:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / float(temperature), dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
        ids = torch.cat((ids, nxt), dim=1)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Omnicoder 2026 native trunk text generation")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--profile", default="ledger_probe")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = get_text_tokenizer(prefer_hf=True)
    model = build_omnicoder2026(args.profile)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
    model.to(args.device).eval()
    model_vocab_size = int(getattr(model, "vocab_size", 0) or getattr(getattr(model, "cfg", None), "vocab_size", 0) or 0)
    text_range = effective_text_token_range(tokenizer=tok, model_vocab_size=model_vocab_size)
    prompt_ids = [int(item) for item in tok.encode(args.prompt)]
    bad_ids = [item for item in prompt_ids if item < 0 or item >= model_vocab_size]
    if bad_ids:
        raise ValueError(f"tokenizer produced ids outside model vocab: examples={bad_ids[:8]} vocab_size={model_vocab_size}")
    x = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
    y = generate_text(model, x, max_new_tokens=args.max_new_tokens, text_range=text_range)
    print(json.dumps({"text": tok.decode(y[0].detach().cpu().tolist()), "tokens": int(y.shape[1])}))


if __name__ == "__main__":
    main()
