from __future__ import annotations

"""
IDK/Guess Reward Shaping Trainer

Trains the policy to prefer honest uncertainty disclaimers with well-reasoned
speculation over confidently wrong statements. Uses a lightweight PPO-style
loop on open-ended QA JSONL where each record provides:
  {"prompt": str, "answer": str, "label": {"correct": bool, "has_idk": bool}}

Reward shaping:
 - Base reward: +1 for correct; -1 for incorrect
 - Uncertainty bonus: +alpha when (has_idk and correct); small +beta when (has_idk and not incorrect)
 - Penalty: -gamma when (confident and incorrect)

The script keeps everything torch-only (no numpy in hot ops) and avoids any
device moves in inner loops; it is a trainer, not an inference hot path.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn

from omnicoder.training.simple_tokenizer import get_text_tokenizer
from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.inference.generate import generate
from omnicoder.utils.logger import get_logger


def _load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                continue
    return rows


def _normalize_text(s: str) -> str:
    return " ".join(s.lower().strip().split())


def _extract_number(s: str) -> str | None:
    import re
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", s)
    return nums[-1] if nums else None


def _is_correct(pred_text: str, gold_answer: str) -> bool:
    # Numeric match if possible, else substring match (normalized)
    pnum = _extract_number(pred_text)
    gnum = _extract_number(gold_answer)
    if pnum is not None and gnum is not None:
        try:
            return float(pnum) == float(gnum)
        except Exception:
            pass
    p = _normalize_text(pred_text)
    g = _normalize_text(gold_answer)
    return (g in p) or (p.endswith(g))


def _wrote_idk(text: str) -> bool:
    t = text.lower()
    return ("i am not sure" in t) or ("i’m not sure" in t) or ("i am uncertain" in t) or ("i don't know" in t) or ("i do not know" in t)


def _compute_reward_v2(pred_text: str, gold_answer: str, alpha: float, beta: float, gamma: float) -> float:
    correct = _is_correct(pred_text, gold_answer)
    wrote_idk = _wrote_idk(pred_text)
    reward = 1.0 if correct else -1.0
    if wrote_idk and correct:
        reward += float(alpha)
    elif wrote_idk and (not correct):
        reward += float(beta)
    else:
        if not correct:
            reward -= float(gamma)
    return float(reward)


def main() -> None:
    ap = argparse.ArgumentParser(description="IDK/Guess reward shaping trainer (PPO-light)")
    ap.add_argument("--jsonl", type=str, required=True)
    ap.add_argument("--mobile_preset", type=str, default="mobile_4gb")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.2)
    ap.add_argument("--out", type=str, default="weights/idk_shaped.pt")
    args = ap.parse_args()

    log = get_logger("omnicoder.idk")
    tok = get_text_tokenizer(prefer_hf=True)
    from omnicoder.inference.generate import build_mobile_model_by_name
    model: OmniTransformer = build_mobile_model_by_name(args.mobile_preset)
    model.to(args.device).train()
    # Lightweight optimizer on all parameters; for LoRA-style, filter on heads
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    rows = _load_jsonl(args.jsonl)
    if not rows:
        raise SystemExit("idk_reward_train: dataset empty")

    for step in range(1, int(args.steps) + 1):
        r = rows[(step - 1) % len(rows)]
        prompt = str(r.get('prompt', ''))
        gold_ans = str(r.get('answer', ''))
        input_ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=args.device)
        # Policy rollout
        out_ids = generate(model, input_ids, max_new_tokens=int(args.max_new_tokens), temperature=0.7, top_k=40, top_p=0.9)
        text = tok.decode(out_ids[0].tolist())
        # Reward shaping against gold answer text
        rew = _compute_reward_v2(text, gold_ans, float(args.alpha), float(args.beta), float(args.gamma))
        # Self-supervised advantage via value head (if available)
        adv = torch.tensor([rew], dtype=torch.float32, device=args.device)
        # Compute policy loss as simple negative log-prob of sampled tokens (last segment)
        model.eval()  # get logits
        with torch.inference_mode():
            out = model(input_ids, use_cache=False)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
        model.train()
        # Align to last sequence (teacher forcing over full out_ids is left as future work)
        # Here we compute a surrogate: encourage next-token distribution towards sampled continuation via cross-entropy
        # Build target as last token of model output range (greedy proxy)
        tgt = torch.argmax(logits[:, -1, :], dim=-1)
        logp = torch.log_softmax(logits[:, -1, :], dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        loss = -(adv.detach() * logp).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step % 10) == 0 or step == 1:
            try:
                log.info("idk_step=%d loss=%.4f reward=%.3f", int(step), float(loss.item()), float(rew))
            except Exception:
                pass

    try:
        from omnicoder.utils.checkpoint import save_with_sidecar  # type: ignore
    except Exception:
        save_with_sidecar = None  # type: ignore
    if callable(save_with_sidecar):
        final = save_with_sidecar(args.out, model.state_dict(), meta={'train_args': {'steps': int(args.steps)}})
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        _safe_save({k: (v.detach().to('cpu') if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}, args.out)
        final = args.out
    print(f"[idk] saved to {final}")


if __name__ == "__main__":
    main()


