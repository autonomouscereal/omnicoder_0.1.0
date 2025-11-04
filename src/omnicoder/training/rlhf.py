from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.config import get_mobile_preset
from omnicoder.training.simple_tokenizer import get_text_tokenizer


def _load_prefs(jsonl_path: str) -> List[dict]:
    p = Path(jsonl_path)
    lines = p.read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Simple RLHF (PPO-like skeleton) with offline preferences")
    ap.add_argument("--prefs", required=True, help="Path to JSONL with fields: prompt, chosen, rejected")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mobile_preset", default="mobile_4gb", choices=["mobile_4gb", "mobile_2gb"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl_coef", type=float, default=0.05, help="KL penalty vs. reference model")
    ap.add_argument("--out", default="weights/omnicoder_rlhf.pt")
    args = ap.parse_args()

    # Reference and trainable policy share init
    preset = get_mobile_preset(args.mobile_preset)
    def build_model():
        return OmniTransformer(
            vocab_size=preset.vocab_size,
            n_layers=preset.n_layers,
            d_model=preset.d_model,
            n_heads=preset.n_heads,
            mlp_dim=preset.mlp_dim,
            n_experts=preset.moe_experts,
            top_k=preset.moe_top_k,
            max_seq_len=1024,
            use_rope=True,
            kv_latent_dim=preset.kv_latent_dim,
            multi_query=preset.multi_query,
            multi_token=1,
        )

    policy = build_model().to(args.device)
    ref = build_model().to(args.device)
    ref.eval()
    tok = get_text_tokenizer(prefer_hf=True)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    prefs = _load_prefs(args.prefs)

    def _logprob_sum(model: OmniTransformer, text: str) -> torch.Tensor:
        ids = torch.tensor([tok.encode(text)], dtype=torch.long, device=args.device)
        logits = model(ids[:, :-1])
        if isinstance(logits, tuple):
            logits = logits[0]
        logprobs = torch.log_softmax(logits, dim=-1)
        lp = logprobs.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).sum(dim=-1)
        return lp.mean()

    for step in range(1, args.steps + 1):
        ex = prefs[(step - 1) % len(prefs)]
        prompt = str(ex.get("prompt", ""))
        chosen = str(ex.get("chosen", ""))
        rejected = str(ex.get("rejected", ""))
        # Compute pairwise preference loss (DPO-style surrogate): encourages policy to rank chosen over rejected
        with torch.no_grad():
            ref_lp_ch = _logprob_sum(ref, chosen)
            ref_lp_rj = _logprob_sum(ref, rejected)
        pol_lp_ch = _logprob_sum(policy, chosen)
        pol_lp_rj = _logprob_sum(policy, rejected)
        # Bradley-Terry loss with KL penalty
        beta = 1.0
        pref_loss = -torch.log(torch.sigmoid(beta * ((pol_lp_ch - ref_lp_ch) - (pol_lp_rj - ref_lp_rj))))
        # KL vs reference on chosen
        kl = torch.relu((pol_lp_ch - ref_lp_ch))
        loss = pref_loss + args.kl_coef * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if step % 10 == 0:
            print(f"step {step}/{args.steps} loss={float(loss.item()):.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    try:
        from omnicoder.utils.checkpoint import save_with_sidecar, maybe_save_best  # type: ignore
    except Exception:
        save_with_sidecar = None  # type: ignore
        maybe_save_best = None  # type: ignore
    if callable(save_with_sidecar):
        final = save_with_sidecar(args.out, policy.state_dict(), meta={'train_args': {'steps': int(args.steps)}})
    else:
        _safe_save(policy.state_dict(), args.out)
        final = args.out
    # Best based on preference loss (lower is better) over last step
    try:
        if callable(maybe_save_best) and 'loss' in locals():
            maybe_save_best(args.out, policy, 'rlhf_pref_loss', float(loss.item()), higher_is_better=False)
    except Exception:
        pass
    print(f"Saved RLHF policy checkpoint to {final}")


if __name__ == "__main__":
    main()

