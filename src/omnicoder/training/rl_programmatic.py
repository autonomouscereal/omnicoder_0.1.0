from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.config import get_mobile_preset
from omnicoder.training.simple_tokenizer import get_text_tokenizer


def _load_tasks(jsonl_path: str) -> List[dict]:
    p = Path(jsonl_path)
    lines = p.read_text(encoding="utf-8").splitlines()
    data = [json.loads(l) for l in lines if l.strip()]
    return data


def _eval_code_reward(candidate_py: str, tests_code: str, timeout_s: int = 3) -> float:
    """
    Execute tests against candidate solution. Returns reward in [0,1] as
    fraction of tests passed if multiple asserts are present.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "sol.py").write_text(candidate_py, encoding="utf-8")
        # Wrap tests to count assertions
        test_wrapper = f"""
passed = 0
total = 0
def _assert(cond):
    global passed, total
    total += 1
    if cond:
        passed += 1

{tests_code}

print(passed, total)
"""
        (tmp / "tests_runner.py").write_text(test_wrapper, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(tmp / "tests_runner.py")],
                cwd=str(tmp),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            out = proc.stdout.strip().split()
            if len(out) >= 2:
                passed, total = int(out[0]), int(out[1])
                if total <= 0:
                    return 0.0
                return float(passed) / float(total)
            return 0.0
        except Exception:
            return 0.0


def _sum_logprob(model: OmniTransformer, input_ids: torch.Tensor) -> torch.Tensor:
    """Return scalar log-prob sum of tokens 1..T conditioned on prefix."""
    with torch.no_grad():
        logits = model(input_ids[:, :-1])  # type: ignore[arg-type]
        if isinstance(logits, tuple):
            logits = logits[0]
        # Gather log-probs of next tokens
        next_logits = logits[:, -1:, :] if logits.size(1) > 0 else logits
        # Compute autoregressive logprobs across sequence
        # For simplicity, use only the last-step logits during sampling; for loss
        # we recompute with full sequence below in the training loop.
    # This function not used in training; kept for potential extensions
    return torch.tensor(0.0, device=next(model.parameters()).device)


def main() -> None:
    ap = argparse.ArgumentParser(description="Programmatic RL (REINFORCE) for code tasks JSONL")
    ap.add_argument("--tasks", required=True, help="Path to JSONL with fields: prompt, tests")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mobile_preset", default="mobile_4gb", choices=["mobile_4gb", "mobile_2gb"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--out", default="weights/omnicoder_rl_programmatic.pt")
    args = ap.parse_args()

    # Build model
    preset = get_mobile_preset(args.mobile_preset)
    model = OmniTransformer(
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
    model.to(args.device)
    model.train()

    tok = get_text_tokenizer(prefer_hf=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    tasks = _load_tasks(args.tasks)
    random.seed(0)

    def _sample(prompt: str) -> Tuple[List[int], torch.Tensor]:
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=args.device)
        generated_ids: List[int] = []
        past_kv = None
        with torch.no_grad():
            for _ in range(args.max_new_tokens):
                step_inp = ids[:, -1:] if ids.size(1) > 1 else ids
                outputs = model(step_inp, past_kv=past_kv, use_cache=True)
                if isinstance(outputs, tuple) and len(outputs) == 3:
                    logits, past_kv, _ = outputs  # type: ignore
                else:
                    logits, past_kv = outputs  # type: ignore
                logits_last = logits[:, -1, :]
                # top-k sampling
                topk_vals, topk_idx = torch.topk(logits_last, k=min(args.top_k, logits_last.size(-1)))
                probs = torch.softmax(topk_vals / max(args.temperature, 1e-5), dim=-1)
                next_local = torch.multinomial(probs, num_samples=1)
                next_id = topk_idx.gather(-1, next_local)
                ids = torch.cat([ids, next_id], dim=1)
                generated_ids.append(int(next_id.item()))
        return generated_ids, ids

    for step in range(1, args.steps + 1):
        task = random.choice(tasks)
        prompt = str(task.get("prompt", ""))
        tests = str(task.get("tests", ""))
        # Generate candidate
        gen_ids, full_ids = _sample(prompt)
        # Convert to python (naive): assume model outputs a full solution module
        candidate_text = tok.decode(gen_ids)
        reward = _eval_code_reward(candidate_text, tests)

        # Compute REINFORCE loss: -R * sum log pi(a_t|s_t)
        # Recompute logits over the full sequence to compute log-probs
        input_ids = full_ids[:, :-1]
        target_ids = full_ids[:, 1:]
        logits = model(input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        logprobs = torch.log_softmax(logits, dim=-1)
        gathered = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        # Only apply reward to the generated segment (exclude the prompt portion)
        gen_len = len(gen_ids)
        seq_len = gathered.size(1)
        mask = torch.zeros_like(gathered)
        mask[:, seq_len - gen_len :] = 1.0
        logprob_sum = (gathered * mask).sum()
        loss = -(reward * logprob_sum)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        print(f"step {step}/{args.steps} reward={reward:.3f} loss={float(loss.item()):.4f}")
        if step % 50 == 0:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            try:
                from omnicoder.utils.checkpoint import save_with_sidecar, maybe_save_best  # type: ignore
            except Exception:
                save_with_sidecar = None  # type: ignore
                maybe_save_best = None  # type: ignore
            if callable(save_with_sidecar):
                save_with_sidecar(args.out, model.state_dict(), meta={'train_args': {'steps': int(args.steps)}})
            else:
                _safe_save(model.state_dict(), args.out)
            try:
                if callable(maybe_save_best):
                    maybe_save_best(args.out, model, 'rl_prog_reward', float(reward), higher_is_better=True)
            except Exception:
                pass

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    try:
        from omnicoder.utils.checkpoint import save_with_sidecar, maybe_save_best  # type: ignore
    except Exception:
        save_with_sidecar = None  # type: ignore
        maybe_save_best = None  # type: ignore
    if callable(save_with_sidecar):
        final = save_with_sidecar(args.out, model.state_dict(), meta={'train_args': {'steps': int(args.steps)}})
    else:
        _safe_save(model.state_dict(), args.out)
        final = args.out
    try:
        if callable(maybe_save_best):
            maybe_save_best(args.out, model, 'rl_prog_reward', float(reward), higher_is_better=True)
    except Exception:
        pass
    print(f"Saved RL-updated checkpoint to {final}")


if __name__ == "__main__":
    main()

