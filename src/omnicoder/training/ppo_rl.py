from __future__ import annotations

"""
Minimal PPO skeleton for text reasoning tasks with programmatic rewards.

This file outlines PPO loops over a prompt dataset, using reward functions plugged
in from eval harnesses (e.g., code pass@k, exact-match, or proxy metrics). It is a
starting point; production PPO requires GAE, clipping, entropy/reg terms, and batching.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Callable

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.config import get_mobile_preset
from omnicoder.training.simple_tokenizer import get_text_tokenizer


def _load_prompts(jsonl: str) -> List[dict]:
    """Load prompts from JSONL; fallback to treating plain lines as prompts."""
    recs: List[dict] = []
    for line in open(jsonl, 'r', encoding='utf-8', errors='ignore'):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                recs.append(obj)
                continue
        except Exception:
            pass
        recs.append({"prompt": line.strip()})
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description='PPO skeleton for reasoning with programmatic rewards')
    ap.add_argument('--prompts', type=str, required=True, help='JSONL with {prompt, target?, tests?}')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--ppo_epochs', type=int, default=3)
    ap.add_argument('--clip_ratio', type=float, default=0.2)
    ap.add_argument('--kl_coef', type=float, default=0.01)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--gae_lambda', type=float, default=0.95)
    ap.add_argument('--entropy_coef', type=float, default=0.01)
    ap.add_argument('--mini_batch', type=int, default=4)
    ap.add_argument('--max_new_tokens', type=int, default=64)
    ap.add_argument('--reward', type=str, default='text', choices=['text','code_exact','code_tests','clip','wer','mos_proxy','code_passk','rm'])
    ap.add_argument('--rm_path', type=str, default='')
    ap.add_argument('--pass_k', type=int, default=5)
    ap.add_argument('--out', type=str, default='weights/omnicoder_ppo.pt')
    args = ap.parse_args()

    tok = get_text_tokenizer(prefer_hf=True)
    # Build a small/mobile policy for PPO
    if args.device == 'cpu' and args.steps <= 4:
        policy = OmniTransformer(
            vocab_size=1024,
            n_layers=1,
            d_model=128,
            n_heads=4,
            mlp_dim=256,
            n_experts=2,
            top_k=1,
            max_seq_len=128,
            kv_latent_dim=64,
            multi_query=True,
            multi_token=1,
        ).to(args.device)
    else:
        p = get_mobile_preset('mobile_4gb')
        policy = OmniTransformer(
            vocab_size=p.vocab_size,
            n_layers=p.n_layers,
            d_model=p.d_model,
            n_heads=p.n_heads,
            mlp_dim=p.mlp_dim,
            n_experts=p.moe_experts,
            top_k=p.moe_top_k,
            max_seq_len=p.max_seq_len,
            kv_latent_dim=p.kv_latent_dim,
            multi_query=p.multi_query,
            multi_token=1,
        ).to(args.device)
    # Keep a frozen reference policy for KL penalty
    ref = None
    try:
        ref = type(policy)(
            vocab_size=getattr(policy, 'vocab_size', 32000),
            n_layers=getattr(policy, 'n_layers', 1),
            d_model=getattr(policy, 'd_model', 128),
            n_heads=getattr(policy, 'n_heads', 4),
            mlp_dim=getattr(policy, 'mlp_dim', 256),
            n_experts=getattr(policy, 'n_experts', 2),
            top_k=getattr(policy, 'top_k', 1),
            max_seq_len=getattr(policy, 'max_seq_len', 128),
            kv_latent_dim=getattr(policy, 'kv_latent_dim', 64),
            multi_query=getattr(policy, 'multi_query', True),
            multi_token=1,
        ).to(args.device)
        ref.load_state_dict(policy.state_dict(), strict=False)
        ref.eval()
    except Exception:
        ref = None
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    prompts = _load_prompts(args.prompts)

    def _sample(text: str) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        ids = torch.tensor([tok.encode(text)], dtype=torch.long, device=args.device)
        out_ids: List[int] = []
        logprobs = []
        past_kv = None
        with torch.no_grad():
            for _ in range(args.max_new_tokens):
                step_inp = ids[:, -1:] if ids.size(1) > 1 else ids
                out = policy(step_inp, past_kv=past_kv, use_cache=True)
                if isinstance(out, tuple):
                    logits, past_kv = out[0], out[1]
                else:
                    logits = out
                dist = torch.distributions.Categorical(logits=logits[:, -1, :])
                nxt = dist.sample().unsqueeze(-1)
                lp = dist.log_prob(nxt.squeeze(-1))
                ids = torch.cat([ids, nxt], dim=1)
                out_ids.append(int(nxt.item()))
                logprobs.append(lp)
        return out_ids, ids, torch.stack(logprobs, dim=1)

    # Reward factory per example
    # Optional reward model head
    rm_head = None
    if args.reward == 'rm' and args.rm_path:
        try:
            blob = torch.load(args.rm_path, map_location='cpu')
            from torch import nn as _nn
            d = int(blob.get('d_model', getattr(policy, 'd_model', 512)))
            rm_head = _nn.Linear(d, 1)
            if 'reward_head' in blob:
                rm_head.load_state_dict(blob['reward_head'])
            rm_head = rm_head.to(args.device).eval()
        except Exception:
            rm_head = None

    def _make_reward(ex: dict) -> Callable[[str], float]:
        targets = ex.get('targets', []) or ([ex.get('target')] if ex.get('target') else [])
        if args.reward == 'text':
            def _r(pred: str) -> float:
                pl = pred.strip().lower()
                return 1.0 if any((t or '').strip().lower() in pl for t in targets) else 0.0
            return _r
        if args.reward == 'code_exact':
            return lambda pred: (1.0 if any(pred.strip() == (t or '').strip() for t in targets) else 0.0)
        if args.reward == 'code_tests':
            tests_code = ex.get('tests', '') or ex.get('tests_code', '')
            if not tests_code:
                return lambda pred: 0.0
            import subprocess, sys, tempfile
            from pathlib import Path as _Path
            def _eval_once(candidate_py: str) -> float:
                with tempfile.TemporaryDirectory() as _td:
                    _tmp = _Path(_td)
                    (_tmp / 'sol.py').write_text(candidate_py, encoding='utf-8')
                    wrapper = """
passed = 0
total = 0
def _assert(cond):
    global passed, total
    total += 1
    if cond:
        passed += 1

""" + str(tests_code) + """
print(passed, total)
"""
                    (_tmp / 'tests_runner.py').write_text(wrapper, encoding='utf-8')
                    try:
                        proc = subprocess.run([sys.executable, str(_tmp / 'tests_runner.py')], cwd=str(_tmp), capture_output=True, text=True, timeout=3)
                        out = proc.stdout.strip().split()
                        if len(out) >= 2:
                            p, t = int(out[0]), int(out[1])
                            return float(p) / float(max(1, t))
                        return 0.0
                    except Exception:
                        return 0.0
            return _eval_once
        if args.reward == 'code_passk':
            tests_code = ex.get('tests', '') or ex.get('tests_code', '')
            k = max(1, int(args.pass_k))
            if not tests_code:
                return lambda pred: 0.0
            # Approximate pass@k by sampling k candidates per step and taking success rate
            import subprocess, sys, tempfile
            from pathlib import Path as _Path
            def _eval_candidate(candidate_py: str) -> float:
                with tempfile.TemporaryDirectory() as _td:
                    _tmp = _Path(_td)
                    (_tmp / 'sol.py').write_text(candidate_py, encoding='utf-8')
                    wrapper = """
passed = 0
total = 0
def _assert(cond):
    global passed, total
    total += 1
    if cond:
        passed += 1

""" + str(tests_code) + """
print(passed, total)
"""
                    (_tmp / 'tests_runner.py').write_text(wrapper, encoding='utf-8')
                    try:
                        proc = subprocess.run([sys.executable, str(_tmp / 'tests_runner.py')], cwd=str(_tmp), capture_output=True, text=True, timeout=3)
                        out = proc.stdout.strip().split()
                        if len(out) >= 2:
                            p, t = int(out[0]), int(out[1])
                            return 1.0 if p == max(1, t) else float(p) / float(max(1, t))
                        return 0.0
                    except Exception:
                        return 0.0
            def _r(_: str) -> float:
                # Generate k samples and compute success rate
                success = 0.0
                for _i in range(k):
                    cand_ids, _, _ = _sample(str(ex.get('prompt','')))
                    cand_text = tok.decode(cand_ids)
                    success += 1.0 if _eval_candidate(cand_text) >= 1.0 else 0.0
                return float(success / float(k))
            return _r
        if args.reward == 'clip':
            img_path = ex.get('image','')
            if not img_path:
                return lambda pred: 0.0
            try:
                import open_clip  # type: ignore
                from PIL import Image  # type: ignore
                model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai')
                tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
                device = args.device
                model = model.to(device).eval()
                def _clip(pred: str) -> float:
                    try:
                        im = Image.open(img_path).convert('RGB')
                        imt = preprocess(im).unsqueeze(0).to(device)
                        with torch.no_grad():
                            image_features = model.encode_image(imt)
                            text = tokenizer([pred]).to(device)
                            text_features = model.encode_text(text)
                            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                            sim = (image_features * text_features).sum(dim=-1).item()
                        return max(0.0, min(1.0, 0.5 * (sim + 1.0)))
                    except Exception:
                        return 0.0
                return _clip
            except Exception:
                return lambda pred: 0.0
        if args.reward == 'wer':
            ref = ex.get('reference', '') or ( (ex.get('target') or '') )
            if not ref:
                return lambda pred: 0.0
            try:
                from jiwer import wer as _wer  # type: ignore
                return lambda pred: float(max(0.0, 1.0 - float(_wer(ref, pred))))
            except Exception:
                import difflib
                return lambda pred: float(difflib.SequenceMatcher(None, ref, pred).ratio())
        if args.reward == 'rm' and rm_head is not None:
            def _rm(pred: str) -> float:
                try:
                    text = (ex.get('prompt', '') + "\n" + pred) if ex.get('prompt') else pred
                    ids = torch.tensor([tok.encode(text)[:512]], dtype=torch.long, device=args.device)
                    with torch.no_grad():
                        out = policy(ids, use_cache=False)
                        hid = out[-1] if isinstance(out, tuple) else policy.embed(ids)  # type: ignore[attr-defined]
                        h = hid.mean(dim=1) if isinstance(hid, torch.Tensor) and hid.dim()==3 else policy.embed(ids).mean(dim=1)  # type: ignore[attr-defined]
                        r = rm_head(h).squeeze(-1)
                    return float(r.item())
                except Exception:
                    return 0.0
            return _rm
        if args.reward == 'mos_proxy':
            def _mos(pred: str) -> float:
                txt = pred.strip()
                n = len(txt)
                if n == 0:
                    return 0.0
                alpha = sum(ch.isalpha() for ch in txt) / float(n)
                len_score = 1.0 - abs(n - 128) / 128.0
                len_score = max(0.0, min(1.0, len_score))
                return float(max(0.0, min(1.0, 0.5 * alpha + 0.5 * len_score)))
            return _mos
        return lambda pred: 0.0

    for step in range(1, args.steps + 1):
        ex = prompts[(step - 1) % len(prompts)]
        prompt = str(ex.get('prompt', ''))
        gen_ids, full_ids, logp_traj = _sample(prompt)

        # Reward: simple exact-match contain if target present; extend to programmatic per-domain metrics.
        rewarder = _make_reward(ex)
        pred = tok.decode(gen_ids)
        reward = float(rewarder(pred))

        # Compute advantages with GAE (single trajectory)
        T = logp_traj.size(1)
        returns = torch.zeros(T, device=args.device)
        adv = torch.zeros(T, device=args.device)
        running_return = torch.tensor(reward, device=args.device)
        last_adv = torch.tensor(0.0, device=args.device)
        for t in reversed(range(T)):
            delta = running_return  # no value function baseline (can plug later)
            last_adv = delta + args.gamma * args.gae_lambda * last_adv
            adv[t] = last_adv
            returns[t] = running_return
        old_logp = logp_traj.detach()

        # PPO epochs over mini-batches
        for _ in range(args.ppo_epochs):
            # Flatten single-trajectory
            idx = torch.randperm(T, device=args.device)
            mb = max(1, int(T // args.mini_batch))
            for i in range(0, T, mb):
                sel = idx[i:i+mb]
                # Recompute action logprobs under current policy
                # Build input_ids prefix for each step token
                # For simplicity, recompute on the whole sequence and index
                input_ids = full_ids[:, :-1]
                logits = policy(input_ids)
                if isinstance(logits, tuple):
                    logits = logits[0]
                dist = torch.distributions.Categorical(logits=logits[:, -T:, :].squeeze(0))
                new_logp = dist.log_prob(torch.tensor(gen_ids[-T:], device=args.device))
                # Optional KL penalty vs reference policy
                kl_term = torch.tensor(0.0, device=args.device)
                if ref is not None and float(args.kl_coef) > 0.0:
                    with torch.no_grad():
                        ref_logits = ref(input_ids)
                        if isinstance(ref_logits, tuple):
                            ref_logits = ref_logits[0]
                        ref_dist = torch.distributions.Categorical(logits=ref_logits[:, -T:, :].squeeze(0))
                        ref_logp = ref_dist.log_prob(torch.tensor(gen_ids[-T:], device=args.device))
                    # KL(new || ref) approximated on selected tokens
                    # Using logp difference as a proxy per action: kl ≈ (new_logp - ref_logp)
                    kl_term = (new_logp[sel] - ref_logp[sel]).mean()
                ratio = torch.exp(new_logp[sel] - old_logp[0, sel])
                unclipped = ratio * adv[sel]
                clipped = torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio) * adv[sel]
                policy_loss = -torch.min(unclipped, clipped).mean()
                entropy = dist.entropy()[sel].mean()
                loss = policy_loss - args.entropy_coef * entropy + float(args.kl_coef) * kl_term
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
        if step % 10 == 0 or step == 1:
            print(f"step {step}/{args.steps} reward={reward:.3f}")

    try:
        from omnicoder.utils.checkpoint import save_with_sidecar  # type: ignore
    except Exception:
        save_with_sidecar = None  # type: ignore
    if callable(save_with_sidecar):
        final = save_with_sidecar(args.out, policy.state_dict(), meta={'train_args': {'steps': int(args.steps)}})
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        _safe_save({k: (v.detach().to('cpu') if torch.is_tensor(v) else v) for k, v in policy.state_dict().items()}, args.out)
        final = args.out
    # Best-checkpoint via reward proxy if available
    try:
        from omnicoder.utils.checkpoint import maybe_save_best  # type: ignore
        if 'reward' in locals():
            maybe_save_best(args.out, policy, 'ppo_reward', float(reward), higher_is_better=True)
    except Exception:
        pass
    print(f"Saved PPO-updated policy to {final}")


if __name__ == '__main__':
    main()


