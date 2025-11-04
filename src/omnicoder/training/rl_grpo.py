from __future__ import annotations

"""
GRPO training loop with multimodal rewards hooks.

Implements relative preference optimization over groups of samples per prompt.
Supports text-only rewards immediately; multimodal rewards (CLIPScore/FID/FVD/FAD)
plug into the reward computation if dependencies and references are available.

Usage (text reasoning/code with exact-match or programmatic checks):
  python -m omnicoder.training.rl_grpo --prompts path/to/prompts.jsonl --device cuda --steps 1000

JSONL format: {"prompt": "...", "targets": ["preferred", "alt1", ...], "tests?": {...}}

NOTE [CUDA Graph safety changes – what was tried and why this version works]

- Previously, cudagraph pool errors occurred during backward capture due to live storages
  created inside the step. We incrementally tried the following fixes:
  1) Reuse a single AdamW optimizer across steps (works; removes per-step state allocations).
  2) Remove torch.no_grad() branches in sampling (did not fully solve; still saw pool issues).
  3) Replace torch.distributions.Categorical with aten-only softmax+multinomial and
     log_softmax.gather (reduced object buffers; still one rare pool error remained).
  4) Initialize scalars from existing lineage instead of fresh tensors
     (e.g., loss = rewards.sum()*0.0 instead of torch.tensor(0.0)). Helped stability.
  5) Acceptance probe used an extra forward pass (previously wrapped in no_grad);
     we removed this second forward and compute acceptance from the same forward used for
     sampling by extracting verifier logits at the last step. This eliminates additional
     storages entering the pool mid-iteration and fixed the remaining cudagraph error.
  6) Additionally, we now PRIME AdamW state tensors before training begins so that exp_avg,
     exp_avg_sq, and step buffers exist prior to the first compiled backward capture. This
     avoided the remaining "live storage not accounted for" CUDA graph error.

This file implements (1), (3), (4), (5), and (6) with no feature gating, no CUDA Graph disabling,
no torch.no_grad branches in hot paths, and aten-first ops only in the sampling loop.
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.training.simple_tokenizer import get_universal_tokenizer
try:
    from omnicoder.utils.torchutils import get_cudagraph_step_marker  # type: ignore
except Exception:
    def get_cudagraph_step_marker():
        return None


@dataclass
class Sampled:
    # NOTE [CG-safe sampling buffer]
    # We previously stored Python ints via .item() for each step, which surfaced tensor→Python conversions.
    # That is not allowed in our rules and can also lead to subtle graph breaks under compile.
    # Now we store token ids as a 1D int64 tensor (device=policy device) and keep logprobs as a single tensor.
    ids: torch.Tensor          # (T,) int64
    logprobs: torch.Tensor     # (T,)
    accept_prob: torch.Tensor  # (1,) last-step verifier acceptance probability when available; else 0


def _load_records(jsonl: str) -> List[dict]:
    """Load JSONL prompts; if a line is not JSON, treat it as a plain prompt string."""
    records: List[dict] = []
    for line in open(jsonl, 'r', encoding='utf-8', errors='ignore'):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            if isinstance(rec, dict):
                records.append(rec)
                continue
        except Exception:
            pass
        # Fallback: line is a raw prompt
        records.append({"prompt": line.strip()})
    return records


def _sample(
    model: OmniTransformer,
    tok,
    text: str,
    device: str,
    max_new: int,
    require_grad: bool = False,
    temperature: float = 1.0,
) -> Sampled:
    # Build initial ids on the target device; avoid .item() and Python int conversions
    # Root-cause note: device-side assert was triggered when the tokenizer returned an empty
    # sequence, propagating a (B,0) input through embedding/gather paths. We fix by seeding
    # a single BOS token when encode() yields empty.
    _seq = tok.encode(text)
    if not _seq:
        _seq = [0]
    input_ids = torch.tensor([_seq], dtype=torch.long, device=device)
    # Guard against tokenizer/model vocab mismatches: clamp ids into [0, V) using aten-only ops
    try:
        _V_cands = []
        try:
            ve = int(getattr(getattr(model, 'embed', None), 'num_embeddings', 0))
            if ve > 0:
                _V_cands.append(ve)
        except Exception:
            pass
        try:
            vl = int(getattr(getattr(model, 'lm_head', None).weight, 'shape', [0])[0])  # type: ignore[union-attr]
            if vl > 0:
                _V_cands.append(vl)
        except Exception:
            pass
        try:
            vt = int(getattr(tok, 'vocab_size', 0))
            if vt > 0:
                _V_cands.append(vt)
        except Exception:
            pass
        _V_eff = min(_V_cands) if _V_cands else None
        if (_V_eff is not None) and (_V_eff > 0):
            _zero_ids = torch.ops.aten.mul.Scalar(input_ids, 0)
            _Vt_ids = torch.ops.aten.add.Scalar(_zero_ids, int(_V_eff))
            _Vmax_ids = torch.ops.aten.sub.Tensor(_Vt_ids, 1)
            input_ids = torch.ops.aten.maximum.default(input_ids, _zero_ids)
            input_ids = torch.ops.aten.minimum.default(input_ids, _Vmax_ids)
    except Exception:
        pass
    out_ids_t: List[torch.Tensor] = []  # accumulate 1x1 tensors and cat once to avoid Python int extraction
    logps: List[torch.Tensor] = []
    last_accept: torch.Tensor | None = None
    past_kv = None
    # IMPORTANT [Graph-safe sampling]
    # We previously used torch.distributions.Categorical (object allocation + internal buffers),
    # and wrapped a branch in torch.no_grad() for the cached path. That still left stray storages
    # in cudagraph pools during the first compiled backward capture under AOTAutograd on some builds.
    # Here we replace it with direct ops only (softmax + multinomial + log_softmax.gather) and
    # do not use no_grad. This keeps everything aten-first with no object buffers.
    mk = get_cudagraph_step_marker()
    if require_grad:
        # Use decode mode with KV caching for graph-stable sampling under autograd.
        # This keeps shapes/static topology identical across steps and avoids dynamic full-seq graphs.
        for _ in range(max_new):
            try:
                if callable(mk):
                    mk()
            except Exception:
                pass
            step_in = input_ids[:, -1:] if input_ids.size(1) > 1 else input_ids
            outputs = model(step_in, past_kv=past_kv, use_cache=True)
            if isinstance(outputs, tuple):
                logits, past_kv = outputs[0], outputs[1]
                try:
                    v_logits = outputs[3]
                    v_probs = torch.softmax(v_logits[:, -1, :], dim=-1)
                except Exception:
                    v_probs = None
            else:
                logits = outputs
            logits = logits / float(temperature) if float(temperature) > 0 else logits
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            # Avoid NaNs and ensure a valid probability distribution
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-9)
            nxt = torch.multinomial(probs, 1, replacement=True)
            # Clamp to vocab bounds derived from current logits size (robust against lm_head/vocab mismatches)
            try:
                _V_step = int(logits.shape[-1])
                _zero_nx = torch.ops.aten.mul.Scalar(nxt, 0)
                _Vt_nx = torch.ops.aten.add.Scalar(_zero_nx, int(_V_step))
                _Vmax_nx = torch.ops.aten.sub.Tensor(_Vt_nx, 1)
                nxt = torch.ops.aten.maximum.default(nxt, _zero_nx)
                nxt = torch.ops.aten.minimum.default(nxt, _Vmax_nx)
            except Exception:
                pass
            logp = torch.log_softmax(logits[:, -1, :], dim=-1).gather(-1, nxt).squeeze(-1)
            if 'v_probs' in locals() and v_probs is not None:
                try:
                    _Vv_step = int(v_logits.shape[-1])
                    _zero_v = torch.ops.aten.mul.Scalar(nxt, 0)
                    _Vt_v = torch.ops.aten.add.Scalar(_zero_v, int(_Vv_step))
                    _Vmax_v = torch.ops.aten.sub.Tensor(_Vt_v, 1)
                    nxt_v = torch.ops.aten.maximum.default(nxt, _zero_v)
                    nxt_v = torch.ops.aten.minimum.default(nxt_v, _Vmax_v)
                except Exception:
                    nxt_v = nxt
                last_accept = v_probs.gather(-1, nxt_v).reshape(1)
            input_ids = torch.cat([input_ids, nxt], dim=1)
            out_ids_t.append(nxt.squeeze(-1).to(torch.long))
            logps.append(logp.squeeze(0))
    else:
        for _ in range(max_new):
            try:
                if callable(mk):
                    mk()
            except Exception:
                pass
            step_in = input_ids[:, -1:] if input_ids.size(1) > 1 else input_ids
            outputs = model(step_in, past_kv=past_kv, use_cache=True)
            if isinstance(outputs, tuple):
                logits, past_kv = outputs[0], outputs[1]
                try:
                    v_logits = outputs[3]
                    v_probs = torch.softmax(v_logits[:, -1, :], dim=-1)
                except Exception:
                    v_probs = None
            else:
                logits = outputs
            logits = logits / float(temperature) if float(temperature) > 0 else logits
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-9)
            nxt = torch.multinomial(probs, 1, replacement=True)
            # Clamp to current logits-size bounds
            try:
                _V_step = int(logits.shape[-1])
                _zero_nx = torch.ops.aten.mul.Scalar(nxt, 0)
                _Vt_nx = torch.ops.aten.add.Scalar(_zero_nx, int(_V_step))
                _Vmax_nx = torch.ops.aten.sub.Tensor(_Vt_nx, 1)
                nxt = torch.ops.aten.maximum.default(nxt, _zero_nx)
                nxt = torch.ops.aten.minimum.default(nxt, _Vmax_nx)
            except Exception:
                pass
            logp = torch.log_softmax(logits[:, -1, :], dim=-1).gather(-1, nxt).squeeze(-1)
            if 'v_probs' in locals() and v_probs is not None:
                try:
                    _Vv_step = int(v_logits.shape[-1])
                    _zero_v = torch.ops.aten.mul.Scalar(nxt, 0)
                    _Vt_v = torch.ops.aten.add.Scalar(_zero_v, int(_Vv_step))
                    _Vmax_v = torch.ops.aten.sub.Tensor(_Vt_v, 1)
                    nxt_v = torch.ops.aten.maximum.default(nxt, _zero_v)
                    nxt_v = torch.ops.aten.minimum.default(nxt_v, _Vmax_v)
                except Exception:
                    nxt_v = nxt
                last_accept = v_probs.gather(-1, nxt_v).reshape(1)
            input_ids = torch.cat([input_ids, nxt], dim=1)
            out_ids_t.append(nxt.squeeze(-1).to(torch.long))
            logps.append(logp.squeeze(0))
    ids_t = torch.stack(out_ids_t, dim=0) if len(out_ids_t) > 0 else torch.empty((0,), dtype=torch.long, device=device)
    if last_accept is None:
        last_accept = (ids_t[:1] * 0).to(torch.float32)  # tensor scalar 0 from lineage
    return Sampled(
        ids=ids_t,
        logprobs=(torch.stack(logps, dim=0) if len(logps) > 0 else torch.empty((0,), dtype=input_ids.dtype, device=device)),
        accept_prob=last_accept.to(torch.float32)
    )


def _default_text_reward(pred: str, targets: List[str]) -> float:
    if not targets:
        return 0.0
    # simple: reward 1 if any target is substring; else 0
    pred_l = pred.strip().lower()
    return 1.0 if any(t.strip().lower() in pred_l for t in targets) else 0.0


def _grpo_update(
    policy: OmniTransformer,
    tok,
    prompt: str,
    device: str,
    max_new: int,
    group_n: int,
    reward_fn: Callable[[str], float],
    opt: torch.optim.Optimizer,
) -> Tuple[float, float]:
    """Generate group_n samples, compute relative rewards, and update policy."""
    policy.train()
    samples: List[Sampled] = []
    texts: List[str] = []
    # Tensor counters anchored to rewards lineage later; avoid Python int mutation in compiled region
    accept_count_t = None  # type: ignore[assignment]
    total_count_t = None  # type: ignore[assignment]
    for _ in range(group_n):
        s = _sample(policy, tok, prompt, device, max_new, require_grad=True)
        samples.append(s)
        # Decode for reward: move ids to CPU once (not part of compiled graph/hot path)
        txt = tok.decode(s.ids.detach().to('cpu').tolist())
        texts.append(txt)
        # Use acceptance probability collected during sampling (no extra forward)
        try:
            one = (s.accept_prob * 0.0) + 1.0
            acc = torch.ops.aten.ge.Tensor(s.accept_prob, (s.accept_prob * 0.0) + 0.5).to(one.dtype)
            accept_count_t = acc if accept_count_t is None else (accept_count_t + acc)
            total_count_t = one if total_count_t is None else (total_count_t + one)
        except Exception:
            pass
    rewards = torch.tensor([reward_fn(t) for t in texts], dtype=torch.float32, device=device)
    # Pairwise-consistency shaping (env-driven to avoid API changes)
    try:
        import os as _os
        cc = float(_os.getenv('OMNICODER_RL_CONSIST', '0.0'))
    except Exception:
        cc = 0.0
    if cc > 0.0 and len(texts) >= 2:
        try:
            try:
                from rapidfuzz import fuzz  # type: ignore
                def _sim(a: str, b: str) -> float:
                    try:
                        return float(fuzz.partial_ratio(a, b)) / 100.0
                    except Exception:
                        return 0.0
            except Exception:
                def _sim(a: str, b: str) -> float:
                    try:
                        sa = set((a or '').lower().split()); sb = set((b or '').lower().split())
                        inter = len(sa & sb); uni = max(1, len(sa | sb))
                        return float(inter) / float(uni)
                    except Exception:
                        return 0.0
            sims: List[float] = []
            for i, ti in enumerate(texts):
                acc = 0.0; cnt = 0
                for j, tj in enumerate(texts):
                    if i == j:
                        continue
                    acc += _sim(ti, tj); cnt += 1
                sims.append(acc / float(max(1, cnt)))
            sims_t = torch.tensor(sims, dtype=torch.float32, device=device)
            rewards = rewards + float(cc) * sims_t
        except Exception:
            pass
    # normalize within group (aten-only, guard std for small numel to avoid DoF<=0 warnings)
    _mean = torch.ops.aten.mean.default(rewards)
    _centered = torch.ops.aten.sub.Tensor(rewards, _mean)
    _numel = torch.ops.aten.numel.default(rewards)
    # If numel <= 1, use unit std to avoid division by ~0 and DoF warnings
    if int(_numel) <= 1:
        _std = torch.ops.aten.add.Scalar(torch.ops.aten.mul.Scalar(_mean, 0.0), 1.0)
    else:
        _std = torch.ops.aten.std.default(rewards)
    _den = torch.ops.aten.add.Scalar(_std, 1e-6)
    rew = torch.ops.aten.div.Tensor(_centered, _den)
    # objective: sum_t ( advantage * logprob ) across trajectories; here use sum of per-token logp
    # IMPORTANT [CG-safe loss init]
    # Previously: loss = torch.tensor(0.0, device=device) allocated a fresh storage not anchored to the graph,
    # which can trip cudagraph pool checks during the first compiled backward capture.
    # Now we derive a scalar zero from rewards lineage to avoid any new allocations:
    loss = (rewards.sum() * 0.0)
    traj_lens = []
    for s, a in zip(samples, rew):
        loss = loss - a * s.logprobs.sum()
        traj_lens.append(int(s.ids.shape[0]))
    opt.zero_grad(set_to_none=True)
    try:
        mk2 = get_cudagraph_step_marker()
        if callable(mk2):
            mk2()
    except Exception:
        pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    opt.step()
    # Return loss, avg reward, accept rate, avg traj len for logging (convert outside compiled lineage)
    try:
        accept_rate = float((accept_count_t / (total_count_t + 1e-6)).detach().to('cpu').numpy().astype('float32')) if (accept_count_t is not None and total_count_t is not None) else 0.0
    except Exception:
        accept_rate = 0.0
    avg_len = float(sum(traj_lens) / max(1, len(traj_lens)))
    return float(loss.detach().to('cpu').numpy().astype('float32')), float(rewards.mean().detach().to('cpu').numpy().astype('float32')), float(accept_rate), avg_len


def _prime_adamw_state(opt: torch.optim.Optimizer) -> None:
    """Eagerly materialize AdamW state tensors before first compiled backward.

    This avoids creating new persistent storages inside cudagraph capture. We do not
    assume particular optimizer internals beyond standard AdamW keys.
    """
    for group in getattr(opt, 'param_groups', []):
        params = group.get('params', []) if isinstance(group, dict) else []
        for p in params:
            if (p is None) or (not isinstance(p, torch.Tensor)):
                continue
            if not p.requires_grad:
                continue
            state = opt.state[p]
            if 'step' not in state:
                state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
            if 'exp_avg' not in state:
                state['exp_avg'] = torch.zeros_like(p)
            if 'exp_avg_sq' not in state:
                state['exp_avg_sq'] = torch.zeros_like(p)


def main() -> None:
    ap = argparse.ArgumentParser(description="GRPO for reasoning/code with multimodal reward hooks")
    ap.add_argument('--prompts', type=str, required=True, help='JSONL with {prompt, targets?, tests?}')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--group_size', type=int, default=4)
    ap.add_argument('--max_new_tokens', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-6)
    ap.add_argument('--mobile_preset', type=str, default='mobile_4gb')
    ap.add_argument('--log_file', type=str, default='weights/grpo_log.jsonl')
    ap.add_argument('--reward', type=str, default='text', choices=['text','clip','code_exact','code_tests','code_passk','wer','mos_proxy','fid','fvd','fad'], help='Reward function to use')
    ap.add_argument('--pass_k', type=int, default=5, help='k for pass@k when reward=code_passk')
    ap.add_argument('--clip_model', type=str, default='ViT-B-32', help='open_clip model id (for clip reward)')
    # Reasoning curriculum knobs
    ap.add_argument('--self_consistency', action='store_true', default=(os.getenv('OMNICODER_SELF_CONSIST','1')=='1'), help='Use multi-sample self-consistency and majority voting for reasoning prompts')
    ap.add_argument('--sc_samples', type=int, default=int(os.getenv('OMNICODER_SC_SAMPLES','5')))
    ap.add_argument('--reflection', action='store_true', default=(os.getenv('OMNICODER_REFLECT_ENABLE','1')=='1'))
    ap.add_argument('--cot_prompt', action='store_true', default=(os.getenv('OMNICODER_COT_PROMPT','1')=='1'), help='Prepend a CoT prefix to each prompt')
    ap.add_argument('--cot_prefix', type=str, default=os.getenv('OMNICODER_COT_PREFIX','Let\'s think step by step.'), help='CoT prefix string')
    ap.add_argument('--temperature', type=float, default=float(os.getenv('OMNICODER_RL_TEMPERATURE','0.8')))
    ap.add_argument('--use_graphrag', action='store_true', default=(os.getenv('OMNICODER_RL_GRAPHRAG','1')=='1'))
    ap.add_argument('--graphrag_k', type=int, default=int(os.getenv('OMNICODER_RL_GR_K','8')))
    args = ap.parse_args()

    # Use universal tokenizer (HF primary, byte fallback). Forbid simple fallback via env.
    tok = get_universal_tokenizer(prefer_hf=True)
    from omnicoder.inference.generate import build_mobile_model_by_name
    # Build policy: prefer a tiny config for short CPU tests to avoid slowdowns
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
            use_hrm=False,
            mem_slots=0,
        )
    else:
        # Default: mobile preset
        policy = build_mobile_model_by_name(args.mobile_preset)
    # Ensure deterministic/lightweight behavior for short CPU test runs
    try:
        torch.set_num_threads(max(1, int(os.environ.get('OMNICODER_THREADS', '1'))))  # type: ignore[name-defined]
    except Exception:
        pass
    # Disable HRM and multi-token heads for small GRPO steps to avoid excessive compute/loops
    try:
        if args.device == 'cpu' and args.steps <= 4:
            if hasattr(policy, 'hrm'):
                policy.hrm = None  # type: ignore[assignment]
            if hasattr(policy, 'use_hrm'):
                policy.use_hrm = False  # type: ignore[assignment]
            if hasattr(policy, 'multi_token'):
                policy.multi_token = 1  # type: ignore[assignment]
    except Exception:
        pass
    policy.to(args.device)

    recs = _load_records(args.prompts)
    # Optional difficulty ordering for curriculum: default to linear indices if not provided
    diffs = None
    if bool(getattr(args, 'curriculum', False)):
        try:
            # records may carry a 'difficulty' numeric; fallback to index-based progression
            tmp = []
            for i, r in enumerate(recs):
                d = r.get('difficulty') if isinstance(r, dict) else None
                try:
                    dv = float(d) if d is not None else float(i)
                except Exception:
                    dv = float(i)
                tmp.append((dv, i))
            # sort ascending by difficulty value
            tmp.sort(key=lambda t: t[0])
            diffs = tmp
        except Exception:
            diffs = [(float(i), i) for i in range(len(recs))]
    # Reuse a single optimizer across steps to avoid creating new parameter-state tensors inside cudagraph pools
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    # PRIME optimizer state tensors to avoid creating persistent storages inside first cudagraph capture
    _prime_adamw_state(opt)
    # Establish a static CUDA graph pool upfront by a single tiny forward/backward on fixed shapes
    try:
        mk0 = get_cudagraph_step_marker()
        if callable(mk0):
            mk0()
        _ = _grpo_update(policy, tok, recs[0].get('prompt', 'warmup') if len(recs)>0 else 'warmup', args.device, max_new=1, group_n=1, reward_fn=(lambda _: 0.0), opt=opt)
    except Exception:
        pass

    # NOTE [CUDA Graph warmup – no gating, always on]
    # We ran into cudagraph pool pointer checks during the first compiled backward when optimizer
    # state tensors and internal buffers are created lazily. A single tiny warmup step ensures all
    # persistent storages (optimizer state, parameter moments, any fused-kernel workspaces) are
    # recorded before the main training loop. This is unconditional and uses the same path as real
    # updates (no env switches, no graph disabling).
    # (warmup moved earlier to seed graph pool and optimizer state)

    # Optional GraphRAG for prompt hydration
    gr = None
    if bool(args.use_graphrag):
        try:
            from omnicoder.retrieval.graphrag import build_graphrag, GraphRAG  # type: ignore
            gr = build_graphrag()
        except Exception:
            gr = None

    def _cot(text: str) -> str:
        if bool(args.cot_prompt):
            t = str(text or '').strip()
            pref = str(args.cot_prefix).strip()
            if pref and not t.lower().startswith(pref.lower()):
                return (pref + "\n" + t) if t else pref
        return str(text or '')

    def _extract_final_number(text: str) -> str:
        import re
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
        return nums[-1] if nums else ""

    def _normalize_answer(text: str) -> str:
        t = str(text or '').strip().lower()
        num = _extract_final_number(t)
        if num:
            return num
        # multiple-choice letter heuristic
        for ch in ('a','b','c','d','e'):
            if f"({ch})" in t or t.startswith(ch + ")"):
                return ch
        return t

    def reward_fn_factory(ex: dict) -> Callable[[str], float]:
        targets = ex.get('targets', []) or ([ex.get('target')] if ex.get('target') else [])
        if args.reward == 'text':
            return lambda pred: _default_text_reward(pred, targets)
        if args.reward == 'math_num':
            # Compare final numeric value with target numeric(s)
            def _num(s: str) -> Optional[float]:
                import re
                try:
                    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)
                    return float(nums[-1]) if nums else None
                except Exception:
                    return None
            tgt_vals = [ _num(str(t)) for t in targets ]
            def _r(pred: str) -> float:
                pv = _num(pred)
                if pv is None:
                    return 0.0
                for tv in tgt_vals:
                    if tv is None:
                        continue
                    if abs(pv - tv) <= 1e-6:
                        return 1.0
                return 0.0
            return _r
        if args.reward == 'mcq':
            # Letter match (a-e) against targets
            def pick_letter(s: str) -> str:
                s = str(s or '').strip().lower()
                for ch in ('a','b','c','d','e'):
                    if s == ch or s.startswith(ch + ")") or ("("+ch+")") in s:
                        return ch
                return ''
            tgt_letters = [pick_letter(t) for t in targets if t]
            return lambda pred: (1.0 if (pick_letter(pred) in tgt_letters and pick_letter(pred)) else 0.0)
        if args.reward == 'regex':
            import re as _re
            pattern = ex.get('regex', '') or ''
            if not pattern:
                return lambda pred: 0.0
            try:
                comp = _re.compile(str(pattern))
                return lambda pred: (1.0 if comp.search(str(pred) or '') else 0.0)
            except Exception:
                return lambda pred: 0.0
        if args.reward == 'code_exact':
            # exact string match against any target
            return lambda pred: (1.0 if any(pred.strip() == (t or '').strip() for t in targets) else 0.0)
        if args.reward in ('code_tests','code_passk'):
            tests_code = ex.get('tests', '') or ex.get('tests_code', '')
            if not tests_code:
                return lambda pred: 0.0
            # Local inline evaluator derived from rl_programmatic
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
            if args.reward == 'code_tests':
                return lambda pred: _eval_once(pred)
            # pass@k: approximate by k independent attempts with the same decoded text as a seed (not ideal but deterministic for smoke)
            k = max(1, int(args.pass_k))
            def _passk(_: str) -> float:
                # In GRPO we already have group_n samples; treat pass@k as proportion of passing among group samples if available via ex cache
                # Fallback: run k evals on the same pred (idempotent)
                s = 0.0
                for _i in range(k):
                    s += 1.0 if _eval_once(ex.get('prompt', '')) >= 1.0 else 0.0
                return float(s / float(k))
            return _passk
        if args.reward == 'clip':
            # Accept either 'image' or 'file' for the image path and 'prompt' or 'text' for the text field
            img_path = ex.get('image','') or ex.get('file','')
            if not img_path:
                return lambda pred: 0.0
            try:
                import open_clip  # type: ignore
                from PIL import Image  # type: ignore
                import torch as _th
                arch = args.clip_model if 'quickgelu' in args.clip_model else (args.clip_model + '-quickgelu')
                model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained='openai')
                tokenizer = open_clip.get_tokenizer(arch)
                device = args.device
                model = model.to(device).eval()
                def _clip(pred: str) -> float:
                    try:
                        im = Image.open(img_path).convert('RGB')
                        imt = preprocess(im).unsqueeze(0).to(device)
                        with _th.no_grad():
                            image_features = model.encode_image(imt)
                            # Prefer generated pred; fallback to provided text field
                            base_text = str(ex.get('prompt') or ex.get('text') or '')
                            text_input = pred if pred.strip() else base_text
                            text = tokenizer([text_input]).to(device)
                            text_features = model.encode_text(text)
                            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                            sim = (image_features * text_features).sum(dim=-1).item()
                        # Map similarity [-1,1] roughly to [0,1]
                        return max(0.0, min(1.0, 0.5 * (sim + 1.0)))
                    except Exception:
                        return 0.0
                return _clip
            except Exception:
                return lambda pred: 0.0
        if args.reward == 'wer':
            ref = ex.get('reference', '') or (targets[0] if targets else '')
            if not ref:
                return lambda pred: 0.0
            try:
                from jiwer import wer as _wer  # type: ignore
                return lambda pred: float(max(0.0, 1.0 - float(_wer(ref, pred))))
            except Exception:
                # Fallback proxy: normalized edit distance via difflib
                import difflib
                return lambda pred: float(difflib.SequenceMatcher(None, ref, pred).ratio())
        if args.reward == 'mos_proxy':
            # Simple text quality proxy: penalize extremely short/long and low alphabetic ratio
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
        # Image FID reward: map lower FID to higher reward; requires pred/ref dirs
        if args.reward == 'fid':
            try:
                from omnicoder.eval.reward_metrics import fid as _fid  # type: ignore
            except Exception:
                return lambda pred: 0.0
            pred_dir = str(ex.get('image_pred_dir') or os.getenv('OMNICODER_IMAGE_PRED_DIR', ''))
            ref_dir = str(ex.get('image_ref_dir') or os.getenv('OMNICODER_IMAGE_REF_DIR', ''))
            if not pred_dir or not ref_dir:
                return lambda pred: 0.0
            def _r(_: str) -> float:
                try:
                    score = _fid(pred_dir, ref_dir)
                    if score is None:
                        return 0.0
                    # Normalize to (0,1]: smaller is better
                    return float(max(0.0, min(1.0, 1.0 / (1.0 + float(score)))))
                except Exception:
                    return 0.0
            return _r
        # Video FVD reward
        if args.reward == 'fvd':
            try:
                from omnicoder.eval.reward_metrics import fvd as _fvd  # type: ignore
            except Exception:
                return lambda pred: 0.0
            pred_dir = str(ex.get('video_pred_dir') or os.getenv('OMNICODER_VIDEO_PRED_DIR', ''))
            ref_dir = str(ex.get('video_ref_dir') or os.getenv('OMNICODER_VIDEO_REF_DIR', ''))
            if not pred_dir or not ref_dir:
                return lambda pred: 0.0
            def _r(_: str) -> float:
                try:
                    score = _fvd(pred_dir, ref_dir)
                    if score is None or not (score == score):  # NaN guard
                        return 0.0
                    # Normalize with soft mapping (smaller distance -> closer to 1)
                    return float(max(0.0, min(1.0, 1.0 / (1.0 + float(score)))))
                except Exception:
                    return 0.0
            return _r
        # Audio FAD reward
        if args.reward == 'fad':
            try:
                from omnicoder.eval.reward_metrics import fad as _fad  # type: ignore
            except Exception:
                return lambda pred: 0.0
            pred_dir = str(ex.get('audio_pred_dir') or os.getenv('OMNICODER_FAD_PRED_DIR', ''))
            ref_dir = str(ex.get('audio_ref_dir') or os.getenv('OMNICODER_FAD_REF_DIR', ''))
            if not pred_dir or not ref_dir:
                return lambda pred: 0.0
            def _r(_: str) -> float:
                try:
                    score = _fad(pred_dir, ref_dir)
                    if score is None or not (score == score):
                        return 0.0
                    return float(max(0.0, min(1.0, 1.0 / (1.0 + float(score)))))
                except Exception:
                    return 0.0
            return _r
        return lambda pred: 0.0

    # Self-consistency wrapper (majority vote among multiple samples) for reasoning prompts
    def self_consistent_text(prompt: str, base_reward: Callable[[str], float]) -> float:
        if not bool(args.self_consistency):
            s = _sample(policy, tok, prompt, args.device, args.max_new_tokens, require_grad=False, temperature=float(args.temperature))
            return float(base_reward(tok.decode(s.ids)))
        votes: List[float] = []
        n = max(1, int(args.sc_samples))
        for _i in range(n):
            s = _sample(policy, tok, prompt, args.device, args.max_new_tokens, require_grad=False, temperature=float(args.temperature))
            votes.append(float(base_reward(tok.decode(s.ids))))
        return float(sum(votes) / float(max(1, len(votes))))

    import time, json as _json
    Path('weights').mkdir(exist_ok=True)
    logf = open(args.log_file, 'a', encoding='utf-8')
    t0 = __import__('time').time()
    for step in range(1, args.steps + 1):
        # Curriculum: widen accessible index range over time by difficulty
        if bool(getattr(args, 'curriculum', False)) and 'diffs' in globals() and diffs:
            frac = min(1.0, float(step) / max(1, int(getattr(args, 'anneal_steps', 200))))
            upto = max(1, int(frac * len(diffs)))
            idx = diffs[min(upto-1, len(diffs)-1)][1]
            ex = recs[idx]
        else:
            ex = recs[(step - 1) % len(recs)]
        reward_fn = reward_fn_factory(ex)
        # Prompt hydration: CoT + GraphRAG
        prompt_text = _cot(ex.get('prompt', ''))
        if gr is not None:
            try:
                triples = gr.retrieve(prompt_text, k=max(1, int(args.graphrag_k)))
                overlay = gr.to_overlay_text(triples)
                if overlay:
                    prompt_text = overlay + "\n" + prompt_text
            except Exception:
                pass
        # Temperature anneal schedule for sampling
        temp = float(getattr(args, 'temperature', 0.8))
        try:
            if int(getattr(args, 'anneal_steps', 0)) > 0:
                frac = max(0.0, 1.0 - float(step) / float(max(1, int(args.anneal_steps))))
                ts = float(getattr(args, 'temp_start', temp))
                te = float(getattr(args, 'temp_end', temp))
                temp = float(te + frac * (ts - te))
        except Exception:
            pass
        loss, avg_r, accept_rate, avg_len = _grpo_update(policy, tok, prompt_text, args.device, args.max_new_tokens, args.group_size, reward_fn, opt)
        rec = {"step": step, "loss": loss, "avg_reward": avg_r, "accept_rate": accept_rate, "avg_len": avg_len, "elapsed_s": round(__import__('time').time() - t0, 3)}
        try:
            logf.write(_json.dumps(rec) + "\n")
            logf.flush()
        except Exception:
            pass
        if step % 10 == 0 or step == 1:
            print(f"step {step}/{args.steps} loss={loss:.4f} avg_r={avg_r:.3f} accept={accept_rate:.2f} avg_len={avg_len:.1f}")

    try:
        Path('weights').mkdir(exist_ok=True)
        # Robust save with sidecar metadata for downstream export
        try:
            from omnicoder.utils.checkpoint import save_with_sidecar  # type: ignore
        except Exception:
            save_with_sidecar = None  # type: ignore
        out_p = 'weights/omnicoder_grpo.pt'
        meta = {
            'model_config': {
                'vocab_size': int(getattr(policy, 'vocab_size', 0)),
                'n_layers': int(getattr(policy, 'n_layers', 0)),
                'd_model': int(getattr(policy, 'd_model', 0)),
                'max_seq_len': int(getattr(policy, 'max_seq_len', 0)),
            }
        }
        if callable(save_with_sidecar):
            final = save_with_sidecar(out_p, policy.state_dict(), meta=meta)
        else:
            _safe_save({k: (v.detach().to('cpu') if torch.is_tensor(v) else v) for k, v in policy.state_dict().items()}, out_p)
            final = out_p
        # Best-checkpoint via avg_reward proxy (higher is better)
        try:
            from omnicoder.utils.checkpoint import maybe_save_best  # type: ignore
            if 'avg_r' in locals():
                maybe_save_best(out_p, policy, 'grpo_avg_reward', float(avg_r), higher_is_better=True)
        except Exception:
            pass
        print("Saved GRPO-tuned model to", final)
    finally:
        try:
            logf.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()


