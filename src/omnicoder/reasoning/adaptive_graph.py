"""
Adaptive Graph-of-Thoughts (AGoT)

Test-time dynamic DAG for adaptive reasoning. This lightweight controller sits
outside the model and, when enabled, performs minimal branch expansion and
reuses short-horizon lookahead results to select the next token.

Design goals:
- No impact when disabled (default).
- O(1) extra memory per token via short caches keyed by recent token tails.
- Bounded compute: width/depth/budget guarded by environment variables.

Environment knobs:
- OMNICODER_AGOT_ENABLE=1 to turn on
- OMNICODER_AGOT_WIDTH (default 3)
- OMNICODER_AGOT_DEPTH (default 2)
- OMNICODER_AGOT_CACHE_TAIL (default 8)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import os
import torch
import time
try:
    from .kv_hints import get_kv_hint, register_kv_hint  # type: ignore
except Exception:
    def get_kv_hint(key, /):
        return None
    def register_kv_hint(key, kv, /):
        return None


@dataclass(slots=True)
class NodeScore:
    margin_sum: float
    proxy_score: float
    cost_tokens: int


class AdaptiveGraphOfThoughts:
    def __init__(self) -> None:
        try:
            self.enabled = os.getenv("OMNICODER_AGOT_ENABLE", "1") == "1"
        except Exception:
            self.enabled = False
        self.width = int(os.getenv("OMNICODER_AGOT_WIDTH", "3"))
        self.depth = int(os.getenv("OMNICODER_AGOT_DEPTH", "2"))
        self.cache_tail = int(os.getenv("OMNICODER_AGOT_CACHE_TAIL", "8"))
        # Optional verifier pruning threshold (soft)
        try:
            self.min_verifier_margin = float(os.getenv("OMNICODER_AGOT_MIN_MARGIN", "0.0"))
        except Exception:
            self.min_verifier_margin = 0.0
        # LRU capacity for score cache
        try:
            self.cache_capacity = int(os.getenv("OMNICODER_AGOT_CACHE_CAP", "2048"))
        except Exception:
            self.cache_capacity = 2048
        # Global compute budget (approximate): max short-rollout tokens per step
        try:
            self.token_budget = int(os.getenv("OMNICODER_AGOT_TOKEN_BUDGET", str(self.width * self.depth)))
        except Exception:
            self.token_budget = max(1, self.width * self.depth)
        # Mixed precision dtype for rollouts on CUDA: bf16 | fp16 | off
        self.mixed_prec = os.getenv("OMNICODER_REASONING_MIXED_PREC", "bf16").strip().lower()
        # Scoring blend and diversity/top-p knobs
        try:
            self.alpha = float(os.getenv("OMNICODER_AGOT_ALPHA", "0.8"))
        except Exception:
            self.alpha = 0.8
        try:
            self.diversity_gamma = float(os.getenv("OMNICODER_AGOT_DIVERSITY_GAMMA", "0.0"))
        except Exception:
            self.diversity_gamma = 0.0
        try:
            self.topp = float(os.getenv("OMNICODER_AGOT_TOPP", "0.0"))
        except Exception:
            self.topp = 0.0
        # Optional wall-clock budget in milliseconds
        try:
            self.ms_budget = int(os.getenv("OMNICODER_AGOT_MS_BUDGET", "0"))
        except Exception:
            self.ms_budget = 0
        # Optional decision telemetry JSONL
        self.log_path = os.getenv("OMNICODER_AGOT_LOG", "").strip()
        # Verbose trace flag
        try:
            self.trace = os.getenv("OMNICODER_TRACE_ENABLE", "1") == "1"
        except Exception:
            self.trace = True
        # Optional distilled verifier head: expects torch.save({'weight': (V,C), 'bias': (V,)})
        self.verifier_W = None
        self.verifier_b = None
        try:
            vpath = os.getenv('OMNICODER_VERIFIER_DISTILL_WEIGHTS', '').strip()
            if vpath:
                ck = torch.load(vpath, map_location='cpu')
                W = ck.get('weight') if isinstance(ck, dict) else None
                b = ck.get('bias') if isinstance(ck, dict) else None
                if isinstance(W, torch.Tensor):
                    self.verifier_W = W.contiguous()
                if isinstance(b, torch.Tensor):
                    self.verifier_b = b.contiguous()
        except Exception:
            self.verifier_W, self.verifier_b = None, None
        # Cache short-horizon scores keyed by recent token tails
        self._score_cache: Dict[Tuple[int, ...], NodeScore] = {}

    def _key_tail(self, ids: torch.Tensor) -> Tuple[int, ...]:
        # ids shape (1, T) long
        try:
            tail = ids[0, -min(int(self.cache_tail), int(ids.size(1))):].tolist()
            return tuple(int(x) for x in tail)
        except Exception:
            return tuple()

    @torch.inference_mode()
    def step(
        self,
        model: torch.nn.Module,
        out_ids: torch.Tensor,
        past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        base_logits: torch.Tensor,
        verifier_logits: Optional[torch.Tensor],
        hidden_out: Optional[torch.Tensor],
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """
        Return one next-token id tensor shaped (1,1) using adaptive short DAG.
        Falls back to greedy argmax if disabled or on error.
        """
        device = base_logits.device
        # Fallback greedy
        def _greedy(logits: torch.Tensor) -> torch.Tensor:
            return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        if not self.enabled:
            return _greedy(base_logits)

        try:
            # Adaptive width from confidence (entropy)
            probs = torch.softmax(base_logits[:, -1, :], dim=-1)
            logp = torch.log(torch.clamp(probs, min=1e-9))
            ent = -torch.sum(probs * logp, dim=-1).item()
            base_w = max(1, int(self.width))
            dyn_w = base_w
            try:
                # increase width for higher entropy up to 2×
                dyn_w = max(1, min(base_w * 2, int(round(base_w * (1.0 + min(1.0, ent / 5.0))))))
            except Exception:
                dyn_w = base_w
            # Candidate selection: optional nucleus (top-p), else top-k
            if self.topp and self.topp > 0.0:
                sorted_p, sorted_i = torch.sort(probs, dim=-1, descending=True)
                cum = torch.cumsum(sorted_p, dim=-1)
                mask = cum <= float(self.topp)
                # Ensure at least one candidate
                if not torch.any(mask):
                    mask[..., 0] = True
                pool = sorted_i[:, mask[0]] if mask.dim() == 2 else sorted_i
                # Clip to width window
                take = min(max(1, dyn_w), int(pool.size(-1)))
                pool = pool[:, :take]
                candidates: List[torch.Tensor] = [pool[:, i : i + 1] for i in range(pool.size(-1))]
            else:
                k = max(1, dyn_w)
                _, topi = torch.topk(probs, k=k, dim=-1)
                candidates: List[torch.Tensor] = [topi[:, i : i + 1] for i in range(k)]
            if self.trace:
                try:
                    import logging as _l
                    _l.getLogger("omnicoder.gen").debug("AGoT: ent=%.4f dyn_w=%d k=%d", float(ent), int(dyn_w), int(len(candidates)))
                except Exception:
                    pass

            # Fast-path batched greedy rollout. Supports verifier head when available
            # This reduces Python-loop overhead by evaluating width candidates together for depth D
            try:
                use_batched = (
                    int(self.depth) > 1 and len(candidates) > 1 and
                    (self.verifier_W is None) and  # skip when only distilled verifier is available
                    os.getenv('OMNICODER_AGOT_BATCH_ROLLOUT', '1') == '1'
                )
            except Exception:
                use_batched = False
            if use_batched:
                try:
                    # Base margin per candidate (prefer verifier when present)
                    ccat = torch.cat(candidates, dim=1)
                    if verifier_logits is not None:
                        base_proxy_vec = torch.softmax(verifier_logits[:, -1, :], dim=-1).gather(-1, ccat).squeeze(0)  # (W,)
                    else:
                        # Prefer distilled verifier margin when available
                        if (self.verifier_W is not None) and isinstance(hidden_out, torch.Tensor):
                            try:
                                tok_idx = ccat[0] if ccat.dim() == 2 else ccat.squeeze(0)
                                h_last = hidden_out[:, -1, :].to(self.verifier_W.device)
                                w_rows = self.verifier_W.index_select(0, tok_idx.to(self.verifier_W.device))  # (W,C)
                                b_rows = self.verifier_b.index_select(0, tok_idx.to(self.verifier_b.device)) if isinstance(self.verifier_b, torch.Tensor) else None  # type: ignore[union-attr]
                                logits_v = torch.matmul(h_last, w_rows.t()) + (b_rows.unsqueeze(0) if isinstance(b_rows, torch.Tensor) else 0.0)
                                base_proxy_vec = torch.sigmoid(logits_v).squeeze(0)
                            except Exception:
                                base_proxy_vec = torch.softmax(base_logits[:, -1, :], dim=-1).gather(-1, ccat).squeeze(0)
                        else:
                            base_proxy_vec = torch.softmax(base_logits[:, -1, :], dim=-1).gather(-1, ccat).squeeze(0)  # (W,)
                    # Prepare batch ids and optional batched KV by expanding along batch dimension
                    tmp_ids_batch = torch.cat([torch.cat([out_ids, cid], dim=1) for cid in candidates], dim=0)
                    tmp_kv_batch = past_kv
                    if isinstance(past_kv, list):
                        try:
                            bs = int(tmp_ids_batch.size(0))
                            tmp_kv_batch = [
                                (
                                    k.expand(bs, -1, -1, -1).contiguous(),
                                    v.expand(bs, -1, -1, -1).contiguous(),
                                )
                                for (k, v) in past_kv
                            ]
                        except Exception:
                            tmp_kv_batch = past_kv
                    margin_sum_vec = base_proxy_vec.clone().to(base_logits.device)
                    cost_steps = 1
                    spent = 0
                    t0 = time.perf_counter()
                    use_autocast = (self.mixed_prec in ("bf16", "fp16")) and ((tmp_ids_batch.device.type == 'cuda') or (base_logits.device.type == 'cuda'))
                    ac_dtype = torch.bfloat16 if self.mixed_prec == 'bf16' else torch.float16
                    steps = 1
                    d = max(1, int(self.depth))
                    while steps < d:
                        if use_autocast:
                            with torch.autocast(device_type='cuda', dtype=ac_dtype):  # type: ignore[arg-type]
                                outs2 = model(tmp_ids_batch[:, -1:], past_kv=tmp_kv_batch, use_cache=True)
                        else:
                            outs2 = model(tmp_ids_batch[:, -1:], past_kv=tmp_kv_batch, use_cache=True)
                        if isinstance(outs2, tuple):
                            log2 = outs2[0]
                            tmp_kv_batch = outs2[1]
                            v2 = outs2[3] if len(outs2) > 3 else None
                        else:
                            log2 = outs2
                            v2 = None
                        pr2 = torch.softmax(log2[:, -1, :], dim=-1)
                        nid = torch.argmax(pr2, dim=-1, keepdim=True)  # (W,1)
                        # Add greedy margin from verifier when available, else proxy prob
                        if v2 is not None:
                            m2 = torch.softmax(v2[:, -1, :], dim=-1).gather(-1, nid).squeeze(1)  # (W,)
                        else:
                            m2 = pr2.gather(-1, nid).squeeze(1)  # (W,)
                        margin_sum_vec += m2
                        tmp_ids_batch = torch.cat([tmp_ids_batch, nid], dim=1)
                        steps += 1
                        cost_steps += 1
                        spent += int(tmp_ids_batch.size(0))
                        if spent >= max(1, int(self.token_budget)):
                            break
                        if int(self.ms_budget) > 0 and (time.perf_counter() - t0) * 1000.0 >= float(self.ms_budget):
                            break
                    # Compose scores and select
                    scores = [(float(m.item()), float(base_proxy_vec[i].item()), int(cost_steps)) for i, m in enumerate(margin_sum_vec)]
                    tail = out_ids[0, -min(int(self.cache_tail), int(out_ids.size(1))):].tolist() if out_ids is not None else []
                    tail_set = set(tail)
                    cand_toks = [int(c.item()) for c in candidates]
                    if scores:
                        margins = [s[0] for s in scores]
                        proxies = [s[1] for s in scores]
                        costs = [s[2] for s in scores]
                        blends = [float(self.alpha) * (m / max(1, c)) + float(1.0 - self.alpha) * float(p) for m, p, c in zip(margins, proxies, costs)]
                        penalties = [float(self.diversity_gamma) * (1.0 if t in tail_set else 0.0) for t in cand_toks]
                        best_i = max(range(len(blends)), key=lambda i: (blends[i] - penalties[i], proxies[i]))
                    else:
                        best_i = 0
                    chosen = candidates[best_i]
                    if self.trace:
                        try:
                            import logging as _l
                            _l.getLogger("omnicoder.gen").debug("AGoT[batch]: chosen=%d norm_margin=%.4f proxy=%.4f", int(chosen.item()), float(scores[best_i][0] / max(1, scores[best_i][2])), float(scores[best_i][1]))
                        except Exception:
                            pass
                    return chosen
                except Exception:
                    # Fall back to per-candidate path on any error
                    pass

            scores: List[Tuple[float, float, int]] = []
            # Evaluate each candidate with short rollouts up to depth D
            spent = 0
            t0 = time.perf_counter()
            # Vectorized proxy gather for initial candidate scores
            try:
                ccat = torch.cat(candidates, dim=1) if candidates else None
            except Exception:
                ccat = None
            base_proxy = None
            if ccat is not None:
                try:
                    base_proxy = torch.softmax(base_logits[:, -1, :], dim=-1).gather(-1, ccat).squeeze(0).tolist()
                except Exception:
                    base_proxy = None
            # Vectorize verifier margins when available
            v_margins: Optional[List[float]] = None
            if verifier_logits is not None and ccat is not None:
                try:
                    v1 = torch.softmax(verifier_logits[:, -1, :], dim=-1)
                    v_margins = v1.gather(-1, ccat).squeeze(0).tolist()
                except Exception:
                    v_margins = None
            # Precompute base tail for score-cache keys to avoid tiny per-candidate torch.cat
            try:
                base_tail: List[int] = out_ids[0, -min(int(self.cache_tail) - 1 if int(self.cache_tail) > 0 else 0, int(out_ids.size(1))):].tolist() if int(self.cache_tail) > 0 else []
            except Exception:
                base_tail = []
            # Precompute candidate token ints for reuse
            try:
                cand_toks: List[int] = [int(c.item()) for c in candidates]
            except Exception:
                cand_toks = []
            # Hoist autocast mode decision
            use_autocast = (self.mixed_prec in ("bf16", "fp16")) and ((out_ids.device.type == 'cuda') or (base_logits.device.type == 'cuda'))
            ac_dtype = torch.bfloat16 if self.mixed_prec == 'bf16' else torch.float16
            for idx_c, cid in enumerate(candidates):
                # Construct tail-key cheaply from python list without building a new tensor
                try:
                    tok = cand_toks[idx_c] if cand_toks else int(cid.item())
                except Exception:
                    tok = None
                if tok is not None and int(self.cache_tail) > 0:
                    tail_list = (base_tail + [tok])[-int(self.cache_tail):]
                    key = tuple(int(x) for x in tail_list)
                else:
                    key = self._key_tail(torch.cat([out_ids, cid], dim=1))
                if key in self._score_cache:
                    ns = self._score_cache[key]
                    scores.append((ns.margin_sum, ns.proxy_score, ns.cost_tokens))
                    continue
                margin_sum = 0.0
                if base_proxy is not None:
                    proxy = float(base_proxy[idx_c])
                else:
                    proxy = float(probs.gather(-1, cid).item())
                cost = 1
                # Accumulate margin from verifier when available
                if verifier_logits is not None:
                    if v_margins is not None:
                        v_margin = float(v_margins[idx_c])
                    else:
                        v1 = torch.softmax(verifier_logits[:, -1, :], dim=-1)
                        v_margin = float(v1.gather(-1, cid).item())
                    # Early prune if extremely low margin
                    if v_margin < self.min_verifier_margin:
                        scores.append((float('-inf'), proxy, cost))
                        # Memoize as dead end
                        self._score_cache[key] = NodeScore(margin_sum=float('-inf'), proxy_score=proxy, cost_tokens=cost)
                        continue
                    margin_sum += v_margin
                else:
                    # Prefer distilled verifier margin over proxy when available
                    if (self.verifier_W is not None) and isinstance(hidden_out, torch.Tensor):
                        try:
                            tok = int(cid.item())
                            h_last = hidden_out[:, -1, :].to(self.verifier_W.device)
                            w_row = self.verifier_W[tok:tok+1, :]
                            b_row = self.verifier_b[tok:tok+1] if isinstance(self.verifier_b, torch.Tensor) else None
                            m = torch.nn.functional.linear(h_last, w_row, b_row).sigmoid().mean().item()
                            margin_sum += float(m)
                        except Exception:
                            margin_sum += proxy
                    else:
                        margin_sum += proxy

                # Short rollout using greedy drafts without committing state
                d = max(1, int(self.depth))
                if d > 1:
                    tmp_ids = torch.cat([out_ids, cid], dim=1)
                    # Reuse KV hint when available
                    hint = get_kv_hint(key)
                    tmp_kv = hint if hint is not None else past_kv
                    steps = 1
                    while steps < d:
                        if use_autocast:
                            with torch.autocast(device_type='cuda', dtype=ac_dtype):  # type: ignore[arg-type]
                                outs2 = model(tmp_ids[:, -1:], past_kv=tmp_kv, use_cache=True)
                        else:
                            outs2 = model(tmp_ids[:, -1:], past_kv=tmp_kv, use_cache=True)
                        if isinstance(outs2, tuple):
                            log2 = outs2[0]
                            tmp_kv = outs2[1]
                            v2 = outs2[3] if len(outs2) > 3 else None
                        else:
                            log2 = outs2
                            v2 = None
                        pr2 = torch.softmax(log2[:, -1, :], dim=-1)
                        nid = torch.argmax(pr2, dim=-1, keepdim=True)
                        if v2 is not None:
                            m2 = float(torch.softmax(v2[:, -1, :], dim=-1).gather(-1, nid).item())
                        else:
                            m2 = float(pr2.gather(-1, nid).item())
                        margin_sum += m2
                        try:
                            register_kv_hint(self._key_tail(tmp_ids), tmp_kv)
                        except Exception:
                            pass
                        tmp_ids = torch.cat([tmp_ids, nid], dim=1)
                        steps += 1
                        cost += 1
                        spent += 1
                        # Token and time budgets
                        if spent >= max(1, int(self.token_budget)):
                            break
                        if int(self.ms_budget) > 0:
                            if (time.perf_counter() - t0) * 1000.0 >= float(self.ms_budget):
                                break

                scores.append((margin_sum, proxy, cost))
                # Memoize shallow score for reuse
                self._score_cache[key] = NodeScore(margin_sum=margin_sum, proxy_score=proxy, cost_tokens=cost)
                # Enforce simple LRU capacity by trimming dict when too large
                if len(self._score_cache) > max(128, int(self.cache_capacity)):
                    # drop ~10% oldest by iter order (Python 3.7+ preserves insertion order)
                    drop_n = max(1, int(0.1 * len(self._score_cache)))
                    for _ in range(drop_n):
                        try:
                            self._score_cache.pop(next(iter(self._score_cache)))
                        except Exception:
                            break

            # Apply diversity penalty against recent tail tokens
            tail = out_ids[0, -min(int(self.cache_tail), int(out_ids.size(1))):].tolist() if out_ids is not None else []
            tail_set = set(tail)
            # Select by blended score with tie-breaker on proxy using precomputed lists
            if scores:
                margins = [s[0] for s in scores]
                proxies = [s[1] for s in scores]
                costs = [s[2] for s in scores]
                blends = [float(self.alpha) * (m / max(1, c)) + float(1.0 - self.alpha) * float(p) for m, p, c in zip(margins, proxies, costs)]
                if float(self.diversity_gamma) != 0.0 and cand_toks:
                    penalties = [float(self.diversity_gamma) * (1.0 if t in tail_set else 0.0) for t in cand_toks]
                else:
                    penalties = [0.0] * len(blends)
                best_i = max(range(len(blends)), key=lambda i: (blends[i] - penalties[i], proxies[i]))
            else:
                best_i = 0
            chosen = candidates[best_i]
            if self.trace:
                try:
                    import logging as _l
                    _l.getLogger("omnicoder.gen").debug("AGoT: chosen=%d norm_margin=%.4f proxy=%.4f", int(chosen.item()), float(scores[best_i][0] / max(1, scores[best_i][2])), float(scores[best_i][1]))
                except Exception:
                    pass
            # Telemetry: write compact JSONL if enabled
            if self.log_path:
                try:
                    rec = {
                        'ent': float(ent), 'width': int(len(candidates)), 'depth': int(self.depth),
                        'token_budget': int(self.token_budget), 'ms_budget': int(self.ms_budget),
                        'alpha': float(self.alpha), 'div_gamma': float(self.diversity_gamma), 'topp': float(self.topp),
                        'chosen': int(chosen.item()),
                    }
                    with open(self.log_path, 'a', encoding='utf-8') as f:
                        f.write(str(rec).replace("'", '"') + "\n")
                except Exception:
                    pass
            return chosen
        except Exception:
            return _greedy(base_logits)


def build_agot() -> AdaptiveGraphOfThoughts:
    return AdaptiveGraphOfThoughts()


