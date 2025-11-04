from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Any
from .factor_graph import build_text_semantic_graph, run_sum_product, semantic_log_marginal_score


@dataclass
class Message:
    target: str
    values: Dict[str, float]


@dataclass
class BeliefPropagation:
    iterations: int | None = None

    def __post_init__(self) -> None:
        if self.iterations is None:
            try:
                self.iterations = int(os.getenv("SFB_BP_ITERS", "10"))
            except Exception:
                self.iterations = 10

    def run(self, factors: List[Any]) -> List[Dict[str, Any]]:
        """
        Execute a very lightweight message pass over factors and return a list of
        message dicts understood by CrossBiasFusion. Each message may contain:
          - prefer_strings: List[str]
          - avoid_strings: List[str]
          - token_bias: Dict[int, float]
        """
        messages: List[Dict[str, Any]] = []
        try:
            # Lazy import solvers to avoid heavy deps at import time
            from omnicoder.sfb.factors.numeric import NumericSolver
            from omnicoder.sfb.factors.code import CodeSolver
            # Optional lightweight logic solver
            LogicSolver = None
            try:
                from omnicoder.sfb.factors.logic import LogicSolver as _LogicSolver  # type: ignore
                LogicSolver = _LogicSolver
            except Exception:
                LogicSolver = None
            try:
                from omnicoder.sfb.factors.vision import VisionSolver  # type: ignore
            except Exception:
                VisionSolver = None  # type: ignore
            try:
                from omnicoder.sfb.factors.audio import AudioSolver  # type: ignore
            except Exception:
                AudioSolver = None  # type: ignore
            try:
                from omnicoder.sfb.factors.video import VideoSolver  # type: ignore
            except Exception:
                VideoSolver = None  # type: ignore

            for f in factors:
                ftype = str(getattr(f, 'meta', {}).get('type', 'generic'))
                if ftype == 'numeric':
                    msg = NumericSolver().solve(f)
                    m = dict(msg.aux)
                    m['score'] = float(msg.score)
                    messages.append(m)
                elif ftype == 'code':
                    msg = CodeSolver().solve(f)
                    m = dict(msg.aux)
                    m['score'] = float(msg.score)
                    messages.append(m)
                elif ftype == 'logic' and LogicSolver is not None:
                    try:
                        msg = LogicSolver().solve(f)  # type: ignore
                        m = dict(msg.aux)
                        m['score'] = float(msg.score)
                        messages.append(m)
                    except Exception:
                        messages.append({})
                elif ftype == 'vision' and VisionSolver is not None:
                    msg = VisionSolver().solve(f)  # type: ignore
                    m = dict(msg.aux)
                    m['score'] = float(msg.score)
                    messages.append(m)
                elif ftype == 'audio' and AudioSolver is not None:
                    msg = AudioSolver().solve(f)  # type: ignore
                    m = dict(msg.aux)
                    m['score'] = float(msg.score)
                    messages.append(m)
                elif ftype == 'video' and VideoSolver is not None:
                    msg = VideoSolver().solve(f)  # type: ignore
                    m = dict(msg.aux)
                    m['score'] = float(msg.score)
                    messages.append(m)
                else:
                    messages.append({})
            # After per-factor solvers, run a tiny semantic graph pass for AMR/SRL
            try:
                iters = 3
                try:
                    iters = max(1, int(os.getenv('SFB_BP_ITERS_SEM', '3')))
                except Exception:
                    iters = 3
                g = build_text_semantic_graph(factors)
                marg = run_sum_product(g, iterations=iters)
                sem_bel = float(semantic_log_marginal_score(marg))
                # Convert highest probability tokens into light prefer_strings hints
                prefer: List[str] = []
                try:
                    # pick up to 5 tokens with largest p(x=1)
                    scores: List[tuple[str, float]] = []
                    for vn, dist in marg.items():
                        p1 = float(dist.get(1, 0.0))
                        if vn.startswith('pred:') and p1 > 0.5:
                            tok = vn.split(':', 1)[-1]
                            scores.append((tok, p1))
                    scores.sort(key=lambda x: x[1], reverse=True)
                    for tok, _ in scores[:5]:
                        prefer.append(tok)
                except Exception:
                    prefer = []
                if prefer or sem_bel != 0.0:
                    messages.append({'prefer_strings': prefer, 'score': float(sem_bel)})
            except Exception:
                pass
        except Exception:
            # Fallback: neutral messages
            for _ in factors:
                messages.append({})
        return messages


