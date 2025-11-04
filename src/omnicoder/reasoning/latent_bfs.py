"""
Latent-space BFS (Continuous Thought)

Provides a token-decoder-free lookahead heuristic by operating on hidden states
returned by the model. This module scores candidate continuations in latent
space, avoiding repeated LM head projection until commit.

When OMNICODER_LATENT_BFS_ENABLE=1, callers can request a small candidate set
and score them via cosine similarity in the last hidden state space.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import os
import torch
import torch.nn.functional as F
try:
    from .kv_hints import get_kv_hint, register_kv_hint  # type: ignore
except Exception:
    def get_kv_hint(key):
        return None
    def register_kv_hint(key, kv):
        return None


class LatentBFS:
    def __init__(self) -> None:
        try:
            self.enabled = os.getenv("OMNICODER_LATENT_BFS_ENABLE", "1") == "1"
        except Exception:
            self.enabled = False
        self.width = int(os.getenv("OMNICODER_LATENT_BFS_WIDTH", "3"))
        self.depth = int(os.getenv("OMNICODER_LATENT_BFS_DEPTH", "2"))
        # Temperature to soften cosine scores into weights
        try:
            self.temp = float(os.getenv("OMNICODER_LATENT_BFS_TEMP", "0.5"))
        except Exception:
            self.temp = 0.5
        try:
            self.beam = int(os.getenv("OMNICODER_LATENT_BFS_BEAM", str(self.width)))
        except Exception:
            self.beam = self.width
        # Mixed precision dtype for rollouts on CUDA: bf16 | fp16 | off
        self.mixed_prec = os.getenv("OMNICODER_REASONING_MIXED_PREC", "bf16").strip().lower()
        try:
            self.trace = os.getenv("OMNICODER_TRACE_ENABLE", "1") == "1"
        except Exception:
            self.trace = True

    @torch.inference_mode()
    def score_candidates(
        self,
        model: torch.nn.Module,
        out_ids: torch.Tensor,
        past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        hidden_out: Optional[torch.Tensor],
        candidates: List[torch.Tensor],
    ) -> List[float]:
        """
        Returns latent-space scores for candidate next-token ids using short
        greedy rollouts in hidden space. Higher is better.
        """
        if not self.enabled or hidden_out is None or not isinstance(hidden_out, torch.Tensor):
            # Fallback uniform scores
            return [1.0 for _ in candidates]

        try:
            h_last = hidden_out[:, -1, :]  # (1, C)
            scores: List[float] = []
            # Beam search in latent space per candidate root
            for cid in candidates:
                tmp_ids = torch.cat([out_ids, cid], dim=1)
                tmp_kv = past_kv
                # Each beam item: (score, ids, kv)
                beam: List[Tuple[float, torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]] = [(0.0, tmp_ids, tmp_kv)]
                dmax = max(1, int(self.depth))
                for _ in range(dmax):
                    new_beam: List[Tuple[float, torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]] = []
                    for (acc, ids_b, kv_b) in beam:
                        use_autocast = (self.mixed_prec in ("bf16", "fp16")) and (ids_b.device.type == 'cuda')
                        ac_dtype = torch.bfloat16 if self.mixed_prec == 'bf16' else torch.float16
                        if use_autocast:
                            with torch.autocast(device_type='cuda', dtype=ac_dtype):  # type: ignore[arg-type]
                                outs2 = model(ids_b[:, -1:], past_kv=kv_b, use_cache=True, return_hidden=True)
                        else:
                            outs2 = model(ids_b[:, -1:], past_kv=kv_b, use_cache=True, return_hidden=True)
                        if isinstance(outs2, tuple):
                            log2 = outs2[0]
                            kv2 = outs2[1]
                            hid2 = outs2[-1]
                        else:
                            log2 = outs2
                            kv2 = kv_b
                            hid2 = None
                        # Compute latent similarity gain
                        gain = 0.0
                        if hid2 is not None and isinstance(hid2, torch.Tensor):
                            h2 = hid2[:, -1, :]
                            gain = float(F.cosine_similarity(h_last, h2, dim=-1).mean().item())
                        # Expand top-k next tokens
                        pr2 = torch.softmax(log2[:, -1, :], dim=-1)
                        kexp = max(1, int(self.width))
                        _, nxt = torch.topk(pr2, k=kexp, dim=-1)
                        for i in range(kexp):
                            nid = nxt[:, i : i + 1]
                            ids2 = torch.cat([ids_b, nid], dim=1)
                            try:
                                register_kv_hint(tuple(ids2[0, -min(8, ids2.size(1)):].tolist()), kv2)
                            except Exception:
                                pass
                            new_beam.append((acc + gain, ids2, kv2))
                    # prune
                    if new_beam:
                        new_beam.sort(key=lambda x: float(x[0]), reverse=True)
                        beam = new_beam[: max(1, int(self.beam))]
                        if self.trace:
                            try:
                                import logging as _l
                                _l.getLogger("omnicoder.gen").debug("LatentBFS: depth_step beam=%d best=%.4f", int(len(beam)), float(beam[0][0]))
                            except Exception:
                                pass
                    else:
                        break
                # Use best beam score as candidate score
                best = max(beam, key=lambda x: float(x[0])) if beam else (0.0, tmp_ids, tmp_kv)
                scores.append(float(best[0]))
            return scores
        except Exception:
            return [1.0 for _ in candidates]


def build_latent_bfs() -> LatentBFS:
    return LatentBFS()


