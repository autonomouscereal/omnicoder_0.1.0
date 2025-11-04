from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.utils.logger import get_logger


@dataclass
class ToTConfig:
    max_depth: int = 8               # maximum thought depth
    branch_factor: int = 4           # K candidates per expansion
    simulations: int = 128           # number of MCTS simulations
    c_puct: float = 1.4              # exploration constant
    temperature: float = 0.8         # sampling temp for proposals
    top_k: int = 40
    top_p: float = 0.9
    verify_threshold: float = 0.0    # verifier acceptance threshold
    use_value_head: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class _TreeNode:
    def __init__(self, parent: Optional["_TreeNode"], token_id: Optional[int]) -> None:
        self.parent = parent
        self.token_id = token_id
        self.children: List["_TreeNode"] = []
        self.N = 0.0    # visits
        self.W = 0.0    # total value
        self.P = 0.0    # prior

    @property
    def Q(self) -> float:
        return self.W / max(1.0, self.N)

    def add_child(self, child: "_TreeNode") -> None:
        self.children.append(child)


class ToTMCTS:
    """
    Tree-of-Thoughts + Monte Carlo Tree Search driver using the same model as
    both policy (proposals) and value (via value_head/verifier).

    This module does not alter the core generation API; it provides a utility
    that can be used by controllers or evaluation scripts to perform deliberate
    search for hard problems. Designed to avoid env reads and heavy IO.
    """

    def __init__(self, model: OmniTransformer, tokenizer: Any, cfg: Optional[ToTConfig] = None) -> None:
        self.model = model
        self.tok = tokenizer
        self.cfg = cfg or ToTConfig()
        self._log = get_logger("omnicoder.tot")
        self.model.eval().to(self.cfg.device)

    def _propose(self, ids: torch.Tensor, k: int) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """
        Propose K candidate next tokens using model logits sampling, and compute
        auxiliary value/verifier signals for evaluation.
        """
        with torch.inference_mode():
            _T = int(ids.shape[1])
            _W = 1024
            _start = _T - _W if _T > _W else 0
            out = self.model(ids[:, _start:], use_cache=True, return_hidden=True)
            if isinstance(out, tuple):
                logits = out[0]
                hidden = out[-1]
            else:
                logits = out
                hidden = None
            last = logits[:, -1, :]
            # Top-k sampling with temperature
            temp = max(1e-5, float(self.cfg.temperature))
            scores = last / temp
            _V = int(scores.shape[-1])
            _kk = int(k) if int(k) <= _V else _V
            _tk = torch.ops.aten.topk.default(scores, int(_kk), -1, True, True)
            topv, topi = _tk[0], _tk[1]
            probs = torch.softmax(topv, dim=-1)
            # Sample without replacement
            _kl = int(topv.shape[-1])
            _ns = int(k) if int(k) <= _kl else _kl
            sel = torch.multinomial(probs, num_samples=int(_ns), replacement=False)
            cand = topi.gather(-1, sel).squeeze(0).tolist()
            # Value via value_head (if present)
            if self.cfg.use_value_head and hasattr(self.model, 'value_head') and getattr(self.model, 'value_head') is not None and hidden is not None:
                vh = self.model.value_head(hidden)  # (B,T,1)
                v_last = vh[:, -1, 0]  # (B,)
            else:
                v_last = torch.zeros((1,), dtype=torch.float32, device=ids.device)
            # Verifier prob for chosen candidates (approximate via model.verifier_head if present)
            if hasattr(self.model, 'verifier_head') and getattr(self.model, 'verifier_head') is not None and hidden is not None:
                ve = torch.softmax(self.model.verifier_head(hidden)[:, -1, :], dim=-1)
                v_cand = ve[0, cand]
            else:
                v_cand = torch.zeros((len(cand),), dtype=torch.float32, device=ids.device)
            return cand, v_last.detach(), v_cand.detach()

    def _evaluate_leaf(self, ids: torch.Tensor) -> float:
        """
        Leaf evaluation: combine value head (state value) and optional verifier
        acceptance on greedy continuation.
        """
        with torch.inference_mode():
            _T = int(ids.shape[1])
            _W = 1024
            _start = _T - _W if _T > _W else 0
            out = self.model(ids[:, _start:], use_cache=True, return_hidden=True)
            if isinstance(out, tuple):
                logits = out[0]
                hidden = out[-1]
            else:
                logits = out
                hidden = None
            v = 0.0
            if self.cfg.use_value_head and hasattr(self.model, 'value_head') and getattr(self.model, 'value_head') is not None and hidden is not None:
                vh = self.model.value_head(hidden)
                v = float(torch.tanh(vh[:, -1, 0]).item())
            # Optional verifier bonus on greedy token
            if hasattr(self.model, 'verifier_head') and getattr(self.model, 'verifier_head') is not None and hidden is not None:
                ve = torch.softmax(self.model.verifier_head(hidden)[:, -1, :], dim=-1)
                g = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                prob = float(ve[0, g].item())
                if prob >= float(self.cfg.verify_threshold):
                    v = v + 0.1 * prob
            v = -1.0 if v < -1.0 else v
            v = 1.0 if v > 1.0 else v
            return v

    def search(self, prompt: str) -> Tuple[List[int], dict]:
        """
        Run ToT+MCTS to generate a token sequence. Returns best token ids and stats.
        """
        stats: dict = {"sims": 0, "expanded": 0}
        root = _TreeNode(parent=None, token_id=None)
        ids0 = torch.tensor([self.tok.encode(prompt)], dtype=torch.long, device=self.cfg.device)
        priors, v_root, v_cand = self._expand(root, ids0)
        stats["expanded"] += 1
        for _ in range(max(1, int(self.cfg.simulations))):
            self._simulate(root, ids0)
            stats["sims"] += 1
        # Extract best path by greedily following highest visit count
        best: List[int] = []
        node = root
        depth = 0
        while node.children and depth < int(self.cfg.max_depth):
            node = max(node.children, key=lambda c: c.N)
            if node.token_id is not None:
                best.append(int(node.token_id))
            depth += 1
        return best, stats

    def _expand(self, node: _TreeNode, ids: torch.Tensor) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        cand, v_last, v_cand = self._propose(ids, self.cfg.branch_factor)
        # Simple prior from softmax over verifier probabilities or uniform fallback
        if v_cand.numel() > 0:
            P = torch.softmax(v_cand, dim=-1).detach().cpu().tolist()
        else:
            P = [1.0 / max(1, len(cand))] * len(cand)
        for t, p in zip(cand, P):
            ch = _TreeNode(parent=node, token_id=int(t))
            ch.P = float(p)
            node.add_child(ch)
        return cand, v_last, v_cand

    def _simulate(self, root: _TreeNode, ids0: torch.Tensor) -> None:
        path: List[_TreeNode] = []
        node = root
        ids = ids0
        depth = 0
        # Selection
        while node.children and depth < int(self.cfg.max_depth):
            totalN = sum(ch.N for ch in node.children) + 1.0
            c = float(self.cfg.c_puct)
            # UCB
            def score(ch: _TreeNode) -> float:
                u = c * ch.P * math.sqrt(totalN) / (1.0 + ch.N)
                return ch.Q + u
            node = max(node.children, key=score)
            if node.token_id is not None:
                nxt = torch.tensor([[node.token_id]], dtype=torch.long, device=ids.device)
                ids = torch.cat([ids, nxt], dim=1)
            path.append(node)
            depth += 1
        # Expansion
        if depth < int(self.cfg.max_depth):
            _, _, _ = self._expand(node, ids)
        # Evaluation
        v = self._evaluate_leaf(ids)
        # Backup
        for n in path:
            n.N += 1.0
            n.W += float(v)


