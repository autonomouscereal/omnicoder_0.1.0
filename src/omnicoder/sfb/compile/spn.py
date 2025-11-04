from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Dict, Tuple
import json
from pathlib import Path
import hashlib


@dataclass
class _SPNNode:
    kind: str  # 'sum' or 'prod' or 'leaf'
    children: List[int]
    weight: float = 1.0
    prob: float = 0.5  # for leaf


@dataclass
class SPNCompiler:
    enabled: bool | None = None
    cache: Dict[str, Any] | None = None
    spn_graphs: Dict[str, List[_SPNNode]] | None = None
    cache_path: str | None = None

    def __post_init__(self) -> None:
        if self.enabled is None:
            self.enabled = os.getenv("SFB_COMPILE_SPN", "1") == "1"
        self.cache = {}
        self.spn_graphs = {}
        # Optionally restore cache from disk
        try:
            self.cache_path = os.getenv('SFB_SPN_CACHE_PATH', '').strip() or None
            if self.cache_path and Path(self.cache_path).exists():
                obj = json.loads(open(self.cache_path, 'r', encoding='utf-8').read())
                c = obj.get('cache') or {}
                self.cache = dict(c)
                graphs = obj.get('spn_graphs') or {}
                self.spn_graphs = {}
                for k, nodes in graphs.items():
                    lst: List[_SPNNode] = []
                    for n in nodes:
                        lst.append(_SPNNode(kind=str(n.get('kind','leaf')), children=list(n.get('children', [])), weight=float(n.get('weight', 1.0)), prob=float(n.get('prob', 0.5))))
                    self.spn_graphs[k] = lst
        except Exception:
            pass

    def _fingerprint(self, factors: List[Any]) -> str:
        try:
            payload = "|".join(sorted([f"{getattr(f,'name','')}:{getattr(f,'meta',{})}" for f in factors]))
        except Exception:
            payload = str(len(factors))
        return hashlib.sha1(payload.encode('utf-8')).hexdigest()

    def maybe_compile(self, factors: List[Any]) -> None:
        if not self.enabled:
            return
        try:
            fp = self._fingerprint(factors)
            if self.cache is None:
                self.cache = {}
            entry = self.cache.get(fp)
            if entry is not None:
                # Touch entry for LRU-style metadata
                entry['hits'] = int(entry.get('hits', 0)) + 1
                entry['last'] = int(os.times().elapsed) if hasattr(os, 'times') else entry.get('last', 0)
                self.cache[fp] = entry
                return
            # Lightweight stand-in: store a summary and mark as seen
            summary = [{
                'name': getattr(f, 'name', ''),
                'type': str(getattr(f, 'meta', {}).get('type', '')),
                'scope': tuple(getattr(f, 'scope', ())),
            } for f in factors]
            self.cache[fp] = {'summary': summary, 'hits': 1, 'last': int(os.times().elapsed) if hasattr(os, 'times') else 0}
            # Attempt to build a typed SPN over leaves derived from factors: product over types, sums over leaves
            try:
                pri_map = {'numeric': 0.65, 'code': 0.60, 'vision': 0.55, 'audio': 0.55, 'video': 0.55, 'semantic': 0.60, 'logic': 0.58}
                nodes: List[_SPNNode] = []
                root = _SPNNode(kind='prod', children=[])
                nodes.append(root)
                root_idx = 0
                type_to_sum_idx: Dict[str, int] = {}
                # Build leaves by inspecting factor metadata when possible
                for f in factors:
                    try:
                        meta = getattr(f, 'meta', {}) or {}
                        ftype = str(meta.get('type', ''))
                        if ftype == 'semantic':
                            mode = str(meta.get('mode', ''))
                            if mode == 'amr':
                                for (src, role, tgt) in (meta.get('triples', []) or [])[:32]:
                                    if str(role).lower() in (':instance', 'instance', 'predicate'):
                                        p = float(pri_map.get('semantic', 0.6))
                                        leaf = _SPNNode(kind='leaf', children=[], prob=p)
                                        nodes.append(leaf)
                                        li = len(nodes) - 1
                                        if 'semantic' not in type_to_sum_idx:
                                            sump = _SPNNode(kind='sum', children=[], weight=1.0)
                                            nodes.append(sump)
                                            type_to_sum_idx['semantic'] = len(nodes) - 1
                                            nodes[root_idx].children.append(type_to_sum_idx['semantic'])
                                        nodes[type_to_sum_idx['semantic']].children.append(li)
                            elif mode == 'srl':
                                roles = meta.get('roles', []) or []
                                for r in roles[:16]:
                                    v = str((r or {}).get('verb', '')).strip().lower()
                                    if v:
                                        p = float(pri_map.get('semantic', 0.6))
                                        leaf = _SPNNode(kind='leaf', children=[], prob=p)
                                        nodes.append(leaf)
                                        li = len(nodes) - 1
                                        if 'semantic' not in type_to_sum_idx:
                                            sump = _SPNNode(kind='sum', children=[], weight=1.0)
                                            nodes.append(sump)
                                            type_to_sum_idx['semantic'] = len(nodes) - 1
                                            nodes[root_idx].children.append(type_to_sum_idx['semantic'])
                                        nodes[type_to_sum_idx['semantic']].children.append(li)
                        else:
                            t = ftype if ftype else 'generic'
                            p = float(pri_map.get(t, 0.5))
                            leaf = _SPNNode(kind='leaf', children=[], prob=p)
                            nodes.append(leaf)
                            li = len(nodes) - 1
                            if t not in type_to_sum_idx:
                                sump = _SPNNode(kind='sum', children=[], weight=1.0)
                                nodes.append(sump)
                                type_to_sum_idx[t] = len(nodes) - 1
                                nodes[root_idx].children.append(type_to_sum_idx[t])
                            nodes[type_to_sum_idx[t]].children.append(li)
                    except Exception:
                        continue
                # Optional structure learning for frequent templates controlled by env
                try:
                    learn = (os.getenv('SFB_SPN_LEARN', '1') == '1')
                except Exception:
                    learn = True
                if learn:
                    # Merge identical leaves under shared sum; lightly reweight observation frequency
                    counts: Dict[str, int] = {}
                    for f in factors:
                        t = str(getattr(f, 'meta', {}).get('type', 'generic'))
                        counts[t] = int(counts.get(t, 0) + 1)
                    # Reweight sums by frequency
                    for k, si in list(type_to_sum_idx.items()):
                        try:
                            nodes[si].weight = float(1.0 + 0.05 * counts.get(k, 0))
                        except Exception:
                            pass
                self.spn_graphs[fp] = nodes
            except Exception:
                pass
        except Exception:
            return

    def register(self, factors: List[Any], messages: List[Dict[str, Any]]) -> None:
        """Store precomputed messages for a factor subgraph once it is observed often.

        Uses an environment threshold SFB_SPN_HIT_THRESHOLD to decide when to keep
        messages for reuse. Before the threshold is met, only a summary is stored.
        """
        if not self.enabled:
            return
        try:
            fp = self._fingerprint(factors)
            if self.cache is None:
                self.cache = {}
            thr = 2
            try:
                thr = max(1, int(os.getenv('SFB_SPN_HIT_THRESHOLD', '2')))
            except Exception:
                thr = 2
            entry = self.cache.get(fp)
            if entry is None:
                # Create a minimal entry; count as one hit
                self.cache[fp] = {
                    'summary': [{
                        'name': getattr(f, 'name', ''),
                        'type': str(getattr(f, 'meta', {}).get('type', '')),
                        'scope': tuple(getattr(f, 'scope', ())),
                    } for f in factors],
                    'hits': 1,
                    'last': int(os.times().elapsed) if hasattr(os, 'times') else 0,
                }
                return
            # Only persist messages when hot enough to be beneficial
            hits = int(entry.get('hits', 0)) + 1
            entry['hits'] = hits
            entry['last'] = int(os.times().elapsed) if hasattr(os, 'times') else entry.get('last', 0)
            if hits >= thr:
                # Keep a compact copy of messages (only supported keys)
                compact: List[Dict[str, Any]] = []
                for m in messages:
                    if not isinstance(m, dict):
                        compact.append({})
                        continue
                    keep: Dict[str, Any] = {}
                    for k in ('prefer_strings', 'avoid_strings', 'token_bias', 'score'):
                        if k in m:
                            keep[k] = m[k]
                    compact.append(keep)
                entry['messages'] = compact
            self.cache[fp] = entry
            # Persist to disk if configured
            try:
                if self.cache_path:
                    obj = {
                        'cache': self.cache,
                        'spn_graphs': {
                            k: [
                                {
                                    'kind': n.kind,
                                    'children': list(n.children),
                                    'weight': float(n.weight),
                                    'prob': float(n.prob),
                                } for n in (self.spn_graphs.get(k, []) if self.spn_graphs else [])
                            ] for k in (self.spn_graphs.keys() if self.spn_graphs else [])
                        },
                    }
                    Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(self.cache_path, 'w', encoding='utf-8') as f:
                        json.dump(obj, f)
            except Exception:
                pass
        except Exception:
            return

    def lookup(self, factors: List[Any]) -> List[Dict[str, Any]] | None:
        """Return cached messages for a given factor set if compiled and hot."""
        if not self.enabled:
            return None
        try:
            fp = self._fingerprint(factors)
            entry = None if self.cache is None else self.cache.get(fp)
            if not entry:
                return None
            msgs = entry.get('messages')
            if isinstance(msgs, list):
                # Touch entry for LRU signal
                entry['hits'] = int(entry.get('hits', 0)) + 1
                entry['last'] = int(os.times().elapsed) if hasattr(os, 'times') else entry.get('last', 0)
                self.cache[fp] = entry
                return msgs  # type: ignore[return-value]
            return None
        except Exception:
            return None

    # Minimal probabilistic circuit (PC) evaluator for unary-only caches
    def pc_score(self, factors: List[Any]) -> float:
        """Return a tractable score using the compiled mini-SPN if available.

        Interprets the value as average leaf probability after flowing upward
        through sum/product structure with uniform sum weights.
        """
        if not self.enabled:
            return 0.0
        try:
            fp = self._fingerprint(factors)
            entry = None if self.cache is None else self.cache.get(fp)
            if not entry:
                return 0.0
            nodes = None if self.spn_graphs is None else self.spn_graphs.get(fp)
            if not nodes:
                return 0.0
            # Prune cache by LRU if necessary
            try:
                max_entries = int(os.getenv('SFB_SPN_CACHE_MAX', '256'))
            except Exception:
                max_entries = 256
            if self.cache is not None and len(self.cache) > max_entries:
                # Evict the oldest
                items = sorted(self.cache.items(), key=lambda kv: int(kv[1].get('last', 0)))
                for k, _ in items[: max(0, len(self.cache) - max_entries)]:
                    try:
                        del self.cache[k]
                        if self.spn_graphs and k in self.spn_graphs:
                            del self.spn_graphs[k]
                    except Exception:
                        continue
            # Calibrate leaves by recent messages when available: small uplift by avg score per type
            try:
                entry_msgs = entry.get('messages', []) or []
                type_score: Dict[str, float] = {}
                type_count: Dict[str, int] = {}
                for m in entry_msgs:
                    t = None
                    if 'token_bias' in m:
                        t = 'code'
                    elif m.get('prefer_strings'):
                        t = 'semantic'
                    s = float(m.get('score', 0.0))
                    if t:
                        type_score[t] = float(type_score.get(t, 0.0) + s)
                        type_count[t] = int(type_count.get(t, 0) + 1)
                avg_score = {k: (type_score[k] / max(1, type_count.get(k, 1))) for k in type_score.keys()}
                nodes2: List[_SPNNode] = []
                for n in nodes:
                    if n.kind == 'leaf':
                        uplift = float(min(0.1, max(0.0, max(avg_score.get('semantic', 0.0), avg_score.get('code', 0.0)))))
                        nodes2.append(_SPNNode(kind=n.kind, children=list(n.children), weight=float(n.weight), prob=float(min(0.99, max(0.01, n.prob + uplift)))))
                    else:
                        nodes2.append(n)
                nodes = nodes2
            except Exception:
                pass
            # Evaluate SPN upward (post-order). For simplicity, assume node 0 is root.
            memo: Dict[int, float] = {}
            def eval_node(i: int) -> float:
                if i in memo:
                    return memo[i]
                n = nodes[i]
                if n.kind == 'leaf':
                    memo[i] = float(max(0.0, min(1.0, n.prob)))
                elif n.kind == 'prod':
                    prod = 1.0
                    if not n.children:
                        prod = 1.0
                    else:
                        for c in n.children:
                            prod *= max(1e-8, eval_node(c))
                    memo[i] = prod
                elif n.kind == 'sum':
                    if not n.children:
                        memo[i] = 0.5
                    else:
                        s = 0.0
                        for c in n.children:
                            s += eval_node(c) * float(max(1e-6, n.weight))
                        memo[i] = float(s / float(len(n.children) * max(1e-6, n.weight)))
                else:
                    memo[i] = 0.0
                return memo[i]
            val = eval_node(0)
            return float(val)
        except Exception:
            return 0.0

