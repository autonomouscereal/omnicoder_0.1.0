from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import os

from . import FactorScore
from ..runtime.sandbox import make_code_sandbox


@dataclass
class CodeSolver:
    def solve(self, factor: Any) -> FactorScore:
        """Run PAL-style compile+unit tests if inputs are provided via env sidecar.

        Environment:
          SFB_CODE_TASKS_JSONL: path to JSONL rows with {candidates:[], tests:"..."}
        If unset, falls back to light token hints.
        """
        tasks = os.getenv('SFB_CODE_TASKS_JSONL', '').strip()
        if not tasks:
            aux: Dict[str, Any] = {"prefer_strings": ["def", "class", "import", "return"], "avoid_strings": []}
            return FactorScore(name="code", score=0.1, aux=aux)
        try:
            import json
            from omnicoder.eval.code_eval import pass_at_k as _pass_at_k  # type: ignore
            rows = [json.loads(l) for l in open(tasks, 'r', encoding='utf-8', errors='ignore') if l.strip()]
            # Evaluate a small subset for latency
            subs = rows[:5]
            ok = 0
            for ex in subs:
                # Optional sandbox execution if configured
                sb_score = 0.0
                try:
                    if os.getenv('SFB_CODE_SANDBOX', '1') == '1':
                        sandbox = make_code_sandbox()
                        # Try the first candidate in a safe runner
                        cand = (ex.get('candidates', []) or [''])[0]
                        tests = str(ex.get('tests', ''))
                        # Multi-file support: provide an optional 'files': [{name, content}]
                        files: List[Dict[str, str]] = ex.get('files', []) or []  # type: ignore
                        if files:
                            # Merge files into code by simple concatenation with guards
                            pre = "\n\n".join([f['content'] for f in files if 'content' in f])
                            cand = pre + "\n\n" + cand
                        res = sandbox.run(cand, tests, timeout=5)
                        sb_score = 1.0 if res.ok else 0.0
                except Exception:
                    sb_score = 0.0
                # Fallback to built-in evaluator for pass@k if sandbox not conclusive
                if sb_score >= 1.0 or _pass_at_k(ex.get('candidates', []), ex.get('tests', ''), k=5, timeout=3):
                    ok += 1
            score = float(ok / max(1, len(subs)))
            aux: Dict[str, Any] = {"prefer_strings": ["def", "class", "import", "return"], "avoid_strings": [], "token_bias": {}, "score": score}
            return FactorScore(name="code", score=score, aux=aux)
        except Exception:
            aux = {"prefer_strings": ["def", "class", "import", "return"], "avoid_strings": []}
            return FactorScore(name="code", score=0.1, aux=aux)


