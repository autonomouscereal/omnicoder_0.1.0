from __future__ import annotations

"""
Programmatic reward utilities used by RL loops.

Includes:
- Code test execution reward (fraction of asserts passing)
- WER-based reward for ASR text (reward = 1 - WER)
- MOS proxy for text responses (simple heuristic)
"""

from pathlib import Path
from typing import Optional


def code_tests_reward(candidate_py: str, tests_code: str, timeout_s: int = 3) -> float:
    import subprocess, sys, tempfile
    from pathlib import Path as _Path
    with tempfile.TemporaryDirectory() as td:
        tmp = _Path(td)
        (tmp / "sol.py").write_text(candidate_py, encoding="utf-8")
        wrapper = f"""
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
        (tmp / "tests_runner.py").write_text(wrapper, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, str(tmp / "tests_runner.py")], cwd=str(tmp), capture_output=True, text=True, timeout=timeout_s)
            out = proc.stdout.strip().split()
            if len(out) >= 2:
                p, t = int(out[0]), int(out[1])
                return float(p) / float(max(1, t))
            return 0.0
        except Exception:
            return 0.0


def wer_from_texts(reference: str, hypothesis: str) -> Optional[float]:
    try:
        from jiwer import wer  # type: ignore
        return float(max(0.0, 1.0 - float(wer(reference, hypothesis))))
    except Exception:
        try:
            import difflib
            return float(difflib.SequenceMatcher(None, reference, hypothesis).ratio())
        except Exception:
            return None


def mos_proxy_from_text(text: str) -> float:
    txt = text.strip()
    n = len(txt)
    if n == 0:
        return 0.0
    alpha = sum(ch.isalpha() for ch in txt) / float(n)
    len_score = 1.0 - abs(n - 128) / 128.0
    len_score = max(0.0, min(1.0, len_score))
    return float(max(0.0, min(1.0, 0.5 * alpha + 0.5 * len_score)))


