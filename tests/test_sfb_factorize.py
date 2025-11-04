from __future__ import annotations

import os

from omnicoder.sfb.factorize import factorize_prompt


def test_factorize_basic_numeric_and_code():
    os.environ['SFB_FACTORIZER'] = 'amr,srl'
    prompt = "Please write python code to sum 2+2 and print the result."
    res = factorize_prompt(prompt)
    names = {f.name for f in res.factors}
    # Heuristics should surface both numeric and code factors
    assert 'numeric_reasoning' in names
    assert 'code_generation' in names
    # Goal priors should pick up code intent via heuristic
    assert res.goal_priors.get('code', 0.0) >= 0.5


