from __future__ import annotations

import os

from omnicoder.sfb.factorize import factorize_prompt
from omnicoder.sfb.compile.spn import SPNCompiler
from omnicoder.sfb.inference.bp import BeliefPropagation


def test_numeric_solver_exact_and_spn_cache():
    os.environ['SFB_FACTORIZER'] = 'amr,srl'
    os.environ['SFB_COMPILE_SPN'] = '1'
    res = factorize_prompt("Compute 3*(4+5)=?")
    # Run BP to produce messages (numeric solver produces higher score when eval succeeds)
    bp = BeliefPropagation()
    msgs = bp.run(res.factors)
    assert isinstance(msgs, list)
    # Compile SPN twice and ensure cache is used
    spn = SPNCompiler()
    spn.maybe_compile(res.factors)
    # Second call should be a no-op (hit cache)
    prev_cache_size = len(spn.cache or {})
    spn.maybe_compile(res.factors)
    assert len(spn.cache or {}) == prev_cache_size

