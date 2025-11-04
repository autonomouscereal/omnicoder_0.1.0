from __future__ import annotations

import os

from omnicoder.sfb.arbiter import ProofMarginArbiter, ProofMarginInputs


def test_proof_margin_with_threshold_env():
    os.environ['SFB_PROOF_MARGIN'] = '0.9'
    arb = ProofMarginArbiter()
    x = ProofMarginInputs(
        llm_confidence=0.2,
        sum_log_factors=0.1,
        verifier_score=0.2,
        retrieval_hits=0,
        code_passk=0.2,
        clip_z=0.1,
        audio_z=0.0,
        video_z=0.0,
    )
    m = arb.compute_margin(x)
    # With a high threshold, accept is likely false
    assert (m >= 0.9) == arb.accept(x)

