from __future__ import annotations

from omnicoder.sfb.arbiter import ProofMarginArbiter, ProofMarginInputs


def test_proof_margin_math_auto():
    arb = ProofMarginArbiter()
    x1 = ProofMarginInputs(
        llm_confidence=0.3,
        sum_log_factors=0.1,
        verifier_score=0.2,
        retrieval_hits=0,
        code_passk=0.0,
        clip_z=0.0,
        audio_z=0.0,
        video_z=0.0,
    )
    m1 = arb.compute_margin(x1)
    x2 = ProofMarginInputs(
        llm_confidence=0.5,
        sum_log_factors=0.2,
        verifier_score=0.2,
        retrieval_hits=0,
        code_passk=0.0,
        clip_z=0.0,
        audio_z=0.0,
        video_z=0.0,
    )
    m2 = arb.compute_margin(x2)
    assert m2 >= m1
    assert arb.accept(x2) is True

