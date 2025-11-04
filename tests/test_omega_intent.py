import os


def test_infer_goals_basic():
    # Ensure omega methods default are enabled
    os.environ.setdefault("OMNI_GOAL_INFER", "rsa,pr,cirl")
    from omnicoder.reasoning.omega_intent import infer_goals

    utterance = "Please explain how to plan and verify code with tests"
    gb = infer_goals(utterance, context=None, k=3)

    assert hasattr(gb, "hypotheses") and isinstance(gb.hypotheses, list)
    assert len(gb.hypotheses) >= 1
    # Posterior values should be in (0,1]
    for h in gb.hypotheses:
        assert 0.0 < float(getattr(h, "posterior", 0.0)) <= 1.0
        assert isinstance(getattr(h, "goal", ""), str)


