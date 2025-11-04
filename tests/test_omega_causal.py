

def test_scm_abduction_and_do():
    from omnicoder.reasoning.omega_causal import build_minimal_scm_for_query, abductive_score

    scm = build_minimal_scm_for_query("why code compile test")
    vals = scm.forward()
    a0 = float(vals["A"])

    # Perfect match observation should yield score 1.0
    s_ok = abductive_score(scm, {"A": a0})
    assert abs(s_ok - 1.0) < 1e-6

    # Mismatch should penalize to 0 when only one observation
    s_bad = abductive_score(scm, {"A": a0 + 0.123})
    assert s_bad == 0.0

    # do-intervention should increase A when increasing P
    vals_do = scm.do({"P": 1.0})
    a1 = float(vals_do["A"])
    assert a1 >= a0
    assert 0.0 <= a1 <= 1.0


