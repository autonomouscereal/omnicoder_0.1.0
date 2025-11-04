import os


def test_sfb_factorize_srl_heuristic():
    os.environ['SFB_ENABLE'] = '1'
    os.environ['SFB_FACTORIZER'] = 'srl'
    from omnicoder.sfb import factorize_prompt
    fact = factorize_prompt("We sort 3 numbers and return the median.")
    assert any(f.meta.get('type') == 'semantic' for f in fact.factors)


def test_sfb_bp_messages_shape():
    os.environ['SFB_ENABLE'] = '1'
    from omnicoder.sfb import BeliefPropagation, factorize_prompt
    fact = factorize_prompt("Write python code to sum two numbers.")
    bp = BeliefPropagation()
    msgs = bp.run(fact.factors)
    assert isinstance(msgs, list)


def test_sfb_spn_cache_roundtrip(tmp_path):
    os.environ['SFB_ENABLE'] = '1'
    os.environ['SFB_SPN_CACHE_PATH'] = str(tmp_path / 'spn_cache.json')
    from omnicoder.sfb import SPNCompiler, factorize_prompt
    spn = SPNCompiler()
    fact = factorize_prompt("If A then B. Also A.")
    spn.maybe_compile(fact.factors)
    msgs = [{'prefer_strings': ['therefore'], 'score': 0.5}]
    spn.register(fact.factors, msgs)
    hit = spn.lookup(fact.factors)
    assert isinstance(hit, list)


def test_sfb_fusion_bias_noop_without_alpha():
    import torch
    from omnicoder.sfb import CrossBiasFusion
    fuser = CrossBiasFusion()
    fuser.alpha = 0.0
    logits = torch.zeros((1, 10))
    out = fuser.apply_messages(logits, [{'prefer_strings': ['test']}])
    assert torch.allclose(out, logits)


def test_sfb_arbiter_margin_monotonic():
    from omnicoder.sfb import ProofMarginArbiter, ProofMarginInputs
    arb = ProofMarginArbiter()
    x = ProofMarginInputs(llm_confidence=0.5, sum_log_factors=0.1, verifier_score=0.2, retrieval_hits=0)
    m1 = arb.compute_margin(x)
    x2 = ProofMarginInputs(llm_confidence=0.6, sum_log_factors=0.2, verifier_score=0.3, retrieval_hits=0)
    m2 = arb.compute_margin(x2)
    assert m2 >= m1


