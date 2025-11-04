

def test_compute_margin_text_and_code():
    from omnicoder.reasoning.omega_verifier import compute_margin

    # Text-only should map to exact overlap-based margin here (1.0)
    m_text = compute_margin({"text": {"hyp": "hello world", "context": "hello"}})
    assert 0.99 <= m_text <= 1.0

    # With code signal (heuristic 0.7), unified avg should be ~0.85
    m_both = compute_margin({
        "text": {"hyp": "hello world", "context": "hello"},
        "code": {"log_str": "All tests PASS"},
    })
    assert abs(m_both - 0.85) < 0.11  # allow small tolerance


