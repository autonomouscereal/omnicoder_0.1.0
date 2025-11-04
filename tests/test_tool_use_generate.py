from omnicoder.inference.generate import _postprocess_tool_use


def test_tool_use_postprocess_replaces_tags():
    text = "Compute: <tool:calculator {\"expr\": \"2+3\"}> end."
    out = _postprocess_tool_use(text)
    assert out != text
    assert "result" in out and "5" in out


