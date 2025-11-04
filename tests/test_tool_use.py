from omnicoder.inference.tool_use import build_default_registry


def test_tool_use_basic_calculator():
    reg = build_default_registry()
    text = "Compute: <tool:calculator {\"expr\": \"2+2*3\"}>"
    out = reg.parse_and_invoke_all(text)
    key = list(out.keys())[0]
    assert "result" in out[key] and abs(out[key]["result"] - 8.0) < 1e-6


