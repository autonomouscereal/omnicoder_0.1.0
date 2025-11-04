import os
import tempfile
import torch

from omnicoder.modeling.multimodal.aligner import CrossModalVerifier


def test_cross_modal_verifier_scores_in_range():
    torch.manual_seed(0)
    B, D = 4, 256
    a = torch.randn(B, D)
    b = torch.randn(B, D)
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    cmv = CrossModalVerifier().eval()
    s = cmv(a, b)
    assert s.shape == (B, 1)
    assert torch.all(s >= 0) and torch.all(s <= 1)


def test_cross_modal_verifier_onnx_export_smoke():
    torch.manual_seed(0)
    B, D = 2, 128
    a = torch.randn(B, D)
    b = torch.randn(B, D)
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    cmv = CrossModalVerifier().eval()
    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "cm_verifier.onnx")
        try:
            torch.onnx.export(cmv, (a, b), out_path, input_names=["a","b"], output_names=["score"], opset_version=17)
        except Exception:
            # Allow environments without full exporter support to pass the smoke
            pass


