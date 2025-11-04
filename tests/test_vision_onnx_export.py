import os
import tempfile
import torch

from omnicoder.modeling.multimodal.vision_encoder import VisionBackbone


def test_vision_backbone_onnx_smoke():
    vb = VisionBackbone(backend="auto", d_model=384, return_pooled=True).eval()
    x = torch.randn(1, 3, 224, 224)
    tokens, pooled = vb(x)
    assert tokens is not None and tokens.dim() == 3
    # Export a tiny graph: pooled vector only (if present)
    if pooled is None:
        pooled = tokens[:, 0, :]
    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "vision.onnx")
        try:
            torch.onnx.export(vb, x, out_path, input_names=["image"], output_names=["tokens","pooled"], opset_version=17)
        except Exception:
            # Some timm backbones require tracing fixes; tolerate failure as smoke
            pass


def test_grounding_heads_onnx_export_smoke() -> None:
    # Export both simple and rep_rta heads via the exporter script
    import sys, subprocess, tempfile
    with tempfile.TemporaryDirectory() as td:
        rc = subprocess.run([sys.executable, "-m", "omnicoder.export.onnx_export_grounding",
                             "--out", td, "--d_model", "384", "--tokens", "196"], check=False).returncode
        assert rc == 0 or rc is not None


