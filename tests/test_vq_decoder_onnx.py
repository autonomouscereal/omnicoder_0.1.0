from __future__ import annotations

from pathlib import Path


def test_image_vq_decoder_onnx_io(tmp_path: Path) -> None:
    # Soft skip if onnxruntime or torch not present
    try:
        import torch  # type: ignore
        import onnxruntime as ort  # type: ignore
    except Exception:
        return

    # Create a tiny fake codebook for a 4x4 grid (hq=wq=4), D=8
    K, D = 16, 8
    patch = 16
    blob = {
        "codebook": torch.randn(K, D),
        "emb_dim": D,
        "patch": patch,
    }
    codebook_path = tmp_path / "cb.pt"
    torch.save(blob, codebook_path)

    # Export ONNX
    onnx_path = tmp_path / "vqdec.onnx"
    import sys
    from omnicoder.export.onnx_export_vqdec import main as vqdec_main
    argv = sys.argv
    try:
        sys.argv = [
            "onnx_export_vqdec",
            "--codebook",
            str(codebook_path),
            "--onnx",
            str(onnx_path),
            "--hq",
            "4",
            "--wq",
            "4",
        ]
        vqdec_main()
    finally:
        sys.argv = argv

    assert onnx_path.exists()

    # Load and run a zero indices grid
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])  # type: ignore
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    import numpy as np  # type: ignore

    indices = np.zeros((1, 4, 4), dtype=np.int64)
    out = sess.run([out_name], {in_name: indices})[0]
    # Expect (B,3,H,W) and non-negative values in [0,1]
    assert out.shape[0] == 1 and out.shape[1] == 3
    assert out.min() >= 0.0 and out.max() <= 1.0


