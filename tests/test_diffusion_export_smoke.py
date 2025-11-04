def test_diffusion_export_onnx_smoke():
    from omnicoder.export.diffusion_export import export_onnx
    # Without a local model id/path this will likely return False; we just ensure it does not crash the process.
    ok = export_onnx(model_id=None, local_path=None, out_dir=__import__('pathlib').Path('weights/sd_export'), opset=17)
    assert ok in (True, False)


