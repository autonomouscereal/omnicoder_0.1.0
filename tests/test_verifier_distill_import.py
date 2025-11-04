def test_import_verifier_distill():
    # Smoke: ensure the training script imports without optional deps
    import importlib
    m = importlib.import_module('omnicoder.training.verifier_distill')
    assert hasattr(m, 'main')


