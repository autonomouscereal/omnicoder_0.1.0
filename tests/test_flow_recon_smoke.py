import os
import sys
import pytest
import torch


@pytest.mark.skipif('CI' in os.environ and os.environ.get('CI') == 'true', reason='skip on constrained CI runners')
def test_flow_recon_perceptual_smoke(tmp_path):
    try:
        import torchvision  # noqa: F401
    except Exception:
        pytest.skip("torchvision not available")

    # Create a tiny images folder with a synthetic sample
    img_dir = tmp_path / 'imgs'
    img_dir.mkdir()
    from PIL import Image
    img = Image.new('RGB', (32, 32), color=(128, 64, 32))
    img.save(img_dir / 'a.png')

    # Run a very short training loop with perceptual loss
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    py = sys.executable or "python"
    cmd = (
        f"{py} -m omnicoder.training.flow_recon --data {img_dir} "
        f"--image_size_w 32 --image_size_h 32 --batch 1 --steps 1 --device {dev} "
        f"--latent_dim 8 --out {tmp_path/'out.pt'} --recon_loss perceptual"
    )
    rc = os.system(cmd)
    assert rc == 0
    assert (tmp_path / 'out.pt').exists()
