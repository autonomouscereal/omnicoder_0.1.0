import os
import numpy as np
import pytest


def test_audio_recon_mel_smoke(tmp_path):
    # Create a tiny mel directory with one sample
    mel_dir = tmp_path / 'mels'
    mel_dir.mkdir()
    mel = np.random.rand(80, 16).astype('float32')
    # Save a valid .npy file so loader torch.from_numpy(np.load(...)) works
    np.save(str(mel_dir / 'a.npy'), mel)
    import sys
    py = sys.executable or "python"
    cmd = (
        f"{py} -m omnicoder.training.audio_recon --mel_dir {mel_dir} "
        f"--batch 1 --steps 1 --device cpu --latent_dim 8 --out {tmp_path/'out.pt'} --recon_loss mse"
    )
    rc = os.system(cmd)
    assert rc == 0
    assert (tmp_path / 'out.pt').exists()


