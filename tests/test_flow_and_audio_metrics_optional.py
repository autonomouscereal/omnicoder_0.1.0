import subprocess, sys


def test_flow_recon_cli_parses_and_runs_one_step(tmp_path):
    # Create a minimal folder; dataset should fall back gracefully when empty
    data = tmp_path / 'images'; data.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, '-m', 'omnicoder.training.flow_recon',
        '--data', str(data), '--steps', '1', '--device', 'cpu', '--fid_metrics'
    ], check=True)


def test_audio_recon_cli_parses_and_runs_one_step(tmp_path):
    mel_dir = tmp_path / 'mels'; mel_dir.mkdir(parents=True, exist_ok=True)
    # create a tiny mel.npy
    import numpy as np
    np.save(str(mel_dir / 'a.npy'), np.zeros((80, 4), dtype='float32'))
    subprocess.run([
        sys.executable, '-m', 'omnicoder.training.audio_recon',
        '--mel_dir', str(mel_dir), '--steps', '1', '--device', 'cpu'
    ], check=True)


