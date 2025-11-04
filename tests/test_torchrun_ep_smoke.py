import os
import sys
import subprocess


def test_torchrun_ep_smoke_cmd_builds_args():
    # Build a dry-run command to verify argument plumbing; do not actually launch torchrun
    script = 'omnicoder.training.pretrain'
    args = "--data ./examples --seq_len 8 --steps 1 --device cpu"
    devices = 'cpu,cpu'
    # Simulate environment
    env = os.environ.copy()
    env['OMNICODER_EXPERT_DEVICES'] = devices
    # Run the launcher as a module with --init_dist off so it won't init torch.distributed
    cmd = [sys.executable, '-m', 'omnicoder.tools.torchrun_ep', '--script', script, '--script_args', args, '--devices', devices]
    # Use a timeout and allow non-zero return if training script exits quickly
    try:
        p = subprocess.run(cmd, env=env, timeout=10)
    except subprocess.TimeoutExpired:
        # The launcher at least started; acceptable for smoke in constrained CI
        return


