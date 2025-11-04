import os
from omnicoder.tools.torchrun_ep import main as ep_main


def test_expert_sharding_launcher_argparse_only(monkeypatch):
    # Dry-run: ensure argparse and env setup do not crash before run_module
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('OMNICODER_EP_DEVICES', 'cpu,cpu')
    # Replace run_module to prevent executing a real script
    import runpy
    called = {}
    def fake_run_module(mod, run_name=None):
        called['mod'] = mod
    monkeypatch.setattr(runpy, 'run_module', fake_run_module)
    import sys
    sys.argv = ['omnicoder.tools.torchrun_ep', '--script', 'omnicoder.training.pretrain', '--script_args', '--steps 1 --device cpu', '--devices', 'cpu,cpu']
    ep_main()
    assert called.get('mod') == 'omnicoder.training.pretrain'


