import os
from omnicoder.inference.generate import main as gen_main
import sys


def test_generate_env_defaults_smoke(monkeypatch):
    # Ensure env-driven defaults don't crash the CLI entrypoint
    argv = sys.argv
    try:
        monkeypatch.setenv('OMNICODER_PROMPT', 'hi')
        monkeypatch.setenv('OMNICODER_MAX_NEW_TOKENS', '1')
        monkeypatch.setenv('OMNICODER_DEVICE', 'cpu')
        sys.argv = ['generate']
        gen_main()
    finally:
        sys.argv = argv

