import sys
from pathlib import Path


def test_pretrain_parses_vg_flags(tmp_path: Path) -> None:
    from omnicoder.training.pretrain import main as pretrain_main
    argv = sys.argv
    try:
        sys.argv = [
            'pretrain',
            '--data', str(tmp_path),
            '--steps', '1',
            '--vg_clipscore',
            '--vg_interval', '1',
        ]
        pretrain_main()
    finally:
        sys.argv = argv


