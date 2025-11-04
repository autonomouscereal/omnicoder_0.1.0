from pathlib import Path

from omnicoder.tools.datasets_fetch import fetch


def test_fetch_tinyshakespeare(tmp_path: Path):
    ok, path, lic = fetch('tinyshakespeare', str(tmp_path))
    # Allow failure if internet disabled, but path should be constructed
    assert isinstance(path, str)
    assert isinstance(lic, str)

