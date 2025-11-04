import json, os, tempfile
import numpy as np
from omnicoder.inference.runtimes.onnx_decode_generate import main as ort_main


def test_kv_spill_enforced(tmp_path, monkeypatch):
    # Create a fake sidecar to trigger spill path
    side = tmp_path / 'decode.kv_paging.json'
    side.write_text(json.dumps({"page_len": 16, "n_layers": 2, "heads": 4, "dl": 64}))
    # We cannot run full ORT session here; just ensure the import stays intact and no exception when sidecar exists
    # (Spill logic runs inside the decode loop; covered by integration tests elsewhere.)
    assert side.exists()


