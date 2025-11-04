import json
import os
import tempfile
from pathlib import Path


def test_onnx_decode_respects_kv_paging_sidecar(monkeypatch):
    # Skip if onnxruntime is not installed; this is a smoke test
    try:
        import onnxruntime  # type: ignore
    except Exception:
        return
    # Create a fake ONNX path and sidecar; the runner should detect sidecar and set window
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        model = td_path / 'omnicoder_decode_step.onnx'
        # Touch model file (we won't run session graph since it's not valid; just check sidecar logic path)
        model.write_bytes(b'')
        sidecar = td_path / 'omnicoder_decode_step.kv_paging.json'
        sidecar.write_text(json.dumps({'page_len': 128, 'n_layers': 12, 'heads_per_layer': [8]*12, 'dl_per_layer': [64]*12}), encoding='utf-8')
        # Run the CLI entrypoint; it should print detection messages and not crash early due to sidecar parsing
        import subprocess, sys
        proc = subprocess.run([sys.executable, '-m', 'omnicoder.inference.runtimes.onnx_decode_generate',
                               '--model', str(model), '--provider', 'CPUExecutionProvider', '--max_new_tokens', '1'],
                               capture_output=True, text=True)
        # The session will likely fail to load an empty ONNX; we focus on sidecar detection
        out = (proc.stdout + proc.stderr)
        assert '[kv] detected kv_paging sidecar' in out or 'kv_paging' in out


