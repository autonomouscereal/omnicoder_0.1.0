import json, os, tempfile, pathlib
from omnicoder.tools.kv_budget_enforce import main as kv_main


def test_kv_budget_enforce_pass(tmp_path):
    sidecar = tmp_path / 'decode.kv_paging.json'
    meta = {"page_len": 256, "n_layers": 4, "heads": 4, "dl": 64}
    sidecar.write_text(json.dumps(meta))
    out = tmp_path / 'summary.json'
    os.environ['OMNICODER_MAX_KV_MB'] = '1024'
    import sys
    sys.argv = ['omnicoder.tools.kv_budget_enforce', '--sidecar', str(sidecar), '--out', str(out)]
    kv_main()
    s = json.loads(out.read_text())
    assert s['pass'] is True


