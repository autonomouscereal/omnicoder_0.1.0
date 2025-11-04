import json
import subprocess
import sys


def test_bench_acceptance_outputs_tps_fields():
    # Run the bench with tiny settings; CPU-only OK
    cmd = [sys.executable, '-m', 'omnicoder.tools.bench_acceptance', '--max_new_tokens', '8']
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    # Parse last JSON line
    lines = [l for l in out.splitlines() if l.strip().startswith('{') and 'tokens_per_second' in l]
    assert lines, out
    j = json.loads(lines[-1])
    assert 'tokens_per_second' in j
    # draft fields may be None if no draft provided, but keys exist when present
    assert 'acceptance_ratio' in j


