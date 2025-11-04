import json, subprocess, tempfile, os, textwrap, random, pathlib

def run_python(code:str, tests:str, timeout=2):
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td)/"sol.py"
        path.write_text(code)
        test_path = pathlib.Path(td)/"test.py"
        test_path.write_text(tests + "\nprint('OK')\n")
        try:
            out = subprocess.run(["python", str(test_path)], timeout=timeout, capture_output=True, text=True)
            ok = (out.returncode == 0) and ("OK" in out.stdout)
            return ok, out.stdout, out.stderr
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT", ""

def pass_at_k(solutions, tests, k=5, timeout=2):
    samples = random.sample(solutions, min(k, len(solutions)))
    for s in samples:
        ok, _, _ = run_python(s, tests, timeout=timeout)
        if ok:
            return True
    return False

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=2)
    args = ap.parse_args()

    data = [json.loads(l) for l in open(args.tasks)]
    passed = 0
    for ex in data:
        ok = pass_at_k(ex["candidates"], ex["tests"], k=args.samples, timeout=args.timeout)
        passed += int(ok)
    print(f"pass@{args.samples}: {passed}/{len(data)} = {passed/len(data):.3f}")

if __name__ == "__main__":
    main()
