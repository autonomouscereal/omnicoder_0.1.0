from __future__ import annotations

"""Android device-side runners (NNAPI ORT and ExecuTorch .pte).

Usage on device (Termux/ADB shell python):
  # ONNX Runtime NNAPI
  python -m omnicoder.inference.runtimes.nnapi_device_runner --model /data/local/tmp/omnicoder_decode_step.onnx --gen_tokens 64
  # ExecuTorch .pte (if executorch available)
  python -m omnicoder.inference.runtimes.nnapi_device_runner --pte /data/local/tmp/omnicoder_decode_step.pte --gen_tokens 64
"""

import argparse
import json
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='')
    ap.add_argument('--pte', type=str, default='')
    ap.add_argument('--gen_tokens', type=int, default=64)
    ap.add_argument('--prompt_len', type=int, default=16)
    ap.add_argument('--vocab_size', type=int, default=32000)
    ap.add_argument('--out', type=str, default='')
    args = ap.parse_args()

    result = {"tokens_per_sec": 0.0, "gen_tokens": args.gen_tokens, "prompt_len": args.prompt_len}
    if args.model:
        import onnxruntime as ort  # type: ignore
        sess = ort.InferenceSession(args.model, providers=["NNAPIExecutionProvider"])  # type: ignore
        input_name = sess.get_inputs()[0].name
        outputs = sess.get_outputs()
        out_names = [o.name for o in outputs]
        ids = np.random.randint(0, args.vocab_size, size=(1, args.prompt_len), dtype=np.int64)
        for t in range(ids.shape[1]):
            step = ids[:, t:t+1]
            feeds = {input_name: step}
            _ = sess.run(out_names, feeds)
        t0 = time.perf_counter()
        for _ in range(args.gen_tokens):
            step = ids[:, -1:]
            feeds = {input_name: step}
            _ = sess.run(out_names, feeds)
            ids = np.concatenate([ids, np.random.randint(0, args.vocab_size, size=(1, 1), dtype=np.int64)], axis=1)
        dt = max(1e-6, time.perf_counter() - t0)
        result["tokens_per_sec"] = float(args.gen_tokens / dt)
    elif args.pte:
        try:
            import executorch  # type: ignore
            from executorch.runtime import Runtime  # type: ignore
            # Pseudocode: actual API may differ and needs adaptation to installed ExecuTorch version
            rt = Runtime()
            prog = rt.load(args.pte)
            # Assuming single input named 'input_ids' and outputs including logits
            ids = np.random.randint(0, args.vocab_size, size=(1, args.prompt_len), dtype=np.int64)
            for t in range(ids.shape[1]):
                _ = prog.run({"input_ids": ids[:, t:t+1]})
            t0 = time.perf_counter()
            for _ in range(args.gen_tokens):
                _ = prog.run({"input_ids": ids[:, -1:]})
                ids = np.concatenate([ids, np.random.randint(0, args.vocab_size, size=(1, 1), dtype=np.int64)], axis=1)
            dt = max(1e-6, time.perf_counter() - t0)
            result["tokens_per_sec"] = float(args.gen_tokens / dt)
        except Exception:
            pass
    payload = json.dumps(result)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(payload)
    print(payload)


if __name__ == '__main__':
    main()


