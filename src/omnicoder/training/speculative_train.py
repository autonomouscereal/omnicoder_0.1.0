from __future__ import annotations

"""
Speculative decoding training: train a compact draft student and tune verifier thresholds.

This script wraps KD for a tiny draft model and then runs the acceptance bench to emit a
recommended threshold that can be written into `profiles/acceptance_thresholds.json`.

Usage:
  python -m omnicoder.training.speculative_train --teacher microsoft/phi-2 \
    --student_preset mobile_2gb --steps 500 --out_ckpt weights/omnicoder_draft_kd.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path


def run(cmd: list[str]) -> int:
    import subprocess
    print('[run]', ' '.join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description='Speculative decoding draft+verify training')
    ap.add_argument('--teacher', type=str, default=os.getenv('OMNICODER_TEACHER', 'microsoft/phi-2'))
    ap.add_argument('--data', type=str, default=os.getenv('OMNICODER_DATA_CODE', 'examples/code_eval'))
    ap.add_argument('--student_preset', type=str, default=os.getenv('OMNICODER_STUDENT_PRESET', 'mobile_2gb'))
    ap.add_argument('--steps', type=int, default=int(os.getenv('OMNICODER_DRAFT_STEPS','500')))
    ap.add_argument('--device', type=str, default=os.getenv('OMNICODER_TRAIN_DEVICE', 'cuda'))
    ap.add_argument('--out_ckpt', type=str, default='weights/omnicoder_draft_kd.pt')
    ap.add_argument('--mobile_preset', type=str, default=os.getenv('OMNICODER_TRAIN_PRESET', 'mobile_4gb'))
    args = ap.parse_args()

    out_root = Path('weights')
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) KD train a tiny draft (LoRA) using existing draft_train wrapper
    rc = run([
        sys.executable, '-m', 'omnicoder.training.draft_train',
        '--data', args.data,
        '--steps', str(int(args.steps)),
        '--device', args.device,
        '--teacher', args.teacher,
        '--student_mobile_preset', args.student_preset,
        '--lora', '--lora_r', '8', '--lora_alpha', '16', '--lora_dropout', '0.05',
        '--out_ckpt', args.out_ckpt,
    ])
    if rc != 0:
        print('[warn] draft KD failed or skipped')

    # 2) Run acceptance bench to produce recommended threshold
    acc_json = out_root / 'draft_acceptance.json'
    rc = run([
        sys.executable, '-m', 'omnicoder.tools.bench_acceptance',
        '--mobile_preset', args.mobile_preset,
        '--max_new_tokens', '64', '--verify_threshold', '0.0', '--verifier_steps', '1',
        '--speculative_draft_len', '1', '--multi_token', '1', '--draft_ckpt', args.out_ckpt,
        '--draft_preset', args.student_preset, '--out_json', str(acc_json),
    ])
    if rc == 0 and acc_json.exists():
        try:
            data = json.loads(acc_json.read_text(encoding='utf-8'))
            thr = float(data.get('recommended_threshold', 0.0)) if isinstance(data, dict) else 0.0
            if thr > 0.0:
                prof = Path('profiles')
                prof.mkdir(parents=True, exist_ok=True)
                acc_path = prof / 'acceptance_thresholds.json'
                cur = {}
                if acc_path.exists():
                    try:
                        cur = json.loads(acc_path.read_text(encoding='utf-8'))
                    except Exception:
                        cur = {}
                cur[str(args.mobile_preset)] = thr
                acc_path.write_text(json.dumps(cur, indent=2), encoding='utf-8')
                print(f'[write] updated {acc_path} with {args.mobile_preset}={thr}')
        except Exception as e:
            print('[warn] threshold update failed:', e)

    print(json.dumps({'status': 'ok', 'draft_ckpt': args.out_ckpt}))


if __name__ == '__main__':
    main()


