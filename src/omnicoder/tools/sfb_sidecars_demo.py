from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    root = Path('examples/sidecars')
    files = {
        'SFB_CLIP_JSONL': root / 'clip_pairs.jsonl',
        'SFB_ASR_JSONL': root / 'asr_pairs.jsonl',
        'SFB_CODE_TASKS_JSONL': root / 'code_tasks.jsonl',
    }
    for k, p in files.items():
        print(f"{k} -> {p} exists={p.exists()}")
    print("\nExport these in your shell before running tests/generate:")
    for k, p in files.items():
        print(f"export {k}={p}")
    print("\nMinimum SFB enabling:")
    print("export SFB_ENABLE=1")
    print("export SFB_FACTORIZER=amr,srl")
    print("export SFB_BLOCK_VERIFY=1")
    print("export SFB_BLOCK_VERIFY_SIZE=4")


if __name__ == "__main__":
    main()


