from __future__ import annotations

"""
Retrieval Tokens API (prototype): build a retrieval context block from a PQ index
and return a string bracketed by [RET]...[/RET] suitable for prepending to a prompt.

Example:
  python -m omnicoder.inference.retrieval_tokens --pq_index weights/unified_index/pq.index \
    --query "How do landmarks enable random-access?" --topk 3
"""

import argparse
import json
from pathlib import Path
from typing import List


def _load_texts(meta_path: Path) -> List[str]:
    if not meta_path.exists():
        return []
    try:
        j = json.loads(meta_path.read_text(encoding='utf-8'))
        if isinstance(j, dict) and isinstance(j.get('docs'), list):
            return [str(x) for x in j['docs']]
    except Exception:
        pass
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description="Build [RET]...[/RET] block from a PQ index for a query")
    ap.add_argument('--pq_index', type=str, required=True, help='Path to a PqRetriever index directory or file')
    ap.add_argument('--query', type=str, required=True, help='User query to retrieve against')
    ap.add_argument('--topk', type=int, default=3)
    args = ap.parse_args()

    try:
        from omnicoder.inference.retrieval_pq import PqRetriever  # type: ignore
    except Exception as e:
        raise SystemExit(f"[error] PqRetriever unavailable: {e}")

    pq = PqRetriever(args.pq_index)
    # Optional doc metadata
    docs_meta = _load_texts(Path(args.pq_index) if Path(args.pq_index).is_dir() else Path(args.pq_index).with_suffix('.json'))
    ids, _ = pq.search(args.query, topk=int(args.topk))
    texts: List[str] = []
    for i in ids:
        try:
            if 0 <= i < len(docs_meta):
                texts.append(docs_meta[i])
        except Exception:
            continue
    block = "[RET]\n" + "\n---\n".join(texts) + "\n[/RET]" if texts else ""
    print(block)


if __name__ == '__main__':
    main()


