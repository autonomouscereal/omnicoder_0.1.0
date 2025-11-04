from __future__ import annotations

"""
Shared Semantic Memory (prototype)

Maintains a small, trainable key→prototype store across modalities. Prototypes
are simple vectors (per modality) with optional text snippets that can be used
to build retrieval context for generation.

CLI usage examples:
  # Add prototypes from JSONL (one per line: {"key":"concept","text":"..."})
  python -m omnicoder.inference.semantic_memory --db weights/semantic_memory.json \
    --add_jsonl examples/semantic.jsonl

  # Query and print a [SEM]...[/SEM] block for a user query
  python -m omnicoder.inference.semantic_memory --db weights/semantic_memory.json \
    --query "long-context landmarks" --topk 3
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class MemoryItem:
    key: str
    text: str
    modality: str = "text"
    # Future: store embedding vectors per modality; for now, text-only


class SemanticMemory:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.items: List[MemoryItem] = []
        self._load()

    def _load(self) -> None:
        if self.db_path.exists():
            try:
                data = json.loads(self.db_path.read_text(encoding='utf-8'))
                if isinstance(data, list):
                    self.items = [MemoryItem(**x) for x in data if isinstance(x, dict)]
            except Exception:
                self.items = []

    def save(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.write_text(json.dumps([asdict(x) for x in self.items], indent=2), encoding='utf-8')

    def add(self, item: MemoryItem) -> None:
        self.items.append(item)

    def add_from_jsonl(self, jsonl_path: str) -> int:
        cnt = 0
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        j = json.loads(line)
                        if isinstance(j, dict) and 'key' in j and 'text' in j:
                            self.add(MemoryItem(key=str(j['key']), text=str(j['text']), modality=str(j.get('modality','text'))))
                            cnt += 1
                    except Exception:
                        continue
        except Exception:
            return cnt
        return cnt

    def query_text(self, query: str, topk: int = 3) -> List[MemoryItem]:
        # Very simple TF-IDF-free proxy: rank by substring matches then length
        scored: List[tuple[float, MemoryItem]] = []
        q = query.lower()
        for it in self.items:
            t = it.text.lower()
            score = 0.0
            if q in t:
                score += 2.0
            for w in q.split():
                if w and w in t:
                    score += 0.2
            score += min(1.0, len(it.text) / 2000.0)
            scored.append((score, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[: max(1, int(topk))]]

    @staticmethod
    def to_block(items: List[MemoryItem]) -> str:
        if not items:
            return ""
        body = "\n---\n".join([f"[{it.modality}] {it.text}" for it in items])
        return f"[SEM]\n{body}\n[/SEM]"


def main() -> None:
    ap = argparse.ArgumentParser(description="Shared semantic memory (prototype)")
    ap.add_argument('--db', type=str, required=True, help='Path to semantic_memory.json database')
    ap.add_argument('--add_jsonl', type=str, default='', help='JSONL with {key,text,modality?}')
    ap.add_argument('--query', type=str, default='', help='Query string')
    ap.add_argument('--topk', type=int, default=3)
    args = ap.parse_args()

    mem = SemanticMemory(args.db)
    if args.add_jsonl.strip():
        n = mem.add_from_jsonl(args.add_jsonl.strip())
        mem.save()
        print(json.dumps({'added': int(n), 'db': args.db}))
    if args.query.strip():
        items = mem.query_text(args.query.strip(), topk=int(args.topk))
        print(SemanticMemory.to_block(items))


if __name__ == '__main__':
    main()


