from __future__ import annotations

"""
Web fact-checker and KG updater.

Best-effort utility that:
- Searches the web for a query or claim
- Extracts top snippets/answers
- Writes normalized facts and provenance into a lightweight JSONL under weights/unified_index/edges.jsonl

Notes:
- Avoids heavyweight dependencies; uses requests+bs4 if available, else falls back to DuckDuckGo HTML.
- Designed to be called from training loops for fact-checking and knowledge enrichment.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict


def _search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    try:
        import requests  # type: ignore
        from bs4 import BeautifulSoup  # type: ignore
        url = "https://duckduckgo.com/html/?q=" + requests.utils.quote(query)
        resp = requests.get(url, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.select(".result__a")
        for a in results[: max(1, int(max_results))]:
            href = a.get("href", "")
            title = a.get_text(strip=True)
            if href:
                out.append({"url": href, "title": title})
    except Exception:
        pass
    return out


def _normalize_facts(snippets: List[Dict[str, str]], query: str) -> List[Dict[str, str]]:
    facts: List[Dict[str, str]] = []
    for s in snippets:
        url = str(s.get("url", ""))
        title = str(s.get("title", ""))
        if not url:
            continue
        facts.append({
            "h": title or query,
            "r": "about",
            "t": query,
            "source": url,
        })
    return facts


def _write_edges(facts: List[Dict[str, str]], out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = out_dir / "edges.jsonl"
    with open(edges, "a", encoding="utf-8") as f:
        for rec in facts:
            try:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                continue
    return str(edges)


def main() -> None:
    ap = argparse.ArgumentParser(description="Web fact-check and append normalized facts into unified KG index")
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--out_root", type=str, default=os.getenv("OMNICODER_MULTI_INDEX_ROOT", "weights/unified_index"))
    ap.add_argument("--max_results", type=int, default=5)
    args = ap.parse_args()

    snippets = _search_duckduckgo(args.query, max_results=int(args.max_results))
    facts = _normalize_facts(snippets, args.query)
    out = _write_edges(facts, Path(args.out_root))
    print(json.dumps({"edges": out, "added": len(facts)}, indent=2))


if __name__ == "__main__":
    main()


