import os, json, numpy as np
from pathlib import Path
from omnicoder.inference.retrieval_pq import PqRetriever


def test_pq_retriever_search(tmp_path: Path):
    # Build small index from temporary text files
    docs = tmp_path / 'docs'
    docs.mkdir(parents=True, exist_ok=True)
    (docs / 'a.txt').write_text('hello world one two three', encoding='utf-8')
    (docs / 'b.txt').write_text('another document with words', encoding='utf-8')
    (docs / 'c.txt').write_text('hello cosmos and world', encoding='utf-8')
    out = tmp_path / 'pq'
    # Pick m that divides TF-IDF dim; for small vocab from tiny docs, use m=2
    pq = PqRetriever.build_from_text_folder(str(docs), chunk_size=16, stride=12, m=2, ks=8, out_dir=str(out))
    res = pq.search('hello world', k=3)
    assert len(res) == 3
    # Ensure return schema (row_index, score, reference)
    row, sc, ref = res[0]
    assert isinstance(row, int)
    assert isinstance(sc, float)
    assert isinstance(ref, str) and ('#chunk' in ref)


