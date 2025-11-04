import os
from pathlib import Path


def test_pq_build_and_search(tmp_path: Path):
    # Build a minimal text folder
    docs = tmp_path / 'docs'
    docs.mkdir()
    (docs / 'a.txt').write_text('vector search is useful for retrieval augmentation', encoding='utf-8')
    (docs / 'b.txt').write_text('mixture of experts routes tokens to experts', encoding='utf-8')
    # Import inside test to avoid import costs if faiss missing
    from omnicoder.inference.retrieval_pq import PqRetriever
    out_dir = tmp_path / 'pq'
    pq = PqRetriever.build_from_text_folder(str(docs), chunk_size=16, stride=12, m=8, ks=16, out_dir=str(out_dir))
    assert (out_dir / 'pq.index.npy').exists() or (out_dir / 'pq.index').exists()
    hits = pq.search('experts routing', k=2, partition_size=128)
    assert len(hits) >= 1

