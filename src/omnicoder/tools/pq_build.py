import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a small PQ index over a folder of .txt files")
    ap.add_argument("root", type=str, help="Folder of .txt files to index")
    ap.add_argument("out_dir", type=str, help="Output directory for the PQ index")
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--nlist", type=int, default=128)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--ks", type=int, default=256)
    ap.add_argument("--max_docs", type=int, default=0)
    ap.add_argument("--from_embeddings", type=str, default="", help="Path to .npy float32 embeddings (N,D)")
    args = ap.parse_args()

    from omnicoder.inference.retrieval_pq import PqRetriever
    if args.from_embeddings:
        import numpy as _np
        emb = _np.load(args.from_embeddings).astype('float32')
        # Create trivial offsets (no source paths). Users can overwrite as needed.
        offsets = [(str(Path(args.root)), i) for i in range(emb.shape[0])]
        pq = PqRetriever.build_from_embeddings(emb, offsets=offsets, m=int(args.m), ks=int(args.ks), out_dir=str(args.out_dir))
    else:
        pq = PqRetriever.build_from_folder(
            args.root,
            args.out_dir,
            dim=int(args.dim),
            nlist=int(args.nlist),
            m=int(args.m),
            ks=int(args.ks),
            max_docs=(int(args.max_docs) if int(args.max_docs) > 0 else None),
        )
    # Write a simple budget profile sidecar
    try:
        pq.write_budget_profile(str(Path(args.out_dir) / 'budget_profile.json'))
    except Exception:
        pass
    print(str(args.out_dir))


if __name__ == "__main__":
    main()


