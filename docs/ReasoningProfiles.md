## Reasoning Profiles (Ω₂)

These presets enable the adaptive reasoning stack with conservative, fast defaults. Adjust as needed per task.

### Math (proofs, step-by-step)
- OMNICODER_AGOT_ENABLE=1, OMNICODER_AGOT_WIDTH=4, OMNICODER_AGOT_DEPTH=3
- OMNICODER_LATENT_BFS_ENABLE=1, OMNICODER_LATENT_BFS_WIDTH=3, OMNICODER_LATENT_BFS_DEPTH=2, OMNICODER_LATENT_BFS_BEAM=3
- OMNICODER_REFLECT_ENABLE=1, OMNICODER_REFLECT_ALPHA=0.1, OMNICODER_REFLECT_ENTROPY_MIN=2.2
- OMNICODER_SYMBOLIC_PLANNER=1, OMNICODER_PLAN_MAX_STEPS=8
- OMNICODER_GRAPHRAG_ENABLE=1 (optional if math KG available)
- OMNICODER_REASONING_MIXED_PREC=bf16

Rationale: wider short DAG and modest latent lookahead improve branching; planner nudges structure; reflection stabilizes under uncertainty.

### Code (algorithms, correctness)
- OMNICODER_AGOT_ENABLE=1, OMNICODER_AGOT_WIDTH=3, OMNICODER_AGOT_DEPTH=2
- OMNICODER_LATENT_BFS_ENABLE=1, OMNICODER_LATENT_BFS_WIDTH=2, OMNICODER_LATENT_BFS_DEPTH=2, OMNICODER_LATENT_BFS_BEAM=2
- OMNICODER_REFLECT_ENABLE=1, OMNICODER_REFLECT_ALPHA=0.08, OMNICODER_REFLECT_ENTROPY_MIN=1.8
- OMNICODER_SYMBOLIC_PLANNER=1, OMNICODER_PLAN_MAX_STEPS=6
- OMNICODER_GRAPHRAG_ENABLE=1 (project-specific KB)
- SFB_REQUIRE_MARGIN_RISE=1

Rationale: accept only improving proof-margins; planner decomposes tasks; moderate exploration balances latency and acceptance.

### Planning (tools, multi-hop, retrieval)
- OMNICODER_AGOT_ENABLE=1, OMNICODER_AGOT_WIDTH=5, OMNICODER_AGOT_DEPTH=3, OMNICODER_AGOT_TOPP=0.9
- OMNICODER_LATENT_BFS_ENABLE=1, OMNICODER_LATENT_BFS_WIDTH=3, OMNICODER_LATENT_BFS_DEPTH=3, OMNICODER_LATENT_BFS_BEAM=4
- OMNICODER_REFLECT_ENABLE=1, OMNICODER_REFLECT_ALPHA=0.12, OMNICODER_REFLECT_ENTROPY_MIN=2.0
- OMNICODER_SYMBOLIC_PLANNER=1, OMNICODER_PLAN_MAX_STEPS=10
- OMNICODER_GRAPHRAG_ENABLE=1, OMNICODER_GRAPHRAG_TOKEN_BIAS=0.03
- OMNICODER_REASONING_MIXED_PREC=bf16

Rationale: higher branching and nucleus sampling explore strategies; GraphRAG anchors entities/relations.

### Operational hints
- Enable `OMNICODER_TRACE_ENABLE=1` to record per-step traces and spot hotspots.
- Keep `OMNICODER_AGOT_TOKEN_BUDGET` aligned with depth×width to bound cost.
- For very large KGs, set FAISS knobs: `OMNICODER_FAISS_NLIST`, `OMNICODER_FAISS_PQ_M`, `OMNICODER_FAISS_NPROBE`.
- Persist certificates with `OMNICODER_CERT_OUT=weights/omega2_certs.jsonl` and analyze trends.


