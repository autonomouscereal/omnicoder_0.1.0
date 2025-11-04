## OmniCoder — State of the Union (2025-08-21)

### Where we are
- Single-button flows exist:
  - `press-play`: build/export/bench; emits decode-step ONNX/Core ML/ExecuTorch artifacts and provider benches.
  - `lets-gooooo`: tests → orchestrated training by time budget → export → provider bench, with optional export-to-phone.
- Architecture: Sparse MoE Transformer with hierarchical/multi-head/GRIN routing; landmark/random-access attention; KV paging/quant (u8/NF4); speculative draft+verify; SCMoE; retrieval PQ/kNN-LM; continuous latents (image/audio) with tiny refiners; temporal video module and AV-sync; cross‑modal verifier and unified pre‑alignment.
- Mobile export: decode-step ONNX with sidecars (kv_paging, kvq, retention, prefetch); Core ML MLProgram; ExecuTorch path; provider microbench with thresholds.
- Auto resources: `OMNICODER_AUTO_RESOURCES=1` scales threads/workers to available cores; avoids underutilization in containers.
- Tests: Broad suite present across exporters, runtimes, routers, long-context, KVQ/paging, video/audio/image heads, and canaries. In some container runs, OS may kill long pytest without error; use focused subsets or raise limits.

### Gaps vs. target (2–4 GB frontier mobile, fully multimodal)
- Device kernels: fused MLA/MQA and int4 GEMM for NNAPI/Core ML/DML (currently partial via DML composite/native; NNAPI/Core ML kernels pending).
- DynamicCache: migrate decode-step export to true DynamicCache once PyTorch stabilizes upstream; remove explicit KV tensors.
- Long-context: YaRN/PI training passes and 32k/128k stability validation; exporter baked params already wired.
- Unified vocab/codebooks: integrity checks across text/image/video/audio; enforce in exporters and runners.
- Mobile apps: Android ExecuTorch NNAPI and iOS Core ML sample apps with tokenizer/KV reuse and streaming UI.
- Dataset/teachers: fill `profiles/{datasets,teachers}.json` with real sources; add checksums and curation.

### What’s new (this pass)
- SFB env knobs added (SFB_ENABLE, SFB_FACTORIZER, SFB_BP_ITERS, SFB_COMPILE_SPN, SFB_MAX_TREEWIDTH, SFB_BLOCK_VERIFY, SFB_GOAL_PRIOR, SFB_PROOF_MARGIN) for optional semantic factorization/verification alongside HRM/Omega.
- ExecutionPlan updated with guidance for long pytest runs in containers.

### Action plan (immediate)
1) Populate dataset/teacher profiles with actual paths/IDs (`profiles/datasets.json`, `profiles/teachers.json`).
2) Enable persistent `/models` volume and set `HF_HOME=/models/hf` across environments.
3) Run `lets-gooooo --budget_hours 1` on a GPU box to produce stage artifacts and provider benches; inspect `TRAINING_SUMMARY.md` and thresholds.
4) If OS kills tests in Docker, run focused subsets (`pytest -k onnx -vv -rA`) or increase container limits.
5) Plan multi-teacher KD and RL schedules for the training bench (2×24 GB GPUs), enabling EP (`tools/torchrun_ep.py`).

### Longer-horizon upgrades
- Implement device-native fused MLA/MQA and int4 GEMM; align AWQ/GPTQ packing with provider kernels.
- Adopt DynamicCache exporter and update runners; add conformance tests.
- Expand RL rewards (CLIPScore/FID/FVD/FAD; code pass@k) and kNN-LM retrieval policies.
- Solidify iOS/Android apps and deploy flow (export-to-phone) with TPS/KV dashboards.


