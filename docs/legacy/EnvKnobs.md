# OmniCoder environment knobs (concise)

This page summarizes commonly used OMNICODER_* environment variables grouped by topic. Copy env.example.txt to .env and edit there. Orchestrators and runners load .env automatically.

- General
  - HF_HOME, TRANSFORMERS_CACHE: default to `/models/hf` when `/models` exists; otherwise `./models/hf`. Prevent re-downloads when running repeatedly.
  - OMNICODER_OUT_ROOT: output root (weights/releases)
  - OMNICODER_SEED: reproducible runs (Python/NumPy/Torch)
  - OMNICODER_AUTO_RESOURCES=1: auto-scale OMP/MKL/Torch threads and DataLoader workers
  - OMNICODER_THREADS: hard override for thread count (takes precedence)
  - OMNICODER_THREADS_FACTOR: scale effective CPUs (default 1.0) when auto-scaling; e.g., 0.5 uses half
  - OMNICODER_WORKERS: hard override for DataLoader workers
  - OMNICODER_WORKERS_MAX: cap for auto-selected DataLoader workers
  - OMNICODER_PROVIDER_THRESHOLD_FRACTION: raise provider TPS thresholds toward measured TPS (default 0.95)
  - EXECUTE_TESTS: set to `false` while another long-running test process is active; set back to `true` before running `lets-gooooo`.

- ONNX runner
  - OMNICODER_ORT_PROVIDER: CPUExecutionProvider|DmlExecutionProvider|CoreMLExecutionProvider|NNAPIExecutionProvider|auto
  - OMNICODER_KVQ: KV quant scheme none|u8|nf4; OMNICODER_KVQ_GROUP: group size
  - OMNICODER_WINDOW_SIZE: sliding window cap; landmarks default on when >0
  - Speculative: OMNICODER_VERIFY_THRESHOLD, OMNICODER_SPEC_DRAFT_LEN, OMNICODER_BLOCK_VERIFY, OMNICODER_BLOCK_VERIFY_SIZE, OMNICODER_DRAFT_VERIFY_THRESHOLD, OMNICODER_TREE_WIDTH, OMNICODER_TREE_DEPTH
  - KV paging/prefetch: OMNICODER_KV_PREFETCH_PREDICTOR, OMNICODER_KV_PREFETCH_KEEP
  - KV spill precision for paged caches: OMNICODER_KV_SPILL_PREC=fp16|bf16 to bound memory in ONNX runner
  - Logit bias: OMNICODER_LOGIT_BIAS_FILE, OMNICODER_LOGIT_BIAS_ALPHA
  - Prompting: OMNICODER_SYSTEM_PROMPT, OMNICODER_PROMPT_TEMPLATE; retrieval: OMNICODER_RETRIEVE_PATH, OMNICODER_RETRIEVE_K, OMNICODER_RETRIEVE_MAX_CHARS

- PyTorch generator
  - Long context: OMNICODER_TARGET_CTX, OMNICODER_WINDOW_SIZE, OMNICODER_MEM_SLOTS, OMNICODER_USE_LANDMARKS=auto
  - KV retention/compress: OMNICODER_KV_RETENTION (sidecar), OMNICODER_KV_COMPRESS_SIDECAR (AE)
  - Speculative: same as ONNX; tree search: OMNICODER_TREE_WIDTH, OMNICODER_TREE_DEPTH
  - kNN-LM: OMNICODER_KNN_CACHE, OMNICODER_KNN_K, OMNICODER_KNN_LAMBDA, OMNICODER_KNN_MAX_ITEMS, OMNICODER_KNN_CACHE_PATH, OMNICODER_KNN_PRUNE_EVERY
  - Cross-modal verifier (image/video): OMNICODER_CM_VERIFIER=1 to enable, OMNICODER_CM_THRESHOLD to set acceptance threshold (default 0.6)

- Training/orchestrators
  - Budget/device: OMNICODER_TRAIN_BUDGET_HOURS, OMNICODER_TRAIN_DEVICE
  - Presets/teachers: OMNICODER_TRAIN_PRESET, OMNICODER_STUDENT_PRESET, OMNICODER_TEACHER
  - Resume/logging/AMP/accum: OMNICODER_PRETRAIN_*, OMNICODER_KD_*, OMNICODER_VL_*
  - RLHF/GRPO/PPO: OMNICODER_ENABLE_GRPO, OMNICODER_ENABLE_PPO, OMNICODER_RUN_RM
  - Provider benches auto-update: OMNICODER_AUTO_UPDATE_THRESHOLDS=1
  - Teachers/datasets profiles: `profiles/teachers.json` (per-domain high-ROI defaults across presets), `profiles/datasets.json` (paths and formats). Backed models/data can be stored in `/models` to avoid re-download.
  - BTM: OMNICODER_BTM_DOMAINS (space-separated domain ckpts), OMNICODER_BTM_ROUTER_STEPS, OMNICODER_BTM_ROUTER_LR
  - EP: OMNICODER_EP_DEVICES (e.g., cuda:0,cuda:1)
  - Expert paging (on-device):
    - OMNICODER_EXPERT_PAGING=1 to enable LRU of experts
    - OMNICODER_EXPERT_PAGING_CAP to set resident experts (fixed)
    - OMNICODER_EXPERT_PAGING_BUDGET_MB to derive capacity from a memory budget
    - OMNICODER_EXPERT_PREFETCH_N to prefetch beyond top‑k based on router probs
    - OMNICODER_EXPERT_PAGING_DIR to persist evicted experts (weight streaming)
    - OMNICODER_EXPERT_PAGING_PERSIST=1 to save on evict; 0 to disable persistence

Refer to env.example.txt for the full catalog and defaults.
 
## Semantic-Factoring Brain (SFB)

Enable the parallel semantic factorization and acceptance stack. All components are optional/fail-safe; missing deps degrade to no-op.

- `SFB_ENABLE=1` – turn on SFB parallel stack
- `SFB_FACTORIZER=amr,srl` – parsing backends (heuristics if heavy deps missing)
- `SFB_BP_ITERS=10` – iterations for lightweight message passing
- `SFB_COMPILE_SPN=1` – compile frequent factor subgraphs (placeholder caching)
- `SFB_MAX_TREEWIDTH=8` – future compiler hint
- `SFB_BIAS_ALPHA=0.2` – strength for messages→logit biasing (falls back to `OMNICODER_LOGIT_BIAS_ALPHA`)
- `SFB_BLOCK_VERIFY=1` – prefer SFB-controlled block-verify gating
- `SFB_BLOCK_VERIFY_SIZE=4` – speculative verification block size (4–8 recommended)
- `SFB_REQUIRE_MARGIN_RISE=1` – only accept if proof-margin is non-decreasing
- `SFB_GOAL_PRIOR=code:0.8,vqa:0.3` – seed pragmatic goal priors
- `SFB_PROOF_MARGIN=auto` – numeric threshold or `auto`
- `SFB_REFRESH_TOKENS=16` – cadence to refresh factor proposals from decoded tail

### Multimodal/metrics sidecars (for gating and factor scores)

- `SFB_CLIP_JSONL` – JSONL with `{file: ..., prompt: ...}` pairs; feeds CLIPScore → `clip_z`
- `SFB_FVD_PRED_DIR`, `SFB_FVD_REF_DIR` – directories of `.mp4` for Fréchet Video Distance → `video_z`
- `SFB_FAD_PRED_DIR`, `SFB_FAD_REF_DIR` – directories of `.wav` for Fréchet Audio Distance → `audio_z`
- `SFB_ASR_JSONL` – JSONL with ASR pairs `{file, ref, hyp?}`; WER → `audio_z`
- `SFB_CODE_TASKS_JSONL` – JSONL with `{candidates: [], tests: "..."}`; pass@k → code factor score and arbiter

Notes:
- Provide any subset; missing inputs default to zeros so the LLM path remains unaffected.
- For ASR/FAD/FVD, install optional deps as needed (see README for packages).

## Vision/DINOv3 knobs

- OMNICODER_VISION_BACKEND: dinov3|auto|timm_vit_tiny|timm_dinov2|siglip (default dinov3 in `.env.example`).
- OMNICODER_DINOV3_VARIANT: DINOv3 ViT variant to load via torch.hub (e.g., vit_base14, vit_large14).
- OMNICODER_DINOV3_REPO: torch.hub repo (default `facebookresearch/dinov3`).

These control `modeling/multimodal/vision_encoder.VisionBackbone` and flow into pre-align and cross-modal alignment trainers by default.