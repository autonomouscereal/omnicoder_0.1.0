# Project Plan (Where we were, where we are, where we're going)

## Where we were
- Initial skeleton with toy MoE transformer, ONNX export, minimal CLI, and early mobile focus.
- Growing README/TODO/CHANGELOG with overlapping content.

## Where we are (0.1.9 series)
- Text path runs locally; decode-step ONNX export and runner validated.
- KV quantization (u8/NF4), long-context export variants, provider microbench harness.
- Multimodal scaffolding with image/video/audio adapters and VQ-VAE trainers.
- One-button orchestrators: `press_play`, `run_training` (time‑budgeted), and `mobile_packager`.
- Teachers/datasets: defaults live in `profiles/teachers.json` and `profiles/datasets.json`.
  - Teachers: Llama‑3.1‑8B text, StarCoder2‑3B code, CLIP ViT‑B/32 VL, faster‑whisper‑medium ASR, Coqui XTTS v2 TTS; draft presets for Qwen2.5 (1–3B) drive speculative decoding acceptance.
  - Datasets: text `.txt` glob, code/VL/VQA JSONL adapters, audio/video sample trees, and a unified `retrieval_multi_index` root for PQ/embeddings.
- Expert paging runtime: env‑gated (`OMNICODER_EXPERT_PAGING=1`) LRU pager with capacity cap or `OMNICODER_EXPERT_PAGING_BUDGET_MB` auto‑derivation and router‑probability prefetch.
 - Vision exports: compact MobileViT/EfficientViT ONNX export with provider maps; lightweight open‑vocab grounding heads (YOLO‑E style) exporter to ONNX/Core ML/ExecuTorch via `export/onnx_export_grounding.py` (integrated into `mobile_packager` with `--vision_export_grounding`).
 - Video generation: keyframe cadence and ORT‑friendly linear interpolation in `VideoGenPipeline`; generation metadata sidecar includes `onnx_video_dir` when the ORT i2v path is used. FVD evaluation lives in `eval/video_eval.py`.

### Verification (2025-08-17)
- Full Docker GPU run: 89/89 tests passed, 25 warnings. Log saved to `tests_logs/docker_pytest_full_after_ptq.txt`.
- Stabilized per-op PTQ presence test by adding `OMNICODER_EXPORT_TINY` exporter knob and making per-op PTQ helper guarantee QDQ presence even on fused graphs.
- Added time-budgeted training probe (`omnicoder.tools.train_probe`) and compose service `train_probe`.

## Where we're going (prioritized)
1) Provider kernels and DynamicCache
   - Fused MLA/MQA kernels (NNAPI/Core ML/DML); align int4 packing and KV u8/NF4.
   - Migrate ONNX exporters to dynamo + DynamicCache; add conformance tests.
2) Long-context stability
   - 32k/128k decode-step canaries; windowed decode policies; YaRN/PI training.
3) Quantization end-to-end
   - AWQ/GPTQ exports; per-op PTQ maps; KV calibration sidecars and tests.
4) Apps and UX
   - Android ExecuTorch NNAPI demo and iOS Core ML demo with tokenizer & streaming UI.
   - Device smokes: Android ADB runner; iOS Core ML smoke helper (`tools.ios_coreml_smoke`) gated by `OMNICODER_IOS_SMOKE=1`.
   - Consolidated quickstart flows via `press-play` console script and `.env` defaults.
5) Training quality
   - KD + RL (GRPO/PPO) across text/code/math/VL/ASR/TTS; verifier-head KD.
   - Data engine for scalable ingestion/filtering/synthesis across modalities (added `training/data/engine.py`).

### Expert Parallel (EP) and BTM Upcycling

- EP validation: the orchestrator auto-probes expert-parallel sharding when ≥2 GPUs are available; `tools/train_probe` also auto-runs an EP probe when `--ep_devices` is not specified and multiple GPUs are detected. Configure devices with `OMNICODER_EP_DEVICES`.
- Branch-Train-Merge (BTM): `training/btm_upcycle.py` merges domain expert checkpoints (code/math/VL/ASR/TTS) into the student MoE and optionally fine-tunes the router. Orchestrator can run BTM automatically when `OMNICODER_BTM_DOMAINS` is set.

### Unified Pre-Alignment

- `training/pre_align_all.py` runs a unified pre-alignment stage over text/image/audio/video (best-effort) and can build a unified multimodal index via `tools.multi_index_build`.
- Orchestrator supports a pre-align stage; artifacts are emitted under `weights/pre_align.pt` and `weights/unified_index` when enabled.

## Milestones
- M1: Decode-step conformance with dynamo + DynamicCache
- M2: Device kernels MVP (one provider) with tokens/s threshold
- M3: 32k stability + KV paging defaults in runners
- M4: Mobile demos shipping (Android/iOS) with streaming UI
- M5: KD student baseline and auto-bench coverage across tasks

See `todo/` and root `TODO.md` for the actionable backlog and bug tracker. Historical entries remain in `CHANGELOG.md`.

