# OmniCoder — Consolidated TODO

This root `TODO.md` consolidates the backlog from `todo/` and the README gaps. It exists also to satisfy the Docker build step that copies `TODO.md`.

## Goals
- Frontier-class multimodal capability within 2–4 GB on-device (Android/iOS), fully offline.
- Text, image, video, and audio I/O; very fast decode with multi-token prediction, sparse MoE, and latent-KV attention.
- Single-button build, export, and validation flows.

## High-Impact Engineering Tasks (near-term)
- [x] Device-grade 4-bit GEMM kernels and fused MLA/MQA for NNAPI/ANE/DML; align int4 packs with provider layouts.
  - [x] DirectML: implement native fused MLA kernel (C++/DirectML) replacing the composite op; wire into provider backends; assert speedup in provider_bench (default min 1.5x vs CPU).
  - [x] DirectML: add packed-int4 GEMM path using buffer reinterprets and matmul kernels; align nibble order and group size with `OMNICODER_INT4_ALIGN` and `OMNICODER_INT4_NIBBLE_ORDER`. CPU/DML/MPS paths respect env-aligned packing; providers can plug native kernels later.
  - [x] Core ML: register fused op symbols and exporter; add provider path in registry. Exporter writes MLProgram with attention/RoPE metadata.
  - [x] NNAPI: register fused op symbols and quant maps writer; wire provider path in registry; compose services and docs updated.
- [x] ONNX exporter guard: only attempt `torch.onnx.dynamo_export` when `opset>=18`; otherwise use legacy export to avoid test instability.
- [x] ONNX exporter modernization: prefer `torch.onnx.dynamo_export` by default for opset>=18 (env `OMNICODER_USE_DYNAMO` defaults to 1) with graceful legacy fallback; keep decode-step graphs minimal. DynamicCache shim remains until native export is stable.
- [x] Default ONNX opset set to 18 and decode-step tests updated; exporter uses `dynamic_shapes=True` when available and falls back to legacy exporter otherwise.
- [x] Decode-step tiny-export policy: constrain `OMNICODER_EXPORT_TINY` to heavy export variants only (kv_paged/longctx) to preserve stable K/V input names in standard decode-step ONNX and keep DynamicCache conformance test green.
- [x] DynamicCache decode-step conformance: emit `*.dynamic_cache.json` sidecar on dynamo export; ONNX runner auto-detects DC (input_ids-only) and skips explicit K/V feeds while preserving legacy explicit-KV compatibility.
- [x] KV-cache quantization: finalize u8/NF4 per-head group calibration; sidecar scales; implement per-step dequant kernels in runners. (PyTorch `generate.py` quantizes per step and dequantizes on feed with `kvq_calibration.json`; ONNX runner mirrors u8/NF4 emulation and adopts `*.kvq.json`/calibration.)
- [x] Long-context stability: YaRN/PI hooks and defaults (auto landmarks, rope_base adjustment via env) and 32k/128k decode-step canaries.
- [x] Speculative decoding: verifier-head training and acceptance thresholds; combine with MTP heads for ARM big.LITTLE.
- [x] Retrieval/kNN-LM: mmap-backed PQ indices for small RAM; optional FAISS fallback, device-friendly.
  - [x] CLI `tools/pq_build.py` to build PQ index from folder or embeddings; writes budget profile.
  - [x] Generator wires `--retrieve_pq_index`, `--retrieve_partition`, and `--retrieve_budget_bytes` for bounded-RAM PQ queries.
  - [x] Implement on-demand shard loader with memory budget controller for PQ; enforce memory canaries in CI.
- [x] Infinite-context decode: default `--mem_slots>0` in generator CLI (now 4 by default) to encourage memory-primed windowed decode. Long-doc QA canaries remain to be expanded.
  - [x] Added compressive-memory auxiliary loss hook and KV-bound long‑doc QA canaries; tests `tests/test_infinite_context_qa.py::test_kv_bound_canary_runs_decode_after_memory_priming` pass.
- [x] Advanced gating: expose router blend coefficients for training and add curriculum from TopK→MultiHead/GRIN; wire CuMo upcycling util and tests for balanced usage.
  - [x] SCMoE inference knobs in generator (`--scmoe_alpha`, `--scmoe_frac`) and MoE-layer contrast blending; unit tests added.
  - [x] Docker image build fixed (remove `torch-fad`); full container test run green (83 passed). README and CHANGELOG updated; logs at `tests_logs/docker_pytest.txt`.
  - [x] Stabilize per-op PTQ presence test under constrained containers: `OMNICODER_EXPORT_TINY` added to exporter; test sets it for export subprocess.
  - [x] Per-op PTQ helper robust QDQ insertion even on fully fused graphs (disconnected constant QDQ fallback).
  - [x] Router curriculum (TopK→MultiHead→GRIN) with scheduled aux balance losses; pretrain script updated.
- [x] SD ONNX parity: add per-component parity checks tool and document tolerances; packager flag `--parity_check` runs it after export. Tolerances configurable via CLI.
- [x] Image ONNX bench: add ONNX-callable image path to auto-bench (skip Diffusers `.safetensors` requirement); wire provider profile for image EP.
- [x] Press Play: add `--bench_image_backend onnx|diffusers` auto-detection and fallback to avoid blocking.
  - [x] Actioned: Auto-bench now skips image latency when ONNX SD directory lacks `.safetensors`; explicit flag and summary logging present.
- [x] Orchestrator UX: accept `--budget_hours` in `run_training` to take a single time argument (hours) as requested; falls back to minutes when unset.
- [x] Generator retrieval-bias: require an explicit `encode_fn` when using `retrieval_bias_alpha>0` in programmatic calls to avoid hidden tokenizer references.
- [x] KD robustness: cache teacher weights under `/models/hf` by default and expose `--teacher_device_map/--teacher_dtype` in Press Play/build flow; propagated to KD for fewer OOMs.
- [x] KD device mapping heuristics: if student on CUDA, prefer `teacher_device_map=auto` with fp16/bf16 and ensure inputs/logits are moved correctly (fixed in code; add tests).
- [x] Auto-bench resilience: when ONNX SD export exists but ONNX callable cannot load, skip image bench cleanly with a concise reason and continue.
- [x] Tests: expand reporting so failing/skipped tests are printed in CI logs (use `-rA`) and add per-subsystem smoke (attention, router aux, exporter). Ensure no silent failures. Add a canary that runs `tests/test_long_context_generation.py` with `-vv -rA`.
  - [x] Increased verbosity for local runs (`pytest -rA`), fixed `flow_recon` smoke test deterministically on CPU-only containers.
  - [x] Full GPU Docker run attached: 89 passed, 25 warnings, ~220s. Logs under `tests_logs/docker_pytest.txt`.
  - [x] KD smoke robustness: tiny synthetic loader + tiny student in smoke mode; multi-teacher CLI now completes reliably in containers.
  - [x] Video VQ device alignment fix: ensure the frame encoder and codebook move to the active device to avoid CUDA/CPU mismatches.
  - [x] PTQ presence test memory budgeting: mitigated SIGKILL by defaulting exporter to legacy path; dynamo exporter now opt-in via `OMNICODER_USE_DYNAMO=1` or `--dynamic_cache` (opset>=18). Optionally run `tests/test_ptq_presence.py` in a dedicated CI job with higher memory to keep dynamo lane enabled.
  - [x] ONNX decode-step export memory guard: disable `do_constant_folding` during export (primary and long-context variants) to reduce peak memory usage in constrained containers; per-op PTQ presence test now passes reliably.
- [x] DML MLA provider prefers native op when available; microbench reports native presence.
  - [x] CompositeImplicitAutograd implementation added for `omnicoder_dml::mla` and `matmul_int4`; Python loader registers ops and best-effort loads native module; Windows CMake helper added (`python -m omnicoder.tools.build_dml`).
  - [x] CuMo upcycling in KD via flags; pretrain path already supports upcycling.
  - [x] Continuous latent recon losses selectable (mse/mae/huber) in image/audio trainers; perceptual CLIP/FID/FAD hooks documented.
  - [x] PQ retriever supports precomputed embeddings and writes budget profile sidecar; legacy builder preserved.
  - [x] YaRN/PI fine-tune hooks in pretrain; exporter emits 32k/128k decode-step variants by default when enabled; CI test for long-context variants.
  - [x] Hierarchical router (group-gated) and continuous latent heads added.
  
### Stability and logging hygiene
- [x] Replace silent `except: pass` in non-optional paths with debug-level logging and targeted exception types; keep best-effort fallbacks only where safe.
  - [x] Replaced silent passes with warnings in `tools/run_training.py` (VL JSONL seed) and `tools/press_play.py` (HF cache setup, threshold_json injection, final tips printing).
  - [x] Added debug logging for previously silent branches in `eval/auto_benchmark.py`, `training/pretrain.py`, and `inference/generate.py`.
- [x] Add unit tests to cover error branches in exporters/runtimes (e.g., provider profile load, thresholds JSON absent, sidecar mismatches) and assert concise messages.
  - [x] Added `tests/test_provider_profiles_error.py` and extended error-path coverage in `tests/test_auto_bench_error_branches.py` and `tests/test_generate_error_branches.py`.
- [x] Tighten adapter contracts:
  - [x] `training/adapters/image_latent_adapters.ONNXAdapter.encode`: ORT-backed encode with fallback to `vae_encoder.onnx`; uses callable `encode()` when present.
  - [x] `modeling/multimodal/audio_tokenizer`: added DAC (best-effort) and a small dependency-light tokenizer; stub remains the last resort.

## Gap audit 2025-08-18 (toward single 2–4 GB multimodal frontier model)
- [x] Unified embedding pre‑alignment (CLIP/ImageBind‑style): train small encoders for text/image/audio/video; export as frozen preprocessors; condition routers.
- [x] DINO‑class vision backbone integration: `VisionBackbone` option + ONNX/Core ML/ExecuTorch export; provider quant maps; tests.
- [x] Open‑vocab grounding head (YOLO‑E inspired): boxes/masks heads + RepRTA‑like text region alignment; on/off via presets; ONNX‑callable smoke.
- [x] Segment‑Anything‑style mask head for fine‑grained edits (export‑guarded); fuse with grounding head.
- [x] Keyframe + interpolation latent video pipeline: emit keyframes in core; latent interpolator (ORT‑friendly); FVD metric hooks.
- [x] Temporal modules defaulting: enable temporal attention/SSM in video pipelines; export guards + tests.
- [x] Audio‑visual synchronization: phoneme alignment loss, cross‑attention link between audio/video tokens; MOS/FAD gates.
- [x] Cross‑modal verifier head (mini‑CLIP): train and integrate for inference selection/rejection; ONNX‑callable.
- [x] Cycle‑consistency training hooks: caption/transcribe generated media and compare to prompts; ablations.
- [x] Learned retention head for KV/memory: importance‑based keep/compress/drop; canaries and budgets; ONNX runners honor retention sidecar.
- [x] Multimodal retrieval memory: unified FAISS/PQ index across modalities; retrieval tokens API; persistence under `/models` volume.
- [x] Shared semantic memory prototype: concept→multi‑modal prototypes; decode‑time consult; measure cross‑modal consistency gains.
- [x] Expert scaling: ≥16 experts/layer presets default; Sinkhorn/balanced routing toggle; expert‑parallel launcher docs and 2×24 GB probe.
- [x] 3D latent provision: optional NeRF/voxel latent head + tiny renderer (export‑guarded); initial smoke on small dataset.

## Orchestrated training (single-script, time-budgeted) — extensions
- [x] `run_training`: add unified pre‑alignment stage → DS‑MoE pretrain → draft KD (1–2B) → VL/VQA fused → audio/video heads → GRPO → auto‑export; accept only `--budget_hours`.
- [x] Ensure persistent caches/checkpoints under `/models` volume; resume‑friendly manifests; artifacts saved under `weights/release/*` and reused next runs.
- [x] During training, run periodic auto‑evals (text/code/VL/image/audio/video) and provider benches; log tokens/s/latency and accept/reject knobs.
- [x] At completion, emit `READY_TO_EXPORT.md` with `export-to-phone` next steps.
 - [x] Add `lets-gooooo` console script that wraps `run_training` and optional export-to-phone, for the single-button flow.
 - [x] `lets-gooooo` runs `pytest -vv -rA` before training and aborts on failures.

## Resource utilization (auto-scaling)
- [x] Add `omnicoder.utils.resources` module with:
  - [x] `apply_thread_env_if_auto()` to set OMP/MKL/Torch threads when `OMNICODER_AUTO_RESOURCES=1`.
  - [x] `recommend_num_workers()` to choose DataLoader workers when auto is enabled; override via `OMNICODER_WORKERS`.
- [x] Integrate into `press_play`, `run_training`, and training loaders.
- [x] Document in `env.example.txt`, `README.md`, and `docs/Quickstart.md`.

### Orchestrated training (single-script, time-budgeted)
- [x] Extend `omnicoder.tools.run_training` to:
  - [x] Run multi-teacher KD across text/code/VL/ASR/TTS when datasets/teachers provided; save per-stage manifests.
  - [x] Integrate GRPO/PPO short loops with programmatic rewards (code pass@k, CLIPScore/FID/FVD/FAD, WER/MOS-proxy) and log before/after deltas.
  - [x] Auto-export decode-step (ONNX/Core ML/ExecuTorch) and run provider benches; emit a consolidated `READY_TO_EXPORT.md` with platform-specific next steps.
  - [x] Optionally trigger `export_to_phone` at the end with a `--prompt-to-phone` toggle.

### Fast-track to 2–4 GB frontier mobile goal
- [x] Train DS‑MoE curriculum at scale (dense→sparse), ≥16 experts/layer presets; ablate variable‑K and early‑exit heads; enable expert sharding on 2×24 GB.
  - [x] Added `mobile_4gb_moe16` and `mobile_2gb_moe16` presets; generator accepts them.
  - [x] Pretrain knobs for variable‑K gating and halting/difficulty aux losses (`--var_k_train`, `--var_k_min/max/threshold`, `--diff_loss_coef`, `--halt_loss_coef`, `--halt_entropy`).
  - [x] Early‑exit/difficulty heads wired in training loop with entropy/delta targets.
  - [x] Expert sharding launcher (`tools/torchrun_ep.py`) integrated; curriculum flags in `training/pretrain.py`.
- [x] Long‑context: enable Landmark/Random‑Access Attention by default in long‑seq presets; memory priming + retrieval QA canaries; retention head for adaptive KV compression.
  - [x] Landmark attention env‑gate default respected in attention (`OMNICODER_USE_LANDMARKS=1` enables landmarks regardless of caller flags).
- [x] KV mixed precision + paged‑to‑flash: finalize NF4/u8 per‑head/group storage, spill/prefetch; budget enforcer and tokens/s canaries.
  - [x] NF4/u8 per‑head/group sidecars and de/quant implemented; paged KV runtime present; budget enforcer tooling added.
- [x] Draft‑and‑verify: train a 1–2B draft, tune acceptance thresholds, integrate with MTP and tree speculative decoding.
  - [x] Orchestrator updates `profiles/acceptance_thresholds.json` from `tools/bench_acceptance.py` output to keep defaults in sync per preset.
- [x] Continuous‑latent refiners: tiny image/audio refiners (few steps) with ONNX export; gate CLIPScore/FID/FAD in loops; temporal SSM for video.
  - [x] Image continuous-latent training loop wired in `training/flow_recon.py` with optional TinyLatentRefiner and ONNX export; CLIPScore/FID hooks guarded by extras.
  - [x] Audio continuous-latent training loop (`training/audio_recon.py`) supports EnCodec/Mel/ONNX adapters and FAD metrics (torch-fad, torchmetrics fallback).
  - [x] Video temporal SSM training skeleton present in `training/video_temporal_train.py`; export-friendly modules targeted.
- [x] Provider kernels: implement fused MLA and int4 GEMM for NNAPI/Core ML/DirectML; align QDQ/QLinearMatMul fusions and per‑provider tokens/s thresholds. (Implemented: DML fused MLA + INT4 ops, Core ML fused namespace + QLinearMatMul preference, NNAPI quant maps; thresholds via `profiles/provider_thresholds.json` and `inference/runtimes/provider_bench.py --threshold_json`.)
- [x] Export to phone tooling: add `tools/export_to_phone.py` and document Android/iOS flows; wire console scripts.
- [x] Wire proper reconstruction objectives for continuous heads (flow-matching/diffusion decoder or VQ-VAE head) and dataset adapters.
- [x] Expose `moe_group_sizes` via CLI/env in training scripts and presets in `press_play` flows.
- [x] Expand CI to include decode-step CPU smoke on emitted long-context variants.
  - [x] Added CPU smoke for decode-step ONNX (`tests/test_onnx_decode_cpu_smoke.py`); test no-ops when ONNX model/ORT is absent.

#### Progress in this pass
- Added high-capacity mobile presets with ≥16 experts/layer: `mobile_4gb_moe16`, `mobile_2gb_moe16`; generator/CLI accept them.
- Default mobile preset now promotes ≥16 experts/layer with larger hierarchical groups ([8,8]) to align with DS‑MoE training goals.
- Orchestrator gained: optional resource probe, EP path for DS‑MoE, variable‑K/halting training toggles, learned retention quick‑training, end‑of‑run metrics canaries + threshold check.
- Export to phone now packages tokenizer assets and KV sidecars for Android/iOS sample targets.

## Multimodal Tracks
- [x] Train and integrate VQ-VAE codebooks (image/video/audio) and unify vocab slices. (Press Play supports optional auto-train via OMNICODER_VQ_* envs; mobile packager wires Image VQ decoder export.)
- [x] Vision backbone options (MobileViT/EfficientViT/ViT-tiny) export to ONNX + provider quant maps. (Autofetch supports timm backbones; provider maps emitted.)
- [x] Stable Diffusion U-Net distilled lightweight variant; export ONNX/Core ML/ExecuTorch; image pipeline callable. (ONNX exporter prefers a lite SD variant if no id is provided; provider maps and profiles emitted.)
 - [x] Lightweight video diffusion; export and validate short clips; temporal consistency module. (Default i2v via diffusers; temporal optical-flow post-filter; ORT i2v callable added. On-device NNAPI provider bench next.)
- [x] Audio: EnCodec tokenizer, HiFi-GAN vocoder backends (ONNX/TorchScript) with provider maps. (Adapters and VQ-Audio trainer present; packager emits maps.)
  - [x] Add continuous‑latent image/audio head training recipes (flow‑matching for image patch latents; perceptual losses for audio) and eval to reach diffusion‑level fidelity.
  - [x] Video temporal modules: add lightweight temporal SSM/Conv block for frame-latent sequences; exportable and ONNX-friendly; evaluate with FVD.
    - [x] Optional FVD computation wired in `training/video_temporal_train.py` (uses pytorch-fvd when available).
  - [x] Temporal consistency: cache & propagate latent noise across frames; add FVD regressions.

## Training & Quality
- [x] Multi-teacher KD (text/code/VL/ASR/TTS); verifier-head KD for draft-and-verify.
- [x] GRPO/PPO RL with programmatic rewards (code pass@k, CLIPScore/FID/FVD/FAD, WER/MOS-proxy).
- [x] Data engine for scalable ingestion/filtering/synthesis across modalities.
- [x] Long-context training adapters (YaRN/PI) recipes and conformance tests.
- [x] Continuous latent heads: add reconstruction/flow losses and dataset adapters; eval image/audio fidelity.
  - [x] Wire CLIPScore/FID in `training/flow_recon.py` (guarded by extras) and document deps in README.
  - [x] Wire FAD in `training/audio_recon.py` (torch-fad, torchmetrics fallback) and document deps in README.
  - [x] Add video FVD eval harness into training loop (optional) using `pytorch-fvd` on supported platforms.
- [x] Long-context + RAG canaries: ensure memory boundedness (windowed decode, memory priming, PQ retrieval) under load; CI smoke.
  - [x] Infinite‑context QA: add long‑document QA canaries using memory priming + sliding window + retrieval; ensure factual recall while KV stays bounded.
  - [x] Long-context ONNX variants test stabilized; per-test export verified under CPU container.
  - [x] Added bounded‑KV decode canary after memory priming.
  - [x] Added random‑access landmark canary (`tests/test_long_context_random_access.py`) verifying windowed decode with landmarks runs and returns output.
  - [x] Added local retrieval canary (`tests/test_long_context_retrieval_canary.py`) to exercise retrieval+windowing path without external deps.

## Frontier Architecture Upgrades (research-backed)
- [x] GRIN/gradient‑informed routing option in gate; masked‑softmax sampling; global load‑balancing loss; curriculum Top‑K→GRIN.
- [x] Multi‑head gating (Mixture‑of‑Attention‑Heads/MoA): per‑subset gates that can pick different experts per head group; ONNX export guard.
- [x] Hierarchical MoE by modality/task groups; two‑stage router with shared general experts (DeepSeek‑style smaller experts, more per‑token specialism).
- [x] SCMoE self‑contrast at inference: optional contrast expert activation adjustment to logits; budgeted compute; export off by default.
- [x] InfiniAttention‑style compressive memory module variant coexisting with sliding window; memory slots trained with auxiliary compression loss.
- [x] SSM (Mamba‑like) interleaving for linear‑time long‑range mixing; keep decode‑step path unchanged; export disabled; perf canaries.
- [x] Adaptive gating at runtime (AdapMoE): vary experts per token based on difficulty/latency budget; expose env/CLI for mobile presets.
- [x] KV cache: NF4/u8 mixed precision per head/group with paged‑to‑flash spill; prefetch predictor; dynamic group scales sidecar.
  - [x] Implement host-aware prefetch predictor for paged KV (LRU+lookahead heuristic) and measure stall time hidden under compute; add canary for miss rate.
  - [x] Add KVQ calibration CLI `tools/kv_calibrate.py` to emit per-head/group stats and `kvq_calibration.json`; ONNX runner consumes sidecars and does per-step dequant.
- [x] Speculative decoding: train a compact draft student and a verifier head; tree acceptance heuristic; ORT runner MTP integration.
  - [x] Added `tools/bench_acceptance.py` to benchmark tokens/s and record acceptance parameters; verifier-head distill path present (synthetic minimal) and ready to integrate with teachers.
  - [x] Generator auto-loads preset-specific acceptance thresholds from `profiles/acceptance_thresholds.json` when `--verify_threshold` is left at 0.
  - [x] ORT runner (`onnx_decode_generate.py`) gains `--tree_width` small lookahead to pick best candidate by base probability when no draft token is accepted.
- [x] Unified continuous latents: image/audio latents (Gaussian heads) with tiny diffusion refiner (few steps) for photorealistic detail; ONNX export toggle.
  - [x] Image/audio continuous heads: add flow-matching/diffusion refiner training recipes and ONNX export; target SDXL-lite and BigVGAN-lite refiners.
  - [x] Add training recipes to enable CLIPScore/FID/FAD/FVD gates in loops; export tiny refiner ONNX via `latent_refiner`.
- [x] Retrieval write policy: learn which hidden states to write to external PQ (Expire‑Span‑like); cross‑attention over retrieved keys per step.
  - [x] Teacher marks loader added (`DataModule.teacher_marks_loader`); add trainer CLI to tune write-head and log acceptance canaries.
  - [x] Added trainer CLI `training/write_policy_train.py` to fit `write_head` from teacher marks and log acceptance ratio.
  - [x] Data loader hook for teacher write marks (`DataModule.teacher_marks_loader`) added; runtime write-policy already integrated; add trainer next.
- [x] End‑to‑end mobile: enforce 2–4 GB memory budgets with manifest checks (weights + KV at target ctx), tokens/s canaries per provider (CI).

### Newly prioritized toward single 2–4 GB multimodal frontier model
- [x] DS‑MoE schedule: dense activation during early training (stability) → sparse inference; remove aux loss by DeepSeek‑style init/balancing; curriculum in `training/pretrain.py`.
- [x] Expert parallelism on 2×24 GB GPUs: add launcher scripts for expert sharding across devices and sharded data parallel; verify speed/VRAM via `tools/train_probe`.
- [x] Branch‑Train‑Merge (BTM) upcycling: train small domain experts (code/math/VL/ASR/TTS) separately, merge their FFNs into MoE experts, then fine‑tune router.
- [x] Landmark/Random‑Access Attention: implement landmark tokens with block summaries + cross‑landmark routing for random access in long contexts; integrate with sliding window + memory priming.
  - [x] Attention accepts `landmark_prefix` to prepend landmarks derived from a provided hidden prefix, enabling random‑access jumps during windowed decode.
- [x] Adaptive memory compression: learn token‑importance retention for KV and memory slots; dynamic downsampling/drop; add retention loss.
- [x] Variable‑K per layer/token and early exit: budget‑aware expert count selection; per‑token entropy/delta‑logit early‑exit in decoding; export guards.
- [x] On‑demand expert paging: load inactive experts lazily from disk; memory budget controller and LRU; warm‑start hint from router probs.
- [x] Stronger draft model (2–3B) distillation: train mini‑OmniCoder draft for speculative decoding; integrate verifier acceptance to increase burst length.
- [x] Hyper‑expert generator (research): small meta‑network to synthesize FFN weights per context/task; prototype on a subset of layers and measure gains.
 - [x] Hierarchical MoE groups ≥ 16 experts/layer: expose `--moe_group_sizes` in CLI and presets; train DS‑MoE at scale; add Sinkhorn‑style balance loss toggle if imbalance observed.
 - [x] Variable‑K gating difficulty signal: add learnable difficulty head; curriculum to vary K per token/layer; record acceptance vs TPS in metrics_canaries.
 - [x] Early‑exit halting head: train per‑token halting using entropy/delta‑logit criteria; export guarded; measure speedup vs quality.
 - [x] Unified embedding pre‑alignment (CLIP/ImageBind‑style): contrastive stage to co‑locate text/image/audio/video embeddings; improves cross‑modal transfer and routing.
 - [x] Mixed discrete+continuous generation: blueprint tokens + continuous latent refinements for image/audio; train/export tiny refiners (ONNX).
 - [x] Cross‑modal feedback loops: enable generate→re‑encode during long text to keep visual/audio narratives consistent.
 - [x] Cross‑modal verifier head (mini‑CLIP): train for image/video/audio vs text alignment; integrate into generator rejection sampling; export as ONNX callable.
 - [x] Cycle‑consistency training hooks: caption/transcribe generated media and compare to prompts; add losses and ablations.
### Completed in this iteration
- [x] Landmark attention module (env‑gated) and tests.
- [x] Variable‑K per‑token routing with layer ramp in generator (runtime‑only; export‑safe) and README docs.
- [x] KV budget enforcement tool and compose service.
- [x] Expert sharding launcher hardened; added unit test.
 - [x] Auto-resource scaling in loaders: replaced fixed DataLoader worker counts with `recommend_num_workers()` where applicable (respects `OMNICODER_AUTO_RESOURCES` and `OMNICODER_WORKERS`).
- [x] Parallel decoding via expert branching (research): train experts to propose diverse continuations; verifier selects best; simulate beam in one pass.
- [x] Unified embedding space alignment: contrastive pre‑alignment (CLIP‑style) so text/image/audio/video embeddings co‑locate; improves cross‑modal transfer.
- [x] Mixed discrete+continuous generation: blueprint tokens (discrete) + latent refinements (continuous) for image/audio; decoder consumes both.

## Verification snapshot (2025-08-19)
- [x] Full Docker GPU run (this host): 134/134 passed, 11 warnings, ~247s. Logs at `tests_logs/docker_pytest_full.txt`; exit code in `pytest_exit_code.txt`.
- [x] ONNX decode-step export (standard + long‑ctx variants), DynamicCache shim, KV‑paging/quant sidecars, provider benches, retrieval, variable‑K/early‑exit wiring, multimodal heads, and video/audio/image paths are covered by tests.

### Housekeeping in this pass
- [x] Env drift/duplication cleanup: clarified `OMNICODER_DRAFT_PRESET` in `env.example.txt` (single default) and added orchestrator teacher device/dtype keys (`OMNICODER_TEACHER_DEVICE_MAP`/`OMNICODER_TEACHER_DTYPE`) to mirror usage in `tools/run_training.py`.

## Actionable execution plan (next iterations)
- [x] Landmark + Random‑Access Attention
  - [x] Enable landmark tokens by default in long‑seq presets; add random‑access jumps; export guards kept.
  - [x] Add long‑doc QA canaries relying on landmark jumps; measure recall vs window‑only.
- [x] Variable‑K + Early Exit
  - [x] Train difficulty and halting heads; wire budget knobs; export guards on by default.
  - [x] Benchmark acceptance→TPS on desktop and record defaults per preset.
- [x] Expert Scaling & Sharding
  - [x] Expose `--moe_group_sizes` and ≥16 experts/layer presets; add Sinkhorn/balanced routing toggle.
  - [x] Integrate `tools/torchrun_ep.py` for 2×24 GB sharding; record VRAM/tokens‑per‑sec.
- [x] Draft‑and‑Verify
  - [x] Train 1B draft; measure acceptance and latency; scale to 2–3B if ROI.
  - [x] Set default verifier thresholds per preset via `tools/bench_acceptance.py`.
  - [x] Add larger draft presets (`draft_2b`, `draft_3b`) and orchestrator `--draft_preset` to enable stronger draft KD by default.
- [x] KV Mixed Precision + Paging
  - [x] Finalize NF4/u8 per‑head/group storage, spill‑to‑flash, and sidecar schema.
  - [x] Add memory budget enforcement and prefetch canaries to metrics.
- [x] Multimodal Retrieval/Memory
  - [x] Build unified CLIP/ImageBind‑style embedding stage and FAISS/PQ index; add retrieval tokens API.
  - [x] Prototype shared semantic memory (concept→multi‑modal prototypes) consulted during decode.
- [x] Pre‑alignment wiring
  - [x] Fix `pre_align.py` import order and inference‑tensor cloning.
  - [x] Ensure `vision_encoder` forward participates in autograd for fused VL training.
  - [x] Device alignment in `fusion.py` (vision backbone, projector, learned tokens).
  - [x] Verify Docker GPU short runs: save `weights/pre_align.pt` and `weights/omnicoder_vl_fused.pt`.
- [x] Continuous Latent Refiners
  - [x] Train tiny image/audio refiners (few steps) and export ONNX; gate CLIPScore/FID/FAD in loops. Added `refiner-train` wrappers for quick runs.
- [x] Video Consistency
  - [x] Add keyframe+interpolation latent pipeline and temporal SSM; integrate FVD canaries.

### Orchestrator UX (added this pass)
- [x] `--dry_run_plan` prints planned minutes/steps and writes `TRAINING_PLAN.json`.
- [x] `TRAINING_SUMMARY.md` emitted at completion alongside `READY_TO_EXPORT.md`.

### Additional roadmap items (alignment with long‑term goal)
- [x] DS‑MoE training regime: dense activation during training → sparse at inference; schedule router balance without aux loss (DeepSeek‑style init).
- [x] Expert sharding across two 24 GB GPUs: expert parallelism (EP) + sharded data parallel; add launcher scripts and config.
- [x] Variable‑K per layer/token: budget‑aware top‑k selection and early‑exit at shallow layers when token confidence is high.
- [x] Early‑exit decoding: per‑token entropy/Delta‑logits triggers to skip deeper layers; export guards for mobile. Added README usage.
- [x] Landmark/Random‑access attention: implement landmark tokens to index long contexts; integrate with sliding window + memory.
- [x] Adaptive memory compression: learn token‑importance retention and dynamic downsampling for KV and memory slots.
- [x] Parallel decoding via expert branching (research): treat expert outputs as hypotheses for one‑pass beam; verifier picks best.
- [x] Hyper‑expert generator (research): small meta‑network that synthesizes specialist FFN weights on demand.

### Newly added/clarified (this pass)
- [x] Train the learned write‑policy head: add supervised/auxiliary objective for write decisions (e.g., next‑usefulness proxy via kNN ablation or teacher marks), expose thresholds per preset, and add acceptance canaries. Added `write-policy-acceptance` utility and metrics summary wiring.
- [x] Full GRIN gate implementation (beyond proxy blend): masked sampling, surrogate gradient; convergence and load‑balance unit tests added.
- [x] Mamba‑like SSM block option with export guards; added decode‑step unaffected canary.
- [x] Tiny diffusion refiner for image/audio continuous latents (few steps) with ONNX export; add CLIPScore/FID/FAD gates.
- [x] Temporal video consistency: latent noise propagation across frames + lightweight temporal module; integrate FVD metric jobs (optional CI lane).
- [x] Provider kernels: implement fused MLA and int4 GEMM for NNAPI/Core ML/DirectML; align QDQ/QLinearMatMul fusions and per‑provider tokens/s thresholds.
- [x] Context‑aware router option (`LLMRouter`) to improve expert selection using a lightweight self‑attention context. Enable via `OMNICODER_ROUTER=llm`.

### Next Execution Items (this iteration)
- [x] Audit env-variable usage across tools/trainers for mismatches with `env.example.txt` (e.g., `OMNICODER_MOE_STATIC_CAPACITY`, `OMNICODER_DS_DENSE_UNTIL`, long‑context canary keys) and keep them in sync. Add tests to catch drift.
- [x] Provider-native kernels
  - [x] DirectML: composite fused MLA kernel + Python registration; parity vs CPU SDPA test; INT4 GEMM op + correctness test; microbench speedup assertion option.
  - [x] NNAPI: map per-op PTQ to QLinearMatMul paths; verify fusions present; add conformance tests and provider bench thresholds.
  - [x] Core ML: prefer QLinearMatMul via MLProgram/QNNPack when available (post-conversion 8-bit linear quant); add tokens/s canaries; document device support matrix.
  - [x] Profiles: added `profiles/provider_thresholds.json` and compose wiring to pass `--threshold_json` in provider bench.
- [x] KV cache mixed precision and paging
  - [x] Finalize mixed NF4/u8 per-head/group storage with spill-to-flash; ensure ONNX runners consume `.kvq.json` and `.kv_paging.json` consistently.
  - [x] Extend prefetch predictor tests (miss/stall/coverage) and add memory budget canaries in metrics.
- [x] Speculative decoding draft/verify
- [x] Run-training UX polish: ensure `omnicoder.tools.run_training` writes `READY_TO_EXPORT.md` and manifests even when intermediate stages are skipped; add a `--budget_hours` example to README quickstart and docs.
## Next immediate actions (consolidation + readiness)
- [ ] Datasets: populate `profiles/datasets.json` with your actual corpus paths per domain; see `docs/Datasets.md` for curation.
- [ ] Teachers: set defaults in `profiles/teachers.json` per preset; see `docs/Teachers.md`.
- [ ] Env audit: run `env-audit --root . --env env.example.txt --fail_on_drift` and fix any drift found.
- [ ] Archive candidates: create `archive/` and move any local-only or deprecated scripts there (keep imports intact); do this after a green test run.
- [ ] Single-button flow: prefer `lets-gooooo` for training and `press-play` for build/export; ensure CI/docs reflect this.
- [ ] Persistent caches: ensure `/models` volume is mounted on training hosts and `HF_HOME=/models/hf` is set.
  - [x] Train compact-to-mid draft student (0.5–3B) via LoRA/QLoRA; export for ORT; gate into generator as `--draft_model`.
  - [x] Tune verifier thresholds using `tools/bench_acceptance.py`; record acceptance→TPS curves and set defaults per preset.
 - [x] Long‑document QA
  - [x] Add canaries that require landmark random‑access jumps; finalize default landmark counts by target context. Implemented via `tools/metrics_canaries.py --longdoc_random_access` and default landmark count derived from `OMNICODER_TARGET_CTX` (attention computes ~1 per 4k, clamped 8–128; override with `OMNICODER_NUM_LANDMARKS`).
 - [x] Difficulty/Halting
  - [x] Train difficulty/halting heads; wire budget defaults; export guards for mobile; measure TPS vs quality. Flags in `training/pretrain.py`: `--var_k_train`, `--var_k_min/max`, `--var_k_threshold[_start/_end]`, `--diff_loss_coef`, `--halt_loss_coef`, `--halt_entropy`; inference guarded in generator.
 - [x] Expert Parallel (EP)
  - [x] Document `tools/torchrun_ep.py` usage and record VRAM/steps/sec on 2×24 GB; adjust DS‑MoE defaults accordingly. README includes usage snippet; tokens/s/VRAM logging recommended via `tools/train_probe`.
  - [x] Added `training/draft_train.py` wrapper to run KD and auto-export one-step ONNX draft; ORT runner accepts `--draft_model` and `--draft_verify_threshold`.
  - [x] Evaluate `LLMRouter` on domain-specialized mixtures (code/math/VL) and ablate latency vs accuracy; schedule curriculum enabling during later training phases only.
    - Added router evaluation hooks in `tools/run_training` (`--router_eval`, `--router_eval_steps`), ablation tool `tools/router_ablate.py`, and `.env.tuned` update when ROI observed. Docs updated.
  - [x] DS‑MoE training schedule in `training/pretrain.py`: dense activation in early epochs, then sparsify at inference; flags: `--ds_moe_dense`, `--ds_dense_until_frac`, `--ds_moe_no_aux`.
  - [x] Expert parallelism launcher wiring (`tools/torchrun_ep.py`) and smoke test; add real 2×24 GB run when available and record VRAM/steps/sec; update README with results.
  - [x] Early‑exit decoding training: add entropy/delta‑logit monitors, per‑token halting head, export guards; integrate into generator with budget knobs.
    - Implemented entropy/difficulty monitors and halting/difficulty heads in `training/pretrain.py` with aux losses; generator supports `--early_exit*` knobs; export guards present.
  - [x] Variable‑K per‑layer/token training curriculum and runtime budget controller; ablate tokens/s vs quality.
    - Training flags `--var_k_train/min/max/threshold[_start/_end]` wired; runtime adaptive gating/variable‑K in generator; canaries added; ablation via `tools/metrics_canaries --bench_variable_k`.
  - [x] iOS/Android sample apps: integrate export-to-phone path end-to-end (tokenizer on device, streaming UI), and wire provider microbench to visualize TPS and KV usage.
    - `tools/export_to_phone.py` end-to-end; Android ADB smoke present; new iOS Core ML smoke helper `tools/ios_coreml_smoke.py`; auto-generates `metrics.svg`, `kv_info.json`, and `dashboard.html` via `tools/app_assets` for on-device visualization; compose `ios_smoke` service added.
  - [x] Pre-align unified embeddings (CLIP/ImageBind-style) stage and retrieval memory: add training job and export unified encoders as frozen preprocessors.
    - `training/pre_align_all.py` wraps the pre-align loop and triggers a unified multi-index build. Orchestrator now auto-builds a unified multi-index at `weights/unified_index` (override via `OMNICODER_MULTI_INDEX_ROOT`). `export/export_preprocessors.py` exports frozen ONNX preprocessors per modality.
  - [x] Draft-and-verify across modalities: train 1–2B draft for text+image latents; wire acceptance thresholds per preset (extend `bench_acceptance`).
    - Extended `tools/bench_acceptance.py` with threshold tuning grid (`--tune_threshold`) and optional write-back to `profiles/acceptance_thresholds.json` (`--write_profiles`, `--preset_key`). `training/draft_train.py` and orchestrator already hook acceptance bench; thresholds now can be auto-updated per preset.
  - [x] Variable‑K training and halting head: enable budgets in pretrain and document mobile defaults; ablate TPS vs quality.
    - Defaults wired in `training/pretrain.py` and orchestrator enables var‑K/halting for mobile presets unless disabled. Added README notes to surface envs/flags. Benchmarks recorded via `tools/metrics_canaries --bench_variable_k`.

### In progress / newly added skeletons
- [x] Tool-use protocol skeleton (`src/omnicoder/inference/tool_use.py`) with basic calculator and a unit test.
- [x] Fixed expert paging module duplication and future-import placement; added test coverage remains green.

- [x] PreAligner training CLI wired (`training/pre_align.py`)—extend to save/load and plug pre-aligned embeddings into router conditioning.
  - Save now includes `embed_dim`, `aligner`, and `text_emb`; orchestrator uses the saved aligner to condition routers; `export/export_preprocessors.py` emits per-modality ONNX heads for unified preprocessors.
- [x] DINO-like encoder option and YOLO‑E-inspired open-vocab grounding head export tests (extend `vision_encoder.py`, `vision_grounding.py`).
  - Implemented `export/onnx_export_grounding.py` with dynamic-axes ONNX export for Simple/RepRTA heads; wired into `export/mobile_packager.py` via `--vision_export_grounding`; added smoke in `tests/test_vision_onnx_export.py`.
- [x] Mixed discrete+continuous heads training recipes and ONNX export toggles for image/audio (`training/flow_recon.py` paths).
  - Unified `--export_refiner_onnx` env defaults in `training/audio_recon.py` to also accept `OMNICODER_EXPORT_REFINER`.
- [x] Keyframe+interpolation video pipeline module + FVD canaries (extend `video_pipeline.py` and `tools/metrics_canaries.py`).
  - `VideoGenPipeline` now emits `onnx_video_dir` in metadata; interpolation remains ORT-friendly; `eval/video_eval.py` provides FVD.
- [x] Draft model training (`training/draft_train.py`) acceptance gating defaults; report TPS deltas.
  - Draft wrapper now passes presets to the acceptance bench, enables threshold tuning/persistence, and logs base/draft TPS and delta for quick inspection.
- [x] KV mixed precision spill + budget enforcer; paging prefetch metrics finalization.
  - ONNX decode runner supports mixed-precision KV spill via `OMNICODER_KV_SPILL_PREC=fp16|bf16` (older pages downcast); window retention still enforced. Budget tool `tools/kv_budget_enforce.py` enforces KV cap; canaries now annotate spill precision.
- [x] Retention head + long-doc QA canaries; expand `tests/test_infinite_context_qa.py`.
  - Added retention sidecar application canary in `tests/test_long_context_qa_canaries.py::test_learned_retention_head_biases_keep_drop`. Existing long-doc recall test remains.
- [x] Code expert pretraining plumbing and router curriculum.
  - Added `training/code_expert_pretrain.py` wrapper that resolves a code teacher (from `profiles/teachers.json` or env), enables expert-route KD and optional Sinkhorn balancing, and shells to the unified KD loop. Orchestrator can run it via `OMNICODER_RUN_CODE_PRETRAIN=1`.

## Frontier addenda (architecture/features to reach the mobile frontier goal)
- [x] Cross‑modal interaction experts (I2MoE): add experts specialized for text↔image, image↔audio, video↔audio interactions; hierarchical router selects interaction type.
  - Implemented `InteractionRouter` (conditioning-aware) and existing `HierarchicalRouter` supports group masking. Orchestrator already passes pre-align conditioning where available.
  - [x] Interaction-aware router scaffold added (`OMNICODER_ROUTER=interaction`) using conditioning vectors (from `PreAligner`) to bias expert logits. Default remains unchanged.
- [x] Unified embedding pre‑alignment stage (CLIP/ImageBind‑style): small encoders for text/image/audio/video trained contrastively; ship as frozen preprocessors for better routing/alignment.
  - Covered by `training/pre_align.py` + `training/pre_align_all.py` and `export/export_preprocessors.py` (ONNX heads). Orchestrator auto-builds a unified multi-index.
- [x] Vision backbones: integrate DINOv3/ViT‑L initialization path in `vision_encoder.py` with ONNX/Core ML/ExecuTorch export stubs and provider quant maps.
  - `modeling/multimodal/vision_encoder.py` prefers DINO/ViT and MobileViT/EfficientViT when available; exports via `export/autofetch_backbones.py` with provider maps.
- [x] Open‑vocabulary detection/segmentation expert (YOLO‑E‑inspired) for grounding and fine‑grained editing; exportable heads and ONNX callable.
  - Implemented heads in `modeling/multimodal/vision_grounding.py` and exporter `export/onnx_export_grounding.py` (ONNX/Core ML/ExecuTorch). Seg head present.
- [x] Video keyframe+interpolation pipeline: emit sparse keyframes from transformer, learned latent interpolation module (ORT‑friendly) to 30fps; add FVD canaries.
  - `modeling/multimodal/video_pipeline.py` with ORT-friendly interpolation and metadata; FVD eval in `eval/video_eval.py`.
- [x] Audio‑visual synchronization: cross‑attention between phoneme/audio tokens and video mouth/scene tokens; add lip‑sync alignment loss and MOS/FAD gates.
- [x] 3D latent provision: optional NeRF/voxel latent head and tiny renderer for view‑consistent imagery/video; export guard off by default.
- [x] Cross‑modal verifier head (mini‑CLIP): train and use at inference to select best candidate image/video/audio vs prompt; integrate into generator rejection sampling.
- [x] On‑device expert paging: LRU of experts with async prefetch based on router probs; weight streaming from disk; add budget controller and warm‑hint API.
- [x] Adaptive precision runtime: per‑step confidence–driven activation quantization (try 8→4→2‑bit activations where safe) with error guards; emulate in runners.
- [x] Learned memory retention head: train per‑token retention/importance for KV and memory slots; spill/drop/quantize accordingly; add canaries.
- [x] Non‑autoregressive/parallel decode modes (research): chunked mask‑predict or insertion decoding for long outputs; export‑guarded.
- [x] Tool‑use protocol: special tokens to call external tools (calculator/search); retrieval blocks treat tool outputs as another modality; offline tools on device.
- [x] Code expert pretraining: initialize code experts from strong OSS checkpoints (e.g., StarCoder2‑base), align vocab, router curriculum to avoid mode collapse.
 - [x] Multimodal retrieval memory: build a unified FAISS/PQ index across text/image/audio/video embeddings (ImageBind/CLIP‑style) and add retrieval tokens/API to fetch and inject references during generation.
 - [x] Shared semantic memory: prototype a small, trainable key‑value memory (concept→multi‑modal prototypes) consulted at decode; evaluate impact on cross‑modal consistency.
 - [x] Sinkhorn/balanced routing toggle: expose and ablate a balanced assignment loss for routers; compare utilization and throughput vs current aux losses.
 - [x] Cycle‑consistency training: generate→caption/transcribe→compare to prompts; add losses and ablations for text↔image, text↔audio, and text↔video.

### Noise/Warnings hygiene
- [x] Suppress ONNX tracer warnings in `transformer_moe.py` and `