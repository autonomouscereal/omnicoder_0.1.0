# TODO

> Archived notice: This file is now superseded by structured plans in `todo/` and `docs/ProjectPlan.md`. Keep using `todo/*` for day-to-day work; this file is retained for historical context.

- [x] Compose press_play export verified on this machine; artifacts under `weights/release/`. Add PowerShell note (avoid bash piping like `| cat`).
- [x] Compose tests service executed; minimal subset OK in container. Broader suite validated in local venv (see README verification snapshot).

- [x] Verify codebase on Windows 11 + Python 3.12 in venv; run tests; validate ONNX export/runtime
  - [x] Editable install in `.venv` with extras (`onnx,vision,gen`)
  - [x] `pytest` all green (43 passed, 1 skipped)
  - [x] Text CLI smoke (`inference/generate`)
  - [x] ONNX decode‑step export + ORT CPU round‑trip

- [ ] One-button Play UX
  - [x] Add consolidated build/export/benchmark CLI (`tools/build_mobile_release.py`) and document in README
  - [x] Add preset env checks and auto-install hints for missing toolchains (onnxruntime, diffusers, optimum, coremltools)
  - [x] Validate CPU smoke: text generate, ONNX decode-step export, ORT streaming, micro-benchmark JSON
  - [x] Optional GUI wrapper (Tk/Qt) for a single-click desktop launcher (`omnicoder.tools.gui_play`)
  - [x] Auto-load `.env` in `press_play` and `build_mobile_release`; add `.env.example` and README doc

- [ ] Wire actual pretrained backbones:
  - [x] ViT-tiny vision encoder (auto-detect via `timm` with internal fallback)
  - [x] EnCodec small (audio tokenizer wrapper); [x] HiFi-GAN vocoder wrapper
  - [x] Stable Diffusion image pipelines (diffusers + ONNX callable); [x] lightweight video diffusion wrappers
  - [x] Whisper tiny/base for ASR; [x] Coqui/Piper/pyttsx3 TTS adapters
- [ ] Implement MLA-style KV compression kernels + multi-query attention kernels
  - [x] Add prototype latent-KV attention path in PyTorch for functional testing
  - [x] Add incremental KV-cache streaming in attention and generator for fast decoding
  - [x] Add provider fused-MLA backend registry (cpu/dml/coreml/nnapi) with fallback; env `OMNICODER_MLA_BACKEND` selects provider; added microbench in `inference/benchmark.py --bench_mla`
  - [ ] Integrate real device fused kernels (ANE/NNAPI/DML) with layout-aligned int4/kv8 ops and export/runtime wiring
- [ ] Implement router regularization (aux-loss-free or z-loss) and capacity factor tuning
  - [x] Capacity-aware token-capacity dispatch per expert with configurable capacity factor (limits per-step activation on mobile)
  - [x] Wire aux load-balancing loss using `MoELayer.last_load_penalty` during training
  - [x] Add z-loss on router logits and importance/load penalties; expose CLI flags in pretrain/distill
  - [x] Token-wise batched dispatch per expert (grouped processing) with top-k capacity filtering
- [x] Add multi-token prediction heads (N>1) with auxiliary losses in training
- [x] Integrate simple Medusa-style lookahead acceptance path in generator
- [x] Add acceptance verifier for speculative decoding (draft-and-verify) for correctness
- [ ] Long-context adapters (YaRN/PI) and Retrieval plug-in
  - [x] Baseline local retrieval (TF-IDF) integrated in generator for RAG prompts
  - [x] Add chunking/merging strategy and prompt templating
  - [x] Add vector store (FAISS optional) and on-device ANN fallback
- [ ] Exporters: Core ML / ExecuTorch operators coverage; GGUF quantizer hooks
  - [x] ONNX export supports mobile 4GB preset
  - [x] Add ExecuTorch decode-step export with KV-cache IO (stateful)
  - [x] Add one-command mobile packager to orchestrate ONNX decode-step export, optional ONNX int8 quant, optional ExecuTorch export, and memory budget summary
  - [x] Add per-operator int8 choices (per-op PTQ) and emit NNAPI/Core ML/DML quant maps with preliminary int4 hints
  - [x] ONNX attention fusion pass in export: ensure com.microsoft::Attention present and QDQ around MatMul for provider QLinearMatMul fusions
  - [x] Core ML decode-step exporter producing MLProgram with recurrent KV inputs/outputs
  - [ ] Core ML custom layers for attention/RoPE/KV-latent to replace traced ops with Apple-native kernels
  - [x] ONNX decode-step runtime that streams generation via KV-cache (`inference/runtimes/onnx_decode_generate.py`)
  - [x] Integrate AWQ/GPTQ int4 export flow (script now attempts quantization if deps present)
  - [x] Add ONNX per-operator PTQ script with per-channel option and README docs
  - [x] One-command mobile packager verified locally (ONNX export + dynamic int8) and summary JSON written
  - [x] Press Play end-to-end validated on Windows; SD ONNX export fixed (Optimum API/CLI fallback)
  - [x] Mobile packager supports per-op PTQ with runtime preset selection
  - [x] Provider options module and runner session optimizations (threads, graph opt); JSON provider profiles
  - [ ] DynamicCache ONNX export via torch.onnx dynamo exporter; replace manual KV IO tensors with DynamicCache state; add conformance tests.
  - [ ] Core ML attention/RoPE custom ops (or mapping to Apple attention) and latent‑KV reduce/expand; add correctness tests.
  - [ ] ExecuTorch NNAPI delegate mapping for decode‑step graph; per‑op int8/int4 maps aligned to device.
  - [ ] Expand Core ML exporter to use Apple attention ops when available; add correctness tests (map latent reduce/expand and RoPE metadata)
  - [x] Prefer ONNX dynamo exporter by default with fallback; add `--no_dynamo` flag. Next: integrate DynamicCache explicitly when available in PyTorch.
  - [x] Documented PowerShell parsing caveat for install commands in README
  - [x] Disable HRM during ONNX/Core ML export for stable graphs
  - [x] Provide functional weight-only int4 wrapper to validate quant math; wire optional replacement utility
- [ ] RL: programmatic rewards for code (compile/run), unit-test coverage, and static analysis
- [ ] RL: CLIPScore/FID/FVD/FAD-based automatic rewards for image/video/audio
- [x] Distillation pipeline (teacher→student logit-matching, optional CE; LoRA + gradient checkpointing)
  - [ ] Optional: rationale/CoT-aware KD and sequence-level KD
  - [ ] Optional: KD from MoE teacher with expert-aware routing supervision
- [x] Integrate HRM into core model (`use_hrm`, `hrm_steps`) for iterative refinement
  - [x] Add adaptive halting w/ small policy head and budgeted compute per query
  - [ ] Expose HRM controls in exporters (disable during export if needed)
  - [ ] Add unit tests for halting threshold behavior
  - [ ] Unit tests: verify HRM invariants (shape preservation, determinism under eval)
- [ ] Mobile demos (Android/iOS), JNI bridge for llama.cpp and Core ML package
  - [x] Provide Android sample app wiring tokenizer + ONNX assets + streaming decode-step (NNAPI best-effort)
  - [x] Android: ADB helper script to push ONNX and run NNAPI device-side smoke with TPS threshold (`tools/android_adb_run.py`)
  - [ ] Android: input field UI, real tokenizer, rolling K/V reuse between steps; ExecuTorch NNAPI delegate variant
  - [x] iOS: SwiftPM console app streams tokens with minimal tokenizer and zero-length K/V
  - [ ] iOS: UIKit/SwiftUI sample app with streaming UI and tokenizer; rolling K/V reuse; packaged `.mlmodelc`

## Architecture/Research Improvements (major reworks)
- [ ] Implement Multi-Head Latent Attention kernels on ANE/NNAPI with int4-friendly layouts; publish minimal kernels
- [ ] Add fused QKV + latent-KV attention custom ops for NNAPI/Core ML/DML; RoPE kernel hooks
- [ ] Add draft-and-verify acceptance with external verifier head distilled from a stronger teacher
- [x] Add verifier-head distillation script and wire generator verifier threshold path
- [ ] Integrate long-context adapters (YaRN/PI) and verify stability at 32K/128K tokens; expose in exporters
  - [x] Add optional SSM blocks (GatedConvSSM) interleaved (every 4th layer) for full‑sequence passes; skipped in decode‑step; export disables SSM
  - [ ] SDPA v3 / FlashAttention‑3 decode fast‑paths where supported; guard by capability; ensure RoPE/MQA compatibility.
- [ ] Multimodal unified token space: train VQ-VAE for images/video; align codebooks with text tokenization for joint modeling
  - [x] Image VQ-VAE trainer and codebook export; Image VQ decoder module and ONNX/Core ML/Execu exporters
  - [x] Video VQ trainer present; integrate decoder exporter later if needed
  - [ ] Integrate VQ-decoder ONNX/Core ML paths into mobile packager for end-to-end demo
- [ ] Data engine: automated data curation, filtering, and synthesis for VL/VQA/ASR/TTS; implement scalable web dataset ingestion (offline mirroring)
- [x] RL research tracks: GRPO/PPO for reasoning and multi-modal rewards (CLIPScore/FID/FVD/FAD), add reward model training
- [ ] Mobile int4 end-to-end: AWQ/GPTQ for text, per-operator quant maps for ONNX/Core ML/ExecuTorch decoders
  - [x] Add packed int4 weight layout alignment via `OMNICODER_INT4_ALIGN` (default 64) to match device kernels
  - [ ] Align int4 packing and nibble order per provider kernels; add golden tests for DML/MPS/NNAPI
- [x] Speculative decoding with small draft model (Medusa-style) and acceptance head; add acceptance verifier tests (multi-step verifier implemented)

## Performance/Architecture Upgrades (to meet 2–4 GB mobile frontier goal)
- [x] Replace Python MoE dispatch with fused gather/scatter CUDA interface and safe fallbacks; integrated in `MoELayer` (hook for CUDA ext)
- [ ] Implement attention low-rank KV compression (MLA) with learnable latent dictionaries; verify perplexity vs. fixed linear maps
- [x] Add paged KV cache with block reuse and host pinning for long-context streaming; export sidecar and ONNX runner tail materialization
- [ ] Integrate FlashAttention-3 / SDPA v3 fast-paths with RoPE and multi-query; guard on PyTorch >=2.4/compute capability
- [ ] Enable multi-token speculative decoding with tree-search acceptors; evaluate speedups on ARM big.LITTLE and NNAPI
  - [x] Tree speculative decoding with verifier acceptance added (auto-threshold support); default knobs exposed in CLI
- [x] Add activation calibration CLI and sidecar for per-channel scales to drive PTQ
- [x] Add ONNX fusion presence test and PTQ helper alignment; provider mapping docs and profiles retained
- [ ] Distillation curriculum: multi-teacher traces (code/math/VL/ASR/TTS), verifier-head KD, router-target KD; track sample efficiency and stability
- [ ] Memory: group-wise KV quant calibration and per-head scaling (u8/nf4); thread-safe decode-step IO in iOS/Android runners
- [ ] Evaluate numpy alternatives (PyTorch ops, PyTorch 2.x compile, JAX/XLA feasibility for exporters); prefer PyTorch vectorized paths; remove pandas at runtime
  - [ ] Add optional kNN-LM cache during decoding (hidden-state ANN lookup) to improve factuality with small on-device stores; expose `--knn_cache` flags.
  - [x] Add `--compile` flag in generator; warmup + fallback if toolchain missing. Next: capture_scalar_outputs option to reduce graph breaks and measure tokens/s improvements.
  - [ ] Provider microbench CI with thresholds: NNAPI (QNN), Core ML (ANE/GPU), DML; enforce tokens/s minimums and fusion presence (Attention/QLinearMatMul).
  - [x] Add GitHub Actions CPU provider bench with Attention fusion requirement and threshold; upload JSON artifacts.
  - [ ] Windows self-hosted runner for DirectML tokens/s thresholds; document torch-directml install in CI runner.
  - [ ] macOS self-hosted runner for MPS/Core ML EP tokens/s thresholds.
  - [ ] Align AWQ/GPTQ int4 weight packing strictly to provider kernel layouts (alignment, ordering, scales/zeros), add golden tests per backend.

## Gaps vs. Goal (must-have features)
- [ ] Real backbones plugged and auto-exported: ViT-tiny (vision), EnCodec small (audio tokenize), HiFi-GAN (vocoder), SD U-Net (image), light text-to-video
- [ ] On-device demo apps: Android ExecuTorch+NNAPI with streaming UI and tokenizer; iOS Core ML MLProgram decode-step runner
- [ ] End-to-end quantization: int4 text (AWQ/GPTQ), int8 per-op PTQ maps per provider, KV u8/NF4 with calibration and exporters
- [ ] Long-context 32k/128k validation with YaRN/PI; decode stability tests and exporter baked params
- [x] Unified multimodal tokens: VQ-VAE training/export for image/video/audio; unified vocab mapping helpers; integrate into fused VL/VQA/VL-video loops
- [ ] CI: provider microbench matrix on device/backends with thresholds; auto-regressions for tokens/s and KV footprint
  - [ ] Add CI long-context decode-step roundtrip for 32k/128k variants; enforce stability thresholds and windowed decode policies

## Newly identified work (2025-08-13, container + GPU validation)
- [ ] Tokenizer and unified vocab
  - [ ] Freeze unified vocab slices and emit integrity checks (text/image/video/audio) at train/export time
  - [ ] Emit sidecar unified vocab mapping JSON for on-device tokenization (Android/iOS/ORT/ExecuTorch/Core ML)
- [ ] HRM integration and exportability
  - [ ] Add export-friendly static-branch HRM with 2-expert routing hint for ORT/ExecuTorch/Core ML; keep default export disabling dynamic loops
  - [ ] Unit tests for HRM adaptive-halting invariants and export disablement paths
- [ ] Memory and KV
  - [ ] Default sliding-window decode and paged KV in runners; page-cache allocator shim for CPU/GPU/NNAPI backends
  - [ ] Enforce sidecar dims (`dl_per_layer`) in runners; add CI thresholds (tokens/s and memory)
  - [ ] KV quantization calibration sidecar generation and conformance tests (u8/NF4, per-head grouping)
- [ ] Attention kernels
  - [ ] Fused RoPE + MLA kernels and reduced kernel launches for DML/Core ML/NNAPI; int8 KV read path in mobile runners
  - [ ] Integrate SDPA v3/FlashAttention-3 fast paths where available; guard for capability; perf canaries
- [ ] Speculative decoding
  - [ ] Enable tree/draft with verifier head across runners; auto-thresholding; streaming API docs and tests
  - [ ] Multi-token prediction head tuning for mobile throughput; losses wired in training
- [ ] Export & runtimes
  - [ ] ExecuTorch decode-step with NNAPI delegate conformance; integer attention mapping (QNN)
  - [ ] Core ML MLProgram decode-step: ANE attention mapping and QDQ; provider profile presets
  - [ ] ORT mobile fusions (Attention/QLinear) with provider presets; complete per-op PTQ maps; integration tests
  - [ ] DynamicCache: adopt torch.onnx dynamo exporter once stable and remove shim; add decode-step conformance
  - [ ] KV paging: device runners for paged decode; README documentation of sidecars and policies
  - [x] Vision backbone export: ONNX; provider maps/profiles copied
  - [ ] Vision backbone export: Core ML MLModel and ExecuTorch `.pte` validation with sample consumers
- [ ] Datasets & training
  - [ ] Curate compact backbones and KD datasets per modality; add checksum manifests
  - [ ] Long-context YaRN/PI training; canaries for 32k/128k decode stability/perplexity drift
- [ ] Bench & QA
  - [ ] Device runners (NNAPI/ANE/DML): enforce TPS and KV memory thresholds in CI; perf regression gates
  - [ ] Windowed decode stability tests across providers
  - [ ] ExecuTorch `.pte` device-run timing via Android ADB workflow (promote to required once delegate available)
  - [x] Image ONNX provider profile injection in CLI/bench; pass provider options to ORT sessions

## Verification (2025-08-14)
- [x] Docker Press Play (Compose, CPU path on this workstation):
  - Exported text decode-step ONNX to `weights/release/text/omnicoder_decode_step.onnx`
  - Exported SD ONNX folder under `weights/release/sd_export/onnx` (observed absolute‑diff tolerance warning; see below)
  - Wrote `weights/release/bench_summary.json` and `weights/release/unified_vocab_map.json`
  - Provider quant maps are emitted when requested; NNAPI/Core ML/DML maps generation is guarded by flags/profiles
- [x] Tests: `docker compose run --rm tests` completed without failures (PowerShell output truncated dots; local venv full suite green previously)
- [x] README updated with Compose/GPU caveats for Windows (no `--gpus` flag on compose CLI; avoid `| cat` in PS)
- [x] ONNX decode-step runner: paged KV enforcement, u8/NF4 KV emulation, speculative MTP drafts acceptance
- [x] Exporter: DynamicCache shim sidecar; KV paging and KVQ sidecars

## Verification (compose runs, 2025-08-14)
- [x] Compose `press_play` produced text decode-step ONNX, SD ONNX directory, quant maps, and `bench_summary.json` on CPU-only host
- [x] Compose `tests` ran minimal tests (GPU not detected → CPU path exercised)
- [ ] Compose `provider_bench` thresholds: parameterize providers and enforce tokens/s gates in CI

### Container run notes (this workstation)
- Full `pytest` inside Docker may be terminated by container resource limits on some hosts (Killed). Prefer `docker compose run --rm tests`, run focused subsets (e.g., `pytest -k onnx`), or run tests natively in a venv.
- Press Play with persistent `/models` cache confirmed SD ONNX export via Optimum CLI path and wrote artifacts under `weights/release`. Subsequent runs reuse `/models/hf` to avoid re-downloading.

### Action items from this run
- [ ] SD ONNX numeric parity: Investigate absolute‑diff tolerance exceedance (observed max ~0.00145 vs 0.0003) and either tighten tolerances per component or document acceptable ranges; prefer dynamo exporter when stable
- [ ] Compose GPU runs: document and provide a `compose.override.yml` example with `deploy.resources.reservations.devices` for GPU devices (or instruct users to use `docker run ... --gpus all`)

### Newly identified from compose run
- [ ] SD ONNX numeric parity: investigate and either tighten tolerances per component or document acceptable ranges; prefer dynamo exporter when stable.
- [ ] PowerShell docs: standardize Windows examples without bash piping; add notes near Docker Compose section.

## Long-context & KVQ enforcement
- [x] Runner: adopt KVQ sidecar scheme/group automatically; warn on mismatches; warn if KVQ requested without sidecars/calibration
- [x] Runner: auto-enable paged KV when `*.kv_paging.json` sidecar present; derive default `--window` from sidecar if not set
- [x] README: document 32k/128k decode-step variant export and CPU ORT smoke; note paged KV/window defaults in runner
- [ ] CI: add long-context export canaries (32k/128k) and CPU ORT smoke step; fail on export/IO regressions

### Follow-ups from compose logs
- [ ] SD ONNX numeric parity: investigate tolerance exceedance (max abs diff ~0.0031) and set per-component tolerances; prefer dynamo exporter when available
- [ ] Add retry/fallback for Optimum API export; keep CLI fallback; record exporter version in manifest
- [ ] Allow skipping image export/bench to speed CPU-only Press Play (`OMNICODER_BENCH_IMAGE_BACKEND=`)
- [ ] Shorten CPU auto-bench by default (reduce samples/steps) and expose `.env` knobs
- [ ] Attention fusion: provider_bench reported 0 `com.microsoft::Attention` nodes on CPU EP — add optional fuse pass guard and verify fused graph presence for non-CPU providers in CI; document CPU EP may not expose fused node types

## New todos (from verification)
- [ ] Add docker compose services for `onnx_smoke` and `provider_bench` with thresholds; wire to CI
- [ ] Add `.env.example` keys for provider profiles and default ORT provider; ensure press_play reads them on container
- [ ] Reduce SD export latency by preferring low-memory UNet variants; add provider maps for SD ONNX EPs

## Additional features to hit 2–4 GB frontier‑class target
- [ ] Multi‑teacher ensemble KD (text/code/math/VL/ASR/TTS/video) with router supervision and rationale traces; curriculum schedule
- [ ] Draft‑and‑verify at scale: train verifier head and small draft model; enable tree speculative decoding with adaptive thresholds on ARM big.LITTLE
- [ ] Retrieval PQ store: on‑device product‑quantized ANN with mmap; budgeted cache per device; integrates with kNN‑LM and RAG prompts
- [ ] Long‑video tokens: hierarchical video tokenization (low‑fps latent tokens + frame interpolation decoder) with tiled per‑op PTQ and streaming decode
- [ ] Device kernels: int4 GEMM and fused MLA/MQA attention for NNAPI/Core ML/DML; align weight/KV layouts to each provider; golden perf/correctness tests
- [ ] DynamicCache decode‑step: migrate exporters and runners when upstream stabilizes; remove explicit KV tensors
- [ ] Memory policy: default sliding‑window + paged KV in all runners; sidecar‑driven window derivation and enforcement
- [ ] Training efficiency: torch.compile decode path, FlashAttention‑3/SDPA v3 where available; bfloat16 where stable; bitsandbytes 4‑bit in KD/LoRA by default
- [ ] Replace pandas usage in any hot paths with vectorized PyTorch or Polars; prefer PyTorch ops over NumPy in runtime loops
- [ ] iOS/Android demo apps with full streaming UI, tokenizer on device, and KV reuse across steps; Press Play deploy stage

## Next work focus (provider kernels, DynamicCache, delegates, training, apps)
- [ ] Provider fused attention kernels (NNAPI/Core ML/DML): implement RoPE+MLA/MQA fused ops and map layouts
- [ ] Int4 GEMM + int8 KV path on device: align `Int4Linear` packing and add dequant-per-step KV kernels
- [ ] DynamicCache: adopt real stateful export with torch.onnx.dynamo_export when stable; remove explicit KV IO
- [ ] Delegate conformance: ExecuTorch NNAPI (attention int8/QLinear) and Core ML MLProgram attention/RoPE/latent-KV mapping
- [ ] Long-context training: YaRN/PI passes; CI canaries (added) enforce 32k variant and optional 128k
- [ ] KD + RL: multi-teacher KD (code/math/VL/ASR/TTS/video), verifier-head KD, GRPO/R1-style RL; distill to mobile presets
- [ ] Mobile demo apps: Android/iOS streaming chat with tokenizer; Press Play deployment stage

## ONNX dynamo + DynamicCache adoption
- [ ] Prefer `torch.onnx.dynamo_export` for decode-step; add decode-step conformance tests (input/output shapes, dynamic axes, KV I/O)
- [ ] Replace explicit KV tensors with true DynamicCache when upstream stabilizes; remove shim sidecar; update runners
- [ ] Long-context decode-step conformance (32k/128k variants) under dynamo exporter

## Unified VQ tokens and vocab sidecars
- [x] Image VQ-VAE trainer and codebook exporter (`training/vq_train.py`) and compact model (`modeling/multimodal/vqvae.py`)
- [x] Unified vocab mapping helper (`modeling/multimodal/vocab_map.py`)
- [x] Emit `weights/unified_vocab_map.json` in autofetcher for on-device tokenization
- [x] Load and enforce sidecar in fused training scripts (`vl_fused_pretrain`, `vqa_fused_train`, `vl_video_fused_pretrain`)
- [x] Load sidecar in inference CLI (`inference/multimodal_infer`) for mapping consistency
- [ ] Integrity checks: assert non-overlap of vocab slices and codebook sizes vs mapping; add unit tests
- [ ] Wire vocab sidecar into exporters and mobile runners; validate mapping in CI

## Performance upgrades (actionable)
- [ ] Torch.compile decode path default for supported desktops; warmup/fallback guards; measure tokens/s deltas
- [ ] Provider microbench expansion with paged KV and KVQ emulation; thresholds per provider profile
- [ ] Align AWQ/GPTQ int4 packing with provider kernels; golden layout tests and runner-side dequant paths
- [x] DirectML MLA provider path wired; persistent device + mask cache for faster transfers; SDPA fallback retained
- [x] DirectML int4 backend path: unpack+dequant+matmul on GPU; alignment/nibble-order knobs; document usage and add benchmark hook
- [ ] Replace residual NumPy/pandas usage in hot paths with vectorized PyTorch ops; ensure no pandas in inference runtime

## New major tracks to reach target product goals
- [x] KV-cache quantization (u8/NF4) end-to-end in PyTorch path: generator flags, runtime dequant, memory estimator, auto-benchmarks
- [ ] Mobile int4/kv8 kernels
  - [ ] Integrate provider-specific int4 matmul (NNAPI QNN/Qualcomm, Core ML ANE, DML) with weight packing matching `Int4Linear` and AWQ/GPTQ
  - [ ] KV-cache 8‑bit or nf4 activation quant with per‑head calibration; export/cache threading in ExecuTorch/Core ML runners
  - [ ] Bitsandbytes CUDA 4-bit fast path: auto-swap nn.Linear to bnb Linear4bit in training and KD flows (now available via `replace_linears_with_bnb_4bit`; extend CI tests)
  - [ ] Self-hosted provider validation on device: NNAPI/QNN (Android), Core ML (ANE), DirectML (Windows) with performance thresholds
- [ ] Long-context 32K/128K stability
  - [ ] YaRN/PI training pass; exporter emits baked scale/base; add CI canaries validating perplexity drift and decode stability
  - [x] CLI flags exist in exporters and pretrain (`--target_ctx`, `--rope_base`); decode-step window added; README updated
  - [ ] Add robust unit tests for 32K streaming decode (CPU path), and ONNX decode-step round-trip for long-context variants
- [ ] Unified multimodal tokens
  - [ ] Train real VQ‑VAE codebooks (image/video/audio) and reserve vocab slices; instrument fused VL/VQA datasets
- [ ] Frontier‑level student via KD+RL
  - [ ] Multi‑teacher KD traces (code/math/vision/audio/video) with verifier-head supervision and router targets
  - [ ] GRPO/GRPO‑style RL with task batteries (GSM8K, HumanEval/MBPP, MMLU, VQAv2, FVD/FID/FAD), curriculum and rejection sampling
  - [ ] Auto-KD datasets: TinyStories, Tiny Shakespeare, small code corpora with license checksums; integrate LoRA/QLoRA with bitsandbytes automatically in KD CLI
- [ ] On-device demo apps
  - [ ] Android app (ExecuTorch NNAPI, streaming UI, tokenizer on device)
  - [ ] iOS app (Core ML MLProgram, streaming UI)
- [ ] "Press Play" UX
  - [ ] Single-button script to fetch small public backbones, run KD for N steps, export, and install demo app packages
- [ ] Bench/QA
  - [x] Provider microbench supports thresholds and fusion checks (Attention/QLinearMatMul), emits JSON; wire into CI on devices to fail on regressions
  - [ ] GitHub Actions CI: install minimal extras, run unit tests, export ONNX decode-step, validate outputs, and run auto-benchmark (CPU)
  - [ ] Add self-hosted runners: Windows (DML) and macOS (MPS) to validate provider backends and enforce tokens/s/KV footprint thresholds

## Roadmap to mobile 2–4GB multimodal frontier

- [ ] Tokenizer and unified vocab
  - [ ] Freeze unified vocab ranges and add integrity checks across text/image/video/audio codebooks
  - [ ] Emit sidecar vocab mapping files for on-device tokenization
- [ ] Model architecture
  - [ ] HRM: integrate hardware routing mixture with low-overhead gating; export-friendly path with static 2-expert hint and ExecuTorch/ORT branching guides
  - [ ] Memory: adopt sliding-window decode defaults and paged KV with page cache allocator (CPU/GPU/NNAPI backends)
  - [ ] Attention: fused RoPE + MLA kernels for DML/CoreML/NNAPI; reduce kernel launches; int8 KV path
  - [ ] Speculative decoding: enable tree/draft with verifier head; add acceptance auto-thresholding and streaming API
  - [ ] Multi-token prediction heads tuned for mobile throughput; train-time losses wired
- [ ] Export & runtimes
  - [ ] ExecuTorch export for decode-step with NNAPI delegate; verify on-device integer attention and QNN acceleration
  - [ ] Core ML MLProgram decode-step with ANE attention mapping and QDQ
  - [ ] ORT Mobile: ensure Attention/QLinear fusions on NNAPI/DML/CoreML EPs; per-op PTQ maps complete
  - [ ] DynamicCache: adopt new torch.export ONNX exporter when stable; remove shim
  - [ ] KV paging: enforce sidecar dims (`dl_per_layer`) in runners; add device runners for paged decode and CI thresholds; document in README
- [ ] Quantization
  - [ ] KV quantization calibration sidecar generation; validate u8/NF4 KV with accuracy & speed on device
  - [ ] Weight quantization: AWQ/GPTQ helpers wired for text model; ExecuTorch/ORT int4/8 friendly layouts
- [ ] Datasets & training
  - [ ] Curate small-scale public backbones and KD datasets across modalities (code/math/vision/audio/video)
  - [ ] Long-context training with YaRN/PI baked; canaries for 32k/128k decode stability and perplexity drift
- [ ] Bench & QA
  - [ ] Device runners (NNAPI/ANE/DML) with enforced TPS and KV memory thresholds in CI
  - [ ] Windowed decode stability and regression tests across providers
  - [ ] ExecuTorch `.pte` device-run timing with threshold in Android ADB workflow (promote from optional to required when runtime available)

## Nice-to-haves (archived/condensed)
- [ ] Add `requirements-windows.txt` and `requirements-linux.txt` minimal sets
- [ ] Optional telemetry-free progress UI (rich/tqdm) across CLIs
- [ ] Artifacts manifest schema versioning and integrity checks

### Implemented in 0.1.2 scaffolding
- [x] Sliding-window attention (`window_size`) for on-device memory control
- [x] Export/training flags for long-context (RoPE `--target_ctx`, `--rope_base`) and decode-step windowed attention
- [x] Runtime-aware per-operator ONNX PTQ presets (generic/nnapi/coreml/dml) and `--onnx_preset` in one-button release
- [x] Speculative decoding supports external `draft_model` or MTP heads with verifier threshold
- [x] VQ codebook trainer `training/vq_train.py`; `ImageVQ` loads saved codebooks via `codebook_path`
- [x] PPO skeleton `training/ppo_rl.py` for programmatic rewards; ready to plug GAE/clipping and task rewards
- [x] Distillation extended: sequence-level KD and expert-route KD stub; CLI flags `--seq_kd`, `--expert_route_kd`
- [x] KD JSONL data loader with rationales and router targets (`training/data/kd_jsonl.py`) and README examples

### Next up (device kernels & delegates)
- [ ] Core ML attention path using Apple attention ops; fuse latent-KV reduce/expand where possible
- [ ] ExecuTorch NNAPI delegate mapping for decode-step graph; per-op int8/int4 maps aligned to device
- [ ] ONNX Runtime EP configs for NNAPI/CoreML/DML loaded via provider hints in mobile runners

## Training UX
- [x] Improve logs: ETA, tokens/s EMA, CUDA mem stats; JSONL log files; periodic checkpointing across pretrain/LoRA/distill
- [ ] Add resume-from-latest checkpoint flags and auto-log rotation
- [ ] Optional WebSocket/TensorBoard streaming of logs during KD

## Packaging/installability
- [x] Gate Coqui `TTS` on platforms/versions with available wheels; document Piper ONNX fallback
- [ ] Add auto-detection/selection of audio backends at runtime with clear error messages and remediation hints
- [ ] Provide `requirements-windows.txt` and `requirements-linux.txt` with minimal, known-good sets

## KVQ calibration schema
- [x] Emit `weights/kvq_calibration.json` with `group` and per-layer entries; optionally store large per-layer stats into sidecar `.npz` files and reference them from JSON.
- [ ] Ensure PyTorch generator and ONNX runner both honor per-head/group scales when calibration is present; write conformance tests.

## Retrieval and kNN‑LM
- [x] Add compact kNN‑LM cache (`inference/knn_cache.py`) with FAISS/NumPy backends; expose blending flags in generator.
- [x] Validate kNN‑LM cache integration in generator and document usage in README; add CLI flags and example snippet.
- [ ] Add training/runtime hooks to populate cache from local docs or conversation memory; evaluate impact on factuality.

## Multimodal image path
- [x] Add CLI flags to select Stable Diffusion HF id or local path, steps, size, and output path
- [x] Add ONNX (diffusers-onnx) loader option for image decoder; allow callable injection for Core ML / ExecuTorch
- [x] Add SD export tooling: ONNX (Optimum pipeline), Core ML VAE decoder, ExecuTorch VAE decoder (`export/diffusion_export.py`)
- [x] Add optional conditioning via `HiddenToImageCond` (aligner) with pooled hidden state; TODO: cross-attn/FiLM conditioning
- [x] Add FiLM-like scale/shift outputs in `HiddenToImageCond` to support decoder modulation
- [x] Fix duplicate video path in CLI; auto-detect `timm` ViT‑tiny if installed; EnCodec audio tokenizer wrapper added
- [x] Implement ONNX Runtime Stable Diffusion callable and CLI wiring (no diffusers required)
- [x] Fix `ImageGenPipeline` backend dispatch for ONNX callable
  - [ ] Lock lightweight U-Net defaults for SD export (document recommended small variants); ensure provider maps for ONNX EPs
  - [ ] Evaluate memory-optimized attention (xformers/Flash) and low-precision UNet where supported for mobile
  - [ ] Wire NNAPI/ANE/GPU provider-specific session options and profiles; add sample configs and per-operator quantization maps
  - [ ] Bundle optional scripts to export video diffusion backbones (ref-only); document compatible HF ids and device RAM constraints

## Multimodal fusion
- [x] Allow core forward to accept pre-embedded features (bypass embedding for fused sequences)
- [x] Add `MultimodalComposer` with vision projector and learned modality tokens (IMG/VID BOS/EOS)
- [x] Add video fusion pathway using `SimpleVideoEncoder` + projector
- [x] Add dataset adapters for VL/VQA to feed fused sequences during training

## Multimodal video/audio
- [x] Add diffusers-based text-to-video pipeline wrapper and CLI integration
- [x] Add image-to-video with seed image generation and motion control
- [x] Add ASR wrapper (faster-whisper/whisper) and TTS wrapper (Coqui/pyttsx3)
- [x] TTS: add Piper CLI support (if installed) with `PIPER_MODEL` env or argument
- [x] Wire ASR/TTS into a unified audio CLI task
  - [x] Add audio WER and image CLIPScore/FID evaluators; document extras in README
  - [x] Video VQ trainer and encoder/decoder; Video VL dataset adapter emitting VQ tokens; fused training loop variant

## Agentic loop
- [ ] Adopt `examples/prompts/agentic_continuous_dev.md` as the development driver
- [ ] Automate periodic runs of eval harnesses and mobile packager; fail CI on regression
- [x] Add GitHub Actions workflow to run smoke tests: importability, tiny CPU generate, ONNX export, ORT roundtrip

### Presets
- [x] `mobile_4gb` preset
- [x] `mobile_2gb` preset and CLI selection
 - [ ] Add JSON/YAML export of presets and a loader for external tools

## New utility
- [x] Add memory estimator for on-device budgeting (`inference/memory_estimator.py`)
  - [x] Add CLI to validate weights folder integrity and required artifacts
  - [x] Add GitHub Actions workflow for smoke tests (imports, tiny generate, ONNX export, ORT roundtrip)
 - [x] Add preset JSON export tool (`tools/presets_export.py`) and weights validator (`tools/weights_validator.py`)
  - [x] Add Dockerfile (CUDA runtime) and GPU quickstart docs for KD/LoRA
  - [x] Auto-benchmark writes JSON summary and validates ONNX outputs
  - [x] HF tokenizer selection via env `OMNICODER_HF_TOKENIZER` with fallback to simple tokenizer
  - [x] Data engine: mirror/index CLI for VL and ASR datasets
  - [x] ONNX fusions: Attention and QDQ packing + QLinearMatMul conversion
  - [x] Provider microbench harness (ORT providers tps)
- [x] Add single-button orchestrator `tools/press_play.py` to run KD (optional), exports, benchmarks, and smokes

## MLC/TVM
- [x] Add `export/mlc_compile.py` wrapper for tvmc-based compilation of decode-step ONNX

## Recently completed
- [x] Minimal runnable text generation loop (`inference/generate.py`) using `OmniTransformer`
- [x] Placeholder tokenizer for smoke tests (`training/tokenizers.py`)
- [x] ONNX export and runtime smoke test for text model
- [x] Minimal pretraining loop over folder of `.txt` files
- [x] Text DataModule and minimal text exact-match eval
- [x] PyTorch dynamic quantization export (int8 CPU)
 - [x] Fixed FAISS retriever TF‑IDF weighting (smoothed log‑IDF) and documented dependency

## Newly identified work from verification pass
- [x] Stabilize SD ONNX export path by trying Optimum API then CLI
- [ ] Add small CPU-only image latency microbench that uses the produced SD ONNX directory (no diffusers dependency)
- [ ] Integrate DynamicCache with ONNX dynamo exporter when PyTorch exposes the stable API (replace manual KV tensors with DynamicCache state to reduce graph size and improve compatibility).
- [x] Harden text dataset loader for KD/LoRA/Pretrain demos to avoid scanning `.venv`, `site-packages`, and other large/hidden directories; handle OS traversal errors gracefully.
- [ ] Add a tiny sample text under `examples/text/` and default demos to that folder to avoid scanning the project root.
- [ ] Replace shim with real `torch.onnx.dynamo_export` DynamicCache once API stabilizes; add conformance tests and CI artifacts for decode-step with DynamicCache state.
- [ ] Add Core ML attention/RoPE custom ops or mapping to Apple attention to avoid traced graph limitations.
- [ ] Implement provider-specific fused MLA kernels (NNAPI/ANE/DML) and align weight/activation layouts, especially for int4 and KV u8/NF4.
  - [x] DirectML path: persistent device + per-shape mask cache to reduce transfers; document usage and microbench.
  - [ ] Core ML/MPS: align latent layout (B*H,T,DL) with exporter and add shape conformance tests.
  - [ ] NNAPI: bind ExecuTorch delegate fused op when available; keep CPU fallback until then.
- [ ] Align int4 packers and KV u8/NF4 layouts across exporters and device runners; add golden tests and provider microbench thresholds per device.
- [ ] Add CI long-context decode-step roundtrip for 32k/128k variants; enforce stability thresholds.
- [ ] Provide Android/iOS sample apps with streaming UI and tokenizer; integrate ExecuTorch NNAPI delegate and Core ML MLProgram runner.
- [x] Train and export VQ‑VAE codebooks and unify vocab slices for multimodal training; add loaders and sample configs.
- [ ] Expand RL/GRPO tasks battery and hook multimodal rewards (CLIPScore/FID/FVD/FAD) into PPO/GRPO scripts; add verifier‑head acceptance metrics.
- [ ] Tighten memory estimator with KV‑quant calibration integration and provider‑specific overheads; output headroom for 2/4 GB presets.
  - [x] Verify one-command mobile packager emits decode-step ONNX, int8 ONNX, provider quant maps, and summary on Windows.
 - [ ] KD import guards documented and tested; add unit that mocks transformers lazy imports to assert no `torchvision` is required for tiny text teachers.

## High-impact performance plan (execution checklist)
- [ ] Fused MLA/MQA provider backends: implement kernels and plumb in `modeling/kernels/mla_providers.py` for NNAPI/Core ML/DML; add golden correctness/perf tests.
- [ ] Int4 weight + KV quant on device: align packing with provider kernels; export sidecars; per‑step dequant kernels in runners.
- [ ] Exporter migration: `torch.onnx.export(..., dynamo=True)` + DynamicCache; ONNX/Core ML decode‑step conformance suite.
- [ ] Long-context canaries: automate 32k/128k decode‑step exports and tokens/s regression thresholds; windowed decode stability tests.
- [ ] Speculative decoding: tree acceptors with verifier; acceptance thresholds tuned for mobile; add tests.
- [ ] Retrieval/kNN‑LM: integrate minimal on-device ANN and caching; add perf/accuracy toggles.

## Major new tasks (2025-08-14)
- [ ] DynamicCache decode‑step export (torch.onnx.dynamo_export) and conformance tests; remove manual KV tensors.
- [ ] True provider kernels for fused MLA/MQA and int4 GEMM on NNAPI/Core ML/DML; align weight/KV packers; golden tests.
- [ ] On‑device ANN PQ index for retrieval/kNN‑LM with mmap and per‑device calibration; expose cache budget controls.
- [ ] Hierarchical video tokens + frame interpolation decoder; tiled per‑op PTQ and streaming decode for 2–4 GB devices.
- [ ] Unified vocab sidecar integrity checks in exporters and mobile runners; enforce contiguous, non‑overlapping ranges.
- [ ] Android ExecuTorch demo app (NNAPI delegate) and iOS Core ML app with tokenizer + streaming UI; Press Play deploy stage.
- [ ] Expand auto‑benchmark: GSM8K/MBPP exact‑match, CLIPScore/FID, FVD/FAD; save artifacts and CSV summaries.
- [ ] Provider microbench CI on self‑hosted devices (Windows DML, macOS MPS) with TPS thresholds and fusion presence gates.
- [ ] Torch.compile default for desktop decode path with warmup/fallback; tokens/s regression canaries.
- [ ] SD ONNX numeric parity investigation; set per‑component tolerances; document in README.
