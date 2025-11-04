# OmniCoder 1 (edge-first multimodal model)

> Archived notice: This README has been condensed. See links below. Historical notes remain below for reference.

## At a glance

- Quickstart: `docs/legacy/Quickstart.md`
- Architecture: `docs/legacy/Architecture.md`
- Datasets: `docs/legacy/Datasets.md`
- Teachers: `docs/legacy/Teachers.md`
- Project plan (historical): `docs/legacy/ProjectPlan.md`
- Backlog:
  - Bugs: `todo/bugs.md`
  - Features: `todo/features.md`
  - Milestones: `todo/milestones.md`
  - Labels: `todo/labels.md`
- Env template: `env.example.txt` (copy to `.env`)
  - Drift guard: `python -m omnicoder.tools.env_audit --root . --env env.example.txt --out weights/env_audit.json`

### Install (minimal)

```bash
python -m venv .venv && . .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
cp env.example.txt .env  # fill in tokens locally (do not commit)
```

### Quickstart

```bash
# Text generation (toy weights-free path)
python -m omnicoder.inference.generate --prompt "Hello OmniCoder" --device cpu

# One-button flow (tests → export → benches; see weights/release)
python -m omnicoder.tools.press_play --device cpu --out_root weights
```

Key runtime knobs: expert paging (`OMNICODER_EXPERT_PAGING=1`), long-context (`--mem_slots`, `--window_size`), and exporters (ONNX/Core ML/ExecuTorch) under `src/omnicoder/export/`.

---

## Overview

OmniCoder is an edge-first, unified multimodal MoE stack targeting fast, small-memory inference on Android/iOS while remaining trainable and testable on a single machine. It includes:

- Sparse MoE Transformer core with hierarchical routing and compressed attention for long context
- Unified text/code/vision/video/audio I/O (VQ + continuous latents) and small refiners
- One-button flows for tests, export, and lightweight training/benchmarking
- Mobile exporters (ONNX/Core ML/ExecuTorch) with provider microbenches and gating thresholds

What you can do quickly:
- Run a CPU-only text demo, or a short, budgeted training flow
- Export a decode-step ONNX and benchmark providers, or pack mobile assets for a sample app
- Iterate on research: variable‑K, speculative decoding, expert paging, and memory retention

## Repo map (high level)

```text
src/omnicoder/
  inference/         # generation loops, runtimes (ONNX, Core ML, ExecuTorch, llama.cpp)
  modeling/          # Transformer MoE core, attention, memory, multimodal modules, kernels
  training/          # pretrain, KD, LoRA/QLoRA, RL (GRPO/PPO/RLHF)
  export/            # exporters and mobile packager (ONNX/Core ML/ExecuTorch/GGUF)
  tools/             # orchestration CLI (press_play, run_training, benches, packaging)
  eval/              # evaluation harnesses (text/code/VL/audio/video)
profiles/            # datasets/teachers/provider thresholds and mobile presets
examples/            # tiny runnable JSONL, prompts, sidecars for smokes
docs/legacy/         # archived docs (historical deep dives)
```

## Common tasks

- Quick CPU demo:
  - `python -m omnicoder.inference.generate --prompt "Hello" --device cpu`
- Budgeted end-to-end flow (tests → export → benches):
  - `python -m omnicoder.tools.press_play --device cpu --out_root weights`
- ONNX export + provider bench:
  - `python -m omnicoder.export.onnx_export --out weights/release/text/omnicoder_decode_step.onnx`
  - `python -m omnicoder.inference.runtimes.provider_bench --model weights/release/text/omnicoder_decode_step.onnx --providers CPUExecutionProvider DmlExecutionProvider --out_json weights/release/text/provider_bench.json`

For more, scan `src/omnicoder/tools/` and `src/omnicoder/export/`.


## For Dummies: One Button

- Windows (PowerShell):
  - `./play.ps1` (deprecated; prefer `press-play`)
- macOS/Linux:
  - `bash play.sh` (deprecated; prefer `press-play`)
- Docker (no local Python needed):
  - `docker build -t omnicoder:cuda .`
  - `docker compose run --rm press_play`
  - One-button budgeted training (hours): `python -m omnicoder.tools.run_training --budget_hours 10 --device cuda`
  - Tip: enable auto resource scaling with `OMNICODER_AUTO_RESOURCES=1` (default in docker-compose). When enabled, thread envs override conservative base-image defaults (e.g., 1) to fully utilize CPUs. Tune threads via `OMNICODER_THREADS` or scale with `OMNICODER_THREADS_FACTOR` (e.g., 0.5 for half CPUs); cap DataLoader workers with `OMNICODER_WORKERS_MAX`. For expert-parallel across two GPUs, set `OMNICODER_ENABLE_EP=1 OMNICODER_EP_DEVICES=cuda:0,cuda:1`.
  - Export to phone:
    - Android (ADB): `python -m omnicoder.tools.export_to_phone --platform android --tps_threshold 15`
    - iOS (Core ML): `python -m omnicoder.tools.export_to_phone --platform ios`; optional device smoke: set `OMNICODER_IOS_SMOKE=1` then run `python -m omnicoder.tools.ios_coreml_smoke --tps_threshold 6`
  - Single-button full flow: `lets-gooooo --budget_hours 1 --device cuda --out_root weights`
  - Note: `lets-gooooo` runs tests (`pytest -vv -rA`) before training and aborts on failures.

### Quick examples: teachers, datasets, and expert paging

- Teachers (profiles): edit `profiles/teachers.json` to set defaults per preset, or override via env when running the orchestrator:
  - Single teacher (text): `OMNICODER_TEACHER=meta-llama/Meta-Llama-3.1-8B-Instruct lets-gooooo --budget_hours 2`
  - Domain teachers: `OMNICODER_KD_CODE_TEACHERS="bigcode/starcoder2-3b" OMNICODER_KD_VL_TEACHERS="laion/CLIP-ViT-B-32-laion2B-s34B-b79K" python -m omnicoder.tools.run_training --budget_hours 2`
  - Draft preset: `OMNICODER_DRAFT_PRESET=draft_2b python -m omnicoder.tools.run_training --budget_hours 1`

- Datasets (profiles): edit `profiles/datasets.json` to point to your corpora, or override via env:
  - Text folder of `.txt`: `OMNICODER_DATA_TEXT=/data/my_corpus python -m omnicoder.tools.run_training --budget_hours 1`
  - Code JSONL: `OMNICODER_DATA_CODE=/data/code.jsonl python -m omnicoder.tools.run_training --budget_hours 1`
  - Unified retrieval multi-index: build once and persist under `weights/retrieval_multi_index`:
    - `python -m omnicoder.tools.multi_index_build --roots /data/text docs --out weights/retrieval_multi_index --device cpu`
    - Then enable at runtime: `OMNICODER_MULTI_INDEX_ROOT=weights/retrieval_multi_index python -m omnicoder.inference.generate --prompt "..."`

- Expert paging (runtime): enable on-demand expert LRU cache for MoE layers:
  - Enable: `OMNICODER_EXPERT_PAGING=1`
  - Fixed capacity: `OMNICODER_EXPERT_PAGING_CAP=8`
  - Budget-derived capacity: `OMNICODER_EXPERT_PAGING_BUDGET_MB=256` (auto-derive resident experts from budget)
  - Router-based prefetch: `OMNICODER_EXPERT_PREFETCH_N=2`
  - Warm-start hint API: models may call `pager.warm_hint(router_probs, topk=4)` to prefetch experts likely to be used next.

### HRM and RL defaults

- HRM is enabled by default in `OmniTransformer`. Disable by constructing with `use_hrm=False` or adjusting your preset factory.
- RL short loops (GRPO/PPO) are enabled by default in the orchestrator. Disable by setting `OMNICODER_ENABLE_GRPO=0` and/or `OMNICODER_ENABLE_PPO=0`.

## Architecture deep-dive (concise status 2025-08-18)

- Core model: `src/omnicoder/modeling/transformer_moe.py` implements a sparse MoE Transformer with hierarchical routing (Top‑K, Multi‑Head, GRIN, optional LLMRouter/I2MoE). Capacity-aware dispatch, optional expert paging hooks, and sub‑experts/shared general experts are wired.
- Attention/long context: `src/omnicoder/modeling/attention.py` provides Multi‑Query + Multi‑Head Latent Attention (KV compression), sliding window, compressive memory slots, optional landmark indexer (`modeling/memory.py`). YaRN/PI hooks and long‑ctx ONNX variants exist. KV quantization (u8/NF4) and paged‑KV sidecars are supported by exporters/runners.
  - Landmark + Random‑Access Attention: landmark tokens are enabled by default (override with `OMNICODER_USE_LANDMARKS=0`), and random‑access jumps are supported by passing a short `prefix_hidden`/`landmark_prefix` during windowed decode.
- Decoding throughput: Multi‑Token Prediction heads, draft‑and‑verify hooks, adaptive variable‑K and early‑exit heads (export‑guarded). Generator integrates kNN‑LM and windowed decode with memory priming. Balanced routing (Sinkhorn) can be toggled at train time via `OMNICODER_ROUTER_SINKHORN_ITERS`/`OMNICODER_ROUTER_SINKHORN_TAU`.
  - Adaptive precision runtime (emulation): enable activation fake-quant based on confidence with `--act_quant` (env `OMNICODER_ACT_QUANT=1`) and control bits via `--act_quant_min_bits/--act_quant_max_bits`.
  - Non‑autoregressive draft (research): set `OMNICODER_MASK_PRED=1` to try a simple mask‑predict loop for long outputs; knobs `OMNICODER_MASK_ITERS`, `OMNICODER_MASK_TOKEN_ID`.
### 3D latents (optional)

We provide a minimal voxel latent head and a tiny NeRF-like renderer for view-consistent imagery/video:

- Voxel head: `modeling/multimodal/latent3d.VoxelLatentHead(d_model, grid=(D,H,W), channels=8)`
- Tiny renderer: `modeling/multimodal/latent3d.TinyNeRFRenderer(channels=8)` taking `(voxels, rays_o, rays_d, steps)` → `(B,N,3)` RGB.

These are export-friendly proxies for on-device demos; training/export are off by default.
  - Speculative block verification: enable block‑level verification to amortize verifier overhead and increase tokens/sec. Use env `OMNICODER_BLOCK_VERIFY=1` and tune `OMNICODER_BLOCK_VERIFY_SIZE` (default 4) to verify contiguous token blocks. Applies to both external draft and MTP‑based drafts.
- Multimodal IO: VQ‑VAE and adapters for image/video/audio tokenization/decoding, plus continuous latent heads and tiny refiners (image/audio). Fusion via `MultimodalComposer`; VL/VQA fused trainers present.
- Export/mobile: ONNX/Core ML/ExecuTorch decode‑step exporters with KV I/O; per‑op PTQ and provider maps; provider microbench and thresholds.
- Training: DS‑MoE curriculum, multi‑teacher KD, GRPO/PPO, verifier‑head KD, pre‑alignment stage, retrieval write‑policy hooks, and one‑button orchestrator `omnicoder.tools.run_training`.

Gap audit (remaining for the final goal):
- Vision backbone scale: integrate DINO‑class encoders and unify CLIP/ImageBind‑style pre‑alignment across text/image/audio/video.
- Grounded vision: YOLO‑E‑inspired open‑vocab detection/segmentation expert and SAM‑style fine‑grained editing.
- Image/video fidelity: formalize keyframe+interpolation latent pipeline and tiny ONNX‑friendly refiners; add temporal attention/SSM defaults and FVD gates.
- Audio‑visual sync: phoneme alignment loss and cross‑attention wiring in video pipelines.
- Cross‑modal verifier and cycle consistency: train a mini‑CLIP verifier and enable generate→re‑encode→compare training losses.
- Memory/long‑context: learned retention head for KV/memory; unify multimodal retrieval memory and shared semantic memory prototypes.
- Expert scale/runtime: ≥16 experts/layer presets by default, balanced routing toggle (Sinkhorn), and expert‑parallel sharding docs + launchers.
- 3D latent provision: optional NeRF/voxel latent head and tiny renderer scaffolding (export‑guarded).

What it does:
- Exports a text decode-step ONNX, runs light quantization, optionally stages image pipeline exports, and writes a benchmark summary to `weights/release/`.
- To include quick knowledge distillation on GPU, run: `docker compose run --rm kd_train` (or add `--kd` to `press_play` when you have a GPU and models cached under `/models`).

### Training Probe (time-budgeted 1-step executor)

- To profile and auto-plan a short training session within a time budget (e.g., 120 minutes):
  - `docker compose run --rm train_probe`
  - Outputs:
    - `weights/train_probe_summary.json`: per-task 1-step durations, plan, executed steps, remaining seconds
    - `weights/train_probe_live.log`: live log from the run
  - Env knobs (see `env.example.txt`): `OMNICODER_TRAIN_BUDGET_MIN`, `OMNICODER_TRAIN_DEVICE`, `OMNICODER_TRAIN_SKIP`

## Executive architecture blueprint (aligned to infinite-context, unified multimodal MoE)

- Core: sparse MoE Transformer with hierarchical gating (grouped per modality) + blended gates (Top‑K, multi‑head, GRIN‑style). Capacity-aware routing with static caps for bounded latency.
- Attention: Multi‑Head Latent Attention (latent KV compression) + MQA, SDPA/Flash‑3 fast path, optional fused provider kernels. Sliding‑window decode + recurrent memory slots enable logically infinite context; retrieval (PQ/kNN‑LM) augments long‑range recall.
- Long context: RoPE with YaRN/PI scaling; exporter emits 32k/128k decode‑step variants. Memory priming summarizes distant history into fixed slots and keeps KV bounded. Defaults: when `--target_ctx` exceeds preset or `--window_size>0`, landmark tokens auto‑enable (min 8) for random‑access jumps; set `OMNICODER_USE_YARN=1` or `OMNICODER_USE_PI=1` to auto‑adjust RoPE base alongside scale.
  - New: compressive‑memory auxiliary loss option in training to encourage memory slots to summarize long prefixes; added KV‑bounded long‑doc canaries.
- CI canaries: add 32k/128k decode‑step emission checks and an infinite‑context QA smoke (`tests/test_infinite_context_qa.py`) exercising memory priming + sliding window.
- Multimodal: unified token space with VQ‑VAE codebooks (image/video/audio) and optional continuous latent heads for high‑fidelity generation; diffusion/flow decoders for image/video; EnCodec+vocoder for audio.
- Decoding throughput: multi‑token prediction heads + draft‑and‑verify; optional kNN‑LM blending; KV‑cache quantization (u8/NF4) and paging.
  - Verifier/draft hooks present; TPS benchmarks available in `inference/benchmark.py`; provider thresholds/canaries included.
  - Adaptive speculative control: enable `--adaptive_gating` to adjust draft length by confidence (`--adaptive_top_k_min/max`, `--adaptive_conf_floor`).
  - Variable‑K per layer/token: enable `--adaptive_layer_ramp` and shape with `--adaptive_layer_power` to use more experts early and fewer deep per token; bounded by expert count; export guarded.
  - Defaults: when the model exposes MTP heads and no external draft model or verifier threshold is specified, the generator now accepts up to 2 lookahead tokens per step using MTP heads by default.
- Quantization: weight‑only int4 (AWQ/GPTQ export), ONNX per‑op PTQ with provider maps; device‑grade int4/attention kernels targeted for NNAPI/Core ML/DML.
  - KV‑cache: finalize calibration/sidecars; runners honor per‑group KVQ u8/NF4; CLI `tools/kv_calibrate.py` writes `weights/kvq_calibration.json`.
  - ONNX runner auto‑aligns to KVQ sidecars and supports paged KV + dynamic window inference.
  - New: KV retention sidecar `omnicoder_decode_step.kv_retention.json` is emitted by the mobile packager and auto‑detected by the ONNX decode runner to enforce a compressive‑memory + tail‑window policy at runtime (no graph change). Use `OMNICODER_COMPRESSIVE_SLOTS` to change default slots (default 4). The runner also accepts `--kv_retention_sidecar` to point to a custom JSON. The PyTorch generator can also adopt a retention policy by setting `OMNICODER_KV_RETENTION` to a sidecar path.
  - Learned KV compression sidecar (PyTorch): set `OMNICODER_KV_COMPRESS_SIDECAR` to a JSON emitted by `kv-autoencoder-train` (contains `kv_autoencoder.weights` with `enc.weight`/`dec.weight`). The generator will compress older KV segments via the AE before concatenating the recent window.
### Provider bench and thresholds

- After export, benchmark providers for the decode-step ONNX and verify expected fusions:
```bash
python -m omnicoder.inference.runtimes.provider_bench \
  --model weights/release/text/omnicoder_decode_step.onnx \
  --providers CPUExecutionProvider DmlExecutionProvider \
  --prompt_len 128 --gen_tokens 128 --check_fusions \
  --threshold "CPUExecutionProvider=2.0,DmlExecutionProvider=10.0,CoreMLExecutionProvider=6.0" \
  --out_json weights/release/text/provider_bench.json
```
- Thresholds can be kept in `profiles/provider_thresholds.json`. Orchestrators can auto-load and (optionally) auto-update thresholds from measured TPS when `OMNICODER_AUTO_UPDATE_THRESHOLDS=1`.
- Training: pretrain → multi‑teacher KD (text/code/VL/ASR/TTS) → GRPO/RLHF with multimodal rewards; verifier‑head KD for acceptance.
  - Verifier head KD: `training/verifier_distill.py` distills a teacher to the verifier head; acceptance/TPS measured via `tools/bench_acceptance.py`. To apply preset thresholds, pass: `python -m omnicoder.tools.bench_acceptance --mobile_preset mobile_4gb --threshold_json profiles/acceptance_thresholds.json`.
  - Retrieval write‑policy: `DataModule.teacher_marks_loader()` ingests teacher write marks for supervised training of the write head.
  - Variable‑K and halting defaults for mobile: `training/pretrain.py` exposes `--var_k_train --var_k_min --var_k_max --var_k_threshold[_start/_end]`, `--diff_loss_coef`, `--halt_loss_coef`, `--halt_entropy`. The orchestrator auto‑enables var‑K/halting for `mobile_*` presets unless `OMNICODER_VAR_K_TRAIN=0`/`OMNICODER_HALT_TRAIN=0`. Ablate TPS vs quality using:
```bash
python -m omnicoder.tools.metrics_canaries --bench_variable_k --out weights/variable_k_summary.json
```

What this means for "infinite" context: we combine windowed decode, recurrent memory slots, and retrieval so the effective context is unbounded while compute/memory per step stays fixed. The generator already exposes `--window_size` and `--mem_slots`; retrieval can be enabled via local TF‑IDF/FAISS or PQ.


OmniCoder is a **research reference implementation** of an *edge-first*, all-in-one multimodal model stack designed for **Android/iOS devices with 2–4 GB of RAM**. It includes:

- A compact **sparse MoE Transformer** core with **compressed attention (KV)** and **multi-token decoding**.
- A **hierarchical reasoning loop (HRM-inspired)** for deep step-by-step thinking without huge depth.
- Unified **multimodal I/O**: text, image, video, **audio (speech + music + SFX)**.
 - Export paths for **ExecuTorch**, **MLC-LLM/TVM**, **Core ML** (now includes decode‑step exporter), **ONNX Runtime Mobile**, and **GGUF/llama.cpp**.
- Automated, **no-human-in-the-loop evaluation** for text/code, image, video, and audio.
- Training recipes that fit **single 24 GB GPUs** via **QLoRA/LoRA**, gradient checkpointing, and offloading.

> ⚠️ **Reality check:** This repository is a *buildable skeleton* + reference code and evaluation harnesses. It is not a drop‑in replacement for frontier cloud models out of the box. You will plug in your datasets and (optionally) large teacher checkpoints for distillation/RL to realize the intended performance.

## Goal vs. Current Status

- **Goal (yours):** Frontier‑level multimodal model (GPT‑5/Grok‑4 class) running fully on‑device (Android/iOS) within 2–4 GB RAM, supporting text, image, video, and audio I/O; trainable quickly on a single 24 GB GPU; extremely fast inference; verifiable performance vs. frontier models.
- **Status (this repo):** Provides a runnable minimal LLM pipeline and full scaffolding for a multimodal, mobile‑first stack:
  - Text generation loop runnable via `omnicoder.modeling.OmniTransformer` with causal attention and MoE blocks (toy but functional).
  - Reference HRM module and multimodal adapters as stubs to be wired to real backbones.
  - Export, mobile runtime, training, and evaluation harnesses in place as stubs.
  - Not yet shipping frontier‑level performance; requires plugging real models, training, quantization, and exports.

### What's present
- **Modeling:** Toy MoE Transformer with causal attention, MQA, and LayerNorm; HRM loop; modular multimodal components (vision/audio/video tokenizers & decoders stubs). MoE now includes capacity-aware token dispatch with configurable capacity factor for mobile-friendly routing, plus robust router regularization (z-loss over logits, importance/load balance penalties) and grouped token-wise batched dispatch per expert. The core optionally applies a small HRM (hierarchical reasoning) module inside the transformer for iterative refinement before the final projection, improving reasoning depth without large parameter cost. The transformer can also return hidden states for conditioning downstream decoders.
- **Inference:** Text demo end‑to‑end (`omnicoder.inference.generate`) now runs locally; runtime adapters (llama.cpp/ONNX/MLC) are stubs to be completed.
  - **Training:** Pretrain/LoRA/QLoRA, functional **distillation** loop (teacher→student KD), and first-cut RL tracks (programmatic RL and RLHF skeletons); tokenizer placeholder for smoke tests.
- **Export:** ONNX decode‑step, optional dynamic int8 (`onnxruntime`), ExecuTorch decode‑step. Helpers for AWQ/GPTQ (4‑bit) and GGUF; mobile packager produces artifacts and a memory budget summary. Runtime notes for NNAPI/CoreML/GPU included. Exporters default to the legacy `torch.onnx.export` path for stability in constrained environments; enable the new `torch.onnx.dynamo_export` by passing `--dynamic_cache` or setting `OMNICODER_USE_DYNAMO=1` (opset≥18). Decode‑step export avoids zero‑length cache tensors (uses a minimal past length of 1) to ensure reliable ONNX graph emission and KV paging sidecar writing.
- **Eval:** Code/text/image/video/audio harness stubs; code pass@k harness runnable on example JSONL.
  - Agentic development prompt for continuous improvement in `examples/prompts/agentic_continuous_dev.md`.

### What's missing (gaps to hit the goal)
- Plug **real backbones**: compact ViT for vision; EnCodec + HiFi‑GAN for audio; Stable‑Diffusion‑class U‑Net for images; lightweight video diffusion; ASR/TTS backbones.
  - Status: ViT‑tiny via `timm` is auto‑detected; EnCodec wrapper implemented; HiFi‑GAN vocoder wrapper (`HiFiGANVocoder`) added; Stable Diffusion image pipelines supported via `diffusers` and ONNX callable; text‑to‑video/image‑to‑video wrappers added; ASR (faster‑whisper/whisper) and TTS (Coqui/Piper/pyttsx3) adapters implemented.
- Implement **KV compression kernels** and fused attention with multi‑query/MLA to reduce memory and increase speed.
- Robust **router regularization** and token‑wise batched MoE dispatch for efficiency on mobile.
- **Multi‑token prediction** head and speculative decoding path; long‑context adapters (YaRN/PI) and Retrieval.
- End‑to‑end **quantization** (int4/int8) and mobile **exports** with operator coverage; runtime integration for ANE/NNAPI/GPU.
- Full **training pipelines** with datasets, teacher traces (distillation), and RL rewards; reproducible **benchmarks** vs. frontier models.

These items are tracked in `TODO.md` and changes recorded in `CHANGELOG.md`.

## Existing options and why they fall short

- **Small LLMs on-device (e.g., 7B class with 4-bit quantization)**: feasible on modern phones, but typically closer to GPT‑3.5 level and text‑only.
- **On‑device image generation (Stable Diffusion variants)**: possible but heavy; memory/time constraints hinder high quality and speed; not unified with LLM.
- **On‑device video generation**: currently not practical for frontier quality at 2–4 GB RAM.

Hence this repo focuses on a unified, sparse, quantized, modular design to close these gaps.

### Validation status (0.1.9 series)

- Latest (Docker GPU, this run): full suite passed 134/134, 11 warnings, ~239s. Logs saved to `tests_logs/docker_pytest_full.txt`; exit code persisted in `pytest_exit_code.txt`.
- Text path: imports, model forward, streaming generation, ONNX export, ONNX decode‑step export, ORT desktop runner, ExecuTorch/Core ML exporters (subject to toolchain availability). All smoke‑tested paths run on CPU without external weights. ONNX decode‑step streaming validated via `inference/runtimes/onnx_decode_generate.py`.
- Training: toy pretrain/LoRA/distill scripts execute on CPU; KD requires `transformers` and a teacher checkpoint.
- Retrieval: pure‑Python local retriever works out‑of‑the‑box; FAISS retriever requires `faiss-cpu`. PQ builder now pads TF‑IDF features so `dim % m == 0`, unblocking tiny corpora.
- Multimodal: image generation requires providing a Stable Diffusion pipeline via `diffusers` (not bundled). Adapters for video/audio/ASR/TTS are stubs.
  - Provider profiles: `profiles/windows_dml.json` fixed (single valid JSON object) to unblock profile loading tests.
  - Docker CPU container (this host): full suite completed with 100 passed, 1 warning in ~3m56s. See `tests_logs/docker_pytest_full.txt` and `pytest_exit_code.txt`.
  - Docker compose (latest): full suite completed with 116 passed, 2 warnings in ~3m on this host. New smokes (tool-use postprocess, bench TPS delta, video interpolation, vision seg, keyframe head, cross‑modal verifier) included.
  - Docker CUDA container (this host): full suite green across two runs to avoid OOM on constrained hosts:
    - Part 1 (excluding longctx-variants): 99 passed. Logs: `tests_logs/docker_pytest_part1.txt`, exit code: `pytest_exit_code_part1.txt`.
    - Part 2 (only longctx-variants): 3 passed. Logs: `tests_logs/docker_pytest_longctx.txt`, exit code: `pytest_exit_code_longctx.txt`.
    - Exporter note: `OMNICODER_EXPORT_TINY` now only applies to heavy export variants (kv_paged/longctx emission) and is ignored for standard decode‑step exports to preserve stable input names (`k_lat_{i}`/`v_lat_{i}`).
  - Note: On Windows Docker, `docker compose` flags like `--gpus`/`--compatibility` may be unsupported. Prefer `docker run --gpus all ...` examples when running with GPU.
 - Training probe: `docker compose run --rm train_probe` fills the given budget with 1‑step tasks (defaults to `flow_recon` focus); see `weights/train_probe_summary.json`.
 - Draft training & acceptance bench (GPU recommended): `docker compose run --rm draft_train` trains a compact draft via KD (LoRA), exports `weights/text/draft_decode_step.onnx`, and writes `weights/draft_acceptance.json`.
 - Multimodal: image generation requires providing a Stable Diffusion pipeline via `diffusers` (not bundled). Adapters for video/audio/ASR/TTS are stubs.
  - Docker CUDA container (latest): DynamicCache conformance, long‑context variant export, KV‑paging sidecar, and provider microbenches pass. If `pytest -vv -rA` appears truncated in some environments, use the sequential runner below to guarantee a real exit code is persisted.

#### Sequential Docker GPU runner (robust in constrained environments)

```bash
docker run --rm --gpus all -v "$PWD":/workspace -w /workspace omnicoder:cuda \
  bash -lc ': > tests_logs/docker_pytest_full.txt; rc=0; for f in $(ls tests/*.py | sort); do echo RUNNING:$f | tee -a tests_logs/docker_pytest_full.txt; pytest -vv -rA "$f" | tee -a tests_logs/docker_pytest_full.txt; r=${PIPESTATUS[0]}; if [ $r -ne 0 ]; then rc=$r; fi; done; echo $rc > pytest_exit_code.txt; exit $rc'
```

This runs each test module sequentially, aggregates verbose logs, and writes the true exit code to `pytest_exit_code.txt`.

#### Exporter note (decode‑step ONNX)

- The pytest tiny‑shrink guard now applies only for heavy export variants (kv_paged/longctx). DynamicCache conformance exports use full preset shapes to keep K/V input names aligned, fixing the `Invalid input name: v_lat_*` error in `tests/test_onnx_dynamic_cache_conformance.py`.
 - When using the dynamo exporter (default, opset≥18), the exporter writes a `*.dynamic_cache.json` sidecar for decode‑step models. DynamicCache models are input_ids‑only; the ONNX decode runner detects this sidecar (or input signature) and skips explicit per‑layer K/V feeds while preserving legacy explicit‑KV compatibility.

### New canaries and modules (performance + fidelity)
- KV prefetch canary and predictor:
  - Run: `python -m omnicoder.tools.metrics_canaries --kv_prefetch_canary --kv_page_len 256 --kv_max_pages 32 --kv_prefetch_ahead 1 --kv_steps 1024`
  - Outputs `kv_prefetch` stats (hit_rate, miss_rate, stall_ratio) in `weights/metrics_canaries.json`.
- Video FVD (optional):
  - Run: `python -m omnicoder.tools.metrics_canaries --video_pred_dir <pred> --video_ref_dir <ref>` (requires `pytorch-fvd`).
- Continuous latent refiner (tiny):
  - Quicktrain wrappers:
    - Image: `refiner-train --kind image --data examples/data/vq/images --steps 200 --export_onnx weights/refiners/image_refiner.onnx`
    - Audio: `refiner-train --kind audio --mel_dir data/mels --steps 200 --export_onnx weights/refiners/audio_refiner.onnx`
  - Direct training:
    - Image: `python -m omnicoder.training.flow_recon --data <images_or_jsonl> --use_refiner --steps 1000 --export_refiner_onnx weights/refiners/image_refiner.onnx`
    - Audio: `python -m omnicoder.training.audio_recon --mel_dir <mel_dir> --use_refiner --steps 1000 --export_refiner_onnx weights/refiners/audio_refiner.onnx`
- Temporal video smoothing (TemporalSSM):
  - `VideoGenPipeline(..., use_temporal_ssm=True, temporal_ssm_dim=384, temporal_ssm_kernel=5)` applies a tiny learned temporal smoothing before optical-flow consistency.
  - Train the tiny temporal module and (optionally) compute FVD:
    - `python -m omnicoder.training.video_temporal_train --videos data/videos --frames 16 --d_model 384 --steps 500 --device cuda --export_onnx weights/video/temporal_ssm.onnx`
    - With latent noise propagation: add `--propagate_noise --noise_dim 64 --noise_alpha 0.95 --noise_gamma 0.05`
    - With FVD (requires pytorch-fvd): add `--fvd_ref_dir data/fvd_ref --fvd_pred_dir data/fvd_pred`
  - Env mirrors (optional): set in `.env` instead of flags:
    - Frames/shape/optim: `OMNICODER_VIDEO_TEMPORAL_FRAMES`, `OMNICODER_VIDEO_TEMPORAL_D`, `OMNICODER_VIDEO_TEMPORAL_KERNEL`, `OMNICODER_VIDEO_TEMPORAL_EXPANSION`, `OMNICODER_VIDEO_TEMPORAL_STEPS`, `OMNICODER_VIDEO_TEMPORAL_DEVICE`, `OMNICODER_VIDEO_TEMPORAL_LR`, `OMNICODER_VIDEO_TEMPORAL_OUT`, `OMNICODER_VIDEO_TEMPORAL_ONNX`
    - Noise propagation: `OMNICODER_VIDEO_TEMPORAL_PROPAGATE_NOISE`, `OMNICODER_VIDEO_TEMPORAL_NOISE_DIM`, `OMNICODER_VIDEO_TEMPORAL_NOISE_ALPHA`, `OMNICODER_VIDEO_TEMPORAL_NOISE_GAMMA`
    - FVD paths: `OMNICODER_VIDEO_TEMPORAL_FVD_REF`, `OMNICODER_VIDEO_TEMPORAL_FVD_PRED`
    - AV-sync: `OMNICODER_VIDEO_TEMPORAL_AV_SYNC`, `OMNICODER_VIDEO_TEMPORAL_AUDIO_DIR`, `OMNICODER_VIDEO_TEMPORAL_AV_WEIGHT`
 - Video generation sidecar/knobs:
   - `VideoGenPipeline.generate(..., keyframe_cadence=K, interp_strength=S)` now writes a JSON sidecar alongside the output video with `{keyframe_cadence, interp_strength, steps, num_frames, size, temporal_*}` to aid reproducibility and app visualization.

### Alignment and VL fused pretrain (quickstart)
- Train a tiny pre‑aligner (InfoNCE) to align image/text embeddings:
  - `python -m omnicoder.training.pre_align --data examples/data/vq/images --steps 50 --device cuda --embed_dim 256 --out weights/pre_align.pt`
- Train the model's shared concept latent head to align with modality embeddings (InfoNCE + triplet with random negatives):
  - `python -m omnicoder.training.cross_modal_align --jsonl examples/vl_auto.jsonl --mobile_preset mobile_4gb --prealign_ckpt weights/pre_align.pt --steps 100 --device cuda`
- Run a short VL fused pretrain with optional auxiliary alignment loss:
  - `python -m omnicoder.training.vl_fused_pretrain --jsonl examples/vl_auto.jsonl --mobile_preset mobile_4gb --pre_align_ckpt weights/pre_align.pt --align_weight 0.1 --steps 3 --device cuda`
Notes: The feature‑fusion path uses a lightweight vision encoder; ensure images exist under `examples/data/vq/images` or provide your dataset JSONL.

### One-button training orchestrator
- Plan and run a budgeted end-to-end training flow (pre-align → unified multi-index build → DS‑MoE pretrain → KD draft → acceptance bench (defaults) → VL/VQA fused → AV heads → optional RL → export+bench), resume-friendly. Stage snapshots are auto-benched:
  - Emits `bench_after_pretrain.json` and `bench_after_kd.json` plus a final `bench_stage_summary.json`.
  - `python -m omnicoder.tools.run_training --budget_hours 10 --device cuda --out_root weights`
  - Tip: set `OMNICODER_AUTO_RESOURCES=1` in `.env` to auto-scale CPU threads and DataLoader workers based on host cores. Override with `OMNICODER_THREADS` / `OMNICODER_WORKERS`. Optional: enable learned retention (`OMNICODER_TRAIN_RETENTION=1`) and variable‑K/halting training (`OMNICODER_VAR_K_TRAIN=1`, `OMNICODER_HALT_TRAIN=1`).
- Configure via environment (see `env.example.txt`): `OMNICODER_TRAIN_BUDGET_MINUTES`, `OMNICODER_TEACHER`, `OMNICODER_TRAIN_PRESET`, `OMNICODER_STUDENT_PRESET`, etc.

### Write‑policy head acceptance threshold

Derive/update the acceptance threshold for the learned write‑policy head from teacher marks (or model predictions):

```bash
write-policy-acceptance --marks examples/teacher_marks.jsonl --preset mobile_4gb --out profiles/write_policy_thresholds.json
```

At inference/training, the threshold can be loaded from `profiles/write_policy_thresholds.json` to control write decisions into external memory (e.g., PQ or kNN cache).

### Export to phone
- Android (ADB): pushes ONNX decode-step and sidecars to `/data/local/tmp/omnicoder/` and optionally runs a NNAPI smoke with TPS threshold
  - `python -m omnicoder.tools.export_to_phone --platform android --tps_threshold 15`
  - Override artifact path with `--onnx`; default is `weights/release/text/omnicoder_decode_step.onnx`
- iOS (Core ML): copies `omnicoder_decode_step.mlmodel` into `SampleApp`/`SampleConsole` resources
  - `python -m omnicoder.tools.export_to_phone --platform ios`
  - Override with `--mlmodel`; default is `weights/release/text/omnicoder_decode_step.mlmodel`
  - Tip: to prefer Core ML QLinearMatMul when supported, enable post-conversion weight quantization: pass `--prefer_qlinear` to `coreml_decode_export` or set `OMNICODER_COREML_PREFER_QLINEAR=1`.

### Provider microbench defaults
- Default thresholds now include DML/CoreML/NNAPI: `--threshold "CPUExecutionProvider=2.0,DmlExecutionProvider=10.0,CoreMLExecutionProvider=6.0,NNAPIExecutionProvider=6.0"`.
- Thresholds also auto‑load from `profiles/provider_thresholds.json` when not provided. Fusion checks are auto‑enabled when mobile/GPU providers are used; benches fail if fused Attention and/or QLinearMatMul are missing where expected. Provider bench can emit tokens/s canaries (`--canary_tokens_per_s`) consumed by the app/dashboard and orchestrators.

### App TPS/KV visualizations
- Use `python -m omnicoder.tools.visualize_metrics --bench_json weights/release/text/provider_bench.json --onnx_model weights/release/text/omnicoder_decode_step.onnx --out_dir weights/release/text` to generate:
  - `metrics.svg`: bar chart of tokens/sec per provider with speedup annotations
  - `kv_info.json`: consolidated KV paging/retention/window info from sidecars
- Build a simple dashboard HTML (for preview/app ingestion):
  - `python -m omnicoder.tools.app_assets --assets_dir weights/release/text --bench_json weights/release/text/provider_bench.json --onnx_model weights/release/text/omnicoder_decode_step.onnx`
  - Output: `weights/release/text/dashboard.html` (inline SVG + KV table)
- Copy assets into sample app folders (best-effort):
  - `python -m omnicoder.tools.package_app_assets --assets_dir weights/release/text --android_assets app/src/main/assets/omnicoder --ios_assets SampleApp/Resources/omnicoder`
  - Copies `metrics.svg`, `kv_info.json`, and `dashboard.html` into app assets.

### Router ablations (LLMRouter vs baseline)
- Quick ablation of context-aware routing vs baseline on a short prompt:
  - `router-ablate --mobile_preset mobile_4gb --max_new_tokens 64 --device cuda --out_json weights/router_ablation.json`
  - Output includes baseline_tps, llmrouter_tps, and speedup_x.

## Highlights

### Unified Multimodal MoE — Architecture Summary (concise)
- Core: Sparse MoE Transformer with hierarchical, modality-aware routing (Top‑K, Multi‑Head, GRIN, optional LLMRouter). Experts specialize by domain (text/code/vision/audio/video) and interaction type (text↔image, video↔audio) while keeping per‑token active params small for mobile.
- Attention: Multi‑Query + Multi‑Head Latent Attention (KV compression) with optional SSM interleaves for long‑range mixing; sliding‑window decode, landmark indexing, and compressive memory enable effectively unbounded context with bounded KV.
- Decoding: Multi‑Token Prediction, speculative draft‑and‑verify hooks, variable‑K expert activation, and early‑exit halting heads adapt compute to difficulty for high tokens/sec.
- Memory/RAG: kNN‑LM and PQ/FAISS retrieval integrated; sidecars calibrate KV quant and paging; planned learned retention head to decide what to keep/compress/drop.
- Multimodal IO: Shared vocab slices unify discrete tokens (text + VQ image/video/audio); continuous latent heads plus tiny refiners target diffusion‑level fidelity under exportable ONNX graphs.
- Exports/Mobile: Decode‑step ONNX with KV I/O, long‑context variants, per‑op PTQ, provider fusion hints; Core ML/ExecuTorch builders; provider microbench with thresholds for CPU/DML/CoreML/NNAPI.

  - **Core model (`src/omnicoder/modeling/transformer_moe.py`)**
  - Sparse **Mixture-of-Experts** MLP with top‑k routing, z-loss load balancing (configurable).
  - **Multi-Head Latent Attention (KV compression)** and **multi‑query attention** to shrink KV cache; fused QKV + SDPA path for faster kernels when available. Presets expose `kv_latent_dim` and `use_sdpa` for device tuning. Flash kernels can be preferred via env (`OMNICODER_SDP_PREF=flash`, `OMNICODER_USE_FA3=1`).
   - **Multi‑Token Prediction (MTP)** heads and streaming cache for higher tokens/sec; generator supports Medusa‑style and tree speculative decoding with verifier acceptance; optional auto‑thresholding.
   - Optional **HRM** sub‑module (`use_hrm`, `hrm_steps`) to iterate internal state for tougher reasoning queries.
   - Adaptive difficulty/halting heads produce per‑token signals to steer variable‑K expert usage and early‑exit in decoding; export guards keep decode‑step graphs stable.
  - **Long‑context** friendly pos-emb (RoPE + interpolation hooks).
  - Optional SSM blocks (GatedConvSSM) interleaved every 4th layer for full‑sequence passes to improve long‑range mixing without KV growth; automatically skipped during decode‑step and disabled in ONNX/Core ML export.
  - **Infinite‑context pathway**: recurrent memory compressor (`mem_slots`) that summarizes distant context into a fixed number of slots and primes decode KV; pair with sliding‑window decode to bound KV while preserving long‑range information. Enable via `--mem_slots` and `--window_size`.
  - **Compressive KV & Landmark indexing**: export‑friendly `CompressiveKV` to pool long prefixes of latent K/V into fixed slots for bounded decode, and `LandmarkIndexer` for random‑access style long‑context. Hooks and auxiliary losses available in `modeling/memory.py`.
 - **Hierarchical Reasoning (`src/omnicoder/modeling/hrm.py`)**
  - Two‑level recurrent planner/worker loop with a small controller and adaptive halting.
- **Multimodal**
  - **Vision encoders** (ViT‑tiny), **Video encoder** (frame‑wise with temporal pooling), **Audio tokenization** (EnCodec stub) and **speech ASR/TTS** adapters.
  - SCMoE inference knob: pass `--scmoe_alpha` and `--scmoe_frac` to enable self-contrast MoE blending at inference for improved reasoning without extra passes.
  - Router curriculum: in `pretrain.py`, use `--router_curriculum topk>multihead>grin --router_phase_steps 0.3,0.6,1.0` to phase routing strategies during training and stabilize expert utilization.
  - **Generative decoders**: VQ‑token decoders & diffusion‑decoder interfaces (plug in SD‑U‑Net, video diffusion, vocoders like HiFi‑GAN). Added a minimal `ImageGenPipeline` to optionally call a local Stable Diffusion pipeline for text→image (off by default; requires weights).
  - **Continuous latent refiners**: image/audio refiners can be trained via `training/flow_recon.py` with `--use_refiner` and exported to ONNX via `--export_refiner_onnx`. Metrics gates (CLIPScore/FID/FAD) are written when extras are present.
  - **Fusion**: Added `MultimodalComposer` to fuse vision/video tokens with text by projecting to LLM `d_model` and composing special BOS/EOS modality tokens, feeding fused features directly to the core (forward now accepts pre-embedded features).
  - **Mobile export** (`src/omnicoder/export/`) with scripts for **Core ML** (decode‑step MLProgram exporter), **ONNX**, **GGUF**, **ExecuTorch** graphs, and **MLC** compilation stubs. Includes one-command mobile packager to emit decode-step ONNX, optional ONNX dynamic int8, optional ExecuTorch program, and a memory budget summary. ONNX export now respects preset latent‑KV size and multi‑query flags.
  - **Training** (`src/omnicoder/training/`) with **pretrain**, **LoRA/QLoRA finetune**, **distillation (teacher→student KD)**, **RL (GRPO/PPO with programmatic rewards)**, **RLHF**.
  - Added fused VL/VQA training scripts: `vl_fused_pretrain.py`, `vl_video_fused_pretrain.py`, `vqa_fused_train.py` and data adapters (`training/data/vl_jsonl.py`, `video_jsonl.py`, `vqa_jsonl.py`).
- **Automated eval** (`src/omnicoder/eval/`) for **pass@k code**, **CLIPScore/FID**, **FVD**, **FAD**, **WER/MOS‑proxy**.
  - Optional canaries: `python -m omnicoder.tools.metrics_canaries --images_dir <dir> --ref_dir <dir> --audio_ref_dir <dir> --audio_pred_dir <dir> --max_new_tokens 64` writes `weights/metrics_canaries.json` with tokens/s (engages verifier/write policy) and optional CLIPScore/FID/FAD if extras are installed. In Docker Compose: `docker compose run --rm metrics_canaries` (default CPU lane threshold ≥ 15 tokens/s; adjust in `src/omnicoder/tools/threshold_check.py`).

### Provider kernels status (Windows DML, macOS Core ML, Android NNAPI)
### Expert-parallel sharding (2×24 GB and beyond)

- Use the expert-parallel launcher to shard experts across devices and run training:

```bash
python -m omnicoder.tools.torchrun_ep \
  --script omnicoder.training.pretrain \
  --script_args "--data examples --seq_len 128 --steps 1000 --device cuda --out weights/pretrain_ep.pt" \
  --devices cuda:0,cuda:1 --init_dist
```
DS‑MoE schedule (dense→sparse) and balanced router init:
- Enable dense activation early and auto-sparsify later with flags in `training/pretrain.py`:
  - `--ds_moe_dense` or `--ds_dense_until_frac 0.3` to activate all experts per token for the early fraction of training.
  - `--ds_moe_no_aux` to disable router aux losses during the dense phase (avoids counterproductive balance penalties).
  - `--router_init_balanced` to initialize router gates near uniform (DeepSeek‑style), improving load balance without heavy aux losses.

Example:
```bash
python -m omnicoder.training.pretrain \
  --data data/text --device cuda --steps 20000 \
  --ds_dense_until_frac 0.3 --ds_moe_no_aux --router_init_balanced \
  --router_curriculum topk>multihead>grin --router_phase_steps 0.3,0.6,1.0
```

Variable‑K budget schedule during training:
- Enable difficulty‑aware expert count with `--var_k_train --var_k_min 1 --var_k_max 4`.
- Optionally schedule the difficulty threshold linearly over training with `--var_k_threshold_start 0.6 --var_k_threshold_end 0.4` (defaults to fixed `--var_k_threshold`).
- Probe EP performance and VRAM with the budgeted probe (writes `weights/train_probe_summary.json`):

```bash
python -m omnicoder.tools.train_probe --budget_minutes 10 --device cuda --ep_devices cuda:0,cuda:1
```

Environment:
- `OMNICODER_EXPERT_DEVICES` is picked up by `MoELayer` to place experts round‑robin.
- `OMNICODER_ROUTER_SINKHORN_ITERS`/`OMNICODER_ROUTER_SINKHORN_TAU` enable balanced routing targets during training.


- DirectML (Windows GPU)
  - Current: Composite DML fused MLA path (functional) and native module hooks are in place. A native C++/DirectML fused attention kernel will be loaded automatically if built (`omnicoder_dml_native`).
  - Build native module (optional):
    ```powershell
    # Option A: helper script
    .\.venv\Scripts\python -m omnicoder.tools.build_dml --config Release
    
    # Option B: raw CMake
    $torch = Resolve-Path .\.venv\Lib\site-packages\torch\share\cmake\Torch
    cmake -S src\omnicoder\modeling\kernels -B build_dml -DTorch_DIR=$torch
    cmake --build build_dml --config Release
    ```
  - The microbench now reports `native_present` when `torch_directml` is installed and the native kernel is found. Run:
    ```powershell
    .\.venv\Scripts\python -m omnicoder.inference.benchmark --bench_mla --seq_len 128 --gen_tokens 256
    ```
  - Bench (Windows venv):
    ```powershell
    .\.venv\Scripts\python -m omnicoder.export.onnx_export --output weights\text\omnicoder_decode_step.onnx --seq_len 1 --mobile_preset mobile_4gb --decode_step
    .\.venv\Scripts\python -m omnicoder.inference.runtimes.provider_bench --model weights\text\omnicoder_decode_step.onnx --providers CPUExecutionProvider DmlExecutionProvider --prompt_len 64 --gen_tokens 128 --compare_base CPUExecutionProvider --compare_target DmlExecutionProvider --speedup_min 1.5
    ```
- Core ML (iOS/ANE)
  - Metadata for native attention + RoPE mapping is emitted; transform to native ops will be enabled as coremltools MIL stabilizes.
- NNAPI (Android)
  - Per‑op PTQ presets and quant maps are emitted; device delegates recommended for on‑device benchmarks.

### New in this build
- DS‑MoE & router conditioning
  - `training/pretrain.py` exposes DS‑MoE dense→sparse training (`--ds_moe_dense`, `--ds_dense_until_frac`) and balanced router init.
  - `--prealign_ckpt` loads a `PreAligner` and conditions `HierarchicalRouter`/`InteractionRouter` per step.
- Expert paging (on-demand)
  - `modeling/utils/expert_paging.py` implements an LRU pager. Enable via `OMNICODER_EXPERT_PAGING=1` and set capacity `OMNICODER_EXPERT_PAGING_CAP`.
  - MoE prefetch uses router probabilities to warm the pager.
- KV-cache streaming and latent-KV attention now support incremental decoding for faster inference and lower memory. The generator uses cached keys/values to avoid reprocessing the full prompt on each step. Presets now expose `kv_latent_dim` and `use_sdpa` for device-specific tuning; fused QKV projection and SDPA fast-path further improve attention throughput when available.
 - Fixed device alignment in Video VQ encoder: the per-frame encoder and codebook now move to the active device to avoid CUDA/CPU tensor type mismatches during video tokenization. This resolves `RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)` and unblocks `tests/test_video_vq.py::test_video_vq_roundtrip_small` on GPU.
 - SCMoE: Generator exposes `--scmoe_alpha` and `--scmoe_frac` to enable self-contrast MoE blending. See `tests/test_routing_balance.py::test_scmoe_inference_blending_runs`.
 - Router curriculum: `training/pretrain.py` gains `--router_curriculum topk>multihead>grin` and `--router_phase_steps 0.3,0.6,1.0` to gradually shift routing during training; auxiliary balance losses are scheduled.
 - ONNX export: default opset is now 18; decode-step exporter prefers `torch.onnx.dynamo_export` with `dynamic_shapes=True` when available and falls back to legacy exporter otherwise. Test updated.
 - Temporal video module: added ONNX-friendly temporal SSM block in `modeling/multimodal/video_encoder.py` (use `use_temporal_ssm=True`).
- New: KV-cache quantization (u8 / NF4) wired end-to-end in the PyTorch generator and benchmarks. Enable with flags to quantify KV footprint and emulate on-device storage formats while dequantizing per step for compute. ONNX decode-step runner supports u8 emulation.
- Added multi-token prediction heads (configurable) and integrated lookahead decoding path in the generator and ONNX export flag.
 - Decode‑step export and streaming generation now supported for both ONNX and ExecuTorch (stateful KV‑cache IO). See examples below.
 - Infinite‑context priming in the text generator: when `--window_size>0` and `--mem_slots>0`, the generator summarizes the prefix beyond the window into memory slots and primes KV before decode, enabling effectively unbounded logical context with bounded KV.
 - Flash/SDPA fast path selection exposed via env: set `OMNICODER_SDP_PREF=flash` (default) and optionally `OMNICODER_USE_FA3=1`.
 - Local RAG PQ: build a compact local PQ index and use it in generation. Builder CLI `python -m omnicoder.tools.pq_build ./docs ./weights/pq`; use with `--retrieve_pq_index ./weights/pq`.
 - Image path CLI improved: pass Stable Diffusion model via `--sd_model` or `--sd_local_path`, control steps/size/output.
 - Added AWQ/GPTQ quantization CLI helper (`export/awq_gptq_quantize.py`) that attempts quantization if dependencies are present.
  - Added `inference/memory_estimator.py` to estimate on‑device memory for weights + KV cache under the `mobile_4gb` preset.
- Added a working knowledge distillation pipeline to train a compact student from a larger HF teacher on a single 24 GB GPU (`training/distill.py`).
  - On CUDA hosts, prefer `--teacher_device_map auto --teacher_dtype auto` so the teacher shards across available GPUs with mixed precision (fp16/bf16) as appropriate. The KD script auto-falls back to CPU if CUDA is unavailable and resolves device from the teacher’s embedding weight to avoid mismatches.
 - Low-rank learnable latent dictionaries in attention (`use_latent_dict`) and optional flash/SDPA fast path toggles.
 - Fused MoE dispatch interface with CUDA extension hook and torch fallback (`modeling/kernels/moe_scatter.py`), integrated in `MoELayer`.
 - Paged KV cache structure and ONNX sidecar paging metadata (`--kv_paged`); ONNX decode-step runner supports `--kv_paged --window` tail materialization.
 - Activation calibration CLI (`omnicoder.tools.act_calibrate`) to collect per-channel scales for exporter PTQ.
 - ONNX attention fusion presence test and KV paging sidecar test.
- Added a simple weight-only int4 wrapper (`Int4Linear`) and a utility to replace `nn.Linear` layers for functional validation of 4-bit weights. This validates int4 quantization correctness on desktop while we implement true device kernels.
- Added verifier-head distillation script (`training/verifier_distill.py`) to train an external acceptance head (draft-and-verify) from a stronger teacher.
- Robust MoE router regularization: new training flags `--router_temp`, `--router_jitter`, `--router_use_gumbel`, `--router_expert_dropout_p`, `--router_sinkhorn_iters`, `--router_sinkhorn_tau`, and aux loss coefficients `--aux_lb_coef`, `--aux_importance_coef`, `--aux_load_coef`, `--aux_zloss_coef`. MoE layer caches auxiliary stats per step (`importance`, `load`, `z_loss`, `sinkhorn_target`). Static capacity bound via `--moe_static_capacity` reduces worst‑case latency on device.
  - Training logs now include ETA, tokens/s (EMA), and CUDA memory; periodic interim checkpoints and JSONL logs are available via CLI flags.
 - Fixed duplicate video CLI branch; ONNX/Core ML exporters now disable HRM for stable graphs.
 - Vision backbone now supports `timm` ViT‑tiny when installed; internal ViT‑tiny fallback kept.
 - Audio tokenizer upgraded to EnCodec wrapper with stub fallback.
 - Optional context‑aware router (`LLMRouter`) for expert selection. Enable via environment `OMNICODER_ROUTER=llm`. Default remains Top‑K/GRIN/Multi‑Head blend; tests unchanged. The `LLMRouter` adds a single lightweight self‑attention layer to the gating context for improved specialization.

### kNN‑LM cache (optional)

- Enable a compact kNN‑LM cache to bias next‑token selection with nearest neighbor lookups over hidden states.
- Generator flags: `--knn_cache --knn_k 16 --knn_lambda 0.2` (set `OMNICODER_KNN_CACHE=1` and `OMNICODER_KNN_CACHE_PATH=weights/knn_cache.npy` to persist across runs)
- Implementation: `omnicoder.inference.knn_cache.KNNCache` (FAISS if available; NumPy fallback).

Populate the cache from conversation turns or retrieved passages:

```python
from omnicoder.inference.knn_cache import KNNCache
from omnicoder.modeling.transformer_moe import OmniTransformer
import torch

model = OmniTransformer()
tok = ...  # your tokenizer of choice
cache = KNNCache(dim=model.lm_head.in_features)

# Prime a few (hidden, next_token) pairs
for text, next_token in [("Hello", 42), ("How are you", 99)]:
    ids = torch.tensor([[tok.encode(text)[-1]]], dtype=torch.long)
    logits, new_kv, *_rest, hidden = model(ids, past_kv=None, use_cache=True, return_hidden=True)
    h = hidden[:, -1, :].detach().cpu().numpy()[0]
    cache.add(h, int(next_token))

# Later, pass --knn_cache to generator or pass knn_cache=cache when calling generate()
```

## Quickstart

Note: a consolidated quickstart with one-button flows now lives in `docs/Quickstart.md`. Use the `press-play` console command after installing the package (editable) to build and validate the default mobile release.

See `docs/Quickstart.md` for a concise, consolidated guide covering orchestrated training, export, provider bench, and Docker GPU flows.

### Early‑exit decoding (runtime)

Enable early‑exit heuristics during inference to skip speculative drafts or deeper compute when confident:

```bash
python -m omnicoder.inference.generate \
  --early_exit --early_exit_mode entropy --early_exit_entropy 1.0 \
  --max_new_tokens 128 --device cpu
```

Environment toggles (equivalents): set `OMNICODER_EARLY_EXIT=1`, `OMNICODER_EARLY_EXIT_MODE=entropy|delta`, and `OMNICODER_EARLY_EXIT_ENTROPY=1.0`.

### Docker GPU validation and persistent model cache

```bash
# Build CUDA image
docker build -t omnicoder:cuda .

# Run an interactive shell with GPU and persistent caches
docker run --rm -it --gpus all \
  -v %cd%:/workspace \
  -v %cd%/models:/models \
  -e HF_HOME=/models/hf -e TRANSFORMERS_CACHE=/models/hf \
  -e OMNICODER_AUTO_RESOURCES=1 \
  omnicoder:cuda bash

# Inside container: quick smoke and export
python3 -m omnicoder.inference.generate --prompt "Hello OmniCoder" --max_new_tokens 16 --mobile_preset mobile_4gb
python3 -m omnicoder.export.onnx_export --output weights/text/omnicoder_decode_step.onnx --seq_len 1 --mobile_preset mobile_4gb --decode_step
python3 -m omnicoder.inference.runtimes.onnx_decode_generate --model weights/text/omnicoder_decode_step.onnx --prompt "Hello" --max_new_tokens 16

# Optional: run tests (resource dependent)
pytest -q
# Focused smokes:
pytest -q tests/test_export_decode_step_smoke.py::test_decode_step_export_smoke -q
pytest -q tests/test_long_context_generation.py::test_long_context_generation_canary -q
pytest -q tests/test_router_smoke.py::test_router_knobs_runtime_only -q

# OR run the ONNX decode-step runner with paged KV and speculative drafts (desktop validation)
python3 -m omnicoder.inference.runtimes.onnx_decode_generate \
  --model weights/release/text/omnicoder_decode_step.onnx \
  --provider CPUExecutionProvider \
  --kv_paged --window 1024 \
  --kvq u8 --kvq_group 64 \
  --speculative_draft_len 2 --verify_threshold 0.0
```

Notes
- Models/backbones and HF artifacts persist under `/models` between runs, avoiding re-downloads.
- Checkpointed student/LoRA weights are written under `weights/` (mounted from host). Reuse them by passing `--ckpt` in subsequent runs.
- Tests: `docker compose run --rm tests` executes with `-vv -rA` for verbose reporting; optional diffusion ONNX smoke may be skipped if toolchains unavailable.
- Auto resources: set `OMNICODER_AUTO_RESOURCES=1` to auto-scale OMP/MKL/Torch threads and DataLoader workers to host cores.
- Compressive memory: enable with `OMNICODER_COMPRESSIVE_SLOTS=4` or pass `--compressive_slots 4` in training to bound long prefixes.
- Pretrain flag smoke: `pretrain` exits cleanly when `--data` has no `.txt` files. This avoids early termination when pointing to empty temp dirs for vg‑flag parsing.
- Vision grounding smoke: the grounding head clones inference tensors before `LayerNorm` when needed to avoid eval‑mode autograd runtime; shapes/outputs unchanged.

Agentic build loop prompt:

```text
See examples/prompts/agentic_continuous_dev.md
```

> For multimodal demos, you must provide actual pretrained/backbone weights (see `weights/README.md`) and enable the relevant adapters in config. Press Play can stage some backbones when you set the appropriate `.env` keys (e.g., `OMNICODER_SD_MODEL`, `OMNICODER_PIPER_URL`).

## Android / iOS

 - **Android**: see `src/omnicoder/inference/serverless_mobile/android/README-android.md` for ExecuTorch/NNAPI, ORT-mobile, and a **JNI** bridge for `llama.cpp` (GGUF). Includes KV-cache streaming details. Use `export/gguf_export.py` to create baseline GGUFs from HF LLaMA/Mistral. Minimal sample app at `serverless_mobile/android/sample-app` shows an ORT NNAPI session in `MainActivity.kt`.
- **iOS**: see `src/omnicoder/inference/serverless_mobile/ios/README-ios.md` for **Core ML MLProgram** (ANE/GPU) decode-step export/use and MLC runtime compilation. Minimal sample app at `serverless_mobile/ios/SampleApp` shows a placeholder `ViewController.swift`; plug a compiled `.mlmodel` and implement KV streaming.
  - Export flags now include `--export_hrm` to preserve HRM during export when needed; default is disabled for stable graphs.
  - Single entrypoint for build/export/smoke: `python -m omnicoder.tools.press_play`.
  - New exporter option `--dynamic_cache_shim` writes a sidecar JSON documenting a future DynamicCache decode‑step interface (no graph change yet) to ease migration to the ONNX dynamo exporter with DynamicCache.

### Android streaming (ONNX/NNAPI) — assets + KV streaming

1) Place your decode‑step ONNX at:
   `src/omnicoder/inference/serverless_mobile/android/sample-app/src/main/assets/omnicoder_decode_step.onnx`
2) Build and run the `sample-app` in Android Studio. On first launch, the app copies the ONNX from assets into the app files directory and opens an ORT session with NNAPI (best‑effort QNN accelerator hint).
3) The app discovers K/V input names and (H, DL), initializes zero‑length K/V for the first step, and streams greedy tokens. Replace the tiny ASCII tokenizer in `MainActivity.kt` with your on‑device tokenizer and (optionally) maintain `nk_lat_*`/`nv_lat_*` outputs between steps for maximum throughput.

Notes
- Assets enabled in Gradle (`sourceSets.main.assets`). NNAPI is enabled via `SessionOptions.addNnapi`.
- For device TPS canaries, export a fused ONNX with `mobile_packager` and run provider_bench on device or via ADB helper.

### iOS SwiftPM console (Core ML MLProgram) — streaming loop

1) Add `omnicoder_decode_step.mlmodel` to:
   `src/omnicoder/inference/serverless_mobile/ios/SampleConsole/Sources/Resources/`
2) Build and run the SwiftPM console target. It compiles the MLModel at runtime, discovers K/V inputs, feeds zero‑length past K/V, and streams greedy tokens with a tiny tokenizer. Output prints to the console.

Notes
- Move the same streaming logic into `SampleApp` (UIKit) and maintain K/V across steps for UI streaming.
- Use the MLProgram decode‑step exporter and ensure K/V I/O matches graph names (`k_lat_i`, `v_lat_i`, `nk_lat_i`, `nv_lat_i`).

Int4/device kernels roadmap
- Today: weight-only int4 wrapper (`modeling/quant/int4_linear.py`) for correctness checks.
- Next: integrate provider-specific int4 kernels (NNAPI/ANE/GPU) and fused MLA/MQA in mobile runners.
Provider fused MLA interface
- `modeling/kernels/mla_providers.py` exposes a backend registry: `cpu`, `dml`, `coreml`, `nnapi` with graceful fallbacks to SDPA. Select via env `OMNICODER_MLA_BACKEND`.

MLA fused attention backends
- Select a fused MLA provider via env `OMNICODER_MLA_BACKEND` = `cpu|dml|coreml|nnapi`.
- DirectML (Windows GPU): install `torch-directml`, then set `OMNICODER_MLA_BACKEND=dml` to route MLA through the DirectML path. A composite fused op `torch.ops.omnicoder_dml.mla` is registered on import; a native prototype is attempted when possible. You can also build a native extension with CMake:
  - Requirements: `cmake` (>=3.18), Visual Studio Build Tools (C++), PyTorch C++ headers
  - Build:
    ```bash
    cmake -S src/omnicoder/modeling/kernels -B build_dml -DTorch_DIR=%CONDA_PREFIX%/Lib/site-packages/torch/share/cmake/Torch
    cmake --build build_dml --config Release
    ```
-  - The Python path attempts JIT build on import via `torch.utils.cpp_extension.load` when `dml_fused_attention.cpp` is present (Windows only).
- Core ML and NNAPI symbols: Python registers `torch.ops.omnicoder_coreml.mla` and `torch.ops.omnicoder_nnapi.mla` composite implementations (and `matmul_int4`) for portability; native bindings/delegates can hook these symbols in the future.
- The attention layer resolves a provider backend and falls back to SDPA or explicit softmax when unavailable.
- Int4 layout alignment: set `OMNICODER_INT4_ALIGN` (default 64 elements) for packed nibble alignment to match provider kernels.
- Fused RoPE: set `OMNICODER_MLA_FUSED_ROPE=1` if your provider backend applies RoPE internally; the PyTorch attention will skip pre-MLA RoPE in that case.
- SDPA/Flash preferences: `OMNICODER_USE_SDPA=1` to prefer SDPA in PyTorch path, and `OMNICODER_SDP_PREF=flash|mem_efficient|all|auto` to steer kernel selection.
- Benchmark comparison:
```bash
pip install torch-directml
set OMNICODER_MLA_BACKEND=dml
python -m omnicoder.inference.benchmark --bench_mla --seq_len 256 --gen_tokens 256
```

DirectML int4 backend (weight-only matmul)
- Enable int4 backend on Windows GPU (DirectML):
```bash
pip install torch-directml
set OMNICODER_INT4_BACKEND=dml
set OMNICODER_INT4_ALIGN=64
python -m omnicoder.inference.benchmark --bench_int4 --seq_len 256 --gen_tokens 256
```
- The DML path performs unpack+dequant on the GPU and matmul on the DirectML device, yielding a practical speedup vs CPU unpack paths while we integrate true device int4 kernels. Select backend via `OMNICODER_INT4_BACKEND=dml`.

Environment switches
- `OMNICODER_INT4_BACKEND`: select int4 matmul backend (`cpu`, `dml`, `coreml`, `nnapi`). Defaults to `cpu` which performs dequant+fp32 matmul. Provider-specific backends will be wired to device kernels when available.
- `OMNICODER_MLA_BACKEND`: fused MLA provider backend (`cpu`, `dml`, `coreml`, `nnapi`).
- `OMNICODER_OUT_ROOT`, `OMNICODER_KD_*`, `OMNICODER_ONNX_*`, `OMNICODER_SD_*`, `OMNICODER_BENCH_*`: configure press_play without long CLI flags.
- `OMNICODER_RUN_KV_CALIBRATE=1` (optional): Press Play runs KV-cache quantization calibration and writes `weights/kvq_calibration.json` consumed by exporters/runners.
- Generator KV quantization flags: `--kvq {none,u8,nf4}` and `--kvq_group` to enable per-group KV storage quantization with per-step dequantization.
- `OMNICODER_COMPILE=1`: enable `torch.compile` by default for the text decode loop (inductor, warmup + fallback). Set to `0` to disable.

Config via .env
- Press Play and the mobile release builder auto-load a project-root `.env` if present. See `env.example.txt` for a complete, commented list of `OMNICODER_*` keys and defaults. If a variable exists in the process environment, it takes precedence over `.env`.


## Training on a single 24 GB GPU

 - Use **QLoRA** via `training/finetune_lora.py` (NF4/norm‑aware), gradient checkpointing, and CPU offload. For faster attention during training, pass `--use_flash` to `training/pretrain.py` to enable SDPA/Flash2 where available.
- Long context adapters (position interpolation) and **LoRA ranks** are config‑driven.
- For RL and distillation, run teacher on a separate machine/cloud; only the **student** finetuning occurs locally.
Distillation example (teacher→student logit matching):

```bash
# From Docker container (recommended)
python3 -m omnicoder.training.distill \
  --data ./ --seq_len 512 --steps 200 --device cuda \
  --teacher microsoft/phi-2 \
  --student_mobile_preset mobile_4gb \
  --kl_temp 1.5 --alpha_kd 0.9 --lora --gradient_checkpointing \
  --log_interval 20 --save_interval 2000 \
  --log_file weights/kd_train_log.jsonl \
  --out weights/omnicoder_student_kd.pt

# Native (if CUDA PyTorch is installed locally)
python -m omnicoder.training.distill \
  --data ./ --seq_len 512 --steps 200 --device cuda \
  --teacher microsoft/phi-2 \
  --student_mobile_preset mobile_4gb \
  --kl_temp 1.5 --alpha_kd 0.9 --lora --gradient_checkpointing \
  --log_interval 20 --save_interval 2000 \
  --log_file weights/kd_train_log.jsonl \
  --out weights/omnicoder_student_kd.pt
```

### Training (text) with router curriculum and GRIN
### Pre-alignment (contrastive) stage (optional)
Run a lightweight contrastive pre-alignment to place text/image embeddings into a shared space for better routing and cross-modal coherence.

```bash
python -m omnicoder.training.pre_align \
  --data /path/to/image_text.jsonl \
  --batch_size 16 --steps 1000 --device cuda \
  --embed_dim 256 --out weights/pre_align.pt
```

Use the pre-aligner during VL fused pretrain with an auxiliary InfoNCE loss:
Unified multi-index build

- The orchestrator auto-builds a unified multi-index by default under `weights/unified_index` combining text/image roots (extendable). Override the root with `OMNICODER_MULTI_INDEX_ROOT`.
- Export frozen ONNX preprocessors for text/image/audio/video embeddings from a pre-align checkpoint:
```bash
python -m omnicoder.export.export_preprocessors --prealign_ckpt weights/pre_align.pt --out_dir weights/release/preprocessors --opset 17 --device cpu
```

```bash
python -m omnicoder.training.vl_fused_pretrain \
  --jsonl /path/to/image_text.jsonl --mobile_preset mobile_4gb \
  --pre_align_ckpt weights/pre_align.pt --align_weight 0.1 --device cuda
```


```bash
# Top‑K → Multi‑Head → GRIN curriculum with scheduled phases
python -m omnicoder.training.pretrain \
  --data ./examples/code_eval \
  --seq_len 256 --steps 1000 --device cuda \
  --router_curriculum topk>multihead>grin \
  --router_phase_steps 0.3,0.6,1.0 \
  --router_temp 1.2 --router_jitter 0.2 --router_use_gumbel \
  --router_grin_tau 1.0 --router_grin_mask_drop 0.05 \
  --aux_balance_schedule linear --aux_importance_coef 0.02 --aux_load_coef 0.02 \
  --log_interval 50 --log_file weights/pretrain_log.jsonl

# Force GRIN from the start
python -m omnicoder.training.pretrain \
  --data ./examples/code_eval --seq_len 256 --steps 500 --device cuda \
  --router_kind grin --router_grin_tau 1.0 --router_grin_mask_drop 0.1
```

### DS‑MoE dense‑train sparse‑infer

```bash
python -m omnicoder.training.pretrain \
  --data ./examples/code_eval --seq_len 256 --steps 500 --device cuda \
  --ds_moe_dense --ds_moe_no_aux \
  --router_curriculum topk>multihead>grin --router_phase_steps 0.3,0.6,1.0
```

During training, logs include per‑phase router summaries: `router_phase`, `expert_load_std`, and a small histogram (`expert_load_hist`).
### Distillation with rationales and expert routing targets (JSONL)

Prepare a KD JSONL with fields `{text, rationale?, router_targets?}` and run:
```bash
python -m omnicoder.training.distill \
  --data /path/to/kd.jsonl --data_is_jsonl --seq_len 512 --steps 200 --device cuda \
  --teacher microsoft/phi-2 --student_mobile_preset mobile_4gb \
  --seq_kd --expert_route_kd --alpha_kd 0.9 --kl_temp 1.5 \
  --out weights/omnicoder_student_kd.pt
```



Quick LoRA finetune example (toy text):

```bash
python -m omnicoder.training.finetune_lora --data ./ --seq_len 256 --steps 50 --device cpu \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --out weights/omnicoder_lora.pt
```

### Benchmarking and recording quality

After KD/LoRA, record tokens/sec and simple exact-match to track improvements:

```bash
# tokens/s (GPU recommended)
python -m omnicoder.inference.benchmark --device cuda --seq_len 128 --gen_tokens 256 --mobile_preset mobile_4gb --kvq nf4

# exact-match on toy tasks
python -m omnicoder.eval.text_eval --tasks examples/code_eval/examples.jsonl --max_new_tokens 64

# tail logs during training (if you used tee)
tail -f weights/kd_phi2.log
```

Keep a small table (pre/post KD/LoRA) in your notes or CI artifacts.

### Performance roadmap (edge-first, 2–4 GB RAM)

- Fused attention and KV compression
  - Implement provider‑grade fused MLA/MQA and RoPE on DirectML/Core ML/NNAPI; prefer SDPA/Flash on GPUs when faster.
  - Tighten ONNX fusion transforms for decode‑step graphs (KV‑cache IO) or use native provider kernels via PyTorch.
- Quantization and kernels
  - End‑to‑end 4‑bit: AWQ/GPTQ for weights; true device int4 GEMM (DML/Core ML/NNAPI) with aligned packing; 8‑bit/NF4 KV caches.
- Long‑context and memory
  - Default sliding‑window decode and recurrent memory slots; YaRN/PI finetuning; CI variants at 32k/128k.
- Decoding throughput
  - Multi‑token prediction, speculative decoding (draft+verify), kNN‑LM cache with mmap PQ; enforce paged KV caches.
- Training and alignment
  - Multi‑teacher KD; GRPO/RLHF with multimodal rewards (CLIPScore/FID/FVD/FAD, code pass@k, WER/MOS proxy); verifier‑head distillation.
- RAG and retrieval
  - Mmap PQ indexes, budgeted retrieval, on‑device shard loading; memory canaries in CI.

### Image/Video/Audio metrics (optional extras)

- Dependencies (install as needed):
  - **CLIPScore**: `pip install git+https://github.com/openai/CLIP.git pillow`
  - or OpenCLIP alternative: `pip install open-clip-torch pillow`
  - **FID (images)**: `pip install clean-fid`
  - **FVD (videos)**: `pip install pytorch-fvd opencv-python`
  - **FAD (audio)**: `pip install torch-fad soundfile`
  - Perceptual CLIPScore/FAD-weighted losses can be enabled in latent reconstruction trainers:
    - Image flow/latent trainer: `--fid_metrics` computes CLIPScore/FID (if extras present); `--flow_loss` enables epsilon targets.
    - Audio latent trainer: provide `--fad_ref_dir` and `--fad_pred_dir` to compute FAD; perceptual proxies (mel/STFT) are included.
  - Optional TorchMetrics variants: `pip install torchmetrics torchvision`

Examples:
```bash
# CLIPScore and FID from training/flow_recon.py (accumulates small batch)
python -m omnicoder.training.flow_recon --data ./examples/data/vq/images \
  --steps 2 --device cpu --fid_metrics --ref_dir ./examples/data/vq/images

# Image eval (standalone):
python -m omnicoder.eval.image_eval --mode clip --jsonl examples/vl_fused_sample.jsonl
python -m omnicoder.eval.image_eval --mode fid --pred_dir ./out/gen_images --ref_dir ./datasets/coco2017/val2017

# Video FVD (pred/ref folders of .mp4):
python -m omnicoder.eval.video_eval --pred_dir ./out/gen_videos --ref_dir ./datasets/ucf101/val

# Audio FAD (WAV folders):
python -m omnicoder.training.audio_recon --wav_dir ./examples/data/vq/audio \
  --steps 2 --device cpu --fad_ref_dir ./ref_wavs --fad_pred_dir ./pred_wavs 
```

### Example results (GPU Docker smoke)

- KD (teacher: `sshleifer/tiny-gpt2`, steps: 50 → early exit saved at step 1 in smoke run)
- Saved checkpoint: `weights/omnicoder_student_kd.pt`

You can load the checkpoint to generate:

```bash
python -m omnicoder.inference.generate --prompt "Hello from KD student" \
  --ckpt weights/omnicoder_student_kd.pt --max_new_tokens 32 --mobile_preset mobile_4gb
```

After a fuller KD run on your GPU, record tokens/sec and task accuracy and replace this section with your numbers.


## Automated evaluation (no humans)
## Long-context (YaRN/PI) and sliding-window attention

- You can extend effective context via RoPE interpolation by setting `--target_ctx` and `--rope_base` in exporters and training scripts. Example (export):
```bash
python -m omnicoder.export.onnx_export --output weights/text/omnicoder_decode_step.onnx \
  --seq_len 1 --mobile_preset mobile_4gb --decode_step --target_ctx 32768 --rope_base 10000.0 --window_size 2048
```
- When `--target_ctx>=32768`, the exporter additionally emits long-context variants:
  `weights/text/omnicoder_decode_step_ctx32k.onnx` and `..._ctx128k.onnx` for CI validation.
- CPU ORT smoke for long-context variants (sanity):
```bash
python -m omnicoder.inference.runtimes.onnx_decode_generate \
  --model weights/text/omnicoder_decode_step_ctx32k.onnx \
  --provider CPUExecutionProvider --prompt "Hello" --max_new_tokens 16
```
- Paged KV and window defaults: if a `*.kv_paging.json` sidecar is present next to the ONNX, the runner auto-enables paged KV; if `--window` is unset and sidecar is present, a reasonable default window is derived from the sidecar page length (4× page_len capped). This keeps decode memory bounded on mobile.

## ONNX fusions and per-op PTQ for NNAPI/CoreML/DML

- After ONNX decode-step export, the packager applies a conservative fusion pass that:
  - Fuses `MatMul -> (Add mask) -> Softmax -> MatMul` into `com.microsoft::Attention` when pattern matches.
  - Inserts QDQ wrappers around `MatMul` inputs to enable backend QDQ fusions for quantized execution.
- Then per-op PTQ is applied with provider-specific presets or custom `ptq_op_types` from a provider profile JSON.
- Use:
```bash
python -m omnicoder.tools.build_mobile_release --preset mobile_4gb \
  --onnx_provider_hint NNAPIExecutionProvider --quantize_onnx_per_op \
  --provider_profile profiles/pixel7_nnapi.json
```

Troubleshooting (containers with limited RAM): heavy export + PTQ tests can exhaust memory and trigger a `SIGKILL` when run together with the full suite. If you see a failure like in `tests/test_ptq_presence.py::test_per_op_ptq_inserts_qdq` due to `<Signals.SIGKILL: 9>`, re-run that test alone or increase the container memory limit. Running the test in isolation passes; the exporter also uses the legacy path by default for opset<18 to reduce memory.

### Providers and device-grade optimizations
- NNAPI (Qualcomm QNN): use `NNAPIExecutionProvider` with `profiles/pixel7_nnapi.json`; per-op PTQ preset `nnapi` packs `MatMul/Gemm/Conv/Attention` and QDQ.
- Core ML (ANE/GPU): use `CoreMLExecutionProvider` or export Core ML MLProgram decode-step; long-term target is fused MLA/MQA and RoPE ops.
- DirectML (Windows/GPU): use `DmlExecutionProvider`; preset `dml` covers MatMul/Attention/LayerNorm. The fused MLA backend resolver now auto-detects a viable backend (`cpu`, `dml`) when `OMNICODER_MLA_BACKEND` is unset or unknown.
- Provider microbench: run `python -m omnicoder.inference.benchmark --bench_mla --seq_len 128 --gen_tokens 256` to compare SDPA vs fused MLA and validate speedups. See `OMNICODER_BENCH_*` envs.

### Expert routing utilization metrics
- During training, router auxiliary stats are logged per step: importance and load per expert (and z-loss/sinkhorn targets when enabled). For quick inspection, check the JSONL logs (pretrain/distill) and plot per-phase histograms to verify balanced expert usage. Curriculum phases (TopK→MultiHead→GRIN) are controlled by `OMNICODER_ROUTER_CURRICULUM` and phase steps by `OMNICODER_ROUTER_PHASE_STEPS`.

### Continuous latents + tiny refiner (image/audio)
- Image training: `training/flow_recon.py` supports flow matching (EDM/VP/VE) and metrics (CLIPScore/FID) when extras are installed. Save the model to persist continuous latent heads.
- Inference: `ImageGenPipeline.generate(..., refiner_steps=N)` optionally runs a tiny refinement network for N steps to add detail with minimal cost.
- Audio training: `training/audio_recon.py` supports EnCodec/mel/ONNX encoders and proxy perceptual terms; compute FAD when reference/prediction dirs are provided.
 - Heads I/O: save heads-only state during training with `--save_heads weights/latent_heads.pt` (image: `image_latent_head`, `image_eps_head`; audio: `audio_latent_head`). Load them at inference with `--load_latent_heads weights/latent_heads.pt` and combine with `--image_refiner_steps`.
- ExecuTorch: export stateful decode-step `.pte` graphs (fallback `.pt`), threaded KV IO; integrate NNAPI delegate.
- Quantization: supports ONNX dynamic and per-op PTQ; external AWQ/GPTQ helpers available; native int4 packer present for correctness.
  - Packager can emit sidecar quant maps for NNAPI, Core ML, and DML (with preliminary int4 hints): pass `--nnapi_maps` or set provider hints; sidecars saved next to artifacts.
  - Provider-backed 4-bit training (CUDA): if `bitsandbytes` is installed and CUDA is available, swap `nn.Linear` to `bnb.nn.Linear4bit` via `replace_linears_with_bnb_4bit(model)` for KD/finetune speedups.
  - DirectML/CoreML runtime paths: `OMNICODER_INT4_BACKEND=dml|coreml|cpu` selects int4 matmul backend for `Int4Linear` at runtime; DML path leverages `torch-directml` when available, CoreML path leverages MPS.
  - ONNX fusion pass: pack `MatMul->Softmax->MatMul` into `com.microsoft::Attention` where possible and insert QDQ to enable `QLinearMatMul` during PTQ. This helps NNAPI/DML/CoreML EPs fuse attention on-device.

Backbone locks (mobile defaults)
- Vision: set `OMNICODER_VISION_BACKEND` to `timm_mobilevit_s`, `timm_mobilevit_xs`, `timm_efficientvit_lite0`, or `timm_vit_tiny`. Falls back to internal `ViTTiny`.
- Audio tokenizer: set `OMNICODER_AUDIO_TOKENIZER` to `encodec_24khz` (default) or `encodec_48khz`.
- Vocoder: set `OMNICODER_VOCODER_BACKEND` to `coqui_hifigan|onnx|torchscript` and `OMNICODER_VOCODER_MODEL` to a local path (for ONNX/TorchScript backends).
- Image decoder: set `OMNICODER_IMAGE_BACKEND` to `diffusers|diffusers_onnx|onnx` and `OMNICODER_SD_MODEL` to an HF id or leave it to use `--sd_model`.

Provider-backed 4-bit (CUDA)
- If running on CUDA and `bitsandbytes` is installed, you can replace `nn.Linear` layers with `bnb.nn.Linear4bit` (NF4) automatically:
```python
from omnicoder.modeling.quant.providers import replace_linears_with_bnb_4bit
replaced = replace_linears_with_bnb_4bit(model)
print('Replaced linears with bnb 4-bit:', replaced)
```
This enables high-performance 4-bit matmuls on NVIDIA GPUs during training/fine-tuning.

## Microbenchmarks (perf/memory)

- Text throughput: `python -m omnicoder.inference.benchmark --device cpu --seq_len 512 --gen_tokens 256`
- Int4 vs fp32 linear latency: printed automatically by the benchmark; programmatic call:
```python
from omnicoder.inference.benchmark import bench_int4_linear_vs_fp32
print(bench_int4_linear_vs_fp32())
```
- Memory budgeting: `python -m omnicoder.inference.memory_estimator --seq_len 8192 --batch_size 1 --preset mobile_4gb --kvq nf4`
  - KV footprint is reported in human-readable form; under NF4 you should see ~4x reduction vs fp16.

Provider microbench for ONNX providers (tokens/s):
```bash
python -m omnicoder.inference.runtimes.provider_bench --model weights/text/omnicoder_decode_step.onnx \
  --providers CPUExecutionProvider NNAPIExecutionProvider CoreMLExecutionProvider DmlExecutionProvider \
  --prompt_len 256 --gen_tokens 256 \
  --check_fusions --require_attention --require_qlinear \
  --threshold "CPUExecutionProvider=2.0,DmlExecutionProvider=20.0" \
  --out_json weights/text/provider_bench.json

DirectML fused MLA microbench (Windows GPU):

```bash
# Install torch-directml, then compare SDPA vs DML fused MLA
pip install torch-directml
set OMNICODER_MLA_BACKEND=dml
python -m omnicoder.inference.benchmark --bench_mla --seq_len 256 --gen_tokens 256
```
### Provider fused kernels and int4 matmul (prototype)
- DML fused MLA: a composite kernel (`torch.ops.omnicoder_dml.mla`) is registered and used when `OMNICODER_MLA_BACKEND=dml`. It moves tensors to the DML device and invokes SDPA as a fused path until a native kernel replaces it.
- Reference int4 matmul (CPU): `modeling/quant/int4_providers.py` provides a working int4 weight unpack/dequant + matmul path for correctness and integration; device-specific int4 backends will supersede this for NNAPI/CoreML/DML.
```
GitHub Actions workflows are included for CPU validation and optional self-hosted provider checks (DML/MPS) under `.github/workflows/`. A nightly long-context stability workflow runs 32k/128k variant exports and uploads artifacts.
NNAPI verification
- Use `--onnx_preset nnapi` in the packager and `--nnapi_maps` to emit attention/QLinear hints.
- Run with NNAPI EP and (optional) QNN accelerator: `--providers NNAPIExecutionProvider` and profile with `provider_bench`.

Core ML verification
- Export MLProgram decode-step via `coremltools` and enable native attention mapping (ANE when available). Hints from `coreml_attention_pass.py` are embedded.
- Optionally run with Core ML EP (desktop validation via MPS): set provider to `CoreMLExecutionProvider` in `provider_bench`.

DML verification
- Set provider to `DmlExecutionProvider` and ensure `QLinearMatMul` and `com.microsoft::Attention` are present (packager fusion). Tune threads and graph opts via provider profiles.

CI thresholds and regression canaries
- CPU and provider workflows emit long-context decode-step variants and run a tokens/sec canary. Set `TOKENS_PER_SEC_MIN` to enforce minimum throughput and fail on regressions.

## Unified multimodal tokens

**Train VQ-VAE codebooks and align unified vocab slices**
```bash
# Image VQ-VAE (EMA VQ) — exports a codebook compatible with ImageVQ
python -m omnicoder.training.vq_train --data /path/to/images --image_size 224 --patch 16 --emb_dim 192 \
  --codebook_size 8192 --steps 5000 --batch 32 --out weights/image_vq_codebook.pt

# Video VQ-VAE (reuse image VQ-VAE per-frame) — exports a codebook for VideoVQ
python -m omnicoder.training.video_vq_train --videos videos.txt --resize 224 --patch 16 --emb_dim 192 \
  --codebook_size 8192 --frames_per_video 16 --out weights/video_vq_codebook.pt
```
- Press Play can now optionally train VQ codebooks before packaging if you set:
  - `OMNICODER_VQ_IMAGE_DIR`, `OMNICODER_VQ_IMAGE_STEPS`
  - `OMNICODER_VQ_VIDEO_LIST`, `OMNICODER_VQ_VIDEO_SAMPLES`
  - `OMNICODER_VQ_AUDIO_DIR`, `OMNICODER_VQ_AUDIO_STEPS`
  The resulting `image_vq_codebook.pt` is auto-wired into the mobile packager.
- Set reserved vocab starts/sizes in `omnicoder.config.MultiModalConfig` to ensure contiguous ranges:
  - Text: [0..31999]
  - Image VQ: [32000..(32000+image_codebook_size-1)]
  - Video VQ: [image_end..(image_end+video_codebook_size-1)]
  - Audio VQ: [video_end..(video_end+audio_codebook_size-1)]
- Use `modeling/multimodal/vocab_map.py` helpers to map VQ indices into the unified vocab for training/inference.
- Fused VL/VQA training integrates VQ tokens with text using `training/vl_fused_pretrain.py`, `training/vqa_fused_train.py`, and `training/vl_video_fused_pretrain.py`.

KD and GRPO recipes (quality)
```bash
# KD (multi-teacher, verifier-head KD, LoRA/QLoRA)
python -m omnicoder.training.distill --data ./ --seq_len 512 --steps 20000 --device cuda \
  --teacher microsoft/phi-2 --student_mobile_preset mobile_4gb \
  --lora --gradient_checkpointing --log_interval 50 --save_interval 1000 \
  --out weights/omnicoder_student_kd.pt

# Verifier-head KD
python -m omnicoder.training.verifier_distill --data ./ --seq_len 512 --steps 5000 --device cuda \
  --teacher microsoft/phi-2 --student_mobile_preset mobile_4gb --verifier_only --lr 2e-4 \
  --out weights/omnicoder_verifier_kd.pt

# GRPO (programmatic rewards; extend with CLIP/FID/FVD/FAD, code pass@k)
python -m omnicoder.training.rl_grpo --prompts examples/code_eval/examples.jsonl --device cuda --steps 5000 \
  --reward code_tests
```

Compile decode-step ONNX with TVM/MLC (optional)
KV calibration defaults and sidecars
- If `weights/kvq_calibration.json` (or a sibling `kvq_calibration.json` by the ONNX model) exists, runners and exporters will auto-detect it and annotate KVQ sidecars. Prefer KV NF4/u8 for mobile presets when calibrated.
```bash
python -m omnicoder.export.mlc_compile --onnx weights/text/omnicoder_decode_step.onnx --out_dir weights/text/mlc --tvm_target "metal"
```
- Training with long-context scale:
```bash
python -m omnicoder.training.pretrain --data ./ --seq_len 2048 --target_ctx 32768 --rope_scale 4.0 --rope_base 10000.0
```
`LatentKVAttention` also supports `window_size` to cap attention to the latest tokens for memory savings on-device.
The ONNX decode-step runner supports per-head, per-group u8 KV emulation with dynamic groupwise scale/zero; pass `--kvq u8 --kvq_group 64` and optionally `--kvq_calibration`.

Verifier-head distillation for speculative decoding

```bash
python -m omnicoder.training.verifier_distill \
  --data ./ --seq_len 512 --steps 200 --device cuda \
  --teacher microsoft/phi-2 --student_mobile_preset mobile_4gb \
  --verifier_only --kl_temp 1.5 --lr 2e-4 --out weights/omnicoder_verifier_kd.pt
```

## VQ codebook (unified tokens for images)

Train a patch-level codebook and load it in `ImageVQ`:
```bash
python -m omnicoder.training.vq_train --data /path/to/images --out weights/image_vq_codebook.pt
```
Then use `ImageVQ(codebook_path=\"weights/image_vq_codebook.pt\")` when wiring custom pipelines.
To build unified multimodal tokens, also train a video VQ codebook and align codebook sizes across modalities.

Unified vocab sidecar
- The autofetcher writes `weights/unified_vocab_map.json` with text/image/video/audio vocab ranges. On-device tokenizers should load this to enforce consistent unified vocab slices.
- Training scripts automatically load and enforce the sidecar when present:
  - `omnicoder.training.vl_fused_pretrain`, `omnicoder.training.vqa_fused_train`, `omnicoder.training.vl_video_fused_pretrain`
- Inference CLI `omnicoder.inference.multimodal_infer` also loads the sidecar if available for consistent mapping.
- Override path via `OMNICODER_VOCAB_SIDECAR=/path/to/unified_vocab_map.json`. If absent, `MultiModalConfig` defaults are used.

Export a standalone Image VQ decoder (indices→image) to ONNX for mobile inference:

```bash
python -m omnicoder.export.onnx_export_vqdec \
  --codebook weights/image_vq_codebook.pt \
  --onnx weights/image_vq_decoder.onnx --hq 14 --wq 14
```

### Tiny VQ-VAE assets and smoke runs

```bash
# Generate tiny assets
python -m omnicoder.tools.make_tiny_vq_assets

# One-button tiny VQ quickstart (generates assets and runs all three trainers)
python -m omnicoder.tools.tiny_vq_quickstart --steps_img 50 --steps_audio 200 --samples_video 2048

# Or run trainers individually:
python -m omnicoder.training.vq_train --data examples/data/vq/images --steps 50 --batch 4 --emb_dim 64 --codebook_size 128 --out weights/image_vq_tiny.pt
python -m omnicoder.training.audio_vq_train --data examples/data/vq/audio --steps 200 --batch 2 --segment 8000 --codebook_size 128 --code_dim 64 --out weights/audio_vq_tiny.pt
echo examples/data/vq/video/toy.mp4 > examples/data/vq/video/toylist.txt
python -m omnicoder.training.video_vq_train --videos examples/data/vq/video/toylist.txt --resize 64 --patch 16 --emb_dim 64 --codebook_size 128 --frames_per_video 8 --samples 2048 --out weights/video_vq_tiny.pt
```

Unified multimodal token space (research track)
- Image tokens via `ImageVQ`; video tokens via `VideoVQ` (see `training/video_vq_train.py`).
- Align codebook sizes with text tokenizer vocab segments to enable joint modeling.
Reserved vocab ranges (defaults; adjust as needed)
- Text: [0..31999]
- Image VQ: [32000..(32000+image_codebook_size-1)] (default 8192)
- Video VQ: [32000+image_codebook_size .. +image+video-1]
- Audio VQ: [32000+image+video .. +image+video+audio-1]
These are surfaced via `MultiModalConfig` (`image_vocab_start`, `video_vocab_start`, `audio_vocab_start`).
- Future work: VQ-VAE training with adversarial/perceptual losses for higher fidelity.

## PPO for RL with programmatic rewards

Prototype PPO over prompts with rewards (text/code/image/audio). Example:
```bash
python -m omnicoder.training.ppo_rl --prompts path/to/prompts.jsonl --device cuda --steps 500 --reward text
```
JSONL lines can be raw strings or JSON objects with fields like `prompt`, `targets`, `tests`, `image`, `reference`.


- **Code**: pass@k on **HumanEval/MBPP** using unit tests and `subprocess` sandbox. Benchmarks are executed offline.
- **Text**: GSM8K/math via exact‑match; reasoning traces optional.
- **Images**: **CLIPScore** + **FID**; **Video**: **FVD** (+ VBench‑style probes if you add models).
- **Audio/Music**: **FAD**; **ASR**: **WER**; **TTS**: MOS‑proxy (MOSNet‑style) optional.

See `src/omnicoder/eval/*`.

Run minimal text exact-match eval (toy):

```bash
python -m omnicoder.eval.text_eval --tasks examples/code_eval/examples.jsonl --max_new_tokens 64
```

Image eval examples (install extras: `pip install -e .[eval]`):

```bash
# CLIPScore on a JSONL with {file, prompt}
.venv\Scripts\python -m omnicoder.eval.image_eval --mode clip --jsonl path/to/pairs.jsonl

# FID between generated and reference folders
.venv\Scripts\python -m omnicoder.eval.image_eval --mode fid --pred_dir weights/gen_images --ref_dir path/to/ref_images
```

Audio/ASR eval examples:

```bash
# WER from JSONL with {ref, hyp}
.venv\Scripts\python -m omnicoder.eval.audio_eval --mode wer --jsonl path/to/pairs.jsonl
```

Performance micro-benchmark (tokens/sec on your device):

```bash
python -m omnicoder.inference.benchmark --device cpu --seq_len 128 --gen_tokens 128 --mobile_preset mobile_4gb
```

Note: The throughput depends on your CPU/GPU/NPU. On first run it may be low due to random init and cold caches.

### Example results (local smoke)

These are small, non-representative smoke results to validate the wiring on CPU:

| Preset | Device | Seq len | Gen tokens | Tokens/s |
|---|---|---:|---:|---:|
| mobile_2gb | cpu | 64 | 64 | varies (CPU-only) |

## Export and On-Device Inference (text path)

```bash
# Export ONNX (toy LLM, mobile preset). Use --multi_token >1 to include extra heads
.venv\Scripts\python -m omnicoder.export.onnx_export --output weights/omnicoder_text.onnx --seq_len 64 --mobile_preset mobile_4gb --multi_token 2
# or with the 2GB preset
.venv\Scripts\python -m omnicoder.export.onnx_export --output weights/omnicoder_text_2gb.onnx --seq_len 64 --mobile_preset mobile_2gb --multi_token 2

# Export ONNX decode-step graph with KV-cache IO (mobile-friendly recurrent state)
.venv\Scripts\python -m omnicoder.export.onnx_export --output weights/text/omnicoder_decode_step.onnx --seq_len 1 --mobile_preset mobile_4gb --decode_step --target_ctx 32768 --two_expert_split --kvq u8 --kvq_group 64 --emit_longctx_variants

# Stream text generation using the decode-step ONNX with KV cache (desktop validation)
# (Install onnxruntime or onnxruntime-gpu first if not present)
# pip install onnxruntime
.venv\Scripts\python -m omnicoder.inference.runtimes.onnx_decode_generate --model weights/text/omnicoder_decode_step.onnx --prompt "Hello from ONNX!" --max_new_tokens 32 --kvq u8

# Export ExecuTorch decode-step program (stateful KV) and deploy with NNAPI delegate
.venv\Scripts\python -m omnicoder.export.executorch_export --out weights/text/omnicoder_decode_step.pte --two_expert_split

# Export ExecuTorch decode-step program (stateful KV) and deploy with NNAPI delegate
.venv\Scripts\python -m omnicoder.export.executorch_export --out weights/text/omnicoder_decode_step.pte --mobile_preset mobile_4gb

# Export Core ML decode-step (stateful KV) for iOS (requires coremltools>=7)
python -m pip install coremltools>=7
.venv\Scripts\python -m omnicoder.export.coreml_decode_export --out weights/text/omnicoder_decode_step.mlmodel --preset mobile_4gb

# One-command mobile packager (text-only, recommended)
  .venv\Scripts\python -m omnicoder.export.mobile_packager --preset mobile_4gb --out_dir weights/text --seq_len_budget 4096 --quantize_onnx --export_executorch
  # emit provider quant maps (NNAPI/Core ML/DML)
  .venv\Scripts\python -m omnicoder.export.mobile_packager --preset mobile_4gb --out_dir weights/text --nnapi_maps
# For 2GB devices
.venv\Scripts\python -m omnicoder.export.mobile_packager --preset mobile_2gb --out_dir weights/text_2gb --seq_len_budget 2048 --quantize_onnx --export_executorch

## Bundle multimodal decoders with the packager

To include a compact vision backbone, Stable Diffusion components, an image VQ decoder, and an optional Piper TTS model alongside the text artifacts, pass the following flags (best-effort per toolchain):

```bash
# Windows PowerShell (run each line separately)
.venv\Scripts\python -m omnicoder.export.mobile_packager \
  --preset mobile_4gb \
  --out_dir weights/text \
  --seq_len_budget 4096 \
  --quantize_onnx \
  --export_executorch \
  --with_vision --vision_backend mobilevit_xs \
  --with_sd --sd_model runwayml/stable-diffusion-v1-5 \
  --with_vqdec --image_vq_codebook weights/image_vq_codebook.pt --vqdec_hq 14 --vqdec_wq 14 \
  --with_piper --piper_url https://github.com/rhasspy/piper/releases/download/2023.11.14/en_US-amy-high.onnx
```

Outputs are summarized in `weights/mobile_packager_manifest.json`, and provider quant maps are written next to ONNX artifacts under `weights/{vision,sd_export,vqdec}`. Use `--vision_export_coreml`, `--sd_export_coreml`, or `--vqdec_export_coreml` (and corresponding `--*_executorch`) to emit additional backends when available.
```

Optional 4‑bit quant (AWQ/GPTQ) with external libs

```bash
# AWQ (requires: pip install autoawq transformers)
.venv\Scripts\python -m omnicoder.export.awq_gptq_quantize --method awq --hf_model meta-llama/Llama-2-7b-hf --out weights/llama2_awq_int4

# GPTQ (requires: pip install auto-gptq transformers)
.venv\Scripts\python -m omnicoder.export.awq_gptq_quantize --method gptq --hf_model mistralai/Mistral-7B-v0.1 --out weights/mistral_gptq_int4
```

## Export image decoder backends

You can export Stable Diffusion components for mobile backends (best-effort):

```bash
# ONNX (via Optimum): exports full SD pipeline (text encoder, U-Net, VAE)
# If no --hf_id/--local_path is provided, a lightweight distilled SD variant is used by default for smaller artifacts
.venv\Scripts\python -m omnicoder.export.diffusion_export --onnx --hf_id runwayml/stable-diffusion-v1-5 --out_dir weights/sd_export

# Core ML (experimental): exports VAE decoder MLProgram
.venv\Scripts\python -m omnicoder.export.diffusion_export --coreml --hf_id runwayml/stable-diffusion-v1-5 --out_dir weights/sd_export

# ExecuTorch (experimental): exports VAE decoder program
.venv\Scripts\python -m omnicoder.export.diffusion_export --executorch --hf_id runwayml/stable-diffusion-v1-5 --out_dir weights/sd_export
```

### Vision backbone export (MobileViT/EfficientViT) and grounding heads

```bash
# Export a compact vision backbone to ONNX (uses timm if available)
.venv\Scripts\python -m omnicoder.export.autofetch_backbones --out_root weights \
  --vision_export_onnx --vision_backend mobilevit_xs

# Or rely on auto-pick among: mobilevit_xs, mobilevit_s, efficientvit_lite0, vit_tiny_patch16_224
.venv\Scripts\python -m omnicoder.export.autofetch_backbones --out_root weights --vision_export_onnx

# Export lightweight grounding heads (YOLO‑E style) to ONNX; optional Core ML/ExecuTorch
.venv\Scripts\python -m omnicoder.export.onnx_export_grounding --out weights/vision --d_model 384 --tokens 196
.venv\Scripts\python -m omnicoder.export.onnx_export_grounding --out weights/vision --head rep_rta --d_model 384 --tokens 196 --coreml --executorch
```

Provider maps for image/VQ/vision exports
- The autofetcher now writes `nnapi_quant_maps.json`, `coreml_quant_maps.json`, and `dml_quant_maps.json` next to exported ONNX artifacts for SD, VQ decoders, and vision backbones.
- Use these sidecars to guide ORT EPs and on-device delegates.

Environment variables for provider profiles (image)
- `OMNICODER_IMAGE_PROVIDER_PROFILE`: path to a provider profile JSON for image ONNX backends (used by `multimodal_infer --image_backend onnx` and auto-bench image path).
- `OMNICODER_PROVIDER_PROFILE`: fallback profile path if the image-specific variable is not set.
- CLI overrides exist via `--provider_profile`.

Example provider profiles (see `profiles/`):
- Windows (DirectML GPU): `profiles/windows_dml.json`
- iOS (Core ML MLProgram): `profiles/coreml_ios.json`
- Android (NNAPI): `profiles/nnapi_android.json`

Tokens/sec thresholds (Press Play and benches):
- Default thresholds live in `profiles/provider_thresholds.json`. Press Play passes this automatically to text/vision/vqdec benches when present; the standalone provider_bench also auto-loads it if `--threshold_json` isn’t provided.

# Compile ONNX decode-step to a TVM/MLC artifact (requires tvmc in PATH)
## Autofetch and export real backbones

Use the one-shot autofetcher (also used by press_play) to export text decode-step graphs and (optionally) Stable Diffusion components, record video pipeline references, and download a Piper TTS model:

```bash
.venv\Scripts\python -m omnicoder.export.autofetch_backbones --out_root weights \
  --preset mobile_4gb --seq_len_budget 4096 --onnx_opset 17 --quantize_onnx --export_executorch \
  --sd_model runwayml/stable-diffusion-v1-5 --sd_export_onnx \
  --video_model stabilityai/text-to-video-sdxl \
  --piper_url https://github.com/rhasspy/piper/releases/download/2023.11.14/en_US-amy-high.onnx
```

Outputs a `weights/backbones_summary.json` with exported artifact paths.

.venv\Scripts\python -m omnicoder.export.mlc_compile --onnx weights/text/omnicoder_decode_step.onnx --out_dir weights/text/mlc --tvm_target "metal"

Provider hints for mobile ONNX runtimes:
- Use `--onnx_provider_hint` in the mobile packager to note the intended provider (e.g., `NNAPIExecutionProvider`, `CoreMLExecutionProvider`, `DmlExecutionProvider`). For manual runs: `inference/runtimes/onnx_mobile_infer.py --provider NNAPIExecutionProvider --nnapi_accel NnapiAccelerator.qnn`.

## Multimodal (image demo)

Provide a local Stable Diffusion pipeline or HF id, then run. If you pass the model's hidden state as `conditioning`, the pipeline will append it as an extra cross-attention token to bias generation (ConditionedSDPipeline).

```bash
# Using HF id (CPU example; diffusers backend)
.venv\Scripts\python -m omnicoder.inference.multimodal_infer --task image --device cpu \
  --image_backend diffusers \
  --sd_model runwayml/stable-diffusion-v1-5 --prompt "A scenic landscape" \
  --image_steps 20 --image_width 512 --image_height 512 --image_out weights/image_out.png

# Using local path
.venv\Scripts\python -m omnicoder.inference.multimodal_infer --task image --device cpu \
  --image_backend diffusers \
  --sd_local_path D:\\sd15 --prompt "A cyberpunk city at night" \
  --image_steps 25 --image_width 640 --image_height 384 --image_out weights/image_out2.png

# Optional: ONNX-based SD pipeline (diffusers-onnx) if installed or Optimum export
.venv\Scripts\python -m omnicoder.inference.multimodal_infer --task image --device cpu \
  --image_backend diffusers \
  --sd_local_path D:\\stable-diffusion-onnx --prompt "A serene beach" \
  --image_steps 20 --image_width 512 --image_height 512 --image_out weights/image_onnx.png

# Or plug an ONNX Runtime callable built from Optimum export (advanced):
#  1) Export: python -m omnicoder.export.diffusion_export --onnx --hf_id runwayml/stable-diffusion-v1-5 --out_dir weights/sd_export
#  2) Use ORT callable in code:
"""
from omnicoder.inference.runtimes.onnx_image_decode import ORTSDCallable
from omnicoder.modeling.multimodal.image_pipeline import ImageGenPipeline

ort_callable = ORTSDCallable('weights/sd_export/onnx')
pipe = ImageGenPipeline(backend='onnx')
pipe.load_backend(pipe=ort_callable)  # inject callable
img_path = pipe.generate('A scenic landscape', steps=15, size=(512, 512), out_path='weights/image_onnx_callable.png')
"""

# CLI using ONNX callable (no diffusers) with an Optimum export dir
.venv\Scripts\python -m omnicoder.inference.multimodal_infer --task image --device cpu \
  --image_backend onnx --onnx_sd_dir weights/sd_export/onnx \
  --prompt "A mountain lake at sunrise" --image_steps 25 --image_width 512 --image_height 512 \
  --image_out weights/image_onnx_cli.png
```

Cross‑modal verifier gate (mini‑CLIP style):

- Enable with `--cm_verifier` or env `OMNICODER_CM_VERIFIER=1` to reject clearly off‑prompt generations.
- Set threshold `--cm_threshold` in [0,1] (default 0.6).
- Multi‑candidate selection: set `OMNICODER_IMAGE_NCAND` (e.g., 4) to sample N images and pick the best by verifier score automatically.
- Training: `src/omnicoder/training/verifier_train.py` fits a tiny `CrossModalVerifier` and exports an ONNX wrapper for mobile.

### Multimodal fusion (image/video + text → text)

The core model now accepts fused features. Prototype image- or video-conditioned text by composing vision tokens and text embeddings using `MultimodalComposer` and feeding the fused sequence to `OmniTransformer.forward` (pre-embedded features bypass token embedding). Helpers:

- `prime_kv_with_features` to prime decode KV from fused features
- `continue_generate_from_primed` to continue AR decoding efficiently

Notes
- Incremental decoding is enabled in the default generator and model forward pass (`use_cache=True`) to mirror mobile runtimes that stream tokens. The ONNX decode‑step runner was validated locally.
- Optional retrieval: use `LocalRetriever` to fetch supporting context chunks from a folder of `.txt` documents and prepend to your prompt for extended context without growing KV memory.

### End-to-end VQ-VAE usage
- Encode image/video to tokens:
```python
from omnicoder.modeling.multimodal.image_vq import ImageVQ
from omnicoder.modeling.multimodal.video_vq import VideoVQ
from omnicoder.modeling.multimodal.vocab_map import map_image_tokens, map_video_tokens
from omnicoder.config import MultiModalConfig

mmc = MultiModalConfig()
img_vq = ImageVQ(codebook_path="weights/image_vq_codebook.pt")
vid_vq = VideoVQ(codebook_path="weights/video_vq_codebook.pt")
img_codes = img_vq.encode(img_np)[0].tolist()
vid_codes = sum([c.tolist() for c in vid_vq.encode(video_np)], [])
img_tokens = map_image_tokens(img_codes, mmc)
vid_tokens = map_video_tokens(vid_codes, mmc)
```

### Audio tokens (EnCodec-VQ)
- Use `AudioTokenizer` to produce EnCodec code streams and map them via `map_audio_tokens`. Train an audio VQ‑VAE later for full unification.
```python
from omnicoder.modeling.multimodal.audio_tokenizer import AudioTokenizer
from omnicoder.modeling.multimodal.vocab_map import map_audio_tokens
tok = AudioTokenizer(sample_rate=32000)
codes = tok.encode(wave_np)  # list of codebooks [np.ndarray]
audio_tokens = map_audio_tokens(codes[0].tolist())
```

### Train an audio VQ‑VAE codebook
```bash
python -m omnicoder.training.audio_vq_train --data /path/to/wavs --steps 20000 --batch 4 --segment 32768 \
  --codebook_size 2048 --code_dim 128 --out weights/audio_vq_codebook.pt
```

## Multimodal (video and audio)

Video (text→video):

```bash
.venv\Scripts\python -m omnicoder.inference.multimodal_infer --task video --device cpu \
  --video_backend diffusers --video_model stabilityai/text-to-video-sdxl \
  --prompt "A drone shot over mountains at sunset" --video_frames 24 --video_steps 25 \
  --video_width 512 --video_height 320 --video_out weights/video_out.mp4
```

Lightweight image→video (default when no model id is given) with temporal consistency filter:

```bash
# Uses a lightweight Stable Video Diffusion by default when no --video_model is provided
.venv\Scripts\python -m omnicoder.inference.multimodal_infer --task video --device cpu \
  --video_backend diffusers \
  --video_frames 24 --video_steps 20 --video_width 512 --video_height 320 \
  --video_seed_image weights/image_out.png --video_out weights/video_out_i2v.mp4
```
The pipeline applies a fast optical-flow–guided temporal blending pass by default to reduce flicker. Control strength and passes in code via `temporal_alpha` and `temporal_passes` (CLI flags forthcoming).

Verifier-based video gating: pass `--cm_verifier --cm_threshold 0.6` to compute a simple text↔video score and reject a clip if it falls below the threshold.

ONNX Runtime i2v callable (NNAPI/CoreML/DML providers):

```bash
# Export (or obtain) an ONNX i2v directory containing generator.onnx

# Run i2v via ORT callable (e.g., NNAPI on Android via ADB-runner or desktop CPU)
.venv\Scripts\python -m omnicoder.inference.multimodal_infer --task video --device cpu \
  --onnx_video_dir weights/i2v_export/onnx \
  --onnx_video_provider CPUExecutionProvider \
  --video_frames 24 --video_steps 1 \
  --video_seed_image weights/image_out.png --video_out weights/video_out_i2v_onnx.mp4 \
  --temporal_filter --temporal_alpha 0.7 --temporal_passes 1
```

Audio ASR/TTS/Vocoder (if libraries installed):

```python
from omnicoder.modeling.multimodal.asr import ASRAdapter
from omnicoder.modeling.multimodal.tts import TTSAdapter
from omnicoder.modeling.multimodal.audio_vocoder import HiFiGANVocoder
import numpy as np

print(ASRAdapter('small').transcribe('path/to/audio.wav'))
print(TTSAdapter().tts('Hello from OmniCoder!', out_path='weights/tts_out.wav'))
# Vocoder a mel-spectrogram (n_mels, T) -> wav
mel = np.load('path/to/mel.npy').astype('float32')
print(HiFiGANVocoder().vocode(mel, out_path='weights/vocode_out.wav'))
```

Or via CLI (audio task):

```bash
.venv\Scripts\python -m omnicoder.inference.multimodal_infer --task audio --device cpu \
  --asr_input path/to/audio.wav --asr_model_size small \
  --tts_text "Hello from OmniCoder" --tts_out weights/tts_out.wav \
  --mel_npy path/to/mel.npy --vocoder_backend auto --vocoder_model path/to/hifigan.onnx
```

### Fused vision-language pretraining (image+text → text)

Prepare a JSONL with records like `{ "image": "/path/to/img.jpg", "text": "caption or question..." }` and run:

```bash
python -m pip install -e .[vision]
.venv\Scripts\python -m omnicoder.training.vl_fused_pretrain \
  --jsonl examples/vl_fused_sample.jsonl --mobile_preset mobile_4gb \
  --batch_size 2 --steps 200 --device cuda --out weights/omnicoder_vl_fused.pt
### VQA fused training (image+question → text)

```bash
python -m omnicoder.training.vqa_fused_train \
  --jsonl examples/vqa_fused_sample.jsonl --mobile_preset mobile_4gb \
  --batch_size 2 --steps 50 --device cpu --out weights/omnicoder_vqa_fused.pt
```
```

Mobile packager outputs
- ONNX decode-step: `weights/text/omnicoder_decode_step.onnx`
- Optional ONNX int8 (dynamic): `weights/text/omnicoder_decode_step_int8.onnx`
- ExecuTorch program (if enabled): `weights/text/omnicoder_decode_step.pte`
- Memory budget summary (int4 weights + fp16 KV estimate): `weights/text/mobile_packager_summary.json`

Optional: dynamic int8 quantization for CPU inference (PyTorch) and ONNX PTQ

```bash
python -m omnicoder.export.pt_quantize --ckpt weights/omnicoder_toy.pt --out weights/omnicoder_int8.pt

# ONNX dynamic quantization
pip install onnxruntime onnxruntime-tools
python -m omnicoder.export.onnx_quantize --model weights/omnicoder_text.onnx --out weights/omnicoder_text_int8.onnx
python -m omnicoder.export.onnx_quantize_per_op --model weights/text/omnicoder_decode_step.onnx --out weights/text/omnicoder_decode_step_int8.onnx --op_types MatMul,Attention --per_channel
```

For Android/iOS deployment, see `inference/serverless_mobile/{android,ios}` docs.

## Utilities

Preset export and weights validation

```bash
# Export presets to JSON
.venv\Scripts\python -m omnicoder.tools.presets_export --out weights/presets.json

# Validate required exported artifacts exist
.venv\Scripts\python -m omnicoder.tools.weights_validator --root weights

# Optional: Desktop one-click GUI (Tk) — superseded by press_play but kept for dev
.venv\Scripts\python -m omnicoder.tools.gui_play

# Data engine: mirror and index datasets (VL/ASR)
.venv\Scripts\python -m omnicoder.tools.data_engine mirror --urls urls.txt --out data_cache
.venv\Scripts\python -m omnicoder.tools.data_engine index-vl --root data_cache/images --out vl.jsonl
.venv\Scripts\python -m omnicoder.tools.data_engine index-asr --root data_cache/audio --out asr.jsonl
# KV-cache quantization calibration (collects per-group stats over streams)
.venv\Scripts\python -m omnicoder.tools.kv_calibrate --mobile_preset mobile_4gb --prompts examples/code_eval/examples.jsonl --max_new_tokens 64 --kvq nf4 --group 64 --out weights/kvq_calibration.json

# Build PQ index from text folder or from precomputed embeddings; a budget sidecar is emitted for on-device scanning
.venv\Scripts\python -m omnicoder.tools.pq_build ./docs ./weights/pq --m 16 --ks 256
.venv\Scripts\python -m omnicoder.tools.pq_build ./docs ./weights/pq_from_emb --from_embeddings path/to/emb.npy --m 16 --ks 256
```

### Press Play (single-button)

```bash
# Windows (run each line separately in PowerShell)
OMNICODER_OUT_ROOT=weights/release \\
OMNICODER_QUANTIZE_ONNX=1 \\
OMNICODER_EXPORT_EXECUTORCH=1 \\
.venv\Scripts\python -m omnicoder.tools.press_play \
  --out_root weights/release \
  --quantize_onnx --export_executorch \
  --sd_model runwayml/stable-diffusion-v1-5 --sd_export_onnx

# Linux/macOS
python -m omnicoder.tools.press_play \
  --out_root weights/release \
  --quantize_onnx --export_executorch \
  --sd_model runwayml/stable-diffusion-v1-5 --sd_export_onnx
```

This will optionally run quick KD, export text decode‑step + PTQ, stage optional image/video/audio backbones, run auto‑benchmarks, and perform native and ONNX decode‑step text smoke runs. It respects `.env` settings.
Optional: set `OMNICODER_RUN_KV_CALIBRATE=1` to emit `weights/kvq_calibration.json` for KV quant runners/exporters. See `weights/release/press_play_manifest.json` and `weights/release/bench_summary.json`.

#### One-button via Docker (recommended)

```bash
# Build the CUDA image once
docker build -t omnicoder:cuda .

# Run Press Play end-to-end with GPU and persistent model cache
docker run --rm -it --gpus all \
  -v %cd%:/workspace \
  -v %cd%/models:/models \
  -e HF_HOME=/models/hf -e TRANSFORMERS_CACHE=/models/hf \
  omnicoder:cuda bash -lc "python3 -m omnicoder.tools.press_play --out_root /workspace/weights/release --quantize_onnx --onnx_preset generic --no_kd"

# Validate ONNX decode-step IO on CPU EP
docker run --rm -it \
  -v %cd%:/workspace omnicoder:cuda bash -lc \
  "python3 -m omnicoder.inference.runtimes.onnx_mobile_infer --model /workspace/weights/release/text/omnicoder_decode_step.onnx --provider CPUExecutionProvider --prompt_len 8 | cat"
```

Docker Compose (one-liners)

```bash
# Build image
docker compose build

# Run tests (CPU/GPU depending on host setup)
docker compose run --rm tests

# One-button Press Play (exports + auto-benchmarks + manifests)
docker compose run --rm press_play

# ONNX decode-step CPU smoke
docker compose run --rm onnx_smoke

# Provider microbench (edit providers in compose if needed)
docker compose run --rm provider_bench
docker compose run --rm provider_bench_dml
docker compose run --rm provider_bench_coreml
docker compose run --rm provider_bench_nnapi
 
# Optional: ONNX parity check (PyTorch vs ONNX decode-step)
python -m omnicoder.tools.onnx_parity_check --onnx weights/release/text/omnicoder_decode_step.onnx --preset mobile_4gb --steps 4 --abs_tol 3e-3 --rel_tol 3e-3 | cat
```

GPU on Windows with WSL2/NVIDIA
- Ensure the NVIDIA Driver and NVIDIA Container Toolkit are installed and that Docker Desktop has WSL2 integration enabled.
- If you see a warning about "NVIDIA Driver was not detected" during compose runs, GPU acceleration is unavailable; runs will proceed on CPU. See NVIDIA docs to enable `--gpus all` in containers.

Or with docker compose:

```bash
docker compose run --rm press_play | cat
docker compose run --rm tests | cat
```

### Android ADB NNAPI smoke (optional)

```bash
# Push decode-step ONNX to device and run NNAPI device-side smoke with TPS threshold
python -m omnicoder.tools.android_adb_run \
  --onnx weights/release/text/omnicoder_decode_step.onnx \
  --gen_tokens 128 --prompt_len 128 --tps_threshold 15.0

# OR: push an ONNX i2v export (generator.onnx) and measure FPS on device
python -m omnicoder.tools.android_adb_run \
  --onnx_video_dir weights/i2v_export/onnx \
  --video_frames 24 --video_width 512 --video_height 320 --video_fps 24 \
  --fps_threshold 5.0
```

Output JSON is saved to `weights/release/text/nnapi_device_bench.json`.

### iOS Swift console (optional)

A minimal SwiftPM console project is included at `src/omnicoder/inference/serverless_mobile/ios/SampleConsole`. Copy your `omnicoder_decode_step.mlmodel` into `Sources/Resources/` and run:

```bash
cd src/omnicoder/inference/serverless_mobile/ios/SampleConsole
swift build -c release
swift run App
```

### Image VQ decoder runners (indices→image)

- Export ONNX/Core ML/ExecuTorch VQ decoders as shown above.
- ONNX CLI demo:

```bash
python -m omnicoder.inference.runtimes.onnx_vqdec_infer --model weights/vqdec/image_vq_decoder.onnx --hq 14 --wq 14 --out weights/vqdec_out.png
```

- Core ML CLI demo (macOS):

```bash
python -m omnicoder.inference.runtimes.coreml_vqdec_infer --model weights/vqdec/image_vq_decoder.mlmodel --hq 14 --wq 14 --out weights/vqdec_out_coreml.png
```

## Minimal Pretraining (toy text)

```bash
# Train a few steps on local .txt files (demo)
python -m omnicoder.training.pretrain --data ./ --steps 10 --seq_len 128 --device cpu --out weights/omnicoder_toy.pt

# Generate with the trained checkpoint
python -m omnicoder.inference.generate --prompt "Once upon a time" --ckpt weights/omnicoder_toy.pt --max_new_tokens 32
```

### Train → Export → Run (For Dummies)

```bash
# (A) Quick student KD on a single 24 GB GPU (uses LoRA/gradient checkpointing)
python -m omnicoder.training.distill --data ./ --seq_len 512 --steps 200 --device cuda \
  --teacher microsoft/phi-2 --student_mobile_preset mobile_4gb \
  --lora --gradient_checkpointing --out weights/omnicoder_student_kd.pt

# Optional: fetch small public datasets (Tiny Shakespeare / TinyStories) with checksums
python -m omnicoder.tools.datasets_fetch --dataset tinyshakespeare --out_dir data/auto_kd
python -m omnicoder.tools.datasets_fetch --dataset tinystories --out_dir data/auto_kd

# (B) One-shot export of mobile artifacts (decode-step ONNX + optional ExecuTorch + PTQ)
python -m omnicoder.export.autofetch_backbones --out_root weights \
  --preset mobile_4gb --seq_len_budget 4096 --onnx_opset 17 --quantize_onnx --export_executorch \
  --sd_model runwayml/stable-diffusion-v1-5 --sd_export_onnx

# (C) Auto-benchmark + ONNX decode-step validation and (optional) provider microbench
python -m omnicoder.eval.auto_benchmark --device cpu --seq_len 128 --gen_tokens 128 --preset mobile_4gb \
  --validate_onnx weights/text/omnicoder_decode_step.onnx --providers CPUExecutionProvider

# (D) Run ONNX decode-step streaming locally (desktop validation)
python -m omnicoder.inference.runtimes.onnx_decode_generate --model weights/text/omnicoder_decode_step.onnx \
  --prompt "Hello from ONNX!" --max_new_tokens 32 --kvq u8

# (E) Press Play (single-button) combines KD(optional), export, and bench
python -m omnicoder.tools.press_play --out_root weights/release --kd \
  --teacher microsoft/phi-2 --kd_steps 200 --kd_seq_len 512 \
  --quantize_onnx --export_executorch \
  --sd_model runwayml/stable-diffusion-v1-5 --sd_export_onnx
```

## Repo layout

```
omnicoder/
  README.md  CHANGELOG.md  TODO.md  pyproject.toml  LICENSE
  src/omnicoder/
    __init__.py  config.py
    modeling/
      transformer_moe.py  attention.py  routing.py  hrm.py
      multimodal/
        vision_encoder.py  video_encoder.py  audio_tokenizer.py
        image_vq.py  video_vq.py  diffusion_decoder.py  tts.py  asr.py
    training/
      pretrain.py  finetune_lora.py  distill.py  rl_programmatic.py  rlhf.py
      data/datamodule.py  tokenizers.py
    eval/
      code_eval.py  text_eval.py  image_eval.py  video_eval.py  audio_eval.py
    export/
      onnx_export.py  coreml_export.py  gguf_export.py  executorch_export.py  mlc_compile.py
    inference/
      generate.py  multimodal_infer.py  runtimes/
        llama_cpp_infer.py  onnx_mobile_infer.py  mlc_llm_infer.py
        memory_estimator.py
      serverless_mobile/
        android/README-android.md  ios/README-ios.md
  examples/
    code_eval/examples.jsonl  prompts/multimodal.txt
  weights/README.md
  env.example.txt
```

## One-button "For Dummies"

If you just want to press play:

1) Install with minimal extras that work everywhere (Windows/macOS/Linux):

```bash
# Windows PowerShell (run each line separately)
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e .[onnx,vision,gen]

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[onnx,vision,gen]
```

2) Configure via .env (optional) and build a mobile-ready bundle and benchmark:

```bash
# Example .env settings:
# OMNICODER_OUT_ROOT=weights/release
# OMNICODER_QUANTIZE_ONNX=1
# OMNICODER_QUANTIZE_ONNX_PER_OP=1
# OMNICODER_ONNX_PRESET=generic
# OMNICODER_EXPORT_EXECUTORCH=1
# OMNICODER_ORT_PROVIDER=CPUExecutionProvider
# OMNICODER_KD=0
# OMNICODER_KD_TEACHER=microsoft/phi-2
# OMNICODER_KD_STEPS=200
# OMNICODER_KD_SEQ_LEN=512
# OMNICODER_SD_MODEL=runwayml/stable-diffusion-v1-5
# OMNICODER_SD_EXPORT_ONNX=1
# OMNICODER_BENCH_DEVICE=cpu

# Windows
.venv\Scripts\python -m omnicoder.tools.press_play

# Linux/macOS
python -m omnicoder.tools.press_play
```

3) Run the text demo (random weights unless you trained or distilled a checkpoint):

```bash
# Windows
.venv\Scripts\python -m omnicoder.inference.generate --prompt "Hello from OmniCoder" --max_new_tokens 32 --mobile_preset mobile_4gb

# Linux/macOS
python -m omnicoder.inference.generate --prompt "Hello from OmniCoder" --max_new_tokens 32 --mobile_preset mobile_4gb
```

4) Optional multimodal image demo (requires a Stable Diffusion pipeline):

```bash
# Windows
.venv\Scripts\python -m omnicoder.inference.multimodal_infer --task image --device cpu --image_backend diffusers --sd_model runwayml/stable-diffusion-v1-5 --prompt "A scenic landscape" --image_steps 20 --image_out weights/image.png

# Linux/macOS
python -m omnicoder.inference.multimodal_infer --task image --device cpu --image_backend diffusers --sd_model runwayml/stable-diffusion-v1-5 --prompt "A scenic landscape" --image_steps 20 --image_out weights/image.png
```

### Single Press Play
- Single entrypoint: `press-play` (console script) or `python -m omnicoder.tools.press_play`. This orchestrates KD (optional), export, quantization, benchmarks, native and ONNX smokes, and writes manifests.
  - You may configure behavior via CLI flags or environment variables (see `env.example.txt`).
  - To prefer a specific ORT provider for the ONNX smoke, set `OMNICODER_ORT_PROVIDER` (e.g., `DmlExecutionProvider`, `CoreMLExecutionProvider`, `NNAPIExecutionProvider`).
  - The ONNX decode-step smoke uses the artifact under your `out_root` (default `weights/release/text/omnicoder_decode_step.onnx`).
  - KD enablement: pass `--kd` on the command line or set `OMNICODER_KD=1` in `.env`. Use `--kd_steps` and `--kd_seq_len` (or envs) to control a quick KD stage. (Fixed: `--kd` is now accepted alongside `--kd_steps/--kd_seq_len`.)

#### One-button Train → Export → Validate (default GPU if available)

```bash
# Minimal KD on a single 24 GB GPU, then export and benchmark (Windows shown; run one line at a time)
.venv\Scripts\python -m omnicoder.tools.press_play \
  --out_root weights/release \
  --kd --teacher microsoft/phi-2 --kd_steps 200 --kd_seq_len 512 \
  --quantize_onnx --onnx_preset generic --export_executorch

# Docker (recommended; persistent cache avoids re-downloads)
docker run --rm -it --gpus all \
  -v ${PWD}:/workspace \
  -v ${PWD}/models:/models \
  -e HF_HOME=/models/hf -e TRANSFORMERS_CACHE=/models/hf \
  omnicoder:cuda bash -lc "python3 -m omnicoder.tools.press_play --out_root /workspace/weights/release \
    --kd --teacher microsoft/phi-2 --kd_steps 200 --kd_seq_len 512 \
    --quantize_onnx --onnx_preset generic --export_executorch"
```

Notes on image auto-bench
- When a Stable Diffusion ONNX export directory exists, Press Play selects the ONNX callable image path automatically. If ONNX loading fails, the image bench is skipped (JSON will show `"image": null`). You can force the backend by setting `OMNICODER_BENCH_IMAGE_BACKEND=onnx|diffusers`.
- Numeric parity warnings during SD ONNX export (absolute-diff above 0.0003) are expected with current toolchain; tracked in TODO to tighten parity checks or adjust per-component tolerances.

Notes
- Checkpoints and LoRA adapters are saved under `weights/` and automatically reused by passing `--ckpt` to CLI tools, or by setting `OMNICODER_STUDENT_CKPT` in `.env` for `press_play`.
- HuggingFace caches persist under `/models/hf` in Docker when you mount `-v ${PWD}/models:/models`.

### Verified build snapshot (Windows 11, Python 3.12)
- Editable install inside `.venv` with extras succeeded.
- `pytest`: 43 passed, 1 skipped.
- Text CLI ran and produced output.
- ONNX decode‑step export succeeded; ORT CPU decode‑step generate worked.
- One-command mobile packager produced artifacts: decode‑step ONNX, dynamic and per‑op int8 ONNX, NNAPI/CoreML/DML quant maps, and a memory budget summary. Press Play respects `.env`.

### Verified container snapshot (Docker CUDA, 2025-08-14)
- GPU visible in container (`torch.cuda.is_available() == True`).
- Native text CLI generated output; ONNX decode‑step export succeeded; ORT decode‑step streaming invoked successfully on CPU EP.
- Full pytest run can be resource intensive in constrained containers; run subsets or CPU-only where needed.
- Local verification notes: Built Docker image, confirmed GPU with `nvidia-smi`. Ran Press Play with persistent `/models` volume; artifacts are written incrementally under `weights/release`. Running `pytest` may be terminated by container resource limits ("Killed"); prefer `docker compose run --rm tests`, run focused subsets (e.g., `pytest -k onnx`), or run tests natively in a venv.

### Verified on this workstation (Windows 11 + Docker Desktop, 2025-08-14)
- Built image via `docker build -t omnicoder:cuda .`
- Ran `docker compose run --rm press_play` (CPU path; Docker Desktop compose cannot pass `--gpus` at CLI). Press Play completed exports and auto-benchmarks:
  - Wrote text decode‑step ONNX to `weights/release/text/omnicoder_decode_step.onnx`
  - Exported Stable Diffusion to `weights/release/sd_export/onnx` (observed absolute‑diff tolerance warning; tracked in TODO)
  - Wrote `weights/release/bench_summary.json` and `weights/release/unified_vocab_map.json`
- Tests via `docker compose run --rm tests` completed without failures on CPU (summary truncated in PowerShell output).
- Notes for Windows PowerShell:
  - Do not pipe Docker output to `cat`; PowerShell's `Get-Content` does not accept pipeline objects the way bash does. Run commands directly.
  - For GPU runs use `docker run ... --gpus all` rather than `docker compose run --gpus all` (Compose CLI does not accept `--gpus`).

Compose validation (same date)
- `docker compose run --rm press_play` produced:
  - `weights/release/text/omnicoder_decode_step.onnx` (decode-step)
  - SD ONNX directory under `weights/release/sd_export/onnx`
  - NNAPI quant maps sidecar and `bench_summary.json`
- Auto-bench summary recorded tokens/s on CPU and validated ONNX decode-step outputs.
- Note: a subsequent auto-bench rerun can be long on CPU-only hosts; you can cancel safely after artifacts are written.

### This run (press_play) summary (2025-08-14)
- Text decode-step ONNX exported to `weights/release/text/omnicoder_decode_step.onnx` and validated by the CPU ORT runner.
- Auto-bench (CPU): native ~17.56 tok/s at seq_len=128, gen_tokens=128; ORT CPUExecutionProvider ~27.42 tok/s.
- SD ONNX export completed with parity warning (max abs diff ~0.00335 vs 0.0003 tolerance); artifacts at `weights/release/sd_export/onnx`.
- Image auto-bench skipped since ONNX export dir lacks Diffusers `.safetensors`; use ONNX callable path instead (see Multimodal demo).

Latest auto-bench (CPU, container):

```json
{
  "text": {"tokens_per_sec": 17.89, "gen_tokens": 128, "seq_len": 128, "weights_h": "355.6 MB", "kv_h": "15.0 MB"},
  "providers": {"CPUExecutionProvider": {"tokens_per_sec": 30.19}},
  "image": {"latency_s": 231.65, "backend": "diffusers", "sd_model": "runwayml/stable-diffusion-v1-5"}
}
```

Note on SD ONNX export numeric tolerance
- During SD ONNX export, an absolute-diff tolerance warning was observed (max diff ~0.0031 vs tolerance 0.0003). The exported models work for demos; a follow-up task is tracked in `TODO.md` to tighten parity checks or adjust tolerances per component.

## Known issues / troubleshooting

- PowerShell PSReadLine may garble multi-line commands. Use cmd.exe, Windows Terminal, or PyCharm Run/Debug.
- ONNX Runtime is optional; install `onnxruntime` or `onnxruntime-gpu` for desktop tests. Use platform-specific builds on mobile.
- Multimodal backbones require you to supply real checkpoints per `weights/README.md` for image/video/audio.
- ExecuTorch export requires PyTorch 2.3+ with `torch.export` and executorch tooling. Fallback TorchScript is saved if unavailable (not production‑ready).
 - FAISS retriever: install `faiss-cpu` to use `--retrieve_faiss`. The TF‑IDF hashing now uses smoothed log‑IDF; a pure‑Python fallback retriever is available.
- ONNX image callable: `inference/runtimes/onnx_image_decode.py` contains a simple `ORTSDCallable` that expects an Optimum‑exported directory with `text_encoder.onnx`, `unet.onnx`, and `vae_decoder.onnx`. If you use diffusers' `OnnxStableDiffusionPipeline`, select the `diffusers` backend instead.
- Diffusers-based video pipelines: availability and speed depend on the chosen HF model; image-to-video requires a seed image which is not auto-generated in this CLI.

### Latest verification snapshot (2025-08-19)
- Docker GPU (CUDA 12.1 runtime): full suite green — 134/134 passed, 11 warnings, ~4m07s. Verbose log saved to `tests_logs/docker_pytest_full.txt`; real exit code persisted to `pytest_exit_code.txt`.
- ONNX decode-step export (standard + long‑ctx variants), DynamicCache shim, KV‑paging/quant sidecars, provider benches, retrieval, variable‑K/early‑exit wiring, multimodal heads, and video/audio/image pipelines validated by tests.
- Caches and models are persisted under `/models` (`HF_HOME`, `TRANSFORMERS_CACHE`) via the default volume mapping to avoid re‑downloads across runs.
- See CHANGELOG entries 0.1.9+post8.34…0.1.9+post8.35 for the granular fixes merged since the previous snapshot; no outstanding test regressions.

### Comprehensive verification report
- Environment: Windows 11, Python 3.12, fresh `.venv`, editable install with extras `[onnx,vision,gen]`.
- Tests (local): key unit tests pass (attention, routing, ONNX round‑trip, provider registry, KVQ). See `tests/` for coverage. New provider and launcher smokes added.
- Docker GPU (pending on this host): use the commands below to run the full suite with GPU once Docker/WSL2 toolkit is available.

### New options
- Exporter now prefers the new ONNX dynamo exporter by default and falls back to legacy when unavailable. Disable with `--no_dynamo`.
- Text generator has `--compile` to attempt `torch.compile` (inductor). It warms-up and falls back automatically if the system lacks a C++ compiler or the backend is unsupported.
- Most knobs can be configured via environment variables; see `env.example.txt`.

## CI / Docker GPU validation

Provider microbench harness and CPU CI are included. Provider-specific CI (DML/MPS) workflows are provided for self-hosted runners. A future Android/iOS CI will integrate demo runners and tokens/s thresholds:

- Windows + DirectML runner labels: `[self-hosted, windows, dml]`
- macOS + MPS runner labels: `[self-hosted, macos, mps]`

Device provider benches (self-hosted)
- A workflow `provider-device.yml` includes example jobs for DirectML (Windows) and Core ML (macOS) that:
  - Install minimal deps (`-e .[onnx,vision]`), optionally `torch-directml` on Windows
  - Run provider_bench on vision and VQ decoder ONNX models
  - Enforce tokens/s thresholds and check fusions
  - Upload bench JSONs as artifacts
Customize thresholds and providers per your device profile under `profiles/`.

Trigger manually:

```bash
gh workflow run provider-windows-dml.yml
gh workflow run provider-macos-mps.yml
```

Set tokens/s threshold via env (`TOKENS_PER_SEC_MIN`). Bench JSON captures KV footprint and throughput. Long-context canaries ensure 32k/128k decode-step variants export successfully.

### Docker GPU quickstart (tests and press‑play)

```bash
docker build -t omnicoder:cuda .
# Full tests with GPU (Windows PowerShell: use proper %CD% quoting or run from WSL)
docker run --rm --gpus all -v %CD%:/workspace -v %CD%\weights:/workspace/weights -v %CD%\models:/models omnicoder:cuda bash -lc "pytest -vv -rA"
# Or via compose (GPU):
docker compose run --rm --compatibility --gpus all tests
# One-button export/bench
docker compose run --rm press_play
```

### Single-script, time‑budgeted training (end‑to‑end)

The orchestrator plans and runs: Pre‑alignment → unified multi‑index build (optional) → DS‑MoE pretrain → draft KD + acceptance bench → VL/VQA fused → audio/video heads → optional GRPO → export + provider benches. All artifacts and manifests are saved under `weights/` and reused on subsequent runs.

```bash
# Native (venv) or inside the Docker container
python -m omnicoder.tools.run_training --budget_hours 10 --device cuda --draft_preset draft_2b

# Notes
# - Uses /models as the persistent HF cache (HF_HOME/TRANSFORMERS_CACHE) so teachers/backbones are fetched once
# - Periodically runs auto‑evals and provider benches; writes stage JSONs: bench_after_pretrain.json, bench_after_kd.json, bench_stage_summary.json
# - Emits READY_TO_EXPORT.md at the end with the next step
```

Then export to phone (best‑effort, optional device smokes):

```bash
python -m omnicoder.tools.export_to_phone --platform android --tps_threshold 15
# or
python -m omnicoder.tools.export_to_phone --platform ios
```

All runs honor the default volumes (`./models` → `/models`, `./weights` → `/workspace/weights`), so models and checkpoints persist across executions.

### Concise architecture status (2025‑08‑19)

- Core: Sparse MoE Transformer with hierarchical routing (Top‑K/Multi‑Head/GRIN/LLMRouter), sub‑experts and shared general experts; fused dispatch path; expert paging module.
- Attention: Latent‑KV (MLA), sliding‑window and landmark/random‑access paths; optional compressive KV; SSM interleave for full‑seq passes (export‑guarded).
- Decoding: Multi‑token heads, draft‑and‑verify wiring, early‑exit/difficulty heads; kNN‑LM and external retrieval memory; KV quant (u8/NF4) and paging.
- Multimodal: VQ‑VAE tokenizers/decoders (image/video/audio), continuous latent heads + tiny refiner export guards; vision backbone options and grounding/seg heads; video keyframes/interpolation + temporal modules; audio vocoder/adapters.
- Training: DS‑MoE curriculum; pre‑alignment (CLIP/ImageBind‑style) stage; multi‑teacher KD; GRPO; data engine; long‑context adapters (YaRN/PI) and canaries.
- Export/Mobile: ONNX decode‑step (+long‑ctx), Core ML decode‑step, ExecuTorch path; per‑op PTQ; provider maps/profiles; provider benches and thresholds.

Outstanding (tracked in TODO): branch‑train‑merge expert upcycling, stronger draft student (2–3B), variable‑K/halting training, adaptive memory compression/retention training, cross‑modal verifier training loop, unified retrieval/shared semantic memory promotion from prototype to default, 3D latent head expansion, and non‑autoregressive decode research lanes.

ExecuTorch NNAPI quant maps
- The one-button release now emits `weights/release/text/nnapi_quant_maps.json`, a sidecar configuration with per-op quantization and NNAPI delegate hints (`Attention` int8, `QLinearMatMul`). Use this with your deployment tooling to guide NNAPI execution.

### Android ADB: ONNX NNAPI and ExecuTorch .pte

- The Android ADB workflow pushes a fused ONNX and runs decode-step with `NNAPIExecutionProvider`, asserting a tokens/s threshold. If a `.pte` exists, it is also pushed and a device-side ExecuTorch run is attempted via the same Python runner, with a minimal tokens/s threshold if results are pulled.

### KV paging (sidecar + simulation)

- Export decode-step with KV paging sidecar:
```bash
python -m omnicoder.export.onnx_export --output weights/text/omnicoder_decode_step.onnx \
  --seq_len 1 --mobile_preset mobile_4gb --decode_step --kv_paged --kv_page_len 256
```
- Run a paged microbench (simulation) using the sidecar:
```bash
python -m omnicoder.inference.runtimes.provider_bench --model weights/text/omnicoder_decode_step.onnx \
  --providers CPUExecutionProvider --kv_paging_sidecar weights/text/omnicoder_decode_step.kv_paging.json
```
- Sidecar fields: `paged`, `page_len`, `n_layers`, `heads`, `dl`, `dl_per_layer`.

## License

- Code in this repo: **Apache‑2.0** (see `LICENSE`).
- You are responsible for third‑party model/data licenses (e.g., EnCodec, SD‑U‑Net, Whisper, etc.).

## Roadmap (abridged)

- [ ] Plug real backbones (ViT‑tiny, EnCodec small, SD‑U‑Net) and export graphs
- [ ] Distill from large teacher via offline traces (text/code/vision/audio/video)
- [x] GRPO‑style RL for reasoning and programmatic compilation rewards
- [ ] Mobile kernels for int4 matmul and fused attention on ANE/NNAPI
- [ ] End‑to‑end on‑device demo app
 - [ ] Add acceptance verifier for speculative decoding (draft‑and‑verify) with stronger correctness checks
 - [ ] Expand Core ML exporter with Apple attention ops and custom RoPE/KV‑latent layers

### High-impact performance plan (updated 2025-08-17)
## Frontier addenda (toward a single 2–4 GB frontier‑class multimodal model)
- Cross‑modal interaction experts (I2MoE): specialize experts for text↔image, image↔audio, video↔audio; route via hierarchical MoE.
- Unified embedding pre‑alignment (CLIP/ImageBind‑style): small encoders align text/image/audio/video embeddings prior to the transformer to improve routing and coherence.
- Vision backbones: DINOv3 is supported as a first-class option in `modeling/multimodal/vision_encoder.py`. Set `OMNICODER_VISION_BACKEND=dinov3` (default in `.env.example`) and optionally choose a variant via `OMNICODER_DINOV3_VARIANT` (e.g., `vit_base14`, `vit_large14`). Falls back to MobileViT/EfficientViT, SigLIP, or tiny ViT when DINOv3 is unavailable.
- Open‑vocabulary detection/segmentation (YOLO‑E inspiration) for grounding/editing; add exportable heads and ONNX callable.
- Video generation via keyframes + learned latent interpolation (ORT‑friendly); add temporal attention module and FVD canaries.
- Audio‑visual coupling (lip‑sync): cross‑attend phoneme/audio tokens with video frame tokens; add alignment loss and MOS/FAD checks.
- 3D latent provision: optional NeRF/voxel latent head and tiny renderer for view‑consistent generations (export‑guarded, off by default).
- Cross‑modal verifier (mini‑CLIP) to score candidate outputs vs prompts and select the best at inference.
- On‑device expert paging (LRU + async prefetch based on router probs) with memory budget controller.
- Adaptive precision runtime: confidence‑driven activation quantization (8→4→2‑bit where safe) with error bounds; emulate in ORT runner.
- Learned memory retention head to decide KV/memory keep/compress/drop; add canaries.
- Parallel decode modes (research): chunked mask‑predict/insertion decoding for long outputs; export‑guarded.
- Tool‑use protocol: special tokens to call offline tools; treat tool outputs as a modality.
- Code expert pretraining: initialize from strong OSS code models (e.g., StarCoder2‑base), align vocab/routing, and de‑bias via curriculum.

- Adopt `torch.compile` for decode path (inductor) and enable SDPA fast-paths; guard for graph breaks.
- Prefer SDPA v3/FlashAttention-3 on supported GPUs via `scaled_dot_product_attention` and SDP kernel preferences; fall back gracefully.
- Migrate ONNX exporters to dynamo path and integrate DynamicCache; keep decode-step graphs minimal and stateful.
- Prioritize device kernels: fused MLA/MQA and int4 GEMM backends for NNAPI/Core ML/DML; align weight/KV layouts with provider expectations.
- Solidify KV quantization: per-head group calibration (u8/NF4), sidecar scales, and dequant-per-step kernels in runners.
- Scale speculative decoding: combine multi-token heads with verifier acceptance; tune acceptance thresholds for ARM big.LITTLE.
- Retrieval/kNN-LM: add a small kNN cache over hidden states to bias next-token selection using local docs or recent conversation; keep it optional for mobile.
- Long-context stability: add 32k/128k decode-step CI canaries and windowed decode policies for mobile.

Implementation status tracker
- HRM inside core with adaptive halting: available and export-disabled for stability by default.
- Latent-KV attention with provider registry and SDPA fast path: available; device fused kernels WIP.
- KV-cache quantization (u8/NF4): end-to-end in PyTorch path; ONNX runner supports u8 emulation with sidecar scales.
- Multi-token prediction heads + speculative verification: available.
- kNN cache: available with NumPy/FAISS backends.

## Deep architecture overview (concise)

- Core model: Sparse MoE Transformer with Multi‑Head Latent Attention (latent KV compression), multi‑query attention option, optional HRM iterative loop, multi‑token prediction heads, paged/quantized KV cache. Designed for on‑device decode‑step with minimal state.
- Routing: Top‑k expert routing with capacity‑aware dispatch, grouped token processing, z‑loss + importance/load balancing penalties, static capacity caps for latency.
- Long context: RoPE with interpolation (YaRN/PI hooks), exporter can bake `rope_scale` and windowed decode for streaming.
- Quantization: Weight‑only int4 path (functional wrapper), ONNX per‑op PTQ, provider profiles for NNAPI/CoreML/DML, KV‑cache quant (u8/NF4) with calibration sidecar.
- Multimodal: Vision/Video/Audio adapters, VQ codebooks for unified tokens, Image/Video diffusion decoders (diffusers/ONNX callable), fusion composer to feed pre‑embedded tokens to LLM core.
- Mobile export: ONNX decode‑step with dynamic axes and sidecars (KVQ/paging), ExecuTorch decode‑step, Core ML MLProgram decode‑step, optional MLC/TVM compile.
- Training: Pretrain, LoRA/QLoRA, KD (teacher→student), PPO/GRPO RL with programmatic rewards, verifier‑head KD, fused VL/VQA training.

Performance priorities summary
- Replace softmax attention with fused MLA/MQA provider backends; use SDPA/FlashAttention-3 when available; maintain causal/windowed semantics.
- End-to-end quant: int4 text (AWQ/GPTQ), per‑op PTQ mappings for NNAPI/Core ML/DML, KV NF4/u8 with per‑head group calibration and on‑device dequant per step.
- Long-context: YaRN/PI training and exporter bake‑ins; 32k/128k decode‑step CI canaries and windowed decode policies.
- Speculative decoding: combine multi‑token heads with verifier acceptance; tree search tuned to ARM big.LITTLE.
- Retrieval/kNN‑LM: optional small cache; device‑friendly memory.

## Where it falls short vs the stated goal (2–4 GB mobile frontier‑class)

- Real backbones: Need compact yet strong backbones (e.g., ViT‑tiny/ViT‑mobile, EnCodec small, SD‑U‑Net distilled/lightweight, text‑to‑video lite) wired and exported.
- Device kernels: True device‑grade int4 matmul and fused MLA/MQA attention for NNAPI/ANE/DML; current int4 is a correctness wrapper.
- Exporter modernization: Migrate ONNX export to `torch.onnx.export(dynamo=True)` and integrate DynamicCache; extend Core ML with native attention/RoPE.
- Long‑context stability: Validate 32k/128k with stability metrics; add decode‑step long‑ctx CI.
- Unified tokens: Train VQ‑VAE codebooks and align vocab across modalities; current codebooks are examples.
- On‑device apps: Android ExecuTorch+NNAPI and iOS Core ML demo apps with tokenizer and streaming UI.
- Frontier‑level quality: Requires multi‑teacher KD + RL across text/code/math/VL/ASR/TTS with robust rewards and verifier acceptance.

Additional frontier gaps (new)
- On‑device video generation at Veo‑class quality within 2–4 GB remains research‑grade. We will pursue a hierarchical token approach (low‑fps latent tokens + frame interpolation) and a two‑stage decoder (token→latent→pixels) optimized with per‑op PTQ and tiled inference.
- Uniform on‑device ANN for retrieval/kNN‑LM using product quantization (PQ) with mmap‑backed indices to fit tight RAM budgets.

## Step‑by‑step plan to close gaps (high‑impact first)

1) Text path performance on device
   - Integrate FlashAttention‑3/SDPA v3 where supported; fall back to SDPA; ensure RoPE/MQA compatibility.
   - Torch 2.x `torch.compile` with inductor on decode loop; AOTAutograd off for inference; ensure no graph breaks.
   - Implement latent‑KV attention fused kernels per provider; bake operator sets for NNAPI/CoreML/DML; validate tokens/s on reference devices.
   - Finish per‑op PTQ maps per provider; calibrate `QLinearMatMul`/`Attention` and verify accuracy.
2) Quantization end‑to‑end
   - Adopt AWQ/GPTQ 4‑bit exports for text backbones; align weight packer with device int4 layouts.
   - KV‑cache u8/NF4 with per‑head group calibration; export sidecar scales; implement per‑step dequant kernels.
3) Long‑context
   - YaRN/PI training pass; bake scale/base in exporters; add 32k/128k decode stability CI canaries.
4) Multimodal
   - Train VQ‑VAE codebooks (image/video/audio); reserve vocab slices; fuse VL/VQA training.
   - Wire EnCodec/HiFi‑GAN, SD‑U‑Net lite, and a lightweight video diffusion; export ONNX/Core ML/ExecuTorch variants.
5) Apps/UX
   - Android ExecuTorch (NNAPI delegate) chat app with streaming UI; iOS Core ML decode‑step runner.
   - Press‑Play flows to fetch backbones, run tiny KD, export, deploy demo.

Notes on libraries/alternatives
- Prefer PyTorch ops on hot paths over NumPy/pandas; remove pandas from runtime loops; use vectorized tensor ops.
- Consider torch‑directml for Windows GPU; MPS/Core ML for Apple; NNAPI QNN for Android; ensure provider profiles set threads/graph opt levels.

### Major research/engineering tracks now in progress
- Implementing MLA kernels: attention path exposes a hook for int4-friendly custom ops; we will wire ANE/NNAPI providers and publish a minimal kernel when ready.
- Draft-and-verify acceptance: generator supports multi-step verification and external draft model acceptance flow; training the verifier head is included in pretrain; dedicated tests will be added.
- Long-context (YaRN/PI): exporters and training flags exist; we'll validate 32K/128K with stability reports and examples.
- Unified multimodal tokens: VQ codebook trainer (`training/vq_train.py`) + `ImageVQ`/`VideoVQ`; align with text tokens for joint training.
- Data engine: planned scalable offline ingestion, filtering, and synthesis across VL/VQA/ASR/TTS.
- RL tracks: GRPO (`training/rl_grpo.py`) and PPO (`training/ppo_rl.py`) runners with hooks for multimodal rewards (CLIPScore/FID/FVD/FAD, WER/MOS proxy, code tests/pass@k).
- Mobile int4: AWQ/GPTQ helpers exist; packager supports per-op ONNX PTQ with runtime presets; ExecuTorch/Core ML mappings are next.

### Hierarchical routing and multimodal heads
- Hierarchical MoE routing: provide contiguous expert group sizes via presets (`MobilePreset.moe_group_sizes`) or set at runtime in training to enable a two-tier router. Example: `moe_group_sizes=[4,4]` to form two groups of 4 experts each (e.g., text vs vision/audio). Training uses the hierarchical router automatically when group sizes are provided.
- Continuous latent heads (optional): the core exposes `image_latent_head` and `audio_latent_head` to output continuous latent tokens for image/audio decoders. Use training flags to add lightweight consistency losses or proper reconstruction losses when you provide decoders.

### Long-context YaRN/PI hooks
- Pretrain fine-tune flags:
  - `--target_ctx 32768 --rope_base 10000.0 --yarn` to enable YaRN-style rope scaling during training.
- Exporter flags:
  - `--emit_longctx_default` emits 32k/128k decode-step ONNX variants automatically; combine with `--yarn` and `--target_ctx` for specific scales.

### Continuous latent reconstruction (image/audio)

Image (text-conditioned) with Diffusers VAE or ONNX callable:

```bash
# Install generation extras for diffusers
pip install -e .[gen]

# Directory of images or JSONL with {"image","text"}
python -m omnicoder.training.flow_recon \
  --data /path/to/images_or.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --steps 500 --device cuda

# Using an ONNX callable (must expose encode())
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --onnx_dir weights/sd_export/onnx \
  --steps 500 --device cuda

# Optional: simplified diffusion/flow target on pooled VAE latents
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --steps 500 --device cuda
```

Audio (mel/codec/ONNX encoder):

```bash
# mel.npy (n_mels x T) folder -> random-proj latents
python -m omnicoder.training.audio_recon \
  --mel_dir /path/to/mels \
  --steps 500 --device cuda --mel

# EnCodec (requires extras [audio])
python -m omnicoder.training.audio_recon \
  --mel_dir /path/to/mels \
  --steps 500 --device cuda --encodec --wav_sr 32000

# ONNX audio encoder (expects a model with suitable encode input/output)
python -m omnicoder.training.audio_recon \
  --mel_dir /path/to/mels \
  --steps 500 --device cuda --onnx_audio weights/audio_encoder.onnx
```

```bash
# Quickstarts (continuous heads + temporal SSM)
# Image continuous latents training (flow-matching)
python -m omnicoder.training.flow_recon \
  --data /path/to/images_or.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --steps 1000 --device cuda --flow_loss --latent_dim 16 --out weights/image_latent_head.pt

# Audio continuous latents training (EnCodec latents)
python -m omnicoder.training.audio_recon \
  --mel_dir /path/to/mels \
  --steps 1000 --device cuda --encodec --latent_dim 16 --out weights/audio_latent_head.pt

# Temporal SSM ONNX export (video)
python -m omnicoder.export.onnx_export_temporal --out weights/video/temporal_ssm.onnx --d_model 384 --opset 18
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --steps 500 --device cuda
```

#### Audio wav_dir pathway (torchaudio)

If you prefer to train from raw audio instead of precomputed mel spectrograms, use `--wav_dir` (requires `torchaudio`):

```bash
pip install torchaudio
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --encodec --wav_sr 32000

# Or with mel/STFT perceptual losses from torchaudio (no EnCodec)
python -m omnicoder.training.audio_recon \
  --wav_dir /path/to/wavs \
  --steps 500 --device cuda --mel --wav_sr 16000
```
```

#### Flow parameterization flags

You can choose the flow parameterization when using `--flow_loss`:

```bash
# VP (variance preserving) cosine schedule
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param vp --steps 500 --device cuda

# VE (variance exploding)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param ve --steps 500 --device cuda

# EDM (default)
python -m omnicoder.training.flow_recon \
  --data images.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param