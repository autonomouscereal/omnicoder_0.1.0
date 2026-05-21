# OmniCoder

OmniCoder is an experimental omnimodal model stack: one research codebase for a
single model family that can ingest and emit text, code, images, video, audio,
structured artifacts, and tool actions. The long-term target is a compact,
edge-capable model that can reason across modalities without depending on a
separate specialist model for every input or output type.

This repository is not a polished model release or a claim of frontier
performance. It is the architecture lab: sparse experts, long-context memory,
multimodal tokenization, mobile/runtime exports, verifier loops, reward
training, and small runnable canaries in one place. Weights and newer training
work may live outside this public tree until they are ready to publish.

## Why This Exists

Most multimodal systems are pipelines: an LLM delegates to an image model, an
audio model, a video model, a detector, a retriever, and a tool runner. That can
work, but it creates brittle handoffs and makes edge deployment painful.
OmniCoder explores a different direction:

- Use a shared reasoning core across modalities.
- Route work through sparse experts instead of waking the whole model.
- Keep long context bounded with compressed memory, retrieval, and KV policies.
- Make text, code, vision, video, audio, and action heads trainable together.
- Export pieces to realistic local runtimes such as ONNX Runtime, Core ML,
  NNAPI-oriented runners, DirectML, ExecuTorch, llama.cpp/GGUF, and mobile app
  bundles.

The practical bet is that a smaller omnimodal model with good routing, memory,
verification, and device-aware exports can be more useful than a collection of
large disconnected models when privacy, latency, bandwidth, and hardware budget
matter.

## Current Capabilities

- Sparse MoE transformer core with hierarchical routing, expert paging hooks,
  variable-K routing experiments, capacity-aware dispatch, and mobile-minded
  runtime switches.
- Long-context mechanisms including sliding-window decode, memory slots,
  landmark/random-access attention experiments, retrieval/PQ/kNN hooks, KV
  quantization, KV retention sidecars, and learned KV compression experiments.
- Multimodal input/output modules for vision, image VQ, image decoding, video
  VQ, video heads, interpolation, audio tokenization, audio VQ-VAE, vocoder,
  ASR/TTS adapters, 3D latent scaffolding, and cross-modal fusion.
- Reasoning and verification experiments including HRM-style refinement,
  reward modeling, GRPO/PPO/RLHF scaffolds, code verification, multi-solution
  generation, verifier distillation, cross-modal verification, and cycle
  consistency checks.
- Export and runtime work for ONNX decode steps, provider benchmarking,
  DirectML, Core ML, NNAPI-style runners, mobile packaging, GGUF/llama.cpp
  adapters, MLC/TVM hooks, Core ML sample apps, and Android/iOS smoke paths.
- Training and data plumbing for pretraining, LoRA/QLoRA, KD, multimodal JSONL,
  VQA/VL/video/audio datasets, teacher profiles, dataset profiles, acceptance
  thresholds, benchmark canaries, and time-budgeted training probes.

## Repository Map

```text
src/omnicoder/
  modeling/          Core transformer, MoE, routing, attention, memory, kernels
  modeling/multimodal/
                     Image, video, audio, grounding, fusion, VQ, latent heads
  inference/         Generation loops and runtime adapters
  export/            ONNX, Core ML, ExecuTorch, GGUF, mobile packaging
  training/          Pretrain, KD, LoRA, reward, verifier, data loaders
  eval/              Benchmarks, canaries, verifier/eval harnesses
  retrieval/         PQ, graph/RAG, prefix hydration
  sfb/               Symbolic/factorized reasoning experiments
  tools/             CLI entrypoints for training, export, benches, packaging
profiles/            Device, provider, dataset, teacher, and threshold presets
examples/            Tiny JSONL and prompt fixtures for smoke testing
docs/                Current docs plus archived legacy notes
tests/               Unit, smoke, export, provider, and architecture canaries
```

## Quick Start

Use a virtual environment. The base install is enough for CPU smoke tests; use
extras only when you need export, audio, vision, or evaluation packages.

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
copy env.example.txt .env
```

On macOS/Linux:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
cp env.example.txt .env
```

Run a weights-free smoke path:

```bash
python -m omnicoder.inference.generate --prompt "Hello OmniCoder" --device cpu
```

Run the one-button development flow:

```bash
python -m omnicoder.tools.press_play --device cpu --out_root weights
```

The one-button flow is intended to exercise tests, exports, benchmarks, and
release artifacts under `weights/`. It is a development harness, not a magic
training recipe.

## Common Workflows

### Train Or Probe

```bash
python -m omnicoder.tools.run_training --budget_hours 1 --device cuda
```

For quick timing and planning:

```bash
python -m omnicoder.tools.train_probe --budget_minutes 120 --device cuda
```

Teacher and dataset profiles live in `profiles/teachers.json` and
`profiles/datasets.json`. Override paths with environment variables when
running local experiments.

### Export And Benchmark

```bash
python -m omnicoder.export.onnx_export --out weights/release/text/omnicoder_decode_step.onnx
python -m omnicoder.inference.runtimes.provider_bench ^
  --model weights/release/text/omnicoder_decode_step.onnx ^
  --providers CPUExecutionProvider DmlExecutionProvider ^
  --out_json weights/release/text/provider_bench.json
```

Provider thresholds live in `profiles/provider_thresholds.json`.

### Package For A Phone

```bash
python -m omnicoder.tools.export_to_phone --platform android --tps_threshold 15
python -m omnicoder.tools.export_to_phone --platform ios --tps_threshold 6
```

Mobile sample code lives under `src/omnicoder/inference/serverless_mobile/`.

### Enable Runtime Experiments

```bash
set OMNICODER_EXPERT_PAGING=1
set OMNICODER_EXPERT_PAGING_BUDGET_MB=256
set OMNICODER_EXPERT_PREFETCH_N=2
set OMNICODER_MULTI_INDEX_ROOT=weights/retrieval_multi_index
```

Useful knobs include expert paging, KV retention sidecars, activation fake
quantization, variable-K routing, landmark attention, memory slots, windowed
decode, and retrieval augmentation.

## Documentation

- [Current Architecture](docs/ARCHITECTURE_CURRENT.md)
- [Current Quickstart](docs/QUICKSTART_CURRENT.md)
- [Legacy Architecture Notes](docs/legacy/Architecture.md)
- [Legacy Dataset Notes](docs/legacy/Datasets.md)
- [Legacy Teacher Notes](docs/legacy/Teachers.md)
- [Backlog](todo/TODO.md)

The legacy docs are retained because they contain useful research notes, but
they do not fully describe the present intent of the project. Start with the
current docs above.

## Status

This is an active research codebase. Some modules are runnable, some are smoke
tested scaffolds, and some are architectural experiments waiting for larger
training runs or unpublished weights. Treat the repo as a map of the model
system and a set of reproducible experiments, not as a packaged consumer model.

## Design Principles

- One model family should reason across all modalities.
- Edge constraints are architecture constraints, not an afterthought.
- Every capability should have a small canary, export path, or benchmark hook.
- Verification and reward loops should be built into the system, not bolted on.
- Runtime truth matters: provider benches, device profiles, and memory budgets
  are first-class artifacts.

## License

See `LICENSE`.
