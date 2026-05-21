# OmniCoder Current Architecture

OmniCoder is organized around a single omnimodal research question: how much
can one compact model family do when routing, memory, verification, and export
are designed together from the start?

## System Shape

```text
Inputs
  text, code, images, video, audio, structured context, tool/action traces
        |
        v
Modality adapters and tokenizers
  text/code tokens, image/video/audio VQ codes, continuous latents, metadata
        |
        v
Shared reasoning core
  sparse MoE transformer, HRM-style refinement, long-context memory,
  retrieval hooks, attention/KV compression, variable expert routing
        |
        v
Output heads and decoders
  text/code, image/video/audio decoders, verifier heads, action/tool heads,
  latent refiners, mobile/export-specific decode steps
        |
        v
Verification, training, and runtime packaging
  reward loops, distillation, acceptance benches, ONNX/Core ML/NNAPI/DML,
  mobile bundles, provider thresholds, runtime sidecars
```

## Core Model

The core lives primarily in `src/omnicoder/modeling/`:

- `transformer_moe.py`: sparse transformer/MoE backbone.
- `attention.py`: causal attention, MQA/MLA style compression, windowing, and
  long-context hooks.
- `routing.py`, `moe_layer.py`, `experts.py`, `hyper_expert.py`: expert
  routing, dispatch, paging, and specialization experiments.
- `memory.py`: landmark and memory primitives for bounded long-context work.
- `hrm.py`: hierarchical refinement module for deeper reasoning without simply
  stacking more layers.
- `quant/`: int4 and KV cache provider experiments.
- `kernels/`: device/provider-oriented kernel and fused op experiments.

The design favors sparse activation and explicit runtime budgets. The goal is
not only to make a model smaller; it is to make the active work per request
match the request.

## Modalities

The multimodal layer lives in `src/omnicoder/modeling/multimodal/`.

Text and code use the standard generation path and code/verifier tooling. Image,
video, and audio modules include tokenizers, VQ/VAE style code paths, encoders,
decoders, continuous latent heads, refiners, grounding, and cross-modal fusion.
Some modules are functional smokes; others are research scaffolds that define
the intended interface for future weights and training runs.

The important architectural move is that modality code feeds a shared reasoning
core instead of permanently isolating each modality in a separate service.

## Long Context And Memory

OmniCoder treats context length as a runtime systems problem:

- sliding-window decode for bounded active KV,
- memory slots for compressed summaries,
- landmark/random-access attention experiments,
- retrieval and PQ/kNN hooks for external recall,
- KV quantization and retention sidecars,
- learned KV compression experiments.

The target is effectively unbounded task context while keeping per-step memory
and compute bounded enough for local devices.

## Training And Verification

Training and evaluation live in `src/omnicoder/training/`, `src/omnicoder/eval/`,
and `src/omnicoder/tools/`.

Current lanes include:

- pretraining and small data-engine loops,
- LoRA/QLoRA and fine-tuning hooks,
- multi-teacher distillation,
- verifier-head distillation,
- reward modeling and GRPO/PPO/RLHF scaffolds,
- code verification,
- multimodal acceptance metrics,
- benchmark and threshold tooling.

Verification is part of the architecture. The repo includes many small tests
because every experimental claim should have a canary: routing balance, long
context, KV policies, export parity, provider thresholds, multimodal VQ paths,
and mobile runtimes.

## Runtime And Export

Export/runtime work is not a side quest. It is the forcing function for the
project.

Important paths:

- `src/omnicoder/export/`: ONNX, Core ML, ExecuTorch, mobile package flows.
- `src/omnicoder/inference/runtimes/`: ONNX, Core ML, NNAPI-style, vLLM,
  llama.cpp, MLC/TVM, provider profiles, and runners.
- `profiles/`: device profiles, thresholds, datasets, teachers, acceptance
  gates, and mobile presets.
- `src/omnicoder/tools/export_to_phone.py`: mobile packaging entrypoint.

Runtime artifacts should carry enough sidecar data to explain what is running:
provider choices, thresholds, KV policies, package assets, and benchmark
results.

## What Is Not Finished

The public repo is an architecture and experiment base. It does not yet publish
the full model weights, large training runs, or every private experiment. Some
docs in `docs/legacy/` describe older assumptions. Prefer this document and the
top-level README when trying to understand the current mission.
