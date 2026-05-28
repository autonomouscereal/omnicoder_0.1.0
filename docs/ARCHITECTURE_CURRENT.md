# OmniCoder Current Architecture

This document summarizes the current 2026 architecture contract. The canonical
deep design note is `docs/Omnicoder2026Redesign.md`.

## Shape

OmniCoder 2026 is a dense, one-trunk omnimodal decoder architecture. The model
is trained over a shared token space that includes text, code, tool traces,
canonical JSON records, route prefixes, and typed media artifact tokens.

```text
inputs and training records
  text, code, tool calls, OCR, image/video/music/TTS specs, traces
        |
        v
canonical ledger/token representation
  assistant targets, media artifact targets, route markers, provenance
        |
        v
dense OmniCoder 2026 trunk
  local/global/compressed attention, long-context state, q4-aware modules
        |
        v
shared token head
  text tokens, tool JSON, route-prefixed media artifact token streams
        |
        v
edge systems
  artifact decoders, tool runners, benchmark harnesses, GGUF/runtime bridges
```

There are no permanent modality adapters in the trunk contract. Codecs,
artifact renderers, and runtime bridges live at the edges so the core model can
learn a unified internal representation.

## Core Model

The active implementation path is `src/omnicoder/modeling/omnicoder2026.py` and
the dense pipeline trainer in
`src/omnicoder/training/pipeline_pretrain_2026_dense.py`.

The design emphasizes:

- Dense depth over sparse expert routing.
- q4-aware training and checkpoint validation.
- Long-context attention and compressed state for the 1M-context target.
- Shared token prediction for text, code, tools, and media artifact streams.
- Route-aware outputs that can be parsed by standard inference and export
  tooling.

## Modalities

Media is represented as supervised route/artifact output, not as separate
in-trunk encoders:

- `image | {...}`
- `video | {...}`
- `music | {...}`
- `tts | {...}`
- `ocr | {...}`

The route prefix is ordinary model text. The JSON/artifact tokens after the
prefix are also learned targets. At inference time,
`src/omnicoder/inference/output_router_2026.py` parses the generated route and
hands media artifact streams to edge decoders or artifact backends.

## Training Contract

Training records are JSONL-first and must preserve assistant/media target
coverage. The current dense path supports:

- message-style SFT records,
- `input_json` + `target_json` records,
- explicit token-id records,
- media artifact token records,
- tool/action traces,
- long-context trace chunks.

Sparse assistant/media labels are not thinned by selected CE. Chunking uses
one-token overlap so target tokens at chunk boundaries still have causal
context.

## Evaluation

Evaluation is split into diagnostics and reportable scoring:

- Target-token diagnostics prove the model can learn the actual supervised
  assistant/media tokens.
- Pipeline sample loss checks heldout loss with the same sparse-target
  semantics as training.
- Batch prediction harnesses produce local-dev or authorized benchmark
  predictions.
- Release gates prevent canaries or private fixtures from being reported as
  public benchmark scores.

## Runtime

The primary deployment target is a q4 GGUF-compatible artifact for llama.cpp
and LM Studio style runtimes. ONNX/Core ML/ExecuTorch/MLC paths remain useful
secondary bridges, but they are not the main release target for the 2026 model.

## Current Limits

This repo has architecture and training infrastructure, not a release-quality
public checkpoint. Real claims still require:

- full-scale curated training,
- scored benchmark artifacts,
- 8K -> 1M context curriculum validation,
- media decoder quality evaluation,
- q4/GGUF runtime validation on target hardware.
