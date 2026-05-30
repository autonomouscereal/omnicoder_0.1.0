# OmniCoder Current Architecture

This document describes the current implemented architecture contract for
OmniCoder 2026. It is intentionally separate from the README so the README can
stay public-facing while this file carries the technical design details.

The deeper research narrative lives in `docs/Omnicoder2026Redesign.md`.

## Design Goals

OmniCoder 2026 is being built around these priorities:

1. Maximize model quality and reasoning capability.
2. Preserve practical speed and memory behavior for 24GB-class local hardware.
3. Keep a single dense trunk for text, code, tools, long context, and media.
4. Train media and text through shared representations instead of isolated
   per-modality mini-models.
5. Export to q4/GGUF-style local runtimes where possible, while retaining a
   native runtime path for features standard runtimes do not support yet.

The current design favors intelligence over speed when the tradeoff is real.
Speed features are kept when they reduce memory or latency without removing the
shared modeling capacity needed for agentic and omnimodal behavior.

## High-Level Flow

```text
raw sources
  text, code, tools, traces, images, video, OCR, speech, audio, music
        |
        v
curation and canonicalization
  quality gates, contamination gates, modality tags, target coverage
        |
        v
typed ledger and native media packets
  text tokens, tool tokens, route tokens, media artifact tokens,
  direct continuous patch/segment features
        |
        v
dense OmniCoder 2026 trunk
  KDA recurrent layers, CSA/HCA sparse latent attention,
  block residual attention, shared FFN depth
        |
        v
shared outputs
  LM tokens, MTP logits, flow/head signals, native media reconstruction
        |
        v
edge systems
  output router, media decoders/renderers, tools, benchmarks,
  native runtime, GGUF bridge
```

The trunk is shared. Media codecs, renderers, OCR engines, audio/video
container handling, and deployment bridges sit outside the trunk.

## 20B-Class Target

The primary training target is `omnicoder2026_20b_1m`, defined in
`src/omnicoder/config_2026.py` and implemented in
`src/omnicoder/modeling/omnicoder2026.py`.

| Field | Current value |
|---|---:|
| Architecture | Dense one-trunk decoder |
| Layers | 64 |
| Hidden width | 4096 |
| Query heads | 32 |
| Head dim | 128 |
| KV heads on sparse attention paths | 1 |
| MLP | SwiGLU |
| MLP hidden | 15360 |
| Estimated trunk parameters | ~=19.57B |
| Estimated total parameters with MTP/media/residual/reasoning heads | ~=22.58B |
| q4 weight estimate | ~=10.51 GiB |
| Typed ledger vocab | 330000 |
| Native context target | 1048576 tokens |
| Layer cycle | `kda, kda, kda, csa, kda, kda, kda, hca` |
| Residual mode | `block_attnres` |
| MTP heads | 2 |
| Native media feature dim | 3072 |
| Native media position dim | 4 |
| Native media type vocab | 16 |

Expanded across 64 layers, the layer cycle gives:

- 48 KDA recurrent-linear layers.
- 8 CSA compressed sparse attention layers.
- 8 HCA heavily compressed attention layers.

The model is intentionally dense. There is no active MoE router or fused expert
dispatch in the current 20B target.

The MLP hidden width is intentionally held at 15360 rather than the earlier
wider MLP budget. This keeps the dense trunk at ~=19.57B parameters while the
enabled MTP heads, block residual modules, native media bridge, adaptive latent
reasoner, flow head, grounding head, and sync head bring the honest target
footprint to ~=22.58B
parameters and a ~=10.51 GiB q4 weight estimate. The point of the 15360 MLP is
not to shrink the model into a weak target; it preserves the headroom needed for
full residual attention, native media heads, and OOM margin instead of chasing a
rounder FFN count.

## Long Context

Native 1M context is not implemented as full dense attention. The current
runtime target combines several memory-bounded mechanisms:

- KDA/Gated-Delta-style recurrent-linear layers for the dominant long-range
  state path.
- CSA layers for compressed sparse global recall.
- HCA layers for cheaper coarse long-range trail memory.
- Local causal attention where exact short-range mixing matters.
- Partial RoPE on trailing head dimensions.
- Chunked prefill, chunked loss, and sparse/global gathers.
- Block residual attention over compressed residual-stream summaries.

This is the core reason the native 1M path is not the same as the GGUF bridge.
Standard local runtimes can carry the q4 short-context compatibility path, but
full native 1M behavior requires the OmniCoder KDA/CSA/HCA state scheduling
unless those runtime features are implemented upstream.

## Residual Attention

The current implemented residual-attention path is `BlockAttentionResidual`.
It is the active default through `residual_mode="block_attnres"`.

Configuration in the 20B target:

- Block size: 128 tokens.
- Maximum summary blocks: 1024.
- Low-rank attention rank: 256.
- Chunk size: 2048 tokens.

Mechanism:

1. Each layer keeps the normal residual update path.
2. The incoming residual stream is summarized into causal blocks.
3. The current update forms low-rank queries.
4. Queries attend to causally visible summary blocks.
5. The residual-context signal is gated and added back to the layer output.
6. The learned scale starts near zero so the path begins close to identity.

This is not a full depth-token attention-logit accumulator. A full residual
attention cache over every token and every layer would be too expensive for the
1M-context/24GB contract. The implemented path is the current hardware-aware
variant: it keeps a residual selection signal while bounding memory by summary
blocks instead of by sequence length times depth.

## Adaptive Latent Reasoning

The current trunk now includes a shared latent reasoner cell for optional
internal deliberation. This is separate from MTP. MTP predicts future tokens for
speed/speculative decoding; the reasoner adds repeated continuous hidden-state
refinement before the final norm/head.

Configuration in the 20B target:

- Latent slots: 8.
- Maximum internal steps: 8.
- Default steps: 0, so normal training/inference stays at baseline cost unless
  a run requests reasoning effort.
- Cell rank: 512 low-rank bottleneck.
- Pool tokens: 1024 bounded context summaries.
- Control heads: difficulty, halt/continue, answer-readiness, verifier margin,
  and tool-readiness.

Mechanism:

1. The current sequence hidden state is pooled into a bounded context summary.
2. Shared learned slots attend to that pooled context through a small low-rank
   update cell.
3. The same cell is reused for 0 to 8 latent steps depending on requested
   reasoning effort.
4. The refined slots produce a low-rank broadcast update back into the hidden
   stream.
5. The learned output scale starts small, keeping the path near identity until
   training teaches it useful effort-dependent behavior.

This keeps reasoning as parameter-shared compute depth rather than adding
another full-vocab verifier head. A 4096-by-330k extra head would be too large
for the 24GB q4 target unless capacity were removed elsewhere.

## Omnimodal Representation

OmniCoder has two media training contracts that share the same trunk.

### Route And Artifact Tokens

Media can be represented as ordinary model output tokens:

- `image | ...`
- `video | ...`
- `audio | ...`
- `music | ...`
- `tts | ...`
- `ocr | ...`

The route prefix, structured artifact descriptor, and media token stream are
all supervised through the shared LM head. The output router then hands the
generated route and artifact payload to edge systems.

This path is useful for deployable artifact generation and for keeping
standard text/tool runtimes understandable.

### Native Continuous Media

The SenseNova-U1-inspired path is a direct patch/segment lane for media:

- Image: patch features with spatial metadata.
- Video: frame/patch features with time and grid metadata.
- Audio: waveform or spectrogram window features.
- Music: audio/music windows with shared time-frequency metadata.
- TTS/speech: speech windows plus speaker/prosody metadata where available.
- OCR: document/page/crop segments with layout metadata.

The implementation is split between:

- `src/omnicoder/tokenization/native_media_2026.py`
- `src/omnicoder/tokenization/native_segments_2026.py`
- `NativeContinuousMediaBridge` in `src/omnicoder/modeling/omnicoder2026.py`

The bridge uses one shared feature projection, one shared position projection,
and a media-type embedding. It does not add separate learned image, video,
audio, music, TTS, or OCR adapters inside the trunk.

Native media features are added to aligned trunk positions. The model can emit:

- normal shared-token outputs through the LM head,
- flow/grounding/sync auxiliary outputs,
- `native_media_reconstruction` plus `native_media_loss` when continuous media
  targets are supplied.

This proves a trainable path for media features. It does not by itself prove
release-quality raw media generation; that still requires full data scale,
decoded artifact evaluation, and modality-specific quality gates.

## Shared-Trunk Training

The target-mask contract is designed to avoid building separate modality
models inside one checkpoint.

The model sees the full prompt/context sequence across modalities. The mask
only decides where supervised loss is charged:

- User/system/developer/tool-observation prompt spans are usually context only.
- Assistant answer tokens are supervised.
- Tool-call tokens are supervised when they are the intended model action.
- Media route and artifact tokens are supervised.
- Native continuous media targets use the shared media reconstruction loss.
- Benchmark inputs and protected eval material stay out of trainable targets.

So masking does not hide modalities from each other. Text can condition media,
media can condition text, tool traces can condition both, and long-context
records can condition all later targets. The mask prevents the trainer from
rewarding prompt copying or benchmark leakage.

## Auxiliary Heads

The model has these output surfaces:

- Shared tied LM head for text, code, tools, route tokens, and media artifact
  tokens.
- MTP heads for future speculative/multi-token decoding validation.
- Flow head for media/refinement supervision.
- Grounding head for spatial or modality grounding signals.
- Sync head for temporal/audio-video alignment signals.
- Native media reconstruction head for direct continuous media targets.

MTP is wired, but production speedup claims require training, acceptance
testing, and quality checks against the base decoder. It should be treated as a
validation feature until those gates pass.

## Speed And Memory Choices

Current speed/memory features:

- Dense depth-first trunk instead of MoE routing.
- KDA recurrent-linear layers to reduce KV growth.
- CSA/HCA sparse global memory instead of full attention at 1M.
- Single-KV compressed attention paths.
- Low-rank projections in sparse/global attention components.
- Chunked loss to avoid full sequence-by-vocab materialization.
- Activation checkpointing in the dense pipeline trainer.
- Low-memory optimizer-in-backward Adafactor path.
- Weighted pipeline placement for fast-card training layouts.
- Q4 fake-quant and QAT hooks for recovery validation.
- MTP heads for future speculative decoding experiments.

Not currently claimed as implemented production paths:

- Samba/Mamba-style state-space blocks.
- YaRN as the main 1M solution.
- DFlash or other 2026 inference kernels.
- Full TurboQuant integration as a complete paper-faithful runtime.
- Stock llama.cpp full native 1M support for KDA/CSA/HCA state.

The q4/TurboQuant lane currently means q4-aware fake quant, recovery training,
and export/runtime validation hooks. It is not a claim that final TurboQuant
runtime speedups have already been integrated and benchmarked.

## Profiling And Diagnostic Readiness

The current training harness has instrumentation for proving where time,
memory, and learning signal are going before another long run is trusted.

Training-step timing can record:

- batch fetch and host-to-device movement,
- input broadcast and pipeline schedule time,
- loss broadcast and optimizer/update time,
- diagnostics, telemetry, log-write overhead, and rank skew.

Checkpoint timing can record:

- state-dict collection,
- CPU copy and optimizer-state collection,
- temporary save and atomic rename,
- marker waits, manifest writes, total bytes, and write throughput.

The 20B profile matrix helper can run bounded no-checkpoint variants for q4
fake quant chunk size, fake-quant-off profiling, activation checkpointing,
pipeline schedule/microbatch choices, target-token budget, and optional block
timing. Fake-quant-off is only allowed for no-save profiling runs that set the
explicit profiling bypass; production and release-contract runs still fail
closed if q4/QAT requirements are skipped.

Checkpoint eval sidecars are intended to run asynchronously after a complete
checkpoint marker. They cover heldout sample loss, target-token diagnostics,
token top-k sanity, decode sanity, media route probes, native media
reconstruction checks, and benchmark adapter handoff. These are engineering
readiness probes, not public benchmark scores.

## Data Gates

The training stack now treats data quality as part of the architecture contract.
Rows intended for training must preserve enough metadata to answer:

- Where did this row come from?
- What modality and target family does it train?
- Is it train, eval, benchmark, quarantine, or research-only material?
- Did quality scoring pass?
- Did contamination and benchmark-holdout scans pass?
- Does it contain nontrivial assistant/media target coverage?

Rows with missing quality metadata, unknown/suspect contamination, fixture or
example paths, protected benchmark markers, placeholder text, one-token junk,
or degenerate prompt-copy targets are rejected or quarantined.

## Evaluation Gates

Evaluation is split into diagnostic and reportable lanes.

Diagnostic gates:

- Import and model-construction checks.
- Target-token diagnostics.
- Heldout sample loss with non-null loss/perplexity.
- Batch decode probes.
- Media route and artifact parsing for image, video, audio, music, TTS, and
  OCR.
- Native media reconstruction checks.

Reportable gates:

- Authorized benchmark snapshots.
- Model-generated predictions.
- Official or benchmark-native scorer output.
- Immutable manifests.
- Contamination and leakage reports.
- Release-gate metadata that prevents canaries from being reported as scores.

The benchmark suite is allowed to fail closed when official scorers or
authorized snapshots are not present.

## Runtime And Export

There are two runtime tracks:

1. Native OmniCoder runtime for full KDA/CSA/HCA, block residual attention,
   native media features, and the 1M context curriculum.
2. GGUF/qwen-compatible bridge for local adoption through llama.cpp and LM
   Studio style tools.

The project target is an easy q4 GGUF-style deployment. The current technical
reality is that full native 1M and media reconstruction paths need custom
runtime support unless standard runtimes add equivalent primitives.

## What Is Implemented Versus Proven

Implemented:

- Dense 20B-ish one-trunk config.
- KDA/CSA/HCA layer cycle.
- Block residual attention module.
- Native continuous media bridge.
- Adaptive latent reasoner cell and control heads.
- MTP heads.
- q4-aware training hooks.
- Assistant/media target masking.
- Data curation fail-closed gates.
- Diagnostic/reportable benchmark separation.
- Phase-timing, checkpoint-I/O, profile-matrix, and checkpoint-sidecar eval
  plumbing.

Proven only at engineering/probe scale so far:

- Model construction.
- Target coverage diagnostics.
- Scratch trainability of selected modality paths.
- Curation and benchmark plumbing behavior.

Not proven yet:

- Frontier-quality text, coding, agentic, image, video, TTS, audio, or music
  output.
- Completed 8K -> 1M context curriculum.
- Reportable public benchmark quality.
- Single-card q4 latency/memory on final exported weights.
- Release-ready decoded media quality.

Those are training, evaluation, and runtime validation goals, not current
release claims.
