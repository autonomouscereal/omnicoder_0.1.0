# Omnicoder 2026 Redesign

Date: 2026-05-23

This document supersedes the old sparse-MoE, SFB/Omega, and ONNX-first
direction. Omnicoder v0.2 is a dense one-trunk, native-1M-context, omnimodal
agent model. The design is source-audited against 2025-2026 papers, model cards,
and release docs; unverified claims stay out of the core architecture.

## Executive Decision

Build Omnicoder as:

```text
raw text/media/tools
  -> edge codecs and typed ledger packets
  -> one dense KDA/CSA/HCA decoder trunk
  -> shared token head + flow/grounding/sync heads
  -> edge decoders, renderers, tools, and GGUF/native runtimes
```

The trunk is dense and shared across text, code, tools, images, video, speech,
audio, and music. It has one embedding table and one block stack. It does not
use MoE fused dispatch, SFB, Q-Former-style in-trunk adapters, CLIP projection
adapters, or separate per-modality reasoning towers.

The edge still needs codecs/tokenizers for pixels, video, waveforms, speech, and
music. Current frontier systems also do this: GPT Image 2, Gemini Omni, Qwen3-
Omni, Cosmos, LTX, and ACE-Step all rely on modality-specific compression or
rendering at the boundary. The fused part is the ledger/trunk/decision space.

## Source Audit

| Claim | Status | Design Use |
|---|---:|---|
| DeepSeek V4 CSA/HCA/mHC | Primary verified, vendor-claimed performance | Use CSA/HCA/mHC primitives, not DeepSeek's MoE scale. |
| Kimi Linear/KDA | Paper and repo verified | Use 3:1 KDA-to-global rhythm and recurrent-linear memory layers. |
| Kimi K2.6 | Primary verified | Treat as a strong current multimodal/agent reference, not proof of KDA internals. |
| MiniMax M2.7 | Primary verified text-agent model | Use as agent-training/long-context reference, not omnimodal generator. |
| MiniMax-01/M1 Lightning Attention | Paper verified | Keep as research reference; do not assume M2.7 uses it. |
| GPT Image 2 / `gpt-image-2` | Primary verified | Use for image generation/editing distillation and eval rubrics where permitted. |
| Gemini Omni | Primary verified | Use as any-input to video-with-audio reference. |
| Gemma 4 | Primary verified | Use as edge/mobile multimodal-input reference, not any-output generation. |
| Grok 4.3 / Imagine / Voice | Primary verified APIs | Use capability-routing lessons; no official architecture claim assumed. |
| Qwen3.6 | Primary verified | Use local Qwen3.6 27B Q4 as a teacher if present. |
| Qwen3.7 | Not primary-verified | Do not bake into architecture. |
| Qwen3-Omni | Primary verified | Use Thinker/Talker, RVQ speech, and temporal multimodal alignment lessons. |
| Qwen3.5-Omni | Not primary-verified | Treat as likely confusion with Qwen2.5-Omni/Qwen3-Omni. |

Primary anchors:

- DeepSeek V4 docs/model cards: https://huggingface.co/docs/transformers/main/model_doc/deepseek_v4 and https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
- Kimi Linear: https://arxiv.org/abs/2510.26692 and https://github.com/MoonshotAI/Kimi-Linear
- Kimi K2.6: https://platform.kimi.ai/docs/guide/kimi-k2-6-quickstart
- MiniMax M2.7: https://www.minimax.io/models/text/m27 and https://github.com/MiniMax-AI/MiniMax-M2.7
- GPT Image 2 docs: https://developers.openai.com/api/docs/guides/image-generation
- Gemini Omni: https://deepmind.google/models/gemini-omni/
- Gemma 4: https://deepmind.google/models/gemma/gemma-4/
- xAI Grok/Imagine/Voice: https://docs.x.ai/developers/models and https://docs.x.ai/developers/model-capabilities/imagine
- Qwen3.6/Qwen3-Omni: https://github.com/QwenLM/Qwen3.6 and https://github.com/QwenLM/Qwen3-Omni
- NVIDIA Cosmos/Nemotron/Curator: https://github.com/NVIDIA/Cosmos-Tokenizer, https://research.nvidia.com/labs/nemotron/Nemotron-3-Super/, and https://docs.nvidia.com/nemo/curator/latest/home/welcome
- TurboQuant/runtime successors: https://arxiv.org/abs/2504.19874, https://arxiv.org/abs/2605.19660, https://arxiv.org/abs/2605.11478, and https://vllm.ai/blog/2026-05-11-turboquant

## Architecture

Production target: `omnicoder2026_dense_kda_csa_hca_mhc_one_trunk`.

Layer cycle:

```text
[kda, kda, kda, csa, kda, kda, kda, hca] x depth
```

The 64-layer primary target expands to 48 KDA layers, 8 CSA layers, and 8 HCA
layers. Smaller 3B/16B profiles keep the same ratio as pilots and fallbacks.

Core pieces:

- `kda`: Kimi Linear-style KDA/Gated-DeltaNet-2 recurrent-linear layers. The
  current correctness path keeps recurrent state in fp32 and has chunk-equivalent
  tests; production kernels can replace the module behind the same block API.
- `csa`: DeepSeek V4-style compressed sparse attention. It combines short exact
  local attention, shared K=V MQA, low-rank Q, grouped low-rank output, trailing
  partial RoPE, sink logits, `compress_rate=4` summaries, prefix retention, and
  causal top-k retrieval.
- `hca`: shared K=V heavily compressed causal summaries at `compress_rate=128`
  for a cheap long-range trail.
- `mHC-lite`: trainable residual/depth gate now, with a production slot for
  Sinkhorn/hyper-connection kernels later.
- `flow_head`: continuous latent targets for image/video/audio/music planning and
  reconstruction losses.
- `grounding_head` and `sync_head`: spatial/time grounding and audio-video sync
  supervision.

## Presets

| Preset | Purpose | Layers | Width | MLP hidden | Pattern | Context |
|---|---:|---:|---:|---:|---|---:|
| `omnicoder2026_native1m_probe` | construction/server probe | 4 | 512 | 1408 | KDA/KDA/CSA/HCA | 1,048,576 |
| `omnicoder2026_full_ledger_probe` | AI-server full-token-ledger training verifier | 4 | 512 | 1408 | KDA/KDA/CSA/HCA | 1,048,576 |
| `omnicoder2026_3b_pilot` | data and QAT pilot | 32 | 2560 | 6912 | 8-layer cycle | 1,048,576 |
| `omnicoder2026_20b_1m` | primary 24GB Q4 native-1M target | 64 | 4096 | 15360 | 8-layer cycle | 1,048,576 |
| `omnicoder2026_16b_1m` | fallback/intermediate profile, not the contract target | 48 | 4096 | 15360 | 8-layer cycle | 1,048,576 |

The active contract target is 20B-ish, not a fixed vanity size. The current
profile keeps `omnicoder2026_20b_1m` at `mlp_dim=15360`, about 19.57B trunk
parameters and about 22.58B total parameters once the enabled MTP heads, block
residual modules, native media bridge, adaptive latent reasoner, flow head,
grounding head, and sync head are counted. That is about 10.51 GiB of Q4 weights and about 13.13 GiB total
native-state estimate for 1M context. The intentional MLP reduction from the
earlier wider MLP budget preserves 24GB headroom for full block residual
attention, native media heads, runtime scratch, and TurboQuant/compressed-state
variance. The small probe is only a full-ledger training and orchestration
verifier; it is not presented as the production model.

## Token Ledger

The vocabulary is a versioned ledger with typed ranges:

| Range | Tokens | Role |
|---|---:|---|
| `text` | 0..127999 | text/code BPE |
| `control` | 128000..132095 | system/task/style/boundary |
| `vision_semantic` | 132096..197631 | low-rate image/video semantics |
| `vision_residual` | 197632..214015 | image/video detail tokens |
| `speech_tts` | 214016..279551 | speech semantic/prosody/codec |
| `audio_music` | 279552..312319 | sound/music codec |
| `music_control` | 312320..320511 | beat/key/tempo/stem/instrument |
| `time_space` | 320512..324607 | time/frame/space/alignment |
| `tool_agent` | 324608..328703 | tool, memory, terminal, verifier |
| `flow` | 328704..329999 | mask/refinement/edit-step tokens |

The trunk receives token IDs and supervised continuous targets. Edge codecs
convert raw media into ledger packets and convert generated packets back into
assets. That keeps the model fused without pretending raw waveforms and pixels
are cheap text tokens.

## Native 1M Context

The memory budget is:

```text
Q4 weights
+ KDA recurrent state
+ CSA/HCA shared K=V compressed latent state on global layers only
+ exact 128-token local windows
+ runtime scratch/offload
```

Explicitly rejected:

- full resident GQA KV for every layer at 1M on a 24GB card;
- YaRN-only claims without trained/evaluated 1M behavior;
- Triton/Inductor/CUDA graphs as correctness requirements;
- stock GGUF as the full native-1M runtime;
- MoE fused dispatch.

The native runtime must support chunked prefill, paged KDA state, sparse
prefix/recent or learned-indexer latent gather, cache quantization gates, and
OpenAI-compatible serving. GGUF remains an adoption bridge for text/tool
checkpoints at shorter context.

## Data Factory

Data and eval metadata live in raw PostgreSQL only. No ORM, Pydantic,
SQLAlchemy, SQLite, or Chroma.

Pipeline:

```text
ingest -> hash/provenance -> modality curation -> quality ledger
       -> contamination scan -> teacher synthesis -> verifier pass
       -> split/materialized training view -> train -> eval quarantine
```

Training mixtures:

- pretraining: text/math/code, media caption/alignment, generation/edit pairs,
  function/tool trajectories, agent traces, and safety/governance;
- SFT: tool traces, software engineering patches/tests, terminal tasks,
  multimodal editing workflows, audio/music reasoning, grounded research;
- RLVR: unit tests, terminal oracles, schema-valid tool calls, browser/UI state,
  ARC-like interactive rewards, and media consistency rewards.

Benchmark prompts, answer keys, hidden tests, private levels, successful eval
trajectories, and judge rubrics are always `eval_protected` or `quarantine`.

## Training Plan

1. Build and verify the 4-layer full-ledger native-1M probe locally and on the
   AI server.
2. Use the data factory to ingest existing traces and build 2025-2026 curated
   buckets with contamination quarantine.
3. Train a 3B pilot on ledger-encoded text/code/tool/media traces.
4. Distill from local teachers: Qwen3.6 27B Q4, Qwen3-Omni, Qwen Image/Edit,
   LTX 2.3, ACE-Step 1.5, and `gpt-image-2` where available/licensed.
5. Scale to the `omnicoder2026_20b_1m` primary target after eval gates are
   stable, using sharded QAT/LoRA-to-full recovery rather than fp16 single-card
   full training.
6. Use SFT for format/tool discipline, preference tuning for localized repair,
   and GRPO/DAPO/Dr.GRPO RLVR only where rewards are deterministic.
7. Finish with fake-Q4/QAT or PTQ+recovery, then quantized eval before release.

## Evaluation

The 2026 harness is registry-based:

- ARC-AGI-3 for interactive reasoning and adaptive world modeling;
- SWE-bench Pro and SWE-bench Live/internal live tasks for repository repair;
- Terminal-Bench 2.x for shell/container autonomy;
- BFCL v4, tau2/tau3, MCPMark/MCP-Atlas for tool calling and stateful agents;
- LiveCodeBench for contamination-aware code generation;
- MMMU-Pro and Video-MME/LVOmniBench/JointAVBench for multimodal reasoning;
- GenEval 2, BizGenEval, CompBench, Pico-Banana-style private slices for
  image generation/editing;
- VBench/VBench++, AVGen-Bench, T2AV-Compass, AudioBench, AudioCapBench, Music
  Arena/Coval-style probes for video/audio/music/TTS.

Every run records model hash, ledger version, codec versions, quant type,
runtime, context length, cache type, prompt template, artifacts, score, latency,
and contamination scan version.

## Deployment

Two release tracks:

1. **Native runtime:** full KDA/CSA/HCA recurrent/sparse-latent 1M behavior.
2. **GGUF bridge:** qwen3-compatible text/tool model for LM Studio and llama.cpp
   adoption, with explicit shorter-context limits and external media tooling.

The GGUF bridge must run in unmodified LM Studio/llama.cpp. The native runtime
can be more aggressive, but it must expose ordinary OpenAI-compatible routes.
