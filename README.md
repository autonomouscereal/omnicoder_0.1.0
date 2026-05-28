# OmniCoder 2026

OmniCoder 2026 is an experimental dense, one-trunk omnimodal agent-model
stack. The project explores a single shared reasoning trunk that can learn
text, code, tool use, images, video, speech/TTS, OCR, audio, music, and
long-context behavior through one token space and one set of model weights,
with modality-specific codecs and decoders kept at the edge.

This repository is an engineering and research workspace. It contains the
architecture, data curation, training orchestration, evaluation harnesses, and
runtime/export plumbing needed to build and validate the model. It does not
currently publish a release-ready checkpoint or reportable public benchmark
claim.

## Status

Current direction:

- Dense 20B-class decoder trunk, designed around q4-aware training and
  deployment.
- Native 1M-context architecture target with sparse/compressed attention and
  staged context-length validation.
- Unified ledger/token contract for text, tools, media artifacts, and typed
  modality routes.
- Assistant/media target masking so supervised loss covers answer tokens and
  media artifact tokens instead of arbitrary prompt positions.
- GGUF/llama.cpp deployment as the primary adoption target, with other
  runtimes treated as secondary bridges.

What is proven in this repo:

- The package imports and core routing/tests run locally.
- The 20B pipeline can load as sharded stages on the AI-server training image.
- Fresh-init scratch diagnostics can learn image, video, music, TTS, and OCR
  route/media targets when the target tokens are covered.
- Evaluation and benchmark harnesses can materialize gated local predictions.

What is not proven yet:

- Release-quality omnimodal generation.
- Reportable public benchmark scores.
- Full 8K -> 1M context curriculum completion.
- Final q4 single-card GGUF runtime and latency validation.

## Architecture

### Current 20B Target

The production-size contract is `omnicoder2026_20b_1m`, defined in
`src/omnicoder/config_2026.py` and implemented in
`src/omnicoder/modeling/omnicoder2026.py`.

| Field | Current value |
|---|---:|
| Architecture | dense one-trunk decoder |
| Layers | 64 |
| Width | 4096 |
| Heads | 32 query heads |
| Head dim | 128 |
| KV style | shared K=V / MQA-style sparse branches |
| MLP | SwiGLU, 16384 hidden |
| Vocab ledger | 330000 typed tokens |
| Native context target | 1048576 tokens |
| Layer cycle | `kda, kda, kda, csa, kda, kda, kda, hca` |

Expanded over 64 layers, the trunk is 48 KDA recurrent-linear layers, 8 CSA
compressed sparse attention layers, and 8 HCA heavily compressed attention
layers. The model is dense: there is no MoE router or fused expert dispatch in
the current target path.

### Long Context

The 1M-context design does not attempt full quadratic attention or full resident
GQA KV cache at every layer. The active long-context path combines:

- KDA/Gated-DeltaNet-2 recurrent-linear layers for the dominant memory path.
- Exact local causal attention over a 128-token window on CSA/HCA layers.
- CSA global recall with `compress_rate=4`, prefix retention, causal recent
  top-k block selection, low-rank Q projection, grouped low-rank output, sink
  logits, and shared K=V summaries.
- HCA long-range trail with `compress_rate=128` for cheap coarse memory.
- Partial RoPE on trailing head dimensions with base `1000000.0`.
- Chunked prefill/runtime expectations for native 1M validation.

This is a native-runtime design. The GGUF path is an adoption bridge for
standard local runners and shorter-context compatibility; true native 1M needs
the OmniCoder 2026 KDA/CSA/HCA runtime path unless llama.cpp grows equivalent
custom support.

### Omnimodal Token Space

```text
text, code, tool traces, media specs, long-context records
        |
        v
typed ledger tokens and canonical training records
        |
        v
dense OmniCoder 2026 decoder trunk
        |
        v
shared token head and route-aware output contract
        |
        v
edge decoders, artifact renderers, tools, benchmarks, and exports
```

All modalities enter the trunk as integer token IDs from one typed ledger and
share the same embeddings, blocks, and token head. The ledger ranges cover:

- `text`: text/code tokens.
- `control`: system, task, style, routing, and boundary tokens.
- `vision_semantic` and `vision_residual`: image/video semantic and detail
  tokens.
- `speech_tts`: speech, prosody, speaker, and TTS codec tokens.
- `audio_music` and `music_control`: audio, music, tempo, key, stem, and
  arrangement tokens.
- `time_space`: frame, pixel-grid, spectrogram, and alignment tokens.
- `tool_agent`: tool calls, memory, terminal, verifier, and agent actions.
- `flow`: mask, denoise, edit-span, and refinement-control tokens.

Media routing is visible in the generated stream, for example `image |`,
`video |`, `music |`, `tts |`, and `ocr |`. The trunk learns to emit the route
and artifact token stream; edge codecs/renderers convert raw pixels, waveforms,
video, and music to and from ledger packets. There are no learned modality
adapters inside the trunk, but raw media codecs still live at the edge.

### Auxiliary Heads

The main output is the shared tied token head. The target model also defines:

- `flow_head` for continuous latent/flow supervision used by media planning and
  reconstruction-style targets.
- `grounding_head` for spatial or modality grounding supervision.
- `sync_head` for temporal/audio-video synchronization signals.
- `mtp_heads` support in the config, currently set to `0` for the active 20B
  target.

### Speed And Memory Optimizations

The current implementation includes:

- Dense depth-first 20B-class sizing instead of MoE routing overhead.
- KDA recurrent-linear layers to avoid per-token KV growth in most layers.
- CSA/HCA compressed sparse global memory instead of full attention at 1M.
- MQA-style shared K=V sparse branches.
- Low-rank Q and grouped low-rank output projections on CSA/HCA layers.
- Chunked sparse/global attention gathers and chunked LM loss.
- Chunked SwiGLU forward/backward support for memory pressure.
- Activation checkpointing in the pipeline trainer.
- Three-rank fast-card pipeline placement with the current 20B lane using
  `16,16,32` layer placement.
- GPipe fallback for one-microbatch stability, with 1F1B reserved for validated
  multi-microbatch runs.
- Low-memory Adafactor optimizer-in-backward path.
- Q4 fake-quant/QAT hooks with chunked fake-quant linear support.
- Docker/NCCL `ipc: host` and explicit distributed timeout/checkpoint sync
  controls for the AI-server training lane.

### Explicit Non-Claims

- Residual attention as a full attention-logit accumulator is not implemented.
  What exists is `mHC-lite`: trainable gated residual/depth scaling around the
  mixer and FFN updates.
- Samba/Mamba/SSM blocks are not in the active 20B target. The closest piece is
  KDA/Gated-DeltaNet-2 recurrent-linear memory.
- YaRN is not the active 1M solution. Older legacy paths mention YaRN/PI, but
  the 2026 target relies on KDA plus CSA/HCA compressed state and still needs
  the staged 8K -> 1M validation ladder.
- Stock GGUF/llama.cpp is not claimed to run full native 1M KDA/CSA/HCA without
  native support. GGUF remains the compatibility/export goal.
- Release-quality image/video/TTS/music generation is not claimed until decoded
  artifacts and scored benchmarks pass.

Key code areas:

- `src/omnicoder/modeling/` - core OmniCoder 2026 model, attention, memory, and
  quantization code.
- `src/omnicoder/training/` - dense pretraining, sharded pipeline training,
  posttraining, and checkpoint handling.
- `src/omnicoder/data_factory/` - curation, trace ingestion, teacher-job
  manifests, integrity filtering, and dataset materialization.
- `src/omnicoder/eval/` - sample loss, target-token diagnostics, benchmark
  plumbing, prediction gates, and release checks.
- `src/omnicoder/inference/` - generation, output routing, runtime helpers, and
  artifact routing.
- `src/omnicoder/export/` - GGUF and other runtime/export bridges.
- `profiles/` - stable config profiles for training, teachers, benchmarks, and
  deployment targets.
- `scripts/` - AI-server orchestration helpers and operational launchers.

## Install

Use a virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[onnx,vision,audio,eval,gen]"
```

Do not commit credentials, private service URLs, API tokens, downloaded model
weights, or local dataset paths.

## Quick Checks

Package/import smoke:

```bash
python -m omnicoder.inference.generate --prompt "Hello OmniCoder" --device cpu
```

Core unit tests:

```bash
python -m pytest tests -q
```

Validate a training orchestration profile:

```bash
training-orchestration-2026 --profile profiles/training_orchestration_2026.json validate
```

Inspect dataset integrity tooling:

```bash
python -m omnicoder.data_factory.dataset_integrity_2026 --help
```

## Main Workflows

Data curation:

```bash
full-harness-2026 run \
  --profile profiles/training_harness_2026.json \
  --trace-input data/raw/agent_memory_events_2026.jsonl \
  --stages ingest_trace,quality_score,contam_scan,export_sft,teacher_jobs \
  --dry-run
```

Dense training entry point:

```bash
pretrain-2026-dense --help
```

Pipeline sample-loss evaluation:

```bash
python -m omnicoder.eval.pipeline_sample_loss_2026 --help
```

Target-token diagnostics:

```bash
python -m omnicoder.eval.pipeline_target_token_diagnostics_2026 --help
```

Batch prediction harness:

```bash
python -m omnicoder.eval.pipeline_checkpoint_batch_predict_2026 --help
```

GGUF export bridge:

```bash
python -m omnicoder.export.gguf_bridge_2026 --help
```

## Documentation

Start here:

- `docs/Omnicoder2026Redesign.md` - active architecture redesign.
- `docs/ARCHITECTURE_CURRENT.md` - concise current architecture summary.
- `docs/DatasetCuration2026.md` - curation and quality contract.
- `docs/TrainingOrchestration2026.md` - training orchestration and AI-server
  runbook.
- `docs/OMNIMODAL_MASKING_CONTRACT_2026.md` - loss-mask and shared-trunk
  training contract for media/text/tool targets.
- `docs/DistillationAndRL2026.md` - teacher distillation and RL/posttraining
  plan.
- `docs/BenchmarkSuite2026.md` - benchmark registry and release-gate contract.

Historical run logs and dated experiment notes should stay in deep docs or
archives, not in the top-level README.

## Benchmarking

The benchmark code distinguishes engineering diagnostics from reportable
scores. Local canaries, synthetic probes, and private dev snapshots are useful
for catching regressions, but they are not leaderboard results. Reportable
results require authorized snapshots, scored prediction artifacts, manifests,
and release-gate metadata.

## Deployment Target

The adoption target is a q4 GGUF-style artifact that can run through standard
local runtimes such as llama.cpp and LM Studio without custom user patches.
The repo also carries ONNX/Core ML/ExecuTorch/MLC experiments, but those are
secondary to the GGUF deployment path for the 2026 model.

## License

See `LICENSE` for repository licensing. Model weights, downloaded datasets, and
teacher outputs may have separate terms; keep their manifests and provenance
with the artifacts that use them.
