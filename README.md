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

The active model path is dense, not sparse MoE. Modality routing is represented
inside the training and generation contract with visible route prefixes and
typed artifact tokens such as image/video/music/TTS/OCR outputs. The trunk is
trained to emit those tokens; external decoders consume them after generation.

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
