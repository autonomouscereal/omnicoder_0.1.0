# OmniCoder 2026

OmniCoder 2026 is an experimental dense, one-trunk omnimodal agent model. It is
being built around one shared token/weight space for text, code, tools, OCR,
image, video, speech/TTS, audio, and music, while codecs, renderers, artifact
decoders, and deployment bridges stay at the edges.

This repository contains the model architecture, data curation stack, training
orchestration, diagnostics, benchmark gates, and export/runtime bridge work for
that target. It does not currently publish release-ready weights or reportable
public benchmark scores.

## Architecture

- Dense 20B-class decoder trunk targeting q4 deployment on 24GB-class GPUs.
- Native 1M-context target using KDA recurrent state, CSA/HCA compressed sparse
  attention, and block residual attention instead of full quadratic attention.
- SenseNova-U1-inspired native continuous media path for direct image, video,
  audio, music, TTS, and OCR patch/segment supervision.
- Shared typed ledger for text, code, tools, media route tokens, media artifact
  tokens, and native media segment alignment.
- Assistant/media target masking so loss is paid on answer and media artifact
  tokens, not arbitrary prompt positions.
- Adaptive latent reasoning slots, MTP heads, q4-aware training hooks, and GGUF
  export plumbing are wired for validation, with production claims gated on
  measured quality and runtime checks.

For the full layer-by-layer contract, see
[docs/ARCHITECTURE_CURRENT.md](docs/ARCHITECTURE_CURRENT.md).

## Current Readiness

Implemented in the repository:

- `omnicoder2026_20b_1m` architecture profile and dense pipeline trainer.
- KDA/CSA/HCA layer scheduling for the 1M-context runtime target.
- Block residual attention and native continuous media bridge code paths.
- Adaptive latent reasoner controls and MTP heads for reasoning/speed
  experiments.
- Target-token diagnostics for assistant, tool, route, and media-token
  coverage.
- Phase-timing, checkpoint-I/O, and checkpoint-sidecar eval hooks for finding
  slow internal phases before resuming large runs.
- Data curation gates for quality, contamination, fixture leakage, and
  benchmark-holdout separation.
- Benchmark adapters that separate diagnostic canaries from reportable scores.

Not yet proven:

- Release-quality omnimodal generation.
- Scored public benchmark results.
- Full 8K -> 1M context curriculum.
- q4 single-card runtime latency, memory, and GGUF compatibility.
- Real decoded image, video, TTS, audio, and music quality from trained weights.

## Quickstart

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

Do not commit credentials, private service URLs, downloaded model weights,
private datasets, generated checkpoints, or local artifact paths.

## Development Checks

Run package and unit checks:

```bash
python -m omnicoder.inference.generate --prompt "Hello OmniCoder" --device cpu
python -m pytest tests -q
```

Validate orchestration profiles:

```bash
training-orchestration-2026 --profile profiles/training_orchestration_2026.json validate
benchmark-suite-2026 --profile profiles/benchmark_suite_2026.json validate
```

Inspect core entry points:

```bash
pretrain-2026-dense --help
python scripts/ai_server_profile_matrix_20b.py --help
python -m omnicoder.eval.pipeline_sample_loss_2026 --help
python -m omnicoder.eval.pipeline_target_token_diagnostics_2026 --help
python -m omnicoder.export.gguf_bridge_2026 --help
```

## Data And Evaluation

Training data is JSONL-first and must preserve provenance, quality scores,
contamination status, modality tags, and split intent. The current curation
path rejects known fixture/example data, protected benchmark material, unknown
or suspect contamination, tiny/placeholder targets, and rows without required
quality metadata.

Benchmarking is split into two lanes:

- Diagnostic checks catch regressions in loading, decoding, sample loss, target
  coverage, and artifact routing.
- Reportable scores require authorized benchmark snapshots, model-generated
  predictions, scorer outputs, immutable manifests, and release-gate metadata.

Diagnostic canaries are useful engineering signals, but they are not public
benchmark claims.

## Repository Map

- `src/omnicoder/modeling/` - core model, attention, residual, media, and
  quantization modules.
- `src/omnicoder/training/` - dense pretraining, pipeline training,
  checkpointing, diagnostics, and orchestration hooks.
- `src/omnicoder/data_factory/` - curation, trace ingestion, teacher manifests,
  dataset expansion, and integrity filtering.
- `src/omnicoder/eval/` - sample loss, target-token diagnostics, benchmark
  adapters, prediction gates, and release checks.
- `src/omnicoder/inference/` - generation, route parsing, output routing, and
  runtime helpers.
- `src/omnicoder/export/` - GGUF and secondary runtime/export bridges.
- `profiles/` - stable model, data, training, benchmark, and deployment
  profiles.
- `scripts/` - local and AI-server orchestration helpers.

## Documentation

- [Current Architecture](docs/ARCHITECTURE_CURRENT.md)
- [Architecture Redesign Notes](docs/Omnicoder2026Redesign.md)
- [Dataset Curation](docs/DatasetCuration2026.md)
- [Training Orchestration](docs/TrainingOrchestration2026.md)
- [Omnimodal Masking Contract](docs/OMNIMODAL_MASKING_CONTRACT_2026.md)
- [Distillation And RL](docs/DistillationAndRL2026.md)
- [Benchmark Suite](docs/BenchmarkSuite2026.md)

Historical run logs and dated experiment notes should stay in deep docs or
archives, not in this README.

## Deployment Target

The adoption target is a q4 GGUF-style artifact that can run through common
local runtimes such as llama.cpp and LM Studio without user-side patches. The
native 1M-context KDA/CSA/HCA path may require OmniCoder-specific runtime
support until standard runtimes can carry the same state and scheduling.

ONNX, Core ML, ExecuTorch, and MLC experiments may remain useful secondary
bridges, but GGUF is the primary release path for local adoption.

## License

See [LICENSE](LICENSE) for repository licensing. Model weights, downloaded
datasets, generated media, and teacher outputs may have separate terms; keep
their manifests and provenance with the artifacts that use them.
