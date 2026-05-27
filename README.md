# OmniCoder 2026

OmniCoder 2026 is an experimental omnimodal agent-model research stack. The
current design target is a dense, one-trunk model family that can share a
reasoning core across text, code, tools, images, video, speech/audio, music,
and long-context state.

This repository is a research and engineering workspace, not a polished model
release. It contains model scaffolding, data-curation tools, training
orchestration, evaluation gates, export/runtime adapters, mobile samples, and
smoke tests. It does not publish a proven frontier-quality checkpoint, and the
local benchmark fixtures in this tree should not be reported as public
leaderboard results.

## Project Goals

- Build a compact omnimodal model architecture around one shared trunk instead
  of a permanent pipeline of unrelated specialist models.
- Preserve explicit modality boundaries at the edge with typed token/codecs,
  media artifacts, tool traces, and ledger-style training records.
- Support native long-context experiments with bounded memory, recurrent or
  compressed state, retrieval hooks, and q4-aware deployment planning.
- Keep training and evaluation auditable through manifests, hashes, curation
  records, contamination checks, and release-gate metadata.
- Make runtime/export work first-class: ONNX, Core ML, NNAPI-style runners,
  DirectML, ExecuTorch, GGUF/llama.cpp bridges, and mobile packaging all live
  in the repo.

## Architecture Overview

The active 2026 contract is documented in
[`docs/Omnicoder2026Redesign.md`](docs/Omnicoder2026Redesign.md). In short:

```text
raw text, code, media, tool traces, and long-context records
        |
        v
typed ledger tokens, modality codecs, and artifact manifests
        |
        v
dense KDA/CSA/HCA/mHC-inspired decoder trunk
        |
        v
shared token head plus flow, grounding, sync, verifier, and tool heads
        |
        v
edge decoders, runtimes, tool adapters, benchmarks, and export packages
```

Important implementation areas:

- `src/omnicoder/modeling/`: core transformer, attention, long-context memory,
  quantization, kernels, and the 2026 dense model path.
- `src/omnicoder/modeling/multimodal/`: image, video, audio, speech, grounding,
  fusion, VQ, latent, and decoder experiments.
- `src/omnicoder/tokenization/`: ledger/tokenizer utilities for typed 2026
  modality ranges.
- `src/omnicoder/inference/output_router_2026.py`: modality-aware output
  routing that keeps text, tool/action JSON, image, video, speech/audio, and
  music generations on the correct ledger lanes before any edge decoder runs.
- `src/omnicoder/data_factory/`: ingestion, curation, integrity scanning,
  trace export, teacher-job generation, and benchmark materialization.
- `src/omnicoder/training/`: staged harnesses, dense pretraining, SFT/QLoRA
  bridges, distillation, reward replay, and orchestration.
- `src/omnicoder/eval/`: sample-loss checks, benchmark-suite plumbing,
  reportable prediction validation, and release gates.
- `src/omnicoder/export/` and `src/omnicoder/inference/runtimes/`: ONNX,
  Core ML, ExecuTorch, GGUF, provider benches, llama.cpp/vLLM/MLC adapters,
  mobile paths, and runtime helpers.
- `profiles/`: device, provider, dataset, teacher, benchmark, and training
  configuration.
- `scripts/`: AI-server launchers and curation/training sidecar helpers.

Data and training metadata are JSONL-first. Where database mirroring is used,
the repo provides raw PostgreSQL schemas under `schemas/`; it does not require
an ORM for the documented 2026 path.

## Current Status

What is present in this repository:

- A packageable Python project named `omnicoder` with console entry points in
  `pyproject.toml`.
- Dense 2026 model scaffolding, token-ledger utilities, and a native-1M-context
  architecture contract.
- Training orchestration for curation, staged dense training, long-context
  checks, posttraining bridges, and sharded checkpoint workflows.
- Data-factory lanes for local traces, media manifests, teacher jobs,
  capability curation, dataset-integrity checks, and benchmark task
  materialization.
- Evaluation harnesses for smoke, sample-loss, pipeline checkpoint checks,
  reportable prediction validation, and release-gate bookkeeping.
- Export/runtime experiments for desktop, server, mobile, and bridge formats.
- Unit and smoke tests across modeling, data, training, export, runtime, and
  benchmark components.

What should be treated carefully:

- The production target is 20B-class and native-1M-context, but public weights
  and public benchmark-quality evidence are not included here.
- Many commands exercise contracts, probes, canaries, or local engineering
  regression paths. Passing them is useful evidence, not a leaderboard claim.
- Some legacy docs describe earlier sparse-MoE or ONNX-first assumptions. Use
  the 2026 docs listed below as the current source of truth.
- Real training runs require appropriate datasets, teacher endpoints, storage,
  GPU placement, and artifact mounts.

Recent local evidence is stored in machine-readable artifacts such as
`pytest_exit_code.txt`, `tests_logs/`, `weights/**/manifests/`,
`weights/**/results*.jsonl`, and run registry outputs when generated. Keep
those artifacts attached to any claim about a specific run.

## Install

Use a virtual environment. The base install is enough for CPU smoke checks;
optional extras are available for ONNX, vision, audio, eval, generation, and
Linux-only Unsloth experiments.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
copy env.example.txt .env
```

macOS/Linux:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
cp env.example.txt .env
```

Optional extras:

```bash
python -m pip install -e ".[onnx,vision,audio,eval,gen]"
```

Do not commit private paths, API tokens, model credentials, or service
passwords. Use environment variables or local vault tooling for secrets.

## Quick Run

CPU package/import smoke:

```bash
python -m omnicoder.inference.generate --prompt "Hello OmniCoder" --device cpu
```

One-button development harness:

```bash
python -m omnicoder.tools.press_play --device cpu --out_root weights
```

The development harness exercises local wiring and writes artifacts under
`weights/`. It is not a full training recipe.

## Data And Curation

Validate the training orchestration profile:

```bash
training-orchestration-2026 --profile profiles/training_orchestration_2026.json validate
```

Run a dry curation/harness pass over trace data:

```powershell
full-harness-2026 run `
  --profile profiles/training_harness_2026.json `
  --trace-input data/raw/agent_memory_events_2026.jsonl `
  --stages ingest_trace,quality_score,contam_scan,export_sft,teacher_jobs `
  --dry-run
```

Run dataset integrity checks against current training inputs:

```bash
python -m omnicoder.data_factory.dataset_integrity_2026 --help
```

AI-server sidecar launchers for trace mining, dataset expansion, teacher jobs,
benchmark materialization, and coverage reports live in `scripts/`. Read
[`docs/TrainingOrchestration2026.md`](docs/TrainingOrchestration2026.md) before
using them against real GPUs or shared training volumes.

## Training Commands

Small legacy/dev training probe:

```bash
python -m omnicoder.tools.run_training --budget_hours 1 --device cuda
```

2026 staged harness:

```powershell
full-harness-2026 run `
  --profile profiles/training_harness_2026.json `
  --trace-input data/raw/agent_memory_events_2026.jsonl `
  --stages all
```

Focused dense 2026 pretraining entry point:

```bash
pretrain-2026-dense --help
```

Production-oriented orchestration entry point:

```bash
training-orchestration-2026 --profile profiles/training_orchestration_2026.json --help
```

On the AI-server lane, the documented launcher is:

```bash
scripts/ai_server_fast_pipeline_20b.sh
```

That launcher is intended for a specific multi-GPU Docker environment and
requires the mounts, checkpoint completeness rules, and artifact expectations
described in `docs/TrainingOrchestration2026.md`.

## Evaluation And Benchmarks

Validate the benchmark profile:

```bash
benchmark-suite-2026 --profile profiles/benchmark_suite_2026.json validate
```

Run a local smoke cycle:

```powershell
benchmark-suite-2026 `
  --profile profiles/benchmark_suite_2026.json `
  --model weights/harness_2026/smoke.pt `
  --out-dir weights/benchmarks_2026/smoke `
  run-smoke `
  --run-id smoke-local `
  --timeout-seconds 30
```

Summarize results:

```powershell
benchmark-suite-2026 `
  --profile profiles/benchmark_suite_2026.json `
  --out-dir weights/benchmarks_2026/smoke `
  summarize `
  --results results.jsonl
```

For sharded or pipeline checkpoints, use the sample-loss and checkpoint tools
documented in
[`docs/TrainingHarness2026.md`](docs/TrainingHarness2026.md) and
[`docs/TrainingOrchestration2026.md`](docs/TrainingOrchestration2026.md).
Reportable benchmark scoring requires authorized task metadata, immutable task
hashes, real model-generated prediction artifacts, and the matching scorer
references.

## Export And Runtime

Export a decode-step ONNX artifact:

```bash
python -m omnicoder.export.onnx_export --out weights/release/text/omnicoder_decode_step.onnx
```

Run a provider bench:

```bash
python -m omnicoder.inference.runtimes.provider_bench \
  --model weights/release/text/omnicoder_decode_step.onnx \
  --providers CPUExecutionProvider \
  --out_json weights/release/text/provider_bench.json
```

Package mobile-oriented artifacts:

```bash
python -m omnicoder.tools.export_to_phone --platform android --tps_threshold 15
python -m omnicoder.tools.export_to_phone --platform ios --tps_threshold 6
```

Mobile sample code lives under
`src/omnicoder/inference/serverless_mobile/`.

## Artifact Paths

Common output locations:

- `weights/harness_2026/<run_id>/`: staged harness data, logs, registry events,
  checkpoints, and smoke outputs.
- `weights/training_orchestration_2026/<run_id>/`: production-oriented
  orchestration outputs and checkpoints.
- `weights/curated_datasets_2026/runs/<run_id>/`: curated training JSONL and
  manifests.
- `weights/external_datasets_2026/runs/<run_id>/`: external dataset expansion
  outputs.
- `weights/data_factory/runs/<run_id>/`: teacher jobs, benchmark
  materialization, curation reports, and sidecar manifests.
- `weights/data_factory/teacher_rollouts/<run_id>/`: generated teacher rollout
  rows and metadata.
- `weights/benchmarks_2026/<run_id>/`: benchmark manifests, JSONL results, and
  summaries.
- `weights/release/`: export artifacts such as ONNX models, provider benches,
  packaged assets, and release metadata.
- `tests_logs/`: local test logs and duration records.

Most generated artifacts are intentionally excluded from source control. Preserve
the run-scoped manifests and hashes when moving or publishing evidence.

## Documentation Map

- [`docs/Omnicoder2026Redesign.md`](docs/Omnicoder2026Redesign.md): current
  model and ledger architecture.
- [`docs/TrainingOrchestration2026.md`](docs/TrainingOrchestration2026.md):
  end-to-end curation, training, sidecar, recovery, and benchmark contract.
- [`docs/TrainingHarness2026.md`](docs/TrainingHarness2026.md): staged harness
  commands and checkpoint notes.
- [`docs/DatasetCuration2026.md`](docs/DatasetCuration2026.md): curation,
  licensing, source policy, and integrity gates.
- [`docs/DistillationAndRL2026.md`](docs/DistillationAndRL2026.md): teacher,
  reward, preference, and posttraining lanes.
- [`docs/BenchmarkSuite2026.md`](docs/BenchmarkSuite2026.md): benchmark-suite
  contracts and release-gate expectations.
- [`docs/AgenticToolTraining2026.md`](docs/AgenticToolTraining2026.md):
  tool-use, trace, preference, reward, and safety-negative training artifacts.
- [`docs/QUICKSTART_CURRENT.md`](docs/QUICKSTART_CURRENT.md): shorter smoke-test
  quickstart.
- `docs/legacy/`: archived notes from earlier design directions.

## Limitations

- No public model card or checkpoint in this repo establishes production-grade
  benchmark quality.
- The local smoke tests and fixtures are mainly regression evidence for wiring,
  schemas, exports, and training/eval contracts.
- The 20B-class path depends on a specific multi-GPU training environment and
  is not expected to run on a normal laptop.
- Some modality modules are scaffolds or canaries awaiting larger training
  runs, real artifacts, or external runtimes.
- Optional dependencies vary by platform; audio, generation, and acceleration
  extras are not equally available on every OS.
- Official or reportable evaluation requires authorized snapshots and scorer
  metadata that are intentionally separate from local smoke fixtures.

## License

See [`LICENSE`](LICENSE).
