# Omnicoder 2026 Benchmark Suite

The 2026 benchmark suite is the release-facing evaluation lane for Omnicoder's
dense native-1M omnimodal agent model. It complements `eval-2026`, which is the
lightweight registry harness, by reserving a stable console entry point for full
benchmark orchestration:

```powershell
benchmark-suite-2026 --help
benchmark-suite-2026 validate
benchmark-suite-2026 list
benchmark-suite-2026 --out-dir weights/benchmarks_2026/smoke run-smoke
benchmark-suite-2026 --out-dir weights/benchmarks_2026/smoke summarize
```

The packaging hook points to:

```text
omnicoder.eval.benchmark_suite_2026:main
```

The default profile is `profiles/benchmark_suite_2026.json`.

## Scope

The suite should cover current 2025-2026 agent, code, tool-use, multimodal, and
generation benchmarks through adapters rather than hardcoding benchmark logic in
the runner. The registry path should remain JSON-first for workstation and AI
server portability, with optional raw PostgreSQL mirroring only when the database
is already provisioned.

Expected adapter families:

- Coding and long-horizon repo repair tasks.
- Terminal/container tasks with hidden tests.
- Tool-use and MCP stateful task suites.
- Simulated user and policy-constrained workflow tasks.
- Multimodal question answering and media understanding.
- Image, video, audio, music, and speech generation/edit preference evals.
- Native-1M retrieval, context-budget, and memory-contamination checks.

Protected prompts, private labels, grader artifacts, hidden tests, solution
traces, and successful evaluation trajectories must stay quarantined from data
factory exports and training traces.

## Current Coverage

The default suite profile is the broad contract harness, while
`profiles/benchmark_registry_2026.json` is the compact release-gate registry
used for high-signal pass/fail planning. Both follow the same quarantine and
reportability rules.

The broad profile covers the active benchmark axes:

- Coding: SWE-bench Pro, SWE-bench Live, LiveCodeBench, private repo maintenance.
- Agent/tool: BFCL v4, tau3/tau2-style stateful tasks, MCPMark-style workflows,
  and Terminal-Bench 2.1-style terminal tasks.
- Reasoning: ARC-AGI-3-style interactive environments and private expert
  reasoning/math refreshes.
- Long context: RULER/InfiniteBench-style retrieval plus LongBench-v2-style
  32K to 1M-plus context reasoning.
- Multimodal understanding: MMMU-Pro, Video-MME-v2, LVBench, LVOmniBench,
  JointAVBench, and AudioBench.
- Generation: ImgEdit-style image editing, VBench/VBench-2.0 video generation,
  audio/speech generation, and Music Arena-style music preference evaluation.
- Safety/deployment: tool-injection, credential, destructive-action, GGUF,
  1M-context, q4 memory, latency, throughput, and artifact-integrity gates.

Smoke mode is contract-backed. It validates that every benchmark, artifact
field, release gate, contamination flag, and output path is wired. It does not
claim benchmark scores until a real adapter command or full benchmark adapter
has run.

Reportable mode is stricter than smoke. A task row must come from an official or
explicitly authorized snapshot descriptor in `reportable_snapshots`, or carry
task-level `snapshot_id`, `snapshot_authorization`, source, and revision
metadata itself. Oracle-only rows are local-only: ARC/SWE/MMMU style task files
must include a separate model output, patch, action trajectory, tool call, or
artifact prediction, or the runner marks the score `local_only` even when the
gold fields are present. The training checkpoint gate writes
`checkpoint_predictions.jsonl` from explicit model/eval-adapter outputs and
passes it to `run-reportable`; missing authorized snapshots still fail closed
under the default `missing_reportable_policy`.

### Benchmark Materialization

Use `benchmark-materialize-2026` to fetch or scan real benchmark suites before
scoring. It writes run-scoped artifacts instead of mutating the canonical
profile:

```powershell
python -m omnicoder.data_factory.benchmark_materializer_2026 `
  --profile profiles/benchmark_suite_2026.json `
  --run-id <run_id> `
  --suite core25 `
  --mode public-dev `
  --download `
  materialize
```

Outputs land under:

```text
weights/data_factory/runs/benchmark_materialization/<run_id>/
  local_2026/*.jsonl
  reportable_2026/*.jsonl
  manifests/benchmark_materialization_manifest.json
```

Public/dev rows from suites such as Terminal-Bench, BFCL, LiveBench,
LiveCodeBench, MMMU-Pro, Video-MME-v2, AudioBench, VBench, and Music Arena are
usable for local regression, training realignment, and failure analysis. They
remain `local_only` until an official or explicitly authorized snapshot is
provided with `snapshot_id`, revision, source, and authorization metadata. The
AI-server sidecar action is opt-in:

```bash
OMNICODER_MATERIALIZE_BENCHMARK_TASKS=1 \
OMNICODER_BENCHMARK_MATERIALIZATION_SUITE=core25 \
OMNICODER_BENCHMARK_MATERIALIZATION_MODE=public-dev \
scripts/ai_server_dataset_training_sidecars_2026.sh benchmark-materialize
```

To evaluate a checkpoint against a run-scoped authorized root, pass it to the
20B launcher:

```bash
OMNICODER_REPORTABLE_TASK_ROOTS=weights/data_factory/runs/benchmark_materialization/<run_id>/reportable_2026 \
OMNICODER_REQUIRE_REPORTABLE_GATE=1 \
scripts/ai_server_fast_pipeline_20b.sh
```

The 20B launcher also accepts the materialization root directly. It will
automatically add `<root>/reportable_2026` when JSONL files exist there; set
`OMNICODER_ALLOW_LOCAL_BENCHMARK_TASK_ROOTS=1` only for public-dev regression
passes that must consume `<root>/local_2026`. Local public-dev rows are useful
for training realignment and nightly debugging, but they are still not
reportable release scores.

### Reportable Prediction Harness

Use `omnicoder.eval.reportable_prediction_harness_2026` to generate the
prediction JSONL consumed by `run-reportable`. The harness validates authorized
task metadata before generation, strips oracle/gold fields from model requests,
and writes rows keyed by `benchmark_id` and `task_id` with one model-output
field such as `prediction`, `model_patch`, `model_actions`, `tool_call`, or
`artifact_path`.

No-network fixture mode is for tests and contract checks only:

```powershell
python -m omnicoder.eval.reportable_prediction_harness_2026 `
  --backend fixture `
  --model fixture-local `
  --tasks data/eval/reportable_2026/mmmu_pro_authorized.jsonl `
  --out weights/benchmarks_2026/predictions/mmmu_fixture_predictions.jsonl
```

Local OpenAI-compatible serving:

```powershell
python -m omnicoder.eval.reportable_prediction_harness_2026 `
  --backend openai-compatible `
  --base-url http://127.0.0.1:8000/v1 `
  --model omnicoder2026-local `
  --tasks data/eval/reportable_2026/mmmu_pro_authorized.jsonl `
  --out weights/benchmarks_2026/predictions/mmmu_predictions.jsonl `
  --force
```

Local checkpoint runner mode invokes a command once per task. The command reads
one sanitized JSON request on stdin and writes either plain text or JSON such as
`{"prediction":"C"}` / `{"model_patch":"diff ..."}` to stdout:

```powershell
python -m omnicoder.eval.reportable_prediction_harness_2026 `
  --backend checkpoint-runner `
  --checkpoint-runner "python tools/local_checkpoint_predict.py" `
  --checkpoint-path weights/release/checkpoint `
  --model weights/release/checkpoint `
  --tasks data/eval/reportable_2026/swe_bench_live_authorized.jsonl `
  --out weights/benchmarks_2026/predictions/swe_live_predictions.jsonl
```

Then score the generated predictions:

```powershell
benchmark-suite-2026 `
  --profile profiles/benchmark_suite_2026.json `
  --model weights/release/checkpoint `
  --out-dir weights/benchmarks_2026/reportable/<run_id> `
  run-reportable `
  --tasks data/eval/reportable_2026 `
  --predictions weights/benchmarks_2026/predictions/swe_live_predictions.jsonl `
  --cycle release
```

The release registry now includes fresh RLVR and media-preference gates for
RLVR Linearity, Nous RLVR Coding, EditReward-Bench, IESBench, Stable Video
Infinity, and text-to-audio human preference evaluation. The latest expansion
also adds ARC-AGI-3 interactive tasks, Terminal-Bench 2.1, BrowserGym/WebArena,
OSWorld, LiveBench math, split MMMU-Pro standard/vision gates, Video-MME-v2,
AudioBench/MMAU/MMAU-Pro, MMAR, IFEval-Audio through VoiceBench, VBench 2.0
faithfulness, VBench trustworthiness, Music Arena, BFCL v4, SWE-Lancer, HLE,
GDPVal, PaperBench, TheAgentCompany, MCP-Bench, and MCPMark/MCP-Universe
contracts. These are eval/release gates only;
their hidden labels, media assets, private states, answer keys, and successful
trajectories do not enter training exports. The broad suite profile includes
matching contract adapters under reasoning, agent/tool, multimodal,
generation, and safety gates so nightly/release planning sees the same
coverage.

## Workflows

### Smoke

Use smoke runs for pull-request checks, local packaging checks, and quick AI
server sanity checks. Smoke should use tiny fixtures and registry validation,
then write JSONL manifests, JSONL results, and `summary.json`.

```powershell
benchmark-suite-2026 `
  --profile profiles/benchmark_suite_2026.json `
  --model weights/harness_2026/smoke.pt `
  --out-dir weights/benchmarks_2026/smoke `
  run-smoke `
  --run-id smoke-local `
  --timeout-seconds 30
```

Minimum smoke gates:

- Load the benchmark registry/profile.
- Confirm every selected adapter declares its protected fields.
- Run at least one deterministic toy or fixture-backed task.
- Emit score, failure taxonomy, hashes, provenance, and quarantine metadata.
- Exit nonzero if protected benchmark material is routed toward training paths.

### Nightly

Use nightly runs for broader adapter coverage, regression tracking, and
performance drift. Nightly should prefer cached datasets and AI-server local
models to avoid unstable external dependencies.

```powershell
benchmark-suite-2026 `
  --profile profiles/benchmark_suite_2026.json `
  --model weights/release `
  --out-dir weights/benchmarks_2026/nightly/%DATE% `
  plan `
  --cycle nightly `
  --mode dry-run
```

Nightly outputs should include:

- Per-adapter JSONL results.
- Aggregate scorecard JSON.
- Latency, token, memory, and cost fields where measurable.
- Failure taxonomy counts.
- Dataset and model hashes.
- Machine-readable release-gate status.

### Release

Use release runs before tagging model, runtime, or dataset releases. Release
runs should use pinned benchmark versions, frozen profiles, and immutable
artifact directories.

```powershell
benchmark-suite-2026 `
  --profile profiles/benchmark_suite_2026.json `
  --model weights/release `
  --out-dir weights/benchmarks_2026/release/<run_id> `
  plan `
  --cycle release `
  --mode command
```

Release gates should fail closed on:

- Missing adapter results.
- Missing model or dataset hashes.
- Any contamination finding against protected benchmark fields.
- Hidden-test or grader execution errors.
- Score regressions beyond the profile threshold.
- Incomplete artifact manifests.

## AI Server Paths

Reference AI-server host: `192.168.50.222`.

Recommended checkout path:

```text
/opt/omnicoder/omnicoder_0.1.0
```

Recommended artifact paths:

```text
/mnt/ai/omnicoder/benchmarks_2026/smoke
/mnt/ai/omnicoder/benchmarks_2026/nightly
/mnt/ai/omnicoder/benchmarks_2026/release
/mnt/ai/omnicoder/models/release
/mnt/ai/omnicoder/datasets/eval_protected
```

Recommended local workstation mirror paths:

```text
G:\omnicoder_0.1.0\weights\benchmarks_2026\smoke
G:\omnicoder_0.1.0\weights\benchmarks_2026\nightly
G:\omnicoder_0.1.0\weights\benchmarks_2026\release
G:\omnicoder_0.1.0\weights\harness_2026\smoke-data
```

Use the server-manager v2 SSH workflow for remote execution and file transfer.
Do not put credentials, benchmark-private data, or grader secrets in committed
profiles.

## Runner Contract

The runner exposes `main()` and supports these commands:

```text
validate
list [--benchmark <id>]
plan --cycle smoke|nightly|release --mode smoke|dry-run|command
run-smoke --cycle smoke|nightly|release --mode smoke|dry-run|command
summarize --results <jsonl>
```

The runner writes:

- `manifests.jsonl`: one benchmark manifest row per selected benchmark.
- `results.jsonl`: one task-result row per selected benchmark.
- `summary.json`: run status, result counts, and release-gate state.
- Optional command stdout/stderr tails for configured real adapters.

This keeps the console hook stable while allowing the adapter set and scoring
contracts to evolve with 2026 benchmark releases.
