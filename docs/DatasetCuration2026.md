# Omnicoder 2026 Dataset Curation

This is the trace and dataset factory for Codex, Claude Code, Hermes, ComfyUI,
agent-memory, and future omnimodal training data. It is JSONL-first for
portability and can mirror curated metadata into raw PostgreSQL. It does not
use ORM, Pydantic, SQLAlchemy, SQLite, or backup databases.

## Design

The pipeline follows the 2025-2026 pattern used by NeMo Curator, Nemotron-CC,
FineWeb-style data factories, and current agent trace work:

1. Collect immutable raw traces and media manifests.
2. Normalize each harness into data-factory records with provenance.
3. Build a canonical curated record with redaction, quality, dedupe,
   contamination, modality, code, tool, and split metadata.
4. Export accepted curated rows into training JSONL.
5. Score quality and scan protected eval/benchmark contamination.
6. Group traces into multi-turn SFT conversations by trace/session id.
7. Generate teacher-job JSONL for local models and multimodal teachers.
8. Emit manifests and a dataset card for every release.

## Core Commands

```powershell
python -m omnicoder.data_factory.memory_trace_collectors_2026 collect-codex `
  --out data/raw/codex_traces_2026/codex.jsonl --limit 50000

python -m omnicoder.data_factory.memory_trace_collectors_2026 collect-claude `
  --out data/raw/claude_traces_2026/claude.jsonl --limit 50000

python -m omnicoder.data_factory.trace_orchestrator_2026 `
  --profile profiles/dataset_curation_2026.json
```

The same entry points are available as console scripts after install:

```powershell
memory-traces-2026 collect-codex --out data/raw/codex_traces_2026/codex.jsonl
trace-orchestrator-2026 --profile profiles/dataset_curation_2026.json
curate-2026 export-training --input curated.jsonl --out training.jsonl
```

## Outputs

- `jsonl/normalized_traces.jsonl`: harness-normalized trace rows.
- `jsonl/curated_traces.jsonl.canonical.jsonl`: full canonical curation rows.
- `jsonl/curated_traces.jsonl`: accepted data-factory training rows.
- `jsonl/rejected_traces.jsonl`: rejected canonical rows with reasons.
- `jsonl/quality_scored.jsonl`: heuristic quality-scored rows.
- `jsonl/contamination_scanned.jsonl`: contamination labels.
- `exports/sft_traces.jsonl`: grouped multi-turn conversations.
- `teacher_jobs/teacher_jobs_2026.jsonl`: teacher critique/distillation jobs.
- `manifests/trace_orchestrator_manifest.json`: release manifest.
- `dataset_card.md`: human-readable release summary.

## External Dataset Expansion

`profiles/dataset_curation_2026.json` now includes
`external_dataset_registry_2026`, a license-tiered 2025-2026 dataset registry.
The registry separates:

- `train`: permissive/attribution sources allowed into training after
  decontamination.
- `research_internal`: useful for local distillation, reward labeling, or
  internal experiments, but not automatically publishable.
- `eval_holdout`: benchmark or official eval material that must not be trained.
- `blocked_until_review`: rows that need a human license/split decision.

The expansion runner materializes rows into the normal ledger-token training
schema and writes family, modality, license, and policy manifests:

```powershell
dataset-expansion-2026 `
  --profile profiles/dataset_curation_2026.json `
  --out-dir weights/external_datasets_2026/latest `
  --download `
  --max-records-per-dataset 1024 `
  build
```

Current high-value registry families:

- Math/reasoning: OpenR1-Math-220k, DAPO-Math-17k-Processed,
  DeepScaleR-Preview, NVIDIA OpenMathReasoning, MathNet,
  NVIDIA AceReason-Math, and OpenThoughts3.
- Coding/SWE/terminal/tool: NVIDIA OpenCodeReasoning-2, SWE-smith, SWE-Gym,
  OpenHands SFT trajectories, Toucan-1.5M, ToolOmni-Data, and
  Terminal-Bench heldout metadata.
- Multimodal generation: OpenGPT-4o-Image, ShareGPT-4o-Image, VideoUFO,
  OpenVid-1M, Emilia-YODAS, AudioSkills, JamendoMaxCaps, and MusicBench.

Rows from external sources are not merged into the 20B target lane merely
because they exist. They must survive redaction, dedupe, benchmark
decontamination, license tiering, and heldout sample-loss checks.

## Quality And Safety

The curation layer stores scores instead of only dropping rows. It tracks
length, diversity, structure, language confidence, provenance, secret findings,
code/tool/media classifications, dedupe hashes, contamination labels, and split
assignment. Secret-bearing rows are redacted and rejected by default.

The SFT exporter groups eligible rows into conversations by trace id and skips
single-message/self-answer traces unless they contain an assistant turn. This
keeps trace training focused on real interactions rather than prompt-equals-
answer artifacts.

## Agentic And RLVR Exports

`agentic-tool-train-2026` now emits both compatibility exports and domain
exports:

- `tool_sft.jsonl`
- `tool_reward.jsonl`
- `tool_preference.jsonl`
- `tool_rlvr.jsonl`
- `math_rlvr.jsonl`
- `code_rlvr.jsonl`
- `terminal_rlvr.jsonl`
- `browser_rlvr.jsonl`
- `tool_safety_negatives.jsonl`

Domain rows carry verifier contracts and reward axes for exact math answers,
code tests, terminal state, browser citations, and tool schema/state validity.
This lets the training sampler weight math, coding, terminal, browser, and
tool RLVR separately instead of treating every agent trajectory as one generic
tool trace.

## PostgreSQL

Apply `schemas/curation_layers_2026.sql` to enable raw PostgreSQL mirroring.
The JSONL path remains the default because it works on the workstation and AI
server without database credentials.
