# Omnicoder 2026 Training Orchestration

This document defines the end-to-end orchestration contract for automated
curation, real multimodal training, q4 recovery, long-context validation, and
benchmark release gates. The production entry point is
`training-orchestration-2026`; it inventories real trace/media sources, exports
per-modality ledger-token JSONL, and runs resumable dense training stages with
loss gates for each required modality.

## Ownership And Guardrails

- Primary orchestration entry point: `full-harness-2026`.
- Lower-level staged runner: `train-2026`.
- Data factory entry points: `memory-traces-2026`, `trace-orchestrator-2026`,
  `ingest-2026`, `curate-2026`, and `teacher-jobs-2026`.
- Verification entry points: `eval-2026`, `metrics-2026`,
  `context-budget-2026`, and `benchmark-suite-2026`.
- Tool-learning entry point: `agentic-tool-train-2026`.
- Post-training handoff: `distill-curriculum-2026`,
  `posttrain-bridge-2026`, and `sft-qlora-2026`.
- Real production orchestration: `training-orchestration-2026 validate`,
  `training-orchestration-2026 inventory`,
  `training-orchestration-2026 curate-real`, and
  `training-orchestration-2026 run-real`.

The training path remains JSONL-first, with raw PostgreSQL mirroring only where
existing schemas and profiles explicitly support it. Protected eval data, hidden
tests, grader labels, answer keys, and successful benchmark trajectories must
remain quarantined from training exports.

The production architecture contract is `omnicoder2026_20b_1m`: a dense
20B-class target sized by the 24GB Q4 native-1M budget, not by an exact fixed
parameter count. The staged AI-server job defaults to `ledger_probe` because it
is the full-330k-token-ledger learning verifier used before scaling; it must not
be reported as the production-size model.

## Orchestration Ladder

1. Collect immutable 2025-2026 traces and media manifests.
2. Normalize records into data-factory JSONL with provenance and source dates.
3. Score quality, redact or reject secret-bearing rows, dedupe, and assign
   splits.
4. Run contamination scans against benchmark-protected registries.
5. Export SFT conversations, tool-training rows, preference rows, reward rows,
   teacher-job JSONL, and media manifests.
6. Run staged training and post-training bridges from accepted exports only.
7. Verify modality learning and recovery behavior before benchmark reporting.
8. Gate release candidates through smoke, nightly, and release benchmark cycles.

The canonical command for a dry-run orchestration check is:

```powershell
full-harness-2026 run `
  --profile profiles/training_harness_2026.json `
  --trace-input data/raw/agent_memory_events_2026.jsonl `
  --stages ingest_trace,ingest_media,quality_score,contam_scan,export_sft,agentic_tool_training,teacher_jobs `
  --dry-run
```

The canonical command for a real AI-server training job is:

```powershell
$env:CUDA_VISIBLE_DEVICES = "0,4,6"
training-orchestration-2026 `
  --profile profiles/training_orchestration_2026.json `
  --out-dir weights/training_orchestration_2026/real_run `
  run-real `
  --device cuda `
  --steps-per-stage 24 `
  --seq-len 192 `
  --batch-size 1 `
  --fake-quant
```

When running this job through Docker on the AI server, mount both the repo and
the host media tree. The curated media rows intentionally keep absolute
`/home/cereal/...` artifact refs so hashes and provenance remain stable; if the
container only mounts `/workspace`, image/video/audio/music artifacts are not
visible and the required media stages will fail with zero records.

```bash
docker run --rm \
  --gpus '"device=0,4,6"' \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /home/cereal/omnicoder_2026_work:/workspace \
  -v /home/cereal:/home/cereal:ro \
  -e PYTHONPATH=/workspace/src \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_SHM_DISABLE=0 \
  -w /workspace \
  omnicoder:cuda-posttrain-2026 \
  /opt/conda/bin/python -m omnicoder.training.training_orchestration_2026 \
    --profile profiles/training_orchestration_2026.json \
    --out-dir weights/training_orchestration_2026/real_run \
    run-real --device cuda --fake-quant
```

For a full resumable training lane, use `run-full`. It performs real curation,
per-modality dense pretraining, checkpointed distillation-job replay, live
native reward/preference/RLVR replay through `reward_replay_2026`, final
all-modality finetuning, and checkpoint benchmark gates. Detached Docker runs
should omit `--rm` so logs and exit state remain available if the run stops.

```bash
docker run -d \
  --name omnicoder_fulltrain_$(date -u +%Y%m%dT%H%M%SZ) \
  --gpus '"device=0,4,6"' \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /home/cereal/omnicoder_2026_work:/workspace \
  -v /home/cereal:/home/cereal:ro \
  -e PYTHONPATH=/workspace/src \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_SHM_DISABLE=0 \
  -w /workspace \
  omnicoder:cuda-posttrain-2026 \
  /opt/conda/bin/python -m omnicoder.training.training_orchestration_2026 \
    --profile profiles/training_orchestration_2026.json \
    --out-dir weights/training_orchestration_2026/full_run \
    run-full \
    --preset omnicoder2026_20b_1m \
    --distributed pipeline_stage \
    --nproc-per-node 3 \
    --rank-device-map 0,1,2 \
    --placement-layer-counts 16,16,32 \
    --pipeline-stage-schedule gpipe \
    --pipeline-microbatches 1 \
    --precision fp16 \
    --init-dtype fp16 \
    --optimizer adafactor \
    --optimizer-in-backward \
    --optimizer-in-backward-update lowmem_adafactor \
    --activation-checkpointing \
    --fake-quant-chunk-rows 64 \
    --fake-quant-max-full-elements 16777216 \
    --steps-per-stage 64 \
    --posttrain-steps 32 \
    --finetune-steps 64 \
    --seq-len 1024 \
    --batch-size 1 \
    --save-interval 32 \
    --fake-quant
```

The repeatable AI-server launcher for this lane is
`scripts/ai_server_fast_pipeline_20b.sh`. It bakes in the fast-card device
selection, Docker IPC/ulimit requirements, 16/16/32 layer placement, GPipe
schedule, low-memory Adafactor, q4 fake-quant hooks, and media-tree mounts:

```bash
cd /home/cereal/omnicoder_2026_work
OMNICODER_RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)" \
OMNICODER_START_STAGE=image \
OMNICODER_STAGE_ORDER=image,video,audio,music,long_context \
OMNICODER_RESUME_CHECKPOINT=weights/training_orchestration_2026/target20b_pipeline_tool_20260524T004357Z/checkpoints/03_tool.pt \
scripts/ai_server_fast_pipeline_20b.sh
```

## Dataset And Teacher Sidecars

Do not start a second synchronous 20B target run while a target container owns
fast GPUs `0,4,6`. Use `scripts/ai_server_dataset_training_sidecars_2026.sh`
for additive work:

```bash
cd /home/cereal/omnicoder_2026_work

# Read-only state check.
scripts/ai_server_dataset_training_sidecars_2026.sh preflight

# Trace mining, ComfyUI artifact indexing, external dataset expansion,
# teacher-job sharding, and P40 teacher rollouts.
OMNICODER_MAX_RECORDS_PER_DATASET=1024 \
OMNICODER_TEACHER_LIMIT=256 \
scripts/ai_server_dataset_training_sidecars_2026.sh all

# Later status check.
scripts/ai_server_dataset_training_sidecars_2026.sh status
```

The sidecar runner writes run-scoped outputs under:

- `weights/curated_datasets_2026/runs/<run_id>`
- `weights/external_datasets_2026/runs/<run_id>`
- `weights/data_factory/trace_orchestrator_2026/teacher_jobs/<run_id>`
- `weights/data_factory/teacher_rollouts/<run_id>`

It promotes `latest` symlinks only after outputs exist. The target training
profile reads external expansion JSONL family files when present, but
`eval_holdout` and `blocked_until_review` files stay out of train paths.

P40 usage policy:

- GPUs `1,2,3`: warm Qwen3.6 27B Q4 OpenAI-compatible teacher rollouts.
- GPU `5`: optional probe/short validation if it is idle and cool.
- CPU: trace export, redaction, dedupe, license-tier manifests, JSONL slicing.
- Fast GPUs `0,4,6`: the active 20B pipeline only.

This keeps the RTX 3090s and RTX 8000 saturated by the target model while P40s
produce agentic, math, code, and tool distillation rows in parallel.

Monitor a detached lane with `scripts/ai_server_monitor_fast_pipeline_2026.sh`.
For older hand-launched containers, set `OMNICODER_NAME_FILTER` to the actual
container prefix.

The AI-server target lane is intentionally disjoint from the P40 sidecar lane.
The target container exposes host GPUs `0,4,6` only, which become container
devices `0,1,2`: RTX 3090, RTX 3090, RTX 8000. The production sharded path uses
`torch.distributed.pipelining.PipelineStage`; rank 0 owns 16 layers, rank 1 owns
16 layers, and rank 2 owns 32 layers plus the final norm/output head. This uses
the RTX 8000 headroom first and keeps P40s out of the synchronous target path.
Launch target pipeline containers with `--ipc=host` or an equivalently large
Docker shared-memory configuration. NCCL uses shared-memory segments during
rank placement checks and early collectives; the default small Docker IPC
namespace can fail before the first training step with `Error while creating
shared memory segment /dev/shm/nccl-*`.

The fit-first schedule is `--pipeline-stage-schedule gpipe
--pipeline-microbatches 1`, which enables per-parameter post-accumulate
low-memory Adafactor updates after each parameter receives its single
microbatch gradient. `Schedule1F1B` is validated only for `pipeline_microbatches
>= 2`; this PyTorch/NCCL build can produce a message-size mismatch with 1F1B and
exactly one microbatch, so the trainer auto-routes that combination to GPipe.
The higher-throughput path is `--pipeline-stage-schedule 1f1b
--pipeline-microbatches 2`, but enable it only after the target fast-card memory
probe confirms the delayed low-memory update fits.

Sharded pipeline checkpoints are directories containing `rank00000.pt`,
`rank00001.pt`, `rank00002.pt`, a `manifest.json`, and a directory-local
`.complete.json`. The manifest must describe the actual pipeline world size and
the rank files must be contiguous from `rank00000.pt` through the final rank;
partial rank directories are treated as incomplete even if a stale completion
marker exists. Native single-file checkpoints still use the sibling
`*.complete.json` marker.

Pipeline checkpoints are evaluated with
`omnicoder.eval.pipeline_sample_loss_2026` under the same three-rank
`torch.distributed.run` layout. The stage gate is real heldout sample loss, not
a smoke check: by default it evaluates up to 32 records per eval/test JSONL file
at the active stage sequence length. Configure this with
`training_plan.heldout_sample_loss_max_records_per_file` or the
`--heldout-max-records-per-file` CLI override. The gate also has a hard timeout
(`training_plan.heldout_sample_loss_timeout_seconds`, default 3600) so failed
ranks cannot wait forever in distributed send/recv. Pipeline eval prints rank-0
record/chunk progress for new runs; older already-running containers may not
show this progress until relaunched.

Live posttraining on pipeline checkpoints also stays distributed. When
`--live-posttraining` receives a sharded checkpoint directory, the orchestrator
uses `omnicoder.training.pipeline_pretrain_2026_dense` for reward-replay/SFT
continuations and writes the next sharded checkpoint under
`checkpoints/posttrain/*_pipeline`; it does not silently downgrade to native
single-file reward replay.

Use the weighted-placement validator only when exercising the older
single-process placement scheduler:

```bash
docker run --rm --gpus '"device=0,4,6"' \
  -v /home/cereal/omnicoder_2026_work:/workspace \
  -w /workspace \
  -e PYTHONPATH=/workspace/src \
  omnicoder:cuda-posttrain-2026 \
  python -m omnicoder.tools.validate_weighted_pipeline_2026 \
    --devices cuda:0,cuda:1
```

Restart controls are explicit. Use `--resume-completed-stages` to skip stages
with complete checkpoints, `--rerun-completed-stages` to force retraining, and
`--start-stage code` or another stage name only when a prior checkpoint or
`--resume-checkpoint` is present. Single-file native checkpoints receive a
sibling `*.complete.json` marker; sharded pipeline checkpoints receive
`.complete.json` inside the checkpoint directory so sidecar eval jobs do not
read half-written files.

Detached sidecar jobs use P40s `1,2,3,5`. Sidecar `training_run` jobs are
restricted to probe/ledger presets, single-device placement, and sidecar output
roots; target-contract 20B training must stay on the fast-card lane. Validate
and plan the configured sidecar jobs before launching:

```bash
cd /home/cereal/omnicoder_2026_work
PYTHONPATH=/home/cereal/omnicoder_2026_work/src \
python3 -m omnicoder.tools.gpu_sidecar_2026 \
  --profile profiles/training_orchestration_2026.json validate

PYTHONPATH=/home/cereal/omnicoder_2026_work/src \
python3 -m omnicoder.tools.gpu_sidecar_2026 \
  --profile profiles/training_orchestration_2026.json plan
```

Launch all detached P40 jobs, or target one explicit sidecar job:

```bash
PYTHONPATH=/home/cereal/omnicoder_2026_work/src \
python3 -m omnicoder.tools.gpu_sidecar_2026 \
  --profile profiles/training_orchestration_2026.json launch

PYTHONPATH=/home/cereal/omnicoder_2026_work/src \
python3 -m omnicoder.tools.gpu_sidecar_2026 \
  --profile profiles/training_orchestration_2026.json \
  --job p40_5_probe_training_run launch
```

Each sidecar process receives its own `CUDA_VISIBLE_DEVICES` value from the
profile, writes under `weights/training_orchestration_2026/gpu_sidecar`, and is
rejected by the launcher if its pinned device overlaps the main GPU set.

When P40s already host LM Studio/llama.cpp teacher servers, reuse those
endpoints as HTTP sidecars instead of loading another CUDA model on top of them.
For example, split Qwen3.6 teacher jobs across `127.0.0.1:18082`, `18084`, and
`18085` with `CUDA_VISIBLE_DEVICES=""` in the client process; the client uses no
CUDA memory, while the resident P40 servers perform the generation. GPU5 can
run probe-scale `ledger_probe` training or verifier jobs independently, but
target 20B checkpoints must never resume from `gpu_sidecar/*` artifacts.

Before starting GPU training, run `curate-real` with the same mounts and confirm
that `image`, `video`, `audio`, and `music` have nonzero train records in
`manifests/curation_manifest.json`.

This creates `train_text.jsonl`, `train_code.jsonl`, `train_tool.jsonl`,
`train_image.jsonl`, `train_video.jsonl`, `train_audio.jsonl`,
`train_music.jsonl`, and `train_long_context.jsonl` from real configured
sources. Each stage resumes from the prior checkpoint and fails if the required
loss trend does not improve. For the 20B target, use the same curated JSONL and
stage gates, then scale with sharded QAT/LoRA-to-full recovery and q4/TurboQuant
deployment validation.

## Automated Curation

Automated curation is handled by the data factory documented in
`docs/DatasetCuration2026.md`. The accepted path is:

- `memory-traces-2026`: collect Codex, Claude Code, Hermes, and agent-memory
  hook exports.
- `trace-orchestrator-2026`: normalize, curate, score, contam-scan, export SFT,
  and create teacher jobs.
- `curate-2026`: run targeted export, scoring, dedupe, and contamination
  operations against JSONL datasets.
- `teacher-jobs-2026`: produce local teacher work items for trace critique,
  tool repair, media captioning, verifier labels, and preference labels.

Every curation output must preserve enough metadata to answer: where did this
row come from, when was the source created, what modality is represented, what
quality gates ran, what contamination checks ran, and whether it is allowed in
training. Rows that contain credentials, protected benchmark material, private
grader outputs, or leaked eval trajectories are rejected or quarantined.

## Training Stages

`training-orchestration-2026` is the production coordinator for real multimodal
learning gates. It exports per-modality token-ledger records from configured
trace files, COCO/CC image-caption JSONL, LibriSpeech/LJSpeech audio, ComfyUI
image/video/audio/music outputs, code corpora, and long-context trace spans.

`full-harness-2026` remains the registry-backed run coordinator for trace SFT,
tool-training exports, teacher jobs, context budgeting, and benchmark wiring.
Its default stages are:

- `ingest_trace`: load normalized agent-memory and harness traces.
- `ingest_media`: load approved image, video, audio, and music manifests.
- `quality_score`: apply heuristic quality and safety gates.
- `contam_scan`: compare candidates with benchmark-protected registries.
- `export_sft`: write grouped chat/tool SFT JSONL.
- `agentic_tool_training`: build tool SFT, preference, reward, RLVR, and safety
  negative artifacts.
- `teacher_jobs`: create critique, repair, caption, verifier, and reward-label
  jobs for local teachers.
- `sft_qlora_bridge`: optional TRL/PEFT bridge and dependency check.
- `native_train`: run `pretrain_2026_dense` on accepted exports.
- `eval_smoke`: run the lightweight registry eval harness.
- `context_budget`: record native 1M and q4 deployment memory estimates.

`train-2026` remains the lower-level stage planner for ingest, ledger encoding,
pretraining, SFT, teacher distillation, preference, long-context, RLVR, QAT,
eval, and GGUF bridge planning. Use it for focused probes; use
`full-harness-2026` when run manifests, registry events, artifact references,
and gate evidence are required.

The dense trainer now accepts either normal text records or explicit
`token_ids`. The real orchestration path uses `token_ids` so image, video,
speech/audio, music, tool, and long-context artifacts enter the same shared
embedding/output space through the Omnicoder 2026 ledger rather than through
separate learned in-trunk adapters.

## Multimodal Learning Verification

Multimodal verification must prove that the model is learning across modality
lanes instead of only preserving text traces. The release evidence should cover:

- Text: SFT loss, trace grouping quality, contamination-clear status, and
  instruction-following smoke evals.
- Code: repo-task fixtures, compile/test pass rates, patch application, and
  regression checks.
- Tools: tool selection, argument validity, observation grounding, stop
  behavior, repair traces, and safety negatives from `agentic-tool-train-2026`.
- Image: prompt adherence, edit preservation, artifact integrity, CLIP-style or
  verifier labels, and private-prompt quarantine.
- Video: temporal consistency, motion checks, black-frame/corruption rejection,
  prompt adherence, and audio-visual grounding where audio is present.
- Audio and music: ASR or caption alignment, clipping/loudness checks, structure
  labels, lyric alignment, and music similarity-risk screening.
- Long context: 32K, 128K, and 1M retrieval/reasoning probes with peak memory,
  position robustness, and context-budget manifests.

Verification artifacts should be JSONL or JSON with hashes for source records,
model checkpoints, generated media, benchmark inputs, and outputs. Generated
media must be decoded and inspected by the relevant adapter before a benchmark
result becomes reportable.

## Q4 Recovery And Deployment Gates

Q4 recovery is a training and evaluation lane, not a claim made after export.
Use the existing q4-aware stages documented in `docs/DistillationAndRL2026.md`,
`docs/AgenticToolTraining2026.md`, and `src/omnicoder/training/orchestrator_2026.py`.

Required evidence:

- q4-aware recovery distillation or QAT/fake-quant stage is represented in the
  run manifest.
- Context-budget output records q4 weights, q4 KV assumptions, and native 1M
  memory estimates through `context-budget-2026`.
- Tool and long-context behaviors are rechecked under q4 or q4-simulated
  deployment conditions.
- GGUF bridge output is labeled as a compatibility bridge and not treated as the
  full native 1M runtime.
- Metrics compare pre-recovery and post-recovery behavior for text, code,
  tools, multimodal generation, and long-context recall.

## Benchmark Gates

`benchmark-suite-2026` is the release-facing benchmark lane documented in
`docs/BenchmarkSuite2026.md`. It covers coding, agent/tool use, reasoning, long
context, multimodal understanding, image generation, video generation, audio
generation, music generation, safety/tool security, and deployment performance.

Use three run cycles:

- Smoke: contract-backed, small fixtures, no heavy downloads, validates adapter
  manifests and quarantine rules.
- Nightly: broader rolling slices, current 2025-2026 tasks, cached datasets, and
  drift tracking.
- Release: pinned profiles, immutable artifacts, private/authorized evals, and
  fail-closed gates.

Minimum commands:

```powershell
benchmark-suite-2026 --profile profiles/benchmark_suite_2026.json validate

benchmark-suite-2026 `
  --profile profiles/benchmark_suite_2026.json `
  --model weights/harness_2026/smoke.pt `
  --out-dir weights/benchmarks_2026/smoke `
  run-smoke `
  --run-id smoke-local `
  --timeout-seconds 30

benchmark-suite-2026 `
  --profile profiles/benchmark_suite_2026.json `
  --out-dir weights/benchmarks_2026/smoke `
  summarize `
  --results results.jsonl
```

Release gates fail closed on missing adapter results, hidden-material exposure,
credential leakage, missing artifact hashes, schema failures, hidden-test or
grader errors, incomplete manifests, or regressions beyond the profile
threshold.

## Registry Evidence

Each orchestration run should leave these machine-readable records:

- Run manifest and stage events from `run-registry-2026`.
- Stage logs summarized by `metrics-2026`.
- Data-factory curation manifest and dataset card.
- SFT, tool, preference, reward, RLVR, teacher-job, and safety-negative JSONL
  exports.
- Native checkpoint and training provenance when `native_train` runs.
- Context-budget JSON for native and q4 deployment estimates.
- Benchmark manifests, results, and summary JSON.

Inspect a run with:

```powershell
run-registry-2026 status --run-id <run_id>
metrics-2026 --log weights/harness_2026/<run_id>/logs/native_train.log
```

## Validation Checklist

- `docs/TrainingOrchestration2026.md` exists and documents only existing 2026
  entry points.
- `pyproject.toml` already exposes the needed console scripts; no new hook was
  added.
- The curation path includes text, code, tool, image, video, audio/music, long
  context, q4 recovery, and benchmark quarantine.
- The verification path includes multimodal learning evidence and artifact
  hashes.
- Benchmark gates use `benchmark-suite-2026` smoke, nightly, and release cycles.
