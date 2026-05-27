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

All optimizer launches now require a dataset-integrity preflight. The curation
policy and balanced posttraining builder reject rows flagged by
`dataset_integrity_2026`; `training_orchestration_2026` also rescans
train-bound JSONL immediately before dense training and posttraining. The queue
script performs the same check after building balanced SFT/RLVR/reward files
and before launching the next container. Positive SynthID/C2PA/Content
Credentials/provenance markers, data-mining restrictions, hidden Unicode prompt
payloads, prompt-injection strings, and poisoning/backdoor/degradation cues are
quarantined instead of silently dropped into training.

## Music, TTS, And ACE-Step Distillation Lane

The posttraining queue can now blend an additional focused music/audio curation
lane produced by `scripts/ai_server_run_music_tts_ace_curation_2026.sh`. This
lane normalizes embedded music/audio bytes into sidecar media files, curates
high-bandwidth text-to-music rows, HiFiTTS/speech rows, and live ACE-Step 1.5
teacher rollouts into `music.clean.jsonl`, `musicbench.clean.jsonl`,
`tts.clean.jsonl`, `ace_rollouts.clean.jsonl`, and
`music_tts_ace_clean.jsonl`.

The supplemental expansion entry point is
`scripts/ai_server_run_music_tts_expansion_2026.sh`. It layers LAION
Orpheus-style expressive TTS parquet rows with extracted WAV artifacts,
JamendoMaxCaps CC-BY/CC-BY-SA music rows with downloaded MP3 artifacts, and
additional ACE-Step 1.5 P40 teacher rollouts onto the latest completed base
music/TTS run. It writes the same required family files so the queue can consume
the expanded directory without changing the training launcher contract.

Audio artifact QA is handled by `scripts/ai_server_audio_manifest_qa_2026.sh`.
It is CPU-only and uses `ffprobe`/`ffmpeg` to gate decodeability, duration,
sample rate, channel count, silence ratio, and clipping risk for music/TTS media
without touching the active training GPUs. Keep the JSONL plus
`.summary.json` next to the run logs.

Live ACE rollouts are routed to the P40 ComfyUI sidecar by default
(`docker-compose.p40.yml`, port `27189`) so the active fast-card 20B optimizer
run is not squeezed by teacher inference. The queue watcher reads
`weights/data_curation_agent_2026/latest_music_tts_ace_curation_dir.txt`, waits
for that sidecar curation PID to finish, then blends the cleaned music/TTS/ACE
rows into the balanced all-modal SFT/RLVR/reward manifests before launching the
next chunk from the latest complete checkpoint. When a music/TTS sidecar is set,
the queue refuses launch unless `tts.clean.jsonl`, `music.clean.jsonl`,
`musicbench.clean.jsonl`, and `ace_rollouts.clean.jsonl` are all nonempty.

The production architecture contract is `omnicoder2026_20b_1m`: a dense
20B-class target sized by the 24GB Q4 native-1M budget, not by an exact fixed
parameter count. The staged AI-server job defaults to `ledger_probe` because it
is the full-330k-token-ledger learning verifier used before scaling; it must not
be reported as the production-size model.

For the fast-card 20B lane, host GPUs `0,4,6` are exposed to the container as
ranks `0,1,2`; P40s stay out of the synchronous pipeline stage. The pipeline
trainer now accepts `OMNICODER2026_DIST_TIMEOUT_SECONDS` for long startup and
first-collective phases, and telemetry records per-rank free/total VRAM,
device capability, visible-device mapping, and local rank. OpenAI-compatible
teacher rollout jobs are launched as HTTP clients with `CUDA_VISIBLE_DEVICES=""`
so P40 service processes, not rollout clients, own GPU memory.

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
    --fake-quant-chunk-rows 16 \
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
schedule, low-memory Adafactor, q4 fake-quant hooks, allocator fragmentation
mitigation, and media-tree mounts:
`OMNICODER_LM_LOSS_CHUNK_TOKENS` defaults to `64` so final-rank language
model logits are chunked below the trainer's 128-token fallback during
20B posttraining. `OMNICODER_FFN_CHUNK_TOKENS` defaults to `256` and is passed
as `OMNICODER2026_FFN_CHUNK_TOKENS` to chunk the SwiGLU sequence dimension
during fake-quant backward recompute.

```bash
cd /home/cereal/omnicoder_2026_work
OMNICODER_RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)" \
OMNICODER_START_STAGE=image \
OMNICODER_STAGE_ORDER=image,video,audio,music,long_context \
OMNICODER_RESUME_CHECKPOINT=weights/training_orchestration_2026/target20b_pipeline_tool_20260524T004357Z/checkpoints/03_tool.pt \
scripts/ai_server_fast_pipeline_20b.sh
```

The launcher defaults to `OMNICODER_MODE=run-full`. Use
`OMNICODER_MODE=run-real` only for the narrower dense-stage plus optional
posttraining lane. Production eval knobs are passed through:

- `OMNICODER_HELDOUT_MAX_RECORDS_PER_FILE=0` evaluates every heldout row per
  modality instead of the bounded profile default.
- `OMNICODER_BENCHMARK_MAX_RECORDS_PER_FILE=0` does the same for checkpoint
  benchmark sample loss.
- `OMNICODER_HELDOUT_SAMPLE_LOSS_TIMEOUT_SECONDS` and
  `OMNICODER_BENCHMARK_SAMPLE_LOSS_TIMEOUT_SECONDS` should be raised for long
  evals on sharded 20B checkpoints.
- `OMNICODER_BENCHMARK_CYCLE`, `OMNICODER_BENCHMARK_MIN_TASKS`, and
  `OMNICODER_BENCHMARK_PREDICTIONS` wire reportable scoring once real
  model-generated predictions exist.
- `OMNICODER_REPORTABLE_TASK_ROOTS` is a comma-separated list of run-scoped
  authorized benchmark task roots, for example
  `weights/data_factory/runs/benchmark_materialization/<run_id>/reportable_2026`.
- `OMNICODER_RERUN_HELDOUT_EVALS=1` invalidates stale heldout sample-loss
  JSON, and resumed stages also reject cached evals when checkpoint, seq_len,
  record cap, or eval/test paths differ.

Pipeline-sharded checkpoints run distributed sample-loss gates immediately.
Reportable prediction scoring remains `pending` for those directories until a
serving/export bridge or prediction artifact is provided. When
`OMNICODER_BENCHMARK_PREDICTIONS` points at a real generated prediction JSONL,
the sharded checkpoint gate runs `run-reportable` directly with those
predictions. Set `OMNICODER_REQUIRE_REPORTABLE_GATE=1` only when a run should
fail closed on missing reportable predictions.

For posttraining-only recovery, do not bend `run-real` or `run-full` into
rerunning dense stages. Use the dedicated `run-posttraining` path after the
active 20B container exits naturally. The resume checkpoint must already be a
complete `omnicoder2026_20b_1m` checkpoint; sharded directories are accepted
only when `.complete.json`, `manifest.json`, all rank files, and per-rank
complete markers are present. This is the correct route after a failed
posttraining stage such as a disk-full `safety_negative_replay` checkpoint
flush: restart from the last complete checkpoint before the failure and slice
the posttraining algorithm order at the failed algorithm.

Posttraining data selection is explicit for balanced all-modal recovery. Use
`python -m omnicoder.data_factory.balanced_allmodal_posttrain_2026` to build
run-scoped SFT, reward, and RLVR JSONL files from curated 2025-2026 sources.
The builder requires nonzero text, code, tool, image, video, audio, music, and
long-context coverage, emits top-level `messages` or prompt/target rows, and
never copies source `token_ids` into the optimizer replay files. Route those
files into live posttraining with `--posttrain-input-jsonl` or
`OMNICODER_POSTTRAIN_INPUT_JSONL`, for example:

```bash
OMNICODER_POSTTRAIN_INPUT_JSONL="reward_weighted_sft_replay=weights/training_orchestration_2026/balanced_allmodal_posttrain_20260526T082100Z/balanced_allmodal_sft.jsonl,grpo_rlvr_replay=weights/training_orchestration_2026/balanced_allmodal_posttrain_20260526T082100Z/balanced_allmodal_rlvr.jsonl"
```

On the AI server, `scripts/ai_server_launch_balanced_allmodal_posttrain_20b.sh`
wraps that path. It finds the latest complete recovery checkpoint, refuses to
start while another fast-card 20B container is active, checks free disk, and
defaults to one 32-step balanced SFT chunk with no periodic 44GB mid-checkpoint
save. Launch additional chunks or GRPO/RLVR chunks by overriding
`OMNICODER_POSTTRAIN_ALGORITHM_ORDER`, `OMNICODER_POSTTRAIN_STEPS`, and
`OMNICODER_SAVE_INTERVAL` after heldout/sample-loss and prediction gates pass.

When a complete checkpoint was saved with an older fast-card layer placement,
the pipeline loader can repartition tensors into the current placement. This is
used to move failed `16,8,40` or `16,14,34` shards back into the current
`16,16,32` layout: each rank loads its own shard first, then streams only
missing layer tensors from the other rank files. Optimizer state is not
restored across that placement change because parameter ordering and ownership
changed. The 2048-token May 25 retries showed that `16,8,40` and `16,14,34`
overfilled the RTX 8000 during fake-quant FFN backward, while earlier
`16,16,32` pressure was tied to 64-row fake-quant chunks on a 3090. The fast
lane now keeps the RTX 8000 largest at 32 layers and uses 16-row chunks.

```bash
cd /home/cereal/omnicoder_2026_work
OMNICODER_MODE=run-posttraining \
OMNICODER_RUN_TAG="resume_posttrain_safety_$(date -u +%Y%m%dT%H%M%SZ)" \
OMNICODER_OUT_DIR="weights/training_orchestration_2026/resume_posttrain_safety_$(date -u +%Y%m%dT%H%M%SZ)" \
OMNICODER_RESUME_CHECKPOINT="weights/training_orchestration_2026/<run>/checkpoints/posttrain/04_orpo_kto_simpo_pair_replay_pipeline" \
OMNICODER_POSTTRAIN_START_ALGORITHM=safety_negative_replay \
OMNICODER_POSTTRAIN_STEPS=32 \
scripts/ai_server_fast_pipeline_20b.sh
```

When launching from a staged clean checkout rather than the mutable
`/home/cereal/omnicoder_2026_work` tree, set
`OMNICODER_REPO=/home/cereal/omnicoder_2026_work/weights/staged_patches/<checkout>`
and keep `OMNICODER_WEIGHTS_ROOT=/home/cereal/omnicoder_2026_work/weights`.
The launcher mounts staged code at `/workspace` and the shared training volume
at `/workspace/weights`, matching the active 20B container layout.

After a posttraining checkpoint completes, run
`scripts/ai_server_run_post_checkpoint_eval_20b.sh` from the AI server for
local-regression evidence. It defaults to the staged all-modal checkout and
shared weights root above, validates a complete three-rank
`omnicoder2026_20b_1m` checkpoint, writes under
`weights/benchmarks_2026/post_checkpoint_eval_<tag>`, and labels all outputs as
engineering regression evidence rather than official/reportable scores. Override
`OMNICODER_EVAL_CHECKPOINT`, `OMNICODER_EVAL_RUN_TAG`,
`OMNICODER_EVAL_GPU_DEVICES`, and `OMNICODER_EVAL_MAX_RECORDS_PER_FILE` to point
at a specific checkpoint, tag, fast-card set, or record cap.

## Dataset And Teacher Sidecars

Do not start a second synchronous 20B target run while a target container owns
fast GPUs `0,4,6`. Use `scripts/ai_server_dataset_training_sidecars_2026.sh`
for additive work:

```bash
cd /home/cereal/omnicoder_2026_work

# Read-only state check.
scripts/ai_server_dataset_training_sidecars_2026.sh preflight

# Trace mining, ComfyUI artifact indexing, external dataset expansion,
# strict local trace export, teacher-job sharding, modality-teacher jobs,
# P40 teacher rollouts, real media-teacher rollouts, and coverage validation.
OMNICODER_MAX_RECORDS_PER_DATASET=1024 \
OMNICODER_TEACHER_LIMIT=256 \
scripts/ai_server_dataset_training_sidecars_2026.sh all

# Build only the strict 2025-2026 Codex/Claude/agent-memory local trace bundle.
scripts/ai_server_dataset_training_sidecars_2026.sh local-traces

# Fresh registry-wave delta without reprocessing the entire registry.
OMNICODER_RUN_ID=external_fresh_wave_delta_$(date -u +%Y%m%dT%H%M%SZ) \
OMNICODER_DATASET_INCLUDE_WAVES=fifth_wave_agentic_rlvr_multimodal_2026_05_24,sixth_wave_formal_code_media_2026_05_24,seventh_wave_agentic_math_code_omni_2026_05_24,eighth_wave_agentic_curation_training_2026_05_24 \
OMNICODER_ENFORCE_DATASET_MINIMA=0 \
OMNICODER_MAX_RECORDS_PER_DATASET=512 \
scripts/ai_server_dataset_training_sidecars_2026.sh external-expansion

# Modality-specific distillation job JSONL for Qwen Image/Edit, LTX, ACE-Step,
# and omni/audio teachers.
scripts/ai_server_dataset_training_sidecars_2026.sh modality-teacher-jobs

# Real artifact-backed Qwen Image/Edit, LTX, and ACE media teacher rollouts.
# Do this at a GPU boundary, not while the 20B target training container is
# actively using the same fast cards.
OMNICODER_MEDIA_TEACHER_ROLLOUT_MODE=live \
OMNICODER_MEDIA_TEACHER_LIMIT=64 \
scripts/ai_server_dataset_training_sidecars_2026.sh media-teacher-rollouts

# Fetch/scan real public-dev benchmark suites into run-scoped local_2026 rows.
OMNICODER_MATERIALIZE_BENCHMARK_TASKS=1 \
OMNICODER_BENCHMARK_MATERIALIZATION_SUITE=core25 \
OMNICODER_BENCHMARK_MATERIALIZATION_MODE=public-dev \
OMNICODER_BENCHMARK_MATERIALIZATION_LIMIT=128 \
scripts/ai_server_dataset_training_sidecars_2026.sh benchmark-materialize

# Read-only proof that artifacts are materialized, not just declared.
OMNICODER_REQUIRE_MEDIA_TEACHER_ROLLOUTS=1 \
OMNICODER_REQUIRE_REPORTABLE_TASKS=1 \
scripts/ai_server_dataset_training_sidecars_2026.sh coverage-report

# Adaptive sample weights and native-1M context ladder for the next train pass.
scripts/ai_server_dataset_training_sidecars_2026.sh mix-plan

# Later status check.
scripts/ai_server_dataset_training_sidecars_2026.sh status
```

The sidecar runner writes run-scoped outputs under:

- `weights/curated_datasets_2026/runs/<run_id>`
- `weights/curated_datasets_2026/runs/<run_id>_local_traces`
- `weights/external_datasets_2026/runs/<run_id>`
- `weights/data_factory/trace_orchestrator_2026/teacher_jobs/<run_id>`
- `weights/data_factory/runs/teacher_jobs/<run_id>/modality`
- `weights/data_factory/teacher_rollouts/<run_id>`
- `weights/data_factory/runs/benchmark_materialization/<run_id>`
- `weights/training_orchestration_2026/runs/<run_id>/manifests/mixture_plan.json`

It promotes `latest` symlinks only after outputs exist. The target training
profile reads external expansion JSONL family files when present, but
`eval_holdout` and `blocked_until_review` files stay out of train paths.
Dataset expansion also rejects synthetic-only train promotion and fail-closes
review, pending, unknown, noncommercial, no-derivatives, holdout, gated,
research, or blocked license markers into non-train buckets. That rule applies
even when a profile entry was accidentally tagged `use_policy: train`.

`coverage-report` runs `omnicoder.data_factory.coverage_validator_2026` and
writes `weights/data_factory/runs/<run_id>/coverage_report.json`. It checks the
actual row counts for all train modality files, strict local traces, external
train rows, agentic SFT/reward/preference/RLVR exports, teacher jobs, modality
teacher jobs, Qwen/P40 rollout outputs, optional Qwen/LTX/ACE media rollout
outputs, mixture plans, and reportable eval task roots. Set
`OMNICODER_COVERAGE_STRICT=1` when missing materialized coverage should stop a
promotion or next-stage launch.
When `benchmark-materialize` has run, coverage also reads
`weights/data_factory/runs/benchmark_materialization/<run_id>/manifests/benchmark_materialization_manifest.json`
and reports local-only versus official/authorized task counts separately.
For stacked benchmark waves, pass each run with repeated
`--benchmark-materialization-root <root>` or
`--benchmark-materialization-manifest <manifest>` flags; the validator
aggregates local-dev and official rows separately and also counts
`<root>/reportable_2026` as a reportable task source when authorized rows are
materialized there. Use `--require-local-benchmark-tasks` for public-dev
regression coverage and `--require-official-reportable-tasks` for release-gate
coverage.
`OMNICODER_REQUIRE_OFFICIAL_REPORTABLE_TASKS=1` prevents public-dev benchmark
rows from satisfying official release-gate coverage.
The benchmark materializer also has a profile audit mode:

```bash
python -m omnicoder.data_factory.benchmark_materializer_2026 \
  --profile profiles/benchmark_suite_2026.json \
  --suite core25 \
  audit-profile \
  --fail-core25 \
  --fail-missing-materializers \
  --fail-known-not-profile \
  --fail-missing-reportable-files
```

Use it before promotion so the native-1M release lane cannot forget a core
benchmark root, snapshot descriptor, local authorized task file, profile
record, or source materializer.

`media-teacher-rollouts` consumes
`weights/data_factory/runs/teacher_jobs/<run_id>/modality/all_modality_teacher_jobs.jsonl`
and writes artifact-backed rows under
`weights/data_factory/teacher_rollouts/<run_id>`, including
`media_teacher_rollouts.jsonl`, `qwen_image_rollouts.jsonl`,
`ltx_video_rollouts.jsonl`, `ace_music_rollouts.jsonl`, and
`media_teacher_rollout_manifest.json`. In `live` mode the action is strict by
default in the sidecar script: failed Qwen/LTX/ACE execution blocks promotion
instead of producing contract-only rows. `dry-run` and `report` exist for
CPU-safe wiring checks only.

## Adaptive Mixture Controller

The static stage order remains the fallback, but `mix-plan` now emits bounded
per-stage weights from curated rows, external manifests, agentic exports,
teacher jobs, and quality/eval signals. The plan includes the native context
ladder `8K -> 32K -> 128K -> 256K -> 512K -> 1M`, flags zero-modality gaps,
and records promotion gates for q4 regression, reward variance, contamination
rejects, and artifact-validation failures. `scripts/ai_server_fast_pipeline_20b.sh`
passes `OMNICODER_ADAPTIVE_WEIGHTS`, `OMNICODER_MIXTURE_PLAN`,
`OMNICODER_CONTEXT_LADDER`, and `OMNICODER_RLVR_ALGOS` into the 20B container.

`run-full` and target `run-real` now add a real long-context curriculum between
dense modality training and posttraining. The stage trains checkpoint rungs for
the configured ladder, defaults to `8K -> 32K -> 128K -> 256K -> 512K -> 1M`,
and resumes each rung from the previous complete checkpoint. Before any rung
starts, `long_context_density_report.json` must pass both token-density and
eligible-row-density checks. The default target contract requires real
long-context rows to reach at least half of each rung's context target, at
least eight eligible rows, and at least 25 percent eligible-row coverage. This
prevents a single long row from masking a mostly padded dataset.

Long-context curation is modality-specific. Curated traces, supplemental text
files, and external datasets use `long_context_target_chars`,
`long_context_text_token_limit`, and `long_context_max_text_file_bytes` instead
of the global short-text caps. Training rows store
`prompt_text_token_count`, `target_text_token_count`, `prompt_char_count`, and
`target_char_count`; curriculum density gates prefer the target count so prompt
or artifact padding cannot inflate the native-1M evidence.

If a posttraining run has already produced a complete checkpoint and only the
native context ladder needs to continue, use `run-long-context` instead of
`run-full` or `run-real`. The mode validates the sharded checkpoint, requires an
existing curation manifest, writes `long_context_resume_summary.json`, and calls
only `run_long_context_curriculum_stage`.

```bash
OMNICODER_MODE=run-long-context \
OMNICODER_RESUME_CHECKPOINT=/workspace/weights/training_orchestration_2026/<run>/checkpoints/posttrain/<complete_stage> \
OMNICODER_CURATION_MANIFEST=/workspace/weights/training_orchestration_2026/<run>/manifests/curation_manifest.json \
OMNICODER_CONTEXT_LADDER=8192,32768,131072,262144,524288,1048576 \
scripts/ai_server_fast_pipeline_20b.sh
```

Reportable benchmark gates can generate model predictions automatically when
authorized reportable tasks exist and no explicit prediction JSONL was supplied.
Set `OMNICODER_BENCHMARK_PREDICTION_BACKEND` plus the matching model, base URL,
API-key environment name, or checkpoint-runner command. The fast-card launcher
quotes the full argv before `bash -lc`, so checkpoint-runner commands with
spaces remain intact inside Docker.

For the 20B launcher, set `OMNICODER_BENCHMARK_MATERIALIZATION_ROOT` to a
run-scoped materializer directory and it will feed `reportable_2026` JSONL rows
into checkpoint evaluation automatically. Public-dev rows under `local_2026`
are opt-in with `OMNICODER_ALLOW_LOCAL_BENCHMARK_TASK_ROOTS=1`; they are for
debugging and realignment, not release reporting. The final `full_run_final`
benchmark gate now fails closed if authorized tasks or model-generated
predictions are missing, so a full run cannot pass by carrying a pending
reportable gate.

When a target training container or sidecar builder is actively running, stage
code/profile updates under `weights/staged_patches/<patch_id>` and apply them
only after a checkpoint and sidecar boundary. Do not overwrite
`profiles/dataset_curation_2026.json`, `scripts/ai_server_dataset_training_sidecars_2026.sh`,
or imported `src/omnicoder/...` modules while the current 20B target or
sidecar Python processes are alive.

P40 usage policy:

- GPUs `1,2,3`: warm Qwen3.6 27B Q4 OpenAI-compatible teacher rollouts.
- GPU `5`: optional probe/short validation if it is idle and cool.
- CPU: trace export, redaction, dedupe, license-tier manifests, JSONL slicing.
- Fast GPUs `0,4,6`: the active 20B pipeline only.

This keeps the RTX 3090s and RTX 8000 saturated by the target model while P40s
produce agentic, math, code, and tool distillation rows in parallel. The
multimodal teacher-job lane creates JSONL work orders for the matching
image/video/audio/music teachers; it does not run those teacher runtimes inside
the Qwen P40 text rollout loop.

Monitor a detached lane with `scripts/ai_server_monitor_fast_pipeline_2026.sh`.
For older hand-launched containers, set `OMNICODER_NAME_FILTER` to the actual
container prefix.

The AI-server target lane is intentionally disjoint from the P40 sidecar lane.
The target container exposes host GPUs `0,4,6` only, which become container
devices `0,1,2`: RTX 3090, RTX 3090, RTX 8000. The production sharded path uses
`torch.distributed.pipelining.PipelineStage`; rank 0 owns 16 layers, rank 1 owns
14 layers, and rank 2 owns 34 layers plus the final norm/output head. This uses
the RTX 8000 headroom first while preserving enough backward-memory slack for
2048-token posttraining, and keeps P40s out of the synchronous target path.
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
single-file reward replay. The bridge manifest must explicitly defer to
`distributed_pipeline_reward_replay` before the sharded optimizer is launched,
and the pipeline loader preserves the same reward/preference/RLVR sample
weights used by native `reward_replay_2026` instead of treating posttraining
JSONL as plain next-token text.

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

## Teacher Distillation Lanes

The 2026 queue now treats live teacher rows as first-class training sources
instead of optional side artifacts. `scripts/ai_server_run_qwen_ltx_distillation_2026.sh`
builds resumable distillation data from:

- Qwen 3.6 27B Q4 through a local OpenAI-compatible llama.cpp server for
  agentic, code, tool, math, text, and long-context correction/reward labels.
- Qwen Image FP8 through ComfyUI for image generation artifact supervision.
- Qwen Image Edit through the validated `TextEncodeQwenImageEdit` workflow with
  a real source image and edit instruction.
- LTX 2.3 22B distilled through ComfyUI for video generation artifact and
  temporal-consistency supervision.

The active 20B trainer keeps the fast cards. Qwen text distillation can run on
an idle P40 partial-offload server while training is active; Qwen Image/Edit and
LTX run after the active trainer exits so ComfyUI has the RTX 3090 path free.

The queued posttraining launcher
`scripts/ai_server_queue_policy_posttrain_after_active_20b.sh` now waits for
the active trainer to exit, resumes or runs the Qwen/Qwen-Image/LTX teacher
distillation, verifies nonempty family files, builds a fresh balanced manifest,
and only then launches the next 20B chunk. Teacher files are inserted before
base curation sources, and the balanced builder accepts `--source-floor` rows
per teacher file before applying shared modality caps. This prevents image
generation rows from crowding out Qwen Image Edit, and prevents broad base
curation from crowding out Qwen 3.6 tool/code/math/text/long-context, LTX, or
agentic trace rows. The active queue also fails closed if
`agentic.clean.jsonl`, `qwen36_text.clean.jsonl`, or
`qwen36_long_context.clean.jsonl` have zero accepted rows after the normal
refusal, eval-holdout, dataset-integrity, watermark/provenance, and media-ref
filters run.

The active balanced defaults now target a larger trainable mix than the old
roughly 20k-row cap slice. Override per-modality caps with
`OMNICODER_POLICY_BALANCED_TEXT_CAP`, `OMNICODER_POLICY_BALANCED_CODE_CAP`,
`OMNICODER_POLICY_BALANCED_TOOL_CAP`, `OMNICODER_POLICY_BALANCED_MATH_CAP`,
`OMNICODER_POLICY_BALANCED_LONG_CONTEXT_CAP`,
`OMNICODER_POLICY_BALANCED_IMAGE_CAP`, `OMNICODER_POLICY_BALANCED_VIDEO_CAP`,
`OMNICODER_POLICY_BALANCED_AUDIO_CAP`, and
`OMNICODER_POLICY_BALANCED_MUSIC_CAP`.

Protected source floors are configurable without weakening filters:
`OMNICODER_BALANCED_AGENTIC_SOURCE_FLOOR`,
`OMNICODER_BALANCED_QWEN_TEXT_SOURCE_FLOOR`,
`OMNICODER_BALANCED_QWEN_LONG_CONTEXT_SOURCE_FLOOR`, and
`OMNICODER_BALANCED_BASE_LONG_CONTEXT_SOURCE_FLOOR`. Media-teacher floors can
be scaled globally with `OMNICODER_MEDIA_TEACHER_SOURCE_FLOOR_SCALE` or by
family with `OMNICODER_MEDIA_TEACHER_IMAGE_SOURCE_FLOOR_SCALE`,
`OMNICODER_MEDIA_TEACHER_VIDEO_SOURCE_FLOOR_SCALE`,
`OMNICODER_MEDIA_TEACHER_AUDIO_SOURCE_FLOOR_SCALE`, and
`OMNICODER_MEDIA_TEACHER_MUSIC_SOURCE_FLOOR_SCALE`.

Fresh integrity-certified sidecar outputs can be added to the next balanced
chunk with `OMNICODER_EXTRA_BALANCED_SOURCES`. Use comma-separated
`modality=/absolute/path/to/file.jsonl` or `modality::/absolute/path` entries.
Use `OMNICODER_EXTRA_BALANCED_SOURCE_FLOORS` with comma-separated
`source_basename.jsonl=count` entries when a small high-value sidecar source
must survive modality caps.
For the 32-row Grok public truth/humor clean file, use the file as an extra
text source and set its source floor to 32, for example
`OMNICODER_EXTRA_BALANCED_SOURCE_FLOORS=grok_public_truth_humor.clean.jsonl=32`.
The queue still runs the dataset-integrity preflight over the final SFT/RLVR
and reward JSONL before launch, so extra OCR, music, TTS, or trace data must
pass the same rejection gate as the base curation sources.

The Qwen 3.6 text lane uses short prompts and compact one-line JSON teacher
targets so the P40 partial-offload servers produce useful throughput. Default
live settings are `OMNICODER_QWEN_TEXT_MAX_TOKENS=224` and
`OMNICODER_QWEN_TEXT_TIMEOUT=420`; higher caps are allowed but should be used
only when the fast-card trainer is idle or a dedicated faster teacher endpoint
is available. The preferred live placement uses all four P40 sidecars on
`127.0.0.1:18081`, `18082`, `18084`, and `18085` with
`OMNICODER_QWEN_GPU_LAYERS=99` so the Q4 teacher uses P40 VRAM instead of
crawling through CPU-heavy partial offload. The rollout client waits through
thermal cooldowns instead of emitting stopped rows.

Qwen Image/Edit and LTX media rollouts are strict live stages by default
(`OMNICODER_MEDIA_STRICT_LIVE=1`). Media failures block the queued launch rather
than creating manifest-only rows. Qwen Image Edit always has a source image: the
script copies the newest available Qwen/image output into the ComfyUI input
folder, or generates a deterministic PNG seed if no prior image exists.

Required teacher family files:

- `qwen36_tool.clean.jsonl`
- `qwen36_code.clean.jsonl`
- `qwen36_math.clean.jsonl`
- `qwen36_text.clean.jsonl`
- `qwen36_long_context.clean.jsonl`
- `qwen_image_generate.clean.jsonl`
- `qwen_image_edit.clean.jsonl`
- `ltx_video.clean.jsonl`

Music/TTS/ACE rows remain required through the music expansion manifest. The
balanced manifest uses teacher-first source order, explicit caps for code,
tool, math, image, video, audio, music, long context, and text, and source
floors for the protected agentic/Qwen/media teacher families.

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
