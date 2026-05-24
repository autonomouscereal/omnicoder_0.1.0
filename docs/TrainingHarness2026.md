# Omnicoder 2026 Training Harness

The 2026 harness is a staged run system, not a single trainer script. It records
run manifests, stage logs, metrics, checkpoints, data exports, teacher-job files,
and eval smoke outputs under a JSONL registry by default. Raw PostgreSQL mirroring
is available through `schemas/training_runs_2026.sql` when an Omnicoder database is
running.

Core command:

```powershell
python -m omnicoder.training.full_harness_2026 run `
  --profile profiles/training_harness_2026.json `
  --trace-input data/raw/agent_memory_events_2026.jsonl `
  --stages all
```

Important stages:

- `ingest_trace`: normalizes Codex/Hermes/Claude memory exports.
- `quality_score`: applies heuristic quality, duplicate, and secret gates.
- `contam_scan`: scans against protected benchmark registry names/rubrics.
- `export_sft`: writes chat-style SFT JSONL.
- `teacher_jobs`: creates local teacher-distillation job JSONL.
- `sft_qlora_bridge`: optional TRL/PEFT/QLoRA bridge trainer or dependency
  check/dry run for Hugging Face-compatible teacher/student checkpoints.
- `native_train`: launches `pretrain_2026_dense.py` against the exported data.
- `eval_smoke`: writes a registry-based eval smoke result.
- `context_budget`: records native-1M memory budget estimates.

The harness currently validates the native Omnicoder trunk and bridge data lane.
Megatron-Bridge, NeMo RL, TRL/PEFT, GRPO/DAPO/RLVR, and QARL are represented in
the recipe contract, but they remain separate execution backends until those
containers are installed and pinned on the AI server.

Supporting commands:

```powershell
python -m omnicoder.training.run_registry_2026 status --run-id <run_id>
python -m omnicoder.training.metrics_2026 --log weights/harness_2026/<run_id>/logs/native_train.log
python -m omnicoder.training.sft_qlora_2026 --check_deps --load_in_4bit
```

Native checkpoints saved by `pretrain_2026_dense.py` now include model weights,
optimizer state, RNG state, global step, data hash, run arguments, architecture
manifest, and runtime provenance.

## Rank-local FSDP checkpoints

FSDP target runs save checkpoint directories with `manifest.json` plus one
`rankNNNNN.pt` file per rank. These are intentionally rank-local; a single
process cannot load them as a normal native `.pt` checkpoint.

Inspect a directory:

```powershell
python -m omnicoder.eval.fsdp_checkpoint_2026 inspect `
  --checkpoint-dir weights/training_orchestration_2026/run/checkpoints/08_long_context.pt
```

Evaluate sample loss directly with the same world size used at save time:

```powershell
python -m torch.distributed.run --nproc_per_node 8 `
  -m omnicoder.eval.sample_loss_2026 `
  --checkpoint weights/training_orchestration_2026/run/checkpoints/08_long_context.pt `
  --data-dir weights/harness_2026/run/data/exports `
  --profile omnicoder2026_20b_1m `
  --out weights/eval/fsdp_sample_loss.json
```

Consolidate once, then reuse the normal single-file eval and benchmark paths:

```powershell
python -m torch.distributed.run --nproc_per_node 8 `
  -m omnicoder.eval.fsdp_checkpoint_2026 consolidate `
  --checkpoint-dir weights/training_orchestration_2026/run/checkpoints/08_long_context.pt `
  --profile omnicoder2026_20b_1m `
  --out weights/training_orchestration_2026/run/checkpoints/08_long_context.consolidated.pt
```

The benchmark smoke harness now fingerprints rank-local FSDP directories from
their manifest and rank-file metadata, so `--model <fsdp-checkpoint-dir>` is
stable for registry/result JSON even before consolidation.
