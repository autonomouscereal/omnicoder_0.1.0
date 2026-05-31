# Optimization Readiness 2026-05-31

This note records the current optimization and proof state for the 20B-class
OmniCoder training path. It is intentionally separate from the README. The
README should stay public-facing; this file is the engineering evidence trail
for the next full-training launch decision.

## Contract Preserved

The optimization pass did not remove the intelligence-oriented architecture
contract:

- Dense one-trunk `omnicoder2026_20b_1m` target.
- 64 layers, 4096 hidden width, 15360 MLP width.
- KDA/CSA/HCA layer cycle.
- Block residual attention remains enabled through `residual_mode=block_attnres`.
- Adaptive latent reasoner remains wired as optional hidden-state deliberation,
  separate from MTP.
- MTP heads remain wired for speculative/speed experiments.
- Native continuous media path and route/artifact-token path remain separate
  output surfaces on the same trunk.
- Assistant/media target masking semantics remain unchanged.
- q4 fake-quant/QAT semantics remain enabled for contract runs.
- Native 1M-context path remains the KDA/CSA/HCA runtime target, not a claim
  that stock GGUF provides full native 1M behavior.

## Implemented Optimizations

The safe changes are implementation-level speed and I/O improvements, not
quality reductions:

- Replaced per-group sparse-attention output projections with
  `QuantAwareGroupedLinear`, a single grouped batched matmul that preserves
  grouped weights and fake-quant behavior.
- Added legacy checkpoint loading from `o_a_groups.*.weight` into the grouped
  projection shape.
- Preallocated local/global attention outputs in fallback loops instead of
  repeatedly building lists and concatenating tensors.
- Added multi-entry RoPE caching keyed by device, dtype, and length.
- Cached block-residual summary position tensors.
- Added an opt-in TorchScript GDN2 scan path for parity testing; it is not a
  default full-training path because full-pipeline profiling exposed backward
  metadata instability.
- Added a tokenized dataset record cache with bounded memory accounting.
- Removed double JSON parsing during JSONL indexing.
- Reused loss-diagnostic target-family counts instead of pulling labels back
  to CPU again during train diagnostics.
- Reduced repeated per-step source-summary payloads by emitting a compact
  reference and live record-cache counters.
- Added launcher passthrough for DataLoader/cache/GDN2 profiling knobs.
- Raised the default q4 fake-quant chunk rows from 256 to 2048 after profiling.
- Promoted the measured fast-card placement default from `16,16,32` to
  `21,21,22` for seq-1024 q4 profiling.

## Verification

Unit and integration checks:

| Check | Result |
|---|---:|
| Focused AI-server container suite | 146 passed, 2 CUDA-only FlexAttention parity tests skipped in CPU-forced run |
| GPU recurrent-path suite | 22 passed |
| Earlier broad AI-server sweep before final doc/checkpoint-load polish | 119 passed |
| Launcher/profile wiring after placement change | passed |
| Grouped projection parity, fake-quant on/off | passed |
| Legacy grouped projection load through full and pipeline checkpoint loaders | passed |
| Legacy grouped projection load shim | passed |
| RoPE multi-entry cache test | passed |
| Dataset record cache and single-parse index tests | passed |
| Diagnostics CPU-sync avoidance test | passed |
| GDN2 JIT scan unit parity | passed |
| Target-count logging with expensive diagnostics disabled | passed |
| Batch-aware profile throughput accounting | passed |

Important caveat: the GDN2 JIT path passed isolated parity tests, but the full
pipeline run hit activation-checkpoint/pipeline backward metadata mismatch on
rank 2. It remains opt-in and experimental.

Latest focused AI-server container suite after loss-timing and target-count
instrumentation: `60 passed in 4.58s`.

Current verification commands:

```bash
docker compose run --rm --no-deps \
  -e CUDA_VISIBLE_DEVICES= \
  -e OMNICODER_ONNX_PROVIDERS=CPUExecutionProvider \
  -e OMNICODER_COMPILE=0 \
  shell bash -lc 'python3 -m pip install --quiet pytest pytest-timeout && python3 -m pytest tests/test_omnicoder2026_initialization.py tests/test_pipeline_pretrain_2026.py tests/test_pipeline_training_telemetry_2026.py tests/test_training_orchestration_2026.py tests/test_profile_matrix_20b_script_2026.py -q'

docker run --rm --gpus all --ipc=host \
  -v "$PWD:/workspace" \
  -w /workspace \
  -e OMNICODER_COMPILE=0 \
  omnicoder:cuda \
  bash -lc 'python3 -m pip install --quiet pytest pytest-timeout && python3 -m pytest tests/test_kda_2026.py -q'
```

## Profile Evidence

All profile runs below used the real 20B-class q4 fake-quant contract path
unless noted otherwise, with no final checkpoint writes.

| Run | Result |
|---|---:|
| seq-256 q4 chunk 256 | ~4.50 train seq-tokens/s |
| seq-256 fake-quant off diagnostic only | ~5.18 train seq-tokens/s |
| seq-256 q4 chunk 2048 | ~5.13 train seq-tokens/s |
| seq-512 q4 chunk 2048 | ~5.77 train seq-tokens/s |
| seq-1024 q4 chunk 2048, old `16,16,32` placement | ~5.64 train seq-tokens/s |
| seq-1024 q4 chunk 2048, `18,18,28` placement | ~6.10 train seq-tokens/s |
| seq-1024 q4 chunk 2048, `20,20,24` placement | ~6.15 train seq-tokens/s |
| seq-1024 q4 chunk 2048, `21,21,22` placement | ~6.22 train seq-tokens/s |
| final default seq-1024 q4 chunk 2048, `21,21,22` | ~5.99 train seq-tokens/s |
| commit `280c257` seq-1024 q4 chunk 2048, no checkpoint | ~6.35 train seq-tokens/s |
| current staged seq-1024 q4 chunk 2048, no checkpoint | ~6.05 train tokens/s |
| current staged seq-1024 q4 block-timing probe | ~5.83 train seq-tokens/s |
| q4 GPipe microbatch-2/batch-2 probe | failed: 3090 rank OOM |
| q4 activation-checkpoint-off probe | failed: 3090 rank OOM |
| q4 GDN2 JIT full-pipeline probe | failed: checkpoint backward metadata mismatch |
| q4 GDN2 compiled full-pipeline probe | timed out before first loss |

The profile matrix showed that `schedule_step` remains the dominant cost.
Batch fetch, host-to-device, telemetry, and log-write timings were small after
the cache and diagnostics changes. Activation-checkpoint-off OOMed, so
activation checkpointing remains required. The 1F1B batch-2 variant was not
stable enough to promote.

Current clean q4 profile:

- Summary: `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_current_q4_seq1024/profile_matrix_summary.json`
- Variant: `fakequant_chunk2048_loss64`
- Loss: `16.594022750854492`
- Step: 1
- Sequence length: 1024
- Batch size: 1
- Max total step time: `169.21987411298323s`
- Max schedule-step time: `169.14335185103118s`
- Training throughput: `6.051298674978949` tokens/s
- Target coverage: `30/30` optimized target tokens, coverage `1.0`
- LM-loss total timing: about `0.079s`
- Selected LM-head CE timing: about `0.012s`
- Selected-position scan timing: about `0.065s`
- Optimizer hook-step timing: about `4.75s`
- Checkpoints written: none

Current q4 block-timing profile:

- Summary: `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_q4_block_timing_targets_seq1024/profile_matrix_summary.json`
- Variant: `block_timing_q4_chunk2048_loss64`
- Loss: `16.72957420349121`
- Target coverage: `24/24` optimized target tokens, coverage `1.0`
- Max total step time: `175.52915790502448s`
- Per-rank forward block totals were only about `13.7s`, `14.2s`, and
  `14.0s`; the remaining wall time is pipeline schedule/backward/recompute
  and waiting, not data loading or LM-head CE.

Remote evidence paths:

- `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_chunk2048_seq512_diag_staged_20260531T024317Z/profile_matrix_summary.json`
- `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_chunk2048_seq1024_staged_20260531T051653Z/profile_matrix_summary.json`
- `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_chunk2048_seq1024_placements_20260531T052159Z/profile_matrix_summary.json`
- `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_defaults_seq1024_20260531T053346Z/profile_matrix_summary.json`
- `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_commit280c257_seq1024/profile_matrix_summary.json`
- `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_current_q4_seq1024/profile_matrix_summary.json`
- `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_q4_block_timing_targets_seq1024/profile_matrix_summary.json`
- `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_schedule_variants_seq1024/profile_matrix_summary.json`
- `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_actckpt_off_seq1024/profile_matrix_summary.json`
- `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_gdn2_jit_seq1024/profile_matrix_summary.json`
- `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_gdn2_variants_seq1024/profile_matrix_summary.json`

The commit-bound profile used `fakequant_chunk2048_loss64`, exited cleanly,
wrote no checkpoint, and produced loss `16.850051879882812`. Loss diagnostics
were disabled for that one-step throughput probe, so target-token coverage was
not collected in that run.

After the target-count instrumentation fix, optimized profiles keep expensive
token-family CE diagnostics off but still broadcast a two-integer
valid/optimized target count from the final rank. This prevents misleading
`0 optimized_target_tokens` logs without reintroducing full label CPU scans or
full diagnostics object broadcasts every step.

## Omnimodal Trainability Proof

A scratch overfit proof trained from fresh initialization on controlled rows for
the shared token/media target path. It is a plumbing and trainability proof, not
a release-quality media-generation claim.

Run directory:

`/home/cereal/omnicoder_2026_work/weights/overfit_proof_2026/omnimodal_overfit_optproof_grouped_hotpath_20260531T024945Z`

Summary:

| Group | Final loss | First-target top-1 |
|---|---:|---:|
| text | ~0.000079 | 1.0 |
| code_tool | ~0.000056 | 1.0 |
| image_ocr | ~0.000072 | 1.0 |
| video | ~0.000057 | 1.0 |
| audio_tts_music | ~0.000053 | 1.0 |
| ledger_all | ~0.000055 | 1.0 |
| omni_all | ~0.000076 | 1.0 |

The proof covered text, code, tool-agent tokens, image/OCR tokens, video/time
tokens, audio/music tokens, TTS/speech tokens, and the full ledger mixture.

## Still Not Proven

These are still gates for a serious full-training or release decision:

- Completed 8K -> 32K -> 128K -> 262K -> 524K -> 1M context ladder.
- Checkpoint-bound heldout sample loss after a real checkpoint.
- Checkpoint-bound target-token diagnostics across all modalities.
- Real decoded media quality for image, video, TTS, audio, and music.
- Reportable benchmark scores from authorized snapshots and official scorers.
- q4 single-card runtime memory, latency, and GGUF bridge validation.
- Full-training stability over long runs with checkpoint cadence enabled.

## Launch Guidance

For the next serious profile/training launch:

- Use the staged code with q4 fake quant enabled.
- Keep activation checkpointing on.
- Keep `FAKE_QUANT_CHUNK_ROWS=2048` unless a longer-context rung proves a
  better value.
- Start fast-card placement at `21,21,22` for short/medium rungs and override
  only when memory profiling says the RTX 8000 needs more headroom.
- Keep GDN2 JIT/compiled scan paths disabled by default.
- Enable checkpoint eval sidecars after complete checkpoint markers.
- Treat public-dev benchmark canaries as diagnostic only.
- Do not claim 1M readiness until the full context ladder has produced finite
  losses and passing recall/probe artifacts.
