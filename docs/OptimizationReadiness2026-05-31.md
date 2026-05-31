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

- Added an SDPA fast path for `BlockAttentionResidual`. Normal token/block
  shapes now use fused scaled-dot-product attention with the same causal
  summary-block mask; the old Python chunked matmul/softmax loop remains as
  the bounded fallback for huge masks.
- Reworked distributed pipeline step routing so rank 0 keeps input IDs,
  the final rank alone receives labels/sample weights by point-to-point
  transfer, and intermediate ranks no longer allocate or receive full target
  tensors.
- Narrowed loss and target-summary sync. The final rank owns loss/target
  diagnostics; rank 0 receives only the scalar/summary needed for logging, and
  all-rank loss broadcast is reserved for checkpoint/final-save boundaries.
- Added cadence-aware final-rank diagnostics so healthy non-diagnostic steps
  avoid full target diagnostics and label CPU pulls.
- Packed token-family count transfer into one compact device-to-host copy
  rather than copying full label tensors to CPU for diagnostics.
- Kept q4 fake-quant STE semantics but moved chunked linear forward/backward
  matmuls onto explicit `torch.mm`/`torch.addmm(..., out=...)` buffers. This
  reduces Python-side tensor assignment and allocator churn while preserving
  the reference STE tests.
- Routed all q4 fake-quant linears, including threshold-edge small matrices,
  through the custom STE linear path. This preserves the same groupwise q4
  forward and STE gradients while avoiding autograd retention of a separate
  full dequantized weight tensor.
- Sparse MQA local attention now attempts native SDPA GQA before falling back
  to expanded K/V tensors, preserving exact local attention semantics while
  avoiding avoidable K/V expansion on runtimes that support `enable_gqa`.
- Added an exact SDPA/additive-bias representation of CSA/HCA sink attention.
  It is gated to FA4-class runtimes or explicit env override; the current
  Ampere-class training cards keep the manual reference path by default because
  measured forward latency was lower there.
- Replaced per-group sparse-attention output projections with
  `QuantAwareGroupedLinear`, a single grouped batched matmul that preserves
  grouped weights and fake-quant behavior.
- Added legacy checkpoint loading from `o_a_groups.*.weight` into the grouped
  projection shape.
- Preallocated local/global attention outputs in fallback loops instead of
  repeatedly building lists and concatenating tensors.
- Preallocated compiled GDN2 chunk outputs and routed tensor-gate KDA/GDN2
  fallbacks through the branch-free tensor scan instead of per-token gate-type
  checks.
- Added an opt-in exact GDN2 checkpoint-scan autograd path behind
  `OMNICODER2026_GDN2_CHECKPOINT_SCAN=1`. It recomputes the fp32 recurrent
  scan in backward so the forward tape does not retain the full per-token
  recurrent-state chain. It remains opt-in pending full 20B profile proof.
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
- Made train loss logging cadence-aware and stopped syncing final-rank loss to
  rank 0 on steps that do not need host-side scalar evidence.
- Made `PipelineLowMemoryAdafactor` opt-in instead of the accidental default
  when `--optimizer adamw` was requested.
- Added launcher passthrough for DataLoader/cache/GDN2 profiling knobs.
- Raised the default q4 fake-quant chunk rows from 256 to 8192 after profiling.
- Kept the production/profile fast-card placement default at `16,16,32`.
  `21,21,22` is slightly useful for short seq-1024 probes but OOMs at
  seq-2048, so it is not the safe default.
- Added cached module-device lookups in the weighted/non-weighted forward
  paths so every block pass does not repeatedly walk parameters to rediscover
  placement.
- Added segmented activation-checkpoint profile variants; they are opt-in
  because the measured segment-2 path did not beat the baseline and segment-4
  failed memory.
- Added a fail-closed pre-full-training proof gate that aggregates target
  coverage, heldout loss, decode/media release gates, reportable benchmark
  evidence, q4 profile evidence, reasoning profile evidence, coverage evidence,
  and GGUF/runtime proof.

## 2026 Forward/Backward Research Alignment

The corrective pass focused on implementation architecture, not superficial
parameter knobs. The current promoted changes line up with the 2025-2026
runtime direction from PyTorch FlexAttention/FlashAttention-4, selective
activation checkpointing, pipeline parallelism, and the Kimi Linear /
Gated DeltaNet-2 / FlashKDA research track:

- Use fused attention kernels where the mask is representable, with exact
  fallbacks for unsupported shapes.
- Avoid blindly recomputing expensive matmuls as a quality-neutral default.
- Remove avoidable full-rank collectives before chasing scheduler knobs.
- Keep the KDA/GDN2 recurrence as the next kernel target, but do not promote
  the current Torch compile/JIT scans into full training until a fused
  chunkwise autograd kernel or FLA-compatible backend passes full-pipeline
  backward parity.

Unpromoted by design:

- No layer/width reduction.
- No removal of residual attention, media paths, MTP heads, q4/QAT path, or
  adaptive latent reasoning.
- No batch-size or chunk-size sweep is treated as the optimization answer.

## Verification

Unit and integration checks:

| Check | Result |
|---|---:|
| Corrective hot-path AI-server CPU suite: KDA, proof gates, model init, q4 fake quant, telemetry, pipeline trainer | 96 passed, 8 CUDA-only tests skipped |
| Corrective pipeline/orchestration AI-server suite after P2P target routing | 131 passed |
| Corrective CUDA fast-card suite for residual SDPA, FlexAttention, q4 fake quant | 8 passed |
| 3-rank CPU/Gloo torchrun smoke with P2P target routing | passed; loss 14.3911 -> 14.3007 over 2 optimizer steps |
| Focused AI-server container suite after proof gate and hot-path patches | 134 passed |
| Earlier focused AI-server container suite | 146 passed, 2 CUDA-only FlexAttention parity tests skipped in CPU-forced run |
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

Latest focused AI-server container suite after proof gate, target masking,
segmented checkpoint, q4 dtype/out-mm, and device-cache patches:
`134 passed in 12.40s`.

Latest corrective proof commands/results:

- CUDA fast-card: `8 passed in 5.22s` for block residual SDPA parity,
  block-mask preservation, backward parity, CUDA FlexAttention parity, and q4
  fake-quant tests.
- CUDA sparse-GQA follow-up: `3 passed in 5.28s` for native SDPA GQA dispatch,
  Flex sparse MQA parity, and residual SDPA backward parity.
- Pipeline/orchestration: `131 passed in 12.65s` after replacing all-rank
  batch/label/weight broadcasts with point-to-point target routing.
- CPU/Gloo distributed smoke: 3 ranks, 2 steps, explicit assistant targets,
  no final checkpoint; completed with positive/decreasing loss and target
  coverage `8/8` on each step.
- Regression sweep: `96 passed, 8 skipped in 6.02s` for KDA CPU parity,
  proof gates, model init, q4 fake quant, telemetry, and pipeline trainer.
- Current hot-path regression subset: `102 passed, 9 skipped in 5.95s` for KDA,
  model initialization, q4 fake quant, pipeline telemetry, and pipeline trainer.
- CUDA hot-path follow-up: `4 passed in 18.32s` for GDN2 checkpoint-scan
  CPU/CUDA gradient parity, GDN2 checkpoint module env-path parity, and GDN2
  compiled-chunk gradient parity. Separate sink-attention/GDN2 CUDA follow-up:
  `3 passed in 18.15s`.
- 3-rank CPU/Gloo torchrun after cadence-aware loss sync and AdamW optimizer
  selection: 2 steps, `14.43017 -> 14.31956`, optimizer logged as `adamw`,
  `optimizer_in_backward_update=""`, final target coverage `5/5`.

Latest local microbench evidence on the AI server RTX 3090-class CUDA path:

- CSA/HCA sink attention, q-heads 32, KV-heads 1, q=512, blocks=128:
  manual reference `0.5283 ms`, SDPA/additive-bias `0.6829 ms`, max abs diff
  `0.00390625`. The SDPA path stays opt-in/FA4-auto instead of default on
  these cards.
- GDN2 tensor-gate scan shape B1/T128/H8/D64: old inline loop comparison
  `31.8634 ms`, branch-free scan `32.8856 ms`, max abs diff `1.52e-05`. This
  patch is kept as a correctness-preserving graph/allocation cleanup, not
  marketed as a current-card throughput win.

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
| current staged seq-1024 safe headroom q4 chunk 8192, `16,16,32` | ~6.26 train seq-tokens/s |
| current staged seq-2048 safe headroom q4 chunk 8192, `16,16,32` | ~6.20 train seq-tokens/s |
| seq-1024 safe headroom q4 chunk 8192 + FFN chunk 1024, `16,16,32` | ~6.43 train seq-tokens/s |
| final staged seq-1024 q4 chunk 8192 + FFN chunk 1024, `16,16,32` | ~6.36 train seq-tokens/s |
| seq-2048 safe headroom q4 chunk 8192 + FFN chunk 1024, `16,16,32` | ~6.22 train seq-tokens/s |
| seq-1024 checkpoint segment-2 q4 chunk 8192 | ~6.12 train seq-tokens/s, slower than baseline |
| seq-1024 checkpoint segment-4 q4 chunk 2048 | failed |
| seq-1024 fake-quant-off diagnostic | ~6.01 train seq-tokens/s, not faster than q4 chunk 8192 |
| q4 GPipe microbatch-2/batch-2 probe | failed: 3090 rank OOM |
| q4 1F1B microbatch-2/batch-2 probe | timed out and was rejected |
| q4 activation-checkpoint-off probe | failed: 3090 rank OOM |
| q4 GDN2 JIT full-pipeline probe | failed: checkpoint backward metadata mismatch |
| q4 GDN2 compiled full-pipeline probe | timed out before first loss |
| final reasoning baseline `fakequant_chunk2048_loss64` | passed, ~6.02 train seq-tokens/s |
| final reasoning effort-2 profile | passed, ~6.42 train seq-tokens/s |
| final reasoning high-effort profile | passed, ~6.32 train seq-tokens/s |
| NCCL P2P-on profile with current q4 default | passed, ~6.31 train seq-tokens/s, slower than P2P-off |

The profile matrix showed that `schedule_step` remains the dominant cost.
Batch fetch, host-to-device, telemetry, and log-write timings were small after
the cache and diagnostics changes. Activation-checkpoint-off OOMed, so
activation checkpointing remains required. The 1F1B batch-2 variant was not
stable enough to promote.

Current best clean q4 seq-1024 profile:

- Summary: `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_ffn1024_globalfast_seq1024_final/profile_matrix_summary.json`
- Variant: `ffn_chunk1024_headroom_q4_chunk8192_loss64`
- Loss: `17.01636505126953`
- Step: global step `1`
- Sequence length: 1024
- Batch size: 1
- Max total step time: `160.91215864697006s`
- Max schedule-step time: `160.85912654199637s`
- Training throughput: `6.363720483338887` sequence tokens/s
- Target coverage: `31/31` optimized target tokens, coverage `1.0`
- LM-loss total timing: about `0.090s`
- Selected LM-head CE timing: about `0.012s`
- Selected-position scan timing: about `0.074s`
- Optimizer hook-step timing: about `4.43s`
- Checkpoints written: none

Current seq-2048 q4 headroom profile:

- Summary: `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_ffn1024_globalfast_seq2048/profile_matrix_summary.json`
- Variant: `ffn_chunk1024_headroom_q4_chunk8192_loss64`
- Loss: `16.856172561645508`
- Step: profile summary did not record a separate local step field
- Sequence length: 2048
- Batch size: 1
- Max total step time: not separately recorded in this profile summary
- Max schedule-step time: `328.99901678902097s`
- Training throughput: `6.2195` sequence tokens/s
- Target coverage: `32/32` optimized target tokens, coverage `1.0`
- LM-loss total timing: about `0.126s`
- Selected-position scan timing: about `0.109s`
- Optimizer hook-step timing: about `6.65s`
- Checkpoints written: none

Current reasoning profile evidence:

- Summary: `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_reasoning_variants_seq1024_final/profile_matrix_summary.json`
- `fakequant_chunk2048_loss64`: loss `16.871400833129883`, target coverage
  `32/32`, no OOM, no checkpoint writes, `6.0201242624766085` sequence
  tokens/s.
- `reasoning_effort2_q4_chunk2048_loss64`: loss `16.635637283325195`,
  target coverage `31/31`, no OOM, no checkpoint writes,
  `6.4221088436572344` sequence tokens/s.
- `reasoning_efforthigh_q4_chunk2048_loss64`: loss `16.809810638427734`,
  target coverage `31/31`, no OOM, no checkpoint writes,
  `6.323432031542531` sequence tokens/s.

Current communication profile evidence:

- Summary: `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_p2p_on_seq1024_final/profile_matrix_summary.json`
- Variant: `p2p_on_ffn_chunk1024_headroom_q4_chunk8192_loss64`
- Status: passed, no OOM, target coverage `31/31`.
- Throughput: `6.306021168981113` sequence tokens/s.
- Baseline comparison: the otherwise matched P2P-off final q4 profile is
  `6.363720483338887` sequence tokens/s with slightly lower rank skew, so
  `NCCL_P2P_DISABLE=1` remains the default.

Corrective microbench evidence:

- Block residual attention, seq 2048 / d_model 512 / rank 64 on RTX 3090:
  old chunked residual-context forward+backward averaged `11.95 ms`; SDPA fast
  path averaged `4.74 ms`, about `2.5x` faster with matching parity tests.
- q4 fake-quant grad-x operand-cast probe produced matching checksums. The
  small-shape timing was neutral (`1.88 ms` vs `1.89 ms`), so the real fix for
  q4/QAT remains a fused dequant-matmul custom autograd kernel; the current
  promoted code keeps STE behavior and avoids extra allocator churn where the
  existing chunked path already uses explicit output buffers.

Current model-contract evidence:

- Summary: `/home/cereal/omnicoder_2026_work/weights/training_runs/model_contract_report_20260531T1127Z.json`
- Status: passed.
- Shape: 64 layers, d_model 4096, MLP 15360, vocab 330000, 2 MTP heads.
- Context ladder configured: 8192, 32768, 131072, 262144, 524288, 1048576.
- Estimate: 22.576B total parameters, 19.566B trunk parameters, 10.51 GiB
  q4 weights, 13.13 GiB estimated native state, and 24GB native fit estimate
  still positive. This is a memory estimate, not yet a GGUF/runtime proof.

Current pre-full-training proof gate:

- Summary: `/home/cereal/omnicoder_2026_work/weights/training_runs/pre_full_training_proof_gate_partial_20260531T1140Z.json`
- Passed checks: `model_contract`, `q4_profile`, `reasoning_profile`.
- Blocked checks: `target_token_coverage`, `heldout_loss_by_modality`,
  `decode_and_media_release_gate`, `data_coverage`, `context_ladder`,
  `reportable_scores`, and `gguf_runtime`.
- Decision: not ready for real full training yet. This is intentional
  fail-closed behavior; the q4/reasoning/model-shape parts are now proven,
  while modality coverage, decode/media quality, 1M ladder, reportable scoring,
  and runtime export still need real artifacts.

Current data coverage audit:

- Summary: `/home/cereal/omnicoder_2026_work/weights/training_runs/data_coverage_latest_composite_20260531T1142Z.json`
- Status: `needs_data`.
- Existing useful counts include curated normalized traces `112000`, external
  train rows `58006`, Qwen agentic/math/code/tool rollout rows `232`, and
  curated modality files for text/code/tool/long-context/image/video/audio/
  music/media-focus.
- Current blockers are external train promotion/integrity-index evidence and
  missing media teacher rollout rows. This means the pool is not empty, but it
  is not yet cleanly promoted as full-training-ready data.

Fastest honest remaining artifact path:

1. Data: run the dataset sidecar coverage/integrity path until the external
   promotion/integrity reports pass and media teacher rollout rows exist.
2. Checkpoint diagnostics: from one complete safe checkpoint, run target-token
   diagnostics and heldout sample loss over eval JSONLs covering text, code,
   tool, math, long-context, image, video, audio, music, TTS, and OCR.
3. Decode/media: generate real checkpoint predictions with nontrivial output
   caps, then run the omnimodal release gate. Current media token artifacts
   prove route/token output only; rendered PNG/MP4/WAV quality still needs a
   real decoder/backend.
4. Long context: run the 8K -> 1M ladder and add a recall/probe aggregation
   artifact with finite loss and pass/fail per rung.
5. Reportable benchmarks: only score authorized/official snapshots with real
   Omnicoder predictions and official scorer artifacts. Canary/public-dev
   plumbing remains diagnostic.
6. Runtime: keep GGUF blocked until a real export plus llama.cpp/LM Studio or
   native bridge runtime proof shows q4 memory under 24GB and usable prefill/
   decode metrics.

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
- `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_headroom8192_lossfast_seq1024/profile_matrix_summary.json`
- `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_headroom8192_lossfast_seq2048/profile_matrix_summary.json`
- `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_ckptoff_q4_vs_fqoff_seq1024/profile_matrix_summary.json`
- `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_ckpt_segments_headroom_seq1024/profile_matrix_summary.json`
- `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_ckpt_segment2_chunk8192_seq1024/profile_matrix_summary.json`
- `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_microbatch2_seq1024/profile_matrix_summary.json`
- `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_ffn1024_globalfast_seq1024_final/profile_matrix_summary.json`
- `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_ffn1024_globalfast_seq2048/profile_matrix_summary.json`
- `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_reasoning_variants_seq1024_final/profile_matrix_summary.json`
- `/home/cereal/omnicoder_2026_work/weights/training_runs/profile_matrix_optproof_p2p_on_seq1024_final/profile_matrix_summary.json`
- `/home/cereal/omnicoder_2026_work/weights/training_runs/model_contract_report_20260531T1127Z.json`
- `/home/cereal/omnicoder_2026_work/weights/training_runs/pre_full_training_proof_gate_partial_20260531T1140Z.json`
- `/home/cereal/omnicoder_2026_work/weights/training_runs/data_coverage_latest_composite_20260531T1142Z.json`

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
- Keep `FAKE_QUANT_CHUNK_ROWS=8192` unless a longer-context rung proves a
  better value.
- Start fast-card placement at `16,16,32` because it passes seq-2048 with the
  current 15360 MLP/full-residual target. Use `21,21,22` only for short
  no-checkpoint probes where OOM risk is acceptable.
- Do not promote microbatch-2, 1F1B, checkpoint-off, checkpoint segment-4, or
  GDN2 JIT/compiled scan paths until a profile proves they are stable and
  faster under the same target contract.
- Keep GDN2 JIT/compiled scan paths disabled by default.
- Enable checkpoint eval sidecars after complete checkpoint markers.
- Treat public-dev benchmark canaries as diagnostic only.
- Run `proof_gates_2026.py` before a serious full-training launch and treat
  missing GGUF/runtime, reportable benchmark, heldout-loss, media decode, or
  context-ladder evidence as a hard block rather than a warning.
- Do not claim 1M readiness until the full context ladder has produced finite
  losses and passing recall/probe artifacts.
