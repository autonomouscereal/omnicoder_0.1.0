# Omnicoder 2026 Rebuild Update

Date: 2026-05-24

This runbook collects the concrete repo updates made during the dense
20B-class, native-1M-context rebuild. It is intentionally operational: what was
changed, where the code lives, how the AI-server lane is launched, what is
currently training, and what evidence exists so far.

## Current Contract

Target profile: `omnicoder2026_20b_1m`

Primary constraints:

- Dense target model, replacing the old sparse-MoE/fused-dispatch target path.
- Native context length of `1048576` tokens.
- 24GB Q4 deployment budget, with exact parameter count governed by the memory
  budget rather than a fixed literal value.
- All-modality input and output training surface: text, code, tool traces,
  image, video, audio, music, and long-context records.
- GGUF-first adoption path where compatible, with native runtime retained for
  true 1M-context and omnimodal output paths that llama.cpp-compatible runtimes
  cannot represent yet.
- No target training on P40s. P40s are sidecars for teacher rollout, data
  curation, and probe-scale verifier work.

## Architecture Direction

The 2026 target no longer treats the old MoE architecture as the production
path. The current design is documented in `docs/Omnicoder2026Redesign.md` and
uses a dense KDA/CSA/HCA/mHC-inspired trunk with shared ledger-token training.

Important design choices:

- One dense trunk for reasoning and modality interaction.
- Modality codecs and renderers stay at the edge; the fused part is the
  ledger/trunk/decision space.
- The training lane optimizes for depth and memory placement first: the RTX
  8000 owns twice the layer count of either RTX 3090 shard.
- Q4/TurboQuant compatibility is trained through fake-quant chunks and
  low-memory update rules, then must be revalidated at export.
- Benchmark and heldout records remain quarantined from training exports.

## AI-Server Target Lane

Canonical launcher:

```bash
scripts/ai_server_fast_pipeline_20b.sh
```

Canonical monitor:

```bash
scripts/ai_server_monitor_fast_pipeline_2026.sh
```

Current fast-card mapping:

| Host GPU | Card | Container rank | Layer placement |
|---:|---|---:|---:|
| 0 | RTX 3090 | 0 | 16 |
| 4 | RTX 3090 | 1 | 8 |
| 6 | RTX 8000 | 2 | 40 plus final norm/head |

The target Docker run must include:

```bash
--gpus '"device=0,4,6"'
--ipc=host
--ulimit memlock=-1
--ulimit stack=67108864
```

The `--ipc=host` setting is required for NCCL shared-memory segments during the
pipeline-stage target lane. Without it, distributed startup can fail before the
model reaches useful training.

## Active Training Run

Current run:

```text
container: omnicoder_target20b_pipeline_media_ipc_20260524T012805Z
run root:  /home/cereal/omnicoder_2026_work/weights/training_orchestration_2026/target20b_pipeline_media_ipc_20260524T012805Z
monitor:   /home/cereal/omnicoder_2026_work/weights/training_orchestration_2026/monitors/target20b_media_ipc_monitor_20260524T050949Z.jsonl
```

The run resumed into media stages after text/code/tool progress and was observed
training the image stage with loss decreasing from roughly `42.1672` to
`19.3426`, best observed `12.8592`, with a complete three-rank checkpoint at:

```text
/home/cereal/omnicoder_2026_work/weights/training_orchestration_2026/target20b_pipeline_media_ipc_20260524T012805Z/checkpoints/04_image.step96
```

GPU memory observed during the target run:

```text
GPU0 RTX 3090: about 22.8GB / 24GB
GPU4 RTX 3090: about 20.7GB / 24GB
GPU6 RTX 8000: about 35.5GB / 46GB
```

This confirms the target job is not the earlier 6B-class verifier lane; it is
using the 20B-class sharded placement profile and putting the largest shard on
the RTX 8000.

The May 25 posttraining retry proved that `16,16,32` can overfill the middle
3090 during fake-quant backward while the RTX 8000 still has room. The launcher
now defaults to `16,8,40`, `OMNICODER_FAKE_QUANT_CHUNK_ROWS=32`, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` so resumed reward-replay and
long-context jobs bias depth and activation pressure toward the 48GB card.

## P40 Sidecar Lane

P40s are intentionally excluded from the synchronous target pipeline to avoid
slowing the 3090/RTX8000 target path or tripping unsupported compiler/Triton
paths. They are used as sidecars instead.

Qwen3.6 27B Q4 llama-server endpoints used for teacher rollout:

```text
127.0.0.1:18082
127.0.0.1:18084
127.0.0.1:18085
```

Curated nonempty teacher rollout export:

```text
/home/cereal/omnicoder_2026_work/weights/curated_datasets_2026/latest/jsonl/train_teacher_rollouts_qwen36_p40.jsonl
```

Current accepted rows: `24`

The older empty-content rollout batch was deliberately excluded.

## Code And Setup Collected In This Commit

Training orchestration:

- `src/omnicoder/training/training_orchestration_2026.py`
- `src/omnicoder/training/pipeline_pretrain_2026_dense.py`
- `src/omnicoder/training/pretrain_2026_dense.py`

Evaluation:

- `src/omnicoder/eval/pipeline_sample_loss_2026.py`
- `src/omnicoder/eval/sample_loss_2026.py`

Data factory / teacher rollout:

- `src/omnicoder/data_factory/openai_teacher_rollout_2026.py`

Profiles and scripts:

- `profiles/training_orchestration_2026.json`
- `scripts/ai_server_fast_pipeline_20b.sh`
- `scripts/ai_server_monitor_fast_pipeline_2026.sh`

Tests:

- `tests/test_training_orchestration_2026.py`

Docs:

- `README.md`
- `docs/TrainingOrchestration2026.md`
- `docs/Omnicoder2026RebuildUpdate.md`

## Behavioral Fixes Included

- Strict sharded checkpoint completeness checks now require the expected
  pipeline world size and contiguous rank files before resume/eval/posttraining.
- Pipeline checkpoint manifests include `world_size`.
- Target runtime rejects P40s in the visible CUDA set for synchronous target
  training.
- Target runtime validates that the largest layer placement lands on the largest
  visible GPU.
- Incomplete sharded resumes are rejected instead of being treated as usable.
- Pipeline sample-loss evaluation supports sharded target checkpoints.
- Sample-loss cross entropy casts logits to fp32 to avoid fp16 infinity results.
- Live posttraining reward replay runs through the distributed pipeline path for
  sharded checkpoints.
- Posttraining status aggregates failed stages instead of treating partial
  success as pass.
- Qwen/OpenAI-compatible teacher rollout treats empty content as failure and
  supports `reasoning_content`, `reasoning`, and `analysis` fallbacks.
- Teacher rollout includes a thermal guard for P40 sidecar runs.

## Verification Evidence

Local focused tests:

```powershell
PYTHONPATH=G:\omnicoder_0.1.0\src python -m pytest G:\omnicoder_0.1.0\tests\test_training_orchestration_2026.py -q
```

Latest expected result:

```text
17 passed
```

Local syntax checks were run with `py_compile` against the changed Python files.

AI-server checks:

- Python compile probes passed inside the CUDA image for changed orchestration
  modules.
- Launcher scripts passed `bash -n`.
- Truncated sharded-checkpoint probe proved complete three-rank checkpoints pass
  and intentionally truncated four-rank expectations fail.
- Server-side pytest was not available in the current
  `omnicoder:cuda-posttrain-2026` image because the image does not include
  `pytest`.

## Operating Notes

Use the monitor script to check target lane state:

```bash
OMNICODER_CONTAINER_NAME=omnicoder_target20b_pipeline_media_ipc_20260524T012805Z \
OMNICODER_OUT_DIR=/home/cereal/omnicoder_2026_work/weights/training_orchestration_2026/target20b_pipeline_media_ipc_20260524T012805Z \
scripts/ai_server_monitor_fast_pipeline_2026.sh
```

Do not resume a target 20B run from `gpu_sidecar/*` artifacts. Sidecar outputs
can feed curation, teacher rows, or verifier/probe reports, but target
checkpoints must come from the fast-card sharded lane.

Do not report benchmark leaderboard quality from plumbing-only gates. The
release gates must run the real benchmark adapters with protected data
quarantine intact before any public benchmark claims are made.
