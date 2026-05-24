# OmniCoder

OmniCoder is an experimental omnimodal model stack: one research codebase for a
single model family that can ingest and emit text, code, images, video, audio,
structured artifacts, and tool actions. The long-term target is a compact,
edge-capable model that can reason across modalities without depending on a
separate specialist model for every input or output type.

This repository is not a polished model release or a claim of frontier
performance. It is the architecture lab: sparse experts, long-context memory,
multimodal tokenization, mobile/runtime exports, verifier loops, reward
training, and small runnable canaries in one place. Weights and newer training
work may live outside this public tree until they are ready to publish.

## 2026 Dense Rebuild Status

The active rebuild is now tracked under the `omnicoder2026_20b_1m` contract:
a dense, 20B-class, native-1M-context omnimodal agent model. The exact
parameter count is governed by the 24GB Q4 deployment budget, all-modality
heads, TurboQuant-style compressed state, and native 1,048,576-token context
requirements.

The current target lane is no longer the old sparse-MoE fused-dispatch path.
It uses a dense KDA/CSA/HCA/mHC-inspired trunk, shared ledger-token training
records, strict sharded checkpoints, pipeline sample-loss evaluation, and live
pipeline reward replay for posttraining. The fast-card AI-server profile maps
host GPUs `0,4,6` to container ranks `0,1,2` with `16,16,32` layer placement,
putting the largest shard and final head on the RTX 8000. P40s are sidecars for
teacher rollout, probe jobs, and curation, not synchronous 20B target shards.

Current rebuild docs:

- `docs/Omnicoder2026Redesign.md`
- `docs/TrainingOrchestration2026.md`
- `docs/Omnicoder2026RebuildUpdate.md`
- `docs/DatasetCuration2026.md`
- `docs/DistillationAndRL2026.md`
- `docs/BenchmarkSuite2026.md`
- `docs/AgenticToolTraining2026.md`

### 2026 Data And Training Sidecars

The data lane now has a license-aware external dataset registry and a
nonblocking AI-server sidecar runner. The registry covers current math,
coding/SWE, terminal/browser/tool, image/editing, video, speech/audio, and
music sources such as OpenR1-Math, DAPO, DeepScaleR, OpenMathReasoning,
OpenThoughts2/3, LIMO, DeepCoder, OpenCodeReasoning, SWE-smith, SWE-smith
trajectories, SWE-Gym, Toucan, Nemotron Terminal, Hermes function-calling,
OpenGPT-4o-Image, ShareGPT-4o-Image, MultiEdit, VideoUFO, OpenVid-1M,
Emilia-YODAS, Granary, AudioSkills, MusicBench, Music Arena, and
AR-Omni-Instruct. The May 2026 expansion also tracks Nemotron-SFT-SWE-v2,
SWE-Hero/SWE-Zero trajectories, SWE-ZERO-12M, R2E-Gym, Jupyter-Agent,
OpenResearcher, WebWalkerQA, Terminal-Bench 2.0 trajectories, CodeTraceBench,
OmniAgent/MAgenIT, Nemotron-Image-Training-v3, PRISM/Innovator VL RL,
Pico-Banana, ImgEdit, VIBE, CompBench, Video-MME, LVBench, PhyWorldBench,
MusicEval, MCIF, Multimodal RewardBench 2, and VoiceAgentBench. Each source is
tagged as `train`,
`research_internal`, `eval_holdout`, or `blocked_until_review` before any row is
eligible for training.

External expansion now reports real rows separately from synthetic seed rows
and can fail the run when required family minima are not met. The current
minimum gates require nonzero real coverage across math, coding, agentic
tool-use, terminal/browser agents, image/editing, video, audio/speech/music,
music, and omnimodal understanding before the dataset symlink can be promoted
as a fresh 20B training source.

Useful entry points:

```bash
dataset-expansion-2026 --profile profiles/dataset_curation_2026.json \
  --out-dir weights/external_datasets_2026/latest \
  --download --max-records-per-dataset 1024 \
  --enforce-requirements build

agentic-tool-train-2026 --profile profiles/agentic_tool_training_2026.json build

distill-curriculum-2026 validate --profile profiles/distillation_curriculum_2026.json

scripts/ai_server_dataset_training_sidecars_2026.sh all
```

The sidecar script keeps the 20B target lane on fast GPUs `0,4,6` and uses CPU
plus P40s for trace collection, dataset expansion, teacher-job sharding, and
Qwen3.6 P40 rollouts. It exports agent-memory audit rows before the trace
orchestrator, collects Codex/Claude/Hermes/LM Studio traces, consumes ComfyUI
manifests as first-class multimodal trace sources, and writes
trace-orchestrator outputs to run-scoped writable directories. It now gates
required trace artifacts, refreshes agentic SFT/reward/preference/RLVR exports
from each trace pass, and refreshes those exports again after Qwen3.6 teacher
rollouts so distillation rows can feed the stable paths in
`weights/agentic_tool_training_2026`. Official/protected benchmark rows remain
release-gate evidence only; missing official metadata now produces `local_only`
benchmark results instead of being misreported as public leaderboard quality.

## Why This Exists

Most multimodal systems are pipelines: an LLM delegates to an image model, an
audio model, a video model, a detector, a retriever, and a tool runner. That can
work, but it creates brittle handoffs and makes edge deployment painful.
OmniCoder explores a different direction:

- Use a shared reasoning core across modalities.
- Route work through sparse experts instead of waking the whole model.
- Keep long context bounded with compressed memory, retrieval, and KV policies.
- Make text, code, vision, video, audio, and action heads trainable together.
- Export pieces to realistic local runtimes such as ONNX Runtime, Core ML,
  NNAPI-oriented runners, DirectML, ExecuTorch, llama.cpp/GGUF, and mobile app
  bundles.

The practical bet is that a smaller omnimodal model with good routing, memory,
verification, and device-aware exports can be more useful than a collection of
large disconnected models when privacy, latency, bandwidth, and hardware budget
matter.

## Current Capabilities

- Sparse MoE transformer core with hierarchical routing, expert paging hooks,
  variable-K routing experiments, capacity-aware dispatch, and mobile-minded
  runtime switches.
- Long-context mechanisms including sliding-window decode, memory slots,
  landmark/random-access attention experiments, retrieval/PQ/kNN hooks, KV
  quantization, KV retention sidecars, and learned KV compression experiments.
- Multimodal input/output modules for vision, image VQ, image decoding, video
  VQ, video heads, interpolation, audio tokenization, audio VQ-VAE, vocoder,
  ASR/TTS adapters, 3D latent scaffolding, and cross-modal fusion.
- Reasoning and verification experiments including HRM-style refinement,
  reward modeling, GRPO/PPO/RLHF scaffolds, code verification, multi-solution
  generation, verifier distillation, cross-modal verification, and cycle
  consistency checks.
- Export and runtime work for ONNX decode steps, provider benchmarking,
  DirectML, Core ML, NNAPI-style runners, mobile packaging, GGUF/llama.cpp
  adapters, MLC/TVM hooks, Core ML sample apps, and Android/iOS smoke paths.
- Training and data plumbing for pretraining, LoRA/QLoRA, KD, multimodal JSONL,
  VQA/VL/video/audio datasets, teacher profiles, dataset profiles, acceptance
  thresholds, benchmark canaries, and time-budgeted training probes.

## Repository Map

```text
src/omnicoder/
  modeling/          Core transformer, MoE, routing, attention, memory, kernels
  modeling/multimodal/
                     Image, video, audio, grounding, fusion, VQ, latent heads
  inference/         Generation loops and runtime adapters
  export/            ONNX, Core ML, ExecuTorch, GGUF, mobile packaging
  training/          Pretrain, KD, LoRA, reward, verifier, data loaders
  eval/              Benchmarks, canaries, verifier/eval harnesses
  retrieval/         PQ, graph/RAG, prefix hydration
  sfb/               Symbolic/factorized reasoning experiments
  tools/             CLI entrypoints for training, export, benches, packaging
profiles/            Device, provider, dataset, teacher, and threshold presets
examples/            Tiny JSONL and prompt fixtures for smoke testing
docs/                Current docs plus archived legacy notes
tests/               Unit, smoke, export, provider, and architecture canaries
```

## Quick Start

Use a virtual environment. The base install is enough for CPU smoke tests; use
extras only when you need export, audio, vision, or evaluation packages.

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
copy env.example.txt .env
```

On macOS/Linux:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
cp env.example.txt .env
```

Run a weights-free smoke path:

```bash
python -m omnicoder.inference.generate --prompt "Hello OmniCoder" --device cpu
```

Run the one-button development flow:

```bash
python -m omnicoder.tools.press_play --device cpu --out_root weights
```

The one-button flow is intended to exercise tests, exports, benchmarks, and
release artifacts under `weights/`. It is a development harness, not a magic
training recipe.

## Common Workflows

### Train Or Probe

```bash
python -m omnicoder.tools.run_training --budget_hours 1 --device cuda
```

For quick timing and planning:

```bash
python -m omnicoder.tools.train_probe --budget_minutes 120 --device cuda
```

Teacher and dataset profiles live in `profiles/teachers.json` and
`profiles/datasets.json`. Override paths with environment variables when
running local experiments.

### Export And Benchmark

```bash
python -m omnicoder.export.onnx_export --out weights/release/text/omnicoder_decode_step.onnx
python -m omnicoder.inference.runtimes.provider_bench ^
  --model weights/release/text/omnicoder_decode_step.onnx ^
  --providers CPUExecutionProvider DmlExecutionProvider ^
  --out_json weights/release/text/provider_bench.json
```

Provider thresholds live in `profiles/provider_thresholds.json`.

### Package For A Phone

```bash
python -m omnicoder.tools.export_to_phone --platform android --tps_threshold 15
python -m omnicoder.tools.export_to_phone --platform ios --tps_threshold 6
```

Mobile sample code lives under `src/omnicoder/inference/serverless_mobile/`.

### Enable Runtime Experiments

```bash
set OMNICODER_EXPERT_PAGING=1
set OMNICODER_EXPERT_PAGING_BUDGET_MB=256
set OMNICODER_EXPERT_PREFETCH_N=2
set OMNICODER_MULTI_INDEX_ROOT=weights/retrieval_multi_index
```

Useful knobs include expert paging, KV retention sidecars, activation fake
quantization, variable-K routing, landmark attention, memory slots, windowed
decode, and retrieval augmentation.

## Documentation

- [Current Architecture](docs/ARCHITECTURE_CURRENT.md)
- [Current Quickstart](docs/QUICKSTART_CURRENT.md)
- [Legacy Architecture Notes](docs/legacy/Architecture.md)
- [Legacy Dataset Notes](docs/legacy/Datasets.md)
- [Legacy Teacher Notes](docs/legacy/Teachers.md)
- [Backlog](todo/TODO.md)

The legacy docs are retained because they contain useful research notes, but
they do not fully describe the present intent of the project. Start with the
current docs above.

## Status

This is an active research codebase. Some modules are runnable, some are smoke
tested scaffolds, and some are architectural experiments waiting for larger
training runs or unpublished weights. Treat the repo as a map of the model
system and a set of reproducible experiments, not as a packaged consumer model.

## Design Principles

- One model family should reason across all modalities.
- Edge constraints are architecture constraints, not an afterthought.
- Every capability should have a small canary, export path, or benchmark hook.
- Verification and reward loops should be built into the system, not bolted on.
- Runtime truth matters: provider benches, device profiles, and memory budgets
  are first-class artifacts.

## License

See `LICENSE`.
