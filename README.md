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
posttraining through `posttrain_bridge_2026` plus pipeline reward replay for
sharded checkpoints. The fast-card AI-server profile maps host GPUs `0,4,6` to
container ranks `0,1,2` with `16,16,32` layer placement, putting the largest
shard and final head on the RTX 8000. P40s are sidecars for teacher rollout,
probe jobs, and curation, not synchronous 20B target shards. Reportable
benchmark gates now require authorized snapshot metadata and separate model
outputs; smoke/contract fixtures are explicitly local-only.

The canonical fast-card launcher now defaults to `run-full`, not the narrower
`run-real` lane. Production runs can widen heldout and benchmark sample-loss
coverage with `OMNICODER_HELDOUT_MAX_RECORDS_PER_FILE` and
`OMNICODER_BENCHMARK_MAX_RECORDS_PER_FILE`; setting either to `0` means all
records rather than the profile's bounded default. Sharded pipeline checkpoint
gates run real distributed sample-loss immediately. They leave reportable
prediction scoring pending when no generated prediction artifact exists, but
consume `OMNICODER_BENCHMARK_PREDICTIONS` once a serving/export path produces
real model outputs for authorized tasks. That lets release gates score sharded
20B checkpoints directly without pretending smoke fixtures are reportable.

The 20B lane now inserts an explicit long-context curriculum after dense
all-modality training and before posttraining. It walks the native context
ladder from `8K` through `1M`, resumes each rung from the previous sharded
checkpoint, runs heldout sample-loss checks per rung, then gates the final
long-context checkpoint. Long-context rows are fail-closed: the curation layer
must produce enough eligible real target tokens per row and enough eligible
rows across the dataset, so one giant row cannot hide mostly padded short
records. Curated traces, supplemental files, and external long-context datasets
now use `long_context_target_chars`, `long_context_text_token_limit`, and
`long_context_max_text_file_bytes` instead of the short global text cap.

Reportable benchmark gates can now generate prediction JSONL automatically for
authorized task snapshots when a real backend is configured through
`OMNICODER_BENCHMARK_PREDICTION_BACKEND` plus the matching model, endpoint, or
checkpoint-runner settings. The launcher preserves argument boundaries for
checkpoint runners with spaces, and generated prediction files are scored by
the existing reportable gate instead of being treated as fixture outputs.
Official/public benchmark materialization is now separate from scoring:
`benchmark-materialize-2026` writes run-scoped task JSONL under
`weights/data_factory/runs/benchmark_materialization/<run_id>/`, separating
`local_2026` public/dev rows from `reportable_2026` authorized rows. The
AI-server sidecar can opt into this with `OMNICODER_MATERIALIZE_BENCHMARK_TASKS=1`;
the 20B launcher can consume those roots with `OMNICODER_REPORTABLE_TASK_ROOTS`
without mutating the live benchmark profile.

The posttraining orchestrator is fail-closed for 20B pipeline replay: a failed
or incomplete sharded optimizer stage stops the remaining replay stack instead
of silently continuing from an older checkpoint. The profile also enables
posttraining checkpoint retention, keeping the active and most recent complete
pipeline shards while pruning older stage shards so long RL stacks do not fill
the AI-server training volume mid-run. For sharded checkpoints, the bridge now
must explicitly authorize `distributed_pipeline_reward_replay`, and the pipeline
trainer carries reward/preference/RLVR sample weights through the loss instead
of flattening those rows into unweighted text.

Posttraining-only recovery is a first-class path. Use
`training-orchestration-2026 run-posttraining` or
`OMNICODER_MODE=run-posttraining scripts/ai_server_fast_pipeline_20b.sh` with a
complete existing `omnicoder2026_20b_1m` checkpoint and, when needed,
`--posttrain-start-algorithm safety_negative_replay` /
`OMNICODER_POSTTRAIN_START_ALGORITHM=safety_negative_replay`. This resumes live
distributed reward/preference/RL replay without rerunning dense pretraining and
without accepting incomplete sharded checkpoints.

Long-context-only recovery is also first-class. Use
`training-orchestration-2026 run-long-context` or
`OMNICODER_MODE=run-long-context scripts/ai_server_fast_pipeline_20b.sh` with
`OMNICODER_RESUME_CHECKPOINT` pointing at a complete sharded checkpoint and
`OMNICODER_CURATION_MANIFEST` pointing at the curated corpus manifest. This
runs only the native context ladder and refuses to rebuild curation, dense
pretraining, distillation, or posttraining implicitly.

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
music sources. The May 24, 2026 expansion adds hard agentic and multimodal
coverage beyond the initial OpenR1/OpenThoughts/SWE-smith/Nemotron/ComfyUI
mix:

- Math and RLVR: DeepMath-103K, AI-MO NuminaMath 1.5, DAPO, DeepScaleR,
  OpenMathReasoning, Polaris Nemotron verifiable math, Korean NuminaMath,
  R-HORIZON, Reasoning Core formal RLVR, UniRRM-RL, Nemotron Math Proofs,
  UltraData-Math, GLM-5.1 reasoning traces, MathVision, AIME 2025/2026
  holdouts, and HLE/HLE-Verified holdouts.
- Coding and SWE agents: SWE-Dev, SWE-Next, DeepSWE/Kimi-K2 trajectories,
  SWE-Swiss repair SFT/RL, SWE-Factory-Gym, SWE-bench Pro/ABS/Multilingual,
  SWE-Lancer, SWE-PolyBench, SWE-bench Live variants, CodeElo, ICPC-Eval,
  JetBrains trajectory analysis rows, SWE-Hero/SWE-Zero, R2E-Gym V1/SFT
  trajectories, OpenHands CodeScout rollouts, AIDev PR traces, SWE-CI,
  Fixbench-RTL, SWE-Synth, and Jupyter-Agent.
- Browser, GUI, terminal, and tool agents: MCP-Atlas, Nemotron RL tool-use,
  WebAgent-R1, WebShepherd, WebExplorer, DeepDive, WebArena Infinity,
  BrowserAgent, Web Agent Graph, WebChain, OSWorld 2, Magic-RICH, TerminalWorld,
  Multi-Docker-Eval, Terminal-Bench 2.0 trajectories, GUI-360, AgentNet,
  Computer Use Large, VideoCUA, AgentSynth, Smol2Operator/Aguvis, tau2/AReaL
  verified tool traces, BFCL/ComplexFuncBench, APEX, WildClawBench, ClawBench,
  CodeTraceBench, Hermes, xLAM, Toucan, NVIDIA ToolScale/When2Call/Nemotron
  Agentic and Cascade RL/SFT data, cleaned Toucan/Hermes/memory-agent/web-QA
  tool SFT, Qwen tool-calling, browser-agent SFT, Terminal-Bench 2 HF
  trajectories, Toolathlon trajectories, Plan-RewardBench, Agentic
  Chain-of-Thought Coding SFT, R2E-Gym verifier/testing-agent trajectories,
  APIGen-MT/WebShaper research rows, PrimeIntellect SYNTHETIC-1 SFT/preference,
  CUA-Gym, A11y-CUA, Telos tool trajectories, MEnvData SWE trajectories,
  JetBrains SWE-Smith trajectories, Kimi-K2 rejection-sampled DeepSWE,
  WebArena Pro trajectory reviews, Mind2Web UTG eval trajectories, Turkish
  mobile function-calling, ScreenSpot-Pro, WorkArena/ScaleCUA blocked review
  rows, and local Codex/Claude/Hermes/agent-memory traces.
- Extra math/coding RLVR: Nemotron RL super blends, Cascade RL SWE/RLHF,
  Nemotron competitive coding, PrimeIntellect verifiable coding/math review
  rows, Math-RLVR 773K, High-Quality-Verifiable-Math-156K review rows,
  RLVR Linearity, Nous RLVR Coding Problems, IFDecorator, NuminaMath-LEAN,
  Kimina/Lean proof rows, FoVer process-verifier labels, TritonBench,
  KernelBench research rows, UTBoost/LiveCodeBench eval rows, ARC-AGI-2
  public-training rows, and SWE-Agent LM 32B R2E-Gym trajectory review rows.
- Multimodal generation and reward: FineVision/FineVisionMax, ScaleEdit,
  GPT-Image-Edit, NHR-Edit, CrispEdit, BAGEL-World, Rapidata image
  preferences, HPDv3, ImgEdit, UniREdit, BLIP3o, UniWorld, text-to-image DPO
  preferences, VideoGen-RewardBench, Rapidata text/image-to-video preferences,
  JavisInst-Omni, Javis AV fine-tune, TTSDS listening tests, SAM Audio data,
  Prompt2MusicBench, OpenMMReasoner, DeepVision, RLFR-VLM, Open-MM-RL,
  MMMU Pro, Video-MME-v2, LVOmniBench, JointAVBench, AVGen-Bench, VBench 2.0,
  PARADE_audio, AudioMC, WildSpeech-Bench, WorldSpeech, Granary,
  NonverbalTTS, Music Arena, Captioned AI Music Snippets, NVIDIA MMOU/QCalEval/
  SAGE/NitroGen references, BLIP3o short/60K data, Rapidata 4o/Imagen4/
  Seedream/Flux/Sora/Genmo/Seedance/Hailuo preference sets, JoyAI
  OpenSpatial, OmniContext, WebSRC, VTC-Bench visual tool chains, FiVE,
  OmniEdit-Bench, OpenAudioBench, Ming audio edit, OmniDoc-TokenBench,
  ChartQAPro, OmniDoc OCR correction, OmniCorpus CC/YT, OmniGUI, OCRBench v2,
  MM-IQ, Real5-OmniDocBench, MMVU,
  VideoVista CoTs, WorldSense/MMOU/MMAU holdouts, AudioSet/zero-shot/adversarial audio
  instruction rows, NVIDIA HiFiTTS2/LongAudio/AF-Think/AF-Chat/MF-Skills,
  SpeechJudge, AudioCoT, EditReward-Bench, IESBench, VideoPhy2, VBench-I2V,
  SVI, MieDB, OpenVE-3M, DocVQA 2026, ChartMuseum, Kirundi and Indonesian
  TTS/speech rows, TrueMuse, tokenized omni/Emu3 review rows, and
  StoryBench/OmniBench reward/eval holdouts.
- Seventh-wave May 24, 2026 additions now cover OpenThoughts-Agent v1 SFT/RL,
  Edge-Agent WebSearch, Exgentic traces, CUDA-Agent-Ops, AI CUDA Engineer,
  CodeX-2M Thinking, GitHub code review, INTELLECT-2 RL, DeepSeek-ProverBench,
  MathArena model-output verifier rows, ECHO/TRIG/MIGE/ReShape/DLE/Inter-Edit,
  OpenS2V-5M, VEFX, VideoGen-Eval, VABench, OmniVideoBench, UniM,
  FysicsWorld, MME-Unify, Zero-To-CAD, ASID-1M, VisCoR, CMI-Pref, VoxEval,
  LongSpeech, UltraEval-Audio, and ATTM 2026. These are tagged as
  `seventh_wave_agentic_math_code_omni_2026_05_24` for delta materialization.
- Eighth-wave May 24, 2026 additions add MCP-Universe and MCPMark trajectories,
  Qwen 3.6 and Qwen agent-distillation tool traces, Computer Use PSAI,
  BrowseCompLongContext, BrowseComp-Plus corpus/QA holdout, PaperBench,
  TheAgentCompany, AudioMarathon, Audio-Alpaca, OpenAudioBench, and
  VideoRewardBench. These are tagged as
  `eighth_wave_agentic_curation_training_2026_05_24`.
- Tenth-wave May 25, 2026 additions add MCPToolBench++ Preview, WebBench,
  mAIME2025, MMLongBench, NoLiMa, LongCodeBench, SagaScale, AcademicEval,
  FineWeb2, Common Pile v0.1, LEMAS, Emilia, AudioBench, MMAU-Pro, MMAR,
  CMI-Bench, MUSE, MPBench, RTV-Bench, and RIVER Bench. These are tagged as
  `tenth_wave_curated_benchmarks_2026_05_25`; FineWeb2 and Common Pile are
  train-eligible, while benchmark and speech-media rows stay eval-only or
  research-internal until licensing and holdout checks pass.
- Eleventh-wave May 25, 2026 additions add LiveMCPBench, SRA-Bench, SkillRet,
  DAPO-Math-17k, Guru RL 92K, MemoryAgentBench, OmniGAIA,
  Omnimodal-Agent-SFT-2K, OmniRAG-Agent, VSTAT, and Tricky TTS under
  `eleventh_wave_agentic_omni_eval_2026_05_25`. The benchmark materializer now
  treats HF file patterns as strict metadata-only constraints so public-dev
  benchmark pulls do not accidentally download full video/audio corpora.
- Twelfth-wave May 25, 2026 additions add AMA-Bench and SMMBench under
  `twelfth_wave_agent_memory_state_2026_05_25`, plus STATE-Bench as a
  benchmark-suite gate. These extend long-horizon agent memory, source-
  distributed multimodal memory, and stateful enterprise tool-use evaluation.
  STATE-Bench materialization now prefers canonical
  `state_bench/domains/*/tasks/*.json` task definitions, preserves split,
  sandbox-state assertions, user simulator rules, and scoring requirements, and
  attaches matching public train trajectories only as reference evidence.
  SMMBench uses pinned raw `Samples/cluster_*/QA_sample.json` task files; its
  HF `default/train` imagefolder split is image-only and is rejected for scored
  benchmark material. HF image/audio/video features are otherwise cast
  metadata-only during curation and benchmark materialization so rows can be
  extracted without media codec failures.
- Thirteenth-wave May 25, 2026 additions add Agentic-MME, MM-ToolBench/TOBench
  tracking, ABC-Bench, LongBench-Pro, MEGA-Bench, StepEval-Audio-360, and
  IndiMathBench to release gates. The dataset registry also stages
  DeepResearch-9K, MMFineReason-1.8M Qwen3-VL Thinking, and Lean Math Formal
  Corpus v4.27.0 as research-internal sources until license/decontamination
  review clears them for weight-bearing training.
- Fourteenth-wave May 25, 2026 additions add MCPVerse, UI-Vision, and
  ViMUL-Bench under `fourteenth_wave_agentic_gui_video_eval_2026_05_25`.
  These cover real MCP tool-server tasks, desktop GUI grounding/action
  prediction, and multilingual audio-video reasoning while remaining
  eval-holdout by default. LongCodeBench materialization now points at the
  concrete `Steefano/LCB` HF source instead of paper-only metadata.

Each source is tagged as `train`, `research_internal`, `eval_only`,
`benchmark_holdout`, or `blocked_until_review` before any row is eligible for
training. Expansion is fail-closed: review, pending, unknown, noncommercial,
no-derivatives, gated, research-only, or holdout license markers cannot
materialize into the `train` bucket even if a profile row is accidentally marked
train.

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

dataset-expansion-2026 --profile profiles/dataset_curation_2026.json \
  --out-dir weights/external_datasets_2026/runs/fresh_wave_delta \
  --download --max-records-per-dataset 512 \
  --include-wave fifth_wave_agentic_rlvr_multimodal_2026_05_24 \
  --include-wave sixth_wave_formal_code_media_2026_05_24 \
  --include-wave seventh_wave_agentic_math_code_omni_2026_05_24 \
  --include-wave eighth_wave_agentic_curation_training_2026_05_24 \
  build

agentic-tool-train-2026 --profile profiles/agentic_tool_training_2026.json build

distill-curriculum-2026 validate --profile profiles/distillation_curriculum_2026.json

python -m omnicoder.training.training_orchestration_2026 \
  --profile profiles/training_orchestration_2026.json mix-plan

python -m omnicoder.data_factory.coverage_validator_2026 \
  --run-id <run_id> --require-media-teacher-rollouts --require-reportable-tasks

scripts/ai_server_dataset_training_sidecars_2026.sh all
```

The coverage validator is read-only. It reports the actual row counts for
curated modality JSONLs, strict local traces, external expansion, agentic
SFT/reward/preference/RLVR exports, teacher jobs, Qwen/P40 rollouts, media
teacher rollout artifacts, mixture plans, and reportable eval task roots. Use
`--strict` only when the run should fail closed instead of producing a report.

The sidecar runner now emits `mixture_plan.json`, a bounded adaptive sampling
plan with native `8K -> 1M` context ladder targets, modality-gap flags, q4
recovery gates, and agentic/multimodal scheduler signals for the next 20B pass.

The sidecar script keeps the 20B target lane on fast GPUs `0,4,6` and uses CPU
plus P40s for trace collection, dataset expansion, teacher-job sharding, and
Qwen3.6 P40 rollouts. It exports agent-memory audit rows before the trace
orchestrator, collects Codex/Claude/Hermes/LM Studio traces, consumes ComfyUI
manifests as first-class multimodal trace sources, and writes
trace-orchestrator outputs to run-scoped writable directories. Memory exports
now use `limit=0` as an unlimited export and target
`data/raw/agent_memory_events_2026.jsonl` explicitly, so workstation or
AI-server PostgreSQL exports can feed the same trace gate without stale
run-scoped ambiguity. The sidecar also has a strict `local-traces` lane that
exports Codex/Claude/agent-memory rows through 2025-2026 date gating and secret
quarantine before they become agentic training rows. The sidecar now gates
required trace artifacts, refuses
synthetic-only train promotion, refreshes
agentic SFT/reward/preference/RLVR exports from each trace pass, parses typed
teacher critiques into corrected responses/tool calls/reward components, and
refreshes those exports again after Qwen3.6 teacher rollouts. New sidecar
outputs write run-scoped first; shared `weights/agentic_tool_training_2026`
promotion is opt-in with `OMNICODER_PROMOTE_SHARED_ARTIFACTS=1`.
The sidecar also builds modality-specific teacher job JSONL for Qwen Image,
Qwen Image Edit, LTX 2.3, ACE-Step 1.5, and omni/audio teachers so image,
video, audio, music, and image-to-video distillation work is routed to the
right teacher family instead of the P40 text/tool rollout path.
Official/protected benchmark rows remain release-gate evidence only; missing
official metadata now produces `local_only` benchmark results instead of being
misreported as public leaderboard quality.

## Why This Exists

Most multimodal systems are pipelines: an LLM delegates to an image model, an
audio model, a video model, a detector, a retriever, and a tool runner. That can
work, but it creates brittle handoffs and makes edge deployment painful.
OmniCoder explores a different direction:

- Use a shared reasoning core across modalities.
- Prefer a dense 20B-class core with depth-biased layers, fake-quant/turboquant
  training paths, and pipeline placement across the fast GPUs.
- Keep long context bounded with compressed memory, retrieval, and KV policies.
- Make text, code, vision, video, audio, and action heads trainable together.
- Export pieces to realistic local runtimes such as ONNX Runtime, Core ML,
  NNAPI-oriented runners, DirectML, ExecuTorch, llama.cpp/GGUF, and mobile app
  bundles.

The practical bet is that a compact dense omnimodal model with good data,
memory, verification, and device-aware exports can be more useful than a
collection of large disconnected models when privacy, latency, bandwidth, and
hardware budget matter.

## Current Capabilities

- Dense 20B-class transformer target with depth-biased placement, 1M native
  context metadata, fake-quant/turboquant-aware training hooks, pipeline
  sharding across RTX 3090/RTX 8000 devices, and GGUF/llama.cpp-oriented
  release contracts.
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
  modeling/          Core transformer, dense attention/memory, quant, kernels
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
