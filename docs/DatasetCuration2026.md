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

Expansion is fail-closed. If `license` or `license_tier` contains review,
pending, unknown, non-commercial, no-derivatives, holdout, gated, research, or
blocked markers, `dataset_expansion_2026.source_use_bucket` prevents the row
from materializing as `train` even if `use_policy` was set too aggressively.
Synthetic fallback seed rows are also demoted out of train and do not satisfy
real-family minimum gates.

The expansion runner materializes rows into the normal ledger-token training
schema and writes family, modality, license, and policy manifests:

```powershell
dataset-expansion-2026 `
  --profile profiles/dataset_curation_2026.json `
  --out-dir weights/external_datasets_2026/latest `
  --download `
  --max-records-per-dataset 1024 `
  --enforce-requirements `
  build
```

For family and modality partitions, the bare JSONL path is train-only
(`math_reasoning.jsonl`, `image.jsonl`). Explicit suffixes carry the wider
policy buckets: `_all`, `_research_internal`, `_eval_holdout`, and
`_blocked_until_review`. Aggregate files follow the same rule with
`train_all_external.jsonl`, `research_internal_all_external.jsonl`,
`eval_holdout_all_external.jsonl`, and `blocked_until_review.jsonl`.

Current high-value registry families:

- Math/reasoning: OpenR1-Math-220k, DAPO-Math-17k-Processed,
  DeepScaleR-Preview, NVIDIA OpenMathReasoning, MathNet,
  NVIDIA AceReason-Math, OpenThoughts2/3, OpenThoughts-114k, LIMO,
  DeepMath-103K, AI-MO NuminaMath 1.5, Polaris Nemotron verifiable math,
  Korean NuminaMath, RLVR Eurus review rows, AIME 2025/2026 holdouts, HLE, and
  Bespoke-Stratos quarantine rows.
- Coding/SWE/terminal/tool: NVIDIA OpenCodeReasoning-2, SWE-smith, SWE-Gym,
  SWE-smith trajectories, OpenHands SFT trajectories, DeepCoder,
  Nemotron-Terminal-Corpus, Nemotron-Terminal-Synthetic-Tasks, Toucan-1.5M,
  Hermes function calling, Hermes agent traces, ToolOmni-Data, xLAM,
  DeepResearchGym search logs, AgentTrove/WebWorldData review rows, and
  Terminal-Bench heldout metadata. The newer May 2026 registry adds
  Nemotron-SFT-SWE-v2, SWE-Hero, SWE-Zero, SWE-ZERO-12M, SWE-Fixer, R2E-Gym,
  Jupyter-Agent, OpenResearcher, WebWalkerQA, DeepSearch-2510,
  BrowseComp-Plus traces, Terminal-Bench 2.0 trajectories, ContextBench
  TraceBench, CodeTraceBench, MCP-Atlas, Nemotron RL tool-use, WebAgent-R1,
  WebShepherd, WebExplorer, DeepDive, WebArena Infinity, BrowserAgent,
  Web Agent Graph, WebChain, OSWorld 2, Magic-RICH, SWE-Dev, SWE-Next,
  DeepSWE/Kimi-K2 trajectories, SWE-Swiss repair SFT/RL, SWE-Factory-Gym,
  SWE-bench Pro/ABS/Multilingual, SWE-Lancer, SWE-PolyBench, SWE-bench Live,
  CodeElo, ICPC-Eval, GUI-360, AgentNet, Computer Use Large, Synthetic
  Computers at Scale, VideoCUA, ExeVR, AgentSynth, Computer Agent Arena,
  Smol2Operator/Aguvis, tau2/AReaL verified tool traces, APEX Agents/SWE,
  WildClawBench, ClawBench, BFCL, ComplexFuncBench, OpenHands CodeScout,
  AIDev, SWE-CI, Fixbench-RTL, SWE-Synth, NVIDIA ToolScale/When2Call/Nemotron
  Agentic/Cascade RL/SFT data, cleaned Toucan/Hermes/memory-agent/web-QA
  tool SFT rows, Qwen tool-calling, BrowserGym-style browser-agent SFT,
  Terminal-Bench 2 HF trajectories, and other MCP/function-calling corpora.
- Multimodal generation: OpenGPT-4o-Image, ShareGPT-4o-Image, Pico-Banana,
  MultiEdit, OpenSubject, VideoUFO, OpenVid-1M, CI-VID, TIP-I2V, VPData,
  Emilia-YODAS, Granary, CapSpeech, AudioSkills, JamendoMaxCaps, MusicBench,
  Music Arena, AR-Omni-Instruct, and Open-MM-RL review rows. The newer
  registry adds OmniAgent/MAgenIT, Nemotron-Image-Training-v3, PRISM/Innovator
  VL RL, RLFR-VLM, FineVision, FineVisionMax, ScaleEdit, GPT-Image-Edit,
  NHR-Edit, CrispEdit, BAGEL-World, Rapidata image preferences,
  HPDv3, ImgEdit, EditReward, UniREdit, BLIP3o, UniWorld, text-to-image DPO
  preferences, image-to-video preferences, VideoGen-RewardBench, Rapidata
  text/image-to-video preferences, JavisInst-Omni, Javis AV fine-tune, TTSDS
  listening tests, SAM Audio data, Prompt2MusicBench, OpenMMReasoner,
  DeepVision, NVIDIA AudioSkills-XL, VIBE, CompBench, ImagenWorld,
  DreamOmni2Bench, MMMU Pro, Video-MME-v2, LVOmniBench, JointAVBench, LVBench,
  PhyWorldBench, MusicEval, MCIF, Multimodal RewardBench 2, AVGen-Bench,
  VBench 2.0, PARADE_audio, AudioMC, WildSpeech-Bench, WorldSpeech,
  NonverbalTTS, Captioned AI Music Snippets, VoiceAgentBench, NVIDIA MMOU,
  QCalEval, SAGE-10K, NitroGen, BLIP3o short/60K, Rapidata frontier image and
  video preference sets, Hailuo image-to-video preferences, VideoVista CoTs,
  WorldSense, MMAU, AudioSet/Zeroshot/Adversarial audio instruction rows,
  NVIDIA HiFiTTS2/LongAudio/AF-Think/AF-Chat/MF-Skills, SpeechJudge, AudioCoT,
  StoryBench, OmniBench, and Multimodal RewardBench v1 holdouts.
- Math/reasoning second wave: R-HORIZON, Reasoning Core formal-reasoning
  environments, UniRRM-RL, Nemotron Math Proofs, UltraData-Math, GLM-5.1
  reasoning traces, MathVision, Nemotron RL super blends, Nemotron 3 Nano RL
  blends, Math-RLVR 773K, PrimeIntellect verifiable review rows, and
  High-Quality-Verifiable-Math review rows are included with train/research/eval
  gates based on license and contamination risk.
- Fourth-wave additions: Toolathlon, Agentic CoT Coding SFT,
  Plan-RewardBench, R2E verifier/testing rows, s1K-1.1, HMMT 2025, MMVU,
  OCRBench v2, MM-IQ, Real5, JoyAI OpenSpatial, OmniContext, VTC-Bench visual
  tool chains, FiVE, OmniEdit-Bench, OpenAudioBench, Ming freeform audio edit,
  Common Voice 22.0, PDMX, Song Describer, OmniDoc-TokenBench, ChartQAPro,
  OmniDoc OCR correction, OmniCorpus CC/YT, OmniGUI, X-LANCE WikiHow/WebSRC,
  APIGen-MT, PrimeIntellect SYNTHETIC-1 SFT and preference rows, WebShaper, and
  DeepGen card rows are now represented in the registry. APIGen-MT, OmniGUI,
  and WebShaper are research/internal until their non-commercial,
  share-alike, or missing-license constraints are resolved. FiVE,
  OmniEdit-Bench, OpenAudioBench, Ming audio edit, OmniDoc, ChartQAPro, and
  VTC-Bench stay in eval-holdout buckets. JoyAI SpatialEdit is blocked until
  license/split review; DeepGen card rows are research-only.
  GitHub-hosted TSV/CSV/JSON/JSONL files can now be materialized through
  `remote_files`, which lets VTC-Bench enter the eval-holdout lane as real
  rows with image refs and ground-truth visual tool trajectories.
- Fifth/sixth-wave additions: CUA-Gym, A11y-CUA, Telos, MEnvData SWE,
  JetBrains SWE-Smith trajectories, DeepSWE Kimi-K2 rejection sampling,
  ElenaFu SWE-agent rows, GELATO OSWorld, WebArena Pro trajectory reviews,
  Mind2Web UTG, MCP tool-calling, Turkish mobile function-calling,
  ScreenSpot-Pro, RLVR Linearity, Nous RLVR Coding, IFDecorator,
  OpenResearcher/OpenSeeker cleaned tool reasoning rows, BrowseComp-plus
  review rows, VideoPhy2, EditReward-Bench, IESBench, SVI, MieDB, OpenVE-3M,
  VBench-I2V, DocVQA 2026, ChartMuseum, GroundUI-18K, Kirundi/Indonesian
  speech, audio preference rows, TrueMuse, tokenized omni/Emu3 review rows,
  LiveCodeBench, KernelBench, TritonBench, TestGenEval, UTBoost,
  NuminaMath-LEAN, Kimina/Lean proof rows, FoVer, MathArena HMMT 2026/USAMO
  2025, and ARC-AGI-2 public-training seeds. These are tagged with
  `registry_wave` so the AI server can materialize only the fresh wave as a
  delta run while the full registry remains the promotion gate.
- Seventh-wave additions: OpenThoughts-Agent v1 SFT/RL, Edge-Agent WebSearch
  260K, Exgentic agent traces, CUDA-Agent-Ops, AI CUDA Engineer Archive,
  CodeX-2M Thinking, GitHub CodeReview, Open-RL, INTELLECT-2 RL,
  DeepSeek-ProverBench, MathArena model-output verifier rows, ECHO 2025,
  TRIG, MIGEBench, ReShapeBench, DeepLookEditBench, Inter-Edit, OpenS2V-5M,
  VEFX, VideoGen-Eval, VABench, OmniVideoBench, UniM, FysicsWorld, MME-Unify,
  Zero-To-CAD, ASID-1M, VisCoR, CMI-Pref, VoxEval, LongSpeech,
  UltraEval-Audio, and ATTM 2026. The wave is tagged
  `seventh_wave_agentic_math_code_omni_2026_05_24` and is designed to add more
  real agentic, math, code, image, video, audio, music, and any-to-any coverage
  without letting protected eval material enter train.
- Eighth-wave additions: MCP-Universe trajectories, MCPMark trajectory logs,
  Qwen 3.6 Plus tool-call trajectories, Qwen agent-distillation trajectories,
  Computer Use PSAI, BrowseCompLongContext, BrowseComp-Plus corpus plus QA
  holdout, PaperBench smoke, TheAgentCompany, AudioMarathon, Audio-Alpaca,
  OpenAudioBench, and VideoRewardBench. The wave is tagged
  `eighth_wave_agentic_curation_training_2026_05_24`. It expands agentic MCP,
  browser, GUI/computer-use, long-context research, audio, and video-reward
  coverage while keeping benchmark answers and successful eval trajectories out
  of training.
- Tenth-wave May 25, 2026 additions: MCPToolBench++ Preview, WebBench,
  mAIME2025, MMLongBench, NoLiMa, LongCodeBench, SagaScale, AcademicEval,
  FineWeb2, Common Pile v0.1, LEMAS, Emilia, AudioBench, MMAU-Pro, MMAR,
  CMI-Bench, MUSE, MPBench, RTV-Bench, and RIVER Bench. The wave is tagged
  `tenth_wave_curated_benchmarks_2026_05_25`. FineWeb2 and Common Pile v0.1
  are train-eligible text pretraining sources with explicit attribution/open
  license tiers. LEMAS and Emilia stay research-internal until speech media
  rights and non-commercial constraints are reviewed. The rest are eval-only
  or benchmark-holdout rows so answers, rubrics, tool traces, and media assets
  cannot leak into the 20B training lane.

`mixture_controller_2026` turns curation metadata into scheduler inputs for the
training profile. It caps synthetic ratios, includes provenance/license/quality
signals, and explicitly states that synthetic-only rows do not satisfy real-data
family minima. Eval-only rows remain held out even when their license is
permissive.

Filtered delta materialization:

```powershell
dataset-expansion-2026 `
  --profile profiles/dataset_curation_2026.json `
  --out-dir weights/external_datasets_2026/runs/fifth_sixth_wave_delta `
  --download `
  --max-records-per-dataset 512 `
  --include-wave fifth_wave_agentic_rlvr_multimodal_2026_05_24 `
  --include-wave sixth_wave_formal_code_media_2026_05_24 `
  --include-wave seventh_wave_agentic_math_code_omni_2026_05_24 `
  --include-wave eighth_wave_agentic_curation_training_2026_05_24 `
  --include-wave tenth_wave_curated_benchmarks_2026_05_25 `
  build
```

Filtered delta runs deliberately skip the global family-minimum requirement
unless `--enforce-requirements` is passed. They are sidecar inputs for fresh
teacher jobs and sampler rows; only a full unfiltered pass can promote
`weights/external_datasets_2026/latest` as the canonical 20B source.

Rows from external sources are not merged into the 20B target lane merely
because they exist. They must survive redaction, dedupe, benchmark
decontamination, license tiering, and heldout sample-loss checks.

`external_dataset_registry_2026.required_real_family_min_records` defines
family-level gates for real downloaded or local rows. Seed prompts still matter
for teacher-job generation and future source bootstrapping, but they do not
count toward the real-data minimum. The expansion manifest includes
`real_families`, `synthetic_seed_families`, and `requirement_report`; CI and the
AI-server sidecar can pass `--enforce-requirements` to reject a run that lacks
real rows for math, coding, agentic tool-use, terminal/browser agents,
image/editing, video, audio/speech/music, music, or omnimodal understanding.

Agent-memory export is PostgreSQL-first. The profile writes the raw export to
`data/raw/agent_memory_events_2026.jsonl` and treats `limit=0` as unlimited so
full Codex/Claude audit history can be collected without silently truncating at
12k rows. If the AI server lacks the `agent_memory_pg` credential, stage a fresh
workstation export into `data/raw/` before starting the next sidecar instead of
letting old run-scoped rows masquerade as current trace coverage.

Local traces are now fail-closed. `builder_2026.strict_trace_dates` and
`reject_unknown_trace_dates` are enabled by default, so Codex, Claude Code, and
agent-memory rows must resolve to 2025-2026 through a row timestamp, path date,
or file mtime. The PostgreSQL exporter uses the shared curation redactor, writes
secret-bearing rows to `data/raw/agent_memory_events_2026.quarantine.jsonl`,
and the SFT exporter rejects `secret_redaction.has_secret` again before
writing grouped conversations. The sidecar action
`scripts/ai_server_dataset_training_sidecars_2026.sh local-traces` builds a
run-scoped strict bundle at
`weights/curated_datasets_2026/runs/<run_id>_local_traces`.

The registry test suite asserts unique names, HF ids, and URLs, verifies the
expanded 2025-2026 source coverage, and checks that unsafe license markers do
not resolve to the train bucket.

`scripts/ai_server_dataset_training_sidecars_2026.sh modality-teacher-jobs`
builds modality-specific teacher job files for Qwen Image, Qwen Image Edit,
LTX 2.3, ACE-Step 1.5, and omni/audio teachers from the current curated and
external JSONL outputs. These jobs are separate from P40 Qwen text/tool
rollouts because raw image/video/audio/music generation and critique need the
matching teacher runtime.

The AI-server sidecar exports agent-memory audit rows before the trace
orchestrator runs, then collects Codex, Claude, Hermes, LM Studio, and ComfyUI
rows. Required trace artifacts are checked for nonzero JSONL rows before a
curation pass can promote outputs. ComfyUI JSONL manifests and media
directories both flow through the trace orchestrator, so generated image,
video, music, and audio artifacts can feed multimodal SFT, reward-labeling, and
teacher-job generation with provenance intact. On the AI server the trace
orchestrator writes to a run-scoped
`weights/data_factory/runs/trace_orchestrator/<run_id>` directory so stale
root-owned artifacts cannot block fresh curation passes. Curated, external,
teacher, and agentic-tool sidecar outputs are also run-scoped by default;
shared-path promotion is explicit opt-in through the sidecar promotion knobs.

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
tool trace. Pure math and pure code verifier rows now emit `math_rlvr.jsonl`
or `code_rlvr.jsonl` without entering generic `tool_sft.jsonl`,
`tool_reward.jsonl`, or `tool_preference.jsonl`.

The AI-server sidecar now runs this exporter immediately after trace curation
and again after P40/Qwen3.6 teacher rollouts. The first pass turns fresh Codex,
Claude, Hermes, LM Studio, ComfyUI, and agent-memory traces into SFT, reward,
preference, and RLVR files. The second pass concatenates contamination-scanned
trace rows with successful teacher-rollout rows so local model critiques become
actual posttraining artifacts. Outputs first land under
`weights/agentic_tool_training_2026/runs/<run_id>`; copying them to shared
`weights/agentic_tool_training_2026/*.jsonl` paths requires explicit shared
promotion opt-in.

## PostgreSQL

Apply `schemas/curation_layers_2026.sql` to enable raw PostgreSQL mirroring.
The JSONL path remains the default because it works on the workstation and AI
server without database credentials.
