# Omnicoder 2026 Distillation And RL Stack

This layer turns curated Codex, Claude Code, agent-memory, ComfyUI, and media
records into teacher jobs and post-training recipes for the dense omnimodal
student. It is JSONL-first and raw-PostgreSQL-compatible.

## Teachers

The active teacher registry lives in
`profiles/distillation_curriculum_2026.json`. It covers:

- `qwen3.6_27b_q4_local` for local reasoning, coding, tool repair, trace
  critique, reward labels, and verifier labels.
- `qwen3_omni_optional` for cross-modal audio/video/image/text alignment.
- `qwen_image_generate` and `qwen_image_edit` through ComfyUI.
- `ltx_2_3` for text-to-video, image-to-video, shot plans, and temporal
  reward labels.
- `ace_step_1_5` for music, lyrics, audio critique, and music rewards.
- Optional frontier/external teachers: DeepSeek V4, MiniMax 2.7, Kimi K2.6,
  Composer 2.5, Gemini Omni, Gemma 4, Grok, and GPT Image 2.

Endpoint values are environment-variable references only. No secrets or
hardcoded credentials are stored in the repo.

## Algorithms

The intended post-training path is:

1. SFT/QLoRA cold start for chat, tool-call, trace, and modal format.
2. Multi-teacher critique distillation for corrected outputs and verifier
   labels.
3. Reward modeling for outcome rewards, process rewards, tool-state rewards,
   and image/video/audio/music artifact rewards.
4. Preference optimization with DPO, ORPO, KTO, SimPO, and IPO-style variants
   depending on available labels.
5. RLVR with GRPO/DAPO-style sampling, Tree-GRPO for agents, and PPO/RLOO
   fallback when value-model infrastructure exists.
6. On-policy multimodal distillation for image/video/audio generation to reduce
   teacher-forced versus rollout mismatch.
7. Long-context retention and q4-aware recovery distillation.

## On-Policy Environment RL

`profiles/distillation_curriculum_2026.json` now includes
`on_policy_distillation_2026`. The new route prioritizes teacher-validated
student/teacher disagreement, reverse-KL token distillation, q4 consistency,
tool repair, long-context compression, audio-video sync, image-edit
preservation, and MCP state recovery.

The posttraining sequence also adds `mcp_environment_rl`, an
Agent-Lightning-style async GRPO lane over MCP, browser, terminal, raw
PostgreSQL, and ComfyUI environments. Successful eval trajectories stay
quarantined; poisoned tool metadata, credential requests, and hidden-answer
leakage become negative replay rows.

## Commands

```powershell
python -m omnicoder.training.distillation_curriculum_2026 validate `
  --profile profiles/distillation_curriculum_2026.json

python -m omnicoder.training.distillation_curriculum_2026 all `
  --profile profiles/distillation_curriculum_2026.json `
  --records weights/data_factory/trace_orchestrator_2026/jsonl/contamination_scanned.jsonl `
  --out-dir weights/distillation_2026

python -m omnicoder.training.posttrain_bridge_2026 `
  --algorithm grpo `
  --model weights/training_orchestration_2026/checkpoints/08_long_context.pt `
  --train_jsonl weights/data_factory/trace_orchestrator_2026/exports/sft_traces.jsonl `
  --out_dir weights/posttrain_2026/grpo `
  --device cuda:0 `
  --max_steps 32 `
  --max_records 512
```

Console scripts are also registered:

- `distill-curriculum-2026`
- `posttrain-bridge-2026`

`posttrain_bridge_2026` is no longer only a manifest validator. Without
`--dry_run` or `--smoke`, it launches the native `reward_replay_2026` optimizer,
writes a trained checkpoint under `--out_dir`, records the loss JSONL, and marks
the manifest `live_training_passed` or `live_training_failed`. Pipeline-sharded
checkpoints use `--defer_optimizer` so the training orchestrator can hand the
same dataset and checkpoint to the distributed pipeline replay path.

Pipeline posttraining is now guarded against stale-checkpoint continuation.
When a live sharded replay stage returns nonzero or fails to write a complete
multi-rank checkpoint, the orchestrator marks that stage failed and skips the
remaining posttraining algorithms instead of resuming them from the previous
successful checkpoint. The training profile also enables posttraining shard
retention with `keep_last_successful: 2` and `delete_incomplete: true`, so long
offline/online replay stacks do not exhaust the AI-server training volume.

## Research Notes

The design follows 2025-2026 post-training trends: online data weighting,
multi-teacher distillation, preference optimization before RL, RL with
verifiable rewards for agents, on-policy distillation for video/multimodal
tasks, and contamination/security filters around tool traces.

Primary references used during design:

- ADAPT online data curation: https://arxiv.org/abs/2605.05227
- Data Mixing Agent: https://arxiv.org/abs/2507.15640
- Demystifying Synthetic Data: https://arxiv.org/abs/2510.01631
- Video-OPD: https://arxiv.org/abs/2602.02994
- TOUCAN tool-agent trajectories: https://arxiv.org/abs/2510.01179
- PROV-AGENT provenance: https://arxiv.org/abs/2508.02866
- MCPTox tool poisoning: https://arxiv.org/abs/2508.14925
- Hugging Face TRL docs: https://huggingface.co/docs/trl/
