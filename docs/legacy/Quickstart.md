# Quickstart (Consolidated)

Prereqs: Python 3.10–3.12, git, optional CUDA/DirectML/MPS.

1) Create and activate a venv, install with extras:
```
python -m venv .venv
. .venv/bin/activate  # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .[onnx,vision,gen]
```

2) Copy env template and adjust as needed (persist caches, enable auto-resources):
```
cp env.example.txt .env  # PowerShell: copy env.example.txt .env
# Recommended edits in .env:
# - HF_HOME=/models/hf
# - TRANSFORMERS_CACHE=/models/hf
# - OMNICODER_AUTO_RESOURCES=1
# - EXECUTE_TESTS=true (set to false only while another test runner is active)
```

3) One-button build/export/validate:
```
press-play --out_root weights/release --quantize_onnx --quantize_onnx_per_op --onnx_preset generic --no_kd
```

4) Optional ONNX decode-step CPU smoke:
```
python -m omnicoder.inference.runtimes.onnx_mobile_infer --model weights/release/text/omnicoder_decode_step.onnx --provider CPUExecutionProvider --prompt_len 8
```

See `README.md` for detailed architecture/roadmap, and `todo/` for the prioritized backlog.
# OmniCoder Quickstart (Consolidated)

This document consolidates setup, training, export, and validation steps. It supersedes scattered quickstarts in the README.

## Environment
- Docker with NVIDIA Container Toolkit (for GPU): `docker info | Select-String -Pattern nvidia`
- Persistent model cache: mount `-v %cd%/models:/models` and set `HF_HOME=/models/hf`.

## One-button training (orchestrated)

Runs: pre-alignment (image/text InfoNCE) → KD draft student (LoRA) → VL fused short run (aux alignment) → optional RL smoke → export + provider bench.

```bash
python -m omnicoder.tools.run_training --budget_hours 1 --device cuda --out_root weights
```

To enable an additional verifier-head KD stage (for draft-and-verify), pass:

```bash
python -m omnicoder.tools.run_training --budget_hours 1 --device cuda --out_root weights \
  --run_verifier_kd --verifier_kd_steps 200
```

Env knobs (see `env.example.txt`):
- `OMNICODER_TRAIN_BUDGET_MINUTES`, `OMNICODER_TRAIN_DEVICE`, `OMNICODER_TRAIN_PRESET`, `OMNICODER_STUDENT_PRESET`, `OMNICODER_TEACHER`
- `OMNICODER_PREALIGN_DATA`, `OMNICODER_PREALIGN_EMBED_DIM`, `OMNICODER_ALIGN_WEIGHT`
- `OMNICODER_VL_JSONL`, `OMNICODER_ONNX_PRESET`

Artifacts:
- `weights/pre_align.pt`, `weights/omnicoder_draft_kd.pt`, `weights/omnicoder_vl_fused.pt`
- `weights/release/` export folder and `weights/draft_acceptance.json`

## Cross-modal verifier (mini-CLIP) for image/video

Enable rejection sampling based on prompt–image/video alignment:

```bash
python -m omnicoder.inference.multimodal_infer --task image --prompt "A scenic landscape" \
  --image_backend diffusers --sd_local_path /path/to/sd \
  --image_out weights/image_out.png \
  --cm_verifier --cm_threshold 0.65

python -m omnicoder.inference.multimodal_infer --task video --prompt "A scenic landscape in motion" \
  --video_backend diffusers --video_local_path /path/to/video_pipeline \
  --video_out weights/video_out.mp4 \
  --cm_verifier --cm_threshold 0.6
```

Notes:
- If `weights/pre_align.pt` exists, its `PreAligner` weights are used; otherwise the default lightweight alignment head is instantiated.
- Threshold is in [0,1]; lower admits more outputs.

## Reinforcement Learning Loops (GRPO/PPO)

GRPO and PPO loops with programmatic rewards (text/code/image/audio) are available. Prompts file can be JSONL with each line a JSON object like `{ "prompt": "...", "targets": ["..."], "tests": "assert ...", "image": "path/to.png", "reference": "text" }`, or simply raw text lines (treated as `{prompt}`).

Examples:

```bash
# GRPO on tiny prompts JSONL
python -m omnicoder.training.rl_grpo --prompts weights/grpo_prompts.jsonl --steps 50 --device cpu --reward text

# Code tests reward (expects `tests` field with Python asserts)
python -m omnicoder.training.rl_grpo --prompts data/code_tasks.jsonl --reward code_tests --steps 50

# PPO skeleton
python -m omnicoder.training.ppo_rl --prompts weights/grpo_prompts.jsonl --steps 50 --device cpu --reward text

# Image-text CLIPScore reward (requires open-clip)
python -m omnicoder.training.rl_grpo --prompts data/image_text.jsonl --reward clip --clip_model ViT-B-32
```

## Long-context + RAG canaries

Run windowed decode and retrieval canaries:

```bash
python -m pytest -q tests/test_long_context_qa_canaries.py::test_infinite_context_qa_recall_with_windowing_and_retrieval -q
```

## Speculative decoding draft+verify training

```bash
python -m omnicoder.training.speculative_train --teacher microsoft/phi-2 --student_preset mobile_2gb --steps 500
```

## Export to mobile

Produces decode-step ONNX, optional int8 ONNX, provider quant maps, and a bench summary.

```bash
python -m omnicoder.tools.build_mobile_release --out_root weights/release \
  --quantize_onnx --onnx_preset generic --no_kd
``;

Optional: Stable Diffusion ONNX export (requires weights/toolchains):
```bash
python -m omnicoder.tools.build_mobile_release --out_root weights/release \
  --quantize_onnx --onnx_preset generic --sd_model runwayml/stable-diffusion-v1-5 --sd_export_onnx
```

## Validate ONNX decode-step + provider bench

```bash
python -m omnicoder.inference.runtimes.onnx_decode_generate \
  --model weights/release/text/omnicoder_decode_step.onnx \
  --provider CPUExecutionProvider --prompt "Hello" --max_new_tokens 16

python -m omnicoder.inference.runtimes.provider_bench \
  --model weights/release/text/omnicoder_decode_step.onnx \
  --providers CPUExecutionProvider \
  --prompt_len 128 --gen_tokens 128 --check_fusions \
  --threshold_json profiles/provider_thresholds.json \
  --out_json weights/release/text/provider_bench.json
```

## Continuous‑latent training (image/audio)

Image latent flow/reconstruction (supports per‑patch flow matching and perceptual proxy):

```bash
python -m omnicoder.training.flow_recon \
  --data examples/vl_auto.jsonl \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --flow_loss --flow_param edm --flow_weight 1.0 --recon_weight 0.5 \
  --patch_loss --latent_dim 16 --steps 200 --device cuda \
  --fid_metrics --ref_dir examples/data/vq/images \
  --out weights/omnicoder_flow_latents.pt

# Save only the image_latent_head for later fine-tuning or export
python -m omnicoder.training.flow_recon \
  --data examples/vl_auto.jsonl --steps 100 --device cpu \
  --save_image_latent_head weights/image_latent_head.pt

# Train with TinyLatentRefiner (temporal)
python -m omnicoder.training.flow_recon \
  --data examples/vl_auto.jsonl --steps 200 --device cuda \
  --use_refiner --refiner_hidden_mult 2 --refiner_temporal \
  --export_refiner_onnx weights/latent_refiner.onnx
```

Audio latent reconstruction with mel/codec adapters and optional FAD metrics:

```bash
python -m omnicoder.training.audio_recon \
  --wav_dir examples/data/vq/audio \
  --mel --encodec --latent_dim 16 --steps 200 --device cuda \
  --recon_loss huber --fad_ref_dir examples/data/vq/audio --fad_pred_dir examples/data/vq/audio \
  --out weights/omnicoder_audio_latents.pt

# Train audio with TinyLatentRefiner and export ONNX
python -m omnicoder.training.audio_recon \
  --wav_dir examples/data/vq/audio --steps 200 --device cpu \
  --use_refiner --refiner_hidden_mult 2 --refiner_temporal \
  --export_refiner_onnx weights/audio_latent_refiner.onnx
```

Notes:
- `flow_recon` now accepts `--patch_loss` to train per‑patch epsilon targets from the VAE latent grid, and `--flow_weight/--recon_weight` to balance objectives.
- Perceptual proxy loss is dependency‑light and gated by `--recon_loss perceptual`.
- Both trainers can optionally save heads‑only state dicts via `--save_heads` for later loading.

## Retrieval write‑policy training

Train the write‑policy head from teacher marks and view metrics:

```bash
python -m omnicoder.training.write_policy_train \
  --marks data/teacher_marks.jsonl --steps 200 --device cpu \
  --out weights/omnicoder_write_head.pt
```

## Draft acceptance benchmark

```bash
python -m omnicoder.tools.bench_acceptance \
  --mobile_preset mobile_4gb --max_new_tokens 64 --verify_threshold 0.0 \
  --verifier_steps 1 --speculative_draft_len 1 --multi_token 1 \
  --draft_ckpt weights/omnicoder_draft_kd.pt --draft_preset mobile_2gb \
  --out_json weights/draft_acceptance.json
```

## Mobile budget checker

Estimate total artifact size + KV memory for a target context and assert a GB budget:

```bash
python -m omnicoder.tools.mobile_budget_check --release_root weights/release \
  --target_ctx 32768 --kvq nf4 --budget_gb 4.0 --tokens_per_second_min 15
```

## Docker GPU flows (recommended)

```bash
docker build -t omnicoder:cuda .
docker run --rm -it --gpus all \
  -v %cd%/models:/models -e HF_HOME=/models/hf omnicoder:cuda bash
```

Inside container:
```bash
python -m omnicoder.tools.run_training --budget_hours 1 --device cuda --out_root /workspace/weights
python -m omnicoder.tools.build_mobile_release --out_root /workspace/weights/release --quantize_onnx --onnx_preset generic
```

## Single-button fully automated flow

```bash
lets-gooooo --budget_hours 1 --device cuda --out_root weights
```

Notes:
- To auto-scale CPU threads and DataLoader workers based on host cores, set `OMNICODER_AUTO_RESOURCES=1` in your `.env` (see `env.example.txt`). Override workers with `OMNICODER_WORKERS` and threads with `OMNICODER_THREADS`.

## Notes
- HF caches persist under `/models` across runs.
- Provider thresholds are configurable via `profiles/provider_thresholds.json`.
- All config flags can be provided via environment variables; see `env.example.txt` for a fully documented template and code links.

# Quickstart

## Install (Windows PowerShell)

Run each line separately:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e .[onnx,vision,gen]
Copy-Item .env.example .env
```

## Run a tiny text demo

```bash
.venv\Scripts\python -m omnicoder.inference.generate --prompt "Hello, OmniCoder!" --max_new_tokens 32 --mobile_preset mobile_4gb
```

## One-button build/export/bench

```bash
.venv\Scripts\python -m omnicoder.tools.press_play --out_root weights/release --quantize_onnx --onnx_preset generic --no_kd
```

Artifacts are written under `weights/release/`. For GPU/Docker usage and provider-specific runners, see `README.md` sections.

