# OmniCoder Execution Plan (Build → Validate → Train → Export)

This document summarizes how to run the project end-to-end once the codebase is built.

## Entrypoints
- press-play: build/export/bench only (no training)
- lets-gooooo: tests → full orchestrated training (time-budgeted) → export → bench; optional export-to-phone

## Environments and caches
- Use a persistent `/models` volume for HF caches to avoid re-downloads:
  - HF_HOME=/models/hf
  - TRANSFORMERS_CACHE=/models/hf
- Outputs: `weights/` (release artifacts under `weights/release`)

## Auto resources
- Set `OMNICODER_AUTO_RESOURCES=1` to auto-scale threads/workers.
- Override with `OMNICODER_THREADS` and `OMNICODER_WORKERS` as needed.

## Single-button training
Run:
```
lets-gooooo --budget_hours <H> --device cuda --out_root weights
```
The launcher runs pytest first; aborts on failures. On resource-constrained containers where the OS may
kill long test runs without an error code, prefer running a focused subset (e.g., `pytest -k onnx -vv -rA`)
or increase container memory/timeout. Then it executes:
1) Pre-align → unified index
2) DS‑MoE pretrain (dense→sparse curriculum)
3) Draft KD (1–3B when available) + acceptance thresholds update
4) VL/VQA fused passes; AV heads
5) RL (GRPO/PPO) short loops with metrics (default on; disable via env)
6) Export decode-step (DynamicCache sidecar) and provider bench (fusion checks enabled for mobile/GPU providers; optional QLinearMatMul assert via `OMNICODER_REQUIRE_QLINEAR=1`)

## Export to phone
- Android (ADB): `python -m omnicoder.tools.export_to_phone --platform android`
- iOS (Core ML): `python -m omnicoder.tools.export_to_phone --platform ios`

## Provider thresholds and fusions
- Thresholds auto-load from `profiles/provider_thresholds.json`.
- Fusion checks are auto-enabled for DML/CoreML/NNAPI providers. Set `OMNICODER_REQUIRE_QLINEAR=1` to assert QLinearMatMul presence where applicable.
- DynamicCache models (input_ids-only) are auto-detected in the provider bench and benchmarked without explicit K/V feeds.

## Video defaults and metrics gates
- Temporal SSM is enabled by default and blends per-frame gains to stabilize sequences.
- Long-form generation defaults to segment chaining: `VideoGenPipeline` preserves a short tail of frames between calls and continues when `continue_from` is omitted.
- FVD gating can be enabled by setting `OMNICODER_ENABLE_VIDEO_METRICS=1` and pointing `OMNICODER_VIDEO_PRED_DIR` and `OMNICODER_VIDEO_REF_DIR` to existing folders; gate with `OMNICODER_MIN_FVD`.
- AV-sync loss during temporal training: enable with `OMNICODER_AV_SYNC=1`; optionally set `OMNICODER_AUDIO_DIR` and `OMNICODER_AV_WEIGHT`.

## App visualizations
Generate TPS/KV artifacts:
```
python -m omnicoder.tools.visualize_metrics \
  --bench_json weights/release/text/provider_bench.json \
  --onnx_model weights/release/text/omnicoder_decode_step.onnx \
  --out_dir weights/release/text
```
Build a simple dashboard and copy into sample apps:
```
python -m omnicoder.tools.app_assets --assets_dir weights/release/text --bench_json weights/release/text/provider_bench.json --onnx_model weights/release/text/omnicoder_decode_step.onnx
python -m omnicoder.tools.package_app_assets --assets_dir weights/release/text --android_assets app/src/main/assets/omnicoder --ios_assets SampleApp/Resources/omnicoder
```

## Activation-quant policy sidecar (optional)

You can emit a small policy sidecar to standardize activation quantization emulation at inference (mirrors PyTorch flags in the ONNX runner):

1) During/after training, write an error thresholds JSON from the pretrain loop (enabled by setting `OMNICODER_ACT_ERR_JSON=weights/act_err.json`). The trainer records a lightweight confidence-derived proxy.

2) Build the policy sidecar (adjust knobs as desired):
```
python -m omnicoder.tools.visualize_metrics \
  --release_root weights/release/text \
  --policy_out weights/release/text/omnicoder_decode_step.act_quant.json \
  --error_json weights/act_err.json \
  --conf_floor 0.3 --min_bits 8 --max_bits 2
```

3) ONNX runner auto-detects `*.act_quant.json` alongside the model and applies the min/max bits and confidence floor to drive emulated activation quant behavior (no graph changes).
