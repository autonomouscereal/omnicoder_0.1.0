Profiles directory
==================

This folder contains optional configuration profiles consumed by tooling.

- provider_thresholds.json: default tokens/sec minimums for provider benches
  used by Press Play and provider_bench when available.
  - CPUExecutionProvider: 2.0
  - DmlExecutionProvider: 10.0
  - CoreMLExecutionProvider: 6.0
  - NNAPIExecutionProvider: 15.0

You can tune these per your hardware and CI expectations. Press Play will pass
this JSON automatically when present for text/vision/vqdec benches; the
standalone provider_bench also auto-loads this file if --threshold_json is not
provided.

Provider profiles

These JSON files describe suggested ONNX Runtime execution providers and options for different devices, and can be consumed by runners and the mobile packager via the `--provider_profile` flag.

Fields:
- `provider`: one of `CPUExecutionProvider`, `NNAPIExecutionProvider`, `CoreMLExecutionProvider`, `DmlExecutionProvider`
- `provider_options`: provider-specific options (e.g., `nnapi_accelerator_name`, `coreml_enable_ane`)
- `intra_op_num_threads`: integer threads hint
- `graph_optimization_level`: ORT graph opt level string
- `ptq_op_types` (optional): list of ONNX op types to quantize for per-op PTQ

Example usage:

```bash
python -m omnicoder.inference.runtimes.onnx_mobile_infer --model weights/text/omnicoder_decode_step.onnx \
  --provider_profile profiles/pixel7_nnapi.json

python -m omnicoder.export.mobile_packager --provider_profile profiles/iphone15_coreml_ane.json \
  --quantize_onnx_per_op --onnx_preset coreml
```

# Provider Profiles

Example JSON profiles for ONNX Runtime mobile providers. Load them with:

```bash
.venv\Scripts\python -m omnicoder.inference.runtimes.onnx_mobile_infer --model weights\text\omnicoder_decode_step.onnx --provider_profile profiles\pixel7_nnapi.json
```

Each profile may include keys:
- provider (e.g., "NNAPIExecutionProvider", "CoreMLExecutionProvider", "DmlExecutionProvider")
- provider_options (dict) passed to ORT session
- intra_op_num_threads (int)
- graph_optimization_level (string: ORT_DISABLE_ALL|ORT_ENABLE_BASIC|ORT_ENABLE_EXTENDED|ORT_ENABLE_ALL)


