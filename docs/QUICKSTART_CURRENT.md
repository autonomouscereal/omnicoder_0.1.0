# OmniCoder Current Quickstart

This quickstart is for understanding and smoke testing the repository. It is
not a full model training recipe.

## 1. Install

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
copy env.example.txt .env
```

macOS/Linux:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
cp env.example.txt .env
```

## 2. Run A CPU Smoke

```bash
python -m omnicoder.inference.generate --prompt "Hello OmniCoder" --device cpu
```

This uses the lightweight path and proves the package imports and generation
loop are wired.

## 3. Run The Development Harness

```bash
python -m omnicoder.tools.press_play --device cpu --out_root weights
```

This is the fastest way to exercise the project shape: tests, exports, provider
checks, and release-style output folders.

## 4. Try Training Plumbing

```bash
python -m omnicoder.tools.run_training --budget_hours 1 --device cuda
```

Point datasets and teachers through `profiles/datasets.json`,
`profiles/teachers.json`, or environment variables. Keep real tokens and
private model paths out of git.

## 5. Export And Bench

```bash
python -m omnicoder.export.onnx_export --out weights/release/text/omnicoder_decode_step.onnx
python -m omnicoder.inference.runtimes.provider_bench --model weights/release/text/omnicoder_decode_step.onnx --providers CPUExecutionProvider --out_json weights/release/text/provider_bench.json
```

Add DirectML/Core ML/NNAPI-oriented providers only where the host supports
them.

## 6. Useful Environment Knobs

```powershell
set OMNICODER_EXPERT_PAGING=1
set OMNICODER_EXPERT_PAGING_BUDGET_MB=256
set OMNICODER_EXPERT_PREFETCH_N=2
set OMNICODER_USE_LANDMARKS=1
set OMNICODER_MULTI_INDEX_ROOT=weights/retrieval_multi_index
```

Use `env.example.txt` as the source of truth for supported switches.

## 7. Where To Look Next

- Architecture: `docs/ARCHITECTURE_CURRENT.md`
- Tools: `src/omnicoder/tools/`
- Export: `src/omnicoder/export/`
- Runtime adapters: `src/omnicoder/inference/runtimes/`
- Multimodal modules: `src/omnicoder/modeling/multimodal/`
- Tests: `tests/`
