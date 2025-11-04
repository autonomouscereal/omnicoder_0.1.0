# Bugs (triage)

- [ ] SD ONNX export numeric tolerance sometimes exceeds threshold (abs diff ~0.0031)
  - Priority: P2
  - Owner: unassigned
  - Code: `src/omnicoder/export/diffusion_export.py`
  - Notes: Prefer dynamo exporter or relax per-component tolerance.

- [ ] Provider bench CPU EP shows 0 fused Attention nodes
  - Priority: P3
  - Owner: unassigned
  - Code: `src/omnicoder/inference/runtimes/provider_bench.py`, `src/omnicoder/export/onnx_fuse.py`
  - Notes: Document CPU EP behavior; validate fusions on non-CPU providers in CI.

