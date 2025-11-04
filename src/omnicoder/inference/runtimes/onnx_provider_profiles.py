from __future__ import annotations

"""
Provider-specific ONNX Runtime session options and per-operator PTQ presets.

This module centralizes EP (Execution Provider) options and quantization op lists
for mobile runtimes: NNAPI, CoreML, and DirectML (DML). Designed to be consumed
by packagers and mobile runners.
"""

from typing import Dict, List, Tuple
import json
from pathlib import Path


def get_provider_options(provider: str, nnapi_accel: str = "", coreml_enable_ane: bool = False) -> Tuple[List[str], List[dict]]:
    p = provider.strip()
    # If a global profile is set (OMNICODER_PROVIDER_PROFILE), try to load provider/options from JSON
    try:
        import os
        prof = os.getenv("OMNICODER_PROVIDER_PROFILE", "").strip()
        if prof:
            jp = Path(prof)
            if jp.exists():
                data = json.loads(jp.read_text(encoding='utf-8'))
                prov = str(data.get("provider", p))
                prov_opts = data.get("provider_options", {})
                return [prov], [prov_opts]
    except Exception as e:
        print(f"[warn] provider profile load failed: {e}")
    if p == "NNAPIExecutionProvider":
        # Enable NNAPI partitioning and prefer fp16 where supported; allow accelerator selection
        opts = {"nnapi_accelerator_name": nnapi_accel} if nnapi_accel else {}
        # Common ORT NNAPI options (may be a no-op depending on build)
        opts.setdefault("use_fp16", True)
        opts.setdefault("partitioning", 1)  # 0 disabled, 1 partition
        return ["NNAPIExecutionProvider"], [opts]
    if p == "CoreMLExecutionProvider":
        # onnxruntime CoreML has limited options via Python; prefer defaults and request ANE
        opts = {"enable_on_subgraph": True}
        if coreml_enable_ane:
            # Some ORT builds recognize this flag; safe to include
            opts["coreml_enable_ane"] = True
        # Prefer FP16 compute for mobile
        opts.setdefault("coreml_compute_units", "ALL")
        return ["CoreMLExecutionProvider"], [opts]
    if p == "DmlExecutionProvider":
        # Configure default DML options
        opts = {}
        return ["DmlExecutionProvider"], [opts]
    # default CPU
    return ["CPUExecutionProvider"], [{}]


def get_ptq_op_types_preset(preset: str) -> List[str]:
    pr = preset.strip().lower()
    if pr == "nnapi":
        # Favor ops commonly supported by NNAPI and quant-friendly
        return ["MatMul", "Gemm", "Conv", "Attention", "Add", "Mul", "Relu", "Softmax"]
    if pr == "coreml":
        return ["MatMul", "Gemm", "Conv", "Attention", "Add", "Mul", "LayerNormalization", "Softmax"]
    if pr == "dml":
        return ["MatMul", "Gemm", "Conv", "Attention", "Add", "Mul", "LayerNormalization", "Relu", "Softmax"]
    return ["MatMul", "Gemm", "Conv", "Attention", "Add", "Mul"]



