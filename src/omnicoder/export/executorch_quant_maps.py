from __future__ import annotations

"""Emit per-operator quantization maps and delegate hints for ExecuTorch NNAPI.

This sidecar JSON can be consumed by deployment tooling to attach NNAPI delegate
preferences and per-op quantization behavior (e.g., prefer QLinearMatMul, int8 Attention).
"""

import json
from pathlib import Path
from typing import Dict, Any


DEFAULT_NNAPI_MAP: Dict[str, Any] = {
    "delegate": "nnapi",
    "preferences": {
        "int8_preferred": True,
        "qlinear_matmul": True,
        "attention_fused": True,
    },
    "ops": {
        "MatMul": {"quantize": True, "per_channel": True},
        "Gemm": {"quantize": True, "per_channel": True},
        "Attention": {"quantize": True, "per_channel": False},
        "Add": {"quantize": True},
        "LayerNormalization": {"quantize": False},
    },
    "int4_hints": {
        "enable": True,
        "weight_layout": "nf4",
        "pack_bits": 4,
        "group_size": 64
    },
}


def write_nnapi_maps(out_path: str | Path, overrides: Dict[str, Any] | None = None) -> str:
    cfg = dict(DEFAULT_NNAPI_MAP)
    if overrides:
        # shallow merge
        cfg.update(overrides)
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(cfg, indent=2))
    return str(outp)


DEFAULT_COREML_MAP: Dict[str, Any] = {
    "delegate": "coreml",
    "preferences": {
        "int8_preferred": True,
        "qlinear_matmul": True,
        "attention_fused": True,
        "use_ane": True
    },
    "ops": {
        "MatMul": {"quantize": True, "per_channel": True},
        "Gemm": {"quantize": True, "per_channel": True},
        "Attention": {"quantize": True, "per_channel": False},
        "LayerNormalization": {"quantize": False}
    },
    "int4_hints": {
        "enable": True,
        "weight_layout": "nf4",
        "pack_bits": 4,
        "group_size": 64
    }
}


def write_coreml_maps(out_path: str | Path, overrides: Dict[str, Any] | None = None) -> str:
    cfg = dict(DEFAULT_COREML_MAP)
    if overrides:
        cfg.update(overrides)
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(cfg, indent=2))
    return str(outp)


DEFAULT_DML_MAP: Dict[str, Any] = {
    "delegate": "dml",
    "preferences": {
        "int8_preferred": True,
        "qlinear_matmul": True,
        "attention_fused": True
    },
    "ops": {
        "MatMul": {"quantize": True, "per_channel": True},
        "Gemm": {"quantize": True, "per_channel": True},
        "Attention": {"quantize": True, "per_channel": False},
        "LayerNormalization": {"quantize": False}
    },
    "int4_hints": {
        "enable": True,
        "weight_layout": "nf4",
        "pack_bits": 4,
        "group_size": 64
    }
}


def write_dml_maps(out_path: str | Path, overrides: Dict[str, Any] | None = None) -> str:
    cfg = dict(DEFAULT_DML_MAP)
    if overrides:
        cfg.update(overrides)
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(cfg, indent=2))
    return str(outp)

def write_nnapi_node_maps(onnx_path: str | Path, out_path: str | Path) -> str:
    """Analyze ONNX and write per-node NNAPI maps."""
    from .onnx_analyze import analyze  # type: ignore
    cfg = analyze(str(onnx_path))
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(cfg, indent=2))
    return str(outp)



