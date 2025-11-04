from __future__ import annotations

"""Analyze an ONNX graph and emit per-node NNAPI quant maps.

Scans for MatMul/Gemm/Attention nodes and builds a node-specific quantization
hint map suitable for ExecuTorch NNAPI delegate tooling.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def analyze(model_path: str) -> Dict[str, Any]:
    try:
        import onnx  # type: ignore
    except Exception as e:
        raise RuntimeError("onnx is required: pip install onnx") from e
    m = onnx.load(model_path)
    nodes = m.graph.node
    node_maps: Dict[str, Any] = {}
    for n in nodes:
        if n.op_type in ("MatMul", "Gemm", "Attention"):
            node_maps[n.name or f"{n.op_type}_{len(node_maps)}"] = {
                "op": n.op_type,
                "quantize": True,
                "per_channel": bool(n.op_type in ("MatMul", "Gemm")),
            }
    return {
        "delegate": "nnapi",
        "nodes": node_maps,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze ONNX and emit NNAPI per-node maps")
    ap.add_argument("--onnx", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights/text/nnapi_nodes.json")
    args = ap.parse_args()
    cfg = analyze(args.onnx)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()


