from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnicoder.config_2026 import get_omnicoder2026_preset
from omnicoder.tokenization.omni_ledger_2026 import DEFAULT_LEDGER


def build_bridge_manifest(profile: str, output_name: str, context_length: int) -> dict:
    preset = get_omnicoder2026_preset(profile)
    bridge_ctx = int(context_length)
    return {
        "schema": "omnicoder2026_gguf_bridge_manifest_v1",
        "profile": preset.name,
        "output_name": output_name,
        "bridge_architecture": preset.gguf_bridge_architecture,
        "native_architecture": preset.architecture,
        "native_context_length": int(preset.max_seq_len),
        "gguf_context_length": bridge_ctx,
        "compatibility_goal": "unmodified LM Studio / llama.cpp first-run compatibility",
        "truth": "GGUF bridge is for text/tool adoption and shorter context; native 1M KDA/CSA/HCA requires Omnicoder runtime support",
        "recommended_llama_cpp_runtime": {
            "weight_quant": "Q4_K_M or mixed Q4/Q5",
            "cache_type_k": "q8_0 for coding/tool quality",
            "cache_type_v": "q4_0 or q8_0 after eval",
            "flags": ["--flash-attn", "--cache-type-k q8_0", "--cache-type-v q4_0", "--kv-offload"],
        },
        "gguf_metadata": {
            "general.architecture": "qwen3",
            "general.name": output_name,
            "llama.context_length": bridge_ctx,
            "tokenizer.chat_template": "qwen/chatml-compatible bridge template",
            "omnicoder.native_architecture": preset.architecture,
            "omnicoder.native_context_length": int(preset.max_seq_len),
            "omnicoder.ledger_schema": DEFAULT_LEDGER.as_metadata()["schema"],
            "omnicoder.full_native_1m_supported_by_stock_llama_cpp": False,
        },
        "ledger": DEFAULT_LEDGER.as_metadata(),
        "blocked_claims": [
            "stock GGUF provides full native 1M context for KDA/CSA/HCA",
            "custom TurboQuant/OScaR KV state is portable in LM Studio without runtime support",
            "media generation works without external codec/rendering tools",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Create Omnicoder 2026 GGUF bridge manifest")
    ap.add_argument("--profile", default="omnicoder2026_20b_1m")
    ap.add_argument("--output_name", default="omnicoder2026-qwen3-bridge")
    ap.add_argument("--context_length", type=int, default=131072)
    ap.add_argument("--out", default="weights/omnicoder2026_gguf_bridge_manifest.json")
    args = ap.parse_args()

    manifest = build_bridge_manifest(args.profile, args.output_name, args.context_length)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "out": args.out, "context_length": int(args.context_length)}))


if __name__ == "__main__":
    main()
