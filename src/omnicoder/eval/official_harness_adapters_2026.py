from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "omnicoder.official_harness_adapters_2026.v1"


OFFICIAL_HARNESS_ADAPTERS: dict[str, dict[str, Any]] = {
    "lm_eval": {
        "name": "lm-evaluation-harness",
        "claim_scope": "official_harness",
        "axis": ["reasoning", "math", "language"],
        "entry_point": "python -m lm_eval",
        "python_modules": ["lm_eval"],
        "package_refs": ["lm-eval"],
        "benchmarks": ["HellaSwag", "ARC", "MMLU", "GSM8K", "IFEval"],
        "requires_official_snapshot": True,
        "heavy_run": True,
    },
    "helm": {
        "name": "HELM",
        "claim_scope": "official_harness",
        "axis": ["reasoning", "language", "safety", "calibration"],
        "entry_point": "helm-run",
        "python_modules": ["helm"],
        "package_refs": ["crfm-helm"],
        "benchmarks": ["HELM scenarios/operators"],
        "requires_official_snapshot": True,
        "heavy_run": True,
    },
    "swe_bench": {
        "name": "SWE-bench harness",
        "claim_scope": "official_harness",
        "axis": ["coding", "agentic_repo_repair"],
        "entry_point": "python -m swebench.harness.run_evaluation",
        "python_modules": ["swebench"],
        "package_refs": ["swebench"],
        "benchmarks": ["SWE-bench Verified", "SWE-bench Live", "SWE-bench Multimodal"],
        "requires_official_snapshot": True,
        "heavy_run": True,
    },
    "terminal_bench": {
        "name": "Terminal-Bench",
        "claim_scope": "official_harness",
        "axis": ["agent_tool", "terminal"],
        "entry_point": "tb run",
        "python_modules": ["terminal_bench"],
        "package_refs": ["terminal-bench"],
        "benchmarks": ["Terminal-Bench"],
        "requires_official_snapshot": True,
        "heavy_run": True,
    },
    "bfcl": {
        "name": "Berkeley Function Calling Leaderboard",
        "claim_scope": "official_harness",
        "axis": ["agent_tool", "function_calling"],
        "entry_point": "python -m bfcl_eval",
        "python_modules": ["bfcl_eval"],
        "package_refs": ["bfcl_eval"],
        "benchmarks": ["BFCL"],
        "requires_official_snapshot": True,
        "heavy_run": True,
    },
    "mmmu": {
        "name": "MMMU/MMMU-Pro",
        "claim_scope": "official_harness",
        "axis": ["multimodal_understanding", "vision_reasoning"],
        "entry_point": "python -m mmmu_eval",
        "python_modules": ["mmmu_eval"],
        "package_refs": ["MMMU official evaluation"],
        "benchmarks": ["MMMU", "MMMU-Pro"],
        "requires_official_snapshot": True,
        "heavy_run": True,
    },
    "ruler": {
        "name": "NVIDIA RULER",
        "claim_scope": "official_harness",
        "axis": ["long_context"],
        "entry_point": "python -m ruler",
        "python_modules": ["ruler"],
        "package_refs": ["RULER official evaluation"],
        "benchmarks": ["RULER"],
        "requires_official_snapshot": True,
        "heavy_run": True,
    },
    "longbench": {
        "name": "LongBench/LongBench v2",
        "claim_scope": "official_harness",
        "axis": ["long_context"],
        "entry_point": "python -m longbench",
        "python_modules": ["longbench"],
        "package_refs": ["LongBench official evaluation"],
        "benchmarks": ["LongBench", "LongBench v2"],
        "requires_official_snapshot": True,
        "heavy_run": True,
    },
    "nolima": {
        "name": "NoLiMa",
        "claim_scope": "official_harness",
        "axis": ["long_context", "needle_retrieval"],
        "entry_point": "python -m nolima",
        "python_modules": ["nolima"],
        "package_refs": ["NoLiMa official evaluation"],
        "benchmarks": ["NoLiMa"],
        "requires_official_snapshot": True,
        "heavy_run": True,
    },
    "vbench": {
        "name": "VBench",
        "claim_scope": "official_harness",
        "axis": ["video_generation"],
        "entry_point": "python -m vbench",
        "python_modules": ["vbench"],
        "package_refs": ["vbench"],
        "benchmarks": ["VBench", "VBench-2.0", "VBench I2V"],
        "requires_official_snapshot": True,
        "heavy_run": True,
    },
    "fad": {
        "name": "Frechet Audio Distance",
        "claim_scope": "official_metric_unavailable",
        "axis": ["audio_generation", "music_generation"],
        "entry_point": "",
        "python_modules": ["frechet_audio_distance"],
        "package_refs": ["operator-provided official FAD implementation"],
        "benchmarks": ["FAD"],
        "requires_official_snapshot": True,
        "heavy_run": True,
        "placeholder_until_installed": True,
    },
    "clap": {
        "name": "CLAPScore",
        "claim_scope": "official_metric_unavailable",
        "axis": ["audio_generation", "music_generation", "audio_text_alignment"],
        "entry_point": "",
        "python_modules": ["laion_clap"],
        "package_refs": ["operator-provided CLAPScore implementation"],
        "benchmarks": ["CLAPScore"],
        "requires_official_snapshot": True,
        "heavy_run": True,
        "placeholder_until_installed": True,
    },
    "mos": {
        "name": "MOS/MOSNet-style speech quality",
        "claim_scope": "official_metric_unavailable",
        "axis": ["tts", "speech_generation"],
        "entry_point": "",
        "python_modules": ["mosnet"],
        "package_refs": ["operator-provided MOS/MOSNet implementation or human MOS protocol"],
        "benchmarks": ["MOS", "MOSNet", "human MOS panel"],
        "requires_official_snapshot": True,
        "heavy_run": True,
        "placeholder_until_installed": True,
    },
}


SCORER_ALIASES = {
    "arc-agi3-official-scorer-2026": "lm_eval",
    "hellaswag-official-eval-2026": "lm_eval",
    "helm-official-eval-2026": "helm",
    "swe-bench-live-official-eval-2026": "swe_bench",
    "swe-bench-official-eval-2026": "swe_bench",
    "terminal-bench-official-eval-2026": "terminal_bench",
    "bfcl-official-eval-2026": "bfcl",
    "mmmu-pro-official-eval-2026": "mmmu",
    "mmmu-official-eval-2026": "mmmu",
    "ruler-official-eval-2026": "ruler",
    "longbench-official-eval-2026": "longbench",
    "nolima-official-eval-2026": "nolima",
    "vbench-official-eval-2026": "vbench",
    "fad-official-eval-2026": "fad",
    "clapscore-official-eval-2026": "clap",
    "mos-official-eval-2026": "mos",
}


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def adapter_status(adapter: dict[str, Any]) -> dict[str, Any]:
    modules = [str(item) for item in adapter.get("python_modules") or [] if item]
    missing = [module for module in modules if not module_available(module)]
    placeholder = bool(adapter.get("placeholder_until_installed"))
    if missing:
        status = "unavailable_until_official_package_installed" if placeholder else "missing_package"
    else:
        status = "available_not_executed"
    return {
        "status": status,
        "available": not missing and not placeholder,
        "official_score": False,
        "diagnostic_only": False,
        "missing_python_modules": missing,
        "note": "adapter manifest only; heavy benchmark execution is a separate operator action",
    }


def manifest_row(adapter_id: str, adapter: dict[str, Any]) -> dict[str, Any]:
    status = adapter_status(adapter)
    return {
        "schema": SCHEMA_VERSION,
        "adapter_id": adapter_id,
        "name": adapter["name"],
        "claim_scope": adapter["claim_scope"],
        "axis": adapter["axis"],
        "entry_point": adapter.get("entry_point", ""),
        "python_modules": adapter.get("python_modules", []),
        "package_refs": adapter.get("package_refs", []),
        "benchmarks": adapter.get("benchmarks", []),
        "requires_official_snapshot": bool(adapter.get("requires_official_snapshot", True)),
        "heavy_run": bool(adapter.get("heavy_run", True)),
        "placeholder_until_installed": bool(adapter.get("placeholder_until_installed", False)),
        "official_score": status["official_score"],
        "diagnostic_only": status["diagnostic_only"],
        "availability": status,
    }


def build_manifest() -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "official_scores_require": [
            "official harness package installed",
            "authorized or official task snapshot manifest",
            "model-generated predictions or artifacts",
            "official scorer output artifact",
            "scorer version and task snapshot hash",
        ],
        "diagnostic_policy": "custom canaries, smokes, proxy metrics, and internal contract scorers are never official scores",
        "adapters": [manifest_row(adapter_id, adapter) for adapter_id, adapter in OFFICIAL_HARNESS_ADAPTERS.items()],
        "scorer_aliases": SCORER_ALIASES,
    }


def write_manifest(path: Path) -> dict[str, Any]:
    manifest = build_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Omnicoder 2026 official benchmark adapter registry")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="Print adapter manifest JSON")
    check = sub.add_parser("check", help="Check one adapter by id")
    check.add_argument("adapter_id")
    write = sub.add_parser("write-manifest", help="Write adapter manifest JSON")
    write.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.cmd == "list":
        print(json.dumps(build_manifest(), indent=2, sort_keys=True))
        return 0
    if args.cmd == "check":
        adapter = OFFICIAL_HARNESS_ADAPTERS.get(args.adapter_id)
        if not adapter:
            print(json.dumps({"status": "unknown_adapter", "adapter_id": args.adapter_id}, sort_keys=True))
            return 2
        print(json.dumps(manifest_row(args.adapter_id, adapter), indent=2, sort_keys=True))
        return 0
    if args.cmd == "write-manifest":
        manifest = write_manifest(Path(args.out))
        print(json.dumps({"status": "ok", "out": args.out, "adapters": len(manifest["adapters"])}, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
