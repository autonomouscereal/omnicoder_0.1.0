from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from omnicoder.data_factory import dataset_expansion_2026 as expansion
from omnicoder.training import training_orchestration_2026 as orch


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _training_profile(root: Path) -> dict:
    return {
        "profile_name": "unit_training",
        "modalities": {name: {"enabled": True} for name in orch.DEFAULT_STAGE_ORDER},
        "real_sources": {},
        "training_plan": {
            "max_records_per_modality": 8,
            "artifact_token_count": {"image": 4, "video": 4, "audio": 4, "music": 4, "tool": 4, "long_context": 4},
            "text_token_limit": 64,
            "target_text_chars": 512,
        },
        "learning_checks": {"min_loss_points": 2, "min_relative_loss_drop": 0.001},
    }


def test_dataset_expansion_materializes_license_tiered_rows(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    _write_json(root / "profiles" / "training_orchestration_2026.json", _training_profile(root))
    _write_jsonl(root / "data" / "math.jsonl", [{"problem": "Solve 2+2.", "answer": "4", "uuid": "m1"}])
    profile = {
        "external_dataset_registry_2026": {
            "training_profile": "profiles/training_orchestration_2026.json",
            "datasets": [
                {
                    "name": "unit_math",
                    "family": "math_reasoning",
                    "target_modality": "text",
                    "local_jsonl": "data/math.jsonl",
                    "license": "Apache-2.0",
                    "license_tier": "permissive",
                    "use_policy": "train",
                    "field_map": {"prompt": ["problem"], "target": ["answer"], "id": ["uuid"]},
                },
                {
                    "name": "unit_eval",
                    "family": "terminal_browser_agents",
                    "target_modality": "tool",
                    "license": "Apache-2.0",
                    "license_tier": "permissive_eval_holdout",
                    "use_policy": "eval_only",
                    "distillation_prompts": [{"instruction": "terminal task", "target": "eval answer"}],
                },
            ],
        }
    }
    _write_json(root / "profiles" / "dataset_curation_2026.json", profile)
    monkeypatch.setattr(expansion, "repo_root", lambda: root)
    manifest = expansion.build_expansion(
        root / "profiles" / "dataset_curation_2026.json",
        root / "weights" / "external",
        type("Args", (), {"download": False, "no_streaming": False, "max_records_per_dataset": 0})(),
    )
    assert manifest["records"]["train"] == 1
    assert manifest["records"]["eval_holdout"] == 1
    assert manifest["families"]["math_reasoning"] == 1
    train_row = json.loads((root / "weights" / "external" / "jsonl" / "train_all_external.jsonl").read_text().splitlines()[0])
    assert train_row["dataset_name"] == "unit_math"
    assert train_row["training_bucket"] == "train"
    assert train_row["license_tier"] == "permissive"


def test_dataset_expansion_falls_back_to_distillation_seeds_after_hf_failure(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    _write_json(root / "profiles" / "training_orchestration_2026.json", _training_profile(root))
    profile = {
        "external_dataset_registry_2026": {
            "training_profile": "profiles/training_orchestration_2026.json",
            "datasets": [
                {
                    "name": "seeded_image",
                    "hf_id": "missing/not-real",
                    "family": "image_generation_editing",
                    "target_modality": "image",
                    "license_tier": "permissive_with_teacher_provenance",
                    "use_policy": "research_internal",
                    "distillation_prompts": [{"instruction": "image prompt decomposition", "target": "image rubric"}],
                }
            ],
        }
    }
    _write_json(root / "profiles" / "dataset_curation_2026.json", profile)
    monkeypatch.setattr(expansion, "repo_root", lambda: root)
    monkeypatch.setattr(expansion, "rows_from_huggingface", lambda entry, limit, streaming: ([], {"status": "failed"}))
    manifest = expansion.build_expansion(
        root / "profiles" / "dataset_curation_2026.json",
        root / "weights" / "external",
        type("Args", (), {"download": True, "no_streaming": False, "max_records_per_dataset": 0})(),
    )
    assert manifest["records"]["research_internal"] == 1
    assert manifest["modalities"]["image"] == 1
    assert manifest["datasets"][0]["synthetic_seed_only"] is True
    assert manifest["synthetic_seed_families"]["image_generation_editing"] == 1


def test_dataset_expansion_reports_required_real_family_minima(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    _write_json(root / "profiles" / "training_orchestration_2026.json", _training_profile(root))
    _write_jsonl(root / "data" / "math.jsonl", [{"problem": "Solve 1+1.", "answer": "2", "uuid": "m1"}])
    profile = {
        "external_dataset_registry_2026": {
            "training_profile": "profiles/training_orchestration_2026.json",
            "required_real_family_min_records": {
                "math_reasoning": {"bucket": "train", "min_real": 1},
                "coding_agentic": {"bucket": "train", "min_real": 1},
            },
            "datasets": [
                {
                    "name": "unit_math",
                    "family": "math_reasoning",
                    "target_modality": "text",
                    "local_jsonl": "data/math.jsonl",
                    "license": "Apache-2.0",
                    "license_tier": "permissive",
                    "use_policy": "train",
                    "field_map": {"prompt": ["problem"], "target": ["answer"], "id": ["uuid"]},
                },
                {
                    "name": "seeded_code",
                    "family": "coding_agentic",
                    "target_modality": "code",
                    "license": "internal",
                    "license_tier": "internal_seed",
                    "use_policy": "train",
                    "distillation_prompts": [{"instruction": "fix bug", "target": "patch"}],
                },
            ],
        }
    }
    _write_json(root / "profiles" / "dataset_curation_2026.json", profile)
    monkeypatch.setattr(expansion, "repo_root", lambda: root)
    manifest = expansion.build_expansion(
        root / "profiles" / "dataset_curation_2026.json",
        root / "weights" / "external",
        type("Args", (), {"download": False, "no_streaming": False, "max_records_per_dataset": 0, "enforce_requirements": False})(),
    )

    assert manifest["status"] == "failed_requirements"
    assert manifest["real_families"]["math_reasoning"] == 1
    assert manifest["synthetic_seed_families"]["coding_agentic"] == 1
    assert manifest["records"]["train"] == 1
    assert manifest["records"]["research_internal"] == 1
    report = manifest["requirement_report"]
    assert report["requirements"]["math_reasoning"]["status"] == "passed"
    assert report["requirements"]["coding_agentic"]["status"] == "failed"


def test_huggingface_iteration_errors_do_not_abort_expansion(monkeypatch) -> None:
    class BrokenDataset:
        def __iter__(self):
            raise ImportError("missing optional decoder")

    def fake_load_dataset(*args, **kwargs):
        return BrokenDataset()

    module = types.ModuleType("datasets")
    module.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    rows, status = expansion.rows_from_huggingface(
        {"hf_id": "unit/broken", "splits": ["train"]},
        limit=4,
        streaming=True,
    )

    assert rows == []
    assert status["status"] == "failed"
    assert "iteration failed" in status["errors"][0]


def test_huggingface_registry_options_are_passed_without_token_leak(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_load_dataset(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return [{"prompt": "p", "target": "t"}]

    module = types.ModuleType("datasets")
    module.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    monkeypatch.setenv("HF_UNIT_TOKEN", "secret-token")

    rows, status = expansion.rows_from_huggingface(
        {
            "hf_id": "unit/options",
            "config": "cfg",
            "revision": "abc123",
            "data_files": {"train": "data/*.jsonl"},
            "verification_mode": "no_checks",
            "trust_remote_code": False,
            "token_env": "HF_UNIT_TOKEN",
            "splits": ["train"],
        },
        limit=1,
        streaming=True,
    )

    assert len(rows) == 1
    assert calls[0]["args"][:2] == ("unit/options", "cfg")
    assert calls[0]["kwargs"]["revision"] == "abc123"
    assert calls[0]["kwargs"]["data_files"] == {"train": "data/*.jsonl"}
    assert calls[0]["kwargs"]["verification_mode"] == "no_checks"
    assert calls[0]["kwargs"]["trust_remote_code"] is False
    assert calls[0]["kwargs"]["token"] == "secret-token"
    assert status["token_env"] == "HF_UNIT_TOKEN"
    assert status["token_used"] is True
    assert "secret-token" not in json.dumps(status)


def test_repo_dataset_registry_covers_new_agentic_and_multimodal_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    names = [entry["name"] for entry in entries]
    assert len(names) == len(set(names))
    for key in ("hf_id", "url"):
        values = [entry[key] for entry in entries if entry.get(key)]
        assert len(values) == len(set(values))
    by_name = {entry["name"]: entry for entry in entries}

    for name in [
        "OpenThoughts2-1M",
        "DeepMath-103K",
        "AI-MO NuminaMath 1.5",
        "Polaris Nemotron Easy Math Verifiable",
        "Polaris Nemotron Medium Math Verifiable",
        "Korean NuminaMath Verifiable 540K",
        "RLVR Eurus 2 Math Fixed",
        "DeepCoder-Preview-Dataset",
        "SWE-Dev Train",
        "Dorothy SWE-Dev",
        "DeepSWE Agent Kimi K2 Trajectories 2.8K",
        "SWE-Factory-Gym",
        "SWE-Next",
        "SWE-Next SFT Trajectories",
        "SWE-Swiss SFT Repair 4K",
        "SWE-Swiss Repair RL 12K",
        "SWE-Universe Repaired Bug Pilot Trajectories",
        "NVIDIA Nemotron-RL Agentic SWE Pivot",
        "Scale-SWE",
        "Nebius SWE-rebench OpenHands Trajectories",
        "NVIDIA OpenCodeReasoning-2",
        "MCP-Atlas",
        "NVIDIA Nemotron-RL Agentic Conversational Tool Use Pivot v1",
        "WebAgent-R1 Distill",
        "WebShepherd PRM Collection",
        "WebExplorer-QA",
        "DeepDive",
        "WebArena Infinity Trajectories",
        "BrowserAgent SeedData",
        "BrowserAgent Data",
        "Web Agent Graph Dataset",
        "WebChain",
        "WebArena Pro Task Intents and Rubrics",
        "Deceptive Web Execution-Time Warnings",
        "CognitiveKernel Pro SFT",
        "NVIDIA Nemotron-Terminal-Corpus",
        "TerminalWorld",
        "OSWorld 2 Ubuntu Trajectories",
        "OSWorld Control Trajectories",
        "Magic-RICH",
        "macOS GUI Agent Train",
        "Multi-Docker-Eval",
        "SWE-bench Pro",
        "SWE-bench Pro ABS",
        "SWE-QA-Pro-Bench",
        "SWE-bench Multilingual",
        "CodeElo",
        "ICPC-Eval",
        "HLE-Verified",
        "MathArena AIME 2026 Holdout",
        "Multi-SWE-bench",
        "SWE-Bench++",
        "EVMbench",
        "SWE-bench Multimodal",
        "SWE-Lancer",
        "SWE-PolyBench",
        "SWE-bench Live MultiLang",
        "SWE-bench Live Windows",
        "JetBrains SWE-bench Agent Trajectories",
        "Hermes Function Calling V1",
        "Tool Use Multiturn Reasoning",
        "OmniEdit-Filtered-1.2M",
        "OmniGen2 X2I2",
        "Common Voice 21.0",
        "ACE-Step Songs",
        "Visual-CoT",
        "MultiEdit",
        "NVIDIA Granary",
        "AR-Omni-Instruct-v0.1",
        "DIVER Training OpenR1 Math 46k",
        "VLM-CapCurriculum TextReasoning",
        "Frugal Thinking RL Data",
        "FleetAI Tool-Use Difficult Envs",
        "FleetAI Thinking Tools Difficult Envs",
        "ATBench Agent Trajectory Safety",
        "InternVL-U ScaleEdit-12M",
        "BLIP3o NEXT Edit Ensemble",
        "Complex Long Video Understanding Reasoning",
        "NVIDIA Nemotron-SFT-SWE-v2",
        "NVIDIA SWE-Hero OpenHands Trajectories",
        "NVIDIA Nemotron-SFT Competitive Programming v2",
        "OpenResearcher Dataset",
        "OpenResearcher Corpus",
        "WebWalkerQA",
        "Text-to-Terminal v2 Tool Reasoning Cleaned",
        "NuminaMath 1.5 RL Verifiable",
        "DeepSearch-2510",
        "SWE-bench-Live OS-bench",
        "ContextBench TraceBench",
        "NJU CodeTraceBench",
        "Terminal-Bench 2.0 Trajectories",
        "OmniAgent MAgenIT Data",
        "NVIDIA Nemotron Image Training v3",
        "PRISM VLM RL Dataset",
        "RLFR Dataset VLM",
        "Innovator VL RL 172K",
        "NVIDIA AudioSkills XL",
        "Pico-Banana-400K",
        "Pico-Banana-400K Apple Research Reference",
        "FineVision",
        "FineVisionMax",
        "GPT-Image-Edit-1.5M",
        "NHR-Edit",
        "CrispEdit-2M",
        "BAGEL-World",
        "ImgEdit 1.2M",
        "Rapidata Open Image Preferences v1 More Results",
        "Text-to-Image DPO Human Preferences Full",
        "VIBE Benchmark",
        "CompBench Complex Editing",
        "ImagenWorld",
        "DreamOmni2Bench",
        "DeepVision-103K",
        "WorldSpeech",
        "NonverbalTTS",
        "Captioned AI Music Snippets",
        "Cambrian-P-Data",
        "MMMU Pro",
        "LongMemEval-V2",
        "AVGen-Bench",
        "TAG-Bench-Video",
        "VGenST-Bench",
        "VBench 2.0",
        "Video-Bench",
        "PARADE_audio",
        "WildSpeech-Bench",
        "AudioMC",
        "AV-SpeakerBench",
        "HLE",
        "Video-MME",
        "Video-MME-v2",
        "LVBench",
        "LVOmniBench",
        "JointAVBench",
        "PhyWorldBench",
        "Image-to-Video Human Preferences Large",
        "MusicEval",
        "MCIF Crosslingual Multimodal Instruction Following",
        "Multimodal RewardBench 2",
        "VoiceAgentBench",
    ]:
        assert name in by_name

    assert by_name["NVIDIA Nemotron-Terminal-Corpus"]["use_policy"] == "train"
    assert by_name["DeepMath-103K"]["use_policy"] == "train"
    assert by_name["AI-MO NuminaMath 1.5"]["use_policy"] == "train"
    assert by_name["Polaris Nemotron Medium Math Verifiable"]["use_policy"] == "train"
    assert by_name["Korean NuminaMath Verifiable 540K"]["use_policy"] == "train"
    assert by_name["RLVR Eurus 2 Math Fixed"]["use_policy"] == "research_internal"
    assert by_name["SWE-Dev Train"]["use_policy"] == "train"
    assert by_name["Dorothy SWE-Dev"]["use_policy"] == "train"
    assert by_name["DeepSWE Agent Kimi K2 Trajectories 2.8K"]["use_policy"] == "train"
    assert by_name["SWE-Factory-Gym"]["use_policy"] == "research_internal"
    assert by_name["SWE-Next"]["use_policy"] == "train"
    assert by_name["SWE-Next SFT Trajectories"]["use_policy"] == "train"
    assert by_name["SWE-Swiss Repair RL 12K"]["use_policy"] == "train"
    assert by_name["SWE-Universe Repaired Bug Pilot Trajectories"]["use_policy"] == "research_internal"
    assert by_name["MCP-Atlas"]["use_policy"] == "train"
    assert by_name["NVIDIA Nemotron-RL Agentic Conversational Tool Use Pivot v1"]["use_policy"] == "train"
    assert by_name["WebAgent-R1 Distill"]["use_policy"] == "train"
    assert by_name["WebShepherd PRM Collection"]["use_policy"] == "research_internal"
    assert by_name["WebExplorer-QA"]["use_policy"] == "train"
    assert by_name["DeepDive"]["use_policy"] == "train"
    assert by_name["WebArena Infinity Trajectories"]["use_policy"] == "train"
    assert by_name["BrowserAgent SeedData"]["use_policy"] == "train"
    assert by_name["BrowserAgent Data"]["use_policy"] == "research_internal"
    assert by_name["Web Agent Graph Dataset"]["use_policy"] == "train"
    assert by_name["WebChain"]["use_policy"] == "research_internal"
    assert by_name["WebArena Pro Task Intents and Rubrics"]["use_policy"] == "eval_only"
    assert by_name["Deceptive Web Execution-Time Warnings"]["use_policy"] == "eval_only"
    assert by_name["CognitiveKernel Pro SFT"]["use_policy"] == "research_internal"
    assert by_name["TerminalWorld"]["use_policy"] == "eval_only"
    assert by_name["OSWorld 2 Ubuntu Trajectories"]["use_policy"] == "research_internal"
    assert by_name["OSWorld Control Trajectories"]["use_policy"] == "eval_only"
    assert by_name["Magic-RICH"]["use_policy"] == "eval_only"
    assert by_name["Multi-Docker-Eval"]["use_policy"] == "eval_only"
    assert by_name["SWE-bench Pro"]["use_policy"] == "eval_only"
    assert by_name["SWE-bench Pro ABS"]["use_policy"] == "eval_only"
    assert by_name["SWE-QA-Pro-Bench"]["use_policy"] == "eval_only"
    assert by_name["SWE-bench Multilingual"]["use_policy"] == "eval_only"
    assert by_name["CodeElo"]["use_policy"] == "eval_only"
    assert by_name["ICPC-Eval"]["use_policy"] == "eval_only"
    assert by_name["HLE-Verified"]["use_policy"] == "eval_only"
    assert by_name["MathArena AIME 2026 Holdout"]["use_policy"] == "eval_only"
    assert by_name["Multi-SWE-bench"]["use_policy"] == "eval_only"
    assert by_name["SWE-Lancer"]["use_policy"] == "eval_only"
    assert by_name["SWE-PolyBench"]["use_policy"] == "eval_only"
    assert by_name["JetBrains SWE-bench Agent Trajectories"]["use_policy"] == "research_internal"
    assert by_name["Scale-SWE"]["use_policy"] == "train"
    assert by_name["SWE-Compass"]["use_policy"] == "benchmark_holdout"
    assert by_name["CoderForge-Preview"]["use_policy"] == "research_internal"
    assert by_name["OmniEdit-Filtered-1.2M"]["use_policy"] == "train"
    assert by_name["AR-Omni-Instruct-v0.1"]["use_policy"] == "research_internal"
    assert by_name["Open-MM-RL"]["use_policy"] == "train"
    assert by_name["RLFR Dataset VLM"]["use_policy"] == "train"
    assert by_name["ATBench Agent Trajectory Safety"]["use_policy"] == "eval_only"
    assert by_name["InternVL-U ScaleEdit-12M"]["use_policy"] == "train"
    assert by_name["GPT-Image-Edit-1.5M"]["use_policy"] == "train"
    assert by_name["NHR-Edit"]["use_policy"] == "train"
    assert by_name["CrispEdit-2M"]["use_policy"] == "train"
    assert by_name["BAGEL-World"]["use_policy"] == "train"
    assert by_name["Rapidata Open Image Preferences v1 More Results"]["use_policy"] == "train"
    assert by_name["Text-to-Image DPO Human Preferences Full"]["use_policy"] == "research_internal"
    assert by_name["ImagenWorld"]["use_policy"] == "eval_only"
    assert by_name["DreamOmni2Bench"]["use_policy"] == "eval_only"
    assert by_name["DeepVision-103K"]["use_policy"] == "research_internal"
    assert by_name["NonverbalTTS"]["use_policy"] == "research_internal"
    assert by_name["Captioned AI Music Snippets"]["use_policy"] == "research_internal"
    assert by_name["Cambrian-P-Data"]["use_policy"] == "train"
    assert by_name["FineVision"]["use_policy"] == "research_internal"
    assert by_name["WorldSpeech"]["use_policy"] == "research_internal"
    assert by_name["AVGen-Bench"]["use_policy"] == "eval_only"
    assert by_name["MMMU Pro"]["use_policy"] == "eval_only"
    assert by_name["LongMemEval-V2"]["use_policy"] == "eval_only"
    assert by_name["Video-MME-v2"]["use_policy"] == "eval_only"
    assert by_name["LVOmniBench"]["use_policy"] == "eval_only"
    assert by_name["JointAVBench"]["use_policy"] == "eval_only"
    assert by_name["Image-to-Video Human Preferences Large"]["use_policy"] == "research_internal"
    assert by_name["AudioMC"]["use_policy"] == "eval_only"
    assert by_name["HLE"]["use_policy"] == "eval_only"
    assert by_name["FineVision"]["license_tier"] == "source_license_varies"
    assert by_name["FineVisionMax"]["license_tier"] == "source_license_varies"
    assert by_name["WorldSpeech"]["license_tier"] == "non_commercial"
    assert by_name["HLE"]["license_tier"] == "benchmark_holdout_review"
    assert by_name["HLE-Verified"]["license_tier"] == "benchmark_holdout_review"
    assert by_name["TerminalWorld"]["license_tier"] == "non_commercial_eval_holdout"
    assert by_name["CodeElo"]["license_tier"] == "permissive_eval_holdout"
    assert by_name["MMMU Pro"]["license_tier"] == "permissive_eval_holdout"
    assert by_name["LVOmniBench"]["license_tier"] == "benchmark_holdout_review"
    assert by_name["JointAVBench"]["license_tier"] == "sharealike_eval_holdout"
    assert by_name["LongMemEval-V2"]["family"] == "long_context"
    assert by_name["AVGen-Bench"]["target_modality"] == "video"
    assert by_name["Video-MME-v2"]["target_modality"] == "video"
    assert by_name["PARADE_audio"]["target_modality"] == "audio"
    assert by_name["OpenGPT-4o-Image"]["use_policy"] == "train"
    assert by_name["ShareGPT-4o-Image"]["use_policy"] == "train"
    assert by_name["NVIDIA Nemotron-SFT-SWE-v2"]["use_policy"] == "train"
    assert by_name["OpenResearcher Dataset"]["use_policy"] == "research_internal"
    assert by_name["WebWalkerQA"]["use_policy"] == "eval_only"
    assert by_name["MCIF Crosslingual Multimodal Instruction Following"]["use_policy"] == "train"
    assert by_name["Terminal-Bench 2.0 Trajectories"]["use_policy"] == "research_internal"
    assert by_name["Video-MME"]["use_policy"] == "eval_only"
    assert by_name["Multimodal RewardBench 2"]["use_policy"] == "eval_only"
    assert by_name["Multimodal RewardBench 2"]["hf_id"] == "rl-research/multimodal-rewardbench-2"

    requirements = profile["external_dataset_registry_2026"]["required_real_family_min_records"]
    for family in [
        "math_reasoning",
        "coding_agentic",
        "agentic_tool_reasoning",
        "terminal_browser_agents",
        "image_generation_editing",
        "video_generation",
        "audio_music_speech",
        "music_generation",
        "omnimodal_understanding",
    ]:
        assert family in requirements


def test_registry_fail_closes_review_and_holdout_rows_from_train_bucket() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    unsafe_markers = (
        "review",
        "pending",
        "unknown",
        "non_commercial",
        "no_derivatives",
        "holdout",
        "gated",
        "research",
        "blocked",
    )

    for entry in entries:
        blob = f"{entry.get('license') or ''} {entry.get('license_tier') or ''}".lower()
        if any(marker in blob for marker in unsafe_markers):
            assert expansion.source_use_bucket(entry) != "train", entry["name"]
