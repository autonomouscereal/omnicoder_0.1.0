"""Materialize real 2025-2026 benchmark task rows.

This module bridges the broad benchmark registry to concrete JSONL task roots.
It deliberately separates public/dev benchmark material from reportable
official or operator-authorized snapshots. Public answer-key data is useful for
local regression and training realignment, but it must not be silently promoted
to release-quality ARC/SWE/MMMU/etc. scores.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    import tomllib
except Exception:  # pragma: no cover - Python < 3.11 fallback path
    tomllib = None  # type: ignore[assignment]


SCHEMA = "omnicoder.benchmark_materializer_2026.v1"
TASK_SCHEMA = "omnicoder.benchmark_task_2026.v1"
DEFAULT_PROFILE = "profiles/benchmark_suite_2026.json"


KNOWN_BENCHMARKS: dict[str, dict[str, Any]] = {
    "agent_bfcl_v4_2026": {
        "source": "https://gorilla.cs.berkeley.edu/leaderboard.html",
        "git": "https://github.com/ShishirPatil/gorilla.git",
        "hf": ["gorilla-llm/Berkeley-Function-Calling-Leaderboard"],
        "kind": "tool",
        "splits": ["test", "validation", "train"],
    },
    "agent_tau3_2026": {
        "source": "https://github.com/sierra-research/tau2-bench",
        "git": "https://github.com/sierra-research/tau2-bench.git",
        "hf": ["HuggingFaceH4/tau2-bench-data"],
        "kind": "agent_tool",
        "splits": ["test", "validation", "train"],
    },
    "agent_mcp_workflows_2026": {
        "source": "https://github.com/eval-sys/mcpmark",
        "git": "https://github.com/eval-sys/mcpmark.git",
        "kind": "tool",
    },
    "agent_mcp_bench_2026": {
        "source": "https://github.com/Accenture/mcp-bench",
        "git": "https://github.com/Accenture/mcp-bench.git",
        "kind": "tool",
    },
    "agent_mcp_atlas_2026": {
        "source": "https://github.com/scaleapi/mcp-atlas",
        "git": "https://github.com/scaleapi/mcp-atlas.git",
        "hf": ["ScaleAI/MCP-Atlas"],
        "kind": "tool",
        "splits": ["train", "test", "validation"],
    },
    "agent_mcp_universe_2026": {
        "source": "https://github.com/SalesforceAIResearch/MCP-Universe",
        "git": "https://github.com/SalesforceAIResearch/MCP-Universe.git",
        "kind": "tool",
    },
    "agent_mcptoolbenchpp_2026": {
        "source": "https://huggingface.co/MCPToolBench",
        "hf": ["MCPToolBench/MCPToolBenchPP"],
        "kind": "tool",
        "splits": ["train", "test", "validation"],
    },
    "agent_livemcpbench_2026": {
        "source": "https://huggingface.co/datasets/ICIP/LiveMCPBench",
        "hf": ["ICIP/LiveMCPBench"],
        "kind": "tool",
        "splits": ["test"],
    },
    "agent_sra_bench_2026": {
        "source": "https://huggingface.co/datasets/WeihangSu/SRA-Bench",
        "hf": [{"id": "WeihangSu/SRA-Bench", "files": ["instances/*.json", "corpus/*.json"]}],
        "kind": "tool",
    },
    "agent_skillret_2026": {
        "source": "https://huggingface.co/datasets/anonymous-ed-benchmark/SKILLRET",
        "hf": [
            {"id": "anonymous-ed-benchmark/SKILLRET", "config": "queries", "splits": ["test", "train"]},
            {"id": "anonymous-ed-benchmark/SKILLRET", "config": "skills", "splits": ["test", "train"]},
        ],
        "kind": "tool",
        "splits": ["test", "train"],
    },
    "agent_webbench_2026": {
        "source": "https://webbench.ai/",
        "kind": "browser",
    },
    "agent_clawbench_browser_2026": {
        "source": "https://github.com/openclaw/clawbench",
        "git": "https://github.com/openclaw/clawbench.git",
        "hf": ["TIGER-Lab/ClawBench", "NAIL-Group/ClawBench"],
        "kind": "browser",
        "splits": ["test", "validation", "train"],
    },
    "agent_terminal_bench_2026": {
        "source": "https://github.com/harbor-framework/terminal-bench-2",
        "git": "https://github.com/harbor-framework/terminal-bench-2.git",
        "hf": ["harborframework/terminal-bench-2.0"],
        "kind": "terminal",
    },
    "agent_terminal_bench_2_1_2026": {
        "source": "https://github.com/harbor-framework/terminal-bench-2-1",
        "git": "https://github.com/harbor-framework/terminal-bench-2-1.git",
        "kind": "terminal",
    },
    "agent_browsergym_webarena_verified_2026": {
        "source": "https://github.com/ServiceNow/BrowserGym",
        "git": "https://github.com/ServiceNow/BrowserGym.git",
        "kind": "browser",
    },
    "agent_osworld_desktop_2026": {
        "source": "https://os-world.github.io/",
        "git": "https://github.com/xlang-ai/OSWorld.git",
        "kind": "desktop",
    },
    "agent_browsecomp_2026": {
        "source": "https://openai.com/index/browsecomp/",
        "kind": "research",
    },
    "agent_browsecomp_long_context_2026": {
        "source": "https://huggingface.co/datasets/openai/BrowseCompLongContext",
        "hf": ["openai/BrowseCompLongContext"],
        "kind": "research",
        "splits": ["test", "validation"],
    },
    "agent_theagentcompany_enterprise_2026": {
        "source": "https://github.com/TheAgentCompany/TheAgentCompany",
        "git": "https://github.com/TheAgentCompany/TheAgentCompany.git",
        "kind": "agent_tool",
    },
    "agent_paperbench_2026": {
        "source": "https://github.com/openai/preparedness/tree/main/project/paperbench",
        "git": "https://github.com/openai/preparedness.git",
        "kind": "research",
    },
    "agent_gdpval_2026": {
        "source": "https://huggingface.co/datasets/openai/gdpval",
        "hf": ["openai/gdpval"],
        "kind": "agent_tool",
        "splits": ["test", "validation", "train"],
    },
    "agent_agencybench_2026": {
        "source": "https://huggingface.co/datasets/GAIR/AgencyBench",
        "git": "https://github.com/GAIR-NLP/AgencyBench.git",
        "hf": [{"id": "GAIR/AgencyBench", "splits": ["train", "test", "validation"]}],
        "kind": "agent_tool",
        "splits": ["train", "test", "validation"],
    },
    "agent_locobench_agent_2026": {
        "source": "https://github.com/SalesforceAIResearch/LoCoBench-Agent",
        "git": "https://github.com/SalesforceAIResearch/LoCoBench-Agent.git",
        "hf": [{"id": "jasonqiu/LoCoBench", "splits": ["train", "test", "validation"]}],
        "kind": "agent_tool",
        "splits": ["train", "test", "validation"],
    },
    "agent_state_bench_2026": {
        "source": "https://github.com/microsoft/STATE-Bench",
        "git": "https://github.com/microsoft/STATE-Bench.git",
        "kind": "agent_tool",
    },
    "coding_swe_bench_live_2026": {
        "source": "https://huggingface.co/datasets/SWE-bench-Live/SWE-bench-Live",
        "git": "https://github.com/microsoft/SWE-bench-Live.git",
        "hf": ["SWE-bench-Live/SWE-bench-Live"],
        "kind": "swe",
        "splits": ["test", "train"],
    },
    "coding_swe_bench_pro_2026": {
        "source": "https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro",
        "git": "https://github.com/scaleapi/SWE-bench_Pro-os.git",
        "hf": ["ScaleAI/SWE-bench_Pro"],
        "kind": "swe",
        "splits": ["test", "train"],
    },
    "coding_livecodebench_2026": {
        "source": "https://github.com/LiveCodeBench/LiveCodeBench",
        "git": "https://github.com/LiveCodeBench/LiveCodeBench.git",
        "hf": ["livecodebench/code_generation_lite", "livecodebench/execution-v2"],
        "kind": "coding",
        "splits": ["test", "validation"],
    },
    "coding_swe_lancer_2026": {
        "source": "https://github.com/openai/SWELancer-Benchmark",
        "git": "https://github.com/openai/SWELancer-Benchmark.git",
        "kind": "swe",
    },
    "coding_swe_rebench_v2_2026": {
        "source": "https://huggingface.co/datasets/nebius/SWE-rebench-V2",
        "hf": ["nebius/SWE-rebench-V2"],
        "kind": "swe",
        "splits": ["train"],
    },
    "coding_swe_mera_2026": {
        "source": "https://huggingface.co/datasets/MERA-evaluation/SWE-MERA",
        "hf": [{"id": "MERA-evaluation/SWE-MERA", "splits": ["dev", "lite", "full", "multilang"]}],
        "kind": "swe",
        "splits": ["dev", "lite", "full", "multilang"],
    },
    "coding_ale_bench_2026": {
        "source": "https://github.com/SakanaAI/ALE-Bench",
        "git": "https://github.com/SakanaAI/ALE-Bench.git",
        "kind": "coding",
    },
    "coding_multi_swe_bench_2026": {
        "source": "https://multi-swe-bench.github.io/",
        "kind": "swe",
    },
    "coding_swe_bench_plus_2026": {
        "source": "https://arxiv.org/abs/2512.17419",
        "kind": "swe",
    },
    "coding_swe_polybench_2026": {
        "source": "https://github.com/amazon-science/SWE-PolyBench",
        "git": "https://github.com/amazon-science/SWE-PolyBench.git",
        "hf": ["AmazonScience/SWE-PolyBench", "AmazonScience/SWE-PolyBench_Verified"],
        "kind": "swe",
        "splits": ["test", "train"],
    },
    "coding_swe_smith_2026": {
        "source": "https://github.com/SWE-bench/SWE-smith",
        "git": "https://github.com/SWE-bench/SWE-smith.git",
        "kind": "swe",
    },
    "coding_octocodingbench_2026": {
        "source": "https://huggingface.co/datasets/MiniMaxAI/OctoCodingBench",
        "hf": [{"id": "MiniMaxAI/OctoCodingBench", "splits": ["train"]}],
        "kind": "agent_tool",
        "splits": ["train"],
    },
    "coding_gittaskbench_2026": {
        "source": "https://huggingface.co/datasets/Nicole-Yi/GitTaskBench",
        "git": "https://huggingface.co/datasets/Nicole-Yi/GitTaskBench",
        "kind": "coding",
    },
    "coding_verisoftbench_2026": {
        "source": "https://huggingface.co/datasets/maxRyeery/VeriSoftBench",
        "hf": [{"id": "maxRyeery/VeriSoftBench", "splits": ["train"]}],
        "kind": "formal_verification",
        "splits": ["train"],
    },
    "coding_nous_rlvr_coding_2026": {
        "source": "https://huggingface.co/datasets/NousResearch/RLVR_Coding_Problems",
        "hf": ["NousResearch/RLVR_Coding_Problems"],
        "kind": "coding",
        "splits": ["train", "test", "validation"],
    },
    "reasoning_arc_agi2_2026": {
        "source": "https://arcprize.org/arc-agi/2",
        "git": "https://github.com/arcprize/ARC-AGI-2.git",
        "kind": "reasoning",
    },
    "reasoning_arc_agi3_2026": {
        "source": "https://arcprize.org/arc-agi/3/",
        "kind": "interactive_reasoning",
    },
    "reasoning_livebench_2026": {
        "source": "https://github.com/LiveBench/LiveBench",
        "git": "https://github.com/LiveBench/LiveBench.git",
        "kind": "reasoning",
    },
    "reasoning_livebench_math_2026": {
        "source": "https://github.com/LiveBench/LiveBench",
        "git": "https://github.com/LiveBench/LiveBench.git",
        "kind": "math",
    },
    "reasoning_hle_rolling_2026": {
        "source": "https://huggingface.co/datasets/cais/hle",
        "hf": ["cais/hle"],
        "kind": "reasoning",
        "splits": ["test", "validation", "train"],
    },
    "reasoning_rlvr_linearity_math_2026": {
        "source": "https://huggingface.co/datasets/Miaow-Lab/RLVR-Linearity-Dataset",
        "hf": ["Miaow-Lab/RLVR-Linearity-Dataset"],
        "kind": "math",
        "splits": ["train", "test", "validation"],
    },
    "reasoning_matharena_2026": {
        "source": "https://matharena.ai/",
        "hf": ["MathArena/aime_2026", "MathArena/hmmt_feb_2026", "MathArena/usamo_2025"],
        "kind": "math",
        "splits": ["train"],
    },
    "reasoning_maime2025_2026": {
        "source": "https://huggingface.co/datasets/LumiOpen/mAIME2025",
        "hf": ["LumiOpen/mAIME2025"],
        "kind": "math",
        "splits": ["test", "validation", "train"],
    },
    "multimodal_mathnet_2026": {
        "source": "https://huggingface.co/datasets/ShadenA/MathNet",
        "hf": [
            {"id": "ShadenA/MathNet", "config": "all", "splits": ["train"]},
            {"id": "ShadenA/MathNet", "splits": ["train"]},
        ],
        "kind": "multimodal_math",
        "splits": ["train"],
    },
    "reasoning_imo_bench_2026": {
        "source": "https://imobench.github.io/",
        "git": "https://github.com/google-deepmind/superhuman.git",
        "kind": "math",
    },
    "long_context_mrcr_2026": {
        "source": "https://huggingface.co/datasets/openai/mrcr",
        "hf": ["openai/mrcr"],
        "kind": "long_context",
        "splits": ["test", "validation"],
    },
    "long_context_ruler_infinitebench_2026": {
        "source": "https://github.com/NVIDIA/RULER",
        "git": "https://github.com/NVIDIA/RULER.git",
        "kind": "long_context",
    },
    "long_context_longbench_v2_2026": {
        "source": "https://github.com/THUDM/LongBench",
        "git": "https://github.com/THUDM/LongBench.git",
        "hf": ["THUDM/LongBench-v2", "THUDM/LongBench"],
        "kind": "long_context",
        "splits": ["test", "validation", "dev"],
    },
    "long_context_graphwalks_2026": {
        "source": "https://huggingface.co/datasets/openai/graphwalks",
        "hf": [{"id": "openai/graphwalks", "splits": ["train"]}],
        "kind": "long_context",
        "splits": ["train"],
    },
    "long_context_longproc_2026": {
        "source": "https://github.com/princeton-pli/LongProc",
        "git": "https://github.com/princeton-pli/LongProc.git",
        "kind": "long_context",
    },
    "long_context_helmet_longproc_2026": {
        "source": "https://pli.princeton.edu/blog/2025/long-input-long-output-holistic-long-context-evaluation-helmet-and-longproc",
        "kind": "long_context",
    },
    "long_context_longcodebench_2026": {
        "source": "https://huggingface.co/papers/2505.07897",
        "kind": "long_context",
    },
    "long_context_nolima_1m_2026": {
        "source": "https://github.com/adobe-research/NoLiMa",
        "git": "https://github.com/adobe-research/NoLiMa.git",
        "hf": ["amodaresi/NoLiMa"],
        "kind": "long_context",
        "splits": ["train", "test", "validation"],
    },
    "long_context_sagascale_2026": {
        "source": "https://arxiv.org/abs/2601.09723",
        "kind": "long_context",
    },
    "long_context_academiceval_2026": {
        "source": "https://github.com/ulab-uiuc/AcademicEval",
        "git": "https://github.com/ulab-uiuc/AcademicEval.git",
        "kind": "long_context",
    },
    "long_context_memoryagentbench_2026": {
        "source": "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench",
        "hf": [
            {
                "id": "ai-hyz/MemoryAgentBench",
                "splits": [
                    "Accurate_Retrieval",
                    "Test_Time_Learning",
                    "Long_Range_Understanding",
                    "Conflict_Resolution",
                ],
            },
        ],
        "kind": "long_context",
        "splits": ["Accurate_Retrieval", "Test_Time_Learning", "Long_Range_Understanding", "Conflict_Resolution"],
    },
    "long_context_ama_bench_2026": {
        "source": "https://huggingface.co/datasets/AMA-bench/AMA-bench",
        "hf": [{"id": "AMA-bench/AMA-bench", "splits": ["test"]}],
        "kind": "long_context",
        "splits": ["test"],
    },
    "multimodal_mmmu_pro_2026": {
        "source": "https://huggingface.co/datasets/MMMU/MMMU_Pro",
        "hf": ["MMMU/MMMU_Pro"],
        "kind": "multimodal_mcq",
        "splits": ["test", "validation", "dev"],
    },
    "multimodal_mmmu_pro_standard_2026": {
        "source": "https://huggingface.co/datasets/MMMU/MMMU_Pro",
        "hf": ["MMMU/MMMU_Pro"],
        "kind": "multimodal_mcq",
        "splits": ["test", "validation", "dev"],
    },
    "multimodal_mmlongbench_2026": {
        "source": "https://huggingface.co/datasets/ZhaoweiWang/MMLongBench",
        "hf": [
            {
                "id": "ZhaoweiWang/MMLongBench",
                "splits": ["test"],
                "files": ["mmlb_data_example/**/*.jsonl"],
            }
        ],
        "kind": "multimodal_long_context",
        "splits": ["test"],
    },
    "multimodal_video_morse500_2026": {
        "source": "https://huggingface.co/datasets/video-reasoning/morse-500",
        "hf": [{"id": "video-reasoning/morse-500", "splits": ["test"], "files": ["*.csv", "*.jsonl", "*.json"]}],
        "kind": "video",
        "splits": ["test"],
    },
    "multimodal_sonic_o1_2026": {
        "source": "https://huggingface.co/datasets/vector-institute/sonic-o1",
        "git": "https://github.com/vectorinstitute/sonic-o1.git",
        "hf": [
            {"id": "vector-institute/sonic-o1", "config": "task1_summarization", "splits": ["train", "test"]},
            {"id": "vector-institute/sonic-o1", "config": "task2_mcq", "splits": ["train", "test"]},
            {"id": "vector-institute/sonic-o1", "config": "task3_temporal_localization", "splits": ["train", "test"]},
            {"id": "vector-institute/sonic-o1", "splits": ["train", "test"]},
        ],
        "kind": "video_audio",
        "splits": ["train", "test"],
    },
    "multimodal_mme_unify_2026": {
        "source": "https://huggingface.co/datasets/wulin222/MME-Unify",
        "git": "https://github.com/MME-Benchmarks/MME-Unify.git",
        "hf": [{"id": "wulin222/MME-Unify", "splits": ["train"]}],
        "kind": "multimodal_mcq",
        "splits": ["train"],
    },
    "multimodal_longspeech_2026": {
        "source": "https://huggingface.co/datasets/ICTNLP/LongSpeech-Eval",
        "hf": [
            {"id": "ICTNLP/LongSpeech-Eval", "splits": ["test", "validation", "train"]},
            {"id": "AIDC-AI/Marco_Longspeech", "splits": ["test", "validation", "train"]},
        ],
        "kind": "audio",
        "splits": ["test", "validation", "train"],
    },
    "multimodal_video_understanding_2026": {
        "source": "https://huggingface.co/datasets/MME-Benchmarks/Video-MME-v2",
        "git": "https://github.com/MME-Benchmarks/Video-MME-v2.git",
        "hf": ["MME-Benchmarks/Video-MME-v2"],
        "kind": "video",
        "splits": ["test", "validation"],
    },
    "multimodal_video_mme_v2_grouped_2026": {
        "source": "https://huggingface.co/datasets/MME-Benchmarks/Video-MME-v2",
        "git": "https://github.com/MME-Benchmarks/Video-MME-v2.git",
        "hf": ["MME-Benchmarks/Video-MME-v2"],
        "kind": "video",
        "splits": ["test", "validation"],
    },
    "multimodal_lvbench_2026": {
        "source": "https://huggingface.co/datasets/zai-org/LVBench",
        "git": "https://github.com/zai-org/LVBench.git",
        "hf": ["zai-org/LVBench"],
        "kind": "video",
        "splits": ["test", "validation"],
    },
    "multimodal_lvomnibench_2026": {
        "source": "https://huggingface.co/datasets/KD-TAO/LVOmniBench",
        "git": "https://github.com/KD-TAO/LVOmniBench.git",
        "hf": ["KD-TAO/LVOmniBench"],
        "kind": "video_audio",
        "splits": ["test", "validation"],
    },
    "multimodal_jointavbench_2026": {
        "source": "https://huggingface.co/datasets/JointAVBench/JointAVBench",
        "git": "https://github.com/roverx12345/JointAVBench.git",
        "hf": ["JointAVBench/JointAVBench"],
        "kind": "video_audio",
        "splits": ["test", "validation", "train"],
    },
    "multimodal_audiobench_2026": {
        "source": "https://github.com/AudioLLMs/AudioBench",
        "git": "https://github.com/AudioLLMs/AudioBench.git",
        "hf": [
            {"id": "hlt-lab/voicebench", "config": "ifeval", "splits": ["test"]},
            "gamma-lab-umd/MMAU-test",
        ],
        "kind": "audio",
        "splits": ["test"],
    },
    "multimodal_audiobench_mmau_2026": {
        "source": "https://huggingface.co/datasets/gamma-lab-umd/MMAU-test",
        "git": "https://github.com/AudioLLMs/AudioBench.git",
        "hf": ["gamma-lab-umd/MMAU-test", "gamma-lab-umd/MMAU-Pro"],
        "kind": "audio",
        "splits": ["test"],
    },
    "multimodal_mmau_pro_2026": {
        "source": "https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro",
        "hf": ["gamma-lab-umd/MMAU-Pro"],
        "kind": "audio",
        "splits": ["test"],
    },
    "multimodal_mmar_audio_music_reasoning_2026": {
        "source": "https://github.com/ddlBoJack/MMAR",
        "git": "https://github.com/ddlBoJack/MMAR.git",
        "kind": "audio",
    },
    "multimodal_mpbench_2026": {
        "source": "https://mpbench.github.io/",
        "kind": "multimodal",
    },
    "multimodal_cmi_bench_music_2026": {
        "source": "https://arxiv.org/abs/2506.12285",
        "kind": "audio",
    },
    "multimodal_muse_music_2026": {
        "source": "https://arxiv.org/abs/2510.19055",
        "kind": "audio",
    },
    "multimodal_rtv_bench_2026": {
        "source": "https://github.com/LJungang/RTV-Bench",
        "git": "https://github.com/LJungang/RTV-Bench.git",
        "hf": [{"id": "RTVBench/RTV-Bench", "splits": ["train", "test", "validation"], "files": ["*.json", "*.jsonl", "*.csv"]}],
        "kind": "video",
        "splits": ["train", "test", "validation"],
    },
    "multimodal_omnigaia_2026": {
        "source": "https://huggingface.co/datasets/RUC-NLPIR/OmniGAIA",
        "hf": ["RUC-NLPIR/OmniGAIA"],
        "kind": "video_audio",
        "splits": ["test"],
    },
    "multimodal_omnirag_agent_2026": {
        "source": "https://huggingface.co/datasets/JackMuX3Y/OmniRAG-Agent",
        "hf": [
            {"id": "JackMuX3Y/OmniRAG-Agent", "config": "Daily-Omni", "splits": ["train"]},
            {"id": "JackMuX3Y/OmniRAG-Agent", "config": "VideoOmniBench", "splits": ["train"]},
            {"id": "JackMuX3Y/OmniRAG-Agent", "config": "WorldSense", "splits": ["train"]},
        ],
        "kind": "video_audio",
        "splits": ["train"],
    },
    "multimodal_vstat_visual_state_tracking_2026": {
        "source": "https://huggingface.co/datasets/VSTAT-NeurIPS2026/VSTAT",
        "hf": [
            {
                "id": "VSTAT-NeurIPS2026/VSTAT",
                "files": [
                    "vstat_qa_clean.json",
                    "youtube_metadata.json",
                    "youtube_resolutions.json",
                    "redactions.json",
                ],
            }
        ],
        "kind": "video",
        "splits": ["train"],
    },
    "multimodal_smmbench_2026": {
        "source": "https://huggingface.co/datasets/HuacanChai/SMMBench",
        "git": "https://github.com/FatCatCHC/SMMBench.git",
        "hf": [
            {
                "id": "HuacanChai/SMMBench",
                "revision": "d19ef39f8b73cea533ad34532c6ba9a70637ea25",
                "files": ["Samples/cluster_*/QA_sample.json"],
            }
        ],
        "kind": "multimodal_agent_memory",
        "splits": ["train"],
    },
    "multimodal_maverix_av_reasoning_2026": {
        "source": "https://maverix-benchmark.github.io/",
        "kind": "video",
    },
    "multimodal_river_video_interaction_2026": {
        "source": "https://github.com/OpenGVLab/RIVER",
        "git": "https://github.com/OpenGVLab/RIVER.git",
        "hf": [
            {
                "id": "OpenGVLab/RIVER",
                "splits": ["train", "test", "validation"],
                "files": ["*.json", "*.jsonl", "*.csv"],
            }
        ],
        "kind": "video",
        "splits": ["train", "test", "validation"],
    },
    "multimodal_audiomarathon_2026": {
        "source": "https://huggingface.co/datasets/AudioMarathon/AudioMarathon",
        "hf": ["AudioMarathon/AudioMarathon", "Hezep/AudioMarathon"],
        "kind": "audio",
        "splits": ["test"],
    },
    "multimodal_rewardbench2_2026": {
        "source": "https://huggingface.co/datasets/rl-research/multimodal-rewardbench-2",
        "hf": [
            {"id": "rl-research/multimodal-rewardbench-2", "config": "edit", "splits": ["test", "validation", "train"]},
            {"id": "rl-research/multimodal-rewardbench-2", "config": "interleaved", "splits": ["test", "validation", "train"]},
            {"id": "rl-research/multimodal-rewardbench-2", "config": "reasoning", "splits": ["test", "validation", "train"]},
            {"id": "rl-research/multimodal-rewardbench-2", "config": "t2i", "splits": ["test", "validation", "train"]},
        ],
        "kind": "multimodal_mcq",
        "splits": ["test", "validation", "train"],
    },
    "long_video_longvt_2026": {
        "source": "https://github.com/EvolvingLMMs-Lab/LongVT",
        "git": "https://github.com/EvolvingLMMs-Lab/LongVT.git",
        "kind": "video_audio",
    },
    "multimodal_rbench_v_visual_reasoning_2026": {
        "source": "https://evalmodels.github.io/rbenchv/",
        "kind": "multimodal_mcq",
    },
    "generation_image_edit_2026": {
        "source": "https://github.com/PKU-YuanGroup/ImgEdit",
        "git": "https://github.com/PKU-YuanGroup/ImgEdit.git",
        "hf": ["sysuyy/ImgEdit"],
        "kind": "image_generation",
        "splits": ["test", "validation", "train"],
    },
    "generation_editreward_bench_2026": {
        "source": "https://github.com/TIGER-AI-Lab/EditReward",
        "kind": "image_generation",
    },
    "safety_iesbench_image_edit_2026": {
        "source": "https://huggingface.co/datasets/CSU-JPG/IESBench",
        "hf": ["CSU-JPG/IESBench"],
        "kind": "image_generation",
        "splits": ["test", "validation", "train"],
    },
    "safety_tool_security_2026": {
        "source": "internal_red_team_jsonl_and_authorized_public_slices",
        "kind": "tool_security",
    },
    "deployment_turboquant_kv_1m_2026": {
        "source": "https://openreview.net/pdf/7d33913c9a4f47c8abb294d6beb85d30124747ca.pdf",
        "kind": "deployment_performance",
    },
    "deployment_performance_2026": {
        "source": "internal_hardware_matrix_jsonl",
        "kind": "deployment_performance",
    },
    "generation_video_2026": {
        "source": "https://github.com/Vchitect/VBench",
        "git": "https://github.com/Vchitect/VBench.git",
        "kind": "video_generation",
    },
    "generation_vbench2_intrinsic_faithfulness_2026": {
        "source": "https://vchitect.github.io/VBench-2.0-project/",
        "git": "https://github.com/Vchitect/VBench.git",
        "kind": "video_generation",
    },
    "generation_vbenchpp_trustworthiness_2026": {
        "source": "https://github.com/Vchitect/VBench",
        "git": "https://github.com/Vchitect/VBench.git",
        "kind": "video_generation",
    },
    "generation_audio_speech_2026": {
        "source": "https://huggingface.co/datasets/hlt-lab/voicebench",
        "git": "https://github.com/AudioLLMs/AudioBench.git",
        "hf": [{"id": "hlt-lab/voicebench", "config": "ifeval", "splits": ["test"]}],
        "kind": "audio_generation",
        "splits": ["test"],
    },
    "generation_ttsds2_2026": {
        "source": "https://huggingface.co/datasets/ttsds/listening_test",
        "hf": [{"id": "ttsds/listening_test", "splits": ["test"]}],
        "kind": "audio_generation",
        "splits": ["test"],
    },
    "generation_emergent_tts_eval_2026": {
        "source": "https://huggingface.co/datasets/bosonai/EmergentTTS-Eval",
        "git": "https://github.com/boson-ai/EmergentTTS-Eval-public.git",
        "hf": [{"id": "bosonai/EmergentTTS-Eval", "splits": ["train", "test", "validation"]}],
        "kind": "audio_generation",
        "splits": ["train", "test", "validation"],
    },
    "generation_long_tts_eval_2026": {
        "source": "https://huggingface.co/datasets/wcy1122/Long-TTS-Eval",
        "git": "https://github.com/dvlab-research/MGM-Omni.git",
        "hf": [
            {
                "id": "wcy1122/Long-TTS-Eval",
                "splits": ["long_tts_eval_en", "long_tts_eval_zh", "hard_tts_eval_en", "hard_tts_eval_zh"],
            }
        ],
        "kind": "audio_generation",
        "splits": ["long_tts_eval_en", "long_tts_eval_zh", "hard_tts_eval_en", "hard_tts_eval_zh"],
    },
    "generation_tta_bench_2026": {
        "source": "https://huggingface.co/datasets/Hui519/TTA-Bench",
        "git": "https://github.com/lcc-404/TTA-Bench-tools.git",
        "hf": [{"id": "Hui519/TTA-Bench", "splits": ["train", "test", "validation"], "files": ["*.json", "*.jsonl", "*.csv"]}],
        "kind": "audio_generation",
        "splits": ["train", "test", "validation"],
    },
    "generation_nv_bench_2026": {
        "source": "https://huggingface.co/datasets/AnonyData/NV-Bench",
        "hf": [
            {"id": "AnonyData/NV-Bench", "config": "zh_single", "splits": ["test"]},
            {"id": "AnonyData/NV-Bench", "config": "zh_multi", "splits": ["test"]},
            {"id": "AnonyData/NV-Bench", "config": "en_single", "splits": ["test"]},
            {"id": "AnonyData/NV-Bench", "config": "en_multi", "splits": ["test"]},
        ],
        "kind": "audio_generation",
        "splits": ["test"],
    },
    "generation_tricky_tts_2026": {
        "source": "https://huggingface.co/datasets/Trelis/tricky-tts-public",
        "hf": ["Trelis/tricky-tts-public"],
        "kind": "audio_generation",
        "splits": ["train"],
    },
    "generation_avgen_bench_2026": {
        "source": "https://microsoft.github.io/AVGen-Bench/",
        "kind": "video_generation",
    },
    "generation_t2i_reasonbench_2026": {
        "source": "https://github.com/KaiyueSun98/T2I-ReasonBench",
        "git": "https://github.com/KaiyueSun98/T2I-ReasonBench.git",
        "kind": "image_generation",
    },
    "generation_complexbench_edit_2026": {
        "source": "https://github.com/llllly26/ComplexBench-Edit",
        "git": "https://github.com/llllly26/ComplexBench-Edit.git",
        "kind": "image_generation",
    },
    "generation_oneig_bench_2026": {
        "source": "https://huggingface.co/datasets/OneIG-Bench/OneIG-Bench",
        "hf": [{"id": "OneIG-Bench/OneIG-Bench", "config": "OneIG-Bench", "splits": ["train"]}],
        "kind": "image_generation",
        "splits": ["test", "validation", "train"],
    },
    "generation_music_2026": {
        "source": "https://huggingface.co/datasets/music-arena/music-arena-dataset",
        "git": "https://github.com/gclef-cmu/music-arena.git",
        "hf": ["music-arena/music-arena-dataset"],
        "kind": "music_generation",
        "splits": ["train", "test", "validation"],
    },
    "generation_music_arena_2026": {
        "source": "https://huggingface.co/datasets/music-arena/music-arena-dataset",
        "git": "https://github.com/gclef-cmu/music-arena.git",
        "hf": ["music-arena/music-arena-dataset"],
        "kind": "music_generation",
        "splits": ["train", "test", "validation"],
    },
    "generation_text_to_audio_pref_2026": {
        "source": "https://huggingface.co/datasets/Rapidata/text-2-audio-human-preference-benchmark",
        "hf": ["Rapidata/text-2-audio-human-preference-benchmark"],
        "kind": "audio_generation",
        "splits": ["train", "test", "validation"],
    },
    "generation_svi_benchmark_2026": {
        "source": "https://huggingface.co/datasets/epfl-vita/svi-benchmark",
        "hf": ["epfl-vita/svi-benchmark"],
        "kind": "video_generation",
        "splits": ["train", "test", "validation"],
    },
}


MODEL_OUTPUT_KEYS = {
    "prediction",
    "model_answer",
    "model_output",
    "model_patch",
    "model_actions",
    "tool_call",
    "artifact_path",
    "output_path",
    "generated_artifact",
    "trajectory",
    "response",
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def stable_hashable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): stable_hashable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [stable_hashable(item) for item in value]
    if isinstance(value, tuple):
        return [stable_hashable(item) for item in value]
    return value


def stable_hash(value: Any) -> str:
    blob = json.dumps(stable_hashable(value), sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "benchmark"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def load_profile(path: str) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root() / candidate
    return read_json(candidate)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
    tmp.replace(path)


def resolve_path(value: str | Path, root: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root or repo_root()) / path


def make_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, list):
        return [make_jsonable(item) for item in value[:200]]
    if isinstance(value, tuple):
        return [make_jsonable(item) for item in value[:200]]
    if isinstance(value, dict):
        return {str(key): make_jsonable(val) for key, val in list(value.items())[:200] if str(key) not in MODEL_OUTPUT_KEYS}
    filename = getattr(value, "filename", None) or getattr(value, "path", None)
    if filename:
        return str(filename)
    return str(value)


def first_value(raw: dict[str, Any], keys: tuple[str, ...]) -> Any:
    casefold_keys = {str(key).casefold(): key for key in raw}
    for key in keys:
        lookup = key if key in raw else casefold_keys.get(str(key).casefold(), key)
        value = raw.get(lookup)
        if value not in (None, "", [], {}):
            return value
    return None


def normalize_choices(value: Any) -> Any:
    value = make_jsonable(value)
    if isinstance(value, dict):
        if isinstance(value.get("text"), list):
            return [str(item) for item in value.get("text") or []]
        ordered = []
        for key in sorted(value):
            if len(str(key)) <= 3:
                ordered.append(str(value[key]))
        return ordered or value
    if isinstance(value, list):
        return [make_jsonable(item) for item in value]
    return value


def normalize_tool_call(value: Any) -> Any:
    value = make_jsonable(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def profile_record_map(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = profile.get("benchmarks") or profile.get("adapters") or []
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        rid = str(record.get("benchmark_id") or record.get("id") or "")
        if rid:
            out[rid] = record
    return out


def snapshot_for(profile: dict[str, Any], benchmark_id: str) -> dict[str, Any]:
    snapshots = profile.get("reportable_snapshots")
    if not isinstance(snapshots, dict):
        return {}
    value = snapshots.get(benchmark_id)
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                return item
    return {}


def reportable_root_for(profile: dict[str, Any], benchmark_id: str, out_root: Path, respect_profile_roots: bool) -> Path:
    roots = profile.get("reportable_task_roots")
    if respect_profile_roots and isinstance(roots, dict):
        value = roots.get(benchmark_id)
        if isinstance(value, list) and value:
            return resolve_path(str(value[0]), repo_root())
        if isinstance(value, str) and value:
            return resolve_path(value, repo_root())
    return out_root / "reportable_2026" / f"{safe_name(benchmark_id)}_authorized.jsonl"


def local_root_for(benchmark_id: str, out_root: Path) -> Path:
    return out_root / "local_2026" / f"{safe_name(benchmark_id)}_public_dev.jsonl"


def selected_benchmark_ids(profile: dict[str, Any], args: argparse.Namespace) -> list[str]:
    if args.benchmark:
        ids = [str(item) for item in args.benchmark]
    elif args.suite == "core25":
        ids = [str(item) for item in profile.get("reportable_core_25") or []]
    elif args.suite == "known":
        ids = [item for item in profile_record_map(profile) if item in KNOWN_BENCHMARKS]
    else:
        ids = list(profile_record_map(profile))
    seen: set[str] = set()
    ordered: list[str] = []
    for item in ids:
        if item and item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def parse_overrides(values: list[str] | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"--source-override must be benchmark_id=path, got {value!r}")
        key, path = value.split("=", 1)
        out[key.strip()] = resolve_path(path.strip(), repo_root())
    return out


def read_jsonl_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
            if len(rows) >= limit:
                break
    return rows


def read_json_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return []

    def flatten(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            out: list[dict[str, Any]] = []
            for item in value:
                out.extend(flatten(item))
                if len(out) >= limit:
                    break
            return out[:limit]
        if isinstance(value, dict):
            if any(value.get(key) not in (None, "", [], {}) for key in ("question", "prompt", "task_description", "instruction", "problem_statement")):
                return [value]
            for key in ("data", "examples", "questions", "items", "tasks", "records", "validation", "test", "train"):
                child = value.get(key)
                if isinstance(child, (list, dict)):
                    rows = flatten(child)
                    if rows:
                        return rows[:limit]
            out = []
            for child in value.values():
                if isinstance(child, (list, dict)):
                    out.extend(flatten(child))
                    if len(out) >= limit:
                        break
            return out[:limit]
        return []

    flattened = flatten(payload)
    if flattened:
        return flattened[:limit]

    candidates: Any = payload
    if isinstance(payload, dict):
        if any(payload.get(key) not in (None, "", [], {}) for key in ("question", "prompt", "task_description", "instruction", "problem_statement")):
            candidates = [payload]
        else:
            for key in ("data", "examples", "questions", "items", "tasks", "records", "validation", "test", "train"):
                if isinstance(payload.get(key), list):
                    candidates = payload[key]
                    break
    if isinstance(candidates, dict):
        candidates = list(candidates.values())
    if not isinstance(candidates, list):
        return []
    rows = [item for item in candidates if isinstance(item, dict)]
    return rows[:limit]


def read_csv_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(dict(row))
            if len(rows) >= limit:
                break
    return rows


def parse_lightweight_yaml(text: str) -> dict[str, Any]:
    row: dict[str, Any] = {}
    current_key = ""
    block: list[str] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if current_key and (line.startswith(" ") or line.startswith("\t")):
            block.append(line.strip())
            continue
        if current_key and block:
            row[current_key] = "\n".join(block).strip()
            current_key = ""
            block = []
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if value in {"|", ">"}:
            current_key = key
            block = []
        elif key and len(value) < 20000:
            row[key] = value
    if current_key and block:
        row[current_key] = "\n".join(block).strip()
    return row


def read_yaml_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
        if isinstance(payload, dict):
            return [payload]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
    except Exception:
        pass
    row = parse_lightweight_yaml(text)
    return [row] if row else []


def read_toml_row(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {}
    if tomllib is not None:
        try:
            payload = tomllib.loads(path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(payload, dict):
                row.update(payload)
        except Exception:
            row = {}
    if not row:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            row[key.strip()] = value.strip().strip("'\"")
    instruction = path.with_name("instruction.md")
    if instruction.exists():
        row.setdefault("instruction", instruction.read_text(encoding="utf-8", errors="ignore").strip())
    readme = path.with_name("README.md")
    if readme.exists():
        row.setdefault("readme", readme.read_text(encoding="utf-8", errors="ignore").strip()[:20000])
    row.setdefault("task_id", path.parent.name)
    row.setdefault("_source_file", str(path))
    return row


def unique_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        key = path.resolve() if path.exists() else path
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def read_mcp_bench_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows: list[dict[str, Any]] = []
    for block in payload.get("server_tasks") or []:
        if not isinstance(block, dict):
            continue
        server_name = str(block.get("server_name") or "").strip()
        servers = block.get("servers")
        if not isinstance(servers, list) or not servers:
            servers = [name.strip() for name in server_name.split("+") if name.strip()]
        for task in block.get("tasks") or []:
            if not isinstance(task, dict):
                continue
            row = dict(task)
            if server_name:
                row.setdefault("server_name", server_name)
            if servers:
                row.setdefault("servers", servers)
                row.setdefault("tools", [{"name": str(name).strip()} for name in servers if str(name).strip()])
            for key in ("combination_name", "combination_type"):
                if block.get(key) not in (None, "", [], {}):
                    row.setdefault(key, block[key])
            row.setdefault("prompt", task.get("task_description") or task.get("fuzzy_description"))
            row.setdefault("_source_file", str(path))
            rows.append(row)
            if len(rows) >= limit:
                return rows
    return rows


def read_mcp_universe_task_row(path: Path) -> dict[str, Any]:
    row = read_json(path)
    row.setdefault("task_id", path.stem)
    row.setdefault("_source_file", str(path))
    row.setdefault("tools", row.get("mcp_servers"))
    row.setdefault("expected", row.get("output_format"))
    return row


def read_agent_company_scenario_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows: list[dict[str, Any]] = []
    workspace_task = path.parent.name
    task_md = path.with_name("task.md")
    checkpoints_md = path.with_name("checkpoints.md")
    dependencies_yml = path.with_name("dependencies.yml")
    task_prompt = task_md.read_text(encoding="utf-8", errors="ignore").strip() if task_md.exists() else ""
    checkpoints = checkpoints_md.read_text(encoding="utf-8", errors="ignore").strip() if checkpoints_md.exists() else ""
    dependencies: Any = None
    if dependencies_yml.exists():
        dep_rows = read_yaml_rows(dependencies_yml)
        dependencies = dep_rows[0] if len(dep_rows) == 1 else dep_rows
    for actor, scenario in payload.items():
        if not isinstance(scenario, dict):
            continue
        extra_info = str(scenario.get("extra_info") or "").strip()
        strategy_hint = str(scenario.get("strategy_hint") or "").strip()
        prompt = extra_info or f"Enterprise workplace scenario for {actor} in task {workspace_task}."
        row = dict(scenario)
        row.setdefault("actor", actor)
        row.setdefault("task_slug", workspace_task)
        row.setdefault("workspace_task", workspace_task)
        row.setdefault("task_id", f"{workspace_task}:{safe_name(str(actor))}")
        row.setdefault("prompt", task_prompt or prompt)
        if extra_info:
            row.setdefault("scenario_extra_info", extra_info)
        if strategy_hint:
            row.setdefault("scenario_strategy_hint", strategy_hint)
        if strategy_hint:
            row.setdefault("answer", strategy_hint)
        if checkpoints:
            row.setdefault("checkpoints", checkpoints)
        if dependencies not in (None, "", [], {}):
            row.setdefault("dependencies", make_jsonable(dependencies))
        row.setdefault("_source_file", str(path))
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def state_bench_conversation_summary(conversation: Any) -> dict[str, Any]:
    if not isinstance(conversation, list):
        return {}
    messages = [item for item in conversation if isinstance(item, dict)]
    if not messages:
        return {}

    system_prompt = ""
    user_turns: list[str] = []
    assistant_turns: list[str] = []
    tool_trace: list[dict[str, Any]] = []
    tool_names: list[str] = []
    for turn_index, message in enumerate(messages):
        role = str(message.get("role") or "").strip()
        content = message.get("content")
        if role == "system" and not system_prompt and isinstance(content, str):
            system_prompt = content.strip()
        elif role == "user" and isinstance(content, str):
            user_turns.append(content.strip())
        elif role == "assistant" and isinstance(content, str):
            assistant_turns.append(content.strip())
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call_index, call in enumerate(calls):
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or call.get("tool_name") or "").strip()
            if name and name not in tool_names:
                tool_names.append(name)
            tool_trace.append(
                {
                    "turn_index": turn_index,
                    "call_index": call_index,
                    "name": name,
                    "arguments": make_jsonable(call.get("arguments")),
                    "result": make_jsonable(call.get("result")),
                }
            )

    user_goal = next((turn for turn in user_turns if "[TASK_DONE]" not in turn), user_turns[0] if user_turns else "")
    final_answer = assistant_turns[-1] if assistant_turns else ""
    return {
        "system_prompt": system_prompt,
        "user_goal": user_goal,
        "final_answer": final_answer,
        "conversation": make_jsonable(messages),
        "trajectory_turns": len(messages),
        "tool_trace": tool_trace,
        "tool_call_count": len(tool_trace),
        "tools": [{"name": name} for name in tool_names],
    }


def read_state_bench_trajectory_row(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    summary = state_bench_conversation_summary(payload.get("conversation"))
    if not summary:
        return {}
    domain = path.parent.name if path.parent.name else ""
    row: dict[str, Any] = {
        "task_id": path.stem,
        "prompt": summary["user_goal"],
        "question": summary["user_goal"],
        "answer": summary["final_answer"],
        "domain": domain,
        "state_bench_split": "train_task_trajectories",
        "conversation": summary["conversation"],
        "trajectory_turns": summary["trajectory_turns"],
        "tool_trace": summary["tool_trace"],
        "tool_call_count": summary["tool_call_count"],
        "tools": summary["tools"],
        "expected_tool_call": summary["tool_trace"],
        "_source_file": str(path),
    }
    if summary["system_prompt"]:
        row["system_prompt"] = summary["system_prompt"]
    return row


def read_state_bench_task_row(path: Path) -> dict[str, Any]:
    raw = read_json(path)
    task_id = str(raw.get("task_id") or path.stem)
    domain_dir = path.parent.parent
    domain = domain_dir.name
    repo_root = domain_dir.parent.parent.parent
    split = "unknown"
    split_file = domain_dir / "splits" / "train_test.json"
    if split_file.exists():
        split_payload = read_json(split_file)
        splits = split_payload.get("splits") if isinstance(split_payload, dict) else None
        if isinstance(splits, dict):
            for split_name, task_ids in splits.items():
                if isinstance(task_ids, list) and task_id in {str(item) for item in task_ids}:
                    split = str(split_name)
                    break
    prompt = first_value(raw, ("opening_message", "task_summary", "task_id")) or task_id
    row = dict(raw)
    row.update(
        {
            "task_id": task_id,
            "domain": domain,
            "split": split,
            "state_bench_split": split,
            "prompt": prompt,
            "question": prompt,
            "scoring": {
                "kind": "state_bench",
                "requires_state_requirements": True,
                "requires_task_requirements_judge": True,
                "requires_ux_judge": True,
            },
            "_source_file": str(path),
        }
    )
    trajectory = repo_root / "datasets" / "train_task_trajectories" / domain / f"{task_id}.json"
    if trajectory.exists():
        summary = state_bench_conversation_summary(read_json(trajectory).get("conversation"))
        if summary:
            if summary["system_prompt"]:
                row.setdefault("system_prompt", summary["system_prompt"])
            row["reference_trajectory"] = {"conversation": summary["conversation"]}
            row["conversation"] = summary["conversation"]
            row["reference_tool_calls"] = summary["tool_trace"]
            row["tool_trace"] = summary["tool_trace"]
            row["tool_call_count"] = summary["tool_call_count"]
            row["tools"] = summary["tools"]
            row["expected_tool_call"] = summary["tool_trace"]
            row["final_assistant_response"] = summary["final_answer"]
            if summary["final_answer"]:
                row["answer"] = summary["final_answer"]
    return row


def special_descriptor_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    files: list[Path] = []
    files.extend(sorted(path.glob("tasks/mcpbench_tasks_*_runner_format.json")))
    files.extend(sorted(path.glob("tests/data/task/*.json")))
    files.extend(sorted(path.glob("mcpuniverse/benchmark/configs/mcpuniverse/**/*.json")))
    files.extend(sorted(path.glob("mcpuniverse/benchmark/configs/mcpmark/configs/**/*.json")))
    files.extend(sorted(path.glob("workspaces/tasks/*/scenarios.json")))
    state_bench_tasks = sorted(path.glob("state_bench/domains/*/tasks/*.json"))
    files.extend(state_bench_tasks)
    if not state_bench_tasks:
        files.extend(sorted(path.glob("datasets/train_task_trajectories/*/*.json")))
    files.extend(sorted(path.glob("Dataset/Samples/cluster_*/QA_sample.json")))
    files.extend(sorted(item for item in path.rglob("meta.json") if "tasks" in item.parts))
    files.extend(sorted(path.rglob("task.toml")))
    return unique_paths(files)


def scan_local_source(path: Path, limit: int) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    files: list[Path]
    special_rows: list[dict[str, Any]] = []
    for descriptor in special_descriptor_files(path):
        if len(special_rows) >= limit:
            break
        try:
            if descriptor.name.startswith("mcpbench_tasks_") and descriptor.name.endswith("_runner_format.json"):
                special_rows.extend(read_mcp_bench_rows(descriptor, limit - len(special_rows)))
            elif (
                (descriptor.parent.name == "task" and "tests" in descriptor.parts)
                or ("mcpuniverse" in descriptor.parts and "configs" in descriptor.parts)
            ):
                special_rows.append(read_mcp_universe_task_row(descriptor))
            elif descriptor.name == "scenarios.json" and "workspaces" in descriptor.parts:
                special_rows.extend(read_agent_company_scenario_rows(descriptor, limit - len(special_rows)))
            elif "state_bench" in descriptor.parts and "domains" in descriptor.parts and "tasks" in descriptor.parts:
                special_rows.append(read_state_bench_task_row(descriptor))
            elif "train_task_trajectories" in descriptor.parts:
                row = read_state_bench_trajectory_row(descriptor)
                if row:
                    special_rows.append(row)
            elif descriptor.name == "QA_sample.json":
                rows = read_json_rows(descriptor, limit - len(special_rows))
                for row in rows:
                    row.setdefault("_source_file", str(descriptor))
                    row.setdefault("cluster_id", descriptor.parent.name)
                special_rows.extend(rows)
            elif descriptor.name == "meta.json":
                row = read_json(descriptor)
                if row:
                    row.setdefault("_source_file", str(descriptor))
                    row.setdefault("task_id", row.get("task_id") or descriptor.parent.name)
                    special_rows.append(row)
            elif descriptor.name == "task.toml":
                special_rows.append(read_toml_row(descriptor))
        except Exception as exc:
            errors.append(f"{descriptor}: {exc}")
    if special_rows:
        return special_rows[:limit], errors

    if path.is_file():
        files = [path]
    elif path.is_dir():
        patterns = ("*.jsonl", "*.json", "*.csv", "*.yaml", "*.yml", "*.toml")
        files = []
        for pattern in patterns:
            files.extend(sorted(path.rglob(pattern)))
    else:
        return [], [f"source path missing: {path}"]

    for file_path in files:
        if len(rows) >= limit:
            break
        try:
            suffix = file_path.suffix.lower()
            if suffix == ".jsonl":
                loaded = read_jsonl_rows(file_path, limit - len(rows))
            elif suffix == ".json":
                loaded = read_json_rows(file_path, limit - len(rows))
            elif suffix == ".csv":
                loaded = read_csv_rows(file_path, limit - len(rows))
            elif suffix in {".yaml", ".yml"}:
                loaded = read_yaml_rows(file_path)
            elif suffix == ".toml":
                loaded = [read_toml_row(file_path)]
            else:
                loaded = []
            for row in loaded:
                row.setdefault("_source_file", str(file_path))
            rows.extend(loaded[: max(0, limit - len(rows))])
        except Exception as exc:
            errors.append(f"{file_path}: {exc}")
    return rows[:limit], errors


def clone_repo(repo: str, cache_root: Path, force: bool) -> tuple[Path | None, str]:
    cache_root.mkdir(parents=True, exist_ok=True)
    name = safe_name(repo.rsplit("/", 1)[-1].removesuffix(".git"))
    target = cache_root / "git" / name
    if target.exists() and force:
        shutil.rmtree(target)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "clone", "--depth", "1", repo, str(target)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=900)
        if proc.returncode != 0:
            return None, (proc.stderr or proc.stdout or "git clone failed")[-2000:]
    return target, ""


def hf_file_rows(
    dataset_id: str,
    entry: dict[str, Any],
    cache_root: Path,
    limit: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    try:
        from huggingface_hub import hf_hub_download, list_repo_files  # type: ignore
    except Exception as exc:
        return [], [f"huggingface_hub package unavailable: {exc}"]

    patterns = [str(item) for item in entry.get("files") or entry.get("file_patterns") or [] if str(item).strip()]
    if not patterns:
        patterns = ["*.jsonl", "*.json", "*.csv"]
    revision = str(entry.get("revision") or "").strip() or None
    try:
        list_kwargs: dict[str, Any] = {"repo_type": "dataset"}
        if revision:
            list_kwargs["revision"] = revision
        files = list_repo_files(dataset_id, **list_kwargs)
    except Exception as exc:
        return [], [f"{dataset_id}: list_repo_files failed: {exc}"]
    selected = [
        name
        for name in files
        if name.lower().endswith((".jsonl", ".json", ".csv"))
        and any(fnmatch.fnmatch(name, pattern) for pattern in patterns)
    ]
    for name in sorted(selected):
        if len(rows) >= limit:
            break
        try:
            download_kwargs: dict[str, Any] = {
                "repo_id": dataset_id,
                "filename": name,
                "repo_type": "dataset",
                "cache_dir": str(cache_root / "hf_files"),
            }
            if revision:
                download_kwargs["revision"] = revision
            local = Path(hf_hub_download(**download_kwargs))
            suffix = local.suffix.lower()
            if suffix == ".jsonl":
                loaded = read_jsonl_rows(local, limit - len(rows))
            elif suffix == ".json":
                loaded = read_json_rows(local, limit - len(rows))
            elif suffix == ".csv":
                loaded = read_csv_rows(local, limit - len(rows))
            else:
                loaded = []
            for idx, item in enumerate(loaded):
                if isinstance(item, dict):
                    item.setdefault("_hf_dataset", dataset_id)
                    item.setdefault("_hf_file", name)
                    item.setdefault("_source_file", str(local))
                    item.setdefault("_source_index", idx)
                    rows.append(item)
                if len(rows) >= limit:
                    break
        except Exception as exc:
            errors.append(f"{dataset_id}:{name}: {exc}")
    if not rows and selected:
        errors.append(f"{dataset_id}: selected raw files yielded no rows")
    elif not selected:
        errors.append(f"{dataset_id}: no raw files matched {patterns}")
    return rows[:limit], errors


def hf_rows(spec: dict[str, Any], cache_root: Path, limit: int) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    try:
        from datasets import Audio, load_dataset  # type: ignore
    except Exception as exc:
        return [], [f"datasets package unavailable: {exc}"]
    try:
        from datasets import Video  # type: ignore
    except Exception:
        Video = None  # type: ignore[assignment]
    try:
        from datasets import Image  # type: ignore
    except Exception:
        Image = None  # type: ignore[assignment]

    for entry in spec.get("hf") or []:
        if isinstance(entry, dict):
            dataset_id = str(entry.get("id") or entry.get("dataset") or "").strip()
            config = str(entry.get("config") or entry.get("name") or "").strip() or None
            splits = entry.get("splits") or spec.get("splits") or ["test", "validation", "train"]
        else:
            dataset_id = str(entry).strip()
            config = None
            splits = spec.get("splits") or ["test", "validation", "train"]
            entry = {"id": dataset_id}
        if not dataset_id:
            continue
        if isinstance(entry, dict) and (entry.get("files") or entry.get("file_patterns")):
            file_rows, file_errors = hf_file_rows(dataset_id, entry, cache_root, limit)
            errors.extend(file_errors)
            if file_rows:
                return file_rows, errors
            continue
        for split in splits:
            try:
                if config:
                    ds = load_dataset(dataset_id, config, split=split, cache_dir=str(cache_root / "hf"))
                else:
                    ds = load_dataset(dataset_id, split=split, cache_dir=str(cache_root / "hf"))
                features = getattr(ds, "features", None) or {}
                for column, feature in features.items():
                    if feature.__class__.__name__ == "Audio":
                        ds = ds.cast_column(column, Audio(decode=False))
                    elif feature.__class__.__name__ == "Video" and Video is not None:
                        ds = ds.cast_column(column, Video(decode=False))
                    elif feature.__class__.__name__ == "Image" and Image is not None:
                        ds = ds.cast_column(column, Image(decode=False))
                rows: list[dict[str, Any]] = []
                for idx, item in enumerate(ds):
                    if isinstance(item, dict):
                        item = dict(item)
                        item.setdefault("_hf_dataset", dataset_id)
                        if config:
                            item.setdefault("_hf_config", config)
                        item.setdefault("_hf_split", split)
                        item.setdefault("_source_index", idx)
                        rows.append(item)
                    if len(rows) >= limit:
                        break
                if rows:
                    return rows, errors
            except Exception as exc:
                label = f"{dataset_id}:{config}:{split}" if config else f"{dataset_id}:{split}"
                errors.append(f"{label}: {exc}")
        file_rows, file_errors = hf_file_rows(dataset_id, entry, cache_root, limit)
        errors.extend(file_errors)
        if file_rows:
            return file_rows, errors
    return [], errors


def collect_source_rows(
    benchmark_id: str,
    spec: dict[str, Any],
    source_override: Path | None,
    cache_root: Path,
    limit: int,
    download: bool,
    force: bool,
) -> tuple[list[dict[str, Any]], list[str], str]:
    if source_override is not None:
        rows, errors = scan_local_source(source_override, limit)
        return rows, errors, str(source_override)
    if not download:
        return [], ["download disabled and no source override supplied"], spec.get("source", benchmark_id)
    if spec.get("hf"):
        rows, errors = hf_rows(spec, cache_root, limit)
        if rows:
            return rows, errors, spec.get("source", str(spec.get("hf")))
    else:
        errors = []
    if spec.get("git"):
        repo_path, error = clone_repo(str(spec["git"]), cache_root, force)
        if repo_path is not None:
            rows, scan_errors = scan_local_source(repo_path, limit)
            return rows, errors + scan_errors, str(repo_path)
        errors.append(error)
    return [], errors or ["no downloadable source configured"], spec.get("source", benchmark_id)


def normalize_task(
    benchmark_id: str,
    raw: dict[str, Any],
    spec: dict[str, Any],
    profile_record: dict[str, Any],
    snapshot: dict[str, Any],
    mode: str,
    source_ref: str,
    index: int,
) -> dict[str, Any] | None:
    kind = str(spec.get("kind") or profile_record.get("adapter_kind") or "").lower()
    task_id = first_value(
        raw,
        (
            "task_id",
            "TASK",
            "Problem ID",
            "Grading ID",
            "id",
            "question_id",
            "instance_id",
            "problem_id",
            "video_id",
            "audio_id",
            "sample_id",
            "annotation_id",
            "uid",
            "item_id",
            "example_id",
            "index",
            "uuid",
            "_source_index",
        ),
    )
    if benchmark_id == "generation_ttsds2_2026" and raw.get("id") not in (None, "", [], {}) and raw.get("annotator") not in (None, "", [], {}):
        task_id = f"{raw.get('id')}:{raw.get('annotator')}"
    prompt = first_value(
        raw,
        (
            "prompt",
            "question",
            "instruction",
            "query",
            "question_text",
            "prompt_text",
            "prompt_en",
            "text_to_synthesize",
            "user_query",
            "problem",
            "Problem",
            "problem_markdown",
            "thm_stmt",
            "target_theorem",
            "task",
            "input_prompt",
            "task_description",
            "fuzzy_description",
            "description",
            "extra_info",
            "strategy_hint",
            "input",
            "text",
            "context",
            "goal",
        ),
    )
    choices = normalize_choices(first_value(raw, ("choices", "choice", "options", "candidates", "endings", "answers", "text_choices", "image_choices")))
    if choices in (None, "", [], {}) and isinstance(raw.get("multi_choice_QA"), dict):
        choices = normalize_choices(first_value(raw["multi_choice_QA"], ("multi_choice_QA_options", "options", "choices")))
    if choices in (None, "", [], {}) and (
        raw.get("response_a_text") not in (None, "", [], {}) or raw.get("response_b_text") not in (None, "", [], {})
    ):
        choices = [make_jsonable(raw.get("response_a_text") or ""), make_jsonable(raw.get("response_b_text") or "")]
    answer = first_value(
        raw,
        (
            "answer",
            "target",
            "gold",
            "label",
            "answer_key",
            "answerKey",
            "correct_answer",
            "expected",
            "expected_answer",
            "reference",
            "reference_output",
            "output",
            "gold_answer",
            "gt_answer",
            "ground_truth",
            "gtfa_claims",
            "ground_truth_proof",
            "answer_nodes",
            "final_answer",
            "Short Answer",
            "Points",
            "Reward",
            "value",
            "mos",
            "score",
            "rating",
            "strategy_hint",
            "chosen",
            "preference",
        ),
    )
    if answer is None and isinstance(raw.get("multi_choice_QA"), dict):
        mcq = raw["multi_choice_QA"]
        answer = first_value(mcq, ("multi_choice_QA_answer", "answer", "label", "target"))
    if prompt is None and raw.get("nums") not in (None, "", [], {}) and raw.get("target") not in (None, "", [], {}):
        prompt = (
            "Solve this Countdown arithmetic task. Use each number at most once "
            f"to reach target {raw.get('target')}. Numbers: {make_jsonable(raw.get('nums'))}."
        )
        answer = first_value(raw, ("solution_text", "solution")) or answer
    if answer is None:
        answer = first_value(raw, ("solution_text", "solution"))
    row: dict[str, Any] = {
        "schema": TASK_SCHEMA,
        "benchmark_id": benchmark_id,
        "task_id": str(task_id or stable_hash({"benchmark_id": benchmark_id, "source": source_ref, "index": index, "raw": raw})[:16]),
        "task_revision": str(raw.get("task_revision") or raw.get("revision") or snapshot.get("task_revision") or snapshot.get("dataset_revision") or f"{safe_name(benchmark_id)}-public-dev-{today()}"),
        "dataset_revision": str(raw.get("dataset_revision") or raw.get("revision") or snapshot.get("dataset_revision") or f"{safe_name(benchmark_id)}-public-dev-{today()}"),
        "source": str(raw.get("source") or raw.get("dataset_source") or snapshot.get("source") or spec.get("source") or source_ref),
        "dataset_source": str(raw.get("dataset_source") or spec.get("source") or source_ref),
        "source_ref": source_ref,
        "source_file": str(raw.get("_source_file") or ""),
        "source_index": raw.get("_source_index", index),
        "collected_at": utc_now(),
        "contamination_class": "protected_eval" if mode == "reportable" and snapshot else "public_dev_eval",
        "reportable": bool(mode == "reportable" and snapshot),
        "local_only": not bool(mode == "reportable" and snapshot),
    }
    if prompt is not None:
        row["prompt"] = make_jsonable(prompt)
        row["question"] = make_jsonable(first_value(raw, ("question",)) or prompt)
    if choices not in (None, "", [], {}):
        row["choices"] = choices
    if answer not in (None, "", [], {}):
        row["answer"] = make_jsonable(answer)

    evidence = raw.get("evidence")
    if isinstance(evidence, dict):
        for media_key, row_key in (
            ("image_evidence", "images"),
            ("video_evidence", "video"),
            ("audio_evidence", "audio"),
            ("text_evidence", "ctxs"),
        ):
            value = evidence.get(media_key)
            if value not in (None, "", [], {}):
                row.setdefault(row_key, make_jsonable(value))

    for media_key in (
        "image",
        "images",
        "image_url",
        "image_file",
        "image_path",
        "images_path",
        "prompt_images",
        "response_a_images",
        "response_b_images",
        "video",
        "video_url",
        "video_file",
        "video_name",
        "video_path",
        "audio",
        "audio_url",
        "audio_file",
        "audio_path",
        "wav",
        "wav_path",
        "prompt_wav",
        "reference_audio",
        "image_list",
        "needle_image_list",
        "subtitles",
        "subtitle",
    ):
        value = raw.get(media_key)
        if value not in (None, "", [], {}):
            row["images" if media_key == "image_list" else media_key] = make_jsonable(value)

    if any(token in kind for token in ("tool", "bfcl", "mcp")):
        tools = first_value(raw, ("tools", "enabled_tools", "functions", "tool_schema", "function", "apis", "mcp_servers", "server_name", "servers"))
        expected = first_value(raw, ("expected_tool_call", "ground_truth", "gtfa_claims", "checklist", "answer", "target", "output_format", "evaluators"))
        if tools not in (None, "", [], {}):
            row["tools"] = normalize_tool_call(tools)
        if expected not in (None, "", [], {}):
            row["expected_tool_call"] = normalize_tool_call(expected)
        for key in (
            "output_format",
            "evaluators",
            "cleanups",
            "server_name",
            "mcp_servers",
            "actor",
            "task_slug",
            "workspace_task",
            "scenario_extra_info",
            "scenario_strategy_hint",
            "checkpoints",
            "dependencies",
            "state_bench_split",
            "split",
            "trajectory_turns",
            "tool_call_count",
            "tool_trace",
            "reference_tool_calls",
            "final_assistant_response",
        ):
            value = raw.get(key)
            if value not in (None, "", [], {}):
                row[key] = make_jsonable(value)

    if any(token in kind for token in ("swe", "repo", "patch", "coding", "formal_verification")):
        for key in (
            "repo",
            "repo_name",
            "repositories",
            "base_commit",
            "base_sha",
            "issue",
            "problem_statement",
            "test_commands",
            "tests",
            "entry_point",
            "file_paths",
            "prompt_file",
            "lean_root",
            "rel_path",
            "imports",
            "thm_stmt",
            "target_theorem",
            "ground_truth_proof",
            "patch",
            "test_patch",
            "hint_text",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
            "command_build",
            "command_test",
            "command_test_small",
            "image_name",
            "meta",
            "timeout",
            "timeouts",
        ):
            value = raw.get(key)
            if value not in (None, "", [], {}):
                row[key] = make_jsonable(value)

    for key in (
        "system_prompt",
        "scaffold",
        "workspace_abs_path",
        "checklist",
        "ctxs",
        "positive_ctxs",
        "problem_type",
        "date_added",
        "category",
        "subcategory",
        "task_type",
        "domain",
        "difficulty",
        "language",
        "word_count",
        "evolution_depth",
        "country",
        "competition",
        "topics_flat",
        "start_time",
        "end_time",
        "duration",
        "demographics",
        "rationale",
        "evidence",
        "evidence_assignment",
        "meta_data",
        "query",
        "text_norm",
        "rubric",
        "subtask_count",
        "source_path",
        "event_count",
        "event_list",
        "event_relation",
        "Fairness_type",
        "scoring",
        "time_limit",
        "memory_limit",
        "conversation",
        "opening_message",
        "task_summary",
        "task_requirements",
        "state_requirements",
        "task_env_path",
        "user_id",
        "now",
        "user_simulator",
        "reference_trajectory",
    ):
        value = first_value(raw, (key,))
        if value not in (None, "", [], {}):
            row[key] = make_jsonable(value)

    if any(token in kind for token in ("terminal", "browser", "desktop")):
        for key in ("setup", "command", "commands", "oracle", "environment", "start_url", "sites", "workspace"):
            value = raw.get(key)
            if value not in (None, "", [], {}):
                row[key] = make_jsonable(value)

    if "generation" in kind:
        row.setdefault("min_bytes", int(raw.get("min_bytes") or 1))
        row.setdefault("expected_artifact_kind", kind)

    if snapshot and mode == "reportable":
        for key in (
            "snapshot_id",
            "official_snapshot_id",
            "authorized_snapshot_id",
            "snapshot_sha256",
            "snapshot_authorization",
            "authorization_ref",
            "license_ref",
        ):
            if snapshot.get(key) not in (None, ""):
                row[key] = snapshot[key]
    if "prompt" not in row and not any(key in row for key in ("image", "video", "audio", "repo", "tools", "command")):
        return None
    row["task_row_sha256"] = stable_hash(row)
    return row


def is_scorable_task(row: dict[str, Any], spec: dict[str, Any]) -> bool:
    kind = str(spec.get("kind") or "").lower()
    if "multimodal_agent_memory" in kind:
        return bool(row.get("prompt") or row.get("question")) and row.get("answer") not in (None, "", [], {})
    return True


def dedupe_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = f"{row.get('benchmark_id')}:{row.get('task_id')}"
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= limit:
            break
    return out


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_profile(args.profile)
    records = profile_record_map(profile)
    overrides = parse_overrides(args.source_override)
    run_id = str(args.run_id or f"benchmark_materialization_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    out_root = resolve_path(str(args.out_root).format(run_id=run_id), repo_root())
    cache_root = resolve_path(str(args.cache_root).format(run_id=run_id), repo_root())
    selected = selected_benchmark_ids(profile, args)
    manifests: list[dict[str, Any]] = []
    total_rows = 0
    for benchmark_id in selected:
        spec = dict(KNOWN_BENCHMARKS.get(benchmark_id) or {})
        record = records.get(benchmark_id, {})
        if not spec:
            spec = {"source": record.get("source") or "profile_only", "kind": record.get("adapter_kind") or record.get("axis") or "unknown"}
        snapshot = snapshot_for(profile, benchmark_id)
        rows_raw, errors, source_ref = collect_source_rows(
            benchmark_id,
            spec,
            overrides.get(benchmark_id),
            cache_root,
            args.limit,
            bool(args.download),
            bool(args.force),
        )
        rows: list[dict[str, Any]] = []
        for idx, raw in enumerate(rows_raw):
            row = normalize_task(benchmark_id, raw, spec, record, snapshot, args.mode, source_ref, idx)
            if row is not None and is_scorable_task(row, spec):
                rows.append(row)
        if rows_raw and not rows:
            errors.append(f"{benchmark_id}: normalized rows failed scorable task contract")
        rows = dedupe_rows(rows, args.limit)
        output_path = (
            reportable_root_for(profile, benchmark_id, out_root, bool(args.write_profile_reportable_roots))
            if args.mode == "reportable" and snapshot
            else local_root_for(benchmark_id, out_root)
        )
        if rows:
            write_jsonl(output_path, rows)
            total_rows += len(rows)
            status = "materialized"
        else:
            status = "needs_data"
        manifests.append(
            {
                "benchmark_id": benchmark_id,
                "status": status,
                "rows": len(rows),
                "raw_rows": len(rows_raw),
                "mode": args.mode,
                "output": str(output_path),
                "source": spec.get("source") or record.get("source"),
                "source_ref": source_ref,
                "reportable": bool(args.mode == "reportable" and snapshot and rows),
                "local_only": not bool(args.mode == "reportable" and snapshot and rows),
                "has_snapshot_descriptor": bool(snapshot),
                "errors": errors[:12],
            }
        )
    summary = {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "profile": str(resolve_path(args.profile, repo_root())),
        "run_id": run_id,
        "mode": args.mode,
        "download": bool(args.download),
        "limit": args.limit,
        "selected": len(selected),
        "materialized": sum(1 for item in manifests if item["status"] == "materialized"),
        "needs_data": sum(1 for item in manifests if item["status"] != "materialized"),
        "rows": total_rows,
        "out_root": str(out_root),
        "cache_root": str(cache_root),
        "records": manifests,
    }
    manifest_path = resolve_path(str(args.manifest_out).format(run_id=run_id), repo_root()) if args.manifest_out else out_root / "manifests" / "benchmark_materialization_manifest.json"
    write_json(manifest_path, summary)
    summary["manifest"] = str(manifest_path)
    return summary


def audit_profile(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_profile(args.profile)
    records = profile_record_map(profile)
    roots = profile.get("reportable_task_roots") if isinstance(profile.get("reportable_task_roots"), dict) else {}
    snapshots = profile.get("reportable_snapshots") if isinstance(profile.get("reportable_snapshots"), dict) else {}
    core_ids = [str(item) for item in profile.get("reportable_core_25") or []]
    selected = selected_benchmark_ids(profile, args)
    selected_set = set(selected)

    missing_materializer = sorted(
        benchmark_id
        for benchmark_id in selected
        if benchmark_id not in KNOWN_BENCHMARKS and benchmark_id not in snapshots and benchmark_id not in roots
    )
    profile_missing_materializer = sorted(benchmark_id for benchmark_id in selected if benchmark_id not in KNOWN_BENCHMARKS)
    known_not_profile = sorted(benchmark_id for benchmark_id in KNOWN_BENCHMARKS if benchmark_id not in records)
    core_missing_profile = sorted(benchmark_id for benchmark_id in core_ids if benchmark_id not in records)
    core_missing_roots = sorted(benchmark_id for benchmark_id in core_ids if benchmark_id not in roots)
    core_missing_snapshots = sorted(benchmark_id for benchmark_id in core_ids if benchmark_id not in snapshots)
    core_missing_materializer = sorted(benchmark_id for benchmark_id in core_ids if benchmark_id in selected_set and benchmark_id not in KNOWN_BENCHMARKS)
    reportable_without_profile = sorted((set(roots) | set(snapshots)) - set(records))

    fail_reasons: list[str] = []
    if getattr(args, "fail_missing_materializers", False) and missing_materializer:
        fail_reasons.append("selected_benchmarks_without_materializer_or_snapshot")
    if getattr(args, "fail_core25", False):
        if core_missing_profile:
            fail_reasons.append("core25_benchmarks_missing_profile_records")
        if core_missing_roots:
            fail_reasons.append("core25_benchmarks_missing_reportable_roots")
        if core_missing_snapshots:
            fail_reasons.append("core25_benchmarks_missing_reportable_snapshots")
        if core_missing_materializer:
            fail_reasons.append("core25_benchmarks_missing_known_materializers")
    if getattr(args, "fail_known_not_profile", False) and known_not_profile:
        fail_reasons.append("known_materializers_missing_profile_records")

    return {
        "schema": "omnicoder.benchmark_materializer_profile_audit_2026.v1",
        "created_at": utc_now(),
        "profile": str(resolve_path(args.profile, repo_root())),
        "suite": str(args.suite),
        "selected": len(selected),
        "profile_benchmarks": len(records),
        "known_materializers": len(KNOWN_BENCHMARKS),
        "reportable_task_roots": len(roots),
        "reportable_snapshots": len(snapshots),
        "status": "failed" if fail_reasons else "passed",
        "fail_reasons": fail_reasons,
        "missing_materializer_or_snapshot": missing_materializer,
        "profile_benchmarks_without_known_materializer": profile_missing_materializer,
        "known_materializers_without_profile_record": known_not_profile,
        "reportable_without_profile_record": reportable_without_profile,
        "core25": {
            "count": len(core_ids),
            "missing_profile_record": core_missing_profile,
            "missing_reportable_task_root": core_missing_roots,
            "missing_reportable_snapshot": core_missing_snapshots,
            "missing_known_materializer": core_missing_materializer,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize official/public benchmark task JSONLs for Omnicoder 2026")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--run-id", default=os.getenv("RUN_ID") or os.getenv("OMNICODER_RUN_ID") or "")
    parser.add_argument("--out-root", default="weights/data_factory/runs/benchmark_materialization/{run_id}")
    parser.add_argument("--cache-root", default="weights/official_benchmarks_2026/cache")
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--benchmark", action="append", help="Benchmark id to materialize; repeatable")
    parser.add_argument("--suite", choices=["known", "core25", "profile"], default="known")
    parser.add_argument("--mode", choices=["public-dev", "reportable"], default="public-dev")
    parser.add_argument("--limit", type=int, default=int(os.getenv("OMNICODER_BENCHMARK_MATERIALIZE_LIMIT", "128")))
    parser.add_argument("--download", action="store_true", help="Allow HF/git downloads. Without this, only source overrides are read.")
    parser.add_argument("--force", action="store_true", help="Refresh git clones before scanning")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if any selected benchmark did not materialize rows")
    parser.add_argument("--write-profile-reportable-roots", action="store_true", help="Write reportable snapshots to paths declared by benchmark_suite_2026.json. Default is run-scoped output.")
    parser.add_argument("--source-override", action="append", help="benchmark_id=local_file_or_dir for authorized/local snapshots")
    sub = parser.add_subparsers(dest="command")
    mat = sub.add_parser("materialize", help="Fetch/scan sources and write task JSONLs")
    mat.set_defaults(func=lambda ns: materialize(ns))
    audit = sub.add_parser("audit-profile", help="Audit benchmark profile materializer and reportable snapshot coverage")
    audit.add_argument("--fail-core25", action="store_true", help="Exit nonzero when reportable_core_25 lacks profile roots or snapshot descriptors")
    audit.add_argument("--fail-missing-materializers", action="store_true", help="Exit nonzero for selected benchmarks with no downloader, snapshot, or task root")
    audit.add_argument("--fail-known-not-profile", action="store_true", help="Exit nonzero when KNOWN_BENCHMARKS entries are absent from the profile")
    audit.set_defaults(func=lambda ns: audit_profile(ns))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        args.command = "materialize"
    func = getattr(args, "func", materialize)
    summary = func(args)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True))
    if args.strict and int(summary.get("needs_data") or 0) > 0:
        return 4
    if summary.get("status") == "failed":
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
