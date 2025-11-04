from __future__ import annotations

import os


def apply_core_defaults(env: dict[str, str]) -> None:
    """Apply centralized defaults for core OMNICODER_* variables to the provided env mapping.

    Only fills missing keys; never overwrites existing values.
    """
    # Logging
    env.setdefault("OMNICODER_LOG_LEVEL", os.getenv("OMNICODER_LOG_LEVEL", "INFO"))
    env.setdefault("OMNICODER_LOG_FILE", os.getenv("OMNICODER_LOG_FILE", "tests_logs/omnicoder.log"))
    # Performance toggles
    env.setdefault("OMNICODER_USE_TORCH_POST", os.getenv("OMNICODER_USE_TORCH_POST", "1"))
    env.setdefault("OMNICODER_USE_SDPA", os.getenv("OMNICODER_USE_SDPA", "1"))
    env.setdefault("OMNICODER_PRETRAIN_USE_FLASH", os.getenv("OMNICODER_PRETRAIN_USE_FLASH", "1"))
    env.setdefault("OMNICODER_PRETRAIN_AMP", os.getenv("OMNICODER_PRETRAIN_AMP", "1"))
    env.setdefault("OMNICODER_KNN_CACHE", os.getenv("OMNICODER_KNN_CACHE", "1"))
    env.setdefault("OMNICODER_GRAPHRAG_ENABLE", os.getenv("OMNICODER_GRAPHRAG_ENABLE", "1"))
    env.setdefault("OMNICODER_DUAL_SUBSTRATE", os.getenv("OMNICODER_DUAL_SUBSTRATE", "1"))
    # Use external KV by default for decode hot path to reduce aliasing and improve TPS
    env.setdefault("OMNICODER_INTERNAL_KV_CACHE", os.getenv("OMNICODER_INTERNAL_KV_CACHE", "0"))
    env.setdefault("OMNICODER_USE_LANDMARKS", os.getenv("OMNICODER_USE_LANDMARKS", "1"))
    env.setdefault("OMNICODER_COMPILE", os.getenv("OMNICODER_COMPILE", "1"))
    # Prefer fullgraph reductions to minimize retraces; can be overridden by user
    env.setdefault("OMNICODER_COMPILE_FULLGRAPH", os.getenv("OMNICODER_COMPILE_FULLGRAPH", "1"))
    env.setdefault("OMNICODER_USE_DYNAMO", os.getenv("OMNICODER_USE_DYNAMO", "1"))
    # Prefer CUDA fused MoE dispatcher when extension is available (safe: falls back automatically)
    env.setdefault("OMNICODER_MOE_CUDA_ENABLE", os.getenv("OMNICODER_MOE_CUDA_ENABLE", "1"))

    # Debug/guard defaults for runaway allocations and mem logs
    env.setdefault("OMNICODER_SHAPE_GUARD", os.getenv("OMNICODER_SHAPE_GUARD", "1"))
    env.setdefault("OMNICODER_SHAPE_MAX_ELEMS", os.getenv("OMNICODER_SHAPE_MAX_ELEMS", "200000000"))
    env.setdefault("OMNICODER_SHAPE_MAX_BYTES", os.getenv("OMNICODER_SHAPE_MAX_BYTES", str(4 * 1024 * 1024 * 1024)))
    env.setdefault("OMNICODER_BENCH_MEM", os.getenv("OMNICODER_BENCH_MEM", "1"))
    # Disk/Tokenizer guards
    env.setdefault("OMNICODER_DISABLE_DISK_CACHE", os.getenv("OMNICODER_DISABLE_DISK_CACHE", "1"))
    env.setdefault("OMNICODER_FORBID_SIMPLE", os.getenv("OMNICODER_FORBID_SIMPLE", "1"))
    # Reasoning stacks
    env.setdefault("OMNICODER_HRM_ENABLE", os.getenv("OMNICODER_HRM_ENABLE", "1"))
    env.setdefault("OMNICODER_SFB_ENABLE", os.getenv("OMNICODER_SFB_ENABLE", "1"))
    env.setdefault("OMNICODER_AGOT_ENABLE", os.getenv("OMNICODER_AGOT_ENABLE", "1"))
    env.setdefault("OMNICODER_BLOCK_VERIFY", os.getenv("OMNICODER_BLOCK_VERIFY", "1"))
    env.setdefault("OMNICODER_HALTING", os.getenv("OMNICODER_HALTING", "1"))
    env.setdefault("OMNICODER_DELIB_HEADS", os.getenv("OMNICODER_DELIB_HEADS", "1"))
    env.setdefault("OMNICODER_SDP_PREF", os.getenv("OMNICODER_SDP_PREF", "flash"))
    env.setdefault("TORCH_INDUCTOR_INSTALL_GXX", os.getenv("TORCH_INDUCTOR_INSTALL_GXX", "1"))
    # Tracing/verbosity guards (keep heavy tracing OFF by default; opt-in only)
    env.setdefault("OMNICODER_GEN_SUPER_VERBOSE", os.getenv("OMNICODER_GEN_SUPER_VERBOSE", "0"))
    env.setdefault("OMNICODER_TIMING_LOG_PER_LAYER", os.getenv("OMNICODER_TIMING_LOG_PER_LAYER", "0"))
    env.setdefault("OMNICODER_TRACE_ALL", os.getenv("OMNICODER_TRACE_ALL", "0"))
    env.setdefault("OMNICODER_TIMING", os.getenv("OMNICODER_TIMING", "1"))
    # Back-compat shim: DECODE_AUX was used in some scripts; map to DUAL_AUX if unset
    if "OMNICODER_DECODE_AUX" in os.environ and "OMNICODER_DUAL_AUX" not in env:
        env.setdefault("OMNICODER_DUAL_AUX", os.getenv("OMNICODER_DECODE_AUX", "0"))


def apply_training_defaults(env: dict[str, str]) -> None:
    env.setdefault("OMNICODER_TRAIN_BUDGET_HOURS", os.getenv("OMNICODER_TRAIN_BUDGET_HOURS", "1"))
    env.setdefault("OMNICODER_TRAIN_DEVICE", os.getenv("OMNICODER_TRAIN_DEVICE", "cuda"))
    env.setdefault("OMNICODER_OUT_ROOT", os.getenv("OMNICODER_OUT_ROOT", "weights"))
    env.setdefault("OMNICODER_TRAIN_PRESET", os.getenv("OMNICODER_TRAIN_PRESET", "mobile_4gb"))
    # Ensure large model presets are used by default for student/draft to avoid tiny smoke models
    env.setdefault("OMNICODER_STUDENT_PRESET", os.getenv("OMNICODER_STUDENT_PRESET", "mobile_4gb"))
    env.setdefault("OMNICODER_DRAFT_PRESET", os.getenv("OMNICODER_DRAFT_PRESET", "mobile_4gb"))


def apply_press_play_defaults(env: dict[str, str]) -> None:
    env.setdefault("OMNICODER_ONNX_OPSET", os.getenv("OMNICODER_ONNX_OPSET", "17"))
    env.setdefault("OMNICODER_SEQ_LEN_BUDGET", os.getenv("OMNICODER_SEQ_LEN_BUDGET", "4096"))
    env.setdefault("OMNICODER_ONNX_PRESET", os.getenv("OMNICODER_ONNX_PRESET", "generic"))


def apply_run_env_defaults(env: dict[str, str]) -> None:
    # Subsystem toggles and reasoning defaults
    env.setdefault("SFB_ENABLE", os.getenv("SFB_ENABLE", "1"))
    env.setdefault("OMNICODER_GRAPHRAG_ENABLE", os.getenv("OMNICODER_GRAPHRAG_ENABLE", "1"))
    env.setdefault("OMNICODER_PERCEIVER_ENABLE", os.getenv("OMNICODER_PERCEIVER_ENABLE", "1"))
    env.setdefault("OMNICODER_ALG_CORE", os.getenv("OMNICODER_ALG_CORE", "1"))
    # Keep consistent with core defaults: external KV by default
    env.setdefault("OMNICODER_INTERNAL_KV_CACHE", os.getenv("OMNICODER_INTERNAL_KV_CACHE", "0"))
    env.setdefault("OMNICODER_MOD_ENABLE", os.getenv("OMNICODER_MOD_ENABLE", "1"))
    env.setdefault("OMNICODER_REASONER", os.getenv("OMNICODER_REASONER", "omega"))
    env.setdefault("OMNI_REASONER", os.getenv("OMNI_REASONER", env.get("OMNICODER_REASONER", "omega")))
    env.setdefault("OMNICODER_RG_MAX_NODES", os.getenv("OMNICODER_RG_MAX_NODES", "64"))
    env.setdefault("OMNICODER_RG_MAX_DEPTH", os.getenv("OMNICODER_RG_MAX_DEPTH", "8"))
    env.setdefault("OMNICODER_RG_BUDGET_TOKENS", os.getenv("OMNICODER_RG_BUDGET_TOKENS", "1024"))
    env.setdefault("OMNICODER_RG_SPECULATIVE_BRANCHES", os.getenv("OMNICODER_RG_SPECULATIVE_BRANCHES", "3"))
    env.setdefault("OMNICODER_RG_ACCEPT_MARGIN", os.getenv("OMNICODER_RG_ACCEPT_MARGIN", "0.0"))
    env.setdefault("OMNI_GOAL_INFER", os.getenv("OMNI_GOAL_INFER", "rsa,pr,cirl"))
    env.setdefault("OMNI_MCTS_BUDGET", os.getenv("OMNI_MCTS_BUDGET", "8"))
    env.setdefault("OMNICODER_CERT_EMIT", os.getenv("OMNICODER_CERT_EMIT", "1"))
    env.setdefault("OMNICODER_DELIB_HEADS", os.getenv("OMNICODER_DELIB_HEADS", "1"))
    env.setdefault("OMNICODER_HALTING", os.getenv("OMNICODER_HALTING", "1"))
    env.setdefault("OMNICODER_HALTING_ALPHA", os.getenv("OMNICODER_HALTING_ALPHA", "0.7"))
    env.setdefault("OMNICODER_HALTING_BETA", os.getenv("OMNICODER_HALTING_BETA", "0.2"))
    env.setdefault("OMNICODER_HALTING_GAMMA", os.getenv("OMNICODER_HALTING_GAMMA", "0.1"))
    env.setdefault("OMNICODER_HALTING_THETA", os.getenv("OMNICODER_HALTING_THETA", "0.85"))
    env.setdefault("OMNICODER_BLOCK_VERIFY", os.getenv("OMNICODER_BLOCK_VERIFY", "1"))
    env.setdefault("OMNICODER_BLOCK_SIZE", os.getenv("OMNICODER_BLOCK_SIZE", "4"))
    env.setdefault("OMNICODER_DYNAMIC_CACHE", os.getenv("OMNICODER_DYNAMIC_CACHE", "1"))
    env.setdefault("OMNICODER_QUANTIZE_ONNX_PER_OP", os.getenv("OMNICODER_QUANTIZE_ONNX_PER_OP", "1"))
    env.setdefault("OMNICODER_FORCE_FETCH", os.getenv("OMNICODER_FORCE_FETCH", "1"))
    # Keep bench diagnostics off by default during normal runs; can be enabled explicitly for profiling
    env.setdefault("OMNICODER_BENCH_DIAG", os.getenv("OMNICODER_BENCH_DIAG", "1"))
    # SFB defaults
    env.setdefault("SFB_FACTORIZER", os.getenv("SFB_FACTORIZER", "amr,srl"))
    env.setdefault("SFB_BP_ITERS", os.getenv("SFB_BP_ITERS", "10"))
    env.setdefault("SFB_BP_ITERS_SEM", os.getenv("SFB_BP_ITERS_SEM", "3"))
    env.setdefault("SFB_COMPILE_SPN", os.getenv("SFB_COMPILE_SPN", "1"))
    env.setdefault("SFB_BLOCK_VERIFY", os.getenv("SFB_BLOCK_VERIFY", "1"))
    env.setdefault("SFB_BLOCK_VERIFY_SIZE", os.getenv("SFB_BLOCK_VERIFY_SIZE", "4"))
    env.setdefault("SFB_BIAS_ALPHA", os.getenv("SFB_BIAS_ALPHA", "0.2"))
    env.setdefault("SFB_SEM_WEIGHT", os.getenv("SFB_SEM_WEIGHT", "1.0"))
    env.setdefault("SFB_ALLOW_NET", os.getenv("SFB_ALLOW_NET", "0"))
    env.setdefault("SFB_LOGIC_Z3", os.getenv("SFB_LOGIC_Z3", "1"))
    # Constraints/factors
    env.setdefault("OMNICODER_FACTORS", os.getenv("OMNICODER_FACTORS", "spatial,temporal,semantic"))
    env.setdefault("OMNICODER_FACTORS_ITERS", os.getenv("OMNICODER_FACTORS_ITERS", "2"))
    env.setdefault("OMNICODER_CONSTRAINTS", os.getenv("OMNICODER_CONSTRAINTS", "text,code,image,video,audio"))
    env.setdefault("OMNICODER_CONSTRAINT_WEIGHT", os.getenv("OMNICODER_CONSTRAINT_WEIGHT", "auto"))
    # Router and dual substrate
    env.setdefault("OMNICODER_ROUTER_TMIN", os.getenv("OMNICODER_ROUTER_TMIN", "0.8"))
    env.setdefault("OMNICODER_ROUTER_TMAX", os.getenv("OMNICODER_ROUTER_TMAX", "1.2"))
    env.setdefault("OMNICODER_ROUTER_TEMP_LAMBDA", os.getenv("OMNICODER_ROUTER_TEMP_LAMBDA", "3.0"))
    env.setdefault("OMNICODER_DUAL_SUBSTRATE", os.getenv("OMNICODER_DUAL_SUBSTRATE", "1"))
    env.setdefault("OMNICODER_DUAL_ALPHA", os.getenv("OMNICODER_DUAL_ALPHA", "1.0"))
    env.setdefault("OMNICODER_DUAL_BETA", os.getenv("OMNICODER_DUAL_BETA", "0.0"))
    env.setdefault("OMNICODER_DUAL_AUX", os.getenv("OMNICODER_DUAL_AUX", "1"))
    # Heavy export lanes
    env.setdefault("OMNICODER_EXPORT_EXECUTORCH", os.getenv("OMNICODER_EXPORT_EXECUTORCH", "1"))
    env.setdefault("OMNICODER_EXPORT_COREML_DECODE", os.getenv("OMNICODER_EXPORT_COREML_DECODE", "1"))
    # Ω2 adaptive
    env.setdefault("OMNICODER_TRACE_ENABLE", os.getenv("OMNICODER_TRACE_ENABLE", "1"))
    env.setdefault("OMNICODER_AGOT_ENABLE", os.getenv("OMNICODER_AGOT_ENABLE", "1"))
    env.setdefault("OMNICODER_LATENT_BFS_ENABLE", os.getenv("OMNICODER_LATENT_BFS_ENABLE", "1"))
    env.setdefault("OMNICODER_REFLECT_ENABLE", os.getenv("OMNICODER_REFLECT_ENABLE", "1"))
    env.setdefault("OMNICODER_SYMBOLIC_PLANNER", os.getenv("OMNICODER_SYMBOLIC_PLANNER", "1"))
    # Retrieval/kNN
    env.setdefault("OMNICODER_KNN_CACHE", os.getenv("OMNICODER_KNN_CACHE", "1"))
    env.setdefault("OMNICODER_KNN_K", os.getenv("OMNICODER_KNN_K", "16"))
    env.setdefault("OMNICODER_KNN_LAMBDA", os.getenv("OMNICODER_KNN_LAMBDA", "0.2"))
    # Fast path
    env.setdefault("OMNICODER_USE_TORCH_POST", os.getenv("OMNICODER_USE_TORCH_POST", "1"))
    env.setdefault("OMNICODER_USE_ONNX", os.getenv("OMNICODER_USE_ONNX", "0"))
    env.setdefault("OMNICODER_TOOL_USE", os.getenv("OMNICODER_TOOL_USE", "1"))
    # Autofetch
    env.setdefault("OMNICODER_AUTOFETCH", os.getenv("OMNICODER_AUTOFETCH", "1"))
    env.setdefault("OMNICODER_FETCH_LIMIT", os.getenv("OMNICODER_FETCH_LIMIT", "1000000"))
    # Adaptive gating
    env.setdefault("OMNICODER_ADAPTIVE_GATING", os.getenv("OMNICODER_ADAPTIVE_GATING", "1"))
    env.setdefault("OMNICODER_ADAPTIVE_TOP_K_MIN", os.getenv("OMNICODER_ADAPTIVE_TOP_K_MIN", "1"))
    env.setdefault("OMNICODER_ADAPTIVE_TOP_K_MAX", os.getenv("OMNICODER_ADAPTIVE_TOP_K_MAX", "4"))
    env.setdefault("OMNICODER_ADAPTIVE_CONF_FLOOR", os.getenv("OMNICODER_ADAPTIVE_CONF_FLOOR", "0.3"))
    env.setdefault("OMNICODER_EARLY_EXIT", os.getenv("OMNICODER_EARLY_EXIT", "1"))
    env.setdefault("OMNICODER_EARLY_EXIT_ENTROPY", os.getenv("OMNICODER_EARLY_EXIT_ENTROPY", "1.0"))
    # Training-side gating
    env.setdefault("OMNICODER_VAR_K_TRAIN", os.getenv("OMNICODER_VAR_K_TRAIN", "1"))
    env.setdefault("OMNICODER_VAR_K_MIN", os.getenv("OMNICODER_VAR_K_MIN", "1"))
    env.setdefault("OMNICODER_VAR_K_MAX", os.getenv("OMNICODER_VAR_K_MAX", "4"))
    env.setdefault("OMNICODER_VAR_K_THRESH", os.getenv("OMNICODER_VAR_K_THRESH", "0.5"))
    env.setdefault("OMNICODER_DIFF_LOSS_COEF", os.getenv("OMNICODER_DIFF_LOSS_COEF", "0.01"))
    env.setdefault("OMNICODER_HALT_LOSS_COEF", os.getenv("OMNICODER_HALT_LOSS_COEF", "0.01"))
    env.setdefault("OMNICODER_HALT_ENTROPY", os.getenv("OMNICODER_HALT_ENTROPY", "1.0"))
    # PPI
    env.setdefault("OMNICODER_PPI_DEFAULT_ON", os.getenv("OMNICODER_PPI_DEFAULT_ON", "1"))
    env.setdefault("OMNICODER_PPI_MODES", os.getenv("OMNICODER_PPI_MODES", "image,audio,video"))
    # Long context
    env.setdefault("OMNICODER_TARGET_CTX", os.getenv("OMNICODER_TARGET_CTX", "32768"))
    env.setdefault("OMNICODER_USE_LANDMARKS", os.getenv("OMNICODER_USE_LANDMARKS", "1"))
    env.setdefault("OMNICODER_WINDOW_SIZE", os.getenv("OMNICODER_WINDOW_SIZE", "512"))
    # Decode-window synonyms used by older scripts/tools; define to avoid env_unknown noise
    env.setdefault("OMNICODER_DECODE_WINDOW", os.getenv("OMNICODER_DECODE_WINDOW", env.get("OMNICODER_WINDOW_SIZE", "0")))
    env.setdefault("OMNICODER_STATIC_DECODE_WINDOW", os.getenv("OMNICODER_STATIC_DECODE_WINDOW", "0"))
    env.setdefault("OMNICODER_MEM_SLOTS", os.getenv("OMNICODER_MEM_SLOTS", "4"))
    # Torch compile
    env.setdefault("OMNICODER_COMPILE", os.getenv("OMNICODER_COMPILE", "1"))
    env.setdefault("OMNICODER_COMPILE_FULLGRAPH", os.getenv("OMNICODER_COMPILE_FULLGRAPH", env.get("OMNICODER_COMPILE_FULLGRAPH", "1")))
    # Enable cudagraphs at runtime; do not skip dynamic graphs (we stabilize shapes)
    env.setdefault("TORCHINDUCTOR_USE_CUDA_GRAPHS", os.getenv("TORCHINDUCTOR_USE_CUDA_GRAPHS", "1"))
    env.setdefault("OMNICODER_USE_SDPA", os.getenv("OMNICODER_USE_SDPA", "1"))
    env.setdefault("TORCH_INDUCTOR_INSTALL_GXX", os.getenv("TORCH_INDUCTOR_INSTALL_GXX", "1"))
    # Pretrain fast path
    env.setdefault("OMNICODER_PRETRAIN_USE_FLASH", os.getenv("OMNICODER_PRETRAIN_USE_FLASH", "1"))
    env.setdefault("OMNICODER_PRETRAIN_AMP", os.getenv("OMNICODER_PRETRAIN_AMP", "1"))


def apply_quality_profile(env: dict[str, str]) -> None:
    """Quality-first profile with strong performance. Prioritizes model quality and stability over aggressive speed tricks.

    Safe, centralized choices that reduce conflicts:
    - Prefer PyTorch runtime with compile/SDPA; avoid ONNX path by default
    - Disable aggressive early exit/adaptive gating for maximum quality
    - Keep block-verify on and modest block size
    - Keep speculative draft conservative with verification
    - Reduce extreme tracing verbosity
    """
    # Runtime: prefer PyTorch (TorchDynamo+Inductor, SDPA) for quality/perf; skip ONNX by default
    env.setdefault("OMNICODER_USE_ONNX", "0")
    env.setdefault("OMNICODER_COMPILE", "1")
    env.setdefault("OMNICODER_USE_DYNAMO", "1")
    env.setdefault("OMNICODER_USE_SDPA", "1")
    env.setdefault("TORCH_INDUCTOR_INSTALL_GXX", "1")
    # Quality guards
    env.setdefault("OMNICODER_BLOCK_VERIFY", "1")
    env.setdefault("OMNICODER_BLOCK_VERIFY_SIZE", "4")
    # Avoid early exit/over-aggressive gating for quality
    env.setdefault("OMNICODER_EARLY_EXIT", "0")
    env.setdefault("OMNICODER_ADAPTIVE_GATING", "0")
    # Speculative decoding kept conservative and verified
    env.setdefault("OMNICODER_SPEC_DRAFT_LEN", "1")
    env.setdefault("OMNICODER_VERIFY_THRESHOLD", "0.2")
    env.setdefault("OMNICODER_DRAFT_VERIFY_THRESHOLD", "0.2")
    # Retrieval/knowledge helpers ON
    env.setdefault("OMNICODER_KNN_CACHE", "1")
    env.setdefault("OMNICODER_GRAPHRAG_ENABLE", "1")
    # Logging/tracing kept light
    env.setdefault("OMNICODER_TRACE_ENABLE", "0")
    env.setdefault("OMNICODER_TRACE_ALL", "0")
    env.setdefault("OMNICODER_GEN_SUPER_VERBOSE", "0")
    env.setdefault("OMNICODER_TIMING_LOG_PER_LAYER", "0")
    env.setdefault("OMNICODER_TIMING", "1")


def apply_profile(env: dict[str, str], name: str | None) -> None:
    name = (name or "quality").strip().lower()
    if name in ("quality", "quality_first", "best"):
        apply_quality_profile(env)
    # Additional profiles could be added (e.g., "throughput", "debug").


