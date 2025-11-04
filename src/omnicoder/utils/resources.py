from __future__ import annotations

import os
from typing import Tuple


def _effective_cpu_count() -> int:
    """Detect effective CPU quota inside containers when possible.

    - Uses sched_getaffinity on POSIX if available to detect cgroup CPU sets.
    - Falls back to os.cpu_count().
    """
    try:
        if hasattr(os, "sched_getaffinity"):
            try:
                return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
            except Exception:
                pass
        return max(1, int(os.cpu_count() or 1))
    except Exception:
        return 1


def detect_cpu_threads() -> int:
    """Return a resource-aware default thread count.

    Behavior when OMNICODER_AUTO_RESOURCES=1:
    - Use all effective CPUs times a scaling factor (OMNICODER_THREADS_FACTOR, default 1.0).
    - Respect explicit OMNICODER_THREADS if provided.

    When auto is disabled, return 1 unless OMNICODER_THREADS is set.
    """
    try:
        # Respect explicit override first
        if "OMNICODER_THREADS" in os.environ:
            return max(1, int(os.environ.get("OMNICODER_THREADS", "1")))
    except Exception:
        pass

    auto = os.environ.get("OMNICODER_AUTO_RESOURCES", "0") == "1"
    eff = _effective_cpu_count()
    if not auto:
        return 1

    try:
        factor = float(os.environ.get("OMNICODER_THREADS_FACTOR", "1.0"))
    except Exception:
        factor = 1.0
    threads = max(1, int(eff * max(0.1, factor)))
    return threads


def apply_thread_env_if_auto() -> Tuple[int, int, int]:
    """If OMNICODER_AUTO_RESOURCES=1, enforce OMP/MKL/TORCH thread envs.

    Behavior:
    - When auto-scaling is enabled, we actively set OMP_NUM_THREADS, MKL_NUM_THREADS,
      and TORCH_NUM_THREADS to the recommended value, overriding prior values from
      base images (e.g., Dockerfile defaults of 1). This ensures full CPU
      utilization inside containers and hosts.
    - If the user explicitly sets OMNICODER_THREADS, that exact value is used.
    - When auto is disabled, we return current values without mutation.

    Returns a tuple of (omp_threads, mkl_threads, torch_threads).
    """
    auto = os.environ.get("OMNICODER_AUTO_RESOURCES", "0") == "1"
    if not auto:
        # Return current values (or 1) without changing when auto is off
        def _get(k: str) -> int:
            try:
                return max(1, int(os.environ.get(k, "1")))
            except Exception:
                return 1
        return (_get("OMP_NUM_THREADS"), _get("MKL_NUM_THREADS"), _get("TORCH_NUM_THREADS"))

    threads = detect_cpu_threads()
    # Enforce threads to override conservative image defaults when auto-scaling
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["TORCH_NUM_THREADS"] = str(threads)
    return (threads, threads, threads)


def recommend_num_workers(memory_gb_hint: float | None = None) -> int:
    """Recommend DataLoader num_workers based on cores and optional memory hint.

    - Defaults to min(cores // 2, 8), at least 2, when auto resources enabled.
    - Falls back to 0 when auto disabled for maximum compatibility.
    - Can be overridden via OMNICODER_WORKERS.
    """
    try:
        if "OMNICODER_WORKERS" in os.environ:
            return max(0, int(os.environ["OMNICODER_WORKERS"]))
    except Exception:
        pass

    auto = os.environ.get("OMNICODER_AUTO_RESOURCES", "0") == "1"
    if not auto:
        return 0

    cores = _effective_cpu_count()
    # Default: use most cores but leave 1 for main thread; cap by OMNICODER_WORKERS_MAX
    base = max(1, cores - 1)
    # Very rough memory-based nudge: if memory hint < 8 GB, reduce by 1
    if memory_gb_hint is not None and memory_gb_hint < 8.0:
        base = max(1, base - 1)
    try:
        max_workers = int(os.environ.get("OMNICODER_WORKERS_MAX", str(base)))
    except Exception:
        max_workers = base
    return int(max(1, min(base, max_workers)))


def gpu_summary() -> str:
    """Return a short GPU summary string; avoids hard torch dependency if missing."""
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return "cuda=False devices=0"
        n = torch.cuda.device_count()
        mems = []
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            mems.append(round(getattr(props, "total_memory", 0) / (1024**3), 1))
        return f"cuda=True devices={n} vram_gb={mems}"
    except Exception:
        return "cuda=unknown devices=0"


def audit_env(prefix: str = "OMNICODER_") -> list[str]:
    """Return a list of environment keys with the given prefix that are set but not recognized.

    Uses the centralized env registry (built from code scan + env.example.txt). Falls back to
    a curated allow list if the registry cannot be built.
    """
    import os as _os
    set_keys = [k for k in _os.environ.keys() if k.startswith(prefix)]
    allow: set[str] = set()
    try:
        from omnicoder.utils.env_registry import build_registry, allowed_keys_from_registry
        reg = build_registry(env_example_path="env.example.txt", root_dir="src")
        allow = set(allowed_keys_from_registry(reg))
    except Exception:
        allow = set()
    if not allow:
        # Minimal fallback to avoid false positives if registry build failed
        allow = set(set_keys)
    unknown = [k for k in set_keys if k not in allow]
    return sorted(unknown)


def select_device(prefer: str | None = None) -> str:
    """Select a best-available device string.

    Order:
    1) Respect OMNICODER_DEVICE if set and available
    2) CUDA if available
    3) DirectML/privateuseone if torch-directml present
    4) CPU
    """
    # 1) Explicit environment override
    try:
        env_dev = os.environ.get("OMNICODER_DEVICE", "").strip().lower()
        if env_dev:
            if env_dev in ("cpu", "cuda", "privateuseone", "dml"):
                # sanity-validate CUDA
                if env_dev == "cuda":
                    try:
                        import torch  # type: ignore
                        if torch.cuda.is_available():
                            return "cuda"
                    except Exception:
                        pass
                elif env_dev in ("privateuseone", "dml"):
                    try:
                        import torch_directml  # type: ignore  # noqa: F401
                        return "privateuseone"
                    except Exception:
                        pass
                elif env_dev == "cpu":
                    return "cpu"
    except Exception:
        pass

    # 2) CUDA
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    # 3) DirectML
    try:
        import torch_directml  # type: ignore  # noqa: F401
        return "privateuseone"
    except Exception:
        pass

    # 4) CPU fallback
    return "cpu"


