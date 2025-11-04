from __future__ import annotations

import os
import sys
import time
import json
import threading
from typing import Any

import pytest


_TEST_TIMES: list[dict[str, Any]] = []


def _env_summary() -> dict[str, Any]:
    try:
        return {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "threads": {
                "active_count": threading.active_count(),
                "TORCH_NUM_THREADS": os.getenv("TORCH_NUM_THREADS", ""),
                "TORCH_NUM_INTEROP_THREADS": os.getenv("TORCH_NUM_INTEROP_THREADS", ""),
                "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS", ""),
                "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS", ""),
                "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS", ""),
            },
        }
    except Exception:
        return {}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    t0 = time.perf_counter()
    outcome = yield
    dt = time.perf_counter() - t0
    rec = {
        "test": item.nodeid,
        "duration_s": float(dt),
        "env": _env_summary() if dt > 10.0 else None,
    }
    _TEST_TIMES.append(rec)
    if dt > 10.0:
        print(json.dumps({"event": "slow_test", **rec}))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    try:
        slow = [r for r in _TEST_TIMES if r.get("duration_s", 0.0) > 10.0]
        if slow:
            print(json.dumps({"event": "slow_tests_summary", "count": len(slow), "tests": slow}, indent=2))
    except Exception:
        pass

import sys
import json
import time
from pathlib import Path
from typing import Any, Dict


# Ensure project src is importable in tests without installing the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Lightweight timing/logger for per-test durations with timestamps.
_TEST_TIMINGS_PATH = Path("tests_logs/test_durations.jsonl")
_TEST_TIMINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
_TEST_START_TIMES: dict[str, float] = {}
_SESSION_START: float | None = None


def pytest_runtest_setup(item):  # type: ignore[override]
    try:
        global _SESSION_START
        if _SESSION_START is None:
            _SESSION_START = time.perf_counter()
        _TEST_START_TIMES[item.nodeid] = time.perf_counter()
        with _TEST_TIMINGS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "event": "start",
                "test": item.nodeid,
                "ts": time.perf_counter(),
            }) + "\n")
    except Exception:
        pass


def pytest_runtest_logreport(report):  # type: ignore[override]
    try:
        if report.when == "call":
            start = _TEST_START_TIMES.pop(report.nodeid, None)
            stop = time.perf_counter()
            duration = float(report.duration) if hasattr(report, 'duration') else (float(stop - start) if start is not None else None)
            payload: Dict[str, Any] = {
                "event": "finish",
                "test": report.nodeid,
                "outcome": report.outcome,
                "duration_s": duration,
                "ts": stop,
            }
            with _TEST_TIMINGS_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
            # Highlight slow tests (>10s) with a separate marker line for easy grep
            if duration is not None and duration > 10.0:
                with _TEST_TIMINGS_PATH.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "event": "slow",
                        "test": report.nodeid,
                        "duration_s": duration,
                        "ts": stop,
                    }) + "\n")
            # TPS check hook: when a test prints a bench.end line, parse the TPS and warn if <200
            try:
                # Lightweight scan last ~100KB of pytest live log for bench.end of this test
                log_path = Path("tests_logs/pytest_full.log")
                if log_path.exists():
                    text = log_path.read_text(encoding="utf-8", errors="ignore")[-100000:]
                    import re as _re
                    m = None
                    for line in text.splitlines()[::-1]:
                        if "bench.end" in line:
                            m = _re.search(r"tps=(\d+\.\d+)", line)
                            if m:
                                tps = float(m.group(1))
                                if tps < 200.0:
                                    with _TEST_TIMINGS_PATH.open("a", encoding="utf-8") as f2:
                                        f2.write(json.dumps({
                                            "event": "low_tps",
                                            "test": report.nodeid,
                                            "tps": tps,
                                            "threshold": 200.0,
                                            "ts": stop,
                                        }) + "\n")
                                break
            except Exception:
                pass
    except Exception:
        pass


def pytest_sessionfinish(session, exitstatus):  # type: ignore[override]
    try:
        stop = time.perf_counter()
        start = _SESSION_START or stop
        with _TEST_TIMINGS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "event": "session_finish",
                "duration_s": float(stop - start),
                "ts": stop,
                "exitstatus": int(exitstatus),
            }) + "\n")
        # Dump aggregated perf counters if enabled
        if os.getenv('OMNICODER_TIMING', '0') == '1':
            try:
                from omnicoder.utils.perf import snapshot as _perf_snapshot  # type: ignore
                snap = _perf_snapshot(reset=False)
                with _TEST_TIMINGS_PATH.open("a", encoding="utf-8") as f2:
                    f2.write(json.dumps({
                        "event": "perf_snapshot",
                        "metrics": snap,
                        "ts": stop,
                    }) + "\n")
            except Exception:
                pass
    except Exception:
        pass

