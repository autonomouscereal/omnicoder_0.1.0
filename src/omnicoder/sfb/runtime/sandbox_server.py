from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn  # type: ignore
except Exception:  # pragma: no cover - import fallback
    FastAPI = None  # type: ignore
    BaseModel = object  # type: ignore
    uvicorn = None  # type: ignore

from .sandbox import LocalIsolatedPythonSandbox
from ...utils.logger import get_logger


class RunRequest(BaseModel):  # type: ignore[misc]
    code: str
    tests: str
    timeout: int = 5


def create_app() -> Any:
    app = FastAPI(title="OmniCoder Sandbox", version="1.0")  # type: ignore[arg-type]
    _LOG = get_logger("omnicoder.sandbox")
    try:
        _LOG.info("sandbox: creating app")
    except Exception:
        pass
    runner = LocalIsolatedPythonSandbox()

    @app.post("/run")
    def run(req: RunRequest) -> Dict[str, Any]:  # type: ignore[no-redef]
        try:
            _LOG.info(
                "sandbox: /run request code_len=%d tests_len=%d timeout=%d",
                len(req.code or ""),
                len(req.tests or ""),
                int(max(1, req.timeout)),
            )
        except Exception:
            pass
        res = runner.run(req.code, req.tests, timeout=int(max(1, req.timeout)))
        try:
            _LOG.info(
                "sandbox: /run completed ok=%s stdout_len=%d stderr_len=%d",
                str(bool(res.ok)),
                len(res.stdout or ""),
                len(res.stderr or ""),
            )
        except Exception:
            pass
        return {"ok": bool(res.ok), "stdout": res.stdout, "stderr": res.stderr}

    @app.get("/healthz")
    def health() -> Dict[str, Any]:  # type: ignore[no-redef]
        try:
            _LOG.info("sandbox: /healthz")
        except Exception:
            pass
        return {"ok": True}

    # Alias for consistency with other services and external tooling
    @app.get("/health")
    def health_alias() -> Dict[str, Any]:  # type: ignore[no-redef]
        try:
            _LOG.info("sandbox: /health (alias)")
        except Exception:
            pass
        return {"ok": True}

    return app


def main() -> None:
    if FastAPI is None or uvicorn is None:
        raise RuntimeError("fastapi/uvicorn not installed in this environment")
    host = os.getenv("SANDBOX_HOST", "0.0.0.0")
    port = int(os.getenv("SANDBOX_PORT", "8088"))
    _LOG = get_logger("omnicoder.sandbox")
    try:
        _LOG.info("sandbox: starting uvicorn host=%s port=%d", host, port)
    except Exception:
        pass
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()


