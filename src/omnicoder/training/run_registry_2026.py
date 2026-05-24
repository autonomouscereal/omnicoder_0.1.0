from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def utc_ms() -> int:
    return int(time.time() * 1000)


def stable_hash_file(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


class JsonlRunRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def manifest_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "run_manifest.json"

    def events_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "events.jsonl"

    def create_run(
        self,
        run_name: str,
        recipe: str,
        profile: str,
        preset: str,
        config: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        rid = run_id or f"{run_name}-{uuid.uuid4().hex[:10]}".replace(" ", "_")
        manifest = {
            "run_id": rid,
            "run_name": run_name,
            "recipe": recipe,
            "profile": profile,
            "preset": preset,
            "status": "created",
            "config": config,
            "metadata": metadata or {},
            "created_at_ms": utc_ms(),
            "updated_at_ms": utc_ms(),
            "started_at_ms": None,
            "finished_at_ms": None,
            "stages": {},
            "metrics": [],
            "artifacts": [],
        }
        write_json(self.manifest_path(rid), manifest)
        self.event(rid, "run_created", {"recipe": recipe, "profile": profile, "preset": preset})
        return manifest

    def load(self, run_id: str) -> dict[str, Any]:
        manifest = read_json(self.manifest_path(run_id))
        if manifest is None:
            raise FileNotFoundError(f"run manifest not found: {self.manifest_path(run_id)}")
        return manifest

    def save(self, manifest: dict[str, Any]) -> None:
        manifest["updated_at_ms"] = utc_ms()
        write_json(self.manifest_path(str(manifest["run_id"])), manifest)

    def update_status(self, run_id: str, status: str, error: str | None = None) -> dict[str, Any]:
        manifest = self.load(run_id)
        manifest["status"] = status
        now = utc_ms()
        if status == "running" and manifest.get("started_at_ms") is None:
            manifest["started_at_ms"] = now
        if status in {"completed", "failed"}:
            manifest["finished_at_ms"] = now
        if error:
            manifest.setdefault("metadata", {})["error"] = error
        self.save(manifest)
        self.event(run_id, "run_status", {"status": status, "error": error})
        return manifest

    def stage(
        self,
        run_id: str,
        stage_name: str,
        status: str,
        command: list[str] | None = None,
        log_path: str | None = None,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manifest = self.load(run_id)
        stage = manifest.setdefault("stages", {}).setdefault(stage_name, {"stage_name": stage_name})
        stage.update(
            {
                "status": status,
                "command": command or stage.get("command") or [],
                "log_path": log_path or stage.get("log_path"),
                "metrics": {**stage.get("metrics", {}), **(metrics or {})},
                "metadata": {**stage.get("metadata", {}), **(metadata or {})},
                "updated_at_ms": utc_ms(),
            }
        )
        if status == "running" and stage.get("started_at_ms") is None:
            stage["started_at_ms"] = utc_ms()
        if status in {"completed", "failed", "skipped"}:
            stage["finished_at_ms"] = utc_ms()
        self.save(manifest)
        self.event(run_id, "stage_status", stage)
        return stage

    def metric(
        self,
        run_id: str,
        name: str,
        value: float | int | None,
        step: int | None = None,
        stage_name: str | None = None,
        unit: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {
            "run_id": run_id,
            "stage_name": stage_name,
            "step": step,
            "name": name,
            "value": float(value) if value is not None else None,
            "unit": unit,
            "metadata": metadata or {},
            "created_at_ms": utc_ms(),
        }
        manifest = self.load(run_id)
        manifest.setdefault("metrics", []).append(record)
        self.save(manifest)
        self.event(run_id, "metric", record)
        return record

    def artifact(
        self,
        run_id: str,
        path: str,
        artifact_type: str,
        stage_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        p = Path(path)
        record = {
            "run_id": run_id,
            "stage_name": stage_name,
            "artifact_type": artifact_type,
            "path": str(p),
            "sha256": stable_hash_file(p),
            "byte_size": p.stat().st_size if p.exists() and p.is_file() else None,
            "metadata": metadata or {},
            "created_at_ms": utc_ms(),
        }
        manifest = self.load(run_id)
        manifest.setdefault("artifacts", []).append(record)
        self.save(manifest)
        self.event(run_id, "artifact", record)
        return record

    def event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        append_jsonl(self.events_path(run_id), {"event_type": event_type, "payload": payload, "created_at_ms": utc_ms()})


def _pg_connect():
    import psycopg2

    return psycopg2.connect(
        host=os.environ.get("OMNICODER_PGHOST", "127.0.0.1"),
        port=int(os.environ.get("OMNICODER_PGPORT", "5432")),
        dbname=os.environ.get("OMNICODER_PGDATABASE", "omnicoder"),
        user=os.environ.get("OMNICODER_PGUSER", "omnicoder"),
        password=os.environ.get("OMNICODER_PGPASSWORD", ""),
        sslmode=os.environ.get("OMNICODER_PGSSLMODE", "prefer"),
    )


@contextmanager
def pg_transaction() -> Iterator[Any]:
    conn = _pg_connect()
    try:
        with conn:
            with conn.cursor() as cur:
                yield cur
    finally:
        conn.close()


def apply_pg_schema(path: str = "schemas/training_runs_2026.sql") -> None:
    sql = Path(path).read_text(encoding="utf-8")
    with pg_transaction() as cur:
        cur.execute(sql)


def mirror_run_to_postgres(manifest: dict[str, Any]) -> None:
    with pg_transaction() as cur:
        cur.execute(
            """
            INSERT INTO training_runs_2026 (
                run_id, run_name, recipe, status, profile, preset, git_commit,
                data_manifest_sha256, config_json, metadata, started_at, finished_at
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,
                    to_timestamp(%s / 1000.0), to_timestamp(%s / 1000.0))
            ON CONFLICT (run_id) DO UPDATE
            SET status=EXCLUDED.status, config_json=EXCLUDED.config_json,
                metadata=EXCLUDED.metadata, updated_at=now(),
                started_at=EXCLUDED.started_at, finished_at=EXCLUDED.finished_at
            """,
            (
                manifest["run_id"],
                manifest["run_name"],
                manifest["recipe"],
                manifest["status"],
                manifest.get("profile"),
                manifest.get("preset"),
                manifest.get("metadata", {}).get("git_commit"),
                manifest.get("metadata", {}).get("data_manifest_sha256"),
                json.dumps(manifest.get("config", {})),
                json.dumps(manifest.get("metadata", {})),
                manifest.get("started_at_ms"),
                manifest.get("finished_at_ms"),
            ),
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Omnicoder 2026 training run registry")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("apply-schema").add_argument("--schema", default="schemas/training_runs_2026.sql")
    init = sub.add_parser("init")
    init.add_argument("--root", default="weights/runs_2026")
    init.add_argument("--run-name", required=True)
    init.add_argument("--recipe", default="native_pretrain")
    init.add_argument("--profile", default="training_harness_2026")
    init.add_argument("--preset", default="probe")
    init.add_argument("--config", default=None)
    init.add_argument("--run-id", default=None)
    status = sub.add_parser("status")
    status.add_argument("--root", default="weights/runs_2026")
    status.add_argument("--run-id", required=True)
    metric = sub.add_parser("metric")
    metric.add_argument("--root", default="weights/runs_2026")
    metric.add_argument("--run-id", required=True)
    metric.add_argument("--name", required=True)
    metric.add_argument("--value", type=float, required=True)
    metric.add_argument("--step", type=int)
    metric.add_argument("--stage")
    args = ap.parse_args()

    if args.cmd == "apply-schema":
        apply_pg_schema(args.schema)
        print(json.dumps({"status": "ok", "schema": args.schema}))
        return
    registry = JsonlRunRegistry(getattr(args, "root", "weights/runs_2026"))
    if args.cmd == "init":
        cfg = read_json(args.config, {}) if args.config else {}
        manifest = registry.create_run(args.run_name, args.recipe, args.profile, args.preset, cfg, run_id=args.run_id)
        print(json.dumps({"status": "ok", "run_id": manifest["run_id"], "manifest": str(registry.manifest_path(manifest["run_id"]))}))
        return
    if args.cmd == "status":
        print(json.dumps(registry.load(args.run_id), indent=2, ensure_ascii=True))
        return
    if args.cmd == "metric":
        print(json.dumps(registry.metric(args.run_id, args.name, args.value, args.step, args.stage), ensure_ascii=True))


if __name__ == "__main__":
    main()
