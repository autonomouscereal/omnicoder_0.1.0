from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class PgConfig:
    host: str
    port: int
    database: str
    user: str
    password: str
    sslmode: str = "prefer"

    @classmethod
    def from_env(cls) -> "PgConfig":
        return cls(
            host=os.environ.get("OMNICODER_PGHOST", "127.0.0.1"),
            port=int(os.environ.get("OMNICODER_PGPORT", "5432")),
            database=os.environ.get("OMNICODER_PGDATABASE", "omnicoder"),
            user=os.environ.get("OMNICODER_PGUSER", "omnicoder"),
            password=os.environ.get("OMNICODER_PGPASSWORD", ""),
            sslmode=os.environ.get("OMNICODER_PGSSLMODE", "prefer"),
        )


def connect(cfg: PgConfig | None = None):
    import psycopg2

    c = cfg or PgConfig.from_env()
    return psycopg2.connect(
        host=c.host,
        port=c.port,
        dbname=c.database,
        user=c.user,
        password=c.password,
        sslmode=c.sslmode,
    )


@contextmanager
def transaction(cfg: PgConfig | None = None) -> Iterator[Any]:
    conn = connect(cfg)
    try:
        with conn:
            with conn.cursor() as cur:
                yield cur
    finally:
        conn.close()


def apply_sql_file(path: str, cfg: PgConfig | None = None) -> None:
    sql = Path(path).read_text(encoding="utf-8")
    with transaction(cfg) as cur:
        cur.execute(sql)


def insert_artifact(path: str, sha256: str, media_type: str, byte_size: int, metadata: dict[str, Any] | None = None, cfg: PgConfig | None = None) -> int:
    with transaction(cfg) as cur:
        cur.execute(
            """
            INSERT INTO artifacts (sha256, path, media_type, byte_size, metadata)
            VALUES (%s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (sha256) DO UPDATE
            SET path = EXCLUDED.path, media_type = EXCLUDED.media_type, byte_size = EXCLUDED.byte_size
            RETURNING artifact_id
            """,
            (sha256, path, media_type, int(byte_size), json.dumps(metadata or {})),
        )
        return int(cur.fetchone()[0])


def insert_dataset(
    name: str,
    namespace: str,
    source_uri: str | None = None,
    source_date: str | None = None,
    license_id: str | None = None,
    terms: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    cfg: PgConfig | None = None,
) -> int:
    with transaction(cfg) as cur:
        cur.execute(
            """
            INSERT INTO datasets (name, namespace, source_uri, source_date, license_id, terms_json, metadata)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
            RETURNING dataset_id
            """,
            (name, namespace, source_uri, source_date, license_id, json.dumps(terms or {}), json.dumps(metadata or {})),
        )
        return int(cur.fetchone()[0])


def insert_agent_run(
    trace_id: str,
    harness: str,
    dataset_id: int | None = None,
    model_name: str | None = None,
    task_family: str | None = None,
    prompt_hash: str | None = None,
    repo_sha: str | None = None,
    env_id: str | None = None,
    outcome: str | None = None,
    reward: float | None = None,
    metrics: dict[str, Any] | None = None,
    cfg: PgConfig | None = None,
) -> int:
    with transaction(cfg) as cur:
        cur.execute(
            """
            INSERT INTO agent_runs (
                dataset_id, trace_id, harness, model_name, task_family, prompt_hash,
                repo_sha, env_id, outcome, reward, metrics
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (trace_id) DO UPDATE
            SET outcome = EXCLUDED.outcome, reward = EXCLUDED.reward, metrics = EXCLUDED.metrics
            RETURNING agent_run_id
            """,
            (
                dataset_id,
                trace_id,
                harness,
                model_name,
                task_family,
                prompt_hash,
                repo_sha,
                env_id,
                outcome,
                reward,
                json.dumps(metrics or {}),
            ),
        )
        return int(cur.fetchone()[0])


def insert_agent_step(
    agent_run_id: int,
    step_index: int,
    role: str,
    action_type: str | None = None,
    content: str | None = None,
    tool_name: str | None = None,
    tool_input: dict[str, Any] | None = None,
    tool_output: dict[str, Any] | None = None,
    exit_code: int | None = None,
    latency_ms: int | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
    cfg: PgConfig | None = None,
) -> int:
    with transaction(cfg) as cur:
        cur.execute(
            """
            INSERT INTO agent_steps (
                agent_run_id, step_index, role, action_type, content, tool_name,
                tool_input, tool_output, exit_code, latency_ms, tokens_in, tokens_out,
                error, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (agent_run_id, step_index) DO UPDATE
            SET role = EXCLUDED.role, action_type = EXCLUDED.action_type, content = EXCLUDED.content,
                tool_name = EXCLUDED.tool_name, tool_input = EXCLUDED.tool_input,
                tool_output = EXCLUDED.tool_output, exit_code = EXCLUDED.exit_code,
                latency_ms = EXCLUDED.latency_ms, tokens_in = EXCLUDED.tokens_in,
                tokens_out = EXCLUDED.tokens_out, error = EXCLUDED.error, metadata = EXCLUDED.metadata
            RETURNING agent_step_id
            """,
            (
                int(agent_run_id),
                int(step_index),
                role,
                action_type,
                content,
                tool_name,
                json.dumps(tool_input or {}),
                json.dumps(tool_output or {}),
                exit_code,
                latency_ms,
                tokens_in,
                tokens_out,
                error,
                json.dumps(metadata or {}),
            ),
        )
        return int(cur.fetchone()[0])


def insert_training_example(
    bucket: str,
    input_json: dict[str, Any],
    target_json: dict[str, Any],
    split_name: str = "train",
    weight: float = 1.0,
    sample_id: int | None = None,
    artifact_id: int | None = None,
    agent_run_id: int | None = None,
    source_date: str | None = None,
    lineage: dict[str, Any] | None = None,
    cfg: PgConfig | None = None,
) -> int:
    with transaction(cfg) as cur:
        cur.execute(
            """
            INSERT INTO training_examples (
                bucket, sample_id, artifact_id, agent_run_id, input_json, target_json,
                weight, split_name, source_date, lineage
            )
            VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s::jsonb)
            RETURNING training_example_id
            """,
            (
                bucket,
                sample_id,
                artifact_id,
                agent_run_id,
                json.dumps(input_json),
                json.dumps(target_json),
                float(weight),
                split_name,
                source_date,
                json.dumps(lineage or {}),
            ),
        )
        return int(cur.fetchone()[0])


def enqueue_teacher_job(
    teacher_name: str,
    job_type: str,
    input_json: dict[str, Any],
    priority: int = 100,
    cfg: PgConfig | None = None,
) -> int:
    with transaction(cfg) as cur:
        cur.execute(
            """
            INSERT INTO teacher_jobs (teacher_name, job_type, input_json, priority)
            VALUES (%s, %s, %s::jsonb, %s)
            RETURNING teacher_job_id
            """,
            (teacher_name, job_type, json.dumps(input_json), int(priority)),
        )
        return int(cur.fetchone()[0])


def claim_teacher_job(teacher_name: str, worker_id: str, cfg: PgConfig | None = None) -> dict[str, Any] | None:
    with transaction(cfg) as cur:
        cur.execute(
            """
            UPDATE teacher_jobs
            SET status='running', locked_by=%s, locked_at=now(), updated_at=now()
            WHERE teacher_job_id = (
                SELECT teacher_job_id FROM teacher_jobs
                WHERE teacher_name=%s AND status='pending'
                ORDER BY priority, teacher_job_id
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING teacher_job_id, teacher_name, job_type, input_json, priority
            """,
            (worker_id, teacher_name),
        )
        row = cur.fetchone()
        if row is None:
            return None
        payload = row[3]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return {
            "teacher_job_id": int(row[0]),
            "teacher_name": row[1],
            "job_type": row[2],
            "input_json": payload,
            "priority": int(row[4]),
        }


def claim_work(stage: str, worker_id: str, cfg: PgConfig | None = None) -> dict[str, Any] | None:
    with transaction(cfg) as cur:
        cur.execute(
            """
            UPDATE work_queue
            SET status='running', locked_by=%s, locked_at=now(), updated_at=now()
            WHERE work_id = (
                SELECT work_id FROM work_queue
                WHERE stage=%s AND status='pending'
                ORDER BY work_id
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING work_id, stage, payload, status, created_at
            """,
            (worker_id, stage),
        )
        row = cur.fetchone()
        if row is None:
            return None
        payload = row[2]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return {"work_id": int(row[0]), "stage": row[1], "payload": payload, "status": row[3], "created_at": row[4].isoformat()}


def complete_work(work_id: int, status: str, metadata: dict[str, Any] | None = None, cfg: PgConfig | None = None) -> None:
    if status not in {"done", "failed"}:
        raise ValueError("status must be done or failed")
    with transaction(cfg) as cur:
        cur.execute(
            """
            UPDATE work_queue
            SET status=%s, updated_at=now(), payload = payload || %s::jsonb
            WHERE work_id=%s
            """,
            (status, json.dumps({"result": metadata or {}}), int(work_id)),
        )
