from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from omnicoder.data_factory import memory_trace_collectors_2026


DEFAULT_OUT = "data/raw/agent_memory_events_2026.jsonl"
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SENSITIVE_KEY_PARTS = memory_trace_collectors_2026.SENSITIVE_KEY_PARTS


def quote_ident(value: str) -> str:
    if not IDENT_RE.match(value):
        raise ValueError(f"unsafe PostgreSQL identifier: {value!r}")
    return '"' + value.replace('"', '""') + '"'


def redact(value: Any) -> Any:
    return memory_trace_collectors_2026.redact(value)


def resolve_password(cfg: dict[str, Any]) -> str:
    env_names = [
        str(cfg.get("password_env") or ""),
        "MEMORY_DB_PASSWORD",
        "AGENT_MEMORY_DB_PASSWORD",
        "OMNICODER_AGENT_MEMORY_PGPASSWORD",
    ]
    for name in env_names:
        if name and os.environ.get(name):
            return str(os.environ[name])
    vault_key = str(cfg.get("vault_key") or cfg.get("credential_vault_key") or "")
    if not vault_key:
        return ""
    candidates = cfg.get("credman_candidates")
    if not isinstance(candidates, list):
        candidates = [
            "/home/cereal/.agents/skills/server-manager/credman.py",
            "/home/cereal/.codex/skills/server-manager/credman.py",
            "C:/Users/cereal/.agents/skills/server-manager/credman.py",
            "C:/Users/cereal/.Codex/skills/server-manager/credman.py",
        ]
    for raw in candidates:
        path = Path(str(raw)).expanduser()
        if not path.exists():
            continue
        result = subprocess.run(
            [sys.executable, str(path), "get", vault_key],
            capture_output=True,
            text=True,
            timeout=float(cfg.get("vault_timeout_seconds") or 30),
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return ""


def connection_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "host": str(os.environ.get("MEMORY_DB_HOST") or cfg.get("host") or "192.168.50.222"),
        "port": int(os.environ.get("MEMORY_DB_PORT") or cfg.get("port") or 25490),
        "dbname": str(os.environ.get("MEMORY_DB_NAME") or cfg.get("database") or cfg.get("dbname") or "agent_memory"),
        "user": str(os.environ.get("MEMORY_DB_USER") or cfg.get("user") or "agent_memory"),
        "password": resolve_password(cfg),
        "sslmode": str(os.environ.get("MEMORY_DB_SSLMODE") or cfg.get("sslmode") or "prefer"),
    }


def connect(cfg: dict[str, Any]):
    import psycopg2
    import psycopg2.extras

    conn_kwargs = connection_config(cfg)
    return psycopg2.connect(**conn_kwargs, cursor_factory=psycopg2.extras.RealDictCursor)


def table_columns(cur: Any, schema: str, table: str) -> list[str]:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """,
        (schema, table),
    )
    return [str(row["column_name"] if isinstance(row, dict) else row[0]) for row in cur.fetchall()]


def timestamp_column(columns: Sequence[str]) -> str | None:
    for name in ("created_at", "timestamp", "event_time", "ts", "time"):
        if name in columns:
            return name
    return None


def space_column(columns: Sequence[str]) -> str | None:
    for name in ("space", "memory_space", "namespace"):
        if name in columns:
            return name
    return None


def build_select_sql(
    *,
    schema: str,
    table: str,
    columns: Sequence[str],
    date_column: str | None,
    space_col: str | None,
    require_date_filter: bool,
    all_spaces: bool,
) -> tuple[str, list[Any]]:
    selected = ", ".join(quote_ident(name) for name in columns)
    sql = f"SELECT {selected} FROM {quote_ident(schema)}.{quote_ident(table)}"
    clauses: list[str] = []
    params: list[Any] = []
    if date_column:
        clauses.append(f"{quote_ident(date_column)} >= %s")
        params.append("__DATE_FLOOR__")
    elif require_date_filter:
        raise ValueError("agent_memory_events has no timestamp column for date_floor enforcement")
    if space_col and not all_spaces:
        clauses.append(f"{quote_ident(space_col)} = %s")
        params.append("__SPACE__")
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    order_col = date_column or columns[0]
    sql += f" ORDER BY {quote_ident(order_col)} ASC LIMIT %s OFFSET %s"
    return sql, params


def normalize_record(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    for key, value in list(payload.items()):
        if hasattr(value, "isoformat"):
            payload[key] = value.isoformat()
    payload.setdefault("event_type", payload.get("event") or payload.get("type") or payload.get("memory_kind"))
    payload.setdefault("content", payload.get("prompt") or payload.get("text") or payload.get("message"))
    payload.setdefault("source_uri", payload.get("source") or payload.get("source_path"))
    return redact(payload)


def export_rows(cfg: dict[str, Any], out_path: Path) -> dict[str, Any]:
    schema = str(cfg.get("schema") or "public")
    table = str(cfg.get("table") or "agent_memory_events")
    date_floor = str(cfg.get("date_floor") or "2025-01-01")
    raw_limit = cfg.get("limit", 12000)
    limit = 12000 if raw_limit in (None, "") else int(raw_limit)
    page_size = max(1, int(cfg.get("page_size") or 1000))
    all_spaces = bool(cfg.get("all_spaces", True))
    requested_space = str(cfg.get("space") or "")
    require_date_filter = bool(cfg.get("require_date_filter", True))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(cfg)
    count = 0
    try:
        with conn:
            with conn.cursor() as cur:
                columns = table_columns(cur, schema, table)
                if not columns:
                    raise RuntimeError(f"no columns discovered for {schema}.{table}")
                date_col = timestamp_column(columns)
                space_col = space_column(columns)
                sql, sentinel_params = build_select_sql(
                    schema=schema,
                    table=table,
                    columns=columns,
                    date_column=date_col,
                    space_col=space_col,
                    require_date_filter=require_date_filter,
                    all_spaces=all_spaces,
                )
                offset = 0
                with out_path.open("w", encoding="utf-8") as handle:
                    while True:
                        remaining = limit - count if limit > 0 else page_size
                        if remaining <= 0:
                            break
                        batch_size = min(page_size, remaining) if limit > 0 else page_size
                        params: list[Any] = []
                        for item in sentinel_params:
                            if item == "__DATE_FLOOR__":
                                params.append(date_floor)
                            elif item == "__SPACE__":
                                params.append(requested_space)
                        params.extend([batch_size, offset])
                        cur.execute(sql, tuple(params))
                        rows = cur.fetchall()
                        if not rows:
                            break
                        for row in rows:
                            item = dict(row)
                            handle.write(json.dumps(normalize_record(item), ensure_ascii=True, sort_keys=True, default=str) + "\n")
                            count += 1
                        offset += len(rows)
                        if len(rows) < batch_size:
                            break
    finally:
        conn.close()
    return {
        "status": "ok",
        "out": str(out_path),
        "records": count,
        "table": f"{schema}.{table}",
        "date_floor": date_floor,
        "all_spaces": all_spaces,
        "space": requested_space or None,
        "limit": limit,
    }


def load_profile_cfg(profile_path: Path | None) -> dict[str, Any]:
    if profile_path is None:
        return {}
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    cfg = payload.get("agent_memory_postgres_export")
    if isinstance(cfg, dict):
        return dict(cfg)
    builder = payload.get("builder_2026")
    if isinstance(builder, dict) and isinstance(builder.get("agent_memory_postgres_export"), dict):
        return dict(builder["agent_memory_postgres_export"])
    return {}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export agent-memory PostgreSQL events to redacted JSONL using raw psycopg2")
    parser.add_argument("--profile")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--date-floor")
    parser.add_argument("--space")
    parser.add_argument("--all-spaces", action="store_true")
    parser.add_argument("--table")
    parser.add_argument("--schema")
    args = parser.parse_args(argv)

    cfg = load_profile_cfg(Path(args.profile)) if args.profile else {}
    for key in ("limit", "date_floor", "space", "table", "schema"):
        value = getattr(args, key)
        if value not in (None, ""):
            cfg[key] = value
    if args.all_spaces:
        cfg["all_spaces"] = True
    result = export_rows(cfg, Path(args.out))
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
