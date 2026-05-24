from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnicoder.data_factory import export_agent_memory_postgres_2026 as exporter


def test_build_select_sql_requires_date_column_when_enforced() -> None:
    with pytest.raises(ValueError):
        exporter.build_select_sql(
            schema="public",
            table="agent_memory_events",
            columns=["event_id", "content"],
            date_column=None,
            space_col=None,
            require_date_filter=True,
            all_spaces=True,
        )


def test_build_select_sql_parameterizes_date_and_space() -> None:
    sql, params = exporter.build_select_sql(
        schema="public",
        table="agent_memory_events",
        columns=["event_id", "created_at", "space", "content"],
        date_column="created_at",
        space_col="space",
        require_date_filter=True,
        all_spaces=False,
    )

    assert "created_at" in sql
    assert "LIMIT %s OFFSET %s" in sql
    assert params == ["__DATE_FLOOR__", "__SPACE__"]


def test_normalize_record_redacts_secret_fields() -> None:
    row = {
        "created_at": "2026-05-24T00:00:00Z",
        "event": "PostToolUse",
        "content": "token=abc123456789012345",
        "metadata": {"password": "super-secret"},
    }

    normalized = exporter.normalize_record(row)

    assert normalized["event_type"] == "PostToolUse"
    assert normalized["metadata"]["password"] == "<redacted>"
    assert "abc123456789012345" not in json.dumps(normalized)


def test_connection_limit_zero_means_unlimited(monkeypatch, tmp_path: Path) -> None:
    observed_limits: list[int] = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=()):
            if "information_schema.columns" in sql:
                return
            observed_limits.append(int(params[-2]))

        def fetchall(self):
            if not observed_limits:
                return [{"column_name": "created_at"}, {"column_name": "content"}]
            return []

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return Cursor()

        def close(self):
            return None

    monkeypatch.setattr(exporter, "connect", lambda cfg: Conn())

    result = exporter.export_rows({"limit": 0, "page_size": 123, "password": "x"}, tmp_path / "events.jsonl")

    assert result["limit"] == 0
    assert observed_limits == [123]
