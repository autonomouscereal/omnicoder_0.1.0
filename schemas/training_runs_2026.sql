-- Omnicoder 2026 training run registry.
-- Raw PostgreSQL only. No ORM, no Pydantic, no SQLAlchemy, no SQLite.

CREATE TABLE IF NOT EXISTS training_runs_2026 (
    run_id TEXT PRIMARY KEY,
    run_name TEXT NOT NULL,
    recipe TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('created', 'running', 'completed', 'failed', 'paused')),
    profile TEXT,
    preset TEXT,
    git_commit TEXT,
    data_manifest_sha256 TEXT,
    config_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS training_stages_2026 (
    stage_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES training_runs_2026(run_id) ON DELETE CASCADE,
    stage_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('planned', 'running', 'completed', 'failed', 'skipped')),
    command_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    log_path TEXT,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (run_id, stage_name)
);

CREATE TABLE IF NOT EXISTS training_metrics_2026 (
    metric_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES training_runs_2026(run_id) ON DELETE CASCADE,
    stage_name TEXT,
    step BIGINT,
    name TEXT NOT NULL,
    value DOUBLE PRECISION,
    unit TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS training_artifacts_2026 (
    artifact_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES training_runs_2026(run_id) ON DELETE CASCADE,
    stage_name TEXT,
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT,
    byte_size BIGINT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_training_metrics_2026_run_step ON training_metrics_2026(run_id, step, name);
CREATE INDEX IF NOT EXISTS idx_training_stages_2026_run ON training_stages_2026(run_id, stage_name);
CREATE INDEX IF NOT EXISTS idx_training_artifacts_2026_run ON training_artifacts_2026(run_id, artifact_type);
