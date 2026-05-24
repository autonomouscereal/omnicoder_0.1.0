-- Omnicoder 2026 benchmark harness schema.
-- Raw PostgreSQL only. No ORM, SQLAlchemy, Pydantic, SQLite, or Chroma.

CREATE TABLE IF NOT EXISTS eval_runs (
  run_id UUID PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  harness_version TEXT NOT NULL,
  git_commit TEXT NOT NULL,
  model_provider TEXT NOT NULL,
  model_id TEXT NOT NULL,
  model_build TEXT,
  model_cutoff_date DATE,
  agent_id TEXT NOT NULL,
  agent_config JSONB NOT NULL,
  benchmark_manifest JSONB NOT NULL,
  contamination_policy JSONB NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('queued','running','passed','failed','invalid'))
);

CREATE TABLE IF NOT EXISTS eval_benchmark_registry (
  benchmark_id TEXT PRIMARY KEY,
  benchmark_version TEXT NOT NULL,
  adapter_kind TEXT NOT NULL,
  source_uri TEXT,
  reportable_split TEXT,
  holdout_policy JSONB NOT NULL DEFAULT '{}'::jsonb,
  release_gate TEXT NOT NULL,
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_task_results (
  result_id UUID PRIMARY KEY,
  run_id UUID NOT NULL REFERENCES eval_runs(run_id),
  benchmark TEXT NOT NULL,
  benchmark_version TEXT NOT NULL,
  split TEXT NOT NULL,
  task_id TEXT NOT NULL,
  task_revision TEXT NOT NULL,
  modalities TEXT[] NOT NULL,
  started_at TIMESTAMPTZ NOT NULL,
  finished_at TIMESTAMPTZ,
  status TEXT NOT NULL CHECK (status IN ('passed','failed','timeout','error','invalid')),
  canonical_score NUMERIC,
  score_json JSONB NOT NULL,
  metrics_json JSONB NOT NULL,
  cost_json JSONB NOT NULL,
  contamination_json JSONB NOT NULL,
  artifact_root TEXT NOT NULL,
  UNIQUE(run_id, benchmark, split, task_id, task_revision)
);

CREATE TABLE IF NOT EXISTS eval_step_events (
  event_id BIGSERIAL PRIMARY KEY,
  result_id UUID NOT NULL REFERENCES eval_task_results(result_id),
  step_index INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  input_sha256 TEXT,
  output_sha256 TEXT,
  tool_name TEXT,
  tool_args_json JSONB,
  observation_json JSONB,
  token_json JSONB,
  wall_ms INTEGER,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_artifacts (
  artifact_id UUID PRIMARY KEY,
  result_id UUID NOT NULL REFERENCES eval_task_results(result_id),
  kind TEXT NOT NULL,
  path TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  bytes BIGINT NOT NULL,
  mime_type TEXT,
  metadata_json JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_quarantine (
  quarantine_id UUID PRIMARY KEY,
  benchmark TEXT NOT NULL,
  artifact_sha256 TEXT NOT NULL,
  artifact_kind TEXT NOT NULL,
  reason TEXT NOT NULL,
  source_run_id UUID,
  source_result_id UUID,
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS eval_task_results_run_idx ON eval_task_results(run_id);
CREATE INDEX IF NOT EXISTS eval_task_results_benchmark_idx ON eval_task_results(benchmark, benchmark_version, split);
CREATE INDEX IF NOT EXISTS eval_step_events_result_idx ON eval_step_events(result_id, step_index);
CREATE INDEX IF NOT EXISTS eval_artifacts_result_idx ON eval_artifacts(result_id);
CREATE INDEX IF NOT EXISTS eval_quarantine_benchmark_idx ON eval_quarantine(benchmark, artifact_sha256);
