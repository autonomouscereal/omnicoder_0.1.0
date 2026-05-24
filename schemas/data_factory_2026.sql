-- Omnicoder 2026 raw PostgreSQL data/eval factory.
-- No ORM, no Pydantic, no SQLAlchemy, no SQLite.

CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    namespace TEXT NOT NULL CHECK (namespace IN ('train', 'eval_protected', 'synthetic', 'trace', 'quarantine')),
    source_uri TEXT,
    source_date DATE,
    license_id TEXT,
    terms_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

ALTER TABLE datasets ADD COLUMN IF NOT EXISTS source_date DATE;
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS terms_json JSONB NOT NULL DEFAULT '{}'::jsonb;
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'datasets_source_date_2025_chk'
    ) THEN
        ALTER TABLE datasets
        ADD CONSTRAINT datasets_source_date_2025_chk
        CHECK (source_date IS NULL OR source_date >= DATE '2025-01-01') NOT VALID;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id BIGSERIAL PRIMARY KEY,
    sha256 CHAR(64) NOT NULL UNIQUE,
    path TEXT NOT NULL,
    media_type TEXT NOT NULL,
    byte_size BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS media_segments (
    media_segment_id BIGSERIAL PRIMARY KEY,
    artifact_id BIGINT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
    segment_index INTEGER NOT NULL,
    start_ms BIGINT,
    end_ms BIGINT,
    transcript TEXT,
    caption TEXT,
    ledger_token_count INTEGER,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (artifact_id, segment_index)
);

CREATE INDEX IF NOT EXISTS idx_media_segments_artifact ON media_segments(artifact_id, segment_index);

CREATE TABLE IF NOT EXISTS samples (
    sample_id BIGSERIAL PRIMARY KEY,
    dataset_id BIGINT NOT NULL REFERENCES datasets(dataset_id),
    sample_hash CHAR(64) NOT NULL UNIQUE,
    modality_set TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    text_content TEXT,
    ledger_token_count INTEGER,
    quality_score DOUBLE PRECISION,
    train_tier TEXT NOT NULL DEFAULT 'candidate',
    contamination_status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_samples_dataset ON samples(dataset_id);
CREATE INDEX IF NOT EXISTS idx_samples_text_trgm ON samples USING gin (text_content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_samples_metadata ON samples USING gin (metadata);

CREATE TABLE IF NOT EXISTS sample_artifacts (
    sample_id BIGINT NOT NULL REFERENCES samples(sample_id) ON DELETE CASCADE,
    artifact_id BIGINT NOT NULL REFERENCES artifacts(artifact_id),
    role TEXT NOT NULL,
    PRIMARY KEY (sample_id, artifact_id, role)
);

CREATE TABLE IF NOT EXISTS trace_sessions (
    trace_session_id BIGSERIAL PRIMARY KEY,
    harness TEXT NOT NULL,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    repo_path TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS agent_runs (
    agent_run_id BIGSERIAL PRIMARY KEY,
    dataset_id BIGINT REFERENCES datasets(dataset_id),
    trace_id TEXT NOT NULL UNIQUE,
    harness TEXT NOT NULL,
    model_name TEXT,
    task_family TEXT,
    prompt_hash CHAR(64),
    repo_sha TEXT,
    env_id TEXT,
    outcome TEXT,
    reward DOUBLE PRECISION,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_steps (
    agent_step_id BIGSERIAL PRIMARY KEY,
    agent_run_id BIGINT NOT NULL REFERENCES agent_runs(agent_run_id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL,
    role TEXT NOT NULL,
    action_type TEXT,
    content TEXT,
    tool_name TEXT,
    tool_input JSONB,
    tool_output JSONB,
    exit_code INTEGER,
    latency_ms INTEGER,
    tokens_in INTEGER,
    tokens_out INTEGER,
    error TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (agent_run_id, step_index)
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_harness ON agent_runs(harness, task_family);
CREATE INDEX IF NOT EXISTS idx_agent_steps_run ON agent_steps(agent_run_id, step_index);

CREATE TABLE IF NOT EXISTS trace_events (
    trace_event_id BIGSERIAL PRIMARY KEY,
    trace_session_id BIGINT REFERENCES trace_sessions(trace_session_id) ON DELETE CASCADE,
    event_index INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    content TEXT,
    tool_name TEXT,
    exit_code INTEGER,
    artifact_id BIGINT REFERENCES artifacts(artifact_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE(trace_session_id, event_index)
);

CREATE TABLE IF NOT EXISTS teacher_runs (
    teacher_run_id BIGSERIAL PRIMARY KEY,
    teacher_model TEXT NOT NULL,
    endpoint TEXT,
    prompt_hash CHAR(64) NOT NULL,
    params JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS teacher_jobs (
    teacher_job_id BIGSERIAL PRIMARY KEY,
    teacher_name TEXT NOT NULL,
    job_type TEXT NOT NULL,
    input_json JSONB NOT NULL,
    output_json JSONB,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'done', 'failed', 'quarantine')),
    priority INTEGER NOT NULL DEFAULT 100,
    locked_by TEXT,
    locked_at TIMESTAMPTZ,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_teacher_jobs_status ON teacher_jobs(status, priority, teacher_job_id);

CREATE TABLE IF NOT EXISTS synthetic_candidates (
    synthetic_id BIGSERIAL PRIMARY KEY,
    source_sample_id BIGINT REFERENCES samples(sample_id),
    teacher_run_id BIGINT NOT NULL REFERENCES teacher_runs(teacher_run_id),
    candidate_hash CHAR(64) NOT NULL UNIQUE,
    content TEXT NOT NULL,
    verifier_status TEXT NOT NULL DEFAULT 'pending',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS verifier_runs (
    verifier_run_id BIGSERIAL PRIMARY KEY,
    sample_id BIGINT REFERENCES samples(sample_id),
    synthetic_id BIGINT REFERENCES synthetic_candidates(synthetic_id),
    verifier_name TEXT NOT NULL,
    passed BOOLEAN NOT NULL,
    score DOUBLE PRECISION,
    evidence_artifact_id BIGINT REFERENCES artifacts(artifact_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS quality_scores (
    quality_score_id BIGSERIAL PRIMARY KEY,
    target_type TEXT NOT NULL,
    target_id BIGINT NOT NULL,
    scorer TEXT NOT NULL,
    score_name TEXT NOT NULL,
    score_value DOUBLE PRECISION,
    label TEXT,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    scored_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_quality_scores_lookup ON quality_scores(target_type, target_id, score_name);

CREATE TABLE IF NOT EXISTS training_examples (
    training_example_id BIGSERIAL PRIMARY KEY,
    bucket TEXT NOT NULL,
    sample_id BIGINT REFERENCES samples(sample_id),
    artifact_id BIGINT REFERENCES artifacts(artifact_id),
    agent_run_id BIGINT REFERENCES agent_runs(agent_run_id),
    input_json JSONB NOT NULL,
    target_json JSONB NOT NULL,
    weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    split_name TEXT NOT NULL CHECK (split_name IN ('train', 'validation', 'eval_holdout', 'quarantine')),
    source_date DATE,
    lineage JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (source_date IS NULL OR source_date >= DATE '2025-01-01')
);

CREATE INDEX IF NOT EXISTS idx_training_examples_bucket ON training_examples(split_name, bucket, weight);
CREATE INDEX IF NOT EXISTS idx_training_examples_lineage ON training_examples USING gin (lineage);

CREATE TABLE IF NOT EXISTS contamination_matches (
    contamination_id BIGSERIAL PRIMARY KEY,
    sample_id BIGINT NOT NULL REFERENCES samples(sample_id) ON DELETE CASCADE,
    benchmark_name TEXT NOT NULL,
    match_type TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    protected_artifact_id BIGINT REFERENCES artifacts(artifact_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_contamination_matches_sample_score ON contamination_matches(sample_id, score DESC);

CREATE TABLE IF NOT EXISTS split_assignments (
    sample_id BIGINT PRIMARY KEY REFERENCES samples(sample_id) ON DELETE CASCADE,
    split_name TEXT NOT NULL CHECK (split_name IN ('train', 'validation', 'holdout', 'eval_protected', 'rejected')),
    reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_split_assignments_split ON split_assignments(split_name, created_at);

CREATE TABLE IF NOT EXISTS export_manifests (
    export_id BIGSERIAL PRIMARY KEY,
    export_name TEXT NOT NULL,
    export_kind TEXT NOT NULL,
    output_path TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_export_manifests_kind_created ON export_manifests(export_kind, created_at DESC);

CREATE TABLE IF NOT EXISTS work_queue (
    work_id BIGSERIAL PRIMARY KEY,
    stage TEXT NOT NULL,
    payload JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'done', 'failed')),
    locked_by TEXT,
    locked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_work_queue_claim ON work_queue(stage, status, work_id);

-- Worker claim pattern:
-- UPDATE work_queue
-- SET status='running', locked_by=$1, locked_at=now(), updated_at=now()
-- WHERE work_id = (
--   SELECT work_id FROM work_queue
--   WHERE stage=$2 AND status='pending'
--   ORDER BY work_id
--   FOR UPDATE SKIP LOCKED
--   LIMIT 1
-- )
-- RETURNING *;
