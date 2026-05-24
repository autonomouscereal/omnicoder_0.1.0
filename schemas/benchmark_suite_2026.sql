-- Omnicoder 2026 comprehensive benchmark suite registry/schema.
-- Raw PostgreSQL mirror for JSONL benchmark events.

CREATE TABLE IF NOT EXISTS benchmark_suites_2026 (
    suite_id TEXT PRIMARY KEY,
    suite_version TEXT NOT NULL,
    profile_path TEXT NOT NULL,
    storage_contract JSONB NOT NULL DEFAULT '{}'::jsonb,
    record_contract JSONB NOT NULL DEFAULT '{}'::jsonb,
    scoring_policy JSONB NOT NULL DEFAULT '{}'::jsonb,
    contamination_controls JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS benchmark_axes_2026 (
    axis_id TEXT PRIMARY KEY,
    suite_id TEXT NOT NULL REFERENCES benchmark_suites_2026(suite_id) ON DELETE CASCADE,
    display_name TEXT NOT NULL,
    purpose TEXT NOT NULL,
    canonical_weight NUMERIC(6,5) NOT NULL CHECK (canonical_weight >= 0 AND canonical_weight <= 1),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS benchmark_definitions_2026 (
    benchmark_id TEXT PRIMARY KEY,
    suite_id TEXT NOT NULL REFERENCES benchmark_suites_2026(suite_id) ON DELETE CASCADE,
    axis_id TEXT NOT NULL REFERENCES benchmark_axes_2026(axis_id) ON DELETE RESTRICT,
    adapter_kind TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    task_format TEXT NOT NULL,
    modalities TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    splits JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics JSONB NOT NULL DEFAULT '[]'::jsonb,
    holdout_policy JSONB NOT NULL DEFAULT '[]'::jsonb,
    release_gate TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS benchmark_release_gates_2026 (
    gate_id TEXT PRIMARY KEY,
    suite_id TEXT NOT NULL REFERENCES benchmark_suites_2026(suite_id) ON DELETE CASCADE,
    required_benchmarks TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    required_checks JSONB NOT NULL DEFAULT '[]'::jsonb,
    failure_policy TEXT NOT NULL DEFAULT 'fail_closed' CHECK (failure_policy IN ('fail_closed','warn','manual_review')),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS benchmark_jsonl_manifests_2026 (
    manifest_id UUID PRIMARY KEY,
    suite_id TEXT NOT NULL REFERENCES benchmark_suites_2026(suite_id) ON DELETE CASCADE,
    manifest_path TEXT NOT NULL,
    jsonl_path TEXT NOT NULL,
    record_count BIGINT NOT NULL CHECK (record_count >= 0),
    sha256 TEXT NOT NULL,
    generated_by TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(suite_id, jsonl_path, sha256)
);

CREATE TABLE IF NOT EXISTS benchmark_runs_2026 (
    run_id UUID PRIMARY KEY,
    suite_id TEXT NOT NULL REFERENCES benchmark_suites_2026(suite_id) ON DELETE RESTRICT,
    suite_version TEXT NOT NULL,
    run_cycle TEXT NOT NULL CHECK (run_cycle IN ('smoke','nightly','release','ad_hoc')),
    run_label TEXT NOT NULL,
    model_provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_build TEXT,
    model_cutoff_date DATE,
    agent_id TEXT,
    agent_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    environment JSONB NOT NULL DEFAULT '{}'::jsonb,
    container_digest TEXT,
    git_commit TEXT,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','running','passed','failed','invalid','cancelled')),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    canonical_score NUMERIC,
    score_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS benchmark_task_results_2026 (
    result_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES benchmark_runs_2026(run_id) ON DELETE CASCADE,
    benchmark_id TEXT NOT NULL REFERENCES benchmark_definitions_2026(benchmark_id) ON DELETE RESTRICT,
    task_id TEXT NOT NULL,
    task_revision TEXT NOT NULL,
    split TEXT NOT NULL,
    phase TEXT NOT NULL CHECK (phase IN ('setup','inference','tool_use','scoring','audit')),
    modalities TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    input_sha256 TEXT NOT NULL,
    output_sha256 TEXT,
    status TEXT NOT NULL CHECK (status IN ('passed','failed','timeout','error','invalid','skipped')),
    canonical_score NUMERIC,
    score_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    contamination_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    cost_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    artifact_root TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(run_id, benchmark_id, task_id, task_revision, split, phase)
);

CREATE TABLE IF NOT EXISTS benchmark_step_events_2026 (
    event_id BIGSERIAL PRIMARY KEY,
    result_id UUID NOT NULL REFERENCES benchmark_task_results_2026(result_id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL CHECK (step_index >= 0),
    event_type TEXT NOT NULL,
    tool_name TEXT,
    tool_args JSONB NOT NULL DEFAULT '{}'::jsonb,
    observation JSONB NOT NULL DEFAULT '{}'::jsonb,
    input_sha256 TEXT,
    output_sha256 TEXT,
    token_usage JSONB NOT NULL DEFAULT '{}'::jsonb,
    wall_ms INTEGER CHECK (wall_ms IS NULL OR wall_ms >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(result_id, step_index)
);

CREATE TABLE IF NOT EXISTS benchmark_artifacts_2026 (
    artifact_id UUID PRIMARY KEY,
    result_id UUID REFERENCES benchmark_task_results_2026(result_id) ON DELETE CASCADE,
    run_id UUID REFERENCES benchmark_runs_2026(run_id) ON DELETE CASCADE,
    benchmark_id TEXT REFERENCES benchmark_definitions_2026(benchmark_id) ON DELETE RESTRICT,
    artifact_kind TEXT NOT NULL,
    uri TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    bytes BIGINT NOT NULL CHECK (bytes >= 0),
    mime_type TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (result_id IS NOT NULL OR run_id IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS benchmark_contamination_audit_2026 (
    audit_id UUID PRIMARY KEY,
    run_id UUID REFERENCES benchmark_runs_2026(run_id) ON DELETE CASCADE,
    result_id UUID REFERENCES benchmark_task_results_2026(result_id) ON DELETE CASCADE,
    benchmark_id TEXT REFERENCES benchmark_definitions_2026(benchmark_id) ON DELETE RESTRICT,
    dataset_revision TEXT,
    release_date DATE,
    model_cutoff_date DATE,
    public_dev_allowed BOOLEAN NOT NULL DEFAULT false,
    hidden_material_exposed BOOLEAN NOT NULL DEFAULT false,
    trajectory_quarantined BOOLEAN NOT NULL DEFAULT false,
    audit_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS benchmark_quarantine_2026 (
    quarantine_id UUID PRIMARY KEY,
    suite_id TEXT NOT NULL REFERENCES benchmark_suites_2026(suite_id) ON DELETE CASCADE,
    benchmark_id TEXT REFERENCES benchmark_definitions_2026(benchmark_id) ON DELETE RESTRICT,
    source_run_id UUID REFERENCES benchmark_runs_2026(run_id) ON DELETE SET NULL,
    source_result_id UUID REFERENCES benchmark_task_results_2026(result_id) ON DELETE SET NULL,
    artifact_sha256 TEXT NOT NULL,
    artifact_kind TEXT NOT NULL,
    reason TEXT NOT NULL,
    release_after DATE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS benchmark_performance_samples_2026 (
    sample_id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES benchmark_runs_2026(run_id) ON DELETE CASCADE,
    benchmark_id TEXT REFERENCES benchmark_definitions_2026(benchmark_id) ON DELETE RESTRICT,
    hardware_profile TEXT NOT NULL,
    runtime_profile TEXT NOT NULL,
    batch_size INTEGER CHECK (batch_size IS NULL OR batch_size > 0),
    input_tokens INTEGER CHECK (input_tokens IS NULL OR input_tokens >= 0),
    output_tokens INTEGER CHECK (output_tokens IS NULL OR output_tokens >= 0),
    time_to_first_token_ms INTEGER CHECK (time_to_first_token_ms IS NULL OR time_to_first_token_ms >= 0),
    tokens_per_second NUMERIC,
    requests_per_second NUMERIC,
    p50_latency_ms INTEGER CHECK (p50_latency_ms IS NULL OR p50_latency_ms >= 0),
    p95_latency_ms INTEGER CHECK (p95_latency_ms IS NULL OR p95_latency_ms >= 0),
    p99_latency_ms INTEGER CHECK (p99_latency_ms IS NULL OR p99_latency_ms >= 0),
    peak_vram_mb INTEGER CHECK (peak_vram_mb IS NULL OR peak_vram_mb >= 0),
    peak_ram_mb INTEGER CHECK (peak_ram_mb IS NULL OR peak_ram_mb >= 0),
    watts_per_request NUMERIC,
    error_rate NUMERIC CHECK (error_rate IS NULL OR (error_rate >= 0 AND error_rate <= 1)),
    sample_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_benchmark_axes_suite_2026 ON benchmark_axes_2026(suite_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_definitions_suite_axis_2026 ON benchmark_definitions_2026(suite_id, axis_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_definitions_release_gate_2026 ON benchmark_definitions_2026(release_gate);
CREATE INDEX IF NOT EXISTS idx_benchmark_jsonl_manifests_suite_2026 ON benchmark_jsonl_manifests_2026(suite_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_suite_status_2026 ON benchmark_runs_2026(suite_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_model_2026 ON benchmark_runs_2026(model_provider, model_id, model_build);
CREATE INDEX IF NOT EXISTS idx_benchmark_task_results_run_2026 ON benchmark_task_results_2026(run_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_task_results_benchmark_2026 ON benchmark_task_results_2026(benchmark_id, split, status);
CREATE INDEX IF NOT EXISTS idx_benchmark_step_events_result_2026 ON benchmark_step_events_2026(result_id, step_index);
CREATE INDEX IF NOT EXISTS idx_benchmark_artifacts_result_2026 ON benchmark_artifacts_2026(result_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_artifacts_hash_2026 ON benchmark_artifacts_2026(sha256);
CREATE INDEX IF NOT EXISTS idx_benchmark_contamination_run_2026 ON benchmark_contamination_audit_2026(run_id, hidden_material_exposed);
CREATE INDEX IF NOT EXISTS idx_benchmark_quarantine_hash_2026 ON benchmark_quarantine_2026(suite_id, artifact_sha256);
CREATE INDEX IF NOT EXISTS idx_benchmark_performance_run_2026 ON benchmark_performance_samples_2026(run_id, hardware_profile, runtime_profile);
