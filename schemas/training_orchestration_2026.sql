-- Omnicoder 2026 automated multimodal training orchestration schema.
-- JSONL is the source of truth; this is an optional raw PostgreSQL mirror.
-- Raw PostgreSQL only. No ORM, no Pydantic, no SQLAlchemy, no SQLite, no Chroma.

CREATE TABLE IF NOT EXISTS training_orchestration_profiles_2026 (
    profile_name TEXT PRIMARY KEY,
    profile_version TEXT NOT NULL,
    profile_path TEXT NOT NULL,
    storage_contract JSONB NOT NULL DEFAULT '{}'::jsonb,
    record_contracts JSONB NOT NULL DEFAULT '{}'::jsonb,
    ai_server JSONB NOT NULL DEFAULT '{}'::jsonb,
    curation_policy JSONB NOT NULL DEFAULT '{}'::jsonb,
    promotion_policy JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS training_orchestration_runs_2026 (
    run_id TEXT PRIMARY KEY,
    profile_name TEXT NOT NULL REFERENCES training_orchestration_profiles_2026(profile_name) ON DELETE RESTRICT,
    run_name TEXT NOT NULL,
    run_kind TEXT NOT NULL CHECK (run_kind IN ('curation','distillation','sft','preference','q4_recovery','benchmark_gate','full_orchestration')),
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','claimed','running','paused','passed','failed','cancelled','quarantined')),
    model_name TEXT,
    model_build TEXT,
    base_model TEXT,
    target_quantization TEXT,
    jsonl_root TEXT NOT NULL,
    artifact_root TEXT NOT NULL,
    scheduler_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    config_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    gate_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    claimed_by TEXT,
    claimed_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS training_orchestration_stages_2026 (
    stage_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES training_orchestration_runs_2026(run_id) ON DELETE CASCADE,
    stage_name TEXT NOT NULL,
    stage_order INTEGER NOT NULL CHECK (stage_order >= 0),
    stage_kind TEXT NOT NULL CHECK (stage_kind IN ('jsonl_ingest','curation','distillation','training','q4_recovery','loss_check','benchmark_gate','promotion','artifact_hash','scheduler')),
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','claimed','running','passed','failed','skipped','cancelled','quarantined')),
    input_jsonl TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    output_jsonl TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    command_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    claimed_by TEXT,
    claimed_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, stage_name)
);

CREATE TABLE IF NOT EXISTS training_jsonl_manifests_2026 (
    manifest_id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES training_orchestration_runs_2026(run_id) ON DELETE CASCADE,
    stage_id TEXT REFERENCES training_orchestration_stages_2026(stage_id) ON DELETE SET NULL,
    manifest_kind TEXT NOT NULL CHECK (manifest_kind IN ('raw_events','normalized','curated','teacher_jobs','teacher_responses','loss_samples','benchmark_results','gate_decisions','scheduler_events','artifact_hashes')),
    jsonl_path TEXT NOT NULL,
    record_count BIGINT NOT NULL CHECK (record_count >= 0),
    sha256 TEXT NOT NULL,
    byte_size BIGINT NOT NULL CHECK (byte_size >= 0),
    schema_name TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (jsonl_path, sha256)
);

CREATE TABLE IF NOT EXISTS training_orchestration_events_2026 (
    event_id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES training_orchestration_runs_2026(run_id) ON DELETE CASCADE,
    stage_id TEXT REFERENCES training_orchestration_stages_2026(stage_id) ON DELETE SET NULL,
    manifest_id TEXT REFERENCES training_jsonl_manifests_2026(manifest_id) ON DELETE SET NULL,
    event_type TEXT NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    record_id TEXT,
    modality TEXT CHECK (modality IS NULL OR modality IN ('text','code','tool','image','video','audio','music','long_context','multimodal')),
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    artifact_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    sha256 TEXT NOT NULL,
    source_jsonl_path TEXT,
    source_line BIGINT CHECK (source_line IS NULL OR source_line > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS training_curated_records_2026 (
    record_id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES training_orchestration_runs_2026(run_id) ON DELETE CASCADE,
    source_id TEXT NOT NULL,
    source_uri TEXT,
    source_date DATE,
    source_license TEXT NOT NULL DEFAULT 'unknown',
    modalities TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    primary_modality TEXT NOT NULL CHECK (primary_modality IN ('text','code','tool','image','video','audio','music','long_context','multimodal')),
    split TEXT NOT NULL CHECK (split IN ('train','validation','eval_holdout','quarantine','rejected')),
    quality_score DOUBLE PRECISION NOT NULL DEFAULT 0.0 CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
    contamination_status TEXT NOT NULL DEFAULT 'unknown' CHECK (contamination_status IN ('clean','suspect','contaminated','unknown')),
    dedupe_key TEXT,
    payload_sha256 TEXT NOT NULL,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    curation_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    artifact_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    rejection_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (source_date IS NULL OR source_date >= DATE '2025-01-01')
);

CREATE TABLE IF NOT EXISTS training_artifacts_2026 (
    artifact_id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES training_orchestration_runs_2026(run_id) ON DELETE CASCADE,
    stage_id TEXT REFERENCES training_orchestration_stages_2026(stage_id) ON DELETE SET NULL,
    record_id TEXT REFERENCES training_curated_records_2026(record_id) ON DELETE SET NULL,
    artifact_kind TEXT NOT NULL CHECK (artifact_kind IN ('jsonl','manifest','checkpoint','adapter','q4_export','image','video','audio','music','log','metric','benchmark_result','dataset_card','model_card','other')),
    uri TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    byte_size BIGINT NOT NULL CHECK (byte_size >= 0),
    mime_type TEXT,
    storage_backend TEXT NOT NULL DEFAULT 'filesystem',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (uri, sha256)
);

CREATE TABLE IF NOT EXISTS training_distillation_jobs_2026 (
    job_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES training_orchestration_runs_2026(run_id) ON DELETE CASCADE,
    record_id TEXT REFERENCES training_curated_records_2026(record_id) ON DELETE SET NULL,
    teacher_name TEXT NOT NULL,
    job_type TEXT NOT NULL,
    modalities TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    priority INTEGER NOT NULL DEFAULT 100,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','claimed','running','completed','failed','cancelled','quarantined')),
    prompt_sha256 TEXT NOT NULL,
    response_sha256 TEXT,
    prompt_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    response_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    score_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    claimed_by TEXT,
    claimed_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS training_loss_samples_2026 (
    sample_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES training_orchestration_runs_2026(run_id) ON DELETE CASCADE,
    stage_id TEXT REFERENCES training_orchestration_stages_2026(stage_id) ON DELETE SET NULL,
    step BIGINT NOT NULL CHECK (step >= 0),
    epoch DOUBLE PRECISION,
    split TEXT NOT NULL DEFAULT 'train' CHECK (split IN ('train','validation','eval_holdout')),
    loss_name TEXT NOT NULL,
    loss_value DOUBLE PRECISION NOT NULL,
    learning_rate DOUBLE PRECISION,
    gradient_norm DOUBLE PRECISION,
    tokens_seen BIGINT CHECK (tokens_seen IS NULL OR tokens_seen >= 0),
    samples_seen BIGINT CHECK (samples_seen IS NULL OR samples_seen >= 0),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, stage_id, step, split, loss_name)
);

CREATE TABLE IF NOT EXISTS training_loss_trend_checks_2026 (
    check_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES training_orchestration_runs_2026(run_id) ON DELETE CASCADE,
    stage_id TEXT REFERENCES training_orchestration_stages_2026(stage_id) ON DELETE SET NULL,
    window_start_step BIGINT NOT NULL CHECK (window_start_step >= 0),
    window_end_step BIGINT NOT NULL CHECK (window_end_step >= window_start_step),
    status TEXT NOT NULL CHECK (status IN ('passed','failed','warn','insufficient_samples')),
    smoothed_loss_start DOUBLE PRECISION,
    smoothed_loss_end DOUBLE PRECISION,
    relative_change DOUBLE PRECISION,
    max_gradient_norm DOUBLE PRECISION,
    nan_or_inf_detected BOOLEAN NOT NULL DEFAULT false,
    check_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS training_benchmark_gate_runs_2026 (
    gate_run_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES training_orchestration_runs_2026(run_id) ON DELETE CASCADE,
    benchmark_suite_id TEXT NOT NULL,
    benchmark_run_ref TEXT,
    cycle TEXT NOT NULL CHECK (cycle IN ('smoke','nightly','release','ad_hoc')),
    status TEXT NOT NULL CHECK (status IN ('queued','running','passed','failed','invalid','cancelled')),
    required_axes TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    score_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    failure_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    artifact_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS training_gate_decisions_2026 (
    decision_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES training_orchestration_runs_2026(run_id) ON DELETE CASCADE,
    decision_kind TEXT NOT NULL CHECK (decision_kind IN ('curation_export','training_continue','q4_export','release_promotion','quarantine')),
    status TEXT NOT NULL CHECK (status IN ('passed','failed','manual_review','blocked')),
    required_checks JSONB NOT NULL DEFAULT '[]'::jsonb,
    passed_checks JSONB NOT NULL DEFAULT '[]'::jsonb,
    failed_checks JSONB NOT NULL DEFAULT '[]'::jsonb,
    decision_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    decided_by TEXT NOT NULL DEFAULT 'orchestrator',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS training_q4_recovery_runs_2026 (
    q4_recovery_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES training_orchestration_runs_2026(run_id) ON DELETE CASCADE,
    source_checkpoint_artifact_id TEXT REFERENCES training_artifacts_2026(artifact_id) ON DELETE SET NULL,
    output_artifact_id TEXT REFERENCES training_artifacts_2026(artifact_id) ON DELETE SET NULL,
    target_quantization TEXT NOT NULL DEFAULT 'q4',
    method TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued','running','passed','failed','cancelled')),
    max_relative_regression DOUBLE PRECISION NOT NULL DEFAULT 0.03,
    regression_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    recovery_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS training_scheduler_queue_2026 (
    queue_id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES training_orchestration_runs_2026(run_id) ON DELETE CASCADE,
    stage_id TEXT REFERENCES training_orchestration_stages_2026(stage_id) ON DELETE CASCADE,
    job_kind TEXT NOT NULL CHECK (job_kind IN ('jsonl_ingest','curation','distillation','training','q4_recovery','loss_check','benchmark_gate','artifact_hash','promotion')),
    resource_pool TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 100,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','claimed','running','completed','failed','cancelled','blocked')),
    not_before TIMESTAMPTZ NOT NULL DEFAULT now(),
    lease_until TIMESTAMPTZ,
    claimed_by TEXT,
    claimed_at TIMESTAMPTZ,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    max_attempts INTEGER NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (run_id IS NOT NULL OR stage_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_training_orchestration_runs_status_2026 ON training_orchestration_runs_2026(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_orchestration_runs_profile_2026 ON training_orchestration_runs_2026(profile_name, run_kind);
CREATE INDEX IF NOT EXISTS idx_training_orchestration_stages_run_2026 ON training_orchestration_stages_2026(run_id, stage_order);
CREATE INDEX IF NOT EXISTS idx_training_jsonl_manifests_run_kind_2026 ON training_jsonl_manifests_2026(run_id, manifest_kind, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_orchestration_events_run_time_2026 ON training_orchestration_events_2026(run_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_training_orchestration_events_record_2026 ON training_orchestration_events_2026(record_id);
CREATE INDEX IF NOT EXISTS idx_training_curated_records_split_quality_2026 ON training_curated_records_2026(split, quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_training_curated_records_modality_2026 ON training_curated_records_2026(primary_modality, split);
CREATE INDEX IF NOT EXISTS idx_training_curated_records_payload_hash_2026 ON training_curated_records_2026(payload_sha256);
CREATE INDEX IF NOT EXISTS idx_training_artifacts_hash_2026 ON training_artifacts_2026(sha256);
CREATE INDEX IF NOT EXISTS idx_training_artifacts_run_kind_2026 ON training_artifacts_2026(run_id, artifact_kind);
CREATE INDEX IF NOT EXISTS idx_training_distillation_jobs_status_2026 ON training_distillation_jobs_2026(status, priority, created_at);
CREATE INDEX IF NOT EXISTS idx_training_distillation_jobs_teacher_2026 ON training_distillation_jobs_2026(teacher_name, job_type);
CREATE INDEX IF NOT EXISTS idx_training_loss_samples_run_step_2026 ON training_loss_samples_2026(run_id, step, loss_name);
CREATE INDEX IF NOT EXISTS idx_training_loss_trend_checks_run_2026 ON training_loss_trend_checks_2026(run_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_benchmark_gate_runs_run_2026 ON training_benchmark_gate_runs_2026(run_id, cycle, status);
CREATE INDEX IF NOT EXISTS idx_training_gate_decisions_run_2026 ON training_gate_decisions_2026(run_id, decision_kind, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_q4_recovery_runs_run_2026 ON training_q4_recovery_runs_2026(run_id, status);
CREATE INDEX IF NOT EXISTS idx_training_scheduler_queue_claim_2026 ON training_scheduler_queue_2026(status, resource_pool, priority, not_before);
CREATE INDEX IF NOT EXISTS idx_training_scheduler_queue_lease_2026 ON training_scheduler_queue_2026(lease_until) WHERE status IN ('claimed','running');

-- Raw PostgreSQL claim pattern:
-- UPDATE training_scheduler_queue_2026
-- SET status = 'claimed',
--     claimed_by = $1,
--     claimed_at = now(),
--     lease_until = now() + ($2::TEXT)::INTERVAL,
--     attempt_count = attempt_count + 1,
--     updated_at = now()
-- WHERE queue_id = (
--     SELECT queue_id
--     FROM training_scheduler_queue_2026
--     WHERE status = 'queued'
--       AND not_before <= now()
--       AND resource_pool = $3
--     ORDER BY priority ASC, created_at ASC
--     FOR UPDATE SKIP LOCKED
--     LIMIT 1
-- )
-- RETURNING *;
