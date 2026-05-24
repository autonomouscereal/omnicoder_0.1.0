-- Omnicoder 2026 distillation and post-training registry.
-- Raw PostgreSQL only.

CREATE TABLE IF NOT EXISTS distillation_teachers_2026 (
    teacher_id BIGSERIAL PRIMARY KEY,
    teacher_name TEXT NOT NULL UNIQUE,
    provider TEXT NOT NULL,
    model_alias TEXT NOT NULL,
    endpoint_env TEXT,
    modalities JSONB NOT NULL DEFAULT '[]'::jsonb,
    job_types JSONB NOT NULL DEFAULT '[]'::jsonb,
    priority INTEGER NOT NULL DEFAULT 100,
    enabled BOOLEAN NOT NULL DEFAULT true,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS distillation_job_manifests_2026 (
    manifest_id BIGSERIAL PRIMARY KEY,
    profile_name TEXT NOT NULL,
    source_records_path TEXT NOT NULL,
    jobs_path TEXT NOT NULL,
    job_count INTEGER NOT NULL,
    teacher_counts JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS posttraining_recipes_2026 (
    recipe_id BIGSERIAL PRIMARY KEY,
    recipe_name TEXT NOT NULL UNIQUE,
    algorithm TEXT NOT NULL,
    trainer TEXT NOT NULL,
    dataset_schema TEXT NOT NULL,
    purpose TEXT NOT NULL,
    hyperparameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    reward_contract JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS posttraining_runs_2026 (
    posttraining_run_id BIGSERIAL PRIMARY KEY,
    recipe_id BIGINT REFERENCES posttraining_recipes_2026(recipe_id) ON DELETE SET NULL,
    run_name TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    model_name TEXT NOT NULL,
    train_jsonl TEXT,
    eval_jsonl TEXT,
    output_dir TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'configured',
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    manifest JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_distillation_teachers_enabled ON distillation_teachers_2026(enabled, priority);
CREATE INDEX IF NOT EXISTS idx_distillation_job_manifests_created ON distillation_job_manifests_2026(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_posttraining_recipes_algorithm ON posttraining_recipes_2026(algorithm);
CREATE INDEX IF NOT EXISTS idx_posttraining_runs_status ON posttraining_runs_2026(status, created_at DESC);
