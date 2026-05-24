-- Omnicoder 2026 agentic tool-calling training registry/schema.
-- Raw PostgreSQL mirror for JSONL tool-training artifacts.

CREATE TABLE IF NOT EXISTS agentic_tool_training_runs_2026 (
    run_id UUID PRIMARY KEY,
    profile_version TEXT NOT NULL,
    source_jsonl TEXT NOT NULL,
    output_dir TEXT NOT NULL,
    model_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('planned','running','passed','failed','invalid')),
    counts JSONB NOT NULL DEFAULT '{}'::jsonb,
    reward_axes JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS agentic_tool_examples_2026 (
    example_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES agentic_tool_training_runs_2026(run_id) ON DELETE CASCADE,
    trace_id TEXT NOT NULL,
    record_hash TEXT NOT NULL,
    training_kind TEXT NOT NULL CHECK (
        training_kind IN (
            'tool_sft',
            'tool_reward',
            'tool_preference',
            'tool_rlvr',
            'tool_safety_negative',
            'math_rlvr',
            'code_rlvr',
            'terminal_rlvr',
            'browser_rlvr',
            'multimodal_rlvr'
        )
    ),
    source_date DATE,
    messages JSONB NOT NULL DEFAULT '[]'::jsonb,
    prompt TEXT,
    chosen TEXT,
    rejected TEXT,
    reward NUMERIC,
    tool_calls JSONB NOT NULL DEFAULT '[]'::jsonb,
    tool_results JSONB NOT NULL DEFAULT '[]'::jsonb,
    risk_labels JSONB NOT NULL DEFAULT '[]'::jsonb,
    quality_score NUMERIC,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(run_id, trace_id, record_hash, training_kind)
);

CREATE TABLE IF NOT EXISTS agentic_tool_reward_events_2026 (
    reward_event_id BIGSERIAL PRIMARY KEY,
    example_id UUID NOT NULL REFERENCES agentic_tool_examples_2026(example_id) ON DELETE CASCADE,
    reward_axis TEXT NOT NULL,
    reward_value NUMERIC NOT NULL,
    verifier JSONB NOT NULL DEFAULT '{}'::jsonb,
    evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agentic_tool_posttrain_manifests_2026 (
    manifest_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES agentic_tool_training_runs_2026(run_id) ON DELETE CASCADE,
    algorithm TEXT NOT NULL,
    train_jsonl TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    dry_run BOOLEAN NOT NULL DEFAULT true,
    q4_recovery_ready BOOLEAN NOT NULL DEFAULT true,
    manifest_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agentic_tool_quarantine_2026 (
    quarantine_id UUID PRIMARY KEY,
    run_id UUID REFERENCES agentic_tool_training_runs_2026(run_id) ON DELETE SET NULL,
    trace_id TEXT,
    record_hash TEXT NOT NULL,
    reason TEXT NOT NULL,
    risk_labels JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_agentic_tool_runs_status_2026 ON agentic_tool_training_runs_2026(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agentic_tool_examples_run_kind_2026 ON agentic_tool_examples_2026(run_id, training_kind);
CREATE INDEX IF NOT EXISTS idx_agentic_tool_examples_trace_2026 ON agentic_tool_examples_2026(trace_id);
CREATE INDEX IF NOT EXISTS idx_agentic_tool_reward_axis_2026 ON agentic_tool_reward_events_2026(reward_axis);
CREATE INDEX IF NOT EXISTS idx_agentic_tool_manifest_algorithm_2026 ON agentic_tool_posttrain_manifests_2026(algorithm);
CREATE INDEX IF NOT EXISTS idx_agentic_tool_quarantine_hash_2026 ON agentic_tool_quarantine_2026(record_hash);
