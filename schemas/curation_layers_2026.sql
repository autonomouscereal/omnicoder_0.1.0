-- Omnicoder 2026 raw PostgreSQL dataset curation layers.
-- Raw PostgreSQL only; no object-mapping or alternate local database layer.

CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS curated_records (
    curated_record_id BIGSERIAL PRIMARY KEY,
    curated_id CHAR(64) NOT NULL UNIQUE,
    normalized_text TEXT NOT NULL DEFAULT '',
    source_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    normalization JSONB NOT NULL DEFAULT '{}'::jsonb,
    secret_redaction JSONB NOT NULL DEFAULT '{}'::jsonb,
    language_classification JSONB NOT NULL DEFAULT '{}'::jsonb,
    code_classification JSONB NOT NULL DEFAULT '{}'::jsonb,
    tool_classification JSONB NOT NULL DEFAULT '{}'::jsonb,
    media_classification JSONB NOT NULL DEFAULT '{}'::jsonb,
    quality JSONB NOT NULL DEFAULT '{}'::jsonb,
    dedupe JSONB NOT NULL DEFAULT '{}'::jsonb,
    contamination JSONB NOT NULL DEFAULT '{}'::jsonb,
    provenance JSONB NOT NULL DEFAULT '{}'::jsonb,
    split_assignment JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE curated_records ADD COLUMN IF NOT EXISTS curated_id CHAR(64);
ALTER TABLE curated_records ADD COLUMN IF NOT EXISTS normalized_text TEXT NOT NULL DEFAULT '';
ALTER TABLE curated_records ADD COLUMN IF NOT EXISTS source_payload JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE curated_records ADD COLUMN IF NOT EXISTS normalization JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE curated_records ADD COLUMN IF NOT EXISTS secret_redaction JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE curated_records ADD COLUMN IF NOT EXISTS language_classification JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE curated_records ADD COLUMN IF NOT EXISTS code_classification JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE curated_records ADD COLUMN IF NOT EXISTS tool_classification JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE curated_records ADD COLUMN IF NOT EXISTS media_classification JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE curated_records ADD COLUMN IF NOT EXISTS quality JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE curated_records ADD COLUMN IF NOT EXISTS dedupe JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE curated_records ADD COLUMN IF NOT EXISTS contamination JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE curated_records ADD COLUMN IF NOT EXISTS provenance JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE curated_records ADD COLUMN IF NOT EXISTS split_assignment JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE curated_records ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();
ALTER TABLE curated_records ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();

CREATE UNIQUE INDEX IF NOT EXISTS uq_curated_records_curated_id ON curated_records(curated_id);
CREATE INDEX IF NOT EXISTS idx_curated_records_text_trgm ON curated_records USING gin (normalized_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_curated_records_quality ON curated_records(((quality->>'label')), (((quality->>'overall')::DOUBLE PRECISION)));
CREATE INDEX IF NOT EXISTS idx_curated_records_split ON curated_records(((split_assignment->>'split')));
CREATE INDEX IF NOT EXISTS idx_curated_records_dedupe_canonical ON curated_records((dedupe->>'canonical_sha256'));
CREATE INDEX IF NOT EXISTS idx_curated_records_provenance ON curated_records USING gin (provenance);

CREATE TABLE IF NOT EXISTS source_provenance_records (
    provenance_id BIGSERIAL PRIMARY KEY,
    curated_record_id BIGINT NOT NULL REFERENCES curated_records(curated_record_id) ON DELETE CASCADE,
    source_name TEXT NOT NULL,
    source_uri TEXT,
    source_date DATE,
    license_id TEXT NOT NULL DEFAULT 'unknown',
    path TEXT,
    line_number INTEGER,
    record_id TEXT,
    raw_record_hash CHAR(64) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (source_date IS NULL OR source_date >= DATE '2025-01-01')
);

CREATE INDEX IF NOT EXISTS idx_source_provenance_records_source ON source_provenance_records(source_name, source_date DESC);
CREATE INDEX IF NOT EXISTS idx_source_provenance_records_hash ON source_provenance_records(raw_record_hash);
CREATE INDEX IF NOT EXISTS idx_source_provenance_records_curated ON source_provenance_records(curated_record_id);

CREATE TABLE IF NOT EXISTS curation_secret_findings (
    secret_finding_id BIGSERIAL PRIMARY KEY,
    curated_record_id BIGINT NOT NULL REFERENCES curated_records(curated_record_id) ON DELETE CASCADE,
    secret_type TEXT NOT NULL,
    secret_hash CHAR(64) NOT NULL,
    start_offset INTEGER,
    end_offset INTEGER,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_curation_secret_findings_record ON curation_secret_findings(curated_record_id);
CREATE INDEX IF NOT EXISTS idx_curation_secret_findings_type_hash ON curation_secret_findings(secret_type, secret_hash);

CREATE TABLE IF NOT EXISTS curation_classifications (
    classification_id BIGSERIAL PRIMARY KEY,
    curated_record_id BIGINT NOT NULL REFERENCES curated_records(curated_record_id) ON DELETE CASCADE,
    classifier_kind TEXT NOT NULL CHECK (classifier_kind IN ('language', 'code', 'tools', 'media')),
    label TEXT NOT NULL,
    confidence DOUBLE PRECISION,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_curation_classifications_lookup ON curation_classifications(classifier_kind, label);
CREATE INDEX IF NOT EXISTS idx_curation_classifications_record ON curation_classifications(curated_record_id);

CREATE TABLE IF NOT EXISTS curation_quality_dimensions (
    quality_dimension_id BIGSERIAL PRIMARY KEY,
    curated_record_id BIGINT NOT NULL REFERENCES curated_records(curated_record_id) ON DELETE CASCADE,
    dimension_name TEXT NOT NULL,
    dimension_value DOUBLE PRECISION NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_curation_quality_dimensions_lookup ON curation_quality_dimensions(dimension_name, dimension_value DESC);
CREATE INDEX IF NOT EXISTS idx_curation_quality_dimensions_record ON curation_quality_dimensions(curated_record_id);

CREATE TABLE IF NOT EXISTS curation_dedupe_signatures (
    dedupe_signature_id BIGSERIAL PRIMARY KEY,
    curated_record_id BIGINT NOT NULL REFERENCES curated_records(curated_record_id) ON DELETE CASCADE,
    signature_type TEXT NOT NULL,
    signature_value TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_curation_dedupe_signatures_value ON curation_dedupe_signatures(signature_type, signature_value);
CREATE INDEX IF NOT EXISTS idx_curation_dedupe_signatures_record ON curation_dedupe_signatures(curated_record_id);

CREATE TABLE IF NOT EXISTS curation_contamination_labels (
    contamination_label_id BIGSERIAL PRIMARY KEY,
    curated_record_id BIGINT NOT NULL REFERENCES curated_records(curated_record_id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (status IN ('clean', 'suspect', 'contaminated')),
    match_type TEXT NOT NULL DEFAULT 'none',
    score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    protected_artifact_id BIGINT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_curation_contamination_labels_status ON curation_contamination_labels(status, score DESC);
CREATE INDEX IF NOT EXISTS idx_curation_contamination_labels_record ON curation_contamination_labels(curated_record_id);

CREATE TABLE IF NOT EXISTS curation_split_assignments (
    curation_split_assignment_id BIGSERIAL PRIMARY KEY,
    curated_record_id BIGINT NOT NULL REFERENCES curated_records(curated_record_id) ON DELETE CASCADE,
    split_name TEXT NOT NULL CHECK (split_name IN ('train', 'validation', 'eval_holdout', 'rejected', 'quarantine')),
    reason TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_curation_split_assignments_split ON curation_split_assignments(split_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_curation_split_assignments_record ON curation_split_assignments(curated_record_id);

CREATE TABLE IF NOT EXISTS curation_export_manifests (
    manifest_id BIGSERIAL PRIMARY KEY,
    export_name TEXT NOT NULL,
    export_kind TEXT NOT NULL,
    output_path TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_curation_export_manifests_kind_created ON curation_export_manifests(export_kind, created_at DESC);

-- Raw PostgreSQL example:
-- INSERT INTO curated_records (curated_id, normalized_text, source_payload, quality, dedupe, provenance, split_assignment)
-- VALUES ($1, $2, $3::jsonb, $4::jsonb, $5::jsonb, $6::jsonb, $7::jsonb)
-- ON CONFLICT (curated_id) DO UPDATE SET updated_at = now();
