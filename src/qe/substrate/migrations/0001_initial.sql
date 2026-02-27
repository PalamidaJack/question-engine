-- __transactional__ = false

CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    subject_entity_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_value TEXT NOT NULL,
    confidence REAL NOT NULL,
    source_service_id TEXT NOT NULL,
    source_envelope_ids TEXT NOT NULL,
    created_at TEXT NOT NULL,
    valid_until TEXT,
    superseded_by TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_claims_subject ON claims(subject_entity_id);
CREATE INDEX IF NOT EXISTS idx_claims_predicate ON claims(predicate);
CREATE INDEX IF NOT EXISTS idx_claims_superseded ON claims(superseded_by);

CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(
    claim_id UNINDEXED,
    subject_entity_id,
    predicate,
    object_value,
    tags,
    content=claims
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    statement TEXT NOT NULL,
    confidence REAL NOT NULL,
    resolution_criteria TEXT NOT NULL,
    resolution_deadline TEXT,
    source_service_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    resolution TEXT NOT NULL DEFAULT 'unresolved',
    resolution_evidence_ids TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_predictions_deadline ON predictions(resolution_deadline);
CREATE INDEX IF NOT EXISTS idx_predictions_status ON predictions(resolution);

CREATE TABLE IF NOT EXISTS null_results (
    null_result_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    query TEXT NOT NULL,
    search_scope TEXT NOT NULL,
    source_service_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    significance TEXT NOT NULL DEFAULT 'low',
    notes TEXT NOT NULL DEFAULT ''
);
