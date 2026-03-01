-- Evidence table for Bayesian belief tracking
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    source TEXT NOT NULL,
    supports BOOLEAN NOT NULL,
    strength REAL NOT NULL,
    timestamp TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (claim_id) REFERENCES claims(claim_id)
);

CREATE INDEX IF NOT EXISTS idx_evidence_claim ON evidence(claim_id);
CREATE INDEX IF NOT EXISTS idx_evidence_timestamp ON evidence(timestamp);

-- Bayesian tracking columns on claims
-- SQLite doesn't support ADD COLUMN IF NOT EXISTS, so we use a safe approach
ALTER TABLE claims ADD COLUMN prior REAL DEFAULT 0.5;
ALTER TABLE claims ADD COLUMN posterior REAL;
ALTER TABLE claims ADD COLUMN evidence_count INTEGER DEFAULT 0;
ALTER TABLE claims ADD COLUMN likelihood_ratio REAL DEFAULT 1.0;
ALTER TABLE claims ADD COLUMN updated_at TEXT;

-- Knowledge graph edges for GraphRAG
CREATE TABLE IF NOT EXISTS knowledge_graph_edges (
    edge_id TEXT PRIMARY KEY,
    source_entity TEXT NOT NULL,
    target_entity TEXT NOT NULL,
    relation TEXT NOT NULL,
    confidence REAL NOT NULL,
    source_claim_ids TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kg_source ON knowledge_graph_edges(source_entity);
CREATE INDEX IF NOT EXISTS idx_kg_target ON knowledge_graph_edges(target_entity);
CREATE INDEX IF NOT EXISTS idx_kg_relation ON knowledge_graph_edges(relation);

-- Hypotheses table
CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id TEXT PRIMARY KEY,
    statement TEXT NOT NULL,
    prior_probability REAL NOT NULL DEFAULT 0.5,
    current_probability REAL NOT NULL DEFAULT 0.5,
    falsification_criteria TEXT NOT NULL DEFAULT '[]',
    experiments TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'active',
    source_contradiction_ids TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_hypotheses_status ON hypotheses(status);
