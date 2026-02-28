CREATE TABLE IF NOT EXISTS memory_entries (
    memory_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    source TEXT NOT NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    expires_at TIMESTAMP,
    superseded_by TEXT
);

CREATE INDEX IF NOT EXISTS idx_memory_category ON memory_entries(category);
CREATE INDEX IF NOT EXISTS idx_memory_key ON memory_entries(key);

CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP,
    last_accessed TIMESTAMP
);

CREATE TABLE IF NOT EXISTS project_claims (
    project_id TEXT,
    claim_id TEXT,
    relevance_score REAL,
    PRIMARY KEY (project_id, claim_id)
);

CREATE TABLE IF NOT EXISTS planning_patterns (
    pattern_id TEXT PRIMARY KEY,
    goal_description TEXT,
    decomposition JSON,
    outcome TEXT,
    total_cost_usd REAL,
    total_time_seconds INTEGER,
    created_at TIMESTAMP
);
