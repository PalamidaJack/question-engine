CREATE TABLE IF NOT EXISTS scout_proposals (
    proposal_id TEXT PRIMARY KEY,
    idea_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    changes_json TEXT NOT NULL DEFAULT '[]',
    test_result_json TEXT,
    impact_assessment TEXT NOT NULL DEFAULT '',
    risk_assessment TEXT NOT NULL DEFAULT '',
    rollback_plan TEXT NOT NULL DEFAULT '',
    branch_name TEXT NOT NULL DEFAULT '',
    worktree_path TEXT NOT NULL DEFAULT '',
    hil_envelope_id TEXT,
    reviewer_feedback TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    decided_at TEXT,
    applied_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_sp_status ON scout_proposals(status);
CREATE INDEX IF NOT EXISTS idx_sp_created ON scout_proposals(created_at);

CREATE TABLE IF NOT EXISTS scout_findings (
    finding_id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    snippet TEXT NOT NULL DEFAULT '',
    full_content TEXT NOT NULL DEFAULT '',
    source_type TEXT NOT NULL,
    relevance_score REAL NOT NULL DEFAULT 0.0,
    discovered_at TEXT NOT NULL,
    tags_json TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_sf_url ON scout_findings(url);
CREATE INDEX IF NOT EXISTS idx_sf_source ON scout_findings(source_type);

CREATE TABLE IF NOT EXISTS scout_feedback (
    record_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    feedback TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT '',
    source_type TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sfb_decision ON scout_feedback(decision);
CREATE INDEX IF NOT EXISTS idx_sfb_category ON scout_feedback(category);
