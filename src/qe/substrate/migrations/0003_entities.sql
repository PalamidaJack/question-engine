-- depends: 0002_fts_triggers
-- __transactional__ = false

CREATE TABLE IF NOT EXISTS entities (
    canonical_name TEXT PRIMARY KEY,
    aliases TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entities_updated ON entities(updated_at);
