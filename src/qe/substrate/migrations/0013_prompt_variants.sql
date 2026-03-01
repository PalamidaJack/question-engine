CREATE TABLE IF NOT EXISTS prompt_variants (
    variant_id TEXT PRIMARY KEY,
    slot_key TEXT NOT NULL,
    content TEXT NOT NULL,
    is_baseline INTEGER NOT NULL DEFAULT 0,
    rollout_pct REAL NOT NULL DEFAULT 100.0,
    parent_variant_id TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    alpha REAL NOT NULL DEFAULT 1.0,
    beta REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pv_slot ON prompt_variants(slot_key);
CREATE INDEX IF NOT EXISTS idx_pv_active ON prompt_variants(active);

CREATE TABLE IF NOT EXISTS prompt_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    variant_id TEXT NOT NULL,
    slot_key TEXT NOT NULL,
    success INTEGER NOT NULL,
    quality_score REAL NOT NULL DEFAULT 0.0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_po_variant ON prompt_outcomes(variant_id);
CREATE INDEX IF NOT EXISTS idx_po_slot ON prompt_outcomes(slot_key);
