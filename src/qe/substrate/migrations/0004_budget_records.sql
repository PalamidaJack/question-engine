-- Budget tracking persistence
-- Stores per-call LLM cost records and monthly summaries

CREATE TABLE IF NOT EXISTS budget_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    month_key TEXT NOT NULL,
    model TEXT NOT NULL,
    cost_usd REAL NOT NULL,
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    service_id TEXT DEFAULT '',
    envelope_id TEXT DEFAULT '',
    created_at TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_budget_month ON budget_records(month_key);
CREATE INDEX IF NOT EXISTS idx_budget_model ON budget_records(month_key, model);

-- Monthly summary for fast lookups on startup
CREATE TABLE IF NOT EXISTS budget_monthly (
    month_key TEXT PRIMARY KEY,
    total_spend_usd REAL NOT NULL DEFAULT 0.0,
    spend_by_model JSON NOT NULL DEFAULT '{}'
);
