CREATE TABLE IF NOT EXISTS free_model_inventory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    context_length INTEGER DEFAULT 32000,
    capabilities TEXT,
    is_available INTEGER DEFAULT 1,
    rate_limit_rpm INTEGER DEFAULT 20,
    rate_limit_rpd INTEGER DEFAULT 200,
    last_checked TEXT,
    last_error TEXT,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    avg_latency_ms REAL DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(provider, model_id)
);

CREATE TABLE IF NOT EXISTS provider_info (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL UNIQUE,
    api_base TEXT,
    requires_api_key INTEGER DEFAULT 1,
    is_active INTEGER DEFAULT 1,
    rate_limit_default_rpm INTEGER DEFAULT 20,
    rate_limit_default_rpd INTEGER DEFAULT 200,
    notes TEXT,
    last_scraped TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    error_code TEXT,
    error_message TEXT,
    error_type TEXT,
    resolved INTEGER DEFAULT 0,
    resolution_notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    resolved_at TEXT
);

CREATE INDEX idx_free_model_provider ON free_model_inventory(provider);
CREATE INDEX idx_free_model_available ON free_model_inventory(is_available);
CREATE INDEX idx_model_errors_unresolved ON model_errors(resolved, created_at);
