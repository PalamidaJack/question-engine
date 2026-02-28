CREATE TABLE IF NOT EXISTS calibration_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    task_type TEXT NOT NULL,
    reported_confidence REAL NOT NULL,
    actual_correct BOOLEAN NOT NULL,
    created_at TIMESTAMP,
    goal_id TEXT,
    subtask_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_calibration ON calibration_records(model, task_type);

CREATE TABLE IF NOT EXISTS routing_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    task_type TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    latency_ms INTEGER,
    cost_usd REAL,
    quality_score REAL,
    created_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_routing ON routing_records(model, task_type);
