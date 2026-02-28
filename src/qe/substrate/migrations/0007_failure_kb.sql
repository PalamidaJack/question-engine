-- Failure knowledge base for learning from failures

CREATE TABLE IF NOT EXISTS failure_records (
    failure_id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    model_used TEXT,
    failure_class TEXT NOT NULL,
    error_summary TEXT,
    context_fingerprint TEXT,
    recovery_strategy TEXT,
    recovery_succeeded BOOLEAN,
    created_at TIMESTAMP NOT NULL,
    goal_id TEXT,
    subtask_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_failure_lookup ON failure_records(failure_class, task_type);
CREATE INDEX IF NOT EXISTS idx_failure_goal ON failure_records(goal_id);
