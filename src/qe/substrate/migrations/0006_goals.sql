-- Goal persistence for crash recovery

CREATE TABLE IF NOT EXISTS goals (
    goal_id TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'planning',
    decomposition JSON,
    subtask_states JSON DEFAULT '{}',
    subtask_results JSON DEFAULT '{}',
    created_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL,
    subtask_states JSON NOT NULL,
    subtask_results JSON NOT NULL,
    created_at TIMESTAMP NOT NULL,
    FOREIGN KEY (goal_id) REFERENCES goals(goal_id)
);

CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);
CREATE INDEX IF NOT EXISTS idx_checkpoints_goal ON checkpoints(goal_id);
