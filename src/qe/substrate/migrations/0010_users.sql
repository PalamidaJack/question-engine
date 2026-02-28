-- Migration 0010: User management for multi-user support (Phase 14)

CREATE TABLE IF NOT EXISTS users (
    user_id        TEXT PRIMARY KEY,
    display_name   TEXT NOT NULL DEFAULT '',
    preferences    TEXT NOT NULL DEFAULT '{}',  -- JSON
    active_project TEXT DEFAULT NULL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS user_sessions (
    session_id     TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    channel        TEXT NOT NULL DEFAULT 'web',
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    last_active    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_user_sessions_user
    ON user_sessions(user_id);
