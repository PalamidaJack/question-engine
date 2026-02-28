-- Embedding vector storage for semantic search and contradiction detection

CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    dimensions INTEGER NOT NULL,
    metadata JSON DEFAULT '{}',
    created_at TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_embeddings_created ON embeddings(created_at);
