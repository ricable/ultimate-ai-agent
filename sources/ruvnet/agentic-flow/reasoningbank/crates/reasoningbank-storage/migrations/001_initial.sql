-- ReasoningBank initial schema
-- Compatible with both rusqlite and sql.js

CREATE TABLE IF NOT EXISTS patterns (
    id TEXT PRIMARY KEY,
    task_category TEXT NOT NULL,
    task_description TEXT NOT NULL,
    strategy TEXT NOT NULL,
    success_score REAL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_patterns_category ON patterns(task_category);
CREATE INDEX IF NOT EXISTS idx_patterns_score ON patterns(success_score DESC);
CREATE INDEX IF NOT EXISTS idx_patterns_created ON patterns(created_at DESC);

-- Embeddings table for similarity search
CREATE TABLE IF NOT EXISTS pattern_embeddings (
    pattern_id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    dimension INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pattern_id) REFERENCES patterns(id) ON DELETE CASCADE
);

-- Statistics table
CREATE TABLE IF NOT EXISTS storage_stats (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_metrics_name ON performance_metrics(metric_name, timestamp DESC);
