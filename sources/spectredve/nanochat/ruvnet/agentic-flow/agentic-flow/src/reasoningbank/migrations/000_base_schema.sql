-- Base Claude Flow Memory Schema
-- Creates foundation tables needed by ReasoningBank

PRAGMA foreign_keys = ON;

-- Core patterns table for storing all learned patterns
CREATE TABLE IF NOT EXISTS patterns (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  pattern_data TEXT NOT NULL,  -- JSON-encoded pattern data
  confidence REAL NOT NULL DEFAULT 0.5,
  usage_count INTEGER NOT NULL DEFAULT 0,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  last_used TEXT
);

CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(type);
CREATE INDEX IF NOT EXISTS idx_patterns_confidence ON patterns(confidence);
CREATE INDEX IF NOT EXISTS idx_patterns_created ON patterns(created_at);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  metric_name TEXT NOT NULL,
  value REAL NOT NULL,
  timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
  metadata TEXT  -- JSON-encoded additional data
);

CREATE INDEX IF NOT EXISTS idx_metrics_name ON performance_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp);

-- Memory namespace table for multi-tenant support
CREATE TABLE IF NOT EXISTS memory_namespaces (
  namespace TEXT PRIMARY KEY,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  metadata TEXT  -- JSON-encoded namespace config
);

-- Session state table for cross-session persistence
CREATE TABLE IF NOT EXISTS session_state (
  session_id TEXT PRIMARY KEY,
  state_data TEXT NOT NULL,  -- JSON-encoded state
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_session_updated ON session_state(updated_at);
