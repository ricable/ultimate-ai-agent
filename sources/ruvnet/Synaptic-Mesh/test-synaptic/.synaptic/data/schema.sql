
-- Node metadata
CREATE TABLE IF NOT EXISTS node_info (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Peer connections
CREATE TABLE IF NOT EXISTS peers (
  id TEXT PRIMARY KEY,
  address TEXT NOT NULL,
  public_key TEXT,
  last_seen TIMESTAMP,
  reputation REAL DEFAULT 1.0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Neural agents
CREATE TABLE IF NOT EXISTS agents (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  type TEXT NOT NULL,
  architecture TEXT,
  state BLOB,
  performance REAL DEFAULT 0.0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  terminated_at TIMESTAMP
);

-- DAG vertices
CREATE TABLE IF NOT EXISTS dag_vertices (
  id TEXT PRIMARY KEY,
  previous_ids TEXT,
  data BLOB,
  signature TEXT,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  confirmations INTEGER DEFAULT 0
);

-- Metrics and telemetry
CREATE TABLE IF NOT EXISTS metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  metric_type TEXT NOT NULL,
  value REAL,
  metadata TEXT,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_peers_last_seen ON peers(last_seen);
CREATE INDEX IF NOT EXISTS idx_agents_type ON agents(type);
CREATE INDEX IF NOT EXISTS idx_dag_timestamp ON dag_vertices(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_type_time ON metrics(metric_type, timestamp);
