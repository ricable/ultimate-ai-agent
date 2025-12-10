-- ReasoningBank Schema Extension for Claude Flow Memory System
-- Paper: https://arxiv.org/html/2509.25140v1
-- Extends existing Claude Flow memory.db at .swarm/memory.db

-- ============================================================================
-- A. Pattern Embeddings for Semantic Retrieval
-- ============================================================================
CREATE TABLE IF NOT EXISTS pattern_embeddings (
  id TEXT PRIMARY KEY,
  model TEXT NOT NULL,           -- embedding model used (e.g., claude-3-5-sonnet, text-embedding-3-large)
  dims INTEGER NOT NULL,         -- vector dimensions
  vector BLOB NOT NULL,          -- float32 array serialized
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (id) REFERENCES patterns(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_pattern_embeddings_model ON pattern_embeddings(model);

-- ============================================================================
-- B. Pattern Relationship Graph (Memory Governance)
-- ============================================================================
CREATE TABLE IF NOT EXISTS pattern_links (
  src_id TEXT NOT NULL,
  dst_id TEXT NOT NULL,
  relation TEXT NOT NULL CHECK(relation IN ('entails', 'contradicts', 'refines', 'duplicate_of')),
  weight REAL DEFAULT 1.0 CHECK(weight BETWEEN 0 AND 1),
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (src_id, dst_id, relation),
  FOREIGN KEY (src_id) REFERENCES patterns(id) ON DELETE CASCADE,
  FOREIGN KEY (dst_id) REFERENCES patterns(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_pattern_links_src ON pattern_links(src_id);
CREATE INDEX IF NOT EXISTS idx_pattern_links_dst ON pattern_links(dst_id);
CREATE INDEX IF NOT EXISTS idx_pattern_links_relation ON pattern_links(relation);

-- ============================================================================
-- C. Task Trajectory Archive with Judgment
-- ============================================================================
CREATE TABLE IF NOT EXISTS task_trajectories (
  task_id TEXT PRIMARY KEY,
  agent_id TEXT,
  query TEXT NOT NULL,
  trajectory_json TEXT NOT NULL,  -- full trajectory with steps, tool calls, outputs
  started_at TEXT,
  ended_at TEXT,
  judge_label TEXT CHECK(judge_label IN ('Success', 'Failure')),
  judge_conf REAL CHECK(judge_conf BETWEEN 0 AND 1),
  judge_reasons TEXT,             -- JSON array of reasoning strings
  matts_run_id TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_trajectories_agent ON task_trajectories(agent_id);
CREATE INDEX IF NOT EXISTS idx_trajectories_label ON task_trajectories(judge_label);
CREATE INDEX IF NOT EXISTS idx_trajectories_matts ON task_trajectories(matts_run_id);
CREATE INDEX IF NOT EXISTS idx_trajectories_created ON task_trajectories(created_at);

-- ============================================================================
-- D. MaTTS Orchestration Runs (Memory-aware Test-Time Scaling)
-- ============================================================================
CREATE TABLE IF NOT EXISTS matts_runs (
  run_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  mode TEXT NOT NULL CHECK(mode IN ('parallel', 'sequential')),
  k INTEGER NOT NULL,             -- scaling factor (rollouts or refinements)
  status TEXT DEFAULT 'completed' CHECK(status IN ('pending', 'running', 'completed', 'failed')),
  summary TEXT,                   -- JSON with outcomes, trajectories, aggregated insights
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_matts_task ON matts_runs(task_id);
CREATE INDEX IF NOT EXISTS idx_matts_mode ON matts_runs(mode);

-- ============================================================================
-- E. Consolidation Run Bookkeeping
-- ============================================================================
CREATE TABLE IF NOT EXISTS consolidation_runs (
  run_id TEXT PRIMARY KEY,
  items_processed INTEGER DEFAULT 0,
  duplicates_found INTEGER DEFAULT 0,
  contradictions_found INTEGER DEFAULT 0,
  items_pruned INTEGER DEFAULT 0,
  duration_ms INTEGER,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_consolidation_created ON consolidation_runs(created_at);

-- ============================================================================
-- F. Views for Monitoring and Debugging
-- ============================================================================

-- Active reasoning memories with usage stats
CREATE VIEW IF NOT EXISTS v_active_memories AS
SELECT
  p.id,
  json_extract(p.pattern_data, '$.title') as title,
  json_extract(p.pattern_data, '$.domain') as domain,
  p.confidence,
  p.usage_count,
  CAST((julianday('now') - julianday(p.created_at)) AS INTEGER) as age_days,
  p.last_used
FROM patterns p
WHERE p.type = 'reasoning_memory'
  AND p.confidence >= 0.3
ORDER BY p.usage_count DESC, p.confidence DESC;

-- Memory contradictions for review
CREATE VIEW IF NOT EXISTS v_memory_contradictions AS
SELECT
  pl.src_id,
  json_extract(p1.pattern_data, '$.title') as src_title,
  pl.dst_id,
  json_extract(p2.pattern_data, '$.title') as dst_title,
  pl.weight as contradiction_weight,
  pl.created_at
FROM pattern_links pl
JOIN patterns p1 ON pl.src_id = p1.id
JOIN patterns p2 ON pl.dst_id = p2.id
WHERE pl.relation = 'contradicts'
ORDER BY pl.weight DESC;

-- Task success rate by agent
CREATE VIEW IF NOT EXISTS v_agent_performance AS
SELECT
  agent_id,
  COUNT(*) as total_tasks,
  SUM(CASE WHEN judge_label = 'Success' THEN 1 ELSE 0 END) as successes,
  ROUND(100.0 * SUM(CASE WHEN judge_label = 'Success' THEN 1 ELSE 0 END) / COUNT(*), 2) as success_rate,
  AVG(judge_conf) as avg_confidence
FROM task_trajectories
GROUP BY agent_id
ORDER BY success_rate DESC;

-- ============================================================================
-- G. Triggers for Automatic Maintenance
-- ============================================================================

-- Auto-delete orphaned embeddings when pattern is deleted
CREATE TRIGGER IF NOT EXISTS trg_cleanup_embeddings
AFTER DELETE ON patterns
BEGIN
  DELETE FROM pattern_embeddings WHERE id = OLD.id;
END;

-- Update last_used timestamp on pattern when used
CREATE TRIGGER IF NOT EXISTS trg_update_last_used
AFTER UPDATE OF usage_count ON patterns
BEGIN
  UPDATE patterns
  SET last_used = CURRENT_TIMESTAMP
  WHERE id = NEW.id;
END;

-- ============================================================================
-- H. Initial Data: Bootstrap with Empty State
-- ============================================================================

-- No seed data needed - memories learned from experience

-- ============================================================================
-- Migration Complete
-- ============================================================================
-- This schema extends Claude Flow's existing patterns table with:
-- 1. Semantic embeddings for retrieval (pattern_embeddings)
-- 2. Memory relationships for governance (pattern_links)
-- 3. Trajectory archive with judgments (task_trajectories)
-- 4. MaTTS orchestration tracking (matts_runs)
-- 5. Consolidation audit trail (consolidation_runs)
--
-- Usage: Run with sqlite3 .swarm/memory.db < 001_reasoningbank_schema.sql
-- ============================================================================
