-- SQLite Schema for Research Jobs Database
-- Replaces Supabase permit_research_jobs table

CREATE TABLE IF NOT EXISTS research_jobs (
  -- Primary identification
  id TEXT PRIMARY KEY,
  
  -- Job configuration
  agent TEXT NOT NULL,
  task TEXT NOT NULL,
  location TEXT,
  
  -- Status tracking
  status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
  progress INTEGER DEFAULT 0 CHECK(progress >= 0 AND progress <= 100),
  current_message TEXT,
  
  -- Tool tracking
  current_tool TEXT,
  tool_count INTEGER DEFAULT 0,
  tool_timestamp TEXT,
  
  -- Execution tracking
  exit_code INTEGER,
  execution_log TEXT,
  
  -- Report data
  report_content TEXT,
  report_format TEXT CHECK(report_format IN ('markdown', 'json', 'html') OR report_format IS NULL),
  report_path TEXT,
  
  -- Error handling
  error_message TEXT,
  retry_count INTEGER DEFAULT 0,
  max_retries INTEGER DEFAULT 3,
  
  -- Performance metrics
  duration_seconds INTEGER,
  tokens_used INTEGER,
  memory_mb INTEGER,
  grounding_score REAL,
  
  -- Timestamps
  created_at TEXT DEFAULT (datetime('now')),
  started_at TEXT,
  completed_at TEXT,
  last_update TEXT DEFAULT (datetime('now')),
  
  -- Custom metadata (JSON string)
  metadata TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_jobs_status ON research_jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_agent ON research_jobs(agent);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON research_jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_completed ON research_jobs(completed_at DESC);

-- View for active jobs
CREATE VIEW IF NOT EXISTS active_jobs AS
SELECT 
  id, agent, task, status, progress, current_message,
  duration_seconds, created_at, started_at
FROM research_jobs
WHERE status IN ('pending', 'running')
ORDER BY created_at DESC;

-- Trigger to update last_update timestamp
CREATE TRIGGER IF NOT EXISTS update_last_update
AFTER UPDATE ON research_jobs
FOR EACH ROW
BEGIN
  UPDATE research_jobs
  SET last_update = datetime('now')
  WHERE id = NEW.id;
END;

-- ============================================================================
-- AGENTDB REASONINGBANK INTEGRATION
-- Self-learning and pattern recognition tables
-- ============================================================================

-- ReasoningBank: Store reasoning patterns and trajectories
CREATE TABLE IF NOT EXISTS reasoningbank_patterns (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  job_id TEXT,

  -- Task information
  task TEXT NOT NULL,
  agent_type TEXT,

  -- Reasoning trajectory
  input TEXT,
  output TEXT,
  reasoning_steps TEXT, -- JSON array of steps

  -- Quality metrics
  reward REAL DEFAULT 0.5 CHECK(reward >= 0 AND reward <= 1),
  success INTEGER DEFAULT 0 CHECK(success IN (0, 1)),
  confidence REAL CHECK(confidence >= 0 AND confidence <= 1),

  -- Learning feedback
  critique TEXT,
  improvements TEXT, -- JSON array of suggested improvements

  -- Performance metrics
  latency_ms INTEGER,
  tokens_used INTEGER,
  memory_mb INTEGER,

  -- Metadata
  created_at TEXT DEFAULT (datetime('now')),
  metadata TEXT, -- JSON

  FOREIGN KEY (job_id) REFERENCES research_jobs(id)
);

-- Learning episodes: Track agent learning over time
CREATE TABLE IF NOT EXISTS learning_episodes (
  id TEXT PRIMARY KEY,
  pattern_id TEXT,

  -- Episode information
  episode_number INTEGER,
  task_category TEXT,
  difficulty_level INTEGER CHECK(difficulty_level >= 1 AND difficulty_level <= 10),

  -- Performance tracking
  initial_score REAL,
  final_score REAL,
  improvement_rate REAL,

  -- Learning metrics
  exploration_rate REAL,
  exploitation_rate REAL,
  learning_rate REAL,

  -- Verdict and judgment
  verdict TEXT CHECK(verdict IN ('success', 'failure', 'partial', 'retry')),
  judgment_score REAL CHECK(judgment_score >= 0 AND judgment_score <= 1),

  -- Timestamps
  started_at TEXT,
  completed_at TEXT,
  duration_seconds INTEGER,

  -- Metadata
  metadata TEXT,

  FOREIGN KEY (pattern_id) REFERENCES reasoningbank_patterns(id)
);

-- Vector embeddings: Semantic search and similarity matching
CREATE TABLE IF NOT EXISTS vector_embeddings (
  id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL,
  source_type TEXT NOT NULL CHECK(source_type IN ('pattern', 'episode', 'task', 'report')),

  -- Vector data
  embedding_model TEXT DEFAULT 'text-embedding-3-small',
  vector_dimensions INTEGER DEFAULT 1536,
  vector_data TEXT, -- JSON array or base64 encoded

  -- Content metadata
  content_text TEXT,
  content_hash TEXT,
  content_length INTEGER,

  -- Similarity metadata
  cluster_id TEXT,
  nearest_neighbors TEXT, -- JSON array of IDs

  -- Timestamps
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT,

  -- Metadata
  metadata TEXT
);

-- Memory distillation: Compressed knowledge from multiple episodes
CREATE TABLE IF NOT EXISTS memory_distillations (
  id TEXT PRIMARY KEY,

  -- Source information
  source_pattern_ids TEXT, -- JSON array of pattern IDs
  task_category TEXT,

  -- Distilled knowledge
  key_insights TEXT, -- JSON array
  success_factors TEXT, -- JSON array
  failure_patterns TEXT, -- JSON array
  best_practices TEXT, -- JSON array

  -- Quality metrics
  confidence_score REAL CHECK(confidence_score >= 0 AND confidence_score <= 1),
  usage_count INTEGER DEFAULT 0,
  success_rate REAL,

  -- Timestamps
  created_at TEXT DEFAULT (datetime('now')),
  last_used_at TEXT,

  -- Metadata
  metadata TEXT
);

-- Pattern associations: Link related patterns for learning
CREATE TABLE IF NOT EXISTS pattern_associations (
  id TEXT PRIMARY KEY,
  pattern_id_a TEXT NOT NULL,
  pattern_id_b TEXT NOT NULL,

  -- Association metrics
  similarity_score REAL CHECK(similarity_score >= 0 AND similarity_score <= 1),
  association_type TEXT CHECK(association_type IN ('similar', 'complementary', 'contrasting', 'sequential')),

  -- Learning value
  learning_value REAL CHECK(learning_value >= 0 AND learning_value <= 1),
  usage_count INTEGER DEFAULT 0,

  -- Timestamps
  created_at TEXT DEFAULT (datetime('now')),
  last_used_at TEXT,

  FOREIGN KEY (pattern_id_a) REFERENCES reasoningbank_patterns(id),
  FOREIGN KEY (pattern_id_b) REFERENCES reasoningbank_patterns(id)
);

-- ============================================================================
-- AGENTDB INDEXES FOR PERFORMANCE
-- ============================================================================

-- ReasoningBank indexes
CREATE INDEX IF NOT EXISTS idx_patterns_session ON reasoningbank_patterns(session_id);
CREATE INDEX IF NOT EXISTS idx_patterns_job ON reasoningbank_patterns(job_id);
CREATE INDEX IF NOT EXISTS idx_patterns_reward ON reasoningbank_patterns(reward DESC);
CREATE INDEX IF NOT EXISTS idx_patterns_success ON reasoningbank_patterns(success);
CREATE INDEX IF NOT EXISTS idx_patterns_created ON reasoningbank_patterns(created_at DESC);

-- Learning episodes indexes
CREATE INDEX IF NOT EXISTS idx_episodes_pattern ON learning_episodes(pattern_id);
CREATE INDEX IF NOT EXISTS idx_episodes_verdict ON learning_episodes(verdict);
CREATE INDEX IF NOT EXISTS idx_episodes_category ON learning_episodes(task_category);
CREATE INDEX IF NOT EXISTS idx_episodes_score ON learning_episodes(judgment_score DESC);

-- Vector embeddings indexes
CREATE INDEX IF NOT EXISTS idx_vectors_source ON vector_embeddings(source_id, source_type);
CREATE INDEX IF NOT EXISTS idx_vectors_cluster ON vector_embeddings(cluster_id);
CREATE INDEX IF NOT EXISTS idx_vectors_hash ON vector_embeddings(content_hash);

-- Memory distillations indexes
CREATE INDEX IF NOT EXISTS idx_distill_category ON memory_distillations(task_category);
CREATE INDEX IF NOT EXISTS idx_distill_confidence ON memory_distillations(confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_distill_usage ON memory_distillations(usage_count DESC);

-- Pattern associations indexes
CREATE INDEX IF NOT EXISTS idx_assoc_pattern_a ON pattern_associations(pattern_id_a);
CREATE INDEX IF NOT EXISTS idx_assoc_pattern_b ON pattern_associations(pattern_id_b);
CREATE INDEX IF NOT EXISTS idx_assoc_similarity ON pattern_associations(similarity_score DESC);

-- ============================================================================
-- AGENTDB VIEWS FOR ANALYTICS
-- ============================================================================

-- Top performing patterns view
CREATE VIEW IF NOT EXISTS top_patterns AS
SELECT
  p.id,
  p.task,
  p.agent_type,
  p.reward,
  p.success,
  p.confidence,
  COUNT(e.id) as episode_count,
  AVG(e.judgment_score) as avg_judgment,
  p.created_at
FROM reasoningbank_patterns p
LEFT JOIN learning_episodes e ON p.id = e.pattern_id
WHERE p.success = 1
GROUP BY p.id
ORDER BY p.reward DESC, avg_judgment DESC
LIMIT 100;

-- Learning progress view
CREATE VIEW IF NOT EXISTS learning_progress AS
SELECT
  task_category,
  COUNT(*) as total_episodes,
  SUM(CASE WHEN verdict = 'success' THEN 1 ELSE 0 END) as successful,
  AVG(judgment_score) as avg_score,
  AVG(improvement_rate) as avg_improvement,
  MAX(completed_at) as last_episode
FROM learning_episodes
GROUP BY task_category
ORDER BY avg_score DESC;

-- Pattern similarity network view
CREATE VIEW IF NOT EXISTS pattern_network AS
SELECT
  pa.id,
  pa.pattern_id_a,
  pa.pattern_id_b,
  pa.similarity_score,
  pa.association_type,
  p1.task as task_a,
  p2.task as task_b,
  pa.learning_value
FROM pattern_associations pa
JOIN reasoningbank_patterns p1 ON pa.pattern_id_a = p1.id
JOIN reasoningbank_patterns p2 ON pa.pattern_id_b = p2.id
WHERE pa.similarity_score > 0.7
ORDER BY pa.similarity_score DESC;
