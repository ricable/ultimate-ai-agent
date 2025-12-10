-- TITAN AgentDB Initialization Script
-- PostgreSQL schema for cognitive memory and vector indexing

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS titan;
CREATE SCHEMA IF NOT EXISTS agentdb;

-- Agent memory table
CREATE TABLE IF NOT EXISTS agentdb.agent_memory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    memory_type VARCHAR(50) NOT NULL, -- 'episodic', 'semantic', 'procedural'
    content JSONB NOT NULL,
    embedding vector(1536), -- OpenAI/Claude embedding dimension
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_agent_memory_agent_id ON agentdb.agent_memory(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_memory_session_id ON agentdb.agent_memory(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_memory_type ON agentdb.agent_memory(memory_type);
CREATE INDEX IF NOT EXISTS idx_agent_memory_created_at ON agentdb.agent_memory(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_memory_embedding ON agentdb.agent_memory USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- RAN optimization history
CREATE TABLE IF NOT EXISTS titan.optimization_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cell_id VARCHAR(50) NOT NULL,
    parameter_name VARCHAR(100) NOT NULL,
    old_value NUMERIC,
    new_value NUMERIC,
    agent_id VARCHAR(255) NOT NULL,
    strategy VARCHAR(100),
    predicted_impact JSONB,
    actual_impact JSONB,
    success BOOLEAN,
    rollback BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_optimization_history_cell_id ON titan.optimization_history(cell_id);
CREATE INDEX IF NOT EXISTS idx_optimization_history_created_at ON titan.optimization_history(created_at DESC);

-- Performance counters
CREATE TABLE IF NOT EXISTS titan.performance_counters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cell_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    counters JSONB NOT NULL,
    rop_period INTEGER, -- Roll-Out Period (1, 2, or 3)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_performance_counters_cell_id ON titan.performance_counters(cell_id);
CREATE INDEX IF NOT EXISTS idx_performance_counters_timestamp ON titan.performance_counters(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_performance_counters_rop ON titan.performance_counters(rop_period);

-- Agent decisions and reflexion
CREATE TABLE IF NOT EXISTS agentdb.agent_decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    decision_type VARCHAR(100) NOT NULL,
    context JSONB NOT NULL,
    decision JSONB NOT NULL,
    reflexion JSONB, -- Self-critique
    outcome JSONB,
    success_score NUMERIC(3,2), -- 0.00 to 1.00
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_decisions_agent_id ON agentdb.agent_decisions(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_created_at ON agentdb.agent_decisions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_success_score ON agentdb.agent_decisions(success_score DESC);

-- Consensus votes (QuDAG)
CREATE TABLE IF NOT EXISTS titan.consensus_votes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    proposal_id UUID NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    vote VARCHAR(20) NOT NULL, -- 'approve', 'reject', 'abstain'
    reasoning TEXT,
    signature TEXT, -- ML-DSA-87 quantum-resistant signature
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_consensus_votes_proposal_id ON titan.consensus_votes(proposal_id);

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA agentdb TO titan;
GRANT ALL PRIVILEGES ON SCHEMA titan TO titan;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA agentdb TO titan;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA titan TO titan;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA agentdb TO titan;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA titan TO titan;

-- Insert seed data for testing
INSERT INTO titan.optimization_history (cell_id, parameter_name, old_value, new_value, agent_id, strategy, success)
VALUES
    ('CELL_001', 'p0Alpha', -70, -65, 'guardian-001', 'gradient-descent', true),
    ('CELL_002', 'qRxLevMin', -120, -115, 'architect-001', 'gnn-optimization', true);

COMMENT ON TABLE agentdb.agent_memory IS 'Persistent memory for AI agents with vector embeddings';
COMMENT ON TABLE titan.optimization_history IS 'History of all RAN parameter optimizations';
COMMENT ON TABLE titan.performance_counters IS 'Performance Management (PM) counter snapshots';
COMMENT ON TABLE agentdb.agent_decisions IS 'Agent decision logs with reflexion for learning';
COMMENT ON TABLE titan.consensus_votes IS 'QuDAG consensus votes for parameter changes';
