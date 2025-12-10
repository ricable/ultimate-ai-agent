-- Titan Cognitive Mesh Memory Schema
-- Compatible with agentdb (postgresql/wasm)

-- Reflexion Store
CREATE TABLE agent_reflexious (
    id UUID PRIMARY KEY,
    agent_id TEXT NOT NULL,
    action_signature TEXT, -- ML-DSA-87 signature for action verification
    outcome_metrics JSONB, -- JSON metrics e.g., { "sinr_delta": 0.5, "bler": 0.01 }
    reflexion_embedding VECTOR(768), -- Embedding for RAG (Requires pgvector)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    context_tags TEXT[] -- Tags for context retrieval
);

-- Consensus Decisions
CREATE TABLE consensus_decisions (
    id UUID PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    vote TEXT CHECK (vote IN ('approve', 'reject', 'abstain')),
    reasoning TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Cluster Quotas
CREATE TABLE cluster_quotas (
    id UUID PRIMARY KEY,
    cluster_id TEXT NOT NULL,
    cell_id TEXT NOT NULL,
    quota_type TEXT NOT NULL, -- 'throughput', 'power', etc.
    value NUMERIC NOT NULL,
    valid_until TIMESTAMP WITH TIME ZONE
);
