-- CRISPR-Cas13 Pipeline Database Schema
-- PostgreSQL 16
-- Version: 1.0.0
-- Last Updated: 2025-10-12

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_trgm"; -- For fuzzy text search

-- =====================================================
-- USERS & AUTHENTICATION
-- =====================================================

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'researcher' CHECK (role IN ('guest', 'researcher', 'bioinformatician', 'admin')),
    institution VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    email_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_login_at TIMESTAMPTZ
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_is_active ON users(is_active);

-- =====================================================
-- EXPERIMENTS
-- =====================================================

CREATE TABLE experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    organism VARCHAR(100) DEFAULT 'Macaca mulatta',
    target_gene VARCHAR(100),
    cas13_variant VARCHAR(50) CHECK (cas13_variant IN ('Cas13a', 'Cas13b', 'Cas13d')),
    guide_rna_sequence TEXT NOT NULL,
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_experiments_user_id ON experiments(user_id);
CREATE INDEX idx_experiments_status ON experiments(status);
CREATE INDEX idx_experiments_created_at ON experiments(created_at DESC);
CREATE INDEX idx_experiments_target_gene ON experiments(target_gene) WHERE target_gene IS NOT NULL;
CREATE INDEX idx_experiments_metadata_gin ON experiments USING gin(metadata jsonb_path_ops);

-- =====================================================
-- SAMPLES
-- =====================================================

CREATE TABLE samples (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    sample_name VARCHAR(255) NOT NULL,
    sample_type VARCHAR(50) CHECK (sample_type IN ('control', 'treatment', 'timepoint_0h', 'timepoint_24h', 'timepoint_48h', 'timepoint_72h')),
    fastq_r1_path TEXT NOT NULL,
    fastq_r2_path TEXT,
    sequencing_platform VARCHAR(100) DEFAULT 'Illumina NovaSeq 6000',
    read_count BIGINT,
    quality_score FLOAT CHECK (quality_score >= 0 AND quality_score <= 100),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT unique_sample_name UNIQUE (experiment_id, sample_name)
);

CREATE INDEX idx_samples_experiment_id ON samples(experiment_id);
CREATE INDEX idx_samples_sample_type ON samples(sample_type);
CREATE INDEX idx_samples_metadata_gin ON samples USING gin(metadata jsonb_path_ops);

-- =====================================================
-- ANALYSIS JOBS
-- =====================================================

CREATE TABLE analysis_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    sample_id UUID REFERENCES samples(id) ON DELETE CASCADE,
    job_type VARCHAR(50) NOT NULL CHECK (job_type IN ('alignment', 'off_target', 'diff_expr', 'immune_response', 'quality_control')),
    status VARCHAR(50) DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled', 'retrying')),
    priority INT DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    worker_pod VARCHAR(255),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    retry_count INT DEFAULT 0,
    max_retries INT DEFAULT 3,
    metrics JSONB DEFAULT '{}', -- execution_time_ms, memory_usage_mb, cpu_usage_percent
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_jobs_experiment_id ON analysis_jobs(experiment_id);
CREATE INDEX idx_jobs_sample_id ON analysis_jobs(sample_id) WHERE sample_id IS NOT NULL;
CREATE INDEX idx_jobs_status ON analysis_jobs(status);
CREATE INDEX idx_jobs_job_type ON analysis_jobs(job_type);
CREATE INDEX idx_jobs_priority ON analysis_jobs(priority DESC) WHERE status = 'queued';
CREATE INDEX idx_jobs_created_at ON analysis_jobs(created_at DESC);

-- =====================================================
-- OFF-TARGET PREDICTIONS
-- =====================================================

CREATE TABLE off_targets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    chromosome VARCHAR(50) NOT NULL,
    position BIGINT NOT NULL CHECK (position > 0),
    strand CHAR(1) CHECK (strand IN ('+', '-')),
    target_sequence TEXT NOT NULL,
    mismatch_count INT NOT NULL CHECK (mismatch_count >= 0),
    cfd_score FLOAT CHECK (cfd_score >= 0 AND cfd_score <= 1),
    mit_score FLOAT CHECK (mit_score >= 0 AND mit_score <= 100),
    gene_symbol VARCHAR(100),
    gene_id VARCHAR(100),
    annotation TEXT,
    distance_to_gene INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_off_targets_job_id ON off_targets(job_id);
CREATE INDEX idx_off_targets_chromosome ON off_targets(chromosome);
CREATE INDEX idx_off_targets_position ON off_targets(position);
CREATE INDEX idx_off_targets_mismatch_count ON off_targets(mismatch_count);
CREATE INDEX idx_off_targets_cfd_score ON off_targets(cfd_score DESC NULLS LAST);
CREATE INDEX idx_off_targets_mit_score ON off_targets(mit_score DESC NULLS LAST);
CREATE INDEX idx_off_targets_gene_symbol ON off_targets(gene_symbol) WHERE gene_symbol IS NOT NULL;
CREATE INDEX idx_off_targets_location ON off_targets(chromosome, position);

-- =====================================================
-- DIFFERENTIAL EXPRESSION RESULTS
-- =====================================================

CREATE TABLE differential_expression (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    gene_id VARCHAR(100) NOT NULL,
    gene_symbol VARCHAR(100),
    gene_name TEXT,
    base_mean FLOAT CHECK (base_mean >= 0),
    log2_fold_change FLOAT NOT NULL,
    lfcse FLOAT CHECK (lfcse >= 0),
    stat FLOAT,
    pvalue FLOAT CHECK (pvalue >= 0 AND pvalue <= 1),
    padj FLOAT CHECK (padj >= 0 AND padj <= 1),
    significant BOOLEAN GENERATED ALWAYS AS (padj < 0.05 AND padj IS NOT NULL) STORED,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_diff_expr_job_id ON differential_expression(job_id);
CREATE INDEX idx_diff_expr_gene_id ON differential_expression(gene_id);
CREATE INDEX idx_diff_expr_gene_symbol ON differential_expression(gene_symbol) WHERE gene_symbol IS NOT NULL;
CREATE INDEX idx_diff_expr_padj ON differential_expression(padj ASC NULLS LAST);
CREATE INDEX idx_diff_expr_log2fc ON differential_expression(log2_fold_change);
CREATE INDEX idx_diff_expr_significant ON differential_expression(significant) WHERE significant = TRUE;

-- =====================================================
-- IMMUNE RESPONSE SIGNATURES
-- =====================================================

CREATE TABLE immune_signatures (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    signature_name VARCHAR(255) NOT NULL,
    signature_source VARCHAR(100) CHECK (signature_source IN ('ImmuneSigDB', 'MSigDB', 'Reactome', 'KEGG', 'GO')),
    signature_id VARCHAR(100),
    enrichment_score FLOAT NOT NULL,
    normalized_enrichment_score FLOAT,
    pvalue FLOAT CHECK (pvalue >= 0 AND pvalue <= 1),
    fdr FLOAT CHECK (fdr >= 0 AND fdr <= 1),
    leading_edge_genes TEXT[], -- Array of gene symbols
    gene_set_size INT,
    overlap_size INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_immune_sig_job_id ON immune_signatures(job_id);
CREATE INDEX idx_immune_sig_name ON immune_signatures(signature_name);
CREATE INDEX idx_immune_sig_source ON immune_signatures(signature_source);
CREATE INDEX idx_immune_sig_fdr ON immune_signatures(fdr ASC NULLS LAST);
CREATE INDEX idx_immune_sig_nes ON immune_signatures(normalized_enrichment_score DESC NULLS LAST);

-- =====================================================
-- AUDIT LOG (Partitioned by Month)
-- =====================================================

CREATE TABLE audit_log (
    id BIGSERIAL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id UUID,
    ip_address INET,
    user_agent TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,

    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

CREATE INDEX idx_audit_log_user_id ON audit_log(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX idx_audit_log_action ON audit_log(action);
CREATE INDEX idx_audit_log_created_at ON audit_log(created_at DESC);
CREATE INDEX idx_audit_log_resource ON audit_log(resource_type, resource_id) WHERE resource_id IS NOT NULL;

-- Create initial partitions (2025)
CREATE TABLE audit_log_2025_01 PARTITION OF audit_log
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE audit_log_2025_02 PARTITION OF audit_log
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE audit_log_2025_03 PARTITION OF audit_log
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');
CREATE TABLE audit_log_2025_04 PARTITION OF audit_log
    FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');
CREATE TABLE audit_log_2025_05 PARTITION OF audit_log
    FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');
CREATE TABLE audit_log_2025_06 PARTITION OF audit_log
    FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');
CREATE TABLE audit_log_2025_07 PARTITION OF audit_log
    FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');
CREATE TABLE audit_log_2025_08 PARTITION OF audit_log
    FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE audit_log_2025_09 PARTITION OF audit_log
    FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE audit_log_2025_10 PARTITION OF audit_log
    FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE audit_log_2025_11 PARTITION OF audit_log
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE audit_log_2025_12 PARTITION OF audit_log
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

-- =====================================================
-- MATERIALIZED VIEWS
-- =====================================================

-- Experiment summary statistics
CREATE MATERIALIZED VIEW experiment_summary AS
SELECT
    e.id,
    e.name,
    e.status,
    COUNT(DISTINCT s.id) AS sample_count,
    COUNT(DISTINCT aj.id) AS job_count,
    COUNT(DISTINCT aj.id) FILTER (WHERE aj.status = 'completed') AS completed_jobs,
    COUNT(DISTINCT aj.id) FILTER (WHERE aj.status = 'failed') AS failed_jobs,
    e.created_at,
    e.completed_at
FROM experiments e
LEFT JOIN samples s ON e.id = s.experiment_id
LEFT JOIN analysis_jobs aj ON e.id = aj.experiment_id
GROUP BY e.id, e.name, e.status, e.created_at, e.completed_at;

CREATE UNIQUE INDEX idx_experiment_summary_id ON experiment_summary(id);

-- User activity statistics
CREATE MATERIALIZED VIEW user_activity_stats AS
SELECT
    u.id,
    u.email,
    u.full_name,
    u.role,
    COUNT(DISTINCT e.id) AS experiment_count,
    COUNT(DISTINCT aj.id) AS total_jobs,
    MAX(e.created_at) AS last_experiment_date,
    MAX(u.last_login_at) AS last_login
FROM users u
LEFT JOIN experiments e ON u.id = e.user_id
LEFT JOIN analysis_jobs aj ON e.id = aj.experiment_id
GROUP BY u.id, u.email, u.full_name, u.role;

CREATE UNIQUE INDEX idx_user_activity_id ON user_activity_stats(id);

-- =====================================================
-- FUNCTIONS & TRIGGERS
-- =====================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at trigger to tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_experiments_updated_at BEFORE UPDATE ON experiments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_samples_updated_at BEFORE UPDATE ON samples
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_jobs_updated_at BEFORE UPDATE ON analysis_jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to create audit log entries
CREATE OR REPLACE FUNCTION audit_log_trigger()
RETURNS TRIGGER AS $$
BEGIN
    IF (TG_OP = 'INSERT') THEN
        INSERT INTO audit_log (user_id, action, resource_type, resource_id, metadata)
        VALUES (NEW.user_id, TG_OP, TG_TABLE_NAME, NEW.id, row_to_json(NEW)::jsonb);
        RETURN NEW;
    ELSIF (TG_OP = 'UPDATE') THEN
        INSERT INTO audit_log (user_id, action, resource_type, resource_id, metadata)
        VALUES (NEW.user_id, TG_OP, TG_TABLE_NAME, NEW.id, jsonb_build_object('old', row_to_json(OLD), 'new', row_to_json(NEW)));
        RETURN NEW;
    ELSIF (TG_OP = 'DELETE') THEN
        INSERT INTO audit_log (user_id, action, resource_type, resource_id, metadata)
        VALUES (OLD.user_id, TG_OP, TG_TABLE_NAME, OLD.id, row_to_json(OLD)::jsonb);
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Apply audit trigger to experiments
CREATE TRIGGER audit_experiments AFTER INSERT OR UPDATE OR DELETE ON experiments
    FOR EACH ROW EXECUTE FUNCTION audit_log_trigger();

-- =====================================================
-- SAMPLE DATA (for development/testing)
-- =====================================================

-- Insert admin user
INSERT INTO users (email, full_name, password_hash, role, email_verified)
VALUES ('admin@example.com', 'Admin User', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYIwNoK8JW2', 'admin', TRUE);

-- Insert researcher user
INSERT INTO users (email, full_name, password_hash, role, email_verified)
VALUES ('researcher@example.com', 'Research Scientist', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYIwNoK8JW2', 'researcher', TRUE);

-- =====================================================
-- GRANTS & PERMISSIONS
-- =====================================================

-- Create application role
CREATE ROLE crispr_app;

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO crispr_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO crispr_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO crispr_app;

-- Grant user to application role
GRANT crispr_app TO crispr_user;
