-- CRISPR-Cas13 Bioinformatics Pipeline - PostgreSQL Database Schema
-- Version: 1.0.0
-- Date: 2025-10-12
-- PostgreSQL 15+ required
-- Extensions: uuid-ossp, pg_trgm (for full-text search)

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ============================================================================
-- CORE TABLES
-- ============================================================================

-- Users table (authentication and authorization)
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL CHECK (username ~ '^[a-z0-9_-]{3,20}$'),
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    institution VARCHAR(200),
    role VARCHAR(20) NOT NULL CHECK (role IN ('admin', 'researcher', 'analyst', 'guest')) DEFAULT 'researcher',
    permissions TEXT[] DEFAULT ARRAY['submit_jobs']::TEXT[],
    api_key_hash VARCHAR(255),
    api_key_created_at TIMESTAMP WITH TIME ZONE,
    last_login TIMESTAMP WITH TIME ZONE,
    account_status VARCHAR(20) NOT NULL CHECK (account_status IN ('active', 'suspended', 'deleted')) DEFAULT 'active',
    storage_quota_gb INTEGER DEFAULT 1000,
    storage_used_gb NUMERIC(10, 2) DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_storage_quota CHECK (storage_quota_gb > 0),
    CONSTRAINT storage_within_quota CHECK (storage_used_gb <= storage_quota_gb)
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_account_status ON users(account_status);

-- Reference genomes table
CREATE TABLE reference_genomes (
    genome_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    species VARCHAR(100) NOT NULL,
    assembly_name VARCHAR(50) NOT NULL,
    assembly_accession VARCHAR(50),
    source VARCHAR(20) NOT NULL CHECK (source IN ('NCBI', 'Ensembl', 'UCSC', 'custom')),
    ensembl_version INTEGER,
    genome_fasta_uri TEXT NOT NULL,
    annotation_gtf_uri TEXT NOT NULL,
    transcriptome_fasta_uri TEXT,
    star_index_uri TEXT,
    hisat2_index_uri TEXT,
    bowtie2_index_uri TEXT,
    total_genome_size BIGINT,
    gene_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_assembly UNIQUE (species, assembly_name, source)
);

CREATE INDEX idx_reference_genomes_species ON reference_genomes(species);
CREATE INDEX idx_reference_genomes_assembly ON reference_genomes(assembly_name);

-- Experiments table
CREATE TABLE experiments (
    experiment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    species VARCHAR(50) NOT NULL CHECK (species IN ('rhesus_macaque', 'cynomolgus_macaque', 'human_cell_line', 'mouse', 'custom')),
    ncbi_taxonomy_id INTEGER,
    tissue_type VARCHAR(50),
    cas13_variant VARCHAR(10) NOT NULL CHECK (cas13_variant IN ('Cas13a', 'Cas13b', 'Cas13c', 'Cas13d')),
    delivery_method VARCHAR(50),
    timepoint_hours INTEGER CHECK (timepoint_hours >= 0),
    replicate_number INTEGER CHECK (replicate_number >= 1),
    control_type VARCHAR(50) CHECK (control_type IN ('mock_treated', 'non_targeting_grna', 'untreated', 'positive_control')),
    experimenter VARCHAR(100),
    institution VARCHAR(200),
    iacuc_protocol VARCHAR(50),
    date_performed TIMESTAMP WITH TIME ZONE NOT NULL,
    metadata JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_experiments_user_id ON experiments(user_id);
CREATE INDEX idx_experiments_species ON experiments(species);
CREATE INDEX idx_experiments_cas13_variant ON experiments(cas13_variant);
CREATE INDEX idx_experiments_date_performed ON experiments(date_performed);
CREATE INDEX idx_experiments_tissue_type ON experiments(tissue_type);
CREATE INDEX idx_experiments_metadata ON experiments USING GIN (metadata);
CREATE INDEX idx_experiments_title_trgm ON experiments USING GIN (title gin_trgm_ops);

-- Guide RNA table
CREATE TABLE guide_rnas (
    grna_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    sequence_rna VARCHAR(30) NOT NULL CHECK (sequence_rna ~ '^[ACGTU]{22,30}$'),
    sequence_dna VARCHAR(30) NOT NULL CHECK (sequence_dna ~ '^[ACGT]{22,30}$'),
    length INTEGER NOT NULL CHECK (length BETWEEN 22 AND 30),
    target_gene VARCHAR(50) NOT NULL,
    target_transcript_id VARCHAR(50),
    target_transcript_biotype VARCHAR(50),
    protospacer_chromosome VARCHAR(10),
    protospacer_start INTEGER,
    protospacer_end INTEGER,
    protospacer_strand CHAR(1) CHECK (protospacer_strand IN ('+', '-')),
    pfs_sequence VARCHAR(6),
    gc_content NUMERIC(5, 4) CHECK (gc_content BETWEEN 0 AND 1),
    secondary_structure_dot_bracket TEXT,
    secondary_structure_mfe NUMERIC(8, 2),
    off_target_potential VARCHAR(10) CHECK (off_target_potential IN ('low', 'medium', 'high')),
    design_tool VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_grna_per_experiment UNIQUE (experiment_id)
);

CREATE INDEX idx_guide_rnas_experiment_id ON guide_rnas(experiment_id);
CREATE INDEX idx_guide_rnas_target_gene ON guide_rnas(target_gene);
CREATE INDEX idx_guide_rnas_sequence_rna ON guide_rnas(sequence_rna);

-- Sequencing runs table
CREATE TABLE sequencing_runs (
    run_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    sample_id VARCHAR(100) NOT NULL,
    sequencing_platform VARCHAR(100) NOT NULL,
    library_prep VARCHAR(100),
    read_type VARCHAR(20) NOT NULL CHECK (read_type IN ('single_end', 'paired_end')),
    read_length INTEGER CHECK (read_length BETWEEN 50 AND 300),
    total_reads BIGINT CHECK (total_reads >= 0),
    flowcell_id VARCHAR(50),
    lane INTEGER CHECK (lane BETWEEN 1 AND 8),
    barcode_index VARCHAR(12) CHECK (barcode_index ~ '^[ACGT]+$'),
    sequencing_date TIMESTAMP WITH TIME ZONE,
    sequencing_facility VARCHAR(200),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_sample_per_experiment UNIQUE (experiment_id, sample_id)
);

CREATE INDEX idx_sequencing_runs_experiment_id ON sequencing_runs(experiment_id);
CREATE INDEX idx_sequencing_runs_sample_id ON sequencing_runs(sample_id);
CREATE INDEX idx_sequencing_runs_platform ON sequencing_runs(sequencing_platform);

-- FASTQ files table
CREATE TABLE fastq_files (
    file_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sequencing_run_id UUID NOT NULL REFERENCES sequencing_runs(run_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    read_pair VARCHAR(10) NOT NULL CHECK (read_pair IN ('R1', 'R2', 'single')),
    file_size BIGINT NOT NULL CHECK (file_size > 0),
    compression VARCHAR(20) NOT NULL CHECK (compression IN ('gzip', 'bzip2', 'uncompressed')) DEFAULT 'gzip',
    md5_checksum CHAR(32) NOT NULL CHECK (md5_checksum ~ '^[a-f0-9]{32}$'),
    sha256_checksum CHAR(64) CHECK (sha256_checksum ~ '^[a-f0-9]{64}$'),
    quality_encoding VARCHAR(20) NOT NULL CHECK (quality_encoding IN ('Phred33', 'Phred64')) DEFAULT 'Phred33',
    upload_date TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    uploaded_by UUID REFERENCES users(user_id),

    CONSTRAINT unique_file_path UNIQUE (file_path)
);

CREATE INDEX idx_fastq_files_sequencing_run_id ON fastq_files(sequencing_run_id);
CREATE INDEX idx_fastq_files_md5 ON fastq_files(md5_checksum);

-- Quality control table
CREATE TABLE quality_control (
    qc_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sequencing_run_id UUID NOT NULL REFERENCES sequencing_runs(run_id) ON DELETE CASCADE,
    status VARCHAR(10) NOT NULL CHECK (status IN ('pass', 'warn', 'fail')),
    total_sequences BIGINT,
    poor_quality_sequences BIGINT,
    sequence_length_min INTEGER,
    sequence_length_max INTEGER,
    sequence_length_mean NUMERIC(8, 2),
    mean_quality_score NUMERIC(5, 2) CHECK (mean_quality_score BETWEEN 0 AND 42),
    gc_content NUMERIC(5, 2) CHECK (gc_content BETWEEN 0 AND 100),
    adapter_content_illumina_universal NUMERIC(5, 2),
    adapter_content_illumina_small_rna NUMERIC(5, 2),
    adapter_content_nextera NUMERIC(5, 2),
    duplication_rate NUMERIC(5, 2) CHECK (duplication_rate BETWEEN 0 AND 100),
    per_base_quality JSONB,
    overrepresented_sequences JSONB,
    report_html_uri TEXT,
    report_json_uri TEXT,
    fastqc_version VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_qc_per_run UNIQUE (sequencing_run_id)
);

CREATE INDEX idx_quality_control_sequencing_run_id ON quality_control(sequencing_run_id);
CREATE INDEX idx_quality_control_status ON quality_control(status);

-- Alignments table
CREATE TABLE alignments (
    alignment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sequencing_run_id UUID NOT NULL REFERENCES sequencing_runs(run_id) ON DELETE CASCADE,
    reference_genome_id UUID NOT NULL REFERENCES reference_genomes(genome_id),
    aligner VARCHAR(20) NOT NULL CHECK (aligner IN ('STAR', 'HISAT2', 'Bowtie2', 'BWA-MEM', 'minimap2')),
    aligner_version VARCHAR(20) NOT NULL,
    alignment_parameters JSONB NOT NULL DEFAULT '{}'::JSONB,
    bam_file_uri TEXT NOT NULL,
    bai_file_uri TEXT,
    total_reads BIGINT,
    mapped_reads BIGINT,
    uniquely_mapped_reads BIGINT,
    multi_mapped_reads BIGINT,
    unmapped_reads BIGINT,
    alignment_rate NUMERIC(5, 4) CHECK (alignment_rate BETWEEN 0 AND 1),
    mean_mapping_quality NUMERIC(5, 2) CHECK (mean_mapping_quality BETWEEN 0 AND 60),
    insert_size_mean NUMERIC(8, 2),
    insert_size_median NUMERIC(8, 2),
    insert_size_std_dev NUMERIC(8, 2),
    mean_coverage NUMERIC(10, 2),
    median_coverage NUMERIC(10, 2),
    covered_bases BIGINT,
    execution_time_seconds INTEGER,
    peak_memory_gb NUMERIC(8, 2),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_alignment_per_run UNIQUE (sequencing_run_id, reference_genome_id, aligner),
    CONSTRAINT valid_read_counts CHECK (
        total_reads = (mapped_reads + unmapped_reads) AND
        mapped_reads = (uniquely_mapped_reads + multi_mapped_reads)
    )
);

CREATE INDEX idx_alignments_sequencing_run_id ON alignments(sequencing_run_id);
CREATE INDEX idx_alignments_reference_genome_id ON alignments(reference_genome_id);
CREATE INDEX idx_alignments_aligner ON alignments(aligner);

-- Target analysis table (on-target activity)
CREATE TABLE target_analysis (
    analysis_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    alignment_id UUID REFERENCES alignments(alignment_id),
    target_gene VARCHAR(50) NOT NULL,
    target_transcript_id VARCHAR(50),
    control_expression_tpm NUMERIC(12, 4) CHECK (control_expression_tpm >= 0),
    treated_expression_tpm NUMERIC(12, 4) CHECK (treated_expression_tpm >= 0),
    fold_change NUMERIC(10, 4),
    log2_fold_change NUMERIC(10, 4),
    knockdown_efficiency NUMERIC(5, 4) CHECK (knockdown_efficiency BETWEEN 0 AND 1),
    p_value DOUBLE PRECISION CHECK (p_value BETWEEN 0 AND 1),
    adjusted_p_value DOUBLE PRECISION CHECK (adjusted_p_value BETWEEN 0 AND 1),
    control_read_count BIGINT,
    treated_read_count BIGINT,
    total_library_size_control BIGINT,
    total_library_size_treated BIGINT,
    transcript_abundance JSONB,
    confidence_level VARCHAR(10) CHECK (confidence_level IN ('high', 'medium', 'low')),
    statistical_method VARCHAR(20) CHECK (statistical_method IN ('DESeq2', 'edgeR', 'limma-voom', 'NOISeq')),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_target_analysis UNIQUE (experiment_id, target_gene)
);

CREATE INDEX idx_target_analysis_experiment_id ON target_analysis(experiment_id);
CREATE INDEX idx_target_analysis_target_gene ON target_analysis(target_gene);
CREATE INDEX idx_target_analysis_p_value ON target_analysis(p_value);
CREATE INDEX idx_target_analysis_knockdown_efficiency ON target_analysis(knockdown_efficiency);

-- Off-target sites table
CREATE TABLE off_target_sites (
    offtarget_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    rank INTEGER NOT NULL CHECK (rank >= 1),
    confidence_score NUMERIC(5, 4) NOT NULL CHECK (confidence_score BETWEEN 0 AND 1),
    gene_symbol VARCHAR(50),
    gene_id VARCHAR(50),
    transcript_id VARCHAR(50),
    chromosome VARCHAR(10) NOT NULL,
    genomic_start INTEGER NOT NULL CHECK (genomic_start > 0),
    genomic_end INTEGER NOT NULL CHECK (genomic_end > genomic_start),
    strand CHAR(1) NOT NULL CHECK (strand IN ('+', '-')),
    aligned_sequence VARCHAR(30) NOT NULL,
    guide_alignment VARCHAR(30) NOT NULL,
    target_alignment VARCHAR(30) NOT NULL,
    alignment_visual VARCHAR(30),
    mismatch_count INTEGER NOT NULL CHECK (mismatch_count >= 0 AND mismatch_count <= 10),
    mismatch_positions INTEGER[],
    seed_mismatches INTEGER CHECK (seed_mismatches >= 0),
    pam_flanking_sequence VARCHAR(6),
    sequence_context TEXT,
    accessibility_score NUMERIC(5, 4) CHECK (accessibility_score BETWEEN 0 AND 1),
    secondary_structure_dot_bracket TEXT,
    secondary_structure_mfe NUMERIC(8, 2),
    expression_level NUMERIC(12, 4) CHECK (expression_level >= 0),
    observed_cleavage BOOLEAN DEFAULT FALSE,
    fold_change NUMERIC(10, 4),
    prediction_method VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_off_target_rank UNIQUE (experiment_id, rank)
);

CREATE INDEX idx_off_target_sites_experiment_id ON off_target_sites(experiment_id);
CREATE INDEX idx_off_target_sites_confidence_score ON off_target_sites(confidence_score DESC);
CREATE INDEX idx_off_target_sites_gene_symbol ON off_target_sites(gene_symbol);
CREATE INDEX idx_off_target_sites_genomic_location ON off_target_sites(chromosome, genomic_start, genomic_end);

-- Immune response analysis table
CREATE TABLE immune_response (
    immune_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    overall_immune_score NUMERIC(8, 4),
    interferon_score NUMERIC(8, 4),
    activated_pathways TEXT[],
    differential_genes JSONB,
    gene_set_enrichment JSONB,
    key_cytokines JSONB,
    pattern_recognition_receptors JSONB,
    interferon_stimulated_genes JSONB,
    heatmap_uri TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_immune_analysis UNIQUE (experiment_id)
);

CREATE INDEX idx_immune_response_experiment_id ON immune_response(experiment_id);
CREATE INDEX idx_immune_response_overall_score ON immune_response(overall_immune_score);
CREATE INDEX idx_immune_response_activated_pathways ON immune_response USING GIN (activated_pathways);

-- Processing jobs table
CREATE TABLE processing_jobs (
    job_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_type VARCHAR(50) NOT NULL CHECK (job_type IN ('qc_only', 'alignment', 'full_pipeline', 'off_target_prediction', 'immune_analysis', 'custom')),
    status VARCHAR(20) NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')) DEFAULT 'queued',
    priority VARCHAR(10) NOT NULL CHECK (priority IN ('low', 'normal', 'high', 'urgent')) DEFAULT 'normal',
    experiment_id UUID REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    sequencing_run_id UUID REFERENCES sequencing_runs(run_id) ON DELETE CASCADE,
    reference_genome_id UUID REFERENCES reference_genomes(genome_id),
    input_files TEXT[],
    output_files TEXT[],
    parameters JSONB DEFAULT '{}'::JSONB,
    progress NUMERIC(5, 4) CHECK (progress BETWEEN 0 AND 1) DEFAULT 0,
    current_stage VARCHAR(100),
    eta_seconds INTEGER,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    execution_time_seconds INTEGER,
    error_message TEXT,
    error_code VARCHAR(50),
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    cpu_hours NUMERIC(10, 4),
    peak_memory_gb NUMERIC(10, 4),
    disk_io_gb NUMERIC(10, 4),
    gpu_hours NUMERIC(10, 4),
    worker_node VARCHAR(100),
    container_image VARCHAR(200),
    notification_email VARCHAR(255),
    created_by UUID NOT NULL REFERENCES users(user_id),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_completion CHECK (
        (status = 'completed' AND completed_at IS NOT NULL) OR
        (status != 'completed' AND completed_at IS NULL)
    ),
    CONSTRAINT valid_start_time CHECK (
        (status IN ('running', 'completed', 'failed', 'cancelled') AND started_at IS NOT NULL) OR
        (status = 'queued' AND started_at IS NULL)
    )
);

CREATE INDEX idx_processing_jobs_status ON processing_jobs(status);
CREATE INDEX idx_processing_jobs_experiment_id ON processing_jobs(experiment_id);
CREATE INDEX idx_processing_jobs_created_by ON processing_jobs(created_by);
CREATE INDEX idx_processing_jobs_created_at ON processing_jobs(created_at DESC);
CREATE INDEX idx_processing_jobs_priority ON processing_jobs(priority);

-- Provenance table (W3C PROV-O)
CREATE TABLE provenance (
    provenance_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id VARCHAR(255) NOT NULL,
    entity_type VARCHAR(50) NOT NULL CHECK (entity_type IN ('fastq_file', 'bam_file', 'vcf_file', 'expression_matrix', 'qc_report', 'analysis_result')),
    activity_id VARCHAR(255) NOT NULL,
    activity_type VARCHAR(50) NOT NULL CHECK (activity_type IN ('quality_control', 'trimming', 'alignment', 'quantification', 'differential_expression', 'off_target_prediction')),
    agent_id VARCHAR(255) NOT NULL,
    agent_type VARCHAR(50) NOT NULL CHECK (agent_type IN ('software_tool', 'user', 'automated_pipeline')),
    tool_name VARCHAR(100),
    tool_version VARCHAR(50),
    tool_parameters JSONB,
    input_entities TEXT[],
    output_entities TEXT[],
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    ended_at TIMESTAMP WITH TIME ZONE,
    execution_environment JSONB,
    provenance_graph_rdf TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_provenance_entity_id ON provenance(entity_id);
CREATE INDEX idx_provenance_activity_id ON provenance(activity_id);
CREATE INDEX idx_provenance_agent_id ON provenance(agent_id);
CREATE INDEX idx_provenance_entity_type ON provenance(entity_type);
CREATE INDEX idx_provenance_activity_type ON provenance(activity_type);

-- Audit log table
CREATE TABLE audit_log (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL,
    user_id UUID REFERENCES users(user_id),
    resource_type VARCHAR(50),
    resource_id VARCHAR(255),
    action VARCHAR(20),
    ip_address INET,
    user_agent TEXT,
    request_method VARCHAR(10),
    request_path TEXT,
    response_status INTEGER,
    details JSONB,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('info', 'warning', 'error', 'critical')) DEFAULT 'info'
);

CREATE INDEX idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX idx_audit_log_event_type ON audit_log(event_type);
CREATE INDEX idx_audit_log_timestamp ON audit_log(timestamp DESC);
CREATE INDEX idx_audit_log_resource ON audit_log(resource_type, resource_id);
CREATE INDEX idx_audit_log_severity ON audit_log(severity);

-- ============================================================================
-- VIEWS
-- ============================================================================

-- Complete experiment view (denormalized for easy querying)
CREATE OR REPLACE VIEW v_experiments_complete AS
SELECT
    e.experiment_id,
    e.title,
    e.description,
    e.species,
    e.tissue_type,
    e.cas13_variant,
    e.date_performed,
    e.experimenter,
    e.institution,
    u.username AS created_by_username,
    g.sequence_rna AS guide_rna_sequence,
    g.target_gene,
    g.target_transcript_id,
    COUNT(DISTINCT sr.run_id) AS sequencing_runs_count,
    COUNT(DISTINCT pj.job_id) AS jobs_count,
    COUNT(DISTINCT CASE WHEN pj.status = 'completed' THEN pj.job_id END) AS completed_jobs_count,
    EXISTS(SELECT 1 FROM target_analysis ta WHERE ta.experiment_id = e.experiment_id) AS has_target_analysis,
    EXISTS(SELECT 1 FROM off_target_sites ots WHERE ots.experiment_id = e.experiment_id) AS has_offtarget_analysis,
    EXISTS(SELECT 1 FROM immune_response ir WHERE ir.experiment_id = e.experiment_id) AS has_immune_analysis,
    e.created_at,
    e.updated_at
FROM
    experiments e
    INNER JOIN users u ON e.user_id = u.user_id
    LEFT JOIN guide_rnas g ON e.experiment_id = g.experiment_id
    LEFT JOIN sequencing_runs sr ON e.experiment_id = sr.experiment_id
    LEFT JOIN processing_jobs pj ON e.experiment_id = pj.experiment_id
GROUP BY
    e.experiment_id, e.title, e.description, e.species, e.tissue_type,
    e.cas13_variant, e.date_performed, e.experimenter, e.institution,
    u.username, g.sequence_rna, g.target_gene, g.target_transcript_id,
    e.created_at, e.updated_at;

-- Active jobs view
CREATE OR REPLACE VIEW v_active_jobs AS
SELECT
    pj.job_id,
    pj.job_type,
    pj.status,
    pj.priority,
    pj.progress,
    pj.current_stage,
    pj.eta_seconds,
    e.title AS experiment_title,
    e.experiment_id,
    u.username AS created_by_username,
    pj.started_at,
    pj.created_at,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - pj.started_at))::INTEGER AS running_time_seconds
FROM
    processing_jobs pj
    INNER JOIN users u ON pj.created_by = u.user_id
    LEFT JOIN experiments e ON pj.experiment_id = e.experiment_id
WHERE
    pj.status IN ('queued', 'running')
ORDER BY
    CASE pj.priority
        WHEN 'urgent' THEN 1
        WHEN 'high' THEN 2
        WHEN 'normal' THEN 3
        WHEN 'low' THEN 4
    END,
    pj.created_at;

-- User storage summary view
CREATE OR REPLACE VIEW v_user_storage AS
SELECT
    u.user_id,
    u.username,
    u.email,
    u.storage_quota_gb,
    u.storage_used_gb,
    ROUND((u.storage_used_gb / u.storage_quota_gb * 100)::NUMERIC, 2) AS storage_used_percent,
    COUNT(DISTINCT e.experiment_id) AS total_experiments,
    COUNT(DISTINCT sr.run_id) AS total_sequencing_runs,
    COUNT(DISTINCT ff.file_id) AS total_files,
    SUM(ff.file_size) / (1024.0 * 1024 * 1024) AS calculated_storage_gb
FROM
    users u
    LEFT JOIN experiments e ON u.user_id = e.user_id
    LEFT JOIN sequencing_runs sr ON e.experiment_id = sr.experiment_id
    LEFT JOIN fastq_files ff ON sr.run_id = ff.sequencing_run_id
WHERE
    u.account_status = 'active'
GROUP BY
    u.user_id, u.username, u.email, u.storage_quota_gb, u.storage_used_gb;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function to automatically update 'updated_at' timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for experiments table
CREATE TRIGGER trigger_experiments_updated_at
BEFORE UPDATE ON experiments
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Trigger for users table
CREATE TRIGGER trigger_users_updated_at
BEFORE UPDATE ON users
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Function to validate guide RNA sequence (DNA/RNA conversion)
CREATE OR REPLACE FUNCTION validate_guide_rna()
RETURNS TRIGGER AS $$
BEGIN
    -- Convert RNA to DNA by replacing U with T
    NEW.sequence_dna = REPLACE(NEW.sequence_rna, 'U', 'T');

    -- Calculate length
    NEW.length = LENGTH(NEW.sequence_rna);

    -- Calculate GC content
    NEW.gc_content = (
        (LENGTH(NEW.sequence_dna) - LENGTH(REPLACE(REPLACE(NEW.sequence_dna, 'G', ''), 'C', '')))::NUMERIC /
        LENGTH(NEW.sequence_dna)::NUMERIC
    );

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_validate_guide_rna
BEFORE INSERT OR UPDATE ON guide_rnas
FOR EACH ROW
EXECUTE FUNCTION validate_guide_rna();

-- Function to update user storage usage
CREATE OR REPLACE FUNCTION update_user_storage()
RETURNS TRIGGER AS $$
DECLARE
    v_user_id UUID;
    v_total_storage NUMERIC;
BEGIN
    -- Get user_id from experiment chain
    SELECT e.user_id INTO v_user_id
    FROM sequencing_runs sr
    INNER JOIN experiments e ON sr.experiment_id = e.experiment_id
    WHERE sr.run_id = NEW.sequencing_run_id;

    -- Calculate total storage for user
    SELECT SUM(ff.file_size) / (1024.0 * 1024 * 1024) INTO v_total_storage
    FROM fastq_files ff
    INNER JOIN sequencing_runs sr ON ff.sequencing_run_id = sr.run_id
    INNER JOIN experiments e ON sr.experiment_id = e.experiment_id
    WHERE e.user_id = v_user_id;

    -- Update user storage
    UPDATE users SET storage_used_gb = COALESCE(v_total_storage, 0)
    WHERE user_id = v_user_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_user_storage
AFTER INSERT ON fastq_files
FOR EACH ROW
EXECUTE FUNCTION update_user_storage();

-- Function to automatically set job execution time
CREATE OR REPLACE FUNCTION set_job_execution_time()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status IN ('completed', 'failed', 'cancelled') AND NEW.started_at IS NOT NULL THEN
        NEW.execution_time_seconds = EXTRACT(EPOCH FROM (NEW.completed_at - NEW.started_at))::INTEGER;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_set_job_execution_time
BEFORE UPDATE ON processing_jobs
FOR EACH ROW
WHEN (NEW.status IN ('completed', 'failed', 'cancelled') AND OLD.status != NEW.status)
EXECUTE FUNCTION set_job_execution_time();

-- ============================================================================
-- PARTITIONING (for large-scale deployments)
-- ============================================================================

-- Partition audit_log by month for better performance
-- Uncomment if expecting >1M audit log entries

-- CREATE TABLE audit_log_template (LIKE audit_log INCLUDING ALL);
--
-- CREATE TABLE audit_log_2025_10 PARTITION OF audit_log_template
-- FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
--
-- -- Create partitions for future months as needed

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Insert default admin user (change password immediately!)
INSERT INTO users (email, username, password_hash, full_name, role, permissions)
VALUES (
    'admin@crispr-pipeline.org',
    'admin',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5oPxZ8l4RMVeC',  -- bcrypt hash of 'changeme123'
    'System Administrator',
    'admin',
    ARRAY['submit_jobs', 'view_all_experiments', 'delete_data', 'manage_users', 'modify_settings']::TEXT[]
);

-- Insert common reference genomes
INSERT INTO reference_genomes (species, assembly_name, assembly_accession, source, genome_fasta_uri, annotation_gtf_uri, transcriptome_fasta_uri, total_genome_size, gene_count)
VALUES
    (
        'Macaca mulatta',
        'rheMac10',
        'GCF_003339765.1',
        'NCBI',
        's3://reference-genomes/rheMac10/genome.fa.gz',
        's3://reference-genomes/rheMac10/genes.gtf.gz',
        's3://reference-genomes/rheMac10/transcriptome.fa.gz',
        2971321462,
        21156
    ),
    (
        'Macaca fascicularis',
        'macFas5',
        'GCF_000364345.1',
        'NCBI',
        's3://reference-genomes/macFas5/genome.fa.gz',
        's3://reference-genomes/macFas5/genes.gtf.gz',
        's3://reference-genomes/macFas5/transcriptome.fa.gz',
        2887435523,
        20317
    ),
    (
        'Homo sapiens',
        'GRCh38',
        'GCA_000001405.29',
        'Ensembl',
        's3://reference-genomes/GRCh38/genome.fa.gz',
        's3://reference-genomes/GRCh38/genes.gtf.gz',
        's3://reference-genomes/GRCh38/transcriptome.fa.gz',
        3099750718,
        20449
    );

-- ============================================================================
-- GRANTS (adjust based on your security model)
-- ============================================================================

-- Create application database user
-- CREATE USER crispr_app WITH PASSWORD 'your_secure_password';

-- Grant permissions
-- GRANT CONNECT ON DATABASE crispr_pipeline TO crispr_app;
-- GRANT USAGE ON SCHEMA public TO crispr_app;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO crispr_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO crispr_app;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO crispr_app;

-- Create read-only analyst role
-- CREATE USER crispr_analyst WITH PASSWORD 'analyst_password';
-- GRANT CONNECT ON DATABASE crispr_pipeline TO crispr_analyst;
-- GRANT USAGE ON SCHEMA public TO crispr_analyst;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO crispr_analyst;

-- ============================================================================
-- COMMENTS (for documentation)
-- ============================================================================

COMMENT ON TABLE experiments IS 'Core experiment metadata including CRISPR-Cas13 design parameters';
COMMENT ON TABLE guide_rnas IS 'Guide RNA sequences and target information';
COMMENT ON TABLE sequencing_runs IS 'Sequencing run metadata linking FASTQ files to experiments';
COMMENT ON TABLE quality_control IS 'FastQC quality metrics for sequencing data';
COMMENT ON TABLE alignments IS 'Read alignment results against reference transcriptome';
COMMENT ON TABLE target_analysis IS 'On-target knockdown efficiency and statistical analysis';
COMMENT ON TABLE off_target_sites IS 'Predicted and observed off-target cleavage sites';
COMMENT ON TABLE immune_response IS 'Innate immune pathway activation analysis';
COMMENT ON TABLE processing_jobs IS 'Asynchronous bioinformatics pipeline jobs';
COMMENT ON TABLE provenance IS 'W3C PROV-O compliant provenance tracking';
COMMENT ON TABLE audit_log IS 'System audit trail for security and compliance';

-- ============================================================================
-- VACUUM AND ANALYZE
-- ============================================================================

VACUUM ANALYZE;
