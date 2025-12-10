# CRISPR-Cas13 Bioinformatics Pipeline - System Requirements Specification

**Version:** 1.0.0
**Date:** 2025-10-12
**Phase:** SPARC Specification
**Status:** Draft

---

## 1. Introduction

### 1.1 Purpose
This document specifies the requirements for a bioinformatics pipeline system designed to process CRISPR-Cas13 RNA-targeting experiments in primate models, with focus on innate immune response analysis, off-target prediction, and experimental provenance tracking.

### 1.2 Scope
The system encompasses:
- **Data Ingestion**: FASTQ sequencing data, experimental metadata, sample annotations
- **Processing Pipeline**: Quality control, alignment, target identification, off-target prediction
- **Analysis**: Immune pathway gene expression analysis, statistical validation
- **Storage**: PostgreSQL for relational data, MongoDB for genomic data
- **API**: RESTful endpoints with real-time monitoring via WebSocket
- **Provenance**: Complete experimental lineage tracking per GA4GH standards

### 1.3 Definitions
- **CRISPR-Cas13**: RNA-targeting CRISPR system for transcript knockdown
- **Off-target**: Unintended RNA cleavage sites with sequence similarity to guide RNA
- **Innate Immune Response**: Type I/III interferon pathway activation
- **Provenance**: Complete audit trail of data processing and analysis steps
- **Guide RNA (gRNA)**: 22-30nt RNA sequence directing Cas13 to target
- **FASTQ**: Standard format for raw sequencing reads with quality scores
- **BAM/SAM**: Binary/text alignment format (GA4GH standard)

### 1.4 Bioinformatics Standards Compliance
- **GA4GH**: Global Alliance for Genomics and Health data standards
- **NCBI**: GenBank/RefSeq sequence formats
- **SAM/BAM**: Sequence Alignment/Map format specification v1.6
- **VCF**: Variant Call Format v4.3 for variant annotations
- **MIAME**: Minimum Information About a Microarray Experiment (adapted for RNA-seq)

---

## 2. Functional Requirements

### 2.1 Data Ingestion (FR-2.1)

#### FR-2.1.1: FASTQ Data Upload
**Priority:** Critical
**Description:** System shall accept paired-end FASTQ files (R1/R2) with Phred33 quality encoding

**Acceptance Criteria:**
- Support gzip compressed FASTQ (.fastq.gz)
- Validate FASTQ format per Illumina/ONT specifications
- Maximum file size: 50GB per file
- Checksum validation (MD5/SHA256)
- Automatic metadata extraction from FASTQ headers

**Validation Rules:**
```yaml
fastq_validation:
  - header_format: "@instrument:run:flowcell:lane:tile:x:y read:filter:control:index"
  - quality_encoding: "Phred33 (ASCII 33-126)"
  - min_read_length: 50
  - max_read_length: 300
  - allowed_bases: "ACGTN"
```

#### FR-2.1.2: Experimental Metadata Ingestion
**Priority:** Critical
**Description:** Accept structured experimental metadata in JSON/YAML format

**Required Fields:**
```yaml
experiment:
  experiment_id: string (UUID)
  title: string
  description: text
  species: enum [rhesus_macaque, cynomolgus_macaque, human_cell_line]
  tissue_type: enum [liver, kidney, brain, blood, cultured_cells]
  cas13_variant: enum [Cas13a, Cas13b, Cas13c, Cas13d]
  guide_rna:
    sequence: string (22-30nt)
    target_gene: string (HGNC symbol)
    target_transcript_id: string (Ensembl/RefSeq)
  delivery_method: enum [AAV, LNP, electroporation]
  timepoint_hours: integer
  replicate_number: integer
  experimenter: string
  institution: string
  date_performed: ISO8601
```

#### FR-2.1.3: Reference Genome Management
**Priority:** High
**Description:** Support multiple reference genomes with automatic indexing

**Supported References:**
- *Macaca mulatta* (rheMac10)
- *Macaca fascicularis* (macFas5)
- *Homo sapiens* (GRCh38/hg38)
- Custom transcriptome assemblies

### 2.2 Processing Pipeline (FR-2.2)

#### FR-2.2.1: Quality Control
**Priority:** Critical
**Description:** Perform automated QC on raw sequencing data

**QC Metrics:**
- Per-base sequence quality (FastQC)
- Adapter contamination detection
- Overrepresented sequences
- GC content distribution
- Duplication rate
- Read length distribution

**Acceptance Criteria:**
- Generate QC report in HTML/JSON format
- Flag samples with mean Phred < 28
- Automatic adapter trimming if >5% contamination

#### FR-2.2.2: Read Alignment
**Priority:** Critical
**Description:** Align reads to reference transcriptome using STAR or HISAT2

**Alignment Parameters:**
```yaml
alignment:
  aligner: STAR
  max_mismatches: 2
  min_intron_length: 20
  max_intron_length: 1000000
  output_format: BAM (coordinate-sorted)
  secondary_alignments: true
  supplementary_alignments: true
```

**Acceptance Criteria:**
- Generate BAM/BAI index files
- Calculate alignment statistics (mapped %, multimappers)
- Store alignment metrics in database

#### FR-2.2.3: Target Site Identification
**Priority:** Critical
**Description:** Identify Cas13 target sites and quantify knockdown efficiency

**Analysis:**
- Map guide RNA to transcriptome with exact/mismatch search
- Quantify transcript abundance (TPM/RPKM)
- Calculate knockdown ratio vs control samples
- Statistical significance testing (DESeq2/edgeR)

#### FR-2.2.4: Off-Target Prediction
**Priority:** High
**Description:** Predict potential off-target sites genome-wide

**Algorithm Requirements:**
- Seed-based search (12-15nt seed region)
- Mismatch tolerance: 0-4 mismatches
- Position-weighted scoring (PAM-proximal > distal)
- Sequence context analysis (secondary structure)

**Confidence Scoring:**
```python
off_target_score = (
    0.4 * seed_match_score +
    0.3 * full_sequence_similarity +
    0.2 * accessibility_score +
    0.1 * expression_level
)
```

**Output:**
- Ranked list of off-target sites (top 1000)
- Confidence score: 0.0-1.0
- Genomic coordinates (BED format)
- Overlapping gene annotations

#### FR-2.2.5: Immune Response Analysis
**Priority:** High
**Description:** Quantify innate immune pathway activation

**Target Pathways:**
- Type I interferon signaling (IRF3/7, STAT1/2, ISGs)
- Type III interferon signaling (IL28/29, IFNλR)
- RIG-I/MDA5 pathway (DDX58, IFIH1, MAVS)
- OAS/RNase L pathway (OAS1/2/3, RNASEL)
- PKR pathway (EIF2AK2)

**Analysis:**
- Gene set enrichment analysis (GSEA)
- Differential expression testing (adjusted p < 0.05)
- Fold-change calculation vs mock-treated controls
- Heatmap generation for immune gene panels

### 2.3 Job Management (FR-2.3)

#### FR-2.3.1: Job Submission
**Priority:** Critical
**Description:** Submit processing jobs with configurable parameters

**Job Configuration:**
```yaml
job:
  job_id: UUID
  job_type: enum [qc_only, alignment, full_pipeline, off_target_only]
  priority: enum [low, normal, high, urgent]
  input_files: array[string]
  reference_genome: string
  parameters: object
  notification_email: string (optional)
```

#### FR-2.3.2: Job Status Tracking
**Priority:** High
**Description:** Real-time job status monitoring

**Status States:**
- `queued`: Job submitted, awaiting resources
- `running`: Active processing
- `completed`: Successful completion
- `failed`: Error occurred (with error details)
- `cancelled`: User-initiated cancellation

**Acceptance Criteria:**
- Status updates every 30 seconds
- Progress percentage estimation
- ETA calculation based on file size
- Resource usage metrics (CPU, memory, disk I/O)

#### FR-2.3.3: Job Cancellation
**Priority:** Medium
**Description:** Allow users to cancel running jobs

**Requirements:**
- Graceful shutdown of running processes
- Cleanup of partial output files
- Preserve logs for debugging
- Immediate status update to `cancelled`

### 2.4 Results Retrieval (FR-2.4)

#### FR-2.4.1: Experiment Results API
**Priority:** High
**Description:** Retrieve processed results for completed experiments

**Response Structure:**
```json
{
  "experiment_id": "uuid",
  "status": "completed",
  "qc_metrics": {
    "total_reads": 45000000,
    "aligned_reads": 42300000,
    "alignment_rate": 0.94,
    "mean_quality": 36.5
  },
  "target_analysis": {
    "knockdown_efficiency": 0.87,
    "p_value": 1.2e-15,
    "tpm_control": 125.3,
    "tpm_treated": 16.2
  },
  "off_targets": [
    {
      "rank": 1,
      "gene": "GAPDH",
      "transcript_id": "ENST00000229239",
      "score": 0.82,
      "mismatches": 2,
      "position": "chr12:6646711-6646733"
    }
  ],
  "immune_response": {
    "interferon_score": 2.35,
    "activated_pathways": ["Type_I_IFN", "RIG-I"],
    "sig_genes": 47
  },
  "files": {
    "alignment_bam": "s3://results/exp123/aligned.bam",
    "qc_report": "s3://results/exp123/qc_report.html",
    "expression_matrix": "s3://results/exp123/expression.tsv"
  }
}
```

#### FR-2.4.2: File Downloads
**Priority:** High
**Description:** Secure download links for result files

**Supported Formats:**
- BAM/BAI (alignment files)
- VCF (variant calls)
- TSV (expression matrices)
- HTML (QC reports)
- BED (off-target coordinates)

**Security:**
- Pre-signed URLs with 1-hour expiration
- User authentication required
- Audit logging of all downloads

### 2.5 Real-Time Monitoring (FR-2.5)

#### FR-2.5.1: WebSocket Job Updates
**Priority:** Medium
**Description:** Real-time job progress via WebSocket connection

**Message Format:**
```json
{
  "job_id": "uuid",
  "timestamp": "2025-10-12T00:15:30Z",
  "event": "progress_update",
  "data": {
    "stage": "alignment",
    "progress": 0.65,
    "eta_seconds": 420,
    "message": "Processing chromosome 7..."
  }
}
```

**Events:**
- `job_started`
- `progress_update`
- `stage_complete`
- `job_completed`
- `job_failed`
- `warning_message`

---

## 3. Non-Functional Requirements

### 3.1 Performance (NFR-3.1)

#### NFR-3.1.1: Throughput
**Description:** System shall process multiple experiments concurrently

**Requirements:**
- 10 concurrent full-pipeline jobs
- 50 concurrent QC-only jobs
- Queue management with priority scheduling

#### NFR-3.1.2: Processing Time
**Description:** Maximum processing time per experiment

**Benchmarks:**
- QC analysis: <15 minutes (30M reads)
- Alignment: <2 hours (30M paired-end reads)
- Off-target prediction: <30 minutes
- Full pipeline: <3 hours (standard experiment)

#### NFR-3.1.3: API Response Time
**Description:** API latency requirements

**Targets:**
- Status queries: <200ms (p95)
- Results retrieval: <500ms (p95)
- Job submission: <1s (p95)
- WebSocket latency: <100ms

### 3.2 Scalability (NFR-3.2)

#### NFR-3.2.1: Data Volume
**Description:** Support for large-scale studies

**Capacity:**
- 10,000 experiments per year
- 500TB total storage (3-year retention)
- 1PB genomic reference data

#### NFR-3.2.2: Concurrent Users
**Description:** Multi-user support

**Requirements:**
- 100 concurrent API users
- 500 authenticated sessions
- Role-based access control (RBAC)

### 3.3 Reliability (NFR-3.3)

#### NFR-3.3.1: Uptime
**Description:** System availability SLA

**Target:** 99.5% uptime (43 hours downtime/year)

**Measures:**
- Automated health checks
- Database replication
- Graceful degradation
- Automatic job retry (transient failures)

#### NFR-3.3.2: Data Integrity
**Description:** Prevent data loss and corruption

**Requirements:**
- Daily automated backups
- Point-in-time recovery (7 days)
- Checksum validation for all files
- Transaction-based database operations

### 3.4 Security (NFR-3.4)

#### NFR-3.4.1: Authentication
**Description:** Secure user authentication

**Requirements:**
- OAuth2/OpenID Connect support
- Multi-factor authentication (MFA)
- API key authentication for programmatic access
- Session timeout: 24 hours

#### NFR-3.4.2: Authorization
**Description:** Fine-grained access control

**Roles:**
- `admin`: Full system access
- `researcher`: Submit jobs, view own results
- `analyst`: Read-only access to all data
- `guest`: View public datasets only

#### NFR-3.4.3: Data Encryption
**Description:** Protect sensitive data

**Requirements:**
- TLS 1.3 for all API traffic
- AES-256 encryption at rest
- Encrypted database connections
- Secure key management (AWS KMS/HashiCorp Vault)

#### NFR-3.4.4: Audit Logging
**Description:** Complete audit trail

**Log Events:**
- User authentication/authorization
- Job submissions/cancellations
- Data access/downloads
- Configuration changes
- System errors

**Retention:** 1 year

### 3.5 Compliance (NFR-3.5)

#### NFR-3.5.1: HIPAA Compliance
**Description:** If processing human samples

**Requirements:**
- PHI de-identification
- Business Associate Agreement (BAA)
- Encrypted storage
- Access audit logs

#### NFR-3.5.2: GDPR Compliance
**Description:** EU data protection

**Requirements:**
- Right to access personal data
- Right to deletion
- Data portability
- Consent management

#### NFR-3.5.3: NIH Data Sharing
**Description:** Compliance with NIH genomic data sharing policy

**Requirements:**
- Data submission to dbGaP (if required)
- Institutional Data Access Committee (DAC)
- Data Use Agreements (DUA)

---

## 4. Data Provenance Requirements

### 4.1 Provenance Tracking (PR-4.1)

#### PR-4.1.1: W3C PROV-O Compliance
**Description:** Implement provenance using W3C PROV ontology

**Entities:**
- Raw FASTQ files
- Reference genomes
- Aligned BAM files
- Expression matrices
- Off-target predictions

**Activities:**
- Quality control
- Read alignment
- Quantification
- Statistical analysis

**Agents:**
- Software tools (versions)
- User accounts
- Automated pipelines

#### PR-4.1.2: Provenance Graph
**Description:** Complete lineage tracking

**Requirements:**
- Directed acyclic graph (DAG) representation
- RDF serialization (Turtle/JSON-LD)
- Queryable via SPARQL
- Visualization in web interface

**Example Provenance Chain:**
```
raw_fastq.gz → [FastQC] → qc_report.html
             → [Trimmomatic] → trimmed_fastq.gz
             → [STAR] → aligned.bam
             → [featureCounts] → expression.tsv
             → [DESeq2] → differential_expression.csv
```

#### PR-4.1.3: Tool Version Tracking
**Description:** Record exact software versions

**Required Metadata:**
```yaml
tool_provenance:
  tool_name: STAR
  version: 2.7.10b
  parameters:
    genomeDir: /ref/rheMac10_star
    outSAMtype: BAM SortedByCoordinate
    quantMode: TranscriptomeSAM
  docker_image: quay.io/biocontainers/star:2.7.10b
  execution_time: 7234 seconds
  exit_code: 0
  stdout_log: s3://logs/job123/star_stdout.log
  stderr_log: s3://logs/job123/star_stderr.log
```

### 4.2 Reproducibility (PR-4.2)

#### PR-4.2.1: Containerized Workflows
**Description:** All tools packaged in Docker containers

**Requirements:**
- Dockerfile for each tool
- Fixed base image tags (no `latest`)
- Multi-stage builds for optimization
- Security scanning (Trivy/Snyk)

#### PR-4.2.2: Workflow Definition
**Description:** Declarative pipeline definition

**Format:** Nextflow DSL2 or Snakemake

**Example:**
```nextflow
process ALIGNMENT {
  container 'quay.io/biocontainers/star:2.7.10b'

  input:
    tuple val(sample_id), path(reads)
    path(genome_index)

  output:
    tuple val(sample_id), path("${sample_id}.bam")

  script:
    """
    STAR --genomeDir ${genome_index} \
         --readFilesIn ${reads} \
         --outSAMtype BAM SortedByCoordinate \
         --runThreadN ${task.cpus}
    """
}
```

---

## 5. System Constraints

### 5.1 Technical Constraints

#### C-5.1.1: Infrastructure
- **Platform:** Linux (Ubuntu 22.04 LTS)
- **Container Runtime:** Docker 24.0+
- **Orchestration:** Kubernetes 1.28+ (optional)
- **Database:** PostgreSQL 15+, MongoDB 7.0+
- **Object Storage:** S3-compatible (AWS/MinIO)

#### C-5.1.2: Programming Languages
- **Backend API:** Python 3.11+ (FastAPI)
- **Pipeline:** Nextflow 23.10+ or Snakemake 7.32+
- **Bioinformatics Tools:** STAR, HISAT2, featureCounts, DESeq2, BLAST+

#### C-5.1.3: Dependencies
- Minimum 32GB RAM for alignment jobs
- 8 CPU cores per concurrent job
- 10TB NVMe storage for temporary files
- 1Gbps network connectivity

### 5.2 Business Constraints

#### C-5.2.1: Budget
- Development: $150,000 (6 months)
- Infrastructure: $5,000/month (cloud costs)
- Maintenance: 1 FTE DevOps engineer

#### C-5.2.2: Timeline
- **Phase 1** (Months 1-2): Core pipeline + API
- **Phase 2** (Months 3-4): Off-target prediction + immune analysis
- **Phase 3** (Months 5-6): Provenance + production deployment

#### C-5.2.3: Team
- 1 Bioinformatics Lead
- 2 Software Engineers
- 1 DevOps Engineer
- 1 QA Analyst

### 5.3 Regulatory Constraints

#### C-5.3.1: Animal Research
- IACUC approval required
- Compliance with USDA Animal Welfare Act
- Detailed experimental protocols

#### C-5.3.2: Data Sharing
- NIH Genomic Data Sharing Policy
- Institutional Data Sharing Agreement
- dbGaP submission (if human-relevant)

---

## 6. Acceptance Criteria Summary

### Phase 1: Core Functionality
- [ ] FASTQ upload with validation (FR-2.1.1)
- [ ] QC analysis pipeline (FR-2.2.1)
- [ ] Read alignment (FR-2.2.2)
- [ ] Job submission API (FR-2.3.1)
- [ ] Status tracking (FR-2.3.2)
- [ ] PostgreSQL schema implemented
- [ ] MongoDB collections created
- [ ] API response time <500ms (NFR-3.1.3)

### Phase 2: Advanced Analysis
- [ ] Target site identification (FR-2.2.3)
- [ ] Off-target prediction (FR-2.2.4)
- [ ] Immune response analysis (FR-2.2.5)
- [ ] Results retrieval API (FR-2.4.1)
- [ ] WebSocket monitoring (FR-2.5.1)
- [ ] Processing time <3 hours (NFR-3.1.2)

### Phase 3: Production Readiness
- [ ] Provenance tracking (PR-4.1)
- [ ] Containerized workflows (PR-4.2.1)
- [ ] Authentication/authorization (NFR-3.4.1/3.4.2)
- [ ] Audit logging (NFR-3.4.4)
- [ ] 99.5% uptime (NFR-3.3.1)
- [ ] Automated backups (NFR-3.3.2)
- [ ] HIPAA compliance (if required) (NFR-3.5.1)

---

## 7. Validation & Testing

### 7.1 Unit Testing
- Minimum 90% code coverage
- Pytest for Python components
- Mock external services (S3, databases)

### 7.2 Integration Testing
- End-to-end pipeline tests with sample data
- API contract testing (Postman/Newman)
- Database migration testing

### 7.3 Performance Testing
- Load testing: 100 concurrent users (Locust)
- Stress testing: Maximum job throughput
- Latency benchmarks: API response times

### 7.4 Validation Data
- **Positive Control:** Known efficient guide RNA (e.g., KRAS-targeting)
- **Negative Control:** Non-targeting scrambled guide RNA
- **Reference Dataset:** Public CRISPR-Cas13 dataset (SRA)

---

## 8. Glossary

| Term | Definition |
|------|------------|
| **BAM** | Binary Alignment/Map format for aligned sequencing reads |
| **Cas13** | RNA-guided RNA-targeting CRISPR effector protein |
| **dbGaP** | Database of Genotypes and Phenotypes (NCBI) |
| **FASTQ** | Text-based format for storing nucleotide sequences with quality scores |
| **GA4GH** | Global Alliance for Genomics and Health |
| **gRNA** | Guide RNA that directs Cas13 to target transcript |
| **IACUC** | Institutional Animal Care and Use Committee |
| **ISG** | Interferon-Stimulated Gene |
| **NGS** | Next-Generation Sequencing |
| **PROV-O** | W3C Provenance Ontology |
| **RPKM** | Reads Per Kilobase per Million mapped reads |
| **TPM** | Transcripts Per Million (normalized expression metric) |

---

## 9. References

1. Abudayyeh, O. O., et al. (2017). "C2c2 is a single-component programmable RNA-guided RNA-targeting CRISPR effector." *Science*, 353(6299).
2. GA4GH Data Working Group. (2023). "GA4GH Data Standards." https://www.ga4gh.org/
3. NCBI. (2023). "Sequence Read Archive Format Guide." https://www.ncbi.nlm.nih.gov/sra/
4. W3C. (2013). "PROV-O: The PROV Ontology." https://www.w3.org/TR/prov-o/
5. NIH. (2023). "Genomic Data Sharing Policy." https://sharing.nih.gov/genomic-data-sharing-policy

---

**Document Control:**
- **Author:** SPARC Specification Agent
- **Reviewers:** Bioinformatics Lead, Software Architect
- **Next Review Date:** 2025-11-12
- **Version History:**
  - v1.0.0 (2025-10-12): Initial specification
