# C4 Context Diagram - CRISPR-Cas13 Pipeline

## System Context

This diagram shows the CRISPR-Cas13 Analysis Pipeline in the context of external users and systems.

```mermaid
C4Context
    title System Context Diagram - CRISPR-Cas13 Analysis Pipeline

    Person(researcher, "Research Scientist", "Designs experiments, analyzes results")
    Person(bioinformatician, "Bioinformatician", "Configures pipelines, validates results")
    Person(admin, "System Administrator", "Monitors health, manages resources")

    System(crispr_pipeline, "CRISPR-Cas13 Pipeline", "Analyzes off-target effects and immune responses from CRISPR-Cas13 experiments")

    System_Ext(ncbi, "NCBI Database", "Reference genomes and annotations")
    System_Ext(ensembl, "Ensembl Genome Browser", "Gene annotations and orthologs")
    System_Ext(immunesigdb, "ImmuneSigDB", "Immune response gene signatures")
    System_Ext(hpc, "HPC Cluster", "High-performance computing resources")

    Rel(researcher, crispr_pipeline, "Submits samples, views results", "HTTPS")
    Rel(bioinformatician, crispr_pipeline, "Configures pipelines, exports data", "HTTPS/CLI")
    Rel(admin, crispr_pipeline, "Monitors system, manages users", "HTTPS/SSH")

    Rel(crispr_pipeline, ncbi, "Downloads reference genomes", "FTP/HTTPS")
    Rel(crispr_pipeline, ensembl, "Queries gene annotations", "REST API")
    Rel(crispr_pipeline, immunesigdb, "Retrieves signature data", "REST API")
    Rel(crispr_pipeline, hpc, "Offloads compute-intensive jobs", "SSH/Slurm")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

## External Systems Integration

### 1. NCBI Database
- **Purpose**: Download reference genomes (e.g., Macaca mulatta rheMac10)
- **Protocol**: FTP, HTTPS
- **Frequency**: Monthly (genome updates)
- **Authentication**: Public access, API key for higher rate limits

### 2. Ensembl Genome Browser
- **Purpose**: Gene annotations, transcript variants, orthologs
- **Protocol**: REST API (`https://rest.ensembl.org/`)
- **Frequency**: Weekly updates
- **Rate Limit**: 15 requests/sec

### 3. ImmuneSigDB
- **Purpose**: Curated immune response gene signatures
- **Protocol**: REST API, bulk download
- **Frequency**: Quarterly updates
- **Data Format**: GMT, JSON

### 4. HPC Cluster (Optional)
- **Purpose**: Offload large-scale alignment jobs
- **Protocol**: SSH, Slurm job scheduler
- **Integration**: Job submission via `sbatch`, result retrieval via `scp`
- **Use Case**: Experiments with >50 samples

## User Roles & Responsibilities

| Role | Responsibilities | Access Level |
|------|-----------------|--------------|
| **Research Scientist** | - Design experiments<br/>- Submit samples<br/>- Interpret results | Read/Write own experiments |
| **Bioinformatician** | - Configure analysis pipelines<br/>- Validate results<br/>- Export raw data | Read all, Write pipelines |
| **System Administrator** | - Monitor system health<br/>- Manage users and permissions<br/>- Perform backups | Full system access |

## Security Boundaries

- **External Firewall**: All external traffic goes through firewall (HTTPS only)
- **API Gateway**: Authentication layer (OAuth2/JWT)
- **Internal Network**: Service-to-service communication uses mTLS
- **Data Encryption**: All data encrypted at rest (AES-256) and in transit (TLS 1.3)

## Data Flow Summary

1. **Ingress**: Users upload FASTQ files via web UI → API Gateway → MinIO S3
2. **Processing**: Jobs orchestrated via Kafka → Processing workers → Results stored in PostgreSQL/MongoDB
3. **Egress**: Users download results via pre-signed S3 URLs (time-limited)
4. **External**: Pipeline fetches reference data from NCBI/Ensembl on-demand (cached in MongoDB)

---

**Diagram Version**: 1.0
**Last Updated**: 2025-10-12
**Next Level**: [C4 Container Diagram](./c4-container.md)
