# C4 Component Diagram - API Gateway & Processing Services

## API Gateway Components

This diagram shows the internal structure of the API Gateway container.

```mermaid
C4Component
    title Component Diagram - API Gateway (Axum/Rust)

    Container_Boundary(api_gateway, "API Gateway") {
        Component(router, "Router", "Axum Router", "Routes HTTP requests to handlers")
        Component(auth_middleware, "Auth Middleware", "Tower Layer", "JWT validation, permission checks")
        Component(rate_limiter, "Rate Limiter", "Tower Layer", "Redis-backed sliding window")
        Component(logger, "Request Logger", "Tower Layer", "Structured logging with tracing")

        Component(auth_handler, "Auth Handler", "Axum Handler", "Login, logout, token refresh")
        Component(experiment_handler, "Experiment Handler", "Axum Handler", "CRUD for experiments")
        Component(sample_handler, "Sample Handler", "Axum Handler", "Upload FASTQ files")
        Component(job_handler, "Job Handler", "Axum Handler", "Submit and monitor jobs")
        Component(result_handler, "Result Handler", "Axum Handler", "Query analysis results")

        Component(db_service, "Database Service", "SQLx", "PostgreSQL connection pool")
        Component(cache_service, "Cache Service", "redis-rs", "Redis client")
        Component(kafka_service, "Kafka Service", "rdkafka", "Message producer")
        Component(storage_service, "Storage Service", "rusoto_s3", "MinIO S3 client")
    }

    ContainerDb(postgres, "PostgreSQL", "Database")
    ContainerDb(redis, "Redis", "Cache")
    ContainerQueue(kafka, "Kafka", "Message Queue")
    ContainerDb(minio, "MinIO", "Object Storage")

    Rel(router, auth_middleware, "Applies", "Layer")
    Rel(router, rate_limiter, "Applies", "Layer")
    Rel(router, logger, "Applies", "Layer")

    Rel(auth_middleware, redis, "Verify session", "TCP")

    Rel(router, auth_handler, "Routes /auth/*")
    Rel(router, experiment_handler, "Routes /experiments/*")
    Rel(router, sample_handler, "Routes /samples/*")
    Rel(router, job_handler, "Routes /jobs/*")
    Rel(router, result_handler, "Routes /results/*")

    Rel(auth_handler, db_service, "Uses")
    Rel(experiment_handler, db_service, "Uses")
    Rel(sample_handler, storage_service, "Uses")
    Rel(job_handler, kafka_service, "Uses")
    Rel(result_handler, db_service, "Uses")

    Rel(db_service, postgres, "Queries", "TCP/5432")
    Rel(cache_service, redis, "Get/Set", "TCP/6379")
    Rel(kafka_service, kafka, "Publish", "TCP/9092")
    Rel(storage_service, minio, "Upload/Download", "HTTPS/9000")

    UpdateLayoutConfig($c4ShapeInRow="4", $c4BoundaryInRow="1")
```

## API Gateway Component Details

### 1. Middleware Stack

**Execution Order** (outermost to innermost):
1. **Request Logger**: Logs all incoming requests (method, path, headers)
2. **Rate Limiter**: Checks Redis for rate limit, returns 429 if exceeded
3. **Auth Middleware**: Validates JWT token, extracts user claims
4. **Router**: Routes request to appropriate handler

**Tower Middleware Example**:
```rust
let app = Router::new()
    .nest("/api/v1", api_routes)
    .layer(
        ServiceBuilder::new()
            .layer(TraceLayer::new_for_http())  // Request logger
            .layer(middleware::from_fn(rate_limit_middleware))
            .layer(middleware::from_fn(auth_middleware))
    );
```

### 2. Route Handlers

**Auth Handler** (`/api/v1/auth/*`)
- `POST /login`: Username/password → JWT token
- `POST /logout`: Invalidate session
- `POST /refresh`: Refresh token → New JWT
- `GET /verify`: Validate current token

**Experiment Handler** (`/api/v1/experiments/*`)
- `GET /`: List experiments (filtered by user)
- `POST /`: Create new experiment
- `GET /:id`: Get experiment details
- `PUT /:id`: Update experiment
- `DELETE /:id`: Delete experiment

**Sample Handler** (`/api/v1/samples/*`)
- `POST /upload`: Upload FASTQ files to MinIO
- `GET /:id`: Get sample metadata
- `DELETE /:id`: Delete sample

**Job Handler** (`/api/v1/jobs/*`)
- `POST /`: Submit analysis job to Kafka
- `GET /:id`: Get job status
- `GET /:id/logs`: Stream job logs
- `DELETE /:id`: Cancel running job

**Result Handler** (`/api/v1/results/*`)
- `GET /off-targets/:job_id`: Query off-target predictions
- `GET /diff-expr/:job_id`: Query differential expression results
- `GET /immune/:job_id`: Query immune response signatures
- `GET /export/:job_id`: Generate CSV/JSON export

### 3. Service Layer

**Database Service**
- **Connection Pool**: 100 max connections (SQLx)
- **Query Builder**: Type-safe queries with `sqlx::query!` macro
- **Transactions**: Automatic rollback on error
- **Migrations**: Versioned schema migrations

**Cache Service**
- **Client**: redis-rs with connection pooling
- **Patterns**:
  - Session storage (`session:{token}`)
  - Rate limiting (`rate_limit:{user}:{minute}`)
  - Job status (`job:{id}`)
- **TTL**: Configurable per key (default: 5 minutes)

**Kafka Service**
- **Producer**: rdkafka FutureProducer (async)
- **Serialization**: JSON with `serde_json`
- **Topics**: `jobs.alignment`, `jobs.off_target`, `jobs.diff_expr`, `jobs.immune`
- **Error Handling**: Retry 3 times with exponential backoff

**Storage Service**
- **Client**: rusoto_s3 (AWS SDK for Rust)
- **Buckets**: `crispr-cas13` (main), `crispr-cas13-backups`
- **Pre-signed URLs**: 1-hour TTL for downloads
- **Multipart Upload**: For files >5GB

---

## Processing Service Components (Alignment Service)

```mermaid
C4Component
    title Component Diagram - Alignment Service (Python/Bowtie2)

    Container_Boundary(alignment_service, "Alignment Service") {
        Component(kafka_consumer, "Kafka Consumer", "kafka-python", "Consumes alignment jobs")
        Component(file_downloader, "File Downloader", "boto3", "Downloads FASTQ from MinIO")
        Component(bowtie2_wrapper, "Bowtie2 Wrapper", "subprocess", "Runs Bowtie2 alignment")
        Component(samtools_wrapper, "Samtools Wrapper", "subprocess", "BAM conversion & indexing")
        Component(qc_module, "QC Module", "FastQC/MultiQC", "Quality control reports")
        Component(file_uploader, "File Uploader", "boto3", "Uploads BAM to MinIO")
        Component(db_updater, "DB Updater", "psycopg2", "Updates job status in PostgreSQL")
        Component(metrics_collector, "Metrics Collector", "prometheus_client", "Exposes metrics")
    }

    ContainerQueue(kafka, "Kafka", "Message Queue")
    ContainerDb(minio, "MinIO", "Object Storage")
    ContainerDb(postgres, "PostgreSQL", "Database")

    Rel(kafka, kafka_consumer, "Delivers jobs", "TCP/9092")
    Rel(kafka_consumer, file_downloader, "Triggers")
    Rel(file_downloader, minio, "Download FASTQ", "HTTPS/9000")

    Rel(kafka_consumer, bowtie2_wrapper, "Executes")
    Rel(bowtie2_wrapper, samtools_wrapper, "Pipes SAM")
    Rel(samtools_wrapper, qc_module, "Generates stats")

    Rel(qc_module, file_uploader, "Upload BAM + reports")
    Rel(file_uploader, minio, "Upload", "HTTPS/9000")

    Rel(kafka_consumer, db_updater, "Update status")
    Rel(db_updater, postgres, "INSERT/UPDATE", "TCP/5432")

    Rel(kafka_consumer, metrics_collector, "Record metrics")

    UpdateLayoutConfig($c4ShapeInRow="4", $c4BoundaryInRow="1")
```

## Alignment Service Workflow

1. **Consume Job**: Kafka consumer receives alignment job message
2. **Download FASTQ**: Download R1/R2 FASTQ files from MinIO to `/tmp`
3. **Run Bowtie2**: Execute Bowtie2 alignment (8 threads)
   ```bash
   bowtie2 -x ref_genome -1 R1.fastq.gz -2 R2.fastq.gz -p 8 --very-sensitive
   ```
4. **Convert to BAM**: Pipe SAM output to samtools
   ```bash
   samtools view -bS - | samtools sort -o sorted.bam
   samtools index sorted.bam
   ```
5. **Quality Control**: Run FastQC on BAM file
6. **Upload Results**: Upload BAM, BAM index, QC report to MinIO
7. **Update Database**: Mark job as completed in PostgreSQL
8. **Publish Event**: Send completion message to Kafka (`jobs.completed`)
9. **Cleanup**: Delete temporary files from `/tmp`

## Error Handling Strategy

**Transient Errors** (network failures, timeouts):
- **Retry**: 3 attempts with exponential backoff (1s, 2s, 4s)
- **Dead Letter Queue**: After 3 failures, move to `jobs.failed` topic

**Permanent Errors** (invalid input, out of memory):
- **Immediate Failure**: Mark job as failed in database
- **Notification**: Send alert to user via email/WebSocket
- **Logging**: Store detailed error message in PostgreSQL

**Resource Exhaustion**:
- **Graceful Degradation**: If disk space <10%, pause processing
- **Auto-Recovery**: Resume when space is freed

---

## Component Communication Patterns

### Synchronous (Request-Response)
- **API Gateway → PostgreSQL**: Direct database queries (SELECT, INSERT, UPDATE)
- **API Gateway → Redis**: Cache lookups (GET, SET)
- **API Gateway → MinIO**: File operations (PUT, GET)

### Asynchronous (Event-Driven)
- **API Gateway → Kafka**: Job submission (fire-and-forget)
- **Kafka → Workers**: Job consumption (at-least-once delivery)
- **Workers → Kafka**: Status updates (completion/failure events)

### Streaming
- **WebSocket Server → Clients**: Real-time job status updates
- **Prometheus → Grafana**: Continuous metrics streaming

---

**Diagram Version**: 1.0
**Last Updated**: 2025-10-12
**See Also**: [Sequence Diagrams](./sequence-diagrams.md)
