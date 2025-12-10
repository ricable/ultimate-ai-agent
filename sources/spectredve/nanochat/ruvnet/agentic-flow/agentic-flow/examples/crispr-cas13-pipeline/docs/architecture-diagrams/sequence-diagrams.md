# Sequence Diagrams - Key Workflows

## 1. User Authentication Flow

```mermaid
sequenceDiagram
    actor User
    participant Web as Web App
    participant API as API Gateway
    participant Redis as Redis
    participant DB as PostgreSQL

    User->>Web: Enter email & password
    Web->>API: POST /api/v1/auth/login
    activate API

    API->>DB: SELECT user WHERE email = ?
    activate DB
    DB-->>API: User record
    deactivate DB

    API->>API: Verify password (bcrypt)

    alt Password Valid
        API->>API: Generate JWT token
        API->>Redis: SET session:{token}
        activate Redis
        Redis-->>API: OK
        deactivate Redis

        API->>DB: INSERT INTO audit_log (login)
        API-->>Web: 200 OK {token, user}
        deactivate API
        Web->>Web: Store token in localStorage
        Web-->>User: Redirect to dashboard
    else Password Invalid
        API-->>Web: 401 Unauthorized
        deactivate API
        Web-->>User: Show error message
    end
```

---

## 2. Experiment Creation & Sample Upload

```mermaid
sequenceDiagram
    actor User
    participant Web as Web App
    participant API as API Gateway
    participant DB as PostgreSQL
    participant MinIO as MinIO S3

    User->>Web: Fill experiment form
    User->>Web: Upload FASTQ files (R1, R2)

    Web->>API: POST /api/v1/experiments
    activate API
    API->>API: Validate JWT token
    API->>DB: INSERT INTO experiments
    activate DB
    DB-->>API: experiment_id
    deactivate DB

    loop For each sample
        Web->>API: POST /api/v1/samples/upload
        activate API
        API->>MinIO: PUT raw-data/{exp_id}/{sample}_R1.fastq.gz
        activate MinIO
        MinIO-->>API: Upload complete
        deactivate MinIO

        API->>DB: INSERT INTO samples
        activate DB
        DB-->>API: sample_id
        deactivate DB
        API-->>Web: 201 Created {sample_id}
        deactivate API
    end

    API-->>Web: 201 Created {experiment_id}
    deactivate API
    Web-->>User: Show success message
```

---

## 3. Job Submission & Processing (Alignment)

```mermaid
sequenceDiagram
    actor User
    participant Web as Web App
    participant API as API Gateway
    participant DB as PostgreSQL
    participant Kafka as Apache Kafka
    participant Worker as Alignment Worker
    participant MinIO as MinIO S3

    User->>Web: Click "Start Analysis"
    Web->>API: POST /api/v1/jobs {type: "alignment"}
    activate API

    API->>DB: INSERT INTO analysis_jobs (status: "queued")
    activate DB
    DB-->>API: job_id
    deactivate DB

    API->>Kafka: Publish to "jobs.alignment"
    activate Kafka
    Kafka-->>API: ACK
    deactivate Kafka

    API-->>Web: 202 Accepted {job_id}
    deactivate API

    Kafka->>Worker: Deliver job message
    activate Worker

    Worker->>DB: UPDATE jobs SET status="running"
    activate DB
    DB-->>Worker: OK
    deactivate DB

    Worker->>MinIO: Download FASTQ files
    activate MinIO
    MinIO-->>Worker: R1.fastq.gz, R2.fastq.gz
    deactivate MinIO

    Worker->>Worker: Run Bowtie2 alignment (8 threads)
    Note over Worker: ~30-60 minutes for 50M reads

    Worker->>Worker: Convert SAM to BAM (samtools)
    Worker->>Worker: Index BAM (samtools)

    Worker->>MinIO: Upload aligned/{sample}.bam
    activate MinIO
    MinIO-->>Worker: Upload complete
    deactivate MinIO

    Worker->>DB: UPDATE jobs SET status="completed"
    activate DB
    DB-->>Worker: OK
    deactivate DB

    Worker->>Kafka: Publish to "jobs.completed"
    activate Kafka
    Kafka-->>Worker: ACK
    deactivate Kafka

    deactivate Worker

    Kafka->>API: Notify via WebSocket
    API->>Web: WebSocket: {job_id, status: "completed"}
    Web-->>User: Show completion notification
```

---

## 4. Real-Time Status Updates (WebSocket)

```mermaid
sequenceDiagram
    actor User
    participant Web as Web App
    participant WS as WebSocket Server
    participant Kafka as Apache Kafka
    participant Worker as Processing Worker

    User->>Web: Open Job Monitor page
    Web->>WS: Connect WebSocket
    activate WS
    WS-->>Web: Connection established

    Web->>WS: Subscribe to job_id=12345
    WS->>WS: Add to subscribers list

    Worker->>Kafka: Publish progress {job_id: 12345, progress: 45%}
    activate Kafka
    Kafka->>WS: Deliver progress update
    deactivate Kafka

    WS->>WS: Filter subscribers (job_id=12345)
    WS->>Web: Send progress update
    Web-->>User: Update progress bar (45%)

    Worker->>Kafka: Publish progress {job_id: 12345, progress: 80%}
    activate Kafka
    Kafka->>WS: Deliver progress update
    deactivate Kafka

    WS->>Web: Send progress update
    Web-->>User: Update progress bar (80%)

    Worker->>Kafka: Publish completion {job_id: 12345, status: "completed"}
    activate Kafka
    Kafka->>WS: Deliver completion event
    deactivate Kafka

    WS->>Web: Send completion event
    Web-->>User: Show "Analysis Complete" notification

    User->>Web: Close page
    Web->>WS: Disconnect WebSocket
    WS->>WS: Remove from subscribers
    deactivate WS
```

---

## 5. Off-Target Prediction Workflow

```mermaid
sequenceDiagram
    participant API as API Gateway
    participant Kafka as Apache Kafka
    participant Worker as Off-Target Worker
    participant MinIO as MinIO S3
    participant Mongo as MongoDB
    participant DB as PostgreSQL

    API->>Kafka: Publish to "jobs.off_target"
    activate Kafka
    Kafka->>Worker: Deliver job message
    deactivate Kafka
    activate Worker

    Worker->>DB: UPDATE jobs SET status="running"
    activate DB
    DB-->>Worker: OK
    deactivate DB

    Worker->>MinIO: Download aligned BAM file
    activate MinIO
    MinIO-->>Worker: sample.bam
    deactivate MinIO

    Worker->>Mongo: Query genome annotations
    activate Mongo
    Mongo-->>Worker: Gene positions, sequences
    deactivate Mongo

    Worker->>Worker: Extract guide RNA from experiment
    Worker->>Worker: Scan BAM for mismatched alignments

    loop For each potential off-target
        Worker->>Worker: Calculate CFD score (ML model)
        Worker->>Worker: Calculate MIT score
        Worker->>DB: INSERT INTO off_targets
        activate DB
        DB-->>Worker: OK
        deactivate DB
    end

    Worker->>DB: UPDATE jobs SET status="completed"
    activate DB
    DB-->>Worker: OK
    deactivate DB

    Worker->>Kafka: Publish to "jobs.completed"
    activate Kafka
    Kafka-->>Worker: ACK
    deactivate Kafka

    deactivate Worker
```

---

## 6. Differential Expression Analysis

```mermaid
sequenceDiagram
    participant API as API Gateway
    participant Kafka as Apache Kafka
    participant Worker as DESeq2 Worker (R)
    participant MinIO as MinIO S3
    participant DB as PostgreSQL

    API->>Kafka: Publish to "jobs.diff_expr"
    activate Kafka
    Kafka->>Worker: Deliver job message
    deactivate Kafka
    activate Worker

    Worker->>DB: UPDATE jobs SET status="running"
    activate DB
    DB-->>Worker: OK
    deactivate DB

    loop For each sample
        Worker->>MinIO: Download BAM file
        activate MinIO
        MinIO-->>Worker: sample.bam
        deactivate MinIO
    end

    Worker->>Worker: Run featureCounts (count reads per gene)
    Note over Worker: ~10-20 minutes for 50M reads

    Worker->>Worker: Create DESeqDataSet
    Worker->>Worker: Run DESeq2 normalization
    Worker->>Worker: Test for differential expression

    loop For each gene
        Worker->>DB: INSERT INTO differential_expression
        activate DB
        DB-->>Worker: OK
        deactivate DB
    end

    Worker->>Worker: Generate plots (MA, volcano, heatmap)
    Worker->>MinIO: Upload plots PDF
    activate MinIO
    MinIO-->>Worker: Upload complete
    deactivate MinIO

    Worker->>DB: UPDATE jobs SET status="completed"
    activate DB
    DB-->>Worker: OK
    deactivate DB

    Worker->>Kafka: Publish to "jobs.completed"
    activate Kafka
    Kafka-->>Worker: ACK
    deactivate Kafka

    deactivate Worker
```

---

## 7. Result Visualization & Export

```mermaid
sequenceDiagram
    actor User
    participant Web as Web App
    participant API as API Gateway
    participant DB as PostgreSQL
    participant MinIO as MinIO S3

    User->>Web: Navigate to Experiment Results
    Web->>API: GET /api/v1/results/off-targets/{job_id}
    activate API

    API->>DB: SELECT * FROM off_targets WHERE job_id=?
    activate DB
    DB-->>API: List of off-targets
    deactivate DB

    API-->>Web: 200 OK {off_targets: [...]}
    deactivate API

    Web->>Web: Render interactive table
    Web-->>User: Display off-targets

    User->>Web: Click "Export to CSV"
    Web->>API: GET /api/v1/results/export/{job_id}?format=csv
    activate API

    API->>DB: SELECT * FROM off_targets, differential_expression
    activate DB
    DB-->>API: Full results
    deactivate DB

    API->>API: Generate CSV file
    API->>MinIO: Upload results/{job_id}/export.csv
    activate MinIO
    MinIO-->>API: Pre-signed URL
    deactivate MinIO

    API-->>Web: 200 OK {download_url}
    deactivate API

    Web->>MinIO: GET download_url
    activate MinIO
    MinIO-->>Web: export.csv
    deactivate MinIO

    Web-->>User: Download CSV file
```

---

## 8. Error Handling & Retry

```mermaid
sequenceDiagram
    participant Kafka as Apache Kafka
    participant Worker as Processing Worker
    participant DB as PostgreSQL
    participant DLQ as Dead Letter Queue

    Kafka->>Worker: Deliver job message (attempt 1)
    activate Worker

    Worker->>Worker: Process job
    Worker->>Worker: ERROR: Network timeout

    Worker->>DB: UPDATE jobs SET status="retrying"
    activate DB
    DB-->>Worker: OK
    deactivate DB

    Worker->>Kafka: NACK (negative acknowledgment)
    deactivate Worker

    Note over Kafka: Wait 5 seconds

    Kafka->>Worker: Deliver job message (attempt 2)
    activate Worker

    Worker->>Worker: Process job
    Worker->>Worker: ERROR: Out of memory

    Worker->>DB: UPDATE jobs SET status="retrying"
    activate DB
    DB-->>Worker: OK
    deactivate DB

    Worker->>Kafka: NACK
    deactivate Worker

    Note over Kafka: Wait 10 seconds

    Kafka->>Worker: Deliver job message (attempt 3)
    activate Worker

    Worker->>Worker: Process job
    Worker->>Worker: ERROR: Invalid BAM file

    Worker->>DB: UPDATE jobs SET status="failed", error_message="Invalid BAM file"
    activate DB
    DB-->>Worker: OK
    deactivate DB

    Worker->>DLQ: Send to "jobs.failed" topic
    activate DLQ
    DLQ-->>Worker: ACK
    deactivate DLQ

    Worker->>Kafka: ACK (final acknowledgment)
    deactivate Worker

    Note over DLQ: Admin reviews failed jobs
```

---

## 9. Monitoring & Alerting

```mermaid
sequenceDiagram
    participant Service as Any Service
    participant Prom as Prometheus
    participant Alert as Alertmanager
    participant Slack as Slack
    participant PD as PagerDuty

    loop Every 15 seconds
        Prom->>Service: Scrape /metrics endpoint
        activate Service
        Service-->>Prom: Metrics (CPU, memory, request rate)
        deactivate Service
    end

    Prom->>Prom: Evaluate alerting rules

    alt CPU > 90% for 5 minutes
        Prom->>Alert: Fire alert: "HighCPU"
        activate Alert

        Alert->>Alert: Check routing rules

        alt Severity: Warning
            Alert->>Slack: Send message to #alerts
            activate Slack
            Slack-->>Alert: OK
            deactivate Slack
        else Severity: Critical
            Alert->>PD: Trigger incident
            activate PD
            PD-->>Alert: Incident created
            deactivate PD

            PD->>PD: Notify on-call engineer
        end

        deactivate Alert
    end
```

---

**Document Version**: 1.0
**Last Updated**: 2025-10-12
**See Also**: [Architecture Overview](../ARCHITECTURE.md)
