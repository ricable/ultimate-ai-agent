# QUIC Architecture Diagrams for AgentDB

This document contains comprehensive architecture diagrams for the QUIC synchronization system using Mermaid notation.

---

## 1. System Overview (C4 Context)

```mermaid
graph TB
    subgraph "AgentDB QUIC Network"
        Agent1[AI Agent 1<br/>Client Node]
        Agent2[AI Agent 2<br/>Client Node]
        Agent3[AI Agent 3<br/>Client Node]
        SyncServer[QUIC Sync Server<br/>Hub Node]

        Agent1 <-->|QUIC/TLS 1.3| SyncServer
        Agent2 <-->|QUIC/TLS 1.3| SyncServer
        Agent3 <-->|QUIC/TLS 1.3| SyncServer
    end

    subgraph "Data Layer"
        DB[(AgentDB<br/>SQLite + Vector)]
        ChangeLog[(Sync Changelog)]
    end

    SyncServer --> DB
    SyncServer --> ChangeLog
    Agent1 -.->|Local DB| DB
    Agent2 -.->|Local DB| DB
    Agent3 -.->|Local DB| DB

    AuthServer[Auth Server<br/>JWT Issuer]
    Monitoring[Monitoring<br/>Prometheus/Grafana]

    Agent1 -.->|Authenticate| AuthServer
    Agent2 -.->|Authenticate| AuthServer
    Agent3 -.->|Authenticate| AuthServer
    SyncServer -->|Metrics| Monitoring

    style SyncServer fill:#4CAF50
    style Agent1 fill:#2196F3
    style Agent2 fill:#2196F3
    style Agent3 fill:#2196F3
    style DB fill:#FF9800
    style ChangeLog fill:#FF9800
```

---

## 2. QUIC Connection Architecture

```mermaid
sequenceDiagram
    participant Client as QUIC Client
    participant Server as QUIC Server
    participant Auth as Auth Service
    participant DB as AgentDB

    Note over Client,Server: Connection Establishment
    Client->>Server: QUIC ClientHello + mTLS Cert
    Server->>Server: Validate Certificate
    Server->>Client: QUIC ServerHello + mTLS Cert
    Client->>Client: Validate Certificate

    Note over Client,Server: TLS 1.3 Handshake Complete

    Client->>Server: Stream 0: JWT Token
    Server->>Auth: Validate JWT
    Auth-->>Server: Token Valid + Claims
    Server->>Client: Auth Success + Capabilities

    Note over Client,Server: Multiplexed Streams Open

    par Stream 1: Episodes
        Client->>Server: Episode Sync (CREATE)
        Server->>DB: Insert Episode
        DB-->>Server: Success
        Server->>Client: ACK
    and Stream 2: Skills
        Client->>Server: Skill Sync (UPDATE)
        Server->>DB: Merge Skill (CRDT)
        DB-->>Server: Merged State
        Server->>Client: Merged Skill
    and Stream 3: Edges
        Client->>Server: Edge Sync (CREATE)
        Server->>DB: Insert Edge
        DB-->>Server: Success
        Server->>Client: ACK
    end

    Note over Client,Server: Periodic Reconciliation
    Client->>Server: Stream 4: Full Recon Request
    Server->>DB: Query Full State
    DB-->>Server: State Dump
    Server->>Client: Full State + Merkle Root
    Client->>Client: Verify & Apply
    Client->>Server: Recon Complete
```

---

## 3. Data Flow Architecture

```mermaid
flowchart TD
    Start([Local Update]) --> CheckType{Data Type?}

    CheckType -->|Episode| VectorClock[Increment Vector Clock]
    CheckType -->|Skill| CRDT[Update CRDT State]
    CheckType -->|Edge| OT[Apply OT Operation]

    VectorClock --> LogChange[Write to Changelog]
    CRDT --> LogChange
    OT --> LogChange

    LogChange --> QueueSync{Auto-Sync<br/>Enabled?}

    QueueSync -->|Yes| SendSync[Send via QUIC Stream]
    QueueSync -->|No| WaitTrigger[Wait for Manual Sync]

    WaitTrigger --> SendSync

    SendSync --> ServerReceive[Server Receives]

    ServerReceive --> Auth{Authorized?}
    Auth -->|No| Reject[Reject: 403 Forbidden]
    Auth -->|Yes| Validate{Valid?}

    Validate -->|No| Reject
    Validate -->|Yes| ConflictCheck{Conflict?}

    ConflictCheck -->|No| Apply[Apply to Server DB]
    ConflictCheck -->|Yes| Resolve[Resolve Conflict]

    Resolve --> Strategy{Strategy?}
    Strategy -->|Vector Clock| VCResolve[Use Causal Ordering]
    Strategy -->|CRDT| CRDTMerge[Merge State]
    Strategy -->|OT| OTTransform[Transform Operations]

    VCResolve --> Apply
    CRDTMerge --> Apply
    OTTransform --> Apply

    Apply --> Broadcast[Broadcast to Other Nodes]
    Broadcast --> ACK[Send ACK to Client]

    ACK --> End([Sync Complete])
    Reject --> End

    style Start fill:#4CAF50
    style End fill:#4CAF50
    style Resolve fill:#FFC107
    style Reject fill:#F44336
```

---

## 4. Conflict Resolution Decision Tree

```mermaid
flowchart TD
    Conflict([Conflict Detected]) --> DataType{Data Type?}

    DataType -->|Episode| VCCompare[Compare Vector Clocks]
    DataType -->|Skill| CRDTCheck[Check CRDT Type]
    DataType -->|Edge| EdgeConflict[Analyze Edge Conflict]

    VCCompare --> VCResult{Comparison<br/>Result?}
    VCResult -->|Before| UseLocal[Use Local Version]
    VCResult -->|After| UseRemote[Use Remote Version]
    VCResult -->|Concurrent| Timestamp{Compare<br/>Timestamps}

    Timestamp -->|Local > Remote| UseLocal
    Timestamp -->|Remote > Local| UseRemote
    Timestamp -->|Equal| NodeID{Compare<br/>Node IDs}

    NodeID -->|Local > Remote| UseLocal
    NodeID -->|Remote > Local| UseRemote

    CRDTCheck --> CRDTType{CRDT Type?}
    CRDTType -->|G-Counter| Sum[Sum Counters]
    CRDTType -->|LWW-Register| LWW[Last Write Wins]
    CRDTType -->|OR-Set| Union[Union Sets]

    Sum --> Merge([Merge Result])
    LWW --> Merge
    Union --> Merge

    EdgeConflict --> EdgeType{Conflict Type?}
    EdgeType -->|Same Metrics| Average[Weighted Average]
    EdgeType -->|Deleted Node| Cascade[Cascade Delete]
    EdgeType -->|Different Evidence| Combine[Combine Evidence]

    Average --> Merge
    Cascade --> Merge
    Combine --> Merge

    UseLocal --> Apply([Apply Resolution])
    UseRemote --> Apply
    Merge --> Apply

    Apply --> Log[Log Resolution]
    Log --> Notify[Notify Clients]
    Notify --> Done([Conflict Resolved])

    style Conflict fill:#F44336
    style Done fill:#4CAF50
    style Merge fill:#2196F3
```

---

## 5. Hub-and-Spoke Topology

```mermaid
graph TB
    subgraph "Hub-and-Spoke Topology"
        Hub[Central QUIC Server<br/>Authoritative State]

        subgraph "Spoke Nodes"
            A1[Agent 1]
            A2[Agent 2]
            A3[Agent 3]
            A4[Agent 4]
            A5[Agent 5]
        end

        A1 <-->|Sync| Hub
        A2 <-->|Sync| Hub
        A3 <-->|Sync| Hub
        A4 <-->|Sync| Hub
        A5 <-->|Sync| Hub
    end

    Hub -.->|Backup| Replica[Standby Replica]

    Note1[✓ Simple conflict resolution<br/>✓ Single source of truth<br/>✓ Centralized monitoring<br/>✗ Single point of failure]

    style Hub fill:#4CAF50
    style A1 fill:#2196F3
    style A2 fill:#2196F3
    style A3 fill:#2196F3
    style A4 fill:#2196F3
    style A5 fill:#2196F3
    style Replica fill:#FF9800
```

---

## 6. Mesh Topology

```mermaid
graph TB
    subgraph "Mesh Topology"
        A[Agent A]
        B[Agent B]
        C[Agent C]
        D[Agent D]
        E[Agent E]

        A <--> B
        A <--> C
        A <--> D
        B <--> C
        B <--> E
        C <--> D
        C <--> E
        D <--> E
    end

    Note2[✓ No single point of failure<br/>✓ Works offline/edge<br/>✓ Peer-to-peer sync<br/>✗ Complex conflict resolution<br/>✗ Higher network overhead]

    style A fill:#2196F3
    style B fill:#2196F3
    style C fill:#2196F3
    style D fill:#2196F3
    style E fill:#2196F3
```

---

## 7. Hierarchical Topology

```mermaid
graph TB
    subgraph "Hierarchical Topology"
        Root[Root Hub<br/>Global Authority]

        subgraph "Regional Hubs"
            R1[Region 1 Hub<br/>US West]
            R2[Region 2 Hub<br/>US East]
            R3[Region 3 Hub<br/>Europe]
        end

        Root <--> R1
        Root <--> R2
        Root <--> R3

        subgraph "Region 1 Nodes"
            A1[Agent 1-1]
            A2[Agent 1-2]
        end

        subgraph "Region 2 Nodes"
            B1[Agent 2-1]
            B2[Agent 2-2]
        end

        subgraph "Region 3 Nodes"
            C1[Agent 3-1]
            C2[Agent 3-2]
        end

        R1 <--> A1
        R1 <--> A2
        R2 <--> B1
        R2 <--> B2
        R3 <--> C1
        R3 <--> C2
    end

    Note3[✓ Scales to 1000+ nodes<br/>✓ Geographic distribution<br/>✓ Reduced latency<br/>✗ Complex setup<br/>✗ Multi-level conflict resolution]

    style Root fill:#4CAF50
    style R1 fill:#FF9800
    style R2 fill:#FF9800
    style R3 fill:#FF9800
```

---

## 8. CRDT State Machine

```mermaid
stateDiagram-v2
    [*] --> Idle

    Idle --> LocalUpdate : Local Write
    LocalUpdate --> CRDTMerge : Receive Remote Update

    state CRDTMerge {
        [*] --> CheckType
        CheckType --> GCounter : G-Counter
        CheckType --> LWWRegister : LWW-Register
        CheckType --> ORSet : OR-Set

        GCounter --> TakeMax : Take Max per Node
        LWWRegister --> CompareTimestamp : Compare Timestamp
        ORSet --> UnionSets : Union with Tombstones

        TakeMax --> MergedState
        CompareTimestamp --> MergedState
        UnionSets --> MergedState

        MergedState --> [*]
    }

    CRDTMerge --> Synced : Merge Complete
    Synced --> Idle : Ready for Next Update

    LocalUpdate --> Syncing : Send to Server
    Syncing --> WaitACK : Waiting for ACK
    WaitACK --> Synced : ACK Received
    WaitACK --> Retry : Timeout
    Retry --> Syncing : Exponential Backoff

    note right of CRDTMerge
        Conflict-Free Merge
        No Manual Resolution Needed
    end note
```

---

## 9. Vector Clock Comparison

```mermaid
flowchart LR
    subgraph "Vector Clock Comparison"
        VC1[Local: A:5, B:3, C:2]
        VC2[Remote: A:4, B:4, C:2]

        VC1 --> Compare{Compare}
        VC2 --> Compare

        Compare --> CheckA{A: 5 vs 4}
        CheckA -->|Local > Remote| AGreater[A: Local Greater]

        Compare --> CheckB{B: 3 vs 4}
        CheckB -->|Local < Remote| BGreater[B: Remote Greater]

        Compare --> CheckC{C: 2 vs 2}
        CheckC -->|Equal| CEqual[C: Equal]

        AGreater --> Result
        BGreater --> Result
        CEqual --> Result

        Result{Final?}
        Result -->|A>R && R>A| Concurrent([Concurrent Updates<br/>Use Timestamp])
        Result -->|All A>R| After([Local After Remote<br/>Use Local])
        Result -->|All R>A| Before([Local Before Remote<br/>Use Remote])
    end

    style Concurrent fill:#FFC107
    style After fill:#4CAF50
    style Before fill:#2196F3
```

---

## 10. Incremental Sync Flow

```mermaid
sequenceDiagram
    participant Client
    participant Changelog
    participant Server
    participant DB

    Note over Client,Server: Incremental Sync (Every 5s)

    Client->>Server: Sync Request<br/>{lastTimestamp: T1, vectorClock: VC1}
    Server->>Changelog: Query changes since T1
    Changelog-->>Server: Delta records (100 records)

    Server->>Server: Filter by Vector Clock<br/>(remove already-seen updates)
    Server->>Client: Delta Batch (50 new records)

    Client->>Client: Apply Changes<br/>(Resolve Conflicts)
    Client->>Client: Update Local DB
    Client->>Client: Increment Vector Clock

    Client->>Server: ACK<br/>{applied: 50, failed: 0}
    Server->>Changelog: Mark records as synced to Client

    alt More Deltas Available
        Server->>Client: Next Batch (50 records)
        Client->>Client: Apply Changes
        Client->>Server: ACK
    else No More Deltas
        Server->>Client: Sync Complete
    end

    Note over Client,Server: Next sync in 5 seconds
```

---

## 11. Full Reconciliation Flow

```mermaid
sequenceDiagram
    participant Client
    participant Server
    participant DB

    Note over Client,Server: Full Reconciliation (Weekly)

    Client->>Server: Reconciliation Request<br/>{dataTypes: [episodes, skills, edges]}
    Server->>DB: Compute State Summary<br/>(Merkle Tree)
    DB-->>Server: {merkleRoot, count, vectorClock}

    Client->>Client: Compute Local State Summary
    Client->>Server: Local Summary<br/>{merkleRoot, count, vectorClock}

    Server->>Server: Compare Merkle Roots

    alt Merkle Roots Match
        Server->>Client: Already in Sync ✓
    else Merkle Roots Differ
        Server->>Client: Request Missing Records
        Client->>Server: Records not on server (delta)

        Server->>Client: Records not on client (delta)
        Client->>Client: Apply Server Records

        Server->>DB: Apply Client Records

        par Conflict Resolution
            Client->>Client: Resolve Local Conflicts
        and
            Server->>DB: Resolve Server Conflicts
        end

        Client->>Client: Recompute Merkle Root
        Server->>DB: Recompute Merkle Root

        Client->>Server: New Merkle Root
        Server->>Client: New Merkle Root

        alt Merkle Roots Now Match
            Server->>Client: Reconciliation Complete ✓
        else Merkle Roots Still Differ
            Server->>Client: Retry Reconciliation
        end
    end

    Note over Client,Server: Next reconciliation in 7 days
```

---

## 12. Authentication & Authorization Flow

```mermaid
sequenceDiagram
    participant Client
    participant Server
    participant Auth
    participant CA

    Note over Client,Server: mTLS Authentication

    Client->>Server: QUIC ClientHello + Client Cert
    Server->>CA: Verify Client Certificate
    CA-->>Server: Certificate Valid ✓

    Server->>Client: QUIC ServerHello + Server Cert
    Client->>CA: Verify Server Certificate
    CA-->>Client: Certificate Valid ✓

    Note over Client,Server: TLS Handshake Complete

    Client->>Auth: Request JWT<br/>{nodeId, certificateThumbprint}
    Auth->>Auth: Validate Identity
    Auth->>Auth: Generate JWT with Claims<br/>{roles, scopes, exp}
    Auth-->>Client: JWT Token

    Client->>Server: Stream 0: JWT Token
    Server->>Auth: Validate JWT Signature
    Auth-->>Server: Valid + Claims

    Server->>Server: Check Token Expiration
    Server->>Server: Parse Scopes

    Server->>Client: Auth Success<br/>{capabilities: [scopes]}

    Note over Client,Server: Authorized Sync Operations

    Client->>Server: Episode Sync (write)
    Server->>Server: Check Scope: episodes:write
    alt Has Scope
        Server->>Client: ACK (200 OK)
    else Missing Scope
        Server->>Client: Forbidden (403)
    end

    Note over Client,Server: Token Refresh (before expiration)

    Client->>Auth: Refresh Token Request
    Auth-->>Client: New JWT Token
```

---

## 13. Monitoring & Observability

```mermaid
graph TB
    subgraph "AgentDB QUIC Network"
        C1[Client 1]
        C2[Client 2]
        Server[Sync Server]
        C1 --> Server
        C2 --> Server
    end

    subgraph "Metrics Collection"
        Server -->|Prometheus| Metrics[Metrics Exporter]
        C1 -->|Metrics| Metrics
        C2 -->|Metrics| Metrics
    end

    subgraph "Monitoring Stack"
        Metrics --> Prometheus[Prometheus Server]
        Prometheus --> Grafana[Grafana Dashboard]
        Prometheus --> AlertManager[Alert Manager]
    end

    subgraph "Dashboards"
        Grafana --> D1[Connection Metrics]
        Grafana --> D2[Sync Performance]
        Grafana --> D3[Conflict Resolution]
        Grafana --> D4[Resource Usage]
    end

    subgraph "Alerts"
        AlertManager --> A1[High Latency Alert]
        AlertManager --> A2[Conflict Storm Alert]
        AlertManager --> A3[Connection Failure Alert]
    end

    A1 -.->|Notify| Ops[Operations Team]
    A2 -.->|Notify| Ops
    A3 -.->|Notify| Ops

    style Server fill:#4CAF50
    style Prometheus fill:#FF9800
    style Grafana fill:#2196F3
    style AlertManager fill:#F44336
```

---

## 14. Deployment Architecture

```mermaid
graph TB
    subgraph "Production Deployment"
        subgraph "Load Balancer"
            LB[NGINX/HAProxy<br/>UDP Load Balancer]
        end

        subgraph "QUIC Server Cluster"
            S1[Server Instance 1]
            S2[Server Instance 2]
            S3[Server Instance 3]
        end

        LB --> S1
        LB --> S2
        LB --> S3

        subgraph "Data Layer"
            Master[(PostgreSQL Master)]
            Replica1[(PostgreSQL Replica 1)]
            Replica2[(PostgreSQL Replica 2)]
            Redis[(Redis Cache)]
        end

        S1 --> Master
        S2 --> Master
        S3 --> Master

        Master --> Replica1
        Master --> Replica2

        S1 --> Redis
        S2 --> Redis
        S3 --> Redis

        subgraph "Auth Service"
            AuthLB[Auth LB]
            Auth1[Auth Server 1]
            Auth2[Auth Server 2]
        end

        AuthLB --> Auth1
        AuthLB --> Auth2

        S1 -.->|Validate JWT| AuthLB
        S2 -.->|Validate JWT| AuthLB
        S3 -.->|Validate JWT| AuthLB

        subgraph "Monitoring"
            Prom[Prometheus]
            Graf[Grafana]
        end

        S1 -->|Metrics| Prom
        S2 -->|Metrics| Prom
        S3 -->|Metrics| Prom
        Prom --> Graf
    end

    subgraph "Clients"
        Client1[AI Agent 1]
        Client2[AI Agent 2]
        ClientN[AI Agent N]
    end

    Client1 --> LB
    Client2 --> LB
    ClientN --> LB

    style LB fill:#4CAF50
    style Master fill:#FF9800
    style Redis fill:#F44336
```

---

## 15. Security Layers

```mermaid
graph TB
    subgraph "Security Layers"
        subgraph "Layer 1: Network"
            L1[Firewall<br/>Only UDP 443 allowed]
        end

        subgraph "Layer 2: Transport"
            L2[mTLS + QUIC<br/>Mutual Certificate Auth<br/>TLS 1.3 Encryption]
        end

        subgraph "Layer 3: Application"
            L3[JWT Authorization<br/>Scope-based Access Control<br/>Role-based Permissions]
        end

        subgraph "Layer 4: Data"
            L4[Field Encryption<br/>Metadata Encryption<br/>At-rest Encryption]
        end

        subgraph "Layer 5: Audit"
            L5[Audit Logging<br/>Intrusion Detection<br/>Anomaly Detection]
        end
    end

    Client[Client Node] --> L1
    L1 --> L2
    L2 --> L3
    L3 --> L4
    L4 --> DB[(AgentDB)]

    L1 -.->|Log| L5
    L2 -.->|Log| L5
    L3 -.->|Log| L5
    L4 -.->|Log| L5

    L5 --> SIEM[SIEM System]

    style L1 fill:#F44336
    style L2 fill:#FF9800
    style L3 fill:#FFC107
    style L4 fill:#4CAF50
    style L5 fill:#2196F3
```

---

## 16. Performance Optimization Pipeline

```mermaid
flowchart LR
    Input[Sync Data] --> Check1{Size > 10KB?}
    Check1 -->|Yes| Compress[Compress with zstd]
    Check1 -->|No| Batch

    Compress --> Batch{Batch Size<br/>< 1000 records?}
    Batch -->|No| Split[Split into Batches]
    Batch -->|Yes| Stream{Stream<br/>Available?}

    Split --> Stream

    Stream -->|Yes| Multiplex[Multiplex on QUIC Stream]
    Stream -->|No| Queue[Queue for Next Stream]

    Multiplex --> Priority{Priority?}
    Priority -->|High| S1[Stream 1: Episodes]
    Priority -->|Medium| S2[Stream 2: Skills]
    Priority -->|Low| S3[Stream 3: Edges]

    S1 --> Send[Send via QUIC]
    S2 --> Send
    S3 --> Send

    Send --> Receive[Server Receives]
    Receive --> Decompress[Decompress if Compressed]
    Decompress --> Apply[Apply to DB]
    Apply --> ACK[Send ACK]

    Queue --> Wait[Wait for Stream]
    Wait --> Stream

    style Input fill:#4CAF50
    style Compress fill:#FF9800
    style Multiplex fill:#2196F3
    style ACK fill:#4CAF50
```

---

## Summary

These diagrams provide comprehensive visualization of:

1. **System Architecture**: Context, containers, and components (C4 model)
2. **Protocol Design**: QUIC connection flow and stream multiplexing
3. **Conflict Resolution**: Decision trees and CRDT state machines
4. **Topology Options**: Hub-and-spoke, mesh, and hierarchical
5. **Sync Strategies**: Incremental and full reconciliation flows
6. **Security**: Multi-layer authentication and authorization
7. **Operations**: Monitoring, deployment, and performance optimization

These diagrams complement the main QUIC Architecture document and provide visual reference for implementation teams.
