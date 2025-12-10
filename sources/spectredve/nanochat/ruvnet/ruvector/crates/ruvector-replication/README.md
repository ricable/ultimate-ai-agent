# Ruvector Replication

[![Crates.io](https://img.shields.io/crates/v/ruvector-replication.svg)](https://crates.io/crates/ruvector-replication)
[![Documentation](https://docs.rs/ruvector-replication/badge.svg)](https://docs.rs/ruvector-replication)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.77%2B-orange.svg)](https://www.rust-lang.org)

**Data replication and synchronization for Ruvector distributed deployments.**

`ruvector-replication` provides vector data replication across nodes with configurable consistency levels, conflict resolution, and synchronization strategies. Part of the [Ruvector](https://github.com/ruvnet/ruvector) ecosystem.

## Why Ruvector Replication?

- **High Availability**: Replicate data across multiple nodes
- **Configurable Consistency**: Tune consistency vs availability
- **Async Replication**: Non-blocking replication for performance
- **Conflict Resolution**: Automatic conflict handling strategies
- **Incremental Sync**: Efficient delta synchronization

## Features

### Core Capabilities

- **Multi-Master Replication**: Write to any node
- **Replica Sets**: Configurable replication factor
- **Change Streams**: Real-time replication events
- **Checkpointing**: Track replication progress
- **Recovery**: Automatic replica recovery

### Advanced Features

- **Quorum Writes**: Configurable write acknowledgment
- **Read Replicas**: Scale read throughput
- **Conflict Resolution**: Last-write-wins, vector clocks, CRDTs
- **Bandwidth Throttling**: Control replication bandwidth
- **Compression**: Reduce network transfer size

## Installation

Add `ruvector-replication` to your `Cargo.toml`:

```toml
[dependencies]
ruvector-replication = "0.1.1"
```

## Quick Start

### Setup Replication

```rust
use ruvector_replication::{Replicator, ReplicationConfig, ConsistencyLevel};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure replication
    let config = ReplicationConfig {
        replication_factor: 3,
        consistency_level: ConsistencyLevel::Quorum,
        sync_interval: Duration::from_millis(100),
        batch_size: 1000,
        compression: true,
        ..Default::default()
    };

    // Create replicator
    let replicator = Replicator::new(config).await?;

    // Start replication
    replicator.start().await?;

    Ok(())
}
```

### Write with Replication

```rust
use ruvector_replication::{Replicator, WriteOptions};

// Write with quorum consistency
let options = WriteOptions {
    consistency: ConsistencyLevel::Quorum,
    timeout: Duration::from_secs(5),
};

replicator.write(vector_entry, options).await?;

// Write with eventual consistency (faster)
let options = WriteOptions {
    consistency: ConsistencyLevel::One,
    ..Default::default()
};

replicator.write(vector_entry, options).await?;
```

### Monitor Replication

```rust
// Get replication lag
let lag = replicator.lag().await?;
println!("Replication lag: {:?}", lag);

// Get replica status
for replica in replicator.replicas().await? {
    println!("{}: {} (lag: {}ms)",
        replica.id,
        replica.status,
        replica.lag_ms
    );
}

// Subscribe to replication events
let mut stream = replicator.events().await?;
while let Some(event) = stream.next().await {
    match event {
        ReplicationEvent::Synced { node_id, entries } => {
            println!("Synced {} entries to {}", entries, node_id);
        }
        ReplicationEvent::Conflict { key, resolution } => {
            println!("Conflict on {}: {:?}", key, resolution);
        }
        _ => {}
    }
}
```

## API Overview

### Core Types

```rust
// Replication configuration
pub struct ReplicationConfig {
    pub replication_factor: usize,
    pub consistency_level: ConsistencyLevel,
    pub sync_interval: Duration,
    pub batch_size: usize,
    pub compression: bool,
    pub conflict_resolution: ConflictResolution,
}

// Consistency levels
pub enum ConsistencyLevel {
    One,      // Write to one replica
    Quorum,   // Write to majority
    All,      // Write to all replicas
}

// Conflict resolution strategies
pub enum ConflictResolution {
    LastWriteWins,
    VectorClock,
    Custom(Box<dyn ConflictResolver>),
}

// Replica information
pub struct ReplicaInfo {
    pub id: NodeId,
    pub status: ReplicaStatus,
    pub lag_ms: u64,
    pub last_sync: DateTime<Utc>,
}
```

### Replicator Operations

```rust
impl Replicator {
    pub async fn new(config: ReplicationConfig) -> Result<Self>;
    pub async fn start(&self) -> Result<()>;
    pub async fn stop(&self) -> Result<()>;

    // Write operations
    pub async fn write(&self, entry: VectorEntry, options: WriteOptions) -> Result<()>;
    pub async fn write_batch(&self, entries: Vec<VectorEntry>, options: WriteOptions) -> Result<()>;

    // Monitoring
    pub async fn lag(&self) -> Result<Duration>;
    pub async fn replicas(&self) -> Result<Vec<ReplicaInfo>>;
    pub async fn events(&self) -> Result<impl Stream<Item = ReplicationEvent>>;

    // Management
    pub async fn add_replica(&self, node_id: NodeId) -> Result<()>;
    pub async fn remove_replica(&self, node_id: NodeId) -> Result<()>;
    pub async fn force_sync(&self, node_id: NodeId) -> Result<()>;
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Replication Flow                       │
│                                                         │
│  Client                                                 │
│    │                                                    │
│    ▼                                                    │
│  ┌──────────┐     Quorum Write    ┌──────────┐         │
│  │ Primary  │────────────────────▶│ Replica 1│         │
│  │          │                     │          │         │
│  │ Vectors  │────────────────────▶│ Vectors  │         │
│  └──────────┘                     └──────────┘         │
│       │                                                 │
│       │        Async Replication                        │
│       └──────────────────────────▶┌──────────┐         │
│                                   │ Replica 2│         │
│                                   │          │         │
│                                   │ Vectors  │         │
│                                   └──────────┘         │
└─────────────────────────────────────────────────────────┘
```

## Related Crates

- **[ruvector-core](../ruvector-core/)** - Core vector database engine
- **[ruvector-cluster](../ruvector-cluster/)** - Clustering and sharding
- **[ruvector-raft](../ruvector-raft/)** - Raft consensus

## Documentation

- **[Main README](../../README.md)** - Complete project overview
- **[API Documentation](https://docs.rs/ruvector-replication)** - Full API reference
- **[GitHub Repository](https://github.com/ruvnet/ruvector)** - Source code

## License

**MIT License** - see [LICENSE](../../LICENSE) for details.

---

<div align="center">

**Part of [Ruvector](https://github.com/ruvnet/ruvector) - Built by [rUv](https://ruv.io)**

[![Star on GitHub](https://img.shields.io/github/stars/ruvnet/ruvector?style=social)](https://github.com/ruvnet/ruvector)

[Documentation](https://docs.rs/ruvector-replication) | [Crates.io](https://crates.io/crates/ruvector-replication) | [GitHub](https://github.com/ruvnet/ruvector)

</div>
