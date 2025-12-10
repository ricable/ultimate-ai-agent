# Ruvector Graph

[![Crates.io](https://img.shields.io/crates/v/ruvector-graph.svg)](https://crates.io/crates/ruvector-graph)
[![Documentation](https://docs.rs/ruvector-graph/badge.svg)](https://docs.rs/ruvector-graph)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.77%2B-orange.svg)](https://www.rust-lang.org)

**Distributed Neo4j-compatible hypergraph database with SIMD optimization and Cypher query support.**

`ruvector-graph` is a high-performance graph database engine that combines the power of hypergraphs with vector embeddings, enabling semantic graph queries and AI-powered graph analytics. Part of the [Ruvector](https://github.com/ruvnet/ruvector) ecosystem.

## Why Ruvector Graph?

- **Neo4j Compatible**: Cypher query language support for familiar graph queries
- **Hypergraph Support**: Model complex relationships with edges connecting multiple nodes
- **Vector-Enhanced**: Combine graph structure with semantic vector search
- **SIMD Optimized**: Hardware-accelerated operations via SimSIMD
- **Distributed Ready**: Built-in support for RAFT consensus and federation
- **WASM Compatible**: Run in browsers with WebAssembly support

## Features

### Core Capabilities

- **Hypergraph Model**: Edges can connect any number of nodes
- **Property Graph**: Rich properties on nodes and edges
- **Cypher Parser**: Full Cypher query language support
- **Vector Embeddings**: Semantic similarity on graph elements
- **HNSW Indexing**: Fast approximate nearest neighbor search
- **Label Indexes**: Roaring bitmap indexes for efficient label lookups

### Advanced Features

- **Distributed Mode**: RAFT consensus for distributed deployments
- **Federation**: Cross-cluster graph queries
- **Compression**: ZSTD and LZ4 support for storage optimization
- **Metrics**: Prometheus integration for monitoring
- **Temporal Graphs**: Time-varying graph support (planned)
- **Full-Text Search**: Text search on properties (planned)

## Installation

Add `ruvector-graph` to your `Cargo.toml`:

```toml
[dependencies]
ruvector-graph = "0.1.1"
```

### Feature Flags

```toml
[dependencies]
# Full feature set
ruvector-graph = { version = "0.1.1", features = ["full"] }

# Minimal WASM-compatible build
ruvector-graph = { version = "0.1.1", default-features = false, features = ["wasm"] }

# Distributed deployment
ruvector-graph = { version = "0.1.1", features = ["distributed"] }
```

Available features:
- `full` (default): Complete feature set with all optimizations
- `simd`: SIMD-optimized operations
- `storage`: Persistent storage with redb
- `async-runtime`: Tokio async support
- `compression`: ZSTD/LZ4 compression
- `distributed`: RAFT consensus support
- `federation`: Cross-cluster federation
- `wasm`: WebAssembly-compatible minimal build
- `metrics`: Prometheus monitoring

## Quick Start

### Create a Graph

```rust
use ruvector_graph::{Graph, Node, Edge, GraphConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new graph
    let config = GraphConfig::default();
    let graph = Graph::new(config)?;

    // Create nodes
    let alice = graph.create_node(Node {
        labels: vec!["Person".to_string()],
        properties: serde_json::json!({
            "name": "Alice",
            "age": 30
        }),
        ..Default::default()
    })?;

    let bob = graph.create_node(Node {
        labels: vec!["Person".to_string()],
        properties: serde_json::json!({
            "name": "Bob",
            "age": 25
        }),
        ..Default::default()
    })?;

    // Create relationship
    graph.create_edge(Edge {
        label: "KNOWS".to_string(),
        source: alice.id,
        target: bob.id,
        properties: serde_json::json!({
            "since": 2020
        }),
        ..Default::default()
    })?;

    Ok(())
}
```

### Cypher Queries

```rust
use ruvector_graph::{Graph, CypherExecutor};

// Execute Cypher query
let executor = CypherExecutor::new(&graph);
let results = executor.execute("
    MATCH (p:Person)-[:KNOWS]->(friend:Person)
    WHERE p.name = 'Alice'
    RETURN friend.name AS name, friend.age AS age
")?;

for row in results {
    println!("Friend: {} (age {})", row["name"], row["age"]);
}
```

### Vector-Enhanced Graph

```rust
use ruvector_graph::{Graph, VectorConfig};

// Enable vector embeddings on nodes
let config = GraphConfig {
    vector_config: Some(VectorConfig {
        dimensions: 384,
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    }),
    ..Default::default()
};

let graph = Graph::new(config)?;

// Create node with embedding
let node = graph.create_node(Node {
    labels: vec!["Document".to_string()],
    properties: serde_json::json!({"title": "Introduction to Graphs"}),
    embedding: Some(vec![0.1, 0.2, 0.3, /* ... 384 dims */]),
    ..Default::default()
})?;

// Semantic similarity search
let similar = graph.search_similar_nodes(
    vec![0.1, 0.2, 0.3, /* query vector */],
    10,  // top-k
    Some(vec!["Document".to_string()]),  // filter by labels
)?;
```

### Hyperedges

```rust
use ruvector_graph::{Graph, Hyperedge};

// Create a hyperedge connecting multiple nodes
let meeting = graph.create_hyperedge(Hyperedge {
    label: "PARTICIPATED_IN".to_string(),
    nodes: vec![alice.id, bob.id, charlie.id],
    properties: serde_json::json!({
        "event": "Team Meeting",
        "date": "2024-01-15"
    }),
    ..Default::default()
})?;
```

## API Overview

### Core Types

```rust
// Node in the graph
pub struct Node {
    pub id: NodeId,
    pub labels: Vec<String>,
    pub properties: serde_json::Value,
    pub embedding: Option<Vec<f32>>,
}

// Edge connecting two nodes
pub struct Edge {
    pub id: EdgeId,
    pub label: String,
    pub source: NodeId,
    pub target: NodeId,
    pub properties: serde_json::Value,
}

// Hyperedge connecting multiple nodes
pub struct Hyperedge {
    pub id: HyperedgeId,
    pub label: String,
    pub nodes: Vec<NodeId>,
    pub properties: serde_json::Value,
}
```

### Graph Operations

```rust
impl Graph {
    // Node operations
    pub fn create_node(&self, node: Node) -> Result<Node>;
    pub fn get_node(&self, id: &NodeId) -> Result<Option<Node>>;
    pub fn update_node(&self, node: Node) -> Result<Node>;
    pub fn delete_node(&self, id: &NodeId) -> Result<bool>;

    // Edge operations
    pub fn create_edge(&self, edge: Edge) -> Result<Edge>;
    pub fn get_edge(&self, id: &EdgeId) -> Result<Option<Edge>>;
    pub fn delete_edge(&self, id: &EdgeId) -> Result<bool>;

    // Traversal
    pub fn neighbors(&self, id: &NodeId, direction: Direction) -> Result<Vec<Node>>;
    pub fn traverse(&self, start: &NodeId, config: TraversalConfig) -> Result<Vec<Path>>;

    // Vector search
    pub fn search_similar_nodes(&self, query: Vec<f32>, k: usize, labels: Option<Vec<String>>) -> Result<Vec<Node>>;
}
```

## Performance

### Benchmarks (1M Nodes, 10M Edges)

```
Operation               Latency (p50)    Throughput
─────────────────────────────────────────────────────
Node lookup             ~0.1ms           100K ops/s
Edge traversal          ~0.5ms           50K ops/s
1-hop neighbors         ~1ms             20K ops/s
Cypher simple query     ~5ms             5K ops/s
Vector similarity       ~2ms             10K ops/s
```

## Related Crates

- **[ruvector-core](../ruvector-core/)** - Core vector database engine
- **[ruvector-graph-node](../ruvector-graph-node/)** - Node.js bindings
- **[ruvector-graph-wasm](../ruvector-graph-wasm/)** - WebAssembly bindings
- **[ruvector-raft](../ruvector-raft/)** - RAFT consensus for distributed mode
- **[ruvector-cluster](../ruvector-cluster/)** - Clustering and sharding

## Documentation

- **[Main README](../../README.md)** - Complete project overview
- **[API Documentation](https://docs.rs/ruvector-graph)** - Full API reference
- **[GitHub Repository](https://github.com/ruvnet/ruvector)** - Source code

## License

**MIT License** - see [LICENSE](../../LICENSE) for details.

---

<div align="center">

**Part of [Ruvector](https://github.com/ruvnet/ruvector) - Built by [rUv](https://ruv.io)**

[![Star on GitHub](https://img.shields.io/github/stars/ruvnet/ruvector?style=social)](https://github.com/ruvnet/ruvector)

[Documentation](https://docs.rs/ruvector-graph) | [Crates.io](https://crates.io/crates/ruvector-graph) | [GitHub](https://github.com/ruvnet/ruvector)

</div>
