# 🧠 ReasoningBank

[![Crates.io](https://img.shields.io/crates/v/reasoningbank?style=flat-square)](https://crates.io/crates/reasoningbank)
[![Documentation](https://img.shields.io/docsrs/reasoningbank?style=flat-square)](https://docs.rs/reasoningbank)
[![License](https://img.shields.io/crates/l/reasoningbank?style=flat-square)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/ruvnet/agentic-flow/ci.yml?style=flat-square)](https://github.com/ruvnet/agentic-flow/actions)
[![Rust Version](https://img.shields.io/badge/rust-1.70%2B-blue?style=flat-square)](https://www.rust-lang.org)

> **Adaptive self-learning reasoning system with embedded storage and QUIC neural bus**

ReasoningBank is a high-performance, self-learning AI reasoning system that learns from experience and improves over time. It combines pattern recognition, adaptive strategy optimization, and distributed neural communication to build smarter AI agents.

---

## 🎯 What is ReasoningBank?

**For Non-Technical Users:**

Think of ReasoningBank as a "memory and learning system" for AI. Just like humans learn from experience and get better at tasks over time, ReasoningBank helps AI systems:

- **Remember** what worked and what didn't in past tasks
- **Learn** from successful strategies and avoid repeating mistakes
- **Adapt** by automatically choosing the best approach for new challenges
- **Share** knowledge efficiently across distributed AI agents

It's like giving your AI a notebook, a teacher, and a communication system all in one.

**For Technical Users:**

ReasoningBank is a Rust-based adaptive learning framework that provides:

- **Pattern Storage**: Embedded SQLite with WAL mode, connection pooling, and optimized queries
- **Adaptive Learning**: Cosine similarity-based pattern matching, strategy scoring, and confidence-weighted recommendations
- **Neural Bus**: QUIC-based communication protocol with Ed25519-signed intent-capped actions, stream multiplexing, and gossip protocols
- **Async-First Design**: Actor pattern for thread-safe async operations with message-passing architecture
- **MCP Integration**: Full Model Context Protocol server with tools and resources for AI agent coordination

---

## ✨ Features

### 🎓 Core Learning Capabilities

- **Pattern Recognition**: Stores and retrieves task patterns with vector embeddings
- **Similarity Matching**: Cosine and Euclidean distance algorithms for finding relevant experiences
- **Strategy Optimization**: Automatically identifies and recommends successful strategies
- **Success Tracking**: Confidence-weighted success rates and outcome analysis
- **Adaptive Recommendations**: Context-aware strategy suggestions based on task characteristics

### 💾 High-Performance Storage

- **Embedded SQLite**: Zero-configuration, file-based or in-memory storage
- **Connection Pooling**: Efficient multi-threaded access with Parking Lot synchronization
- **Write-Ahead Logging (WAL)**: Concurrent reads during writes for maximum throughput
- **Optimized Queries**: Indexed searches, prepared statements, and efficient similarity calculations
- **Migration System**: Automatic schema versioning and upgrades

### 🚀 QUIC Neural Bus

- **0-RTT Connections**: Sub-millisecond connection establishment
- **Stream Multiplexing**: Parallel data streams without head-of-line blocking
- **Intent-Capped Actions**: Ed25519-signed authorization with spend caps and scope restrictions
- **Reasoning Streams**: Separate channels for tokens, traces, rubrics, and verification
- **Gossip Protocol**: Decentralized knowledge sharing across agent networks
- **Snapshot Streaming**: Efficient large-state transfer with chunking

### 🔧 Developer Experience

- **Async/Await Support**: Full Tokio integration with spawn_blocking for sync operations
- **Actor Pattern**: Thread-safe message passing eliminates Send/Sync issues
- **Type-Safe API**: Strongly-typed interfaces with comprehensive error handling
- **MCP Tools**: Ready-to-use server with pattern storage, retrieval, analysis, and optimization tools
- **Extensive Tests**: 60+ unit and integration tests with 90%+ coverage
- **Performance Benchmarks**: Criterion-based benchmarking for storage, learning, and neural bus operations

---

## 📦 Installation

Add ReasoningBank to your `Cargo.toml`:

```toml
[dependencies]
reasoningbank-core = "0.1.0"          # Core reasoning engine
reasoningbank-storage = "0.1.0"       # SQLite storage layer
reasoningbank-learning = "0.1.0"      # Adaptive learning system
reasoningbank-network = "0.1.0"       # QUIC neural bus
reasoningbank-mcp = "0.1.0"          # MCP server integration
```

### Requirements

- **Rust**: 1.70 or later
- **OS**: Linux, macOS, Windows (SQLite bundled for all platforms)
- **Features**: `tokio` async runtime, `rusqlite` with bundled SQLite

---

## 🚀 Quick Start

### Basic Pattern Storage and Learning

```rust
use reasoningbank_core::{Pattern, TaskOutcome, ReasoningEngine, EngineConfig};
use reasoningbank_learning::{AdaptiveLearner, LearningConfig};
use reasoningbank_storage::{SqliteStorage, StorageConfig};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    // Initialize storage
    let storage_config = StorageConfig {
        database_path: PathBuf::from("reasoningbank.db"),
        max_connections: 10,
        enable_wal: true,
        cache_size_kb: 8192,
    };
    let storage = SqliteStorage::new(storage_config)?;

    // Create reasoning engine and learner
    let engine = ReasoningEngine::new(EngineConfig::default());
    let mut learner = AdaptiveLearner::new(
        engine,
        storage,
        LearningConfig::default(),
    );

    // Store a successful task pattern
    let pattern = Pattern::new(
        "Implement REST API endpoint".to_string(),
        "backend_development".to_string(),
        "test_driven_development".to_string(),
    ).with_outcome(TaskOutcome::success(0.95));

    let insight = learner.learn_from_task(&pattern)?;
    println!("Learning insight: {:?}", insight);

    // Get recommendations for a new task
    let applied = learner.apply_learning(
        "Build a new API endpoint",
        "backend_development",
    )?;

    println!("Recommended strategy: {}", applied.recommended_strategy);
    println!("Confidence: {:.2}%", applied.confidence * 100.0);

    Ok(())
}
```

---

## 📚 Documentation

### Architecture Overview

ReasoningBank uses a modular architecture with six primary crates:

- **reasoningbank-core**: Core reasoning engine with pattern matching
- **reasoningbank-storage**: SQLite storage layer with connection pooling
- **reasoningbank-learning**: Adaptive learning system with similarity matching
- **reasoningbank-network**: QUIC neural bus with intent verification
- **reasoningbank-mcp**: MCP server with tools and resources
- **reasoningbank-wasm**: WASM bindings (placeholder for future web support)

### Performance

Based on Criterion benchmarks:

- **Storage**: ~200-300 µs per pattern storage, ~50-100 µs per lookup
- **Learning**: ~2.6 ms per learning operation, ~4.7 ms per recommendation
- **Neural Bus**: ~5-10 µs per frame encode/decode (1KB payload)

---

## 🧪 Testing

```bash
# Run all tests
cargo test --all-features

# Run benchmarks
cargo bench

# Generate benchmark HTML reports
cargo bench -- --save-baseline baseline-v1
```

**Test Coverage**: 60+ tests across all crates with 90%+ coverage

---

## 🤝 Contributing

Contributions are welcome! Please:

- Write tests for new features
- Run `cargo fmt` and `cargo clippy` before committing
- Update documentation for API changes
- Add benchmarks for performance-critical code

---

## 📄 License

Dual-licensed under MIT or Apache-2.0 at your option.

---

## 🙏 Acknowledgments

Built with [Rust](https://www.rust-lang.org/), [SQLite](https://www.sqlite.org/), [Quinn](https://github.com/quinn-rs/quinn), [Tokio](https://tokio.rs/), and [Criterion](https://github.com/bheisler/criterion.rs).

---

<p align="center">
  <sub>Built with ❤️ by the ReasoningBank contributors</sub>
</p>
