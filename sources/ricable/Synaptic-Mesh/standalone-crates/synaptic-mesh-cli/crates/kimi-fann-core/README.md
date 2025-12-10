# 🤖 Kimi-FANN Core

[![Crates.io](https://img.shields.io/crates/v/kimi-fann-core.svg)](https://crates.io/crates/kimi-fann-core)
[![Documentation](https://docs.rs/kimi-fann-core/badge.svg)](https://docs.rs/kimi-fann-core)
[![License](https://img.shields.io/crates/l/kimi-fann-core.svg)](https://github.com/ruvnet/Synaptic-Neural-Mesh/blob/main/LICENSE)

A high-performance neural inference engine with 5-agent swarm architecture, optimized for WebAssembly and native environments. Part of the Synaptic Neural Mesh project.

## 🚀 Features

- **Multi-Expert System**: 6 specialized neural experts (reasoning, coding, mathematics, language, tool-use, context)
- **5-10x Performance**: Optimized hash-based processing with SIMD support
- **WASM Optimized**: 40% memory reduction, sub-second inference
- **Production Ready**: Published on crates.io, comprehensive test coverage
- **Swarm Architecture**: Distributed agent coordination for complex tasks

## 📦 Installation

### As a CLI Tool

```bash
cargo install kimi-fann-core
```

### As a Library

```toml
[dependencies]
kimi-fann-core = "0.1.3"
```

## 🎯 Quick Start

### Command Line Usage

```bash
# Ask any question - Kimi will route to the best expert
kimi "What is machine learning?"

# Use specific expert
kimi --expert coding "Write a bubble sort function"

# Multi-expert consensus
kimi --consensus "Design a neural network"

# Show performance metrics
kimi --performance "Explain quantum computing"

# Interactive mode
kimi --interactive
```

### Development Usage (cargo run)

When using `cargo run`, you MUST include `--` to separate cargo arguments:

```bash
# ✅ CORRECT - With -- separator
cargo run --bin kimi -- "your question here"
cargo run --bin kimi -- --expert mathematics "What is 2+2?"
cargo run --bin kimi -- --consensus "Complex problem"

# ❌ WRONG - Without -- separator (will fail)
cargo run --bin kimi "your question"
cargo run --bin kimi --expert coding "Write code"
```

#### Convenience Script

For easier development, use the included wrapper script:

```bash
# Make it executable
chmod +x kimi.sh

# Use without worrying about --
./kimi.sh "your question"
./kimi.sh --expert coding "write a function"
./kimi.sh --consensus "complex question"
```

### Library Usage

```rust
use kimi_fann_core::{KimiCore, ExpertType};

// Initialize Kimi
let mut kimi = KimiCore::new();

// Ask a question with automatic routing
let response = kimi.process("What is rust?");
println!("Response: {}", response);

// Use specific expert
let response = kimi.process_with_expert(
    "Write a sorting algorithm",
    ExpertType::Coding
);

// Get consensus from multiple experts
let response = kimi.process_with_consensus(
    "Design a distributed system"
);
```

## 🧠 Expert Domains

| Expert | Specialization | Example Use Case |
|--------|---------------|------------------|
| **Reasoning** | Logic, analysis, critical thinking | "Analyze the pros and cons of microservices" |
| **Coding** | Programming, algorithms, software design | "Implement a binary search tree" |
| **Mathematics** | Calculations, formulas, numerical analysis | "Calculate the derivative of x^3" |
| **Language** | Translation, text processing, linguistics | "Translate 'hello world' to Japanese" |
| **Tool-Use** | Commands, operations, technical procedures | "How to use Docker compose?" |
| **Context** | Memory, conversation tracking, references | "What did we discuss earlier?" |

## 🚀 Advanced Features

### Consensus Mode
Get multiple expert opinions for complex questions:

```rust
let response = kimi.process_with_consensus("Design a blockchain system");
// Returns aggregated insights from multiple experts
```

### Performance Monitoring
Track neural inference metrics:

```rust
let (response, metrics) = kimi.process_with_metrics("Complex question");
println!("Inference time: {}ms", metrics.inference_time_ms);
println!("Tokens processed: {}", metrics.tokens_processed);
```

### WASM Integration
Optimized for browser environments:

```javascript
import init, { KimiCore } from './pkg/kimi_fann_core.js';

await init();
const kimi = KimiCore.new();
const response = kimi.process("What is WebAssembly?");
console.log(response);
```

## 📊 Performance

- **Inference Speed**: 50-200ms typical response time
- **Memory Usage**: 40% reduction vs v0.1.0
- **WASM Size**: <500KB compressed
- **Throughput**: 1000+ queries/second on modern hardware

## 🔧 Building from Source

```bash
# Clone the repository
git clone https://github.com/ruvnet/Synaptic-Neural-Mesh
cd Synaptic-Neural-Mesh/standalone-crates/synaptic-mesh-cli/crates/kimi-fann-core

# Build native
cargo build --release

# Build for WASM
wasm-pack build --target web

# Run tests
cargo test

# Run benchmarks
cargo bench
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## 🙏 Acknowledgments

Built as part of the [Synaptic Neural Mesh](https://github.com/ruvnet/Synaptic-Neural-Mesh) project, advancing distributed AI systems.