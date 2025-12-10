# 🧠 Kimi-FANN-Core

**Kimi-K2 micro-expert implementation using ruv-FANN with WASM support**

[![Crates.io](https://img.shields.io/crates/v/kimi-fann-core.svg)](https://crates.io/crates/kimi-fann-core)
[![Documentation](https://docs.rs/kimi-fann-core/badge.svg)](https://docs.rs/kimi-fann-core)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

## 🚀 Overview

`kimi-fann-core` is a specialized neural network implementation designed for Kimi-K2 to Rust-WASM conversion. It provides micro-expert architectures with efficient memory management, compression, and high-performance WASM support.

## ✨ Key Features

- **🔬 Micro-Expert Architecture**: Specialized small neural networks for specific domains
- **⚡ WASM Optimization**: Highly optimized for WebAssembly with SIMD support
- **🗜️ Compression**: Advanced compression for expert storage and transfer
- **🚀 Performance**: Sub-100ms inference times
- **🔄 Memory Efficient**: LRU caching and smart memory management
- **📡 Router System**: Intelligent routing between micro-experts

## 🛠️ Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
kimi-fann-core = "0.1.0"
```

For WASM projects:

```toml
[dependencies]
kimi-fann-core = { version = "0.1.0", features = ["wasm-opt"] }
```

## 📖 Usage

### Basic Expert Creation

```rust
use kimi_fann_core::Expert;

// Create a reasoning expert
let expert = Expert::new()
    .with_domain("reasoning")
    .with_layers(&[32, 64, 32])
    .build()?;

// Train the expert
expert.train(&training_data)?;

// Use for inference
let result = expert.forward(&input)?;
```

### WASM Integration

```rust
use wasm_bindgen::prelude::*;
use kimi_fann_core::WasmExpert;

#[wasm_bindgen]
pub fn create_kimi_expert() -> WasmExpert {
    WasmExpert::new()
        .with_compression(true)
        .with_simd(true)
        .build()
}
```

## 🏗️ Architecture

### Expert Domains

- **🧠 Reasoning**: Logic and problem-solving
- **💻 Coding**: Programming and code generation  
- **🗣️ Language**: Natural language processing
- **🔢 Mathematics**: Mathematical computations
- **🛠️ Tool-Use**: External tool integration
- **📚 Context**: Context understanding and memory

### Router System

```rust
use kimi_fann_core::Router;

let router = Router::new()
    .add_expert("reasoning", reasoning_expert)
    .add_expert("coding", coding_expert)
    .build();

// Route input to appropriate expert
let result = router.route(&input)?;
```

## 🎯 Performance

- **Inference Time**: < 100ms for most tasks
- **Memory Usage**: < 50MB per expert
- **Compression Ratio**: 10:1 for expert storage
- **WASM Bundle Size**: < 2MB optimized

## 🔧 Features

- `std` - Standard library support (default)
- `parallel` - Parallel processing with tokio
- `compression` - Expert compression support
- `simd` - SIMD acceleration
- `wasm-opt` - WASM optimizations
- `no-std` - No standard library (embedded)

## 🤝 Integration

Works seamlessly with:

- **Synaptic Neural Mesh**: Distributed AI coordination
- **Synaptic Neural WASM**: WASM runtime optimization
- **Kimi-K2 API**: Direct integration with Kimi services

## 📊 Benchmarks

```bash
cargo bench
```

## 🧪 Testing

```bash
# Run all tests
cargo test

# WASM tests
wasm-pack test --node
```

## 📚 Documentation

- [API Documentation](https://docs.rs/kimi-fann-core)
- [Architecture Guide](docs/architecture.md)
- [WASM Integration](docs/wasm.md)
- [Performance Tuning](docs/performance.md)

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md).

## 📄 License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## 🔗 Related Projects

- [Synaptic Neural Mesh](https://github.com/ruvnet/Synaptic-Neural-Mesh)
- [Kimi-K2 Integration](https://github.com/ruvnet/Synaptic-Neural-Mesh/tree/main/plans/Kimi-K2)
- [Synaptic Market](https://github.com/ruvnet/Synaptic-Neural-Mesh/tree/main/plans/synaptic-market)

---

**Built with ❤️ for the Kimi-K2 ecosystem**