# ruvector-attention

Advanced attention mechanisms for vector search and geometric AI.

[![Crates.io](https://img.shields.io/crates/v/ruvector-attention.svg)](https://crates.io/crates/ruvector-attention)
[![Documentation](https://docs.rs/ruvector-attention/badge.svg)](https://docs.rs/ruvector-attention)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

## Features

- 🚀 **High-Performance**: SIMD-accelerated attention computations
- 🎯 **Ergonomic API**: Fluent builder pattern and preset configurations
- 📦 **Modular Design**: Mix and match attention mechanisms
- 🔧 **Flexible**: Support for standard, sparse, graph, and geometric attention
- 🧠 **Advanced**: MoE routing, hyperbolic attention, and more

## Supported Attention Mechanisms

### Standard Attention
- **Scaled Dot-Product**: `softmax(QK^T / √d)V`
- **Multi-Head**: Parallel attention heads with diverse representations

### Sparse Attention (Memory Efficient)
- **Flash Attention**: O(n) memory complexity with tiled computation
- **Linear Attention**: O(n) complexity using kernel approximation
- **Local-Global**: Sliding window + global tokens (Longformer-style)

### Geometric Attention
- **Hyperbolic Attention**: Attention in hyperbolic space for hierarchical data
- **Mixed Curvature**: Dynamic curvature for complex geometries

### Graph Attention
- **Edge-Featured GAT**: Graph attention with edge features
- **RoPE**: Rotary Position Embeddings for graphs

### Mixture-of-Experts
- **MoE Attention**: Learned routing to specialized expert modules
- **Top-k Routing**: Efficient expert selection

## Quick Start

```rust
use ruvector_attention::sdk::*;

// Simple multi-head attention
let attention = multi_head(768, 12)
    .dropout(0.1)
    .causal(true)
    .build()?;

// Use preset configurations
let bert = AttentionPreset::Bert.builder(768).build()?;
let gpt = AttentionPreset::Gpt.builder(768).build()?;

// Build pipelines with normalization
let pipeline = AttentionPipeline::new()
    .add_attention(attention)
    .add_norm(NormType::LayerNorm)
    .add_residual();

// Compute attention
let query = vec![0.5; 768];
let keys = vec![&query[..]; 10];
let values = vec![&query[..]; 10];

let output = pipeline.run(&query, &keys, &values)?;
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ruvector-attention = "0.1"
```

Or with specific features:

```toml
[dependencies]
ruvector-attention = { version = "0.1", features = ["simd", "wasm"] }
```

## SDK Overview

### Builder API

The builder provides a fluent interface for configuring attention:

```rust
use ruvector_attention::sdk::*;

// Flash attention for long sequences
let flash = flash(1024, 128)  // dim, block_size
    .causal(true)
    .dropout(0.1)
    .build()?;

// Linear attention for O(n) complexity
let linear = linear(512, 256)  // dim, num_features
    .build()?;

// MoE attention with 8 experts
let moe = moe(512, 8, 2)  // dim, num_experts, top_k
    .expert_capacity(1.25)
    .jitter_noise(0.01)
    .build()?;

// Hyperbolic attention for hierarchies
let hyperbolic = hyperbolic(512, -1.0)  // dim, curvature
    .build()?;
```

### Pipeline API

Compose attention with pre/post processing:

```rust
use ruvector_attention::sdk::*;

let attention = multi_head(768, 12).build()?;

let pipeline = AttentionPipeline::new()
    .add_norm(NormType::LayerNorm)     // Pre-normalization
    .add_attention(attention)           // Attention layer
    .add_dropout(0.1)                   // Dropout
    .add_residual()                     // Residual connection
    .add_norm(NormType::RMSNorm);      // Post-normalization

let output = pipeline.run(&query, &keys, &values)?;
```

### Preset Configurations

Pre-configured attention for popular models:

```rust
use ruvector_attention::sdk::presets::*;

// Model-specific presets
let bert = AttentionPreset::Bert.builder(768).build()?;
let gpt = AttentionPreset::Gpt.builder(768).build()?;
let longformer = AttentionPreset::Longformer.builder(512).build()?;
let flash = AttentionPreset::FlashOptimized.builder(1024).build()?;
let t5 = AttentionPreset::T5.builder(768).build()?;
let vit = AttentionPreset::ViT.builder(768).build()?;

// Smart selection based on use case
let attention = for_sequences(512, max_len).build()?;  // Auto-select by length
let graph_attn = for_graphs(256, hierarchical).build()?;  // Graph attention
let fast_attn = for_large_scale(1024).build()?;  // Flash attention

// By model name
let bert = from_model_name("bert", 768)?;
let gpt2 = from_model_name("gpt2", 768)?;
```

## Architecture

```
ruvector-attention/
├── src/
│   ├── lib.rs                 # Main crate entry
│   ├── error.rs              # Error types
│   ├── traits.rs             # Core attention traits
│   ├── attention/            # Standard attention
│   │   ├── scaled_dot_product.rs
│   │   └── multi_head.rs
│   ├── sparse/               # Sparse attention
│   │   ├── flash.rs
│   │   ├── linear.rs
│   │   └── local_global.rs
│   ├── graph/                # Graph attention
│   │   ├── edge_featured.rs
│   │   └── rope.rs
│   ├── hyperbolic/           # Geometric attention
│   │   ├── hyperbolic_attention.rs
│   │   └── poincare.rs
│   ├── moe/                  # Mixture-of-Experts
│   │   ├── expert.rs
│   │   ├── router.rs
│   │   └── moe_attention.rs
│   ├── training/             # Training utilities
│   │   ├── loss.rs
│   │   ├── optimizer.rs
│   │   └── curriculum.rs
│   └── sdk/                  # High-level SDK
│       ├── builder.rs        # Fluent builder API
│       ├── pipeline.rs       # Composable pipelines
│       └── presets.rs        # Model presets
```

## Examples

### Transformer Block

```rust
use ruvector_attention::sdk::*;

fn create_transformer_block(dim: usize) -> AttentionResult<AttentionPipeline> {
    let attention = multi_head(dim, 12)
        .dropout(0.1)
        .build()?;

    Ok(AttentionPipeline::new()
        .add_norm(NormType::LayerNorm)
        .add_attention(attention)
        .add_dropout(0.1)
        .add_residual())
}
```

### Long Context Processing

```rust
use ruvector_attention::sdk::*;

fn create_long_context_attention(dim: usize, max_len: usize)
    -> AttentionResult<Box<dyn Attention>> {
    if max_len <= 2048 {
        multi_head(dim, 12).build()
    } else if max_len <= 16384 {
        local_global(dim, 512).build()
    } else {
        linear(dim, dim / 4).build()
    }
}
```

### Graph Neural Network

```rust
use ruvector_attention::sdk::*;

fn create_graph_attention(dim: usize, is_tree: bool)
    -> AttentionResult<Box<dyn Attention>> {
    if is_tree {
        hyperbolic(dim, -1.0).build()  // Hyperbolic for tree-like
    } else {
        multi_head(dim, 8).build()     // Standard for general graphs
    }
}
```

## Performance

### Complexity Comparison

| Mechanism | Time | Memory | Use Case |
|-----------|------|--------|----------|
| Scaled Dot-Product | O(n²) | O(n²) | Short sequences |
| Multi-Head | O(n²) | O(n²) | Standard transformers |
| Flash Attention | O(n²) | O(n) | Long sequences |
| Linear Attention | O(n) | O(n) | Very long sequences |
| Local-Global | O(n·w) | O(n·w) | Document processing |
| Hyperbolic | O(n²) | O(n²) | Hierarchical data |
| MoE | O(n²/E) | O(n²) | Specialized tasks |

Where:
- `n` = sequence length
- `w` = local window size
- `E` = number of experts

### Benchmarks

On a typical workload (batch_size=32, seq_len=512, dim=768):

- **Flash Attention**: 2.3x faster, 5x less memory than standard
- **Linear Attention**: O(n) scaling for sequences >4096
- **Local-Global**: 60% of standard attention cost for w=256

## Features

- `simd` - SIMD acceleration (default, enabled)
- `wasm` - WebAssembly support
- `napi` - Node.js bindings

## Documentation

- [SDK Guide](docs/SDK_GUIDE.md) - Comprehensive SDK usage guide
- [API Documentation](https://docs.rs/ruvector-attention) - Full API reference
- [Examples](examples/) - Working code examples

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Citation

If you use this crate in your research, please cite:

```bibtex
@software{ruvector_attention,
  title = {ruvector-attention: Advanced Attention Mechanisms for Vector Search},
  author = {ruvector contributors},
  year = {2025},
  url = {https://github.com/ruvnet/ruvector}
}
```

## Related Projects

- [ruvector](../ruvector) - Core vector search engine
- [ruvector-graph](../ruvector-graph) - Graph neural networks
- [ruvector-gnn](../ruvector-gnn) - Geometric neural networks
