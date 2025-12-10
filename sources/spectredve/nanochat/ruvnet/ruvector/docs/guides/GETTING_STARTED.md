# Getting Started with Ruvector

## What is Ruvector?

Ruvector is a high-performance, Rust-native vector database designed for modern AI applications. It provides:

- **10-100x performance improvements** over Python/TypeScript implementations
- **Sub-millisecond latency** with HNSW indexing and SIMD optimization
- **AgenticDB API compatibility** for seamless migration
- **Multi-platform deployment** (Rust, Node.js, WASM/Browser, CLI)
- **Advanced features** including quantization, hybrid search, and causal memory

## Quick Start

### Installation

#### Rust
```bash
# Add to Cargo.toml
[dependencies]
ruvector-core = "0.1.0"
```

#### Node.js
```bash
npm install ruvector
# or
yarn add ruvector
```

#### CLI
```bash
cargo install ruvector-cli
```

### Basic Usage

#### Rust
```rust
use ruvector_core::{VectorDB, VectorEntry, SearchQuery, DbOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new vector database
    let mut options = DbOptions::default();
    options.dimensions = 128;
    options.storage_path = "./vectors.db".to_string();

    let db = VectorDB::new(options)?;

    // Insert a vector
    let entry = VectorEntry {
        id: None,
        vector: vec![0.1; 128],
        metadata: None,
    };
    let id = db.insert(entry)?;
    println!("Inserted vector: {}", id);

    // Search for similar vectors
    let query = SearchQuery {
        vector: vec![0.1; 128],
        k: 10,
        filter: None,
        include_vectors: false,
    };
    let results = db.search(&query)?;

    for (i, result) in results.iter().enumerate() {
        println!("{}. ID: {}, Distance: {}", i + 1, result.id, result.distance);
    }

    Ok(())
}
```

#### Node.js
```javascript
const { VectorDB } = require('ruvector');

async function main() {
    // Create a new vector database
    const db = new VectorDB({
        dimensions: 128,
        storagePath: './vectors.db',
        distanceMetric: 'cosine'
    });

    // Insert a vector
    const id = await db.insert({
        vector: new Float32Array(128).fill(0.1),
        metadata: { text: 'Example document' }
    });
    console.log('Inserted vector:', id);

    // Search for similar vectors
    const results = await db.search({
        vector: new Float32Array(128).fill(0.1),
        k: 10
    });

    results.forEach((result, i) => {
        console.log(`${i + 1}. ID: ${result.id}, Distance: ${result.distance}`);
    });
}

main().catch(console.error);
```

#### CLI
```bash
# Create a database
ruvector create --path ./vectors.db --dimensions 128

# Insert vectors from a JSON file
ruvector insert --db ./vectors.db --input vectors.json --format json

# Search for similar vectors
ruvector search --db ./vectors.db --query "[0.1, 0.2, ...]" --top-k 10

# Show database info
ruvector info --db ./vectors.db
```

## Core Concepts

### 1. Vector Database

A vector database stores high-dimensional vectors (embeddings) and enables fast similarity search. Common use cases:
- **Semantic search**: Find similar documents, images, or audio
- **Recommendation systems**: Find similar products or content
- **RAG (Retrieval Augmented Generation)**: Retrieve relevant context for LLMs
- **Agent memory**: Store and retrieve experiences for AI agents

### 2. Distance Metrics

Ruvector supports multiple distance metrics:
- **Euclidean (L2)**: Standard distance in Euclidean space
- **Cosine**: Measures angle between vectors (normalized dot product)
- **Dot Product**: Inner product (useful for pre-normalized vectors)
- **Manhattan (L1)**: Sum of absolute differences

### 3. HNSW Indexing

Hierarchical Navigable Small World (HNSW) provides:
- **O(log n) search complexity**
- **95%+ recall** with proper tuning
- **Sub-millisecond latency** for millions of vectors

Key parameters:
- `m`: Connections per node (16-64, default 32)
- `ef_construction`: Build quality (100-400, default 200)
- `ef_search`: Search quality (50-500, default 100)

### 4. Quantization

Reduce memory usage with quantization:
- **Scalar (int8)**: 4x compression, 97-99% recall
- **Product**: 8-16x compression, 90-95% recall
- **Binary**: 32x compression, 80-90% recall (filtering)

### 5. AgenticDB Features

Advanced features for AI agents:
- **Reflexion Memory**: Self-critique episodes for learning
- **Skill Library**: Reusable action patterns
- **Causal Memory**: Cause-effect relationships
- **Learning Sessions**: RL training data

## Next Steps

- [Installation Guide](INSTALLATION.md) - Detailed installation instructions
- [Basic Tutorial](BASIC_TUTORIAL.md) - Step-by-step tutorial
- [Advanced Features](ADVANCED_FEATURES.md) - Hybrid search, quantization, filtering
- [AgenticDB Migration Guide](../MIGRATION.md) - Migrate from agenticDB
- [API Reference](../api/) - Complete API documentation
- [Examples](../../examples/) - Working code examples

## Performance Tips

1. **Choose the right distance metric**: Cosine for normalized embeddings, Euclidean otherwise
2. **Tune HNSW parameters**: Higher `m` and `ef_construction` for better recall
3. **Enable quantization**: Reduces memory 4-32x with minimal accuracy loss
4. **Batch operations**: Use `insert_batch()` for better throughput
5. **Memory-map large datasets**: Set `mmap_vectors: true` for datasets larger than RAM

## Common Issues

### Out of Memory
- Enable quantization to reduce memory usage
- Use memory-mapped vectors for large datasets
- Reduce `max_elements` or increase available RAM

### Slow Search
- Lower `ef_search` for faster (but less accurate) search
- Enable quantization for cache-friendly operations
- Check if SIMD is enabled (`RUSTFLAGS="-C target-cpu=native"`)

### Low Recall
- Increase `ef_construction` during index building
- Increase `ef_search` during queries
- Use full-precision vectors instead of quantization

## Community & Support

- **GitHub**: [https://github.com/ruvnet/ruvector](https://github.com/ruvnet/ruvector)
- **Issues**: [https://github.com/ruvnet/ruvector/issues](https://github.com/ruvnet/ruvector/issues)
- **Documentation**: [https://docs.rs/ruvector-core](https://docs.rs/ruvector-core)

## License

Ruvector is licensed under the MIT License. See [LICENSE](../../LICENSE) for details.
