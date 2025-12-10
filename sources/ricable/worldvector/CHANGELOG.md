# Changelog

All notable changes to Ruvector will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation suite
  - Getting Started guide
  - Installation guide
  - Basic tutorial
  - Advanced features guide
  - Architecture documentation
  - API references for all platforms
  - 10+ code examples
  - Contributing guide
  - Migration guide from AgenticDB

## [0.1.0] - 2025-11-19

### Added

#### Phase 1: Foundation (Completed)
- Core vector database implementation with redb storage
- Memory-mapped vector access via memmap2
- SIMD-optimized distance metrics (Euclidean, Cosine, Dot Product, Manhattan)
- Basic flat index for exact search
- Initial test suite and benchmarks
- CLI scaffolding

#### Phase 2: HNSW Indexing (Completed)
- HNSW (Hierarchical Navigable Small World) graph implementation
- Integration with hnsw_rs crate
- Parallel index construction using rayon
- Zero-copy serialization with rkyv
- Batch insert operations
- Scalar quantization (int8) for 4x memory compression
- Comprehensive HNSW integration tests
- Performance benchmarks:
  - Distance metrics: 200-300x speedup with SimSIMD
  - HNSW search: Sub-millisecond latency for 1M vectors
  - Batch operations: 10-100x faster than individual operations

#### Phase 3: AgenticDB Compatibility (Completed)
- Full AgenticDB API implementation with 5-table schema:
  - `vectors_table`: Core embeddings with metadata
  - `reflexion_episodes`: Self-critique memory for agent learning
  - `skills_library`: Reusable action patterns
  - `causal_edges`: Hypergraph-based cause-effect relationships
  - `learning_sessions`: RL training data with 9 algorithms
- Reflexion Memory API:
  - Store/retrieve self-critique episodes
  - Semantic search over critiques
  - Learning from past mistakes
- Skill Library:
  - Create and search skills
  - Auto-consolidation from successful patterns
  - Usage tracking and success metrics
- Causal Memory Graph:
  - Add causal relationships with confidence scores
  - Query with utility function (similarity + uplift - latency)
  - Hypergraph support for n-ary relationships
- Learning Sessions:
  - 9 RL algorithms (Q-Learning, SARSA, DQN, Policy Gradient, Actor-Critic, PPO, Decision Transformer, MCTS, Model-Based)
  - Experience replay storage
  - Prediction with conformal confidence intervals
- Complete AgenticDB demo application
- 10-100x performance improvement over original agenticDB

#### Phase 4: Advanced Features (Completed)
- Product Quantization (PQ):
  - 8-16x memory compression
  - 90-95% recall preservation
  - Configurable subspaces and codebook size
- Filtered Search:
  - Pre-filtering strategy (efficient for selective filters)
  - Post-filtering strategy (better for loose constraints)
  - Complex filter expressions (AND, OR, NOT, comparison operators)
- Hybrid Search:
  - Vector similarity + BM25 keyword scoring
  - Configurable weight balancing
  - Integrated text indexing
- MMR (Maximal Marginal Relevance):
  - Diversity-aware result ranking
  - Configurable relevance vs. diversity trade-off
- Conformal Prediction:
  - Distribution-free confidence intervals
  - Calibration-based uncertainty quantification
  - Adaptive top-k selection
- Advanced integration tests
- Performance monitoring and metrics

#### Phase 5: Multi-Platform Deployment (Completed)
- **Node.js Bindings** (ruvector-node):
  - NAPI-RS integration for high-performance native bindings
  - Complete TypeScript type definitions
  - Async/await API
  - Zero-copy buffer sharing with Float32Array
  - Automatic platform-specific binary selection
  - npm package ready
- **WASM Module** (ruvector-wasm):
  - wasm-bindgen integration
  - Browser-compatible vector database
  - SIMD detection and dual builds (SIMD/non-SIMD)
  - Web Workers support for parallelism
  - IndexedDB persistence integration
  - React example application
  - Vanilla JS example
- **CLI Tool** (ruvector-cli):
  - Create, insert, search, info, benchmark commands
  - JSON, CSV, NPY format support
  - Progress bars and colored output
  - Configuration file support
  - Shell completions (bash, zsh, fish)
- **Cross-platform builds**:
  - Linux (x64, arm64)
  - macOS (x64, arm64)
  - Windows (x64, arm64)
  - WASM (browser, Node.js)

#### Phase 6: Performance Optimization (Completed)
- SIMD intrinsics optimization:
  - AVX2 support for x86_64
  - ARM NEON support
  - Runtime feature detection
  - Fallback implementations
- Lock-free data structures:
  - Concurrent HNSW reads
  - Lock-free query queues
  - Atomic reference counting
- Cache-optimized layouts:
  - Structure-of-Arrays (SoA) format
  - 64-byte cache line alignment
  - Prefetching hints
- Arena allocators:
  - Batch allocation/deallocation
  - Reduced memory fragmentation
- Comprehensive benchmarking suite:
  - Distance metrics benchmark
  - HNSW search benchmark
  - Batch operations benchmark
  - Quantization benchmark
  - Memory usage benchmark
  - Latency percentiles
  - Throughput measurements

### Performance Achievements

- **10-100x faster** than Python/TypeScript implementations
- **Sub-millisecond latency** (p50 < 0.8ms for 1M vectors)
- **95%+ recall** with HNSW (ef_search=100)
- **4-32x memory compression** with quantization
- **200-300x distance calculation speedup** with SIMD
- **Near-linear scaling** to CPU core count
- **Instant loading** with memory-mapped vectors and rkyv

### Documentation

- Comprehensive README with technical plan
- Rustdoc comments for all public APIs
- AgenticDB API documentation
- Phase implementation summaries
- Performance tuning guides
- Build optimization guides
- Test suite documentation
- WASM API documentation

### Dependencies

- **Core**: redb, memmap2, hnsw_rs, simsimd, rayon, crossbeam
- **Serialization**: rkyv, bincode, serde, serde_json
- **Node.js**: napi, napi-derive
- **WASM**: wasm-bindgen, wasm-bindgen-futures, js-sys, web-sys
- **Async**: tokio
- **Utilities**: thiserror, anyhow, tracing
- **Math**: ndarray, rand, rand_distr
- **CLI**: clap, indicatif, console
- **Testing**: criterion, proptest, mockall
- **Performance**: dashmap, parking_lot, once_cell

### Known Limitations

- Single-node only (no distributed queries yet)
- Write operations require exclusive lock
- Maximum 10M vectors by default (configurable)
- Advanced features (hypergraphs, learned indexes) in experimental state

### Breaking Changes

None (initial release)

## Future Roadmap

### v0.2.0 (Planned)
- Distributed query processing
- Horizontal scaling with sharding
- GPU acceleration for distance calculations
- Improved quantization algorithms
- Enhanced hypergraph support
- Temporal indexes for time-series

### v0.3.0 (Planned)
- Learned index structures (hybrid with HNSW)
- Neural hash functions
- Enhanced causal inference
- Model-based RL integration
- Real-time index updates
- Streaming data support

### v1.0.0 (Future)
- Production-grade distributed system
- High availability and replication
- Advanced AI agent features
- Neuromorphic hardware support
- Complete documentation and examples
- Enterprise support options

## Contributing

We welcome contributions! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## License

Ruvector is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [hnsw_rs](https://github.com/jean-pierreBoth/hnsw_rs) - HNSW implementation
- [simsimd](https://github.com/ashvardanian/simsimd) - SIMD distance metrics
- [redb](https://github.com/cberner/redb) - Embedded database
- [NAPI-RS](https://napi.rs/) - Node.js bindings
- [wasm-bindgen](https://github.com/rustwasm/wasm-bindgen) - WASM bindings
- AgenticDB team for API design inspiration

---

For questions or issues, please visit: https://github.com/ruvnet/ruvector/issues
