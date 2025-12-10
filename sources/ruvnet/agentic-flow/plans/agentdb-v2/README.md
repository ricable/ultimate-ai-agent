# AgentDB v2: RuVector Integration Plan

## Overview

AgentDB v2 integrates RuVector as an optional high-performance backend with automatic detection. When RuVector is installed, it becomes the default vector engine, providing 8x faster search and 2-32x memory reduction.

## Design Principle

```
┌─────────────────────────────────────────────────────────────┐
│                      AgentDB v2                              │
├─────────────────────────────────────────────────────────────┤
│  agentdb init                    # Auto-detect backend       │
│  agentdb init --backend=ruvector # Force RuVector           │
│  agentdb init --backend=hnswlib  # Force legacy             │
└─────────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
   ┌───────────────┐               ┌───────────────┐
   │   RuVector    │               │   hnswlib     │
   │   (Default)   │               │   (Fallback)  │
   │               │               │               │
   │ • 61µs search │               │ • 500µs search│
   │ • GNN learning│               │ • Basic HNSW  │
   │ • Compression │               │ • No compress │
   │ • Graph/Cypher│               │ • No graph    │
   └───────────────┘               └───────────────┘
```

## Installation Modes

### Mode 1: Auto-Detection (Recommended)
```bash
npm install agentdb

# If @ruvector/core is available → uses RuVector
# Otherwise → uses hnswlib-node
```

### Mode 2: Explicit RuVector
```bash
npm install agentdb @ruvector/core @ruvector/gnn
agentdb init --backend=ruvector
```

### Mode 3: Legacy Only
```bash
npm install agentdb
agentdb init --backend=hnswlib
```

## Feature Matrix

| Feature | hnswlib (Fallback) | RuVector (Default) |
|---------|-------------------|-------------------|
| Vector search | ✅ | ✅ |
| HNSW indexing | ✅ | ✅ (faster) |
| Batch operations | ✅ | ✅ (faster) |
| Persistence | ✅ | ✅ |
| Tiered compression | ❌ | ✅ |
| GNN self-learning | ❌ | ✅ |
| Graph queries | ❌ | ✅ |
| Distributed mode | ❌ | ✅ |
| WASM fallback | ❌ | ✅ |

## Performance Targets

| Metric | hnswlib | RuVector | Improvement |
|--------|---------|----------|-------------|
| Search (k=10) | 500µs | 61µs | 8.2x |
| Search (k=100) | 2.1ms | 164µs | 12.8x |
| Insert throughput | 5K/s | 47K/s | 9.4x |
| Memory (100K vec) | 412MB | 48MB | 8.6x |
| Index build | 8.4s | 2.1s | 4.0x |

## Project Structure

```
plans/agentdb-v2/
├── README.md                 # This file
├── ARCHITECTURE.md           # Technical architecture
├── IMPLEMENTATION.md         # Step-by-step implementation
├── API.md                    # API design and interfaces
├── MIGRATION.md              # Migration guide
├── benchmarks/
│   ├── BENCHMARK_PLAN.md     # Benchmarking strategy
│   ├── baseline.json         # Performance baselines
│   └── scenarios/            # Test scenarios
├── security/
│   ├── SECURITY_CHECKLIST.md # Security review checklist
│   ├── THREAT_MODEL.md       # Threat modeling
│   └── audit-config.json     # Security scan config
├── tests/
│   ├── REGRESSION_PLAN.md    # Regression test strategy
│   ├── TEST_MATRIX.md        # Platform test matrix
│   └── fixtures/             # Test data fixtures
└── workflows/
    ├── ci.yml                # Main CI pipeline
    ├── platform-builds.yml   # Platform-specific builds
    ├── benchmarks.yml        # Automated benchmarks
    ├── security-scan.yml     # Security scanning
    └── release.yml           # Release workflow
```

## Implementation Phases

### Phase 1: Core Integration (Week 1-2)
- [ ] Backend abstraction interface
- [ ] RuVector adapter implementation
- [ ] Auto-detection logic
- [ ] CLI init command updates
- [ ] Unit tests for both backends

### Phase 2: Enhanced Features (Week 3-4)
- [ ] GNN integration for ReasoningBank
- [ ] Tiered compression support
- [ ] Graph query adapter (optional)
- [ ] Performance benchmarks

### Phase 3: CI/CD & Quality (Week 5)
- [ ] GitHub Actions workflows
- [ ] Platform-specific builds
- [ ] Security scanning
- [ ] Regression test suite
- [ ] Documentation

### Phase 4: Release (Week 6)
- [ ] Beta release
- [ ] Performance validation
- [ ] Migration guide
- [ ] GA release

## Success Criteria

### Performance
- [ ] Search latency < 100µs (p50)
- [ ] Throughput > 10K QPS
- [ ] Memory reduction > 4x with compression

### Quality
- [ ] 100% backward compatibility
- [ ] Zero security vulnerabilities (critical/high)
- [ ] Test coverage > 80%
- [ ] All platforms pass CI

### Usability
- [ ] Auto-detection works seamlessly
- [ ] Clear error messages on failure
- [ ] Migration path documented

## Quick Links

- [Architecture](./ARCHITECTURE.md)
- [Implementation Guide](./IMPLEMENTATION.md)
- [API Design](./API.md)
- [Benchmark Plan](./benchmarks/BENCHMARK_PLAN.md)
- [Security Checklist](./security/SECURITY_CHECKLIST.md)
- [Regression Tests](./tests/REGRESSION_PLAN.md)
- [CI Workflows](./workflows/)

## Dependencies

### Required
```json
{
  "dependencies": {
    "better-sqlite3": "^11.10.0",
    "hnswlib-node": "^3.0.0"
  }
}
```

### Optional (RuVector)
```json
{
  "optionalDependencies": {
    "@ruvector/core": "^0.1.15",
    "@ruvector/gnn": "^0.1.15",
    "@ruvector/graph-node": "^0.1.x"
  }
}
```

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| RuVector instability | Medium | High | Keep hnswlib fallback |
| Platform build failures | Medium | Medium | WASM fallback |
| Performance regression | Low | High | Automated benchmarks |
| Breaking API changes | Low | High | Strict versioning |
| Security vulnerabilities | Low | Critical | Automated scanning |
