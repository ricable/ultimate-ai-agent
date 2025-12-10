# DAA Performance Benchmarks

Comprehensive performance benchmarking suite comparing native NAPI-rs vs WASM implementations across the DAA ecosystem.

## 📊 Overview

This benchmark suite provides:

- **Crypto Benchmarks**: ML-KEM-768, ML-DSA, BLAKE3 hashing, quantum fingerprinting
- **Orchestrator Benchmarks**: Workflow execution, MRAP loop, rules evaluation, event processing
- **Prime ML Benchmarks**: Gradient aggregation, model updates, zero-copy operations, Byzantine-tolerant training

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
npm install

# Build native bindings (optional, for native vs WASM comparison)
cd ../qudag/qudag-napi && npm run build
cd ../../benchmarks

# Build WASM bindings
cd ../qudag/qudag-wasm && wasm-pack build --target nodejs
cd ../../benchmarks
```

### Running Benchmarks

```bash
# Run all benchmarks and generate reports
npm run bench:compare

# Or run individual benchmark suites
npm run bench:crypto          # WASM crypto benchmarks
npm run bench:crypto-compare  # Native vs WASM comparison
npm run bench:orchestrator    # Orchestrator benchmarks
npm run bench:prime           # Prime ML benchmarks

# Run native Rust benchmarks
npm run bench:native
```

### Generating Reports

```bash
# Generate HTML report
npm run report:generate

# Create visualizations
npm run report:visualize

# Generate everything
npm run report:html

# Open report in browser
open reports/benchmark_report.html
```

## 📁 Directory Structure

```
benchmarks/
├── crypto/                      # Cryptographic operation benchmarks
│   ├── crypto_native.rs         # Native Rust benchmarks
│   ├── crypto_wasm.js           # WASM JavaScript benchmarks
│   └── crypto_compare.js        # Native vs WASM comparison
├── orchestrator/                # Workflow and orchestration benchmarks
│   ├── orchestrator_benchmarks.rs
│   └── orchestrator_wasm.js
├── prime/                       # Machine learning benchmarks
│   ├── prime_ml_benchmarks.rs
│   └── prime_wasm.js
├── scripts/                     # Utilities and runners
│   ├── compare_runner.js        # Comprehensive runner
│   ├── generate_report.js       # HTML report generator
│   └── visualize.js             # Chart generation
├── utils/                       # Statistical analysis
│   ├── statistics.js            # Statistical functions
│   └── analyzer.js              # Result analyzer
└── reports/                     # Generated reports and charts
    ├── benchmark_report.html
    ├── crypto_comparison.json
    ├── orchestrator_results.json
    ├── prime_ml_results.json
    └── *.png                    # Generated charts
```

## 🔬 Benchmark Categories

### 1. Cryptographic Operations

**ML-KEM-768 (Key Encapsulation)**
- Key generation
- Encapsulation
- Decapsulation
- Full key exchange workflow

**ML-DSA (Digital Signatures)**
- Signing
- Verification
- Full signature workflow

**BLAKE3 Hashing**
- Various data sizes: 1KB, 10KB, 100KB, 1MB, 10MB
- Throughput measurement

**Quantum Fingerprinting**
- Fingerprint generation at various data sizes
- Collision resistance testing

**Expected Results:**
- Native NAPI-rs: **2-5x faster** than WASM
- Target latency: <10ms for most operations
- Throughput: 100+ ops/sec for key generation

### 2. Orchestrator Performance

**Workflow Operations**
- Simple workflow creation
- Complex workflow with dependencies
- MRAP loop execution (Monitor → Reason → Act → Reflect → Plan)
- Parallel workflow execution

**Rules Evaluation**
- Simple rule evaluation
- Complex ruleset with 100+ rules
- Dynamic rule addition

**Event Processing**
- Event throughput at 10, 100, 1000, 10000 events
- Queue management
- State monitoring

**Expected Results:**
- Workflow creation: <5ms
- MRAP loop execution: <50ms
- Event processing: 1000+ events/sec

### 3. Prime ML Operations

**Gradient Aggregation**
- Federated averaging with 5, 10, 20, 50, 100 nodes
- Various gradient sizes: 1K, 10K, 100K parameters
- Byzantine-tolerant trimmed mean aggregation

**Model Updates**
- Model sizes: 1K, 10K, 100K, 1M parameters
- Learning rate application
- Zero-copy tensor operations

**Distributed Training**
- Training coordination
- Gradient compression (top-k sparsification)
- Model serialization/deserialization

**Expected Results:**
- Gradient aggregation: <100ms for 50 nodes
- Model update: <10ms for 100K parameters
- Scalability efficiency: >80% up to 20 nodes

## 📈 Performance Metrics

### Primary Metrics

- **Throughput**: Operations per second (ops/sec)
- **Latency**: Mean execution time (ms)
- **Speedup**: Native performance / WASM performance
- **p50, p95, p99**: Latency percentiles

### Statistical Analysis

- Mean, median, standard deviation
- Confidence intervals (95%)
- Relative standard error (RSE)
- Regression detection

### Visualization

- Speedup comparison charts
- Throughput comparison charts
- Scalability analysis
- Performance trends

## 🎯 Target Performance Goals

### Crypto Operations (Native vs WASM)

| Operation | WASM | Native (Target) | Speedup |
|-----------|------|-----------------|---------|
| ML-KEM Keygen | 5.2ms | 1.8ms | 2.9x |
| ML-KEM Encapsulate | 3.1ms | 1.1ms | 2.8x |
| ML-DSA Sign | 4.5ms | 1.5ms | 3.0x |
| ML-DSA Verify | 3.8ms | 1.3ms | 2.9x |
| BLAKE3 Hash (1MB) | 8.2ms | 2.1ms | 3.9x |

### Orchestrator Operations

| Operation | Target | Acceptable |
|-----------|--------|------------|
| Workflow Creation | <5ms | <10ms |
| MRAP Loop | <50ms | <100ms |
| Event Processing (1000 events) | <1s | <2s |
| Rule Evaluation (100 rules) | <10ms | <20ms |

### Prime ML Operations

| Operation | Target | Acceptable |
|-----------|--------|------------|
| Gradient Aggregation (10 nodes) | <50ms | <100ms |
| Model Update (100K params) | <10ms | <20ms |
| Zero-Copy Operation (1MB) | <1ms | <5ms |

## 🔧 Configuration

### Criterion.rs Configuration

```toml
[profile.release]
lto = true
codegen-units = 1
opt-level = 3
```

### Benchmark.js Configuration

```javascript
const suite = new Benchmark.Suite({
  async: true,
  minSamples: 100,
  maxTime: 5
});
```

## 📊 Report Interpretation

### Speedup Indicators

- **🟢 Excellent (3x+)**: Strong recommendation for native bindings
- **🟡 Good (2-3x)**: Recommended for performance-critical paths
- **🟠 Modest (<2x)**: Consider WASM for better portability

### Statistical Significance

- **RSE < 1%**: Very consistent, highly reliable
- **RSE < 5%**: Consistent, reliable
- **RSE > 10%**: High variance, may need more samples

### Regression Detection

Automatically detects performance regressions:
- **Threshold**: 5% performance degradation
- **Alerts**: Highlighted in reports
- **CI Integration**: Can fail builds on regression

## 🧪 Testing Strategy

### Continuous Integration

```yaml
# .github/workflows/benchmarks.yml
- name: Run Benchmarks
  run: |
    npm run bench:compare
    npm run report:generate

- name: Check for Regressions
  run: |
    node utils/analyzer.js --threshold 0.05
```

### Local Development

```bash
# Quick smoke test (reduced iterations)
BENCH_QUICK=1 npm run bench:crypto-compare

# Full benchmark suite (takes ~10-15 minutes)
npm run bench:compare

# Continuous monitoring
watch -n 300 'npm run bench:compare && npm run report:generate'
```

## 📖 API Documentation

### Statistics Module

```javascript
import stats from './utils/statistics.js';

// Basic statistics
const avg = stats.mean(values);
const med = stats.median(values);
const p95 = stats.percentile(values, 95);

// Performance analysis
const speedup = stats.speedup(baselineTime, optimizedTime);
const improvement = stats.improvementPercentage(baseline, optimized);

// Regression detection
const regression = stats.detectRegression(baseline, current, 0.05);
```

### Analyzer Module

```javascript
import { analyzeSamples, compareBenchmarks } from './utils/analyzer.js';

// Analyze raw samples
const analysis = analyzeSamples(samples);
// Returns: { mean, median, stdDev, percentiles, confidenceInterval }

// Compare two benchmarks
const comparison = compareBenchmarks(baseline, current);
// Returns: { speedup, improvement, regression, verdict }
```

## 🐛 Troubleshooting

### Native Bindings Not Found

```bash
# Rebuild native bindings
cd ../qudag/qudag-napi
npm run build

# Verify bindings
node -e "require('@daa/qudag-native')"
```

### WASM Build Errors

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Rebuild WASM
cd ../qudag/qudag-wasm
wasm-pack build --target nodejs --out-dir pkg-node
```

### Chart Generation Errors

```bash
# Install canvas dependencies (Ubuntu/Debian)
sudo apt-get install build-essential libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev

# Install canvas dependencies (macOS)
brew install pkg-config cairo pango libpng jpeg giflib librsvg
```

### Missing Reports Directory

```bash
mkdir -p reports
```

## 🤝 Contributing

### Adding New Benchmarks

1. **Rust benchmarks**: Add to `crypto/`, `orchestrator/`, or `prime/`
2. **JavaScript benchmarks**: Create corresponding `.js` files
3. **Update Cargo.toml**: Add new `[[bench]]` entry
4. **Update package.json**: Add new script

### Benchmark Best Practices

- Use `black_box()` to prevent optimization
- Warm up before measuring
- Run sufficient iterations (100+ samples)
- Test multiple data sizes
- Document expected performance

## 📚 References

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [Benchmark.js Documentation](https://benchmarkjs.com/docs)
- [NAPI-rs Performance Guide](https://napi.rs/docs/concepts/performance)
- [WASM Performance Best Practices](https://web.dev/webassembly-best-practices/)

## 📄 License

MIT OR Apache-2.0

---

**Generated by DAA Benchmarking Suite**
For more information, visit [github.com/ruvnet/daa](https://github.com/ruvnet/daa)
