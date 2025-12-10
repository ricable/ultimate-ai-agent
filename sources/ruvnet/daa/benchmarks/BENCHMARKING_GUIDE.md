# DAA Benchmarking Guide

## Overview

This guide provides detailed information on interpreting benchmark results, understanding performance metrics, and making informed decisions about native vs WASM implementations.

## Table of Contents

1. [Understanding Benchmark Metrics](#understanding-benchmark-metrics)
2. [Interpreting Results](#interpreting-results)
3. [Performance Goals and Targets](#performance-goals-and-targets)
4. [Common Performance Patterns](#common-performance-patterns)
5. [Optimization Strategies](#optimization-strategies)
6. [When to Use Native vs WASM](#when-to-use-native-vs-wasm)
7. [Regression Analysis](#regression-analysis)
8. [Best Practices](#best-practices)

## Understanding Benchmark Metrics

### Primary Metrics

#### 1. Throughput (ops/sec)
- **What it measures**: Number of operations completed per second
- **Higher is better**: More operations = better performance
- **Use case**: Sustained workload performance

```
Example: 1,500 ops/sec means 1,500 operations completed every second
```

#### 2. Latency (ms)
- **What it measures**: Time to complete a single operation
- **Lower is better**: Less time = faster response
- **Use case**: Single operation responsiveness

```
Example: 2.5ms means each operation takes 2.5 milliseconds
```

#### 3. Speedup (x)
- **What it measures**: Performance ratio between implementations
- **Higher is better**: 3x means 3 times faster
- **Use case**: Comparing native vs WASM

```
Example: 2.8x speedup means native is 2.8 times faster than WASM
```

### Statistical Metrics

#### Mean (Average)
- Average performance across all samples
- Good for overall performance assessment
- Can be skewed by outliers

#### Median (p50)
- Middle value when samples are sorted
- More resistant to outliers than mean
- Better for typical case performance

#### Percentiles (p95, p99)
- p95: 95% of operations complete within this time
- p99: 99% of operations complete within this time
- Critical for understanding worst-case performance

```
Example Performance Distribution:
- Mean: 2.5ms
- Median (p50): 2.3ms (typical case)
- p95: 4.2ms (95% complete within)
- p99: 8.1ms (worst 1% take longer)
```

#### Standard Deviation (σ)
- Measure of variability in performance
- Lower is better (more consistent)
- High variance may indicate unstable performance

#### Relative Standard Error (RSE)
- Percentage variation from mean
- < 1%: Very consistent
- < 5%: Acceptable consistency
- > 10%: High variance, may need investigation

## Interpreting Results

### Speedup Analysis

| Speedup | Rating | Recommendation |
|---------|--------|----------------|
| 5x+ | Outstanding | Always use native for this operation |
| 3-5x | Excellent | Strongly recommend native |
| 2-3x | Good | Recommend native for hot paths |
| 1.5-2x | Modest | Consider native for CPU-intensive tasks |
| <1.5x | Marginal | WASM may be sufficient, prioritize portability |

### Latency Analysis

#### Crypto Operations

| Operation | Excellent | Good | Acceptable | Needs Optimization |
|-----------|-----------|------|------------|--------------------|
| ML-KEM Keygen | <2ms | <5ms | <10ms | >10ms |
| ML-KEM Encap | <1ms | <3ms | <5ms | >5ms |
| ML-DSA Sign | <2ms | <5ms | <10ms | >10ms |
| BLAKE3 (1MB) | <2ms | <5ms | <10ms | >10ms |

#### Orchestrator Operations

| Operation | Excellent | Good | Acceptable | Needs Optimization |
|-----------|-----------|------|------------|--------------------|
| Workflow Create | <5ms | <10ms | <20ms | >20ms |
| MRAP Loop | <50ms | <100ms | <200ms | >200ms |
| Event Process (1K) | <500ms | <1s | <2s | >2s |

#### Prime ML Operations

| Operation | Excellent | Good | Acceptable | Needs Optimization |
|-----------|-----------|------|------------|--------------------|
| Gradient Agg (10) | <30ms | <50ms | <100ms | >100ms |
| Model Update (100K) | <5ms | <10ms | <20ms | >20ms |

### Consistency Analysis

#### RSE (Relative Standard Error) Guidelines

- **< 1%**: Very stable, production-ready
- **1-3%**: Stable, acceptable for most use cases
- **3-5%**: Moderate variance, monitor for regressions
- **5-10%**: High variance, investigate causes
- **> 10%**: Unstable, requires optimization

### Scalability Analysis

#### Evaluating Distributed Training Scalability

**Ideal Scalability**: Linear scaling with node count
- 2 nodes: 2x throughput
- 4 nodes: 4x throughput
- 8 nodes: 8x throughput

**Real-World Scalability**: Account for coordination overhead

| Efficiency | Rating | Example |
|------------|--------|---------|
| >90% | Excellent | 8 nodes achieve 7.2x+ speedup |
| 70-90% | Good | 8 nodes achieve 5.6-7.2x speedup |
| 50-70% | Acceptable | 8 nodes achieve 4.0-5.6x speedup |
| <50% | Poor | 8 nodes achieve <4x speedup |

```
Scalability Efficiency = (Actual Speedup / Ideal Speedup) × 100%

Example:
- 10 nodes, ideal speedup: 10x
- Actual throughput improvement: 7x
- Efficiency: (7 / 10) × 100% = 70%
```

## Performance Goals and Targets

### Crypto Operations (Native)

Based on NIST PQC requirements and real-world usage:

| Operation | Target | Stretch Goal | Notes |
|-----------|--------|--------------|-------|
| ML-KEM-768 Keygen | <2ms | <1ms | One-time operation |
| ML-KEM-768 Encap | <1.5ms | <1ms | Per session |
| ML-KEM-768 Decap | <1.5ms | <1ms | Per session |
| ML-DSA Sign | <2ms | <1ms | Per transaction |
| ML-DSA Verify | <1.5ms | <1ms | Per transaction |
| BLAKE3 (1MB) | <3ms | <2ms | Hashing throughput |

### Orchestrator Operations

| Operation | Target | Stretch Goal | Notes |
|-----------|--------|--------------|-------|
| Simple Workflow | <5ms | <2ms | Low complexity |
| Complex Workflow | <20ms | <10ms | High complexity |
| MRAP Loop | <100ms | <50ms | Full cycle |
| Event (1000/s) | <1s | <500ms | Throughput |

### Prime ML Operations

| Operation | Target | Stretch Goal | Notes |
|-----------|--------|--------------|-------|
| Gradient Agg (10 nodes) | <50ms | <30ms | Small cluster |
| Gradient Agg (50 nodes) | <200ms | <100ms | Large cluster |
| Model Update (100K) | <10ms | <5ms | Mid-size model |
| Model Update (1M) | <50ms | <30ms | Large model |

## Common Performance Patterns

### 1. Fixed Cost Operations
- **Pattern**: Constant time regardless of data size
- **Example**: Key generation, signature verification setup
- **Optimization**: Focus on algorithm efficiency

### 2. Linear Scaling Operations
- **Pattern**: Performance scales linearly with data size
- **Example**: Hashing, encryption, model updates
- **Optimization**: Batch processing, SIMD, parallelization

### 3. Quadratic Operations
- **Pattern**: Performance scales with O(n²)
- **Example**: Naive gradient aggregation, some sorts
- **Optimization**: Better algorithms, early termination

### 4. Communication-Bound Operations
- **Pattern**: Performance limited by network/IPC
- **Example**: Distributed training coordination
- **Optimization**: Reduce round trips, compression, batching

## Optimization Strategies

### Native (NAPI-rs) Optimizations

#### 1. Zero-Copy Operations
```rust
#[napi]
pub fn process_buffer(buffer: Buffer) -> Result<Buffer> {
    // Use buffer reference directly, no copy
    let slice = buffer.as_ref();
    // Process in-place when possible
    Ok(buffer)
}
```

#### 2. Parallel Processing
```rust
use rayon::prelude::*;

#[napi]
pub fn parallel_aggregate(gradients: Vec<Vec<f32>>) -> Vec<f32> {
    gradients.par_iter()
        .fold(|| vec![0.0; size], |acc, grad| {
            // Parallel aggregation
        })
        .reduce(|| vec![0.0; size], |a, b| {
            // Combine results
        })
}
```

#### 3. Batch Processing
```rust
#[napi]
pub fn batch_sign(messages: Vec<String>) -> Vec<Signature> {
    messages.iter()
        .map(|msg| sign(msg))
        .collect()
}
```

### WASM Optimizations

#### 1. TypedArrays for Performance
```javascript
// Use TypedArrays instead of regular arrays
const data = new Float32Array(1000000);
const view = new DataView(data.buffer);
```

#### 2. Memory Pooling
```javascript
// Reuse buffers instead of allocating
const pool = new BufferPool();
const buffer = pool.acquire(size);
// ... use buffer ...
pool.release(buffer);
```

#### 3. Minimize JS ↔ WASM Crossings
```javascript
// ❌ Bad: Multiple crossings
for (let i = 0; i < 1000; i++) {
    wasm.process_item(i);
}

// ✅ Good: Single crossing
wasm.process_batch(items);
```

### General Optimizations

#### 1. Algorithm Selection
- Choose appropriate algorithms for data size
- Consider time/space tradeoffs
- Profile before optimizing

#### 2. Caching
```javascript
// Cache expensive computations
const cache = new Map();
function expensiveOp(key) {
    if (cache.has(key)) return cache.get(key);
    const result = doExpensiveWork(key);
    cache.set(key, result);
    return result;
}
```

#### 3. Lazy Initialization
```javascript
// Delay expensive setup until needed
class Crypto {
    #mlkem = null;

    get mlkem() {
        if (!this.#mlkem) {
            this.#mlkem = new MlKem768();
        }
        return this.#mlkem;
    }
}
```

## When to Use Native vs WASM

### Use Native (NAPI-rs) When:

✅ **Performance is Critical**
- High-frequency operations (>1000 ops/sec)
- Latency-sensitive applications (<10ms requirement)
- CPU-intensive workloads (crypto, ML training)

✅ **Node.js Environment**
- Server-side applications
- CLI tools
- Build tools

✅ **Native Features Needed**
- Multi-threading with full OS support
- Direct file system access
- Native library integration

✅ **Observed Speedup >2x**
- Benchmark shows significant performance gain
- Justifies additional complexity

### Use WASM When:

✅ **Cross-Platform Portability**
- Browser compatibility required
- No build tools on target system
- Wide platform support needed

✅ **Security Sandboxing**
- Untrusted code execution
- Isolated environment required
- Memory safety critical

✅ **Easy Distribution**
- npm-only installation
- No native compilation needed
- Simplified CI/CD

✅ **Modest Performance Requirements**
- <100 ops/sec
- Latency >100ms acceptable
- Native speedup <1.5x

### Hybrid Approach

Recommended for most projects:

```javascript
// Auto-detect best implementation
import { detectPlatform, loadCrypto } from 'daa-sdk';

const platform = detectPlatform(); // 'native' or 'wasm'
const crypto = await loadCrypto();  // Loads best available

// Native in Node.js, WASM in browser
```

## Regression Analysis

### Detecting Regressions

**Threshold**: 5% performance degradation

```javascript
const baseline = 100; // ops/sec
const current = 93;   // ops/sec

const change = (current - baseline) / baseline;
// -0.07 = -7% (regression detected)
```

### Types of Regressions

#### 1. Algorithm Regression
- **Cause**: Less efficient algorithm introduced
- **Detection**: Large, consistent slowdown
- **Fix**: Revert to previous algorithm or optimize new one

#### 2. Memory Regression
- **Cause**: Memory leaks, excessive allocation
- **Detection**: Slowdown over time, increased GC
- **Fix**: Profile memory usage, fix leaks

#### 3. Concurrency Regression
- **Cause**: Lock contention, race conditions
- **Detection**: Performance degrades with load
- **Fix**: Reduce lock scope, use lock-free structures

#### 4. I/O Regression
- **Cause**: Increased I/O operations
- **Detection**: Higher latency variance
- **Fix**: Batch operations, reduce I/O

### CI/CD Integration

```yaml
# Fail build on >5% regression
- name: Check Regressions
  run: |
    npm run bench:compare
    node utils/analyzer.js --threshold 0.05 --fail-on-regression
```

## Best Practices

### 1. Consistent Environment
- Same hardware for benchmarks
- Isolated system (no other processes)
- Consistent OS and Node.js versions

### 2. Statistical Rigor
- Run sufficient samples (100+ recommended)
- Report confidence intervals
- Check for outliers

### 3. Realistic Workloads
- Use production-like data sizes
- Test with real-world patterns
- Include both best and worst cases

### 4. Continuous Monitoring
- Benchmark on every major change
- Track performance over time
- Set up regression alerts

### 5. Document Findings
- Record baseline performance
- Document optimization rationale
- Track performance goals

### 6. Context Matters
- Benchmark in target environment
- Consider cold vs warm cache
- Account for JIT warm-up

## Conclusion

Effective benchmarking requires:
1. **Understanding metrics**: Know what you're measuring
2. **Setting goals**: Define acceptable performance
3. **Analyzing results**: Interpret data correctly
4. **Making decisions**: Choose right implementation
5. **Continuous monitoring**: Catch regressions early

Use this guide to make informed performance decisions and optimize DAA implementations effectively.

## Additional Resources

- [Criterion.rs Book](https://bheisler.github.io/criterion.rs/book/)
- [Benchmark.js Best Practices](https://benchmarkjs.com/docs#best-practices)
- [NAPI-rs Performance Tips](https://napi.rs/docs/concepts/performance)
- [WASM Performance](https://web.dev/webassembly-performance/)

---

**Questions or Issues?**
Open an issue at [github.com/ruvnet/daa/issues](https://github.com/ruvnet/daa/issues)
