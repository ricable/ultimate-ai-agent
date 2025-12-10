# Test Suite - CRISPR-Cas13 Pipeline

## 🎯 Overview

Comprehensive test and benchmark suite for the CRISPR-Cas13 bioinformatics pipeline, ensuring code quality, performance, and reliability.

## 📊 Test Coverage Summary

| Component | Unit Tests | Integration Tests | Benchmarks | Coverage Target |
|-----------|------------|-------------------|------------|-----------------|
| data-models | ✅ | ✅ | - | >85% |
| alignment-engine | ✅ | ✅ | ✅ | >85% |
| offtarget-predictor | ✅ | ✅ | ✅ | >85% |
| immune-analyzer | ✅ | ✅ | ✅ | >85% |
| api-service | ✅ | ✅ | ✅ | >85% |
| processing-orchestrator | ✅ | ✅ | - | >80% |

## 🧪 Test Types

### 1. Unit Tests
Located in each crate's `tests/` directory:
- **data-models:** Serialization, validation, type conversions
- **alignment-engine:** Algorithm correctness, edge cases, CIGAR strings
- **offtarget-predictor:** ML inference, feature extraction, scoring
- **immune-analyzer:** Normalization, DE analysis, statistical tests
- **api-service:** Endpoint handlers, authentication, validation

### 2. Integration Tests
Located in workspace `tests/` directory:
- **End-to-end pipeline workflows**
- **Database integration** (PostgreSQL, MongoDB, Redis)
- **Message queue integration** (Kafka)
- **API integration** (REST + WebSocket)
- **Distributed system coordination**

### 3. Property-Based Tests
Using `proptest` for algorithmic correctness:
- Alignment score bounds and monotonicity
- Normalization sum invariants
- Statistical test properties
- Serialization roundtrip preservation

### 4. Performance Benchmarks
Using `Criterion.rs` for micro-benchmarks:
- **Alignment throughput:** Target >10K reads/sec
- **Off-target prediction:** Target >100K sites/sec
- **DE analysis:** Target <30 sec for 20K genes
- **API latency:** Target p95 <200ms

### 5. Load Tests
Realistic traffic simulation:
- **k6:** JavaScript-based load testing
- **Locust:** Python-based distributed load testing

## 🚀 Quick Start

### Run All Tests
```bash
# Unit + integration tests
cargo test --all-features

# With output
cargo test --all-features -- --nocapture

# Specific crate
cargo test -p alignment-engine
```

### Run Benchmarks
```bash
# All benchmarks
cargo bench --all-features

# Specific benchmark
cargo bench --bench alignment_benchmark

# Save baseline
cargo bench -- --save-baseline main
```

### Run Load Tests
```bash
# k6 load test
k6 run --vus 100 --duration 5m tests/load_testing_k6.js

# Locust load test (web UI)
locust -f tests/load_testing_locust.py --host=http://localhost:8080

# Locust headless
locust -f tests/load_testing_locust.py \
  --host=http://localhost:8080 \
  --users 100 \
  --spawn-rate 10 \
  --run-time 5m \
  --headless
```

### Generate Coverage Report
```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate report
cargo tarpaulin --all-features --workspace --out Html

# View report
open tarpaulin-report.html
```

## 📁 Test File Structure

```
tests/
├── README.md                          # This file
├── integration_test.rs                # End-to-end integration tests
├── property_tests.rs                  # Property-based tests (proptest)
├── load_testing_k6.js                 # k6 load testing script
└── load_testing_locust.py             # Locust load testing script

benches/
├── alignment_benchmark.rs             # Alignment throughput benchmarks
├── offtarget_prediction_benchmark.rs  # Off-target prediction benchmarks
├── immune_analysis_benchmark.rs       # Immune analysis benchmarks
└── api_benchmark.rs                   # API performance benchmarks

crates/*/tests/
├── models_test.rs                     # data-models unit tests
├── alignment_test.rs                  # alignment-engine unit tests
├── prediction_test.rs                 # offtarget-predictor unit tests
├── analysis_test.rs                   # immune-analyzer unit tests
└── api_test.rs                        # api-service unit tests
```

## 🎯 Performance Targets

| Metric | Target | Current Status |
|--------|--------|----------------|
| Alignment throughput | >10,000 reads/sec | 🔄 TBD |
| Off-target throughput | >100,000 sites/sec | 🔄 TBD |
| DE analysis (20K genes) | <30 seconds | 🔄 TBD |
| API latency (p95) | <200 ms | 🔄 TBD |
| API throughput | >1,000 req/sec | 🔄 TBD |
| End-to-end (1M reads) | <30 minutes | 🔄 TBD |
| Code coverage | >85% | 🔄 TBD |

## 🔧 Test Infrastructure

### Dependencies
```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
proptest = "1.5"
mockall = "0.13"
wiremock = "0.6"
tokio-test = "0.4"
```

### CI/CD
Tests run automatically on:
- ✅ Every push to `main`/`develop`
- ✅ Every pull request
- ✅ Nightly builds

See [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) for details.

### Test Services
Integration tests require:
- PostgreSQL 16
- MongoDB 7
- Redis 7
- Kafka 3.x

Use Docker Compose:
```bash
docker-compose up -d
cargo test --test integration_test
docker-compose down
```

## 📖 Documentation

- **[TESTING_GUIDE.md](../docs/TESTING_GUIDE.md)** - Comprehensive testing guide
- **[BENCHMARKS.md](../docs/BENCHMARKS.md)** - Performance benchmark results
- **[CI.yml](../.github/workflows/ci.yml)** - CI/CD configuration

## ✍️ Writing Tests

### Unit Test Template
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature() {
        // Arrange
        let input = setup_test_data();

        // Act
        let result = function_under_test(input);

        // Assert
        assert_eq!(result, expected);
    }
}
```

### Integration Test Template
```rust
#[tokio::test]
async fn test_workflow() {
    // Setup
    let app = setup_test_app().await;

    // Execute
    let response = app.request().send().await;

    // Verify
    assert_eq!(response.status(), 200);

    // Cleanup
    teardown().await;
}
```

### Benchmark Template
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark(c: &mut Criterion) {
    c.bench_function("operation", |b| {
        b.iter(|| black_box(function_to_benchmark()));
    });
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
```

## 🐛 Debugging Tests

```bash
# Run with debug output
cargo test -- --nocapture

# Run single test
cargo test test_specific_case

# Run with backtrace
RUST_BACKTRACE=1 cargo test

# Run tests serially (avoid race conditions)
cargo test -- --test-threads=1
```

## 📊 Viewing Results

### Test Reports
```bash
# Run tests with JSON output
cargo test -- -Z unstable-options --format json

# Generate HTML test report
cargo install cargo-test-report
cargo test-report
```

### Benchmark Reports
```bash
# View Criterion HTML reports
open target/criterion/report/index.html

# Compare with baseline
cargo bench -- --baseline main
```

### Load Test Reports
```bash
# k6 results
k6 run --out json=results.json tests/load_testing_k6.js

# Locust results
locust -f tests/load_testing_locust.py \
  --headless \
  --html report.html \
  --csv results
```

## 🔍 Test Best Practices

1. **Independence:** Tests should not depend on each other
2. **Isolation:** Use mocks/stubs for external dependencies
3. **Speed:** Unit tests should be fast (<100ms each)
4. **Readability:** Use descriptive test names
5. **Coverage:** Aim for >85% code coverage
6. **Determinism:** Tests should produce consistent results
7. **Cleanup:** Always clean up resources (files, connections)

## 🚨 Common Issues

### Issue: Tests fail in CI but pass locally
**Solution:** Check for environment differences, timing issues, or missing test services.

### Issue: Flaky tests
**Solution:** Look for race conditions, timing dependencies, or shared state. Use `cargo test -- --test-threads=1` to isolate.

### Issue: Slow test suite
**Solution:** Profile tests, parallelize where possible, use test fixtures efficiently.

### Issue: Out of memory during tests
**Solution:** Run tests sequentially or increase system resources.

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/your-org/crispr-cas13-pipeline/issues)
- **Discussions:** [GitHub Discussions](https://github.com/your-org/crispr-cas13-pipeline/discussions)
- **Documentation:** [docs/](../docs/)

## 📝 Contributing

When adding new features:
1. ✅ Write unit tests (>85% coverage)
2. ✅ Add integration tests for workflows
3. ✅ Add benchmarks for performance-critical code
4. ✅ Update documentation
5. ✅ Ensure CI passes

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.

---

**Last Updated:** 2024-10-12
**Test Suite Version:** v1.0.0
**Maintainers:** Testing & QA Team

🎯 **Goal:** Comprehensive testing ensures reliability, performance, and maintainability of the CRISPR-Cas13 pipeline.
