# Testing Guide - CRISPR-Cas13 Pipeline

## 📚 Overview

This guide provides comprehensive instructions for running, writing, and maintaining tests for the CRISPR-Cas13 bioinformatics pipeline.

---

## 🧪 Test Structure

### Test Organization

```
crispr-cas13-pipeline/
├── crates/
│   ├── data-models/
│   │   └── tests/
│   │       └── models_test.rs         # Unit tests for data structures
│   ├── alignment-engine/
│   │   └── tests/
│   │       └── alignment_test.rs      # Unit tests for alignment algorithms
│   ├── offtarget-predictor/
│   │   └── tests/
│   │       └── prediction_test.rs     # Unit tests for ML predictions
│   ├── immune-analyzer/
│   │   └── tests/
│   │       └── analysis_test.rs       # Unit tests for DE analysis
│   └── api-service/
│       └── tests/
│           └── api_test.rs            # Unit tests for API endpoints
├── tests/
│   ├── integration_test.rs            # End-to-end integration tests
│   ├── property_tests.rs              # Property-based tests with proptest
│   ├── load_testing_k6.js             # k6 load testing script
│   └── load_testing_locust.py         # Locust load testing script
└── benches/
    ├── alignment_benchmark.rs          # Criterion benchmarks for alignment
    ├── offtarget_prediction_benchmark.rs  # Benchmarks for predictions
    ├── immune_analysis_benchmark.rs    # Benchmarks for DE analysis
    └── api_benchmark.rs               # Benchmarks for API performance
```

---

## 🚀 Running Tests

### Unit Tests

Run all unit tests:
```bash
cargo test --lib --all-features
```

Run tests for a specific crate:
```bash
cargo test -p data-models
cargo test -p alignment-engine
cargo test -p offtarget-predictor
```

Run with output:
```bash
cargo test -- --nocapture
```

Run specific test by name:
```bash
cargo test test_perfect_alignment
```

### Integration Tests

Run all integration tests:
```bash
cargo test --test '*' --all-features
```

Run specific integration test file:
```bash
cargo test --test integration_test
```

**Note:** Integration tests require running services (PostgreSQL, MongoDB, Redis, Kafka). Use Docker Compose:
```bash
docker-compose up -d
cargo test --test integration_test
docker-compose down
```

### Property-Based Tests

Run property tests (uses proptest):
```bash
cargo test --test property_tests
```

Increase test cases for more thorough testing:
```bash
PROPTEST_CASES=10000 cargo test --test property_tests
```

### Doc Tests

Run documentation examples:
```bash
cargo test --doc --all-features
```

---

## 📊 Benchmarks

### Running Benchmarks

Run all benchmarks:
```bash
cargo bench --all-features
```

Run specific benchmark suite:
```bash
cargo bench --bench alignment_benchmark
cargo bench --bench offtarget_prediction_benchmark
cargo bench --bench immune_analysis_benchmark
cargo bench --bench api_benchmark
```

Save baseline for comparison:
```bash
cargo bench --all-features -- --save-baseline main
```

Compare with baseline:
```bash
cargo bench --all-features -- --baseline main
```

### Viewing Benchmark Results

Benchmark reports are generated in `target/criterion/`:
```bash
# Open HTML report in browser
open target/criterion/report/index.html

# View text summary
cat target/criterion/*/report/index.txt
```

---

## 🌐 Load Testing

### k6 Load Testing

Install k6:
```bash
# macOS
brew install k6

# Ubuntu/Debian
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6
```

Run load test:
```bash
# Basic test
k6 run tests/load_testing_k6.js

# Custom configuration
k6 run --vus 100 --duration 5m tests/load_testing_k6.js

# With environment variables
k6 run --vus 50 --duration 2m \
  --env API_URL=http://localhost:8080 \
  --env AUTH_TOKEN=your-token-here \
  tests/load_testing_k6.js

# Output results to file
k6 run tests/load_testing_k6.js --out json=load-test-results.json
```

### Locust Load Testing

Install Locust:
```bash
pip install locust
```

Run load test:
```bash
# Web UI mode (recommended)
locust -f tests/load_testing_locust.py --host=http://localhost:8080

# Then open browser to http://localhost:8089

# Headless mode
locust -f tests/load_testing_locust.py \
  --host=http://localhost:8080 \
  --users 100 \
  --spawn-rate 10 \
  --run-time 5m \
  --headless

# With output files
locust -f tests/load_testing_locust.py \
  --host=http://localhost:8080 \
  --users 100 \
  --spawn-rate 10 \
  --run-time 5m \
  --headless \
  --html locust-report.html \
  --csv locust-results
```

---

## 📈 Code Coverage

### Generating Coverage Reports

Install tarpaulin:
```bash
cargo install cargo-tarpaulin
```

Generate coverage report:
```bash
# Basic coverage
cargo tarpaulin --all-features --workspace

# Generate HTML report
cargo tarpaulin --all-features --workspace --out Html

# Generate multiple formats
cargo tarpaulin --all-features --workspace \
  --out Html --out Xml --out Lcov

# Exclude integration tests
cargo tarpaulin --lib --all-features --workspace

# Upload to Codecov
cargo tarpaulin --all-features --workspace --out Xml
bash <(curl -s https://codecov.io/bash)
```

View coverage report:
```bash
open tarpaulin-report.html
```

### Coverage Targets

Our coverage targets:
- **Statements:** >80%
- **Branches:** >75%
- **Functions:** >80%
- **Lines:** >80%

Check if coverage meets threshold:
```bash
COVERAGE=$(cargo tarpaulin --all-features --workspace --out Stdout | grep -oP '\d+\.\d+(?=%)' | head -1)
if (( $(echo "$COVERAGE < 80" | bc -l) )); then
  echo "Coverage $COVERAGE% is below 80% threshold"
  exit 1
fi
```

---

## ✍️ Writing Tests

### Unit Test Template

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_name() {
        // Arrange: Set up test data
        let input = create_test_input();

        // Act: Execute the code under test
        let result = function_to_test(input);

        // Assert: Verify the results
        assert_eq!(result, expected_value);
    }

    #[test]
    #[should_panic(expected = "error message")]
    fn test_error_condition() {
        // Test that function panics with expected message
        panic_inducing_function();
    }
}
```

### Integration Test Template

```rust
#[tokio::test]
async fn test_end_to_end_workflow() {
    // Setup: Initialize test environment
    let test_db = setup_test_database().await;
    let app = create_test_app(test_db.clone()).await;

    // Execute: Run the workflow
    let response = app
        .post("/api/v1/jobs")
        .json(&test_payload)
        .send()
        .await
        .unwrap();

    // Verify: Check results
    assert_eq!(response.status(), 201);
    let job_id = response.json::<Job>().await.unwrap().id;

    // Cleanup: Tear down test environment
    test_db.cleanup().await;
}
```

### Property-Based Test Template

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_property_holds(
        input in arbitrary_input_strategy()
    ) {
        // Property: Define what should always be true
        let result = function_to_test(input);

        // Assert the property holds
        prop_assert!(property_condition(result));
    }
}
```

### Benchmark Template

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_function(c: &mut Criterion) {
    let test_data = generate_test_data();

    c.bench_function("operation_name", |b| {
        b.iter(|| {
            // Code to benchmark
            black_box(function_to_benchmark(black_box(&test_data)));
        });
    });
}

criterion_group!(benches, benchmark_function);
criterion_main!(benches);
```

---

## 🎯 Test Best Practices

### General Guidelines

1. **Test Naming:** Use descriptive names that explain what is being tested
   - ✅ `test_alignment_handles_single_mismatch`
   - ❌ `test_1`

2. **Test Independence:** Each test should be independent and not rely on others
   - Use setup/teardown for shared resources
   - Clean up after tests

3. **Test Data:** Use realistic test data
   - Avoid magic numbers
   - Use constants or test data builders

4. **Assertions:** Use specific assertions
   - ✅ `assert_eq!(result, expected)`
   - ❌ `assert!(result == expected)`

5. **Test Coverage:** Aim for comprehensive coverage
   - Happy path
   - Edge cases
   - Error conditions
   - Boundary values

### Rust-Specific Best Practices

1. **Use `#[cfg(test)]` for test modules:**
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;
       // tests here
   }
   ```

2. **Use test helpers:**
   ```rust
   #[cfg(test)]
   mod test_helpers {
       pub fn create_test_data() -> TestData {
           // ...
       }
   }
   ```

3. **Mock external dependencies:**
   ```rust
   use mockall::automock;

   #[automock]
   trait Database {
       fn query(&self, sql: &str) -> Result<Vec<Row>>;
   }
   ```

4. **Use async tests for async code:**
   ```rust
   #[tokio::test]
   async fn test_async_function() {
       let result = async_function().await;
       assert!(result.is_ok());
   }
   ```

---

## 🔍 Debugging Tests

### Running Tests with Debug Output

```bash
# Show all output
cargo test -- --nocapture

# Show test names as they run
cargo test -- --nocapture --test-threads=1

# Run with backtrace
RUST_BACKTRACE=1 cargo test

# Run with full backtrace
RUST_BACKTRACE=full cargo test
```

### Using println! in Tests

```rust
#[test]
fn test_with_debug_output() {
    let value = compute_value();
    println!("Computed value: {:?}", value);
    assert_eq!(value, expected);
}
```

Run with `--nocapture` to see output.

### Using dbg! Macro

```rust
#[test]
fn test_with_dbg() {
    let intermediate = dbg!(compute_intermediate());
    let result = dbg!(process(intermediate));
    assert_eq!(result, expected);
}
```

---

## 🤖 CI/CD Integration

### GitHub Actions

Tests run automatically on:
- Push to `main` or `develop`
- Pull requests

See [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) for configuration.

### Running CI Locally

Simulate CI environment:
```bash
# Install act (GitHub Actions local runner)
brew install act  # macOS
# or
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run CI workflow
act push
```

---

## 📝 Test Documentation

### Documenting Tests

```rust
/// Tests that the alignment algorithm correctly handles single mismatches.
///
/// # Test Case
/// - Input: Two sequences differing by one base
/// - Expected: Alignment identifies mismatch position correctly
/// - Verify: CIGAR string shows single substitution
#[test]
fn test_single_mismatch_alignment() {
    // test implementation
}
```

### Test Plans

Major features should have test plans documenting:
- Test objectives
- Test cases
- Expected results
- Prerequisites
- Execution steps

See [`docs/test-plans/`](../docs/test-plans/) for examples.

---

## 🛠️ Troubleshooting

### Common Issues

**Issue:** Tests fail in CI but pass locally
- **Solution:** Check for environment differences (dependencies, OS, etc.)

**Issue:** Flaky tests (intermittent failures)
- **Solution:** Check for timing dependencies, race conditions, or shared state

**Issue:** Slow tests
- **Solution:** Profile tests, parallelize where possible, use mocking

**Issue:** Out of memory during tests
- **Solution:** Run tests sequentially (`--test-threads=1`), increase system resources

### Getting Help

- Check test output for error messages
- Run with `RUST_BACKTRACE=1` for stack traces
- Review test logs in CI artifacts
- Ask in team chat or GitHub discussions

---

## 📚 Additional Resources

- [Rust Testing Book](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [Proptest Book](https://proptest-rs.github.io/proptest/)
- [k6 Documentation](https://k6.io/docs/)
- [Locust Documentation](https://docs.locust.io/)

---

**Last Updated:** 2024-10-12
**Version:** v1.0.0

For questions or contributions, see [CONTRIBUTING.md](../CONTRIBUTING.md)
