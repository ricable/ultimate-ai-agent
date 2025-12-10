# Climate Prediction Testing Guide

## Overview

This document describes the comprehensive testing strategy for the Climate Prediction system, ensuring 80%+ code coverage and production-ready quality.

## Test Structure

```
tests/
├── common/
│   └── test_helpers.rs          # Shared utilities and mocks
├── fixtures/
│   ├── sample_data.json         # Test data
│   └── mock_responses/          # Mock API responses
├── integration/
│   ├── data_ingestion_test.rs   # API client & data fetching tests
│   ├── model_inference_test.rs  # ML model tests
│   ├── api_endpoints_test.rs    # HTTP endpoint tests
│   └── end_to_end_test.rs       # Full workflow tests
├── property_tests.rs            # Property-based testing
benches/
└── performance_bench.rs         # Performance benchmarks
```

## Running Tests

### Quick Start

```bash
# Run all tests
cargo test

# Run with coverage
cargo tarpaulin --all-features --workspace

# Run specific test suite
cargo test --test data_ingestion_test

# Run benchmarks
cargo bench
```

### Comprehensive Test Script

```bash
# Run full test suite with coverage report
./scripts/run_tests.sh
```

This script runs:
- Unit tests
- Integration tests
- Property-based tests
- Code quality checks (fmt, clippy)
- Coverage analysis (80%+ threshold)
- Benchmark compilation

## Test Categories

### 1. Unit Tests

Located in source files (`src/*.rs`) using `#[cfg(test)]` modules.

**What they test:**
- Individual function behavior
- Edge cases and boundary conditions
- Error handling
- Data validation logic

**Example:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_validation() {
        assert!(validate_temperature(22.5).is_ok());
        assert!(validate_temperature(150.0).is_err());
    }
}
```

### 2. Integration Tests

Located in `tests/integration/*.rs`.

#### Data Ingestion Tests (`data_ingestion_test.rs`)

Tests API client behavior:
- ✅ Basic HTTP requests
- ✅ Retry logic on failures
- ✅ Rate limiting handling
- ✅ Authentication
- ✅ Data validation
- ✅ Concurrent requests
- ✅ Caching mechanisms
- ✅ Timeout handling

**Key Tests:**
- `test_api_client_retry_logic()` - Verifies exponential backoff
- `test_rate_limit_handling()` - Checks 429 response handling
- `test_concurrent_api_requests()` - Validates thread safety

#### Model Inference Tests (`model_inference_test.rs`)

Tests ML model behavior:
- ✅ Model loading
- ✅ Basic inference
- ✅ Prediction accuracy (RMSE validation)
- ✅ Confidence scoring
- ✅ Batch processing
- ✅ Edge cases (empty input, extreme values)
- ✅ Multi-parameter predictions
- ✅ Feature importance

**Key Tests:**
- `test_model_inference_accuracy()` - RMSE < 5°C threshold
- `test_model_confidence_scores()` - Confidence ∈ [0, 1]
- `test_batch_inference()` - Validates parallel predictions

#### API Endpoints Tests (`api_endpoints_test.rs`)

Tests HTTP API behavior:
- ✅ Health check endpoint
- ✅ Current weather endpoint
- ✅ Prediction endpoint
- ✅ Input validation (400 errors)
- ✅ Authentication (401 errors)
- ✅ Rate limiting (429 errors)
- ✅ CORS headers
- ✅ Pagination
- ✅ Error response format
- ✅ Content negotiation

**Key Tests:**
- `test_prediction_endpoint_invalid_location()` - Lat/lon validation
- `test_rate_limiting_endpoint()` - Quota enforcement
- `test_error_response_format()` - Consistent error structure

#### End-to-End Tests (`end_to_end_test.rs`)

Tests complete workflows:
- ✅ Full prediction pipeline (fetch → process → predict → return)
- ✅ Error recovery workflows
- ✅ High load scenarios (50+ concurrent requests)
- ✅ Data pipeline integrity
- ✅ Multi-location predictions
- ✅ Time series forecasting
- ✅ Cache effectiveness

**Key Tests:**
- `test_complete_prediction_workflow()` - 4-step pipeline validation
- `test_high_load_scenario()` - 50 concurrent requests
- `test_time_series_forecast()` - 24-hour forecast continuity

### 3. Property-Based Tests

Located in `tests/property_tests.rs` using `proptest`.

Tests invariants across random inputs:
- ✅ Temperature bounds (-100°C to 100°C)
- ✅ Humidity bounds (0% to 100%)
- ✅ Pressure bounds (800-1200 hPa)
- ✅ Non-negative wind speed
- ✅ Non-negative precipitation
- ✅ Latitude bounds (-90° to 90°)
- ✅ Longitude bounds (-180° to 180°)
- ✅ Celsius ↔ Fahrenheit conversion reversibility
- ✅ Data processing preserves array length
- ✅ Moving average within min/max bounds
- ✅ Prediction continuity (no huge jumps)
- ✅ Confidence scores ∈ [0, 1]

**Example:**
```rust
proptest! {
    #[test]
    fn test_temperature_bounds(temp in -100.0f64..100.0f64) {
        let validated = validate_temperature(temp);
        prop_assert!(validated.is_ok());
    }
}
```

### 4. Performance Benchmarks

Located in `benches/performance_bench.rs` using `criterion`.

Benchmarks:
- **Data Ingestion** - 100, 1K, 10K data points
- **Data Validation** - 100 vs 1000 records
- **Feature Extraction** - 24h, 168h (1 week), 720h (1 month)
- **Model Inference** - 24h vs 168h input
- **Batch Predictions** - 10, 50, 100 batch sizes
- **Data Aggregation** - Stats calculation, moving averages
- **Concurrent Operations** - 10 parallel predictions
- **Memory Efficiency** - Clone vs reference

**Run benchmarks:**
```bash
cargo bench --bench performance_bench
```

**View results:**
```
target/criterion/report/index.html
```

## Coverage Requirements

**Minimum Coverage: 80%**

Coverage includes:
- **Statements**: 80%+
- **Branches**: 75%+
- **Functions**: 80%+
- **Lines**: 80%+

### Measuring Coverage

```bash
# Generate HTML report
cargo tarpaulin --all-features --workspace --out Html --output-dir coverage

# Open report
open coverage/index.html  # macOS
xdg-open coverage/index.html  # Linux
```

### CI/CD Integration

GitHub Actions workflow (`.github/workflows/test.yml`):
- ✅ Runs on every push/PR
- ✅ Parallel test execution
- ✅ Coverage enforcement
- ✅ Benchmark compilation check
- ✅ Code quality (fmt, clippy)
- ✅ Uploads coverage to Codecov

## Mock Services

### MockHttpClient

Simulates HTTP responses for testing:

```rust
let mock_client = MockHttpClient::new();

mock_client.add_response(MockResponse {
    status: 200,
    body: r#"{"temperature": 22.5}"#.to_string(),
    headers: vec![],
}).await;

let response = mock_client.get_next_response().await;
```

### Test Fixtures

Pre-defined test data in `tests/fixtures/`:
- `sample_data.json` - Historical climate data
- `mock_responses/api_success.json` - Successful API response
- `mock_responses/api_rate_limit.json` - Rate limit response

## Best Practices

### 1. Test Independence

Each test should be independent and idempotent:

```rust
#[tokio::test]
async fn test_something() {
    let config = create_test_config();  // Fresh config
    let mock = MockHttpClient::new();   // Fresh mock

    // Test logic...
}
```

### 2. Clear Assertions

Use descriptive assertions:

```rust
assert!(
    prediction >= -100.0 && prediction <= 100.0,
    "Prediction {} outside valid temperature range",
    prediction
);
```

### 3. Test Data Builders

Use helper functions for consistent test data:

```rust
let data = generate_sample_climate_data(24);  // 24 hours
let config = create_test_config();
```

### 4. Edge Cases

Always test boundary conditions:

```rust
#[tokio::test]
async fn test_edge_cases() {
    // Empty input
    let empty = predict_temperature(&[]);
    assert!(empty.is_none());

    // Extreme values
    let extreme = predict_temperature(&[-89.2, 56.7]);
    assert!(extreme.is_some());
}
```

### 5. Performance Awareness

Ensure tests run quickly:

```rust
#[tokio::test]
async fn test_fast_operation() {
    let start = std::time::Instant::now();

    // Test logic...

    assert!(start.elapsed() < Duration::from_millis(100));
}
```

## ReasoningBank Integration

All tests integrate with ReasoningBank for coordination:

```bash
# Pre-task registration
npx claude-flow@alpha hooks pre-task --description "test suite implementation"

# Post-edit tracking
npx claude-flow@alpha hooks post-edit \
  --file "tests/integration/end_to_end_test.rs" \
  --memory-key "climate/tests/e2e"

# Session management
npx claude-flow@alpha hooks session-end --export-metrics true
```

## Troubleshooting

### Tests Hanging

```bash
# Set shorter timeout
cargo test -- --test-threads=1 --nocapture
```

### Coverage Not Generated

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Clean build
cargo clean
cargo tarpaulin --all-features --workspace
```

### Benchmark Failures

```bash
# Compile only (no run)
cargo bench --no-run

# Run specific benchmark
cargo bench --bench performance_bench -- data_ingestion
```

## Continuous Improvement

### Adding New Tests

1. Create test file in appropriate directory
2. Follow naming convention: `*_test.rs`
3. Use common test helpers from `tests/common/test_helpers.rs`
4. Add to `./scripts/run_tests.sh` if needed
5. Ensure coverage stays ≥80%

### Reviewing Test Results

```bash
# View test output
cargo test -- --nocapture

# View coverage report
open coverage/index.html

# View benchmark results
open target/criterion/report/index.html
```

## Summary

✅ **100+ test cases** across unit, integration, property, and E2E tests
✅ **80%+ code coverage** enforced via CI/CD
✅ **Performance benchmarks** for critical paths
✅ **Mock services** for external dependencies
✅ **Property-based testing** for invariants
✅ **CI/CD integration** with GitHub Actions
✅ **ReasoningBank hooks** for coordination

This comprehensive test suite ensures production-ready quality and reliability for the Climate Prediction system.
