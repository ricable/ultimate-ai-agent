# Neuro-Divergent Stress Tests

This directory contains comprehensive stress tests designed to push the neuro-divergent library to its limits and ensure robustness under extreme conditions.

## Test Categories

### 1. Large Dataset Tests (`large_dataset_tests.rs`)
- Tests with millions of time series and data points
- Memory efficiency validation
- Streaming and batch processing capabilities
- Performance benchmarking

### 2. Edge Case Tests (`edge_case_tests.rs`)
- Empty datasets, single points, all NaN values
- Infinite values, duplicate timestamps, unsorted data
- Extreme time ranges and special characters
- Boundary value testing

### 3. Resource Limit Tests (`resource_limit_tests.rs`)
- Memory exhaustion scenarios
- File handle limits
- Thread pool saturation
- Stack overflow protection
- Disk space handling

### 4. Concurrent Usage Tests (`concurrent_usage_tests.rs`)
- Thread-safe operations
- Race condition detection
- Parallel model training
- Lock-free data structures
- Producer-consumer patterns

### 5. Failure Recovery Tests (`failure_recovery_tests.rs`)
- Training interruption and recovery
- Corrupted model file handling
- Network failure simulation
- Transaction rollback
- Graceful degradation

### 6. Fuzz Tests (`fuzz_tests.rs`)
- Property-based testing
- Random input generation
- API sequence fuzzing
- Edge case discovery

### 7. Memory Stress Tests (`memory_stress_tests.rs`)
- Memory leak detection
- Fragmentation analysis
- Allocation tracking
- Streaming memory patterns

## Running the Tests

### Quick Start
```bash
# Run all standard stress tests
cargo test --release --test '*stress*'

# Run with the helper script
./tests/run_stress_tests.sh

# Run specific category
./tests/run_stress_tests.sh --category large_dataset

# Include resource-intensive tests (be careful!)
./tests/run_stress_tests.sh --ignored
```

### Individual Test Commands
```bash
# Large dataset tests (standard)
cargo test --release --test large_dataset_tests

# Large dataset tests (including ignored/expensive tests)
cargo test --release --test large_dataset_tests -- --ignored

# Run specific test
cargo test --release --test edge_case_tests test_empty_dataset

# Run with output
cargo test --release --test concurrent_usage_tests -- --nocapture

# Run with single thread (for debugging)
cargo test --release --test resource_limit_tests -- --test-threads=1
```

## Resource Requirements

### Standard Tests
- Memory: 2-4 GB
- Disk: 1 GB free space
- Time: ~5 minutes

### Resource-Intensive Tests (--ignored)
- Memory: 8-16 GB recommended
- Disk: 10+ GB free space
- Time: 30-60 minutes
- CPU: Multi-core recommended

## Interpreting Results

### Success Indicators
- ✅ All tests pass without panics
- ✅ Memory usage stays within bounds
- ✅ Resources are properly cleaned up
- ✅ Error messages are descriptive

### Warning Signs
- ⚠️ Excessive memory growth
- ⚠️ Slow test execution
- ⚠️ Resource cleanup failures
- ⚠️ Unclear error messages

### Failure Analysis
1. Check the test output for specific error messages
2. Look for panics or unwrap failures
3. Monitor system resources during test execution
4. Review the stress test report for patterns

## Adding New Stress Tests

1. Choose the appropriate test file based on category
2. Follow the existing test patterns
3. Use property-based testing where applicable
4. Document resource requirements
5. Add to the stress test report

Example template:
```rust
#[test]
#[ignore] // If resource intensive
fn test_new_stress_scenario() {
    // Setup
    let config = StressConfig::default();
    
    // Execute stress scenario
    let result = stress_operation(config);
    
    // Verify robustness
    assert!(result.is_ok() || result.is_err()); // Should not panic
    assert!(resources_cleaned_up());
}
```

## Troubleshooting

### Out of Memory
- Reduce test data sizes
- Enable only one test at a time
- Check for memory leaks
- Use memory profiling tools

### Test Timeouts
- Increase timeout limits
- Run tests individually
- Check for deadlocks
- Profile performance bottlenecks

### Platform-Specific Issues
- File handle limits: Increase ulimits on Unix
- Stack size: Adjust thread stack size
- Path lengths: Use shorter paths on Windows

## Continuous Integration

Add to your CI pipeline:
```yaml
stress-tests:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v2
    - uses: actions-rs/toolchain@v1
    - name: Run stress tests
      run: |
        cd neuro-divergent
        cargo test --release --test '*stress*'
```

## See Also
- [Stress Test Report](../../docs/STRESS_TEST_REPORT.md) - Detailed analysis and findings
- [Contributing Guide](../../CONTRIBUTING.md) - How to contribute stress tests
- [Performance Guide](../../docs/PERFORMANCE.md) - Performance optimization tips