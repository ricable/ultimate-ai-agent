# Neuro-Divergent Integration Tests

This directory contains comprehensive integration tests for the neuro-divergent neural forecasting library.

## Test Categories

### 1. Workflow Tests (`workflow_tests.rs`)
End-to-end tests for complete forecasting workflows including:
- Loading CSV data
- Creating multiple models (LSTM, NBEATS, MLP)
- Model fitting
- Forecast generation
- Accuracy evaluation
- Large dataset handling
- Model persistence
- Error handling

### 2. Ensemble Tests (`ensemble_tests.rs`)
Tests for multi-model ensemble scenarios:
- Different model types working together
- Model averaging and weighting
- Ensemble prediction aggregation
- Performance comparisons
- Robustness to outliers
- Dynamic model selection
- Probabilistic ensembles

### 3. Cross-Validation Tests (`cross_validation_tests.rs`)
Tests for cross-validation workflows:
- Time series cross-validation
- Model selection via CV
- Hyperparameter tuning
- Different window strategies (expanding, sliding)
- Seasonal splits
- Stability testing

### 4. Persistence Tests (`persistence_tests.rs`)
Tests for model persistence and loading:
- Saving and loading trained models
- Model state serialization
- Cross-version compatibility
- Incremental learning from saved states
- Checkpoint management
- Model compression
- Export to different formats

### 5. Preprocessing Tests (`preprocessing_tests.rs`)
Tests for data preprocessing pipelines:
- Data scaling and normalization
- Feature engineering
- Missing value handling
- Outlier detection and treatment
- Data validation
- Exogenous variable preprocessing

### 6. API Compatibility Tests (`api_compatibility_tests.rs`)
Tests ensuring compatibility with Python NeuralForecast:
- Replicating Python examples
- Identical API surface
- All configuration options
- Output format validation
- Cross-validation compatibility
- Save/load functionality

## Running Tests

### Run all integration tests:
```bash
cargo test --test integration
```

### Run specific test module:
```bash
cargo test --test integration workflow_tests
```

### Run with verbose output:
```bash
cargo test --test integration -- --nocapture
```

### Run ignored (slow) tests:
```bash
cargo test --test integration -- --ignored
```

### Run with specific features:
```bash
cargo test --test integration --features "gpu,async"
```

## Performance Benchmarks

Some tests include performance benchmarks that are marked with `#[ignore]`. These can be run explicitly:

```bash
cargo test --test integration bench_ -- --ignored --nocapture
```

## Test Data

Tests generate synthetic data to ensure reproducibility. The data generation functions create:
- Multiple time series with different patterns
- Trend and seasonality components
- Exogenous variables
- Missing values and outliers (for preprocessing tests)

## Writing New Tests

When adding new integration tests:

1. Follow the existing naming convention
2. Use descriptive test names that explain what is being tested
3. Add appropriate documentation
4. Mark slow tests with `#[ignore]`
5. Clean up temporary files/directories
6. Use `Result<(), Box<dyn std::error::Error>>` for better error handling

## Test Coverage Goals

Our integration tests aim to cover:
- ✅ Basic workflows (single and multi-model)
- ✅ Advanced features (exogenous variables, probabilistic forecasting)
- ✅ Performance at scale (large datasets)
- ✅ Error conditions and edge cases
- ✅ API compatibility with Python NeuralForecast
- ✅ Model persistence and recovery
- ✅ Cross-validation and hyperparameter tuning
- ✅ Data preprocessing pipelines
- ✅ Ensemble methods and strategies

## Known Issues

- GPU tests require CUDA/ROCm setup
- Some probabilistic tests may have non-deterministic results
- Large dataset tests require significant memory (mark with `#[ignore]`)

## Contributing

When contributing new integration tests:
1. Ensure tests are deterministic (use fixed seeds)
2. Mock external dependencies when possible
3. Add comments explaining complex test scenarios
4. Update this README with new test categories