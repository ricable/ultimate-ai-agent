# Troubleshooting Guide: Common Migration Issues and Solutions

This guide provides comprehensive solutions to common issues encountered during migration from Python NeuralForecast to Rust neuro-divergent.

## Table of Contents

1. [Installation and Setup Issues](#installation-and-setup-issues)
2. [Data Format and Conversion Problems](#data-format-and-conversion-problems)
3. [Model Migration Issues](#model-migration-issues)
4. [Performance and Memory Problems](#performance-and-memory-problems)
5. [API Compatibility Issues](#api-compatibility-issues)
6. [Deployment and Runtime Errors](#deployment-and-runtime-errors)
7. [Integration Problems](#integration-problems)
8. [Debugging Techniques](#debugging-techniques)

## Installation and Setup Issues

### Issue: Rust Compilation Errors

**Problem**: Build fails with linker errors or missing dependencies

**Symptoms**:
```
error: linking with `cc` failed: exit status: 1
= note: /usr/bin/ld: cannot find -lpq
= note: /usr/bin/ld: cannot find -lssl
```

**Solutions**:

1. **Install system dependencies**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libssl-dev

# CentOS/RHEL
sudo yum groupinstall -y "Development Tools"
sudo yum install -y openssl-devel

# macOS
brew install openssl
export OPENSSL_DIR=/usr/local/opt/openssl
```

2. **Configure linker**:
```bash
# Add to ~/.cargo/config.toml
[target.x86_64-unknown-linux-gnu]
linker = "clang"
rustflags = ["-C", "link-arg=-fuse-ld=lld"]
```

3. **Clear build cache**:
```bash
cargo clean
cargo build --release
```

### Issue: Memory Errors During Compilation

**Problem**: Compilation runs out of memory on large projects

**Symptoms**:
```
error: could not compile `neuro-divergent` due to previous error
killed (signal 9)
```

**Solutions**:

1. **Reduce parallel compilation**:
```bash
# Add to ~/.cargo/config.toml
[build]
jobs = 2  # Reduce from default

# Or set environment variable
export CARGO_BUILD_JOBS=2
```

2. **Enable incremental compilation**:
```bash
export CARGO_INCREMENTAL=1
```

3. **Use release mode with debug info**:
```toml
[profile.release]
debug = true
lto = "thin"  # Instead of "fat"
```

### Issue: Version Compatibility Problems

**Problem**: Incompatible Rust version or dependencies

**Symptoms**:
```
error[E0554]: `#![feature(generic_associated_types)]` is incomplete
```

**Solutions**:

1. **Update Rust version**:
```bash
rustup update stable
rustup default stable
```

2. **Check minimum supported Rust version (MSRV)**:
```toml
# In Cargo.toml
[package]
rust-version = "1.70.0"
```

3. **Update dependencies**:
```bash
cargo update
cargo build
```

## Data Format and Conversion Problems

### Issue: pandas to polars Conversion Failures

**Problem**: Data types don't convert correctly between pandas and polars

**Symptoms**:
```python
PolarsError: conversion from `object` to `str` failed
```

**Solutions**:

1. **Explicit type conversion**:
```python
import pandas as pd
import polars as pl

# Fix mixed types before conversion
df_pandas['unique_id'] = df_pandas['unique_id'].astype(str)
df_pandas['y'] = pd.to_numeric(df_pandas['y'], errors='coerce')
df_pandas['ds'] = pd.to_datetime(df_pandas['ds'])

# Convert to polars
df_polars = pl.from_pandas(df_pandas)
```

2. **Handle null values**:
```python
# Clean null values before conversion
df_pandas = df_pandas.dropna(subset=['unique_id', 'ds', 'y'])
df_polars = pl.from_pandas(df_pandas)
```

3. **Use schema specification**:
```rust
use polars::prelude::*;

let schema = Schema::from_iter([
    ("unique_id", DataType::String),
    ("ds", DataType::Datetime(TimeUnit::Nanoseconds, None)),
    ("y", DataType::Float64),
]);

let df = LazyFrame::scan_csv("data.csv", ScanArgsCSV {
    schema: Some(Arc::new(schema)),
    ..Default::default()
})?.collect()?;
```

### Issue: Date Parsing Errors

**Problem**: DateTime columns not parsing correctly

**Symptoms**:
```
PolarsError: could not parse `2023-01-01T00:00:00` as date
```

**Solutions**:

1. **Specify date format**:
```rust
let df = df.with_columns([
    col("ds").str().strptime(StrptimeOptions {
        format: Some("%Y-%m-%d %H:%M:%S".to_string()),
        strict: false,
        exact: true,
        cache: true,
    })
]);
```

2. **Handle multiple date formats**:
```python
# Python preprocessing
df['ds'] = pd.to_datetime(df['ds'], infer_datetime_format=True)

# Then convert to polars
df_polars = pl.from_pandas(df)
```

3. **Manual date parsing**:
```rust
let df = df.with_columns([
    col("ds").map(
        |s| {
            s.str()?
                .into_iter()
                .map(|opt_s| {
                    opt_s.and_then(|s| {
                        chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
                            .ok()
                            .map(|dt| dt.timestamp_nanos())
                    })
                })
                .collect::<Int64Chunked>()
                .into_series()
        },
        GetOutput::from_type(DataType::Int64),
    ).cast(DataType::Datetime(TimeUnit::Nanoseconds, None))
]);
```

### Issue: Large Dataset Memory Issues

**Problem**: Out of memory when processing large datasets

**Symptoms**:
```
thread 'main' panicked at 'called `Result::unwrap()` on an `Err` value: OutOfMemory'
```

**Solutions**:

1. **Use lazy evaluation**:
```rust
// Don't collect immediately
let df = LazyFrame::scan_csv("large_file.csv", Default::default())?
    .filter(col("y").gt(0))
    .select([col("unique_id"), col("ds"), col("y")]);

// Process in chunks
let chunks = df.slice(0, 10000).collect()?;
```

2. **Stream processing**:
```rust
use polars::prelude::*;

fn process_large_file(path: &str) -> PolarsResult<()> {
    let mut reader = CsvReader::from_path(path)?
        .infer_schema(Some(1000))
        .has_header(true)
        .batch_size(10000);
    
    while let Some(batch) = reader.next_batch()? {
        // Process each batch
        process_batch(batch)?;
    }
    
    Ok(())
}
```

## Model Migration Issues

### Issue: Model Parameter Incompatibilities

**Problem**: Python model parameters don't map directly to Rust

**Symptoms**:
```
Error: Unknown parameter 'early_stop_patience_steps' for LSTM model
```

**Solutions**:

1. **Use parameter mapping**:
```rust
// Create mapping function
fn map_python_params(py_params: &PythonParams) -> LSTMConfig {
    LSTMConfig {
        horizon: py_params.h,
        input_size: py_params.input_size,
        hidden_size: py_params.hidden_size,
        // Map Python parameter names to Rust
        early_stopping_patience: py_params.early_stop_patience_steps,
        validation_check_steps: py_params.val_check_steps,
        ..Default::default()
    }
}
```

2. **Use builder pattern with validation**:
```rust
let model = LSTM::builder()
    .horizon(12)
    .input_size(24)
    .hidden_size(128)
    .validate_and_build()?;  // Validates parameters
```

### Issue: Model Accuracy Differences

**Problem**: Rust model produces different results than Python

**Symptoms**:
```
Assertion failed: MAE difference > tolerance (0.1 > 0.01)
```

**Solutions**:

1. **Check random seeds**:
```rust
// Ensure deterministic behavior
let model = LSTM::builder()
    .random_seed(Some(42))
    .build()?;

// Set global random seed
use rand::SeedableRng;
let mut rng = rand::rngs::StdRng::seed_from_u64(42);
```

2. **Verify data preprocessing**:
```rust
// Ensure identical preprocessing
let df = df.with_columns([
    // Use same normalization as Python
    ((col("y") - lit(mean)) / lit(std)).alias("y_normalized")
]);
```

3. **Compare intermediate results**:
```rust
// Add debug logging
pub fn debug_model_state(model: &LSTM, data: &DataFrame) {
    println!("Model weights: {:?}", model.get_weights());
    println!("Input data shape: {:?}", data.shape());
    println!("Input data sample: {:?}", data.head(Some(5)));
}
```

### Issue: Model Loading/Saving Errors

**Problem**: Cannot load saved Python models in Rust

**Symptoms**:
```
Error: Cannot deserialize Python pickle format
```

**Solutions**:

1. **Export Python model to neutral format**:
```python
# Python: Save to ONNX or custom format
import torch

# Save model parameters
torch.save(model.state_dict(), 'model_params.pt')

# Save configuration
import json
with open('model_config.json', 'w') as f:
    json.dump(model.get_config(), f)
```

2. **Create conversion utility**:
```rust
// Rust: Load from neutral format
use serde_json;

pub fn load_from_python_export(params_path: &str, config_path: &str) -> Result<LSTM> {
    // Load configuration
    let config_str = std::fs::read_to_string(config_path)?;
    let config: ModelConfig = serde_json::from_str(&config_str)?;
    
    // Create model
    let mut model = LSTM::from_config(config)?;
    
    // Load parameters (implement parameter loading)
    model.load_parameters(params_path)?;
    
    Ok(model)
}
```

## Performance and Memory Problems

### Issue: Slower Than Expected Performance

**Problem**: Rust version not showing expected speedup

**Symptoms**:
```
Training time: 10 minutes (expected: 3-4 minutes)
```

**Solutions**:

1. **Check build configuration**:
```toml
# Ensure optimized build
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"
```

2. **Enable CPU-specific optimizations**:
```bash
# Set CPU target
export RUSTFLAGS="-C target-cpu=native"
cargo build --release
```

3. **Profile performance**:
```bash
# Install profiler
cargo install cargo-profiler

# Profile application
cargo profiler --release --bin neuro-divergent
```

4. **Check parallel processing**:
```rust
// Ensure parallel processing is enabled
use rayon::prelude::*;

// Set thread count
rayon::ThreadPoolBuilder::new()
    .num_threads(num_cpus::get())
    .build_global()
    .unwrap();
```

### Issue: Memory Leaks

**Problem**: Memory usage grows over time

**Symptoms**:
```
Memory usage: 1GB -> 5GB -> 10GB (growing)
```

**Solutions**:

1. **Use memory profiling**:
```bash
# Install valgrind
sudo apt-get install valgrind

# Run with memory checking
valgrind --tool=memcheck --leak-check=full ./target/release/neuro-divergent
```

2. **Check for reference cycles**:
```rust
// Use Weak references where appropriate
use std::rc::{Rc, Weak};
use std::sync::{Arc, Weak as SyncWeak};

// Avoid cycles
struct Parent {
    children: Vec<Rc<Child>>,
}

struct Child {
    parent: Weak<Parent>,  // Use Weak to avoid cycle
}
```

3. **Implement proper cleanup**:
```rust
impl Drop for Model {
    fn drop(&mut self) {
        // Clean up resources
        self.cleanup_gpu_memory();
        self.close_file_handles();
    }
}
```

## API Compatibility Issues

### Issue: Missing Python API Methods

**Problem**: Python methods not available in Rust API

**Symptoms**:
```
Error: method `cross_validation` not found for type `NeuralForecast`
```

**Solutions**:

1. **Check API mapping documentation**:
```rust
// Use correct Rust method name
let cv_results = nf.cross_validation(CrossValidationConfig {
    n_windows: 3,
    horizon: 12,
    ..Default::default()
})?;
```

2. **Implement wrapper functions**:
```rust
// Create Python-like API
impl NeuralForecast {
    pub fn cross_validation_python_compat(
        &mut self,
        df: DataFrame,
        n_windows: usize,
        h: usize,
    ) -> Result<DataFrame> {
        self.cross_validation(CrossValidationConfig {
            data: df,
            n_windows,
            horizon: h,
            ..Default::default()
        })
    }
}
```

### Issue: Different Error Handling

**Problem**: Rust error handling differs from Python exceptions

**Symptoms**:
```
Python: try/except blocks
Rust: Result<T, E> types
```

**Solutions**:

1. **Use consistent error handling**:
```rust
use anyhow::{Context, Result};

fn train_model(data: DataFrame) -> Result<NeuralForecast> {
    let model = LSTM::builder()
        .horizon(12)
        .build()
        .context("Failed to create LSTM model")?;
    
    model.fit(data)
        .context("Failed to train model")?;
    
    Ok(model)
}

// Usage
match train_model(df) {
    Ok(model) => println!("Training successful"),
    Err(e) => eprintln!("Training failed: {:#}", e),
}
```

2. **Create error conversion utilities**:
```rust
#[derive(Debug, thiserror::Error)]
pub enum MigrationError {
    #[error("Data validation failed: {0}")]
    DataValidation(String),
    #[error("Model training failed: {0}")]
    Training(String),
    #[error("Prediction failed: {0}")]
    Prediction(String),
}

// Convert from various error types
impl From<PolarsError> for MigrationError {
    fn from(err: PolarsError) -> Self {
        MigrationError::DataValidation(err.to_string())
    }
}
```

## Deployment and Runtime Errors

### Issue: Container Build Failures

**Problem**: Docker build fails for Rust application

**Symptoms**:
```
Error: failed to solve: process "/bin/sh -c cargo build --release" did not complete
```

**Solutions**:

1. **Optimize Dockerfile**:
```dockerfile
# Multi-stage build
FROM rust:1.70 as builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Cache dependencies
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release && \
    rm -rf src

# Copy source and build
COPY src ./src
RUN touch src/main.rs && \
    cargo build --release

# Runtime stage
FROM debian:bullseye-slim
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/neuro-divergent /usr/local/bin/
CMD ["neuro-divergent"]
```

2. **Use .dockerignore**:
```
# .dockerignore
target/
.git/
*.md
.gitignore
Dockerfile
.dockerignore
```

### Issue: Runtime Panics

**Problem**: Application crashes with panic in production

**Symptoms**:
```
thread 'main' panicked at 'index out of bounds: the len is 3 but the index is 5'
```

**Solutions**:

1. **Add bounds checking**:
```rust
// Instead of direct indexing
let value = data[index];  // Can panic

// Use safe methods
let value = data.get(index).ok_or("Index out of bounds")?;
```

2. **Set up panic handler**:
```rust
use std::panic;

fn main() {
    panic::set_hook(Box::new(|panic_info| {
        eprintln!("Application panicked: {}", panic_info);
        // Log to monitoring system
        // Send alert
    }));
    
    // Rest of application
}
```

3. **Use defensive programming**:
```rust
fn safe_prediction(model: &Model, data: DataFrame) -> Result<DataFrame> {
    // Validate inputs
    if data.is_empty() {
        return Err("Empty data provided".into());
    }
    
    if data.width() < model.required_columns() {
        return Err("Insufficient columns in data".into());
    }
    
    // Proceed with prediction
    model.predict(data)
}
```

## Integration Problems

### Issue: MLflow Integration Failures

**Problem**: Cannot log metrics to MLflow

**Symptoms**:
```
Error: Failed to connect to MLflow server at http://localhost:5000
```

**Solutions**:

1. **Check MLflow server status**:
```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Test connectivity
curl http://localhost:5000/api/2.0/mlflow/experiments/list
```

2. **Configure timeout and retry**:
```rust
use reqwest::Client;
use std::time::Duration;

let client = Client::builder()
    .timeout(Duration::from_secs(30))
    .build()?;

// Retry logic
for attempt in 1..=3 {
    match client.post(&mlflow_url).send().await {
        Ok(response) => break,
        Err(e) if attempt < 3 => {
            eprintln!("Attempt {} failed: {}, retrying...", attempt, e);
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
        Err(e) => return Err(e.into()),
    }
}
```

### Issue: Kubernetes Deployment Problems

**Problem**: Pod fails to start in Kubernetes

**Symptoms**:
```
Error: CrashLoopBackOff
Error: ImagePullBackOff
```

**Solutions**:

1. **Check image and registry**:
```bash
# Debug pod
kubectl describe pod <pod-name>
kubectl logs <pod-name>

# Check image exists
docker pull neuro-divergent:latest
```

2. **Fix resource limits**:
```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "100m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

3. **Add health checks**:
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10
readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
```

## Debugging Techniques

### Enable Debug Logging

```rust
// Add to main.rs
use tracing::{info, debug, error};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn init_logging() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "debug".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();
}

fn main() {
    init_logging();
    info!("Starting application");
    
    // Your code here
}
```

### Performance Profiling

```bash
# Install flamegraph
cargo install flamegraph

# Generate flamegraph
cargo flamegraph --bin neuro-divergent

# Install perf (Linux)
sudo apt-get install linux-perf

# Profile with perf
perf record --call-graph=dwarf ./target/release/neuro-divergent
perf report
```

### Memory Debugging

```bash
# Use AddressSanitizer
RUSTFLAGS="-Z sanitizer=address" cargo run --target x86_64-unknown-linux-gnu

# Use Miri for undefined behavior detection
RUSTUP_TOOLCHAIN=nightly cargo miri test
```

### Integration Testing

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_end_to_end_workflow() {
        // Load test data
        let df = load_test_data("test_data.csv").unwrap();
        
        // Train model
        let mut nf = create_test_model();
        nf.fit(df.clone()).unwrap();
        
        // Make prediction
        let predictions = nf.predict().unwrap();
        
        // Validate results
        assert!(!predictions.is_empty());
        assert_eq!(predictions.width(), 3); // ds, unique_id, prediction
    }
}
```

---

**Quick Debugging Checklist**:
- ✅ Check Rust version compatibility
- ✅ Verify system dependencies installed
- ✅ Confirm data format compatibility
- ✅ Validate model parameters
- ✅ Enable debug logging
- ✅ Check resource limits
- ✅ Verify network connectivity
- ✅ Test with minimal examples
- ✅ Review error messages carefully
- ✅ Compare with working Python version