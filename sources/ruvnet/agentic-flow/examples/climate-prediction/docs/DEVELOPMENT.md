# Development Guide

This guide covers development workflows, contribution guidelines, and best practices for the Climate Prediction System.

## Table of Contents

1. [Development Environment](#development-environment)
2. [Project Structure](#project-structure)
3. [Development Workflow](#development-workflow)
4. [Testing](#testing)
5. [Code Style](#code-style)
6. [Contributing](#contributing)
7. [Release Process](#release-process)

## Development Environment

### Prerequisites

- **Rust 1.70+** with cargo
- **Node.js 18+** and npm 9+
- **Git 2.30+**
- **Docker** (optional, for containerized development)
- **PostgreSQL 14+** (for development database)

### IDE Setup

**Recommended: Visual Studio Code**

Install extensions:
```bash
code --install-extension rust-lang.rust-analyzer
code --install-extension vadimcn.vscode-lldb
code --install-extension tamasfe.even-better-toml
code --install-extension serayuzgur.crates
```

**VS Code settings.json:**
```json
{
  "rust-analyzer.cargo.features": "all",
  "rust-analyzer.checkOnSave.command": "clippy",
  "rust-analyzer.inlayHints.enable": true,
  "editor.formatOnSave": true,
  "editor.rulers": [100],
  "[rust]": {
    "editor.defaultFormatter": "rust-lang.rust-analyzer"
  }
}
```

### Clone and Setup

```bash
# Clone repository
git clone https://github.com/yourusername/climate-prediction.git
cd climate-prediction

# Install Rust dependencies
cargo fetch

# Install development tools
cargo install cargo-watch cargo-tarpaulin cargo-audit cargo-outdated

# Install pre-commit hooks
./scripts/install-hooks.sh

# Setup development database (optional)
./scripts/setup-dev-db.sh
```

### Environment Configuration

Create `.env.development`:

```bash
# Server
SERVER_HOST=127.0.0.1
SERVER_PORT=8080
RUST_LOG=debug

# Database (optional)
DATABASE_URL=postgres://climate:password@localhost/climate_dev

# Models
MODEL_PATH=./models
MODEL_CACHE_SIZE=1000

# ReasoningBank
REASONINGBANK_BACKEND=wasm
REASONINGBANK_MEMORY_TTL=3600

# Development
RUST_BACKTRACE=1
ENABLE_HOT_RELOAD=true
```

## Project Structure

```
climate-prediction/
├── src/
│   ├── main.rs              # Application entry point
│   ├── lib.rs               # Library root
│   ├── api/                 # API server implementation
│   │   ├── mod.rs
│   │   ├── handlers.rs      # Request handlers
│   │   ├── routes.rs        # Route definitions
│   │   └── middleware.rs    # Auth, logging, etc.
│   ├── models/              # Prediction models
│   │   ├── mod.rs
│   │   ├── neural.rs        # Neural network model
│   │   ├── arima.rs         # ARIMA time series
│   │   └── ensemble.rs      # Ensemble coordinator
│   ├── core/                # Core business logic
│   │   ├── mod.rs
│   │   ├── predictor.rs     # Main prediction engine
│   │   ├── data_processor.rs
│   │   └── validator.rs
│   ├── reasoningbank/       # ReasoningBank integration
│   │   ├── mod.rs
│   │   ├── memory.rs        # Memory store
│   │   └── patterns.rs      # Pattern learning
│   └── utils/               # Utilities
│       ├── mod.rs
│       ├── config.rs        # Configuration
│       └── logging.rs       # Logging setup
├── tests/                   # Integration tests
│   ├── integration_tests.rs
│   ├── api_tests.rs
│   └── model_tests.rs
├── benches/                 # Benchmarks
│   └── prediction_bench.rs
├── examples/                # Example programs
│   ├── first_prediction.rs
│   ├── batch_processing.rs
│   └── custom_model.rs
├── models/                  # Pre-trained models
│   ├── neural.wasm
│   └── arima.json
├── docs/                    # Documentation
├── scripts/                 # Utility scripts
├── Cargo.toml              # Rust dependencies
└── config.toml             # Application config
```

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/my-feature

# Start development server with hot reload
cargo watch -x 'run --bin climate-api'

# In another terminal, run tests continuously
cargo watch -x test
```

### 2. Making Changes

**Add new prediction model:**

```rust
// src/models/my_model.rs
use crate::core::{PredictionConfig, PredictionResult};
use anyhow::Result;

pub struct MyModel {
    // Model state
}

impl MyModel {
    pub fn new() -> Result<Self> {
        Ok(Self {})
    }

    pub async fn predict(&self, config: &PredictionConfig) -> Result<PredictionResult> {
        // Implementation
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_prediction() {
        let model = MyModel::new().unwrap();
        // Test implementation
    }
}
```

**Add model to registry:**

```rust
// src/models/mod.rs
pub mod my_model;

use my_model::MyModel;

pub enum ModelType {
    Neural,
    Arima,
    Hybrid,
    Ensemble,
    MyModel,  // Add new model
}
```

### 3. Running Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_prediction

# Run with output
cargo test -- --nocapture

# Run integration tests only
cargo test --test integration_tests

# Run benchmarks
cargo bench
```

### 4. Code Quality

```bash
# Format code
cargo fmt

# Check formatting
cargo fmt -- --check

# Run clippy linter
cargo clippy -- -D warnings

# Check for security vulnerabilities
cargo audit

# Check for outdated dependencies
cargo outdated
```

### 5. Building

```bash
# Debug build (fast compilation)
cargo build

# Release build (optimized)
cargo build --release

# Build with specific features
cargo build --features "experimental"

# Build documentation
cargo doc --open
```

## Testing

### Unit Tests

Located in same file as code using `#[cfg(test)]`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation() {
        let result = validate_coordinates(40.7128, -74.0060);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_async_prediction() {
        let predictor = ClimatePredictor::new(ModelType::Neural).unwrap();
        let config = PredictionConfig::default();
        let result = predictor.predict(&config).await;
        assert!(result.is_ok());
    }
}
```

### Integration Tests

Located in `tests/` directory:

```rust
// tests/api_tests.rs
use climate_prediction::api::Server;
use reqwest::Client;

#[tokio::test]
async fn test_prediction_endpoint() {
    // Start test server
    let server = Server::new_test().await;

    // Make request
    let client = Client::new();
    let response = client
        .post(&format!("{}/api/predictions", server.url()))
        .json(&serde_json::json!({
            "latitude": 40.7128,
            "longitude": -74.0060,
            "days_ahead": 7
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body: serde_json::Value = response.json().await.unwrap();
    assert!(body["prediction"]["temperature"].is_number());
}
```

### Property-Based Testing

Using `proptest`:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_coordinates_always_valid(
        lat in -90.0..=90.0,
        lon in -180.0..=180.0
    ) {
        let result = validate_coordinates(lat, lon);
        assert!(result.is_ok());
    }
}
```

### Coverage Reports

```bash
# Generate coverage report
cargo tarpaulin --out Html --output-dir coverage

# View report
open coverage/index.html
```

### Benchmarking

```rust
// benches/prediction_bench.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use climate_prediction::{ClimatePredictor, ModelType, PredictionConfig};

fn prediction_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let predictor = ClimatePredictor::new(ModelType::Ensemble).unwrap();
    let config = PredictionConfig::default();

    c.bench_function("ensemble prediction", |b| {
        b.to_async(&rt).iter(|| async {
            predictor.predict(black_box(&config)).await.unwrap()
        });
    });
}

criterion_group!(benches, prediction_benchmark);
criterion_main!(benches);
```

Run benchmarks:

```bash
cargo bench
```

## Code Style

### Formatting Rules

Follow Rust standard style (enforced by `rustfmt`):

```toml
# rustfmt.toml
max_width = 100
hard_tabs = false
tab_spaces = 4
newline_style = "Unix"
use_small_heuristics = "Default"
reorder_imports = true
reorder_modules = true
remove_nested_parens = true
edition = "2021"
```

### Naming Conventions

- **Types**: `PascalCase` (e.g., `ClimatePredictor`)
- **Functions**: `snake_case` (e.g., `predict_weather`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `MAX_PREDICTIONS`)
- **Modules**: `snake_case` (e.g., `data_processor`)

### Documentation

All public items must have documentation:

```rust
/// Predicts weather conditions for a given location and timeframe.
///
/// # Arguments
///
/// * `config` - Configuration specifying location and prediction parameters
///
/// # Returns
///
/// Returns a `Result` containing the prediction or an error.
///
/// # Errors
///
/// Returns `Err` if:
/// - Coordinates are invalid
/// - Model inference fails
/// - Network request times out
///
/// # Examples
///
/// ```
/// use climate_prediction::{ClimatePredictor, PredictionConfig, ModelType};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let predictor = ClimatePredictor::new(ModelType::Ensemble)?;
/// let config = PredictionConfig::default();
/// let prediction = predictor.predict(&config).await?;
/// # Ok(())
/// # }
/// ```
pub async fn predict(&self, config: &PredictionConfig) -> Result<PredictionResult> {
    // Implementation
}
```

### Error Handling

Use `anyhow` for application errors and `thiserror` for library errors:

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PredictionError {
    #[error("Invalid coordinates: lat={lat}, lon={lon}")]
    InvalidCoordinates { lat: f64, lon: f64 },

    #[error("Model inference failed: {0}")]
    InferenceFailed(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
```

## Contributing

### Pull Request Process

1. **Fork and branch**: Create feature branch from `main`
2. **Implement**: Write code following style guide
3. **Test**: Ensure all tests pass and add new tests
4. **Document**: Update documentation and add examples
5. **Commit**: Use conventional commit messages
6. **Push**: Push to your fork
7. **PR**: Open pull request with description

### Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Examples:**
```
feat(api): Add batch prediction endpoint

Implement /api/predictions/batch endpoint for requesting
predictions for multiple locations in a single request.

Closes #123
```

```
fix(neural): Fix memory leak in WASM inference

The neural model was not properly releasing memory after
each inference, causing memory usage to grow over time.

Fixes #456
```

### Code Review

All contributions require code review:

- At least one approval from maintainer
- All tests passing
- Code coverage maintained
- Documentation updated
- No merge conflicts

### Development Environment

Use pre-commit hooks to ensure quality:

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Format code
cargo fmt -- --check || exit 1

# Run clippy
cargo clippy -- -D warnings || exit 1

# Run tests
cargo test || exit 1

echo "✅ Pre-commit checks passed"
```

## Release Process

### Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Release Checklist

1. **Update version** in `Cargo.toml`
2. **Update CHANGELOG.md** with release notes
3. **Run full test suite**: `cargo test --all-features`
4. **Run benchmarks**: `cargo bench`
5. **Build release**: `cargo build --release`
6. **Tag release**: `git tag v1.0.0`
7. **Push tag**: `git push origin v1.0.0`
8. **Publish to crates.io**: `cargo publish`
9. **Create GitHub release** with binaries
10. **Update documentation** site

### Release Automation

Use GitHub Actions for automated releases:

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - run: cargo test --all-features
      - run: cargo build --release
      - uses: softprops/action-gh-release@v1
        with:
          files: target/release/climate-api
```

## Debugging

### Enable Debug Logging

```bash
RUST_LOG=debug cargo run
```

### Use LLDB Debugger

**VS Code launch.json:**
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "lldb",
      "request": "launch",
      "name": "Debug",
      "cargo": {
        "args": ["build", "--bin=climate-api"]
      },
      "args": [],
      "cwd": "${workspaceFolder}"
    }
  ]
}
```

### Performance Profiling

```bash
# Install flamegraph
cargo install flamegraph

# Generate flamegraph
cargo flamegraph --bin climate-api

# View flamegraph.svg
open flamegraph.svg
```

## Additional Resources

- [Rust Book](https://doc.rust-lang.org/book/)
- [Cargo Book](https://doc.rust-lang.org/cargo/)
- [Async Book](https://rust-lang.github.io/async-book/)
- [API Guidelines](https://rust-lang.github.io/api-guidelines/)

---

**Happy coding! Thank you for contributing to Climate Prediction System.** 🚀
