# Climate Prediction System

A modular Rust-based climate prediction system with ML inference, physics-informed constraints, and REST API.

## Architecture

This project follows a layered, modular architecture:

```
┌────────────────────────────────────┐
│  CLI + API (Presentation)          │
├────────────────────────────────────┤
│  Models + Physics (Domain)         │
├────────────────────────────────────┤
│  Data Sources (Infrastructure)     │
├────────────────────────────────────┤
│  Core Types & Traits (Foundation)  │
└────────────────────────────────────┘
```

## Crates

- **climate-core**: Core types, traits, and errors
- **climate-data**: Data ingestion from external APIs
- **climate-models**: ML model inference with Candle
- **climate-physics**: Physics-informed constraints
- **climate-api**: REST API server with Axum
- **climate-cli**: Command-line interface

## Quick Start

### Prerequisites
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install dependencies
cargo build
```

### Running Tests
```bash
# Run all tests
cargo test --workspace

# Run specific crate tests
cargo test -p climate-core
```

### CLI Usage
```bash
# Single prediction
cargo run -p climate-cli -- predict \
  --location 40.7128,-74.0060 \
  --variable temperature \
  --days 7

# Batch predictions
cargo run -p climate-cli -- batch \
  --file locations.json \
  --output predictions.json

# List available models
cargo run -p climate-cli -- models list
```

### API Server
```bash
# Start server
cargo run -p climate-api

# In another terminal, make requests
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "location": {"latitude": 40.7128, "longitude": -74.0060},
    "time_range": {
      "start": "2024-01-01T00:00:00Z",
      "end": "2024-01-07T00:00:00Z"
    },
    "variables": ["Temperature"]
  }'
```

## Configuration

Create `config/default.toml`:
```toml
[api]
host = "127.0.0.1"
port = 8080

[data_sources]
openweather_api_key = "${OPENWEATHER_API_KEY}"

[models]
default_model = "climate-v1"
model_dir = "./models"
```

## Development

### Project Structure
```
climate-prediction/
├── Cargo.toml              # Workspace definition
├── crates/
│   ├── climate-core/       # Foundation layer
│   ├── climate-data/       # Data ingestion
│   ├── climate-models/     # ML inference
│   ├── climate-physics/    # Physics constraints
│   ├── climate-api/        # REST API
│   └── climate-cli/        # CLI tool
├── docs/                   # Architecture docs
├── tests/                  # Integration tests
└── config/                 # Configuration files
```

### Adding a New Data Source
```rust
use climate_core::{DataSource, Result};

pub struct MyDataSource { }

#[async_trait]
impl DataSource for MyDataSource {
    async fn fetch_observations(...) -> Result<Vec<Observation>> {
        // Implementation
    }
}
```

### Adding a New Model
```rust
use climate_core::{PredictionModel, Result};

pub struct MyModel { }

#[async_trait]
impl PredictionModel for MyModel {
    async fn predict(...) -> Result<Vec<Prediction>> {
        // Implementation
    }
}
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System design and decisions
- [API Design](docs/API-DESIGN.md) - REST API specification
- [Trait Design](docs/TRAIT-DESIGN.md) - Extensibility guide
- [ADR-001](docs/ADR-001-crate-structure.md) - Crate structure decision
- [ADR-002](docs/ADR-002-error-handling.md) - Error handling strategy
- [ADR-003](docs/ADR-003-async-runtime.md) - Async runtime selection

## Performance

- Single prediction: <100ms p95 latency
- Batch predictions: >1000 locations/second
- Model loading: <5 seconds cold start

## ReasoningBank Integration

This project uses ReasoningBank for coordination:

```bash
# Session management
npx claude-flow@alpha hooks session-restore --session-id "climate-swarm"

# Track changes
npx claude-flow@alpha hooks post-edit \
  --file "crates/climate-core/src/lib.rs" \
  --memory-key "climate/core/types"

# Export metrics
npx claude-flow@alpha hooks session-end --export-metrics true
```

## License

MIT OR Apache-2.0

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure `cargo test` passes
5. Submit a pull request
