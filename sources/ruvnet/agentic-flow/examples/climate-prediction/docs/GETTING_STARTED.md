# Getting Started with Climate Prediction System

This guide will walk you through setting up and using the Climate Prediction System from scratch.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [First Prediction](#first-prediction)
5. [Using the API](#using-the-api)
6. [ReasoningBank Integration](#reasoningbank-integration)
7. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **CPU**: 2 cores, 2.0 GHz
- **RAM**: 4 GB
- **Storage**: 1 GB free space
- **OS**: Linux, macOS, or Windows (WSL2)

### Recommended Requirements
- **CPU**: 4+ cores, 3.0 GHz
- **RAM**: 8 GB
- **Storage**: 5 GB free space (for model storage)
- **OS**: Ubuntu 22.04 LTS or macOS 13+

### Software Dependencies
- **Rust**: 1.70+ (with cargo)
- **Node.js**: 18+ (for agentic-flow)
- **npm**: 9+
- **Git**: 2.30+

## Installation

### Step 1: Install Rust

```bash
# Install Rust using rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Source environment
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

### Step 2: Install Node.js and npm

**Ubuntu/Debian:**
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

**macOS:**
```bash
brew install node
```

**Verify:**
```bash
node --version
npm --version
```

### Step 3: Install agentic-flow

```bash
# Install globally
npm install -g agentic-flow@latest

# Verify installation
npx agentic-flow --version
```

### Step 4: Install Claude Flow

```bash
# Install Claude Flow alpha version
npm install -g @ruv/claude-flow@alpha

# Verify installation
npx claude-flow@alpha --version
```

### Step 5: Clone and Build Project

```bash
# Clone repository
git clone https://github.com/yourusername/climate-prediction.git
cd climate-prediction

# Install Rust dependencies
cargo fetch

# Build in release mode
cargo build --release

# Run tests to verify
cargo test
```

**Build should complete in 2-5 minutes depending on your system.**

## Configuration

### Step 1: Create Configuration File

```bash
# Copy example configuration
cp config.example.toml config.toml

# Edit configuration
nano config.toml  # or your preferred editor
```

### Step 2: Basic Configuration

Edit `config.toml`:

```toml
[server]
host = "127.0.0.1"  # Use 0.0.0.0 for external access
port = 8080
workers = 4  # Set to number of CPU cores

[models]
# Model files will be downloaded on first run
neural_path = "./models/neural.wasm"
arima_config = "./models/arima.json"

# Ensemble weights (must sum to 1.0)
ensemble_weights = [0.5, 0.3, 0.2]  # Neural, ARIMA, Hybrid

[reasoningbank]
enabled = true
backend = "auto"  # Auto-detect best backend
memory_ttl = 86400  # 24 hours

[cache]
enabled = true
ttl = 300  # 5 minutes
max_size = 1000  # Cache up to 1000 predictions

[logging]
level = "info"  # debug, info, warn, error
format = "json"  # json or pretty
```

### Step 3: Environment Variables (Optional)

Create `.env` file:

```bash
# Server configuration
SERVER_HOST=127.0.0.1
SERVER_PORT=8080

# API keys (if using external data sources)
WEATHER_API_KEY=your_api_key_here

# ReasoningBank configuration
REASONINGBANK_BACKEND=wasm
REASONINGBANK_MEMORY_TTL=86400

# Logging
RUST_LOG=info
RUST_BACKTRACE=1
```

## First Prediction

### Step 1: Start the Server

```bash
# Run the API server
cargo run --release --bin climate-api

# Server will start on http://127.0.0.1:8080
# Check logs for confirmation
```

Expected output:
```
[2025-10-14T10:00:00Z INFO] Starting Climate Prediction API
[2025-10-14T10:00:00Z INFO] Loading models...
[2025-10-14T10:00:00Z INFO] Neural model loaded: 2.1MB
[2025-10-14T10:00:00Z INFO] ARIMA model loaded
[2025-10-14T10:00:00Z INFO] ReasoningBank initialized (backend: wasm)
[2025-10-14T10:00:00Z INFO] Server listening on http://127.0.0.1:8080
```

### Step 2: Make Your First Prediction (CLI)

Create `examples/first_prediction.rs`:

```rust
use climate_prediction::{ClimatePredictor, ModelType, PredictionConfig};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize predictor with ensemble model
    let predictor = ClimatePredictor::new(ModelType::Ensemble)?;

    // Configure prediction for New York City
    let config = PredictionConfig {
        latitude: 40.7128,
        longitude: -74.0060,
        days_ahead: 7,
        include_uncertainty: true,
    };

    println!("🌤️  Getting 7-day weather prediction for NYC...\n");

    // Get prediction
    let prediction = predictor.predict(&config).await?;

    // Display results
    println!("📊 Prediction Results:");
    println!("   Temperature: {:.1}°C", prediction.temperature);
    println!("   Feels Like: {:.1}°C", prediction.feels_like);
    println!("   Humidity: {}%", prediction.humidity);
    println!("   Precipitation: {}mm", prediction.precipitation);
    println!("   Confidence: {:.1}%", prediction.confidence * 100.0);
    println!("   Model Used: {}", prediction.model_name);

    if let Some(uncertainty) = prediction.uncertainty {
        println!("\n📈 Uncertainty Range:");
        println!("   Min: {:.1}°C", prediction.temperature - uncertainty);
        println!("   Max: {:.1}°C", prediction.temperature + uncertainty);
    }

    Ok(())
}
```

Run it:

```bash
cargo run --example first_prediction
```

### Step 3: Make Prediction via API

Using curl:

```bash
curl -X POST http://localhost:8080/api/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 40.7128,
    "longitude": -74.0060,
    "days_ahead": 7,
    "include_uncertainty": true
  }'
```

Response:

```json
{
  "prediction": {
    "temperature": 22.5,
    "feels_like": 21.8,
    "humidity": 65,
    "precipitation": 2.3,
    "wind_speed": 12.5,
    "confidence": 0.89,
    "model_name": "ensemble",
    "uncertainty": 1.2
  },
  "metadata": {
    "timestamp": "2025-10-14T10:30:00Z",
    "location": {
      "latitude": 40.7128,
      "longitude": -74.0060
    },
    "processing_time_ms": 42
  }
}
```

## Using the API

### Authentication

Generate an API key:

```bash
curl -X POST http://localhost:8080/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "secure_password"
  }'
```

Response includes API key:

```json
{
  "api_key": "cpk_1234567890abcdef",
  "expires_at": "2026-10-14T10:30:00Z"
}
```

Use API key in requests:

```bash
curl -X POST http://localhost:8080/api/predictions \
  -H "Authorization: Bearer cpk_1234567890abcdef" \
  -H "Content-Type: application/json" \
  -d '{"latitude": 40.7128, "longitude": -74.0060, "days_ahead": 7}'
```

### Batch Predictions

Request predictions for multiple locations:

```bash
curl -X POST http://localhost:8080/api/predictions/batch \
  -H "Authorization: Bearer cpk_1234567890abcdef" \
  -H "Content-Type: application/json" \
  -d '{
    "locations": [
      {"latitude": 40.7128, "longitude": -74.0060},
      {"latitude": 34.0522, "longitude": -118.2437},
      {"latitude": 41.8781, "longitude": -87.6298}
    ],
    "days_ahead": 7
  }'
```

### Historical Data

Retrieve historical predictions:

```bash
curl -X GET "http://localhost:8080/api/predictions/history?lat=40.7128&lon=-74.0060&days=30" \
  -H "Authorization: Bearer cpk_1234567890abcdef"
```

## ReasoningBank Integration

### Enabling Adaptive Learning

ReasoningBank automatically learns from prediction patterns:

```rust
use climate_prediction::reasoningbank::{ReasoningBank, MemoryEntry};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize ReasoningBank
    let rb = ReasoningBank::new("auto").await?;

    // Store prediction result for learning
    let entry = MemoryEntry {
        key: "prediction/nyc/2025-10-14".to_string(),
        value: serde_json::to_value(&prediction)?,
        timestamp: chrono::Utc::now(),
        metadata: Some(json!({
            "location": "NYC",
            "model": "ensemble",
            "accuracy": 0.95
        })),
    };

    rb.store("climate/predictions", entry).await?;

    // Retrieve patterns
    let patterns = rb.query_patterns("climate/predictions", 30).await?;
    println!("Learned patterns: {}", patterns.len());

    Ok(())
}
```

### Using Pre-task Hooks

Automatically store predictions using Claude Flow hooks:

```bash
# Before making predictions
npx claude-flow@alpha hooks pre-task \
  --description "7-day forecast for NYC"

# After predictions (automatically stores in ReasoningBank)
npx claude-flow@alpha hooks post-edit \
  --file "prediction_results.json" \
  --memory-key "climate/nyc/forecast"
```

### Querying Learned Patterns

```bash
# Search for similar weather patterns
npx claude-flow@alpha hooks search \
  --query "warm sunny weather patterns" \
  --namespace "climate/predictions" \
  --limit 10
```

## Troubleshooting

### Issue: Server Won't Start

**Symptoms:**
```
Error: Address already in use (os error 98)
```

**Solution:**
```bash
# Check if port 8080 is in use
lsof -i :8080

# Kill the process
kill -9 <PID>

# Or change port in config.toml
port = 8081
```

### Issue: Model Loading Fails

**Symptoms:**
```
Error: Failed to load neural model: File not found
```

**Solution:**
```bash
# Create models directory
mkdir -p models

# Download models (run this script)
./scripts/download_models.sh

# Or specify different model path in config.toml
neural_path = "/path/to/neural.wasm"
```

### Issue: Low Prediction Accuracy

**Symptoms:**
- Confidence scores consistently below 70%
- High uncertainty values

**Solution:**
```bash
# Retrain models with recent data
cargo run --bin train_models -- --dataset recent_data.csv

# Adjust ensemble weights in config.toml
ensemble_weights = [0.6, 0.2, 0.2]  # Increase neural weight

# Enable ReasoningBank learning
reasoningbank.enabled = true
```

### Issue: Out of Memory

**Symptoms:**
```
Error: Cannot allocate memory
```

**Solution:**
```toml
# Reduce cache size in config.toml
[cache]
max_size = 500  # Reduced from 1000

# Reduce worker count
[server]
workers = 2  # Reduced from 4
```

### Issue: Slow Predictions

**Symptoms:**
- Response times > 200ms
- High CPU usage

**Solution:**
```toml
# Enable caching
[cache]
enabled = true
ttl = 600  # Cache for 10 minutes

# Increase workers
[server]
workers = 8

# Optimize model settings
[models]
neural_batch_size = 32  # Process multiple requests together
```

### Getting Help

If you encounter issues not covered here:

1. **Check logs**: `tail -f logs/climate-api.log`
2. **Enable debug mode**: Set `RUST_LOG=debug` in `.env`
3. **Run diagnostics**: `cargo run --bin diagnostics`
4. **Search issues**: [GitHub Issues](https://github.com/yourusername/climate-prediction/issues)
5. **Ask community**: [Discord Server](https://discord.gg/climate-prediction)

## Next Steps

Now that you have the system running:

1. **Explore API**: Read [API.md](API.md) for complete endpoint reference
2. **Review examples**: Check [examples/](examples/) for more use cases
3. **Learn architecture**: Study [ARCHITECTURE.md](ARCHITECTURE.md)
4. **Deploy to production**: Follow [DEPLOYMENT.md](DEPLOYMENT.md)
5. **Contribute**: Read [DEVELOPMENT.md](DEVELOPMENT.md)

---

**Congratulations! You're ready to build weather prediction applications.** 🎉
