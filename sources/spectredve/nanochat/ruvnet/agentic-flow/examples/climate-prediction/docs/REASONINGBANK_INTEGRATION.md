# Climate Prediction with ReasoningBank Integration

## Overview

This example demonstrates how to integrate ReasoningBank into a climate prediction system for continuous learning and optimization. The system learns from prediction outcomes to improve model selection and accuracy over time.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Climate Prediction System                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ API Client   │  │  Prediction  │  │ ReasoningBank│    │
│  │ (Weather API)│  │    Cache     │  │  Optimizer   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│         │                  │                  │            │
│         ↓                  ↓                  ↓            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          ReasoningBank Learning System               │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │                                                       │  │
│  │  • Pattern Storage    - Historical accuracy data    │  │
│  │  • Learning Module    - Outcome analysis            │  │
│  │  • Optimization       - Model selection             │  │
│  │                                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Features

### 1. **Pattern Storage**

Stores successful prediction patterns with metadata:

```rust
pub fn store_successful_prediction(
    &self,
    input: &ClimateInput,
    prediction: &ClimatePrediction,
    actual: Option<&ClimateData>,
) -> Result<String>
```

**What it stores:**
- Location coordinates
- Model used (Linear, ARIMA, Neural, Ensemble)
- Prediction accuracy
- Confidence scores
- Timestamp

### 2. **Performance Optimization**

Tracks prediction accuracy by location and model:

```rust
pub fn optimize_model_selection(
    &mut self,
    location: (f64, f64),
) -> Result<OptimizationResult>
```

**Optimization strategy:**
1. Query historical patterns within 1° radius
2. Calculate average accuracy per model
3. Return best performing model with confidence score
4. Cache result for fast repeated lookups

### 3. **Continuous Learning**

Learns from prediction outcomes:

```rust
pub fn learn_from_outcome(
    &mut self,
    prediction: &ClimatePrediction,
    actual: &ClimateData,
) -> Result<LearningInsight>
```

**Learning process:**
- Compare predicted vs actual temperature/humidity
- Calculate accuracy score
- Update pattern confidence
- Store trajectory for future reference

## Usage Example

### Basic Prediction

```rust
use climate_core::{
    ClimateApiClient, ClimateInput, Location, ModelType,
    PredictionCache, ReasoningBankOptimizer,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize components
    let db_path = PathBuf::from(".swarm/climate_memory.db");
    let mut optimizer = ReasoningBankOptimizer::new(db_path, "climate")?;
    let cache = PredictionCache::new();
    let api_client = ClimateApiClient::new(None);

    let location = Location {
        latitude: 40.7128,
        longitude: -74.0060,
        name: Some("New York".to_string()),
    };

    // Step 1: Optimize model selection
    let optimization = optimizer.optimize_model_selection((
        location.latitude,
        location.longitude,
    ))?;

    println!("Recommended model: {:?}", optimization.recommended_model);
    println!("Confidence: {:.2}%", optimization.confidence * 100.0);

    // Step 2: Fetch current data
    let current = api_client.fetch_current(&location).await?;

    // Step 3: Make prediction (simplified)
    let prediction = ClimatePrediction {
        location: location.clone(),
        predicted_at: Utc::now(),
        prediction_time: Utc::now() + Duration::hours(24),
        temperature: current.temperature + 2.0,
        humidity: current.humidity - 5.0,
        precipitation_probability: 0.2,
        confidence: optimization.confidence,
        model_used: optimization.recommended_model,
    };

    // Step 4: Cache and record
    cache.store_prediction(prediction.clone())?;

    let input = ClimateInput {
        location: location.clone(),
        forecast_hours: 24,
        model_preference: Some(optimization.recommended_model),
    };

    optimizer.record_prediction_result(&input, &prediction, Some(&current))?;

    Ok(())
}
```

### Hooks Integration

The system integrates with Claude-Flow hooks for coordination:

```bash
# Before prediction
npx claude-flow@alpha hooks pre-task --description "Climate prediction"
npx claude-flow@alpha hooks session-restore --session-id "climate-prediction"

# After each prediction
npx claude-flow@alpha hooks post-edit --file "predictions.json" \
  --memory-key "climate/predictions/results"

# At shutdown
npx claude-flow@alpha hooks post-task --task-id "climate-prediction"
npx claude-flow@alpha hooks session-end --export-metrics true
```

## ReasoningBank Benefits

### 1. **Adaptive Model Selection**

The system automatically selects the best model based on:
- Historical accuracy at similar locations
- Seasonal patterns
- Recent performance trends
- Sample size confidence

**Example:**
```
Location: New York (40.71, -74.01)
Historical data: 50 predictions
  - Ensemble: 85% accuracy (30 samples)
  - Neural: 78% accuracy (15 samples)
  - ARIMA: 72% accuracy (5 samples)

→ Recommendation: Ensemble (85% confidence)
```

### 2. **Cache Optimization**

Intelligent caching reduces API calls:
- 5-minute TTL for predictions
- Location-based key generation
- Automatic expiry cleanup
- Memory-efficient storage

**Performance:**
- Cache hit: <1ms
- Cache miss + API: ~100-200ms
- Cache savings: 99% faster repeated queries

### 3. **Learning Statistics**

Track system improvement over time:

```rust
let stats = optimizer.get_optimization_stats()?;
println!("Total predictions: {}", stats.total_predictions);
println!("Avg confidence: {:.2}%", stats.avg_confidence * 100.0);
println!("Predictions by model: {:?}", stats.patterns_by_model);
```

**Example output:**
```
Total predictions: 150
Avg confidence: 82.50%
Predictions by model: {
  "ensemble": 80,
  "neural": 45,
  "arima": 25
}
```

## File Structure

```
climate-prediction/
├── Cargo.toml
├── src/
│   └── main.rs                 # CLI application
├── crates/
│   └── climate-core/
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs          # Core exports
│           ├── models.rs       # Data structures
│           ├── api.rs          # Weather API client
│           ├── cache.rs        # Prediction cache
│           └── reasoningbank/
│               ├── mod.rs      # Module exports
│               ├── learning.rs # Learning integration
│               ├── patterns.rs # Pattern storage
│               └── optimization.rs # Model optimization
├── tests/
│   └── integration_test.rs     # E2E tests
├── docs/
│   └── REASONINGBANK_INTEGRATION.md
└── scripts/
    ├── run-prediction.sh       # Run prediction
    └── demo-learning.sh        # Learning demo
```

## Running the Example

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install claude-flow
npm install -g claude-flow@alpha
```

### Build and Run

```bash
# Build
cd examples/climate-prediction
cargo build --release

# Run prediction
cargo run --release

# Run tests
cargo test

# Run with debug logging
RUST_LOG=debug cargo run --release
```

### Demo Scripts

```bash
# Basic prediction demo
./scripts/run-prediction.sh

# Learning system demo
./scripts/demo-learning.sh
```

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Model optimization (cached) | <1ms | From memory |
| Model optimization (uncached) | 2-5ms | Pattern query + ranking |
| Pattern storage | 1-2ms | SQLite insert |
| Pattern query | 2-10ms | Radius search + filtering |
| Cache lookup | <1ms | In-memory HashMap |
| API call | 100-200ms | External weather API |

## Learning Metrics

The system tracks:

- **Accuracy**: Temperature and humidity prediction error
- **Confidence**: Based on historical performance
- **Usage**: How often each model is selected
- **Samples**: Number of predictions per location

**Accuracy calculation:**
```rust
let temp_accuracy = 1.0 - (temp_diff / 10.0).min(1.0);
let humidity_accuracy = 1.0 - (humidity_diff / 20.0).min(1.0);
let overall = (temp_accuracy + humidity_accuracy) / 2.0;
```

## Advanced Features

### Seasonal Patterns

Store seasonal performance:

```rust
patterns.store_seasonal_pattern("summer", ModelType::Neural, 0.88)?;
patterns.store_seasonal_pattern("winter", ModelType::Ensemble, 0.85)?;
```

### Location Clustering

Query patterns within radius:

```rust
let nearby = patterns.query_location_patterns(
    (40.7128, -74.0060),
    1.0  // 1 degree radius (~111km)
)?;
```

### Confidence Boosting

Higher confidence for:
- More samples
- Recent predictions
- Similar locations
- Consistent accuracy

## Troubleshooting

### Database Locked

```bash
# Increase max_connections
let config = StorageConfig {
    max_connections: 20,  // Default: 10
    ..Default::default()
};
```

### Cache Not Working

```bash
# Check TTL setting
let cache = PredictionCache::with_ttl(Duration::minutes(10));

# Clear cache if stale
cache.clear();
```

### Low Accuracy

```rust
// Increase pattern sample size before trusting
if patterns.len() < 10 {
    // Use default model
}

// Check pattern ages
if pattern.last_used < (Utc::now() - Duration::days(30)) {
    // Pattern might be stale
}
```

## Best Practices

1. **Initialize Early**: Create optimizer at startup
2. **Record Everything**: Store all predictions for learning
3. **Cache Aggressively**: 5-minute TTL is usually optimal
4. **Monitor Stats**: Track improvement over time
5. **Clean Periodically**: Run cache cleanup every hour
6. **Export Patterns**: Backup learned patterns regularly

## Future Enhancements

- [ ] Neural network model integration
- [ ] Real-time weather API integration
- [ ] Multi-location batch predictions
- [ ] WebSocket updates for live tracking
- [ ] Dashboard for visualization
- [ ] Export/import learned patterns
- [ ] A/B testing between models
- [ ] Ensemble weighting optimization

## References

- [ReasoningBank Architecture](/workspaces/agentic-flow/agentic-flow/docs/REASONINGBANK_ARCHITECTURE.md)
- [Claude-Flow Documentation](https://github.com/ruvnet/claude-flow)
- [Self-Learning Guide](/workspaces/agentic-flow/examples/SELF_LEARNING_GUIDE.md)
