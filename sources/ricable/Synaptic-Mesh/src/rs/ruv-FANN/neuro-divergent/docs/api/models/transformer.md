# Transformer Models

Transformer models leverage attention mechanisms to capture complex temporal relationships and handle long sequences efficiently. These models excel at modeling intricate patterns and dependencies in time series data.

## Transformer (Standard)

Standard transformer architecture adapted for time series forecasting, using self-attention to model temporal dependencies.

### Architecture

- **Self-Attention**: Captures relationships between all time steps
- **Position Encoding**: Temporal position information
- **Multi-Head Attention**: Multiple attention patterns
- **Feed-Forward Networks**: Non-linear transformations

### Configuration

```rust
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub horizon: usize,
    pub input_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_encoder_layers: usize,
    pub num_decoder_layers: usize,
    pub dim_feedforward: usize,
    pub dropout: f64,
    pub activation: ActivationType,
    pub use_positional_encoding: bool,
}
```

### Usage Examples

#### Basic Transformer

```rust
use neuro_divergent::models::{Transformer, TransformerConfig};

let transformer = Transformer::builder()
    .horizon(12)
    .input_size(48)
    .d_model(128)
    .num_heads(8)
    .num_encoder_layers(6)
    .num_decoder_layers(6)
    .dim_feedforward(512)
    .dropout(0.1)
    .build()?;
```

#### Efficient Transformer

```rust
let efficient_transformer = Transformer::builder()
    .horizon(24)
    .input_size(96)
    .d_model(64)          // Smaller model
    .num_heads(4)         // Fewer attention heads
    .num_encoder_layers(3) // Fewer layers
    .num_decoder_layers(3)
    .dim_feedforward(256)
    .dropout(0.1)
    .build()?;
```

## Informer

Efficient transformer for long sequences using ProbSparse attention and distillation mechanisms.

### Configuration

```rust
#[derive(Debug, Clone)]
pub struct InformerConfig {
    pub horizon: usize,
    pub input_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_encoder_layers: usize,
    pub num_decoder_layers: usize,
    pub factor: usize,
    pub distil: bool,
    pub attention_type: AttentionType,
}

#[derive(Debug, Clone)]
pub enum AttentionType {
    ProbSparse,
    Full,
}
```

### Usage Examples

```rust
let informer = Informer::builder()
    .horizon(96)          // Long horizon
    .input_size(384)      // Long input sequence
    .d_model(128)
    .num_heads(8)
    .num_encoder_layers(3)
    .num_decoder_layers(2)
    .factor(5)            // ProbSparse factor
    .distil(true)         // Enable distillation
    .attention_type(AttentionType::ProbSparse)
    .build()?;
```

### Key Features

- **ProbSparse Attention**: O(L log L) complexity instead of O(L²)
- **Distillation**: Reduces sequence length between layers
- **Long Sequence Efficiency**: Optimized for very long sequences

## Autoformer

Transformer with autocorrelation mechanism for better seasonal pattern modeling.

### Configuration

```rust
#[derive(Debug, Clone)]
pub struct AutoformerConfig {
    pub horizon: usize,
    pub input_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_encoder_layers: usize,
    pub num_decoder_layers: usize,
    pub moving_avg_kernel: usize,
    pub autocorr_factor: usize,
}
```

### Usage Examples

```rust
let autoformer = Autoformer::builder()
    .horizon(24)
    .input_size(96)
    .d_model(128)
    .num_heads(8)
    .num_encoder_layers(2)
    .num_decoder_layers(1)
    .moving_avg_kernel(25)
    .autocorr_factor(1)
    .build()?;
```

### Key Features

- **Auto-Correlation**: Discovers period-based dependencies
- **Decomposition**: Built-in trend/seasonal decomposition
- **Seasonal Modeling**: Specialized for seasonal patterns

## TFT (Temporal Fusion Transformer)

Advanced transformer with variable selection, interpretability, and multi-horizon capabilities.

### Configuration

```rust
#[derive(Debug, Clone)]
pub struct TFTConfig {
    pub horizon: usize,
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_encoder_layers: usize,
    pub dropout: f64,
    pub use_static_features: bool,
    pub use_future_features: bool,
    pub quantiles: Vec<f64>,
    pub add_relative_index: bool,
}
```

### Usage Examples

#### Full TFT Configuration

```rust
let tft = TFT::builder()
    .horizon(12)
    .input_size(48)
    .hidden_size(128)
    .num_heads(4)
    .num_encoder_layers(1)
    .dropout(0.1)
    .use_static_features(true)
    .use_future_features(true)
    .quantiles(vec![0.1, 0.5, 0.9])
    .add_relative_index(true)
    .build()?;
```

#### TFT for Financial Forecasting

```rust
let financial_tft = TFT::builder()
    .horizon(5)
    .input_size(60)
    .hidden_size(256)
    .num_heads(8)
    .num_encoder_layers(2)
    .dropout(0.2)
    .use_static_features(true)  // Company metadata
    .use_future_features(true)  // Economic indicators
    .quantiles(vec![0.05, 0.25, 0.5, 0.75, 0.95])  // Risk modeling
    .build()?;
```

### Key Features

- **Variable Selection**: Automatic feature importance
- **Interpretability**: Attention weights and feature importance
- **Probabilistic**: Native quantile forecasting
- **Multi-Horizon**: Efficient multi-step forecasting

## Training Considerations

### Memory Management

```rust
// Transformer models require careful memory management
let training_config = TrainingConfig::builder()
    .batch_size(16)       // Smaller batches for transformers
    .gradient_accumulation_steps(4)  // Effective batch size of 64
    .mixed_precision(true)  // Use FP16 to save memory
    .max_epochs(50)
    .learning_rate(0.0001)  // Lower learning rate
    .build()?;
```

### Attention Optimization

```rust
// Configure attention for efficiency
let transformer = Transformer::builder()
    .horizon(24)
    .input_size(96)
    .d_model(128)
    .num_heads(8)
    .attention_dropout(0.1)
    .flash_attention(true)  // Optimized attention implementation
    .build()?;
```

### Learning Rate Scheduling

```rust
// Warmup scheduler often works well for transformers
let scheduler_config = SchedulerConfig::WarmupCosineAnnealingLR {
    warmup_epochs: 10,
    max_epochs: 100,
    eta_min: 1e-6,
};
```

## Performance Comparison

### Computational Complexity

| Model | Attention Complexity | Memory Usage | Training Speed | Long Sequences |
|-------|---------------------|--------------|----------------|----------------|
| Transformer | O(L²) | High | Slow | Poor |
| Informer | O(L log L) | Medium | Medium | Excellent |
| Autoformer | O(L log L) | Medium | Medium | Good |
| TFT | O(L²) | High | Slow | Medium |

### Best Use Cases

- **Transformer**: Complex patterns with moderate sequence length
- **Informer**: Very long sequences (>1000 steps)
- **Autoformer**: Strong seasonal patterns
- **TFT**: Multivariate forecasting with interpretability needs

## Advanced Features

### Attention Visualization

```rust
// Extract attention weights for interpretation
let forecasts = tft.predict_with_attention(&data)?;
let attention_weights = forecasts.attention_weights()?;

// Visualize which time steps are most important
plot_attention_weights(&attention_weights)?;
```

### Variable Importance

```rust
// TFT provides variable importance scores
let importance_scores = tft.get_variable_importance()?;
for (variable, importance) in importance_scores {
    println!("{}: {:.4}", variable, importance);
}
```

### Multi-Quantile Forecasting

```rust
// Generate probabilistic forecasts
let tft = TFT::builder()
    .quantiles(vec![0.1, 0.25, 0.5, 0.75, 0.9])
    .build()?;

let forecasts = tft.predict(&data)?;
let quantile_forecasts = forecasts.quantile_forecasts()?;

// Access different quantiles
let median_forecast = quantile_forecasts.get("0.5")?;
let lower_bound = quantile_forecasts.get("0.1")?;
let upper_bound = quantile_forecasts.get("0.9")?;
```

## Integration Examples

### Multi-Scale Forecasting

```rust
// Use different transformers for different horizons
let short_horizon_model = Transformer::builder()
    .horizon(7)
    .input_size(28)
    .d_model(64)
    .num_heads(4)
    .build()?;

let long_horizon_model = Informer::builder()
    .horizon(56)
    .input_size(168)
    .d_model(128)
    .num_heads(8)
    .build()?;

let ensemble = NeuralForecast::builder()
    .with_model(Box::new(short_horizon_model))
    .with_model(Box::new(long_horizon_model))
    .build()?;
```

### Interpretable Business Forecasting

```rust
// TFT for interpretable business forecasting
let business_tft = TFT::builder()
    .horizon(12)
    .input_size(48)
    .hidden_size(160)
    .num_heads(4)
    .use_static_features(true)    // Product categories
    .use_future_features(true)    // Promotions, holidays
    .quantiles(vec![0.1, 0.5, 0.9]) // Risk assessment
    .add_relative_index(true)
    .build()?;

// Train and generate interpretable forecasts
let mut nf = NeuralForecast::builder()
    .with_model(Box::new(business_tft))
    .with_frequency(Frequency::Monthly)
    .build()?;

nf.fit(business_data)?;
let forecasts = nf.predict()?;

// Extract insights
let variable_importance = forecasts.get_variable_importance()?;
let attention_patterns = forecasts.get_attention_patterns()?;
```

Transformer models provide state-of-the-art capabilities for complex time series forecasting, with each variant optimized for different scenarios: standard Transformers for general use, Informer for long sequences, Autoformer for seasonal data, and TFT for interpretable multivariate forecasting.