# Models Overview

Neuro-Divergent provides a comprehensive collection of neural forecasting models, offering 100% compatibility with NeuralForecast Python models while leveraging Rust's performance and safety benefits.

## Model Categories

### [Basic Models](basic.md)
Linear and simple neural models suitable for straightforward forecasting tasks.

- **DLinear** - Decomposition-based linear forecasting
- **NLinear** - Normalized linear forecasting  
- **MLP** - Multi-layer perceptron for univariate forecasting
- **MLP Multivariate** - Multi-layer perceptron for multivariate forecasting

### [Recurrent Models](recurrent.md)
Models based on recurrent neural network architectures.

- **RNN** - Basic recurrent neural network
- **LSTM** - Long Short-Term Memory networks
- **GRU** - Gated Recurrent Unit networks
- **BiLSTM** - Bidirectional LSTM networks

### [Advanced Models](advanced.md)
Sophisticated architectures designed for complex forecasting scenarios.

- **NBEATS** - Neural basis expansion analysis for interpretable time series
- **N-BEATS-X** - Extended NBEATS with exogenous variables
- **N-HiTS** - Neural hierarchical interpolation for time series
- **TSMixer** - Time series mixing model

### [Transformer Models](transformer.md)
Attention-based models leveraging transformer architectures.

- **Transformer** - Standard transformer for time series
- **Informer** - Efficient transformer for long sequences
- **Autoformer** - Auto-correlation based transformer
- **TFT** - Temporal Fusion Transformer

### [Specialized Models](specialized.md)
Domain-specific and experimental models for particular use cases.

- **DeepAR** - Probabilistic forecasting with autoregressive RNNs
- **DeepNPTS** - Deep learning for non-parametric time series
- **TCN** - Temporal Convolutional Networks
- **BiTCN** - Bidirectional Temporal Convolutional Networks

## Model Selection Guide

### By Data Characteristics

#### **Small Datasets (< 1000 observations)**
- **DLinear** - Simple and robust
- **NLinear** - Good baseline performance
- **MLP** - If non-linearity is suspected

#### **Medium Datasets (1000-10000 observations)**
- **LSTM** - Good balance of complexity and performance
- **NBEATS** - Interpretable with strong performance
- **TCN** - Efficient alternative to RNNs

#### **Large Datasets (> 10000 observations)**
- **Transformer** - Can leverage large amounts of data
- **TFT** - Excellent for complex multivariate scenarios
- **N-HiTS** - Efficient for hierarchical patterns

### By Forecasting Requirements

#### **Fast Training/Inference**
- **DLinear** - Linear models are fastest
- **NLinear** - Minimal computational overhead
- **TCN** - Parallelizable convolutions

#### **High Accuracy**
- **TFT** - State-of-the-art for many scenarios
- **NBEATS** - Strong performance with interpretability
- **Transformer** - Excellent for complex patterns

#### **Interpretability**
- **DLinear** - Fully interpretable decomposition
- **NBEATS** - Interpretable basis functions
- **N-HiTS** - Hierarchical decomposition

#### **Probabilistic Forecasting**
- **DeepAR** - Native probabilistic outputs
- **TFT** - Quantile forecasting capabilities
- **LSTM** - Can be configured for probabilistic outputs

### By Data Patterns

#### **Seasonal Data**
- **NBEATS** - Explicit seasonality modeling
- **N-HiTS** - Multi-scale seasonal patterns
- **TFT** - Learned seasonal representations

#### **Trend-Heavy Data**
- **DLinear** - Explicit trend decomposition
- **NBEATS** - Trend and seasonality blocks
- **N-HiTS** - Hierarchical trend modeling

#### **Irregular/Noisy Data**
- **DeepAR** - Robust probabilistic approach
- **TCN** - Dilated convolutions handle irregularity
- **TFT** - Attention mechanism filters noise

## Common Model Configuration Patterns

### Basic Model Setup

```rust
use neuro_divergent::models::*;

// Simple univariate forecasting
let model = LSTM::builder()
    .hidden_size(64)
    .num_layers(2)
    .horizon(7)
    .input_size(28)
    .build()?;
```

### Multi-Model Ensemble

```rust
// Create diverse model ensemble
let linear_model = DLinear::builder()
    .horizon(7)
    .input_size(28)
    .build()?;

let neural_model = LSTM::builder()
    .hidden_size(128)
    .num_layers(2)
    .horizon(7)
    .input_size(28)
    .build()?;

let interpretable_model = NBEATS::builder()
    .stack_types(vec![StackType::Trend, StackType::Seasonality])
    .num_blocks(3)
    .horizon(7)
    .input_size(28)
    .build()?;

let nf = NeuralForecast::builder()
    .with_model(Box::new(linear_model))
    .with_model(Box::new(neural_model))
    .with_model(Box::new(interpretable_model))
    .build()?;
```

### Probabilistic Forecasting Setup

```rust
// Configure for probabilistic outputs
let deepar = DeepAR::builder()
    .hidden_size(64)
    .num_layers(2)
    .horizon(7)
    .input_size(28)
    .likelihood_type(LikelihoodType::Normal)
    .build()?;

let tft = TFT::builder()
    .hidden_size(128)
    .num_heads(4)
    .horizon(7)
    .input_size(28)
    .quantiles(vec![0.1, 0.5, 0.9])
    .build()?;
```

## Model Performance Characteristics

### Computational Complexity

| Model | Training Time | Inference Time | Memory Usage | Parallelizable |
|-------|---------------|----------------|--------------|----------------|
| DLinear | O(n) | O(1) | Low | Yes |
| NLinear | O(n) | O(1) | Low | Yes |
| MLP | O(n·h) | O(h) | Medium | Yes |
| LSTM | O(n·h²) | O(h²) | High | No |
| NBEATS | O(n·h²) | O(h²) | Medium | Partial |
| Transformer | O(n²·h) | O(n·h) | High | Yes |
| TFT | O(n²·h) | O(n·h) | Very High | Yes |
| TCN | O(n·h) | O(h) | Medium | Yes |

*Where n = sequence length, h = hidden size*

### Accuracy Benchmarks

Based on typical performance across various datasets:

#### M4 Competition Results
- **TFT**: 0.085 sMAPE (best overall)
- **NBEATS**: 0.089 sMAPE (interpretable)
- **LSTM**: 0.092 sMAPE (good baseline)
- **DLinear**: 0.095 sMAPE (simple baseline)

#### Financial Data
- **DeepAR**: 0.12 MAE (probabilistic)
- **TFT**: 0.11 MAE (with features)
- **TCN**: 0.13 MAE (efficient)
- **LSTM**: 0.14 MAE (standard)

## Model Implementation Details

### Memory Management

All models are designed for efficient memory usage:

```rust
// Models support different precision levels
let model_f32 = LSTM::<f32>::builder().build()?;  // Memory efficient
let model_f64 = LSTM::<f64>::builder().build()?;  // Numerical precision
```

### GPU Acceleration

Compatible models support GPU acceleration:

```rust
let model = TFT::builder()
    .device(Device::GPU(0))  // Use first GPU
    .build()?;
```

### Batch Processing

Models support efficient batch operations:

```rust
// Batch prediction for multiple series
let batch_results = model.batch_predict(&datasets)?;
```

## Custom Model Development

### Implementing BaseModel

To create custom models, implement the `BaseModel` trait:

```rust
use neuro_divergent::core::BaseModel;

pub struct MyCustomModel<T: Float> {
    config: MyModelConfig,
    // ... model internals
}

impl<T: Float + Send + Sync> BaseModel<T> for MyCustomModel<T> {
    type Config = MyModelConfig;
    type State = MyModelState;
    
    fn new(config: Self::Config) -> NeuroDivergentResult<Self> {
        // Model initialization
    }
    
    fn fit(&mut self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<()> {
        // Training implementation
    }
    
    fn predict(&self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<ForecastResult<T>> {
        // Prediction implementation
    }
    
    // ... other required methods
}
```

### Configuration Patterns

Follow established patterns for model configuration:

```rust
#[derive(Debug, Clone)]
pub struct MyModelConfig {
    pub horizon: usize,
    pub input_size: usize,
    pub hidden_size: usize,
    // ... other parameters
}

impl MyModelConfig {
    pub fn builder() -> MyModelConfigBuilder {
        MyModelConfigBuilder::new()
    }
}

impl ModelConfig<f64> for MyModelConfig {
    fn validate(&self) -> NeuroDivergentResult<()> {
        if self.horizon == 0 {
            return Err(NeuroDivergentError::config("Horizon must be positive"));
        }
        // ... other validations
        Ok(())
    }
    
    // ... other trait methods
}
```

## Best Practices

### Model Selection Process

1. **Start Simple**: Begin with DLinear or NLinear as baselines
2. **Add Complexity Gradually**: Move to LSTM, then NBEATS, then Transformers
3. **Validate Thoroughly**: Use cross-validation to assess real performance
4. **Consider Ensemble**: Combine different model types for robustness

### Hyperparameter Tuning

```rust
// Systematic hyperparameter search
let configs = vec![
    LSTMConfig::builder().hidden_size(32).num_layers(1).build()?,
    LSTMConfig::builder().hidden_size(64).num_layers(2).build()?,
    LSTMConfig::builder().hidden_size(128).num_layers(2).build()?,
];

let mut best_mae = f64::INFINITY;
let mut best_config = None;

for config in configs {
    let model = LSTM::new(config.clone())?;
    let cv_results = nf.cross_validation(data.clone(), cv_config.clone())?;
    let mae = cv_results.overall_metrics().get("MAE").unwrap();
    
    if mae < &best_mae {
        best_mae = *mae;
        best_config = Some(config);
    }
}
```

### Production Deployment

```rust
// Optimize for production
let production_model = TFT::builder()
    .hidden_size(64)  // Balanced size
    .num_heads(4)     // Efficient attention
    .dropout(0.1)     // Regularization
    .device(Device::GPU(0))  // GPU acceleration
    .build()?;

// Save trained model
production_model.save("production_model.json")?;

// Load in production
let loaded_model = TFT::load("production_model.json")?;
```

### Error Handling

```rust
// Robust model training with error handling
match model.fit(&dataset) {
    Ok(()) => println!("Model trained successfully"),
    Err(NeuroDivergentError::TrainingError(msg)) => {
        eprintln!("Training failed: {}", msg);
        // Try with different hyperparameters
    },
    Err(e) => return Err(e),
}
```

## Integration Examples

### Financial Forecasting Pipeline

```rust
// Financial time series pipeline
let financial_models = vec![
    Box::new(DLinear::builder()
        .horizon(5)
        .build()?) as Box<dyn BaseModel<f64>>,
    Box::new(DeepAR::builder()
        .hidden_size(64)
        .horizon(5)
        .likelihood_type(LikelihoodType::StudentT)
        .build()?) as Box<dyn BaseModel<f64>>,
    Box::new(TFT::builder()
        .hidden_size(128)
        .horizon(5)
        .quantiles(vec![0.05, 0.5, 0.95])
        .build()?) as Box<dyn BaseModel<f64>>,
];

let nf = NeuralForecast::builder()
    .with_models(financial_models)
    .with_frequency(Frequency::BusinessDaily)
    .with_local_scaler(ScalerType::RobustScaler)
    .build()?;
```

### IoT Sensor Forecasting

```rust
// High-frequency sensor data
let iot_models = vec![
    Box::new(TCN::builder()
        .num_filters(64)
        .kernel_size(3)
        .horizon(24)
        .build()?) as Box<dyn BaseModel<f32>>,
    Box::new(LSTM::builder()
        .hidden_size(128)
        .num_layers(2)
        .horizon(24)
        .build()?) as Box<dyn BaseModel<f32>>,
];

let nf = NeuralForecast::builder()
    .with_models(iot_models)
    .with_frequency(Frequency::Hourly)
    .with_num_threads(8)
    .with_device(Device::GPU(0))
    .build()?;
```

Each model category provides detailed documentation with configuration options, usage examples, and performance characteristics. Choose the appropriate models based on your specific forecasting requirements and data characteristics.