# Advanced Models

Advanced models represent state-of-the-art forecasting architectures that incorporate sophisticated designs for complex temporal patterns, interpretability, and high performance on challenging datasets.

## NBEATS (Neural Basis Expansion Analysis for Time Series)

NBEATS is a interpretable neural architecture that decomposes time series into trend and seasonality components using learnable basis functions.

### Architecture

- **Stacking**: Multiple blocks organized in stacks
- **Block Types**: Trend and Seasonality blocks with specific basis functions
- **Forecasting**: Direct multi-step forecasting capability
- **Interpretability**: Decomposition into meaningful components

### Configuration

```rust
#[derive(Debug, Clone)]
pub struct NBEATSConfig {
    pub horizon: usize,
    pub input_size: usize,
    pub stack_types: Vec<StackType>,
    pub num_blocks: usize,
    pub num_layers: usize,
    pub layer_widths: Vec<usize>,
    pub share_weights_in_stack: bool,
    pub expansion_coefficient_dim: usize,
    pub trend_polynomial_degree: usize,
}

#[derive(Debug, Clone)]
pub enum StackType {
    Trend,
    Seasonality,
    Generic,
}
```

### Usage Examples

#### Basic NBEATS

```rust
use neuro_divergent::models::{NBEATS, NBEATSConfig, StackType};

let nbeats = NBEATS::builder()
    .horizon(12)
    .input_size(48)
    .stack_types(vec![StackType::Trend, StackType::Seasonality])
    .num_blocks(3)
    .num_layers(4)
    .layer_widths(vec![512, 512, 512, 512])
    .build()?;
```

#### Interpretable NBEATS

```rust
let interpretable_nbeats = NBEATS::builder()
    .horizon(24)
    .input_size(168)  // Weekly data
    .stack_types(vec![
        StackType::Trend,
        StackType::Seasonality,
        StackType::Generic
    ])
    .num_blocks(3)
    .share_weights_in_stack(true)
    .trend_polynomial_degree(3)
    .build()?;
```

### Key Features

- **Interpretable Decomposition**: Explicit trend and seasonal components
- **Multi-Step Forecasting**: Direct horizon forecasting without recursion
- **Hierarchical Structure**: Multiple stacks for different patterns
- **Strong Performance**: State-of-the-art accuracy on many datasets

## N-BEATS-X (Extended NBEATS)

Enhanced version of NBEATS with support for exogenous variables and improved architecture.

### Usage Examples

```rust
let nbeatsx = NBEATSX::builder()
    .horizon(12)
    .input_size(48)
    .num_exogenous_features(5)
    .stack_types(vec![StackType::Trend, StackType::Seasonality])
    .num_blocks(3)
    .exogenous_handling(ExogenousHandling::Concatenate)
    .build()?;
```

## N-HiTS (Neural Hierarchical Interpolation for Time Series)

N-HiTS extends NBEATS with hierarchical interpolation for improved efficiency on long horizons.

### Configuration

```rust
#[derive(Debug, Clone)]
pub struct NHiTSConfig {
    pub horizon: usize,
    pub input_size: usize,
    pub n_pool_kernel_size: Vec<usize>,
    pub n_freq_downsample: Vec<usize>,
    pub interpolation_mode: InterpolationMode,
    pub num_blocks: usize,
    pub num_layers: usize,
    pub layer_widths: Vec<usize>,
}
```

### Usage Examples

```rust
let nhits = NHiTS::builder()
    .horizon(96)  // Long horizon
    .input_size(384)
    .n_pool_kernel_size(vec![2, 2, 1])
    .n_freq_downsample(vec![8, 4, 1])
    .interpolation_mode(InterpolationMode::Linear)
    .num_blocks(3)
    .build()?;
```

### Key Features

- **Hierarchical Processing**: Multi-scale decomposition
- **Efficient Long Horizons**: Optimized for long-term forecasting
- **Interpolation**: Smart upsampling for forecast generation

## TSMixer (Time Series Mixing)

Simple yet effective architecture based on MLP-Mixer principles adapted for time series.

### Configuration

```rust
#[derive(Debug, Clone)]
pub struct TSMixerConfig {
    pub horizon: usize,
    pub input_size: usize,
    pub n_block: usize,
    pub ff_dim: usize,
    pub dropout: f64,
    pub norm_type: NormType,
    pub activation: ActivationType,
}
```

### Usage Examples

```rust
let tsmixer = TSMixer::builder()
    .horizon(12)
    .input_size(48)
    .n_block(8)
    .ff_dim(256)
    .dropout(0.1)
    .norm_type(NormType::BatchNorm)
    .activation(ActivationType::ReLU)
    .build()?;
```

## Training and Optimization

### Advanced Training Configurations

```rust
// Training config for advanced models
let training_config = TrainingConfig::builder()
    .max_epochs(100)
    .batch_size(32)
    .learning_rate(0.001)
    .optimizer(OptimizerConfig::Adam {
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01
    })
    .scheduler(SchedulerConfig::CosineAnnealingLR {
        t_max: 100,
        eta_min: 1e-6
    })
    .early_stopping_patience(15)
    .build()?;
```

### Loss Functions

```rust
// Specialized loss functions for advanced models
let loss_config = LossFunctionConfig::MASE {  // Mean Absolute Scaled Error
    seasonality: 24
};

// Or quantile loss for probabilistic forecasting
let quantile_loss = LossFunctionConfig::QuantileLoss {
    quantiles: vec![0.1, 0.5, 0.9]
};
```

## Model Comparison

### Performance Characteristics

| Model | Training Time | Memory Usage | Interpretability | Long Horizons |
|-------|---------------|--------------|------------------|---------------|
| NBEATS | Medium | Medium | High | Good |
| N-BEATS-X | Medium | Medium | High | Good |
| N-HiTS | Fast | Low | Medium | Excellent |
| TSMixer | Fast | Low | Low | Good |

### Best Use Cases

- **NBEATS**: When interpretability is crucial
- **N-BEATS-X**: NBEATS with exogenous variables
- **N-HiTS**: Long-horizon forecasting with efficiency
- **TSMixer**: Simple and effective baseline

## Integration Examples

### Interpretable Forecasting Pipeline

```rust
// Setup for interpretable forecasting
let interpretable_model = NBEATS::builder()
    .horizon(12)
    .input_size(48)
    .stack_types(vec![StackType::Trend, StackType::Seasonality])
    .num_blocks(3)
    .interpretable(true)
    .build()?;

let mut nf = NeuralForecast::builder()
    .with_model(Box::new(interpretable_model))
    .with_frequency(Frequency::Monthly)
    .build()?;

nf.fit(data)?;
let forecasts = nf.predict()?;

// Extract interpretable components
let trend_component = forecasts.get_trend_component()?;
let seasonal_component = forecasts.get_seasonal_component()?;
```

### Long-Horizon Forecasting

```rust
// Optimized for long-term forecasting
let long_horizon_model = NHiTS::builder()
    .horizon(96)  // 4 days hourly
    .input_size(672)  // 4 weeks hourly
    .n_pool_kernel_size(vec![4, 2, 1])
    .n_freq_downsample(vec![8, 4, 1])
    .interpolation_mode(InterpolationMode::Linear)
    .build()?;
```

Advanced models provide sophisticated capabilities for complex forecasting scenarios, with NBEATS offering interpretability, N-HiTS optimizing for long horizons, and TSMixer providing a simple yet effective alternative.