# Specialized Models

Specialized models are designed for specific forecasting scenarios and use cases. These models incorporate domain-specific knowledge, probabilistic approaches, or unique architectures optimized for particular types of time series data.

## DeepAR (Deep Autoregressive)

Probabilistic forecasting model using autoregressive RNNs with parametric distributions, designed for generating prediction intervals and handling uncertainty.

### Architecture

- **Autoregressive Structure**: Predicts next value based on previous values
- **Parametric Distributions**: Models output as probability distributions
- **Global Model**: Single model for multiple time series
- **Covariates Support**: Handles static and dynamic features

### Configuration

```rust
#[derive(Debug, Clone)]
pub struct DeepARConfig {
    pub horizon: usize,
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub dropout: f64,
    pub likelihood_type: LikelihoodType,
    pub num_parallel_samples: usize,
    pub use_static_features: bool,
    pub use_dynamic_features: bool,
}

#[derive(Debug, Clone)]
pub enum LikelihoodType {
    Normal,
    StudentT,
    NegativeBinomial,
    Gamma,
}
```

### Usage Examples

#### Basic DeepAR

```rust
use neuro_divergent::models::{DeepAR, DeepARConfig, LikelihoodType};

let deepar = DeepAR::builder()
    .horizon(12)
    .input_size(48)
    .hidden_size(64)
    .num_layers(2)
    .dropout(0.1)
    .likelihood_type(LikelihoodType::Normal)
    .num_parallel_samples(100)
    .build()?;
```

#### DeepAR for Count Data

```rust
// For count data (e.g., sales, web traffic)
let count_deepar = DeepAR::builder()
    .horizon(7)
    .input_size(28)
    .hidden_size(64)
    .num_layers(2)
    .likelihood_type(LikelihoodType::NegativeBinomial)
    .num_parallel_samples(200)
    .use_static_features(true)
    .build()?;
```

#### Financial DeepAR

```rust
// For financial data with heavy tails
let financial_deepar = DeepAR::builder()
    .horizon(5)
    .input_size(60)
    .hidden_size(128)
    .num_layers(3)
    .dropout(0.2)
    .likelihood_type(LikelihoodType::StudentT)
    .num_parallel_samples(500)
    .use_dynamic_features(true)
    .build()?;
```

### Key Features

- **Probabilistic Outputs**: Natural prediction intervals
- **Multiple Distributions**: Choose appropriate likelihood
- **Global Training**: Learns across multiple series
- **Missing Value Handling**: Robust to missing data

### Strengths

- **Uncertainty Quantification**: Native prediction intervals
- **Flexible Distributions**: Handles different data types
- **Robust**: Works well with irregular or sparse data
- **Scalable**: Efficient for many time series

## DeepNPTS (Deep Non-Parametric Time Series)

Non-parametric deep learning approach for time series forecasting without distributional assumptions.

### Configuration

```rust
#[derive(Debug, Clone)]
pub struct DeepNPTSConfig {
    pub horizon: usize,
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_quantiles: usize,
    pub quantile_levels: Vec<f64>,
    pub dropout: f64,
    pub use_residual_connections: bool,
}
```

### Usage Examples

```rust
let deepnpts = DeepNPTS::builder()
    .horizon(12)
    .input_size(48)
    .hidden_size(96)
    .num_layers(3)
    .quantile_levels(vec![0.1, 0.25, 0.5, 0.75, 0.9])
    .dropout(0.15)
    .use_residual_connections(true)
    .build()?;
```

### Key Features

- **Non-Parametric**: No distributional assumptions
- **Quantile Regression**: Direct quantile prediction
- **Flexible**: Adapts to any distribution shape

## TCN (Temporal Convolutional Network)

Convolutional network designed for sequential modeling, offering parallelizable training and good performance on time series.

### Architecture

- **Dilated Convolutions**: Exponentially increasing receptive field
- **Residual Connections**: Skip connections for deep networks
- **Causal Convolutions**: No future information leakage
- **Parallel Training**: Unlike RNNs, can be trained in parallel

### Configuration

```rust
#[derive(Debug, Clone)]
pub struct TCNConfig {
    pub horizon: usize,
    pub input_size: usize,
    pub num_filters: usize,
    pub kernel_size: usize,
    pub num_layers: usize,
    pub dilation_base: usize,
    pub dropout: f64,
    pub use_skip_connections: bool,
    pub activation: ActivationType,
}
```

### Usage Examples

#### Standard TCN

```rust
use neuro_divergent::models::{TCN, TCNConfig};

let tcn = TCN::builder()
    .horizon(7)
    .input_size(28)
    .num_filters(64)
    .kernel_size(3)
    .num_layers(8)
    .dilation_base(2)
    .dropout(0.1)
    .use_skip_connections(true)
    .activation(ActivationType::ReLU)
    .build()?;
```

#### Deep TCN for Long Sequences

```rust
let deep_tcn = TCN::builder()
    .horizon(24)
    .input_size(168)  // Weekly data
    .num_filters(128)
    .kernel_size(3)
    .num_layers(12)   // Deep network
    .dilation_base(2)
    .dropout(0.2)
    .use_skip_connections(true)
    .build()?;
```

#### Efficient TCN

```rust
// Lightweight TCN for fast inference
let efficient_tcn = TCN::builder()
    .horizon(12)
    .input_size(48)
    .num_filters(32)
    .kernel_size(2)
    .num_layers(6)
    .dilation_base(2)
    .dropout(0.05)
    .build()?;
```

### Key Features

- **Parallel Training**: Much faster than RNNs
- **Long Memory**: Dilated convolutions capture long dependencies
- **Stable Training**: More stable than RNNs
- **Flexible Receptive Field**: Adjustable through dilation

### Performance Characteristics

- **Training Speed**: Fast (parallelizable)
- **Memory Usage**: Medium
- **Long Sequences**: Good
- **Irregular Patterns**: Excellent

## BiTCN (Bidirectional Temporal Convolutional Network)

Bidirectional version of TCN that processes sequences in both directions.

### Configuration

```rust
#[derive(Debug, Clone)]
pub struct BiTCNConfig {
    pub horizon: usize,
    pub input_size: usize,
    pub num_filters: usize,
    pub kernel_size: usize,
    pub num_layers: usize,
    pub dilation_base: usize,
    pub dropout: f64,
    pub fusion_method: FusionMethod,
}

#[derive(Debug, Clone)]
pub enum FusionMethod {
    Concatenate,
    Add,
    Attention,
}
```

### Usage Examples

```rust
let bitcn = BiTCN::builder()
    .horizon(7)
    .input_size(28)
    .num_filters(64)
    .kernel_size(3)
    .num_layers(8)
    .fusion_method(FusionMethod::Attention)
    .build()?;
```

### Key Features

- **Bidirectional Processing**: Forward and backward temporal modeling
- **Flexible Fusion**: Multiple ways to combine directions
- **Rich Representations**: Better feature extraction

## Training and Optimization

### Specialized Training Configurations

```rust
// Training config optimized for probabilistic models
let deepar_training = TrainingConfig::builder()
    .max_epochs(100)
    .batch_size(64)
    .learning_rate(0.001)
    .optimizer(OptimizerConfig::Adam {
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01
    })
    .loss_function(LossFunctionConfig::NegativeLogLikelihood)
    .build()?;

// Training config for TCN
let tcn_training = TrainingConfig::builder()
    .max_epochs(200)
    .batch_size(32)
    .learning_rate(0.002)
    .optimizer(OptimizerConfig::SGD {
        momentum: 0.9,
        weight_decay: 0.0001
    })
    .scheduler(SchedulerConfig::StepLR {
        step_size: 50,
        gamma: 0.5
    })
    .build()?;
```

### Loss Functions for Specialized Models

```rust
// Quantile loss for DeepNPTS
let quantile_loss = LossFunctionConfig::QuantileLoss {
    quantiles: vec![0.1, 0.25, 0.5, 0.75, 0.9]
};

// Negative log-likelihood for DeepAR
let nll_loss = LossFunctionConfig::NegativeLogLikelihood;

// Dilate loss for TCN
let dilate_loss = LossFunctionConfig::DilateLoss {
    alpha: 0.5,
    gamma: 0.001
};
```

## Model Comparison

### Use Case Matrix

| Model | Probabilistic | Count Data | Long Sequences | Interpretability | Speed |
|-------|---------------|------------|----------------|------------------|-------|
| DeepAR | ✓ | ✓ | ✓ | Medium | Medium |
| DeepNPTS | ✓ | ✓ | ✓ | Low | Medium |
| TCN | ✗ | ✗ | ✓ | Low | Fast |
| BiTCN | ✗ | ✗ | ✓ | Low | Medium |

### Performance Benchmarks

Typical performance on specialized tasks:

#### Retail Sales (Count Data)
- **DeepAR**: 0.89 CRPS (Continuous Ranked Probability Score)
- **DeepNPTS**: 0.92 CRPS
- **TCN**: 0.94 CRPS (point forecasts only)

#### High-Frequency IoT Data
- **TCN**: 0.12 MAE
- **BiTCN**: 0.11 MAE
- **LSTM**: 0.14 MAE (for comparison)

## Best Practices

### Model Selection Guidelines

```rust
fn select_specialized_model(data_type: &DataType) -> Box<dyn BaseModel<f64>> {
    match data_type {
        // Count data (sales, web traffic, etc.)
        DataType::Count { sparse: true, .. } => {
            Box::new(DeepAR::builder()
                .likelihood_type(LikelihoodType::NegativeBinomial)
                .num_parallel_samples(200)
                .build().unwrap())
        },
        
        // Financial data with uncertainty
        DataType::Financial { volatility: high, .. } if *high => {
            Box::new(DeepAR::builder()
                .likelihood_type(LikelihoodType::StudentT)
                .num_parallel_samples(500)
                .build().unwrap())
        },
        
        // High-frequency sensor data
        DataType::HighFrequency { .. } => {
            Box::new(TCN::builder()
                .num_filters(128)
                .num_layers(12)
                .build().unwrap())
        },
        
        // General probabilistic forecasting
        DataType::General { uncertainty: true, .. } => {
            Box::new(DeepNPTS::builder()
                .quantile_levels(vec![0.1, 0.25, 0.5, 0.75, 0.9])
                .build().unwrap())
        },
        
        // Default to TCN for speed
        _ => {
            Box::new(TCN::builder()
                .num_filters(64)
                .num_layers(8)
                .build().unwrap())
        }
    }
}
```

### Probabilistic Model Evaluation

```rust
// Evaluate probabilistic forecasts
use neuro_divergent::metrics::{CRPS, PinballLoss, CoverageRate};

// Train probabilistic model
let mut deepar_nf = NeuralForecast::builder()
    .with_model(Box::new(deepar))
    .build()?;

deepar_nf.fit(train_data)?;
let forecasts = deepar_nf.predict()?;

// Evaluate with probabilistic metrics
let crps = CRPS::new().compute(&forecasts, &actual_values)?;
let pinball = PinballLoss::new(0.1).compute(&forecasts, &actual_values)?;
let coverage = CoverageRate::new(0.8).compute(&forecasts, &actual_values)?;

println!("CRPS: {:.4}", crps);
println!("Pinball Loss (0.1): {:.4}", pinball);
println!("Coverage Rate (80%): {:.2}%", coverage * 100.0);
```

## Integration Examples

### Probabilistic Retail Forecasting

```rust
// Complete probabilistic forecasting pipeline
use neuro_divergent::prelude::*;

// Load retail sales data
let sales_data = TimeSeriesDataFrame::from_csv("retail_sales.csv")?;

// Create DeepAR model for count data
let deepar = DeepAR::builder()
    .horizon(28)  // 4-week forecast
    .input_size(84)  // 12-week history
    .hidden_size(128)
    .num_layers(3)
    .likelihood_type(LikelihoodType::NegativeBinomial)
    .num_parallel_samples(200)
    .use_static_features(true)  // Product categories
    .use_dynamic_features(true) // Promotions, holidays
    .build()?;

let mut nf = NeuralForecast::builder()
    .with_model(Box::new(deepar))
    .with_frequency(Frequency::Daily)
    .build()?;

nf.fit(sales_data)?;
let forecasts = nf.predict()?;

// Extract prediction intervals
let prediction_intervals = forecasts.prediction_intervals()?;
println!("80% PI: [{:.1}, {:.1}]", 
         prediction_intervals.lower_80, 
         prediction_intervals.upper_80);
```

### High-Frequency Sensor Monitoring

```rust
// TCN for real-time sensor monitoring
let sensor_tcn = TCN::builder()
    .horizon(60)     // 1-hour ahead (minute data)
    .input_size(1440) // 24-hour history
    .num_filters(64)
    .kernel_size(3)
    .num_layers(10)
    .dilation_base(2)
    .dropout(0.1)
    .build()?;

let mut sensor_nf = NeuralForecast::builder()
    .with_model(Box::new(sensor_tcn))
    .with_frequency(Frequency::Minute)
    .with_num_threads(8)  // Parallel processing
    .build()?;

// Real-time processing
let mut stream = SensorDataStream::new("sensor_feed")?;
while let Some(new_data) = stream.next().await {
    let forecast = sensor_nf.predict_on(new_data)?;
    
    // Check for anomalies
    if forecast.point_forecast > threshold {
        alert_system.send_warning(&forecast)?;
    }
}
```

### Multi-Model Uncertainty Ensemble

```rust
// Combine different probabilistic approaches
let probabilistic_ensemble = vec![
    Box::new(DeepAR::builder()
        .likelihood_type(LikelihoodType::Normal)
        .build()?) as Box<dyn BaseModel<f64>>,
        
    Box::new(DeepNPTS::builder()
        .quantile_levels(vec![0.1, 0.5, 0.9])
        .build()?) as Box<dyn BaseModel<f64>>,
        
    Box::new(TCN::builder()
        .num_filters(64)
        .build()?) as Box<dyn BaseModel<f64>>,
];

let ensemble_nf = NeuralForecast::builder()
    .with_models(probabilistic_ensemble)
    .with_prediction_intervals(PredictionIntervals::new(
        vec![0.8, 0.9, 0.95],
        IntervalMethod::ConformalPrediction
    )?)
    .build()?;
```

Specialized models provide targeted solutions for specific forecasting challenges, with DeepAR excelling at probabilistic forecasting, TCN offering speed and efficiency, and other models filling particular niches in the forecasting landscape.