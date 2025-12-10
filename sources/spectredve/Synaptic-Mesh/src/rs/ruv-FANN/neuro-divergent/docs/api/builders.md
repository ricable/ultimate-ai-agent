# Builder Patterns and Fluent API

Neuro-Divergent extensively uses the builder pattern to provide a fluent, ergonomic API for configuring models, training setups, and data processing pipelines. This approach ensures type safety, provides sensible defaults, and enables easy configuration validation.

## Overview

Builder patterns are used throughout the library for:

- **Model Configuration**: Creating and configuring neural network models
- **Data Processing**: Setting up data transformation pipelines
- **Training Configuration**: Configuring optimizers, schedulers, and callbacks
- **Evaluation Setup**: Configuring metrics and cross-validation

## Core Builder Concepts

### Basic Builder Pattern

All builders follow a consistent pattern:

```rust
// Standard builder pattern
let config = ModelConfig::builder()
    .parameter1(value1)
    .parameter2(value2)
    .build()?;  // Returns Result<Config, Error>

// Alternative with new()
let model = Model::new(config)?;

// Or combined
let model = Model::builder()
    .parameter1(value1)
    .parameter2(value2)
    .build()?;
```

### Validation at Build Time

```rust
// Validation happens during build()
let result = LSTMConfig::builder()
    .horizon(0)  // Invalid value
    .build();

match result {
    Ok(config) => { /* use config */ },
    Err(NeuroDivergentError::ConfigError(msg)) => {
        eprintln!("Configuration error: {}", msg);
    }
}
```

## Model Builders

### LSTM Builder

```rust
use neuro_divergent::models::{LSTM, LSTMConfig};

// Basic LSTM
let lstm = LSTM::builder()
    .horizon(7)
    .input_size(28)
    .build()?;

// Advanced LSTM configuration
let advanced_lstm = LSTM::builder()
    .horizon(12)
    .input_size(48)
    .hidden_size(128)
    .num_layers(3)
    .dropout(0.2)
    .bidirectional(false)
    .use_bias(true)
    .batch_first(true)
    .activation(ActivationType::Tanh)
    .recurrent_activation(ActivationType::Sigmoid)
    .build()?;

// Configuration-based approach
let config = LSTMConfig::builder()
    .horizon(7)
    .input_size(28)
    .hidden_size(64)
    .num_layers(2)
    .dropout(0.1)
    .build()?;

let lstm = LSTM::new(config)?;
```

### NBEATS Builder

```rust
use neuro_divergent::models::{NBEATS, StackType};

// Interpretable NBEATS
let nbeats = NBEATS::builder()
    .horizon(12)
    .input_size(48)
    .stack_types(vec![StackType::Trend, StackType::Seasonality])
    .num_blocks(3)
    .num_layers(4)
    .layer_widths(vec![512, 512, 512, 512])
    .share_weights_in_stack(true)
    .expansion_coefficient_dim(32)
    .trend_polynomial_degree(3)
    .build()?;

// Generic NBEATS
let generic_nbeats = NBEATS::builder()
    .horizon(24)
    .input_size(168)
    .stack_types(vec![
        StackType::Generic,
        StackType::Generic,
        StackType::Generic
    ])
    .num_blocks(3)
    .num_layers(4)
    .layer_widths(vec![256, 256, 256, 256])
    .build()?;
```

### Transformer Builder

```rust
use neuro_divergent::models::Transformer;

// Standard transformer
let transformer = Transformer::builder()
    .horizon(12)
    .input_size(48)
    .d_model(128)
    .num_heads(8)
    .num_encoder_layers(6)
    .num_decoder_layers(6)
    .dim_feedforward(512)
    .dropout(0.1)
    .activation(ActivationType::ReLU)
    .use_positional_encoding(true)
    .build()?;

// Efficient transformer for production
let efficient_transformer = Transformer::builder()
    .horizon(24)
    .input_size(96)
    .d_model(64)          // Smaller model dimension
    .num_heads(4)         // Fewer attention heads
    .num_encoder_layers(3) // Fewer layers
    .num_decoder_layers(3)
    .dim_feedforward(256)
    .dropout(0.1)
    .attention_dropout(0.1)
    .flash_attention(true)  // Optimized attention
    .build()?;
```

### TFT Builder

```rust
use neuro_divergent::models::TFT;

// Full TFT with all features
let tft = TFT::builder()
    .horizon(12)
    .input_size(48)
    .hidden_size(128)
    .num_heads(4)
    .num_encoder_layers(1)
    .dropout(0.1)
    .use_static_features(true)
    .use_future_features(true)
    .use_historical_features(true)
    .quantiles(vec![0.1, 0.5, 0.9])
    .add_relative_index(true)
    .variable_selection_threshold(0.01)
    .build()?;

// Simple TFT for point forecasting
let simple_tft = TFT::builder()
    .horizon(7)
    .input_size(28)
    .hidden_size(64)
    .num_heads(2)
    .use_static_features(false)
    .use_future_features(false)
    .build()?;
```

## NeuralForecast Builder

### Basic NeuralForecast Setup

```rust
use neuro_divergent::{NeuralForecast, Frequency, ScalerType, Device};

// Single model forecasting
let nf = NeuralForecast::builder()
    .with_model(Box::new(lstm))
    .with_frequency(Frequency::Daily)
    .build()?;

// Multi-model ensemble
let ensemble_nf = NeuralForecast::builder()
    .with_models(vec![
        Box::new(lstm),
        Box::new(nbeats),
        Box::new(transformer),
    ])
    .with_frequency(Frequency::Hourly)
    .with_local_scaler(ScalerType::StandardScaler)
    .with_num_threads(8)
    .with_device(Device::GPU(0))
    .build()?;
```

### Advanced NeuralForecast Configuration

```rust
use neuro_divergent::config::{PredictionIntervals, IntervalMethod};

// Production forecasting setup
let production_nf = NeuralForecast::builder()
    .with_models(production_models)
    .with_frequency(Frequency::BusinessDaily)
    .with_local_scaler(ScalerType::RobustScaler)  // Robust to outliers
    .with_num_threads(16)
    .with_prediction_intervals(
        PredictionIntervals::new(
            vec![0.8, 0.9, 0.95],
            IntervalMethod::ConformalPrediction
        )?
    )
    .with_device(Device::GPU(0))
    .validation_enabled(true)
    .early_stopping_patience(15)
    .build()?;
```

## Training Configuration Builders

### Training Config Builder

```rust
use neuro_divergent::training::{TrainingConfig, OptimizerConfig, SchedulerConfig};

// Basic training configuration
let training_config = TrainingConfig::builder()
    .max_epochs(100)
    .batch_size(32)
    .learning_rate(0.001)
    .build()?;

// Advanced training setup
let advanced_training = TrainingConfig::builder()
    .max_epochs(200)
    .batch_size(64)
    .learning_rate(0.001)
    .optimizer(OptimizerConfig::Adam {
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01,
    })
    .scheduler(SchedulerConfig::CosineAnnealingLR {
        t_max: 200,
        eta_min: 1e-6,
    })
    .early_stopping_patience(20)
    .gradient_clipping(1.0)
    .validation_split(0.2)
    .mixed_precision(true)
    .build()?;
```

### Optimizer Builder

```rust
use neuro_divergent::training::{Adam, SGD, AdamW};

// Adam optimizer
let adam = Adam::builder()
    .learning_rate(0.001)
    .beta1(0.9)
    .beta2(0.999)
    .epsilon(1e-8)
    .weight_decay(0.01)
    .amsgrad(false)
    .build()?;

// SGD with momentum
let sgd = SGD::builder()
    .learning_rate(0.01)
    .momentum(0.9)
    .dampening(0.0)
    .weight_decay(0.0001)
    .nesterov(true)
    .build()?;

// AdamW for transformers
let adamw = AdamW::builder()
    .learning_rate(0.0001)
    .beta1(0.9)
    .beta2(0.999)
    .weight_decay(0.01)
    .build()?;
```

## Data Processing Builders

### TimeSeriesDataFrame Builder

```rust
use neuro_divergent::data::{TimeSeriesDataFrame, TimeSeriesSchema};

// Schema builder
let schema = TimeSeriesSchema::builder()
    .unique_id_col("series_id")
    .ds_col("timestamp")
    .y_col("value")
    .static_features(vec!["category".to_string(), "region".to_string()])
    .exogenous_features(vec!["temperature".to_string(), "holiday".to_string()])
    .build()?;

// DataFrameBuilder (conceptual)
let df = TimeSeriesDataFrame::builder()
    .from_csv("data.csv")
    .schema(schema)
    .frequency(Frequency::Daily)
    .validate_on_load(true)
    .build()?;
```

### Feature Engineering Builder

```rust
use neuro_divergent::features::{FeatureEngine, LagsConfig, RollingConfig};

// Feature engineering pipeline
let features = FeatureEngine::builder()
    .lags(LagsConfig::builder()
        .lags(vec![1, 2, 3, 7, 14])
        .build()?)
    .rolling_stats(RollingConfig::builder()
        .windows(vec![7, 14, 30])
        .stats(vec!["mean", "std", "min", "max"])
        .build()?)
    .temporal_features(true)
    .calendar_features(true)
    .build()?;

let enhanced_data = features.transform(&raw_data)?;
```

## Cross-Validation Builders

### Cross-Validation Config

```rust
use neuro_divergent::evaluation::CrossValidationConfig;

// Time series cross-validation
let cv_config = CrossValidationConfig::builder()
    .n_windows(5)
    .horizon(12)
    .step_size(6)
    .test_size(24)
    .season_length(12)
    .refit(true)
    .random_seed(42)
    .build()?;

// Expanding window CV
let expanding_cv = CrossValidationConfig::builder()
    .strategy(CVStrategy::ExpandingWindow)
    .initial_train_size(200)
    .horizon(12)
    .step_size(12)
    .max_windows(10)
    .build()?;
```

## Evaluation Builders

### Model Evaluator Builder

```rust
use neuro_divergent::evaluation::{ModelEvaluator, MetricConfig};

// Comprehensive evaluation setup
let evaluator = ModelEvaluator::builder()
    .point_metrics(vec![
        MetricConfig::MAE,
        MetricConfig::RMSE,
        MetricConfig::MAPE,
        MetricConfig::MASE { seasonality: 12 },
    ])
    .probabilistic_metrics(vec![
        MetricConfig::CRPS,
        MetricConfig::PinballLoss { quantiles: vec![0.1, 0.9] },
    ])
    .cross_validation(cv_config)
    .significance_level(0.05)
    .bootstrap_samples(1000)
    .build()?;
```

## Custom Builder Implementation

### Creating Custom Builders

```rust
// Example: Custom model builder
#[derive(Debug, Default)]
pub struct MyModelBuilder {
    horizon: Option<usize>,
    input_size: Option<usize>,
    hidden_size: Option<usize>,
    dropout: Option<f64>,
    custom_param: Option<String>,
}

impl MyModelBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn horizon(mut self, horizon: usize) -> Self {
        self.horizon = Some(horizon);
        self
    }
    
    pub fn input_size(mut self, input_size: usize) -> Self {
        self.input_size = Some(input_size);
        self
    }
    
    pub fn hidden_size(mut self, hidden_size: usize) -> Self {
        self.hidden_size = Some(hidden_size);
        self
    }
    
    pub fn dropout(mut self, dropout: f64) -> Self {
        self.dropout = Some(dropout);
        self
    }
    
    pub fn custom_param(mut self, param: impl Into<String>) -> Self {
        self.custom_param = Some(param.into());
        self
    }
    
    pub fn build(self) -> NeuroDivergentResult<MyModel> {
        let config = MyModelConfig {
            horizon: self.horizon.ok_or_else(|| 
                NeuroDivergentError::config("horizon is required"))?,
            input_size: self.input_size.ok_or_else(|| 
                NeuroDivergentError::config("input_size is required"))?,
            hidden_size: self.hidden_size.unwrap_or(64),
            dropout: self.dropout.unwrap_or(0.1),
            custom_param: self.custom_param.unwrap_or_else(|| "default".to_string()),
        };
        
        // Validate configuration
        config.validate()?;
        
        MyModel::new(config)
    }
}

// Enable builder() method on the model
impl MyModel {
    pub fn builder() -> MyModelBuilder {
        MyModelBuilder::new()
    }
}
```

### Builder with Conditional Configuration

```rust
pub struct ConditionalModelBuilder {
    base_config: BaseModelConfig,
    model_type: Option<ModelType>,
}

impl ConditionalModelBuilder {
    pub fn for_model_type(mut self, model_type: ModelType) -> Self {
        self.model_type = Some(model_type);
        self
    }
    
    pub fn build(self) -> NeuroDivergentResult<Box<dyn BaseModel<f64>>> {
        let model_type = self.model_type.ok_or_else(|| 
            NeuroDivergentError::config("model_type is required"))?;
        
        match model_type {
            ModelType::LSTM => {
                let lstm_config = LSTMConfig::from_base_config(&self.base_config)?;
                Ok(Box::new(LSTM::new(lstm_config)?))
            },
            ModelType::NBEATS => {
                let nbeats_config = NBEATSConfig::from_base_config(&self.base_config)?;
                Ok(Box::new(NBEATS::new(nbeats_config)?))
            },
            ModelType::Transformer => {
                let transformer_config = TransformerConfig::from_base_config(&self.base_config)?;
                Ok(Box::new(Transformer::new(transformer_config)?))
            },
        }
    }
}
```

## Builder Best Practices

### Validation Strategy

```rust
impl ModelConfigBuilder {
    pub fn build(self) -> NeuroDivergentResult<ModelConfig> {
        // Step 1: Check required fields
        let horizon = self.horizon.ok_or_else(|| 
            NeuroDivergentError::config("horizon is required"))?;
        
        // Step 2: Apply defaults
        let hidden_size = self.hidden_size.unwrap_or(64);
        let dropout = self.dropout.unwrap_or(0.1);
        
        // Step 3: Cross-validate parameters
        if horizon > self.input_size.unwrap_or(1) {
            return Err(NeuroDivergentError::config(
                "horizon cannot be larger than input_size"
            ));
        }
        
        // Step 4: Create and validate config
        let config = ModelConfig {
            horizon,
            hidden_size,
            dropout,
            // ... other fields
        };
        
        config.validate()?;
        Ok(config)
    }
}
```

### Fluent API Patterns

```rust
// Method chaining with validation
impl NeuralForecastBuilder {
    pub fn add_model_if<P>(mut self, predicate: P, model: Box<dyn BaseModel<f64>>) -> Self 
    where 
        P: FnOnce() -> bool,
    {
        if predicate() {
            self.models.push(model);
        }
        self
    }
    
    pub fn configure_for_scenario(mut self, scenario: ForecastingScenario) -> Self {
        match scenario {
            ForecastingScenario::HighFrequency => {
                self.local_scaler_type = Some(ScalerType::RobustScaler);
                self.num_threads = Some(8);
            },
            ForecastingScenario::Financial => {
                self.local_scaler_type = Some(ScalerType::StandardScaler);
                self.prediction_intervals = Some(PredictionIntervals::new(
                    vec![0.95, 0.99], 
                    IntervalMethod::ConformalPrediction
                ).unwrap());
            },
            ForecastingScenario::IoT => {
                self.device = Device::GPU(0);
                self.num_threads = Some(16);
            },
        }
        self
    }
}

// Usage
let nf = NeuralForecast::builder()
    .configure_for_scenario(ForecastingScenario::Financial)
    .add_model_if(|| use_lstm, Box::new(lstm_model))
    .add_model_if(|| use_transformer, Box::new(transformer_model))
    .with_frequency(Frequency::Daily)
    .build()?;
```

### Type-Safe Builder Patterns

```rust
// Phantom types for type-safe builders
use std::marker::PhantomData;

pub struct ModelBuilder<State> {
    config: ModelConfig,
    _state: PhantomData<State>,
}

pub struct NeedsHorizon;
pub struct NeedsInputSize;
pub struct Ready;

impl ModelBuilder<NeedsHorizon> {
    pub fn new() -> Self {
        Self {
            config: ModelConfig::default(),
            _state: PhantomData,
        }
    }
    
    pub fn horizon(mut self, horizon: usize) -> ModelBuilder<NeedsInputSize> {
        self.config.horizon = horizon;
        ModelBuilder {
            config: self.config,
            _state: PhantomData,
        }
    }
}

impl ModelBuilder<NeedsInputSize> {
    pub fn input_size(mut self, input_size: usize) -> ModelBuilder<Ready> {
        self.config.input_size = input_size;
        ModelBuilder {
            config: self.config,
            _state: PhantomData,
        }
    }
}

impl ModelBuilder<Ready> {
    pub fn hidden_size(mut self, hidden_size: usize) -> Self {
        self.config.hidden_size = hidden_size;
        self
    }
    
    pub fn build(self) -> NeuroDivergentResult<Model> {
        Model::new(self.config)
    }
}

// Usage - enforces required parameters at compile time
let model = Model::builder()
    .horizon(7)        // Required first
    .input_size(28)    // Required second
    .hidden_size(64)   // Optional
    .build()?;         // Only available after required params
```

The builder pattern system in Neuro-Divergent provides a powerful, type-safe, and user-friendly way to configure complex forecasting pipelines while ensuring validation and maintaining flexibility for different use cases.