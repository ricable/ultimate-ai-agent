# Configuration System

The configuration system in Neuro-Divergent provides comprehensive management of model parameters, training settings, data processing options, and system-level configurations through a type-safe, hierarchical approach.

## Overview

The configuration system consists of several layers:

- **Model Configurations**: Specific to each model type
- **Training Configurations**: Training algorithms, optimizers, loss functions
- **Data Configurations**: Preprocessing, validation, feature engineering
- **System Configurations**: Hardware, performance, logging settings

## Core Configuration Types

### Frequency Configuration

Time series frequency specification compatible with pandas/NeuralForecast.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Frequency {
    // High frequency
    Nanosecond,     // "ns"
    Microsecond,    // "us"  
    Millisecond,    // "ms"
    Second,         // "s"
    Minute,         // "min"
    
    // Common frequencies
    Hourly,         // "H"
    Daily,          // "D"
    Weekly,         // "W"
    Monthly,        // "M"
    Quarterly,      // "Q"
    Yearly,         // "Y"
    
    // Business frequencies
    BusinessDaily,    // "B"
    BusinessMonthly,  // "BM"
    BusinessQuarterly, // "BQ"
    
    // Custom frequency
    Custom(String),
}
```

#### Usage Examples

```rust
use neuro_divergent::config::Frequency;

// Parse from string (pandas compatible)
let freq = Frequency::from_str("D")?;  // Daily
let freq = Frequency::from_str("H")?;  // Hourly
let freq = Frequency::from_str("M")?;  // Monthly

// Get pandas-compatible string
println!("{}", Frequency::Daily.to_pandas_str());  // "D"

// Get duration between periods
let duration = Frequency::Hourly.duration();  // 1 hour

// Check if business frequency
let is_business = Frequency::BusinessDaily.is_business();  // true

// Get seasonal period
let seasonal = Frequency::Daily.seasonal_period();  // Some(7) for weekly seasonality
```

### Scaler Configuration

Data preprocessing and scaling options.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalerType {
    StandardScaler,    // Zero mean, unit variance
    MinMaxScaler,      // Scale to [0, 1]
    RobustScaler,      // Median and IQR
    IdentityScaler,    // No scaling
}
```

#### Usage Examples

```rust
use neuro_divergent::config::ScalerType;

// Choose scaler based on data characteristics
let scaler = if data.has_outliers() {
    ScalerType::RobustScaler
} else if data.is_non_negative() {
    ScalerType::MinMaxScaler
} else {
    ScalerType::StandardScaler
};

// Get all available scalers
let all_scalers = ScalerType::all();
```

### Prediction Intervals Configuration

Configuration for probabilistic forecasting and uncertainty quantification.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionIntervals {
    pub confidence_levels: Vec<f64>,
    pub method: IntervalMethod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntervalMethod {
    Quantile,              // Quantile regression
    ConformalPrediction,   // Conformal prediction
    Bootstrap,             // Bootstrap sampling
}
```

#### Usage Examples

```rust
use neuro_divergent::config::{PredictionIntervals, IntervalMethod};

// Create prediction intervals
let intervals = PredictionIntervals::new(
    vec![0.8, 0.9, 0.95],  // 80%, 90%, 95% confidence
    IntervalMethod::ConformalPrediction
)?;

// Default intervals (80%, 90%, 95% with quantile method)
let default_intervals = PredictionIntervals::default();

// Get quantile levels from confidence levels
let quantiles = intervals.quantile_levels();  // [0.025, 0.05, 0.1, 0.9, 0.95, 0.975]
```

### Loss Function Configuration

Comprehensive loss function options for different forecasting scenarios.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossFunction {
    MSE,        // Mean Squared Error
    MAE,        // Mean Absolute Error
    MAPE,       // Mean Absolute Percentage Error
    SMAPE,      // Symmetric Mean Absolute Percentage Error
    Huber,      // Huber Loss (robust to outliers)
    Quantile,   // Quantile Loss
}
```

#### Usage Examples

```rust
use neuro_divergent::config::LossFunction;

// Choose loss function based on data characteristics
let loss = if data.has_outliers() {
    LossFunction::Huber  // Robust to outliers
} else if data.is_percentage() {
    LossFunction::MAPE   // Good for percentage data
} else {
    LossFunction::MSE    // Standard choice
};

// Check if loss supports probabilistic outputs
let supports_prob = LossFunction::Quantile.supports_probabilistic();  // true

// Get all available loss functions
let all_losses = LossFunction::all();
```

### Optimizer Configuration

Optimizer selection and hyperparameter configuration.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerType {
    SGD,        // Stochastic Gradient Descent
    Adam,       // Adam optimizer
    AdamW,      // Adam with weight decay
    RMSprop,    // RMSprop optimizer
    Adagrad,    // Adagrad optimizer
}
```

#### Usage Examples

```rust
use neuro_divergent::config::OptimizerType;

// Select optimizer
let optimizer = OptimizerType::Adam;  // Most common choice

// For large models, AdamW often works better
let optimizer_large = OptimizerType::AdamW;

// For simple models, SGD can be sufficient
let optimizer_simple = OptimizerType::SGD;

// Get all available optimizers
let all_optimizers = OptimizerType::all();
```

### Device Configuration

Hardware acceleration and computation device selection.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    CPU,         // CPU computation
    GPU(usize),  // GPU computation with device index
}

impl Default for Device {
    fn default() -> Self {
        Self::CPU
    }
}
```

#### Usage Examples

```rust
use neuro_divergent::config::Device;

// CPU computation (default)
let device = Device::CPU;

// GPU computation
let gpu_device = Device::GPU(0);  // First GPU

// Display device
println!("Using device: {}", device);  // "cpu" or "gpu:0"
```

## Learning Rate Scheduling

Adaptive learning rate strategies for improved training.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig<T: Float> {
    pub scheduler_type: SchedulerType,
    pub step_size: Option<usize>,
    pub gamma: Option<T>,
    pub milestones: Option<Vec<usize>>,
    pub patience: Option<usize>,
    pub factor: Option<T>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchedulerType {
    StepLR,            // Step learning rate decay
    MultiStepLR,       // Multi-step learning rate decay
    ExponentialLR,     // Exponential learning rate decay
    ReduceLROnPlateau, // Reduce on plateau
    CosineAnnealingLR, // Cosine annealing
}
```

### Usage Examples

```rust
use neuro_divergent::config::{SchedulerConfig, SchedulerType};

// Step LR scheduler
let step_scheduler = SchedulerConfig {
    scheduler_type: SchedulerType::StepLR,
    step_size: Some(30),
    gamma: Some(0.1),
    ..Default::default()
};

// Reduce on plateau
let plateau_scheduler = SchedulerConfig {
    scheduler_type: SchedulerType::ReduceLROnPlateau,
    patience: Some(10),
    factor: Some(0.5),
    ..Default::default()
};

// Cosine annealing
let cosine_scheduler = SchedulerConfig {
    scheduler_type: SchedulerType::CosineAnnealingLR,
    step_size: Some(100), // T_max
    ..Default::default()
};
```

## Early Stopping Configuration

Prevent overfitting with early stopping mechanisms.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig<T: Float> {
    pub monitor: String,
    pub patience: usize,
    pub min_delta: T,
    pub mode: EarlyStoppingMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EarlyStoppingMode {
    Min,  // Monitor for minimum (e.g., loss)
    Max,  // Monitor for maximum (e.g., accuracy)
}
```

### Usage Examples

```rust
use neuro_divergent::config::{EarlyStoppingConfig, EarlyStoppingMode};

// Monitor validation loss
let early_stopping = EarlyStoppingConfig::new(
    "val_loss".to_string(),
    10,                           // patience
    0.001,                        // min_delta
    EarlyStoppingMode::Min        // minimize loss
);

// Default early stopping
let default_early_stopping = EarlyStoppingConfig::<f64>::default();
```

## Cross-Validation Configuration

Time series-aware cross-validation setup.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationConfig {
    pub n_windows: usize,
    pub h: usize,
    pub step_size: Option<usize>,
    pub test_size: Option<usize>,
    pub season_length: Option<usize>,
    pub refit: bool,
}
```

### Usage Examples

```rust
use neuro_divergent::config::CrossValidationConfig;

// Basic cross-validation
let cv_config = CrossValidationConfig::new(3, 12);  // 3 windows, 12-step horizon

// Advanced cross-validation
let advanced_cv = CrossValidationConfig::new(5, 24)
    .with_step_size(6)         // 6-step increments
    .with_test_size(48)        // 48-step test sets
    .with_season_length(12)    // Monthly seasonality
    .with_refit(false);        // Don't refit models

// Validate configuration
cv_config.validate()?;
```

## Generic Configuration Values

Type-safe configuration parameter storage.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigValue<T: Float> {
    Float(T),
    Int(i64),
    String(String),
    Bool(bool),
    FloatVec(Vec<T>),
    IntVec(Vec<i64>),
    StringVec(Vec<String>),
}
```

### Usage Examples

```rust
use neuro_divergent::config::ConfigValue;

// Create different value types
let learning_rate = ConfigValue::Float(0.001);
let batch_size = ConfigValue::Int(32);
let model_name = ConfigValue::String("LSTM".to_string());
let use_dropout = ConfigValue::Bool(true);
let layer_sizes = ConfigValue::IntVec(vec![64, 32, 16]);

// Extract values with type checking
if let Some(lr) = learning_rate.as_float() {
    println!("Learning rate: {}", lr);
}

if let Some(batch) = batch_size.as_int() {
    println!("Batch size: {}", batch);
}

if let Some(name) = model_name.as_string() {
    println!("Model: {}", name);
}
```

## Model-Specific Configurations

### Generic Model Configuration

Base configuration container for any model type.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericModelConfig<T: Float> {
    pub model_type: String,
    pub horizon: usize,
    pub input_size: usize,
    pub parameters: HashMap<String, ConfigValue<T>>,
}
```

#### Usage Examples

```rust
use neuro_divergent::config::{GenericModelConfig, ConfigValue};
use std::collections::HashMap;

// Create generic configuration
let mut config = GenericModelConfig::new(
    "LSTM".to_string(),
    12,  // horizon
    48   // input_size
);

// Add parameters
config.set_parameter("hidden_size", ConfigValue::Int(64));
config.set_parameter("num_layers", ConfigValue::Int(2));
config.set_parameter("dropout", ConfigValue::Float(0.1));
config.set_parameter("bidirectional", ConfigValue::Bool(false));

// Retrieve parameters
if let Some(hidden_size) = config.get_parameter("hidden_size") {
    if let Some(size) = hidden_size.as_int() {
        println!("Hidden size: {}", size);
    }
}

// Get all parameter keys
let keys = config.parameter_keys();
println!("Available parameters: {:?}", keys);
```

## Configuration Validation

### Validation Patterns

```rust
impl<T: Float> ModelConfig<T> for LSTMConfig {
    fn validate(&self) -> NeuroDivergentResult<()> {
        // Validate horizon
        if self.horizon == 0 {
            return Err(NeuroDivergentError::config("Horizon must be positive"));
        }
        
        // Validate input size
        if self.input_size == 0 {
            return Err(NeuroDivergentError::config("Input size must be positive"));
        }
        
        // Validate hidden size
        if self.hidden_size == 0 {
            return Err(NeuroDivergentError::config("Hidden size must be positive"));
        }
        
        // Validate dropout rate
        if self.dropout < 0.0 || self.dropout >= 1.0 {
            return Err(NeuroDivergentError::config(
                format!("Dropout must be in [0, 1), got {}", self.dropout)
            ));
        }
        
        // Validate number of layers
        if self.num_layers == 0 {
            return Err(NeuroDivergentError::config("Number of layers must be positive"));
        }
        
        Ok(())
    }
}
```

### Cross-Configuration Validation

```rust
fn validate_training_config(
    model_config: &dyn ModelConfig<f64>,
    training_config: &TrainingConfig
) -> NeuroDivergentResult<()> {
    // Check if batch size is reasonable for input size
    if training_config.batch_size > model_config.input_size() {
        return Err(NeuroDivergentError::config(
            "Batch size should not exceed input size"
        ));
    }
    
    // Check learning rate ranges
    if training_config.learning_rate <= 0.0 || training_config.learning_rate > 1.0 {
        return Err(NeuroDivergentError::config(
            "Learning rate must be in (0, 1]"
        ));
    }
    
    // Validate horizon compatibility
    if model_config.horizon() > model_config.input_size() {
        log::warn!("Horizon ({}) larger than input size ({})", 
                   model_config.horizon(), model_config.input_size());
    }
    
    Ok(())
}
```

## Configuration Serialization

### JSON Serialization

```rust
use serde_json;

// Serialize configuration to JSON
let config = LSTMConfig::builder()
    .horizon(12)
    .hidden_size(64)
    .build()?;

let json_string = serde_json::to_string_pretty(&config)?;
println!("{}", json_string);

// Deserialize from JSON
let loaded_config: LSTMConfig = serde_json::from_str(&json_string)?;
```

### TOML Configuration Files

```rust
use toml;

// Serialize to TOML
let toml_string = toml::to_string(&config)?;

// Save to file
std::fs::write("model_config.toml", toml_string)?;

// Load from file
let toml_content = std::fs::read_to_string("model_config.toml")?;
let loaded_config: LSTMConfig = toml::from_str(&toml_content)?;
```

### Example TOML Configuration

```toml
# model_config.toml
model_type = "LSTM"
horizon = 12
input_size = 48
hidden_size = 64
num_layers = 2
dropout = 0.1
bidirectional = false

[training]
max_epochs = 100
batch_size = 32
learning_rate = 0.001
optimizer = "Adam"

[early_stopping]
monitor = "val_loss"
patience = 10
min_delta = 0.001
mode = "Min"
```

## Environment-Based Configuration

### Configuration from Environment Variables

```rust
use std::env;

fn load_config_from_env() -> NeuroDivergentResult<LSTMConfig> {
    let horizon = env::var("LSTM_HORIZON")
        .map_err(|_| NeuroDivergentError::config("LSTM_HORIZON not set"))?
        .parse::<usize>()
        .map_err(|_| NeuroDivergentError::config("Invalid LSTM_HORIZON"))?;
    
    let hidden_size = env::var("LSTM_HIDDEN_SIZE")
        .unwrap_or_else(|_| "64".to_string())
        .parse::<usize>()
        .map_err(|_| NeuroDivergentError::config("Invalid LSTM_HIDDEN_SIZE"))?;
    
    LSTMConfig::builder()
        .horizon(horizon)
        .hidden_size(hidden_size)
        .build()
}
```

### Configuration Profiles

```rust
#[derive(Debug, Clone)]
pub enum ConfigProfile {
    Development,
    Testing,
    Production,
    Custom(String),
}

fn get_profile_config(profile: ConfigProfile) -> LSTMConfig {
    match profile {
        ConfigProfile::Development => {
            LSTMConfig::builder()
                .horizon(7)
                .hidden_size(32)      // Small for fast iteration
                .num_layers(1)
                .dropout(0.1)
                .build().unwrap()
        },
        
        ConfigProfile::Testing => {
            LSTMConfig::builder()
                .horizon(12)
                .hidden_size(64)      // Balanced
                .num_layers(2)
                .dropout(0.2)
                .build().unwrap()
        },
        
        ConfigProfile::Production => {
            LSTMConfig::builder()
                .horizon(24)
                .hidden_size(128)     // Large for best performance
                .num_layers(3)
                .dropout(0.15)
                .build().unwrap()
        },
        
        ConfigProfile::Custom(name) => {
            // Load from file or database
            load_custom_config(&name).unwrap_or_else(|_| {
                ConfigProfile::get_profile_config(ConfigProfile::Production)
            })
        }
    }
}
```

## Best Practices

### Configuration Validation

```rust
// Always validate configurations early
let config = LSTMConfig::builder()
    .horizon(12)
    .hidden_size(64)
    .build()?;  // Validation happens here

config.validate()?;  // Additional validation if needed
```

### Default Values

```rust
// Provide sensible defaults
impl Default for LSTMConfig {
    fn default() -> Self {
        Self {
            horizon: 1,
            input_size: 1,
            hidden_size: 64,
            num_layers: 2,
            dropout: 0.1,
            bidirectional: false,
            use_bias: true,
            batch_first: true,
        }
    }
}
```

### Configuration Documentation

```rust
/// LSTM model configuration
///
/// # Fields
///
/// * `horizon` - Number of future time steps to forecast (required > 0)
/// * `input_size` - Length of input sequence (required > 0)
/// * `hidden_size` - Size of LSTM hidden state (default: 64, range: 1-2048)
/// * `num_layers` - Number of LSTM layers (default: 2, range: 1-10)
/// * `dropout` - Dropout rate between layers (default: 0.1, range: [0, 1))
/// * `bidirectional` - Use bidirectional LSTM (default: false)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSTMConfig {
    // ... fields
}
```

The configuration system provides a comprehensive, type-safe approach to managing all aspects of neural forecasting, from model hyperparameters to system-level settings, with validation, serialization, and environment integration capabilities.