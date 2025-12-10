# Core Traits

The core traits define the fundamental interfaces that all forecasting models must implement, providing a consistent and type-safe interface for neural forecasting operations.

## Overview

The trait system is designed around three primary abstractions:

- **`BaseModel<T>`** - The main model interface
- **`ModelConfig<T>`** - Configuration and parameter management
- **`ForecastingEngine<T>`** - Advanced forecasting capabilities

## BaseModel Trait

The `BaseModel<T>` trait is the core interface that all forecasting models must implement.

### Definition

```rust
pub trait BaseModel<T: Float + Send + Sync>: Send + Sync {
    type Config: ModelConfig<T>;
    type State: ModelState<T>;
    
    // Core methods
    fn new(config: Self::Config) -> NeuroDivergentResult<Self>;
    fn fit(&mut self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<()>;
    fn predict(&self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<ForecastResult<T>>;
    
    // Additional methods...
}
```

### Required Methods

#### `new(config: Self::Config) -> NeuroDivergentResult<Self>`

Creates a new model instance with the given configuration.

**Parameters:**
- `config` - The model configuration

**Returns:**
- `Ok(Self)` - A new model instance
- `Err(NeuroDivergentError)` - If the configuration is invalid

**Example:**
```rust
use neuro_divergent::models::{LSTM, LSTMConfig};

let config = LSTMConfig::builder()
    .hidden_size(128)
    .num_layers(2)
    .horizon(12)
    .input_size(24)
    .build()?;

let model = LSTM::new(config)?;
```

#### `fit(&mut self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<()>`

Trains the model on the provided dataset.

**Parameters:**
- `data` - The training dataset

**Returns:**
- `Ok(())` - If training succeeds
- `Err(NeuroDivergentError)` - If training fails

**Example:**
```rust
let mut model = LSTM::new(config)?;
let dataset = TimeSeriesDataset::from_csv("train.csv", schema)?;
model.fit(&dataset)?;
```

#### `predict(&self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<ForecastResult<T>>`

Generates forecasts for the given dataset.

**Parameters:**
- `data` - The dataset to generate forecasts for

**Returns:**
- `Ok(ForecastResult<T>)` - Forecast results
- `Err(NeuroDivergentError)` - If prediction fails

**Example:**
```rust
let test_data = TimeSeriesDataset::from_csv("test.csv", schema)?;
let forecasts = model.predict(&test_data)?;

println!("Generated {} forecasts", forecasts.forecasts.len());
```

### State Management Methods

#### `state(&self) -> &Self::State`

Returns the current model state for serialization.

#### `restore_state(&mut self, state: Self::State) -> NeuroDivergentResult<()>`

Restores the model from a saved state.

#### `reset(&mut self) -> NeuroDivergentResult<()>`

Resets the model to its initial untrained state.

**Example:**
```rust
// Save model state
let state = model.state().clone();

// Reset model
model.reset()?;

// Restore previous state
model.restore_state(state)?;
```

### Metadata and Validation Methods

#### `metadata(&self) -> ModelMetadata`

Returns model metadata including capabilities and information.

#### `validate_data(&self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<()>`

Validates that the input data is compatible with this model.

#### `is_trained(&self) -> bool`

Checks if the model has been trained and is ready for prediction.

#### `parameter_count(&self) -> usize`

Returns the total number of parameters in the model.

**Example:**
```rust
let metadata = model.metadata();
println!("Model: {}", metadata.model_type);
println!("Parameters: {}", model.parameter_count());
println!("Trained: {}", model.is_trained());
```

## ModelConfig Trait

The `ModelConfig<T>` trait defines the interface for model configuration objects.

### Definition

```rust
pub trait ModelConfig<T: Float>: Clone + Send + Sync + 'static {
    fn validate(&self) -> NeuroDivergentResult<()>;
    fn horizon(&self) -> usize;
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
    fn exogenous_config(&self) -> &ExogenousConfig;
    fn model_type(&self) -> &str;
    
    // Parameter conversion methods
    fn to_parameters(&self) -> HashMap<String, ConfigParameter<T>>;
    fn from_parameters(params: HashMap<String, ConfigParameter<T>>) -> NeuroDivergentResult<Self>;
    
    // Builder pattern support
    fn builder() -> ConfigBuilder<Self, T>;
}
```

### Core Methods

#### `validate(&self) -> NeuroDivergentResult<()>`

Validates all configuration parameters.

**Returns:**
- `Ok(())` - If configuration is valid
- `Err(NeuroDivergentError)` - With details about validation failures

#### `horizon(&self) -> usize`

Returns the forecast horizon (number of steps to predict).

#### `input_size(&self) -> usize`

Returns the input window size (number of historical steps to use).

#### `output_size(&self) -> usize`

Returns the output size (usually equal to horizon for single-target forecasting).

### Parameter Management

#### `to_parameters(&self) -> HashMap<String, ConfigParameter<T>>`

Converts the configuration to a generic parameter map for serialization.

#### `from_parameters(params: HashMap<String, ConfigParameter<T>>) -> NeuroDivergentResult<Self>`

Creates a configuration from a generic parameter map.

**Example:**
```rust
let config = LSTMConfig::builder()
    .hidden_size(128)
    .num_layers(2)
    .horizon(12)
    .build()?;

// Convert to parameters
let params = config.to_parameters();

// Recreate from parameters
let restored_config = LSTMConfig::from_parameters(params)?;
```

## ForecastingEngine Trait

The `ForecastingEngine<T>` trait provides advanced forecasting capabilities for batch operations and probabilistic forecasting.

### Definition

```rust
pub trait ForecastingEngine<T: Float + Send + Sync>: Send + Sync {
    fn batch_predict(&self, datasets: &[TimeSeriesDataset<T>]) -> NeuroDivergentResult<Vec<ForecastResult<T>>>;
    fn predict_intervals(&self, data: &TimeSeriesDataset<T>, confidence_levels: &[f64]) -> NeuroDivergentResult<IntervalForecast<T>>;
    fn predict_quantiles(&self, data: &TimeSeriesDataset<T>, quantiles: &[f64]) -> NeuroDivergentResult<QuantileForecast<T>>;
    fn predict_multi_horizon(&self, data: &TimeSeriesDataset<T>, horizons: &[usize]) -> NeuroDivergentResult<MultiHorizonForecast<T>>;
}
```

### Batch Operations

#### `batch_predict(&self, datasets: &[TimeSeriesDataset<T>]) -> NeuroDivergentResult<Vec<ForecastResult<T>>>`

Performs batch prediction for multiple time series efficiently.

**Parameters:**
- `datasets` - Vector of datasets to predict

**Returns:**
- `Ok(Vec<ForecastResult<T>>)` - Vector of forecast results, one for each input dataset
- `Err(NeuroDivergentError)` - If any prediction fails

**Example:**
```rust
let datasets = vec![dataset1, dataset2, dataset3];
let results = engine.batch_predict(&datasets)?;

for (i, result) in results.iter().enumerate() {
    println!("Dataset {}: {} forecasts", i, result.forecasts.len());
}
```

### Probabilistic Forecasting

#### `predict_intervals(&self, data: &TimeSeriesDataset<T>, confidence_levels: &[f64]) -> NeuroDivergentResult<IntervalForecast<T>>`

Generates prediction intervals with specified confidence levels.

**Parameters:**
- `data` - The dataset to predict
- `confidence_levels` - Confidence levels (e.g., [0.8, 0.9, 0.95])

**Returns:**
- `Ok(IntervalForecast<T>)` - Interval forecasts with confidence bounds
- `Err(NeuroDivergentError)` - If prediction fails

**Example:**
```rust
let confidence_levels = vec![0.8, 0.9, 0.95];
let interval_forecast = engine.predict_intervals(&data, &confidence_levels)?;

println!("Point forecasts: {:?}", interval_forecast.forecasts);
println!("Lower bounds: {:?}", interval_forecast.lower_bounds);
println!("Upper bounds: {:?}", interval_forecast.upper_bounds);
```

#### `predict_quantiles(&self, data: &TimeSeriesDataset<T>, quantiles: &[f64]) -> NeuroDivergentResult<QuantileForecast<T>>`

Generates quantile forecasts for specified quantile levels.

**Parameters:**
- `data` - The dataset to predict
- `quantiles` - Quantile levels (e.g., [0.1, 0.5, 0.9])

**Example:**
```rust
let quantiles = vec![0.1, 0.5, 0.9];
let quantile_forecast = engine.predict_quantiles(&data, &quantiles)?;

for (quantile, forecasts) in &quantile_forecast.quantile_forecasts {
    println!("Quantile {}: {:?}", quantile, forecasts);
}
```

## Data Structures

### ForecastResult<T>

Contains forecast predictions and associated metadata.

```rust
pub struct ForecastResult<T: Float> {
    pub forecasts: Vec<T>,
    pub timestamps: Vec<DateTime<Utc>>,
    pub series_id: String,
    pub model_name: String,
    pub generated_at: DateTime<Utc>,
    pub metadata: Option<HashMap<String, String>>,
}
```

### IntervalForecast<T>

Contains prediction intervals with confidence bounds.

```rust
pub struct IntervalForecast<T: Float> {
    pub forecasts: Vec<T>,
    pub lower_bounds: Vec<Vec<T>>,
    pub upper_bounds: Vec<Vec<T>>,
    pub confidence_levels: Vec<f64>,
    pub timestamps: Vec<DateTime<Utc>>,
    pub series_id: String,
    pub model_name: String,
    pub generated_at: DateTime<Utc>,
}
```

### ModelMetadata

Contains model information and capabilities.

```rust
pub struct ModelMetadata {
    pub model_type: String,
    pub version: String,
    pub description: String,
    pub capabilities: ModelCapabilities,
    pub parameter_count: usize,
    pub memory_requirements: Option<usize>,
    pub training_time: Option<f64>,
    pub metadata: HashMap<String, String>,
}
```

### ModelCapabilities

Describes what features a model supports.

```rust
pub struct ModelCapabilities {
    pub supports_future_exogenous: bool,
    pub supports_historical_exogenous: bool,
    pub supports_static_exogenous: bool,
    pub supports_multivariate: bool,
    pub supports_probabilistic: bool,
    pub supports_quantile: bool,
    pub supports_recursive: bool,
    pub supports_parallel_training: bool,
    pub supports_online_learning: bool,
}
```

## Implementation Guidelines

### Custom Model Implementation

To implement a custom model, you need to:

1. **Define Configuration Structure**
```rust
#[derive(Debug, Clone)]
pub struct MyModelConfig {
    pub horizon: usize,
    pub input_size: usize,
    pub hidden_size: usize,
    // ... other parameters
}

impl ModelConfig<f64> for MyModelConfig {
    fn validate(&self) -> NeuroDivergentResult<()> {
        if self.horizon == 0 {
            return Err(NeuroDivergentError::config("Horizon must be > 0"));
        }
        // ... other validations
        Ok(())
    }
    
    // ... implement other required methods
}
```

2. **Define Model State**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MyModelState {
    pub model_type: String,
    pub version: u32,
    pub weights: Vec<f64>,
    pub trained_at: Option<DateTime<Utc>>,
    // ... other state data
}

impl ModelState<f64> for MyModelState {
    fn model_type(&self) -> &str {
        &self.model_type
    }
    
    // ... implement other required methods
}
```

3. **Implement BaseModel**
```rust
pub struct MyModel {
    config: MyModelConfig,
    state: MyModelState,
    // ... internal model data
}

impl BaseModel<f64> for MyModel {
    type Config = MyModelConfig;
    type State = MyModelState;
    
    fn new(config: Self::Config) -> NeuroDivergentResult<Self> {
        config.validate()?;
        
        Ok(Self {
            config,
            state: MyModelState::new(),
            // ... initialize internal state
        })
    }
    
    fn fit(&mut self, data: &TimeSeriesDataset<f64>) -> NeuroDivergentResult<()> {
        // Training implementation
        Ok(())
    }
    
    fn predict(&self, data: &TimeSeriesDataset<f64>) -> NeuroDivergentResult<ForecastResult<f64>> {
        // Prediction implementation
        Ok(ForecastResult {
            // ... create forecast result
        })
    }
    
    // ... implement other required methods
}
```

### Error Handling

All trait methods return `NeuroDivergentResult<T>` for consistent error handling:

```rust
// Configuration errors
Err(NeuroDivergentError::config("Invalid parameter value"))

// Training errors  
Err(NeuroDivergentError::training("Convergence failed"))

// Prediction errors
Err(NeuroDivergentError::prediction("Model not trained"))

// Data errors
Err(NeuroDivergentError::data("Missing required columns"))
```

### Thread Safety

All traits require `Send + Sync` for thread safety:

- Models can be shared across threads
- Training and prediction can be parallelized
- State serialization is thread-safe

### Type Safety

Generic type parameter `T: Float + Send + Sync` ensures:

- Numeric operations are well-defined
- Models work with both f32 and f64
- Thread safety is maintained
- Serialization is supported

## Best Practices

### Configuration Validation

Always validate configuration parameters comprehensively:

```rust
impl ModelConfig<T> for MyConfig {
    fn validate(&self) -> NeuroDivergentResult<()> {
        if self.horizon == 0 {
            return Err(NeuroDivergentError::config("Horizon must be positive"));
        }
        if self.input_size == 0 {
            return Err(NeuroDivergentError::config("Input size must be positive"));
        }
        if self.learning_rate <= 0.0 || self.learning_rate >= 1.0 {
            return Err(NeuroDivergentError::config("Learning rate must be in (0, 1)"));
        }
        Ok(())
    }
}
```

### Data Validation

Always validate input data compatibility:

```rust
fn validate_data(&self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<()> {
    if data.is_empty() {
        return Err(NeuroDivergentError::data("Dataset is empty"));
    }
    
    let expected_features = self.config.input_size();
    if data.feature_count() != expected_features {
        return Err(NeuroDivergentError::data(
            format!("Expected {} features, got {}", expected_features, data.feature_count())
        ));
    }
    
    Ok(())
}
```

### State Management

Implement proper state serialization for model persistence:

```rust
impl ModelState<T> for MyState {
    fn is_compatible(&self, config: &dyn ModelConfig<T>) -> bool {
        config.model_type() == self.model_type() &&
        config.input_size() == self.input_size &&
        config.output_size() == self.output_size
    }
}
```

This trait system provides a robust, type-safe foundation for implementing neural forecasting models while maintaining compatibility with the NeuralForecast Python API.