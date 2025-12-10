# NeuralForecast API

The `NeuralForecast` class is the primary user interface for neural forecasting operations, providing 100% compatibility with the NeuralForecast Python library while leveraging Rust's performance and safety guarantees.

## Overview

`NeuralForecast` manages a collection of forecasting models and provides high-level operations for training, prediction, and evaluation. It handles data preprocessing, model coordination, and result aggregation automatically.

### Key Features

- **Multi-Model Support**: Manage multiple forecasting models simultaneously
- **Automatic Preprocessing**: Built-in data scaling and preprocessing
- **Parallel Processing**: Multi-threaded training and prediction
- **Cross-Validation**: Time series-aware cross-validation
- **Prediction Intervals**: Probabilistic forecasting with confidence bounds
- **Model Persistence**: Save and load trained models

## Class Definition

```rust
pub struct NeuralForecast<T: Float + Send + Sync> {
    models: Vec<Box<dyn BaseModel<T>>>,
    frequency: Frequency,
    local_scaler_type: Option<ScalerType>,
    num_threads: Option<usize>,
    prediction_intervals: Option<PredictionIntervals>,
    device: Device,
    is_fitted: bool,
    training_schema: Option<TimeSeriesSchema>,
    model_metadata: HashMap<String, ModelInfo>,
}
```

## Construction

### Builder Pattern (Recommended)

```rust
use neuro_divergent::{NeuralForecast, models::LSTM, Frequency, ScalerType, Device};

let nf = NeuralForecast::builder()
    .with_model(Box::new(lstm_model))
    .with_frequency(Frequency::Daily)
    .with_local_scaler(ScalerType::StandardScaler)
    .with_num_threads(4)
    .with_device(Device::CPU)
    .build()?;
```

### Direct Constructor

```rust
let models = vec![Box::new(lstm_model), Box::new(nbeats_model)];
let nf = NeuralForecast::new(models, Frequency::Daily)?;
```

## Core Methods

### Training

#### `fit(&mut self, data: TimeSeriesDataFrame<T>) -> NeuroDivergentResult<()>`

Trains all models on the provided time series data.

**Parameters:**
- `data` - Time series data for training

**Returns:**
- `Ok(())` - If training succeeds for all models
- `Err(NeuroDivergentError)` - If any model training fails

**Example:**
```rust
// Load training data
let data = TimeSeriesDataFrame::from_csv("train_data.csv")?;

// Fit all models
nf.fit(data)?;

println!("All models trained successfully");
```

**Notes:**
- Models are trained in parallel if `num_threads > 1`
- Data is automatically preprocessed using the configured scaler
- Training schema is stored for validation of future data
- Model metadata is updated with training statistics

#### `fit_with_validation(&mut self, data: TimeSeriesDataFrame<T>, validation_config: ValidationConfig) -> NeuroDivergentResult<()>`

Trains models with custom validation configuration.

**Parameters:**
- `data` - Training data
- `validation_config` - Validation settings (split ratio, shuffle, etc.)

**Example:**
```rust
let validation_config = ValidationConfig::new()
    .with_validation_split(0.2)
    .with_shuffle(true)
    .with_random_seed(42);

nf.fit_with_validation(data, validation_config)?;
```

### Prediction

#### `predict(&self) -> NeuroDivergentResult<ForecastDataFrame<T>>`

Generates forecasts using all fitted models.

**Returns:**
- `Ok(ForecastDataFrame<T>)` - Forecast results from all models
- `Err(NeuroDivergentError)` - If models are not fitted or prediction fails

**Example:**
```rust
// Generate forecasts
let forecasts = nf.predict()?;

// Access forecasts by model
for model_name in nf.model_names() {
    let model_forecasts = forecasts.get_model_forecasts(&model_name)?;
    println!("{}: {:?}", model_name, model_forecasts);
}
```

#### `predict_on(&self, data: TimeSeriesDataFrame<T>) -> NeuroDivergentResult<ForecastDataFrame<T>>`

Generates forecasts for new input data.

**Parameters:**
- `data` - New time series data to forecast

**Example:**
```rust
// Load new data
let new_data = TimeSeriesDataFrame::from_csv("new_data.csv")?;

// Generate forecasts for new data
let forecasts = nf.predict_on(new_data)?;
```

#### `predict_with_config(&self, config: PredictionConfig) -> NeuroDivergentResult<ForecastDataFrame<T>>`

Generates forecasts with custom prediction configuration.

**Parameters:**
- `config` - Prediction settings (intervals, sampling, etc.)

**Example:**
```rust
let prediction_config = PredictionConfig::new()
    .with_intervals()
    .with_num_samples(1000)
    .with_temperature(0.8);

let forecasts = nf.predict_with_config(prediction_config)?;
```

### Cross-Validation

#### `cross_validation(&mut self, data: TimeSeriesDataFrame<T>, config: CrossValidationConfig) -> NeuroDivergentResult<CrossValidationDataFrame<T>>`

Performs time series cross-validation for model evaluation.

**Parameters:**
- `data` - Full dataset for cross-validation
- `config` - Cross-validation configuration

**Returns:**
- `Ok(CrossValidationDataFrame<T>)` - Cross-validation results
- `Err(NeuroDivergentError)` - If validation fails

**Example:**
```rust
let cv_config = CrossValidationConfig::new(3, 12)  // 3 windows, 12-step horizon
    .with_step_size(1)
    .with_refit(false);

let cv_results = nf.cross_validation(data, cv_config)?;

// Analyze results
println!("CV cutoffs: {:?}", cv_results.cutoffs());
let metrics = cv_results.metrics();
for (model_name, model_metrics) in metrics {
    println!("{}: MAE = {:.4}", model_name, model_metrics.get("MAE").unwrap());
}
```

### Utility Methods

#### `fit_predict(&mut self, train_data: TimeSeriesDataFrame<T>) -> NeuroDivergentResult<ForecastDataFrame<T>>`

Convenience method that fits models and generates predictions in one call.

**Example:**
```rust
let forecasts = nf.fit_predict(train_data)?;
```

#### `model_names(&self) -> Vec<String>`

Returns the names of all models in the ensemble.

#### `num_models(&self) -> usize`

Returns the number of models in the ensemble.

#### `is_fitted(&self) -> bool`

Checks if all models have been trained.

#### `frequency(&self) -> Frequency`

Returns the time series frequency.

**Example:**
```rust
println!("Models: {:?}", nf.model_names());
println!("Model count: {}", nf.num_models());
println!("Fitted: {}", nf.is_fitted());
println!("Frequency: {}", nf.frequency());
```

### Model Management

#### `get_model(&self, name: &str) -> Option<&dyn BaseModel<T>>`

Gets a reference to a specific model by name.

#### `get_model_mut(&mut self, name: &str) -> Option<&mut dyn BaseModel<T>>`

Gets a mutable reference to a specific model by name.

**Example:**
```rust
if let Some(lstm_model) = nf.get_model("LSTM") {
    let metadata = lstm_model.metadata();
    println!("LSTM parameters: {}", metadata.parameter_count);
}

// Modify model (if mutable reference needed)
if let Some(lstm_model) = nf.get_model_mut("LSTM") {
    lstm_model.reset()?;
}
```

#### `reset(&mut self) -> NeuroDivergentResult<()>`

Resets all models to their untrained state.

**Example:**
```rust
// Reset all models
nf.reset()?;
assert!(!nf.is_fitted());
```

### Persistence

#### `save<P: AsRef<Path>>(&self, path: P) -> NeuroDivergentResult<()>`

Saves all models and metadata to file.

#### `load<P: AsRef<Path>>(path: P) -> NeuroDivergentResult<Self>`

Loads models and metadata from file.

**Example:**
```rust
// Save trained models
nf.save("models.json")?;

// Load models later
let loaded_nf = NeuralForecast::load("models.json")?;
```

## Builder Pattern

The `NeuralForecastBuilder` provides a fluent API for configuration.

### Builder Methods

#### `with_models(models: Vec<Box<dyn BaseModel<T>>>) -> Self`

Sets the model ensemble.

#### `with_model(model: Box<dyn BaseModel<T>>) -> Self`

Adds a single model to the ensemble.

#### `with_frequency(frequency: Frequency) -> Self`

Sets the time series frequency.

#### `with_local_scaler(scaler_type: ScalerType) -> Self`

Sets the preprocessing scaler type.

#### `with_num_threads(num_threads: usize) -> Self`

Sets the number of threads for parallel processing.

#### `with_prediction_intervals(intervals: PredictionIntervals) -> Self`

Configures prediction intervals.

#### `with_device(device: Device) -> Self`

Sets the computation device (CPU/GPU).

#### `build(self) -> NeuroDivergentResult<NeuralForecast<T>>`

Builds the `NeuralForecast` instance.

### Example: Complete Builder Usage

```rust
use neuro_divergent::{
    NeuralForecast, models::{LSTM, NBEATS}, 
    Frequency, ScalerType, Device, PredictionIntervals, IntervalMethod
};

// Create models
let lstm = LSTM::builder()
    .hidden_size(128)
    .num_layers(2)
    .horizon(12)
    .input_size(24)
    .build()?;

let nbeats = NBEATS::builder()
    .stack_types(vec![StackType::Trend, StackType::Seasonality])
    .num_blocks(3)
    .horizon(12)
    .input_size(24)
    .build()?;

// Create prediction intervals
let intervals = PredictionIntervals::new(
    vec![0.8, 0.9, 0.95],
    IntervalMethod::ConformalPrediction
)?;

// Build NeuralForecast instance
let nf = NeuralForecast::builder()
    .with_model(Box::new(lstm))
    .with_model(Box::new(nbeats))
    .with_frequency(Frequency::Monthly)
    .with_local_scaler(ScalerType::StandardScaler)
    .with_num_threads(4)
    .with_prediction_intervals(intervals)
    .with_device(Device::CPU)
    .build()?;
```

## Configuration Types

### ValidationConfig

Configuration for training validation.

```rust
pub struct ValidationConfig {
    pub validation_split: Option<f64>,
    pub shuffle: bool,
    pub random_seed: Option<u64>,
}

impl ValidationConfig {
    pub fn new() -> Self;
    pub fn with_validation_split(self, split: f64) -> Self;
    pub fn with_shuffle(self, shuffle: bool) -> Self;
    pub fn with_random_seed(self, seed: u64) -> Self;
}
```

### PredictionConfig

Configuration for prediction generation.

```rust
pub struct PredictionConfig {
    pub include_intervals: bool,
    pub num_samples: Option<usize>,
    pub temperature: Option<f64>,
}

impl PredictionConfig {
    pub fn new() -> Self;
    pub fn with_intervals(self) -> Self;
    pub fn with_num_samples(self, num_samples: usize) -> Self;
    pub fn with_temperature(self, temperature: f64) -> Self;
}
```

## Error Handling

All methods return `NeuroDivergentResult<T>` for comprehensive error handling:

### Common Error Types

- **ConfigError**: Invalid configuration parameters
- **DataError**: Data validation or compatibility issues
- **TrainingError**: Model training failures
- **PredictionError**: Prediction generation failures

### Example Error Handling

```rust
match nf.fit(data) {
    Ok(()) => println!("Training successful"),
    Err(NeuroDivergentError::TrainingError(msg)) => {
        eprintln!("Training failed: {}", msg);
    },
    Err(NeuroDivergentError::DataError(msg)) => {
        eprintln!("Data error: {}", msg);
    },
    Err(e) => eprintln!("Other error: {}", e),
}
```

## Performance Considerations

### Memory Usage

- Models are stored as boxed trait objects for flexibility
- Data is processed using Polars for memory efficiency
- Lazy evaluation where possible to minimize memory footprint

### Parallel Processing

```rust
// Enable parallel training and prediction
let nf = NeuralForecast::builder()
    .with_num_threads(8)  // Use 8 threads
    .build()?;
```

### GPU Acceleration

```rust
// Use GPU for compatible models
let nf = NeuralForecast::builder()
    .with_device(Device::GPU(0))  // Use first GPU
    .build()?;
```

## Integration Examples

### Basic Forecasting Workflow

```rust
use neuro_divergent::prelude::*;

// 1. Create models
let lstm = LSTM::builder().hidden_size(64).horizon(7).build()?;
let nbeats = NBEATS::builder().num_blocks(2).horizon(7).build()?;

// 2. Create NeuralForecast instance
let mut nf = NeuralForecast::builder()
    .with_model(Box::new(lstm))
    .with_model(Box::new(nbeats))
    .with_frequency(Frequency::Daily)
    .build()?;

// 3. Load and prepare data
let data = TimeSeriesDataFrame::from_csv("data.csv")?;

// 4. Train models
nf.fit(data.clone())?;

// 5. Generate forecasts
let forecasts = nf.predict()?;

// 6. Evaluate with cross-validation
let cv_config = CrossValidationConfig::new(3, 7);
let cv_results = nf.cross_validation(data, cv_config)?;
```

### Financial Forecasting Example

```rust
// Configure for financial time series
let nf = NeuralForecast::builder()
    .with_frequency(Frequency::BusinessDaily)
    .with_local_scaler(ScalerType::RobustScaler)  // Robust to outliers
    .with_prediction_intervals(PredictionIntervals::new(
        vec![0.95, 0.99],  // High confidence for risk management
        IntervalMethod::ConformalPrediction
    )?)
    .build()?;

// Train on historical financial data
let financial_data = TimeSeriesDataFrame::from_csv("stock_prices.csv")?;
nf.fit(financial_data)?;

// Generate forecasts with confidence intervals
let forecasts = nf.predict()?;
let intervals = forecasts.prediction_intervals()?;
```

### IoT Sensor Forecasting Example

```rust
// Configure for high-frequency sensor data
let nf = NeuralForecast::builder()
    .with_frequency(Frequency::Minute)
    .with_num_threads(16)  // High parallelism for real-time processing
    .with_device(Device::GPU(0))  // GPU acceleration
    .build()?;

// Stream processing
let sensor_data = TimeSeriesDataFrame::from_streaming_source()?;
nf.fit(sensor_data)?;

// Real-time prediction
let current_data = get_current_sensor_data()?;
let forecasts = nf.predict_on(current_data)?;
```

## Thread Safety

`NeuralForecast` is `Send + Sync` and can be safely shared across threads:

```rust
use std::sync::Arc;
use std::thread;

let nf = Arc::new(fitted_neural_forecast);

let handles: Vec<_> = (0..4).map(|i| {
    let nf = Arc::clone(&nf);
    thread::spawn(move || {
        let data = load_test_data(i)?;
        nf.predict_on(data)
    })
}).collect();

for handle in handles {
    let forecasts = handle.join().unwrap()?;
    // Process forecasts
}
```

## Best Practices

### Model Selection

```rust
// Use complementary models for ensemble forecasting
let linear_model = DLinear::builder().horizon(12).build()?;
let nonlinear_model = LSTM::builder().hidden_size(128).horizon(12).build()?;
let interpretable_model = NBEATS::builder().interpretable(true).horizon(12).build()?;

let nf = NeuralForecast::builder()
    .with_model(Box::new(linear_model))
    .with_model(Box::new(nonlinear_model))
    .with_model(Box::new(interpretable_model))
    .build()?;
```

### Data Preprocessing

```rust
// Choose scaler based on data characteristics
let scaler = if data.has_outliers() {
    ScalerType::RobustScaler  // Robust to outliers
} else if data.is_stationary() {
    ScalerType::StandardScaler  // Zero mean, unit variance
} else {
    ScalerType::MinMaxScaler  // Scale to [0, 1]
};

let nf = NeuralForecast::builder()
    .with_local_scaler(scaler)
    .build()?;
```

### Error Recovery

```rust
// Handle partial training failures
match nf.fit(data) {
    Ok(()) => {
        // All models trained successfully
    },
    Err(NeuroDivergentError::TrainingError(msg)) => {
        // Some models may have failed
        println!("Training issues: {}", msg);
        
        // Check which models are trained
        for model_name in nf.model_names() {
            if let Some(model) = nf.get_model(&model_name) {
                println!("{}: trained = {}", model_name, model.is_trained());
            }
        }
    },
    Err(e) => return Err(e),
}
```

The `NeuralForecast` class provides a comprehensive, user-friendly interface for neural forecasting while maintaining the performance and safety benefits of Rust.