# neuro-divergent-core

[![Crates.io](https://img.shields.io/crates/v/neuro-divergent-core.svg)](https://crates.io/crates/neuro-divergent-core)
[![Documentation](https://docs.rs/neuro-divergent-core/badge.svg)](https://docs.rs/neuro-divergent-core)
[![License](https://img.shields.io/crates/l/neuro-divergent-core.svg)](LICENSE)

**Core foundation for neuro-divergent neural forecasting library** - A high-performance, type-safe neural forecasting system built on the ruv-FANN foundation.

## Technical Introduction

`neuro-divergent-core` provides the foundational abstractions, data structures, and integration layer for advanced neural time series forecasting. Built on Rust's zero-cost abstractions and ruv-FANN's neural network infrastructure, this crate delivers type-safe, memory-efficient forecasting capabilities without sacrificing performance.

The core design philosophy emphasizes:
- **Type Safety**: Generic traits over `f32`/`f64` with compile-time guarantees
- **Zero-Cost Abstractions**: Trait-based polymorphism without runtime overhead
- **Memory Safety**: Rust's ownership model ensures safe concurrent operations
- **Integration**: Seamless bridge to ruv-FANN's optimized neural networks

## Features Overview

### 🧠 **Neural Network Integration**
- **ruv-FANN Bridge**: `NetworkAdapter` provides time series-specific preprocessing/postprocessing
- **Training Pipeline**: `TrainingBridge` integrates forecasting workflows with ruv-FANN algorithms
- **Activation Mapping**: Type-safe activation function configuration and validation

### 📊 **Time Series Data Structures**
- **`TimeSeriesDataFrame`**: Polars-backed data structure for efficient time series operations
- **`TimeSeriesDataset`**: Internal representation optimized for neural network training
- **`TimeSeriesSchema`**: Flexible schema system supporting exogenous variables and static features

### ⚡ **High-Performance Core Traits**
- **`BaseModel<T>`**: Universal forecasting interface with generic floating-point support
- **`ForecastingEngine<T>`**: Advanced capabilities including batch prediction and probabilistic forecasting
- **`ModelConfig<T>`**: Type-safe configuration system with validation and serialization

### 🔧 **Configuration Management**
- **System Configuration**: Comprehensive settings for performance, memory, and parallelism
- **Model Configuration**: Flexible parameter management with builder patterns
- **Validation**: Built-in configuration validation with detailed error reporting

### 🛡️ **Error Handling**
- **`NeuroDivergentError`**: Comprehensive error types covering all operation aspects
- **Context-Rich Errors**: Detailed error information with field-level specificity
- **Error Builder**: Fluent API for constructing detailed error instances

## Architecture Details

### Core Traits Hierarchy

```rust
// Universal model interface with generic floating-point support
pub trait BaseModel<T: Float + Send + Sync>: Send + Sync {
    type Config: ModelConfig<T>;
    type State: ModelState<T>;
    
    fn fit(&mut self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<()>;
    fn predict(&self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<ForecastResult<T>>;
    fn cross_validation(&mut self, data: &TimeSeriesDataset<T>, config: CrossValidationConfig) 
        -> NeuroDivergentResult<CrossValidationResult<T>>;
}

// Advanced forecasting capabilities
pub trait ForecastingEngine<T: Float + Send + Sync>: Send + Sync {
    fn batch_predict(&self, datasets: &[TimeSeriesDataset<T>]) 
        -> NeuroDivergentResult<Vec<ForecastResult<T>>>;
    fn predict_intervals(&self, data: &TimeSeriesDataset<T>, confidence_levels: &[f64]) 
        -> NeuroDivergentResult<IntervalForecast<T>>;
    fn predict_quantiles(&self, data: &TimeSeriesDataset<T>, quantiles: &[f64]) 
        -> NeuroDivergentResult<QuantileForecast<T>>;
}
```

### Time Series Data Architecture

```rust
// Main data structure equivalent to pandas DataFrame
pub struct TimeSeriesDataFrame<T: Float> {
    pub data: DataFrame,              // Polars-backed storage
    pub schema: TimeSeriesSchema,     // Column definitions and types
}

// Internal optimized representation
pub struct TimeSeriesDataset<T: Float> {
    pub unique_ids: Vec<String>,
    pub series_data: HashMap<String, SeriesData<T>>,
    pub static_features: Option<HashMap<String, Vec<T>>>,
    pub schema: TimeSeriesSchema,
}

// Individual series data with exogenous support
pub struct SeriesData<T: Float> {
    pub timestamps: Vec<DateTime<Utc>>,
    pub target_values: Vec<T>,
    pub historical_exogenous: Option<Array2<T>>,  // (time_steps, n_features)
    pub future_exogenous: Option<Array2<T>>,      // (future_steps, n_features)
}
```

### Integration Layer

```rust
// Bridge between forecasting and ruv-FANN networks
pub struct NetworkAdapter<T: Float> {
    network: Network<T>,
    input_preprocessor: Box<dyn InputPreprocessor<T>>,
    output_postprocessor: Box<dyn OutputPostprocessor<T>>,
    activation_mapper: ActivationMapper,
}

// Training integration with ruv-FANN algorithms
pub struct TrainingBridge<T: Float> {
    algorithm: Box<dyn TrainingAlgorithm<T>>,
    config: TrainingBridgeConfig<T>,
    state: Arc<Mutex<TrainingBridgeState<T>>>,
}
```

## Usage Examples

### Basic Time Series Operations

```rust
use neuro_divergent_core::prelude::*;
use chrono::{DateTime, Utc};

// Create time series schema
let schema = TimeSeriesSchema::new("series_id", "timestamp", "value")
    .with_static_features(vec!["category".to_string()])
    .with_historical_exogenous(vec!["temperature".to_string(), "humidity".to_string()]);

// Load data from CSV
let ts_dataframe = TimeSeriesDataFrame::<f64>::from_csv("data.csv", schema)?;

// Convert to internal dataset format
let dataset = ts_dataframe.to_dataset()?;

// Validate data integrity
let validation_report = ts_dataframe.validate()?;
if !validation_report.is_valid {
    for error in validation_report.errors {
        eprintln!("Validation error: {}", error.message);
    }
}
```

### Neural Network Integration

```rust
use neuro_divergent_core::{integration::*, prelude::*};
use ruv_fann::{Network, NetworkBuilder, ActivationFunction};

// Create ruv-FANN network
let network = NetworkBuilder::new()
    .input_layer(24)           // 24-hour lookback window
    .hidden_layer(64)          // Hidden layer with 64 neurons
    .hidden_layer(32)          // Second hidden layer
    .output_layer(12)          // 12-hour forecast horizon
    .activation_function(ActivationFunction::Sigmoid)
    .build::<f64>()?;

// Wrap with forecasting adapter
let mut adapter = NetworkAdapter::from_network(network)
    .with_input_processor(Box::new(StandardPreprocessor::new()))
    .with_output_processor(Box::new(ForecastPostprocessor::new()));

// Prepare time series input
let input = TimeSeriesInput {
    target_history: recent_values,
    exogenous_history: Some(historical_features),
    exogenous_future: Some(future_features),
    static_features: Some(static_vars),
    timestamps: timestamps.clone(),
    series_id: "sensor_001".to_string(),
    metadata: HashMap::new(),
};

// Generate forecast
let output = adapter.forward(&input)?;
let forecast = adapter.output_postprocessor.process(
    &output.raw_outputs, 
    &postprocessor_context
)?;
```

### Model Configuration

```rust
use neuro_divergent_core::config::*;

// Build model configuration
let config = ModelConfigBuilder::<f64>::new()
    .with_model_type("lstm_forecaster")
    .with_horizon(12)                    // 12-step forecast horizon
    .with_input_size(24)                 // 24-step input window
    .with_parameter("hidden_size", ConfigParameter::Integer(64))
    .with_parameter("learning_rate", ConfigParameter::Float(0.001))
    .with_parameter("dropout_rate", ConfigParameter::Float(0.2))
    .with_exogenous_config(ExogenousConfig {
        static_features: vec!["category".to_string()],
        historical_features: vec!["temperature".to_string(), "humidity".to_string()],
        future_features: vec!["weather_forecast".to_string()],
        auto_encode_categorical: true,
        max_categorical_cardinality: Some(100),
    })
    .build()?;

// Validate configuration
config.validate()?;

// Save configuration
config.save("model_config.json")?;

// Load configuration
let loaded_config = GenericModelConfig::<f64>::load("model_config.json")?;
```

### Error Handling with Context

```rust
use neuro_divergent_core::error::*;

// Rich error construction
let error = ErrorBuilder::data("Invalid time series data")
    .context("field", "timestamp")
    .context("series_id", "sensor_001")
    .source(std::io::Error::new(std::io::ErrorKind::NotFound, "File missing"))
    .build();

// Pattern matching on error types
match error {
    NeuroDivergentError::DataError { message, field, .. } => {
        eprintln!("Data error in field {:?}: {}", field, message);
    }
    NeuroDivergentError::NetworkError(net_err) => {
        eprintln!("Network integration error: {}", net_err);
    }
    _ => eprintln!("Other error: {}", error),
}

// Convenient error macros
let validation_error = data_error!("Missing required column", field = "timestamp");
let config_error = config_error!("Invalid horizon value", "horizon" => "0");
```

## API Documentation

### Key Traits

#### `BaseModel<T>`
Universal interface for forecasting models with type-safe operations:

```rust
impl<T: Float + Send + Sync> BaseModel<T> for MyForecastModel {
    type Config = MyModelConfig<T>;
    type State = MyModelState<T>;
    
    fn fit(&mut self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<()> {
        // Training implementation
    }
    
    fn predict(&self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<ForecastResult<T>> {
        // Prediction implementation
    }
}
```

#### `ModelConfig<T>`
Type-safe configuration management with validation:

```rust
impl<T: Float> ModelConfig<T> for MyModelConfig<T> {
    fn validate(&self) -> NeuroDivergentResult<()> {
        if self.horizon() == 0 {
            return Err(config_error!("Horizon must be greater than 0"));
        }
        Ok(())
    }
    
    fn to_parameters(&self) -> HashMap<String, ConfigParameter<T>> {
        // Convert to generic parameter map
    }
}
```

#### `ModelState<T>`
Serializable model state for persistence:

```rust
impl<T: Float> ModelState<T> for MyModelState<T> {
    fn model_type(&self) -> &str { "my_model" }
    fn version(&self) -> u32 { 1 }
    fn is_compatible(&self, config: &dyn ModelConfig<T>) -> bool {
        // Compatibility check
    }
}
```

### Core Data Types

#### `TimeSeriesDataFrame<T>`
High-level interface for time series data with Polars backend:

```rust
// Create from various sources
let df = TimeSeriesDataFrame::<f64>::from_csv("data.csv", schema)?;
let df = TimeSeriesDataFrame::<f64>::from_parquet("data.parquet", schema)?;
let df = TimeSeriesDataFrame::<f64>::from_polars(polars_df, schema)?;

// Data operations
let unique_ids = df.unique_ids()?;
let filtered = df.filter_date_range(start_date, end_date)?;
let series_subset = df.filter_by_id("series_001")?;

// Validation and export
let report = df.validate()?;
df.to_csv("output.csv")?;
df.to_parquet("output.parquet")?;
```

#### `NetworkAdapter<T>`
Integration bridge with ruv-FANN networks:

```rust
let adapter = NetworkAdapter::from_network(ruv_fann_network)
    .with_input_processor(Box::new(custom_preprocessor))
    .with_output_processor(Box::new(custom_postprocessor));

// Forward pass with preprocessing
let output = adapter.forward(&time_series_input)?;

// Access underlying network
let network = adapter.network();
let network_mut = adapter.network_mut();
```

## Integration Guide

### Building on neuro-divergent-core

Other crates in the neuro-divergent ecosystem build upon this foundation:

```toml
[dependencies]
neuro-divergent-core = "0.1"
```

#### Implementing Custom Models

```rust
use neuro_divergent_core::prelude::*;

pub struct LSTMForecaster<T: Float> {
    config: LSTMConfig<T>,
    state: LSTMState<T>,
    network_adapter: NetworkAdapter<T>,
}

impl<T: Float + Send + Sync> BaseModel<T> for LSTMForecaster<T> {
    type Config = LSTMConfig<T>;
    type State = LSTMState<T>;
    
    fn new(config: Self::Config) -> NeuroDivergentResult<Self> {
        // Build ruv-FANN network based on config
        let network = NetworkBuilder::new()
            .input_layer(config.input_size())
            .hidden_layer(config.hidden_size)
            .output_layer(config.output_size())
            .build()?;
            
        let adapter = NetworkAdapter::from_network(network);
        
        Ok(Self {
            config,
            state: LSTMState::new(),
            network_adapter: adapter,
        })
    }
    
    fn fit(&mut self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<()> {
        // Use TrainingBridge for ruv-FANN integration
        let mut bridge = TrainingBridge::new(
            Box::new(ruv_fann::training::BackpropagationAlgorithm::new())
        );
        
        let training_config = TrainingBridgeConfig {
            max_epochs: self.config.epochs,
            learning_rate: self.config.learning_rate,
            target_error: self.config.target_error,
            validation_split: Some(self.config.validation_split),
            early_stopping: Some(self.config.early_stopping.clone()),
            batch_size: self.config.batch_size,
            shuffle: true,
        };
        
        let result = bridge.train_network(
            self.network_adapter.network_mut(),
            data,
            &training_config,
        )?;
        
        self.state.update_from_training(result);
        Ok(())
    }
}
```

#### Extending Data Processing

```rust
use neuro_divergent_core::integration::*;

pub struct AdvancedPreprocessor<T: Float> {
    config: PreprocessorConfig<T>,
    scaler: StandardScaler<T>,
    feature_engineer: FeatureEngineer<T>,
}

impl<T: Float> InputPreprocessor<T> for AdvancedPreprocessor<T> {
    fn process(&self, input: &TimeSeriesInput<T>) -> NeuroDivergentResult<Vec<T>> {
        // 1. Apply scaling
        let scaled = self.scaler.transform(&input.target_history)?;
        
        // 2. Engineer features
        let mut features = self.feature_engineer.create_features(&scaled)?;
        
        // 3. Include exogenous variables
        if let Some(exog) = &input.exogenous_history {
            let exog_scaled = self.scaler.transform_exogenous(exog)?;
            features.extend_from_slice(&exog_scaled);
        }
        
        // 4. Add static features
        if let Some(static_feat) = &input.static_features {
            features.extend_from_slice(static_feat);
        }
        
        Ok(features)
    }
    
    fn output_size(&self) -> usize {
        self.config.window_size + 
        self.config.exogenous_size + 
        self.config.static_size
    }
}
```

## Performance Considerations

### Memory Usage

- **Zero-Copy Operations**: Extensive use of array views and references to minimize allocations
- **Memory Pools**: Optional memory pooling for high-frequency operations
- **Streaming Processing**: Support for processing large datasets without full memory loading

```rust
// Memory-efficient data loading
let lazy_df = LazyFrame::scan_csv("large_dataset.csv", ScanArgsCSV::default())
    .filter(col("timestamp").gt(lit(start_date)))
    .select([col("series_id"), col("timestamp"), col("value")])
    .collect()?;
```

### Thread Safety

All core types implement `Send + Sync` for safe concurrent operations:

```rust
// Safe parallel processing
dataset.series_data.par_iter()
    .map(|(id, series)| {
        let forecast = model.predict_series(series)?;
        Ok((id.clone(), forecast))
    })
    .collect::<NeuroDivergentResult<HashMap<_, _>>>()?;
```

### Optimization Guidelines

1. **Generic Specialization**: Use `f32` for memory-constrained environments, `f64` for precision-critical applications
2. **Batch Operations**: Prefer batch prediction over individual series forecasting
3. **Configuration Caching**: Validate and cache configurations to avoid repeated validation overhead
4. **Network Reuse**: Reuse `NetworkAdapter` instances across predictions to amortize initialization costs

```rust
// Optimized batch prediction
let forecasts = forecasting_engine.batch_predict(&datasets)?;

// Efficient configuration management
let config = ModelConfigBuilder::<f32>::new()
    .with_model_type("efficient_lstm")
    .build()?;
config.validate()?; // Validate once, reuse many times
```

## Advanced Usage

### Custom Error Types

Extend the error system for domain-specific errors:

```rust
use neuro_divergent_core::error::*;

#[derive(Debug, thiserror::Error)]
pub enum MyCustomError {
    #[error("Custom model error: {message}")]
    ModelSpecific { message: String },
}

impl From<MyCustomError> for NeuroDivergentError {
    fn from(err: MyCustomError) -> Self {
        ErrorBuilder::prediction(err.to_string())
            .context("custom_error_type", "model_specific")
            .build()
    }
}
```

### Custom Preprocessing Pipelines

```rust
pub struct PipelinePreprocessor<T: Float> {
    steps: Vec<Box<dyn PreprocessingStep<T>>>,
}

impl<T: Float> PipelinePreprocessor<T> {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }
    
    pub fn add_step(mut self, step: Box<dyn PreprocessingStep<T>>) -> Self {
        self.steps.push(step);
        self
    }
}

impl<T: Float> InputPreprocessor<T> for PipelinePreprocessor<T> {
    fn process(&self, input: &TimeSeriesInput<T>) -> NeuroDivergentResult<Vec<T>> {
        let mut data = input.target_history.clone();
        
        for step in &self.steps {
            data = step.process(data)?;
        }
        
        Ok(data)
    }
}
```

### Multi-Model Ensembles

```rust
pub struct EnsembleForecaster<T: Float> {
    models: Vec<Box<dyn BaseModel<T>>>,
    weights: Vec<T>,
    aggregation_method: AggregationMethod,
}

impl<T: Float + Send + Sync> BaseModel<T> for EnsembleForecaster<T> {
    type Config = EnsembleConfig<T>;
    type State = EnsembleState<T>;
    
    fn predict(&self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<ForecastResult<T>> {
        let predictions: Vec<_> = self.models
            .iter()
            .map(|model| model.predict(data))
            .collect::<Result<Vec<_>, _>>()?;
        
        let aggregated = self.aggregate_predictions(predictions)?;
        Ok(aggregated)
    }
}
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-org/ruv-FANN.git
cd ruv-FANN/neuro-divergent/neuro-divergent-core

# Run tests
cargo test

# Run benchmarks
cargo bench

# Check documentation
cargo doc --open

# Lint and format
cargo clippy
cargo fmt
```

### Performance Benchmarks

```bash
# Run core benchmarks
cargo bench --bench core_benchmarks

# Profile memory usage
cargo test --release --features=profiling
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Related Crates

- [`ruv-fann`](../ruv-fann/) - High-performance neural network library
- [`neuro-divergent-models`](../neuro-divergent-models/) - Pre-built forecasting models
- [`neuro-divergent-data`](../neuro-divergent-data/) - Advanced data processing utilities
- [`neuro-divergent-cli`](../neuro-divergent-cli/) - Command-line interface