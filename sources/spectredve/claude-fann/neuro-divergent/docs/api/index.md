# Neuro-Divergent API Documentation

Welcome to the comprehensive API documentation for **Neuro-Divergent**, a high-performance neural forecasting library that provides 100% compatibility with the NeuralForecast Python API while delivering the performance and safety benefits of Rust.

## Overview

Neuro-Divergent is built on the ruv-FANN neural network foundation and offers high-performance neural forecasting capabilities with a user-friendly API. The library provides a complete ecosystem for time series forecasting with neural networks.

### Key Features

- **100% NeuralForecast API Compatibility**: Drop-in replacement for Python users
- **High Performance**: Rust performance with SIMD optimization  
- **Memory Safety**: Zero-cost abstractions with compile-time guarantees
- **Async Support**: Asynchronous training and prediction
- **Multiple Model Support**: LSTM, NBEATS, DeepAR, Transformers, and more
- **Extensible Architecture**: Easy to add custom models and components

## Quick Start

```rust
use neuro_divergent::{NeuralForecast, models::LSTM, data::TimeSeriesDataFrame, Frequency};

// Create LSTM model
let lstm = LSTM::builder()
    .hidden_size(128)
    .num_layers(2)
    .dropout(0.1)
    .horizon(12)
    .input_size(24)
    .build()?;

// Create NeuralForecast instance
let mut nf = NeuralForecast::builder()
    .with_model(Box::new(lstm))
    .with_frequency(Frequency::Monthly)
    .build()?;

// Load your time series data
let data = TimeSeriesDataFrame::from_csv("data.csv")?;

// Fit the model
nf.fit(data.clone())?;

// Generate forecasts
let forecasts = nf.predict()?;
```

## Architecture

The library is structured in several layers:

- **API Layer**: User-facing interface with NeuralForecast compatibility
- **Model Layer**: Neural network model implementations  
- **Core Layer**: Base traits and abstractions
- **Data Layer**: Time series data handling and preprocessing
- **Foundation Layer**: Integration with ruv-FANN neural networks

## API Reference

### Core Components

- [**Core Traits**](core-traits.md) - BaseModel, ModelConfig, ForecastingEngine and fundamental abstractions
- [**NeuralForecast**](neural-forecast.md) - Main API class for forecasting operations
- [**Data Types**](data-types.md) - TimeSeriesDataFrame, schemas, and data structures
- [**Configuration**](configurations.md) - System configuration and settings
- [**Error Handling**](errors.md) - Error types and handling strategies

### Models

- [**Models Overview**](models/index.md) - Complete guide to all available models
- [**Basic Models**](models/basic.md) - Simple linear and MLP models
- [**Recurrent Models**](models/recurrent.md) - LSTM, GRU, and RNN variants
- [**Advanced Models**](models/advanced.md) - NBEATS, N-HiTS, and sophisticated architectures
- [**Transformer Models**](models/transformer.md) - Attention-based forecasting models
- [**Specialized Models**](models/specialized.md) - Domain-specific and experimental models

### Training & Optimization

- [**Training System**](training.md) - Optimizers, loss functions, and training algorithms
- [**Evaluation**](evaluation.md) - Metrics, cross-validation, and model assessment
- [**Builders**](builders.md) - Fluent API and builder patterns for easy configuration

## Usage Patterns

### Basic Workflow

1. **Data Preparation**: Load and preprocess time series data
2. **Model Selection**: Choose appropriate forecasting models
3. **Configuration**: Set up training parameters and system settings
4. **Training**: Fit models to historical data
5. **Evaluation**: Assess model performance using cross-validation
6. **Forecasting**: Generate predictions for future periods

### Advanced Features

- **Ensemble Forecasting**: Combine multiple models for better accuracy
- **Cross-Validation**: Robust model evaluation with time series splits
- **Prediction Intervals**: Quantify uncertainty in forecasts
- **Custom Models**: Extend the library with your own model implementations

## Performance Considerations

### Memory Usage

- **Streaming Processing**: Handle large datasets with constant memory usage
- **Lazy Evaluation**: Polars-based lazy computation for efficiency
- **Zero-Copy Operations**: Minimize memory allocations where possible

### Computational Performance

- **SIMD Optimization**: Vectorized operations for numerical computations
- **Parallel Processing**: Multi-threaded training and prediction
- **GPU Support**: Optional GPU acceleration for compatible models

### Scalability

- **Batch Processing**: Process multiple time series efficiently
- **Async Support**: Non-blocking operations for concurrent workloads
- **Memory-Mapped Files**: Handle datasets larger than available RAM

## Integration

### Data Formats

- **CSV**: Standard comma-separated values
- **Parquet**: Efficient columnar storage format
- **JSON**: Flexible semi-structured data
- **Polars DataFrames**: Native integration with Polars ecosystem

### External Libraries

- **ruv-FANN**: Core neural network engine
- **Polars**: High-performance data manipulation
- **Chrono**: Date and time handling
- **Serde**: Serialization and configuration management

## Type Safety

Neuro-Divergent leverages Rust's type system for compile-time guarantees:

- **Generic Types**: Support for different floating-point precisions (f32, f64)
- **Trait Bounds**: Ensure proper numeric behavior across all operations
- **Error Types**: Comprehensive error handling with specific error categories
- **Memory Safety**: Eliminate data races and memory leaks at compile time

## Examples by Use Case

### Financial Forecasting
```rust
// Configure for daily financial data with volatility modeling
let config = NeuralForecast::builder()
    .with_frequency(Frequency::BusinessDaily)
    .with_local_scaler(ScalerType::RobustScaler)
    .with_prediction_intervals(PredictionIntervals::new(
        vec![0.95, 0.99], 
        IntervalMethod::ConformalPrediction
    )?);
```

### IoT Sensor Data
```rust
// High-frequency sensor data with minute-level predictions
let config = NeuralForecast::builder()
    .with_frequency(Frequency::Minute)
    .with_num_threads(8)
    .with_device(Device::GPU(0));
```

### Retail Demand Forecasting
```rust
// Weekly retail data with seasonal patterns
let config = NeuralForecast::builder()
    .with_frequency(Frequency::Weekly)
    .with_local_scaler(ScalerType::StandardScaler);
```

## Version Compatibility

| Version | Rust Version | Features |
|---------|-------------|----------|
| 0.1.x   | 1.70+       | Core functionality, basic models |
| 0.2.x   | 1.72+       | Advanced models, GPU support |
| 0.3.x   | 1.74+       | Transformer models, async API |

## Contributing

The API is designed for extensibility. Key extension points:

- **Custom Models**: Implement the `BaseModel` trait
- **Custom Loss Functions**: Implement the `LossFunction` trait  
- **Custom Optimizers**: Implement the `Optimizer` trait
- **Custom Data Transforms**: Implement the `DataTransform` trait

## Support

- **Documentation**: Complete API reference with examples
- **Type Safety**: Comprehensive compile-time checking
- **Error Messages**: Detailed error information with context
- **Performance**: Built-in benchmarking and profiling support

---

*This documentation covers all public APIs. For implementation details and internal architecture, see the source code documentation generated with `cargo doc`.*