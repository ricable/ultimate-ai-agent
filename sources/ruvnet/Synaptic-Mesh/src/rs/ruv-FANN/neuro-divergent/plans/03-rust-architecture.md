# Rust Architecture Design for neuro-divergent

This document outlines the complete Rust architecture for the neuro-divergent library, providing 100% API parity with NeuralForecast while leveraging the high-performance ruv-FANN neural network foundation.

## Architecture Overview

The neuro-divergent library follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Public API Layer                        │
│  (NeuralForecast compatibility + Rust-native extensions)   │
├─────────────────────────────────────────────────────────────┤
│                  Model Implementation Layer                 │
│        (LSTM, NBEATS, DeepAR, Transformers, etc.)         │
├─────────────────────────────────────────────────────────────┤
│                    Core Abstraction Layer                  │
│     (Traits, Configuration, Training, Prediction)          │
├─────────────────────────────────────────────────────────────┤
│                   Data Processing Layer                    │
│    (Time Series, Scaling, Validation, Serialization)      │
├─────────────────────────────────────────────────────────────┤
│                 Neural Network Foundation                  │
│                     (ruv-FANN)                            │
└─────────────────────────────────────────────────────────────┘
```

## Module Hierarchy

### Core Modules

```rust
neuro_divergent/
├── src/
│   ├── lib.rs                    // Main library entry point
│   ├── core/                     // Core abstractions and traits
│   │   ├── mod.rs
│   │   ├── base_model.rs         // BaseModel trait definition
│   │   ├── forecasting.rs        // Forecasting interfaces
│   │   ├── training.rs           // Training abstractions
│   │   ├── config.rs             // Configuration management
│   │   └── registry.rs           // Model registry system
│   ├── models/                   // Neural forecasting models
│   │   ├── mod.rs
│   │   ├── lstm.rs               // LSTM implementation
│   │   ├── nbeats.rs             // NBEATS implementation
│   │   ├── deepar.rs             // DeepAR implementation
│   │   ├── transformer.rs        // Transformer variants
│   │   ├── nhits.rs              // NHITS implementation
│   │   └── rnn.rs                // Basic RNN variants
│   ├── data/                     // Data handling and processing
│   │   ├── mod.rs
│   │   ├── dataset.rs            // TimeSeriesDataset
│   │   ├── preprocessing.rs      // Data preprocessing
│   │   ├── scaling.rs            // Scaling and normalization
│   │   ├── validation.rs         // Data validation
│   │   └── transforms.rs         // Data transformations
│   ├── training/                 // Training infrastructure
│   │   ├── mod.rs
│   │   ├── optimizers.rs         // Optimizer implementations
│   │   ├── losses.rs             // Loss functions
│   │   ├── schedulers.rs         // Learning rate schedulers
│   │   ├── callbacks.rs          // Training callbacks
│   │   └── metrics.rs            // Evaluation metrics
│   ├── utils/                    // Utilities and helpers
│   │   ├── mod.rs
│   │   ├── math.rs               // Mathematical utilities
│   │   ├── time.rs               // Time series utilities
│   │   ├── distributions.rs      // Probability distributions
│   │   └── serialization.rs      // Model serialization
│   ├── integration/              // ruv-FANN integration
│   │   ├── mod.rs
│   │   ├── network_adapter.rs    // Adapter for ruv-FANN networks
│   │   ├── activation_bridge.rs  // Activation function bridge
│   │   └── training_bridge.rs    // Training algorithm bridge
│   └── errors.rs                 // Error handling
```

## Core Trait Definitions

### BaseModel Trait

```rust
use std::collections::HashMap;
use ruv_fann::Network;
use crate::data::TimeSeriesDataset;
use crate::errors::NeuroDivergentResult;

/// Core trait that all forecasting models must implement
pub trait BaseModel<T: Float + Send + Sync>: Send + Sync {
    type Config: ModelConfig<T>;
    type State: ModelState<T>;
    
    /// Create a new model instance with configuration
    fn new(config: Self::Config) -> NeuroDivergentResult<Self> where Self: Sized;
    
    /// Fit the model to training data
    fn fit(&mut self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<()>;
    
    /// Generate forecasts for the given dataset
    fn predict(&self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<ForecastResult<T>>;
    
    /// Perform cross-validation
    fn cross_validation(
        &mut self, 
        data: &TimeSeriesDataset<T>,
        config: CrossValidationConfig
    ) -> NeuroDivergentResult<CrossValidationResult<T>>;
    
    /// Get model configuration
    fn config(&self) -> &Self::Config;
    
    /// Get internal model state for serialization
    fn state(&self) -> &Self::State;
    
    /// Restore model from saved state
    fn restore_state(&mut self, state: Self::State) -> NeuroDivergentResult<()>;
    
    /// Reset model to initial state
    fn reset(&mut self) -> NeuroDivergentResult<()>;
    
    /// Get model metadata
    fn metadata(&self) -> ModelMetadata;
    
    /// Validate input data compatibility
    fn validate_data(&self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<()>;
}
```

### ModelConfig Trait

```rust
/// Configuration trait for model parameters
pub trait ModelConfig<T: Float>: Clone + Send + Sync {
    /// Validate configuration parameters
    fn validate(&self) -> NeuroDivergentResult<()>;
    
    /// Get forecast horizon
    fn horizon(&self) -> usize;
    
    /// Get input window size
    fn input_size(&self) -> usize;
    
    /// Get output size
    fn output_size(&self) -> usize;
    
    /// Get exogenous variable configuration
    fn exogenous_config(&self) -> &ExogenousConfig;
    
    /// Convert to builder pattern for fluent configuration
    fn builder() -> ConfigBuilder<Self, T> where Self: Sized;
}
```

### ForecastingEngine Trait

```rust
/// High-level forecasting engine trait
pub trait ForecastingEngine<T: Float + Send + Sync>: Send + Sync {
    /// Batch prediction for multiple time series
    fn batch_predict(
        &self,
        datasets: &[TimeSeriesDataset<T>]
    ) -> NeuroDivergentResult<Vec<ForecastResult<T>>>;
    
    /// Probabilistic forecasting with prediction intervals
    fn predict_intervals(
        &self,
        data: &TimeSeriesDataset<T>,
        confidence_levels: &[f64]
    ) -> NeuroDivergentResult<IntervalForecast<T>>;
    
    /// Quantile forecasting
    fn predict_quantiles(
        &self,
        data: &TimeSeriesDataset<T>,
        quantiles: &[f64]
    ) -> NeuroDivergentResult<QuantileForecast<T>>;
}
```

## Integration with ruv-FANN

### NetworkAdapter

```rust
/// Adapter for integrating ruv-FANN networks with forecasting models
pub struct NetworkAdapter<T: Float> {
    network: Network<T>,
    input_preprocessor: Box<dyn InputPreprocessor<T>>,
    output_postprocessor: Box<dyn OutputPostprocessor<T>>,
    activation_mapper: ActivationMapper,
}

impl<T: Float> NetworkAdapter<T> {
    /// Create adapter from ruv-FANN network
    pub fn from_network(network: Network<T>) -> Self;
    
    /// Configure input preprocessing
    pub fn with_input_processor(
        mut self, 
        processor: Box<dyn InputPreprocessor<T>>
    ) -> Self;
    
    /// Configure output postprocessing
    pub fn with_output_processor(
        mut self, 
        processor: Box<dyn OutputPostprocessor<T>>
    ) -> Self;
    
    /// Forward pass through the network
    pub fn forward(&mut self, input: &TimeSeriesInput<T>) -> NeuroDivergentResult<NetworkOutput<T>>;
    
    /// Get underlying network for direct access
    pub fn network(&self) -> &Network<T>;
    
    /// Get mutable reference to underlying network
    pub fn network_mut(&mut self) -> &mut Network<T>;
}
```

### Training Integration

```rust
/// Bridge between neuro-divergent training and ruv-FANN training algorithms
pub struct TrainingBridge<T: Float> {
    algorithm: Box<dyn ruv_fann::TrainingAlgorithm<T>>,
    loss_function: Box<dyn LossFunction<T>>,
    optimizer_config: OptimizerConfig<T>,
}

impl<T: Float> TrainingBridge<T> {
    /// Train network using ruv-FANN algorithms
    pub fn train_network(
        &mut self,
        network: &mut Network<T>,
        data: &TimeSeriesDataset<T>,
        config: &TrainingConfig<T>
    ) -> NeuroDivergentResult<TrainingResult<T>>;
    
    /// Setup cascade correlation training
    pub fn setup_cascade_training(
        &mut self,
        config: CascadeConfig<T>
    ) -> NeuroDivergentResult<()>;
}
```

## Model Implementations

### LSTM Model Structure

```rust
/// LSTM-based forecasting model
pub struct LSTM<T: Float> {
    config: LSTMConfig<T>,
    state: LSTMState<T>,
    network_adapter: NetworkAdapter<T>,
    encoder: LSTMEncoder<T>,
    decoder: LSTMDecoder<T>,
    loss_function: Box<dyn LossFunction<T>>,
}

#[derive(Clone)]
pub struct LSTMConfig<T: Float> {
    pub horizon: usize,
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub dropout: T,
    pub bidirectional: bool,
    pub encoder_hidden_size: usize,
    pub decoder_hidden_size: usize,
    pub max_steps: usize,
    pub learning_rate: T,
    pub exogenous_config: ExogenousConfig,
    pub scaler_type: ScalerType,
}

impl<T: Float + Send + Sync> BaseModel<T> for LSTM<T> {
    type Config = LSTMConfig<T>;
    type State = LSTMState<T>;
    
    // Implementation of BaseModel methods...
}
```

### NBEATS Model Structure

```rust
/// NBEATS interpretable forecasting model
pub struct NBEATS<T: Float> {
    config: NBEATSConfig<T>,
    state: NBEATSState<T>,
    stacks: Vec<NBEATSStack<T>>,
    basis_functions: HashMap<StackType, Box<dyn BasisFunction<T>>>,
}

#[derive(Clone)]
pub struct NBEATSConfig<T: Float> {
    pub horizon: usize,
    pub input_size: usize,
    pub stack_types: Vec<StackType>,
    pub n_blocks: Vec<usize>,
    pub mlp_units: Vec<Vec<usize>>,
    pub shared_weights: bool,
    pub activation: ActivationFunction,
    pub max_steps: usize,
    pub learning_rate: T,
}

/// Stack types for NBEATS
#[derive(Debug, Clone, PartialEq)]
pub enum StackType {
    Identity,
    Trend,
    Seasonality,
    Generic,
}

/// Trait for basis function implementations
pub trait BasisFunction<T: Float>: Send + Sync {
    fn generate_basis(&self, theta: &[T], backcast_length: usize, forecast_length: usize) -> (Vec<T>, Vec<T>);
    fn basis_size(&self) -> usize;
}
```

## Data Processing Layer

### TimeSeriesDataset

```rust
/// Core time series dataset structure
#[derive(Debug, Clone)]
pub struct TimeSeriesDataset<T: Float> {
    pub data: DataFrame<T>,
    pub target_col: String,
    pub time_col: String,
    pub unique_id_col: String,
    pub static_features: Option<HashMap<String, Vec<T>>>,
    pub exogenous_features: Option<DataFrame<T>>,
    pub metadata: DatasetMetadata,
}

impl<T: Float> TimeSeriesDataset<T> {
    /// Create new dataset from DataFrame
    pub fn new(
        data: DataFrame<T>,
        target_col: String,
        time_col: String,
        unique_id_col: String
    ) -> NeuroDivergentResult<Self>;
    
    /// Add static features
    pub fn with_static_features(mut self, features: HashMap<String, Vec<T>>) -> Self;
    
    /// Add exogenous features
    pub fn with_exogenous_features(mut self, features: DataFrame<T>) -> Self;
    
    /// Validate dataset structure
    pub fn validate(&self) -> NeuroDivergentResult<()>;
    
    /// Split dataset for training/testing
    pub fn train_test_split(&self, test_size: f64) -> NeuroDivergentResult<(Self, Self)>;
    
    /// Create time-based splits for cross-validation
    pub fn time_series_split(&self, n_splits: usize) -> NeuroDivergentResult<Vec<(Self, Self)>>;
    
    /// Get unique time series identifiers
    pub fn unique_ids(&self) -> Vec<String>;
    
    /// Filter by unique ID
    pub fn filter_by_id(&self, id: &str) -> NeuroDivergentResult<Self>;
    
    /// Apply preprocessing transformations
    pub fn preprocess(&self, config: &PreprocessingConfig<T>) -> NeuroDivergentResult<Self>;
}
```

### Scaling and Preprocessing

```rust
/// Scaler trait for data normalization
pub trait Scaler<T: Float>: Send + Sync {
    fn fit(&mut self, data: &[T]) -> NeuroDivergentResult<()>;
    fn transform(&self, data: &[T]) -> NeuroDivergentResult<Vec<T>>;
    fn inverse_transform(&self, data: &[T]) -> NeuroDivergentResult<Vec<T>>;
    fn fit_transform(&mut self, data: &[T]) -> NeuroDivergentResult<Vec<T>>;
}

/// Standard scaler implementation
pub struct StandardScaler<T: Float> {
    mean: T,
    std: T,
    fitted: bool,
}

/// Min-max scaler implementation
pub struct MinMaxScaler<T: Float> {
    min_val: T,
    max_val: T,
    scale: T,
    fitted: bool,
}

/// Robust scaler using median and IQR
pub struct RobustScaler<T: Float> {
    median: T,
    iqr: T,
    fitted: bool,
}
```

## Error Handling and Memory Management

### Error Types

```rust
/// Comprehensive error handling for neuro-divergent
#[derive(Error, Debug)]
pub enum NeuroDivergentError {
    #[error("Model configuration error: {0}")]
    ConfigError(String),
    
    #[error("Data validation error: {0}")]
    DataError(String),
    
    #[error("Training error: {0}")]
    TrainingError(String),
    
    #[error("Prediction error: {0}")]
    PredictionError(String),
    
    #[error("Network integration error: {0}")]
    NetworkError(#[from] ruv_fann::NetworkError),
    
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Memory allocation error: {0}")]
    MemoryError(String),
    
    #[error("Compatibility error: {0}")]
    CompatibilityError(String),
}

pub type NeuroDivergentResult<T> = Result<T, NeuroDivergentError>;
```

### Memory Management Strategy

```rust
/// Memory pool for efficient allocation during training
pub struct MemoryPool<T: Float> {
    buffers: Vec<Vec<T>>,
    available: Vec<usize>,
    total_allocated: usize,
    max_allocation: usize,
}

impl<T: Float> MemoryPool<T> {
    /// Create new memory pool with size limit
    pub fn with_limit(max_mb: usize) -> Self;
    
    /// Allocate buffer of specified size
    pub fn allocate(&mut self, size: usize) -> NeuroDivergentResult<BufferHandle>;
    
    /// Return buffer to pool
    pub fn deallocate(&mut self, handle: BufferHandle);
    
    /// Get current memory usage
    pub fn memory_usage(&self) -> MemoryUsage;
    
    /// Cleanup unused buffers
    pub fn cleanup(&mut self);
}
```

## Performance Optimization Strategies

### Parallel Processing

```rust
/// Parallel execution configuration
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    pub max_threads: Option<usize>,
    pub batch_size: usize,
    pub enable_model_parallelism: bool,
    pub enable_data_parallelism: bool,
}

/// Parallel batch processing for multiple time series
pub struct BatchProcessor<T: Float> {
    thread_pool: rayon::ThreadPool,
    config: ParallelConfig,
}

impl<T: Float + Send + Sync> BatchProcessor<T> {
    /// Process multiple datasets in parallel
    pub fn process_batch<M: BaseModel<T> + Clone>(
        &self,
        model: &M,
        datasets: &[TimeSeriesDataset<T>]
    ) -> NeuroDivergentResult<Vec<ForecastResult<T>>>;
    
    /// Parallel cross-validation
    pub fn parallel_cross_validation<M: BaseModel<T> + Clone>(
        &self,
        model: &mut M,
        data: &TimeSeriesDataset<T>,
        config: CrossValidationConfig
    ) -> NeuroDivergentResult<CrossValidationResult<T>>;
}
```

### SIMD Optimization

```rust
/// SIMD-optimized operations for time series processing
pub mod simd {
    use std::simd::*;
    
    /// SIMD-optimized dot product
    pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32;
    
    /// SIMD-optimized element-wise operations
    pub fn elementwise_add_f32(a: &[f32], b: &[f32], result: &mut [f32]);
    
    /// SIMD-optimized activation functions
    pub fn sigmoid_f32(input: &[f32], output: &mut [f32]);
    
    /// SIMD-optimized matrix multiplication
    pub fn matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize);
}
```

## Integration Points with ruv-FANN

### Network Configuration Mapping

```rust
/// Maps NeuralForecast model configurations to ruv-FANN network structures
pub struct NetworkConfigMapper<T: Float> {
    activation_mapping: HashMap<String, ruv_fann::ActivationFunction>,
    layer_mapping: LayerMappingStrategy,
}

impl<T: Float> NetworkConfigMapper<T> {
    /// Convert LSTM config to ruv-FANN network
    pub fn lstm_to_network(&self, config: &LSTMConfig<T>) -> NeuroDivergentResult<NetworkBuilder<T>>;
    
    /// Convert NBEATS config to ruv-FANN network
    pub fn nbeats_to_network(&self, config: &NBEATSConfig<T>) -> NeuroDivergentResult<NetworkBuilder<T>>;
    
    /// Generic model config to network conversion
    pub fn model_to_network<C: ModelConfig<T>>(
        &self, 
        config: &C
    ) -> NeuroDivergentResult<NetworkBuilder<T>>;
}
```

### Training Algorithm Integration

```rust
/// Integrates neuro-divergent training with ruv-FANN algorithms
pub struct TrainingIntegrator<T: Float> {
    ruv_fann_trainer: Box<dyn ruv_fann::TrainingAlgorithm<T>>,
    loss_adapter: LossAdapter<T>,
    callback_bridge: CallbackBridge<T>,
}

impl<T: Float> TrainingIntegrator<T> {
    /// Setup training with ruv-FANN backend
    pub fn setup_training(
        &mut self,
        model_config: &dyn ModelConfig<T>,
        training_config: &TrainingConfig<T>
    ) -> NeuroDivergentResult<()>;
    
    /// Execute training epoch
    pub fn train_epoch(
        &mut self,
        network: &mut Network<T>,
        data: &TimeSeriesDataset<T>
    ) -> NeuroDivergentResult<TrainingMetrics<T>>;
}
```

This architecture provides a solid foundation for implementing the neuro-divergent library with complete NeuralForecast compatibility while leveraging the high-performance ruv-FANN backend for neural network operations.