# API Design for neuro-divergent

This document details the Rust API signatures that provide 100% compatibility with NeuralForecast's Python API while leveraging Rust's type system for safety and performance.

## Public API Surface

### Main NeuralForecast Class

```rust
/// Main entry point for neural forecasting, equivalent to Python's NeuralForecast class
pub struct NeuralForecast<T: Float + Send + Sync> {
    models: Vec<Box<dyn BaseModel<T>>>,
    frequency: Frequency,
    local_scaler_type: Option<ScalerType>,
    num_threads: Option<usize>,
    prediction_intervals: Option<PredictionIntervals>,
}

impl<T: Float + Send + Sync> NeuralForecast<T> {
    /// Create new NeuralForecast instance with models
    /// 
    /// # Example
    /// ```rust
    /// use neuro_divergent::{NeuralForecast, models::LSTM};
    /// 
    /// let nf = NeuralForecast::new()
    ///     .with_models(vec![
    ///         Box::new(LSTM::new(lstm_config)?),
    ///     ])
    ///     .with_frequency(Frequency::Monthly)
    ///     .build()?;
    /// ```
    pub fn new() -> NeuralForecastBuilder<T> {
        NeuralForecastBuilder::new()
    }
    
    /// Fit models to the provided time series data
    /// 
    /// Equivalent to Python: `nf.fit(df)`
    pub fn fit(&mut self, data: TimeSeriesDataFrame<T>) -> NeuroDivergentResult<()> {
        self.fit_with_validation(data, ValidationConfig::default())
    }
    
    /// Fit with custom validation configuration
    pub fn fit_with_validation(
        &mut self, 
        data: TimeSeriesDataFrame<T>,
        validation_config: ValidationConfig
    ) -> NeuroDivergentResult<()>;
    
    /// Generate forecasts for all fitted models
    /// 
    /// Equivalent to Python: `nf.predict()`
    pub fn predict(&self) -> NeuroDivergentResult<ForecastDataFrame<T>> {
        self.predict_with_config(PredictionConfig::default())
    }
    
    /// Generate forecasts with custom configuration
    pub fn predict_with_config(
        &self, 
        config: PredictionConfig
    ) -> NeuroDivergentResult<ForecastDataFrame<T>>;
    
    /// Predict on new data
    /// 
    /// Equivalent to Python: `nf.predict(df=new_data)`
    pub fn predict_on(
        &self, 
        data: TimeSeriesDataFrame<T>
    ) -> NeuroDivergentResult<ForecastDataFrame<T>>;
    
    /// Cross-validation for model evaluation
    /// 
    /// Equivalent to Python: `nf.cross_validation(df, n_windows=3)`
    pub fn cross_validation(
        &mut self,
        data: TimeSeriesDataFrame<T>,
        config: CrossValidationConfig
    ) -> NeuroDivergentResult<CrossValidationDataFrame<T>>;
    
    /// Fit and predict in one step
    pub fn fit_predict(
        &mut self,
        train_data: TimeSeriesDataFrame<T>
    ) -> NeuroDivergentResult<ForecastDataFrame<T>>;
    
    /// Get model by name
    pub fn get_model(&self, name: &str) -> Option<&dyn BaseModel<T>>;
    
    /// Get mutable model by name
    pub fn get_model_mut(&mut self, name: &str) -> Option<&mut dyn BaseModel<T>>;
    
    /// List all model names
    pub fn model_names(&self) -> Vec<String>;
    
    /// Save models to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> NeuroDivergentResult<()>;
    
    /// Load models from file
    pub fn load<P: AsRef<Path>>(path: P) -> NeuroDivergentResult<Self>;
}
```

### Builder Pattern for NeuralForecast

```rust
/// Builder for constructing NeuralForecast instances with fluent API
pub struct NeuralForecastBuilder<T: Float + Send + Sync> {
    models: Vec<Box<dyn BaseModel<T>>>,
    frequency: Option<Frequency>,
    local_scaler_type: Option<ScalerType>,
    num_threads: Option<usize>,
    prediction_intervals: Option<PredictionIntervals>,
}

impl<T: Float + Send + Sync> NeuralForecastBuilder<T> {
    pub fn new() -> Self;
    
    /// Add models to the forecasting ensemble
    pub fn with_models(mut self, models: Vec<Box<dyn BaseModel<T>>>) -> Self;
    
    /// Add a single model
    pub fn with_model(mut self, model: Box<dyn BaseModel<T>>) -> Self;
    
    /// Set the frequency of the time series
    pub fn with_frequency(mut self, frequency: Frequency) -> Self;
    
    /// Set local scaler type for data preprocessing
    pub fn with_local_scaler(mut self, scaler_type: ScalerType) -> Self;
    
    /// Set number of threads for parallel processing
    pub fn with_num_threads(mut self, num_threads: usize) -> Self;
    
    /// Configure prediction intervals
    pub fn with_prediction_intervals(mut self, intervals: PredictionIntervals) -> Self;
    
    /// Build the NeuralForecast instance
    pub fn build(self) -> NeuroDivergentResult<NeuralForecast<T>>;
}
```

## Core Type Definitions

### Time Series Data Structures

```rust
/// Main data structure for time series data, equivalent to pandas DataFrame
#[derive(Debug, Clone)]
pub struct TimeSeriesDataFrame<T: Float> {
    pub data: polars::DataFrame,
    pub schema: TimeSeriesSchema,
    phantom: PhantomData<T>,
}

impl<T: Float> TimeSeriesDataFrame<T> {
    /// Create from Polars DataFrame
    pub fn from_polars(
        df: polars::DataFrame, 
        schema: TimeSeriesSchema
    ) -> NeuroDivergentResult<Self>;
    
    /// Create from CSV file
    pub fn from_csv<P: AsRef<Path>>(
        path: P, 
        schema: TimeSeriesSchema
    ) -> NeuroDivergentResult<Self>;
    
    /// Create from Parquet file
    pub fn from_parquet<P: AsRef<Path>>(
        path: P, 
        schema: TimeSeriesSchema
    ) -> NeuroDivergentResult<Self>;
    
    /// Get unique time series identifiers
    pub fn unique_ids(&self) -> Vec<String>;
    
    /// Filter by date range
    pub fn filter_date_range(
        &self, 
        start: DateTime<Utc>, 
        end: DateTime<Utc>
    ) -> NeuroDivergentResult<Self>;
    
    /// Filter by unique ID
    pub fn filter_by_id(&self, id: &str) -> NeuroDivergentResult<Self>;
    
    /// Add exogenous variables
    pub fn with_exogenous(
        mut self, 
        exogenous_data: polars::DataFrame
    ) -> NeuroDivergentResult<Self>;
    
    /// Convert to internal dataset format
    pub fn to_dataset(&self) -> NeuroDivergentResult<TimeSeriesDataset<T>>;
    
    /// Get number of time series
    pub fn n_series(&self) -> usize;
    
    /// Get time range
    pub fn time_range(&self) -> (DateTime<Utc>, DateTime<Utc>);
    
    /// Validate data integrity
    pub fn validate(&self) -> NeuroDivergentResult<ValidationReport>;
}

/// Schema definition for time series data
#[derive(Debug, Clone)]
pub struct TimeSeriesSchema {
    pub unique_id_col: String,
    pub ds_col: String,           // Date/time column
    pub y_col: String,            // Target variable column
    pub static_features: Vec<String>,
    pub exogenous_features: Vec<String>,
}

impl TimeSeriesSchema {
    /// Create schema with required columns
    pub fn new(
        unique_id_col: impl Into<String>,
        ds_col: impl Into<String>,
        y_col: impl Into<String>
    ) -> Self;
    
    /// Add static features
    pub fn with_static_features(mut self, features: Vec<String>) -> Self;
    
    /// Add exogenous features
    pub fn with_exogenous_features(mut self, features: Vec<String>) -> Self;
    
    /// Validate schema against DataFrame
    pub fn validate_dataframe(&self, df: &polars::DataFrame) -> NeuroDivergentResult<()>;
}
```

### Forecast Results

```rust
/// Results from forecasting operations
#[derive(Debug, Clone)]
pub struct ForecastDataFrame<T: Float> {
    pub data: polars::DataFrame,
    pub models: Vec<String>,
    pub forecast_horizon: usize,
    pub confidence_levels: Option<Vec<f64>>,
    phantom: PhantomData<T>,
}

impl<T: Float> ForecastDataFrame<T> {
    /// Get forecasts for specific model
    pub fn get_model_forecasts(&self, model_name: &str) -> NeuroDivergentResult<polars::DataFrame>;
    
    /// Get all model forecasts
    pub fn get_all_forecasts(&self) -> Vec<(String, polars::DataFrame)>;
    
    /// Convert to point forecasts only
    pub fn to_point_forecasts(&self) -> polars::DataFrame;
    
    /// Get prediction intervals if available
    pub fn get_prediction_intervals(&self, model_name: &str) -> Option<polars::DataFrame>;
    
    /// Export to CSV
    pub fn to_csv<P: AsRef<Path>>(&self, path: P) -> NeuroDivergentResult<()>;
    
    /// Export to Parquet
    pub fn to_parquet<P: AsRef<Path>>(&self, path: P) -> NeuroDivergentResult<()>;
    
    /// Calculate forecast accuracy metrics
    pub fn calculate_metrics(
        &self, 
        actual_data: &TimeSeriesDataFrame<T>
    ) -> NeuroDivergentResult<AccuracyMetrics<T>>;
}

/// Cross-validation results
#[derive(Debug, Clone)]
pub struct CrossValidationDataFrame<T: Float> {
    pub data: polars::DataFrame,
    pub cutoffs: Vec<DateTime<Utc>>,
    pub models: Vec<String>,
    pub metrics: AccuracyMetrics<T>,
}
```

## Model Configuration Types

### Generic Model Configuration

```rust
/// Base configuration trait for all models
pub trait ModelConfig<T: Float>: Clone + Send + Sync + 'static {
    /// Model name identifier
    fn model_name() -> &'static str where Self: Sized;
    
    /// Validate configuration parameters
    fn validate(&self) -> NeuroDivergentResult<()>;
    
    /// Get forecast horizon
    fn horizon(&self) -> usize;
    
    /// Get input window size
    fn input_size(&self) -> usize;
    
    /// Convert to generic configuration
    fn to_generic(&self) -> GenericModelConfig<T>;
    
    /// Create default configuration
    fn default_config() -> Self where Self: Sized;
}

/// Generic configuration container
#[derive(Debug, Clone)]
pub struct GenericModelConfig<T: Float> {
    pub model_type: String,
    pub horizon: usize,
    pub input_size: usize,
    pub parameters: HashMap<String, ConfigValue<T>>,
}

/// Configuration value types
#[derive(Debug, Clone)]
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

### LSTM Configuration

```rust
/// Configuration for LSTM models
#[derive(Debug, Clone)]
pub struct LSTMConfig<T: Float> {
    // Required parameters
    pub horizon: usize,
    pub input_size: usize,
    
    // Architecture parameters
    pub hidden_size: usize,
    pub num_layers: usize,
    pub dropout: T,
    pub bidirectional: bool,
    
    // Encoder-decoder parameters
    pub encoder_hidden_size: usize,
    pub decoder_hidden_size: usize,
    
    // Training parameters
    pub max_steps: usize,
    pub learning_rate: T,
    pub weight_decay: T,
    pub gradient_clip_val: Option<T>,
    
    // Data parameters
    pub scaler_type: ScalerType,
    pub static_features: Option<Vec<String>>,
    pub hist_exog_features: Option<Vec<String>>,
    pub futr_exog_features: Option<Vec<String>>,
    
    // Advanced parameters
    pub loss_function: LossFunction,
    pub optimizer: OptimizerType,
    pub scheduler: Option<SchedulerConfig<T>>,
    pub early_stopping: Option<EarlyStoppingConfig<T>>,
    
    // Inference parameters
    pub prediction_intervals: Option<PredictionIntervals>,
    pub num_samples: usize,
}

impl<T: Float> LSTMConfig<T> {
    /// Create default LSTM configuration
    pub fn default_with_horizon(horizon: usize) -> Self;
    
    /// Builder pattern for fluent configuration
    pub fn builder() -> LSTMConfigBuilder<T>;
    
    /// Set architecture parameters
    pub fn with_architecture(
        mut self,
        hidden_size: usize,
        num_layers: usize,
        dropout: T
    ) -> Self;
    
    /// Set training parameters
    pub fn with_training(
        mut self,
        max_steps: usize,
        learning_rate: T
    ) -> Self;
    
    /// Enable bidirectional LSTM
    pub fn bidirectional(mut self) -> Self;
    
    /// Set exogenous features
    pub fn with_exogenous_features(
        mut self,
        static_features: Option<Vec<String>>,
        hist_exog_features: Option<Vec<String>>,
        futr_exog_features: Option<Vec<String>>
    ) -> Self;
}

impl<T: Float> ModelConfig<T> for LSTMConfig<T> {
    fn model_name() -> &'static str { "LSTM" }
    fn horizon(&self) -> usize { self.horizon }
    fn input_size(&self) -> usize { self.input_size }
    
    fn validate(&self) -> NeuroDivergentResult<()> {
        if self.horizon == 0 {
            return Err(NeuroDivergentError::ConfigError(
                "Horizon must be greater than 0".to_string()
            ));
        }
        if self.hidden_size == 0 {
            return Err(NeuroDivergentError::ConfigError(
                "Hidden size must be greater than 0".to_string()
            ));
        }
        if self.dropout < T::zero() || self.dropout >= T::one() {
            return Err(NeuroDivergentError::ConfigError(
                "Dropout must be in range [0, 1)".to_string()
            ));
        }
        Ok(())
    }
    
    fn to_generic(&self) -> GenericModelConfig<T> {
        // Convert to generic configuration
        todo!()
    }
    
    fn default_config() -> Self {
        Self::default_with_horizon(1)
    }
}
```

### NBEATS Configuration

```rust
/// Configuration for NBEATS models
#[derive(Debug, Clone)]
pub struct NBEATSConfig<T: Float> {
    // Required parameters
    pub horizon: usize,
    pub input_size: usize,
    
    // Architecture parameters
    pub stack_types: Vec<StackType>,
    pub n_blocks: Vec<usize>,
    pub mlp_units: Vec<Vec<usize>>,
    pub shared_weights: bool,
    pub activation: ActivationFunction,
    
    // Training parameters
    pub max_steps: usize,
    pub learning_rate: T,
    pub weight_decay: T,
    pub loss_function: LossFunction,
    
    // Data parameters
    pub scaler_type: ScalerType,
    pub static_features: Option<Vec<String>>,
    
    // Interpretability parameters
    pub include_trend: bool,
    pub include_seasonality: bool,
    pub seasonality_period: Option<usize>,
    
    // Advanced parameters
    pub optimizer: OptimizerType,
    pub scheduler: Option<SchedulerConfig<T>>,
    pub early_stopping: Option<EarlyStoppingConfig<T>>,
}

impl<T: Float> NBEATSConfig<T> {
    /// Create interpretable NBEATS configuration
    pub fn interpretable(horizon: usize, input_size: usize) -> Self {
        Self {
            horizon,
            input_size,
            stack_types: vec![StackType::Trend, StackType::Seasonality],
            n_blocks: vec![3, 3],
            mlp_units: vec![vec![512, 512], vec![512, 512]],
            shared_weights: true,
            activation: ActivationFunction::ReLU,
            max_steps: 1000,
            learning_rate: T::from(0.001).unwrap(),
            weight_decay: T::from(1e-3).unwrap(),
            loss_function: LossFunction::MAE,
            scaler_type: ScalerType::StandardScaler,
            static_features: None,
            include_trend: true,
            include_seasonality: true,
            seasonality_period: None,
            optimizer: OptimizerType::Adam,
            scheduler: None,
            early_stopping: None,
        }
    }
    
    /// Create generic NBEATS configuration
    pub fn generic(horizon: usize, input_size: usize) -> Self {
        Self {
            stack_types: vec![StackType::Generic; 4],
            n_blocks: vec![3; 4],
            include_trend: false,
            include_seasonality: false,
            ..Self::interpretable(horizon, input_size)
        }
    }
    
    /// Builder pattern for configuration
    pub fn builder() -> NBEATSConfigBuilder<T>;
}

impl<T: Float> ModelConfig<T> for NBEATSConfig<T> {
    fn model_name() -> &'static str { "NBEATS" }
    fn horizon(&self) -> usize { self.horizon }
    fn input_size(&self) -> usize { self.input_size }
    
    fn validate(&self) -> NeuroDivergentResult<()> {
        if self.stack_types.len() != self.n_blocks.len() {
            return Err(NeuroDivergentError::ConfigError(
                "stack_types and n_blocks must have the same length".to_string()
            ));
        }
        if self.stack_types.len() != self.mlp_units.len() {
            return Err(NeuroDivergentError::ConfigError(
                "stack_types and mlp_units must have the same length".to_string()
            ));
        }
        Ok(())
    }
    
    fn to_generic(&self) -> GenericModelConfig<T> {
        todo!()
    }
    
    fn default_config() -> Self {
        Self::interpretable(1, 1)
    }
}
```

## Supporting Types

### Frequency and Time Handling

```rust
/// Time series frequency enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Frequency {
    // High frequency
    Nanosecond,
    Microsecond,
    Millisecond,
    Second,
    Minute,
    
    // Common frequencies
    Hourly,
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Yearly,
    
    // Business frequencies
    BusinessDaily,
    BusinessMonthly,
    BusinessQuarterly,
    
    // Custom frequency with period
    Custom(String),
}

impl Frequency {
    /// Parse frequency from string (e.g., "D", "H", "M", "Q", "Y")
    pub fn from_str(s: &str) -> NeuroDivergentResult<Self>;
    
    /// Convert to pandas-compatible frequency string
    pub fn to_pandas_str(&self) -> &str;
    
    /// Get the duration between periods
    pub fn duration(&self) -> chrono::Duration;
    
    /// Check if frequency is business-related
    pub fn is_business(&self) -> bool;
}
```

### Scaling and Preprocessing Types

```rust
/// Available scaler types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalerType {
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    IdentityScaler,  // No scaling
}

/// Prediction intervals configuration
#[derive(Debug, Clone)]
pub struct PredictionIntervals {
    pub confidence_levels: Vec<f64>,
    pub method: IntervalMethod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntervalMethod {
    Quantile,
    ConformalPrediction,
    Bootstrap,
}
```

### Training Configuration

```rust
/// Training configuration for models
#[derive(Debug, Clone)]
pub struct TrainingConfig<T: Float> {
    pub max_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: T,
    pub weight_decay: T,
    pub gradient_clip_val: Option<T>,
    pub early_stopping: Option<EarlyStoppingConfig<T>>,
    pub validation_split: Option<T>,
    pub shuffle: bool,
    pub num_workers: usize,
    pub device: Device,
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig<T: Float> {
    pub monitor: String,
    pub patience: usize,
    pub min_delta: T,
    pub mode: EarlyStoppingMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarlyStoppingMode {
    Min,
    Max,
}

/// Device specification for computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    CPU,
    GPU(usize), // GPU index
}
```

### Loss Functions and Optimizers

```rust
/// Available loss functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossFunction {
    MSE,      // Mean Squared Error
    MAE,      // Mean Absolute Error
    MAPE,     // Mean Absolute Percentage Error
    SMAPE,    // Symmetric Mean Absolute Percentage Error
    Huber,    // Huber Loss
    Quantile, // Quantile Loss
    Custom(fn(&[f32], &[f32]) -> f32), // Custom loss function
}

/// Available optimizer types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerType {
    SGD,
    Adam,
    AdamW,
    RMSprop,
    Adagrad,
}

/// Learning rate scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig<T: Float> {
    pub scheduler_type: SchedulerType<T>,
    pub step_size: Option<usize>,
    pub gamma: Option<T>,
    pub milestones: Option<Vec<usize>>,
    pub patience: Option<usize>,
    pub factor: Option<T>,
}

#[derive(Debug, Clone)]
pub enum SchedulerType<T: Float> {
    StepLR,
    MultiStepLR,
    ExponentialLR,
    ReduceLROnPlateau,
    CosineAnnealingLR,
    Custom(Box<dyn Fn(usize, T) -> T + Send + Sync>),
}
```

## Cross-Validation Configuration

```rust
/// Cross-validation configuration
#[derive(Debug, Clone)]
pub struct CrossValidationConfig {
    pub n_windows: usize,
    pub h: usize,                    // Forecast horizon for each window
    pub step_size: Option<usize>,    // Step size between windows
    pub test_size: Option<usize>,    // Size of test set
    pub season_length: Option<usize>, // Seasonal length for time series split
    pub refit: bool,                 // Whether to refit models for each window
}

impl Default for CrossValidationConfig {
    fn default() -> Self {
        Self {
            n_windows: 3,
            h: 1,
            step_size: None,
            test_size: None,
            season_length: None,
            refit: true,
        }
    }
}
```

## Model Registry and Factory

```rust
/// Model factory for creating models by name
pub struct ModelFactory;

impl ModelFactory {
    /// Create model by name with configuration
    pub fn create_model<T: Float + Send + Sync>(
        model_name: &str,
        config: GenericModelConfig<T>
    ) -> NeuroDivergentResult<Box<dyn BaseModel<T>>>;
    
    /// List available model names
    pub fn available_models() -> Vec<&'static str>;
    
    /// Register custom model type
    pub fn register_model<M, C>(name: &str) 
    where 
        M: BaseModel<T> + 'static,
        C: ModelConfig<T> + 'static;
}

/// Model registry for tracking available models
pub struct ModelRegistry {
    models: HashMap<String, Box<dyn ModelConstructor>>,
}

trait ModelConstructor: Send + Sync {
    fn construct(&self, config: GenericModelConfig<f64>) -> NeuroDivergentResult<Box<dyn BaseModel<f64>>>;
}
```

This API design provides a comprehensive, type-safe interface that maintains compatibility with NeuralForecast's Python API while leveraging Rust's strengths in performance and memory safety. The extensive use of builder patterns, generic types, and trait objects ensures flexibility while maintaining compile-time safety guarantees.