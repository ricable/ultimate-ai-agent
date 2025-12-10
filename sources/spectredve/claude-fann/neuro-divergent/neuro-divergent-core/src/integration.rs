//! Integration layer with ruv-FANN neural networks.
//!
//! This module provides the bridge between neuro-divergent's time series forecasting
//! capabilities and ruv-FANN's neural network infrastructure, enabling seamless
//! integration of existing neural network functionality with forecasting-specific features.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2};
use num_traits::Float;
use serde::{Deserialize, Serialize};

use ruv_fann::{
    ActivationFunction, Network, NetworkBuilder, TrainingAlgorithm, TrainingData,
};

use crate::{
    data::{SeriesData, TimeSeriesDataset, TimeSeriesDataFrame},
    error::{ErrorBuilder, NetworkIntegrationError, NeuroDivergentError, NeuroDivergentResult},
    traits::{ForecastResult, TrainingStatistics},
};

/// Adapter for integrating ruv-FANN networks with forecasting models
///
/// This adapter wraps a ruv-FANN Network and provides time series-specific
/// preprocessing and postprocessing capabilities.
pub struct NetworkAdapter<T: Float + Send + Sync + 'static> {
    /// The underlying ruv-FANN network
    network: Network<T>,
    /// Input preprocessing pipeline
    input_preprocessor: Box<dyn InputPreprocessor<T>>,
    /// Output postprocessing pipeline
    output_postprocessor: Box<dyn OutputPostprocessor<T>>,
    /// Activation function mapper
    activation_mapper: ActivationMapper,
    /// Network configuration metadata
    config: NetworkAdapterConfig<T>,
    /// Training state
    training_state: Option<NetworkTrainingState<T>>,
}

/// Configuration for NetworkAdapter
#[derive(Debug, Clone)]
pub struct NetworkAdapterConfig<T: Float + Send + Sync + 'static> {
    /// Input dimension
    pub input_size: usize,
    /// Output dimension
    pub output_size: usize,
    /// Hidden layer sizes
    pub hidden_layers: Vec<usize>,
    /// Activation functions per layer
    pub activations: Vec<ActivationFunction>,
    /// Whether to use bias terms
    pub use_bias: bool,
    /// Dropout rates per layer (if supported)
    pub dropout_rates: Option<Vec<T>>,
    /// Network type identifier
    pub network_type: String,
}

/// Training state for NetworkAdapter
#[derive(Debug, Clone)]
pub struct NetworkTrainingState<T: Float + Send + Sync + 'static> {
    /// Current epoch
    pub current_epoch: usize,
    /// Training loss history
    pub loss_history: Vec<T>,
    /// Validation loss history
    pub validation_loss: Option<Vec<T>>,
    /// Learning rate history
    pub learning_rate_history: Vec<T>,
    /// Best validation loss
    pub best_validation_loss: Option<T>,
    /// Best epoch
    pub best_epoch: Option<usize>,
    /// Training start time
    pub training_start: chrono::DateTime<chrono::Utc>,
    /// Last update time
    pub last_update: chrono::DateTime<chrono::Utc>,
    /// Training completed flag
    pub training_completed: bool,
}

/// Input preprocessing trait for time series data
pub trait InputPreprocessor<T: Float + Send + Sync + 'static>: Send + Sync {
    /// Process input time series data for network consumption
    fn process(&self, input: &TimeSeriesInput<T>) -> NeuroDivergentResult<Vec<T>>;
    
    /// Get the expected output size after preprocessing
    fn output_size(&self) -> usize;
    
    /// Get preprocessing configuration
    fn config(&self) -> &PreprocessorConfig<T>;
    
    /// Reset internal state (if any)
    fn reset(&mut self);
}

/// Output postprocessing trait for network outputs
pub trait OutputPostprocessor<T: Float + Send + Sync + 'static>: Send + Sync {
    /// Process network output to generate forecasts
    fn process(&self, output: &[T], context: &PostprocessorContext<T>) -> NeuroDivergentResult<ForecastResult<T>>;
    
    /// Get postprocessing configuration
    fn config(&self) -> &PostprocessorConfig;
    
    /// Reset internal state (if any)
    fn reset(&mut self);
}

/// Input data structure for time series processing
#[derive(Debug, Clone)]
pub struct TimeSeriesInput<T: Float + Send + Sync + 'static> {
    /// Historical target values
    pub target_history: Vec<T>,
    /// Historical exogenous variables (if any)
    pub exogenous_history: Option<Array2<T>>,
    /// Future exogenous variables (if any)
    pub exogenous_future: Option<Array2<T>>,
    /// Static features
    pub static_features: Option<Vec<T>>,
    /// Timestamps
    pub timestamps: Vec<chrono::DateTime<chrono::Utc>>,
    /// Series identifier
    pub series_id: String,
    /// Input context metadata
    pub metadata: HashMap<String, String>,
}

/// Context for output postprocessing
#[derive(Debug, Clone)]
pub struct PostprocessorContext<T: Float + Send + Sync + 'static> {
    /// Original input data
    pub input: TimeSeriesInput<T>,
    /// Forecast horizon
    pub horizon: usize,
    /// Future timestamps
    pub future_timestamps: Vec<chrono::DateTime<chrono::Utc>>,
    /// Model name
    pub model_name: String,
    /// Additional context
    pub metadata: HashMap<String, String>,
}

/// Configuration for input preprocessors
#[derive(Debug, Clone)]
pub struct PreprocessorConfig<T: Float + Send + Sync + 'static> {
    /// Scaling method
    pub scaling_method: ScalingMethod,
    /// Window size for input
    pub window_size: usize,
    /// Whether to include trend features
    pub include_trend: bool,
    /// Whether to include seasonal features
    pub include_seasonal: bool,
    /// Lag features to include
    pub lag_features: Vec<usize>,
    /// Rolling window features
    pub rolling_features: Option<RollingFeatureConfig>,
    /// Normalization parameters
    pub normalization: Option<NormalizationParams<T>>,
}

/// Configuration for output postprocessors
#[derive(Debug, Clone)]
pub struct PostprocessorConfig {
    /// Forecast horizon
    pub horizon: usize,
    /// Whether to denormalize outputs
    pub denormalize: bool,
    /// Confidence interval settings
    pub confidence_intervals: Option<Vec<f64>>,
    /// Whether to apply trend adjustment
    pub trend_adjustment: bool,
    /// Seasonal adjustment settings
    pub seasonal_adjustment: Option<SeasonalAdjustmentConfig>,
}

/// Scaling methods for preprocessing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingMethod {
    /// No scaling
    None,
    /// Standard scaling (z-score)
    Standard,
    /// Min-max scaling
    MinMax,
    /// Robust scaling
    Robust,
    /// Custom scaling with parameters
    Custom(CustomScalingParams),
}

/// Custom scaling parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomScalingParams {
    /// Scaling factor
    pub scale: f64,
    /// Offset value
    pub offset: f64,
}

/// Rolling feature configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingFeatureConfig {
    /// Window sizes for rolling features
    pub window_sizes: Vec<usize>,
    /// Statistics to compute
    pub statistics: Vec<RollingStatistic>,
}

/// Rolling statistics to compute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RollingStatistic {
    /// Mean
    Mean,
    /// Standard deviation
    Std,
    /// Minimum
    Min,
    /// Maximum
    Max,
    /// Median
    Median,
}

/// Normalization parameters
#[derive(Debug, Clone)]
pub struct NormalizationParams<T: Float + Send + Sync + 'static> {
    /// Mean values per feature
    pub means: Vec<T>,
    /// Standard deviation values per feature
    pub stds: Vec<T>,
    /// Minimum values per feature (for MinMax scaling)
    pub mins: Option<Vec<T>>,
    /// Maximum values per feature (for MinMax scaling)
    pub maxs: Option<Vec<T>>,
}

/// Seasonal adjustment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalAdjustmentConfig {
    /// Seasonal period
    pub period: usize,
    /// Adjustment method
    pub method: SeasonalAdjustmentMethod,
}

/// Seasonal adjustment methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeasonalAdjustmentMethod {
    /// Additive seasonal adjustment
    Additive,
    /// Multiplicative seasonal adjustment
    Multiplicative,
    /// No adjustment
    None,
}

/// Activation function mapper for compatibility
pub struct ActivationMapper {
    /// Mapping from string names to ruv-FANN activation functions
    function_map: HashMap<String, ActivationFunction>,
}

/// Bridge between neuro-divergent training and ruv-FANN training algorithms
pub struct TrainingBridge<T: Float + Send + Sync + 'static> {
    /// ruv-FANN training algorithm
    algorithm: Box<dyn TrainingAlgorithm<T>>,
    /// Training configuration
    config: TrainingBridgeConfig<T>,
    /// Training state tracking
    state: Arc<Mutex<TrainingBridgeState<T>>>,
}

/// Configuration for training bridge
#[derive(Debug, Clone)]
pub struct TrainingBridgeConfig<T: Float + Send + Sync + 'static> {
    /// Maximum number of epochs
    pub max_epochs: usize,
    /// Learning rate
    pub learning_rate: T,
    /// Target error threshold
    pub target_error: T,
    /// Validation split ratio
    pub validation_split: Option<T>,
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig<T>>,
    /// Batch size for training
    pub batch_size: Option<usize>,
    /// Shuffle data between epochs
    pub shuffle: bool,
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig<T: Float + Send + Sync + 'static> {
    /// Patience (epochs to wait)
    pub patience: usize,
    /// Minimum improvement threshold
    pub min_delta: T,
    /// Metric to monitor ('loss' or 'validation_loss')
    pub monitor: String,
    /// Mode ('min' or 'max')
    pub mode: String,
}

/// Training bridge state
#[derive(Debug, Clone)]
pub struct TrainingBridgeState<T: Float + Send + Sync + 'static> {
    /// Current epoch
    pub epoch: usize,
    /// Training metrics
    pub metrics: TrainingStatistics<T>,
    /// Early stopping state
    pub early_stopping_state: Option<EarlyStoppingState>,
    /// Training active flag
    pub training_active: bool,
}

/// Early stopping state tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingState {
    /// Current patience counter
    pub patience_counter: usize,
    /// Best metric value seen
    pub best_metric: f64,
    /// Should stop training
    pub should_stop: bool,
}

/// Network output wrapper
#[derive(Debug, Clone)]
pub struct NetworkOutput<T: Float + Send + Sync + 'static> {
    /// Raw network outputs
    pub raw_outputs: Vec<T>,
    /// Processed outputs
    pub processed_outputs: Option<Vec<T>>,
    /// Confidence scores (if available)
    pub confidence_scores: Option<Vec<T>>,
    /// Output metadata
    pub metadata: HashMap<String, String>,
}

impl<T: Float + Send + Sync + 'static + 'static> NetworkAdapter<T> {
    /// Create adapter from ruv-FANN network
    pub fn from_network(network: Network<T>) -> Self {
        let config = NetworkAdapterConfig {
            input_size: network.num_inputs(),
            output_size: network.num_outputs(),
            hidden_layers: Vec::new(), // TODO: Extract from network
            activations: Vec::new(),    // TODO: Extract from network
            use_bias: true,
            dropout_rates: None,
            network_type: "feedforward".to_string(),
        };

        Self {
            network,
            input_preprocessor: Box::new(DefaultInputPreprocessor::<T>::new()),
            output_postprocessor: Box::new(DefaultOutputPostprocessor::<T>::new()),
            activation_mapper: ActivationMapper::new(),
            config,
            training_state: None,
        }
    }

    /// Configure input preprocessing
    pub fn with_input_processor(
        mut self,
        processor: Box<dyn InputPreprocessor<T>>,
    ) -> Self {
        self.input_preprocessor = processor;
        self
    }

    /// Configure output postprocessing
    pub fn with_output_processor(
        mut self,
        processor: Box<dyn OutputPostprocessor<T>>,
    ) -> Self {
        self.output_postprocessor = processor;
        self
    }

    /// Forward pass through the network
    pub fn forward(&mut self, input: &TimeSeriesInput<T>) -> NeuroDivergentResult<NetworkOutput<T>> {
        // Preprocess input
        let processed_input = self.input_preprocessor.process(input)
            .map_err(|e| NetworkIntegrationError::ValidationError {
                message: format!("Input preprocessing failed: {}", e),
                layer: None,
            })?;

        // Ensure input size matches network expectations
        if processed_input.len() != self.config.input_size {
            return Err(NetworkIntegrationError::ArchitectureMismatch {
                expected: format!("input size {}", self.config.input_size),
                found: format!("input size {}", processed_input.len()),
            }.into());
        }

        // Run through network
        let raw_outputs = self.network.run(&processed_input);

        Ok(NetworkOutput {
            raw_outputs,
            processed_outputs: None,
            confidence_scores: None,
            metadata: HashMap::new(),
        })
    }

    /// Get underlying network for direct access
    pub fn network(&self) -> &Network<T> {
        &self.network
    }

    /// Get mutable reference to underlying network
    pub fn network_mut(&mut self) -> &mut Network<T> {
        &mut self.network
    }

    /// Get adapter configuration
    pub fn config(&self) -> &NetworkAdapterConfig<T> {
        &self.config
    }

    /// Get training state
    pub fn training_state(&self) -> Option<&NetworkTrainingState<T>> {
        self.training_state.as_ref()
    }

    /// Reset adapter state
    pub fn reset(&mut self) {
        self.input_preprocessor.reset();
        self.output_postprocessor.reset();
        self.training_state = None;
    }

}


impl<T: Float + Send + Sync + 'static> TrainingBridge<T> {
    /// Create new training bridge with algorithm
    pub fn new(algorithm: Box<dyn TrainingAlgorithm<T>>) -> Self {
        let config = TrainingBridgeConfig {
            max_epochs: 1000,
            learning_rate: T::from(0.001).unwrap_or_else(|| T::one() / T::from(1000).unwrap()),
            target_error: T::from(0.001).unwrap_or_else(|| T::one() / T::from(1000).unwrap()),
            validation_split: None,
            early_stopping: None,
            batch_size: None,
            shuffle: true,
        };

        let state = Arc::new(Mutex::new(TrainingBridgeState {
            epoch: 0,
            metrics: TrainingStatistics::default(),
            early_stopping_state: None,
            training_active: false,
        }));

        Self {
            algorithm,
            config,
            state,
        }
    }

    /// Train network using ruv-FANN algorithms
    pub fn train_network(
        &mut self,
        network: &mut Network<T>,
        data: &TimeSeriesDataset<T>,
        config: &TrainingBridgeConfig<T>,
    ) -> NeuroDivergentResult<T> {
        // Convert time series data to ruv-FANN training format
        let training_data = self.prepare_training_data(data)?;

        // Update configuration
        self.config = config.clone();

        // Initialize training state
        {
            let mut state = self.state.lock().unwrap();
            state.training_active = true;
            state.epoch = 0;
            state.metrics = TrainingStatistics::default();
        }

        // Execute training loop
        let mut final_error = T::infinity();
        for epoch in 0..self.config.max_epochs {
            let epoch_error = self.algorithm.train_epoch(network, &training_data)
                .map_err(|e| NetworkIntegrationError::TrainingAlgorithmError {
                    message: format!("Training epoch {} failed: {}", epoch, e),
                    algorithm: Some("ruv-fann".to_string()),
                })?;
            
            final_error = epoch_error;
            
            // Update training state
            {
                let mut state = self.state.lock().unwrap();
                state.epoch = epoch + 1;
            }
            
            // Check for early stopping
            if epoch_error < T::from(0.001).unwrap() {
                break;
            }
        }

        // Update final state
        {
            let mut state = self.state.lock().unwrap();
            state.training_active = false;
        }

        Ok(final_error)
    }

    /// Prepare time series data for ruv-FANN training
    fn prepare_training_data(&self, dataset: &TimeSeriesDataset<T>) -> NeuroDivergentResult<TrainingData<T>> {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        for series in dataset.series_data.values() {
            // Create sliding windows for training
            let window_size = 10; // TODO: Make configurable
            let horizon = 1;      // TODO: Make configurable

            for i in 0..(series.target_values.len().saturating_sub(window_size + horizon)) {
                let input_window = &series.target_values[i..i + window_size];
                let output_window = &series.target_values[i + window_size..i + window_size + horizon];

                inputs.push(input_window.to_vec());
                outputs.push(output_window.to_vec());
            }
        }

        Ok(TrainingData { inputs, outputs })
    }

    /// Setup cascade correlation training
    pub fn setup_cascade_training(
        &mut self,
        _config: CascadeConfig<T>,
    ) -> NeuroDivergentResult<()> {
        // TODO: Implement cascade correlation configuration
        Ok(())
    }

    /// Get training state
    pub fn state(&self) -> Arc<Mutex<TrainingBridgeState<T>>> {
        Arc::clone(&self.state)
    }

    /// Check if training should continue
    pub fn should_continue_training(&self) -> bool {
        let state = self.state.lock().unwrap();
        state.training_active && state.epoch < self.config.max_epochs
    }
}

/// Cascade correlation training configuration
#[derive(Debug, Clone)]
pub struct CascadeConfig<T: Float + Send + Sync + 'static> {
    /// Maximum number of candidate neurons
    pub max_candidates: usize,
    /// Correlation threshold
    pub correlation_threshold: T,
    /// Training epochs per candidate
    pub epochs_per_candidate: usize,
    /// Minimum improvement threshold
    pub min_improvement: T,
}

impl ActivationMapper {
    /// Create new activation mapper
    pub fn new() -> Self {
        let mut function_map = HashMap::new();
        
        // Map common activation function names
        function_map.insert("linear".to_string(), ActivationFunction::Linear);
        function_map.insert("sigmoid".to_string(), ActivationFunction::Sigmoid);
        function_map.insert("sigmoid_symmetric".to_string(), ActivationFunction::SigmoidSymmetric);
        function_map.insert("gaussian".to_string(), ActivationFunction::Gaussian);
        function_map.insert("gaussian_symmetric".to_string(), ActivationFunction::GaussianSymmetric);
        function_map.insert("elliot".to_string(), ActivationFunction::Elliot);
        function_map.insert("elliot_symmetric".to_string(), ActivationFunction::ElliotSymmetric);
        function_map.insert("linear_piece".to_string(), ActivationFunction::LinearPiece);
        function_map.insert("linear_piece_symmetric".to_string(), ActivationFunction::LinearPieceSymmetric);
        function_map.insert("sin_symmetric".to_string(), ActivationFunction::SinSymmetric);
        function_map.insert("cos_symmetric".to_string(), ActivationFunction::CosSymmetric);

        Self { function_map }
    }

    /// Map string name to activation function
    pub fn map_function(&self, name: &str) -> Option<ActivationFunction> {
        self.function_map.get(name).copied()
    }

    /// Get all available function names
    pub fn available_functions(&self) -> Vec<&String> {
        self.function_map.keys().collect()
    }
}

/// Default input preprocessor implementation
pub struct DefaultInputPreprocessor<T: Float + Send + Sync + 'static> {
    config: PreprocessorConfig<T>,
}

impl<T: Float + Send + Sync + 'static> DefaultInputPreprocessor<T> {
    /// Create a new default input preprocessor
    pub fn new() -> Self {
        Self {
            config: PreprocessorConfig {
                scaling_method: ScalingMethod::Standard,
                window_size: 10,
                include_trend: false,
                include_seasonal: false,
                lag_features: Vec::new(),
                rolling_features: None,
                normalization: None,
            },
        }
    }
}

impl<T: Float + Send + Sync + 'static> InputPreprocessor<T> for DefaultInputPreprocessor<T> {
    fn process(&self, input: &TimeSeriesInput<T>) -> NeuroDivergentResult<Vec<T>> {
        // Simple default preprocessing: just return the target history
        let window_size = self.config.window_size.min(input.target_history.len());
        Ok(input.target_history[input.target_history.len() - window_size..].to_vec())
    }

    fn output_size(&self) -> usize {
        self.config.window_size
    }

    fn config(&self) -> &PreprocessorConfig<T> {
        &self.config
    }

    fn reset(&mut self) {
        // Nothing to reset in default implementation
    }
}

/// Default output postprocessor implementation
pub struct DefaultOutputPostprocessor<T: Float + Send + Sync + 'static> {
    config: PostprocessorConfig,
    _phantom: PhantomData<T>,
}

impl<T: Float + Send + Sync + 'static> DefaultOutputPostprocessor<T> {
    /// Create a new default output postprocessor
    pub fn new() -> Self {
        Self {
            config: PostprocessorConfig {
                horizon: 1,
                denormalize: false,
                confidence_intervals: None,
                trend_adjustment: false,
                seasonal_adjustment: None,
            },
            _phantom: PhantomData,
        }
    }
}

impl<T: Float + Send + Sync + 'static> OutputPostprocessor<T> for DefaultOutputPostprocessor<T> {
    fn process(&self, output: &[T], context: &PostprocessorContext<T>) -> NeuroDivergentResult<ForecastResult<T>> {
        Ok(ForecastResult {
            forecasts: output.to_vec(),
            timestamps: context.future_timestamps.clone(),
            series_id: context.input.series_id.clone(),
            model_name: context.model_name.clone(),
            generated_at: chrono::Utc::now(),
            metadata: None,
        })
    }

    fn config(&self) -> &PostprocessorConfig {
        &self.config
    }

    fn reset(&mut self) {
        // Nothing to reset in default implementation
    }
}

impl Default for ActivationMapper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruv_fann::NetworkBuilder;

    #[test]
    fn test_activation_mapper() {
        let mapper = ActivationMapper::new();
        
        assert_eq!(mapper.map_function("sigmoid"), Some(ActivationFunction::Sigmoid));
        assert_eq!(mapper.map_function("linear"), Some(ActivationFunction::Linear));
        assert_eq!(mapper.map_function("nonexistent"), None);
        
        let functions = mapper.available_functions();
        assert!(!functions.is_empty());
        assert!(functions.contains(&&"sigmoid".to_string()));
    }

    #[test]
    fn test_network_adapter_creation() {
        let network = NetworkBuilder::new()
            .input_layer(3)
            .hidden_layer(5)
            .output_layer(1)
            .build::<f64>()
            .unwrap();

        let adapter = NetworkAdapter::from_network(network);
        assert_eq!(adapter.config.input_size, 3);
        assert_eq!(adapter.config.output_size, 1);
    }

    #[test]
    fn test_training_bridge_creation() {
        let algorithm = Box::new(ruv_fann::training::BackpropagationAlgorithm::new());
        let bridge = TrainingBridge::<f64>::new(algorithm);
        
        assert_eq!(bridge.config.max_epochs, 1000);
        assert!(bridge.config.shuffle);
    }

    #[test]
    fn test_default_preprocessor() {
        let mut preprocessor = DefaultInputPreprocessor::<f64>::new();
        
        let input = TimeSeriesInput {
            target_history: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            exogenous_history: None,
            exogenous_future: None,
            static_features: None,
            timestamps: Vec::new(),
            series_id: "test".to_string(),
            metadata: HashMap::new(),
        };

        let processed = preprocessor.process(&input).unwrap();
        assert_eq!(processed.len(), 5); // Should return all values since window_size is 10
        assert_eq!(processed, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_configuration_serialization() {
        let config = NetworkAdapterConfig::<f64> {
            input_size: 10,
            output_size: 1,
            hidden_layers: vec![20, 15],
            activations: vec![ActivationFunction::Sigmoid, ActivationFunction::Linear],
            use_bias: true,
            dropout_rates: Some(vec![0.1, 0.2]),
            network_type: "feedforward".to_string(),
        };

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: NetworkAdapterConfig<f64> = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(config.input_size, deserialized.input_size);
        assert_eq!(config.output_size, deserialized.output_size);
        assert_eq!(config.hidden_layers, deserialized.hidden_layers);
    }
}