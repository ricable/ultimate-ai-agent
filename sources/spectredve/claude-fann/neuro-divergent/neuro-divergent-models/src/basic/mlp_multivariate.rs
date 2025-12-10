//! MLPMultivariate: Multi-Layer Perceptron for Multivariate Time Series Forecasting
//!
//! This module implements a feedforward neural network specifically designed for
//! multivariate time series forecasting. It can handle multiple related time series
//! either with shared weights or individual networks for each series.

use crate::core::{BaseModel, ModelConfig, ModelError, ModelResult, ForecastResult, TimeSeriesData, TrainingConfig};
use crate::utils::{Scaler, ScalerType, create_scaler};
use num_traits::Float;
use ruv_fann::{Network, NetworkBuilder, ActivationFunction, TrainingData, TrainingAlgorithm};
use ruv_fann::training::{IncrementalBackprop, BatchBackprop, Rprop, Quickprop};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Configuration for MLPMultivariate forecasting model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPMultivariateConfig<T: Float> {
    /// Number of time series to handle
    pub n_series: usize,
    /// Input window size (number of historical values to use per series)
    pub input_size: usize,
    /// Forecast horizon (number of future values to predict per series)
    pub horizon: usize,
    /// Hidden layer sizes
    pub hidden_layers: Vec<usize>,
    /// Activation function for hidden layers
    pub activation: ActivationFunction,
    /// Learning rate for training
    pub learning_rate: T,
    /// Maximum training epochs
    pub max_epochs: usize,
    /// Type of scaler to use for data preprocessing
    pub scaler_type: ScalerType,
    /// Training algorithm to use
    pub training_algorithm: TrainingAlgorithmType,
    /// Whether to use shared weights across all series or individual networks
    pub shared_weights: bool,
    /// Early stopping patience
    pub patience: Option<usize>,
    /// Validation split fraction
    pub validation_split: T,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    _phantom: PhantomData<T>,
}

/// Available training algorithms (reused from MLP)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingAlgorithmType {
    IncrementalBackprop,
    BatchBackprop,
    Rprop,
    Quickprop,
}

impl<T: Float> MLPMultivariateConfig<T> {
    /// Create a new MLPMultivariate configuration
    pub fn new(n_series: usize, input_size: usize, horizon: usize) -> Self {
        Self {
            n_series,
            input_size,
            horizon,
            hidden_layers: vec![64, 32],
            activation: ActivationFunction::ReLU,
            learning_rate: T::from(0.001).unwrap(),
            max_epochs: 100,
            scaler_type: ScalerType::Standard,
            training_algorithm: TrainingAlgorithmType::IncrementalBackprop,
            shared_weights: true,
            patience: Some(10),
            validation_split: T::from(0.2).unwrap(),
            seed: None,
            _phantom: PhantomData,
        }
    }
    
    /// Set hidden layer sizes
    pub fn with_hidden_layers(mut self, hidden_layers: Vec<usize>) -> Self {
        self.hidden_layers = hidden_layers;
        self
    }
    
    /// Set activation function
    pub fn with_activation(mut self, activation: ActivationFunction) -> Self {
        self.activation = activation;
        self
    }
    
    /// Set learning rate
    pub fn with_learning_rate(mut self, learning_rate: T) -> Self {
        self.learning_rate = learning_rate;
        self
    }
    
    /// Set maximum epochs
    pub fn with_max_epochs(mut self, max_epochs: usize) -> Self {
        self.max_epochs = max_epochs;
        self
    }
    
    /// Set scaler type
    pub fn with_scaler(mut self, scaler_type: ScalerType) -> Self {
        self.scaler_type = scaler_type;
        self
    }
    
    /// Set training algorithm
    pub fn with_training_algorithm(mut self, algorithm: TrainingAlgorithmType) -> Self {
        self.training_algorithm = algorithm;
        self
    }
    
    /// Set whether to use shared weights
    pub fn with_shared_weights(mut self, shared: bool) -> Self {
        self.shared_weights = shared;
        self
    }
    
    /// Set early stopping patience
    pub fn with_patience(mut self, patience: usize) -> Self {
        self.patience = Some(patience);
        self
    }
    
    /// Disable early stopping
    pub fn without_early_stopping(mut self) -> Self {
        self.patience = None;
        self
    }
    
    /// Set validation split
    pub fn with_validation_split(mut self, split: T) -> Self {
        self.validation_split = split;
        self
    }
    
    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    
    /// Create a builder for fluent configuration
    pub fn builder() -> MLPMultivariateBuilder<T> {
        MLPMultivariateBuilder::new()
    }
}

impl<T: Float> ModelConfig<T> for MLPMultivariateConfig<T> {
    fn horizon(&self) -> usize {
        self.horizon
    }
    
    fn input_size(&self) -> usize {
        self.input_size
    }
    
    fn validate(&self) -> ModelResult<()> {
        if self.n_series == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "n_series".to_string(),
                value: self.n_series.to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        
        if self.input_size == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "input_size".to_string(),
                value: self.input_size.to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        
        if self.horizon == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "horizon".to_string(),
                value: self.horizon.to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        
        if self.hidden_layers.is_empty() {
            return Err(ModelError::InvalidParameter {
                parameter: "hidden_layers".to_string(),
                value: "empty".to_string(),
                reason: "must have at least one hidden layer".to_string(),
            });
        }
        
        for (i, &size) in self.hidden_layers.iter().enumerate() {
            if size == 0 {
                return Err(ModelError::InvalidParameter {
                    parameter: format!("hidden_layers[{}]", i),
                    value: size.to_string(),
                    reason: "must be greater than 0".to_string(),
                });
            }
        }
        
        if self.learning_rate <= T::zero() {
            return Err(ModelError::InvalidParameter {
                parameter: "learning_rate".to_string(),
                value: "non-positive".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        
        if self.max_epochs == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "max_epochs".to_string(),
                value: self.max_epochs.to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        
        if self.validation_split < T::zero() || self.validation_split >= T::one() {
            return Err(ModelError::InvalidParameter {
                parameter: "validation_split".to_string(),
                value: "out of range".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }
        
        Ok(())
    }
    
    fn parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("n_series".to_string(), self.n_series.to_string());
        params.insert("input_size".to_string(), self.input_size.to_string());
        params.insert("horizon".to_string(), self.horizon.to_string());
        params.insert("hidden_layers".to_string(), format!("{:?}", self.hidden_layers));
        params.insert("activation".to_string(), format!("{:?}", self.activation));
        params.insert("learning_rate".to_string(), format!("{:?}", self.learning_rate));
        params.insert("max_epochs".to_string(), self.max_epochs.to_string());
        params.insert("scaler_type".to_string(), format!("{:?}", self.scaler_type));
        params.insert("training_algorithm".to_string(), format!("{:?}", self.training_algorithm));
        params.insert("shared_weights".to_string(), self.shared_weights.to_string());
        params.insert("patience".to_string(), format!("{:?}", self.patience));
        params.insert("validation_split".to_string(), format!("{:?}", self.validation_split));
        params.insert("seed".to_string(), format!("{:?}", self.seed));
        params
    }
    
    fn model_type(&self) -> &'static str {
        "MLPMultivariate"
    }
}

/// Builder for MLPMultivariate configuration
pub struct MLPMultivariateBuilder<T: Float> {
    config: MLPMultivariateConfig<T>,
}

impl<T: Float> MLPMultivariateBuilder<T> {
    pub fn new() -> Self {
        Self {
            config: MLPMultivariateConfig::new(1, 1, 1), // Will be overridden
        }
    }
    
    pub fn n_series(mut self, n_series: usize) -> Self {
        self.config.n_series = n_series;
        self
    }
    
    pub fn input_size(mut self, input_size: usize) -> Self {
        self.config.input_size = input_size;
        self
    }
    
    pub fn horizon(mut self, horizon: usize) -> Self {
        self.config.horizon = horizon;
        self
    }
    
    pub fn hidden_layers(mut self, layers: Vec<usize>) -> Self {
        self.config.hidden_layers = layers;
        self
    }
    
    pub fn activation(mut self, activation: ActivationFunction) -> Self {
        self.config.activation = activation;
        self
    }
    
    pub fn learning_rate(mut self, lr: T) -> Self {
        self.config.learning_rate = lr;
        self
    }
    
    pub fn max_epochs(mut self, epochs: usize) -> Self {
        self.config.max_epochs = epochs;
        self
    }
    
    pub fn scaler(mut self, scaler_type: ScalerType) -> Self {
        self.config.scaler_type = scaler_type;
        self
    }
    
    pub fn training_algorithm(mut self, algorithm: TrainingAlgorithmType) -> Self {
        self.config.training_algorithm = algorithm;
        self
    }
    
    pub fn shared_weights(mut self, shared: bool) -> Self {
        self.config.shared_weights = shared;
        self
    }
    
    pub fn patience(mut self, patience: usize) -> Self {
        self.config.patience = Some(patience);
        self
    }
    
    pub fn validation_split(mut self, split: T) -> Self {
        self.config.validation_split = split;
        self
    }
    
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }
    
    pub fn build(self) -> MLPMultivariateConfig<T> {
        self.config
    }
}

/// Data structure for multivariate time series
#[derive(Debug, Clone)]
pub struct MultivariateTimeSeriesData<T: Float> {
    /// Target values for each series (n_series x time_steps)
    pub targets: Vec<Vec<T>>,
    /// Optional exogenous features
    pub exogenous: Option<Vec<Vec<Vec<T>>>>, // n_series x time_steps x n_features
    /// Optional static features for each series
    pub static_features: Option<Vec<Vec<T>>>, // n_series x n_static_features
}

impl<T: Float> MultivariateTimeSeriesData<T> {
    /// Create new multivariate time series data
    pub fn new(targets: Vec<Vec<T>>) -> ModelResult<Self> {
        if targets.is_empty() {
            return Err(ModelError::DataError("Target data cannot be empty".to_string()));
        }
        
        // Check that all series have the same length
        let first_len = targets[0].len();
        for (i, series) in targets.iter().enumerate() {
            if series.len() != first_len {
                return Err(ModelError::DataError(format!(
                    "All time series must have the same length. Series {} has length {}, expected {}",
                    i, series.len(), first_len
                )));
            }
        }
        
        Ok(Self {
            targets,
            exogenous: None,
            static_features: None,
        })
    }
    
    /// Add exogenous features
    pub fn with_exogenous(mut self, exogenous: Vec<Vec<Vec<T>>>) -> ModelResult<Self> {
        if exogenous.len() != self.targets.len() {
            return Err(ModelError::DataError(format!(
                "Exogenous features length ({}) must match number of series ({})",
                exogenous.len(),
                self.targets.len()
            )));
        }
        
        for (i, series_exog) in exogenous.iter().enumerate() {
            if series_exog.len() != self.targets[i].len() {
                return Err(ModelError::DataError(format!(
                    "Exogenous features for series {} have length {}, expected {}",
                    i, series_exog.len(), self.targets[i].len()
                )));
            }
        }
        
        self.exogenous = Some(exogenous);
        Ok(self)
    }
    
    /// Add static features
    pub fn with_static_features(mut self, static_features: Vec<Vec<T>>) -> Self {
        self.static_features = Some(static_features);
        self
    }
    
    /// Get number of series
    pub fn n_series(&self) -> usize {
        self.targets.len()
    }
    
    /// Get length of time series
    pub fn len(&self) -> usize {
        if self.targets.is_empty() {
            0
        } else {
            self.targets[0].len()
        }
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty() || self.targets[0].is_empty()
    }
}

/// MLPMultivariate forecasting model
pub struct MLPMultivariate<T: Float> {
    config: MLPMultivariateConfig<T>,
    networks: Vec<Network<T>>,
    scalers: Vec<Box<dyn Scaler<T>>>,
    is_fitted: bool,
}

impl<T: Float + Send + Sync> MLPMultivariate<T> {
    /// Create network architecture for a single series
    fn create_single_network(&self) -> ModelResult<Network<T>> {
        let mut builder = NetworkBuilder::new();
        
        // Input layer (can include features from other series if shared weights)
        let input_size = if self.config.shared_weights {
            self.config.input_size * self.config.n_series
        } else {
            self.config.input_size
        };
        
        builder = builder.input_layer(input_size);
        
        // Hidden layers
        for &hidden_size in &self.config.hidden_layers {
            builder = builder.hidden_layer_with_activation(hidden_size, self.config.activation, T::one());
        }
        
        // Output layer
        builder = builder.output_layer_with_activation(self.config.horizon, ActivationFunction::Linear, T::one());
        
        builder.build().map_err(ModelError::from)
    }
    
    /// Prepare training data for multivariate series
    fn prepare_training_data(&mut self, data: &MultivariateTimeSeriesData<T>) -> ModelResult<Vec<TrainingData<T>>> {
        let mut training_datasets = Vec::new();
        
        for (series_idx, series_data) in data.targets.iter().enumerate() {
            // Scale the series data
            let scaled_data = self.scalers[series_idx].fit_transform(series_data)?;
            
            // Create windows
            if scaled_data.len() < self.config.input_size + self.config.horizon {
                return Err(ModelError::DataError(format!(
                    "Series {} is too short for input_size ({}) + horizon ({})",
                    series_idx,
                    self.config.input_size,
                    self.config.horizon
                )));
            }
            
            let mut training_data = TrainingData::new();
            
            // Create sliding windows
            for i in 0..=scaled_data.len() - self.config.input_size - self.config.horizon {
                let input_window = if self.config.shared_weights {
                    // Concatenate features from all series at the same time points
                    let mut combined_input = Vec::new();
                    for s_idx in 0..self.config.n_series {
                        let start = i;
                        let end = i + self.config.input_size;
                        if end <= data.targets[s_idx].len() {
                            let series_scaled = self.scalers[s_idx].transform(&data.targets[s_idx][start..end])?;
                            combined_input.extend(series_scaled);
                        }
                    }
                    combined_input
                } else {
                    // Just use the current series
                    scaled_data[i..i + self.config.input_size].to_vec()
                };
                
                let target_window = scaled_data[i + self.config.input_size..i + self.config.input_size + self.config.horizon].to_vec();
                
                training_data.add_sample(input_window, target_window)?;
            }
            
            training_datasets.push(training_data);
        }
        
        Ok(training_datasets)
    }
    
    /// Train networks with the configured algorithm
    fn train_networks(&mut self, training_datasets: &[TrainingData<T>]) -> ModelResult<()> {
        for (net_idx, training_data) in training_datasets.iter().enumerate() {
            if net_idx >= self.networks.len() {
                break;
            }
            
            // Create the appropriate training algorithm
            let mut trainer: Box<dyn TrainingAlgorithm<T>> = match self.config.training_algorithm {
                TrainingAlgorithmType::IncrementalBackprop => {
                    Box::new(IncrementalBackprop::new(self.config.learning_rate))
                },
                TrainingAlgorithmType::BatchBackprop => {
                    Box::new(BatchBackprop::new(self.config.learning_rate))
                },
                TrainingAlgorithmType::Rprop => {
                    Box::new(Rprop::new())
                },
                TrainingAlgorithmType::Quickprop => {
                    Box::new(Quickprop::new(self.config.learning_rate))
                },
            };
            
            let mut best_error = T::infinity();
            let mut patience_counter = 0;
            
            for epoch in 0..self.config.max_epochs {
                let epoch_error = trainer.train_epoch(&mut self.networks[net_idx], training_data)
                    .map_err(|e| ModelError::TrainingError(format!("Training failed for network {} at epoch {}: {}", net_idx, epoch, e)))?;
                
                // Early stopping logic
                if let Some(patience) = self.config.patience {
                    if epoch_error < best_error {
                        best_error = epoch_error;
                        patience_counter = 0;
                    } else {
                        patience_counter += 1;
                        if patience_counter >= patience {
                            break; // Early stopping
                        }
                    }
                }
                
                // Log progress occasionally
                if epoch % 10 == 0 {
                    eprintln!("Network {} - Epoch {}: Error = {:?}", net_idx, epoch, epoch_error);
                }
            }
        }
        
        Ok(())
    }
}

impl<T: Float + Send + Sync> BaseModel<T> for MLPMultivariate<T> {
    type Config = MLPMultivariateConfig<T>;
    
    fn new(config: Self::Config) -> ModelResult<Self> {
        config.validate()?;
        
        // Create networks
        let networks = if config.shared_weights {
            // Single shared network
            vec![Self::create_single_network(&Self { 
                config: config.clone(), 
                networks: Vec::new(), 
                scalers: Vec::new(), 
                is_fitted: false 
            })?]
        } else {
            // Individual network for each series
            let mut nets = Vec::new();
            for _ in 0..config.n_series {
                nets.push(Self::create_single_network(&Self { 
                    config: config.clone(), 
                    networks: Vec::new(), 
                    scalers: Vec::new(), 
                    is_fitted: false 
                })?);
            }
            nets
        };
        
        // Create scalers for each series
        let scalers: Vec<Box<dyn Scaler<T>>> = (0..config.n_series)
            .map(|_| create_scaler(config.scaler_type))
            .collect();
        
        Ok(Self {
            config,
            networks,
            scalers,
            is_fitted: false,
        })
    }
    
    fn fit(&mut self, data: &TimeSeriesData<T>) -> ModelResult<()> {
        // Convert univariate data to multivariate format for compatibility
        let mv_data = MultivariateTimeSeriesData::new(vec![data.target.clone()])?;
        self.fit_multivariate(&mv_data)
    }
    
    fn predict(&self, data: &TimeSeriesData<T>) -> ModelResult<ForecastResult<T>> {
        // Convert univariate data to multivariate format
        let mv_data = MultivariateTimeSeriesData::new(vec![data.target.clone()])?;
        let mv_result = self.predict_multivariate(&mv_data)?;
        
        // Return first series forecast
        if mv_result.is_empty() {
            return Err(ModelError::PredictionError("No forecasts generated".to_string()));
        }
        
        Ok(ForecastResult::new(mv_result[0].clone()))
    }
    
    fn config(&self) -> &Self::Config {
        &self.config
    }
    
    fn is_fitted(&self) -> bool {
        self.is_fitted
    }
    
    fn reset(&mut self) -> ModelResult<()> {
        for scaler in &mut self.scalers {
            scaler.reset();
        }
        self.is_fitted = false;
        Ok(())
    }
}

impl<T: Float + Send + Sync> MLPMultivariate<T> {
    /// Fit on multivariate data
    pub fn fit_multivariate(&mut self, data: &MultivariateTimeSeriesData<T>) -> ModelResult<()> {
        if data.n_series() != self.config.n_series {
            return Err(ModelError::DataError(format!(
                "Expected {} series, got {}",
                self.config.n_series,
                data.n_series()
            )));
        }
        
        if data.len() < self.config.input_size + self.config.horizon {
            return Err(ModelError::DataError(format!(
                "Data length ({}) is too short for input_size ({}) + horizon ({})",
                data.len(),
                self.config.input_size,
                self.config.horizon
            )));
        }
        
        // Prepare training data
        let training_datasets = self.prepare_training_data(data)?;
        
        // Train networks
        self.train_networks(&training_datasets)?;
        
        self.is_fitted = true;
        Ok(())
    }
    
    /// Predict on multivariate data
    pub fn predict_multivariate(&self, data: &MultivariateTimeSeriesData<T>) -> ModelResult<Vec<Vec<T>>> {
        if !self.is_fitted {
            return Err(ModelError::PredictionError("Model has not been fitted".to_string()));
        }
        
        if data.n_series() != self.config.n_series {
            return Err(ModelError::DataError(format!(
                "Expected {} series, got {}",
                self.config.n_series,
                data.n_series()
            )));
        }
        
        let mut forecasts = Vec::new();
        
        for (series_idx, series_data) in data.targets.iter().enumerate() {
            if series_data.len() < self.config.input_size {
                return Err(ModelError::PredictionError(format!(
                    "Series {} has insufficient data for prediction. Need at least {} values, got {}",
                    series_idx,
                    self.config.input_size,
                    series_data.len()
                )));
            }
            
            // Scale input data
            let scaled_data = self.scalers[series_idx].transform(series_data)?;
            
            // Prepare input for prediction
            let input_window = if self.config.shared_weights {
                // Use shared network (index 0) with concatenated features
                let mut combined_input = Vec::new();
                for s_idx in 0..self.config.n_series {
                    let series_len = data.targets[s_idx].len();
                    let start = series_len - self.config.input_size;
                    let series_scaled = self.scalers[s_idx].transform(&data.targets[s_idx][start..])?;
                    combined_input.extend(series_scaled);
                }
                combined_input
            } else {
                // Use individual network for this series
                let start = scaled_data.len() - self.config.input_size;
                scaled_data[start..].to_vec()
            };
            
            // Choose appropriate network
            let network_idx = if self.config.shared_weights { 0 } else { series_idx };
            
            // Make prediction
            let scaled_forecast = self.networks[network_idx].run(&input_window);
            
            // Inverse scale
            let forecast = self.scalers[series_idx].inverse_transform(&scaled_forecast)?;
            
            forecasts.push(forecast);
        }
        
        Ok(forecasts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TimeSeriesData;
    use approx::assert_relative_eq;

    #[test]
    fn test_mlp_multivariate_config_creation() {
        let config = MLPMultivariateConfig::new(3, 24, 12)
            .with_hidden_layers(vec![64, 32])
            .with_learning_rate(0.01)
            .with_shared_weights(false);
        
        assert_eq!(config.n_series, 3);
        assert_eq!(config.input_size, 24);
        assert_eq!(config.horizon, 12);
        assert_eq!(config.hidden_layers, vec![64, 32]);
        assert_relative_eq!(config.learning_rate, 0.01);
        assert!(!config.shared_weights);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_multivariate_data_creation() {
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let series2 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let series3 = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        
        let data = MultivariateTimeSeriesData::new(vec![series1, series2, series3]);
        assert!(data.is_ok());
        
        let data = data.unwrap();
        assert_eq!(data.n_series(), 3);
        assert_eq!(data.len(), 5);
        assert!(!data.is_empty());
    }

    #[test]
    fn test_multivariate_data_validation() {
        // Test mismatched series lengths
        let series1 = vec![1.0, 2.0, 3.0];
        let series2 = vec![10.0, 20.0]; // Different length
        
        let data = MultivariateTimeSeriesData::new(vec![series1, series2]);
        assert!(data.is_err());
    }

    #[test]
    fn test_mlp_multivariate_creation() {
        let config = MLPMultivariateConfig::new(2, 10, 5);
        let model = MLPMultivariate::<f64>::new(config);
        assert!(model.is_ok());
        
        let model = model.unwrap();
        assert!(!model.is_fitted());
        assert_eq!(model.config().n_series, 2);
        assert_eq!(model.config().input_size, 10);
        assert_eq!(model.config().horizon, 5);
    }

    #[test]
    fn test_mlp_multivariate_shared_vs_individual() {
        let config_shared = MLPMultivariateConfig::new(3, 10, 5).with_shared_weights(true);
        let model_shared = MLPMultivariate::<f64>::new(config_shared).unwrap();
        assert_eq!(model_shared.networks.len(), 1); // Single shared network
        
        let config_individual = MLPMultivariateConfig::new(3, 10, 5).with_shared_weights(false);
        let model_individual = MLPMultivariate::<f64>::new(config_individual).unwrap();
        assert_eq!(model_individual.networks.len(), 3); // Individual networks
    }

    #[test]
    fn test_mlp_multivariate_workflow() {
        // Create synthetic multivariate data
        let series1: Vec<f64> = (0..30).map(|i| (i as f64 * 0.1).sin()).collect();
        let series2: Vec<f64> = (0..30).map(|i| (i as f64 * 0.1).cos()).collect();
        
        let mv_data = MultivariateTimeSeriesData::new(vec![series1, series2]).unwrap();
        
        let config = MLPMultivariateConfig::new(2, 10, 3)
            .with_hidden_layers(vec![20, 10])
            .with_shared_weights(true)
            .with_max_epochs(5); // Small number for testing
        
        let mut model = MLPMultivariate::new(config).unwrap();
        
        // Test fitting
        let fit_result = model.fit_multivariate(&mv_data);
        assert!(fit_result.is_ok());
        assert!(model.is_fitted());
        
        // Test prediction
        let prediction = model.predict_multivariate(&mv_data);
        assert!(prediction.is_ok());
        
        let forecasts = prediction.unwrap();
        assert_eq!(forecasts.len(), 2); // Two series
        assert_eq!(forecasts[0].len(), 3); // Horizon = 3
        assert_eq!(forecasts[1].len(), 3);
        
        // Test reset
        assert!(model.reset().is_ok());
        assert!(!model.is_fitted());
    }

    #[test]
    fn test_mlp_multivariate_univariate_compatibility() {
        // Test that the model works with univariate data through the BaseModel interface
        let data = TimeSeriesData::new((0..25).map(|i| i as f64 * 0.1).collect());
        
        let config = MLPMultivariateConfig::new(1, 10, 5)
            .with_max_epochs(5);
        
        let mut model = MLPMultivariate::new(config).unwrap();
        
        // Test fitting
        let fit_result = model.fit(&data);
        assert!(fit_result.is_ok());
        assert!(model.is_fitted());
        
        // Test prediction
        let prediction = model.predict(&data);
        assert!(prediction.is_ok());
        
        let forecast = prediction.unwrap();
        assert_eq!(forecast.len(), 5);
    }

    #[test]
    fn test_mlp_multivariate_builder() {
        let config = MLPMultivariateBuilder::new()
            .n_series(4)
            .input_size(20)
            .horizon(10)
            .hidden_layers(vec![128, 64, 32])
            .activation(ActivationFunction::Tanh)
            .learning_rate(0.005)
            .shared_weights(false)
            .max_epochs(200)
            .build();
        
        assert_eq!(config.n_series, 4);
        assert_eq!(config.input_size, 20);
        assert_eq!(config.horizon, 10);
        assert_eq!(config.hidden_layers, vec![128, 64, 32]);
        assert_eq!(config.activation, ActivationFunction::Tanh);
        assert_relative_eq!(config.learning_rate, 0.005);
        assert!(!config.shared_weights);
        assert_eq!(config.max_epochs, 200);
    }
}