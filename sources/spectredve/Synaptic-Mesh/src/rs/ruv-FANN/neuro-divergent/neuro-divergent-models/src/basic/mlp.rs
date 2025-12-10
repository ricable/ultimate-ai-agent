//! Multi-Layer Perceptron (MLP) for Time Series Forecasting
//!
//! This module implements a feedforward neural network (MLP) adapted for time series forecasting.
//! The model uses sliding windows of historical data to predict future values.

use crate::core::{BaseModel, ModelConfig, ModelError, ModelResult, ForecastResult, TimeSeriesData, TrainingConfig};
use crate::utils::{Scaler, ScalerType, create_scaler, create_windows};
use num_traits::Float;
use ruv_fann::{Network, NetworkBuilder, ActivationFunction, TrainingData, TrainingAlgorithm};
use ruv_fann::training::{IncrementalBackprop, BatchBackprop, Rprop, Quickprop};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Configuration for MLP forecasting model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPConfig<T: Float> {
    /// Input window size (number of historical values to use)
    pub input_size: usize,
    /// Forecast horizon (number of future values to predict)
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
    /// Early stopping patience
    pub patience: Option<usize>,
    /// Validation split fraction
    pub validation_split: T,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    _phantom: PhantomData<T>,
}

/// Available training algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingAlgorithmType {
    IncrementalBackprop,
    BatchBackprop,
    Rprop,
    Quickprop,
}

impl<T: Float> MLPConfig<T> {
    /// Create a new MLP configuration with basic parameters
    pub fn new(input_size: usize, horizon: usize) -> Self {
        Self {
            input_size,
            horizon,
            hidden_layers: vec![64, 32],
            activation: ActivationFunction::ReLU,
            learning_rate: T::from(0.001).unwrap(),
            max_epochs: 100,
            scaler_type: ScalerType::Standard,
            training_algorithm: TrainingAlgorithmType::IncrementalBackprop,
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
    pub fn builder() -> MLPBuilder<T> {
        MLPBuilder::new()
    }
}

impl<T: Float> ModelConfig<T> for MLPConfig<T> {
    fn horizon(&self) -> usize {
        self.horizon
    }
    
    fn input_size(&self) -> usize {
        self.input_size
    }
    
    fn validate(&self) -> ModelResult<()> {
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
        params.insert("input_size".to_string(), self.input_size.to_string());
        params.insert("horizon".to_string(), self.horizon.to_string());
        params.insert("hidden_layers".to_string(), format!("{:?}", self.hidden_layers));
        params.insert("activation".to_string(), format!("{:?}", self.activation));
        params.insert("learning_rate".to_string(), format!("{:?}", self.learning_rate));
        params.insert("max_epochs".to_string(), self.max_epochs.to_string());
        params.insert("scaler_type".to_string(), format!("{:?}", self.scaler_type));
        params.insert("training_algorithm".to_string(), format!("{:?}", self.training_algorithm));
        params.insert("patience".to_string(), format!("{:?}", self.patience));
        params.insert("validation_split".to_string(), format!("{:?}", self.validation_split));
        params.insert("seed".to_string(), format!("{:?}", self.seed));
        params
    }
    
    fn model_type(&self) -> &'static str {
        "MLP"
    }
}

/// Builder for MLP configuration
pub struct MLPBuilder<T: Float> {
    config: MLPConfig<T>,
}

impl<T: Float> MLPBuilder<T> {
    pub fn new() -> Self {
        Self {
            config: MLPConfig::new(1, 1), // Will be overridden
        }
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
    
    pub fn build(self) -> MLPConfig<T> {
        self.config
    }
}

/// MLP forecasting model
pub struct MLP<T: Float> {
    config: MLPConfig<T>,
    network: Option<Network<T>>,
    scaler: Box<dyn Scaler<T>>,
    is_fitted: bool,
}

impl<T: Float + Send + Sync> MLP<T> {
    /// Create network architecture based on configuration
    fn create_network(&self) -> ModelResult<Network<T>> {
        let mut builder = NetworkBuilder::new();
        
        // Input layer
        builder = builder.input_layer(self.config.input_size);
        
        // Hidden layers
        for &hidden_size in &self.config.hidden_layers {
            builder = builder.hidden_layer_with_activation(hidden_size, self.config.activation, T::one());
        }
        
        // Output layer (linear activation for regression)
        builder = builder.output_layer_with_activation(self.config.horizon, ActivationFunction::Linear, T::one());
        
        builder.build().map_err(ModelError::from)
    }
    
    /// Prepare training data from time series
    fn prepare_training_data(&mut self, data: &TimeSeriesData<T>) -> ModelResult<TrainingData<T>> {
        // Scale the target data
        let scaled_target = self.scaler.fit_transform(&data.target)?;
        
        // Create sliding windows
        let (inputs, targets) = create_windows(&scaled_target, self.config.input_size, self.config.horizon, 1)?;
        
        if inputs.is_empty() {
            return Err(ModelError::DataError("No training windows could be created".to_string()));
        }
        
        // Convert to TrainingData format
        let mut training_data = TrainingData::new();
        for (input, target) in inputs.iter().zip(targets.iter()) {
            training_data.add_sample(input.clone(), target.clone())?;
        }
        
        Ok(training_data)
    }
    
    /// Train the network with the configured algorithm
    fn train_network(&mut self, training_data: &TrainingData<T>) -> ModelResult<()> {
        let network = self.network.as_mut().ok_or_else(|| 
            ModelError::TrainingError("Network not initialized".to_string()))?;
        
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
            let epoch_error = trainer.train_epoch(network, training_data)
                .map_err(|e| ModelError::TrainingError(format!("Training failed at epoch {}: {}", epoch, e)))?;
            
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
                eprintln!("Epoch {}: Error = {:?}", epoch, epoch_error);
            }
        }
        
        Ok(())
    }
}

impl<T: Float + Send + Sync> BaseModel<T> for MLP<T> {
    type Config = MLPConfig<T>;
    
    fn new(config: Self::Config) -> ModelResult<Self> {
        config.validate()?;
        
        let scaler = create_scaler(config.scaler_type);
        
        Ok(Self {
            config,
            network: None,
            scaler,
            is_fitted: false,
        })
    }
    
    fn fit(&mut self, data: &TimeSeriesData<T>) -> ModelResult<()> {
        self.validate_input(data)?;
        
        // Create and initialize network
        let network = self.create_network()?;
        self.network = Some(network);
        
        // Prepare training data
        let training_data = self.prepare_training_data(data)?;
        
        // Train the network
        self.train_network(&training_data)?;
        
        self.is_fitted = true;
        Ok(())
    }
    
    fn predict(&self, data: &TimeSeriesData<T>) -> ModelResult<ForecastResult<T>> {
        if !self.is_fitted {
            return Err(ModelError::PredictionError("Model has not been fitted".to_string()));
        }
        
        self.validate_input(data)?;
        
        let network = self.network.as_ref().ok_or_else(|| 
            ModelError::PredictionError("Network not initialized".to_string()))?;
        
        // Scale input data
        let scaled_data = self.scaler.transform(&data.target)?;
        
        // Use the last window of data for prediction
        if scaled_data.len() < self.config.input_size {
            return Err(ModelError::PredictionError(format!(
                "Insufficient data for prediction. Need at least {} values, got {}",
                self.config.input_size,
                scaled_data.len()
            )));
        }
        
        let input_window = &scaled_data[scaled_data.len() - self.config.input_size..];
        
        // Make prediction
        let scaled_forecast = network.run(input_window);
        
        // Inverse transform to original scale
        let forecast = self.scaler.inverse_transform(&scaled_forecast)?;
        
        // Create result with metadata
        let mut metadata = HashMap::new();
        metadata.insert("model_type".to_string(), "MLP".to_string());
        metadata.insert("input_size".to_string(), self.config.input_size.to_string());
        metadata.insert("horizon".to_string(), self.config.horizon.to_string());
        
        Ok(ForecastResult::new(forecast).with_metadata("model_type".to_string(), "MLP".to_string()))
    }
    
    fn config(&self) -> &Self::Config {
        &self.config
    }
    
    fn is_fitted(&self) -> bool {
        self.is_fitted
    }
    
    fn reset(&mut self) -> ModelResult<()> {
        self.network = None;
        self.scaler.reset();
        self.is_fitted = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TimeSeriesData;
    use approx::assert_relative_eq;

    #[test]
    fn test_mlp_config_creation() {
        let config = MLPConfig::new(10, 5)
            .with_hidden_layers(vec![64, 32])
            .with_learning_rate(0.01)
            .with_max_epochs(50);
        
        assert_eq!(config.input_size, 10);
        assert_eq!(config.horizon, 5);
        assert_eq!(config.hidden_layers, vec![64, 32]);
        assert_relative_eq!(config.learning_rate, 0.01);
        assert_eq!(config.max_epochs, 50);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_mlp_config_validation() {
        // Test invalid input_size
        let config = MLPConfig::new(0, 5);
        assert!(config.validate().is_err());
        
        // Test invalid horizon
        let config = MLPConfig::new(10, 0);
        assert!(config.validate().is_err());
        
        // Test empty hidden layers
        let config = MLPConfig::new(10, 5).with_hidden_layers(vec![]);
        assert!(config.validate().is_err());
        
        // Test zero hidden layer size
        let config = MLPConfig::new(10, 5).with_hidden_layers(vec![64, 0, 32]);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_mlp_builder() {
        let config = MLPBuilder::new()
            .input_size(20)
            .horizon(10)
            .hidden_layers(vec![128, 64])
            .activation(ActivationFunction::Tanh)
            .learning_rate(0.005)
            .max_epochs(200)
            .build();
        
        assert_eq!(config.input_size, 20);
        assert_eq!(config.horizon, 10);
        assert_eq!(config.hidden_layers, vec![128, 64]);
        assert_eq!(config.activation, ActivationFunction::Tanh);
        assert_relative_eq!(config.learning_rate, 0.005);
        assert_eq!(config.max_epochs, 200);
    }

    #[test]
    fn test_mlp_creation() {
        let config = MLPConfig::new(10, 5);
        let model = MLP::<f64>::new(config);
        assert!(model.is_ok());
        
        let model = model.unwrap();
        assert!(!model.is_fitted());
        assert_eq!(model.config().input_size, 10);
        assert_eq!(model.config().horizon, 5);
    }

    #[test]
    fn test_mlp_input_validation() {
        let config = MLPConfig::new(10, 5);
        let model = MLP::<f64>::new(config).unwrap();
        
        // Test with insufficient data
        let short_data = TimeSeriesData::new(vec![1.0, 2.0, 3.0]); // Only 3 values, need 10
        assert!(model.validate_input(&short_data).is_err());
        
        // Test with sufficient data
        let good_data = TimeSeriesData::new((0..20).map(|i| i as f64).collect());
        assert!(model.validate_input(&good_data).is_ok());
    }

    #[test]
    fn test_mlp_workflow() {
        // Create a simple sine wave dataset
        let data: Vec<f64> = (0..100)
            .map(|i| (i as f64 * 0.1).sin())
            .collect();
        let ts_data = TimeSeriesData::new(data);
        
        let config = MLPConfig::new(10, 5)
            .with_hidden_layers(vec![20, 10])
            .with_max_epochs(5); // Small number for testing
        
        let mut model = MLP::new(config).unwrap();
        
        // Test fitting
        let fit_result = model.fit(&ts_data);
        assert!(fit_result.is_ok());
        assert!(model.is_fitted());
        
        // Test prediction
        let prediction = model.predict(&ts_data);
        assert!(prediction.is_ok());
        
        let forecast = prediction.unwrap();
        assert_eq!(forecast.len(), 5); // Should predict 5 steps ahead
        
        // Test reset
        assert!(model.reset().is_ok());
        assert!(!model.is_fitted());
    }
}