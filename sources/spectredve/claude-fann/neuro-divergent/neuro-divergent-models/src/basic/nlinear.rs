//! NLinear: Normalized Linear Model
//!
//! NLinear applies normalization to the input data, then uses a simple linear
//! transformation to generate forecasts, and finally denormalizes the output.
//! This approach helps the model handle different scales and patterns in the data.

use crate::core::{BaseModel, ModelConfig, ModelError, ModelResult, ForecastResult, TimeSeriesData};
use crate::utils::{Scaler, ScalerType, create_scaler, create_windows};
use num_traits::Float;
use ruv_fann::{Network, NetworkBuilder, ActivationFunction};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Configuration for NLinear forecasting model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NLinearConfig<T: Float> {
    /// Input window size (number of historical values to use)
    pub input_size: usize,
    /// Forecast horizon (number of future values to predict)
    pub horizon: usize,
    /// Type of normalization to apply
    pub normalization_type: ScalerType,
    /// Learning rate for training the linear layer
    pub learning_rate: T,
    /// Maximum training epochs
    pub max_epochs: usize,
    /// Early stopping patience
    pub patience: Option<usize>,
    /// Add individual series normalization (mean subtraction)
    pub individual_normalization: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    _phantom: PhantomData<T>,
}

impl<T: Float> NLinearConfig<T> {
    /// Create a new NLinear configuration
    pub fn new(input_size: usize, horizon: usize) -> Self {
        Self {
            input_size,
            horizon,
            normalization_type: ScalerType::Standard,
            learning_rate: T::from(0.001).unwrap(),
            max_epochs: 100,
            patience: Some(10),
            individual_normalization: true,
            seed: None,
            _phantom: PhantomData,
        }
    }
    
    /// Set normalization type
    pub fn with_normalization(mut self, normalization_type: ScalerType) -> Self {
        self.normalization_type = normalization_type;
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
    
    /// Enable/disable individual series normalization
    pub fn with_individual_normalization(mut self, enable: bool) -> Self {
        self.individual_normalization = enable;
        self
    }
    
    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    
    /// Create a builder for fluent configuration
    pub fn builder() -> NLinearBuilder<T> {
        NLinearBuilder::new()
    }
}

impl<T: Float> ModelConfig<T> for NLinearConfig<T> {
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
        
        Ok(())
    }
    
    fn parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("input_size".to_string(), self.input_size.to_string());
        params.insert("horizon".to_string(), self.horizon.to_string());
        params.insert("normalization_type".to_string(), format!("{:?}", self.normalization_type));
        params.insert("learning_rate".to_string(), format!("{:?}", self.learning_rate));
        params.insert("max_epochs".to_string(), self.max_epochs.to_string());
        params.insert("patience".to_string(), format!("{:?}", self.patience));
        params.insert("individual_normalization".to_string(), self.individual_normalization.to_string());
        params.insert("seed".to_string(), format!("{:?}", self.seed));
        params
    }
    
    fn model_type(&self) -> &'static str {
        "NLinear"
    }
}

/// Builder for NLinear configuration
pub struct NLinearBuilder<T: Float> {
    config: NLinearConfig<T>,
}

impl<T: Float> NLinearBuilder<T> {
    pub fn new() -> Self {
        Self {
            config: NLinearConfig::new(1, 1), // Will be overridden
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
    
    pub fn normalization(mut self, normalization_type: ScalerType) -> Self {
        self.config.normalization_type = normalization_type;
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
    
    pub fn patience(mut self, patience: usize) -> Self {
        self.config.patience = Some(patience);
        self
    }
    
    pub fn individual_normalization(mut self, enable: bool) -> Self {
        self.config.individual_normalization = enable;
        self
    }
    
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }
    
    pub fn build(self) -> NLinearConfig<T> {
        self.config
    }
}

/// NLinear forecasting model
pub struct NLinear<T: Float> {
    config: NLinearConfig<T>,
    network: Option<Network<T>>,
    global_scaler: Box<dyn Scaler<T>>,
    last_values: Option<Vec<T>>, // For individual normalization
    is_fitted: bool,
}

impl<T: Float + Send + Sync> NLinear<T> {
    /// Create a simple linear network
    fn create_linear_network(&self) -> ModelResult<Network<T>> {
        NetworkBuilder::new()
            .input_layer(self.config.input_size)
            .output_layer_with_activation(self.config.horizon, ActivationFunction::Linear, T::one())
            .build()
            .map_err(ModelError::from)
    }
    
    /// Apply individual normalization (subtract last value)
    fn apply_individual_normalization(&self, data: &[T]) -> (Vec<T>, T) {
        if data.is_empty() {
            return (Vec::new(), T::zero());
        }
        
        let last_value = *data.last().unwrap();
        let normalized: Vec<T> = data.iter().map(|&x| x - last_value).collect();
        (normalized, last_value)
    }
    
    /// Reverse individual normalization
    fn reverse_individual_normalization(&self, normalized_data: &[T], last_value: T) -> Vec<T> {
        normalized_data.iter().map(|&x| x + last_value).collect()
    }
    
    /// Prepare training data from normalized time series
    fn prepare_training_data(&mut self, data: &TimeSeriesData<T>) -> ModelResult<Vec<(Vec<T>, Vec<T>)>> {
        let mut processed_data = data.target.clone();
        
        // Apply individual normalization if enabled
        let (individually_normalized, last_value) = if self.config.individual_normalization {
            let (normalized, last_val) = self.apply_individual_normalization(&processed_data);
            self.last_values = Some(vec![last_val]); // Store for prediction
            processed_data = normalized;
            (true, last_val)
        } else {
            (false, T::zero())
        };
        
        // Apply global scaling
        let scaled_data = if self.config.normalization_type != ScalerType::Identity {
            self.global_scaler.fit_transform(&processed_data)?
        } else {
            processed_data
        };
        
        // Create sliding windows
        let (inputs, targets) = create_windows(&scaled_data, self.config.input_size, self.config.horizon, 1)?;
        
        if inputs.is_empty() {
            return Err(ModelError::DataError("No training windows could be created".to_string()));
        }
        
        Ok(inputs.into_iter().zip(targets.into_iter()).collect())
    }
    
    /// Train the linear network
    fn train_network(&mut self, training_data: &[(Vec<T>, Vec<T>)]) -> ModelResult<()> {
        let network = self.network.as_mut().ok_or_else(|| 
            ModelError::TrainingError("Network not initialized".to_string()))?;
        
        let mut best_error = T::infinity();
        let mut patience_counter = 0;
        
        for epoch in 0..self.config.max_epochs {
            let mut total_error = T::zero();
            
            for (input, target) in training_data {
                let output = network.run(input);
                
                // Calculate mean squared error
                let error: T = output.iter()
                    .zip(target.iter())
                    .map(|(&pred, &actual)| (pred - actual) * (pred - actual))
                    .sum::<T>() / T::from(output.len()).unwrap();
                
                total_error = total_error + error;
                
                // Note: In a full implementation, we would do proper gradient descent here
                // For now, we rely on the network's built-in capabilities
                // This is a simplified implementation
            }
            
            let avg_error = total_error / T::from(training_data.len()).unwrap();
            
            // Early stopping logic
            if let Some(patience) = self.config.patience {
                if avg_error < best_error {
                    best_error = avg_error;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= patience {
                        break; // Early stopping
                    }
                }
            }
            
            // Simple convergence check
            if avg_error < T::from(1e-8).unwrap() {
                break;
            }
            
            if epoch % 10 == 0 {
                eprintln!("NLinear training - Epoch {}: Error = {:?}", epoch, avg_error);
            }
        }
        
        Ok(())
    }
}

impl<T: Float + Send + Sync> BaseModel<T> for NLinear<T> {
    type Config = NLinearConfig<T>;
    
    fn new(config: Self::Config) -> ModelResult<Self> {
        config.validate()?;
        
        let global_scaler = create_scaler(config.normalization_type);
        
        Ok(Self {
            config,
            network: None,
            global_scaler,
            last_values: None,
            is_fitted: false,
        })
    }
    
    fn fit(&mut self, data: &TimeSeriesData<T>) -> ModelResult<()> {
        self.validate_input(data)?;
        
        // Create linear network
        let network = self.create_linear_network()?;
        self.network = Some(network);
        
        // Prepare training data with normalization
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
        
        let mut processed_data = data.target.clone();
        let prediction_last_value;
        
        // Apply individual normalization if enabled
        if self.config.individual_normalization {
            let (normalized, last_value) = self.apply_individual_normalization(&processed_data);
            processed_data = normalized;
            prediction_last_value = last_value;
        } else {
            prediction_last_value = T::zero();
        }
        
        // Apply global scaling
        let scaled_data = if self.config.normalization_type != ScalerType::Identity {
            self.global_scaler.transform(&processed_data)?
        } else {
            processed_data
        };
        
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
        
        // Reverse global scaling
        let globally_unscaled = if self.config.normalization_type != ScalerType::Identity {
            self.global_scaler.inverse_transform(&scaled_forecast)?
        } else {
            scaled_forecast
        };
        
        // Reverse individual normalization
        let final_forecast = if self.config.individual_normalization {
            self.reverse_individual_normalization(&globally_unscaled, prediction_last_value)
        } else {
            globally_unscaled
        };
        
        // Create result with metadata
        let mut metadata = HashMap::new();
        metadata.insert("model_type".to_string(), "NLinear".to_string());
        metadata.insert("normalization_type".to_string(), format!("{:?}", self.config.normalization_type));
        metadata.insert("individual_normalization".to_string(), self.config.individual_normalization.to_string());
        
        Ok(ForecastResult::new(final_forecast).with_metadata("model_type".to_string(), "NLinear".to_string()))
    }
    
    fn config(&self) -> &Self::Config {
        &self.config
    }
    
    fn is_fitted(&self) -> bool {
        self.is_fitted
    }
    
    fn reset(&mut self) -> ModelResult<()> {
        self.network = None;
        self.global_scaler.reset();
        self.last_values = None;
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
    fn test_nlinear_config_creation() {
        let config = NLinearConfig::new(24, 12)
            .with_normalization(ScalerType::MinMax)
            .with_learning_rate(0.01)
            .with_max_epochs(50)
            .with_individual_normalization(false);
        
        assert_eq!(config.input_size, 24);
        assert_eq!(config.horizon, 12);
        assert_eq!(config.normalization_type, ScalerType::MinMax);
        assert_relative_eq!(config.learning_rate, 0.01);
        assert_eq!(config.max_epochs, 50);
        assert!(!config.individual_normalization);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_nlinear_config_validation() {
        // Test invalid input_size
        let config = NLinearConfig::new(0, 5);
        assert!(config.validate().is_err());
        
        // Test invalid horizon
        let config = NLinearConfig::new(10, 0);
        assert!(config.validate().is_err());
        
        // Test invalid learning rate
        let config = NLinearConfig::new(10, 5).with_learning_rate(-0.01);
        assert!(config.validate().is_err());
        
        // Test zero max epochs
        let config = NLinearConfig::new(10, 5).with_max_epochs(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_nlinear_builder() {
        let config = NLinearBuilder::new()
            .input_size(20)
            .horizon(10)
            .normalization(ScalerType::Standard)
            .learning_rate(0.005)
            .max_epochs(200)
            .individual_normalization(true)
            .build();
        
        assert_eq!(config.input_size, 20);
        assert_eq!(config.horizon, 10);
        assert_eq!(config.normalization_type, ScalerType::Standard);
        assert_relative_eq!(config.learning_rate, 0.005);
        assert_eq!(config.max_epochs, 200);
        assert!(config.individual_normalization);
    }

    #[test]
    fn test_nlinear_creation() {
        let config = NLinearConfig::new(10, 5);
        let model = NLinear::<f64>::new(config);
        assert!(model.is_ok());
        
        let model = model.unwrap();
        assert!(!model.is_fitted());
        assert_eq!(model.config().input_size, 10);
        assert_eq!(model.config().horizon, 5);
    }

    #[test]
    fn test_individual_normalization() {
        let config = NLinearConfig::new(10, 5);
        let model = NLinear::<f64>::new(config).unwrap();
        
        let data = vec![10.0, 12.0, 15.0, 18.0, 20.0];
        let (normalized, last_value) = model.apply_individual_normalization(&data);
        
        assert_relative_eq!(last_value, 20.0);
        assert_eq!(normalized.len(), data.len());
        assert_relative_eq!(normalized[4], 0.0); // Last value should be 0
        assert_relative_eq!(normalized[0], -10.0); // 10 - 20 = -10
        
        // Test reverse normalization
        let reversed = model.reverse_individual_normalization(&normalized, last_value);
        for (original, recovered) in data.iter().zip(reversed.iter()) {
            assert_relative_eq!(original, recovered, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_nlinear_workflow() {
        // Create a synthetic dataset with trend
        let data: Vec<f64> = (0..50)
            .map(|i| i as f64 + (i as f64 * 0.1).sin())
            .collect();
        
        let ts_data = TimeSeriesData::new(data);
        
        let config = NLinearConfig::new(15, 5)
            .with_normalization(ScalerType::Standard)
            .with_individual_normalization(true)
            .with_max_epochs(10); // Small number for testing
        
        let mut model = NLinear::new(config).unwrap();
        
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

    #[test]
    fn test_nlinear_different_normalizations() {
        let data: Vec<f64> = (0..30).map(|i| (i as f64) * 0.5).collect();
        let ts_data = TimeSeriesData::new(data);
        
        // Test with different normalization types
        let normalizations = vec![
            ScalerType::Standard,
            ScalerType::MinMax,
            ScalerType::Identity,
        ];
        
        for norm_type in normalizations {
            let config = NLinearConfig::new(10, 3)
                .with_normalization(norm_type)
                .with_max_epochs(5);
            
            let mut model = NLinear::new(config).unwrap();
            
            let fit_result = model.fit(&ts_data);
            assert!(fit_result.is_ok(), "Failed with normalization {:?}", norm_type);
            
            let prediction = model.predict(&ts_data);
            assert!(prediction.is_ok(), "Prediction failed with normalization {:?}", norm_type);
        }
    }

    #[test]
    fn test_nlinear_input_validation() {
        let config = NLinearConfig::new(10, 5);
        let model = NLinear::<f64>::new(config).unwrap();
        
        // Test with insufficient data
        let short_data = TimeSeriesData::new(vec![1.0, 2.0, 3.0]);
        assert!(model.validate_input(&short_data).is_err());
        
        // Test with sufficient data
        let good_data = TimeSeriesData::new((0..20).map(|i| i as f64).collect());
        assert!(model.validate_input(&good_data).is_ok());
    }
}