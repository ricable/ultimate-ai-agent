//! Core traits and types for neural forecasting models
//!
//! This module defines the fundamental interfaces that all forecasting models must implement,
//! along with common data structures and error types.

use num_traits::Float;
use ruv_fann::{Network, TrainingData};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Result type for model operations
pub type ModelResult<T> = Result<T, ModelError>;

/// Error types for model operations
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Training error: {0}")]
    TrainingError(String),
    
    #[error("Prediction error: {0}")]
    PredictionError(String),
    
    #[error("Data validation error: {0}")]
    DataError(String),
    
    #[error("Network error: {0}")]
    NetworkError(#[from] ruv_fann::NetworkError),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Invalid parameter: {parameter} = {value}, reason: {reason}")]
    InvalidParameter {
        parameter: String,
        value: String,
        reason: String,
    },
}

/// Time series data structure for model input/output
#[derive(Debug, Clone)]
pub struct TimeSeriesData<T: Float> {
    /// Historical values of the target variable
    pub target: Vec<T>,
    /// Optional exogenous features (same length as target)
    pub exogenous: Option<Vec<Vec<T>>>,
    /// Optional static features (constant across time)
    pub static_features: Option<Vec<T>>,
    /// Time index (optional, for validation)
    pub time_index: Option<Vec<usize>>,
}

impl<T: Float> TimeSeriesData<T> {
    /// Create new time series data with just target values
    pub fn new(target: Vec<T>) -> Self {
        Self {
            target,
            exogenous: None,
            static_features: None,
            time_index: None,
        }
    }
    
    /// Add exogenous features
    pub fn with_exogenous(mut self, exogenous: Vec<Vec<T>>) -> ModelResult<Self> {
        if !exogenous.is_empty() && exogenous.len() != self.target.len() {
            return Err(ModelError::DataError(format!(
                "Exogenous features length ({}) must match target length ({})",
                exogenous.len(),
                self.target.len()
            )));
        }
        self.exogenous = Some(exogenous);
        Ok(self)
    }
    
    /// Add static features
    pub fn with_static_features(mut self, static_features: Vec<T>) -> Self {
        self.static_features = Some(static_features);
        self
    }
    
    /// Get length of the time series
    pub fn len(&self) -> usize {
        self.target.len()
    }
    
    /// Check if the time series is empty
    pub fn is_empty(&self) -> bool {
        self.target.is_empty()
    }
    
    /// Validate the data structure
    pub fn validate(&self) -> ModelResult<()> {
        if self.target.is_empty() {
            return Err(ModelError::DataError("Target data cannot be empty".to_string()));
        }
        
        if let Some(ref exog) = self.exogenous {
            if !exog.is_empty() && exog.len() != self.target.len() {
                return Err(ModelError::DataError(format!(
                    "Exogenous features length ({}) must match target length ({})",
                    exog.len(),
                    self.target.len()
                )));
            }
            
            // Check that all exogenous feature vectors have the same length
            if let Some(first) = exog.first() {
                let expected_len = first.len();
                for (i, features) in exog.iter().enumerate() {
                    if features.len() != expected_len {
                        return Err(ModelError::DataError(format!(
                            "All exogenous feature vectors must have the same length. \
                             Vector at index {} has length {}, expected {}",
                            i, features.len(), expected_len
                        )));
                    }
                }
            }
        }
        
        Ok(())
    }
}

/// Forecast results from model prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResult<T: Float> {
    /// Point forecasts
    pub forecasts: Vec<T>,
    /// Optional prediction intervals (confidence bounds)
    pub prediction_intervals: Option<HashMap<String, (Vec<T>, Vec<T>)>>,
    /// Model metadata
    pub metadata: HashMap<String, String>,
}

impl<T: Float> ForecastResult<T> {
    /// Create new forecast result with just point forecasts
    pub fn new(forecasts: Vec<T>) -> Self {
        Self {
            forecasts,
            prediction_intervals: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Add prediction intervals
    pub fn with_intervals(mut self, intervals: HashMap<String, (Vec<T>, Vec<T>)>) -> Self {
        self.prediction_intervals = Some(intervals);
        self
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    /// Get forecast length
    pub fn len(&self) -> usize {
        self.forecasts.len()
    }
    
    /// Check if forecast is empty
    pub fn is_empty(&self) -> bool {
        self.forecasts.is_empty()
    }
}

/// Configuration trait that all model configurations must implement
pub trait ModelConfig<T: Float>: Clone + Send + Sync {
    /// Get the forecast horizon (number of steps to predict)
    fn horizon(&self) -> usize;
    
    /// Get the input size (lookback window)
    fn input_size(&self) -> usize;
    
    /// Validate the configuration parameters
    fn validate(&self) -> ModelResult<()>;
    
    /// Get model-specific parameters as a hash map
    fn parameters(&self) -> HashMap<String, String>;
    
    /// Get the model type name
    fn model_type(&self) -> &'static str;
}

/// Base trait that all forecasting models must implement
pub trait BaseModel<T: Float + Send + Sync>: Send + Sync {
    /// Associated configuration type
    type Config: ModelConfig<T>;
    
    /// Create a new model instance with the given configuration
    fn new(config: Self::Config) -> ModelResult<Self> where Self: Sized;
    
    /// Fit the model to training data
    fn fit(&mut self, data: &TimeSeriesData<T>) -> ModelResult<()>;
    
    /// Generate forecasts for the given data
    fn predict(&self, data: &TimeSeriesData<T>) -> ModelResult<ForecastResult<T>>;
    
    /// Get the model configuration
    fn config(&self) -> &Self::Config;
    
    /// Check if the model has been fitted
    fn is_fitted(&self) -> bool;
    
    /// Reset the model to unfitted state
    fn reset(&mut self) -> ModelResult<()>;
    
    /// Validate input data compatibility with model
    fn validate_input(&self, data: &TimeSeriesData<T>) -> ModelResult<()> {
        data.validate()?;
        
        let config = self.config();
        
        if data.len() < config.input_size() {
            return Err(ModelError::DataError(format!(
                "Input data length ({}) is less than required input size ({})",
                data.len(),
                config.input_size()
            )));
        }
        
        Ok(())
    }
    
    /// Get model summary information
    fn summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();
        let config = self.config();
        
        summary.insert("model_type".to_string(), config.model_type().to_string());
        summary.insert("horizon".to_string(), config.horizon().to_string());
        summary.insert("input_size".to_string(), config.input_size().to_string());
        summary.insert("fitted".to_string(), self.is_fitted().to_string());
        
        // Add model-specific parameters
        for (key, value) in config.parameters() {
            summary.insert(key, value);
        }
        
        summary
    }
}

/// Training configuration for models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig<T: Float> {
    /// Maximum number of training epochs
    pub max_epochs: usize,
    /// Learning rate
    pub learning_rate: T,
    /// Early stopping patience
    pub patience: Option<usize>,
    /// Validation split (fraction of data to use for validation)
    pub validation_split: Option<T>,
    /// Batch size for training
    pub batch_size: Option<usize>,
    /// Whether to shuffle training data
    pub shuffle: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl<T: Float> Default for TrainingConfig<T> {
    fn default() -> Self {
        Self {
            max_epochs: 100,
            learning_rate: T::from(0.001).unwrap(),
            patience: Some(10),
            validation_split: Some(T::from(0.2).unwrap()),
            batch_size: None,
            shuffle: true,
            seed: None,
        }
    }
}

impl<T: Float> TrainingConfig<T> {
    /// Create new training configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set maximum epochs
    pub fn with_max_epochs(mut self, max_epochs: usize) -> Self {
        self.max_epochs = max_epochs;
        self
    }
    
    /// Set learning rate
    pub fn with_learning_rate(mut self, learning_rate: T) -> Self {
        self.learning_rate = learning_rate;
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
        self.validation_split = Some(split);
        self
    }
    
    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }
    
    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    
    /// Validate the training configuration
    pub fn validate(&self) -> ModelResult<()> {
        if self.max_epochs == 0 {
            return Err(ModelError::InvalidParameter {
                parameter: "max_epochs".to_string(),
                value: self.max_epochs.to_string(),
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
        
        if let Some(split) = self.validation_split {
            if split <= T::zero() || split >= T::one() {
                return Err(ModelError::InvalidParameter {
                    parameter: "validation_split".to_string(),
                    value: "out of range".to_string(),
                    reason: "must be between 0 and 1".to_string(),
                });
            }
        }
        
        if let Some(batch_size) = self.batch_size {
            if batch_size == 0 {
                return Err(ModelError::InvalidParameter {
                    parameter: "batch_size".to_string(),
                    value: batch_size.to_string(),
                    reason: "must be greater than 0".to_string(),
                });
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_time_series_data_creation() {
        let data = TimeSeriesData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(data.len(), 5);
        assert!(!data.is_empty());
        assert!(data.validate().is_ok());
    }

    #[test]
    fn test_time_series_data_with_exogenous() {
        let data = TimeSeriesData::new(vec![1.0, 2.0, 3.0])
            .with_exogenous(vec![
                vec![0.1, 0.2],
                vec![0.3, 0.4],
                vec![0.5, 0.6],
            ]);
        
        assert!(data.is_ok());
        let data = data.unwrap();
        assert!(data.validate().is_ok());
    }

    #[test]
    fn test_forecast_result() {
        let result = ForecastResult::new(vec![1.0, 2.0, 3.0])
            .with_metadata("model".to_string(), "test".to_string());
        
        assert_eq!(result.len(), 3);
        assert!(!result.is_empty());
        assert_eq!(result.metadata.get("model"), Some(&"test".to_string()));
    }

    #[test]
    fn test_training_config() {
        let config = TrainingConfig::<f64>::new()
            .with_max_epochs(200)
            .with_learning_rate(0.01)
            .with_patience(15);
        
        assert_eq!(config.max_epochs, 200);
        assert_relative_eq!(config.learning_rate, 0.01);
        assert_eq!(config.patience, Some(15));
        assert!(config.validate().is_ok());
    }
}