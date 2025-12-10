//! Specialized neural forecasting models module
//!
//! This module contains implementations of advanced neural forecasting models with unique
//! architectures, built on top of the ruv-FANN foundation. Each model provides specialized
//! capabilities for different types of time series forecasting tasks.

use std::collections::HashMap;
use std::fmt;
use ruv_fann::Network;
use crate::Float;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

// Model implementations
pub mod deepar;
pub mod deepnpts;
pub mod tcn;
pub mod bitcn;
pub mod timesnet;
pub mod stemgnn;
pub mod tsmixer;
pub mod tsmixerx;
pub mod timellm;

// Re-export models
pub use deepar::DeepAR;
pub use deepnpts::DeepNPTS;
pub use tcn::TCN;
pub use bitcn::BiTCN;
pub use timesnet::TimesNet;
pub use stemgnn::StemGNN;
pub use tsmixer::TSMixer;
pub use tsmixerx::TSMixerx;
pub use timellm::TimeLLM;

/// Errors that can occur during model operations
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Training error: {0}")]
    TrainingError(String),
    
    #[error("Prediction error: {0}")]
    PredictionError(String),
    
    #[error("Data processing error: {0}")]
    DataError(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Invalid input dimensions: expected {expected}, got {actual}")]
    DimensionError { expected: usize, actual: usize },
    
    #[error("Model not trained")]
    NotTrainedError,
    
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
}

/// Time series data structure for model input/output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesData<T: Float> {
    /// Timestamps for each observation
    pub timestamps: Vec<DateTime<Utc>>,
    /// Target variable values
    pub values: Vec<T>,
    /// Static features (constant across time)
    pub static_features: Option<Vec<T>>,
    /// Historical exogenous variables
    pub exogenous_historical: Option<Vec<Vec<T>>>,
    /// Future known exogenous variables
    pub exogenous_future: Option<Vec<Vec<T>>>,
    /// Series identifier
    pub series_id: String,
}

impl<T: Float> TimeSeriesData<T> {
    pub fn new(series_id: String, timestamps: Vec<DateTime<Utc>>, values: Vec<T>) -> Self {
        Self {
            timestamps,
            values,
            static_features: None,
            exogenous_historical: None,
            exogenous_future: None,
            series_id,
        }
    }
    
    pub fn with_static_features(mut self, features: Vec<T>) -> Self {
        self.static_features = Some(features);
        self
    }
    
    pub fn with_exogenous_historical(mut self, exog: Vec<Vec<T>>) -> Self {
        self.exogenous_historical = Some(exog);
        self
    }
    
    pub fn with_exogenous_future(mut self, exog: Vec<Vec<T>>) -> Self {
        self.exogenous_future = Some(exog);
        self
    }
    
    pub fn len(&self) -> usize {
        self.values.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Prediction result with forecasts and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult<T: Float> {
    /// Point forecasts
    pub forecasts: Vec<T>,
    /// Forecast timestamps
    pub timestamps: Vec<DateTime<Utc>>,
    /// Series identifier
    pub series_id: String,
    /// Prediction intervals (optional)
    pub intervals: Option<PredictionIntervals<T>>,
    /// Model metadata
    pub metadata: HashMap<String, String>,
}

/// Prediction intervals for uncertainty quantification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionIntervals<T: Float> {
    /// Lower bounds for different confidence levels
    pub lower_bounds: HashMap<String, Vec<T>>,
    /// Upper bounds for different confidence levels  
    pub upper_bounds: HashMap<String, Vec<T>>,
    /// Confidence levels (e.g., "0.8", "0.9", "0.95")
    pub confidence_levels: Vec<String>,
}

/// Training configuration for models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig<T: Float> {
    /// Maximum number of training epochs
    pub max_epochs: usize,
    /// Learning rate
    pub learning_rate: T,
    /// Batch size
    pub batch_size: usize,
    /// Early stopping patience
    pub patience: Option<usize>,
    /// Validation split ratio
    pub validation_split: Option<T>,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl<T: Float> Default for TrainingConfig<T> {
    fn default() -> Self {
        Self {
            max_epochs: 100,
            learning_rate: T::from(0.001).unwrap(),
            batch_size: 32,
            patience: Some(10),
            validation_split: Some(T::from(0.2).unwrap()),
            random_seed: None,
        }
    }
}

/// Base trait that all specialized forecasting models must implement
pub trait BaseModel<T: Float>: Send + Sync {
    type Config: ModelConfig<T>;
    
    /// Create a new model instance with the given configuration
    fn new(config: Self::Config) -> Result<Self, ModelError> where Self: Sized;
    
    /// Train the model on the provided time series data
    fn fit(&mut self, data: &TimeSeriesData<T>, config: &TrainingConfig<T>) -> Result<(), ModelError>;
    
    /// Generate predictions for the given input data
    fn predict(&self, data: &TimeSeriesData<T>) -> Result<PredictionResult<T>, ModelError>;
    
    /// Check if the model has been trained
    fn is_trained(&self) -> bool;
    
    /// Reset the model to untrained state
    fn reset(&mut self) -> Result<(), ModelError>;
    
    /// Get model configuration
    fn config(&self) -> &Self::Config;
    
    /// Validate input data compatibility
    fn validate_input(&self, data: &TimeSeriesData<T>) -> Result<(), ModelError>;
    
    /// Get model name/identifier
    fn name(&self) -> &'static str;
    
    /// Get model metadata
    fn metadata(&self) -> HashMap<String, String> {
        let mut meta = HashMap::new();
        meta.insert("name".to_string(), self.name().to_string());
        meta.insert("trained".to_string(), self.is_trained().to_string());
        meta
    }
}

/// Configuration trait for model parameters
pub trait ModelConfig<T: Float>: Clone + Send + Sync + fmt::Debug {
    /// Validate configuration parameters
    fn validate(&self) -> Result<(), ModelError>;
    
    /// Get the forecast horizon
    fn horizon(&self) -> usize;
    
    /// Get the input window size
    fn input_size(&self) -> usize;
    
    /// Get the number of static features
    fn static_features_size(&self) -> usize {
        0
    }
    
    /// Get the number of exogenous features
    fn exogenous_features_size(&self) -> usize {
        0
    }
}

/// Probabilistic forecasting capability
pub trait ProbabilisticForecasting<T: Float> {
    /// Generate probabilistic forecasts with prediction intervals
    fn predict_with_intervals(
        &self,  
        data: &TimeSeriesData<T>,
        confidence_levels: &[T]
    ) -> Result<PredictionResult<T>, ModelError>;
    
    /// Generate quantile forecasts
    fn predict_quantiles(
        &self,
        data: &TimeSeriesData<T>, 
        quantiles: &[T]
    ) -> Result<PredictionResult<T>, ModelError>;
    
    /// Sample from the predictive distribution
    fn sample_predictions(
        &self,
        data: &TimeSeriesData<T>,
        num_samples: usize
    ) -> Result<Vec<Vec<T>>, ModelError>;
}

/// Multi-variate time series forecasting capability
pub trait MultivariateForecasting<T: Float> {
    /// Predict multiple target variables simultaneously
    fn predict_multivariate(
        &self,
        data: &[TimeSeriesData<T>]
    ) -> Result<Vec<PredictionResult<T>>, ModelError>;
}

/// Interpretability and explainability features
pub trait ModelInterpretability<T: Float> {
    /// Get feature importance scores
    fn feature_importance(&self) -> Result<HashMap<String, T>, ModelError>;
    
    /// Get attention weights (for attention-based models)
    fn attention_weights(&self, data: &TimeSeriesData<T>) -> Result<Vec<Vec<T>>, ModelError>;
    
    /// Get decomposition components (trend, seasonal, etc.)
    fn decompose(&self, data: &TimeSeriesData<T>) -> Result<HashMap<String, Vec<T>>, ModelError>;
}

/// Transfer learning capabilities
pub trait TransferLearning<T: Float> {
    /// Fine-tune model on new data
    fn fine_tune(&mut self, data: &TimeSeriesData<T>, config: &TrainingConfig<T>) -> Result<(), ModelError>;
    
    /// Extract features for transfer to other models
    fn extract_features(&self, data: &TimeSeriesData<T>) -> Result<Vec<T>, ModelError>;
}

/// Model serialization and persistence
pub trait ModelPersistence<T: Float> {
    /// Save model to bytes
    fn save_to_bytes(&self) -> Result<Vec<u8>, ModelError>;
    
    /// Load model from bytes
    fn load_from_bytes(bytes: &[u8]) -> Result<Self, ModelError> where Self: Sized;
    
    /// Save model to file
    fn save_to_file(&self, path: &str) -> Result<(), ModelError> {
        let bytes = self.save_to_bytes()?;
        std::fs::write(path, bytes).map_err(|e| ModelError::DataError(e.to_string()))?;
        Ok(())
    }
    
    /// Load model from file
    fn load_from_file(path: &str) -> Result<Self, ModelError> where Self: Sized {
        let bytes = std::fs::read(path).map_err(|e| ModelError::DataError(e.to_string()))?;
        Self::load_from_bytes(&bytes)
    }
}

/// Utility functions for common operations
pub mod utils {
    use super::*;
    
    /// Create lagged features from time series data
    pub fn create_lagged_features<T: Float>(values: &[T], lags: &[usize]) -> Vec<Vec<T>> {
        let mut features = Vec::new();
        for &lag in lags {
            if lag < values.len() {
                let lagged: Vec<T> = values.iter().skip(lag).cloned().collect();
                features.push(lagged);
            }
        }
        features
    }
    
    /// Normalize time series values
    pub fn normalize<T: Float>(values: &[T]) -> (Vec<T>, T, T) {
        let mean = values.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(values.len()).unwrap();
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .fold(T::zero(), |acc, x| acc + x) / T::from(values.len()).unwrap();
        let std = variance.sqrt();
        
        let normalized: Vec<T> = values.iter()
            .map(|&x| (x - mean) / std)
            .collect();
            
        (normalized, mean, std)
    }
    
    /// Denormalize time series values
    pub fn denormalize<T: Float>(normalized: &[T], mean: T, std: T) -> Vec<T> {
        normalized.iter()
            .map(|&x| x * std + mean)
            .collect()
    }
    
    /// Split time series data for training and validation
    pub fn train_test_split<T: Float>(
        data: &TimeSeriesData<T>, 
        test_size: T
    ) -> Result<(TimeSeriesData<T>, TimeSeriesData<T>), ModelError> {
        let total_len = data.len();
        let test_len = (T::from(total_len).unwrap() * test_size).to_usize().unwrap();
        let train_len = total_len - test_len;
        
        if train_len == 0 || test_len == 0 {
            return Err(ModelError::DataError("Invalid split ratio".to_string()));
        }
        
        let train_data = TimeSeriesData {
            timestamps: data.timestamps[..train_len].to_vec(),
            values: data.values[..train_len].to_vec(),
            static_features: data.static_features.clone(),
            exogenous_historical: data.exogenous_historical.as_ref()
                .map(|exog| exog.iter().map(|series| series[..train_len].to_vec()).collect()),
            exogenous_future: data.exogenous_future.clone(),
            series_id: data.series_id.clone(),
        };
        
        let test_data = TimeSeriesData {
            timestamps: data.timestamps[train_len..].to_vec(),
            values: data.values[train_len..].to_vec(),
            static_features: data.static_features.clone(),
            exogenous_historical: data.exogenous_historical.as_ref()
                .map(|exog| exog.iter().map(|series| series[train_len..].to_vec()).collect()),
            exogenous_future: data.exogenous_future.clone(),
            series_id: data.series_id.clone(),
        };
        
        Ok((train_data, test_data))
    }
}