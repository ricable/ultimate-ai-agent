//! Core foundation traits and structures for neural forecasting models
//!
//! This module defines the fundamental interfaces that all neural forecasting models
//! must implement, providing a unified API for training, prediction, and evaluation.

use std::collections::HashMap;
use std::marker::PhantomData;
use chrono::{DateTime, Utc};
use num_traits::Float;
use ruv_fann::{Network, TrainingData};
use crate::errors::{NeuroDivergentError, NeuroDivergentResult};

/// Core trait that all neural forecasting models must implement
pub trait BaseModel<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>: Send + Sync {
    /// Get the model's name/identifier
    fn name(&self) -> &str;
    
    /// Get the model's configuration as a generic structure
    fn config(&self) -> &dyn ModelConfig<T>;
    
    /// Check if the model has been trained
    fn is_trained(&self) -> bool;
    
    /// Train the model on the provided data
    fn fit(&mut self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<TrainingMetrics<T>>;
    
    /// Generate forecasts for the given input
    fn predict(&self, input: &TimeSeriesInput<T>) -> NeuroDivergentResult<ForecastOutput<T>>;
    
    /// Generate forecasts for multiple time series
    fn predict_batch(&self, inputs: &[TimeSeriesInput<T>]) -> NeuroDivergentResult<Vec<ForecastOutput<T>>>;
    
    /// Validate the model on validation data
    fn validate(&self, data: &TimeSeriesDataset<T>) -> NeuroDivergentResult<ValidationMetrics<T>>;
    
    /// Save the model state to bytes
    fn save_state(&self) -> NeuroDivergentResult<Vec<u8>>;
    
    /// Load the model state from bytes
    fn load_state(&mut self, state: &[u8]) -> NeuroDivergentResult<()>;
    
    /// Get the input size expected by the model
    fn input_size(&self) -> usize;
    
    /// Get the forecast horizon
    fn horizon(&self) -> usize;
    
    /// Reset the model to untrained state
    fn reset(&mut self);
    
    /// Get training history if available
    fn training_history(&self) -> Option<&TrainingHistory<T>>;
    
    /// Get model-specific metrics
    fn model_metrics(&self) -> HashMap<String, f64>;
}

/// Configuration trait for all model configurations
pub trait ModelConfig<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>: Send + Sync {
    /// Get the model type identifier
    fn model_type(&self) -> &'static str;
    
    /// Get the forecast horizon
    fn horizon(&self) -> usize;
    
    /// Get the input size
    fn input_size(&self) -> usize;
    
    /// Validate the configuration
    fn validate(&self) -> NeuroDivergentResult<()>;
    
    /// Convert to a generic configuration map
    fn to_map(&self) -> HashMap<String, ConfigValue<T>>;
    
    /// Create from a generic configuration map
    fn from_map(map: HashMap<String, ConfigValue<T>>) -> NeuroDivergentResult<Self>
    where
        Self: Sized;
}

/// Generic configuration value for model parameters
#[derive(Debug, Clone)]
pub enum ConfigValue<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    Float(T),
    Int(i64),
    UInt(usize),
    String(String),
    Bool(bool),
    FloatVec(Vec<T>),
    IntVec(Vec<i64>),
    UIntVec(Vec<usize>),
    StringVec(Vec<String>),
}

/// Adapter trait for integrating ruv-FANN networks with time series models
pub trait NetworkAdapter<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    /// Convert time series input to network input format
    fn prepare_input(&self, ts_input: &TimeSeriesInput<T>) -> NeuroDivergentResult<Vec<T>>;
    
    /// Convert network output to forecast output format
    fn process_output(&self, network_output: Vec<T>) -> NeuroDivergentResult<ForecastOutput<T>>;
    
    /// Create training data for the underlying network
    fn create_training_data(&self, dataset: &TimeSeriesDataset<T>) -> NeuroDivergentResult<TrainingData<T>>;
    
    /// Get the underlying network reference
    fn network(&self) -> &Network<T>;
    
    /// Get the underlying network mutable reference
    fn network_mut(&mut self) -> &mut Network<T>;
}

/// Time series input structure for models
#[derive(Debug, Clone)]
pub struct TimeSeriesInput<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    /// Historical target values
    pub historical_targets: Vec<T>,
    
    /// Static exogenous features (constant across time)
    pub static_features: Option<Vec<T>>,
    
    /// Historical exogenous features (time-varying, known history)
    pub historical_features: Option<Vec<Vec<T>>>,
    
    /// Future exogenous features (time-varying, known future)
    pub future_features: Option<Vec<Vec<T>>>,
    
    /// Unique identifier for the time series
    pub unique_id: Option<String>,
    
    /// Time index for the last historical point
    pub last_timestamp: Option<DateTime<Utc>>,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> TimeSeriesInput<T> {
    /// Create a new time series input with just historical targets
    pub fn new(historical_targets: Vec<T>) -> Self {
        Self {
            historical_targets,
            static_features: None,
            historical_features: None,
            future_features: None,
            unique_id: None,
            last_timestamp: None,
        }
    }
    
    /// Add static features
    pub fn with_static_features(mut self, features: Vec<T>) -> Self {
        self.static_features = Some(features);
        self
    }
    
    /// Add historical exogenous features
    pub fn with_historical_features(mut self, features: Vec<Vec<T>>) -> Self {
        self.historical_features = Some(features);
        self
    }
    
    /// Add future exogenous features
    pub fn with_future_features(mut self, features: Vec<Vec<T>>) -> Self {
        self.future_features = Some(features);
        self
    }
    
    /// Set unique identifier
    pub fn with_id<S: Into<String>>(mut self, id: S) -> Self {
        self.unique_id = Some(id.into());
        self
    }
    
    /// Set last timestamp
    pub fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.last_timestamp = Some(timestamp);
        self
    }
    
    /// Get the total input dimension
    pub fn input_dimension(&self) -> usize {
        let mut dim = self.historical_targets.len();
        
        if let Some(static_features) = &self.static_features {
            dim += static_features.len();
        }
        
        if let Some(hist_features) = &self.historical_features {
            dim += hist_features.iter().map(|f| f.len()).sum::<usize>();
        }
        
        if let Some(fut_features) = &self.future_features {
            dim += fut_features.iter().map(|f| f.len()).sum::<usize>();
        }
        
        dim
    }
    
    /// Validate the input structure
    pub fn validate(&self) -> NeuroDivergentResult<()> {
        if self.historical_targets.is_empty() {
            return Err(NeuroDivergentError::data("Historical targets cannot be empty"));
        }
        
        // Check feature alignment
        if let Some(hist_features) = &self.historical_features {
            if hist_features.len() != self.historical_targets.len() {
                return Err(NeuroDivergentError::dimension_mismatch(
                    self.historical_targets.len(),
                    hist_features.len(),
                ));
            }
        }
        
        Ok(())
    }
}

/// Forecast output structure from models
#[derive(Debug, Clone)]
pub struct ForecastOutput<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    /// Point forecasts for each horizon step
    pub forecasts: Vec<T>,
    
    /// Prediction intervals if available
    pub prediction_intervals: Option<Vec<PredictionInterval<T>>>,
    
    /// Model confidence scores
    pub confidence_scores: Option<Vec<T>>,
    
    /// Forecast timestamps
    pub timestamps: Option<Vec<DateTime<Utc>>>,
    
    /// Unique identifier for the time series
    pub unique_id: Option<String>,
    
    /// Additional model-specific outputs
    pub additional_outputs: HashMap<String, Vec<T>>,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> ForecastOutput<T> {
    /// Create a new forecast output with point forecasts
    pub fn new(forecasts: Vec<T>) -> Self {
        Self {
            forecasts,
            prediction_intervals: None,
            confidence_scores: None,
            timestamps: None,
            unique_id: None,
            additional_outputs: HashMap::new(),
        }
    }
    
    /// Add prediction intervals
    pub fn with_intervals(mut self, intervals: Vec<PredictionInterval<T>>) -> Self {
        self.prediction_intervals = Some(intervals);
        self
    }
    
    /// Add confidence scores
    pub fn with_confidence(mut self, scores: Vec<T>) -> Self {
        self.confidence_scores = Some(scores);
        self
    }
    
    /// Add timestamps
    pub fn with_timestamps(mut self, timestamps: Vec<DateTime<Utc>>) -> Self {
        self.timestamps = Some(timestamps);
        self
    }
    
    /// Set unique identifier
    pub fn with_id<S: Into<String>>(mut self, id: S) -> Self {
        self.unique_id = Some(id.into());
        self
    }
    
    /// Add additional output
    pub fn with_additional<S: Into<String>>(mut self, name: S, values: Vec<T>) -> Self {
        self.additional_outputs.insert(name.into(), values);
        self
    }
    
    /// Get forecast horizon
    pub fn horizon(&self) -> usize {
        self.forecasts.len()
    }
    
    /// Validate the output structure
    pub fn validate(&self) -> NeuroDivergentResult<()> {
        if self.forecasts.is_empty() {
            return Err(NeuroDivergentError::prediction("Forecasts cannot be empty"));
        }
        
        // Check prediction intervals alignment
        if let Some(intervals) = &self.prediction_intervals {
            if intervals.len() != self.forecasts.len() {
                return Err(NeuroDivergentError::dimension_mismatch(
                    self.forecasts.len(),
                    intervals.len(),
                ));
            }
        }
        
        // Check confidence scores alignment
        if let Some(scores) = &self.confidence_scores {
            if scores.len() != self.forecasts.len() {
                return Err(NeuroDivergentError::dimension_mismatch(
                    self.forecasts.len(),
                    scores.len(),
                ));
            }
        }
        
        // Check timestamps alignment
        if let Some(timestamps) = &self.timestamps {
            if timestamps.len() != self.forecasts.len() {
                return Err(NeuroDivergentError::dimension_mismatch(
                    self.forecasts.len(),
                    timestamps.len(),
                ));
            }
        }
        
        Ok(())
    }
}

/// Prediction interval structure
#[derive(Debug, Clone)]
pub struct PredictionInterval<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    pub lower_bound: T,
    pub upper_bound: T,
    pub confidence_level: f64,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> PredictionInterval<T> {
    pub fn new(lower_bound: T, upper_bound: T, confidence_level: f64) -> Self {
        Self {
            lower_bound,
            upper_bound,
            confidence_level,
        }
    }
    
    pub fn width(&self) -> T {
        self.upper_bound - self.lower_bound
    }
    
    pub fn contains(&self, value: T) -> bool {
        value >= self.lower_bound && value <= self.upper_bound
    }
}

/// Dataset structure for training and validation
#[derive(Debug, Clone)]
pub struct TimeSeriesDataset<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    /// Training examples
    pub samples: Vec<TimeSeriesSample<T>>,
    
    /// Dataset metadata
    pub metadata: DatasetMetadata,
}

/// Individual time series sample for training
#[derive(Debug, Clone)]
pub struct TimeSeriesSample<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    /// Input for the model
    pub input: TimeSeriesInput<T>,
    
    /// Target values to predict
    pub target: Vec<T>,
    
    /// Sample weight for training
    pub weight: Option<T>,
}

/// Dataset metadata
#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    pub name: Option<String>,
    pub description: Option<String>,
    pub num_series: usize,
    pub total_samples: usize,
    pub feature_names: Vec<String>,
    pub target_names: Vec<String>,
}

/// Training metrics returned after model training
#[derive(Debug, Clone)]
pub struct TrainingMetrics<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    pub final_loss: T,
    pub epochs_completed: usize,
    pub training_time_seconds: f64,
    pub best_validation_loss: Option<T>,
    pub early_stopped: bool,
    pub convergence_achieved: bool,
}

/// Validation metrics for model evaluation
#[derive(Debug, Clone)]
pub struct ValidationMetrics<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    pub mse: T,
    pub mae: T,
    pub mape: Option<T>,
    pub smape: Option<T>,
    pub r2_score: Option<T>,
    pub directional_accuracy: Option<T>,
    pub custom_metrics: HashMap<String, T>,
}

/// Training history tracking
#[derive(Debug, Clone)]
pub struct TrainingHistory<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    pub epoch_losses: Vec<T>,
    pub validation_losses: Vec<T>,
    pub learning_rates: Vec<T>,
    pub training_times: Vec<f64>,
    pub custom_metrics: HashMap<String, Vec<T>>,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> TrainingHistory<T> {
    pub fn new() -> Self {
        Self {
            epoch_losses: Vec::new(),
            validation_losses: Vec::new(),
            learning_rates: Vec::new(),
            training_times: Vec::new(),
            custom_metrics: HashMap::new(),
        }
    }
    
    pub fn add_epoch(&mut self, loss: T, val_loss: Option<T>, lr: T, time: f64) {
        self.epoch_losses.push(loss);
        if let Some(val_loss) = val_loss {
            self.validation_losses.push(val_loss);
        }
        self.learning_rates.push(lr);
        self.training_times.push(time);
    }
    
    pub fn add_custom_metric(&mut self, name: String, value: T) {
        self.custom_metrics.entry(name).or_insert_with(Vec::new).push(value);
    }
    
    pub fn best_validation_loss(&self) -> Option<T> {
        self.validation_losses.iter().fold(None, |acc, &loss| {
            match acc {
                None => Some(loss),
                Some(best) => Some(best.min(loss)),
            }
        })
    }
    
    pub fn total_training_time(&self) -> f64 {
        self.training_times.iter().sum()
    }
    
    pub fn num_epochs(&self) -> usize {
        self.epoch_losses.len()
    }
}

/// Validation configuration for model evaluation
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub validation_split: f64,
    pub shuffle: bool,
    pub stratify: bool,
    pub random_seed: Option<u64>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            validation_split: 0.2,
            shuffle: true,
            stratify: false,
            random_seed: None,
        }
    }
}

/// Utility trait for recurrent models to manage temporal state
pub trait RecurrentState<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static>: Send + Sync {
    /// Reset the internal state
    fn reset(&mut self);
    
    /// Get the current state
    fn get_state(&self) -> Vec<T>;
    
    /// Set the state
    fn set_state(&mut self, state: Vec<T>) -> NeuroDivergentResult<()>;
    
    /// Get the state dimension
    fn state_dimension(&self) -> usize;
    
    /// Clone the current state
    fn clone_state(&self) -> Box<dyn RecurrentState<T>>;
}

/// Sequence processing utilities for variable-length sequences
pub struct SequenceProcessor<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> {
    phantom: PhantomData<T>,
}

impl<T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> SequenceProcessor<T> {
    /// Pad sequences to the same length
    pub fn pad_sequences(
        sequences: &[Vec<T>], 
        max_length: Option<usize>, 
        padding_value: T
    ) -> Vec<Vec<T>> {
        let target_length = max_length.unwrap_or_else(|| {
            sequences.iter().map(|s| s.len()).max().unwrap_or(0)
        });
        
        sequences.iter()
            .map(|seq| {
                let mut padded = seq.clone();
                padded.resize(target_length, padding_value);
                padded
            })
            .collect()
    }
    
    /// Create attention masks for padded sequences
    pub fn create_attention_masks(
        sequences: &[Vec<T>], 
        padded_length: usize
    ) -> Vec<Vec<bool>> {
        sequences.iter()
            .map(|seq| {
                let mut mask = vec![true; seq.len()];
                mask.resize(padded_length, false);
                mask
            })
            .collect()
    }
    
    /// Truncate sequences to maximum length
    pub fn truncate_sequences(
        sequences: &[Vec<T>], 
        max_length: usize
    ) -> Vec<Vec<T>> {
        sequences.iter()
            .map(|seq| {
                if seq.len() > max_length {
                    seq[..max_length].to_vec()
                } else {
                    seq.clone()
                }
            })
            .collect()
    }
}