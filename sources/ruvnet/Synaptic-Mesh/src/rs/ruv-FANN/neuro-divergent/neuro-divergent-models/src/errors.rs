//! Error types for neuro-divergent models
//!
//! This module provides comprehensive error handling for all neural forecasting operations.

use thiserror::Error;
use ruv_fann::{NetworkError, TrainingError};

/// Result type for neuro-divergent operations
pub type NeuroDivergentResult<T> = Result<T, NeuroDivergentError>;

/// Comprehensive error type for neuro-divergent models
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
    
    #[error("Network error: {0}")]
    NetworkError(#[from] NetworkError),
    
    #[error("Internal ruv-FANN training error: {0}")]
    RuvFannTrainingError(#[from] TrainingError),
    
    #[error("Time series error: {0}")]
    TimeSeriesError(String),
    
    #[error("Sequence processing error: {0}")]
    SequenceError(String),
    
    #[error("State management error: {0}")]
    StateError(String),
    
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Invalid sequence length: {0}")]
    InvalidSequenceLength(usize),
    
    #[error("Missing required feature: {0}")]
    MissingFeature(String),
    
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[cfg(feature = "polars")]
    #[error("Polars error: {0}")]
    PolarsError(#[from] polars::error::PolarsError),
    
    #[error("Custom error: {0}")]
    Custom(String),
}

impl NeuroDivergentError {
    /// Create a custom error with a message
    pub fn custom<T: Into<String>>(message: T) -> Self {
        Self::Custom(message.into())
    }
    
    /// Create a configuration error
    pub fn config<T: Into<String>>(message: T) -> Self {
        Self::ConfigError(message.into())
    }
    
    /// Create a data validation error
    pub fn data<T: Into<String>>(message: T) -> Self {
        Self::DataError(message.into())
    }
    
    /// Create a training error
    pub fn training<T: Into<String>>(message: T) -> Self {
        Self::TrainingError(message.into())
    }
    
    /// Create a prediction error
    pub fn prediction<T: Into<String>>(message: T) -> Self {
        Self::PredictionError(message.into())
    }
    
    /// Create a time series error
    pub fn time_series<T: Into<String>>(message: T) -> Self {
        Self::TimeSeriesError(message.into())
    }
    
    /// Create a sequence processing error
    pub fn sequence<T: Into<String>>(message: T) -> Self {
        Self::SequenceError(message.into())
    }
    
    /// Create a state management error
    pub fn state<T: Into<String>>(message: T) -> Self {
        Self::StateError(message.into())
    }
    
    /// Create a dimension mismatch error
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }
    
    /// Create an invalid sequence length error
    pub fn invalid_sequence_length(length: usize) -> Self {
        Self::InvalidSequenceLength(length)
    }
    
    /// Create a missing feature error
    pub fn missing_feature<T: Into<String>>(feature: T) -> Self {
        Self::MissingFeature(feature.into())
    }
    
    /// Check if this is a recoverable error
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::ConfigError(_) => false,
            Self::DataError(_) => false,
            Self::TrainingError(_) => true,
            Self::PredictionError(_) => true,
            Self::NetworkError(_) => false,
            Self::RuvFannTrainingError(_) => true,
            Self::TimeSeriesError(_) => false,
            Self::SequenceError(_) => true,
            Self::StateError(_) => true,
            Self::DimensionMismatch { .. } => false,
            Self::InvalidSequenceLength(_) => false,
            Self::MissingFeature(_) => false,
            Self::IoError(_) => false,
            Self::SerializationError(_) => false,
            #[cfg(feature = "polars")]
            Self::PolarsError(_) => false,
            Self::Custom(_) => true,
        }
    }
    
    /// Get error category for logging and monitoring
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::ConfigError(_) => ErrorCategory::Configuration,
            Self::DataError(_) => ErrorCategory::Data,
            Self::TrainingError(_) => ErrorCategory::Training,
            Self::PredictionError(_) => ErrorCategory::Prediction,
            Self::NetworkError(_) => ErrorCategory::Network,
            Self::RuvFannTrainingError(_) => ErrorCategory::Training,
            Self::TimeSeriesError(_) => ErrorCategory::Data,
            Self::SequenceError(_) => ErrorCategory::Processing,
            Self::StateError(_) => ErrorCategory::State,
            Self::DimensionMismatch { .. } => ErrorCategory::Validation,
            Self::InvalidSequenceLength(_) => ErrorCategory::Validation,
            Self::MissingFeature(_) => ErrorCategory::Configuration,
            Self::IoError(_) => ErrorCategory::IO,
            Self::SerializationError(_) => ErrorCategory::IO,
            #[cfg(feature = "polars")]
            Self::PolarsError(_) => ErrorCategory::Data,
            Self::Custom(_) => ErrorCategory::Other,
        }
    }
}

/// Error categories for classification and handling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    Configuration,
    Data,
    Training,
    Prediction,
    Network,
    Processing,
    State,
    Validation,
    IO,
    Other,
}

impl std::fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Configuration => write!(f, "Configuration"),
            Self::Data => write!(f, "Data"),
            Self::Training => write!(f, "Training"),
            Self::Prediction => write!(f, "Prediction"),
            Self::Network => write!(f, "Network"),
            Self::Processing => write!(f, "Processing"),
            Self::State => write!(f, "State"),
            Self::Validation => write!(f, "Validation"),
            Self::IO => write!(f, "IO"),
            Self::Other => write!(f, "Other"),
        }
    }
}

/// Context for error reporting and debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub operation: String,
    pub model_name: Option<String>,
    pub epoch: Option<usize>,
    pub batch: Option<usize>,
    pub additional_info: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    pub fn new<T: Into<String>>(operation: T) -> Self {
        Self {
            operation: operation.into(),
            model_name: None,
            epoch: None,
            batch: None,
            additional_info: std::collections::HashMap::new(),
        }
    }
    
    pub fn with_model<T: Into<String>>(mut self, model_name: T) -> Self {
        self.model_name = Some(model_name.into());
        self
    }
    
    pub fn with_epoch(mut self, epoch: usize) -> Self {
        self.epoch = Some(epoch);
        self
    }
    
    pub fn with_batch(mut self, batch: usize) -> Self {
        self.batch = Some(batch);
        self
    }
    
    pub fn with_info<K, V>(mut self, key: K, value: V) -> Self 
    where 
        K: Into<String>,
        V: Into<String>,
    {
        self.additional_info.insert(key.into(), value.into());
        self
    }
}

/// Trait for adding context to errors
pub trait ErrorContextExt<T> {
    fn with_context(self, context: ErrorContext) -> NeuroDivergentResult<T>;
}

impl<T> ErrorContextExt<T> for NeuroDivergentResult<T> {
    fn with_context(self, context: ErrorContext) -> NeuroDivergentResult<T> {
        self.map_err(|e| {
            let context_info = format!(
                "Operation: {}, Model: {}, Epoch: {}, Batch: {}",
                context.operation,
                context.model_name.unwrap_or_else(|| "Unknown".to_string()),
                context.epoch.map(|e| e.to_string()).unwrap_or_else(|| "N/A".to_string()),
                context.batch.map(|b| b.to_string()).unwrap_or_else(|| "N/A".to_string())
            );
            
            match e {
                NeuroDivergentError::Custom(msg) => {
                    NeuroDivergentError::custom(format!("{} | Context: {}", msg, context_info))
                }
                other => other,
            }
        })
    }
}