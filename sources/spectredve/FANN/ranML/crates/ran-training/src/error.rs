//! Error handling for ran-training crate

use std::fmt;

/// Result type alias for training operations
pub type TrainingResult<T> = Result<T, TrainingError>;

/// Training-specific error types
#[derive(Debug, Clone)]
pub enum TrainingError {
    /// Data loading or parsing errors
    DataError(String),
    /// Invalid neural network architecture
    InvalidArchitecture(String),
    /// Training process failures
    TrainingFailed(String),
    /// Network not trained when prediction attempted
    NetworkNotTrained,
    /// Prediction computation failed
    PredictionFailed(String),
    /// Invalid input data for training or prediction
    InvalidInput(String),
    /// Configuration validation errors
    ConfigError(String),
    /// File I/O errors
    IoError(String),
    /// CSV parsing errors
    CsvError(String),
    /// FANN library errors
    FannError(String),
}

impl fmt::Display for TrainingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrainingError::DataError(msg) => write!(f, "Data error: {}", msg),
            TrainingError::InvalidArchitecture(msg) => write!(f, "Invalid architecture: {}", msg),
            TrainingError::TrainingFailed(msg) => write!(f, "Training failed: {}", msg),
            TrainingError::NetworkNotTrained => write!(f, "Network has not been trained"),
            TrainingError::PredictionFailed(msg) => write!(f, "Prediction failed: {}", msg),
            TrainingError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            TrainingError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            TrainingError::IoError(msg) => write!(f, "I/O error: {}", msg),
            TrainingError::CsvError(msg) => write!(f, "CSV error: {}", msg),
            TrainingError::FannError(msg) => write!(f, "FANN error: {}", msg),
        }
    }
}

impl std::error::Error for TrainingError {}

// Convert from common error types
impl From<std::io::Error> for TrainingError {
    fn from(error: std::io::Error) -> Self {
        TrainingError::IoError(error.to_string())
    }
}

impl From<csv::Error> for TrainingError {
    fn from(error: csv::Error) -> Self {
        TrainingError::CsvError(error.to_string())
    }
}

impl From<serde_json::Error> for TrainingError {
    fn from(error: serde_json::Error) -> Self {
        TrainingError::DataError(format!("JSON error: {}", error))
    }
}

impl From<serde_yaml::Error> for TrainingError {
    fn from(error: serde_yaml::Error) -> Self {
        TrainingError::DataError(format!("YAML error: {}", error))
    }
}

impl From<ruv_fann::NetworkError> for TrainingError {
    fn from(error: ruv_fann::NetworkError) -> Self {
        TrainingError::FannError(error.to_string())
    }
}

impl From<ruv_fann::TrainingError> for TrainingError {
    fn from(error: ruv_fann::TrainingError) -> Self {
        TrainingError::FannError(error.to_string())
    }
}

#[cfg(feature = "wasm")]
impl From<TrainingError> for wasm_bindgen::JsValue {
    fn from(error: TrainingError) -> Self {
        wasm_bindgen::JsValue::from_str(&error.to_string())
    }
}