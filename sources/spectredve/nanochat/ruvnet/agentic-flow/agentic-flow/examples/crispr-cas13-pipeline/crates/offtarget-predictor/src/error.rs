//! Error types for off-target prediction

use thiserror::Error;

/// Result type for prediction operations
pub type Result<T> = std::result::Result<T, PredictionError>;

/// Errors that can occur during off-target prediction
#[derive(Debug, Error)]
pub enum PredictionError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Feature extraction error: {0}")]
    FeatureExtraction(String),

    #[error("Invalid configuration: {0}")]
    Config(String),

    #[error("Data model error: {0}")]
    DataModel(#[from] data_models::error::DataModelError),

    #[error("Alignment error: {0}")]
    Alignment(#[from] alignment_engine::error::AlignmentError),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Computation error: {0}")]
    Computation(String),
}
