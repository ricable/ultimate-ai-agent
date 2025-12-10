//! Error types for immune analysis

use thiserror::Error;

/// Result type for analysis operations
pub type Result<T> = std::result::Result<T, AnalysisError>;

/// Errors that can occur during immune analysis
#[derive(Debug, Error)]
pub enum AnalysisError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Statistical error: {0}")]
    Statistical(String),

    #[error("Data model error: {0}")]
    DataModel(#[from] data_models::error::DataModelError),

    #[error("Invalid configuration: {0}")]
    Config(String),

    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    #[error("Computation error: {0}")]
    Computation(String),
}
