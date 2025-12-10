//! Error types for data models

use thiserror::Error;

/// Result type for data model operations
pub type Result<T> = std::result::Result<T, DataModelError>;

/// Errors that can occur when working with data models
#[derive(Debug, Error)]
pub enum DataModelError {
    #[error("Invalid sequence: {0}")]
    InvalidSequence(String),

    #[error("Invalid quality score: {0}")]
    InvalidQualityScore(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Invalid coordinate: {0}")]
    InvalidCoordinate(String),

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Invalid range: start={start}, end={end}")]
    InvalidRange { start: u64, end: u64 },

    #[error("Parse error: {0}")]
    ParseError(String),
}
