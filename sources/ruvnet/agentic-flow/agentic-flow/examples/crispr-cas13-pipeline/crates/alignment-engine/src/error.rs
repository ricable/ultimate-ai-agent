//! Error types for alignment engine

use thiserror::Error;

/// Result type for alignment operations
pub type Result<T> = std::result::Result<T, AlignmentError>;

/// Errors that can occur during alignment
#[derive(Debug, Error)]
pub enum AlignmentError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid reference file: {0}")]
    InvalidReference(String),

    #[error("Alignment failed: {0}")]
    AlignmentFailed(String),

    #[error("Invalid BAM file: {0}")]
    InvalidBam(String),

    #[error("Data model error: {0}")]
    DataModel(#[from] data_models::error::DataModelError),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Resource not found: {0}")]
    NotFound(String),

    #[error("Parse error: {0}")]
    Parse(String),
}
