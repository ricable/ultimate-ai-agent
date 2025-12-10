//! Error handling for Kimi-FANN core
//!
//! This module defines the error types used throughout the Kimi-FANN system.

use thiserror::Error;

/// Main error type for Kimi-FANN operations
#[derive(Error, Debug)]
pub enum KimiError {
    #[error("Expert error: {0}")]
    ExpertError(String),

    #[error("Routing error: {0}")]
    RoutingError(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("Compression error: {0}")]
    CompressionError(String),

    #[error("Execution error: {0}")]
    ExecutionError(String),

    #[error("Context error: {0}")]
    ContextError(String),

    #[error("Neural network error: {0}")]
    NeuralNetworkError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("WASM error: {0}")]
    WasmError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Timeout error: operation took longer than {timeout}ms")]
    TimeoutError { timeout: u64 },

    #[error("Expert not found: {expert_id}")]
    ExpertNotFound { expert_id: u32 },

    #[error("Invalid expert state: {details}")]
    InvalidExpertState { details: String },
}

/// Result type alias for Kimi-FANN operations
pub type Result<T> = std::result::Result<T, KimiError>;

impl From<serde_json::Error> for KimiError {
    fn from(err: serde_json::Error) -> Self {
        KimiError::SerializationError(err.to_string())
    }
}

impl From<std::io::Error> for KimiError {
    fn from(err: std::io::Error) -> Self {
        KimiError::NetworkError(err.to_string())
    }
}