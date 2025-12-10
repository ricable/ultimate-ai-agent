//! Error types for the climate prediction system

use thiserror::Error;

/// Result type alias for climate operations
pub type Result<T> = std::result::Result<T, ClimateError>;

/// Main error type for the climate prediction system
#[derive(Error, Debug)]
pub enum ClimateError {
    #[error("Data source error: {0}")]
    DataSource(String),

    #[error("Model inference error: {0}")]
    ModelInference(String),

    #[error("Physics constraint violation: {0}")]
    PhysicsConstraint(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Missing data: {0}")]
    MissingData(String),

    #[error("Out of bounds: {0}")]
    OutOfBounds(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl ClimateError {
    /// Create a data source error
    pub fn data_source(msg: impl Into<String>) -> Self {
        Self::DataSource(msg.into())
    }

    /// Create a model inference error
    pub fn model_inference(msg: impl Into<String>) -> Self {
        Self::ModelInference(msg.into())
    }

    /// Create a physics constraint error
    pub fn physics_constraint(msg: impl Into<String>) -> Self {
        Self::PhysicsConstraint(msg.into())
    }

    /// Create an invalid input error
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }
}
