use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Configuration error: {0}")]
    Config(#[from] config::ConfigError),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Machine learning error: {0}")]
    MachineLearning(String),
    
    #[error("Feature extraction error: {0}")]
    FeatureExtraction(String),
    
    #[error("Signal processing error: {0}")]
    SignalProcessing(String),
    
    #[error("Model training error: {0}")]
    ModelTraining(String),
    
    #[error("Classification error: {0}")]
    Classification(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Data validation error: {0}")]
    DataValidation(String),
    
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
    
    #[error("Performance threshold not met: expected {expected}, got {actual}")]
    PerformanceThreshold { expected: f64, actual: f64 },
    
    #[error("gRPC error: {0}")]
    Grpc(#[from] tonic::Status),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("General error: {0}")]
    General(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, Error>;