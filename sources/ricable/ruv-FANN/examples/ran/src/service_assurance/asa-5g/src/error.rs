use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Configuration error: {0}")]
    Config(#[from] config::ConfigError),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Machine learning error: {0}")]
    MachineLearning(String),
    
    #[error("Prediction error: {0}")]
    Prediction(String),
    
    #[error("Signal analysis error: {0}")]
    SignalAnalysis(String),
    
    #[error("Model training error: {0}")]
    ModelTraining(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Data validation error: {0}")]
    DataValidation(String),
    
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
    
    #[error("Performance threshold not met: expected {expected}, got {actual}")]
    PerformanceThreshold { expected: f64, actual: f64 },
    
    #[error("ENDC configuration error: {0}")]
    EndcConfiguration(String),
    
    #[error("Bearer setup error: {0}")]
    BearerSetup(String),
    
    #[error("Signal quality degradation: {0}")]
    SignalQuality(String),
    
    #[error("Network congestion detected: {0}")]
    NetworkCongestion(String),
    
    #[error("gRPC error: {0}")]
    Grpc(#[from] tonic::Status),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("General error: {0}")]
    General(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, Error>;