//! Error types for neural network operations

use thiserror::Error;

/// Main error type for neural network operations
#[derive(Error, Debug)]
pub enum NeuralError {
    /// Network creation error
    #[error("Network creation failed: {0}")]
    NetworkCreation(String),

    /// Model loading error
    #[error("Model loading failed: {0}")]
    ModelLoading(String),

    /// Model saving error
    #[error("Model saving failed: {0}")]
    ModelSaving(String),

    /// Inference error
    #[error("Inference failed: {0}")]
    InferenceError(String),

    /// Training error
    #[error("Training failed: {0}")]
    TrainingError(String),

    /// Invalid input error
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Feature extraction error
    #[error("Feature extraction failed: {0}")]
    FeatureExtraction(String),

    /// Model not found error
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Data validation error
    #[error("Data validation failed: {0}")]
    DataValidation(String),

    /// Resource error (memory, GPU, etc.)
    #[error("Resource error: {0}")]
    Resource(String),

    /// Timeout error
    #[error("Operation timed out: {0}")]
    Timeout(String),

    /// Concurrency error
    #[error("Concurrency error: {0}")]
    Concurrency(String),

    /// Integration error with RAN core
    #[error("RAN integration error: {0}")]
    RanIntegration(#[from] ran_core::RanError),

    /// FANN library error
    #[error("FANN error: {0}")]
    FannError(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Generic error
    #[error("Neural network error: {0}")]
    Generic(String),
}

impl NeuralError {
    /// Create a network creation error
    pub fn network_creation<S: Into<String>>(message: S) -> Self {
        NeuralError::NetworkCreation(message.into())
    }

    /// Create a model loading error
    pub fn model_loading<S: Into<String>>(message: S) -> Self {
        NeuralError::ModelLoading(message.into())
    }

    /// Create an inference error
    pub fn inference_error<S: Into<String>>(message: S) -> Self {
        NeuralError::InferenceError(message.into())
    }

    /// Create a training error
    pub fn training_error<S: Into<String>>(message: S) -> Self {
        NeuralError::TrainingError(message.into())
    }

    /// Create an invalid input error
    pub fn invalid_input<S: Into<String>>(message: S) -> Self {
        NeuralError::InvalidInput(message.into())
    }

    /// Create a feature extraction error
    pub fn feature_extraction<S: Into<String>>(message: S) -> Self {
        NeuralError::FeatureExtraction(message.into())
    }

    /// Create a configuration error
    pub fn configuration<S: Into<String>>(message: S) -> Self {
        NeuralError::Configuration(message.into())
    }

    /// Get error category
    pub fn category(&self) -> ErrorCategory {
        match self {
            NeuralError::NetworkCreation(_) => ErrorCategory::Creation,
            NeuralError::ModelLoading(_) | NeuralError::ModelSaving(_) => ErrorCategory::Persistence,
            NeuralError::InferenceError(_) => ErrorCategory::Inference,
            NeuralError::TrainingError(_) => ErrorCategory::Training,
            NeuralError::InvalidInput(_) | NeuralError::DataValidation(_) => ErrorCategory::Validation,
            NeuralError::FeatureExtraction(_) => ErrorCategory::FeatureExtraction,
            NeuralError::ModelNotFound(_) => ErrorCategory::ModelManagement,
            NeuralError::Configuration(_) => ErrorCategory::Configuration,
            NeuralError::SerializationError(_) => ErrorCategory::Serialization,
            NeuralError::Resource(_) => ErrorCategory::Resource,
            NeuralError::Timeout(_) => ErrorCategory::Timeout,
            NeuralError::Concurrency(_) => ErrorCategory::Concurrency,
            NeuralError::RanIntegration(_) => ErrorCategory::Integration,
            NeuralError::FannError(_) => ErrorCategory::Library,
            NeuralError::Io(_) => ErrorCategory::IO,
            NeuralError::Generic(_) => ErrorCategory::Generic,
        }
    }

    /// Check if the error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            NeuralError::NetworkCreation(_) => false,
            NeuralError::ModelLoading(_) => false,
            NeuralError::ModelSaving(_) => true,
            NeuralError::InferenceError(_) => true,
            NeuralError::TrainingError(_) => true,
            NeuralError::InvalidInput(_) => false,
            NeuralError::FeatureExtraction(_) => true,
            NeuralError::ModelNotFound(_) => false,
            NeuralError::Configuration(_) => false,
            NeuralError::SerializationError(_) => false,
            NeuralError::DataValidation(_) => false,
            NeuralError::Resource(_) => true,
            NeuralError::Timeout(_) => true,
            NeuralError::Concurrency(_) => true,
            NeuralError::RanIntegration(_) => true,
            NeuralError::FannError(_) => true,
            NeuralError::Io(_) => true,
            NeuralError::Generic(_) => false,
        }
    }

    /// Get error severity
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            NeuralError::NetworkCreation(_) | NeuralError::ModelNotFound(_) |
            NeuralError::Configuration(_) => ErrorSeverity::Critical,
            
            NeuralError::ModelLoading(_) | NeuralError::TrainingError(_) |
            NeuralError::Resource(_) => ErrorSeverity::High,
            
            NeuralError::InferenceError(_) | NeuralError::FeatureExtraction(_) |
            NeuralError::Timeout(_) | NeuralError::Concurrency(_) => ErrorSeverity::Medium,
            
            NeuralError::ModelSaving(_) | NeuralError::SerializationError(_) |
            NeuralError::Io(_) => ErrorSeverity::Low,
            
            _ => ErrorSeverity::Medium,
        }
    }
}

/// Error categories for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Network creation errors
    Creation,
    /// Model persistence errors
    Persistence,
    /// Inference errors
    Inference,
    /// Training errors
    Training,
    /// Input validation errors
    Validation,
    /// Feature extraction errors
    FeatureExtraction,
    /// Model management errors
    ModelManagement,
    /// Configuration errors
    Configuration,
    /// Serialization errors
    Serialization,
    /// Resource errors
    Resource,
    /// Timeout errors
    Timeout,
    /// Concurrency errors
    Concurrency,
    /// Integration errors
    Integration,
    /// Library errors
    Library,
    /// IO errors
    IO,
    /// Generic errors
    Generic,
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Low severity - operation can continue
    Low,
    /// Medium severity - may affect performance
    Medium,
    /// High severity - requires attention
    High,
    /// Critical severity - system may be unusable
    Critical,
}

impl std::fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorSeverity::Low => write!(f, "LOW"),
            ErrorSeverity::Medium => write!(f, "MEDIUM"),
            ErrorSeverity::High => write!(f, "HIGH"),
            ErrorSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Result type for neural network operations
pub type NeuralResult<T> = Result<T, NeuralError>;

/// Error context for better error reporting
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Operation that failed
    pub operation: String,
    /// Model type involved
    pub model_type: Option<String>,
    /// Additional context
    pub context: std::collections::HashMap<String, String>,
    /// Timestamp when error occurred
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new<S: Into<String>>(operation: S) -> Self {
        Self {
            operation: operation.into(),
            model_type: None,
            context: std::collections::HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Add model type information
    pub fn with_model_type<S: Into<String>>(mut self, model_type: S) -> Self {
        self.model_type = Some(model_type.into());
        self
    }

    /// Add context information
    pub fn with_context<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        self.context.insert(key.into(), value.into());
        self
    }

    /// Convert to formatted string
    pub fn to_string(&self) -> String {
        let mut parts = vec![format!("Operation: {}", self.operation)];
        
        if let Some(ref model_type) = self.model_type {
            parts.push(format!("Model Type: {}", model_type));
        }
        
        if !self.context.is_empty() {
            let context_str: Vec<String> = self.context
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            parts.push(format!("Context: {}", context_str.join(", ")));
        }
        
        parts.push(format!("Timestamp: {}", self.timestamp.format("%Y-%m-%d %H:%M:%S UTC")));
        
        parts.join(", ")
    }
}

/// Specialized error types for different operations
pub mod specialized {
    use super::*;

    /// Training-specific errors
    #[derive(Error, Debug)]
    pub enum TrainingError {
        #[error("Insufficient training data: {0}")]
        InsufficientData(String),
        
        #[error("Invalid training configuration: {0}")]
        InvalidConfig(String),
        
        #[error("Training convergence failed: {0}")]
        ConvergenceFailed(String),
        
        #[error("Validation failed: {0}")]
        ValidationFailed(String),
        
        #[error("Early stopping triggered: {0}")]
        EarlyStopping(String),
    }

    /// Inference-specific errors
    #[derive(Error, Debug)]
    pub enum InferenceError {
        #[error("Model not loaded")]
        ModelNotLoaded,
        
        #[error("Input shape mismatch: expected {expected}, got {actual}")]
        InputShapeMismatch { expected: usize, actual: usize },
        
        #[error("Inference timeout after {timeout_ms}ms")]
        Timeout { timeout_ms: u64 },
        
        #[error("GPU inference failed: {0}")]
        GpuError(String),
        
        #[error("Batch size too large: {size}")]
        BatchSizeTooLarge { size: usize },
    }

    /// Feature extraction errors
    #[derive(Error, Debug)]
    pub enum FeatureError {
        #[error("Missing required feature: {0}")]
        MissingFeature(String),
        
        #[error("Invalid feature value: {feature} = {value}")]
        InvalidValue { feature: String, value: String },
        
        #[error("Feature scaling failed: {0}")]
        ScalingFailed(String),
        
        #[error("Feature dimension mismatch: expected {expected}, got {actual}")]
        DimensionMismatch { expected: usize, actual: usize },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = NeuralError::network_creation("Test error");
        assert_eq!(error.category(), ErrorCategory::Creation);
        assert!(!error.is_recoverable());
        assert_eq!(error.severity(), ErrorSeverity::Critical);
    }

    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("test_operation")
            .with_model_type("ThroughputPredictor")
            .with_context("input_size", "10")
            .with_context("output_size", "1");
        
        let context_str = context.to_string();
        assert!(context_str.contains("Operation: test_operation"));
        assert!(context_str.contains("Model Type: ThroughputPredictor"));
        assert!(context_str.contains("input_size=10"));
    }

    #[test]
    fn test_error_severity_ordering() {
        assert!(ErrorSeverity::Critical > ErrorSeverity::High);
        assert!(ErrorSeverity::High > ErrorSeverity::Medium);
        assert!(ErrorSeverity::Medium > ErrorSeverity::Low);
    }

    #[test]
    fn test_specialized_errors() {
        let training_error = specialized::TrainingError::InsufficientData("Not enough data".to_string());
        assert!(training_error.to_string().contains("Insufficient training data"));
        
        let inference_error = specialized::InferenceError::InputShapeMismatch { expected: 10, actual: 5 };
        assert!(inference_error.to_string().contains("Input shape mismatch"));
        
        let feature_error = specialized::FeatureError::MissingFeature("cell_load".to_string());
        assert!(feature_error.to_string().contains("Missing required feature"));
    }
}