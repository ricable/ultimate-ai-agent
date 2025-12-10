//! Error types for RAN forecasting

use std::fmt;
use thiserror::Error;

/// Result type for forecasting operations
pub type ForecastResult<T> = Result<T, ForecastError>;

/// Error types for RAN forecasting operations
#[derive(Error, Debug, Clone)]
pub enum ForecastError {
    /// Data-related errors
    #[error("Data error: {0}")]
    DataError(String),

    /// Model-related errors
    #[error("Model error: {0}")]
    ModelError(String),

    /// Model not fitted error
    #[error("Model has not been fitted yet")]
    ModelNotFitted,

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Training errors
    #[error("Training error: {0}")]
    TrainingError(String),

    /// Prediction errors
    #[error("Prediction error: {0}")]
    PredictionError(String),

    /// Validation errors
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// IO errors
    #[error("IO error: {0}")]
    IoError(String),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Integration errors with neuro-divergent
    #[error("Integration error: {0}")]
    IntegrationError(String),

    /// Math/computation errors
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Resource errors (memory, disk, etc.)
    #[error("Resource error: {0}")]
    ResourceError(String),

    /// Timeout errors
    #[error("Operation timed out: {0}")]
    TimeoutError(String),

    /// Concurrency errors
    #[error("Concurrency error: {0}")]
    ConcurrencyError(String),

    /// Feature extraction errors
    #[error("Feature extraction error: {0}")]
    FeatureExtractionError(String),

    /// Adapter errors
    #[error("Adapter error: {0}")]
    AdapterError(String),

    /// Invalid parameter errors
    #[error("Invalid parameter '{parameter}': {message}")]
    InvalidParameter {
        parameter: String,
        message: String,
    },

    /// Range errors
    #[error("Value out of range: {value} not in [{min}, {max}]")]
    OutOfRange {
        value: f64,
        min: f64,
        max: f64,
    },

    /// Dimension mismatch errors
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },

    /// Incompatible data types
    #[error("Incompatible data types: {message}")]
    IncompatibleTypes {
        message: String,
    },

    /// Multiple errors combined
    #[error("Multiple errors: {errors:?}")]
    MultipleErrors {
        errors: Vec<ForecastError>,
    },
}

impl ForecastError {
    /// Create a data error
    pub fn data_error<S: Into<String>>(message: S) -> Self {
        Self::DataError(message.into())
    }

    /// Create a model error
    pub fn model_error<S: Into<String>>(message: S) -> Self {
        Self::ModelError(message.into())
    }

    /// Create a configuration error
    pub fn config_error<S: Into<String>>(message: S) -> Self {
        Self::Configuration(message.into())
    }

    /// Create a training error
    pub fn training_error<S: Into<String>>(message: S) -> Self {
        Self::TrainingError(message.into())
    }

    /// Create a prediction error
    pub fn prediction_error<S: Into<String>>(message: S) -> Self {
        Self::PredictionError(message.into())
    }

    /// Create a validation error
    pub fn validation_error<S: Into<String>>(message: S) -> Self {
        Self::ValidationError(message.into())
    }

    /// Create an integration error
    pub fn integration_error<S: Into<String>>(message: S) -> Self {
        Self::IntegrationError(message.into())
    }

    /// Create a computation error
    pub fn computation_error<S: Into<String>>(message: S) -> Self {
        Self::ComputationError(message.into())
    }

    /// Create an invalid parameter error
    pub fn invalid_parameter<S: Into<String>>(parameter: S, message: S) -> Self {
        Self::InvalidParameter {
            parameter: parameter.into(),
            message: message.into(),
        }
    }

    /// Create an out of range error
    pub fn out_of_range(value: f64, min: f64, max: f64) -> Self {
        Self::OutOfRange { value, min, max }
    }

    /// Create a dimension mismatch error
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }

    /// Create an incompatible types error
    pub fn incompatible_types<S: Into<String>>(message: S) -> Self {
        Self::IncompatibleTypes {
            message: message.into(),
        }
    }

    /// Combine multiple errors
    pub fn multiple_errors(errors: Vec<ForecastError>) -> Self {
        Self::MultipleErrors { errors }
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::DataError(_) => false,
            Self::ModelError(_) => false,
            Self::ModelNotFitted => true,
            Self::Configuration(_) => false,
            Self::TrainingError(_) => true,
            Self::PredictionError(_) => true,
            Self::ValidationError(_) => false,
            Self::IoError(_) => true,
            Self::SerializationError(_) => false,
            Self::IntegrationError(_) => true,
            Self::ComputationError(_) => true,
            Self::ResourceError(_) => true,
            Self::TimeoutError(_) => true,
            Self::ConcurrencyError(_) => true,
            Self::FeatureExtractionError(_) => true,
            Self::AdapterError(_) => true,
            Self::InvalidParameter { .. } => false,
            Self::OutOfRange { .. } => false,
            Self::DimensionMismatch { .. } => false,
            Self::IncompatibleTypes { .. } => false,
            Self::MultipleErrors { errors } => errors.iter().any(|e| e.is_recoverable()),
        }
    }

    /// Get error category
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::DataError(_) => ErrorCategory::Data,
            Self::ModelError(_) | Self::ModelNotFitted => ErrorCategory::Model,
            Self::Configuration(_) => ErrorCategory::Configuration,
            Self::TrainingError(_) => ErrorCategory::Training,
            Self::PredictionError(_) => ErrorCategory::Prediction,
            Self::ValidationError(_) => ErrorCategory::Validation,
            Self::IoError(_) => ErrorCategory::IO,
            Self::SerializationError(_) => ErrorCategory::Serialization,
            Self::IntegrationError(_) => ErrorCategory::Integration,
            Self::ComputationError(_) => ErrorCategory::Computation,
            Self::ResourceError(_) => ErrorCategory::Resource,
            Self::TimeoutError(_) => ErrorCategory::Timeout,
            Self::ConcurrencyError(_) => ErrorCategory::Concurrency,
            Self::FeatureExtractionError(_) => ErrorCategory::FeatureExtraction,
            Self::AdapterError(_) => ErrorCategory::Adapter,
            Self::InvalidParameter { .. } => ErrorCategory::Parameter,
            Self::OutOfRange { .. } => ErrorCategory::Parameter,
            Self::DimensionMismatch { .. } => ErrorCategory::Parameter,
            Self::IncompatibleTypes { .. } => ErrorCategory::Parameter,
            Self::MultipleErrors { .. } => ErrorCategory::Multiple,
        }
    }

    /// Get error severity
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::DataError(_) => ErrorSeverity::High,
            Self::ModelError(_) => ErrorSeverity::High,
            Self::ModelNotFitted => ErrorSeverity::Medium,
            Self::Configuration(_) => ErrorSeverity::High,
            Self::TrainingError(_) => ErrorSeverity::Medium,
            Self::PredictionError(_) => ErrorSeverity::Medium,
            Self::ValidationError(_) => ErrorSeverity::High,
            Self::IoError(_) => ErrorSeverity::Medium,
            Self::SerializationError(_) => ErrorSeverity::Low,
            Self::IntegrationError(_) => ErrorSeverity::Medium,
            Self::ComputationError(_) => ErrorSeverity::Medium,
            Self::ResourceError(_) => ErrorSeverity::High,
            Self::TimeoutError(_) => ErrorSeverity::Low,
            Self::ConcurrencyError(_) => ErrorSeverity::Medium,
            Self::FeatureExtractionError(_) => ErrorSeverity::Medium,
            Self::AdapterError(_) => ErrorSeverity::Medium,
            Self::InvalidParameter { .. } => ErrorSeverity::High,
            Self::OutOfRange { .. } => ErrorSeverity::Medium,
            Self::DimensionMismatch { .. } => ErrorSeverity::High,
            Self::IncompatibleTypes { .. } => ErrorSeverity::High,
            Self::MultipleErrors { errors } => {
                errors.iter()
                    .map(|e| e.severity())
                    .max()
                    .unwrap_or(ErrorSeverity::Low)
            }
        }
    }
}

/// Error category classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorCategory {
    Data,
    Model,
    Configuration,
    Training,
    Prediction,
    Validation,
    IO,
    Serialization,
    Integration,
    Computation,
    Resource,
    Timeout,
    Concurrency,
    FeatureExtraction,
    Adapter,
    Parameter,
    Multiple,
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCategory::Data => write!(f, "Data"),
            ErrorCategory::Model => write!(f, "Model"),
            ErrorCategory::Configuration => write!(f, "Configuration"),
            ErrorCategory::Training => write!(f, "Training"),
            ErrorCategory::Prediction => write!(f, "Prediction"),
            ErrorCategory::Validation => write!(f, "Validation"),
            ErrorCategory::IO => write!(f, "IO"),
            ErrorCategory::Serialization => write!(f, "Serialization"),
            ErrorCategory::Integration => write!(f, "Integration"),
            ErrorCategory::Computation => write!(f, "Computation"),
            ErrorCategory::Resource => write!(f, "Resource"),
            ErrorCategory::Timeout => write!(f, "Timeout"),
            ErrorCategory::Concurrency => write!(f, "Concurrency"),
            ErrorCategory::FeatureExtraction => write!(f, "FeatureExtraction"),
            ErrorCategory::Adapter => write!(f, "Adapter"),
            ErrorCategory::Parameter => write!(f, "Parameter"),
            ErrorCategory::Multiple => write!(f, "Multiple"),
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Low => write!(f, "Low"),
            ErrorSeverity::Medium => write!(f, "Medium"),
            ErrorSeverity::High => write!(f, "High"),
            ErrorSeverity::Critical => write!(f, "Critical"),
        }
    }
}

// Conversion implementations for common error types
impl From<std::io::Error> for ForecastError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err.to_string())
    }
}

impl From<serde_json::Error> for ForecastError {
    fn from(err: serde_json::Error) -> Self {
        Self::SerializationError(err.to_string())
    }
}

impl From<anyhow::Error> for ForecastError {
    fn from(err: anyhow::Error) -> Self {
        Self::IntegrationError(err.to_string())
    }
}

// Integration with neuro-divergent errors
impl From<neuro_divergent_core::error::NeuroDivergentError> for ForecastError {
    fn from(err: neuro_divergent_core::error::NeuroDivergentError) -> Self {
        Self::IntegrationError(format!("Neuro-divergent error: {}", err))
    }
}

impl From<ran_core::RanError> for ForecastError {
    fn from(err: ran_core::RanError) -> Self {
        Self::DataError(format!("RAN core error: {}", err))
    }
}

/// Error context for better debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Error location (function, file, line)
    pub location: String,
    /// Additional context information
    pub context: std::collections::HashMap<String, String>,
    /// Timestamp when error occurred
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ErrorContext {
    /// Create new error context
    pub fn new<S: Into<String>>(location: S) -> Self {
        Self {
            location: location.into(),
            context: std::collections::HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
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

    /// Add multiple context entries
    pub fn with_contexts<I, K, V>(mut self, contexts: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        for (key, value) in contexts {
            self.context.insert(key.into(), value.into());
        }
        self
    }
}

/// Extended error with context
#[derive(Debug, Clone)]
pub struct ContextualError {
    /// The underlying error
    pub error: ForecastError,
    /// Error context
    pub context: ErrorContext,
}

impl fmt::Display for ContextualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (at {})", self.error, self.context.location)?;
        if !self.context.context.is_empty() {
            write!(f, " - Context: {:?}", self.context.context)?;
        }
        Ok(())
    }
}

impl std::error::Error for ContextualError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Macro for creating contextual errors
#[macro_export]
macro_rules! forecast_error {
    ($error:expr, $location:expr) => {
        ContextualError {
            error: $error,
            context: ErrorContext::new($location),
        }
    };
    ($error:expr, $location:expr, $($key:expr => $value:expr),*) => {
        ContextualError {
            error: $error,
            context: ErrorContext::new($location)
                $(.with_context($key, $value))*,
        }
    };
}

/// Result type with contextual errors
pub type ContextualResult<T> = Result<T, ContextualError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forecast_error_creation() {
        let error = ForecastError::data_error("test message");
        assert!(matches!(error, ForecastError::DataError(_)));
        assert_eq!(error.to_string(), "Data error: test message");
    }

    #[test]
    fn test_forecast_error_properties() {
        let error = ForecastError::TrainingError("training failed".to_string());
        assert!(error.is_recoverable());
        assert_eq!(error.category(), ErrorCategory::Training);
        assert_eq!(error.severity(), ErrorSeverity::Medium);
    }

    #[test]
    fn test_error_category_display() {
        assert_eq!(format!("{}", ErrorCategory::Data), "Data");
        assert_eq!(format!("{}", ErrorCategory::Model), "Model");
        assert_eq!(format!("{}", ErrorCategory::Training), "Training");
    }

    #[test]
    fn test_error_severity_ordering() {
        assert!(ErrorSeverity::Low < ErrorSeverity::Medium);
        assert!(ErrorSeverity::Medium < ErrorSeverity::High);
        assert!(ErrorSeverity::High < ErrorSeverity::Critical);
    }

    #[test]
    fn test_multiple_errors_severity() {
        let errors = vec![
            ForecastError::TrainingError("low".to_string()),
            ForecastError::Configuration("high".to_string()),
        ];
        let multi_error = ForecastError::multiple_errors(errors);
        assert_eq!(multi_error.severity(), ErrorSeverity::High);
    }

    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("test_function")
            .with_context("key1", "value1")
            .with_context("key2", "value2");
        
        assert_eq!(context.location, "test_function");
        assert_eq!(context.context.len(), 2);
        assert_eq!(context.context.get("key1"), Some(&"value1".to_string()));
    }

    #[test]
    fn test_contextual_error() {
        let error = ForecastError::ModelNotFitted;
        let context = ErrorContext::new("prediction_function");
        let contextual = ContextualError { error, context };
        
        assert!(contextual.to_string().contains("Model has not been fitted yet"));
        assert!(contextual.to_string().contains("prediction_function"));
    }

    #[test]
    fn test_error_conversions() {
        // Test IO error conversion
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let forecast_error: ForecastError = io_error.into();
        assert!(matches!(forecast_error, ForecastError::IoError(_)));

        // Test JSON error conversion
        let json_error = serde_json::Error::syntax(serde_json::error::ErrorCode::TrailingComma, 0, 0);
        let forecast_error: ForecastError = json_error.into();
        assert!(matches!(forecast_error, ForecastError::SerializationError(_)));
    }

    #[test]
    fn test_parameter_errors() {
        let error = ForecastError::invalid_parameter("learning_rate", "must be positive");
        assert!(matches!(error, ForecastError::InvalidParameter { .. }));
        assert!(!error.is_recoverable());
        assert_eq!(error.severity(), ErrorSeverity::High);

        let range_error = ForecastError::out_of_range(5.0, 0.0, 1.0);
        assert!(matches!(range_error, ForecastError::OutOfRange { .. }));
        assert_eq!(range_error.severity(), ErrorSeverity::Medium);

        let dim_error = ForecastError::dimension_mismatch(10, 5);
        assert!(matches!(dim_error, ForecastError::DimensionMismatch { .. }));
        assert_eq!(dim_error.severity(), ErrorSeverity::High);
    }
}