//! Comprehensive error handling for the neuro-divergent library.
//!
//! This module provides detailed error types that cover all aspects of neural forecasting
//! operations, from data validation to model training and prediction errors.

use std::fmt;
use thiserror::Error;

/// Result type alias for neuro-divergent operations
pub type NeuroDivergentResult<T> = Result<T, NeuroDivergentError>;

/// Comprehensive error types for neuro-divergent operations
#[derive(Error, Debug)]
pub enum NeuroDivergentError {
    /// Configuration errors in model setup
    #[error("Model configuration error: {message}")]
    ConfigError {
        /// Error message describing the configuration issue
        message: String,
        /// Optional source error
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Data validation and processing errors
    #[error("Data validation error: {message}")]
    DataError {
        /// Error message describing the data issue
        message: String,
        /// Optional field name where the error occurred
        field: Option<String>,
        /// Optional source error
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Training-related errors
    #[error("Training error: {message}")]
    TrainingError {
        /// Error message describing the training issue
        message: String,
        /// Training epoch where the error occurred (if applicable)
        epoch: Option<usize>,
        /// Optional source error
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Prediction and inference errors
    #[error("Prediction error: {message}")]
    PredictionError {
        /// Error message describing the prediction issue
        message: String,
        /// Optional model name where the error occurred
        model_name: Option<String>,
        /// Optional source error
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Integration errors with ruv-FANN
    #[error("Network integration error: {0}")]
    NetworkError(#[from] NetworkIntegrationError),

    /// I/O errors (file operations, network, etc.)
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization and deserialization errors
    #[error("Serialization error: {message}")]
    SerializationError {
        /// Error message describing the serialization issue
        message: String,
        /// Optional format information
        format: Option<String>,
        /// Optional source error
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Memory allocation and management errors
    #[error("Memory error: {message}")]
    MemoryError {
        /// Error message describing the memory issue
        message: String,
        /// Optional memory usage information
        memory_usage: Option<usize>,
        /// Optional source error
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Compatibility errors between components
    #[error("Compatibility error: {message}")]
    CompatibilityError {
        /// Error message describing the compatibility issue
        message: String,
        /// Optional component names involved
        components: Option<Vec<String>>,
        /// Optional source error
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Mathematical computation errors
    #[error("Mathematical error: {message}")]
    MathError {
        /// Error message describing the mathematical issue
        message: String,
        /// Optional operation name
        operation: Option<String>,
        /// Optional source error
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Parallel processing errors
    #[error("Parallel processing error: {message}")]
    ParallelError {
        /// Error message describing the parallel processing issue
        message: String,
        /// Optional thread information
        thread_info: Option<String>,
        /// Optional source error
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Time series specific errors
    #[error("Time series error: {message}")]
    TimeSeriesError {
        /// Error message describing the time series issue
        message: String,
        /// Optional series identifier
        series_id: Option<String>,
        /// Optional timestamp information
        timestamp: Option<chrono::DateTime<chrono::Utc>>,
        /// Optional source error
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}

/// Specific errors related to network integration with ruv-FANN
#[derive(Error, Debug)]
pub enum NetworkIntegrationError {
    /// Network architecture mismatch
    #[error("Network architecture mismatch: expected {expected}, found {found}")]
    ArchitectureMismatch {
        /// Expected network architecture
        expected: String,
        /// Found network architecture
        found: String,
    },

    /// Network training algorithm error
    #[error("Training algorithm error: {message}")]
    TrainingAlgorithmError {
        /// Error message
        message: String,
        /// Algorithm name
        algorithm: Option<String>,
    },

    /// Network I/O error
    #[error("Network I/O error: {message}")]
    NetworkIoError {
        /// Error message
        message: String,
        /// File path if applicable
        path: Option<String>,
    },

    /// Network validation error
    #[error("Network validation error: {message}")]
    ValidationError {
        /// Error message
        message: String,
        /// Layer information if applicable
        layer: Option<usize>,
    },

    /// Network activation function error
    #[error("Activation function error: {message}")]
    ActivationError {
        /// Error message
        message: String,
        /// Function name
        function: Option<String>,
    },
}

/// Error builder for creating detailed error instances
pub struct ErrorBuilder {
    error_type: ErrorType,
    message: String,
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
    context: std::collections::HashMap<String, String>,
}

/// Internal error type enumeration for the builder
enum ErrorType {
    Config,
    Data,
    Training,
    Prediction,
    Memory,
    Compatibility,
    Math,
    Parallel,
    TimeSeries,
    Serialization,
}

impl ErrorBuilder {
    /// Create a new configuration error builder
    pub fn config<S: Into<String>>(message: S) -> Self {
        Self {
            error_type: ErrorType::Config,
            message: message.into(),
            source: None,
            context: std::collections::HashMap::new(),
        }
    }

    /// Create a new data error builder
    pub fn data<S: Into<String>>(message: S) -> Self {
        Self {
            error_type: ErrorType::Data,
            message: message.into(),
            source: None,
            context: std::collections::HashMap::new(),
        }
    }

    /// Create a new training error builder
    pub fn training<S: Into<String>>(message: S) -> Self {
        Self {
            error_type: ErrorType::Training,
            message: message.into(),
            source: None,
            context: std::collections::HashMap::new(),
        }
    }

    /// Create a new prediction error builder
    pub fn prediction<S: Into<String>>(message: S) -> Self {
        Self {
            error_type: ErrorType::Prediction,
            message: message.into(),
            source: None,
            context: std::collections::HashMap::new(),
        }
    }

    /// Create a new memory error builder
    pub fn memory<S: Into<String>>(message: S) -> Self {
        Self {
            error_type: ErrorType::Memory,
            message: message.into(),
            source: None,
            context: std::collections::HashMap::new(),
        }
    }

    /// Create a new time series error builder
    pub fn time_series<S: Into<String>>(message: S) -> Self {
        Self {
            error_type: ErrorType::TimeSeries,
            message: message.into(),
            source: None,
            context: std::collections::HashMap::new(),
        }
    }

    /// Add a source error
    pub fn source<E: std::error::Error + Send + Sync + 'static>(mut self, source: E) -> Self {
        self.source = Some(Box::new(source));
        self
    }

    /// Add context information
    pub fn context<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }

    /// Build the error
    pub fn build(self) -> NeuroDivergentError {
        match self.error_type {
            ErrorType::Config => NeuroDivergentError::ConfigError {
                message: self.message,
                source: self.source,
            },
            ErrorType::Data => NeuroDivergentError::DataError {
                message: self.message,
                field: self.context.get("field").cloned(),
                source: self.source,
            },
            ErrorType::Training => NeuroDivergentError::TrainingError {
                message: self.message,
                epoch: self.context.get("epoch").and_then(|s| s.parse().ok()),
                source: self.source,
            },
            ErrorType::Prediction => NeuroDivergentError::PredictionError {
                message: self.message,
                model_name: self.context.get("model_name").cloned(),
                source: self.source,
            },
            ErrorType::Memory => NeuroDivergentError::MemoryError {
                message: self.message,
                memory_usage: self.context.get("memory_usage").and_then(|s| s.parse().ok()),
                source: self.source,
            },
            ErrorType::Compatibility => NeuroDivergentError::CompatibilityError {
                message: self.message,
                components: self.context.get("components")
                    .map(|s| s.split(',').map(|s| s.trim().to_string()).collect()),
                source: self.source,
            },
            ErrorType::Math => NeuroDivergentError::MathError {
                message: self.message,
                operation: self.context.get("operation").cloned(),
                source: self.source,
            },
            ErrorType::Parallel => NeuroDivergentError::ParallelError {
                message: self.message,
                thread_info: self.context.get("thread_info").cloned(),
                source: self.source,
            },
            ErrorType::TimeSeries => NeuroDivergentError::TimeSeriesError {
                message: self.message,
                series_id: self.context.get("series_id").cloned(),
                timestamp: self.context.get("timestamp")
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                    .map(|dt| dt.with_timezone(&chrono::Utc)),
                source: self.source,
            },
            ErrorType::Serialization => NeuroDivergentError::SerializationError {
                message: self.message,
                format: self.context.get("format").cloned(),
                source: self.source,
            },
        }
    }
}

/// Convenience macros for error creation
#[macro_export]
macro_rules! config_error {
    ($msg:expr) => {
        $crate::error::ErrorBuilder::config($msg).build()
    };
    ($msg:expr, $($key:expr => $value:expr),+) => {
        {
            let mut builder = $crate::error::ErrorBuilder::config($msg);
            $(
                builder = builder.context($key, $value);
            )+
            builder.build()
        }
    };
}

/// Create a data error with the given message
#[macro_export]
macro_rules! data_error {
    ($msg:expr) => {
        $crate::error::ErrorBuilder::data($msg).build()
    };
    ($msg:expr, field = $field:expr) => {
        $crate::error::ErrorBuilder::data($msg).context("field", $field).build()
    };
    ($msg:expr, $($key:expr => $value:expr),+) => {
        {
            let mut builder = $crate::error::ErrorBuilder::data($msg);
            $(
                builder = builder.context($key, $value);
            )+
            builder.build()
        }
    };
}

/// Create a training error with the given message
#[macro_export]
macro_rules! training_error {
    ($msg:expr) => {
        $crate::error::ErrorBuilder::training($msg).build()
    };
    ($msg:expr, epoch = $epoch:expr) => {
        $crate::error::ErrorBuilder::training($msg).context("epoch", $epoch.to_string()).build()
    };
    ($msg:expr, $($key:expr => $value:expr),+) => {
        {
            let mut builder = $crate::error::ErrorBuilder::training($msg);
            $(
                builder = builder.context($key, $value);
            )+
            builder.build()
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_builder_config() {
        let error = ErrorBuilder::config("Test configuration error")
            .context("parameter", "learning_rate")
            .build();

        match error {
            NeuroDivergentError::ConfigError { message, .. } => {
                assert_eq!(message, "Test configuration error");
            }
            _ => panic!("Expected ConfigError"),
        }
    }

    #[test]
    fn test_error_builder_data() {
        let error = ErrorBuilder::data("Test data error")
            .context("field", "target_column")
            .build();

        match error {
            NeuroDivergentError::DataError { message, field, .. } => {
                assert_eq!(message, "Test data error");
                assert_eq!(field, Some("target_column".to_string()));
            }
            _ => panic!("Expected DataError"),
        }
    }

    #[test]
    fn test_error_macros() {
        let error = config_error!("Configuration problem");
        assert!(matches!(error, NeuroDivergentError::ConfigError { .. }));

        let error = data_error!("Data problem", field = "timestamp");
        match error {
            NeuroDivergentError::DataError { field, .. } => {
                assert_eq!(field, Some("timestamp".to_string()));
            }
            _ => panic!("Expected DataError"),
        }
    }

    #[test]
    fn test_network_integration_error() {
        let error = NetworkIntegrationError::ArchitectureMismatch {
            expected: "3-5-1".to_string(),
            found: "3-4-1".to_string(),
        };
        
        let error_string = error.to_string();
        assert!(error_string.contains("3-5-1"));
        assert!(error_string.contains("3-4-1"));
    }

    #[test]
    fn test_error_chaining() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let error = ErrorBuilder::data("Could not read data file")
            .source(io_error)
            .build();

        match error {
            NeuroDivergentError::DataError { source, .. } => {
                assert!(source.is_some());
            }
            _ => panic!("Expected DataError"),
        }
    }
}