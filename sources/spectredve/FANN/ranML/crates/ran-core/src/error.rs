//! Error types and handling for RAN operations

use std::fmt;
use thiserror::Error;

/// Main error type for RAN operations
#[derive(Error, Debug)]
pub enum RanError {
    /// Network configuration error
    #[error("Network configuration error: {message}")]
    NetworkConfig { message: String },

    /// Invalid network element
    #[error("Invalid network element: {element_type} with id {element_id}")]
    InvalidNetworkElement {
        element_type: String,
        element_id: String,
    },

    /// Measurement error
    #[error("Measurement error: {message}")]
    Measurement { message: String },

    /// Optimization error
    #[error("Optimization error: {message}")]
    Optimization { message: String },

    /// Resource allocation error
    #[error("Resource allocation error: {resource_type} - {message}")]
    ResourceAllocation {
        resource_type: String,
        message: String,
    },

    /// Performance metric error
    #[error("Performance metric error: {metric_type} - {message}")]
    PerformanceMetric {
        metric_type: String,
        message: String,
    },

    /// Time series data error
    #[error("Time series data error: {message}")]
    TimeSeriesData { message: String },

    /// Model configuration error
    #[error("Model configuration error: {message}")]
    ModelConfig { message: String },

    /// Validation error
    #[error("Validation error: {field} - {message}")]
    Validation { field: String, message: String },

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// UUID parsing error
    #[error("UUID parsing error: {0}")]
    UuidParsing(#[from] uuid::Error),

    /// Generic error with context
    #[error("RAN error: {0}")]
    Generic(#[from] anyhow::Error),
}

impl RanError {
    /// Create a new network configuration error
    pub fn network_config<S: Into<String>>(message: S) -> Self {
        RanError::NetworkConfig {
            message: message.into(),
        }
    }

    /// Create a new invalid network element error
    pub fn invalid_network_element<S: Into<String>>(element_type: S, element_id: S) -> Self {
        RanError::InvalidNetworkElement {
            element_type: element_type.into(),
            element_id: element_id.into(),
        }
    }

    /// Create a new measurement error
    pub fn measurement<S: Into<String>>(message: S) -> Self {
        RanError::Measurement {
            message: message.into(),
        }
    }

    /// Create a new optimization error
    pub fn optimization<S: Into<String>>(message: S) -> Self {
        RanError::Optimization {
            message: message.into(),
        }
    }

    /// Create a new resource allocation error
    pub fn resource_allocation<S: Into<String>>(resource_type: S, message: S) -> Self {
        RanError::ResourceAllocation {
            resource_type: resource_type.into(),
            message: message.into(),
        }
    }

    /// Create a new performance metric error
    pub fn performance_metric<S: Into<String>>(metric_type: S, message: S) -> Self {
        RanError::PerformanceMetric {
            metric_type: metric_type.into(),
            message: message.into(),
        }
    }

    /// Create a new time series data error
    pub fn time_series_data<S: Into<String>>(message: S) -> Self {
        RanError::TimeSeriesData {
            message: message.into(),
        }
    }

    /// Create a new model configuration error
    pub fn model_config<S: Into<String>>(message: S) -> Self {
        RanError::ModelConfig {
            message: message.into(),
        }
    }

    /// Create a new validation error
    pub fn validation<S: Into<String>>(field: S, message: S) -> Self {
        RanError::Validation {
            field: field.into(),
            message: message.into(),
        }
    }

    /// Get the error category
    pub fn category(&self) -> ErrorCategory {
        match self {
            RanError::NetworkConfig { .. } => ErrorCategory::Configuration,
            RanError::InvalidNetworkElement { .. } => ErrorCategory::Validation,
            RanError::Measurement { .. } => ErrorCategory::Measurement,
            RanError::Optimization { .. } => ErrorCategory::Optimization,
            RanError::ResourceAllocation { .. } => ErrorCategory::Resource,
            RanError::PerformanceMetric { .. } => ErrorCategory::Metric,
            RanError::TimeSeriesData { .. } => ErrorCategory::Data,
            RanError::ModelConfig { .. } => ErrorCategory::Configuration,
            RanError::Validation { .. } => ErrorCategory::Validation,
            RanError::Io(_) => ErrorCategory::IO,
            RanError::Serialization(_) => ErrorCategory::Serialization,
            RanError::UuidParsing(_) => ErrorCategory::Parsing,
            RanError::Generic(_) => ErrorCategory::Generic,
        }
    }

    /// Check if the error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            RanError::NetworkConfig { .. } => false,
            RanError::InvalidNetworkElement { .. } => false,
            RanError::Measurement { .. } => true,
            RanError::Optimization { .. } => true,
            RanError::ResourceAllocation { .. } => true,
            RanError::PerformanceMetric { .. } => true,
            RanError::TimeSeriesData { .. } => true,
            RanError::ModelConfig { .. } => false,
            RanError::Validation { .. } => false,
            RanError::Io(_) => true,
            RanError::Serialization(_) => false,
            RanError::UuidParsing(_) => false,
            RanError::Generic(_) => false,
        }
    }
}

/// Error category for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Configuration-related errors
    Configuration,
    /// Validation errors
    Validation,
    /// Measurement errors
    Measurement,
    /// Optimization errors
    Optimization,
    /// Resource management errors
    Resource,
    /// Performance metric errors
    Metric,
    /// Data-related errors
    Data,
    /// IO errors
    IO,
    /// Serialization errors
    Serialization,
    /// Parsing errors
    Parsing,
    /// Generic errors
    Generic,
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCategory::Configuration => write!(f, "Configuration"),
            ErrorCategory::Validation => write!(f, "Validation"),
            ErrorCategory::Measurement => write!(f, "Measurement"),
            ErrorCategory::Optimization => write!(f, "Optimization"),
            ErrorCategory::Resource => write!(f, "Resource"),
            ErrorCategory::Metric => write!(f, "Metric"),
            ErrorCategory::Data => write!(f, "Data"),
            ErrorCategory::IO => write!(f, "IO"),
            ErrorCategory::Serialization => write!(f, "Serialization"),
            ErrorCategory::Parsing => write!(f, "Parsing"),
            ErrorCategory::Generic => write!(f, "Generic"),
        }
    }
}

/// Result type for RAN operations
pub type RanResult<T> = Result<T, RanError>;

/// Error context for better error reporting
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Operation that failed
    pub operation: String,
    /// Component that failed
    pub component: String,
    /// Additional context information
    pub context: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new(operation: String, component: String) -> Self {
        Self {
            operation,
            component,
            context: std::collections::HashMap::new(),
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

    /// Convert to a formatted string
    pub fn to_string(&self) -> String {
        let mut result = format!("Operation: {}, Component: {}", self.operation, self.component);
        
        if !self.context.is_empty() {
            result.push_str(", Context: ");
            let context_str: Vec<String> = self.context
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            result.push_str(&context_str.join(", "));
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = RanError::network_config("Invalid configuration");
        assert_eq!(error.category(), ErrorCategory::Configuration);
        assert!(!error.is_recoverable());
    }

    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("measure".to_string(), "gNodeB".to_string())
            .with_context("cell_id", "12345")
            .with_context("metric", "throughput");
        
        let context_str = context.to_string();
        assert!(context_str.contains("Operation: measure"));
        assert!(context_str.contains("Component: gNodeB"));
        assert!(context_str.contains("cell_id=12345"));
        assert!(context_str.contains("metric=throughput"));
    }

    #[test]
    fn test_error_recoverability() {
        let measurement_error = RanError::measurement("Sensor unavailable");
        assert!(measurement_error.is_recoverable());
        
        let config_error = RanError::network_config("Invalid parameter");
        assert!(!config_error.is_recoverable());
    }
}