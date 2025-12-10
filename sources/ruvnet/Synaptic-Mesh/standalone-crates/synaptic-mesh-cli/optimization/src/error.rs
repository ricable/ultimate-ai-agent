//! Error handling for the optimization suite

use thiserror::Error;
use wasm_bindgen::prelude::*;

/// Result type for optimization operations
pub type Result<T> = std::result::Result<T, OptimizationError>;

/// Comprehensive error types for optimization operations
#[derive(Error, Debug)]
pub enum OptimizationError {
    #[error("Neural network error: {0}")]
    Neural(String),
    
    #[error("WASM optimization error: {0}")]
    Wasm(String),
    
    #[error("Memory optimization error: {0}")]
    Memory(String),
    
    #[error("Performance monitoring error: {0}")]
    Performance(String),
    
    #[error("Browser compatibility error: {0}")]
    BrowserCompat(String),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("WASM binding error: {0}")]
    WasmBinding(String),
    
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Resource unavailable: {0}")]
    ResourceUnavailable(String),
    
    #[error("Timeout error: operation took too long")]
    Timeout,
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<OptimizationError> for JsValue {
    fn from(error: OptimizationError) -> Self {
        JsValue::from_str(&error.to_string())
    }
}

impl From<serde_wasm_bindgen::Error> for OptimizationError {
    fn from(error: serde_wasm_bindgen::Error) -> Self {
        OptimizationError::Serialization(serde_json::Error::custom(error.to_string()))
    }
}

impl From<kimi_fann_core::error::RuntimeError> for OptimizationError {
    fn from(error: kimi_fann_core::error::RuntimeError) -> Self {
        OptimizationError::Neural(error.to_string())
    }
}

/// Convert JavaScript errors to optimization errors
#[wasm_bindgen]
pub fn js_error_to_optimization_error(js_error: &JsValue) -> OptimizationError {
    if let Some(error_string) = js_error.as_string() {
        OptimizationError::WasmBinding(error_string)
    } else {
        OptimizationError::WasmBinding("Unknown JavaScript error".to_string())
    }
}

/// Error context for better debugging
pub struct ErrorContext {
    pub operation: String,
    pub module: String,
    pub details: Option<String>,
}

impl ErrorContext {
    pub fn new(module: &str, operation: &str) -> Self {
        Self {
            operation: operation.to_string(),
            module: module.to_string(),
            details: None,
        }
    }
    
    pub fn with_details(mut self, details: &str) -> Self {
        self.details = Some(details.to_string());
        self
    }
    
    pub fn to_error(&self, error: OptimizationError) -> OptimizationError {
        let context = if let Some(details) = &self.details {
            format!("[{}::{}] {} ({})", self.module, self.operation, error, details)
        } else {
            format!("[{}::{}] {}", self.module, self.operation, error)
        };
        
        match error {
            OptimizationError::Neural(_) => OptimizationError::Neural(context),
            OptimizationError::Wasm(_) => OptimizationError::Wasm(context),
            OptimizationError::Memory(_) => OptimizationError::Memory(context),
            OptimizationError::Performance(_) => OptimizationError::Performance(context),
            OptimizationError::BrowserCompat(_) => OptimizationError::BrowserCompat(context),
            _ => OptimizationError::Internal(context),
        }
    }
}

/// Macro for creating errors with context
#[macro_export]
macro_rules! optimization_error {
    ($kind:ident, $($arg:tt)*) => {
        $crate::error::OptimizationError::$kind(format!($($arg)*))
    };
}

/// Macro for wrapping results with context
#[macro_export]
macro_rules! with_context {
    ($result:expr, $module:expr, $operation:expr) => {
        $result.map_err(|e| {
            $crate::error::ErrorContext::new($module, $operation).to_error(e)
        })
    };
    ($result:expr, $module:expr, $operation:expr, $details:expr) => {
        $result.map_err(|e| {
            $crate::error::ErrorContext::new($module, $operation)
                .with_details($details)
                .to_error(e)
        })
    };
}