//! Error types for QUIC transport

use thiserror::Error;

/// QUIC transport errors
#[derive(Error, Debug)]
pub enum QuicError {
    /// Connection error
    #[error("Connection error: {0}")]
    Connection(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Stream error
    #[error("Stream error: {0}")]
    Stream(String),

    /// Timeout error
    #[error("Operation timed out: {0}")]
    Timeout(String),

    /// Protocol error
    #[error("Protocol error: {0}")]
    Protocol(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// TLS error
    #[error("TLS error: {0}")]
    Tls(String),

    /// Connection pool error
    #[error("Connection pool error: {0}")]
    Pool(String),

    /// WASM error
    #[cfg(feature = "wasm")]
    #[error("WASM error: {0}")]
    Wasm(String),
}

impl QuicError {
    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            QuicError::Timeout(_) | QuicError::Pool(_)
        )
    }

    /// Get error category for logging
    pub fn category(&self) -> &'static str {
        match self {
            QuicError::Connection(_) => "connection",
            QuicError::Io(_) => "io",
            QuicError::ConfigError(_) => "config",
            QuicError::InvalidConfig(_) => "invalid_config",
            QuicError::Stream(_) => "stream",
            QuicError::Timeout(_) => "timeout",
            QuicError::Protocol(_) => "protocol",
            QuicError::Serialization(_) => "serialization",
            QuicError::Tls(_) => "tls",
            QuicError::Pool(_) => "pool",
            #[cfg(feature = "wasm")]
            QuicError::Wasm(_) => "wasm",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_recoverable() {
        let timeout_err = QuicError::Timeout("test".to_string());
        assert!(timeout_err.is_recoverable());

        let config_err = QuicError::ConfigError("test".to_string());
        assert!(!config_err.is_recoverable());
    }

    #[test]
    fn test_error_category() {
        let err = QuicError::Connection("test".to_string());
        assert_eq!(err.category(), "connection");
    }
}
