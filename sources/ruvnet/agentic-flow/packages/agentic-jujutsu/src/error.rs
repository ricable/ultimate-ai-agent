//! Error types for agentic-jujutsu

use thiserror::Error;

/// Result type alias for agentic-jujutsu operations
pub type Result<T> = std::result::Result<T, JJError>;

/// Error types for Jujutsu operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum JJError {
    /// jj command not found or not installed
    #[error("jj command not found. Please install Jujutsu: https://github.com/jj-vcs/jj")]
    JJNotFound,

    /// jj command execution failed
    #[error("jj command failed: {0}")]
    CommandFailed(String),

    /// Failed to parse jj output
    #[error("Failed to parse jj output: {0}")]
    ParseError(String),

    /// Operation not found in log
    #[error("Operation {0} not found")]
    OperationNotFound(String),

    /// Conflict resolution failed
    #[error("Conflict resolution failed: {0}")]
    ConflictResolutionFailed(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Unknown error
    #[error("Unknown error: {0}")]
    Unknown(String),

    /// MCP protocol error
    #[error("MCP error: {0}")]
    MCPError(String),

    /// Cryptographic operation error
    #[error("Crypto error: {0}")]
    CryptoError(String),
}

impl JJError {
    /// Get error message as string
    pub fn message(&self) -> String {
        self.to_string()
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            JJError::CommandFailed(_) | JJError::ConflictResolutionFailed(_)
        )
    }
}

impl From<std::io::Error> for JJError {
    fn from(err: std::io::Error) -> Self {
        JJError::IoError(err.to_string())
    }
}

impl From<serde_json::Error> for JJError {
    fn from(err: serde_json::Error) -> Self {
        JJError::SerializationError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = JJError::JJNotFound;
        assert!(err.to_string().contains("jj command not found"));
    }

    #[test]
    fn test_recoverable() {
        assert!(JJError::CommandFailed("test".into()).is_recoverable());
        assert!(!JJError::JJNotFound.is_recoverable());
    }
}
