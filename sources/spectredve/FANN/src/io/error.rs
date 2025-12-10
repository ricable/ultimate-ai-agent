//! Error types for I/O operations

use std::fmt;
use std::io;

/// Result type for I/O operations
pub type IoResult<T> = Result<T, IoError>;

/// Error types for I/O operations
#[derive(Debug)]
pub enum IoError {
    /// I/O error (file not found, permission denied, etc.)
    Io(io::Error),
    /// Invalid file format
    InvalidFileFormat(String),
    /// Parse error
    ParseError(String),
    /// Serialization error
    SerializationError(String),
    /// Decompression error
    CompressionError(String),
    /// Invalid network structure
    InvalidNetwork(String),
    /// Invalid training data
    InvalidTrainingData(String),
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IoError::Io(err) => write!(f, "I/O error: {err}"),
            IoError::InvalidFileFormat(msg) => write!(f, "Invalid file format: {msg}"),
            IoError::ParseError(msg) => write!(f, "Parse error: {msg}"),
            IoError::SerializationError(msg) => write!(f, "Serialization error: {msg}"),
            IoError::CompressionError(msg) => write!(f, "Compression error: {msg}"),
            IoError::InvalidNetwork(msg) => write!(f, "Invalid network: {msg}"),
            IoError::InvalidTrainingData(msg) => write!(f, "Invalid training data: {msg}"),
        }
    }
}

impl std::error::Error for IoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            IoError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for IoError {
    fn from(err: io::Error) -> Self {
        IoError::Io(err)
    }
}

#[cfg(feature = "serde")]
impl From<serde_json::Error> for IoError {
    fn from(err: serde_json::Error) -> Self {
        IoError::SerializationError(err.to_string())
    }
}

#[cfg(feature = "binary")]
impl From<bincode::Error> for IoError {
    fn from(err: bincode::Error) -> Self {
        IoError::SerializationError(format!("Bincode error: {err}"))
    }
}

impl From<std::num::ParseFloatError> for IoError {
    fn from(err: std::num::ParseFloatError) -> Self {
        IoError::ParseError(format!("Float parse error: {err}"))
    }
}

impl From<std::num::ParseIntError> for IoError {
    fn from(err: std::num::ParseIntError) -> Self {
        IoError::ParseError(format!("Integer parse error: {err}"))
    }
}
