//! # ReasoningBank Core
//!
//! Core reasoning engine with pattern matching and similarity scoring.
//!
//! This crate provides the foundational types and algorithms for storing,
//! retrieving, and analyzing reasoning patterns with adaptive learning capabilities.

pub mod engine;
pub mod pattern;
pub mod similarity;

pub use engine::{ReasoningEngine, EngineConfig};
pub use pattern::{Pattern, PatternMetadata, TaskOutcome};
pub use similarity::{SimilarityMetrics, VectorEmbedding};

use thiserror::Error;

/// Result type for reasoning operations
pub type Result<T> = std::result::Result<T, ReasoningError>;

/// Errors that can occur in reasoning operations
#[derive(Error, Debug)]
pub enum ReasoningError {
    #[error("Pattern not found: {0}")]
    PatternNotFound(String),

    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    InvalidEmbeddingDimension { expected: usize, actual: usize },

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Internal error: {0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ReasoningError::PatternNotFound("test-pattern".to_string());
        assert!(err.to_string().contains("test-pattern"));
    }
}
