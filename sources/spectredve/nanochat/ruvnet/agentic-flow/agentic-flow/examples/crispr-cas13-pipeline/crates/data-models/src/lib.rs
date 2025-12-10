//! # Data Models
//!
//! Shared data structures for the CRISPR-Cas13 bioinformatics pipeline.
//! This crate provides common types used across all pipeline components.

pub mod error;
pub mod expression;
pub mod metadata;
pub mod sequencing;
pub mod targets;

pub use error::{DataModelError, Result};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_type_alias() {
        let ok_result: Result<i32> = Ok(42);
        assert!(ok_result.is_ok());
        assert_eq!(ok_result.unwrap(), 42);

        let err_result: Result<i32> = Err(DataModelError::ValidationError("test error".to_string()));
        assert!(err_result.is_err());
    }
}
