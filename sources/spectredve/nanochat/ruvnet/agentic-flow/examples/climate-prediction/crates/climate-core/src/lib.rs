//! # Climate Core
//!
//! Core types, traits, and error definitions for the climate prediction system.
//! This crate provides the foundational abstractions that all other crates depend on.

pub mod error;
pub mod types;
pub mod traits;

pub use error::{ClimateError, Result};
pub use types::*;
pub use traits::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_location_creation() {
        let loc = Location {
            latitude: 40.7128,
            longitude: -74.0060,
            altitude: Some(10.0),
        };
        assert_eq!(loc.latitude, 40.7128);
    }
}
