//! # Rights-Preserving Countermeasure Platform
//!
//! A production-grade Rust platform for AI governance, auditing, and rights-preserving oversight
//! with differential privacy guarantees.
//!
//! ## Features
//!
//! - **Privacy by Design**: Differential privacy with SmartNoise integration
//! - **Policy Enforcement**: Open Policy Agent (OPA) integration
//! - **Cryptographic Audit**: Immutable blockchain-style audit trails
//! - **Federated Learning**: Privacy-preserving distributed model training
//! - **Bias Detection**: Automated fairness testing
//! - **Emergency Controls**: Multi-layer kill switch mechanisms
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use rights_preserving_platform::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Initialize platform
//!     let platform = Platform::builder()
//!         .with_privacy(PrivacyConfig::default())
//!         .build()?;
//!
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod api;
pub mod audit;
pub mod governance;
pub mod privacy;
pub mod services;

// Re-export individual module types
pub use api::gateway;
pub use audit::logger;
pub use governance::policy;
pub use privacy::differential;
pub use services::grpc;

/// Re-exports of commonly used types for convenience
pub mod prelude {
    //! Prelude module for convenient imports
    //!
    //! This module re-exports commonly used types and traits.

    pub use crate::api::*;
    pub use crate::audit::*;
    pub use crate::governance::*;
    pub use crate::privacy::*;
    pub use crate::services::*;

    /// Common Result type
    pub type Result<T> = std::result::Result<T, Error>;

    /// Platform error type
    #[derive(Debug, thiserror::Error)]
    pub enum Error {
        /// Privacy error
        #[error("Privacy error: {0}")]
        Privacy(String),

        /// Policy error
        #[error("Policy error: {0}")]
        Policy(String),

        /// Audit error
        #[error("Audit error: {0}")]
        Audit(String),

        /// Configuration error
        #[error("Configuration error: {0}")]
        Config(String),

        /// I/O error
        #[error("I/O error: {0}")]
        Io(#[from] std::io::Error),

        /// Other error
        #[error("{0}")]
        Other(String),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prelude_imports() {
        use prelude::*;
        // Test that prelude exports work
        let _ = Error::Other("test".to_string());
    }
}
