//! RAN Optimizer Library
//!
//! Public API for RAN optimization functions that can be used by other modules.

pub mod prompts;
pub mod metrics;
pub mod recommendations;

use serde::{Deserialize, Serialize};

/// Re-export main types
pub use prompts::*;
pub use metrics::*;
pub use recommendations::*;

/// RAN Optimizer Result type
pub type Result<T> = std::result::Result<T, RanOptimizerError>;

/// RAN Optimizer Error types
#[derive(Debug)]
pub enum RanOptimizerError {
    InferenceError(String),
    ConfigError(String),
    ParseError(String),
    IoError(std::io::Error),
}

impl std::fmt::Display for RanOptimizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InferenceError(msg) => write!(f, "Inference error: {}", msg),
            Self::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Self::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for RanOptimizerError {}

impl From<std::io::Error> for RanOptimizerError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");
