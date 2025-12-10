//! High-performance QUIC transport for agentic-flow
//!
//! This crate provides a production-ready QUIC implementation using quinn
//! with WASM support for browser-based agents.

#![warn(missing_docs)]

pub mod error;
pub mod types;

// Native-only modules
#[cfg(not(target_family = "wasm"))]
pub mod client;
#[cfg(not(target_family = "wasm"))]
pub mod server;

// WASM stub module
#[cfg(target_family = "wasm")]
pub mod wasm_stub;

// WASM-only modules
#[cfg(all(target_family = "wasm", feature = "wasm"))]
pub mod wasm;

// Re-exports
pub use error::QuicError;
pub use types::{ConnectionConfig, MessageType, QuicMessage, PoolStats};

/// Result type alias for QUIC operations
pub type Result<T> = std::result::Result<T, QuicError>;

#[cfg(not(target_family = "wasm"))]
pub use client::QuicClient;
#[cfg(not(target_family = "wasm"))]
pub use server::QuicServer;

#[cfg(target_family = "wasm")]
pub use wasm_stub::QuicClient;

#[cfg(all(target_family = "wasm", feature = "wasm"))]
pub use wasm::WasmQuicClient;

/// Initialize the QUIC transport with configuration validation
pub fn init(config: &ConnectionConfig) -> Result<()> {
    if config.max_idle_timeout_ms < 1000 {
        return Err(QuicError::InvalidConfig(
            "max_idle_timeout_ms must be >= 1000ms".to_string(),
        ));
    }

    if config.max_concurrent_streams == 0 {
        return Err(QuicError::InvalidConfig(
            "max_concurrent_streams must be > 0".to_string(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_valid_config() {
        let config = ConnectionConfig::default();
        assert!(init(&config).is_ok());
    }

    #[test]
    fn test_init_invalid_timeout() {
        let mut config = ConnectionConfig::default();
        config.max_idle_timeout_ms = 500;
        assert!(init(&config).is_err());
    }
}
