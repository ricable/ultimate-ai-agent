//! WASM stub implementation
//!
//! This provides a minimal API surface for WASM compatibility.
//! Full QUIC requires WebTransport which isn't widely supported yet.

use crate::{
    error::QuicError,
    types::{ConnectionConfig, QuicMessage, PoolStats},
    Result,
};

/// WASM-compatible QUIC client stub
#[derive(Clone)]
pub struct QuicClient {
    config: ConnectionConfig,
}

impl QuicClient {
    /// Create new QUIC client (WASM stub)
    pub async fn new(config: ConnectionConfig) -> Result<Self> {
        Ok(Self { config })
    }

    /// Connect to server (WASM stub - not implemented)
    pub async fn connect(&self, _addr: std::net::SocketAddr) -> Result<()> {
        Err(QuicError::Connection(
            "QUIC not supported in WASM - use WebTransport or native build".to_string()
        ))
    }

    /// Send message (WASM stub - not implemented)
    pub async fn send_message(
        &self,
        _addr: std::net::SocketAddr,
        _message: QuicMessage,
    ) -> Result<()> {
        Err(QuicError::Connection(
            "QUIC not supported in WASM - use WebTransport or native build".to_string()
        ))
    }

    /// Receive message (WASM stub - not implemented)
    pub async fn recv_message(
        &self,
        _addr: std::net::SocketAddr,
    ) -> Result<QuicMessage> {
        Err(QuicError::Connection(
            "QUIC not supported in WASM - use WebTransport or native build".to_string()
        ))
    }

    /// Get pool statistics (WASM stub - returns zeros)
    pub async fn pool_stats(&self) -> PoolStats {
        PoolStats::default()
    }

    /// Close connections (WASM stub - no-op)
    pub async fn close(&self) {
        // No-op in WASM stub
    }
}
