//! # ReasoningBank Network
//!
//! QUIC-based networking layer with 0-RTT and stream multiplexing.
//!
//! This crate provides high-performance networking using the QUIC protocol
//! for distributed reasoning and pattern synchronization.
//!
//! ## Features
//!
//! - **Basic QUIC**: Simple client/server with message passing
//! - **Neural Bus** (native-only): Advanced multi-stream architecture with:
//!   - Role-based stream multiplexing
//!   - Intent-capped actions with Ed25519 signatures
//!   - Application-level priority queues
//!   - Gossip and snapshot streaming
//!   - Reasoning streams with parallel verification

pub mod quic;

#[cfg(not(target_arch = "wasm32"))]
pub mod neural_bus;

pub use quic::{QuicServer, QuicClient, QuicConfig, NetworkMessage};

#[cfg(not(target_arch = "wasm32"))]
pub use neural_bus::{
    NeuralBus, NeuralBusConfig, Frame, FrameType, FrameHeader,
    Intent, IntentVerifier, Scope, Priority, PriorityQueue,
    StreamRole, NeuralStream, ReasoningStreams,
    GossipManager, SnapshotManager,
};

#[cfg(not(target_arch = "wasm32"))]
pub use neural_bus::gossip::GossipMessage;

use thiserror::Error;

/// Result type for network operations
pub type Result<T> = std::result::Result<T, NetworkError>;

/// Errors that can occur in network operations
#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Connection error: {0}")]
    Connection(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("TLS error: {0}")]
    Tls(String),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("Internal error: {0}")]
    Internal(String),
}
