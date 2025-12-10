//! Core data types for QUIC transport

use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use bytes::Bytes;

/// Configuration for QUIC connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionConfig {
    /// Server name for SNI
    pub server_name: String,

    /// Maximum idle timeout in milliseconds
    pub max_idle_timeout_ms: u64,

    /// Maximum concurrent bidirectional streams
    pub max_concurrent_streams: u32,

    /// Enable 0-RTT connection establishment
    pub enable_0rtt: bool,
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            server_name: "localhost".to_string(),
            max_idle_timeout_ms: 30_000,
            max_concurrent_streams: 100,
            enable_0rtt: true,
        }
    }
}

/// QUIC message structure for agent communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuicMessage {
    /// Message ID for tracking
    pub id: String,

    /// Message type identifier
    pub msg_type: MessageType,

    /// Payload data
    pub payload: Bytes,

    /// Optional metadata
    pub metadata: Option<serde_json::Value>,

    /// Timestamp in milliseconds
    pub timestamp: u64,
}

/// Message types for agent communication
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MessageType {
    /// Task assignment
    Task,

    /// Result response
    Result,

    /// Status update
    Status,

    /// Coordination message
    Coordination,

    /// Heartbeat ping
    Heartbeat,

    /// Custom message type
    Custom(String),
}

/// Connection metadata
#[derive(Debug, Clone)]
pub struct ConnectionMeta {
    /// Remote socket address
    pub remote_addr: SocketAddr,

    /// Connection ID
    pub connection_id: u64,

    /// Round-trip time in microseconds
    pub rtt_us: Option<u64>,

    /// Bytes sent
    pub bytes_sent: u64,

    /// Bytes received
    pub bytes_received: u64,
}

/// Stream information
#[derive(Debug, Clone)]
pub struct StreamInfo {
    /// Stream ID
    pub id: u64,

    /// Whether stream is bidirectional
    pub bidirectional: bool,

    /// Stream state
    pub state: StreamState,
}

/// Stream states
#[derive(Debug, Clone, PartialEq)]
pub enum StreamState {
    /// Stream is open for read/write
    Open,

    /// Stream is sending data
    Sending,

    /// Stream is receiving data
    Receiving,

    /// Stream is closed
    Closed,

    /// Stream encountered an error
    Error,
}

/// Connection pool statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PoolStats {
    /// Active connections
    pub active: usize,

    /// Idle connections
    pub idle: usize,

    /// Total connections created
    pub total_created: u64,

    /// Total connections closed
    pub total_closed: u64,

    /// Current streams
    pub current_streams: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ConnectionConfig::default();
        assert_eq!(config.max_concurrent_streams, 100);
        assert!(config.enable_0rtt);
    }

    #[test]
    fn test_message_type() {
        let msg_type = MessageType::Task;
        assert_eq!(msg_type, MessageType::Task);

        let custom = MessageType::Custom("test".to_string());
        assert!(matches!(custom, MessageType::Custom(_)));
    }
}
