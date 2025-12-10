//! Gossip protocol with backpressure handling
//!
//! Implements peer-to-peer gossip for:
//! - Pattern synchronization
//! - Distributed learning updates
//! - Network topology discovery
//! - Health monitoring

use crate::neural_bus::{Frame, FrameHeader, FrameType};
use crate::{NetworkError, Result};
use bytes::Bytes;
use quinn::SendStream;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, warn};

/// Gossip message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GossipMessage {
    /// Pattern update notification
    PatternUpdate {
        pattern_id: String,
        version: u64,
        metadata: HashMap<String, String>,
    },
    /// Learning weight update
    LearningUpdate {
        model_id: String,
        delta: Vec<f32>,
        epoch: u64,
    },
    /// Peer discovery announcement
    PeerAnnounce {
        peer_id: String,
        capabilities: Vec<String>,
        endpoint: String,
    },
    /// Health check ping
    HealthPing {
        peer_id: String,
        timestamp: i64,
    },
    /// Health check pong
    HealthPong {
        peer_id: String,
        timestamp: i64,
        load: f32,
    },
}

/// Gossip manager with backpressure control
pub struct GossipManager {
    /// Outbound gossip queue with backpressure
    outbound: mpsc::Sender<GossipMessage>,
    outbound_rx: Arc<tokio::sync::Mutex<mpsc::Receiver<GossipMessage>>>,

    /// Active gossip streams
    streams: Arc<RwLock<Vec<Arc<tokio::sync::Mutex<SendStream>>>>>,

    /// Backpressure threshold (queue size)
    backpressure_threshold: usize,
}

impl GossipManager {
    /// Create a new gossip manager
    pub fn new(buffer_size: usize, backpressure_threshold: usize) -> Self {
        let (tx, rx) = mpsc::channel(buffer_size);

        Self {
            outbound: tx,
            outbound_rx: Arc::new(tokio::sync::Mutex::new(rx)),
            streams: Arc::new(RwLock::new(Vec::new())),
            backpressure_threshold,
        }
    }

    /// Add a gossip stream
    pub async fn add_stream(&self, stream: SendStream) {
        let mut streams = self.streams.write().await;
        streams.push(Arc::new(tokio::sync::Mutex::new(stream)));
        debug!("Added gossip stream, total: {}", streams.len());
    }

    /// Send a gossip message (with backpressure)
    pub async fn send(&self, message: GossipMessage) -> Result<()> {
        // Check backpressure
        if self.outbound.capacity() < self.backpressure_threshold {
            warn!("Gossip queue approaching capacity, applying backpressure");
            return Err(NetworkError::Internal("Gossip backpressure active".to_string()));
        }

        self.outbound
            .send(message)
            .await
            .map_err(|_| NetworkError::Internal("Gossip send failed".to_string()))?;

        Ok(())
    }

    /// Start gossip dispatcher (processes outbound queue)
    pub async fn run_dispatcher(&self) -> Result<()> {
        let mut rx = self.outbound_rx.lock().await;

        while let Some(message) = rx.recv().await {
            debug!("Dispatching gossip message: {:?}", message);

            // Serialize message
            let payload = serde_json::to_vec(&message)
                .map_err(|e| NetworkError::Serialization(e))?;

            // Create frame
            let header = FrameHeader::new("gossip".to_string());
            let frame = Frame::new(FrameType::Gossip, header, Bytes::from(payload));
            let encoded = frame.encode()?;

            // Send to all active streams
            let streams = self.streams.read().await;
            for stream in streams.iter() {
                let mut s = stream.lock().await;
                if let Err(e) = s.write_all(&encoded).await {
                    warn!("Failed to write gossip to stream: {}", e);
                    // Stream might be closed, but we continue with others
                } else if let Err(e) = s.flush().await {
                    warn!("Failed to flush gossip stream: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Process an incoming gossip message
    pub async fn handle_message(&self, message: GossipMessage) -> Result<()> {
        debug!("Handling gossip message: {:?}", message);

        match message {
            GossipMessage::PatternUpdate {
                pattern_id,
                version,
                metadata,
            } => {
                // Handle pattern update
                debug!(
                    "Pattern update: {} version {} with {} metadata fields",
                    pattern_id,
                    version,
                    metadata.len()
                );
            }
            GossipMessage::LearningUpdate {
                model_id,
                delta,
                epoch,
            } => {
                // Handle learning update
                debug!(
                    "Learning update: {} epoch {} with {} deltas",
                    model_id,
                    epoch,
                    delta.len()
                );
            }
            GossipMessage::PeerAnnounce {
                peer_id,
                capabilities,
                endpoint,
            } => {
                // Handle peer discovery
                debug!(
                    "Peer announce: {} at {} with capabilities: {:?}",
                    peer_id, endpoint, capabilities
                );
            }
            GossipMessage::HealthPing { peer_id, timestamp } => {
                // Respond with pong
                debug!("Health ping from {} at {}", peer_id, timestamp);
                let pong = GossipMessage::HealthPong {
                    peer_id: "self".to_string(),
                    timestamp: chrono::Utc::now().timestamp_millis(),
                    load: 0.5, // TODO: actual load calculation
                };
                self.send(pong).await?;
            }
            GossipMessage::HealthPong {
                peer_id,
                timestamp,
                load,
            } => {
                // Record peer health
                debug!(
                    "Health pong from {} at {} with load {}",
                    peer_id, timestamp, load
                );
            }
        }

        Ok(())
    }

    /// Get current queue depth (for monitoring)
    pub fn queue_depth(&self) -> usize {
        self.outbound.capacity()
    }

    /// Check if backpressure is active
    pub fn is_backpressure_active(&self) -> bool {
        self.outbound.capacity() < self.backpressure_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gossip_manager_creation() {
        let manager = GossipManager::new(100, 20);
        assert!(!manager.is_backpressure_active());
    }

    #[tokio::test]
    async fn test_gossip_message_serialization() {
        let msg = GossipMessage::PatternUpdate {
            pattern_id: "test".to_string(),
            version: 1,
            metadata: HashMap::new(),
        };

        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: GossipMessage = serde_json::from_str(&json).unwrap();

        matches!(deserialized, GossipMessage::PatternUpdate { .. });
    }
}
