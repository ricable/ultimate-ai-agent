//! Snapshot streaming with chunk-based transfers
//!
//! Implements efficient bulk data transfer for:
//! - Database snapshots
//! - Model checkpoints
//! - Pattern archives
//! - State synchronization

use crate::neural_bus::{Frame, FrameHeader, FrameType};
use crate::{NetworkError, Result};
use bytes::{Bytes, BytesMut};
use quinn::{RecvStream, SendStream};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;
// AsyncReadExt and AsyncWriteExt imported for potential future use
use tokio::sync::Mutex;
use tracing::{debug, info};

/// Default chunk size (1MB)
const DEFAULT_CHUNK_SIZE: usize = 1024 * 1024;

/// Snapshot metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    /// Snapshot identifier
    pub id: String,
    /// Total size in bytes
    pub total_size: u64,
    /// Number of chunks
    pub chunk_count: u64,
    /// Chunk size
    pub chunk_size: usize,
    /// SHA256 checksum of complete snapshot
    pub checksum: String,
    /// Snapshot type
    pub snapshot_type: String,
    /// Timestamp
    pub timestamp: i64,
}

/// Snapshot chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotChunk {
    /// Snapshot ID
    pub snapshot_id: String,
    /// Chunk index
    pub chunk_index: u64,
    /// Chunk data
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
    /// SHA256 checksum of this chunk
    pub checksum: String,
}

impl SnapshotChunk {
    /// Create a new chunk with checksum
    pub fn new(snapshot_id: String, chunk_index: u64, data: Vec<u8>) -> Self {
        let checksum = format!("{:x}", Sha256::digest(&data));
        Self {
            snapshot_id,
            chunk_index,
            data,
            checksum,
        }
    }

    /// Verify chunk checksum
    pub fn verify(&self) -> bool {
        let computed = format!("{:x}", Sha256::digest(&self.data));
        computed == self.checksum
    }
}

/// Snapshot manager for streaming large data
pub struct SnapshotManager {
    chunk_size: usize,
}

impl SnapshotManager {
    /// Create a new snapshot manager
    pub fn new(chunk_size: Option<usize>) -> Self {
        Self {
            chunk_size: chunk_size.unwrap_or(DEFAULT_CHUNK_SIZE),
        }
    }

    /// Stream a snapshot to a send stream
    pub async fn send_snapshot(
        &self,
        stream: Arc<Mutex<SendStream>>,
        snapshot_id: String,
        data: Bytes,
    ) -> Result<()> {
        info!(
            "Sending snapshot {} ({} bytes)",
            snapshot_id,
            data.len()
        );

        let total_size = data.len() as u64;
        let chunk_count = (total_size + self.chunk_size as u64 - 1) / self.chunk_size as u64;

        // Compute overall checksum
        let checksum = format!("{:x}", Sha256::digest(&data));

        // Send metadata first
        let metadata = SnapshotMetadata {
            id: snapshot_id.clone(),
            total_size,
            chunk_count,
            chunk_size: self.chunk_size,
            checksum: checksum.clone(),
            snapshot_type: "generic".to_string(),
            timestamp: chrono::Utc::now().timestamp_millis(),
        };

        let metadata_payload = serde_json::to_vec(&metadata)
            .map_err(|e| NetworkError::Serialization(e))?;

        let mut header = FrameHeader::new("snapshot_metadata".to_string());
        header.topic = Some(snapshot_id.clone());

        let metadata_frame = Frame::new(
            FrameType::Snapshot,
            header,
            Bytes::from(metadata_payload),
        );

        {
            let mut s = stream.lock().await;
            metadata_frame.write_to(&mut *s).await?;
        }

        debug!("Sent snapshot metadata: {} chunks", chunk_count);

        // Send chunks
        let mut offset = 0;
        let mut chunk_index = 0;

        while offset < data.len() {
            let end = std::cmp::min(offset + self.chunk_size, data.len());
            let chunk_data = data.slice(offset..end).to_vec();

            let chunk = SnapshotChunk::new(
                snapshot_id.clone(),
                chunk_index,
                chunk_data,
            );

            let chunk_payload = serde_json::to_vec(&chunk)
                .map_err(|e| NetworkError::Serialization(e))?;

            let mut chunk_header = FrameHeader::new("snapshot_chunk".to_string());
            chunk_header.topic = Some(format!("{}:{}", snapshot_id, chunk_index));

            let chunk_frame = Frame::new(
                FrameType::Snapshot,
                chunk_header,
                Bytes::from(chunk_payload),
            );

            {
                let mut s = stream.lock().await;
                chunk_frame.write_to(&mut *s).await?;
            }

            debug!(
                "Sent chunk {}/{} ({} bytes)",
                chunk_index + 1,
                chunk_count,
                end - offset
            );

            offset = end;
            chunk_index += 1;
        }

        info!("Snapshot {} sent successfully", snapshot_id);
        Ok(())
    }

    /// Receive a snapshot from a receive stream
    pub async fn receive_snapshot(
        &self,
        stream: Arc<Mutex<RecvStream>>,
    ) -> Result<(SnapshotMetadata, Bytes)> {
        // Receive metadata
        let metadata_frame = {
            let mut s = stream.lock().await;
            Frame::read_from(&mut *s).await?
        };

        let metadata: SnapshotMetadata = serde_json::from_slice(&metadata_frame.payload)
            .map_err(|e| NetworkError::Serialization(e))?;

        info!(
            "Receiving snapshot {} ({} bytes, {} chunks)",
            metadata.id, metadata.total_size, metadata.chunk_count
        );

        // Receive chunks
        let mut data = BytesMut::with_capacity(metadata.total_size as usize);
        let mut received_chunks = 0;

        while received_chunks < metadata.chunk_count {
            let chunk_frame = {
                let mut s = stream.lock().await;
                Frame::read_from(&mut *s).await?
            };

            let chunk: SnapshotChunk = serde_json::from_slice(&chunk_frame.payload)
                .map_err(|e| NetworkError::Serialization(e))?;

            // Verify chunk
            if !chunk.verify() {
                return Err(NetworkError::Internal(format!(
                    "Chunk {} checksum verification failed",
                    chunk.chunk_index
                )));
            }

            data.extend_from_slice(&chunk.data);
            received_chunks += 1;

            debug!(
                "Received chunk {}/{} ({} bytes)",
                received_chunks,
                metadata.chunk_count,
                chunk.data.len()
            );
        }

        // Verify overall checksum
        let computed_checksum = format!("{:x}", Sha256::digest(&data));
        if computed_checksum != metadata.checksum {
            return Err(NetworkError::Internal(
                "Snapshot checksum verification failed".to_string(),
            ));
        }

        info!("Snapshot {} received successfully", metadata.id);
        Ok((metadata, data.freeze()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_chunk_creation() {
        let data = vec![1, 2, 3, 4, 5];
        let chunk = SnapshotChunk::new("test".to_string(), 0, data.clone());

        assert_eq!(chunk.snapshot_id, "test");
        assert_eq!(chunk.chunk_index, 0);
        assert_eq!(chunk.data, data);
        assert!(chunk.verify());
    }

    #[test]
    fn test_snapshot_chunk_verification() {
        let mut chunk = SnapshotChunk::new("test".to_string(), 0, vec![1, 2, 3]);
        assert!(chunk.verify());

        // Corrupt data
        chunk.data[0] = 99;
        assert!(!chunk.verify());
    }

    #[test]
    fn test_snapshot_metadata_serialization() {
        let metadata = SnapshotMetadata {
            id: "test".to_string(),
            total_size: 1000,
            chunk_count: 10,
            chunk_size: 100,
            checksum: "abc123".to_string(),
            snapshot_type: "test".to_string(),
            timestamp: 123456789,
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let deserialized: SnapshotMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, "test");
        assert_eq!(deserialized.total_size, 1000);
    }
}
