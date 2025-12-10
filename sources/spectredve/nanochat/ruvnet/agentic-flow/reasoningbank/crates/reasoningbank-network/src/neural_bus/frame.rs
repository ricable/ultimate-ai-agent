//! Frame protocol for neural bus
//!
//! Wire format:
//! - version: u16 (2 bytes)
//! - type: u8 (1 byte)
//! - header_len: u32 (4 bytes)
//! - payload_len: u32 (4 bytes)
//! - header: JSON (header_len bytes)
//! - payload: bytes (payload_len bytes)

use crate::{NetworkError, Result};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Protocol version
pub const PROTOCOL_VERSION: u16 = 1;

/// Frame header size (version + type + header_len + payload_len)
pub const FRAME_HEADER_SIZE: usize = 2 + 1 + 4 + 4; // 11 bytes

/// Frame types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum FrameType {
    /// Control frame (connection management)
    Control = 0,
    /// Request frame
    Request = 1,
    /// Response frame
    Response = 2,
    /// Gossip message
    Gossip = 3,
    /// Snapshot chunk
    Snapshot = 4,
    /// Telemetry data
    Telemetry = 5,
    /// Token stream (reasoning)
    Token = 6,
    /// Trace stream (reasoning)
    Trace = 7,
    /// Rubric stream (reasoning)
    Rubric = 8,
    /// Verification stream (reasoning)
    Verify = 9,
}

impl FrameType {
    pub fn from_u8(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::Control),
            1 => Some(Self::Request),
            2 => Some(Self::Response),
            3 => Some(Self::Gossip),
            4 => Some(Self::Snapshot),
            5 => Some(Self::Telemetry),
            6 => Some(Self::Token),
            7 => Some(Self::Trace),
            8 => Some(Self::Rubric),
            9 => Some(Self::Verify),
            _ => None,
        }
    }

    pub fn is_mutating(&self) -> bool {
        matches!(self, Self::Request | Self::Control | Self::Snapshot)
    }
}

/// Frame header (JSON-encoded metadata)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameHeader {
    /// Key ID for signature verification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kid: Option<String>,

    /// Timestamp (Unix milliseconds)
    pub ts: i64,

    /// Nonce for replay protection
    pub nonce: String,

    /// Scope (intent-based authorization)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,

    /// Spend cap (for intent-capped actions)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cap: Option<u64>,

    /// Ed25519 signature (base64)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sig: Option<String>,

    /// Operation name
    pub op: String,

    /// Topic (for pub/sub patterns)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topic: Option<String>,

    /// Protocol version
    pub version: u16,
}

impl FrameHeader {
    /// Create a new frame header
    pub fn new(op: String) -> Self {
        Self {
            kid: None,
            ts: chrono::Utc::now().timestamp_millis(),
            nonce: uuid::Uuid::new_v4().to_string(),
            scope: None,
            cap: None,
            sig: None,
            op,
            topic: None,
            version: PROTOCOL_VERSION,
        }
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| NetworkError::Serialization(e))
    }

    /// Deserialize from JSON
    pub fn from_json(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(|e| NetworkError::Serialization(e))
    }
}

/// Complete frame with header and payload
#[derive(Debug, Clone)]
pub struct Frame {
    pub version: u16,
    pub frame_type: FrameType,
    pub header: FrameHeader,
    pub payload: Bytes,
}

impl Frame {
    /// Create a new frame
    pub fn new(frame_type: FrameType, header: FrameHeader, payload: Bytes) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            frame_type,
            header,
            payload,
        }
    }

    /// Encode frame to bytes
    pub fn encode(&self) -> Result<Bytes> {
        let header_json = self.header.to_json()?;
        let header_len = header_json.len() as u32;
        let payload_len = self.payload.len() as u32;

        let total_size = FRAME_HEADER_SIZE + header_len as usize + payload_len as usize;
        let mut buf = BytesMut::with_capacity(total_size);

        // Write frame header
        buf.put_u16(self.version);
        buf.put_u8(self.frame_type as u8);
        buf.put_u32(header_len);
        buf.put_u32(payload_len);

        // Write header JSON
        buf.put_slice(&header_json);

        // Write payload
        buf.put_slice(&self.payload);

        Ok(buf.freeze())
    }

    /// Decode frame from bytes
    pub fn decode(mut data: Bytes) -> Result<Self> {
        if data.len() < FRAME_HEADER_SIZE {
            return Err(NetworkError::Internal("Incomplete frame header".to_string()));
        }

        // Read frame header
        let version = data.get_u16();
        let frame_type_byte = data.get_u8();
        let header_len = data.get_u32() as usize;
        let payload_len = data.get_u32() as usize;

        // Validate version
        if version != PROTOCOL_VERSION {
            return Err(NetworkError::Internal(format!(
                "Unsupported protocol version: {}",
                version
            )));
        }

        // Parse frame type
        let frame_type = FrameType::from_u8(frame_type_byte)
            .ok_or_else(|| NetworkError::Internal("Invalid frame type".to_string()))?;

        // Check remaining data
        if data.len() < header_len + payload_len {
            return Err(NetworkError::Internal("Incomplete frame data".to_string()));
        }

        // Read header JSON
        let header_data = data.split_to(header_len);
        let header = FrameHeader::from_json(&header_data)?;

        // Read payload
        let payload = data.split_to(payload_len);

        Ok(Self {
            version,
            frame_type,
            header,
            payload,
        })
    }

    /// Write frame to async writer
    pub async fn write_to<W: AsyncWriteExt + Unpin>(&self, writer: &mut W) -> Result<()> {
        let encoded = self.encode()?;
        writer.write_all(&encoded).await
            .map_err(|e| NetworkError::Connection(e.to_string()))?;
        writer.flush().await
            .map_err(|e| NetworkError::Connection(e.to_string()))?;
        Ok(())
    }

    /// Read frame from async reader
    pub async fn read_from<R: AsyncReadExt + Unpin>(reader: &mut R) -> Result<Self> {
        // Read frame header
        let mut header_buf = [0u8; FRAME_HEADER_SIZE];
        reader.read_exact(&mut header_buf).await
            .map_err(|e| NetworkError::Connection(e.to_string()))?;

        let version = u16::from_be_bytes([header_buf[0], header_buf[1]]);
        let frame_type_byte = header_buf[2];
        let header_len = u32::from_be_bytes([
            header_buf[3],
            header_buf[4],
            header_buf[5],
            header_buf[6],
        ]) as usize;
        let payload_len = u32::from_be_bytes([
            header_buf[7],
            header_buf[8],
            header_buf[9],
            header_buf[10],
        ]) as usize;

        // Validate version
        if version != PROTOCOL_VERSION {
            return Err(NetworkError::Internal(format!(
                "Unsupported protocol version: {}",
                version
            )));
        }

        // Parse frame type
        let frame_type = FrameType::from_u8(frame_type_byte)
            .ok_or_else(|| NetworkError::Internal("Invalid frame type".to_string()))?;

        // Read header JSON
        let mut header_data = vec![0u8; header_len];
        reader.read_exact(&mut header_data).await
            .map_err(|e| NetworkError::Connection(e.to_string()))?;
        let header = FrameHeader::from_json(&header_data)?;

        // Read payload
        let mut payload_data = vec![0u8; payload_len];
        reader.read_exact(&mut payload_data).await
            .map_err(|e| NetworkError::Connection(e.to_string()))?;
        let payload = Bytes::from(payload_data);

        Ok(Self {
            version,
            frame_type,
            header,
            payload,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_encode_decode() {
        let header = FrameHeader::new("test_op".to_string());
        let payload = Bytes::from("test payload");
        let frame = Frame::new(FrameType::Request, header, payload.clone());

        let encoded = frame.encode().unwrap();
        let decoded = Frame::decode(encoded).unwrap();

        assert_eq!(decoded.version, PROTOCOL_VERSION);
        assert_eq!(decoded.frame_type, FrameType::Request);
        assert_eq!(decoded.header.op, "test_op");
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn test_frame_type_is_mutating() {
        assert!(FrameType::Request.is_mutating());
        assert!(FrameType::Control.is_mutating());
        assert!(!FrameType::Response.is_mutating());
        assert!(!FrameType::Token.is_mutating());
    }
}
