//! Binary serialization support using bincode

use crate::io::error::{IoError, IoResult};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

/// Read binary data from a reader
pub fn read_binary<T, R>(reader: &mut R) -> IoResult<T>
where
    T: for<'de> Deserialize<'de>,
    R: Read,
{
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer)?;

    let value = bincode::deserialize(&buffer)?;
    Ok(value)
}

/// Write binary data to a writer
pub fn write_binary<T, W>(data: &T, writer: &mut W) -> IoResult<()>
where
    T: Serialize,
    W: Write,
{
    let encoded = bincode::serialize(data)?;
    writer.write_all(&encoded)?;
    Ok(())
}

/// Binary format configuration
#[derive(Debug, Clone)]
pub struct BinaryConfig {
    /// Use little endian byte order
    pub little_endian: bool,
    /// Use variable length encoding for integers
    pub varint_encoding: bool,
}

impl BinaryConfig {
    /// Create a new binary config with default settings
    pub fn new() -> Self {
        Self {
            little_endian: true,
            varint_encoding: false,
        }
    }

    /// Create a config optimized for size (variable length encoding)
    pub fn compact() -> Self {
        Self {
            little_endian: true,
            varint_encoding: true,
        }
    }

    /// Create a config optimized for speed (fixed length encoding)
    pub fn fast() -> Self {
        Self {
            little_endian: true,
            varint_encoding: false,
        }
    }
}

impl Default for BinaryConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Binary reader with configuration
pub struct BinaryReader {
    config: BinaryConfig,
}

impl BinaryReader {
    /// Create a new binary reader with default config
    pub fn new() -> Self {
        Self {
            config: BinaryConfig::new(),
        }
    }

    /// Create a new binary reader with custom config
    pub fn with_config(config: BinaryConfig) -> Self {
        Self { config }
    }

    /// Read data from a reader
    pub fn read<T, R>(&self, reader: &mut R) -> IoResult<T>
    where
        T: for<'de> Deserialize<'de>,
        R: Read,
    {
        read_binary(reader)
    }

    /// Read data with size limit to prevent memory exhaustion
    pub fn read_with_limit<T, R>(&self, reader: &mut R, limit: u64) -> IoResult<T>
    where
        T: for<'de> Deserialize<'de>,
        R: Read,
    {
        let mut buffer = Vec::new();
        let mut limited_reader = reader.take(limit);
        limited_reader.read_to_end(&mut buffer)?;

        if buffer.len() as u64 == limit {
            return Err(IoError::SerializationError(
                "Data exceeds size limit".to_string(),
            ));
        }

        let value = bincode::deserialize(&buffer)?;
        Ok(value)
    }
}

impl Default for BinaryReader {
    fn default() -> Self {
        Self::new()
    }
}

/// Binary writer with configuration
pub struct BinaryWriter {
    config: BinaryConfig,
}

impl BinaryWriter {
    /// Create a new binary writer with default config
    pub fn new() -> Self {
        Self {
            config: BinaryConfig::new(),
        }
    }

    /// Create a new binary writer with custom config
    pub fn with_config(config: BinaryConfig) -> Self {
        Self { config }
    }

    /// Write data to a writer
    pub fn write<T, W>(&self, data: &T, writer: &mut W) -> IoResult<()>
    where
        T: Serialize,
        W: Write,
    {
        write_binary(data, writer)
    }

    /// Get the size of serialized data without writing
    pub fn serialized_size<T>(&self, data: &T) -> IoResult<u64>
    where
        T: Serialize,
    {
        let size = bincode::serialized_size(data)?;
        Ok(size)
    }
}

impl Default for BinaryWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Utilities for binary format inspection
pub mod inspect {
    use super::*;

    /// Get the size of data when serialized
    pub fn serialized_size<T>(data: &T) -> IoResult<u64>
    where
        T: Serialize,
    {
        let size = bincode::serialized_size(data)?;
        Ok(size)
    }

    /// Check if data can be serialized without errors
    pub fn validate_serializable<T>(data: &T) -> IoResult<()>
    where
        T: Serialize,
    {
        bincode::serialize(data)?;
        Ok(())
    }
}
