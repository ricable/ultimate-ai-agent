//! Compressed tensor storage for neural network data
//! 
//! Provides efficient storage and retrieval of tensor data with compression

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use memmap2::{Mmap, MmapOptions};

/// Tensor data type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorDataType {
    Float32,
    Float16,
    Int32,
    Int16,
    Int8,
    UInt8,
}

impl TensorDataType {
    pub fn size_of(&self) -> usize {
        match self {
            TensorDataType::Float32 | TensorDataType::Int32 => 4,
            TensorDataType::Float16 | TensorDataType::Int16 => 2,
            TensorDataType::Int8 | TensorDataType::UInt8 => 1,
        }
    }
}

/// Tensor shape and metadata
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub shape: Vec<usize>,
    pub data_type: TensorDataType,
    pub compressed: bool,
    pub compression_ratio: f32,
}

impl TensorMeta {
    pub fn new(shape: Vec<usize>, data_type: TensorDataType) -> Self {
        Self {
            shape,
            data_type,
            compressed: false,
            compression_ratio: 1.0,
        }
    }

    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size_bytes(&self) -> usize {
        self.element_count() * self.data_type.size_of()
    }
}

/// Compressed tensor storage
pub struct TensorStorage {
    /// Tensor metadata
    pub meta: TensorMeta,
    /// Compressed data
    pub data: Vec<u8>,
    /// Memory-mapped file for large tensors
    pub mmap: Option<Mmap>,
}

impl TensorStorage {
    /// Create new tensor storage
    pub fn new(meta: TensorMeta) -> Self {
        Self {
            meta,
            data: Vec::new(),
            mmap: None,
        }
    }

    /// Store tensor data with compression
    pub fn store_compressed(&mut self, data: &[f32], compression_level: u32) -> Result<(), Box<dyn std::error::Error>> {
        let raw_bytes = self.float_slice_to_bytes(data);
        
        // Compress using zstd
        let compressed = zstd::bulk::compress(&raw_bytes, compression_level as i32)?;
        
        self.data = compressed;
        self.meta.compressed = true;
        self.meta.compression_ratio = raw_bytes.len() as f32 / self.data.len() as f32;
        
        Ok(())
    }

    /// Load tensor data with decompression
    pub fn load_decompressed(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        if self.meta.compressed {
            let decompressed = zstd::bulk::decompress(&self.data, self.meta.size_bytes())?;
            Ok(self.bytes_to_float_vec(&decompressed))
        } else {
            Ok(self.bytes_to_float_vec(&self.data))
        }
    }

    /// Store tensor to file with memory mapping
    pub fn store_to_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);
        
        // Write metadata
        self.write_metadata(&mut writer)?;
        
        // Write data
        writer.write_all(&self.data)?;
        writer.flush()?;
        
        // Memory map the file
        let file = File::open(&path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        self.mmap = Some(mmap);
        
        Ok(())
    }

    /// Load tensor from file with memory mapping
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(&path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        
        // Read metadata
        let mut cursor = std::io::Cursor::new(&mmap[..]);
        let meta = Self::read_metadata(&mut cursor)?;
        
        // Read data
        let data_start = cursor.position() as usize;
        let data = mmap[data_start..].to_vec();
        
        Ok(Self {
            meta,
            data,
            mmap: Some(mmap),
        })
    }

    /// Convert float slice to bytes
    fn float_slice_to_bytes(&self, data: &[f32]) -> Vec<u8> {
        match self.meta.data_type {
            TensorDataType::Float32 => {
                let mut bytes = Vec::with_capacity(data.len() * 4);
                for &value in data {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                bytes
            }
            TensorDataType::Float16 => {
                let mut bytes = Vec::with_capacity(data.len() * 2);
                for &value in data {
                    let f16_value = half::f16::from_f32(value);
                    bytes.extend_from_slice(&f16_value.to_le_bytes());
                }
                bytes
            }
            TensorDataType::Int8 => {
                data.iter()
                    .map(|&x| (x.clamp(-128.0, 127.0) as i8) as u8)
                    .collect()
            }
            TensorDataType::UInt8 => {
                data.iter()
                    .map(|&x| (x.clamp(0.0, 255.0) as u8))
                    .collect()
            }
            _ => panic!("Unsupported data type conversion"),
        }
    }

    /// Convert bytes to float vector
    fn bytes_to_float_vec(&self, bytes: &[u8]) -> Vec<f32> {
        match self.meta.data_type {
            TensorDataType::Float32 => {
                bytes.chunks(4)
                    .map(|chunk| {
                        let array: [u8; 4] = chunk.try_into().unwrap();
                        f32::from_le_bytes(array)
                    })
                    .collect()
            }
            TensorDataType::Float16 => {
                bytes.chunks(2)
                    .map(|chunk| {
                        let array: [u8; 2] = chunk.try_into().unwrap();
                        half::f16::from_le_bytes(array).to_f32()
                    })
                    .collect()
            }
            TensorDataType::Int8 => {
                bytes.iter()
                    .map(|&b| (b as i8) as f32)
                    .collect()
            }
            TensorDataType::UInt8 => {
                bytes.iter()
                    .map(|&b| b as f32)
                    .collect()
            }
            _ => panic!("Unsupported data type conversion"),
        }
    }

    /// Write tensor metadata
    fn write_metadata<W: Write>(&self, writer: &mut W) -> Result<(), Box<dyn std::error::Error>> {
        // Magic number
        writer.write_all(b"TENS")?;
        
        // Version
        writer.write_all(&1u32.to_le_bytes())?;
        
        // Shape
        writer.write_all(&(self.meta.shape.len() as u32).to_le_bytes())?;
        for &dim in &self.meta.shape {
            writer.write_all(&(dim as u64).to_le_bytes())?;
        }
        
        // Data type
        writer.write_all(&(self.meta.data_type as u8).to_le_bytes())?;
        
        // Compression info
        writer.write_all(&[if self.meta.compressed { 1 } else { 0 }])?;
        writer.write_all(&self.meta.compression_ratio.to_le_bytes())?;
        
        // Data length
        writer.write_all(&(self.data.len() as u64).to_le_bytes())?;
        
        Ok(())
    }

    /// Read tensor metadata
    fn read_metadata<R: Read>(reader: &mut R) -> Result<TensorMeta, Box<dyn std::error::Error>> {
        // Magic number
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"TENS" {
            return Err("Invalid tensor file format".into());
        }
        
        // Version
        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let _version = u32::from_le_bytes(version_bytes);
        
        // Shape
        let mut shape_len_bytes = [0u8; 4];
        reader.read_exact(&mut shape_len_bytes)?;
        let shape_len = u32::from_le_bytes(shape_len_bytes) as usize;
        
        let mut shape = Vec::with_capacity(shape_len);
        for _ in 0..shape_len {
            let mut dim_bytes = [0u8; 8];
            reader.read_exact(&mut dim_bytes)?;
            shape.push(u64::from_le_bytes(dim_bytes) as usize);
        }
        
        // Data type
        let mut data_type_bytes = [0u8; 1];
        reader.read_exact(&mut data_type_bytes)?;
        let data_type = match data_type_bytes[0] {
            0 => TensorDataType::Float32,
            1 => TensorDataType::Float16,
            2 => TensorDataType::Int32,
            3 => TensorDataType::Int16,
            4 => TensorDataType::Int8,
            5 => TensorDataType::UInt8,
            _ => return Err("Unknown data type".into()),
        };
        
        // Compression info
        let mut compressed_bytes = [0u8; 1];
        reader.read_exact(&mut compressed_bytes)?;
        let compressed = compressed_bytes[0] != 0;
        
        let mut compression_ratio_bytes = [0u8; 4];
        reader.read_exact(&mut compression_ratio_bytes)?;
        let compression_ratio = f32::from_le_bytes(compression_ratio_bytes);
        
        // Data length
        let mut data_len_bytes = [0u8; 8];
        reader.read_exact(&mut data_len_bytes)?;
        let _data_len = u64::from_le_bytes(data_len_bytes);
        
        Ok(TensorMeta {
            shape,
            data_type,
            compressed,
            compression_ratio,
        })
    }
}

/// Tensor batch for neural network training
pub struct TensorBatch {
    /// Input tensors
    pub inputs: Vec<TensorStorage>,
    /// Target tensors
    pub targets: Vec<TensorStorage>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl TensorBatch {
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            targets: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add input tensor
    pub fn add_input(&mut self, tensor: TensorStorage) {
        self.inputs.push(tensor);
    }

    /// Add target tensor
    pub fn add_target(&mut self, tensor: TensorStorage) {
        self.targets.push(tensor);
    }

    /// Save batch to directory
    pub fn save_to_dir<P: AsRef<Path>>(&mut self, dir: P) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(&dir)?;
        
        // Save inputs
        for (i, tensor) in self.inputs.iter_mut().enumerate() {
            let path = dir.as_ref().join(format!("input_{}.tensor", i));
            tensor.store_to_file(path)?;
        }
        
        // Save targets
        for (i, tensor) in self.targets.iter_mut().enumerate() {
            let path = dir.as_ref().join(format!("target_{}.tensor", i));
            tensor.store_to_file(path)?;
        }
        
        // Save metadata
        let metadata_path = dir.as_ref().join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&self.metadata)?;
        std::fs::write(metadata_path, metadata_json)?;
        
        Ok(())
    }

    /// Load batch from directory
    pub fn load_from_dir<P: AsRef<Path>>(dir: P) -> Result<Self, Box<dyn std::error::Error>> {
        let mut batch = Self::new();
        
        // Load inputs
        let mut i = 0;
        loop {
            let path = dir.as_ref().join(format!("input_{}.tensor", i));
            if path.exists() {
                let tensor = TensorStorage::load_from_file(path)?;
                batch.add_input(tensor);
                i += 1;
            } else {
                break;
            }
        }
        
        // Load targets
        let mut i = 0;
        loop {
            let path = dir.as_ref().join(format!("target_{}.tensor", i));
            if path.exists() {
                let tensor = TensorStorage::load_from_file(path)?;
                batch.add_target(tensor);
                i += 1;
            } else {
                break;
            }
        }
        
        // Load metadata
        let metadata_path = dir.as_ref().join("metadata.json");
        if metadata_path.exists() {
            let metadata_json = std::fs::read_to_string(metadata_path)?;
            batch.metadata = serde_json::from_str(&metadata_json)?;
        }
        
        Ok(batch)
    }
}

/// Tensor dataset for managing multiple batches
pub struct TensorDataset {
    pub batches: Vec<TensorBatch>,
    pub batch_size: usize,
    pub shuffle: bool,
}

impl TensorDataset {
    pub fn new(batch_size: usize, shuffle: bool) -> Self {
        Self {
            batches: Vec::new(),
            batch_size,
            shuffle,
        }
    }

    /// Add batch to dataset
    pub fn add_batch(&mut self, batch: TensorBatch) {
        self.batches.push(batch);
    }

    /// Get iterator over batches
    pub fn iter_batches(&self) -> std::slice::Iter<TensorBatch> {
        self.batches.iter()
    }

    /// Get total number of samples
    pub fn len(&self) -> usize {
        self.batches.len() * self.batch_size
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_tensor_storage() {
        let meta = TensorMeta::new(vec![2, 3], TensorDataType::Float32);
        let mut storage = TensorStorage::new(meta);
        
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        storage.store_compressed(&data, 3).unwrap();
        
        let loaded = storage.load_decompressed().unwrap();
        assert_eq!(data, loaded);
    }

    #[test]
    fn test_tensor_batch() {
        let mut batch = TensorBatch::new();
        
        let meta = TensorMeta::new(vec![2, 2], TensorDataType::Float32);
        let mut tensor = TensorStorage::new(meta);
        let data = vec![1.0, 2.0, 3.0, 4.0];
        tensor.store_compressed(&data, 3).unwrap();
        
        batch.add_input(tensor);
        
        let dir = tempdir().unwrap();
        batch.save_to_dir(dir.path()).unwrap();
        
        let loaded_batch = TensorBatch::load_from_dir(dir.path()).unwrap();
        assert_eq!(loaded_batch.inputs.len(), 1);
    }

    #[test]
    fn test_tensor_dataset() {
        let mut dataset = TensorDataset::new(32, true);
        
        let batch = TensorBatch::new();
        dataset.add_batch(batch);
        
        assert_eq!(dataset.len(), 32);
        assert!(!dataset.is_empty());
    }
}