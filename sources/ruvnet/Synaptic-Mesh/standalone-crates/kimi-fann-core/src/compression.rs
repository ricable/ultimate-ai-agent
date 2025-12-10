//! Expert Compression and Decompression
//!
//! This module implements efficient compression for micro-expert storage
//! and streaming in WASM environments.

use crate::*;
use synaptic_neural_wasm::Activation;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Compression method
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CompressionMethod {
    /// No compression (for development/testing)
    None,
    /// LZ4 compression (fast compression/decompression)
    Lz4,
    /// DEFLATE compression (better ratio)
    Deflate,
    /// Custom neural network weight compression
    NeuralCompression,
}

impl Default for CompressionMethod {
    fn default() -> Self {
        CompressionMethod::Lz4
    }
}

/// Quantization settings for neural weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Number of bits for weight quantization (8, 16, or 32)
    pub bits: u8,
    /// Whether to use dynamic range quantization
    pub dynamic_range: bool,
    /// Scale factor for quantization
    pub scale_factor: f32,
    /// Zero point for quantization
    pub zero_point: i32,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            bits: 8,
            dynamic_range: true,
            scale_factor: 1.0,
            zero_point: 0,
        }
    }
}

/// Compression metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionMetadata {
    /// Original size in bytes
    pub original_size: usize,
    /// Compressed size in bytes
    pub compressed_size: usize,
    /// Compression method used
    pub method: CompressionMethod,
    /// Quantization configuration
    pub quantization: Option<QuantizationConfig>,
    /// Expert domain
    pub domain: ExpertDomain,
    /// Specialization
    pub specialization: Specialization,
    /// Parameter count
    pub parameter_count: usize,
    /// Confidence threshold
    pub confidence_threshold: f32,
    /// Compression timestamp
    pub timestamp: f64,
    /// Checksum for integrity verification
    pub checksum: u32,
}

/// Compressed expert data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedExpert {
    /// Expert identifier
    pub id: ExpertId,
    /// Compressed weight data
    pub compressed_weights: Vec<u8>,
    /// Network topology information
    pub network_topology: NetworkTopology,
    /// Compression metadata
    pub metadata: CompressionMetadata,
}

/// Network topology description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkTopology {
    /// Input layer size
    pub input_size: usize,
    /// Hidden layer sizes
    pub hidden_layers: Vec<usize>,
    /// Output layer size
    pub output_size: usize,
    /// Activation functions
    pub activations: Vec<String>, // Serialized activation names
    /// Layer types
    pub layer_types: Vec<String>,
}

/// Expert compression engine
pub struct ExpertCompressor {
    /// Default compression method
    method: CompressionMethod,
    /// Quantization configuration
    quantization: QuantizationConfig,
    /// Compression cache
    cache: HashMap<ExpertId, CompressedExpert>,
    /// Maximum cache size
    max_cache_size: usize,
}

impl ExpertCompressor {
    /// Create a new expert compressor
    pub fn new(method: CompressionMethod, quantization: QuantizationConfig) -> Self {
        Self {
            method,
            quantization,
            cache: HashMap::new(),
            max_cache_size: 100,
        }
    }

    /// Compress a micro-expert
    pub fn compress_expert(&mut self, expert: &KimiMicroExpert) -> Result<CompressedExpert> {
        // Check cache first
        if let Some(cached) = self.cache.get(&expert.id()) {
            return Ok(cached.clone());
        }

        let weights = self.extract_weights(expert)?;
        let topology = self.extract_topology(expert)?;

        // Quantize weights if configured
        let (quantized_weights, quant_config) = if self.quantization.bits < 32 {
            let quantized = self.quantize_weights(&weights)?;
            (quantized, Some(self.quantization.clone()))
        } else {
            (self.weights_to_bytes(&weights)?, None)
        };

        // Compress the quantized weights
        let compressed_data = match self.method {
            CompressionMethod::None => quantized_weights,
            CompressionMethod::Lz4 => self.compress_lz4(&quantized_weights)?,
            CompressionMethod::Deflate => self.compress_deflate(&quantized_weights)?,
            CompressionMethod::NeuralCompression => self.compress_neural(&quantized_weights)?,
        };

        let checksum = self.calculate_checksum(&compressed_data);

        let metadata = CompressionMetadata {
            original_size: weights.len() * std::mem::size_of::<f32>(),
            compressed_size: compressed_data.len(),
            method: self.method,
            quantization: quant_config,
            domain: expert.domain,
            specialization: expert.specialization.clone(),
            parameter_count: expert.parameter_count(),
            confidence_threshold: expert.confidence_threshold(),
            timestamp: Utils::now(),
            checksum,
        };

        let compressed_expert = CompressedExpert {
            id: expert.id(),
            compressed_weights: compressed_data,
            network_topology: topology,
            metadata,
        };

        // Cache the compressed expert
        self.cache_compressed_expert(compressed_expert.clone());

        Ok(compressed_expert)
    }

    /// Decompress a micro-expert
    pub fn decompress_expert(&self, compressed: &CompressedExpert) -> Result<KimiMicroExpert> {
        // Verify checksum
        let checksum = self.calculate_checksum(&compressed.compressed_weights);
        if checksum != compressed.metadata.checksum {
            return Err(KimiError::CompressionError(
                "Checksum verification failed".to_string()
            ));
        }

        // Decompress weights
        let decompressed_data = match compressed.metadata.method {
            CompressionMethod::None => compressed.compressed_weights.clone(),
            CompressionMethod::Lz4 => self.decompress_lz4(&compressed.compressed_weights)?,
            CompressionMethod::Deflate => self.decompress_deflate(&compressed.compressed_weights)?,
            CompressionMethod::NeuralCompression => self.decompress_neural(&compressed.compressed_weights)?,
        };

        // Dequantize if needed
        let weights = if let Some(quant_config) = &compressed.metadata.quantization {
            self.dequantize_weights(&decompressed_data, quant_config)?
        } else {
            self.bytes_to_weights(&decompressed_data)?
        };

        // Reconstruct the expert
        self.reconstruct_expert(compressed, weights)
    }

    /// Extract weights from expert (placeholder)
    fn extract_weights(&self, _expert: &KimiMicroExpert) -> Result<Vec<f32>> {
        // In a real implementation, this would extract actual weights from the neural network
        // For now, return placeholder weights
        Ok(vec![0.0; 1000]) // Placeholder
    }

    /// Extract network topology from expert
    fn extract_topology(&self, expert: &KimiMicroExpert) -> Result<NetworkTopology> {
        // This is a simplified extraction - in reality, we'd inspect the actual network
        Ok(NetworkTopology {
            input_size: 100, // Placeholder
            hidden_layers: vec![64, 32],
            output_size: 10,
            activations: vec!["ReLU".to_string(), "ReLU".to_string(), "Linear".to_string()],
            layer_types: vec!["Dense".to_string(), "Dense".to_string(), "Dense".to_string()],
        })
    }

    /// Quantize floating-point weights to reduced precision
    fn quantize_weights(&self, weights: &[f32]) -> Result<Vec<u8>> {
        match self.quantization.bits {
            8 => self.quantize_to_8bit(weights),
            16 => self.quantize_to_16bit(weights),
            _ => Err(KimiError::CompressionError(
                format!("Unsupported quantization bits: {}", self.quantization.bits)
            )),
        }
    }

    /// Quantize to 8-bit integers
    fn quantize_to_8bit(&self, weights: &[f32]) -> Result<Vec<u8>> {
        if self.quantization.dynamic_range {
            // Dynamic range quantization
            let min_val = weights.iter().copied().fold(f32::INFINITY, f32::min);
            let max_val = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = (max_val - min_val) / 255.0;
            
            if scale == 0.0 {
                return Ok(vec![0; weights.len()]);
            }

            let quantized: Vec<u8> = weights.iter()
                .map(|&w| ((w - min_val) / scale).round().clamp(0.0, 255.0) as u8)
                .collect();

            // Store scale and offset for dequantization
            let mut result = Vec::with_capacity(weights.len() + 8);
            result.extend_from_slice(&min_val.to_le_bytes());
            result.extend_from_slice(&scale.to_le_bytes());
            result.extend_from_slice(&quantized);

            Ok(result)
        } else {
            // Fixed range quantization
            let scale = self.quantization.scale_factor;
            let zero_point = self.quantization.zero_point;
            
            let quantized: Vec<u8> = weights.iter()
                .map(|&w| ((w / scale) + zero_point as f32).round().clamp(0.0, 255.0) as u8)
                .collect();

            Ok(quantized)
        }
    }

    /// Quantize to 16-bit integers
    fn quantize_to_16bit(&self, weights: &[f32]) -> Result<Vec<u8>> {
        let min_val = weights.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = (max_val - min_val) / 65535.0;
        
        if scale == 0.0 {
            return Ok(vec![0; weights.len() * 2 + 8]);
        }

        let quantized: Vec<u16> = weights.iter()
            .map(|&w| ((w - min_val) / scale).round().clamp(0.0, 65535.0) as u16)
            .collect();

        let mut result = Vec::with_capacity(weights.len() * 2 + 8);
        result.extend_from_slice(&min_val.to_le_bytes());
        result.extend_from_slice(&scale.to_le_bytes());
        
        for &val in &quantized {
            result.extend_from_slice(&val.to_le_bytes());
        }

        Ok(result)
    }

    /// Dequantize weights back to f32
    fn dequantize_weights(&self, data: &[u8], config: &QuantizationConfig) -> Result<Vec<f32>> {
        match config.bits {
            8 => self.dequantize_from_8bit(data, config),
            16 => self.dequantize_from_16bit(data),
            _ => Err(KimiError::CompressionError(
                format!("Unsupported quantization bits: {}", config.bits)
            )),
        }
    }

    /// Dequantize from 8-bit
    fn dequantize_from_8bit(&self, data: &[u8], config: &QuantizationConfig) -> Result<Vec<f32>> {
        if config.dynamic_range {
            if data.len() < 8 {
                return Err(KimiError::CompressionError("Invalid quantized data".to_string()));
            }

            let min_val = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            let scale = f32::from_le_bytes([data[4], data[5], data[6], data[7]]);
            let quantized = &data[8..];

            let weights: Vec<f32> = quantized.iter()
                .map(|&q| min_val + (q as f32) * scale)
                .collect();

            Ok(weights)
        } else {
            let scale = config.scale_factor;
            let zero_point = config.zero_point;

            let weights: Vec<f32> = data.iter()
                .map(|&q| (q as f32 - zero_point as f32) * scale)
                .collect();

            Ok(weights)
        }
    }

    /// Dequantize from 16-bit
    fn dequantize_from_16bit(&self, data: &[u8]) -> Result<Vec<f32>> {
        if data.len() < 8 {
            return Err(KimiError::CompressionError("Invalid quantized data".to_string()));
        }

        let min_val = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let scale = f32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let quantized_data = &data[8..];

        if quantized_data.len() % 2 != 0 {
            return Err(KimiError::CompressionError("Invalid 16-bit quantized data".to_string()));
        }

        let mut weights = Vec::with_capacity(quantized_data.len() / 2);
        for chunk in quantized_data.chunks_exact(2) {
            let q = u16::from_le_bytes([chunk[0], chunk[1]]);
            weights.push(min_val + (q as f32) * scale);
        }

        Ok(weights)
    }

    /// Convert weights to bytes
    fn weights_to_bytes(&self, weights: &[f32]) -> Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(weights.len() * 4);
        for &weight in weights {
            bytes.extend_from_slice(&weight.to_le_bytes());
        }
        Ok(bytes)
    }

    /// Convert bytes back to weights
    fn bytes_to_weights(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        if bytes.len() % 4 != 0 {
            return Err(KimiError::CompressionError("Invalid weight data length".to_string()));
        }

        let mut weights = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            let weight = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            weights.push(weight);
        }
        Ok(weights)
    }

    /// LZ4 compression
    fn compress_lz4(&self, data: &[u8]) -> Result<Vec<u8>> {
        lz4_flex::compress_prepend_size(data)
            .map_err(|e| KimiError::CompressionError(format!("LZ4 compression failed: {}", e)))
    }

    /// LZ4 decompression
    fn decompress_lz4(&self, data: &[u8]) -> Result<Vec<u8>> {
        lz4_flex::decompress_size_prepended(data)
            .map_err(|e| KimiError::CompressionError(format!("LZ4 decompression failed: {}", e)))
    }

    /// DEFLATE compression
    fn compress_deflate(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write;

        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data)
            .map_err(|e| KimiError::CompressionError(format!("DEFLATE write failed: {}", e)))?;
        encoder.finish()
            .map_err(|e| KimiError::CompressionError(format!("DEFLATE compression failed: {}", e)))
    }

    /// DEFLATE decompression
    fn decompress_deflate(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::read::DeflateDecoder;
        use std::io::Read;

        let mut decoder = DeflateDecoder::new(data);
        let mut result = Vec::new();
        decoder.read_to_end(&mut result)
            .map_err(|e| KimiError::CompressionError(format!("DEFLATE decompression failed: {}", e)))?;
        Ok(result)
    }

    /// Neural-specific compression (placeholder)
    fn compress_neural(&self, data: &[u8]) -> Result<Vec<u8>> {
        // This would implement neural network specific compression techniques
        // For now, fall back to LZ4
        self.compress_lz4(data)
    }

    /// Neural-specific decompression (placeholder)
    fn decompress_neural(&self, data: &[u8]) -> Result<Vec<u8>> {
        // This would implement neural network specific decompression techniques
        // For now, fall back to LZ4
        self.decompress_lz4(data)
    }

    /// Calculate checksum for integrity verification
    fn calculate_checksum(&self, data: &[u8]) -> u32 {
        // Simple CRC32-like checksum
        let mut checksum = 0u32;
        for &byte in data {
            checksum = checksum.wrapping_mul(31).wrapping_add(byte as u32);
        }
        checksum
    }

    /// Reconstruct expert from compressed data
    fn reconstruct_expert(&self, compressed: &CompressedExpert, weights: Vec<f32>) -> Result<KimiMicroExpert> {
        // Create expert configuration
        let architecture = NetworkArchitecture {
            input_size: compressed.network_topology.input_size,
            hidden_layers: compressed.network_topology.hidden_layers.clone(),
            output_size: compressed.network_topology.output_size,
            activations: compressed.network_topology.activations.iter()
                .map(|s| match s.as_str() {
                    "ReLU" => Activation::ReLU,
                    "Sigmoid" => Activation::Sigmoid,
                    "Tanh" => Activation::Tanh,
                    _ => Activation::Linear,
                })
                .collect(),
            dropout_rates: vec![],
        };

        let config = ExpertConfig {
            id: compressed.id,
            domain: compressed.metadata.domain,
            specialization: compressed.metadata.specialization.clone(),
            architecture,
            training_config: TrainingConfig {
                learning_rate: 0.001,
                batch_size: 32,
                epochs: 100,
                regularization: 0.01,
            },
            performance_thresholds: PerformanceThresholds::default(),
        };

        let config_json = serde_json::to_string(&config)?;
        let mut expert = KimiMicroExpert::new(&config_json)
            .map_err(|e| KimiError::ExpertError(format!("Failed to create expert: {:?}", e)))?;

        // In a real implementation, we would restore the actual weights here
        // For now, the expert is created with random weights

        Ok(expert)
    }

    /// Cache a compressed expert
    fn cache_compressed_expert(&mut self, compressed: CompressedExpert) {
        // Maintain cache size limit
        if self.cache.len() >= self.max_cache_size {
            // Remove oldest entry (simplified LRU)
            let oldest_key = self.cache.keys().next().copied();
            if let Some(key) = oldest_key {
                self.cache.remove(&key);
            }
        }

        self.cache.insert(compressed.id, compressed);
    }

    /// Get compression ratio
    pub fn get_compression_ratio(&self, compressed: &CompressedExpert) -> f32 {
        if compressed.metadata.original_size > 0 {
            compressed.metadata.compressed_size as f32 / compressed.metadata.original_size as f32
        } else {
            1.0
        }
    }

    /// Get compression statistics
    pub fn get_compression_stats(&self) -> CompressionStats {
        let mut total_original = 0;
        let mut total_compressed = 0;
        let mut count = 0;

        for compressed in self.cache.values() {
            total_original += compressed.metadata.original_size;
            total_compressed += compressed.metadata.compressed_size;
            count += 1;
        }

        CompressionStats {
            cached_experts: count,
            total_original_size: total_original,
            total_compressed_size: total_compressed,
            average_ratio: if total_original > 0 {
                total_compressed as f32 / total_original as f32
            } else {
                1.0
            },
            cache_memory_usage: self.estimate_cache_memory(),
        }
    }

    /// Estimate cache memory usage
    fn estimate_cache_memory(&self) -> usize {
        self.cache.values()
            .map(|compressed| {
                compressed.compressed_weights.len() + 
                std::mem::size_of::<CompressedExpert>()
            })
            .sum()
    }
}

impl Default for ExpertCompressor {
    fn default() -> Self {
        Self::new(CompressionMethod::default(), QuantizationConfig::default())
    }
}

/// Compression statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct CompressionStats {
    pub cached_experts: usize,
    pub total_original_size: usize,
    pub total_compressed_size: usize,
    pub average_ratio: f32,
    pub cache_memory_usage: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_8bit() {
        let compressor = ExpertCompressor::default();
        let weights = vec![0.0, 0.5, 1.0, -0.5, -1.0];
        
        let quantized = compressor.quantize_to_8bit(&weights).unwrap();
        assert!(quantized.len() > weights.len()); // Includes scale and offset
        
        let dequantized = compressor.dequantize_from_8bit(&quantized, &QuantizationConfig::default()).unwrap();
        assert_eq!(dequantized.len(), weights.len());
        
        // Check approximate equality (quantization introduces some error)
        for (original, restored) in weights.iter().zip(dequantized.iter()) {
            assert!((original - restored).abs() < 0.1);
        }
    }

    #[test]
    fn test_lz4_compression() {
        let compressor = ExpertCompressor::default();
        let data = vec![0u8; 1000]; // Highly compressible data
        
        let compressed = compressor.compress_lz4(&data).unwrap();
        assert!(compressed.len() < data.len());
        
        let decompressed = compressor.decompress_lz4(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_checksum_calculation() {
        let compressor = ExpertCompressor::default();
        let data1 = vec![1, 2, 3, 4, 5];
        let data2 = vec![1, 2, 3, 4, 6];
        
        let checksum1 = compressor.calculate_checksum(&data1);
        let checksum2 = compressor.calculate_checksum(&data2);
        
        assert_ne!(checksum1, checksum2);
        
        // Same data should produce same checksum
        let checksum3 = compressor.calculate_checksum(&data1);
        assert_eq!(checksum1, checksum3);
    }
}