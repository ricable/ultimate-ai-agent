# PFS Logs - Log Anomaly Detection Networks

## Overview

The PFS Logs module implements specialized neural architectures for log anomaly detection, specifically optimized for Ericsson network equipment logs. It features transformer-based models with sliding window attention, online learning capabilities, and custom attention kernels for efficient processing.

## Key Features

### 🧠 Neural Architecture
- **Transformer-based log sequence models** with multi-head attention
- **Sliding window attention** for efficient long sequence processing
- **Quantized embeddings** for memory efficiency
- **Custom attention kernels** optimized for log patterns

### 📝 Log Processing
- **Ericsson-specific parsers** for AMOS commands (alt, lget, cvc)
- **Syslog format parsing** with multiple pattern support
- **Semi-structured text extraction** with key-value pairs
- **Byte-pair encoding (BPE)** tokenization

### 🔄 Online Learning
- **Incremental model updates** for continuous learning
- **Adaptive learning rate** with momentum
- **Experience replay buffer** for stable training
- **Real-time anomaly threshold adaptation**

### 🎯 Anomaly Detection
- **Ensemble methods** combining multiple detection approaches
- **Autoencoder-based reconstruction** error analysis
- **Isolation forest** for outlier detection
- **Statistical models** with Mahalanobis distance

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Log Input     │───▶│  BPE Tokenizer   │───▶│ Transformer     │
│                 │    │                  │    │ Encoder         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Anomaly Score   │◀───│ Anomaly Scorer   │◀───│ Feature         │
│                 │    │                  │    │ Extractor       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

```rust
use ran_opt::pfs_logs::{LogAnomalyDetector, DetectorConfig};

// Initialize detector with custom configuration
let config = DetectorConfig {
    sequence_length: 256,
    embedding_dim: 128,
    num_heads: 8,
    num_layers: 4,
    window_size: 32,
    anomaly_threshold: 0.85,
    ..Default::default()
};

let mut detector = LogAnomalyDetector::new(config);

// Process logs
let logs = vec![
    "2024-01-04 10:15:23 AMOS alt cell=12345 state=active".to_string(),
    "2024-01-04 10:15:24 ERROR: Connection timeout on node RBS_01".to_string(),
];

let results = detector.process_logs(&logs);

for result in results {
    println!("Score: {:.4}, Type: {}", result.score, result.log_type);
    if result.score > 0.8 {
        println!("🚨 Anomaly detected!");
    }
}
```

## Components

### 1. LogAnomalyDetector

The main entry point for log anomaly detection.

**Key Methods:**
- `new(config)` - Create new detector
- `process_logs(&logs)` - Process batch of logs
- `incremental_update(&logs, &labels)` - Train incrementally
- `save_checkpoint(&path)` - Save model state
- `load_checkpoint(&path)` - Load model state

### 2. BPETokenizer

Byte-pair encoding tokenizer optimized for log data.

**Features:**
- Custom vocabulary for log patterns
- Ericsson-specific tokenization
- Semi-structured text handling
- Efficient encoding/decoding

### 3. SlidingWindowAttention

Memory-efficient attention mechanism for long sequences.

**Optimizations:**
- Fixed window size for O(n) complexity
- Flash attention implementation
- Sparse attention patterns
- Custom CUDA kernels (when available)

### 4. EricssonLogParser

Specialized parser for Ericsson network logs.

**Supported Formats:**
- AMOS commands (alt, lget, cvc)
- Syslog RFC3164/RFC5424
- Custom structured formats
- Key-value pair extraction

### 5. AnomalyScorer

Multi-model ensemble for anomaly scoring.

**Models:**
- Autoencoder reconstruction error
- Isolation Forest
- One-Class SVM
- Statistical outlier detection

## Configuration

### DetectorConfig

```rust
pub struct DetectorConfig {
    pub sequence_length: usize,    // Max sequence length (default: 512)
    pub embedding_dim: usize,      // Embedding dimensions (default: 256)
    pub num_heads: usize,          // Attention heads (default: 8)
    pub num_layers: usize,         // Transformer layers (default: 6)
    pub hidden_dim: usize,         // Hidden layer size (default: 1024)
    pub window_size: usize,        // Attention window (default: 64)
    pub vocab_size: usize,         // Vocabulary size (default: 50000)
    pub quantization_bits: u8,     // Embedding quantization (default: 8)
    pub anomaly_threshold: f32,    // Detection threshold (default: 0.95)
    pub learning_rate: f32,        // Learning rate (default: 0.001)
    pub buffer_size: usize,        // Experience buffer (default: 10000)
}
```

## Examples

### Basic Anomaly Detection

```rust
use ran_opt::pfs_logs::{LogAnomalyDetector, DetectorConfig};

let detector = LogAnomalyDetector::new(DetectorConfig::default());

let logs = vec![
    "2024-01-04 10:15:23 INFO: Normal operation".to_string(),
    "2024-01-04 10:15:24 ERROR: Critical failure".to_string(),
];

let results = detector.process_logs(&logs);
assert!(results[1].score > results[0].score); // Error has higher score
```

### Incremental Learning

```rust
// Train on labeled data
let training_logs = vec![
    "2024-01-04 11:00:00 INFO: Normal log".to_string(),
    "2024-01-04 11:00:01 ERROR: Anomalous log".to_string(),
];
let labels = vec![false, true]; // false = normal, true = anomaly

detector.incremental_update(&training_logs, &labels);
```

### AMOS Command Processing

```rust
let amos_logs = vec![
    "2024-01-04 10:15:23 AMOS alt cell=12345 state=active power=85.2".to_string(),
    "2024-01-04 10:15:24 AMOS lget mo=RncFunction=1,UtranCell=12345".to_string(),
    "2024-01-04 10:15:25 AMOS cvc connect node=RBS_01 status=ok".to_string(),
];

let results = detector.process_logs(&amos_logs);

for result in results {
    if result.log_type == "AMOS" {
        println!("AMOS command detected: {:?}", result.detected_patterns);
    }
}
```

### Model Persistence

```rust
// Save model
detector.save_checkpoint("model.bin")?;

// Load model
let loaded_detector = LogAnomalyDetector::load_checkpoint("model.bin")?;
```

## Performance Optimizations

### Memory Efficiency
- Quantized embeddings reduce memory by 75%
- Sliding window attention: O(n) vs O(n²)
- Experience replay buffer with LRU eviction
- Streaming processing for large log files

### Speed Optimizations
- Custom SIMD attention kernels
- Batch processing with vectorization
- Sparse attention patterns
- Flash attention for long sequences

### Model Compression
- 8-bit quantization for embeddings
- Pruning of attention weights
- Knowledge distillation for smaller models
- Dynamic vocabulary pruning

## Benchmarks

Running the benchmarks:

```bash
cargo bench --bench pfs_logs_bench
```

Expected performance on modern hardware:
- **Processing**: >1000 logs/second
- **Memory**: <500MB for 50K vocabulary
- **Latency**: <10ms per log
- **Accuracy**: >95% on Ericsson test set

## Integration with Other Modules

### With DTM Power
```rust
// Monitor power-related log anomalies
let power_logs = extract_power_logs(&all_logs);
let power_anomalies = detector.process_logs(&power_logs);
dtm_power_manager.adjust_for_anomalies(&power_anomalies);
```

### With PFS Data
```rust
// Process compressed log archives
let compressed_logs = pfs_data.decompress_logs(&archive)?;
let anomalies = detector.process_logs(&compressed_logs);
```

### With AFM Detect
```rust
// Combine log and traffic anomalies
let log_anomalies = log_detector.process_logs(&logs);
let traffic_anomalies = afm_detector.detect(&traffic_data);
let combined_score = combine_anomaly_scores(&log_anomalies, &traffic_anomalies);
```

## Testing

Run all tests:
```bash
cargo test pfs_logs
```

Integration tests:
```bash
cargo test --test pfs_logs_integration_test
```

## Contributing

1. Ensure all tests pass
2. Add benchmarks for new features
3. Update documentation
4. Follow Rust coding standards
5. Add examples for new functionality

## License

This module is part of the RAN-Opt project and follows the same licensing terms.