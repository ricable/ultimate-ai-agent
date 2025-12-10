//! Memory usage profiling tests for neuro-divergent
//!
//! This module contains tests that profile memory usage patterns across different
//! components of the library to identify memory leaks, excessive allocations,
//! and optimization opportunities.

use neuro_divergent::models::*;
use neuro_divergent::data::*;
use neuro_divergent::training::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::collections::HashMap;
use std::fmt;

/// Custom allocator to track memory allocations
struct TrackingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK_MEMORY: AtomicUsize = AtomicUsize::new(0);
static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ret = System.alloc(layout);
        if !ret.is_null() {
            let size = layout.size();
            ALLOCATED.fetch_add(size, Ordering::SeqCst);
            ALLOCATION_COUNT.fetch_add(1, Ordering::SeqCst);
            
            // Update peak memory
            let current = ALLOCATED.load(Ordering::SeqCst) - DEALLOCATED.load(Ordering::SeqCst);
            let mut peak = PEAK_MEMORY.load(Ordering::SeqCst);
            while current > peak {
                match PEAK_MEMORY.compare_exchange_weak(
                    peak,
                    current,
                    Ordering::SeqCst,
                    Ordering::SeqCst
                ) {
                    Ok(_) => break,
                    Err(p) => peak = p,
                }
            }
        }
        ret
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        DEALLOCATED.fetch_add(layout.size(), Ordering::SeqCst);
        DEALLOCATION_COUNT.fetch_add(1, Ordering::SeqCst);
    }
}

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

/// Memory statistics structure
#[derive(Debug, Clone)]
struct MemoryStats {
    allocated_bytes: usize,
    deallocated_bytes: usize,
    current_usage: usize,
    peak_usage: usize,
    allocation_count: usize,
    deallocation_count: usize,
}

impl MemoryStats {
    fn capture() -> Self {
        let allocated = ALLOCATED.load(Ordering::SeqCst);
        let deallocated = DEALLOCATED.load(Ordering::SeqCst);
        Self {
            allocated_bytes: allocated,
            deallocated_bytes: deallocated,
            current_usage: allocated.saturating_sub(deallocated),
            peak_usage: PEAK_MEMORY.load(Ordering::SeqCst),
            allocation_count: ALLOCATION_COUNT.load(Ordering::SeqCst),
            deallocation_count: DEALLOCATION_COUNT.load(Ordering::SeqCst),
        }
    }
    
    fn reset() {
        ALLOCATED.store(0, Ordering::SeqCst);
        DEALLOCATED.store(0, Ordering::SeqCst);
        PEAK_MEMORY.store(0, Ordering::SeqCst);
        ALLOCATION_COUNT.store(0, Ordering::SeqCst);
        DEALLOCATION_COUNT.store(0, Ordering::SeqCst);
    }
    
    fn diff(&self, other: &Self) -> Self {
        Self {
            allocated_bytes: self.allocated_bytes - other.allocated_bytes,
            deallocated_bytes: self.deallocated_bytes - other.deallocated_bytes,
            current_usage: self.current_usage.saturating_sub(other.current_usage),
            peak_usage: self.peak_usage.max(other.peak_usage),
            allocation_count: self.allocation_count - other.allocation_count,
            deallocation_count: self.deallocation_count - other.deallocation_count,
        }
    }
}

impl fmt::Display for MemoryStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Memory Stats:\n")?;
        write!(f, "  Current Usage: {:.2} MB\n", self.current_usage as f64 / 1_048_576.0)?;
        write!(f, "  Peak Usage: {:.2} MB\n", self.peak_usage as f64 / 1_048_576.0)?;
        write!(f, "  Total Allocated: {:.2} MB\n", self.allocated_bytes as f64 / 1_048_576.0)?;
        write!(f, "  Total Deallocated: {:.2} MB\n", self.deallocated_bytes as f64 / 1_048_576.0)?;
        write!(f, "  Allocations: {}\n", self.allocation_count)?;
        write!(f, "  Deallocations: {}\n", self.deallocation_count)?;
        write!(f, "  Memory Leaked: {:.2} MB", 
               (self.allocated_bytes.saturating_sub(self.deallocated_bytes)) as f64 / 1_048_576.0)
    }
}

/// Macro to profile memory usage of a code block
macro_rules! profile_memory {
    ($name:expr, $code:block) => {{
        let start_stats = MemoryStats::capture();
        let result = $code;
        let end_stats = MemoryStats::capture();
        let diff_stats = end_stats.diff(&start_stats);
        println!("\n=== Memory Profile: {} ===", $name);
        println!("{}", diff_stats);
        (result, diff_stats)
    }};
}

#[test]
fn test_model_creation_memory() {
    MemoryStats::reset();
    
    println!("\n===== MODEL CREATION MEMORY USAGE =====\n");
    
    // Test MLP creation
    let (_, mlp_stats) = profile_memory!("MLP Creation", {
        let config = MLPConfig::new(100, 50)
            .with_hidden_layers(vec![256, 128, 64]);
        MLP::new(config).unwrap()
    });
    
    // Test LSTM creation
    let (_, lstm_stats) = profile_memory!("LSTM Creation", {
        let config = LSTMConfig::new(100, 50)
            .with_hidden_size(256)
            .with_num_layers(3);
        LSTM::new(config).unwrap()
    });
    
    // Test NBEATS creation
    let (_, nbeats_stats) = profile_memory!("NBEATS Creation", {
        let config = NBEATSConfig::new(100, 50)
            .with_num_stacks(3)
            .with_num_blocks(3)
            .with_layer_size(256);
        NBEATS::new(config).unwrap()
    });
    
    // Test TFT creation
    let (_, tft_stats) = profile_memory!("TFT Creation", {
        let config = TFTConfig::new(100, 50)
            .with_hidden_size(256)
            .with_num_heads(8);
        TFT::new(config).unwrap()
    });
    
    // Assert reasonable memory usage
    assert!(mlp_stats.peak_usage < 10_000_000, "MLP uses too much memory");
    assert!(lstm_stats.peak_usage < 20_000_000, "LSTM uses too much memory");
    assert!(nbeats_stats.peak_usage < 30_000_000, "NBEATS uses too much memory");
    assert!(tft_stats.peak_usage < 40_000_000, "TFT uses too much memory");
}

#[test]
fn test_training_memory_usage() {
    MemoryStats::reset();
    
    println!("\n===== TRAINING MEMORY USAGE =====\n");
    
    // Generate training data
    let training_data = generate_training_data(100, 1000, 5);
    
    // Test training with different batch sizes
    let batch_sizes = vec![8, 32, 128];
    
    for batch_size in batch_sizes {
        let (_, stats) = profile_memory!(format!("Training with batch_size={}", batch_size), {
            let config = MLPConfig::new(24, 12)
                .with_hidden_layers(vec![64, 32]);
            let mut model = MLP::new(config).unwrap();
            
            let mut trainer = Trainer::new()
                .with_optimizer(OptimizerType::Adam(Adam::new(0.001, 0.9, 0.999)))
                .with_loss_function(MSELoss::new())
                .build();
            
            let training_config = TrainingConfig {
                max_epochs: 5,
                batch_size,
                validation_frequency: 5,
                patience: None,
                gradient_clip: Some(1.0),
                mixed_precision: false,
                seed: Some(42),
                device: DeviceConfig::Cpu { num_threads: None },
                checkpoint: CheckpointConfig {
                    enabled: false,
                    save_frequency: 10,
                    keep_best_only: true,
                    monitor_metric: "loss".to_string(),
                    mode: CheckpointMode::Min,
                },
            };
            
            trainer.train(&mut model, &training_data, &training_config).unwrap()
        });
        
        // Memory should scale somewhat with batch size but not linearly
        let expected_max = 50_000_000 + (batch_size as usize * 100_000);
        assert!(stats.peak_usage < expected_max, 
                "Training with batch_size={} uses too much memory", batch_size);
    }
}

#[test]
fn test_data_processing_memory() {
    MemoryStats::reset();
    
    println!("\n===== DATA PROCESSING MEMORY USAGE =====\n");
    
    let data_sizes = vec![1000, 10000, 100000];
    
    for size in data_sizes {
        // Test preprocessing pipeline
        let (_, stats) = profile_memory!(format!("Preprocessing {} samples", size), {
            let data: Vec<f64> = (0..size)
                .map(|i| (i as f64 * 0.1).sin() * 10.0 + 50.0)
                .collect();
            
            // Standard scaling
            let mut scaler = StandardScaler::default();
            scaler.fit(&data).unwrap();
            let scaled = scaler.transform(&data).unwrap();
            
            // Feature engineering
            let lags = vec![1, 2, 3, 6, 12, 24];
            let lag_features = generate_lag_features(&scaled, &lags);
            
            // Rolling statistics
            let windows = vec![7, 14, 30];
            let mut rolling_features = Vec::new();
            for window in windows {
                rolling_features.push(rolling_mean(&scaled, window));
                rolling_features.push(rolling_std(&scaled, window));
            }
            
            (scaled, lag_features, rolling_features)
        });
        
        // Memory usage should be proportional to data size
        let expected_max = size * 1000; // Rough estimate
        assert!(stats.peak_usage < expected_max,
                "Preprocessing {} samples uses too much memory", size);
    }
}

#[test]
fn test_memory_leak_detection() {
    MemoryStats::reset();
    
    println!("\n===== MEMORY LEAK DETECTION =====\n");
    
    // Run multiple iterations to detect leaks
    let iterations = 10;
    let mut iteration_stats = Vec::new();
    
    for i in 0..iterations {
        let start_stats = MemoryStats::capture();
        
        // Create and destroy models repeatedly
        for _ in 0..5 {
            let config = MLPConfig::new(50, 25)
                .with_hidden_layers(vec![64, 32]);
            let model = MLP::new(config).unwrap();
            
            // Make some predictions
            let input = vec![0.5; 50];
            let _ = model.predict(&input);
            
            // Model should be dropped here
        }
        
        let end_stats = MemoryStats::capture();
        let diff = end_stats.diff(&start_stats);
        iteration_stats.push(diff.current_usage);
        
        println!("Iteration {}: Memory delta = {} bytes", i + 1, diff.current_usage);
    }
    
    // Check that memory usage is stable (no significant growth)
    let first_half_avg: f64 = iteration_stats[..iterations/2].iter()
        .map(|&x| x as f64).sum::<f64>() / (iterations/2) as f64;
    let second_half_avg: f64 = iteration_stats[iterations/2..].iter()
        .map(|&x| x as f64).sum::<f64>() / (iterations/2) as f64;
    
    let growth_rate = (second_half_avg - first_half_avg) / first_half_avg;
    assert!(growth_rate.abs() < 0.1, "Potential memory leak detected: {:.2}% growth", growth_rate * 100.0);
}

#[test]
fn test_batch_processing_memory_efficiency() {
    MemoryStats::reset();
    
    println!("\n===== BATCH PROCESSING MEMORY EFFICIENCY =====\n");
    
    let total_samples = 10000;
    let batch_sizes = vec![100, 500, 1000, 2000];
    
    for batch_size in batch_sizes {
        let (_, stats) = profile_memory!(format!("Batch processing with size={}", batch_size), {
            let dataset = generate_large_dataset(total_samples);
            
            let loader = DataLoader::new(dataset)
                .with_batch_size(batch_size)
                .with_shuffle(false);
            
            let mut processed = 0;
            for batch in loader {
                // Simulate processing
                let _sum: f64 = batch.iter()
                    .flat_map(|series| series.values())
                    .sum();
                processed += batch.len();
            }
            
            processed
        });
        
        // Memory usage should not scale linearly with batch size
        // It should be relatively constant + batch overhead
        let base_memory = 5_000_000; // 5MB base
        let per_batch_memory = batch_size * 1000; // 1KB per sample in batch
        let expected_max = base_memory + per_batch_memory;
        
        assert!(stats.peak_usage < expected_max,
                "Batch processing with size={} uses too much memory", batch_size);
    }
}

#[test]
fn test_model_parameter_memory() {
    MemoryStats::reset();
    
    println!("\n===== MODEL PARAMETER MEMORY =====\n");
    
    // Test different model sizes
    let layer_configs = vec![
        ("tiny", vec![32, 16]),
        ("small", vec![128, 64]),
        ("medium", vec![256, 128, 64]),
        ("large", vec![512, 256, 128, 64]),
    ];
    
    for (size_name, layers) in layer_configs {
        let (model, stats) = profile_memory!(format!("MLP {} model", size_name), {
            let config = MLPConfig::new(100, 10)
                .with_hidden_layers(layers.clone());
            MLP::new(config).unwrap()
        });
        
        // Calculate expected memory based on parameter count
        let param_count = model.count_parameters();
        let expected_memory = param_count * std::mem::size_of::<f64>();
        let overhead_factor = 2.0; // Allow 2x for overhead
        
        println!("  Parameters: {}", param_count);
        println!("  Expected memory: {:.2} MB", expected_memory as f64 / 1_048_576.0);
        println!("  Actual peak: {:.2} MB", stats.peak_usage as f64 / 1_048_576.0);
        
        assert!(stats.peak_usage < (expected_memory as f64 * overhead_factor) as usize,
                "Model {} uses more memory than expected", size_name);
    }
}

/// Helper function to generate training data
fn generate_training_data(
    num_series: usize,
    length: usize,
    features: usize,
) -> TrainingData<f64> {
    let mut inputs = Vec::with_capacity(num_series);
    let mut targets = Vec::with_capacity(num_series);
    let mut metadata = Vec::with_capacity(num_series);
    
    for series_idx in 0..num_series {
        let mut series_inputs = Vec::with_capacity(length);
        let mut series_targets = Vec::with_capacity(length);
        
        for t in 0..length {
            let mut input_features = Vec::with_capacity(features);
            for f in 0..features {
                let value = ((t + series_idx * 10) as f64 * 0.1 + f as f64).sin() * 10.0 + 50.0;
                input_features.push(vec![value]);
            }
            series_inputs.push(input_features);
            
            let target = ((t + series_idx * 10) as f64 * 0.15).sin() * 15.0 + 60.0;
            series_targets.push(vec![vec![target]]);
        }
        
        inputs.push(series_inputs);
        targets.push(series_targets);
        
        metadata.push(TimeSeriesMetadata {
            id: format!("series_{}", series_idx),
            frequency: "H".to_string(),
            seasonal_periods: vec![24, 168],
            scale: Some(1.0),
        });
    }
    
    TrainingData {
        inputs,
        targets,
        exogenous: None,
        static_features: None,
        metadata,
    }
}

/// Helper function to generate a large dataset
fn generate_large_dataset(num_samples: usize) -> TimeSeriesDataset<f64> {
    use chrono::{TimeZone, Utc};
    
    let mut dataset = TimeSeriesDataset::new();
    let samples_per_series = 100;
    let num_series = num_samples / samples_per_series;
    
    for s in 0..num_series {
        let timestamps: Vec<_> = (0..samples_per_series)
            .map(|i| Utc.ymd_opt(2023, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap() + chrono::Duration::hours(i as i64))
            .collect();
        
        let values: Vec<f64> = (0..samples_per_series)
            .map(|i| ((i + s * 10) as f64 * 0.1).sin() * 10.0 + 50.0)
            .collect();
        
        let series = TimeSeriesDatasetBuilder::new(format!("series_{}", s))
            .with_frequency("H".to_string())
            .with_values(values)
            .with_timestamps(timestamps)
            .build()
            .unwrap();
        
        dataset.add_series(series);
    }
    
    dataset
}

#[test]
fn test_memory_pool_efficiency() {
    MemoryStats::reset();
    
    println!("\n===== MEMORY POOL EFFICIENCY =====\n");
    
    // Test repeated allocations with pooling
    let (_, pooled_stats) = profile_memory!("Pooled allocations", {
        let mut pool = Vec::with_capacity(100);
        
        // Pre-allocate buffers
        for _ in 0..100 {
            pool.push(Vec::<f64>::with_capacity(1000));
        }
        
        // Reuse buffers
        for _ in 0..1000 {
            let mut buffer = pool.pop().unwrap_or_else(|| Vec::with_capacity(1000));
            buffer.clear();
            
            // Use buffer
            for i in 0..1000 {
                buffer.push(i as f64);
            }
            
            // Return to pool
            if pool.len() < 100 {
                pool.push(buffer);
            }
        }
    });
    
    // Test without pooling
    let (_, unpooled_stats) = profile_memory!("Unpooled allocations", {
        for _ in 0..1000 {
            let mut buffer = Vec::with_capacity(1000);
            
            // Use buffer
            for i in 0..1000 {
                buffer.push(i as f64);
            }
            
            // Buffer dropped here
        }
    });
    
    println!("Pooled allocations: {}", pooled_stats.allocation_count);
    println!("Unpooled allocations: {}", unpooled_stats.allocation_count);
    
    // Pooled should have significantly fewer allocations
    assert!(pooled_stats.allocation_count < unpooled_stats.allocation_count / 5,
            "Memory pooling not effective");
}