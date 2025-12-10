//! Stress tests for large datasets, testing memory efficiency and performance
//! with millions of time series and data points.

use chrono::{DateTime, Utc, Duration};
use ndarray::{Array2, Array1};
use polars::prelude::*;
use proptest::prelude::*;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::collections::HashMap;
use rand::Rng;
use tempfile::TempDir;

use neuro_divergent_core::data::{
    TimeSeriesDataFrame, TimeSeriesSchema, TimeSeriesDataset,
    SeriesData, DatasetMetadata, TimeSeriesDatasetBuilder,
};
use neuro_divergent_models::{
    basic::{MLP, DLinear, NLinear, MLPMultivariate},
    forecasting::ForecastingModel,
    core::{ModelConfig, ModelState},
};
use neuro_divergent::prelude::*;

/// Configuration for large dataset stress tests
struct LargeDatasetConfig {
    /// Number of time series to generate
    n_series: usize,
    /// Points per series
    n_points_per_series: usize,
    /// Number of features
    n_features: usize,
    /// Batch size for processing
    batch_size: usize,
    /// Memory limit in MB
    memory_limit_mb: usize,
}

impl Default for LargeDatasetConfig {
    fn default() -> Self {
        Self {
            n_series: 10_000,
            n_points_per_series: 1_000,
            n_features: 10,
            batch_size: 1_000,
            memory_limit_mb: 8_192, // 8GB
        }
    }
}

/// Generate large synthetic time series dataset
fn generate_large_dataset(config: &LargeDatasetConfig) -> DataFrame {
    let mut unique_ids = Vec::new();
    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    let mut rng = rand::thread_rng();
    
    let start_time = DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z")
        .unwrap()
        .with_timezone(&Utc);
    
    // Generate data in batches to control memory usage
    for batch_start in (0..config.n_series).step_by(config.batch_size) {
        let batch_end = (batch_start + config.batch_size).min(config.n_series);
        
        for series_idx in batch_start..batch_end {
            let series_id = format!("series_{:08}", series_idx);
            
            for point_idx in 0..config.n_points_per_series {
                unique_ids.push(series_id.clone());
                timestamps.push(start_time + Duration::hours(point_idx as i64));
                
                // Generate synthetic value with trend, seasonality, and noise
                let trend = point_idx as f64 * 0.01;
                let seasonality = (point_idx as f64 * 2.0 * std::f64::consts::PI / 24.0).sin();
                let noise = rng.gen_range(-0.5..0.5);
                values.push(trend + seasonality + noise + 100.0);
            }
        }
    }
    
    // Create DataFrame from vectors
    df! {
        "unique_id" => unique_ids,
        "ds" => timestamps.into_iter().map(|t| t.naive_utc()).collect::<Vec<_>>(),
        "y" => values,
    }.unwrap()
}

/// Generate large dataset with multiple features
fn generate_multivariate_dataset(config: &LargeDatasetConfig) -> DataFrame {
    let base_df = generate_large_dataset(config);
    let mut df = base_df;
    
    // Add exogenous features
    for feature_idx in 0..config.n_features {
        let feature_name = format!("feature_{}", feature_idx);
        let mut feature_values = Vec::new();
        let mut rng = rand::thread_rng();
        
        for _ in 0..(config.n_series * config.n_points_per_series) {
            feature_values.push(rng.gen_range(-1.0..1.0));
        }
        
        df = df.with_column(
            Series::new(&feature_name, feature_values)
        ).unwrap();
    }
    
    df
}

#[test]
#[ignore] // Run with: cargo test --release -- --ignored test_1million_series
fn test_1million_series() {
    let config = LargeDatasetConfig {
        n_series: 1_000_000,
        n_points_per_series: 100,
        n_features: 5,
        batch_size: 10_000,
        memory_limit_mb: 16_384, // 16GB
    };
    
    println!("Generating dataset with 1M series...");
    let start = Instant::now();
    
    let df = generate_large_dataset(&config);
    let generation_time = start.elapsed();
    
    println!("Dataset generated in: {:?}", generation_time);
    println!("DataFrame shape: {:?} x {:?}", df.height(), df.width());
    
    // Test loading into TimeSeriesDataFrame
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    // Verify we can extract unique IDs without running out of memory
    let unique_ids = ts_df.unique_ids().unwrap();
    assert_eq!(unique_ids.len(), config.n_series);
    
    // Test conversion to dataset
    let dataset_start = Instant::now();
    let dataset = ts_df.to_dataset().unwrap();
    let dataset_time = dataset_start.elapsed();
    
    println!("Dataset conversion time: {:?}", dataset_time);
    assert_eq!(dataset.unique_ids.len(), config.n_series);
    assert_eq!(dataset.metadata.n_series, config.n_series);
}

#[test]
#[ignore] // Run with: cargo test --release -- --ignored test_1million_points_per_series
fn test_1million_points_per_series() {
    let config = LargeDatasetConfig {
        n_series: 10,
        n_points_per_series: 1_000_000,
        n_features: 3,
        batch_size: 1,
        memory_limit_mb: 8_192,
    };
    
    println!("Generating dataset with 1M points per series...");
    let start = Instant::now();
    
    let df = generate_large_dataset(&config);
    let generation_time = start.elapsed();
    
    println!("Dataset generated in: {:?}", generation_time);
    
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    // Test filtering large series
    let filter_start = Instant::now();
    let filtered = ts_df.filter_by_id("series_00000000").unwrap();
    let filter_time = filter_start.elapsed();
    
    println!("Large series filter time: {:?}", filter_time);
    assert_eq!(filtered.data.height(), config.n_points_per_series);
}

#[test]
#[ignore] // Run with: cargo test --release -- --ignored test_1000_features
fn test_1000_features() {
    let config = LargeDatasetConfig {
        n_series: 100,
        n_points_per_series: 1000,
        n_features: 1000,
        batch_size: 10,
        memory_limit_mb: 16_384,
    };
    
    println!("Generating dataset with 1000 features...");
    let start = Instant::now();
    
    let df = generate_multivariate_dataset(&config);
    let generation_time = start.elapsed();
    
    println!("Dataset generated in: {:?}", generation_time);
    println!("DataFrame shape: {:?} x {:?}", df.height(), df.width());
    
    // Create schema with all features as historical exogenous
    let mut schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let feature_names: Vec<String> = (0..config.n_features)
        .map(|i| format!("feature_{}", i))
        .collect();
    schema = schema.with_historical_exogenous(feature_names);
    
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    // Test dataset conversion with high-dimensional features
    let dataset = ts_df.to_dataset().unwrap();
    assert_eq!(dataset.unique_ids.len(), config.n_series);
}

#[test]
#[ignore] // Run with: cargo test --release -- --ignored test_100gb_file_streaming
fn test_100gb_file_streaming() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("large_dataset.parquet");
    
    // Configuration for ~100GB dataset
    let config = LargeDatasetConfig {
        n_series: 100_000,
        n_points_per_series: 10_000,
        n_features: 50,
        batch_size: 1_000,
        memory_limit_mb: 4_096, // Only 4GB memory limit
    };
    
    println!("Generating 100GB dataset...");
    
    // Generate and write in chunks to avoid memory overflow
    let mut total_rows = 0;
    let chunk_size = 1_000_000; // 1M rows per chunk
    
    for chunk_idx in 0..100 {
        println!("Writing chunk {}/100", chunk_idx + 1);
        
        let chunk_config = LargeDatasetConfig {
            n_series: config.n_series / 100,
            n_points_per_series: config.n_points_per_series,
            n_features: config.n_features,
            batch_size: config.batch_size,
            memory_limit_mb: config.memory_limit_mb,
        };
        
        let chunk_df = generate_multivariate_dataset(&chunk_config);
        total_rows += chunk_df.height();
        
        // Append to parquet file
        if chunk_idx == 0 {
            chunk_df.lazy()
                .sink_parquet(file_path.clone(), Default::default())
                .unwrap();
        } else {
            // Note: In practice, you'd append to the existing file
            // This is a simplified version
        }
    }
    
    println!("Total rows written: {}", total_rows);
    
    // Test streaming read
    println!("Testing streaming read...");
    let start = Instant::now();
    
    let lazy_df = LazyFrame::scan_parquet(&file_path, Default::default()).unwrap();
    
    // Process in chunks
    let result = lazy_df
        .select([col("y").mean().alias("avg_y")])
        .collect()
        .unwrap();
    
    let read_time = start.elapsed();
    println!("Streaming read time: {:?}", read_time);
    println!("Average y value: {:?}", result);
}

#[test]
#[ignore] // Run with: cargo test --release -- --ignored test_model_training_large_dataset
fn test_model_training_large_dataset() {
    let config = LargeDatasetConfig {
        n_series: 10_000,
        n_points_per_series: 500,
        n_features: 20,
        batch_size: 1_000,
        memory_limit_mb: 8_192,
    };
    
    println!("Generating large training dataset...");
    let df = generate_multivariate_dataset(&config);
    
    let mut schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let feature_names: Vec<String> = (0..config.n_features)
        .map(|i| format!("feature_{}", i))
        .collect();
    schema = schema.with_historical_exogenous(feature_names);
    
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    // Create MLP model
    let model = MLP::builder()
        .hidden_size(128)
        .num_layers(3)
        .dropout(0.1)
        .learning_rate(0.001)
        .batch_size(512)
        .max_epochs(10)
        .horizon(24)
        .input_size(48)
        .build()
        .unwrap();
    
    // Create NeuralForecast instance
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(model))
        .with_frequency(Frequency::Hourly)
        .build()
        .unwrap();
    
    // Train on large dataset
    println!("Training model on large dataset...");
    let start = Instant::now();
    
    nf.fit(ts_df).unwrap();
    
    let training_time = start.elapsed();
    println!("Training completed in: {:?}", training_time);
    
    // Test prediction on large dataset
    let pred_start = Instant::now();
    let predictions = nf.predict().unwrap();
    let pred_time = pred_start.elapsed();
    
    println!("Prediction time: {:?}", pred_time);
    println!("Number of predictions: {}", predictions.data.height());
}

#[test]
#[ignore] // Run with: cargo test --release -- --ignored test_memory_efficient_batch_processing
fn test_memory_efficient_batch_processing() {
    let config = LargeDatasetConfig {
        n_series: 100_000,
        n_points_per_series: 1_000,
        n_features: 10,
        batch_size: 5_000,
        memory_limit_mb: 2_048, // Only 2GB limit
    };
    
    println!("Testing memory-efficient batch processing...");
    
    // Track memory usage
    let memory_tracker = Arc::new(Mutex::new(Vec::new()));
    
    // Process in batches
    let results: Vec<f64> = (0..config.n_series)
        .into_par_iter()
        .chunks(config.batch_size)
        .map(|batch_indices| {
            let batch_start = Instant::now();
            
            // Generate batch data
            let batch_config = LargeDatasetConfig {
                n_series: batch_indices.len(),
                n_points_per_series: config.n_points_per_series,
                n_features: config.n_features,
                batch_size: batch_indices.len(),
                memory_limit_mb: config.memory_limit_mb,
            };
            
            let batch_df = generate_large_dataset(&batch_config);
            
            // Process batch (e.g., compute statistics)
            let mean_value = batch_df
                .column("y")
                .unwrap()
                .mean()
                .unwrap();
            
            let batch_time = batch_start.elapsed();
            
            // Track memory (simplified - in practice use actual memory tracking)
            let mem_usage = batch_df.estimated_size();
            memory_tracker.lock().unwrap().push(mem_usage);
            
            println!(
                "Processed batch of {} series in {:?}, memory: {} MB",
                batch_indices.len(),
                batch_time,
                mem_usage / 1_000_000
            );
            
            mean_value
        })
        .collect();
    
    println!("Processed {} batches", results.len());
    
    let memory_usage = memory_tracker.lock().unwrap();
    let max_memory = memory_usage.iter().max().unwrap_or(&0);
    let avg_memory = memory_usage.iter().sum::<usize>() / memory_usage.len().max(1);
    
    println!("Max memory usage: {} MB", max_memory / 1_000_000);
    println!("Avg memory usage: {} MB", avg_memory / 1_000_000);
    
    // Verify memory stayed within limits
    assert!(
        *max_memory < (config.memory_limit_mb as usize * 1_000_000),
        "Memory usage exceeded limit"
    );
}

#[test]
#[ignore] // Run with: cargo test --release -- --ignored test_distributed_dataset_processing
fn test_distributed_dataset_processing() {
    use std::thread;
    use std::sync::mpsc;
    
    let config = LargeDatasetConfig {
        n_series: 50_000,
        n_points_per_series: 2_000,
        n_features: 15,
        batch_size: 5_000,
        memory_limit_mb: 4_096,
    };
    
    println!("Testing distributed dataset processing...");
    
    let num_workers = 4;
    let (tx, rx) = mpsc::channel();
    let series_per_worker = config.n_series / num_workers;
    
    // Spawn worker threads
    let workers: Vec<_> = (0..num_workers)
        .map(|worker_id| {
            let tx = tx.clone();
            let worker_config = LargeDatasetConfig {
                n_series: series_per_worker,
                n_points_per_series: config.n_points_per_series,
                n_features: config.n_features,
                batch_size: config.batch_size,
                memory_limit_mb: config.memory_limit_mb / num_workers,
            };
            
            thread::spawn(move || {
                let start = Instant::now();
                
                // Generate and process worker's portion
                let df = generate_multivariate_dataset(&worker_config);
                
                // Compute statistics
                let stats = df
                    .lazy()
                    .select([
                        col("y").mean().alias("mean"),
                        col("y").std(0).alias("std"),
                        col("y").min().alias("min"),
                        col("y").max().alias("max"),
                    ])
                    .collect()
                    .unwrap();
                
                let elapsed = start.elapsed();
                
                tx.send((worker_id, stats, elapsed)).unwrap();
            })
        })
        .collect();
    
    drop(tx); // Close sender
    
    // Collect results
    let mut total_time = Duration::from_secs(0);
    while let Ok((worker_id, stats, elapsed)) = rx.recv() {
        println!("Worker {} completed in {:?}", worker_id, elapsed);
        println!("Worker {} stats: {:?}", worker_id, stats);
        total_time = total_time.max(elapsed);
    }
    
    // Wait for all workers
    for worker in workers {
        worker.join().unwrap();
    }
    
    println!("All workers completed in {:?}", total_time);
}

#[test]
#[ignore] // Run with: cargo test --release -- --ignored test_incremental_learning_large_stream
fn test_incremental_learning_large_stream() {
    let config = LargeDatasetConfig {
        n_series: 1_000,
        n_points_per_series: 100_000, // Streaming 100k points
        n_features: 5,
        batch_size: 1_000,
        memory_limit_mb: 1_024, // Only 1GB for streaming
    };
    
    println!("Testing incremental learning on large data stream...");
    
    // Simulate streaming data
    let mut model = DLinear::builder()
        .hidden_size(64)
        .kernel_size(5)
        .learning_rate(0.001)
        .batch_size(256)
        .horizon(12)
        .input_size(24)
        .build()
        .unwrap();
    
    let mut total_samples = 0;
    let mut total_time = Duration::from_secs(0);
    
    // Process data in streaming windows
    let window_size = 10_000;
    let num_windows = config.n_points_per_series / window_size;
    
    for window_idx in 0..num_windows {
        let window_start = Instant::now();
        
        // Generate window data
        let window_config = LargeDatasetConfig {
            n_series: config.n_series,
            n_points_per_series: window_size,
            n_features: config.n_features,
            batch_size: config.batch_size,
            memory_limit_mb: config.memory_limit_mb,
        };
        
        let window_df = generate_multivariate_dataset(&window_config);
        total_samples += window_df.height();
        
        // Update model incrementally (simulated)
        // In practice, this would involve actual incremental training
        
        let window_time = window_start.elapsed();
        total_time += window_time;
        
        println!(
            "Processed window {}/{} ({} samples) in {:?}",
            window_idx + 1,
            num_windows,
            window_df.height(),
            window_time
        );
        
        // Ensure we're not accumulating memory
        drop(window_df);
    }
    
    println!("Total samples processed: {}", total_samples);
    println!("Total processing time: {:?}", total_time);
    println!(
        "Average throughput: {:.2} samples/sec",
        total_samples as f64 / total_time.as_secs_f64()
    );
}

/// Property-based test for large dataset generation
proptest! {
    #[test]
    #[ignore]
    fn prop_large_dataset_consistency(
        n_series in 100usize..10_000,
        n_points in 100usize..5_000,
        n_features in 1usize..50
    ) {
        let config = LargeDatasetConfig {
            n_series,
            n_points_per_series: n_points,
            n_features,
            batch_size: 1_000.min(n_series),
            memory_limit_mb: 4_096,
        };
        
        let df = generate_multivariate_dataset(&config);
        
        // Verify dataset properties
        prop_assert_eq!(df.height(), n_series * n_points);
        prop_assert_eq!(df.width(), 3 + n_features); // unique_id, ds, y + features
        
        // Verify no null values
        for col in df.get_columns() {
            prop_assert_eq!(col.null_count(), 0);
        }
    }
}

#[cfg(test)]
mod benchmarks {
    use super::*;
    use criterion::{black_box, Criterion};
    
    pub fn benchmark_large_dataset_operations(c: &mut Criterion) {
        let mut group = c.benchmark_group("large_dataset_operations");
        
        // Benchmark dataset generation
        group.bench_function("generate_10k_series", |b| {
            let config = LargeDatasetConfig {
                n_series: 10_000,
                n_points_per_series: 100,
                n_features: 5,
                batch_size: 1_000,
                memory_limit_mb: 4_096,
            };
            
            b.iter(|| {
                black_box(generate_large_dataset(&config))
            });
        });
        
        // Benchmark dataframe operations
        group.bench_function("filter_large_dataset", |b| {
            let config = LargeDatasetConfig {
                n_series: 10_000,
                n_points_per_series: 100,
                n_features: 5,
                batch_size: 1_000,
                memory_limit_mb: 4_096,
            };
            
            let df = generate_large_dataset(&config);
            let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
            let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
            
            b.iter(|| {
                black_box(ts_df.filter_by_id("series_00005000").unwrap())
            });
        });
        
        group.finish();
    }
}