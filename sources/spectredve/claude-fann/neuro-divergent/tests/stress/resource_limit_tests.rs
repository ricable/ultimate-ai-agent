//! Resource limit stress tests for memory, CPU, file handles, and other system resources.
//! Tests graceful degradation and recovery under resource constraints.

use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};
use std::thread;
use std::time::{Duration, Instant};
use std::fs::{File, OpenOptions};
use std::io::{Write, BufWriter};
use std::collections::HashMap;
use std::path::PathBuf;
use tempfile::{TempDir, NamedTempFile};
use sysinfo::{System, SystemExt};
use chrono::{DateTime, Utc};
use ndarray::Array2;
use polars::prelude::*;
use rayon::prelude::*;

use neuro_divergent_core::data::{
    TimeSeriesDataFrame, TimeSeriesSchema, TimeSeriesDataset,
};
use neuro_divergent_models::{
    basic::{MLP, DLinear},
    forecasting::ForecastingModel,
    core::ModelConfig,
};
use neuro_divergent::prelude::*;

/// Monitor system resources during test execution
struct ResourceMonitor {
    initial_memory: usize,
    peak_memory: AtomicUsize,
    initial_handles: usize,
    peak_handles: AtomicUsize,
    monitoring: Arc<Mutex<bool>>,
}

impl ResourceMonitor {
    fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        
        let initial_memory = system.used_memory() as usize * 1024; // Convert from KB to bytes
        
        Self {
            initial_memory,
            peak_memory: AtomicUsize::new(initial_memory),
            initial_handles: count_file_handles(),
            peak_handles: AtomicUsize::new(count_file_handles()),
            monitoring: Arc::new(Mutex::new(true)),
        }
    }
    
    fn start_monitoring(&self) -> thread::JoinHandle<()> {
        let peak_memory = Arc::new(self.peak_memory.clone());
        let peak_handles = Arc::new(self.peak_handles.clone());
        let monitoring = self.monitoring.clone();
        
        thread::spawn(move || {
            let mut system = System::new_all();
            
            while *monitoring.lock().unwrap() {
                system.refresh_all();
                
                let current_memory = system.used_memory() as usize * 1024;
                peak_memory.fetch_max(current_memory, Ordering::Relaxed);
                
                let current_handles = count_file_handles();
                peak_handles.fetch_max(current_handles, Ordering::Relaxed);
                
                thread::sleep(Duration::from_millis(100));
            }
        })
    }
    
    fn stop_monitoring(&self) {
        *self.monitoring.lock().unwrap() = false;
    }
    
    fn get_report(&self) -> ResourceReport {
        let mut system = System::new_all();
        system.refresh_all();
        
        let current_memory = system.used_memory() as usize * 1024;
        
        ResourceReport {
            initial_memory_mb: self.initial_memory / 1_000_000,
            peak_memory_mb: self.peak_memory.load(Ordering::Relaxed) / 1_000_000,
            current_memory_mb: current_memory / 1_000_000,
            memory_increase_mb: current_memory.saturating_sub(self.initial_memory) / 1_000_000,
            initial_file_handles: self.initial_handles,
            peak_file_handles: self.peak_handles.load(Ordering::Relaxed),
            current_file_handles: count_file_handles(),
        }
    }
}

#[derive(Debug)]
struct ResourceReport {
    initial_memory_mb: usize,
    peak_memory_mb: usize,
    current_memory_mb: usize,
    memory_increase_mb: usize,
    initial_file_handles: usize,
    peak_file_handles: usize,
    current_file_handles: usize,
}

/// Count open file handles (Linux specific, returns 0 on other platforms)
fn count_file_handles() -> usize {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_dir("/proc/self/fd")
            .map(|entries| entries.count())
            .unwrap_or(0)
    }
    
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

/// Test memory exhaustion scenarios
#[test]
#[ignore] // Run with: cargo test --release -- --ignored test_memory_exhaustion
fn test_memory_exhaustion() {
    let monitor = ResourceMonitor::new();
    let _handle = monitor.start_monitoring();
    
    // Try to allocate increasingly large datasets until we approach memory limits
    let mut allocations = Vec::new();
    let mut allocation_size = 1_000_000; // Start with 1MB
    
    for i in 0..100 {
        println!("Allocation attempt {}: {} MB", i, allocation_size / 1_000_000);
        
        // Check current memory usage
        let report = monitor.get_report();
        println!("Current memory: {} MB, Peak: {} MB", 
                 report.current_memory_mb, report.peak_memory_mb);
        
        // Stop if we're using too much memory
        if report.current_memory_mb > 4096 { // 4GB limit
            println!("Approaching memory limit, stopping allocations");
            break;
        }
        
        // Try to allocate
        match generate_large_array(allocation_size) {
            Ok(array) => {
                allocations.push(array);
                allocation_size = (allocation_size as f64 * 1.5) as usize;
            }
            Err(e) => {
                println!("Allocation failed: {:?}", e);
                break;
            }
        }
    }
    
    monitor.stop_monitoring();
    let final_report = monitor.get_report();
    
    println!("Final resource report: {:?}", final_report);
    
    // Verify graceful handling
    assert!(final_report.peak_memory_mb < 5000); // Should not exceed reasonable limits
}

/// Generate large array with error handling
fn generate_large_array(size: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let elements = size / std::mem::size_of::<f64>();
    
    // Use try_reserve to handle allocation failure gracefully
    let mut vec = Vec::new();
    vec.try_reserve(elements)?;
    
    // Fill with data
    vec.extend((0..elements).map(|i| i as f64));
    
    Ok(vec)
}

/// Test file handle exhaustion
#[test]
#[ignore] // Run with: cargo test --release -- --ignored test_file_handle_exhaustion
fn test_file_handle_exhaustion() {
    let temp_dir = TempDir::new().unwrap();
    let monitor = ResourceMonitor::new();
    let _handle = monitor.start_monitoring();
    
    let mut files = Vec::new();
    let max_files = 10_000; // Try to open many files
    
    for i in 0..max_files {
        let file_path = temp_dir.path().join(format!("test_file_{}.txt", i));
        
        match File::create(&file_path) {
            Ok(file) => {
                files.push(file);
                
                if i % 100 == 0 {
                    let report = monitor.get_report();
                    println!("Opened {} files, current handles: {}", 
                             i, report.current_file_handles);
                }
            }
            Err(e) => {
                println!("Failed to create file {}: {:?}", i, e);
                break;
            }
        }
        
        // Check if we're approaching system limits
        let report = monitor.get_report();
        if report.current_file_handles > 5000 {
            println!("Approaching file handle limit");
            break;
        }
    }
    
    monitor.stop_monitoring();
    let final_report = monitor.get_report();
    
    println!("Opened {} files", files.len());
    println!("Peak file handles: {}", final_report.peak_file_handles);
    
    // Clean up
    drop(files);
    
    // Verify handles are released
    thread::sleep(Duration::from_millis(100));
    let cleanup_report = monitor.get_report();
    assert!(cleanup_report.current_file_handles < final_report.peak_file_handles);
}

/// Test thread pool saturation
#[test]
fn test_thread_pool_saturation() {
    let num_threads = 1000; // Try to create many threads
    let mut handles = Vec::new();
    
    let counter = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();
    
    for i in 0..num_threads {
        let counter_clone = counter.clone();
        
        match thread::Builder::new()
            .name(format!("worker-{}", i))
            .stack_size(1024 * 1024) // 1MB stack
            .spawn(move || {
                // Simulate work
                thread::sleep(Duration::from_millis(10));
                counter_clone.fetch_add(1, Ordering::Relaxed);
            }) {
            Ok(handle) => handles.push(handle),
            Err(e) => {
                println!("Failed to spawn thread {}: {:?}", i, e);
                break;
            }
        }
    }
    
    println!("Successfully spawned {} threads", handles.len());
    
    // Wait for completion
    for handle in handles {
        handle.join().unwrap();
    }
    
    let elapsed = start.elapsed();
    let completed = counter.load(Ordering::Relaxed);
    
    println!("Completed {} tasks in {:?}", completed, elapsed);
}

/// Test stack overflow scenarios
#[test]
fn test_stack_overflow_protection() {
    // Test deep recursion that could cause stack overflow
    fn recursive_computation(depth: usize, max_depth: usize) -> Result<f64, String> {
        if depth >= max_depth {
            return Ok(depth as f64);
        }
        
        // Check stack space (simplified)
        let stack_var = [0u8; 1024]; // 1KB on stack
        let _use_stack = stack_var[0]; // Prevent optimization
        
        // Catch potential stack overflow
        match std::panic::catch_unwind(|| {
            recursive_computation(depth + 1, max_depth)
        }) {
            Ok(result) => result,
            Err(_) => Err(format!("Stack overflow at depth {}", depth)),
        }
    }
    
    // Test various depths
    let test_depths = vec![100, 1_000, 10_000, 100_000];
    
    for max_depth in test_depths {
        println!("Testing recursion depth: {}", max_depth);
        
        match recursive_computation(0, max_depth) {
            Ok(result) => println!("Completed successfully: {}", result),
            Err(e) => println!("Failed as expected: {}", e),
        }
    }
}

/// Test disk space exhaustion
#[test]
#[ignore] // Run with: cargo test --release -- --ignored test_disk_space_exhaustion
fn test_disk_space_exhaustion() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("large_file.dat");
    
    // Try to write a very large file
    let chunk_size = 1024 * 1024 * 100; // 100MB chunks
    let max_size = 1024 * 1024 * 1024 * 10; // 10GB max
    
    match OpenOptions::new()
        .create(true)
        .write(true)
        .open(&file_path) {
        Ok(file) => {
            let mut writer = BufWriter::new(file);
            let chunk = vec![0u8; chunk_size];
            let mut total_written = 0;
            
            loop {
                match writer.write_all(&chunk) {
                    Ok(_) => {
                        total_written += chunk_size;
                        
                        if total_written % (chunk_size * 10) == 0 {
                            println!("Written {} MB", total_written / 1_000_000);
                        }
                        
                        if total_written >= max_size {
                            println!("Reached max size limit");
                            break;
                        }
                    }
                    Err(e) => {
                        println!("Write failed after {} MB: {:?}", 
                                 total_written / 1_000_000, e);
                        break;
                    }
                }
            }
            
            match writer.flush() {
                Ok(_) => println!("Successfully flushed {} MB", total_written / 1_000_000),
                Err(e) => println!("Flush failed: {:?}", e),
            }
        }
        Err(e) => {
            println!("Failed to create file: {:?}", e);
        }
    }
}

/// Test memory fragmentation
#[test]
fn test_memory_fragmentation() {
    let monitor = ResourceMonitor::new();
    let _handle = monitor.start_monitoring();
    
    // Create many small allocations and deallocations to cause fragmentation
    let iterations = 1000;
    let mut allocations = HashMap::new();
    
    for i in 0..iterations {
        // Allocate various sizes
        let size = (i % 100 + 1) * 1000;
        let data = vec![i as f64; size];
        allocations.insert(i, data);
        
        // Randomly deallocate some
        if i > 10 && i % 3 == 0 {
            allocations.remove(&(i - 10));
        }
        
        if i % 100 == 0 {
            let report = monitor.get_report();
            println!("Iteration {}: {} MB used, {} allocations active",
                     i, report.current_memory_mb, allocations.len());
        }
    }
    
    monitor.stop_monitoring();
    let final_report = monitor.get_report();
    println!("Final memory report: {:?}", final_report);
}

/// Test concurrent memory pressure
#[test]
fn test_concurrent_memory_pressure() {
    let num_threads = 8;
    let allocations_per_thread = 100;
    let allocation_size = 1_000_000; // 1MB per allocation
    
    let barrier = Arc::new(std::sync::Barrier::new(num_threads));
    let mut handles = Vec::new();
    
    for thread_id in 0..num_threads {
        let barrier_clone = barrier.clone();
        
        let handle = thread::spawn(move || {
            let mut local_allocations = Vec::new();
            
            // Wait for all threads to start
            barrier_clone.wait();
            
            for i in 0..allocations_per_thread {
                match generate_large_array(allocation_size) {
                    Ok(array) => {
                        local_allocations.push(array);
                        
                        if i % 10 == 0 {
                            println!("Thread {} allocated {} arrays", thread_id, i);
                        }
                    }
                    Err(e) => {
                        println!("Thread {} allocation failed: {:?}", thread_id, e);
                        break;
                    }
                }
                
                // Add some computation to stress CPU as well
                let sum: f64 = local_allocations.iter()
                    .flat_map(|arr| arr.iter())
                    .sum();
                
                if sum.is_nan() {
                    println!("Unexpected NaN in thread {}", thread_id);
                }
            }
            
            local_allocations.len()
        });
        
        handles.push(handle);
    }
    
    // Collect results
    let total_allocations: usize = handles.into_iter()
        .map(|h| h.join().unwrap())
        .sum();
    
    println!("Total successful allocations across all threads: {}", total_allocations);
}

/// Test model training under memory constraints
#[test]
#[ignore] // Run with: cargo test --release -- --ignored test_model_training_memory_limit
fn test_model_training_memory_limit() {
    let monitor = ResourceMonitor::new();
    let _handle = monitor.start_monitoring();
    
    // Generate dataset that will stress memory during training
    let n_series = 5000;
    let n_points = 1000;
    let n_features = 50;
    
    println!("Generating large dataset for training...");
    let df = generate_dataset_for_training(n_series, n_points, n_features);
    
    let mut schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let feature_names: Vec<String> = (0..n_features)
        .map(|i| format!("feature_{}", i))
        .collect();
    schema = schema.with_historical_exogenous(feature_names);
    
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    let initial_report = monitor.get_report();
    println!("Memory before training: {} MB", initial_report.current_memory_mb);
    
    // Create model with large architecture to stress memory
    let model = MLP::builder()
        .hidden_size(512)
        .num_layers(5)
        .dropout(0.1)
        .learning_rate(0.001)
        .batch_size(1024) // Large batch size
        .max_epochs(5)
        .horizon(24)
        .input_size(48)
        .build()
        .unwrap();
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(model))
        .with_frequency(Frequency::Hourly)
        .build()
        .unwrap();
    
    // Attempt training with memory monitoring
    println!("Starting model training...");
    let train_start = Instant::now();
    
    match nf.fit(ts_df) {
        Ok(_) => {
            let train_time = train_start.elapsed();
            println!("Training completed in {:?}", train_time);
            
            let post_train_report = monitor.get_report();
            println!("Memory after training: {} MB", post_train_report.current_memory_mb);
            println!("Peak memory during training: {} MB", post_train_report.peak_memory_mb);
        }
        Err(e) => {
            println!("Training failed: {:?}", e);
            let failure_report = monitor.get_report();
            println!("Memory at failure: {} MB", failure_report.current_memory_mb);
        }
    }
    
    monitor.stop_monitoring();
}

/// Helper to generate dataset for training tests
fn generate_dataset_for_training(n_series: usize, n_points: usize, n_features: usize) -> DataFrame {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    let mut unique_ids = Vec::new();
    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    let mut features = vec![Vec::new(); n_features];
    
    for series_idx in 0..n_series {
        let series_id = format!("series_{:04}", series_idx);
        
        for point_idx in 0..n_points {
            unique_ids.push(series_id.clone());
            timestamps.push((Utc::now() + chrono::Duration::hours(point_idx as i64)).naive_utc());
            values.push(rng.gen_range(0.0..100.0));
            
            for feat_idx in 0..n_features {
                features[feat_idx].push(rng.gen_range(-1.0..1.0));
            }
        }
    }
    
    let mut df = df! {
        "unique_id" => unique_ids,
        "ds" => timestamps,
        "y" => values,
    }.unwrap();
    
    for (feat_idx, feat_values) in features.into_iter().enumerate() {
        df = df.with_column(
            Series::new(&format!("feature_{}", feat_idx), feat_values)
        ).unwrap();
    }
    
    df
}

/// Test resource cleanup
#[test]
fn test_resource_cleanup() {
    let temp_dir = TempDir::new().unwrap();
    let monitor = ResourceMonitor::new();
    
    let initial_report = monitor.get_report();
    println!("Initial resources - Memory: {} MB, Handles: {}",
             initial_report.current_memory_mb, initial_report.current_file_handles);
    
    // Create resources in a scope
    {
        // Allocate memory
        let _large_vec = vec![0u8; 100_000_000]; // 100MB
        
        // Open files
        let mut files = Vec::new();
        for i in 0..10 {
            let path = temp_dir.path().join(format!("file_{}.txt", i));
            files.push(File::create(path).unwrap());
        }
        
        let during_report = monitor.get_report();
        println!("During allocation - Memory: {} MB, Handles: {}",
                 during_report.current_memory_mb, during_report.current_file_handles);
        
        // Resources should be active here
        assert!(during_report.current_memory_mb > initial_report.current_memory_mb);
    }
    
    // Force cleanup
    thread::sleep(Duration::from_millis(100));
    
    let final_report = monitor.get_report();
    println!("After cleanup - Memory: {} MB, Handles: {}",
             final_report.current_memory_mb, final_report.current_file_handles);
    
    // Verify resources were released
    assert!(final_report.current_memory_mb <= initial_report.current_memory_mb + 10);
}