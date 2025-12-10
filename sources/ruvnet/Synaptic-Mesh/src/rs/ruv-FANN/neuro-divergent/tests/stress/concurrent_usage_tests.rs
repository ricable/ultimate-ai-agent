//! Concurrent usage stress tests for thread safety, race conditions, and parallel processing.
//! Tests the library's behavior under heavy concurrent load.

use std::sync::{Arc, Mutex, RwLock, Barrier, atomic::{AtomicUsize, AtomicBool, Ordering}};
use std::thread;
use std::time::{Duration, Instant};
use std::collections::{HashMap, HashSet};
use crossbeam::channel::{bounded, unbounded};
use parking_lot::Mutex as ParkingMutex;
use rayon::prelude::*;
use chrono::{DateTime, Utc};
use polars::prelude::*;
use dashmap::DashMap;
use once_cell::sync::Lazy;

use neuro_divergent_core::data::{
    TimeSeriesDataFrame, TimeSeriesSchema, TimeSeriesDataset,
};
use neuro_divergent_models::{
    basic::{MLP, DLinear, NLinear},
    forecasting::ForecastingModel,
    core::{ModelConfig, ModelState},
};
use neuro_divergent::prelude::*;

/// Global model registry for concurrent access
static MODEL_REGISTRY: Lazy<DashMap<String, Arc<Mutex<Box<dyn ForecastingModel + Send + Sync>>>>> = 
    Lazy::new(|| DashMap::new());

/// Test concurrent model training
#[test]
fn test_concurrent_model_training() {
    let num_models = 10;
    let num_threads = 4;
    let barrier = Arc::new(Barrier::new(num_threads));
    let completed = Arc::new(AtomicUsize::new(0));
    
    // Generate shared dataset
    let df = generate_test_dataset(1000, 100, 5);
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let ts_df = Arc::new(TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap());
    
    let mut handles = Vec::new();
    
    for thread_id in 0..num_threads {
        let barrier_clone = barrier.clone();
        let completed_clone = completed.clone();
        let ts_df_clone = ts_df.clone();
        
        let handle = thread::spawn(move || {
            // Wait for all threads to start
            barrier_clone.wait();
            
            let models_per_thread = num_models / num_threads;
            let start_idx = thread_id * models_per_thread;
            
            for model_idx in start_idx..(start_idx + models_per_thread) {
                let model_name = format!("model_{}", model_idx);
                
                // Create different model types
                let model: Box<dyn ForecastingModel + Send + Sync> = match model_idx % 3 {
                    0 => Box::new(
                        MLP::builder()
                            .hidden_size(64)
                            .num_layers(2)
                            .horizon(12)
                            .input_size(24)
                            .build()
                            .unwrap()
                    ),
                    1 => Box::new(
                        DLinear::builder()
                            .hidden_size(32)
                            .kernel_size(3)
                            .horizon(12)
                            .input_size(24)
                            .build()
                            .unwrap()
                    ),
                    _ => Box::new(
                        NLinear::builder()
                            .hidden_size(48)
                            .horizon(12)
                            .input_size(24)
                            .build()
                            .unwrap()
                    ),
                };
                
                // Train model
                let mut nf = NeuralForecast::builder()
                    .with_model(model)
                    .with_frequency(Frequency::Hourly)
                    .build()
                    .unwrap();
                
                match nf.fit((*ts_df_clone).clone()) {
                    Ok(_) => {
                        completed_clone.fetch_add(1, Ordering::Relaxed);
                        println!("Thread {} trained {}", thread_id, model_name);
                    }
                    Err(e) => {
                        println!("Thread {} failed to train {}: {:?}", thread_id, model_name, e);
                    }
                }
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    let total_completed = completed.load(Ordering::Relaxed);
    println!("Successfully trained {} models concurrently", total_completed);
    assert_eq!(total_completed, num_models);
}

/// Test concurrent predictions
#[test]
fn test_concurrent_predictions() {
    // Train a model first
    let df = generate_test_dataset(100, 200, 3);
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    let model = DLinear::builder()
        .hidden_size(64)
        .kernel_size(5)
        .horizon(24)
        .input_size(48)
        .build()
        .unwrap();
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(model))
        .with_frequency(Frequency::Hourly)
        .build()
        .unwrap();
    
    nf.fit(ts_df.clone()).unwrap();
    
    // Share the trained model across threads
    let nf_arc = Arc::new(Mutex::new(nf));
    let num_threads = 8;
    let predictions_per_thread = 100;
    let barrier = Arc::new(Barrier::new(num_threads));
    
    let mut handles = Vec::new();
    let prediction_times = Arc::new(Mutex::new(Vec::new()));
    
    for thread_id in 0..num_threads {
        let nf_clone = nf_arc.clone();
        let barrier_clone = barrier.clone();
        let times_clone = prediction_times.clone();
        
        let handle = thread::spawn(move || {
            let mut local_times = Vec::new();
            
            // Wait for all threads to start
            barrier_clone.wait();
            
            for i in 0..predictions_per_thread {
                let start = Instant::now();
                
                // Lock and make prediction
                let mut nf = nf_clone.lock().unwrap();
                match nf.predict() {
                    Ok(_predictions) => {
                        let elapsed = start.elapsed();
                        local_times.push(elapsed);
                        
                        if i % 10 == 0 {
                            println!("Thread {} completed {} predictions", thread_id, i);
                        }
                    }
                    Err(e) => {
                        println!("Thread {} prediction failed: {:?}", thread_id, e);
                    }
                }
            }
            
            times_clone.lock().unwrap().extend(local_times);
        });
        
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let all_times = prediction_times.lock().unwrap();
    let avg_time = all_times.iter().sum::<Duration>() / all_times.len() as u32;
    println!("Average prediction time under concurrent load: {:?}", avg_time);
}

/// Test race conditions in model registry
#[test]
fn test_model_registry_race_conditions() {
    let num_operations = 1000;
    let num_threads = 10;
    let barrier = Arc::new(Barrier::new(num_threads));
    
    let mut handles = Vec::new();
    
    for thread_id in 0..num_threads {
        let barrier_clone = barrier.clone();
        
        let handle = thread::spawn(move || {
            barrier_clone.wait();
            
            for i in 0..num_operations {
                let key = format!("model_{}_{}", thread_id, i % 10);
                
                // Randomly choose operation
                match i % 4 {
                    0 => {
                        // Insert
                        let model = Box::new(
                            DLinear::builder()
                                .hidden_size(32)
                                .horizon(12)
                                .input_size(24)
                                .build()
                                .unwrap()
                        );
                        MODEL_REGISTRY.insert(
                            key.clone(), 
                            Arc::new(Mutex::new(model as Box<dyn ForecastingModel + Send + Sync>))
                        );
                    }
                    1 => {
                        // Read
                        if let Some(model) = MODEL_REGISTRY.get(&key) {
                            let _state = model.lock().unwrap().state();
                        }
                    }
                    2 => {
                        // Update
                        if let Some(model) = MODEL_REGISTRY.get(&key) {
                            // Simulate state update
                            let _model = model.lock().unwrap();
                        }
                    }
                    _ => {
                        // Remove
                        MODEL_REGISTRY.remove(&key);
                    }
                }
            }
        });
        
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("Registry size after concurrent operations: {}", MODEL_REGISTRY.len());
}

/// Test parallel data processing
#[test]
fn test_parallel_data_processing() {
    let n_series = 10_000;
    let n_points = 100;
    
    // Generate large dataset
    let df = generate_test_dataset(n_series, n_points, 5);
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    let start = Instant::now();
    
    // Process series in parallel
    let unique_ids = ts_df.unique_ids().unwrap();
    let results: Vec<_> = unique_ids
        .par_iter()
        .map(|series_id| {
            // Filter to single series
            let series_df = ts_df.filter_by_id(series_id).unwrap();
            
            // Perform some computation
            let dataset = series_df.to_dataset().unwrap();
            let series_data = &dataset.series_data[series_id];
            
            // Calculate statistics
            let mean = series_data.target_values.iter().sum::<f64>() / series_data.length as f64;
            let max = series_data.target_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min = series_data.target_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            
            (series_id.clone(), mean, min, max)
        })
        .collect();
    
    let elapsed = start.elapsed();
    
    println!("Processed {} series in parallel in {:?}", results.len(), elapsed);
    println!("Average time per series: {:?}", elapsed / n_series as u32);
}

/// Test concurrent model sharing
#[test]
fn test_concurrent_model_sharing() {
    // Create a shared model state
    let model = Arc::new(RwLock::new(
        MLP::builder()
            .hidden_size(128)
            .num_layers(3)
            .horizon(24)
            .input_size(48)
            .build()
            .unwrap()
    ));
    
    let num_readers = 10;
    let num_writers = 2;
    let operations = 100;
    
    let barrier = Arc::new(Barrier::new(num_readers + num_writers));
    let mut handles = Vec::new();
    
    // Spawn reader threads
    for reader_id in 0..num_readers {
        let model_clone = model.clone();
        let barrier_clone = barrier.clone();
        
        let handle = thread::spawn(move || {
            barrier_clone.wait();
            
            for i in 0..operations {
                let model = model_clone.read().unwrap();
                let config = model.config();
                
                // Simulate reading model state
                assert_eq!(config.horizon(), 24);
                
                if i % 20 == 0 {
                    println!("Reader {} completed {} reads", reader_id, i);
                }
                
                thread::sleep(Duration::from_micros(10));
            }
        });
        
        handles.push(handle);
    }
    
    // Spawn writer threads
    for writer_id in 0..num_writers {
        let model_clone = model.clone();
        let barrier_clone = barrier.clone();
        
        let handle = thread::spawn(move || {
            barrier_clone.wait();
            
            for i in 0..operations {
                let mut model = model_clone.write().unwrap();
                
                // Simulate updating model state
                // In practice, this would be updating weights, etc.
                let _state = model.state();
                
                if i % 20 == 0 {
                    println!("Writer {} completed {} writes", writer_id, i);
                }
                
                thread::sleep(Duration::from_millis(1));
            }
        });
        
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("Concurrent model sharing test completed successfully");
}

/// Test producer-consumer pattern for streaming data
#[test]
fn test_streaming_data_processing() {
    let (tx, rx) = bounded(100); // Bounded channel for backpressure
    let num_producers = 4;
    let num_consumers = 2;
    let data_per_producer = 1000;
    
    let barrier = Arc::new(Barrier::new(num_producers + num_consumers));
    let processed = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();
    
    // Spawn producers
    for producer_id in 0..num_producers {
        let tx_clone = tx.clone();
        let barrier_clone = barrier.clone();
        
        let handle = thread::spawn(move || {
            barrier_clone.wait();
            
            for i in 0..data_per_producer {
                let data = generate_streaming_data_point(producer_id, i);
                
                match tx_clone.send_timeout(data, Duration::from_secs(1)) {
                    Ok(_) => {
                        if i % 100 == 0 {
                            println!("Producer {} sent {} items", producer_id, i);
                        }
                    }
                    Err(e) => {
                        println!("Producer {} failed to send: {:?}", producer_id, e);
                    }
                }
            }
        });
        
        handles.push(handle);
    }
    
    drop(tx); // Close sender side
    
    // Spawn consumers
    for consumer_id in 0..num_consumers {
        let rx_clone = rx.clone();
        let barrier_clone = barrier.clone();
        let processed_clone = processed.clone();
        
        let handle = thread::spawn(move || {
            barrier_clone.wait();
            
            let mut local_count = 0;
            while let Ok(data) = rx_clone.recv() {
                // Process data
                let _result = process_streaming_data(data);
                local_count += 1;
                
                if local_count % 100 == 0 {
                    println!("Consumer {} processed {} items", consumer_id, local_count);
                }
            }
            
            processed_clone.fetch_add(local_count, Ordering::Relaxed);
        });
        
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let total_processed = processed.load(Ordering::Relaxed);
    let expected = num_producers * data_per_producer;
    println!("Total processed: {} (expected: {})", total_processed, expected);
    assert_eq!(total_processed, expected);
}

/// Test atomic operations for statistics collection
#[test]
fn test_atomic_statistics_collection() {
    use std::sync::atomic::{AtomicU64, AtomicBool};
    
    #[derive(Default)]
    struct Statistics {
        count: AtomicUsize,
        sum: AtomicU64,
        min: AtomicU64,
        max: AtomicU64,
    }
    
    let stats = Arc::new(Statistics::default());
    stats.min.store(u64::MAX, Ordering::Relaxed);
    
    let num_threads = 8;
    let values_per_thread = 10_000;
    let barrier = Arc::new(Barrier::new(num_threads));
    
    let mut handles = Vec::new();
    
    for thread_id in 0..num_threads {
        let stats_clone = stats.clone();
        let barrier_clone = barrier.clone();
        
        let handle = thread::spawn(move || {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            
            barrier_clone.wait();
            
            for _ in 0..values_per_thread {
                let value = rng.gen_range(0..1000);
                let value_u64 = value as u64;
                
                // Update statistics atomically
                stats_clone.count.fetch_add(1, Ordering::Relaxed);
                stats_clone.sum.fetch_add(value_u64, Ordering::Relaxed);
                
                // Update min
                let mut current_min = stats_clone.min.load(Ordering::Relaxed);
                loop {
                    if value_u64 >= current_min {
                        break;
                    }
                    match stats_clone.min.compare_exchange_weak(
                        current_min,
                        value_u64,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(x) => current_min = x,
                    }
                }
                
                // Update max
                let mut current_max = stats_clone.max.load(Ordering::Relaxed);
                loop {
                    if value_u64 <= current_max {
                        break;
                    }
                    match stats_clone.max.compare_exchange_weak(
                        current_max,
                        value_u64,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(x) => current_max = x,
                    }
                }
            }
        });
        
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let final_count = stats.count.load(Ordering::Relaxed);
    let final_sum = stats.sum.load(Ordering::Relaxed);
    let final_min = stats.min.load(Ordering::Relaxed);
    let final_max = stats.max.load(Ordering::Relaxed);
    
    println!("Statistics collected atomically:");
    println!("  Count: {}", final_count);
    println!("  Sum: {}", final_sum);
    println!("  Min: {}", final_min);
    println!("  Max: {}", final_max);
    println!("  Average: {:.2}", final_sum as f64 / final_count as f64);
    
    assert_eq!(final_count, num_threads * values_per_thread);
}

/// Test lock-free data structures
#[test]
fn test_lock_free_concurrent_access() {
    use crossbeam::queue::ArrayQueue;
    
    let queue = Arc::new(ArrayQueue::new(1000));
    let num_producers = 4;
    let num_consumers = 2;
    let items_per_producer = 1000;
    
    let produced = Arc::new(AtomicUsize::new(0));
    let consumed = Arc::new(AtomicUsize::new(0));
    let done = Arc::new(AtomicBool::new(false));
    
    let mut handles = Vec::new();
    
    // Producers
    for producer_id in 0..num_producers {
        let queue_clone = queue.clone();
        let produced_clone = produced.clone();
        
        let handle = thread::spawn(move || {
            for i in 0..items_per_producer {
                let item = (producer_id, i);
                
                // Retry on full queue
                while let Err(_) = queue_clone.push(item) {
                    thread::yield_now();
                }
                
                produced_clone.fetch_add(1, Ordering::Relaxed);
            }
        });
        
        handles.push(handle);
    }
    
    // Consumers
    for consumer_id in 0..num_consumers {
        let queue_clone = queue.clone();
        let consumed_clone = consumed.clone();
        let done_clone = done.clone();
        
        let handle = thread::spawn(move || {
            let mut local_count = 0;
            
            loop {
                if let Some((_producer_id, _item_id)) = queue_clone.pop() {
                    local_count += 1;
                    
                    if local_count % 100 == 0 {
                        println!("Consumer {} processed {} items", consumer_id, local_count);
                    }
                } else if done_clone.load(Ordering::Relaxed) {
                    break;
                } else {
                    thread::yield_now();
                }
            }
            
            consumed_clone.fetch_add(local_count, Ordering::Relaxed);
        });
        
        handles.push(handle);
    }
    
    // Wait for producers
    for i in 0..num_producers {
        handles[i].join().unwrap();
    }
    
    // Signal completion
    done.store(true, Ordering::Relaxed);
    
    // Wait for consumers
    for i in num_producers..(num_producers + num_consumers) {
        handles[i].join().unwrap();
    }
    
    let total_produced = produced.load(Ordering::Relaxed);
    let total_consumed = consumed.load(Ordering::Relaxed);
    
    println!("Total produced: {}, Total consumed: {}", total_produced, total_consumed);
    assert_eq!(total_produced, total_consumed);
}

/// Test deadlock prevention
#[test]
fn test_deadlock_prevention() {
    let lock1 = Arc::new(ParkingMutex::new(0));
    let lock2 = Arc::new(ParkingMutex::new(0));
    
    let num_threads = 4;
    let operations = 100;
    let completed = Arc::new(AtomicUsize::new(0));
    
    let mut handles = Vec::new();
    
    for thread_id in 0..num_threads {
        let lock1_clone = lock1.clone();
        let lock2_clone = lock2.clone();
        let completed_clone = completed.clone();
        
        let handle = thread::spawn(move || {
            for i in 0..operations {
                // Use consistent lock ordering to prevent deadlock
                if thread_id % 2 == 0 {
                    // Always acquire lock1 first, then lock2
                    let mut val1 = lock1_clone.lock();
                    let mut val2 = lock2_clone.lock();
                    
                    *val1 += 1;
                    *val2 += 1;
                    
                    drop(val2);
                    drop(val1);
                } else {
                    // Also acquire lock1 first (consistent ordering)
                    let mut val1 = lock1_clone.lock();
                    let mut val2 = lock2_clone.lock();
                    
                    *val1 -= 1;
                    *val2 -= 1;
                    
                    drop(val2);
                    drop(val1);
                }
                
                if i % 20 == 0 {
                    println!("Thread {} completed {} operations", thread_id, i);
                }
            }
            
            completed_clone.fetch_add(1, Ordering::Relaxed);
        });
        
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let total_completed = completed.load(Ordering::Relaxed);
    println!("All {} threads completed without deadlock", total_completed);
    assert_eq!(total_completed, num_threads);
}

/// Helper functions
fn generate_test_dataset(n_series: usize, n_points: usize, n_features: usize) -> DataFrame {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    let mut unique_ids = Vec::new();
    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    
    for series_idx in 0..n_series {
        let series_id = format!("series_{:04}", series_idx);
        
        for point_idx in 0..n_points {
            unique_ids.push(series_id.clone());
            timestamps.push((Utc::now() + chrono::Duration::hours(point_idx as i64)).naive_utc());
            values.push(rng.gen_range(0.0..100.0));
        }
    }
    
    df! {
        "unique_id" => unique_ids,
        "ds" => timestamps,
        "y" => values,
    }.unwrap()
}

#[derive(Debug, Clone)]
struct StreamingDataPoint {
    producer_id: usize,
    sequence: usize,
    timestamp: DateTime<Utc>,
    value: f64,
}

fn generate_streaming_data_point(producer_id: usize, sequence: usize) -> StreamingDataPoint {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    StreamingDataPoint {
        producer_id,
        sequence,
        timestamp: Utc::now(),
        value: rng.gen_range(0.0..100.0),
    }
}

fn process_streaming_data(data: StreamingDataPoint) -> f64 {
    // Simulate processing
    data.value * 2.0 + data.sequence as f64
}