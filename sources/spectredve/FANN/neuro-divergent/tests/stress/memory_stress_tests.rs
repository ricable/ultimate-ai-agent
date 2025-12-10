//! Memory stress tests for detecting leaks, excessive allocations, and memory usage patterns.
//! Tests memory efficiency and identifies potential memory-related issues.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use chrono::{DateTime, Utc};
use polars::prelude::*;
// use jemalloc_ctl::{stats, epoch}; // Optional - requires jemalloc feature

use neuro_divergent_core::data::{
    TimeSeriesDataFrame, TimeSeriesSchema, TimeSeriesDataset,
};
use neuro_divergent_models::{
    basic::{MLP, DLinear},
    forecasting::ForecastingModel,
    core::ModelConfig,
};
use neuro_divergent::prelude::*;

/// Custom allocator for tracking memory allocations
struct TrackingAllocator {
    allocated: AtomicUsize,
    deallocated: AtomicUsize,
    peak: AtomicUsize,
    allocations: AtomicUsize,
    deallocations: AtomicUsize,
}

impl TrackingAllocator {
    const fn new() -> Self {
        Self {
            allocated: AtomicUsize::new(0),
            deallocated: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            allocations: AtomicUsize::new(0),
            deallocations: AtomicUsize::new(0),
        }
    }
    
    fn current_usage(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
            .saturating_sub(self.deallocated.load(Ordering::Relaxed))
    }
    
    fn peak_usage(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }
    
    fn allocation_count(&self) -> usize {
        self.allocations.load(Ordering::Relaxed)
    }
    
    fn deallocation_count(&self) -> usize {
        self.deallocations.load(Ordering::Relaxed)
    }
    
    fn reset_stats(&self) {
        self.allocated.store(0, Ordering::Relaxed);
        self.deallocated.store(0, Ordering::Relaxed);
        self.peak.store(0, Ordering::Relaxed);
        self.allocations.store(0, Ordering::Relaxed);
        self.deallocations.store(0, Ordering::Relaxed);
    }
}

static ALLOCATOR: TrackingAllocator = TrackingAllocator::new();

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let ptr = System.alloc(layout);
        
        if !ptr.is_null() {
            self.allocated.fetch_add(size, Ordering::Relaxed);
            self.allocations.fetch_add(1, Ordering::Relaxed);
            
            // Update peak
            let current = self.current_usage();
            let mut peak = self.peak.load(Ordering::Relaxed);
            while current > peak {
                match self.peak.compare_exchange_weak(
                    peak,
                    current,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(p) => peak = p,
                }
            }
        }
        
        ptr
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let size = layout.size();
        System.dealloc(ptr, layout);
        self.deallocated.fetch_add(size, Ordering::Relaxed);
        self.deallocations.fetch_add(1, Ordering::Relaxed);
    }
}

/// Memory usage snapshot
#[derive(Debug, Clone)]
struct MemorySnapshot {
    timestamp: Instant,
    allocated_bytes: usize,
    peak_bytes: usize,
    allocation_count: usize,
    deallocation_count: usize,
    label: String,
}

/// Memory profiler for tracking usage over time
struct MemoryProfiler {
    snapshots: Arc<Mutex<Vec<MemorySnapshot>>>,
    sampling_interval: Duration,
    running: Arc<AtomicBool>,
}

impl MemoryProfiler {
    fn new(sampling_interval: Duration) -> Self {
        Self {
            snapshots: Arc::new(Mutex::new(Vec::new())),
            sampling_interval,
            running: Arc::new(AtomicBool::new(false)),
        }
    }
    
    fn start(&self) -> thread::JoinHandle<()> {
        self.running.store(true, Ordering::Relaxed);
        let snapshots = self.snapshots.clone();
        let running = self.running.clone();
        let interval = self.sampling_interval;
        
        thread::spawn(move || {
            while running.load(Ordering::Relaxed) {
                let snapshot = MemorySnapshot {
                    timestamp: Instant::now(),
                    allocated_bytes: ALLOCATOR.current_usage(),
                    peak_bytes: ALLOCATOR.peak_usage(),
                    allocation_count: ALLOCATOR.allocation_count(),
                    deallocation_count: ALLOCATOR.deallocation_count(),
                    label: "auto".to_string(),
                };
                
                snapshots.lock().unwrap().push(snapshot);
                thread::sleep(interval);
            }
        })
    }
    
    fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }
    
    fn add_snapshot(&self, label: &str) {
        let snapshot = MemorySnapshot {
            timestamp: Instant::now(),
            allocated_bytes: ALLOCATOR.current_usage(),
            peak_bytes: ALLOCATOR.peak_usage(),
            allocation_count: ALLOCATOR.allocation_count(),
            deallocation_count: ALLOCATOR.deallocation_count(),
            label: label.to_string(),
        };
        
        self.snapshots.lock().unwrap().push(snapshot);
    }
    
    fn get_report(&self) -> MemoryReport {
        let snapshots = self.snapshots.lock().unwrap().clone();
        
        if snapshots.is_empty() {
            return MemoryReport::default();
        }
        
        let peak_usage = snapshots.iter()
            .map(|s| s.peak_bytes)
            .max()
            .unwrap_or(0);
        
        let total_allocations = snapshots.last()
            .map(|s| s.allocation_count)
            .unwrap_or(0);
        
        let total_deallocations = snapshots.last()
            .map(|s| s.deallocation_count)
            .unwrap_or(0);
        
        let leaked_bytes = snapshots.last()
            .map(|s| s.allocated_bytes)
            .unwrap_or(0);
        
        MemoryReport {
            peak_usage_mb: peak_usage as f64 / 1_000_000.0,
            total_allocations,
            total_deallocations,
            leaked_bytes,
            leaked_mb: leaked_bytes as f64 / 1_000_000.0,
            snapshots,
        }
    }
}

#[derive(Debug, Default)]
struct MemoryReport {
    peak_usage_mb: f64,
    total_allocations: usize,
    total_deallocations: usize,
    leaked_bytes: usize,
    leaked_mb: f64,
    snapshots: Vec<MemorySnapshot>,
}

/// Test for memory leaks in data loading
#[test]
fn test_memory_leak_data_loading() {
    ALLOCATOR.reset_stats();
    let profiler = MemoryProfiler::new(Duration::from_millis(100));
    let _handle = profiler.start();
    
    profiler.add_snapshot("start");
    
    // Repeatedly load and drop data
    for i in 0..100 {
        let df = generate_test_dataset(100, 100);
        let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
        
        {
            let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
            let _ = ts_df.to_dataset();
        } // Data should be dropped here
        
        if i % 10 == 0 {
            profiler.add_snapshot(&format!("iteration_{}", i));
            println!("Iteration {}: Current memory: {:.2} MB", 
                     i, ALLOCATOR.current_usage() as f64 / 1_000_000.0);
        }
    }
    
    profiler.add_snapshot("end");
    profiler.stop();
    
    let report = profiler.get_report();
    println!("Memory Report:");
    println!("  Peak usage: {:.2} MB", report.peak_usage_mb);
    println!("  Total allocations: {}", report.total_allocations);
    println!("  Total deallocations: {}", report.total_deallocations);
    println!("  Leaked: {:.2} MB", report.leaked_mb);
    
    // Check for memory leaks
    let start_memory = report.snapshots.first().unwrap().allocated_bytes;
    let end_memory = report.snapshots.last().unwrap().allocated_bytes;
    let growth = end_memory.saturating_sub(start_memory);
    
    assert!(
        growth < 10_000_000, // Allow 10MB growth
        "Memory leak detected: {} MB growth", growth as f64 / 1_000_000.0
    );
}

/// Test for memory leaks in model training
#[test]
#[ignore] // Run with: cargo test --release -- --ignored test_memory_leak_model_training
fn test_memory_leak_model_training() {
    ALLOCATOR.reset_stats();
    let profiler = MemoryProfiler::new(Duration::from_millis(500));
    let _handle = profiler.start();
    
    profiler.add_snapshot("start");
    
    // Generate dataset once
    let df = generate_test_dataset(100, 500);
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    profiler.add_snapshot("data_loaded");
    
    // Train multiple models
    for i in 0..10 {
        println!("Training model {}", i);
        
        {
            let model = DLinear::builder()
                .hidden_size(64)
                .kernel_size(3)
                .learning_rate(0.001)
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
            
            let _ = nf.fit(ts_df.clone());
            let _ = nf.predict();
        } // Model should be dropped here
        
        profiler.add_snapshot(&format!("model_{}_complete", i));
        
        // Force garbage collection (if available)
        thread::sleep(Duration::from_millis(100));
    }
    
    profiler.add_snapshot("end");
    profiler.stop();
    
    let report = profiler.get_report();
    
    // Analyze memory growth pattern
    let labeled_snapshots: Vec<_> = report.snapshots.iter()
        .filter(|s| s.label != "auto")
        .collect();
    
    for window in labeled_snapshots.windows(2) {
        let prev = &window[0];
        let curr = &window[1];
        let growth = curr.allocated_bytes.saturating_sub(prev.allocated_bytes);
        
        println!("{} -> {}: {:.2} MB growth",
                 prev.label, curr.label,
                 growth as f64 / 1_000_000.0);
    }
    
    // Check overall memory growth
    let data_loaded_memory = labeled_snapshots.iter()
        .find(|s| s.label == "data_loaded")
        .unwrap()
        .allocated_bytes;
    
    let end_memory = report.snapshots.last().unwrap().allocated_bytes;
    let growth = end_memory.saturating_sub(data_loaded_memory);
    
    assert!(
        growth < 50_000_000, // Allow 50MB growth for model artifacts
        "Excessive memory growth in model training: {} MB", 
        growth as f64 / 1_000_000.0
    );
}

/// Test memory fragmentation
#[test]
fn test_memory_fragmentation() {
    ALLOCATOR.reset_stats();
    let mut allocations: VecDeque<Vec<u8>> = VecDeque::new();
    
    let initial_memory = ALLOCATOR.current_usage();
    
    // Create fragmentation pattern
    for i in 0..1000 {
        // Allocate various sizes
        let size = match i % 4 {
            0 => 1024,        // 1KB
            1 => 1024 * 10,   // 10KB
            2 => 1024 * 100,  // 100KB
            _ => 1024 * 1000, // 1MB
        };
        
        allocations.push_back(vec![0u8; size]);
        
        // Deallocate some to create holes
        if i > 10 && i % 3 == 0 {
            allocations.pop_front();
        }
        
        if i % 100 == 0 {
            println!("Iteration {}: {} allocations, {:.2} MB used",
                     i, allocations.len(),
                     ALLOCATOR.current_usage() as f64 / 1_000_000.0);
        }
    }
    
    let fragmented_memory = ALLOCATOR.current_usage();
    
    // Clear all allocations
    allocations.clear();
    thread::sleep(Duration::from_millis(100));
    
    let final_memory = ALLOCATOR.current_usage();
    let leaked = final_memory.saturating_sub(initial_memory);
    
    println!("Fragmentation test results:");
    println!("  Peak memory during fragmentation: {:.2} MB", 
             fragmented_memory as f64 / 1_000_000.0);
    println!("  Memory after cleanup: {:.2} MB",
             final_memory as f64 / 1_000_000.0);
    println!("  Leaked: {} bytes", leaked);
    
    assert!(leaked < 1_000_000, "Memory leak after fragmentation test");
}

/// Test memory usage with different batch sizes
#[test]
fn test_memory_usage_batch_sizes() {
    let batch_sizes = vec![1, 10, 100, 1000, 10000];
    let mut results = Vec::new();
    
    for &batch_size in &batch_sizes {
        ALLOCATOR.reset_stats();
        
        // Generate data in batches
        let total_series = 10_000;
        let points_per_series = 100;
        
        let start_memory = ALLOCATOR.current_usage();
        let mut peak_memory = 0;
        
        for batch_start in (0..total_series).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(total_series);
            let batch_series = batch_end - batch_start;
            
            let df = generate_test_dataset(batch_series, points_per_series);
            let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
            let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
            
            // Process batch
            let _ = ts_df.to_dataset();
            
            peak_memory = peak_memory.max(ALLOCATOR.current_usage());
            
            // Clear batch
            drop(ts_df);
        }
        
        let end_memory = ALLOCATOR.current_usage();
        
        results.push(BatchSizeResult {
            batch_size,
            peak_memory_mb: peak_memory as f64 / 1_000_000.0,
            final_memory_mb: end_memory as f64 / 1_000_000.0,
            memory_efficiency: (end_memory - start_memory) as f64 / (total_series * points_per_series) as f64,
        });
    }
    
    println!("Batch size memory usage results:");
    for result in &results {
        println!("  Batch size {}: Peak {:.2} MB, Final {:.2} MB, Efficiency {:.2} bytes/point",
                 result.batch_size,
                 result.peak_memory_mb,
                 result.final_memory_mb,
                 result.memory_efficiency);
    }
    
    // Verify larger batches use more peak memory
    for window in results.windows(2) {
        assert!(window[1].peak_memory_mb >= window[0].peak_memory_mb * 0.9);
    }
}

#[derive(Debug)]
struct BatchSizeResult {
    batch_size: usize,
    peak_memory_mb: f64,
    final_memory_mb: f64,
    memory_efficiency: f64,
}

/// Test memory pressure handling
#[test]
fn test_memory_pressure_handling() {
    ALLOCATOR.reset_stats();
    
    let mut allocations = Vec::new();
    let allocation_size = 10_000_000; // 10MB per allocation
    let mut allocation_failures = 0;
    
    println!("Testing memory pressure handling...");
    
    // Try to allocate until we hit limits
    for i in 0..1000 {
        // Try to allocate
        match std::panic::catch_unwind(|| {
            vec![0u8; allocation_size]
        }) {
            Ok(vec) => {
                allocations.push(vec);
                
                if i % 10 == 0 {
                    let current_mb = ALLOCATOR.current_usage() as f64 / 1_000_000.0;
                    println!("Allocated {} chunks, total: {:.2} MB", i + 1, current_mb);
                }
            }
            Err(_) => {
                allocation_failures += 1;
                println!("Allocation {} failed", i);
                
                if allocation_failures > 5 {
                    println!("Too many allocation failures, stopping");
                    break;
                }
            }
        }
        
        // Check if we're approaching limits
        let current_usage = ALLOCATOR.current_usage();
        if current_usage > 1_000_000_000 { // 1GB limit for test
            println!("Reached memory limit at {:.2} MB", current_usage as f64 / 1_000_000.0);
            break;
        }
    }
    
    let peak_memory = ALLOCATOR.peak_usage();
    println!("Peak memory usage: {:.2} MB", peak_memory as f64 / 1_000_000.0);
    
    // Test recovery - drop half the allocations
    let half = allocations.len() / 2;
    allocations.truncate(half);
    
    thread::sleep(Duration::from_millis(100));
    
    let after_release = ALLOCATOR.current_usage();
    println!("Memory after releasing half: {:.2} MB", after_release as f64 / 1_000_000.0);
    
    assert!(after_release < peak_memory);
}

/// Test cyclic reference detection
#[test]
fn test_cyclic_reference_memory() {
    use std::rc::Rc;
    use std::cell::RefCell;
    
    #[derive(Debug)]
    struct Node {
        value: i32,
        next: Option<Rc<RefCell<Node>>>,
    }
    
    ALLOCATOR.reset_stats();
    let initial_memory = ALLOCATOR.current_usage();
    
    // Create cyclic references
    {
        let node1 = Rc::new(RefCell::new(Node { value: 1, next: None }));
        let node2 = Rc::new(RefCell::new(Node { value: 2, next: None }));
        
        // Create cycle
        node1.borrow_mut().next = Some(node2.clone());
        node2.borrow_mut().next = Some(node1.clone());
        
        // Nodes go out of scope but memory might leak due to cycle
    }
    
    thread::sleep(Duration::from_millis(100));
    
    let after_cycle = ALLOCATOR.current_usage();
    let leaked = after_cycle.saturating_sub(initial_memory);
    
    println!("Memory leaked due to cyclic references: {} bytes", leaked);
    
    // Note: Some leakage is expected with Rc cycles
    // This test is more for demonstration
}

/// Test memory usage patterns during streaming
#[test]
fn test_streaming_memory_patterns() {
    use std::sync::mpsc;
    
    ALLOCATOR.reset_stats();
    let profiler = MemoryProfiler::new(Duration::from_millis(50));
    let _handle = profiler.start();
    
    let (tx, rx) = mpsc::channel();
    let stream_duration = Duration::from_secs(2);
    
    profiler.add_snapshot("start");
    
    // Producer thread
    let producer = thread::spawn(move || {
        let start = Instant::now();
        let mut count = 0;
        
        while start.elapsed() < stream_duration {
            let data = generate_test_dataset(10, 100);
            tx.send(data).unwrap();
            count += 1;
            thread::sleep(Duration::from_millis(10));
        }
        
        count
    });
    
    // Consumer thread
    let consumer = thread::spawn(move || {
        let mut count = 0;
        let mut buffer = VecDeque::with_capacity(10);
        
        while let Ok(data) = rx.recv_timeout(Duration::from_millis(100)) {
            buffer.push_back(data);
            
            // Process when buffer is full
            if buffer.len() >= 5 {
                let batch: Vec<_> = buffer.drain(..5).collect();
                // Simulate processing
                thread::sleep(Duration::from_millis(20));
                drop(batch);
            }
            
            count += 1;
        }
        
        // Process remaining
        drop(buffer);
        count
    });
    
    let produced = producer.join().unwrap();
    let consumed = consumer.join().unwrap();
    
    profiler.add_snapshot("end");
    profiler.stop();
    
    println!("Streaming test: Produced {}, Consumed {}", produced, consumed);
    
    let report = profiler.get_report();
    
    // Analyze memory pattern
    let memory_samples: Vec<_> = report.snapshots.iter()
        .map(|s| s.allocated_bytes as f64 / 1_000_000.0)
        .collect();
    
    if memory_samples.len() > 2 {
        let avg_memory = memory_samples.iter().sum::<f64>() / memory_samples.len() as f64;
        let max_memory = memory_samples.iter().fold(0.0, |a, &b| a.max(b));
        let min_memory = memory_samples.iter().fold(f64::MAX, |a, &b| a.min(b));
        
        println!("Memory statistics during streaming:");
        println!("  Average: {:.2} MB", avg_memory);
        println!("  Maximum: {:.2} MB", max_memory);
        println!("  Minimum: {:.2} MB", min_memory);
        println!("  Range: {:.2} MB", max_memory - min_memory);
        
        // Memory usage should be relatively stable during streaming
        assert!(
            max_memory - min_memory < 50.0,
            "Memory usage too volatile during streaming"
        );
    }
}

/// Helper function to generate test dataset
fn generate_test_dataset(n_series: usize, n_points: usize) -> DataFrame {
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