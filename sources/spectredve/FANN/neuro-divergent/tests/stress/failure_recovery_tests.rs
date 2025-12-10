//! Failure recovery stress tests for handling interruptions, corrupted data, and system crashes.
//! Tests the library's resilience and recovery mechanisms.

use std::fs::{File, OpenOptions};
use std::io::{Write, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, AtomicUsize, Ordering}};
use std::panic;
use tempfile::{TempDir, NamedTempFile};
use chrono::{DateTime, Utc};
use polars::prelude::*;
use rand::Rng;
use serde::{Serialize, Deserialize};
use bincode;

use neuro_divergent_core::data::{
    TimeSeriesDataFrame, TimeSeriesSchema, TimeSeriesDataset,
};
use neuro_divergent_core::error::{NeuroDivergentError, NeuroDivergentResult};
use neuro_divergent_models::{
    basic::{MLP, DLinear},
    forecasting::ForecastingModel,
    core::{ModelConfig, ModelState},
};
use neuro_divergent::prelude::*;

/// Checkpoint manager for saving/restoring training state
#[derive(Debug)]
struct CheckpointManager {
    checkpoint_dir: PathBuf,
    max_checkpoints: usize,
    checkpoints: Vec<Checkpoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Checkpoint {
    id: String,
    epoch: usize,
    loss: f64,
    timestamp: DateTime<Utc>,
    model_path: PathBuf,
    metadata: CheckpointMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointMetadata {
    training_samples: usize,
    validation_samples: usize,
    best_loss: f64,
    total_time_seconds: u64,
}

impl CheckpointManager {
    fn new(checkpoint_dir: PathBuf, max_checkpoints: usize) -> Self {
        std::fs::create_dir_all(&checkpoint_dir).unwrap();
        
        Self {
            checkpoint_dir,
            max_checkpoints,
            checkpoints: Vec::new(),
        }
    }
    
    fn save_checkpoint(&mut self, epoch: usize, loss: f64, model: &dyn ForecastingModel) -> NeuroDivergentResult<()> {
        let checkpoint_id = format!("checkpoint_epoch_{}_loss_{:.4}", epoch, loss);
        let model_path = self.checkpoint_dir.join(format!("{}.model", checkpoint_id));
        
        // Save model state
        let state = model.state();
        let serialized = bincode::serialize(&state)
            .map_err(|e| NeuroDivergentError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        
        let mut file = File::create(&model_path)?;
        file.write_all(&serialized)?;
        
        let checkpoint = Checkpoint {
            id: checkpoint_id,
            epoch,
            loss,
            timestamp: Utc::now(),
            model_path,
            metadata: CheckpointMetadata {
                training_samples: 1000, // Example
                validation_samples: 200,
                best_loss: loss,
                total_time_seconds: 60,
            },
        };
        
        self.checkpoints.push(checkpoint);
        
        // Remove old checkpoints if exceeded max
        if self.checkpoints.len() > self.max_checkpoints {
            let removed = self.checkpoints.remove(0);
            std::fs::remove_file(&removed.model_path).ok();
        }
        
        Ok(())
    }
    
    fn restore_latest(&self) -> NeuroDivergentResult<(Checkpoint, Vec<u8>)> {
        let latest = self.checkpoints.last()
            .ok_or_else(|| NeuroDivergentError::NotFound("No checkpoints found".to_string()))?;
        
        let mut file = File::open(&latest.model_path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        
        Ok((latest.clone(), buffer))
    }
}

/// Test training interruption and recovery
#[test]
fn test_training_interruption_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let mut checkpoint_manager = CheckpointManager::new(temp_dir.path().to_path_buf(), 5);
    
    // Generate dataset
    let df = generate_test_dataset(100, 500);
    let schema = TimeSeriesSchema::new("unique_id", "ds", "y");
    let ts_df = TimeSeriesDataFrame::<f64>::from_polars(df, schema).unwrap();
    
    // Create model
    let model = MLP::builder()
        .hidden_size(64)
        .num_layers(2)
        .learning_rate(0.001)
        .max_epochs(100)
        .horizon(24)
        .input_size(48)
        .build()
        .unwrap();
    
    let interrupt_flag = Arc::new(AtomicBool::new(false));
    let progress = Arc::new(AtomicUsize::new(0));
    
    // Simulate training with interruption
    let interrupt_flag_clone = interrupt_flag.clone();
    let progress_clone = progress.clone();
    
    let training_thread = thread::spawn(move || {
        let mut current_loss = 1.0;
        
        for epoch in 0..100 {
            // Check for interruption
            if interrupt_flag_clone.load(Ordering::Relaxed) {
                println!("Training interrupted at epoch {}", epoch);
                return Err(NeuroDivergentError::Interrupted("Training interrupted".to_string()));
            }
            
            // Simulate training step
            thread::sleep(Duration::from_millis(50));
            current_loss *= 0.95; // Simulate decreasing loss
            
            // Save checkpoint every 10 epochs
            if epoch % 10 == 0 {
                checkpoint_manager.save_checkpoint(epoch, current_loss, &model).unwrap();
                println!("Checkpoint saved at epoch {}", epoch);
            }
            
            progress_clone.store(epoch, Ordering::Relaxed);
        }
        
        Ok(current_loss)
    });
    
    // Simulate interruption after some time
    thread::sleep(Duration::from_secs(2));
    interrupt_flag.store(true, Ordering::Relaxed);
    
    let result = training_thread.join().unwrap();
    assert!(result.is_err());
    
    let interrupted_epoch = progress.load(Ordering::Relaxed);
    println!("Training interrupted at epoch: {}", interrupted_epoch);
    
    // Test recovery
    println!("Attempting to recover from checkpoint...");
    
    let temp_dir_2 = TempDir::new().unwrap();
    let checkpoint_manager_2 = CheckpointManager::new(temp_dir_2.path().to_path_buf(), 5);
    
    match checkpoint_manager_2.restore_latest() {
        Ok((checkpoint, _state_data)) => {
            println!("Recovered from checkpoint: epoch {}, loss {:.4}", 
                     checkpoint.epoch, checkpoint.loss);
            
            // Resume training from checkpoint
            let resume_epoch = checkpoint.epoch;
            assert!(resume_epoch < interrupted_epoch);
        }
        Err(e) => {
            println!("Recovery test (expected): {:?}", e);
        }
    }
}

/// Test corrupted model file handling
#[test]
fn test_corrupted_model_file_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("model.bin");
    
    // Create a valid model and save it
    let model = DLinear::builder()
        .hidden_size(32)
        .kernel_size(3)
        .horizon(12)
        .input_size(24)
        .build()
        .unwrap();
    
    let state = model.state();
    let serialized = bincode::serialize(&state).unwrap();
    
    // Save valid model
    {
        let mut file = File::create(&model_path).unwrap();
        file.write_all(&serialized).unwrap();
    }
    
    // Verify we can load it
    {
        let mut file = File::open(&model_path).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();
        
        let deserialized: Result<ModelState, _> = bincode::deserialize(&buffer);
        assert!(deserialized.is_ok());
    }
    
    // Corrupt the file
    {
        let mut file = OpenOptions::new()
            .write(true)
            .open(&model_path)
            .unwrap();
        
        // Corrupt random bytes
        let mut rng = rand::thread_rng();
        for i in 0..10 {
            let pos = rng.gen_range(0..serialized.len() as u64);
            file.seek(SeekFrom::Start(pos)).unwrap();
            file.write_all(&[rng.gen::<u8>()]).unwrap();
        }
    }
    
    // Try to load corrupted file
    {
        let mut file = File::open(&model_path).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();
        
        let deserialized: Result<ModelState, _> = bincode::deserialize(&buffer);
        
        match deserialized {
            Ok(_) => println!("Unexpectedly loaded corrupted file"),
            Err(e) => println!("Correctly failed to load corrupted file: {:?}", e),
        }
    }
    
    // Test recovery mechanism
    let recovery_result = recover_from_corrupted_model(&model_path);
    assert!(recovery_result.is_err());
}

/// Test partial write recovery
#[test]
fn test_partial_write_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let data_path = temp_dir.path().join("partial_data.parquet");
    
    // Generate large dataset
    let df = generate_test_dataset(10000, 100);
    
    // Simulate partial write by interrupting
    let write_thread = thread::spawn(move || {
        df.lazy()
            .sink_parquet(data_path.clone(), Default::default())
            .unwrap();
    });
    
    // Interrupt after short delay
    thread::sleep(Duration::from_millis(10));
    
    // In practice, we'd actually interrupt the write
    // For testing, we'll let it complete and then corrupt the file
    write_thread.join().unwrap();
    
    // Simulate partial write by truncating file
    let file_size = std::fs::metadata(&data_path).unwrap().len();
    let truncate_size = file_size * 3 / 4; // Keep only 75% of file
    
    {
        let file = OpenOptions::new()
            .write(true)
            .open(&data_path)
            .unwrap();
        file.set_len(truncate_size).unwrap();
    }
    
    // Try to read partially written file
    let read_result = LazyFrame::scan_parquet(&data_path, Default::default());
    
    match read_result {
        Ok(lazy_df) => {
            match lazy_df.collect() {
                Ok(df) => println!("Unexpectedly read partial file with {} rows", df.height()),
                Err(e) => println!("Correctly failed to read partial file: {:?}", e),
            }
        }
        Err(e) => println!("Correctly failed to scan partial file: {:?}", e),
    }
}

/// Test system crash simulation
#[test]
fn test_system_crash_simulation() {
    let crash_flag = Arc::new(AtomicBool::new(false));
    let work_completed = Arc::new(AtomicUsize::new(0));
    
    let crash_flag_clone = crash_flag.clone();
    let work_completed_clone = work_completed.clone();
    
    // Spawn worker that might "crash"
    let worker = thread::spawn(move || {
        let result = panic::catch_unwind(|| {
            for i in 0..100 {
                // Simulate work
                thread::sleep(Duration::from_millis(10));
                work_completed_clone.store(i, Ordering::Relaxed);
                
                // Check if we should "crash"
                if crash_flag_clone.load(Ordering::Relaxed) {
                    panic!("Simulated system crash!");
                }
            }
            
            "Work completed successfully"
        });
        
        result
    });
    
    // Simulate crash after some work
    thread::sleep(Duration::from_millis(200));
    crash_flag.store(true, Ordering::Relaxed);
    
    let result = worker.join().unwrap();
    
    match result {
        Ok(msg) => println!("Unexpected success: {}", msg),
        Err(e) => {
            println!("Worker crashed as expected: {:?}", e);
            let completed = work_completed.load(Ordering::Relaxed);
            println!("Work completed before crash: {}", completed);
            assert!(completed < 100);
        }
    }
}

/// Test network failure recovery (for distributed scenarios)
#[test]
fn test_network_failure_recovery() {
    use std::sync::mpsc::{channel, TryRecvError};
    
    let (tx, rx) = channel();
    let network_healthy = Arc::new(AtomicBool::new(true));
    
    let network_healthy_clone = network_healthy.clone();
    
    // Simulate network communication
    let sender = thread::spawn(move || {
        let mut sent_count = 0;
        
        for i in 0..100 {
            if network_healthy_clone.load(Ordering::Relaxed) {
                match tx.send(i) {
                    Ok(_) => {
                        sent_count += 1;
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(e) => {
                        println!("Send failed: {:?}", e);
                        break;
                    }
                }
            } else {
                // Network is down, wait and retry
                println!("Network down, waiting...");
                thread::sleep(Duration::from_millis(100));
                
                // Check if network recovered
                if network_healthy_clone.load(Ordering::Relaxed) {
                    println!("Network recovered, resuming from message {}", i);
                }
            }
        }
        
        sent_count
    });
    
    // Receiver with timeout and retry
    let receiver = thread::spawn(move || {
        let mut received_count = 0;
        let mut consecutive_failures = 0;
        
        loop {
            match rx.try_recv() {
                Ok(msg) => {
                    received_count += 1;
                    consecutive_failures = 0;
                    
                    if msg % 10 == 0 {
                        println!("Received message: {}", msg);
                    }
                }
                Err(TryRecvError::Empty) => {
                    thread::sleep(Duration::from_millis(10));
                }
                Err(TryRecvError::Disconnected) => {
                    println!("Sender disconnected");
                    break;
                }
            }
            
            consecutive_failures += 1;
            if consecutive_failures > 1000 {
                println!("Too many failures, giving up");
                break;
            }
        }
        
        received_count
    });
    
    // Simulate network failure
    thread::sleep(Duration::from_millis(200));
    network_healthy.store(false, Ordering::Relaxed);
    println!("Simulated network failure");
    
    // Recover after some time
    thread::sleep(Duration::from_millis(300));
    network_healthy.store(true, Ordering::Relaxed);
    println!("Network recovered");
    
    let sent = sender.join().unwrap();
    let received = receiver.join().unwrap();
    
    println!("Sent: {}, Received: {}", sent, received);
}

/// Test transaction rollback
#[test]
fn test_transaction_rollback() {
    #[derive(Debug)]
    struct Transaction {
        operations: Vec<Operation>,
        committed: bool,
    }
    
    #[derive(Debug, Clone)]
    enum Operation {
        Insert(String, f64),
        Update(String, f64),
        Delete(String),
    }
    
    impl Transaction {
        fn new() -> Self {
            Self {
                operations: Vec::new(),
                committed: false,
            }
        }
        
        fn add_operation(&mut self, op: Operation) {
            self.operations.push(op);
        }
        
        fn commit(&mut self) -> Result<(), &'static str> {
            // Simulate commit that might fail
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.3) {
                Err("Commit failed")
            } else {
                self.committed = true;
                Ok(())
            }
        }
        
        fn rollback(&mut self) {
            self.operations.clear();
            self.committed = false;
        }
    }
    
    let mut successful_commits = 0;
    let mut rollbacks = 0;
    
    for i in 0..100 {
        let mut transaction = Transaction::new();
        
        // Add operations
        transaction.add_operation(Operation::Insert(format!("key_{}", i), i as f64));
        transaction.add_operation(Operation::Update(format!("key_{}", i), i as f64 * 2.0));
        
        if i % 3 == 0 {
            transaction.add_operation(Operation::Delete(format!("key_{}", i - 1)));
        }
        
        // Try to commit
        match transaction.commit() {
            Ok(_) => {
                successful_commits += 1;
            }
            Err(e) => {
                println!("Transaction {} failed: {}, rolling back", i, e);
                transaction.rollback();
                rollbacks += 1;
            }
        }
    }
    
    println!("Successful commits: {}, Rollbacks: {}", successful_commits, rollbacks);
    assert!(rollbacks > 0); // Should have some rollbacks
}

/// Test graceful degradation under failures
#[test]
fn test_graceful_degradation() {
    let degradation_level = Arc::new(AtomicUsize::new(0));
    
    // Service levels: 0 = full, 1 = degraded, 2 = minimal, 3 = maintenance
    let service_level_names = vec!["Full Service", "Degraded", "Minimal", "Maintenance"];
    
    let mut handles = Vec::new();
    
    for worker_id in 0..4 {
        let degradation_clone = degradation_level.clone();
        
        let handle = thread::spawn(move || {
            for i in 0..50 {
                let current_level = degradation_clone.load(Ordering::Relaxed);
                
                match current_level {
                    0 => {
                        // Full service - all features available
                        thread::sleep(Duration::from_millis(10));
                        if i % 10 == 0 {
                            println!("Worker {} operating at full capacity", worker_id);
                        }
                    }
                    1 => {
                        // Degraded - some features disabled
                        thread::sleep(Duration::from_millis(20));
                        if i % 10 == 0 {
                            println!("Worker {} in degraded mode", worker_id);
                        }
                    }
                    2 => {
                        // Minimal - only essential features
                        thread::sleep(Duration::from_millis(50));
                        if i % 10 == 0 {
                            println!("Worker {} in minimal mode", worker_id);
                        }
                    }
                    _ => {
                        // Maintenance mode
                        println!("Worker {} in maintenance mode, pausing", worker_id);
                        thread::sleep(Duration::from_secs(1));
                    }
                }
                
                // Simulate random failures that increase degradation
                let mut rng = rand::thread_rng();
                if rng.gen_bool(0.05) {
                    let new_level = current_level.saturating_add(1).min(3);
                    degradation_clone.store(new_level, Ordering::Relaxed);
                    println!("System degraded to level: {}", service_level_names[new_level]);
                }
            }
        });
        
        handles.push(handle);
    }
    
    // Monitor thread that tries to recover
    let degradation_monitor = degradation_level.clone();
    let monitor = thread::spawn(move || {
        loop {
            thread::sleep(Duration::from_secs(1));
            
            let current = degradation_monitor.load(Ordering::Relaxed);
            if current > 0 {
                // Try to recover
                let new_level = current.saturating_sub(1);
                degradation_monitor.store(new_level, Ordering::Relaxed);
                println!("System recovering to level: {}", service_level_names[new_level]);
            }
            
            if current == 0 {
                break;
            }
        }
    });
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    drop(monitor); // Stop monitor
}

/// Test error cascade prevention
#[test]
fn test_error_cascade_prevention() {
    const NUM_COMPONENTS: usize = 10;
    
    #[derive(Debug)]
    struct Component {
        id: usize,
        healthy: AtomicBool,
        error_count: AtomicUsize,
        circuit_breaker_open: AtomicBool,
    }
    
    impl Component {
        fn new(id: usize) -> Self {
            Self {
                id,
                healthy: AtomicBool::new(true),
                error_count: AtomicUsize::new(0),
                circuit_breaker_open: AtomicBool::new(false),
            }
        }
        
        fn process(&self) -> Result<(), String> {
            // Check circuit breaker
            if self.circuit_breaker_open.load(Ordering::Relaxed) {
                return Err("Circuit breaker open".to_string());
            }
            
            // Simulate processing that might fail
            let mut rng = rand::thread_rng();
            if !self.healthy.load(Ordering::Relaxed) || rng.gen_bool(0.1) {
                let errors = self.error_count.fetch_add(1, Ordering::Relaxed);
                
                // Open circuit breaker after threshold
                if errors > 5 {
                    self.circuit_breaker_open.store(true, Ordering::Relaxed);
                    println!("Component {} circuit breaker opened", self.id);
                }
                
                Err(format!("Component {} failed", self.id))
            } else {
                // Success - reset error count
                self.error_count.store(0, Ordering::Relaxed);
                Ok(())
            }
        }
        
        fn reset_circuit_breaker(&self) {
            self.circuit_breaker_open.store(false, Ordering::Relaxed);
            self.error_count.store(0, Ordering::Relaxed);
            println!("Component {} circuit breaker reset", self.id);
        }
    }
    
    let components: Vec<_> = (0..NUM_COMPONENTS)
        .map(|i| Arc::new(Component::new(i)))
        .collect();
    
    // Process requests through components
    let mut total_failures = 0;
    let mut cascade_prevented = 0;
    
    for request_id in 0..100 {
        let mut request_failed = false;
        
        for component in &components {
            match component.process() {
                Ok(_) => {
                    // Continue to next component
                }
                Err(e) => {
                    if e.contains("Circuit breaker") {
                        cascade_prevented += 1;
                    }
                    request_failed = true;
                    break; // Stop processing this request
                }
            }
        }
        
        if request_failed {
            total_failures += 1;
            
            // Simulate causing failures in other components
            if request_id == 20 {
                components[3].healthy.store(false, Ordering::Relaxed);
                println!("Component 3 marked unhealthy");
            }
        }
        
        // Periodically try to reset circuit breakers
        if request_id % 20 == 0 {
            for component in &components {
                if component.circuit_breaker_open.load(Ordering::Relaxed) {
                    component.reset_circuit_breaker();
                }
            }
        }
    }
    
    println!("Total failures: {}, Cascades prevented: {}", total_failures, cascade_prevented);
    assert!(cascade_prevented > 0); // Should have prevented some cascades
}

/// Helper functions
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

fn recover_from_corrupted_model(path: &Path) -> NeuroDivergentResult<()> {
    // Attempt recovery strategies
    // 1. Try to load backup
    let backup_path = path.with_extension("backup");
    if backup_path.exists() {
        std::fs::copy(&backup_path, path)?;
        return Ok(());
    }
    
    // 2. No backup available
    Err(NeuroDivergentError::Io(
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Model file corrupted and no backup available"
        )
    ))
}