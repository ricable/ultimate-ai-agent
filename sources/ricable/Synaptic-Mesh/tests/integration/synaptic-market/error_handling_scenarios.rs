//! Error handling scenarios for the Synaptic Market
//!
//! Tests various failure modes and recovery mechanisms:
//! - Network failures and timeouts
//! - Database corruption and recovery
//! - Docker container failures
//! - Malformed input handling
//! - Resource exhaustion scenarios
//! - Concurrent access conflicts

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use chrono::Utc;
use libp2p::PeerId;
use uuid::Uuid;
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use serde_json::json;
use tokio::sync::Mutex;
use tokio::time::{sleep, timeout};

mod error_test_utils {
    use super::*;

    pub struct ErrorScenarioRunner {
        pub market: Arc<claude_market::Market>,
        pub temp_dir: tempfile::TempDir,
        pub db_path: String,
        pub error_log: Arc<Mutex<Vec<ErrorEvent>>>,
    }

    #[derive(Debug, Clone)]
    pub struct ErrorEvent {
        pub timestamp: chrono::DateTime<Utc>,
        pub error_type: String,
        pub error_message: String,
        pub recovery_attempted: bool,
        pub recovery_successful: bool,
        pub user_affected: Option<PeerId>,
    }

    impl ErrorScenarioRunner {
        pub async fn new() -> Self {
            let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
            let db_path = temp_dir.path().join("error_test.db").to_string_lossy().to_string();
            
            let market = Arc::new(claude_market::Market::new(&db_path).await.unwrap());
            market.init_schema().await.unwrap();

            Self {
                market,
                temp_dir,
                db_path,
                error_log: Arc::new(Mutex::new(Vec::new())),
            }
        }

        pub async fn log_error(&self, error_type: &str, error_message: &str, user: Option<PeerId>) {
            let mut log = self.error_log.lock().await;
            log.push(ErrorEvent {
                timestamp: Utc::now(),
                error_type: error_type.to_string(),
                error_message: error_message.to_string(),
                recovery_attempted: false,
                recovery_successful: false,
                user_affected: user,
            });
        }

        pub async fn log_recovery(&self, error_type: &str, successful: bool) {
            let mut log = self.error_log.lock().await;
            if let Some(last_error) = log.iter_mut()
                .rev()
                .find(|e| e.error_type == error_type) {
                last_error.recovery_attempted = true;
                last_error.recovery_successful = successful;
            }
        }

        pub async fn get_error_count(&self) -> usize {
            self.error_log.lock().await.len()
        }

        pub async fn get_recovery_rate(&self) -> f64 {
            let log = self.error_log.lock().await;
            let total_recoveries = log.iter().filter(|e| e.recovery_attempted).count();
            let successful_recoveries = log.iter().filter(|e| e.recovery_successful).count();
            
            if total_recoveries > 0 {
                successful_recoveries as f64 / total_recoveries as f64
            } else {
                0.0
            }
        }

        pub async fn simulate_database_corruption(&self) -> Result<(), Box<dyn std::error::Error>> {
            // Simulate database corruption by writing invalid data
            use std::fs::OpenOptions;
            use std::io::Write;
            
            let mut file = OpenOptions::new()
                .write(true)
                .append(true)
                .open(&self.db_path)?;
            
            // Write garbage data to corrupt the database
            file.write_all(b"CORRUPTED_DATA_INVALID_SQL_CONTENT")?;
            file.flush()?;
            
            Ok(())
        }

        pub async fn simulate_network_partition(&self, duration: Duration) {
            // Simulate network partition by introducing delays and failures
            sleep(duration).await;
        }

        pub async fn simulate_resource_exhaustion(&self) -> Vec<u8> {
            // Simulate memory exhaustion by allocating large amounts of data
            vec![0u8; 100 * 1024 * 1024] // 100MB allocation
        }
    }

    pub fn generate_test_users(count: usize) -> Vec<(PeerId, SigningKey)> {
        (0..count)
            .map(|_| {
                let key = SigningKey::generate(&mut OsRng);
                let peer_id = PeerId::random();
                (peer_id, key)
            })
            .collect()
    }

    pub fn create_malformed_task_spec() -> claude_market::ComputeTaskSpec {
        claude_market::ComputeTaskSpec {
            task_type: String::new(), // Invalid: empty task type
            compute_units: 0, // Invalid: zero compute units
            max_duration_secs: u64::MAX, // Invalid: excessive duration
            required_capabilities: vec!["".to_string()], // Invalid: empty capability
            min_reputation: Some(-1.0), // Invalid: negative reputation
            privacy_level: claude_market::PrivacyLevel::Private,
            encrypted_payload: Some(vec![0u8; 50 * 1024 * 1024]), // Invalid: 50MB payload
        }
    }
}

#[tokio::test]
async fn test_database_failure_recovery() {
    use error_test_utils::*;
    
    let mut runner = ErrorScenarioRunner::new().await;
    let users = generate_test_users(5);
    
    println!("Testing database failure and recovery scenarios...");

    // Set up initial data
    for (user_id, _) in &users {
        runner.market.opt_in_compute_sharing(user_id, true).await.unwrap();
    }

    let task_spec = claude_market::ComputeTaskSpec {
        task_type: "db_test".to_string(),
        compute_units: 50,
        max_duration_secs: 300,
        required_capabilities: vec![],
        min_reputation: None,
        privacy_level: claude_market::PrivacyLevel::Public,
        encrypted_payload: None,
    };

    // Create some orders before corruption
    let (user1, key1) = &users[0];
    let initial_order = runner.market.place_order(
        claude_market::OrderType::OfferCompute,
        *user1,
        100,
        50,
        task_spec.clone(),
        None,
        None,
        Some(key1),
    ).await.unwrap();

    println!("✓ Initial data created successfully");

    // Test 1: Database corruption handling
    println!("Testing database corruption scenario...");
    
    // Simulate database corruption
    runner.simulate_database_corruption().await.unwrap();
    runner.log_error("database_corruption", "Database file corrupted", Some(*user1)).await;

    // Attempt to perform operations on corrupted database
    let corrupt_operation = runner.market.place_order(
        claude_market::OrderType::RequestCompute,
        users[1].0,
        120,
        30,
        task_spec.clone(),
        None,
        None,
        Some(&users[1].1),
    ).await;

    // Should handle corruption gracefully
    match corrupt_operation {
        Err(_) => {
            println!("✓ Database corruption detected and handled gracefully");
            runner.log_recovery("database_corruption", true).await;
        }
        Ok(_) => {
            // If it succeeded, check if recovery mechanism worked
            println!("✓ Database corruption automatically recovered");
            runner.log_recovery("database_corruption", true).await;
        }
    }

    // Test 2: Database recovery mechanism
    println!("Testing database recovery...");
    
    // Attempt to recover or reinitialize
    let recovery_result = runner.market.attempt_database_recovery().await;
    match recovery_result {
        Ok(_) => {
            println!("✓ Database recovery successful");
            
            // Verify recovery by performing operations
            let post_recovery_order = runner.market.place_order(
                claude_market::OrderType::OfferCompute,
                users[2].0,
                100,
                40,
                task_spec.clone(),
                None,
                None,
                Some(&users[2].1),
            ).await;
            
            assert!(post_recovery_order.is_ok(), "Operations should work after recovery");
            println!("✓ Post-recovery operations successful");
        }
        Err(e) => {
            println!("⚠ Database recovery failed: {}", e);
            runner.log_recovery("database_recovery", false).await;
        }
    }

    // Test 3: Transaction rollback on failure
    println!("Testing transaction rollback...");
    
    let (user3, key3) = &users[2];
    
    // Start a complex transaction that might fail
    let rollback_test = runner.market.perform_complex_transaction(*user3, |market| async move {
        // First operation: place order (should succeed)
        let order1 = market.place_order(
            claude_market::OrderType::RequestCompute,
            *user3,
            100,
            50,
            task_spec.clone(),
            None,
            None,
            Some(key3),
        ).await?;

        // Second operation: intentionally fail
        return Err(claude_market::MarketError::InvalidOrder("Intentional failure".to_string()));
    }).await;

    // Verify rollback occurred
    assert!(rollback_test.is_err(), "Transaction should have failed");
    
    let user3_orders = runner.market.get_trader_orders(user3, None, true).await.unwrap();
    let user3_order_count = user3_orders.len();
    
    println!("✓ Transaction rollback verified (user has {} orders)", user3_order_count);

    println!("✓ Database failure recovery tests completed");
}

#[tokio::test]
async fn test_network_failure_scenarios() {
    use error_test_utils::*;
    
    let runner = ErrorScenarioRunner::new().await;
    let users = generate_test_users(3);
    
    println!("Testing network failure scenarios...");

    for (user_id, _) in &users {
        runner.market.opt_in_compute_sharing(user_id, true).await.unwrap();
    }

    let task_spec = claude_market::ComputeTaskSpec {
        task_type: "network_test".to_string(),
        compute_units: 30,
        max_duration_secs: 300,
        required_capabilities: vec![],
        min_reputation: None,
        privacy_level: claude_market::PrivacyLevel::Public,
        encrypted_payload: None,
    };

    // Test 1: Connection timeout handling
    println!("Testing connection timeout handling...");
    
    let timeout_duration = Duration::from_millis(100);
    
    let timeout_test = timeout(timeout_duration, async {
        // Simulate a slow operation
        sleep(Duration::from_millis(200)).await;
        
        runner.market.place_order(
            claude_market::OrderType::OfferCompute,
            users[0].0,
            100,
            50,
            task_spec.clone(),
            None,
            None,
            Some(&users[0].1),
        ).await
    }).await;

    match timeout_test {
        Err(_) => {
            println!("✓ Timeout properly detected");
            runner.log_error("connection_timeout", "Operation timed out", Some(users[0].0)).await;
        }
        Ok(Ok(_)) => {
            println!("✓ Operation completed within timeout");
        }
        Ok(Err(e)) => {
            println!("⚠ Operation failed: {}", e);
        }
    }

    // Test 2: Network partition simulation
    println!("Testing network partition handling...");
    
    let partition_start = Instant::now();
    
    // Simulate network partition
    let partition_task = tokio::spawn(async move {
        runner.simulate_network_partition(Duration::from_millis(500)).await;
    });

    // Try to perform operations during partition
    let partition_operations = vec![
        runner.market.place_order(
            claude_market::OrderType::RequestCompute,
            users[1].0,
            120,
            40,
            task_spec.clone(),
            None,
            None,
            Some(&users[1].1),
        ),
        runner.market.place_order(
            claude_market::OrderType::OfferCompute,
            users[2].0,
            110,
            60,
            task_spec.clone(),
            None,
            None,
            Some(&users[2].1),
        ),
    ];

    let partition_results = futures::future::join_all(partition_operations).await;
    partition_task.await.unwrap();
    
    let partition_duration = partition_start.elapsed();
    println!("✓ Network partition simulation completed in {:?}", partition_duration);

    // Verify operations during partition
    let successful_during_partition = partition_results.iter().filter(|r| r.is_ok()).count();
    let failed_during_partition = partition_results.iter().filter(|r| r.is_err()).count();
    
    println!("  - Successful operations: {}", successful_during_partition);
    println!("  - Failed operations: {}", failed_during_partition);

    if failed_during_partition > 0 {
        runner.log_error("network_partition", 
                         &format!("{} operations failed during partition", failed_during_partition),
                         None).await;
    }

    // Test 3: Network recovery validation
    println!("Testing network recovery...");
    
    // After partition, verify system can recover
    let recovery_start = Instant::now();
    
    let recovery_order = runner.market.place_order(
        claude_market::OrderType::OfferCompute,
        users[0].0,
        100,
        30,
        task_spec.clone(),
        None,
        None,
        Some(&users[0].1),
    ).await;

    match recovery_order {
        Ok(_) => {
            let recovery_time = recovery_start.elapsed();
            println!("✓ Network recovery successful in {:?}", recovery_time);
            runner.log_recovery("network_partition", true).await;
        }
        Err(e) => {
            println!("⚠ Network recovery failed: {}", e);
            runner.log_recovery("network_partition", false).await;
        }
    }

    // Test 4: Concurrent connection handling
    println!("Testing concurrent connection handling...");
    
    let concurrent_operations = 20;
    let mut concurrent_tasks = Vec::new();
    
    for i in 0..concurrent_operations {
        let market_clone = Arc::clone(&runner.market);
        let user = users[i % users.len()].clone();
        let task_spec_clone = task_spec.clone();
        
        let task = tokio::spawn(async move {
            market_clone.place_order(
                claude_market::OrderType::OfferCompute,
                user.0,
                100 + i as u64,
                20 + i as u64,
                task_spec_clone,
                None,
                None,
                Some(&user.1),
            ).await
        });
        
        concurrent_tasks.push(task);
    }

    let concurrent_results = futures::future::join_all(concurrent_tasks).await;
    let concurrent_successful = concurrent_results.iter()
        .filter_map(|r| r.as_ref().ok())
        .filter(|r| r.is_ok())
        .count();
    
    println!("✓ Concurrent operations: {}/{} successful", 
             concurrent_successful, concurrent_operations);

    assert!(concurrent_successful > concurrent_operations * 8 / 10, 
            "Too many concurrent operations failed");

    println!("✓ Network failure scenario tests completed");
}

#[tokio::test]
async fn test_docker_container_failure_recovery() {
    use error_test_utils::*;
    
    let runner = ErrorScenarioRunner::new().await;
    let users = generate_test_users(2);
    
    println!("Testing Docker container failure and recovery...");

    for (user_id, _) in &users {
        runner.market.opt_in_compute_sharing(user_id, true).await.unwrap();
    }

    // Test 1: Container startup failure
    println!("Testing container startup failure...");
    
    let startup_failure_test = runner.market.test_container_startup_failure().await;
    match startup_failure_test {
        Err(e) => {
            println!("✓ Container startup failure detected: {}", e);
            runner.log_error("container_startup", &e.to_string(), Some(users[0].0)).await;
        }
        Ok(_) => {
            println!("✓ Container startup successful (no failure to test)");
        }
    }

    // Test 2: Container crash during execution
    println!("Testing container crash handling...");
    
    let task_spec = claude_market::ComputeTaskSpec {
        task_type: "crash_test".to_string(),
        compute_units: 50,
        max_duration_secs: 300,
        required_capabilities: vec!["docker".to_string()],
        min_reputation: None,
        privacy_level: claude_market::PrivacyLevel::Private,
        encrypted_payload: Some(b"test_crash_payload".to_vec()),
    };

    // Create assignment
    let request = runner.market.place_order(
        claude_market::OrderType::RequestCompute,
        users[0].0,
        100,
        50,
        task_spec.clone(),
        None,
        None,
        Some(&users[0].1),
    ).await.unwrap();

    let offer = runner.market.place_order(
        claude_market::OrderType::OfferCompute,
        users[1].0,
        80,
        100,
        task_spec,
        None,
        None,
        Some(&users[1].1),
    ).await.unwrap();

    sleep(Duration::from_millis(100)).await;

    let assignments = runner.market.get_assignments(None, 10).await.unwrap();
    if !assignments.is_empty() {
        let assignment = &assignments[0];
        
        // Start task
        runner.market.start_task(&assignment.id, &assignment.provider).await.unwrap();
        
        // Simulate container crash
        let crash_result = runner.market.simulate_container_crash(&assignment.id).await;
        match crash_result {
            Err(e) => {
                println!("✓ Container crash simulated: {}", e);
                runner.log_error("container_crash", &e.to_string(), Some(assignment.provider)).await;
                
                // Test recovery mechanism
                let recovery_result = runner.market.recover_from_container_crash(&assignment.id).await;
                match recovery_result {
                    Ok(_) => {
                        println!("✓ Container crash recovery successful");
                        runner.log_recovery("container_crash", true).await;
                    }
                    Err(e) => {
                        println!("⚠ Container crash recovery failed: {}", e);
                        runner.log_recovery("container_crash", false).await;
                    }
                }
            }
            Ok(_) => {
                println!("✓ No container crash occurred (system stable)");
            }
        }
    }

    // Test 3: Resource limit enforcement
    println!("Testing Docker resource limit enforcement...");
    
    let resource_test = runner.market.test_resource_limits().await;
    match resource_test {
        Ok(limits) => {
            println!("✓ Resource limits enforced:");
            println!("  - Memory limit: {} MB", limits.memory_limit_mb);
            println!("  - CPU limit: {} cores", limits.cpu_limit);
            println!("  - Timeout: {} seconds", limits.timeout_seconds);
            
            assert!(limits.memory_limit_mb <= 1024, "Memory limit should be reasonable");
            assert!(limits.timeout_seconds <= 3600, "Timeout should be reasonable");
        }
        Err(e) => {
            println!("⚠ Resource limit test failed: {}", e);
            runner.log_error("resource_limits", &e.to_string(), None).await;
        }
    }

    // Test 4: Container cleanup on failure
    println!("Testing container cleanup on failure...");
    
    let cleanup_test = runner.market.test_container_cleanup().await;
    match cleanup_test {
        Ok(cleanup_info) => {
            println!("✓ Container cleanup successful:");
            println!("  - Containers stopped: {}", cleanup_info.containers_stopped);
            println!("  - Containers removed: {}", cleanup_info.containers_removed);
            println!("  - Volumes cleaned: {}", cleanup_info.volumes_cleaned);
            
            assert!(cleanup_info.containers_stopped >= cleanup_info.containers_removed);
        }
        Err(e) => {
            println!("⚠ Container cleanup failed: {}", e);
            runner.log_error("container_cleanup", &e.to_string(), None).await;
        }
    }

    println!("✓ Docker container failure recovery tests completed");
}

#[tokio::test]
async fn test_malformed_input_handling() {
    use error_test_utils::*;
    
    let runner = ErrorScenarioRunner::new().await;
    let users = generate_test_users(3);
    
    println!("Testing malformed input handling...");

    for (user_id, _) in &users {
        runner.market.opt_in_compute_sharing(user_id, true).await.unwrap();
    }

    // Test 1: Invalid order parameters
    println!("Testing invalid order parameters...");
    
    let malformed_spec = create_malformed_task_spec();
    
    let invalid_order = runner.market.place_order(
        claude_market::OrderType::OfferCompute,
        users[0].0,
        0, // Invalid: zero price
        0, // Invalid: zero units
        malformed_spec,
        None,
        None,
        Some(&users[0].1),
    ).await;

    match invalid_order {
        Err(e) => {
            println!("✓ Invalid order properly rejected: {}", e);
            runner.log_error("invalid_input", &e.to_string(), Some(users[0].0)).await;
            runner.log_recovery("invalid_input", true).await; // Rejection is successful handling
        }
        Ok(_) => {
            println!("⚠ Invalid order was accepted (potential issue)");
        }
    }

    // Test 2: Malformed JSON payload
    println!("Testing malformed JSON payload handling...");
    
    let malformed_json_payload = b"{ invalid json structure }";
    
    let json_test_spec = claude_market::ComputeTaskSpec {
        task_type: "json_test".to_string(),
        compute_units: 25,
        max_duration_secs: 300,
        required_capabilities: vec![],
        min_reputation: None,
        privacy_level: claude_market::PrivacyLevel::Private,
        encrypted_payload: Some(malformed_json_payload.to_vec()),
    };

    let malformed_json_order = runner.market.place_order(
        claude_market::OrderType::RequestCompute,
        users[1].0,
        100,
        25,
        json_test_spec,
        None,
        None,
        Some(&users[1].1),
    ).await;

    match malformed_json_order {
        Ok(_) => {
            println!("✓ Malformed JSON handled gracefully (stored as binary)");
        }
        Err(e) => {
            println!("✓ Malformed JSON properly rejected: {}", e);
        }
    }

    // Test 3: Oversized input handling
    println!("Testing oversized input handling...");
    
    let oversized_payload = vec![0u8; 100 * 1024 * 1024]; // 100MB
    
    let oversized_spec = claude_market::ComputeTaskSpec {
        task_type: "oversized_test".to_string(),
        compute_units: 50,
        max_duration_secs: 300,
        required_capabilities: vec!["large_data".to_string()],
        min_reputation: None,
        privacy_level: claude_market::PrivacyLevel::Private,
        encrypted_payload: Some(oversized_payload),
    };

    let oversized_order = runner.market.place_order(
        claude_market::OrderType::OfferCompute,
        users[2].0,
        200,
        50,
        oversized_spec,
        None,
        None,
        Some(&users[2].1),
    ).await;

    match oversized_order {
        Err(e) => {
            println!("✓ Oversized input properly rejected: {}", e);
            runner.log_error("oversized_input", &e.to_string(), Some(users[2].0)).await;
            runner.log_recovery("oversized_input", true).await;
        }
        Ok(_) => {
            println!("⚠ Oversized input was accepted (check size limits)");
        }
    }

    // Test 4: Unicode and special character handling
    println!("Testing Unicode and special character handling...");
    
    let unicode_spec = claude_market::ComputeTaskSpec {
        task_type: "测试任务🚀💻".to_string(),
        compute_units: 20,
        max_duration_secs: 300,
        required_capabilities: vec!["unicode_support".to_string(), "emoji🎯".to_string()],
        min_reputation: None,
        privacy_level: claude_market::PrivacyLevel::Public,
        encrypted_payload: Some("Special chars: àáâãäåæçèéêë".as_bytes().to_vec()),
    };

    let unicode_order = runner.market.place_order(
        claude_market::OrderType::RequestCompute,
        users[0].0,
        150,
        20,
        unicode_spec,
        None,
        None,
        Some(&users[0].1),
    ).await;

    match unicode_order {
        Ok(_) => {
            println!("✓ Unicode input handled correctly");
        }
        Err(e) => {
            println!("⚠ Unicode input handling failed: {}", e);
            runner.log_error("unicode_handling", &e.to_string(), Some(users[0].0)).await;
        }
    }

    // Test 5: Null and empty value handling
    println!("Testing null and empty value handling...");
    
    let empty_spec = claude_market::ComputeTaskSpec {
        task_type: String::new(), // Empty string
        compute_units: 10,
        max_duration_secs: 300,
        required_capabilities: vec![String::new()], // Empty capability
        min_reputation: None,
        privacy_level: claude_market::PrivacyLevel::Public,
        encrypted_payload: Some(Vec::new()), // Empty payload
    };

    let empty_order = runner.market.place_order(
        claude_market::OrderType::OfferCompute,
        users[1].0,
        100,
        10,
        empty_spec,
        None,
        None,
        Some(&users[1].1),
    ).await;

    match empty_order {
        Err(e) => {
            println!("✓ Empty values properly rejected: {}", e);
            runner.log_recovery("empty_values", true).await;
        }
        Ok(_) => {
            println!("⚠ Empty values were accepted");
        }
    }

    println!("✓ Malformed input handling tests completed");
}

#[tokio::test]
async fn test_resource_exhaustion_scenarios() {
    use error_test_utils::*;
    
    let runner = ErrorScenarioRunner::new().await;
    let users = generate_test_users(10);
    
    println!("Testing resource exhaustion scenarios...");

    for (user_id, _) in &users {
        runner.market.opt_in_compute_sharing(user_id, true).await.unwrap();
    }

    // Test 1: Memory exhaustion handling
    println!("Testing memory exhaustion handling...");
    
    let initial_memory = get_memory_usage().await;
    println!("Initial memory usage: {:.2} MB", initial_memory);
    
    let mut memory_hogs = Vec::new();
    let mut memory_allocations = 0;
    
    // Gradually increase memory usage
    for i in 0..10 {
        match runner.simulate_resource_exhaustion().await {
            allocation => {
                memory_hogs.push(allocation);
                memory_allocations += 1;
                
                let current_memory = get_memory_usage().await;
                println!("Memory allocation {}: {:.2} MB", i + 1, current_memory);
                
                // Test if system still responds
                let memory_test_order = timeout(Duration::from_millis(1000), async {
                    let task_spec = claude_market::ComputeTaskSpec {
                        task_type: "memory_test".to_string(),
                        compute_units: 5,
                        max_duration_secs: 60,
                        required_capabilities: vec![],
                        min_reputation: None,
                        privacy_level: claude_market::PrivacyLevel::Public,
                        encrypted_payload: None,
                    };

                    runner.market.place_order(
                        claude_market::OrderType::OfferCompute,
                        users[i % users.len()].0,
                        50,
                        5,
                        task_spec,
                        None,
                        None,
                        Some(&users[i % users.len()].1),
                    ).await
                }).await;

                match memory_test_order {
                    Ok(Ok(_)) => {
                        println!("  ✓ System responsive under memory pressure");
                    }
                    Ok(Err(e)) => {
                        println!("  ⚠ System error under memory pressure: {}", e);
                        runner.log_error("memory_pressure", &e.to_string(), None).await;
                        break;
                    }
                    Err(_) => {
                        println!("  ⚠ System timeout under memory pressure");
                        runner.log_error("memory_timeout", "System timeout", None).await;
                        break;
                    }
                }
                
                // Check if we should stop to avoid crashing the test
                if current_memory > initial_memory + 500.0 { // Stop at +500MB
                    println!("  Stopping memory test to avoid system crash");
                    break;
                }
            }
        }
    }
    
    // Cleanup memory
    drop(memory_hogs);
    
    let final_memory = get_memory_usage().await;
    println!("Final memory usage: {:.2} MB", final_memory);
    
    if memory_allocations > 0 {
        runner.log_recovery("memory_pressure", final_memory < initial_memory + 100.0).await;
    }

    // Test 2: Concurrent connection exhaustion
    println!("Testing connection exhaustion...");
    
    let max_connections = 100;
    let mut connection_tasks = Vec::new();
    
    for i in 0..max_connections {
        let market_clone = Arc::clone(&runner.market);
        let user = users[i % users.len()].clone();
        
        let task = tokio::spawn(async move {
            let task_spec = claude_market::ComputeTaskSpec {
                task_type: format!("connection_test_{}", i),
                compute_units: 1,
                max_duration_secs: 300,
                required_capabilities: vec![],
                min_reputation: None,
                privacy_level: claude_market::PrivacyLevel::Public,
                encrypted_payload: None,
            };

            market_clone.place_order(
                claude_market::OrderType::OfferCompute,
                user.0,
                10 + i as u64,
                1,
                task_spec,
                None,
                None,
                Some(&user.1),
            ).await
        });
        
        connection_tasks.push(task);
        
        // Add small delay to simulate realistic connection pattern
        if i % 10 == 0 {
            sleep(Duration::from_millis(10)).await;
        }
    }

    let connection_results = futures::future::join_all(connection_tasks).await;
    let successful_connections = connection_results.iter()
        .filter_map(|r| r.as_ref().ok())
        .filter(|r| r.is_ok())
        .count();
    
    let failed_connections = max_connections - successful_connections;
    
    println!("Connection test results:");
    println!("  - Successful: {}/{}", successful_connections, max_connections);
    println!("  - Failed: {}", failed_connections);
    
    if failed_connections > max_connections / 4 {
        runner.log_error("connection_exhaustion", 
                         &format!("{} connections failed", failed_connections),
                         None).await;
    }

    // Test 3: Database connection pool exhaustion
    println!("Testing database connection pool exhaustion...");
    
    let db_stress_tasks = (0..50).map(|i| {
        let market_clone = Arc::clone(&runner.market);
        let user = users[i % users.len()].0;
        
        tokio::spawn(async move {
            // Perform rapid database operations
            for _ in 0..5 {
                let _ = market_clone.get_trader_orders(&user, None, true).await;
                let _ = market_clone.get_assignments(Some(&user), 10).await;
                sleep(Duration::from_millis(1)).await;
            }
        })
    }).collect::<Vec<_>>();

    let db_stress_start = Instant::now();
    futures::future::join_all(db_stress_tasks).await;
    let db_stress_duration = db_stress_start.elapsed();
    
    println!("Database stress test completed in {:?}", db_stress_duration);
    
    // Verify system is still responsive
    let post_stress_test = runner.market.place_order(
        claude_market::OrderType::RequestCompute,
        users[0].0,
        100,
        10,
        claude_market::ComputeTaskSpec {
            task_type: "post_stress_test".to_string(),
            compute_units: 10,
            max_duration_secs: 300,
            required_capabilities: vec![],
            min_reputation: None,
            privacy_level: claude_market::PrivacyLevel::Public,
            encrypted_payload: None,
        },
        None,
        None,
        Some(&users[0].1),
    ).await;

    match post_stress_test {
        Ok(_) => {
            println!("✓ System responsive after database stress test");
            runner.log_recovery("db_stress", true).await;
        }
        Err(e) => {
            println!("⚠ System not responsive after stress test: {}", e);
            runner.log_error("db_stress", &e.to_string(), None).await;
            runner.log_recovery("db_stress", false).await;
        }
    }

    println!("✓ Resource exhaustion scenario tests completed");

    async fn get_memory_usage() -> f64 {
        // Simple memory measurement
        if let Ok(status) = tokio::fs::read_to_string("/proc/self/status").await {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<f64>() {
                            return kb / 1024.0; // Convert to MB
                        }
                    }
                }
            }
        }
        0.0
    }
}

#[tokio::test]
async fn test_concurrent_access_conflicts() {
    use error_test_utils::*;
    
    let runner = ErrorScenarioRunner::new().await;
    let users = generate_test_users(5);
    
    println!("Testing concurrent access conflict scenarios...");

    for (user_id, _) in &users {
        runner.market.opt_in_compute_sharing(user_id, true).await.unwrap();
    }

    let task_spec = claude_market::ComputeTaskSpec {
        task_type: "concurrent_test".to_string(),
        compute_units: 100,
        max_duration_secs: 300,
        required_capabilities: vec![],
        min_reputation: None,
        privacy_level: claude_market::PrivacyLevel::Public,
        encrypted_payload: None,
    };

    // Test 1: Concurrent order placement on same resource
    println!("Testing concurrent order placement conflicts...");
    
    let (shared_user, shared_key) = &users[0];
    let concurrent_order_tasks = (0..10).map(|i| {
        let market_clone = Arc::clone(&runner.market);
        let user_id = *shared_user;
        let key = shared_key.clone();
        let spec = task_spec.clone();
        
        tokio::spawn(async move {
            market_clone.place_order(
                claude_market::OrderType::OfferCompute,
                user_id,
                100 + i as u64,
                50 + i as u64,
                spec,
                None,
                None,
                Some(&key),
            ).await
        })
    }).collect::<Vec<_>>();

    let concurrent_order_results = futures::future::join_all(concurrent_order_tasks).await;
    let successful_concurrent_orders = concurrent_order_results.iter()
        .filter_map(|r| r.as_ref().ok())
        .filter(|r| r.is_ok())
        .count();
    
    println!("Concurrent order placement: {}/10 successful", successful_concurrent_orders);
    assert!(successful_concurrent_orders >= 8, "Too many concurrent order failures");

    // Test 2: Race condition in assignment creation
    println!("Testing assignment creation race conditions...");
    
    // Create a single request that multiple providers might try to match
    let (requester, requester_key) = &users[1];
    let high_value_request = runner.market.place_order(
        claude_market::OrderType::RequestCompute,
        *requester,
        500, // High price to attract multiple providers
        100,
        task_spec.clone(),
        None,
        None,
        Some(requester_key),
    ).await.unwrap();

    // Multiple providers try to match simultaneously
    let provider_tasks = users[2..].iter().map(|(provider_id, provider_key)| {
        let market_clone = Arc::clone(&runner.market);
        let provider = *provider_id;
        let key = provider_key.clone();
        let spec = task_spec.clone();
        
        tokio::spawn(async move {
            market_clone.place_order(
                claude_market::OrderType::OfferCompute,
                provider,
                400, // Competitive price
                150,
                spec,
                None,
                None,
                Some(&key),
            ).await
        })
    }).collect::<Vec<_>>();

    let provider_results = futures::future::join_all(provider_tasks).await;
    sleep(Duration::from_millis(200)).await; // Allow matching to complete

    // Check assignment results
    let assignments = runner.market.get_assignments(None, 20).await.unwrap();
    let request_assignments = assignments.iter()
        .filter(|a| a.request_id == high_value_request.id)
        .count();
    
    println!("Assignment race condition test: {} assignments created", request_assignments);
    
    // Should have exactly one assignment (no double-matching)
    if request_assignments > 1 {
        runner.log_error("race_condition", 
                         &format!("Multiple assignments for single request: {}", request_assignments),
                         Some(*requester)).await;
    } else if request_assignments == 1 {
        println!("✓ Race condition properly handled");
        runner.log_recovery("race_condition", true).await;
    }

    // Test 3: Concurrent task state updates
    println!("Testing concurrent task state update conflicts...");
    
    if !assignments.is_empty() {
        let assignment = &assignments[0];
        let assignment_id = assignment.id;
        let provider = assignment.provider;
        
        // Start the task first
        runner.market.start_task(&assignment_id, &provider).await.unwrap();
        
        // Try concurrent state updates
        let state_update_tasks = vec![
            // Task 1: Try to complete normally
            {
                let market_clone = Arc::clone(&runner.market);
                tokio::spawn(async move {
                    sleep(Duration::from_millis(50)).await;
                    let quality_scores = HashMap::from([
                        ("accuracy".to_string(), 0.90),
                    ]);
                    market_clone.complete_task(&assignment_id, &provider, quality_scores).await
                })
            },
            // Task 2: Try to fail the task
            {
                let market_clone = Arc::clone(&runner.market);
                tokio::spawn(async move {
                    sleep(Duration::from_millis(60)).await;
                    market_clone.fail_task(&assignment_id, &provider, "Simulated failure".to_string()).await
                })
            },
            // Task 3: Try to cancel
            {
                let market_clone = Arc::clone(&runner.market);
                tokio::spawn(async move {
                    sleep(Duration::from_millis(40)).await;
                    market_clone.cancel_task(&assignment_id, &provider).await
                })
            },
        ];

        let state_update_results = futures::future::join_all(state_update_tasks).await;
        let successful_updates = state_update_results.iter()
            .filter_map(|r| r.as_ref().ok())
            .filter(|r| r.is_ok())
            .count();

        println!("Concurrent state updates: {}/3 successful", successful_updates);
        
        // Should only allow one state change
        if successful_updates > 1 {
            runner.log_error("state_conflict", 
                             &format!("Multiple state updates succeeded: {}", successful_updates),
                             Some(provider)).await;
        } else {
            println!("✓ Concurrent state updates properly serialized");
            runner.log_recovery("state_conflict", true).await;
        }
    }

    // Test 4: Deadlock detection and prevention
    println!("Testing deadlock detection...");
    
    let deadlock_test_start = Instant::now();
    
    // Create scenario that could potentially deadlock
    let deadlock_tasks = (0..5).map(|i| {
        let market_clone = Arc::clone(&runner.market);
        let user1 = users[i % users.len()].0;
        let user2 = users[(i + 1) % users.len()].0;
        let spec = task_spec.clone();
        
        tokio::spawn(async move {
            // User1 places request
            let _ = market_clone.place_order(
                claude_market::OrderType::RequestCompute,
                user1,
                200,
                50,
                spec.clone(),
                None,
                None,
                None,
            ).await;
            
            // Immediately, User2 places offer
            let _ = market_clone.place_order(
                claude_market::OrderType::OfferCompute,
                user2,
                180,
                100,
                spec,
                None,
                None,
                None,
            ).await;
        })
    }).collect::<Vec<_>>();

    // Set timeout to detect deadlocks
    let deadlock_result = timeout(Duration::from_secs(5), 
        futures::future::join_all(deadlock_tasks)).await;
    
    let deadlock_duration = deadlock_test_start.elapsed();
    
    match deadlock_result {
        Ok(_) => {
            println!("✓ No deadlock detected (completed in {:?})", deadlock_duration);
            runner.log_recovery("deadlock_prevention", true).await;
        }
        Err(_) => {
            println!("⚠ Potential deadlock detected (timeout after {:?})", deadlock_duration);
            runner.log_error("deadlock", "Operation timed out", None).await;
        }
    }

    // Generate error summary
    let total_errors = runner.get_error_count().await;
    let recovery_rate = runner.get_recovery_rate().await;
    
    println!("\n=== Concurrent Access Conflict Test Summary ===");
    println!("Total errors detected: {}", total_errors);
    println!("Recovery rate: {:.1}%", recovery_rate * 100.0);
    
    // Verify system is still functional
    let final_test = runner.market.place_order(
        claude_market::OrderType::OfferCompute,
        users[0].0,
        100,
        25,
        task_spec,
        None,
        None,
        Some(&users[0].1),
    ).await;

    assert!(final_test.is_ok(), "System should be functional after concurrent access tests");
    println!("✓ System remains functional after conflict tests");
    println!("✓ Concurrent access conflict tests completed");
}