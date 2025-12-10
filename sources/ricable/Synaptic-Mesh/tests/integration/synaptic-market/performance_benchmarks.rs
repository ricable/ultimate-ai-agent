//! Performance benchmarks for the Synaptic Market
//!
//! Tests system performance under various load conditions:
//! - High-frequency order placement and matching
//! - Large-scale concurrent operations
//! - Memory usage and optimization
//! - Database performance
//! - Network efficiency simulation

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use chrono::Utc;
use libp2p::PeerId;
use uuid::Uuid;
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use serde_json::json;
use tokio::sync::{Mutex, Semaphore};
use tokio::time::sleep;

mod benchmark_utils {
    use super::*;

    pub struct BenchmarkConfig {
        pub num_users: usize,
        pub num_orders_per_user: usize,
        pub concurrent_limit: usize,
        pub order_types_ratio: f64, // 0.0 = all requests, 1.0 = all offers
        pub task_types: Vec<String>,
    }

    impl Default for BenchmarkConfig {
        fn default() -> Self {
            Self {
                num_users: 100,
                num_orders_per_user: 10,
                concurrent_limit: 50,
                order_types_ratio: 0.5,
                task_types: vec![
                    "code_generation".to_string(),
                    "data_analysis".to_string(),
                    "ml_training".to_string(),
                    "testing".to_string(),
                    "documentation".to_string(),
                ],
            }
        }
    }

    pub struct BenchmarkResult {
        pub total_operations: u64,
        pub successful_operations: u64,
        pub failed_operations: u64,
        pub total_duration: Duration,
        pub operations_per_second: f64,
        pub average_latency: Duration,
        pub p95_latency: Duration,
        pub p99_latency: Duration,
        pub memory_usage_mb: f64,
        pub assignments_created: u64,
    }

    impl BenchmarkResult {
        pub fn print_summary(&self, test_name: &str) {
            println!("\n=== {} Benchmark Results ===", test_name);
            println!("Total operations: {}", self.total_operations);
            println!("Successful: {} ({:.2}%)", 
                     self.successful_operations, 
                     (self.successful_operations as f64 / self.total_operations as f64) * 100.0);
            println!("Failed: {}", self.failed_operations);
            println!("Duration: {:?}", self.total_duration);
            println!("Operations/sec: {:.2}", self.operations_per_second);
            println!("Average latency: {:?}", self.average_latency);
            println!("P95 latency: {:?}", self.p95_latency);
            println!("P99 latency: {:?}", self.p99_latency);
            println!("Memory usage: {:.2} MB", self.memory_usage_mb);
            println!("Assignments created: {}", self.assignments_created);
            println!("================================\n");
        }
    }

    pub fn create_test_environment() -> (tempfile::TempDir, String) {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let db_path = temp_dir.path().join("benchmark.db").to_string_lossy().to_string();
        (temp_dir, db_path)
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

    pub fn create_task_spec(task_type: &str, size_factor: u64) -> claude_market::ComputeTaskSpec {
        claude_market::ComputeTaskSpec {
            task_type: task_type.to_string(),
            compute_units: (10 + size_factor * 10),
            max_duration_secs: 300 + size_factor * 60,
            required_capabilities: vec![
                format!("skill_{}", size_factor % 5),
                task_type.to_string(),
            ],
            min_reputation: if size_factor % 3 == 0 { Some(70.0) } else { None },
            privacy_level: match size_factor % 3 {
                0 => claude_market::PrivacyLevel::Public,
                1 => claude_market::PrivacyLevel::Private,
                _ => claude_market::PrivacyLevel::Confidential,
            },
            encrypted_payload: if size_factor % 4 == 0 {
                Some(vec![0u8; (size_factor * 100) as usize])
            } else {
                None
            },
        }
    }

    pub async fn measure_memory_usage() -> f64 {
        // Simple memory measurement - in production would use more sophisticated tools
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
        0.0 // Default if measurement fails
    }

    pub fn calculate_percentile(mut latencies: Vec<Duration>, percentile: f64) -> Duration {
        if latencies.is_empty() {
            return Duration::from_millis(0);
        }
        
        latencies.sort();
        let index = ((latencies.len() as f64 * percentile / 100.0).ceil() as usize).saturating_sub(1);
        latencies[index]
    }
}

#[tokio::test]
async fn benchmark_order_placement_throughput() {
    use benchmark_utils::*;
    
    let (_temp_dir, db_path) = create_test_environment();
    let market = Arc::new(claude_market::Market::new(&db_path).await.unwrap());
    market.init_schema().await.unwrap();

    let config = BenchmarkConfig {
        num_users: 200,
        num_orders_per_user: 5,
        concurrent_limit: 100,
        ..Default::default()
    };

    println!("Starting order placement throughput benchmark...");
    println!("Users: {}, Orders per user: {}, Concurrent limit: {}", 
             config.num_users, config.num_orders_per_user, config.concurrent_limit);

    // Generate test users
    let users = generate_test_users(config.num_users);
    
    // Opt in all users
    for (user_id, _) in &users {
        market.opt_in_compute_sharing(user_id, true).await.unwrap();
    }

    let semaphore = Arc::new(Semaphore::new(config.concurrent_limit));
    let mut latencies = Vec::new();
    let mut successful_operations = 0u64;
    let mut failed_operations = 0u64;

    let start_time = Instant::now();
    let start_memory = measure_memory_usage().await;

    // Create tasks for all operations
    let mut tasks = Vec::new();
    
    for (user_idx, (user_id, signing_key)) in users.iter().enumerate() {
        for order_idx in 0..config.num_orders_per_user {
            let market_clone = Arc::clone(&market);
            let semaphore_clone = Arc::clone(&semaphore);
            let user_id = *user_id;
            let signing_key = signing_key.clone();
            let task_type = config.task_types[order_idx % config.task_types.len()].clone();
            
            let task = tokio::spawn(async move {
                let _permit = semaphore_clone.acquire().await.unwrap();
                
                let operation_start = Instant::now();
                
                let task_spec = create_task_spec(&task_type, (user_idx + order_idx) as u64);
                
                let order_type = if (user_idx + order_idx) % 2 == 0 {
                    claude_market::OrderType::RequestCompute
                } else {
                    claude_market::OrderType::OfferCompute
                };
                
                let price = 50 + ((user_idx + order_idx) % 100) as u64;
                let units = 10 + ((user_idx + order_idx) % 50) as u64;
                
                let result = market_clone.place_order(
                    order_type,
                    user_id,
                    price,
                    units,
                    task_spec,
                    None,
                    None,
                    Some(&signing_key),
                ).await;
                
                let latency = operation_start.elapsed();
                (result.is_ok(), latency)
            });
            
            tasks.push(task);
        }
    }

    // Execute all tasks and collect results
    let results = futures::future::join_all(tasks).await;
    
    for result in results {
        match result {
            Ok((success, latency)) => {
                latencies.push(latency);
                if success {
                    successful_operations += 1;
                } else {
                    failed_operations += 1;
                }
            }
            Err(_) => {
                failed_operations += 1;
            }
        }
    }

    let total_duration = start_time.elapsed();
    let end_memory = measure_memory_usage().await;
    
    // Wait for any background processing
    sleep(Duration::from_millis(500)).await;
    
    // Check assignments created
    let assignments = market.get_assignments(None, 10000).await.unwrap();

    let benchmark_result = BenchmarkResult {
        total_operations: successful_operations + failed_operations,
        successful_operations,
        failed_operations,
        total_duration,
        operations_per_second: successful_operations as f64 / total_duration.as_secs_f64(),
        average_latency: Duration::from_nanos(
            latencies.iter().map(|d| d.as_nanos() as u64).sum::<u64>() / latencies.len().max(1) as u64
        ),
        p95_latency: calculate_percentile(latencies.clone(), 95.0),
        p99_latency: calculate_percentile(latencies, 99.0),
        memory_usage_mb: end_memory - start_memory,
        assignments_created: assignments.len() as u64,
    };

    benchmark_result.print_summary("Order Placement Throughput");

    // Performance assertions
    assert!(benchmark_result.operations_per_second > 50.0, 
            "Throughput too low: {} ops/sec", benchmark_result.operations_per_second);
    assert!(benchmark_result.p95_latency < Duration::from_millis(1000), 
            "P95 latency too high: {:?}", benchmark_result.p95_latency);
    assert!(benchmark_result.successful_operations > benchmark_result.total_operations * 8 / 10, 
            "Success rate too low: {}%", 
            (benchmark_result.successful_operations * 100 / benchmark_result.total_operations));
}

#[tokio::test]
async fn benchmark_matching_engine_performance() {
    use benchmark_utils::*;
    
    let (_temp_dir, db_path) = create_test_environment();
    let market = Arc::new(claude_market::Market::new(&db_path).await.unwrap());
    market.init_schema().await.unwrap();

    println!("Starting matching engine performance benchmark...");

    let num_requests = 500;
    let num_offers = 500;
    let users = generate_test_users(100);

    // Opt in all users
    for (user_id, _) in &users {
        market.opt_in_compute_sharing(user_id, true).await.unwrap();
    }

    // Phase 1: Create all offers first
    println!("Creating {} offers...", num_offers);
    let offer_start = Instant::now();
    
    for i in 0..num_offers {
        let (user_id, signing_key) = &users[i % users.len()];
        let task_spec = create_task_spec("benchmark_task", i as u64);
        
        market.place_order(
            claude_market::OrderType::OfferCompute,
            *user_id,
            50 + (i % 100) as u64, // Varied pricing
            20 + (i % 80) as u64,  // Varied capacity
            task_spec,
            None,
            None,
            Some(signing_key),
        ).await.unwrap();
    }
    
    let offer_duration = offer_start.elapsed();
    println!("Created {} offers in {:?}", num_offers, offer_duration);

    // Phase 2: Create requests that should trigger matching
    println!("Creating {} requests to trigger matching...", num_requests);
    let matching_start = Instant::now();
    let mut matching_latencies = Vec::new();
    
    for i in 0..num_requests {
        let request_start = Instant::now();
        
        let (user_id, signing_key) = &users[(i + 50) % users.len()]; // Different users for requests
        let task_spec = create_task_spec("benchmark_task", i as u64);
        
        market.place_order(
            claude_market::OrderType::RequestCompute,
            *user_id,
            80 + (i % 50) as u64, // Higher prices to ensure matching
            15 + (i % 35) as u64,
            task_spec,
            None,
            None,
            Some(signing_key),
        ).await.unwrap();
        
        matching_latencies.push(request_start.elapsed());
    }
    
    let total_matching_duration = matching_start.elapsed();
    
    // Wait for all matching to complete
    sleep(Duration::from_millis(1000)).await;
    
    // Check results
    let assignments = market.get_assignments(None, 10000).await.unwrap();
    
    let matching_result = BenchmarkResult {
        total_operations: num_requests as u64,
        successful_operations: assignments.len() as u64,
        failed_operations: num_requests as u64 - assignments.len() as u64,
        total_duration: total_matching_duration,
        operations_per_second: num_requests as f64 / total_matching_duration.as_secs_f64(),
        average_latency: Duration::from_nanos(
            matching_latencies.iter().map(|d| d.as_nanos() as u64).sum::<u64>() 
            / matching_latencies.len().max(1) as u64
        ),
        p95_latency: calculate_percentile(matching_latencies.clone(), 95.0),
        p99_latency: calculate_percentile(matching_latencies, 99.0),
        memory_usage_mb: measure_memory_usage().await,
        assignments_created: assignments.len() as u64,
    };

    matching_result.print_summary("Matching Engine Performance");

    // Calculate matching rate
    let matching_rate = assignments.len() as f64 / num_requests as f64;
    println!("Matching rate: {:.2}% ({} assignments from {} requests)", 
             matching_rate * 100.0, assignments.len(), num_requests);

    // Performance assertions
    assert!(matching_result.operations_per_second > 20.0, 
            "Matching throughput too low: {} ops/sec", matching_result.operations_per_second);
    assert!(matching_rate > 0.5, "Matching rate too low: {:.2}%", matching_rate * 100.0);
    assert!(matching_result.average_latency < Duration::from_millis(500), 
            "Average matching latency too high: {:?}", matching_result.average_latency);
}

#[tokio::test]
async fn benchmark_concurrent_task_execution() {
    use benchmark_utils::*;
    
    let (_temp_dir, db_path) = create_test_environment();
    let market = Arc::new(claude_market::Market::new(&db_path).await.unwrap());
    market.init_schema().await.unwrap();

    println!("Starting concurrent task execution benchmark...");

    let num_concurrent_tasks = 100;
    let users = generate_test_users(20);

    // Opt in all users
    for (user_id, _) in &users {
        market.opt_in_compute_sharing(user_id, true).await.unwrap();
    }

    // Create matching request/offer pairs
    let mut assignment_ids = Vec::new();
    
    for i in 0..num_concurrent_tasks {
        let requester = &users[i % users.len()];
        let provider = &users[(i + 1) % users.len()];
        
        let task_spec = create_task_spec("concurrent_task", i as u64);
        
        // Create request
        let request = market.place_order(
            claude_market::OrderType::RequestCompute,
            requester.0,
            100,
            50,
            task_spec.clone(),
            None,
            None,
            Some(&requester.1),
        ).await.unwrap();
        
        // Create matching offer
        let offer = market.place_order(
            claude_market::OrderType::OfferCompute,
            provider.0,
            80,
            100,
            task_spec,
            None,
            None,
            Some(&provider.1),
        ).await.unwrap();
    }
    
    // Wait for assignments to be created
    sleep(Duration::from_millis(500)).await;
    let assignments = market.get_assignments(None, num_concurrent_tasks as u32 * 2).await.unwrap();
    
    println!("Created {} assignments for concurrent execution", assignments.len());

    // Phase 1: Start all tasks concurrently
    let start_time = Instant::now();
    let mut start_tasks = Vec::new();
    
    for assignment in &assignments {
        let market_clone = Arc::clone(&market);
        let assignment_id = assignment.id;
        let provider = assignment.provider;
        
        let task = tokio::spawn(async move {
            let start_time = Instant::now();
            let result = market_clone.start_task(&assignment_id, &provider).await;
            (result.is_ok(), start_time.elapsed())
        });
        
        start_tasks.push(task);
    }
    
    // Execute all start operations
    let start_results = futures::future::join_all(start_tasks).await;
    let start_duration = start_time.elapsed();
    
    let successful_starts = start_results.iter()
        .filter_map(|r| r.as_ref().ok())
        .filter(|(success, _)| *success)
        .count();
    
    println!("Started {}/{} tasks in {:?}", successful_starts, assignments.len(), start_duration);

    // Phase 2: Simulate task execution with varying completion times
    sleep(Duration::from_millis(100)).await; // Simulate processing time

    // Phase 3: Complete all tasks concurrently
    let completion_start = Instant::now();
    let mut completion_tasks = Vec::new();
    
    for assignment in &assignments {
        let market_clone = Arc::clone(&market);
        let assignment_id = assignment.id;
        let provider = assignment.provider;
        
        let task = tokio::spawn(async move {
            let completion_time = Instant::now();
            
            let quality_scores = HashMap::from([
                ("accuracy".to_string(), 0.90 + (rand::random::<f64>() * 0.1)),
                ("completeness".to_string(), 0.85 + (rand::random::<f64>() * 0.15)),
            ]);
            
            let result = market_clone.complete_task(&assignment_id, &provider, quality_scores).await;
            (result.is_ok(), completion_time.elapsed())
        });
        
        completion_tasks.push(task);
    }
    
    // Execute all completion operations
    let completion_results = futures::future::join_all(completion_tasks).await;
    let total_completion_duration = completion_start.elapsed();
    
    let successful_completions = completion_results.iter()
        .filter_map(|r| r.as_ref().ok())
        .filter(|(success, _)| *success)
        .count();
    
    let completion_latencies: Vec<Duration> = completion_results.iter()
        .filter_map(|r| r.as_ref().ok())
        .map(|(_, latency)| *latency)
        .collect();

    let execution_result = BenchmarkResult {
        total_operations: assignments.len() as u64,
        successful_operations: successful_completions as u64,
        failed_operations: assignments.len() as u64 - successful_completions as u64,
        total_duration: total_completion_duration,
        operations_per_second: successful_completions as f64 / total_completion_duration.as_secs_f64(),
        average_latency: Duration::from_nanos(
            completion_latencies.iter().map(|d| d.as_nanos() as u64).sum::<u64>() 
            / completion_latencies.len().max(1) as u64
        ),
        p95_latency: calculate_percentile(completion_latencies.clone(), 95.0),
        p99_latency: calculate_percentile(completion_latencies, 99.0),
        memory_usage_mb: measure_memory_usage().await,
        assignments_created: successful_completions as u64,
    };

    execution_result.print_summary("Concurrent Task Execution");

    // Verify task completion consistency
    let final_assignments = market.get_assignments(None, num_concurrent_tasks as u32 * 2).await.unwrap();
    let completed_count = final_assignments.iter()
        .filter(|a| matches!(a.status, claude_market::AssignmentStatus::Completed))
        .count();
    
    println!("Final completion status: {}/{} tasks completed", completed_count, assignments.len());

    // Performance assertions
    assert!(execution_result.operations_per_second > 30.0, 
            "Task completion throughput too low: {} ops/sec", execution_result.operations_per_second);
    assert!(successful_completions > assignments.len() * 8 / 10, 
            "Task completion rate too low: {}%", 
            (successful_completions * 100 / assignments.len()));
}

#[tokio::test]
async fn benchmark_database_performance() {
    use benchmark_utils::*;
    
    let (_temp_dir, db_path) = create_test_environment();
    let market = Arc::new(claude_market::Market::new(&db_path).await.unwrap());
    market.init_schema().await.unwrap();

    println!("Starting database performance benchmark...");

    let num_operations = 10000;
    let concurrent_queries = 50;
    let users = generate_test_users(100);

    // Opt in all users
    for (user_id, _) in &users {
        market.opt_in_compute_sharing(user_id, true).await.unwrap();
    }

    // Phase 1: Insert performance
    println!("Testing insert performance with {} operations...", num_operations);
    let insert_start = Instant::now();
    
    for i in 0..num_operations {
        let (user_id, signing_key) = &users[i % users.len()];
        let task_spec = create_task_spec("db_test", i as u64);
        
        market.place_order(
            claude_market::OrderType::OfferCompute,
            *user_id,
            50 + (i % 200) as u64,
            10 + (i % 90) as u64,
            task_spec,
            None,
            None,
            Some(signing_key),
        ).await.unwrap();
        
        if i % 1000 == 0 {
            println!("Inserted {} orders...", i);
        }
    }
    
    let insert_duration = insert_start.elapsed();
    let insert_rate = num_operations as f64 / insert_duration.as_secs_f64();
    
    println!("Insert performance: {} ops/sec ({:?} total)", insert_rate, insert_duration);

    // Phase 2: Query performance under load
    println!("Testing concurrent query performance...");
    let query_start = Instant::now();
    let mut query_tasks = Vec::new();
    
    for i in 0..concurrent_queries {
        let market_clone = Arc::clone(&market);
        let user_id = users[i % users.len()].0;
        
        let task = tokio::spawn(async move {
            let query_start = Instant::now();
            
            // Mix different types of queries
            match i % 4 {
                0 => {
                    // Get trader orders
                    let _ = market_clone.get_trader_orders(&user_id, None, true).await;
                }
                1 => {
                    // Get assignments
                    let _ = market_clone.get_assignments(Some(&user_id), 100).await;
                }
                2 => {
                    // Get price discovery
                    let _ = market_clone.get_price_discovery("db_test").await;
                }
                3 => {
                    // Get order book (expensive query)
                    let _ = market_clone.get_order_book().await;
                }
                _ => {}
            }
            
            query_start.elapsed()
        });
        
        query_tasks.push(task);
    }
    
    let query_results = futures::future::join_all(query_tasks).await;
    let query_duration = query_start.elapsed();
    
    let query_latencies: Vec<Duration> = query_results.into_iter()
        .filter_map(|r| r.ok())
        .collect();
    
    let avg_query_latency = Duration::from_nanos(
        query_latencies.iter().map(|d| d.as_nanos() as u64).sum::<u64>() 
        / query_latencies.len().max(1) as u64
    );

    // Phase 3: Complex aggregation performance
    println!("Testing complex aggregation queries...");
    let aggregation_start = Instant::now();
    
    // Test price discovery update (complex aggregation)
    market.update_price_discovery("db_test").await.unwrap();
    let price_data = market.get_price_discovery("db_test").await.unwrap();
    
    let aggregation_duration = aggregation_start.elapsed();
    
    println!("Database benchmark results:");
    println!("  Insert rate: {:.2} ops/sec", insert_rate);
    println!("  Average query latency: {:?}", avg_query_latency);
    println!("  P95 query latency: {:?}", calculate_percentile(query_latencies.clone(), 95.0));
    println!("  P99 query latency: {:?}", calculate_percentile(query_latencies, 99.0));
    println!("  Aggregation time: {:?}", aggregation_duration);
    println!("  Total orders in DB: {}", num_operations);
    
    if let Some(price_info) = price_data {
        println!("  Price discovery - Volume: {}, Assignments: {}", 
                 price_info.total_volume, price_info.assignment_count);
    }

    // Performance assertions
    assert!(insert_rate > 100.0, "Insert rate too low: {} ops/sec", insert_rate);
    assert!(avg_query_latency < Duration::from_millis(100), 
            "Average query latency too high: {:?}", avg_query_latency);
    assert!(aggregation_duration < Duration::from_millis(1000), 
            "Aggregation too slow: {:?}", aggregation_duration);
}

#[tokio::test]
async fn benchmark_memory_usage_and_scalability() {
    use benchmark_utils::*;
    
    let (_temp_dir, db_path) = create_test_environment();
    let market = Arc::new(claude_market::Market::new(&db_path).await.unwrap());
    market.init_schema().await.unwrap();

    println!("Starting memory usage and scalability benchmark...");

    let initial_memory = measure_memory_usage().await;
    println!("Initial memory usage: {:.2} MB", initial_memory);

    let mut memory_samples = Vec::new();
    let batch_size = 1000;
    let num_batches = 10;
    
    let users = generate_test_users(500);
    
    // Opt in all users
    for (user_id, _) in &users {
        market.opt_in_compute_sharing(user_id, true).await.unwrap();
    }
    
    let post_init_memory = measure_memory_usage().await;
    memory_samples.push(("Initial + Users", post_init_memory));

    // Load test in batches to monitor memory growth
    for batch in 0..num_batches {
        println!("Processing batch {} of {}...", batch + 1, num_batches);
        let batch_start = Instant::now();
        
        // Create orders in batch
        for i in 0..batch_size {
            let global_index = batch * batch_size + i;
            let (user_id, signing_key) = &users[global_index % users.len()];
            
            let task_spec = claude_market::ComputeTaskSpec {
                task_type: format!("batch_task_{}", global_index % 10),
                compute_units: 10 + (global_index % 100) as u64,
                max_duration_secs: 300,
                required_capabilities: vec![format!("skill_{}", global_index % 5)],
                min_reputation: None,
                privacy_level: claude_market::PrivacyLevel::Public,
                encrypted_payload: if global_index % 10 == 0 {
                    Some(vec![0u8; 1024]) // 1KB payload for some orders
                } else {
                    None
                },
            };
            
            let order_type = if global_index % 2 == 0 {
                claude_market::OrderType::RequestCompute
            } else {
                claude_market::OrderType::OfferCompute
            };
            
            market.place_order(
                order_type,
                *user_id,
                50 + (global_index % 150) as u64,
                20 + (global_index % 80) as u64,
                task_spec,
                None,
                None,
                Some(signing_key),
            ).await.unwrap();
        }
        
        let batch_duration = batch_start.elapsed();
        let batch_memory = measure_memory_usage().await;
        
        memory_samples.push((
            &format!("Batch {} ({} orders)", batch + 1, (batch + 1) * batch_size),
            batch_memory
        ));
        
        println!("  Batch {} completed in {:?}, memory: {:.2} MB", 
                 batch + 1, batch_duration, batch_memory);
        
        // Process any background tasks
        sleep(Duration::from_millis(100)).await;
    }

    // Final measurements
    let total_orders = num_batches * batch_size;
    let final_memory = measure_memory_usage().await;
    let assignments = market.get_assignments(None, (total_orders * 2) as u32).await.unwrap();
    
    // Memory analysis
    println!("\nMemory Usage Analysis:");
    for (phase, memory) in &memory_samples {
        println!("  {}: {:.2} MB (+{:.2} MB)", 
                 phase, memory, memory - initial_memory);
    }
    
    let memory_per_order = (final_memory - initial_memory) / total_orders as f64;
    let total_memory_growth = final_memory - initial_memory;
    
    println!("\nScalability Metrics:");
    println!("  Total orders: {}", total_orders);
    println!("  Total assignments: {}", assignments.len());
    println!("  Total memory growth: {:.2} MB", total_memory_growth);
    println!("  Memory per order: {:.4} MB", memory_per_order);
    println!("  Final memory usage: {:.2} MB", final_memory);

    // Test cleanup efficiency
    println!("\nTesting cleanup efficiency...");
    let cleanup_start = Instant::now();
    
    // Simulate cleanup by expiring old orders
    market.process_expired_auctions().await.unwrap();
    
    let cleanup_duration = cleanup_start.elapsed();
    let post_cleanup_memory = measure_memory_usage().await;
    
    println!("  Cleanup duration: {:?}", cleanup_duration);
    println!("  Memory after cleanup: {:.2} MB ({:.2} MB reduction)", 
             post_cleanup_memory, final_memory - post_cleanup_memory);

    // Performance assertions
    assert!(memory_per_order < 0.01, "Memory usage per order too high: {:.4} MB", memory_per_order);
    assert!(total_memory_growth < 100.0, "Total memory growth too high: {:.2} MB", total_memory_growth);
    assert!(cleanup_duration < Duration::from_millis(5000), 
            "Cleanup too slow: {:?}", cleanup_duration);
    
    // Verify system is still responsive after load
    let responsiveness_start = Instant::now();
    let test_user = &users[0];
    let test_order = market.place_order(
        claude_market::OrderType::OfferCompute,
        test_user.0,
        100,
        10,
        create_task_spec("responsiveness_test", 1),
        None,
        None,
        Some(&test_user.1),
    ).await;
    let responsiveness_time = responsiveness_start.elapsed();
    
    assert!(test_order.is_ok(), "System not responsive after load test");
    assert!(responsiveness_time < Duration::from_millis(1000), 
            "System response too slow after load: {:?}", responsiveness_time);
    
    println!("✓ System remains responsive after load test: {:?}", responsiveness_time);
}

#[tokio::test]
async fn benchmark_reputation_system_performance() {
    use benchmark_utils::*;
    
    let (_temp_dir, db_path) = create_test_environment();
    let market = Arc::new(claude_market::Market::new(&db_path).await.unwrap());
    market.init_schema().await.unwrap();

    println!("Starting reputation system performance benchmark...");

    let num_users = 1000;
    let events_per_user = 20;
    let users = generate_test_users(num_users);

    // Initialize reputation system
    market.reputation.init_schema().await.unwrap();

    // Opt in all users
    for (user_id, _) in &users {
        market.opt_in_compute_sharing(user_id, true).await.unwrap();
    }

    // Phase 1: Reputation event processing performance
    println!("Processing {} reputation events for {} users...", events_per_user, num_users);
    let reputation_start = Instant::now();
    let mut reputation_latencies = Vec::new();
    
    for (user_id, _) in &users {
        for event_idx in 0..events_per_user {
            let event_start = Instant::now();
            
            let event = match event_idx % 5 {
                0 => claude_market::reputation::ReputationEvent::TradeCompleted,
                1 => claude_market::reputation::ReputationEvent::SLAViolation,
                2 => claude_market::reputation::ReputationEvent::FastResponse,
                3 => claude_market::reputation::ReputationEvent::HighQualityWork,
                _ => claude_market::reputation::ReputationEvent::TradeCompleted,
            };
            
            let quality_score = if event_idx % 5 == 0 {
                Some(80.0 + (event_idx as f64 * 2.0))
            } else {
                None
            };
            
            market.reputation.record_event(user_id, event, quality_score, None).await.unwrap();
            reputation_latencies.push(event_start.elapsed());
        }
    }
    
    let reputation_duration = reputation_start.elapsed();
    let total_events = num_users * events_per_user;
    let reputation_ops_per_sec = total_events as f64 / reputation_duration.as_secs_f64();

    // Phase 2: Reputation lookup performance
    println!("Testing reputation lookup performance...");
    let lookup_start = Instant::now();
    let mut lookup_latencies = Vec::new();
    
    for (user_id, _) in &users {
        let lookup_time = Instant::now();
        let reputation = market.reputation.get_reputation(user_id).await.unwrap();
        lookup_latencies.push(lookup_time.elapsed());
        
        // Verify reputation calculation
        assert!(reputation.score >= 0.0 && reputation.score <= 100.0);
        assert!(reputation.total_trades > 0);
    }
    
    let lookup_duration = lookup_start.elapsed();
    let lookup_ops_per_sec = num_users as f64 / lookup_duration.as_secs_f64();

    // Phase 3: Bulk reputation-based filtering
    println!("Testing reputation-based order filtering...");
    let filtering_start = Instant::now();
    
    // Create orders that require reputation filtering
    let high_reputation_users: Vec<_> = users.iter().take(100).collect();
    
    for (user_id, signing_key) in &high_reputation_users {
        let task_spec = claude_market::ComputeTaskSpec {
            task_type: "high_rep_task".to_string(),
            compute_units: 50,
            max_duration_secs: 300,
            required_capabilities: vec![],
            min_reputation: Some(70.0), // Require high reputation
            privacy_level: claude_market::PrivacyLevel::Confidential,
            encrypted_payload: None,
        };
        
        market.place_order(
            claude_market::OrderType::RequestCompute,
            **user_id,
            200,
            50,
            task_spec,
            None,
            None,
            Some(signing_key),
        ).await.unwrap();
    }
    
    let filtering_duration = filtering_start.elapsed();

    // Results
    let reputation_result = BenchmarkResult {
        total_operations: total_events as u64,
        successful_operations: total_events as u64,
        failed_operations: 0,
        total_duration: reputation_duration,
        operations_per_second: reputation_ops_per_sec,
        average_latency: Duration::from_nanos(
            reputation_latencies.iter().map(|d| d.as_nanos() as u64).sum::<u64>() 
            / reputation_latencies.len().max(1) as u64
        ),
        p95_latency: calculate_percentile(reputation_latencies.clone(), 95.0),
        p99_latency: calculate_percentile(reputation_latencies, 99.0),
        memory_usage_mb: measure_memory_usage().await,
        assignments_created: 0,
    };

    reputation_result.print_summary("Reputation Event Processing");

    println!("Reputation Lookup Performance:");
    println!("  Lookup rate: {:.2} ops/sec", lookup_ops_per_sec);
    println!("  Average lookup latency: {:?}", 
             Duration::from_nanos(lookup_latencies.iter().map(|d| d.as_nanos() as u64).sum::<u64>() 
                                   / lookup_latencies.len().max(1) as u64));
    println!("  P95 lookup latency: {:?}", calculate_percentile(lookup_latencies.clone(), 95.0));
    println!("  Reputation filtering time: {:?}", filtering_duration);

    // Performance assertions
    assert!(reputation_ops_per_sec > 500.0, 
            "Reputation processing too slow: {} ops/sec", reputation_ops_per_sec);
    assert!(lookup_ops_per_sec > 1000.0, 
            "Reputation lookup too slow: {} ops/sec", lookup_ops_per_sec);
    assert!(reputation_result.average_latency < Duration::from_millis(10), 
            "Average reputation processing latency too high: {:?}", reputation_result.average_latency);
    assert!(filtering_duration < Duration::from_millis(5000), 
            "Reputation filtering too slow: {:?}", filtering_duration);
}