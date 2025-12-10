use ran_opt::pfs_logs::{LogAnomalyDetector, DetectorConfig};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the log anomaly detector
    let config = DetectorConfig {
        sequence_length: 256,
        embedding_dim: 128,
        num_heads: 8,
        num_layers: 4,
        hidden_dim: 512,
        window_size: 32,
        vocab_size: 10000,
        quantization_bits: 8,
        anomaly_threshold: 0.85,
        learning_rate: 0.001,
        buffer_size: 5000,
    };

    let mut detector = LogAnomalyDetector::new(config);

    // Sample Ericsson log data
    let sample_logs = vec![
        "2024-01-04 10:15:23 INFO: AMOS alt cell=12345 state=active power=85.2".to_string(),
        "2024-01-04 10:15:24 INFO: Normal system operation, all services running".to_string(),
        "2024-01-04 10:15:25 INFO: lget mo=RncFunction=1,UtranCell=12345 administrativeState=UNLOCKED".to_string(),
        "2024-01-04 10:15:26 WARN: High CPU usage detected: cpu=92.5% memory=78.3%".to_string(),
        "2024-01-04 10:15:27 INFO: cvc connect node=RBS_01 status=connected".to_string(),
        "2024-01-04 10:15:28 ERROR: Connection timeout on node RBS_02 after 30 seconds".to_string(),
        "2024-01-04 10:15:29 ERROR: Failed to execute AMOS command: alt cell=99999 - Cell not found".to_string(),
        "2024-01-04 10:15:30 INFO: Backup completed successfully, 1.2GB transferred".to_string(),
        "2024-01-04 10:15:31 DEBUG: Heartbeat received from all managed nodes".to_string(),
        "2024-01-04 10:15:32 CRITICAL: System temperature exceeded threshold: temp=85.7°C".to_string(),
        "2024-01-04 10:15:33 INFO: Traffic load balancing activated for sector 7".to_string(),
        "2024-01-04 10:15:34 ERROR: Database connection pool exhausted, retrying...".to_string(),
        "2024-01-04 10:15:35 INFO: Scheduled maintenance window started".to_string(),
        "2024-01-04 10:15:36 WARN: Unusual traffic pattern detected in cell 54321".to_string(),
        "2024-01-04 10:15:37 INFO: Performance metrics: throughput=1.2Gbps latency=5.2ms".to_string(),
    ];

    println!("=== Log Anomaly Detection Results ===\n");

    // Process logs and detect anomalies
    let results = detector.process_logs(&sample_logs);

    for (i, result) in results.iter().enumerate() {
        println!("Log {}: {}", i + 1, sample_logs[i]);
        println!("  Timestamp: {}", result.timestamp);
        println!("  Log Type: {}", result.log_type);
        println!("  Anomaly Score: {:.4}", result.score);
        println!("  Detected Patterns: {:?}", result.detected_patterns);
        
        if result.score > 0.8 {
            println!("  🚨 HIGH ANOMALY DETECTED!");
        } else if result.score > 0.6 {
            println!("  ⚠️  Medium anomaly");
        } else {
            println!("  ✅ Normal");
        }
        println!();
    }

    // Demonstrate incremental learning
    println!("=== Incremental Learning Example ===\n");
    
    let training_logs = vec![
        "2024-01-04 11:00:00 INFO: Normal operation continues".to_string(),
        "2024-01-04 11:00:01 ERROR: Critical system failure - immediate attention required".to_string(),
        "2024-01-04 11:00:02 INFO: System recovery successful".to_string(),
        "2024-01-04 11:00:03 ERROR: Out of memory exception in process PID 1234".to_string(),
    ];

    let labels = vec![false, true, false, true]; // false = normal, true = anomaly

    println!("Training on new log data...");
    detector.incremental_update(&training_logs, &labels);
    println!("Training completed!\n");

    // Test on similar patterns
    let test_logs = vec![
        "2024-01-04 12:00:00 ERROR: Critical system failure - service disruption".to_string(),
        "2024-01-04 12:00:01 INFO: Routine maintenance completed".to_string(),
        "2024-01-04 12:00:02 ERROR: Memory allocation failed for process PID 5678".to_string(),
    ];

    println!("Testing on new logs after training:");
    let test_results = detector.process_logs(&test_logs);

    for (i, result) in test_results.iter().enumerate() {
        println!("Test Log {}: {}", i + 1, test_logs[i]);
        println!("  Anomaly Score: {:.4}", result.score);
        if result.score > 0.8 {
            println!("  🚨 HIGH ANOMALY DETECTED!");
        } else if result.score > 0.6 {
            println!("  ⚠️  Medium anomaly");
        } else {
            println!("  ✅ Normal");
        }
        println!();
    }

    // Save model checkpoint
    println!("=== Model Persistence ===\n");
    let checkpoint_path = "pfs_logs_model.bin";
    
    if let Err(e) = detector.save_checkpoint(checkpoint_path) {
        eprintln!("Failed to save checkpoint: {}", e);
    } else {
        println!("Model checkpoint saved to: {}", checkpoint_path);
    }

    // Load model checkpoint
    match LogAnomalyDetector::load_checkpoint(checkpoint_path) {
        Ok(loaded_detector) => {
            println!("Model checkpoint loaded successfully!");
            
            // Test loaded model
            let loaded_results = loaded_detector.process_logs(&vec![
                "2024-01-04 13:00:00 INFO: Testing loaded model".to_string(),
            ]);
            
            println!("Loaded model test result: score = {:.4}", loaded_results[0].score);
        }
        Err(e) => {
            eprintln!("Failed to load checkpoint: {}", e);
        }
    }

    // Generate synthetic anomalous log patterns
    println!("\n=== Synthetic Anomaly Testing ===\n");
    
    let synthetic_logs = generate_synthetic_logs();
    let synthetic_results = detector.process_logs(&synthetic_logs);

    println!("Synthetic log analysis:");
    for (i, result) in synthetic_results.iter().enumerate() {
        println!("Pattern {}: score = {:.4}, type = {}", 
                 i + 1, result.score, result.log_type);
    }

    // Performance analysis
    println!("\n=== Performance Analysis ===\n");
    
    let start_time = std::time::Instant::now();
    let large_batch: Vec<String> = (0..1000)
        .map(|i| format!("2024-01-04 14:{:02}:{:02} INFO: Batch processing log {}", 
                        i / 60, i % 60, i))
        .collect();
    
    let batch_results = detector.process_logs(&large_batch);
    let duration = start_time.elapsed();
    
    println!("Processed {} logs in {:?}", large_batch.len(), duration);
    println!("Average time per log: {:?}", duration / large_batch.len() as u32);
    
    let anomaly_count = batch_results.iter()
        .filter(|r| r.score > 0.8)
        .count();
    
    println!("Detected {} anomalies out of {} logs ({:.2}%)", 
             anomaly_count, large_batch.len(), 
             (anomaly_count as f64 / large_batch.len() as f64) * 100.0);

    // Write detailed results to file
    write_detailed_results(&results, "anomaly_detection_results.json")?;
    
    println!("\nDetailed results written to: anomaly_detection_results.json");
    println!("Log anomaly detection example completed successfully!");

    Ok(())
}

fn generate_synthetic_logs() -> Vec<String> {
    vec![
        // Normal patterns
        "2024-01-04 15:00:00 INFO: System startup completed".to_string(),
        "2024-01-04 15:00:01 INFO: All services initialized".to_string(),
        
        // Anomalous patterns
        "2024-01-04 15:00:02 ERROR: Stack overflow in kernel module".to_string(),
        "2024-01-04 15:00:03 CRITICAL: Hardware failure detected on CPU core 3".to_string(),
        "2024-01-04 15:00:04 ERROR: Segmentation fault in process manager".to_string(),
        
        // Unusual AMOS patterns
        "2024-01-04 15:00:05 AMOS alt cell=99999999 state=UNKNOWN_STATE".to_string(),
        "2024-01-04 15:00:06 AMOS lget mo=INVALID_MO_PATH".to_string(),
        
        // Suspicious network activity
        "2024-01-04 15:00:07 WARN: 1000 failed login attempts from IP 192.168.1.100".to_string(),
        "2024-01-04 15:00:08 ERROR: DDoS attack detected, dropping packets".to_string(),
        
        // Resource exhaustion
        "2024-01-04 15:00:09 CRITICAL: Disk space exhausted on /var/log partition".to_string(),
        "2024-01-04 15:00:10 ERROR: Memory leak detected in service XYZ".to_string(),
    ]
}

fn write_detailed_results(
    results: &[ran_opt::pfs_logs::AnomalyResult], 
    filename: &str
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(filename)?;
    
    writeln!(file, "[")?;
    for (i, result) in results.iter().enumerate() {
        writeln!(file, "  {{")?;
        writeln!(file, "    \"index\": {},", i)?;
        writeln!(file, "    \"timestamp\": \"{}\",", result.timestamp)?;
        writeln!(file, "    \"log_type\": \"{}\",", result.log_type)?;
        writeln!(file, "    \"anomaly_score\": {:.6},", result.score)?;
        writeln!(file, "    \"detected_patterns\": {:?},", result.detected_patterns)?;
        writeln!(file, "    \"attention_weights_shape\": {:?}", result.attention_weights.shape())?;
        
        if i < results.len() - 1 {
            writeln!(file, "  }},")?;
        } else {
            writeln!(file, "  }}")?;
        }
    }
    writeln!(file, "]")?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_detection_pipeline() {
        let config = DetectorConfig::default();
        let mut detector = LogAnomalyDetector::new(config);
        
        let test_logs = vec![
            "2024-01-04 10:00:00 INFO: Normal log".to_string(),
            "2024-01-04 10:00:01 ERROR: Critical error occurred".to_string(),
        ];
        
        let results = detector.process_logs(&test_logs);
        
        assert_eq!(results.len(), 2);
        assert!(results[1].score > results[0].score); // Error should have higher score
    }
}