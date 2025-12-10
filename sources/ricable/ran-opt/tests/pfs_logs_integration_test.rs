use ran_opt::pfs_logs::{LogAnomalyDetector, DetectorConfig};

#[test]
fn test_log_anomaly_detector_integration() {
    let config = DetectorConfig {
        sequence_length: 64,
        embedding_dim: 32,
        num_heads: 2,
        num_layers: 1,
        hidden_dim: 64,
        window_size: 8,
        vocab_size: 1000,
        quantization_bits: 8,
        anomaly_threshold: 0.8,
        learning_rate: 0.001,
        buffer_size: 100,
    };

    let mut detector = LogAnomalyDetector::new(config);

    // Test normal logs
    let normal_logs = vec![
        "2024-01-04 10:15:23 INFO: System startup completed".to_string(),
        "2024-01-04 10:15:24 INFO: All services initialized".to_string(),
        "2024-01-04 10:15:25 INFO: Normal operation in progress".to_string(),
    ];

    let results = detector.process_logs(&normal_logs);
    assert_eq!(results.len(), 3);
    
    // Test anomalous logs
    let anomalous_logs = vec![
        "2024-01-04 10:15:26 ERROR: Critical system failure detected".to_string(),
        "2024-01-04 10:15:27 CRITICAL: Hardware malfunction in CPU core 3".to_string(),
        "2024-01-04 10:15:28 ERROR: Out of memory exception in kernel".to_string(),
    ];

    let anomaly_results = detector.process_logs(&anomalous_logs);
    assert_eq!(anomaly_results.len(), 3);

    // Test AMOS commands
    let amos_logs = vec![
        "2024-01-04 10:15:29 AMOS alt cell=12345 state=active".to_string(),
        "2024-01-04 10:15:30 AMOS lget mo=RncFunction=1,UtranCell=12345".to_string(),
        "2024-01-04 10:15:31 AMOS cvc connect node=RBS_01 status=ok".to_string(),
    ];

    let amos_results = detector.process_logs(&amos_logs);
    assert_eq!(amos_results.len(), 3);

    // Verify that all results have the expected structure
    for result in &results {
        assert!(!result.timestamp.is_empty());
        assert!(!result.log_type.is_empty());
        assert!(result.score >= 0.0 && result.score <= 1.0);
        assert_eq!(result.attention_weights.ndim(), 2);
    }
}

#[test]
fn test_incremental_learning() {
    let config = DetectorConfig {
        sequence_length: 32,
        embedding_dim: 16,
        num_heads: 2,
        num_layers: 1,
        hidden_dim: 32,
        window_size: 8,
        vocab_size: 500,
        quantization_bits: 8,
        anomaly_threshold: 0.8,
        learning_rate: 0.01,
        buffer_size: 50,
    };

    let mut detector = LogAnomalyDetector::new(config);

    // Initial processing
    let initial_logs = vec![
        "2024-01-04 10:00:00 INFO: Normal operation".to_string(),
        "2024-01-04 10:00:01 ERROR: Test error".to_string(),
    ];

    let initial_results = detector.process_logs(&initial_logs);
    let initial_error_score = initial_results[1].score;

    // Train the model
    let training_logs = vec![
        "2024-01-04 11:00:00 INFO: Normal operation continues".to_string(),
        "2024-01-04 11:00:01 ERROR: Critical system failure".to_string(),
        "2024-01-04 11:00:02 INFO: Recovery successful".to_string(),
        "2024-01-04 11:00:03 ERROR: Another critical failure".to_string(),
    ];

    let labels = vec![false, true, false, true];
    detector.incremental_update(&training_logs, &labels);

    // Test after training
    let test_logs = vec![
        "2024-01-04 12:00:00 ERROR: Similar critical failure".to_string(),
    ];

    let test_results = detector.process_logs(&test_logs);
    
    // The model should learn to better identify similar patterns
    assert!(test_results[0].score >= 0.0 && test_results[0].score <= 1.0);
}

#[test]
fn test_model_persistence() {
    let config = DetectorConfig {
        sequence_length: 32,
        embedding_dim: 16,
        num_heads: 2,
        num_layers: 1,
        hidden_dim: 32,
        window_size: 8,
        vocab_size: 500,
        quantization_bits: 8,
        anomaly_threshold: 0.8,
        learning_rate: 0.001,
        buffer_size: 50,
    };

    let detector = LogAnomalyDetector::new(config);

    // Save checkpoint
    let checkpoint_path = "test_checkpoint.bin";
    detector.save_checkpoint(checkpoint_path).expect("Failed to save checkpoint");

    // Load checkpoint
    let loaded_detector = LogAnomalyDetector::load_checkpoint(checkpoint_path)
        .expect("Failed to load checkpoint");

    // Test that loaded model works
    let test_logs = vec![
        "2024-01-04 10:00:00 INFO: Test log".to_string(),
    ];

    let results = loaded_detector.process_logs(&test_logs);
    assert_eq!(results.len(), 1);
    assert!(results[0].score >= 0.0 && results[0].score <= 1.0);

    // Clean up
    std::fs::remove_file(checkpoint_path).ok();
}

#[test]
fn test_ericsson_log_patterns() {
    let config = DetectorConfig::default();
    let detector = LogAnomalyDetector::new(config);

    // Test various Ericsson log patterns
    let ericsson_logs = vec![
        "2024-01-04 10:15:23 AMOS alt cell=12345 state=active power=85.2".to_string(),
        "2024-01-04 10:15:24 AMOS lget mo=RncFunction=1,UtranCell=12345 administrativeState=UNLOCKED".to_string(),
        "2024-01-04 10:15:25 AMOS cvc connect node=RBS_01 status=connected ip=192.168.1.100".to_string(),
        "2024-01-04 10:15:26 SYSLOG <134>Jan  4 10:15:26 hostname process: System message".to_string(),
        "2024-01-04 10:15:27 INFO: Performance metrics: cpu=85.2% memory=78.3% throughput=1.5Gbps".to_string(),
    ];

    let results = detector.process_logs(&ericsson_logs);
    assert_eq!(results.len(), 5);

    // Check that AMOS commands are properly identified
    for result in &results {
        if result.log_type == "AMOS" {
            assert!(result.detected_patterns.contains(&"AMOS_COMMAND".to_string()));
        }
    }
}

#[test]
fn test_anomaly_scoring_consistency() {
    let config = DetectorConfig {
        sequence_length: 32,
        embedding_dim: 16,
        num_heads: 2,
        num_layers: 1,
        hidden_dim: 32,
        window_size: 8,
        vocab_size: 500,
        quantization_bits: 8,
        anomaly_threshold: 0.8,
        learning_rate: 0.001,
        buffer_size: 50,
    };

    let detector = LogAnomalyDetector::new(config);

    // Test that the same log produces consistent scores
    let test_log = vec![
        "2024-01-04 10:15:23 ERROR: Critical system failure".to_string(),
    ];

    let results1 = detector.process_logs(&test_log);
    let results2 = detector.process_logs(&test_log);

    assert_eq!(results1.len(), 1);
    assert_eq!(results2.len(), 1);
    
    // Scores should be identical for the same input
    assert!((results1[0].score - results2[0].score).abs() < 1e-6);
}

#[test]
fn test_large_batch_processing() {
    let config = DetectorConfig {
        sequence_length: 32,
        embedding_dim: 16,
        num_heads: 2,
        num_layers: 1,
        hidden_dim: 32,
        window_size: 8,
        vocab_size: 500,
        quantization_bits: 8,
        anomaly_threshold: 0.8,
        learning_rate: 0.001,
        buffer_size: 50,
    };

    let detector = LogAnomalyDetector::new(config);

    // Generate a large batch of logs
    let large_batch: Vec<String> = (0..100)
        .map(|i| {
            if i % 10 == 0 {
                format!("2024-01-04 10:{:02}:{:02} ERROR: Error log {}", i / 60, i % 60, i)
            } else {
                format!("2024-01-04 10:{:02}:{:02} INFO: Normal log {}", i / 60, i % 60, i)
            }
        })
        .collect();

    let results = detector.process_logs(&large_batch);
    assert_eq!(results.len(), 100);

    // Verify all results are valid
    for result in &results {
        assert!(result.score >= 0.0 && result.score <= 1.0);
        assert!(!result.timestamp.is_empty());
        assert!(!result.log_type.is_empty());
    }
}

#[test]
fn test_tokenizer_integration() {
    use ran_opt::pfs_logs::tokenizer::BPETokenizer;

    let mut tokenizer = BPETokenizer::new(1000);

    // Test basic tokenization
    let test_text = "2024-01-04 10:15:23 ERROR: Connection failed";
    let tokens = tokenizer.encode(test_text);
    let decoded = tokenizer.decode(&tokens);

    assert!(!tokens.is_empty());
    assert!(decoded.contains("ERROR"));

    // Test Ericsson-specific tokenization
    let ericsson_log = "2024-01-04 10:15:23 AMOS alt cell=12345 state=active";
    let ericsson_tokens = tokenizer.tokenize_ericsson_log(ericsson_log);
    
    assert!(!ericsson_tokens.is_empty());
    assert!(ericsson_tokens.len() > tokens.len()); // Should have more tokens due to structured parsing
}

#[test]
fn test_attention_mechanism() {
    use ran_opt::pfs_logs::attention::SlidingWindowAttention;
    use ndarray::Array3;

    let attention = SlidingWindowAttention::new(32, 4, 8);
    
    let batch_size = 1;
    let seq_len = 16;
    let embedding_dim = 32;
    
    let query = Array3::from_shape_fn((batch_size, seq_len, embedding_dim), |(_, _, _)| {
        rand::random::<f32>()
    });
    let key = query.clone();
    let value = query.clone();
    
    let output = attention.forward(&query, &key, &value);
    
    assert_eq!(output.shape(), &[batch_size, seq_len, embedding_dim]);
    
    // Test attention weights extraction
    let weights = attention.get_attention_weights(&query, &key);
    assert_eq!(weights.shape(), &[batch_size, 4, seq_len]); // 4 heads
}

#[test]
fn test_log_parser_integration() {
    use ran_opt::pfs_logs::parser::EricssonLogParser;

    let parser = EricssonLogParser::new();

    // Test AMOS command parsing
    let amos_log = "2024-01-04 10:15:23 AMOS alt cell=12345 state=active";
    let entry = parser.parse(amos_log).expect("Failed to parse AMOS log");
    
    assert_eq!(entry.log_type, "AMOS");
    assert!(entry.amos_command.is_some());
    
    let amos_cmd = entry.amos_command.unwrap();
    assert_eq!(amos_cmd.command, "alt");
    assert_eq!(amos_cmd.parameters.get("cell"), Some(&"12345".to_string()));
    assert_eq!(amos_cmd.parameters.get("state"), Some(&"active".to_string()));

    // Test structured data extraction
    let structured_log = "INFO: Performance data cpu=85.2 memory=78.3 throughput=1.5Gbps";
    let structured_entry = parser.parse(structured_log).expect("Failed to parse structured log");
    
    assert!(structured_entry.structured_data.contains_key("cpu"));
    assert!(structured_entry.structured_data.contains_key("memory"));
    assert!(structured_entry.structured_data.contains_key("throughput"));

    // Test numerical value extraction
    let numerical_values = parser.extract_numerical_values(&structured_log);
    assert!(numerical_values.contains_key("cpu_usage"));
    assert!(numerical_values.contains_key("memory_usage"));
    assert_eq!(numerical_values.get("cpu_usage"), Some(&85.2));
    assert_eq!(numerical_values.get("memory_usage"), Some(&78.3));
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_processing_performance() {
        let config = DetectorConfig {
            sequence_length: 64,
            embedding_dim: 32,
            num_heads: 2,
            num_layers: 1,
            hidden_dim: 64,
            window_size: 8,
            vocab_size: 1000,
            quantization_bits: 8,
            anomaly_threshold: 0.8,
            learning_rate: 0.001,
            buffer_size: 100,
        };

        let detector = LogAnomalyDetector::new(config);

        // Generate test logs
        let test_logs: Vec<String> = (0..1000)
            .map(|i| format!("2024-01-04 10:{:02}:{:02} INFO: Test log {}", i / 60, i % 60, i))
            .collect();

        let start = Instant::now();
        let results = detector.process_logs(&test_logs);
        let duration = start.elapsed();

        assert_eq!(results.len(), 1000);
        
        // Should process at least 100 logs per second
        let logs_per_second = 1000.0 / duration.as_secs_f64();
        assert!(logs_per_second > 100.0, "Processing rate too slow: {:.2} logs/sec", logs_per_second);
        
        println!("Processed {} logs in {:?} ({:.2} logs/sec)", 
                 test_logs.len(), duration, logs_per_second);
    }
}