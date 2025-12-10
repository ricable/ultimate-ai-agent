use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ran_opt::pfs_logs::{LogAnomalyDetector, DetectorConfig};
use std::time::Duration;

fn benchmark_log_processing(c: &mut Criterion) {
    let config = DetectorConfig {
        sequence_length: 128,
        embedding_dim: 64,
        num_heads: 4,
        num_layers: 2,
        hidden_dim: 256,
        window_size: 16,
        vocab_size: 5000,
        quantization_bits: 8,
        anomaly_threshold: 0.8,
        learning_rate: 0.001,
        buffer_size: 1000,
    };

    let mut detector = LogAnomalyDetector::new(config);

    // Generate sample logs
    let sample_logs = generate_sample_logs();
    
    let mut group = c.benchmark_group("log_processing");
    group.measurement_time(Duration::from_secs(10));
    
    // Benchmark different batch sizes
    for batch_size in [1, 10, 50, 100, 500].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));
        
        let logs: Vec<String> = sample_logs.iter()
            .cycle()
            .take(*batch_size)
            .cloned()
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("process_logs", batch_size),
            &logs,
            |b, logs| {
                b.iter(|| {
                    detector.process_logs(black_box(logs))
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_incremental_learning(c: &mut Criterion) {
    let config = DetectorConfig {
        sequence_length: 64,
        embedding_dim: 32,
        num_heads: 2,
        num_layers: 1,
        hidden_dim: 128,
        window_size: 8,
        vocab_size: 1000,
        quantization_bits: 8,
        anomaly_threshold: 0.8,
        learning_rate: 0.01,
        buffer_size: 500,
    };

    let mut detector = LogAnomalyDetector::new(config);

    let training_logs = vec![
        "2024-01-04 10:00:00 INFO: Normal operation".to_string(),
        "2024-01-04 10:00:01 ERROR: Critical failure".to_string(),
    ];
    
    let labels = vec![false, true];

    c.bench_function("incremental_update", |b| {
        b.iter(|| {
            detector.incremental_update(
                black_box(&training_logs),
                black_box(&labels)
            );
        });
    });
}

fn benchmark_tokenization(c: &mut Criterion) {
    use ran_opt::pfs_logs::tokenizer::BPETokenizer;
    
    let mut tokenizer = BPETokenizer::new(5000);
    
    // Train tokenizer
    let corpus = generate_sample_logs();
    tokenizer.train(&corpus, 100);
    
    let test_texts = vec![
        "Simple log message",
        "2024-01-04 10:15:23 ERROR: Connection timeout on node RBS_01",
        "AMOS alt cell=12345 state=active power=85.2 throughput=1.5Gbps",
        "Complex log with many parameters: cpu=85.2% memory=78.3% disk=45.1% network=1.2Gbps temperature=65.4°C",
    ];

    let mut group = c.benchmark_group("tokenization");
    
    for (i, text) in test_texts.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("encode", i),
            text,
            |b, text| {
                b.iter(|| {
                    tokenizer.encode(black_box(text))
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_attention_mechanisms(c: &mut Criterion) {
    use ran_opt::pfs_logs::attention::{SlidingWindowAttention, CustomAttentionKernel, AttentionConfig};
    use ndarray::Array3;
    
    let attention = SlidingWindowAttention::new(128, 8, 32);
    
    let mut group = c.benchmark_group("attention");
    
    // Benchmark different sequence lengths
    for seq_len in [16, 32, 64, 128].iter() {
        let query = Array3::from_shape_fn((1, *seq_len, 128), |(_, _, _)| {
            rand::random::<f32>()
        });
        let key = query.clone();
        let value = query.clone();
        
        group.bench_with_input(
            BenchmarkId::new("sliding_window", seq_len),
            &(*seq_len, &query, &key, &value),
            |b, (_, q, k, v)| {
                b.iter(|| {
                    attention.forward(black_box(q), black_box(k), black_box(v))
                });
            },
        );
    }
    
    // Benchmark custom attention kernel
    let kernel = CustomAttentionKernel::new(AttentionConfig::default());
    
    for seq_len in [16, 32, 64].iter() {
        let query = ndarray::Array2::from_shape_fn((*seq_len, 64), |(_, _)| {
            rand::random::<f32>()
        });
        let key = query.clone();
        let value = query.clone();
        
        group.bench_with_input(
            BenchmarkId::new("custom_kernel", seq_len),
            &(*seq_len, &query, &key, &value),
            |b, (_, q, k, v)| {
                b.iter(|| {
                    kernel.compute_attention(black_box(q), black_box(k), black_box(v), 16, 0.125)
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_anomaly_scoring(c: &mut Criterion) {
    use ran_opt::pfs_logs::anomaly::AnomalyScorer;
    use ndarray::Array3;
    
    let scorer = AnomalyScorer::new(128, 0.8);
    
    let mut group = c.benchmark_group("anomaly_scoring");
    
    for seq_len in [16, 32, 64, 128].iter() {
        let encoded = Array3::from_shape_fn((1, *seq_len, 128), |(_, _, _)| {
            rand::random::<f32>()
        });
        
        group.bench_with_input(
            BenchmarkId::new("score", seq_len),
            &encoded,
            |b, encoded| {
                b.iter(|| {
                    scorer.score(black_box(encoded))
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_log_parsing(c: &mut Criterion) {
    use ran_opt::pfs_logs::parser::EricssonLogParser;
    
    let parser = EricssonLogParser::new();
    
    let test_logs = vec![
        "2024-01-04 10:15:23 INFO: Normal operation",
        "2024-01-04 10:15:24 ERROR: Connection failed",
        "2024-01-04 10:15:25 AMOS alt cell=12345 state=active",
        "2024-01-04 10:15:26 AMOS lget mo=RncFunction=1,UtranCell=12345",
        "2024-01-04 10:15:27 AMOS cvc connect node=RBS_01 status=ok",
        "<134>Jan  4 10:15:28 hostname process: syslog format message",
        "Complex log with key=value pairs: cpu=85.2 memory=78.3 throughput=1.5Gbps",
    ];
    
    let mut group = c.benchmark_group("log_parsing");
    
    for (i, log) in test_logs.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("parse", i),
            log,
            |b, log| {
                b.iter(|| {
                    parser.parse(black_box(log))
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_model_persistence(c: &mut Criterion) {
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
    
    c.bench_function("save_checkpoint", |b| {
        b.iter(|| {
            let path = format!("benchmark_checkpoint_{}.bin", rand::random::<u32>());
            detector.save_checkpoint(black_box(&path)).ok();
            // Clean up
            std::fs::remove_file(&path).ok();
        });
    });
}

fn benchmark_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    
    // Benchmark memory usage with different model sizes
    for (embedding_dim, hidden_dim) in [(32, 64), (64, 128), (128, 256), (256, 512)].iter() {
        let config = DetectorConfig {
            sequence_length: 64,
            embedding_dim: *embedding_dim,
            num_heads: 4,
            num_layers: 2,
            hidden_dim: *hidden_dim,
            window_size: 16,
            vocab_size: 5000,
            quantization_bits: 8,
            anomaly_threshold: 0.8,
            learning_rate: 0.001,
            buffer_size: 1000,
        };
        
        group.bench_with_input(
            BenchmarkId::new("model_creation", format!("{}_{}", embedding_dim, hidden_dim)),
            &config,
            |b, config| {
                b.iter(|| {
                    let _detector = LogAnomalyDetector::new(black_box(config.clone()));
                });
            },
        );
    }
    
    group.finish();
}

fn generate_sample_logs() -> Vec<String> {
    vec![
        "2024-01-04 10:15:23 INFO: System startup completed".to_string(),
        "2024-01-04 10:15:24 ERROR: Connection timeout on node RBS_01".to_string(),
        "2024-01-04 10:15:25 AMOS alt cell=12345 state=active power=85.2".to_string(),
        "2024-01-04 10:15:26 WARN: High CPU usage detected: cpu=92.5%".to_string(),
        "2024-01-04 10:15:27 INFO: lget mo=RncFunction=1,UtranCell=12345".to_string(),
        "2024-01-04 10:15:28 DEBUG: Heartbeat received from managed nodes".to_string(),
        "2024-01-04 10:15:29 ERROR: Database connection failed".to_string(),
        "2024-01-04 10:15:30 INFO: Backup completed successfully".to_string(),
        "2024-01-04 10:15:31 CRITICAL: System temperature exceeded threshold".to_string(),
        "2024-01-04 10:15:32 INFO: Traffic load balancing activated".to_string(),
        "2024-01-04 10:15:33 AMOS cvc connect node=RBS_02 status=connected".to_string(),
        "2024-01-04 10:15:34 WARN: Memory usage above 80%: memory=85.7%".to_string(),
        "2024-01-04 10:15:35 INFO: Performance metrics collected".to_string(),
        "2024-01-04 10:15:36 ERROR: Service restart required".to_string(),
        "2024-01-04 10:15:37 INFO: Scheduled maintenance window started".to_string(),
    ]
}

criterion_group!(
    benches,
    benchmark_log_processing,
    benchmark_incremental_learning,
    benchmark_tokenization,
    benchmark_attention_mechanisms,
    benchmark_anomaly_scoring,
    benchmark_log_parsing,
    benchmark_model_persistence,
    benchmark_memory_usage,
);

criterion_main!(benches);