use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use candle_core::{Device, Tensor};
use ran_opt::afm_detect::{AFMDetector, DetectionMode};

fn create_test_data(batch_size: usize, feature_dim: usize, device: &Device) -> candle_core::Result<Tensor> {
    Tensor::randn(0.0, 1.0, &[batch_size, feature_dim], device)
}

fn bench_detection_modes(c: &mut Criterion) {
    let mut group = c.benchmark_group("detection_modes");
    
    let device = Device::Cpu;
    let detector = AFMDetector::new(64, 16, device.clone()).unwrap();
    
    let modes = vec![
        DetectionMode::KpiKqi,
        DetectionMode::HardwareDegradation,
        DetectionMode::ThermalPower,
        DetectionMode::Combined,
    ];
    
    for mode in modes {
        group.bench_with_input(
            BenchmarkId::new("detection_mode", format!("{:?}", mode)),
            &mode,
            |b, mode| {
                let data = create_test_data(10, 64, &device).unwrap();
                b.iter(|| {
                    detector.detect(black_box(&data), black_box(*mode), None).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_batch_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_sizes");
    
    let device = Device::Cpu;
    let detector = AFMDetector::new(64, 16, device.clone()).unwrap();
    
    let batch_sizes = vec![1, 5, 10, 20, 50, 100];
    
    for batch_size in batch_sizes {
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &batch_size,
            |b, &batch_size| {
                let data = create_test_data(batch_size, 64, &device).unwrap();
                b.iter(|| {
                    detector.detect(black_box(&data), DetectionMode::Combined, None).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_feature_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_dimensions");
    
    let device = Device::Cpu;
    let feature_dims = vec![32, 64, 128, 256, 512];
    
    for feature_dim in feature_dims {
        let detector = AFMDetector::new(feature_dim, feature_dim / 4, device.clone()).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("feature_dim", feature_dim),
            &feature_dim,
            |b, &feature_dim| {
                let data = create_test_data(10, feature_dim, &device).unwrap();
                b.iter(|| {
                    detector.detect(black_box(&data), DetectionMode::Combined, None).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_with_history(c: &mut Criterion) {
    let mut group = c.benchmark_group("with_history");
    
    let device = Device::Cpu;
    let detector = AFMDetector::new(64, 16, device.clone()).unwrap();
    
    let history_lengths = vec![10, 50, 100, 200, 500];
    
    for history_len in history_lengths {
        group.bench_with_input(
            BenchmarkId::new("history_length", history_len),
            &history_len,
            |b, &history_len| {
                let data = create_test_data(10, 64, &device).unwrap();
                let history = create_test_data(history_len, 64, &device).unwrap();
                b.iter(|| {
                    detector.detect(black_box(&data), DetectionMode::Combined, Some(black_box(&history))).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_training_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("training");
    
    let device = Device::Cpu;
    let mut detector = AFMDetector::new(64, 16, device.clone()).unwrap();
    
    // Benchmark training on different dataset sizes
    let dataset_sizes = vec![100, 500, 1000, 2000];
    
    for dataset_size in dataset_sizes {
        group.bench_with_input(
            BenchmarkId::new("training_dataset_size", dataset_size),
            &dataset_size,
            |b, &dataset_size| {
                let data = create_test_data(dataset_size, 64, &device).unwrap();
                b.iter(|| {
                    detector.train_on_normal(black_box(&data), 10, 0.01).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_individual_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("components");
    
    let device = Device::Cpu;
    let detector = AFMDetector::new(64, 16, device.clone()).unwrap();
    let data = create_test_data(10, 64, &device).unwrap();
    
    // This would require making internal components public for benchmarking
    // For now, we'll benchmark the overall detection
    group.bench_function("overall_detection", |b| {
        b.iter(|| {
            detector.detect(black_box(&data), DetectionMode::Combined, None).unwrap()
        });
    });
    
    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    
    let device = Device::Cpu;
    
    // Benchmark memory usage for different model sizes
    let model_sizes = vec![
        (32, 8),
        (64, 16),
        (128, 32),
        (256, 64),
        (512, 128),
    ];
    
    for (input_dim, latent_dim) in model_sizes {
        group.bench_with_input(
            BenchmarkId::new("model_size", format!("{}_{}", input_dim, latent_dim)),
            &(input_dim, latent_dim),
            |b, &(input_dim, latent_dim)| {
                b.iter(|| {
                    let detector = AFMDetector::new(input_dim, latent_dim, device.clone()).unwrap();
                    let data = create_test_data(10, input_dim, &device).unwrap();
                    detector.detect(black_box(&data), DetectionMode::Combined, None).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_concurrent_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_detection");
    
    let device = Device::Cpu;
    let detector = AFMDetector::new(64, 16, device.clone()).unwrap();
    
    // Benchmark concurrent detection calls
    let num_threads = vec![1, 2, 4, 8];
    
    for thread_count in num_threads {
        group.bench_with_input(
            BenchmarkId::new("concurrent_threads", thread_count),
            &thread_count,
            |b, &thread_count| {
                b.iter(|| {
                    let handles: Vec<_> = (0..thread_count).map(|_| {
                        let detector = &detector;
                        let data = create_test_data(10, 64, &device).unwrap();
                        std::thread::spawn(move || {
                            detector.detect(&data, DetectionMode::Combined, None).unwrap()
                        })
                    }).collect();
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn bench_real_world_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_world_scenarios");
    
    let device = Device::Cpu;
    let detector = AFMDetector::new(64, 16, device.clone()).unwrap();
    
    // Scenario 1: High-frequency monitoring (1-second intervals)
    group.bench_function("high_frequency_monitoring", |b| {
        let data = create_test_data(1, 64, &device).unwrap();
        b.iter(|| {
            detector.detect(black_box(&data), DetectionMode::KpiKqi, None).unwrap()
        });
    });
    
    // Scenario 2: Batch processing (hourly aggregated data)
    group.bench_function("batch_processing", |b| {
        let data = create_test_data(3600, 64, &device).unwrap(); // 1 hour of second-level data
        b.iter(|| {
            detector.detect(black_box(&data), DetectionMode::Combined, None).unwrap()
        });
    });
    
    // Scenario 3: Historical analysis with long history
    group.bench_function("historical_analysis", |b| {
        let data = create_test_data(10, 64, &device).unwrap();
        let history = create_test_data(86400, 64, &device).unwrap(); // 24 hours of history
        b.iter(|| {
            detector.detect(black_box(&data), DetectionMode::HardwareDegradation, Some(black_box(&history))).unwrap()
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_detection_modes,
    bench_batch_sizes,
    bench_feature_dimensions,
    bench_with_history,
    bench_training_performance,
    bench_individual_components,
    bench_memory_usage,
    bench_concurrent_detection,
    bench_real_world_scenarios
);

criterion_main!(benches);