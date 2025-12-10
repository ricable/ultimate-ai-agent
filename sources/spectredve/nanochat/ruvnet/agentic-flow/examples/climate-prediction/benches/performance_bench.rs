// Performance benchmarks using Criterion
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

// Mock data structures (in real implementation, import from crates)
#[derive(Clone, Debug)]
struct ClimateDataPoint {
    timestamp: chrono::DateTime<chrono::Utc>,
    temperature: f64,
    humidity: f64,
    pressure: f64,
    wind_speed: f64,
    precipitation: f64,
}

fn generate_sample_data(size: usize) -> Vec<ClimateDataPoint> {
    (0..size)
        .map(|i| ClimateDataPoint {
            timestamp: chrono::Utc::now() - chrono::Duration::days(i as i64),
            temperature: 20.0 + (i as f64 * 0.1),
            humidity: 60.0,
            pressure: 1013.25,
            wind_speed: 5.0,
            precipitation: 0.0,
        })
        .collect()
}

fn benchmark_data_ingestion(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_ingestion");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let data = generate_sample_data(size);
                black_box(data)
            });
        });
    }

    group.finish();
}

fn benchmark_data_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_validation");

    let data_100 = generate_sample_data(100);
    let data_1000 = generate_sample_data(1000);

    group.bench_function("validate_100", |b| {
        b.iter(|| {
            let valid = data_100.iter().all(|point| {
                point.temperature >= -100.0
                    && point.temperature <= 100.0
                    && point.humidity >= 0.0
                    && point.humidity <= 100.0
            });
            black_box(valid)
        });
    });

    group.bench_function("validate_1000", |b| {
        b.iter(|| {
            let valid = data_1000.iter().all(|point| {
                point.temperature >= -100.0
                    && point.temperature <= 100.0
                    && point.humidity >= 0.0
                    && point.humidity <= 100.0
            });
            black_box(valid)
        });
    });

    group.finish();
}

fn benchmark_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_extraction");

    for size in [24, 168, 720].iter() {
        // 1 day, 1 week, 1 month
        let data = generate_sample_data(*size);

        group.bench_with_input(
            BenchmarkId::new("extract_features", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let features: Vec<f32> = data
                        .iter()
                        .flat_map(|point| {
                            vec![
                                point.temperature as f32,
                                point.humidity as f32,
                                point.pressure as f32,
                                point.wind_speed as f32,
                                point.precipitation as f32,
                            ]
                        })
                        .collect();
                    black_box(features)
                });
            },
        );
    }

    group.finish();
}

fn benchmark_model_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_inference");

    let data_24 = generate_sample_data(24);
    let features_24: Vec<f32> = data_24
        .iter()
        .flat_map(|p| vec![p.temperature as f32])
        .collect();

    let data_168 = generate_sample_data(168);
    let features_168: Vec<f32> = data_168
        .iter()
        .flat_map(|p| vec![p.temperature as f32])
        .collect();

    group.bench_function("inference_24h", |b| {
        b.iter(|| {
            let prediction = mock_inference(&features_24);
            black_box(prediction)
        });
    });

    group.bench_function("inference_168h", |b| {
        b.iter(|| {
            let prediction = mock_inference(&features_168);
            black_box(prediction)
        });
    });

    group.finish();
}

fn benchmark_batch_predictions(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_predictions");

    for batch_size in [10, 50, 100].iter() {
        let batches: Vec<_> = (0..*batch_size)
            .map(|_| generate_sample_data(24))
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batches,
            |b, batches| {
                b.iter(|| {
                    let predictions: Vec<_> = batches
                        .iter()
                        .map(|data| {
                            let features: Vec<f32> =
                                data.iter().map(|p| p.temperature as f32).collect();
                            mock_inference(&features)
                        })
                        .collect();
                    black_box(predictions)
                });
            },
        );
    }

    group.finish();
}

fn benchmark_data_aggregation(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_aggregation");

    let data = generate_sample_data(1000);

    group.bench_function("calculate_stats", |b| {
        b.iter(|| {
            let temps: Vec<f64> = data.iter().map(|p| p.temperature).collect();
            let avg = temps.iter().sum::<f64>() / temps.len() as f64;
            let min = temps.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = temps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            black_box((avg, min, max))
        });
    });

    group.bench_function("moving_average", |b| {
        b.iter(|| {
            let temps: Vec<f64> = data.iter().map(|p| p.temperature).collect();
            let window = 7;
            let moving_avg: Vec<f64> = temps
                .windows(window)
                .map(|w| w.iter().sum::<f64>() / window as f64)
                .collect();

            black_box(moving_avg)
        });
    });

    group.finish();
}

fn benchmark_concurrent_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_operations");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("parallel_predictions", |b| {
        b.iter(|| {
            let handles: Vec<_> = (0..10)
                .map(|_| {
                    std::thread::spawn(|| {
                        let data = generate_sample_data(24);
                        let features: Vec<f32> =
                            data.iter().map(|p| p.temperature as f32).collect();
                        mock_inference(&features)
                    })
                })
                .collect();

            let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
            black_box(results)
        });
    });

    group.finish();
}

fn benchmark_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    group.bench_function("clone_large_dataset", |b| {
        let data = generate_sample_data(10000);
        b.iter(|| {
            let cloned = data.clone();
            black_box(cloned)
        });
    });

    group.bench_function("reference_dataset", |b| {
        let data = generate_sample_data(10000);
        b.iter(|| {
            let referenced = &data;
            let count = referenced.len();
            black_box(count)
        });
    });

    group.finish();
}

// Helper functions
fn mock_inference(features: &[f32]) -> f32 {
    if features.is_empty() {
        return 0.0;
    }
    features.iter().sum::<f32>() / features.len() as f32
}

criterion_group!(
    benches,
    benchmark_data_ingestion,
    benchmark_data_validation,
    benchmark_feature_extraction,
    benchmark_model_inference,
    benchmark_batch_predictions,
    benchmark_data_aggregation,
    benchmark_concurrent_operations,
    benchmark_memory_efficiency
);

criterion_main!(benches);
