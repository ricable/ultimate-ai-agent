//! Prime ML benchmarks for distributed training operations
//! Measures performance of gradient aggregation, model updates, and zero-copy operations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use daa_prime_core::types::*;
use tokio::runtime::Runtime;
use rand::Rng;
use std::collections::HashMap;

/// Generate random gradient data for testing
fn generate_gradients(count: usize, size: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| (0..size).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

/// Benchmark gradient aggregation with different numbers of nodes
fn bench_gradient_aggregation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Gradient Aggregation");

    for node_count in [5, 10, 20, 50, 100].iter() {
        for gradient_size in [1000, 10000, 100000].iter() {
            group.throughput(Throughput::Elements(*node_count as u64));

            group.bench_with_input(
                BenchmarkId::new(
                    format!("federated_avg_{}nodes", node_count),
                    gradient_size
                ),
                &(*node_count, *gradient_size),
                |b, &(node_count, gradient_size)| {
                    let gradients = generate_gradients(node_count, gradient_size);

                    b.iter(|| {
                        // Federated averaging
                        let mut aggregated = vec![0.0f32; gradient_size];
                        for gradient in &gradients {
                            for (i, &val) in gradient.iter().enumerate() {
                                aggregated[i] += val;
                            }
                        }
                        for val in &mut aggregated {
                            *val /= node_count as f32;
                        }
                        black_box(aggregated)
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark gradient aggregation with Byzantine fault tolerance
fn bench_byzantine_aggregation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Byzantine-Tolerant Aggregation");

    for node_count in [10, 20, 50].iter() {
        let gradient_size = 10000;
        group.throughput(Throughput::Elements(*node_count as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}nodes", node_count)),
            node_count,
            |b, &node_count| {
                let gradients = generate_gradients(node_count, gradient_size);

                b.iter(|| {
                    // Trimmed mean aggregation (Byzantine-tolerant)
                    let trim_ratio = 0.1;
                    let trim_count = (node_count as f32 * trim_ratio) as usize;

                    let mut aggregated = vec![0.0f32; gradient_size];
                    for i in 0..gradient_size {
                        let mut values: Vec<f32> = gradients.iter().map(|g| g[i]).collect();
                        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                        // Remove outliers
                        let trimmed = &values[trim_count..values.len() - trim_count];
                        let sum: f32 = trimmed.iter().sum();
                        aggregated[i] = sum / trimmed.len() as f32;
                    }

                    black_box(aggregated)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark model update operations
fn bench_model_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("Model Updates");

    for model_size in [1000, 10000, 100000, 1000000].iter() {
        group.throughput(Throughput::Bytes(*model_size as u64 * 4)); // f32 = 4 bytes

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}params", model_size)),
            model_size,
            |b, &model_size| {
                let mut model = vec![0.5f32; model_size];
                let gradient = generate_gradients(1, model_size)[0].clone();
                let learning_rate = 0.01f32;

                b.iter(|| {
                    // Apply gradient update
                    for (param, grad) in model.iter_mut().zip(gradient.iter()) {
                        *param -= learning_rate * grad;
                    }
                    black_box(&model)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark zero-copy operations using shared memory
fn bench_zero_copy_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zero-Copy Operations");

    for data_size in [1024, 10240, 102400, 1024000].iter() {
        group.throughput(Throughput::Bytes(*data_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}bytes", data_size)),
            data_size,
            |b, &data_size| {
                let data = vec![0u8; data_size];

                b.iter(|| {
                    // Simulate zero-copy by using references
                    let slice = &data[..];
                    // Process without copying
                    let checksum: u64 = slice.iter().map(|&b| b as u64).sum();
                    black_box(checksum)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark distributed training coordination
#[cfg(feature = "prime-trainer")]
fn bench_training_coordination(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("Training Coordination");

    group.bench_function("coordinate_5_nodes", |b| {
        b.to_async(&rt).iter(|| async {
            use daa_prime_trainer::Coordinator;

            let coordinator = Coordinator::new(5).await.unwrap();

            // Simulate training round
            coordinator.start_round().await.unwrap();
            coordinator.wait_for_gradients().await.unwrap();
            let aggregated = coordinator.aggregate_gradients().await.unwrap();
            coordinator.distribute_update(&aggregated).await.unwrap();

            black_box(())
        });
    });

    group.bench_function("coordinate_20_nodes", |b| {
        b.to_async(&rt).iter(|| async {
            use daa_prime_trainer::Coordinator;

            let coordinator = Coordinator::new(20).await.unwrap();

            coordinator.start_round().await.unwrap();
            coordinator.wait_for_gradients().await.unwrap();
            let aggregated = coordinator.aggregate_gradients().await.unwrap();
            coordinator.distribute_update(&aggregated).await.unwrap();

            black_box(())
        });
    });

    group.finish();
}

/// Benchmark gradient compression
fn bench_gradient_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("Gradient Compression");

    for gradient_size in [1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*gradient_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}params", gradient_size)),
            gradient_size,
            |b, &gradient_size| {
                let gradient = generate_gradients(1, gradient_size)[0].clone();

                b.iter(|| {
                    // Top-k sparsification (compression)
                    let k = gradient_size / 10; // Keep top 10%
                    let mut indices: Vec<(usize, f32)> = gradient
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| (i, v.abs()))
                        .collect();

                    indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    let sparse: Vec<(usize, f32)> = indices[..k]
                        .iter()
                        .map(|&(i, _)| (i, gradient[i]))
                        .collect();

                    black_box(sparse)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark model serialization and deserialization
fn bench_model_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("Model Serialization");

    for model_size in [1000, 10000, 100000].iter() {
        group.throughput(Throughput::Bytes(*model_size as u64 * 4));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}params", model_size)),
            model_size,
            |b, &model_size| {
                let model = vec![0.5f32; model_size];

                b.iter(|| {
                    // Serialize to bytes
                    let bytes = model
                        .iter()
                        .flat_map(|&f| f.to_le_bytes())
                        .collect::<Vec<u8>>();

                    // Deserialize back
                    let restored: Vec<f32> = bytes
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();

                    black_box(restored)
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "prime-trainer")]
criterion_group!(
    prime_benches,
    bench_gradient_aggregation,
    bench_byzantine_aggregation,
    bench_model_updates,
    bench_zero_copy_operations,
    bench_training_coordination,
    bench_gradient_compression,
    bench_model_serialization
);

#[cfg(not(feature = "prime-trainer"))]
criterion_group!(
    prime_benches,
    bench_gradient_aggregation,
    bench_byzantine_aggregation,
    bench_model_updates,
    bench_zero_copy_operations,
    bench_gradient_compression,
    bench_model_serialization
);

criterion_main!(prime_benches);
