//! Benchmarks for HNSW index operations
//!
//! Compares ruvector HNSW implementation against pgvector equivalents

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use ruvector_postgres::index::hnsw::{HnswConfig, HnswIndex};
use ruvector_postgres::distance::DistanceMetric;

// ============================================================================
// Test Data Generation
// ============================================================================

fn generate_random_vectors(n: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            (0..dims)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect()
        })
        .collect()
}

fn generate_clustered_vectors(n: usize, dims: usize, num_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Generate cluster centers
    let centers: Vec<Vec<f32>> = (0..num_clusters)
        .map(|_| {
            (0..dims)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect()
        })
        .collect();

    // Generate vectors around centers
    (0..n)
        .map(|_| {
            let center = &centers[rng.random_range(0..num_clusters)];
            center
                .iter()
                .map(|&c| c + rng.random_range(-0.1..0.1))
                .collect()
        })
        .collect()
}

// ============================================================================
// HNSW Build Benchmarks
// ============================================================================

fn bench_hnsw_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_build");
    group.sample_size(10); // Reduce sample size for slow benchmarks

    for &dims in [128, 384, 768, 1536].iter() {
        for &n in [1000, 10000, 100000].iter() {
            let vectors = generate_random_vectors(n, dims, 42);

            group.bench_with_input(
                BenchmarkId::new(format!("{}d", dims), n),
                &vectors,
                |bench, vecs| {
                    bench.iter(|| {
                        let config = HnswConfig {
                            m: 16,
                            m0: 32,
                            ef_construction: 64,
                            max_elements: n,
                            metric: DistanceMetric::Euclidean,
                            seed: 42,
                            ..Default::default()
                        };

                        let mut index = HnswIndex::new(config);
                        for (id, vec) in vecs.iter().enumerate() {
                            index.insert(id as u64, vec);
                        }
                        black_box(index)
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_hnsw_build_ef_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_build_ef_construction");
    group.sample_size(10);

    let dims = 768;
    let n = 10000;
    let vectors = generate_random_vectors(n, dims, 42);

    for &ef in [16, 32, 64, 128, 256].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(ef),
            &ef,
            |bench, &ef_val| {
                bench.iter(|| {
                    let config = HnswConfig {
                        m: 16,
                        m0: 32,
                        ef_construction: ef_val,
                        max_elements: n,
                        metric: DistanceMetric::Euclidean,
                        seed: 42,
                        ..Default::default()
                    };

                    let mut index = HnswIndex::new(config);
                    for (id, vec) in vectors.iter().enumerate() {
                        index.insert(id as u64, vec);
                    }
                    black_box(index)
                });
            },
        );
    }

    group.finish();
}

fn bench_hnsw_build_m_parameter(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_build_m_parameter");
    group.sample_size(10);

    let dims = 768;
    let n = 10000;
    let vectors = generate_random_vectors(n, dims, 42);

    for &m in [8, 12, 16, 24, 32, 48].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(m),
            &m,
            |bench, &m_val| {
                bench.iter(|| {
                    let config = HnswConfig {
                        m: m_val,
                        m0: m_val * 2,
                        ef_construction: 64,
                        max_elements: n,
                        metric: DistanceMetric::Euclidean,
                        seed: 42,
                        ..Default::default()
                    };

                    let mut index = HnswIndex::new(config);
                    for (id, vec) in vectors.iter().enumerate() {
                        index.insert(id as u64, vec);
                    }
                    black_box(index)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// HNSW Search Benchmarks
// ============================================================================

fn bench_hnsw_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search");

    for &dims in [128, 384, 768, 1536].iter() {
        for &n in [10000, 100000, 1000000].iter() {
            let vectors = generate_random_vectors(n, dims, 42);
            let query = generate_random_vectors(1, dims, 999)[0].clone();

            let config = HnswConfig {
                m: 16,
                m0: 32,
                ef_construction: 64,
                ef_search: 40,
                max_elements: n,
                metric: DistanceMetric::Euclidean,
                seed: 42,
                ..Default::default()
            };

            let mut index = HnswIndex::new(config);
            for (id, vec) in vectors.iter().enumerate() {
                index.insert(id as u64, vec);
            }

            group.bench_with_input(
                BenchmarkId::new(format!("{}d", dims), n),
                &(&index, &query),
                |bench, (idx, q)| {
                    bench.iter(|| {
                        black_box(idx.search(q, 10))
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_hnsw_search_ef_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search_ef");

    let dims = 768;
    let n = 100000;
    let vectors = generate_random_vectors(n, dims, 42);
    let queries = generate_random_vectors(100, dims, 999);

    // Build index once
    let config = HnswConfig {
        m: 16,
        m0: 32,
        ef_construction: 64,
        ef_search: 40, // Will be overridden
        max_elements: n,
        metric: DistanceMetric::Euclidean,
        seed: 42,
        ..Default::default()
    };

    let mut index = HnswIndex::new(config);
    for (id, vec) in vectors.iter().enumerate() {
        index.insert(id as u64, vec);
    }

    for &ef in [10, 20, 40, 80, 160, 320].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(ef),
            &ef,
            |bench, &ef_val| {
                bench.iter(|| {
                    for query in &queries {
                        black_box(index.search_with_ef(query, 10, ef_val));
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_hnsw_search_k_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search_k");

    let dims = 768;
    let n = 100000;
    let vectors = generate_random_vectors(n, dims, 42);
    let query = generate_random_vectors(1, dims, 999)[0].clone();

    let config = HnswConfig {
        m: 16,
        m0: 32,
        ef_construction: 64,
        ef_search: 100,
        max_elements: n,
        metric: DistanceMetric::Euclidean,
        seed: 42,
        ..Default::default()
    };

    let mut index = HnswIndex::new(config);
    for (id, vec) in vectors.iter().enumerate() {
        index.insert(id as u64, vec);
    }

    for &k in [1, 5, 10, 20, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(k),
            &k,
            |bench, &k_val| {
                bench.iter(|| {
                    black_box(index.search(&query, k_val))
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Recall Accuracy Benchmarks
// ============================================================================

fn bench_hnsw_recall(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_recall");
    group.sample_size(10);

    let dims = 768;
    let n = 10000;
    let vectors = generate_clustered_vectors(n, dims, 20, 42);
    let queries = generate_random_vectors(100, dims, 999);

    // Build index
    let config = HnswConfig {
        m: 16,
        m0: 32,
        ef_construction: 64,
        ef_search: 40,
        max_elements: n,
        metric: DistanceMetric::Euclidean,
        seed: 42,
        ..Default::default()
    };

    let mut index = HnswIndex::new(config);
    for (id, vec) in vectors.iter().enumerate() {
        index.insert(id as u64, vec);
    }

    // Compute ground truth (brute force)
    let compute_ground_truth = |query: &[f32], k: usize| -> Vec<u64> {
        let mut distances: Vec<(u64, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(id, vec)| {
                let dist = vec
                    .iter()
                    .zip(query)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                (id as u64, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.iter().take(k).map(|(id, _)| *id).collect()
    };

    for &ef in [10, 20, 40, 80, 160].iter() {
        group.bench_with_input(
            BenchmarkId::new("recall@10", ef),
            &ef,
            |bench, &ef_val| {
                bench.iter(|| {
                    let mut total_recall = 0.0;
                    for query in &queries {
                        let ground_truth = compute_ground_truth(query, 10);
                        let results = index.search_with_ef(query, 10, ef_val);

                        let hits = results
                            .iter()
                            .filter(|r| ground_truth.contains(&r.id))
                            .count();

                        total_recall += hits as f32 / 10.0;
                    }
                    black_box(total_recall / queries.len() as f32)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Memory Usage Benchmarks
// ============================================================================

fn bench_hnsw_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_memory");
    group.sample_size(10);

    for &dims in [128, 384, 768, 1536].iter() {
        for &n in [1000, 10000, 100000].iter() {
            let vectors = generate_random_vectors(n, dims, 42);

            group.bench_with_input(
                BenchmarkId::new(format!("{}d", dims), n),
                &vectors,
                |bench, vecs| {
                    bench.iter(|| {
                        let config = HnswConfig {
                            m: 16,
                            m0: 32,
                            ef_construction: 64,
                            max_elements: n,
                            metric: DistanceMetric::Euclidean,
                            seed: 42,
                            ..Default::default()
                        };

                        let mut index = HnswIndex::new(config);
                        for (id, vec) in vecs.iter().enumerate() {
                            index.insert(id as u64, vec);
                        }

                        let memory_bytes = index.memory_usage();
                        let memory_per_vec = memory_bytes as f64 / n as f64;
                        black_box(memory_per_vec)
                    });
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// Distance Metric Comparison
// ============================================================================

fn bench_hnsw_distance_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_metrics");
    group.sample_size(10);

    let dims = 768;
    let n = 10000;
    let vectors = generate_random_vectors(n, dims, 42);
    let query = generate_random_vectors(1, dims, 999)[0].clone();

    for metric in [
        DistanceMetric::Euclidean,
        DistanceMetric::Cosine,
        DistanceMetric::InnerProduct,
    ] {
        let config = HnswConfig {
            m: 16,
            m0: 32,
            ef_construction: 64,
            ef_search: 40,
            max_elements: n,
            metric,
            seed: 42,
            ..Default::default()
        };

        let mut index = HnswIndex::new(config);
        for (id, vec) in vectors.iter().enumerate() {
            index.insert(id as u64, vec);
        }

        let metric_name = match metric {
            DistanceMetric::Euclidean => "l2",
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::InnerProduct => "inner_product",
        };

        group.bench_with_input(
            BenchmarkId::new("search", metric_name),
            &(&index, &query),
            |bench, (idx, q)| {
                bench.iter(|| {
                    black_box(idx.search(q, 10))
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Parallel Search Benchmark
// ============================================================================

fn bench_hnsw_parallel_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_parallel");

    let dims = 768;
    let n = 100000;
    let vectors = generate_random_vectors(n, dims, 42);
    let queries = generate_random_vectors(1000, dims, 999);

    let config = HnswConfig {
        m: 16,
        m0: 32,
        ef_construction: 64,
        ef_search: 40,
        max_elements: n,
        metric: DistanceMetric::Euclidean,
        seed: 42,
        ..Default::default()
    };

    let mut index = HnswIndex::new(config);
    for (id, vec) in vectors.iter().enumerate() {
        index.insert(id as u64, vec);
    }

    group.bench_function("sequential", |bench| {
        bench.iter(|| {
            for query in &queries {
                black_box(index.search(query, 10));
            }
        });
    });

    group.bench_function("parallel_rayon", |bench| {
        use rayon::prelude::*;
        bench.iter(|| {
            queries.par_iter().for_each(|query| {
                black_box(index.search(query, 10));
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_hnsw_build,
    bench_hnsw_build_ef_construction,
    bench_hnsw_build_m_parameter,
    bench_hnsw_search,
    bench_hnsw_search_ef_values,
    bench_hnsw_search_k_values,
    bench_hnsw_recall,
    bench_hnsw_memory,
    bench_hnsw_distance_metrics,
    bench_hnsw_parallel_search,
);

criterion_main!(benches);
