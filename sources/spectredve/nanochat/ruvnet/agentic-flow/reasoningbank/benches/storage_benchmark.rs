use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use reasoningbank_core::{Pattern, TaskOutcome};
use reasoningbank_storage::{SqliteStorage, StorageConfig};
use std::path::PathBuf;

fn bench_pattern_storage(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage");

    // Setup
    let temp_dir = std::env::temp_dir();
    let db_path = temp_dir.join("bench_storage.db");
    let _ = std::fs::remove_file(&db_path);

    let config = StorageConfig {
        database_path: db_path.clone(),
        max_connections: 10,
        enable_wal: true,
        cache_size_kb: 8192,
    };
    let storage = SqliteStorage::new(config).unwrap();

    // Benchmark pattern insertion
    group.bench_function("store_pattern", |b| {
        let mut counter = 0;
        b.iter(|| {
            let pattern = Pattern::new(
                format!("Task {}", counter),
                "benchmark".to_string(),
                "strategy-1".to_string(),
            ).with_outcome(TaskOutcome::success(0.95));
            counter += 1;
            storage.store_pattern(black_box(&pattern)).unwrap();
        });
    });

    // Store some patterns for retrieval benchmarks
    for i in 0..1000 {
        let pattern = Pattern::new(
            format!("Lookup task {}", i),
            "benchmark".to_string(),
            "strategy-1".to_string(),
        );
        storage.store_pattern(&pattern).unwrap();
    }

    // Benchmark pattern retrieval
    let patterns = storage.get_patterns_by_category("benchmark", 100).unwrap();
    let first_id = patterns[0].id;

    group.bench_function("get_pattern", |b| {
        b.iter(|| {
            storage.get_pattern(black_box(&first_id)).unwrap();
        });
    });

    // Benchmark category search
    group.bench_function("get_by_category", |b| {
        b.iter(|| {
            storage.get_patterns_by_category(black_box("benchmark"), black_box(10)).unwrap();
        });
    });

    group.finish();
    let _ = std::fs::remove_file(&db_path);
}

fn bench_similarity_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity");

    let temp_dir = std::env::temp_dir();
    let db_path = temp_dir.join("bench_similarity.db");
    let _ = std::fs::remove_file(&db_path);

    let config = StorageConfig {
        database_path: db_path.clone(),
        max_connections: 10,
        enable_wal: true,
        cache_size_kb: 8192,
    };
    let storage = SqliteStorage::new(config).unwrap();

    // Store patterns with embeddings
    for i in 0..100 {
        let mut pattern = Pattern::new(
            format!("Similarity task {}", i),
            "similarity".to_string(),
            "strategy-1".to_string(),
        );
        // Add dummy embedding
        pattern.embedding = Some(vec![0.1 * i as f32; 128]);
        storage.store_pattern(&pattern).unwrap();
    }

    // Benchmark retrieval of all patterns
    for size in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                storage.get_all_patterns(black_box(Some(size))).unwrap();
            });
        });
    }

    group.finish();
    let _ = std::fs::remove_file(&db_path);
}

criterion_group!(benches, bench_pattern_storage, bench_similarity_search);
criterion_main!(benches);
