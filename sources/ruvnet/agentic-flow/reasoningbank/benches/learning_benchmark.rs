use criterion::{black_box, criterion_group, criterion_main, Criterion};
use reasoningbank_core::{Pattern, TaskOutcome, ReasoningEngine, EngineConfig};
use reasoningbank_learning::{AdaptiveLearner, LearningConfig};
use reasoningbank_storage::{SqliteStorage, StorageConfig};
use std::path::PathBuf;

fn bench_learning_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("learning");

    // Setup
    let temp_dir = std::env::temp_dir();
    let db_path = temp_dir.join("bench_learning.db");
    let _ = std::fs::remove_file(&db_path);

    let storage_config = StorageConfig {
        database_path: db_path.clone(),
        max_connections: 10,
        enable_wal: true,
        cache_size_kb: 8192,
    };

    let engine = ReasoningEngine::new(EngineConfig::default());
    let storage = SqliteStorage::new(storage_config).unwrap();
    let mut learner = AdaptiveLearner::new(engine, storage, LearningConfig::default());

    // Pre-populate with some patterns
    for i in 0..50 {
        let pattern = Pattern::new(
            format!("Learning task {}", i),
            "learning".to_string(),
            format!("strategy-{}", i % 5),
        ).with_outcome(TaskOutcome::success(0.8 + (i % 10) as f64 * 0.02));
        learner.learn_from_task(&pattern).unwrap();
    }

    // Benchmark learning from task
    group.bench_function("learn_from_task", |b| {
        let mut counter = 50;
        b.iter(|| {
            let pattern = Pattern::new(
                format!("New task {}", counter),
                "learning".to_string(),
                "strategy-1".to_string(),
            ).with_outcome(TaskOutcome::success(0.9));
            counter += 1;
            learner.learn_from_task(black_box(&pattern)).unwrap();
        });
    });

    // Benchmark applying learning
    group.bench_function("apply_learning", |b| {
        b.iter(|| {
            learner.apply_learning(
                black_box("Implement feature X"),
                black_box("learning")
            ).unwrap();
        });
    });

    // Benchmark statistics retrieval
    group.bench_function("get_stats", |b| {
        b.iter(|| {
            learner.get_learning_stats().unwrap();
        });
    });

    group.finish();
    let _ = std::fs::remove_file(&db_path);
}

criterion_group!(benches, bench_learning_operations);
criterion_main!(benches);
