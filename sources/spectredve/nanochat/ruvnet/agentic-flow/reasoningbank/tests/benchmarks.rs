//! Performance benchmarks for ReasoningBank

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use reasoningbank_core::{Pattern, TaskOutcome, ReasoningEngine};
use reasoningbank_storage::SqliteStorage;

fn benchmark_pattern_storage(c: &mut Criterion) {
    let storage = SqliteStorage::in_memory().unwrap();
    let engine = ReasoningEngine::default();

    c.bench_function("store_pattern", |b| {
        b.iter(|| {
            let pattern = Pattern::new(
                black_box("Test task".to_string()),
                black_box("testing".to_string()),
                black_box("test-strategy".to_string()),
            ).with_outcome(TaskOutcome::success(1.0));

            let prepared = engine.prepare_pattern(pattern).unwrap();
            storage.store_pattern(&prepared).unwrap();
        });
    });
}

fn benchmark_pattern_retrieval(c: &mut Criterion) {
    let storage = SqliteStorage::in_memory().unwrap();
    let engine = ReasoningEngine::default();

    // Prepare test data
    let mut ids = Vec::new();
    for i in 0..100 {
        let pattern = Pattern::new(
            format!("Task {}", i),
            "testing".to_string(),
            "test-strategy".to_string(),
        );
        let prepared = engine.prepare_pattern(pattern).unwrap();
        ids.push(prepared.id);
        storage.store_pattern(&prepared).unwrap();
    }

    c.bench_function("retrieve_pattern", |b| {
        b.iter(|| {
            let id = black_box(&ids[50]);
            storage.get_pattern(id).unwrap();
        });
    });
}

fn benchmark_similarity_search(c: &mut Criterion) {
    let storage = SqliteStorage::in_memory().unwrap();
    let engine = ReasoningEngine::default();

    // Prepare test data
    for i in 0..100 {
        let pattern = Pattern::new(
            format!("Task description number {}", i),
            "testing".to_string(),
            format!("strategy-{}", i % 5),
        ).with_outcome(TaskOutcome::partial(0.8, 1.0));

        let prepared = engine.prepare_pattern(pattern).unwrap();
        storage.store_pattern(&prepared).unwrap();
    }

    let candidates = storage.get_all_patterns(None).unwrap();

    c.bench_function("similarity_search", |b| {
        b.iter(|| {
            let query = Pattern::new(
                black_box("Find similar task".to_string()),
                black_box("testing".to_string()),
                String::new(),
            );
            let prepared = engine.prepare_pattern(query).unwrap();
            engine.find_similar(&prepared, black_box(&candidates));
        });
    });
}

fn benchmark_embedding_generation(c: &mut Criterion) {
    let engine = ReasoningEngine::default();

    c.bench_function("generate_embedding", |b| {
        b.iter(|| {
            let pattern = Pattern::new(
                black_box("This is a test task description for benchmarking".to_string()),
                black_box("testing".to_string()),
                black_box("test-strategy".to_string()),
            );
            engine.prepare_pattern(pattern).unwrap();
        });
    });
}

fn benchmark_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");

    for size in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let storage = SqliteStorage::in_memory().unwrap();
            let engine = ReasoningEngine::default();

            b.iter(|| {
                for i in 0..size {
                    let pattern = Pattern::new(
                        format!("Task {}", i),
                        "testing".to_string(),
                        "test-strategy".to_string(),
                    );
                    let prepared = engine.prepare_pattern(pattern).unwrap();
                    storage.store_pattern(&prepared).unwrap();
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_pattern_storage,
    benchmark_pattern_retrieval,
    benchmark_similarity_search,
    benchmark_embedding_generation,
    benchmark_batch_operations,
);

criterion_main!(benches);
