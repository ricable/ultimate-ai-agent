//! Performance benchmarks for operations

use agentic_jujutsu::{JJOperation, JJOperationLog, OperationType};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn benchmark_operation_creation(c: &mut Criterion) {
    c.bench_function("create_operation", |b| {
        b.iter(|| {
            JJOperation::new(
                black_box("op_12345678".to_string()),
                black_box("user@example.com".to_string()),
                black_box("Test operation description".to_string()),
            )
        });
    });
}

fn benchmark_operation_serialization(c: &mut Criterion) {
    let op = JJOperation::new(
        "op_12345678".to_string(),
        "user@example.com".to_string(),
        "Test operation".to_string(),
    );

    c.bench_function("serialize_operation", |b| {
        b.iter(|| serde_json::to_string(black_box(&op)).unwrap());
    });
}

fn benchmark_operation_deserialization(c: &mut Criterion) {
    let json = r#"{
        "id": "op_12345678",
        "op_type": "Commit",
        "user": "user@example.com",
        "timestamp": "2024-01-01T00:00:00Z",
        "description": "Test operation",
        "parents": []
    }"#;

    c.bench_function("deserialize_operation", |b| {
        b.iter(|| serde_json::from_str::<JJOperation>(black_box(json)).unwrap());
    });
}

fn benchmark_operation_log_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("operation_log_add");

    for size in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut log = JJOperationLog::new(size);
                for i in 0..100 {
                    let op = JJOperation::new(
                        format!("op_{}", i),
                        "user@test.com".to_string(),
                        "desc".to_string(),
                    );
                    log.add_operation(black_box(op));
                }
            });
        });
    }

    group.finish();
}

fn benchmark_operation_log_search(c: &mut Criterion) {
    let mut log = JJOperationLog::new(10000);

    // Populate log
    for i in 0..1000 {
        let op = JJOperation::new(
            format!("op_{:08}", i),
            "user@test.com".to_string(),
            "desc".to_string(),
        );
        log.add_operation(op);
    }

    c.bench_function("operation_log_get_by_id", |b| {
        b.iter(|| log.get_by_id(black_box("op_00000500")));
    });
}

fn benchmark_operation_log_filter(c: &mut Criterion) {
    let mut log = JJOperationLog::new(10000);

    // Populate log with mixed operation types
    for i in 0..1000 {
        let mut op = JJOperation::new(
            format!("op_{}", i),
            "user@test.com".to_string(),
            "desc".to_string(),
        );

        op.op_type = if i % 3 == 0 {
            OperationType::Commit
        } else if i % 3 == 1 {
            OperationType::Rebase
        } else {
            OperationType::Merge
        };

        log.add_operation(op);
    }

    c.bench_function("operation_log_filter_by_type", |b| {
        b.iter(|| log.filter_by_type(black_box(OperationType::Commit)));
    });
}

fn benchmark_operation_log_recent(c: &mut Criterion) {
    let mut log = JJOperationLog::new(10000);

    for i in 0..1000 {
        let op = JJOperation::new(
            format!("op_{}", i),
            "user@test.com".to_string(),
            "desc".to_string(),
        );
        log.add_operation(op);
    }

    let mut group = c.benchmark_group("operation_log_recent");

    for count in [1, 10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, &count| {
            b.iter(|| log.recent(black_box(count)));
        });
    }

    group.finish();
}

fn benchmark_operation_parse_from_log(c: &mut Criterion) {
    let log_lines = vec![
        "@  qpvuntsm test@example.com 2024-01-01 12:00:00.000 -08:00 abc123 describe commit",
        "◉  sqpuoqvx test@example.com 2024-01-01 11:00:00.000 -08:00 def456 rebase operation",
        "@  xyzabc12 user@test.org 2024-01-02 10:30:00.000 -08:00 ghijkl merge branches",
    ];

    c.bench_function("parse_operation_from_log", |b| {
        b.iter(|| {
            for line in &log_lines {
                JJOperation::parse_from_log(black_box(line));
            }
        });
    });
}

fn benchmark_operation_timestamp_iso(c: &mut Criterion) {
    let op = JJOperation::new(
        "op_id".to_string(),
        "user@test.com".to_string(),
        "desc".to_string(),
    );

    c.bench_function("operation_timestamp_iso", |b| {
        b.iter(|| black_box(&op).timestamp_iso());
    });
}

fn benchmark_memory_usage(c: &mut Criterion) {
    c.bench_function("operation_log_memory_1k", |b| {
        b.iter(|| {
            let mut log = JJOperationLog::new(1000);
            for i in 0..1000 {
                let op = JJOperation::new(
                    format!("op_{}", i),
                    "user@test.com".to_string(),
                    format!("Description for operation {}", i),
                );
                log.add_operation(op);
            }
            black_box(log);
        });
    });
}

fn benchmark_operation_log_clear(c: &mut Criterion) {
    c.bench_function("operation_log_clear", |b| {
        b.iter_with_setup(
            || {
                let mut log = JJOperationLog::new(1000);
                for i in 0..1000 {
                    log.add_operation(JJOperation::new(
                        format!("op_{}", i),
                        "user@test.com".to_string(),
                        "desc".to_string(),
                    ));
                }
                log
            },
            |mut log| {
                log.clear();
                black_box(log);
            },
        );
    });
}

criterion_group!(
    benches,
    benchmark_operation_creation,
    benchmark_operation_serialization,
    benchmark_operation_deserialization,
    benchmark_operation_log_add,
    benchmark_operation_log_search,
    benchmark_operation_log_filter,
    benchmark_operation_log_recent,
    benchmark_operation_parse_from_log,
    benchmark_operation_timestamp_iso,
    benchmark_memory_usage,
    benchmark_operation_log_clear,
);

criterion_main!(benches);
