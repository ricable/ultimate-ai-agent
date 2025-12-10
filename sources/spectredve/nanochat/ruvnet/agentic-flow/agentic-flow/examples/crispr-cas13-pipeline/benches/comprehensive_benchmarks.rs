//! Comprehensive performance benchmarks for critical paths

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

/// Benchmark sequence alignment performance
fn bench_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("alignment");

    for size in [100, 500, 1000, 5000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let sequences = generate_sequences(size);
            b.iter(|| {
                for seq in &sequences {
                    align_sequence(black_box(seq));
                }
            });
        });
    }

    group.finish();
}

/// Benchmark off-target prediction scoring
fn bench_offtarget_scoring(c: &mut Criterion) {
    let mut group = c.benchmark_group("offtarget_scoring");

    let guide = "ACGUACGUACGUACGUACGUACG";
    let targets = generate_target_sites(1000);

    group.throughput(Throughput::Elements(1000));
    group.bench_function("score_1000_targets", |b| {
        b.iter(|| {
            for target in &targets {
                calculate_offtarget_score(black_box(guide), black_box(target));
            }
        });
    });

    group.finish();
}

/// Benchmark differential expression analysis
fn bench_differential_expression(c: &mut Criterion) {
    let mut group = c.benchmark_group("differential_expression");

    for n_genes in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*n_genes as u64));
        group.bench_with_input(
            BenchmarkId::new("deseq_analysis", n_genes),
            n_genes,
            |b, &n| {
                let control = generate_expression_data(n);
                let treated = generate_expression_data(n);
                b.iter(|| {
                    run_deseq_analysis(black_box(&control), black_box(&treated));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark TPM normalization
fn bench_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalization");

    for size in [100, 1000, 10000, 50000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let counts = generate_count_data(size);
            b.iter(|| {
                normalize_tpm(black_box(&counts));
            });
        });
    }

    group.finish();
}

/// Benchmark API request handling
fn bench_api_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("api_requests");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("job_creation", |b| {
        b.iter(|| {
            create_job_request(black_box("test_job"));
        });
    });

    group.bench_function("job_status_query", |b| {
        b.iter(|| {
            query_job_status(black_box("job_123"));
        });
    });

    group.finish();
}

/// Benchmark database operations
fn bench_database_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("database");

    group.bench_function("insert_job", |b| {
        b.iter(|| {
            insert_job_record(black_box("job_123"), black_box("pending"));
        });
    });

    group.bench_function("query_jobs", |b| {
        b.iter(|| {
            query_jobs_by_status(black_box("running"));
        });
    });

    group.bench_function("update_job_status", |b| {
        b.iter(|| {
            update_job_status(black_box("job_123"), black_box("completed"));
        });
    });

    group.finish();
}

/// Benchmark message queue operations
fn bench_message_queue(c: &mut Criterion) {
    let mut group = c.benchmark_group("message_queue");

    for batch_size in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("kafka_publish", batch_size),
            batch_size,
            |b, &size| {
                let messages = generate_messages(size);
                b.iter(|| {
                    publish_messages(black_box(&messages));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark feature extraction
fn bench_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_extraction");

    let guide = "ACGUACGUACGUACGUACGUACG";

    group.bench_function("gc_content", |b| {
        b.iter(|| {
            calculate_gc_content(black_box(guide));
        });
    });

    group.bench_function("melting_temp", |b| {
        b.iter(|| {
            calculate_melting_temp(black_box(guide));
        });
    });

    group.bench_function("secondary_structure", |b| {
        b.iter(|| {
            predict_secondary_structure(black_box(guide));
        });
    });

    group.finish();
}

/// Benchmark statistical calculations
fn bench_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistics");

    let group1 = generate_expression_data(1000);
    let group2 = generate_expression_data(1000);

    group.bench_function("t_test", |b| {
        b.iter(|| {
            welch_ttest(black_box(&group1), black_box(&group2));
        });
    });

    let p_values = generate_p_values(10000);

    group.bench_function("fdr_correction", |b| {
        b.iter(|| {
            benjamini_hochberg_fdr(black_box(&p_values), 0.05);
        });
    });

    group.finish();
}

/// Benchmark parallel processing
fn bench_parallel_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel");

    for n_tasks in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*n_tasks as u64));
        group.bench_with_input(
            BenchmarkId::new("parallel_alignment", n_tasks),
            n_tasks,
            |b, &n| {
                let sequences = generate_sequences(n);
                b.iter(|| {
                    parallel_align(black_box(&sequences));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory operations
fn bench_memory_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory");

    group.bench_function("large_allocation", |b| {
        b.iter(|| {
            let _data: Vec<u8> = vec![0; 1_000_000];
        });
    });

    group.bench_function("serialization", |b| {
        let data = generate_expression_data(1000);
        b.iter(|| {
            serialize_data(black_box(&data));
        });
    });

    group.bench_function("deserialization", |b| {
        let json = r#"{"genes": [1, 2, 3, 4, 5]}"#;
        b.iter(|| {
            deserialize_data(black_box(json));
        });
    });

    group.finish();
}

// Helper functions
fn generate_sequences(n: usize) -> Vec<String> {
    (0..n).map(|_| "ACGTACGTACGT".to_string()).collect()
}

fn generate_target_sites(n: usize) -> Vec<String> {
    (0..n).map(|_| "ACGUACGUACGUACGUACGUACG".to_string()).collect()
}

fn generate_expression_data(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 * 1.5).collect()
}

fn generate_count_data(n: usize) -> Vec<u32> {
    (0..n).map(|i| (i * 10) as u32).collect()
}

fn generate_messages(n: usize) -> Vec<String> {
    (0..n).map(|i| format!(r#"{{"job_id": "job_{}", "action": "process"}}"#, i)).collect()
}

fn generate_p_values(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64 / n as f64)).collect()
}

fn align_sequence(_seq: &str) -> i32 {
    42 // Mock alignment score
}

fn calculate_offtarget_score(_guide: &str, _target: &str) -> f64 {
    0.85 // Mock score
}

fn run_deseq_analysis(_control: &[f64], _treated: &[f64]) -> Vec<f64> {
    vec![1.5, 2.0, 0.5] // Mock fold changes
}

fn normalize_tpm(counts: &[u32]) -> Vec<f64> {
    let total: u32 = counts.iter().sum();
    counts.iter().map(|&c| (c as f64 / total as f64) * 1_000_000.0).collect()
}

fn create_job_request(_name: &str) -> String {
    "job_id_123".to_string()
}

fn query_job_status(_job_id: &str) -> String {
    "running".to_string()
}

fn insert_job_record(_job_id: &str, _status: &str) -> bool {
    true
}

fn query_jobs_by_status(_status: &str) -> Vec<String> {
    vec!["job1".to_string(), "job2".to_string()]
}

fn update_job_status(_job_id: &str, _status: &str) -> bool {
    true
}

fn publish_messages(_messages: &[String]) -> bool {
    true
}

fn calculate_gc_content(seq: &str) -> f64 {
    let gc = seq.chars().filter(|&c| c == 'G' || c == 'C').count();
    (gc as f64 / seq.len() as f64) * 100.0
}

fn calculate_melting_temp(seq: &str) -> f64 {
    seq.len() as f64 * 2.0 + 40.0
}

fn predict_secondary_structure(_seq: &str) -> f64 {
    -5.2 // Mock free energy
}

fn welch_ttest(_g1: &[f64], _g2: &[f64]) -> (f64, f64) {
    (2.5, 0.01) // Mock t-stat and p-value
}

fn benjamini_hochberg_fdr(p_values: &[f64], _alpha: f64) -> Vec<f64> {
    p_values.to_vec() // Mock FDR values
}

fn parallel_align(sequences: &[String]) -> Vec<i32> {
    sequences.iter().map(|s| align_sequence(s)).collect()
}

fn serialize_data(_data: &[f64]) -> String {
    "serialized".to_string()
}

fn deserialize_data(_json: &str) -> Vec<i32> {
    vec![1, 2, 3]
}

criterion_group!(
    benches,
    bench_alignment,
    bench_offtarget_scoring,
    bench_differential_expression,
    bench_normalization,
    bench_api_throughput,
    bench_database_ops,
    bench_message_queue,
    bench_feature_extraction,
    bench_statistics,
    bench_parallel_processing,
    bench_memory_ops
);

criterion_main!(benches);
