// Criterion benchmark for alignment engine
// Measures throughput (reads/second) and latency for alignment operations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
// use crispr_cas13_pipeline::alignment_engine::*; // TODO: Uncomment when module exists

/// Generate synthetic sequence data for benchmarking
fn generate_sequence(length: usize) -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let bases = ['A', 'C', 'G', 'T'];
    (0..length).map(|_| bases[rng.gen_range(0..4)]).collect()
}

/// Generate multiple sequences for batch benchmarking
fn generate_sequences(count: usize, length: usize) -> Vec<String> {
    (0..count).map(|_| generate_sequence(length)).collect()
}

/// Benchmark: Single sequence alignment
fn bench_single_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_alignment");

    // Test different sequence lengths
    for length in [50, 100, 150, 200, 500, 1000].iter() {
        let reference = generate_sequence(*length);
        let query = generate_sequence(*length);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}bp", length)),
            length,
            |b, &_len| {
                b.iter(|| {
                    // TODO: Replace with actual alignment function
                    // align(&reference, &query)
                    black_box(&reference);
                    black_box(&query);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Batch alignment throughput
fn bench_batch_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_alignment");

    // Test different batch sizes
    for batch_size in [100, 1_000, 10_000, 100_000].iter() {
        let reference = generate_sequence(150); // Typical read length
        let queries = generate_sequences(*batch_size, 150);

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_reads", batch_size)),
            batch_size,
            |b, &_size| {
                b.iter(|| {
                    // TODO: Replace with actual batch alignment function
                    // align_batch(&reference, &queries)
                    for query in &queries {
                        black_box(&reference);
                        black_box(query);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Parallel alignment throughput
fn bench_parallel_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_alignment");

    // Test scaling with thread count
    let reference = generate_sequence(150);
    let queries = generate_sequences(10_000, 150);

    for thread_count in [1, 2, 4, 8].iter() {
        group.throughput(Throughput::Elements(10_000));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_threads", thread_count)),
            thread_count,
            |b, &threads| {
                b.iter(|| {
                    // TODO: Replace with actual parallel alignment function
                    // align_parallel(&reference, &queries, threads)
                    black_box(threads);
                    black_box(&reference);
                    black_box(&queries);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Alignment with quality scores
fn bench_quality_aware_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality_aware_alignment");

    let reference = generate_sequence(150);
    let query = generate_sequence(150);
    let quality_scores: Vec<u8> = (0..150).map(|_| 30).collect(); // Phred 30

    group.throughput(Throughput::Elements(1));
    group.bench_function("with_quality_scores", |b| {
        b.iter(|| {
            // TODO: Replace with actual quality-aware alignment
            // align_with_quality(&reference, &query, &quality_scores)
            black_box(&reference);
            black_box(&query);
            black_box(&quality_scores);
        });
    });

    group.finish();
}

/// Benchmark: Alignment with complex indels
fn bench_indel_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("indel_alignment");

    // Generate sequences with insertions/deletions
    let reference = generate_sequence(150);
    let query_with_indels = {
        let mut seq = generate_sequence(140);
        seq.push_str(&generate_sequence(20)); // Simulate complex indel pattern
        seq
    };

    group.throughput(Throughput::Elements(1));
    group.bench_function("complex_indels", |b| {
        b.iter(|| {
            // TODO: Replace with actual alignment function
            // align(&reference, &query_with_indels)
            black_box(&reference);
            black_box(&query_with_indels);
        });
    });

    group.finish();
}

/// Benchmark: CIGAR string generation
fn bench_cigar_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cigar_generation");

    let alignment_length = 150;

    group.throughput(Throughput::Elements(1));
    group.bench_function("cigar_string", |b| {
        b.iter(|| {
            // TODO: Replace with actual CIGAR generation
            // generate_cigar_string(alignment_data)
            black_box(alignment_length);
        });
    });

    group.finish();
}

/// Benchmark: Memory usage for large batch alignment
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    // Test with 1 million reads (memory pressure test)
    let reference = generate_sequence(150);
    let large_batch = generate_sequences(1_000_000, 150);

    group.throughput(Throughput::Elements(1_000_000));
    group.sample_size(10); // Reduce sample size for expensive benchmark
    group.bench_function("1M_reads", |b| {
        b.iter(|| {
            // TODO: Replace with actual streaming alignment
            // align_streaming(&reference, &large_batch)
            black_box(&reference);
            black_box(&large_batch);
        });
    });

    group.finish();
}

// Configure Criterion
criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(std::time::Duration::from_secs(10));
    targets =
        bench_single_alignment,
        bench_batch_alignment,
        bench_parallel_alignment,
        bench_quality_aware_alignment,
        bench_indel_alignment,
        bench_cigar_generation,
        bench_memory_efficiency
}

criterion_main!(benches);
