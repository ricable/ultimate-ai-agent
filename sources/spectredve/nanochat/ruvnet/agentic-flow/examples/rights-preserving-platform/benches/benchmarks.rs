use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_stub(c: &mut Criterion) {
    c.bench_function("stub", |b| {
        b.iter(|| black_box(1 + 1));
    });
}

criterion_group!(benches, benchmark_stub);
criterion_main!(benches);
