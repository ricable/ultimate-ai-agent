use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use reasoningbank_network::neural_bus::{Frame, FrameType, FrameHeader};
use bytes::Bytes;

fn bench_frame_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_bus");

    // Create test data
    let header = FrameHeader::new("benchmark_op".to_string());
    let payload = Bytes::from(vec![0u8; 1024]); // 1KB payload

    // Benchmark frame encoding
    group.bench_function("frame_encode", |b| {
        b.iter(|| {
            let frame = Frame::new(
                black_box(FrameType::Request),
                black_box(header.clone()),
                black_box(payload.clone())
            );
            let _ = frame.encode();
        });
    });

    // Benchmark frame decoding
    let frame = Frame::new(FrameType::Request, header.clone(), payload.clone());
    let encoded = frame.encode().unwrap();

    group.bench_function("frame_decode", |b| {
        b.iter(|| {
            Frame::decode(black_box(encoded.clone())).unwrap();
        });
    });

    // Benchmark different payload sizes
    for size in [256, 1024, 4096, 16384].iter() {
        let payload = Bytes::from(vec![0u8; *size]);
        group.bench_with_input(BenchmarkId::new("frame_encode_size", size), size, |b, _| {
            b.iter(|| {
                let frame = Frame::new(
                    black_box(FrameType::Request),
                    black_box(header.clone()),
                    black_box(payload.clone())
                );
                let _ = frame.encode();
            });
        });
    }

    // Benchmark frame type checking
    group.bench_function("frame_type_is_mutating", |b| {
        b.iter(|| {
            black_box(FrameType::Request.is_mutating());
        });
    });

    group.bench_function("frame_type_from_u8", |b| {
        b.iter(|| {
            black_box(FrameType::from_u8(1));
        });
    });

    group.finish();
}

criterion_group!(benches, bench_frame_operations);
criterion_main!(benches);
