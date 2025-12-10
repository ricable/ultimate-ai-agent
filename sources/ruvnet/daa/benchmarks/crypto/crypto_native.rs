//! Native Rust crypto benchmarks using Criterion
//! Measures performance of quantum-resistant operations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use qudag_crypto::{MlKem768, MlDsa, QuantumFingerprint};
use blake3;
use rand::rngs::OsRng;
use rand::RngCore;

/// Benchmark ML-KEM-768 key generation
fn bench_ml_kem_keygen(c: &mut Criterion) {
    let mut group = c.benchmark_group("ML-KEM-768 Key Generation");

    group.bench_function("keygen", |b| {
        b.iter(|| {
            let (public_key, secret_key) = MlKem768::generate_keypair(&mut OsRng);
            black_box((public_key, secret_key))
        });
    });

    group.finish();
}

/// Benchmark ML-KEM-768 encapsulation
fn bench_ml_kem_encapsulate(c: &mut Criterion) {
    let mut group = c.benchmark_group("ML-KEM-768 Encapsulation");

    let (public_key, _) = MlKem768::generate_keypair(&mut OsRng);

    group.bench_function("encapsulate", |b| {
        b.iter(|| {
            let (ciphertext, shared_secret) = MlKem768::encapsulate(&public_key, &mut OsRng);
            black_box((ciphertext, shared_secret))
        });
    });

    group.finish();
}

/// Benchmark ML-KEM-768 decapsulation
fn bench_ml_kem_decapsulate(c: &mut Criterion) {
    let mut group = c.benchmark_group("ML-KEM-768 Decapsulation");

    let (public_key, secret_key) = MlKem768::generate_keypair(&mut OsRng);
    let (ciphertext, _) = MlKem768::encapsulate(&public_key, &mut OsRng);

    group.bench_function("decapsulate", |b| {
        b.iter(|| {
            let shared_secret = MlKem768::decapsulate(&ciphertext, &secret_key);
            black_box(shared_secret)
        });
    });

    group.finish();
}

/// Benchmark ML-DSA signing
fn bench_ml_dsa_sign(c: &mut Criterion) {
    let mut group = c.benchmark_group("ML-DSA Signing");

    let (public_key, secret_key) = MlDsa::generate_keypair(&mut OsRng);
    let message = b"Hello, quantum-resistant world!";

    group.bench_function("sign", |b| {
        b.iter(|| {
            let signature = MlDsa::sign(black_box(message), &secret_key);
            black_box(signature)
        });
    });

    group.finish();
}

/// Benchmark ML-DSA verification
fn bench_ml_dsa_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("ML-DSA Verification");

    let (public_key, secret_key) = MlDsa::generate_keypair(&mut OsRng);
    let message = b"Hello, quantum-resistant world!";
    let signature = MlDsa::sign(message, &secret_key);

    group.bench_function("verify", |b| {
        b.iter(|| {
            let valid = MlDsa::verify(black_box(message), &signature, &public_key);
            black_box(valid)
        });
    });

    group.finish();
}

/// Benchmark BLAKE3 hashing at various data sizes
fn bench_blake3_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("BLAKE3 Hashing");

    for size_kb in [1, 10, 100, 1024, 10240].iter() {
        let size = size_kb * 1024;
        let mut data = vec![0u8; size];
        OsRng.fill_bytes(&mut data);

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(format!("{}KB", size_kb)), &data, |b, data| {
            b.iter(|| {
                let hash = blake3::hash(black_box(data));
                black_box(hash)
            });
        });
    }

    group.finish();
}

/// Benchmark quantum fingerprint generation
fn bench_quantum_fingerprint(c: &mut Criterion) {
    let mut group = c.benchmark_group("Quantum Fingerprint");

    for size_kb in [1, 10, 100, 1024].iter() {
        let size = size_kb * 1024;
        let mut data = vec![0u8; size];
        OsRng.fill_bytes(&mut data);

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(format!("{}KB", size_kb)), &data, |b, data| {
            b.iter(|| {
                let fingerprint = QuantumFingerprint::generate(black_box(data));
                black_box(fingerprint)
            });
        });
    }

    group.finish();
}

/// Benchmark full key exchange workflow
fn bench_key_exchange_workflow(c: &mut Criterion) {
    let mut group = c.benchmark_group("Full Key Exchange Workflow");

    group.bench_function("complete_handshake", |b| {
        b.iter(|| {
            // Alice generates keypair
            let (alice_pk, alice_sk) = MlKem768::generate_keypair(&mut OsRng);

            // Bob encapsulates
            let (ciphertext, bob_shared_secret) = MlKem768::encapsulate(&alice_pk, &mut OsRng);

            // Alice decapsulates
            let alice_shared_secret = MlKem768::decapsulate(&ciphertext, &alice_sk);

            black_box((bob_shared_secret, alice_shared_secret))
        });
    });

    group.finish();
}

/// Benchmark signature workflow
fn bench_signature_workflow(c: &mut Criterion) {
    let mut group = c.benchmark_group("Full Signature Workflow");

    let message = b"Transaction data to be signed";

    group.bench_function("sign_and_verify", |b| {
        b.iter(|| {
            // Generate keypair
            let (public_key, secret_key) = MlDsa::generate_keypair(&mut OsRng);

            // Sign
            let signature = MlDsa::sign(black_box(message), &secret_key);

            // Verify
            let valid = MlDsa::verify(black_box(message), &signature, &public_key);

            black_box(valid)
        });
    });

    group.finish();
}

criterion_group!(
    crypto_benches,
    bench_ml_kem_keygen,
    bench_ml_kem_encapsulate,
    bench_ml_kem_decapsulate,
    bench_ml_dsa_sign,
    bench_ml_dsa_verify,
    bench_blake3_hash,
    bench_quantum_fingerprint,
    bench_key_exchange_workflow,
    bench_signature_workflow
);

criterion_main!(crypto_benches);
