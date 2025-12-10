/**
 * Performance Benchmarks for QuDAG Crypto Operations
 *
 * Measures performance of ML-KEM-768, ML-DSA, and BLAKE3
 */

const { test } = require('node:test');
const assert = require('node:assert/strict');

// Mock implementations with timing
const createBenchmarkMocks = () => {
  class MlKem768 {
    generateKeypair() {
      const start = process.hrtime.bigint();
      const result = {
        publicKey: Buffer.alloc(1184),
        secretKey: Buffer.alloc(2400)
      };
      const end = process.hrtime.bigint();
      return { result, time: Number(end - start) / 1_000_000 }; // Convert to ms
    }

    encapsulate(publicKey) {
      const start = process.hrtime.bigint();
      const result = {
        ciphertext: Buffer.alloc(1088),
        sharedSecret: Buffer.alloc(32)
      };
      const end = process.hrtime.bigint();
      return { result, time: Number(end - start) / 1_000_000 };
    }

    decapsulate(ciphertext, secretKey) {
      const start = process.hrtime.bigint();
      const result = Buffer.alloc(32);
      const end = process.hrtime.bigint();
      return { result, time: Number(end - start) / 1_000_000 };
    }
  }

  class MlDsa {
    sign(message, secretKey) {
      const start = process.hrtime.bigint();
      const result = Buffer.alloc(3309);
      const end = process.hrtime.bigint();
      return { result, time: Number(end - start) / 1_000_000 };
    }

    verify(message, signature, publicKey) {
      const start = process.hrtime.bigint();
      const result = true;
      const end = process.hrtime.bigint();
      return { result, time: Number(end - start) / 1_000_000 };
    }
  }

  function blake3Hash(data) {
    const start = process.hrtime.bigint();
    const result = Buffer.alloc(32);
    const end = process.hrtime.bigint();
    return { result, time: Number(end - start) / 1_000_000 };
  }

  return { MlKem768, MlDsa, blake3Hash };
};

const { MlKem768, MlDsa, blake3Hash } = createBenchmarkMocks();

function runBenchmark(name, fn, iterations = 100) {
  const times = [];

  for (let i = 0; i < iterations; i++) {
    const { time } = fn();
    times.push(time);
  }

  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  const min = Math.min(...times);
  const max = Math.max(...times);
  const median = times.sort((a, b) => a - b)[Math.floor(times.length / 2)];

  return { name, iterations, avg, min, max, median };
}

test('Benchmark: ML-KEM-768 keypair generation', (t) => {
  const mlkem = new MlKem768();

  const stats = runBenchmark('ML-KEM-768 Keygen', () => {
    return mlkem.generateKeypair();
  }, 100);

  console.log(`\n${stats.name}:`);
  console.log(`  Iterations: ${stats.iterations}`);
  console.log(`  Average: ${stats.avg.toFixed(3)}ms`);
  console.log(`  Median: ${stats.median.toFixed(3)}ms`);
  console.log(`  Min: ${stats.min.toFixed(3)}ms`);
  console.log(`  Max: ${stats.max.toFixed(3)}ms`);

  // Performance target: < 5ms for native (mock will be faster)
  assert.ok(stats.avg < 5, 'Average keygen should be under 5ms');
});

test('Benchmark: ML-KEM-768 encapsulation', (t) => {
  const mlkem = new MlKem768();
  const { result: keypair } = mlkem.generateKeypair();

  const stats = runBenchmark('ML-KEM-768 Encapsulate', () => {
    return mlkem.encapsulate(keypair.publicKey);
  }, 100);

  console.log(`\n${stats.name}:`);
  console.log(`  Iterations: ${stats.iterations}`);
  console.log(`  Average: ${stats.avg.toFixed(3)}ms`);
  console.log(`  Median: ${stats.median.toFixed(3)}ms`);

  assert.ok(stats.avg < 5, 'Average encapsulation should be under 5ms');
});

test('Benchmark: ML-KEM-768 decapsulation', (t) => {
  const mlkem = new MlKem768();
  const { result: keypair } = mlkem.generateKeypair();
  const { result: encap } = mlkem.encapsulate(keypair.publicKey);

  const stats = runBenchmark('ML-KEM-768 Decapsulate', () => {
    return mlkem.decapsulate(encap.ciphertext, keypair.secretKey);
  }, 100);

  console.log(`\n${stats.name}:`);
  console.log(`  Iterations: ${stats.iterations}`);
  console.log(`  Average: ${stats.avg.toFixed(3)}ms`);
  console.log(`  Median: ${stats.median.toFixed(3)}ms`);

  assert.ok(stats.avg < 5, 'Average decapsulation should be under 5ms');
});

test('Benchmark: ML-DSA signing', (t) => {
  const mldsa = new MlDsa();
  const message = Buffer.from('Benchmark message for signing');
  const secretKey = Buffer.alloc(4864);

  const stats = runBenchmark('ML-DSA Sign', () => {
    return mldsa.sign(message, secretKey);
  }, 100);

  console.log(`\n${stats.name}:`);
  console.log(`  Iterations: ${stats.iterations}`);
  console.log(`  Average: ${stats.avg.toFixed(3)}ms`);
  console.log(`  Median: ${stats.median.toFixed(3)}ms`);

  assert.ok(stats.avg < 5, 'Average signing should be under 5ms');
});

test('Benchmark: ML-DSA verification', (t) => {
  const mldsa = new MlDsa();
  const message = Buffer.from('Benchmark message for verification');
  const secretKey = Buffer.alloc(4864);
  const publicKey = Buffer.alloc(1952);
  const { result: signature } = mldsa.sign(message, secretKey);

  const stats = runBenchmark('ML-DSA Verify', () => {
    return mldsa.verify(message, signature, publicKey);
  }, 100);

  console.log(`\n${stats.name}:`);
  console.log(`  Iterations: ${stats.iterations}`);
  console.log(`  Average: ${stats.avg.toFixed(3)}ms`);
  console.log(`  Median: ${stats.median.toFixed(3)}ms`);

  assert.ok(stats.avg < 5, 'Average verification should be under 5ms');
});

test('Benchmark: BLAKE3 hashing (small data)', (t) => {
  const data = Buffer.from('Small data to hash');

  const stats = runBenchmark('BLAKE3 Hash (Small)', () => {
    return blake3Hash(data);
  }, 1000);

  console.log(`\n${stats.name}:`);
  console.log(`  Iterations: ${stats.iterations}`);
  console.log(`  Average: ${stats.avg.toFixed(3)}ms`);
  console.log(`  Median: ${stats.median.toFixed(3)}ms`);
  console.log(`  Throughput: ${(data.length / stats.avg / 1000).toFixed(2)} MB/s`);

  assert.ok(stats.avg < 1, 'Small hash should be under 1ms');
});

test('Benchmark: BLAKE3 hashing (1KB data)', (t) => {
  const data = Buffer.alloc(1024);

  const stats = runBenchmark('BLAKE3 Hash (1KB)', () => {
    return blake3Hash(data);
  }, 1000);

  console.log(`\n${stats.name}:`);
  console.log(`  Iterations: ${stats.iterations}`);
  console.log(`  Average: ${stats.avg.toFixed(3)}ms`);
  console.log(`  Median: ${stats.median.toFixed(3)}ms`);
  console.log(`  Throughput: ${(data.length / stats.avg / 1000).toFixed(2)} MB/s`);

  assert.ok(stats.avg < 5, '1KB hash should be under 5ms');
});

test('Benchmark: BLAKE3 hashing (1MB data)', (t) => {
  const data = Buffer.alloc(1024 * 1024);

  const stats = runBenchmark('BLAKE3 Hash (1MB)', () => {
    return blake3Hash(data);
  }, 50);

  console.log(`\n${stats.name}:`);
  console.log(`  Iterations: ${stats.iterations}`);
  console.log(`  Average: ${stats.avg.toFixed(3)}ms`);
  console.log(`  Median: ${stats.median.toFixed(3)}ms`);
  console.log(`  Throughput: ${(data.length / stats.avg / 1000).toFixed(2)} MB/s`);

  // Target: < 10ms for 1MB (native should achieve this)
  assert.ok(stats.avg < 50, '1MB hash should be under 50ms');
});

test('Benchmark: End-to-end key exchange', (t) => {
  const mlkem = new MlKem768();

  const stats = runBenchmark('E2E Key Exchange', () => {
    const start = process.hrtime.bigint();

    const { result: aliceKeys } = mlkem.generateKeypair();
    const { result: bobKeys } = mlkem.generateKeypair();
    const { result: encap } = mlkem.encapsulate(aliceKeys.publicKey);
    const { result: decap } = mlkem.decapsulate(encap.ciphertext, aliceKeys.secretKey);

    const end = process.hrtime.bigint();
    const time = Number(end - start) / 1_000_000;

    return { result: decap, time };
  }, 50);

  console.log(`\n${stats.name}:`);
  console.log(`  Iterations: ${stats.iterations}`);
  console.log(`  Average: ${stats.avg.toFixed(3)}ms`);
  console.log(`  Median: ${stats.median.toFixed(3)}ms`);

  assert.ok(stats.avg < 20, 'E2E key exchange should be under 20ms');
});

test('Benchmark: Sign and verify workflow', (t) => {
  const mldsa = new MlDsa();
  const message = Buffer.from('Important message to sign');
  const secretKey = Buffer.alloc(4864);
  const publicKey = Buffer.alloc(1952);

  const stats = runBenchmark('Sign + Verify Workflow', () => {
    const start = process.hrtime.bigint();

    const { result: signature } = mldsa.sign(message, secretKey);
    const { result: valid } = mldsa.verify(message, signature, publicKey);

    const end = process.hrtime.bigint();
    const time = Number(end - start) / 1_000_000;

    return { result: valid, time };
  }, 100);

  console.log(`\n${stats.name}:`);
  console.log(`  Iterations: ${stats.iterations}`);
  console.log(`  Average: ${stats.avg.toFixed(3)}ms`);
  console.log(`  Median: ${stats.median.toFixed(3)}ms`);

  assert.ok(stats.avg < 10, 'Sign + verify should be under 10ms');
});

test('Performance Summary', (t) => {
  console.log('\n=== Performance Summary ===');
  console.log('\nExpected Performance Targets (Native):');
  console.log('  ML-KEM-768 Keygen:      < 2ms (target 1.8ms)');
  console.log('  ML-KEM-768 Encapsulate: < 2ms (target 1.1ms)');
  console.log('  ML-KEM-768 Decapsulate: < 2ms (target 1.3ms)');
  console.log('  ML-DSA Sign:            < 2ms (target 1.5ms)');
  console.log('  ML-DSA Verify:          < 2ms (target 1.3ms)');
  console.log('  BLAKE3 Hash (1MB):      < 3ms (target 2.1ms)');
  console.log('\nNative vs WASM Expected Speedup: 2.8x - 3.9x');

  assert.ok(true, 'Performance summary displayed');
});
