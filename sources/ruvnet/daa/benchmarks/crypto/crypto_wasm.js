#!/usr/bin/env node
/**
 * WASM crypto benchmarks using benchmark.js
 * Measures performance of quantum-resistant operations in WASM
 */

import Benchmark from 'benchmark';
import chalk from 'chalk';
import { randomBytes } from 'crypto';

// Try to load WASM bindings
let Crypto;
try {
  const wasm = await import('qudag-wasm');
  Crypto = wasm;
} catch (err) {
  console.error(chalk.red('Failed to load qudag-wasm. Make sure it is built:'));
  console.error(chalk.yellow('  cd ../qudag/qudag-wasm && wasm-pack build --target nodejs'));
  process.exit(1);
}

const suite = new Benchmark.Suite('WASM Crypto Benchmarks');

// Prepare test data
const testData1KB = randomBytes(1024);
const testData10KB = randomBytes(10 * 1024);
const testData100KB = randomBytes(100 * 1024);
const testData1MB = randomBytes(1024 * 1024);
const testData10MB = randomBytes(10 * 1024 * 1024);

console.log(chalk.blue.bold('\n🔬 WASM Crypto Benchmarks\n'));
console.log(chalk.gray('Testing quantum-resistant cryptography operations...\n'));

// ML-KEM-768 benchmarks
suite.add('WASM ML-KEM-768 Keygen', () => {
  const mlkem = new Crypto.MlKem768();
  const keypair = mlkem.generateKeypair();
});

suite.add('WASM ML-KEM-768 Encapsulate', () => {
  const mlkem = new Crypto.MlKem768();
  const keypair = mlkem.generateKeypair();
  const result = mlkem.encapsulate(keypair.publicKey);
});

suite.add('WASM ML-KEM-768 Decapsulate', () => {
  const mlkem = new Crypto.MlKem768();
  const keypair = mlkem.generateKeypair();
  const { ciphertext, sharedSecret } = mlkem.encapsulate(keypair.publicKey);
  const decrypted = mlkem.decapsulate(ciphertext, keypair.secretKey);
});

// ML-DSA benchmarks
suite.add('WASM ML-DSA Sign', () => {
  const mldsa = new Crypto.MlDsa();
  const keypair = mldsa.generateKeypair();
  const message = new TextEncoder().encode('Hello, quantum-resistant world!');
  const signature = mldsa.sign(message, keypair.secretKey);
});

suite.add('WASM ML-DSA Verify', () => {
  const mldsa = new Crypto.MlDsa();
  const keypair = mldsa.generateKeypair();
  const message = new TextEncoder().encode('Hello, quantum-resistant world!');
  const signature = mldsa.sign(message, keypair.secretKey);
  const valid = mldsa.verify(message, signature, keypair.publicKey);
});

// BLAKE3 benchmarks at various sizes
suite.add('WASM BLAKE3 Hash 1KB', () => {
  const hash = Crypto.blake3Hash(testData1KB);
});

suite.add('WASM BLAKE3 Hash 10KB', () => {
  const hash = Crypto.blake3Hash(testData10KB);
});

suite.add('WASM BLAKE3 Hash 100KB', () => {
  const hash = Crypto.blake3Hash(testData100KB);
});

suite.add('WASM BLAKE3 Hash 1MB', () => {
  const hash = Crypto.blake3Hash(testData1MB);
});

suite.add('WASM BLAKE3 Hash 10MB', () => {
  const hash = Crypto.blake3Hash(testData10MB);
});

// Quantum fingerprint benchmarks
suite.add('WASM Quantum Fingerprint 1KB', () => {
  const fingerprint = Crypto.quantumFingerprint(testData1KB);
});

suite.add('WASM Quantum Fingerprint 10KB', () => {
  const fingerprint = Crypto.quantumFingerprint(testData10KB);
});

suite.add('WASM Quantum Fingerprint 100KB', () => {
  const fingerprint = Crypto.quantumFingerprint(testData100KB);
});

suite.add('WASM Quantum Fingerprint 1MB', () => {
  const fingerprint = Crypto.quantumFingerprint(testData1MB);
});

// Full workflow benchmarks
suite.add('WASM Full Key Exchange', () => {
  const mlkem = new Crypto.MlKem768();

  // Alice generates keypair
  const aliceKeypair = mlkem.generateKeypair();

  // Bob encapsulates
  const { ciphertext, sharedSecret: bobSecret } = mlkem.encapsulate(aliceKeypair.publicKey);

  // Alice decapsulates
  const aliceSecret = mlkem.decapsulate(ciphertext, aliceKeypair.secretKey);
});

suite.add('WASM Full Signature Workflow', () => {
  const mldsa = new Crypto.MlDsa();
  const message = new TextEncoder().encode('Transaction data to be signed');

  // Generate keypair
  const keypair = mldsa.generateKeypair();

  // Sign
  const signature = mldsa.sign(message, keypair.secretKey);

  // Verify
  const valid = mldsa.verify(message, signature, keypair.publicKey);
});

// Event handlers
suite.on('cycle', (event) => {
  const bench = event.target;
  const name = bench.name;
  const opsPerSec = bench.hz.toFixed(2);
  const rme = bench.stats.rme.toFixed(2);
  const runs = bench.stats.sample.length;

  console.log(chalk.green('✓'), chalk.white(name));
  console.log(chalk.gray(`  ${opsPerSec} ops/sec ±${rme}% (${runs} runs sampled)`));
});

suite.on('complete', function() {
  console.log(chalk.blue.bold('\n📊 WASM Benchmark Results Summary\n'));

  const results = this.filter('fastest').map(function(bench) {
    return {
      name: bench.name,
      hz: bench.hz,
      mean: bench.stats.mean * 1000, // Convert to ms
      median: bench.stats.median * 1000,
      p95: bench.stats.sample.sort((a, b) => a - b)[Math.floor(bench.stats.sample.length * 0.95)] * 1000,
      p99: bench.stats.sample.sort((a, b) => a - b)[Math.floor(bench.stats.sample.length * 0.99)] * 1000,
    };
  });

  console.log(chalk.cyan('Fastest operations:'));
  results.forEach(result => {
    console.log(chalk.white(`  ${result.name}: ${result.hz.toFixed(2)} ops/sec`));
  });

  // Save results to JSON
  const fs = await import('fs/promises');
  await fs.writeFile(
    'reports/wasm_crypto_results.json',
    JSON.stringify({ timestamp: new Date().toISOString(), results: this.map(b => ({
      name: b.name,
      hz: b.hz,
      stats: {
        mean: b.stats.mean * 1000,
        median: b.stats.median * 1000,
        rme: b.stats.rme,
        samples: b.stats.sample.length
      }
    })) }, null, 2)
  );

  console.log(chalk.gray('\nResults saved to reports/wasm_crypto_results.json'));
});

suite.on('error', (event) => {
  console.error(chalk.red('Error:'), event.target.error);
});

// Run benchmarks
suite.run({ async: true });
