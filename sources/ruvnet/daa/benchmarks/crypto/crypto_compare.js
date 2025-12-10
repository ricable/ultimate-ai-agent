#!/usr/bin/env node
/**
 * Compare native NAPI-rs vs WASM crypto performance
 * Runs both implementations and generates comparison report
 */

import Benchmark from 'benchmark';
import chalk from 'chalk';
import { randomBytes } from 'crypto';
import { writeFile } from 'fs/promises';

console.log(chalk.blue.bold('\n🏁 Native vs WASM Crypto Performance Comparison\n'));

// Try to load both implementations
let NativeCrypto, WasmCrypto;
let hasNative = false, hasWasm = false;

try {
  NativeCrypto = await import('@daa/qudag-native');
  hasNative = true;
  console.log(chalk.green('✓ Native NAPI-rs bindings loaded'));
} catch (err) {
  console.log(chalk.yellow('⚠ Native bindings not available'));
  console.log(chalk.gray('  Build with: cd ../qudag/qudag-napi && npm run build'));
}

try {
  WasmCrypto = await import('qudag-wasm');
  hasWasm = true;
  console.log(chalk.green('✓ WASM bindings loaded'));
} catch (err) {
  console.log(chalk.yellow('⚠ WASM bindings not available'));
  console.log(chalk.gray('  Build with: cd ../qudag/qudag-wasm && wasm-pack build --target nodejs'));
}

if (!hasNative && !hasWasm) {
  console.error(chalk.red('\n✗ No implementations available. Build at least one implementation first.'));
  process.exit(1);
}

console.log('');

// Prepare test data
const testData1KB = randomBytes(1024);
const testData10KB = randomBytes(10 * 1024);
const testData100KB = randomBytes(100 * 1024);
const testData1MB = randomBytes(1024 * 1024);
const testData10MB = randomBytes(10 * 1024 * 1024);
const testMessage = new TextEncoder().encode('Hello, quantum-resistant world!');

const results = {
  timestamp: new Date().toISOString(),
  platform: {
    node: process.version,
    arch: process.arch,
    platform: process.platform,
  },
  implementations: {
    native: hasNative,
    wasm: hasWasm,
  },
  benchmarks: []
};

/**
 * Run a benchmark comparing native and WASM implementations
 */
async function runComparison(name, nativeFn, wasmFn) {
  return new Promise((resolve) => {
    const suite = new Benchmark.Suite(name);
    const benchResults = { name, implementations: {} };

    console.log(chalk.cyan(`\n📊 ${name}`));

    if (hasNative && nativeFn) {
      suite.add('Native', nativeFn);
    }

    if (hasWasm && wasmFn) {
      suite.add('WASM', wasmFn);
    }

    suite.on('cycle', (event) => {
      const bench = event.target;
      const impl = bench.name;
      const meanMs = bench.stats.mean * 1000;
      const hz = bench.hz.toFixed(2);

      benchResults.implementations[impl] = {
        hz: bench.hz,
        mean: meanMs,
        median: bench.stats.median * 1000,
        rme: bench.stats.rme,
        samples: bench.stats.sample.length,
      };

      console.log(chalk.white(`  ${impl}: ${hz} ops/sec (${meanMs.toFixed(3)}ms/op) ±${bench.stats.rme.toFixed(2)}%`));
    });

    suite.on('complete', function() {
      if (hasNative && hasWasm) {
        const nativeMean = benchResults.implementations.Native?.mean || 0;
        const wasmMean = benchResults.implementations.WASM?.mean || 0;

        if (nativeMean > 0 && wasmMean > 0) {
          const speedup = wasmMean / nativeMean;
          benchResults.speedup = speedup;

          if (speedup > 1) {
            console.log(chalk.green(`  🚀 Native is ${speedup.toFixed(2)}x faster than WASM`));
          } else {
            console.log(chalk.yellow(`  ⚠️  WASM is ${(1/speedup).toFixed(2)}x faster than Native`));
          }
        }
      }

      results.benchmarks.push(benchResults);
      resolve();
    });

    suite.run({ async: true });
  });
}

// Run all comparisons
async function runAllComparisons() {
  // ML-KEM-768 Key Generation
  await runComparison(
    'ML-KEM-768 Key Generation',
    hasNative ? () => {
      const mlkem = new NativeCrypto.Crypto.MlKem768();
      mlkem.generateKeypair();
    } : null,
    hasWasm ? () => {
      const mlkem = new WasmCrypto.MlKem768();
      mlkem.generateKeypair();
    } : null
  );

  // ML-KEM-768 Encapsulation
  const nativeKeypair = hasNative ? new NativeCrypto.Crypto.MlKem768().generateKeypair() : null;
  const wasmKeypair = hasWasm ? new WasmCrypto.MlKem768().generateKeypair() : null;

  await runComparison(
    'ML-KEM-768 Encapsulation',
    hasNative ? () => {
      const mlkem = new NativeCrypto.Crypto.MlKem768();
      mlkem.encapsulate(nativeKeypair.publicKey);
    } : null,
    hasWasm ? () => {
      const mlkem = new WasmCrypto.MlKem768();
      mlkem.encapsulate(wasmKeypair.publicKey);
    } : null
  );

  // ML-DSA Signing
  const nativeDsaKeypair = hasNative ? new NativeCrypto.Crypto.MlDsa().generateKeypair() : null;
  const wasmDsaKeypair = hasWasm ? new WasmCrypto.MlDsa().generateKeypair() : null;

  await runComparison(
    'ML-DSA Signing',
    hasNative ? () => {
      const mldsa = new NativeCrypto.Crypto.MlDsa();
      mldsa.sign(testMessage, nativeDsaKeypair.secretKey);
    } : null,
    hasWasm ? () => {
      const mldsa = new WasmCrypto.MlDsa();
      mldsa.sign(testMessage, wasmDsaKeypair.secretKey);
    } : null
  );

  // BLAKE3 Hashing at various sizes
  for (const [label, data] of [
    ['1KB', testData1KB],
    ['10KB', testData10KB],
    ['100KB', testData100KB],
    ['1MB', testData1MB],
    ['10MB', testData10MB]
  ]) {
    await runComparison(
      `BLAKE3 Hash ${label}`,
      hasNative ? () => NativeCrypto.Crypto.blake3Hash(data) : null,
      hasWasm ? () => WasmCrypto.blake3Hash(data) : null
    );
  }

  // Quantum Fingerprint
  for (const [label, data] of [
    ['1KB', testData1KB],
    ['10KB', testData10KB],
    ['100KB', testData100KB],
    ['1MB', testData1MB]
  ]) {
    await runComparison(
      `Quantum Fingerprint ${label}`,
      hasNative ? () => NativeCrypto.Crypto.quantumFingerprint(data) : null,
      hasWasm ? () => WasmCrypto.quantumFingerprint(data) : null
    );
  }

  // Generate summary
  console.log(chalk.blue.bold('\n📈 Performance Summary\n'));

  if (hasNative && hasWasm) {
    const avgSpeedup = results.benchmarks
      .filter(b => b.speedup)
      .reduce((sum, b) => sum + b.speedup, 0) / results.benchmarks.filter(b => b.speedup).length;

    console.log(chalk.green(`Average Native speedup: ${avgSpeedup.toFixed(2)}x`));

    const fastest = results.benchmarks
      .filter(b => b.speedup && b.speedup > 1)
      .sort((a, b) => b.speedup - a.speedup)
      .slice(0, 3);

    if (fastest.length > 0) {
      console.log(chalk.cyan('\nTop 3 Native speedups:'));
      fastest.forEach((bench, i) => {
        console.log(chalk.white(`  ${i + 1}. ${bench.name}: ${bench.speedup.toFixed(2)}x faster`));
      });
    }
  }

  // Save results
  await writeFile(
    'reports/crypto_comparison.json',
    JSON.stringify(results, null, 2)
  );

  console.log(chalk.gray('\n💾 Results saved to reports/crypto_comparison.json\n'));
}

runAllComparisons().catch(err => {
  console.error(chalk.red('Error running comparisons:'), err);
  process.exit(1);
});
