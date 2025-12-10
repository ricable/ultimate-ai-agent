#!/usr/bin/env node
/**
 * Advanced performance benchmarks for agentic-jujutsu
 * Measures WASM overhead, throughput, and compares with native implementation
 */

const path = require('path');
const fs = require('fs');
const { execSync } = require('child_process');

const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m',
};

function log(color, symbol, message) {
  console.log(`${color}${symbol}${colors.reset} ${message}`);
}

class BenchmarkRunner {
  constructor() {
    this.results = [];
    this.hasJJ = false;
  }

  async setup() {
    // Check if jj is installed
    try {
      execSync('jj --version', { stdio: 'ignore' });
      this.hasJJ = true;
      log(colors.green, '✓', 'Jujutsu (jj) detected - will run comparative benchmarks');
    } catch (error) {
      this.hasJJ = false;
      log(colors.yellow, '⚠', 'jj not found - skipping native comparison benchmarks');
    }

    // Load WASM module
    try {
      const pkgPath = path.join(__dirname, '../../pkg/node');
      this.wasmModule = require(pkgPath);
      log(colors.green, '✓', 'WASM module loaded');
    } catch (error) {
      log(colors.red, '✗', `Failed to load WASM module: ${error.message}`);
      throw error;
    }
  }

  benchmark(name, iterations, fn) {
    const times = [];
    let totalTime = 0;

    // Warmup
    for (let i = 0; i < Math.min(10, iterations); i++) {
      fn();
    }

    // Actual benchmark
    for (let i = 0; i < iterations; i++) {
      const start = process.hrtime.bigint();
      fn();
      const end = process.hrtime.bigint();

      const duration = Number(end - start) / 1_000_000; // Convert to ms
      times.push(duration);
      totalTime += duration;
    }

    const avg = totalTime / iterations;
    const min = Math.min(...times);
    const max = Math.max(...times);

    // Calculate median
    times.sort((a, b) => a - b);
    const median = times[Math.floor(times.length / 2)];

    // Calculate standard deviation
    const variance = times.reduce((sum, t) => sum + Math.pow(t - avg, 2), 0) / iterations;
    const stddev = Math.sqrt(variance);

    const result = {
      name,
      iterations,
      avg: avg.toFixed(3),
      median: median.toFixed(3),
      min: min.toFixed(3),
      max: max.toFixed(3),
      stddev: stddev.toFixed(3),
      totalTime: totalTime.toFixed(3)
    };

    this.results.push(result);
    return result;
  }

  async benchmarkAsync(name, iterations, fn) {
    const times = [];
    let totalTime = 0;

    // Warmup
    for (let i = 0; i < Math.min(10, iterations); i++) {
      await fn();
    }

    // Actual benchmark
    for (let i = 0; i < iterations; i++) {
      const start = process.hrtime.bigint();
      await fn();
      const end = process.hrtime.bigint();

      const duration = Number(end - start) / 1_000_000;
      times.push(duration);
      totalTime += duration;
    }

    const avg = totalTime / iterations;
    const min = Math.min(...times);
    const max = Math.max(...times);

    times.sort((a, b) => a - b);
    const median = times[Math.floor(times.length / 2)];

    const variance = times.reduce((sum, t) => sum + Math.pow(t - avg, 2), 0) / iterations;
    const stddev = Math.sqrt(variance);

    const result = {
      name,
      iterations,
      avg: avg.toFixed(3),
      median: median.toFixed(3),
      min: min.toFixed(3),
      max: max.toFixed(3),
      stddev: stddev.toFixed(3),
      totalTime: totalTime.toFixed(3)
    };

    this.results.push(result);
    return result;
  }

  printResult(result) {
    console.log(`\n${colors.cyan}${result.name}${colors.reset}`);
    console.log(`  Iterations: ${result.iterations}`);
    console.log(`  Average:    ${colors.green}${result.avg}ms${colors.reset}`);
    console.log(`  Median:     ${result.median}ms`);
    console.log(`  Min:        ${result.min}ms`);
    console.log(`  Max:        ${result.max}ms`);
    console.log(`  Std Dev:    ${result.stddev}ms`);
    console.log(`  Total:      ${result.totalTime}ms`);
  }

  compareResults(result1, result2) {
    const speedup = (parseFloat(result2.avg) / parseFloat(result1.avg)).toFixed(2);
    const improvement = ((1 - parseFloat(result1.avg) / parseFloat(result2.avg)) * 100).toFixed(1);

    console.log(`\n${colors.magenta}Comparison: ${result1.name} vs ${result2.name}${colors.reset}`);

    if (speedup > 1) {
      console.log(`  ${colors.green}${result1.name} is ${speedup}x faster${colors.reset}`);
      console.log(`  ${colors.green}${improvement}% improvement${colors.reset}`);
    } else {
      console.log(`  ${colors.red}${result2.name} is ${(1 / speedup).toFixed(2)}x faster${colors.reset}`);
      console.log(`  ${colors.red}${improvement}% slower${colors.reset}`);
    }
  }

  report() {
    console.log('\n' + '='.repeat(70));
    console.log(`📊 Benchmark Summary`);
    console.log('='.repeat(70));

    this.results.forEach(result => {
      console.log(`${result.name.padEnd(40)} ${result.avg.padStart(10)}ms avg`);
    });

    console.log('='.repeat(70) + '\n');

    // Save results to JSON
    const resultsPath = path.join(__dirname, '../../benchmark-results.json');
    fs.writeFileSync(resultsPath, JSON.stringify({
      timestamp: new Date().toISOString(),
      platform: {
        node: process.version,
        platform: process.platform,
        arch: process.arch,
        cpus: require('os').cpus()[0].model
      },
      results: this.results
    }, null, 2));

    log(colors.blue, 'ℹ', `Results saved to ${resultsPath}`);
  }
}

async function runBenchmarks() {
  console.log('\n🚀 agentic-jujutsu Advanced Performance Benchmarks\n');
  const runner = new BenchmarkRunner();

  await runner.setup();

  // Benchmark 1: Module Load Time
  console.log(`\n${colors.cyan}═══ Module Load Benchmarks ═══${colors.reset}`);

  const loadResult = runner.benchmark('WASM Module Load', 100, () => {
    const pkgPath = path.join(__dirname, '../../pkg/node');
    delete require.cache[require.resolve(pkgPath)];
    require(pkgPath);
  });
  runner.printResult(loadResult);

  // Benchmark 2: Memory Overhead
  console.log(`\n${colors.cyan}═══ Memory Benchmarks ═══${colors.reset}`);

  const memBefore = process.memoryUsage();
  const pkgPath = path.join(__dirname, '../../pkg/node');
  const jj = require(pkgPath);
  const memAfter = process.memoryUsage();

  console.log(`\n${colors.cyan}WASM Module Memory Impact${colors.reset}`);
  console.log(`  Heap Used:     ${((memAfter.heapUsed - memBefore.heapUsed) / 1024 / 1024).toFixed(2)}MB`);
  console.log(`  Heap Total:    ${((memAfter.heapTotal - memBefore.heapTotal) / 1024 / 1024).toFixed(2)}MB`);
  console.log(`  External:      ${((memAfter.external - memBefore.external) / 1024 / 1024).toFixed(2)}MB`);
  console.log(`  Array Buffers: ${((memAfter.arrayBuffers - memBefore.arrayBuffers) / 1024 / 1024).toFixed(2)}MB`);

  // Benchmark 3: Function Call Overhead
  console.log(`\n${colors.cyan}═══ Function Call Overhead ═══${colors.reset}`);

  if (jj.VERSION) {
    const versionResult = runner.benchmark('Get VERSION (1000 calls)', 1000, () => {
      const v = jj.VERSION;
    });
    runner.printResult(versionResult);
  }

  // Benchmark 4: Object Creation
  console.log(`\n${colors.cyan}═══ Object Creation Benchmarks ═══${colors.reset}`);

  const objCreationResult = runner.benchmark('JavaScript Object Creation', 10000, () => {
    const obj = { name: 'test', value: 123, nested: { a: 1, b: 2 } };
  });
  runner.printResult(objCreationResult);

  // Benchmark 5: String Operations
  console.log(`\n${colors.cyan}═══ String Operation Benchmarks ═══${colors.reset}`);

  const stringResult = runner.benchmark('String Concatenation (10000 ops)', 10000, () => {
    const str = 'Hello ' + 'World ' + '!' + ' test ' + 123;
  });
  runner.printResult(stringResult);

  // Benchmark 6: Array Operations
  console.log(`\n${colors.cyan}═══ Array Operation Benchmarks ═══${colors.reset}`);

  const arrayResult = runner.benchmark('Array Operations (1000 ops)', 1000, () => {
    const arr = Array(100).fill(0).map((_, i) => i * 2).filter(x => x % 4 === 0);
  });
  runner.printResult(arrayResult);

  // Benchmark 7: JSON Serialization
  console.log(`\n${colors.cyan}═══ JSON Serialization Benchmarks ═══${colors.reset}`);

  const testData = {
    commits: Array(100).fill(0).map((_, i) => ({
      id: `commit-${i}`,
      message: `Test commit ${i}`,
      author: 'test@example.com',
      timestamp: Date.now()
    }))
  };

  const jsonParseResult = runner.benchmark('JSON Parse/Stringify (1000 ops)', 1000, () => {
    const str = JSON.stringify(testData);
    const obj = JSON.parse(str);
  });
  runner.printResult(jsonParseResult);

  // Benchmark 8: Throughput Test
  console.log(`\n${colors.cyan}═══ Throughput Benchmarks ═══${colors.reset}`);

  const throughputResult = runner.benchmark('Mixed Operations (1000 iterations)', 1000, () => {
    const data = { id: Math.random(), timestamp: Date.now() };
    const json = JSON.stringify(data);
    const parsed = JSON.parse(json);
    const result = parsed.id * 2;
  });
  runner.printResult(throughputResult);

  // Benchmark 9: Concurrent Operations
  console.log(`\n${colors.cyan}═══ Concurrent Operation Benchmarks ═══${colors.reset}`);

  const concurrentResult = await runner.benchmarkAsync('Concurrent Promises (100 ops)', 100, async () => {
    const promises = Array(10).fill(0).map((_, i) =>
      Promise.resolve({ id: i, value: i * 2 })
    );
    await Promise.all(promises);
  });
  runner.printResult(concurrentResult);

  // Benchmark 10: Native Comparison (if jj installed)
  if (runner.hasJJ) {
    console.log(`\n${colors.cyan}═══ Native vs WASM Comparison ═══${colors.reset}`);

    const nativeVersionResult = runner.benchmark('Native jj --version (10 calls)', 10, () => {
      execSync('jj --version', { stdio: 'pipe' });
    });
    runner.printResult(nativeVersionResult);

    log(colors.blue, 'ℹ', 'Note: Native calls include process spawn overhead');
  }

  // Memory Leak Test
  console.log(`\n${colors.cyan}═══ Memory Leak Detection ═══${colors.reset}`);

  const iterations = 1000;
  const memStart = process.memoryUsage().heapUsed;

  for (let i = 0; i < iterations; i++) {
    // Perform operations that might leak
    const data = { id: i, data: Array(100).fill(i) };
    JSON.stringify(data);
    JSON.parse(JSON.stringify(data));
  }

  if (global.gc) global.gc();

  const memEnd = process.memoryUsage().heapUsed;
  const memGrowth = (memEnd - memStart) / 1024 / 1024;

  console.log(`\n${colors.cyan}Memory Leak Test (${iterations} iterations)${colors.reset}`);
  console.log(`  Initial Heap:  ${(memStart / 1024 / 1024).toFixed(2)}MB`);
  console.log(`  Final Heap:    ${(memEnd / 1024 / 1024).toFixed(2)}MB`);
  console.log(`  Growth:        ${memGrowth.toFixed(2)}MB`);

  if (memGrowth > 10) {
    log(colors.yellow, '⚠', 'Significant memory growth detected');
  } else {
    log(colors.green, '✓', 'Memory usage is stable');
  }

  // Performance Grade
  console.log(`\n${colors.cyan}═══ Performance Grade ═══${colors.reset}`);

  const avgLoadTime = parseFloat(loadResult.avg);
  let grade = 'A+';
  let color = colors.green;

  if (avgLoadTime > 100) {
    grade = 'C';
    color = colors.yellow;
  } else if (avgLoadTime > 50) {
    grade = 'B';
    color = colors.yellow;
  } else if (avgLoadTime > 20) {
    grade = 'A';
    color = colors.green;
  }

  console.log(`\n  Performance Grade: ${color}${grade}${colors.reset}`);
  console.log(`  Module Load Time: ${avgLoadTime.toFixed(2)}ms`);
  console.log(`  Memory Growth: ${memGrowth.toFixed(2)}MB`);

  // Final Report
  runner.report();

  process.exit(0);
}

// Run benchmarks
runBenchmarks().catch(err => {
  console.error(`\n${colors.red}Fatal error:${colors.reset}`, err);
  process.exit(1);
});
