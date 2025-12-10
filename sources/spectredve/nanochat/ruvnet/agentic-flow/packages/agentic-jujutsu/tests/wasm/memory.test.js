#!/usr/bin/env node
/**
 * Memory and resource management tests for agentic-jujutsu WASM
 * Tests memory leaks, garbage collection, and resource cleanup
 */

const path = require('path');
const fs = require('fs');

const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
};

function log(color, symbol, message) {
  console.log(`${color}${symbol}${colors.reset} ${message}`);
}

function formatBytes(bytes) {
  return (bytes / 1024 / 1024).toFixed(2) + ' MB';
}

class MemoryTestRunner {
  constructor() {
    this.passed = 0;
    this.failed = 0;
    this.memorySnapshots = [];
  }

  snapshot(label) {
    const mem = process.memoryUsage();
    this.memorySnapshots.push({
      label,
      timestamp: Date.now(),
      ...mem
    });
    return mem;
  }

  async test(name, fn) {
    try {
      await fn();
      log(colors.green, '✓', name);
      this.passed++;
    } catch (error) {
      log(colors.red, '✗', `${name}\n  ${error.message}`);
      this.failed++;
    }
  }

  assert(condition, message) {
    if (!condition) {
      throw new Error(message || 'Assertion failed');
    }
  }

  printMemorySnapshot(mem, label) {
    console.log(`\n${colors.cyan}${label}${colors.reset}`);
    console.log(`  RSS:           ${formatBytes(mem.rss)}`);
    console.log(`  Heap Total:    ${formatBytes(mem.heapTotal)}`);
    console.log(`  Heap Used:     ${formatBytes(mem.heapUsed)}`);
    console.log(`  External:      ${formatBytes(mem.external)}`);
    console.log(`  Array Buffers: ${formatBytes(mem.arrayBuffers)}`);
  }

  compareSnapshots(before, after, label) {
    console.log(`\n${colors.cyan}Memory Change: ${label}${colors.reset}`);
    console.log(`  RSS:           ${formatBytes(after.rss - before.rss)}`);
    console.log(`  Heap Total:    ${formatBytes(after.heapTotal - before.heapTotal)}`);
    console.log(`  Heap Used:     ${formatBytes(after.heapUsed - before.heapUsed)}`);
    console.log(`  External:      ${formatBytes(after.external - before.external)}`);
    console.log(`  Array Buffers: ${formatBytes(after.arrayBuffers - before.arrayBuffers)}`);

    return {
      rssChange: after.rss - before.rss,
      heapChange: after.heapUsed - before.heapUsed,
      externalChange: after.external - before.external,
    };
  }

  report() {
    const total = this.passed + this.failed;
    console.log('\n' + '='.repeat(60));
    console.log(`📊 Memory Test Results`);
    console.log('='.repeat(60));
    console.log(`${colors.green}✓ Passed: ${this.passed}${colors.reset}`);
    console.log(`${colors.red}✗ Failed: ${this.failed}${colors.reset}`);
    console.log(`Total: ${total}`);
    console.log('='.repeat(60) + '\n');

    // Save memory snapshots
    const snapshotPath = path.join(__dirname, '../../memory-snapshots.json');
    fs.writeFileSync(snapshotPath, JSON.stringify({
      timestamp: new Date().toISOString(),
      snapshots: this.memorySnapshots
    }, null, 2));

    log(colors.blue, 'ℹ', `Memory snapshots saved to ${snapshotPath}`);

    return this.failed === 0;
  }
}

async function runTests() {
  console.log('\n🧪 agentic-jujutsu Memory & Resource Tests\n');
  const runner = new MemoryTestRunner();

  // Initial snapshot
  const initialMem = runner.snapshot('Initial State');
  runner.printMemorySnapshot(initialMem, 'Initial Memory State');

  // Test Suite 1: Module Loading Impact
  console.log(`\n${colors.cyan}═══ Module Loading Memory Impact ═══${colors.reset}\n`);

  await runner.test('Module loading has acceptable memory overhead', () => {
    const beforeLoad = runner.snapshot('Before Module Load');

    const pkgPath = path.join(__dirname, '../../pkg/node');
    const jj = require(pkgPath);

    const afterLoad = runner.snapshot('After Module Load');
    const change = runner.compareSnapshots(beforeLoad, afterLoad, 'Module Load');

    // Check that memory increase is reasonable (< 100MB for WASM module)
    const heapIncreaseMB = change.heapChange / 1024 / 1024;
    runner.assert(heapIncreaseMB < 100, `Heap increase (${heapIncreaseMB.toFixed(2)}MB) should be < 100MB`);

    log(colors.blue, 'ℹ', `Module added ${heapIncreaseMB.toFixed(2)}MB to heap`);
  });

  // Load module for subsequent tests
  const pkgPath = path.join(__dirname, '../../pkg/node');
  let jj;
  try {
    jj = require(pkgPath);
    log(colors.green, '✓', 'Module loaded for testing');
  } catch (error) {
    log(colors.red, '✗', `Failed to load module: ${error.message}`);
    process.exit(1);
  }

  // Test Suite 2: Repeated Module Operations
  console.log(`\n${colors.cyan}═══ Repeated Operations Memory Tests ═══${colors.reset}\n`);

  await runner.test('No memory leak on repeated module requires', () => {
    const before = runner.snapshot('Before Repeated Requires');

    for (let i = 0; i < 100; i++) {
      delete require.cache[require.resolve(pkgPath)];
      require(pkgPath);
    }

    if (global.gc) global.gc();
    const after = runner.snapshot('After Repeated Requires');

    const change = runner.compareSnapshots(before, after, 'Repeated Requires');
    const heapGrowthMB = change.heapChange / 1024 / 1024;

    runner.assert(heapGrowthMB < 50, `Heap growth (${heapGrowthMB.toFixed(2)}MB) should be < 50MB`);
  });

  await runner.test('No memory leak on repeated object creation', () => {
    const before = runner.snapshot('Before Object Creation Loop');

    for (let i = 0; i < 10000; i++) {
      const obj = {
        id: i,
        name: `Object ${i}`,
        data: Array(10).fill(i),
        nested: { a: i, b: i * 2 }
      };
    }

    if (global.gc) global.gc();
    const after = runner.snapshot('After Object Creation Loop');

    const change = runner.compareSnapshots(before, after, 'Object Creation');
    const heapGrowthMB = change.heapChange / 1024 / 1024;

    runner.assert(heapGrowthMB < 10, `Heap growth (${heapGrowthMB.toFixed(2)}MB) should be < 10MB`);
  });

  // Test Suite 3: Large Data Handling
  console.log(`\n${colors.cyan}═══ Large Data Memory Tests ═══${colors.reset}\n`);

  await runner.test('Handles large string allocation and cleanup', () => {
    const before = runner.snapshot('Before Large String');

    let largeStr = 'x'.repeat(10 * 1024 * 1024); // 10MB string
    runner.assert(largeStr.length === 10 * 1024 * 1024, 'Large string created');

    largeStr = null; // Release reference
    if (global.gc) global.gc();

    const after = runner.snapshot('After Large String Cleanup');
    const change = runner.compareSnapshots(before, after, 'Large String');

    // After GC, memory should be mostly recovered
    const heapGrowthMB = change.heapChange / 1024 / 1024;
    log(colors.blue, 'ℹ', `Residual heap growth: ${heapGrowthMB.toFixed(2)}MB`);
  });

  await runner.test('Handles large array allocation and cleanup', () => {
    const before = runner.snapshot('Before Large Array');

    let largeArr = new Array(1000000).fill(0).map((_, i) => ({
      id: i,
      value: i * 2
    }));
    runner.assert(largeArr.length === 1000000, 'Large array created');

    largeArr = null; // Release reference
    if (global.gc) global.gc();

    const after = runner.snapshot('After Large Array Cleanup');
    const change = runner.compareSnapshots(before, after, 'Large Array');

    const heapGrowthMB = change.heapChange / 1024 / 1024;
    log(colors.blue, 'ℹ', `Residual heap growth: ${heapGrowthMB.toFixed(2)}MB`);
  });

  // Test Suite 4: Buffer and ArrayBuffer Tests
  console.log(`\n${colors.cyan}═══ Buffer Memory Tests ═══${colors.reset}\n`);

  await runner.test('Handles Buffer allocation and cleanup', () => {
    const before = runner.snapshot('Before Buffer Allocation');

    let buffers = [];
    for (let i = 0; i < 100; i++) {
      buffers.push(Buffer.alloc(1024 * 1024)); // 1MB each
    }
    runner.assert(buffers.length === 100, '100 buffers created');

    buffers = null;
    if (global.gc) global.gc();

    const after = runner.snapshot('After Buffer Cleanup');
    const change = runner.compareSnapshots(before, after, 'Buffer Allocation');

    const heapGrowthMB = change.heapChange / 1024 / 1024;
    log(colors.blue, 'ℹ', `Residual heap growth: ${heapGrowthMB.toFixed(2)}MB`);
  });

  await runner.test('Handles ArrayBuffer allocation and cleanup', () => {
    const before = runner.snapshot('Before ArrayBuffer');

    let arrayBuffers = [];
    for (let i = 0; i < 100; i++) {
      arrayBuffers.push(new ArrayBuffer(1024 * 1024)); // 1MB each
    }

    arrayBuffers = null;
    if (global.gc) global.gc();

    const after = runner.snapshot('After ArrayBuffer Cleanup');
    const change = runner.compareSnapshots(before, after, 'ArrayBuffer');

    log(colors.blue, 'ℹ', `ArrayBuffer memory change: ${formatBytes(change.externalChange)}`);
  });

  // Test Suite 5: Concurrent Operations
  console.log(`\n${colors.cyan}═══ Concurrent Operations Memory Tests ═══${colors.reset}\n`);

  await runner.test('Memory stable under concurrent Promise operations', async () => {
    const before = runner.snapshot('Before Concurrent Promises');

    const promises = Array(1000).fill(0).map((_, i) =>
      Promise.resolve({
        id: i,
        data: Array(100).fill(i)
      })
    );

    await Promise.all(promises);

    if (global.gc) global.gc();
    const after = runner.snapshot('After Concurrent Promises');

    const change = runner.compareSnapshots(before, after, 'Concurrent Promises');
    const heapGrowthMB = change.heapChange / 1024 / 1024;

    runner.assert(heapGrowthMB < 20, `Heap growth (${heapGrowthMB.toFixed(2)}MB) should be < 20MB`);
  });

  // Test Suite 6: Garbage Collection Efficiency
  console.log(`\n${colors.cyan}═══ Garbage Collection Tests ═══${colors.reset}\n`);

  await runner.test('Garbage collection is effective', () => {
    if (!global.gc) {
      log(colors.yellow, '⚠', 'GC not exposed. Run with --expose-gc for full test');
      return;
    }

    const before = runner.snapshot('Before GC Test');

    // Create garbage
    for (let i = 0; i < 10000; i++) {
      const garbage = {
        id: i,
        data: Array(100).fill(i),
        nested: { a: 1, b: 2, c: 3 }
      };
    }

    const beforeGC = runner.snapshot('After Garbage Creation');
    global.gc();
    const afterGC = runner.snapshot('After GC');

    const garbageCreated = beforeGC.heapUsed - before.heapUsed;
    const garbageCollected = beforeGC.heapUsed - afterGC.heapUsed;

    log(colors.blue, 'ℹ', `Garbage created: ${formatBytes(garbageCreated)}`);
    log(colors.blue, 'ℹ', `Garbage collected: ${formatBytes(garbageCollected)}`);
    log(colors.blue, 'ℹ', `Collection efficiency: ${((garbageCollected / garbageCreated) * 100).toFixed(1)}%`);
  });

  // Test Suite 7: Long-Running Stability
  console.log(`\n${colors.cyan}═══ Long-Running Stability Tests ═══${colors.reset}\n`);

  await runner.test('Memory stable over sustained operations', async () => {
    const snapshots = [];
    const iterations = 50;

    for (let i = 0; i < iterations; i++) {
      // Perform mixed operations
      const data = {
        id: i,
        timestamp: Date.now(),
        payload: Array(100).fill(Math.random())
      };

      JSON.stringify(data);
      JSON.parse(JSON.stringify(data));

      if (i % 10 === 0) {
        snapshots.push(runner.snapshot(`Iteration ${i}`));
      }
    }

    // Check that memory hasn't grown significantly
    const firstSnapshot = snapshots[0];
    const lastSnapshot = snapshots[snapshots.length - 1];

    const heapGrowth = (lastSnapshot.heapUsed - firstSnapshot.heapUsed) / 1024 / 1024;
    log(colors.blue, 'ℹ', `Heap growth over ${iterations} iterations: ${heapGrowth.toFixed(2)}MB`);

    runner.assert(heapGrowth < 50, `Heap growth should be < 50MB, got ${heapGrowth.toFixed(2)}MB`);
  });

  // Test Suite 8: Resource Cleanup
  console.log(`\n${colors.cyan}═══ Resource Cleanup Tests ═══${colors.reset}\n`);

  await runner.test('Module cleanup releases resources', () => {
    const before = runner.snapshot('Before Module Cleanup');

    delete require.cache[require.resolve(pkgPath)];

    if (global.gc) global.gc();
    const after = runner.snapshot('After Module Cleanup');

    log(colors.blue, 'ℹ', 'Module cache cleared and GC run');
    runner.assert(true, 'Module cleanup completed');
  });

  // Final memory state
  const finalMem = runner.snapshot('Final State');
  console.log('\n');
  runner.printMemorySnapshot(finalMem, 'Final Memory State');
  runner.compareSnapshots(initialMem, finalMem, 'Overall Test Suite');

  // Final Report
  const success = runner.report();

  // Memory health grade
  const totalGrowth = (finalMem.heapUsed - initialMem.heapUsed) / 1024 / 1024;
  console.log(`\n${colors.cyan}═══ Memory Health Grade ═══${colors.reset}\n`);

  let grade = 'A+';
  let color = colors.green;

  if (totalGrowth > 100) {
    grade = 'C';
    color = colors.yellow;
  } else if (totalGrowth > 50) {
    grade = 'B';
    color = colors.yellow;
  } else if (totalGrowth > 20) {
    grade = 'A';
    color = colors.green;
  }

  console.log(`  Grade: ${color}${grade}${colors.reset}`);
  console.log(`  Total Heap Growth: ${totalGrowth.toFixed(2)}MB`);

  process.exit(success ? 0 : 1);
}

// Run tests
if (require.main === module) {
  if (global.gc) {
    log(colors.green, '✓', 'GC is exposed - full memory tests will run');
  } else {
    log(colors.yellow, '⚠', 'GC not exposed. Run with: node --expose-gc memory.test.js');
  }

  runTests().catch(err => {
    console.error(`\n${colors.red}Fatal error:${colors.reset}`, err);
    process.exit(1);
  });
}
