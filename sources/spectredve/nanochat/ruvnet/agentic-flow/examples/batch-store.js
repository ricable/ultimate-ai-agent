#!/usr/bin/env node
/**
 * Batch Storage Helper
 * Stores multiple patterns efficiently
 */

const { execSync } = require('child_process');

const patterns = process.argv[2] ? JSON.parse(process.argv[2]) : [];
const namespace = process.argv[3] || 'default';

console.log(`Batch storing ${patterns.length} patterns...`);

const startTime = Date.now();
let success = 0;
let failed = 0;

patterns.forEach((pattern, idx) => {
  try {
    execSync(
      `npx claude-flow@alpha memory store "${pattern.key}" "${pattern.value}" --namespace ${namespace} --reasoningbank`,
      { stdio: 'pipe' }
    );
    success++;
    process.stdout.write(`\rProgress: ${idx + 1}/${patterns.length}`);
  } catch (error) {
    failed++;
  }
});

const duration = Date.now() - startTime;
const avgTime = duration / patterns.length;

console.log(`\n\n✅ Completed: ${success} stored, ${failed} failed`);
console.log(`⏱️  Total time: ${duration}ms (avg: ${avgTime.toFixed(2)}ms per entry)`);
console.log(`🚀 Throughput: ${(patterns.length / (duration / 1000)).toFixed(2)} entries/sec`);
