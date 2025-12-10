#!/usr/bin/env node
/**
 * Batch Query Helper
 * Execute multiple queries efficiently
 */

const { execSync } = require('child_process');

function batchQuery(queries, namespace = 'default') {
  console.log(`Executing ${queries.length} queries...`);

  const results = [];
  const startTime = Date.now();

  queries.forEach((query, idx) => {
    try {
      const result = execSync(
        `npx claude-flow@alpha memory query "${query}" --namespace ${namespace} --reasoningbank`,
        { encoding: 'utf8', stdio: 'pipe' }
      );
      results.push({ query, result, success: true });
      process.stdout.write(`\rProgress: ${idx + 1}/${queries.length}`);
    } catch (error) {
      results.push({ query, error: error.message, success: false });
    }
  });

  const duration = Date.now() - startTime;
  const avgTime = duration / queries.length;

  console.log(`\n\n✅ Completed ${results.length} queries`);
  console.log(`⏱️  Average time: ${avgTime.toFixed(2)}ms per query`);
  console.log(`🚀 Throughput: ${(queries.length / (duration / 1000)).toFixed(2)} queries/sec`);

  return results;
}

// CLI usage
if (require.main === module) {
  const queries = process.argv[2] ? JSON.parse(process.argv[2]) : [];
  const namespace = process.argv[3] || 'default';

  if (queries.length === 0) {
    console.log('Usage: node batch-query.js '["query1","query2"]' [namespace]');
    process.exit(1);
  }

  batchQuery(queries, namespace);
}

module.exports = { batchQuery };
