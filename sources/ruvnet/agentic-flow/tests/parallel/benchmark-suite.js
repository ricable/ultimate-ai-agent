#!/usr/bin/env node
/**
 * Comprehensive Benchmark Suite for Parallel Execution
 * Tests all topologies and measures performance improvements
 */

const { meshSwarmTest } = require('./mesh-swarm-test');
const { hierarchicalSwarmTest } = require('./hierarchical-swarm-test');
const { ringSwarmTest } = require('./ring-swarm-test');
const fs = require('fs');
const path = require('path');

const ITERATIONS = parseInt(process.env.ITERATIONS || '3');
const BENCHMARK_MODE = process.env.BENCHMARK_MODE === 'true';

async function runBenchmark(name, testFn, iterations = ITERATIONS) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`Running: ${name} (${iterations} iterations)`);
  console.log('='.repeat(70));

  const results = [];
  const times = [];

  for (let i = 0; i < iterations; i++) {
    console.log(`\n--- Iteration ${i + 1}/${iterations} ---`);
    const startTime = Date.now();

    try {
      const result = await testFn();
      const executionTime = Date.now() - startTime;

      results.push(result);
      times.push(executionTime);

      console.log(`✅ Iteration ${i + 1} completed in ${executionTime}ms`);
    } catch (error) {
      console.error(`❌ Iteration ${i + 1} failed:`, error.message);
      times.push(null);
    }

    // Brief pause between iterations
    if (i < iterations - 1) {
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }

  // Calculate statistics
  const validTimes = times.filter(t => t !== null);
  const avgTime = validTimes.length > 0
    ? validTimes.reduce((a, b) => a + b, 0) / validTimes.length
    : 0;
  const minTime = validTimes.length > 0 ? Math.min(...validTimes) : 0;
  const maxTime = validTimes.length > 0 ? Math.max(...validTimes) : 0;
  const successRate = (validTimes.length / iterations) * 100;

  return {
    name,
    iterations,
    results,
    times,
    statistics: {
      avgTimeMs: Math.round(avgTime),
      minTimeMs: minTime,
      maxTimeMs: maxTime,
      successRate,
      validResults: validTimes.length
    }
  };
}

async function compareTopologies() {
  console.log('\n🏁 Starting Comprehensive Benchmark Suite');
  console.log('='.repeat(70));
  console.log(`Iterations per test: ${ITERATIONS}`);
  console.log(`Benchmark mode: ${BENCHMARK_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log('='.repeat(70));

  const benchmarkResults = {
    timestamp: new Date().toISOString(),
    config: {
      iterations: ITERATIONS,
      benchmarkMode: BENCHMARK_MODE
    },
    tests: {}
  };

  try {
    // Test 1: Mesh Topology
    const meshResults = await runBenchmark('Mesh Topology', meshSwarmTest);
    benchmarkResults.tests.mesh = meshResults;

    // Test 2: Hierarchical Topology
    const hierarchicalResults = await runBenchmark('Hierarchical Topology', hierarchicalSwarmTest);
    benchmarkResults.tests.hierarchical = hierarchicalResults;

    // Test 3: Ring Topology
    const ringResults = await runBenchmark('Ring Topology', ringSwarmTest);
    benchmarkResults.tests.ring = ringResults;

    // Generate comparison report
    console.log('\n\n📊 BENCHMARK COMPARISON REPORT');
    console.log('='.repeat(70));

    const topologies = ['mesh', 'hierarchical', 'ring'];
    const comparisonTable = topologies.map(topology => {
      const stats = benchmarkResults.tests[topology].statistics;
      return {
        topology: topology.charAt(0).toUpperCase() + topology.slice(1),
        avgTime: `${stats.avgTimeMs}ms`,
        minTime: `${stats.minTimeMs}ms`,
        maxTime: `${stats.maxTimeMs}ms`,
        successRate: `${stats.successRate.toFixed(1)}%`,
        validResults: `${stats.validResults}/${ITERATIONS}`
      };
    });

    console.table(comparisonTable);

    // Calculate speedup metrics
    const baselineTime = benchmarkResults.tests.mesh.statistics.avgTimeMs;
    console.log('\n⚡ Speedup Analysis (vs Mesh baseline)');
    console.log('─'.repeat(70));

    topologies.forEach(topology => {
      const avgTime = benchmarkResults.tests[topology].statistics.avgTimeMs;
      const speedup = baselineTime / avgTime;
      const faster = ((1 - (avgTime / baselineTime)) * 100).toFixed(1);

      console.log(`${topology.padEnd(15)}: ${speedup.toFixed(2)}x speedup (${faster}% ${faster >= 0 ? 'faster' : 'slower'})`);
    });

    // Performance grade
    console.log('\n🏆 Performance Grades');
    console.log('─'.repeat(70));

    topologies.forEach(topology => {
      const stats = benchmarkResults.tests[topology].statistics;
      const grade = gradePerformance(stats.successRate, stats.avgTimeMs, baselineTime);
      console.log(`${topology.padEnd(15)}: ${grade.grade} (${grade.description})`);
    });

    // Recommendations
    console.log('\n💡 Recommendations');
    console.log('─'.repeat(70));

    const recommendations = generateRecommendations(benchmarkResults);
    recommendations.forEach((rec, i) => {
      console.log(`${i + 1}. ${rec}`);
    });

    // Save results to file
    const resultsDir = path.join(__dirname, '../../benchmark-results');
    if (!fs.existsSync(resultsDir)) {
      fs.mkdirSync(resultsDir, { recursive: true });
    }

    const resultsFile = path.join(resultsDir, `benchmark-${Date.now()}.json`);
    fs.writeFileSync(resultsFile, JSON.stringify(benchmarkResults, null, 2));
    console.log(`\n📝 Results saved to: ${resultsFile}`);

    // Generate markdown report
    const markdownReport = generateMarkdownReport(benchmarkResults, comparisonTable);
    const reportFile = path.join(resultsDir, `benchmark-${Date.now()}.md`);
    fs.writeFileSync(reportFile, markdownReport);
    console.log(`📄 Markdown report saved to: ${reportFile}`);

    console.log('\n✅ Benchmark suite completed successfully\n');
    return benchmarkResults;

  } catch (error) {
    console.error('\n❌ Benchmark suite failed:', error);
    throw error;
  }
}

function gradePerformance(successRate, avgTime, baselineTime) {
  const speedup = baselineTime / avgTime;

  if (successRate >= 90 && speedup >= 0.9) {
    return { grade: 'A', description: 'Excellent performance' };
  } else if (successRate >= 75 && speedup >= 0.75) {
    return { grade: 'B', description: 'Good performance' };
  } else if (successRate >= 60 && speedup >= 0.6) {
    return { grade: 'C', description: 'Acceptable performance' };
  } else if (successRate >= 50) {
    return { grade: 'D', description: 'Poor performance' };
  } else {
    return { grade: 'F', description: 'Failed performance' };
  }
}

function generateRecommendations(benchmarkResults) {
  const recommendations = [];
  const tests = benchmarkResults.tests;

  // Find best performing topology
  const topologies = Object.keys(tests);
  const bestTopology = topologies.reduce((best, current) => {
    const bestTime = tests[best].statistics.avgTimeMs;
    const currentTime = tests[current].statistics.avgTimeMs;
    return currentTime < bestTime ? current : best;
  });

  recommendations.push(`Best topology for this workload: ${bestTopology} (${tests[bestTopology].statistics.avgTimeMs}ms avg)`);

  // Check success rates
  topologies.forEach(topology => {
    const successRate = tests[topology].statistics.successRate;
    if (successRate < 80) {
      recommendations.push(`${topology} has low success rate (${successRate.toFixed(1)}%) - investigate failures`);
    }
  });

  // Parallel execution benefits
  const meshTime = tests.mesh.statistics.avgTimeMs;
  const hierarchicalTime = tests.hierarchical.statistics.avgTimeMs;
  if (hierarchicalTime < meshTime * 0.7) {
    recommendations.push(`Hierarchical shows strong parallel benefits (${((1 - hierarchicalTime/meshTime) * 100).toFixed(1)}% faster) - good for delegated tasks`);
  }

  return recommendations;
}

function generateMarkdownReport(benchmarkResults, comparisonTable) {
  const timestamp = new Date(benchmarkResults.timestamp).toLocaleString();

  return `# Parallel Execution Benchmark Report

**Generated:** ${timestamp}
**Iterations:** ${benchmarkResults.config.iterations}
**Mode:** ${benchmarkResults.config.benchmarkMode ? 'Benchmark' : 'Standard'}

## Topology Comparison

| Topology | Avg Time | Min Time | Max Time | Success Rate | Valid Results |
|----------|----------|----------|----------|--------------|---------------|
${comparisonTable.map(row =>
  `| ${row.topology} | ${row.avgTime} | ${row.minTime} | ${row.maxTime} | ${row.successRate} | ${row.validResults} |`
).join('\n')}

## Speedup Analysis

${Object.keys(benchmarkResults.tests).map(topology => {
  const avgTime = benchmarkResults.tests[topology].statistics.avgTimeMs;
  const baselineTime = benchmarkResults.tests.mesh.statistics.avgTimeMs;
  const speedup = (baselineTime / avgTime).toFixed(2);
  const faster = ((1 - (avgTime / baselineTime)) * 100).toFixed(1);
  return `- **${topology}**: ${speedup}x speedup (${faster}% ${faster >= 0 ? 'faster' : 'slower'} than baseline)`;
}).join('\n')}

## Performance Grades

${Object.keys(benchmarkResults.tests).map(topology => {
  const stats = benchmarkResults.tests[topology].statistics;
  const baselineTime = benchmarkResults.tests.mesh.statistics.avgTimeMs;
  const grade = gradePerformance(stats.successRate, stats.avgTimeMs, baselineTime);
  return `- **${topology}**: ${grade.grade} - ${grade.description}`;
}).join('\n')}

## Recommendations

${generateRecommendations(benchmarkResults).map((rec, i) => `${i + 1}. ${rec}`).join('\n')}

## Raw Results

\`\`\`json
${JSON.stringify(benchmarkResults, null, 2)}
\`\`\`
`;
}

// Run benchmark suite
if (require.main === module) {
  compareTopologies()
    .then(results => {
      console.log('✅ All benchmarks completed');
      process.exit(0);
    })
    .catch(error => {
      console.error('❌ Benchmark failed:', error);
      process.exit(1);
    });
}

module.exports = { compareTopologies, runBenchmark };
