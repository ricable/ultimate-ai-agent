#!/usr/bin/env node

/**
 * Analyze Morph LLM benchmark results and generate insights
 */

const fs = require('fs');
const path = require('path');

const resultsPath = path.join(__dirname, 'results', 'morph-baseline-results.json');
const results = JSON.parse(fs.readFileSync(resultsPath, 'utf8'));

console.log('\n' + '='.repeat(70));
console.log('MORPH LLM BENCHMARK ANALYSIS');
console.log('='.repeat(70));

// Overall Statistics
console.log('\n📊 OVERALL PERFORMANCE');
console.log('-'.repeat(70));
console.log(`Success Rate: ${results.successful_tests}/${results.total_tests} (${(results.successful_tests/results.total_tests*100).toFixed(1)}%)`);
console.log(`Average Latency: ${results.performance.avg_latency_ms}ms`);
console.log(`Total Time: ${(results.performance.total_time_ms/1000).toFixed(1)}s`);
console.log(`Total API Time: ${(results.performance.total_api_time_ms/1000).toFixed(1)}s`);

// Token Usage
console.log('\n💰 RESOURCE USAGE');
console.log('-'.repeat(70));
console.log(`Applies: ${results.usage.applies_used}/500 (${(results.usage.applies_used/500*100).toFixed(1)}%)`);
console.log(`Input Tokens: ${results.usage.input_tokens}/300000 (${(results.usage.input_tokens/300000*100).toFixed(2)}%)`);
console.log(`Output Tokens: ${results.usage.output_tokens}/100000 (${(results.usage.output_tokens/100000*100).toFixed(2)}%)`);
console.log(`Avg Input/Test: ${(results.usage.input_tokens/results.total_tests).toFixed(1)} tokens`);
console.log(`Avg Output/Test: ${(results.usage.output_tokens/results.total_tests).toFixed(1)} tokens`);

// Category Analysis
console.log('\n📂 CATEGORY BREAKDOWN');
console.log('-'.repeat(70));

const categories = {};
results.results.forEach(test => {
  if (!categories[test.category]) {
    categories[test.category] = {
      tests: [],
      totalLatency: 0,
      totalInput: 0,
      totalOutput: 0
    };
  }
  categories[test.category].tests.push(test);
  categories[test.category].totalLatency += test.latency;
  categories[test.category].totalInput += test.tokens.input;
  categories[test.category].totalOutput += test.tokens.output;
});

Object.entries(categories)
  .sort((a, b) => a[1].totalLatency/a[1].tests.length - b[1].totalLatency/b[1].tests.length)
  .forEach(([category, data]) => {
    const avgLatency = (data.totalLatency / data.tests.length).toFixed(0);
    const avgInput = (data.totalInput / data.tests.length).toFixed(0);
    const avgOutput = (data.totalOutput / data.tests.length).toFixed(0);
    console.log(`${category.padEnd(25)} | ${data.tests.length} tests | ${avgLatency}ms avg | ${avgInput}+${avgOutput} tokens`);
  });

// Fastest/Slowest Tests
console.log('\n⚡ PERFORMANCE EXTREMES');
console.log('-'.repeat(70));

const sortedByLatency = [...results.results].sort((a, b) => a.latency - b.latency);
console.log('Fastest Tests:');
sortedByLatency.slice(0, 3).forEach((test, i) => {
  console.log(`  ${i+1}. ${test.description} (${test.latency}ms)`);
});

console.log('\nSlowest Tests:');
sortedByLatency.slice(-3).reverse().forEach((test, i) => {
  console.log(`  ${i+1}. ${test.description} (${test.latency}ms)`);
});

// Token Efficiency
console.log('\n🔤 TOKEN EFFICIENCY');
console.log('-'.repeat(70));

const sortedByTokens = [...results.results].sort((a, b) =>
  (a.tokens.input + a.tokens.output) - (b.tokens.input + b.tokens.output)
);

console.log('Most Token-Efficient:');
sortedByTokens.slice(0, 3).forEach((test, i) => {
  const total = test.tokens.input + test.tokens.output;
  console.log(`  ${i+1}. ${test.description} (${total} tokens)`);
});

console.log('\nMost Token-Intensive:');
sortedByTokens.slice(-3).reverse().forEach((test, i) => {
  const total = test.tokens.input + test.tokens.output;
  console.log(`  ${i+1}. ${test.description} (${total} tokens)`);
});

// Projections
console.log('\n📈 MONTHLY PROJECTIONS');
console.log('-'.repeat(70));

const testsPerMonth = Math.floor(500 / (results.usage.applies_used / results.total_tests));
const inputTokensPerMonth = (results.usage.input_tokens / results.total_tests) * testsPerMonth;
const outputTokensPerMonth = (results.usage.output_tokens / results.total_tests) * testsPerMonth;

console.log(`Maximum Tests/Month: ${testsPerMonth} (limited by applies)`);
console.log(`Projected Input Tokens: ${inputTokensPerMonth.toFixed(0)}/300000 (${(inputTokensPerMonth/300000*100).toFixed(1)}%)`);
console.log(`Projected Output Tokens: ${outputTokensPerMonth.toFixed(0)}/100000 (${(outputTokensPerMonth/100000*100).toFixed(1)}%)`);
console.log(`Bottleneck: Applies limit`);

// Cost Analysis
console.log('\n💵 COST ANALYSIS (MORPH FREE PLAN)');
console.log('-'.repeat(70));
console.log(`Current usage: ${results.usage.applies_used} applies`);
console.log(`Remaining budget: ${results.usage.remaining_budget.applies} applies`);
console.log(`Tests remaining: ${Math.floor(results.usage.remaining_budget.applies / (results.usage.applies_used / results.total_tests))}`);
console.log(`Days until depletion (at 10 tests/day): ${Math.floor(results.usage.remaining_budget.applies / 10)} days`);

// Agent Booster Opportunity
console.log('\n🚀 AGENT BOOSTER OPPORTUNITY');
console.log('-'.repeat(70));

const networkLatency = 200; // estimated average network latency
const localProcessingTime = results.performance.avg_latency_ms - networkLatency;
const estimatedAgentBoosterLatency = Math.max(50, localProcessingTime * 0.5);
const speedupRatio = results.performance.avg_latency_ms / estimatedAgentBoosterLatency;

console.log(`Current Morph Latency: ${results.performance.avg_latency_ms}ms`);
console.log(`  - Network latency: ~${networkLatency}ms`);
console.log(`  - Processing time: ~${localProcessingTime}ms`);
console.log(`\nEstimated Agent Booster Latency: ${estimatedAgentBoosterLatency}ms (local)`);
console.log(`Potential Speedup: ${speedupRatio.toFixed(1)}x faster`);
console.log(`Latency Reduction: ${(results.performance.avg_latency_ms - estimatedAgentBoosterLatency).toFixed(0)}ms saved per test`);
console.log(`\nCost Savings:`);
console.log(`  - Morph: Limited to 500 applies/month`);
console.log(`  - Agent Booster: Unlimited (zero marginal cost)`);
console.log(`  - Monthly Savings: Unlimited vs 500 test cap`);

console.log('\n' + '='.repeat(70) + '\n');
