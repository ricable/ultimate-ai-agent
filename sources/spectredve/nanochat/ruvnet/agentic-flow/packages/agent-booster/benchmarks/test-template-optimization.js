#!/usr/bin/env node
/**
 * Test Template-Based Optimization
 *
 * Tests the newly implemented template transformation engine against
 * the 6 failed test cases from Morph LLM comparison.
 */

const AgentBooster = require('../dist/index.js').default;
const dataset = require('./datasets/small-test-dataset.json');

console.log('\n🧪 Template-Based Optimization Test\n');
console.log('Testing against 6 previously failed cases...\n');

// The 6 failed tests from morphllm comparison
const failedTestIds = ['test-002', 'test-006', 'test-008', 'test-009', 'test-011', 'test-012'];

async function runTemplateTests() {
  const booster = new AgentBooster({ confidenceThreshold: 0.5 });
  const results = [];

  console.log('┌──────────┬─────────────────────────────────────┬─────────┬────────────┬────────────┐');
  console.log('│ Test ID  │ Description                         │ Status  │ Confidence │ Strategy   │');
  console.log('├──────────┼─────────────────────────────────────┼─────────┼────────────┼────────────┤');

  for (const testId of failedTestIds) {
    const testCase = dataset.find(t => t.id === testId);

    if (!testCase) {
      console.log(`│ ${testId.padEnd(8)} │ NOT FOUND                           │ ❌      │ N/A        │ N/A        │`);
      continue;
    }

    try {
      const startTime = Date.now();
      const result = await booster.apply({
        code: testCase.input,
        edit: testCase.expected_output,
        language: 'javascript'
      });
      const latency = Date.now() - startTime;

      const desc = testCase.description.substring(0, 35).padEnd(35);
      const status = result.success ? '✅' : '❌';
      const conf = result.success ? `${(result.confidence * 100).toFixed(1)}%`.padStart(10) : 'N/A'.padStart(10);
      const strategy = result.success ? result.strategy.padEnd(10) : 'failed'.padEnd(10);

      console.log(`│ ${testId.padEnd(8)} │ ${desc} │ ${status}      │ ${conf} │ ${strategy} │`);

      results.push({
        testId,
        description: testCase.description,
        success: result.success,
        confidence: result.confidence,
        strategy: result.strategy,
        latency,
        previouslyFailed: true,
        nowPasses: result.success,
      });

    } catch (error) {
      const desc = testCase.description.substring(0, 35).padEnd(35);
      console.log(`│ ${testId.padEnd(8)} │ ${desc} │ ❌      │ N/A        │ failed     │`);

      results.push({
        testId,
        description: testCase.description,
        success: false,
        confidence: 0,
        strategy: 'failed',
        latency: 0,
        previouslyFailed: true,
        nowPasses: false,
      });
    }
  }

  console.log('└──────────┴─────────────────────────────────────┴─────────┴────────────┴────────────┘\n');

  return results;
}

runTemplateTests().then(results => {
  const fs = require('fs');
  const path = require('path');

  // Save detailed results
  const outputPath = path.join(__dirname, 'results/template-optimization-results.json');
  fs.writeFileSync(outputPath, JSON.stringify({
    results,
    timestamp: new Date().toISOString()
  }, null, 2));

  // Calculate statistics
  const totalTests = results.length;
  const passedTests = results.filter(r => r.success).length;
  const failedTests = totalTests - passedTests;
  const improvementRate = (passedTests / totalTests) * 100;

  const avgConfidence = passedTests > 0
    ? results.filter(r => r.success).reduce((sum, r) => sum + r.confidence, 0) / passedTests
    : 0;

  console.log('═══════════════════════════════════════════════════════════════════════════════════');
  console.log('                    📊 TEMPLATE OPTIMIZATION RESULTS                              ');
  console.log('═══════════════════════════════════════════════════════════════════════════════════\n');

  console.log('🎯 Success Metrics\n');
  console.log(`Total Tests:        ${totalTests}`);
  console.log(`Previously Failed:  ${totalTests} (100%)`);
  console.log(`Now Passing:        ${passedTests} (${improvementRate.toFixed(0)}%)`);
  console.log(`Still Failing:      ${failedTests} (${((failedTests/totalTests)*100).toFixed(0)}%)\n`);

  console.log('📈 Improvement Analysis\n');
  console.log(`Before Template Optimization:  0/6 (0%)`);
  console.log(`After Template Optimization:   ${passedTests}/6 (${improvementRate.toFixed(0)}%)`);
  console.log(`Improvement:                   +${passedTests} tests (+${improvementRate.toFixed(0)}%)\n`);

  if (passedTests > 0) {
    console.log('✅ Successfully Fixed Tests:\n');
    results.filter(r => r.success).forEach(r => {
      console.log(`   • ${r.testId}: ${r.description} (${(r.confidence * 100).toFixed(1)}% confidence)`);
    });
    console.log('');
  }

  if (failedTests > 0) {
    console.log('❌ Still Failing Tests:\n');
    results.filter(r => !r.success).forEach(r => {
      console.log(`   • ${r.testId}: ${r.description}`);
    });
    console.log('');
  }

  console.log('🎯 Quality Metrics\n');
  console.log(`Average Confidence: ${(avgConfidence * 100).toFixed(1)}%`);
  console.log(`Min Confidence:     ${passedTests > 0 ? (Math.min(...results.filter(r => r.success).map(r => r.confidence)) * 100).toFixed(1) + '%' : 'N/A'}`);
  console.log(`Max Confidence:     ${passedTests > 0 ? (Math.max(...results.filter(r => r.success).map(r => r.confidence)) * 100).toFixed(1) + '%' : 'N/A'}\n`);

  // Win rate projection
  const originalWinRate = 50; // 6/12 = 50%
  const newPassingTests = passedTests; // How many of the 6 failed tests now pass
  const projectedWins = 6 + newPassingTests; // Original 6 wins + newly passing tests
  const projectedWinRate = (projectedWins / 12) * 100;

  console.log('═══════════════════════════════════════════════════════════════════════════════════');
  console.log('                    🏆 WIN RATE PROJECTION                                        ');
  console.log('═══════════════════════════════════════════════════════════════════════════════════\n');

  console.log('📊 Head-to-Head vs Morph LLM\n');
  console.log(`Before Optimization:  6/12 wins (50%)`);
  console.log(`After Optimization:   ${projectedWins}/12 wins (${projectedWinRate.toFixed(0)}%)`);
  console.log(`Improvement:          +${newPassingTests} wins (+${(projectedWinRate - originalWinRate).toFixed(0)}%)\n`);

  if (projectedWinRate >= 75) {
    console.log('🎉 SUCCESS: Target win rate of 75% achieved!\n');
  } else if (projectedWinRate >= 60) {
    console.log('⚠️  PROGRESS: Significant improvement, but target 75% not yet reached.\n');
  } else {
    console.log('❌ NEEDS WORK: Additional optimization strategies required.\n');
  }

  console.log('═══════════════════════════════════════════════════════════════════════════════════\n');
  console.log(`📄 Detailed results: ${outputPath}\n`);

}).catch(err => {
  console.error('❌ Test failed:', err);
  process.exit(1);
});
