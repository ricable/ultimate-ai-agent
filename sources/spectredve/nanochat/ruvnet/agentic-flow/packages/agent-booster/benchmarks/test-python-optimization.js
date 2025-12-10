#!/usr/bin/env node
/**
 * Python Support Optimization Test
 *
 * Tests various Python code transformations to optimize the parser
 */

const AgentBooster = require('../dist/index.js').default;

console.log('\n🐍 Python Support Optimization Test\n');

const pythonTests = [
  {
    name: 'Add type hints to function',
    code: `def add(x, y):
    return x + y`,
    edit: `def add(x: int, y: int) -> int:
    return x + y`,
  },
  {
    name: 'Add docstring to function',
    code: `def calculate(a, b):
    return a * b`,
    edit: `def calculate(a, b):
    """Multiply two numbers"""
    return a * b`,
  },
  {
    name: 'Add class method',
    code: `class Calculator:
    def __init__(self):
        self.result = 0`,
    edit: `class Calculator:
    def __init__(self):
        self.result = 0

    def add(self, x):
        self.result += x`,
  },
  {
    name: 'Convert to async function',
    code: `def fetch_data(url):
    response = requests.get(url)
    return response.json()`,
    edit: `async def fetch_data(url):
    response = await requests.get(url)
    return response.json()`,
  },
  {
    name: 'Add error handling',
    code: `def divide(a, b):
    return a / b`,
    edit: `def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None`,
  },
  {
    name: 'Add decorator',
    code: `def process_data(data):
    return data.strip()`,
    edit: `@staticmethod
def process_data(data):
    return data.strip()`,
  },
  {
    name: 'Refactor list comprehension',
    code: `result = []
for i in range(10):
    result.append(i * 2)`,
    edit: `result = [i * 2 for i in range(10)]`,
  },
  {
    name: 'Add property decorator',
    code: `class Person:
    def __init__(self, name):
        self._name = name`,
    edit: `class Person:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name`,
  },
];

async function runPythonTests() {
  const booster = new AgentBooster({ confidenceThreshold: 0.4 }); // Lower threshold for Python
  const results = [];

  console.log('┌────────────────────────────────────────┬───────────┬──────────────┬──────────────┐');
  console.log('│ Test                                   │ Latency   │ Confidence   │ Strategy     │');
  console.log('├────────────────────────────────────────┼───────────┼──────────────┼──────────────┤');

  for (const test of pythonTests) {
    const startTime = Date.now();

    try {
      const result = await booster.apply({
        code: test.code,
        edit: test.edit,
        language: 'python',
      });

      const latency = Date.now() - startTime;

      console.log(
        `│ ${test.name.padEnd(38)} │ ${(latency + 'ms').padStart(9)} │ ${((result.confidence * 100).toFixed(1) + '%').padStart(12)} │ ${result.strategy.padEnd(12)} │`
      );

      results.push({
        name: test.name,
        success: result.success,
        confidence: result.confidence,
        latency: latency,
        strategy: result.strategy,
        output: result.output,
      });
    } catch (error) {
      console.log(
        `│ ${test.name.padEnd(38)} │ ${('ERROR').padStart(9)} │ ${('0%').padStart(12)} │ ${'failed'.padEnd(12)} │`
      );

      results.push({
        name: test.name,
        success: false,
        confidence: 0,
        latency: 0,
        strategy: 'failed',
        error: error.message,
      });
    }
  }

  console.log('└────────────────────────────────────────┴───────────┴──────────────┴──────────────┘\n');

  // Summary
  const successful = results.filter(r => r.success).length;
  const failed = results.filter(r => !r.success).length;
  const avgLatency = results.reduce((sum, r) => sum + r.latency, 0) / results.length;
  const avgConfidence = results.filter(r => r.success).reduce((sum, r) => sum + r.confidence, 0) / successful || 0;

  console.log('📊 Python Optimization Summary\n');
  console.log(`Total Tests:         ${results.length}`);
  console.log(`Successful:          ${successful} (${((successful/results.length)*100).toFixed(0)}%)`);
  console.log(`Failed:              ${failed} (${((failed/results.length)*100).toFixed(0)}%)`);
  console.log(`Average Latency:     ${avgLatency.toFixed(0)}ms`);
  console.log(`Average Confidence:  ${(avgConfidence * 100).toFixed(1)}%\n`);

  // Strategy breakdown
  const strategies = {};
  results.filter(r => r.success).forEach(r => {
    strategies[r.strategy] = (strategies[r.strategy] || 0) + 1;
  });

  console.log('📂 Strategy Distribution\n');
  Object.entries(strategies).forEach(([strategy, count]) => {
    console.log(`  ${strategy}: ${count} (${((count/successful)*100).toFixed(0)}%)`);
  });
  console.log();

  // Save results
  const fs = require('fs');
  const path = require('path');
  const outputPath = path.join(__dirname, 'results/python-optimization-results.json');
  fs.writeFileSync(outputPath, JSON.stringify({ results, timestamp: new Date().toISOString() }, null, 2));
  console.log(`📄 Results saved to: ${outputPath}\n`);

  return results;
}

runPythonTests().catch(err => {
  console.error('❌ Test failed:', err);
  process.exit(1);
});
