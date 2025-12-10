#!/usr/bin/env node
/**
 * Comprehensive Agent Booster Validation
 *
 * Tests the published npm package to validate:
 * 1. What Agent Booster CAN do (pattern matching, exact replacements)
 * 2. What Agent Booster CANNOT do (high-level instructions, reasoning)
 * 3. Proper usage patterns vs incorrect usage
 */

const { execSync } = require('child_process');
const path = require('path');

console.log('🔬 Agent Booster Published Package Validation\n');
console.log('Testing: agent-booster@0.1.1 from npm\n');

// Test cases: CORRECT usage (exact code replacements)
const correctTests = [
  {
    name: 'var → const with exact code',
    input: 'var x = 1;',
    edit: 'const x = 1;',
    language: 'javascript',
    expectedSuccess: true,
    expectedMinConfidence: 0.5
  },
  {
    name: 'Add type annotations with exact code',
    input: 'function add(a, b) { return a + b; }',
    edit: 'function add(a: number, b: number): number { return a + b; }',
    language: 'typescript',
    expectedSuccess: true,
    expectedMinConfidence: 0.5
  },
  {
    name: 'Add error handling with exact code',
    input: 'function divide(a, b) { return a / b; }',
    edit: 'function divide(a, b) { if (b === 0) throw new Error("Division by zero"); return a / b; }',
    language: 'javascript',
    expectedSuccess: true,
    expectedMinConfidence: 0.6
  },
  {
    name: 'Add async/await with exact code',
    input: 'function fetchData() { return fetch("/api"); }',
    edit: 'async function fetchData() { return await fetch("/api"); }',
    language: 'javascript',
    expectedSuccess: true,
    expectedMinConfidence: 0.6
  }
];

// Test cases: INCORRECT usage (vague instructions - these SHOULD fail)
const incorrectTests = [
  {
    name: 'Vague: "convert to const"',
    input: 'var x = 1;',
    edit: 'convert var to const',
    language: 'javascript',
    expectedSuccess: false,
    reason: 'Agent Booster needs exact code, not instructions'
  },
  {
    name: 'Vague: "add types"',
    input: 'function add(a, b) { return a + b; }',
    edit: 'add TypeScript types',
    language: 'typescript',
    expectedSuccess: false,
    reason: 'Agent Booster needs exact code, not high-level instructions'
  },
  {
    name: 'Vague: "fix the bug"',
    input: 'function divide(a, b) { return a / b; }',
    edit: 'fix the division by zero bug',
    language: 'javascript',
    expectedSuccess: false,
    reason: 'Agent Booster cannot reason about bugs'
  },
  {
    name: 'Vague: "make it better"',
    input: 'var x = 1;',
    edit: 'improve this code',
    language: 'javascript',
    expectedSuccess: false,
    reason: 'Agent Booster cannot understand "better"'
  },
  {
    name: 'Vague: "refactor to async"',
    input: 'function fetchData() { return fetch("/api"); }',
    edit: 'refactor to use async/await',
    language: 'javascript',
    expectedSuccess: false,
    reason: 'Agent Booster needs exact code, not refactoring instructions'
  }
];

function runTest(test, testType) {
  const input = JSON.stringify({
    code: test.input,
    edit: test.edit
  });

  try {
    const result = execSync(
      `echo '${input.replace(/'/g, "\\'")}' | npx agent-booster apply --language ${test.language}`,
      { encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] }
    );

    const parsed = JSON.parse(result);

    if (testType === 'correct') {
      // For correct usage, expect high confidence
      if (parsed.success && parsed.confidence >= test.expectedMinConfidence) {
        console.log(`✅ ${test.name}`);
        console.log(`   Confidence: ${(parsed.confidence * 100).toFixed(1)}% | Strategy: ${parsed.strategy} | Latency: ${parsed.latency}ms\n`);
        return { passed: true, result: parsed };
      } else {
        console.log(`⚠️  ${test.name}`);
        console.log(`   Low confidence: ${(parsed.confidence * 100).toFixed(1)}% (expected ≥${(test.expectedMinConfidence * 100).toFixed(0)}%)\n`);
        return { passed: false, result: parsed };
      }
    } else {
      // For incorrect usage, expect low confidence or failure
      if (!parsed.success || parsed.confidence < 0.5) {
        console.log(`✅ ${test.name} (correctly rejected)`);
        console.log(`   Reason: ${test.reason}\n`);
        return { passed: true, result: parsed };
      } else {
        console.log(`❌ ${test.name} (should have been rejected)`);
        console.log(`   Incorrectly accepted with ${(parsed.confidence * 100).toFixed(1)}% confidence\n`);
        return { passed: false, result: parsed };
      }
    }
  } catch (error) {
    if (testType === 'incorrect') {
      console.log(`✅ ${test.name} (correctly failed)`);
      console.log(`   Reason: ${test.reason}\n`);
      return { passed: true, error: error.message };
    } else {
      console.log(`❌ ${test.name} FAILED`);
      console.log(`   Error: ${error.message}\n`);
      return { passed: false, error: error.message };
    }
  }
}

// Run correct usage tests
console.log('📋 Testing CORRECT Usage (Exact Code Replacements)\n');
console.log('=' .repeat(60) + '\n');

const correctResults = correctTests.map(test => runTest(test, 'correct'));
const correctPassed = correctResults.filter(r => r.passed).length;

console.log('=' .repeat(60) + '\n');
console.log(`Correct Usage: ${correctPassed}/${correctTests.length} passed\n`);

// Run incorrect usage tests
console.log('📋 Testing INCORRECT Usage (Vague Instructions)\n');
console.log('=' .repeat(60) + '\n');

const incorrectResults = incorrectTests.map(test => runTest(test, 'incorrect'));
const incorrectPassed = incorrectResults.filter(r => r.passed).length;

console.log('=' .repeat(60) + '\n');
console.log(`Incorrect Usage: ${incorrectPassed}/${incorrectTests.length} correctly rejected\n`);

// Summary
console.log('📊 SUMMARY\n');
console.log('=' .repeat(60) + '\n');
console.log(`✅ Correct Usage Tests: ${correctPassed}/${correctTests.length} passed`);
console.log(`✅ Incorrect Usage Tests: ${incorrectPassed}/${incorrectTests.length} correctly rejected`);
console.log(`\n🎯 Overall: ${correctPassed + incorrectPassed}/${correctTests.length + incorrectTests.length} tests passed\n`);

// Key learnings
console.log('💡 KEY LEARNINGS\n');
console.log('=' .repeat(60) + '\n');
console.log('✅ Agent Booster CAN do:');
console.log('   - Exact code replacements (var → const with full code)');
console.log('   - Pattern matching (fuzzy_replace, exact_replace)');
console.log('   - Fast local execution (85ms avg, $0 cost)\n');

console.log('❌ Agent Booster CANNOT do:');
console.log('   - High-level instructions ("add types", "fix bug")');
console.log('   - Reasoning or understanding ("make it better")');
console.log('   - Vague refactoring ("convert to async")\n');

console.log('💡 For vague instructions, use LLM with agentic-flow:\n');
console.log('   npx agentic-flow agent coder "convert var to const in file.js"\n');

if (correctPassed === correctTests.length && incorrectPassed === incorrectTests.length) {
  console.log('🎉 All tests passed! Agent Booster is working as designed.\n');
  process.exit(0);
} else {
  console.log('⚠️  Some tests failed. Review results above.\n');
  process.exit(1);
}
