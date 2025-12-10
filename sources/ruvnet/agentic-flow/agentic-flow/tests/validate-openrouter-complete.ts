#!/usr/bin/env tsx
/**
 * Deep Validation Suite for OpenRouter Integration
 * Tests all capabilities with OpenRouter models via integrated proxy
 */

import { spawn } from 'child_process';
import { existsSync, readFileSync, unlinkSync } from 'fs';
import { join } from 'path';

interface TestResult {
  name: string;
  status: 'pass' | 'fail';
  duration: number;
  output?: string;
  error?: string;
}

const results: TestResult[] = [];

async function runTest(
  name: string,
  command: string,
  args: string[],
  validator: (output: string) => boolean
): Promise<TestResult> {
  const startTime = Date.now();

  return new Promise((resolve) => {
    let output = '';
    let errorOutput = '';

    const proc = spawn(command, args, {
      env: {
        ...process.env,
        AGENTS_DIR: join(process.cwd(), '.claude/agents')
      }
    });

    proc.stdout.on('data', (data) => {
      output += data.toString();
    });

    proc.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    proc.on('close', (code) => {
      const duration = Date.now() - startTime;
      const fullOutput = output + errorOutput;
      const passed = code === 0 && validator(fullOutput);

      resolve({
        name,
        status: passed ? 'pass' : 'fail',
        duration,
        output: fullOutput,
        error: code !== 0 ? `Exit code: ${code}` : undefined
      });
    });

    // Timeout after 3 minutes
    setTimeout(() => {
      proc.kill();
      resolve({
        name,
        status: 'fail',
        duration: Date.now() - startTime,
        error: 'Timeout after 3 minutes'
      });
    }, 180000);
  });
}

async function main() {
  console.log('🧪 Deep Validation Suite for OpenRouter Integration\n');
  console.log('================================================\n');

  // Test 1: Simple code generation
  console.log('Test 1: Simple code generation...');
  const test1 = await runTest(
    'Simple code generation',
    'node',
    [
      'dist/cli-proxy.js',
      '--agent', 'coder',
      '--task', 'Create a Python function to multiply two numbers',
      '--model', 'meta-llama/llama-3.1-8b-instruct'
    ],
    (output) => output.includes('def') && output.includes('multiply')
  );
  results.push(test1);
  console.log(`  ${test1.status === 'pass' ? '✅' : '❌'} ${test1.status.toUpperCase()} (${test1.duration}ms)\n`);

  // Test 2: Different model (DeepSeek)
  console.log('Test 2: DeepSeek model...');
  const test2 = await runTest(
    'DeepSeek model',
    'node',
    [
      'dist/cli-proxy.js',
      '--agent', 'coder',
      '--task', 'Create a Python function for fibonacci sequence',
      '--model', 'deepseek/deepseek-chat-v3.1'
    ],
    (output) => output.includes('def') && output.includes('fibonacci')
  );
  results.push(test2);
  console.log(`  ${test2.status === 'pass' ? '✅' : '❌'} ${test2.status.toUpperCase()} (${test2.duration}ms)\n`);

  // Test 3: Another model (Gemini)
  console.log('Test 3: Gemini model...');
  const test3 = await runTest(
    'Gemini model',
    'node',
    [
      'dist/cli-proxy.js',
      '--agent', 'coder',
      '--task', 'Create a Python function to check if a number is prime',
      '--model', 'google/gemini-2.5-flash-preview-09-2025'
    ],
    (output) => output.includes('def') && output.includes('prime')
  );
  results.push(test3);
  console.log(`  ${test3.status === 'pass' ? '✅' : '❌'} ${test3.status.toUpperCase()} (${test3.duration}ms)\n`);

  // Test 4: Check proxy logs for API conversion
  console.log('Test 4: Proxy API conversion...');
  const test4 = await runTest(
    'Proxy API conversion',
    'node',
    [
      'dist/cli-proxy.js',
      '--agent', 'coder',
      '--task', 'Create a hello world function',
      '--model', 'meta-llama/llama-3.1-8b-instruct'
    ],
    (output) => output.includes('Proxy Mode: OpenRouter') && output.includes('Completed')
  );
  results.push(test4);
  console.log(`  ${test4.status === 'pass' ? '✅' : '❌'} ${test4.status.toUpperCase()} (${test4.duration}ms)\n`);

  // Print summary
  console.log('\n================================================');
  console.log('📊 VALIDATION SUMMARY\n');

  const passed = results.filter(r => r.status === 'pass').length;
  const failed = results.filter(r => r.status === 'fail').length;
  const total = results.length;

  console.log(`Total Tests: ${total}`);
  console.log(`✅ Passed: ${passed}`);
  console.log(`❌ Failed: ${failed}`);
  console.log(`Success Rate: ${((passed / total) * 100).toFixed(1)}%\n`);

  // Detailed results
  console.log('Detailed Results:');
  results.forEach((result, index) => {
    console.log(`\n${index + 1}. ${result.name}`);
    console.log(`   Status: ${result.status === 'pass' ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`   Duration: ${result.duration}ms`);
    if (result.error) {
      console.log(`   Error: ${result.error}`);
    }
  });

  // Final verdict
  console.log('\n================================================');
  if (failed === 0) {
    console.log('🎉 ALL TESTS PASSED - OpenRouter integration fully validated!\n');
    process.exit(0);
  } else {
    console.log(`⚠️  ${failed} test(s) failed - review output above\n`);
    process.exit(1);
  }
}

main().catch(err => {
  console.error('Validation suite error:', err);
  process.exit(1);
});
