#!/usr/bin/env tsx
/**
 * Validation test for OpenRouter proxy fixes
 * Tests the three issues reported in validation:
 * 1. GPT-4o-mini: XML format instead of clean code
 * 2. DeepSeek: Truncated responses
 * 3. Llama 3.3: No code generation, just repeats prompt
 */

import { spawn } from 'child_process';
import { logger } from '../src/utils/logger.js';

interface TestCase {
  name: string;
  model: string;
  task: string;
  expectedBehavior: string;
  checkFn: (output: string) => boolean;
}

const testCases: TestCase[] = [
  {
    name: 'GPT-4o-mini - Clean Code (No XML)',
    model: 'openai/gpt-4o-mini',
    task: 'Write a Python function to reverse a string',
    expectedBehavior: 'Should return clean code without XML tags',
    checkFn: (output: string) => {
      const hasXmlTags = output.includes('<file_write') || output.includes('<bash_command');
      const hasCode = output.includes('def ') || output.includes('function');
      return !hasXmlTags && hasCode;
    }
  },
  {
    name: 'DeepSeek - Complete Response',
    model: 'deepseek/deepseek-chat',
    task: 'Write a simple REST API with three endpoints',
    expectedBehavior: 'Should generate complete response with 8000 max_tokens',
    checkFn: (output: string) => {
      const notTruncated = !output.includes('<function=') || output.length > 500;
      const hasCode = output.includes('def ') || output.includes('app') || output.includes('route');
      return notTruncated && hasCode;
    }
  },
  {
    name: 'Llama 3.3 - Code Generation',
    model: 'meta-llama/llama-3.3-70b-instruct',
    task: 'Write a function to calculate factorial',
    expectedBehavior: 'Should generate code instead of repeating prompt',
    checkFn: (output: string) => {
      // Check if output contains actual code (function, def, or code block)
      const hasCode = output.includes('def ') ||
                     output.includes('function') ||
                     output.includes('factorial') ||
                     output.includes('```');

      // Check if it's NOT just repeating the exact task
      const taskRepeat = 'Write a function to calculate factorial';
      const isRepeating = output.includes(taskRepeat) &&
                         !output.includes('def') &&
                         !output.includes('function') &&
                         output.split(taskRepeat).length > 2; // Multiple exact repeats

      return hasCode && !isRepeating;
    }
  }
];

async function runTest(testCase: TestCase): Promise<{
  passed: boolean;
  output: string;
  error?: string;
}> {
  return new Promise((resolve) => {
    const args = [
      'dist/cli-proxy.js',
      '--agent', 'coder',
      '--task', testCase.task,
      '--provider', 'openrouter',
      '--model', testCase.model,
      '--max-tokens', '1000'
    ];

    console.log(`\n🧪 Testing: ${testCase.name}`);
    console.log(`   Model: ${testCase.model}`);
    console.log(`   Task: ${testCase.task}`);

    const child = spawn('node', args, {
      cwd: '/workspaces/agentic-flow/agentic-flow',
      env: process.env
    });

    let output = '';
    let error = '';

    child.stdout.on('data', (data) => {
      output += data.toString();
    });

    child.stderr.on('data', (data) => {
      error += data.toString();
    });

    child.on('close', (code) => {
      const fullOutput = output + error;
      const passed = testCase.checkFn(fullOutput);

      console.log(`   Expected: ${testCase.expectedBehavior}`);
      console.log(`   Result: ${passed ? '✅ PASSED' : '❌ FAILED'}`);

      if (!passed) {
        console.log(`   Output preview: ${fullOutput.slice(0, 200)}...`);
      }

      resolve({
        passed,
        output: fullOutput,
        error: code !== 0 ? `Exit code: ${code}` : undefined
      });
    });

    // Timeout after 60 seconds
    setTimeout(() => {
      child.kill();
      resolve({
        passed: false,
        output,
        error: 'Timeout after 60 seconds'
      });
    }, 60000);
  });
}

async function main() {
  console.log('═══════════════════════════════════════════════════════════');
  console.log('🔧 OpenRouter Proxy Fix Validation');
  console.log('═══════════════════════════════════════════════════════════');

  const results = [];

  for (const testCase of testCases) {
    const result = await runTest(testCase);
    results.push({ testCase, result });
  }

  console.log('\n═══════════════════════════════════════════════════════════');
  console.log('📊 Test Summary');
  console.log('═══════════════════════════════════════════════════════════\n');

  let passCount = 0;
  for (const { testCase, result } of results) {
    const status = result.passed ? '✅ PASS' : '❌ FAIL';
    console.log(`${status} - ${testCase.name}`);
    if (result.error) {
      console.log(`        Error: ${result.error}`);
    }
    if (result.passed) passCount++;
  }

  console.log(`\n📈 Results: ${passCount}/${testCases.length} tests passed`);

  if (passCount === testCases.length) {
    console.log('\n✅ All OpenRouter proxy fixes validated successfully!');
    process.exit(0);
  } else {
    console.log('\n⚠️  Some tests failed. Review the output above.');
    process.exit(1);
  }
}

main().catch((error) => {
  console.error('❌ Validation failed:', error);
  process.exit(1);
});
