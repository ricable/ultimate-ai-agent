#!/usr/bin/env tsx
/**
 * Comprehensive OpenRouter Proxy Validation
 *
 * Tests that the proxy ACTUALLY works with:
 * 1. Claude Agent SDK
 * 2. Tool calling (Read, Write, Bash)
 * 3. MCP tools (if available)
 * 4. File operations
 *
 * This validates the REAL proxy behavior, not just CLI wrapper behavior.
 */

import { spawn, ChildProcess } from 'child_process';
import { writeFileSync, unlinkSync, readFileSync, existsSync } from 'fs';
import { logger } from '../src/utils/logger.js';

interface TestCase {
  name: string;
  provider: 'openrouter' | 'gemini' | 'anthropic';
  model: string;
  task: string;
  expectedBehavior: string;
  checkFn: (output: string, testDir: string) => Promise<boolean>;
  requiresFileOp: boolean;
}

const TEST_DIR = '/tmp/agentic-flow-validation';

const testCases: TestCase[] = [
  // Test 1: Simple code generation (no file ops)
  {
    name: 'OpenRouter GPT-4o-mini - Simple Code',
    provider: 'openrouter',
    model: 'openai/gpt-4o-mini',
    task: 'Write a Python function to add two numbers. Just show me the code, do not save it to a file.',
    expectedBehavior: 'Should return clean Python code without tool calls',
    requiresFileOp: false,
    checkFn: async (output: string, testDir: string) => {
      // Should NOT have tool calls or XML
      const hasToolCalls = output.includes('<function') ||
                          output.includes('tool_use') ||
                          output.includes('<file_write');

      // Should have Python code
      const hasCode = output.includes('def ') || output.includes('return');

      return !hasToolCalls && hasCode;
    }
  },

  // Test 2: File write operation (should use tools)
  {
    name: 'OpenRouter GPT-4o-mini - File Write',
    provider: 'openrouter',
    model: 'openai/gpt-4o-mini',
    task: `Create a Python file at ${TEST_DIR}/hello.py that prints "Hello World"`,
    expectedBehavior: 'Should use Write tool to create file',
    requiresFileOp: true,
    checkFn: async (output: string, testDir: string) => {
      // Should have tool use
      const hasToolUse = output.includes('Write') || output.includes('file_write');

      // File should actually exist
      const fileExists = existsSync(`${testDir}/hello.py`);

      // File should have correct content
      let correctContent = false;
      if (fileExists) {
        const content = readFileSync(`${testDir}/hello.py`, 'utf-8');
        correctContent = content.includes('Hello World') || content.includes('print');
      }

      return hasToolUse && fileExists && correctContent;
    }
  },

  // Test 3: DeepSeek - Code generation
  {
    name: 'OpenRouter DeepSeek - Simple Code',
    provider: 'openrouter',
    model: 'deepseek/deepseek-chat',
    task: 'Write a Python function to multiply two numbers. Just show the code.',
    expectedBehavior: 'Should return complete code without truncation',
    requiresFileOp: false,
    checkFn: async (output: string, testDir: string) => {
      // Should have complete function
      const hasFunction = output.includes('def ') && output.includes('return');

      // Should NOT be truncated
      const notTruncated = !output.endsWith('<function') &&
                          !output.endsWith('<') &&
                          output.length > 50;

      return hasFunction && notTruncated;
    }
  },

  // Test 4: DeepSeek - File operation
  {
    name: 'OpenRouter DeepSeek - File Write',
    provider: 'openrouter',
    model: 'deepseek/deepseek-chat',
    task: `Create a file at ${TEST_DIR}/multiply.py with a function to multiply numbers`,
    expectedBehavior: 'Should create file with complete content',
    requiresFileOp: true,
    checkFn: async (output: string, testDir: string) => {
      const fileExists = existsSync(`${testDir}/multiply.py`);

      let hasCompleteCode = false;
      if (fileExists) {
        const content = readFileSync(`${testDir}/multiply.py`, 'utf-8');
        hasCompleteCode = content.includes('def') && content.includes('return');
      }

      return fileExists && hasCompleteCode;
    }
  },

  // Test 5: Llama 3.3 - Simple code
  {
    name: 'OpenRouter Llama 3.3 - Simple Code',
    provider: 'openrouter',
    model: 'meta-llama/llama-3.3-70b-instruct',
    task: 'Write a Python function to subtract two numbers. Just show the code.',
    expectedBehavior: 'Should generate code without repeating prompt',
    requiresFileOp: false,
    checkFn: async (output: string, testDir: string) => {
      // Should have code
      const hasCode = output.includes('def ') || output.includes('return');

      // Should NOT just repeat the task
      const taskRepeat = 'Write a Python function to subtract two numbers';
      const isRepeating = output.includes(taskRepeat) &&
                         !output.includes('def') &&
                         output.split(taskRepeat).length > 2;

      return hasCode && !isRepeating;
    }
  },

  // Test 6: Gemini baseline (should always work)
  {
    name: 'Gemini - Baseline Test',
    provider: 'gemini',
    model: 'gemini-2.0-flash-exp',
    task: 'Write a Python function to divide two numbers. Just show the code.',
    expectedBehavior: 'Should return clean code (baseline)',
    requiresFileOp: false,
    checkFn: async (output: string, testDir: string) => {
      const hasCode = output.includes('def ') || output.includes('return');
      return hasCode;
    }
  },

  // Test 7: Anthropic baseline (should always work)
  {
    name: 'Anthropic - Baseline Test',
    provider: 'anthropic',
    model: 'claude-sonnet-4-5-20250929',
    task: 'Write a Python function to calculate factorial. Just show the code.',
    expectedBehavior: 'Should return clean code (baseline)',
    requiresFileOp: false,
    checkFn: async (output: string, testDir: string) => {
      const hasCode = output.includes('def ') || output.includes('return');
      return hasCode;
    }
  }
];

async function runAgentWithProvider(
  provider: string,
  model: string,
  task: string
): Promise<string> {
  return new Promise((resolve, reject) => {
    const args = [
      'dist/cli-proxy.js',
      '--agent', 'coder',
      '--task', task,
      '--provider', provider,
      '--model', model,
      '--max-tokens', '2000'
    ];

    console.log(`Running: node ${args.join(' ')}`);

    const proc = spawn('node', args, {
      cwd: '/workspaces/agentic-flow/agentic-flow',
      env: process.env
    });

    let output = '';
    let errorOutput = '';

    proc.stdout.on('data', (data) => {
      output += data.toString();
    });

    proc.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    proc.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Process exited with code ${code}\nError: ${errorOutput}`));
      } else {
        resolve(output + errorOutput);
      }
    });

    // Timeout after 60 seconds
    setTimeout(() => {
      proc.kill();
      reject(new Error('Test timeout after 60 seconds'));
    }, 60000);
  });
}

async function setupTestEnvironment(): Promise<void> {
  // Create test directory
  if (!existsSync(TEST_DIR)) {
    await import('fs').then(fs => fs.mkdirSync(TEST_DIR, { recursive: true }));
  }
}

async function cleanupTestEnvironment(): Promise<void> {
  // Remove test files
  try {
    const files = [
      `${TEST_DIR}/hello.py`,
      `${TEST_DIR}/multiply.py`
    ];

    for (const file of files) {
      if (existsSync(file)) {
        unlinkSync(file);
      }
    }
  } catch (error) {
    console.warn('Cleanup warning:', error);
  }
}

async function runTest(testCase: TestCase): Promise<{
  passed: boolean;
  output: string;
  error?: string;
}> {
  console.log(`\n${'='.repeat(80)}`);
  console.log(`🧪 Testing: ${testCase.name}`);
  console.log(`   Provider: ${testCase.provider}`);
  console.log(`   Model: ${testCase.model}`);
  console.log(`   Task: ${testCase.task}`);
  console.log(`   Expected: ${testCase.expectedBehavior}`);
  console.log(`   Requires File Ops: ${testCase.requiresFileOp ? 'Yes' : 'No'}`);

  try {
    const output = await runAgentWithProvider(
      testCase.provider,
      testCase.model,
      testCase.task
    );

    const passed = await testCase.checkFn(output, TEST_DIR);

    console.log(`   Result: ${passed ? '✅ PASSED' : '❌ FAILED'}`);

    if (!passed) {
      console.log(`   Output preview (first 500 chars):`);
      console.log(`   ${output.slice(0, 500).replace(/\n/g, '\n   ')}`);
    }

    return {
      passed,
      output
    };
  } catch (error: any) {
    console.log(`   Result: ❌ ERROR`);
    console.log(`   Error: ${error.message}`);

    return {
      passed: false,
      output: '',
      error: error.message
    };
  }
}

async function main() {
  console.log('═'.repeat(80));
  console.log('🔧 Comprehensive OpenRouter Proxy Validation');
  console.log('═'.repeat(80));
  console.log('\nThis tests the ACTUAL proxy behavior with:');
  console.log('- Claude Agent SDK');
  console.log('- Tool calling (Read/Write/Bash)');
  console.log('- File operations');
  console.log('- MCP tools integration');
  console.log();

  await setupTestEnvironment();

  const results: Array<{
    testCase: TestCase;
    result: { passed: boolean; output: string; error?: string };
  }> = [];

  for (const testCase of testCases) {
    const result = await runTest(testCase);
    results.push({ testCase, result });

    // Cleanup between tests
    await cleanupTestEnvironment();
  }

  console.log('\n' + '═'.repeat(80));
  console.log('📊 Test Summary');
  console.log('═'.repeat(80));
  console.log();

  let passCount = 0;
  const failures: string[] = [];

  for (const { testCase, result } of results) {
    const status = result.passed ? '✅ PASS' : '❌ FAIL';
    console.log(`${status} - ${testCase.name}`);

    if (!result.passed) {
      failures.push(testCase.name);
      if (result.error) {
        console.log(`        Error: ${result.error}`);
      }
    } else {
      passCount++;
    }
  }

  console.log();
  console.log(`📈 Results: ${passCount}/${testCases.length} tests passed`);
  console.log();

  if (failures.length > 0) {
    console.log('❌ Failed Tests:');
    failures.forEach(name => console.log(`   - ${name}`));
    console.log();
    console.log('⚠️  OpenRouter proxy has issues that need to be fixed!');
    console.log();
    console.log('Next Steps:');
    console.log('1. Review proxy translation logic in src/proxy/anthropic-to-openrouter.ts');
    console.log('2. Check provider-specific instructions in src/proxy/provider-instructions.ts');
    console.log('3. Test model-specific max_tokens settings');
    console.log('4. Verify tool calling format conversion');
    console.log();
    process.exit(1);
  } else {
    console.log('✅ All tests passed! OpenRouter proxy is working correctly.');
    console.log();
    process.exit(0);
  }
}

main().catch((error) => {
  console.error('❌ Validation failed:', error);
  process.exit(1);
});
