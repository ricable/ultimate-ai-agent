#!/usr/bin/env tsx
/**
 * Test Claude Code with Non-Tool Model Emulation
 *
 * Simulates what would happen if Claude Code used a non-tool model
 * with the emulation layer integrated.
 */

import { ToolEmulator, executeEmulation, Tool, ToolCall } from '../src/proxy/tool-emulation.js';
import { detectModelCapabilities } from '../src/utils/modelCapabilities.js';

const OPENROUTER_KEY = process.env.OPENROUTER_API_KEY;

if (!OPENROUTER_KEY) {
  console.error('❌ OPENROUTER_API_KEY required');
  console.log('\nSet your API key:');
  console.log('  export OPENROUTER_API_KEY="sk-or-..."');
  process.exit(1);
}

// Simulate Claude Code's tool catalog (simplified)
const claudeCodeTools: Tool[] = [
  {
    name: 'read_file',
    description: 'Read contents of a file',
    input_schema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'File path to read' }
      },
      required: ['path']
    }
  },
  {
    name: 'write_file',
    description: 'Write content to a file',
    input_schema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'File path' },
        content: { type: 'string', description: 'Content to write' }
      },
      required: ['path', 'content']
    }
  },
  {
    name: 'bash',
    description: 'Execute bash command',
    input_schema: {
      type: 'object',
      properties: {
        command: { type: 'string', description: 'Command to execute' }
      },
      required: ['command']
    }
  }
];

// Mock tool executor
async function executeTool(toolCall: ToolCall): Promise<any> {
  console.log(`\n   🔧 Executing: ${toolCall.name}(${JSON.stringify(toolCall.arguments)})`);

  // Simulate tool execution
  switch (toolCall.name) {
    case 'read_file':
      return {
        content: '// Example file content\nfunction hello() {\n  console.log("Hello");\n}',
        path: toolCall.arguments.path
      };

    case 'write_file':
      return {
        success: true,
        path: toolCall.arguments.path,
        bytes_written: toolCall.arguments.content.length
      };

    case 'bash':
      return {
        stdout: 'Command executed successfully',
        stderr: '',
        exit_code: 0
      };

    default:
      throw new Error(`Unknown tool: ${toolCall.name}`);
  }
}

// OpenRouter API call
async function callOpenRouter(model: string, prompt: string): Promise<string> {
  console.log(`\n   📡 Calling OpenRouter: ${model}`);

  const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${OPENROUTER_KEY}`,
      'Content-Type': 'application/json',
      'HTTP-Referer': 'https://agentic.flow',
      'X-Title': 'Claude Code Tool Emulation Test'
    },
    body: JSON.stringify({
      model,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.3,
      max_tokens: 2000
    })
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`OpenRouter API error: ${response.status} ${error}`);
  }

  const data = await response.json();
  return data.choices[0].message.content;
}

// Test scenarios
const testCases = [
  {
    name: 'Simple File Read',
    model: 'mistralai/mistral-7b-instruct',
    task: 'Read the file at path "/home/user/example.js"',
    expectedTool: 'read_file'
  },
  {
    name: 'File Write',
    model: 'meta-llama/llama-2-13b-chat',
    task: 'Write "Hello World" to the file "/tmp/test.txt"',
    expectedTool: 'write_file'
  },
  {
    name: 'Bash Command',
    model: 'mistralai/mistral-7b-instruct',
    task: 'Run the command "ls -la" using bash',
    expectedTool: 'bash'
  }
];

async function runTest(testCase: typeof testCases[0]) {
  console.log('\n' + '═'.repeat(80));
  console.log(`🧪 Test: ${testCase.name}`);
  console.log('═'.repeat(80));
  console.log(`Model: ${testCase.model}`);
  console.log(`Task: ${testCase.task}`);

  // Detect capabilities
  const capabilities = detectModelCapabilities(testCase.model);
  console.log(`\n📊 Model Capabilities:`);
  console.log(`   Native Tools: ${capabilities.supportsNativeTools ? '✅' : '❌'}`);
  console.log(`   Requires Emulation: ${capabilities.requiresEmulation ? '🔧 YES' : 'NO'}`);
  console.log(`   Strategy: ${capabilities.emulationStrategy.toUpperCase()}`);

  if (!capabilities.requiresEmulation) {
    console.log(`\n⚠️  This model supports native tools - emulation not needed`);
    return { skipped: true };
  }

  // Create emulator
  const emulator = new ToolEmulator(claudeCodeTools, capabilities.emulationStrategy as 'react' | 'prompt');

  console.log(`\n⚙️  Starting emulation with ${capabilities.emulationStrategy.toUpperCase()} strategy...`);

  try {
    const result = await executeEmulation(
      emulator,
      testCase.task,
      (prompt) => callOpenRouter(testCase.model, prompt),
      executeTool,
      { maxIterations: 3, verbose: true }
    );

    console.log('\n' + '─'.repeat(80));
    console.log('📊 RESULTS');
    console.log('─'.repeat(80));
    console.log(`\n✅ Tool Calls: ${result.toolCalls.length}`);

    if (result.toolCalls.length > 0) {
      const firstTool = result.toolCalls[0];
      console.log(`   Tool Used: ${firstTool.name}`);
      console.log(`   Arguments: ${JSON.stringify(firstTool.arguments, null, 2)}`);

      const correct = firstTool.name === testCase.expectedTool;
      console.log(`   Expected: ${testCase.expectedTool}`);
      console.log(`   ${correct ? '✅ CORRECT' : '❌ INCORRECT'}`);
    }

    if (result.finalAnswer) {
      console.log(`\n📝 Final Answer: ${result.finalAnswer.substring(0, 200)}...`);
    }

    console.log(`\n📈 Confidence: ${(result.confidence * 100).toFixed(1)}%`);

    return {
      success: true,
      toolCallCount: result.toolCalls.length,
      correct: result.toolCalls[0]?.name === testCase.expectedTool,
      confidence: result.confidence
    };

  } catch (error: any) {
    console.error(`\n❌ Test failed: ${error.message}`);
    return { success: false, error: error.message };
  }
}

// Run all tests
async function main() {
  console.log('\n' + '═'.repeat(80));
  console.log('🧪 CLAUDE CODE TOOL EMULATION TEST');
  console.log('═'.repeat(80));
  console.log('\nSimulates Claude Code using non-tool models with emulation layer\n');

  const results: any[] = [];

  for (const testCase of testCases) {
    const result = await runTest(testCase);
    results.push({ ...testCase, ...result });

    // Wait between tests to avoid rate limits
    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  // Summary
  console.log('\n\n' + '═'.repeat(80));
  console.log('📊 TEST SUMMARY');
  console.log('═'.repeat(80) + '\n');

  const successCount = results.filter(r => r.success && !r.skipped).length;
  const skippedCount = results.filter(r => r.skipped).length;
  const totalCount = results.length - skippedCount;

  console.log(`Tests Completed: ${successCount}/${totalCount}`);
  console.log(`Tests Skipped: ${skippedCount}\n`);

  results.forEach((result, i) => {
    const status = result.skipped ? '⏭️' : result.success ? '✅' : '❌';
    console.log(`${status} Test ${i + 1}: ${result.name} (${result.model})`);

    if (result.success && !result.skipped) {
      console.log(`   Tool Calls: ${result.toolCallCount}`);
      console.log(`   Accuracy: ${result.correct ? '✅ Correct' : '❌ Incorrect'}`);
      console.log(`   Confidence: ${(result.confidence * 100).toFixed(1)}%`);
    } else if (result.error) {
      console.log(`   Error: ${result.error}`);
    }
    console.log('');
  });

  console.log('═'.repeat(80) + '\n');

  if (successCount === totalCount && totalCount > 0) {
    console.log('🎉 All tests passed! Tool emulation works for Claude Code.\n');
  } else if (totalCount === 0) {
    console.log('⚠️  All models already support native tools - no emulation needed.\n');
  } else {
    console.log('⚠️  Some tests failed. Review the errors above.\n');
  }
}

main().catch(console.error);
