#!/usr/bin/env tsx
/**
 * Tool Emulation Test
 *
 * Tests tool emulation with models that don't support native function calling:
 * - mistralai/mistral-7b-instruct (no native tools)
 * - thudm/glm-4-9b:free (free tier, no tools)
 *
 * Compares with a tool-capable model:
 * - deepseek/deepseek-chat (native tools)
 */

import { ToolEmulator, executeEmulation, Tool, ToolCall } from '../src/proxy/tool-emulation.js';
import { detectModelCapabilities, getCapabilityReport } from '../src/utils/modelCapabilities.js';

// Mock tools for testing
const testTools: Tool[] = [
  {
    name: 'calculate',
    description: 'Perform mathematical calculations',
    input_schema: {
      type: 'object',
      properties: {
        expression: {
          type: 'string',
          description: 'Math expression to evaluate (e.g., "2 + 2", "10 * 5")'
        }
      },
      required: ['expression']
    }
  },
  {
    name: 'get_weather',
    description: 'Get current weather for a location',
    input_schema: {
      type: 'object',
      properties: {
        location: {
          type: 'string',
          description: 'City name or coordinates'
        },
        units: {
          type: 'string',
          description: 'Temperature units (celsius or fahrenheit)'
        }
      },
      required: ['location']
    }
  },
  {
    name: 'search_web',
    description: 'Search the web for information',
    input_schema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Search query'
        }
      },
      required: ['query']
    }
  }
];

// Mock tool executor
async function executeTool(toolCall: ToolCall): Promise<any> {
  console.log(`   🔧 Executing: ${toolCall.name}(${JSON.stringify(toolCall.arguments)})`);

  switch (toolCall.name) {
    case 'calculate':
      try {
        // Simple eval for demo (don't use in production!)
        const result = eval(toolCall.arguments.expression);
        return { result, expression: toolCall.arguments.expression };
      } catch (e: any) {
        return { error: `Invalid expression: ${e.message}` };
      }

    case 'get_weather':
      // Mock weather data
      return {
        location: toolCall.arguments.location,
        temperature: 22,
        units: toolCall.arguments.units || 'celsius',
        conditions: 'Partly cloudy',
        humidity: 65
      };

    case 'search_web':
      // Mock search results
      return {
        query: toolCall.arguments.query,
        results: [
          { title: 'Result 1', snippet: 'This is a mock search result.' },
          { title: 'Result 2', snippet: 'Another mock result for testing.' }
        ]
      };

    default:
      throw new Error(`Unknown tool: ${toolCall.name}`);
  }
}

// Mock OpenRouter API call
async function callOpenRouter(model: string, prompt: string): Promise<string> {
  const OPENROUTER_KEY = process.env.OPENROUTER_API_KEY;

  if (!OPENROUTER_KEY) {
    throw new Error('OPENROUTER_API_KEY environment variable required');
  }

  console.log(`\n   📡 Calling OpenRouter: ${model}`);
  console.log(`   Prompt length: ${prompt.length} characters\n`);

  const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${OPENROUTER_KEY}`,
      'Content-Type': 'application/json',
      'HTTP-Referer': 'https://agentic.flow',
      'X-Title': 'Agentic Flow Tool Emulation Test'
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: 'user', content: prompt }
      ],
      temperature: 0.7,
      max_tokens: 1000
    })
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`OpenRouter API error: ${response.status} ${error}`);
  }

  const data = await response.json();
  return data.choices[0].message.content;
}

// Test function
async function testModel(
  modelId: string,
  userQuestion: string,
  strategy: 'react' | 'prompt'
) {
  console.log('\n' + '═'.repeat(80));
  console.log(`\n🧪 Testing Model: ${modelId}`);
  console.log(`Strategy: ${strategy.toUpperCase()}`);
  console.log(getCapabilityReport(modelId));

  // Create emulator
  const emulator = new ToolEmulator(testTools, strategy);

  console.log(`\n💬 User Question: "${userQuestion}"\n`);

  try {
    // Execute emulation
    const result = await executeEmulation(
      emulator,
      userQuestion,
      (prompt) => callOpenRouter(modelId, prompt),
      executeTool,
      { maxIterations: 5, verbose: true }
    );

    // Display results
    console.log('\n' + '─'.repeat(80));
    console.log('📊 RESULTS');
    console.log('─'.repeat(80));
    console.log(`\n✅ Tool Calls: ${result.toolCalls.length}`);

    if (result.toolCalls.length > 0) {
      result.toolCalls.forEach((call, i) => {
        console.log(`\n   ${i + 1}. ${call.name}`);
        console.log(`      Arguments: ${JSON.stringify(call.arguments, null, 2).split('\n').join('\n      ')}`);
      });
    }

    if (result.reasoning) {
      console.log(`\n💭 Reasoning: ${result.reasoning.substring(0, 200)}...`);
    }

    if (result.finalAnswer) {
      console.log(`\n📝 Final Answer:\n   ${result.finalAnswer}`);
    }

    console.log(`\n📈 Confidence Score: ${(result.confidence * 100).toFixed(1)}%`);

    return result;

  } catch (error: any) {
    console.error(`\n❌ Test failed: ${error.message}`);
    throw error;
  }
}

// Main test suite
async function runTests() {
  console.log('\n' + '═'.repeat(80));
  console.log('🧪 TOOL EMULATION TEST SUITE');
  console.log('═'.repeat(80));
  console.log('\nTesting tool emulation with non-tool-capable models\n');

  const testCases = [
    {
      model: 'mistralai/mistral-7b-instruct',
      strategy: 'react' as const,
      question: 'What is 15 multiplied by 23?'
    },
    {
      model: 'mistralai/mistral-7b-instruct',
      strategy: 'prompt' as const,
      question: 'Calculate 100 divided by 4'
    },
    {
      model: 'thudm/glm-4-9b:free',
      strategy: 'react' as const,
      question: 'What is the weather in Tokyo?'
    },
    {
      model: 'mistralai/mistral-7b-instruct',
      strategy: 'react' as const,
      question: 'First calculate 5 + 5, then search the web for information about that number.'
    }
  ];

  const results: any[] = [];

  for (const testCase of testCases) {
    try {
      const result = await testModel(
        testCase.model,
        testCase.question,
        testCase.strategy
      );
      results.push({
        model: testCase.model,
        strategy: testCase.strategy,
        question: testCase.question,
        success: true,
        toolCallCount: result.toolCalls.length,
        confidence: result.confidence
      });
    } catch (error: any) {
      results.push({
        model: testCase.model,
        strategy: testCase.strategy,
        question: testCase.question,
        success: false,
        error: error.message
      });
    }

    // Wait between tests to avoid rate limits
    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  // Summary
  console.log('\n\n' + '═'.repeat(80));
  console.log('📊 TEST SUMMARY');
  console.log('═'.repeat(80) + '\n');

  const successCount = results.filter(r => r.success).length;
  const totalCount = results.length;

  console.log(`Tests Passed: ${successCount}/${totalCount} (${(successCount / totalCount * 100).toFixed(1)}%)\n`);

  results.forEach((result, i) => {
    const status = result.success ? '✅' : '❌';
    console.log(`${status} Test ${i + 1}: ${result.model} (${result.strategy})`);

    if (result.success) {
      console.log(`   Tool Calls: ${result.toolCallCount}`);
      console.log(`   Confidence: ${(result.confidence * 100).toFixed(1)}%`);
    } else {
      console.log(`   Error: ${result.error}`);
    }
    console.log('');
  });

  console.log('═'.repeat(80) + '\n');

  if (successCount === totalCount) {
    console.log('🎉 All tests passed! Tool emulation is working correctly.\n');
  } else {
    console.log('⚠️  Some tests failed. Check the errors above.\n');
  }
}

// Run tests
runTests().catch(console.error);
