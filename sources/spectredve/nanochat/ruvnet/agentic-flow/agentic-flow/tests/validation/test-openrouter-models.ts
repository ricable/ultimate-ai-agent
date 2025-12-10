// Test popular OpenRouter models with tool instruction parsing
import { writeFileSync } from 'fs';

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const PROXY_URL = 'http://localhost:3000';

// Popular models from https://openrouter.ai/models
const POPULAR_MODELS = [
  'meta-llama/llama-3.1-8b-instruct',           // Llama 3.1 8B (free)
  'google/gemini-flash-1.5',                     // Gemini Flash 1.5
  'mistralai/mistral-7b-instruct',              // Mistral 7B
  'meta-llama/llama-3.1-70b-instruct',          // Llama 3.1 70B
  'qwen/qwen-2.5-7b-instruct'                   // Qwen 2.5 7B
];

async function testModel(model: string): Promise<{
  model: string;
  success: boolean;
  toolUsesDetected: number;
  error?: string;
  fileCreated?: boolean;
}> {
  console.log(`\n📦 Testing: ${model}`);

  const anthropicRequest = {
    model,
    max_tokens: 1024,
    system: 'You are a helpful coding assistant.',
    messages: [
      {
        role: 'user',
        content: `Create a test-${model.replace(/[^a-z0-9]/gi, '-')}.txt file with the text "Hello from ${model}!"`
      }
    ]
  };

  try {
    const response = await fetch(`${PROXY_URL}/v1/messages`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify(anthropicRequest)
    });

    if (!response.ok) {
      const error = await response.text();
      console.error(`  ❌ HTTP ${response.status}: ${error.substring(0, 200)}`);
      return {
        model,
        success: false,
        toolUsesDetected: 0,
        error: `HTTP ${response.status}`
      };
    }

    const result = await response.json();
    const toolUses = result.content?.filter((c: any) => c.type === 'tool_use') || [];

    console.log(`  ✅ Response received`);
    console.log(`  📊 Tool uses detected: ${toolUses.length}`);

    if (toolUses.length > 0) {
      toolUses.forEach((tool: any, i: number) => {
        console.log(`     Tool ${i + 1}: ${tool.name} - ${JSON.stringify(tool.input).substring(0, 60)}...`);
      });
    }

    return {
      model,
      success: true,
      toolUsesDetected: toolUses.length,
      fileCreated: toolUses.some((t: any) => t.name === 'Write')
    };

  } catch (error: any) {
    console.error(`  ❌ Error: ${error.message}`);
    return {
      model,
      success: false,
      toolUsesDetected: 0,
      error: error.message
    };
  }
}

async function runTests() {
  console.log('🧪 Testing popular OpenRouter models with tool instructions\n');
  console.log(`Testing ${POPULAR_MODELS.length} models...\n`);

  const results = [];

  for (const model of POPULAR_MODELS) {
    const result = await testModel(model);
    results.push(result);
    // Rate limit - wait between requests
    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  console.log('\n\n═══════════════════════════════════════');
  console.log('📊 TEST SUMMARY');
  console.log('═══════════════════════════════════════\n');

  const successful = results.filter(r => r.success);
  const withTools = results.filter(r => r.toolUsesDetected > 0);
  const filesCreated = results.filter(r => r.fileCreated);

  console.log(`Total Models Tested: ${results.length}`);
  console.log(`Successful Responses: ${successful.length}`);
  console.log(`Models Using Tools: ${withTools.length}`);
  console.log(`Files Created: ${filesCreated.length}\n`);

  console.log('Detailed Results:\n');
  results.forEach(r => {
    const status = r.success ? '✅' : '❌';
    const tools = r.toolUsesDetected > 0 ? `🔧 ${r.toolUsesDetected} tools` : '⚠️  No tools';
    console.log(`${status} ${r.model.padEnd(45)} ${tools}`);
    if (r.error) {
      console.log(`   Error: ${r.error}`);
    }
  });

  // Save results to file
  writeFileSync(
    'openrouter-model-test-results.json',
    JSON.stringify(results, null, 2)
  );

  console.log('\n\n✅ Results saved to: openrouter-model-test-results.json');
}

runTests().catch(err => {
  console.error('Test suite failed:', err);
  process.exit(1);
});
