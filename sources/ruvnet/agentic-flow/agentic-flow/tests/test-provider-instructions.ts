// Test provider-specific tool instructions with individual models
import fetch from 'node-fetch';

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const PROXY_URL = 'http://localhost:3000';

if (!OPENROUTER_API_KEY) {
  console.error('❌ Error: OPENROUTER_API_KEY environment variable required');
  process.exit(1);
}

// Test models representing each provider
const TEST_MODELS = [
  { id: 'anthropic/claude-sonnet-4', name: 'Claude Sonnet 4', provider: 'anthropic' },
  { id: 'openai/gpt-4o-mini', name: 'GPT-4o-mini', provider: 'openai' },
  { id: 'google/gemini-2.5-flash', name: 'Gemini 2.5 Flash', provider: 'google' },
  { id: 'meta-llama/llama-3.1-8b-instruct', name: 'Llama 3.1 8B', provider: 'meta-llama' },
  { id: 'deepseek/deepseek-chat-v3.1:free', name: 'DeepSeek V3.1', provider: 'deepseek' },
  { id: 'mistralai/mistral-7b-instruct', name: 'Mistral 7B', provider: 'mistralai' },
  { id: 'x-ai/grok-4-fast', name: 'Grok 4 Fast', provider: 'x-ai' }
];

interface TestResult {
  modelId: string;
  modelName: string;
  provider: string;
  success: boolean;
  toolUsesDetected: number;
  responseTime: number;
  instructionsUsed: string;
  error?: string;
  responsePreview?: string;
}

async function testModel(model: typeof TEST_MODELS[0]): Promise<TestResult> {
  console.log(`\n📦 Testing: ${model.name} (${model.provider})`);

  const anthropicRequest = {
    model: model.id,
    max_tokens: 512,
    system: 'You are a helpful coding assistant.',
    messages: [
      {
        role: 'user',
        content: `Create a file named ${model.provider}-test.txt with the text "Hello from ${model.name}!"`
      }
    ]
  };

  const startTime = Date.now();

  try {
    const response = await fetch(`${PROXY_URL}/v1/messages`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify(anthropicRequest)
    });

    const responseTime = Date.now() - startTime;

    if (!response.ok) {
      const error = await response.text();
      console.error(`  ❌ HTTP ${response.status}: ${error.substring(0, 100)}`);
      return {
        modelId: model.id,
        modelName: model.name,
        provider: model.provider,
        success: false,
        toolUsesDetected: 0,
        responseTime,
        instructionsUsed: 'N/A',
        error: `HTTP ${response.status}`
      };
    }

    const result = await response.json();
    const toolUses = result.content?.filter((c: any) => c.type === 'tool_use') || [];
    const textContent = result.content?.filter((c: any) => c.type === 'text')
      .map((c: any) => c.text).join(' ').substring(0, 150) || '';

    console.log(`  ✅ Response (${responseTime}ms)`);
    console.log(`  📊 Tool uses: ${toolUses.length}`);
    if (toolUses.length > 0) {
      console.log(`  🔧 Tools: ${toolUses.map((t: any) => t.name).join(', ')}`);
    }
    if (toolUses.length === 0 && textContent) {
      console.log(`  📝 Text: ${textContent}...`);
    }

    return {
      modelId: model.id,
      modelName: model.name,
      provider: model.provider,
      success: true,
      toolUsesDetected: toolUses.length,
      responseTime,
      instructionsUsed: `${model.provider.toUpperCase()}_INSTRUCTIONS`,
      responsePreview: textContent
    };

  } catch (error: any) {
    const responseTime = Date.now() - startTime;
    console.error(`  ❌ Error: ${error.message}`);
    return {
      modelId: model.id,
      modelName: model.name,
      provider: model.provider,
      success: false,
      toolUsesDetected: 0,
      responseTime,
      instructionsUsed: 'N/A',
      error: error.message
    };
  }
}

async function runTests() {
  console.log('🧪 Testing Provider-Specific Tool Instructions\n');
  console.log('Testing one model from each provider family...\n');

  const results: TestResult[] = [];

  for (const model of TEST_MODELS) {
    const result = await testModel(model);
    results.push(result);

    // Rate limit between requests
    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  console.log('\n\n' + '═'.repeat(80));
  console.log('📊 PROVIDER INSTRUCTION TEST RESULTS');
  console.log('═'.repeat(80) + '\n');

  // Group by success
  const successful = results.filter(r => r.success && r.toolUsesDetected > 0);
  const failed = results.filter(r => !r.success || r.toolUsesDetected === 0);

  console.log('✅ SUCCESSFUL (with tool usage):');
  successful.forEach(r => {
    console.log(`  - ${r.provider.padEnd(12)} | ${r.modelName.padEnd(25)} | ${r.toolUsesDetected} tools | ${r.responseTime}ms`);
  });

  console.log('\n❌ NEEDS OPTIMIZATION:');
  failed.forEach(r => {
    const reason = r.error || (r.toolUsesDetected === 0 ? 'No tool usage' : 'Unknown');
    console.log(`  - ${r.provider.padEnd(12)} | ${r.modelName.padEnd(25)} | ${reason}`);
  });

  console.log(`\n📈 Summary:`);
  console.log(`   Total Models: ${results.length}`);
  console.log(`   With Tool Usage: ${successful.length}/${results.length} (${((successful.length / results.length) * 100).toFixed(1)}%)`);
  console.log(`   Avg Response Time: ${Math.round(results.filter(r => r.success).reduce((sum, r) => sum + r.responseTime, 0) / results.filter(r => r.success).length)}ms`);

  return results;
}

runTests().catch(err => {
  console.error('Test suite failed:', err);
  process.exit(1);
});
