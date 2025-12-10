// Test free OpenRouter models with tool instruction parsing
import { writeFileSync } from 'fs';

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const PROXY_URL = 'http://localhost:3000';

// Free models from https://openrouter.ai/models?q=free (2025)
const FREE_MODELS = [
  'google/gemini-2.0-flash-exp:free',               // Gemini 2.0 Flash (Free)
  'deepseek/deepseek-r1:free',                      // DeepSeek R1 (Free)
  'deepseek/deepseek-r1-distill-llama-70b:free',   // DeepSeek R1 Distill Llama 70B (Free)
  'deepseek/deepseek-r1-distill-qwen-14b:free',    // DeepSeek R1 Distill Qwen 14B (Free)
  'meta-llama/llama-3.1-8b-instruct',               // Llama 3.1 8B (already tested - paid)
  'mistralai/mistral-7b-instruct',                  // Mistral 7B (already tested - paid)
  'qwen/qwen-2.5-7b-instruct'                       // Qwen 2.5 7B (already tested - paid)
];

async function testModel(model: string): Promise<{
  model: string;
  success: boolean;
  toolUsesDetected: number;
  error?: string;
  fileCreated?: boolean;
  responseTime?: number;
}> {
  console.log(`\n📦 Testing: ${model}`);

  const anthropicRequest = {
    model,
    max_tokens: 512,
    system: 'You are a helpful coding assistant.',
    messages: [
      {
        role: 'user',
        content: `Create a test-${model.split('/')[1].split(':')[0]}.txt file with "Hello from ${model.split('/')[1]}!"`
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
      console.error(`  ❌ HTTP ${response.status}: ${error.substring(0, 150)}`);
      return {
        model,
        success: false,
        toolUsesDetected: 0,
        error: `HTTP ${response.status}`,
        responseTime
      };
    }

    const result = await response.json();
    const toolUses = result.content?.filter((c: any) => c.type === 'tool_use') || [];

    console.log(`  ✅ Response received (${responseTime}ms)`);
    console.log(`  📊 Tool uses detected: ${toolUses.length}`);

    if (toolUses.length > 0) {
      toolUses.forEach((tool: any, i: number) => {
        const inputPreview = JSON.stringify(tool.input).substring(0, 50);
        console.log(`     Tool ${i + 1}: ${tool.name} - ${inputPreview}...`);
      });
    }

    return {
      model,
      success: true,
      toolUsesDetected: toolUses.length,
      fileCreated: toolUses.some((t: any) => t.name === 'Write'),
      responseTime
    };

  } catch (error: any) {
    const responseTime = Date.now() - startTime;
    console.error(`  ❌ Error: ${error.message}`);
    return {
      model,
      success: false,
      toolUsesDetected: 0,
      error: error.message,
      responseTime
    };
  }
}

async function runTests() {
  console.log('🧪 Testing FREE OpenRouter models with tool instructions\n');
  console.log(`Testing ${FREE_MODELS.length} free models...\n`);

  const results = [];

  for (const model of FREE_MODELS) {
    const result = await testModel(model);
    results.push(result);
    // Rate limit - wait between requests
    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  console.log('\n\n═══════════════════════════════════════');
  console.log('📊 FREE MODELS TEST SUMMARY');
  console.log('═══════════════════════════════════════\n');

  const successful = results.filter(r => r.success);
  const withTools = results.filter(r => r.toolUsesDetected > 0);
  const filesCreated = results.filter(r => r.fileCreated);
  const avgResponseTime = successful.reduce((sum, r) => sum + (r.responseTime || 0), 0) / successful.length;

  console.log(`Total Models Tested: ${results.length}`);
  console.log(`Successful Responses: ${successful.length}`);
  console.log(`Models Using Tools: ${withTools.length}`);
  console.log(`Files Created: ${filesCreated.length}`);
  console.log(`Avg Response Time: ${Math.round(avgResponseTime)}ms\n`);

  console.log('Detailed Results:\n');
  results.forEach(r => {
    const status = r.success ? '✅' : '❌';
    const tools = r.toolUsesDetected > 0 ? `🔧 ${r.toolUsesDetected} tools` : '⚠️  No tools';
    const time = r.responseTime ? `${r.responseTime}ms` : 'N/A';
    console.log(`${status} ${r.model.padEnd(50)} ${tools.padEnd(15)} ${time}`);
    if (r.error) {
      console.log(`   Error: ${r.error}`);
    }
  });

  // Save results to file
  writeFileSync(
    'openrouter-free-models-test-results.json',
    JSON.stringify(results, null, 2)
  );

  console.log('\n\n✅ Results saved to: openrouter-free-models-test-results.json');

  // Summary stats
  const toolSuccessRate = (withTools.length / successful.length * 100).toFixed(1);
  console.log(`\n📈 Tool Usage Success Rate: ${toolSuccessRate}% of successful models`);
}

runTests().catch(err => {
  console.error('Test suite failed:', err);
  process.exit(1);
});
