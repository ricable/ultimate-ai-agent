// Test top 20 OpenRouter models with tool instruction parsing
import { writeFileSync } from 'fs';

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const PROXY_URL = 'http://localhost:3000';

// Top 20 models from https://openrouter.ai/models (by token usage)
const TOP_20_MODELS = [
  { id: 'x-ai/grok-code-fast-1', name: 'Grok Code Fast 1', provider: 'x-ai', free: false },
  { id: 'x-ai/grok-4-fast:free', name: 'Grok 4 Fast (free)', provider: 'x-ai', free: true },
  { id: 'anthropic/claude-sonnet-4', name: 'Claude Sonnet 4', provider: 'anthropic', free: false },
  { id: 'google/gemini-2.5-flash', name: 'Gemini 2.5 Flash', provider: 'google', free: false },
  { id: 'anthropic/claude-sonnet-4.5', name: 'Claude Sonnet 4.5', provider: 'anthropic', free: false },
  { id: 'deepseek/deepseek-chat-v3.1:free', name: 'DeepSeek V3.1 (free)', provider: 'deepseek', free: true },
  { id: 'openai/gpt-4.1-mini', name: 'GPT-4.1 Mini', provider: 'openai', free: false },
  { id: 'google/gemini-2.0-flash-exp:free', name: 'Gemini 2.0 Flash', provider: 'google', free: true },
  { id: 'google/gemini-2.5-pro', name: 'Gemini 2.5 Pro', provider: 'google', free: false },
  { id: 'google/gemini-2.5-flash-lite', name: 'Gemini 2.5 Flash Lite', provider: 'google', free: false },
  { id: 'deepseek/deepseek-v3.2-exp', name: 'DeepSeek V3.2 (exp)', provider: 'deepseek', free: false },
  { id: 'google/gemma-2-27b-it', name: 'Gemma 2 27B', provider: 'google', free: false },
  { id: 'openai/gpt-5', name: 'GPT-5', provider: 'openai', free: false },
  { id: 'anthropic/claude-3.7-sonnet', name: 'Claude 3.7 Sonnet', provider: 'anthropic', free: false },
  { id: 'openai/gpt-oss-120b', name: 'gpt-oss-120b', provider: 'openai', free: false },
  { id: 'openai/gpt-oss-20b', name: 'gpt-oss-20b', provider: 'openai', free: false },
  { id: 'x-ai/grok-4-fast', name: 'Grok 4 Fast', provider: 'x-ai', free: false },
  { id: 'openai/gpt-4o-mini', name: 'GPT-4o-mini', provider: 'openai', free: false },
  { id: 'meta-llama/llama-3.1-8b-instruct', name: 'Llama 3.1 8B Instruct', provider: 'meta-llama', free: false },
  { id: 'zhipu/glm-4.6', name: 'GLM 4.6', provider: 'z-ai', free: false }
];

interface TestResult {
  rank: number;
  modelId: string;
  modelName: string;
  provider: string;
  free: boolean;
  success: boolean;
  toolUsesDetected: number;
  fileCreated: boolean;
  responseTime?: number;
  error?: string;
  responseText?: string;
  nativeToolCalling?: boolean;
}

async function testModel(model: typeof TOP_20_MODELS[0], rank: number): Promise<TestResult> {
  console.log(`\n[${rank}/20] 📦 Testing: ${model.name} (${model.id})`);

  const anthropicRequest = {
    model: model.id,
    max_tokens: 512,
    system: 'You are a helpful coding assistant.',
    messages: [
      {
        role: 'user',
        content: `Create a test-${rank}.txt file with the text "Hello from ${model.name}!"`
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
      const errorMsg = error.substring(0, 200);
      console.error(`  ❌ HTTP ${response.status}: ${errorMsg}`);
      return {
        rank,
        modelId: model.id,
        modelName: model.name,
        provider: model.provider,
        free: model.free,
        success: false,
        toolUsesDetected: 0,
        fileCreated: false,
        responseTime,
        error: `HTTP ${response.status}`
      };
    }

    const result = await response.json();
    const toolUses = result.content?.filter((c: any) => c.type === 'tool_use') || [];
    const textContent = result.content?.filter((c: any) => c.type === 'text')
      .map((c: any) => c.text).join(' ').substring(0, 150) || '';

    // Detect if model has native tool calling (Anthropic, some OpenAI)
    const nativeToolCalling = model.provider === 'anthropic' ||
                              (model.provider === 'openai' && model.id.includes('gpt-4'));

    console.log(`  ✅ Response (${responseTime}ms)`);
    console.log(`  📊 Tool uses: ${toolUses.length}`);
    if (toolUses.length === 0 && textContent) {
      console.log(`  📝 Text: ${textContent}...`);
    }

    return {
      rank,
      modelId: model.id,
      modelName: model.name,
      provider: model.provider,
      free: model.free,
      success: true,
      toolUsesDetected: toolUses.length,
      fileCreated: toolUses.some((t: any) => t.name === 'Write'),
      responseTime,
      responseText: textContent,
      nativeToolCalling
    };

  } catch (error: any) {
    const responseTime = Date.now() - startTime;
    console.error(`  ❌ Error: ${error.message}`);
    return {
      rank,
      modelId: model.id,
      modelName: model.name,
      provider: model.provider,
      free: model.free,
      success: false,
      toolUsesDetected: 0,
      fileCreated: false,
      responseTime,
      error: error.message
    };
  }
}

function generateMatrix(results: TestResult[]): string {
  let matrix = `# Top 20 OpenRouter Models - Tool Calling Functionality Matrix

Generated: ${new Date().toISOString()}

## Summary Statistics

- **Total Models Tested:** ${results.length}
- **Successful Responses:** ${results.filter(r => r.success).length}
- **Models Using Tools:** ${results.filter(r => r.toolUsesDetected > 0).length}
- **Tool Success Rate:** ${((results.filter(r => r.toolUsesDetected > 0).length / results.filter(r => r.success).length) * 100).toFixed(1)}%
- **Free Models:** ${results.filter(r => r.free).length}
- **Avg Response Time:** ${Math.round(results.filter(r => r.success).reduce((sum, r) => sum + (r.responseTime || 0), 0) / results.filter(r => r.success).length)}ms

## Functionality Matrix

| Rank | Model | Provider | Free | Status | Tools | Native | Response Time | Notes |
|------|-------|----------|------|--------|-------|--------|---------------|-------|
`;

  results.forEach(r => {
    const status = r.success ? '✅' : '❌';
    const tools = r.toolUsesDetected > 0 ? `🔧 ${r.toolUsesDetected}` : '⚠️ 0';
    const native = r.nativeToolCalling ? '✓' : '✗';
    const free = r.free ? '✓' : '✗';
    const time = r.responseTime ? `${r.responseTime}ms` : 'N/A';
    const notes = r.error ? `Error: ${r.error}` : (r.toolUsesDetected === 0 && r.responseText ? `Text only: ${r.responseText.substring(0, 40)}...` : 'OK');

    matrix += `| ${r.rank} | ${r.modelName} | ${r.provider} | ${free} | ${status} | ${tools} | ${native} | ${time} | ${notes} |\n`;
  });

  matrix += `\n## Models Requiring Custom Instructions\n\nBased on test results, the following models may need model-specific tool instructions:\n\n`;

  const needsCustom = results.filter(r => r.success && r.toolUsesDetected === 0);
  needsCustom.forEach(r => {
    matrix += `### ${r.modelName} (${r.modelId})\n`;
    matrix += `- **Provider:** ${r.provider}\n`;
    matrix += `- **Issue:** Responded with text but didn't use structured commands\n`;
    matrix += `- **Response:** ${r.responseText}\n`;
    matrix += `- **Recommendation:** Create provider-specific prompt template\n\n`;
  });

  matrix += `\n## Provider-Specific Recommendations\n\n`;

  const byProvider = results.reduce((acc, r) => {
    if (!acc[r.provider]) acc[r.provider] = [];
    acc[r.provider].push(r);
    return acc;
  }, {} as Record<string, TestResult[]>);

  Object.entries(byProvider).forEach(([provider, models]) => {
    const toolSuccess = models.filter(m => m.toolUsesDetected > 0).length;
    const total = models.filter(m => m.success).length;
    const rate = total > 0 ? ((toolSuccess / total) * 100).toFixed(1) : '0.0';

    matrix += `### ${provider}\n`;
    matrix += `- **Tool Success Rate:** ${rate}% (${toolSuccess}/${total})\n`;
    matrix += `- **Models Tested:** ${models.map(m => m.modelName).join(', ')}\n`;
    if (toolSuccess < total) {
      matrix += `- **Action:** Consider provider-specific instruction template\n`;
    }
    matrix += `\n`;
  });

  return matrix;
}

async function runTests() {
  console.log('🧪 Testing Top 20 OpenRouter Models\n');
  console.log('This will test the most popular models by token usage...\n');

  const results: TestResult[] = [];

  for (let i = 0; i < TOP_20_MODELS.length; i++) {
    const model = TOP_20_MODELS[i];
    const result = await testModel(model, i + 1);
    results.push(result);

    // Rate limit - wait between requests
    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  console.log('\n\n' + '═'.repeat(80));
  console.log('📊 TEST COMPLETE');
  console.log('═'.repeat(80) + '\n');

  // Generate and save matrix
  const matrix = generateMatrix(results);

  writeFileSync('TOP20_MODELS_MATRIX.md', matrix);
  writeFileSync('top20-models-test-results.json', JSON.stringify(results, null, 2));

  console.log('✅ Results saved to:');
  console.log('   - TOP20_MODELS_MATRIX.md (functionality matrix)');
  console.log('   - top20-models-test-results.json (detailed results)');

  // Print quick summary
  console.log(`\n📈 Quick Summary:`);
  console.log(`   Successful: ${results.filter(r => r.success).length}/20`);
  console.log(`   Using Tools: ${results.filter(r => r.toolUsesDetected > 0).length}/20`);
  console.log(`   Need Custom Instructions: ${results.filter(r => r.success && r.toolUsesDetected === 0).length}`);
}

runTests().catch(err => {
  console.error('Test suite failed:', err);
  process.exit(1);
});
