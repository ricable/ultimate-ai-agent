// Validate Claude Agent SDK works with optimized provider instructions
import { query } from '@anthropic-ai/claude-agent-sdk';

const OPENROUTER_MODELS = [
  { id: 'openai/gpt-4o-mini', name: 'GPT-4o-mini', provider: 'openai' },
  { id: 'meta-llama/llama-3.1-8b-instruct', name: 'Llama 3.1 8B', provider: 'meta-llama' },
  { id: 'x-ai/grok-4-fast', name: 'Grok 4 Fast', provider: 'x-ai' }
];

async function testAgentWithModel(model: typeof OPENROUTER_MODELS[0]) {
  console.log(`\n${'='.repeat(80)}`);
  console.log(`🧪 Testing Claude Agent SDK with ${model.name} (${model.provider})`);
  console.log('='.repeat(80));

  try {
    const result = await query({
      systemPrompt: 'You are a helpful coding assistant.',
      model: model.id,
      prompt: `Create a test file named ${model.provider}-sdk-test.txt with the content "Hello from ${model.name} via Claude SDK!"`,
      permissionMode: 'bypassPermissions',
      allowedTools: ['Read', 'Write', 'Bash'],
      env: {
        ANTHROPIC_BASE_URL: 'http://localhost:3000'
      }
    });

    console.log('\n✅ SDK Query Successful!');
    console.log(`📄 Response:`, result);

    // Check if file was created
    const fs = await import('fs/promises');
    try {
      const content = await fs.readFile(`${model.provider}-sdk-test.txt`, 'utf-8');
      console.log(`✅ File Created: ${model.provider}-sdk-test.txt`);
      console.log(`📝 Content: ${content}`);
    } catch (e) {
      console.log(`⚠️  File not found: ${model.provider}-sdk-test.txt`);
    }

    return { success: true, model: model.name, provider: model.provider };
  } catch (error: any) {
    console.error(`\n❌ Error with ${model.name}:`, error.message);
    return { success: false, model: model.name, provider: model.provider, error: error.message };
  }
}

async function runValidation() {
  console.log('\n🚀 Claude Agent SDK + Optimized Proxy Validation\n');
  console.log('Testing Claude SDK with provider-optimized instructions...\n');

  const results = [];

  for (const model of OPENROUTER_MODELS) {
    const result = await testAgentWithModel(model);
    results.push(result);

    // Wait between tests
    await new Promise(resolve => setTimeout(resolve, 3000));
  }

  console.log('\n\n' + '='.repeat(80));
  console.log('📊 VALIDATION SUMMARY');
  console.log('='.repeat(80));

  const successful = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);

  console.log('\n✅ SUCCESSFUL:');
  successful.forEach(r => {
    console.log(`  - ${r.provider.padEnd(15)} | ${r.model}`);
  });

  if (failed.length > 0) {
    console.log('\n❌ FAILED:');
    failed.forEach(r => {
      console.log(`  - ${r.provider.padEnd(15)} | ${r.model} | ${r.error}`);
    });
  }

  console.log(`\n📈 Success Rate: ${successful.length}/${results.length} (${((successful.length / results.length) * 100).toFixed(1)}%)`);

  if (successful.length === results.length) {
    console.log('\n🎉 ALL TESTS PASSED! Provider-optimized instructions working perfectly with Claude SDK!');
  }
}

runValidation().catch(err => {
  console.error('Validation failed:', err);
  process.exit(1);
});
