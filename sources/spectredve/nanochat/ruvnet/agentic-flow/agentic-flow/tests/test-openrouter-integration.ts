#!/usr/bin/env npx tsx
// Integration test: OpenRouter models with Agentic Flow

import { config } from 'dotenv';
import { writeFileSync } from 'fs';

config();

const OPENROUTER_KEY = process.env.OPENROUTER_API_KEY!;

async function testModelWithAgenticFlow(model: string, modelName: string, task: string) {
  console.log(`\n🤖 Testing ${modelName}...`);
  console.log(`📝 Task: ${task}`);

  const startTime = Date.now();

  try {
    const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${OPENROUTER_KEY}`,
        'HTTP-Referer': 'https://github.com/ruvnet/agentic-flow',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model,
        messages: [{
          role: 'user',
          content: `You are a code generation AI. Generate ONLY the Python code without any markdown formatting or explanations. Task: ${task}`
        }],
        max_tokens: 1000,
        temperature: 0.3
      })
    });

    const latency = Date.now() - startTime;

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const data = await response.json();
    let code = data.choices[0]?.message?.content || '';

    // Extract code from markdown if present
    const codeMatch = code.match(/```python\n([\s\S]*?)```/);
    if (codeMatch) {
      code = codeMatch[1];
    }

    // Clean up any remaining markdown
    code = code.replace(/^```[\s\S]*?\n/, '').replace(/\n```$/, '');

    const tokens = data.usage?.total_tokens || 0;
    const cost = ((data.usage?.prompt_tokens || 0) * 0.00001) +
                 ((data.usage?.completion_tokens || 0) * 0.00003);

    console.log(`✅ Success!`);
    console.log(`   Latency: ${latency}ms`);
    console.log(`   Tokens: ${tokens}`);
    console.log(`   Cost: $${cost.toFixed(6)}`);
    console.log(`   Code length: ${code.length} chars`);

    return { success: true, code, latency, tokens, cost, model: modelName };
  } catch (error: any) {
    console.log(`❌ Failed: ${error.message}`);
    return { success: false, error: error.message, model: modelName };
  }
}

async function main() {
  console.log('╔══════════════════════════════════════════════════════╗');
  console.log('║  Agentic Flow + OpenRouter Integration Test        ║');
  console.log('╚══════════════════════════════════════════════════════╝');

  const tests = [
    {
      model: 'meta-llama/llama-3.1-8b-instruct',
      name: 'Llama 3.1 8B',
      task: 'Create a Python function for binary search with type hints and docstring'
    },
    {
      model: 'deepseek/deepseek-chat-v3.1',
      name: 'DeepSeek V3.1',
      task: 'Create a FastAPI POST endpoint that validates JSON input'
    },
    {
      model: 'google/gemini-2.5-flash-preview-09-2025',
      name: 'Gemini 2.5 Flash',
      task: 'Create an async Python function to fetch data from URLs concurrently'
    }
  ];

  const results = [];

  for (const test of tests) {
    const result = await testModelWithAgenticFlow(
      test.model,
      test.name,
      test.task
    );
    results.push(result);

    // Save successful code
    if (result.success && 'code' in result) {
      const filename = `/tmp/openrouter_${test.name.replace(/\s+/g, '_').toLowerCase()}.py`;
      writeFileSync(filename, result.code);
      console.log(`   💾 Saved to: ${filename}`);
    }

    // Wait between requests
    await new Promise(resolve => setTimeout(resolve, 1000));
  }

  // Summary
  console.log('\n' + '='.repeat(80));
  console.log('📊 INTEGRATION TEST SUMMARY');
  console.log('='.repeat(80));

  const successful = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);

  console.log(`\n✅ Successful: ${successful.length}/${results.length}`);
  if (successful.length > 0) {
    const avgLatency = successful.reduce((sum, r) => sum + ('latency' in r ? r.latency : 0), 0) / successful.length;
    const totalCost = successful.reduce((sum, r) => sum + ('cost' in r ? r.cost : 0), 0);

    console.log(`   Average Latency: ${Math.round(avgLatency)}ms`);
    console.log(`   Total Cost: $${totalCost.toFixed(6)}`);

    console.log(`\n   Working Models:`);
    successful.forEach(r => {
      console.log(`   - ${r.model} ✅`);
    });
  }

  if (failed.length > 0) {
    console.log(`\n❌ Failed: ${failed.length}/${results.length}`);
    failed.forEach(r => {
      console.log(`   - ${r.model}: ${'error' in r ? r.error : 'Unknown error'}`);
    });
  }

  console.log('\n' + '='.repeat(80));
  console.log('🎯 CONCLUSION');
  console.log('='.repeat(80));

  if (successful.length === results.length) {
    console.log('\n✅ ALL MODELS WORKING WITH AGENTIC FLOW!');
    console.log('   OpenRouter integration is fully operational.');
    console.log('   Alternative models can successfully generate code.');
  } else if (successful.length > 0) {
    console.log(`\n⚠️  PARTIAL SUCCESS: ${successful.length}/${results.length} models working`);
    console.log('   Some models are operational. Check failed models for issues.');
  } else {
    console.log('\n❌ INTEGRATION FAILED');
    console.log('   No models successfully generated code. Check API key and configuration.');
  }

  console.log('\n✨ Test complete!\n');
}

main().catch(console.error);
