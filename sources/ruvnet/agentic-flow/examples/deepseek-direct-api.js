#!/usr/bin/env node

/**
 * Direct DeepSeek Chat API Test
 * Uses OpenRouter API directly to test DeepSeek capabilities
 */

const https = require('https');

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const MODEL = 'deepseek/deepseek-chat';

if (!OPENROUTER_API_KEY) {
  console.error('❌ Error: OPENROUTER_API_KEY environment variable not set');
  process.exit(1);
}

console.log('╔════════════════════════════════════════════════════════════════╗');
console.log('║         DeepSeek Chat Direct API Test                         ║');
console.log('╚════════════════════════════════════════════════════════════════╝\n');

console.log('✅ Configuration:');
console.log(`   Model: ${MODEL}`);
console.log(`   API Key: ${OPENROUTER_API_KEY.substring(0, 15)}...`);
console.log(`   Cost: $0.14/$0.28 per 1M tokens\n`);

const requestBody = JSON.stringify({
  model: MODEL,
  messages: [
    {
      role: 'user',
      content: 'Write a simple hello world function in Python with a docstring. Keep it concise.'
    }
  ],
  max_tokens: 300,
  temperature: 0.7
});

const options = {
  hostname: 'openrouter.ai',
  port: 443,
  path: '/api/v1/chat/completions',
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
    'HTTP-Referer': 'https://github.com/ruvnet/agentic-flow',
    'X-Title': 'Agentic Flow - DeepSeek Test',
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(requestBody)
  }
};

console.log('📝 Task: Write a hello world function in Python');
console.log('⏳ Sending request to OpenRouter...\n');

const startTime = Date.now();

const req = https.request(options, (res) => {
  let data = '';

  res.on('data', (chunk) => {
    data += chunk;
  });

  res.on('end', () => {
    const endTime = Date.now();
    const duration = ((endTime - startTime) / 1000).toFixed(2);

    try {
      const response = JSON.parse(data);

      if (response.error) {
        console.error('❌ API Error:', response.error.message);
        process.exit(1);
      }

      console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
      console.log('✅ Response received!');
      console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

      const message = response.choices[0].message;
      console.log('Generated Code:\n');
      console.log(message.content);
      console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

      // Usage stats
      if (response.usage) {
        const inputTokens = response.usage.prompt_tokens || 0;
        const outputTokens = response.usage.completion_tokens || 0;
        const totalTokens = inputTokens + outputTokens;

        // Cost calculation for DeepSeek Chat
        const inputCost = (inputTokens / 1000000) * 0.14;
        const outputCost = (outputTokens / 1000000) * 0.28;
        const totalCost = inputCost + outputCost;

        // Claude cost for comparison
        const claudeInputCost = (inputTokens / 1000000) * 3.00;
        const claudeOutputCost = (outputTokens / 1000000) * 15.00;
        const claudeTotalCost = claudeInputCost + claudeOutputCost;

        const savings = ((claudeTotalCost - totalCost) / claudeTotalCost * 100).toFixed(1);

        console.log('\n📊 Usage Statistics:');
        console.log(`   Input Tokens:  ${inputTokens.toLocaleString()}`);
        console.log(`   Output Tokens: ${outputTokens.toLocaleString()}`);
        console.log(`   Total Tokens:  ${totalTokens.toLocaleString()}`);
        console.log(`   Duration:      ${duration}s`);
        console.log('\n💰 Cost Analysis:');
        console.log(`   DeepSeek Cost: $${totalCost.toFixed(6)}`);
        console.log(`   Claude Cost:   $${claudeTotalCost.toFixed(6)}`);
        console.log(`   Savings:       ${savings}% ($${(claudeTotalCost - totalCost).toFixed(6)})`);
      }

      console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
      console.log('✨ DeepSeek Chat test completed successfully!');
      console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

      console.log('🚀 Next Steps:');
      console.log('   1. Use with agentic-flow agents: --model "deepseek/deepseek-chat"');
      console.log('   2. Try other agents: coder, reviewer, backend-dev, ml-developer');
      console.log('   3. Compare with other models: llama, gemini, claude\n');

    } catch (e) {
      console.error('❌ Failed to parse response:', e.message);
      console.error('Raw response:', data);
      process.exit(1);
    }
  });
});

req.on('error', (e) => {
  console.error('❌ Request failed:', e.message);
  process.exit(1);
});

req.write(requestBody);
req.end();
