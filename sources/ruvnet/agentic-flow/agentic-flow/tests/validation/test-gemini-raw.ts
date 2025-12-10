// Raw test of Gemini proxy with tool instruction parsing
import { logger } from './src/utils/logger.js';

const GEMINI_API_KEY = process.env.GOOGLE_GEMINI_API_KEY;
const PROXY_URL = 'http://localhost:3001';

async function testGeminiRaw() {
  console.log('🧪 Testing Gemini proxy with tool instructions (RAW)...\n');

  const anthropicRequest = {
    model: 'gemini-2.0-flash-exp',
    max_tokens: 1024,
    system: 'You are a helpful coding assistant.',
    messages: [
      {
        role: 'user',
        content: 'Create a hello.js file with a simple hello world function that prints "Hello from Gemini!"'
      }
    ]
  };

  console.log('📤 Sending Anthropic-format request to Gemini proxy...\n');
  console.log('Request:', JSON.stringify(anthropicRequest, null, 2));

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
      console.error(`❌ Proxy error (${response.status}):`, error);
      process.exit(1);
    }

    const result = await response.json();

    console.log('\n📥 Received Anthropic-format response:\n');
    console.log(JSON.stringify(result, null, 2));

    // Check for tool uses in content
    const toolUses = result.content?.filter((c: any) => c.type === 'tool_use') || [];

    console.log('\n📊 Analysis:');
    console.log(`   Total content blocks: ${result.content?.length || 0}`);
    console.log(`   Tool uses detected: ${toolUses.length}`);

    if (toolUses.length > 0) {
      console.log('\n✅ SUCCESS! Tool uses detected:');
      toolUses.forEach((tool: any, i: number) => {
        console.log(`\n   Tool ${i + 1}:`);
        console.log(`     Name: ${tool.name}`);
        console.log(`     Input:`, JSON.stringify(tool.input, null, 6));
      });
    } else {
      console.log('\n⚠️  No tool uses detected. Response text:');
      const textBlocks = result.content?.filter((c: any) => c.type === 'text') || [];
      textBlocks.forEach((block: any) => {
        console.log(`\n${block.text}`);
      });
    }

  } catch (error: any) {
    console.error('\n❌ Test failed:', error.message);
    process.exit(1);
  }
}

testGeminiRaw().catch(err => {
  console.error(err);
  process.exit(1);
});
