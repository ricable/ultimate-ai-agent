// Test Claude Agent SDK with OpenRouter proxy using baseURL
import { query } from "@anthropic-ai/claude-agent-sdk";

async function testOpenRouterProxy() {
  console.log('🧪 Testing Claude Agent SDK with OpenRouter proxy...\n');

  // Configure environment for OpenRouter routing
  const originalApiKey = process.env.ANTHROPIC_API_KEY;
  const originalBaseUrl = process.env.ANTHROPIC_BASE_URL;

  try {
    // Set up OpenRouter configuration via our proxy
    // The proxy translates Anthropic API format to OpenRouter format
    process.env.ANTHROPIC_API_KEY = 'proxy-key'; // Proxy handles the real key
    process.env.ANTHROPIC_BASE_URL = 'http://localhost:3000';

    console.log('📡 Configuration:');
    console.log(`   Base URL: ${process.env.ANTHROPIC_BASE_URL}`);
    console.log(`   API Key: ${process.env.ANTHROPIC_API_KEY?.substring(0, 15)}...`);
    console.log(`   Model: meta-llama/llama-3.1-8b-instruct\n`);

    console.log('🚀 Sending query via SDK...\n');

    const result = query({
      prompt: "Say 'Hello from OpenRouter!' and nothing else.",
      options: {
        model: 'meta-llama/llama-3.1-8b-instruct',
        permissionMode: 'bypassPermissions',
        mcpServers: {} // No MCP servers for simple test
      }
    });

    let output = '';
    for await (const msg of result) {
      if (msg.type === 'assistant') {
        const chunk = msg.message.content?.map((c: any) =>
          c.type === 'text' ? c.text : ''
        ).join('') || '';
        output += chunk;
        process.stdout.write(chunk);
      }
    }

    console.log('\n\n✅ SUCCESS! Claude Agent SDK successfully routed to OpenRouter');
    console.log(`📝 Response length: ${output.length} characters`);

  } catch (error: any) {
    console.error('\n❌ FAILED:', error.message);
    throw error;
  } finally {
    // Restore original environment
    if (originalApiKey) process.env.ANTHROPIC_API_KEY = originalApiKey;
    else delete process.env.ANTHROPIC_API_KEY;

    if (originalBaseUrl) process.env.ANTHROPIC_BASE_URL = originalBaseUrl;
    else delete process.env.ANTHROPIC_BASE_URL;
  }
}

testOpenRouterProxy().catch(err => {
  console.error(err);
  process.exit(1);
});
