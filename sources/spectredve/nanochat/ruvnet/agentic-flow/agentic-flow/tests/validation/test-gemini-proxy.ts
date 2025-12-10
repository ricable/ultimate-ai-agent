// Test Claude Agent SDK with Gemini proxy using baseURL
import { query } from "@anthropic-ai/claude-agent-sdk";

async function testGeminiProxy() {
  console.log('🧪 Testing Claude Agent SDK with Gemini proxy...\n');

  // Configure environment for Gemini routing
  const originalApiKey = process.env.ANTHROPIC_API_KEY;
  const originalBaseUrl = process.env.ANTHROPIC_BASE_URL;

  try {
    // Set up Gemini configuration via our proxy
    // The proxy translates Anthropic API format to Gemini format
    process.env.ANTHROPIC_API_KEY = 'proxy-key'; // Proxy handles the real key
    process.env.ANTHROPIC_BASE_URL = 'http://localhost:3001';

    console.log('📡 Configuration:');
    console.log(`   Base URL: ${process.env.ANTHROPIC_BASE_URL}`);
    console.log(`   API Key: ${process.env.ANTHROPIC_API_KEY?.substring(0, 15)}...`);
    console.log(`   Model: gemini-2.0-flash-exp\n`);

    console.log('🚀 Sending query via SDK...\n');

    const result = query({
      prompt: "Say 'Hello from Gemini!' and nothing else.",
      options: {
        model: 'gemini-2.0-flash-exp',
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

    console.log('\n\n✅ SUCCESS! Claude Agent SDK successfully routed to Gemini');
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

testGeminiProxy().catch(err => {
  console.error(err);
  process.exit(1);
});
