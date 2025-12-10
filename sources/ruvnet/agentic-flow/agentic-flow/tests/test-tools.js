// Test script to verify all 87 MCP tools are accessible via SDK
import { query } from '@anthropic-ai/claude-agent-sdk';
import { claudeFlowSdkServer } from './dist/mcp/claudeFlowSdkServer.js';

async function testToolDiscovery() {
  console.log('🔍 Testing MCP tool discovery via Claude Agent SDK...\n');

  try {
    const result = query({
      prompt: 'List all available tools from both MCP servers. Show tool names grouped by server.',
      options: {
        permissionMode: 'bypassPermissions',
        mcpServers: {
          // In-SDK server: 7 basic tools
          'claude-flow-sdk': claudeFlowSdkServer,

          // Full MCP server: All tools via subprocess
          'claude-flow': {
            command: 'npx',
            args: ['claude-flow@alpha', 'mcp', 'start'],
            env: {
              ...process.env,
              MCP_AUTO_START: 'true'
            }
          }
        }
      }
    });

    let output = '';
    for await (const msg of result) {
      if (msg.type === 'assistant') {
        const chunk = msg.message.content?.map(c => c.type === 'text' ? c.text : '').join('') || '';
        output += chunk;
        process.stdout.write(chunk);
      } else if (msg.type === 'system' && msg.subtype === 'init') {
        console.log('\\n📊 System Initialization:');
        console.log(`   Tools: ${msg.tools.length} total`);
        console.log(`   MCP Servers: ${msg.mcp_servers.map(s => `${s.name} (${s.status})`).join(', ')}`);
        console.log('\\n' + '='.repeat(80) + '\\n');
      }
    }

    console.log('\\n\\n✅ Tool discovery test completed!\\n');
  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
}

testToolDiscovery();
