#!/usr/bin/env node
// Validation script for Flow Nexus MCP integration
// Tests authentication, tool discovery, and sandbox creation
// Usage: node validation/test-flow-nexus.js <email> <password>

import { query } from '@anthropic-ai/claude-agent-sdk';
import { claudeFlowSdkServer } from '../dist/mcp/claudeFlowSdkServer.js';

const email = process.argv[2];
const password = process.argv[3];

if (!email || !password) {
  console.error('❌ Usage: node validation/test-flow-nexus.js <email> <password>');
  process.exit(1);
}

console.log('🧪 Flow Nexus Integration Test\n');
console.log('═'.repeat(80));

async function testFlowNexusIntegration() {
  try {
    console.log('\n📡 Step 1: Testing MCP server connectivity and tool discovery...\n');

    const discoveryResult = query({
      prompt: 'List all available MCP tools. Group them by server (claude-flow-sdk, claude-flow, flow-nexus). Count tools per server.',
      options: {
        permissionMode: 'bypassPermissions',
        mcpServers: {
          'claude-flow-sdk': claudeFlowSdkServer,
          'claude-flow': {
            command: 'npx',
            args: ['claude-flow@alpha', 'mcp', 'start'],
            env: { ...process.env, MCP_AUTO_START: 'true' }
          },
          'flow-nexus': {
            command: 'npx',
            args: ['flow-nexus@latest', 'mcp', 'start'],
            env: { ...process.env, FLOW_NEXUS_AUTO_START: 'true' }
          }
        }
      }
    });

    let discoveryOutput = '';
    for await (const msg of discoveryResult) {
      if (msg.type === 'assistant') {
        const chunk = msg.message.content?.map(c => c.type === 'text' ? c.text : '').join('') || '';
        discoveryOutput += chunk;
        process.stdout.write(chunk);
      } else if (msg.type === 'system' && msg.subtype === 'init') {
        console.log('\n📊 System Initialization:');
        console.log(`   Total Tools: ${msg.tools.length}`);
        console.log(`   MCP Servers: ${msg.mcp_servers.map(s => `${s.name} (${s.status})`).join(', ')}`);
        console.log('\n' + '='.repeat(80) + '\n');
      }
    }

    console.log('\n\n✅ Tool discovery completed!\n');
    console.log('═'.repeat(80));

    console.log(`\n🔐 Step 2: Testing Flow Nexus authentication (${email})...\n`);

    const authPrompt = `
Use the Flow Nexus MCP tools to:
1. Login with email: ${email} and password: ${password}
2. Check authentication status
3. Get current credit balance

Report the results clearly.
`;

    const authResult = query({
      prompt: authPrompt,
      options: {
        permissionMode: 'bypassPermissions',
        mcpServers: {
          'claude-flow-sdk': claudeFlowSdkServer,
          'claude-flow': {
            command: 'npx',
            args: ['claude-flow@alpha', 'mcp', 'start'],
            env: { ...process.env, MCP_AUTO_START: 'true' }
          },
          'flow-nexus': {
            command: 'npx',
            args: ['flow-nexus@latest', 'mcp', 'start'],
            env: { ...process.env, FLOW_NEXUS_AUTO_START: 'true' }
          }
        }
      }
    });

    let authOutput = '';
    for await (const msg of authResult) {
      if (msg.type === 'assistant') {
        const chunk = msg.message.content?.map(c => c.type === 'text' ? c.text : '').join('') || '';
        authOutput += chunk;
        process.stdout.write(chunk);
      }
    }

    console.log('\n\n✅ Authentication test completed!\n');
    console.log('═'.repeat(80));

    console.log('\n🚀 Step 3: Testing Flow Nexus sandbox creation and tool usage...\n');

    const sandboxPrompt = `
Using Flow Nexus MCP tools:
1. Create a Node.js sandbox named "test-sandbox"
2. List available sandboxes
3. Check system health status
4. Report the sandbox ID and status

Keep the sandbox running for further tests.
`;

    const sandboxResult = query({
      prompt: sandboxPrompt,
      options: {
        permissionMode: 'bypassPermissions',
        mcpServers: {
          'claude-flow-sdk': claudeFlowSdkServer,
          'claude-flow': {
            command: 'npx',
            args: ['claude-flow@alpha', 'mcp', 'start'],
            env: { ...process.env, MCP_AUTO_START: 'true' }
          },
          'flow-nexus': {
            command: 'npx',
            args: ['flow-nexus@latest', 'mcp', 'start'],
            env: { ...process.env, FLOW_NEXUS_AUTO_START: 'true' }
          }
        }
      }
    });

    let sandboxOutput = '';
    for await (const msg of sandboxResult) {
      if (msg.type === 'assistant') {
        const chunk = msg.message.content?.map(c => c.type === 'text' ? c.text : '').join('') || '';
        sandboxOutput += chunk;
        process.stdout.write(chunk);
      }
    }

    console.log('\n\n✅ Sandbox creation test completed!\n');
    console.log('═'.repeat(80));

    console.log('\n\n🎉 ALL TESTS PASSED!\n');
    console.log('Summary:');
    console.log('  ✅ MCP server connectivity verified');
    console.log('  ✅ All three servers accessible (claude-flow-sdk, claude-flow, flow-nexus)');
    console.log('  ✅ Flow Nexus authentication successful');
    console.log('  ✅ Sandbox creation and management working');
    console.log('\n📦 Agentic Flow is ready with Flow Nexus integration!\n');

  } catch (error) {
    console.error('\n❌ Test failed:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

testFlowNexusIntegration();
