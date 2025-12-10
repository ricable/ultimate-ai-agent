// Test MCP server connection with Claude Agent SDK
import { query } from "@anthropic-ai/claude-agent-sdk";
import { claudeFlowSdkServer } from "../src/mcp/claudeFlowSdkServer.js";

async function testMCPConnection() {
  console.log('🧪 Testing MCP Server Connection...\n');

  // Test 1: In-SDK MCP server (should work)
  console.log('Test 1: In-SDK MCP Server (claude-flow-sdk)');
  try {
    const result = query({
      prompt: "List all available MCP tools. What tools do you have access to from MCP servers?",
      options: {
        systemPrompt: "You are a helpful assistant. List all MCP tools available to you.",
        model: process.env.COMPLETION_MODEL || 'claude-sonnet-4-5-20250929',
        permissionMode: 'bypassPermissions',
        mcpServers: {
          'claude-flow-sdk': claudeFlowSdkServer
        }
      }
    });

    let output = '';
    for await (const msg of result) {
      if (msg.type === 'assistant') {
        const chunk = msg.message.content?.map((c: any) => c.type === 'text' ? c.text : '').join('') || '';
        output += chunk;
      }
    }

    console.log('✅ In-SDK Server Response:', output.substring(0, 500));
  } catch (error: any) {
    console.error('❌ In-SDK Server Error:', error.message);
  }

  console.log('\n---\n');

  // Test 2: External MCP server via stdio (current issue)
  console.log('Test 2: External stdio MCP Server (claude-flow)');
  try {
    const result = query({
      prompt: "Use the claude-flow MCP server to store test-key=test-value in memory, then retrieve it.",
      options: {
        systemPrompt: "You are a helpful assistant with access to MCP memory tools.",
        model: process.env.COMPLETION_MODEL || 'claude-sonnet-4-5-20250929',
        permissionMode: 'bypassPermissions',
        mcpServers: {
          'claude-flow': {
            type: 'stdio',
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
        const chunk = msg.message.content?.map((c: any) => c.type === 'text' ? c.text : '').join('') || '';
        output += chunk;
      }
    }

    console.log('✅ External Server Response:', output.substring(0, 500));
  } catch (error: any) {
    console.error('❌ External Server Error:', error.message);
  }

  console.log('\n---\n');

  // Test 3: Check what tools are actually available
  console.log('Test 3: Tool Availability Check');
  try {
    const result = query({
      prompt: "Please tell me exactly which tools you have access to. List them by name.",
      options: {
        systemPrompt: "List all tools available to you, including MCP tools.",
        model: process.env.COMPLETION_MODEL || 'claude-sonnet-4-5-20250929',
        permissionMode: 'bypassPermissions',
        mcpServers: {
          'claude-flow-sdk': claudeFlowSdkServer,
          'claude-flow': {
            type: 'stdio',
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
        const chunk = msg.message.content?.map((c: any) => c.type === 'text' ? c.text : '').join('') || '';
        output += chunk;
      }
    }

    console.log('✅ Available Tools:', output);
  } catch (error: any) {
    console.error('❌ Tool Check Error:', error.message);
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  testMCPConnection().catch(console.error);
}
