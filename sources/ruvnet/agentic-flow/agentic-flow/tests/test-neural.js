// Test neural_train tool with bypassPermissions
import { query } from '@anthropic-ai/claude-agent-sdk';
import { claudeFlowSdkServer } from './dist/mcp/claudeFlowSdkServer.js';

async function testNeuralTool() {
  console.log('🧠 Testing neural_train with bypassPermissions mode...\n');

  try {
    const result = query({
      prompt: 'Use the neural_train tool to train a convergent cognitive pattern for 3 iterations. Report the training results.',
      options: {
        permissionMode: 'bypassPermissions', // Auto-approve tools
        mcpServers: {
          'claude-flow-sdk': claudeFlowSdkServer,
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

    for await (const msg of result) {
      if (msg.type === 'assistant') {
        const chunk = msg.message.content?.map(c => c.type === 'text' ? c.text : '').join('') || '';
        process.stdout.write(chunk);
      } else if (msg.type === 'system' && msg.subtype === 'init') {
        console.log(`📊 Initialized with ${msg.tools.length} tools, permissionMode: ${msg.permissionMode}\n`);
        console.log('='.repeat(80) + '\n');
      }
    }

    console.log('\n\n✅ Neural tool test completed!\n');
  } catch (error) {
    console.error('❌ Error:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

testNeuralTool();
