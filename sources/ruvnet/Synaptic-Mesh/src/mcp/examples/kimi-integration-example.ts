/**
 * Kimi-K2 Integration Examples
 * Demonstrates how to use the new MCP tools for AI model interactions
 */

import { SynapticMCPServer } from '../synaptic-mcp-server.js';
import { KimiClient, KimiMultiProvider } from '../../js/synaptic-cli/lib/kimi-client.js';

async function demonstrateKimiIntegration() {
  console.log('🤖 Kimi-K2 Integration Demo');
  console.log('================================\n');

  // Initialize the MCP server with Kimi integration
  const mcpServer = new SynapticMCPServer();
  
  // Example 1: Test all providers
  console.log('1. Testing Provider Connections...');
  const providerTest = await mcpServer.executeTool('kimi_provider_test', {
    providers: ['moonshot', 'openrouter', 'local'],
    timeout: 30000
  });
  console.log('Provider Test Results:', JSON.stringify(providerTest, null, 2));
  console.log('');

  // Example 2: List available models
  console.log('2. Listing Available Models...');
  const modelList = await mcpServer.executeTool('kimi_model_list', {
    provider: 'all'
  });
  console.log('Available Models:', JSON.stringify(modelList, null, 2));
  console.log('');

  // Example 3: Basic chat completion
  console.log('3. Basic Chat Completion...');
  const chatResponse = await mcpServer.executeTool('kimi_chat_completion', {
    provider: 'moonshot',
    messages: [
      { role: 'system', content: 'You are a helpful AI assistant specialized in neural networks.' },
      { role: 'user', content: 'Explain what a synaptic neural mesh is in simple terms.' }
    ],
    temperature: 0.7,
    max_tokens: 500
  });
  console.log('Chat Response:', JSON.stringify(chatResponse, null, 2));
  console.log('');

  // Example 4: Chat with tool calling
  console.log('4. Chat with Tool Calling...');
  const toolChat = await mcpServer.executeTool('kimi_chat_completion', {
    provider: 'moonshot',
    messages: [
      { role: 'system', content: 'You are an AI assistant that can analyze neural mesh status. Use the mesh_status tool when asked about mesh information.' },
      { role: 'user', content: 'What is the current status of the neural mesh?' }
    ],
    tools: [
      {
        type: 'function',
        function: {
          name: 'mesh_status',
          description: 'Get the current status of the neural mesh',
          parameters: {
            type: 'object',
            properties: {
              meshId: { type: 'string', description: 'Optional mesh ID to check' },
              metrics: { 
                type: 'array',
                items: { type: 'string' },
                description: 'Metrics to include in status'
              }
            }
          }
        }
      }
    ],
    tool_choice: 'auto',
    temperature: 0.3
  });
  console.log('Tool Chat Response:', JSON.stringify(toolChat, null, 2));

  // Example 5: Execute tool calls if present
  if (toolChat.success && toolChat.tool_calls && toolChat.tool_calls.length > 0) {
    console.log('5. Executing Tool Calls...');
    const toolExecution = await mcpServer.executeTool('kimi_tool_execution', {
      tool_calls: toolChat.tool_calls,
      available_tools: ['mesh_status', 'neuron_spawn', 'synapse_create']
    });
    console.log('Tool Execution Results:', JSON.stringify(toolExecution, null, 2));
    console.log('');
  }

  // Example 6: Context management
  console.log('6. Context Management...');
  const longConversation = [
    { role: 'system', content: 'You are a neural network expert.' },
    { role: 'user', content: 'Tell me about neural networks.' },
    { role: 'assistant', content: 'Neural networks are...' },
    { role: 'user', content: 'How do they learn?' },
    { role: 'assistant', content: 'They learn through...' },
    // Add many more messages to simulate a long conversation
    ...Array(50).fill(null).map((_, i) => [
      { role: 'user', content: `Question ${i + 1}: What about aspect ${i + 1}?` },
      { role: 'assistant', content: `Answer ${i + 1}: Here's information about aspect ${i + 1}...` }
    ]).flat()
  ];

  const contextManagement = await mcpServer.executeTool('kimi_context_management', {
    messages: longConversation,
    context_window: 128000,
    strategy: 'sliding_window'
  });
  console.log('Context Management:', JSON.stringify(contextManagement, null, 2));
  console.log('');

  console.log('✅ Kimi-K2 Integration Demo Complete!');
}

async function demonstrateDirectClientUsage() {
  console.log('\n🔧 Direct Client Usage Demo');
  console.log('============================\n');

  // Create a multi-provider client
  const multiProvider = new KimiMultiProvider();

  // Add providers
  multiProvider.addProvider('moonshot', {
    provider: 'moonshot',
    apiKey: process.env.MOONSHOT_API_KEY,
    model: 'moonshot-v1-128k'
  });

  multiProvider.addProvider('openrouter', {
    provider: 'openrouter',
    apiKey: process.env.OPENROUTER_API_KEY,
    model: 'anthropic/claude-3.5-sonnet'
  });

  // Test all providers
  console.log('Testing all providers...');
  const results = await multiProvider.testAllProviders();
  console.log('Test Results:', JSON.stringify(results, null, 2));

  // Get the best available provider
  const bestProvider = await multiProvider.getBestProvider();
  if (bestProvider) {
    console.log(`\nUsing best provider: ${bestProvider.name}`);
    
    // Make a chat request
    const response = await bestProvider.client.chatCompletion({
      model: 'moonshot-v1-128k',
      messages: [
        { role: 'user', content: 'Hello! How can AI help with neural mesh orchestration?' }
      ],
      temperature: 0.7,
      max_tokens: 200
    });

    console.log('Response:', JSON.stringify(response.choices[0]?.message, null, 2));
  } else {
    console.log('No providers available');
  }
}

async function demonstrateStreamingExample() {
  console.log('\n🌊 Streaming Chat Demo');
  console.log('======================\n');

  const client = new KimiClient({
    provider: 'moonshot',
    apiKey: process.env.MOONSHOT_API_KEY,
    model: 'moonshot-v1-128k'
  });

  console.log('Starting streaming chat...');
  
  await client.streamChatCompletion({
    model: 'moonshot-v1-128k',
    messages: [
      { role: 'user', content: 'Explain the concept of a synaptic neural mesh in detail.' }
    ],
    stream: true
  }, (chunk) => {
    if (chunk.choices && chunk.choices[0]?.delta?.content) {
      process.stdout.write(chunk.choices[0].delta.content);
    }
  });

  console.log('\n\n✅ Streaming complete!');
}

// Run the demos
async function main() {
  try {
    await demonstrateKimiIntegration();
    await demonstrateDirectClientUsage();
    await demonstrateStreamingExample();
  } catch (error) {
    console.error('Demo failed:', error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

export {
  demonstrateKimiIntegration,
  demonstrateDirectClientUsage,
  demonstrateStreamingExample
};