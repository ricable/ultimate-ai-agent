# Synaptic Neural Mesh MCP Server

## Kimi-K2 Integration

The Synaptic Neural Mesh MCP Server now includes comprehensive integration with Kimi-K2 AI models through multiple providers.

### Features

- **Multi-Provider Support**: Moonshot AI, OpenRouter, and local LLM support
- **Tool Calling**: Full function calling capabilities
- **128k Context Window**: Extended context management for long conversations
- **Streaming Responses**: Real-time response streaming
- **Error Handling**: Robust error handling and fallback mechanisms
- **Context Management**: Intelligent context window management

### Providers

#### Moonshot AI
- Native Kimi-K2 models
- 128k context window
- Optimized for Chinese and English
- Best performance for neural mesh tasks

#### OpenRouter
- Access to multiple AI providers
- Claude, GPT-4, Llama, and more
- Up to 200k context window
- Fallback options

#### Local (Ollama)
- Privacy-focused local inference
- No API costs
- Offline capabilities
- Hardware-dependent performance

### Installation

1. **Environment Setup**:
```bash
# Required for Moonshot AI
export MOONSHOT_API_KEY="your-moonshot-api-key"

# Required for OpenRouter
export OPENROUTER_API_KEY="your-openrouter-api-key"

# Optional for local LLM
export LOCAL_LLM_URL="http://localhost:11434/v1"
```

2. **Install Dependencies**:
```bash
npm install
```

3. **Start MCP Server**:
```bash
npm run start:mcp
```

### Usage

#### MCP Tools

##### Chat Completion
```typescript
await mcpServer.executeTool('kimi_chat_completion', {
  provider: 'moonshot',
  messages: [
    { role: 'system', content: 'You are a neural network expert.' },
    { role: 'user', content: 'Explain synaptic neural meshes.' }
  ],
  temperature: 0.7,
  max_tokens: 1000
});
```

##### Tool Calling
```typescript
await mcpServer.executeTool('kimi_chat_completion', {
  provider: 'moonshot',
  messages: [
    { role: 'user', content: 'Get the current mesh status.' }
  ],
  tools: [{
    type: 'function',
    function: {
      name: 'mesh_status',
      description: 'Get neural mesh status',
      parameters: {
        type: 'object',
        properties: {
          meshId: { type: 'string' }
        }
      }
    }
  }],
  tool_choice: 'auto'
});
```

##### Context Management
```typescript
await mcpServer.executeTool('kimi_context_management', {
  messages: longConversation,
  context_window: 128000,
  strategy: 'sliding_window'
});
```

##### Provider Testing
```typescript
await mcpServer.executeTool('kimi_provider_test', {
  providers: ['moonshot', 'openrouter', 'local'],
  timeout: 30000
});
```

##### Model Listing
```typescript
await mcpServer.executeTool('kimi_model_list', {
  provider: 'all'
});
```

#### Direct Client Usage

```typescript
import { KimiClient, KimiMultiProvider } from '../js/synaptic-cli/lib/kimi-client.js';

// Single provider
const client = new KimiClient({
  provider: 'moonshot',
  apiKey: process.env.MOONSHOT_API_KEY,
  model: 'moonshot-v1-128k'
});

const response = await client.chatCompletion({
  model: 'moonshot-v1-128k',
  messages: [{ role: 'user', content: 'Hello!' }]
});

// Multi-provider with fallback
const multiProvider = new KimiMultiProvider();
multiProvider.addProvider('moonshot', { ... });
multiProvider.addProvider('openrouter', { ... });

const bestProvider = await multiProvider.getBestProvider();
```

#### Streaming Responses

```typescript
await client.streamChatCompletion({
  model: 'moonshot-v1-128k',
  messages: [{ role: 'user', content: 'Explain neural networks...' }],
  stream: true
}, (chunk) => {
  if (chunk.choices?.[0]?.delta?.content) {
    process.stdout.write(chunk.choices[0].delta.content);
  }
});
```

### Neural Mesh Integration

The Kimi integration seamlessly works with existing neural mesh tools:

```typescript
// AI can call neural mesh functions
const response = await client.chatCompletion({
  messages: [
    { role: 'user', content: 'Create a new neural mesh and add 10 neurons' }
  ],
  tools: [
    { type: 'function', function: { name: 'mesh_initialize', ... } },
    { type: 'function', function: { name: 'neuron_spawn', ... } }
  ]
});

// Execute the tool calls
if (response.choices[0].message.tool_calls) {
  for (const toolCall of response.choices[0].message.tool_calls) {
    const result = await client.executeToolCall(toolCall, availableTools);
    console.log('Tool result:', result);
  }
}
```

### Configuration

Provider configuration is stored in `config/kimi-providers.json`:

```json
{
  "providers": {
    "moonshot": {
      "baseUrl": "https://api.moonshot.cn/v1",
      "defaultModel": "moonshot-v1-128k",
      "contextWindow": 128000
    }
  }
}
```

### Error Handling

The integration includes comprehensive error handling:

- **Network timeouts**: Configurable timeout with automatic retries
- **API errors**: Graceful handling of 4xx/5xx responses
- **Fallback providers**: Automatic fallback to alternative providers
- **Context limits**: Intelligent context window management
- **Rate limiting**: Built-in rate limit handling

### Testing

Run the test suite:

```bash
npm test src/mcp/tests/kimi-integration.test.ts
```

Run examples:

```bash
npm run example:kimi
```

### Performance

- **Token efficiency**: Smart context management reduces token usage
- **Response speed**: Optimized for low latency responses
- **Concurrent requests**: Support for parallel API calls
- **Caching**: Response caching for repeated queries

### Security

- **API key protection**: Environment variable storage
- **Input validation**: Comprehensive input sanitization
- **Rate limiting**: Built-in protection against abuse
- **Local option**: Privacy-focused local inference

### Troubleshooting

#### Common Issues

1. **API Key Issues**:
```bash
# Check environment variables
echo $MOONSHOT_API_KEY
echo $OPENROUTER_API_KEY
```

2. **Local LLM Connection**:
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags
```

3. **Context Window Exceeded**:
- Use `kimi_context_management` tool
- Implement conversation summarization
- Split large requests

4. **Provider Unavailable**:
- Test providers with `kimi_provider_test`
- Configure fallback providers
- Check network connectivity

#### Debug Mode

Enable debug logging:

```bash
export DEBUG=kimi:*
npm run start:mcp
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### License

MIT License - see LICENSE file for details.