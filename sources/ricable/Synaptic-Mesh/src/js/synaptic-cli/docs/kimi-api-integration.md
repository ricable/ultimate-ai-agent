# Kimi-K2 API Integration Guide

## Overview

The Synaptic Neural Mesh CLI now includes real integration with Kimi-K2 AI models through two major providers:

- **Moonshot AI** - Direct access to Kimi models with 128k context window
- **OpenRouter** - Alternative provider with multiple model options

## Features

### 🚀 Real API Integration
- **128k Context Window** - Full document analysis and processing
- **Real-time Responses** - Actual AI-powered interactions
- **Tool Calling** - Function calling capabilities
- **Streaming Support** - Real-time response streaming (coming soon)
- **Rate Limiting** - Built-in retry logic and rate limiting
- **Error Handling** - Comprehensive error recovery

### 🎯 Supported Operations
- **Chat Interface** - Interactive conversations with AI
- **Code Generation** - AI-powered code creation
- **File Analysis** - Deep code analysis and suggestions
- **Document Processing** - Text analysis and summarization
- **Multi-modal** - Text and image processing capabilities

## Quick Start

### 1. Install Dependencies

```bash
cd /workspaces/Synaptic-Neural-Mesh/src/js/synaptic-cli
npm install
```

### 2. Configure API Keys

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# For Moonshot AI (Recommended)
KIMI_PROVIDER=moonshot
KIMI_API_KEY=your_moonshot_api_key_here

# For OpenRouter (Alternative)
# KIMI_PROVIDER=openrouter
# OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 3. Initialize Kimi Integration

```bash
# Interactive setup
synaptic-mesh kimi init --interactive

# Or quick setup
synaptic-mesh kimi init --api-key YOUR_API_KEY --provider moonshot
```

### 4. Connect and Test

```bash
# Connect to API
synaptic-mesh kimi connect

# Test with a simple chat
synaptic-mesh kimi chat "Hello, can you help me with JavaScript?"

# Generate some code
synaptic-mesh kimi generate --prompt "Create a REST API endpoint" --lang javascript

# Analyze a file
synaptic-mesh kimi analyze --file ./src/commands/kimi.ts --type quality
```

## API Providers

### Moonshot AI (Recommended)

**Advantages:**
- Direct access to Kimi models
- 128k context window
- Optimized for Chinese and English
- Cost-effective pricing

**Getting Started:**
1. Visit [Moonshot AI Platform](https://platform.moonshot.cn/)
2. Create an account and get your API key
3. Set `KIMI_PROVIDER=moonshot` and `KIMI_API_KEY=your_key`

**Available Models:**
- `moonshot-v1-128k` - Full 128k context (Recommended)
- `moonshot-v1-32k` - 32k context for faster responses
- `moonshot-v1-8k` - 8k context for simple tasks

### OpenRouter

**Advantages:**
- Multiple model options (Claude, GPT, etc.)
- Unified API for different providers
- Transparent pricing

**Getting Started:**
1. Visit [OpenRouter](https://openrouter.ai/)
2. Create an account and get your API key
3. Set `KIMI_PROVIDER=openrouter` and `OPENROUTER_API_KEY=your_key`

**Available Models:**
- `anthropic/claude-3.5-sonnet` - High-quality reasoning
- `anthropic/claude-3-opus` - Maximum capability
- `anthropic/claude-3-haiku` - Fast responses

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KIMI_PROVIDER` | API provider (moonshot/openrouter) | moonshot |
| `KIMI_API_KEY` | Primary API key | - |
| `KIMI_MODEL_VERSION` | Model to use | moonshot-v1-128k |
| `KIMI_MAX_TOKENS` | Maximum tokens per request | 128000 |
| `KIMI_TEMPERATURE` | Model temperature (0.0-2.0) | 0.7 |
| `KIMI_TIMEOUT` | Request timeout in ms | 60000 |
| `KIMI_RETRY_ATTEMPTS` | Number of retry attempts | 3 |

### Feature Flags

| Variable | Description | Default |
|----------|-------------|---------|
| `KIMI_FEATURE_MULTIMODAL` | Enable multi-modal processing | true |
| `KIMI_FEATURE_CODE_GENERATION` | Enable code generation | true |
| `KIMI_FEATURE_DOCUMENT_ANALYSIS` | Enable document analysis | true |
| `KIMI_FEATURE_STREAMING` | Enable streaming responses | true |
| `KIMI_FEATURE_TOOL_CALLING` | Enable tool calling | true |

## Usage Examples

### Interactive Chat

```bash
# Start interactive session
synaptic-mesh kimi chat --interactive

# Single message
synaptic-mesh kimi chat "Explain async/await in JavaScript"

# Include file context
synaptic-mesh kimi chat "Review this code" --file ./src/utils/helper.js
```

### Code Generation

```bash
# Generate a function
synaptic-mesh kimi generate \
  --prompt "Create a function to validate email addresses" \
  --lang javascript \
  --output ./src/utils/validation.js

# Generate with optimization
synaptic-mesh kimi generate \
  --prompt "Create a React component for user profile" \
  --lang javascript \
  --optimize \
  --template component
```

### File Analysis

```bash
# Quality analysis
synaptic-mesh kimi analyze \
  --file ./src/components/App.tsx \
  --type quality \
  --save-report analysis-report.json

# Security analysis
synaptic-mesh kimi analyze \
  --directory ./src \
  --type security \
  --format json

# Performance analysis
synaptic-mesh kimi analyze \
  --file ./src/utils/performance.js \
  --type performance
```

### Advanced Features

```bash
# Check status and health
synaptic-mesh kimi status --verbose --health-check

# Deploy with AI assistance
synaptic-mesh kimi deploy \
  --environment production \
  --platform vercel \
  --auto-optimize \
  --monitoring
```

## Programming Interface

### Basic Usage

```javascript
import { KimiClient } from './src/core/kimi-client.js';

const client = new KimiClient({
  provider: 'moonshot',
  apiKey: 'your-api-key',
  modelVersion: 'moonshot-v1-128k',
  maxTokens: 128000,
  temperature: 0.7
});

// Connect
await client.connect();

// Chat
const response = await client.chat('Hello, world!');
console.log(response);

// Generate code
const { code, explanation } = await client.generateCode(
  'Create a function to sort an array',
  'javascript',
  { optimize: true, includeTests: true }
);

// Analyze file
const analysis = await client.analyzeFile(
  'example.js',
  fileContent,
  'quality'
);

// Disconnect
client.disconnect();
```

### Event Handling

```javascript
client.on('connected', ({ sessionId }) => {
  console.log('Connected with session:', sessionId);
});

client.on('response', ({ content, usage, model }) => {
  console.log('Response received:', content);
  console.log('Tokens used:', usage.total_tokens);
});

client.on('error', (error) => {
  console.error('API Error:', error.message);
});

client.on('api_call', ({ success, tokens, model }) => {
  if (success) {
    console.log(`API call successful: ${tokens} tokens used`);
  }
});
```

### Advanced Configuration

```javascript
const client = new KimiClient({
  provider: 'moonshot',
  apiKey: process.env.MOONSHOT_API_KEY,
  modelVersion: 'moonshot-v1-128k',
  maxTokens: 128000,
  temperature: 0.7,
  timeout: 60000,
  retryAttempts: 3,
  rateLimitDelay: 1000,
  features: {
    multiModal: true,
    codeGeneration: true,
    documentAnalysis: true,
    imageProcessing: true,
    streaming: true,
    toolCalling: true
  }
});
```

## Testing

### Unit Tests

```bash
# Run unit tests
npm test tests/unit/kimi-client.test.js

# Run with coverage
npm test -- --coverage
```

### Integration Tests

```bash
# Set up API keys for testing
export MOONSHOT_API_KEY="your-test-key"
export OPENROUTER_API_KEY="your-test-key"

# Run integration tests
npm test tests/integration/kimi-api-integration.test.js
```

### Manual Testing

```bash
# Test connection
synaptic-mesh kimi connect

# Test chat
synaptic-mesh kimi chat "Test message"

# Test code generation
synaptic-mesh kimi generate --prompt "Test function" --lang javascript

# Test file analysis
echo "function test() { return true; }" > test.js
synaptic-mesh kimi analyze --file test.js
```

## Performance and Optimization

### Context Management

The client automatically manages conversation context:

- **History Limiting**: Keeps last 20 exchanges to maintain context window
- **Memory Efficient**: Only stores essential conversation data
- **Smart Chunking**: Automatically splits large inputs

### Rate Limiting

Built-in rate limiting handles API quotas:

- **Exponential Backoff**: Intelligent retry strategy
- **Rate Limit Detection**: Automatic throttling based on headers
- **Request Queuing**: Manages concurrent requests

### Error Recovery

Robust error handling ensures reliability:

- **Automatic Retries**: Up to 3 retry attempts with backoff
- **Graceful Degradation**: Fallback responses for errors
- **Connection Recovery**: Automatic reconnection on network issues

## Security Considerations

### API Key Management

- **Environment Variables**: Store keys in `.env` files
- **Key Rotation**: Support for rotating API keys
- **Encryption**: Optional encryption for stored keys
- **Validation**: Automatic key validation and testing

### Data Privacy

- **No Logging**: Sensitive data not logged by default
- **Local Processing**: Conversation history stored locally only
- **Secure Transmission**: HTTPS for all API communications
- **Memory Cleanup**: Automatic cleanup of sensitive data

## Troubleshooting

### Common Issues

**Connection Errors:**
```bash
# Check API key
synaptic-mesh kimi status --verbose

# Test with minimal request
synaptic-mesh kimi chat "test" --format text
```

**Rate Limiting:**
```bash
# Check rate limit status
synaptic-mesh kimi status --health-check

# Reduce request frequency
export KIMI_RATE_LIMIT_DELAY=2000
```

**Timeout Issues:**
```bash
# Increase timeout
export KIMI_TIMEOUT=120000

# Use smaller context
export KIMI_MAX_TOKENS=32000
```

### Debug Mode

Enable debug logging:

```bash
export DEBUG_API_CALLS=true
export LOG_LEVEL=debug
synaptic-mesh kimi chat "Debug test"
```

### API Monitoring

Check API usage and performance:

```bash
# Get detailed status
synaptic-mesh kimi status --verbose --health-check

# Monitor API calls
node -e "
const client = require('./lib/core/kimi-client');
const c = new client.KimiClient({provider:'moonshot', apiKey:process.env.KIMI_API_KEY});
c.on('api_call', console.log);
c.connect().then(() => c.chat('test')).then(console.log);
"
```

## Migration from Mock Implementation

If you were using the previous mock implementation:

1. **Update Configuration**: Add real API keys to `.env`
2. **Test Connection**: Run `synaptic-mesh kimi connect`
3. **Update Code**: Real API responses may differ from mock responses
4. **Error Handling**: Add proper error handling for API failures
5. **Rate Limiting**: Be aware of API quotas and limits

## Support and Resources

- **Documentation**: [Full API Documentation](./api-reference.md)
- **Examples**: Check `./examples/kimi-integration/`
- **Issues**: Report bugs on GitHub Issues
- **Community**: Join discussions in GitHub Discussions

## Roadmap

### Coming Soon
- **Streaming Responses**: Real-time response streaming
- **Function Calling**: Advanced tool integration
- **Image Processing**: Multi-modal image analysis
- **Batch Processing**: Efficient batch operations
- **Custom Models**: Support for fine-tuned models

### Future Enhancements
- **Local Model Support**: Offline model integration
- **Plugin System**: Extensible functionality
- **Advanced Monitoring**: Detailed analytics dashboard
- **Team Collaboration**: Shared configurations and sessions