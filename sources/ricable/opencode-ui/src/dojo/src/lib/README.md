# OpenCode API Client

A comprehensive TypeScript client for communicating with the OpenCode Go backend, providing real-time session management, multi-provider support, and advanced tool integration.

## Features

- 🔄 **Real-time Communication**: WebSocket-based streaming for live updates
- 🌐 **Multi-Provider Support**: 75+ AI providers including Anthropic, OpenAI, Google, Groq
- 📝 **Session Management**: Full session lifecycle with SQLite persistence
- 🛠️ **Tool System**: Execute tools with MCP server integration
- ⚙️ **Configuration**: Schema-validated configuration management
- 💻 **LSP Integration**: Language server protocol support
- 📊 **Analytics**: Usage tracking and cost analysis
- 🔧 **Custom Commands**: User-defined automation commands
- 🔍 **Health Monitoring**: Connection status and provider health checks

## Quick Start

```typescript
import { openCodeClient } from '@/lib/opencode-client';

// Check server health
const health = await openCodeClient.healthCheck();
console.log('Server status:', health.status);

// Get available providers
const providers = await openCodeClient.getProviders();
console.log('Available providers:', providers.length);

// Create a new session
const session = await openCodeClient.createSession({
  provider: 'anthropic',
  model: 'claude-3-5-sonnet-20241022',
  max_tokens: 8000,
  temperature: 0.7
});

// Send a message
await openCodeClient.sendMessage(session.id, 'Hello, Claude!');

// Subscribe to real-time updates
const unsubscribe = openCodeClient.subscribeToSession(session.id, (update) => {
  console.log('Session update:', update);
});
```

## Core Concepts

### Sessions

Sessions represent conversations with AI providers. Each session maintains:
- Message history
- Configuration settings
- Provider and model selection
- Cost tracking
- Tool execution history

```typescript
// Create session with custom configuration
const session = await openCodeClient.createSession({
  provider: 'openai',
  model: 'gpt-4o',
  max_tokens: 4000,
  temperature: 0.5,
  system_prompt: 'You are a helpful coding assistant.',
  tools_enabled: true
});

// Share session with others
const shareLink = await openCodeClient.shareSession(session.id, {
  expires_in_hours: 24,
  password: 'optional-password'
});
```

### Providers

OpenCode supports 75+ AI providers through a unified interface:

```typescript
// Get all providers
const providers = await openCodeClient.getProviders();

// Authenticate with a provider
const authResult = await openCodeClient.authenticateProvider('anthropic', {
  apiKey: 'your-api-key'
});

// Monitor provider health
const health = await openCodeClient.getProviderHealth();
const metrics = await openCodeClient.getProviderMetrics();
```

### Tool System

Execute tools and manage MCP servers:

```typescript
// Get available tools
const tools = await openCodeClient.getTools();

// Execute a tool
const result = await openCodeClient.executeTool('file_read', {
  path: '/src/index.ts'
}, session.id);

// Manage MCP servers
const mcpServers = await openCodeClient.getMCPServers();
await openCodeClient.addMCPServer({
  name: 'GitHub MCP',
  type: 'stdio',
  command: 'npx',
  args: ['@modelcontextprotocol/server-github']
});
```

### Real-time Updates

Subscribe to various event streams:

```typescript
// Session updates (messages, status changes)
const sessionUnsub = openCodeClient.subscribeToSession(sessionId, (update) => {
  switch (update.type) {
    case 'message':
      console.log('New message:', update.data);
      break;
    case 'status':
      console.log('Status change:', update.data);
      break;
    case 'tool_execution':
      console.log('Tool execution:', update.data);
      break;
  }
});

// Provider updates (health, authentication)
const providerUnsub = openCodeClient.subscribeToProviderUpdates((update) => {
  console.log('Provider update:', update);
});

// Tool execution updates
const toolUnsub = openCodeClient.subscribeToToolExecutions((update) => {
  console.log('Tool update:', update);
});
```

## Configuration Management

OpenCode uses a comprehensive configuration system:

```typescript
// Get current configuration
const config = await openCodeClient.getConfig();

// Update configuration
await openCodeClient.updateConfig({
  theme: 'dark',
  autoshare: true,
  providers: {
    anthropic: { 
      apiKey: 'new-key',
      disabled: false 
    }
  }
});

// Validate configuration
const validation = await openCodeClient.validateConfig(config);
if (!validation.valid) {
  console.error('Configuration errors:', validation.errors);
}
```

## LSP Integration

Language Server Protocol support for enhanced development:

```typescript
// Get LSP servers
const lspServers = await openCodeClient.getLSPServers();

// Get diagnostics for current file
const diagnostics = await openCodeClient.getDiagnostics('/src/index.ts');

// Control LSP servers
await openCodeClient.enableLSPServer('typescript');
await openCodeClient.restartLSPServer('python');
```

## Custom Commands

Create and manage custom automation commands:

```typescript
// Get custom commands
const commands = await openCodeClient.getCustomCommands();

// Execute a command
const result = await openCodeClient.executeCommand('format_code', {
  file: '/src/index.ts'
});

// Create new command
await openCodeClient.createCustomCommand({
  name: 'Run Tests',
  description: 'Execute test suite',
  command: 'npm',
  args: { script: 'test' },
  enabled: true,
  shortcuts: ['Ctrl+T']
});
```

## Usage Analytics

Track usage and costs across providers:

```typescript
// Get usage statistics
const stats = await openCodeClient.getUsageStats();
console.log('Total cost this month:', stats.this_month.cost);

// Get detailed cost breakdown
const costBreakdown = await openCodeClient.getCostBreakdown('month');

// Export usage data
const exportData = await openCodeClient.exportUsageData('csv');
```

## Streaming Messages

Real-time streaming for better user experience:

```typescript
await openCodeClient.sendStreamMessage(
  sessionId,
  'Write a React component',
  (chunk) => {
    switch (chunk.type) {
      case 'delta':
        // Append incremental content
        appendToResponse(chunk.content);
        break;
      case 'complete':
        // Message completed
        finalizeResponse(chunk.metadata);
        break;
      case 'error':
        // Handle streaming error
        handleError(chunk.error);
        break;
    }
  },
  {
    model_config: { temperature: 0.7 },
    tools_enabled: true
  }
);
```

## Error Handling

Robust error handling with typed exceptions:

```typescript
import { OpenCodeAPIError } from '@/lib/opencode-client';

try {
  await openCodeClient.sendMessage(sessionId, 'Hello');
} catch (error) {
  if (error instanceof OpenCodeAPIError) {
    console.error('API Error:', error.message);
    console.error('Status Code:', error.status);
    console.error('Error Code:', error.code);
    
    // Handle specific error types
    if (error.status === 429) {
      // Rate limit exceeded
      await delay(error.retry_after || 1000);
    }
  }
}
```

## Event System

Custom event system for component communication:

```typescript
// Listen for events
const unsubscribe = openCodeClient.on('session_created', (session) => {
  console.log('New session created:', session.id);
});

// One-time event listener
openCodeClient.once('connection_restored', () => {
  console.log('Connection restored!');
});

// Emit custom events
openCodeClient.emit('custom_event', { data: 'value' });

// Remove listeners
openCodeClient.off('session_created');
unsubscribe();
```

## Type Safety

Full TypeScript support with comprehensive type definitions:

```typescript
import type {
  Session,
  Provider,
  ToolResult,
  SessionConfig,
  OpenCodeConfig,
  LSPDiagnostic
} from '@/lib/opencode-client';

// Type guards for runtime validation
import { isValidSession, isValidProvider } from '@/lib/opencode-client';

const session: unknown = await fetchSession();
if (isValidSession(session)) {
  // session is now typed as Session
  console.log('Session ID:', session.id);
}
```

## Connection Management

Automatic reconnection and connection monitoring:

```typescript
// Get connection status
const status = openCodeClient.getConnectionStatus();
console.log('Connection status:', status.status);
console.log('Active WebSockets:', status.activeWebSockets);

// Test connection manually
const connectionTest = await openCodeClient.testConnection();
if (connectionTest.success) {
  console.log('Latency:', connectionTest.latency, 'ms');
}

// Handle connection events
openCodeClient.on('connection_status_change', ({ status }) => {
  console.log('Connection status changed to:', status);
});

openCodeClient.on('websocket_reconnect_failed', ({ sessionId }) => {
  console.error('Failed to reconnect session:', sessionId);
});
```

## Best Practices

### Resource Management

```typescript
// Always clean up subscriptions
const unsubscribe = openCodeClient.subscribeToSession(sessionId, handler);

// In cleanup (useEffect, component unmount, etc.)
return () => {
  unsubscribe();
};

// Graceful shutdown
await openCodeClient.shutdown();
```

### Error Recovery

```typescript
// Implement retry logic for transient errors
async function retryOperation<T>(
  operation: () => Promise<T>,
  maxRetries: number = 3
): Promise<T> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await operation();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      await delay(1000 * Math.pow(2, i)); // Exponential backoff
    }
  }
  throw new Error('Max retries exceeded');
}
```

### Performance Optimization

```typescript
// Use pagination for large datasets
const messages = await openCodeClient.getSessionMessages(sessionId, {
  limit: 50,
  offset: 0
});

// Batch operations when possible
const [providers, tools, config] = await Promise.all([
  openCodeClient.getProviders(),
  openCodeClient.getTools(),
  openCodeClient.getConfig()
]);
```

## Testing

The client includes comprehensive test coverage. Run tests with:

```bash
npm test src/lib/__tests__/opencode-client.test.ts
```

See the test file for usage examples and edge cases.

## API Reference

For complete API documentation, see the TypeScript definitions in the source files. The client provides full IntelliSense support in modern editors.

## Contributing

When extending the client:

1. Add new methods following the existing patterns
2. Include comprehensive TypeScript types
3. Add mock implementations for development
4. Write tests for new functionality
5. Update this documentation

## License

This OpenCode API client is part of the OpenCode project and follows the same licensing terms.