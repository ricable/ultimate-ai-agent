# Federated MCP API Reference

## Table of Contents
1. [FederationProxy API](#federationproxy-api)
2. [AuthManager API](#authmanager-api)
3. [Type Definitions](#type-definitions)
4. [Error Handling](#error-handling)

## FederationProxy API

### Class: FederationProxy

The main class for managing federated server connections.

#### Constructor

```typescript
constructor(secret: string)
```

Parameters:
- `secret: string` - The secret key used for JWT token generation and validation

Example:
```typescript
const proxy = new FederationProxy("your-secret-key");
```

#### Method: registerServer

Registers a new server in the federation network.

```typescript
async registerServer(config: FederationConfig): Promise<void>
```

Parameters:
- `config: FederationConfig` - Server configuration object

Example:
```typescript
await proxy.registerServer({
  serverId: "server-1",
  endpoints: {
    control: "ws://localhost:3000",
    data: "http://localhost:3001"
  },
  auth: {
    type: "jwt",
    config: { secret: "your-secret-key" }
  }
});
```

#### Method: removeServer

Removes a server from the federation network.

```typescript
async removeServer(serverId: string): Promise<void>
```

Parameters:
- `serverId: string` - Unique identifier of the server to remove

Example:
```typescript
await proxy.removeServer("server-1");
```

#### Method: getConnectedServers

Returns an array of connected server IDs.

```typescript
getConnectedServers(): string[]
```

Returns:
- `string[]` - Array of server IDs

Example:
```typescript
const servers = proxy.getConnectedServers();
console.log("Connected servers:", servers);
```

## AuthManager API

### Class: AuthManager

Handles authentication and token management.

#### Constructor

```typescript
constructor(secret: string)
```

Parameters:
- `secret: string` - Secret key for token generation and validation

Example:
```typescript
const authManager = new AuthManager("your-secret-key");
```

#### Method: createToken

Creates a JWT token for server authentication.

```typescript
async createToken(payload: Record<string, unknown>): Promise<string>
```

Parameters:
- `payload: Record<string, unknown>` - Token payload data

Returns:
- `Promise<string>` - Generated JWT token

Example:
```typescript
const token = await authManager.createToken({
  serverId: "server-1",
  type: "federation"
});
```

#### Method: verifyToken

Verifies a JWT token.

```typescript
async verifyToken(token: string): Promise<Record<string, unknown>>
```

Parameters:
- `token: string` - JWT token to verify

Returns:
- `Promise<Record<string, unknown>>` - Decoded token payload

Example:
```typescript
const payload = await authManager.verifyToken(token);
```

## Type Definitions

### Interface: FederationConfig

Configuration for a federated server.

```typescript
interface FederationConfig {
  serverId: string;
  endpoints: {
    control: string;
    data: string;
  };
  auth: {
    type: 'jwt' | 'oauth2';
    config: Record<string, unknown>;
  };
}
```

Fields:
- `serverId: string` - Unique identifier for the server
- `endpoints` - Server endpoints
  - `control: string` - WebSocket control endpoint
  - `data: string` - HTTP data endpoint
- `auth` - Authentication configuration
  - `type: 'jwt' | 'oauth2'` - Authentication type
  - `config: Record<string, unknown>` - Authentication-specific configuration

### Interface: MCPCapabilities

Defines server capabilities.

```typescript
interface MCPCapabilities {
  resources: boolean;
  prompts: boolean;
  tools: boolean;
  sampling: boolean;
}
```

### Interface: ServerInfo

Server information structure.

```typescript
interface ServerInfo {
  name: string;
  version: string;
  capabilities: MCPCapabilities;
}
```

## Error Handling

### Connection Errors

The system throws specific errors for various connection scenarios:

```typescript
// Connection timeout error
new Error('Connection timeout')

// Authentication error
new Error('Invalid token')

// Server error
new Error('Server connection failed')
```

### Error Types

1. **ConnectionError**
   - Thrown when WebSocket connection fails
   - Includes connection details and server ID

2. **AuthenticationError**
   - Thrown for authentication failures
   - Includes token validation details

3. **TimeoutError**
   - Thrown when connection or operation timeouts occur
   - Includes timeout duration and operation type

### Error Handling Example

```typescript
try {
  await proxy.registerServer(config);
} catch (error) {
  if (error.message.includes('timeout')) {
    // Handle timeout error
  } else if (error.message.includes('token')) {
    // Handle authentication error
  } else {
    // Handle general error
  }
}
```

## WebSocket Events

### Connection Events

```typescript
ws.onopen = () => {
  // Connection established
}

ws.onclose = () => {
  // Connection closed
}

ws.onerror = (error) => {
  // Connection error
}
```

### Message Events

```typescript
ws.onmessage = (event) => {
  // Handle incoming message
}
```

## Best Practices

1. **Error Handling**
   ```typescript
   try {
     await proxy.registerServer(config);
   } catch (error) {
     console.error('Registration failed:', error);
     // Implement retry logic or fallback
   }
   ```

2. **Connection Management**
   ```typescript
   const servers = proxy.getConnectedServers();
   if (servers.includes(serverId)) {
     await proxy.removeServer(serverId);
   }
   ```

3. **Token Management**
   ```typescript
   const token = await authManager.createToken({
     serverId,
     exp: Math.floor(Date.now() / 1000) + (60 * 60) // 1 hour expiration
   });
   ```

## Rate Limiting

The system implements rate limiting for various operations:

```typescript
// Example rate limit configuration
const rateLimits = {
  connections: 100,    // Max concurrent connections
  messages: 1000,      // Messages per minute
  tokens: 100         // Token generations per minute
};
```

## Security Considerations

1. **Token Security**
   - Use strong secrets
   - Implement token expiration
   - Validate token signatures

2. **Connection Security**
   - Use WSS (WebSocket Secure)
   - Implement connection timeouts
   - Validate server certificates

3. **Data Security**
   - Validate message payloads
   - Implement message encryption
   - Use secure protocols
