# Federated MCP Documentation

Welcome to the Federated MCP documentation. This documentation provides comprehensive information about the Federated MCP system, its architecture, implementation details, API references, usage guidelines, and development instructions.

## Table of Contents

1. [Architecture](./architecture.md)
   - System Overview
   - Components
   - Communication Flow
   - Security Model

2. [Implementation](./implementation.md)
   - Core Components
   - Federation Protocol
   - Authentication
   - WebSocket Communication
   - Error Handling

3. [API Reference](./api.md)
   - Federation Proxy API
   - Authentication API
   - Server Configuration
   - Type Definitions

4. [Usage Guide](./usage.md)
   - Getting Started
   - Configuration
   - Server Registration
   - Connection Management
   - Best Practices

5. [Development Guide](./development.md)
   - Setup Development Environment
   - Running Tests
   - Contributing Guidelines
   - Code Style
   - Release Process

6. [Edge Functions](./edge_functions.md)
   - Available Functions
   - Provider Configuration
   - Deployment Guide
   - Monitoring & Logs

## Quick Start

```typescript
import { FederationProxy } from "../packages/proxy/federation.ts";

// Initialize the federation proxy with a secret key
const proxy = new FederationProxy("your-secret-key");

// Configure a server
const config = {
  serverId: "server-1",
  endpoints: {
    control: "ws://localhost:3000",
    data: "http://localhost:3001"
  },
  auth: {
    type: "jwt",
    config: { secret: "your-secret-key" }
  }
};

// Register the server
await proxy.registerServer(config);

// Get connected servers
const servers = proxy.getConnectedServers();
console.log("Connected servers:", servers);
```

## System Requirements

- Deno 1.x or higher
- Secure WebSocket support
- JWT authentication capabilities

## Support

For issues, feature requests, or contributions, please:
1. Check the existing documentation
2. Review the [Development Guide](./development.md)
3. Submit issues through the project's issue tracker

## License

This project is licensed under the terms specified in the project's LICENSE file.
