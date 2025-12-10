# Federated MCP Usage Guide

## Getting Started

### Installation

1. Ensure Deno is installed:
```bash
curl -fsSL https://deno.land/x/install/install.sh | sh
```

2. Import the required modules:
```typescript
import { FederationProxy } from "../packages/proxy/federation.ts";
import { FederationConfig } from "../packages/core/types.ts";
```

### Basic Setup

1. Create a Federation Proxy instance:
```typescript
const SECRET_KEY = "your-secure-secret-key";
const proxy = new FederationProxy(SECRET_KEY);
```

2. Configure a server:
```typescript
const serverConfig: FederationConfig = {
  serverId: "server-1",
  endpoints: {
    control: "ws://localhost:3000",
    data: "http://localhost:3001"
  },
  auth: {
    type: "jwt",
    config: { secret: SECRET_KEY }
  }
};
```

3. Register the server:
```typescript
try {
  await proxy.registerServer(serverConfig);
  console.log("Server registered successfully");
} catch (error) {
  console.error("Registration failed:", error);
}
```

## Common Use Cases

### 1. Managing Multiple Servers

```typescript
// Register multiple servers
const servers = [
  {
    serverId: "server-1",
    endpoints: {
      control: "ws://localhost:3000",
      data: "http://localhost:3001"
    }
  },
  {
    serverId: "server-2",
    endpoints: {
      control: "ws://localhost:3002",
      data: "http://localhost:3003"
    }
  }
];

for (const server of servers) {
  const config: FederationConfig = {
    ...server,
    auth: {
      type: "jwt",
      config: { secret: SECRET_KEY }
    }
  };
  
  try {
    await proxy.registerServer(config);
    console.log(`${server.serverId} registered successfully`);
  } catch (error) {
    console.error(`Failed to register ${server.serverId}:`, error);
  }
}
```

### 2. Server Health Monitoring

```typescript
// Check connected servers periodically
setInterval(() => {
  const connectedServers = proxy.getConnectedServers();
  console.log("Connected servers:", connectedServers);
  
  // Implement health checks
  for (const serverId of connectedServers) {
    checkServerHealth(serverId);
  }
}, 60000); // Check every minute
```

### 3. Error Recovery

```typescript
async function registerWithRetry(config: FederationConfig, maxRetries = 3) {
  let attempts = 0;
  
  while (attempts < maxRetries) {
    try {
      await proxy.registerServer(config);
      console.log(`${config.serverId} registered successfully`);
      return;
    } catch (error) {
      attempts++;
      console.error(`Attempt ${attempts} failed:`, error);
      
      if (attempts === maxRetries) {
        throw new Error(`Failed to register after ${maxRetries} attempts`);
      }
      
      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, 1000 * attempts));
    }
  }
}
```

## Best Practices

### 1. Security

#### Secure Secret Management
```typescript
// Load secret from environment variable
const SECRET_KEY = Deno.env.get("MCP_SECRET_KEY");
if (!SECRET_KEY) {
  throw new Error("MCP_SECRET_KEY environment variable not set");
}
```

#### Token Validation
```typescript
import { AuthManager } from "../packages/core/auth.ts";

const authManager = new AuthManager(SECRET_KEY);

// Validate incoming tokens
async function validateToken(token: string) {
  try {
    const payload = await authManager.verifyToken(token);
    return payload;
  } catch (error) {
    console.error("Token validation failed:", error);
    return null;
  }
}
```

### 2. Connection Management

#### Graceful Shutdown
```typescript
// Handle process termination
async function shutdown() {
  const servers = proxy.getConnectedServers();
  
  for (const serverId of servers) {
    try {
      await proxy.removeServer(serverId);
      console.log(`${serverId} disconnected successfully`);
    } catch (error) {
      console.error(`Failed to disconnect ${serverId}:`, error);
    }
  }
}

// Handle termination signals
Deno.addSignalListener("SIGINT", shutdown);
Deno.addSignalListener("SIGTERM", shutdown);
```

#### Connection Monitoring
```typescript
function monitorConnections() {
  const servers = proxy.getConnectedServers();
  
  console.log(`Active connections: ${servers.length}`);
  console.log("Connected servers:", servers);
  
  // Implement additional monitoring logic
}

// Monitor connections periodically
setInterval(monitorConnections, 30000);
```

### 3. Error Handling

#### Comprehensive Error Handling
```typescript
async function handleServerOperation(operation: () => Promise<void>) {
  try {
    await operation();
  } catch (error) {
    if (error instanceof WebSocket.Error) {
      console.error("WebSocket error:", error);
      // Handle WebSocket specific errors
    } else if (error.message.includes("timeout")) {
      console.error("Operation timed out:", error);
      // Handle timeout errors
    } else {
      console.error("Unknown error:", error);
      // Handle other errors
    }
  }
}
```

## Configuration Examples

### 1. Development Configuration
```typescript
const devConfig: FederationConfig = {
  serverId: "dev-server",
  endpoints: {
    control: "ws://localhost:3000",
    data: "http://localhost:3001"
  },
  auth: {
    type: "jwt",
    config: {
      secret: "dev-secret",
      expiresIn: "1h"
    }
  }
};
```

### 2. Production Configuration
```typescript
const prodConfig: FederationConfig = {
  serverId: "prod-server",
  endpoints: {
    control: "wss://production.example.com/ws",
    data: "https://production.example.com/api"
  },
  auth: {
    type: "jwt",
    config: {
      secret: Deno.env.get("PROD_SECRET"),
      expiresIn: "24h"
    }
  }
};
```

## Testing

### 1. Connection Testing
```typescript
async function testConnection(config: FederationConfig) {
  try {
    await proxy.registerServer(config);
    const servers = proxy.getConnectedServers();
    console.assert(servers.includes(config.serverId), "Server not connected");
    await proxy.removeServer(config.serverId);
  } catch (error) {
    console.error("Connection test failed:", error);
  }
}
```

### 2. Load Testing
```typescript
async function loadTest(numConnections: number) {
  const results = [];
  
  for (let i = 0; i < numConnections; i++) {
    const config: FederationConfig = {
      serverId: `test-server-${i}`,
      endpoints: {
        control: `ws://localhost:${3000 + i}`,
        data: `http://localhost:${4000 + i}`
      },
      auth: {
        type: "jwt",
        config: { secret: SECRET_KEY }
      }
    };
    
    results.push(proxy.registerServer(config));
  }
  
  await Promise.allSettled(results);
}
```

## Troubleshooting

### Common Issues and Solutions

1. **Connection Failures**
   ```typescript
   // Check if server is reachable
   async function checkServerAvailability(url: string) {
     try {
       const response = await fetch(url);
       return response.ok;
     } catch {
       return false;
     }
   }
   ```

2. **Authentication Issues**
   ```typescript
   // Verify token is valid
   async function verifyAuthentication(config: FederationConfig) {
     const authManager = new AuthManager(config.auth.config.secret as string);
     try {
       const token = await authManager.createToken({ test: true });
       await authManager.verifyToken(token);
       return true;
     } catch {
       return false;
     }
   }
   ```

3. **Performance Issues**
   ```typescript
   // Monitor connection latency
   async function checkLatency(serverId: string) {
     const start = Date.now();
     await proxy.removeServer(serverId);
     await proxy.registerServer(config);
     return Date.now() - start;
   }
   ```

## Maintenance

### Regular Maintenance Tasks

1. **Connection Cleanup**
   ```typescript
   async function cleanupConnections() {
     const servers = proxy.getConnectedServers();
     for (const serverId of servers) {
       // Implement cleanup logic
       await proxy.removeServer(serverId);
     }
   }
   ```

2. **Health Checks**
   ```typescript
   async function performHealthCheck() {
     const servers = proxy.getConnectedServers();
     const results = new Map();
     
     for (const serverId of servers) {
       // Implement health check logic
       results.set(serverId, "healthy");
     }
     
     return results;
   }
   ```

Remember to regularly:
- Monitor system performance
- Update security configurations
- Perform connection maintenance
- Review error logs
- Test recovery procedures
