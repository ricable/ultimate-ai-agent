# FastMCP Technical Implementation Patterns

## Core Architectural Patterns Extracted from Research

### 1. Server Composition Patterns

#### Static Import Pattern (Python FastMCP)
```python
# Pattern: Static server composition with prefixing
from fastmcp import FastMCP
import asyncio

# Define subservers
weather_mcp = FastMCP(name="WeatherService")

@weather_mcp.tool()
def get_forecast(city: str) -> dict:
    return {"city": city, "forecast": "Sunny"}

@weather_mcp.resource("data://cities/supported")
def list_supported_cities() -> list[str]:
    return ["London", "Paris", "Tokyo"]

# Main server with static composition
main_mcp = FastMCP(name="MainApp")

async def setup():
    # Components copied with prefix - no live updates
    await main_mcp.import_server("weather", weather_mcp)

# Result: main_mcp contains "weather_get_forecast" tool
```

#### Dynamic Mount Pattern (Python FastMCP)
```python
# Pattern: Live-linked server mounting
main_mcp = FastMCP(name="MainAppLive")
dynamic_mcp = FastMCP(name="DynamicService")

@dynamic_mcp.tool()
def initial_tool():
    return "Initial Tool Exists"

# Live mounting - changes reflected immediately
main_mcp.mount("dynamic", dynamic_mcp)

# Add tool after mounting - available immediately
@dynamic_mcp.tool()
def added_later():
    return "Tool Added Dynamically!"

# Tools accessible as "dynamic_initial_tool", "dynamic_added_later"
```

#### Proxy Mount Pattern (Python FastMCP)
```python
# Pattern: Remote server proxying
from fastmcp import Client

# Create proxy for remote server
remote_proxy = FastMCP.as_proxy(Client("http://example.com/mcp"))

# Mount with full lifecycle preservation
main_server.mount("remote", remote_proxy, as_proxy=True)

# Multi-server composite proxy
config = {
    "mcpServers": {
        "weather": {"url": "https://weather-api.example.com/mcp"},
        "calendar": {"url": "https://calendar-api.example.com/mcp"}
    }
}
composite_proxy = FastMCP.as_proxy(config, name="Composite Proxy")
```

### 2. Session Management Patterns

#### TypeScript Session Handling
```typescript
// Pattern: Event-driven session management
const server = new FastMCP({
  name: "My Server",
  version: "1.0.0"
});

server.on("connect", (event) => {
  const session = event.session;
  console.log("Client connected:", session.id);
  
  // Access session properties
  console.log("Client capabilities:", session.clientCapabilities);
  console.log("Client roots:", session.roots);
  console.log("Logging level:", session.loggingLevel);
  
  // Listen for session events
  session.on("rootsChanged", (event) => {
    console.log("Roots updated:", event.roots);
  });
  
  session.on("error", (event) => {
    console.error("Session error:", event.error);
  });
});

server.on("disconnect", (event) => {
  console.log("Client disconnected:", event.session.id);
});
```

#### Python Context Injection Pattern
```python
# Pattern: Context-aware tool execution
from fastmcp import FastMCP, Context

mcp = FastMCP(name="ContextDemo")

@mcp.tool()
async def process_file(file_uri: str, ctx: Context) -> str:
    """Tool with injected context for server capabilities"""
    # Context provides access to server state and capabilities
    await ctx.log_info(f"Processing file: {file_uri}")
    return "Processed file"

@mcp.resource("resource://user-data")
async def get_user_data(ctx: Context) -> dict:
    """Resource with context for request-specific data"""
    user_id = ctx.get_user_id()  # Example context method
    return {"user_id": user_id}
```

### 3. Health Monitoring and Discovery Patterns

#### TypeScript Health Check Configuration
```typescript
// Pattern: Comprehensive health monitoring
const server = new FastMCP({
  name: "Distributed Node",
  version: "1.0.0",
  health: {
    enabled: true,
    message: "healthy",
    path: "/healthz",
    status: 200
  },
  ping: {
    enabled: true,
    intervalMs: 10000,
    logLevel: "debug"
  },
  roots: {
    enabled: true  // Enable client capability discovery
  }
});

await server.start({
  transportType: "httpStream",
  httpStream: { port: 8080 }
});
```

#### Python Multi-Transport Pattern
```python
# Pattern: Flexible transport configuration
from fastmcp import FastMCP, Client
from fastmcp.client.transports import (
    SSETransport, 
    StreamableHTTPTransport,
    NpxStdioTransport
)

# HTTP Streaming for high throughput
http_client = Client(StreamableHTTPTransport("http://localhost:8080/stream"))

# SSE for real-time events
sse_client = Client(SSETransport("http://localhost:8080/sse"))

# NPX for Node.js packages
npx_client = Client(NpxStdioTransport(package="mcp-server-package"))

# Auto-inferred transport
auto_client = Client("http://localhost:8080/sse")  # Automatically uses SSE
```

### 4. Load Balancing and Task Distribution

#### Progress Reporting Pattern (TypeScript)
```typescript
// Pattern: Granular progress tracking
server.addTool({
  name: "processData",
  description: "Process data with streaming updates",
  parameters: z.object({
    datasetSize: z.number(),
  }),
  annotations: {
    streamingHint: true,
  },
  execute: async (args, { streamContent, reportProgress }) => {
    const total = args.datasetSize;

    for (let i = 0; i < total; i++) {
      // Report numeric progress for load balancing decisions
      await reportProgress({ progress: i, total });

      // Stream intermediate results
      if (i % 10 === 0) {
        await streamContent({
          type: "text",
          text: `Processed ${i} of ${total} items...\n`,
        });
      }

      await new Promise((resolve) => setTimeout(resolve, 50));
    }

    return "Processing complete!";
  },
});
```

#### Python Async Tool Pattern
```python
# Pattern: Non-blocking tool execution
import aiohttp
from fastmcp import FastMCP

mcp = FastMCP()

@mcp.tool()
async def fetch_weather(city: str) -> dict:
    """Asynchronous tool for I/O operations"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/weather/{city}") as response:
            response.raise_for_status()
            return await response.json()

@mcp.tool()
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Synchronous tool for CPU-bound tasks"""
    # CPU-intensive calculation
    return 42.5
```

### 5. Resource Template and State Management

#### Dynamic Resource Templates (TypeScript)
```typescript
// Pattern: Parameterized resource templates with auto-completion
server.addResourceTemplate({
  uriTemplate: "file:///logs/{name}.log",
  name: "Application Logs",
  mimeType: "text/plain",
  arguments: [
    {
      name: "name",
      description: "Name of the log",
      required: true,
      complete: async (value) => {
        // Dynamic auto-completion based on available logs
        const availableLogs = await getAvailableLogNames();
        return {
          values: availableLogs.filter(log => log.startsWith(value))
        };
      }
    }
  ],
  async load({ name }) {
    return {
      text: await readLogFile(name),
    };
  }
});
```

#### Python Multi-URI Resource Pattern
```python
# Pattern: Multiple URI templates for same resource
from fastmcp import FastMCP

mcp = FastMCP(name="DataServer")

@mcp.resource("users://email/{email}")
@mcp.resource("users://name/{name}")
def lookup_user(name: str | None = None, email: str | None = None) -> dict:
    """Resource accessible via multiple identifiers"""
    if email:
        return find_user_by_email(email)
    elif name:
        return find_user_by_name(name)
    else:
        return {"error": "No lookup parameters provided"}
```

### 6. Error Handling and Fault Tolerance

#### Granular Error Handling (Python)
```python
# Pattern: Differentiated error handling
from fastmcp import FastMCP
from fastmcp.exceptions import ResourceError

mcp = FastMCP(name="DataServer")

@mcp.resource("resource://safe-error")
def fail_with_details() -> str:
    # ResourceError details always sent to clients
    raise ResourceError("Unable to retrieve data: file not found")

@mcp.resource("resource://masked-error")
def fail_with_masked_details() -> str:
    # ValueError details masked based on server settings
    raise ValueError("Sensitive internal file path: /etc/secrets.conf")

@mcp.resource("data://{id}")
def get_data_by_id(id: str) -> dict:
    if id == "secure":
        raise ValueError("Cannot access secure data")
    elif id == "missing":
        raise ResourceError("Data ID 'missing' not found in database")
    return {"id": id, "value": "data"}
```

#### TypeScript User Error Pattern
```typescript
// Pattern: User-facing error handling
import { UserError } from "fastmcp";

server.addTool({
  name: "download",
  description: "Download a file",
  parameters: z.object({
    url: z.string(),
  }),
  execute: async (args) => {
    if (args.url.startsWith("https://blocked.com")) {
      throw new UserError("This URL is not allowed");
    }
    
    try {
      return await downloadFile(args.url);
    } catch (error) {
      // System errors are automatically masked
      throw error;
    }
  }
});
```

### 7. Authentication and Security Patterns

#### TypeScript Authentication Pattern
```typescript
// Pattern: Custom authentication with session data
const server = new FastMCP({
  name: "Secure Server",
  version: "1.0.0",
  authenticate: (request) => {
    const apiKey = request.headers["x-api-key"];
    const bearerToken = request.headers["authorization"];

    if (apiKey !== "valid-key" && !isValidBearerToken(bearerToken)) {
      throw new Response(null, {
        status: 401,
        statusText: "Unauthorized"
      });
    }

    // Return session data accessible in tools
    return {
      id: extractUserId(apiKey || bearerToken),
      permissions: getPermissions(apiKey || bearerToken)
    };
  }
});

server.addTool({
  name: "secureOperation",
  execute: async (args, { session }) => {
    if (!session.permissions.includes("admin")) {
      throw new UserError("Insufficient permissions");
    }
    return `Hello, user ${session.id}!`;
  }
});
```

#### Python Roots Management Pattern
```python
# Pattern: Dynamic client capability discovery
from fastmcp import Client
from fastmcp.client.roots import RequestContext

async def dynamic_roots_callback(context: RequestContext) -> list[str]:
    """Dynamic roots based on request context"""
    user_id = context.get_user_id()
    user_permissions = await get_user_permissions(user_id)
    
    roots = ["/shared"]
    if "admin" in user_permissions:
        roots.append("/admin")
    if "user_data" in user_permissions:
        roots.append(f"/users/{user_id}")
    
    return roots

client = Client(
    server_config,
    roots=dynamic_roots_callback
)
```

### 8. Configuration and Deployment Patterns

#### Multi-Server Configuration (Python)
```python
# Pattern: Unified multi-server client
from fastmcp import Client

config = {
    "mcpServers": {
        "weather": {
            "url": "https://weather-api.example.com/mcp",
            "transport": "streamable-http"
        },
        "database": {
            "command": "python",
            "args": ["./db_server.py"],
            "env": {"DB_URL": "postgresql://..."}
        },
        "ai_service": {
            "command": "npx",
            "args": ["tsx", "./ai_server.ts"]
        }
    }
}

# Single client manages all servers
client = Client(config)

async def main():
    async with client:
        # Tools prefixed by server name
        weather = await client.call_tool("weather_get_forecast", {"city": "London"})
        data = await client.call_tool("database_query", {"sql": "SELECT * FROM users"})
        ai_result = await client.call_tool("ai_service_analyze", {"text": "Hello"})
```

#### ASGI Integration Pattern (Python)
```python
# Pattern: FastMCP in existing web applications
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

# Create MCP server
mcp = FastMCP("MyServer")

@mcp.tool()
def api_tool():
    return "API response"

# Custom middleware
custom_middleware = [
    Middleware(CORSMiddleware, allow_origins=["*"]),
]

# Create ASGI app
mcp_app = mcp.http_app(middleware=custom_middleware)

# Integrate with existing Starlette app
app = Starlette(
    routes=[
        Mount("/api", app=existing_api_app),
        Mount("/mcp", app=mcp_app),
    ],
    lifespan=mcp_app.lifespan,  # Critical for proper initialization
)
```

## Implementation Guidelines for SAFLA

### 1. Adopt Composition Patterns
- Use **static import** for stable service integrations
- Use **dynamic mounting** for runtime service discovery
- Use **proxy mounting** for remote service orchestration

### 2. Implement Session Management
- Event-driven architecture for connection lifecycle
- Context injection for tool execution
- Dynamic capability discovery

### 3. Build Health Monitoring
- HTTP health endpoints for load balancers
- Ping mechanisms for connection health
- Roots management for client capabilities

### 4. Design for Fault Tolerance
- Granular error handling with ResourceError vs ValueError
- User-facing errors with UserError
- Authentication with session data

### 5. Support Multiple Transports
- HTTP Streaming for high throughput
- SSE for real-time events
- Stdio for local processes
- Auto-detection based on configuration

These patterns provide a solid foundation for implementing distributed MCP orchestration in SAFLA, leveraging proven FastMCP architectural decisions while extending them for distributed scenarios.