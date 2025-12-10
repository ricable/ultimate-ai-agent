# FastMCP Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Build the Project
```bash
cd docker/claude-agent-sdk
npm run build
```

### 2. Choose Your Transport

#### Option A: stdio (Local MCP)
```bash
# Start server
node dist/mcp/fastmcp/servers/stdio-full.js

# Test all 6 tools
./scripts/test-claude-flow-sdk.sh
```

#### Option B: HTTP + SSE (Web/Remote)
```bash
# Start server
node dist/mcp/fastmcp/servers/http-streaming.js

# Test in another terminal
curl http://localhost:3000/health
curl -N http://localhost:3000/events
```

### 3. Use the Tools

## 📦 Available Tools

### Memory Management
```bash
# Store
{"name":"memory_store","arguments":{"key":"mykey","value":"myvalue","namespace":"default"}}

# Retrieve
{"name":"memory_retrieve","arguments":{"key":"mykey","namespace":"default"}}

# Search
{"name":"memory_search","arguments":{"pattern":"my*","limit":10}}
```

### Swarm Coordination
```bash
# Initialize
{"name":"swarm_init","arguments":{"topology":"mesh","maxAgents":5,"strategy":"balanced"}}

# Spawn agent
{"name":"agent_spawn","arguments":{"type":"researcher","capabilities":["analysis","research"]}}

# Orchestrate
{"name":"task_orchestrate","arguments":{"task":"Analyze data","strategy":"parallel","priority":"high"}}
```

## 🔌 Integration

### Claude Desktop
Add to `~/.config/claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "claude-flow-sdk": {
      "command": "node",
      "args": ["/absolute/path/to/dist/mcp/fastmcp/servers/stdio-full.js"]
    }
  }
}
```

### HTTP Client (JavaScript)
```javascript
async function callTool(name, args) {
  const res = await fetch('http://localhost:3000/mcp', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      jsonrpc: '2.0',
      id: 1,
      method: 'tools/call',
      params: { name, arguments: args }
    })
  });
  const data = await res.json();
  return JSON.parse(data.result.content[0].text);
}

// Example usage
const result = await callTool('memory_store', {
  key: 'test',
  value: 'Hello FastMCP!',
  namespace: 'demo'
});
console.log(result); // { success: true, key: 'test', ... }
```

### SSE Streaming (JavaScript)
```javascript
const events = new EventSource('http://localhost:3000/events');

events.addEventListener('connected', (e) => {
  console.log('Connected!', JSON.parse(e.data));
});

events.addEventListener('progress', (e) => {
  const { progress, message } = JSON.parse(e.data);
  console.log(`${Math.round(progress * 100)}%: ${message}`);
});

events.addEventListener('ping', (e) => {
  console.log('Server ping:', new Date().toISOString());
});
```

## 📊 Tool Reference

### memory_store
Store persistent values with optional TTL.

**Parameters**:
- `key` (string, required) - Storage key
- `value` (string, required) - Value to store
- `namespace` (string, optional) - Namespace, default: "default"
- `ttl` (number, optional) - Time-to-live in seconds

**Example**:
```json
{
  "key": "user-session",
  "value": "session-123",
  "namespace": "auth",
  "ttl": 3600
}
```

### memory_retrieve
Get stored values from memory.

**Parameters**:
- `key` (string, required) - Storage key
- `namespace` (string, optional) - Namespace, default: "default"

**Example**:
```json
{
  "key": "user-session",
  "namespace": "auth"
}
```

### memory_search
Search for keys matching patterns.

**Parameters**:
- `pattern` (string, required) - Search pattern (supports wildcards)
- `namespace` (string, optional) - Namespace to search
- `limit` (number, optional) - Max results, default: 10

**Example**:
```json
{
  "pattern": "user-*",
  "namespace": "auth",
  "limit": 5
}
```

### swarm_init
Initialize multi-agent swarm.

**Parameters**:
- `topology` (enum, required) - "mesh" | "hierarchical" | "ring" | "star"
- `maxAgents` (number, optional) - Max agents, default: 8
- `strategy` (enum, optional) - "balanced" | "specialized" | "adaptive", default: "balanced"

**Example**:
```json
{
  "topology": "mesh",
  "maxAgents": 10,
  "strategy": "adaptive"
}
```

### agent_spawn
Create new agent in swarm.

**Parameters**:
- `type` (enum, required) - "researcher" | "coder" | "analyst" | "optimizer" | "coordinator"
- `capabilities` (array, optional) - Agent capabilities
- `name` (string, optional) - Custom agent name

**Example**:
```json
{
  "type": "researcher",
  "capabilities": ["data-analysis", "pattern-recognition"],
  "name": "research-agent-001"
}
```

### task_orchestrate
Coordinate task across swarm.

**Parameters**:
- `task` (string, required) - Task description
- `strategy` (enum, optional) - "parallel" | "sequential" | "adaptive", default: "adaptive"
- `priority` (enum, optional) - "low" | "medium" | "high" | "critical", default: "medium"
- `maxAgents` (number, optional) - Max agents for this task

**Example**:
```json
{
  "task": "Analyze user behavior patterns",
  "strategy": "parallel",
  "priority": "high",
  "maxAgents": 5
}
```

## 🧪 Testing

### Automated Test Suite
```bash
./scripts/test-claude-flow-sdk.sh
```

Expected output:
```
🧪 Testing FastMCP claude-flow-sdk Server (6 tools)...
✅ Test 1: memory_store - PASS
✅ Test 2: memory_retrieve - PASS
✅ Test 3: memory_search - PASS
✅ Test 4: swarm_init - PASS
✅ Test 5: agent_spawn - PASS
✅ Test 6: task_orchestrate - PASS
```

### Manual Testing (HTTP)
```bash
# Start server
node dist/mcp/fastmcp/servers/http-streaming.js

# Test in another terminal
curl -X POST http://localhost:3000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "memory_store",
      "arguments": {
        "key": "test",
        "value": "Hello World"
      }
    }
  }'
```

## 🔍 Debugging

### Enable Verbose Logging
```bash
DEBUG=fastmcp:* node dist/mcp/fastmcp/servers/http-streaming.js
```

### Check Server Status
```bash
# Health check
curl http://localhost:3000/health

# List tools
curl -X POST http://localhost:3000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

### Common Issues

**Port 3000 in use**:
```bash
lsof -i :3000
kill -9 <PID>
```

**Tool execution fails**:
```bash
# Test claude-flow directly
npx claude-flow@alpha memory store "test" "value"

# Check installation
npx claude-flow@alpha --version
```

**stdio server silent**:
```bash
# Check build
npm run build

# Verify files
ls -la dist/mcp/fastmcp/servers/
```

## 📈 Performance Tips

1. **Use appropriate transport**:
   - stdio for local MCP clients (lower latency)
   - HTTP for web apps (better scalability)

2. **Batch operations**:
   - Group related tool calls
   - Use namespaces to organize data

3. **Set reasonable limits**:
   - Configure TTL for temporary data
   - Limit search results
   - Optimize swarm size

4. **Monitor resources**:
   - Check memory usage
   - Monitor response times
   - Track error rates

## 🚀 Next Steps

1. **Explore Advanced Features**:
   - Progress reporting with SSE
   - Authentication middleware
   - Rate limiting

2. **Build Your Application**:
   - Integrate into Claude Desktop
   - Create web dashboard
   - Build custom tools

3. **Contribute**:
   - Add new tools
   - Improve documentation
   - Share examples

## 📚 Resources

- Full Documentation: `docs/fastmcp-implementation.md`
- FastMCP Library: https://github.com/jlowin/fastmcp
- MCP Specification: https://spec.modelcontextprotocol.io/
- Claude Flow: https://github.com/ruvnet/claude-flow

---

**Status**: ✅ Production Ready
**Version**: 1.0.0
**Tools**: 6 (Memory × 3, Swarm × 3)
**Transports**: stdio, HTTP+SSE
