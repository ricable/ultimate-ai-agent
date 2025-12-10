# MCP Tools Quick Start Guide

**Get started with MCP tools in 5 minutes** ⚡

---

## What You'll Learn

- Install and configure MCP servers
- Make your first successful tool call
- Understand the correct API format
- Verify everything is working

**Time Required:** 5 minutes

---

## Prerequisites Checklist

Before starting, ensure you have:

- ✅ **Claude Desktop** or **Claude Code** installed
- ✅ **Node.js 18+** (`node --version`)
- ✅ **npm 9+** (`npm --version`)
- ✅ **Terminal/Command Line** access

---

## 1. Installation (1 minute)

### Add Your First MCP Server

**Claude Flow** is the best starting point (no authentication required):

```bash
# Add Claude Flow MCP server
claude mcp add claude-flow npx claude-flow@alpha mcp start
```

**Verify installation:**
```bash
# List installed servers
claude mcp list
```

You should see `claude-flow` in the list.

---

## 2. Your First Tool Call (2 minutes)

### ✅ CORRECT API Format

**IMPORTANT:** MCP tools use the `mcp__server__tool` format, NOT `query()` wrapper.

```javascript
// ✅ CORRECT: Direct MCP tool invocation
mcp__claude-flow__memory_usage({
  action: "store",
  key: "test",
  value: "Hello, MCP!",
  namespace: "quickstart"
})
```

```javascript
// ❌ WRONG: SDK pattern (don't use this)
query({
  mcp: {
    server: 'claude-flow',
    tool: 'memory_usage',
    params: { ... }
  }
})
```

### Test It Now

Try this simple "Hello World" example:

**Step 1: Store a value**
```javascript
mcp__claude-flow__memory_usage({
  action: "store",
  key: "first-test",
  value: "I just used MCP tools!",
  namespace: "quickstart"
})
```

**Expected Response:**
```json
{
  "success": true,
  "action": "store",
  "key": "first-test",
  "value": "I just used MCP tools!",
  "namespace": "quickstart",
  "timestamp": "2025-10-22T14:01:30.033Z"
}
```

**Step 2: Retrieve the value**
```javascript
mcp__claude-flow__memory_usage({
  action: "retrieve",
  key: "first-test",
  namespace: "quickstart"
})
```

**Expected Response:**
```json
{
  "success": true,
  "action": "retrieve",
  "key": "first-test",
  "value": "I just used MCP tools!",
  "found": true,
  "namespace": "quickstart"
}
```

🎉 **Success!** You just used your first MCP tool!

---

## 3. Quick Verification (1 minute)

### Verify Server Status

```javascript
// Check if Claude Flow is running
mcp__claude-flow__swarm_status()
```

**Expected Response:**
```json
{
  "success": true,
  "active_swarms": 0,
  "total_agents": 0,
  "status": "ready"
}
```

### Try a Simple Swarm

```javascript
// Initialize a swarm
mcp__claude-flow__swarm_init({
  topology: "mesh",
  maxAgents: 3,
  strategy: "balanced"
})
```

**Expected Response:**
```json
{
  "success": true,
  "swarmId": "swarm-abc123",
  "topology": "mesh",
  "maxAgents": 3,
  "status": "initialized"
}
```

---

## 4. Add More Servers (Optional)

### Flow Nexus (Cloud Features)

**Requires registration** - provides sandboxes, templates, and cloud execution.

```bash
# Add Flow Nexus server
claude mcp add flow-nexus npx flow-nexus@latest mcp start
```

**Register an account:**
```javascript
mcp__flow-nexus__user_register({
  email: "your-email@example.com",
  password: "secure-password"
})
```

**Login:**
```javascript
mcp__flow-nexus__user_login({
  email: "your-email@example.com",
  password: "secure-password"
})
```

**Test it:**
```javascript
// Check authentication status
mcp__flow-nexus__auth_status({
  detailed: true
})
```

### AgentDB (Advanced Memory)

**No authentication required** - provides causal memory, skills, and learning.

AgentDB is integrated with Claude Flow, so it's already available!

```javascript
// Store a reflexion episode
mcp__agentdb__reflexion_store({
  session_id: "test-session",
  task: "write_hello_world",
  reward: 1.0,
  success: true,
  critique: "Simple and effective implementation"
})
```

**Expected Response:**
```json
{
  "success": true,
  "episodeId": 1,
  "task": "write_hello_world",
  "reward": 1.0
}
```

### Agentic Payments (Payment Authorization)

**Requires keypair** - enables autonomous agent payment authorization.

```bash
# Add Agentic Payments server
claude mcp add agentic-payments npx agentic-payments mcp start
```

**Generate identity:**
```javascript
mcp__agentic-payments__generate_agent_identity({
  include_private_key: false  // Don't include private key in response
})
```

**Expected Response:**
```json
{
  "public_key": "ed25519:abc123...",
  "agent_id": "agent-xyz789"
}
```

---

## 5. Common Patterns (1 minute)

### Pattern 1: Memory Storage

```javascript
// Store configuration
mcp__claude-flow__memory_usage({
  action: "store",
  key: "app-config",
  value: JSON.stringify({
    apiUrl: "https://api.example.com",
    timeout: 5000
  }),
  namespace: "config"
})

// Retrieve configuration
const config = mcp__claude-flow__memory_usage({
  action: "retrieve",
  key: "app-config",
  namespace: "config"
})
```

### Pattern 2: Multi-Agent Orchestration

```javascript
// 1. Initialize swarm
mcp__claude-flow__swarm_init({
  topology: "hierarchical",
  maxAgents: 5
})

// 2. Spawn agents
mcp__claude-flow__agent_spawn({
  type: "researcher",
  capabilities: ["search", "analyze"]
})

// 3. Assign task
mcp__claude-flow__task_orchestrate({
  task: "Research best practices for API design",
  strategy: "adaptive",
  priority: "high"
})
```

### Pattern 3: Learning from Experience

```javascript
// 1. Store episode
mcp__agentdb__reflexion_store({
  session_id: "learning-1",
  task: "optimize_query",
  reward: 0.85,
  success: true,
  critique: "Query optimization reduced latency by 40%"
})

// 2. Retrieve similar experiences
mcp__agentdb__reflexion_retrieve({
  task: "database optimization",
  k: 5,
  only_successes: true,
  min_reward: 0.7
})

// 3. Create reusable skill
mcp__agentdb__skill_create({
  name: "query_optimizer",
  description: "Optimize SQL queries using indexing",
  success_rate: 0.85
})
```

---

## Troubleshooting

### Issue: "Tool not found"

**Symptom:** `Error: Unknown tool 'memory_usage'`

**Solution:**
1. Check server name is correct: `mcp__claude-flow__` (with hyphens)
2. Verify server is running: `claude mcp list`
3. Restart server: `claude mcp restart claude-flow`

### Issue: "AUTH_REQUIRED"

**Symptom:** `Error: AUTH_REQUIRED - Authentication needed`

**Solution:**
- **Flow Nexus:** Run `mcp__flow-nexus__user_login()`
- **Agentic Payments:** Generate identity with `generate_agent_identity()`

### Issue: "Connection timeout"

**Symptom:** Request hangs or times out

**Solution:**
1. Check internet connection
2. Verify server is installed: `npm list -g flow-nexus`
3. Update to latest version: `npm update -g flow-nexus`
4. Check system resources (memory, CPU)

### Issue: "Invalid parameters"

**Symptom:** `Error: INVALID_PARAMS - Required parameter missing`

**Solution:**
1. Check parameter names match documentation exactly
2. Ensure required parameters are provided
3. Verify parameter types (string vs number vs boolean)

**Example:**
```javascript
// ❌ WRONG: Missing required parameter
mcp__claude-flow__memory_usage({
  action: "store",
  key: "test"
  // Missing 'value' parameter!
})

// ✅ CORRECT: All required parameters
mcp__claude-flow__memory_usage({
  action: "store",
  key: "test",
  value: "Hello!"
})
```

---

## Quick Reference: Tool Naming

| Server | Prefix | Example Tool |
|--------|--------|-------------|
| Claude Flow | `mcp__claude-flow__` | `swarm_init` |
| Flow Nexus | `mcp__flow-nexus__` | `sandbox_create` |
| AgentDB | `mcp__agentdb__` | `reflexion_store` |
| Agentic Payments | `mcp__agentic-payments__` | `create_active_mandate` |

**Format:** `mcp__<server-name>__<tool_name>`

**Note:** Server names use hyphens (`claude-flow`), tool names use underscores (`swarm_init`)

---

## Next Steps

Now that you have MCP tools working, explore:

### 📚 Full Documentation
- **[MCP-TOOLS.md](/workspaces/agentic-flow/docs/guides/MCP-TOOLS.md)** - Complete reference (233+ tools)
- **[MCP-AUTHENTICATION.md](/workspaces/agentic-flow/docs/guides/MCP-AUTHENTICATION.md)** - Authentication setup
- **[MCP-TROUBLESHOOTING.md](/workspaces/agentic-flow/docs/guides/MCP-TROUBLESHOOTING.md)** - Common issues
- **[MCP-EXAMPLES.md](/workspaces/agentic-flow/docs/guides/MCP-EXAMPLES.md)** - Real-world examples

### 🎯 Try These Next
1. **Memory Management** - Store and retrieve context across sessions
2. **Swarm Orchestration** - Coordinate multiple agents
3. **Cloud Execution** - Run code in sandboxes
4. **Learning Systems** - Build self-improving agents

### 🔧 Advanced Features
- **Neural Networks** - Train and deploy AI models
- **GitHub Integration** - Automate repository tasks
- **Payment Authorization** - Enable autonomous payments
- **Causal Learning** - Learn from cause-effect relationships

---

## Server Comparison

| Feature | Claude Flow | Flow Nexus | AgentDB | Agentic Payments |
|---------|-------------|------------|---------|------------------|
| **Authentication** | ❌ None | ✅ Required | ❌ None | ✅ Keypair |
| **Cost** | 🆓 Free | 💰 Freemium | 🆓 Free | 💵 Transaction fees |
| **Tools** | 101 | 96 | 9 | 12 |
| **Best For** | Orchestration | Cloud execution | Memory | Payments |
| **Use Offline** | ✅ Yes | ❌ No | ✅ Yes | ❌ No |

---

## Performance Expectations

| Operation | Expected Time | Typical Use Case |
|-----------|--------------|------------------|
| Memory store/retrieve | < 10ms | Configuration, state |
| Swarm initialization | < 100ms | Multi-agent tasks |
| Agent spawn | < 50ms | Adding workers |
| Task orchestration | < 200ms | Coordinating work |
| Sandbox creation | < 3s | Cloud execution |
| Neural inference | < 50ms | AI predictions |

---

## Success Checklist

✅ **You're ready to use MCP tools if:**

- [x] You can call `mcp__claude-flow__memory_usage` successfully
- [x] You understand the `mcp__server__tool` format
- [x] You can retrieve stored values
- [x] You've verified server status
- [x] You know how to troubleshoot errors

---

## Getting Help

### Documentation
- **Architecture:** [mcp-tools-architecture.md](/workspaces/agentic-flow/docs/mcp-tools-architecture.md)
- **AgentDB Verification:** [agentdb-tools-verification.md](/workspaces/agentic-flow/docs/agentdb-tools-verification.md)

### Community
- **GitHub Issues:** https://github.com/ruvnet/claude-flow/issues
- **Discord:** https://discord.gg/claude-flow
- **Flow Nexus:** https://flow-nexus.ruv.io

### Common Questions

**Q: Which server should I start with?**
A: Claude Flow - no authentication, core features, 101 tools.

**Q: Do I need all servers?**
A: No. Start with Claude Flow, add others as needed.

**Q: Are MCP tools free?**
A: Claude Flow and AgentDB are free. Flow Nexus has free tier. Agentic Payments has transaction fees.

**Q: Can I use MCP tools offline?**
A: Claude Flow and AgentDB work offline. Flow Nexus and Agentic Payments require internet.

**Q: How do I update servers?**
A: Run `npm update -g <package-name>` (e.g., `npm update -g flow-nexus`)

---

## Quick Start Summary

**1. Install Claude Flow** → `claude mcp add claude-flow npx claude-flow@alpha mcp start`

**2. Test memory** → `mcp__claude-flow__memory_usage({ action: "store", key: "test", value: "hello" })`

**3. Verify status** → `mcp__claude-flow__swarm_status()`

**4. Read full docs** → [MCP-TOOLS.md](/workspaces/agentic-flow/docs/guides/MCP-TOOLS.md)

---

**Congratulations! You're now ready to build with MCP tools.** 🚀

**Time spent:** 5 minutes
**Tools mastered:** 3+
**Next milestone:** Build your first swarm

---

**Last Updated:** October 22, 2025
**Version:** 1.0.0
**Status:** ✅ Production Ready
