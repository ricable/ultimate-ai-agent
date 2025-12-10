# 🔧 MCP Tools: Complete Reference Guide

**~135 tools across 5 servers • Universal AI agent capabilities**

---

## 📑 Quick Navigation

[← Back to Main README](../../README.md) | [Quick Start →](#-quick-start-5-minutes) | [Authentication →](./MCP-AUTHENTICATION.md) | [Troubleshooting →](./MCP-TROUBLESHOOTING.md)

---

## 🚀 Quick Start (5 minutes)

### Step 1: Prerequisites (1 minute)

```bash
# Verify you have Node.js 18+
node --version

# Check Claude CLI is installed
claude --version
```

### Step 2: Add MCP Servers (2 minutes)

```bash
# 1. Claude Flow - Core orchestration (required)
claude mcp add claude-flow npx claude-flow@alpha mcp start

# 2. Flow Nexus - Cloud features (optional, requires registration)
claude mcp add flow-nexus npx flow-nexus@latest mcp start

# 3. Agentic Payments - Payment authorization (optional)
# Built-in with agentic-flow, auto-detected

# 4. Agentic Flow - Agent execution (optional)
claude mcp add agentic-flow npx agentic-flow@latest mcp start

# Verify servers are running
claude mcp list
```

### Step 3: First Tool Call (2 minutes)

```javascript
// Initialize a swarm (no auth required)
mcp__claude-flow__swarm_init({
  topology: 'mesh',
  maxAgents: 8,
  strategy: 'balanced'
});

// Response:
// {
//   success: true,
//   swarmId: 'swarm-abc123',
//   topology: 'mesh',
//   maxAgents: 8
// }
```

**✅ Success!** You're now using MCP tools. Continue to [Authentication Setup](#-prerequisites--authentication) for advanced features.

---

## 📋 Prerequisites & Authentication

### Setup Overview

| Server | Auth Required | Setup Time | Documentation |
|--------|---------------|------------|---------------|
| **Claude Flow** | ❌ No | 0 min | Ready to use |
| **AgentDB** | ❌ No (local) | 0 min | Built-in with Claude Flow |
| **Agentic Flow** | ❌ No | 0 min | Built-in package |
| **Flow Nexus** | ✅ Yes | 2 min | [Auth Guide](./MCP-AUTHENTICATION.md) |
| **Agentic Payments** | ✅ Keypair | 1 min | [Payment Setup](./MCP-AUTHENTICATION.md#agentic-payments) |

### Quick Authentication Setup

**Flow Nexus (for cloud features):**
```bash
# Register once
npx flow-nexus@latest register

# Login each session
npx flow-nexus@latest login
```

**Agentic Payments (for payment features):**
```javascript
// Generate keypair once
const identity = mcp__agentic-payments__generate_agent_identity({
  include_private_key: true
});
// Store private key securely!
```

**→ Full setup guide:** [MCP-AUTHENTICATION.md](./MCP-AUTHENTICATION.md)

---

## 🎯 What are MCP Tools?

MCP (Model Context Protocol) tools are standardized AI agent capabilities that enable agents to interact with external systems, services, and data sources. Think of them as "skills" or "abilities" that agents can use to perform tasks beyond text generation.

### The Problem

Traditional AI agents are limited to:
- **Text-only responses**: Can't interact with real systems
- **No memory**: Forget everything between sessions
- **No external data**: Limited to training data
- **No automation**: Can't execute actions
- **Isolated**: Can't coordinate with other agents

**Example**: An agent reviewing code can't actually run tests, check GitHub PRs, or update documentation - it can only suggest these actions.

### The Solution

MCP tools provide standardized interfaces for:
- **System Integration**: File I/O, Git operations, API calls
- **Data Access**: Databases, search engines, web scraping
- **Agent Coordination**: Swarm management, task orchestration
- **Memory & Learning**: Persistent storage, pattern recognition
- **Automation**: Workflow execution, event handling

**Results**:
- Agents can execute real actions, not just suggest them
- Cross-session memory enables learning and improvement
- Multi-agent coordination for complex tasks
- Integration with external services

---

## 🚀 MCP Server Overview

### Available Servers (Verified Tool Counts)

| Server | Tools | Focus Area | Auth | Installation |
|--------|-------|------------|------|--------------|
| **Claude Flow** | 46 | Orchestration, memory, neural | None | `claude mcp add claude-flow npx claude-flow@alpha mcp start` |
| **AgentDB** | 9 | Advanced memory, causal learning | None (local) | Built-in with Claude Flow |
| **Flow Nexus** | 70+ | Cloud execution, sandboxes, templates | Required | `claude mcp add flow-nexus npx flow-nexus@latest mcp start` |
| **Agentic Payments** | 10 | Payment authorization, mandates | Keypair | Built-in with agentic-flow |
| **Agentic Flow** | ~10 | Agent execution, optimization | None | `claude mcp add agentic-flow npx agentic-flow@latest mcp start` |

**Total**: ~135+ verified MCP tools

**Note**: Tool counts verified through source code analysis (October 2025). See [AgentDB Verification Report](../agentdb-tools-verification.md) and [Tool Architecture](../mcp-tools-architecture.md) for details.

---

## 📚 Tool Categories

### Category Overview

| Category | Tools | Primary Server | Auth Required |
|----------|-------|----------------|---------------|
| [1. AgentDB - Advanced Memory](#1%EF%B8%8F⃣-agentdb---advanced-memory-9-tools) | 9 | AgentDB | No |
| [2. Swarm & Agent Orchestration](#2%EF%B8%8F⃣-swarm--agent-orchestration-26-tools) | 26 | Claude Flow | No |
| [3. Memory & Learning](#3%EF%B8%8F⃣-memory--learning-20-tools) | 20 | Claude Flow | No |
| [4. Neural Networks & AI](#4%EF%B8%8F⃣-neural-networks--ai-15-tools) | 15 | Claude Flow, Flow Nexus | Mixed |
| [5. Cloud Execution & Sandboxes](#5%EF%B8%8F⃣-cloud-execution--sandboxes-24-tools) | 24 | Flow Nexus | Yes |
| [6. GitHub Integration](#6%EF%B8%8F⃣-github-integration-16-tools) | 16 | Claude Flow | No |
| [7. Payment Authorization](#7%EF%B8%8F⃣-payment-authorization-10-tools) | 10 | Agentic Payments | Keypair |
| [8. Workflow Automation](#8%EF%B8%8F⃣-workflow-automation-22-tools) | 22 | Flow Nexus | Yes |
| [9. Performance & Monitoring](#9%EF%B8%8F⃣-performance--monitoring-18-tools) | 18 | Claude Flow | No |
| [10. App Store & Templates](#-app-store--templates-12-tools) | 12 | Flow Nexus | Yes |

---

## 1️⃣ AgentDB - Advanced Memory (9 tools)

State-of-the-art agent memory with causal learning and explainable recall.

### Reflexion Memory (Self-Critique Learning)

#### `reflexion_store`

Store an episode with self-critique for learning from experience.

```javascript
// Store successful authentication implementation
mcp__agentdb__reflexion_store({
  session_id: 'session-123',
  task: 'implement_oauth2_authentication',
  reward: 0.95,
  success: true,
  critique: 'OAuth2 PKCE flow worked perfectly for mobile apps',
  input: 'Need secure authentication for mobile app',
  output: 'Implemented OAuth2 with authorization code + PKCE',
  latency_ms: 1200,
  tokens: 500
});
```

**Response:**
```json
{
  "success": true,
  "episodeId": 42,
  "message": "✅ Stored episode #42\nTask: implement_oauth2_authentication\nReward: 0.95\nSuccess: true"
}
```

**Common Errors:**
- `INVALID_REWARD`: Reward must be 0-1
- `MISSING_SESSION`: Provide session_id

**See Also:**
- [`reflexion_retrieve`](#reflexion_retrieve) - Find similar past episodes
- [`skill_create`](#skill_create) - Convert successful episodes to skills

---

#### `reflexion_retrieve`

Retrieve relevant past episodes for learning from experience.

```javascript
// Find past successful authentication implementations
mcp__agentdb__reflexion_retrieve({
  task: 'authentication implementation',
  k: 10,
  only_successes: true,
  min_reward: 0.8
});
```

**Response:**
```json
{
  "success": true,
  "episodes": [
    {
      "id": 42,
      "task": "implement_oauth2_authentication",
      "reward": 0.95,
      "similarity": 0.87,
      "critique": "OAuth2 PKCE flow worked perfectly..."
    }
  ],
  "count": 10
}
```

---

### Skill Library (Reusable Agent Skills)

#### `skill_create`

Create a reusable skill in the skill library.

```javascript
// Create JWT authentication skill
mcp__agentdb__skill_create({
  name: 'jwt_auth',
  description: 'Generate secure JWT tokens with 24h expiry',
  code: "const jwt = require('jsonwebtoken'); jwt.sign(payload, secret, {expiresIn: '24h'});",
  success_rate: 0.92
});
```

**Response:**
```json
{
  "success": true,
  "skillId": 15,
  "message": "✅ Created skill #15: jwt_auth"
}
```

---

#### `skill_search`

Search for applicable skills by semantic similarity.

```javascript
// Find authentication-related skills
mcp__agentdb__skill_search({
  task: 'authentication security',
  k: 5,
  min_success_rate: 0.8
});
```

**Response:**
```json
{
  "success": true,
  "skills": [
    {
      "id": 15,
      "name": "jwt_auth",
      "description": "Generate secure JWT tokens...",
      "successRate": 0.92,
      "similarity": 0.85
    }
  ]
}
```

---

### Causal Memory (Cause-Effect Learning)

#### `causal_add_edge`

Add a causal relationship between actions and outcomes.

```javascript
// Record that adding tests improves code quality
mcp__causal_add_edge({
  cause: 'add_unit_tests',
  effect: 'code_quality_improvement',
  uplift: 0.25,
  confidence: 0.95,
  sample_size: 100
});
```

**Response:**
```json
{
  "success": true,
  "edgeId": 7,
  "message": "✅ Added causal edge #7\nadd_unit_tests → code_quality_improvement\nUplift: 0.25"
}
```

---

#### `causal_query`

Query causal effects to understand what actions cause outcomes.

```javascript
// Check if adding tests causally improves code quality
mcp__causal_query({
  cause: 'add_unit_tests',
  effect: 'code_quality_improvement',
  min_confidence: 0.7,
  min_uplift: 0.1,
  limit: 5
});
```

**Response:**
```json
{
  "success": true,
  "edges": [
    {
      "id": 7,
      "cause": "add_unit_tests",
      "effect": "code_quality_improvement",
      "uplift": 0.25,
      "confidence": 0.95,
      "sampleSize": 100
    }
  ]
}
```

---

### Explainable Recall (Provenance Certificates)

#### `recall_with_certificate`

Retrieve memories with causal utility scoring and provenance certificate.

```javascript
// Retrieve memories with causal utility weighting
mcp__recall_with_certificate({
  query: 'successful API optimization strategies',
  k: 5,
  alpha: 0.6,  // 60% weight on similarity
  beta: 0.3,   // 30% weight on causal uplift
  gamma: 0.1   // 10% weight on recency
});
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "type": "skill",
      "id": 15,
      "content": "jwt_auth implementation...",
      "similarity": 0.85,
      "uplift": 0.25,
      "score": 0.78
    }
  ],
  "certificateId": "cert-abc123",
  "count": 5
}
```

---

### Nightly Learner (Pattern Discovery)

#### `learner_discover`

Automatically discover causal patterns from episode history.

```javascript
// Discover causal patterns from past episodes
mcp__learner_discover({
  min_attempts: 3,
  min_success_rate: 0.6,
  min_confidence: 0.7,
  dry_run: true  // Preview first
});
```

**Response:**
```json
{
  "success": true,
  "patterns": [
    {
      "cause": "add_unit_tests",
      "effect": "code_quality_improvement",
      "uplift": 0.25,
      "sampleSize": 100,
      "confidence": 0.95
    }
  ],
  "count": 5,
  "dryRun": true
}
```

---

### Database Utilities

#### `db_stats`

Get database statistics showing record counts.

```javascript
// Get database statistics
mcp__db_stats({});
```

**Response:**
```json
{
  "success": true,
  "stats": {
    "causalEdges": 42,
    "experiments": 156,
    "observations": 1024,
    "episodes": 387,
    "skills": 23
  }
}
```

---

## 2️⃣ Swarm & Agent Orchestration (26 tools)

Coordinate multiple agents for complex tasks.

### Core Swarm Tools

#### `swarm_init`

Initialize swarm with specific topology.

```javascript
// Initialize mesh topology swarm
mcp__claude-flow__swarm_init({
  topology: 'mesh',      // mesh, hierarchical, ring, star
  maxAgents: 8,
  strategy: 'balanced'   // balanced, specialized, adaptive
});
```

**Response:**
```json
{
  "success": true,
  "swarmId": "swarm-abc123",
  "topology": "mesh",
  "maxAgents": 8,
  "strategy": "balanced"
}
```

**Common Errors:**
- `INVALID_TOPOLOGY`: Must be mesh, hierarchical, ring, or star
- `MAX_AGENTS_EXCEEDED`: maxAgents must be 1-100

**See Also:**
- [`agent_spawn`](#agent_spawn) - Add agents to swarm
- [`task_orchestrate`](#task_orchestrate) - Assign tasks
- [`swarm_status`](#swarm_status) - Monitor progress

---

#### `agent_spawn`

Spawn specialized agent in swarm.

```javascript
// Spawn backend developer agent
mcp__claude-flow__agent_spawn({
  type: 'coder',         // researcher, analyst, optimizer, coordinator
  name: 'backend-dev',
  capabilities: ['api-design', 'database', 'testing']
});
```

**Response:**
```json
{
  "success": true,
  "agentId": "agent-456",
  "type": "coder",
  "name": "backend-dev",
  "capabilities": ["api-design", "database", "testing"]
}
```

---

#### `task_orchestrate`

Orchestrate complex task across swarm.

```javascript
// Build REST API with authentication
mcp__claude-flow__task_orchestrate({
  task: 'Build REST API with authentication',
  strategy: 'adaptive',  // parallel, sequential, adaptive
  priority: 'high',
  maxAgents: 5
});
```

**Response:**
```json
{
  "success": true,
  "taskId": "task-789",
  "strategy": "adaptive",
  "priority": "high",
  "assignedAgents": ["agent-456", "agent-457"]
}
```

---

### Monitoring & Status

#### `swarm_status`

Get swarm status and details.

```javascript
// Check swarm health
mcp__claude-flow__swarm_status({
  swarmId: 'swarm-abc123'
});
```

**Response:**
```json
{
  "success": true,
  "swarmId": "swarm-abc123",
  "status": "active",
  "topology": "mesh",
  "activeAgents": 5,
  "maxAgents": 8,
  "tasksRunning": 3,
  "tasksCompleted": 12
}
```

---

#### `agent_list`

List active agents in swarm.

```javascript
// List all active agents
mcp__claude-flow__agent_list({
  filter: 'active'  // all, active, idle, busy
});
```

**Response:**
```json
{
  "success": true,
  "agents": [
    {
      "agentId": "agent-456",
      "type": "coder",
      "status": "busy",
      "currentTask": "task-789"
    }
  ],
  "count": 5
}
```

---

#### `agent_metrics`

Get agent performance metrics.

```javascript
// Get agent performance
mcp__claude-flow__agent_metrics({
  agentId: 'agent-456',
  metric: 'performance'  // cpu, memory, tasks, performance
});
```

**Response:**
```json
{
  "success": true,
  "agentId": "agent-456",
  "metrics": {
    "tasksCompleted": 12,
    "successRate": 0.92,
    "avgLatencyMs": 1200,
    "tokensUsed": 15000
  }
}
```

---

### Advanced Swarm Operations

#### `swarm_scale`

Scale swarm up or down.

```javascript
// Scale to 15 agents
mcp__claude-flow__swarm_scale({
  swarmId: 'swarm-abc123',
  targetSize: 15
});
```

**Response:**
```json
{
  "success": true,
  "swarmId": "swarm-abc123",
  "previousSize": 8,
  "newSize": 15,
  "spawned": 7
}
```

---

#### `swarm_destroy`

Destroy swarm and cleanup resources.

```javascript
// Cleanup swarm
mcp__claude-flow__swarm_destroy({
  swarmId: 'swarm-abc123'
});
```

**Response:**
```json
{
  "success": true,
  "swarmId": "swarm-abc123",
  "agentsTerminated": 15,
  "message": "Swarm destroyed successfully"
}
```

---

## 3️⃣ Memory & Learning (20 tools)

Persistent memory across sessions with learning capabilities.

### Memory Storage & Retrieval

#### `memory_usage`

Store, retrieve, list, or delete memory with TTL and namespacing.

**Store memory:**
```javascript
// Store API design pattern
mcp__claude-flow__memory_usage({
  action: 'store',
  key: 'api-design-pattern',
  value: JSON.stringify({
    pattern: 'REST with JWT auth',
    successRate: 0.95,
    usageCount: 47
  }),
  namespace: 'backend-patterns',
  ttl: 2592000  // 30 days in seconds
});
```

**Response:**
```json
{
  "success": true,
  "action": "store",
  "key": "api-design-pattern",
  "namespace": "backend-patterns"
}
```

**Retrieve memory:**
```javascript
// Retrieve pattern
mcp__claude-flow__memory_usage({
  action: 'retrieve',
  key: 'api-design-pattern',
  namespace: 'backend-patterns'
});
```

**Response:**
```json
{
  "success": true,
  "action": "retrieve",
  "key": "api-design-pattern",
  "value": "{\"pattern\":\"REST with JWT auth\",\"successRate\":0.95,\"usageCount\":47}",
  "namespace": "backend-patterns"
}
```

**List memories:**
```javascript
// List all in namespace
mcp__claude-flow__memory_usage({
  action: 'list',
  namespace: 'backend-patterns'
});
```

**Delete memory:**
```javascript
// Delete old pattern
mcp__claude-flow__memory_usage({
  action: 'delete',
  key: 'old-pattern',
  namespace: 'backend-patterns'
});
```

---

#### `memory_search`

Search memories with pattern matching.

```javascript
// Search for authentication patterns
mcp__claude-flow__memory_search({
  pattern: 'authentication',
  namespace: 'backend-patterns',
  limit: 10
});
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "key": "api-design-pattern",
      "value": "{\"pattern\":\"REST with JWT auth\"...}",
      "similarity": 0.87
    }
  ],
  "count": 10
}
```

---

## 4️⃣ Neural Networks & AI (15 tools)

Train and deploy neural networks with WASM acceleration.

### Neural Training

#### `neural_train`

Train neural patterns with WASM acceleration.

```javascript
// Train coordination patterns
mcp__claude-flow__neural_train({
  pattern_type: 'coordination',  // optimization, prediction
  training_data: JSON.stringify({
    inputs: [[1, 0], [0, 1], [1, 1]],
    outputs: [[1], [1], [0]]
  }),
  epochs: 50
});
```

**Response:**
```json
{
  "success": true,
  "modelId": "model-123",
  "patternType": "coordination",
  "epochs": 50,
  "finalLoss": 0.012
}
```

---

#### `neural_status`

Check neural network training status.

```javascript
// Check model status
mcp__claude-flow__neural_status({
  modelId: 'model-123'
});
```

**Response:**
```json
{
  "success": true,
  "modelId": "model-123",
  "status": "trained",
  "accuracy": 0.95,
  "trainingTime": 1200
}
```

---

### Neural Inference

#### `neural_predict`

Run inference on trained model.

```javascript
// Predict with model
mcp__claude-flow__neural_predict({
  modelId: 'model-123',
  input: JSON.stringify({ features: [1, 0, 1] })
});
```

**Response:**
```json
{
  "success": true,
  "modelId": "model-123",
  "prediction": [0.92],
  "confidence": 0.95
}
```

---

## 5️⃣ Cloud Execution & Sandboxes (24 tools)

Execute code in isolated cloud environments (Flow Nexus).

**⚠️ Authentication Required:** [Setup Guide](./MCP-AUTHENTICATION.md#flow-nexus)

### Sandbox Management

#### `sandbox_create`

Create execution sandbox with environment variables.

```javascript
// Create Node.js sandbox
mcp__flow-nexus__sandbox_create({
  template: 'node',  // python, react, nextjs, claude-code
  name: 'api-dev',
  env_vars: {
    DATABASE_URL: 'postgresql://...',
    API_KEY: 'sk-...'
  },
  timeout: 3600  // 1 hour
});
```

**Response:**
```json
{
  "success": true,
  "sandbox_id": "sandbox-789",
  "template": "node",
  "status": "running"
}
```

**Common Errors:**
- `AUTH_REQUIRED`: Login first with `user_login`
- `INVALID_TEMPLATE`: Must be node, python, react, nextjs, or claude-code
- `INSUFFICIENT_CREDITS`: Add credits to account

---

#### `sandbox_execute`

Execute code in sandbox.

```javascript
// Run code in sandbox
mcp__flow-nexus__sandbox_execute({
  sandbox_id: 'sandbox-789',
  code: 'console.log("Hello from sandbox!")',
  language: 'javascript',
  timeout: 60
});
```

**Response:**
```json
{
  "success": true,
  "sandbox_id": "sandbox-789",
  "output": "Hello from sandbox!\n",
  "exitCode": 0,
  "executionTime": 250
}
```

---

#### `sandbox_upload`

Upload file to sandbox.

```javascript
// Upload config file
mcp__flow-nexus__sandbox_upload({
  sandbox_id: 'sandbox-789',
  file_path: '/app/config.json',
  content: JSON.stringify({ port: 3000 })
});
```

**Response:**
```json
{
  "success": true,
  "sandbox_id": "sandbox-789",
  "file_path": "/app/config.json",
  "size": 15
}
```

---

## 6️⃣ GitHub Integration (16 tools)

Comprehensive GitHub workflow automation.

### Repository Management

#### `github_repo_analyze`

Analyze repository code quality, performance, or security.

```javascript
// Analyze code quality
mcp__claude-flow__github_repo_analyze({
  repo: 'ruvnet/agentic-flow',
  analysis_type: 'code_quality'  // performance, security
});
```

**Response:**
```json
{
  "success": true,
  "repo": "ruvnet/agentic-flow",
  "analysisType": "code_quality",
  "score": 0.87,
  "issues": [
    {
      "file": "src/index.ts",
      "line": 42,
      "message": "Consider adding error handling"
    }
  ]
}
```

---

## 7️⃣ Payment Authorization (10 tools)

AI-native payment authorization with Active Mandates.

**⚠️ Keypair Required:** [Setup Guide](./MCP-AUTHENTICATION.md#agentic-payments)

### Mandate Management

#### `create_active_mandate`

Create payment mandate with spend caps and merchant restrictions.

```javascript
// Create monthly mandate
mcp__agentic-payments__create_active_mandate({
  agent: 'shopping-bot@agentics',
  holder: 'user-123',
  amount: 12000,  // $120.00 in cents
  currency: 'USD',
  period: 'monthly',
  kind: 'intent',
  merchant_allow: ['amazon.com', 'ebay.com'],
  expires_at: '2025-01-12T00:00:00Z'
});
```

**Response:**
```json
{
  "success": true,
  "mandateId": "mandate-789",
  "agent": "shopping-bot@agentics",
  "amount": 12000,
  "period": "monthly"
}
```

---

## 8️⃣ Workflow Automation (22 tools)

Event-driven workflow execution with message queues (Flow Nexus).

**⚠️ Authentication Required:** [Setup Guide](./MCP-AUTHENTICATION.md#flow-nexus)

### Workflow Creation & Execution

#### `workflow_create`

Create event-driven workflow.

```javascript
// Create code review pipeline
mcp__flow-nexus__workflow_create({
  name: 'code-review-pipeline',
  steps: [
    { type: 'checkout', params: { branch: 'main' } },
    { type: 'test', params: { coverage: 80 } },
    { type: 'review', params: { reviewers: 2 } },
    { type: 'merge', params: { strategy: 'squash' } }
  ],
  triggers: ['pull_request.opened', 'push.main']
});
```

**Response:**
```json
{
  "success": true,
  "workflow_id": "workflow-456",
  "name": "code-review-pipeline",
  "steps": 4
}
```

---

## 9️⃣ Performance & Monitoring (18 tools)

Real-time metrics, benchmarks, and optimization.

### Performance Tracking

#### `performance_report`

Generate performance report with metrics.

```javascript
// Get 24h performance report
mcp__claude-flow__performance_report({
  timeframe: '24h',  // 7d, 30d
  format: 'detailed'  // summary, json
});
```

**Response:**
```json
{
  "success": true,
  "timeframe": "24h",
  "metrics": {
    "totalRequests": 1250,
    "avgLatencyMs": 150,
    "p95LatencyMs": 350,
    "errorRate": 0.02
  }
}
```

---

## 🔟 App Store & Templates (12 tools)

Deploy pre-built applications and templates (Flow Nexus).

**⚠️ Authentication Required:** [Setup Guide](./MCP-AUTHENTICATION.md#flow-nexus)

### Template Management

#### `template_list`

List available deployment templates.

```javascript
// List web app templates
mcp__flow-nexus__template_list({
  category: 'web-apps',
  featured: true,
  limit: 20
});
```

**Response:**
```json
{
  "success": true,
  "templates": [
    {
      "template_id": "template-123",
      "name": "React + Express Starter",
      "category": "web-apps",
      "featured": true
    }
  ],
  "count": 20
}
```

---

## 🛠️ Response Types & Error Handling

### Standard Response Structure

**Success Response:**
```typescript
interface SuccessResponse<T> {
  success: true;
  data: T;
  metadata?: {
    timestamp: string;
    latency_ms: number;
    cache_hit?: boolean;
  };
}
```

**Error Response:**
```typescript
interface ErrorResponse {
  success: false;
  error: {
    code: string;           // e.g., "AUTH_REQUIRED"
    message: string;        // Human-readable
    details?: object;       // Additional context
    suggestion?: string;    // How to fix
  };
}
```

### Common Error Codes

| Code | Meaning | Solution |
|------|---------|----------|
| `AUTH_REQUIRED` | Authentication needed | Run login/register first |
| `INVALID_PARAMS` | Bad parameters | Check parameter types/values |
| `NOT_FOUND` | Resource doesn't exist | Verify ID/name |
| `RATE_LIMITED` | Too many requests | Wait and retry |
| `INSUFFICIENT_CREDITS` | Not enough balance | Add credits or upgrade tier |
| `SERVER_ERROR` | Internal error | Retry or report issue |

### Error Handling Pattern

```javascript
try {
  const result = mcp__flow-nexus__sandbox_create({
    template: 'node',
    name: 'test-sandbox'
  });

  if (result.success) {
    console.log('Sandbox created:', result.data.sandbox_id);
  }
} catch (error) {
  if (error.code === 'AUTH_REQUIRED') {
    // Handle authentication
    mcp__flow-nexus__user_login({ ... });
    // Retry operation
  } else if (error.code === 'RATE_LIMITED') {
    // Wait and retry
    await sleep(error.details.retry_after_ms);
  } else {
    // Log and report
    console.error('Unexpected error:', error);
  }
}
```

---

## 📊 Tool Performance Benchmarks

### Latency Comparison

**Note:** These benchmarks represent typical performance under normal conditions. Actual latency may vary based on network conditions, system load, and data volume.

| Tool Category | Average Latency | P99 Latency | Throughput | Server |
|---------------|----------------|-------------|------------|--------|
| **Memory Operations** | 5ms | 12ms | 10K ops/sec | Claude Flow |
| **Swarm Management** | 50ms | 120ms | 500 ops/sec | Claude Flow |
| **Neural Inference** | 15ms | 35ms | 2K ops/sec | Claude Flow |
| **Sandbox Creation** | 2s | 5s | 50 ops/min | Flow Nexus |
| **GitHub Integration** | 200ms | 800ms | 100 ops/sec | Claude Flow |
| **AgentDB Queries** | 10ms | 25ms | 5K ops/sec | AgentDB |

### Cost Optimization

| Operation | Without MCP | With MCP | Savings |
|-----------|------------|----------|---------|
| **Code Review** | $0.15 (LLM) | $0.02 (cached) | 87% |
| **Memory Retrieval** | $0.05 (LLM) | $0.00 (local) | 100% |
| **Task Orchestration** | $0.30 (manual) | $0.05 (auto) | 83% |

**Optimization Tips:**
- Batch operations when possible
- Reuse sandboxes instead of creating new ones
- Use memory caching for frequent queries
- Enable WASM acceleration for neural operations

---

## 🔗 Related Documentation

### Core Components
- [← Back to Main README](../../README.md)
- [Quick Start Guide](./MCP-QUICKSTART.md) - 5-minute getting started
- [Authentication Setup](./MCP-AUTHENTICATION.md) - Flow Nexus & Agentic Payments auth
- [Troubleshooting Guide](./MCP-TROUBLESHOOTING.md) - Common issues & solutions
- [Usage Examples](./MCP-EXAMPLES.md) - Real-world patterns

### Technical References
- [AgentDB Verification Report](../agentdb-tools-verification.md) - Tool count verification
- [MCP Tools Architecture](../mcp-tools-architecture.md) - Design decisions
- [Agent Booster (Code Transformations)](./AGENT-BOOSTER.md)
- [ReasoningBank (Learning Memory)](./REASONINGBANK.md)
- [Multi-Model Router (Cost Optimization)](./MULTI-MODEL-ROUTER.md)

### Advanced Topics
- [Deployment Options](./DEPLOYMENT.md)
- [Performance Benchmarks](../../benchmarks/README.md)

### External Resources
- [MCP Specification](https://modelcontextprotocol.io)
- [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk)
- [Flow Nexus Platform](https://flow-nexus.ruv.io)

---

## ❓ FAQ

### Getting Started

**Q: Which MCP server should I start with?**
A: Claude Flow (no auth required, core orchestration features).

**Q: Do I need all servers?**
A: No. Start with Claude Flow, add others as needed:
- Flow Nexus: Cloud execution, sandboxes
- Agentic Payments: Payment authorization
- AgentDB: Advanced memory (included in Claude Flow)

**Q: How do I know which tools I have access to?**
A: Run `claude mcp list` to see installed servers.

### Authentication

**Q: Why does Flow Nexus require registration?**
A: Cloud resources (sandboxes, storage) require account management.

**Q: Can I use Flow Nexus tools without registering?**
A: No. Registration is required for cloud features.

**Q: Is my password stored securely?**
A: Yes. Passwords are hashed with bcrypt, never stored in plaintext.

### Pricing & Limits

**Q: Are MCP tools free?**
A: Mixed:
- Claude Flow: Free, unlimited
- AgentDB: Free, local storage
- Flow Nexus: Free tier + paid upgrades
- Agentic Payments: Transaction fees

**Q: What are Flow Nexus rate limits?**
A: Free tier: 100 requests/hour. Paid tiers: Higher limits.

---

## 🤝 Contributing

MCP tools are part of the agentic-flow ecosystem. Contributions welcome!

**Areas for Contribution:**
- Additional tool implementations
- Performance optimizations
- New MCP server integrations
- Documentation improvements
- Usage examples

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - see [LICENSE](../../LICENSE) for details.

---

**Access ~135 AI agent capabilities across 5 MCP servers. Universal integration.** 🔧

[← Back to Main README](../../README.md)
