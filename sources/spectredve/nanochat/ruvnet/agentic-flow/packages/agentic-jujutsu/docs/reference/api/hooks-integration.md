# Agentic-Flow Hooks Integration

Complete guide for integrating agentic-jujutsu with the agentic-flow hooks system.

## Overview

The hooks integration enables automatic tracking of all jj operations within agentic-flow workflows, providing:

- **Automatic operation logging**: Every file edit, commit, and merge is tracked
- **AgentDB synchronization**: Operation history is stored for pattern learning
- **Multi-agent coordination**: Agents can see and learn from each other's work
- **Conflict-free collaboration**: jj's merge capabilities enable safe concurrent work

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Agentic-Flow                        │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │  Agent 1  │  │  Agent 2  │  │  Agent 3  │         │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘          │
│        │             │              │                │
│        └─────────────┼──────────────┘                │
│                      │                               │
│              ┌───────▼────────┐                      │
│              │  Hooks System   │                     │
│              └───────┬────────┘                      │
└──────────────────────┼───────────────────────────────┘
                       │
           ┌───────────▼────────────┐
           │ JJHooksIntegration     │
           ├────────────────────────┤
           │ - on_pre_task()        │
           │ - on_post_edit()       │
           │ - on_post_task()       │
           │ - on_conflict()        │
           └───────────┬────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼────────┐ ┌──▼─────┐ ┌─────▼──────┐
│   JJ Wrapper    │ │ AgentDB│ │   Memory   │
│  (Operations)   │ │  Sync  │ │   Store    │
└─────────────────┘ └────────┘ └────────────┘
```

## Setup

### 1. Installation

```bash
# Install agentic-jujutsu
cd packages/agentic-jujutsu
cargo build --release

# Install jj CLI tool
cargo install --git https://github.com/jj-vcs/jj.git jj-cli
```

### 2. Configuration

Create a configuration file or use environment variables:

```bash
# Enable AgentDB sync
export AGENTDB_SYNC_ENABLED=true
export AGENTDB_SYNC_FILE=/path/to/agentdb-episodes.jsonl

# Configure jj
export JJ_PATH=/usr/local/bin/jj
export JJ_REPO_PATH=/path/to/repo
```

### 3. Initialize Repository

```bash
# Initialize jj repository
jj init --git

# Configure agent information
jj config set --user user.name "AI Agent"
jj config set --user user.email "agent@example.com"
```

## Hook Points

### 1. Pre-Task Hook

Called before an agent begins work on a task.

**Purpose:**
- Initialize session state
- Prepare repository
- Create session marker
- Restore previous context

**Usage:**

```bash
jj-agent-hook pre-task \
  --agent-id coder-1 \
  --session-id swarm-001 \
  --description "Implement authentication"
```

**Rust API:**

```rust
use agentic_jujutsu::{JJHooksIntegration, HookContext};

let ctx = HookContext::new(
    "coder-1".to_string(),
    "swarm-001".to_string(),
    "Implement authentication".to_string(),
);

let event = integration.on_pre_task(ctx).await?;
```

**TypeScript API:**

```typescript
await integration.onPreTask("Implement authentication");
```

### 2. Post-Edit Hook

Called after a file edit operation.

**Purpose:**
- Auto-commit changes
- Log operation
- Sync to AgentDB
- Update coordination state

**Usage:**

```bash
jj-agent-hook post-edit \
  --file src/auth.rs \
  --agent-id coder-1 \
  --session-id swarm-001
```

**Rust API:**

```rust
let operation = integration.on_post_edit("src/auth.rs", ctx).await?;
```

**TypeScript API:**

```typescript
await integration.onPostEdit("src/auth.rs", "Added login function");
```

### 3. Post-Task Hook

Called after task execution completes.

**Purpose:**
- Gather session operations
- Generate summary
- Store in memory
- Clean up session state

**Usage:**

```bash
jj-agent-hook post-task \
  --agent-id coder-1 \
  --session-id swarm-001
```

**Rust API:**

```rust
let operations = integration.on_post_task(ctx).await?;
```

**TypeScript API:**

```typescript
const operations = await integration.onPostTask();
```

### 4. Conflict Detection Hook

Called when a merge conflict is detected.

**Purpose:**
- Notify coordination system
- Log for learning
- Trigger resolution workflow

**Usage:**

```bash
jj-agent-hook detect-conflicts \
  --agent-id coder-1 \
  --session-id swarm-001
```

**Rust API:**

```rust
let event = integration.on_conflict_detected(conflicts, ctx).await?;
```

**TypeScript API:**

```typescript
await integration.onConflictDetected(['src/auth.rs', 'src/user.rs']);
```

## AgentDB Sync

### Episode Structure

Each jj operation is stored in AgentDB as an episode:

```json
{
  "sessionId": "swarm-001",
  "task": "Implement authentication",
  "agentId": "coder-1",
  "input": "Requirements: JWT-based auth",
  "output": "Created auth module with login/logout",
  "critique": "Good structure, needs rate limiting",
  "success": true,
  "reward": 0.95,
  "latencyMs": 1500,
  "tokensUsed": 250,
  "operation": {
    "id": "op-123",
    "operationType": "Describe",
    "description": "Added authentication",
    "timestamp": 1699564800,
    "user": "coder-1",
    "metadata": {}
  },
  "timestamp": 1699564800
}
```

### Querying Patterns

```rust
// Query similar past operations
let episodes = agentdb_sync
    .query_similar_operations("implement authentication", 5)
    .await?;

// Get statistics for a task type
let stats = agentdb_sync
    .get_task_statistics("authentication")
    .await?;

println!("Success rate: {:.2}%", stats.success_rate() * 100.0);
```

## Integration Patterns

### Pattern 1: Single Agent with Hooks

```typescript
import { createHooksIntegration, withHooks } from '@agentic-flow/jujutsu';

const integration = await createHooksIntegration(
    config,
    'session-001',
    'my-agent',
    true // Enable AgentDB
);

const { result, operations } = await withHooks(
    integration,
    'Implement feature X',
    async (hooks) => {
        // Do work
        await hooks.onPostEdit('src/feature.rs');
        await hooks.onPostEdit('tests/feature_test.rs');

        return { success: true };
    }
);
```

### Pattern 2: Multi-Agent Coordination

```rust
// Agent 1: Backend
let backend_ctx = HookContext::new(
    "backend-agent".to_string(),
    session_id.clone(),
    "Implement API".to_string(),
);
backend_integration.on_pre_task(backend_ctx.clone()).await?;
backend_integration.on_post_edit("src/api.rs", backend_ctx.clone()).await?;
backend_integration.on_post_task(backend_ctx).await?;

// Agent 2: Frontend (concurrent)
let frontend_ctx = HookContext::new(
    "frontend-agent".to_string(),
    session_id.clone(),
    "Create UI".to_string(),
);
frontend_integration.on_pre_task(frontend_ctx.clone()).await?;
frontend_integration.on_post_edit("ui/App.tsx", frontend_ctx.clone()).await?;
frontend_integration.on_post_task(frontend_ctx).await?;
```

### Pattern 3: Learning from History

```typescript
// Import historical operations
const ops = await jj.getOperations(100);
for (const op of ops) {
    await agentdb.storeEpisode({
        sessionId: 'history-import',
        task: op.description,
        operation: op,
        success: true,
        reward: 1.0,
    });
}

// Query for similar work
const similar = await agentdb.searchPatterns('refactor authentication');

// Use learned patterns
for (const episode of similar) {
    console.log('Past approach:', episode.output);
}
```

## Best Practices

### 1. Session Management

- Use consistent session IDs across related agents
- Store session context in memory
- Clean up after post-task hook

### 2. Operation Logging

- Provide descriptive operation messages
- Include relevant metadata
- Use consistent naming conventions

### 3. Conflict Resolution

- Monitor for conflicts regularly
- Use conflict hooks for learning
- Implement automatic resolution strategies

### 4. Performance

- Batch operations when possible
- Use async/await for parallelism
- Cache frequently accessed data

### 5. AgentDB Integration

- Enable sync for learning-critical operations
- Set appropriate reward scores
- Include critique for failed attempts

## Troubleshooting

### Issue: Hooks not executing

**Solution:**
```bash
# Check jj installation
which jj

# Verify hook configuration
echo $JJ_PATH
echo $JJ_REPO_PATH

# Test hook manually
jj-agent-hook pre-task --agent-id test --session-id test --description "test"
```

### Issue: AgentDB sync failing

**Solution:**
```bash
# Check environment
echo $AGENTDB_SYNC_ENABLED
echo $AGENTDB_SYNC_FILE

# Verify file permissions
touch $AGENTDB_SYNC_FILE
ls -la $AGENTDB_SYNC_FILE
```

### Issue: Operations not visible

**Solution:**
```bash
# Check jj log
jj log

# Verify operation log
jj op log

# Check session operations
jj-agent-hook query-history --session-id swarm-001
```

## Examples

See the `examples/integration/` directory for complete examples:

- `multi_agent_workflow.rs` - Multiple agents collaborating
- `agentdb_learning.ts` - Learning from operation history
- `concurrent_agents.ts` - Concurrent agent execution

## API Reference

### Rust

- `JJHooksIntegration` - Main integration struct
- `HookContext` - Context for hook execution
- `JJHookEvent` - Hook event data
- `AgentDBSync` - AgentDB synchronization

### TypeScript

- `JJHooksIntegration` - Integration class
- `HookContext` - Hook context interface
- `JJHookEvent` - Event interface
- `createHooksIntegration()` - Factory function
- `withHooks()` - Lifecycle helper

## Performance Metrics

| Operation | Time (avg) | Memory |
|-----------|------------|--------|
| Pre-task hook | 5ms | 1KB |
| Post-edit hook | 10ms | 2KB |
| Post-task hook | 15ms | 5KB |
| AgentDB sync | 20ms | 3KB |

## Future Enhancements

- [ ] Real-time MCP integration for AgentDB
- [ ] Automatic conflict resolution strategies
- [ ] Visual operation timeline
- [ ] Cross-repository coordination
- [ ] WASM-based browser support

## Support

- Issues: https://github.com/ruvnet/agentic-flow/issues
- Documentation: https://docs.agentic-flow.dev
- Examples: `/examples/integration/`
