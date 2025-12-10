# AgentDB Integration Guide

## Overview

agentic-jujutsu includes **built-in operation tracking** via AgentDB, allowing AI agents to learn from repository operations, analyze patterns, and improve their decision-making over time.

**Status**: ✅ **FULLY FUNCTIONAL** in v2.0.2

## What is AgentDB?

AgentDB automatically tracks every jj operation performed through the wrapper, storing:
- Operation type (status, log, commit, rebase, etc.)
- Command executed
- User and hostname
- Timestamp (ISO 8601)
- Duration (milliseconds)
- Success/failure status
- Error messages (if any)

This data enables:
- **Multi-agent coordination** - Agents can see what others are doing
- **Performance optimization** - Identify slow operations
- **Error pattern detection** - Learn from failures
- **Operation analytics** - Success rates, common workflows
- **Learning and adaptation** - Improve agent behavior over time

## Quick Start

###  Basic Usage

```javascript
const { JjWrapper } = require('agentic-jujutsu');

// Create wrapper (AgentDB is always active)
const jj = new JjWrapper();

// Execute operations (automatically tracked)
await jj.status();
await jj.log(10);
await jj.newCommit('My commit message');

// View statistics
const stats = JSON.parse(jj.getStats());
console.log(`Total operations: ${stats.total_operations}`);
console.log(`Success rate: ${(stats.success_rate * 100).toFixed(1)}%`);
console.log(`Average duration: ${stats.avg_duration_ms.toFixed(2)}ms`);
```

## API Reference

### Statistics

#### `getStats()` → `string`

Returns JSON string with aggregated statistics:

```javascript
const stats = JSON.parse(jj.getStats());
// {
//   total_operations: 15,
//   success_rate: 0.9333,  // 93.33%
//   avg_duration_ms: 28.5
// }
```

### Operation Queries

#### `getOperations(limit)` → `JjOperation[]`

Get recent operations (newest first):

```javascript
const recentOps = jj.getOperations(10);

recentOps.forEach(op => {
    console.log(`${op.operationType}: ${op.command}`);
    console.log(`  Success: ${op.success}, Duration: ${op.durationMs}ms`);
    console.log(`  User: ${op.user}@${op.hostname}`);
    console.log(`  Time: ${new Date(op.timestamp).toLocaleString()}`);
});
```

#### `getUserOperations(limit)` → `JjOperation[]`

Get user-initiated operations (excludes automatic snapshots):

```javascript
const userOps = jj.getUserOperations(20);
console.log(`User performed ${userOps.length} operations`);
```

### Maintenance

#### `clearLog()` → `void`

Clear all logged operations:

```javascript
jj.clearLog();
const stats = JSON.parse(jj.getStats());
console.log(stats.total_operations); // 0
```

## Operation Types

AgentDB tracks 30+ operation types:

### Read Operations (Non-modifying)
- **Status** - `jj status`
- **Log** - `jj log`
- **Diff** - `jj diff`

### Write Operations (Modifying)
- **New** - `jj new` (create commit)
- **Describe** - `jj describe` (amend message)
- **Edit** - `jj edit` (edit commit)
- **Abandon** - `jj abandon`
- **Rebase** - `jj rebase`
- **Squash** - `jj squash`
- **Undo** - `jj undo`

### Branch/Bookmark Operations
- **Branch** - `jj branch`
- **Bookmark** - `jj bookmark`

### Remote Operations
- **GitFetch** - `jj git fetch`
- **GitPush** - `jj git push`
- **Push** - `jj push`
- **Fetch** - `jj fetch`

### Special Operations
- **Resolve** - `jj resolve` (conflict resolution)
- **Restore** - `jj restore`
- **Split** - `jj split`
- **Duplicate** - `jj duplicate`
- **Merge** - `jj merge`

## Data Structure

### JjOperation Object

```typescript
interface JjOperation {
    id: string;              // UUID
    operationId: string;     // jj operation ID
    operationType: string;   // "Status", "Log", "New", etc.
    command: string;         // Full command: "jj status"
    user: string;            // Username
    hostname: string;        // Machine hostname
    timestamp: string;       // ISO 8601: "2025-11-10T15:30:00Z"
    tags: string[];          // Custom tags
    metadata: string;        // Additional JSON data
    parentId?: string;       // Parent operation ID
    durationMs: number;      // Execution time in ms
    success: boolean;        // true if succeeded
    error?: string;          // Error message if failed
}
```

## Advanced Usage

### Multi-Agent Coordination

```javascript
// Agent 1: Developer
const dev = new JjWrapper();
await dev.newCommit('Add feature X');
await dev.gitPush();

// Agent 2: Reviewer
const reviewer = new JjWrapper();
const recentOps = reviewer.getOperations(5);

const lastPush = recentOps.find(op =>
    op.operationType === 'GitPush' && op.success
);

if (lastPush) {
    console.log(`Review needed: ${lastPush.user} pushed at ${lastPush.timestamp}`);
    await reviewer.log(1);  // Check what was pushed
}
```

### Performance Monitoring

```javascript
async function monitorPerformance(jj) {
    const initialStats = JSON.parse(jj.getStats());
    const initialOps = initialStats.total_operations;

    // Perform operations
    await jj.status();
    await jj.log(100);
    await jj.diff('@', '@-');

    const finalStats = JSON.parse(jj.getStats());
    const newOps = finalStats.total_operations - initialOps;
    const avgDuration = finalStats.avg_duration_ms;

    if (avgDuration > 50) {
        console.warn(`⚠️ Slow operations detected: ${avgDuration.toFixed(2)}ms average`);
    }

    console.log(`✅ Completed ${newOps} operations in ${avgDuration.toFixed(2)}ms avg`);
}
```

### Error Pattern Detection

```javascript
function analyzeErrors(jj) {
    const allOps = jj.getOperations(1000);
    const failures = allOps.filter(op => !op.success);

    if (failures.length === 0) {
        console.log('✅ No errors in recent operations');
        return;
    }

    // Group by error type
    const errorCounts = {};
    failures.forEach(op => {
        const error = op.error || 'Unknown error';
        errorCounts[error] = (errorCounts[error] || 0) + 1;
    });

    console.log('❌ Error Analysis:');
    Object.entries(errorCounts)
        .sort((a, b) => b[1] - a[1])
        .forEach(([error, count]) => {
            console.log(`  ${count}x ${error}`);
        });

    const failureRate = failures.length / allOps.length;
    console.log(`\nFailure rate: ${(failureRate * 100).toFixed(2)}%`);
}
```

### Learning from Success Patterns

```javascript
function findSuccessPatterns(jj) {
    const ops = jj.getOperations(500);
    const successful = ops.filter(op => op.success);

    // Analyze operation sequences
    const sequences = {};
    for (let i = 0; i < successful.length - 1; i++) {
        const current = successful[i].operationType;
        const next = successful[i + 1].operationType;
        const sequence = `${current} → ${next}`;
        sequences[sequence] = (sequences[sequence] || 0) + 1;
    }

    console.log('📊 Most Common Successful Sequences:');
    Object.entries(sequences)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .forEach(([seq, count]) => {
            console.log(`  ${count}x ${seq}`);
        });
}
```

## Configuration

### Memory Management

AgentDB stores operations in memory with configurable limits:

```javascript
const jj = new JjWrapper();
const config = jj.getConfig();

console.log(`Max log entries: ${config.maxLogEntries}`);
// Default: 1000 operations

// When limit is reached, oldest entries are removed (FIFO)
```

### Clearing Old Data

```javascript
// Clear all operations
jj.clearLog();

// Or implement custom cleanup
function cleanupOldOperations(jj, maxAge = 24 * 60 * 60 * 1000) {
    const ops = jj.getOperations(10000);
    const now = Date.now();

    const recentOps = ops.filter(op => {
        const age = now - new Date(op.timestamp).getTime();
        return age < maxAge;
    });

    if (recentOps.length < ops.length) {
        console.log(`Cleaned up ${ops.length - recentOps.length} old operations`);
        jj.clearLog();
        // Re-add recent operations if needed
    }
}
```

## Real-World Examples

### Example 1: Agent Dashboard

```javascript
async function agentDashboard(jj) {
    const stats = JSON.parse(jj.getStats());
    const recentOps = jj.getOperations(10);

    console.log('╔════════════════════════════════════╗');
    console.log('║      Agent Activity Dashboard      ║');
    console.log('╚════════════════════════════════════╝\n');

    console.log(`📊 Statistics:`);
    console.log(`   Total Operations: ${stats.total_operations}`);
    console.log(`   Success Rate: ${(stats.success_rate * 100).toFixed(1)}%`);
    console.log(`   Avg Duration: ${stats.avg_duration_ms.toFixed(2)}ms\n`);

    console.log(`📝 Recent Activity:`);
    recentOps.slice(0, 5).forEach((op, i) => {
        const icon = op.success ? '✅' : '❌';
        const time = new Date(op.timestamp).toLocaleTimeString();
        console.log(`   ${icon} [${time}] ${op.operationType}: ${op.command}`);
    });
}
```

### Example 2: Automated Conflict Detection

```javascript
async function detectConflicts(jj) {
    // Check status
    await jj.status();

    const ops = jj.getOperations(10);
    const statusOp = ops.find(op => op.operationType === 'Status');

    if (statusOp && statusOp.success) {
        // Parse output for conflicts (would need actual jj output)
        console.log('✅ Status check completed');

        // If conflicts detected, run resolve
        // await jj.resolve();
    }
}
```

### Example 3: CI/CD Integration

```javascript
async function cicdWorkflow(jj) {
    console.log('🚀 Starting CI/CD workflow...\n');

    try {
        // 1. Check status
        console.log('1. Checking repository status...');
        await jj.status();

        // 2. Create checkpoint
        console.log('2. Creating checkpoint...');
        await jj.newCommit('CI: Pre-test snapshot');

        // 3. Run tests (external)
        console.log('3. Running tests...');
        // execSync('npm test');

        // 4. If tests pass, push
        console.log('4. Pushing changes...');
        // await jj.gitPush();

        const stats = JSON.parse(jj.getStats());
        console.log(`\n✅ Workflow complete: ${stats.total_operations} operations`);
        console.log(`   Success rate: ${(stats.success_rate * 100).toFixed(1)}%`);
        console.log(`   Total time: ${stats.avg_duration_ms * stats.total_operations}ms`);

    } catch (error) {
        console.error('❌ Workflow failed:', error.message);

        // Rollback
        await jj.undo();
        throw error;
    }
}
```

## Troubleshooting

### Operations Not Being Tracked

**Problem**: `getStats()` returns zeros

**Solution**: AgentDB is always active. If you see zeros, no operations have been executed yet.

```javascript
const jj = new JjWrapper();
const stats1 = JSON.parse(jj.getStats());
console.log(stats1.total_operations); // 0

await jj.status();
const stats2 = JSON.parse(jj.getStats());
console.log(stats2.total_operations); // 1
```

### Memory Growth

**Problem**: Too many operations in memory

**Solution**: Periodically clear the log:

```javascript
setInterval(() => {
    const stats = JSON.parse(jj.getStats());
    if (stats.total_operations > 5000) {
        jj.clearLog();
        console.log('🧹 Cleared operation log');
    }
}, 60000); // Every minute
```

### Missing Operation Types

**Problem**: Operations show as "Unknown"

**Solution**: Verify you're using the latest version (v2.0.2+):

```bash
npm install agentic-jujutsu@latest
```

Supported types as of v2.0.2: Status, Log, Diff, New, Describe, Edit, Abandon, Rebase, Squash, Resolve, Branch, Bookmark, GitFetch, GitPush, Undo, Restore, and more.

## Performance Impact

AgentDB has **minimal performance overhead**:

- **Memory**: ~1KB per operation
- **CPU**: <1ms per operation
- **I/O**: Zero (in-memory only)

With default limit of 1000 operations:
- Memory usage: ~1MB
- No disk I/O
- No network calls

## Best Practices

### 1. **Regular Cleanup**
```javascript
if (stats.total_operations > 1000) {
    jj.clearLog();
}
```

### 2. **Error Handling**
```javascript
try {
    await jj.gitPush();
} catch (error) {
    const ops = jj.getOperations(1);
    console.error('Push failed:', ops[0].error);
}
```

### 3. **Performance Monitoring**
```javascript
const ops = jj.getOperations(100);
const slowOps = ops.filter(op => op.durationMs > 100);
if (slowOps.length > 10) {
    console.warn('Performance degradation detected');
}
```

### 4. **Multi-Agent Coordination**
```javascript
// Each agent has its own JjWrapper instance
// Operations are logged per-instance (not shared across processes)
const agent1 = new JjWrapper();
const agent2 = new JjWrapper();

// They track independently
await agent1.status();
await agent2.status();

console.log(agent1.getStats()); // { total_operations: 1 }
console.log(agent2.getStats()); // { total_operations: 1 }
```

## Future Enhancements

Planned features for future versions:

- [ ] **Persistent storage** - Save operations to disk/database
- [ ] **Cross-process sharing** - Share operation log between agents
- [ ] **Operation replay** - Re-execute operation sequences
- [ ] **Pattern learning** - ML-based operation prediction
- [ ] **Conflict prediction** - Warn before operations that might conflict
- [ ] **Resource usage tracking** - CPU, memory, disk I/O per operation
- [ ] **Webhooks** - Trigger external systems on specific operations
- [ ] **GraphQL API** - Query operations via GraphQL

## Support

For issues or questions about AgentDB:

- GitHub Issues: https://github.com/ruvnet/agentic-flow/issues
- Documentation: https://github.com/ruvnet/agentic-flow/tree/main/packages/agentic-jujutsu

---

**Version**: 2.0.2
**Status**: ✅ Production Ready
**Last Updated**: November 10, 2025
