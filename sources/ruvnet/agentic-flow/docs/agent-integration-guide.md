# Agent Integration Guide: Self-Learning Swarm Optimization

**Version:** 2.0.0
**Date:** 2025-11-02
**Purpose:** Guide for integrating self-learning swarm optimization into agentic-flow agents

---

## Overview

This guide shows how to add self-learning swarm optimization capabilities to existing agents, enabling them to:
- Automatically select optimal swarm topologies
- Learn from execution patterns over time
- Receive AI-recommended configurations
- Track and improve performance

---

## Top 10 Priority Agents for Integration

Based on usage patterns and benefit potential:

1. **coder** - Most frequently used, benefits from parallel code generation
2. **researcher** - Multi-domain research ideal for mesh/hierarchical topologies
3. **reviewer** - Large-scale code reviews benefit from parallel execution
4. **tester** - Parallel test generation and execution
5. **task-orchestrator** - Natural fit for swarm coordination
6. **system-architect** - Multi-domain architecture analysis
7. **backend-dev** - Microservice development with parallel agents
8. **code-review-swarm** - Already swarm-based, add self-learning
9. **github-modes** - Multi-repo operations benefit from parallelization
10. **swarm-memory-manager** - Distributed memory coordination

---

## Integration Template

### Option 1: Agent Markdown Frontmatter (Recommended)

Add YAML frontmatter to agent definition files:

```yaml
---
name: agent-name
description: Agent description
version: 2.0.0
concurrency: true
batch_size: 5
self_learning: true
adaptive_topology: true
reasoningbank_enabled: true
max_agents: 10
---

# Agent Name

## Concurrent Execution Capabilities ✨ NEW

This agent supports parallel execution via CLI subprocesses and ReasoningBank coordination.

See `/agentic-flow/src/prompts/parallel-execution-guide.md` for detailed patterns.

### Automatic Optimization

```typescript
import { autoSelectSwarmConfig } from '../hooks/swarm-learning-optimizer';
import { reasoningBank } from '../reasoningbank';

// Get AI-recommended configuration
const config = await autoSelectSwarmConfig(
  reasoningBank,
  'Your task description',
  {
    taskComplexity: 'high',  // low, medium, high, critical
    estimatedAgentCount: 10
  }
);

console.log(`Recommended: ${config.recommendedTopology}`);
console.log(`Expected speedup: ${config.expectedSpeedup}x`);
console.log(`Confidence: ${config.confidence}`);
```

### CLI Subprocess Spawning

```bash
# Spawn parallel subagents
npx agentic-flow --agent agent-name --task "subtask-1" --output reasoningbank:swarm/task-id/agent-1
npx agentic-flow --agent agent-name --task "subtask-2" --output reasoningbank:swarm/task-id/agent-2
```

### Memory Coordination

```typescript
// Subagent stores results
await reasoningBank.storePattern({
  sessionId: 'swarm/task-123/agent-1',
  task: 'Subtask description',
  output: findings,
  reward: 0.95,
  success: true
});

// Parent retrieves all results
const allResults = await reasoningBank.searchPatterns('swarm/task-123', { k: 10 });

// Synthesize final report
const report = synthesizeResults(allResults);
```

### Store Execution Patterns for Learning

```typescript
import { SwarmLearningOptimizer } from '../hooks/swarm-learning-optimizer';

const optimizer = new SwarmLearningOptimizer(reasoningBank);

// After swarm execution
await optimizer.storeExecutionPattern(
  'Task description',
  {
    topology: 'hierarchical',
    agentCount: 8,
    batchSize: 4,
    totalTimeMs: executionTime,
    successRate: 95.0,
    speedup: 3.5,
    taskComplexity: 'high',
    operations: 8
  },
  true  // success
);
```

## Original Agent Instructions

[Rest of agent definition...]
```

### Option 2: Agent System Prompt Enhancement

For agents defined in TypeScript/JavaScript:

```typescript
export const agentSystemPrompt = `
You are an AI agent specialized in [domain].

## CONCURRENT EXECUTION CAPABILITIES

You can spawn parallel subagents via CLI:

\`\`\`typescript
import { autoSelectSwarmConfig } from './hooks/swarm-learning-optimizer';
import { reasoningBank } from './reasoningbank';

// Get optimized configuration
const config = await autoSelectSwarmConfig(reasoningBank, taskDescription, {
  taskComplexity: 'high',
  estimatedAgentCount: 10
});

// Spawn parallel subagents
const results = await Promise.all([
  exec('npx agentic-flow --agent worker --task "task-1"'),
  exec('npx agentic-flow --agent worker --task "task-2"'),
  exec('npx agentic-flow --agent worker --task "task-3"')
]);
\`\`\`

See /agentic-flow/src/prompts/parallel-execution-guide.md for complete patterns.

## CORE CAPABILITIES

[Original agent capabilities...]
`;
```

---

## Step-by-Step Integration Process

### Step 1: Analyze Agent Usage Pattern

Determine if agent benefits from parallelization:

**✅ Good Candidates:**
- Tasks can be decomposed into independent subtasks
- High volume operations (1000+ files, multiple domains)
- Time-consuming operations that can run concurrently
- Multi-domain analysis or research

**❌ Poor Candidates:**
- Purely sequential tasks with dependencies
- Single-operation tasks (already fast)
- Tasks requiring constant user interaction

### Step 2: Add YAML Frontmatter

```yaml
---
name: coder
version: 2.0.0
concurrency: true              # Enable parallel execution
batch_size: 5                  # Default batch size
self_learning: true            # Enable auto-optimization
adaptive_topology: true        # Use AI-recommended topology
reasoningbank_enabled: true    # Enable cross-process memory
max_agents: 10                 # Maximum concurrent agents
---
```

### Step 3: Add Documentation Section

Insert after the frontmatter:

```markdown
## 🚀 Concurrent Execution Capabilities

This agent supports parallel execution via CLI subprocesses and self-learning optimization.

### Key Features:
- **Automatic Topology Selection**: AI recommends optimal swarm configuration
- **Pattern Learning**: Learns from past executions (0.6 → 0.95 confidence)
- **Expected Speedup**: 3.5-5.0x with hierarchical topology
- **ReasoningBank Coordination**: Cross-process memory sharing

### Usage Example:

See `/agentic-flow/src/prompts/parallel-execution-guide.md` for complete patterns.
```

### Step 4: Test Integration

```bash
# Test with basic task
npx agentic-flow --agent coder --task "Implement simple REST API"

# Test with parallel-friendly task
npx agentic-flow --agent coder --task "Refactor 50 modules in /src directory"
```

### Step 5: Store Execution Pattern

After running the agent, the system should automatically store patterns:

```typescript
// This should be integrated into the agent execution flow
await optimizer.storeExecutionPattern(taskDescription, metrics, success);
```

---

## Example: Integrating "coder" Agent

### Before (v1.0):

```yaml
---
name: coder
description: Implementation specialist for writing clean, efficient code
---

# Coder Agent

You are an expert software engineer focused on writing high-quality code.

[Instructions...]
```

### After (v2.0):

```yaml
---
name: coder
description: Implementation specialist for writing clean, efficient code
version: 2.0.0
concurrency: true
batch_size: 5
self_learning: true
adaptive_topology: true
reasoningbank_enabled: true
max_agents: 10
---

# Coder Agent

## 🚀 Concurrent Execution Capabilities

This agent supports parallel execution for large-scale code generation tasks.

### When to Use Parallel Execution:
- Refactoring multiple modules (10+)
- Implementing multiple microservices
- Generating test suites for large codebases
- Multi-file code transformations

### Auto-Optimization Example:

\`\`\`typescript
import { autoSelectSwarmConfig } from '../hooks/swarm-learning-optimizer';

// Get AI-recommended configuration for your task
const config = await autoSelectSwarmConfig(
  reasoningBank,
  'Refactor 50 utility modules to TypeScript',
  {
    taskComplexity: 'high',
    estimatedAgentCount: 10
  }
);

// Expected output:
// Recommended: hierarchical
// Expected speedup: 3.8x
// Confidence: 0.87
\`\`\`

### Performance Expectations:
- **Single module**: Use standard execution
- **10-50 modules**: 3-4x speedup with hierarchical topology
- **50+ modules**: 4-5x speedup with hierarchical + QUIC

See `/agentic-flow/src/prompts/parallel-execution-guide.md` for detailed patterns.

---

## Core Capabilities

You are an expert software engineer focused on writing high-quality code.

[Original instructions...]
```

---

## Testing Integrated Agents

### Validation Checklist

For each integrated agent:

- [ ] YAML frontmatter added with correct flags
- [ ] Documentation section added
- [ ] Examples provided for parallel execution
- [ ] Agent tested with small task (verify no regression)
- [ ] Agent tested with parallel-friendly task
- [ ] Execution pattern stored in ReasoningBank
- [ ] Performance improvement validated

### Test Commands

```bash
# 1. Test basic functionality (no regression)
npx agentic-flow --agent <name> --task "Simple task"

# 2. Test parallel execution
npx agentic-flow --agent <name> --task "Complex parallel-friendly task"

# 3. Check ReasoningBank patterns
npx agentic-flow reasoningbank status

# 4. View optimization statistics
# (Requires SwarmLearningOptimizer integration in agent execution flow)
```

---

## Rollout Strategy

### Phase 3A: Core Agents (Week 1)
- ✅ coder
- ✅ researcher
- ✅ reviewer
- ✅ tester

### Phase 3B: Specialized Agents (Week 2)
- ✅ task-orchestrator
- ✅ system-architect
- ✅ backend-dev

### Phase 3C: Swarm Agents (Week 3)
- ✅ code-review-swarm
- ✅ github-modes
- ✅ swarm-memory-manager

### Success Metrics

Track for each integrated agent:
- **Execution Time Improvement**: Target 3-5x speedup
- **Success Rate**: Maintain ≥90%
- **Learning Progress**: Confidence 0.6 → 0.85+ over 50 executions
- **User Satisfaction**: No regression complaints

---

## Troubleshooting

### Issue: Parallel execution not faster

**Diagnosis:**
- Task too small (overhead > benefit)
- Dependencies prevent parallelization
- Batch size too small

**Solution:**
- Increase minimum task size for parallel execution
- Adjust batch size in YAML frontmatter
- Use sequential execution for small tasks

### Issue: Low confidence recommendations

**Diagnosis:**
- Insufficient execution patterns (< 10)
- High variability in task types
- New task domain

**Solution:**
- Use default recommendations until 10+ patterns learned
- Check alternative recommendations
- Manually specify topology for critical tasks

### Issue: Execution patterns not storing

**Diagnosis:**
- ReasoningBank not initialized
- Missing integration in agent execution flow

**Solution:**
```bash
# Initialize ReasoningBank
npx agentic-flow reasoningbank init

# Verify status
npx agentic-flow reasoningbank status
```

---

## Best Practices

### 1. Start Small
- Integrate one agent at a time
- Test thoroughly before moving to next
- Monitor performance metrics

### 2. Document Changes
- Update agent version to 2.0.0
- Add clear examples
- Document expected speedup

### 3. Monitor Learning
- Track confidence evolution
- Review learned patterns
- Adjust recommendations based on data

### 4. Provide Fallbacks
- Keep sequential execution as option
- Use default recommendations when confidence low
- Allow manual topology override

---

## Next Steps

1. **Select First Agent**: Start with `coder` (highest usage)
2. **Follow Integration Template**: Use YAML frontmatter approach
3. **Test Thoroughly**: Validate no regressions
4. **Monitor Performance**: Track speedup and learning
5. **Iterate**: Apply learnings to next agent

---

## Support & Resources

- **Parallel Execution Guide**: `/agentic-flow/src/prompts/parallel-execution-guide.md`
- **Optimization Report**: `/docs/swarm-optimization-report.md`
- **Implementation Report**: `/docs/parallel-execution-implementation.md`
- **GitHub Issue**: [#43](https://github.com/ruvnet/agentic-flow/issues/43)

---

**Status:** 📋 Integration guide ready for Phase 3 rollout
**Recommendation:** Start with `coder` agent integration and validate before scaling to remaining 9 agents
