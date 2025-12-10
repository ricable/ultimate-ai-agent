# Research Swarm Architecture - v1.1.0 Proposal

## Problem Statement

**Current Issue**: Package named "research-swarm" but default `research` command uses single-agent execution.

**User Requirement**: "its called research swarm, it should use a swarm approach"

## Current Architecture (v1.0.1)

### Single-Agent Mode (Default)
```bash
research-swarm research researcher "task"
```
- Spawns ONE agent via `run-researcher-local.js`
- Executes task sequentially through 5 phases
- No parallel processing
- No multi-perspective analysis

### Swarm Mode (Optional)
```bash
research-swarm swarm "task1" "task2" "task3" --concurrent 3
```
- Spawns MULTIPLE agents for DIFFERENT tasks
- Parallel execution with configurable concurrency
- Triggers learning session after ≥2 successful completions

## Proposed Architecture (v1.1.0)

### Default Swarm Behavior

When user provides a single research task, automatically decompose into multiple swarm perspectives:

#### Swarm Decomposition Strategy

**For any research task, spawn 3-5 specialized agents:**

1. **Explorer Agent** (20% of time)
   - Broad survey and topic mapping
   - Identify key domains and sub-topics
   - Fast breadth-first search

2. **Depth Analyst** (30% of time)
   - Deep dive into core concepts
   - Technical details and mechanisms
   - Slow depth-first search

3. **Verification Agent** (20% of time)
   - Cross-reference findings
   - Fact-checking and citation verification
   - Source quality assessment

4. **Trend Analyst** (15% of time)
   - Temporal analysis (historical → current → future)
   - Pattern recognition across time
   - Emerging trends identification

5. **Synthesis Agent** (15% of time)
   - Combine findings from all agents
   - Resolve conflicts and contradictions
   - Generate unified report

### Implementation Phases

#### Phase 1: Parallel Swarm Research (v1.1.0)

**Change `research` command behavior:**

```javascript
// OLD (v1.0.1)
research-swarm research researcher "task"
  → Spawns 1 agent

// NEW (v1.1.0)
research-swarm research researcher "task"
  → Spawns 5 specialized agents in parallel
  → Each agent runs for depth/5 time
  → Synthesis agent combines results
```

**Architecture:**

```
User Task: "Analyze quantum computing trends"
    ↓
Swarm Decomposition
    ↓
┌─────────────────────────────────────────────────┐
│  Explorer (20%) → Broad survey                  │
│  Depth (30%)    → Technical deep dive          │
│  Verifier (20%) → Fact checking                │
│  Trend (15%)    → Temporal analysis            │
│  Synthesis (15%) → Unified report               │
└─────────────────────────────────────────────────┘
    ↓
Pattern Learning (ReasoningBank)
    ↓
Final Report (Markdown)
```

#### Phase 2: Configuration Options (v1.1.1)

**Add command-line options:**

```bash
# Use swarm by default (5 agents)
research-swarm research researcher "task"

# Override with single-agent mode
research-swarm research researcher "task" --single-agent

# Configure swarm size
research-swarm research researcher "task" --swarm-size 3

# Custom swarm composition
research-swarm research researcher "task" --swarm-agents "explorer,depth,verifier"
```

#### Phase 3: Adaptive Swarm (v1.2.0)

**Intelligent agent selection based on task complexity:**

- Simple tasks (depth 1-3): 3 agents (explorer, depth, synthesis)
- Medium tasks (depth 4-6): 5 agents (all)
- Complex tasks (depth 7-10): 7+ agents (add domain specialists)

**Example:**

```javascript
// Automatic swarm sizing
if (depth <= 3) {
  swarmAgents = ['explorer', 'depth', 'synthesis'];
} else if (depth <= 6) {
  swarmAgents = ['explorer', 'depth', 'verifier', 'trend', 'synthesis'];
} else {
  swarmAgents = ['explorer', 'depth', 'verifier', 'trend', 'domain-expert', 'critic', 'synthesis'];
}
```

## Technical Implementation

### File Changes Required

#### 1. `/bin/cli.js` (lines 27-73)

**OLD:**
```javascript
.action(async (agent, task, options) => {
  const child = spawn('node', [runnerPath, agent, task], { env, stdio: 'inherit' });
});
```

**NEW:**
```javascript
.option('--single-agent', 'Use single-agent mode (default: swarm)')
.option('--swarm-size <number>', 'Number of swarm agents', '5')
.action(async (agent, task, options) => {
  if (options.singleAgent) {
    // Legacy single-agent mode
    const child = spawn('node', [runnerPath, agent, task], { env, stdio: 'inherit' });
  } else {
    // NEW: Swarm decomposition
    const swarmTasks = decomposeTask(task, parseInt(options.swarmSize));
    await executeSwarm(swarmTasks, options);
  }
});
```

#### 2. `/lib/swarm-decomposition.js` (NEW)

```javascript
export function decomposeTask(task, swarmSize = 5) {
  return [
    {
      agent: 'explorer',
      task: `Broad survey: ${task}`,
      focus: 'broad',
      timePercent: 0.20
    },
    {
      agent: 'depth-analyst',
      task: `Deep analysis: ${task}`,
      focus: 'narrow',
      timePercent: 0.30
    },
    {
      agent: 'verifier',
      task: `Verify findings: ${task}`,
      focus: 'balanced',
      timePercent: 0.20
    },
    {
      agent: 'trend-analyst',
      task: `Temporal trends: ${task}`,
      focus: 'broad',
      timePercent: 0.15
    },
    {
      agent: 'synthesizer',
      task: `Synthesize all findings: ${task}`,
      focus: 'balanced',
      timePercent: 0.15,
      dependsOn: ['explorer', 'depth-analyst', 'verifier', 'trend-analyst']
    }
  ];
}
```

#### 3. `/lib/swarm-executor.js` (NEW)

```javascript
export async function executeSwarm(swarmTasks, options) {
  const { spawn } = await import('child_process');
  const maxConcurrent = 4; // Run first 4 in parallel, synthesis waits

  // Execute non-synthesis agents in parallel
  const primaryAgents = swarmTasks.filter(t => t.agent !== 'synthesizer');
  const synthesisAgent = swarmTasks.find(t => t.agent === 'synthesizer');

  const results = await Promise.all(
    primaryAgents.map(task => runSwarmAgent(task, options))
  );

  // Run synthesis agent after all others complete
  if (synthesisAgent) {
    synthesisAgent.inputReports = results.map(r => r.reportPath);
    await runSwarmAgent(synthesisAgent, options);
  }

  // Trigger learning session
  await runLearningSession();
}
```

### Migration Path

#### v1.0.1 → v1.1.0 (Backward Compatible)

**Default behavior changes, but single-agent mode still available:**

```bash
# v1.0.1 behavior (can still be accessed)
research-swarm research researcher "task" --single-agent

# v1.1.0 default (swarm mode)
research-swarm research researcher "task"
```

#### v1.1.0 → v1.2.0 (Adaptive)

**Automatic swarm sizing based on task complexity:**

```bash
# Automatically selects swarm size based on depth/time
research-swarm research researcher "task" --depth 8
  → Spawns 7 agents (complex task)

research-swarm research researcher "task" --depth 2
  → Spawns 3 agents (simple task)
```

## Benefits

### Performance
- **Parallel execution**: 3-5x faster for complex research
- **Resource utilization**: Better CPU/memory usage
- **Adaptive scaling**: Match resources to task complexity

### Quality
- **Multi-perspective analysis**: Reduces blind spots
- **Built-in verification**: Cross-checking between agents
- **Conflict resolution**: Synthesis agent identifies contradictions

### Learning
- **Pattern diversity**: Multiple approaches stored in ReasoningBank
- **Failure analysis**: Identify which perspectives work best
- **Continuous improvement**: Learn from swarm dynamics

## Risks & Mitigations

### Risk 1: Increased Complexity
**Mitigation**: Provide `--single-agent` fallback for simple tasks

### Risk 2: Higher API Costs
**Mitigation**:
- Adaptive swarm sizing (3 agents for simple tasks)
- Token budget per agent = total budget / swarm size

### Risk 3: Breaking Changes
**Mitigation**:
- v1.1.0 is backward compatible via `--single-agent`
- Gradual migration path
- Clear documentation

## Success Metrics

### v1.1.0 Goals
- ✅ Default `research` command uses swarm (5 agents)
- ✅ `--single-agent` option for backward compatibility
- ✅ Parallel execution with synthesis phase
- ✅ ReasoningBank stores all agent patterns

### v1.2.0 Goals
- ✅ Adaptive swarm sizing (3-7 agents)
- ✅ Task complexity detection
- ✅ Domain-specific agent spawning
- ✅ Swarm performance metrics

## Timeline

- **v1.0.1**: Current (single-agent default) ✅
- **v1.1.0**: Swarm-by-default (1-2 weeks)
- **v1.1.1**: Configuration options (1 week)
- **v1.2.0**: Adaptive swarm (2-3 weeks)

## Conclusion

Making swarm the default behavior aligns the package name with its functionality. The proposed architecture:

1. ✅ Matches user expectations ("research swarm" = swarm behavior)
2. ✅ Provides backward compatibility (`--single-agent`)
3. ✅ Improves research quality (multi-perspective)
4. ✅ Enables future enhancements (adaptive sizing)

**Recommendation**: Proceed with v1.1.0 implementation.
