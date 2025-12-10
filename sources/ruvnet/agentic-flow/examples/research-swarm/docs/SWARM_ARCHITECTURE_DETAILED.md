# Multi-Agent Swarm Workflow Architecture - Detailed Diagram

**Version**: 1.2.2
**Date**: November 4, 2025
**Document Type**: Technical Architecture Reference

---

## 🎯 Overview

This document provides a comprehensive visual breakdown of the research-swarm multi-agent architecture, including all components, data flows, decision points, and integration layers.

---

## 📊 Complete System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                          USER INPUT & ENTRY POINTS                              │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   CLI Tool   │  │  JavaScript  │  │ MCP Server   │  │ NPX Command  │      │
│  │ (Commander)  │  │   Package    │  │  (stdio/http)│  │   (Direct)   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │                 │               │
│         └─────────────────┴─────────────────┴─────────────────┘               │
│                                    ▼                                           │
└────────────────────────────────────────────────────────────────────────────────┘
                                     │
                  ┌──────────────────┴──────────────────┐
                  │                                     │
                  ▼                                     ▼
┌────────────────────────────────────────┐  ┌────────────────────────────────────┐
│    GOAL-ORIENTED PLANNING (v1.2.0)     │  │   DIRECT RESEARCH (v1.1.0)         │
│           GOALIE SDK                   │  │      Swarm-by-Default              │
└────────────────┬───────────────────────┘  └────────────────┬───────────────────┘
                 │                                           │
                 │  ┌─────────────────────────────────────┐ │
                 │  │   COMPONENT 1: TASK ANALYZER        │ │
                 │  ├─────────────────────────────────────┤ │
                 └─►│ • Parse user goal/task              │◄┘
                    │ • Estimate complexity (low/med/high)│
                    │ • Determine depth (1-10 scale)      │
                    │ • Calculate time budget             │
                    │ • Detect task type (research/code/  │
                    │   analysis/synthesis)               │
                    └────────────────┬────────────────────┘
                                     │
                    ┌────────────────▼────────────────────┐
                    │   DECISION POINT 1                  │
                    │   Is goal complex enough for GOAP?  │
                    └─────┬──────────────────────┬────────┘
                          │ YES                  │ NO
                          │ (complex goal)       │ (simple task)
                          ▼                      ▼
        ┌─────────────────────────────┐  ┌─────────────────────────┐
        │  COMPONENT 2: GOALIE GOAP   │  │  COMPONENT 3: DIRECT    │
        │  Goal Decomposition         │  │  Task Decomposition     │
        ├─────────────────────────────┤  ├─────────────────────────┤
        │ • Execute: npx goalie       │  │ • Parse task directly   │
        │ • Algorithm: GOAP planning  │  │ • No sub-goals          │
        │ • Output: Sub-goals (2-10)  │  │ • Single decomposition  │
        │ • Per sub-goal:             │  │ • Agent role assignment │
        │   - Description             │  └────────────┬────────────┘
        │   - Complexity (L/M/H/VH)   │               │
        │   - Dependencies            │               │
        │   - Success criteria        │               │
        │ • Fallback: Manual split    │               │
        └────────────┬────────────────┘               │
                     │                                │
                     └────────────┬───────────────────┘
                                  │
                    ┌─────────────▼──────────────────┐
                    │   COMPONENT 4: SWARM SIZING    │
                    │   Adaptive Agent Allocation    │
                    ├────────────────────────────────┤
                    │ Complexity → Agent Count:      │
                    │                                │
                    │ • Low (depth 1-3):    3 agents│
                    │   - Explorer (40%)             │
                    │   - Depth Analyst (40%)        │
                    │   - Synthesizer (20%)          │
                    │                                │
                    │ • Medium (depth 4-6): 5 agents│
                    │   - Explorer (20%)             │
                    │   - Depth Analyst (30%)        │
                    │   - Verifier (20%)             │
                    │   - Trend Analyst (15%)        │
                    │   - Synthesizer (15%)          │
                    │                                │
                    │ • High (depth 7-8):   7 agents│
                    │   + Domain Expert (10%)        │
                    │   + Critic (10%)               │
                    │                                │
                    │ • Very High (depth 9-10):     │
                    │   7 agents + recursive loops   │
                    └────────────┬───────────────────┘
                                 │
                ┌────────────────▼────────────────┐
                │   COMPONENT 5: AGENT SPAWNING  │
                │   Multi-Process Execution      │
                ├────────────────────────────────┤
                │ For each agent role:           │
                │                                │
                │ 1. Create job record (SQLite)  │
                │ 2. Assign:                     │
                │    - Unique job ID (UUID)      │
                │    - Agent role                │
                │    - Task slice                │
                │    - Time allocation (%)       │
                │    - Context window            │
                │ 3. Spawn Node process          │
                │ 4. Set up IPC channels         │
                │ 5. Initialize progress tracking│
                └────────────┬───────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
        ▼                                         ▼
┌────────────────────┐              ┌────────────────────────┐
│  COMPONENT 6A:     │              │  COMPONENT 6B:         │
│  Priority Queue    │              │  Execution Scheduler   │
├────────────────────┤              ├────────────────────────┤
│ Phase 1: RESEARCH  │◄────────────►│ • Max concurrent: 4    │
│ - Explorer         │              │ • Round-robin spawn    │
│ - Depth Analyst    │              │ • Resource monitoring  │
│ - Trend Analyst    │              │ • Backpressure control │
│                    │              │ • Error retry (3x)     │
│ Phase 2: VERIFY    │              └────────────────────────┘
│ - Verifier         │
│ - Domain Expert    │
│                    │
│ Phase 3: SYNTHESIZE│
│ - Critic           │
│ - Synthesizer      │
└─────────┬──────────┘
          │
          │  ┌──────────────────────────────────────────────────────────┐
          │  │         COMPONENT 7: PARALLEL AGENT EXECUTION            │
          └─►├──────────────────────────────────────────────────────────┤
             │                                                           │
             │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
             │  │  Agent 1    │  │  Agent 2    │  │  Agent 3    │     │
             │  │  Explorer   │  │  Depth      │  │  Verifier   │     │
             │  │             │  │  Analyst    │  │             │     │
             │  │  Process 1  │  │  Process 2  │  │  Process 3  │     │
             │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
             │         │                │                │             │
             │         └────────┬───────┴────────┬───────┘             │
             │                  │                │                     │
             │                  ▼                ▼                     │
             │         ┌──────────────────────────────┐               │
             │         │  COMPONENT 8: AI PROVIDER    │               │
             │         │  Multi-Provider Router       │               │
             │         ├──────────────────────────────┤               │
             │         │ Provider Selection:          │               │
             │         │                              │               │
             │         │ • Anthropic Claude           │               │
             │         │   - Sonnet 4.5 (default)     │               │
             │         │   - Best quality, deep       │               │
             │         │                              │               │
             │         │ • Google Gemini              │               │
             │         │   - Real-time grounding      │               │
             │         │   - Google Search integration│               │
             │         │   - Latest information       │               │
             │         │                              │               │
             │         │ • OpenRouter                 │               │
             │         │   - 200+ models              │               │
             │         │   - Cost optimization        │               │
             │         │   - Perplexity support       │               │
             │         │                              │               │
             │         │ • Custom MCP Servers         │               │
             │         │   - Brave Search             │               │
             │         │   - Custom tools             │               │
             │         └──────────┬───────────────────┘               │
             │                    │                                   │
             │                    ▼                                   │
             │         ┌──────────────────────────────┐               │
             │         │  COMPONENT 9: RESEARCH LOOP  │               │
             │         │  Per-Agent Execution         │               │
             │         ├──────────────────────────────┤               │
             │         │ For each agent:              │               │
             │         │                              │               │
             │         │ 1. Initialize context        │               │
             │         │ 2. Load task + role prompt   │               │
             │         │ 3. Execute research phases:  │               │
             │         │    a. Initial exploration    │               │
             │         │    b. Deep dive              │               │
             │         │    c. Verification           │               │
             │         │    d. Citation check         │               │
             │         │    e. Synthesis              │               │
             │         │                              │               │
             │         │ 4. Progress updates (5s)     │               │
             │         │ 5. Memory checkpoints        │               │
             │         │ 6. Error handling            │               │
             │         │ 7. Token tracking            │               │
             │         └──────────┬───────────────────┘               │
             │                    │                                   │
             └────────────────────┼───────────────────────────────────┘
                                  │
                    ┌─────────────▼──────────────────┐
                    │   COMPONENT 10: DATABASE LAYER │
                    │   Hybrid Storage Architecture  │
                    ├────────────────────────────────┤
                    │                                │
                    │  ┌──────────────────────────┐  │
                    │  │  AgentDB (SQLite + WAL)  │  │
                    │  ├──────────────────────────┤  │
                    │  │ • Local-first (fast)     │  │
                    │  │ • 3,848 ops/sec          │  │
                    │  │ • HNSW vector search     │  │
                    │  │ • ReasoningBank patterns │  │
                    │  │ • Job state tracking     │  │
                    │  │ • Progress updates       │  │
                    │  │                          │  │
                    │  │ Tables:                  │  │
                    │  │ - research_jobs          │  │
                    │  │ - reasoningbank_patterns │  │
                    │  │ - learning_episodes      │  │
                    │  │ - vector_embeddings      │  │
                    │  │ - memory_distillations   │  │
                    │  │ - pattern_associations   │  │
                    │  └────────┬─────────────────┘  │
                    │           │ (Optional Sync)    │
                    │           ▼                    │
                    │  ┌──────────────────────────┐  │
                    │  │  Supabase (PostgreSQL)   │  │
                    │  ├──────────────────────────┤  │
                    │  │ • Multi-tenant (RLS)     │  │
                    │  │ • Real-time subscriptions│  │
                    │  │ • pgvector support       │  │
                    │  │ • Persistent storage     │  │
                    │  │ • Batch sync (2s)        │  │
                    │  │ • Progress throttling    │  │
                    │  │                          │  │
                    │  │ Table:                   │  │
                    │  │ - permit_research_jobs   │  │
                    │  └──────────────────────────┘  │
                    └────────────┬───────────────────┘
                                 │
                    ┌────────────▼────────────────┐
                    │  COMPONENT 11: SELF-LEARNING│
                    │  ReasoningBank Integration  │
                    ├─────────────────────────────┤
                    │ After each agent completes: │
                    │                             │
                    │ 1. Extract trajectory:      │
                    │    - Task input             │
                    │    - Reasoning steps        │
                    │    - Output quality         │
                    │    - Confidence scores      │
                    │                             │
                    │ 2. Calculate reward:        │
                    │    reward = (quality * 0.4) │
                    │           + (speed * 0.2)   │
                    │           + (citations * 0.2)│
                    │           + (depth * 0.2)   │
                    │                             │
                    │ 3. Store pattern:           │
                    │    - Session ID             │
                    │    - Task type              │
                    │    - Success/failure        │
                    │    - Critique               │
                    │    - Improvements           │
                    │                             │
                    │ 4. Generate embeddings      │
                    │ 5. HNSW indexing            │
                    │ 6. Pattern associations     │
                    └────────────┬────────────────┘
                                 │
                    ┌────────────▼────────────────┐
                    │  COMPONENT 12: AGGREGATION  │
                    │  Swarm Result Synthesis     │
                    ├─────────────────────────────┤
                    │ Wait for all agents to      │
                    │ complete, then:             │
                    │                             │
                    │ 1. Collect outputs:         │
                    │    - Per-agent findings     │
                    │    - Confidence scores      │
                    │    - Citations              │
                    │    - Verification results   │
                    │                             │
                    │ 2. Detect conflicts:        │
                    │    - Compare findings       │
                    │    - Flag contradictions    │
                    │    - Weight by confidence   │
                    │                             │
                    │ 3. Merge insights:          │
                    │    - Deduplicate content    │
                    │    - Rank by importance     │
                    │    - Preserve sources       │
                    │                             │
                    │ 4. Cross-verify:            │
                    │    - Check citation overlap │
                    │    - Validate claims        │
                    │    - Calculate consensus    │
                    │                             │
                    │ 5. Quality scoring:         │
                    │    - Depth analysis         │
                    │    - Citation quality       │
                    │    - Novelty detection      │
                    │    - Completeness check     │
                    └────────────┬────────────────┘
                                 │
                    ┌────────────▼────────────────┐
                    │  COMPONENT 13: SYNTHESIS    │
                    │  Master Report Generation   │
                    ├─────────────────────────────┤
                    │ Final report structure:     │
                    │                             │
                    │ # Executive Summary         │
                    │ - Key findings (3-5)        │
                    │ - Confidence: X.XX/1.0      │
                    │ - Sources: N verified       │
                    │                             │
                    │ # Methodology               │
                    │ - Swarm size: N agents      │
                    │ - Execution time: Xm Ys     │
                    │ - Provider(s): [list]       │
                    │                             │
                    │ # Detailed Findings         │
                    │ Per agent contributions:    │
                    │ - Explorer: [findings]      │
                    │ - Depth Analyst: [analysis] │
                    │ - Verifier: [validation]    │
                    │ - etc.                      │
                    │                             │
                    │ # Cross-Agent Insights      │
                    │ - Consensus findings        │
                    │ - Divergent views           │
                    │ - Novel discoveries         │
                    │                             │
                    │ # Verification               │
                    │ - Citation count: N         │
                    │ - Verification rate: X%     │
                    │ - Confidence breakdown      │
                    │                             │
                    │ # Learning Patterns         │
                    │ - Successful strategies     │
                    │ - Challenges encountered    │
                    │ - Future improvements       │
                    │                             │
                    │ # Appendix                  │
                    │ - Full citations            │
                    │ - Agent metrics             │
                    │ - Raw outputs (optional)    │
                    └────────────┬────────────────┘
                                 │
                    ┌────────────▼────────────────┐
                    │  COMPONENT 14: OUTPUT       │
                    │  Multi-Format Export        │
                    ├─────────────────────────────┤
                    │ Export formats:             │
                    │                             │
                    │ • Markdown (default)        │
                    │   - GitHub-flavored         │
                    │   - Mermaid diagrams        │
                    │   - Syntax highlighting     │
                    │                             │
                    │ • JSON (structured)         │
                    │   - Machine-readable        │
                    │   - API integration         │
                    │   - Schema validation       │
                    │                             │
                    │ • HTML (web)                │
                    │   - Interactive             │
                    │   - Collapsible sections    │
                    │   - Search functionality    │
                    │                             │
                    │ Storage:                    │
                    │ - ./data/reports/           │
                    │ - Database (report_content) │
                    │ - Supabase (optional)       │
                    └────────────┬────────────────┘
                                 │
                    ┌────────────▼────────────────┐
                    │  COMPONENT 15: METRICS      │
                    │  Performance Tracking       │
                    ├─────────────────────────────┤
                    │ Tracked metrics:            │
                    │                             │
                    │ • Duration: Total time      │
                    │ • Tokens: Per agent + total │
                    │ • Cost: Provider breakdown  │
                    │ • Quality: Grounding score  │
                    │ • Citations: Count + verify │
                    │ • Errors: Count + types     │
                    │ • Retries: Per agent        │
                    │ • Memory: Peak usage        │
                    │ • Latency: P50/P95/P99      │
                    │                             │
                    │ Stored in:                  │
                    │ - research_jobs table       │
                    │ - Adapter metrics (Supabase)│
                    │ - Learning episodes         │
                    └─────────────────────────────┘
```

---

## 🔍 Component Breakdown

### **COMPONENT 1: Task Analyzer**

**Purpose**: Parse and classify incoming research requests

**Inputs**:
- User goal/task string
- Optional flags (depth, time, focus)

**Processing**:
1. **Complexity Estimation**:
   ```javascript
   function estimateComplexity(task) {
     const keywords = {
       high: ['comprehensive', 'analyze', 'evaluate', 'compare'],
       medium: ['research', 'investigate', 'explore', 'study'],
       low: ['list', 'identify', 'define', 'summarize']
     };

     // Check for complex patterns
     if (task.match(/\b(and|or|versus|compare|contrast)\b/gi)) {
       return 'high';
     }

     // Default based on keywords
     return detectKeywords(task, keywords);
   }
   ```

2. **Depth Calculation**:
   - Depth 1-3: Quick reference, definitions
   - Depth 4-6: Standard research, fact-finding
   - Depth 7-8: Deep analysis, comprehensive reports
   - Depth 9-10: PhD-level research, exhaustive

3. **Time Budget Allocation**:
   - Default: 120 minutes
   - Per depth level: ~15 minutes
   - Adjusted by swarm size

**Outputs**:
- Complexity level (low/medium/high/very-high)
- Recommended depth (1-10)
- Time allocation per phase
- Task type classification

**File**: `/lib/swarm-decomposition.js:estimateComplexity()`

---

### **COMPONENT 2: GOALIE GOAP (Goal-Oriented Action Planning)**

**Purpose**: Decompose complex goals into achievable sub-goals

**Algorithm**: GOAP (Goal-Oriented Action Planning)
- Uses preconditions and effects
- Finds optimal path to goal
- Handles dependencies between sub-goals

**Process**:
1. **Execute GOALIE**:
   ```bash
   npx goalie search "<goal>" --max-results 10
   ```

2. **Parse Output**:
   ```javascript
   {
     subGoals: [
       {
         description: "Research consensus mechanisms",
         complexity: "high",
         dependencies: [],
         successCriteria: "Understand PoW, PoS, PoA, BFT"
       },
       {
         description: "Analyze Layer 2 solutions",
         complexity: "medium",
         dependencies: ["consensus-mechanisms"],
         successCriteria: "Compare rollups, state channels"
       }
     ]
   }
   ```

3. **Fallback Strategy**:
   - If GOALIE unavailable: Manual decomposition
   - If GOALIE fails: Single-agent mode
   - If timeout: Use cached patterns

**Performance**:
- Overhead: 30-60 seconds
- Cost reduction: 15-30% (through better task allocation)
- Success rate: 85-90%

**File**: `/lib/goalie-integration.js:decomposeGoal()`

---

### **COMPONENT 3: Direct Task Decomposition**

**Purpose**: Handle simple tasks without GOAP overhead

**When Used**:
- Single, well-defined questions
- Low complexity (depth 1-3)
- User specifies `--single-agent`

**Processing**:
```javascript
function directDecompose(task, depth) {
  const roles = [];

  // Always include explorer for initial survey
  roles.push({
    role: 'explorer',
    focus: 'broad-survey',
    timeAllocation: 0.3
  });

  // Add depth analyst for depth > 3
  if (depth >= 4) {
    roles.push({
      role: 'depth-analyst',
      focus: 'technical-dive',
      timeAllocation: 0.4
    });
  }

  // Always synthesize
  roles.push({
    role: 'synthesizer',
    focus: 'unified-report',
    timeAllocation: 0.3
  });

  return roles;
}
```

**File**: `/lib/swarm-decomposition.js:decomposeTask()`

---

### **COMPONENT 4: Swarm Sizing (Adaptive Agent Allocation)**

**Purpose**: Determine optimal number of agents based on complexity

**Sizing Matrix**:

| Complexity | Depth | Agents | Roles |
|------------|-------|--------|-------|
| **Low** | 1-3 | 3 | Explorer, Depth Analyst, Synthesizer |
| **Medium** | 4-6 | 5 | + Verifier, Trend Analyst |
| **High** | 7-8 | 7 | + Domain Expert, Critic |
| **Very High** | 9-10 | 7 | + Recursive loops, multi-pass |

**Agent Role Descriptions**:

1. **Explorer** (20-40%):
   - Broad survey of topic
   - Identify key areas
   - Map knowledge landscape

2. **Depth Analyst** (30-40%):
   - Technical deep dive
   - Detailed analysis
   - Expert-level content

3. **Verifier** (20%):
   - Fact-checking
   - Citation validation
   - Cross-reference verification

4. **Trend Analyst** (15%):
   - Temporal patterns
   - Historical context
   - Future predictions

5. **Domain Expert** (10%):
   - Specialized knowledge
   - Technical accuracy
   - Industry insights

6. **Critic** (10%):
   - Challenge assumptions
   - Devil's advocate
   - Quality control

7. **Synthesizer** (15-20%):
   - Combine findings
   - Resolve conflicts
   - Generate final report

**Cost Optimization**:
```javascript
// Adaptive sizing reduces cost by 15-30%
const estimatedCost = {
  singleAgent: depth * $0.50,
  staticSwarm: depth * 5 * $0.50,
  adaptiveSwarm: depth * dynamicSize * $0.50
};

// Example: Depth 5, Medium complexity
// Single: 5 * $0.50 = $2.50
// Static (5 agents): 5 * 5 * $0.50 = $12.50
// Adaptive (3-4 agents): 5 * 3.5 * $0.50 = $8.75
// Savings: ~30%
```

**File**: `/lib/swarm-decomposition.js:decomposeTask()`

---

### **COMPONENT 5: Agent Spawning (Multi-Process Execution)**

**Purpose**: Launch parallel Node.js processes for each agent

**Process**:
1. **Create Job Record**:
   ```javascript
   const jobId = createJob({
     id: `job-${uuid()}`,
     agent: role.name,
     task: role.taskSlice,
     config: {
       depth,
       timeAllocation: role.timePercent,
       provider: 'anthropic',
       model: 'claude-sonnet-4-5'
     }
   });
   ```

2. **Spawn Process**:
   ```javascript
   const agent = spawn('node', [
     'run-researcher-local.js',
     role.name,
     role.task,
     '--depth', depth,
     '--time', timeAllocation
   ], {
     cwd: __dirname,
     env: { ...process.env, AGENT_ROLE: role.name }
   });
   ```

3. **Set Up IPC**:
   - stdout: Progress updates
   - stderr: Error reporting
   - IPC channel: Memory sharing

4. **Initialize Tracking**:
   - Progress: 0%
   - Status: 'pending'
   - Start time: Date.now()

**Parallel Execution**:
- Max concurrent: 4 agents
- Round-robin spawning
- Backpressure control
- Resource monitoring

**File**: `/lib/swarm-executor.js:executeSwarm()`

---

### **COMPONENT 6A: Priority Queue**

**Purpose**: Order agent execution by dependency and priority

**Queue Structure**:
```javascript
const queue = {
  phase1_research: [
    { priority: 1, role: 'explorer', deps: [] },
    { priority: 1, role: 'depth-analyst', deps: [] },
    { priority: 1, role: 'trend-analyst', deps: [] }
  ],
  phase2_verify: [
    { priority: 2, role: 'verifier', deps: ['explorer', 'depth-analyst'] },
    { priority: 2, role: 'domain-expert', deps: ['depth-analyst'] }
  ],
  phase3_synthesize: [
    { priority: 3, role: 'critic', deps: ['verifier'] },
    { priority: 3, role: 'synthesizer', deps: ['all'] }
  ]
};
```

**Scheduling**:
1. Phase 1 (Research): Can run in parallel
2. Phase 2 (Verify): Waits for Phase 1 completion
3. Phase 3 (Synthesize): Waits for Phase 2 completion

**File**: `/lib/swarm-executor.js:priorityQueue`

---

### **COMPONENT 6B: Execution Scheduler**

**Purpose**: Manage concurrent execution and resource limits

**Features**:
- **Max Concurrent**: 4 agents at once
- **Round-Robin**: Fair scheduling
- **Backpressure**: Pause if queue > 10
- **Retry Logic**: 3 attempts on failure
- **Timeout**: Kill after 2x time budget

**Code**:
```javascript
async function scheduleExecution(agents, maxConcurrent = 4) {
  const queue = [...agents];
  const running = new Set();
  const completed = [];

  while (queue.length > 0 || running.size > 0) {
    // Spawn up to maxConcurrent
    while (running.size < maxConcurrent && queue.length > 0) {
      const agent = queue.shift();
      const process = spawnAgent(agent);
      running.add(process);

      process.on('exit', () => {
        running.delete(process);
        completed.push(agent);
      });
    }

    // Wait for any to complete
    await Promise.race([...running].map(p => p.promise));
  }

  return completed;
}
```

**File**: `/lib/swarm-executor.js:executeSwarm()`

---

### **COMPONENT 7: Parallel Agent Execution**

**Purpose**: Run multiple research agents concurrently

**Architecture**:
```
Main Process (Orchestrator)
    ├─► Process 1 (Explorer)
    ├─► Process 2 (Depth Analyst)
    ├─► Process 3 (Verifier)
    └─► Process 4 (Trend Analyst)
```

**Inter-Process Communication**:
- **Progress**: Every 5 seconds via IPC
- **Memory**: Shared via SQLite database
- **Errors**: Reported via stderr
- **Results**: Written to database

**Resource Monitoring**:
```javascript
// Track resource usage per agent
const metrics = {
  cpu: process.cpuUsage(),
  memory: process.memoryUsage(),
  tokens: countTokens(output),
  latency: Date.now() - startTime
};
```

**File**: `/lib/swarm-executor.js`, `/run-researcher-local.js`

---

### **COMPONENT 8: AI Provider (Multi-Provider Router)**

**Purpose**: Route requests to appropriate AI provider

**Provider Selection Logic**:
```javascript
function selectProvider(task, config) {
  if (config.provider) return config.provider;

  // Auto-select based on task type
  if (task.includes('latest') || task.includes('recent')) {
    return 'gemini'; // Real-time grounding
  }

  if (task.includes('technical') || task.includes('code')) {
    return 'anthropic'; // Best for technical content
  }

  if (config.costOptimize) {
    return 'openrouter'; // Cost-effective
  }

  return 'anthropic'; // Default
}
```

**Provider Capabilities**:

| Provider | Strengths | Use Cases |
|----------|-----------|-----------|
| **Anthropic Claude** | Deep analysis, technical accuracy | Default, code review, research |
| **Google Gemini** | Real-time info, grounding | Latest news, current events |
| **OpenRouter** | Cost, variety (200+ models) | Budget-conscious, experimentation |
| **Custom MCP** | Specialized tools | Domain-specific research |

**File**: `/run-researcher-local.js:selectProvider()`

---

### **COMPONENT 9: Research Loop (Per-Agent Execution)**

**Purpose**: Execute research task with multi-phase approach

**Research Phases**:

1. **Initial Exploration** (15% time):
   ```
   - Broad topic survey
   - Identify key areas
   - Map knowledge landscape
   - Find reliable sources
   ```

2. **Deep Dive** (40% time):
   ```
   - Technical analysis
   - Detailed investigation
   - Expert-level content
   - Primary sources
   ```

3. **Verification** (20% time):
   ```
   - Fact-checking
   - Cross-reference claims
   - Validate citations
   - Check recency
   ```

4. **Citation Check** (15% time):
   ```
   - Verify URLs exist
   - Check source authority
   - Validate date ranges
   - Assess bias
   ```

5. **Synthesis** (10% time):
   ```
   - Compile findings
   - Resolve conflicts
   - Generate report
   - Add confidence scores
   ```

**Progress Updates**:
```javascript
// Update every 5 seconds
setInterval(() => {
  const progress = calculateProgress(phase, elapsed);
  updateProgress(jobId, progress, `Phase: ${phase.name}`);
}, 5000);
```

**File**: `/run-researcher-local.js:executeResearch()`

---

### **COMPONENT 10: Database Layer (Hybrid Storage)**

**Purpose**: Store job state, patterns, and results

**Storage Architecture**:

```
┌─────────────────────────┐
│ AgentDB (SQLite + WAL)  │
│ - Fast local storage    │
│ - 3,848 ops/sec         │
└────────┬────────────────┘
         │ Optional sync (2s batch)
         ▼
┌─────────────────────────┐
│ Supabase (PostgreSQL)   │
│ - Multi-tenant (RLS)    │
│ - Real-time updates     │
└─────────────────────────┘
```

**SQLite Tables** (6 tables):

1. **research_jobs**: Job metadata and state
   - Columns: id, agent, task, status, progress, report_content
   - Indexes: status, agent, created_at

2. **reasoningbank_patterns**: Learning patterns
   - Columns: id, session_id, task, reward, success, critique
   - Indexes: session_id, reward, success

3. **learning_episodes**: Performance tracking
   - Columns: id, pattern_id, verdict, judgment_score
   - Indexes: pattern_id, verdict, judgment_score

4. **vector_embeddings**: Semantic search
   - Columns: id, source_id, embedding_data, content_hash
   - Indexes: source_id, content_hash

5. **memory_distillations**: Compressed knowledge
   - Columns: id, key_insights, success_rate, confidence_score
   - Indexes: confidence_score, usage_count

6. **pattern_associations**: Relationship graph
   - Columns: id, pattern_id_a, pattern_id_b, similarity_score
   - Indexes: pattern_id_a, pattern_id_b, similarity_score

**Performance**:
- **SQLite**: 3,848 ops/sec (WAL mode)
- **Supabase Sync**: 2s batch interval
- **HNSW Search**: 150x faster than linear

**Files**: `/lib/db-utils.js`, `/schema/research-jobs.sql`

---

### **COMPONENT 11: Self-Learning (ReasoningBank)**

**Purpose**: Learn from research patterns for continuous improvement

**Learning Pipeline**:

1. **Trajectory Extraction**:
   ```javascript
   const trajectory = {
     input: task,
     reasoning: steps,
     output: report,
     quality: score,
     confidence: confidence
   };
   ```

2. **Reward Calculation**:
   ```javascript
   function calculateReward(result) {
     return (
       result.quality * 0.4 +      // Output quality
       result.speed * 0.2 +         // Time efficiency
       result.citations * 0.2 +     // Source quality
       result.depth * 0.2           // Analysis depth
     );
   }
   ```

3. **Pattern Storage**:
   ```javascript
   storePattern({
     sessionId,
     task,
     reward,
     success: reward > 0.7,
     critique: generateCritique(result),
     improvements: suggestImprovements(result)
   });
   ```

4. **Vector Embedding**:
   - Generate embeddings (1536 dimensions)
   - HNSW indexing (150x faster)
   - Similarity search (cosine distance)

5. **Pattern Association**:
   - Link similar patterns (similarity > 0.7)
   - Complementary patterns (different approaches)
   - Sequential patterns (task chains)

**Learning Metrics**:
- **Success Rate**: % of successful tasks
- **Improvement Rate**: Change in reward over time
- **Pattern Reuse**: How often patterns are retrieved
- **Confidence Growth**: Increase in confidence scores

**File**: `/lib/reasoningbank-integration.js`

---

### **COMPONENT 12: Aggregation (Swarm Result Synthesis)**

**Purpose**: Combine results from all agents into unified view

**Aggregation Process**:

1. **Collect Outputs**:
   ```javascript
   const results = await Promise.all(agents.map(a => a.result));
   ```

2. **Detect Conflicts**:
   ```javascript
   function detectConflicts(results) {
     const claims = extractClaims(results);
     const conflicts = [];

     for (const claim of claims) {
       const supporting = results.filter(r => supports(r, claim));
       const contradicting = results.filter(r => contradicts(r, claim));

       if (contradicting.length > 0) {
         conflicts.push({
           claim,
           support: supporting.length,
           contradict: contradicting.length,
           confidence: calculateConsensus(supporting, contradicting)
         });
       }
     }

     return conflicts;
   }
   ```

3. **Merge Insights**:
   ```javascript
   function mergeInsights(results) {
     const merged = {
       findings: [],
       citations: new Set(),
       confidence: {}
     };

     // Deduplicate by content similarity
     for (const result of results) {
       for (const finding of result.findings) {
         if (!isDuplicate(finding, merged.findings)) {
           merged.findings.push({
             ...finding,
             sources: [result.agent],
             confidence: result.confidence
           });
         }
       }
     }

     // Rank by importance
     merged.findings.sort((a, b) =>
       b.confidence * b.sources.length -
       a.confidence * a.sources.length
     );

     return merged;
   }
   ```

4. **Cross-Verify**:
   - Check citation overlap
   - Validate claims across agents
   - Calculate consensus percentage

5. **Quality Scoring**:
   ```javascript
   const qualityScore = {
     depth: analyzeDepth(merged),        // 0-1
     citations: validateCitations(merged), // 0-1
     novelty: detectNovelty(merged),     // 0-1
     completeness: checkCompleteness(merged) // 0-1
   };

   const overall = Object.values(qualityScore)
     .reduce((a, b) => a + b) / 4;
   ```

**Output**:
```javascript
{
  findings: [...],
  conflicts: [...],
  consensus: 0.85,
  qualityScore: 0.87,
  citations: [...],
  agentBreakdown: {...}
}
```

**File**: `/lib/swarm-executor.js:aggregateResults()`

---

### **COMPONENT 13: Synthesis (Master Report Generation)**

**Purpose**: Generate comprehensive final report

**Report Structure**:

```markdown
# Research Report: [Task Title]

## Executive Summary
[3-5 key findings with confidence scores]

**Overall Confidence**: 0.87/1.0
**Verified Sources**: 42
**Swarm Size**: 5 agents
**Execution Time**: 8m 42s

## Methodology
- **Approach**: Multi-agent swarm with GOAP planning
- **Agents**: Explorer, Depth Analyst, Verifier, Trend Analyst, Synthesizer
- **Provider**: Anthropic Claude Sonnet 4.5
- **Verification**: 3-layer cross-check with citation validation

## Detailed Findings

### 1. [Finding Title] (Confidence: 0.92)
[Detailed content from Explorer and Depth Analyst]

**Supporting Evidence**:
- [Citation 1]
- [Citation 2]

**Agent Consensus**: 4/5 agents agree

---

### 2. [Finding Title] (Confidence: 0.85)
[Content...]

## Cross-Agent Insights

### Areas of Consensus
- [Finding with 100% agent agreement]
- [Finding with 80%+ agreement]

### Divergent Views
- **Explorer vs Critic**: [Different perspectives on X]
- **Resolution**: [How conflict was resolved]

### Novel Discoveries
- [Findings that emerged from agent interaction]

## Verification Report

### Citation Quality
- **Total Citations**: 42
- **Verified URLs**: 40/42 (95%)
- **Primary Sources**: 28 (67%)
- **Recency**: Avg 2.3 years old

### Fact-Checking
- **Claims Verified**: 38/40 (95%)
- **Contradictions**: 2 (flagged)
- **Confidence**: High

## Learning Patterns

### Successful Strategies
- [What worked well in this research]

### Challenges Encountered
- [Issues faced and how resolved]

### Future Improvements
- [Suggestions for similar tasks]

## Performance Metrics
- **Total Duration**: 8m 42s
- **Tokens Used**: 124,350
- **Estimated Cost**: $2.15
- **Provider Breakdown**: Anthropic 100%

## Appendix

### A. Full Citations
[Complete list with URLs and access dates]

### B. Agent Breakdown
| Agent | Findings | Confidence | Tokens |
|-------|----------|------------|--------|
| Explorer | 12 | 0.82 | 28,000 |
| Depth Analyst | 18 | 0.91 | 45,000 |
| ... | ... | ... | ... |

### C. Raw Outputs (Optional)
[Individual agent reports]
```

**Generation Process**:
1. Template selection based on task type
2. Content population from aggregated results
3. Markdown formatting with syntax highlighting
4. Quality review pass
5. Export in requested format (MD/JSON/HTML)

**File**: `/run-researcher-local.js:generateReport()`

---

### **COMPONENT 14: Output (Multi-Format Export)**

**Purpose**: Export report in user-requested format

**Supported Formats**:

1. **Markdown** (default):
   ```markdown
   # Report Title
   ## Section
   - Bullet points
   - **Bold** and *italic*
   - [Links](url)
   - ```code blocks```
   ```

2. **JSON** (machine-readable):
   ```json
   {
     "title": "Report Title",
     "executive_summary": "...",
     "findings": [
       {
         "title": "...",
         "content": "...",
         "confidence": 0.92,
         "citations": [...]
       }
     ],
     "metadata": {
       "duration": 522,
       "tokens": 124350,
       "cost": 2.15
     }
   }
   ```

3. **HTML** (web-ready):
   ```html
   <!DOCTYPE html>
   <html>
     <head>
       <title>Research Report</title>
       <style>/* Styling */</style>
     </head>
     <body>
       <h1>Report Title</h1>
       <details>
         <summary>Executive Summary</summary>
         <p>Content...</p>
       </details>
     </body>
   </html>
   ```

**Storage Locations**:
1. **File System**: `./data/reports/report-{jobId}.{format}`
2. **Database**: `research_jobs.report_content` (text)
3. **Supabase** (optional): `permit_research_jobs.report_content`

**File**: `/lib/db-utils.js:markComplete()`

---

### **COMPONENT 15: Metrics (Performance Tracking)**

**Purpose**: Track and report performance metrics

**Tracked Metrics**:

```javascript
const metrics = {
  // Timing
  duration_seconds: 522,
  phase_durations: {
    planning: 45,
    research: 320,
    verification: 89,
    synthesis: 68
  },

  // Resources
  tokens_used: 124350,
  tokens_per_agent: {
    explorer: 28000,
    depth_analyst: 45000,
    verifier: 22000,
    trend_analyst: 18000,
    synthesizer: 11350
  },
  memory_mb: 1250,
  memory_peak: 1450,

  // Cost
  estimated_cost: 2.15,
  provider_breakdown: {
    anthropic: 2.15
  },
  cost_per_agent: {...},

  // Quality
  grounding_score: 0.87,
  citation_count: 42,
  verified_citations: 40,
  confidence_avg: 0.85,

  // Performance
  errors: 2,
  retries: 3,
  success_rate: 0.95,
  latency_p50: 450,
  latency_p95: 890,
  latency_p99: 1200
};
```

**Storage**:
- **research_jobs table**: Core metrics
- **learning_episodes**: Performance tracking over time
- **Supabase adapter**: Real-time metrics (if enabled)

**Reporting**:
```javascript
// Print metrics summary
console.log(`
╔═══════════════════════════════════════╗
║      RESEARCH METRICS                  ║
╚═══════════════════════════════════════╝
Duration:        8m 42s
Tokens:          124,350
Cost:            $2.15
Quality:         0.87/1.0
Citations:       42 (95% verified)
Success Rate:    95%
╚═══════════════════════════════════════╝
`);
```

**File**: `/lib/db-utils.js`, `/lib/permit-platform-adapter.js`

---

## 📈 Data Flow Diagram

```
┌─────────┐
│  USER   │
└────┬────┘
     │ 1. Submit task
     ▼
┌─────────────────┐
│  Task Analyzer  │
└────┬────────────┘
     │ 2. Complexity = High
     ▼
┌─────────────────┐
│  GOALIE GOAP    │◄─── GOALIE SDK (npx goalie)
└────┬────────────┘
     │ 3. Sub-goals: [A, B, C]
     ▼
┌─────────────────┐
│  Swarm Sizing   │
└────┬────────────┘
     │ 4. Allocate 7 agents
     ▼
┌─────────────────────────────────────┐
│  Agent Spawning                     │
│  ├─► Explorer      (Process 1)      │
│  ├─► Depth Analyst (Process 2)      │
│  ├─► Verifier      (Process 3)      │
│  ├─► Trend Analyst (Process 4)      │
│  ├─► Domain Expert (Process 5)      │
│  ├─► Critic        (Process 6)      │
│  └─► Synthesizer   (Process 7)      │
└────┬────────────────────────────────┘
     │ 5. Execute in parallel (max 4 concurrent)
     │
     ├──────────────┬──────────────┬──────────────┐
     ▼              ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Agent 1  │  │ Agent 2  │  │ Agent 3  │  │ Agent 4  │
│ Research │  │ Research │  │ Research │  │ Research │
│ Loop     │  │ Loop     │  │ Loop     │  │ Loop     │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │             │
     │             │             │             │
     └──────┬──────┴──────┬──────┴──────┬──────┘
            │             │             │
            ▼             ▼             ▼
       ┌────────────────────────────────┐
       │   AI Provider Router           │
       │   (Anthropic/Gemini/OpenRouter)│
       └────────────┬───────────────────┘
                    │ 6. Generate findings
                    │
       ┌────────────▼───────────────────┐
       │   AgentDB (SQLite)             │
       │   - Store results               │
       │   - Track progress              │
       │   - Learn patterns              │
       └────────────┬───────────────────┘
                    │ Optional: Sync (2s batch)
       ┌────────────▼───────────────────┐
       │   Supabase (PostgreSQL)        │
       │   - Multi-tenant storage        │
       │   - Real-time updates           │
       └────────────┬───────────────────┘
                    │
       ┌────────────▼───────────────────┐
       │   Result Aggregation           │
       │   - Merge findings              │
       │   - Detect conflicts            │
       │   - Cross-verify                │
       └────────────┬───────────────────┘
                    │ 7. Synthesize
       ┌────────────▼───────────────────┐
       │   Report Generation            │
       │   - Executive summary           │
       │   - Detailed findings           │
       │   - Verification report         │
       └────────────┬───────────────────┘
                    │ 8. Export
       ┌────────────▼───────────────────┐
       │   Multi-Format Output          │
       │   - Markdown (default)          │
       │   - JSON (API)                  │
       │   - HTML (web)                  │
       └────────────┬───────────────────┘
                    │
       ┌────────────▼───────────────────┐
       │   ReasoningBank Learning       │
       │   - Store patterns              │
       │   - Calculate rewards           │
       │   - Update knowledge graph      │
       └────────────────────────────────┘
                    │ 9. Return to user
                    ▼
               ┌─────────┐
               │  USER   │
               └─────────┘
```

---

## 🔄 Execution Flow Timeline

```
Time → 0s         30s        60s        90s        180s       300s       450s      522s
       │          │          │          │          │          │          │         │
       ├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼─────────┤
       │          │          │          │          │          │          │         │
USER   │►Task     │          │          │          │          │          │         │◄Report
       │          │          │          │          │          │          │         │
GOALIE │ ►Decomp  │          │          │          │          │          │         │
       │          │          │          │          │          │          │         │
Agent1 │          │►Research ├──────────├──────────►Complete  │          │         │
Agent2 │          │►Research ├──────────├──────────►Complete  │          │         │
Agent3 │          │          │►Research ├──────────├──────────►Complete  │         │
Agent4 │          │          │►Research ├──────────├──────────►Complete  │         │
Agent5 │          │          │          │►Verify   ├──────────►Complete  │         │
Agent6 │          │          │          │          │►Verify   ├──────────►Complete│
Agent7 │          │          │          │          │          │►Synthesize►Complete│
       │          │          │          │          │          │          │         │
DB     │►Init     ├──────────Progress updates (every 5s)──────────────────►Store   │
Learn  │          │          │          │          │          │          │►Pattern │
       │          │          │          │          │          │          │         │
       └──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴─────────┘

Legend:
►  = Start event
├  = Progress checkpoint
└  = Completion
```

---

## 🎯 Decision Tree

```
                      ┌────────────────┐
                      │  User Input    │
                      └────────┬───────┘
                               │
                  ┌────────────▼────────────┐
                  │ Complexity Estimation   │
                  └─┬─────────────────────┬─┘
                    │                     │
            ┌───────▼────────┐    ┌──────▼─────────┐
            │ Simple Task    │    │ Complex Goal   │
            │ (depth 1-3)    │    │ (depth 4-10)   │
            └───────┬────────┘    └──────┬─────────┘
                    │                     │
                    │              ┌──────▼─────────┐
                    │              │  Use GOALIE?   │
                    │              └─┬────────────┬─┘
                    │                │ YES        │ NO
                    │         ┌──────▼──────┐    │
                    │         │ GOAP Decomp │    │
                    │         └──────┬──────┘    │
                    │                │           │
                    └────────┬───────┴───────────┘
                             │
                   ┌─────────▼──────────┐
                   │  Swarm Size?       │
                   └─┬────────┬────────┬┘
                     │        │        │
            ┌────────▼──┐ ┌──▼────┐ ┌▼────────┐
            │ 3 agents  │ │5 agents│ │7 agents │
            │ (Low)     │ │(Medium)│ │(High)   │
            └────────┬──┘ └──┬─────┘ └┬────────┘
                     │       │        │
                     └───────┼────────┘
                             │
                   ┌─────────▼──────────┐
                   │  Priority Queue    │
                   │  Phase 1: Research │
                   │  Phase 2: Verify   │
                   │  Phase 3: Synthesize│
                   └─────────┬──────────┘
                             │
                   ┌─────────▼──────────┐
                   │ Parallel Execution │
                   │ (max 4 concurrent) │
                   └─────────┬──────────┘
                             │
                   ┌─────────▼──────────┐
                   │  All Complete?     │
                   └─┬────────────────┬─┘
                     │ NO             │ YES
              ┌──────▼──────┐         │
              │ Wait & Retry│         │
              └──────┬──────┘         │
                     │                │
                     └────────────────┘
                                      │
                            ┌─────────▼──────────┐
                            │  Aggregate Results │
                            └─────────┬──────────┘
                                      │
                            ┌─────────▼──────────┐
                            │ Generate Report    │
                            └─────────┬──────────┘
                                      │
                            ┌─────────▼──────────┐
                            │  Export Format?    │
                            └─┬────────┬────────┬┘
                              │        │        │
                      ┌───────▼──┐ ┌──▼───┐ ┌─▼────┐
                      │ Markdown │ │ JSON │ │ HTML │
                      └───────┬──┘ └──┬───┘ └─┬────┘
                              │       │       │
                              └───────┼───────┘
                                      │
                            ┌─────────▼──────────┐
                            │  Return to User    │
                            └────────────────────┘
```

---

## 📋 Configuration Matrix

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| **depth** | 5 | 1-10 | Research thoroughness |
| **time** | 120 min | 10-480 | Time budget |
| **swarm_size** | 5 | 3-7 | Agent count |
| **max_concurrent** | 4 | 1-8 | Parallel execution |
| **provider** | anthropic | any | AI model selection |
| **anti_hallucination** | high | low/med/high | Verification level |
| **citations** | true | bool | Require sources |
| **ed2551** | true | bool | Enhanced mode |

---

## 🎯 Use Case Examples

### Use Case 1: Simple Question (3 agents, 5 minutes)

```
Task: "What are webhooks?"
Complexity: Low
Depth: 3
Agents: 3 (Explorer, Depth, Synthesizer)
Timeline: 5 minutes
Cost: ~$0.50
```

### Use Case 2: Standard Research (5 agents, 15 minutes)

```
Task: "Compare REST vs GraphQL APIs"
Complexity: Medium
Depth: 5
Agents: 5 (Explorer, Depth, Verifier, Trend, Synthesizer)
Timeline: 15 minutes
Cost: ~$2.00
```

### Use Case 3: Deep Analysis (7 agents, 30 minutes)

```
Task: "Comprehensive AI safety governance analysis"
Complexity: High
Depth: 8
Agents: 7 (all roles)
Timeline: 30 minutes
Cost: ~$5.00
```

### Use Case 4: GOALIE Goal-Oriented (7 agents, 45 minutes)

```
Goal: "Research blockchain scalability solutions"
Sub-goals: 3 (Consensus, Layer 2, Implementations)
Complexity: Very High
Depth: 7
Agents: 7 per sub-goal (adaptive)
Timeline: 45 minutes
Cost: ~$8.00
```

---

## 🔗 Related Documentation

- **Architecture Overview**: `/docs/SWARM_ARCHITECTURE.md`
- **GOALIE Integration**: `/docs/GOALIE_INTEGRATION.md`
- **Permit Platform**: `/docs/PERMIT_PLATFORM_INTEGRATION.md`
- **ReasoningBank**: `/docs/research-swarm-updates.md`
- **Production Validation**: `/docs/PRODUCTION_VALIDATION.md`

---

## 📞 Support

For questions about this architecture:
- 📖 [Full Documentation](https://github.com/ruvnet/agentic-flow/tree/main/examples/research-swarm)
- 🐛 [Report Issues](https://github.com/ruvnet/agentic-flow/issues)
- 💬 [Discussions](https://github.com/ruvnet/agentic-flow/discussions)

---

**Document Version**: 1.0
**Last Updated**: November 4, 2025
**research-swarm**: v1.2.2
