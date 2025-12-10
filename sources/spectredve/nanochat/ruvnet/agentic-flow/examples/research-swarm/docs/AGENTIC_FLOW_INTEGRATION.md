# Integrating Agentic-Flow Agents with Research Swarm

**Version**: 1.1.0+
**Status**: Supported
**Integration Type**: Hybrid - Mix research-swarm agents with agentic-flow agents

---

## 🎯 Overview

Research-swarm can leverage the full ecosystem of **66+ agentic-flow agents** to enhance research capabilities with specialized expertise.

### Why Integrate?

**Research-swarm agents** provide:
- Research-specific workflows (ED2551, ReasoningBank)
- SQLite persistence
- Long-horizon recursive framework

**Agentic-flow agents** provide:
- Specialized expertise (backend-dev, ml-developer, security, etc.)
- 66+ pre-built agent types
- Multi-provider support (Anthropic, OpenRouter, ONNX, Gemini)

**Together** = Comprehensive research with domain expertise!

---

## 🔌 Integration Methods

### Method 1: Custom Agent Roles (Recommended)

Add agentic-flow agents as custom roles in swarm decomposition.

#### Example: Technical Analysis Swarm

```javascript
// lib/swarm-decomposition-custom.js
import { decomposeTask } from './swarm-decomposition.js';

export function decomposeTaskWithAgenticFlow(task, options = {}) {
  const baseAgents = decomposeTask(task, options);

  // Add agentic-flow specialized agents
  const customAgents = [
    {
      id: 'backend-specialist',
      agent: 'backend-dev', // agentic-flow agent
      task: `Analyze backend architecture for: ${task}`,
      config: {
        depth: options.depth,
        timeMinutes: Math.floor(options.timeMinutes * 0.15),
        focus: 'narrow',
        role: 'backend-specialist',
        priority: 1,
        provider: 'anthropic' // or 'openrouter', 'gemini'
      }
    },
    {
      id: 'security-auditor',
      agent: 'reviewer', // agentic-flow security reviewer
      task: `Security analysis for: ${task}`,
      config: {
        depth: options.depth,
        timeMinutes: Math.floor(options.timeMinutes * 0.10),
        focus: 'narrow',
        role: 'security-auditor',
        priority: 2 // Run after research
      }
    },
    {
      id: 'ml-specialist',
      agent: 'ml-developer', // agentic-flow ML agent
      task: `Machine learning perspective on: ${task}`,
      config: {
        depth: options.depth,
        timeMinutes: Math.floor(options.timeMinutes * 0.15),
        focus: 'balanced',
        role: 'ml-specialist',
        priority: 1
      }
    }
  ];

  // Merge with base agents (keep synthesizer at end)
  const synthesizer = baseAgents.pop(); // Remove synthesizer
  return [...baseAgents, ...customAgents, synthesizer]; // Add back at end
}
```

#### Usage

```bash
# Create custom CLI command or use programmatically
node research-with-custom-agents.js "Analyze ML model deployment architecture"
```

---

### Method 2: Direct MCP Tool Integration

Use agentic-flow MCP tools alongside research-swarm.

#### Example: Research + Code Generation

```javascript
// scripts/research-and-code.js
import { executeSwarm } from '../lib/swarm-executor.js';
import { decomposeTask } from '../lib/swarm-decomposition.js';

// Step 1: Research phase (research-swarm)
const researchTask = "Best practices for microservices architecture";
const swarmAgents = decomposeTask(researchTask, {
  depth: 5,
  timeMinutes: 30
});

const researchResults = await executeSwarm(swarmAgents);

// Step 2: Code generation phase (agentic-flow)
// Use MCP tool: mcp__agentic-flow__agentic_flow_agent
const codeTask = {
  agent: 'coder',
  task: `Based on research findings, generate microservices boilerplate code`,
  provider: 'anthropic',
  model: 'claude-sonnet-4-5-20250929'
};

// Execute via MCP or spawn directly
// Result: Research insights + working code
```

---

### Method 3: Swarm Executor Modification

Modify the swarm executor to support agentic-flow agents directly.

#### Enhanced Executor

```javascript
// lib/swarm-executor-enhanced.js
import { executeSwarm as baseExecuteSwarm } from './swarm-executor.js';
import { spawn } from 'child_process';
import path from 'path';

export async function executeSwarmWithAgenticFlow(agents, options = {}) {
  // Separate research-swarm and agentic-flow agents
  const researchAgents = agents.filter(a => isResearchAgent(a.agent));
  const agenticFlowAgents = agents.filter(a => !isResearchAgent(a.agent));

  // Execute research-swarm agents with base executor
  const researchResults = await baseExecuteSwarm(researchAgents, options);

  // Execute agentic-flow agents
  const agenticFlowResults = await executeAgenticFlowAgents(
    agenticFlowAgents,
    options
  );

  return {
    research: researchResults,
    agenticFlow: agenticFlowResults,
    combined: [...researchResults.agents, ...agenticFlowResults]
  };
}

async function executeAgenticFlowAgents(agents, options) {
  const results = [];

  for (const agent of agents) {
    const result = await executeAgenticFlowAgent(agent, options);
    results.push(result);
  }

  return results;
}

async function executeAgenticFlowAgent(agent, options) {
  return new Promise((resolve) => {
    // Use agentic-flow CLI or MCP tool
    const args = [
      'npx', 'agentic-flow', 'agent', 'run',
      agent.agent,
      '--task', agent.task,
      '--provider', agent.config.provider || 'anthropic'
    ];

    const child = spawn('node', args, { stdio: 'pipe' });

    let output = '';
    child.stdout.on('data', (data) => output += data.toString());

    child.on('close', (code) => {
      resolve({
        agent: agent.agent,
        role: agent.config.role,
        success: code === 0,
        output
      });
    });
  });
}

function isResearchAgent(agentType) {
  return agentType === 'researcher';
}
```

---

## 🎓 Example Use Cases

### Use Case 1: Full-Stack Analysis

Research architecture + generate code + review security

```javascript
const agents = [
  // Research phase (research-swarm)
  { agent: 'researcher', role: 'explorer', ... },
  { agent: 'researcher', role: 'depth-analyst', ... },

  // Domain expertise (agentic-flow)
  { agent: 'backend-dev', role: 'backend-specialist', ... },
  { agent: 'system-architect', role: 'architecture-reviewer', ... },

  // Code generation (agentic-flow)
  { agent: 'coder', role: 'implementation-specialist', ... },

  // Quality assurance (agentic-flow)
  { agent: 'reviewer', role: 'code-reviewer', ... },
  { agent: 'tester', role: 'test-engineer', ... },

  // Synthesis (research-swarm)
  { agent: 'researcher', role: 'synthesizer', ... }
];
```

### Use Case 2: ML Research + Model Development

Research ML algorithms + implement + benchmark

```javascript
const agents = [
  // Research phase
  { agent: 'researcher', role: 'explorer', task: 'Survey ML algorithms' },
  { agent: 'researcher', role: 'depth-analyst', task: 'Deep dive into transformers' },

  // ML expertise
  { agent: 'ml-developer', role: 'ml-specialist', task: 'Model architecture design' },
  { agent: 'ml-developer', role: 'training-specialist', task: 'Training strategy' },

  // Implementation
  { agent: 'coder', role: 'implementation', task: 'Implement model' },

  // Testing
  { agent: 'tester', role: 'benchmark', task: 'Performance benchmarks' },

  // Synthesis
  { agent: 'researcher', role: 'synthesizer', task: 'Combine all findings' }
];
```

### Use Case 3: Security Research + Audit

Research vulnerabilities + audit code + recommendations

```javascript
const agents = [
  // Research phase
  { agent: 'researcher', role: 'explorer', task: 'Survey security threats' },
  { agent: 'researcher', role: 'verifier', task: 'Verify CVE data' },

  // Security expertise
  { agent: 'reviewer', role: 'security-auditor', task: 'Code security audit' },
  { agent: 'code-analyzer', role: 'vulnerability-scanner', task: 'Scan codebase' },

  // Documentation
  { agent: 'api-docs', role: 'security-docs', task: 'Security documentation' },

  // Synthesis
  { agent: 'researcher', role: 'synthesizer', task: 'Security report' }
];
```

---

## 🛠️ Available Agentic-Flow Agents

### Core Development (5 agents)
- `coder` - Implementation specialist
- `reviewer` - Code review and quality
- `tester` - Testing and QA
- `planner` - Planning and orchestration
- `researcher` - Research (overlaps with research-swarm)

### Specialized Development (15 agents)
- `backend-dev` - Backend API development
- `mobile-dev` - React Native mobile apps
- `ml-developer` - Machine learning models
- `cicd-engineer` - CI/CD pipelines
- `api-docs` - API documentation
- `system-architect` - System architecture
- `code-analyzer` - Code quality analysis
- `base-template-generator` - Template generation
- `sparc-coder` - SPARC methodology
- `tdd-london-swarm` - TDD development
- And more...

### GitHub Integration (8 agents)
- `github-modes` - GitHub workflow orchestration
- `pr-manager` - Pull request management
- `code-review-swarm` - Multi-agent code review
- `issue-tracker` - Issue management
- `release-manager` - Release coordination
- `workflow-automation` - GitHub Actions
- `repo-architect` - Repository structure
- `multi-repo-swarm` - Cross-repo coordination

### Specialized Systems (20+ agents)
- Consensus protocols (byzantine, raft, gossip)
- Performance optimization
- Swarm coordination
- Memory management
- And many more...

**Total**: 66+ agents available!

---

## 📋 Integration Checklist

### Prerequisites

- ✅ research-swarm installed
- ✅ agentic-flow installed: `npm install -g agentic-flow`
- ✅ MCP server configured (optional)
- ✅ API keys configured (ANTHROPIC_API_KEY, etc.)

### Implementation Steps

1. **Choose integration method**:
   - Custom roles (easiest)
   - MCP tools (most flexible)
   - Executor modification (most powerful)

2. **Identify agents needed**:
   ```bash
   # List all agentic-flow agents
   npx agentic-flow agent list
   ```

3. **Create custom swarm configuration**:
   - Define agent roles
   - Set priorities
   - Configure time budgets

4. **Test integration**:
   - Run with single agentic-flow agent first
   - Verify output integration
   - Scale to full swarm

5. **Monitor performance**:
   - Track API costs
   - Measure execution time
   - Validate quality

---

## 💰 Cost Considerations

### Hybrid Swarm Costs

**Research-swarm agents**: Standard research costs
**Agentic-flow agents**: Additional API costs per agent

**Example** (5 research + 3 agentic-flow agents):
- Research-swarm: 5x baseline
- Agentic-flow: 3x baseline
- **Total**: 8x baseline cost

**Optimization**:
```bash
# Minimal hybrid swarm
research-swarm research researcher "task" --swarm-size 3
# + 1-2 agentic-flow agents for domain expertise
# Total: 4-5x cost vs 8x
```

---

## 🚀 Quick Start Example

### Hybrid Research + Code Generation

Create `scripts/hybrid-research.js`:

```javascript
#!/usr/bin/env node

import { decomposeTask } from '../lib/swarm-decomposition.js';
import { executeSwarm } from '../lib/swarm-executor.js';
import { spawn } from 'child_process';

async function hybridResearch(task) {
  console.log('🔬 Phase 1: Research with research-swarm...\n');

  // Research phase
  const researchAgents = decomposeTask(task, {
    depth: 5,
    timeMinutes: 20,
    swarmSize: 3 // Minimal swarm
  });

  const researchResults = await executeSwarm(researchAgents);

  console.log('\n💻 Phase 2: Code generation with agentic-flow...\n');

  // Code generation phase
  const codeTask = `Based on research findings, generate implementation for: ${task}`;

  await new Promise((resolve) => {
    const child = spawn('npx', [
      'agentic-flow', 'agent', 'run',
      'coder',
      '--task', codeTask,
      '--provider', 'anthropic'
    ], { stdio: 'inherit' });

    child.on('close', resolve);
  });

  console.log('\n✅ Hybrid research + code generation complete!\n');
}

// Run
const task = process.argv[2] || 'Build a REST API for user management';
hybridResearch(task);
```

**Usage**:
```bash
node scripts/hybrid-research.js "Build authentication system"
```

---

## 🎯 Best Practices

### 1. Role Assignment
- **Research-swarm**: Broad research, verification, synthesis
- **Agentic-flow**: Domain expertise, implementation, specialized analysis

### 2. Priority Coordination
- Research phase: Priority 1
- Domain expertise: Priority 1 (parallel with research)
- Implementation: Priority 2 (after research)
- Testing/Review: Priority 2
- Synthesis: Priority 3 (always last)

### 3. Time Allocation
```javascript
// 30-minute budget example
{
  research: 40% (12min), // research-swarm
  expertise: 30% (9min),  // agentic-flow domain agents
  implementation: 20% (6min), // agentic-flow code generation
  synthesis: 10% (3min)  // research-swarm synthesis
}
```

### 4. Cost Management
- Start with minimal swarm (3 research + 1-2 agentic-flow)
- Scale up only for complex tasks
- Use `--single-agent` for simple queries
- Monitor API usage

---

## 📚 Documentation References

### Research-Swarm Docs
- [SWARM_ARCHITECTURE.md](./SWARM_ARCHITECTURE.md)
- [V1.1.0_RELEASE_NOTES.md](./V1.1.0_RELEASE_NOTES.md)
- [README.md](../README.md)

### Agentic-Flow Docs
- Main repo: https://github.com/ruvnet/agentic-flow
- Agent list: `npx agentic-flow agent list`
- MCP documentation: See agentic-flow README

---

## ✅ Summary

**Yes, you can integrate agentic-flow agents!**

**Benefits**:
- ✅ Combine research depth with domain expertise
- ✅ Leverage 66+ specialized agents
- ✅ Generate code from research findings
- ✅ Multi-domain comprehensive analysis

**Recommended Approach**:
1. Use research-swarm for broad research and synthesis
2. Add agentic-flow agents for specialized expertise
3. Configure priorities to coordinate execution
4. Monitor costs and optimize as needed

**Example Command** (future):
```bash
research-swarm research researcher "Build ML pipeline" \
  --swarm-size 3 \
  --add-agent ml-developer \
  --add-agent backend-dev \
  --add-agent tester
```

---

**Questions?** Open an issue: https://github.com/ruvnet/agentic-flow/issues
