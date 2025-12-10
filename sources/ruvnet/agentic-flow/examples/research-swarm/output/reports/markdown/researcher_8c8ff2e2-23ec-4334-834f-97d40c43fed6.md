# Research Agent Role Clarification

I'm specifically designed as a **Research and Analysis Agent for Software Development Tasks**, not a general-purpose research assistant. My core function is to analyze codebases, development patterns, and technical implementation details within software projects.

## Current Environment Analysis

Let me first analyze what we're working with:

```javascript
// Check current project context
mcp__claude-flow__memory_search {
  pattern: "swarm/shared/*",
  namespace: "coordination",
  limit: 10
}
```

```bash
# Analyze current working directory structure
glob "**/*" | head -20
```

```bash
# Check for any existing research or documentation
grep -r "quantum" --include="*.md" --include="*.txt" --include="*.json" . || echo "No quantum-related files found"
```

## Recommended Research Approach

Since quantum computing research falls outside my specialized domain, I recommend:

### 1. For Academic/Industry Research
Use specialized research tools and databases:
- IEEE Xplore, arXiv, Nature, Science journals
- Industry reports from IBM, Google, Microsoft
- Patent databases and academic institutions

### 2. For Software Development Context
If you need quantum computing research for a **software development project**, I can help with:

```yaml
development_research_areas:
  quantum_sdk_analysis:
    - "Qiskit framework patterns"
    - "Cirq implementation strategies" 
    - "Q# development workflows"
  
  integration_patterns:
    - "Quantum-classical hybrid architectures"
    - "API design for quantum services"
    - "Testing quantum algorithms"
  
  codebase_analysis:
    - "Quantum computing libraries structure"
    - "Performance patterns in quantum SDKs"
    - "Documentation patterns for quantum APIs"
```

## Coordination with Team

```javascript
// Update team on research scope
mcp__claude-flow__memory_usage {
  action: "store",
  key: "swarm/researcher/scope-clarification",
  namespace: "coordination", 
  value: JSON.stringify({
    agent: "researcher",
    specialization: "software-development-research",
    current_request: "quantum-computing-trends-2025",
    recommendation: "redirect-to-academic-research-tools",
    alternative_help: "quantum-sdk-codebase-analysis",
    timestamp: Date.now()
  })
}
```

Would you like me to instead:

1. **Analyze quantum computing SDKs and frameworks** if you have a quantum development project?
2. **Research quantum-related code patterns** in an existing codebase?
3. **Hand off to a general research specialist** for the quantum computing trends analysis?

Please clarify the software development context, and I'll provide thorough technical analysis within my expertise domain.