# Phase 1: Discovery Prompt

## Objective
Analyze the codebase and understand the existing structure, patterns, and capabilities before making changes.

---

## Prompt Template

```
You are an expert software architect analyzing a unified AI agent development platform.

## Context
This monorepo contains 35+ projects from ricable, spectredve, and ruvnet repositories,
organized using the SPARC Framework (Specification, Pseudocode, Architecture, Refinement, Completion).

## Your Task
Perform comprehensive discovery analysis:

### 1. Codebase Structure Analysis
- Identify all major components and their relationships
- Map dependencies between packages
- Document the technology stack used

### 2. Pattern Recognition
- Identify common design patterns across projects
- Note architectural consistencies and inconsistencies
- Find shared utilities and abstractions

### 3. Integration Points
- List all MCP protocol integrations
- Document API endpoints and their purposes
- Identify database schemas and data flows

### 4. Quality Assessment
- Review test coverage across packages
- Identify technical debt and legacy code
- Note security considerations

## Output Format
Provide your analysis as:

1. **Executive Summary** (2-3 paragraphs)
2. **Component Map** (hierarchical list)
3. **Key Findings** (bullet points)
4. **Recommendations** (prioritized list)
5. **Questions for Clarification** (if any)

## Constraints
- Focus on facts from the codebase, not assumptions
- Cite specific files/lines when making observations
- Prioritize findings by impact on the unified platform goal
```

---

## Example Usage

```bash
# Using with Claude Code
claude "Run the discovery phase prompt on /Users/cedric/dev/ultimate-ai-agent"

# Using with SPARC CLI
sparc analyze --phase=discovery --path=/Users/cedric/dev/ultimate-ai-agent
```

## Expected Outputs

After running this prompt, you should have:
- [ ] Complete inventory of all packages and their purposes
- [ ] Dependency graph showing package relationships
- [ ] List of shared patterns and utilities
- [ ] Technical debt inventory
- [ ] Prioritized recommendations for unification

## Next Phase
Once discovery is complete, proceed to [02-design.md](02-design.md) for architecture planning.
