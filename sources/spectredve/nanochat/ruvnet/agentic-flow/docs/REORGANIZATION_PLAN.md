# Documentation Reorganization Plan

## Current State
- **241 markdown files** across **33 directories**
- **3.9MB** total size
- Significant duplication between `releases/`, `version-releases/`, and root-level release files
- Validation reports scattered across multiple directories
- Feature docs mixed with architecture, guides, and reports

## Proposed Structure

```
docs/
├── README.md                          # Main navigation hub
├── INDEX.md                           # Quick reference index (keep)
├── CLAUDE.md                          # Claude Code config (keep)
├── LICENSE                            # License file (keep)
│
├── features/                          # Feature-specific documentation
│   ├── agentdb/                       # AgentDB feature
│   ├── reasoningbank/                 # ReasoningBank feature
│   ├── quic/                          # QUIC transport
│   ├── federation/                    # Federation features
│   ├── agent-booster/                 # Agent Booster
│   └── router/                        # Multi-model router
│
├── architecture/                      # System architecture docs (keep & enhance)
│   ├── README.md
│   └── [existing architecture docs]
│
├── guides/                            # User guides (keep & organize)
│   ├── README.md
│   ├── getting-started/
│   ├── mcp/
│   ├── deployment/
│   └── advanced/
│
├── api/                               # API documentation
│   ├── README.md
│   ├── mcp-tools/
│   └── cli/
│
├── development/                       # Developer documentation
│   ├── README.md
│   ├── integrations/
│   ├── testing/
│   └── contributing/
│
├── releases/                          # Release documentation
│   ├── README.md
│   ├── current/
│   ├── archive/
│   └── planning/
│
├── validation/                        # Validation & verification
│   ├── README.md
│   ├── reports/
│   └── benchmarks/
│
└── archive/                           # Historical/deprecated docs (keep)
    └── [outdated documents]
```

## Consolidation Actions

### 1. Feature Documentation
- **AgentDB**: Merge `agentdb/` + related root files
- **ReasoningBank**: Consolidate `reasoningbank/` docs
- **QUIC**: Merge `quic/` + `plans/QUIC/` + root QUIC files
- **Federation**: Move architecture/FEDERATION-*.md to features/federation/
- **Agent Booster**: Move `plans/agent-booster/` to features/

### 2. Release Documentation
- Consolidate: `releases/`, `version-releases/`, root release files
- Create: `current/` (active releases), `archive/` (historical)

### 3. Validation & Testing
- Merge: `validation/`, `validation-reports/`, `testing/`, `reports/`
- Organize by: validation type and feature

### 4. Guides
- Reorganize into logical subsections
- Create getting-started guides
- Separate MCP, deployment, and advanced topics

### 5. Remove Duplicates
- Multiple validation reports for same versions
- Duplicate QUIC documentation
- Redundant integration status docs

## Files to Archive
- Old validation reports (pre-v1.5.0)
- Superseded implementation plans
- Outdated status reports
- Legacy integration docs

## Benefits
- Clear feature-based organization
- Reduced duplication (~30% reduction expected)
- Easier navigation
- Better maintenance
- Logical grouping by purpose
