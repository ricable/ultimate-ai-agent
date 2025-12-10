# Documentation Swarm Coordinator

This directory contains the documentation swarm coordinator for Nova Medicina project.

## Quick Start

```bash
# Run the coordinator
npm run generate-docs

# Or directly
node docs/generate-docs.js
```

## What It Does

The coordinator sets up a swarm of 6 concurrent documentation agents:

1. **README Writer** → `README.md`
   - Project overview with badges and quick start
   - Target: Developers and technical users

2. **Patient Guide** → `PATIENT_GUIDE.md`
   - Simple, friendly language for patients
   - Target: Patients and caregivers

3. **Provider Guide** → `PROVIDER_GUIDE.md`
   - Technical guide for healthcare providers
   - Target: Healthcare providers and IT staff

4. **API Documentation** → `API.md`
   - Complete API reference with examples
   - Target: Developers integrating the API

5. **Examples Generator** → `EXAMPLES.md`
   - Practical usage scenarios and code examples
   - Target: Developers implementing the project

6. **Safety Documentation** → `SAFETY.md`
   - Medical disclaimers and legal warnings
   - Target: All users

## Coordination via Memory

All agents coordinate through `.swarm-memory/` directory:

```
.swarm-memory/
├── manifest.json                    # Agent tracking
├── project_metadata.json           # Project info
├── coordination_rules.json         # Consistency rules
├── specs_*.json                    # Agent specifications
└── agent-prompts_all.json         # Full agent prompts
```

## Using with Claude Code

After running the coordinator, copy the Task() calls from the output and paste them into Claude Code in a **single message**:

```javascript
Task("Generate comprehensive README...", "...", "coder")
Task("Create patient-friendly documentation...", "...", "coder")
Task("Generate technical guide...", "...", "coder")
Task("Document all API endpoints...", "...", "coder")
Task("Create practical usage scenarios...", "...", "coder")
Task("Generate safety disclaimers...", "...", "coder")
```

All agents will execute concurrently with memory coordination.

## Memory Hooks

Each agent automatically:

### Before Starting
```bash
npx claude-flow@alpha hooks session-restore --session-id "docswarm-xxx"
```

### During Work
```bash
npx claude-flow@alpha hooks post-edit --file "filename.md" --memory-key "swarm/agent/progress"
npx claude-flow@alpha hooks notify --message "agent: Completed section X"
```

### After Completion
```bash
npx claude-flow@alpha hooks post-task --task-id "docswarm-xxx-agent"
```

## Monitoring Progress

```bash
# Watch memory updates
tail -f .swarm-memory/*.json

# Check manifest
cat .swarm-memory/manifest.json

# View agent prompts
cat .swarm-memory/agent-prompts_all.json
```

## Output Files

All documentation will be generated in `/home/user/agentic-flow/nova-medicina/docs/`:

- `README.md` - Main project documentation
- `PATIENT_GUIDE.md` - Patient-friendly guide
- `PROVIDER_GUIDE.md` - Healthcare provider guide
- `API.md` - API reference
- `EXAMPLES.md` - Usage examples
- `SAFETY.md` - Safety and legal disclaimers

## Architecture

```
docs/
├── generate-docs.js           # Coordinator script
├── README-COORDINATOR.md      # This file
└── .swarm-memory/            # Coordination memory
    ├── manifest.json         # Agent tracking
    ├── specs_*.json          # Agent specifications
    └── coordination_rules.json  # Consistency rules
```

## Benefits

- **Concurrent Execution**: All 6 agents run in parallel
- **Memory Coordination**: Consistent terminology across docs
- **Quality Assurance**: Built-in checklists for each agent
- **Cross-References**: Agents aware of each other's work
- **Session Tracking**: Full audit trail via hooks

## Session ID

Each run generates a unique session ID (e.g., `docswarm-1762579255454`) for tracking and coordination.

---

**Created by**: Documentation Swarm Coordinator v1.0.0
**Framework**: agentic-flow with claude-flow hooks
**Project**: Nova Medicina AI Healthcare System
