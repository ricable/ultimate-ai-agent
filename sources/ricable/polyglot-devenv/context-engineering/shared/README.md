# Context Engineering Shared Resources

The **shared** directory contains resources used by both workspace (generation) and devpod (execution) environments.

## Structure

- **examples/** - Reference examples and patterns
  - `dojo/` - CopilotKit integration examples and patterns
  - `multi_agent_prp.md` - Multi-agent system example
  - `python-api-example.md` - Complete FastAPI implementation example
- **utils/** - Common utilities and tools
  - `argument-parser.nu` - Nushell argument parsing utilities
  - `command-builder.nu` - Command construction helpers
  - `dojo-integrator.nu` - CopilotKit/Dojo integration tools
  - `simple-parser.nu` - Basic parsing utilities
  - `template-engine.nu` - Template processing engine
- **schemas/** - Validation schemas and data models
- **docs/** - Shared documentation and implementation guides
  - `composite-template-system.md` - Template system architecture
  - `environment-adapter-implementation.md` - Environment adaptation patterns
  - `prp-builder-implementation.md` - PRP construction guidelines
  - `security-implementation.md` - Security patterns and practices
  - `version-control-and-scalability-implementation.md` - Scaling strategies

## Purpose

These shared resources provide:

1. **Common Patterns**: Reusable examples and implementations
2. **Utility Functions**: Tools used across workspace and devpod
3. **Validation**: Schemas for ensuring consistency
4. **Documentation**: Implementation guides and architectural decisions

## Examples

### CopilotKit Dojo Integration

The `examples/dojo/` directory contains a complete Next.js application demonstrating:
- **Agentic Chat**: Interactive AI conversations
- **Generative UI**: Dynamic interface generation
- **Human in the Loop**: Human oversight and intervention
- **Predictive State Updates**: Anticipatory state management
- **Shared State**: Cross-component state synchronization
- **Tool-based Generative UI**: Tool-driven interface generation

### Multi-Agent Systems

`examples/multi_agent_prp.md` demonstrates:
- **Agent Coordination**: Multiple AI agents working together
- **Task Distribution**: Parallel task execution across agents
- **State Synchronization**: Shared state management
- **Communication Patterns**: Inter-agent messaging

### Python API Examples

`examples/python-api-example.md` provides:
- **FastAPI Patterns**: Modern async API development
- **Database Integration**: SQLAlchemy async patterns
- **Authentication**: JWT-based security
- **Testing**: Comprehensive test coverage

## Utilities

### Nushell Tools

- **argument-parser.nu**: Parse command-line arguments with validation
- **command-builder.nu**: Construct complex commands dynamically
- **dojo-integrator.nu**: Integrate CopilotKit patterns into PRPs
- **template-engine.nu**: Process templates with variable substitution

### Usage Examples

```bash
# Use argument parser
nu shared/utils/argument-parser.nu --input "feature.md" --env "python"

# Build commands dynamically
nu shared/utils/command-builder.nu --command "devbox" --args "run test"

# Integrate dojo patterns
nu shared/utils/dojo-integrator.nu --feature "chat" --template "typescript"
```

## Integration

Shared resources are accessed from both workspace and devpod:

```bash
# From workspace
cd context-engineering/workspace
nu ../shared/utils/template-engine.nu --template templates/python_prp.md

# From devpod
cd context-engineering/devpod/environments/python
nu ../../../shared/utils/argument-parser.nu --validate
```

## Documentation

The `docs/` directory contains implementation guides:
- **Architecture**: How the system is designed
- **Patterns**: Common implementation patterns
- **Security**: Security considerations and practices
- **Scaling**: How to scale the system

These documents inform both workspace PRP generation and devpod execution patterns.