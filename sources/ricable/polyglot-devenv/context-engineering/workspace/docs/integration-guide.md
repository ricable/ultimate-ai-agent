# Context Engineering Integration Guide

This guide explains how to integrate and use the context engineering framework within the polyglot development environment.

## Overview

The context engineering framework provides two main Claude Code commands:
- `/generate-prp` - Generates comprehensive Product Requirements Prompts
- `/execute-prp` - Executes PRPs to implement features

These commands work seamlessly with the existing polyglot environment structure and automation systems.

## Quick Start

### 1. Generate a PRP

```bash
# For a Python feature
/generate-prp features/user-api.md --env dev-env/python

# For a cross-environment feature
/generate-prp features/monitoring-dashboard.md --env multi

# For a specific template
/generate-prp features/cli-tool.md --template nushell
```

### 2. Execute a PRP

```bash
# Execute with validation
/execute-prp context-engineering/PRPs/user-api-python.md --validate

# Execute with monitoring
/execute-prp context-engineering/PRPs/user-api-python.md --monitor

# Execute for specific environment
/execute-prp context-engineering/PRPs/cli-tool-nushell.md --env dev-env/nushell
```

## Directory Structure

```
context-engineering/
├── templates/
│   ├── prp_base.md           # Polyglot-adapted base template
│   ├── python_prp.md         # Python-specific template
│   ├── typescript_prp.md     # TypeScript-specific template
│   ├── rust_prp.md           # Rust-specific template
│   ├── go_prp.md             # Go-specific template
│   └── nushell_prp.md        # Nushell-specific template
├── PRPs/                     # Generated PRPs
├── examples/                 # Example PRPs for reference
│   └── python-api-example.md # Complete FastAPI example
└── docs/                     # Documentation
    └── integration-guide.md  # This file
```

## Creating Feature Requests

Before generating a PRP, create a feature request file describing what you want to build:

### Feature Request Template

```markdown
# Feature: [Feature Name]

## FEATURE:
[Detailed description of what needs to be built]

## EXAMPLES:
[Reference existing code patterns in the polyglot environment]
- `python-env/src/services/example_service.py` - Service layer patterns
- `typescript-env/src/types/example.types.ts` - Type definitions
- `nushell-env/scripts/existing-script.nu` - Automation patterns

## DOCUMENTATION:
[Include relevant documentation URLs]
- FastAPI documentation: https://fastapi.tiangolo.com/
- TypeScript handbook: https://www.typescriptlang.org/docs/
- Nushell book: https://www.nushell.sh/book/

## OTHER CONSIDERATIONS:
- Environment: [dev-env/python|dev-env/typescript|dev-env/rust|dev-env/go|dev-env/nushell|multi]
- Integration points: [List environments this feature will interact with]
- Performance requirements: [Any specific performance needs]
- Security considerations: [Authentication, authorization, data validation]
```

### Example Feature Request

```markdown
# Feature: User Management API

## FEATURE:
Build a complete user management REST API with CRUD operations, JWT authentication, and PostgreSQL database integration. The API should include user registration, login, profile management, and admin operations.

## EXAMPLES:
- `dev-env/python/src/main.py` - FastAPI application structure
- `dev-env/python/src/models/` - Existing Pydantic model patterns
- Reference the FastAPI documentation patterns for async endpoints

## DOCUMENTATION:
- FastAPI: https://fastapi.tiangolo.com/tutorial/
- SQLAlchemy async: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
- Pydantic v2: https://docs.pydantic.dev/

## OTHER CONSIDERATIONS:
- Environment: dev-env/python
- Database: PostgreSQL with async SQLAlchemy
- Authentication: JWT-based with secure password hashing
- Testing: Comprehensive test coverage with pytest-asyncio
- Deployment: Container-ready with environment configuration
```

## Template Selection

The system automatically selects the appropriate template based on the target environment:

### Template Selection Logic

1. **Single Environment**: Uses language-specific template
   - `dev-env/python` → `python_prp.md`
   - `dev-env/typescript` → `typescript_prp.md`
   - `dev-env/rust` → `rust_prp.md`
   - `dev-env/go` → `go_prp.md`
   - `dev-env/nushell` → `nushell_prp.md`

2. **Multi-Environment**: Uses base template with cross-environment sections
   - `multi` → `prp_base.md` with polyglot considerations

3. **Manual Override**: Specify template explicitly
   - `--template python` → Force Python template regardless of environment

## Environment-Specific Features

### Python Environment
- **FastAPI Integration**: Async endpoints with dependency injection
- **Database Support**: SQLAlchemy async with Alembic migrations
- **Package Management**: uv exclusively (no pip/poetry/pipenv)
- **Testing**: pytest-asyncio with comprehensive coverage
- **Validation**: ruff formatting, mypy type checking

### TypeScript Environment
- **Type Safety**: Strict TypeScript mode, no `any` types
- **Modern Node.js**: ES modules, async/await patterns
- **Testing**: Jest with comprehensive test patterns
- **Quality**: ESLint + Prettier with strict rules
- **Error Handling**: Result pattern or proper error types

### Rust Environment
- **Memory Safety**: Leverage ownership system effectively
- **Async Runtime**: Tokio for async operations
- **Error Handling**: Result<T, E> with custom error types
- **Testing**: Cargo test with comprehensive scenarios
- **Performance**: Zero-cost abstractions

### Go Environment
- **Simplicity**: Clean, readable Go code
- **Context Usage**: context.Context for cancellation/timeouts
- **Error Handling**: Explicit error checking
- **Testing**: Table-driven tests
- **Interfaces**: Small, focused interface design

### Nushell Environment
- **Structured Data**: Leverage built-in data structures
- **Type Safety**: Type hints for all parameters
- **Automation**: Cross-environment orchestration
- **Pipeline Design**: Functions for data pipeline composition
- **Error Handling**: Graceful error handling with exit codes

## Integration with Existing Systems

### Hooks Automation
The context engineering system integrates with existing hooks:

```json
{
  "PostToolUse": [
    {
      "matcher": "generate-prp|execute-prp",
      "hooks": [
        {
          "type": "command",
          "command": "nu nushell-env/scripts/performance-analytics.nu measure 'context-engineering' 'prp-generation'"
        }
      ]
    }
  ]
}
```

### Intelligence Scripts
PRPs leverage existing intelligence scripts:

- **Performance Monitoring**: `nushell-env/scripts/performance-analytics.nu`
- **Security Scanning**: `nushell-env/scripts/security-scanner.nu`
- **Dependency Management**: `nushell-env/scripts/dependency-monitor.nu`
- **Environment Drift**: `nushell-env/scripts/environment-drift.nu`

### Validation Integration
Validation commands use existing polyglot tooling:

```bash
# Python
cd python-env && devbox run format && devbox run lint && devbox run test

# Cross-environment
nu nushell-env/scripts/validate-all.nu parallel
```

## Workflow Examples

### Single-Environment Feature

1. **Create Feature Request**
   ```bash
   # Create features/user-auth.md with requirements
   ```

2. **Generate PRP**
   ```bash
   /generate-prp features/user-auth.md --env dev-env/python
   ```

3. **Review and Execute**
   ```bash
   # Review generated PRP in context-engineering/PRPs/
   /execute-prp context-engineering/PRPs/user-auth-python.md --validate
   ```

### Cross-Environment Feature

1. **Create Multi-Environment Request**
   ```bash
   # Create features/monitoring-system.md
   ```

2. **Generate Comprehensive PRP**
   ```bash
   /generate-prp features/monitoring-system.md --env multi
   ```

3. **Execute with Monitoring**
   ```bash
   /execute-prp context-engineering/PRPs/monitoring-system-multi.md --monitor
   ```

### Iterative Development

1. **Start with Basic PRP**
   ```bash
   /generate-prp features/basic-api.md --env dev-env/python
   ```

2. **Execute and Test**
   ```bash
   /execute-prp context-engineering/PRPs/basic-api-python.md
   ```

3. **Enhance and Iterate**
   ```bash
   # Modify the PRP for additional features
   /execute-prp context-engineering/PRPs/enhanced-api-python.md --validate
   ```

## Best Practices

### Writing Effective Feature Requests

1. **Be Specific**: Detailed requirements with clear success criteria
2. **Reference Examples**: Point to existing code patterns in the environment
3. **Include Documentation**: Provide relevant URLs and resources
4. **Consider Integration**: Think about cross-environment interactions
5. **Security First**: Include security and validation requirements

### PRP Generation Tips

1. **Environment Research**: Let the system analyze existing patterns
2. **Template Selection**: Choose the most appropriate template
3. **Validation Gates**: Ensure all validation commands are executable
4. **Context Completeness**: Include all necessary documentation and examples

### Execution Best Practices

1. **Review PRPs**: Always review generated PRPs before execution
2. **Use Validation**: Enable validation flags for quality assurance
3. **Monitor Performance**: Use monitoring for complex implementations
4. **Iterative Approach**: Start simple and enhance incrementally

### Error Recovery

1. **Validation Failures**: Review error messages and fix issues systematically
2. **Implementation Blocks**: Use additional research and documentation
3. **Environment Issues**: Verify devbox setup and dependencies
4. **Cross-Environment**: Check integration points and communication

## Advanced Usage

### Custom Templates

Create custom templates for specific use cases:

```bash
# Copy and modify existing template
cp context-engineering/templates/python_prp.md context-engineering/templates/custom_api_prp.md

# Use custom template
/generate-prp features/special-api.md --template custom_api
```

### PRP Versioning

Track PRP evolution for complex features:

```bash
# Generate initial version
/generate-prp features/complex-feature.md --env python-env
# Creates: context-engineering/PRPs/complex-feature-python.md

# Generate enhanced version
/generate-prp features/complex-feature-v2.md --env python-env
# Creates: context-engineering/PRPs/complex-feature-v2-python.md
```

### Integration Testing

Test cross-environment integration:

```bash
# Generate PRPs for multiple environments
/generate-prp features/api-backend.md --env dev-env/python
/generate-prp features/web-frontend.md --env dev-env/typescript
/generate-prp features/monitoring.md --env dev-env/nushell

# Execute in order with dependency management
/execute-prp context-engineering/PRPs/api-backend-python.md
/execute-prp context-engineering/PRPs/web-frontend-typescript.md
/execute-prp context-engineering/PRPs/monitoring-nushell.md
```

## Troubleshooting

### Common Issues

1. **Template Not Found**
   - Verify template exists in `context-engineering/templates/`
   - Check template name spelling

2. **Environment Detection Failed**
   - Ensure feature request mentions target environment
   - Use `--env` flag to override

3. **Validation Failures**
   - Check devbox environment is activated
   - Verify all dependencies are installed
   - Review validation command syntax

4. **Cross-Environment Issues**
   - Verify all referenced environments exist
   - Check integration point configurations
   - Test environment switching manually

### Debug Commands

```bash
# Check template availability
ls context-engineering/templates/

# Verify environment setup
cd dev-env/python && devbox shell && devbox run --version

# Test validation commands manually
cd dev-env/python && devbox run lint && devbox run test

# Check cross-environment validation
nu dev-env/nushell/scripts/validate-all.nu parallel
```

## Contributing

### Adding New Templates

1. **Create Template**: Based on existing language-specific templates
2. **Test Template**: Generate and execute test PRPs
3. **Document Usage**: Add to this guide
4. **Integration**: Update command logic if needed

### Improving Existing Templates

1. **Identify Issues**: Common failure patterns or missing context
2. **Update Templates**: Add better examples and validation
3. **Test Changes**: Verify with multiple feature types
4. **Document Updates**: Update this guide with improvements

### Reporting Issues

1. **Template Issues**: Problems with generated PRPs
2. **Execution Failures**: Command or validation failures
3. **Integration Problems**: Cross-environment issues
4. **Documentation**: Unclear or missing documentation

For issues and contributions, use the project's standard Git workflow and issue tracking system.