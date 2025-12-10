# Python DevPod Environment

Configuration for Python-based PRP execution in containerized DevPod environments.

## Environment Specification

- **Python Version**: 3.12.11
- **Package Manager**: uv 0.7.19 (exclusively)
- **Container**: `polyglot-python-devpod-{timestamp}-{N}`
- **Base Environment**: `dev-env/python`

## DevPod Integration

### Provisioning

```bash
# Single workspace
/devpod-python
# Creates: polyglot-python-devpod-20250107-121657-1

# Multiple workspaces (parallel development)
/devpod-python 3
# Creates: 
#   polyglot-python-devpod-20250107-121657-1
#   polyglot-python-devpod-20250107-121704-2  
#   polyglot-python-devpod-20250107-121710-3
```

### Resource Limits

- **Max per Command**: 10 workspaces
- **Max Total**: 15 containers across all environments
- **Max Python**: 5 concurrent Python containers
- **Performance**: ~5s provisioning time per workspace

## VS Code Integration

### Extensions

Auto-installed extensions:
- **Python** - Core Python support
- **Pylance** - Advanced Python language server
- **Python Debugger** - Debugging capabilities
- **autopep8** - Code formatting
- **ESLint** - Linting support

### Configuration

Workspace settings automatically configured:
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.formatting.provider": "autopep8",
  "editor.formatOnSave": true
}
```

## PRP Execution

### Directory Structure

```
environments/python/
├── PRPs/                    # PRPs ready for execution
├── configs/                 # Python-specific configurations
├── templates/               # Environment execution templates
├── monitoring/              # Python-specific monitoring
└── results/                 # Execution results and artifacts
```

### Execution Workflow

1. **Receive PRP**: Generated PRP moves from workspace to `PRPs/`
2. **Validate Environment**: Ensure Python DevPod is provisioned
3. **Execute PRP**: Run PRP in isolated container environment
4. **Monitor Execution**: Track performance, resource usage, quality
5. **Collect Results**: Store results in `results/` and report to `../execution/`

### Example Execution

```bash
# Execute PRP in Python DevPod
/devpod-python
/execute-prp context-engineering/devpod/environments/python/PRPs/user-api-python.md --validate

# Monitor execution
nu context-engineering/devpod/monitoring/python-monitor.nu --workspace polyglot-python-devpod-20250107-121657-1
```

## Framework Patterns

### FastAPI Applications

Standard patterns for FastAPI-based PRPs:
- **Async/Await**: All endpoints use async patterns
- **Dependency Injection**: Proper DI for database, auth, etc.
- **SQLAlchemy Async**: Database integration with async SQLAlchemy
- **Pydantic v2**: Data validation and serialization
- **pytest-asyncio**: Comprehensive async testing

### Package Management

- **uv exclusively**: No pip, poetry, or pipenv
- **Virtual environments**: Automatic venv creation and management
- **Lock files**: uv.lock for reproducible builds
- **Fast installs**: Leverages uv's performance optimizations

### Quality Standards

- **ruff**: Formatting and linting (88 character limit)
- **mypy**: Type checking (strict mode)
- **pytest**: Testing with async support
- **Coverage**: Minimum 80% code coverage

## Performance Optimization

### Container Optimization

- **Layer Caching**: Efficient Docker layer management
- **Pre-installed Dependencies**: Common packages pre-installed
- **Fast Startup**: ~5s container provisioning
- **Resource Management**: Automatic cleanup and optimization

### Development Performance

```bash
# Personal performance aliases (add to CLAUDE.local.md)
alias py-perf="nu context-engineering/devpod/monitoring/python-monitor.nu --perf"
alias py-metrics="nu dev-env/nushell/scripts/performance-analytics.nu report --env python"
alias py-optimize="nu dev-env/nushell/scripts/performance-analytics.nu optimize --env python"
```

## Security Features

### Container Security

- **Isolation**: Complete filesystem and network isolation
- **Read-only**: Core system files are read-only
- **User Privileges**: Non-root execution
- **Resource Limits**: CPU and memory constraints

### Code Security

- **Secret Scanning**: Automatic detection of secrets and keys
- **Dependency Scanning**: Known vulnerability detection
- **Input Validation**: Pydantic-based validation at boundaries
- **Security Headers**: Automatic security header injection

## Monitoring & Observability

### Performance Tracking

- **Build Times**: Track uv install and build performance
- **Test Execution**: Monitor pytest execution times
- **Resource Usage**: CPU, memory, disk usage tracking
- **Quality Metrics**: Coverage, lint scores, type checking results

### Alerts & Notifications

- **Performance Degradation**: Alert on slow builds or tests
- **Quality Issues**: Notify on coverage drops or lint failures
- **Security Issues**: Immediate alerts on security scan failures
- **Resource Limits**: Warning on approaching container limits

## Personal Integration

### Workflow Aliases

```bash
# Add to CLAUDE.local.md for personal productivity
alias py-prp="personal-prp-workflow \$1 python"
alias py-dev="/devpod-python && code ."
alias py-test="/devpod-python && uv run pytest"
alias py-lint="/devpod-python && uv run ruff check"
alias py-format="/devpod-python && uv run ruff format"
```

### Development Patterns

```python
# Personal FastAPI patterns for PRPs
from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
import asyncio

# Standard async endpoint pattern
@app.post("/api/resource")
async def create_resource(
    resource: ResourceCreate,
    db: AsyncSession = Depends(get_db)
) -> ResourceResponse:
    # Implementation with proper error handling
    pass
```

## Troubleshooting

### Common Issues

1. **Container Provisioning Fails**
   - Check Docker daemon status
   - Verify available disk space
   - Review resource limits

2. **PRP Execution Errors**
   - Validate PRP syntax and structure
   - Check environment dependencies
   - Review execution logs

3. **Performance Issues**
   - Monitor container resource usage
   - Check for memory leaks in Python code
   - Optimize database queries

### Debug Commands

```bash
# Check container status
devpod list | grep python

# Monitor container resources
devpod logs polyglot-python-devpod-20250107-121657-1

# Debug PRP execution
/devpod-python && /execute-prp PRPs/debug-test.md --verbose --debug
```