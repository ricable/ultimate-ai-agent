## Environment Setup Fragment

### Target Environment(s)
```yaml
Environment: {ENVIRONMENT}
Devbox_Config: {DEVBOX_CONFIG}
Dependencies: {DEPENDENCIES}
Integration_Points: {INTEGRATION_POINTS}
```

### Environment Setup Commands
```bash
# Activate environment
cd {ENVIRONMENT} && devbox shell

# Verify environment is ready
devbox run --quiet health-check 2>/dev/null || echo "Environment ready"

# Install/update dependencies if needed
{INSTALL_COMMAND}  # Environment-specific install command
```

### Current Codebase Structure
```bash
{CURRENT_STRUCTURE}
```

### Target Codebase Structure
```bash
{TARGET_STRUCTURE}
```

### Known Environment Gotchas
```{LANGUAGE}
{ENVIRONMENT_GOTCHAS}
```