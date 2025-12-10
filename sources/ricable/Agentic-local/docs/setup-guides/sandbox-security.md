# Sandbox Security Guide

## Overview

When AI agents generate and execute code, security becomes paramount. This guide explains the multi-layered security approach used in the Docker sandbox environment.

## Security Architecture

```
┌─────────────────────────────────────────────┐
│          AI Agent (agentic-flow)            │
│  "Generate a Python data analysis script"  │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│         Sandbox Manager (MCP)               │
│  - Validates code                           │
│  - Applies resource limits                  │
│  - Monitors execution                       │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│        Docker Container (Isolated)          │
│  ┌───────────────────────────────────────┐ │
│  │ Security Boundaries:                  │ │
│  │ ✓ Read-only root filesystem          │ │
│  │ ✓ Network disabled                    │ │
│  │ ✓ No new privileges                   │ │
│  │ ✓ All capabilities dropped            │ │
│  │ ✓ Memory limit: 2GB                   │ │
│  │ ✓ CPU limit: 2 cores                  │ │
│  │ ✓ Execution timeout: 5 minutes        │ │
│  └───────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

## Threat Model

### Threats Mitigated

1. **Malicious Code Execution**
   - Agent generates code that attempts to access host file system
   - **Mitigation**: Read-only root filesystem, isolated volumes

2. **Data Exfiltration**
   - Code attempts to send sensitive data over network
   - **Mitigation**: Network isolation (`--network none`)

3. **Resource Exhaustion**
   - Infinite loops or memory bombs
   - **Mitigation**: CPU/memory limits, execution timeouts

4. **Privilege Escalation**
   - Code attempts to gain root access
   - **Mitigation**: `--security-opt no-new-privileges`, all capabilities dropped

5. **Container Escape**
   - Exploiting Docker vulnerabilities
   - **Mitigation**: Latest Docker version, minimal attack surface

### Residual Risks

⚠️ **Not Fully Protected Against:**

1. **Side-Channel Attacks**: Timing attacks or cache-based attacks (mitigated by ephemeral containers)
2. **Zero-Day Docker Vulnerabilities**: Keep Docker updated
3. **Host Kernel Exploits**: Use Docker Desktop on macOS for additional VM isolation

## Configuration

### Recommended Settings (Production)

```json
{
  "mcpServers": {
    "docker-sandbox": {
      "env": {
        "SANDBOX_TYPE": "docker",
        "SANDBOX_MEMORY_LIMIT": "2g",
        "SANDBOX_CPU_LIMIT": "2",
        "SANDBOX_TIMEOUT": "300000",
        "SANDBOX_NETWORK": "none",
        "ALLOWED_LANGUAGES": "javascript,typescript,python",
        "SANDBOX_VOLUMES": "./sandbox-volumes"
      }
    }
  }
}
```

### Paranoid Settings (Maximum Security)

For highly sensitive environments:

```json
{
  "mcpServers": {
    "docker-sandbox": {
      "env": {
        "SANDBOX_TYPE": "docker",
        "SANDBOX_MEMORY_LIMIT": "512m",
        "SANDBOX_CPU_LIMIT": "1",
        "SANDBOX_TIMEOUT": "60000",
        "SANDBOX_NETWORK": "none",
        "ALLOWED_LANGUAGES": "javascript",
        "SANDBOX_VOLUMES": "./sandbox-volumes",
        "DISABLE_STDOUT_STREAMING": "true",
        "MAX_OUTPUT_SIZE": "10240"
      }
    }
  }
}
```

### Development Settings (Relaxed)

For local testing only:

```json
{
  "mcpServers": {
    "docker-sandbox": {
      "env": {
        "SANDBOX_TYPE": "docker",
        "SANDBOX_MEMORY_LIMIT": "4g",
        "SANDBOX_CPU_LIMIT": "4",
        "SANDBOX_TIMEOUT": "600000",
        "SANDBOX_NETWORK": "bridge",
        "ALLOWED_LANGUAGES": "javascript,typescript,python,rust",
        "SANDBOX_VOLUMES": "./sandbox-volumes"
      }
    }
  }
}
```

## Docker Security Options Explained

### Read-Only Root Filesystem

```bash
--read-only
```

**Purpose**: Prevents code from modifying system files
**Impact**: Code can only write to explicitly mounted volumes or tmpfs

### Network Isolation

```bash
--network none
```

**Purpose**: Completely disables networking
**Alternative**: `--network bridge` with firewall rules for selective access

### Capabilities

```bash
--cap-drop ALL
```

**Purpose**: Removes all Linux capabilities (root-level permissions)
**Dropped capabilities include**:
- CAP_NET_RAW (raw sockets)
- CAP_SYS_ADMIN (mount, etc.)
- CAP_SYS_PTRACE (debugging other processes)

### Security Options

```bash
--security-opt no-new-privileges
```

**Purpose**: Prevents privilege escalation via setuid binaries

### Resource Limits

```bash
--memory 2g
--cpus 2
```

**Purpose**: Prevent DoS via resource exhaustion
**Tuning**: Adjust based on expected workload

## Testing Security

Run the test suite to verify security boundaries:

```bash
npm run sandbox:test
```

### Expected Results

| Test | Expected Outcome | Security Property |
|------|------------------|-------------------|
| Basic Execution | ✅ Success | Functionality verified |
| Python Execution | ✅ Success | Multi-language support |
| Error Handling | ✅ Errors caught | Graceful failure |
| Network Isolation | ✅ Network blocked | Data exfiltration prevented |
| Resource Limits | ✅ OOM or timeout | DoS prevention |
| File System | ❌ Read-only errors | Host protection |
| Agent Code | ✅ Code executes safely | Real-world simulation |

## Integration with Agentic Flow

### Secure Agent Configuration

```javascript
import { AgenticFlow } from 'agentic-flow';
import { DockerSandbox } from './src/sandbox/docker-sandbox.js';

const sandbox = new DockerSandbox({
  memoryLimit: '2g',
  cpuLimit: '2',
  timeout: 300000,
  network: 'none'
});

const agent = new AgenticFlow({
  provider: 'custom',
  baseURL: process.env.GAIANET_ENDPOINT,
  model: 'Qwen2.5-Coder-32B-Instruct',

  // Hook for code execution
  onCodeGenerated: async (code, language) => {
    console.log('Executing generated code in sandbox...');
    const result = await sandbox.execute(code, language);

    if (!result.success) {
      console.error('Execution failed:', result.stderr);
      // Agent will see the error and attempt to fix
      return { error: result.stderr };
    }

    console.log('Execution succeeded:', result.stdout);
    return { output: result.stdout };
  }
});

// Agent generates and tests code automatically
await agent.run('Create a function to parse CSV files');
```

## Monitoring and Alerting

### Log Analysis

Monitor sandbox logs for suspicious activity:

```bash
# Watch for excessive failures (potential attack)
tail -f logs/sandbox.log | grep "ERROR"

# Monitor resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

### Alerting Rules

Set up alerts for:

1. **High Failure Rate**: >10% of executions failing
2. **Timeout Frequency**: >5% of executions timing out
3. **Resource Limits Hit**: Container repeatedly hitting memory/CPU caps
4. **Unusual Patterns**: Same error repeated >100 times

## Advanced: Nested Sandboxing

For extreme security, run Docker inside a VM:

```
Host macOS
  └─> Docker Desktop (VM)
       └─> Docker Container (sandbox)
            └─> Agent Code Execution
```

**Setup**:
1. Docker Desktop on macOS already provides VM isolation
2. For Linux hosts, consider using Firecracker or gVisor

## Compliance Considerations

### GDPR / Data Privacy

- ✅ Ephemeral containers (no persistent data)
- ✅ Network isolation (no data leakage)
- ⚠️ Log retention policies needed

### SOC 2 / Security Certifications

- ✅ Principle of least privilege
- ✅ Defense in depth (multiple security layers)
- ✅ Audit logging
- ⚠️ Regular security assessments required

## Incident Response

### If Sandbox Escape Suspected

1. **Immediately kill all containers**:
   ```bash
   docker kill $(docker ps -q --filter "name=sandbox-*")
   ```

2. **Inspect logs**:
   ```bash
   docker events --since 1h | grep sandbox
   ```

3. **Check host file system** for unauthorized changes:
   ```bash
   find ./sandbox-volumes -type f -mmin -60
   ```

4. **Update Docker** and rebuild containers:
   ```bash
   brew upgrade --cask docker
   docker pull node:20-alpine
   docker pull python:3.11-alpine
   ```

## Best Practices Summary

✅ **DO**:
- Use read-only root filesystem
- Disable networking (or whitelist specific domains)
- Set memory and CPU limits
- Use ephemeral containers (--rm)
- Keep Docker updated
- Monitor execution logs
- Test security boundaries regularly

❌ **DON'T**:
- Mount host directories with write access
- Run containers with `--privileged`
- Use `--cap-add` unless absolutely necessary
- Share Docker socket with containers
- Trust AI-generated code without sandboxing
- Ignore timeout or resource limit violations

## Further Reading

- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [OWASP Docker Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)

---

**Remember**: Security is a process, not a product. Regularly review and update your sandbox configuration as threats evolve.
