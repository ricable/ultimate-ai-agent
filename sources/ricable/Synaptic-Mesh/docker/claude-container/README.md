# Claude Secure Container

A minimal, secure Docker container for running Claude AI tasks in a sandboxed environment.

## 🔒 Security Features

### Compliance Requirements Met
- ✅ **Local API Access Only**: Each node runs Claude with their own API credentials
- ✅ **No Shared Secrets**: No API keys or proxy access shared between nodes
- ✅ **Secure Sandbox**: Read-only filesystem with tmpfs workspace
- ✅ **User Control**: User maintains full control with their own API keys
- ✅ **Network Isolation**: API access only, no lateral network access
- ✅ **No Persistent State**: Tmpfs workspace cleared on container restart

### Security Hardening
- **Alpine Linux Base**: Minimal attack surface (< 50MB)
- **Non-root User**: Runs as uid:1000, no privileged access
- **Read-only Filesystem**: Container filesystem is immutable
- **Resource Limits**: 512MB RAM, 0.5 CPU cores maximum
- **Capability Dropping**: All Linux capabilities dropped except network
- **Tmpfs Workspace**: 100MB temporary workspace, cleared on restart
- **Security Options**: `no-new-privileges` enabled
- **Network Restriction**: Only API endpoints allowed

## 🚀 Quick Start

### Prerequisites
- Docker installed
- Claude API key from Anthropic

### 1. Set API Key
```bash
export CLAUDE_API_KEY="your-anthropic-api-key"
# OR
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 2. Run Interactive Mode
```bash
./run-container.sh interactive
```

Send JSON tasks via stdin:
```json
{"id": "1", "prompt": "Hello Claude!"}
```

### 3. Run Batch Mode
```bash
./run-container.sh batch example-tasks.json
```

## 📋 Usage Examples

### Interactive Mode
```bash
# Start interactive container
./run-container.sh interactive

# Send a task (stdin)
{"id": "task-1", "prompt": "Explain quantum computing in simple terms"}

# Receive response (stdout)
{
  "type": "task_result",
  "taskId": "task-1", 
  "success": true,
  "response": "Quantum computing is...",
  "metadata": {
    "model": "claude-3-sonnet-20240229",
    "usage": {...},
    "timestamp": "2024-01-15T10:30:00.000Z"
  }
}
```

### Batch Mode
Create `tasks.json`:
```json
{"id": "1", "prompt": "What is machine learning?"}
{"id": "2", "prompt": "Explain neural networks"}
{"id": "3", "prompt": "Benefits of containerization"}
```

Run batch:
```bash
./run-container.sh batch tasks.json > results.json
```

### Docker Compose
```bash
# Set API key in environment
export CLAUDE_API_KEY="your-key"

# Run with docker-compose
docker-compose up claude-container
```

## 🔧 Configuration

### Security Configuration (`security-config.json`)
```json
{
  "maxMemoryMB": 512,
  "maxExecutionTimeMs": 300000,
  "allowedApis": ["api.anthropic.com"],
  "workspaceDir": "/tmp/claude-work",
  "readOnlyMode": true,
  "networkRestricted": true,
  "rateLimiting": {
    "maxRequestsPerMinute": 20,
    "maxRequestsPerHour": 100
  }
}
```

### Environment Variables
- `CLAUDE_API_KEY` or `ANTHROPIC_API_KEY` (required)
- `NODE_ENV=production` (set automatically)
- `CLAUDE_SANDBOX_MODE=true` (security flag)
- `CLAUDE_NETWORK_RESTRICTED=true` (network isolation)
- `CLAUDE_FILESYSTEM_READONLY=true` (filesystem protection)

## 🛡️ Security Details

### Filesystem Security
- **Read-only root**: Container filesystem is immutable
- **Tmpfs workspace**: `/tmp/claude-work` (100MB, cleared on restart)
- **No persistent data**: No data survives container restart
- **User ownership**: All files owned by non-root user (1000:1000)

### Network Security
- **API whitelist**: Only `api.anthropic.com` allowed
- **No lateral access**: Cannot access other containers/hosts
- **DNS restrictions**: Only essential DNS queries permitted
- **Firewall**: Drop all traffic except HTTPS to API

### Runtime Security
- **Non-root user**: Runs as `claude:claude` (1000:1000)
- **No privileges**: `--security-opt no-new-privileges`
- **Capability drop**: All Linux capabilities removed except network
- **Resource limits**: 512MB RAM, 0.5 CPU cores
- **Process isolation**: Single process per container

### API Security
- **User-provided keys**: No shared or embedded API keys
- **Rate limiting**: 20 requests/minute, 100 requests/hour
- **Request validation**: All inputs sanitized and validated
- **Response filtering**: Outputs validated before return
- **Audit logging**: All API calls logged for compliance

## 🔍 Security Audit

Run security audit:
```bash
./run-container.sh audit
```

This performs:
- Container configuration validation
- Vulnerability scanning (if Trivy installed)
- Security posture verification
- Compliance checking

## 📊 Monitoring

### Health Checks
Container includes built-in health checks:
```bash
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}"
```

### Logging
Structured JSON logging to stdout:
```json
{
  "type": "task_result",
  "taskId": "...",
  "timestamp": "...",
  "success": true,
  "metadata": {...}
}
```

### Resource Monitoring
```bash
# Monitor resource usage
docker stats claude-secure-container
```

## 🔄 Synaptic Market Integration

This container is designed for the Synaptic Neural Mesh marketplace:

1. **Node Deployment**: Each market node runs their own container
2. **API Key Management**: Nodes use their own Anthropic API keys
3. **Task Distribution**: Tasks sent via JSON streaming interface
4. **Result Collection**: Responses collected via stdout
5. **Security Isolation**: No shared state between nodes

### Market Compliance
- ✅ **Decentralized**: No central API proxy
- ✅ **Privacy**: No data sharing between nodes
- ✅ **Security**: Sandboxed execution environment
- ✅ **Auditability**: Full logging and monitoring
- ✅ **Scalability**: Lightweight container deployment

## 🚨 Security Considerations

### Production Deployment
1. **API Key Rotation**: Regularly rotate API keys
2. **Image Updates**: Keep base image updated for security patches
3. **Network Segmentation**: Deploy in isolated network segments
4. **Monitoring**: Monitor for unusual API usage patterns
5. **Backup**: No persistent data, but monitor configurations

### Threat Model
- **Malicious Tasks**: Input validation prevents code injection
- **Resource Exhaustion**: Resource limits prevent DoS
- **Data Exfiltration**: Read-only filesystem prevents data theft
- **Privilege Escalation**: Non-root user prevents system compromise
- **Network Attacks**: Network isolation limits lateral movement

## 📚 Documentation

- [Security Architecture](security-architecture.md)
- [API Reference](api-reference.md)
- [Deployment Guide](deployment-guide.md)
- [Troubleshooting](troubleshooting.md)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add security tests
4. Submit pull request
5. Security review required

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Links

- [Synaptic Neural Mesh](https://github.com/ruvnet/synaptic-neural-mesh)
- [Anthropic API](https://docs.anthropic.com/)
- [Docker Security](https://docs.docker.com/engine/security/)
- [Container Best Practices](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)