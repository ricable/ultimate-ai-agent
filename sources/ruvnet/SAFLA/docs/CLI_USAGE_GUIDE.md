# SAFLA CLI Management System - Complete Usage Guide

## Overview

The SAFLA CLI Management System provides comprehensive command-line interface for managing all aspects of the SAFLA (Self-Aware Feedback Loop Algorithm) system. This guide covers all available commands, interactive features, and advanced usage patterns.

## Installation & Setup

### Quick Start
```bash
# Make CLI executable
chmod +x safla/cli_main.py

# Run interactive setup wizard
python safla/cli_main.py setup

# Start the system
python safla/cli_main.py system start
```

### Advanced Setup
```bash
# Custom configuration
python safla/cli_main.py --config custom_config.env setup

# Debug mode
python safla/cli_main.py --debug system validate

# Quiet mode (minimal output)
python safla/cli_main.py --quiet system status
```

## Command Reference

### System Management

#### `system status`
Display comprehensive system status and health information.

```bash
# Basic status
python safla/cli_main.py system status

# Detailed status with performance metrics
python safla/cli_main.py system status --detailed

# JSON output for automation
python safla/cli_main.py system status --format json

# YAML output
python safla/cli_main.py system status --format yaml
```

**Example Output:**
```
SAFLA System Status
┏━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┓
┃ Component ┃ Status  ┃ Health   ┃ Uptime  ┃ CPU %  ┃ Memory MB  ┃
┡━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━┩
│ Memory    │ running │ healthy  │ 2h 30m  │ 15.2   │ 245.6      │
│ Cognition │ running │ healthy  │ 2h 30m  │ 22.4   │ 512.8      │
│ Safety    │ running │ healthy  │ 2h 30m  │ 8.7    │ 128.3      │
│ Mcp       │ running │ healthy  │ 2h 30m  │ 12.5   │ 189.2      │
│ Agents    │ running │ healthy  │ 2h 30m  │ 18.9   │ 367.4      │
└───────────┴─────────┴──────────┴─────────┴────────┴────────────┘

Overall Status: System Health: Healthy
```

#### `system start`
Start SAFLA system or specific components.

```bash
# Start entire system
python safla/cli_main.py system start

# Start specific component
python safla/cli_main.py system start --component memory

# Start as daemon (background process)
python safla/cli_main.py system start --daemon
```

#### `system stop`
Stop SAFLA system or specific components.

```bash
# Stop entire system
python safla/cli_main.py system stop

# Stop specific component
python safla/cli_main.py system stop --component mcp

# Force stop (immediate)
python safla/cli_main.py system stop --force
```

#### `system restart`
Restart system components.

```bash
# Restart entire system
python safla/cli_main.py system restart

# Restart specific component
python safla/cli_main.py system restart --component cognition
```

#### `system validate`
Comprehensive system validation and health checks.

```bash
# Basic validation
python safla/cli_main.py system validate

# Save validation report
python safla/cli_main.py system validate --output validation_report.json
```

### Configuration Management

#### `config show`
Display current configuration.

```bash
# Show all configuration
python safla/cli_main.py config show

# Show specific key
python safla/cli_main.py config show --key SAFLA_DEBUG

# Different output formats
python safla/cli_main.py config show --format json
python safla/cli_main.py config show --format yaml
python safla/cli_main.py config show --format env
```

#### `config set`
Set configuration values.

```bash
# Set configuration value
python safla/cli_main.py config set SAFLA_DEBUG true

# Set and save to file
python safla/cli_main.py config set SAFLA_WORKER_THREADS 8 --persistent

# Set nested configuration
python safla/cli_main.py config set memory.vector_dim 768
```

#### `config edit`
Edit configuration in your preferred editor.

```bash
# Edit with default editor
python safla/cli_main.py config edit

# Set custom editor
EDITOR=vim python safla/cli_main.py config edit
```

#### `config backup`
Backup current configuration.

```bash
# Backup to default location
python safla/cli_main.py config backup

# Backup to custom directory
python safla/cli_main.py config backup --backup-dir /path/to/backups
```

#### `config restore`
Restore configuration from backup.

```bash
# Restore from backup file
python safla/cli_main.py config restore backup_20240101_120000.json
```

### Monitoring & Metrics

#### `monitor live`
Live monitoring dashboard with real-time updates.

```bash
# Live monitoring with 5-second refresh
python safla/cli_main.py monitor live

# Custom refresh interval
python safla/cli_main.py monitor live --refresh 10

# Monitor for specific duration
python safla/cli_main.py monitor live --duration 300
```

#### `monitor logs`
View system logs.

```bash
# View recent logs
python safla/cli_main.py monitor logs

# Follow logs in real-time
python safla/cli_main.py monitor logs --follow

# Show specific number of lines
python safla/cli_main.py monitor logs --lines 100

# Monitor specific component
python safla/cli_main.py monitor logs --component mcp
```

#### `monitor metrics`
Display system performance metrics.

```bash
# Basic metrics
python safla/cli_main.py monitor metrics

# Detailed metrics
python safla/cli_main.py monitor metrics --detailed

# JSON output for monitoring tools
python safla/cli_main.py monitor metrics --format json
```

#### `monitor performance`
Real-time performance monitoring.

```bash
# Monitor overall performance
python safla/cli_main.py monitor performance

# Monitor specific component
python safla/cli_main.py monitor performance --component memory

# Monitor for specific duration
python safla/cli_main.py monitor performance --duration 120
```

### Optimization

#### `optimize analyze`
Analyze system for optimization opportunities.

```bash
# Full system analysis
python safla/cli_main.py optimize analyze

# Auto-apply safe optimizations
python safla/cli_main.py optimize analyze --auto

# Analyze specific component
python safla/cli_main.py optimize analyze --component memory
```

#### `optimize apply`
Apply system optimizations.

```bash
# Apply general optimizations
python safla/cli_main.py optimize apply

# Target specific optimization area
python safla/cli_main.py optimize apply --target memory

# Aggressive optimization mode
python safla/cli_main.py optimize apply --aggressive
```

#### `optimize memory`
Memory-specific optimizations.

```bash
# Optimize memory usage
python safla/cli_main.py optimize memory
```

#### `optimize cache`
Cache performance optimizations.

```bash
# Optimize cache performance
python safla/cli_main.py optimize cache
```

### Benchmarking

#### `benchmark run`
Run performance benchmarks.

```bash
# Quick benchmark suite
python safla/cli_main.py benchmark run --suite quick

# Standard benchmark suite
python safla/cli_main.py benchmark run --suite standard

# Comprehensive benchmark suite
python safla/cli_main.py benchmark run --suite comprehensive

# Save results to file
python safla/cli_main.py benchmark run --output benchmark_results.json

# Compare with previous results
python safla/cli_main.py benchmark run --compare previous_results.json
```

#### `benchmark component`
Benchmark specific components.

```bash
# Benchmark memory system
python safla/cli_main.py benchmark component --component memory

# Custom iteration count
python safla/cli_main.py benchmark component --component cognition --iterations 1000
```

#### `benchmark stress`
Run stress tests.

```bash
# Standard stress test
python safla/cli_main.py benchmark stress

# Custom duration and load
python safla/cli_main.py benchmark stress --duration 600 --load-level 0.9
```

### Agent Management

#### `agents list`
List deployed agents.

```bash
# List all agents
python safla/cli_main.py agents list

# JSON output
python safla/cli_main.py agents list --format json
```

#### `agents deploy`
Deploy new agents.

```bash
# Deploy agent with defaults
python safla/cli_main.py agents deploy my-agent

# Deploy with configuration file
python safla/cli_main.py agents deploy my-agent --config-file agent_config.json

# Deploy with multiple replicas
python safla/cli_main.py agents deploy my-agent --replicas 3

# Deploy with resource requirements
python safla/cli_main.py agents deploy my-agent --resources '{"cpu": 2, "memory": "4Gi"}'
```

#### `agents scale`
Scale agent deployments.

```bash
# Scale replicas
python safla/cli_main.py agents scale my-agent --replicas 5

# Update resource requirements
python safla/cli_main.py agents scale my-agent --resources '{"cpu": 4, "memory": "8Gi"}'
```

#### `agents remove`
Remove deployed agents.

```bash
# Remove agent (with confirmation)
python safla/cli_main.py agents remove my-agent

# Force removal
python safla/cli_main.py agents remove my-agent --force
```

#### `agents logs`
View agent logs.

```bash
# View agent logs
python safla/cli_main.py agents logs my-agent

# Follow logs
python safla/cli_main.py agents logs my-agent --follow

# Show specific number of lines
python safla/cli_main.py agents logs my-agent --lines 200
```

### Interactive Features

#### `dashboard`
Launch interactive dashboard.

```bash
# Launch full-featured dashboard (requires textual)
python safla/cli_main.py dashboard

# If textual is not available, falls back to simple dashboard
```

**Dashboard Features:**
- Real-time system monitoring
- Interactive component management  
- Live performance metrics
- Agent deployment controls
- System logs viewing
- One-click system operations

#### `setup`
Interactive setup wizard.

```bash
# Run setup wizard
python safla/cli_main.py setup
```

The setup wizard guides you through:
- System configuration
- Performance settings
- Memory configuration
- Safety settings
- MCP server setup
- Security configuration
- Monitoring setup

### Utility Commands

#### `doctor`
Comprehensive system health check and diagnostics.

```bash
# Run full system diagnostics
python safla/cli_main.py doctor
```

Performs:
- System resource checks
- Component health validation
- Configuration verification
- Network connectivity tests
- Storage health checks
- Performance baseline tests
- Security audit

#### `version`
Show version information.

```bash
# Show version info
python safla/cli_main.py version

# JSON format
python safla/cli_main.py version --format json
```

#### `search`
Search commands and documentation.

```bash
# Search for commands
python safla/cli_main.py search memory

# Interactive search
python safla/cli_main.py search
```

#### `help-menu`
Show comprehensive help menu.

```bash
# Show full help menu
python safla/cli_main.py help-menu
```

## Advanced Usage Patterns

### Automation & Scripting

#### Health Check Script
```bash
#!/bin/bash
# health_check.sh

# Check system health
STATUS=$(python safla/cli_main.py system status --format json | jq -r '.health')

if [ "$STATUS" != "healthy" ]; then
    echo "System unhealthy, attempting restart..."
    python safla/cli_main.py system restart
    
    # Wait and check again
    sleep 30
    NEW_STATUS=$(python safla/cli_main.py system status --format json | jq -r '.health')
    
    if [ "$NEW_STATUS" != "healthy" ]; then
        echo "System still unhealthy after restart. Manual intervention required."
        exit 1
    fi
fi

echo "System healthy"
```

#### Performance Monitoring Script
```bash
#!/bin/bash
# monitor_performance.sh

while true; do
    # Collect metrics
    python safla/cli_main.py monitor metrics --format json > metrics_$(date +%s).json
    
    # Check for performance issues
    CPU_USAGE=$(python safla/cli_main.py monitor metrics --format json | jq -r '.system.cpu_percent')
    
    if (( $(echo "$CPU_USAGE > 90" | bc -l) )); then
        echo "High CPU usage detected: $CPU_USAGE%"
        python safla/cli_main.py optimize analyze --auto
    fi
    
    sleep 60
done
```

#### Backup Script
```bash
#!/bin/bash
# backup_safla.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/safla_$DATE"

mkdir -p "$BACKUP_DIR"

# Backup configuration
python safla/cli_main.py config backup --backup-dir "$BACKUP_DIR"

# Export system status
python safla/cli_main.py system status --format json > "$BACKUP_DIR/system_status.json"

# Export metrics
python safla/cli_main.py monitor metrics --format json > "$BACKUP_DIR/metrics.json"

echo "Backup completed: $BACKUP_DIR"
```

### Monitoring Integration

#### Prometheus Metrics Export
```bash
# Export metrics for Prometheus
python safla/cli_main.py monitor metrics --format json | \
    jq -r '.system | to_entries[] | "safla_\(.key) \(.value)"' > metrics.prom
```

#### Grafana Dashboard Query
```bash
# Query for Grafana dashboard
while true; do
    python safla/cli_main.py monitor metrics --format json | \
        curl -X POST -H "Content-Type: application/json" \
        -d @- http://grafana-server:3000/api/metrics
    sleep 30
done
```

### Multi-Environment Management

#### Development Environment
```bash
# Development setup
python safla/cli_main.py --config dev.env setup
python safla/cli_main.py --debug system start
```

#### Production Environment
```bash
# Production setup
python safla/cli_main.py --config prod.env --quiet system start --daemon
python safla/cli_main.py monitor live --refresh 60 > monitoring.log 2>&1 &
```

#### Testing Environment
```bash
# Testing setup
python safla/cli_main.py --config test.env system start
python safla/cli_main.py benchmark run --suite comprehensive
python safla/cli_main.py system stop
```

## Configuration Management

### Environment Variables

All configuration can be managed through environment variables:

```bash
# Basic settings
export SAFLA_DEBUG=true
export SAFLA_LOG_LEVEL=DEBUG
export SAFLA_WORKER_THREADS=8

# Performance settings
export SAFLA_MEMORY_POOL_SIZE=1024
export SAFLA_CACHE_SIZE=512
export SAFLA_ENABLE_OPTIMIZATIONS=true

# Memory settings
export SAFLA_VECTOR_DIMENSIONS=512,768,1024,1536
export SAFLA_MAX_MEMORIES=20000
export SAFLA_SIMILARITY_THRESHOLD=0.85

# Safety settings
export SAFLA_MEMORY_LIMIT=2000000000
export SAFLA_CPU_LIMIT=0.95
export SAFLA_ROLLBACK_ENABLED=true

# MCP settings
export SAFLA_HOST=localhost
export SAFLA_PORT=8000
export SAFLA_MCP_TIMEOUT=60

# Security settings
export SAFLA_ENABLE_RATE_LIMITING=true
export SAFLA_API_RATE_LIMIT=2000
export JWT_SECRET_KEY=your-secret-key

# Monitoring settings
export SAFLA_ENABLE_MONITORING=true
export SAFLA_ENABLE_METRICS=true
export SAFLA_METRICS_INTERVAL=30
```

### Configuration Files

#### Development Configuration (`dev.env`)
```bash
SAFLA_DEBUG=true
SAFLA_LOG_LEVEL=DEBUG
SAFLA_WORKER_THREADS=4
SAFLA_MEMORY_POOL_SIZE=256
SAFLA_CACHE_SIZE=128
SAFLA_MAX_MEMORIES=5000
SAFLA_ENABLE_MONITORING=true
SAFLA_DEV_MODE=true
SAFLA_HOT_RELOAD=true
```

#### Production Configuration (`prod.env`)
```bash
SAFLA_DEBUG=false
SAFLA_LOG_LEVEL=INFO
SAFLA_WORKER_THREADS=16
SAFLA_MEMORY_POOL_SIZE=2048
SAFLA_CACHE_SIZE=1024
SAFLA_MAX_MEMORIES=50000
SAFLA_ENABLE_OPTIMIZATIONS=true
SAFLA_ENABLE_MONITORING=true
SAFLA_ENABLE_METRICS=true
SAFLA_ENABLE_SSL=true
```

#### Testing Configuration (`test.env`)
```bash
SAFLA_DEBUG=false
SAFLA_LOG_LEVEL=WARNING
SAFLA_WORKER_THREADS=2
SAFLA_MEMORY_POOL_SIZE=128
SAFLA_CACHE_SIZE=64
SAFLA_MAX_MEMORIES=1000
SAFLA_TEST_MODE=true
```

## Troubleshooting

### Common Issues

#### System Won't Start
```bash
# Check system validation
python safla/cli_main.py system validate

# Check configuration
python safla/cli_main.py config show

# Try individual components
python safla/cli_main.py system start --component memory
python safla/cli_main.py system start --component safety
```

#### Performance Issues
```bash
# Run optimization analysis
python safla/cli_main.py optimize analyze

# Check resource usage
python safla/cli_main.py monitor metrics --detailed

# Run performance monitoring
python safla/cli_main.py monitor performance --duration 300
```

#### Configuration Problems
```bash
# Validate configuration
python safla/cli_main.py doctor

# Reset to defaults
python safla/cli_main.py config restore default_config.json

# Interactive configuration
python safla/cli_main.py setup
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Debug mode for all commands
python safla/cli_main.py --debug system start

# Set debug in configuration
python safla/cli_main.py config set SAFLA_DEBUG true --persistent
```

### Log Analysis

```bash
# View recent logs
python safla/cli_main.py monitor logs --lines 1000

# Filter logs by component
python safla/cli_main.py monitor logs --component mcp --follow

# Export logs for analysis
python safla/cli_main.py monitor logs --lines 10000 > system_logs.txt
```

## Best Practices

### Regular Maintenance

```bash
# Daily health check
python safla/cli_main.py doctor

# Weekly optimization
python safla/cli_main.py optimize analyze --auto

# Monthly benchmarking
python safla/cli_main.py benchmark run --suite comprehensive

# Regular backups
python safla/cli_main.py config backup
```

### Performance Monitoring

```bash
# Set up continuous monitoring
python safla/cli_main.py monitor live --refresh 10 > monitoring.log 2>&1 &

# Regular performance checks
*/15 * * * * python safla/cli_main.py monitor metrics --format json >> metrics.log
```

### Security

```bash
# Regular security checks
python safla/cli_main.py doctor | grep -i security

# Ensure secure configuration
python safla/cli_main.py config show | grep -E "(SECRET|PASSWORD|TOKEN)"
```

## Integration Examples

### Docker Integration
```dockerfile
FROM python:3.11

COPY . /app
WORKDIR /app

# Install dependencies
RUN pip install -r requirements.txt

# Setup SAFLA CLI
RUN chmod +x safla/cli_main.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python safla/cli_main.py system status --format json | jq -r '.health' | grep -q healthy

# Start command
CMD python safla/cli_main.py system start --daemon && \
    python safla/cli_main.py monitor live
```

### Kubernetes Integration
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: safla-config
data:
  .env: |
    SAFLA_DEBUG=false
    SAFLA_LOG_LEVEL=INFO
    SAFLA_ENABLE_MONITORING=true

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: safla
spec:
  template:
    spec:
      containers:
      - name: safla
        image: safla:latest
        command: ["python", "safla/cli_main.py", "system", "start", "--daemon"]
        livenessProbe:
          exec:
            command:
            - python
            - safla/cli_main.py
            - system
            - status
            - --format
            - json
          initialDelaySeconds: 30
          periodSeconds: 30
```

### CI/CD Integration
```yaml
# .github/workflows/safla.yml
name: SAFLA Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: pip install -r requirements.txt
    
    - name: Validate installation
      run: python safla/cli_main.py system validate
    
    - name: Run benchmarks
      run: python safla/cli_main.py benchmark run --suite quick
    
    - name: Check system health
      run: python safla/cli_main.py doctor
```

This comprehensive CLI system provides complete control over SAFLA with both command-line and interactive interfaces, making it suitable for development, testing, production, and automation scenarios.