# Common Issues & Troubleshooting

Comprehensive troubleshooting guide for Synaptic Neural Mesh. This guide covers common problems, their causes, and step-by-step solutions.

## 🔧 Quick Diagnostic Commands

Before diving into specific issues, run these diagnostic commands:

```bash
# Check overall system status
npx synaptic-mesh status --detailed

# Validate configuration
npx synaptic-mesh config validate

# Check network connectivity
npx synaptic-mesh peer ping --debug

# View recent logs
npx synaptic-mesh logs --tail 50
```

## 🚨 Installation & Setup Issues

### Issue: `npx synaptic-mesh` command not found

**Symptoms:**
- Command not found error
- Package not installing correctly

**Causes:**
- NPM installation failure
- Missing Node.js or incompatible version
- PATH configuration issues

**Solutions:**

1. **Check Node.js version:**
```bash
node --version  # Should be 18.0.0 or higher
npm --version   # Should be 8.0.0 or higher
```

2. **Reinstall globally:**
```bash
npm uninstall -g synaptic-mesh
npm install -g synaptic-mesh@alpha
```

3. **Use npx directly:**
```bash
npx --yes synaptic-mesh@alpha init --force
```

4. **Check npm configuration:**
```bash
npm config get prefix
npm config get registry
```

### Issue: Permission denied during installation

**Symptoms:**
- EACCES errors during npm install
- Cannot create directories

**Solutions:**

1. **Use npx (recommended):**
```bash
npx synaptic-mesh@alpha init
```

2. **Fix npm permissions:**
```bash
# On macOS/Linux
sudo chown -R $(whoami) ~/.npm
sudo chown -R $(whoami) /usr/local/lib/node_modules

# On Windows (as Administrator)
npm config set prefix %APPDATA%/npm
```

3. **Use Node version manager:**
```bash
# Install nvm and use it
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18
```

### Issue: WASM modules failed to load

**Symptoms:**
- Error loading neural network modules
- "WebAssembly module compilation failed"

**Causes:**
- Unsupported architecture
- Missing WASM support
- Corrupted module files

**Solutions:**

1. **Check WASM support:**
```bash
node -e "console.log(typeof WebAssembly !== 'undefined')"  # Should output 'true'
```

2. **Reinstall with fresh modules:**
```bash
npx synaptic-mesh init --force --template minimal
```

3. **Manual WASM verification:**
```bash
cd .synaptic/wasm
ls -la *.wasm
file *.wasm  # Should show WebAssembly binary
```

## 🌐 Network & Connectivity Issues

### Issue: Cannot start P2P networking

**Symptoms:**
- "Port already in use" error
- "Cannot bind to address" error
- Network startup timeout

**Solutions:**

1. **Check port availability:**
```bash
# Check if port is in use
netstat -tulpn | grep 8080
lsof -i :8080

# Use different port
npx synaptic-mesh start --port 8081
```

2. **Check firewall settings:**
```bash
# Linux (ufw)
sudo ufw allow 8080
sudo ufw status

# macOS
sudo pfctl -f /etc/pf.conf

# Windows
netsh advfirewall firewall add rule name="Synaptic Mesh" dir=in action=allow protocol=TCP localport=8080
```

3. **Network interface binding:**
```bash
# Bind to all interfaces
npx synaptic-mesh start --bind 0.0.0.0 --port 8080

# Bind to specific interface
npx synaptic-mesh start --bind 192.168.1.100 --port 8080
```

### Issue: Cannot connect to peers

**Symptoms:**
- "Connection timeout" errors
- Zero connected peers
- "Peer unreachable" messages

**Debugging Steps:**

1. **Check peer address format:**
```bash
# Correct format
/ip4/192.168.1.100/tcp/8080/p2p/12D3KooWAbc123...

# Test connection
npx synaptic-mesh peer connect "/ip4/192.168.1.100/tcp/8080/p2p/12D3KooW..." --timeout 60
```

2. **Network connectivity test:**
```bash
# Test basic connectivity
ping 192.168.1.100
telnet 192.168.1.100 8080

# Test from mesh
npx synaptic-mesh peer ping --target 192.168.1.100:8080
```

3. **NAT traversal issues:**
```bash
# Enable UPnP
npx synaptic-mesh start --upnp

# Use STUN servers
npx synaptic-mesh start --stun stun.l.google.com:19302

# Manual port forwarding required for some routers
```

### Issue: High latency or slow performance

**Symptoms:**
- High response times (>500ms)
- Slow consensus
- Poor inference performance

**Solutions:**

1. **Network optimization:**
```bash
# Optimize topology
npx synaptic-mesh mesh topology optimize --strategy performance

# Reduce max peers
npx synaptic-mesh config set network.maxPeers 20

# Enable connection pooling
npx synaptic-mesh config set network.connectionPool true
```

2. **Performance monitoring:**
```bash
# Real-time monitoring
npx synaptic-mesh status --watch --refresh 2

# Network latency test
npx synaptic-mesh peer ping --all --detailed
```

## 🧠 Neural Network Issues

### Issue: Neural agents fail to spawn

**Symptoms:**
- "Insufficient memory" errors
- "WASM module failed to initialize"
- Spawn timeout errors

**Solutions:**

1. **Check memory availability:**
```bash
# Check system memory
free -h  # Linux
vm_stat  # macOS

# Check mesh memory usage
npx synaptic-mesh status --detailed | grep -i memory
```

2. **Adjust memory limits:**
```bash
# Spawn with smaller memory limit
npx synaptic-mesh neural spawn --type mlp --memory 32MB

# Configure global memory limit
npx synaptic-mesh config set neural.memoryLimit "128MB"
```

3. **Clean up zombie agents:**
```bash
# List all agents
npx synaptic-mesh neural list --detailed

# Kill inactive agents
npx synaptic-mesh neural kill --filter status=idle --force
```

### Issue: Poor inference performance

**Symptoms:**
- Inference time >100ms
- High CPU usage
- Memory leaks

**Solutions:**

1. **Optimize neural architecture:**
```bash
# Use optimized SIMD version
npx synaptic-mesh neural spawn --type mlp --simd true

# Reduce model size
npx synaptic-mesh neural spawn --type mlp --layers [784,64,10]
```

2. **Performance profiling:**
```bash
# Enable performance monitoring
npx synaptic-mesh start --metrics --profile

# View performance metrics
curl http://localhost:9090/metrics | grep neural
```

### Issue: Training fails or stalls

**Symptoms:**
- Training progress stuck at 0%
- "Training coordinator disconnected"
- Loss not decreasing

**Solutions:**

1. **Check training configuration:**
```bash
# Validate dataset
npx synaptic-mesh neural validate-dataset --path ./data/training.json

# Check training status
npx synaptic-mesh neural training status --id training_xyz
```

2. **Restart training with debugging:**
```bash
# Start with debug mode
npx synaptic-mesh neural train --debug --dataset ./data/ --epochs 10
```

## 📊 DAG & Consensus Issues

### Issue: Consensus not reaching finality

**Symptoms:**
- Transactions stuck in pending state
- "Consensus timeout" errors
- Network height not advancing

**Solutions:**

1. **Check validator status:**
```bash
# View consensus status
npx synaptic-mesh dag status --validators

# Check network synchronization
npx synaptic-mesh dag sync-status
```

2. **Restart consensus:**
```bash
# Force consensus restart
npx synaptic-mesh dag consensus restart

# Rejoin consensus network
npx synaptic-mesh dag consensus join --bootstrap
```

### Issue: High transaction fees or delays

**Symptoms:**
- Transactions taking too long to confirm
- High fee requirements

**Solutions:**

1. **Optimize transaction submission:**
```bash
# Submit with higher priority
npx synaptic-mesh dag submit "data" --priority high --fee 2000

# Batch multiple transactions
npx synaptic-mesh dag batch-submit --file transactions.json
```

## 🔧 Configuration Issues

### Issue: Configuration validation errors

**Symptoms:**
- "Invalid configuration" errors
- Mesh won't start due to config issues

**Solutions:**

1. **Validate and fix configuration:**
```bash
# Validate current config
npx synaptic-mesh config validate --verbose

# Reset to defaults
npx synaptic-mesh config reset --backup

# Load working configuration
npx synaptic-mesh config load --template enterprise
```

2. **Common configuration fixes:**
```bash
# Fix port conflicts
npx synaptic-mesh config set network.port 8081

# Fix memory limits
npx synaptic-mesh config set neural.memoryLimit "256MB"

# Fix network ID
npx synaptic-mesh config set network.networkId "mainnet"
```

## 🐳 Docker & Container Issues

### Issue: Docker container won't start

**Symptoms:**
- Container exits immediately
- "Permission denied" in container logs

**Solutions:**

1. **Check Docker configuration:**
```bash
# Pull latest image
docker pull synaptic-mesh:alpha

# Run with debug
docker run -it --rm synaptic-mesh:alpha --debug

# Check container logs
docker logs <container-id>
```

2. **Fix volume mounting:**
```bash
# Correct volume syntax
docker run -v ~/.synaptic:/app/.synaptic synaptic-mesh:alpha

# Fix permissions
sudo chown -R 1000:1000 ~/.synaptic
```

### Issue: Kubernetes deployment failures

**Symptoms:**
- Pods stuck in pending state
- "ImagePullBackOff" errors

**Solutions:**

1. **Check Kubernetes configuration:**
```bash
# Check pod status
kubectl get pods -n synaptic-mesh

# View pod logs
kubectl logs -n synaptic-mesh <pod-name>

# Describe pod for events
kubectl describe pod -n synaptic-mesh <pod-name>
```

2. **Common K8s fixes:**
```yaml
# Fix resource limits
resources:
  limits:
    memory: "512Mi"
    cpu: "500m"
  requests:
    memory: "256Mi"
    cpu: "250m"
```

## 🔍 Debugging Tools & Techniques

### Enable Debug Mode

```bash
# Start with full debugging
npx synaptic-mesh start --debug --log-level debug --profile

# Debug specific components
export SYNAPTIC_DEBUG="neural,network,dag"
npx synaptic-mesh start
```

### Log Analysis

```bash
# View real-time logs
npx synaptic-mesh logs --follow --level error

# Search logs for errors
npx synaptic-mesh logs --grep "error" --since "1h"

# Export logs for analysis
npx synaptic-mesh logs --export --format json > debug.log
```

### Network Diagnostics

```bash
# Test network stack
npx synaptic-mesh network diagnose

# Trace peer connections
npx synaptic-mesh peer trace --target <peer-id>

# Bandwidth test
npx synaptic-mesh network speed-test --peers 5
```

### Performance Profiling

```bash
# CPU profiling
npx synaptic-mesh profile --type cpu --duration 60s

# Memory profiling
npx synaptic-mesh profile --type memory --output memory.prof

# Network profiling
npx synaptic-mesh profile --type network --detailed
```

## 🚨 Emergency Recovery

### Complete System Reset

```bash
# Stop all processes
npx synaptic-mesh stop --force

# Backup current state
cp -r ~/.synaptic ~/.synaptic.backup

# Reset to clean state
rm -rf ~/.synaptic
npx synaptic-mesh init --force

# Restore data if needed
cp ~/.synaptic.backup/data/* ~/.synaptic/data/
```

### Network Partition Recovery

```bash
# Check partition status
npx synaptic-mesh dag status --partition-check

# Force rejoin network
npx synaptic-mesh mesh rejoin --bootstrap <bootstrap-address>

# Sync from trusted peer
npx synaptic-mesh dag sync --peer <trusted-peer-id>
```

### Data Corruption Recovery

```bash
# Check data integrity
npx synaptic-mesh data verify --all

# Repair corrupted data
npx synaptic-mesh data repair --backup

# Restore from backup
npx synaptic-mesh data restore --from backup.tar.gz
```

## 📞 Getting Help

### Community Support

- **GitHub Issues**: [Report bugs and issues](https://github.com/ruvnet/Synaptic-Neural-Mesh/issues)
- **Discussions**: [Community help and questions](https://github.com/ruvnet/Synaptic-Neural-Mesh/discussions)
- **Discord**: [Real-time community support](https://discord.gg/synaptic-mesh)

### Diagnostic Information

When reporting issues, include:

```bash
# Generate diagnostic report
npx synaptic-mesh diagnose --full --output diagnostic.json

# System information
npx synaptic-mesh info --system --network --config
```

### Log Collection

```bash
# Collect all relevant logs
npx synaptic-mesh collect-logs --since "24h" --output logs.tar.gz

# Include configuration (remove sensitive data)
npx synaptic-mesh config export --sanitized >> diagnostic.json
```

---

This troubleshooting guide covers the most common issues. For specific problems not covered here, check our [FAQ](faq.md) or reach out to the community for help.

**Next**: [FAQ](faq.md) for frequently asked questions and quick answers.