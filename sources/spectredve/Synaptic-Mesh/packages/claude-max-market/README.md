# Claude Max Market - NPX Wrapper

NPX wrapper for Docker orchestration and market integration with comprehensive compliance features for Claude Max capacity sharing.

## 🔐 Compliance Features

This implementation is designed to be **fully compliant** with Anthropic's Terms of Service by ensuring:

- ✅ **No shared API keys** - Each user maintains their own Claude credentials
- ✅ **Peer orchestrated model** - Tasks route to willing participants, not centralized accounts  
- ✅ **Voluntary participation** - Users opt-in explicitly with granular controls
- ✅ **User control & transparency** - Full approval and audit mechanisms
- ✅ **Token incentives** - RUV tokens reward contribution, not access purchase

## 📦 Installation

```bash
# Install globally via NPX
npx @synaptic-neural-mesh/claude-max-market --help

# Or install locally
npm install -g @synaptic-neural-mesh/claude-max-market
```

## 🚀 Quick Start

### 1. Review Legal Terms

```bash
npx claude-max-market terms
```

### 2. Opt-in with User Consent

```bash
npx claude-max-market opt-in --claude-jobs --max-daily 5 --max-tokens 1000
```

### 3. Build/Pull Docker Image

```bash
npx claude-max-market docker:build
# or
npx claude-max-market docker:pull
```

### 4. Execute Claude Job

```bash
npx claude-max-market execute --prompt "Hello Claude" --model claude-3-sonnet-20240229
```

## 📋 Available Commands

### Legal & Compliance

```bash
# Display legal notice and usage policy
claude-max-market terms

# Run compliance verification
claude-max-market compliance-check

# Generate audit report
claude-max-market audit --format json --output audit.json
```

### User Consent & Control

```bash
# Opt into Claude job processing
claude-max-market opt-in [options]
  --claude-jobs          Allow Claude job processing
  --max-daily <n>        Maximum daily tasks (default: 5)
  --max-tokens <n>       Maximum tokens per task (default: 1000)

# Opt out completely
claude-max-market opt-out

# Configure usage limits
claude-max-market limits --daily 10 --tokens 2000 --timeout 600

# View current status
claude-max-market status
```

### Docker Orchestration

```bash
# Build Claude container image
claude-max-market docker:build [options]
  --tag <name>           Image tag (default: synaptic-mesh/claude-max)
  --no-cache             Build without cache

# Pull pre-built image
claude-max-market docker:pull [options]
  --tag <name>           Image tag

# System health check
claude-max-market health

# Clean up containers
claude-max-market clean [--force]
```

### Job Execution

```bash
# Execute Claude job with approval
claude-max-market execute [options]
  --prompt <text>        Claude prompt
  --file <path>          Input file path
  --model <name>         Claude model (default: claude-3-sonnet-20240229)
  --max-tokens <n>       Maximum tokens (default: 1000)
  --approve-all          Auto-approve (not recommended)

# View execution logs
claude-max-market logs [options]
  --follow               Follow log output
  --tail <n>             Number of lines (default: 50)
```

### Market Integration

```bash
# Advertise available capacity
claude-max-market advertise [options]
  --slots <n>            Available execution slots (default: 1)
  --price <n>            Price per task in RUV tokens (default: 5)

# Place bid for task execution
claude-max-market bid [options]
  --task-id <id>         Task ID to bid on
  --max-price <n>        Maximum price to pay
```

### Security & Encryption

```bash
# Encrypt job payload
claude-max-market encrypt --input <file> --output <file>

# Decrypt job payload
claude-max-market decrypt --input <file> --output <file>
```

### Configuration

```bash
# Manage configuration
claude-max-market config [options]
  --set <key=value>      Set configuration value
  --get <key>            Get configuration value
  --list                 List all configuration
```

## 🔧 Configuration

The system uses a comprehensive configuration system with these main sections:

### Docker Configuration
```json
{
  "docker": {
    "image": "synaptic-mesh/claude-max:latest",
    "memory": "512m",
    "cpuShares": 512,
    "timeout": 300000
  }
}
```

### Security Settings
```json
{
  "security": {
    "enableEncryption": true,
    "keyRotationDays": 30,
    "maxSessionAge": 3600000
  }
}
```

### Usage Limits
```json
{
  "limits": {
    "dailyTasks": 5,
    "dailyTokens": 5000,
    "maxTokensPerTask": 1000,
    "concurrentJobs": 1
  }
}
```

### Compliance Settings
```json
{
  "compliance": {
    "requireApproval": true,
    "logAllActivity": true,
    "enableAuditTrail": true,
    "checkInterval": 3600000
  }
}
```

## 🛡️ Security Features

### Container Isolation
- Read-only filesystem with tmpfs workspace
- Network isolation (API access only)
- Resource limits (512MB RAM, limited CPU)
- Non-root user execution
- No persistent secrets storage

### Encryption
- End-to-end payload encryption
- RSA + AES hybrid encryption for peer-to-peer
- Secure key generation and rotation
- No plaintext transmission

### Access Control
- User approval required for each job (configurable)
- Granular usage limits and controls
- Comprehensive audit logging
- Immediate opt-out capability

## 📊 Usage Tracking

The system provides detailed usage analytics:

```bash
# View current usage status
claude-max-market status

# Generate analytics report
node -e "
const { UsageTracker } = require('./src/tracking/usageTracker.js');
const tracker = new UsageTracker();
tracker.generateAnalytics().then(console.log);
"
```

### Tracked Metrics
- Daily tasks executed
- Token consumption
- Execution times
- Success/failure rates
- Model usage distribution
- Compliance adherence

## 🏥 Health Monitoring

```bash
# System health check
claude-max-market health

# Continuous monitoring (example)
while true; do
  claude-max-market health
  sleep 60
done
```

## 📝 Audit & Compliance

### Audit Trail
All activities are logged for compliance:
- User consent and opt-in events
- Job executions and approvals
- Configuration changes
- System access and errors

### Compliance Reports
```bash
# Generate compliance report
claude-max-market audit --format text

# Export detailed JSON report
claude-max-market audit --format json --output compliance-report.json
```

## 🐳 Docker Usage

### Build Custom Image
```dockerfile
FROM synaptic-mesh/claude-max:latest

# Add custom configurations
COPY custom-config.json /app/

# Set environment
ENV CLAUDE_API_KEY=""
ENV MAX_DAILY_TASKS=10
```

### Run Container
```bash
# Run with environment variables
docker run -e CLAUDE_API_KEY="your-key" \
           -e MAX_DAILY_TASKS=5 \
           --read-only \
           --tmpfs /tmp \
           synaptic-mesh/claude-max execute --prompt "Hello"

# Run with mounted config
docker run -v $(pwd)/config:/app/config:ro \
           synaptic-mesh/claude-max
```

## 🌐 Market Integration

### Capacity Advertising
```bash
# Advertise 3 slots at 5 RUV each
claude-max-market advertise --slots 3 --price 5
```

### Bidding System
```bash
# Bid up to 10 RUV for a specific task
claude-max-market bid --task-id abc123 --max-price 10
```

### Market Statistics
```javascript
const { MarketIntegration } = require('./src/market/integration.js');
const market = new MarketIntegration();

// View market stats
market.getMarketStats().then(stats => {
  console.log('Active offers:', stats.activeOffers);
  console.log('Average price:', stats.averagePrice);
  console.log('Node reputation:', stats.nodeReputation);
});
```

## 🔌 API Integration

### Programmatic Usage
```javascript
import { ClaudeMaxMarket } from '@synaptic-neural-mesh/claude-max-market';

const market = new ClaudeMaxMarket();

// Initialize with user consent
await market.setupOptIn({
  maxDaily: 5,
  maxTokens: 1000
});

// Execute job
const result = await market.executeJob({
  prompt: 'Explain quantum computing',
  model: 'claude-3-sonnet-20240229',
  maxTokens: 500
});

console.log('Result:', result.response);
console.log('Tokens used:', result.usage.totalTokens);
```

## ⚠️ Important Legal Notice

**COMPLIANCE DISCLAIMER**: This software facilitates voluntary peer-to-peer coordination between users who already have their own individual Claude Max subscriptions. It does NOT:

- Share, proxy, or resell access to Claude Max
- Transmit API keys or credentials between users
- Provide centralized Claude access
- Violate Anthropic's Terms of Service

Each user maintains full control of their own Claude account and can revoke participation at any time.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup
```bash
git clone https://github.com/ruvnet/synaptic-neural-mesh
cd packages/claude-max-market
npm install
npm run test
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/ruvnet/synaptic-neural-mesh/issues)
- **Documentation**: [Full Docs](https://synaptic-neural-mesh.org/docs)
- **Compliance**: [Legal Notice](https://synaptic-neural-mesh.org/legal)

## 🔗 Related Projects

- [Synaptic Neural Mesh](https://github.com/ruvnet/synaptic-neural-mesh) - Main repository
- [Claude Flow](https://github.com/ruvnet/claude-flow) - Claude orchestration framework
- [RUV Swarm](https://github.com/ruvnet/ruv-swarm) - Distributed computation platform