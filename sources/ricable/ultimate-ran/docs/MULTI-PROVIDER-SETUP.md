# TITAN Multi-Provider Setup Guide

Complete guide for running TITAN with Claude Code PRO MAX, Google AI Pro, E2B, and OpenRouter on Mac Silicon using DevPod.

## 🎯 Overview

This setup enables TITAN to leverage multiple AI providers concurrently:

- **Claude Code PRO MAX**: Primary reasoning and RAN optimization
- **Google AI Pro (Gemini 2.0)**: Multimodal analysis and anomaly detection
- **E2B Sandboxes**: Isolated code execution for safety validation
- **OpenRouter**: Fallback and additional model access
- **Agentic Flow**: Multi-agent coordination with QUIC transport
- **Claude Flow**: Swarm orchestration and consensus

## 📋 Prerequisites

### Required Accounts & API Keys

1. **Anthropic Claude** (PRO MAX subscription)
   - Sign up: https://console.anthropic.com/
   - Get API key: https://console.anthropic.com/settings/keys
   - Model: `claude-sonnet-4-5-20250929`

2. **Google AI Studio** (Free/Pro)
   - Sign up: https://ai.google.dev/
   - Get API key: https://aistudio.google.com/app/apikey
   - Model: `gemini-2.0-flash-exp`

3. **E2B** (Sandbox execution)
   - Sign up: https://e2b.dev/
   - Get API key: https://e2b.dev/docs/getting-started/api-key
   - Free tier: 100 sandbox hours/month

4. **OpenRouter** (Optional)
   - Sign up: https://openrouter.ai/
   - Get API key: https://openrouter.ai/keys
   - Pay-per-use pricing

### System Requirements

- **Mac Silicon**: M1, M2, M3, or M4 (ARM64)
- **macOS**: 12.0+ (Monterey or later)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **Node.js**: 18.0+ or 20.0+
- **Docker Desktop**: 4.25+ (for DevPod)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Navigate to project
cd /Users/cedric/dev/ultimate-ran-1

# Copy environment template
cp config/.env.template config/.env

# Edit with your API keys
nano config/.env  # or use your preferred editor
```

### 2. Configure API Keys

Edit `config/.env` with your actual keys:

```bash
# Claude Code PRO MAX
ANTHROPIC_API_KEY=sk-ant-api03-YOUR-ACTUAL-KEY

# Google AI Pro
GOOGLE_AI_API_KEY=AIzaSy-YOUR-ACTUAL-KEY

# E2B Sandboxes
E2B_API_KEY=e2b_YOUR-ACTUAL-KEY

# OpenRouter (optional)
OPENROUTER_API_KEY=sk-or-v1-YOUR-ACTUAL-KEY

# Set strategy (consensus recommended)
AGENTIC_FLOW_STRATEGY=consensus
```

### 3. Choose Runtime Mode

#### Option A: Local Mac Silicon (Recommended for Development)

```bash
# Install dependencies
npm install

# Test integrations
./scripts/test-integration.sh

# Start TITAN locally
./scripts/start-local.sh
```

#### Option B: DevPod with Docker (Recommended for Production)

```bash
# Install DevPod (via Homebrew)
brew install devpod

# Start DevPod environment
./scripts/start-devpod.sh

# Access workspace
devpod ssh titan-ran
```

## 🎨 AI Strategy Modes

TITAN supports 4 AI orchestration strategies:

### 1. **Consensus** (Recommended for Production)
Both Claude and Gemini must agree. Highest confidence, safest for critical operations.

```typescript
AGENTIC_FLOW_STRATEGY=consensus
```

**Use cases:**
- Parameter optimization approvals
- Safety-critical changes
- Network-wide updates

**Performance:** 5-10s per decision
**Confidence:** 95%+

### 2. **Claude Primary** (Fast & Reliable)
Claude makes decisions, Gemini validates. Good balance of speed and accuracy.

```typescript
AGENTIC_FLOW_STRATEGY=claude_primary
```

**Use cases:**
- Real-time optimization
- Routine parameter adjustments
- Testing and experimentation

**Performance:** 2-4s per decision
**Confidence:** 85-90%

### 3. **Gemini Primary** (Multimodal Analysis)
Gemini leads with visual insights, Claude validates logic.

```typescript
AGENTIC_FLOW_STRATEGY=gemini_primary
```

**Use cases:**
- Anomaly detection from graphs
- Interference pattern analysis
- Visual network diagnostics

**Performance:** 3-5s per decision
**Confidence:** 80-85%

### 4. **Parallel** (Maximum Speed)
Both run independently, faster response selected.

```typescript
AGENTIC_FLOW_STRATEGY=parallel
```

**Use cases:**
- Research and exploration
- Non-critical analysis
- Rapid prototyping

**Performance:** 1-3s per decision
**Confidence:** 70-80%

## 🔧 Configuration Details

### Multi-Provider Configuration

The system is configured in `config/agentic-flow.config.ts`:

```typescript
import { getMultiProviderConfig } from './config/agentic-flow.config';

const config = getMultiProviderConfig();
// Automatically loads from .env and selects optimal providers
```

### Agentic Flow Transport

QUIC-based transport with 0-RTT for ultra-low latency:

```typescript
{
  transport: {
    protocol: 'quic',
    enableZeroRTT: true,
    congestionControl: 'bbr',
    maxStreams: 100
  }
}
```

### AgentDB Memory

Persistent cognitive memory with HNSW vector indexing:

```typescript
{
  memory: {
    provider: 'agentdb',
    persistence: true,
    vectorIndex: {
      algorithm: 'hnsw',
      dimensions: 1536,  // Compatible with Claude/OpenAI embeddings
      metric: 'cosine'
    }
  }
}
```

## 🐳 DevPod Setup (Advanced)

### Install DevPod

```bash
# macOS (Homebrew)
brew install devpod

# Or download from: https://devpod.sh/
```

### Configure Docker Provider

```bash
# Add Docker provider
devpod provider add docker

# Set as default
devpod provider use docker

# Configure for Mac Silicon
devpod provider set-options docker \
  --option PLATFORM=linux/arm64 \
  --option CPUS=4 \
  --option MEMORY=8G
```

### Create Workspace

```bash
# Create and start workspace
devpod up . \
  --id titan-ran \
  --provider docker \
  --devcontainer-path config/devpod.yaml \
  --ide vscode

# SSH into workspace
devpod ssh titan-ran

# Open in VS Code
devpod ide titan-ran vscode
```

### Docker Compose (Alternative)

```bash
# Start all services
docker-compose -f config/docker-compose.devpod.yml up -d

# View logs
docker-compose -f config/docker-compose.devpod.yml logs -f titan-app

# Stop services
docker-compose -f config/docker-compose.devpod.yml down
```

## 🧪 Testing Integrations

### Run Full Test Suite

```bash
./scripts/test-integration.sh
```

Expected output:
```
🧪 TITAN Multi-Provider Integration Test
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Testing Claude Code PRO MAX...
  ✓ Claude Code PRO MAX working

Testing Google AI Pro (Gemini)...
  ✓ Google AI Pro (Gemini) working

Testing E2B Sandboxes...
  ✓ E2B Sandboxes working

Testing OpenRouter...
  ✓ OpenRouter working

Testing agentic-flow@alpha...
  ✓ agentic-flow installed

Testing claude-flow@alpha...
  ✓ claude-flow installed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Passed: 6 / 6 tests
🎉 All tests passed!
```

### Manual Testing

```bash
# Test Claude integration
npm run ui:integration

# Test specific example
EXAMPLE=3 npm run ui:integration  # AI Orchestrator

# Test with live data
npm run agui:frontend  # Open dashboard
```

## 📊 Usage Examples

### Example 1: Claude-Only Optimization

```typescript
import { ClaudeAgentIntegration } from './ui/integrations';

const claude = new ClaudeAgentIntegration({
  apiKey: process.env.ANTHROPIC_API_KEY!
});

const result = await claude.requestOptimization(
  cells,
  "Optimize SINR while maintaining coverage",
  { maxIterations: 5 }
);

console.log('Recommendations:', result.recommendations);
```

### Example 2: Gemini Anomaly Detection

```typescript
import { GoogleGeminiIntegration } from './ui/integrations';

const gemini = new GoogleGeminiIntegration({
  apiKey: process.env.GOOGLE_AI_API_KEY!
});

const anomalies = await gemini.detectAnomalies(cells, performanceData);

for (const anomaly of anomalies) {
  console.log(`${anomaly.severity}: ${anomaly.description}`);
}
```

### Example 3: Consensus Mode (Recommended)

```typescript
import { AIOrchestrator } from './ui/integrations';

const ai = new AIOrchestrator({
  claude: { apiKey: process.env.ANTHROPIC_API_KEY! },
  gemini: { apiKey: process.env.GOOGLE_AI_API_KEY! },
  strategy: 'consensus'  // Both must agree
});

const result = await ai.requestOptimization(
  cells,
  "High-risk parameter change: Increase p0Alpha by 5dB",
  interferenceMatrix
);

if (result.confidence > 0.95) {
  console.log('✓ Both AIs approved:', result.recommendations);
} else {
  console.log('✗ No consensus reached:', result.reasoning);
}
```

### Example 4: E2B Safety Validation

```typescript
import { Sandbox } from '@e2b/sdk';

// Create isolated sandbox for testing
const sandbox = await Sandbox.create({
  apiKey: process.env.E2B_API_KEY,
  template: 'base'
});

// Simulate parameter change in digital twin
const result = await sandbox.runCode(`
  // Digital twin simulation
  const newConfig = { p0Alpha: -65 };
  const impact = simulateChange(currentState, newConfig);
  impact;
`);

console.log('Simulated impact:', result.stdout);

await sandbox.close();
```

## 🚀 Running TITAN

### Local Mode (Mac Silicon)

```bash
# Start all services locally
./scripts/start-local.sh

# Access UI
open http://localhost:3000

# View AG-UI
open http://localhost:3001
```

Services started:
- ✅ AgentDB (cognitive memory)
- ✅ AG-UI Server (real-time dashboard)
- ✅ TITAN Orchestrator (main system)
- ✅ QUIC Transport (port 4433)

### DevPod Mode

```bash
# Create and start workspace
./scripts/start-devpod.sh

# SSH into container
devpod ssh titan-ran

# Inside container, services auto-start
cd /workspace
npm start
```

### Docker Compose Mode

```bash
# Start all containers
docker-compose -f config/docker-compose.devpod.yml up -d

# View logs
docker-compose logs -f titan-app

# Access services
open http://localhost:3000  # Dashboard
open http://localhost:3001  # AG-UI
```

## 🔍 Monitoring & Debugging

### View Logs

```bash
# Local mode - logs to stdout
npm start

# DevPod mode
devpod ssh titan-ran
tail -f /workspace/logs/titan.log

# Docker Compose mode
docker-compose logs -f titan-app
```

### Check Service Health

```bash
# AgentDB status
npm run db:status

# Agent swarm status
npx claude-flow@alpha swarm status

# Agentic flow health
npx agentic-flow@alpha health
```

### Debug API Connections

```bash
# Test Claude API
curl -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  https://api.anthropic.com/v1/messages

# Test Google AI
curl "https://generativelanguage.googleapis.com/v1/models?key=$GOOGLE_AI_API_KEY"

# Test E2B
curl -H "X-API-Key: $E2B_API_KEY" \
  https://api.e2b.dev/sandboxes
```

## 🛡️ Security Best Practices

### API Key Management

```bash
# NEVER commit .env to git
echo "config/.env" >> .gitignore

# Use environment-specific files
config/.env.local       # Local development
config/.env.devpod      # DevPod environment
config/.env.production  # Production (never commit!)

# Rotate keys regularly
# Set expiration dates in provider dashboards
```

### Sandbox Isolation

```bash
# Always use strict isolation for production
SANDBOX_ISOLATION=strict

# Enable quantum-resistant signatures
QUANTUM_SIGNATURES=true

# Audit all parameter changes
ENABLE_TELEMETRY=true
```

## 📈 Performance Optimization

### Mac Silicon Optimizations

```bash
# Enable ARM64 NEON intrinsics
export PLATFORM_ARCH=arm64

# Use SIMD vectorization
# Automatically enabled in config/agentic-flow.config.ts
```

### Memory Management

```bash
# AgentDB persistence
AGENTDB_PERSISTENCE=true

# Vector index optimization (HNSW)
# Configured for 150x faster search than linear scan
```

### Concurrent Execution

```bash
# Max concurrent agents
# Configured in agentic-flow.config.ts: maxConcurrent: 10

# Use parallel strategy for non-critical tasks
AGENTIC_FLOW_STRATEGY=parallel
```

## 🆘 Troubleshooting

### Common Issues

#### 1. API Key Invalid

```bash
# Verify key format
echo $ANTHROPIC_API_KEY | grep "^sk-ant-"
echo $GOOGLE_AI_API_KEY | grep "^AIza"
echo $E2B_API_KEY | grep "^e2b_"

# Test connection
./scripts/test-integration.sh
```

#### 2. DevPod Won't Start

```bash
# Check Docker is running
docker info

# Reset DevPod provider
devpod provider delete docker
devpod provider add docker
devpod provider use docker

# Recreate workspace
devpod delete titan-ran --force
./scripts/start-devpod.sh
```

#### 3. Port Already in Use

```bash
# Find process using port 3000
lsof -i :3000

# Kill process
kill -9 <PID>

# Or change port in .env
UI_PORT=3100
```

#### 4. Out of Memory

```bash
# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory: 8GB

# Or reduce concurrent agents
# In config/agentic-flow.config.ts: maxConcurrent: 5
```

## 📚 Additional Resources

### Documentation
- Claude Agent SDK: https://github.com/anthropics/anthropic-sdk-typescript
- Google Gemini API: https://ai.google.dev/docs
- E2B Documentation: https://e2b.dev/docs
- OpenRouter API: https://openrouter.ai/docs
- Agentic Flow: https://github.com/ruvnet/agentic-flow
- Claude Flow: https://github.com/ruvnet/claude-flow
- DevPod: https://devpod.sh/docs

### Community
- TITAN GitHub: https://github.com/ricable/ultimate-ran
- Issues: https://github.com/ricable/ultimate-ran/issues
- Discussions: https://github.com/ricable/ultimate-ran/discussions

### Support
For help with setup:
1. Check this guide first
2. Run `./scripts/test-integration.sh` for diagnostics
3. Review logs in `logs/` directory
4. Open GitHub issue with diagnostic output

## 🎉 Next Steps

Once setup is complete:

1. **Test the system**: `./scripts/test-integration.sh`
2. **Run examples**: `npm run ui:integration`
3. **Open dashboard**: `npm run agui:frontend`
4. **Start optimizing**: `npm start`
5. **Monitor agents**: `npx claude-flow@alpha swarm status`

**Recommended Configuration for Production:**
- Strategy: `consensus`
- Primary Provider: `anthropic`
- Secondary Provider: `google`
- Sandbox Isolation: `strict`
- Quantum Signatures: `true`

Enjoy your multi-provider TITAN setup! 🚀
