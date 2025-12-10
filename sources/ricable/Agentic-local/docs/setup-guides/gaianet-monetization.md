# GaiaNet Monetization Guide

## Overview

GaiaNet transforms your local AI inference node into a revenue-generating asset by allowing you to serve AI requests from the decentralized network while earning crypto rewards.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Your Mac Silicon                      │
│                                                          │
│  ┌────────────────┐         ┌─────────────────────┐   │
│  │  Agentic Flow  │────────▶│   GaiaNet Node      │   │
│  │  (Your Agents) │         │  (WasmEdge + MLX)   │   │
│  └────────────────┘         └──────────┬──────────┘   │
│         │                              │               │
│         │ Local inference              │ Public API    │
│         │ (Free)                       │ (Earn crypto) │
│         │                              │               │
└─────────┼──────────────────────────────┼───────────────┘
          │                              │
          │                              ▼
          │                   ┌──────────────────┐
          │                   │  GaiaNet Network │
          │                   │  (Public clients)│
          │                   └──────────────────┘
          │                              │
          │                              ▼
          │                   ┌──────────────────┐
          └──────────────────▶│  Smart Contract  │
                              │  Reward Pool     │
                              └──────────────────┘
```

## Economic Model

### Gaia Points System

During the testnet/early mainnet phase, nodes earn **Gaia Points** based on:

1. **Availability (Uptime)**
   - Nodes must maintain >95% uptime to qualify for full rewards
   - Downtime penalties scale linearly
   - Formula: `availability_score = uptime_hours / total_hours`

2. **Throughput (Tokens Processed)**
   - Rewards based on total tokens served
   - Premium multipliers for specialized domains
   - Formula: `throughput_score = tokens_processed * domain_multiplier`

3. **Quality (Response Validation)**
   - Client feedback on response quality
   - Latency measurements
   - Error rate tracking

### Token Generation Event (TGE)

- **Expected Timeframe**: Q2-Q3 2025
- **Conversion**: Gaia Points → GAIA tokens
- **Initial Staking Requirement**: TBD (likely 1000-5000 GAIA tokens)

### Earning Potential

Based on current network data (as of Dec 2024):

| Hardware | Daily Requests | Daily Points | Monthly Estimate |
|----------|---------------|--------------|------------------|
| M1 Mac Mini (16GB) | 500-1000 | 50-100 | 1,500-3,000 pts |
| M2 Mac Studio (32GB) | 2000-4000 | 200-400 | 6,000-12,000 pts |
| M3 Max (64GB) | 5000-10000 | 500-1000 | 15,000-30,000 pts |

*Note: These are estimates. Actual earnings depend on network demand, domain selection, and model performance.*

## Setup Instructions

### 1. Install GaiaNet Node

```bash
# Run the automated setup script
./scripts/setup-gaianet.sh

# Or manually:
curl -sSfL 'https://github.com/GaiaNet-AI/gaianet-node/releases/latest/download/install.sh' | bash
```

### 2. Configure for Qwen 2.5 Coder

For optimal performance with coding agents:

```bash
# Initialize with Qwen 2.5 Coder 32B (adjust size based on your RAM)
gaianet init --config https://raw.githubusercontent.com/GaiaNet-AI/node-configs/main/qwen2.5-coder-32b-instruct/config.json

# Set correct prompt template (CRITICAL for Qwen models)
gaianet config --prompt-template chatml

# Optimize context window for coding tasks
gaianet config --ctx-size 32768

# Enable GPU acceleration (MLX on Mac)
gaianet config --use-gpu true
```

### 3. Start Your Node

```bash
# Start the node
gaianet start

# Verify it's running
gaianet info

# Check logs for any errors
gaianet log
```

Your node will be available at: `http://localhost:8080/v1`

### 4. Register with GaiaNet Network

1. Visit the [GaiaNet Registration Portal](https://www.gaianet.ai/register)
2. Connect your wallet (MetaMask recommended)
3. Enter your node ID (get via `gaianet info`)
4. Select your domain (choose "developer-tools" for coding agents)
5. Set your public endpoint (or use GaiaNet's proxy)

### 5. Configure Firewall & Networking

For macOS:

```bash
# Allow incoming connections on port 8080
# System Settings > Network > Firewall > Options
# Add: WasmEdge (Allow incoming connections)

# Or via command line (requires sudo):
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /path/to/wasmedge
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /path/to/wasmedge
```

## Dual-Mode Operation

The key advantage: **Use your own node for free while earning from external requests.**

### Configure Agentic Flow for Local Node

In your `.env` file:

```bash
LLM_PROVIDER=local
GAIANET_ENDPOINT=http://localhost:8080/v1
GAIANET_MODEL=Qwen2.5-Coder-32B-Instruct
```

### Agent Configuration Example

```javascript
import { AgenticFlow } from 'agentic-flow';

const agent = new AgenticFlow({
  provider: 'custom',
  baseURL: process.env.GAIANET_ENDPOINT,
  model: process.env.GAIANET_MODEL,
  // Your requests hit localhost - no cost
});

// Your agent uses local inference - ZERO API COSTS
const result = await agent.run('Generate a REST API in Express.js');
```

Meanwhile, your node serves external requests and earns points!

## Monitoring & Optimization

### Dashboard

Access your earnings dashboard:
```
https://www.gaianet.ai/dashboard?node_id=YOUR_NODE_ID
```

Metrics tracked:
- Total points earned
- Request volume (hourly/daily)
- Average latency
- Error rate
- Uptime percentage

### Performance Optimization

#### 1. Model Selection

| Use Case | Recommended Model | RAM Required |
|----------|------------------|--------------|
| Fast responses, high volume | Qwen 2.5 Coder 7B | 16GB |
| Balanced quality/speed | Qwen 2.5 Coder 14B | 24GB |
| Maximum quality | Qwen 2.5 Coder 32B | 48GB+ |

#### 2. Context Window Tuning

```bash
# For short code snippets (higher throughput)
gaianet config --ctx-size 8192

# For full file analysis (better quality)
gaianet config --ctx-size 32768

# For entire codebases (premium requests)
gaianet config --ctx-size 65536
```

#### 3. Temperature Settings

```bash
# For code generation (deterministic)
gaianet config --temperature 0.2

# For creative coding (exploratory)
gaianet config --temperature 0.7
```

### Resource Management

Monitor your Mac's performance:

```bash
# Watch CPU/GPU usage
top -o cpu

# Check memory pressure
memory_pressure

# Monitor GPU (if available)
sudo powermetrics --samplers gpu_power
```

Set limits to prevent thermal throttling:

```bash
# Limit batch size
gaianet config --batch-size 512

# Limit concurrent requests
gaianet config --max-requests 4
```

## Domain Selection Strategy

Domains affect your earning multiplier:

| Domain | Multiplier | Competition | Best For |
|--------|-----------|-------------|----------|
| general-purpose | 1.0x | High | Broadest reach |
| developer-tools | 1.5x | Medium | Qwen Coder models |
| data-science | 1.3x | Medium | Analysis-focused |
| web3-dev | 1.8x | Low | Solidity, smart contracts |

**Recommendation for Qwen Coder**: Join `developer-tools` or `web3-dev` domains.

## Staking (Post-TGE)

Once GAIA tokens launch:

### Benefits of Staking

1. **Reward Multiplier**: Staked nodes earn 2-5x base rewards
2. **Priority Routing**: Higher-paying requests routed to staked nodes first
3. **Governance**: Vote on network parameters

### Risks

1. **Slashing**: Persistent downtime or malicious behavior results in stake loss
2. **Lock-up Period**: Tokens locked for 30-90 days
3. **Opportunity Cost**: Staked tokens can't be traded

### Recommended Staking Strategy

```
Phase 1 (Months 1-3): Accumulate points, don't stake immediately
Phase 2 (Months 4-6): Stake 50% of tokens, keep 50% liquid
Phase 3 (Months 7+): Adjust based on ROI and network maturity
```

## Troubleshooting

### Node Not Earning Points

**Check registration:**
```bash
gaianet info
# Verify "Network Status: Connected"
```

**Verify public endpoint:**
```bash
curl http://YOUR_PUBLIC_IP:8080/v1/models
```

**Check logs for errors:**
```bash
gaianet log | grep ERROR
```

### Low Request Volume

1. **Switch domain**: Try `web3-dev` for less competition
2. **Improve uptime**: Set up auto-restart on crashes
3. **Upgrade hardware**: Faster responses = more requests

### High Error Rate

1. **Check model compatibility:**
   ```bash
   gaianet config --show
   # Ensure prompt_template = "chatml" for Qwen
   ```

2. **Monitor memory:**
   ```bash
   # If seeing OOM errors, reduce context size
   gaianet config --ctx-size 16384
   ```

3. **Update WasmEdge:**
   ```bash
   ./scripts/setup-wasmedge-mlx.sh
   ```

## Advanced: RAG-Enhanced Node

Boost your node's value with specialized knowledge:

```bash
# Add a custom knowledge base
gaianet rag import --source ./knowledge-base --collection rust-std-lib

# Update node description
gaianet config --description "Rust expert with std lib knowledge"

# Restart to apply
gaianet restart
```

This allows your node to answer specialized queries other nodes can't, commanding premium rates.

## Security Considerations

### Never Expose These

- Private keys / wallet seeds
- Node admin API (keep on localhost only)
- File system access (WasmEdge sandbox prevents this, but verify)

### Best Practices

1. **Run as non-root user**
2. **Use firewall rules** to limit exposed ports
3. **Monitor logs** for suspicious activity
4. **Backup config** regularly
5. **Update promptly** when security patches released

## Summary

By running a GaiaNet node with Qwen 2.5 Coder on your Mac Silicon:

✅ **Zero-cost local inference** for your own agents
✅ **Earn crypto rewards** from external requests
✅ **Utilize idle GPU** when not developing
✅ **Participate in decentralized AI** infrastructure

Expected ROI timeline:
- **Month 1-3**: Setup and optimization, minimal earnings
- **Month 4-6**: Steady point accumulation (1000-5000 points/month)
- **Post-TGE**: Convert to tokens, stake for multiplier
- **Year 1+**: Potential passive income stream ($100-$500/month estimated)

Start with the automated script:
```bash
./scripts/setup-gaianet.sh
```

Monitor your node:
```bash
gaianet info
gaianet log
```

Join the network:
https://www.gaianet.ai/register
