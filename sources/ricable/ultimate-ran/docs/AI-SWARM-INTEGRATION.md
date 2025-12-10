# AI Swarm Integration - Claude Agent SDK + Google ADK
## Consensus-Based Multi-Agent Reasoning for TITAN RAN

This document explains how to use the AI Swarm Coordinator in the TITAN AG-UI for consensus-based RAN optimization using both Claude Agent SDK and Google Generative AI (Gemini).

---

## 🎯 Overview

The **AI Swarm Coordinator** orchestrates multiple AI agents (Claude + Gemini) in a distributed swarm to perform consensus-based reasoning for network optimization decisions. This ensures higher confidence and more robust decision-making through multi-agent validation.

### Key Features

- **Dual AI Integration**: Claude 3.5 Sonnet + Google Gemini 2.0 Flash
- **Consensus Voting**: Multi-agent approval system with configurable thresholds
- **Multiple Topologies**: Hierarchical, Mesh, and Consensus network patterns
- **Real-time Monitoring**: Live agent status and task tracking in AG-UI
- **Reasoning Transparency**: Full visibility into agent votes and confidence levels

---

## 🚀 Quick Start

### 1. **Configure API Keys**

Create or update your environment configuration:

```bash
# Copy template
cp config/.env.template config/.env

# Add your API keys
ANTHROPIC_API_KEY=sk-ant-...    # Claude Agent SDK
GOOGLE_AI_API_KEY=AIza...       # Google AI Studio
```

**Getting API Keys:**
- **Claude**: [Anthropic Console](https://console.anthropic.com/) - Requires Claude Code PRO MAX subscription
- **Gemini**: [Google AI Studio](https://aistudio.google.com/app/apikey) - Free tier available (15 RPM)

### 2. **Initialize the Swarm**

Access the TITAN Dashboard:

```bash
npm run agui:frontend
# Navigate to http://localhost:8080
```

In the AG-UI:
1. Select swarm topology (Consensus recommended)
2. Click **"Initialize Swarm"**
3. Wait for agents to spawn (typically 3-6 agents)

### 3. **Monitor Agent Activity**

The dashboard displays:
- **Agent Cards**: Real-time status of each Claude/Gemini agent
- **Consensus History**: Past voting decisions with confidence scores
- **Task Queue**: Active and completed optimization tasks

---

## 📊 Swarm Topologies

### **Consensus (Recommended)**
- **Pattern**: Mixed Claude + Gemini agents vote on decisions
- **Use Case**: Critical optimizations requiring high confidence
- **Threshold**: 75% agent agreement required
- **Agents**: 2 Claude + 2 Gemini (4 total)

```typescript
consensusThreshold: 0.75 // 75% must agree
```

### **Hierarchical**
- **Pattern**: One coordinator agent leads, others assist
- **Use Case**: Fast optimization with central authority
- **Agents**: 1 Coordinator + 2-4 Workers

### **Mesh**
- **Pattern**: All agents collaborate peer-to-peer
- **Use Case**: Maximum redundancy and fault tolerance
- **Agents**: Up to 6 agents (configurable)

---

## 🤖 Agent Roles

Each agent is assigned a role for specialized tasks:

| Role | Responsibility | Example |
|------|---------------|---------|
| **Analyzer** | Performance analysis and anomaly detection | "Low SINR detected in Cell ABC-123" |
| **Optimizer** | Parameter tuning recommendations | "Increase power_dbm from 30 to 33" |
| **Validator** | Safety checks and 3GPP compliance | "Change is within bounds [-130, 46]" |
| **Coordinator** | Task distribution and consensus building | "Initiating vote on approval request" |

---

## 📡 How Consensus Works

### Approval Validation Flow

```
1. Approval Request Submitted
   ↓
2. Swarm Coordinator selects validator agents (2 Claude + 2 Gemini)
   ↓
3. Each agent analyzes request independently:
   - Claude: Deep reasoning with tool use
   - Gemini: Multimodal analysis with risk assessment
   ↓
4. Agents cast votes (APPROVE/REJECT) with confidence scores
   ↓
5. Consensus Decision:
   ✅ APPROVED:     75%+ agents approve
   ❌ REJECTED:     75%+ agents reject
   ⚠️  NEEDS REVIEW: < 75% agreement
   ↓
6. Final decision logged with full transparency
```

### Example Consensus Result

```json
{
  "decision": "approved",
  "confidence": 0.88,
  "votes": [
    {"agent_id": "claude-1", "vote": true, "confidence": 0.92},
    {"agent_id": "claude-2", "vote": true, "confidence": 0.85},
    {"agent_id": "gemini-1", "vote": true, "confidence": 0.87},
    {"agent_id": "gemini-2", "vote": false, "confidence": 0.88}
  ],
  "finalReasoning": "Consensus reached with 75% approval..."
}
```

**Decision**: APPROVED (3/4 agents = 75%)

---

## 🛠️ API Integration

### Initialize Swarm

```typescript
import { AISwarmCoordinator } from './ui';

const swarm = new AISwarmCoordinator({
  claude: {
    apiKey: process.env.ANTHROPIC_API_KEY!,
    model: 'claude-3-5-sonnet-20241022'
  },
  gemini: {
    apiKey: process.env.GOOGLE_AI_API_KEY!,
    model: 'gemini-2.0-flash-exp'
  },
  topology: 'consensus',
  consensusThreshold: 0.75,
  maxAgents: 6,
  enableLearning: true
});
```

### Request Consensus Optimization

```typescript
const result = await swarm.requestConsensusOptimization(
  cells,                    // Array of CellStatus objects
  "Maximize SINR coverage", // Optimization objective
  interferenceMatrix        // Optional: cell interference data
);

console.log(`Decision: ${result.consensus.decision}`);
console.log(`Confidence: ${result.consensus.confidence}`);
console.log(`Recommendations: ${result.recommendations.length}`);
```

### Validate Approval with Consensus

```typescript
const consensus = await swarm.validateWithConsensus(approvalRequest);

if (consensus.decision === 'approved') {
  // Execute parameter change
  console.log('✅ Approved by AI swarm');
} else {
  console.log('❌ Rejected - requires human review');
}
```

---

## 🎨 AG-UI Dashboard

### Visual Components

#### 1. **Swarm Control Panel**
- Topology selector (Consensus/Hierarchical/Mesh)
- Initialize and Reset buttons
- Real-time statistics (Total/Active/Busy agents)

#### 2. **Agent Cards**
Each agent displays:
- **Agent ID**: e.g., "claude-1", "gemini-2"
- **Type**: Claude (purple border) or Gemini (blue border)
- **Role**: Analyzer, Optimizer, Validator, Coordinator
- **Status**: 🟢 Idle, 🟡 Busy, 🔴 Error
- **Confidence**: Current confidence level (0-100%)
- **Tasks Completed**: Total number of completed tasks

#### 3. **Consensus History**
Shows recent voting decisions:
- **Decision**: APPROVED (green), REJECTED (red), NEEDS REVIEW (orange)
- **Confidence**: Average confidence across all votes
- **Vote Breakdown**: Individual agent votes with reasoning

---

## ⚙️ Configuration Options

### Consensus Threshold

Controls what percentage of agents must agree:

```typescript
consensusThreshold: 0.75  // 75% (Recommended for production)
consensusThreshold: 0.5   // 50% (Faster, lower confidence)
consensusThreshold: 0.9   // 90% (Highest confidence, slower)
```

### Agent Learning

Enable agents to learn from past decisions:

```typescript
enableLearning: true  // Agents update confidence based on outcomes
```

### Maximum Agents

Limit swarm size for cost/performance balance:

```typescript
maxAgents: 4   // Budget-friendly (2 Claude + 2 Gemini)
maxAgents: 6   // Balanced (3 + 3)
maxAgents: 8   // High reliability (4 + 4)
```

---

## 🔐 Security & Privacy

- **API Keys**: Stored in environment variables, never in code
- **Request Isolation**: Each agent operates in isolated context
- **Audit Trail**: All consensus decisions logged with timestamps
- **Rate Limiting**: Respects provider API limits (Claude: 50 RPM, Gemini: 15 RPM)

---

## 📈 Performance Metrics

### Typical Latencies

| Operation | Latency | Agents |
|-----------|---------|--------|
| Single Agent | 800-1200ms | 1 |
| Consensus (2+2) | 1500-2500ms | 4 |
| Mesh (3+3) | 2000-3500ms | 6 |

### Cost Estimates

Based on average tokens per request:

- **Claude 3.5 Sonnet**: ~$0.003 per consensus vote
- **Gemini 2.0 Flash**: ~$0.0001 per consensus vote
- **4-Agent Consensus**: ~$0.006 per decision

---

## 🧪 Testing

### Manual Testing

1. Initialize swarm in AG-UI
2. Submit an approval request
3. Watch agents vote in real-time
4. Verify consensus decision matches threshold

### Automated Testing

```bash
npm run ui:integration   # Test AI integrations
npm run test:consensus   # Test consensus logic
```

---

## 🐛 Troubleshooting

### Issue: "Swarm initialization failed"

**Cause**: Missing or invalid API keys

**Solution**:
```bash
# Check environment variables
echo $ANTHROPIC_API_KEY
echo $GOOGLE_AI_API_KEY

# Verify keys are valid
curl -H "x-api-key: $ANTHROPIC_API_KEY" https://api.anthropic.com/v1/models
```

### Issue: "Agent stuck in BUSY status"

**Cause**: API timeout or network error

**Solution**: Click "Reset Swarm" to reinitialize all agents

### Issue: "Low confidence scores"

**Cause**: Ambiguous optimization objectives

**Solution**: Provide clearer objectives with specific KPI targets:
- ❌ "Improve network"
- ✅ "Maximize SINR to 15dB in downtown cells"

---

## 📚 Further Reading

- [Claude Agent SDK Documentation](https://docs.anthropic.com/agent-sdk)
- [Google Gemini API Guide](https://ai.google.dev/docs)
- [TITAN Architecture Overview](./architecture-status-report.md)
- [Multi-Provider Setup](./MULTI-PROVIDER-SETUP.md)

---

## 🤝 Contributing

Contributions to improve AI swarm coordination are welcome!

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## 📄 License

TITAN RAN Platform © 2025 - MIT License
