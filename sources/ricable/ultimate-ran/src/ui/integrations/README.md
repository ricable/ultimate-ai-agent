# AI Integrations for TITAN Dashboard

This directory contains integrations for AI-powered RAN optimization using:
- **Claude Agent SDK** (@anthropic-ai/sdk)
- **Google Generative AI** (@google/generative-ai) - Gemini

## Overview

The TITAN Dashboard uses a hybrid AI approach combining Claude's advanced reasoning with Gemini's multimodal capabilities for optimal RAN network optimization.

## Files

- **`claude-agent-integration.ts`** - Claude Agent SDK integration with tool use
- **`google-adk-integration.ts`** - Google Generative AI (Gemini) integration
- **`ai-orchestrator.ts`** - Hybrid orchestrator combining both AI engines
- **`example.ts`** - Complete usage examples
- **`index.ts`** - Module exports

## Quick Start

### 1. Set API Keys

```bash
export ANTHROPIC_API_KEY="your-claude-api-key"
export GOOGLE_AI_API_KEY="your-gemini-api-key"
```

### 2. Install Dependencies

Dependencies are already in package.json:
```bash
npm install
```

### 3. Build TypeScript

```bash
npm run build
```

### 4. Run Examples

```bash
# Run all examples
npm run ui:integration

# Run specific example
EXAMPLE=1 npm run ui:integration  # Claude only
EXAMPLE=2 npm run ui:integration  # Gemini only
EXAMPLE=3 npm run ui:integration  # AI Orchestrator (Hybrid)
EXAMPLE=4 npm run ui:integration  # Full Dashboard Integration
```

## Usage Examples

### Claude Agent SDK Only

```typescript
import { ClaudeAgentIntegration } from './integrations';

const claude = new ClaudeAgentIntegration({
  apiKey: process.env.ANTHROPIC_API_KEY!
});

const result = await claude.requestOptimization(
  cells,
  'Improve SINR for degraded cells',
  { min_power: -10, max_power: 10 }
);

console.log('Recommendations:', result.recommendations);
console.log('Reasoning:', result.reasoning);
console.log('Confidence:', result.confidence);
```

### Google Gemini Only

```typescript
import { GoogleADKIntegration } from './integrations';

const gemini = new GoogleADKIntegration({
  apiKey: process.env.GOOGLE_AI_API_KEY!
});

const analysis = await gemini.analyzeNetworkPerformance(
  cells,
  interferenceMatrix
);

console.log('Analysis:', analysis.analysis);
console.log('Recommendations:', analysis.recommendations);
console.log('Visual Insights:', analysis.visualInsights);
```

### AI Orchestrator (Recommended)

```typescript
import { AIOrchestrator } from './integrations';

const ai = new AIOrchestrator({
  claude: { apiKey: process.env.ANTHROPIC_API_KEY! },
  gemini: { apiKey: process.env.GOOGLE_AI_API_KEY! },
  strategy: 'consensus'  // 'claude_primary' | 'gemini_primary' | 'consensus' | 'parallel'
});

// Get optimization with consensus
const result = await ai.requestOptimization(
  cells,
  'Maximize throughput while maintaining QoS',
  interferenceMatrix
);

console.log('Source:', result.source);  // 'consensus'
console.log('Recommendations:', result.recommendations);
console.log('Confidence:', result.confidence);
```

### Dashboard Integration

```typescript
import { TitanDashboard } from '../titan-dashboard';
import { AIOrchestrator } from './integrations';

const dashboard = new TitanDashboard({ port: 8080 });
const ai = new AIOrchestrator({
  claude: { apiKey: claudeKey },
  gemini: { apiKey: geminiKey },
  strategy: 'parallel'
});

await dashboard.start();

// Use AI to analyze and optimize
const optimization = await ai.requestOptimization(
  degradedCells,
  'Improve SINR and reduce interference'
);

// Add events to dashboard
for (const rec of optimization.recommendations) {
  dashboard.addOptimizationEvent({
    id: `ai_opt_${Date.now()}_${rec.cell_id}`,
    timestamp: new Date().toISOString(),
    event_type: 'gnn_decision',
    cell_ids: [rec.cell_id],
    parameters_changed: [rec],
    reasoning: optimization.reasoning,
    confidence: optimization.confidence,
    status: 'pending'
  });
}

// Create approval request
const approval = dashboard.createApprovalRequest({
  action: 'AI-recommended optimization',
  target: targetCells,
  changes: optimization.recommendations,
  riskLevel: 'medium',
  justification: optimization.reasoning
});

// Validate with AI
const validation = await ai.validateApprovalRequest(approval);
console.log('Claude Decision:', validation.claudeDecision.approved);
console.log('Gemini Decision:', validation.geminiDecision.approved);
console.log('Final Decision:', validation.finalDecision);
```

## Orchestration Strategies

### 1. `claude_primary`
- Uses Claude Agent SDK as the primary decision maker
- Fast, reliable reasoning with tool use
- Best for: Structured parameter optimization

### 2. `gemini_primary`
- Uses Gemini as the primary decision maker
- Multimodal analysis with visual insights
- Best for: Pattern recognition and anomaly detection

### 3. `consensus` (Recommended)
- Both AIs must agree on recommendations
- Higher confidence, lower risk
- Returns only parameters where both AIs agree
- Best for: Production deployments

### 4. `parallel`
- Runs both AIs in parallel and combines results
- Maximum coverage of optimization opportunities
- Weighted confidence scoring
- Best for: Comprehensive analysis

## Features

### Claude Agent SDK Features
- ✅ Tool use with structured schemas (Zod validation)
- ✅ RAN parameter optimization within 3GPP bounds
- ✅ Interference pattern analysis
- ✅ KPI impact prediction
- ✅ Cell performance analysis
- ✅ Approval request validation
- ✅ Conversation history management

### Google Gemini Features
- ✅ Multimodal network performance analysis
- ✅ Anomaly detection with severity classification
- ✅ Optimization strategy generation
- ✅ Visual insight extraction
- ✅ Function calling for parameter optimization
- ✅ Interference prediction
- ✅ Historical baseline comparison

### AI Orchestrator Features
- ✅ Hybrid intelligence combining both AIs
- ✅ Multiple orchestration strategies
- ✅ Consensus-based decision making
- ✅ Parallel execution for performance
- ✅ Weighted confidence scoring
- ✅ Comprehensive network insights
- ✅ Dual validation for approvals

## API Reference

### Claude Agent SDK

#### `requestOptimization(cells, objective, constraints?)`
Request RAN parameter optimization recommendations.

#### `analyzeCellPerformance(cell)`
Analyze a specific cell's performance and suggest improvements.

#### `validateApprovalRequest(request)`
Validate an approval request using Claude's judgment.

### Google ADK

#### `analyzeNetworkPerformance(cells, interferenceMatrix?)`
Analyze overall network performance with multimodal insights.

#### `detectAnomalies(cells, historicalBaseline?)`
Detect performance anomalies across cells.

#### `generateOptimizationStrategy(objective, cells, constraints?)`
Generate a comprehensive optimization strategy with timeline.

### AI Orchestrator

#### `requestOptimization(cells, objective, interferenceMatrix?)`
Request optimization using the configured strategy.

#### `analyzeCellPerformance(cell)`
Analyze cell performance with both AIs.

#### `validateApprovalRequest(request)`
Validate approval request with dual AI assessment.

#### `getNetworkInsights(cells, interferenceMatrix?)`
Get comprehensive network insights from both AIs.

## Environment Variables

```bash
# Required for Claude integration
ANTHROPIC_API_KEY="sk-ant-..."

# Required for Gemini integration
GOOGLE_AI_API_KEY="AIza..."
# Alternative name
GEMINI_API_KEY="AIza..."

# Optional: Run specific example
EXAMPLE="1"  # 1-4
```

## Best Practices

1. **Use Consensus Mode in Production**
   - Higher confidence and safety
   - Both AIs must agree on changes

2. **Set Appropriate Constraints**
   - Always specify 3GPP-compliant bounds
   - Use constraints to limit risk

3. **Validate High-Risk Changes**
   - Use `validateApprovalRequest()` for risky changes
   - Require dual AI approval

4. **Monitor Confidence Scores**
   - Only apply recommendations with >80% confidence
   - Review low-confidence suggestions manually

5. **Clear Conversation History**
   - Call `clearHistory()` between optimization sessions
   - Prevents context pollution

## Integration with TITAN Components

### SPARC Methodology
The AI integrations follow SPARC methodology:
- **S**pecification: Define optimization objectives
- **P**seudocode: AI generates parameter change logic
- **A**rchitecture: Validates 3GPP compliance
- **R**efinement: Iterative optimization
- **C**ompletion: Execute with approval workflow

### Guardian Agent
AI recommendations are validated by Guardian Agent:
```typescript
const approval = dashboard.createApprovalRequest({...});
const validation = await ai.validateApprovalRequest(approval);

if (validation.finalDecision && validation.claudeDecision.approved) {
  // Pass to Guardian for digital twin simulation
  await guardianAgent.simulateChange(approval);
}
```

### 3-ROP Closed Loop
AI monitors optimization results across 3 ROPs:
- **ROP 1**: Baseline measurement
- **ROP 2**: Compare actual vs predicted KPI impact
- **ROP 3**: Confirm success or trigger rollback

## Troubleshooting

### "API key not set"
```bash
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_AI_API_KEY="your-key"
```

### "Cannot find module"
```bash
npm run build  # Build TypeScript first
```

### "Tool use error"
- Check Zod schema matches tool input
- Verify 3GPP bounds are respected

### "Low confidence scores"
- Provide more cell context
- Include historical baseline data
- Use interferenceMatrix for better analysis

## Performance

- **Claude API**: ~2-5s per optimization request
- **Gemini API**: ~3-7s per analysis
- **Consensus Mode**: ~5-10s (parallel execution)
- **Parallel Mode**: ~5-8s (parallel execution)

## License

Proprietary - Ericsson Autonomous Networks Division

## Support

For issues or questions:
- GitHub: https://github.com/ericsson/titan-ran
- Docs: https://titan-ran.ericsson.com/docs/ai-integrations
