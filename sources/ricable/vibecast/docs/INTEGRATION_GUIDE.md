# Integration Guide: NPX Tools & Claude Agent SDK

This guide explains how to integrate the platform with various NPX tools and extend it with additional Claude capabilities.

## NPX Tools Integration

### agentic-flow

Use `agentic-flow` for defining complex agentic workflows:

```bash
npx agentic-flow init
```

Create a workflow definition:

```yaml
# workflows/ran-optimization.yaml
name: ran-optimization
agents:
  - name: kpi-analyzer
    type: claude
    model: claude-sonnet-4-5-20250929
  - name: config-generator
    type: claude
    model: claude-sonnet-4-5-20250929

flow:
  - step: analyze-kpis
    agent: kpi-analyzer
    input: ${kpi_data}
  - step: generate-config
    agent: config-generator
    input: ${analyze-kpis.output}
    depends_on: [analyze-kpis]
```

### agentdb

Use `agentdb` for persistent agent memory and state:

```typescript
import { AgentDB } from 'agentdb';

const db = new AgentDB('./data/agentdb');

// Store agent conversation history
await db.store('ran-docs-agent', {
  conversationId: 'conv-001',
  messages: conversationHistory,
  metadata: { nodeId: 'gNB-001' }
});

// Retrieve past conversations for context
const pastConversations = await db.query('ran-docs-agent', {
  nodeId: 'gNB-001'
});
```

### claude-flow@alpha

Use `claude-flow@alpha` for advanced Claude orchestration:

```bash
npx claude-flow@alpha run --workflow ran-analysis.yaml
```

Workflow definition:

```yaml
# ran-analysis.yaml
version: alpha
orchestrator: claude-sonnet-4-5-20250929

tasks:
  - id: document-extraction
    agent: ran-docs
    prompt: "Extract all KPI definitions from the technical manual"

  - id: kpi-analysis
    agent: kpi-analyzer
    prompt: "Analyze KPI trends using the definitions from ${document-extraction}"
    dependencies: [document-extraction]

  - id: optimization
    agent: config-management
    prompt: "Generate optimization proposals based on ${kpi-analysis}"
    dependencies: [kpi-analysis]
```

### ruv-fann

Use `ruv-fann` for neural network-based prediction:

```bash
npx ruv-fann train --data kpi-data.csv --output model.fann
```

Integration with the platform:

```typescript
import { FannPredictor } from 'ruv-fann';

// In your custom predictor
class FannBasedPredictor extends BasePredictor {
  private fannModel: FannPredictor;

  constructor() {
    this.fannModel = new FannPredictor('model.fann');
  }

  async predict(data: MultiVariatePoint[]): Promise<PredictionResult[]> {
    const features = this.extractFeatures(data);
    const predictions = await this.fannModel.predict(features);
    return this.convertToPredictionResults(predictions);
  }
}
```

## Claude Agent SDK Integration

### Creating Custom Agents

Extend the base agent for custom functionality:

```typescript
import { BaseAgent, AgentResult } from './agents/base-agent.js';

class CustomRanAgent extends BaseAgent<MyInput, MyOutput> {
  constructor() {
    super({
      name: 'Custom-RAN-Agent',
      role: 'Custom RAN Specialist',
      systemPrompt: 'You are an expert in...',
      temperature: 0.5,
    });
  }

  async execute(input: MyInput): Promise<AgentResult<MyOutput>> {
    const validated = await this.validateInput(input, MyInputSchema);
    const response = await this.sendMessage(this.buildPrompt(validated));
    return this.parseResponse(response);
  }
}
```

### Claude Skills

Create reusable Claude skills:

```typescript
// src/skills/custom-skill.ts
export class CustomRanSkill {
  async execute(input: any): Promise<any> {
    // Skill implementation
    return result;
  }
}

// Register in orchestrator
orchestrator.registerSkill('custom-skill', new CustomRanSkill());
```

### Claude Subagents

Use subagents for specialized subtasks:

```typescript
class MainAnalysisAgent extends BaseAgent {
  private subagents: Map<string, BaseAgent>;

  constructor() {
    super(config);
    this.subagents = new Map([
      ['trend-analysis', new TrendAnalysisSubagent()],
      ['anomaly-detection', new AnomalyDetectionSubagent()],
      ['forecasting', new ForecastingSubagent()],
    ]);
  }

  async execute(input: any): Promise<any> {
    // Delegate to subagents based on task type
    const results = await Promise.all(
      Array.from(this.subagents.values()).map(agent =>
        agent.execute(input)
      )
    );

    // Aggregate subagent results
    return this.aggregateResults(results);
  }
}
```

## Advanced Orchestration Patterns

### Parallel Agent Execution

```typescript
const orchestrator = new MultiAgentOrchestrator();

// Execute multiple agents in parallel
const results = await orchestrator.orchestrate({
  taskType: 'custom',
  requiredAgents: ['ran-docs', 'kpi-analyzer', 'alarm-fault'],
  input: {
    'ran-docs': { query: 'Parameter definitions' },
    'kpi-analyzer': { measurements: kpiData },
    'alarm-fault': { alarms: alarmData },
  },
});
```

### Sequential Workflows

```typescript
// Step 1: Analyze alarms
const alarmAnalysis = await orchestrator.orchestrate({
  taskType: 'alarm-response',
  input: { alarms },
});

// Step 2: Use alarm insights for optimization
const optimization = await orchestrator.orchestrate({
  taskType: 'optimization',
  input: {
    node,
    kpiMeasurements,
    performanceIssues: extractIssues(alarmAnalysis),
  },
});
```

### Conditional Workflows

```typescript
async function adaptiveWorkflow(data: any) {
  const orchestrator = new MultiAgentOrchestrator();

  // Initial analysis
  const analysis = await orchestrator.orchestrate({
    taskType: 'full-analysis',
    input: data,
  });

  // Conditional branching based on results
  if (hasHighSeverityAlarms(analysis)) {
    return await orchestrator.orchestrate({
      taskType: 'alarm-response',
      input: { alarms: analysis.alarms },
    });
  } else if (hasPerformanceIssues(analysis)) {
    return await orchestrator.orchestrate({
      taskType: 'optimization',
      input: data,
    });
  }

  return analysis;
}
```

## Data Persistence

### Agent State Management

```typescript
import { AgentStateManager } from './services/state-manager.js';

const stateManager = new AgentStateManager('./data/state');

// Save agent state
await stateManager.saveState('kpi-analyzer', {
  lastAnalysis: Date.now(),
  baseline: kpiBaseline,
  models: trainedModels,
});

// Restore state
const state = await stateManager.loadState('kpi-analyzer');
```

### Time Series Data Storage

```typescript
import { TimeSeriesStore } from './services/ts-store.js';

const tsStore = new TimeSeriesStore('./data/timeseries');

// Store aggregated data
await tsStore.storeAggregated('gNB-001', 'Hourly', hourlyData);
await tsStore.storeAggregated('gNB-001', 'Daily', dailyData);

// Query historical data
const historical = await tsStore.query('gNB-001', {
  granularity: 'Hourly',
  startTime: Date.now() - 7 * 24 * 3600 * 1000,
  endTime: Date.now(),
});
```

## Event-Driven Architecture

```typescript
import { EventEmitter } from 'events';

class RanEventBus extends EventEmitter {
  constructor() {
    super();
    this.setupListeners();
  }

  private setupListeners() {
    this.on('alarm:critical', this.handleCriticalAlarm);
    this.on('kpi:anomaly', this.handleKpiAnomaly);
    this.on('prediction:degradation', this.handleDegradation);
  }

  private async handleCriticalAlarm(alarm: Alarm) {
    const orchestrator = new MultiAgentOrchestrator();
    await orchestrator.orchestrate({
      taskType: 'alarm-response',
      input: { alarms: [alarm] },
      priority: 10,
    });
  }
}

const eventBus = new RanEventBus();
eventBus.emit('alarm:critical', criticalAlarm);
```

## Monitoring & Observability

```typescript
import { logger } from './core/logger.js';

// Agent execution monitoring
class MonitoredAgent extends BaseAgent {
  async execute(input: any): Promise<any> {
    const startTime = Date.now();

    try {
      const result = await super.execute(input);

      logger.info('Agent execution completed', {
        agent: this.getName(),
        executionTimeMs: Date.now() - startTime,
        success: result.success,
      });

      return result;
    } catch (error) {
      logger.error('Agent execution failed', {
        agent: this.getName(),
        executionTimeMs: Date.now() - startTime,
        error,
      });
      throw error;
    }
  }
}
```

## Best Practices

### 1. Agent Specialization
- Keep agents focused on specific domains
- Use subagents for complex tasks
- Leverage skills for reusable functionality

### 2. Error Handling
```typescript
try {
  const result = await agent.execute(input);
  if (!result.success) {
    // Fallback strategy
    const fallbackResult = await fallbackAgent.execute(input);
  }
} catch (error) {
  // Graceful degradation
  logger.error('Agent failed, using default behavior', { error });
}
```

### 3. Rate Limiting
```typescript
import pLimit from 'p-limit';

const limit = pLimit(10); // Max 10 concurrent agent calls

const results = await Promise.all(
  tasks.map(task =>
    limit(() => agent.execute(task))
  )
);
```

### 4. Caching
```typescript
const cache = new Map<string, any>();

async function cachedAgentExecution(agent: BaseAgent, input: any) {
  const cacheKey = JSON.stringify(input);

  if (cache.has(cacheKey)) {
    return cache.get(cacheKey);
  }

  const result = await agent.execute(input);
  cache.set(cacheKey, result);

  return result;
}
```

## Testing

```typescript
import { describe, it, expect } from 'vitest';

describe('RAN Agents', () => {
  it('should analyze KPIs correctly', async () => {
    const agent = new KpiAnalyzerAgent();
    const result = await agent.execute({
      measurements: mockKpiData,
      analysisType: 'trend',
    });

    expect(result.success).toBe(true);
    expect(result.data.trends).toBeDefined();
  });
});
```

## Deployment

### Docker Support

```dockerfile
FROM node:18-alpine

WORKDIR /app

# Install Rust for building the engine
RUN apk add --no-cache rust cargo

COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

CMD ["npm", "start"]
```

### Environment Configuration

```bash
# Production
NODE_ENV=production
ANTHROPIC_API_KEY=prod_key
LOG_LEVEL=warn

# Staging
NODE_ENV=staging
ANTHROPIC_API_KEY=staging_key
LOG_LEVEL=info

# Development
NODE_ENV=development
ANTHROPIC_API_KEY=dev_key
LOG_LEVEL=debug
```

## Further Resources

- [Claude API Documentation](https://docs.anthropic.com)
- [Rust Time Series Libraries](https://docs.rs)
- [Multi-Agent Systems Patterns](https://patterns.arc.net)
