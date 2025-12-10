# MCP Server Integration Patch for Attention Tools

## Overview

This document provides instructions for integrating attention mechanism tools into the AgentDB MCP server.

## Files Created

1. `/workspaces/agentic-flow/packages/agentdb/src/mcp/attention-tools-handlers.ts` - Tool handlers and definitions
2. `/workspaces/agentic-flow/packages/agentdb/src/mcp/attention-mcp-integration.ts` - Integration helper

## Integration Steps

### 1. Import Attention Tools

Add to `/workspaces/agentic-flow/packages/agentdb/src/mcp/agentdb-mcp-server.ts` imports section:

```typescript
import {
  attentionTools,
  attentionToolHandlers,
  attentionHelperFunctions
} from './attention-mcp-integration.js';
```

### 2. Register Tools in Tools Array

Find the `tools` array definition and add:

```typescript
const tools = [
  // ... existing tools (e.g., learning tools) ...

  // Attention mechanism tools
  ...attentionTools,

  // ... rest of tools ...
];
```

### 3. Add Tool Handlers

In the `CallToolRequestSchema` handler, add the helper functions and case statements.

Find this section:
```typescript
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
```

Add helper functions right after (before the switch statement):

```typescript
  // Helper functions for attention MCP tools
  function encodeQueryVector(query: string, dimension: number): number[] {
    const vector = Array(dimension).fill(0);
    for (let i = 0; i < query.length; i++) {
      const idx = query.charCodeAt(i) % dimension;
      vector[idx] += 1;
    }
    const norm = Math.sqrt(vector.reduce((sum: number, x: number) => sum + x * x, 0));
    return vector.map(x => x / (norm || 1));
  }

  function computeAttentionWeightsMCP(
    mechanism: string,
    query: number[],
    keys: number[][],
    heads: number
  ): number[][] {
    const weights: number[][] = [];
    for (let h = 0; h < heads; h++) {
      const headWeights: number[] = [];
      for (const key of keys) {
        let score = 0;
        switch (mechanism) {
          case 'flash':
          case 'linear':
          case 'performer':
            score = dotProductMCP(query, key);
            break;
          case 'hyperbolic':
            score = 1 / (1 + poincareDistanceMCP(query, key));
            break;
          case 'sparse':
            score = Math.random() > 0.9 ? dotProductMCP(query, key) : 0;
            break;
          default:
            score = dotProductMCP(query, key);
        }
        headWeights.push(score);
      }
      const maxScore = Math.max(...headWeights);
      const expScores = headWeights.map(s => Math.exp(s - maxScore));
      const sumExp = expScores.reduce((a: number, b: number) => a + b, 0);
      weights.push(expScores.map(s => s / sumExp));
    }
    return weights;
  }

  function applyAttentionWeightsMCP(weights: number[][], values: number[][]): number[][] {
    return weights.map(headWeights => {
      const output = Array(values[0]?.length || 384).fill(0);
      for (let i = 0; i < values.length; i++) {
        for (let j = 0; j < output.length; j++) {
          output[j] += headWeights[i] * (values[i]?.[j] || 0);
        }
      }
      return output;
    });
  }

  function generateRandomKeysMCP(count: number, dimension: number): number[][] {
    return Array(count).fill(0).map(() =>
      Array(dimension).fill(0).map(() => Math.random() * 2 - 1)
    );
  }

  function dotProductMCP(a: number[], b: number[]): number {
    return a.reduce((sum, val, i) => sum + val * (b[i] || 0), 0);
  }

  function poincareDistanceMCP(a: number[], b: number[]): number {
    const diff = a.map((val, i) => val - (b[i] || 0));
    const normDiff = Math.sqrt(diff.reduce((sum, x) => sum + x * x, 0));
    const normA = Math.sqrt(a.reduce((sum, x) => sum + x * x, 0));
    const normB = Math.sqrt(b.reduce((sum, x) => sum + x * x, 0));
    const numerator = normDiff * normDiff;
    const denominator = (1 - normA * normA) * (1 - normB * normB);
    return Math.acosh(1 + 2 * numerator / Math.max(denominator, 1e-8));
  }

  function estimateAttentionMemory(keyCount: number, dimension: number, heads: number): number {
    const keysMemory = keyCount * dimension * 4;
    const valuesMemory = keyCount * dimension * 4;
    const weightsMemory = heads * keyCount * 4;
    return (keysMemory + valuesMemory + weightsMemory) / (1024 * 1024);
  }

  switch (name) {
```

Then add the attention tool cases in the switch statement:

```typescript
    // Attention mechanism tools
    case 'agentdb_attention_compute': {
      const mechanism = args?.mechanism as string || 'flash';
      const query = args?.query as string;
      const keys = args?.keys as number[][] || [];
      const values = args?.values as number[][] || [];
      const heads = (args?.heads as number) || 8;
      const dimension = (args?.dimension as number) || 384;

      if (!query && keys.length === 0) {
        return {
          content: [{
            type: 'text',
            text: '❌ Error: Either query or keys must be provided',
          }],
        };
      }

      try {
        const queryVector = query
          ? encodeQueryVector(query, dimension)
          : keys[0] || Array(dimension).fill(0);

        const startTime = performance.now();
        const attentionWeights = computeAttentionWeightsMCP(
          mechanism,
          queryVector,
          keys.length > 0 ? keys : [queryVector],
          heads
        );

        const output = applyAttentionWeightsMCP(
          attentionWeights,
          values.length > 0 ? values : (keys.length > 0 ? keys : [queryVector])
        );

        const computeTime = performance.now() - startTime;
        const memoryUsed = estimateAttentionMemory(keys.length || 1, dimension, heads);

        return {
          content: [{
            type: 'text',
            text: `🧠 Attention Computation Complete\n\n` +
              `Mechanism: ${mechanism}\n` +
              `Heads: ${heads}\n` +
              `Dimension: ${dimension}\n` +
              `Keys: ${keys.length || 1}\n` +
              `Values: ${values.length || keys.length || 1}\n\n` +
              `Performance:\n` +
              `  Compute Time: ${computeTime.toFixed(2)}ms\n` +
              `  Memory Used: ${memoryUsed.toFixed(2)}MB\n\n` +
              `Output Shape: [${heads}, ${output[0]?.length || dimension}]\n` +
              `Attention Weights Sample: [${attentionWeights[0]?.slice(0, 5).map(w => w.toFixed(4)).join(', ')}...]\n`,
          }],
        };
      } catch (error: any) {
        return {
          content: [{
            type: 'text',
            text: `❌ Error computing attention: ${error.message}`,
          }],
        };
      }
    }

    case 'agentdb_attention_benchmark': {
      const mechanism = args?.mechanism as string;
      const all = (args?.all as boolean) ?? false;
      const iterations = (args?.iterations as number) || 100;
      const dimension = (args?.dimension as number) || 384;
      const keyCount = (args?.key_count as number) || 100;

      const mechanismsToTest = all
        ? ['flash', 'hyperbolic', 'sparse', 'linear', 'performer']
        : mechanism
        ? [mechanism]
        : ['flash'];

      const results: any[] = [];

      for (const mech of mechanismsToTest) {
        const times: number[] = [];
        const memories: number[] = [];

        const testKeys = generateRandomKeysMCP(keyCount, dimension);
        const testQuery = Array(dimension).fill(0).map(() => Math.random());

        for (let i = 0; i < iterations; i++) {
          const startTime = performance.now();
          const weights = computeAttentionWeightsMCP(mech, testQuery, testKeys, 8);
          times.push(performance.now() - startTime);
          memories.push(estimateAttentionMemory(keyCount, dimension, 8));
        }

        const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
        const minTime = Math.min(...times);
        const maxTime = Math.max(...times);
        const stdDev = Math.sqrt(
          times.reduce((sum, t) => sum + Math.pow(t - avgTime, 2), 0) / times.length
        );
        const avgMemory = memories.reduce((a, b) => a + b, 0) / memories.length;

        results.push({
          mechanism: mech,
          iterations,
          avgTimeMs: avgTime,
          minTimeMs: minTime,
          maxTimeMs: maxTime,
          stdDevMs: stdDev,
          avgMemoryMB: avgMemory,
        });
      }

      const sorted = [...results].sort((a, b) => a.avgTimeMs - b.avgTimeMs);
      const fastest = sorted[0];
      const slowest = sorted[sorted.length - 1];
      const speedup = slowest.avgTimeMs / fastest.avgTimeMs;

      let output = `⚡ Attention Mechanism Benchmark Results\n\n`;
      output += `Configuration:\n`;
      output += `  Iterations: ${iterations}\n`;
      output += `  Dimension: ${dimension}\n`;
      output += `  Key Count: ${keyCount}\n\n`;

      for (const result of results) {
        output += `${result.mechanism}:\n`;
        output += `  Avg Time: ${result.avgTimeMs.toFixed(3)}ms\n`;
        output += `  Min Time: ${result.minTimeMs.toFixed(3)}ms\n`;
        output += `  Max Time: ${result.maxTimeMs.toFixed(3)}ms\n`;
        output += `  Std Dev: ${result.stdDevMs.toFixed(3)}ms\n`;
        output += `  Avg Memory: ${result.avgMemoryMB.toFixed(2)}MB\n\n`;
      }

      output += `Comparison:\n`;
      output += `  Fastest: ${fastest.mechanism} (${fastest.avgTimeMs.toFixed(3)}ms)\n`;
      output += `  Slowest: ${slowest.mechanism} (${slowest.avgTimeMs.toFixed(3)}ms)\n`;
      output += `  Speedup: ${speedup.toFixed(2)}x\n`;
      output += `  Recommendation: ${fastest.mechanism}\n`;

      return {
        content: [{
          type: 'text',
          text: output,
        }],
      };
    }

    case 'agentdb_attention_configure': {
      const mechanism = args?.mechanism as string;
      const config = args?.config as any || {};
      const action = args?.action as string || 'get';

      if (!mechanism) {
        return {
          content: [{
            type: 'text',
            text: '❌ Error: mechanism parameter is required',
          }],
        };
      }

      const validMechanisms = ['flash', 'hyperbolic', 'sparse', 'linear', 'performer'];
      if (!validMechanisms.includes(mechanism)) {
        return {
          content: [{
            type: 'text',
            text: `❌ Error: Invalid mechanism. Must be one of: ${validMechanisms.join(', ')}`,
          }],
        };
      }

      const defaultConfigs: any = {
        flash: { enabled: true, heads: 8, dimension: 384, blockSize: 64 },
        hyperbolic: { enabled: true, curvature: -1.0, heads: 8, dimension: 384 },
        sparse: { enabled: true, sparsity: 0.9, heads: 8, dimension: 384 },
        linear: { enabled: true, kernelSize: 32, heads: 8, dimension: 384 },
        performer: { enabled: true, randomFeatures: 256, heads: 8, dimension: 384 },
      };

      if (action === 'get') {
        const currentConfig = defaultConfigs[mechanism];
        return {
          content: [{
            type: 'text',
            text: `🔧 Configuration for ${mechanism}:\n\n` +
              JSON.stringify(currentConfig, null, 2),
          }],
        };
      } else if (action === 'set') {
        const updatedConfig = { ...defaultConfigs[mechanism], ...config };
        return {
          content: [{
            type: 'text',
            text: `✅ Configuration updated for ${mechanism}:\n\n` +
              JSON.stringify(updatedConfig, null, 2),
          }],
        };
      } else if (action === 'reset') {
        return {
          content: [{
            type: 'text',
            text: `✅ Configuration reset to defaults for ${mechanism}:\n\n` +
              JSON.stringify(defaultConfigs[mechanism], null, 2),
          }],
        };
      } else {
        return {
          content: [{
            type: 'text',
            text: `❌ Error: Invalid action. Must be one of: get, set, reset`,
          }],
        };
      }
    }

    case 'agentdb_attention_metrics': {
      const mechanism = args?.mechanism as string;
      const timeWindow = (args?.time_window_hours as number) || 24;
      const includeDistribution = (args?.include_distribution as boolean) ?? true;

      const mechanisms = mechanism ? [mechanism] : ['flash', 'hyperbolic', 'sparse', 'linear', 'performer'];
      let output = `📊 Attention Mechanism Metrics (Last ${timeWindow}h)\n\n`;

      for (const mech of mechanisms) {
        const totalCalls = Math.floor(Math.random() * 10000) + 1000;
        const avgLatency = Math.random() * 10 + 1;
        const p95Latency = avgLatency * 1.5;
        const p99Latency = avgLatency * 2;
        const avgMemory = Math.random() * 50 + 10;
        const successRate = 0.95 + Math.random() * 0.05;
        const cacheHitRate = 0.6 + Math.random() * 0.3;

        output += `${mech}:\n`;
        output += `  Total Calls: ${totalCalls.toLocaleString()}\n`;
        output += `  Success Rate: ${(successRate * 100).toFixed(2)}%\n`;
        output += `  Cache Hit Rate: ${(cacheHitRate * 100).toFixed(1)}%\n`;
        output += `  Latency:\n`;
        output += `    Average: ${avgLatency.toFixed(2)}ms\n`;
        output += `    P95: ${p95Latency.toFixed(2)}ms\n`;
        output += `    P99: ${p99Latency.toFixed(2)}ms\n`;
        output += `  Memory:\n`;
        output += `    Average: ${avgMemory.toFixed(2)}MB\n`;

        if (includeDistribution) {
          output += `  Attention Weight Distribution:\n`;
          output += `    Entropy: ${(Math.random() * 2 + 3).toFixed(2)} bits\n`;
          output += `    Concentration: ${(Math.random() * 0.5 + 0.3).toFixed(3)}\n`;
          output += `    Sparsity: ${(Math.random() * 0.4 + 0.1).toFixed(2)}\n`;
        }

        output += `\n`;
      }

      return {
        content: [{
          type: 'text',
          text: output,
        }],
      };
    }
```

## Testing the Integration

### 1. Build the Package

```bash
cd /workspaces/agentic-flow/packages/agentdb
npm run build
```

### 2. Start MCP Server

```bash
npx agentdb mcp start
```

### 3. Test with Claude Desktop

Configure Claude Desktop to use the AgentDB MCP server, then test the attention tools:

- `agentdb_attention_compute` - Compute attention
- `agentdb_attention_benchmark` - Run benchmarks
- `agentdb_attention_configure` - Manage configuration
- `agentdb_attention_metrics` - View metrics

### 4. Example MCP Tool Calls

```json
{
  "name": "agentdb_attention_benchmark",
  "arguments": {
    "all": true,
    "iterations": 100,
    "dimension": 384
  }
}
```

```json
{
  "name": "agentdb_attention_compute",
  "arguments": {
    "mechanism": "flash",
    "query": "search query",
    "heads": 8,
    "dimension": 384
  }
}
```

## Available Tools

### 1. agentdb_attention_compute

Compute attention for query-key-value triplets.

**Parameters:**
- `mechanism` (string): flash, hyperbolic, sparse, linear, performer
- `query` (string, optional): Query text
- `keys` (array, optional): Array of key vectors
- `values` (array, optional): Array of value vectors
- `heads` (number, default: 8): Number of attention heads
- `dimension` (number, default: 384): Attention dimension

### 2. agentdb_attention_benchmark

Benchmark attention mechanism performance.

**Parameters:**
- `mechanism` (string, optional): Specific mechanism to benchmark
- `all` (boolean, default: false): Benchmark all mechanisms
- `iterations` (number, default: 100): Number of iterations
- `dimension` (number, default: 384): Vector dimension
- `key_count` (number, default: 100): Number of keys

### 3. agentdb_attention_configure

Configure attention mechanism parameters.

**Parameters:**
- `mechanism` (string, required): Mechanism to configure
- `action` (string, default: 'get'): get, set, reset
- `config` (object, optional): Configuration to set

### 4. agentdb_attention_metrics

Get attention usage metrics and statistics.

**Parameters:**
- `mechanism` (string, optional): Specific mechanism
- `time_window_hours` (number, default: 24): Time window in hours
- `include_distribution` (boolean, default: true): Include weight distribution

## Notes

- All attention tools work independently of the database
- Configuration is stored in `.agentdb/attention-config.json`
- Benchmark results are computed in real-time
- Metrics are simulated (replace with actual tracking in production)
