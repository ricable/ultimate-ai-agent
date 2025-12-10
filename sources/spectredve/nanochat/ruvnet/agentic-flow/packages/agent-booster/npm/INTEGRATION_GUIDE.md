# Agent Booster Integration Guide

Complete guide for integrating Agent Booster into your projects.

## Quick Start

### 1. Installation

```bash
# Install the SDK
npm install agent-booster

# Or install globally for CLI
npm install -g agent-booster-cli
```

### 2. Basic Usage

```javascript
const agentBooster = require('agent-booster');

async function optimizePrompt() {
  const prompt = 'Your verbose prompt here...';
  const optimized = await agentBooster.apply(prompt);
  console.log(optimized);
}

optimizePrompt();
```

### 3. Check Runtime

```javascript
const runtime = agentBooster.getRuntime();
console.log(`Using: ${runtime}`); // 'native' or 'wasm'
```

## Integration Patterns

### Pattern 1: Express.js Middleware

Optimize prompts in API endpoints:

```javascript
const express = require('express');
const agentBooster = require('agent-booster');

const app = express();

// Middleware to optimize prompts
async function optimizePromptMiddleware(req, res, next) {
  if (req.body.prompt) {
    try {
      req.body.optimizedPrompt = await agentBooster.apply(req.body.prompt, {
        strategy: 'balanced'
      });
      next();
    } catch (error) {
      res.status(500).json({ error: 'Prompt optimization failed' });
    }
  } else {
    next();
  }
}

app.post('/api/chat', optimizePromptMiddleware, async (req, res) => {
  // Use req.body.optimizedPrompt for LLM call
  const response = await callLLM(req.body.optimizedPrompt);
  res.json({ response });
});

app.listen(3000);
```

### Pattern 2: Batch Processing Service

Process multiple prompts efficiently:

```javascript
const agentBooster = require('agent-booster');

class PromptOptimizationService {
  constructor() {
    this.queue = [];
    this.processing = false;
  }

  async add(prompt) {
    return new Promise((resolve) => {
      this.queue.push({ prompt, resolve });
      this.processQueue();
    });
  }

  async processQueue() {
    if (this.processing || this.queue.length === 0) return;

    this.processing = true;

    // Process in batches of 10
    while (this.queue.length > 0) {
      const batch = this.queue.splice(0, 10);
      const prompts = batch.map(item => item.prompt);

      try {
        const optimized = await agentBooster.batchApply(prompts);

        // Resolve all promises
        batch.forEach((item, idx) => {
          item.resolve(optimized[idx]);
        });
      } catch (error) {
        // Handle errors
        batch.forEach(item => item.resolve(null));
      }
    }

    this.processing = false;
  }
}

// Usage
const service = new PromptOptimizationService();

async function handleRequest(prompt) {
  const optimized = await service.add(prompt);
  return optimized;
}
```

### Pattern 3: CLI Integration

Use as a pre-processing step:

```bash
#!/bin/bash
# optimize-and-run.sh

# Optimize prompts from file
agent-booster batch prompts.txt -o optimized.txt

# Use optimized prompts with your tool
your-llm-tool --prompts optimized.txt
```

### Pattern 4: Real-time Stream Processing

Optimize prompts in real-time:

```javascript
const agentBooster = require('agent-booster');
const { Transform } = require('stream');

class PromptOptimizationStream extends Transform {
  constructor(options = {}) {
    super({ objectMode: true });
    this.options = options;
  }

  async _transform(chunk, encoding, callback) {
    try {
      const optimized = await agentBooster.apply(chunk.prompt, this.options);
      callback(null, { ...chunk, optimized });
    } catch (error) {
      callback(error);
    }
  }
}

// Usage with streams
const { pipeline } = require('stream');
const fs = require('fs');

pipeline(
  fs.createReadStream('prompts.jsonl'),
  new PromptOptimizationStream({ strategy: 'balanced' }),
  fs.createWriteStream('optimized.jsonl'),
  (error) => {
    if (error) console.error('Pipeline failed:', error);
    else console.log('Pipeline succeeded');
  }
);
```

### Pattern 5: Worker Thread Pool

Parallel processing with workers:

```javascript
const { Worker } = require('worker_threads');
const path = require('path');

class PromptOptimizationPool {
  constructor(workerCount = 4) {
    this.workers = [];
    this.queue = [];
    this.activeWorkers = new Set();

    for (let i = 0; i < workerCount; i++) {
      const worker = new Worker(path.join(__dirname, 'optimization-worker.js'));
      this.workers.push(worker);
      this.setupWorker(worker);
    }
  }

  setupWorker(worker) {
    worker.on('message', ({ id, result }) => {
      const task = this.activeWorkers.get(id);
      if (task) {
        task.resolve(result);
        this.activeWorkers.delete(id);
        this.processQueue();
      }
    });

    worker.on('error', (error) => {
      console.error('Worker error:', error);
    });
  }

  async optimize(prompt, options = {}) {
    return new Promise((resolve, reject) => {
      const id = Math.random().toString(36);
      this.queue.push({ id, prompt, options, resolve, reject });
      this.processQueue();
    });
  }

  processQueue() {
    const availableWorker = this.workers.find(w => !this.activeWorkers.has(w));
    const task = this.queue.shift();

    if (availableWorker && task) {
      this.activeWorkers.set(availableWorker, task);
      availableWorker.postMessage({
        id: task.id,
        prompt: task.prompt,
        options: task.options
      });
    }
  }
}

// optimization-worker.js
const { parentPort } = require('worker_threads');
const agentBooster = require('agent-booster');

parentPort.on('message', async ({ id, prompt, options }) => {
  try {
    const result = await agentBooster.apply(prompt, options);
    parentPort.postMessage({ id, result });
  } catch (error) {
    parentPort.postMessage({ id, error: error.message });
  }
});
```

## Framework Integrations

### Next.js API Routes

```javascript
// pages/api/optimize.js
import agentBooster from 'agent-booster';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { prompt, strategy = 'balanced' } = req.body;
    const optimized = await agentBooster.apply(prompt, { strategy });

    res.status(200).json({ optimized });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
}
```

### NestJS Service

```typescript
// optimization.service.ts
import { Injectable } from '@nestjs/common';
import * as agentBooster from 'agent-booster';
import type { OptimizationOptions } from 'agent-booster';

@Injectable()
export class OptimizationService {
  async optimizePrompt(
    prompt: string,
    options?: OptimizationOptions
  ): Promise<string> {
    return agentBooster.apply(prompt, options);
  }

  async optimizeBatch(
    prompts: string[],
    options?: OptimizationOptions
  ): Promise<string[]> {
    return agentBooster.batchApply(prompts, options);
  }

  async analyzePrompt(prompt: string) {
    return agentBooster.analyze(prompt);
  }

  getSystemInfo() {
    return agentBooster.getVersion();
  }
}
```

### AWS Lambda

```javascript
// lambda/optimize.js
const agentBooster = require('agent-booster');

exports.handler = async (event) => {
  try {
    const { prompt, strategy = 'balanced' } = JSON.parse(event.body);

    const optimized = await agentBooster.apply(prompt, { strategy });

    return {
      statusCode: 200,
      body: JSON.stringify({ optimized })
    };
  } catch (error) {
    return {
      statusCode: 500,
      body: JSON.stringify({ error: error.message })
    };
  }
};
```

### Cloudflare Workers

```javascript
// worker.js
import agentBooster from 'agent-booster';

addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request));
});

async function handleRequest(request) {
  if (request.method !== 'POST') {
    return new Response('Method not allowed', { status: 405 });
  }

  try {
    const { prompt, strategy = 'balanced' } = await request.json();
    const optimized = await agentBooster.apply(prompt, { strategy });

    return new Response(JSON.stringify({ optimized }), {
      headers: { 'Content-Type': 'application/json' }
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}
```

## Performance Optimization

### 1. Pre-initialize Runtime

```javascript
// At application startup
const agentBooster = require('agent-booster');
agentBooster.initialize(); // Load runtime once

// Later in your application
const optimized = await agentBooster.apply(prompt); // Fast!
```

### 2. Use Batch Processing

```javascript
// ❌ Slow: Sequential processing
for (const prompt of prompts) {
  results.push(await agentBooster.apply(prompt));
}

// ✅ Fast: Batch processing
const results = await agentBooster.batchApply(prompts);
```

### 3. Cache Results

```javascript
const cache = new Map();

async function optimizeWithCache(prompt, options = {}) {
  const cacheKey = JSON.stringify({ prompt, options });

  if (cache.has(cacheKey)) {
    return cache.get(cacheKey);
  }

  const optimized = await agentBooster.apply(prompt, options);
  cache.set(cacheKey, optimized);

  return optimized;
}
```

### 4. Use Workers for Heavy Loads

See Worker Thread Pool pattern above for parallel processing.

## Error Handling

### Graceful Degradation

```javascript
async function robustOptimization(prompt) {
  try {
    return await agentBooster.apply(prompt);
  } catch (error) {
    // Log error
    console.error('Optimization failed:', error.message);

    // Fall back to original prompt
    return prompt;
  }
}
```

### Retry Logic

```javascript
async function optimizeWithRetry(prompt, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await agentBooster.apply(prompt);
    } catch (error) {
      if (i === maxRetries - 1) throw error;

      // Wait before retry
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
    }
  }
}
```

### Circuit Breaker

```javascript
class CircuitBreaker {
  constructor(threshold = 5, timeout = 60000) {
    this.failureCount = 0;
    this.threshold = threshold;
    this.timeout = timeout;
    this.state = 'CLOSED';
    this.nextAttempt = Date.now();
  }

  async execute(fn) {
    if (this.state === 'OPEN') {
      if (Date.now() < this.nextAttempt) {
        throw new Error('Circuit breaker is OPEN');
      }
      this.state = 'HALF_OPEN';
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  onSuccess() {
    this.failureCount = 0;
    this.state = 'CLOSED';
  }

  onFailure() {
    this.failureCount++;
    if (this.failureCount >= this.threshold) {
      this.state = 'OPEN';
      this.nextAttempt = Date.now() + this.timeout;
    }
  }
}

const breaker = new CircuitBreaker();

async function safeOptimize(prompt) {
  return breaker.execute(() => agentBooster.apply(prompt));
}
```

## Monitoring & Observability

### Add Metrics

```javascript
const prometheus = require('prom-client');

const optimizationDuration = new prometheus.Histogram({
  name: 'prompt_optimization_duration_ms',
  help: 'Duration of prompt optimization in milliseconds'
});

const optimizationCounter = new prometheus.Counter({
  name: 'prompt_optimizations_total',
  help: 'Total number of prompt optimizations',
  labelNames: ['runtime', 'strategy']
});

async function observedOptimize(prompt, options = {}) {
  const timer = optimizationDuration.startTimer();
  const runtime = agentBooster.getRuntime();

  try {
    const result = await agentBooster.apply(prompt, options);

    optimizationCounter.inc({
      runtime: runtime || 'unknown',
      strategy: options.strategy || 'balanced'
    });

    return result;
  } finally {
    timer();
  }
}
```

### Structured Logging

```javascript
const winston = require('winston');

const logger = winston.createLogger({
  format: winston.format.json(),
  transports: [new winston.transports.Console()]
});

async function loggedOptimize(prompt, options = {}) {
  const startTime = Date.now();
  const runtime = agentBooster.getRuntime();

  logger.info('Starting optimization', {
    promptLength: prompt.length,
    strategy: options.strategy,
    runtime
  });

  try {
    const result = await agentBooster.apply(prompt, options);
    const duration = Date.now() - startTime;

    logger.info('Optimization completed', {
      duration,
      originalLength: prompt.length,
      optimizedLength: result.length,
      savings: prompt.length - result.length
    });

    return result;
  } catch (error) {
    logger.error('Optimization failed', {
      error: error.message,
      duration: Date.now() - startTime
    });
    throw error;
  }
}
```

## Testing

### Unit Tests

```javascript
const assert = require('assert');
const agentBooster = require('agent-booster');

describe('Prompt Optimization', () => {
  it('should optimize a prompt', async () => {
    const prompt = 'Test prompt';
    const result = await agentBooster.apply(prompt);

    assert(typeof result === 'string');
    assert(result.length > 0);
  });

  it('should handle batch processing', async () => {
    const prompts = ['Test 1', 'Test 2'];
    const results = await agentBooster.batchApply(prompts);

    assert(Array.isArray(results));
    assert(results.length === prompts.length);
  });

  it('should analyze prompts', async () => {
    const prompt = 'Test prompt';
    const analysis = await agentBooster.analyze(prompt);

    assert(typeof analysis.originalTokens === 'number');
    assert(typeof analysis.optimizedTokens === 'number');
    assert(typeof analysis.savings === 'number');
  });
});
```

### Integration Tests

```javascript
const request = require('supertest');
const app = require('./app');

describe('Optimization API', () => {
  it('POST /api/optimize should optimize prompt', async () => {
    const response = await request(app)
      .post('/api/optimize')
      .send({ prompt: 'Test prompt', strategy: 'balanced' })
      .expect(200);

    assert(response.body.optimized);
  });
});
```

## Best Practices

1. **Pre-initialize at startup** - Load runtime once when application starts
2. **Use batch processing** - Process multiple prompts together for better performance
3. **Cache results** - Avoid re-optimizing the same prompts
4. **Handle errors gracefully** - Fall back to original prompt if optimization fails
5. **Monitor performance** - Track optimization duration and success rates
6. **Use appropriate strategy** - Balance between compression and quality
7. **Set timeouts** - Prevent long-running optimizations from blocking
8. **Test with real data** - Validate optimizations with production prompts

## Troubleshooting

See [README.md](./README.md#troubleshooting) for common issues and solutions.

## Support

- [Documentation](./README.md)
- [GitHub Issues](https://github.com/yourusername/agentic-flow/issues)
- [Examples](./agent-booster/examples.js)
