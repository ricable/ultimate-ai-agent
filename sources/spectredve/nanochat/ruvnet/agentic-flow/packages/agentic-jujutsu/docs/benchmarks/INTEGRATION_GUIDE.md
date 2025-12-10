# Integration Guide
## Connecting Benchmarking System to Existing Infrastructure

**Version:** 1.0.0
**Date:** 2025-11-09

---

## Overview

This guide describes how to integrate the benchmarking system with the existing agentic-jujutsu codebase, Claude Flow coordination, and AgentDB learning infrastructure.

---

## Integration Points

### 1. Existing Code Integration

#### Hooks System Integration

**Location:** `/workspaces/agentic-flow/packages/agentic-jujutsu/src/hooks.rs`

The benchmarking system integrates with the existing hooks infrastructure:

```typescript
// benchmarks/framework/hooks-integration.ts
import { execSync } from 'child_process';

export class BenchmarkHooks {
  private sessionId: string;

  constructor(sessionId?: string) {
    this.sessionId = sessionId || `benchmark-${Date.now()}`;
  }

  async preTask(description: string): Promise<void> {
    execSync(
      `npx claude-flow@alpha hooks pre-task --description "${description}"`,
      { stdio: 'inherit' }
    );
  }

  async postEdit(file: string, memoryKey: string): Promise<void> {
    execSync(
      `npx claude-flow@alpha hooks post-edit --file "${file}" --memory-key "${memoryKey}"`,
      { stdio: 'inherit' }
    );
  }

  async postTask(taskId: string): Promise<void> {
    execSync(
      `npx claude-flow@alpha hooks post-task --task-id "${taskId}"`,
      { stdio: 'inherit' }
    );
  }

  async sessionRestore(): Promise<void> {
    execSync(
      `npx claude-flow@alpha hooks session-restore --session-id "${this.sessionId}"`,
      { stdio: 'inherit' }
    );
  }

  async sessionEnd(exportMetrics: boolean = true): Promise<void> {
    execSync(
      `npx claude-flow@alpha hooks session-end --export-metrics ${exportMetrics}`,
      { stdio: 'inherit' }
    );
  }

  async notify(message: string): Promise<void> {
    execSync(
      `npx claude-flow@alpha hooks notify --message "${message}"`,
      { stdio: 'inherit' }
    );
  }
}
```

**Usage in Benchmark Runner:**
```typescript
// benchmarks/framework/runner.ts
import { BenchmarkHooks } from './hooks-integration';

export class BenchmarkRunner {
  private hooks: BenchmarkHooks;

  async run(config: BenchmarkConfig): Promise<void> {
    this.hooks = new BenchmarkHooks(config.agentdb.sessionId);

    // Pre-task hook
    await this.hooks.preTask(
      `benchmark-suite: ${config.tests.suites.join(', ')}`
    );

    try {
      // Restore previous session state
      await this.hooks.sessionRestore();

      // Run benchmarks
      for (const suite of config.tests.suites) {
        await this.runSuite(suite);

        // Notify progress
        await this.hooks.notify(`Completed suite: ${suite}`);
      }

      // Save results
      await this.saveResults();

      // Post-task hook
      await this.hooks.postTask('benchmark-run');

    } finally {
      // End session with metrics export
      await this.hooks.sessionEnd(true);
    }
  }
}
```

---

### 2. AgentDB Integration

**Location:** `/workspaces/agentic-flow/packages/agentic-jujutsu/src/agentdb_sync.rs`

#### Pattern Storage

```typescript
// benchmarks/analysis/agentdb-learning.ts
import { execSync } from 'child_process';

export class AgentDBLearning {
  async storePattern(result: ProcessedResult): Promise<void> {
    const pattern: BenchmarkPattern = {
      sessionId: `benchmark-${result.testId}`,
      task: `${result.operation}-${result.context.vcsType}-benchmark`,
      input: JSON.stringify({
        vcsType: result.context.vcsType,
        operation: result.operation,
        repoSize: result.context.repoSize,
        repoMetadata: result.context.repoMetadata
      }),
      output: JSON.stringify({
        executionTime: result.metrics.executionTime,
        memoryPeak: result.metrics.memoryPeak,
        ioOperations: result.metrics.ioOperations,
        success: result.quality.success
      }),
      reward: this.calculateReward(result),
      critique: this.generateCritique(result),
      latencyMs: result.metrics.executionTime,
      tokensUsed: 0,
      success: result.quality.success
    };

    // Store via CLI (uses AgentDB from agentic-jujutsu)
    const cmd = `npx agentdb-cli pattern-store \
      --session-id "${pattern.sessionId}" \
      --task "${pattern.task}" \
      --input '${pattern.input}' \
      --output '${pattern.output}' \
      --reward ${pattern.reward} \
      --success ${pattern.success}`;

    execSync(cmd, { stdio: 'inherit' });
  }

  async searchPatterns(operation: string, k: number = 10): Promise<Pattern[]> {
    const cmd = `npx agentdb-cli pattern-search \
      --task "${operation}-benchmark" \
      --k ${k} \
      --only-successes true`;

    const output = execSync(cmd, { encoding: 'utf-8' });
    return JSON.parse(output);
  }

  private calculateReward(result: ProcessedResult): number {
    // Multi-factor reward calculation
    const factors = {
      // Speed factor (faster = better)
      speed: this.normalizeSpeed(result.metrics.executionTime),

      // Memory efficiency (lower = better)
      memory: 1 - this.normalizeMemory(result.metrics.memoryPeak),

      // Reliability (success/failure)
      reliability: result.quality.success ? 1 : 0,

      // I/O efficiency
      ioEfficiency: this.calculateIOEfficiency(result.metrics.ioOperations)
    };

    // Weighted average
    return (
      factors.speed * 0.4 +
      factors.memory * 0.3 +
      factors.reliability * 0.2 +
      factors.ioEfficiency * 0.1
    );
  }

  private generateCritique(result: ProcessedResult): string {
    const parts: string[] = [];

    // Performance assessment
    if (result.comparison.jjVsGit.speedupFactor > 2) {
      parts.push(`Excellent performance: ${result.comparison.jjVsGit.speedupFactor.toFixed(1)}x faster than Git.`);
    } else if (result.comparison.jjVsGit.speedupFactor > 1) {
      parts.push(`Good performance: ${result.comparison.jjVsGit.speedupFactor.toFixed(1)}x faster than Git.`);
    } else {
      parts.push(`Slower than Git: ${(1 / result.comparison.jjVsGit.speedupFactor).toFixed(1)}x slower.`);
    }

    // Memory assessment
    if (result.comparison.jjVsGit.memoryRatio < 0.8) {
      parts.push(`Memory efficient: ${(result.comparison.jjVsGit.memoryRatio * 100).toFixed(0)}% of Git's usage.`);
    }

    // Recommendations
    if (result.insights.bottlenecks.length > 0) {
      parts.push(`Found ${result.insights.bottlenecks.length} optimization opportunities.`);
    }

    return parts.join(' ');
  }

  private normalizeSpeed(executionTimeMs: number): number {
    // Normalize to 0-1 range (faster = higher reward)
    // Assume 10s is "slow", 100ms is "fast"
    const maxTime = 10000;
    const minTime = 100;
    return Math.max(0, Math.min(1, 1 - (executionTimeMs - minTime) / (maxTime - minTime)));
  }

  private normalizeMemory(memoryBytes: number): number {
    // Normalize to 0-1 range
    // Assume 2GB is "high", 64MB is "low"
    const maxMemory = 2 * 1024 * 1024 * 1024;
    const minMemory = 64 * 1024 * 1024;
    return Math.max(0, Math.min(1, (memoryBytes - minMemory) / (maxMemory - minMemory)));
  }

  private calculateIOEfficiency(ioOps: IOMetrics): number {
    // Ratio of useful I/O to total I/O
    const totalOps = ioOps.readOps + ioOps.writeOps;
    const totalBytes = ioOps.readBytes + ioOps.writeBytes;

    if (totalOps === 0) return 0;

    // Higher bytes per operation = more efficient
    const bytesPerOp = totalBytes / totalOps;

    // Normalize (4KB per op is average, 64KB is good)
    const avgBytesPerOp = 4 * 1024;
    const goodBytesPerOp = 64 * 1024;

    return Math.max(0, Math.min(1, (bytesPerOp - avgBytesPerOp) / (goodBytesPerOp - avgBytesPerOp)));
  }
}
```

---

### 3. WASM Integration

**Location:** `/workspaces/agentic-flow/packages/agentic-jujutsu/src/wasm.rs`

The benchmarking system can leverage WASM for browser-based benchmarking:

```typescript
// benchmarks/framework/wasm-runner.ts
import { JJWrapper } from '@agentic-flow/jujutsu';

export class WASMBenchmarkRunner {
  private jj: JJWrapper;

  async initialize(): Promise<void> {
    this.jj = await JJWrapper.new();
  }

  async benchmarkOperation(operation: string): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const startMemory = (performance as any).memory?.usedJSHeapSize || 0;

    // Execute operation via WASM
    const result = await this.executeOperation(operation);

    const endTime = performance.now();
    const endMemory = (performance as any).memory?.usedJSHeapSize || 0;

    return {
      operation,
      executionTime: endTime - startTime,
      memoryDelta: endMemory - startMemory,
      success: result.exitCode === 0,
      output: result.stdout
    };
  }

  private async executeOperation(operation: string): Promise<any> {
    switch (operation) {
      case 'status':
        return await this.jj.status();
      case 'log':
        return await this.jj.log();
      // ... other operations
      default:
        throw new Error(`Unknown operation: ${operation}`);
    }
  }
}
```

---

### 4. Existing Test Infrastructure

**Location:** `/workspaces/agentic-flow/packages/agentic-jujutsu/tests/`

Integration with existing test suites:

```typescript
// tests/integration/benchmark-integration.test.ts
import { BenchmarkRunner } from '../../benchmarks/framework/runner';
import { expect } from 'chai';

describe('Benchmark Integration', () => {
  it('should execute basic benchmark suite', async () => {
    const runner = new BenchmarkRunner({
      id: 'test-benchmark',
      name: 'Integration Test Benchmark',
      execution: {
        iterations: 3,
        warmupRuns: 1,
        timeout: 30000,
        parallelism: 2
      },
      tests: {
        suites: ['basic-operations']
      },
      environment: {
        docker: {
          baseImages: {
            jujutsu: 'benchmark-jujutsu:test',
            git: 'benchmark-git:test',
            gitWorktrees: 'benchmark-git-worktrees:test'
          },
          resources: {
            cpus: 1,
            memory: 1024 * 1024 * 1024 // 1GB
          },
          network: 'none',
          volumes: []
        }
      },
      dataGeneration: {
        repoSizes: ['small'],
        cacheFixtures: true
      },
      output: {
        formats: ['json'],
        destination: './test-results',
        archiveResults: false,
        compression: false
      },
      analysis: {
        statisticalSignificance: 0.05,
        confidenceLevel: 0.95,
        outlierDetection: true,
        trendAnalysis: false
      },
      agentdb: {
        enabled: false,
        sessionId: 'test-session',
        storePatterns: false,
        queryHistorical: false
      }
    });

    const results = await runner.run();

    expect(results).to.have.length.greaterThan(0);
    expect(results[0].quality.success).to.be.true;
  });
});
```

---

### 5. Claude Flow Coordination

**Integration with MCP Tools:**

```typescript
// benchmarks/framework/swarm-coordination.ts
export class SwarmCoordination {
  async initializeBenchmarkSwarm(
    config: BenchmarkConfig
  ): Promise<string> {
    // Initialize swarm via MCP
    const swarmId = await mcp.swarm_init({
      topology: 'mesh',
      maxAgents: config.execution.parallelism,
      strategy: 'balanced'
    });

    // Spawn analyzer agents
    await mcp.agent_spawn({
      type: 'researcher',
      name: 'performance-analyzer',
      swarmId
    });

    await mcp.agent_spawn({
      type: 'analyst',
      name: 'statistical-analyzer',
      swarmId
    });

    await mcp.agent_spawn({
      type: 'optimizer',
      name: 'bottleneck-detector',
      swarmId
    });

    return swarmId;
  }

  async orchestrateBenchmark(
    swarmId: string,
    suite: TestSuite
  ): Promise<void> {
    await mcp.task_orchestrate({
      task: `Execute benchmark suite: ${suite.name}`,
      strategy: 'parallel',
      priority: 'high',
      dependencies: suite.tests.map(t => ({
        id: t.id,
        dependsOn: t.dependsOn || []
      }))
    });
  }

  async monitorProgress(swarmId: string): Promise<SwarmStatus> {
    return await mcp.swarm_status({ swarmId });
  }
}
```

---

## Configuration Files

### Docker Compose Integration

```yaml
# benchmarks/docker/docker-compose.yml
version: '3.8'

services:
  jujutsu-bench:
    build:
      context: .
      dockerfile: Dockerfile.jujutsu
    environment:
      - BENCHMARK_MODE=true
      - AGENTDB_PATH=/data/agentdb.sqlite
    volumes:
      - ./data:/data
      - ./results:/results
      - ../../src:/app/src:ro  # Mount existing source
    networks:
      - benchmark-net
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G

  git-bench:
    build:
      context: .
      dockerfile: Dockerfile.git
    environment:
      - BENCHMARK_MODE=true
    volumes:
      - ./data:/data
      - ./results:/results
    networks:
      - benchmark-net
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G

  metrics-collector:
    build:
      context: .
      dockerfile: Dockerfile.metrics
    volumes:
      - ./results:/results
      - ../../.claude-flow/metrics:/metrics  # Existing metrics dir
    depends_on:
      - jujutsu-bench
      - git-bench

networks:
  benchmark-net:
    driver: bridge
```

### Package.json Integration

```json
{
  "scripts": {
    "bench": "npm run build && node benchmarks/framework/cli.js",
    "bench:run": "npm run bench -- run --config benchmarks/config/default.json",
    "bench:analyze": "npm run bench -- analyze --results benchmarks/results/latest.json",
    "bench:report": "npm run bench -- report --format markdown",
    "bench:docker": "docker-compose -f benchmarks/docker/docker-compose.yml up",
    "bench:clean": "rm -rf benchmarks/results/* benchmarks/data/fixtures/*"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "typescript": "^5.0.0",
    "tsx": "^4.0.0"
  }
}
```

---

## CLI Integration

### Main Benchmark CLI

```typescript
#!/usr/bin/env node
// benchmarks/framework/cli.ts

import { Command } from 'commander';
import { BenchmarkRunner } from './runner';
import { ReportGenerator } from './reporter';
import { readFile } from 'fs/promises';

const program = new Command();

program
  .name('benchmark')
  .description('Agentic Jujutsu Benchmarking CLI')
  .version('1.0.0');

program
  .command('run')
  .description('Run benchmarks')
  .option('-c, --config <path>', 'Config file path')
  .option('-s, --suite <name>', 'Specific suite to run')
  .option('--docker', 'Run in Docker containers')
  .action(async (options) => {
    const config = JSON.parse(
      await readFile(options.config, 'utf-8')
    );

    if (options.suite) {
      config.tests.suites = [options.suite];
    }

    const runner = new BenchmarkRunner(config);
    await runner.run();

    console.log('Benchmarks completed!');
  });

program
  .command('analyze')
  .description('Analyze benchmark results')
  .requiredOption('-r, --results <path>', 'Results file path')
  .action(async (options) => {
    const results = JSON.parse(
      await readFile(options.results, 'utf-8')
    );

    const analyzer = new AnalysisPipeline();
    const processed = await analyzer.process(results);

    console.log('Analysis complete!');
    console.log(JSON.stringify(processed, null, 2));
  });

program
  .command('report')
  .description('Generate benchmark report')
  .requiredOption('-r, --results <path>', 'Results file path')
  .option('-f, --format <type>', 'Report format (markdown|json|html)', 'markdown')
  .option('-o, --output <path>', 'Output file path')
  .action(async (options) => {
    const results = JSON.parse(
      await readFile(options.results, 'utf-8')
    );

    const generator = new ReportGenerator();
    const report = await generator.generate(results, {
      format: options.format
    });

    if (options.output) {
      await writeFile(options.output, report);
    } else {
      console.log(report);
    }
  });

program.parse();
```

---

## GitHub Actions Integration

```yaml
# .github/workflows/benchmarks.yml
name: Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    # Run daily at midnight UTC
    - cron: '0 0 * * *'

jobs:
  benchmark:
    runs-on: ubuntu-latest
    timeout-minutes: 120

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install dependencies
        run: |
          cd packages/agentic-jujutsu
          npm install

      - name: Build WASM
        run: |
          cd packages/agentic-jujutsu
          npm run build

      - name: Run benchmarks
        run: |
          cd packages/agentic-jujutsu
          npm run bench:run

      - name: Generate report
        run: |
          cd packages/agentic-jujutsu
          npm run bench:report -- --output benchmarks/results/report.md

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: packages/agentic-jujutsu/benchmarks/results/

      - name: Comment PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync(
              'packages/agentic-jujutsu/benchmarks/results/report.md',
              'utf-8'
            );

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## 📊 Benchmark Results\n\n${report}`
            });
```

---

## Summary

The benchmarking system integrates seamlessly with:

1. **Existing Rust code**: Uses JJWrapper, hooks, and AgentDB sync
2. **Hooks system**: Pre/post task coordination
3. **AgentDB**: Pattern storage and learning
4. **WASM**: Browser-based benchmarking
5. **Test infrastructure**: Integration tests
6. **Claude Flow**: Swarm coordination
7. **CI/CD**: GitHub Actions automation

All integration points are designed to be:
- ✅ **Non-invasive**: Doesn't modify existing code
- ✅ **Modular**: Can be used independently or together
- ✅ **Extensible**: Easy to add new integrations
- ✅ **Observable**: Full hooks and metrics integration

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-09
