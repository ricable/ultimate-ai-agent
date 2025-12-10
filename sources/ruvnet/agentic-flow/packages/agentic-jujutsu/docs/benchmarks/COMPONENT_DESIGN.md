# Component Interaction Design
## Benchmarking System Components

**Version:** 1.0.0
**Date:** 2025-11-09

---

## Component Interaction Diagram

```
                              ┌──────────────────────────────────┐
                              │  Benchmark Orchestrator (Main)   │
                              │  • Task scheduling               │
                              │  • Resource allocation           │
                              │  • Result aggregation            │
                              └────────┬─────────────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         │                             │                             │
         ↓                             ↓                             ↓
┌────────────────┐          ┌────────────────┐          ┌────────────────┐
│ Test Planner   │          │ Data Generator │          │ Test Executor  │
│                │          │                │          │                │
│ • Load suites  │→────────→│ • Create repos │→────────→│ • Run Docker   │
│ • Validate     │          │ • Gen commits  │          │ • Collect data │
│ • Schedule     │          │ • Gen branches │          │ • Monitor      │
└────────────────┘          └────────────────┘          └────────┬───────┘
                                                                  │
                                                                  ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                     Docker Environment Coordinator                        │
│  ┌─────────────┐      ┌─────────────┐      ┌──────────────┐             │
│  │  Jujutsu    │      │     Git     │      │ Git-Worktrees│             │
│  │  Container  │      │  Container  │      │  Container   │             │
│  │             │      │             │      │              │             │
│  │ ┌─────────┐ │      │ ┌─────────┐ │      │ ┌──────────┐ │             │
│  │ │ Profiler│ │      │ │ Profiler│ │      │ │ Profiler │ │             │
│  │ └─────────┘ │      │ └─────────┘ │      │ └──────────┘ │             │
│  └──────┬──────┘      └──────┬──────┘      └──────┬───────┘             │
└─────────┼─────────────────────┼──────────────────────┼──────────────────┘
          │                     │                      │
          └─────────────────────┼──────────────────────┘
                                ↓
                   ┌────────────────────────┐
                   │ Metrics Collector Hub  │
                   │ • Aggregate metrics    │
                   │ • Normalize data       │
                   │ • Timestamp sync       │
                   └────────┬───────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ↓                  ↓                  ↓
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│  Performance   │ │ Code Quality   │ │   Security     │
│  Profiler      │ │   Analyzer     │ │   Scanner      │
│                │ │                │ │                │
│ • CPU/Memory   │ │ • Complexity   │ │ • Vuln scan    │
│ • I/O ops      │ │ • Coverage     │ │ • Secret leak  │
│ • Latency      │ │ • Tech debt    │ │ • Best practice│
└────────┬───────┘ └────────┬───────┘ └────────┬───────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            ↓
                   ┌────────────────────────┐
                   │ Statistical Engine     │
                   │ • T-tests              │
                   │ • Regression           │
                   │ • Outlier detection    │
                   └────────┬───────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ↓                  ↓                  ↓
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ Speed Optimizer│ │ AgentDB Store  │ │ Report Gen     │
│                │ │                │ │                │
│ • Bottlenecks  │ │ • Patterns     │ │ • Markdown     │
│ • Hot paths    │ │ • Learning     │ │ • JSON         │
│ • A/B tests    │ │ • Predictions  │ │ • HTML         │
└────────────────┘ └────────────────┘ └────────────────┘
```

---

## Detailed Component Interactions

### 1. Benchmark Orchestrator

**Purpose:** Central controller that manages the entire benchmark lifecycle

**Interactions:**
```typescript
class BenchmarkOrchestrator {
  // Inputs
  receive(config: BenchmarkConfig): void
  receive(suites: TestSuite[]): void

  // Outputs
  emit(event: 'started', data: { suites: string[] }): void
  emit(event: 'progress', data: { completed: number; total: number }): void
  emit(event: 'completed', data: { results: ProcessedResult[] }): void
  emit(event: 'error', error: Error): void

  // Component Coordination
  async coordinate(): Promise<void> {
    // 1. Plan tests
    const plan = await this.testPlanner.createPlan(this.config);

    // 2. Generate data
    const fixtures = await this.dataGenerator.generate(plan.repoSizes);

    // 3. Execute tests
    const rawResults = await this.testExecutor.runAll(plan, fixtures);

    // 4. Analyze results
    const processed = await this.analyzer.process(rawResults);

    // 5. Store patterns
    await this.agentdb.store(processed);

    // 6. Generate reports
    await this.reporter.generate(processed);
  }
}
```

**Dependencies:**
- Test Planner
- Data Generator
- Test Executor
- Analysis Pipeline
- AgentDB
- Report Generator

**State Management:**
```typescript
interface OrchestratorState {
  phase: 'idle' | 'planning' | 'generating' | 'executing' | 'analyzing' | 'reporting';
  currentSuite: string;
  completedTests: number;
  totalTests: number;
  errors: Error[];
  startTime: number;
  estimatedEndTime: number;
}
```

---

### 2. Test Planner

**Purpose:** Creates optimized test execution plan

**Interactions:**
```typescript
class TestPlanner {
  // Inputs
  async loadSuites(pattern: string): Promise<TestSuite[]>
  async validateSuite(suite: TestSuite): Promise<ValidationResult>

  // Outputs
  async createPlan(config: BenchmarkConfig): Promise<ExecutionPlan>

  // Optimization Logic
  private optimizePlan(tests: Test[]): Test[] {
    // 1. Group by resource requirements
    const grouped = this.groupByResources(tests);

    // 2. Parallelize independent tests
    const parallelGroups = this.identifyParallelGroups(grouped);

    // 3. Order by execution time (longest first)
    return this.orderByDuration(parallelGroups);
  }
}

interface ExecutionPlan {
  totalTests: number;
  estimatedDuration: number;
  parallelGroups: TestGroup[];
  requiredResources: ResourceRequirements;
  dependencies: DependencyGraph;
}
```

**Communication Pattern:**
```
User Config → TestPlanner.loadSuites()
           → TestPlanner.validateSuite()
           → TestPlanner.createPlan()
           → ExecutionPlan → Orchestrator
```

---

### 3. Data Generator

**Purpose:** Create realistic repository fixtures

**Interactions:**
```typescript
class DataGenerator {
  // Inputs
  async generate(spec: RepoSpec): Promise<RepositoryFixture>

  // Output
  return {
    path: string;
    metadata: {
      commits: number;
      branches: number;
      files: number;
      totalSize: number;
    }
  }

  // Generation Strategy
  async generateRepo(spec: RepoSpec): Promise<void> {
    // 1. Initialize repository
    await this.initRepo(spec.vcsType);

    // 2. Generate file tree
    await this.generateFiles(spec.fileCount);

    // 3. Create commit history
    await this.generateCommits(spec.commitCount);

    // 4. Create branches
    await this.generateBranches(spec.branchCount);

    // 5. Simulate realistic patterns
    await this.simulateWorkflow(spec.workflow);
  }
}
```

**Data Flow:**
```
RepoSpec → CodeGenerator → FileTree
        → CommitGenerator → CommitHistory
        → BranchGenerator → BranchStructure
        → RepositoryFixture
```

---

### 4. Test Executor

**Purpose:** Run benchmarks in Docker containers

**Interactions:**
```typescript
class TestExecutor {
  private dockerManager: DockerManager;
  private metricsCollector: MetricsCollector;

  // Execute single test
  async executeTest(test: Test, fixture: RepositoryFixture): Promise<RawMetric> {
    // 1. Start container
    const container = await this.dockerManager.startContainer(test.vcsType);

    // 2. Load fixture
    await container.loadFixture(fixture);

    // 3. Start metrics collection
    const collector = await this.metricsCollector.attach(container);

    // 4. Execute test command
    const result = await container.exec(test.command);

    // 5. Stop collection
    const metrics = await collector.stop();

    // 6. Cleanup
    await container.stop();

    return { result, metrics };
  }

  // Parallel execution
  async runParallel(tests: Test[]): Promise<RawMetric[]> {
    const pools = this.createContainerPools(tests.length);
    return Promise.all(tests.map((test, i) =>
      this.executeTest(test, pools[i])
    ));
  }
}
```

**Docker Integration:**
```typescript
class DockerManager {
  async startContainer(vcsType: string): Promise<Container> {
    // 1. Pull image
    await this.docker.pull(`benchmark-${vcsType}:latest`);

    // 2. Create container with resource limits
    const container = await this.docker.createContainer({
      Image: `benchmark-${vcsType}:latest`,
      HostConfig: {
        Memory: 2 * 1024 * 1024 * 1024, // 2GB
        NanoCpus: 2 * 1e9, // 2 CPUs
        NetworkMode: 'none'
      }
    });

    // 3. Start container
    await container.start();

    // 4. Wait for ready
    await this.waitForReady(container);

    return new ContainerWrapper(container);
  }
}
```

---

### 5. Metrics Collector

**Purpose:** Gather performance data from containers

**Interactions:**
```typescript
class MetricsCollector {
  async attach(container: Container): Promise<ActiveCollector> {
    // Start collectors
    const cpuCollector = new CPUCollector(container);
    const memoryCollector = new MemoryCollector(container);
    const ioCollector = new IOCollector(container);

    await Promise.all([
      cpuCollector.start(),
      memoryCollector.start(),
      ioCollector.start()
    ]);

    return new ActiveCollector({
      cpu: cpuCollector,
      memory: memoryCollector,
      io: ioCollector
    });
  }
}

class CPUCollector {
  async collect(): Promise<CPUMetrics> {
    // Read /proc/stat in container
    const stats = await this.container.exec('cat /proc/stat');

    return {
      user: parseFloat(stats.user),
      system: parseFloat(stats.system),
      idle: parseFloat(stats.idle),
      timestamp: Date.now()
    };
  }
}
```

**Polling Strategy:**
```typescript
interface CollectionStrategy {
  interval: number; // ms between samples
  duration: number; // total collection time
  aggregation: 'mean' | 'sum' | 'max' | 'p95';
}

// High-frequency for short tests
const shortTestStrategy: CollectionStrategy = {
  interval: 10, // 10ms
  duration: 5000, // 5s
  aggregation: 'mean'
};

// Low-frequency for long tests
const longTestStrategy: CollectionStrategy = {
  interval: 1000, // 1s
  duration: 300000, // 5min
  aggregation: 'p95'
};
```

---

### 6. Analysis Pipeline

**Purpose:** Process raw metrics into insights

**Component Flow:**
```typescript
class AnalysisPipeline {
  async process(rawMetrics: RawMetric[]): Promise<ProcessedResult[]> {
    // 1. Clean and normalize
    const cleaned = await this.cleaner.clean(rawMetrics);

    // 2. Statistical analysis
    const stats = await this.statisticalEngine.analyze(cleaned);

    // 3. Comparative analysis
    const comparison = await this.comparator.compare(stats);

    // 4. Pattern detection
    const patterns = await this.patternDetector.detect(comparison);

    // 5. Optimization recommendations
    const recommendations = await this.optimizer.recommend(patterns);

    return { stats, comparison, patterns, recommendations };
  }
}
```

**Statistical Engine:**
```typescript
class StatisticalEngine {
  async analyze(metrics: CleanedMetric[]): Promise<Statistics> {
    return {
      descriptive: this.calculateDescriptive(metrics),
      inferential: this.performInference(metrics),
      timeSeries: this.analyzeTimeSeries(metrics)
    };
  }

  private calculateDescriptive(data: number[]): DescriptiveStats {
    return {
      mean: this.mean(data),
      median: this.median(data),
      mode: this.mode(data),
      stdDev: this.stdDev(data),
      variance: this.variance(data),
      skewness: this.skewness(data),
      kurtosis: this.kurtosis(data)
    };
  }

  private performInference(
    jjData: number[],
    gitData: number[]
  ): InferentialStats {
    const tTest = this.tTest(jjData, gitData);
    const mannWhitney = this.mannWhitneyU(jjData, gitData);
    const cohensD = this.cohensD(jjData, gitData);

    return {
      tTest,
      mannWhitney,
      effectSize: cohensD,
      confidenceInterval: this.confidenceInterval(jjData, 0.95)
    };
  }
}
```

---

### 7. Speed Optimizer

**Purpose:** Identify performance bottlenecks and optimization opportunities

**Interactions:**
```typescript
class SpeedOptimizer {
  async analyzeBottlenecks(
    metrics: ProcessedResult
  ): Promise<Bottleneck[]> {
    const hotPaths = await this.identifyHotPaths(metrics);
    const slowOperations = this.findSlowOperations(metrics);
    const resourceWaste = this.detectWaste(metrics);

    return [...hotPaths, ...slowOperations, ...resourceWaste];
  }

  async recommend(
    bottlenecks: Bottleneck[]
  ): Promise<Recommendation[]> {
    return bottlenecks.map(bn => {
      // Match bottleneck to optimization strategy
      const strategy = this.strategies.find(s =>
        s.applicableTo(bn)
      );

      return {
        bottleneck: bn,
        strategy: strategy.name,
        expectedImprovement: strategy.estimateGain(bn),
        implementation: strategy.generatePlan(bn)
      };
    });
  }
}

interface OptimizationStrategy {
  name: string;
  applicableTo(bottleneck: Bottleneck): boolean;
  estimateGain(bottleneck: Bottleneck): number;
  generatePlan(bottleneck: Bottleneck): ImplementationPlan;
}
```

**Hot Path Detection:**
```typescript
class HotPathDetector {
  async identify(trace: ExecutionTrace): Promise<HotPath[]> {
    // 1. Build call graph
    const callGraph = this.buildCallGraph(trace);

    // 2. Calculate cumulative time
    const cumulativeTimes = this.calculateCumulative(callGraph);

    // 3. Find critical paths (>10% total time)
    const criticalPaths = cumulativeTimes.filter(
      path => path.percentage > 0.1
    );

    // 4. Analyze optimization potential
    return criticalPaths.map(path => ({
      path,
      optimizationPotential: this.assessPotential(path),
      techniques: this.suggestTechniques(path)
    }));
  }
}
```

---

### 8. AgentDB Integration

**Purpose:** Store patterns and enable learning

**Interactions:**
```typescript
class BenchmarkLearning {
  async storeResults(result: ProcessedResult): Promise<void> {
    // Convert to pattern format
    const pattern: BenchmarkPattern = {
      sessionId: `benchmark-${result.testId}`,
      task: `${result.operation}-benchmark`,
      input: JSON.stringify({
        vcsType: result.vcsType,
        repoSize: result.repoSize,
        operation: result.operation
      }),
      output: JSON.stringify({
        executionTime: result.metrics.executionTime,
        memoryUsage: result.metrics.memoryPeak,
        success: result.success
      }),
      reward: this.calculateReward(result),
      critique: this.generateCritique(result),
      latencyMs: result.metrics.executionTime,
      tokensUsed: 0,
      success: result.success
    };

    await agentdb.patternStore(pattern);
  }

  async queryHistorical(operation: string): Promise<Insights> {
    // Retrieve similar benchmarks
    const patterns = await agentdb.patternSearch({
      task: `${operation}-benchmark`,
      k: 50,
      onlySuccesses: true
    });

    // Analyze trends
    return {
      historicalMean: this.calculateMean(patterns),
      trend: this.detectTrend(patterns),
      regressions: this.findRegressions(patterns),
      improvements: this.findImprovements(patterns)
    };
  }

  private calculateReward(result: ProcessedResult): number {
    // Multi-factor reward
    const factors = {
      speed: this.normalizeSpeed(result.metrics.executionTime),
      memory: this.normalizeMemory(result.metrics.memoryPeak),
      reliability: result.success ? 1 : 0,
      efficiency: this.calculateEfficiency(result)
    };

    return (
      factors.speed * 0.4 +
      factors.memory * 0.3 +
      factors.reliability * 0.2 +
      factors.efficiency * 0.1
    );
  }
}
```

**Pattern Schema:**
```typescript
interface BenchmarkPattern {
  sessionId: string;
  task: string; // "commit-benchmark", "merge-benchmark", etc.
  input: string; // JSON: { vcsType, repoSize, operation }
  output: string; // JSON: { time, memory, success }
  reward: number; // 0-1 (higher = better)
  critique: string; // Analysis of performance
  latencyMs: number; // Execution time
  tokensUsed: number; // Always 0 for benchmarks
  success: boolean;
}
```

---

### 9. Report Generator

**Purpose:** Create human-readable and machine-parseable reports

**Interactions:**
```typescript
class ReportGenerator {
  async generate(results: ProcessedResult[]): Promise<Report> {
    // Generate multiple formats
    const markdown = await this.generateMarkdown(results);
    const json = await this.generateJSON(results);
    const html = await this.generateHTML(results);

    // Save to disk
    await Promise.all([
      this.save('latest.md', markdown),
      this.save('latest.json', json),
      this.save('latest.html', html)
    ]);

    // Archive previous reports
    await this.archive();

    return { markdown, json, html };
  }

  private async generateMarkdown(
    results: ProcessedResult[]
  ): Promise<string> {
    const sections = [
      this.createSummary(results),
      this.createComparisonTable(results),
      this.createPerformanceCharts(results),
      this.createRecommendations(results),
      this.createDetailedResults(results)
    ];

    return sections.join('\n\n---\n\n');
  }
}
```

**Template Structure:**
```markdown
# Benchmark Report: {Date}

## Executive Summary
- Total tests: {count}
- Jujutsu avg speedup: {factor}x
- Memory efficiency: {percentage}
- Recommendations: {count}

## Performance Comparison
| Operation | Jujutsu | Git | Git-Worktrees | Winner |
|-----------|---------|-----|---------------|--------|
| ...       | ...     | ... | ...           | ...    |

## Charts
{Performance graphs}

## Detailed Results
{Per-test breakdowns}

## Optimization Recommendations
{Actionable insights}
```

---

## Data Flow Sequences

### Sequence 1: Complete Benchmark Run

```
User → Orchestrator.start(config)
     → TestPlanner.createPlan()
     → DataGenerator.generate()
     → TestExecutor.runParallel([
         DockerManager.startContainer('jujutsu'),
         DockerManager.startContainer('git'),
         DockerManager.startContainer('git-worktrees')
       ])
     → MetricsCollector.collect()
     → AnalysisPipeline.process()
     → StatisticalEngine.analyze()
     → SpeedOptimizer.recommend()
     → AgentDB.storePattern()
     → ReportGenerator.generate()
     → User receives Report
```

### Sequence 2: Single Test Execution

```
TestExecutor.executeTest(test)
  → DockerManager.startContainer(vcsType)
  → Container.loadFixture(repo)
  → MetricsCollector.attach(container)
  → Container.exec(command) [TIMED]
    → CPUCollector.poll() [every 10ms]
    → MemoryCollector.poll() [every 10ms]
    → IOCollector.poll() [every 10ms]
  → Container.exec() completes
  → MetricsCollector.stop()
  → MetricsCollector.aggregate()
  → DockerManager.stopContainer()
  → Return RawMetric
```

### Sequence 3: Pattern Learning

```
AnalysisPipeline.complete(results)
  → BenchmarkLearning.storeResults(result)
  → AgentDB.patternStore({
      task: "operation-benchmark",
      reward: calculated,
      ...
    })
  → AgentDB.patternSearch({
      task: "operation-benchmark",
      k: 50
    })
  → BenchmarkLearning.analyzeHistory(patterns)
  → Insights returned
```

---

## Error Handling

### Error Propagation

```typescript
class ErrorHandler {
  async handleTestFailure(error: Error, context: TestContext): Promise<void> {
    // 1. Log error
    await this.logger.error('Test failed', { error, context });

    // 2. Save error report
    await this.saveErrorReport(error, context);

    // 3. Attempt recovery
    if (this.isRecoverable(error)) {
      return this.retry(context);
    }

    // 4. Store failure pattern
    await this.agentdb.storeFailure({
      task: context.testName,
      error: error.message,
      reward: 0,
      success: false
    });

    // 5. Continue with next test
    throw new RecoverableError(error);
  }
}
```

### Retry Strategy

```typescript
interface RetryConfig {
  maxAttempts: 3;
  backoffMs: [1000, 5000, 10000];
  retryableErrors: ['ETIMEOUT', 'ECONNRESET', 'DOCKER_ERROR'];
}

async function withRetry<T>(
  fn: () => Promise<T>,
  config: RetryConfig
): Promise<T> {
  for (let attempt = 0; attempt < config.maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      if (
        attempt === config.maxAttempts - 1 ||
        !config.retryableErrors.includes(error.code)
      ) {
        throw error;
      }
      await sleep(config.backoffMs[attempt]);
    }
  }
}
```

---

## Performance Optimization

### Caching Strategy

```typescript
class BenchmarkCache {
  private cache = new Map<string, CachedResult>();

  async get(testId: string): Promise<CachedResult | null> {
    const cached = this.cache.get(testId);
    if (!cached) return null;

    // Check if still valid (1 hour TTL)
    if (Date.now() - cached.timestamp > 3600000) {
      this.cache.delete(testId);
      return null;
    }

    return cached;
  }

  async set(testId: string, result: RawMetric): Promise<void> {
    this.cache.set(testId, {
      result,
      timestamp: Date.now()
    });
  }
}
```

### Parallel Processing

```typescript
class ParallelExecutor {
  async runParallel(
    tests: Test[],
    maxConcurrency: number = 4
  ): Promise<RawMetric[]> {
    const queue = [...tests];
    const results: RawMetric[] = [];
    const running: Promise<RawMetric>[] = [];

    while (queue.length > 0 || running.length > 0) {
      // Start new tests up to maxConcurrency
      while (running.length < maxConcurrency && queue.length > 0) {
        const test = queue.shift()!;
        running.push(this.executeTest(test));
      }

      // Wait for at least one to complete
      const result = await Promise.race(running);
      results.push(result);

      // Remove completed from running
      const index = running.findIndex(p => p === result);
      running.splice(index, 1);
    }

    return results;
  }
}
```

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-09
