# Data Models and Schemas
## Benchmarking System Data Structures

**Version:** 1.0.0
**Date:** 2025-11-09

---

## Core Data Models

### 1. Benchmark Configuration

```typescript
interface BenchmarkConfig {
  // Metadata
  id: string;
  name: string;
  description: string;
  version: string;
  createdAt: string;

  // Execution settings
  execution: {
    iterations: number; // Number of runs per test
    warmupRuns: number; // Runs to discard before measurement
    timeout: number; // Max time per test (ms)
    parallelism: number; // Max concurrent tests
    randomSeed?: number; // For reproducibility
  };

  // Environment configuration
  environment: {
    docker: {
      baseImages: {
        jujutsu: string;
        git: string;
        gitWorktrees: string;
      };
      resources: ResourceConstraints;
      network: 'none' | 'bridge' | 'host';
      volumes: VolumeMount[];
    };
  };

  // Test selection
  tests: {
    suites: string[]; // Suite names or patterns
    include?: string[]; // Specific test IDs
    exclude?: string[]; // Tests to skip
    tags?: string[]; // Filter by tags
  };

  // Data generation
  dataGeneration: {
    repoSizes: ('small' | 'medium' | 'large')[];
    customFixtures?: string[]; // Paths to custom repos
    cacheFixtures: boolean;
  };

  // Output settings
  output: {
    formats: ('json' | 'markdown' | 'html')[];
    destination: string;
    archiveResults: boolean;
    compression: boolean;
  };

  // Analysis options
  analysis: {
    statisticalSignificance: number; // p-value threshold
    confidenceLevel: number; // e.g., 0.95
    outlierDetection: boolean;
    trendAnalysis: boolean;
  };

  // Learning integration
  agentdb: {
    enabled: boolean;
    sessionId: string;
    storePatterns: boolean;
    queryHistorical: boolean;
  };
}

interface ResourceConstraints {
  cpus: number; // Number of CPU cores
  memory: number; // Memory in bytes
  memoryReservation: number; // Soft limit
  diskQuota?: number; // Disk space limit
  ioWeight?: number; // I/O priority (10-1000)
}

interface VolumeMount {
  source: string; // Host path
  target: string; // Container path
  readonly: boolean;
}
```

### 2. Test Suite Definition

```typescript
interface TestSuite {
  // Metadata
  id: string;
  name: string;
  description: string;
  category: TestCategory;
  tags: string[];

  // Test configuration
  tests: Test[];
  setup?: SetupScript;
  teardown?: TeardownScript;

  // Requirements
  requirements: {
    minRepoSize: RepoSize;
    requiredFeatures: string[];
    estimatedDuration: number; // ms
  };
}

type TestCategory =
  | 'basic-operations'
  | 'workflow-simulation'
  | 'scale-testing'
  | 'edge-cases';

type RepoSize = 'small' | 'medium' | 'large' | 'custom';

interface Test {
  id: string;
  name: string;
  description: string;

  // Execution
  command: string; // Command to run
  expectedExitCode: number;
  timeout: number; // Override default timeout

  // Variants
  vcsTypes: ('jujutsu' | 'git' | 'git-worktrees')[];

  // Metrics to collect
  metrics: MetricType[];

  // Validation
  validation?: {
    outputMatches?: RegExp;
    minExecutionTime?: number;
    maxExecutionTime?: number;
    maxMemoryUsage?: number;
  };

  // Dependencies
  dependsOn?: string[]; // Test IDs that must run first
}

type MetricType =
  | 'execution-time'
  | 'cpu-usage'
  | 'memory-usage'
  | 'io-operations'
  | 'disk-usage'
  | 'network-traffic';

interface SetupScript {
  description: string;
  commands: string[];
  timeout: number;
}

interface TeardownScript {
  description: string;
  commands: string[];
  alwaysRun: boolean; // Run even on failure
}
```

### 3. Repository Fixture

```typescript
interface RepositoryFixture {
  // Metadata
  id: string;
  name: string;
  size: RepoSize;
  createdAt: string;

  // Location
  path: string; // Absolute path to repository
  vcsType: 'jujutsu' | 'git';

  // Repository properties
  metadata: {
    commits: number;
    branches: number;
    tags: number;
    files: number;
    totalSize: number; // bytes
    largestFile: number; // bytes
  };

  // Content characteristics
  characteristics: {
    codeFiles: number;
    binaryFiles: number;
    avgCommitSize: number;
    avgBranchDepth: number;
    conflictFrequency: number; // 0-1
  };

  // Generation info
  generation: {
    strategy: 'realistic' | 'synthetic' | 'imported';
    seed: number; // For reproducibility
    parameters: GenerationParameters;
  };
}

interface GenerationParameters {
  // Repository structure
  fileTree: {
    maxDepth: number;
    filesPerDir: number;
    dirCount: number;
  };

  // Commit history
  commits: {
    count: number;
    timeSpan: number; // days
    authorsCount: number;
    avgFilesPerCommit: number;
  };

  // Branching structure
  branches: {
    count: number;
    mergeFrequency: number; // 0-1
    branchLifetime: number; // commits
  };

  // Content
  content: {
    codeLanguages: string[];
    avgFileSize: number;
    binaryRatio: number; // 0-1
  };
}
```

### 4. Raw Metrics

```typescript
interface RawMetric {
  // Test identification
  testId: string;
  runId: string; // Unique ID for this specific run
  iteration: number; // Which iteration (1-N)
  timestamp: string; // ISO 8601

  // Test context
  context: {
    vcsType: 'jujutsu' | 'git' | 'git-worktrees';
    operation: string;
    repoSize: RepoSize;
    repoId: string;
  };

  // Execution results
  execution: {
    exitCode: number;
    signalCode?: string;
    stdout: string;
    stderr: string;
    startTime: number; // Unix timestamp ms
    endTime: number; // Unix timestamp ms
    durationMs: number;
  };

  // Performance metrics
  performance: {
    cpu: CPUMetrics;
    memory: MemoryMetrics;
    io: IOMetrics;
    disk: DiskMetrics;
  };

  // Environment snapshot
  environment: {
    containerId: string;
    cpuCores: number;
    totalMemory: number;
    diskType: string;
    kernelVersion: string;
  };

  // Quality indicators
  quality: {
    success: boolean;
    errorType?: string;
    warningCount: number;
    retryCount: number;
  };
}

interface CPUMetrics {
  userTime: number; // ms
  systemTime: number; // ms
  totalTime: number; // ms
  percentage: number; // 0-100
  contextSwitches: number;
  timeSeries?: TimeSeries<number>; // Optional detailed trace
}

interface MemoryMetrics {
  rss: number; // Resident Set Size (bytes)
  vms: number; // Virtual Memory Size (bytes)
  heap: number; // Heap usage (bytes)
  external: number; // External memory (bytes)
  peak: number; // Peak memory usage (bytes)
  timeSeries?: TimeSeries<number>;
}

interface IOMetrics {
  readOps: number;
  writeOps: number;
  readBytes: number;
  writeBytes: number;
  readLatency: number; // avg ms
  writeLatency: number; // avg ms
  timeSeries?: TimeSeries<IOSample>;
}

interface IOSample {
  timestamp: number;
  readOps: number;
  writeOps: number;
}

interface DiskMetrics {
  initialSize: number; // bytes
  finalSize: number; // bytes
  delta: number; // bytes
  inodes: number;
}

interface TimeSeries<T> {
  interval: number; // ms between samples
  samples: T[];
}
```

### 5. Processed Results

```typescript
interface ProcessedResult {
  // Test identification
  testSuite: string;
  testName: string;
  operation: string;
  timestamp: string;

  // Aggregated metrics by VCS type
  jujutsu: PerformanceStats;
  git: PerformanceStats;
  gitWorktrees: PerformanceStats;

  // Comparative analysis
  comparison: {
    jjVsGit: ComparisonMetrics;
    jjVsGitWorktrees: ComparisonMetrics;
    gitVsGitWorktrees: ComparisonMetrics;
  };

  // Quality metrics
  quality: {
    reliability: number; // Success rate 0-1
    variance: number; // Coefficient of variation
    outliers: number; // Number of outliers detected
  };

  // Insights
  insights: {
    winner: 'jujutsu' | 'git' | 'git-worktrees';
    confidence: number; // 0-1
    recommendations: Recommendation[];
    bottlenecks: Bottleneck[];
  };

  // Learning data
  learning: {
    reward: number; // 0-1
    patterns: Pattern[];
    historicalComparison: TrendAnalysis;
  };
}

interface PerformanceStats {
  // Central tendency
  mean: number;
  median: number;
  mode?: number;

  // Dispersion
  stdDev: number;
  variance: number;
  range: [number, number]; // [min, max]

  // Percentiles
  percentiles: {
    p25: number;
    p50: number;
    p75: number;
    p90: number;
    p95: number;
    p99: number;
  };

  // Sample info
  sampleSize: number;
  validSamples: number;
  outliers: number[];

  // Distribution
  distribution: {
    skewness: number;
    kurtosis: number;
    normality: boolean; // Shapiro-Wilk test
  };
}

interface ComparisonMetrics {
  // Performance ratios
  speedupFactor: number; // baseline / comparison
  memoryRatio: number; // comparison / baseline
  ioEfficiency: number; // ops per byte

  // Statistical significance
  statistical: {
    significant: boolean;
    pValue: number;
    confidenceInterval: [number, number];
    effectSize: number; // Cohen's d
    power: number; // Statistical power
  };

  // Practical significance
  practical: {
    absoluteDifference: number;
    relativeDifference: number; // percentage
    threshold: number; // Minimum meaningful difference
    meaningful: boolean;
  };
}

interface Recommendation {
  id: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  category: 'performance' | 'quality' | 'security' | 'usability';
  title: string;
  description: string;
  rationale: string;
  expectedImpact: {
    metric: string;
    improvement: number; // percentage or absolute
    confidence: number; // 0-1
  };
  implementation: {
    effort: 'low' | 'medium' | 'high';
    steps: string[];
    risks: string[];
  };
}

interface Bottleneck {
  id: string;
  type: 'cpu' | 'memory' | 'io' | 'algorithm';
  location: string; // Code path or operation
  severity: number; // 0-1
  impact: {
    timePercentage: number; // % of total time
    occurences: number;
  };
  optimization: {
    technique: string;
    expectedGain: number; // percentage
    complexity: 'trivial' | 'moderate' | 'complex';
  };
}

interface Pattern {
  type: 'performance' | 'error' | 'usage';
  description: string;
  frequency: number;
  examples: string[];
}

interface TrendAnalysis {
  direction: 'improving' | 'degrading' | 'stable';
  slope: number; // Rate of change
  r2: number; // Goodness of fit
  forecast: {
    nextValue: number;
    confidence: number;
  };
}
```

### 6. AgentDB Integration Schema

```typescript
interface BenchmarkPattern {
  // Required AgentDB fields
  sessionId: string; // Format: "benchmark-{timestamp}"
  task: string; // Format: "{operation}-{vcsType}-benchmark"
  input: string; // JSON-serialized input parameters
  output: string; // JSON-serialized results
  reward: number; // 0-1, calculated from performance
  success: boolean;

  // Optional metadata
  critique?: string; // Analysis of the result
  latencyMs?: number; // Execution time
  tokensUsed?: number; // Always 0 for benchmarks
}

// Example pattern storage
const examplePattern: BenchmarkPattern = {
  sessionId: 'benchmark-1699564800000',
  task: 'commit-jujutsu-benchmark',
  input: JSON.stringify({
    operation: 'commit',
    vcsType: 'jujutsu',
    repoSize: 'medium',
    files: 100,
    commitMessage: '50 chars'
  }),
  output: JSON.stringify({
    executionTime: 45,
    memoryPeak: 128,
    ioOperations: 250,
    success: true
  }),
  reward: 0.92, // High reward for good performance
  success: true,
  critique: 'Excellent performance. 2.1x faster than Git with lower memory usage.',
  latencyMs: 45,
  tokensUsed: 0
};

// Pattern query interface
interface PatternQuery {
  task: string; // Search by task pattern
  k?: number; // Number of results
  minReward?: number; // Filter by minimum reward
  onlySuccesses?: boolean; // Only successful runs
  onlyFailures?: boolean; // Only failed runs
}

// Pattern search result
interface PatternSearchResult {
  patterns: BenchmarkPattern[];
  statistics: {
    totalFound: number;
    avgReward: number;
    successRate: number;
    avgLatency: number;
  };
  insights: {
    bestPerforming: BenchmarkPattern;
    worstPerforming: BenchmarkPattern;
    trend: 'improving' | 'degrading' | 'stable';
  };
}
```

### 7. Report Schema

```typescript
interface BenchmarkReport {
  // Metadata
  metadata: {
    reportId: string;
    generatedAt: string;
    benchmarkConfig: string; // Config ID
    version: string;
    author: string;
  };

  // Summary
  summary: {
    totalTests: number;
    successfulTests: number;
    failedTests: number;
    totalDuration: number; // ms
    overallWinner: 'jujutsu' | 'git' | 'git-worktrees';
    avgSpeedup: number;
    keyFindings: string[];
  };

  // Detailed results
  results: ProcessedResult[];

  // Comparative analysis
  comparison: {
    byOperation: OperationComparison[];
    byRepoSize: RepoCategoryComparison[];
    overall: OverallComparison;
  };

  // Visualizations
  charts: {
    performanceComparison: ChartData;
    memoryUsage: ChartData;
    scalability: ChartData;
  };

  // Recommendations
  recommendations: {
    critical: Recommendation[];
    important: Recommendation[];
    optional: Recommendation[];
  };

  // Historical context
  historical: {
    previousReports: string[]; // Report IDs
    trendsOverTime: TrendData[];
    regressions: Regression[];
    improvements: Improvement[];
  };

  // Appendices
  appendices: {
    rawData: string; // Path to raw data
    methodology: string;
    environmentDetails: EnvironmentSnapshot;
    reproducibility: ReproducibilityGuide;
  };
}

interface OperationComparison {
  operation: string;
  jujutsu: number;
  git: number;
  gitWorktrees: number;
  winner: string;
  speedup: number;
}

interface RepoCategoryComparison {
  size: RepoSize;
  averageSpeedup: number;
  results: OperationComparison[];
}

interface OverallComparison {
  avgExecutionTime: {
    jujutsu: number;
    git: number;
    gitWorktrees: number;
  };
  avgMemoryUsage: {
    jujutsu: number;
    git: number;
    gitWorktrees: number;
  };
  winRate: {
    jujutsu: number;
    git: number;
    gitWorktrees: number;
  };
}

interface ChartData {
  type: 'bar' | 'line' | 'scatter' | 'box';
  title: string;
  data: any; // Plotly.js compatible data
  layout: any; // Plotly.js layout
}

interface Regression {
  test: string;
  previousValue: number;
  currentValue: number;
  degradation: number; // percentage
  severity: 'minor' | 'moderate' | 'severe';
}

interface Improvement {
  test: string;
  previousValue: number;
  currentValue: number;
  improvement: number; // percentage
  significance: 'minor' | 'moderate' | 'major';
}

interface EnvironmentSnapshot {
  os: string;
  kernel: string;
  cpu: string;
  memory: string;
  disk: string;
  docker: string;
  jujutsu: string;
  git: string;
}

interface ReproducibilityGuide {
  dockerImages: string[];
  commands: string[];
  configFile: string;
  seed: number;
}
```

---

## Database Schemas

### SQLite Schema for AgentDB Integration

```sql
-- Benchmark results table
CREATE TABLE benchmark_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  test_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  timestamp INTEGER NOT NULL,
  vcs_type TEXT NOT NULL,
  operation TEXT NOT NULL,
  repo_size TEXT NOT NULL,

  -- Performance metrics
  execution_time_ms REAL NOT NULL,
  cpu_user_ms REAL NOT NULL,
  cpu_system_ms REAL NOT NULL,
  memory_peak_bytes INTEGER NOT NULL,
  io_read_ops INTEGER NOT NULL,
  io_write_ops INTEGER NOT NULL,
  disk_delta_bytes INTEGER NOT NULL,

  -- Quality
  success INTEGER NOT NULL, -- 0 or 1
  exit_code INTEGER NOT NULL,
  retry_count INTEGER NOT NULL,

  -- Calculated
  reward REAL NOT NULL,

  -- Indexes
  INDEX idx_vcs_operation (vcs_type, operation),
  INDEX idx_timestamp (timestamp),
  INDEX idx_success (success)
);

-- Patterns table (mirrors AgentDB structure)
CREATE TABLE benchmark_patterns (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  task TEXT NOT NULL,
  input TEXT NOT NULL, -- JSON
  output TEXT NOT NULL, -- JSON
  reward REAL NOT NULL,
  critique TEXT,
  latency_ms REAL,
  success INTEGER NOT NULL,
  created_at INTEGER NOT NULL,

  INDEX idx_task (task),
  INDEX idx_reward (reward),
  INDEX idx_created_at (created_at)
);

-- Trends table
CREATE TABLE performance_trends (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  operation TEXT NOT NULL,
  vcs_type TEXT NOT NULL,
  date TEXT NOT NULL, -- YYYY-MM-DD
  avg_time_ms REAL NOT NULL,
  avg_memory_bytes REAL NOT NULL,
  sample_count INTEGER NOT NULL,

  INDEX idx_operation_vcs (operation, vcs_type),
  INDEX idx_date (date)
);

-- Recommendations table
CREATE TABLE recommendations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  report_id TEXT NOT NULL,
  priority TEXT NOT NULL,
  category TEXT NOT NULL,
  title TEXT NOT NULL,
  description TEXT NOT NULL,
  expected_impact REAL NOT NULL,
  created_at INTEGER NOT NULL,

  INDEX idx_report (report_id),
  INDEX idx_priority (priority)
);
```

---

## JSON Schema Examples

### Benchmark Configuration JSON

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["id", "name", "execution", "environment", "tests"],
  "properties": {
    "id": {
      "type": "string",
      "pattern": "^benchmark-[0-9a-f]{8}$"
    },
    "name": {
      "type": "string",
      "minLength": 1,
      "maxLength": 100
    },
    "execution": {
      "type": "object",
      "required": ["iterations", "warmupRuns", "timeout"],
      "properties": {
        "iterations": {
          "type": "integer",
          "minimum": 1,
          "maximum": 1000
        },
        "warmupRuns": {
          "type": "integer",
          "minimum": 0,
          "maximum": 10
        },
        "timeout": {
          "type": "integer",
          "minimum": 1000,
          "maximum": 3600000
        }
      }
    },
    "tests": {
      "type": "object",
      "required": ["suites"],
      "properties": {
        "suites": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "minItems": 1
        }
      }
    }
  }
}
```

### Raw Metric JSON Example

```json
{
  "testId": "commit-basic-001",
  "runId": "run-1699564800-abc123",
  "iteration": 3,
  "timestamp": "2025-11-09T18:00:00.000Z",
  "context": {
    "vcsType": "jujutsu",
    "operation": "commit",
    "repoSize": "medium",
    "repoId": "fixture-medium-001"
  },
  "execution": {
    "exitCode": 0,
    "startTime": 1699564800000,
    "endTime": 1699564800045,
    "durationMs": 45
  },
  "performance": {
    "cpu": {
      "userTime": 35,
      "systemTime": 8,
      "totalTime": 43,
      "percentage": 95.6,
      "contextSwitches": 12
    },
    "memory": {
      "rss": 134217728,
      "vms": 268435456,
      "heap": 67108864,
      "external": 4194304,
      "peak": 134217728
    },
    "io": {
      "readOps": 45,
      "writeOps": 23,
      "readBytes": 1048576,
      "writeBytes": 524288,
      "readLatency": 0.5,
      "writeLatency": 1.2
    },
    "disk": {
      "initialSize": 10485760,
      "finalSize": 11010048,
      "delta": 524288,
      "inodes": 150
    }
  },
  "quality": {
    "success": true,
    "warningCount": 0,
    "retryCount": 0
  }
}
```

---

## File Format Standards

### CSV Export Format

```csv
test_id,run_id,timestamp,vcs_type,operation,execution_time_ms,memory_peak_mb,success
commit-001,run-001,2025-11-09T18:00:00Z,jujutsu,commit,45,128,true
commit-001,run-001,2025-11-09T18:00:01Z,git,commit,95,156,true
commit-001,run-001,2025-11-09T18:00:02Z,git-worktrees,commit,87,142,true
```

### Markdown Report Template

````markdown
# Benchmark Report: {date}

## Executive Summary

- **Total Tests**: {count}
- **Success Rate**: {percentage}%
- **Overall Winner**: {vcs_type}
- **Average Speedup**: {factor}x

## Performance Comparison

| Operation | Jujutsu (ms) | Git (ms) | Git-WT (ms) | Winner | Speedup |
|-----------|--------------|----------|-------------|--------|---------|
| commit    | 45           | 95       | 87          | JJ     | 2.1x    |
| merge     | 120          | 280      | 250         | JJ     | 2.3x    |

## Key Findings

1. **Finding 1**: Description
2. **Finding 2**: Description

## Recommendations

### Critical
- Recommendation 1
- Recommendation 2

## Detailed Results

```json
{detailed results}
```
````

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-09
