# Benchmarking Guide: Agentic-Jujutsu
## Running, Interpreting, and Contributing to Performance Benchmarks

**Version:** 1.0.0
**Last Updated:** 2025-11-09
**Status:** Design Complete - Implementation Pending

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Docker Setup](#docker-setup)
4. [Running Benchmarks](#running-benchmarks)
5. [Interpreting Results](#interpreting-results)
6. [Comparison Methodology](#comparison-methodology)
7. [Expected Performance Baselines](#expected-performance-baselines)
8. [Contributing Benchmarks](#contributing-benchmarks)
9. [Advanced Configuration](#advanced-configuration)

---

## Overview

### What is This Benchmarking System?

The Agentic-Jujutsu benchmarking system provides comprehensive performance comparison between:
- **Jujutsu VCS** - Modern version control with operation log
- **Git** - Traditional version control
- **Git-Worktrees** - Git with worktree optimization

### Key Features

1. **Multi-Dimensional Analysis**
   - Performance metrics (CPU, memory, I/O)
   - Code quality assessment
   - Security scanning
   - Speed optimization
   - Self-learning via AgentDB

2. **Statistical Rigor**
   - T-tests for significance (p < 0.05)
   - Effect size calculations (Cohen's d)
   - Confidence intervals (95%)
   - Outlier detection

3. **Docker Isolation**
   - Fair benchmarking with resource constraints
   - Reproducible test environments
   - Network isolation for accuracy

4. **Scalability**
   - Horizontal: 4-32+ concurrent containers
   - Vertical: Dynamic resource allocation
   - Auto-scaling based on load

---

## Quick Start

### Prerequisites

**Required:**
- Docker 20.10+
- Docker Compose 2.0+
- Node.js 20+
- 8GB RAM minimum
- 20GB free disk space

**Optional:**
- Jujutsu CLI (for native benchmarks)
- Git 2.40+ (for comparison)

---

### 5-Minute Setup

**IMPORTANT:** The benchmark framework is currently in design phase. These instructions will work once implementation is complete (Week 2 of roadmap).

```bash
# Step 1: Navigate to package directory
cd /workspaces/agentic-flow/packages/agentic-jujutsu

# Step 2: Install dependencies
npm install

# Step 3: Build WASM (requires fix - see troubleshooting)
npm run build

# Step 4: Set up Docker environments
cd benchmarks/docker
docker-compose build

# Step 5: Run basic benchmark suite
cd ../..
npm run bench:run -- --suite basic-operations

# Step 6: View results
npm run bench:report -- --format markdown
```

**Expected Output:**
```
✅ Benchmark suite completed
📊 Results: benchmarks/results/latest.json
📄 Report: benchmarks/results/latest.md
```

---

## Docker Setup

### Container Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Docker Compose Setup                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Jujutsu    │  │     Git      │  │ Git-Worktrees│  │
│  │  Container   │  │  Container   │  │  Container   │  │
│  │              │  │              │  │              │  │
│  │ • jj CLI     │  │ • git CLI    │  │ • git CLI    │  │
│  │ • Rust 1.75+ │  │ • Alpine     │  │ • Worktrees  │  │
│  │ • Monitors   │  │ • Monitors   │  │ • Monitors   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │           Metrics Collector Container            │   │
│  │  • Node.js 20                                    │   │
│  │  • Prometheus exporters                          │   │
│  │  • StatsD aggregation                            │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

### Docker Compose Configuration

**File:** `benchmarks/docker/docker-compose.yml` (to be created)

```yaml
version: '3.8'

services:
  # Jujutsu benchmark container
  jujutsu-bench:
    build:
      context: .
      dockerfile: Dockerfile.jujutsu
    container_name: jujutsu-bench
    environment:
      - BENCHMARK_MODE=true
      - RUST_BACKTRACE=1
      - AGENTDB_PATH=/data/agentdb.sqlite
    volumes:
      - ./data:/data
      - ./results:/results
      - ../../src:/app/src:ro
    networks:
      - benchmark-net
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G

  # Git benchmark container
  git-bench:
    build:
      context: .
      dockerfile: Dockerfile.git
    container_name: git-bench
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
        reservations:
          cpus: '1.0'
          memory: 1G

  # Git-Worktrees benchmark container
  git-worktrees-bench:
    build:
      context: .
      dockerfile: Dockerfile.git-worktrees
    container_name: git-worktrees-bench
    environment:
      - BENCHMARK_MODE=true
      - GIT_USE_WORKTREES=true
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
        reservations:
          cpus: '1.0'
          memory: 1G

  # Metrics collection container
  metrics-collector:
    build:
      context: .
      dockerfile: Dockerfile.metrics
    container_name: metrics-collector
    volumes:
      - ./results:/results
      - ../../.claude-flow/metrics:/metrics
    networks:
      - benchmark-net
    depends_on:
      - jujutsu-bench
      - git-bench
      - git-worktrees-bench

networks:
  benchmark-net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16
```

---

### Dockerfile Examples

#### Jujutsu Container

**File:** `benchmarks/docker/Dockerfile.jujutsu`

```dockerfile
# Build stage
FROM rust:1.75-slim as builder

RUN apt-get update && apt-get install -y \
    git curl build-essential libssl-dev pkg-config

# Install jj from source
RUN cargo install --git https://github.com/martinvonz/jj.git jj-cli

# Runtime stage
FROM debian:bookworm-slim

# Copy jj binary
COPY --from=builder /usr/local/cargo/bin/jj /usr/local/bin/

# Install monitoring tools
RUN apt-get update && apt-get install -y \
    time \
    procps \
    sysstat \
    strace \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Environment
ENV RUST_BACKTRACE=1
ENV PATH=/usr/local/bin:$PATH

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s \
  CMD jj --version || exit 1

CMD ["/bin/bash"]
```

---

#### Git Container

**File:** `benchmarks/docker/Dockerfile.git`

```dockerfile
FROM alpine:latest

# Install git and monitoring tools
RUN apk add --no-cache \
    git \
    bash \
    time \
    procps \
    sysstat \
    strace

WORKDIR /workspace

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s \
  CMD git --version || exit 1

CMD ["/bin/bash"]
```

---

### Building Docker Images

```bash
# Navigate to Docker directory
cd /workspaces/agentic-flow/packages/agentic-jujutsu/benchmarks/docker

# Build all images
docker-compose build

# Or build specific image
docker-compose build jujutsu-bench

# Verify images
docker images | grep bench

# Expected output:
# jujutsu-bench       latest   ...   ...   ...
# git-bench          latest   ...   ...   ...
# git-worktrees-bench latest   ...   ...   ...
# metrics-collector  latest   ...   ...   ...
```

---

### Starting Containers

```bash
# Start all containers
docker-compose up -d

# Check status
docker-compose ps

# Expected output:
# NAME                   STATUS      PORTS
# jujutsu-bench         running
# git-bench             running
# git-worktrees-bench   running
# metrics-collector     running

# View logs
docker-compose logs -f jujutsu-bench

# Stop containers
docker-compose down
```

---

## Running Benchmarks

### Command-Line Interface

**Basic Usage:**
```bash
npm run bench:run [options]
```

**Options:**
- `--suite <name>` - Run specific suite
- `--config <path>` - Custom config file
- `--iterations <n>` - Number of iterations (default: 10)
- `--warmup <n>` - Warmup runs (default: 3)
- `--parallel <n>` - Parallel containers (default: 1)
- `--docker` - Run in Docker (default: true)
- `--output <path>` - Results output path

---

### Running Specific Suites

#### 1. Basic Operations Suite

```bash
# Run all basic operations (15 tests)
npm run bench:run -- --suite basic-operations

# Run specific test
npm run bench:run -- --suite basic-operations --test commit

# With custom iterations
npm run bench:run -- --suite basic-operations --iterations 20
```

**Tests Included:**
- Repository initialization
- File staging
- Commit creation
- Branch management
- Merge operations
- Rebase workflows
- Cherry-pick
- Stash operations
- Tag management
- Remote operations
- Log queries
- Diff generation
- Blame analysis
- Bisect search
- Cleanup/gc

---

#### 2. Workflow Simulation Suite

```bash
# Run all workflow simulations (10 tests)
npm run bench:run -- --suite workflow-simulation

# Run specific workflow
npm run bench:run -- --suite workflow-simulation --test feature-development
```

**Tests Included:**
- Feature branch workflow
- Gitflow workflow
- Trunk-based development
- Hotfix process
- Code review simulation
- Release process
- Monorepo operations
- Submodule management
- Large file handling
- CI/CD integration

---

#### 3. Scale Testing Suite

```bash
# Run scale tests (8 tests)
npm run bench:run -- --suite scale-testing

# With larger repository
npm run bench:run -- --suite scale-testing --repo-size large
```

**Tests Included:**
- Large repository (1M+ commits)
- Many branches (10k+)
- Deep history (100k+ commits)
- Large files (GB+ blobs)
- Concurrent operations
- Network latency simulation
- Disk I/O limits
- Memory constraints

---

#### 4. Edge Cases Suite

```bash
# Run edge case tests (7 tests)
npm run bench:run -- --suite edge-cases
```

**Tests Included:**
- Merge conflicts
- Binary file conflicts
- Detached HEAD states
- Repository recovery
- Interrupted operations
- Concurrent modifications
- Permission edge cases

---

### Running All Suites

```bash
# Run complete benchmark suite (40 tests)
npm run bench:run -- --all

# With parallel execution (4 containers)
npm run bench:run -- --all --parallel 4

# With full statistics
npm run bench:run -- --all --iterations 30 --warmup 5
```

**Expected Duration:**
- Sequential: ~30 minutes
- Parallel (4 containers): ~10 minutes
- Parallel (8 containers): ~5 minutes

---

### Custom Configuration

**Create config file:** `benchmarks/config/custom.json`

```json
{
  "id": "custom-benchmark",
  "name": "Custom Performance Test",
  "description": "Focused benchmark for specific operations",

  "execution": {
    "iterations": 20,
    "warmupRuns": 5,
    "timeout": 60000,
    "parallelism": 4
  },

  "tests": {
    "suites": ["basic-operations"],
    "includeTests": ["commit", "merge", "rebase"],
    "excludeTests": []
  },

  "environment": {
    "docker": {
      "baseImages": {
        "jujutsu": "jujutsu-bench:latest",
        "git": "git-bench:latest",
        "gitWorktrees": "git-worktrees-bench:latest"
      },
      "resources": {
        "cpus": 2,
        "memory": 2147483648
      },
      "network": "none",
      "volumes": [
        "./data:/data",
        "./results:/results"
      ]
    }
  },

  "dataGeneration": {
    "repoSizes": ["small", "medium"],
    "cacheFixtures": true,
    "repoMetadata": {
      "small": {
        "commitCount": 100,
        "branchCount": 10,
        "fileCount": 50
      },
      "medium": {
        "commitCount": 1000,
        "branchCount": 50,
        "fileCount": 500
      }
    }
  },

  "output": {
    "formats": ["json", "markdown", "html"],
    "destination": "./benchmarks/results",
    "archiveResults": true,
    "compression": true
  },

  "analysis": {
    "statisticalSignificance": 0.05,
    "confidenceLevel": 0.95,
    "outlierDetection": true,
    "trendAnalysis": true
  },

  "agentdb": {
    "enabled": true,
    "sessionId": "custom-benchmark-session",
    "storePatterns": true,
    "queryHistorical": true
  }
}
```

**Run with custom config:**
```bash
npm run bench:run -- --config benchmarks/config/custom.json
```

---

## Interpreting Results

### Result Files

After running benchmarks, you'll find:

```
benchmarks/results/
├── raw/
│   ├── 2025-11-09-basic-operations-jujutsu.json
│   ├── 2025-11-09-basic-operations-git.json
│   └── 2025-11-09-basic-operations-git-worktrees.json
├── processed/
│   └── 2025-11-09-analysis.json
└── reports/
    ├── latest.md
    ├── latest.html
    └── latest.json
```

---

### Reading the Report

**Example Report:** `benchmarks/results/latest.md`

```markdown
# Benchmark Report: Basic Operations
**Date:** 2025-11-09
**Suite:** basic-operations
**Iterations:** 10

## Executive Summary

- **Total Tests:** 15
- **Jujutsu Wins:** 12 (80%)
- **Git Wins:** 2 (13%)
- **Ties:** 1 (7%)
- **Average Speedup:** 3.2x faster

## Detailed Results

### 1. Repository Initialization

| Metric | Jujutsu | Git | Git-Worktrees | Winner |
|--------|---------|-----|---------------|--------|
| Mean (ms) | 45 | 180 | 175 | Jujutsu (4.0x) |
| Median (ms) | 42 | 178 | 172 | Jujutsu (4.2x) |
| Std Dev (ms) | 5 | 12 | 11 | Jujutsu |
| p95 (ms) | 52 | 195 | 190 | Jujutsu |
| Memory (MB) | 12 | 25 | 24 | Jujutsu (2.1x) |

**Statistical Significance:** p = 0.001 (< 0.05) ✅
**Effect Size (Cohen's d):** 2.8 (Large effect)
**Verdict:** Jujutsu is significantly faster

### 2. Commit Operation

| Metric | Jujutsu | Git | Git-Worktrees | Winner |
|--------|---------|-----|---------------|--------|
| Mean (ms) | 85 | 250 | 240 | Jujutsu (2.9x) |
| Median (ms) | 82 | 245 | 235 | Jujutsu (3.0x) |
| Std Dev (ms) | 8 | 18 | 15 | Jujutsu |
| p95 (ms) | 95 | 275 | 265 | Jujutsu |
| Memory (MB) | 18 | 35 | 32 | Jujutsu (1.9x) |

**Statistical Significance:** p = 0.002 (< 0.05) ✅
**Effect Size (Cohen's d):** 2.5 (Large effect)
**Verdict:** Jujutsu is significantly faster

## Summary Statistics

### Overall Performance

- **Fastest Operations:** Jujutsu (80% of tests)
- **Most Memory Efficient:** Jujutsu (87% of tests)
- **Most Reliable:** All systems (100% success rate)

### Key Insights

1. Jujutsu excels at lightweight operations (init, branch, status)
2. Merge operations show 3.3x speedup
3. Memory usage consistently lower (40-60% reduction)
4. All tests show statistical significance (p < 0.05)

### Recommendations

1. **Use Jujutsu for:**
   - Frequent branch operations
   - Large repositories with deep history
   - Memory-constrained environments

2. **Use Git when:**
   - Ecosystem compatibility required
   - Existing tooling integration needed
   - Team familiarity with Git workflows
```

---

### Understanding Metrics

#### Execution Time Metrics

- **Mean:** Average execution time across all iterations
- **Median:** Middle value (50th percentile) - more robust to outliers
- **Std Dev:** Standard deviation - measure of variability
- **p95/p99:** 95th/99th percentile - worst-case performance

**Interpretation:**
- Lower is better
- Low std dev indicates consistent performance
- Focus on median for typical performance
- Check p95/p99 for worst-case scenarios

---

#### Statistical Significance

- **p-value:** Probability results occurred by chance
  - p < 0.05: Statistically significant (95% confidence)
  - p < 0.01: Highly significant (99% confidence)
  - p >= 0.05: Not significant (could be random)

- **Cohen's d (Effect Size):**
  - d < 0.2: Small effect
  - 0.2 ≤ d < 0.8: Medium effect
  - d ≥ 0.8: Large effect

**Example:**
```
p = 0.001, Cohen's d = 2.8
→ Highly significant with large effect
→ Results are real and meaningful
```

---

#### Memory Metrics

- **RSS (Resident Set Size):** Total physical memory used
- **Heap:** Memory allocated for objects
- **Peak:** Maximum memory used during execution

**Interpretation:**
- Lower is better
- Monitor peak for memory-constrained environments
- Heap usage indicates allocation efficiency

---

#### I/O Metrics

- **Read Ops:** Number of read operations
- **Write Ops:** Number of write operations
- **Read Bytes:** Total bytes read
- **Write Bytes:** Total bytes written

**Interpretation:**
- Lower ops with higher bytes = more efficient (fewer, larger I/O)
- Monitor for disk-intensive operations
- SSD vs HDD can affect results significantly

---

## Comparison Methodology

### Statistical Testing

#### T-Test for Means

Used to determine if performance differences are statistically significant:

```typescript
// Null hypothesis: No difference between Jujutsu and Git
// Alternative hypothesis: Jujutsu is faster than Git

const tTest = performTTest(jjResults, gitResults);

if (tTest.pValue < 0.05) {
  console.log('Statistically significant difference');
  console.log(`Jujutsu is ${speedup}x faster with ${(1 - tTest.pValue) * 100}% confidence`);
}
```

**Criteria:**
- Alpha level: 0.05 (5% chance of false positive)
- Two-tailed test (can be faster or slower)
- Assumes normal distribution (validated via Shapiro-Wilk test)

---

#### Effect Size Calculation

Measures practical significance beyond statistical significance:

```typescript
const cohenD = calculateCohenD(jjMean, gitMean, pooledStdDev);

if (cohenD >= 0.8) {
  console.log('Large practical effect - meaningful difference');
} else if (cohenD >= 0.2) {
  console.log('Medium effect - noticeable difference');
} else {
  console.log('Small effect - minimal practical difference');
}
```

---

#### Confidence Intervals

Provides range of plausible values for true performance:

```typescript
const ci = calculateConfidenceInterval(results, 0.95);

console.log(`95% CI: [${ci.lower}ms, ${ci.upper}ms]`);
// Interpretation: 95% confident true mean is in this range
```

---

### Outlier Detection

Identifies and handles anomalous results:

```typescript
const outliers = detectOutliers(results, method: 'IQR');
// IQR method: Values beyond 1.5 * IQR from quartiles

if (outliers.length > 0) {
  console.log(`Detected ${outliers.length} outliers`);
  console.log('Rerun with outliers removed for robustness');
}
```

**Outlier Handling:**
1. Identify outliers using IQR or Z-score method
2. Report outliers separately
3. Provide results with and without outliers
4. Investigate causes (system load, disk contention, etc.)

---

### Reproducibility

Ensuring consistent, repeatable results:

**Environmental Controls:**
- Isolated Docker containers
- Fixed resource limits (CPU, memory)
- Network isolation (no external traffic)
- Fresh repository state per iteration

**Statistical Controls:**
- Warmup runs (discard first N iterations)
- Multiple iterations (10-30 recommended)
- Randomized test order (reduce bias)
- Time-based partitioning (avoid clock skew)

**Example:**
```bash
# Good reproducibility setup
npm run bench:run \
  --iterations 20 \
  --warmup 5 \
  --docker \
  --parallel 1  # Sequential for fairness

# Results should vary < 5% across runs
```

---

## Expected Performance Baselines

### Operation-Level Baselines

Based on design specifications (theoretical):

| Operation | Jujutsu | Git | Speedup | Category |
|-----------|---------|-----|---------|----------|
| init | 40-50ms | 150-200ms | 3-4x | Lightweight |
| status | 30-40ms | 100-150ms | 3-4x | Lightweight |
| commit | 80-100ms | 250-350ms | 3x | Medium |
| branch create | 20-30ms | 80-120ms | 4x | Lightweight |
| merge (no conflict) | 150-200ms | 500-700ms | 3x | Heavy |
| rebase | 200-300ms | 800-1200ms | 4x | Heavy |
| log (100 commits) | 50-80ms | 300-500ms | 5x | Query |
| diff (50 files) | 100-150ms | 400-600ms | 4x | Query |

**Note:** These are theoretical estimates. Actual performance CANNOT be measured until implementation is complete and WASM build is fixed.

---

### Repository Size Impact

| Repo Size | Commits | Files | Jujutsu Init | Git Init | Speedup |
|-----------|---------|-------|--------------|----------|---------|
| Small | 100 | 50 | ~40ms | ~180ms | 4.5x |
| Medium | 1,000 | 500 | ~80ms | ~400ms | 5x |
| Large | 10,000 | 5,000 | ~200ms | ~1,200ms | 6x |
| X-Large | 100,000+ | 50,000+ | ~500ms | ~5,000ms | 10x |

**Scaling Characteristics:**
- Jujutsu: Sub-linear scaling (better for large repos)
- Git: Linear to super-linear scaling
- Speedup increases with repository size

---

### Memory Footprint

| Operation | Jujutsu | Git | Reduction |
|-----------|---------|-----|-----------|
| init | 10-15MB | 20-30MB | 50% |
| commit | 15-20MB | 30-40MB | 50% |
| merge | 25-35MB | 50-70MB | 50% |
| log | 20-30MB | 40-60MB | 50% |

**Memory Efficiency:**
- Jujutsu consistently uses 40-60% less memory
- Especially efficient for large repositories
- Better for memory-constrained environments

---

## Contributing Benchmarks

### Adding New Tests

#### Step 1: Create Test File

**File:** `benchmarks/suites/01-basic-operations/my-test.bench.ts`

```typescript
import { BenchmarkTest, BenchmarkContext } from '../../framework/types';

export const myTest: BenchmarkTest = {
  id: 'my-operation',
  name: 'My Custom Operation',
  description: 'Tests performance of my custom operation',
  category: 'basic-operations',

  setup: async (context: BenchmarkContext) => {
    // Prepare test environment
    await context.initRepository();
    await context.createFiles(['file1.txt', 'file2.txt']);
  },

  execute: async (context: BenchmarkContext) => {
    // Execute operation being benchmarked
    const result = await context.jj.execute(['my-operation', '--arg']);
    return result;
  },

  teardown: async (context: BenchmarkContext) => {
    // Clean up test environment
    await context.cleanup();
  },

  validate: async (context: BenchmarkContext, result: any) => {
    // Validate operation succeeded correctly
    return result.success && result.stdout.includes('expected output');
  },

  metrics: ['execution-time', 'memory-usage', 'cpu-usage', 'io-operations']
};
```

---

#### Step 2: Register Test

**File:** `benchmarks/suites/01-basic-operations/index.ts`

```typescript
import { myTest } from './my-test.bench';

export const basicOperationsSuite = {
  id: 'basic-operations',
  name: 'Basic Operations',
  description: 'Core VCS operations',
  tests: [
    // ... existing tests ...
    myTest
  ]
};
```

---

#### Step 3: Run and Validate

```bash
# Run your test
npm run bench:run -- --suite basic-operations --test my-operation

# Verify results
npm run bench:report -- --format markdown

# Check results file
cat benchmarks/results/latest.md | grep "my-operation"
```

---

### Best Practices for Benchmark Tests

**DO:**
- ✅ Use isolated environments (Docker containers)
- ✅ Run warmup iterations (discard first N)
- ✅ Test with multiple repository sizes
- ✅ Validate operation correctness
- ✅ Document test purpose and methodology
- ✅ Use consistent naming conventions

**DON'T:**
- ❌ Benchmark without warmup
- ❌ Mix different repository states
- ❌ Ignore statistical significance
- ❌ Hardcode file paths
- ❌ Skip validation steps
- ❌ Test with external dependencies

---

### Submitting Benchmarks

```bash
# Create branch
git checkout -b benchmarks/my-new-test

# Add test files
git add benchmarks/suites/*/my-test.bench.ts

# Run tests locally
npm run bench:run -- --suite my-suite
npm test

# Commit with descriptive message
git commit -m "feat(benchmarks): Add benchmark for my operation

- Tests performance of my-operation
- Covers small/medium/large repositories
- Includes statistical validation"

# Push and create PR
git push origin benchmarks/my-new-test
gh pr create --title "Add benchmark for my operation"
```

---

## Advanced Configuration

### Parallel Execution

```bash
# Run with 4 parallel containers (4x faster)
npm run bench:run -- --parallel 4

# Run with 8 parallel containers (8x faster, requires 16+ CPU cores)
npm run bench:run -- --parallel 8

# Auto-scale based on CPU
npm run bench:run -- --parallel auto
```

**Scaling Considerations:**
- 1 container: ~30min for full suite
- 4 containers: ~8min for full suite
- 8 containers: ~4min for full suite
- 16+ containers: Diminishing returns (overhead)

---

### Custom Resource Limits

**Docker Compose Override:**

```yaml
# benchmarks/docker/docker-compose.override.yml
services:
  jujutsu-bench:
    deploy:
      resources:
        limits:
          cpus: '4.0'      # 4 CPU cores
          memory: 4G       # 4GB RAM
        reservations:
          cpus: '2.0'
          memory: 2G
```

**Apply override:**
```bash
docker-compose -f docker-compose.yml -f docker-compose.override.yml up
```

---

### AgentDB Learning Integration

```bash
# Enable learning
npm run bench:run -- --agentdb-enabled

# Query historical patterns
npm run bench:analyze -- --historical --task "commit-operation"

# Generate optimization recommendations
npm run bench:optimize -- --based-on-history
```

**Example Output:**
```
📊 Historical Analysis: commit-operation

Past 30 runs:
- Average execution time: 85ms
- Trend: -5% (improving)
- Best performance: 72ms (2025-11-08)
- Worst performance: 102ms (2025-11-01)

💡 Recommendations:
1. Use smaller batches for commits (< 100 files)
2. Avoid commits during high system load
3. Consider pre-warming repository cache
```

---

### CI/CD Integration

**GitHub Actions Example:**

```yaml
# .github/workflows/benchmarks.yml
name: Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight

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
          npm run bench:run -- --parallel 4

      - name: Generate report
        run: |
          cd packages/agentic-jujutsu
          npm run bench:report -- --output benchmarks/results/report.md

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results-${{ github.sha }}
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

## Troubleshooting

### Docker Build Failures

**Problem:**
```
ERROR: failed to solve: failed to compute cache key
```

**Solution:**
```bash
# Clear Docker build cache
docker builder prune -a

# Rebuild from scratch
docker-compose build --no-cache
```

---

### Permission Errors

**Problem:**
```
EACCES: permission denied, open '/results/output.json'
```

**Solution:**
```bash
# Fix permissions on results directory
sudo chown -R $USER:$USER benchmarks/results
chmod -R 755 benchmarks/results
```

---

### High Variance in Results

**Problem:**
```
Standard deviation > 20% of mean
```

**Solution:**
```bash
# Increase iterations
npm run bench:run -- --iterations 30 --warmup 10

# Reduce system load
# - Close other applications
# - Disable background services
# - Run during low-activity hours

# Use dedicated hardware
# - No VMs (use bare metal)
# - SSD instead of HDD
# - Adequate RAM (8GB+ recommended)
```

---

## Next Steps

1. **Fix WASM Build** - Follow main troubleshooting guide
2. **Set Up Docker** - Build and verify containers
3. **Run Basic Benchmarks** - Start with small suites
4. **Analyze Results** - Understand performance characteristics
5. **Contribute** - Add new tests and share results

---

**Document Status:** ✅ COMPLETE
**Last Updated:** 2025-11-09
**Implementation Status:** Design Phase (Week 0 of 10-week roadmap)
**Maintainer:** Agentic Flow Team
