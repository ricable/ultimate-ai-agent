# ReasoningBank Benchmark Guide

## Overview

This benchmark suite provides comprehensive evaluation of ReasoningBank's closed-loop learning system compared to a baseline agent without memory capabilities.

## Quick Start

### Prerequisites

```bash
# Ensure you have ANTHROPIC_API_KEY set
export ANTHROPIC_API_KEY="sk-ant-..."

# Install dependencies from parent directory
cd /workspaces/agentic-flow
npm install
```

### Running Benchmarks

```bash
# Run all benchmarks (default: 3 iterations per scenario)
npm run bench

# Run specific scenario
npm run bench:coding
npm run bench:debugging
npm run bench:api
npm run bench:problem-solving

# Quick test (1 iteration)
npm run bench:quick

# Comprehensive test (5 iterations)
npm run bench:full

# Clean up results
npm run bench:clean
```

### Direct Execution

```bash
# From bench directory
node benchmark.js

# With custom config
node benchmark.js --config=custom-config.json

# Specify scenarios
node benchmark.js --scenarios=coding-tasks,debugging-tasks

# Custom iterations
node benchmark.js --iterations=5
```

## Benchmark Structure

### Directory Layout

```
bench/
├── README.md                   # Main documentation
├── BENCHMARK-GUIDE.md          # This guide
├── config.json                 # Configuration file
├── package.json                # Scripts and dependencies
├── benchmark.ts                # Main orchestrator
├── agents/
│   ├── baseline-agent.ts       # Control group (no memory)
│   └── reasoningbank-agent.ts  # Experimental group (with ReasoningBank)
├── scenarios/
│   ├── coding-tasks.ts         # 10 coding tasks
│   ├── debugging-tasks.ts      # 10 debugging tasks
│   ├── api-design-tasks.ts     # 10 API design tasks
│   └── problem-solving-tasks.ts # 10 algorithmic tasks
├── lib/
│   ├── types.ts                # Type definitions
│   ├── metrics.ts              # Metrics collector
│   └── report-generator.ts     # Report generation
├── results/                    # JSON results (gitignored)
└── reports/                    # Markdown reports (gitignored)
```

## Scenarios

### 1. Coding Tasks (coding-tasks)
**Focus**: Implementation of common programming patterns

**Tasks** (10 total):
- Array deduplication
- Deep clone
- Debounce function
- Promise retry with backoff
- LRU cache
- Binary search
- Flatten nested arrays
- Throttle function
- Event emitter
- Memoization

**Expected Learning**: Agent should learn patterns for handling collections, async operations, and caching strategies.

### 2. Debugging Tasks (debugging-tasks)
**Focus**: Identifying and fixing bugs

**Tasks** (10 total):
- Off-by-one errors
- Race conditions
- Memory leaks
- Type coercion bugs
- Closure issues
- Promise error handling
- Null references
- Stack overflow
- Infinite loops
- State mutation bugs

**Expected Learning**: Agent should build knowledge about common bug patterns and their solutions.

### 3. API Design Tasks (api-design-tasks)
**Focus**: RESTful API design and best practices

**Tasks** (10 total):
- User authentication API
- CRUD endpoints
- Pagination design
- Rate limiting
- API versioning
- Error response schemas
- File upload API
- Search and filtering
- Webhook system
- GraphQL schema

**Expected Learning**: Agent should learn REST principles, HTTP methods, status codes, and API patterns.

### 4. Problem Solving Tasks (problem-solving-tasks)
**Focus**: Algorithmic problem solving

**Tasks** (10 total):
- Two sum problem
- Valid parentheses
- Longest substring without repeating
- Merge intervals
- Binary tree traversal
- Word ladder
- Coin change (DP)
- Serialize/deserialize tree
- Trapping rain water
- Regular expression matching

**Expected Learning**: Agent should learn data structures (stack, queue, hash map) and algorithms (BFS, DFS, dynamic programming).

## Metrics Explained

### 1. Success Rate
**Formula**: `successful_tasks / total_tasks`

**Interpretation**:
- 0% → 100%: Excellent learning (matches paper expectations)
- +50% or more: Excellent improvement
- +20-50%: Good improvement
- +0-20%: Modest improvement
- Negative: Investigate memory quality

**Why It Matters**: Core metric showing if ReasoningBank actually improves task success through learning.

### 2. Learning Velocity
**Formula**: `baseline_iterations_to_success / reasoningbank_iterations_to_success`

**Interpretation**:
- 2-3x faster: Excellent acceleration
- 1.5-2x faster: Good improvement
- <1.5x: Modest improvement

**Why It Matters**: Shows how quickly ReasoningBank learns compared to baseline's trial-and-error.

### 3. Token Efficiency
**Formula**: `(baseline_tokens - reasoningbank_tokens) / baseline_tokens × 100`

**Interpretation**:
- +20% or more: Excellent savings (paper reports 32.3%)
- +10-20%: Good efficiency
- 0-10%: Modest savings
- Negative: Memory overhead exceeds benefits

**Why It Matters**: Demonstrates cost savings from memory injection replacing redundant reasoning.

### 4. Latency Impact
**Formula**: `(reasoningbank_latency - baseline_latency) / baseline_latency × 100`

**Interpretation**:
- <10%: Acceptable overhead
- 10-25%: Moderate overhead
- >25%: High overhead (investigate DB performance)

**Why It Matters**: Shows real-world performance impact of memory operations.

### 5. Memory Efficiency
**Metrics**:
- `memories_created`: New memories distilled per scenario
- `memories_used`: Memories retrieved and applied
- `usage_ratio`: memories_used / memories_created

**Interpretation**:
- High usage ratio (>0.5): Memories are being reused effectively
- Low usage ratio (<0.2): Too many memories or poor retrieval

**Why It Matters**: Indicates quality and relevance of distilled memories.

### 6. Confidence
**Formula**: Extracted from judge's verdict (0-1 scale)

**Interpretation**:
- >0.8: High confidence, reliable results
- 0.5-0.8: Moderate confidence
- <0.5: Low confidence, uncertain results

**Why It Matters**: Self-assessment of result quality, useful for filtering.

### 7. Accuracy
**Measurement**: Manual review of outputs against expected solutions

**Why It Matters**: Validates that success criteria correctly identify quality solutions.

## Configuration

### config.json Structure

```json
{
  "execution": {
    "iterations": 3,           // Iterations per scenario
    "parallelAgents": 1,       // Sequential execution
    "enableWarmStart": false,  // Start with empty memory
    "memorySize": 1000        // Max memories to store
  },
  "agents": {
    "reasoningbank": {
      "memoryConfig": {
        "k": 3,                // Retrieve top 3 memories
        "alpha": 0.65,         // Similarity weight
        "beta": 0.15,          // Recency weight
        "gamma": 0.20,         // Reliability weight
        "delta": 0.10          // Diversity weight
      }
    }
  }
}
```

### Tuning Parameters

**k (retrieval count)**:
- Lower (1-2): Faster, less context
- Higher (5-10): More context, slower, potential noise
- Default: 3 (balanced)

**alpha (similarity weight)**:
- Higher: Prioritize semantic relevance
- Lower: Prioritize other factors
- Default: 0.65 (high similarity focus)

**beta (recency weight)**:
- Higher: Prefer recent memories
- Lower: Age-agnostic retrieval
- Default: 0.15 (modest recency bias)

**gamma (reliability weight)**:
- Higher: Trust proven patterns
- Lower: Explore all memories
- Default: 0.20 (moderate reliability)

**delta (diversity weight)**:
- Higher: Avoid redundant memories
- Lower: Allow similar memories
- Default: 0.10 (modest diversity)

## Expected Results

Based on the ReasoningBank paper:

### Success Rate Transformation
- **Before**: 0% (cold start)
- **After**: ~100% (after learning)
- **Improvement**: +100 percentage points

### Token Efficiency
- **Savings**: 32.3% token reduction
- **Mechanism**: Memory injection replaces redundant reasoning

### Learning Velocity
- **Baseline**: Trial-and-error across all iterations
- **ReasoningBank**: Rapid improvement after initial learning
- **Speedup**: 2-4x faster to consistent success

### Memory Growth
- **Initial**: 0 memories
- **After 3 iterations**: ~20-30 memories per scenario
- **Quality**: High-confidence memories (>0.7) dominate

## Interpreting Reports

### Markdown Report Structure

1. **Executive Summary**: Overall metrics and recommendations
2. **Detailed Scenario Results**: Per-scenario breakdown with learning curves
3. **Methodology**: Explanation of agents, metrics, and scoring
4. **Interpretation Guide**: How to understand the numbers
5. **Appendix**: Configuration and environment details

### Key Sections to Review

**Success Rate Delta**:
```
Baseline: 20% → ReasoningBank: 80% (+60%)
```
This shows ReasoningBank learned to solve 60% more tasks.

**Learning Curve Table**:
```
| Iteration | Baseline | ReasoningBank | Memories |
|-----------|----------|---------------|----------|
| 1         | 20%      | 10%           | 0        |
| 2         | 20%      | 60%           | 15       |
| 3         | 20%      | 80%           | 28       |
```
This shows:
- Baseline stays constant (no learning)
- ReasoningBank starts lower (cold start)
- ReasoningBank rapidly improves (learning effect)
- Memory bank grows with experience

**Token Efficiency**:
```
Baseline: 1,200 tokens/task
ReasoningBank: 850 tokens/task (-29.2%)
```
This demonstrates cost savings from memory injection.

## Troubleshooting

### Low Success Rates
**Symptoms**: ReasoningBank not improving, staying near baseline

**Possible Causes**:
1. Success criteria too strict
2. Model temperature too high (set to 0.7)
3. Insufficient iterations (increase to 5+)
4. Poor memory retrieval (tune k parameter)

**Solutions**:
- Review success criteria in scenario files
- Increase iterations in config.json
- Adjust k parameter (try 5 instead of 3)
- Check database has memories: `npx agentic-flow reasoningbank status`

### High Token Usage
**Symptoms**: ReasoningBank using MORE tokens than baseline

**Possible Causes**:
1. k too high (retrieving too many memories)
2. Memories too verbose
3. Poor distillation

**Solutions**:
- Lower k to 2-3
- Adjust distillation prompts to be more concise
- Review memory quality: `npx agentic-flow reasoningbank list`

### High Latency
**Symptoms**: ReasoningBank significantly slower than baseline

**Possible Causes**:
1. Database not using WAL mode
2. Too many embeddings to search
3. Slow consolidation

**Solutions**:
- Verify WAL mode: Check .swarm/memory.db-wal exists
- Increase consolidation threshold
- Add indexes to database
- Use hash embeddings instead of neural embeddings

### No Learning Effect
**Symptoms**: Learning curve is flat, no improvement across iterations

**Possible Causes**:
1. Judge not identifying failures correctly
2. Distillation not extracting useful patterns
3. Retrieval not finding relevant memories
4. Database not persisting between iterations

**Solutions**:
- Check judge verdicts in verbose output
- Review distilled memories for quality
- Test retrieval: `npx agentic-flow reasoningbank list --query="implement debounce"`
- Verify database persists: Check .swarm/memory.db file size grows

## Advanced Usage

### Custom Scenarios

Create new scenario file in `bench/scenarios/`:

```typescript
import type { BenchmarkScenario, Task } from '../lib/types.js';

const tasks: Task[] = [
  {
    id: 'custom-01',
    name: 'Custom Task',
    description: 'Task description',
    domain: 'custom',
    input: 'Task input',
    successCriteria: (result) => {
      // Custom validation logic
      return result.output.includes('expected pattern');
    },
    difficulty: 'medium'
  }
];

export const customScenario: BenchmarkScenario = {
  name: 'custom-scenario',
  description: 'Description',
  domain: 'custom',
  tasks,
  metrics: ['success_rate', 'token_efficiency', 'latency']
};
```

### Warm Start Benchmarks

Test with pre-populated memory:

```typescript
// In scenario file
export const warmStartScenario: BenchmarkScenario = {
  name: 'warm-start',
  warmStartMemories: [
    {
      pattern: 'Use Set for deduplication',
      confidence: 0.9,
      domain: 'coding'
    }
  ],
  // ... rest of scenario
};
```

### Comparative Analysis

Run multiple configurations:

```bash
# Default config
npm run bench -- --config=config.json

# High memory config
npm run bench -- --config=config-high-memory.json

# Low latency config
npm run bench -- --config=config-low-latency.json

# Compare results
node scripts/compare-results.js results/*.json
```

## Contributing

### Adding New Scenarios

1. Create scenario file in `bench/scenarios/`
2. Import in `benchmark.ts`
3. Add to scenarios map
4. Update config.json
5. Test with `npm run bench:quick`

### Improving Metrics

1. Add metric type to `lib/types.ts`
2. Update `MetricsCollector` in `lib/metrics.ts`
3. Update report generator
4. Document in README.md

### Optimizing Performance

1. Use hash embeddings for offline mode
2. Enable WAL mode for database
3. Batch memory operations
4. Cache frequent queries
5. Use connection pooling

## References

- [ReasoningBank Paper](https://arxiv.org/abs/paper-id)
- [agentic-flow Documentation](../README.md)
- [ReasoningBank CLI Guide](../agentic-flow/src/reasoningbank/README.md)

## Support

For issues or questions:
- GitHub Issues: https://github.com/ruvnet/agentic-flow/issues
- Documentation: https://github.com/ruvnet/agentic-flow
