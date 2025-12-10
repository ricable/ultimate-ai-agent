# ReasoningBank - Adaptive Learning System

> Self-learning AI system with pattern recognition and continuous improvement

## 📚 Documentation

### Core Documentation
- [ReasoningBank Architecture](REASONINGBANK_ARCHITECTURE.md) - System design and components
- [Reasoning Agents](REASONING-AGENTS.md) - Agent implementation details
- [Backend Systems](REASONINGBANK_BACKENDS.md) - Storage and persistence
- [Integration Plan](REASONINGBANK_INTEGRATION_PLAN.md) - Integration with agentic-flow

### Implementation & Status
- [Implementation Status](REASONINGBANK_IMPLEMENTATION_STATUS.md)
- [Fixes Applied](REASONINGBANK_FIXES.md)
- [Investigation Report](REASONINGBANK_INVESTIGATION.md)

### Validation & Benchmarks
- [ReasoningBank Validation](REASONINGBANK-VALIDATION.md)
- [Benchmark Results](REASONINGBANK-BENCHMARK-RESULTS.md)
- [Benchmarking Guide](REASONINGBANK-BENCHMARK.md)
- [Memory Validation Report](MEMORY_VALIDATION_REPORT.md)

### Integration
- [CLI Integration](REASONINGBANK-CLI-INTEGRATION.md)
- [Demo Guide](REASONINGBANK-DEMO.md)

## 🚀 Quick Start

### Basic Usage
```bash
# Store a reasoning pattern
npx claude-flow reasoningbank store \
  --task "Solve authentication bug" \
  --reward 0.95 \
  --success true

# Search for similar patterns
npx claude-flow reasoningbank search \
  --task "Fix login issue" \
  --k 5

# Get pattern statistics
npx claude-flow reasoningbank stats \
  --task "authentication"
```

## 🎯 Key Features

### Pattern Learning
- **Trajectory Tracking** - Record complete problem-solving paths
- **Verdict Judgment** - Evaluate solution quality
- **Memory Distillation** - Compress and optimize learned patterns
- **Similarity Scoring** - Find relevant past experiences

### Adaptive Improvement
- **Experience Replay** - Learn from past successes and failures
- **Meta-Learning** - Transfer knowledge across domains
- **Continuous Refinement** - Improve strategies over time
- **Context-Aware Retrieval** - Match patterns to current situations

---

**Back to**: [Features](../README.md) | [Main Documentation](../../README.md)
