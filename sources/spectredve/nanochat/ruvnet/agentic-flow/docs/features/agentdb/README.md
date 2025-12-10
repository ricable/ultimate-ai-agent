# AgentDB - Vector Database & Learning Systems

> High-performance vector database with 150x faster search and 9 reinforcement learning algorithms

## 📚 Documentation

### Core Documentation
- [CLI Guide](CLI_GUIDE.md) - Command-line interface reference
- [MCP Tools Implementation](MCP_TOOLS_IMPLEMENTATION.md) - MCP integration details
- [Learning Tools](LEARNING_TOOLS_6-10.md) - Reinforcement learning tools
- [Core Tools Implementation](CORE_TOOLS_6-10_IMPLEMENTATION.md) - Core functionality

### Features
- [Causal Recall Search](CAUSAL_RECALL_SEARCH_IMPLEMENTATION.md) - Advanced search capabilities
- [Test Suite](TEST_SUITE_SUMMARY.md) - Testing and validation

### Integration & Architecture
- [AgentDB Integration Plan](AGENTDB_INTEGRATION_PLAN.md)
- [Integration Complete](AGENTDB-INTEGRATION-COMPLETE.md)
- [Learning Systems Report](AGENTDB-LEARNING-SYSTEMS-REPORT.md)
- [MCP Security Audit](AGENTDB-MCP-SECURITY-AUDIT.md)
- [WASM Vector Implementation](WASM-VECTOR-IMPLEMENTATION.md)

### Validation
- [Code Quality Analysis](agentdb-code-quality-analysis.md)
- [Tools Verification](agentdb-tools-verification.md)

## 🚀 Quick Start

### Installation
```bash
npm install agentic-flow
```

### Basic Usage
```bash
# Initialize AgentDB
npx claude-flow agentdb init

# Store vectors
npx claude-flow agentdb store --embedding "[0.1, 0.2, ...]" --metadata "document"

# Search
npx claude-flow agentdb search --query "[0.1, 0.2, ...]" --k 5
```

## 🎯 Key Features

### Performance
- **150x faster** vector search vs traditional databases
- **HNSW indexing** for approximate nearest neighbor search
- **Quantization** support (4-32x memory reduction)
- **WASM acceleration** for compute-intensive operations

### Learning Algorithms
1. Decision Transformer
2. Q-Learning
3. SARSA
4. Actor-Critic
5. Policy Gradient
6. Monte Carlo
7. Temporal Difference
8. Deep Q-Network (DQN)
9. Proximal Policy Optimization (PPO)

### MCP Integration
- Vector storage and retrieval
- Pattern search and matching
- Learning algorithm execution
- Real-time training feedback

## 📊 Performance Benchmarks

See [Benchmarks](../../validation/benchmarks/README.md) for detailed performance data.

## 🔗 Related Documentation
- [ReasoningBank Integration](../reasoningbank/README.md)
- [MCP Tools Guide](../../guides/mcp/MCP-TOOLS.md)
- [Architecture Overview](../../architecture/README.md)

## 📝 Version History
- [v1.3.0 Release](VERSION_1.3.0_RELEASE.md)

---

**Back to**: [Features](../README.md) | [Main Documentation](../../README.md)
