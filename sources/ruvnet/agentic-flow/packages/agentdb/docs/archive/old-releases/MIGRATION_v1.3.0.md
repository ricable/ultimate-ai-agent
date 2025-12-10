# Migration Guide: AgentDB v1.2.2 → v1.3.0

**Release Date:** October 22, 2025
**Breaking Changes:** None
**New Tools:** 15 (10 learning + 5 core AgentDB)
**Total Tools:** 29 (up from 14)

---

## 📋 Overview

AgentDB v1.3.0 adds **two major feature categories**:

1. **Learning System Tools (10)** - Full reinforcement learning pipeline with 9 algorithms
2. **Core AgentDB Tools (5)** - Advanced database management and reasoning patterns

**Backward Compatibility:** ✅ **100% compatible** with v1.2.2. All existing tools work identically. This is an **additive release** with no breaking changes.

---

## 🚀 Quick Upgrade

### Step 1: Update Package

```bash
# NPM
npm install agentdb@1.3.0

# Yarn
yarn add agentdb@1.3.0

# PNPM
pnpm add agentdb@1.3.0

# Global (for MCP server)
npm install -g agentdb@1.3.0
```

### Step 2: Verify Installation

```bash
# Check version
npx agentdb --version
# Should output: 1.3.0

# List MCP tools
npx agentdb mcp list
# Should show 29 tools
```

### Step 3: Restart MCP Server

If using Claude Desktop or MCP integration:

```bash
# Restart Claude Desktop
# OR reload MCP server configuration
```

**That's it!** Your existing code continues to work. New tools are available immediately.

---

## 🆕 What's New in v1.3.0

### Category 1: Learning System Tools (10 NEW)

Full reinforcement learning pipeline for adaptive AI agents.

#### Session Management (2 tools)

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `learning_start_session` | Initialize RL session with algorithm | `user_id`, `session_type` (q-learning/dqn/ppo/etc), `config` |
| `learning_end_session` | Finalize and save trained policy | `session_id` |

**Example:**
```json
{
  "name": "learning_start_session",
  "arguments": {
    "user_id": "agent-123",
    "session_type": "q-learning",
    "config": {
      "learning_rate": 0.01,
      "discount_factor": 0.99,
      "exploration_rate": 0.1
    }
  }
}
```

#### Adaptive Intelligence (3 tools)

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `learning_predict` | Get AI-recommended action | `session_id`, `state` |
| `learning_feedback` | Submit reward signal | `session_id`, `state`, `action`, `reward`, `success` |
| `learning_train` | Batch policy training | `session_id`, `epochs`, `batch_size` |

**Example:**
```json
{
  "name": "learning_predict",
  "arguments": {
    "session_id": "session-abc123",
    "state": "user authentication failed"
  }
}
```

#### Analytics & Advanced (5 tools)

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `learning_metrics` | Performance tracking | `session_id`, `time_window_days`, `include_trends` |
| `learning_transfer` | Transfer knowledge | `source_session`, `target_session`, `min_similarity` |
| `learning_explain` | Explainable recommendations | `query`, `explain_depth`, `include_evidence` |
| `experience_record` | Experience replay | `session_id`, `tool_name`, `action`, `outcome`, `reward` |
| `reward_signal` | Automated reward calculation | `success`, `efficiency_score`, `quality_score` |

**Supported RL Algorithms:**
- **Q-Learning** - Simple state-action value learning
- **SARSA** - On-policy temporal difference learning
- **DQN** - Deep Q-Network with experience replay
- **Policy Gradient** - Direct policy optimization
- **Actor-Critic** - Hybrid value and policy learning
- **PPO** - Proximal Policy Optimization
- **Decision Transformer** - Transformer-based RL
- **MCTS** - Monte Carlo Tree Search
- **Model-Based** - Learn environment model

---

### Category 2: Core AgentDB Tools (5 NEW)

Advanced database management and reasoning pattern storage.

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `agentdb_stats` | Comprehensive database statistics | `detailed` (boolean) |
| `agentdb_pattern_store` | Store reasoning patterns | `taskType`, `approach`, `successRate` |
| `agentdb_pattern_search` | Search patterns semantically | `task`, `k`, `threshold`, `filters` |
| `agentdb_pattern_stats` | Pattern analytics | (none) |
| `agentdb_clear_cache` | Cache management | `cache_type` (all/patterns/stats) |

**Example:**
```json
{
  "name": "agentdb_pattern_store",
  "arguments": {
    "taskType": "code_review",
    "approach": "Security-first analysis followed by code quality checks",
    "successRate": 0.95,
    "tags": ["security", "quality"]
  }
}
```

---

## 🔄 Migration Checklist

### For MCP Users (Claude Desktop, etc.)

- [x] **Update AgentDB:** `npm install -g agentdb@1.3.0`
- [x] **Restart Claude Desktop:** Reload MCP server
- [x] **Verify Tools:** Check that 29 tools are available
- [ ] **Explore Learning Tools:** Try `learning_start_session`
- [ ] **Explore Pattern Tools:** Try `agentdb_pattern_store`

### For CLI Users

- [x] **Update Package:** `npm install agentdb@1.3.0`
- [x] **Verify Version:** `npx agentdb --version`
- [ ] **Explore New Commands:** (Note: Learning tools currently MCP-only)

### For SDK Users

- [x] **Update Package:** Update in `package.json`
- [x] **Review New APIs:** Check `LearningSystem` and `ReasoningBank` controllers
- [ ] **Integrate Learning:** Add RL capabilities to your agents
- [ ] **Store Patterns:** Use `ReasoningBank` for reasoning patterns

---

## 📊 Tool Count Comparison

| Version | Core Vector DB | Core AgentDB | Frontier Memory | Learning System | **Total** |
|---------|----------------|--------------|-----------------|-----------------|-----------|
| v1.2.2 | 5 | 0 | 9 | 0 | **14** |
| v1.3.0 | 5 | **5 NEW** | 9 | **10 NEW** | **29** |

---

## 🎯 Use Cases for New Tools

### Learning System Use Cases

**1. Adaptive Debugging Assistant**
```javascript
// Start learning session
learning_start_session({
  user_id: "dev-123",
  session_type: "q-learning",
  config: { learning_rate: 0.01 }
});

// Get recommendation
learning_predict({
  session_id: "session-1",
  state: "authentication error in production"
});

// Provide feedback
learning_feedback({
  session_id: "session-1",
  state: "authentication error",
  action: "check JWT token expiry",
  reward: 0.95,
  success: true
});

// Train policy
learning_train({
  session_id: "session-1",
  epochs: 50
});
```

**2. Cross-Domain Knowledge Transfer**
```javascript
// Transfer debugging patterns to API troubleshooting
learning_transfer({
  source_session: "debug-session",
  target_session: "api-session",
  source_task: "database connection errors",
  target_task: "API timeout errors",
  min_similarity: 0.7,
  transfer_type: "all"
});
```

**3. Explainable Recommendations**
```javascript
// Get recommendations with evidence
learning_explain({
  query: "how to fix authentication failures",
  explain_depth: "detailed",
  include_evidence: true,
  include_causal: true
});
```

### Core AgentDB Use Cases

**1. Reasoning Pattern Library**
```javascript
// Store successful reasoning approach
agentdb_pattern_store({
  taskType: "code_review",
  approach: "1. Security scan, 2. Performance analysis, 3. Code quality",
  successRate: 0.92,
  tags: ["security", "performance"]
});

// Search for relevant patterns
agentdb_pattern_search({
  task: "review authentication code",
  k: 5,
  threshold: 0.7,
  filters: { taskType: "code_review", minSuccessRate: 0.8 }
});
```

**2. Database Health Monitoring**
```javascript
// Get comprehensive statistics
agentdb_stats({ detailed: true });

// Returns:
// - Table counts (episodes, skills, patterns, sessions)
// - Storage usage (MB)
// - Recent activity (7 days)
// - Pattern analytics
```

**3. Performance Optimization**
```javascript
// Clear cache after bulk updates
agentdb_clear_cache({ cache_type: "patterns" });

// Refresh statistics
agentdb_pattern_stats();
```

---

## 🔧 Configuration Changes

### New Environment Variables (Optional)

```bash
# Learning System Configuration
export AGENTDB_LEARNING_RATE=0.01        # Default learning rate
export AGENTDB_DISCOUNT_FACTOR=0.99      # Discount factor (gamma)
export AGENTDB_EXPLORATION_RATE=0.1      # Epsilon for ε-greedy
export AGENTDB_BATCH_SIZE=32             # Batch size for training

# Pattern Storage Configuration
export AGENTDB_PATTERN_CACHE_SIZE=1000   # Max cached patterns
export AGENTDB_PATTERN_TTL=86400         # Pattern cache TTL (seconds)
```

**Note:** These are optional. Defaults work well for most use cases.

---

## 📖 Documentation Updates

### Updated Documentation

- **[MCP_TOOLS.md](./docs/MCP_TOOLS.md)** - Complete reference for all 29 tools
- **[README.md](./README.md)** - Updated with v1.3.0 features
- **[CHANGELOG.md](./CHANGELOG.md)** - Full v1.3.0 release notes

### New Examples

- **Adaptive Learning Example:** Build agents that learn from experience
- **Transfer Learning Example:** Reuse knowledge across domains
- **Pattern Library Example:** Store and retrieve reasoning patterns

---

## ⚠️ Known Issues

### None for v1.3.0

All 29 tools are production-ready and fully tested.

---

## 🧪 Testing Your Migration

### Test 1: Verify Tool Count

```bash
# Should show 29 tools
npx agentdb mcp list | grep -c "Tool:"
```

### Test 2: Try Learning System

```javascript
// Start a simple Q-learning session
mcp_call("learning_start_session", {
  user_id: "test-user",
  session_type: "q-learning",
  config: {
    learning_rate: 0.01,
    discount_factor: 0.99
  }
});

// Should return session_id
```

### Test 3: Try Pattern Storage

```javascript
// Store a pattern
mcp_call("agentdb_pattern_store", {
  taskType: "test_task",
  approach: "Test approach",
  successRate: 0.9
});

// Search for the pattern
mcp_call("agentdb_pattern_search", {
  task: "test task",
  k: 5
});

// Should find the stored pattern
```

### Test 4: Verify Backward Compatibility

```javascript
// All v1.2.2 tools should still work
mcp_call("agentdb_insert", {
  text: "Test vector",
  session_id: "test"
});

mcp_call("agentdb_search", {
  query: "test",
  k: 5
});

// Should work identically to v1.2.2
```

---

## 🚦 Performance Impact

### Learning System

| Operation | Latency | Notes |
|-----------|---------|-------|
| `learning_start_session` | 10-50ms | Creates session + initializes policy |
| `learning_predict` | 3-10ms | Cached policy lookup |
| `learning_feedback` | 5-15ms | Updates Q-values incrementally |
| `learning_train` (50 epochs) | 1-2s | Optimized batch learning |
| `learning_metrics` | 20-100ms | Aggregates statistics |

### Core AgentDB

| Operation | Latency | Notes |
|-----------|---------|-------|
| `agentdb_stats` | 50-100ms | Queries all table counts |
| `agentdb_pattern_store` | 100-300ms | Generates embedding + stores |
| `agentdb_pattern_search` | 200-500ms | Vector similarity search |
| `agentdb_pattern_stats` | 50-150ms | Aggregates pattern analytics |
| `agentdb_clear_cache` | <50ms | In-memory operation |

**Storage Impact:**
- Learning sessions: ~100KB per session (includes policy + experiences)
- Reasoning patterns: ~2KB per pattern (includes embedding)

---

## 💡 Best Practices

### Learning System

1. **Start Simple:** Begin with Q-learning for basic state-action mapping
2. **Provide Feedback:** Always provide feedback after actions for policy improvement
3. **Train Regularly:** Use `learning_train` after collecting 50-100 experiences
4. **Monitor Metrics:** Track `learning_metrics` to ensure policy is improving
5. **Transfer Knowledge:** Use `learning_transfer` for similar tasks to accelerate learning
6. **Explain Decisions:** Use `learning_explain` to build trust and debug

### Pattern Storage

1. **Tag Patterns:** Use consistent tags for easier filtering
2. **Track Success Rates:** Update success rates as you use patterns
3. **Search Semantically:** Use natural language queries for pattern search
4. **Monitor Stats:** Check `agentdb_pattern_stats` to identify gaps
5. **Clear Cache:** Use `agentdb_clear_cache` after bulk pattern updates

---

## 🆘 Troubleshooting

### Issue: "Tool not found" error

**Solution:** Restart MCP server or Claude Desktop

```bash
# For Claude Desktop: Quit and restart app
# For custom MCP server: Kill and restart process
```

### Issue: Learning session not persisting

**Solution:** Always call `learning_end_session` before terminating

```javascript
// Proper session cleanup
learning_end_session({ session_id: "your-session-id" });
```

### Issue: Pattern search returning no results

**Solution:** Check similarity threshold and filters

```javascript
// Lower threshold if no results
agentdb_pattern_search({
  task: "your query",
  threshold: 0.5,  // Lower threshold
  k: 10
});
```

### Issue: High latency on `agentdb_stats`

**Solution:** Use non-detailed mode for faster results

```javascript
// Faster statistics (no storage metrics)
agentdb_stats({ detailed: false });
```

---

## 📞 Support

- **Documentation:** [docs/MCP_TOOLS.md](./docs/MCP_TOOLS.md)
- **Issues:** [GitHub Issues](https://github.com/ruvnet/agentic-flow/issues)
- **Discussions:** [GitHub Discussions](https://github.com/ruvnet/agentic-flow/discussions)

---

## 🎉 Next Steps

1. ✅ Complete migration using checklist above
2. 📚 Read [MCP_TOOLS.md](./docs/MCP_TOOLS.md) for complete tool reference
3. 🧪 Experiment with learning tools in Claude Desktop
4. 🔍 Store reasoning patterns for your common tasks
5. 📊 Monitor learning progress with `learning_metrics`
6. 🚀 Build adaptive agents with transfer learning

---

**Congratulations!** You're now ready to use AgentDB v1.3.0 with all 29 production-ready MCP tools.

---

**Migration Guide Version:** 1.0
**AgentDB Version:** 1.3.0
**Date:** October 22, 2025
**Status:** ✅ Production Ready
