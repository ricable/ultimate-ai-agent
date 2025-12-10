# AgentDB Learning System Tools 6-10

**Version:** 1.4.0
**Implementation Date:** 2025-10-22
**Status:** ✅ Complete

## Overview

Extended AgentDB Learning System with 5 advanced tools for performance metrics, transfer learning, explainable AI (XAI), experience recording, and reward shaping.

## New Tools Implemented

### 6. `learning_metrics` - Performance Metrics Analysis

**Purpose:** Get comprehensive learning performance metrics with time windows and trend analysis.

**Parameters:**
- `session_id` (optional): Filter metrics by specific session
- `time_window_days` (default: 7): Time window in days
- `include_trends` (default: true): Include trend analysis
- `group_by` (default: 'task'): Group by task/session/skill

**Returns:**
```json
{
  "timeWindow": { "days": 7, "startTimestamp": 123, "endTimestamp": 456 },
  "overall": {
    "totalEpisodes": 100,
    "avgReward": 0.75,
    "successRate": 0.85,
    "minReward": 0.0,
    "maxReward": 1.0,
    "avgLatencyMs": 150
  },
  "groupedMetrics": [
    { "key": "task_name", "count": 50, "avgReward": 0.8, "successRate": 0.9 }
  ],
  "trends": [
    { "date": "2025-10-15", "count": 10, "avgReward": 0.7, "successRate": 0.8 }
  ],
  "policyImprovement": {
    "versions": 5,
    "qValueImprovement": 0.15
  }
}
```

**Use Cases:**
- Monitor learning progress over time
- Identify high-performing tasks/sessions
- Track policy improvement
- Analyze success rates and rewards

---

### 7. `learning_transfer` - Transfer Learning

**Purpose:** Transfer knowledge between sessions or tasks, enabling reuse across different contexts.

**Parameters:**
- `source_session` (optional): Source session ID
- `target_session` (optional): Target session ID
- `source_task` (optional): Source task pattern
- `target_task` (optional): Target task pattern
- `min_similarity` (default: 0.7): Minimum similarity threshold (0-1)
- `transfer_type` (default: 'all'): Type of transfer (episodes/skills/causal_edges/all)
- `max_transfers` (default: 10): Maximum items to transfer

**Returns:**
```json
{
  "success": true,
  "transferred": {
    "episodes": 8,
    "skills": 12,
    "causalEdges": 5,
    "details": [
      { "type": "episode", "id": 123, "similarity": 0.85 }
    ]
  },
  "source": { "session": "session-123", "task": null },
  "target": { "session": "session-456", "task": null },
  "minSimilarity": 0.7,
  "transferType": "all"
}
```

**Use Cases:**
- Reuse learned knowledge in new contexts
- Bootstrap new learning sessions
- Cross-domain knowledge transfer
- Accelerate learning with prior experience

---

### 8. `learning_explain` - Explainable AI (XAI)

**Purpose:** Explain action recommendations with confidence scores and supporting evidence.

**Parameters:**
- `query` (required): Task description to get recommendations for
- `k` (default: 5): Number of recommendations
- `explain_depth` (default: 'detailed'): Explanation level (summary/detailed/full)
- `include_confidence` (default: true): Include confidence scores
- `include_evidence` (default: true): Include supporting evidence
- `include_causal` (default: true): Include causal reasoning chains

**Returns:**
```json
{
  "query": "optimize database query",
  "recommendations": [
    {
      "action": "add_index",
      "confidence": 0.85,
      "avgReward": 0.92,
      "successRate": 0.95,
      "supportingExamples": 10,
      "evidence": [
        {
          "episodeId": 123,
          "state": "slow_query",
          "reward": 0.95,
          "success": true,
          "similarity": 0.88,
          "timestamp": 1729000000
        }
      ]
    }
  ],
  "reasoning": {
    "similarExperiencesFound": 15,
    "avgSimilarity": 0.82,
    "uniqueActions": 5
  },
  "causalChains": [
    { "fromMemoryType": "add_index", "toMemoryType": "fast_query", "uplift": 0.45 }
  ]
}
```

**Use Cases:**
- Understand AI decision-making
- Debug model predictions
- Build trust with transparent recommendations
- Audit AI behavior

---

### 9. `experience_record` - Tool Execution Recording

**Purpose:** Record tool execution as experience for offline learning and experience replay.

**Parameters:**
- `session_id` (required): Session identifier
- `tool_name` (required): Name of the tool executed
- `action` (required): Action taken or tool parameters
- `state_before` (optional): System state before action (JSON)
- `state_after` (optional): System state after action (JSON)
- `outcome` (required): Outcome description
- `reward` (required): Reward signal (0-1)
- `success` (required): Whether action succeeded
- `latency_ms` (optional): Execution latency
- `metadata` (optional): Additional metadata (JSON)

**Returns:**
```json
{
  "experienceId": 789,
  "message": "Experience recorded for offline learning"
}
```

**Use Cases:**
- Log tool executions for learning
- Build experience replay buffers
- Train policies from historical data
- Offline reinforcement learning

---

### 10. `reward_signal` - Reward Calculation

**Purpose:** Calculate reward signals based on success, efficiency, quality, and causal impact.

**Parameters:**
- `episode_id` (optional): Episode ID for causal impact
- `success` (required): Whether outcome was successful
- `target_achieved` (default: true): Whether target was achieved
- `efficiency_score` (default: 0.5): Efficiency score (0-1)
- `quality_score` (default: 0.5): Quality score (0-1)
- `time_taken_ms` (optional): Time taken in milliseconds
- `expected_time_ms` (optional): Expected time in milliseconds
- `include_causal` (default: true): Include causal impact
- `reward_function` (default: 'standard'): Function type (standard/sparse/dense/shaped)

**Reward Functions:**
1. **Standard**: Balanced weighted combination
2. **Sparse**: Only reward on full success
3. **Dense**: Partial rewards for progress
4. **Shaped**: Time-efficient reward shaping

**Returns:**
```json
{
  "reward": 0.85,
  "rewardFunction": "standard",
  "factors": {
    "success": true,
    "targetAchieved": true,
    "efficiencyScore": 0.8,
    "qualityScore": 0.9,
    "timeEfficiency": 1.2,
    "causalImpact": 0.05
  }
}
```

**Use Cases:**
- Calculate training rewards
- Reward shaping for RL
- Multi-objective optimization
- Causal impact analysis

---

## Implementation Details

### Controller Methods

**File:** `/workspaces/agentic-flow/packages/agentdb/src/controllers/LearningSystem.ts`

```typescript
// Method signatures
async getMetrics(options: {...}): Promise<any>
async transferLearning(options: {...}): Promise<any>
async explainAction(options: {...}): Promise<any>
async recordExperience(options: {...}): Promise<number>
calculateReward(options: {...}): number
```

### MCP Server Integration

**File:** `/workspaces/agentic-flow/packages/agentdb/src/mcp/agentdb-mcp-server.ts`

All 5 tools are fully integrated into the MCP server with:
- Proper parameter validation
- Comprehensive error handling
- Rich formatted output
- User-friendly responses

### Database Schema

Uses existing `learning_experiences`, `learning_sessions`, and `learning_policies` tables. No schema changes required.

---

## Usage Examples

### Example 1: Monitor Learning Performance

```typescript
// Get 7-day metrics grouped by task
{
  "name": "learning_metrics",
  "arguments": {
    "time_window_days": 7,
    "include_trends": true,
    "group_by": "task"
  }
}
```

### Example 2: Transfer Knowledge

```typescript
// Transfer episodes from one session to another
{
  "name": "learning_transfer",
  "arguments": {
    "source_session": "session-123",
    "target_session": "session-456",
    "min_similarity": 0.7,
    "transfer_type": "all",
    "max_transfers": 10
  }
}
```

### Example 3: Get Explainable Recommendations

```typescript
// Get AI recommendations with full explanations
{
  "name": "learning_explain",
  "arguments": {
    "query": "optimize database query performance",
    "k": 5,
    "explain_depth": "detailed",
    "include_evidence": true,
    "include_causal": true
  }
}
```

### Example 4: Record Tool Execution

```typescript
// Log tool execution for learning
{
  "name": "experience_record",
  "arguments": {
    "session_id": "session-123",
    "tool_name": "database_optimizer",
    "action": "add_index",
    "outcome": "query_speed_improved",
    "reward": 0.95,
    "success": true,
    "latency_ms": 150
  }
}
```

### Example 5: Calculate Shaped Reward

```typescript
// Calculate reward with time efficiency
{
  "name": "reward_signal",
  "arguments": {
    "success": true,
    "target_achieved": true,
    "efficiency_score": 0.9,
    "quality_score": 0.85,
    "time_taken_ms": 100,
    "expected_time_ms": 150,
    "reward_function": "shaped",
    "include_causal": true
  }
}
```

---

## Advanced Features

### 1. Transfer Learning
- Semantic similarity matching
- Cross-domain knowledge transfer
- Policy/Q-value transfer
- Experience replay transfer

### 2. Explainable AI (XAI)
- Evidence-based recommendations
- Confidence scoring
- Causal reasoning chains
- Multiple explanation depths

### 3. Reward Shaping
- 4 reward functions (standard/sparse/dense/shaped)
- Time efficiency bonuses
- Multi-objective optimization
- Causal impact integration

### 4. Performance Analytics
- Time-series trend analysis
- Policy improvement tracking
- Success rate monitoring
- Latency analysis

---

## Integration with Existing Tools

These tools complement the existing 5 Learning System tools:
1. `learning_start_session` - Start RL session
2. `learning_end_session` - End RL session
3. `learning_predict` - Get action predictions
4. `learning_feedback` - Submit feedback
5. `learning_train` - Train policy

Together they provide a complete RL system with:
- **Session management** (tools 1-2)
- **Prediction & feedback** (tools 3-4)
- **Training** (tool 5)
- **Metrics & monitoring** (tool 6)
- **Transfer learning** (tool 7)
- **Explainability** (tool 8)
- **Experience logging** (tool 9)
- **Reward calculation** (tool 10)

---

## Server Information

**Version:** 1.4.0
**Total Tools:** 29 tools
- 5 core vector DB tools
- 9 frontier memory tools
- 10 learning system tools
- 5 AgentDB tools

**Startup Message:**
```
🚀 AgentDB MCP Server v1.4.0 running on stdio
📦 29 tools available (5 core vector DB + 9 frontier + 10 learning + 5 AgentDB tools)
🧠 Embedding service initialized
🎓 Learning system ready (9 RL algorithms)
✨ New learning tools: metrics, transfer, explain, experience_record, reward_signal
🔬 Extended features: transfer learning, XAI explanations, reward shaping
```

---

## Testing & Verification

### Build Status
✅ TypeScript compilation successful
✅ All dependencies resolved
✅ MCP server starts without errors

### Coordination Hooks
✅ Pre-task hook executed
✅ Post-edit hooks executed for both files
✅ Notification hooks completed
✅ Session-end hook with metrics
✅ Post-task hook completed

### Memory Storage
✅ Stored in namespace: `agentdb-v1.3.0`
✅ Key: `learning-tools-6-10`
✅ All edits tracked in coordination memory

---

## Performance Characteristics

### Time Complexity
- **learning_metrics**: O(n) where n = episodes in time window
- **learning_transfer**: O(n * m) where n = source episodes, m = similarity checks
- **learning_explain**: O(n * k) where n = experiences, k = embedding comparisons
- **experience_record**: O(1) - single insert
- **reward_signal**: O(1) - constant time calculation

### Space Complexity
- All tools use existing database tables
- No additional storage overhead
- Efficient SQL queries with proper indexing

---

## Future Enhancements

1. **Distributed Transfer Learning**: Transfer across distributed nodes
2. **Neural XAI**: Deep learning-based explanations
3. **Adaptive Reward Shaping**: Self-adjusting reward functions
4. **Real-time Metrics**: Streaming performance dashboards
5. **Multi-agent Transfer**: Knowledge sharing between agents

---

## References

- **ReasoningBank Paper**: Adaptive learning with pattern recognition
- **Transfer Learning**: Cross-domain knowledge reuse
- **Explainable AI**: LIME, SHAP, attention mechanisms
- **Reward Shaping**: Ng et al. (1999) policy invariance
- **Experience Replay**: Lin (1992) reinforcement learning

---

**Implementation Complete:** 2025-10-22
**Author:** Claude Code Agent (Coder Specialist)
**Status:** Production Ready ✅
