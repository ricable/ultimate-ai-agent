# ReasoningBank - Advanced Self-Learning Guide

**Version**: 2.1.0
**Date**: November 10, 2025
**Feature**: Self-learning AI with pattern recognition and adaptive decision making

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Concepts](#core-concepts)
4. [API Reference](#api-reference)
5. [Advanced Examples](#advanced-examples)
6. [Best Practices](#best-practices)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)

---

## Overview

ReasoningBank is an advanced learning system that enables AI agents to:

- **Learn from Experience**: Track trajectories of operations with outcomes
- **Recognize Patterns**: Automatically discover recurring successful sequences
- **Make Predictions**: Suggest optimal approaches based on historical data
- **Improve Over Time**: Adapt strategies through continuous feedback
- **Share Knowledge**: Transfer learnings across agents and sessions

### Key Benefits

✅ **Zero Configuration** - Works out of the box
✅ **Automatic Learning** - No manual pattern definition required
✅ **Transparent Reasoning** - Explains suggestions with supporting evidence
✅ **Failure Tolerance** - Learns from both success and failure
✅ **Memory Efficient** - Intelligent data pruning and compression

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                     JjWrapper                           │
│  ┌────────────────────────────────────────────────┐   │
│  │            Operation Tracking                   │   │
│  │  ┌──────────────────────────────────────────┐ │   │
│  │  │      JJOperation Log (AgentDB)          │ │   │
│  │  └──────────────────────────────────────────┘ │   │
│  └────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────┐   │
│  │            ReasoningBank Engine                 │   │
│  │  ┌──────────────────────────────────────────┐ │   │
│  │  │  Trajectory Storage (1000 max)          │ │   │
│  │  │  ┌────────────┐  ┌────────────┐        │ │   │
│  │  │  │Trajectory 1│  │Trajectory 2│  ...   │ │   │
│  │  │  └────────────┘  └────────────┘        │ │   │
│  │  └──────────────────────────────────────────┘ │   │
│  │  ┌──────────────────────────────────────────┐ │   │
│  │  │  Pattern Extraction & Matching          │ │   │
│  │  │  - Success rate calculation             │ │   │
│  │  │  - Confidence scoring                   │ │   │
│  │  │  - Similarity matching                  │ │   │
│  │  └──────────────────────────────────────────┘ │   │
│  │  ┌──────────────────────────────────────────┐ │   │
│  │  │  Decision Engine                        │ │   │
│  │  │  - Pattern selection                    │ │   │
│  │  │  - Outcome prediction                   │ │   │
│  │  │  - Reasoning generation                 │ │   │
│  │  └──────────────────────────────────────────┘ │   │
│  └────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Operation Execution** → Logged to AgentDB
2. **Trajectory Tracking** → Groups operations by task
3. **Pattern Discovery** → Analyzes successful trajectories
4. **Knowledge Base** → Stores patterns with metadata
5. **Decision Making** → Queries patterns for suggestions

---

## Core Concepts

### 1. Trajectories

A **Trajectory** is a complete record of a task execution:

```typescript
interface Trajectory {
    id: string;                              // Unique identifier
    task: string;                            // "Implement authentication"
    operations: JjOperation[];               // [New, Describe, Commit, ...]
    initialContext: Record<string, string>;  // {branch: "main", ...}
    finalContext: Record<string, string>;    // {branch: "feature/auth", ...}
    successScore: number;                    // 0.9 (0.0-1.0)
    startedAt: string;                       // ISO timestamp
    completedAt: string;                     // ISO timestamp
    tags: string[];                          // ["auth", "feature"]
    reward: number;                          // Calculated reward signal
    critique?: string;                       // "Clean code, good tests"
}
```

**Reward Calculation**:
```
reward = success_score * 0.7 + efficiency_bonus * 0.3

efficiency_bonus = 1 / (1 + ln(duration_minutes))
```

### 2. Patterns

A **Pattern** is an extracted successful sequence:

```typescript
interface Pattern {
    id: string;
    name: string;                           // "Feature development workflow"
    operationSequence: OperationType[];     // [New, Describe, Rebase, Squash]
    successRate: number;                    // 0.87 (average success)
    observationCount: number;               // 12 times observed
    avgDurationMs: number;                  // 3500ms average
    successfulContexts: Record<string, string>[]; // Similar contexts
    confidence: number;                     // 0.85 (based on observations)
}
```

**Confidence Calculation**:
```
confidence = min(ln(observations) / 5, 1.0) * success_rate
```

### 3. Decision Suggestions

A **DecisionSuggestion** provides AI-powered recommendations:

```typescript
interface DecisionSuggestion {
    recommendedOperations: OperationType[]; // [Rebase, Resolve, Squash]
    confidence: number;                     // 0.87
    expectedSuccessRate: number;            // 0.91
    estimatedDurationMs: number;            // 3200
    supportingPatterns: string[];           // ["pattern-uuid-1", ...]
    reasoning: string;                      // "Based on 12 observations..."
}
```

---

## API Reference

### Starting a Trajectory

```javascript
jj.startTrajectory(task: string): string
```

**Purpose**: Begin tracking operations for a specific task.

**Parameters**:
- `task`: Descriptive task name (e.g., "Fix authentication bug")

**Returns**: Trajectory ID (UUID)

**Example**:
```javascript
const trajectoryId = jj.startTrajectory('Implement OAuth2 login');
console.log(`Tracking: ${trajectoryId}`);
```

**Notes**:
- Starting a new trajectory replaces any active trajectory
- Captures initial context (status, branch, etc.)
- Automatically timestamped

---

### Adding Operations

```javascript
jj.addToTrajectory(): void
```

**Purpose**: Add recent operations from AgentDB to active trajectory.

**Example**:
```javascript
// Perform operations
await jj.branchCreate('feature/oauth');
await jj.newCommit('Add OAuth scaffolding');
await jj.describe('Implement OAuth2 client');

// Add to trajectory
jj.addToTrajectory();
```

**Notes**:
- Only adds operations not already in trajectory
- Gets last 10 operations from AgentDB
- Idempotent - safe to call multiple times

---

### Finalizing Trajectories

```javascript
jj.finalizeTrajectory(successScore: number, critique?: string): void
```

**Purpose**: Complete trajectory with outcome and store for learning.

**Parameters**:
- `successScore`: 0.0-1.0 (0.0 = complete failure, 1.0 = perfect success)
- `critique`: Optional self-reflection or lessons learned

**Example**:
```javascript
jj.finalizeTrajectory(0.85, `
    OAuth2 implemented successfully.
    Lessons learned:
    - Token refresh logic was tricky
    - Need better error messages
    - Tests helped catch edge cases
`);
```

**Notes**:
- Calculates reward signal automatically
- Extracts patterns if success_score >= 0.7
- Trajectory becomes available for queries

---

### Getting Suggestions

```javascript
jj.getSuggestion(task: string): string
```

**Purpose**: Get AI decision recommendation for a task.

**Parameters**:
- `task`: Task description to get suggestion for

**Returns**: JSON string of `DecisionSuggestion`

**Example**:
```javascript
const suggestionJson = jj.getSuggestion('Implement logout feature');
const suggestion = JSON.parse(suggestionJson);

console.log(`Confidence: ${(suggestion.confidence * 100).toFixed(1)}%`);
console.log(`Expected success: ${(suggestion.expectedSuccessRate * 100).toFixed(1)}%`);
console.log(`Reasoning: ${suggestion.reasoning}`);

if (suggestion.confidence > 0.7) {
    console.log('High confidence - following suggestion:');
    console.log(suggestion.recommendedOperations);
}
```

**Selection Algorithm**:
```
score = success_rate * confidence
best_pattern = max(patterns, key=score)
```

---

### Learning Statistics

```javascript
jj.getLearningStats(): string
```

**Purpose**: Get comprehensive learning metrics.

**Returns**: JSON string of `LearningStats`

**Example**:
```javascript
const stats = JSON.parse(jj.getLearningStats());

console.log(`Total trajectories: ${stats.totalTrajectories}`);
console.log(`Patterns discovered: ${stats.totalPatterns}`);
console.log(`Average success: ${(stats.avgSuccessRate * 100).toFixed(1)}%`);
console.log(`Improvement rate: ${(stats.improvementRate * 100).toFixed(1)}%`);
console.log(`Predictions made: ${stats.predictionsMade}`);
console.log(`Accuracy: ${(stats.predictionAccuracy * 100).toFixed(1)}%`);
```

---

### Getting Patterns

```javascript
jj.getPatterns(): string
```

**Purpose**: Retrieve all discovered patterns.

**Returns**: JSON string of `Pattern[]`

**Example**:
```javascript
const patterns = JSON.parse(jj.getPatterns());

patterns.forEach((pattern, idx) => {
    console.log(`\nPattern ${idx + 1}: ${pattern.name}`);
    console.log(`  Success rate: ${(pattern.successRate * 100).toFixed(1)}%`);
    console.log(`  Observations: ${pattern.observationCount}`);
    console.log(`  Confidence: ${(pattern.confidence * 100).toFixed(1)}%`);
    console.log(`  Sequence: ${pattern.operationSequence.join(' → ')}`);
    console.log(`  Avg duration: ${pattern.avgDurationMs.toFixed(2)}ms`);
});
```

---

### Querying Trajectories

```javascript
jj.queryTrajectories(task: string, limit: number): string
```

**Purpose**: Find similar past trajectories.

**Parameters**:
- `task`: Task description to match against
- `limit`: Maximum number of results

**Returns**: JSON string of `Trajectory[]` (sorted by similarity)

**Example**:
```javascript
const trajectoriesJson = jj.queryTrajectories('feature implementation', 5);
const trajectories = JSON.parse(trajectoriesJson);

trajectories.forEach((traj, idx) => {
    console.log(`\nMatch ${idx + 1}:`);
    console.log(`  Task: ${traj.task}`);
    console.log(`  Success: ${(traj.successScore * 100).toFixed(0)}%`);
    console.log(`  Operations: ${traj.operations.length}`);
    console.log(`  Duration: ${calcDuration(traj)}s`);

    if (traj.critique) {
        console.log(`  Lessons: ${traj.critique}`);
    }
});
```

**Similarity Algorithm**:
```javascript
// Word overlap similarity
words1 = task1.toLowerCase().split(' ');
words2 = task2.toLowerCase().split(' ');
matches = intersection(words1, words2).length;
similarity = matches / max(words1.length, words2.length);
```

---

### Resetting Learning

```javascript
jj.resetLearning(): void
```

**Purpose**: Clear all trajectories and patterns.

**Example**:
```javascript
// Start fresh
jj.resetLearning();

const stats = JSON.parse(jj.getLearningStats());
console.log(stats.totalTrajectories); // 0
console.log(stats.totalPatterns);     // 0
```

**Use Cases**:
- Testing and development
- Switching to different repository
- Removing bad learning data

---

## Advanced Examples

### Example 1: Continuous Learning Loop

```javascript
const jj = new JjWrapper();

async function developFeature(featureName) {
    // Start trajectory
    const trajId = jj.startTrajectory(`Develop ${featureName}`);

    // Get suggestion from past experiences
    const suggestion = JSON.parse(jj.getSuggestion(`Develop ${featureName}`));

    if (suggestion.confidence > 0.6) {
        console.log(`💡 Suggestion (${(suggestion.confidence * 100).toFixed(0)}% confidence):`);
        console.log(suggestion.reasoning);
    }

    try {
        // Perform development operations
        await jj.branchCreate(`feature/${featureName}`);
        await jj.newCommit(`Start ${featureName}`);
        await jj.describe(`Implement ${featureName}`);
        await jj.execute(['git', 'push']);

        // Add to trajectory
        jj.addToTrajectory();

        // Run tests
        const testsPass = await runTests();

        // Finalize with result
        jj.finalizeTrajectory(
            testsPass ? 0.9 : 0.6,
            testsPass
                ? 'Feature complete, all tests passing'
                : 'Feature works but tests need improvement'
        );

        return true;
    } catch (error) {
        jj.addToTrajectory();
        jj.finalizeTrajectory(0.3, `Failed: ${error.message}`);
        return false;
    }
}

// Develop multiple features, learning from each
const features = ['user-auth', 'user-profile', 'user-settings'];

for (const feature of features) {
    await developFeature(feature);
}

// Check learning progress
const stats = JSON.parse(jj.getLearningStats());
console.log(`\nLearning Progress:`);
console.log(`  Trajectories: ${stats.totalTrajectories}`);
console.log(`  Patterns: ${stats.totalPatterns}`);
console.log(`  Success rate: ${(stats.avgSuccessRate * 100).toFixed(1)}%`);
```

### Example 2: Multi-Agent Collaborative Learning

```javascript
// Central knowledge base
class KnowledgeBase {
    constructor() {
        this.jj = new JjWrapper();
    }

    recordExperience(agent, task, operations, success, critique) {
        this.jj.startTrajectory(`${agent}: ${task}`);

        // Simulate operations
        operations.forEach(op => {
            // Add operation to log (simplified)
        });

        this.jj.addToTrajectory();
        this.jj.finalizeTrajectory(success, critique);
    }

    getSuggestionFor(agent, task) {
        // Query similar tasks from all agents
        const trajectories = JSON.parse(
            this.jj.queryTrajectories(task, 10)
        );

        // Get suggestion
        const suggestion = JSON.parse(this.jj.getSuggestion(task));

        // Combine with similar experiences
        return {
            suggestion,
            similar_experiences: trajectories.map(t => ({
                agent: t.task.split(':')[0],
                success: t.successScore,
                lessons: t.critique,
            })),
        };
    }

    getStats() {
        return JSON.parse(this.jj.getLearningStats());
    }
}

// Agents using shared knowledge
const kb = new KnowledgeBase();

// Agent 1: Developer
async function developerAgent(task) {
    const guidance = kb.getSuggestionFor('developer', task);

    console.log('Checking knowledge base...');
    if (guidance.similar_experiences.length > 0) {
        console.log('Found similar experiences:');
        guidance.similar_experiences.forEach(exp => {
            console.log(`  - ${exp.agent}: ${(exp.success * 100).toFixed(0)}%`);
        });
    }

    // Perform work...
    const success = await performDevelopment(task);

    // Share learning
    kb.recordExperience('developer', task, [], success, 'Completed successfully');
}

// Agent 2: Reviewer
async function reviewerAgent(prNumber) {
    const guidance = kb.getSuggestionFor('reviewer', `Review PR ${prNumber}`);

    // Use suggestions to guide review...
    const success = await performReview(prNumber, guidance);

    kb.recordExperience('reviewer', `Review PR ${prNumber}`, [], success, 'Review complete');
}

// Agents learn from each other
await developerAgent('Implement feature X');
await reviewerAgent(123);
await developerAgent('Implement feature Y'); // Benefits from both experiences

console.log('\nCollective Learning Stats:');
console.log(kb.getStats());
```

### Example 3: Adaptive Retry with Learning

```javascript
async function retryWithLearning(task, maxAttempts = 3) {
    const jj = new JjWrapper();

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        console.log(`\n=== Attempt ${attempt}/${maxAttempts} ===`);

        // Start trajectory
        jj.startTrajectory(`${task} - attempt ${attempt}`);

        // Get suggestion based on previous attempts
        const suggestion = JSON.parse(jj.getSuggestion(task));

        if (attempt > 1) {
            console.log(`Learning from previous attempts:`);
            console.log(`  Confidence: ${(suggestion.confidence * 100).toFixed(1)}%`);
            console.log(`  Expected success: ${(suggestion.expectedSuccessRate * 100).toFixed(1)}%`);
            console.log(`  ${suggestion.reasoning}`);
        }

        try {
            // Perform task with adapted strategy
            const result = await performTaskWithStrategy(
                task,
                suggestion.recommendedOperations
            );

            jj.addToTrajectory();
            jj.finalizeTrajectory(0.9, `Success on attempt ${attempt}`);

            console.log(`✅ Task succeeded!`);
            return result;
        } catch (error) {
            jj.addToTrajectory();
            jj.finalizeTrajectory(
                0.3,
                `Attempt ${attempt} failed: ${error.message}. ${
                    attempt < maxAttempts ? 'Will adapt strategy.' : 'Giving up.'
                }`
            );

            if (attempt === maxAttempts) {
                throw new Error(`Task failed after ${maxAttempts} attempts`);
            }

            // Learn from failure before retrying
            console.log(`❌ Failed: ${error.message}`);
            console.log(`Analyzing failure...`);

            // Query similar failures
            const failures = JSON.parse(jj.queryTrajectories(task, 5))
                .filter(t => t.successScore < 0.5);

            if (failures.length > 0) {
                console.log(`Found ${failures.length} similar failure(s):`);
                failures.forEach(f => {
                    console.log(`  - ${f.critique}`);
                });
            }
        }
    }
}

// Use adaptive retry
await retryWithLearning('Deploy to production', 3);
```

---

## Best Practices

### 1. Task Naming

**❌ Bad**:
```javascript
jj.startTrajectory('task1');
jj.startTrajectory('work');
jj.startTrajectory('fix');
```

**✅ Good**:
```javascript
jj.startTrajectory('Implement OAuth2 authentication');
jj.startTrajectory('Fix JWT token expiration bug');
jj.startTrajectory('Refactor user service for better testability');
```

**Why**: Descriptive names enable better pattern matching and knowledge retrieval.

### 2. Success Scoring

**❌ Bad** (Binary thinking):
```javascript
jj.finalizeTrajectory(1.0); // Always perfect
jj.finalizeTrajectory(0.0); // Always failure
```

**✅ Good** (Nuanced assessment):
```javascript
// Feature works perfectly, clean code, good tests
jj.finalizeTrajectory(0.95);

// Feature works, but needs refactoring
jj.finalizeTrajectory(0.75);

// Feature works, but has issues
jj.finalizeTrajectory(0.6);

// Feature partially works, major issues
jj.finalizeTrajectory(0.4);

// Complete failure
jj.finalizeTrajectory(0.1);
```

**Why**: Nuanced scores help identify patterns in varying success levels.

### 3. Critique Quality

**❌ Bad**:
```javascript
jj.finalizeTrajectory(0.8, 'Good');
jj.finalizeTrajectory(0.6, 'Had issues');
```

**✅ Good**:
```javascript
jj.finalizeTrajectory(0.8, `
    Successful implementation:
    ✓ Clean code structure
    ✓ All tests passing
    ✓ Good performance

    Areas for improvement:
    - Error messages could be clearer
    - Need more edge case tests
    - Documentation incomplete

    Next time:
    - Write docs during development
    - Consider error cases earlier
`);
```

**Why**: Detailed critiques provide valuable context for future decisions.

### 4. Pattern Discovery Threshold

Adjust success threshold based on domain:

```javascript
// High-stakes tasks (production deployments)
const highStakes = new JjWrapper();
highStakes.min_success_threshold = 0.85;

// Regular development tasks
const regular = new JjWrapper();
regular.min_success_threshold = 0.7;

// Experimental/learning tasks
const experimental = new JjWrapper();
experimental.min_success_threshold = 0.5;
```

### 5. Confidence Thresholds

Use appropriate confidence thresholds:

```javascript
const suggestion = JSON.parse(jj.getSuggestion(task));

if (suggestion.confidence > 0.9) {
    // Very high confidence - auto-apply
    await applyAutomatically(suggestion.recommendedOperations);
} else if (suggestion.confidence > 0.7) {
    // High confidence - apply with review
    console.log('Recommendation:', suggestion.reasoning);
    await applyWithReview(suggestion.recommendedOperations);
} else if (suggestion.confidence > 0.5) {
    // Medium confidence - suggest only
    console.log('Consider:', suggestion.reasoning);
} else {
    // Low confidence - gather more data
    console.log('Insufficient data for recommendation');
}
```

---

## Performance Tuning

### Memory Management

```javascript
// Default: 1000 trajectories
// Adjust based on available memory:

// Low memory environment (embedded systems)
config.max_trajectories = 100;

// Standard environment
config.max_trajectories = 1000;

// High memory environment (server)
config.max_trajectories = 10000;
```

**Memory Usage**:
```
memory_mb = (trajectories * 2KB + patterns * 1KB) / 1024
```

### Pattern Extraction Performance

```javascript
// Patterns extracted when success_score >= threshold
// Lower threshold = more patterns, more CPU
// Higher threshold = fewer patterns, less CPU

// CPU-constrained
config.min_success_threshold = 0.85;

// Balanced
config.min_success_threshold = 0.7;

// Learn from everything
config.min_success_threshold = 0.5;
```

### Query Optimization

```javascript
// Limit results for faster queries
const trajectories = JSON.parse(
    jj.queryTrajectories(task, 5) // Only top 5
);

// vs slower but more comprehensive
const trajectories = JSON.parse(
    jj.queryTrajectories(task, 50) // All matches
);
```

---

## Troubleshooting

### Problem: No patterns discovered

**Symptoms**:
```javascript
const patterns = JSON.parse(jj.getPatterns());
console.log(patterns.length); // 0
```

**Solutions**:
1. Check trajectory count: Need at least 2-3 trajectories
2. Check success scores: Must be >= 0.7 by default
3. Check similarity: Patterns need similar operation sequences

```javascript
// Debug
const stats = JSON.parse(jj.getLearningStats());
console.log(`Trajectories: ${stats.totalTrajectories}`);

const trajectories = JSON.parse(jj.queryTrajectories('', 100));
const successful = trajectories.filter(t => t.successScore >= 0.7);
console.log(`Successful: ${successful.length}`);
```

### Problem: Low confidence suggestions

**Symptoms**:
```javascript
const suggestion = JSON.parse(jj.getSuggestion(task));
console.log(suggestion.confidence); // < 0.5
```

**Solutions**:
1. **Not enough observations**: Need more trajectories
2. **Inconsistent results**: Success rates varying
3. **Different contexts**: Tasks too dissimilar

```javascript
// Check pattern quality
const patterns = JSON.parse(jj.getPatterns());
patterns.forEach(p => {
    console.log(`${p.name}:`);
    console.log(`  Observations: ${p.observationCount}`);
    console.log(`  Success rate: ${p.successRate}`);
    console.log(`  Confidence: ${p.confidence}`);
});
```

### Problem: Memory usage too high

**Symptoms**:
- Process memory growing over time
- Slow performance

**Solutions**:

```javascript
// 1. Reduce trajectory limit
config.max_trajectories = 500;

// 2. Periodic cleanup
setInterval(() => {
    const stats = JSON.parse(jj.getLearningStats());
    if (stats.totalTrajectories > 800) {
        // Keep only high-quality trajectories
        // (requires custom implementation)
    }
}, 3600000); // Every hour

// 3. Reset when switching contexts
if (switchingToNewRepo) {
    jj.resetLearning();
}
```

### Problem: Suggestions not relevant

**Symptoms**:
- Recommended operations don't match task
- Low expected success rates

**Solutions**:

```javascript
// 1. Check task description similarity
const trajectories = JSON.parse(jj.queryTrajectories(task, 10));
trajectories.forEach(t => {
    console.log(`Task: ${t.task}`);
    console.log(`Similarity: ${calculateSimilarity(task, t.task)}`);
});

// 2. Use more specific task descriptions
// Bad: "Fix bug"
// Good: "Fix authentication token expiration bug in user service"

// 3. Query specific patterns
const patterns = JSON.parse(jj.getPatterns());
const relevant = patterns.filter(p =>
    p.name.toLowerCase().includes(keywordFromTask)
);
```

---

## Conclusion

ReasoningBank provides powerful self-learning capabilities that enable AI agents to continuously improve from experience. By tracking trajectories, discovering patterns, and providing intelligent suggestions, it creates a feedback loop that drives adaptive behavior and knowledge accumulation.

### Key Takeaways

1. **Start Simple**: Begin with basic trajectory tracking, let patterns emerge naturally
2. **Quality Over Quantity**: Better critiques and accurate scoring beats more trajectories
3. **Learn from Failures**: Failed attempts are valuable learning opportunities
4. **Use Confidence**: Trust high-confidence suggestions, investigate low ones
5. **Monitor Progress**: Check learning stats to verify improvement over time

### Next Steps

- Experiment with trajectory tracking in your workflows
- Tune confidence thresholds for your use case
- Build multi-agent systems that share knowledge
- Integrate with CI/CD for continuous learning
- Explore persistence options for long-term memory

---

**Documentation**: `docs/REASONING_BANK_GUIDE.md`
**Tests**: `tests/reasoning-bank.test.js`
**Examples**: See README.md § ReasoningBank

For questions or issues: https://github.com/ruvnet/agentic-flow/issues
