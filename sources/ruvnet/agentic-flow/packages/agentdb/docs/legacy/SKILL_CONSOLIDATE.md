# AgentDB Skill Consolidate Tool

## Overview

The `skill_consolidate` tool is an advanced AgentDB feature that automatically creates reusable skills from successful reflexion episodes using ML-inspired pattern extraction.

## Features

### 1. **Automatic Skill Generation**
- Analyzes reflexion episodes with high success rates
- Groups similar successful episodes by task
- Automatically creates skill definitions with learned patterns

### 2. **ML Pattern Extraction**
Five intelligent pattern analysis techniques:

#### a) **Keyword Frequency Analysis**
- Extracts common techniques from episode outputs
- Uses NLP-inspired stop word filtering
- Identifies frequently used keywords (appearing 2+ times)
- Example: `Common techniques: authentication, validation, error_handling`

#### b) **Critique Pattern Analysis**
- Analyzes successful episode critiques
- Extracts key success indicators
- Filters out noise with frequency thresholds

#### c) **Reward Distribution Analysis**
- Analyzes consistency of high rewards
- Calculates high-reward ratio
- Identifies stable performance patterns
- Example: `High consistency (75% above average)`

#### d) **Metadata Pattern Extraction**
- Identifies consistent configuration parameters
- Extracts common metadata fields across episodes
- Example: `Consistent environment: production`

#### e) **Learning Curve Analysis**
- Temporal analysis of episode performance
- Compares first half vs second half rewards
- Identifies improvement trends
- Examples:
  - `Strong learning curve (+25% improvement)`
  - `Stable performance (±3%)`

### 3. **Pattern Confidence Scoring**
- Calculates confidence based on sample size and success rate
- Uses sigmoid-like function for smooth scaling
- Saturates at 10 samples for stability
- Formula: `confidence = min(sampleSize/10, 1.0) * successRate`

## Usage

### CLI Command

```bash
agentdb skill consolidate [min-attempts] [min-reward] [time-window-days] [extract-patterns]
```

**Parameters:**
- `min-attempts` (default: 3) - Minimum number of successful episodes required
- `min-reward` (default: 0.7) - Minimum reward threshold (0.0-1.0)
- `time-window-days` (default: 7) - Time window to analyze in days
- `extract-patterns` (default: true) - Enable/disable pattern extraction

### Examples

```bash
# Basic usage with defaults (3 attempts, 0.7 reward, 7 days, patterns enabled)
agentdb skill consolidate

# Higher quality threshold
agentdb skill consolidate 5 0.8 14 true

# Disable pattern extraction for faster processing
agentdb skill consolidate 3 0.7 7 false

# Lower threshold to capture more skills
agentdb skill consolidate 2 0.6 30 true
```

### Programmatic Usage

```typescript
import { SkillLibrary } from 'agentdb';

const result = await skillLibrary.consolidateEpisodesIntoSkills({
  minAttempts: 3,
  minReward: 0.7,
  timeWindowDays: 7,
  extractPatterns: true
});

console.log(`Created: ${result.created} skills`);
console.log(`Updated: ${result.updated} skills`);

// Analyze extracted patterns
result.patterns.forEach(pattern => {
  console.log(`Task: ${pattern.task}`);
  console.log(`Patterns: ${pattern.commonPatterns.join(', ')}`);
  console.log(`Indicators: ${pattern.successIndicators.join(', ')}`);
});
```

## Output Format

### Console Output Example

```
🔄 Consolidating Episodes into Skills with Pattern Extraction
ℹ Min Attempts: 3
ℹ Min Reward: 0.7
ℹ Time Window: 7 days
ℹ Pattern Extraction: Enabled

✅ Created 5 new skills, updated 2 existing skills in 234ms

════════════════════════════════════════════════════════════════════════════════
Extracted Patterns:
════════════════════════════════════════════════════════════════════════════════

#1: implement_authentication
  Avg Reward: 0.87
  Common Patterns:
    • Common techniques: oauth2, jwt, validation, session
    • Consistent framework: express
  Success Indicators:
    • error_handling
    • security
    • High consistency (83% above average)
    • Strong learning curve (+18% improvement)
────────────────────────────────────────────────────────────────────────────────

#2: database_migration
  Avg Reward: 0.92
  Common Patterns:
    • Common techniques: schema, transaction, rollback
    • Consistent database: postgresql
  Success Indicators:
    • backup
    • testing
    • Stable performance (±2%)
────────────────────────────────────────────────────────────────────────────────
```

### Return Value Structure

```typescript
{
  created: number,        // Number of new skills created
  updated: number,        // Number of existing skills updated
  patterns: Array<{
    task: string,
    commonPatterns: string[],
    successIndicators: string[],
    avgReward: number
  }>
}
```

## Skill Metadata

Generated skills include rich metadata:

```typescript
{
  name: "implement_authentication",
  description: "Skill learned from 5 successful episodes. Common patterns: oauth2, jwt, validation",
  signature: {
    inputs: { task: 'string' },
    outputs: { result: 'any' }
  },
  successRate: 0.87,
  uses: 5,
  avgReward: 0.87,
  avgLatencyMs: 1250,
  createdFromEpisode: 42,
  metadata: {
    sourceEpisodes: [38, 39, 40, 41, 42],
    autoGenerated: true,
    consolidatedAt: 1729606015000,
    extractedPatterns: [
      "Common techniques: oauth2, jwt, validation, session",
      "Consistent framework: express"
    ],
    successIndicators: [
      "error_handling",
      "security",
      "High consistency (83% above average)",
      "Strong learning curve (+18% improvement)"
    ],
    patternConfidence: 0.435  // Based on 5 samples with 0.87 success rate
  }
}
```

## Integration with ReflexionMemory

### Workflow

1. **Store Episodes** - Use `reflexion_store` to record episodes with critiques
2. **Accumulate Data** - Let episodes accumulate over time (7+ days recommended)
3. **Consolidate** - Run `skill_consolidate` to extract learned patterns
4. **Retrieve Skills** - Use `skill_search` to find relevant skills for new tasks

### Example Complete Workflow

```bash
# 1. Store successful episodes over time
agentdb reflexion store "session-1" "implement_auth" 0.95 true "Used OAuth2 with JWT"
agentdb reflexion store "session-2" "implement_auth" 0.85 true "Added refresh tokens"
agentdb reflexion store "session-3" "implement_auth" 0.90 true "Implemented rate limiting"

# 2. Wait for more episodes to accumulate...

# 3. Consolidate into skills
agentdb skill consolidate 3 0.8 7

# 4. Search for relevant skills later
agentdb skill search "authentication" 5
```

## Performance Considerations

### Speed
- Pattern extraction adds ~50-100ms per skill
- Scales linearly with number of episodes analyzed
- Typical performance: 200-500ms for 5-10 skills

### Memory
- Keyword frequency analysis uses in-memory maps
- Scales well up to thousands of episodes
- Automatic pruning available via `skill_prune`

### Optimization Tips
1. **Disable patterns for speed**: Set `extract-patterns=false` for 2-3x faster processing
2. **Narrow time window**: Use shorter windows (7 days) for focused learning
3. **Higher thresholds**: Increase `min-reward` to reduce candidate set
4. **Regular pruning**: Run `skill_prune` monthly to remove stale skills

## Best Practices

### 1. Episode Quality
- Store detailed critiques and outputs for better pattern extraction
- Include metadata for consistent parameter detection
- Use structured output formats when possible

### 2. Consolidation Frequency
- Run weekly for active learning systems
- Run monthly for stable production systems
- Run after major feature additions

### 3. Threshold Tuning
- Start conservative (min-attempts=5, min-reward=0.8)
- Lower thresholds as confidence builds
- Monitor skill usage with `skill_search`

### 4. Pattern Interpretation
- Use extracted patterns as hints, not rules
- Validate patterns with domain knowledge
- Combine with human expertise for critical systems

## Troubleshooting

### No Skills Created

**Symptoms**: `Created 0 new skills`

**Solutions**:
1. Lower `min-reward` threshold (try 0.6)
2. Increase `time-window-days` (try 14 or 30)
3. Check episode data: `agentdb reflexion retrieve <task>`
4. Verify success rate: `agentdb db stats`

### Poor Pattern Quality

**Symptoms**: Generic or unhelpful patterns

**Solutions**:
1. Store more detailed episode outputs
2. Include structured critique messages
3. Increase `min-attempts` for better statistics
4. Add meaningful metadata to episodes

### Duplicate Skills

**Symptoms**: Multiple skills for same task

**Solutions**:
1. Skills are deduplicated by name automatically
2. Use consistent task naming in episodes
3. Run `skill_prune` to remove low-usage duplicates

## Technical Details

### Pattern Extraction Algorithm

1. **Text Processing**
   - Tokenization with regex: `\b[a-z0-9_-]+\b`
   - Stop word filtering (35 common words)
   - Minimum word length: 4 characters
   - Minimum frequency: 2 occurrences

2. **Statistical Analysis**
   - Frequency counting with Map data structure
   - Top-k selection (5 for patterns, 3 for critiques)
   - Reward distribution analysis with percentiles

3. **Temporal Analysis**
   - Episode sorting by ID (temporal order)
   - Split-half comparison (first 50% vs last 50%)
   - Percentage improvement calculation

### Database Schema Integration

Skills are stored in the `skills` table with:
- Vector embeddings for similarity search
- JSON metadata with extracted patterns
- Usage statistics (success_rate, uses, avg_reward)
- Temporal tracking (created_at, last_used_at)

## See Also

- [Reflexion Memory](./REFLEXION_MEMORY.md) - Episode storage and retrieval
- [Skill Library](./SKILL_LIBRARY.md) - Skill management and search
- [CLI Guide](./CLI_GUIDE.md) - Complete CLI reference
- [AgentDB API](./API.md) - Programmatic usage

## Research References

Based on concepts from:
- **Voyager** (Wang et al., 2023) - Lifelong learning skill library
- **Reflexion** (Shinn et al., 2023) - Self-improvement through critique
- Research paper: https://arxiv.org/abs/2305.16291
