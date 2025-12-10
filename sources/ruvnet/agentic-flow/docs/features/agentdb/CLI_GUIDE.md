# AgentDB CLI Guide - Frontier Features

Complete command-line interface for AgentDB's state-of-the-art memory features.

## Installation

```bash
npm install -g agentic-flow@latest
# or
npx agentic-flow agentdb <command>
```

## Quick Start

```bash
# Set database path (optional, defaults to ./agentdb.db)
export AGENTDB_PATH="./my-agent-memory.db"

# Add a causal edge
npx agentdb causal add-edge "add_tests" "code_quality" 0.25

# Run nightly learner to discover patterns
npx agentdb learner run

# Query with causal utility
npx agentdb recall with-certificate "implement authentication"
```

---

## Causal Commands

### `causal add-edge`

Manually add a causal relationship between two variables.

**Syntax:**
```bash
agentdb causal add-edge <cause> <effect> <uplift> [confidence] [sample-size]
```

**Parameters:**
- `cause` - Cause variable (string)
- `effect` - Effect variable (string)
- `uplift` - Causal effect size (number, e.g., 0.25 = 25% improvement)
- `confidence` - Statistical confidence (optional, default: 0.95)
- `sample-size` - Number of observations (optional, default: 0)

**Example:**
```bash
# Adding tests improves code quality by 25% with 95% confidence
agentdb causal add-edge "add_tests" "code_quality" 0.25 0.95 100
```

---

### `causal experiment create`

Create a new A/B experiment to test a hypothesis.

**Syntax:**
```bash
agentdb causal experiment create <name> <cause> <effect>
```

**Parameters:**
- `name` - Experiment name (string)
- `cause` - Treatment variable (string)
- `effect` - Outcome variable (string)

**Example:**
```bash
# Test if code reviews reduce bug rate
agentdb causal experiment create "code-review-bugs" "code_review" "bug_rate"
```

---

### `causal experiment add-observation`

Record an observation in an A/B experiment.

**Syntax:**
```bash
agentdb causal experiment add-observation <experiment-id> <is-treatment> <outcome> [context]
```

**Parameters:**
- `experiment-id` - Experiment ID (number)
- `is-treatment` - Treatment group flag (true/false)
- `outcome` - Observed outcome value (number)
- `context` - Additional context JSON (optional, string)

**Example:**
```bash
# Treatment group: code review done, outcome: 0.12 bugs/KLOC
agentdb causal experiment add-observation 1 true 0.12 '{"reviewer": "senior"}'

# Control group: no code review, outcome: 0.35 bugs/KLOC
agentdb causal experiment add-observation 1 false 0.35 '{"reviewer": "none"}'
```

---

### `causal experiment calculate`

Calculate uplift and statistical significance for an experiment.

**Syntax:**
```bash
agentdb causal experiment calculate <experiment-id>
```

**Parameters:**
- `experiment-id` - Experiment ID (number)

**Example:**
```bash
agentdb causal experiment calculate 1
```

**Output:**
```
📈 Calculating Uplift

ℹ Treatment Mean: 0.120
ℹ Control Mean: 0.350
✅ Uplift: -0.230 (65.7% reduction)
ℹ 95% CI: [-0.280, -0.180]
ℹ p-value: 0.0001
ℹ Sample Sizes: 50 treatment, 50 control
✅ Result is statistically significant (p < 0.05)
```

---

### `causal query`

Query causal edges with filters.

**Syntax:**
```bash
agentdb causal query [cause] [effect] [min-confidence] [min-uplift] [limit]
```

**Parameters:**
- `cause` - Filter by cause (optional, string)
- `effect` - Filter by effect (optional, string)
- `min-confidence` - Minimum confidence (optional, default: 0.7)
- `min-uplift` - Minimum uplift magnitude (optional, default: 0.1)
- `limit` - Max results (optional, default: 10)

**Examples:**
```bash
# All high-confidence edges
agentdb causal query "" "" 0.8

# Edges affecting code quality
agentdb causal query "" "code_quality" 0.7

# Effects of adding tests
agentdb causal query "add_tests" "" 0.7 0.1 5
```

---

## Recall Commands

### `recall with-certificate`

Retrieve episodes with causal utility and provenance certificate.

**Syntax:**
```bash
agentdb recall with-certificate <query> [k] [alpha] [beta] [gamma]
```

**Parameters:**
- `query` - Search query (string)
- `k` - Number of results (optional, default: 12)
- `alpha` - Similarity weight (optional, default: 0.7)
- `beta` - Uplift weight (optional, default: 0.2)
- `gamma` - Latency penalty (optional, default: 0.1)

**Utility Formula:**
```
U = α * similarity + β * uplift - γ * latency_cost
```

**Examples:**
```bash
# Default weights: 70% similarity, 20% causal uplift, 10% latency penalty
agentdb recall with-certificate "implement authentication" 10

# Prioritize causal impact over similarity
agentdb recall with-certificate "fix bug" 12 0.5 0.4 0.1

# Pure similarity (no causal ranking)
agentdb recall with-certificate "refactor code" 10 1.0 0.0 0.0
```

**Output:**
```
🔍 Causal Recall with Certificate
ℹ Query: "implement authentication"
ℹ k: 10

================================================================================
Results (10)
================================================================================

#1: Episode 142
  Task: implement_oauth2_authentication
  Similarity: 0.892
  Uplift: 0.156
  Utility: 0.735
  Reward: 0.88

#2: Episode 89
  Task: add_jwt_tokens
  Similarity: 0.845
  Uplift: 0.203
  Utility: 0.733
  Reward: 0.91

...

================================================================================
ℹ Certificate ID: cert-1729512345-abc123
ℹ Provenance Hash: 2f7a8e1d3c4b9...
ℹ Retrieved at: 2025-10-21T14:25:45.123Z
✅ Completed in 75ms
```

---

## Learner Commands

### `learner run`

Discover causal edges from episode patterns using doubly robust estimator.

**Syntax:**
```bash
agentdb learner run [min-attempts] [min-success-rate] [min-confidence] [dry-run]
```

**Parameters:**
- `min-attempts` - Minimum episode attempts per pattern (optional, default: 3)
- `min-success-rate` - Minimum success rate (optional, default: 0.6)
- `min-confidence` - Minimum statistical confidence (optional, default: 0.7)
- `dry-run` - Preview without saving (optional, "true"/"false", default: false)

**Algorithm:**
Uses doubly robust causal estimator:
```
τ̂(x) = μ₁(x) - μ₀(x) + [a*(y-μ₁(x))/e(x)] - [(1-a)*(y-μ₀(x))/(1-e(x))]
```

**Examples:**
```bash
# Default: conservative discovery
agentdb learner run

# Aggressive discovery (lower thresholds)
agentdb learner run 2 0.5 0.6

# Preview mode (don't save edges)
agentdb learner run 3 0.6 0.7 true

# High-confidence only
agentdb learner run 5 0.7 0.9
```

**Output:**
```
🌙 Running Nightly Learner
ℹ Min Attempts: 3
ℹ Min Success Rate: 0.6
ℹ Min Confidence: 0.7
✅ Discovered 12 causal edges in 8.3s

================================================================================
#1: add_error_handling → code_stability
  Uplift: 0.234 (CI: 0.88)
  Sample size: 47
─────────────────────────────────────────────────────────────────────────────
#2: write_tests → bug_detection_early
  Uplift: 0.189 (CI: 0.82)
  Sample size: 35
...
```

---

### `learner prune`

Remove low-quality or stale causal edges.

**Syntax:**
```bash
agentdb learner prune [min-confidence] [min-uplift] [max-age-days]
```

**Parameters:**
- `min-confidence` - Minimum confidence to keep (optional, default: 0.5)
- `min-uplift` - Minimum uplift magnitude (optional, default: 0.05)
- `max-age-days` - Maximum age in days (optional, default: 90)

**Examples:**
```bash
# Conservative pruning (keep most edges)
agentdb learner prune

# Aggressive pruning
agentdb learner prune 0.7 0.1 30

# Remove old edges only
agentdb learner prune 0.0 0.0 60
```

---

## Reflexion Commands

### `reflexion store`

Store an episode with self-critique for agent self-improvement.

**Syntax:**
```bash
agentdb reflexion store <session-id> <task> <reward> <success> [critique] [input] [output] [latency-ms] [tokens]
```

**Parameters:**
- `session-id` - Session identifier (string)
- `task` - Task description (string)
- `reward` - Reward score 0-1 (number)
- `success` - Task success flag (true/false)
- `critique` - Self-critique or lesson learned (optional, string)
- `input` - Task input data (optional, string)
- `output` - Task output data (optional, string)
- `latency-ms` - Execution time in ms (optional, number)
- `tokens` - Tokens consumed (optional, number)

**Example:**
```bash
# Store successful authentication implementation
agentdb reflexion store "session-1" "implement_oauth2" 0.95 true \
  "Used industry-standard OAuth2 flow with refresh tokens" \
  "Requirements: secure authentication" \
  "JWT-based OAuth2 implementation"

# Store failed attempt with critique
agentdb reflexion store "session-2" "fix_memory_leak" 0.3 false \
  "Forgot to cleanup event listeners - must always call removeEventListener()" \
  "Memory growing unbounded" \
  "Attempted fix with WeakMap but missed cleanup"
```

---

### `reflexion retrieve`

Retrieve relevant past episodes for learning from experience.

**Syntax:**
```bash
agentdb reflexion retrieve <task> [k] [min-reward] [only-failures] [only-successes]
```

**Parameters:**
- `task` - Query task (string)
- `k` - Number of episodes to retrieve (optional, default: 10)
- `min-reward` - Minimum reward threshold (optional, default: 0.0)
- `only-failures` - Return only failures (optional, true/false)
- `only-successes` - Return only successes (optional, true/false)

**Examples:**
```bash
# Get top 10 authentication episodes
agentdb reflexion retrieve "implement authentication" 10

# Get only successful episodes with high reward
agentdb reflexion retrieve "bug fix" 20 0.8 false true

# Get only failures to learn from mistakes
agentdb reflexion retrieve "performance optimization" 15 0.0 true false
```

**Output:**
```
🔍 Retrieving Episodes

ℹ Query: "implement authentication"
ℹ k: 10

================================================================================
Results (3)
================================================================================

#1: Episode 142
  Task: implement_oauth2_authentication
  Similarity: 0.892
  Reward: 0.95
  Success: Yes
  Critique: "Used industry-standard OAuth2 flow with refresh tokens"

#2: Episode 89
  Task: add_jwt_tokens
  Similarity: 0.845
  Reward: 0.91
  Success: Yes
  Critique: "Implemented secure token rotation"

...
```

---

### `reflexion critique-summary`

Get aggregated critique summary from past episodes.

**Syntax:**
```bash
agentdb reflexion critique-summary <task> [only-failures]
```

**Parameters:**
- `task` - Task pattern (string)
- `only-failures` - Focus on failure lessons (optional, true/false)

**Example:**
```bash
# Get all lessons about authentication
agentdb reflexion critique-summary "authentication"

# Get failure lessons about bug fixes
agentdb reflexion critique-summary "bug_fix" true
```

**Output:**
```
📝 Critique Summary

ℹ Task: "authentication"
ℹ Episodes analyzed: 25

================================================================================
Key Lessons
================================================================================

• Always use refresh tokens for long-lived sessions (15 episodes)
• CSRF protection required for OAuth2 flows (12 episodes)
• Never store tokens in localStorage, use httpOnly cookies (10 episodes)
• Rate-limit authentication attempts to prevent brute force (8 episodes)

✅ Success rate: 88.0%
✅ Avg reward: 0.87
```

---

### `reflexion prune`

Clean up old or low-value episodes.

**Syntax:**
```bash
agentdb reflexion prune [max-age-days] [max-reward]
```

**Parameters:**
- `max-age-days` - Remove episodes older than X days (optional, default: 90)
- `max-reward` - Remove episodes with reward below X (optional, default: 0.3)

**Examples:**
```bash
# Conservative: remove old episodes only
agentdb reflexion prune 180

# Aggressive: remove low-value episodes
agentdb reflexion prune 30 0.5

# Default: 90 days, reward < 0.3
agentdb reflexion prune
```

---

## Skill Commands

### `skill create`

Create a reusable skill from experience.

**Syntax:**
```bash
agentdb skill create <name> <description> [code]
```

**Parameters:**
- `name` - Skill name (string)
- `description` - What this skill does (string)
- `code` - Implementation code (optional, string)

**Example:**
```bash
# Create authentication skill
agentdb skill create "jwt_authentication" \
  "Generate and validate JWT tokens with refresh rotation" \
  "function generateJWT(payload) { /* code */ }"

# Create skill without code (metadata only)
agentdb skill create "rate_limiting" \
  "Implement token-bucket rate limiting for API endpoints"
```

---

### `skill search`

Find applicable skills by semantic similarity.

**Syntax:**
```bash
agentdb skill search <query> [k]
```

**Parameters:**
- `query` - Search query (string)
- `k` - Number of results (optional, default: 10)

**Example:**
```bash
# Find authentication skills
agentdb skill search "secure user login" 5

# Find performance optimization skills
agentdb skill search "reduce latency" 10
```

**Output:**
```
🔍 Searching Skills

ℹ Query: "secure user login"
ℹ k: 5

================================================================================
Results (5)
================================================================================

#1: jwt_authentication
  Description: Generate and validate JWT tokens with refresh rotation
  Similarity: 0.912
  Success Rate: 95.3%
  Uses: 47
  Avg Reward: 0.91
  Avg Latency: 125ms

#2: oauth2_flow
  Description: Complete OAuth2 authorization code flow
  Similarity: 0.875
  Success Rate: 88.1%
  Uses: 32
  Avg Reward: 0.87
  Avg Latency: 280ms

...
```

---

### `skill consolidate`

Automatically create skills from successful episodes.

**Syntax:**
```bash
agentdb skill consolidate [min-reward] [min-success-rate]
```

**Parameters:**
- `min-reward` - Minimum episode reward (optional, default: 0.8)
- `min-success-rate` - Minimum success rate for patterns (optional, default: 0.7)

**Examples:**
```bash
# Conservative: high-quality skills only
agentdb skill consolidate 0.9 0.8

# Balanced: default thresholds
agentdb skill consolidate

# Aggressive: capture more patterns
agentdb skill consolidate 0.6 0.6
```

**Output:**
```
🎓 Consolidating Skills

ℹ Min Reward: 0.8
ℹ Min Success Rate: 0.7
✅ Created 7 new skills from successful episodes

================================================================================
#1: error_handling_pattern
  Episodes: 15
  Success rate: 93.3%
  Avg reward: 0.89
─────────────────────────────────────────────────────────────────────────────
#2: database_transaction
  Episodes: 12
  Success rate: 91.7%
  Avg reward: 0.87
...
```

---

### `skill prune`

Remove underperforming skills.

**Syntax:**
```bash
agentdb skill prune [min-success-rate] [max-latency-ms]
```

**Parameters:**
- `min-success-rate` - Keep skills above this rate (optional, default: 0.5)
- `max-latency-ms` - Remove skills slower than this (optional, default: 5000)

**Examples:**
```bash
# Conservative: remove obvious failures only
agentdb skill prune 0.3 10000

# Balanced: default thresholds
agentdb skill prune

# Aggressive: keep high performers only
agentdb skill prune 0.8 500
```

---

## Database Commands

### `db stats`

Show database statistics and record counts.

**Syntax:**
```bash
agentdb db stats
```

**Output:**
```
📊 Database Statistics

================================================================================
causal_edges: 47 records
causal_experiments: 3 records
causal_observations: 150 records
certificates: 1,234 records
provenance_lineage: 567 records
episodes: 2,891 records
================================================================================
```

---

## Environment Variables

### `AGENTDB_PATH`

Database file location.

**Default:** `./agentdb.db`

**Example:**
```bash
export AGENTDB_PATH="/var/data/agent-memory.db"
agentdb causal query
```

---

## Complete Example Workflow

```bash
# 1. Create an experiment
agentdb causal experiment create \
  "test-tdd-quality" \
  "use_tdd" \
  "code_quality_score"

# 2. Collect data (treatment group: TDD used)
for i in {1..30}; do
  agentdb causal experiment add-observation \
    1 true 0.85 '{"methodology": "TDD"}'
done

# 3. Collect data (control group: no TDD)
for i in {1..30}; do
  agentdb causal experiment add-observation \
    1 false 0.65 '{"methodology": "ad-hoc"}'
done

# 4. Calculate uplift
agentdb causal experiment calculate 1

# 5. Query all high-impact practices
agentdb causal query "" "code_quality_score" 0.8 0.15

# 6. Use causal recall for decision-making
agentdb recall with-certificate \
  "improve code quality" 10 0.5 0.4 0.1

# 7. Discover more patterns automatically
agentdb learner run 3 0.6 0.7

# 8. Clean up old/weak edges
agentdb learner prune 0.6 0.08 90

# 9. Check database stats
agentdb db stats
```

---

## Performance

All operations are optimized for sub-100ms latency:

| Operation | Target | Typical |
|-----------|--------|---------|
| Add edge | < 5ms | 2-3ms |
| Calculate uplift | < 100ms | 60-80ms |
| Causal query | < 20ms | 12-15ms |
| Recall with certificate | < 100ms | 75-90ms |
| Learner discovery | < 60s | 30-45s |

---

## Tips & Best Practices

### 1. Experiment Design
- Use **minimum 30 observations** per group for statistical power
- Record **context** for reproducibility
- Test **one variable** at a time (isolate cause)

### 2. Query Optimization
- Set **tight filters** (min-confidence, min-uplift) to reduce noise
- Use **specific queries** instead of broad searches
- Limit results (`[limit]`) for faster responses

### 3. Utility Weights
- **α (similarity)**: How relevant is the episode?
- **β (uplift)**: How much causal impact does it have?
- **γ (latency)**: Penalty for slow retrieval

**Recommended presets:**
```bash
# Research mode (maximize causal understanding)
agentdb recall with-certificate "query" 20 0.3 0.6 0.1

# Production mode (fast, relevant results)
agentdb recall with-certificate "query" 10 0.8 0.1 0.1

# Balanced mode (default)
agentdb recall with-certificate "query" 12 0.7 0.2 0.1
```

### 4. Learner Scheduling
Run nightly learner **daily** or **weekly**:
```bash
# Crontab: Every night at 2 AM
0 2 * * * AGENTDB_PATH=/data/agentdb.db agentdb learner run >> /var/log/learner.log 2>&1

# Weekly cleanup (Sundays at 3 AM)
0 3 * * 0 AGENTDB_PATH=/data/agentdb.db agentdb learner prune
```

---

## Troubleshooting

### Error: "Not initialized"
**Solution:** Database schema not loaded. Run:
```bash
# Initialize schema manually
sqlite3 agentdb.db < agentic-flow/src/agentdb/schemas/frontier-schema.sql
```

### Error: "Assertion failed"
**Solution:** Insufficient data for statistical test. Add more observations:
```bash
# Need at least 10 per group for t-test
agentdb causal experiment add-observation 1 true 0.5
agentdb causal experiment add-observation 1 false 0.7
```

### Slow queries
**Solution:** Optimize database:
```bash
sqlite3 agentdb.db "PRAGMA optimize;"
sqlite3 agentdb.db "ANALYZE;"
```

---

## See Also

- [AgentDB README](../../agentic-flow/src/agentdb/README.md) - Architecture overview
- [Optimization Guide](../../agentic-flow/src/agentdb/OPTIMIZATION_GUIDE.md) - Performance tuning
- [Frontier Validation](../../agentic-flow/src/agentdb/docs/FRONTIER_FEATURES_VALIDATION.md) - Test results

---

**Need help?** Open an issue at https://github.com/ruvnet/agentic-flow/issues
