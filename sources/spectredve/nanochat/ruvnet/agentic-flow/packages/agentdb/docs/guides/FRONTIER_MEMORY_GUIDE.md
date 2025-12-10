# Frontier Memory Features - Complete Guide

> Advanced memory patterns for cognitive AI agents

This guide covers all six frontier memory features in AgentDB v1.2.1, with detailed explanations, use cases, and best practices.

---

## Table of Contents

1. [Reflexion Memory](#1-reflexion-memory)
2. [Skill Library](#2-skill-library)
3. [Causal Memory Graph](#3-causal-memory-graph)
4. [Explainable Recall](#4-explainable-recall)
5. [Causal Recall](#5-causal-recall)
6. [Nightly Learner](#6-nightly-learner)

---

## 1. Reflexion Memory

**Learn from experience with self-critique and episodic replay**

### What is Reflexion Memory?

Reflexion memory implements episodic replay with self-generated critiques, allowing agents to learn from both successes and failures. Each episode is stored with:

- Task description
- Input/output data
- Success/failure status
- Reward score (0-1)
- Self-critique (what worked, what didn't)
- Performance metrics (latency, tokens)

### Commands

#### Store Episode

```bash
agentdb reflexion store <session-id> <task> <reward> <success> \
  [critique] [input] [output] [latency-ms] [tokens]
```

**Parameters:**
- `session-id` - Unique session identifier (e.g., "session-1", "debug-001")
- `task` - Task description (e.g., "fix_auth_bug", "implement_feature")
- `reward` - Success score 0.0-1.0 (higher is better)
- `success` - Boolean: `true` or `false`
- `critique` (optional) - Self-generated critique or lesson learned
- `input` (optional) - Input data or context
- `output` (optional) - Output result or solution
- `latency-ms` (optional) - Execution time in milliseconds
- `tokens` (optional) - Token count used

**Examples:**

```bash
# Successful debugging session
agentdb reflexion store "debug-001" "fix_auth_timeout" 0.95 true \
  "Increasing JWT expiry from 1h to 24h resolved random logouts" \
  "Users getting logged out randomly after 1 hour" \
  "Changed JWT_EXPIRY from 3600 to 86400 in config" \
  1800 350

# Failed attempt
agentdb reflexion store "debug-002" "fix_auth_timeout" 0.2 false \
  "Increasing rate limits didn't address the root cause - was a JWT expiry issue" \
  "Users getting logged out randomly" \
  "Tried increasing rate limits from 100/min to 1000/min" \
  3600 500

# Successful feature implementation
agentdb reflexion store "feature-005" "implement_oauth" 0.92 true \
  "OAuth2 PKCE flow with refresh tokens worked perfectly for mobile apps" \
  "Need secure authentication for mobile app" \
  "Implemented OAuth2 with authorization code + PKCE" \
  7200 1200
```

#### Retrieve Episodes

```bash
agentdb reflexion retrieve <task> [k] [min-reward] [only-failures] [only-successes]
```

**Parameters:**
- `task` - Query task (semantic search)
- `k` (optional) - Number of episodes to retrieve (default: 10)
- `min-reward` (optional) - Minimum reward threshold (default: 0.0)
- `only-failures` (optional) - Only failed episodes (`true`/`false`, default: false)
- `only-successes` (optional) - Only successful episodes (`true`/`false`, default: false)

**Examples:**

```bash
# Find similar authentication debugging sessions
agentdb reflexion retrieve "authentication timeout issues" 10 0.7

# Learn from failures only (min-reward 0.0, only failures)
agentdb reflexion retrieve "authentication" 10 0.0 true false

# Best practices only (min-reward 0.8, only successes)
agentdb reflexion retrieve "authentication" 10 0.8 false true
```

#### Critique Summary

```bash
agentdb reflexion critique-summary <task> [only-failures]
```

**Parameters:**
- `task` - Task to summarize critiques for
- `only-failures` (optional) - Only include failure critiques (default: false)

**Examples:**

```bash
# Get all critique lessons for authentication
agentdb reflexion critique-summary "authentication" false

# Learn specifically from failures
agentdb reflexion critique-summary "authentication" true
```

#### Prune Episodes

```bash
agentdb reflexion prune [max-age-days] [max-reward]
```

**Parameters:**
- `max-age-days` (optional) - Remove episodes older than N days (default: 90)
- `max-reward` (optional) - Remove episodes below reward threshold (default: 0.5)

**Examples:**

```bash
# Remove old, low-quality episodes (90+ days old, <0.5 reward)
agentdb reflexion prune 90 0.5

# More aggressive pruning (30 days, <0.7 reward)
agentdb reflexion prune 30 0.7
```

### Use Cases

1. **Debugging History**: Store every debugging session with critique
2. **Feature Development**: Track what approaches worked vs didn't
3. **Code Review**: Remember patterns that led to bugs
4. **Performance Optimization**: Learn which optimizations had impact
5. **Architecture Decisions**: Document why certain choices succeeded/failed

### Best Practices

✅ **DO:**
- Write detailed critiques explaining WHY something worked/failed
- Include specific details (file names, parameter values, error messages)
- Set reward scores consistently (0-1 scale)
- Store both successes AND failures

❌ **DON'T:**
- Skip the critique field (it's the most valuable part)
- Use vague task descriptions ("fixed bug" → "fixed JWT timeout in auth middleware")
- Forget to mark failures (they're often more educational)

---

## 2. Skill Library

**Auto-consolidate successful patterns into reusable skills**

### What is the Skill Library?

The skill library transforms repeated successful task executions into parameterized, reusable skills. Think of it as building a personal code library from experience.

Each skill tracks:
- Name and description
- Implementation code
- Success rate (updated over time)
- Usage count
- Average reward and latency

### Commands

#### Create Skill

```bash
agentdb skill create <name> <description> [code]
```

**Parameters:**
- `name` - Unique skill identifier (e.g., "jwt_auth", "rate_limiter")
- `description` - What the skill does
- `code` (optional) - Implementation code or pseudocode

**Examples:**

```bash
# JWT authentication skill
agentdb skill create "jwt_auth" "Generate secure JWT tokens with 24h expiry" \
  "const jwt = require('jsonwebtoken'); jwt.sign(payload, SECRET, {expiresIn: '24h'});"

# Rate limiting skill
agentdb skill create "redis_rate_limit" "Redis-based rate limiting with token bucket" \
  "const redis = require('redis'); // Token bucket implementation with INCR + EXPIRE"

# Error handling skill
agentdb skill create "retry_with_backoff" "Exponential backoff retry logic" \
  "async function retryWithBackoff(fn, maxRetries=3) { /* implementation */ }"
```

#### Search Skills

```bash
agentdb skill search <query> [k]
```

**Parameters:**
- `query` - Semantic search query
- `k` (optional) - Number of skills to return (default: 5)

**Examples:**

```bash
# Find authentication-related skills
agentdb skill search "authentication security" 5

# Find rate limiting approaches
agentdb skill search "rate limiting API protection" 3

# Find error handling patterns
agentdb skill search "error retry resilience" 5
```

#### Auto-Consolidate Skills

```bash
agentdb skill consolidate [min-attempts] [min-reward] [time-window-days]
```

**Parameters:**
- `min-attempts` (optional) - Minimum episode attempts (default: 3)
- `min-reward` (optional) - Minimum average reward (default: 0.7)
- `time-window-days` (optional) - Look back N days (default: 7)

**Examples:**

```bash
# Default: 3+ attempts, 70%+ reward, last 7 days
agentdb skill consolidate

# More selective: 5+ attempts, 80%+ reward, last 14 days
agentdb skill consolidate 5 0.8 14

# Find any patterns: 2+ attempts, 60%+ reward, last 30 days
agentdb skill consolidate 2 0.6 30
```

**What Gets Consolidated:**
This command scans reflexion episodes for repeated successful patterns:
- Groups episodes by similar tasks
- Identifies patterns attempted 3+ times with 70%+ success
- Auto-creates skills from the successful approaches
- Extracts common code patterns

#### Prune Skills

```bash
agentdb skill prune [min-uses] [min-success-rate] [max-age-days]
```

**Parameters:**
- `min-uses` (optional) - Minimum usage count (default: 3)
- `min-success-rate` (optional) - Minimum success rate (default: 0.4)
- `max-age-days` (optional) - Maximum age (default: 60)

**Examples:**

```bash
# Default: prune skills with <3 uses, <40% success, or >60 days old
agentdb skill prune

# More aggressive: <5 uses, <60% success, or >30 days
agentdb skill prune 5 0.6 30
```

### Use Cases

1. **Code Generation**: Reuse proven patterns for common tasks
2. **Architecture Patterns**: Document successful design patterns
3. **API Integration**: Store working API call patterns
4. **Testing Strategies**: Reuse effective test approaches
5. **Deployment Scripts**: Consolidate deployment workflows

### Best Practices

✅ **DO:**
- Give skills descriptive names (what they do, not what they are)
- Include actual code snippets when possible
- Let auto-consolidation find patterns for you
- Regularly search skills before implementing new features

❌ **DON'T:**
- Create overly specific skills (too narrow to reuse)
- Skip the description field
- Keep underperforming skills (use prune)
- Forget to update skills as patterns evolve

---

## 3. Causal Memory Graph

**Intervention-based causality with p(y|do(x)) semantics**

### What is Causal Memory?

Causal memory learns cause-and-effect relationships between agent actions using Pearl's do-calculus. Instead of just correlation (p(y|x)), it tracks interventional probabilities (p(y|do(x))) - what happens when you actively DO something.

**Key Concepts:**
- **Causal Edge**: A → B means "doing A causes effect B"
- **Uplift**: The causal effect magnitude (treatment - control)
- **Confidence**: Statistical confidence in the relationship
- **Experiment**: A/B test to measure causal effects

### Commands

#### Manual Edge Creation

```bash
agentdb causal add-edge <cause> <effect> <uplift> [confidence] [sample-size]
```

**Parameters:**
- `cause` - Intervention/treatment (e.g., "add_tests", "optimize_query")
- `effect` - Outcome (e.g., "code_quality", "response_time")
- `uplift` - Measured effect size (can be negative)
- `confidence` (optional) - Statistical confidence 0-1 (default: 0.9)
- `sample-size` (optional) - Number of observations (default: 10)

**Examples:**

```bash
# Adding tests improves code quality (uplift = 0.25, confidence = 0.95)
agentdb causal add-edge "add_tests" "code_quality" 0.25 0.95 100

# Code reviews reduce bugs (uplift = -0.30, negative means fewer bugs)
agentdb causal add-edge "code_review" "bug_rate" -0.30 0.92 80

# Caching improves response time (uplift = -150ms)
agentdb causal add-edge "add_caching" "response_time" -150 0.98 50
```

#### Query Causal Edges

```bash
agentdb causal query <cause> [effect] [min-confidence] [min-uplift] [limit]
```

**Parameters:**
- `cause` - Intervention to query (REQUIRED)
- `effect` (optional) - Specific outcome to filter
- `min-confidence` (optional) - Minimum confidence (default: 0.7)
- `min-uplift` (optional) - Minimum absolute uplift (default: 0.1)
- `limit` (optional) - Max results (default: 10)

**Examples:**

```bash
# What are the effects of adding tests?
agentdb causal query "add_tests" "" 0.7 0.1 10

# Does adding tests improve code quality specifically?
agentdb causal query "add_tests" "code_quality" 0.7 0.1 10

# High-confidence edges only (95%+)
agentdb causal query "add_tests" "" 0.95 0.1 10
```

#### A/B Experiments

**Step 1: Create Experiment**

```bash
agentdb causal experiment create <name> <cause> <effect>
```

**Parameters:**
- `name` - Experiment identifier
- `cause` - Treatment variable
- `effect` - Outcome variable

**Example:**

```bash
agentdb causal experiment create "test_coverage_quality" "add_tests" "code_quality"
```

**Step 2: Record Observations**

```bash
agentdb causal experiment add-observation <experiment-id> <is-treatment> <outcome> [context]
```

**Parameters:**
- `experiment-id` - Experiment ID (returned from create)
- `is-treatment` - `true` for treatment group, `false` for control
- `outcome` - **NUMERIC** outcome value (e.g., 0.85, -150, 42)
- `context` (optional) - JSON context metadata

**⚠️ IMPORTANT**: Outcomes MUST be numeric values!

**Examples:**

```bash
# Treatment group (with tests) - quality scores 0-1
agentdb causal experiment add-observation 1 true 0.85
agentdb causal experiment add-observation 1 true 0.88
agentdb causal experiment add-observation 1 true 0.90

# Control group (no tests) - quality scores 0-1
agentdb causal experiment add-observation 1 false 0.65
agentdb causal experiment add-observation 1 false 0.60
agentdb causal experiment add-observation 1 false 0.62
```

**Step 3: Calculate Causal Effect**

```bash
agentdb causal experiment calculate <experiment-id>
```

**Example:**

```bash
agentdb causal experiment calculate 1
```

**Output:**

```
📈 Calculating Uplift
Experiment: Does add_tests causally affect code_quality?
Treatment Mean: 0.877
Control Mean: 0.625
✅ Uplift: 0.252
95% CI: [0.210, 0.293]
p-value: 0.0030
Sample Sizes: 3 treatment, 2 control
✅ Result is statistically significant (p < 0.05)
```

**What You Get:**
- **Treatment Mean**: Average outcome in treatment group
- **Control Mean**: Average outcome in control group
- **Uplift**: Causal effect (treatment - control)
- **95% CI**: Confidence interval for uplift
- **p-value**: Statistical significance (p < 0.05 = significant)
- **Sample Sizes**: Observations in each group

### Use Cases

1. **Code Quality**: Does adding tests improve code quality?
2. **Performance**: Does caching reduce response time?
3. **Debugging**: Which debugging strategies fix bugs?
4. **Architecture**: Do certain patterns reduce coupling?
5. **Process**: Does code review reduce bug rates?

### Best Practices

✅ **DO:**
- Use numeric outcomes (quality scores 0-1, latency in ms, bug counts)
- Run experiments with adequate sample sizes (10+ per group)
- Check statistical significance (p < 0.05)
- Interpret confidence intervals, not just point estimates

❌ **DON'T:**
- Use string outcomes ("good", "bad") - must be numeric
- Draw conclusions from tiny samples (< 5 observations)
- Ignore p-values (could be random chance)
- Confuse correlation with causation (use experiments!)

### Statistical Interpretation

**P-value interpretation:**
- `p < 0.001` - Extremely strong evidence
- `p < 0.01` - Very strong evidence
- `p < 0.05` - Strong evidence (standard threshold)
- `p > 0.05` - Weak/no evidence (could be chance)

**Confidence Interval interpretation:**
- If CI doesn't include 0 → effect is real
- Narrow CI → precise estimate
- Wide CI → need more data

---

## 4. Explainable Recall

**Provenance tracking with cryptographic Merkle proofs**

### What is Explainable Recall?

Every memory retrieval comes with a "certificate" explaining why those specific memories were selected, backed by cryptographic proof of completeness using Merkle trees.

**Certificate Contents:**
- 🔐 Cryptographic proof (SHA-256 Merkle root)
- 📊 Retrieval parameters used (k, α, β, γ)
- ✅ Completeness score (0.0-1.0)
- ⏱️ Query execution latency
- 🎯 Retrieved memory IDs with scores

### Command

```bash
agentdb recall with-certificate <query> [k] [alpha] [beta] [gamma]
```

**Parameters:**
- `query` - Semantic search query
- `k` (optional) - Number of results (default: 12)
- `alpha` (optional) - Similarity weight (default: 0.7)
- `beta` (optional) - Uplift weight (default: 0.2)
- `gamma` (optional) - Latency penalty (default: 0.1)

**Examples:**

```bash
# Default balanced retrieval
agentdb recall with-certificate "successful API optimization" 5

# Emphasize similarity over causality (α=0.9, β=0.1)
agentdb recall with-certificate "database performance" 10 0.9 0.1 0.0

# Emphasize what works over what's similar (α=0.5, β=0.4)
agentdb recall with-certificate "debugging strategies" 8 0.5 0.4 0.1

# Ignore latency, focus on similarity + causality (γ=0.0)
agentdb recall with-certificate "authentication patterns" 5 0.7 0.3 0.0
```

### Output Format

```
🔍 Causal Recall with Certificate
Query: "successful API optimization"
k: 5

════════════════════════════════════════════════════════════
Results (3)

#1: episode 42
  Content: Implemented Redis caching for frequently accessed endpoints...
  Similarity: 0.912
  Uplift: 0.450
  Utility: 0.829

#2: episode 17
  Content: Optimized database queries with proper indexing...
  Similarity: 0.887
  Uplift: 0.380
  Utility: 0.797

#3: episode 9
  Content: Added connection pooling to reduce latency...
  Similarity: 0.856
  Uplift: 0.320
  Utility: 0.763

════════════════════════════════════════════════════════════
Certificate ID: 9516f6115248be471ada6086f3edf41c62421ced47165a819064517da731ae71
Query: successful API optimization
Completeness: 0.87
✅ Completed in 12ms
```

### Certificate Verification

The certificate provides:

1. **Merkle Root Hash**: Cryptographic proof of retrieved memories
2. **Completeness Score**: How much of the search space was covered (0-1)
3. **Query Parameters**: Exact α, β, γ weights used
4. **Execution Metadata**: Latency, timestamp

You can verify:
- The certificate matches the retrieved memories (hash verification)
- The completeness score is high (>0.8 is good)
- The query parameters match your intent

### Use Cases

1. **Debugging Decisions**: Why did the agent choose this approach?
2. **Audit Trail**: Prove which memories influenced decisions
3. **Trust Building**: Show users why recommendations were made
4. **Performance Analysis**: Understand retrieval completeness
5. **Research**: Analyze which memories are most influential

### Best Practices

✅ **DO:**
- Check completeness scores (>0.8 is comprehensive)
- Use certificates for audit logs
- Verify Merkle roots for critical decisions
- Store certificate IDs with important retrievals

❌ **DON'T:**
- Ignore low completeness scores (<0.5)
- Trust retrievals without checking certificates
- Forget to log certificate IDs for debugging

---

## 5. Causal Recall

**Utility-based reranking combining similarity, causality, and latency**

### What is Causal Recall?

Standard vector search returns the most similar memories. Causal recall reranks by actual utility, considering:

- **Similarity**: How relevant is this memory? (semantic matching)
- **Uplift**: Does this approach actually work? (causal effectiveness)
- **Latency**: How expensive was this approach? (efficiency)

### Utility Formula

```
U = α · similarity + β · uplift − γ · latency
```

**Parameters:**
- **α (alpha)**: Weight for semantic similarity (default: 0.7)
- **β (beta)**: Weight for causal uplift (default: 0.2)
- **γ (gamma)**: Penalty for execution latency (default: 0.1)

### Command

```bash
agentdb recall with-certificate <query> [k] [alpha] [beta] [gamma]
```

Same command as explainable recall - it includes both features!

### Parameter Tuning

**Similarity-focused (α=0.9, β=0.1, γ=0.0):**
```bash
agentdb recall with-certificate "authentication" 10 0.9 0.1 0.0
```
Use when: You want the most relevant memories, regardless of effectiveness.

**Effectiveness-focused (α=0.5, β=0.4, γ=0.1):**
```bash
agentdb recall with-certificate "debugging" 10 0.5 0.4 0.1
```
Use when: You want approaches that actually work, even if less similar.

**Balanced (α=0.7, β=0.2, γ=0.1):**
```bash
agentdb recall with-certificate "optimization" 10 0.7 0.2 0.1
```
Use when: You want a good mix of relevance and effectiveness (DEFAULT).

**Speed-focused (α=0.6, β=0.2, γ=0.2):**
```bash
agentdb recall with-certificate "algorithms" 10 0.6 0.2 0.2
```
Use when: Execution time is critical (high latency penalty).

### Example Output

```
#1: episode 42
  Content: Implemented Redis caching...
  Similarity: 0.912  (high relevance)
  Uplift: 0.450      (very effective)
  Utility: 0.829     (α·0.912 + β·0.450 - γ·0.012)

#2: episode 17
  Content: Optimized database queries...
  Similarity: 0.887  (high relevance)
  Uplift: 0.380      (effective)
  Utility: 0.797     (α·0.887 + β·0.380 - γ·0.008)
```

### Use Cases

1. **Debugging**: Retrieve fixes that actually worked
2. **Optimization**: Find effective performance improvements
3. **Architecture**: Choose patterns with proven benefits
4. **Learning**: Study high-uplift approaches
5. **Decision Making**: Balance relevance with effectiveness

### Best Practices

✅ **DO:**
- Adjust α, β, γ based on your task
- Use high β when effectiveness matters most
- Use high γ when speed is critical
- Check both similarity AND uplift scores

❌ **DON'T:**
- Use default weights blindly
- Ignore uplift scores (they show what works)
- Set all weights to 1.0 (they should sum to ~1.0)
- Forget the latency penalty for time-critical tasks

---

## 6. Nightly Learner

**Automated pattern discovery while you sleep**

### What is the Nightly Learner?

The nightly learner runs automated causal discovery on your episode history, finding patterns you didn't explicitly program. It's like having a data scientist analyze your agent's performance overnight.

**What It Discovers:**
- Causal edges between memory types
- Successful skill patterns (auto-consolidation)
- Statistical significance (p-values, confidence intervals)
- Uplift measurements (effect sizes)

### Commands

#### Discover Patterns

```bash
agentdb learner run [min-attempts] [min-success-rate] [min-confidence] [dry-run]
```

**Parameters:**
- `min-attempts` (optional) - Minimum pattern occurrences (default: 3)
- `min-success-rate` (optional) - Minimum success rate (default: 0.6)
- `min-confidence` (optional) - Statistical confidence (default: 0.7)
- `dry-run` (optional) - Show what would be created without creating (default: false)

**Examples:**

```bash
# Dry-run to see what would be discovered (safe to run anytime)
agentdb learner run 3 0.6 0.7 true

# Actually discover and create patterns (DEFAULT)
agentdb learner run

# More selective discovery (5+ attempts, 80%+ success)
agentdb learner run 5 0.8 0.8

# Find any patterns (2+ attempts, 50%+ success)
agentdb learner run 2 0.5 0.6
```

**What Gets Created:**
1. **Causal Edges**: A → B relationships with uplift measurements
2. **Skills**: Auto-consolidated from successful patterns
3. **Statistics**: Confidence intervals, p-values, sample sizes

**Output:**

```
🌙 Running Nightly Learner
Min Attempts: 3
Min Success Rate: 0.6
Min Confidence: 0.7

Discovered Patterns:
  • add_tests → code_quality (uplift: 0.25, p=0.003, n=12)
  • code_review → bug_rate (uplift: -0.30, p=0.001, n=18)
  • add_caching → response_time (uplift: -120ms, p=0.008, n=9)

Created 3 causal edges
Auto-consolidated 2 skills

✅ Discovered 3 causal edges in 2.4s
```

#### Prune Low-Quality Edges

```bash
agentdb learner prune [min-confidence] [min-uplift] [max-age-days]
```

**Parameters:**
- `min-confidence` (optional) - Minimum confidence to keep (default: 0.5)
- `min-uplift` (optional) - Minimum absolute uplift (default: 0.05)
- `max-age-days` (optional) - Maximum age to keep (default: 90)

**Examples:**

```bash
# Default pruning (confidence <0.5, uplift <0.05, age >90 days)
agentdb learner prune

# Aggressive quality filter (confidence <0.8, uplift <0.2)
agentdb learner prune 0.8 0.2 60

# Time-based only (remove >30 days old)
agentdb learner prune 0.0 0.0 30
```

### Discovery Algorithm

The learner uses **doubly robust causal estimation**:

1. **Pattern Mining**: Groups similar task episodes
2. **A/B Comparison**: Compares treatment vs control groups
3. **Statistical Testing**: Calculates p-values and confidence intervals
4. **Effect Estimation**: Measures uplift with regression + propensity scoring
5. **Quality Filtering**: Only keeps statistically significant patterns

### Use Cases

1. **Overnight Analysis**: Run before bed, wake up to insights
2. **CI/CD Integration**: Run after deployments to learn patterns
3. **Weekly Reports**: Schedule to run weekly for pattern reports
4. **Quality Maintenance**: Auto-prune old patterns regularly
5. **Knowledge Discovery**: Find patterns you didn't know existed

### Best Practices

✅ **DO:**
- Run dry-run first to see what would be discovered
- Schedule to run overnight or during low-activity periods
- Regularly prune low-quality patterns
- Review discovered patterns for insights
- Adjust thresholds based on your data volume

❌ **DON'T:**
- Run on empty databases (need 10+ episodes minimum)
- Set thresholds too low (min-confidence <0.5 is noise)
- Ignore discovered patterns (they're valuable insights!)
- Run too frequently (once per day/week is enough)
- Forget to prune old patterns (quality degrades over time)

### Interpretation

**Good Pattern:**
```
add_tests → code_quality (uplift: 0.25, p=0.003, n=12)
```
- Uplift of 0.25 (meaningful effect)
- p=0.003 (highly significant, p<0.01)
- n=12 (adequate sample size)
- **Action**: Trust this pattern, it's robust

**Marginal Pattern:**
```
refactor → maintainability (uplift: 0.08, p=0.045, n=5)
```
- Uplift of 0.08 (small effect)
- p=0.045 (barely significant, p≈0.05)
- n=5 (small sample)
- **Action**: Collect more data before trusting

**Noise:**
```
random_action → outcome (uplift: 0.03, p=0.32, n=3)
```
- Uplift of 0.03 (tiny effect)
- p=0.32 (not significant, p>0.05)
- n=3 (very small sample)
- **Action**: Ignore, likely random chance

---

## Integration Patterns

### Pattern 1: Learning Loop

```bash
# 1. Execute tasks and store episodes
agentdb reflexion store "session-1" "fix_bug" 0.95 true "Used debugger effectively"
agentdb reflexion store "session-2" "fix_bug" 0.90 true "Added logging statements"

# 2. Retrieve similar episodes when needed
agentdb reflexion retrieve "debugging" 10 0.7

# 3. Overnight: discover patterns
agentdb learner run 3 0.6 0.7

# 4. Query discovered causality
agentdb causal query "add_logging" "bug_fixed" 0.7 0.1 10

# 5. Apply insights to new tasks
agentdb recall with-certificate "fix new bug" 5 0.5 0.4 0.1
```

### Pattern 2: Skill Building

```bash
# 1. Create skills manually
agentdb skill create "jwt_auth" "Secure JWT generation" "code..."

# 2. Execute tasks using skills
agentdb reflexion store "session-1" "implement_auth" 0.92 true "Used jwt_auth skill"

# 3. Auto-consolidate new skills
agentdb skill consolidate 3 0.7 7

# 4. Search for applicable skills
agentdb skill search "authentication" 5

# 5. Prune underperforming skills
agentdb skill prune 3 0.4 60
```

### Pattern 3: A/B Testing Workflow

```bash
# 1. Create experiment
agentdb causal experiment create "caching_perf" "add_caching" "response_time"

# 2. Collect data (alternating treatment/control)
agentdb causal experiment add-observation 1 true 150   # with cache
agentdb causal experiment add-observation 1 false 280  # without cache
agentdb causal experiment add-observation 1 true 145   # with cache
agentdb causal experiment add-observation 1 false 290  # without cache

# 3. Calculate causal effect
agentdb causal experiment calculate 1

# 4. If significant, add to knowledge base
agentdb causal add-edge "add_caching" "response_time" -130 0.95 4

# 5. Retrieve when optimizing
agentdb recall with-certificate "optimize performance" 5 0.7 0.3 0.0
```

---

## Troubleshooting

### Issue: No results from recall

**Possible causes:**
- Empty database (run `agentdb db stats`)
- Query too specific (try broader terms)
- Thresholds too strict (lower α, β, γ)

**Solution:**
```bash
# Check database
agentdb db stats

# Try broader query
agentdb recall with-certificate "general query" 10

# Lower thresholds
agentdb recall with-certificate "query" 10 0.5 0.0 0.0
```

### Issue: Learner finds no patterns

**Possible causes:**
- Insufficient episodes (need 10+ minimum)
- min-attempts too high (lower to 2)
- min-success-rate too high (try 0.5)

**Solution:**
```bash
# Check episode count
agentdb db stats

# Lower thresholds
agentdb learner run 2 0.5 0.6 true  # dry-run first
```

### Issue: Experiment shows no significance

**Possible causes:**
- Sample size too small (need 5+ per group)
- Effect size too small (real effect might be tiny)
- High variance in data (need more samples)

**Solution:**
```bash
# Collect more observations
agentdb causal experiment add-observation 1 true 0.85
agentdb causal experiment add-observation 1 true 0.88
# ... collect 10+ per group

# Re-calculate
agentdb causal experiment calculate 1
```

---

## Next Steps

1. **Start Simple**: Begin with reflexion memory and skill library
2. **Build History**: Collect 20-30 episodes before using learner
3. **Run Experiments**: Test causal hypotheses with A/B tests
4. **Automate**: Schedule nightly learner for pattern discovery
5. **Optimize**: Tune α, β, γ for your specific use cases

For complete CLI reference, see [CLI_GUIDE.md](./CLI_GUIDE.md).

For API usage, see [API.md](./API.md).

For MCP integration, see [MCP.md](./MCP.md).
