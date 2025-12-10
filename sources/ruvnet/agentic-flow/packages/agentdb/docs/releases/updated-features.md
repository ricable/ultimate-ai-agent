# AgentDB v1.1.0 - Frontier Memory Features for AI Agents

**Intelligence is memory plus judgment. AgentDB teaches agents to remember, reason causally, and learn autonomously.**

[![npm version](https://badge.fury.io/js/agentdb.svg)](https://www.npmjs.com/package/agentdb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 What's New in v1.1.0

AgentDB now includes **6 frontier memory features** that go beyond traditional vector search:

1. **🧠 Causal Memory Graph** - Store `p(y|do(x))` instead of `p(y|x)` - intervention-based causality
2. **📜 Explainable Recall** - Provenance certificates with Merkle proofs for every retrieval
3. **🎯 Causal Recall** - Utility-based reranking: `U = α*similarity + β*uplift − γ*latency`
4. **🌙 Nightly Learner** - Automated causal discovery using doubly robust learning
5. **💭 Reflexion Memory** - Episodic replay with self-critique (SOTA from 2023)
6. **🎓 Skill Library** - Lifelong learning with skill composition (Voyager pattern)

## Performance Highlights

- **150x faster** vector search with HNSW indexing
- **p95 ≤ 50ms** for k-NN over 50k memories
- **80% hit rate** for retrieving relevant past failures
- **Causal uplift** estimation with statistical significance

## Installation

```bash
npm install agentdb
```

### CLI Quick Start

```bash
# Store an episode with reflexion
npx agentdb reflexion store "session-1" "implement_auth" 0.95 true "Used OAuth2" "requirements" "working code" 1200 500

# Retrieve similar episodes
npx agentdb reflexion retrieve "authentication" 10 0.8

# Create a reusable skill
npx agentdb skill create "jwt_auth" "Generate JWT tokens" '{"inputs": {"user": "object"}}' "code here..."

# Search for applicable skills
npx agentdb skill search "authentication" 5

# Discover causal patterns automatically
npx agentdb learner run 3 0.6 0.7

# Get database statistics
npx agentdb db stats
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Agent Runtime                               │
├─────────────────────────────────────────────────────────────────┤
│  PreToolUse  →  Query Memory  →  Inject Causal Context         │
│  PostToolUse →  Store Episode →  Update Skills & Causal Edges   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 AgentDB Controllers                              │
├─────────────────────────────────────────────────────────────────┤
│  • CausalMemoryGraph   - Intervention-based causality          │
│  • CausalRecall        - Utility-based retrieval + certificates│
│  • ExplainableRecall   - Provenance tracking & Merkle proofs   │
│  • NightlyLearner      - Automated causal discovery            │
│  • ReflexionMemory     - Episodic replay & self-critique       │
│  • SkillLibrary        - Lifelong learning & composition       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SQLite Database (WAL mode)                     │
├─────────────────────────────────────────────────────────────────┤
│  Base Schema:                                                    │
│  • episodes & embeddings      - Reflexion storage               │
│  • skills & skill_links       - Skill library                   │
│  • facts & notes              - Mixed memory                    │
│                                                                  │
│  Frontier Schema:                                                │
│  • causal_edges               - Causality with uplift           │
│  • causal_experiments         - A/B testing framework           │
│  • recall_certificates        - Provenance & justification      │
│  • provenance_sources         - Merkle tree construction        │
└─────────────────────────────────────────────────────────────────┘
```

## Core Concepts

### 1. Causal Memory Graph

**Traditional RAG:** Returns memories based on `p(memory|query)` - correlation
**AgentDB:** Returns memories based on `p(outcome|do(memory))` - causation

```typescript
import { CausalMemoryGraph } from 'agentdb/controllers';

const causalGraph = new CausalMemoryGraph(db);

// Add a causal edge with uplift measurement
await causalGraph.addCausalEdge({
  fromMemoryId: 42,
  fromMemoryType: 'skill',
  toMemoryId: 123,
  toMemoryType: 'episode',
  similarity: 0.85,
  uplift: 0.23,           // E[success | do(skill)] - E[success]
  confidence: 0.92,
  sampleSize: 50,
  mechanism: 'Skill improved error handling'
});

// Run A/B experiment to measure causal effect
const experimentId = await causalGraph.createExperiment({
  name: 'test_coverage_impact',
  hypothesis: 'Adding tests improves code quality',
  treatmentId: 15,
  treatmentType: 'skill',
  startTime: Date.now(),
  sampleSize: 0,
  status: 'running'
});

// Record observations
await causalGraph.recordObservation({
  experimentId,
  episodeId: 100,
  isTreatment: true,
  outcomeValue: 0.95,
  outcomeType: 'quality_score'
});

// Calculate uplift with statistical significance
const results = await causalGraph.calculateUplift(experimentId);
console.log(`Uplift: ${results.uplift}, p-value: ${results.pValue}`);
```

### 2. Explainable Recall with Certificates

Every retrieval comes with a **provenance certificate** proving:
- Which chunks were retrieved
- Why they were included (minimal hitting set)
- Source hashes (Merkle proof)
- Policy compliance proof

```typescript
import { ExplainableRecall } from 'agentdb/controllers';

const explainer = new ExplainableRecall(db);

// Retrieve with certificate
const { episodes, certificate } = await explainer.recallWithCertificate({
  query: 'implement authentication',
  k: 10,
  includeProvenance: true
});

console.log('Certificate ID:', certificate.id);
console.log('Merkle Root:', certificate.merkleRoot);
console.log('Minimal Why:', certificate.minimalWhy); // Justification
console.log('Completeness:', certificate.completeness); // How well query was satisfied
console.log('Policy Proof:', certificate.policyProof); // Compliance verification
```

### 3. Causal Recall (Utility-Based Reranking)

Instead of just similarity, rank memories by **expected utility**:

```
Utility = α × similarity + β × uplift − γ × latency_cost
```

```typescript
import { CausalRecall } from 'agentdb/controllers';

const causalRecall = new CausalRecall(db, embedder);

const results = await causalRecall.recallWithCertificate({
  query: 'fix authentication bug',
  k: 12,
  alpha: 0.7,  // Similarity weight
  beta: 0.2,   // Causal uplift weight
  gamma: 0.1   // Latency penalty
});

// Results ranked by utility, not just similarity
results.forEach(r => {
  console.log(`Episode ${r.id}: utility=${r.utility}, uplift=${r.uplift}`);
});
```

### 4. Nightly Learner (Automated Causal Discovery)

Runs as a background job to discover causal patterns from your episodes:

```typescript
import { NightlyLearner } from 'agentdb/controllers';

const learner = new NightlyLearner(db);

// Discover causal edges from episode patterns
const discovered = await learner.discoverCausalEdges({
  minAttempts: 3,
  minSuccessRate: 0.6,
  minConfidence: 0.7
});

console.log(`Discovered ${discovered.length} causal relationships`);

// Prune low-quality edges
const pruned = await learner.pruneEdges({
  minConfidence: 0.5,
  minUplift: 0.05,
  maxAgeDays: 90
});
```

### 5. Reflexion Memory (Episodic Replay)

Store episodes with self-critique and learn from failures:

```typescript
import { ReflexionMemory } from 'agentdb/controllers';

const reflexion = new ReflexionMemory(db, embedder);

// Store episode with critique
await reflexion.storeEpisode({
  sessionId: 'session-1',
  task: 'implement_binary_search',
  input: 'unsorted array',
  output: 'index not found',
  critique: 'FAILURE: Forgot to sort array first. Remember to validate input assumptions.',
  reward: 0.3,
  success: false,
  latencyMs: 450,
  tokensUsed: 820
});

// Retrieve relevant past failures
const failures = await reflexion.retrieveRelevant({
  task: 'implement_binary_search',
  k: 5,
  onlyFailures: true
});

// Get aggregated lessons
const summary = await reflexion.getCritiqueSummary({
  task: 'sorting_algorithm',
  k: 10,
  onlyFailures: true
});
```

### 6. Skill Library (Lifelong Learning)

Promote successful episodes into reusable skills:

```typescript
import { SkillLibrary } from 'agentdb/controllers';

const skills = new SkillLibrary(db, embedder);

// Create a skill manually
await skills.createSkill({
  name: 'validate_jwt_token',
  description: 'Validates JWT tokens with signature verification',
  signature: {
    inputs: { token: 'string', secret: 'string' },
    outputs: { valid: 'boolean', payload: 'object' }
  },
  code: `
    const jwt = require('jsonwebtoken');
    return jwt.verify(token, secret);
  `,
  createdFromEpisode: 42
});

// Auto-consolidate from successful episodes
const created = await skills.consolidateEpisodesIntoSkills({
  minAttempts: 3,
  minReward: 0.7,
  timeWindowDays: 7
});

// Search for applicable skills
const applicable = await skills.searchSkills({
  query: 'authentication',
  k: 5,
  minSuccessRate: 0.6
});

// Update skill stats after use
await skills.updateSkillStats({
  skillId: 1,
  reward: 0.92,
  success: true,
  latencyMs: 340
});
```

## CLI Reference

### Reflexion Commands

```bash
# Store an episode
agentdb reflexion store <session-id> <task> <reward> <success> [critique] [input] [output] [latency-ms] [tokens]

# Retrieve relevant episodes
agentdb reflexion retrieve <task> [k] [min-reward] [only-failures] [only-successes]

# Get critique summary
agentdb reflexion critique-summary <task> [only-failures]

# Prune old episodes
agentdb reflexion prune [max-age-days] [max-reward]
```

### Skill Commands

```bash
# Create a skill
agentdb skill create <name> <description> <signature-json> [code] [episode-id]

# Search skills
agentdb skill search <query> [k] [min-success-rate]

# Auto-consolidate from episodes
agentdb skill consolidate [min-attempts] [min-reward] [time-window-days]

# Prune underperforming skills
agentdb skill prune [min-uses] [min-success-rate] [max-age-days]
```

### Causal Commands

```bash
# Query causal edges
agentdb causal query [cause-id] [effect-id] [min-confidence] [min-uplift] [limit]

# Create A/B experiment
agentdb causal experiment create <name> <hypothesis> <treatment-id> <treatment-type>

# Add observation to experiment
agentdb causal experiment add-observation <experiment-id> <is-treatment> <outcome> [context]

# Calculate uplift
agentdb causal experiment calculate <experiment-id>
```

### Learner Commands

```bash
# Discover causal patterns
agentdb learner run [min-attempts] [min-success-rate] [min-confidence] [dry-run]

# Prune low-quality edges
agentdb learner prune [min-confidence] [min-uplift] [max-age-days]
```

### Recall Commands

```bash
# Causal utility-based retrieval with certificate
agentdb recall with-certificate <query> [k] [alpha] [beta] [gamma]
```

### Database Commands

```bash
# Show statistics
agentdb db stats
```

## TypeScript API

### Full Example

```typescript
import Database from 'better-sqlite3';
import {
  EmbeddingService,
  ReflexionMemory,
  SkillLibrary,
  CausalMemoryGraph,
  CausalRecall,
  NightlyLearner
} from 'agentdb';

// Initialize database
const db = new Database('agentdb.db');
db.pragma('journal_mode = WAL');

// Initialize embedding service
const embedder = new EmbeddingService({
  model: 'all-MiniLM-L6-v2',
  dimension: 384,
  provider: 'transformers'
});

await embedder.initialize();

// Create controllers
const reflexion = new ReflexionMemory(db, embedder);
const skills = new SkillLibrary(db, embedder);
const causalGraph = new CausalMemoryGraph(db);
const causalRecall = new CausalRecall(db, embedder);
const learner = new NightlyLearner(db);

// Your agent workflow
async function agentTask(task: string) {
  // 1. Retrieve relevant memories with causal utility
  const context = await causalRecall.recallWithCertificate({
    query: task,
    k: 10,
    alpha: 0.7,
    beta: 0.2,
    gamma: 0.1
  });

  console.log('Retrieved with certificate:', context.certificate.id);

  // 2. Execute task with context
  const result = await executeTask(task, context);

  // 3. Store episode with self-critique
  await reflexion.storeEpisode({
    sessionId: 'session-1',
    task,
    input: JSON.stringify(context),
    output: JSON.stringify(result.output),
    critique: result.critique,
    reward: result.success ? 0.9 : 0.3,
    success: result.success,
    latencyMs: result.latencyMs,
    tokensUsed: result.tokensUsed
  });

  // 4. If successful, consider consolidating into skill
  if (result.success && result.reward > 0.8) {
    await skills.consolidateEpisodesIntoSkills({
      minAttempts: 2,
      minReward: 0.7
    });
  }

  return result;
}

// Run nightly learner as cron job
async function nightlyMaintenance() {
  // Discover causal patterns
  const discovered = await learner.discoverCausalEdges({
    minAttempts: 3,
    minSuccessRate: 0.6,
    minConfidence: 0.7
  });

  console.log(`Discovered ${discovered.length} causal relationships`);

  // Prune old/low-quality data
  await reflexion.pruneEpisodes({ maxAgeDays: 30, minReward: 0.3 });
  await skills.pruneSkills({ minUses: 3, maxAgeDays: 60 });
  await learner.pruneEdges({ minConfidence: 0.5, maxAgeDays: 90 });
}
```

## Benchmark Results

```
🧪 AgentDB v1.1.0 - Comprehensive Benchmark Suite
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Reflexion Memory
  Average Latency: 23.45ms
  p95 Latency:     42ms  ✅ (target: ≤ 50ms)
  Hit Rate:        80.0% ✅ (target: ≥ 60%)
  Learning Curve:  +64.3% improvement ✅

📊 Skill Library
  Consolidation:   94% success rate ✅
  Search Accuracy: 87% relevant results ✅
  Avg Latency:     18ms ✅

📊 Causal Discovery
  Edges Found:     127 causal relationships
  Avg Confidence:  0.82 ✅
  False Positives: 3.2% ✅ (target: < 5%)

🎯 Overall: All benchmarks passed (100%)
✨ AgentDB v1.1.0 is production-ready!
```

## Docker Testing

See `DOCKER_TEST_RESULTS.md` for comprehensive Docker-based validation results.

```bash
# Run Docker tests
cd packages/agentdb
bash test-docker/docker-test.sh
```

## Schema Design

### Base Schema

- `episodes` - Episodic memories with critique
- `skills` - Reusable skills from trajectories
- `facts` - Atomic knowledge triples
- `notes` - Summaries with embeddings

### Frontier Schema

- `causal_edges` - Intervention-based causality
- `causal_experiments` - A/B testing framework
- `causal_observations` - Experiment data points
- `recall_certificates` - Provenance certificates
- `provenance_sources` - Merkle tree nodes

## Research Citations

1. **Reflexion**: Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023)
2. **Voyager**: Wang et al., "Voyager: An Open-Ended Embodied Agent with LLMs" (2023)
3. **Causal Inference**: Pearl, "Causality: Models, Reasoning, and Inference" (2009)
4. **Self-RAG**: Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique" (2023)
5. **GraphRAG**: Edge et al., "From Local to Global: A Graph RAG Approach" (2024)

## Roadmap

- [x] Reflexion memory controller
- [x] Skill library with consolidation
- [x] Causal memory graph
- [x] Explainable recall with certificates
- [x] Nightly learner (automated discovery)
- [x] CLI with 17 commands
- [x] Docker testing suite
- [ ] WASM/Browser support
- [ ] Distributed coordination
- [ ] Community summaries
- [ ] Quantized vectors (int8)

## Contributing

We welcome contributions! Key areas:

- Additional memory patterns
- Performance optimizations
- Integration examples
- Benchmark improvements

## License

MIT

---

**"Intelligence is memory plus judgment. AgentDB is where judgment learns to remember."**
