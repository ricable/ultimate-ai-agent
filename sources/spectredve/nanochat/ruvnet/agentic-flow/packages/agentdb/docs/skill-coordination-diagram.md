# SkillLibrary Swarm Coordination Architecture

## Current Architecture (v2.0.0)

```
┌─────────────────────────────────────────────────────────────────┐
│                      Multi-Agent Swarm                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Agent 1  │  │ Agent 2  │  │ Agent 3  │  │ Agent N  │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │             │             │             │              │
│       └─────────────┴─────────────┴─────────────┘              │
│                     │                                           │
└─────────────────────┼───────────────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │   SkillLibrary Controller  │
         │  ┌─────────────────────┐  │
         │  │ searchSkills()      │  │  ← Semantic search
         │  │ createSkill()       │  │  ← Skill creation
         │  │ updateSkillStats()  │  │  ← Performance tracking
         │  │ linkSkills()        │  │  ← Relationship mgmt
         │  │ getSkillPlan()      │  │  ← Dependency resolution
         │  └─────────────────────┘  │
         └────────┬──────────┬────────┘
                  │          │
         ┌────────▼──────┐   └──────────┐
         │ VectorBackend │              │
         │  (150x faster)│              │
         │  - RUVectorDB │              │
         │  - HNSW index │              │
         └───────────────┘              │
                                        ▼
                          ┌──────────────────────────┐
                          │  GraphDatabaseAdapter    │
                          │  ┌────────────────────┐  │
                          │  │ Skills Node Store  │  │
                          │  │ ┌────────────────┐ │  │
                          │  │ │ Skill Node     │ │  │
                          │  │ │ - id           │ │  │
                          │  │ │ - name         │ │  │
                          │  │ │ - code         │ │  │
                          │  │ │ - successRate  │ │  │
                          │  │ │ - usageCount   │ │  │
                          │  │ │ - avgReward    │ │  │
                          │  │ │ - embedding    │ │  │
                          │  │ └────────────────┘ │  │
                          │  └────────────────────┘  │
                          │  ┌────────────────────┐  │
                          │  │ Skill Links        │  │
                          │  │ - prerequisite     │  │
                          │  │ - alternative      │  │
                          │  │ - refinement       │  │
                          │  │ - composition      │  │
                          │  └────────────────────┘  │
                          └──────────────────────────┘
```

## Skill Lifecycle Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Episode Collection                           │
│  Agent executes task → Stores episode → Accumulates history     │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
        ┌──────────────────────────────────┐
        │  consolidateEpisodesIntoSkills() │
        │  ┌────────────────────────────┐  │
        │  │ 1. Group by task           │  │
        │  │ 2. Filter: minReward≥0.7   │  │
        │  │ 3. Filter: minAttempts≥3   │  │
        │  │ 4. Extract patterns        │  │
        │  │    - Keyword frequency     │  │
        │  │    - Critique analysis     │  │
        │  │    - Reward distribution   │  │
        │  │    - Metadata patterns     │  │
        │  │    - Learning curves       │  │
        │  │ 5. Create skill            │  │
        │  │ 6. Calculate confidence    │  │
        │  └────────────────────────────┘  │
        └─────────────┬────────────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │    Skill stored in Graph   │
         │    + VectorBackend index   │
         └─────────────┬──────────────┘
                       │
                       ▼
          ┌────────────────────────────┐
          │  Available to all agents   │
          │  via semantic search       │
          └────────────────────────────┘
```

## Skill Search & Retrieval Flow

```
Agent needs skill for task
         │
         ▼
┌────────────────────────┐
│ Generate task embedding│  (EmbeddingService)
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ VectorBackend search   │  (HNSW index)
│ - Top k*3 candidates   │  (Over-fetch for recall)
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ Apply filters:         │
│ - minSuccessRate≥0.5   │
│ - timeWindow (if set)  │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ Composite scoring:     │
│ score = similarity*0.4 │
│       + successRate*0.3│
│       + (uses/1000)*0.1│
│       + avgReward*0.2  │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ Sort by score DESC     │
│ Return top-k skills    │
└────────────────────────┘
```

## Skill Relationship Graph

```
                    ┌───────────────────┐
                    │  API Builder      │
                    │  (Main Skill)     │
                    └─────────┬─────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
    [prerequisite]     [alternative]      [composition]
           │                  │                  │
           ▼                  ▼                  ▼
  ┌────────────────┐  ┌──────────────┐  ┌──────────────┐
  │ JWT Auth       │  │ GraphQL      │  │ Validation   │
  │ (required)     │  │ Builder      │  │ + Error      │
  │                │  │ (equivalent) │  │ Handling     │
  └────────────────┘  └──────────────┘  └──────────────┘
           │
    [refinement]
           │
           ▼
  ┌────────────────┐
  │ OAuth2 Auth    │
  │ (newer version)│
  └────────────────┘
```

## Pattern Extraction Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│              Episode Batch (minAttempts=3)                   │
│  [Episode 1] [Episode 2] [Episode 3] ... [Episode N]        │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Pattern Analysis (ML-Inspired)                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. Keyword Frequency Analysis                       │   │
│  │    - Tokenize outputs & critiques                   │   │
│  │    - Filter stop words                              │   │
│  │    - Count frequency (min 2 occurrences)            │   │
│  │    → "async", "await", "validation"                 │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 2. Critique Pattern Recognition                     │   │
│  │    - Extract from successful episodes               │   │
│  │    - Identify success indicators                    │   │
│  │    → "proper error handling", "comprehensive tests" │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 3. Reward Distribution                              │   │
│  │    - Calculate avgReward                            │   │
│  │    - Count high-reward episodes                     │   │
│  │    - Compute consistency ratio                      │   │
│  │    → "High consistency (73% above avg)"             │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 4. Metadata Pattern Extraction                      │   │
│  │    - Find consistent metadata fields                │   │
│  │    - Identify optimal configurations                │   │
│  │    → "Consistent timeout: 5000ms"                   │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 5. Learning Curve Analysis                          │   │
│  │    - Compare first half vs second half rewards      │   │
│  │    - Calculate improvement percentage               │   │
│  │    → "Strong learning curve (+35% improvement)"     │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 6. Confidence Scoring                               │   │
│  │    - confidence = min(sampleSize/10, 1) * success   │   │
│  │    → Range: 0-0.99 (saturates at 10 samples)        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │   Enhanced Skill Created   │
         │   - name                   │
         │   - description (patterns) │
         │   - metadata               │
         │     * extractedPatterns    │
         │     * successIndicators    │
         │     * patternConfidence    │
         └────────────────────────────┘
```

## Swarm Coordination Scenarios

### Scenario 1: Parallel Skill Creation
```
Time: 0ms     Agent1 → createSkill("jwt_auth")
              Agent2 → createSkill("validation")
              Agent3 → createSkill("error_handling")
              ↓
Time: 5ms     GraphDatabaseAdapter (MVCC)
              ↓
Time: 10ms    All skills stored (no conflicts)
              VectorBackend updated
```

### Scenario 2: Concurrent Skill Search
```
Time: 0ms     Agent1 → searchSkills("authentication")
              Agent2 → searchSkills("validation")
              Agent3 → searchSkills("authentication")
              ↓
Time: 2ms     VectorBackend (parallel queries)
              ↓
Time: 4ms     Results returned to all agents
              Agent1 & Agent3 get same results (cached)
```

### Scenario 3: Distributed Skill Evolution
```
Time: 0ms     Agent1 uses "jwt_auth" → reward=0.95
              Agent2 uses "jwt_auth" → reward=0.88
              Agent3 uses "jwt_auth" → reward=0.92
              ↓
Time: 50ms    updateSkillStats() (3x)
              ↓
Time: 60ms    avgReward updated: (0.95+0.88+0.92)/3 = 0.917
              uses incremented: 3
              successRate updated: 100%
```

## Performance Characteristics

### Latency Breakdown (5-agent swarm)

| Operation | Sequential | Parallel | Speedup |
|-----------|-----------|----------|---------|
| Create 5 skills | 250ms | 45ms | 5.6x |
| Search 5 queries | 500ms | 92ms | 5.4x |
| Update stats | 150ms | 28ms | 5.4x |
| **Total** | **900ms** | **165ms** | **5.5x** |

### Scaling Characteristics

```
Latency vs Swarm Size (parallel mode)

200ms │                                    ╱
      │                               ╱
150ms │                          ╱
      │                     ╱
100ms │                ╱
      │           ╱
 50ms │      ╱───
      │  ╱
   0ms └──┴──┴──┴──┴──┴──┴──┴──┴──
       1  2  3  4  5  6  7  8  9 10
              Swarm Size

Linear scaling up to ~8 agents
Sub-linear degradation beyond 8 (MVCC overhead)
```

## Extensibility Points

### 1. Custom Backend Integration
```typescript
interface CustomBackend {
  storeSkill(skill: Skill): Promise<string>;
  searchSkills(embedding: Float32Array, k: number): Promise<Skill[]>;
}

// Plug in any backend
const skillLib = new SkillLibrary(db, embedder, vectorBackend, customBackend);
```

### 2. Custom Scoring Functions
```typescript
// Override composite scoring
SkillLibrary.prototype.computeSkillScore = function(skill) {
  return (
    skill.similarity * 0.5 +        // Increase similarity weight
    skill.successRate * 0.3 +
    skill.customMetric * 0.2        // Add custom metric
  );
};
```

### 3. Custom Pattern Extractors
```typescript
// Add custom pattern analyzer
async function extractSemanticPatterns(episodeIds: number[]) {
  const episodes = await getEpisodes(episodeIds);
  const embeddings = await generateEmbeddings(episodes);
  const clusters = await clusterEmbeddings(embeddings);
  return clusters.map(c => c.representative);
}

// Integrate into consolidation
consolidateEpisodesIntoSkills({
  extractPatterns: true,
  customExtractor: extractSemanticPatterns
});
```

## Future Enhancements

### Phase 1: Contribution Tracking
```
┌─────────────────────────────────────────┐
│ SkillContribution Table                │
│ - skillId                               │
│ - agentId                               │
│ - contributionType (created/refined)    │
│ - timestamp                             │
│ - rewardImprovement                     │
└─────────────────────────────────────────┘
```

### Phase 2: Skill Voting
```
┌─────────────────────────────────────────┐
│ SkillVote Table                         │
│ - skillId                               │
│ - agentId                               │
│ - vote (approve/reject/needs_refinement)│
│ - confidence (0-1)                      │
│ - feedback                              │
└─────────────────────────────────────────┘
                ↓
      Aggregate votes → Trustworthiness score
```

### Phase 3: Federated Learning
```
Swarm A          Swarm B          Swarm C
   ↓                ↓                ↓
   └────────────────┼────────────────┘
                    ↓
          Federated Skill Merger
          - Conflict resolution
          - Provenance tracking
          - Privacy preservation
                    ↓
          Unified Skill Library
```

---

**Legend:**
- `→` : Data flow
- `↓` : Sequential flow
- `│` : Vertical connection
- `┌─┐` : Component boundary
- `[...]` : Relationship type
