# AgentDB SDK Guide - Programmatic API

> Complete reference for using AgentDB in Node.js/TypeScript applications

This guide covers programmatic usage of AgentDB, including all frontier memory features with code examples.

---

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Core API](#core-api)
3. [Reflexion Memory API](#reflexion-memory-api)
4. [Skill Library API](#skill-library-api)
5. [Causal Memory API](#causal-memory-api)
6. [Recall API](#recall-api)
7. [Nightly Learner API](#nightly-learner-api)
8. [TypeScript Types](#typescript-types)
9. [Advanced Usage](#advanced-usage)

---

## Installation & Setup

### Installation

```bash
npm install agentdb
```

### Basic Initialization

```typescript
import { AgentDB } from 'agentdb';

// Initialize with default database path (./agentdb.db)
const db = new AgentDB();

// Or specify custom path
const db = new AgentDB('./my-agent.db');

// Initialize all components
await db.initialize();
```

### TypeScript Configuration

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "node",
    "esModuleInterop": true,
    "strict": true
  }
}
```

---

## Core API

### AgentDB Class

```typescript
import { AgentDB } from 'agentdb';

class AgentDB {
  constructor(dbPath?: string);

  // Initialize database and all controllers
  async initialize(): Promise<void>;

  // Access controllers
  reflexion: ReflexionMemory;
  skills: SkillLibrary;
  causal: CausalMemoryGraph;
  recall: CausalRecall;
  learner: NightlyLearner;

  // Database stats
  getStats(): DatabaseStats;

  // Close database
  close(): void;
}
```

### Complete Example

```typescript
import { AgentDB } from 'agentdb';

async function main() {
  // Initialize
  const agent = new AgentDB('./my-agent.db');
  await agent.initialize();

  // Store reflexion episode
  const episodeId = agent.reflexion.store({
    sessionId: 'session-1',
    task: 'implement_auth',
    reward: 0.95,
    success: true,
    critique: 'OAuth2 worked perfectly',
    input: 'Need secure authentication',
    output: 'Implemented OAuth2 with PKCE',
    latencyMs: 1200,
    tokens: 500
  });

  // Retrieve similar episodes
  const episodes = await agent.reflexion.retrieve({
    task: 'authentication',
    k: 10,
    minReward: 0.7
  });

  console.log(`Found ${episodes.length} similar episodes`);

  // Close when done
  agent.close();
}

main().catch(console.error);
```

---

## Reflexion Memory API

### Store Episode

```typescript
interface EpisodeData {
  sessionId: string;
  task: string;
  reward: number;        // 0-1
  success: boolean;
  critique?: string;
  input?: string;
  output?: string;
  latencyMs?: number;
  tokens?: number;
}

// Store episode
const episodeId = agent.reflexion.store({
  sessionId: 'debug-001',
  task: 'fix_auth_timeout',
  reward: 0.95,
  success: true,
  critique: 'Increasing JWT expiry from 1h to 24h fixed the issue',
  input: 'Users getting logged out randomly',
  output: 'Changed JWT_EXPIRY to 86400',
  latencyMs: 1800,
  tokens: 350
});

console.log(`Stored episode ${episodeId}`);
```

### Retrieve Episodes

```typescript
interface RetrieveParams {
  task: string;
  k?: number;
  minReward?: number;
  onlyFailures?: boolean;
  onlySuccesses?: boolean;
}

interface Episode {
  id: number;
  sessionId: string;
  task: string;
  reward: number;
  success: boolean;
  critique?: string;
  input?: string;
  output?: string;
  latencyMs?: number;
  tokens?: number;
  createdAt: number;
  embedding?: Float32Array;
  similarity?: number;
}

// Retrieve similar episodes
const episodes = await agent.reflexion.retrieve({
  task: 'authentication timeout',
  k: 10,
  minReward: 0.7
});

episodes.forEach(ep => {
  console.log(`Episode ${ep.id}: ${ep.task} (reward: ${ep.reward})`);
  console.log(`  Critique: ${ep.critique}`);
  console.log(`  Similarity: ${ep.similarity?.toFixed(3)}`);
});
```

### Critique Summary

```typescript
interface CritiqueSummary {
  task: string;
  totalEpisodes: number;
  successRate: number;
  avgReward: number;
  critiques: string[];
  commonPatterns: string[];
}

const summary = await agent.reflexion.getCritiqueSummary({
  task: 'authentication',
  onlyFailures: true  // Learn from failures
});

console.log(`Success Rate: ${summary.successRate.toFixed(2)}`);
console.log(`Lessons Learned:`);
summary.critiques.forEach((c, i) => {
  console.log(`  ${i + 1}. ${c}`);
});
```

### Prune Episodes

```typescript
const pruned = agent.reflexion.prune({
  maxAgeDays: 90,
  maxReward: 0.5
});

console.log(`Pruned ${pruned} episodes`);
```

---

## Skill Library API

### Create Skill

```typescript
interface SkillData {
  name: string;
  description: string;
  code?: string;
  schema?: object;
  version?: number;
}

const skillId = agent.skills.create({
  name: 'jwt_auth',
  description: 'Generate secure JWT tokens with 24h expiry',
  code: `
    const jwt = require('jsonwebtoken');

    function generateToken(payload, secret) {
      return jwt.sign(payload, secret, {
        expiresIn: '24h',
        algorithm: 'HS256'
      });
    }
  `,
  schema: {
    inputs: { payload: 'object', secret: 'string' },
    outputs: { token: 'string' }
  },
  version: 1
});

console.log(`Created skill ${skillId}`);
```

### Search Skills

```typescript
interface SearchParams {
  query: string;
  k?: number;
  minSuccessRate?: number;
}

interface Skill {
  id: number;
  name: string;
  description: string;
  code?: string;
  schema?: object;
  version: number;
  uses: number;
  successRate: number;
  avgReward: number;
  avgLatency: number;
  similarity?: number;
}

const skills = await agent.skills.search({
  query: 'authentication security',
  k: 5,
  minSuccessRate: 0.5
});

skills.forEach(skill => {
  console.log(`${skill.name}: ${skill.description}`);
  console.log(`  Success: ${(skill.successRate * 100).toFixed(1)}%`);
  console.log(`  Uses: ${skill.uses}`);
  console.log(`  Similarity: ${skill.similarity?.toFixed(3)}`);
});
```

### Update Skill Stats

```typescript
// Record skill usage
agent.skills.recordUsage({
  skillId: 1,
  success: true,
  reward: 0.92,
  latencyMs: 150,
  sessionId: 'session-1'
});
```

### Auto-Consolidate

```typescript
interface ConsolidateParams {
  minAttempts?: number;
  minReward?: number;
  timeWindowDays?: number;
}

const newSkills = await agent.skills.consolidate({
  minAttempts: 3,
  minReward: 0.7,
  timeWindowDays: 7
});

console.log(`Created ${newSkills.length} new skills from patterns`);
newSkills.forEach(skill => {
  console.log(`  - ${skill.name}: ${skill.description}`);
});
```

### Prune Skills

```typescript
const pruned = agent.skills.prune({
  minUses: 3,
  minSuccessRate: 0.4,
  maxAgeDays: 60
});

console.log(`Pruned ${pruned} underperforming skills`);
```

---

## Causal Memory API

### Add Causal Edge

```typescript
interface CausalEdge {
  fromMemoryId: number;
  fromMemoryType: 'episode' | 'skill' | 'note' | 'fact';
  toMemoryId: number;
  toMemoryType: 'episode' | 'skill' | 'note' | 'fact';
  similarity: number;
  uplift?: number;
  confidence: number;
  sampleSize?: number;
  mechanism?: string;
  metadata?: Record<string, any>;
}

const edgeId = agent.causal.addCausalEdge({
  fromMemoryId: 0,
  fromMemoryType: 'episode',
  toMemoryId: 0,
  toMemoryType: 'episode',
  similarity: 0.0,
  uplift: 0.25,
  confidence: 0.95,
  sampleSize: 100,
  mechanism: 'Adding tests improves code quality through better coverage',
  metadata: { experiment: 'test_coverage_study' }
});
```

### Query Causal Effects

```typescript
interface CausalQuery {
  interventionMemoryId: number;
  interventionMemoryType: string;
  outcomeMemoryId?: number;
  minConfidence?: number;
  minUplift?: number;
}

const edges = agent.causal.queryCausalEffects({
  interventionMemoryId: 0,
  interventionMemoryType: 'add_tests',
  minConfidence: 0.7,
  minUplift: 0.1
});

edges.forEach(edge => {
  console.log(`${edge.fromMemoryType} → ${edge.toMemoryType}`);
  console.log(`  Uplift: ${edge.uplift?.toFixed(3)}`);
  console.log(`  Confidence: ${edge.confidence.toFixed(2)}`);
  console.log(`  Sample Size: ${edge.sampleSize}`);
});
```

### A/B Experiments

**Create Experiment:**

```typescript
interface CausalExperiment {
  name: string;
  hypothesis: string;
  treatmentId: number;
  treatmentType: string;
  controlId?: number;
  startTime: number;
  sampleSize: number;
  status: 'running' | 'completed' | 'failed';
  metadata?: Record<string, any>;
}

const experimentId = agent.causal.createExperiment({
  name: 'test_coverage_quality',
  hypothesis: 'Does adding tests causally affect code quality?',
  treatmentId: 0,
  treatmentType: 'add_tests',
  startTime: Math.floor(Date.now() / 1000),
  sampleSize: 0,
  status: 'running',
  metadata: { effect: 'code_quality' }
});
```

**Record Observations:**

```typescript
interface CausalObservation {
  experimentId: number;
  episodeId: number;
  isTreatment: boolean;
  outcomeValue: number;
  outcomeType: 'reward' | 'success' | 'latency';
  context?: Record<string, any>;
}

// Treatment group
agent.causal.recordObservation({
  experimentId: 1,
  episodeId: 42,
  isTreatment: true,
  outcomeValue: 0.85,
  outcomeType: 'reward'
});

// Control group
agent.causal.recordObservation({
  experimentId: 1,
  episodeId: 43,
  isTreatment: false,
  outcomeValue: 0.65,
  outcomeType: 'reward'
});
```

**Calculate Uplift:**

```typescript
interface UpliftResult {
  uplift: number;
  pValue: number;
  confidenceInterval: [number, number];
}

const result = agent.causal.calculateUplift(experimentId);

console.log(`Uplift: ${result.uplift.toFixed(3)}`);
console.log(`95% CI: [${result.confidenceInterval[0].toFixed(3)}, ${result.confidenceInterval[1].toFixed(3)}]`);
console.log(`p-value: ${result.pValue.toFixed(4)}`);

if (result.pValue < 0.05) {
  console.log('✅ Statistically significant!');
} else {
  console.log('⚠️ Not statistically significant');
}
```

---

## Recall API

### Causal Recall with Certificate

```typescript
interface RecallParams {
  query: string;
  k?: number;
  alpha?: number;  // Similarity weight (default: 0.7)
  beta?: number;   // Uplift weight (default: 0.2)
  gamma?: number;  // Latency penalty (default: 0.1)
}

interface CausalRecallResult {
  query: string;
  candidates: RecallCandidate[];
  certificate: ProvenanceCertificate;
  latencyMs: number;
}

interface RecallCandidate {
  id: number;
  type: string;
  content: string;
  similarity: number;
  uplift?: number;
  utility: number;
  latency?: number;
}

interface ProvenanceCertificate {
  certificateId: string;
  queryHash: string;
  merkleRoot: string;
  completeness: number;
  timestamp: number;
  parameters: {
    k: number;
    alpha: number;
    beta: number;
    gamma: number;
  };
  retrievedIds: number[];
}

// Perform recall
const result = await agent.recall.retrieveWithCertificate({
  query: 'successful API optimization',
  k: 5,
  alpha: 0.7,
  beta: 0.2,
  gamma: 0.1
});

console.log(`Found ${result.candidates.length} results`);
result.candidates.forEach((c, i) => {
  console.log(`#${i + 1}: ${c.type} ${c.id}`);
  console.log(`  Similarity: ${c.similarity.toFixed(3)}`);
  console.log(`  Uplift: ${c.uplift?.toFixed(3) || 'N/A'}`);
  console.log(`  Utility: ${c.utility.toFixed(3)}`);
});

console.log(`\nCertificate ID: ${result.certificate.certificateId}`);
console.log(`Completeness: ${result.certificate.completeness.toFixed(2)}`);
console.log(`Latency: ${result.latencyMs}ms`);
```

### Verify Certificate

```typescript
const isValid = agent.recall.verifyCertificate(result.certificate);

if (isValid) {
  console.log('✅ Certificate is valid');
} else {
  console.log('❌ Certificate verification failed');
}
```

---

## Nightly Learner API

### Discover Patterns

```typescript
interface LearnerConfig {
  minAttempts?: number;
  minSuccessRate?: number;
  minConfidence?: number;
  dryRun?: boolean;
}

interface DiscoveredPattern {
  cause: string;
  effect: string;
  uplift: number;
  pValue: number;
  sampleSize: number;
  confidence: number;
}

// Dry-run to see what would be discovered
const dryRunResults = await agent.learner.discoverPatterns({
  minAttempts: 3,
  minSuccessRate: 0.6,
  minConfidence: 0.7,
  dryRun: true
});

console.log(`Would discover ${dryRunResults.length} patterns:`);
dryRunResults.forEach(p => {
  console.log(`  ${p.cause} → ${p.effect} (uplift: ${p.uplift.toFixed(3)}, p=${p.pValue.toFixed(4)})`);
});

// Actually discover and create
const patterns = await agent.learner.discoverPatterns({
  minAttempts: 3,
  minSuccessRate: 0.6,
  minConfidence: 0.7,
  dryRun: false
});

console.log(`Discovered ${patterns.length} causal edges`);
```

### Prune Low-Quality Edges

```typescript
const pruned = agent.learner.pruneEdges({
  minConfidence: 0.5,
  minUplift: 0.05,
  maxAgeDays: 90
});

console.log(`Pruned ${pruned} low-quality edges`);
```

---

## TypeScript Types

### Complete Type Definitions

```typescript
// Episode types
interface Episode {
  id?: number;
  sessionId: string;
  task: string;
  reward: number;
  success: boolean;
  critique?: string;
  input?: string;
  output?: string;
  latencyMs?: number;
  tokens?: number;
  createdAt?: number;
  embedding?: Float32Array;
}

// Skill types
interface Skill {
  id?: number;
  name: string;
  description: string;
  code?: string;
  schema?: object;
  version: number;
  uses: number;
  successRate: number;
  avgReward: number;
  avgLatency: number;
  createdAt?: number;
  updatedAt?: number;
}

// Causal types
interface CausalEdge {
  id?: number;
  fromMemoryId: number;
  fromMemoryType: 'episode' | 'skill' | 'note' | 'fact';
  toMemoryId: number;
  toMemoryType: 'episode' | 'skill' | 'note' | 'fact';
  similarity: number;
  uplift?: number;
  confidence: number;
  sampleSize?: number;
  evidenceIds?: string[];
  experimentIds?: string[];
  confounderScore?: number;
  mechanism?: string;
  metadata?: Record<string, any>;
}

interface CausalExperiment {
  id?: number;
  name: string;
  hypothesis: string;
  treatmentId: number;
  treatmentType: string;
  controlId?: number;
  startTime: number;
  endTime?: number;
  sampleSize: number;
  treatmentMean?: number;
  controlMean?: number;
  uplift?: number;
  pValue?: number;
  confidenceIntervalLow?: number;
  confidenceIntervalHigh?: number;
  status: 'running' | 'completed' | 'failed';
  metadata?: Record<string, any>;
}

// Recall types
interface CausalRecallResult {
  query: string;
  candidates: RecallCandidate[];
  certificate: ProvenanceCertificate;
  latencyMs: number;
}

interface RecallCandidate {
  id: number;
  type: string;
  content: string;
  similarity: number;
  uplift?: number;
  utility: number;
  latency?: number;
}

interface ProvenanceCertificate {
  certificateId: string;
  queryHash: string;
  merkleRoot: string;
  completeness: number;
  timestamp: number;
  parameters: {
    k: number;
    alpha: number;
    beta: number;
    gamma: number;
  };
  retrievedIds: number[];
}
```

---

## Advanced Usage

### Custom Embedding Service

```typescript
import { EmbeddingService } from 'agentdb';

// Create custom embedding service
const embeddings = new EmbeddingService();

// Generate embedding
const text = 'implement authentication';
const embedding = await embeddings.embed(text);

console.log(`Embedding dimension: ${embedding.length}`);
```

### Batch Operations

```typescript
// Batch store episodes
const episodes = [
  { sessionId: 's1', task: 'task1', reward: 0.9, success: true },
  { sessionId: 's2', task: 'task2', reward: 0.85, success: true },
  { sessionId: 's3', task: 'task3', reward: 0.92, success: true }
];

const ids = episodes.map(ep => agent.reflexion.store(ep));
console.log(`Stored ${ids.length} episodes`);

// Batch retrieve
const tasks = ['task1', 'task2', 'task3'];
const results = await Promise.all(
  tasks.map(task => agent.reflexion.retrieve({ task, k: 5 }))
);
```

### Event Handling

```typescript
// Listen for events (if implemented)
agent.on('episode:stored', (episode) => {
  console.log(`New episode: ${episode.task}`);
});

agent.on('skill:created', (skill) => {
  console.log(`New skill: ${skill.name}`);
});

agent.on('pattern:discovered', (pattern) => {
  console.log(`Discovered: ${pattern.cause} → ${pattern.effect}`);
});
```

### Database Stats

```typescript
interface DatabaseStats {
  episodes: number;
  skills: number;
  causalEdges: number;
  experiments: number;
  observations: number;
  certificates: number;
}

const stats = agent.getStats();

console.log('Database Statistics:');
console.log(`  Episodes: ${stats.episodes}`);
console.log(`  Skills: ${stats.skills}`);
console.log(`  Causal Edges: ${stats.causalEdges}`);
console.log(`  Experiments: ${stats.experiments}`);
console.log(`  Observations: ${stats.observations}`);
```

### Transaction Support

```typescript
// Begin transaction
agent.db.exec('BEGIN TRANSACTION');

try {
  // Perform multiple operations
  const ep1 = agent.reflexion.store({ /* ... */ });
  const ep2 = agent.reflexion.store({ /* ... */ });
  const skill = agent.skills.create({ /* ... */ });

  // Commit transaction
  agent.db.exec('COMMIT');
} catch (error) {
  // Rollback on error
  agent.db.exec('ROLLBACK');
  throw error;
}
```

### Export/Import Data

```typescript
// Export to JSON
const data = {
  episodes: agent.reflexion.getAll(),
  skills: agent.skills.getAll(),
  edges: agent.causal.getAllEdges(),
  experiments: agent.causal.getAllExperiments()
};

const json = JSON.stringify(data, null, 2);
await fs.writeFile('agent-memory.json', json);

// Import from JSON
const imported = JSON.parse(await fs.readFile('agent-memory.json', 'utf-8'));

imported.episodes.forEach(ep => agent.reflexion.store(ep));
imported.skills.forEach(skill => agent.skills.create(skill));
imported.edges.forEach(edge => agent.causal.addCausalEdge(edge));
```

---

## Complete Application Example

```typescript
import { AgentDB } from 'agentdb';

class IntelligentAgent {
  private db: AgentDB;

  constructor(dbPath: string = './agent.db') {
    this.db = new AgentDB(dbPath);
  }

  async initialize() {
    await this.db.initialize();
  }

  async executeTask(task: string, input: string): Promise<string> {
    const startTime = Date.now();

    // 1. Search for applicable skills
    const skills = await this.db.skills.search({
      query: task,
      k: 3,
      minSuccessRate: 0.7
    });

    console.log(`Found ${skills.length} applicable skills`);

    // 2. Retrieve similar past episodes
    const pastEpisodes = await this.db.reflexion.retrieve({
      task,
      k: 5,
      minReward: 0.7
    });

    console.log(`Found ${pastEpisodes.length} similar episodes`);

    // 3. Get causal insights
    const causalEdges = this.db.causal.queryCausalEffects({
      interventionMemoryId: 0,
      interventionMemoryType: task,
      minConfidence: 0.7
    });

    console.log(`Found ${causalEdges.length} causal relationships`);

    // 4. Execute task (your implementation)
    const output = await this.performTask(task, input, skills, pastEpisodes);
    const success = this.evaluateSuccess(output);
    const reward = this.calculateReward(output);

    // 5. Store episode
    const episodeId = this.db.reflexion.store({
      sessionId: `session-${Date.now()}`,
      task,
      reward,
      success,
      critique: this.generateCritique(output, success),
      input,
      output,
      latencyMs: Date.now() - startTime,
      tokens: this.countTokens(input + output)
    });

    console.log(`Stored episode ${episodeId}`);

    // 6. Update skill statistics if skill was used
    if (skills.length > 0) {
      this.db.skills.recordUsage({
        skillId: skills[0].id!,
        success,
        reward,
        latencyMs: Date.now() - startTime,
        sessionId: `session-${Date.now()}`
      });
    }

    return output;
  }

  async runNightlyLearning() {
    console.log('Running nightly pattern discovery...');

    // Discover patterns
    const patterns = await this.db.learner.discoverPatterns({
      minAttempts: 3,
      minSuccessRate: 0.6,
      minConfidence: 0.7
    });

    console.log(`Discovered ${patterns.length} patterns`);

    // Consolidate skills
    const newSkills = await this.db.skills.consolidate({
      minAttempts: 3,
      minReward: 0.7,
      timeWindowDays: 7
    });

    console.log(`Created ${newSkills.length} new skills`);

    // Prune old data
    const prunedEpisodes = this.db.reflexion.prune({
      maxAgeDays: 90,
      maxReward: 0.5
    });

    const prunedSkills = this.db.skills.prune({
      minUses: 3,
      minSuccessRate: 0.4,
      maxAgeDays: 60
    });

    const prunedEdges = this.db.learner.pruneEdges({
      minConfidence: 0.5,
      minUplift: 0.05,
      maxAgeDays: 90
    });

    console.log(`Pruned ${prunedEpisodes} episodes, ${prunedSkills} skills, ${prunedEdges} edges`);
  }

  private async performTask(
    task: string,
    input: string,
    skills: any[],
    pastEpisodes: any[]
  ): Promise<string> {
    // Your implementation here
    return 'Task output';
  }

  private evaluateSuccess(output: string): boolean {
    // Your evaluation logic
    return true;
  }

  private calculateReward(output: string): number {
    // Your reward calculation
    return 0.85;
  }

  private generateCritique(output: string, success: boolean): string {
    // Your critique generation
    return success ? 'Worked well' : 'Could be improved';
  }

  private countTokens(text: string): number {
    // Simple approximation
    return Math.ceil(text.length / 4);
  }

  close() {
    this.db.close();
  }
}

// Usage
async function main() {
  const agent = new IntelligentAgent('./my-agent.db');
  await agent.initialize();

  // Execute tasks
  await agent.executeTask('implement_auth', 'Need OAuth2 authentication');
  await agent.executeTask('optimize_query', 'Slow database queries');

  // Run nightly learning
  await agent.runNightlyLearning();

  agent.close();
}

main().catch(console.error);
```

---

## Error Handling

```typescript
import { AgentDB, AgentDBError } from 'agentdb';

try {
  const agent = new AgentDB('./agent.db');
  await agent.initialize();

  const episode = agent.reflexion.store({
    sessionId: 'test',
    task: 'test_task',
    reward: 0.9,
    success: true
  });

} catch (error) {
  if (error instanceof AgentDBError) {
    console.error(`AgentDB Error: ${error.message}`);
    console.error(`Code: ${error.code}`);
  } else {
    console.error(`Unexpected error: ${error}`);
  }
}
```

---

## Performance Tips

1. **Batch Operations**: Use transactions for multiple inserts
2. **Index Usage**: Queries use proper SQLite indexes
3. **WASM SIMD**: Vector operations are 150x faster
4. **Connection Pooling**: Reuse AgentDB instance
5. **Lazy Loading**: Only load embeddings when needed

---

## Next Steps

- See [CLI Guide](./CLI_GUIDE.md) for command-line usage
- See [Frontier Memory Guide](./FRONTIER_MEMORY_GUIDE.md) for concepts
- See [MCP Guide](./MCP.md) for Claude Desktop integration
- Check [examples/](../examples/) for more code samples

---

## Support

- **GitHub**: [anthropics/agentic-flow](https://github.com/anthropics/agentic-flow)
- **Issues**: [Report bugs](https://github.com/anthropics/agentic-flow/issues)
- **Discussions**: [Ask questions](https://github.com/anthropics/agentic-flow/discussions)
