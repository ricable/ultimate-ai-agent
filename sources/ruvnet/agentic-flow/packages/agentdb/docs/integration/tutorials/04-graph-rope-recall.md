# Graph-RoPE Recall Deep Dive

Master graph-enhanced rotary position encoding for building connected knowledge bases with contextual memory recall.

## What is Graph-RoPE?

Graph-RoPE combines two powerful techniques:

1. **Graph Structure**: Explicit relationships between memories
2. **Rotary Position Encoding (RoPE)**: Position-aware embeddings that preserve sequential context

This enables:
- **Contextual Recall**: Find memories through their connections
- **Path-Based Search**: Explore relationships across multiple hops
- **Position Awareness**: Maintain sequential ordering information
- **Efficient Traversal**: O(log N) graph queries with indexing

## Core Concepts

### Memory Graphs

Memories are nodes, relationships are edges:

```
Memory A ----0.9----> Memory B
    |                    |
   0.7                  0.8
    |                    |
    v                    v
Memory C <----0.6---- Memory D
```

Edge weights represent relationship strength (0-1).

### Rotary Position Encoding

RoPE preserves position information in the embedding space:

```typescript
// Standard embedding: Just semantic meaning
embedding = [0.1, 0.3, -0.2, ...]

// RoPE embedding: Semantic + positional
ropeEmbedding = rotate(embedding, position, theta)
```

The `theta` parameter controls how quickly position information decays with distance.

## Building Memory Graphs

### Basic Graph Construction

```typescript
import Database from 'better-sqlite3';
import { AttentionService } from '@agentic/agentdb';

const db = new Database('knowledge.db');
const attention = new AttentionService(db, {
  enableGraphRoPE: true,
  ropeTheta: 10000,      // Position decay parameter
  graphDensity: 0.1,     // Target 10% edge density
  maxGraphHops: 3        // Max traversal depth
});

function embed(text: string): Float32Array {
  // Your embedding model here
  return new Float32Array(1536);
}

// Create memories
const memories = [
  { id: 1, text: "JavaScript is a programming language", vector: embed("JavaScript language") },
  { id: 2, text: "Python is a programming language", vector: embed("Python language") },
  { id: 3, text: "React is a JavaScript framework", vector: embed("React JavaScript") },
  { id: 4, text: "Django is a Python framework", vector: embed("Django Python") },
  { id: 5, text: "TypeScript extends JavaScript", vector: embed("TypeScript JavaScript") }
];

// Build the initial graph
await attention.graphRoPE.buildMemoryGraph(
  memories.map(m => ({
    id: m.id,
    vector: m.vector,
    metadata: { text: m.text, type: 'fact' }
  }))
);

console.log('Memory graph constructed');

// Add explicit relationships
await attention.graphRoPE.addEdge(3, 1, 0.95);  // React -> JavaScript (strong)
await attention.graphRoPE.addEdge(4, 2, 0.95);  // Django -> Python (strong)
await attention.graphRoPE.addEdge(5, 1, 0.90);  // TypeScript -> JavaScript (strong)
await attention.graphRoPE.addEdge(1, 2, 0.30);  // JavaScript -> Python (weak)

const stats = attention.graphRoPE.getGraphStats();
console.log(`Graph: ${stats.nodeCount} nodes, ${stats.edgeCount} edges`);
console.log(`Density: ${(stats.density * 100).toFixed(1)}%`);
console.log(`Avg degree: ${stats.avgDegree.toFixed(2)}`);
```

### Graph-Aware Search

```typescript
// Search with graph context
const query = embed("What frameworks are available?");

const results = await attention.graphRoPE.graphAwareSearch(
  query,
  10,   // Top 10 results
  2     // Explore up to 2 hops
);

results.forEach((result, i) => {
  console.log(`\n${i + 1}. ${result.metadata.text}`);
  console.log(`   RoPE score: ${result.ropeScore.toFixed(4)}`);
  console.log(`   Path length: ${result.pathLength} hops`);
  console.log(`   Connected to: ${result.connectedIds.length} memories`);

  if (result.connectedIds.length > 0) {
    console.log(`   Connections: ${result.connectedIds.slice(0, 3).join(', ')}`);
  }
});
```

Expected output:
```
1. React is a JavaScript framework
   RoPE score: 0.9234
   Path length: 0 hops
   Connected to: 1 memories
   Connections: 1

2. Django is a Python framework
   RoPE score: 0.9112
   Path length: 0 hops
   Connected to: 1 memories
   Connections: 2

3. JavaScript is a programming language
   RoPE score: 0.8756
   Path length: 1 hops
   Connected to: 3 memories
   Connections: 3, 5, 2
```

## Advanced Patterns

### 1. Knowledge Graph Construction

```typescript
// Build a comprehensive knowledge graph
interface KnowledgeEntry {
  concept: string;
  definition: string;
  relatedConcepts: Array<{ concept: string; relationship: string; strength: number }>;
}

const knowledge: KnowledgeEntry[] = [
  {
    concept: "Machine Learning",
    definition: "AI systems that learn from data",
    relatedConcepts: [
      { concept: "Neural Networks", relationship: "implements", strength: 0.9 },
      { concept: "Data Science", relationship: "part-of", strength: 0.8 },
      { concept: "Artificial Intelligence", relationship: "subset-of", strength: 0.95 }
    ]
  },
  {
    concept: "Neural Networks",
    definition: "Computing systems inspired by biological neural networks",
    relatedConcepts: [
      { concept: "Deep Learning", relationship: "enables", strength: 0.95 },
      { concept: "Backpropagation", relationship: "uses", strength: 0.85 },
      { concept: "Machine Learning", relationship: "used-in", strength: 0.9 }
    ]
  },
  {
    concept: "Deep Learning",
    definition: "ML with multiple layers of neural networks",
    relatedConcepts: [
      { concept: "Neural Networks", relationship: "uses", strength: 0.95 },
      { concept: "Computer Vision", relationship: "powers", strength: 0.85 },
      { concept: "NLP", relationship: "powers", strength: 0.85 }
    ]
  }
];

// Create concept map
const conceptMap = new Map<string, number>();
const memories = [];

for (const entry of knowledge) {
  const id = memories.length + 1;
  conceptMap.set(entry.concept, id);

  memories.push({
    id,
    vector: embed(`${entry.concept}: ${entry.definition}`),
    metadata: {
      concept: entry.concept,
      definition: entry.definition,
      type: 'knowledge'
    }
  });
}

// Build graph
await attention.graphRoPE.buildMemoryGraph(memories);

// Add relationships
for (const entry of knowledge) {
  const sourceId = conceptMap.get(entry.concept)!;

  for (const related of entry.relatedConcepts) {
    const targetId = conceptMap.get(related.concept);
    if (targetId) {
      await attention.graphRoPE.addEdge(
        sourceId,
        targetId,
        related.strength
      );

      console.log(`${entry.concept} --[${related.relationship}]--> ${related.concept} (${related.strength})`);
    }
  }
}

// Query with graph traversal
const query = embed("How does deep learning work?");
const results = await attention.graphRoPE.graphAwareSearch(query, 5, 3);

console.log('\nKnowledge path:');
results.forEach(r => {
  console.log(`${r.metadata.concept}: ${r.metadata.definition}`);
  console.log(`  Via ${r.pathLength} hop(s), score: ${r.ropeScore.toFixed(4)}`);
});
```

### 2. Citation and Reference Networks

```typescript
// Build academic citation network
interface Paper {
  id: number;
  title: string;
  abstract: string;
  references: number[];  // IDs of cited papers
  year: number;
}

const papers: Paper[] = [
  {
    id: 1,
    title: "Attention Is All You Need",
    abstract: "We propose the Transformer, a model architecture...",
    references: [],
    year: 2017
  },
  {
    id: 2,
    title: "BERT: Pre-training of Deep Bidirectional Transformers",
    abstract: "We introduce BERT, a new language representation model...",
    references: [1],
    year: 2018
  },
  {
    id: 3,
    title: "GPT-3: Language Models are Few-Shot Learners",
    abstract: "We train GPT-3, an autoregressive language model...",
    references: [1, 2],
    year: 2020
  }
];

// Create paper memories
const paperMemories = papers.map(paper => ({
  id: paper.id,
  vector: embed(`${paper.title} ${paper.abstract}`),
  metadata: {
    title: paper.title,
    abstract: paper.abstract,
    year: paper.year,
    type: 'paper'
  }
}));

await attention.graphRoPE.buildMemoryGraph(paperMemories);

// Add citation edges
for (const paper of papers) {
  for (const refId of paper.references) {
    // Citation strength based on recency
    const citedPaper = papers.find(p => p.id === refId)!;
    const yearGap = paper.year - citedPaper.year;
    const strength = Math.max(0.5, 1.0 - (yearGap * 0.1));

    await attention.graphRoPE.addEdge(paper.id, refId, strength);
    console.log(`"${paper.title}" cites "${citedPaper.title}" (strength: ${strength.toFixed(2)})`);
  }
}

// Find related papers through citations
const query = embed("Transformer architecture for NLP");
const related = await attention.graphRoPE.graphAwareSearch(query, 5, 2);

console.log('\nRelated papers via citations:');
related.forEach(r => {
  console.log(`${r.metadata.title} (${r.metadata.year})`);
  console.log(`  Path: ${r.pathLength} citations, Score: ${r.ropeScore.toFixed(4)}`);
});
```

### 3. Conversational Context Graph

```typescript
// Build conversation flow graph
interface Message {
  id: number;
  speaker: 'user' | 'assistant';
  text: string;
  replyTo?: number;  // ID of message being replied to
  topic?: string;
}

const conversation: Message[] = [
  { id: 1, speaker: 'user', text: "What is React?", topic: 'react' },
  { id: 2, speaker: 'assistant', text: "React is a JavaScript library for building UIs", replyTo: 1, topic: 'react' },
  { id: 3, speaker: 'user', text: "How do I create components?", replyTo: 2, topic: 'react' },
  { id: 4, speaker: 'assistant', text: "You can create components using functions or classes", replyTo: 3, topic: 'react' },
  { id: 5, speaker: 'user', text: "What about state management?", replyTo: 4, topic: 'state' },
  { id: 6, speaker: 'assistant', text: "React provides useState hook for state", replyTo: 5, topic: 'state' }
];

// Create message memories with positional encoding
const messageMemories = conversation.map((msg, index) => ({
  id: msg.id,
  vector: embed(msg.text),
  metadata: {
    ...msg,
    position: index  // Sequential position for RoPE
  }
}));

await attention.graphRoPE.buildMemoryGraph(messageMemories);

// Add reply-to edges
for (const msg of conversation) {
  if (msg.replyTo) {
    await attention.graphRoPE.addEdge(
      msg.id,
      msg.replyTo,
      0.9  // Strong reply relationship
    );
  }
}

// Add topic-based edges
for (let i = 0; i < conversation.length; i++) {
  for (let j = i + 1; j < conversation.length; j++) {
    if (conversation[i].topic === conversation[j].topic) {
      const distance = j - i;
      const strength = Math.max(0.3, 1.0 - (distance * 0.1));

      await attention.graphRoPE.addEdge(
        conversation[j].id,
        conversation[i].id,
        strength
      );
    }
  }
}

// Retrieve conversation context
const query = embed("How do I manage component state?");
const context = await attention.graphRoPE.graphAwareSearch(query, 5, 3);

console.log('Conversation context:');
context.forEach(c => {
  console.log(`[${c.metadata.speaker}] ${c.metadata.text}`);
  console.log(`  Position: ${c.metadata.position}, Path: ${c.pathLength} hops`);
});
```

### 4. Hierarchical Concept Maps

```typescript
// Build multi-level concept hierarchy with cross-links
interface Concept {
  id: number;
  name: string;
  level: number;  // 0=domain, 1=category, 2=concept, 3=detail
  parent?: number;
  relatedTo?: Array<{ id: number; type: string }>;
}

const concepts: Concept[] = [
  // Level 0: Domain
  { id: 1, name: "Programming", level: 0 },

  // Level 1: Categories
  { id: 2, name: "Web Development", level: 1, parent: 1 },
  { id: 3, name: "Data Science", level: 1, parent: 1 },

  // Level 2: Concepts
  { id: 4, name: "Frontend Frameworks", level: 2, parent: 2,
    relatedTo: [{ id: 7, type: 'uses' }] },
  { id: 5, name: "Backend Frameworks", level: 2, parent: 2 },
  { id: 6, name: "Machine Learning", level: 2, parent: 3 },

  // Level 3: Details
  { id: 7, name: "React", level: 3, parent: 4,
    relatedTo: [{ id: 8, type: 'competes-with' }] },
  { id: 8, name: "Vue", level: 3, parent: 4,
    relatedTo: [{ id: 7, type: 'competes-with' }] },
  { id: 9, name: "Express", level: 3, parent: 5 },
  { id: 10, name: "TensorFlow", level: 3, parent: 6 }
];

const conceptMemories = concepts.map(c => ({
  id: c.id,
  vector: embed(c.name),
  metadata: { ...c, type: 'concept' }
}));

await attention.graphRoPE.buildMemoryGraph(conceptMemories);

// Add hierarchical edges (parent-child)
for (const concept of concepts) {
  if (concept.parent) {
    await attention.graphRoPE.addEdge(
      concept.id,
      concept.parent,
      0.95  // Strong hierarchical link
    );
  }

  // Add cross-hierarchy edges
  if (concept.relatedTo) {
    for (const related of concept.relatedTo) {
      const strength = related.type === 'uses' ? 0.7 : 0.5;
      await attention.graphRoPE.addEdge(concept.id, related.id, strength);
    }
  }
}

// Multi-hop exploration
async function exploreConcept(conceptId: number, hops: number = 2) {
  const concept = concepts.find(c => c.id === conceptId)!;
  const query = embed(concept.name);

  const results = await attention.graphRoPE.graphAwareSearch(query, 10, hops);

  console.log(`\nExploring: ${concept.name}`);
  console.log('Connected concepts:');

  const byLevel = new Map<number, typeof results>();
  results.forEach(r => {
    if (!byLevel.has(r.metadata.level)) {
      byLevel.set(r.metadata.level, []);
    }
    byLevel.get(r.metadata.level)!.push(r);
  });

  for (const [level, items] of byLevel) {
    console.log(`\n  Level ${level}:`);
    items.forEach(item => {
      console.log(`    - ${item.metadata.name} (${item.pathLength} hops, ${item.ropeScore.toFixed(3)})`);
    });
  }
}

await exploreConcept(7, 3);  // Explore React with 3 hops
```

## Path Finding and Traversal

### Shortest Path Between Memories

```typescript
// Find shortest path between two concepts
async function findPath(
  fromId: number,
  toId: number,
  maxHops: number = 5
): Promise<number[]> {
  // BFS to find shortest path
  const queue: Array<{ id: number; path: number[] }> = [{ id: fromId, path: [fromId] }];
  const visited = new Set<number>([fromId]);

  while (queue.length > 0) {
    const { id, path } = queue.shift()!;

    if (id === toId) {
      return path;
    }

    if (path.length > maxHops) {
      continue;
    }

    // Get connected nodes
    const edges = await db.all(
      'SELECT target_id FROM graph_edges WHERE source_id = ?',
      id
    );

    for (const edge of edges) {
      if (!visited.has(edge.target_id)) {
        visited.add(edge.target_id);
        queue.push({
          id: edge.target_id,
          path: [...path, edge.target_id]
        });
      }
    }
  }

  return [];  // No path found
}

// Example: Find path from React to TensorFlow
const path = await findPath(7, 10);  // React -> ... -> TensorFlow

console.log('Path found:');
for (let i = 0; i < path.length; i++) {
  const concept = concepts.find(c => c.id === path[i])!;
  const arrow = i < path.length - 1 ? ' -> ' : '';
  console.log(`${concept.name}${arrow}`);
}
// Output: React -> Frontend Frameworks -> Web Development -> Programming -> Data Science -> Machine Learning -> TensorFlow
```

### Community Detection

```typescript
// Find clusters of related memories
async function detectCommunities(minClusterSize: number = 3) {
  const stats = attention.graphRoPE.getGraphStats();
  const communities: number[][] = [];

  // Simple community detection: connected components
  const visited = new Set<number>();

  for (let nodeId = 1; nodeId <= stats.nodeCount; nodeId++) {
    if (visited.has(nodeId)) continue;

    const community = await exploreCommunity(nodeId, visited);

    if (community.length >= minClusterSize) {
      communities.push(community);
    }
  }

  return communities;
}

async function exploreCommunity(
  startId: number,
  visited: Set<number>
): Promise<number[]> {
  const community = [];
  const queue = [startId];

  while (queue.length > 0) {
    const id = queue.shift()!;

    if (visited.has(id)) continue;

    visited.add(id);
    community.push(id);

    // Get neighbors
    const edges = await db.all(
      'SELECT target_id FROM graph_edges WHERE source_id = ? AND weight > 0.5',
      id
    );

    for (const edge of edges) {
      if (!visited.has(edge.target_id)) {
        queue.push(edge.target_id);
      }
    }
  }

  return community;
}

// Detect and display communities
const communities = await detectCommunities(2);

console.log(`Found ${communities.length} communities:`);
communities.forEach((community, i) => {
  console.log(`\nCommunity ${i + 1} (${community.length} members):`);
  community.forEach(id => {
    const concept = concepts.find(c => c.id === id);
    if (concept) {
      console.log(`  - ${concept.name}`);
    }
  });
});
```

## Performance Optimization

### Graph Indexing

```typescript
// Create indexes for fast graph traversal
db.exec(`
  CREATE INDEX IF NOT EXISTS idx_graph_source
  ON graph_edges(source_id, weight DESC);

  CREATE INDEX IF NOT EXISTS idx_graph_target
  ON graph_edges(target_id);

  CREATE INDEX IF NOT EXISTS idx_graph_weight
  ON graph_edges(weight DESC);

  CREATE INDEX IF NOT EXISTS idx_memory_metadata
  ON graph_memories(json_extract(metadata, '$.type'));
`);

console.log('Graph indexes created');
```

### Batch Edge Creation

```typescript
// Efficiently add many edges
async function batchAddEdges(edges: Array<{
  source: number;
  target: number;
  weight: number;
}>) {
  const transaction = db.transaction((edges) => {
    for (const edge of edges) {
      attention.graphRoPE.addEdge(edge.source, edge.target, edge.weight);
    }
  });

  transaction(edges);
  console.log(`Added ${edges.length} edges in batch`);
}

// Example: Add 1000 edges efficiently
const edges = [];
for (let i = 1; i <= 1000; i++) {
  edges.push({
    source: Math.floor(Math.random() * 100) + 1,
    target: Math.floor(Math.random() * 100) + 1,
    weight: Math.random()
  });
}

await batchAddEdges(edges);
```

## Statistics and Analytics

### Graph Metrics

```typescript
const stats = attention.graphRoPE.getGraphStats();

console.log('Graph Statistics:');
console.log(`Nodes: ${stats.nodeCount}`);
console.log(`Edges: ${stats.edgeCount}`);
console.log(`Density: ${(stats.density * 100).toFixed(2)}%`);
console.log(`Average degree: ${stats.avgDegree.toFixed(2)}`);
console.log(`Max degree: ${stats.maxDegree}`);

console.log('\nDegree distribution:');
stats.degreeDistribution.forEach((count, degree) => {
  const bar = '█'.repeat(Math.min(50, count));
  console.log(`${degree.toString().padStart(3)}: ${bar} (${count})`);
});

console.log('\nConnectivity:');
console.log(`Connected components: ${stats.componentCount}`);
console.log(`Largest component: ${stats.largestComponentSize} nodes`);
console.log(`Average clustering coefficient: ${stats.avgClustering.toFixed(4)}`);
```

## Best Practices

### 1. Edge Weight Selection

✅ **Good:**
```typescript
// Meaningful weights based on relationship type
await attention.graphRoPE.addEdge(id1, id2, 0.95);  // Direct citation
await attention.graphRoPE.addEdge(id1, id3, 0.70);  // Mentioned in text
await attention.graphRoPE.addEdge(id1, id4, 0.40);  // Weak association
```

❌ **Bad:**
```typescript
// All edges have same weight
await attention.graphRoPE.addEdge(id1, id2, 0.5);
await attention.graphRoPE.addEdge(id1, id3, 0.5);
```

### 2. Graph Density

✅ **Good:**
```typescript
// Sparse graph with meaningful connections
attention.enableFeatures({ graphDensity: 0.1 });  // 10% density
```

❌ **Bad:**
```typescript
// Overly dense graph (slow traversal)
attention.enableFeatures({ graphDensity: 0.9 });  // 90% density
```

### 3. Hop Limits

✅ **Good:**
```typescript
// Reasonable hop limit
const results = await graphAwareSearch(query, 10, 3);  // 3 hops
```

❌ **Bad:**
```typescript
// Excessive hops (exponential growth)
const results = await graphAwareSearch(query, 10, 10);  // 10 hops
```

## Summary

You've learned:
- ✅ Building memory graphs with RoPE encoding
- ✅ Graph-aware search and traversal
- ✅ Advanced patterns (citations, conversations, hierarchies)
- ✅ Path finding and community detection
- ✅ Performance optimization for large graphs
- ✅ Analytics and metrics

Next: [MoE Routing Tutorial](05-moe-routing.md)
