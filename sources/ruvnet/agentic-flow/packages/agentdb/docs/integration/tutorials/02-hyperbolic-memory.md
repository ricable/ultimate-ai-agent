# Hyperbolic Memory Deep Dive

Learn how to build hierarchical knowledge bases using hyperbolic geometry for efficient memory organization and retrieval.

## What is Hyperbolic Memory?

Hyperbolic memory uses hyperbolic geometry (negative curvature space) to organize information hierarchically. This allows:

- **Natural hierarchies**: Parent-child relationships with geometric meaning
- **Efficient search**: Logarithmic complexity for hierarchical queries
- **Semantic distance**: Distance reflects both similarity and hierarchy
- **Scalability**: Billions of items with constant-time access

## Core Concepts

### Hierarchy Depth

- **Depth 0**: Root concepts (categories, domains)
- **Depth 1**: Major topics (subcategories)
- **Depth 2+**: Specific details (facts, examples)

### Hyperbolic Distance

Combines semantic similarity with hierarchical position:

```
hyperbolic_distance = sqrt(euclidean_distance² + depth_difference²)
```

Memories at the same depth are compared semantically. Memories at different depths account for their hierarchical relationship.

## Building a Knowledge Hierarchy

### Example: Technical Documentation

```typescript
import Database from 'better-sqlite3';
import { AttentionService } from '@agentic/agentdb';

const db = new Database('knowledge.db');
const attention = new AttentionService(db, {
  enableHyperbolic: true,
  maxHierarchyDepth: 5,
  hyperbolicCurvature: -1.0
});

// Helper function (use real embeddings in production)
function embed(text: string): Float32Array {
  // Your embedding model here
  return new Float32Array(1536);
}

// Level 0: Programming languages
const jsId = await attention.hyperbolic.storeWithHierarchy(
  embed("JavaScript programming language"),
  { name: "JavaScript", type: "language" },
  0
);

const pythonId = await attention.hyperbolic.storeWithHierarchy(
  embed("Python programming language"),
  { name: "Python", type: "language" },
  0
);

// Level 1: Language features
const jsAsyncId = await attention.hyperbolic.storeWithHierarchy(
  embed("JavaScript asynchronous programming"),
  { name: "Async JavaScript", type: "feature", parent: jsId },
  1
);

const pythonAsyncId = await attention.hyperbolic.storeWithHierarchy(
  embed("Python asynchronous programming"),
  { name: "Async Python", type: "feature", parent: pythonId },
  1
);

// Level 2: Specific topics
const promisesId = await attention.hyperbolic.storeWithHierarchy(
  embed("JavaScript Promises for handling async operations"),
  {
    name: "Promises",
    type: "concept",
    parent: jsAsyncId,
    code: "const promise = new Promise((resolve, reject) => {...})"
  },
  2
);

const asyncAwaitId = await attention.hyperbolic.storeWithHierarchy(
  embed("JavaScript async/await syntax for cleaner async code"),
  {
    name: "Async/Await",
    type: "concept",
    parent: jsAsyncId,
    code: "async function fetchData() { await response; }"
  },
  2
);

console.log('Hierarchy built successfully!');
```

### Visualizing the Hierarchy

```
Depth 0: Languages
├── JavaScript (jsId)
│   └── Depth 1: Async JavaScript (jsAsyncId)
│       ├── Depth 2: Promises (promisesId)
│       └── Depth 2: Async/Await (asyncAwaitId)
└── Python (pythonId)
    └── Depth 1: Async Python (pythonAsyncId)
```

## Hierarchical Search

### Basic Search

```typescript
// Query: How do I handle async in JavaScript?
const query = embed("JavaScript asynchronous operations");

const results = await attention.hyperbolic.hierarchicalSearch(
  query,
  10,    // Top 10 results
  3      // Search up to depth 3
);

results.forEach(result => {
  const indent = '  '.repeat(result.depth);
  console.log(`${indent}${result.metadata.name}`);
  console.log(`${indent}Score: ${result.hyperbolicScore.toFixed(4)}`);
  console.log(`${indent}Depth: ${result.depth}`);
});
```

Expected output:
```
JavaScript
  Score: 0.8500
  Depth: 0
  Async JavaScript
    Score: 0.9200
    Depth: 1
    Async/Await
      Score: 0.9500
      Depth: 2
    Promises
      Score: 0.9300
      Depth: 2
```

### Depth-Limited Search

Search only specific hierarchy levels:

```typescript
// Search only root level (languages)
const rootResults = await attention.hyperbolic.hierarchicalSearch(
  embed("programming language"),
  5,
  0  // maxDepth = 0 (root only)
);

console.log('Root level only:');
rootResults.forEach(r => console.log(r.metadata.name));
// Output: JavaScript, Python

// Search only depth 1 and below (features and deeper)
const featureResults = await attention.hyperbolic.hierarchicalSearch(
  embed("async programming"),
  5,
  1  // maxDepth = 1
);

console.log('Features and language:');
featureResults.forEach(r => console.log(`${r.metadata.name} (depth ${r.depth})`));
// Output: Async JavaScript (depth 1), JavaScript (depth 0)
```

### Path-Based Search

Find entire paths from root to leaf:

```typescript
// Get full context path for a memory
async function getMemoryPath(memoryId: number): Promise<any[]> {
  const path = [];
  let currentId = memoryId;

  while (currentId) {
    const memory = await db.get(
      'SELECT * FROM hyperbolic_memory WHERE id = ?',
      currentId
    );
    path.unshift(memory);
    currentId = memory.metadata?.parent;
  }

  return path;
}

const path = await getMemoryPath(promisesId);
console.log('Full context path:');
path.forEach(node => {
  const indent = '  '.repeat(node.depth);
  console.log(`${indent}${node.metadata.name}`);
});
// Output:
// JavaScript
//   Async JavaScript
//     Promises
```

## Advanced Patterns

### 1. Multi-Language Knowledge Base

```typescript
// Store documentation for multiple languages
const languages = [
  { name: 'JavaScript', features: ['Promises', 'Async/Await', 'Generators'] },
  { name: 'Python', features: ['Async/Await', 'Generators', 'Coroutines'] },
  { name: 'Rust', features: ['Futures', 'Async/Await', 'Tokio'] }
];

for (const lang of languages) {
  // Root: Language
  const langId = await attention.hyperbolic.storeWithHierarchy(
    embed(`${lang.name} programming language`),
    { name: lang.name, type: 'language' },
    0
  );

  // Level 1: Features
  for (const feature of lang.features) {
    await attention.hyperbolic.storeWithHierarchy(
      embed(`${lang.name} ${feature}`),
      { name: feature, type: 'feature', language: lang.name, parent: langId },
      1
    );
  }
}

// Cross-language search
const results = await attention.hyperbolic.hierarchicalSearch(
  embed("async await syntax"),
  5
);

console.log('Async/await across languages:');
results.forEach(r => {
  console.log(`${r.metadata.language}: ${r.metadata.name}`);
});
// Output:
// JavaScript: Async/Await
// Python: Async/Await
// Rust: Async/Await
```

### 2. Document Hierarchy

```typescript
// Build document structure
const doc = {
  title: "Complete Guide to React",
  chapters: [
    {
      title: "Getting Started",
      sections: [
        { title: "Installation", content: "npm install react..." },
        { title: "First Component", content: "function App() {...}" }
      ]
    },
    {
      title: "Advanced Patterns",
      sections: [
        { title: "Custom Hooks", content: "function useCustomHook() {...}" },
        { title: "Context API", content: "const MyContext = createContext()..." }
      ]
    }
  ]
};

// Store hierarchically
const docId = await attention.hyperbolic.storeWithHierarchy(
  embed(doc.title),
  { title: doc.title, type: 'document' },
  0
);

for (const chapter of doc.chapters) {
  const chapterId = await attention.hyperbolic.storeWithHierarchy(
    embed(chapter.title),
    { title: chapter.title, type: 'chapter', parent: docId },
    1
  );

  for (const section of chapter.sections) {
    await attention.hyperbolic.storeWithHierarchy(
      embed(section.content),
      {
        title: section.title,
        content: section.content,
        type: 'section',
        parent: chapterId
      },
      2
    );
  }
}

// Search with document context
const results = await attention.hyperbolic.hierarchicalSearch(
  embed("How do I create custom hooks?"),
  5
);

results.forEach(result => {
  if (result.metadata.type === 'section') {
    console.log(`Found in: ${result.metadata.title}`);
    console.log(`Content: ${result.metadata.content.substring(0, 50)}...`);
  }
});
```

### 3. Taxonomic Classification

```typescript
// Build biological taxonomy
const taxonomy = {
  kingdom: 'Animalia',
  phylum: 'Chordata',
  class: 'Mammalia',
  order: 'Carnivora',
  family: 'Felidae',
  genus: 'Panthera',
  species: 'Panthera leo (Lion)'
};

let parentId = null;
let depth = 0;

for (const [rank, name] of Object.entries(taxonomy)) {
  const id = await attention.hyperbolic.storeWithHierarchy(
    embed(`${rank}: ${name}`),
    { rank, name, type: 'taxonomy', parent: parentId },
    depth
  );
  parentId = id;
  depth++;
}

// Find specific classification
const results = await attention.hyperbolic.hierarchicalSearch(
  embed("What family do lions belong to?"),
  5
);

results.forEach(r => {
  if (r.metadata.rank === 'family') {
    console.log(`Family: ${r.metadata.name}`);
  }
});
// Output: Family: Felidae
```

### 4. Product Catalog

```typescript
// E-commerce product hierarchy
const catalog = [
  {
    category: 'Electronics',
    subcategories: [
      {
        name: 'Computers',
        products: [
          { name: 'Laptop Pro 15"', price: 1299, specs: '16GB RAM, 512GB SSD' },
          { name: 'Desktop Workstation', price: 1999, specs: '32GB RAM, 1TB SSD' }
        ]
      },
      {
        name: 'Smartphones',
        products: [
          { name: 'Phone X', price: 999, specs: '6.1" display, 128GB' },
          { name: 'Phone Pro Max', price: 1299, specs: '6.7" display, 256GB' }
        ]
      }
    ]
  }
];

for (const category of catalog) {
  // Depth 0: Category
  const catId = await attention.hyperbolic.storeWithHierarchy(
    embed(category.category),
    { name: category.category, type: 'category' },
    0
  );

  for (const subcategory of category.subcategories) {
    // Depth 1: Subcategory
    const subId = await attention.hyperbolic.storeWithHierarchy(
      embed(subcategory.name),
      { name: subcategory.name, type: 'subcategory', parent: catId },
      1
    );

    for (const product of subcategory.products) {
      // Depth 2: Product
      await attention.hyperbolic.storeWithHierarchy(
        embed(`${product.name} ${product.specs}`),
        {
          ...product,
          type: 'product',
          parent: subId
        },
        2
      );
    }
  }
}

// Search for products
const results = await attention.hyperbolic.hierarchicalSearch(
  embed("laptop with 16GB RAM"),
  5
);

results
  .filter(r => r.metadata.type === 'product')
  .forEach(r => {
    console.log(`${r.metadata.name} - $${r.metadata.price}`);
    console.log(`Specs: ${r.metadata.specs}`);
  });
```

## Dynamic Hierarchy Updates

### Moving Nodes

```typescript
// Move a memory to a different parent
async function reparentMemory(
  memoryId: number,
  newParentId: number,
  newDepth: number
) {
  // Update hierarchy
  await attention.hyperbolic.updateHierarchy(memoryId, newDepth);

  // Update parent reference
  const memory = await db.get(
    'SELECT metadata FROM hyperbolic_memory WHERE id = ?',
    memoryId
  );

  const metadata = JSON.parse(memory.metadata);
  metadata.parent = newParentId;

  await db.run(
    'UPDATE hyperbolic_memory SET metadata = ? WHERE id = ?',
    JSON.stringify(metadata),
    memoryId
  );
}

// Example: Move "Promises" from "Async JavaScript" to "Advanced Topics"
const advancedId = await attention.hyperbolic.storeWithHierarchy(
  embed("Advanced JavaScript Topics"),
  { name: "Advanced Topics", type: "category" },
  0
);

await reparentMemory(promisesId, advancedId, 1);
```

### Pruning Hierarchies

```typescript
// Remove entire subtree
async function pruneSubtree(rootId: number) {
  // Get all descendants
  const descendants = await db.all(`
    WITH RECURSIVE tree AS (
      SELECT id, metadata FROM hyperbolic_memory WHERE id = ?
      UNION ALL
      SELECT h.id, h.metadata
      FROM hyperbolic_memory h
      JOIN tree t ON json_extract(h.metadata, '$.parent') = t.id
    )
    SELECT id FROM tree
  `, rootId);

  // Delete all descendants
  for (const node of descendants) {
    await db.run('DELETE FROM hyperbolic_memory WHERE id = ?', node.id);
  }

  console.log(`Pruned ${descendants.length} nodes`);
}

// Remove entire Python branch
await pruneSubtree(pythonId);
```

## Performance Optimization

### Batch Insertions

```typescript
// Efficient bulk insertion
async function bulkInsertHierarchy(nodes: Array<{
  text: string;
  metadata: any;
  depth: number;
}>) {
  const transaction = db.transaction((nodes) => {
    for (const node of nodes) {
      attention.hyperbolic.storeWithHierarchy(
        embed(node.text),
        node.metadata,
        node.depth
      );
    }
  });

  transaction(nodes);
}

// Use for large datasets
const nodes = [];
for (let i = 0; i < 10000; i++) {
  nodes.push({
    text: `Document ${i}`,
    metadata: { id: i, type: 'document' },
    depth: i % 3  // Distribute across depths
  });
}

await bulkInsertHierarchy(nodes);
```

### Indexing Strategy

```typescript
// Create indexes for common queries
db.exec(`
  CREATE INDEX IF NOT EXISTS idx_hyperbolic_depth
  ON hyperbolic_memory(depth);

  CREATE INDEX IF NOT EXISTS idx_hyperbolic_parent
  ON hyperbolic_memory(json_extract(metadata, '$.parent'));

  CREATE INDEX IF NOT EXISTS idx_hyperbolic_type
  ON hyperbolic_memory(json_extract(metadata, '$.type'));
`);
```

## Statistics and Analytics

### Hierarchy Analysis

```typescript
const stats = attention.hyperbolic.getHierarchyStats();

console.log(`Total nodes: ${stats.totalNodes}`);
console.log(`Max depth: ${stats.maxDepth}`);
console.log(`Average depth: ${stats.avgDepth.toFixed(2)}`);

console.log('\nDepth distribution:');
stats.depthDistribution.forEach((count, depth) => {
  const bar = '█'.repeat(count / 10);
  console.log(`Depth ${depth}: ${bar} (${count} nodes)`);
});
```

Output:
```
Total nodes: 1543
Max depth: 4
Average depth: 2.34

Depth distribution:
Depth 0: ████ (45 nodes)
Depth 1: ████████ (82 nodes)
Depth 2: ████████████████ (165 nodes)
Depth 3: ██████████ (101 nodes)
Depth 4: ███ (30 nodes)
```

### Search Analytics

```typescript
// Analyze search patterns
const queries = [
  "JavaScript async",
  "Python generators",
  "React hooks"
];

for (const query of queries) {
  const results = await attention.hyperbolic.hierarchicalSearch(
    embed(query),
    10
  );

  console.log(`\nQuery: "${query}"`);
  console.log(`Results: ${results.length}`);

  const avgDepth = results.reduce((sum, r) => sum + r.depth, 0) / results.length;
  console.log(`Average result depth: ${avgDepth.toFixed(2)}`);

  const avgScore = results.reduce((sum, r) => sum + r.hyperbolicScore, 0) / results.length;
  console.log(`Average score: ${avgScore.toFixed(4)}`);
}
```

## Best Practices

### 1. Hierarchy Design

✅ **Good:**
```typescript
// Clear depth semantics
const languageId = await store(embed("JavaScript"), {}, 0);  // Category
const featureId = await store(embed("Async"), {parent: languageId}, 1);  // Feature
const conceptId = await store(embed("Promises"), {parent: featureId}, 2);  // Specific
```

❌ **Bad:**
```typescript
// Unclear depth usage
const id1 = await store(embed("JavaScript"), {}, 3);  // Why depth 3?
const id2 = await store(embed("Promises"), {}, 0);   // Root should be category
```

### 2. Metadata Organization

✅ **Good:**
```typescript
{
  name: "Promises",
  type: "concept",
  parent: asyncId,
  code: "...",
  references: ["MDN", "ECMAScript spec"]
}
```

❌ **Bad:**
```typescript
{
  data: "Promises and stuff",  // Unstructured
  info: { ... }  // Nested without clear purpose
}
```

### 3. Search Strategies

✅ **Good:**
```typescript
// Specific depth range for targeted search
const results = await hierarchicalSearch(query, 10, 2);
```

❌ **Bad:**
```typescript
// Always searching entire hierarchy
const results = await hierarchicalSearch(query, 100, 10);
```

## Summary

You've learned:
- ✅ How hyperbolic geometry enables hierarchical memory
- ✅ Building knowledge hierarchies at scale
- ✅ Advanced search patterns and depth control
- ✅ Dynamic hierarchy updates and maintenance
- ✅ Performance optimization strategies
- ✅ Analytics and monitoring

Next: [Flash Consolidation Tutorial](03-flash-consolidation.md)
