# MoE Routing Deep Dive

Master Mixture of Experts routing for building specialized retrieval systems with intelligent query distribution.

## What is MoE Routing?

Mixture of Experts (MoE) routing automatically directs queries to specialized expert models:

- **Specialization**: Each expert focuses on specific domains or content types
- **Intelligent Routing**: Queries automatically go to the most relevant experts
- **Scalability**: Add experts without retraining the entire system
- **Load Balancing**: Distribute queries efficiently across experts

## Core Concepts

### Experts

Each expert specializes in a specific domain:

```
Query: "How do I fix authentication bug?"
  ↓ (routing)
Expert 1: Security & Auth ⭐ (0.92 confidence)
Expert 2: Backend Systems   (0.15 confidence)
Expert 3: Frontend UI       (0.03 confidence)
Expert 4: Database          (0.08 confidence)
```

### Router Network

The router learns which expert(s) to activate:

```typescript
router(query) = {
  expert1: 0.92,  // High confidence
  expert2: 0.15,  // Low confidence
  expert3: 0.03,  // Very low
  expert4: 0.08
}

// Top-2 activation
activeExperts = [expert1, expert2]
```

### Load Balancing

Prevents expert overload:

```
Without balancing:
Expert 1: 80% of queries ❌ (overloaded)
Expert 2: 15% of queries
Expert 3:  5% of queries

With balancing:
Expert 1: 40% of queries ✅ (balanced)
Expert 2: 35% of queries
Expert 3: 25% of queries
```

## Creating Expert Systems

### Basic Expert Setup

```typescript
import Database from 'better-sqlite3';
import { AttentionService } from '@agentic/agentdb';

const db = new Database('experts.db');
const attention = new AttentionService(db, {
  enableMoE: true,
  moeExpertCount: 8,
  moeTopK: 2,           // Activate top 2 experts
  moeLoadBalance: true  // Enable load balancing
});

function embed(text: string): Float32Array {
  // Your embedding model here
  return new Float32Array(1536);
}

// Create specialized experts
const experts = [
  {
    name: 'JavaScript Expert',
    specialization: 'javascript',
    trainingData: [
      "JavaScript promises for async operations",
      "React hooks and state management",
      "Node.js event loop explained",
      "TypeScript type system basics",
      "ES6 arrow functions and closures"
    ]
  },
  {
    name: 'Python Expert',
    specialization: 'python',
    trainingData: [
      "Python list comprehensions",
      "Django ORM queries",
      "Python async/await syntax",
      "Flask RESTful API design",
      "NumPy array operations"
    ]
  },
  {
    name: 'Database Expert',
    specialization: 'database',
    trainingData: [
      "SQL join operations explained",
      "MongoDB aggregation pipelines",
      "PostgreSQL indexing strategies",
      "Redis caching patterns",
      "Database normalization"
    ]
  },
  {
    name: 'Security Expert',
    specialization: 'security',
    trainingData: [
      "JWT authentication implementation",
      "OAuth 2.0 flow explained",
      "SQL injection prevention",
      "CORS configuration best practices",
      "Password hashing with bcrypt"
    ]
  }
];

// Initialize experts
for (const expert of experts) {
  const vectors = expert.trainingData.map(text => embed(text));

  const expertId = await attention.moe.addExpert(
    expert.name,
    expert.specialization,
    vectors
  );

  console.log(`Created expert ${expertId}: ${expert.name}`);
}

console.log('Expert system initialized');
```

### Query Routing

```typescript
// Route queries to appropriate experts
const queries = [
  "How do I implement JWT authentication?",
  "What are Python decorators?",
  "How to optimize SQL queries?",
  "React useEffect cleanup function"
];

for (const queryText of queries) {
  console.log(`\nQuery: "${queryText}"`);

  const query = embed(queryText);
  const results = await attention.moe.routeQuery(
    query,
    3,  // 3 results per expert
    2   // Query top 2 experts
  );

  // Group by expert
  const byExpert = new Map<string, typeof results>();
  results.forEach(r => {
    if (!byExpert.has(r.expertName)) {
      byExpert.set(r.expertName, []);
    }
    byExpert.get(r.expertName)!.push(r);
  });

  // Display results
  byExpert.forEach((expertResults, expertName) => {
    console.log(`\n  ${expertName} (confidence: ${expertResults[0].routingScore.toFixed(3)}):`);
    expertResults.forEach((r, i) => {
      console.log(`    ${i + 1}. ${r.metadata.text?.substring(0, 50)}...`);
      console.log(`       Score: ${r.score.toFixed(4)}`);
    });
  });
}
```

Expected output:
```
Query: "How do I implement JWT authentication?"

  Security Expert (confidence: 0.934):
    1. JWT authentication implementation...
       Score: 0.9123
    2. OAuth 2.0 flow explained...
       Score: 0.7845
    3. Password hashing with bcrypt...
       Score: 0.6234

  JavaScript Expert (confidence: 0.156):
    1. Node.js event loop explained...
       Score: 0.4523
```

## Advanced Patterns

### 1. Multi-Domain Knowledge Base

```typescript
// Build comprehensive knowledge base with specialized experts
interface Domain {
  name: string;
  subdomain: string;
  documents: string[];
}

const domains: Domain[] = [
  {
    name: 'Programming',
    subdomain: 'Frontend',
    documents: [
      "React component lifecycle methods",
      "Vue.js reactive data system",
      "Angular dependency injection",
      "CSS Grid layout techniques",
      "Webpack bundling optimization"
    ]
  },
  {
    name: 'Programming',
    subdomain: 'Backend',
    documents: [
      "REST API design principles",
      "GraphQL query optimization",
      "Microservices architecture patterns",
      "Message queue systems (RabbitMQ)",
      "API rate limiting strategies"
    ]
  },
  {
    name: 'DevOps',
    subdomain: 'CI/CD',
    documents: [
      "GitHub Actions workflow automation",
      "Docker multi-stage builds",
      "Kubernetes deployment strategies",
      "Jenkins pipeline configuration",
      "Terraform infrastructure as code"
    ]
  },
  {
    name: 'Data Science',
    subdomain: 'Machine Learning',
    documents: [
      "Supervised learning algorithms",
      "Neural network architectures",
      "Feature engineering techniques",
      "Model evaluation metrics",
      "Hyperparameter tuning strategies"
    ]
  }
];

// Create expert for each subdomain
for (const domain of domains) {
  const expertName = `${domain.name} - ${domain.subdomain} Expert`;
  const vectors = domain.documents.map(doc => embed(doc));

  await attention.moe.addExpert(
    expertName,
    domain.subdomain.toLowerCase().replace(/\s+/g, '_'),
    vectors
  );

  console.log(`Created: ${expertName} (${domain.documents.length} docs)`);
}

// Query with automatic routing
const multiDomainQueries = [
  "How do I deploy a containerized application?",
  "What are best practices for API design?",
  "How to build a neural network?",
  "React state management patterns"
];

for (const query of multiDomainQueries) {
  const results = await attention.moe.routeQuery(embed(query), 2, 3);

  console.log(`\nQuery: "${query}"`);
  console.log('Routed to:');

  const experts = [...new Set(results.map(r => r.expertName))];
  experts.forEach(name => {
    const expertResults = results.filter(r => r.expertName === name);
    console.log(`  - ${name}: ${expertResults[0].routingScore.toFixed(3)} confidence`);
  });
}
```

### 2. Dynamic Expert Creation

```typescript
// Automatically create experts based on content clustering
async function autoCreateExperts(
  documents: Array<{ text: string; category: string }>,
  expertsPerCategory: number = 1
) {
  // Group by category
  const byCategory = new Map<string, typeof documents>();

  documents.forEach(doc => {
    if (!byCategory.has(doc.category)) {
      byCategory.set(doc.category, []);
    }
    byCategory.get(doc.category)!.push(doc);
  });

  // Create expert for each category
  for (const [category, docs] of byCategory) {
    if (docs.length < 5) {
      console.log(`Skipping ${category}: too few documents (${docs.length})`);
      continue;
    }

    const vectors = docs.map(d => embed(d.text));
    const expertName = `${category} Expert`;

    await attention.moe.addExpert(
      expertName,
      category.toLowerCase().replace(/\s+/g, '_'),
      vectors
    );

    console.log(`Auto-created: ${expertName} (${docs.length} documents)`);
  }
}

// Example: Auto-create from document corpus
const corpus = [
  { text: "React hooks tutorial", category: "Frontend" },
  { text: "Vue.js components", category: "Frontend" },
  { text: "Angular services", category: "Frontend" },
  { text: "CSS animations", category: "Frontend" },
  { text: "HTML semantics", category: "Frontend" },

  { text: "Express.js routing", category: "Backend" },
  { text: "Django views", category: "Backend" },
  { text: "Node.js streams", category: "Backend" },
  { text: "Flask blueprints", category: "Backend" },
  { text: "FastAPI endpoints", category: "Backend" },

  { text: "PostgreSQL optimization", category: "Database" },
  { text: "MongoDB schemas", category: "Database" },
  { text: "Redis caching", category: "Database" },
  { text: "MySQL indexes", category: "Database" },
  { text: "SQLite performance", category: "Database" }
];

await autoCreateExperts(corpus);
```

### 3. Hierarchical Expert System

```typescript
// Create hierarchical expert structure
interface ExpertHierarchy {
  parent: string;
  children: Array<{
    name: string;
    specialization: string;
    data: string[];
  }>;
}

const hierarchy: ExpertHierarchy[] = [
  {
    parent: 'Web Development',
    children: [
      {
        name: 'React Developer',
        specialization: 'react',
        data: [
          "React hooks and custom hooks",
          "React Context API",
          "React Router navigation",
          "React performance optimization",
          "React testing with Jest"
        ]
      },
      {
        name: 'Vue Developer',
        specialization: 'vue',
        data: [
          "Vue Composition API",
          "Vuex state management",
          "Vue Router setup",
          "Vue directives",
          "Vue 3 features"
        ]
      },
      {
        name: 'Angular Developer',
        specialization: 'angular',
        data: [
          "Angular services and DI",
          "RxJS observables in Angular",
          "Angular forms (reactive/template)",
          "Angular routing guards",
          "NgRx state management"
        ]
      }
    ]
  },
  {
    parent: 'Backend Development',
    children: [
      {
        name: 'Node.js Developer',
        specialization: 'nodejs',
        data: [
          "Express middleware chains",
          "Node.js clustering",
          "Stream processing in Node",
          "Node.js error handling",
          "PM2 process management"
        ]
      },
      {
        name: 'Python Developer',
        specialization: 'python_backend',
        data: [
          "Django class-based views",
          "FastAPI async endpoints",
          "Flask application factory",
          "SQLAlchemy ORM",
          "Celery task queues"
        ]
      }
    ]
  }
];

// Create hierarchical experts
for (const level of hierarchy) {
  console.log(`\n${level.parent}:`);

  for (const child of level.children) {
    const vectors = child.data.map(text => embed(text));

    await attention.moe.addExpert(
      child.name,
      child.specialization,
      vectors
    );

    console.log(`  Created: ${child.name} (${child.data.length} items)`);
  }
}

// Query with hierarchy awareness
async function hierarchicalQuery(queryText: string) {
  const query = embed(queryText);
  const results = await attention.moe.routeQuery(query, 3, 5);

  // Group by parent domain
  const byParent = new Map<string, typeof results>();

  for (const result of results) {
    // Find parent domain for this expert
    let parent = 'Unknown';
    for (const level of hierarchy) {
      if (level.children.some(c => c.name === result.expertName)) {
        parent = level.parent;
        break;
      }
    }

    if (!byParent.has(parent)) {
      byParent.set(parent, []);
    }
    byParent.get(parent)!.push(result);
  }

  console.log(`\nQuery: "${queryText}"`);
  byParent.forEach((results, parent) => {
    console.log(`\n${parent}:`);
    results.forEach(r => {
      console.log(`  ${r.expertName}: ${r.routingScore.toFixed(3)}`);
    });
  });
}

await hierarchicalQuery("How do I handle async operations in frontend?");
```

### 4. Expert Specialization Learning

```typescript
// Train experts from query history
interface QueryHistory {
  query: string;
  expertUsed: string;
  feedback: 'positive' | 'negative';
  timestamp: Date;
}

class AdaptiveExpertSystem {
  private queryHistory: QueryHistory[] = [];

  async recordQuery(
    query: string,
    expertName: string,
    feedback: 'positive' | 'negative'
  ) {
    this.queryHistory.push({
      query,
      expertUsed: expertName,
      feedback,
      timestamp: new Date()
    });
  }

  async analyzeExpertPerformance() {
    const expertStats = new Map<string, {
      totalQueries: number;
      positiveQueries: number;
      successRate: number;
    }>();

    for (const entry of this.queryHistory) {
      if (!expertStats.has(entry.expertUsed)) {
        expertStats.set(entry.expertUsed, {
          totalQueries: 0,
          positiveQueries: 0,
          successRate: 0
        });
      }

      const stats = expertStats.get(entry.expertUsed)!;
      stats.totalQueries++;
      if (entry.feedback === 'positive') {
        stats.positiveQueries++;
      }
      stats.successRate = stats.positiveQueries / stats.totalQueries;
    }

    return expertStats;
  }

  async suggestExpertRefinement() {
    const stats = await this.analyzeExpertPerformance();

    console.log('Expert Performance Analysis:');
    for (const [expertName, data] of stats) {
      console.log(`\n${expertName}:`);
      console.log(`  Total queries: ${data.totalQueries}`);
      console.log(`  Success rate: ${(data.successRate * 100).toFixed(1)}%`);

      if (data.successRate < 0.5 && data.totalQueries > 10) {
        console.log(`  ⚠️  Recommendation: Retrain or merge with other expert`);
      } else if (data.successRate > 0.9 && data.totalQueries > 20) {
        console.log(`  ✅ Recommendation: Consider splitting into sub-experts`);
      }
    }
  }

  async optimizeExperts() {
    // Automatically optimize based on query patterns
    const stats = await this.analyzeExpertPerformance();

    for (const [expertName, data] of stats) {
      if (data.successRate < 0.4 && data.totalQueries > 15) {
        console.log(`Retraining low-performing expert: ${expertName}`);

        // Get positive examples from history
        const positiveQueries = this.queryHistory
          .filter(h => h.expertUsed === expertName && h.feedback === 'positive')
          .map(h => h.query);

        if (positiveQueries.length > 0) {
          const vectors = positiveQueries.map(q => embed(q));

          // Retrain expert (in practice, you'd replace the expert)
          console.log(`  Retraining with ${vectors.length} positive examples`);
        }
      }
    }
  }
}

// Usage
const adaptiveSystem = new AdaptiveExpertSystem();

// Simulate query history
await adaptiveSystem.recordQuery(
  "React hooks example",
  "React Developer",
  "positive"
);
await adaptiveSystem.recordQuery(
  "Vue composition API",
  "React Developer",  // Wrong expert!
  "negative"
);
await adaptiveSystem.recordQuery(
  "Angular services",
  "Angular Developer",
  "positive"
);

// Analyze and optimize
await adaptiveSystem.analyzeExpertPerformance();
await adaptiveSystem.suggestExpertRefinement();
await adaptiveSystem.optimizeExperts();
```

## Expert Management

### Adding and Removing Experts

```typescript
// Add new expert at runtime
async function addDynamicExpert(
  name: string,
  specialization: string,
  documents: string[]
) {
  const vectors = documents.map(doc => embed(doc));

  const expertId = await attention.moe.addExpert(
    name,
    specialization,
    vectors
  );

  console.log(`Added expert ${expertId}: ${name}`);
  return expertId;
}

// Remove underperforming expert
async function removeExpert(expertId: number) {
  // In practice, you'd implement this in the MoE system
  await db.run('DELETE FROM moe_experts WHERE id = ?', expertId);
  await db.run('DELETE FROM moe_expert_data WHERE expert_id = ?', expertId);

  console.log(`Removed expert ${expertId}`);
}

// Example usage
const newExpertId = await addDynamicExpert(
  'Go Developer',
  'golang',
  [
    "Go goroutines and channels",
    "Go interfaces explained",
    "Go error handling patterns"
  ]
);

// Later, if expert underperforms...
// await removeExpert(newExpertId);
```

### Expert Merging

```typescript
// Merge similar experts
async function mergeExperts(
  expert1Id: number,
  expert2Id: number,
  newName: string
) {
  // Get data from both experts
  const expert1Data = await db.all(
    'SELECT vector FROM moe_expert_data WHERE expert_id = ?',
    expert1Id
  );

  const expert2Data = await db.all(
    'SELECT vector FROM moe_expert_data WHERE expert_id = ?',
    expert2Id
  );

  // Combine vectors
  const allVectors = [
    ...expert1Data.map(row => new Float32Array(row.vector)),
    ...expert2Data.map(row => new Float32Array(row.vector))
  ];

  // Create merged expert
  const mergedId = await attention.moe.addExpert(
    newName,
    'merged',
    allVectors
  );

  console.log(`Merged experts ${expert1Id} and ${expert2Id} into ${mergedId}`);

  // Remove old experts
  await removeExpert(expert1Id);
  await removeExpert(expert2Id);

  return mergedId;
}
```

## Performance Optimization

### Load Balancing Analysis

```typescript
const stats = attention.moe.getExpertStats();

console.log('Expert Load Distribution:');

// Calculate total queries
const totalQueries = stats.reduce((sum, s) => sum + s.queryCount, 0);

// Analyze distribution
stats.forEach(expert => {
  const loadPercentage = (expert.queryCount / totalQueries) * 100;
  const bar = '█'.repeat(Math.floor(loadPercentage / 2));

  console.log(`${expert.expertName.padEnd(30)} ${bar} ${loadPercentage.toFixed(1)}%`);
  console.log(`  Queries: ${expert.queryCount}, Avg confidence: ${expert.avgConfidence.toFixed(3)}`);
});

// Check for imbalance
const avgLoad = totalQueries / stats.length;
const imbalanced = stats.filter(s =>
  s.queryCount > avgLoad * 1.5 || s.queryCount < avgLoad * 0.5
);

if (imbalanced.length > 0) {
  console.log('\n⚠️  Load imbalance detected:');
  imbalanced.forEach(expert => {
    const ratio = expert.queryCount / avgLoad;
    console.log(`  ${expert.expertName}: ${ratio.toFixed(2)}x average load`);
  });
}
```

### Routing Optimization

```typescript
// Optimize routing based on performance
const optimization = await attention.moe.optimizeRouting();

console.log('Routing Optimization Results:');
console.log(`Experts rebalanced: ${optimization.rebalanced}`);
console.log(`Experts merged: ${optimization.merged}`);
console.log(`Experts split: ${optimization.splitExperts}`);
console.log(`Performance improvement: ${optimization.improvement.toFixed(1)}%`);

// Recommendations
if (optimization.improvement > 10) {
  console.log('✅ Significant improvement achieved');
} else if (optimization.improvement > 5) {
  console.log('⚠️  Modest improvement, consider more training data');
} else {
  console.log('❌ Minimal improvement, review expert definitions');
}
```

## Best Practices

### 1. Expert Specialization

✅ **Good:**
```typescript
// Clear, distinct specializations
await attention.moe.addExpert('React Expert', 'react', reactVectors);
await attention.moe.addExpert('Vue Expert', 'vue', vueVectors);
await attention.moe.addExpert('Angular Expert', 'angular', angularVectors);
```

❌ **Bad:**
```typescript
// Overlapping, unclear specializations
await attention.moe.addExpert('Frontend Expert', 'frontend', allFrontendVectors);
await attention.moe.addExpert('UI Expert', 'ui', uiVectors);  // Too similar
```

### 2. Top-K Selection

✅ **Good:**
```typescript
// Query 2-3 experts for diverse perspectives
const results = await attention.moe.routeQuery(query, 5, 2);
```

❌ **Bad:**
```typescript
// Querying too many experts (slow, diluted results)
const results = await attention.moe.routeQuery(query, 5, 10);
```

### 3. Training Data Quality

✅ **Good:**
```typescript
// High-quality, representative data
const trainingData = [
  "Complete React hooks guide with examples",
  "Advanced React patterns: render props and HOCs",
  "React performance optimization techniques"
];
```

❌ **Bad:**
```typescript
// Low-quality, generic data
const trainingData = [
  "React",
  "React is good",
  "React tutorial"
];
```

## Summary

You've learned:
- ✅ Creating specialized expert systems
- ✅ Intelligent query routing and load balancing
- ✅ Dynamic expert management and optimization
- ✅ Hierarchical expert architectures
- ✅ Adaptive learning from query history
- ✅ Performance monitoring and optimization

Next: [Migration Guide](../MIGRATION.md)
