# Flash Consolidation Deep Dive

Master memory consolidation using Flash Attention for efficient storage and ultra-fast retrieval with sliding window mechanisms.

## What is Flash Consolidation?

Flash Consolidation applies Flash Attention's sliding window approach to memory management:

- **Memory Compression**: Consolidate many memories into efficient representations
- **O(N) Complexity**: Linear time consolidation instead of O(N²) full attention
- **Constant Memory**: Fixed window size regardless of dataset size
- **Fast Retrieval**: Logarithmic query time with WASM acceleration

## Core Concepts

### Sliding Window Attention

Instead of attending to all memories simultaneously, Flash uses a sliding window:

```
Memory sequence: [M1, M2, M3, M4, M5, M6, M7, M8]
Window size: 3

Windows:
[M1, M2, M3]
    [M2, M3, M4]
        [M3, M4, M5]
            [M4, M5, M6]
                [M5, M6, M7]
                    [M6, M7, M8]
```

Each window is consolidated independently, then combined efficiently.

### Multi-Head Attention

Multiple attention heads capture different aspects:

```
Head 1: Semantic similarity
Head 2: Temporal proximity
Head 3: Entity relationships
Head 4: Topic clustering
```

## Basic Consolidation

### Example: Consolidating Conversation History

```typescript
import Database from 'better-sqlite3';
import { AttentionService } from '@agentic/agentdb';

const db = new Database('memories.db');
const attention = new AttentionService(db, {
  enableFlash: true,
  flashWindowSize: 128,    // Window size in tokens
  flashHeadCount: 8        // Number of attention heads
});

function embed(text: string): Float32Array {
  // Your embedding model here
  return new Float32Array(1536);
}

// Simulate a conversation
const conversation = [
  "User: Hello, I need help with JavaScript",
  "Assistant: I'd be happy to help with JavaScript. What specifically?",
  "User: How do Promises work?",
  "Assistant: Promises handle async operations. They have three states...",
  "User: Can you show an example?",
  "Assistant: Sure! Here's a basic Promise: new Promise((resolve, reject) => {...})",
  "User: What about async/await?",
  "Assistant: async/await is syntactic sugar over Promises...",
  "User: That makes sense, thanks!",
  "Assistant: You're welcome! Any other questions?"
];

// Convert to embeddings
const vectors = conversation.map(msg => embed(msg));

console.log(`Original: ${vectors.length} messages`);

// Consolidate the conversation
const consolidated = await attention.flash.consolidateMemories(
  vectors,
  128  // Window size
);

console.log(`Consolidated into ${consolidated.consolidatedCount} segments`);
console.log(`Compression ratio: ${consolidated.compressionRatio.toFixed(2)}x`);
console.log(`Memory saved: ${((1 - 1/consolidated.compressionRatio) * 100).toFixed(1)}%`);
```

Expected output:
```
Original: 10 messages
Consolidated into 3 segments
Compression ratio: 3.33x
Memory saved: 70.0%
```

### Querying Consolidated Memories

```typescript
// Query the consolidated conversation
const query = embed("What did we discuss about Promises?");

const results = await attention.flash.queryConsolidated(query, 5);

results.forEach((result, i) => {
  console.log(`\n${i + 1}. Score: ${result.flashScore.toFixed(4)}`);
  console.log(`   Window: tokens ${result.windowInfo.start}-${result.windowInfo.end}`);
  console.log(`   Head activations: ${result.headActivations.join(', ')}`);
  console.log(`   Content: ${result.metadata.text?.substring(0, 60)}...`);
});
```

Output:
```
1. Score: 0.9234
   Window: tokens 64-192
   Head activations: 0.92, 0.85, 0.78, 0.91, 0.88, 0.82, 0.79, 0.86
   Content: Assistant: Promises handle async operations. They have three...

2. Score: 0.8756
   Window: tokens 192-320
   Head activations: 0.88, 0.81, 0.75, 0.87, 0.84, 0.79, 0.76, 0.83
   Content: Sure! Here's a basic Promise: new Promise((resolve, reje...
```

## Advanced Patterns

### 1. Document Chunking and Consolidation

```typescript
// Process long documents efficiently
async function consolidateDocument(document: string, chunkSize: number = 512) {
  // Split into chunks
  const chunks = [];
  for (let i = 0; i < document.length; i += chunkSize) {
    chunks.push(document.substring(i, i + chunkSize));
  }

  console.log(`Document split into ${chunks.length} chunks`);

  // Convert to embeddings
  const vectors = chunks.map(chunk => embed(chunk));

  // Consolidate with appropriate window size
  const windowSize = Math.min(256, chunks.length * 2);
  const consolidated = await attention.flash.consolidateMemories(
    vectors,
    windowSize
  );

  console.log(`Consolidated ${chunks.length} chunks into ${consolidated.consolidatedCount} segments`);

  return {
    original: chunks,
    consolidated,
    compressionRatio: consolidated.compressionRatio
  };
}

// Example: Process research paper
const paper = `
  [Long research paper text...]
  Introduction: Machine learning has revolutionized...
  Methods: We employed a novel approach...
  Results: Our experiments demonstrate...
  Discussion: These findings suggest...
  Conclusion: In summary, we have shown...
`;

const result = await consolidateDocument(paper, 512);
console.log(`Compression: ${result.compressionRatio.toFixed(2)}x`);

// Query the paper
const query = embed("What were the main findings?");
const findings = await attention.flash.queryConsolidated(query, 3);

findings.forEach(finding => {
  console.log(`Section: ${finding.metadata.section}`);
  console.log(`Content: ${finding.metadata.text}`);
});
```

### 2. Temporal Memory Consolidation

```typescript
// Consolidate memories by time periods
interface TimestampedMemory {
  timestamp: Date;
  text: string;
  vector: Float32Array;
}

async function consolidateByTimeWindow(
  memories: TimestampedMemory[],
  windowHours: number = 24
) {
  // Sort by timestamp
  memories.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

  // Group by time windows
  const windows = [];
  let currentWindow = [];
  let windowStart = memories[0].timestamp;

  for (const memory of memories) {
    const hoursDiff = (memory.timestamp.getTime() - windowStart.getTime()) / (1000 * 60 * 60);

    if (hoursDiff > windowHours) {
      windows.push(currentWindow);
      currentWindow = [];
      windowStart = memory.timestamp;
    }

    currentWindow.push(memory);
  }
  windows.push(currentWindow);

  console.log(`Created ${windows.length} time-based windows`);

  // Consolidate each window
  const consolidated = [];
  for (let i = 0; i < windows.length; i++) {
    const window = windows[i];
    const vectors = window.map(m => m.vector);

    const result = await attention.flash.consolidateMemories(vectors);

    consolidated.push({
      timeRange: {
        start: window[0].timestamp,
        end: window[window.length - 1].timestamp
      },
      originalCount: window.length,
      consolidated: result
    });

    console.log(`Window ${i + 1}: ${window.length} memories -> ${result.consolidatedCount} segments`);
  }

  return consolidated;
}

// Example: Consolidate week of activity logs
const logs: TimestampedMemory[] = [
  { timestamp: new Date('2024-01-01T09:00'), text: 'User logged in', vector: embed('login') },
  { timestamp: new Date('2024-01-01T09:15'), text: 'User viewed dashboard', vector: embed('dashboard') },
  // ... more logs
  { timestamp: new Date('2024-01-07T18:00'), text: 'User logged out', vector: embed('logout') }
];

const consolidated = await consolidateByTimeWindow(logs, 24);

// Query specific time period
const query = embed("What happened on January 3rd?");
const results = await attention.flash.queryConsolidated(query, 5);
```

### 3. Multi-Source Consolidation

```typescript
// Consolidate memories from different sources
interface SourcedMemory {
  source: string;  // 'docs', 'code', 'chat', 'web'
  text: string;
  vector: Float32Array;
  metadata: Record<string, any>;
}

async function consolidateMultiSource(memories: SourcedMemory[]) {
  // Group by source
  const bySource = new Map<string, SourcedMemory[]>();

  for (const memory of memories) {
    if (!bySource.has(memory.source)) {
      bySource.set(memory.source, []);
    }
    bySource.get(memory.source)!.push(memory);
  }

  // Consolidate each source separately
  const consolidatedBySour = new Map();

  for (const [source, sourceMemories] of bySource) {
    const vectors = sourceMemories.map(m => m.vector);
    const consolidated = await attention.flash.consolidateMemories(vectors);

    consolidatedBySource.set(source, {
      original: sourceMemories,
      consolidated,
      compressionRatio: consolidated.compressionRatio
    });

    console.log(`${source}: ${sourceMemories.length} -> ${consolidated.consolidatedCount} (${consolidated.compressionRatio.toFixed(2)}x)`);
  }

  return consolidatedBySource;
}

// Example: Knowledge base from multiple sources
const memories: SourcedMemory[] = [
  { source: 'docs', text: 'API documentation for endpoints...', vector: embed('api docs'), metadata: {} },
  { source: 'code', text: 'function handleRequest() {...}', vector: embed('code'), metadata: {} },
  { source: 'chat', text: 'User asked about authentication...', vector: embed('chat'), metadata: {} },
  { source: 'web', text: 'Stack Overflow answer about JWT...', vector: embed('jwt'), metadata: {} }
];

const consolidated = await consolidateMultiSource(memories);

// Query prioritizing certain sources
const query = embed("How do I authenticate API requests?");
const results = await attention.flash.queryConsolidated(query, 10);

// Filter and rank by source priority
const sourcePriority = { 'docs': 3, 'code': 2, 'chat': 1, 'web': 1 };
const ranked = results
  .map(r => ({
    ...r,
    priorityScore: r.flashScore * (sourcePriority[r.metadata.source] || 1)
  }))
  .sort((a, b) => b.priorityScore - a.priorityScore);
```

### 4. Incremental Consolidation

```typescript
// Efficiently add new memories to existing consolidation
class IncrementalConsolidator {
  private buffer: Float32Array[] = [];
  private bufferSize: number;
  private consolidatedSegments: any[] = [];

  constructor(
    private attention: AttentionService,
    bufferSize: number = 100
  ) {
    this.bufferSize = bufferSize;
  }

  async addMemory(vector: Float32Array, metadata: any) {
    this.buffer.push(vector);

    // Consolidate when buffer is full
    if (this.buffer.length >= this.bufferSize) {
      await this.flush();
    }
  }

  async flush() {
    if (this.buffer.length === 0) return;

    console.log(`Consolidating ${this.buffer.length} buffered memories...`);

    const consolidated = await this.attention.flash.consolidateMemories(
      this.buffer
    );

    this.consolidatedSegments.push(consolidated);
    this.buffer = [];

    console.log(`Total consolidated segments: ${this.consolidatedSegments.length}`);
  }

  async query(query: Float32Array, k: number) {
    // Query all consolidated segments
    const allResults = [];

    for (const segment of this.consolidatedSegments) {
      const results = await this.attention.flash.queryConsolidated(query, k);
      allResults.push(...results);
    }

    // Also query buffer (not consolidated yet)
    // ... handle buffer separately ...

    // Merge and rank results
    allResults.sort((a, b) => b.flashScore - a.flashScore);
    return allResults.slice(0, k);
  }

  getStats() {
    return {
      bufferedCount: this.buffer.length,
      consolidatedSegments: this.consolidatedSegments.length,
      totalMemories: this.consolidatedSegments.reduce(
        (sum, seg) => sum + seg.sourceCount,
        this.buffer.length
      )
    };
  }
}

// Usage
const consolidator = new IncrementalConsolidator(attention, 100);

// Add memories as they arrive
for await (const event of eventStream) {
  await consolidator.addMemory(
    embed(event.text),
    event.metadata
  );
}

// Flush remaining buffer
await consolidator.flush();

// Query
const results = await consolidator.query(
  embed("What events occurred?"),
  10
);

console.log(consolidator.getStats());
```

## Performance Optimization

### Window Size Tuning

```typescript
// Benchmark different window sizes
async function benchmarkWindowSizes(vectors: Float32Array[]) {
  const sizes = [64, 128, 256, 512, 1024];

  console.log('Window Size | Time (ms) | Compression | Memory (MB)');
  console.log('--------------------------------------------------------');

  for (const size of sizes) {
    const startTime = performance.now();
    const startMem = process.memoryUsage().heapUsed;

    const result = await attention.flash.consolidateMemories(vectors, size);

    const endTime = performance.now();
    const endMem = process.memoryUsage().heapUsed;

    console.log(
      `${size.toString().padEnd(11)} | ` +
      `${(endTime - startTime).toFixed(2).padEnd(9)} | ` +
      `${result.compressionRatio.toFixed(2).padEnd(11)} | ` +
      `${((endMem - startMem) / 1024 / 1024).toFixed(2)}`
    );
  }
}

// Find optimal window size for your data
const testVectors = Array(1000).fill(0).map(() =>
  new Float32Array(1536).map(() => Math.random())
);

await benchmarkWindowSizes(testVectors);
```

Expected output:
```
Window Size | Time (ms) | Compression | Memory (MB)
--------------------------------------------------------
64          | 145.23    | 2.45        | 12.34
128         | 178.45    | 3.12        | 15.67
256         | 234.56    | 4.23        | 21.23
512         | 345.67    | 5.34        | 32.45
1024        | 567.89    | 6.45        | 54.32
```

### Batch Processing

```typescript
// Process large datasets in batches
async function batchConsolidate(
  vectors: Float32Array[],
  batchSize: number = 1000
) {
  const batches = [];

  for (let i = 0; i < vectors.length; i += batchSize) {
    batches.push(vectors.slice(i, i + batchSize));
  }

  console.log(`Processing ${batches.length} batches...`);

  const results = [];
  for (let i = 0; i < batches.length; i++) {
    const batch = batches[i];
    const result = await attention.flash.consolidateMemories(batch);
    results.push(result);

    console.log(`Batch ${i + 1}/${batches.length}: ${batch.length} -> ${result.consolidatedCount}`);
  }

  return results;
}

// Process 100k vectors efficiently
const largeDataset = Array(100000).fill(0).map(() =>
  new Float32Array(1536).map(() => Math.random())
);

const results = await batchConsolidate(largeDataset, 5000);
```

## Statistics and Monitoring

### Consolidation Analytics

```typescript
const stats = attention.flash.getConsolidationStats();

console.log('Flash Consolidation Statistics:');
console.log(`Total consolidations: ${stats.totalConsolidations}`);
console.log(`Average compression ratio: ${stats.avgCompressionRatio.toFixed(2)}x`);
console.log(`Total memories processed: ${stats.totalMemoriesProcessed}`);
console.log(`Total consolidated segments: ${stats.totalSegments}`);
console.log(`Average window utilization: ${(stats.avgWindowUtilization * 100).toFixed(1)}%`);

console.log('\nPer-head statistics:');
stats.headStats.forEach((head, i) => {
  console.log(`Head ${i + 1}: avg activation = ${head.avgActivation.toFixed(4)}`);
});

console.log('\nMemory savings:');
const originalSize = stats.totalMemoriesProcessed * 1536 * 4; // float32
const consolidatedSize = stats.totalSegments * 1536 * 4;
const saved = originalSize - consolidatedSize;
console.log(`Original: ${(originalSize / 1024 / 1024).toFixed(2)} MB`);
console.log(`Consolidated: ${(consolidatedSize / 1024 / 1024).toFixed(2)} MB`);
console.log(`Saved: ${(saved / 1024 / 1024).toFixed(2)} MB (${((saved / originalSize) * 100).toFixed(1)}%)`);
```

### Query Performance Analysis

```typescript
// Analyze query performance
async function analyzeQueryPerformance(queries: Float32Array[], k: number = 10) {
  const results = [];

  for (const query of queries) {
    const startTime = performance.now();
    const queryResults = await attention.flash.queryConsolidated(query, k);
    const endTime = performance.now();

    results.push({
      queryTime: endTime - startTime,
      resultCount: queryResults.length,
      avgScore: queryResults.reduce((sum, r) => sum + r.flashScore, 0) / queryResults.length,
      topScore: queryResults[0]?.flashScore || 0
    });
  }

  const avgQueryTime = results.reduce((sum, r) => sum.queryTime, 0) / results.length;
  const avgResultCount = results.reduce((sum, r) => sum.resultCount, 0) / results.length;
  const avgTopScore = results.reduce((sum, r) => sum.topScore, 0) / results.length;

  console.log('Query Performance Analysis:');
  console.log(`Queries tested: ${queries.length}`);
  console.log(`Avg query time: ${avgQueryTime.toFixed(2)} ms`);
  console.log(`Avg results returned: ${avgResultCount.toFixed(1)}`);
  console.log(`Avg top score: ${avgTopScore.toFixed(4)}`);

  return results;
}

const testQueries = [
  embed("What is machine learning?"),
  embed("How do neural networks work?"),
  embed("Explain backpropagation")
];

await analyzeQueryPerformance(testQueries, 10);
```

## Best Practices

### 1. Window Size Selection

✅ **Good:**
```typescript
// Match window size to content length
const shortConversations = await attention.flash.consolidateMemories(vectors, 64);
const documentChunks = await attention.flash.consolidateMemories(vectors, 256);
const longTranscripts = await attention.flash.consolidateMemories(vectors, 512);
```

❌ **Bad:**
```typescript
// One size for all content
const everything = await attention.flash.consolidateMemories(vectors, 128);
```

### 2. Incremental Updates

✅ **Good:**
```typescript
// Buffer and batch consolidate
const consolidator = new IncrementalConsolidator(attention, 100);
for (const vector of newVectors) {
  await consolidator.addMemory(vector, metadata);
}
```

❌ **Bad:**
```typescript
// Reconsolidate everything on each add
for (const vector of newVectors) {
  await attention.flash.consolidateMemories([...allVectors, vector]);
}
```

### 3. Query Optimization

✅ **Good:**
```typescript
// Request only what you need
const results = await attention.flash.queryConsolidated(query, 5);
```

❌ **Bad:**
```typescript
// Over-fetching
const results = await attention.flash.queryConsolidated(query, 1000);
const filtered = results.slice(0, 5);
```

## Summary

You've learned:
- ✅ Flash Attention's sliding window mechanism
- ✅ Memory consolidation strategies
- ✅ Advanced multi-source and temporal patterns
- ✅ Performance optimization techniques
- ✅ Incremental consolidation for streaming data
- ✅ Analytics and monitoring

Next: [Graph-RoPE Recall Tutorial](04-graph-rope-recall.md)
