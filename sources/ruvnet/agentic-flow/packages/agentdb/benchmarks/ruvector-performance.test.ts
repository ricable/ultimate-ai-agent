/**
 * RuVector Performance Benchmarks
 *
 * Comprehensive benchmark suite validating optimal RuVector integration
 * across all AgentDB components:
 * - @ruvector/core: Vector database performance
 * - @ruvector/graph-node: Graph database performance
 * - @ruvector/gnn: GNN operations performance
 * - SDK integration: ReflexionMemory, SkillLibrary, etc.
 */

import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import { performance } from 'perf_hooks';

const BENCH_DIR = path.join(process.cwd(), 'bench-data');
const RESULTS: Record<string, any> = {};

// Ensure benchmark directory exists
if (!fs.existsSync(BENCH_DIR)) {
  fs.mkdirSync(BENCH_DIR, { recursive: true });
}

/**
 * Benchmark helper
 */
function benchmark(name: string, fn: () => Promise<void> | void, iterations: number = 1): Promise<number> {
  return new Promise(async (resolve) => {
    const start = performance.now();

    for (let i = 0; i < iterations; i++) {
      await fn();
    }

    const end = performance.now();
    const duration = end - start;
    const avgDuration = duration / iterations;
    const opsPerSec = iterations / (duration / 1000);

    RESULTS[name] = {
      iterations,
      totalDurationMs: duration.toFixed(2),
      avgDurationMs: avgDuration.toFixed(4),
      opsPerSec: Math.floor(opsPerSec)
    };

    console.log(`\n📊 ${name}`);
    console.log(`   Iterations: ${iterations}`);
    console.log(`   Total: ${duration.toFixed(2)}ms`);
    console.log(`   Average: ${avgDuration.toFixed(4)}ms`);
    console.log(`   Throughput: ${Math.floor(opsPerSec).toLocaleString()} ops/sec`);

    resolve(opsPerSec);
  });
}

describe('RuVector Core (@ruvector/core) Performance', () => {
  it('should benchmark vector insert performance', async () => {
    const { VectorDB } = await import('@ruvector/core');

    const db = new VectorDB({
      dimensions: 384,
      maxElements: 10000,
      distanceMetric: 'Cosine'
    });

    // Single insert
    const singleInsert = await benchmark('Vector Insert (single)', async () => {
      await db.insert('bench-single', new Float32Array(384).map(() => Math.random()));
    }, 100);

    expect(singleInsert).toBeGreaterThan(1000); // At least 1K ops/sec

    // Batch insert
    const batchEntries = Array.from({ length: 100 }, (_, i) => ({
      id: `batch-${i}`,
      vector: new Float32Array(384).map(() => Math.random())
    }));

    const batchInsert = await benchmark('Vector Insert (batch 100)', async () => {
      await db.insertBatch(batchEntries);
    }, 10);

    // Batch throughput should be 10K+ ops/sec (100 vectors per batch * throughput)
    const batchThroughput = batchInsert * 100;
    expect(batchThroughput).toBeGreaterThan(10000);

    console.log(`   → Batch throughput: ${batchThroughput.toLocaleString()} vectors/sec`);
  });

  it('should benchmark vector search performance', async () => {
    const { VectorDB } = await import('@ruvector/core');

    const db = new VectorDB({
      dimensions: 384,
      maxElements: 10000,
      distanceMetric: 'Cosine'
    });

    // Insert test data
    for (let i = 0; i < 1000; i++) {
      await db.insert(`search-${i}`, new Float32Array(384).map(() => Math.random()));
    }

    const queryVector = new Float32Array(384).map(() => Math.random());

    // Search k=10
    const search10 = await benchmark('Vector Search (k=10)', async () => {
      await db.search(queryVector, 10);
    }, 100);

    expect(search10).toBeGreaterThan(1000); // At least 1K searches/sec

    // Search k=100
    const search100 = await benchmark('Vector Search (k=100)', async () => {
      await db.search(queryVector, 100);
    }, 100);

    expect(search100).toBeGreaterThan(500); // At least 500 searches/sec
  });
});

describe('RuVector Graph Database (@ruvector/graph-node) Performance', () => {
  it('should benchmark node creation performance', async () => {
    const { GraphDatabase } = await import('@ruvector/graph-node');

    const db = new GraphDatabase({
      dimensions: 384,
      distanceMetric: 'Cosine',
      storagePath: path.join(BENCH_DIR, 'bench-graph.db')
    });

    const embedding = new Float32Array(384).map(() => Math.random());

    // Single node creation
    const singleNode = await benchmark('Graph Node Create (single)', async () => {
      await db.createNode({
        id: `node-${Math.random()}`,
        embedding,
        labels: ['Test'],
        properties: { type: 'benchmark' }
      });
    }, 100);

    expect(singleNode).toBeGreaterThan(1000); // At least 1K ops/sec

    // Batch node creation
    const nodes = Array.from({ length: 100 }, (_, i) => ({
      id: `batch-node-${i}`,
      embedding: new Float32Array(384).map(() => Math.random()),
      labels: ['Test'],
      properties: { type: 'benchmark', index: i.toString() }
    }));

    const edges: any[] = [];

    const batchCreate = await benchmark('Graph Node Create (batch 100)', async () => {
      await db.batchInsert({ nodes, edges });
    }, 10);

    const batchThroughput = batchCreate * 100;
    expect(batchThroughput).toBeGreaterThan(50000); // At least 50K nodes/sec

    console.log(`   → Batch throughput: ${batchThroughput.toLocaleString()} nodes/sec`);
  });

  it('should benchmark Cypher query performance', async () => {
    const { GraphDatabase } = await import('@ruvector/graph-node');

    const dbPath = path.join(BENCH_DIR, 'bench-cypher.db');
    const db = new GraphDatabase({
      dimensions: 384,
      distanceMetric: 'Cosine',
      storagePath: dbPath
    });

    // Insert test data
    for (let i = 0; i < 100; i++) {
      await db.createNode({
        id: `cypher-${i}`,
        embedding: new Float32Array(384).map(() => Math.random()),
        labels: ['Test'],
        properties: { index: i.toString(), type: 'benchmark' }
      });
    }

    // Simple query
    const simpleQuery = await benchmark('Cypher Query (MATCH simple)', async () => {
      await db.query('MATCH (n:Test) RETURN n LIMIT 10');
    }, 100);

    expect(simpleQuery).toBeGreaterThan(500); // At least 500 queries/sec

    // Complex query with filtering
    const complexQuery = await benchmark('Cypher Query (MATCH with WHERE)', async () => {
      await db.query('MATCH (n:Test) WHERE n.type = "benchmark" RETURN n LIMIT 10');
    }, 100);

    expect(complexQuery).toBeGreaterThan(300); // At least 300 queries/sec
  });

  it('should benchmark graph traversal performance', async () => {
    const { GraphDatabase } = await import('@ruvector/graph-node');

    const db = new GraphDatabase({
      dimensions: 384,
      distanceMetric: 'Cosine',
      storagePath: path.join(BENCH_DIR, 'bench-traversal.db')
    });

    // Create a small graph
    const nodeIds: string[] = [];
    for (let i = 0; i < 50; i++) {
      const id = await db.createNode({
        id: `trav-${i}`,
        embedding: new Float32Array(384).map(() => Math.random()),
        labels: ['Traversal'],
        properties: { index: i.toString() }
      });
      nodeIds.push(id);
    }

    // Create edges
    for (let i = 0; i < 45; i++) {
      await db.createEdge({
        from: nodeIds[i],
        to: nodeIds[i + 1],
        description: 'next',
        embedding: new Float32Array(384).map(() => Math.random()),
        confidence: 0.9
      });
    }

    // Benchmark k-hop traversal
    const traversal = await benchmark('Graph Traversal (2-hop)', async () => {
      await db.kHop(nodeIds[0], 2);
    }, 100);

    expect(traversal).toBeGreaterThan(100); // At least 100 traversals/sec
  });
});

describe('RuVector GNN (@ruvector/gnn) Performance', () => {
  it('should benchmark GNN forward pass performance', async () => {
    const { RuvectorLayer } = await import('@ruvector/gnn');

    const layer = new RuvectorLayer(384, 512, 4, 0.1);
    const nodeEmbedding = new Float32Array(384).map(() => Math.random());
    const neighborEmbeddings = [
      new Float32Array(384).map(() => Math.random()),
      new Float32Array(384).map(() => Math.random()),
      new Float32Array(384).map(() => Math.random())
    ];
    const edgeWeights = [0.8, 0.6, 0.9];

    const forwardPass = await benchmark('GNN Forward Pass (3 neighbors)', async () => {
      layer.forward(nodeEmbedding, neighborEmbeddings, edgeWeights);
    }, 1000);

    expect(forwardPass).toBeGreaterThan(500); // At least 500 forward passes/sec
  });

  it('should benchmark tensor compression performance', async () => {
    const { TensorCompress } = await import('@ruvector/gnn');

    const compressor = new TensorCompress();
    const tensor = new Float32Array(384).map(() => Math.random());

    const compression = await benchmark('Tensor Compression', async () => {
      compressor.compress(tensor, 0.5);
    }, 1000);

    expect(compression).toBeGreaterThan(1000); // At least 1K compressions/sec

    const compressed = compressor.compress(tensor, 0.5);

    const decompression = await benchmark('Tensor Decompression', async () => {
      compressor.decompress(compressed);
    }, 1000);

    expect(decompression).toBeGreaterThan(1000); // At least 1K decompressions/sec
  });

  it('should benchmark differentiable search performance', async () => {
    const { DifferentiableSearch } = await import('@ruvector/gnn');

    const searcher = new DifferentiableSearch();
    const query = new Float32Array(128).map(() => Math.random());
    const candidates = Array.from({ length: 100 }, () =>
      new Float32Array(128).map(() => Math.random())
    );

    const search = await benchmark('Differentiable Search (100 candidates)', async () => {
      searcher.search(query, candidates, 10);
    }, 100);

    expect(search).toBeGreaterThan(100); // At least 100 searches/sec
  });
});

describe('AgentDB SDK Integration Performance', () => {
  it('should benchmark ReflexionMemory with GraphDatabase', async () => {
    const { createUnifiedDatabase } = await import('../src/db-unified.js');
    const { ReflexionMemory } = await import('../src/controllers/ReflexionMemory.js');
    const { EmbeddingService } = await import('../src/controllers/EmbeddingService.js');

    const embedder = new EmbeddingService({
      model: 'Xenova/all-MiniLM-L6-v2',
      dimension: 384,
      provider: 'transformers'
    });
    await embedder.initialize();

    const db = await createUnifiedDatabase(
      path.join(BENCH_DIR, 'bench-reflexion.graph'),
      embedder,
      { forceMode: 'graph' }
    );

    const graphDb = db.getGraphDatabase()!;
    const reflexion = new ReflexionMemory(graphDb as any, embedder, undefined, undefined, graphDb as any);

    // Benchmark episode storage
    const storeEpisode = await benchmark('ReflexionMemory Store Episode', async () => {
      await reflexion.storeEpisode({
        sessionId: 'bench',
        task: `task-${Math.random()}`,
        reward: Math.random(),
        success: true,
        input: 'benchmark input',
        output: 'benchmark output'
      });
    }, 50);

    expect(storeEpisode).toBeGreaterThan(10); // At least 10 episodes/sec

    // Benchmark episode retrieval
    const retrieveEpisode = await benchmark('ReflexionMemory Retrieve Episodes', async () => {
      await reflexion.retrieveRelevant({ task: 'benchmark', k: 10 });
    }, 50);

    expect(retrieveEpisode).toBeGreaterThan(10); // At least 10 retrievals/sec

    db.close();
  });

  it('should benchmark SkillLibrary with GraphDatabase', async () => {
    const { createUnifiedDatabase } = await import('../src/db-unified.js');
    const { SkillLibrary } = await import('../src/controllers/SkillLibrary.js');
    const { EmbeddingService } = await import('../src/controllers/EmbeddingService.js');

    const embedder = new EmbeddingService({
      model: 'Xenova/all-MiniLM-L6-v2',
      dimension: 384,
      provider: 'transformers'
    });
    await embedder.initialize();

    const db = await createUnifiedDatabase(
      path.join(BENCH_DIR, 'bench-skills.graph'),
      embedder,
      { forceMode: 'graph' }
    );

    const graphDb = db.getGraphDatabase()!;
    const skills = new SkillLibrary(graphDb as any, embedder, graphDb as any);

    // Benchmark skill creation
    const createSkill = await benchmark('SkillLibrary Create Skill', async () => {
      await skills.createSkill({
        name: `skill-${Math.random()}`,
        description: 'benchmark skill',
        code: 'function bench() { return true; }',
        successRate: Math.random()
      });
    }, 50);

    expect(createSkill).toBeGreaterThan(10); // At least 10 skills/sec

    // Benchmark skill search
    const searchSkill = await benchmark('SkillLibrary Search Skills', async () => {
      await skills.searchSkills({ query: 'benchmark', k: 10 });
    }, 50);

    expect(searchSkill).toBeGreaterThan(10); // At least 10 searches/sec

    db.close();
  });
});

// After all tests, write results to file
afterAll(() => {
  const resultsPath = path.join(BENCH_DIR, 'benchmark-results.json');
  fs.writeFileSync(resultsPath, JSON.stringify(RESULTS, null, 2));

  console.log(`\n\n📊 BENCHMARK RESULTS SUMMARY\n`);
  console.log('═'.repeat(70));

  for (const [name, data] of Object.entries(RESULTS)) {
    console.log(`\n${name}:`);
    console.log(`  Throughput: ${data.opsPerSec.toLocaleString()} ops/sec`);
    console.log(`  Avg Latency: ${data.avgDurationMs}ms`);
  }

  console.log(`\n${'═'.repeat(70)}`);
  console.log(`\n✅ Results saved to: ${resultsPath}\n`);
});
