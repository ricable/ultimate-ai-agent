# AgentDB v2 Regression Test Plan

## Overview

Comprehensive regression testing strategy to ensure backward compatibility and feature parity between RuVector and hnswlib backends.

## Test Categories

### 1. Backend Parity Tests

Ensure both backends produce equivalent results for identical operations.

```typescript
// packages/agentdb/tests/regression/backend-parity.test.ts

import { describe, it, expect, beforeAll } from 'vitest';
import { createBackend } from '../../src/backends/factory.js';

describe('Backend Parity', () => {
  let ruvector: VectorBackend;
  let hnswlib: VectorBackend;

  const testVectors = generateTestVectors(1000, 384);
  const queries = generateTestVectors(100, 384);

  beforeAll(async () => {
    ruvector = await createBackend('ruvector', { dimension: 384, metric: 'cosine' });
    hnswlib = await createBackend('hnswlib', { dimension: 384, metric: 'cosine' });

    // Insert identical data
    for (const { id, embedding } of testVectors) {
      ruvector.insert(id, embedding);
      hnswlib.insert(id, embedding);
    }
  });

  describe('Search Result Parity', () => {
    it('should return same top-1 result', () => {
      for (const query of queries.slice(0, 10)) {
        const rvResult = ruvector.search(query.embedding, 1)[0];
        const hwResult = hnswlib.search(query.embedding, 1)[0];

        expect(rvResult.id).toBe(hwResult.id);
      }
    });

    it('should return same top-10 results (order may vary for ties)', () => {
      for (const query of queries.slice(0, 10)) {
        const rvResults = ruvector.search(query.embedding, 10);
        const hwResults = hnswlib.search(query.embedding, 10);

        const rvIds = new Set(rvResults.map(r => r.id));
        const hwIds = new Set(hwResults.map(r => r.id));

        // At least 90% overlap (HNSW is approximate)
        const overlap = [...rvIds].filter(id => hwIds.has(id)).length;
        expect(overlap).toBeGreaterThanOrEqual(9);
      }
    });

    it('should produce similar similarity scores', () => {
      for (const query of queries.slice(0, 10)) {
        const rvResult = ruvector.search(query.embedding, 1)[0];
        const hwResult = hnswlib.search(query.embedding, 1)[0];

        // Scores should be within 1%
        expect(Math.abs(rvResult.similarity - hwResult.similarity)).toBeLessThan(0.01);
      }
    });
  });

  describe('Insert/Remove Parity', () => {
    it('should maintain count after insertions', () => {
      const initialRv = ruvector.getStats().count;
      const initialHw = hnswlib.getStats().count;

      expect(initialRv).toBe(initialHw);

      // Insert new vectors
      const newVectors = generateTestVectors(100, 384);
      for (const { id, embedding } of newVectors) {
        ruvector.insert(id, embedding);
        hnswlib.insert(id, embedding);
      }

      expect(ruvector.getStats().count).toBe(initialRv + 100);
      expect(hnswlib.getStats().count).toBe(initialHw + 100);
    });

    it('should handle removals correctly', () => {
      const idToRemove = testVectors[0].id;

      ruvector.remove(idToRemove);
      hnswlib.remove(idToRemove);

      // ID should not appear in search results
      const query = testVectors[0].embedding;
      const rvResults = ruvector.search(query, 10);
      const hwResults = hnswlib.search(query, 10);

      expect(rvResults.find(r => r.id === idToRemove)).toBeUndefined();
      expect(hwResults.find(r => r.id === idToRemove)).toBeUndefined();
    });
  });
});
```

### 2. API Compatibility Tests

Ensure v2 API is fully backward compatible with v1.

```typescript
// packages/agentdb/tests/regression/api-compat.test.ts

describe('API Backward Compatibility', () => {
  describe('ReasoningBank API', () => {
    it('should support v1 storePattern signature', async () => {
      const pattern = {
        taskType: 'code_review',
        approach: 'Review for bugs and style',
        successRate: 0.85
      };

      // v1 signature still works
      const id = await reasoningBank.storePattern(pattern);
      expect(id).toBeDefined();
    });

    it('should support v1 searchPatterns signature', async () => {
      // v1 signature with query object
      const results = await reasoningBank.searchPatterns({
        taskEmbedding: queryVector,
        k: 10,
        threshold: 0.7
      });

      expect(Array.isArray(results)).toBe(true);
    });

    it('should support v2 searchPatterns with GNN option', async () => {
      // v2 enhanced signature
      const results = await reasoningBank.searchPatterns('review code', 10, {
        useGNN: true,
        threshold: 0.7
      });

      expect(Array.isArray(results)).toBe(true);
    });
  });

  describe('HNSWIndex API', () => {
    it('should maintain v1 HNSWIndex interface', async () => {
      // Old import path should still work
      const { HNSWIndex } = await import('agentdb');

      const index = new HNSWIndex(db, {
        dimension: 384,
        metric: 'cosine'
      });

      await index.buildIndex();
      const results = await index.search(queryVector, 10);

      expect(results).toBeDefined();
    });
  });

  describe('CLI Compatibility', () => {
    it('should support legacy CLI commands', async () => {
      // Existing commands still work
      const { execSync } = await import('child_process');

      execSync('npx agentdb --version');
      execSync('npx agentdb benchmark --help');
    });
  });
});
```

### 3. Persistence Tests

Ensure data survives restarts and is compatible between backends.

```typescript
// packages/agentdb/tests/regression/persistence.test.ts

describe('Persistence', () => {
  const tempDir = '/tmp/agentdb-test-' + Date.now();

  describe('Save/Load Cycle', () => {
    it('should persist and restore RuVector index', async () => {
      const backend = await createBackend('ruvector', { dimension: 384, metric: 'cosine' });

      // Insert data
      for (let i = 0; i < 1000; i++) {
        backend.insert(`vec-${i}`, generateVector(384));
      }

      // Save
      await backend.save(`${tempDir}/ruvector-index`);
      backend.close();

      // Load in new instance
      const restored = await createBackend('ruvector', { dimension: 384, metric: 'cosine' });
      await restored.load(`${tempDir}/ruvector-index`);

      expect(restored.getStats().count).toBe(1000);

      // Verify search works
      const results = restored.search(generateVector(384), 10);
      expect(results.length).toBe(10);
    });

    it('should persist and restore hnswlib index', async () => {
      const backend = await createBackend('hnswlib', { dimension: 384, metric: 'cosine' });

      for (let i = 0; i < 1000; i++) {
        backend.insert(`vec-${i}`, generateVector(384));
      }

      await backend.save(`${tempDir}/hnswlib-index`);
      backend.close();

      const restored = await createBackend('hnswlib', { dimension: 384, metric: 'cosine' });
      await restored.load(`${tempDir}/hnswlib-index`);

      expect(restored.getStats().count).toBe(1000);
    });
  });

  describe('SQLite Compatibility', () => {
    it('should read v1 SQLite database', async () => {
      // Copy v1 fixture database
      const v1DbPath = './tests/fixtures/v1-database.sqlite';

      const instance = await init({
        backend: 'auto',
        dbPath: v1DbPath
      });

      // Verify v1 data accessible
      const patterns = await instance.reasoningBank.searchPatterns('test', 10);
      expect(patterns.length).toBeGreaterThan(0);
    });
  });
});
```

### 4. Edge Case Tests

```typescript
// packages/agentdb/tests/regression/edge-cases.test.ts

describe('Edge Cases', () => {
  describe('Empty Database', () => {
    it('should handle search on empty index', async () => {
      const backend = await createBackend('ruvector', { dimension: 384, metric: 'cosine' });

      const results = backend.search(generateVector(384), 10);
      expect(results).toEqual([]);
    });
  });

  describe('Single Vector', () => {
    it('should find the only vector', async () => {
      const backend = await createBackend('ruvector', { dimension: 384, metric: 'cosine' });
      const vec = generateVector(384);

      backend.insert('only-one', vec);
      const results = backend.search(vec, 10);

      expect(results.length).toBe(1);
      expect(results[0].id).toBe('only-one');
    });
  });

  describe('Duplicate IDs', () => {
    it('should overwrite on duplicate insert', async () => {
      const backend = await createBackend('ruvector', { dimension: 384, metric: 'cosine' });

      backend.insert('dup', generateVector(384));
      backend.insert('dup', generateVector(384));

      expect(backend.getStats().count).toBe(1);
    });
  });

  describe('Large Batch', () => {
    it('should handle 100K vector batch', async () => {
      const backend = await createBackend('ruvector', { dimension: 384, metric: 'cosine' });

      const batch = [];
      for (let i = 0; i < 100000; i++) {
        batch.push({ id: `vec-${i}`, embedding: generateVector(384) });
      }

      backend.insertBatch(batch);
      expect(backend.getStats().count).toBe(100000);
    }, 60000); // 60s timeout
  });

  describe('Boundary Values', () => {
    it('should handle k=1 search', async () => {
      const backend = await createBackend('ruvector', { dimension: 384, metric: 'cosine' });
      backend.insert('test', generateVector(384));

      const results = backend.search(generateVector(384), 1);
      expect(results.length).toBe(1);
    });

    it('should handle k larger than index size', async () => {
      const backend = await createBackend('ruvector', { dimension: 384, metric: 'cosine' });

      for (let i = 0; i < 5; i++) {
        backend.insert(`vec-${i}`, generateVector(384));
      }

      const results = backend.search(generateVector(384), 100);
      expect(results.length).toBe(5);
    });

    it('should handle threshold=1.0', async () => {
      const backend = await createBackend('ruvector', { dimension: 384, metric: 'cosine' });
      const vec = generateVector(384);

      backend.insert('exact', vec);
      const results = backend.search(vec, 10, { threshold: 0.9999 });

      expect(results.length).toBe(1);
    });
  });
});
```

### 5. Metrics Accuracy

```typescript
// packages/agentdb/tests/regression/metrics.test.ts

describe('Distance Metrics', () => {
  const backends = ['ruvector', 'hnswlib'] as const;

  for (const backendType of backends) {
    describe(backendType, () => {
      describe('Cosine', () => {
        it('should return 1.0 similarity for identical vectors', async () => {
          const backend = await createBackend(backendType, { dimension: 3, metric: 'cosine' });
          const vec = new Float32Array([1, 0, 0]);

          backend.insert('test', vec);
          const results = backend.search(vec, 1);

          expect(results[0].similarity).toBeCloseTo(1.0, 4);
        });

        it('should return 0.0 similarity for orthogonal vectors', async () => {
          const backend = await createBackend(backendType, { dimension: 3, metric: 'cosine' });

          backend.insert('x', new Float32Array([1, 0, 0]));
          const results = backend.search(new Float32Array([0, 1, 0]), 1);

          expect(results[0].similarity).toBeCloseTo(0.0, 4);
        });
      });

      describe('L2 (Euclidean)', () => {
        it('should return 0 distance for identical vectors', async () => {
          const backend = await createBackend(backendType, { dimension: 3, metric: 'l2' });
          const vec = new Float32Array([1, 2, 3]);

          backend.insert('test', vec);
          const results = backend.search(vec, 1);

          expect(results[0].distance).toBeCloseTo(0.0, 4);
        });
      });

      describe('Inner Product', () => {
        it('should rank by dot product', async () => {
          const backend = await createBackend(backendType, { dimension: 3, metric: 'ip' });

          backend.insert('high', new Float32Array([1, 1, 1]));
          backend.insert('low', new Float32Array([0.1, 0.1, 0.1]));

          const results = backend.search(new Float32Array([1, 1, 1]), 2);

          expect(results[0].id).toBe('high');
        });
      });
    });
  }
});
```

## Test Matrix

### Platform Matrix

| Platform | Node 18 | Node 20 | Node 22 |
|----------|---------|---------|---------|
| Linux x64 | ✅ | ✅ | ✅ |
| Linux ARM64 | ✅ | ✅ | ✅ |
| macOS x64 | ✅ | ✅ | ✅ |
| macOS ARM64 | ✅ | ✅ | ✅ |
| Windows x64 | ✅ | ✅ | ✅ |

### Backend Matrix

| Test Suite | RuVector Native | RuVector WASM | hnswlib |
|------------|-----------------|---------------|---------|
| Parity | ✅ | ✅ | ✅ |
| Persistence | ✅ | ✅ | ✅ |
| Edge Cases | ✅ | ✅ | ✅ |
| Performance | ✅ | ⚠️ (slower) | ✅ |

## Running Tests

```bash
# Full regression suite
npm run test:regression

# Specific category
npm run test:regression -- --filter parity
npm run test:regression -- --filter persistence

# Specific backend
AGENTDB_BACKEND=ruvector npm run test:regression
AGENTDB_BACKEND=hnswlib npm run test:regression

# With coverage
npm run test:regression -- --coverage
```

## CI Integration

See [workflows/ci.yml](../workflows/ci.yml) for GitHub Actions configuration.

## Test Fixtures

```
tests/fixtures/
├── v1-database.sqlite      # Legacy v1 database
├── vectors-1k.json         # 1000 test vectors
├── vectors-10k.json        # 10000 test vectors
├── queries-100.json        # 100 test queries
└── expected-results.json   # Expected search results
```

## Coverage Requirements

| Category | Minimum Coverage |
|----------|-----------------|
| Backends | 90% |
| Controllers | 85% |
| CLI | 80% |
| Utils | 75% |
| **Overall** | **85%** |
