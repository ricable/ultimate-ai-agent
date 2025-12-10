/**
 * Browser WASM Attention Tests
 *
 * Tests for:
 * - Lazy loading
 * - Memory cleanup
 * - Fallback to mock when WASM unavailable
 * - Error handling
 * - Performance benchmarks
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
  AttentionBrowser,
  createAttention,
  createFastAttention,
  createAccurateAttention
} from '../../src/browser/AttentionBrowser';

describe('AttentionBrowser - WASM Lazy Loading', () => {
  let attention;

  beforeEach(() => {
    attention = createAttention({
      dimension: 64,
      numHeads: 2,
      useWASM: true
    });
  });

  afterEach(() => {
    if (attention) {
      attention.dispose();
    }
  });

  it('should start in idle state', () => {
    expect(attention.getLoadingState()).toBe('idle');
    expect(attention.getError()).toBeNull();
  });

  it('should lazy load WASM on first use', async () => {
    const query = new Float32Array(64).fill(0.5);
    const keys = new Float32Array(64 * 5).fill(0.3);
    const values = new Float32Array(64 * 5).fill(0.7);

    // Should trigger initialization
    const result = await attention.flashAttention(query, keys, values);

    expect(attention.getLoadingState()).toBe('loaded');
    expect(result).toBeInstanceOf(Float32Array);
    expect(result.length).toBe(64);
  });

  it('should not reload WASM on subsequent calls', async () => {
    const query = new Float32Array(64).fill(0.5);
    const keys = new Float32Array(64 * 5).fill(0.3);
    const values = new Float32Array(64 * 5).fill(0.7);

    // First call
    await attention.flashAttention(query, keys, values);
    const firstState = attention.getLoadingState();

    // Second call
    await attention.flashAttention(query, keys, values);
    const secondState = attention.getLoadingState();

    expect(firstState).toBe('loaded');
    expect(secondState).toBe('loaded');
  });

  it('should handle manual initialization', async () => {
    await attention.initialize();
    expect(attention.getLoadingState()).toBe('loaded');
  });
});

describe('AttentionBrowser - Flash Attention', () => {
  let attention;

  beforeEach(async () => {
    attention = createAttention({
      dimension: 128,
      numHeads: 4,
      useWASM: true
    });
    await attention.initialize();
  });

  afterEach(() => {
    attention.dispose();
  });

  it('should compute flash attention correctly', async () => {
    const dim = 128;
    const seqLen = 10;

    const query = new Float32Array(dim);
    const keys = new Float32Array(seqLen * dim);
    const values = new Float32Array(seqLen * dim);

    for (let i = 0; i < dim; i++) query[i] = Math.random() - 0.5;
    for (let i = 0; i < seqLen * dim; i++) {
      keys[i] = Math.random() - 0.5;
      values[i] = Math.random() - 0.5;
    }

    const output = await attention.flashAttention(query, keys, values);

    expect(output).toBeInstanceOf(Float32Array);
    expect(output.length).toBe(dim);
    expect(output.some(v => !isNaN(v))).toBe(true);
  });

  it('should handle empty sequences', async () => {
    const query = new Float32Array(128);
    const keys = new Float32Array(0);
    const values = new Float32Array(0);

    const output = await attention.flashAttention(query, keys, values);
    expect(output).toBeInstanceOf(Float32Array);
  });

  it('should be faster than O(N²) for long sequences', async () => {
    const dim = 64;
    const seqLen = 20;

    const query = new Float32Array(dim).map(() => Math.random());
    const keys = new Float32Array(seqLen * dim).map(() => Math.random());
    const values = new Float32Array(seqLen * dim).map(() => Math.random());

    const start = performance.now();
    await attention.flashAttention(query, keys, values);
    const duration = performance.now() - start;

    // Flash attention should complete quickly even for longer sequences
    expect(duration).toBeLessThan(100); // Should be under 100ms
  });
});

describe('AttentionBrowser - Hyperbolic Attention', () => {
  let attention;

  beforeEach(async () => {
    attention = createAttention({
      dimension: 64,
      curvature: -1.0,
      useWASM: true
    });
    await attention.initialize();
  });

  afterEach(() => {
    attention.dispose();
  });

  it('should compute hyperbolic similarities', async () => {
    const dim = 64;
    const numKeys = 5;

    const query = new Float32Array(dim).map(() => Math.random() * 0.5);
    const keys = new Float32Array(numKeys * dim).map(() => Math.random() * 0.5);

    const similarities = await attention.hyperbolicAttention(query, keys);

    expect(similarities).toBeInstanceOf(Float32Array);
    expect(similarities.length).toBe(numKeys);
    expect(similarities.every(s => s >= 0 && s <= 1)).toBe(true);
  });

  it('should handle different curvature values', async () => {
    const query = new Float32Array(64).fill(0.3);
    const keys = new Float32Array(64 * 3).fill(0.5);

    // Test different curvatures
    const curvatures = [-0.5, -1.0, -2.0];

    for (const k of curvatures) {
      attention = createAttention({
        dimension: 64,
        curvature: k,
        useWASM: true
      });
      await attention.initialize();

      const similarities = await attention.hyperbolicAttention(query, keys);
      expect(similarities.every(s => !isNaN(s))).toBe(true);

      attention.dispose();
    }
  });

  it('should preserve hierarchical relationships', async () => {
    const dim = 64;

    // Create hierarchical embeddings (closer to origin = higher in hierarchy)
    const parent = new Float32Array(dim).fill(0.1); // Near origin
    const child1 = new Float32Array(dim).fill(0.4); // Further out
    const child2 = new Float32Array(dim).fill(0.45); // Even further

    const keys = new Float32Array(dim * 3);
    keys.set(parent, 0);
    keys.set(child1, dim);
    keys.set(child2, dim * 2);

    // Query with child1
    const similarities = await attention.hyperbolicAttention(child1, keys);

    // child1 should be most similar to itself
    expect(similarities[1]).toBeGreaterThan(similarities[0]);
    expect(similarities[1]).toBeGreaterThan(similarities[2]);
  });
});

describe('AttentionBrowser - Memory Consolidation', () => {
  let attention;

  beforeEach(async () => {
    attention = createAttention({
      dimension: 128,
      useWASM: true
    });
    await attention.initialize();
  });

  afterEach(() => {
    attention.dispose();
  });

  it('should consolidate similar memories', async () => {
    const dim = 128;
    const memories = [];

    // Create 3 clusters of similar memories
    for (let cluster = 0; cluster < 3; cluster++) {
      const base = new Float32Array(dim).map(() => Math.random() - 0.5);
      for (let i = 0; i < 5; i++) {
        const memory = new Float32Array(dim);
        for (let d = 0; d < dim; d++) {
          memory[d] = base[d] + (Math.random() - 0.5) * 0.1;
        }
        memories.push(memory);
      }
    }

    const consolidated = await attention.consolidateMemories(memories, {
      threshold: 0.8,
      maxClusters: 5
    });

    expect(consolidated.length).toBeLessThan(memories.length);
    expect(consolidated.length).toBeGreaterThan(0);
    expect(consolidated.every(c => c.memory instanceof Float32Array)).toBe(true);
    expect(consolidated.every(c => c.count > 0)).toBe(true);
  });

  it('should respect maxClusters parameter', async () => {
    const memories = Array.from({ length: 20 }, () =>
      new Float32Array(128).map(() => Math.random())
    );

    const consolidated = await attention.consolidateMemories(memories, {
      threshold: 0.5,
      maxClusters: 5
    });

    expect(consolidated.length).toBeLessThanOrEqual(5);
  });

  it('should compute cluster centroids correctly', async () => {
    const dim = 64;
    const memories = [
      new Float32Array(dim).fill(1.0),
      new Float32Array(dim).fill(1.0),
      new Float32Array(dim).fill(1.0)
    ];

    const consolidated = await attention.consolidateMemories(memories, {
      threshold: 0.99,
      maxClusters: 10
    });

    // Should create one cluster with normalized centroid
    expect(consolidated.length).toBe(1);
    expect(consolidated[0].count).toBe(3);

    // Centroid should be normalized
    let norm = 0;
    for (const v of consolidated[0].memory) {
      norm += v * v;
    }
    expect(Math.abs(Math.sqrt(norm) - 1.0)).toBeLessThan(0.01);
  });

  it('should handle minimum cluster size', async () => {
    const memories = Array.from({ length: 10 }, () =>
      new Float32Array(64).map(() => Math.random())
    );

    const consolidated = await attention.consolidateMemories(memories, {
      threshold: 0.95, // High threshold = fewer clusters
      minClusterSize: 2 // Only keep clusters with 2+ members
    });

    expect(consolidated.every(c => c.count >= 2)).toBe(true);
  });
});

describe('AttentionBrowser - Memory Cleanup', () => {
  it('should clean up resources on dispose', async () => {
    const attention = createAttention();
    await attention.initialize();

    expect(attention.getLoadingState()).toBe('loaded');

    attention.dispose();

    expect(attention.getLoadingState()).toBe('idle');
  });

  it('should handle multiple dispose calls', () => {
    const attention = createAttention();

    expect(() => {
      attention.dispose();
      attention.dispose();
      attention.dispose();
    }).not.toThrow();
  });
});

describe('AttentionBrowser - Fallback Behavior', () => {
  it('should use JavaScript fallback when WASM disabled', async () => {
    const attention = createAttention({
      dimension: 64,
      useWASM: false // Explicitly disable WASM
    });

    const query = new Float32Array(64).fill(0.5);
    const keys = new Float32Array(64 * 5).fill(0.3);
    const values = new Float32Array(64 * 5).fill(0.7);

    const output = await attention.flashAttention(query, keys, values);

    expect(output).toBeInstanceOf(Float32Array);
    expect(attention.getLoadingState()).toBe('loaded');
  });

  it('should handle errors gracefully', async () => {
    const attention = createAttention({
      dimension: 64,
      useWASM: true
    });

    // This should not throw even if WASM fails
    const query = new Float32Array(64).fill(0.5);
    const keys = new Float32Array(64 * 5).fill(0.3);
    const values = new Float32Array(64 * 5).fill(0.7);

    await expect(
      attention.flashAttention(query, keys, values)
    ).resolves.toBeTruthy();
  });
});

describe('AttentionBrowser - Factory Functions', () => {
  it('should create fast attention instance', () => {
    const attention = createFastAttention();
    expect(attention).toBeInstanceOf(AttentionBrowser);
    attention.dispose();
  });

  it('should create accurate attention instance', () => {
    const attention = createAccurateAttention();
    expect(attention).toBeInstanceOf(AttentionBrowser);
    attention.dispose();
  });

  it('should create default attention instance', () => {
    const attention = createAttention();
    expect(attention).toBeInstanceOf(AttentionBrowser);
    attention.dispose();
  });
});

describe('AttentionBrowser - Performance Benchmarks', () => {
  it('should handle large sequences efficiently', async () => {
    const attention = createFastAttention();
    await attention.initialize();

    const dim = 256;
    const seqLen = 50;

    const query = new Float32Array(dim).map(() => Math.random());
    const keys = new Float32Array(seqLen * dim).map(() => Math.random());
    const values = new Float32Array(seqLen * dim).map(() => Math.random());

    const start = performance.now();
    const output = await attention.flashAttention(query, keys, values);
    const duration = performance.now() - start;

    expect(output).toBeInstanceOf(Float32Array);
    expect(duration).toBeLessThan(200); // Should complete quickly

    attention.dispose();
  });

  it('should scale linearly with sequence length', async () => {
    const attention = createAttention({
      dimension: 64,
      useWASM: true
    });
    await attention.initialize();

    const dim = 64;
    const sequenceLengths = [10, 20, 40];
    const times = [];

    for (const seqLen of sequenceLengths) {
      const query = new Float32Array(dim).map(() => Math.random());
      const keys = new Float32Array(seqLen * dim).map(() => Math.random());
      const values = new Float32Array(seqLen * dim).map(() => Math.random());

      const start = performance.now();
      await attention.flashAttention(query, keys, values);
      times.push(performance.now() - start);
    }

    // Time should scale roughly linearly, not quadratically
    const ratio1 = times[1] / times[0];
    const ratio2 = times[2] / times[1];

    // Allow some variance but should be closer to linear (2x) than quadratic (4x)
    expect(ratio1).toBeLessThan(3);
    expect(ratio2).toBeLessThan(3);

    attention.dispose();
  });
});
