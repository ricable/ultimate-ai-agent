/**
 * rUvector - High-performance vector database
 *
 * Smart loader that tries native bindings first, falls back to WASM
 */

import type {
  Vector,
  SearchResult,
  IndexStats,
  CreateIndexOptions,
  SearchOptions,
  BatchInsertOptions,
  BackendInfo
} from '../types';

let backend: any;
let backendType: 'native' | 'wasm' = 'wasm';

/**
 * Try to load the native backend first, fall back to WASM
 */
function loadBackend() {
  if (backend) {
    return backend;
  }

  // Try native bindings first
  try {
    backend = require('@ruvector/core');
    backendType = 'native';
    console.log('✓ Loaded native rUvector bindings');
    return backend;
  } catch (e) {
    // Native not available, try WASM
    try {
      backend = require('@ruvector/wasm');
      backendType = 'wasm';
      console.warn('⚠ Native bindings not available, using WASM fallback');
      console.warn('  For better performance, install @ruvector/core');
      return backend;
    } catch (wasmError) {
      throw new Error(
        'Failed to load rUvector backend. Please ensure either @ruvector/core or @ruvector/wasm is installed.\n' +
        `Native error: ${e}\n` +
        `WASM error: ${wasmError}`
      );
    }
  }
}

/**
 * VectorIndex class that wraps the backend
 */
export class VectorIndex {
  private index: any;

  constructor(options: CreateIndexOptions) {
    const backend = loadBackend();
    this.index = new backend.VectorIndex(options);
  }

  async insert(vector: Vector): Promise<void> {
    return this.index.insert(vector);
  }

  async insertBatch(vectors: Vector[], options?: BatchInsertOptions): Promise<void> {
    if (this.index.insertBatch) {
      return this.index.insertBatch(vectors, options);
    }

    // Fallback for backends without batch support
    const batchSize = options?.batchSize || 1000;
    const total = vectors.length;

    for (let i = 0; i < total; i += batchSize) {
      const batch = vectors.slice(i, Math.min(i + batchSize, total));
      await Promise.all(batch.map(v => this.insert(v)));

      if (options?.progressCallback) {
        options.progressCallback(Math.min(i + batchSize, total) / total);
      }
    }
  }

  async search(query: number[], options?: SearchOptions): Promise<SearchResult[]> {
    return this.index.search(query, options);
  }

  async get(id: string): Promise<Vector | null> {
    return this.index.get(id);
  }

  async delete(id: string): Promise<boolean> {
    return this.index.delete(id);
  }

  async stats(): Promise<IndexStats> {
    return this.index.stats();
  }

  async save(path: string): Promise<void> {
    return this.index.save(path);
  }

  static async load(path: string): Promise<VectorIndex> {
    const backend = loadBackend();
    const index = await backend.VectorIndex.load(path);
    const wrapper = Object.create(VectorIndex.prototype);
    wrapper.index = index;
    return wrapper;
  }

  async clear(): Promise<void> {
    return this.index.clear();
  }

  async optimize(): Promise<void> {
    if (this.index.optimize) {
      return this.index.optimize();
    }
    // No-op for backends without optimization
  }
}

/**
 * Utility functions
 */
export const Utils = {
  cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Vectors must have the same dimension');
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  },

  euclideanDistance(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Vectors must have the same dimension');
    }

    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = a[i] - b[i];
      sum += diff * diff;
    }

    return Math.sqrt(sum);
  },

  normalize(vector: number[]): number[] {
    const norm = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return vector.map(val => val / norm);
  },

  randomVector(dimension: number): number[] {
    const vector = new Array(dimension);
    for (let i = 0; i < dimension; i++) {
      vector[i] = Math.random() * 2 - 1;
    }
    return this.normalize(vector);
  }
};

/**
 * Get backend information
 */
export function getBackendInfo(): BackendInfo {
  loadBackend();

  const features: string[] = [];

  if (backendType === 'native') {
    features.push('SIMD', 'Multi-threading', 'Memory-mapped I/O');
  } else {
    features.push('Browser-compatible', 'No native dependencies');
  }

  return {
    type: backendType,
    version: require('../package.json').version,
    features
  };
}

/**
 * Check if native bindings are available
 */
export function isNativeAvailable(): boolean {
  try {
    require.resolve('@ruvector/core');
    return true;
  } catch {
    return false;
  }
}

// Default export
export default VectorIndex;

// Re-export types
export type {
  Vector,
  SearchResult,
  IndexStats,
  CreateIndexOptions,
  SearchOptions,
  BatchInsertOptions,
  BackendInfo
};
