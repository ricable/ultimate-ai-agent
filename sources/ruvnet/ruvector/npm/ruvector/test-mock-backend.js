/**
 * Mock backend for testing the main ruvector package
 * Simulates both native and WASM backends
 */

class MockVectorIndex {
  constructor(options) {
    this.options = options;
    this.vectors = new Map();
    this._stats = {
      vectorCount: 0,
      dimension: options.dimension,
      indexType: options.indexType || 'hnsw',
      memoryUsage: 0
    };
  }

  async insert(vector) {
    if (vector.values.length !== this.options.dimension) {
      throw new Error(`Vector dimension mismatch: expected ${this.options.dimension}, got ${vector.values.length}`);
    }
    this.vectors.set(vector.id, vector);
    this._stats.vectorCount = this.vectors.size;
    this._stats.memoryUsage = this.vectors.size * this.options.dimension * 4; // Rough estimate
  }

  async insertBatch(vectors, options = {}) {
    const batchSize = options.batchSize || 1000;
    const total = vectors.length;

    for (let i = 0; i < total; i += batchSize) {
      const batch = vectors.slice(i, Math.min(i + batchSize, total));
      await Promise.all(batch.map(v => this.insert(v)));

      if (options.progressCallback) {
        options.progressCallback(Math.min(i + batchSize, total) / total);
      }
    }
  }

  async search(query, options = {}) {
    const k = options.k || 10;
    const results = [];

    // Simple cosine similarity
    for (const [id, vector] of this.vectors.entries()) {
      const score = this._cosineSimilarity(query, vector.values);
      results.push({ id, score, metadata: vector.metadata });
    }

    // Sort by score descending and return top k
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, k);
  }

  _cosineSimilarity(a, b) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  async get(id) {
    return this.vectors.get(id) || null;
  }

  async delete(id) {
    const result = this.vectors.delete(id);
    if (result) {
      this._stats.vectorCount = this.vectors.size;
      this._stats.memoryUsage = this.vectors.size * this.options.dimension * 4;
    }
    return result;
  }

  stats() {
    return { ...this._stats };
  }

  async save(path) {
    // Mock save - just log
    console.log(`Mock: Saving index to ${path}`);
  }

  static async load(path) {
    // Mock load - create empty index
    console.log(`Mock: Loading index from ${path}`);
    return new MockVectorIndex({ dimension: 384, indexType: 'hnsw' });
  }

  async clear() {
    this.vectors.clear();
    this._stats.vectorCount = 0;
    this._stats.memoryUsage = 0;
  }

  async optimize() {
    // Mock optimize
    console.log('Mock: Optimizing index');
  }
}

module.exports = { VectorIndex: MockVectorIndex };
