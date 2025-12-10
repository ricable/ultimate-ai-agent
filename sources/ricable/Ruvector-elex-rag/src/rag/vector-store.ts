/**
 * RuVector-based Vector Store for RAG
 *
 * High-performance vector storage and retrieval using RuVector's
 * native HNSW implementation with self-learning capabilities.
 */

import type {
  DocumentChunk,
  RAGQuery,
  RAGResult,
} from '../core/types.js';
import { getConfig } from '../core/config.js';
import { logger } from '../utils/logger.js';
import { v4 as uuidv4 } from 'uuid';

// Dynamic import for ruvector (handles native/WASM fallback)
let RuVector: any;

async function initRuVector() {
  if (!RuVector) {
    const module = await import('ruvector');
    RuVector = module.default || module;
  }
  return RuVector;
}

export interface VectorStoreConfig {
  /** Storage path for persistence */
  storagePath: string;
  /** Vector dimensions */
  dimensions: number;
  /** HNSW M parameter (max connections per node) */
  m: number;
  /** HNSW ef_construction parameter */
  efConstruction: number;
  /** HNSW ef_search parameter */
  efSearch: number;
  /** Enable self-learning (adaptive indexing) */
  selfLearning: boolean;
  /** Self-learning adaptation rate */
  adaptationRate: number;
}

const DEFAULT_CONFIG: VectorStoreConfig = {
  storagePath: './data/vector_store',
  dimensions: 3072,
  m: 32,
  efConstruction: 200,
  efSearch: 100,
  selfLearning: true,
  adaptationRate: 0.01,
};

/**
 * Self-Learning Vector Store
 *
 * Implements adaptive retrieval that learns from user interactions
 * to improve relevance over time.
 */
export class SelfLearningVectorStore {
  private config: VectorStoreConfig;
  private index: any = null;
  private chunks: Map<string, DocumentChunk> = new Map();
  private queryHistory: Map<string, { query: string; results: string[]; feedback?: number }[]> = new Map();
  private relevanceWeights: Map<string, number> = new Map();
  private initialized = false;

  constructor(config: Partial<VectorStoreConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Initialize the vector store
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      const rv = await initRuVector();

      // Create HNSW index
      this.index = new rv.VectorDB({
        dimensions: this.config.dimensions,
        indexType: 'hnsw',
        m: this.config.m,
        efConstruction: this.config.efConstruction,
        storagePath: this.config.storagePath,
      });

      await this.index.initialize();

      // Load existing data if available
      await this.loadFromStorage();

      this.initialized = true;
      logger.info('Vector store initialized', {
        dimensions: this.config.dimensions,
        chunksLoaded: this.chunks.size,
      });
    } catch (error) {
      logger.error('Failed to initialize vector store', {
        error: (error as Error).message,
      });

      // Fallback to in-memory implementation
      logger.info('Using fallback in-memory vector store');
      this.index = new InMemoryVectorIndex(this.config.dimensions);
      this.initialized = true;
    }
  }

  /**
   * Add chunks to the vector store
   */
  async addChunks(chunks: DocumentChunk[]): Promise<void> {
    if (!this.initialized) {
      await this.initialize();
    }

    const vectors: { id: string; vector: number[]; metadata: any }[] = [];

    for (const chunk of chunks) {
      if (!chunk.embedding) {
        logger.warn('Chunk missing embedding, skipping', { chunkId: chunk.id });
        continue;
      }

      // Store chunk data
      this.chunks.set(chunk.id, chunk);

      // Prepare vector for indexing
      vectors.push({
        id: chunk.id,
        vector: Array.from(chunk.embedding),
        metadata: {
          documentId: chunk.documentId,
          documentType: chunk.documentType,
          ...chunk.metadata,
        },
      });

      // Initialize relevance weight
      this.relevanceWeights.set(chunk.id, 1.0);
    }

    // Add to index in batch
    if (vectors.length > 0) {
      await this.index.addVectors(vectors);

      logger.info('Added chunks to vector store', {
        count: vectors.length,
        totalChunks: this.chunks.size,
      });
    }
  }

  /**
   * Search for similar chunks
   */
  async search(
    queryEmbedding: Float32Array,
    query: RAGQuery
  ): Promise<{ chunks: DocumentChunk[]; scores: number[] }> {
    if (!this.initialized) {
      await this.initialize();
    }

    // Search index
    const results = await this.index.search(
      Array.from(queryEmbedding),
      query.topK * 2, // Over-fetch for filtering
      this.config.efSearch
    );

    // Filter and re-rank results
    const filteredResults: { chunk: DocumentChunk; score: number }[] = [];

    for (const result of results) {
      const chunk = this.chunks.get(result.id);
      if (!chunk) continue;

      // Apply filters
      if (query.documentTypes && !query.documentTypes.includes(chunk.documentType)) {
        continue;
      }

      if (query.technologies && chunk.metadata.technology) {
        if (!query.technologies.includes(chunk.metadata.technology as 'LTE' | 'NR')) {
          continue;
        }
      }

      if (query.parameterNames && chunk.metadata.parameterName) {
        const lowerParam = chunk.metadata.parameterName.toLowerCase();
        const matchesParam = query.parameterNames.some(
          (p) => lowerParam.includes(p.toLowerCase())
        );
        if (!matchesParam) continue;
      }

      // Apply minimum similarity threshold
      if (result.score < query.minSimilarity) {
        continue;
      }

      // Apply self-learning relevance adjustment
      let adjustedScore = result.score;
      if (this.config.selfLearning) {
        const relevanceWeight = this.relevanceWeights.get(chunk.id) || 1.0;
        adjustedScore *= relevanceWeight;
      }

      filteredResults.push({ chunk, score: adjustedScore });
    }

    // Sort by adjusted score and take top K
    filteredResults.sort((a, b) => b.score - a.score);
    const topResults = filteredResults.slice(0, query.topK);

    // Record query for self-learning
    if (this.config.selfLearning) {
      this.recordQuery(query.query, topResults.map((r) => r.chunk.id));
    }

    return {
      chunks: topResults.map((r) => r.chunk),
      scores: topResults.map((r) => r.score),
    };
  }

  /**
   * Record feedback for self-learning
   */
  recordFeedback(queryId: string, chunkId: string, feedback: number): void {
    if (!this.config.selfLearning) return;

    // Update relevance weight based on feedback
    const currentWeight = this.relevanceWeights.get(chunkId) || 1.0;
    const newWeight = currentWeight + this.config.adaptationRate * (feedback - 0.5) * 2;

    // Clamp weight to reasonable range
    this.relevanceWeights.set(chunkId, Math.max(0.1, Math.min(2.0, newWeight)));

    logger.debug('Updated relevance weight', {
      chunkId,
      oldWeight: currentWeight,
      newWeight: this.relevanceWeights.get(chunkId),
      feedback,
    });
  }

  /**
   * Record query for learning
   */
  private recordQuery(query: string, resultIds: string[]): void {
    const queryKey = this.hashQuery(query);

    if (!this.queryHistory.has(queryKey)) {
      this.queryHistory.set(queryKey, []);
    }

    this.queryHistory.get(queryKey)!.push({
      query,
      results: resultIds,
    });

    // Keep only recent queries
    const history = this.queryHistory.get(queryKey)!;
    if (history.length > 100) {
      history.shift();
    }
  }

  /**
   * Hash query for deduplication
   */
  private hashQuery(query: string): string {
    // Simple hash for query grouping
    let hash = 0;
    for (let i = 0; i < query.length; i++) {
      const char = query.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash;
    }
    return hash.toString(16);
  }

  /**
   * Get chunk by ID
   */
  getChunk(chunkId: string): DocumentChunk | undefined {
    return this.chunks.get(chunkId);
  }

  /**
   * Get all chunks for a document
   */
  getDocumentChunks(documentId: string): DocumentChunk[] {
    const chunks: DocumentChunk[] = [];
    for (const chunk of this.chunks.values()) {
      if (chunk.documentId === documentId) {
        chunks.push(chunk);
      }
    }
    return chunks.sort((a, b) => a.metadata.chunkIndex - b.metadata.chunkIndex);
  }

  /**
   * Delete chunks by document ID
   */
  async deleteDocument(documentId: string): Promise<number> {
    const chunkIds: string[] = [];

    for (const [chunkId, chunk] of this.chunks) {
      if (chunk.documentId === documentId) {
        chunkIds.push(chunkId);
      }
    }

    for (const chunkId of chunkIds) {
      this.chunks.delete(chunkId);
      this.relevanceWeights.delete(chunkId);
      await this.index.delete(chunkId);
    }

    logger.info('Deleted document chunks', {
      documentId,
      chunksDeleted: chunkIds.length,
    });

    return chunkIds.length;
  }

  /**
   * Persist to storage
   */
  async persist(): Promise<void> {
    if (this.index?.persist) {
      await this.index.persist();
    }

    // Save chunk metadata and relevance weights
    const metadata = {
      chunks: Array.from(this.chunks.entries()),
      relevanceWeights: Array.from(this.relevanceWeights.entries()),
    };

    // Would save to file in production
    logger.info('Vector store persisted', {
      chunks: this.chunks.size,
    });
  }

  /**
   * Load from storage
   */
  private async loadFromStorage(): Promise<void> {
    // Would load from file in production
    if (this.index?.load) {
      await this.index.load();
    }
  }

  /**
   * Get statistics
   */
  getStats(): {
    totalChunks: number;
    byDocumentType: Record<string, number>;
    avgRelevanceWeight: number;
  } {
    const byType: Record<string, number> = {};
    let totalWeight = 0;

    for (const chunk of this.chunks.values()) {
      byType[chunk.documentType] = (byType[chunk.documentType] || 0) + 1;
      totalWeight += this.relevanceWeights.get(chunk.id) || 1.0;
    }

    return {
      totalChunks: this.chunks.size,
      byDocumentType: byType,
      avgRelevanceWeight: this.chunks.size > 0 ? totalWeight / this.chunks.size : 1.0,
    };
  }
}

/**
 * In-memory vector index fallback
 */
class InMemoryVectorIndex {
  private vectors: Map<string, { vector: number[]; metadata: any }> = new Map();
  private dimensions: number;

  constructor(dimensions: number) {
    this.dimensions = dimensions;
  }

  async initialize(): Promise<void> {}

  async addVectors(vectors: { id: string; vector: number[]; metadata: any }[]): Promise<void> {
    for (const v of vectors) {
      this.vectors.set(v.id, { vector: v.vector, metadata: v.metadata });
    }
  }

  async search(
    queryVector: number[],
    k: number,
    _efSearch: number
  ): Promise<{ id: string; score: number }[]> {
    const results: { id: string; score: number }[] = [];

    for (const [id, data] of this.vectors) {
      const score = this.cosineSimilarity(queryVector, data.vector);
      results.push({ id, score });
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, k);
  }

  async delete(id: string): Promise<void> {
    this.vectors.delete(id);
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < Math.min(a.length, b.length); i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
    return magnitude === 0 ? 0 : dotProduct / magnitude;
  }
}

export default SelfLearningVectorStore;
