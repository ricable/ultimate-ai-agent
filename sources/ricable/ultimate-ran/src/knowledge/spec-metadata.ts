/**
 * 3GPP Spec Metadata Integration with AgentDB
 *
 * Integrates OrganizedProgrammers/3GPPSpecMetadata dataset with AgentDB
 * for semantic search, dependency tracking, and RAG-based 3GPP compliance.
 *
 * Provides:
 * - Schema for 3GPP spec metadata storage
 * - Vector embeddings for semantic search (<10ms latency)
 * - Graph-based dependency queries
 * - Section-level retrieval for RAG
 *
 * @module knowledge/spec-metadata
 * @version 7.0.0-alpha.1
 */

import { EventEmitter } from 'events';

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * 3GPP Technical Specification metadata
 */
export interface ThreeGPPSpec {
  /** Spec number (e.g., "TS 38.331", "TS 28.552") */
  specNumber: string;

  /** Version string (e.g., "17.4.0") */
  version: string;

  /** Release version (e.g., "Rel-17", "Rel-18") */
  release: string;

  /** Full title of the specification */
  title: string;

  /** Working group (e.g., "RAN2", "SA5", "CT4") */
  workingGroup: string;

  /** Current status */
  status: 'active' | 'withdrawn' | 'draft' | 'frozen';

  /** Scope/abstract of the specification */
  scope: string;

  /** Keywords for semantic search */
  keywords: string[];

  /** Referenced/dependent spec numbers */
  dependencies: string[];

  /** Last update timestamp */
  lastUpdate: Date;

  /** Embedding vector for semantic search */
  embedding?: number[];

  /** Additional metadata */
  metadata?: {
    /** Download URL */
    downloadUrl?: string;
    /** File size in bytes */
    fileSize?: number;
    /** ETSI document reference */
    etsiRef?: string;
    /** Primary domain (RAN, CN, SA, etc.) */
    domain?: 'RAN' | 'CN' | 'SA' | 'CT' | 'SEC';
  };
}

/**
 * Spec section with detailed content
 */
export interface SpecSection {
  /** Parent spec number */
  specNumber: string;

  /** Section number (e.g., "5.3.2.1") */
  sectionNumber: string;

  /** Section title */
  title: string;

  /** Section content/text */
  content: string;

  /** Tables within this section */
  tables: TableDefinition[];

  /** Figures/diagrams */
  figures: FigureDefinition[];

  /** ASN.1 code blocks */
  asn1Blocks: ASN1Block[];

  /** Embedding for section content */
  embedding?: number[];

  /** Section level (depth) */
  level: number;

  /** Parent section number */
  parentSection?: string;

  /** Child section numbers */
  childSections?: string[];
}

/**
 * Table definition from spec
 */
export interface TableDefinition {
  /** Table identifier */
  tableId: string;

  /** Table caption/title */
  caption: string;

  /** Column headers */
  headers: string[];

  /** Table rows (array of arrays) */
  rows: string[][];

  /** Notes/footnotes */
  notes?: string[];
}

/**
 * Figure/diagram definition
 */
export interface FigureDefinition {
  /** Figure identifier */
  figureId: string;

  /** Figure caption */
  caption: string;

  /** Description text */
  description?: string;

  /** Image URL or data URI */
  imageUrl?: string;

  /** Figure type */
  type?: 'diagram' | 'chart' | 'flowchart' | 'state-machine' | 'sequence';
}

/**
 * ASN.1 code block
 */
export interface ASN1Block {
  /** Block identifier */
  blockId: string;

  /** ASN.1 module name */
  moduleName?: string;

  /** ASN.1 code */
  code: string;

  /** Defined types in this block */
  definedTypes?: string[];

  /** Referenced types */
  referencedTypes?: string[];
}

/**
 * Spec search result
 */
export interface SpecSearchResult {
  /** The matched spec or section */
  item: ThreeGPPSpec | SpecSection;

  /** Type of result */
  type: 'spec' | 'section';

  /** Similarity score (0-1) */
  similarity: number;

  /** Distance metric */
  distance: number;

  /** Highlighted excerpt */
  excerpt?: string;
}

/**
 * Query filters for spec search
 */
export interface SpecQueryFilters {
  /** Filter by release */
  release?: string;

  /** Filter by working group */
  workingGroup?: string;

  /** Filter by status */
  status?: 'active' | 'withdrawn' | 'draft' | 'frozen';

  /** Filter by domain */
  domain?: 'RAN' | 'CN' | 'SA' | 'CT' | 'SEC';

  /** Only return specs updated after this date */
  updatedAfter?: Date;

  /** Minimum similarity threshold (0-1) */
  minSimilarity?: number;
}

/**
 * Dependency graph node
 */
export interface DependencyNode {
  /** Spec number */
  specNumber: string;

  /** Direct dependencies */
  dependencies: string[];

  /** Specs that depend on this one */
  dependents: string[];

  /** Depth in dependency tree */
  depth: number;
}

// ============================================================================
// Spec Metadata Store
// ============================================================================

/**
 * SpecMetadataStore - AgentDB integration for 3GPP specs
 *
 * Provides fast semantic search over 3GPP specifications with
 * <10ms latency using HNSW vector indices.
 */
export class SpecMetadataStore extends EventEmitter {
  private dimension: number = 768;  // bge-base-en-v1.5 embeddings
  private dbPath: string;

  // In-memory caches
  private specs: Map<string, ThreeGPPSpec>;
  private sections: Map<string, SpecSection>;
  private dependencies: Map<string, DependencyNode>;

  // HNSW indices
  private specIndex: HNSWIndex;
  private sectionIndex: HNSWIndex;

  // Performance tracking
  private searchLatencies: number[];

  constructor(dbPath: string = './titan-ran.db') {
    super();

    this.dbPath = dbPath;
    this.specs = new Map();
    this.sections = new Map();
    this.dependencies = new Map();
    this.searchLatencies = [];

    // Initialize HNSW indices with RAN-optimized parameters
    this.specIndex = new HNSWIndex({
      dimension: this.dimension,
      metric: 'cosine',
      M: 32,              // Max connections per layer
      efConstruction: 200, // Construction quality
      efSearch: 100        // Search speed/quality balance
    });

    this.sectionIndex = new HNSWIndex({
      dimension: this.dimension,
      metric: 'cosine',
      M: 48,               // Higher for section granularity
      efConstruction: 300,
      efSearch: 150
    });

    console.log('[SpecMetadata] Initialized 3GPP spec metadata store');
    console.log(`[SpecMetadata] Database: ${this.dbPath}`);
  }

  /**
   * Initialize the store and load existing specs
   */
  async initialize(): Promise<void> {
    console.log('[SpecMetadata] Initializing spec metadata store...');

    // In production: load from AgentDB
    // npx agentdb@alpha query --db ./titan-ran.db --table specs

    this.emit('initialized');
    console.log('[SpecMetadata] Initialization complete');
  }

  // ========================================================================
  // Indexing Operations
  // ========================================================================

  /**
   * Index a 3GPP specification with embeddings
   *
   * @param spec - The spec to index
   */
  async indexSpec(spec: ThreeGPPSpec): Promise<void> {
    const startTime = performance.now();

    // Generate embedding if not provided
    if (!spec.embedding) {
      spec.embedding = await this.generateSpecEmbedding(spec);
    }

    // Validate embedding dimension
    if (spec.embedding.length !== this.dimension) {
      throw new Error(`Embedding dimension mismatch: expected ${this.dimension}, got ${spec.embedding.length}`);
    }

    // Store in memory cache
    this.specs.set(spec.specNumber, spec);

    // Add to HNSW index
    await this.specIndex.insert(spec.specNumber, new Float32Array(spec.embedding));

    // Update dependency graph
    this.updateDependencyGraph(spec);

    const latency = performance.now() - startTime;
    console.log(`[SpecMetadata] Indexed spec ${spec.specNumber} in ${latency.toFixed(2)}ms`);

    this.emit('spec_indexed', {
      specNumber: spec.specNumber,
      latency
    });
  }

  /**
   * Index a spec section with embeddings
   *
   * @param section - The section to index
   */
  async indexSection(section: SpecSection): Promise<void> {
    const startTime = performance.now();

    // Generate embedding if not provided
    if (!section.embedding) {
      section.embedding = await this.generateSectionEmbedding(section);
    }

    // Validate dimension
    if (section.embedding.length !== this.dimension) {
      throw new Error(`Embedding dimension mismatch: expected ${this.dimension}, got ${section.embedding.length}`);
    }

    // Create unique ID for section
    const sectionId = `${section.specNumber}#${section.sectionNumber}`;

    // Store in memory
    this.sections.set(sectionId, section);

    // Add to HNSW index
    await this.sectionIndex.insert(sectionId, new Float32Array(section.embedding));

    const latency = performance.now() - startTime;
    console.log(`[SpecMetadata] Indexed section ${sectionId} in ${latency.toFixed(2)}ms`);

    this.emit('section_indexed', {
      sectionId,
      latency
    });
  }

  /**
   * Bulk index multiple specs (efficient batch operation)
   *
   * @param specs - Array of specs to index
   */
  async bulkIndexSpecs(specs: ThreeGPPSpec[]): Promise<void> {
    const startTime = performance.now();

    console.log(`[SpecMetadata] Bulk indexing ${specs.length} specs...`);

    for (const spec of specs) {
      await this.indexSpec(spec);
    }

    const latency = performance.now() - startTime;
    const avgLatency = latency / specs.length;

    console.log(`[SpecMetadata] Bulk indexed ${specs.length} specs in ${latency.toFixed(2)}ms (avg: ${avgLatency.toFixed(2)}ms/spec)`);

    this.emit('bulk_index_complete', {
      count: specs.length,
      totalLatency: latency,
      avgLatency
    });
  }

  /**
   * Bulk index multiple sections
   *
   * @param sections - Array of sections to index
   */
  async bulkIndexSections(sections: SpecSection[]): Promise<void> {
    const startTime = performance.now();

    console.log(`[SpecMetadata] Bulk indexing ${sections.length} sections...`);

    for (const section of sections) {
      await this.indexSection(section);
    }

    const latency = performance.now() - startTime;
    const avgLatency = latency / sections.length;

    console.log(`[SpecMetadata] Bulk indexed ${sections.length} sections in ${latency.toFixed(2)}ms (avg: ${avgLatency.toFixed(2)}ms/section)`);

    this.emit('bulk_section_index_complete', {
      count: sections.length,
      totalLatency: latency,
      avgLatency
    });
  }

  // ========================================================================
  // Query Operations
  // ========================================================================

  /**
   * Find relevant specs using semantic similarity
   *
   * Performance target: <10ms
   *
   * @param query - Natural language query
   * @param k - Number of results to return
   * @param filters - Optional filters
   */
  async findRelevantSpecs(
    query: string,
    k: number = 5,
    filters?: SpecQueryFilters
  ): Promise<SpecSearchResult[]> {
    const startTime = performance.now();

    // Generate query embedding
    const queryEmbedding = await this.generateQueryEmbedding(query);

    // Search spec index
    const candidates = await this.specIndex.search(
      new Float32Array(queryEmbedding),
      k * 2  // Get more candidates for filtering
    );

    // Map to specs and apply filters
    const mappedResults = candidates
      .map(c => {
        const spec = this.specs.get(c.id);
        if (!spec) return null;

        return {
          item: spec,
          type: 'spec' as const,
          similarity: this.distanceToSimilarity(c.distance),
          distance: c.distance,
          excerpt: this.generateSpecExcerpt(spec, query)
        };
      })
      .filter(r => r !== null);

    let results: SpecSearchResult[] = mappedResults as SpecSearchResult[];

    // Apply filters
    if (filters) {
      results = this.applyFilters(results, filters);
    }

    // Take top k
    results = results.slice(0, k);

    const latency = performance.now() - startTime;
    this.searchLatencies.push(latency);

    console.log(`[SpecMetadata] Found ${results.length} relevant specs in ${latency.toFixed(2)}ms`);

    if (latency > 10) {
      console.warn(`[SpecMetadata] WARNING: Search latency ${latency.toFixed(2)}ms exceeds 10ms target`);
    }

    this.emit('search_complete', {
      query,
      resultsCount: results.length,
      latency
    });

    return results;
  }

  /**
   * Find relevant sections within specs
   *
   * @param query - Natural language query
   * @param specId - Optional spec number to limit search
   * @param k - Number of results
   */
  async findRelevantSections(
    query: string,
    specId?: string,
    k: number = 5
  ): Promise<SpecSearchResult[]> {
    const startTime = performance.now();

    // Generate query embedding
    const queryEmbedding = await this.generateQueryEmbedding(query);

    // Search section index
    const candidates = await this.sectionIndex.search(
      new Float32Array(queryEmbedding),
      k * 2
    );

    // Map to sections and filter by specId if provided
    const mappedResults = candidates
      .map(c => {
        const section = this.sections.get(c.id);
        if (!section) return null;

        // Filter by specId if provided
        if (specId && section.specNumber !== specId) {
          return null;
        }

        return {
          item: section,
          type: 'section' as const,
          similarity: this.distanceToSimilarity(c.distance),
          distance: c.distance,
          excerpt: this.generateSectionExcerpt(section, query)
        };
      })
      .filter(r => r !== null);

    let results: SpecSearchResult[] = mappedResults as SpecSearchResult[];

    // Take top k
    results = results.slice(0, k);

    const latency = performance.now() - startTime;

    console.log(`[SpecMetadata] Found ${results.length} relevant sections in ${latency.toFixed(2)}ms`);

    this.emit('section_search_complete', {
      query,
      specId,
      resultsCount: results.length,
      latency
    });

    return results;
  }

  // ========================================================================
  // Graph-based Queries
  // ========================================================================

  /**
   * Get dependency tree for a spec (transitive dependencies)
   *
   * @param specId - Spec number
   * @param maxDepth - Maximum depth to traverse (default: 3)
   */
  async getDependencyTree(
    specId: string,
    maxDepth: number = 3
  ): Promise<ThreeGPPSpec[]> {
    const visited = new Set<string>();
    const result: ThreeGPPSpec[] = [];

    const traverse = (currentId: string, depth: number) => {
      if (depth > maxDepth || visited.has(currentId)) return;

      visited.add(currentId);
      const spec = this.specs.get(currentId);

      if (!spec) return;

      result.push(spec);

      // Traverse dependencies
      for (const depId of spec.dependencies) {
        traverse(depId, depth + 1);
      }
    };

    traverse(specId, 0);

    console.log(`[SpecMetadata] Found ${result.length} dependencies for ${specId}`);

    return result;
  }

  /**
   * Get specs that reference this spec (reverse dependencies)
   *
   * @param specId - Spec number
   */
  async getReferencingSpecs(specId: string): Promise<ThreeGPPSpec[]> {
    const node = this.dependencies.get(specId);

    if (!node) {
      console.warn(`[SpecMetadata] No dependency node found for ${specId}`);
      return [];
    }

    const referencingSpecs = node.dependents
      .map(id => this.specs.get(id))
      .filter((spec): spec is ThreeGPPSpec => spec !== undefined);

    console.log(`[SpecMetadata] Found ${referencingSpecs.length} specs referencing ${specId}`);

    return referencingSpecs;
  }

  /**
   * Find circular dependencies in spec graph
   */
  detectCircularDependencies(): Array<string[]> {
    const cycles: Array<string[]> = [];
    const visited = new Set<string>();
    const recursionStack = new Set<string>();

    const dfs = (specId: string, path: string[]): void => {
      visited.add(specId);
      recursionStack.add(specId);
      path.push(specId);

      const spec = this.specs.get(specId);
      if (!spec) return;

      for (const depId of spec.dependencies) {
        if (!visited.has(depId)) {
          dfs(depId, [...path]);
        } else if (recursionStack.has(depId)) {
          // Found cycle
          const cycleStart = path.indexOf(depId);
          cycles.push(path.slice(cycleStart));
        }
      }

      recursionStack.delete(specId);
    };

    for (const specId of this.specs.keys()) {
      if (!visited.has(specId)) {
        dfs(specId, []);
      }
    }

    if (cycles.length > 0) {
      console.warn(`[SpecMetadata] Found ${cycles.length} circular dependencies`);
    }

    return cycles;
  }

  // ========================================================================
  // Statistics and Utilities
  // ========================================================================

  /**
   * Get statistics about indexed specs
   */
  getStats(): {
    specsCount: number;
    sectionsCount: number;
    avgDependencies: number;
    avgSearchLatency: number;
    releases: string[];
    workingGroups: string[];
  } {
    const specs = Array.from(this.specs.values());

    const avgDeps = specs.length > 0
      ? specs.reduce((sum, s) => sum + s.dependencies.length, 0) / specs.length
      : 0;

    const releases = [...new Set(specs.map(s => s.release))];
    const workingGroups = [...new Set(specs.map(s => s.workingGroup))];

    const avgLatency = this.searchLatencies.length > 0
      ? this.searchLatencies.reduce((sum, lat) => sum + lat, 0) / this.searchLatencies.length
      : 0;

    return {
      specsCount: this.specs.size,
      sectionsCount: this.sections.size,
      avgDependencies: avgDeps,
      avgSearchLatency: avgLatency,
      releases: releases.sort(),
      workingGroups: workingGroups.sort()
    };
  }

  /**
   * Get a spec by its number
   */
  getSpec(specNumber: string): ThreeGPPSpec | undefined {
    return this.specs.get(specNumber);
  }

  /**
   * Get a section by spec number and section number
   */
  getSection(specNumber: string, sectionNumber: string): SpecSection | undefined {
    const sectionId = `${specNumber}#${sectionNumber}`;
    return this.sections.get(sectionId);
  }

  // ========================================================================
  // Private Helper Methods
  // ========================================================================

  /**
   * Generate embedding for a spec (title + scope + keywords)
   */
  private async generateSpecEmbedding(spec: ThreeGPPSpec): Promise<number[]> {
    // Combine spec metadata into text for embedding
    const text = [
      spec.specNumber,
      spec.title,
      spec.scope,
      ...spec.keywords,
      spec.workingGroup,
      spec.release
    ].join(' ');

    // In production: call embedding service (e.g., bge-base-en-v1.5)
    // For now: generate mock embedding
    return this.generateMockEmbedding(text);
  }

  /**
   * Generate embedding for a section (chunked content)
   */
  private async generateSectionEmbedding(section: SpecSection): Promise<number[]> {
    // Chunk section content (max 512 tokens)
    const text = this.chunkText(
      [
        section.specNumber,
        section.sectionNumber,
        section.title,
        section.content
      ].join(' '),
      512
    );

    // In production: call embedding service
    return this.generateMockEmbedding(text);
  }

  /**
   * Generate embedding for a query
   */
  private async generateQueryEmbedding(query: string): Promise<number[]> {
    // In production: call embedding service
    return this.generateMockEmbedding(query);
  }

  /**
   * Generate mock embedding (for development)
   */
  private generateMockEmbedding(text: string): number[] {
    const embedding = new Array(this.dimension);

    // Simple hash-based mock embedding
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      hash = ((hash << 5) - hash) + text.charCodeAt(i);
      hash = hash & hash;
    }

    for (let i = 0; i < this.dimension; i++) {
      embedding[i] = Math.sin(hash * (i + 1)) * 0.5 + 0.5;
    }

    // Normalize
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    for (let i = 0; i < this.dimension; i++) {
      embedding[i] /= norm;
    }

    return embedding;
  }

  /**
   * Chunk text to max tokens (simple word-based chunking)
   */
  private chunkText(text: string, maxTokens: number): string {
    const words = text.split(/\s+/);
    return words.slice(0, maxTokens).join(' ');
  }

  /**
   * Convert distance to similarity score (0-1)
   */
  private distanceToSimilarity(distance: number): number {
    // Cosine distance: 1 - cosine_similarity
    return Math.max(0, Math.min(1, 1 - distance));
  }

  /**
   * Generate excerpt for spec search result
   */
  private generateSpecExcerpt(spec: ThreeGPPSpec, query: string): string {
    // Return first 200 chars of scope
    return spec.scope.substring(0, 200) + (spec.scope.length > 200 ? '...' : '');
  }

  /**
   * Generate excerpt for section search result
   */
  private generateSectionExcerpt(section: SpecSection, query: string): string {
    // Return first 200 chars of content
    return section.content.substring(0, 200) + (section.content.length > 200 ? '...' : '');
  }

  /**
   * Apply filters to search results
   */
  private applyFilters(
    results: SpecSearchResult[],
    filters: SpecQueryFilters
  ): SpecSearchResult[] {
    return results.filter(r => {
      if (r.type !== 'spec') return true;

      const spec = r.item as ThreeGPPSpec;

      if (filters.release && spec.release !== filters.release) {
        return false;
      }

      if (filters.workingGroup && spec.workingGroup !== filters.workingGroup) {
        return false;
      }

      if (filters.status && spec.status !== filters.status) {
        return false;
      }

      if (filters.domain && spec.metadata?.domain !== filters.domain) {
        return false;
      }

      if (filters.updatedAfter && spec.lastUpdate < filters.updatedAfter) {
        return false;
      }

      if (filters.minSimilarity && r.similarity < filters.minSimilarity) {
        return false;
      }

      return true;
    });
  }

  /**
   * Update dependency graph with new spec
   */
  private updateDependencyGraph(spec: ThreeGPPSpec): void {
    // Create or update node for this spec
    if (!this.dependencies.has(spec.specNumber)) {
      this.dependencies.set(spec.specNumber, {
        specNumber: spec.specNumber,
        dependencies: spec.dependencies,
        dependents: [],
        depth: 0
      });
    }

    const node = this.dependencies.get(spec.specNumber)!;
    node.dependencies = spec.dependencies;

    // Update dependent references
    for (const depId of spec.dependencies) {
      if (!this.dependencies.has(depId)) {
        this.dependencies.set(depId, {
          specNumber: depId,
          dependencies: [],
          dependents: [spec.specNumber],
          depth: 0
        });
      } else {
        const depNode = this.dependencies.get(depId)!;
        if (!depNode.dependents.includes(spec.specNumber)) {
          depNode.dependents.push(spec.specNumber);
        }
      }
    }
  }
}

// ============================================================================
// HNSW Index (Simplified Implementation)
// ============================================================================

interface HNSWConfig {
  dimension: number;
  metric: 'cosine' | 'euclidean' | 'dotproduct';
  M: number;
  efConstruction: number;
  efSearch: number;
}

class HNSWIndex {
  private config: HNSWConfig;
  private vectors: Map<string, Float32Array>;
  private graph: Map<string, Set<string>>;

  constructor(config: HNSWConfig) {
    this.config = config;
    this.vectors = new Map();
    this.graph = new Map();
  }

  async insert(id: string, vector: Float32Array): Promise<void> {
    this.vectors.set(id, vector);

    if (!this.graph.has(id)) {
      this.graph.set(id, new Set());
    }

    // Connect to nearest neighbors
    const neighbors = await this.search(vector, this.config.M);
    for (const neighbor of neighbors) {
      if (neighbor.id !== id) {
        this.graph.get(id)!.add(neighbor.id);
        this.graph.get(neighbor.id)?.add(id);
      }
    }
  }

  async search(query: Float32Array, k: number): Promise<Array<{ id: string; distance: number }>> {
    const results: Array<{ id: string; distance: number }> = [];

    for (const [id, vector] of this.vectors) {
      const distance = this.distance(query, vector);
      results.push({ id, distance });
    }

    results.sort((a, b) => a.distance - b.distance);
    return results.slice(0, k);
  }

  private distance(a: Float32Array, b: Float32Array): number {
    if (this.config.metric === 'cosine') {
      let dot = 0, normA = 0, normB = 0;
      for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
      }
      return 1 - (dot / (Math.sqrt(normA) * Math.sqrt(normB)));
    }

    // Euclidean
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += (a[i] - b[i]) ** 2;
    }
    return Math.sqrt(sum);
  }
}

// ============================================================================
// Exports
// ============================================================================

// export {
//   SpecMetadataStore,
//   type ThreeGPPSpec,
//   type SpecSection,
//   type TableDefinition,
//   type FigureDefinition,
//   type ASN1Block,
//   type SpecSearchResult,
//   type SpecQueryFilters,
//   type DependencyNode
// };
