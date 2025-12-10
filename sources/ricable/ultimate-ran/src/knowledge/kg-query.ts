/**
 * Knowledge Graph Query Interface with RuvLLM Integration
 *
 * Provides natural language and structured query interface for 3GPP spec knowledge graph.
 * Integrates with RuvVector for semantic search and RuvLLM for natural language understanding.
 *
 * Features:
 * - Natural language to graph query translation
 * - Cypher-like and SPARQL-like query syntax
 * - Graph traversal with pattern matching
 * - Semantic understanding via RuvLLM
 * - Cross-spec analysis and relationship explanation
 *
 * @module knowledge/kg-query
 * @version 7.0.0-alpha.1
 */

import { EventEmitter } from 'events';

// ============================================================================
// Type Definitions - Graph Data Model
// ============================================================================

/**
 * Base node in the knowledge graph (3GPP specs, parameters, IEs)
 */
export interface GraphMLNode {
  id: string;
  type: 'spec' | 'section' | 'parameter' | 'ie' | 'procedure' | 'concept';
  label: string;
  properties: Record<string, any>;
  embedding?: Float32Array;  // Vector embedding for semantic search
  metadata: {
    source?: string;  // e.g., "TS 38.331", "TS 28.552"
    version?: string;
    section?: string;
    tags?: string[];
  };
}

/**
 * Edge representing relationship between nodes
 */
export interface GraphMLEdge {
  id: string;
  source: string;  // Node ID
  target: string;  // Node ID
  type: string;    // e.g., "contains", "references", "controls", "affects"
  properties: Record<string, any>;
  weight?: number;
}

/**
 * Complete knowledge graph structure
 */
export interface KnowledgeGraph {
  nodes: Map<string, GraphMLNode>;
  edges: Map<string, GraphMLEdge[]>;  // Adjacency list: nodeId -> outgoing edges
  reverseEdges: Map<string, GraphMLEdge[]>;  // Reverse adjacency: nodeId -> incoming edges
  metadata: {
    totalNodes: number;
    totalEdges: number;
    specs: string[];
    lastUpdated: Date;
  };
}

/**
 * 3GPP Specification section
 */
export interface SpecSection {
  id: string;
  specId: string;  // e.g., "TS-38.331"
  section: string;  // e.g., "6.2.2"
  title: string;
  content: string;
  subsections: string[];
  parameters: string[];  // Parameter IDs
  procedures: string[];  // Procedure IDs
  references: string[];  // References to other sections
}

// ============================================================================
// Type Definitions - Query Results
// ============================================================================

/**
 * Result from natural language query
 */
export interface QueryResult {
  query: string;
  answer: string;
  nodes: GraphMLNode[];
  edges: GraphMLEdge[];
  paths?: GraphPath[];
  confidence: number;
  reasoning: string;
  executionTime: number;
  sourceSpecs: string[];
}

/**
 * Result from Cypher-like query
 */
export interface CypherResult {
  query: string;
  matches: CypherMatch[];
  totalMatches: number;
  executionTime: number;
}

export interface CypherMatch {
  nodes: Map<string, GraphMLNode>;  // Variable name -> node
  edges: Map<string, GraphMLEdge>;   // Variable name -> edge
  properties: Record<string, any>;    // Extracted properties
}

/**
 * Result from SPARQL-like query
 */
export interface SPARQLResult {
  query: string;
  bindings: SPARQLBinding[];
  totalBindings: number;
  executionTime: number;
}

export interface SPARQLBinding {
  [variable: string]: GraphMLNode | string | number | boolean;
}

/**
 * Path through the knowledge graph
 */
export interface GraphPath {
  nodes: GraphMLNode[];
  edges: GraphMLEdge[];
  length: number;
  weight: number;
  explanation?: string;
}

// ============================================================================
// Type Definitions - Query Patterns
// ============================================================================

/**
 * Traversal pattern for graph exploration
 */
export interface TraversalPattern {
  direction: 'outgoing' | 'incoming' | 'both';
  edgeTypes?: string[];
  maxDepth: number;
  filter?: (node: GraphMLNode) => boolean;
  collectPaths?: boolean;
}

/**
 * Parsed query from natural language
 */
export interface ParsedQuery {
  intent: 'find' | 'relate' | 'compare' | 'explain' | 'list';
  entities: string[];  // Extracted entities (e.g., "P0", "SINR", "RRCReconfiguration")
  relationships: string[];  // Extracted relationships
  constraints: QueryConstraint[];
  specs: string[];  // Relevant specs
  traversalPattern?: TraversalPattern;
}

export interface QueryConstraint {
  field: string;
  operator: 'equals' | 'contains' | 'greaterThan' | 'lessThan' | 'matches';
  value: any;
}

// ============================================================================
// KG Query Interface - Main Query Engine
// ============================================================================

/**
 * Natural language and structured query interface for knowledge graph
 */
export class KGQueryInterface extends EventEmitter {
  private graph: KnowledgeGraph;
  private ruvllmBridge: KGRuvLLMBridge;
  private indexPath: string;

  constructor(graph: KnowledgeGraph, indexPath: string = './ruvector-kg.db') {
    super();
    this.graph = graph;
    this.indexPath = indexPath;
    this.ruvllmBridge = new KGRuvLLMBridge(indexPath);

    console.log('[KG-Query] Initialized knowledge graph query interface');
    console.log(`[KG-Query] Total nodes: ${graph.metadata.totalNodes}`);
    console.log(`[KG-Query] Total edges: ${graph.metadata.totalEdges}`);
    console.log(`[KG-Query] Specs: ${graph.metadata.specs.join(', ')}`);
  }

  /**
   * Natural language query interface
   *
   * @example
   * await query("What parameters control uplink power in 5G NR?")
   * await query("How does P0 relate to SINR?")
   * await query("List all IEs in RRCReconfiguration")
   */
  async query(question: string): Promise<QueryResult> {
    const startTime = performance.now();

    console.log(`[KG-Query] Processing natural language query: "${question}"`);

    // Parse question using RuvLLM
    const parsedQuery = await this.ruvllmBridge.parseQuestion(question);
    console.log(`[KG-Query] Parsed intent: ${parsedQuery.intent}`);
    console.log(`[KG-Query] Entities: ${parsedQuery.entities.join(', ')}`);

    // Execute graph traversal based on intent
    let nodes: GraphMLNode[] = [];
    let edges: GraphMLEdge[] = [];
    let paths: GraphPath[] | undefined;

    switch (parsedQuery.intent) {
      case 'find':
        ({ nodes, edges } = await this.executeFindQuery(parsedQuery));
        break;

      case 'relate':
        ({ nodes, edges, paths } = await this.executeRelateQuery(parsedQuery));
        break;

      case 'compare':
        ({ nodes, edges } = await this.executeCompareQuery(parsedQuery));
        break;

      case 'explain':
        ({ nodes, edges, paths } = await this.executeExplainQuery(parsedQuery));
        break;

      case 'list':
        ({ nodes, edges } = await this.executeListQuery(parsedQuery));
        break;

      default:
        throw new Error(`Unknown query intent: ${parsedQuery.intent}`);
    }

    // Generate answer using RuvLLM
    const sourceSpecs = this.extractSourceSpecs(nodes);
    const answer = await this.ruvllmBridge.generateAnswer(question, nodes, sourceSpecs);

    const executionTime = performance.now() - startTime;

    const result: QueryResult = {
      query: question,
      answer,
      nodes,
      edges,
      paths,
      confidence: this.calculateConfidence(nodes, edges),
      reasoning: this.generateReasoning(parsedQuery, nodes, edges),
      executionTime,
      sourceSpecs
    };

    this.emit('query-complete', result);
    return result;
  }

  /**
   * Execute Cypher-like query
   *
   * @example
   * await cypher("MATCH (p:Parameter)-[:CONTROLS]->(m:Metric) WHERE p.name = 'P0' RETURN p, m")
   * await cypher("MATCH (s:Spec)-[:CONTAINS]->(sec:Section) WHERE s.id = 'TS-38.331' RETURN sec")
   */
  async cypher(query: string): Promise<CypherResult> {
    const startTime = performance.now();

    console.log(`[KG-Query] Executing Cypher query: ${query}`);

    // Parse Cypher query
    const pattern = this.parseCypherQuery(query);

    // Execute pattern matching
    const matches = await this.executePatternMatch(pattern);

    const executionTime = performance.now() - startTime;

    return {
      query,
      matches,
      totalMatches: matches.length,
      executionTime
    };
  }

  /**
   * Execute SPARQL-like query
   *
   * @example
   * await sparql("SELECT ?param WHERE { ?param rdf:type 'Parameter' . ?param controls ?metric }")
   * await sparql("SELECT ?ie WHERE { ?msg contains ?ie . ?msg rdf:type 'RRCReconfiguration' }")
   */
  async sparql(query: string): Promise<SPARQLResult> {
    const startTime = performance.now();

    console.log(`[KG-Query] Executing SPARQL query: ${query}`);

    // Parse SPARQL query
    const pattern = this.parseSPARQLQuery(query);

    // Execute triple pattern matching
    const bindings = await this.executeTripleMatch(pattern);

    const executionTime = performance.now() - startTime;

    return {
      query,
      bindings,
      totalBindings: bindings.length,
      executionTime
    };
  }

  /**
   * Traverse graph from starting node with pattern
   *
   * @example
   * await traverse("P0-PUSCH", {
   *   direction: 'outgoing',
   *   edgeTypes: ['affects', 'controls'],
   *   maxDepth: 3,
   *   filter: (node) => node.type === 'parameter' || node.type === 'metric'
   * })
   */
  async traverse(
    startNodeId: string,
    pattern: TraversalPattern
  ): Promise<GraphMLNode[]> {
    console.log(`[KG-Query] Traversing from node: ${startNodeId}`);

    const startNode = this.graph.nodes.get(startNodeId);
    if (!startNode) {
      throw new Error(`Start node not found: ${startNodeId}`);
    }

    const visited = new Set<string>();
    const result: GraphMLNode[] = [];
    const queue: Array<{ node: GraphMLNode; depth: number; path: GraphMLNode[] }> = [
      { node: startNode, depth: 0, path: [startNode] }
    ];

    while (queue.length > 0) {
      const { node, depth, path } = queue.shift()!;

      if (visited.has(node.id) || depth > pattern.maxDepth) {
        continue;
      }

      visited.add(node.id);

      // Apply filter
      if (!pattern.filter || pattern.filter(node)) {
        result.push(node);
      }

      // Get neighbors based on direction
      const neighbors = this.getNeighbors(node.id, pattern.direction, pattern.edgeTypes);

      for (const neighbor of neighbors) {
        if (!visited.has(neighbor.id)) {
          queue.push({
            node: neighbor,
            depth: depth + 1,
            path: [...path, neighbor]
          });
        }
      }
    }

    console.log(`[KG-Query] Traversal found ${result.length} nodes`);
    return result;
  }

  // ============================================================================
  // Private Methods - Query Execution
  // ============================================================================

  /**
   * Execute FIND query (e.g., "What parameters control uplink power?")
   */
  private async executeFindQuery(parsed: ParsedQuery): Promise<{
    nodes: GraphMLNode[];
    edges: GraphMLEdge[];
  }> {
    const nodes: GraphMLNode[] = [];
    const edges: GraphMLEdge[] = [];

    // Find nodes matching entities
    for (const entity of parsed.entities) {
      const matchingNodes = this.findNodesByEntity(entity);
      nodes.push(...matchingNodes);
    }

    // Find related nodes based on relationships
    for (const node of nodes) {
      const relatedNodes = await this.findRelatedNodes(node, parsed.relationships);
      nodes.push(...relatedNodes.nodes);
      edges.push(...relatedNodes.edges);
    }

    return { nodes: this.deduplicateNodes(nodes), edges: this.deduplicateEdges(edges) };
  }

  /**
   * Execute RELATE query (e.g., "How does P0 relate to SINR?")
   */
  private async executeRelateQuery(parsed: ParsedQuery): Promise<{
    nodes: GraphMLNode[];
    edges: GraphMLEdge[];
    paths: GraphPath[];
  }> {
    if (parsed.entities.length < 2) {
      throw new Error('RELATE query requires at least 2 entities');
    }

    const [sourceEntity, targetEntity] = parsed.entities;
    const sourceNodes = this.findNodesByEntity(sourceEntity);
    const targetNodes = this.findNodesByEntity(targetEntity);

    if (sourceNodes.length === 0 || targetNodes.length === 0) {
      return { nodes: [], edges: [], paths: [] };
    }

    // Find paths between source and target
    const paths = await this.findPaths(sourceNodes[0], targetNodes[0], 5);

    const nodes: GraphMLNode[] = [];
    const edges: GraphMLEdge[] = [];

    for (const path of paths) {
      nodes.push(...path.nodes);
      edges.push(...path.edges);
    }

    return {
      nodes: this.deduplicateNodes(nodes),
      edges: this.deduplicateEdges(edges),
      paths
    };
  }

  /**
   * Execute COMPARE query (e.g., "Compare LTE and NR power control")
   */
  private async executeCompareQuery(parsed: ParsedQuery): Promise<{
    nodes: GraphMLNode[];
    edges: GraphMLEdge[];
  }> {
    const nodes: GraphMLNode[] = [];
    const edges: GraphMLEdge[] = [];

    // Find nodes for each entity to compare
    for (const entity of parsed.entities) {
      const matchingNodes = this.findNodesByEntity(entity);
      nodes.push(...matchingNodes);

      // Get related nodes for context
      for (const node of matchingNodes) {
        const related = await this.findRelatedNodes(node, ['contains', 'implements']);
        nodes.push(...related.nodes);
        edges.push(...related.edges);
      }
    }

    return { nodes: this.deduplicateNodes(nodes), edges: this.deduplicateEdges(edges) };
  }

  /**
   * Execute EXPLAIN query (e.g., "Explain power control in 5G NR")
   */
  private async executeExplainQuery(parsed: ParsedQuery): Promise<{
    nodes: GraphMLNode[];
    edges: GraphMLEdge[];
    paths: GraphPath[];
  }> {
    const nodes: GraphMLNode[] = [];
    const edges: GraphMLEdge[] = [];
    const paths: GraphPath[] = [];

    // Find all nodes related to entities
    for (const entity of parsed.entities) {
      const matchingNodes = this.findNodesByEntity(entity);
      nodes.push(...matchingNodes);

      // Traverse from each node to build explanation graph
      for (const node of matchingNodes) {
        const traversed = await this.traverse(node.id, {
          direction: 'both',
          maxDepth: 2,
          collectPaths: true
        });
        nodes.push(...traversed);
      }
    }

    return {
      nodes: this.deduplicateNodes(nodes),
      edges: this.deduplicateEdges(edges),
      paths
    };
  }

  /**
   * Execute LIST query (e.g., "List all IEs in RRCReconfiguration")
   */
  private async executeListQuery(parsed: ParsedQuery): Promise<{
    nodes: GraphMLNode[];
    edges: GraphMLEdge[];
  }> {
    const nodes: GraphMLNode[] = [];
    const edges: GraphMLEdge[] = [];

    // Find container node (e.g., RRCReconfiguration)
    const containerEntity = parsed.entities.find(e =>
      e.toLowerCase().includes('rrc') ||
      e.toLowerCase().includes('message') ||
      e.toLowerCase().includes('spec')
    );

    if (containerEntity) {
      const containerNodes = this.findNodesByEntity(containerEntity);

      for (const container of containerNodes) {
        // Find all contained nodes
        const contained = this.getNeighbors(container.id, 'outgoing', ['contains', 'includes']);
        nodes.push(container, ...contained);

        // Get edges
        const outgoingEdges = this.graph.edges.get(container.id) || [];
        edges.push(...outgoingEdges.filter(e =>
          e.type === 'contains' || e.type === 'includes'
        ));
      }
    }

    return { nodes: this.deduplicateNodes(nodes), edges: this.deduplicateEdges(edges) };
  }

  // ============================================================================
  // Private Methods - Graph Operations
  // ============================================================================

  /**
   * Find nodes matching an entity name
   */
  private findNodesByEntity(entity: string): GraphMLNode[] {
    const normalizedEntity = entity.toLowerCase();
    const results: GraphMLNode[] = [];

    for (const node of this.graph.nodes.values()) {
      const normalizedLabel = node.label.toLowerCase();
      const normalizedId = node.id.toLowerCase();

      if (
        normalizedLabel.includes(normalizedEntity) ||
        normalizedId.includes(normalizedEntity) ||
        node.properties.name?.toLowerCase().includes(normalizedEntity)
      ) {
        results.push(node);
      }
    }

    return results;
  }

  /**
   * Find nodes related to a given node
   */
  private async findRelatedNodes(
    node: GraphMLNode,
    relationships: string[]
  ): Promise<{ nodes: GraphMLNode[]; edges: GraphMLEdge[] }> {
    const nodes: GraphMLNode[] = [];
    const edges: GraphMLEdge[] = [];

    const outgoingEdges = this.graph.edges.get(node.id) || [];
    const incomingEdges = this.graph.reverseEdges.get(node.id) || [];

    for (const edge of [...outgoingEdges, ...incomingEdges]) {
      if (relationships.length === 0 || relationships.includes(edge.type)) {
        edges.push(edge);

        const relatedNodeId = edge.source === node.id ? edge.target : edge.source;
        const relatedNode = this.graph.nodes.get(relatedNodeId);
        if (relatedNode) {
          nodes.push(relatedNode);
        }
      }
    }

    return { nodes, edges };
  }

  /**
   * Find paths between two nodes using BFS
   */
  private async findPaths(
    source: GraphMLNode,
    target: GraphMLNode,
    maxDepth: number
  ): Promise<GraphPath[]> {
    const paths: GraphPath[] = [];
    const queue: Array<{
      current: GraphMLNode;
      path: GraphMLNode[];
      edges: GraphMLEdge[];
      depth: number;
    }> = [{ current: source, path: [source], edges: [], depth: 0 }];

    const visited = new Set<string>();

    while (queue.length > 0) {
      const { current, path, edges: pathEdges, depth } = queue.shift()!;

      if (depth > maxDepth) continue;

      if (current.id === target.id && depth > 0) {
        paths.push({
          nodes: path,
          edges: pathEdges,
          length: depth,
          weight: this.calculatePathWeight(pathEdges)
        });
        continue;
      }

      const stateKey = `${current.id}-${depth}`;
      if (visited.has(stateKey)) continue;
      visited.add(stateKey);

      const neighbors = this.getNeighbors(current.id, 'outgoing');
      for (const neighbor of neighbors) {
        const connectingEdge = (this.graph.edges.get(current.id) || []).find(
          e => e.target === neighbor.id
        );

        if (connectingEdge && !path.some(n => n.id === neighbor.id)) {
          queue.push({
            current: neighbor,
            path: [...path, neighbor],
            edges: [...pathEdges, connectingEdge],
            depth: depth + 1
          });
        }
      }
    }

    // Sort paths by weight (lower is better)
    return paths.sort((a, b) => a.weight - b.weight).slice(0, 5);
  }

  /**
   * Get neighbor nodes based on direction and edge types
   */
  private getNeighbors(
    nodeId: string,
    direction: 'outgoing' | 'incoming' | 'both',
    edgeTypes?: string[]
  ): GraphMLNode[] {
    const neighbors: GraphMLNode[] = [];

    if (direction === 'outgoing' || direction === 'both') {
      const outgoingEdges = this.graph.edges.get(nodeId) || [];
      for (const edge of outgoingEdges) {
        if (!edgeTypes || edgeTypes.includes(edge.type)) {
          const neighbor = this.graph.nodes.get(edge.target);
          if (neighbor) neighbors.push(neighbor);
        }
      }
    }

    if (direction === 'incoming' || direction === 'both') {
      const incomingEdges = this.graph.reverseEdges.get(nodeId) || [];
      for (const edge of incomingEdges) {
        if (!edgeTypes || edgeTypes.includes(edge.type)) {
          const neighbor = this.graph.nodes.get(edge.source);
          if (neighbor) neighbors.push(neighbor);
        }
      }
    }

    return neighbors;
  }

  // ============================================================================
  // Private Methods - Query Parsing (Simplified)
  // ============================================================================

  /**
   * Parse Cypher-like query (simplified version)
   */
  private parseCypherQuery(query: string): any {
    // Simplified Cypher parser
    // In production, use a full parser like cypher-query-builder
    return {
      matches: [],
      where: [],
      returns: []
    };
  }

  /**
   * Execute pattern matching for Cypher
   */
  private async executePatternMatch(pattern: any): Promise<CypherMatch[]> {
    // Simplified pattern matching
    // In production, implement full Cypher pattern matching
    return [];
  }

  /**
   * Parse SPARQL-like query (simplified version)
   */
  private parseSPARQLQuery(query: string): any {
    // Simplified SPARQL parser
    // In production, use a full parser like sparqljs
    return {
      select: [],
      where: []
    };
  }

  /**
   * Execute triple pattern matching for SPARQL
   */
  private async executeTripleMatch(pattern: any): Promise<SPARQLBinding[]> {
    // Simplified triple matching
    // In production, implement full SPARQL triple matching
    return [];
  }

  // ============================================================================
  // Private Methods - Utilities
  // ============================================================================

  private deduplicateNodes(nodes: GraphMLNode[]): GraphMLNode[] {
    const seen = new Set<string>();
    return nodes.filter(node => {
      if (seen.has(node.id)) return false;
      seen.add(node.id);
      return true;
    });
  }

  private deduplicateEdges(edges: GraphMLEdge[]): GraphMLEdge[] {
    const seen = new Set<string>();
    return edges.filter(edge => {
      if (seen.has(edge.id)) return false;
      seen.add(edge.id);
      return true;
    });
  }

  private extractSourceSpecs(nodes: GraphMLNode[]): string[] {
    const specs = new Set<string>();
    for (const node of nodes) {
      if (node.metadata.source) {
        specs.add(node.metadata.source);
      }
    }
    return Array.from(specs);
  }

  private calculateConfidence(nodes: GraphMLNode[], edges: GraphMLEdge[]): number {
    // Simple confidence based on result count
    if (nodes.length === 0) return 0;
    if (nodes.length === 1) return 0.5;
    if (edges.length > 0) return 0.9;
    return 0.7;
  }

  private generateReasoning(
    parsed: ParsedQuery,
    nodes: GraphMLNode[],
    edges: GraphMLEdge[]
  ): string {
    return `Found ${nodes.length} nodes and ${edges.length} relationships ` +
           `matching intent '${parsed.intent}' for entities: ${parsed.entities.join(', ')}`;
  }

  private calculatePathWeight(edges: GraphMLEdge[]): number {
    // Simple weight calculation (can be enhanced with edge weights)
    return edges.reduce((sum, edge) => sum + (edge.weight || 1), 0);
  }
}

// ============================================================================
// RuvLLM Bridge - Semantic Understanding
// ============================================================================

/**
 * Bridge between knowledge graph and RuvLLM for semantic understanding
 */
export class KGRuvLLMBridge extends EventEmitter {
  private indexPath: string;
  private embeddings: Map<string, Float32Array>;

  constructor(indexPath: string = './ruvector-kg.db') {
    super();
    this.indexPath = indexPath;
    this.embeddings = new Map();

    console.log('[KG-RuvLLM] Initialized RuvLLM bridge');
    console.log(`[KG-RuvLLM] Index path: ${indexPath}`);
  }

  /**
   * Parse natural language question into structured query
   *
   * Uses RuvLLM to understand intent and extract entities/relationships
   */
  async parseQuestion(question: string): Promise<ParsedQuery> {
    console.log(`[KG-RuvLLM] Parsing question: "${question}"`);

    // Extract intent
    const intent = this.extractIntent(question);

    // Extract entities using NER
    const entities = this.extractEntities(question);

    // Extract relationships
    const relationships = this.extractRelationships(question);

    // Identify relevant specs
    const specs = this.identifySpecs(question);

    // Build traversal pattern if needed
    const traversalPattern = this.buildTraversalPattern(intent, question);

    const parsed: ParsedQuery = {
      intent,
      entities,
      relationships,
      constraints: [],
      specs,
      traversalPattern
    };

    console.log(`[KG-RuvLLM] Parsed query:`, parsed);
    return parsed;
  }

  /**
   * Generate natural language answer from graph results
   */
  async generateAnswer(
    question: string,
    graphResults: GraphMLNode[],
    sourceSpecs: string[]
  ): Promise<string> {
    console.log(`[KG-RuvLLM] Generating answer for ${graphResults.length} results`);

    if (graphResults.length === 0) {
      return `I couldn't find any information in the knowledge graph to answer: "${question}"`;
    }

    // Build context from graph results
    const context = this.buildContext(graphResults);

    // Generate answer (in production, call actual RuvLLM)
    const answer = this.synthesizeAnswer(question, context, sourceSpecs);

    return answer;
  }

  /**
   * Explain path through knowledge graph
   */
  async explainPath(path: GraphMLNode[]): Promise<string> {
    if (path.length === 0) return 'Empty path';
    if (path.length === 1) return `Single node: ${path[0].label}`;

    const explanations: string[] = [];

    for (let i = 0; i < path.length - 1; i++) {
      const current = path[i];
      const next = path[i + 1];

      explanations.push(
        `${current.label} (${current.type}) → ${next.label} (${next.type})`
      );
    }

    return `Path: ${explanations.join(' → ')}`;
  }

  // ============================================================================
  // Private Methods - NLP Processing
  // ============================================================================

  /**
   * Extract query intent from natural language
   */
  private extractIntent(question: string): ParsedQuery['intent'] {
    const lower = question.toLowerCase();

    if (lower.includes('what') || lower.includes('which') || lower.includes('find')) {
      return 'find';
    }
    if (lower.includes('how') && (lower.includes('relate') || lower.includes('connect'))) {
      return 'relate';
    }
    if (lower.includes('compare') || lower.includes('difference')) {
      return 'compare';
    }
    if (lower.includes('explain') || lower.includes('describe')) {
      return 'explain';
    }
    if (lower.includes('list') || lower.includes('show all')) {
      return 'list';
    }

    return 'find';  // Default
  }

  /**
   * Extract named entities from question
   */
  private extractEntities(question: string): string[] {
    const entities: string[] = [];

    // Common RAN parameters
    const ranParams = ['P0', 'alpha', 'SINR', 'RSRP', 'RSRQ', 'CQI', 'MCS', 'PRB',
                       'BLER', 'throughput', 'latency', 'tilt', 'power', 'beamweight'];

    // Common IEs and procedures
    const ranIEs = ['RRCReconfiguration', 'RRCSetup', 'RRCReestablishment',
                    'MeasurementReport', 'HandoverCommand'];

    // Common concepts
    const ranConcepts = ['uplink', 'downlink', 'power control', 'handover',
                        'interference', 'mobility', 'beam management'];

    // Extract parameters
    for (const param of ranParams) {
      if (question.toLowerCase().includes(param.toLowerCase())) {
        entities.push(param);
      }
    }

    // Extract IEs
    for (const ie of ranIEs) {
      if (question.toLowerCase().includes(ie.toLowerCase())) {
        entities.push(ie);
      }
    }

    // Extract concepts
    for (const concept of ranConcepts) {
      if (question.toLowerCase().includes(concept.toLowerCase())) {
        entities.push(concept);
      }
    }

    // Extract spec references (e.g., "TS 38.331")
    const specRegex = /TS\s*\d+\.\d+/gi;
    const specMatches = question.match(specRegex);
    if (specMatches) {
      entities.push(...specMatches);
    }

    return entities;
  }

  /**
   * Extract relationships from question
   */
  private extractRelationships(question: string): string[] {
    const relationships: string[] = [];
    const lower = question.toLowerCase();

    if (lower.includes('control') || lower.includes('controls')) {
      relationships.push('controls');
    }
    if (lower.includes('affect') || lower.includes('affects')) {
      relationships.push('affects');
    }
    if (lower.includes('contain') || lower.includes('includes')) {
      relationships.push('contains');
    }
    if (lower.includes('reference') || lower.includes('refers')) {
      relationships.push('references');
    }
    if (lower.includes('implement') || lower.includes('implements')) {
      relationships.push('implements');
    }

    return relationships;
  }

  /**
   * Identify relevant 3GPP specs from question
   */
  private identifySpecs(question: string): string[] {
    const specs: string[] = [];
    const lower = question.toLowerCase();

    // Extract explicit spec references
    const specRegex = /TS\s*(\d+\.\d+)/gi;
    const matches = question.match(specRegex);
    if (matches) {
      return matches.map(m => m.replace(/\s/g, ''));
    }

    // Infer specs from keywords
    if (lower.includes('rrc') || lower.includes('radio resource control')) {
      specs.push('TS 38.331');
    }
    if (lower.includes('power') || lower.includes('physical layer')) {
      specs.push('TS 38.213', 'TS 38.214');
    }
    if (lower.includes('measurement') || lower.includes('mobility')) {
      specs.push('TS 38.331', 'TS 38.300');
    }
    if (lower.includes('mom') || lower.includes('managed object')) {
      specs.push('TS 28.552', 'TS 28.541');
    }

    return specs;
  }

  /**
   * Build traversal pattern from intent and question
   */
  private buildTraversalPattern(
    intent: ParsedQuery['intent'],
    question: string
  ): TraversalPattern | undefined {
    if (intent === 'relate' || intent === 'explain') {
      return {
        direction: 'both',
        maxDepth: 3,
        collectPaths: true
      };
    }

    if (intent === 'list') {
      return {
        direction: 'outgoing',
        edgeTypes: ['contains', 'includes'],
        maxDepth: 1
      };
    }

    return undefined;
  }

  /**
   * Build context string from graph results
   */
  private buildContext(nodes: GraphMLNode[]): string {
    const contextParts: string[] = [];

    for (const node of nodes.slice(0, 10)) {  // Limit to top 10
      let part = `- ${node.label} (${node.type})`;

      if (node.metadata.source) {
        part += ` from ${node.metadata.source}`;
      }

      if (node.properties.description) {
        part += `: ${node.properties.description}`;
      }

      contextParts.push(part);
    }

    return contextParts.join('\n');
  }

  /**
   * Synthesize natural language answer
   */
  private synthesizeAnswer(
    question: string,
    context: string,
    sourceSpecs: string[]
  ): string {
    // In production, this would call RuvLLM for generation
    // For now, provide a structured response

    const specsText = sourceSpecs.length > 0
      ? ` (from ${sourceSpecs.join(', ')})`
      : '';

    return `Based on the knowledge graph${specsText}, here are the relevant results:\n\n${context}\n\n` +
           `These results were found by analyzing the 3GPP specification knowledge graph.`;
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a sample knowledge graph for testing
 */
export function createSampleKnowledgeGraph(): KnowledgeGraph {
  const nodes = new Map<string, GraphMLNode>();
  const edges = new Map<string, GraphMLEdge[]>();
  const reverseEdges = new Map<string, GraphMLEdge[]>();

  // Sample nodes
  const p0Node: GraphMLNode = {
    id: 'param-p0-pusch',
    type: 'parameter',
    label: 'P0-PUSCH',
    properties: {
      name: 'P0-PUSCH',
      description: 'Target received power for PUSCH',
      unit: 'dBm',
      range: [-202, 24]
    },
    metadata: {
      source: 'TS 38.213',
      section: '7.1.1',
      tags: ['power-control', 'uplink']
    }
  };

  const sinrNode: GraphMLNode = {
    id: 'metric-sinr',
    type: 'concept',
    label: 'SINR',
    properties: {
      name: 'Signal-to-Interference-plus-Noise Ratio',
      description: 'Ratio of signal power to interference plus noise',
      unit: 'dB'
    },
    metadata: {
      source: 'TS 38.215',
      tags: ['measurement', 'quality']
    }
  };

  const rrcReconfig: GraphMLNode = {
    id: 'ie-rrc-reconfig',
    type: 'ie',
    label: 'RRCReconfiguration',
    properties: {
      name: 'RRCReconfiguration',
      description: 'Main RRC message for reconfiguring radio resources'
    },
    metadata: {
      source: 'TS 38.331',
      section: '6.2.2',
      tags: ['rrc', 'signaling']
    }
  };

  nodes.set(p0Node.id, p0Node);
  nodes.set(sinrNode.id, sinrNode);
  nodes.set(rrcReconfig.id, rrcReconfig);

  // Sample edges
  const edge1: GraphMLEdge = {
    id: 'edge-1',
    source: p0Node.id,
    target: sinrNode.id,
    type: 'affects',
    properties: {
      relationship: 'controls',
      strength: 0.9
    },
    weight: 1
  };

  edges.set(p0Node.id, [edge1]);
  reverseEdges.set(sinrNode.id, [edge1]);

  return {
    nodes,
    edges,
    reverseEdges,
    metadata: {
      totalNodes: nodes.size,
      totalEdges: 1,
      specs: ['TS 38.213', 'TS 38.215', 'TS 38.331'],
      lastUpdated: new Date()
    }
  };
}
