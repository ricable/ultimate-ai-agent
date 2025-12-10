/**
 * 3GPP Knowledge Graph GraphML Parser
 *
 * Parses GraphML knowledge graphs from otellm/3gpp_knowledgeGraph format
 * Extracts entities, relationships, and semantic structures from:
 * - TS 23.xxx (Architecture)
 * - TS 24.xxx (Protocols)
 * - TS 28.xxx (Management)
 * - TS 29.xxx (APIs/Interfaces)
 * - TS 36.xxx (LTE)
 * - TS 38.xxx (NR/5G)
 *
 * @module graphml-parser
 */

import { z } from 'zod';
import * as fs from 'fs/promises';
import * as path from 'path';

// ============================================================================
// Core Types and Interfaces
// ============================================================================

/**
 * Node types in 3GPP knowledge graph
 */
export type NodeType =
  | '3gpp_spec'      // Top-level specification (e.g., TS 38.331)
  | 'section'        // Specification section (e.g., 5.2.1)
  | 'term'           // Defined term or concept
  | 'procedure'      // Protocol procedure or state machine
  | 'parameter'      // Configuration parameter
  | 'ie'             // Information Element (ASN.1)
  | 'message';       // Protocol message

/**
 * Edge types representing relationships
 */
export type EdgeType =
  | 'references'     // Spec A references Spec B
  | 'defines'        // Section defines a term/IE
  | 'contains'       // Parent contains child (hierarchy)
  | 'implements'     // Procedure implements a concept
  | 'extends';       // IE extends another IE (inheritance)

/**
 * GraphML Node representing a 3GPP entity
 */
export interface GraphMLNode {
  id: string;
  label: string;
  type: NodeType;
  attributes: Record<string, string>;
}

/**
 * GraphML Edge representing a relationship
 */
export interface GraphMLEdge {
  id: string;
  source: string;
  target: string;
  type: EdgeType;
  weight?: number;
}

/**
 * Complete Knowledge Graph structure
 */
export interface KnowledgeGraph {
  nodes: Map<string, GraphMLNode>;
  edges: GraphMLEdge[];
  metadata: {
    release: string; // R15, R16, R17, R18
    series: string;  // 23, 24, 28, 29, 36, 38
  };
}

/**
 * Adjacency list for efficient graph traversal
 */
export interface AdjacencyList {
  outgoing: Map<string, string[]>;  // nodeId -> [targetIds]
  incoming: Map<string, string[]>;  // nodeId -> [sourceIds]
}

// ============================================================================
// Zod Schemas for Validation
// ============================================================================

const NodeTypeSchema = z.enum([
  '3gpp_spec',
  'section',
  'term',
  'procedure',
  'parameter',
  'ie',
  'message'
]);

const EdgeTypeSchema = z.enum([
  'references',
  'defines',
  'contains',
  'implements',
  'extends'
]);

const GraphMLNodeSchema = z.object({
  id: z.string(),
  label: z.string(),
  type: NodeTypeSchema,
  attributes: z.record(z.string(), z.string()),
});

const GraphMLEdgeSchema = z.object({
  id: z.string(),
  source: z.string(),
  target: z.string(),
  type: EdgeTypeSchema,
  weight: z.number().optional(),
});

// ============================================================================
// 3GPP Specification Series Detection
// ============================================================================

/**
 * Detect 3GPP specification series from spec ID
 */
export function detect3GPPSeries(specId: string): string {
  const match = specId.match(/TS\s*(\d{2})\.\d{3}/i);
  if (!match) return 'unknown';

  const series = match[1];
  switch (series) {
    case '23': return 'Architecture';
    case '24': return 'Protocols';
    case '28': return 'Management';
    case '29': return 'APIs';
    case '36': return 'LTE';
    case '38': return 'NR';
    default: return series;
  }
}

/**
 * Extract release from spec version (e.g., R15, R16, R17, R18)
 */
export function extractRelease(version: string): string {
  const match = version.match(/R(\d{2})/i);
  return match ? `R${match[1]}` : 'R15'; // Default to R15
}

// ============================================================================
// GraphML XML Parser
// ============================================================================

/**
 * Simple but robust XML parser for GraphML format
 * Handles the specific structure of 3GPP knowledge graphs
 */
export class GraphMLXMLParser {
  /**
   * Parse GraphML XML string into nodes and edges
   */
  parse(xml: string): { nodes: GraphMLNode[]; edges: GraphMLEdge[] } {
    const nodes: GraphMLNode[] = [];
    const edges: GraphMLEdge[] = [];

    // Parse nodes
    const nodeMatches = Array.from(xml.matchAll(/<node\s+id="([^"]+)"[^>]*>([\s\S]*?)<\/node>/g));
    for (const match of nodeMatches) {
      const [, id, content] = match;
      const node = this.parseNode(id, content);
      if (node) nodes.push(node);
    }

    // Parse edges
    const edgeMatches = Array.from(xml.matchAll(/<edge\s+id="([^"]+)"\s+source="([^"]+)"\s+target="([^"]+)"[^>]*>([\s\S]*?)<\/edge>/g));
    for (const match of edgeMatches) {
      const [, id, source, target, content] = match;
      const edge = this.parseEdge(id, source, target, content);
      if (edge) edges.push(edge);
    }

    return { nodes, edges };
  }

  /**
   * Parse individual node element
   */
  private parseNode(id: string, content: string): GraphMLNode | null {
    const attributes: Record<string, string> = {};

    // Extract data elements (GraphML format: <data key="keyname">value</data>)
    const dataMatches = Array.from(content.matchAll(/<data\s+key="([^"]+)">([^<]*)<\/data>/g));
    for (const match of dataMatches) {
      const [, key, value] = match;
      attributes[key] = value.trim();
    }

    // Determine node type
    const typeStr = attributes.type || attributes.nodeType || '3gpp_spec';
    const type = this.normalizeNodeType(typeStr);

    // Extract label
    const label = attributes.label || attributes.name || id;

    return {
      id,
      label,
      type,
      attributes,
    };
  }

  /**
   * Parse individual edge element
   */
  private parseEdge(id: string, source: string, target: string, content: string): GraphMLEdge | null {
    const attributes: Record<string, string> = {};

    // Extract data elements
    const dataMatches = Array.from(content.matchAll(/<data\s+key="([^"]+)">([^<]*)<\/data>/g));
    for (const match of dataMatches) {
      const [, key, value] = match;
      attributes[key] = value.trim();
    }

    // Determine edge type
    const typeStr = attributes.type || attributes.edgeType || 'references';
    const type = this.normalizeEdgeType(typeStr);

    // Extract weight if present
    const weight = attributes.weight ? parseFloat(attributes.weight) : undefined;

    return {
      id,
      source,
      target,
      type,
      weight,
    };
  }

  /**
   * Normalize node type string to valid NodeType
   */
  private normalizeNodeType(typeStr: string): NodeType {
    const normalized = typeStr.toLowerCase().replace(/[_-]/g, '');

    if (normalized.includes('spec')) return '3gpp_spec';
    if (normalized.includes('section')) return 'section';
    if (normalized.includes('term') || normalized.includes('definition')) return 'term';
    if (normalized.includes('procedure') || normalized.includes('state')) return 'procedure';
    if (normalized.includes('parameter') || normalized.includes('param')) return 'parameter';
    if (normalized.includes('ie') || normalized.includes('informationelement')) return 'ie';
    if (normalized.includes('message') || normalized.includes('msg')) return 'message';

    return '3gpp_spec'; // Default
  }

  /**
   * Normalize edge type string to valid EdgeType
   */
  private normalizeEdgeType(typeStr: string): EdgeType {
    const normalized = typeStr.toLowerCase().replace(/[_-]/g, '');

    if (normalized.includes('reference')) return 'references';
    if (normalized.includes('define')) return 'defines';
    if (normalized.includes('contain') || normalized.includes('parent')) return 'contains';
    if (normalized.includes('implement')) return 'implements';
    if (normalized.includes('extend') || normalized.includes('inherit')) return 'extends';

    return 'references'; // Default
  }

  /**
   * Extract metadata from GraphML header
   */
  extractMetadata(xml: string): { release: string; series: string } {
    let release = 'R15';
    let series = '38';

    // Try to extract from graph attributes
    const graphMatch = xml.match(/<graph[^>]*>/);
    if (graphMatch) {
      const releaseMatch = graphMatch[0].match(/release="([^"]+)"/i);
      const seriesMatch = graphMatch[0].match(/series="([^"]+)"/i);

      if (releaseMatch) release = releaseMatch[1];
      if (seriesMatch) series = seriesMatch[1];
    }

    // Try to infer from content
    const releaseInContent = xml.match(/R(\d{2})/);
    if (releaseInContent) release = `R${releaseInContent[1]}`;

    const seriesInContent = xml.match(/TS\s*(\d{2})\.\d{3}/);
    if (seriesInContent) series = seriesInContent[1];

    return { release, series };
  }
}

// ============================================================================
// 3GPP Knowledge Graph Class
// ============================================================================

/**
 * Main class for managing and querying 3GPP knowledge graphs
 */
export class ThreeGPPKnowledgeGraph {
  private graph: KnowledgeGraph;
  private adjacency: AdjacencyList;
  private parser: GraphMLXMLParser;

  constructor() {
    this.graph = {
      nodes: new Map(),
      edges: [],
      metadata: { release: 'R15', series: '38' },
    };
    this.adjacency = {
      outgoing: new Map(),
      incoming: new Map(),
    };
    this.parser = new GraphMLXMLParser();
  }

  // ==========================================================================
  // Loading Methods
  // ==========================================================================

  /**
   * Load knowledge graph from GraphML file
   */
  async loadFromGraphML(filePath: string): Promise<void> {
    const xml = await fs.readFile(filePath, 'utf-8');
    this.parseGraphML(xml);
  }

  /**
   * Load knowledge graph from URL (fetch GraphML)
   */
  async loadFromURL(url: string): Promise<void> {
    // Use dynamic import to avoid issues in environments without fetch
    const fetch = (await import('node-fetch')).default;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Failed to fetch GraphML from ${url}: ${response.statusText}`);
    }

    const xml = await response.text();
    this.parseGraphML(xml);
  }

  /**
   * Parse GraphML XML string and build graph
   */
  private parseGraphML(xml: string): void {
    // Extract metadata
    this.graph.metadata = this.parser.extractMetadata(xml);

    // Parse nodes and edges
    const { nodes, edges } = this.parser.parse(xml);

    // Build node map
    this.graph.nodes.clear();
    for (const node of nodes) {
      this.graph.nodes.set(node.id, node);
    }

    // Store edges
    this.graph.edges = edges;

    // Build adjacency lists
    this.buildAdjacencyLists();

    console.log(`Loaded 3GPP Knowledge Graph: ${nodes.length} nodes, ${edges.length} edges`);
    console.log(`Release: ${this.graph.metadata.release}, Series: ${this.graph.metadata.series}`);
  }

  /**
   * Build adjacency lists for efficient traversal
   */
  private buildAdjacencyLists(): void {
    this.adjacency.outgoing.clear();
    this.adjacency.incoming.clear();

    for (const edge of this.graph.edges) {
      // Outgoing edges
      if (!this.adjacency.outgoing.has(edge.source)) {
        this.adjacency.outgoing.set(edge.source, []);
      }
      this.adjacency.outgoing.get(edge.source)!.push(edge.target);

      // Incoming edges
      if (!this.adjacency.incoming.has(edge.target)) {
        this.adjacency.incoming.set(edge.target, []);
      }
      this.adjacency.incoming.get(edge.target)!.push(edge.source);
    }
  }

  // ==========================================================================
  // Query Methods
  // ==========================================================================

  /**
   * Find a specification node by ID (e.g., "TS38331", "TS-38.331")
   */
  findSpec(specId: string): GraphMLNode | null {
    // Normalize spec ID
    const normalized = specId.replace(/[.\s-]/g, '').toUpperCase();

    // Try direct lookup
    const entries = Array.from(this.graph.nodes.entries());
    for (const [id, node] of entries) {
      const nodeIdNormalized = id.replace(/[.\s-]/g, '').toUpperCase();
      if (nodeIdNormalized === normalized || nodeIdNormalized.includes(normalized)) {
        if (node.type === '3gpp_spec') {
          return node;
        }
      }
    }

    // Try by label
    const nodes = Array.from(this.graph.nodes.values());
    for (const node of nodes) {
      if (node.type === '3gpp_spec' && node.label.replace(/[.\s-]/g, '').toUpperCase().includes(normalized)) {
        return node;
      }
    }

    return null;
  }

  /**
   * Find all related nodes within a given depth (BFS traversal)
   */
  findRelated(nodeId: string, depth: number = 1): GraphMLNode[] {
    const visited = new Set<string>();
    const related: GraphMLNode[] = [];
    const queue: Array<{ id: string; currentDepth: number }> = [{ id: nodeId, currentDepth: 0 }];

    while (queue.length > 0) {
      const { id, currentDepth } = queue.shift()!;

      if (visited.has(id) || currentDepth > depth) continue;
      visited.add(id);

      const node = this.graph.nodes.get(id);
      if (node && id !== nodeId) {
        related.push(node);
      }

      if (currentDepth < depth) {
        // Add outgoing neighbors
        const outgoing = this.adjacency.outgoing.get(id) || [];
        for (const targetId of outgoing) {
          if (!visited.has(targetId)) {
            queue.push({ id: targetId, currentDepth: currentDepth + 1 });
          }
        }

        // Add incoming neighbors
        const incoming = this.adjacency.incoming.get(id) || [];
        for (const sourceId of incoming) {
          if (!visited.has(sourceId)) {
            queue.push({ id: sourceId, currentDepth: currentDepth + 1 });
          }
        }
      }
    }

    return related;
  }

  /**
   * Find shortest path between two nodes (BFS)
   */
  findPath(fromId: string, toId: string): GraphMLNode[] {
    if (fromId === toId) {
      const node = this.graph.nodes.get(fromId);
      return node ? [node] : [];
    }

    const visited = new Set<string>();
    const queue: Array<{ id: string; path: string[] }> = [{ id: fromId, path: [fromId] }];

    while (queue.length > 0) {
      const { id, path } = queue.shift()!;

      if (id === toId) {
        // Found path - convert IDs to nodes
        return path
          .map(nodeId => this.graph.nodes.get(nodeId))
          .filter((node): node is GraphMLNode => node !== undefined);
      }

      if (visited.has(id)) continue;
      visited.add(id);

      // Explore outgoing edges
      const neighbors = this.adjacency.outgoing.get(id) || [];
      for (const neighborId of neighbors) {
        if (!visited.has(neighborId)) {
          queue.push({ id: neighborId, path: [...path, neighborId] });
        }
      }
    }

    return []; // No path found
  }

  /**
   * Find nodes by type
   */
  findByType(type: NodeType): GraphMLNode[] {
    const result: GraphMLNode[] = [];
    const nodes = Array.from(this.graph.nodes.values());
    for (const node of nodes) {
      if (node.type === type) result.push(node);
    }
    return result;
  }

  /**
   * Find nodes by attribute value
   */
  findByAttribute(key: string, value: string): GraphMLNode[] {
    const result: GraphMLNode[] = [];
    const nodes = Array.from(this.graph.nodes.values());
    for (const node of nodes) {
      if (node.attributes[key] === value) result.push(node);
    }
    return result;
  }

  /**
   * Search nodes by label (fuzzy search)
   */
  searchByLabel(query: string): GraphMLNode[] {
    const lowerQuery = query.toLowerCase();
    const result: GraphMLNode[] = [];
    const nodes = Array.from(this.graph.nodes.values());
    for (const node of nodes) {
      if (node.label.toLowerCase().includes(lowerQuery)) {
        result.push(node);
      }
    }
    return result;
  }

  // ==========================================================================
  // Semantic Search Preparation (for ruvector)
  // ==========================================================================

  /**
   * Generate embedding text for a node (for vector indexing)
   * Combines label, type, and attributes into searchable text
   */
  getEmbeddingText(nodeId: string): string {
    const node = this.graph.nodes.get(nodeId);
    if (!node) return '';

    const parts: string[] = [];

    // Add label
    parts.push(`Label: ${node.label}`);

    // Add type
    parts.push(`Type: ${node.type}`);

    // Add attributes
    for (const [key, value] of Object.entries(node.attributes)) {
      if (value) {
        parts.push(`${key}: ${value}`);
      }
    }

    // Add relationship context
    const outgoing = this.adjacency.outgoing.get(nodeId) || [];
    const incoming = this.adjacency.incoming.get(nodeId) || [];

    if (outgoing.length > 0) {
      const targets = outgoing
        .map(id => this.graph.nodes.get(id)?.label)
        .filter(Boolean)
        .slice(0, 5); // Limit context
      parts.push(`Related to: ${targets.join(', ')}`);
    }

    if (incoming.length > 0) {
      const sources = incoming
        .map(id => this.graph.nodes.get(id)?.label)
        .filter(Boolean)
        .slice(0, 5);
      parts.push(`Referenced by: ${sources.join(', ')}`);
    }

    return parts.join('\n');
  }

  /**
   * Export all nodes with embedding text for bulk indexing
   */
  exportForVectorIndexing(): Array<{ id: string; text: string; metadata: Record<string, any> }> {
    const result: Array<{ id: string; text: string; metadata: Record<string, any> }> = [];
    const nodes = Array.from(this.graph.nodes.values());
    for (const node of nodes) {
      result.push({
        id: node.id,
        text: this.getEmbeddingText(node.id),
        metadata: {
          label: node.label,
          type: node.type,
          release: this.graph.metadata.release,
          series: this.graph.metadata.series,
          ...node.attributes,
        },
      });
    }
    return result;
  }

  // ==========================================================================
  // Graph Statistics and Analysis
  // ==========================================================================

  /**
   * Get graph statistics
   */
  getStats(): {
    nodeCount: number;
    edgeCount: number;
    nodeTypeDistribution: Record<NodeType, number>;
    edgeTypeDistribution: Record<EdgeType, number>;
    avgDegree: number;
  } {
    const nodeTypeDistribution: Record<string, number> = {};
    const edgeTypeDistribution: Record<string, number> = {};

    // Count node types
    const nodes = Array.from(this.graph.nodes.values());
    for (const node of nodes) {
      nodeTypeDistribution[node.type] = (nodeTypeDistribution[node.type] || 0) + 1;
    }

    // Count edge types
    for (const edge of this.graph.edges) {
      edgeTypeDistribution[edge.type] = (edgeTypeDistribution[edge.type] || 0) + 1;
    }

    // Calculate average degree
    const totalDegree = Array.from(this.adjacency.outgoing.values())
      .reduce((sum, neighbors) => sum + neighbors.length, 0);
    const avgDegree = this.graph.nodes.size > 0 ? totalDegree / this.graph.nodes.size : 0;

    return {
      nodeCount: this.graph.nodes.size,
      edgeCount: this.graph.edges.length,
      nodeTypeDistribution: nodeTypeDistribution as Record<NodeType, number>,
      edgeTypeDistribution: edgeTypeDistribution as Record<EdgeType, number>,
      avgDegree,
    };
  }

  /**
   * Get subgraph for a specific spec series
   */
  getSubgraphBySeries(series: string): ThreeGPPKnowledgeGraph {
    const subgraph = new ThreeGPPKnowledgeGraph();

    // Filter nodes matching series
    const seriesPattern = new RegExp(`TS\\s*${series}\\.\\d{3}`, 'i');
    const matchingNodes: GraphMLNode[] = [];
    const nodes = Array.from(this.graph.nodes.values());
    for (const node of nodes) {
      if (seriesPattern.test(node.label) || seriesPattern.test(node.id)) {
        matchingNodes.push(node);
      }
    }

    const nodeIds = new Set(matchingNodes.map(n => n.id));

    // Add matching nodes
    for (const node of matchingNodes) {
      subgraph.graph.nodes.set(node.id, node);
    }

    // Add edges between matching nodes
    subgraph.graph.edges = this.graph.edges.filter(
      edge => nodeIds.has(edge.source) && nodeIds.has(edge.target)
    );

    // Copy metadata
    subgraph.graph.metadata = { ...this.graph.metadata, series };

    // Build adjacency lists
    subgraph.buildAdjacencyLists();

    return subgraph;
  }

  // ==========================================================================
  // Getters
  // ==========================================================================

  get nodes(): Map<string, GraphMLNode> {
    return this.graph.nodes;
  }

  get edges(): GraphMLEdge[] {
    return this.graph.edges;
  }

  get metadata(): { release: string; series: string } {
    return this.graph.metadata;
  }
}

// ============================================================================
// Export Utilities
// ============================================================================

/**
 * Create a new knowledge graph instance
 */
export function createKnowledgeGraph(): ThreeGPPKnowledgeGraph {
  return new ThreeGPPKnowledgeGraph();
}

/**
 * Load knowledge graph from file
 */
export async function loadKnowledgeGraph(filePath: string): Promise<ThreeGPPKnowledgeGraph> {
  const kg = new ThreeGPPKnowledgeGraph();
  await kg.loadFromGraphML(filePath);
  return kg;
}

/**
 * Load knowledge graph from URL
 */
export async function loadKnowledgeGraphFromURL(url: string): Promise<ThreeGPPKnowledgeGraph> {
  const kg = new ThreeGPPKnowledgeGraph();
  await kg.loadFromURL(url);
  return kg;
}

// ============================================================================
// ASN.1 and IE Extraction Utilities
// ============================================================================

/**
 * Extract ASN.1 definitions from node attributes
 */
export function extractASN1Definition(node: GraphMLNode): {
  name: string;
  type: string;
  definition: string;
} | null {
  if (node.type !== 'ie') return null;

  return {
    name: node.attributes.asn1Name || node.label,
    type: node.attributes.asn1Type || 'SEQUENCE',
    definition: node.attributes.asn1Definition || node.attributes.definition || '',
  };
}

/**
 * Extract parameter ranges from node attributes
 */
export function extractParameterRange(node: GraphMLNode): {
  min: number | null;
  max: number | null;
  unit: string | null;
} {
  if (node.type !== 'parameter') {
    return { min: null, max: null, unit: null };
  }

  const min = node.attributes.min ? parseFloat(node.attributes.min) : null;
  const max = node.attributes.max ? parseFloat(node.attributes.max) : null;
  const unit = node.attributes.unit || null;

  return { min, max, unit };
}

/**
 * Extract procedure state machine from node
 */
export function extractProcedureStates(node: GraphMLNode): string[] {
  if (node.type !== 'procedure') return [];

  const statesStr = node.attributes.states || node.attributes.stateMachine || '';
  return statesStr
    .split(/[,;]/)
    .map(s => s.trim())
    .filter(s => s.length > 0);
}

// ============================================================================
// Example Usage and Testing
// ============================================================================

/**
 * Example: Load and query a 3GPP knowledge graph
 */
export async function exampleUsage() {
  const kg = createKnowledgeGraph();

  // Example GraphML (minimal structure)
  const exampleGraphML = `<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <graph id="3gpp_knowledge" edgedefault="directed" release="R18" series="38">
    <node id="TS38331">
      <data key="label">TS 38.331 - RRC Protocol Specification</data>
      <data key="type">3gpp_spec</data>
      <data key="release">R18</data>
    </node>
    <node id="RRCSetup">
      <data key="label">RRCSetup</data>
      <data key="type">message</data>
      <data key="asn1Type">SEQUENCE</data>
    </node>
    <node id="SIB1">
      <data key="label">SystemInformationBlockType1</data>
      <data key="type">ie</data>
      <data key="asn1Name">SIB1</data>
    </node>
    <edge id="e1" source="TS38331" target="RRCSetup">
      <data key="type">defines</data>
    </edge>
    <edge id="e2" source="TS38331" target="SIB1">
      <data key="type">defines</data>
    </edge>
  </graph>
</graphml>`;

  // In real usage, load from file:
  // await kg.loadFromGraphML('./3gpp-knowledge.graphml');

  console.log('3GPP Knowledge Graph Parser - Ready for integration with ruvector');
}

// ============================================================================
// Default Export
// ============================================================================

export default ThreeGPPKnowledgeGraph;
