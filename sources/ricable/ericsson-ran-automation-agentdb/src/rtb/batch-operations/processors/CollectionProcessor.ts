/**
 * Collection Processor for Multi-Node Configuration Support
 *
 * Processes node collections with scope filters, wildcard patterns, and intelligent
 * node selection for Ericsson RAN batch operations.
 */

import {
  NodeCollection,
  NodePattern,
  ScopeFilter,
  FilterCondition,
  BatchExecutionContext
} from '../core/types';

import { ScopeFilterEngine } from './ScopeFilterEngine';
import { WildcardProcessor } from './WildcardProcessor';

/**
 * Processed node information
 */
export interface ProcessedNode {
  /** Node identifier */
  id: string;
  /** Node name */
  name: string;
  /** Node type */
  nodeType: string;
  /** Network element type */
  neType: string;
  /** Node status */
  status: string;
  /** Synchronization status */
  syncStatus: string;
  /** Node location */
  location?: string;
  /** Node version */
  version?: string;
  /** Node vendor */
  vendor?: string;
  /** Additional node attributes */
  attributes: Record<string, any>;
  /** Processing metadata */
  metadata: {
    source: string;
    patternMatch: string;
    filterResults: Record<string, boolean>;
    processingTime: number;
  };
}

/**
 * Collection processing result
 */
export interface CollectionProcessingResult {
  /** Original collection */
  collection: NodeCollection;
  /** Processed nodes */
  nodes: ProcessedNode[];
  /** Processing statistics */
  statistics: {
    totalNodes: number;
    includedNodes: number;
    excludedNodes: number;
    filteredNodes: number;
    processingTime: number;
    patternsProcessed: number;
    filtersApplied: number;
  };
  /** Processing errors */
  errors: CollectionProcessingError[];
}

/**
 * Collection processing error
 */
export interface CollectionProcessingError {
  /** Error identifier */
  id: string;
  /** Error type */
  type: 'pattern_error' | 'filter_error' | 'node_resolution_error' | 'validation_error';
  /** Error message */
  message: string;
  /** Source of error */
  source: string;
  /** Error severity */
  severity: 'low' | 'medium' | 'high' | 'critical';
  /** Timestamp */
  timestamp: Date;
  /** Additional context */
  context?: Record<string, any>;
}

/**
 * Collection Processor
 */
export class CollectionProcessor {
  private scopeFilterEngine: ScopeFilterEngine;
  private wildcardProcessor: WildcardProcessor;
  private nodeCache: Map<string, ProcessedNode[]> = new Map();
  private patternCache: Map<string, string[]> = new Map();

  constructor() {
    this.scopeFilterEngine = new ScopeFilterEngine();
    this.wildcardProcessor = new WildcardProcessor();
  }

  /**
   * Process a node collection with scope filters
   */
  public async processCollection(
    collection: NodeCollection,
    scopeFilters: ScopeFilter[],
    context: BatchExecutionContext
  ): Promise<CollectionProcessingResult> {
    const startTime = Date.now();
    const errors: CollectionProcessingError[] = [];

    try {
      console.log(`Processing collection ${collection.id} with ${collection.nodePatterns.length} patterns`);

      // Resolve nodes from patterns
      const resolvedNodes = await this.resolveNodesFromPatterns(collection.nodePatterns, errors);

      // Apply scope filters
      const filteredNodes = await this.applyScopeFilters(resolvedNodes, scopeFilters, errors);

      // Sort and prioritize nodes
      const prioritizedNodes = this.prioritizeNodes(filteredNodes, collection);

      // Validate final node set
      const validatedNodes = await this.validateNodes(prioritizedNodes, errors);

      const processingTime = Date.now() - startTime;

      const statistics = {
        totalNodes: resolvedNodes.length,
        includedNodes: validatedNodes.length,
        excludedNodes: resolvedNodes.length - validatedNodes.length,
        filteredNodes: resolvedNodes.length - filteredNodes.length,
        processingTime,
        patternsProcessed: collection.nodePatterns.length,
        filtersApplied: scopeFilters.length
      };

      const result: CollectionProcessingResult = {
        collection,
        nodes: validatedNodes,
        statistics,
        errors
      };

      console.log(`Collection processing completed: ${validatedNodes.length} nodes from ${resolvedNodes.length} candidates`);

      return result;

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error(`Collection processing failed: ${errorMessage}`);

      errors.push({
        id: `collection_error_${Date.now()}`,
        type: 'validation_error',
        message: errorMessage,
        source: 'CollectionProcessor.processCollection',
        severity: 'critical',
        timestamp: new Date(),
        context: { collectionId: collection.id }
      });

      const processingTime = Date.now() - startTime;

      return {
        collection,
        nodes: [],
        statistics: {
          totalNodes: 0,
          includedNodes: 0,
          excludedNodes: 0,
          filteredNodes: 0,
          processingTime,
          patternsProcessed: collection.nodePatterns.length,
          filtersApplied: scopeFilters.length
        },
        errors
      };
    }
  }

  /**
   * Resolve nodes from patterns
   */
  private async resolveNodesFromPatterns(
    patterns: NodePattern[],
    errors: CollectionProcessingError[]
  ): Promise<ProcessedNode[]> {
    const allNodes: ProcessedNode[] = [];
    const processedPatterns = new Set<string>();

    for (const pattern of patterns) {
      try {
        if (processedPatterns.has(pattern.pattern)) {
          continue; // Skip duplicate patterns
        }
        processedPatterns.add(pattern.pattern);

        const patternNodes = await this.resolvePattern(pattern, errors);
        allNodes.push(...patternNodes);

      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        console.error(`Pattern resolution failed for ${pattern.pattern}: ${errorMessage}`);

        errors.push({
          id: `pattern_error_${Date.now()}_${pattern.id}`,
          type: 'pattern_error',
          message: errorMessage,
          source: `Pattern:${pattern.id}`,
          severity: 'medium',
          timestamp: new Date(),
          context: { pattern: pattern.pattern, type: pattern.type }
        });
      }
    }

    // Remove duplicates while preserving order
    const uniqueNodes = this.removeDuplicateNodes(allNodes);

    return uniqueNodes;
  }

  /**
   * Resolve a single pattern to nodes
   */
  private async resolvePattern(
    pattern: NodePattern,
    errors: CollectionProcessingError[]
  ): Promise<ProcessedNode[]> {
    switch (pattern.type) {
      case 'wildcard':
        return await this.wildcardProcessor.processWildcard(pattern, errors);

      case 'regex':
        return await this.processRegexPattern(pattern, errors);

      case 'list':
        return await this.processListPattern(pattern, errors);

      case 'query':
        return await this.processQueryPattern(pattern, errors);

      case 'cognitive':
        return await this.processCognitivePattern(pattern, errors);

      default:
        throw new Error(`Unsupported pattern type: ${pattern.type}`);
    }
  }

  /**
   * Process regex pattern
   */
  private async processRegexPattern(
    pattern: NodePattern,
    errors: CollectionProcessingError[]
  ): Promise<ProcessedNode[]> {
    try {
      const regex = new RegExp(pattern.pattern);
      const allNodes = await this.getAllNodes();

      const matchedNodes = allNodes.filter(node =>
        regex.test(node.id) || regex.test(node.name)
      );

      return this.applyPatternFilters(matchedNodes, pattern);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Invalid regex';
      throw new Error(`Regex pattern processing failed: ${errorMessage}`);
    }
  }

  /**
   * Process list pattern
   */
  private async processListPattern(
    pattern: NodePattern,
    errors: CollectionProcessingError[]
  ): Promise<ProcessedNode[]> {
    const nodeIds = pattern.pattern.split(',').map(id => id.trim());
    const nodes: ProcessedNode[] = [];

    for (const nodeId of nodeIds) {
      try {
        const node = await this.getNodeById(nodeId);
        if (node) {
          nodes.push(node);
        }
      } catch (error) {
        console.warn(`Node ${nodeId} not found: ${error}`);
        // Continue with other nodes
      }
    }

    return this.applyPatternFilters(nodes, pattern);
  }

  /**
   * Process query pattern
   */
  private async processQueryPattern(
    pattern: NodePattern,
    errors: CollectionProcessingError[]
  ): Promise<ProcessedNode[]> {
    try {
      // Parse query pattern (e.g., "nodeType=ENB,location=Paris")
      const query = this.parseQueryPattern(pattern.pattern);
      const allNodes = await this.getAllNodes();

      const matchedNodes = allNodes.filter(node =>
        this.matchesQuery(node, query)
      );

      return this.applyPatternFilters(matchedNodes, pattern);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Invalid query';
      throw new Error(`Query pattern processing failed: ${errorMessage}`);
    }
  }

  /**
   * Process cognitive pattern (AI-enhanced node selection)
   */
  private async processCognitivePattern(
    pattern: NodePattern,
    errors: CollectionProcessingError[]
  ): Promise<ProcessedNode[]> {
    try {
      // Cognitive patterns use ML/AI for intelligent node selection
      const allNodes = await this.getAllNodes();

      // Apply cognitive algorithms for pattern matching
      const cognitiveMatches = await this.applyCognitiveMatching(allNodes, pattern.pattern);

      return this.applyPatternFilters(cognitiveMatches, pattern);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Cognitive processing failed';
      throw new Error(`Cognitive pattern processing failed: ${errorMessage}`);
    }
  }

  /**
   * Apply pattern filters (inclusions/exclusions)
   */
  private applyPatternFilters(nodes: ProcessedNode[], pattern: NodePattern): ProcessedNode[] {
    let filteredNodes = [...nodes];

    // Apply exclusions
    if (pattern.exclusions && pattern.exclusions.length > 0) {
      filteredNodes = filteredNodes.filter(node => {
        return !pattern.exclusions!.some(exclusion =>
          this.matchesPattern(node, exclusion)
        );
      });
    }

    // Apply inclusions
    if (pattern.inclusions && pattern.inclusions.length > 0) {
      filteredNodes = filteredNodes.filter(node => {
        return pattern.inclusions!.some(inclusion =>
          this.matchesPattern(node, inclusion)
        );
      });
    }

    return filteredNodes;
  }

  /**
   * Check if node matches a pattern
   */
  private matchesPattern(node: ProcessedNode, pattern: string): boolean {
    // Simple wildcard matching
    if (pattern.includes('*')) {
      const regex = new RegExp(pattern.replace(/\*/g, '.*'));
      return regex.test(node.id) || regex.test(node.name);
    }

    // Exact match
    return node.id === pattern || node.name === pattern;
  }

  /**
   * Apply scope filters to nodes
   */
  private async applyScopeFilters(
    nodes: ProcessedNode[],
    scopeFilters: ScopeFilter[],
    errors: CollectionProcessingError[]
  ): Promise<ProcessedNode[]> {
    if (scopeFilters.length === 0) {
      return nodes;
    }

    let filteredNodes = [...nodes];

    // Sort filters by priority
    const sortedFilters = [...scopeFilters].sort((a, b) => b.priority - a.priority);

    for (const filter of sortedFilters) {
      try {
        const filterResult = await this.scopeFilterEngine.applyFilter(filteredNodes, filter);

        // Apply filter action
        switch (filter.action) {
          case 'include':
            filteredNodes = filterResult.matchedNodes;
            break;
          case 'exclude':
            filteredNodes = filterResult.nonMatchedNodes;
            break;
          case 'prioritize':
            // Move matching nodes to the front
            const prioritized = [
              ...filterResult.matchedNodes,
              ...filterResult.nonMatchedNodes.filter(node =>
                !filterResult.matchedNodes.some(matched => matched.id === node.id)
              )
            ];
            filteredNodes = prioritized;
            break;
        }

      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        console.error(`Scope filter application failed: ${errorMessage}`);

        errors.push({
          id: `filter_error_${Date.now()}_${filter.id}`,
          type: 'filter_error',
          message: errorMessage,
          source: `Filter:${filter.id}`,
          severity: 'medium',
          timestamp: new Date(),
          context: { filterType: filter.type, action: filter.action }
        });
      }
    }

    return filteredNodes;
  }

  /**
   * Prioritize nodes based on collection and node attributes
   */
  private prioritizeNodes(nodes: ProcessedNode[], collection: NodeCollection): ProcessedNode[] {
    // Sort nodes based on multiple criteria
    return nodes.sort((a, b) => {
      // Priority 1: Node status (prefer active nodes)
      const statusPriority = this.getStatusPriority(a.status) - this.getStatusPriority(b.status);
      if (statusPriority !== 0) return statusPriority;

      // Priority 2: Synchronization status (prefer synchronized nodes)
      const syncPriority = this.getSyncPriority(a.syncStatus) - this.getSyncPriority(b.syncStatus);
      if (syncPriority !== 0) return syncPriority;

      // Priority 3: Node type (prefer certain types)
      const typePriority = this.getNodeTypePriority(a.nodeType) - this.getNodeTypePriority(b.nodeType);
      if (typePriority !== 0) return typePriority;

      // Priority 4: Alphabetical order
      return a.id.localeCompare(b.id);
    });
  }

  /**
   * Validate nodes
   */
  private async validateNodes(
    nodes: ProcessedNode[],
    errors: CollectionProcessingError[]
  ): Promise<ProcessedNode[]> {
    const validatedNodes: ProcessedNode[] = [];

    for (const node of nodes) {
      try {
        if (this.isValidNode(node)) {
          validatedNodes.push(node);
        } else {
          console.warn(`Node ${node.id} failed validation`);
          errors.push({
            id: `validation_error_${Date.now()}_${node.id}`,
            type: 'validation_error',
            message: `Node failed validation checks`,
            source: `Node:${node.id}`,
            severity: 'low',
            timestamp: new Date(),
            context: { nodeType: node.nodeType, status: node.status }
          });
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Validation error';
        console.error(`Node validation failed for ${node.id}: ${errorMessage}`);

        errors.push({
          id: `validation_error_${Date.now()}_${node.id}`,
          type: 'validation_error',
          message: errorMessage,
          source: `Node:${node.id}`,
          severity: 'medium',
          timestamp: new Date()
        });
      }
    }

    return validatedNodes;
  }

  /**
   * Check if node is valid
   */
  private isValidNode(node: ProcessedNode): boolean {
    // Basic validation checks
    if (!node.id || !node.name || !node.nodeType) {
      return false;
    }

    // Check if node is accessible
    if (node.status === 'unreachable' || node.status === 'maintenance') {
      return false;
    }

    // Check synchronization status
    if (node.syncStatus === 'out_of_sync' || node.syncStatus === 'unknown') {
      return false;
    }

    return true;
  }

  /**
   * Remove duplicate nodes
   */
  private removeDuplicateNodes(nodes: ProcessedNode[]): ProcessedNode[] {
    const seen = new Set<string>();
    const uniqueNodes: ProcessedNode[] = [];

    for (const node of nodes) {
      if (!seen.has(node.id)) {
        seen.add(node.id);
        uniqueNodes.push(node);
      }
    }

    return uniqueNodes;
  }

  /**
   * Get all nodes (mock implementation)
   */
  private async getAllNodes(): Promise<ProcessedNode[]> {
    // In production, this would query the actual network management system
    // For now, return mock data

    const cacheKey = 'all_nodes';
    if (this.nodeCache.has(cacheKey)) {
      return this.nodeCache.get(cacheKey)!;
    }

    const mockNodes: ProcessedNode[] = [
      {
        id: 'ERBS001',
        name: 'Paris-Central-ENB',
        nodeType: 'ENB',
        neType: 'ENB',
        status: 'active',
        syncStatus: 'synchronized',
        location: 'Paris',
        version: '21B',
        vendor: 'Ericsson',
        attributes: {
          ip: '192.168.1.1',
          model: 'AIR 6488',
          capacity: 'high'
        },
        metadata: {
          source: 'NMS',
          patternMatch: 'ERBS*',
          filterResults: {},
          processingTime: 0
        }
      },
      {
        id: 'ERBS002',
        name: 'Paris-North-ENB',
        nodeType: 'ENB',
        neType: 'ENB',
        status: 'active',
        syncStatus: 'synchronized',
        location: 'Paris',
        version: '21B',
        vendor: 'Ericsson',
        attributes: {
          ip: '192.168.1.2',
          model: 'AIR 6488',
          capacity: 'medium'
        },
        metadata: {
          source: 'NMS',
          patternMatch: 'ERBS*',
          filterResults: {},
          processingTime: 0
        }
      },
      {
        id: 'GNB001',
        name: 'Paris-Central-GNB',
        nodeType: 'GNB',
        neType: 'GNB',
        status: 'active',
        syncStatus: 'synchronized',
        location: 'Paris',
        version: '22A',
        vendor: 'Ericsson',
        attributes: {
          ip: '192.168.2.1',
          model: 'AIR 6653',
          capacity: 'high'
        },
        metadata: {
          source: 'NMS',
          patternMatch: 'GNB*',
          filterResults: {},
          processingTime: 0
        }
      },
      {
        id: 'ERBS003',
        name: 'Lyon-Central-ENB',
        nodeType: 'ENB',
        neType: 'ENB',
        status: 'active',
        syncStatus: 'out_of_sync',
        location: 'Lyon',
        version: '20B',
        vendor: 'Ericsson',
        attributes: {
          ip: '192.168.1.3',
          model: 'AIR 6488',
          capacity: 'medium'
        },
        metadata: {
          source: 'NMS',
          patternMatch: 'ERBS*',
          filterResults: {},
          processingTime: 0
        }
      }
    ];

    this.nodeCache.set(cacheKey, mockNodes);
    return mockNodes;
  }

  /**
   * Get node by ID
   */
  private async getNodeById(nodeId: string): Promise<ProcessedNode | null> {
    const allNodes = await this.getAllNodes();
    return allNodes.find(node => node.id === nodeId) || null;
  }

  /**
   * Parse query pattern
   */
  private parseQueryPattern(query: string): Record<string, string> {
    const conditions: Record<string, string> = {};

    // Split by commas and parse each condition
    const parts = query.split(',');

    for (const part of parts) {
      const [key, value] = part.split('=').map(s => s.trim());
      if (key && value) {
        conditions[key] = value;
      }
    }

    return conditions;
  }

  /**
   * Check if node matches query
   */
  private matchesQuery(node: ProcessedNode, query: Record<string, string>): boolean {
    for (const [key, value] of Object.entries(query)) {
      const nodeValue = this.getNodeAttributeValue(node, key);

      if (nodeValue === undefined || nodeValue !== value) {
        return false;
      }
    }

    return true;
  }

  /**
   * Get node attribute value
   */
  private getNodeAttributeValue(node: ProcessedNode, attribute: string): string | undefined {
    switch (attribute) {
      case 'nodeType':
        return node.nodeType;
      case 'neType':
        return node.neType;
      case 'status':
        return node.status;
      case 'syncStatus':
        return node.syncStatus;
      case 'location':
        return node.location;
      case 'version':
        return node.version;
      case 'vendor':
        return node.vendor;
      default:
        return node.attributes[attribute];
    }
  }

  /**
   * Apply cognitive matching (mock implementation)
   */
  private async applyCognitiveMatching(
    nodes: ProcessedNode[],
    cognitivePattern: string
  ): Promise<ProcessedNode[]> {
    // In production, this would use actual AI/ML algorithms
    // For now, implement simple heuristic matching

    const pattern = cognitivePattern.toLowerCase();

    return nodes.filter(node => {
      // Simple cognitive matching based on patterns
      const nodeData = `${node.id} ${node.name} ${node.nodeType} ${node.location} ${JSON.stringify(node.attributes)}`.toLowerCase();

      // Check for semantic matches
      if (pattern.includes('high') && node.attributes.capacity === 'high') {
        return true;
      }

      if (pattern.includes('paris') && node.location === 'Paris') {
        return true;
      }

      if (pattern.includes('5g') && node.nodeType === 'GNB') {
        return true;
      }

      if (pattern.includes('4g') && node.nodeType === 'ENB') {
        return true;
      }

      // Check for fuzzy matches
      const similarity = this.calculateStringSimilarity(pattern, nodeData);
      return similarity > 0.3; // 30% similarity threshold
    });
  }

  /**
   * Calculate string similarity (simple implementation)
   */
  private calculateStringSimilarity(str1: string, str2: string): number {
    const longer = str1.length > str2.length ? str1 : str2;
    const shorter = str1.length > str2.length ? str2 : str1;

    if (longer.length === 0) return 1.0;

    const editDistance = this.calculateEditDistance(longer, shorter);
    return (longer.length - editDistance) / longer.length;
  }

  /**
   * Calculate edit distance
   */
  private calculateEditDistance(str1: string, str2: string): number {
    const matrix = Array(str2.length + 1).fill(null).map(() => Array(str1.length + 1).fill(null));

    for (let i = 0; i <= str1.length; i++) matrix[0][i] = i;
    for (let j = 0; j <= str2.length; j++) matrix[j][0] = j;

    for (let j = 1; j <= str2.length; j++) {
      for (let i = 1; i <= str1.length; i++) {
        const indicator = str1[i - 1] === str2[j - 1] ? 0 : 1;
        matrix[j][i] = Math.min(
          matrix[j][i - 1] + 1,
          matrix[j - 1][i] + 1,
          matrix[j - 1][i - 1] + indicator
        );
      }
    }

    return matrix[str2.length][str1.length];
  }

  /**
   * Get status priority
   */
  private getStatusPriority(status: string): number {
    const priorities: Record<string, number> = {
      'active': 4,
      'standby': 3,
      'maintenance': 2,
      'unreachable': 1,
      'unknown': 0
    };

    return priorities[status] || 0;
  }

  /**
   * Get sync priority
   */
  private getSyncPriority(syncStatus: string): number {
    const priorities: Record<string, number> = {
      'synchronized': 3,
      'synchronizing': 2,
      'out_of_sync': 1,
      'unknown': 0
    };

    return priorities[syncStatus] || 0;
  }

  /**
   * Get node type priority
   */
  private getNodeTypePriority(nodeType: string): number {
    const priorities: Record<string, number> = {
      'GNB': 4,  // 5G nodes prioritized
      'ENB': 3,  // 4G nodes
      'RNC': 2,  // 3G nodes
      'BSC': 1,  // 2G nodes
      'default': 0
    };

    return priorities[nodeType] || priorities['default'];
  }

  /**
   * Clear cache
   */
  public clearCache(): void {
    this.nodeCache.clear();
    this.patternCache.clear();
  }

  /**
   * Get cache statistics
   */
  public getCacheStatistics(): {
    nodeCacheSize: number;
    patternCacheSize: number;
    cacheKeys: string[];
  } {
    return {
      nodeCacheSize: this.nodeCache.size,
      patternCacheSize: this.patternCache.size,
      cacheKeys: Array.from(this.nodeCache.keys())
    };
  }
}