/**
 * Scope Filter Engine for Intelligent Node Filtering
 *
 * Provides advanced filtering capabilities for node selection based on various
 * criteria including sync status, NE type, version, location, and performance metrics.
 */

import {
  ScopeFilter,
  FilterCondition,
  ProcessedNode
} from '../core/types';

/**
 * Filter application result
 */
export interface FilterResult {
  /** Nodes that matched the filter */
  matchedNodes: ProcessedNode[];
  /** Nodes that did not match the filter */
  nonMatchedNodes: ProcessedNode[];
  /** Filter application statistics */
  statistics: {
    totalNodes: number;
    matchedNodes: number;
    nonMatchedNodes: number;
    filterTime: number;
    conditionsEvaluated: number;
  };
  /** Filter evaluation details */
  evaluationDetails: FilterEvaluationDetail[];
}

/**
 * Filter evaluation detail
 */
export interface FilterEvaluationDetail {
  /** Node identifier */
  nodeId: string;
  /** Condition evaluation results */
  conditionResults: Record<string, boolean>;
  /** Overall match result */
  matched: boolean;
  /** Evaluation time in milliseconds */
  evaluationTime: number;
}

/**
 * Scope Filter Engine
 */
export class ScopeFilterEngine {
  private filterCache: Map<string, FilterResult> = new Map();
  private customFilters: Map<string, CustomFilterFunction> = new Map();

  constructor() {
    this.initializeBuiltInFilters();
  }

  /**
   * Apply a scope filter to nodes
   */
  public async applyFilter(
    nodes: ProcessedNode[],
    filter: ScopeFilter
  ): Promise<FilterResult> {
    const startTime = Date.now();
    const cacheKey = this.generateCacheKey(nodes, filter);

    // Check cache first
    if (this.filterCache.has(cacheKey)) {
      const cachedResult = this.filterCache.get(cacheKey)!;
      console.log(`Using cached filter result for ${filter.id}`);
      return cachedResult;
    }

    console.log(`Applying scope filter ${filter.id} (${filter.type}) to ${nodes.length} nodes`);

    const evaluationDetails: FilterEvaluationDetail[] = [];
    const matchedNodes: ProcessedNode[] = [];
    const nonMatchedNodes: ProcessedNode[] = [];
    let conditionsEvaluated = 0;

    // Evaluate each node against the filter
    for (const node of nodes) {
      const nodeStartTime = Date.now();
      const conditionResults: Record<string, boolean> = {};

      try {
        const matched = await this.evaluateNodeAgainstFilter(node, filter, conditionResults);
        conditionsEvaluated += Object.keys(conditionResults).length;

        const evaluationDetail: FilterEvaluationDetail = {
          nodeId: node.id,
          conditionResults,
          matched,
          evaluationTime: Date.now() - nodeStartTime
        };

        evaluationDetails.push(evaluationDetail);

        if (matched) {
          matchedNodes.push(node);
        } else {
          nonMatchedNodes.push(node);
        }

      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        console.error(`Filter evaluation failed for node ${node.id}: ${errorMessage}`);

        // Treat evaluation errors as non-matching
        nonMatchedNodes.push(node);

        evaluationDetails.push({
          nodeId: node.id,
          conditionResults: { 'error': false },
          matched: false,
          evaluationTime: Date.now() - nodeStartTime
        });
      }
    }

    const filterTime = Date.now() - startTime;

    const result: FilterResult = {
      matchedNodes,
      nonMatchedNodes,
      statistics: {
        totalNodes: nodes.length,
        matchedNodes: matchedNodes.length,
        nonMatchedNodes: nonMatchedNodes.length,
        filterTime,
        conditionsEvaluated
      },
      evaluationDetails
    };

    // Cache the result
    this.filterCache.set(cacheKey, result);

    console.log(`Filter ${filter.id} completed: ${matchedNodes.length}/${nodes.length} nodes matched in ${filterTime}ms`);

    return result;
  }

  /**
   * Evaluate a node against a filter
   */
  private async evaluateNodeAgainstFilter(
    node: ProcessedNode,
    filter: ScopeFilter,
    conditionResults: Record<string, boolean>
  ): Promise<boolean> {
    switch (filter.type) {
      case 'sync_status':
        return this.evaluateSyncStatusFilter(node, filter.condition, conditionResults);

      case 'ne_type':
        return this.evaluateNeTypeFilter(node, filter.condition, conditionResults);

      case 'vendor':
        return this.evaluateVendorFilter(node, filter.condition, conditionResults);

      case 'version':
        return this.evaluateVersionFilter(node, filter.condition, conditionResults);

      case 'location':
        return this.evaluateLocationFilter(node, filter.condition, conditionResults);

      case 'performance':
        return await this.evaluatePerformanceFilter(node, filter.condition, conditionResults);

      case 'custom':
        return await this.evaluateCustomFilter(node, filter, conditionResults);

      default:
        throw new Error(`Unsupported filter type: ${filter.type}`);
    }
  }

  /**
   * Evaluate sync status filter
   */
  private evaluateSyncStatusFilter(
    node: ProcessedNode,
    condition: FilterCondition,
    conditionResults: Record<string, boolean>
  ): boolean {
    const syncStatus = node.syncStatus;
    const result = this.evaluateCondition(syncStatus, condition);
    conditionResults['sync_status'] = result;
    return result;
  }

  /**
   * Evaluate NE type filter
   */
  private evaluateNeTypeFilter(
    node: ProcessedNode,
    condition: FilterCondition,
    conditionResults: Record<string, boolean>
  ): boolean {
    const neType = node.neType;
    const result = this.evaluateCondition(neType, condition);
    conditionResults['ne_type'] = result;
    return result;
  }

  /**
   * Evaluate vendor filter
   */
  private evaluateVendorFilter(
    node: ProcessedNode,
    condition: FilterCondition,
    conditionResults: Record<string, boolean>
  ): boolean {
    const vendor = node.vendor || 'unknown';
    const result = this.evaluateCondition(vendor, condition);
    conditionResults['vendor'] = result;
    return result;
  }

  /**
   * Evaluate version filter
   */
  private evaluateVersionFilter(
    node: ProcessedNode,
    condition: FilterCondition,
    conditionResults: Record<string, boolean>
  ): boolean {
    const version = node.version || 'unknown';
    const result = this.evaluateCondition(version, condition);
    conditionResults['version'] = result;
    return result;
  }

  /**
   * Evaluate location filter
   */
  private evaluateLocationFilter(
    node: ProcessedNode,
    condition: FilterCondition,
    conditionResults: Record<string, boolean>
  ): boolean {
    const location = node.location || 'unknown';
    const result = this.evaluateCondition(location, condition);
    conditionResults['location'] = result;
    return result;
  }

  /**
   * Evaluate performance filter
   */
  private async evaluatePerformanceFilter(
    node: ProcessedNode,
    condition: FilterCondition,
    conditionResults: Record<string, boolean>
  ): Promise<boolean> {
    // Get performance metrics for the node
    const performanceMetrics = await this.getNodePerformanceMetrics(node);

    let result = false;

    switch (condition.attribute) {
      case 'cpu_usage':
        result = this.evaluateCondition(performanceMetrics.cpuUsage.toString(), condition);
        break;
      case 'memory_usage':
        result = this.evaluateCondition(performanceMetrics.memoryUsage.toString(), condition);
        break;
      case 'throughput':
        result = this.evaluateCondition(performanceMetrics.throughput.toString(), condition);
        break;
      case 'latency':
        result = this.evaluateCondition(performanceMetrics.latency.toString(), condition);
        break;
      case 'error_rate':
        result = this.evaluateCondition(performanceMetrics.errorRate.toString(), condition);
        break;
      default:
        throw new Error(`Unsupported performance metric: ${condition.attribute}`);
    }

    conditionResults[`performance_${condition.attribute}`] = result;
    return result;
  }

  /**
   * Evaluate custom filter
   */
  private async evaluateCustomFilter(
    node: ProcessedNode,
    filter: ScopeFilter,
    conditionResults: Record<string, boolean>
  ): Promise<boolean> {
    const customFilter = this.customFilters.get(filter.type);

    if (!customFilter) {
      throw new Error(`Custom filter not found: ${filter.type}`);
    }

    const result = await customFilter(node, filter.condition);
    conditionResults[`custom_${filter.type}`] = result;
    return result;
  }

  /**
   * Evaluate a single condition
   */
  private evaluateCondition(
    value: string,
    condition: FilterCondition
  ): boolean {
    // Handle logical operators
    if (condition.conditions && condition.conditions.length > 0) {
      return this.evaluateLogicalCondition(value, condition);
    }

    // Simple condition evaluation
    switch (condition.operator) {
      case 'eq':
        return value === condition.value.toString();

      case 'ne':
        return value !== condition.value.toString();

      case 'gt':
        return this.compareNumeric(value, condition.value) > 0;

      case 'gte':
        return this.compareNumeric(value, condition.value) >= 0;

      case 'lt':
        return this.compareNumeric(value, condition.value) < 0;

      case 'lte':
        return this.compareNumeric(value, condition.value) <= 0;

      case 'in':
        const values = Array.isArray(condition.value) ? condition.value : [condition.value];
        return values.includes(value);

      case 'not_in':
        const excludeValues = Array.isArray(condition.value) ? condition.value : [condition.value];
        return !excludeValues.includes(value);

      case 'contains':
        return value.toLowerCase().includes(condition.value.toString().toLowerCase());

      case 'regex':
        try {
          const regex = new RegExp(condition.value.toString());
          return regex.test(value);
        } catch (error) {
          console.error(`Invalid regex pattern: ${condition.value}`);
          return false;
        }

      default:
        throw new Error(`Unsupported operator: ${condition.operator}`);
    }
  }

  /**
   * Evaluate logical condition
   */
  private evaluateLogicalCondition(value: string, condition: FilterCondition): boolean {
    const results = condition.conditions!.map(cond => this.evaluateCondition(value, cond));

    switch (condition.logicalOperator) {
      case 'and':
        return results.every(result => result);

      case 'or':
        return results.some(result => result);

      case 'not':
        return !results[0]; // NOT applies to first condition

      default:
        return results[0];
    }
  }

  /**
   * Compare numeric values
   */
  private compareNumeric(value1: string, value2: any): number {
    const num1 = parseFloat(value1);
    const num2 = parseFloat(value2.toString());

    if (isNaN(num1) || isNaN(num2)) {
      throw new Error(`Cannot compare non-numeric values: ${value1}, ${value2}`);
    }

    return num1 - num2;
  }

  /**
   * Get node performance metrics (mock implementation)
   */
  private async getNodePerformanceMetrics(node: ProcessedNode): Promise<{
    cpuUsage: number;
    memoryUsage: number;
    throughput: number;
    latency: number;
    errorRate: number;
  }> {
    // In production, this would query actual performance monitoring systems
    // For now, return mock data with some realistic variations

    const baseMetrics = {
      cpuUsage: 30 + Math.random() * 40,  // 30-70%
      memoryUsage: 40 + Math.random() * 30,  // 40-70%
      throughput: 800 + Math.random() * 400,  // 800-1200 Mbps
      latency: 5 + Math.random() * 15,  // 5-20 ms
      errorRate: Math.random() * 2  // 0-2%
    };

    // Add node-specific variations
    if (node.attributes.capacity === 'high') {
      baseMetrics.cpuUsage *= 0.8;  // High capacity nodes use less CPU
      baseMetrics.throughput *= 1.5;  // Higher throughput
    }

    if (node.nodeType === 'GNB') {
      baseMetrics.latency *= 0.7;  // 5G nodes have lower latency
      baseMetrics.throughput *= 1.2;  // Higher throughput
    }

    return baseMetrics;
  }

  /**
   * Generate cache key
   */
  private generateCacheKey(nodes: ProcessedNode[], filter: ScopeFilter): string {
    const nodeIds = nodes.map(node => node.id).sort().join(',');
    const filterString = `${filter.type}_${filter.condition.attribute}_${filter.condition.operator}_${filter.condition.value}`;
    const hash = this.simpleHash(nodeIds + filterString);
    return `filter_${filter.id}_${hash}`;
  }

  /**
   * Simple hash function
   */
  private simpleHash(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }

  /**
   * Initialize built-in filters
   */
  private initializeBuiltInFilters(): void {
    // Active nodes filter
    this.addCustomFilter('active_nodes', async (node, condition) => {
      return node.status === 'active';
    });

    // High capacity nodes filter
    this.addCustomFilter('high_capacity', async (node, condition) => {
      return node.attributes.capacity === 'high';
    });

    // Low latency nodes filter
    this.addCustomFilter('low_latency', async (node, condition) => {
      const metrics = await this.getNodePerformanceMetrics(node);
      return metrics.latency < 10; // Less than 10ms
    });

    // High throughput nodes filter
    this.addCustomFilter('high_throughput', async (node, condition) => {
      const metrics = await this.getNodePerformanceMetrics(node);
      return metrics.throughput > 1000; // Greater than 1000 Mbps
    });

    // Recent version filter
    this.addCustomFilter('recent_version', async (node, condition) => {
      if (!node.version) return false;

      // Extract version number (e.g., "21B" -> 21)
      const versionMatch = node.version.match(/(\d+)/);
      if (!versionMatch) return false;

      const versionNumber = parseInt(versionMatch[1]);
      const currentYear = new Date().getFullYear() - 2000; // Convert to 2-digit year
      return versionNumber >= currentYear - 1; // Within last 2 versions
    });

    // Healthy nodes filter
    this.addCustomFilter('healthy_nodes', async (node, condition) => {
      const metrics = await this.getNodePerformanceMetrics(node);
      return (
        node.status === 'active' &&
        node.syncStatus === 'synchronized' &&
        metrics.cpuUsage < 80 &&
        metrics.memoryUsage < 80 &&
        metrics.errorRate < 1
      );
    });
  }

  /**
   * Add custom filter
   */
  public addCustomFilter(name: string, filterFunction: CustomFilterFunction): void {
    this.customFilters.set(name, filterFunction);
    console.log(`Custom filter registered: ${name}`);
  }

  /**
   * Remove custom filter
   */
  public removeCustomFilter(name: string): boolean {
    return this.customFilters.delete(name);
  }

  /**
   * Get available filters
   */
  public getAvailableFilters(): string[] {
    return [
      'sync_status',
      'ne_type',
      'vendor',
      'version',
      'location',
      'performance',
      ...Array.from(this.customFilters.keys())
    ];
  }

  /**
   * Clear cache
   */
  public clearCache(): void {
    this.filterCache.clear();
  }

  /**
   * Get cache statistics
   */
  public getCacheStatistics(): {
    cacheSize: number;
    cacheKeys: string[];
    totalCacheHits: number;
  } {
    return {
      cacheSize: this.filterCache.size,
      cacheKeys: Array.from(this.filterCache.keys()),
      totalCacheHits: 0 // In production, track actual cache hits
    };
  }

  /**
   * Validate filter configuration
   */
  public validateFilter(filter: ScopeFilter): string[] {
    const errors: string[] = [];

    // Check filter type
    const availableTypes = this.getAvailableFilters();
    if (!availableTypes.includes(filter.type)) {
      errors.push(`Invalid filter type: ${filter.type}. Available types: ${availableTypes.join(', ')}`);
    }

    // Validate condition
    if (!filter.condition) {
      errors.push('Filter condition is required');
    } else {
      const conditionErrors = this.validateCondition(filter.condition);
      errors.push(...conditionErrors);
    }

    // Validate action
    const validActions = ['include', 'exclude', 'prioritize'];
    if (!validActions.includes(filter.action)) {
      errors.push(`Invalid filter action: ${filter.action}. Valid actions: ${validActions.join(', ')}`);
    }

    return errors;
  }

  /**
   * Validate condition
   */
  private validateCondition(condition: FilterCondition): string[] {
    const errors: string[] = [];

    if (!condition.attribute) {
      errors.push('Condition attribute is required');
    }

    if (!condition.operator) {
      errors.push('Condition operator is required');
    }

    if (condition.value === undefined || condition.value === null) {
      errors.push('Condition value is required');
    }

    // Validate operator
    const validOperators = ['eq', 'ne', 'gt', 'gte', 'lt', 'lte', 'in', 'not_in', 'contains', 'regex'];
    if (condition.operator && !validOperators.includes(condition.operator)) {
      errors.push(`Invalid operator: ${condition.operator}. Valid operators: ${validOperators.join(', ')}`);
    }

    // Validate logical operator if specified
    if (condition.logicalOperator) {
      const validLogicalOperators = ['and', 'or', 'not'];
      if (!validLogicalOperators.includes(condition.logicalOperator)) {
        errors.push(`Invalid logical operator: ${condition.logicalOperator}. Valid operators: ${validLogicalOperators.join(', ')}`);
      }
    }

    return errors;
  }
}

/**
 * Custom filter function type
 */
export type CustomFilterFunction = (
  node: ProcessedNode,
  condition: FilterCondition
) => Promise<boolean>;