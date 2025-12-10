/**
 * Inheritance Resolver - Advanced Template Inheritance Chain Processing
 *
 * Handles complex inheritance hierarchies, circular dependency detection,
 * parameter conflict resolution, and template merging strategies.
 */

import {
  RTBTemplate,
  TemplateMeta,
  CustomFunction,
  ConditionOperator,
  EvaluationOperator
} from '../../types/rtb-types';
import {
  TemplatePriorityInfo,
  TemplateInheritanceChain,
  ParameterConflict,
  TemplateResolutionContext,
  TemplatePriority,
  ValidationError,
  ValidationWarning
} from './priority-engine';
import { TemplateRegistry, RegistryTemplateMeta } from './template-registry';

/**
 * Inheritance resolution strategy
 */
export enum InheritanceStrategy {
  OVERRIDE = 'override',         // Child overrides parent
  MERGE = 'merge',              // Merge values intelligently
  APPEND = 'append',            // Append arrays/lists
  INTERSECT = 'intersect',      // Keep only common values
  CUSTOM = 'custom'             // Use custom resolver function
}

/**
 * Circular dependency information
 */
export interface CircularDependency {
  chain: string[];
  cycleStart: number;
  severity: 'error' | 'warning';
  resolution: 'break_cycle' | 'ignore_cycle' | 'raise_error';
  message: string;
}

/**
 * Inheritance graph node
 */
export interface InheritanceNode {
  name: string;
  template: RTBTemplate;
  priority: TemplatePriorityInfo;
  children: Set<string>;
  parents: Set<string>;
  depth: number;
  visited: boolean;
  visiting: boolean;
  resolved: boolean;
}

/**
 * Merge conflict resolution rule
 */
export interface MergeConflictRule {
  parameterPattern: RegExp;
  strategy: InheritanceStrategy;
  customResolver?: (values: any[], contexts: any[]) => any;
  priority?: 'highest' | 'lowest' | 'average';
  conditions?: Record<string, any>;
}

/**
 * Inheritance resolution options
 */
export interface InheritanceResolutionOptions {
  strategy?: InheritanceStrategy;
  conflictRules?: MergeConflictRule[];
  allowCircularDependencies?: boolean;
  maxInheritanceDepth?: number;
  preserveComments?: boolean;
  validateInheritance?: boolean;
  trackResolution?: boolean;
  optimizeMerging?: boolean;
}

/**
 * Parameter merge context
 */
export interface ParameterMergeContext {
  parameter: string;
  values: Array<{
    value: any;
    source: string;
    priority: number;
    path: string[];
  }>;
  mergedValue?: any;
  conflicts: ParameterConflict[];
  warnings: string[];
}

/**
 * Inheritance analysis result
 */
export interface InheritanceAnalysisResult {
  templateName: string;
  inheritanceDepth: number;
  totalDependencies: number;
  circularDependencies: CircularDependency[];
  parameterConflicts: ParameterConflict[];
  mergeComplexity: number;
  estimatedProcessingTime: number;
  recommendations: string[];
}

/**
 * Inheritance Resolver
 *
 * Advanced system for resolving complex template inheritance chains,
 * detecting circular dependencies, and managing parameter conflicts.
 */
export class InheritanceResolver {
  private registry: TemplateRegistry;
  private inheritanceGraph = new Map<string, InheritanceNode>();
  private resolutionCache = new Map<string, TemplateInheritanceChain>();
  private analysisCache = new Map<string, InheritanceAnalysisResult>();
  private conflictRules: Map<string, MergeConflictRule[]> = new Map();

  constructor(registry: TemplateRegistry) {
    this.registry = registry;
    this.initializeDefaultConflictRules();
  }

  /**
   * Resolve template inheritance with advanced conflict handling
   */
  async resolveInheritance(
    templateName: string,
    options: InheritanceResolutionOptions = {}
  ): Promise<TemplateInheritanceChain> {
    const cacheKey = this.generateResolutionCacheKey(templateName, options);

    // Check cache first
    if (this.resolutionCache.has(cacheKey)) {
      return this.resolutionCache.get(cacheKey)!;
    }

    // Build inheritance graph
    await this.buildInheritanceGraph();

    // Validate inheritance structure
    const validationErrors = await this.validateInheritanceStructure(templateName, options);
    if (validationErrors.length > 0 && !options.allowCircularDependencies) {
      throw new Error(
        `Inheritance validation failed: ${validationErrors.map(e => e.message).join(', ')}`
      );
    }

    // Resolve inheritance chain
    const chain = await this.resolveInheritanceChain(templateName, options);

    // Cache result
    this.resolutionCache.set(cacheKey, chain);

    return chain;
  }

  /**
   * Analyze template inheritance complexity
   */
  async analyzeInheritance(
    templateName: string
  ): Promise<InheritanceAnalysisResult> {
    // Check cache first
    if (this.analysisCache.has(templateName)) {
      return this.analysisCache.get(templateName)!;
    }

    // Build inheritance graph
    await this.buildInheritanceGraph();

    const node = this.inheritanceGraph.get(templateName);
    if (!node) {
      throw new Error(`Template '${templateName}' not found in inheritance graph`);
    }

    // Calculate inheritance depth
    const depth = this.calculateInheritanceDepth(templateName);

    // Count dependencies
    const dependencies = this.getAllDependencies(templateName);

    // Detect circular dependencies
    const circularDeps = this.detectCircularDependencies(templateName);

    // Analyze parameter conflicts
    const conflicts = await this.analyzeParameterConflicts(templateName);

    // Calculate merge complexity
    const complexity = this.calculateMergeComplexity(templateName);

    // Estimate processing time
    const processingTime = this.estimateProcessingTime(templateName, complexity);

    // Generate recommendations
    const recommendations = this.generateRecommendations(
      templateName, depth, dependencies, circularDeps, conflicts
    );

    const result: InheritanceAnalysisResult = {
      templateName,
      inheritanceDepth: depth,
      totalDependencies: dependencies.length,
      circularDependencies: circularDeps,
      parameterConflicts: conflicts,
      mergeComplexity: complexity,
      estimatedProcessingTime: processingTime,
      recommendations
    };

    // Cache result
    this.analysisCache.set(templateName, result);

    return result;
  }

  /**
   * Build inheritance graph from all registered templates
   */
  private async buildInheritanceGraph(): Promise<void> {
    // Clear existing graph
    this.inheritanceGraph.clear();

    // Get all templates from registry
    const templates = this.registry.getRegisteredTemplates();

    // Create nodes for all templates
    for (const [name, priority] of templates) {
      const template = this.registry.getTemplate(name);
      if (template) {
        this.inheritanceGraph.set(name, {
          name,
          template,
          priority,
          children: new Set(),
          parents: new Set(),
          depth: 0,
          visited: false,
          visiting: false,
          resolved: false
        });
      }
    }

    // Build parent-child relationships
    for (const [name, node] of this.inheritanceGraph) {
      const dependencies = this.extractTemplateDependencies(node.template);
      for (const dep of dependencies) {
        const parentNode = this.inheritanceGraph.get(dep);
        if (parentNode) {
          node.parents.add(dep);
          parentNode.children.add(name);
        }
      }
    }

    // Calculate depths
    this.calculateDepths();
  }

  /**
   * Extract template dependencies
   */
  private extractTemplateDependencies(template: RTBTemplate): string[] {
    const dependencies: string[] = [];

    // From metadata
    if (template.meta?.inherits_from) {
      if (Array.isArray(template.meta.inherits_from)) {
        dependencies.push(...template.meta.inherits_from);
      } else {
        dependencies.push(template.meta.inherits_from);
      }
    }

    // From evaluations
    if (template.evaluations) {
      for (const [key, evalOp] of Object.entries(template.evaluations)) {
        // Extract template references from evaluation expressions
        const refs = this.extractTemplateReferences(evalOp.eval);
        dependencies.push(...refs);
      }
    }

    // From conditions
    if (template.conditions) {
      for (const [key, condition] of Object.entries(template.conditions)) {
        const refs = this.extractTemplateReferences(condition.if);
        dependencies.push(...refs);
        if (typeof condition.then === 'string') {
          const thenRefs = this.extractTemplateReferences(condition.then);
          dependencies.push(...thenRefs);
        }
        if (typeof condition.else === 'string') {
          const elseRefs = this.extractTemplateReferences(condition.else);
          dependencies.push(...elseRefs);
        }
      }
    }

    return [...new Set(dependencies)];
  }

  /**
   * Extract template references from expression
   */
  private extractTemplateReferences(expression: string): string[] {
    const templateRefs: string[] = [];

    // Pattern to match template names (CamelCase ending with Template)
    const templatePattern = /\b([A-Z][a-zA-Z0-9_]*Template)\b/g;
    let match;

    while ((match = templatePattern.exec(expression)) !== null) {
      templateRefs.push(match[1]);
    }

    return templateRefs;
  }

  /**
   * Calculate node depths in inheritance graph
   */
  private calculateDepths(): void {
    // Topological sort to calculate depths
    const visited = new Set<string>();
    const processing = new Set<string>();

    const calculateDepth = (nodeName: string): number => {
      if (visited.has(nodeName)) {
        return this.inheritanceGraph.get(nodeName)!.depth;
      }

      if (processing.has(nodeName)) {
        // Circular dependency detected
        return 0;
      }

      processing.add(nodeName);

      const node = this.inheritanceGraph.get(nodeName)!;
      let maxParentDepth = 0;

      for (const parentName of node.parents) {
        const parentDepth = calculateDepth(parentName);
        maxParentDepth = Math.max(maxParentDepth, parentDepth);
      }

      node.depth = maxParentDepth + 1;
      processing.delete(nodeName);
      visited.add(nodeName);

      return node.depth;
    };

    // Calculate depths for all nodes
    for (const nodeName of this.inheritanceGraph.keys()) {
      calculateDepth(nodeName);
    }
  }

  /**
   * Resolve inheritance chain with topological sort
   */
  private async resolveInheritanceChain(
    templateName: string,
    options: InheritanceResolutionOptions
  ): Promise<TemplateInheritanceChain> {
    const node = this.inheritanceGraph.get(templateName);
    if (!node) {
      throw new Error(`Template '${templateName}' not found`);
    }

    // Build inheritance chain using topological sort
    const chain: TemplatePriorityInfo[] = [];
    const conflicts: ParameterConflict[] = [];
    const warnings: string[] = [];
    const visited = new Set<string>();

    const buildChain = (nodeName: string, currentChain: string[]): void => {
      if (visited.has(nodeName)) {
        return;
      }

      // Check for circular dependencies
      if (currentChain.includes(nodeName)) {
        const cycleStart = currentChain.indexOf(nodeName);
        const cycle = currentChain.slice(cycleStart).concat(nodeName);
        warnings.push(`Circular dependency detected: ${cycle.join(' -> ')}`);
        return;
      }

      const currentNode = this.inheritanceGraph.get(nodeName)!;

      // Process parents first (inheritance order)
      for (const parentName of currentNode.parents) {
        buildChain(parentName, currentChain.concat(nodeName));
      }

      // Add current node to chain
      chain.push(currentNode.priority);
      visited.add(nodeName);
    };

    buildChain(templateName, []);

    // Merge templates in inheritance order
    const resolvedTemplate = await this.mergeTemplates(chain, conflicts, warnings, options);

    return {
      templateName,
      chain: chain.sort((a, b) => a.level - b.level),
      resolvedTemplate,
      conflicts,
      warnings
    };
  }

  /**
   * Merge templates using specified strategy
   */
  private async mergeTemplates(
    chain: TemplatePriorityInfo[],
    conflicts: ParameterConflict[],
    warnings: string[],
    options: InheritanceResolutionOptions
  ): Promise<RTBTemplate> {
    let mergedTemplate: RTBTemplate = {
      configuration: {},
      custom: [],
      conditions: {},
      evaluations: {}
    };

    // Process templates in priority order (highest to lowest)
    const sortedChain = [...chain].sort((a, b) => a.level - b.level);

    for (const priorityInfo of sortedChain) {
      const template = this.registry.getTemplate(
        this.findTemplateByPriority(priorityInfo)
      );

      if (!template) {
        warnings.push(`Template not found for priority: ${priorityInfo.category}`);
        continue;
      }

      // Merge template into result
      await this.mergeTemplate(
        mergedTemplate,
        template,
        priorityInfo,
        conflicts,
        warnings,
        options
      );
    }

    return mergedTemplate;
  }

  /**
   * Merge two templates
   */
  private async mergeTemplate(
    target: RTBTemplate,
    source: RTBTemplate,
    priorityInfo: TemplatePriorityInfo,
    conflicts: ParameterConflict[],
    warnings: string[],
    options: InheritanceResolutionOptions
  ): Promise<void> {
    // Merge configuration
    await this.mergeConfiguration(
      target.configuration || {},
      source.configuration || {},
      priorityInfo,
      conflicts,
      options
    );

    // Merge custom functions
    if (source.custom) {
      target.custom = target.custom || [];
      for (const func of source.custom) {
        const existing = target.custom.find(f => f.name === func.name);
        if (existing) {
          conflicts.push({
            parameter: `custom_function:${func.name}`,
            templates: ['target', 'source'],
            values: [existing, func],
            resolvedValue: func, // Source wins
            resolutionStrategy: 'highest_priority',
            reason: `Function override from ${priorityInfo.category}`
          });
          // Replace existing function
          const index = target.custom.indexOf(existing);
          target.custom[index] = func;
        } else {
          target.custom.push(func);
        }
      }
    }

    // Merge conditions
    if (source.conditions) {
      target.conditions = target.conditions || {};
      for (const [key, condition] of Object.entries(source.conditions)) {
        if (target.conditions[key]) {
          conflicts.push({
            parameter: `condition:${key}`,
            templates: ['target', 'source'],
            values: [target.conditions[key], condition],
            resolvedValue: condition, // Source wins
            resolutionStrategy: 'highest_priority',
            reason: `Condition override from ${priorityInfo.category}`
          });
        }
        target.conditions[key] = condition;
      }
    }

    // Merge evaluations
    if (source.evaluations) {
      target.evaluations = target.evaluations || {};
      for (const [key, evaluation] of Object.entries(source.evaluations)) {
        if (target.evaluations[key]) {
          conflicts.push({
            parameter: `evaluation:${key}`,
            templates: ['target', 'source'],
            values: [target.evaluations[key], evaluation],
            resolvedValue: evaluation, // Source wins
            resolutionStrategy: 'highest_priority',
            reason: `Evaluation override from ${priorityInfo.category}`
          });
        }
        target.evaluations[key] = evaluation;
      }
    }

    // Merge metadata
    if (source.meta) {
      target.meta = { ...target.meta, ...source.meta };
    }
  }

  /**
   * Merge configuration objects
   */
  private async mergeConfiguration(
    target: Record<string, any>,
    source: Record<string, any>,
    priorityInfo: TemplatePriorityInfo,
    conflicts: ParameterConflict[],
    options: InheritanceResolutionOptions
  ): Promise<void> {
    for (const [key, sourceValue] of Object.entries(source)) {
      if (target[key] !== undefined) {
        // Conflict detected
        const conflict = await this.resolveParameterConflict(
          key,
          target[key],
          sourceValue,
          priorityInfo,
          options
        );
        conflicts.push(conflict);
        target[key] = conflict.resolvedValue;
      } else {
        target[key] = sourceValue;
      }
    }
  }

  /**
   * Resolve parameter conflict using rules
   */
  private async resolveParameterConflict(
    parameter: string,
    targetValue: any,
    sourceValue: any,
    priorityInfo: TemplatePriorityInfo,
    options: InheritanceResolutionOptions
  ): Promise<ParameterConflict> {
    const strategy = options.strategy || InheritanceStrategy.OVERRIDE;
    let resolvedValue = sourceValue;
    let resolutionStrategy: ParameterConflict['resolutionStrategy'] = 'highest_priority';

    // Check for custom conflict rules
    const customRule = this.findConflictRule(parameter);
    if (customRule) {
      resolvedValue = await this.applyConflictRule(
        customRule,
        [targetValue, sourceValue],
        [priorityInfo]
      );
      resolutionStrategy = 'custom';
    } else {
      // Apply default strategy
      switch (strategy) {
        case InheritanceStrategy.MERGE:
          resolvedValue = this.mergeValues(targetValue, sourceValue);
          resolutionStrategy = 'merge';
          break;
        case InheritanceStrategy.APPEND:
          resolvedValue = this.appendValues(targetValue, sourceValue);
          resolutionStrategy = 'merge';
          break;
        case InheritanceStrategy.INTERSECT:
          resolvedValue = this.intersectValues(targetValue, sourceValue);
          resolutionStrategy = 'merge';
          break;
        default:
          // OVERRIDE strategy - source wins
          resolvedValue = sourceValue;
          break;
      }
    }

    return {
      parameter,
      templates: ['target', 'source'],
      values: [targetValue, sourceValue],
      resolvedValue,
      resolutionStrategy,
      reason: `Resolution using ${strategy} strategy from ${priorityInfo.category}`
    };
  }

  /**
   * Merge values intelligently
   */
  private mergeValues(target: any, source: any): any {
    // If values are the same, return source
    if (JSON.stringify(target) === JSON.stringify(source)) {
      return source;
    }

    // Array merging
    if (Array.isArray(target) && Array.isArray(source)) {
      return [...new Set([...target, ...source])];
    }

    // Object merging
    if (typeof target === 'object' && typeof source === 'object' &&
        target !== null && source !== null) {
      return { ...target, ...source };
    }

    // Default: return source (higher priority)
    return source;
  }

  /**
   * Append values
   */
  private appendValues(target: any, source: any): any {
    if (Array.isArray(target) && Array.isArray(source)) {
      return [...target, ...source];
    }

    if (typeof target === 'string' && typeof source === 'string') {
      return target + source;
    }

    return source;
  }

  /**
   * Intersect values
   */
  private intersectValues(target: any, source: any): any {
    if (Array.isArray(target) && Array.isArray(source)) {
      return target.filter(value => source.includes(value));
    }

    if (typeof target === 'object' && typeof source === 'object' &&
        target !== null && source !== null) {
      const result: any = {};
      for (const key of Object.keys(target)) {
        if (key in source) {
          result[key] = source[key];
        }
      }
      return result;
    }

    return target;
  }

  /**
   * Validate inheritance structure
   */
  private async validateInheritanceStructure(
    templateName: string,
    options: InheritanceResolutionOptions
  ): Promise<ValidationError[]> {
    const errors: ValidationError[] = [];

    // Check for circular dependencies
    const circularDeps = this.detectCircularDependencies(templateName);
    for (const circDep of circularDeps) {
      if (circDep.severity === 'error') {
        errors.push({
          code: 'CIRCULAR_DEPENDENCY',
          message: circDep.message,
          template: templateName,
          severity: 'error'
        });
      }
    }

    // Check inheritance depth
    const depth = this.calculateInheritanceDepth(templateName);
    const maxDepth = options.maxInheritanceDepth || 10;
    if (depth > maxDepth) {
      errors.push({
        code: 'MAX_DEPTH_EXCEEDED',
        message: `Inheritance depth ${depth} exceeds maximum allowed depth of ${maxDepth}`,
        template: templateName,
        severity: 'warning'
      });
    }

    return errors;
  }

  /**
   * Detect circular dependencies
   */
  private detectCircularDependencies(templateName: string): CircularDependency[] {
    const circularDeps: CircularDependency[] = [];
    const visited = new Set<string>();
    const recursionStack = new Set<string>();
    const path: string[] = [];

    const detectCycle = (nodeName: string): void => {
      visited.add(nodeName);
      recursionStack.add(nodeName);
      path.push(nodeName);

      const node = this.inheritanceGraph.get(nodeName);
      if (node) {
        for (const parentName of node.parents) {
          if (!visited.has(parentName)) {
            detectCycle(parentName);
          } else if (recursionStack.has(parentName)) {
            // Circular dependency found
            const cycleStart = path.indexOf(parentName);
            const cycle = path.slice(cycleStart);
            circularDeps.push({
              chain: [...cycle, parentName],
              cycleStart,
              severity: 'error',
              resolution: 'break_cycle',
              message: `Circular dependency: ${cycle.join(' -> ')} -> ${parentName}`
            });
          }
        }
      }

      recursionStack.delete(nodeName);
      path.pop();
    };

    detectCycle(templateName);
    return circularDeps;
  }

  /**
   * Calculate inheritance depth
   */
  private calculateInheritanceDepth(templateName: string): number {
    const node = this.inheritanceGraph.get(templateName);
    return node ? node.depth : 0;
  }

  /**
   * Get all dependencies for a template
   */
  private getAllDependencies(templateName: string): string[] {
    const dependencies = new Set<string>();
    const visited = new Set<string>();

    const collectDependencies = (nodeName: string): void => {
      if (visited.has(nodeName)) return;
      visited.add(nodeName);

      const node = this.inheritanceGraph.get(nodeName);
      if (node) {
        for (const parentName of node.parents) {
          dependencies.add(parentName);
          collectDependencies(parentName);
        }
      }
    };

    collectDependencies(templateName);
    return Array.from(dependencies);
  }

  /**
   * Analyze parameter conflicts
   */
  private async analyzeParameterConflicts(templateName: string): Promise<ParameterConflict[]> {
    // This would analyze potential conflicts before resolution
    // Implementation would simulate the merge process
    return [];
  }

  /**
   * Calculate merge complexity
   */
  private calculateMergeComplexity(templateName: string): number {
    const node = this.inheritanceGraph.get(templateName);
    if (!node) return 0;

    let complexity = 0;
    const visited = new Set<string>();

    const calculateComplexity = (nodeName: string): number => {
      if (visited.has(nodeName)) return 0;
      visited.add(nodeName);

      const currentNode = this.inheritanceGraph.get(nodeName)!;
      let nodeComplexity = Object.keys(currentNode.template.configuration || {}).length;

      for (const parentName of currentNode.parents) {
        nodeComplexity += calculateComplexity(parentName);
      }

      return nodeComplexity;
    };

    complexity = calculateComplexity(templateName);
    return complexity;
  }

  /**
   * Estimate processing time
   */
  private estimateProcessingTime(templateName: string, complexity: number): number {
    // Simple estimation based on complexity
    return Math.max(10, complexity * 0.1); // Minimum 10ms
  }

  /**
   * Generate optimization recommendations
   */
  private generateRecommendations(
    templateName: string,
    depth: number,
    dependencies: string[],
    circularDeps: CircularDependency[],
    conflicts: ParameterConflict[]
  ): string[] {
    const recommendations: string[] = [];

    if (depth > 5) {
      recommendations.push('Consider reducing inheritance depth for better performance');
    }

    if (dependencies.length > 10) {
      recommendations.push('High number of dependencies may impact maintainability');
    }

    if (circularDeps.length > 0) {
      recommendations.push('Resolve circular dependencies to ensure proper template resolution');
    }

    if (conflicts.length > 20) {
      recommendations.push('Many parameter conflicts detected - consider template restructuring');
    }

    return recommendations;
  }

  /**
   * Find template by priority info
   */
  private findTemplateByPriority(priorityInfo: TemplatePriorityInfo): string {
    // This would need to be implemented based on registry structure
    // For now, return a placeholder
    return priorityInfo.source || priorityInfo.category;
  }

  /**
   * Find conflict rule for parameter
   */
  private findConflictRule(parameter: string): MergeConflictRule | undefined {
    for (const [pattern, rules] of this.conflictRules) {
      const regex = new RegExp(pattern);
      if (regex.test(parameter)) {
        return rules[0]; // Return first matching rule
      }
    }
    return undefined;
  }

  /**
   * Apply conflict rule
   */
  private async applyConflictRule(
    rule: MergeConflictRule,
    values: any[],
    contexts: any[]
  ): Promise<any> {
    if (rule.customResolver) {
      return rule.customResolver(values, contexts);
    }

    switch (rule.strategy) {
      case InheritanceStrategy.MERGE:
        return this.mergeValues(values[0], values[1]);
      case InheritanceStrategy.APPEND:
        return this.appendValues(values[0], values[1]);
      case InheritanceStrategy.INTERSECT:
        return this.intersectValues(values[0], values[1]);
      default:
        return values[values.length - 1]; // Last value (highest priority)
    }
  }

  /**
   * Initialize default conflict rules
   */
  private initializeDefaultConflictRules(): void {
    // Array parameters should be merged
    this.conflictRules.set('.*List$|.*Array$|.*Items$', [{
      parameterPattern: /.*List$|.*Array$|.*Items$/,
      strategy: InheritanceStrategy.MERGE,
      priority: 'highest'
    }]);

    // Boolean parameters should use AND logic
    this.conflictRules.set('.*Enabled$|.*Active$|.*Flag$', [{
      parameterPattern: /.*Enabled$|.*Active$|.*Flag$/,
      strategy: InheritanceStrategy.CUSTOM,
      customResolver: (values) => values.some(v => v === true), // Any true = true
      priority: 'highest'
    }]);

    // Numeric parameters should use highest
    this.conflictRules.set('.*Threshold$|.*Limit$|.*Max$|.*Min$', [{
      parameterPattern: /.*Threshold$|.*Limit$|.*Max$|.*Min$/,
      strategy: InheritanceStrategy.CUSTOM,
      customResolver: (values) => Math.max(...values.filter(v => typeof v === 'number')),
      priority: 'highest'
    }]);
  }

  /**
   * Generate resolution cache key
   */
  private generateResolutionCacheKey(
    templateName: string,
    options: InheritanceResolutionOptions
  ): string {
    const optionsHash = JSON.stringify(options);
    return `${templateName}:${Buffer.from(optionsHash).toString('base64')}`;
  }

  /**
   * Clear all caches
   */
  clearCaches(): void {
    this.resolutionCache.clear();
    this.analysisCache.clear();
  }

  /**
   * Get inheritance graph statistics
   */
  getGraphStats(): {
    totalNodes: number;
    totalEdges: number;
    maxDepth: number;
    circularDependencies: number;
    averageBranchingFactor: number;
  } {
    let totalEdges = 0;
    let maxDepth = 0;
    let totalBranching = 0;

    for (const node of this.inheritanceGraph.values()) {
      totalEdges += node.parents.size;
      maxDepth = Math.max(maxDepth, node.depth);
      totalBranching += node.children.size;
    }

    return {
      totalNodes: this.inheritanceGraph.size,
      totalEdges,
      maxDepth,
      circularDependencies: 0, // Would need to calculate this
      averageBranchingFactor: this.inheritanceGraph.size > 0 ? totalBranching / this.inheritanceGraph.size : 0
    };
  }
}