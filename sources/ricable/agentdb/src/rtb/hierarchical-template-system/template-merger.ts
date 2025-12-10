/**
 * Template Merger Engine - Hierarchical Template Inheritance and Merging System
 *
 * Core engine for merging multiple RTB templates with intelligent inheritance resolution
 * and conflict handling. Supports complex inheritance chains with priority-based resolution.
 */

import { RTBTemplate, TemplateMeta, CustomFunction, ConditionOperator, EvaluationOperator } from '../../types/rtb-types';
import { TemplateConflict, ConflictType, ResolutionStrategyType, MergeResult, MergeContext } from './types';
import { ConflictDetector } from './conflict-detector';
import { ResolutionEngine } from './resolution-engine';
import { MergeValidator } from './merge-validator';
import { Logger } from '../../utils/logger';

export interface MergeOptions {
  /** Strategy for handling conflicts */
  conflictResolution: 'auto' | 'interactive' | 'strict';
  /** Preserve template metadata during merge */
  preserveMetadata: boolean;
  /** Validate merged template */
  validateResult: boolean;
  /** Enable deep merging for nested objects */
  deepMerge: boolean;
  /** Custom resolvers for specific conflicts */
  customResolvers: Record<string, (conflict: TemplateConflict) => any>;
  /** Performance optimization for large template sets */
  batchMode: boolean;
  /** Enable caching for repeated merges */
  enableCache: boolean;
}

export interface InheritanceChain {
  templates: RTBTemplate[];
  priorities: number[];
  inheritanceDepth: number;
  hasCircularDependency: boolean;
  circularPath?: string[];
}

export class TemplateMerger {
  private conflictDetector: ConflictDetector;
  private resolutionEngine: ResolutionEngine;
  private mergeValidator: MergeValidator;
  private mergeCache: Map<string, MergeResult>;
  private logger: Logger;

  constructor() {
    this.conflictDetector = new ConflictDetector();
    this.resolutionEngine = new ResolutionEngine();
    this.mergeValidator = new MergeValidator();
    this.mergeCache = new Map();
    this.logger = new Logger('TemplateMerger');
  }

  /**
   * Merge multiple templates with inheritance resolution
   */
  async mergeTemplates(
    templates: RTBTemplate[],
    options: Partial<MergeOptions> = {}
  ): Promise<MergeResult> {
    const mergeOptions: MergeOptions = {
      conflictResolution: 'auto',
      preserveMetadata: true,
      validateResult: true,
      deepMerge: true,
      customResolvers: {},
      batchMode: false,
      enableCache: true,
      ...options
    };

    const context: MergeContext = {
      templates,
      options: mergeOptions,
      startTime: Date.now(),
      mergeStats: {
        totalTemplates: templates.length,
        conflictsDetected: 0,
        conflictsResolved: 0,
        resolutionsApplied: [],
        processingTime: 0
      }
    };

    this.logger.info(`Starting merge of ${templates.length} templates`, {
      templateNames: templates.map(t => t.meta?.description || 'Unnamed'),
      options: mergeOptions
    });

    try {
      // Check cache first
      const cacheKey = this.generateCacheKey(templates, mergeOptions);
      if (mergeOptions.enableCache && this.mergeCache.has(cacheKey)) {
        this.logger.debug('Using cached merge result');
        return this.mergeCache.get(cacheKey)!;
      }

      // Build inheritance chain
      const inheritanceChain = await this.buildInheritanceChain(templates);
      if (inheritanceChain.hasCircularDependency) {
        throw new Error(`Circular dependency detected: ${inheritanceChain.circularPath?.join(' -> ')}`);
      }

      // Detect conflicts
      const conflicts = await this.conflictDetector.detectConflicts(inheritanceChain.templates);
      context.mergeStats.conflictsDetected = conflicts.length;

      this.logger.info(`Detected ${conflicts.length} conflicts during merge`);

      // Resolve conflicts
      const resolvedTemplates = await this.resolveConflicts(
        inheritanceChain.templates,
        conflicts,
        context
      );

      // Perform the actual merge
      const mergedTemplate = await this.performMerge(resolvedTemplates, context);

      // Validate result if requested
      if (mergeOptions.validateResult) {
        const validationResult = await this.mergeValidator.validateMergedTemplate(
          mergedTemplate,
          context
        );

        if (!validationResult.isValid) {
          throw new Error(`Merged template validation failed: ${validationResult.errors.join(', ')}`);
        }
      }

      // Create result
      const result: MergeResult = {
        template: mergedTemplate,
        conflicts: conflicts,
        resolvedConflicts: conflicts.filter(c => c.resolved),
        unresolvedConflicts: conflicts.filter(c => !c.resolved),
        mergeStats: {
          ...context.mergeStats,
          processingTime: Date.now() - context.startTime
        },
        inheritanceChain: inheritanceChain,
        validationResult: mergeOptions.validateResult ?
          await this.mergeValidator.validateMergedTemplate(mergedTemplate, context) :
          undefined
      };

      // Cache result
      if (mergeOptions.enableCache) {
        this.mergeCache.set(cacheKey, result);
      }

      this.logger.info(`Template merge completed successfully`, {
        processingTime: result.mergeStats.processingTime,
        conflictsResolved: result.resolvedConflicts.length,
        validationResult: result.validationResult?.isValid
      });

      return result;

    } catch (error) {
      this.logger.error('Template merge failed', { error, context });
      throw error;
    }
  }

  /**
   * Build inheritance chain from templates
   */
  private async buildInheritanceChain(templates: RTBTemplate[]): Promise<InheritanceChain> {
    const templateMap = new Map(templates.map(t => [t.meta?.description || t.meta?.source || 'unknown', t]));
    const visited = new Set<string>();
    const visiting = new Set<string>();
    const chain: RTBTemplate[] = [];
    const priorities: number[] = [];
    const inheritanceGraph = new Map<string, string[]>();

    // Build inheritance graph
    for (const template of templates) {
      const templateId = template.meta?.description || template.meta?.source || 'unknown';
      const inheritsFrom = template.meta?.inherits_from;

      if (inheritsFrom) {
        const parents = Array.isArray(inheritsFrom) ? inheritsFrom : [inheritsFrom];
        inheritanceGraph.set(templateId, parents);
      }
    }

    // Detect circular dependencies and build chain
    const visit = async (templateId: string, depth: number = 0): Promise<void> => {
      if (visiting.has(templateId)) {
        throw new Error(`Circular dependency detected involving ${templateId}`);
      }

      if (visited.has(templateId)) {
        return;
      }

      visiting.add(templateId);

      const template = templateMap.get(templateId);
      if (!template) {
        this.logger.warn(`Template not found: ${templateId}`);
        return;
      }

      // Visit parents first
      const parents = inheritanceGraph.get(templateId) || [];
      for (const parentId of parents) {
        await visit(parentId, depth + 1);
      }

      visiting.delete(templateId);
      visited.add(templateId);

      // Add to chain (parents come first, then children)
      if (!chain.includes(template)) {
        chain.push(template);
        priorities.push(template.meta?.priority || 0);
      }
    };

    // Visit all templates
    for (const templateId of templateMap.keys()) {
      await visit(templateId);
    }

    return {
      templates: chain,
      priorities,
      inheritanceDepth: Math.max(...chain.map(t => this.getInheritanceDepth(t, inheritanceGraph))),
      hasCircularDependency: false
    };
  }

  /**
   * Get inheritance depth for a template
   */
  private getInheritanceDepth(template: RTBTemplate, graph: Map<string, string[]>): number {
    const templateId = template.meta?.description || template.meta?.source || 'unknown';
    const parents = graph.get(templateId) || [];

    if (parents.length === 0) {
      return 0;
    }

    return 1 + Math.max(...parents.map(parentId => {
      const parentTemplate = Array.from(graph.keys()).find(id => id === parentId);
      return parentTemplate ? this.getInheritanceDepth({ meta: { description: parentId } } as RTBTemplate, graph) : 0;
    }));
  }

  /**
   * Resolve conflicts using resolution engine
   */
  private async resolveConflicts(
    templates: RTBTemplate[],
    conflicts: TemplateConflict[],
    context: MergeContext
  ): Promise<RTBTemplate[]> {
    const resolvedTemplates = [...templates];

    for (const conflict of conflicts) {
      try {
        const resolution = await this.resolutionEngine.resolveConflict(conflict, context);

        if (resolution.resolved) {
          // Apply resolution to the affected template
          const templateIndex = resolvedTemplates.findIndex(t =>
            t.meta?.description === conflict.templates[conflict.resolvedTemplateIndex!]
          );

          if (templateIndex !== -1) {
            resolvedTemplates[templateIndex] = this.applyResolution(
              resolvedTemplates[templateIndex],
              conflict,
              resolution.value
            );
          }

          context.mergeStats.conflictsResolved++;
          context.mergeStats.resolutionsApplied.push({
            conflict: conflict.parameter,
            strategy: resolution.strategy,
            template: conflict.templates[conflict.resolvedTemplateIndex!]
          });
        }
      } catch (error) {
        this.logger.error(`Failed to resolve conflict for ${conflict.parameter}`, { error, conflict });
        conflict.resolved = false;
      }
    }

    return resolvedTemplates;
  }

  /**
   * Apply resolution to a template
   */
  private applyResolution(
    template: RTBTemplate,
    conflict: TemplateConflict,
    resolutionValue: any
  ): RTBTemplate {
    const updatedTemplate = { ...template };

    // Navigate to the parameter path and apply resolution
    const pathParts = conflict.parameter.split('.');
    let current = updatedTemplate.configuration;

    for (let i = 0; i < pathParts.length - 1; i++) {
      if (!current[pathParts[i]]) {
        current[pathParts[i]] = {};
      }
      current = current[pathParts[i]];
    }

    current[pathParts[pathParts.length - 1]] = resolutionValue;

    return updatedTemplate;
  }

  /**
   * Perform the actual merge operation
   */
  private async performMerge(templates: RTBTemplate[], context: MergeContext): Promise<RTBTemplate> {
    if (templates.length === 0) {
      throw new Error('No templates to merge');
    }

    if (templates.length === 1) {
      return templates[0];
    }

    // Start with the base template (lowest priority)
    let mergedTemplate = { ...templates[0] };

    // Merge templates in order of priority (low to high)
    for (let i = 1; i < templates.length; i++) {
      mergedTemplate = await this.mergeTwoTemplates(mergedTemplate, templates[i], context);
    }

    // Merge custom functions
    mergedTemplate.custom = this.mergeCustomFunctions(
      templates.map(t => t.custom || [])
    );

    // Merge conditions
    mergedTemplate.conditions = this.mergeConditions(
      templates.map(t => t.conditions || {})
    );

    // Merge evaluations
    mergedTemplate.evaluations = this.mergeEvaluations(
      templates.map(t => t.evaluations || {})
    );

    // Merge metadata
    if (context.options.preserveMetadata) {
      mergedTemplate.meta = this.mergeMetadata(
        templates.map(t => t.meta).filter(Boolean) as TemplateMeta[]
      );
    }

    return mergedTemplate;
  }

  /**
   * Merge two templates
   */
  private async mergeTwoTemplates(
    base: RTBTemplate,
    override: RTBTemplate,
    context: MergeContext
  ): Promise<RTBTemplate> {
    const merged: RTBTemplate = {
      ...base,
      configuration: context.options.deepMerge ?
        this.deepMerge(base.configuration || {}, override.configuration || {}) :
        { ...base.configuration, ...override.configuration }
    };

    return merged;
  }

  /**
   * Deep merge two objects
   */
  private deepMerge(target: any, source: any): any {
    const result = { ...target };

    for (const key in source) {
      if (source.hasOwnProperty(key)) {
        if (this.isObject(source[key]) && this.isObject(result[key])) {
          result[key] = this.deepMerge(result[key], source[key]);
        } else {
          result[key] = source[key];
        }
      }
    }

    return result;
  }

  /**
   * Check if value is an object
   */
  private isObject(value: any): boolean {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
  }

  /**
   * Merge custom functions from multiple templates
   */
  private mergeCustomFunctions(functionsArray: CustomFunction[][]): CustomFunction[] {
    const functionMap = new Map<string, CustomFunction>();

    for (const functions of functionsArray) {
      for (const func of functions) {
        // Higher priority templates override lower priority ones
        functionMap.set(func.name, func);
      }
    }

    return Array.from(functionMap.values());
  }

  /**
   * Merge conditions from multiple templates
   */
  private mergeConditions(conditionsArray: Record<string, ConditionOperator>[]): Record<string, ConditionOperator> {
    const merged: Record<string, ConditionOperator> = {};

    for (const conditions of conditionsArray) {
      for (const [key, condition] of Object.entries(conditions)) {
        // Later templates override earlier ones
        merged[key] = condition;
      }
    }

    return merged;
  }

  /**
   * Merge evaluations from multiple templates
   */
  private mergeEvaluations(evaluationsArray: Record<string, EvaluationOperator>[]): Record<string, EvaluationOperator> {
    const merged: Record<string, EvaluationOperator> = {};

    for (const evaluations of evaluationsArray) {
      for (const [key, evaluation] of Object.entries(evaluations)) {
        // Later templates override earlier ones
        merged[key] = evaluation;
      }
    }

    return merged;
  }

  /**
   * Merge metadata from multiple templates
   */
  private mergeMetadata(metadataArray: TemplateMeta[]): TemplateMeta {
    if (metadataArray.length === 0) {
      return {} as TemplateMeta;
    }

    const merged: TemplateMeta = { ...metadataArray[0] };

    for (let i = 1; i < metadataArray.length; i++) {
      const meta = metadataArray[i];

      // Merge arrays
      if (meta.author) {
        merged.author = [...(merged.author || []), ...meta.author];
      }

      if (meta.tags) {
        merged.tags = [...(merged.tags || []), ...meta.tags];
      }

      if (meta.inherits_from) {
        if (Array.isArray(merged.inherits_from)) {
          merged.inherits_from = [...merged.inherits_from];
        } else if (merged.inherits_from) {
          merged.inherits_from = [merged.inherits_from];
        }

        const inherits = Array.isArray(meta.inherits_from) ? meta.inherits_from : [meta.inherits_from];
        merged.inherits_from = [...(Array.isArray(merged.inherits_from) ? merged.inherits_from : []), ...inherits];
      }

      // Override with higher priority values
      if (meta.version !== undefined) merged.version = meta.version;
      if (meta.description !== undefined) merged.description = meta.description;
      if (meta.environment !== undefined) merged.environment = meta.environment;
      if (meta.priority !== undefined) merged.priority = meta.priority;
      if (meta.source !== undefined) merged.source = meta.source;
    }

    // Remove duplicates
    if (merged.author) {
      merged.author = [...new Set(merged.author)];
    }

    if (merged.tags) {
      merged.tags = [...new Set(merged.tags)];
    }

    return merged;
  }

  /**
   * Generate cache key for template set
   */
  private generateCacheKey(templates: RTBTemplate[], options: MergeOptions): string {
    const templateHashes = templates.map(t =>
      t.meta?.description || t.meta?.source || 'unknown'
    ).sort().join('|');

    const optionsHash = JSON.stringify(options);

    return `${templateHashes}:${optionsHash}`;
  }

  /**
   * Clear merge cache
   */
  public clearCache(): void {
    this.mergeCache.clear();
    this.logger.info('Merge cache cleared');
  }

  /**
   * Get merge cache statistics
   */
  public getCacheStats(): { size: number; keys: string[] } {
    return {
      size: this.mergeCache.size,
      keys: Array.from(this.mergeCache.keys())
    };
  }
}