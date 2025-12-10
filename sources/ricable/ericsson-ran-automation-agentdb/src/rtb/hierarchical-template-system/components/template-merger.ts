/**
 * Template Merger - Priority-Based Conflict Resolution Implementation
 *
 * Handles merging of multiple templates with intelligent conflict resolution.
 * Supports multiple resolution strategies and maintains template integrity
 * throughout the merging process.
 *
 * Features:
 * - Priority-based parameter resolution
 * - Custom function merging with conflict detection
 * - Conditional logic preservation and combination
 * - Deep configuration merging with conflict tracking
 * - Performance optimization for large template sets
 */

import {
  PriorityTemplate,
  TemplatePriority,
  TemplateConflict,
  ConflictResolutionStrategy,
  TemplateWarning,
  ConditionOperator,
  CustomFunction,
  EvaluationOperator,
  TemplateChainLink,
  TemplateProcessingMetrics
} from '../interfaces';

import { HierarchicalTemplateEngineConfig } from '../interfaces';

/**
 * Template Merger implementation
 */
export class TemplateMerger {
  private config: HierarchicalTemplateEngineConfig;

  constructor(config: HierarchicalTemplateEngineConfig) {
    this.config = config;
  }

  /**
   * Merge multiple templates with conflict resolution
   */
  async merge(
    templates: PriorityTemplate[],
    strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.HIGHEST_PRIORITY_WINS
  ): Promise<PriorityTemplate> {
    const startTime = Date.now();

    if (templates.length === 0) {
      throw new Error('No templates provided for merging');
    }

    if (templates.length === 1) {
      return templates[0];
    }

    // Sort templates by priority (lowest number = highest priority)
    const sortedTemplates = [...templates].sort((a, b) => a.priority - b.priority);

    // Detect conflicts
    const conflicts = await this.detectConflicts(sortedTemplates);

    // Resolve conflicts based on strategy
    const resolvedConflicts = await this.resolveConflicts(conflicts, strategy);

    // Merge template metadata
    const mergedMeta = this.mergeMetadata(sortedTemplates, resolvedConflicts);

    // Merge configurations
    const mergedConfiguration = this.mergeConfigurations(sortedTemplates, resolvedConflicts);

    // Merge custom functions
    const mergedCustomFunctions = await this.mergeCustomFunctions(
      sortedTemplates.map(t => t.custom || [])
    );

    // Merge conditions
    const mergedConditions = await this.mergeConditions(
      sortedTemplates.map(t => t.conditions || {})
    );

    // Merge evaluations
    const mergedEvaluations = this.mergeEvaluations(
      sortedTemplates.map(t => t.evaluations || {})
    );

    // Create merged template
    const mergedTemplate: PriorityTemplate = {
      meta: mergedMeta,
      custom: mergedCustomFunctions,
      configuration: mergedConfiguration,
      conditions: mergedConditions,
      evaluations: mergedEvaluations,
      priority: Math.min(...sortedTemplates.map(t => t.priority)),
      inheritanceChain: this.buildMergedInheritanceChain(sortedTemplates),
      conflictResolution: strategy,
      validationRules: this.mergeValidationRules(sortedTemplates)
    };

    // Record processing time
    const processingTime = Date.now() - startTime;
    if (this.config.performanceMonitoring) {
      this.recordMergingMetrics(sortedTemplates, processingTime, resolvedConflicts.length);
    }

    console.log(`[TemplateMerger] Merged ${templates.length} templates in ${processingTime}ms with ${resolvedConflicts.length} conflicts resolved`);

    return mergedTemplate;
  }

  /**
   * Resolve parameter conflicts using specified strategy
   */
  async resolveConflicts(
    conflicts: TemplateConflict[],
    strategy: ConflictResolutionStrategy
  ): Promise<TemplateConflict[]> {
    const resolvedConflicts: TemplateConflict[] = [];

    for (const conflict of conflicts) {
      try {
        const resolved = await this.resolveConflict(conflict, strategy);
        resolvedConflicts.push(resolved);
      } catch (error) {
        console.error(`[TemplateMerger] Error resolving conflict for ${conflict.parameterPath}:`, error);
        // Add conflict as unresolved with error
        resolvedConflicts.push({
          ...conflict,
          resolutionReason: `Error during resolution: ${(error as Error).message}`
        });
      }
    }

    return resolvedConflicts;
  }

  /**
   * Merge custom functions from multiple templates
   */
  async mergeCustomFunctions(functions: CustomFunction[][]): Promise<CustomFunction[]> {
    const functionMap = new Map<string, CustomFunction>();
    const functionConflicts: Array<{ name: string; functions: CustomFunction[] }> = [];

    // Group functions by name
    for (const templateFunctions of functions) {
      for (const func of templateFunctions) {
        if (functionMap.has(func.name)) {
          // Track conflicts
          const existingConflict = functionConflicts.find(c => c.name === func.name);
          if (existingConflict) {
            existingConflict.functions.push(func);
          } else {
            functionConflicts.push({
              name: func.name,
              functions: [functionMap.get(func.name)!, func]
            });
          }
        } else {
          functionMap.set(func.name, func);
        }
      }
    }

    // Resolve function conflicts
    for (const conflict of functionConflicts) {
      const resolvedFunction = await this.resolveFunctionConflict(conflict.functions);
      functionMap.set(conflict.name, resolvedFunction);
    }

    return Array.from(functionMap.values());
  }

  /**
   * Merge conditional logic from multiple templates
   */
  async mergeConditions(conditions: Record<string, ConditionOperator>[]): Promise<Record<string, ConditionOperator>> {
    const mergedConditions: Record<string, ConditionOperator> = {};
    const conditionConflicts: Array<{ key: string; conditions: ConditionOperator[] }> = [];

    // Group conditions by key
    for (const templateConditions of conditions) {
      for (const [key, condition] of Object.entries(templateConditions)) {
        if (mergedConditions[key]) {
          // Track conflicts
          const existingConflict = conditionConflicts.find(c => c.key === key);
          if (existingConflict) {
            existingConflict.conditions.push(condition);
          } else {
            conditionConflicts.push({
              key,
              conditions: [mergedConditions[key], condition]
            });
          }
        } else {
          mergedConditions[key] = condition;
        }
      }
    }

    // Resolve condition conflicts
    for (const conflict of conditionConflicts) {
      const resolvedCondition = await this.resolveConditionConflict(conflict.conditions);
      mergedConditions[conflict.key] = resolvedCondition;
    }

    return mergedConditions;
  }

  // ============================================================================
  // PRIVATE METHODS
  // ============================================================================

  /**
   * Detect conflicts between templates
   */
  private async detectConflicts(templates: PriorityTemplate[]): Promise<TemplateConflict[]> {
    const conflicts: TemplateConflict[] = [];
    const parameterMap = new Map<string, Array<{ template: PriorityTemplate; value: any }>>();

    // Build parameter map
    for (const template of templates) {
      for (const [parameterPath, value] of Object.entries(template.configuration)) {
        if (!parameterMap.has(parameterPath)) {
          parameterMap.set(parameterPath, []);
        }
        parameterMap.get(parameterPath)!.push({ template, value });
      }
    }

    // Find conflicts (parameters with different values across templates)
    for (const [parameterPath, values] of parameterMap) {
      if (values.length > 1) {
        const uniqueValues = new Set(values.map(v => JSON.stringify(v.value)));
        if (uniqueValues.size > 1) {
          conflicts.push({
            parameterPath,
            conflictingTemplates: values.map(v => ({
              templateId: v.template.meta.version,
              priority: v.template.priority,
              value: v.value
            })),
            resolutionStrategy: ConflictResolutionStrategy.HIGHEST_PRIORITY_WINS
          });
        }
      }
    }

    return conflicts;
  }

  /**
   * Resolve a single conflict using the specified strategy
   */
  private async resolveConflict(
    conflict: TemplateConflict,
    strategy: ConflictResolutionStrategy
  ): Promise<TemplateConflict> {
    const resolvedConflict = { ...conflict };

    switch (strategy) {
      case ConflictResolutionStrategy.HIGHEST_PRIORITY_WINS:
        // Sort by priority (lowest number = highest priority)
        const highestPriority = conflict.conflictingTemplates.reduce((min, current) =>
          current.priority < min.priority ? current : min
        );
        resolvedConflict.resolvedValue = highestPriority.value;
        resolvedConflict.resolutionReason = `Selected value from template '${highestPriority.templateId}' with highest priority ${highestPriority.priority}`;
        break;

      case ConflictResolutionStrategy.LOWEST_PRIORITY_WINS:
        // Sort by priority (highest number = lowest priority)
        const lowestPriority = conflict.conflictingTemplates.reduce((max, current) =>
          current.priority > max.priority ? current : max
        );
        resolvedConflict.resolvedValue = lowestPriority.value;
        resolvedConflict.resolutionReason = `Selected value from template '${lowestPriority.templateId}' with lowest priority ${lowestPriority.priority}`;
        break;

      case ConflictResolutionStrategy.MERGE_WITH_WARNING:
        // Attempt to merge values if they're objects, otherwise use highest priority
        const values = conflict.conflictingTemplates.map(t => t.value);
        if (values.every(v => typeof v === 'object' && v !== null)) {
          resolvedConflict.resolvedValue = this.deepMergeObjects(...values as Record<string, any>[]);
          resolvedConflict.resolutionReason = 'Merged object values from conflicting templates';
        } else {
          const highestPriority = conflict.conflictingTemplates.reduce((min, current) =>
            current.priority < min.priority ? current : min
          );
          resolvedConflict.resolvedValue = highestPriority.value;
          resolvedConflict.resolutionReason = `Cannot merge non-object values, used highest priority template '${highestPriority.templateId}'`;
        }
        break;

      case ConflictResolutionStrategy.CUSTOM_FUNCTION:
        // For now, fall back to highest priority
        // In a full implementation, this would invoke custom resolution functions
        const highestPriorityCustom = conflict.conflictingTemplates.reduce((min, current) =>
          current.priority < min.priority ? current : min
        );
        resolvedConflict.resolvedValue = highestPriorityCustom.value;
        resolvedConflict.resolutionReason = `Custom resolution not implemented, used highest priority template '${highestPriorityCustom.templateId}'`;
        break;

      case ConflictResolutionStrategy.FAIL_ON_CONFLICT:
        throw new Error(`Conflict detected for parameter '${conflict.parameterPath}' and strategy is FAIL_ON_CONFLICT`);

      case ConflictResolutionStrategy.CONFLICT_LOGGING:
        // Log conflict and use highest priority
        const highestPriorityLog = conflict.conflictingTemplates.reduce((min, current) =>
          current.priority < min.priority ? current : min
        );
        resolvedConflict.resolvedValue = highestPriorityLog.value;
        resolvedConflict.resolutionReason = `Conflict logged, used highest priority template '${highestPriorityLog.templateId}'`;
        console.warn(`[TemplateMerger] Conflict logged: ${conflict.parameterPath}`, conflict);
        break;

      default:
        throw new Error(`Unknown conflict resolution strategy: ${strategy}`);
    }

    return resolvedConflict;
  }

  /**
   * Merge template metadata
   */
  private mergeMetadata(
    templates: PriorityTemplate[],
    resolvedConflicts: TemplateConflict[]
  ): any {
    const highestPriorityTemplate = templates[0]; // Already sorted by priority
    const allTags = new Set<string>();
    const allAuthors = new Set<string>();

    // Collect all tags and authors
    for (const template of templates) {
      if (template.meta.tags) {
        template.meta.tags.forEach(tag => allTags.add(tag));
      }
      if (template.meta.author) {
        template.meta.author.forEach(author => allAuthors.add(author));
      }
    }

    return {
      version: `merged_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      author: Array.from(allAuthors),
      description: `Merged template: ${templates.map(t => t.meta.description).filter(Boolean).join('; ')}`,
      tags: Array.from(allTags),
      priority: Math.min(...templates.map(t => t.priority)),
      inherits_from: templates.map(t => t.meta.version),
      conflictResolution: resolvedConflicts.length > 0 ? 'resolved' : 'none',
      mergedFrom: templates.map(t => ({
        version: t.meta.version,
        priority: t.priority,
        description: t.meta.description
      })),
      conflictsResolved: resolvedConflicts.length,
      validationRules: []
    };
  }

  /**
   * Merge configurations with conflict resolution
   */
  private mergeConfigurations(
    templates: PriorityTemplate[],
    resolvedConflicts: TemplateConflict[]
  ): Record<string, any> {
    const mergedConfig: Record<string, any> = {};
    const resolvedPaths = new Set(resolvedConflicts.map(c => c.parameterPath));

    // Create a map of resolved values
    const resolvedValues = new Map<string, any>();
    for (const conflict of resolvedConflicts) {
      if (conflict.resolvedValue !== undefined) {
        resolvedValues.set(conflict.parameterPath, conflict.resolvedValue);
      }
    }

    // Process templates in priority order (highest to lowest)
    for (const template of templates) {
      for (const [parameterPath, value] of Object.entries(template.configuration)) {
        // Skip if this parameter was resolved by conflict resolution
        if (resolvedPaths.has(parameterPath)) {
          mergedConfig[parameterPath] = resolvedValues.get(parameterPath);
        } else if (!mergedConfig.hasOwnProperty(parameterPath)) {
          // First template (highest priority) sets the value
          mergedConfig[parameterPath] = value;
        }
        // Subsequent templates only set values for parameters that haven't been set yet
      }
    }

    return mergedConfig;
  }

  /**
   * Merge evaluation operators
   */
  private mergeEvaluations(evaluations: Record<string, EvaluationOperator>[]): Record<string, EvaluationOperator> {
    const mergedEvaluations: Record<string, EvaluationOperator> = {};

    for (const templateEvaluations of evaluations) {
      for (const [key, evaluation] of Object.entries(templateEvaluations)) {
        if (!mergedEvaluations[key]) {
          mergedEvaluations[key] = evaluation;
        }
      }
    }

    return mergedEvaluations;
  }

  /**
   * Merge validation rules
   */
  private mergeValidationRules(templates: PriorityTemplate[]): any[] {
    const allRules: any[] = [];
    const ruleNames = new Set<string>();

    for (const template of templates) {
      if (template.meta.validationRules) {
        for (const rule of template.meta.validationRules) {
          if (!ruleNames.has(rule.ruleId)) {
            allRules.push(rule);
            ruleNames.add(rule.ruleId);
          }
        }
      }
    }

    return allRules;
  }

  /**
   * Build merged inheritance chain
   */
  private buildMergedInheritanceChain(templates: PriorityTemplate[]): TemplateChainLink[] {
    const chain: TemplateChainLink[] = [];

    for (const template of templates) {
      chain.push({
        templateId: template.meta.version,
        priority: template.priority,
        appliedAt: new Date(),
        appliedParameters: Object.keys(template.configuration),
        overriddenParameters: [], // Would be calculated during resolution
        conflicts: []
      });
    }

    return chain;
  }

  /**
   * Resolve function conflicts
   */
  private async resolveFunctionConflict(functions: CustomFunction[]): Promise<CustomFunction> {
    // For now, use the function from the highest priority template
    // In a more sophisticated implementation, we could attempt to merge function bodies
    return functions[0];
  }

  /**
   * Resolve condition conflicts
   */
  private async resolveConditionConflict(conditions: ConditionOperator[]): Promise<ConditionOperator> {
    // For now, use the condition from the highest priority template
    // In a more sophisticated implementation, we could combine conditions
    return conditions[0];
  }

  /**
   * Deep merge objects
   */
  private deepMergeObjects(...objects: Record<string, any>[]): Record<string, any> {
    const result: Record<string, any> = {};

    for (const obj of objects) {
      for (const [key, value] of Object.entries(obj)) {
        if (value && typeof value === 'object' && !Array.isArray(value)) {
          if (result[key] && typeof result[key] === 'object' && !Array.isArray(result[key])) {
            result[key] = this.deepMergeObjects(result[key], value);
          } else {
            result[key] = value;
          }
        } else {
          result[key] = value;
        }
      }
    }

    return result;
  }

  /**
   * Record merging metrics
   */
  private recordMergingMetrics(templates: PriorityTemplate[], processingTime: number, conflictCount: number): void {
    if (!this.config.performanceMonitoring) return;

    const metrics: TemplateProcessingMetrics = {
      templateId: `merge_${Date.now()}`,
      processingTime,
      memoryUsage: process.memoryUsage().heapUsed,
      parameterCount: templates.reduce((sum, t) => sum + Object.keys(t.configuration).length, 0),
      conflictCount,
      warningCount: 0,
      cacheHits: 0,
      cacheMisses: 0
    };

    // Store metrics (in a real implementation, this would use a proper metrics store)
    console.log(`[TemplateMerger] Metrics:`, metrics);
  }
}