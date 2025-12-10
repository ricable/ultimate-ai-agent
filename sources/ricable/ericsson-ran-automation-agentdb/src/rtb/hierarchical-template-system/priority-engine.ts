/**
 * Priority Template Engine - Core Template Inheritance System
 *
 * Handles priority-based template inheritance with support for complex
 * inheritance chains, parameter resolution, and conflict management.
 */

import {
  RTBTemplate,
  TemplateMeta,
  CustomFunction,
  ConditionOperator,
  EvaluationOperator,
  RTBParameter
} from '../../types/rtb-types';

/**
 * Template priority levels (0 = highest, 80 = lowest)
 */
export enum TemplatePriority {
  AGENT_OVERRIDE = 0,      // Agent-specific overrides
  CONTEXT_SPECIFIC = 10,   // Context-specific templates
  AGENTDB_LEARNED = 20,    // AgentDB learned patterns
  FEATURE_SPECIFIC = 30,   // Feature-specific templates
  TECHNOLOGY = 40,         // Technology-specific (4G/5G)
  SCENARIO = 50,           // Scenario-based (urban/mobility/sleep)
  VARIANT = 60,            // Variant templates
  BASE = 70,               // Base templates
  DEFAULT = 80             // Default fallback
}

/**
 * Template priority information
 */
export interface TemplatePriorityInfo {
  level: number;
  category: string;
  source: string;
  inherits_from?: string | string[];
  metadata?: TemplateMeta;
  resolvedAt?: Date;
}

/**
 * Template inheritance chain
 */
export interface TemplateInheritanceChain {
  templateName: string;
  chain: TemplatePriorityInfo[];
  resolvedTemplate: RTBTemplate;
  conflicts: ParameterConflict[];
  warnings: string[];
}

/**
 * Parameter conflict information
 */
export interface ParameterConflict {
  parameter: string;
  templates: string[];
  values: any[];
  resolvedValue: any;
  resolutionStrategy: 'highest_priority' | 'merge' | 'custom';
  reason: string;
}

/**
 * Template resolution context
 */
export interface TemplateResolutionContext {
  includeInactive?: boolean;
  validateConstraints?: boolean;
  preserveConditions?: boolean;
  mergeStrategy?: 'override' | 'merge' | 'append';
  priorityOverride?: (template: string) => number;
  environment?: string;
  featureFlags?: Record<string, boolean>;
}

/**
 * Template validation result
 */
export interface TemplateValidationResult {
  isValid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
  metadata: {
    validationTime: number;
    parametersValidated: number;
    constraintsChecked: number;
  };
}

/**
 * Validation error
 */
export interface ValidationError {
  code: string;
  message: string;
  template?: string;
  parameter?: string;
  severity: 'error' | 'warning';
  path?: string;
}

/**
 * Validation warning
 */
export interface ValidationWarning {
  code: string;
  message: string;
  template?: string;
  parameter?: string;
  suggestion?: string;
}

/**
 * Priority Template Engine
 *
 * Core engine for handling priority-based template inheritance,
 * parameter resolution, and conflict management.
 */
export class PriorityTemplateEngine {
  private templates = new Map<string, RTBTemplate>();
  private templatePriorities = new Map<string, TemplatePriorityInfo>();
  private inheritanceCache = new Map<string, TemplateInheritanceChain>();
  private parameterCache = new Map<string, Map<string, any>>();
  private validationCache = new Map<string, TemplateValidationResult>();

  /**
   * Register a template with priority information
   */
  registerTemplate(
    name: string,
    template: RTBTemplate,
    priority: TemplatePriorityInfo
  ): void {
    // Validate template structure
    this.validateTemplateStructure(template);

    // Store template and priority info
    this.templates.set(name, template);
    this.templatePriorities.set(name, {
      ...priority,
      metadata: template.meta,
      resolvedAt: new Date()
    });

    // Clear related caches
    this.clearCacheForTemplate(name);
  }

  /**
   * Resolve template inheritance chain
   */
  resolveInheritanceChain(
    templateName: string,
    context: TemplateResolutionContext = {}
  ): TemplateInheritanceChain {
    const cacheKey = this.generateCacheKey(templateName, context);

    // Check cache first
    if (this.inheritanceCache.has(cacheKey)) {
      return this.inheritanceCache.get(cacheKey)!;
    }

    const template = this.templates.get(templateName);
    if (!template) {
      throw new Error(`Template '${templateName}' not found`);
    }

    const priorityInfo = this.templatePriorities.get(templateName)!;
    const chain: TemplatePriorityInfo[] = [priorityInfo];
    const visited = new Set<string>([templateName]);
    const conflicts: ParameterConflict[] = [];
    const warnings: string[] = [];

    // Build inheritance chain
    const currentTemplate = { ...template };
    const processedTemplates = [templateName];

    // Process inheritance hierarchy
    this.processInheritanceHierarchy(
      templateName,
      currentTemplate,
      chain,
      visited,
      processedTemplates,
      conflicts,
      warnings,
      context
    );

    // Resolve parameter conflicts
    const resolvedTemplate = this.resolveParameterConflicts(
      currentTemplate,
      chain,
      conflicts,
      context
    );

    const result: TemplateInheritanceChain = {
      templateName,
      chain: chain.sort((a, b) => a.level - b.level),
      resolvedTemplate,
      conflicts,
      warnings
    };

    // Cache result
    this.inheritanceCache.set(cacheKey, result);

    return result;
  }

  /**
   * Process inheritance hierarchy recursively
   */
  private processInheritanceHierarchy(
    templateName: string,
    currentTemplate: RTBTemplate,
    chain: TemplatePriorityInfo[],
    visited: Set<string>,
    processedTemplates: string[],
    conflicts: ParameterConflict[],
    warnings: string[],
    context: TemplateResolutionContext
  ): void {
    const priorityInfo = this.templatePriorities.get(templateName);
    if (!priorityInfo?.inherits_from) {
      return;
    }

    const parents = Array.isArray(priorityInfo.inherits_from)
      ? priorityInfo.inherits_from
      : [priorityInfo.inherits_from];

    for (const parentName of parents) {
      // Check for circular dependencies
      if (visited.has(parentName)) {
        warnings.push(`Circular dependency detected: ${templateName} -> ${parentName}`);
        continue;
      }

      const parentTemplate = this.templates.get(parentName);
      if (!parentTemplate) {
        warnings.push(`Parent template '${parentName}' not found for '${templateName}'`);
        continue;
      }

      const parentPriority = this.templatePriorities.get(parentName)!;

      // Validate inheritance priority (parent should have lower priority)
      if (parentPriority.level <= priorityInfo.level) {
        warnings.push(
          `Invalid priority: Parent '${parentName}' (${parentPriority.level}) ` +
          `has higher or equal priority than child '${templateName}' (${priorityInfo.level})`
        );
      }

      visited.add(parentName);
      processedTemplates.push(parentName);
      chain.push(parentPriority);

      // Merge parent template
      this.mergeTemplateData(currentTemplate, parentTemplate, conflicts, context);

      // Recursively process parent hierarchy
      this.processInheritanceHierarchy(
        parentName,
        currentTemplate,
        chain,
        visited,
        processedTemplates,
        conflicts,
        warnings,
        context
      );

      visited.delete(parentName);
    }
  }

  /**
   * Merge template data from parent to child
   */
  private mergeTemplateData(
    child: RTBTemplate,
    parent: RTBTemplate,
    conflicts: ParameterConflict[],
    context: TemplateResolutionContext
  ): void {
    // Merge configuration parameters
    for (const [key, value] of Object.entries(parent.configuration || {})) {
      if (child.configuration[key] !== undefined) {
        // Conflict detected
        conflicts.push({
          parameter: key,
          templates: ['parent', 'child'],
          values: [value, child.configuration[key]],
          resolvedValue: child.configuration[key], // Child wins by default
          resolutionStrategy: 'highest_priority',
          reason: 'Child template overrides parent parameter'
        });
      } else {
        child.configuration[key] = value;
      }
    }

    // Merge conditions
    if (parent.conditions && context.preserveConditions !== false) {
      child.conditions = { ...parent.conditions, ...child.conditions };
    }

    // Merge evaluations
    if (parent.evaluations && context.preserveConditions !== false) {
      child.evaluations = { ...parent.evaluations, ...child.evaluations };
    }

    // Merge custom functions
    if (parent.custom && child.custom) {
      for (const func of parent.custom) {
        if (!child.custom.find(f => f.name === func.name)) {
          child.custom.push(func);
        }
      }
    }
  }

  /**
   * Resolve parameter conflicts using priority rules
   */
  private resolveParameterConflicts(
    template: RTBTemplate,
    chain: TemplatePriorityInfo[],
    conflicts: ParameterConflict[],
    context: TemplateResolutionContext
  ): RTBTemplate {
    const resolved = { ...template };

    for (const conflict of conflicts) {
      switch (context.mergeStrategy || 'override') {
        case 'override':
          // Highest priority wins
          resolved.configuration[conflict.parameter] = conflict.resolvedValue;
          break;

        case 'merge':
          // Try to merge values if possible
          const mergedValue = this.mergeParameterValues(
            conflict.parameter,
            conflict.values,
            conflict.templates,
            chain
          );
          resolved.configuration[conflict.parameter] = mergedValue;
          break;

        case 'append':
          // Append values (for arrays)
          if (Array.isArray(conflict.resolvedValue)) {
            resolved.configuration[conflict.parameter] = [
              ...conflict.values.filter(v => Array.isArray(v)).flat(),
              ...conflict.resolvedValue
            ];
          }
          break;
      }
    }

    return resolved;
  }

  /**
   * Merge parameter values intelligently
   */
  private mergeParameterValues(
    parameter: string,
    values: any[],
    templates: string[],
    chain: TemplatePriorityInfo[]
  ): any {
    // If all values are the same, use that value
    if (values.every(v => JSON.stringify(v) === JSON.stringify(values[0]))) {
      return values[0];
    }

    // For arrays, merge and deduplicate
    if (values.some(v => Array.isArray(v))) {
      const merged = values
        .filter(v => Array.isArray(v))
        .flat()
        .filter((value, index, self) =>
          self.findIndex(v => JSON.stringify(v) === JSON.stringify(value)) === index
        );
      return merged;
    }

    // For objects, merge properties
    if (values.some(v => typeof v === 'object' && v !== null)) {
      const merged = {};
      for (const value of values) {
        if (typeof value === 'object' && value !== null) {
          Object.assign(merged, value);
        }
      }
      return merged;
    }

    // Default: use highest priority value
    return values[values.length - 1];
  }

  /**
   * Validate template structure
   */
  private validateTemplateStructure(template: RTBTemplate): void {
    if (!template || typeof template !== 'object') {
      throw new Error('Template must be a valid object');
    }

    if (!template.configuration || typeof template.configuration !== 'object') {
      throw new Error('Template must have a configuration object');
    }

    // Validate custom functions if present
    if (template.custom) {
      for (const func of template.custom) {
        if (!func.name || !func.args || !func.body) {
          throw new Error(`Invalid custom function: ${func.name}`);
        }
      }
    }
  }

  /**
   * Generate cache key for template resolution
   */
  private generateCacheKey(templateName: string, context: TemplateResolutionContext): string {
    const contextHash = JSON.stringify(context);
    return `${templateName}:${Buffer.from(contextHash).toString('base64')}`;
  }

  /**
   * Clear cache for specific template
   */
  private clearCacheForTemplate(templateName: string): void {
    // Clear inheritance cache
    for (const [key] of this.inheritanceCache) {
      if (key.startsWith(templateName + ':')) {
        this.inheritanceCache.delete(key);
      }
    }

    // Clear parameter cache
    this.parameterCache.clear();
    this.validationCache.clear();
  }

  /**
   * Get all registered templates
   */
  getRegisteredTemplates(): Map<string, TemplatePriorityInfo> {
    return new Map(this.templatePriorities);
  }

  /**
   * Get template by name
   */
  getTemplate(name: string): RTBTemplate | undefined {
    return this.templates.get(name);
  }

  /**
   * Get template priority info
   */
  getTemplatePriority(name: string): TemplatePriorityInfo | undefined {
    return this.templatePriorities.get(name);
  }

  /**
   * Check if template exists
   */
  hasTemplate(name: string): boolean {
    return this.templates.has(name);
  }

  /**
   * Remove template
   */
  removeTemplate(name: string): boolean {
    const removed = this.templates.delete(name) && this.templatePriorities.delete(name);
    if (removed) {
      this.clearCacheForTemplate(name);
    }
    return removed;
  }

  /**
   * Clear all caches
   */
  clearAllCaches(): void {
    this.inheritanceCache.clear();
    this.parameterCache.clear();
    this.validationCache.clear();
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): {
    inheritanceCache: number;
    parameterCache: number;
    validationCache: number;
  } {
    return {
      inheritanceCache: this.inheritanceCache.size,
      parameterCache: this.parameterCache.size,
      validationCache: this.validationCache.size
    };
  }
}