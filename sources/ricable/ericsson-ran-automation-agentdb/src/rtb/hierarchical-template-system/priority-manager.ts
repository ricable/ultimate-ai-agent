/**
 * Priority Manager - Advanced Parameter Override and Priority Management
 *
 * Handles sophisticated parameter override logic, priority-based conflict resolution,
 * and dynamic priority adjustment strategies for RTB templates.
 */

import {
  RTBTemplate,
  RTBParameter,
  ConstraintSpec
} from '../../types/rtb-types';
import {
  TemplatePriorityInfo,
  TemplatePriority,
  ParameterConflict,
  TemplateResolutionContext
} from './priority-engine';
import { TemplateRegistry } from './template-registry';

/**
 * Priority adjustment rule
 */
export interface PriorityAdjustmentRule {
  name: string;
  condition: (context: PriorityContext) => boolean;
  adjustment: (priority: number, context: PriorityContext) => number;
  description: string;
  enabled: boolean;
  precedence: number; // Higher precedence rules are applied first
}

/**
 * Parameter override rule
 */
export interface ParameterOverrideRule {
  parameterPattern: RegExp;
  condition?: (context: OverrideContext) => boolean;
  strategy: OverrideStrategy;
  priority?: number; // Fixed priority for this parameter
  description: string;
  enabled: boolean;
}

/**
 * Override strategy
 */
export enum OverrideStrategy {
  HIGHEST_PRIORITY = 'highest_priority',
  LOWEST_PRIORITY = 'lowest_priority',
  MERGE_ALL = 'merge_all',
  CONCATENATE = 'concatenate',
  AVERAGE = 'average',
  SUM = 'sum',
  CUSTOM = 'custom'
}

/**
 * Priority evaluation context
 */
export interface PriorityContext {
  parameter: string;
  templateName: string;
  environment?: string;
  featureFlags?: Record<string, boolean>;
  metadata?: Record<string, any>;
  timestamp: Date;
  inheritanceDepth: number;
}

/**
 * Parameter override context
 */
export interface OverrideContext {
  parameter: string;
  values: Array<{
    value: any;
    source: string;
    priority: number;
    metadata?: Record<string, any>;
  }>;
  templateName: string;
  resolutionContext: TemplateResolutionContext;
}

/**
 * Priority rule result
 */
export interface PriorityRuleResult {
  originalPriority: number;
  adjustedPriority: number;
  appliedRules: Array<{
    name: string;
    adjustment: number;
    reason: string;
  }>;
  finalPriority: number;
}

/**
 * Parameter override result
 */
export interface ParameterOverrideResult {
  parameter: string;
  strategy: OverrideStrategy;
  originalValues: any[];
  resolvedValue: any;
  appliedRules: Array<{
    pattern: string;
    strategy: string;
    reason: string;
  }>;
  conflicts: ParameterConflict[];
  warnings: string[];
}

/**
 * Priority conflict resolution strategy
 */
export interface ConflictResolutionStrategy {
  name: string;
  canResolve: (conflicts: ParameterConflict[]) => boolean;
  resolve: (conflicts: ParameterConflict[], context: TemplateResolutionContext) => ParameterConflict[];
  priority: number; // Higher priority strategies are tried first
}

/**
 * Priority cache entry
 */
export interface PriorityCacheEntry {
  key: string;
  result: PriorityRuleResult | ParameterOverrideResult;
  timestamp: Date;
  ttl: number; // Time to live in milliseconds
  hitCount: number;
}

/**
 * Priority Manager
 *
 * Advanced system for managing parameter priorities, override rules,
 * and conflict resolution strategies with dynamic priority adjustment.
 */
export class PriorityManager {
  private registry: TemplateRegistry;
  private priorityRules: PriorityAdjustmentRule[] = [];
  private overrideRules: ParameterOverrideRule[] = [];
  private conflictStrategies: ConflictResolutionStrategy[] = [];
  private priorityCache = new Map<string, PriorityCacheEntry>();
  private customResolvers = new Map<string, (values: any[], context: OverrideContext) => any>();

  constructor(registry: TemplateRegistry) {
    this.registry = registry;
    this.initializeDefaultRules();
    this.initializeDefaultStrategies();
  }

  /**
   * Calculate adjusted priority for a template parameter
   */
  async calculateAdjustedPriority(
    templateName: string,
    parameter: string,
    basePriority: number,
    context: PriorityContext
  ): Promise<PriorityRuleResult> {
    const cacheKey = this.generatePriorityCacheKey(templateName, parameter, context);

    // Check cache first
    const cached = this.getFromCache(cacheKey);
    if (cached) {
      return cached as PriorityRuleResult;
    }

    const result: PriorityRuleResult = {
      originalPriority: basePriority,
      adjustedPriority: basePriority,
      appliedRules: [],
      finalPriority: basePriority
    };

    // Apply priority adjustment rules in precedence order
    const sortedRules = [...this.priorityRules]
      .filter(rule => rule.enabled)
      .sort((a, b) => b.precedence - a.precedence);

    for (const rule of sortedRules) {
      try {
        if (rule.condition(context)) {
          const adjustedValue = rule.adjustment(result.adjustedPriority, context);
          const adjustment = adjustedValue - result.adjustedPriority;

          result.adjustedPriority = adjustedValue;
          result.appliedRules.push({
            name: rule.name,
            adjustment,
            reason: rule.description
          });
        }
      } catch (error) {
        console.warn(`Priority rule '${rule.name}' failed:`, error);
      }
    }

    result.finalPriority = Math.max(0, Math.min(80, result.adjustedPriority));

    // Cache result
    this.setCache(cacheKey, result, 300000); // 5 minutes TTL

    return result;
  }

  /**
   * Resolve parameter overrides using strategy rules
   */
  async resolveParameterOverride(
    templateName: string,
    parameter: string,
    values: Array<{
      value: any;
      source: string;
      priority: number;
      metadata?: Record<string, any>;
    }>,
    context: TemplateResolutionContext
  ): Promise<ParameterOverrideResult> {
    const cacheKey = this.generateOverrideCacheKey(templateName, parameter, values, context);

    // Check cache first
    const cached = this.getFromCache(cacheKey);
    if (cached) {
      return cached as ParameterOverrideResult;
    }

    const overrideContext: OverrideContext = {
      parameter,
      values,
      templateName,
      resolutionContext: context
    };

    const result: ParameterOverrideResult = {
      parameter,
      strategy: OverrideStrategy.HIGHEST_PRIORITY,
      originalValues: values.map(v => v.value),
      resolvedValue: null,
      appliedRules: [],
      conflicts: [],
      warnings: []
    };

    // Find applicable override rules
    const applicableRules = this.overrideRules.filter(rule =>
      rule.enabled &&
      rule.parameterPattern.test(parameter) &&
      (!rule.condition || rule.condition(overrideContext))
    );

    // Sort rules by specificity (more specific patterns first)
    applicableRules.sort((a, b) => {
      const aSpecificity = this.calculatePatternSpecificity(a.parameterPattern);
      const bSpecificity = this.calculatePatternSpecificity(b.parameterPattern);
      return bSpecificity - aSpecificity;
    });

    // Apply first matching rule
    if (applicableRules.length > 0) {
      const rule = applicableRules[0];
      result.strategy = rule.strategy;

      result.resolvedValue = await this.applyOverrideStrategy(
        rule.strategy,
        values,
        overrideContext
      );

      result.appliedRules.push({
        pattern: rule.parameterPattern.source,
        strategy: rule.strategy,
        reason: rule.description
      });

      if (rule.priority !== undefined) {
        // Override priorities for all values
        values.forEach(v => v.priority = rule.priority!);
      }
    } else {
      // Default strategy: highest priority wins
      result.strategy = OverrideStrategy.HIGHEST_PRIORITY;
      const highestPriorityValue = values.reduce((prev, current) =>
        current.priority > prev.priority ? current : prev
      );
      result.resolvedValue = highestPriorityValue.value;
    }

    // Detect and log conflicts
    if (values.length > 1) {
      const conflicts = this.detectConflicts(values);
      result.conflicts.push(...conflicts);

      // Attempt to resolve conflicts
      if (result.conflicts.length > 0) {
        const resolvedConflicts = await this.resolveConflicts(result.conflicts, context);
        result.conflicts = resolvedConflicts;
      }
    }

    // Cache result
    this.setCache(cacheKey, result, 600000); // 10 minutes TTL

    return result;
  }

  /**
   * Apply override strategy to values
   */
  private async applyOverrideStrategy(
    strategy: OverrideStrategy,
    values: Array<{
      value: any;
      source: string;
      priority: number;
      metadata?: Record<string, any>;
    }>,
    context: OverrideContext
  ): Promise<any> {
    if (values.length === 0) return null;
    if (values.length === 1) return values[0].value;

    switch (strategy) {
      case OverrideStrategy.HIGHEST_PRIORITY:
        const highest = values.reduce((prev, current) =>
          current.priority > prev.priority ? current : prev
        );
        return highest.value;

      case OverrideStrategy.LOWEST_PRIORITY:
        const lowest = values.reduce((prev, current) =>
          current.priority < prev.priority ? current : prev
        );
        return lowest.value;

      case OverrideStrategy.MERGE_ALL:
        return this.mergeAllValues(values);

      case OverrideStrategy.CONCATENATE:
        return this.concatenateValues(values);

      case OverrideStrategy.AVERAGE:
        return this.averageValues(values);

      case OverrideStrategy.SUM:
        return this.sumValues(values);

      case OverrideStrategy.CUSTOM:
        const customResolver = this.customResolvers.get(context.parameter);
        if (customResolver) {
          return customResolver(values.map(v => v.value), context);
        }
        // Fall back to highest priority
        return values[values.length - 1].value;

      default:
        return values[values.length - 1].value;
    }
  }

  /**
   * Merge all values intelligently
   */
  private mergeAllValues(values: Array<{ value: any; source: string; priority: number }>): any {
    const allValues = values.map(v => v.value);

    // If all values are the same, return that value
    if (allValues.every(v => JSON.stringify(v) === JSON.stringify(allValues[0]))) {
      return allValues[0];
    }

    // Array merging
    if (allValues.every(v => Array.isArray(v))) {
      const merged = allValues.flat();
      return [...new Set(merged)]; // Remove duplicates
    }

    // Object merging
    if (allValues.every(v => typeof v === 'object' && v !== null)) {
      const merged = {};
      for (const value of allValues) {
        Object.assign(merged, value);
      }
      return merged;
    }

    // String merging
    if (allValues.every(v => typeof v === 'string')) {
      return allValues.join(' | ');
    }

    // Default: return highest priority value
    const highest = values.reduce((prev, current) =>
      current.priority > prev.priority ? current : prev
    );
    return highest.value;
  }

  /**
   * Concatenate values
   */
  private concatenateValues(values: Array<{ value: any; source: string; priority: number }>): any {
    const allValues = values.map(v => v.value);

    if (allValues.every(v => Array.isArray(v))) {
      return allValues.flat();
    }

    if (allValues.every(v => typeof v === 'string')) {
      return allValues.join('');
    }

    // For other types, create array
    return allValues;
  }

  /**
   * Average numeric values
   */
  private averageValues(values: Array<{ value: any; source: string; priority: number }>): any {
    const numericValues = values
      .map(v => v.value)
      .filter(v => typeof v === 'number' && !isNaN(v));

    if (numericValues.length === 0) {
      return values[values.length - 1].value; // Fall back to highest priority
    }

    return numericValues.reduce((sum, val) => sum + val, 0) / numericValues.length;
  }

  /**
   * Sum numeric values
   */
  private sumValues(values: Array<{ value: any; source: string; priority: number }>): any {
    const numericValues = values
      .map(v => v.value)
      .filter(v => typeof v === 'number' && !isNaN(v));

    if (numericValues.length === 0) {
      return values[values.length - 1].value; // Fall back to highest priority
    }

    return numericValues.reduce((sum, val) => sum + val, 0);
  }

  /**
   * Detect conflicts between values
   */
  private detectConflicts(values: Array<{
    value: any;
    source: string;
    priority: number;
    metadata?: Record<string, any>;
  }>): ParameterConflict[] {
    const conflicts: ParameterConflict[] = [];

    if (values.length < 2) return conflicts;

    // Group values by type and content
    const valueGroups = new Map<string, typeof values>();
    for (const v of values) {
      const key = this.getValueGroupKey(v.value);
      if (!valueGroups.has(key)) {
        valueGroups.set(key, []);
      }
      valueGroups.get(key)!.push(v);
    }

    // If we have multiple groups, there's a conflict
    if (valueGroups.size > 1) {
      const groups = Array.from(valueGroups.values());
      const primaryGroup = groups[groups.length - 1]; // Highest priority group

      for (let i = 0; i < groups.length - 1; i++) {
        const group = groups[i];
        conflicts.push({
          parameter: '', // Will be filled by caller
          templates: group.map(v => v.source),
          values: group.map(v => v.value),
          resolvedValue: primaryGroup[0]?.value,
          resolutionStrategy: 'highest_priority',
          reason: `Value conflict between ${groups.length} different value types/contents`
        });
      }
    }

    return conflicts;
  }

  /**
   * Get grouping key for value comparison
   */
  private getValueGroupKey(value: any): string {
    if (value === null || value === undefined) return 'null';
    if (typeof value === 'boolean') return 'boolean';
    if (typeof value === 'number') return 'number';
    if (typeof value === 'string') return `string:${value.substring(0, 50)}`;
    if (Array.isArray(value)) return `array:${value.length}`;
    if (typeof value === 'object') return `object:${Object.keys(value).length}`;
    return 'unknown';
  }

  /**
   * Resolve conflicts using available strategies
   */
  private async resolveConflicts(
    conflicts: ParameterConflict[],
    context: TemplateResolutionContext
  ): Promise<ParameterConflict[]> {
    const resolvedConflicts = [...conflicts];

    // Apply conflict resolution strategies in priority order
    const sortedStrategies = [...this.conflictStrategies]
      .sort((a, b) => b.priority - a.priority);

    for (const strategy of sortedStrategies) {
      if (strategy.canResolve(resolvedConflicts)) {
        try {
          resolvedConflicts.splice(0, resolvedConflicts.length,
            ...strategy.resolve(resolvedConflicts, context)
          );
          break; // Stop after first successful resolution
        } catch (error) {
          console.warn(`Conflict resolution strategy '${strategy.name}' failed:`, error);
        }
      }
    }

    return resolvedConflicts;
  }

  /**
   * Calculate pattern specificity (higher = more specific)
   */
  private calculatePatternSpecificity(pattern: RegExp): number {
    const source = pattern.source;

    // More specific patterns have higher specificity scores
    let specificity = 0;

    // Exact matches are most specific
    if (source.startsWith('^') && source.endsWith('$')) {
      specificity += 10;
    }

    // More characters = more specific
    specificity += source.length * 0.1;

    // More character classes = more specific
    const charClassMatches = source.match(/\[[^\]]*\]/g);
    if (charClassMatches) {
      specificity += charClassMatches.length * 2;
    }

    // More quantifiers = more specific
    const quantifierMatches = source.match(/[+*?{]/g);
    if (quantifierMatches) {
      specificity += quantifierMatches.length * 1;
    }

    return specificity;
  }

  /**
   * Add priority adjustment rule
   */
  addPriorityRule(rule: PriorityAdjustmentRule): void {
    this.priorityRules.push(rule);
    this.clearCache(); // Clear cache as rules have changed
  }

  /**
   * Add parameter override rule
   */
  addOverrideRule(rule: ParameterOverrideRule): void {
    this.overrideRules.push(rule);
    this.clearCache(); // Clear cache as rules have changed
  }

  /**
   * Add custom resolver for parameter
   */
  addCustomResolver(parameter: string, resolver: (values: any[], context: OverrideContext) => any): void {
    this.customResolvers.set(parameter, resolver);
    this.clearCache(); // Clear cache as resolvers have changed
  }

  /**
   * Remove priority rule
   */
  removePriorityRule(name: string): boolean {
    const index = this.priorityRules.findIndex(rule => rule.name === name);
    if (index > -1) {
      this.priorityRules.splice(index, 1);
      this.clearCache();
      return true;
    }
    return false;
  }

  /**
   * Remove override rule
   */
  removeOverrideRule(pattern: string): boolean {
    const index = this.overrideRules.findIndex(rule => rule.parameterPattern.source === pattern);
    if (index > -1) {
      this.overrideRules.splice(index, 1);
      this.clearCache();
      return true;
    }
    return false;
  }

  /**
   * Enable/disable rule
   */
  toggleRule(name: string, enabled: boolean): boolean {
    // Try priority rules first
    const priorityRule = this.priorityRules.find(rule => rule.name === name);
    if (priorityRule) {
      priorityRule.enabled = enabled;
      this.clearCache();
      return true;
    }

    // Try override rules
    const overrideRule = this.overrideRules.find(rule => rule.parameterPattern.source === name);
    if (overrideRule) {
      overrideRule.enabled = enabled;
      this.clearCache();
      return true;
    }

    return false;
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): {
    size: number;
    hitRate: number;
    totalHits: number;
    totalMisses: number;
  } {
    let totalHits = 0;
    let totalMisses = 0;

    for (const entry of this.priorityCache.values()) {
      totalHits += entry.hitCount;
    }

    return {
      size: this.priorityCache.size,
      hitRate: totalHits > 0 ? totalHits / (totalHits + totalMisses) : 0,
      totalHits,
      totalMisses
    };
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.priorityCache.clear();
  }

  /**
   * Generate priority cache key
   */
  private generatePriorityCacheKey(
    templateName: string,
    parameter: string,
    context: PriorityContext
  ): string {
    const contextHash = JSON.stringify({
      environment: context.environment,
      featureFlags: context.featureFlags,
      metadata: context.metadata,
      inheritanceDepth: context.inheritanceDepth
    });
    return `priority:${templateName}:${parameter}:${Buffer.from(contextHash).toString('base64')}`;
  }

  /**
   * Generate override cache key
   */
  private generateOverrideCacheKey(
    templateName: string,
    parameter: string,
    values: Array<{ value: any; source: string; priority: number }>,
    context: TemplateResolutionContext
  ): string {
    const valuesHash = JSON.stringify(values.map(v => v.value));
    const contextHash = JSON.stringify(context);
    return `override:${templateName}:${parameter}:${Buffer.from(valuesHash + contextHash).toString('base64')}`;
  }

  /**
   * Get value from cache
   */
  private getFromCache(key: string): any | null {
    const entry = this.priorityCache.get(key);
    if (!entry) return null;

    // Check if entry is expired
    if (Date.now() - entry.timestamp.getTime() > entry.ttl) {
      this.priorityCache.delete(key);
      return null;
    }

    // Update hit count
    entry.hitCount++;
    return entry.result;
  }

  /**
   * Set value in cache
   */
  private setCache(key: string, result: any, ttl: number): void {
    // Remove oldest entries if cache is too large
    if (this.priorityCache.size > 1000) {
      const oldestKey = this.priorityCache.keys().next().value;
      this.priorityCache.delete(oldestKey);
    }

    this.priorityCache.set(key, {
      key,
      result,
      timestamp: new Date(),
      ttl,
      hitCount: 0
    });
  }

  /**
   * Initialize default priority adjustment rules
   */
  private initializeDefaultRules(): void {
    // Environment-based adjustments
    this.priorityRules.push({
      name: 'production_high_priority',
      condition: (context) => context.environment === 'production',
      adjustment: (priority) => Math.max(0, priority - 5), // Lower number = higher priority
      description: 'Increase priority for production environment',
      enabled: true,
      precedence: 100
    });

    // Feature flag adjustments
    this.priorityRules.push({
      name: 'feature_flag_boost',
      condition: (context) => {
        return context.featureFlags &&
               Object.keys(context.featureFlags).some(flag => context.featureFlags![flag]);
      },
      adjustment: (priority, context) => {
        const activeFlags = Object.values(context.featureFlags || {}).filter(Boolean).length;
        return Math.max(0, priority - (activeFlags * 2)); // Boost for each active flag
      },
      description: 'Boost priority based on active feature flags',
      enabled: true,
      precedence: 90
    });

    // Inheritance depth penalty
    this.priorityRules.push({
      name: 'inheritance_depth_penalty',
      condition: (context) => context.inheritanceDepth > 3,
      adjustment: (priority, context) => priority + (context.inheritanceDepth - 3),
      description: 'Apply penalty for deep inheritance chains',
      enabled: true,
      precedence: 50
    });

    // Time-based adjustments
    this.priorityRules.push({
      name: 'recent_template_boost',
      condition: (context) => {
        const recentThreshold = new Date(Date.now() - 24 * 60 * 60 * 1000); // 24 hours
        return context.timestamp > recentThreshold;
      },
      adjustment: (priority) => Math.max(0, priority - 2),
      description: 'Boost priority for recently updated templates',
      enabled: true,
      precedence: 30
    });
  }

  /**
   * Initialize default override strategies
   */
  private initializeDefaultStrategies(): void {
    // Array merge strategy
    this.overrideRules.push({
      parameterPattern: /.*List$|.*Array$|.*Items$|.*Elements$/,
      strategy: OverrideStrategy.MERGE_ALL,
      description: 'Merge array-type parameters',
      enabled: true
    });

    // Boolean AND strategy
    this.overrideRules.push({
      parameterPattern: /.*Enabled$|.*Active$|.*Flag$|.*Enabled$|.*Disabled$/,
      strategy: OverrideStrategy.CUSTOM,
      description: 'Boolean parameters use AND logic',
      enabled: true
    });

    // Numeric max strategy
    this.overrideRules.push({
      parameterPattern: /.*Threshold$|.*Limit$|.*Maximum$|.*Max$|.*Count$/,
      strategy: OverrideStrategy.CUSTOM,
      description: 'Numeric limits use maximum value',
      enabled: true
    });

    // Numeric min strategy
    this.overrideRules.push({
      parameterPattern: /.*Minimum$|.*Min$|.*Delay$|.*Timeout$/,
      strategy: OverrideStrategy.CUSTOM,
      description: 'Numeric minimums use minimum value',
      enabled: true
    });

    // String concatenation strategy
    this.overrideRules.push({
      parameterPattern: /.*Name$|.*Title$|.*Label$|.*Description$/,
      strategy: OverrideStrategy.CONCATENATE,
      description: 'String parameters are concatenated',
      enabled: true
    });

    // Add custom resolvers
    this.customResolvers.set('.*Enabled$|.*Active$|.*Flag$|.*Enabled$|.*Disabled$',
      (values) => values.every(v => v === true)); // AND logic

    this.customResolvers.set('.*Threshold$|.*Limit$|.*Maximum$|.*Max$|.*Count$',
      (values) => Math.max(...values.filter(v => typeof v === 'number')));

    this.customResolvers.set('.*Minimum$|.*Min$|.*Delay$|.*Timeout$',
      (values) => Math.min(...values.filter(v => typeof v === 'number')));
  }

  /**
   * Initialize default conflict resolution strategies
   */
  private initializeDefaultStrategies(): void {
    // Add this method to set up default conflict resolution strategies
    this.conflictStrategies = [
      {
        name: 'priority_based_resolution',
        canResolve: (conflicts) => conflicts.length > 0,
        resolve: (conflicts) => conflicts, // Return as-is, priority already applied
        priority: 100
      }
    ];
  }

  /**
   * Get all priority rules
   */
  getPriorityRules(): PriorityAdjustmentRule[] {
    return [...this.priorityRules];
  }

  /**
   * Get all override rules
   */
  getOverrideRules(): ParameterOverrideRule[] {
    return [...this.overrideRules];
  }

  /**
   * Get conflict strategies
   */
  getConflictStrategies(): ConflictResolutionStrategy[] {
    return [...this.conflictStrategies];
  }
}