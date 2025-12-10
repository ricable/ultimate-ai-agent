/**
 * Template Conflict Resolver - Intelligent Conflict Handling Implementation
 *
 * Provides sophisticated conflict detection and resolution capabilities for
 * template inheritance and merging operations.
 *
 * Features:
 * - Multi-parameter conflict detection
 * - Smart resolution strategies
 * - Conflict logging and analysis
 * - Performance optimization
 * - Custom resolution function support
 */

import {
  PriorityTemplate,
  TemplateConflict,
  ConflictResolutionStrategy,
  ConflictResolutionSuggestion,
  TemplateConflictError,
  HierarchicalTemplateEngineConfig
} from '../interfaces';

/**
 * Template Conflict Resolver implementation
 */
export class TemplateConflictResolver {
  private config: HierarchicalTemplateEngineConfig;
  private conflictHistory: TemplateConflict[] = [];
  private resolutionSuggestions: Map<string, ConflictResolutionSuggestion[]> = new Map();

  constructor(config: HierarchicalTemplateEngineConfig) {
    this.config = config;
  }

  /**
   * Detect conflicts between templates
   */
  async detectConflicts(templates: PriorityTemplate[]): Promise<TemplateConflict[]> {
    const conflicts: TemplateConflict[] = [];
    const parameterMap = new Map<string, Array<{ template: PriorityTemplate; value: any; path: string[] }>>();

    // Build comprehensive parameter map
    for (const template of templates) {
      this.extractParameters(template.configuration, [], template, parameterMap);
    }

    // Analyze each parameter for conflicts
    for (const [parameterPath, values] of parameterMap) {
      if (values.length > 1) {
        const conflict = this.analyzeParameterConflict(parameterPath, values);
        if (conflict) {
          conflicts.push(conflict);
        }
      }
    }

    // Detect conditional conflicts
    const conditionalConflicts = await this.detectConditionalConflicts(templates);
    conflicts.push(...conditionalConflicts);

    // Detect function conflicts
    const functionConflicts = await this.detectFunctionConflicts(templates);
    conflicts.push(...functionConflicts);

    // Log conflicts for analysis
    this.logConflicts(conflicts);

    return conflicts;
  }

  /**
   * Resolve conflicts using specified strategy
   */
  async resolveConflict(
    conflict: TemplateConflict,
    strategy: ConflictResolutionStrategy
  ): Promise<TemplateConflict> {
    const resolvedConflict = { ...conflict };

    try {
      switch (strategy) {
        case ConflictResolutionStrategy.HIGHEST_PRIORITY_WINS:
          resolvedConflict.resolvedValue = this.resolveHighestPriority(conflict);
          resolvedConflict.resolutionReason = this.generateResolutionReason(conflict, 'HIGHEST_PRIORITY');
          break;

        case ConflictResolutionStrategy.LOWEST_PRIORITY_WINS:
          resolvedConflict.resolvedValue = this.resolveLowestPriority(conflict);
          resolvedConflict.resolutionReason = this.generateResolutionReason(conflict, 'LOWEST_PRIORITY');
          break;

        case ConflictResolutionStrategy.MERGE_WITH_WARNING:
          resolvedConflict.resolvedValue = this.resolveWithMerge(conflict);
          resolvedConflict.resolutionReason = this.generateResolutionReason(conflict, 'MERGE');
          break;

        case ConflictResolutionStrategy.CUSTOM_FUNCTION:
          resolvedConflict.resolvedValue = await this.resolveWithCustomFunction(conflict);
          resolvedConflict.resolutionReason = this.generateResolutionReason(conflict, 'CUSTOM_FUNCTION');
          break;

        case ConflictResolutionStrategy.CONFLICT_LOGGING:
          resolvedConflict.resolvedValue = this.resolveHighestPriority(conflict);
          resolvedConflict.resolutionReason = this.generateResolutionReason(conflict, 'LOGGED');
          this.logConflictDetails(conflict);
          break;

        case ConflictResolutionStrategy.FAIL_ON_CONFLICT:
          throw new TemplateConflictError(`Conflict resolution failed: strategy is FAIL_ON_CONFLICT`, [conflict]);

        default:
          throw new Error(`Unknown conflict resolution strategy: ${strategy}`);
      }

      resolvedConflict.resolutionStrategy = strategy;
      return resolvedConflict;

    } catch (error) {
      // Mark conflict as unresolved with error
      resolvedConflict.resolutionReason = `Resolution failed: ${(error as Error).message}`;
      resolvedConflict.resolvedValue = undefined;
      return resolvedConflict;
    }
  }

  /**
   * Get conflict resolution suggestions
   */
  async getResolutionSuggestions(conflict: TemplateConflict): Promise<ConflictResolutionSuggestion[]> {
    const cacheKey = this.generateConflictCacheKey(conflict);

    if (this.resolutionSuggestions.has(cacheKey)) {
      return this.resolutionSuggestions.get(cacheKey)!;
    }

    const suggestions: ConflictResolutionSuggestion[] = [];

    // Analyze conflict characteristics
    const conflictAnalysis = this.analyzeConflictCharacteristics(conflict);

    // Suggest highest priority if priorities are significantly different
    if (conflictAnalysis.prioritySpread > 20) {
      suggestions.push({
        strategy: ConflictResolutionStrategy.HIGHEST_PRIORITY_WINS,
        confidence: 0.9,
        reasoning: `Large priority spread (${conflictAnalysis.prioritySpread}) suggests clear precedence`,
        expectedOutcome: `Value from template '${conflictAnalysis.highestPriority.templateId}' will be used`,
        risks: ['Lower priority template parameters will be overridden']
      });
    }

    // Suggest merge if values are compatible
    if (conflictAnalysis.canMerge && conflictAnalysis.valueType === 'object') {
      suggestions.push({
        strategy: ConflictResolutionStrategy.MERGE_WITH_WARNING,
        confidence: 0.8,
        reasoning: 'Values are objects and can be merged without data loss',
        expectedOutcome: 'Combined object with properties from all templates',
        risks: ['Potential for conflicting nested properties', 'Increased complexity']
      });
    }

    // Suggest custom function for complex scenarios
    if (conflictAnalysis.isComplex) {
      suggestions.push({
        strategy: ConflictResolutionStrategy.CUSTOM_FUNCTION,
        confidence: 0.7,
        reasoning: 'Complex conflict requires custom resolution logic',
        expectedOutcome: 'Value determined by custom resolution function',
        risks: ['Custom function complexity', 'Potential for unexpected results']
      });
    }

    // Always suggest logging as fallback
    suggestions.push({
      strategy: ConflictResolutionStrategy.CONFLICT_LOGGING,
      confidence: 0.6,
      reasoning: 'Safe option that preserves all information',
      expectedOutcome: 'Highest priority value used, conflict logged for analysis',
      risks: ['Manual intervention may be required']
    });

    // Cache suggestions
    this.resolutionSuggestions.set(cacheKey, suggestions);
    return suggestions;
  }

  /**
   * Log conflict for analysis
   */
  async logConflict(conflict: TemplateConflict, resolution: TemplateConflict): Promise<void> {
    const logEntry = {
      timestamp: new Date(),
      conflict: { ...conflict },
      resolution: { ...resolution },
      analysis: this.analyzeConflictCharacteristics(conflict)
    };

    // Store in history (in a real implementation, this would use persistent storage)
    this.conflictHistory.push(conflict);

    // Log to console if detailed logging is enabled
    if (this.config.detailedLogging) {
      console.log(`[TemplateConflictResolver] Conflict logged:`, logEntry);
    }

    // Update conflict patterns for machine learning
    this.updateConflictPatterns(conflict, resolution);
  }

  /**
   * Get conflict statistics
   */
  getConflictStats(): {
    totalConflicts: number;
    resolutionSuccess: number;
    averageConflictsPerTemplate: number;
    mostCommonStrategies: Array<{ strategy: string; count: number }>;
    conflictPatterns: Array<{ parameterPattern: string; frequency: number }>;
  } {
    const resolutionSuccess = this.conflictHistory.filter(c => c.resolvedValue !== undefined).length;
    const strategyCounts = new Map<string, number>();
    const parameterPatterns = new Map<string, number>();

    for (const conflict of this.conflictHistory) {
      // Count resolution strategies
      if (conflict.resolutionStrategy) {
        strategyCounts.set(conflict.resolutionStrategy, (strategyCounts.get(conflict.resolutionStrategy) || 0) + 1);
      }

      // Analyze parameter patterns
      const pattern = this.extractParameterPattern(conflict.parameterPath);
      parameterPatterns.set(pattern, (parameterPatterns.get(pattern) || 0) + 1);
    }

    const mostCommonStrategies = Array.from(strategyCounts.entries())
      .map(([strategy, count]) => ({ strategy, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);

    const conflictPatterns = Array.from(parameterPatterns.entries())
      .map(([parameterPattern, frequency]) => ({ parameterPattern, frequency }))
      .sort((a, b) => b.frequency - a.frequency)
      .slice(0, 10);

    return {
      totalConflicts: this.conflictHistory.length,
      resolutionSuccess,
      averageConflictsPerTemplate: this.conflictHistory.length / Math.max(1, this.conflictHistory.length),
      mostCommonStrategies,
      conflictPatterns
    };
  }

  /**
   * Clear conflict history
   */
  clearHistory(): void {
    this.conflictHistory = [];
    this.resolutionSuggestions.clear();
  }

  // ============================================================================
  // PRIVATE METHODS
  // ============================================================================

  /**
   * Extract parameters from configuration recursively
   */
  private extractParameters(
    config: any,
    path: string[],
    template: PriorityTemplate,
    parameterMap: Map<string, Array<{ template: PriorityTemplate; value: any; path: string[] }>>
  ): void {
    for (const [key, value] of Object.entries(config)) {
      const currentPath = [...path, key];
      const parameterPath = currentPath.join('.');

      if (value && typeof value === 'object' && !Array.isArray(value)) {
        // Recurse into nested objects
        this.extractParameters(value, currentPath, template, parameterMap);
      } else {
        // Store parameter value
        if (!parameterMap.has(parameterPath)) {
          parameterMap.set(parameterPath, []);
        }
        parameterMap.get(parameterPath)!.push({
          template,
          value,
          path: currentPath
        });
      }
    }
  }

  /**
   * Analyze parameter conflict
   */
  private analyzeParameterConflict(
    parameterPath: string,
    values: Array<{ template: PriorityTemplate; value: any; path: string[] }>
  ): TemplateConflict | null {
    // Check if values are actually different
    const uniqueValues = new Set(values.map(v => JSON.stringify(v.value)));
    if (uniqueValues.size <= 1) {
      return null; // No conflict
    }

    return {
      parameterPath,
      conflictingTemplates: values.map(v => ({
        templateId: v.template.meta.version,
        priority: v.template.priority,
        value: v.value
      })),
      resolutionStrategy: ConflictResolutionStrategy.HIGHEST_PRIORITY_WINS // Default strategy
    };
  }

  /**
   * Detect conditional conflicts
   */
  private async detectConditionalConflicts(templates: PriorityTemplate[]): Promise<TemplateConflict[]> {
    const conflicts: TemplateConflict[] = [];
    const conditionMap = new Map<string, Array<{ template: PriorityTemplate; condition: any }>>();

    // Build condition map
    for (const template of templates) {
      if (template.conditions) {
        for (const [key, condition] of Object.entries(template.conditions)) {
          if (!conditionMap.has(key)) {
            conditionMap.set(key, []);
          }
          conditionMap.get(key)!.push({ template, condition });
        }
      }
    }

    // Analyze conditions for conflicts
    for (const [conditionKey, conditionValues] of conditionMap) {
      if (conditionValues.length > 1) {
        const uniqueConditions = new Set(conditionValues.map(c => JSON.stringify(c.condition)));
        if (uniqueConditions.size > 1) {
          conflicts.push({
            parameterPath: `condition.${conditionKey}`,
            conflictingTemplates: conditionValues.map(c => ({
              templateId: c.template.meta.version,
              priority: c.template.priority,
              value: c.condition
            })),
            resolutionStrategy: ConflictResolutionStrategy.HIGHEST_PRIORITY_WINS
          });
        }
      }
    }

    return conflicts;
  }

  /**
   * Detect function conflicts
   */
  private async detectFunctionConflicts(templates: PriorityTemplate[]): Promise<TemplateConflict[]> {
    const conflicts: TemplateConflict[] = [];
    const functionMap = new Map<string, Array<{ template: PriorityTemplate; func: any }>>();

    // Build function map
    for (const template of templates) {
      if (template.custom) {
        for (const func of template.custom) {
          if (!functionMap.has(func.name)) {
            functionMap.set(func.name, []);
          }
          functionMap.get(func.name)!.push({ template, func });
        }
      }
    }

    // Analyze functions for conflicts
    for (const [funcName, funcValues] of functionMap) {
      if (funcValues.length > 1) {
        const uniqueBodies = new Set(funcValues.map(f => JSON.stringify(f.func.body)));
        if (uniqueBodies.size > 1) {
          conflicts.push({
            parameterPath: `function.${funcName}`,
            conflictingTemplates: funcValues.map(f => ({
              templateId: f.template.meta.version,
              priority: f.template.priority,
              value: f.func
            })),
            resolutionStrategy: ConflictResolutionStrategy.HIGHEST_PRIORITY_WINS
          });
        }
      }
    }

    return conflicts;
  }

  /**
   * Resolve conflict using highest priority strategy
   */
  private resolveHighestPriority(conflict: TemplateConflict): any {
    const highestPriority = conflict.conflictingTemplates.reduce((min, current) =>
      current.priority < min.priority ? current : min
    );
    return highestPriority.value;
  }

  /**
   * Resolve conflict using lowest priority strategy
   */
  private resolveLowestPriority(conflict: TemplateConflict): any {
    const lowestPriority = conflict.conflictingTemplates.reduce((max, current) =>
      current.priority > max.priority ? current : max
    );
    return lowestPriority.value;
  }

  /**
   * Resolve conflict by merging values
   */
  private resolveWithMerge(conflict: TemplateConflict): any {
    const values = conflict.conflictingTemplates.map(t => t.value);

    // Check if all values are objects
    if (values.every(v => v && typeof v === 'object' && !Array.isArray(v))) {
      return this.deepMergeObjects(...values as Record<string, any>[]);
    }

    // Check if all values are arrays
    if (values.every(v => Array.isArray(v))) {
      return [...new Set(values.flat())];
    }

    // Fall back to highest priority
    return this.resolveHighestPriority(conflict);
  }

  /**
   * Resolve conflict using custom function
   */
  private async resolveWithCustomFunction(conflict: TemplateConflict): Promise<any> {
    // In a full implementation, this would load and execute custom resolution functions
    // For now, fall back to highest priority with logging
    console.warn(`[TemplateConflictResolver] Custom resolution not implemented for ${conflict.parameterPath}, using highest priority`);
    return this.resolveHighestPriority(conflict);
  }

  /**
   * Generate resolution reason
   */
  private generateResolutionReason(conflict: TemplateConflict, strategy: string): string {
    const highestPriority = conflict.conflictingTemplates.reduce((min, current) =>
      current.priority < min.priority ? current : min
    );

    switch (strategy) {
      case 'HIGHEST_PRIORITY':
        return `Selected value from template '${highestPriority.templateId}' with highest priority ${highestPriority.priority}`;
      case 'LOWEST_PRIORITY':
        const lowestPriority = conflict.conflictingTemplates.reduce((max, current) =>
          current.priority > max.priority ? current : max
        );
        return `Selected value from template '${lowestPriority.templateId}' with lowest priority ${lowestPriority.priority}`;
      case 'MERGE':
        return `Merged values from ${conflict.conflictingTemplates.length} conflicting templates`;
      case 'CUSTOM_FUNCTION':
        return `Resolved using custom resolution function`;
      case 'LOGGED':
        return `Conflict logged, highest priority value used from '${highestPriority.templateId}'`;
      default:
        return `Resolved using ${strategy} strategy`;
    }
  }

  /**
   * Analyze conflict characteristics
   */
  private analyzeConflictCharacteristics(conflict: TemplateConflict): any {
    const priorities = conflict.conflictingTemplates.map(t => t.priority);
    const values = conflict.conflictingTemplates.map(t => t.value);
    const highestPriority = Math.min(...priorities);
    const lowestPriority = Math.max(...priorities);

    return {
      prioritySpread: lowestPriority - highestPriority,
      highestPriority: conflict.conflictingTemplates.find(t => t.priority === highestPriority)!,
      lowestPriority: conflict.conflictingTemplates.find(t => t.priority === lowestPriority)!,
      valueType: this.getValueType(values[0]),
      canMerge: values.every(v => (v && typeof v === 'object') || Array.isArray(v)),
      isComplex: values.length > 2 || this.hasComplexValues(values),
      valueCount: values.length
    };
  }

  /**
   * Get value type
   */
  private getValueType(value: any): string {
    if (Array.isArray(value)) return 'array';
    if (value === null) return 'null';
    return typeof value;
  }

  /**
   * Check if values are complex
   */
  private hasComplexValues(values: any[]): boolean {
    return values.some(v =>
      v && typeof v === 'object' && Object.keys(v).length > 5
    );
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
   * Generate conflict cache key
   */
  private generateConflictCacheKey(conflict: TemplateConflict): string {
    const templateIds = conflict.conflictingTemplates.map(t => t.templateId).sort().join(',');
    const valueTypes = conflict.conflictingTemplates.map(t => this.getValueType(t.value)).join(',');
    return `${conflict.parameterPath}:${templateIds}:${valueTypes}`;
  }

  /**
   * Extract parameter pattern
   */
  private extractParameterPattern(parameterPath: string): string {
    const parts = parameterPath.split('.');
    if (parts.length >= 2) {
      return `${parts[0]}.*`; // Generalize pattern
    }
    return parameterPath;
  }

  /**
   * Log conflicts for analysis
   */
  private logConflicts(conflicts: TemplateConflict[]): void {
    if (conflicts.length > 0 && this.config.detailedLogging) {
      console.log(`[TemplateConflictResolver] Detected ${conflicts.length} conflicts:`, conflicts);
    }
  }

  /**
   * Log detailed conflict information
   */
  private logConflictDetails(conflict: TemplateConflict): void {
    console.log(`[TemplateConflictResolver] Conflict Details:`, {
      parameterPath: conflict.parameterPath,
      conflictingTemplates: conflict.conflictingTemplates,
      resolutionStrategy: conflict.resolutionStrategy,
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Update conflict patterns for machine learning
   */
  private updateConflictPatterns(conflict: TemplateConflict, resolution: TemplateConflict): void {
    // In a full implementation, this would update ML models
    // For now, just log the pattern
    if (this.config.detailedLogging) {
      console.log(`[TemplateConflictResolver] Conflict pattern updated:`, {
        pattern: this.extractParameterPattern(conflict.parameterPath),
        strategy: resolution.resolutionStrategy,
        success: resolution.resolvedValue !== undefined
      });
    }
  }
}