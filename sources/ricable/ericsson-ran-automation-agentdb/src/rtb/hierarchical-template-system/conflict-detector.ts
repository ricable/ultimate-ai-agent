/**
 * Conflict Detection System for Template Merging
 *
 * Identifies and categorizes conflicts between templates during the merging process.
 * Supports advanced conflict detection with pattern recognition and ML-based predictions.
 */

import { RTBTemplate, TemplateMeta } from '../../types/rtb-types';
import { TemplateConflict, ConflictType, ConflictContext, ConflictPattern } from './types';
import { Logger } from '../../utils/logger';

export interface ConflictDetectionOptions {
  /** Enable ML-based conflict prediction */
  enableMLPrediction: boolean;
  /** Strict mode for type checking */
  strictTypeChecking: boolean;
  /** Detect structural conflicts */
  detectStructuralConflicts: boolean;
  /** Analyze conditional logic conflicts */
  analyzeConditionalConflicts: boolean;
  /** Cache conflict patterns */
  cachePatterns: boolean;
  /** Maximum recursion depth for nested object analysis */
  maxRecursionDepth: number;
}

export interface ConflictDetectionResult {
  /** Detected conflicts */
  conflicts: TemplateConflict[];
  /** Conflict patterns identified */
  patterns: ConflictPattern[];
  /** Detection statistics */
  stats: ConflictDetectionStats;
}

export interface ConflictDetectionStats {
  /** Total parameters analyzed */
  totalParameters: number;
  /** Total conflicts detected */
  totalConflicts: number;
  /** Conflicts by type */
  conflictsByType: Record<ConflictType, number>;
  /** Conflicts by severity */
  conflictsBySeverity: Record<string, number>;
  /** Detection time in milliseconds */
  detectionTime: number;
  /** Pattern matches found */
  patternMatches: number;
}

export class ConflictDetector {
  private logger: Logger;
  private conflictPatterns: Map<string, ConflictPattern>;
  private options: ConflictDetectionOptions;

  constructor(options: Partial<ConflictDetectionOptions> = {}) {
    this.logger = new Logger('ConflictDetector');
    this.conflictPatterns = new Map();
    this.options = {
      enableMLPrediction: false,
      strictTypeChecking: true,
      detectStructuralConflicts: true,
      analyzeConditionalConflicts: true,
      cachePatterns: true,
      maxRecursionDepth: 10,
      ...options
    };
  }

  /**
   * Detect conflicts between multiple templates
   */
  async detectConflicts(templates: RTBTemplate[]): Promise<TemplateConflict[]> {
    const startTime = Date.now();
    this.logger.info(`Starting conflict detection for ${templates.length} templates`);

    const conflicts: TemplateConflict[] = [];
    const parameterMap = new Map<string, Array<{ template: RTBTemplate; value: any; priority: number; index: number }>>();

    // Build parameter map
    for (let i = 0; i < templates.length; i++) {
      const template = templates[i];
      const priority = template.meta?.priority || 0;

      this.extractParameters(template.configuration || {}, '', parameterMap, template, priority, i);
    }

    // Detect conflicts for each parameter
    for (const [parameter, values] of parameterMap.entries()) {
      if (values.length > 1) {
        const parameterConflicts = await this.detectParameterConflicts(parameter, values);
        conflicts.push(...parameterConflicts);
      }
    }

    // Detect structural conflicts
    if (this.options.detectStructuralConflicts) {
      const structuralConflicts = await this.detectStructuralConflicts(templates);
      conflicts.push(...structuralConflicts);
    }

    // Detect conditional conflicts
    if (this.options.analyzeConditionalConflicts) {
      const conditionalConflicts = await this.detectConditionalConflicts(templates);
      conflicts.push(...conditionalConflicts);
    }

    // Detect function conflicts
    const functionConflicts = await this.detectFunctionConflicts(templates);
    conflicts.push(...functionConflicts);

    // Detect metadata conflicts
    const metadataConflicts = await this.detectMetadataConflicts(templates);
    conflicts.push(...metadataConflicts);

    // Analyze and categorize conflicts
    await this.analyzeConflicts(conflicts);

    const detectionTime = Date.now() - startTime;
    this.logger.info(`Conflict detection completed`, {
      totalConflicts: conflicts.length,
      detectionTime,
      conflictsByType: this.getConflictsByType(conflicts)
    });

    return conflicts;
  }

  /**
   * Extract parameters from template configuration
   */
  private extractParameters(
    obj: any,
    path: string,
    parameterMap: Map<string, Array<{ template: RTBTemplate; value: any; priority: number; index: number }>>,
    template: RTBTemplate,
    priority: number,
    templateIndex: number,
    depth: number = 0
  ): void {
    if (depth > this.options.maxRecursionDepth) {
      this.logger.warn(`Maximum recursion depth exceeded at path: ${path}`);
      return;
    }

    for (const [key, value] of Object.entries(obj)) {
      const currentPath = path ? `${path}.${key}` : key;

      if (this.isObject(value) && !Array.isArray(value) && !this.isSpecialValue(value)) {
        this.extractParameters(value, currentPath, parameterMap, template, priority, templateIndex, depth + 1);
      } else {
        if (!parameterMap.has(currentPath)) {
          parameterMap.set(currentPath, []);
        }
        parameterMap.get(currentPath)!.push({
          template,
          value,
          priority,
          index: templateIndex
        });
      }
    }
  }

  /**
   * Detect conflicts for a specific parameter
   */
  private async detectParameterConflicts(
    parameter: string,
    values: Array<{ template: RTBTemplate; value: any; priority: number; index: number }>
  ): Promise<TemplateConflict[]> {
    const conflicts: TemplateConflict[] = [];
    const uniqueValues = new Map();

    // Group by value equality
    for (const { template, value, priority, index } of values) {
      const valueKey = this.getValueKey(value);
      if (!uniqueValues.has(valueKey)) {
        uniqueValues.set(valueKey, []);
      }
      uniqueValues.get(valueKey).push({ template, value, priority, index });
    }

    // If we have multiple unique values, there's a conflict
    if (uniqueValues.size > 1) {
      const conflictGroups = Array.from(uniqueValues.values());
      const allValues = conflictGroups.flat();
      const templates = allValues.map(v => v.template.meta?.description || v.template.meta?.source || 'unknown');
      const priorities = allValues.map(v => v.priority);
      const conflictValues = allValues.map(v => v.value);

      const conflictType = this.determineConflictType(conflictValues);
      const severity = this.determineSeverity(parameter, conflictType, priorities);

      const conflict: TemplateConflict = {
        parameter,
        templates,
        priorities,
        conflictType,
        values: conflictValues,
        resolution: {
          strategy: 'highest_priority',
          reasoning: `Conflict detected for parameter ${parameter} with ${uniqueValues.size} different values`,
          resolved: false
        },
        resolved: false,
        context: await this.buildConflictContext(parameter, conflictValues, templates),
        severity,
        requiresManualIntervention: severity === 'critical' || conflictType === 'conditional'
      };

      conflicts.push(conflict);
    }

    return conflicts;
  }

  /**
   * Detect structural conflicts between templates
   */
  private async detectStructuralConflicts(templates: RTBTemplate[]): Promise<TemplateConflict[]> {
    const conflicts: TemplateConflict[] = [];
    const structureMap = new Map<string, Array<{ template: RTBTemplate; structure: any; priority: number }>>();

    // Analyze template structures
    for (const template of templates) {
      const structure = this.analyzeStructure(template.configuration || {});
      const structureKey = JSON.stringify(structure);

      if (!structureMap.has(structureKey)) {
        structureMap.set(structureKey, []);
      }

      structureMap.get(structureKey)!.push({
        template,
        structure,
        priority: template.meta?.priority || 0
      });
    }

    // Detect structural conflicts
    if (structureMap.size > 1) {
      const structures = Array.from(structureMap.values());
      const templates = structures.map(s => s[0].template.meta?.description || 'unknown');
      const priorities = structures.map(s => s[0].priority);

      const conflict: TemplateConflict = {
        parameter: '__template_structure__',
        templates,
        priorities,
        conflictType: 'structure',
        values: structures.map(s => s[0].structure),
        resolution: {
          strategy: 'merge',
          reasoning: 'Templates have different structural layouts - deep merge required',
          resolved: false
        },
        resolved: false,
        severity: 'medium',
        requiresManualIntervention: false
      };

      conflicts.push(conflict);
    }

    return conflicts;
  }

  /**
   * Detect conditional logic conflicts
   */
  private async detectConditionalConflicts(templates: RTBTemplate[]): Promise<TemplateConflict[]> {
    const conflicts: TemplateConflict[] = [];
    const conditionMap = new Map<string, Array<{ template: RTBTemplate; condition: any; priority: number }>>();

    // Collect all conditions
    for (const template of templates) {
      const conditions = template.conditions || {};
      for (const [key, condition] of Object.entries(conditions)) {
        if (!conditionMap.has(key)) {
          conditionMap.set(key, []);
        }
        conditionMap.get(key)!.push({
          template,
          condition,
          priority: template.meta?.priority || 0
        });
      }
    }

    // Detect condition conflicts
    for (const [conditionKey, conditionList] of conditionMap.entries()) {
      if (conditionList.length > 1) {
        const templates = conditionList.map(c => c.template.meta?.description || 'unknown');
        const priorities = conditionList.map(c => c.priority);
        const values = conditionList.map(c => c.condition);

        // Check for contradictory conditions
        if (this.hasContradictoryConditions(values)) {
          const conflict: TemplateConflict = {
            parameter: `condition.${conditionKey}`,
            templates,
            priorities,
            conflictType: 'conditional',
            values,
            resolution: {
              strategy: 'conditional',
              reasoning: `Contradictory conditions detected for ${conditionKey}`,
              resolved: false
            },
            resolved: false,
            severity: 'high',
            requiresManualIntervention: true
          };

          conflicts.push(conflict);
        }
      }
    }

    return conflicts;
  }

  /**
   * Detect function conflicts
   */
  private async detectFunctionConflicts(templates: RTBTemplate[]): Promise<TemplateConflict[]> {
    const conflicts: TemplateConflict[] = [];
    const functionMap = new Map<string, Array<{ template: RTBTemplate; func: any; priority: number }>>();

    // Collect all custom functions
    for (const template of templates) {
      const functions = template.custom || [];
      for (const func of functions) {
        if (!functionMap.has(func.name)) {
          functionMap.set(func.name, []);
        }
        functionMap.get(func.name)!.push({
          template,
          func,
          priority: template.meta?.priority || 0
        });
      }
    }

    // Detect function conflicts
    for (const [funcName, funcList] of functionMap.entries()) {
      if (funcList.length > 1) {
        const templates = funcList.map(f => f.template.meta?.description || 'unknown');
        const priorities = funcList.map(f => f.priority);
        const values = funcList.map(f => f.func);

        // Check for different implementations
        if (this.hasDifferentFunctionImplementations(values)) {
          const conflict: TemplateConflict = {
            parameter: `function.${funcName}`,
            templates,
            priorities,
            conflictType: 'function',
            values,
            resolution: {
              strategy: 'highest_priority',
              reasoning: `Different implementations detected for function ${funcName}`,
              resolved: false
            },
            resolved: false,
            severity: 'medium',
            requiresManualIntervention: true
          };

          conflicts.push(conflict);
        }
      }
    }

    return conflicts;
  }

  /**
   * Detect metadata conflicts
   */
  private async detectMetadataConflicts(templates: RTBTemplate[]): Promise<TemplateConflict[]> {
    const conflicts: TemplateConflict[] = [];
    const metadataFields = ['version', 'environment', 'author', 'tags'];

    for (const field of metadataFields) {
      const values = templates.map(t => t.meta?.[field]).filter(v => v !== undefined);
      if (values.length > 1) {
        const uniqueValues = new Set(values.map(v => Array.isArray(v) ? JSON.stringify(v.sort()) : v));

        if (uniqueValues.size > 1) {
          const templatesWithValues = templates.filter(t => t.meta?.[field] !== undefined);
          const templateNames = templatesWithValues.map(t => t.meta?.description || 'unknown');
          const priorities = templatesWithValues.map(t => t.meta?.priority || 0);

          const conflict: TemplateConflict = {
            parameter: `metadata.${field}`,
            templates: templateNames,
            priorities,
            conflictType: 'metadata',
            values: Array.from(uniqueValues),
            resolution: {
              strategy: field === 'tags' || field === 'author' ? 'merge' : 'highest_priority',
              reasoning: `Metadata conflict detected for field ${field}`,
              resolved: false
            },
            resolved: false,
            severity: 'low',
            requiresManualIntervention: false
          };

          conflicts.push(conflict);
        }
      }
    }

    return conflicts;
  }

  /**
   * Analyze and enhance conflicts with additional context
   */
  private async analyzeConflicts(conflicts: TemplateConflict[]): Promise<void> {
    for (const conflict of conflicts) {
      // Apply ML prediction if enabled
      if (this.options.enableMLPrediction) {
        const recommendedStrategy = await this.predictResolutionStrategy(conflict);
        if (recommendedStrategy && conflict.resolution.strategy !== recommendedStrategy) {
          conflict.resolution.strategy = recommendedStrategy;
          conflict.resolution.reasoning += ` (ML recommended: ${recommendedStrategy})`;
        }
      }

      // Update pattern cache
      if (this.options.cachePatterns) {
        this.updateConflictPatterns(conflict);
      }
    }
  }

  /**
   * Build conflict context
   */
  private async buildConflictContext(
    parameter: string,
    values: any[],
    templates: string[]
  ): Promise<ConflictContext> {
    const parameterType = this.inferParameterType(values);
    const historicalConflicts = this.getHistoricalConflictCount(parameter);
    const recommendedResolution = this.getRecommendedResolution(parameter, values);

    return {
      inheritanceDepth: parameter.split('.').length,
      inheritancePath: parameter.split('.'),
      parameterType,
      constraints: [], // Could be enhanced with actual constraint extraction
      historicalConflicts,
      recommendedResolution
    };
  }

  /**
   * Determine conflict type based on values
   */
  private determineConflictType(values: any[]): ConflictType {
    const types = values.map(v => this.getValueType(v));
    const uniqueTypes = new Set(types);

    if (uniqueTypes.size > 1) {
      return 'type';
    }

    const mainType = types[0];
    if (mainType === 'object') {
      return 'structure';
    }

    return 'value';
  }

  /**
   * Determine conflict severity
   */
  private determineSeverity(parameter: string, conflictType: ConflictType, priorities: number[]): 'low' | 'medium' | 'high' | 'critical' {
    if (conflictType === 'conditional') {
      return 'critical';
    }

    if (conflictType === 'function') {
      return 'high';
    }

    if (conflictType === 'structure') {
      return 'medium';
    }

    // Check priority differences
    const maxPriority = Math.max(...priorities);
    const minPriority = Math.min(...priorities);
    if (maxPriority - minPriority > 5) {
      return 'medium';
    }

    return 'low';
  }

  /**
   * Get value key for comparison
   */
  private getValueKey(value: any): string {
    if (value === null || value === undefined) {
      return 'null';
    }

    if (typeof value === 'object') {
      return JSON.stringify(value, Object.keys(value).sort());
    }

    return String(value);
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
   * Infer parameter type from values
   */
  private inferParameterType(values: any[]): string | undefined {
    const types = values.map(v => this.getValueType(v));
    const typeCounts = types.reduce((acc, type) => {
      acc[type] = (acc[type] || 0) + 1;
      return acc;
    }, {});

    return Object.keys(typeCounts).reduce((a, b) => typeCounts[a] > typeCounts[b] ? a : b);
  }

  /**
   * Check if value is an object
   */
  private isObject(value: any): boolean {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
  }

  /**
   * Check if value is a special value that should not be recursed into
   */
  private isSpecialValue(value: any): boolean {
    return value instanceof Date || value instanceof RegExp || value instanceof Function;
  }

  /**
   * Analyze template structure
   */
  private analyzeStructure(obj: any, depth: number = 0): any {
    if (depth > 5 || !this.isObject(obj)) {
      return this.getValueType(obj);
    }

    const structure: any = {};
    for (const [key, value] of Object.entries(obj)) {
      if (this.isObject(value)) {
        structure[key] = this.analyzeStructure(value, depth + 1);
      } else {
        structure[key] = this.getValueType(value);
      }
    }

    return structure;
  }

  /**
   * Check for contradictory conditions
   */
  private hasContradictoryConditions(conditions: any[]): boolean {
    // Simple implementation - could be enhanced with more sophisticated logic
    const conditionsStr = conditions.map(c => JSON.stringify(c));
    return new Set(conditionsStr).size > 1;
  }

  /**
   * Check for different function implementations
   */
  private hasDifferentFunctionImplementations(functions: any[]): boolean {
    const implementations = functions.map(f => JSON.stringify(f.body || ''));
    return new Set(implementations).size > 1;
  }

  /**
   * Predict resolution strategy using ML (placeholder)
   */
  private async predictResolutionStrategy(conflict: TemplateConflict): Promise<ResolutionStrategyType | null> {
    // Placeholder for ML-based prediction
    // Could be enhanced with actual ML model
    return null;
  }

  /**
   * Update conflict patterns cache
   */
  private updateConflictPatterns(conflict: TemplateConflict): void {
    const patternKey = `${conflict.conflictType}:${conflict.parameter.split('.')[0]}`;

    if (!this.conflictPatterns.has(patternKey)) {
      this.conflictPatterns.set(patternKey, {
        pattern: patternKey,
        frequency: 0,
        commonResolution: conflict.resolution.strategy,
        successRate: 0
      });
    }

    const pattern = this.conflictPatterns.get(patternKey)!;
    pattern.frequency++;
  }

  /**
   * Get historical conflict count
   */
  private getHistoricalConflictCount(parameter: string): number {
    const patternKey = `${parameter.split('.')[0]}`;
    const pattern = this.conflictPatterns.get(patternKey);
    return pattern ? pattern.frequency : 0;
  }

  /**
   * Get recommended resolution
   */
  private getRecommendedResolution(parameter: string, values: any[]): ResolutionStrategyType {
    const patternKey = `${parameter.split('.')[0]}`;
    const pattern = this.conflictPatterns.get(patternKey);
    return pattern ? pattern.commonResolution : 'highest_priority';
  }

  /**
   * Get conflicts by type
   */
  private getConflictsByType(conflicts: TemplateConflict[]): Record<ConflictType, number> {
    return conflicts.reduce((acc, conflict) => {
      acc[conflict.conflictType] = (acc[conflict.conflictType] || 0) + 1;
      return acc;
    }, {} as Record<ConflictType, number>);
  }

  /**
   * Get conflict patterns
   */
  public getConflictPatterns(): ConflictPattern[] {
    return Array.from(this.conflictPatterns.values());
  }

  /**
   * Clear conflict patterns cache
   */
  public clearPatterns(): void {
    this.conflictPatterns.clear();
  }
}