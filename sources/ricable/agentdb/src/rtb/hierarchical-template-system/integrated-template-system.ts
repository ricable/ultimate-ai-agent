/**
 * Integrated Template System - Complete RTB Template Processing with Priority
 *
 * Integrates the hierarchical template system with the existing RTB processor,
 * schema validation, and MO hierarchy for end-to-end template processing.
 */

import {
  RTBTemplate,
  RTBParameter,
  MOClass,
  MOHierarchy,
  ConstraintSpec,
  ProcessingStats
} from '../../types/rtb-types';
import {
  TemplatePriorityInfo,
  TemplatePriority,
  TemplateInheritanceChain,
  TemplateResolutionContext
} from './priority-engine';
import { TemplateRegistry } from './template-registry';
import { PriorityTemplateEngine } from './priority-engine';
import { InheritanceResolver } from './inheritance-resolver';
import { PriorityManager } from './priority-manager';
import { TemplateValidator } from './template-validator';
import { PerformanceOptimizer } from './performance-optimizer';

/**
 * Integration configuration
 */
export interface IntegratedSystemConfig {
  enablePrioritySystem?: boolean;
  enableValidation?: boolean;
  enablePerformanceOptimization?: boolean;
  enableSchemaValidation?: boolean;
  enableMOValidation?: boolean;
  defaultPriority?: TemplatePriority;
  maxInheritanceDepth?: number;
  cacheConfig?: any;
  validationConfig?: any;
  performanceConfig?: any;
}

/**
 * Template processing result
 */
export interface TemplateProcessingResult {
  template: RTBTemplate;
  inheritanceChain: TemplateInheritanceChain;
  validationResult?: any;
  processingStats: ProcessingStats;
  appliedOptimizations: string[];
  warnings: string[];
  errors: string[];
}

/**
 * MO-aware template context
 */
export interface MOTemplateContext extends TemplateResolutionContext {
  moHierarchy?: MOHierarchy;
  moClasses?: Map<string, MOClass>;
  parameterDefinitions?: Map<string, RTBParameter>;
  constraints?: Map<string, ConstraintSpec[]>;
  ldnPatterns?: Map<string, any>;
}

/**
 * Schema validation result
 */
export interface SchemaValidationResult {
  isValid: boolean;
  errors: Array<{
    parameter: string;
    value: any;
    constraint: string;
    message: string;
    severity: 'error' | 'warning';
  }>;
  warnings: Array<{
    parameter: string;
    message: string;
    suggestion?: string;
  }>;
  validatedParameters: number;
  totalParameters: number;
}

/**
 * Integrated Template System
 *
 * Complete integration of hierarchical template system with existing RTB processor,
 * providing end-to-end template processing with priority inheritance, validation,
 * and performance optimization.
 */
export class IntegratedTemplateSystem {
  private registry: TemplateRegistry;
  private priorityEngine: PriorityTemplateEngine;
  private inheritanceResolver: InheritanceResolver;
  private priorityManager: PriorityManager;
  private templateValidator: TemplateValidator;
  private performanceOptimizer: PerformanceOptimizer;
  private config: Required<IntegratedSystemConfig>;

  // Integration with existing RTB components
  private moHierarchy?: MOHierarchy;
  private parameterDefinitions = new Map<string, RTBParameter>();
  private constraintValidators = new Map<string, ConstraintSpec[]>();

  constructor(config: IntegratedSystemConfig = {}) {
    this.config = {
      enablePrioritySystem: config.enablePrioritySystem ?? true,
      enableValidation: config.enableValidation ?? true,
      enablePerformanceOptimization: config.enablePerformanceOptimization ?? true,
      enableSchemaValidation: config.enableSchemaValidation ?? true,
      enableMOValidation: config.enableMOValidation ?? true,
      defaultPriority: config.defaultPriority ?? TemplatePriority.BASE,
      maxInheritanceDepth: config.maxInheritanceDepth ?? 10,
      cacheConfig: config.cacheConfig || {},
      validationConfig: config.validationConfig || {},
      performanceConfig: config.performanceConfig || {}
    };

    // Initialize components
    this.registry = new TemplateRegistry();
    this.priorityEngine = new PriorityTemplateEngine();
    this.inheritanceResolver = new InheritanceResolver(this.registry);
    this.priorityManager = new PriorityManager(this.registry);
    this.templateValidator = new TemplateValidator(this.registry, this.config.validationConfig);
    this.performanceOptimizer = new PerformanceOptimizer(this.registry, this.config.performanceConfig);

    this.initializeIntegration();
  }

  /**
   * Process template with full system integration
   */
  async processTemplate(
    templateName: string,
    context: MOTemplateContext = {}
  ): Promise<TemplateProcessingResult> {
    const startTime = Date.now();
    const stats: ProcessingStats = {
      xmlProcessingTime: 0,
      hierarchyProcessingTime: 0,
      validationTime: 0,
      totalParameters: 0,
      totalMOClasses: this.moHierarchy?.classes.size || 0,
      totalRelationships: this.moHierarchy?.relationships.size || 0,
      memoryUsage: 0,
      errorCount: 0,
      warningCount: 0
    };

    const result: TemplateProcessingResult = {
      template: {} as RTBTemplate,
      inheritanceChain: {} as TemplateInheritanceChain,
      processingStats: stats,
      appliedOptimizations: [],
      warnings: [],
      errors: []
    };

    try {
      // Step 1: Resolve template inheritance (with performance optimization)
      const resolutionStart = Date.now();
      result.inheritanceChain = await this.performanceOptimizer.resolveTemplateOptimized(
        templateName,
        context
      );
      result.template = result.inheritanceChain.resolvedTemplate;
      stats.hierarchyProcessingTime = Date.now() - resolutionStart;
      result.appliedOptimizations.push('priority_inheritance_resolution');

      // Step 2: Validate template structure and constraints
      if (this.config.enableValidation) {
        const validationStart = Date.now();
        const validationResult = await this.validateTemplateWithSchema(
          result.template,
          templateName,
          context
        );
        result.validationResult = validationResult;
        stats.validationTime = Date.now() - validationStart;
        result.appliedOptimizations.push('template_validation');

        // Update statistics
        stats.errorCount += validationResult.errors?.length || 0;
        stats.warningCount += validationResult.warnings?.length || 0;
        result.errors.push(...(validationResult.errors?.map(e => e.message) || []));
        result.warnings.push(...(validationResult.warnings?.map(w => w.message) || []));
      }

      // Step 3: Validate against MO hierarchy and constraints
      if (this.config.enableMOValidation && context.moHierarchy) {
        const moValidationStart = Date.now();
        const moValidationResult = await this.validateAgainstMOHierarchy(
          result.template,
          context
        );
        stats.hierarchyProcessingTime += Date.now() - moValidationStart;
        result.appliedOptimizations.push('mo_hierarchy_validation');

        // Update statistics
        stats.errorCount += moValidationResult.errors?.length || 0;
        stats.warningCount += moValidationResult.warnings?.length || 0;
        result.errors.push(...(moValidationResult.errors?.map(e => e.message) || []));
        result.warnings.push(...(moValidationResult.warnings?.map(w => w.message) || []));
      }

      // Step 4: Calculate total parameters
      stats.totalParameters = Object.keys(result.template.configuration || {}).length;

      // Step 5: Update memory usage
      stats.memoryUsage = this.calculateMemoryUsage();

      // Add final optimization note
      if (this.config.enablePerformanceOptimization) {
        result.appliedOptimizations.push('performance_optimization');
      }

      return result;
    } catch (error) {
      stats.errorCount++;
      result.errors.push(`Template processing failed: ${error}`);
      throw error;
    } finally {
      // Total processing time would be calculated here
      stats.xmlProcessingTime = Date.now() - startTime;
    }
  }

  /**
   * Register template with full integration
   */
  async registerTemplate(
    name: string,
    template: RTBTemplate,
    priority: TemplatePriorityInfo,
    context: MOTemplateContext = {}
  ): Promise<void> {
    // Validate template before registration
    if (this.config.enableValidation) {
      const validationResult = await this.templateValidator.validateTemplate(
        name,
        template,
        priority
      );

      if (!validationResult.isValid && this.config.validationConfig?.strictMode) {
        throw new Error(
          `Template validation failed: ${validationResult.totalErrors} errors found`
        );
      }
    }

    // Register with priority engine
    this.priorityEngine.registerTemplate(name, template, priority);

    // Register with registry for search and indexing
    await this.registry.registerTemplate(name, template, priority);

    // Update indexes if performance optimization is enabled
    if (this.config.enablePerformanceOptimization) {
      await this.performanceOptimizer.buildIndexes();
    }
  }

  /**
   * Batch process multiple templates
   */
  async batchProcessTemplates(
    templateNames: string[],
    context: MOTemplateContext = {}
  ): Promise<TemplateProcessingResult[]> {
    if (!this.config.enablePerformanceOptimization) {
      // Fallback to sequential processing
      const results: TemplateProcessingResult[] = [];
      for (const name of templateNames) {
        const result = await this.processTemplate(name, context);
        results.push(result);
      }
      return results;
    }

    // Use performance optimizer for batch processing
    const batchResult = await this.performanceOptimizer.batchResolveTemplates(
      templateNames,
      context
    );

    // Process each resolved template
    const results: TemplateProcessingResult[] = [];
    for (let i = 0; i < batchResult.results.length; i++) {
      const inheritanceChain = batchResult.results[i];
      try {
        // Create full processing result for each template
        const result = await this.createProcessingResult(
          inheritanceChain.templateName,
          inheritanceChain,
          context
        );
        results.push(result);
      } catch (error) {
        // Create error result
        results.push({
          template: {} as RTBTemplate,
          inheritanceChain,
          processingStats: {
            xmlProcessingTime: 0,
            hierarchyProcessingTime: 0,
            validationTime: 0,
            totalParameters: 0,
            totalMOClasses: 0,
            totalRelationships: 0,
            memoryUsage: 0,
            errorCount: 1,
            warningCount: 0
          },
          appliedOptimizations: [],
          warnings: [],
          errors: [`Batch processing failed: ${error}`]
        });
      }
    }

    return results;
  }

  /**
   * Set MO hierarchy for validation
   */
  setMOHierarchy(moHierarchy: MOHierarchy): void {
    this.moHierarchy = moHierarchy;

    // Extract parameter definitions and constraints from MO hierarchy
    for (const [className, moClass] of moHierarchy.classes) {
      for (const attribute of moClass.attributes) {
        const paramDef = this.parameterDefinitions.get(attribute);
        if (!paramDef) {
          // This would typically be populated from XML parsing
          // For now, create a basic parameter definition
          this.parameterDefinitions.set(attribute, {
            id: attribute,
            name: attribute,
            vsDataType: 'string',
            type: 'string',
            hierarchy: [className],
            source: 'mo_hierarchy',
            extractedAt: new Date()
          });
        }
      }
    }
  }

  /**
   * Set parameter definitions for schema validation
   */
  setParameterDefinitions(parameters: Map<string, RTBParameter>): void {
    this.parameterDefinitions = new Map(parameters);
  }

  /**
   * Set constraint validators
   */
  setConstraintValidators(validators: Map<string, ConstraintSpec[]>): void {
    this.constraintValidators = new Map(validators);
  }

  /**
   * Search templates with integrated features
   */
  async searchTemplates(filter: any, context: MOTemplateContext = {}): Promise<any> {
    let result;

    if (this.config.enablePerformanceOptimization) {
      result = await this.performanceOptimizer.searchTemplatesOptimized(filter);
    } else {
      result = await this.registry.searchTemplates(filter);
    }

    // Enhance results with additional metadata if available
    if (result.templates && context.moHierarchy) {
      result.templates = result.templates.map((template: any) => ({
        ...template,
        moValidation: this.getMOValidationInfo(template.template, context)
      }));
    }

    return result;
  }

  /**
   * Get system statistics
   */
  getSystemStats(): {
    registry: any;
    priority: any;
    performance: any;
    validation: any;
    integration: {
      moClasses: number;
      parameterDefinitions: number;
      constraintValidators: number;
    };
  } {
    return {
      registry: this.registry.getRegistryStats(),
      priority: {
        templates: this.priorityEngine.getRegisteredTemplates().size,
        cacheStats: this.priorityEngine.getCacheStats()
      },
      performance: this.performanceOptimizer.getPerformanceMetrics(),
      validation: this.templateValidator.getValidationStats(),
      integration: {
        moClasses: this.moHierarchy?.classes.size || 0,
        parameterDefinitions: this.parameterDefinitions.size,
        constraintValidators: this.constraintValidators.size
      }
    };
  }

  // Private helper methods

  private initializeIntegration(): void {
    // Set up event handlers and optimizations
    if (this.config.enablePerformanceOptimization) {
      // Preload common templates
      setTimeout(async () => {
        const commonTemplates = ['base', 'default', 'urban', 'mobility'];
        await this.performanceOptimizer.preloadTemplates(commonTemplates);
      }, 1000);
    }
  }

  private async validateTemplateWithSchema(
    template: RTBTemplate,
    templateName: string,
    context: MOTemplateContext
  ): Promise<any> {
    const validationResults = {
      isValid: true,
      errors: [] as any[],
      warnings: [] as any[],
      validatedParameters: 0,
      totalParameters: Object.keys(template.configuration || {}).length
    };

    if (!template.configuration) return validationResults;

    for (const [paramName, paramValue] of Object.entries(template.configuration)) {
      const paramDef = this.parameterDefinitions.get(paramName);
      if (!paramDef) {
        validationResults.warnings.push({
          parameter: paramName,
          message: `Parameter '${paramName}' not found in schema definitions`
        });
        continue;
      }

      // Validate parameter against schema
      const paramValidation = this.validateParameter(paramDef, paramValue);
      if (!paramValidation.isValid) {
        validationResults.errors.push(...paramValidation.errors);
        validationResults.isValid = false;
      }

      validationResults.warnings.push(...paramValidation.warnings);
      validationResults.validatedParameters++;
    }

    return validationResults;
  }

  private validateParameter(paramDef: RTBParameter, value: any): {
    isValid: boolean;
    errors: any[];
    warnings: any[];
  } {
    const result = {
      isValid: true,
      errors: [] as any[],
      warnings: [] as any[]
    };

    // Type validation
    if (paramDef.type && typeof value !== paramDef.type) {
      result.errors.push({
        parameter: paramDef.name,
        value,
        constraint: 'type',
        message: `Expected type ${paramDef.type}, got ${typeof value}`,
        severity: 'error'
      });
      result.isValid = false;
    }

    // Constraint validation
    const constraints = this.constraintValidators.get(paramDef.name) || paramDef.constraints;
    if (constraints && Array.isArray(constraints)) {
      for (const constraint of constraints) {
        const constraintResult = this.validateConstraint(constraint, value);
        if (!constraintResult.isValid) {
          result.errors.push({
            parameter: paramDef.name,
            value,
            constraint: constraint.type,
            message: constraintResult.message || `Constraint violation: ${constraint.type}`,
            severity: constraint.severity || 'error'
          });
          if (constraint.severity === 'error') {
            result.isValid = false;
          }
        }
      }
    }

    return result;
  }

  private validateConstraint(constraint: ConstraintSpec, value: any): {
    isValid: boolean;
    message?: string;
  } {
    switch (constraint.type) {
      case 'required':
        return {
          isValid: value !== null && value !== undefined && value !== '',
          message: 'Value is required'
        };

      case 'range':
        if (typeof value === 'number' && typeof constraint.value === 'object') {
          const { min, max } = constraint.value;
          if (min !== undefined && value < min) {
            return {
              isValid: false,
              message: `Value ${value} is below minimum ${min}`
            };
          }
          if (max !== undefined && value > max) {
            return {
              isValid: false,
              message: `Value ${value} is above maximum ${max}`
            };
          }
        }
        return { isValid: true };

      case 'enum':
        if (Array.isArray(constraint.value)) {
          return {
            isValid: constraint.value.includes(value),
            message: `Value ${value} not in allowed values: ${constraint.value.join(', ')}`
          };
        }
        return { isValid: true };

      case 'pattern':
        if (typeof value === 'string' && constraint.value instanceof RegExp) {
          return {
            isValid: constraint.value.test(value),
            message: `Value ${value} does not match required pattern`
          };
        }
        return { isValid: true };

      default:
        return { isValid: true };
    }
  }

  private async validateAgainstMOHierarchy(
    template: RTBTemplate,
    context: MOTemplateContext
  ): Promise<any> {
    const validationResults = {
      isValid: true,
      errors: [] as any[],
      warnings: [] as any[]
    };

    if (!context.moHierarchy || !template.configuration) {
      return validationResults;
    }

    // Validate that configuration parameters reference valid MO classes/attributes
    for (const [paramName, paramValue] of Object.entries(template.configuration)) {
      // Check if parameter corresponds to a valid MO attribute
      const isValidMOParam = this.parameterDefinitions.has(paramName);
      if (!isValidMOParam) {
        validationResults.warnings.push({
          parameter: paramName,
          message: `Parameter '${paramName}' does not correspond to a known MO attribute`,
          suggestion: 'Verify parameter name or add to MO hierarchy definitions'
        });
      }
    }

    return validationResults;
  }

  private getMOValidationInfo(template: RTBTemplate, context: MOTemplateContext): any {
    if (!context.moHierarchy || !template.configuration) {
      return null;
    }

    const moInfo = {
      validParameters: 0,
      invalidParameters: 0,
      moClasses: new Set<string>(),
      attributes: new Set<string>()
    };

    for (const paramName of Object.keys(template.configuration)) {
      if (this.parameterDefinitions.has(paramName)) {
        moInfo.validParameters++;
        const paramDef = this.parameterDefinitions.get(paramName)!;
        moInfo.attributes.add(paramName);
        paramDef.hierarchy.forEach(moClass => moInfo.moClasses.add(moClass));
      } else {
        moInfo.invalidParameters++;
      }
    }

    return moInfo;
  }

  private async createProcessingResult(
    templateName: string,
    inheritanceChain: TemplateInheritanceChain,
    context: MOTemplateContext
  ): Promise<TemplateProcessingResult> {
    const processingStats: ProcessingStats = {
      xmlProcessingTime: 0,
      hierarchyProcessingTime: 0,
      validationTime: 0,
      totalParameters: Object.keys(inheritanceChain.resolvedTemplate.configuration || {}).length,
      totalMOClasses: context.moHierarchy?.classes.size || 0,
      totalRelationships: context.moHierarchy?.relationships.size || 0,
      memoryUsage: this.calculateMemoryUsage(),
      errorCount: 0,
      warningCount: 0
    };

    return {
      template: inheritanceChain.resolvedTemplate,
      inheritanceChain,
      processingStats,
      appliedOptimizations: ['batch_processing', 'priority_inheritance'],
      warnings: inheritanceChain.warnings,
      errors: []
    };
  }

  private calculateMemoryUsage(): number {
    // Simplified memory calculation
    return Math.floor(Math.random() * 50000000); // Placeholder
  }

  /**
   * Clear all caches and reset system
   */
  clearSystem(): void {
    this.registry.clear();
    this.priorityEngine.clearAllCaches();
    this.performanceOptimizer.clearOptimizations();
    this.templateValidator.clearCache();
  }
}