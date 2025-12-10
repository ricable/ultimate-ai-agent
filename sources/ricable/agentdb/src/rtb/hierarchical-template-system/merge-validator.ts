/**
 * Merge Validation System for Template Merging
 *
 * Comprehensive validation system for merged templates ensuring data integrity,
* constraint compliance, and structural consistency. Supports advanced validation
* with custom rules and real-time validation feedback.
 */

import { RTBTemplate, TemplateMeta, ConstraintSpec } from '../../types/rtb-types';
import { ValidationResult, ValidationError, ValidationWarning, ValidationStats, MergeContext } from './types';
import { Logger } from '../../utils/logger';

export interface ValidationOptions {
  /** Enable strict validation mode */
  strictMode: boolean;
  /** Validate parameter constraints */
  validateConstraints: boolean;
  /** Validate template structure */
  validateStructure: boolean;
  /** Validate inheritance consistency */
  validateInheritance: boolean;
  /** Validate custom functions */
  validateFunctions: boolean;
  /** Maximum validation depth for nested objects */
  maxValidationDepth: number;
  /** Custom validation rules */
  customValidators: Record<string, ValidationRule>;
  /** Enable performance optimization */
  enableOptimizations: boolean;
}

export interface ValidationRule {
  name: string;
  description: string;
  validator: (value: any, context: ValidationRuleContext) => ValidationResult;
  priority: number;
  applicableTo: string[]; // Parameter patterns
}

export interface ValidationRuleContext {
  parameter: string;
  value: any;
  template: RTBTemplate;
  mergeContext: MergeContext;
  inheritanceDepth: number;
  parentPath?: string;
}

export interface ValidationReport {
  validationResult: ValidationResult;
  validationStats: ValidationStats;
  performanceMetrics: ValidationPerformanceMetrics;
  recommendations: ValidationRecommendation[];
}

export interface ValidationPerformanceMetrics {
  totalValidationTime: number;
  phaseTimes: {
    structureValidation: number;
    constraintValidation: number;
    inheritanceValidation: number;
    functionValidation: number;
    customValidation: number;
  };
  memoryUsage: {
    peak: number;
    average: number;
  };
  parametersValidated: number;
  validationDepth: number;
}

export interface ValidationRecommendation {
  type: 'fix' | 'improvement' | 'optimization';
  parameter: string;
  message: string;
  suggestedAction?: string;
  priority: 'low' | 'medium' | 'high';
}

export class MergeValidator {
  private logger: Logger;
  private options: ValidationOptions;
  private customValidators: Map<string, ValidationRule>;
  private constraintValidators: Map<string, (value: any, constraint: any) => boolean>;

  constructor(options: Partial<ValidationOptions> = {}) {
    this.logger = new Logger('MergeValidator');
    this.options = {
      strictMode: true,
      validateConstraints: true,
      validateStructure: true,
      validateInheritance: true,
      validateFunctions: true,
      maxValidationDepth: 10,
      customValidators: {},
      enableOptimizations: true,
      ...options
    };

    this.customValidators = new Map();
    this.initializeConstraintValidators();
    this.initializeBuiltinValidators();
  }

  /**
   * Validate merged template comprehensively
   */
  async validateMergedTemplate(
    template: RTBTemplate,
    mergeContext: MergeContext
  ): Promise<ValidationResult> {
    const startTime = Date.now();
    this.logger.info(`Starting validation of merged template: ${template.meta?.description || 'Unknown'}`);

    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];
    let totalParameters = 0;

    try {
      // Structure validation
      if (this.options.validateStructure) {
        const structureResult = await this.validateTemplateStructure(template, mergeContext);
        errors.push(...structureResult.errors);
        warnings.push(...structureResult.warnings);
        totalParameters += structureResult.parametersValidated;
      }

      // Constraint validation
      if (this.options.validateConstraints) {
        const constraintResult = await this.validateTemplateConstraints(template, mergeContext);
        errors.push(...constraintResult.errors);
        warnings.push(...constraintResult.warnings);
        totalParameters += constraintResult.parametersValidated;
      }

      // Inheritance validation
      if (this.options.validateInheritance) {
        const inheritanceResult = await this.validateInheritanceConsistency(template, mergeContext);
        errors.push(...inheritanceResult.errors);
        warnings.push(...inheritanceResult.warnings);
      }

      // Function validation
      if (this.options.validateFunctions) {
        const functionResult = await this.validateCustomFunctions(template, mergeContext);
        errors.push(...functionResult.errors);
        warnings.push(...functionResult.warnings);
      }

      // Custom validation
      const customResult = await this.validateWithCustomRules(template, mergeContext);
      errors.push(...customResult.errors);
      warnings.push(...customResult.warnings);

      const validationTime = Date.now() - startTime;

      const stats: ValidationStats = {
        totalParameters,
        errorCount: errors.length,
        warningCount: warnings.length,
        validationTime
      };

      this.logger.info(`Template validation completed`, {
        isValid: errors.length === 0,
        errors: errors.length,
        warnings: warnings.length,
        validationTime
      });

      return {
        isValid: errors.length === 0 || !this.options.strictMode,
        errors,
        warnings,
        stats
      };

    } catch (error) {
      this.logger.error('Template validation failed', { error });

      return {
        isValid: false,
        errors: [{
          message: `Validation process failed: ${error.message}`,
          parameter: '__validation__',
          severity: 'error',
          type: 'schema',
          source: 'MergeValidator',
          suggestion: 'Check template structure and try again'
        }],
        warnings: [],
        stats: {
          totalParameters: 0,
          errorCount: 1,
          warningCount: 0,
          validationTime: Date.now() - startTime
        }
      };
    }
  }

  /**
   * Validate template structure
   */
  private async validateTemplateStructure(
    template: RTBTemplate,
    mergeContext: MergeContext
  ): Promise<{ errors: ValidationError[]; warnings: ValidationWarning[]; parametersValidated: number }> {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];
    let parametersValidated = 0;

    // Validate configuration structure
    if (template.configuration) {
      const configResult = await this.validateObjectStructure(
        template.configuration,
        '',
        template,
        mergeContext,
        0
      );
      errors.push(...configResult.errors);
      warnings.push(...configResult.warnings);
      parametersValidated += configResult.parametersValidated;
    }

    // Validate metadata structure
    if (template.meta) {
      const metaResult = await this.validateMetadataStructure(template.meta, template, mergeContext);
      errors.push(...metaResult.errors);
      warnings.push(...metaResult.warnings);
    }

    return { errors, warnings, parametersValidated };
  }

  /**
   * Validate object structure recursively
   */
  private async validateObjectStructure(
    obj: any,
    path: string,
    template: RTBTemplate,
    mergeContext: MergeContext,
    depth: number
  ): Promise<{ errors: ValidationError[]; warnings: ValidationWarning[]; parametersValidated: number }> {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];
    let parametersValidated = 0;

    if (depth > this.options.maxValidationDepth) {
      warnings.push({
        message: `Maximum validation depth exceeded at path: ${path}`,
        parameter: path,
        severity: 'warning',
        type: 'structure',
        source: 'MergeValidator'
      });
      return { errors, warnings, parametersValidated };
    }

    for (const [key, value] of Object.entries(obj)) {
      const currentPath = path ? `${path}.${key}` : key;
      parametersValidated++;

      if (value === null || value === undefined) {
        if (this.options.strictMode) {
          errors.push({
            message: `Null or undefined value at path: ${currentPath}`,
            parameter: currentPath,
            severity: 'error',
            type: 'structure',
            source: 'MergeValidator',
            suggestion: 'Provide a valid value or remove the parameter'
          });
        } else {
          warnings.push({
            message: `Null or undefined value at path: ${currentPath}`,
            parameter: currentPath,
            severity: 'warning',
            type: 'structure',
            source: 'MergeValidator'
          });
        }
      } else if (typeof value === 'object' && !Array.isArray(value)) {
        const nestedResult = await this.validateObjectStructure(
          value,
          currentPath,
          template,
          mergeContext,
          depth + 1
        );
        errors.push(...nestedResult.errors);
        warnings.push(...nestedResult.warnings);
        parametersValidated += nestedResult.parametersValidated;
      }
    }

    return { errors, warnings, parametersValidated };
  }

  /**
   * Validate metadata structure
   */
  private async validateMetadataStructure(
    meta: TemplateMeta,
    template: RTBTemplate,
    mergeContext: MergeContext
  ): Promise<{ errors: ValidationError[]; warnings: ValidationWarning[] }> {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    // Required fields
    if (!meta.version) {
      warnings.push({
        message: 'Template version is not specified',
        parameter: 'meta.version',
        severity: 'warning',
        type: 'structure',
        source: 'MergeValidator'
      });
    }

    if (!meta.author || meta.author.length === 0) {
      warnings.push({
        message: 'Template author is not specified',
        parameter: 'meta.author',
        severity: 'warning',
        type: 'structure',
        source: 'MergeValidator'
      });
    }

    // Version format validation
    if (meta.version && !this.isValidVersionFormat(meta.version)) {
      errors.push({
        message: `Invalid version format: ${meta.version}`,
        parameter: 'meta.version',
        severity: 'error',
        type: 'structure',
        source: 'MergeValidator',
        suggestion: 'Use semantic versioning (e.g., 1.0.0)'
      });
    }

    // Priority validation
    if (meta.priority !== undefined && (typeof meta.priority !== 'number' || meta.priority < 0)) {
      errors.push({
        message: `Invalid priority value: ${meta.priority}`,
        parameter: 'meta.priority',
        severity: 'error',
        type: 'structure',
        source: 'MergeValidator',
        suggestion: 'Priority must be a non-negative number'
      });
    }

    return { errors, warnings };
  }

  /**
   * Validate template constraints
   */
  private async validateTemplateConstraints(
    template: RTBTemplate,
    mergeContext: MergeContext
  ): Promise<{ errors: ValidationError[]; warnings: ValidationWarning[]; parametersValidated: number }> {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];
    let parametersValidated = 0;

    // This would validate against actual constraints from the XML schema
    // For now, we'll do basic type validation

    if (template.configuration) {
      const constraintResult = await this.validateParameterConstraints(
        template.configuration,
        '',
        template,
        mergeContext,
        0
      );
      errors.push(...constraintResult.errors);
      warnings.push(...constraintResult.warnings);
      parametersValidated += constraintResult.parametersValidated;
    }

    return { errors, warnings, parametersValidated };
  }

  /**
   * Validate parameter constraints recursively
   */
  private async validateParameterConstraints(
    obj: any,
    path: string,
    template: RTBTemplate,
    mergeContext: MergeContext,
    depth: number
  ): Promise<{ errors: ValidationError[]; warnings: ValidationWarning[]; parametersValidated: number }> {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];
    let parametersValidated = 0;

    if (depth > this.options.maxValidationDepth) {
      return { errors, warnings, parametersValidated };
    }

    for (const [key, value] of Object.entries(obj)) {
      const currentPath = path ? `${path}.${key}` : key;
      parametersValidated++;

      // Basic type validation
      if (value !== null && value !== undefined) {
        const typeValidation = this.validateParameterType(value, currentPath);
        if (!typeValidation.isValid) {
          errors.push(...typeValidation.errors);
          warnings.push(...typeValidation.warnings);
        }
      }

      // Recursive validation for objects
      if (typeof value === 'object' && !Array.isArray(value) && value !== null) {
        const nestedResult = await this.validateParameterConstraints(
          value,
          currentPath,
          template,
          mergeContext,
          depth + 1
        );
        errors.push(...nestedResult.errors);
        warnings.push(...nestedResult.warnings);
        parametersValidated += nestedResult.parametersValidated;
      }
    }

    return { errors, warnings, parametersValidated };
  }

  /**
   * Validate parameter type
   */
  private validateParameterType(value: any, path: string): { isValid: boolean; errors: ValidationError[]; warnings: ValidationWarning[] } {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    // Check for circular references
    if (typeof value === 'object' && value !== null) {
      try {
        JSON.stringify(value);
      } catch (error) {
        errors.push({
          message: `Circular reference detected at path: ${path}`,
          parameter: path,
          severity: 'error',
          type: 'structure',
          source: 'MergeValidator',
          suggestion: 'Remove circular reference in the data structure'
        });
      }
    }

    // Check for extremely large values
    if (typeof value === 'string' && value.length > 10000) {
      warnings.push({
        message: `Very large string value at path: ${path} (${value.length} characters)`,
        parameter: path,
        severity: 'warning',
        type: 'structure',
        source: 'MergeValidator'
      });
    }

    // Check for very deep arrays
    if (Array.isArray(value) && value.length > 1000) {
      warnings.push({
        message: `Very large array at path: ${path} (${value.length} items)`,
        parameter: path,
        severity: 'warning',
        type: 'structure',
        source: 'MergeValidator'
      });
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }

  /**
   * Validate inheritance consistency
   */
  private async validateInheritanceConsistency(
    template: RTBTemplate,
    mergeContext: MergeContext
  ): Promise<{ errors: ValidationError[]; warnings: ValidationWarning[] }> {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    if (!template.meta?.inherits_from) {
      return { errors, warnings };
    }

    const inheritsFrom = Array.isArray(template.meta.inherits_from)
      ? template.meta.inherits_from
      : [template.meta.inherits_from];

    // Check for self-inheritance
    if (inheritsFrom.includes(template.meta?.description || template.meta?.source || '')) {
      errors.push({
        message: 'Template cannot inherit from itself',
        parameter: 'meta.inherits_from',
        severity: 'error',
        type: 'circular',
        source: 'MergeValidator',
        suggestion: 'Remove self-reference from inheritance list'
      });
    }

    // Check for circular inheritance (simplified check)
    const visited = new Set<string>();
    const hasCircular = this.checkCircularInheritance(template, inheritsFrom, visited, mergeContext);

    if (hasCircular) {
      errors.push({
        message: 'Circular inheritance detected',
        parameter: 'meta.inherits_from',
        severity: 'error',
        type: 'circular',
        source: 'MergeValidator',
        suggestion: 'Reorganize template hierarchy to remove circular dependencies'
      });
    }

    return { errors, warnings };
  }

  /**
   * Check for circular inheritance
   */
  private checkCircularInheritance(
    template: RTBTemplate,
    parentNames: string[],
    visited: Set<string>,
    mergeContext: MergeContext
  ): boolean {
    const templateName = template.meta?.description || template.meta?.source || '';

    if (visited.has(templateName)) {
      return true;
    }

    visited.add(templateName);

    // This is a simplified check - in a real implementation,
    // we would traverse the actual template hierarchy
    for (const parentName of parentNames) {
      if (visited.has(parentName)) {
        return true;
      }
    }

    return false;
  }

  /**
   * Validate custom functions
   */
  private async validateCustomFunctions(
    template: RTBTemplate,
    mergeContext: MergeContext
  ): Promise<{ errors: ValidationError[]; warnings: ValidationWarning[] }> {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    if (!template.custom) {
      return { errors, warnings };
    }

    for (const func of template.custom) {
      // Validate function name
      if (!func.name || typeof func.name !== 'string') {
        errors.push({
          message: 'Invalid function name',
          parameter: `custom.${func.name}`,
          severity: 'error',
          type: 'structure',
          source: 'MergeValidator'
        });
        continue;
      }

      // Validate function arguments
      if (!Array.isArray(func.args)) {
        errors.push({
          message: `Function arguments must be an array: ${func.name}`,
          parameter: `custom.${func.name}.args`,
          severity: 'error',
          type: 'structure',
          source: 'MergeValidator'
        });
      }

      // Validate function body
      if (!Array.isArray(func.body) || func.body.length === 0) {
        errors.push({
          message: `Function body must be a non-empty array: ${func.name}`,
          parameter: `custom.${func.name}.body`,
          severity: 'error',
          type: 'structure',
          source: 'MergeValidator'
        });
      }
    }

    return { errors, warnings };
  }

  /**
   * Validate with custom rules
   */
  private async validateWithCustomRules(
    template: RTBTemplate,
    mergeContext: MergeContext
  ): Promise<{ errors: ValidationError[]; warnings: ValidationWarning[] }> {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    for (const [pattern, rule] of this.customValidators.entries()) {
      try {
        const matchingPaths = this.findMatchingPaths(template, pattern);

        for (const path of matchingPaths) {
          const value = this.getValueAtPath(template, path);
          const context: ValidationRuleContext = {
            parameter: path,
            value,
            template,
            mergeContext,
            inheritanceDepth: path.split('.').length
          };

          const result = rule.validator(value, context);
          errors.push(...result.errors);
          warnings.push(...result.warnings);
        }
      } catch (error) {
        this.logger.error(`Custom validator failed: ${rule.name}`, { error });
        errors.push({
          message: `Custom validation rule failed: ${rule.name}`,
          parameter: pattern,
          severity: 'error',
          type: 'custom',
          source: 'MergeValidator'
        });
      }
    }

    return { errors, warnings };
  }

  /**
   * Find paths matching a pattern
   */
  private findMatchingPaths(template: RTBTemplate, pattern: string): string[] {
    const paths: string[] = [];

    const traverse = (obj: any, path: string = '') => {
      for (const [key, value] of Object.entries(obj)) {
        const currentPath = path ? `${path}.${key}` : key;

        if (currentPath.includes(pattern) || pattern.includes(currentPath)) {
          paths.push(currentPath);
        }

        if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
          traverse(value, currentPath);
        }
      }
    };

    if (template.configuration) {
      traverse(template.configuration);
    }

    return paths;
  }

  /**
   * Get value at a specific path
   */
  private getValueAtPath(template: RTBTemplate, path: string): any {
    const pathParts = path.split('.');
    let current = template.configuration;

    for (const part of pathParts) {
      if (current && typeof current === 'object' && part in current) {
        current = current[part];
      } else {
        return undefined;
      }
    }

    return current;
  }

  /**
   * Check if version format is valid
   */
  private isValidVersionFormat(version: string): boolean {
    // Semantic versioning pattern
    const semverPattern = /^\d+\.\d+\.\d+(-[a-zA-Z0-9\-\.]+)?(\+[a-zA-Z0-9\-\.]+)?$/;
    return semverPattern.test(version);
  }

  /**
   * Initialize constraint validators
   */
  private initializeConstraintValidators(): void {
    this.constraintValidators = new Map();

    // Range validator
    this.constraintValidators.set('range', (value: any, constraint: any) => {
      if (typeof value !== 'number') return false;
      const { min, max } = constraint;
      return value >= min && value <= max;
    });

    // Enum validator
    this.constraintValidators.set('enum', (value: any, constraint: any) => {
      return constraint.values.includes(value);
    });

    // Pattern validator
    this.constraintValidators.set('pattern', (value: any, constraint: any) => {
      if (typeof value !== 'string') return false;
      const regex = new RegExp(constraint.pattern);
      return regex.test(value);
    });

    // Length validator
    this.constraintValidators.set('length', (value: any, constraint: any) => {
      const length = Array.isArray(value) ? value.length : (typeof value === 'string' ? value.length : 0);
      const { min, max } = constraint;
      return length >= (min || 0) && length <= (max || Infinity);
    });
  }

  /**
   * Initialize built-in validators
   */
  private initializeBuiltinValidators(): void {
    // Add built-in validation rules here
  }

  /**
   * Register custom validation rule
   */
  public registerCustomValidator(rule: ValidationRule): void {
    this.customValidators.set(rule.name, rule);
    this.logger.info(`Registered custom validator: ${rule.name}`);
  }

  /**
   * Unregister custom validation rule
   */
  public unregisterCustomValidator(name: string): void {
    this.customValidators.delete(name);
    this.logger.info(`Unregistered custom validator: ${name}`);
  }

  /**
   * Get registered validators
   */
  public getRegisteredValidators(): ValidationRule[] {
    return Array.from(this.customValidators.values());
  }

  /**
   * Clear all custom validators
   */
  public clearCustomValidators(): void {
    this.customValidators.clear();
    this.logger.info('All custom validators cleared');
  }
}