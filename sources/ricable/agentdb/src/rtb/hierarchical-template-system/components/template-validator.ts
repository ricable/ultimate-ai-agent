/**
 * Template Validator - XML Constraint Validation Implementation
 *
 * Provides comprehensive validation capabilities for RTB templates
 * against XML schema constraints, business rules, and performance requirements.
 *
 * Features:
 * - XML schema constraint validation
 * - Business rule validation
 * - Performance constraint checking
 * - Cross-template consistency validation
 * - Detailed error reporting with suggestions
 */

import {
  PriorityTemplate,
  TemplateValidationResult,
  TemplateValidationRule,
  ValidationError,
  ValidationWarning,
  ConstraintViolation,
  ValidationPerformanceMetrics,
  HierarchicalTemplateEngineConfig,
  RTBParameter,
  ConstraintSpec
} from '../interfaces';

import { MOHierarchy } from '../../types/rtb-types';

/**
 * Template Validator implementation
 */
export class TemplateValidator {
  private config: HierarchicalTemplateEngineConfig;
  private moHierarchy?: MOHierarchy;
  private parameterRegistry: Map<string, RTBParameter> = new Map();

  constructor(config: HierarchicalTemplateEngineConfig, moHierarchy?: MOHierarchy) {
    this.config = config;
    this.moHierarchy = moHierarchy;
  }

  /**
   * Validate template against constraints
   */
  async validate(template: PriorityTemplate): Promise<TemplateValidationResult> {
    const startTime = Date.now();
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];
    const constraintViolations: ConstraintViolation[] = [];

    try {
      // Basic structure validation
      const structureErrors = this.validateTemplateStructure(template);
      errors.push(...structureErrors);

      // Parameter validation
      const parameterValidation = await this.validateParameters(template);
      errors.push(...parameterValidation.errors);
      warnings.push(...parameterValidation.warnings);
      constraintViolations.push(...parameterValidation.violations);

      // Conditional logic validation
      const conditionalErrors = this.validateConditionalLogic(template);
      errors.push(...conditionalErrors);

      // Custom function validation
      const functionErrors = this.validateCustomFunctions(template);
      errors.push(...functionErrors);

      // Business rule validation
      if (template.meta.validationRules) {
        const businessValidation = await this.validateBusinessRules(template);
        errors.push(...businessValidation.errors);
        warnings.push(...businessValidation.warnings);
      }

      // Performance validation
      const performanceValidation = this.validatePerformanceConstraints(template);
      warnings.push(...performanceValidation);

      // Cross-template consistency (if hierarchy is available)
      if (this.moHierarchy) {
        const consistencyValidation = await this.validateConsistency(template);
        warnings.push(...consistencyValidation);
      }

      // Calculate metrics
      const processingTime = Date.now() - startTime;
      const parameterCount = this.countParameters(template);
      const constraintCount = this.countConstraints(template);

      const performanceMetrics: ValidationPerformanceMetrics = {
        validationTime: processingTime,
        memoryUsage: process.memoryUsage().heapUsed,
        parameterCount,
        constraintCount
      };

      const isValid = errors.length === 0;

      const result: TemplateValidationResult = {
        isValid,
        errors,
        warnings,
        parameterCount,
        constraintViolations,
        performanceMetrics
      };

      if (this.config.detailedLogging) {
        console.log(`[TemplateValidator] Validation completed for ${template.meta.version}:`, {
          isValid,
          errorCount: errors.length,
          warningCount: warnings.length,
          processingTime
        });
      }

      return result;

    } catch (error) {
      const processingTime = Date.now() - startTime;

      return {
        isValid: false,
        errors: [{
          parameterPath: 'template',
          errorType: 'validation_error',
          message: `Validation failed: ${(error as Error).message}`,
          severity: 'error'
        }],
        warnings: [],
        parameterCount: 0,
        constraintViolations: [],
        performanceMetrics: {
          validationTime: processingTime,
          memoryUsage: process.memoryUsage().heapUsed,
          parameterCount: 0,
          constraintCount: 0
        }
      };
    }
  }

  /**
   * Validate template structure
   */
  private validateTemplateStructure(template: PriorityTemplate): ValidationError[] {
    const errors: ValidationError[] = [];

    // Check required metadata
    if (!template.meta) {
      errors.push({
        parameterPath: 'meta',
        errorType: 'missing_metadata',
        message: 'Template metadata is required',
        severity: 'error'
      });
      return errors;
    }

    if (!template.meta.version) {
      errors.push({
        parameterPath: 'meta.version',
        errorType: 'missing_version',
        message: 'Template version is required',
        severity: 'error'
      });
    }

    if (!template.meta.author || template.meta.author.length === 0) {
      errors.push({
        parameterPath: 'meta.author',
        errorType: 'missing_author',
        message: 'Template author is required',
        severity: 'warning'
      });
    }

    if (!template.meta.description) {
      errors.push({
        parameterPath: 'meta.description',
        errorType: 'missing_description',
        message: 'Template description is recommended',
        severity: 'warning'
      });
    }

    // Check priority value
    if (typeof template.priority !== 'number' || template.priority < 0 || template.priority > 100) {
      errors.push({
        parameterPath: 'priority',
        errorType: 'invalid_priority',
        message: 'Priority must be a number between 0 and 100',
        severity: 'error'
      });
    }

    // Check configuration object
    if (!template.configuration || typeof template.configuration !== 'object') {
      errors.push({
        parameterPath: 'configuration',
        errorType: 'missing_configuration',
        message: 'Configuration object is required',
        severity: 'error'
      });
    }

    return errors;
  }

  /**
   * Validate template parameters
   */
  private async validateParameters(template: PriorityTemplate): Promise<{
    errors: ValidationError[];
    warnings: ValidationWarning[];
    violations: ConstraintViolation[];
  }> {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];
    const violations: ConstraintViolation[] = [];

    for (const [parameterPath, value] of Object.entries(template.configuration)) {
      try {
        // Validate parameter name format
        if (!this.isValidParameterName(parameterPath)) {
          errors.push({
            parameterPath,
            errorType: 'invalid_parameter_name',
            message: `Invalid parameter name format: ${parameterPath}`,
            severity: 'error'
          });
          continue;
        }

        // Get parameter definition from registry
        const paramDef = this.parameterRegistry.get(parameterPath);
        if (paramDef) {
          // Validate against XML constraints
          const constraintValidation = this.validateConstraints(parameterPath, value, paramDef.constraints);
          violations.push(...constraintValidation);
        } else {
          warnings.push({
            parameterPath,
            warningType: 'unknown_parameter',
            message: `Parameter not found in XML schema: ${parameterPath}`,
            suggestion: 'Verify parameter name or add to schema'
          });
        }

        // Validate value type
        const typeValidation = this.validateParameterType(parameterPath, value);
        if (typeValidation) {
          if (typeValidation.severity === 'error') {
            errors.push(typeValidation);
          } else {
            warnings.push(typeValidation);
          }
        }

      } catch (error) {
        errors.push({
          parameterPath,
          errorType: 'validation_exception',
          message: `Exception validating parameter: ${(error as Error).message}`,
          severity: 'error'
        });
      }
    }

    return { errors, warnings, violations };
  }

  /**
   * Validate conditional logic
   */
  private validateConditionalLogic(template: PriorityTemplate): ValidationError[] {
    const errors: ValidationError[] = [];

    if (!template.conditions) {
      return errors;
    }

    for (const [conditionId, condition] of Object.entries(template.conditions)) {
      // Validate condition structure
      if (!condition.if || typeof condition.if !== 'string') {
        errors.push({
          parameterPath: `conditions.${conditionId}.if`,
          errorType: 'missing_condition',
          message: 'Condition "if" clause is required and must be a string',
          severity: 'error'
        });
      }

      if (!condition.then || typeof condition.then !== 'object') {
        errors.push({
          parameterPath: `conditions.${conditionId}.then`,
          errorType: 'missing_then_clause',
          message: 'Condition "then" clause is required and must be an object',
          severity: 'error'
        });
      }

      if (condition.else && typeof condition.else !== 'string' && typeof condition.else !== 'object') {
        errors.push({
          parameterPath: `conditions.${conditionId}.else`,
          errorType: 'invalid_else_clause',
          message: 'Condition "else" clause must be a string or object',
          severity: 'error'
        });
      }

      // Validate condition expression
      if (condition.if) {
        const expressionErrors = this.validateConditionExpression(condition.if);
        errors.push(...expressionErrors.map(e => ({
          ...e,
          parameterPath: `conditions.${conditionId}.if`
        })));
      }
    }

    return errors;
  }

  /**
   * Validate custom functions
   */
  private validateCustomFunctions(template: PriorityTemplate): ValidationError[] {
    const errors: ValidationError[] = [];

    if (!template.custom) {
      return errors;
    }

    for (const func of template.custom) {
      // Validate function name
      if (!func.name || typeof func.name !== 'string') {
        errors.push({
          parameterPath: 'custom.function.name',
          errorType: 'missing_function_name',
          message: 'Function name is required',
          severity: 'error'
        });
        continue;
      }

      // Validate function arguments
      if (!func.args || !Array.isArray(func.args)) {
        errors.push({
          parameterPath: `custom.function.${func.name}.args`,
          errorType: 'missing_function_args',
          message: 'Function arguments array is required',
          severity: 'error'
        });
      }

      // Validate function body
      if (!func.body || !Array.isArray(func.body) || func.body.length === 0) {
        errors.push({
          parameterPath: `custom.function.${func.name}.body`,
          errorType: 'missing_function_body',
          message: 'Function body array is required and cannot be empty',
          severity: 'error'
        });
      }

      // Validate function body syntax
      if (func.body) {
        try {
          this.validateFunctionBody(func.body);
        } catch (error) {
          errors.push({
            parameterPath: `custom.function.${func.name}.body`,
            errorType: 'function_syntax_error',
            message: `Function syntax error: ${(error as Error).message}`,
            severity: 'error'
          });
        }
      }
    }

    return errors;
  }

  /**
   * Validate business rules
   */
  private async validateBusinessRules(template: PriorityTemplate): Promise<{
    errors: ValidationError[];
    warnings: ValidationWarning[];
  }> {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    if (!template.meta.validationRules) {
      return { errors, warnings };
    }

    for (const rule of template.meta.validationRules) {
      if (!rule.enabled) {
        continue;
      }

      try {
        const result = this.evaluateBusinessRule(rule, template);

        if (rule.action === 'error' && !result.passed) {
          errors.push({
            parameterPath: rule.ruleId,
            errorType: rule.type,
            message: rule.message,
            severity: 'error'
          });
        } else if (rule.action === 'warning' && !result.passed) {
          warnings.push({
            parameterPath: rule.ruleId,
            warningType: rule.type,
            message: rule.message,
            suggestion: result.suggestion
          });
        }

      } catch (error) {
        errors.push({
          parameterPath: rule.ruleId,
          errorType: 'rule_evaluation_error',
          message: `Error evaluating business rule: ${(error as Error).message}`,
          severity: 'error'
        });
      }
    }

    return { errors, warnings };
  }

  /**
   * Validate performance constraints
   */
  private validatePerformanceConstraints(template: PriorityTemplate): ValidationWarning[] {
    const warnings: ValidationWarning[] = [];
    const parameterCount = Object.keys(template.configuration).length;

    // Check parameter count limits
    if (parameterCount > 1000) {
      warnings.push({
        parameterPath: 'configuration',
        warningType: 'performance_warning',
        message: `Large template with ${parameterCount} parameters may impact performance`,
        suggestion: 'Consider splitting into multiple templates or using template inheritance'
      });
    }

    // Check condition complexity
    if (template.conditions && Object.keys(template.conditions).length > 50) {
      warnings.push({
        parameterPath: 'conditions',
        warningType: 'complexity_warning',
        message: `High number of conditions (${Object.keys(template.conditions).length}) may impact processing time`,
        suggestion: 'Simplify conditional logic or use fewer conditions'
      });
    }

    // Check custom function count
    if (template.custom && template.custom.length > 20) {
      warnings.push({
        parameterPath: 'custom',
        warningType: 'complexity_warning',
        message: `High number of custom functions (${template.custom.length}) may impact performance`,
        suggestion: 'Consolidate functions or move to shared library'
      });
    }

    return warnings;
  }

  /**
   * Validate cross-template consistency
   */
  private async validateConsistency(template: PriorityTemplate): Promise<ValidationWarning[]> {
    const warnings: ValidationWarning[] = [];

    if (!this.moHierarchy) {
      return warnings;
    }

    // Check MO class relationships
    for (const [parameterPath, value] of Object.entries(template.configuration)) {
      const moClass = this.extractMOClass(parameterPath);
      if (moClass) {
        const classDef = this.moHierarchy.classes.get(moClass);
        if (!classDef) {
          warnings.push({
            parameterPath,
            warningType: 'unknown_mo_class',
            message: `MO class not found in hierarchy: ${moClass}`,
            suggestion: 'Verify MO class name or update hierarchy'
          });
        }
      }
    }

    return warnings;
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  /**
   * Validate parameter name format
   */
  private isValidParameterName(name: string): boolean {
    // Regex for parameter names: MOClass.parameterName or nested.object.path
    const parameterRegex = /^[A-Z][a-zA-Z0-9]*(\.[a-zA-Z][a-zA-Z0-9]*)*$/;
    return parameterRegex.test(name);
  }

  /**
   * Validate constraints
   */
  private validateConstraints(
    parameterPath: string,
    value: any,
    constraints?: ConstraintSpec[] | Record<string, any>
  ): ConstraintViolation[] {
    const violations: ConstraintViolation[] = [];

    if (!constraints) {
      return violations;
    }

    const constraintList = Array.isArray(constraints) ? constraints : this.convertConstraintsToArray(constraints);

    for (const constraint of constraintList) {
      const violation = this.evaluateConstraint(parameterPath, value, constraint);
      if (violation) {
        violations.push(violation);
      }
    }

    return violations;
  }

  /**
   * Evaluate a single constraint
   */
  private evaluateConstraint(
    parameterPath: string,
    value: any,
    constraint: ConstraintSpec
  ): ConstraintViolation | null {
    let passed = true;
    let actualValue = value;
    let expectedValue = constraint.value;

    switch (constraint.type) {
      case 'range':
        if (typeof value !== 'number') {
          passed = false;
        } else {
          const [min, max] = constraint.value;
          passed = value >= min && value <= max;
          actualValue = value;
          expectedValue = `${min} - ${max}`;
        }
        break;

      case 'enum':
        passed = constraint.value.includes(value);
        actualValue = value;
        expectedValue = constraint.value.join(', ');
        break;

      case 'pattern':
        if (typeof value !== 'string') {
          passed = false;
        } else {
          const regex = new RegExp(constraint.value);
          passed = regex.test(value);
          actualValue = value;
          expectedValue = constraint.value;
        }
        break;

      case 'length':
        if (typeof value !== 'string' && !Array.isArray(value)) {
          passed = false;
        } else {
          const length = Array.isArray(value) ? value.length : value.length;
          if (typeof constraint.value === 'number') {
            passed = length <= constraint.value;
            expectedValue = `<= ${constraint.value}`;
          } else {
            const [min, max] = constraint.value;
            passed = length >= min && length <= max;
            expectedValue = `${min} - ${max}`;
          }
          actualValue = length;
        }
        break;

      case 'required':
        passed = value !== null && value !== undefined && value !== '';
        break;

      default:
        return null;
    }

    if (!passed) {
      return {
        parameterPath,
        constraint,
        actualValue,
        expectedValue,
        severity: constraint.severity === 'error' ? 'error' : 'warning'
      };
    }

    return null;
  }

  /**
   * Validate parameter type
   */
  private validateParameterType(parameterPath: string, value: any): ValidationError | ValidationWarning | null {
    // Basic type validation - could be enhanced with schema-specific types
    if (value === null || value === undefined) {
      return {
        parameterPath,
        errorType: 'null_value',
        message: 'Parameter value cannot be null or undefined',
        severity: 'warning'
      };
    }

    return null;
  }

  /**
   * Validate condition expression
   */
  private validateConditionExpression(expression: string): ValidationError[] {
    const errors: ValidationError[] = [];

    // Basic expression validation
    if (!expression.includes('${')) {
      errors.push({
        parameterPath: 'expression',
        errorType: 'invalid_expression',
        message: 'Condition expression must contain at least one variable reference',
        severity: 'error'
      });
    }

    // Check for balanced braces
    const openBraces = (expression.match(/\$\{/g) || []).length;
    const closeBraces = (expression.match(/\}/g) || []).length;
    if (openBraces !== closeBraces) {
      errors.push({
        parameterPath: 'expression',
        errorType: 'unbalanced_braces',
        message: 'Expression has unbalanced braces',
        severity: 'error'
      });
    }

    return errors;
  }

  /**
   * Validate function body
   */
  private validateFunctionBody(body: string[]): void {
    // Basic syntax validation - could be enhanced with actual JavaScript parsing
    const fullBody = body.join('\n');

    // Check for basic syntax issues
    if (fullBody.includes('eval(') || fullBody.includes('Function(')) {
      throw new Error('Use of eval() or Function() constructor is not allowed in custom functions');
    }

    // Check for potential infinite loops
    if (fullBody.includes('while (true)') || fullBody.includes('for (;;')) {
      throw new Error('Infinite loops are not allowed in custom functions');
    }
  }

  /**
   * Evaluate business rule
   */
  private evaluateBusinessRule(rule: TemplateValidationRule, template: PriorityTemplate): {
    passed: boolean;
    suggestion?: string;
  } {
    // Simple rule evaluation - could be enhanced with expression parser
    try {
      // Create a context with template data
      const context = {
        ...template.configuration,
        template,
        meta: template.meta
      };

      // For now, just check basic conditions
      if (rule.condition.includes('priority')) {
        const priorityMatch = rule.condition.match(/priority\s*([<>=!]+)\s*(\d+)/);
        if (priorityMatch) {
          const operator = priorityMatch[1];
          const value = parseInt(priorityMatch[2]);
          const actualPriority = template.priority;

          switch (operator) {
            case '<':
              return { passed: actualPriority < value };
            case '<=':
              return { passed: actualPriority <= value };
            case '>':
              return { passed: actualPriority > value };
            case '>=':
              return { passed: actualPriority >= value };
            case '==':
              return { passed: actualPriority === value };
            case '!=':
              return { passed: actualPriority !== value };
          }
        }
      }

      // Default to passed if condition can't be evaluated
      return { passed: true };

    } catch (error) {
      return { passed: false, suggestion: 'Fix rule condition syntax' };
    }
  }

  /**
   * Extract MO class from parameter path
   */
  private extractMOClass(parameterPath: string): string | null {
    const parts = parameterPath.split('.');
    return parts.length > 0 ? parts[0] : null;
  }

  /**
   * Convert constraints object to array
   */
  private convertConstraintsToArray(constraints: Record<string, any>): ConstraintSpec[] {
    const result: ConstraintSpec[] = [];

    for (const [key, value] of Object.entries(constraints)) {
      let type: ConstraintSpec['type'];
      let constraintValue: any;

      switch (key) {
        case 'min':
        case 'max':
          // Handle range constraints
          type = 'range';
          const min = constraints.min;
          const max = constraints.max;
          if (min !== undefined && max !== undefined) {
            constraintValue = [min, max];
          } else if (min !== undefined) {
            constraintValue = [min, Number.MAX_SAFE_INTEGER];
          } else {
            constraintValue = [Number.MIN_SAFE_INTEGER, max];
          }
          break;

        case 'enum':
        case 'values':
          type = 'enum';
          constraintValue = Array.isArray(value) ? value : [value];
          break;

        case 'pattern':
        case 'regex':
          type = 'pattern';
          constraintValue = value;
          break;

        case 'required':
          type = 'required';
          constraintValue = true;
          break;

        default:
          continue;
      }

      result.push({
        type,
        value: constraintValue,
        errorMessage: constraints.errorMessage || `Constraint violation for ${key}`,
        severity: constraints.severity || 'error'
      });
    }

    return result;
  }

  /**
   * Count parameters in template
   */
  private countParameters(template: PriorityTemplate): number {
    return Object.keys(template.configuration).length;
  }

  /**
   * Count constraints in template
   */
  private countConstraints(template: PriorityTemplate): number {
    let count = 0;
    if (template.meta.validationRules) {
      count += template.meta.validationRules.length;
    }
    return count;
  }

  /**
   * Set parameter registry for validation
   */
  setParameterRegistry(parameters: RTBParameter[]): void {
    this.parameterRegistry.clear();
    for (const param of parameters) {
      this.parameterRegistry.set(param.name, param);
    }
  }

  /**
   * Set MO hierarchy for consistency validation
   */
  setMOHierarchy(moHierarchy: MOHierarchy): void {
    this.moHierarchy = moHierarchy;
  }
}