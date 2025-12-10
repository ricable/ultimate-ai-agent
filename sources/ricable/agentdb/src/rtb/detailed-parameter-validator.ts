import { RTBParameter, MOHierarchy, LDNHierarchy, ReservedByHierarchy, ConstraintSpec } from '../types/rtb-types';

export interface ValidationRule {
  id: string;
  name: string;
  description: string;
  severity: 'error' | 'warning' | 'info';
  validate: (param: RTBParameter, context?: ValidationContext) => ValidationResult;
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  info: string[];
  suggestions: string[];
}

export interface ValidationContext {
  moHierarchy?: MOHierarchy;
  ldnHierarchy?: LDNHierarchy;
  reservedByHierarchy?: ReservedByHierarchy;
  timestamp?: Date;
  environment?: 'production' | 'staging' | 'development';
}

export interface ValidationReport {
  totalParameters: number;
  validParameters: number;
  invalidParameters: number;
  validationErrors: number;
  validationWarnings: number;
  validationSuggestions: number;
  processingTime: number;
  detailedResults: Map<string, ValidationResult>;
  summary: {
    bySeverity: Record<string, number>;
    byType: Record<string, number>;
    topErrors: Array<{ error: string; count: number }>;
  };
}

export class DetailedParameterValidator {
  private rules: Map<string, ValidationRule> = new Map();
  private context: ValidationContext = {};

  constructor() {
    this.initializeDefaultRules();
  }

  setContext(context: ValidationContext): void {
    this.context = { ...this.context, ...context };
  }

  addRule(rule: ValidationRule): void {
    this.rules.set(rule.id, rule);
  }

  removeRule(ruleId: string): void {
    this.rules.delete(ruleId);
  }

  validateParameters(parameters: RTBParameter[]): ValidationReport {
    const startTime = Date.now();
    const detailedResults = new Map<string, ValidationResult>();
    let validationErrors = 0;
    let validationWarnings = 0;
    let validationSuggestions = 0;

    console.log('[DetailedParameterValidator] Starting parameter validation...');

    // Validate each parameter
    parameters.forEach(param => {
      const result = this.validateParameter(param);
      detailedResults.set(param.name, result);

      if (!result.valid) {
        validationErrors++;
      }

      validationWarnings += result.warnings.length;
      validationSuggestions += result.suggestions.length;
    });

    // Perform cross-parameter validations
    const crossValidationResults = this.performCrossParameterValidations(parameters);
    this.mergeCrossValidationResults(detailedResults, crossValidationResults);

    const processingTime = Date.now() - startTime;

    const report: ValidationReport = {
      totalParameters: parameters.length,
      validParameters: parameters.length - validationErrors,
      invalidParameters: validationErrors,
      validationErrors,
      validationWarnings,
      validationSuggestions,
      processingTime,
      detailedResults,
      summary: this.generateSummary(detailedResults)
    };

    console.log(`[DetailedParameterValidator] Validation completed in ${processingTime}ms`);
    return report;
  }

  private validateParameter(param: RTBParameter): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];
    const info: string[] = [];
    const suggestions: string[] = [];

    // Apply all validation rules
    for (const [ruleId, rule] of this.rules) {
      try {
        const result = rule.validate(param, this.context);

        if (!result.valid) {
          errors.push(...result.errors);
        }

        warnings.push(...result.warnings);
        info.push(...result.info);
        suggestions.push(...result.suggestions);
      } catch (error) {
        console.error(`[DetailedParameterValidator] Rule ${ruleId} failed for parameter ${param.name}:`, error);
        errors.push(`Validation rule ${ruleId} failed: ${error instanceof Error ? error.message : String(error)}`);
      }
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings,
      info,
      suggestions
    };
  }

  private performCrossParameterValidations(parameters: RTBParameter[]): Map<string, ValidationResult> {
    const crossResults = new Map<string, ValidationResult>();

    // Hierarchical dependencies
    this.validateHierarchicalDependencies(parameters, crossResults);

    // ReservedBy relationships
    if (this.context.reservedByHierarchy) {
      this.validateReservedByRelationships(parameters, crossResults);
    }

    // LDN navigation conflicts
    if (this.context.ldnHierarchy) {
      this.validateLDNNavigationConflicts(parameters, crossResults);
    }

    // Parameter naming conflicts
    this.validateNamingConflicts(parameters, crossResults);

    // Type compatibility
    this.validateTypeCompatibility(parameters, crossResults);

    return crossResults;
  }

  private validateHierarchicalDependencies(parameters: RTBParameter[], crossResults: Map<string, ValidationResult>): void {
    const paramByName = new Map(parameters.map(p => [p.name, p]));

    parameters.forEach(param => {
      if (!param.hierarchy || param.hierarchy.length === 0) {
        return;
      }

      // Check if parent parameters exist
      for (let i = 0; i < param.hierarchy.length - 1; i++) {
        const parentName = param.hierarchy[i];
        const parentParam = paramByName.get(parentName);

        if (!parentParam) {
          this.addToCrossResult(crossResults, param.name, {
            valid: false,
            errors: [`Parent parameter '${parentName}' not found`],
            warnings: [],
            info: [],
            suggestions: [`Consider creating parameter '${parentName}'`]
          });
        }
      }

      // Check if child parameters are properly defined
      const potentialChildren = parameters.filter(p =>
        p.hierarchy && p.hierarchy.length > param.hierarchy!.length &&
        p.hierarchy.slice(0, param.hierarchy!.length).join('.') === param.hierarchy!.join('.')
      );

      if (potentialChildren.length > 0) {
        this.addToCrossResult(crossResults, param.name, {
          valid: true,
          errors: [],
          warnings: [`Has ${potentialChildren.length} child parameters`],
          info: [],
          suggestions: ['Consider defining child parameter constraints']
        });
      }
    });
  }

  private validateReservedByRelationships(parameters: RTBParameter[], crossResults: Map<string, ValidationResult>): void {
    if (!this.context.reservedByHierarchy) return;

    const paramByName = new Map(parameters.map(p => [p.name, p]));

    for (const [relationshipId, relationship] of this.context.reservedByHierarchy.relationships) {
      const sourceParam = paramByName.get(relationship.sourceClass);
      const targetParam = paramByName.get(relationship.targetClass);

      if (!sourceParam) {
        continue; // Source parameter not in current set
      }

      if (!targetParam) {
        this.addToCrossResult(crossResults, sourceParam.name, {
          valid: false,
          errors: [`Reserved parameter '${relationship.targetClass}' not found`],
          warnings: [],
          info: [],
          suggestions: [`Create parameter '${relationship.targetClass}' to satisfy reservation`]
        });
      } else {
        // Validate cardinality constraints
        const constraints = this.context.reservedByHierarchy.constraintValidation.get(relationshipId);
        if (constraints) {
          this.validateCardinalityConstraints(sourceParam, targetParam, constraints, crossResults);
        }
      }
    }
  }

  private validateCardinalityConstraints(
    sourceParam: RTBParameter,
    targetParam: RTBParameter,
    constraints: any,
    crossResults: Map<string, ValidationResult>
  ): void {
    // Implement actual cardinality validation logic
    // This would check if the parameters meet the cardinality requirements

    const errors: string[] = [];
    const warnings: string[] = [];

    // Example validation logic (would be more sophisticated in reality)
    if (constraints.validatorType === 'range') {
      const minRule = constraints.rules.find((r: any) => r.type === 'min');
      const maxRule = constraints.rules.find((r: any) => r.type === 'max');

      if (minRule && maxRule) {
        // Check if parameters meet range requirements
        const sourceCount = this.getParameterCount(sourceParam);
        const targetCount = this.getParameterCount(targetParam);

        if (sourceCount < minRule.value) {
          errors.push(`Source parameter count (${sourceCount}) is below minimum (${minRule.value})`);
        }

        if (targetCount > maxRule.value && maxRule.value !== Infinity) {
          errors.push(`Target parameter count (${targetCount}) exceeds maximum (${maxRule.value})`);
        }
      }
    }

    if (errors.length > 0) {
      this.addToCrossResult(crossResults, sourceParam.name, {
        valid: false,
        errors,
        warnings,
        info: [],
        suggestions: ['Adjust parameter values to meet cardinality constraints']
      });
    }
  }

  private validateLDNNavigationConflicts(parameters: RTBParameter[], crossResults: Map<string, ValidationResult>): void {
    if (!this.context.ldnHierarchy) return;

    // Find parameters with conflicting LDN paths
    const conflicts = new Map<string, string[]>();

    parameters.forEach(param => {
      if (!param.navigationPaths) return;

      param.navigationPaths.forEach(path => {
        if (!conflicts.has(path)) {
          conflicts.set(path, []);
        }
        conflicts.get(path)!.push(param.name);
      });
    });

    // Check for conflicts
    for (const [path, paramNames] of conflicts) {
      if (paramNames.length > 1) {
        paramNames.forEach(paramName => {
          this.addToCrossResult(crossResults, paramName, {
            valid: false,
            errors: [`LDN path '${path}' is shared by multiple parameters`],
            warnings: [],
            info: [],
            suggestions: ['Consider renaming parameters to resolve conflict']
          });
        });
      }
    }
  }

  private validateNamingConflicts(parameters: RTBParameter[], crossResults: Map<string, ValidationResult>): void {
    const nameConflicts = new Map<string, string[]>();

    parameters.forEach(param => {
      if (!nameConflicts.has(param.name)) {
        nameConflicts.set(param.name, []);
      }
      nameConflicts.get(param.name)!.push(param.name);
    });

    // Check for duplicate names
    for (const [name, names] of nameConflicts) {
      if (names.length > 1) {
        names.forEach(paramName => {
          this.addToCrossResult(crossResults, paramName, {
            valid: false,
            errors: [`Duplicate parameter name '${name}' found`],
            warnings: [],
            info: [],
            suggestions: ['Rename parameter to ensure uniqueness']
          });
        });
      }
    }
  }

  private validateTypeCompatibility(parameters: RTBParameter[], crossResults: Map<string, ValidationResult>): void {
    // Validate parameter types based on their hierarchy and relationships
    const paramByName = new Map(parameters.map(p => [p.name, p]));

    parameters.forEach(param => {
      if (!param.hierarchy || param.hierarchy.length === 0) {
        return;
      }

      // Check if parameter type matches parent expectations
      const parentName = param.hierarchy[param.hierarchy.length - 2];
      if (parentName) {
        const parentParam = paramByName.get(parentName);
        if (parentParam) {
          const compatibility = this.checkTypeCompatibility(parentParam, param);
          if (!compatibility.valid) {
            this.addToCrossResult(crossResults, param.name, {
              valid: false,
              errors: [compatibility.error],
              warnings: compatibility.warnings,
              info: [],
              suggestions: compatibility.suggestions
            });
          }
        }
      }
    });
  }

  private checkTypeCompatibility(parentParam: RTBParameter, childParam: RTBParameter): {
    valid: boolean;
    error?: string;
    warnings: string[];
    suggestions: string[];
  } {
    // Implement type compatibility logic
    const typeCompatibilityRules: Record<string, Record<string, boolean>> = {
      'number': { 'number': true, 'any[]': true, 'object': true },
      'string': { 'string': true, 'any[]': true },
      'boolean': { 'boolean': true, 'any[]': true },
      'object': { 'object': true, 'any[]': true },
      'any[]': { 'any[]': true }
    };

    const isCompatible = typeCompatibilityRules[parentParam.type]?.[childParam.type] ?? false;

    if (!isCompatible) {
      return {
        valid: false,
        error: `Type incompatibility: parent '${parentParam.type}' and child '${childParam.type}'`,
        warnings: [`Consider adjusting types to maintain compatibility`],
        suggestions: ['Review parameter type hierarchy or convert to compatible types']
      };
    }

    return {
      valid: true,
      warnings: [],
      suggestions: []
    };
  }

  private addToCrossResult(
    crossResults: Map<string, ValidationResult>,
    paramName: string,
    result: ValidationResult
  ): void {
    const existing = crossResults.get(paramName);
    if (existing) {
      crossResults.set(paramName, this.mergeValidationResults(existing, result));
    } else {
      crossResults.set(paramName, result);
    }
  }

  private mergeValidationResults(a: ValidationResult, b: ValidationResult): ValidationResult {
    return {
      valid: a.valid && b.valid,
      errors: [...a.errors, ...b.errors],
      warnings: [...a.warnings, ...b.warnings],
      info: [...a.info, ...b.info],
      suggestions: [...a.suggestions, ...b.suggestions]
    };
  }

  private getParameterCount(param: RTBParameter): number {
    // Implement logic to determine parameter count
    // This would depend on the actual parameter structure and values
    return 1; // Default to 1 for now
  }

  private generateSummary(detailedResults: Map<string, ValidationResult>): ValidationReport['summary'] {
    const bySeverity: Record<string, number> = { error: 0, warning: 0, info: 0 };
    const byType: Record<string, number> = {};
    const errorCounts: Map<string, number> = new Map();

    for (const result of detailedResults.values()) {
      if (result.errors.length > 0) {
        bySeverity.error += result.errors.length;
        result.errors.forEach(error => {
          errorCounts.set(error, (errorCounts.get(error) || 0) + 1);
        });
      }

      if (result.warnings.length > 0) {
        bySeverity.warning += result.warnings.length;
      }

      if (result.info.length > 0) {
        bySeverity.info += result.info.length;
      }
    }

    // Get top errors
    const topErrors = Array.from(errorCounts.entries())
      .map(([error, count]) => ({ error, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);

    return {
      bySeverity,
      byType,
      topErrors
    };
  }

  private initializeDefaultRules(): void {
    // Rule 1: Basic parameter structure validation
    this.addRule({
      id: 'basic-structure',
      name: 'Basic Parameter Structure',
      description: 'Validates basic parameter structure and required fields',
      severity: 'error',
      validate: (param) => {
        const errors: string[] = [];
        const warnings: string[] = [];
        const info: string[] = [];
        const suggestions: string[] = [];

        if (!param.name || param.name.trim() === '') {
          errors.push('Parameter name is required');
        }

        if (!param.vsDataType || param.vsDataType.trim() === '') {
          errors.push('vsDataType is required');
        }

        if (!param.type || param.type.trim() === '') {
          errors.push('Parameter type is required');
        }

        if (!param.extractedAt) {
          errors.push('Extracted timestamp is required');
        }

        if (!param.hierarchy || param.hierarchy.length === 0) {
          warnings.push('Parameter has no hierarchy information');
          suggestions.push('Consider adding hierarchical structure for better organization');
        }

        if (param.name && param.name.length > 100) {
          warnings.push('Parameter name is very long (>100 characters)');
          suggestions.push('Consider using a shorter, more descriptive name');
        }

        if (param.description && param.description.length > 500) {
          warnings.push('Parameter description is very long (>500 characters)');
          suggestions.push('Consider splitting the description into multiple parts');
        }

        return {
          valid: errors.length === 0,
          errors,
          warnings,
          info,
          suggestions
        };
      }
    });

    // Rule 2: Type validation
    this.addRule({
      id: 'type-validation',
      name: 'Type Validation',
      description: 'Validates parameter types and values',
      severity: 'error',
      validate: (param) => {
        const errors: string[] = [];
        const warnings: string[] = [];
        const info: string[] = [];
        const suggestions: string[] = [];

        if (param.defaultValue !== undefined) {
          const typeValidation = this.validateTypeValue(param.defaultValue, param.type);
          if (!typeValidation.valid) {
            errors.push(typeValidation.error);
            suggestions.push(typeValidation.suggestion);
          }
        }

        if (param.constraints) {
          for (const constraint of param.constraints) {
            const constraintValidation = this.validateConstraint(constraint, param.type);
            if (!constraintValidation.valid) {
              errors.push(constraintValidation.error);
              suggestions.push(constraintValidation.suggestion);
            }
          }
        }

        return {
          valid: errors.length === 0,
          errors,
          warnings,
          info,
          suggestions
        };
      }
    });

    // Rule 3: Naming convention validation
    this.addRule({
      id: 'naming-convention',
      name: 'Naming Convention',
      description: 'Validates parameter naming conventions',
      severity: 'warning',
      validate: (param) => {
        const errors: string[] = [];
        const warnings: string[] = [];
        const info: string[] = [];
        const suggestions: string[] = [];

        // Check naming convention
        if (!/^[a-zA-Z][a-zA-Z0-9_]*$/.test(param.name)) {
          errors.push('Parameter name contains invalid characters');
          suggestions.push('Use only alphanumeric characters and underscores, starting with a letter');
        }

        // Check for common prefixes
        const commonPrefixes = ['gnb', 'lte', 'enb', 'eutran', 'utran'];
        if (!commonPrefixes.some(prefix => param.name.toLowerCase().startsWith(prefix))) {
          warnings.push('Parameter name does not follow common Ericsson naming conventions');
          suggestions.push('Consider using standard Ericsson prefixes like gnb_, lte_, enb_, etc.');
        }

        return {
          valid: errors.length === 0,
          errors,
          warnings,
          info,
          suggestions
        };
      }
    });

    // Rule 4: Performance validation
    this.addRule({
      id: 'performance-validation',
      name: 'Performance Validation',
      description: 'Validates parameter performance implications',
      severity: 'warning',
      validate: (param) => {
        const errors: string[] = [];
        const warnings: string[] = [];
        const info: string[] = [];
        const suggestions: string[] = [];

        // Check for parameters that might impact performance
        if (param.type === 'any[]' && (!param.constraints || param.constraints.length === 0)) {
          warnings.push('Unconstrained array parameters may impact performance');
          suggestions.push('Consider adding size constraints or performance limits');
        }

        if (param.type === 'object' && !param.description) {
          warnings.push('Object parameters without descriptions may be unclear');
          suggestions.push('Add descriptions to explain object structure and usage');
        }

        return {
          valid: errors.length === 0,
          errors,
          warnings,
          info,
          suggestions
        };
      }
    });

    // Rule 5: Documentation validation
    this.addRule({
      id: 'documentation-validation',
      name: 'Documentation Validation',
      description: 'Validates parameter documentation quality',
      severity: 'info',
      validate: (param) => {
        const errors: string[] = [];
        const warnings: string[] = [];
        const info: string[] = [];
        const suggestions: string[] = [];

        if (!param.description || param.description.trim() === '') {
          warnings.push('Parameter has no description');
          suggestions.push('Add a descriptive explanation of the parameter purpose and usage');
        } else {
          // Check description quality
          if (param.description.length < 50) {
            warnings.push('Parameter description is very short');
            suggestions.push('Provide more detailed information about parameter usage');
          }

          if (!param.description.includes('.')) {
            warnings.push('Parameter description is not properly formatted');
            suggestions.push('Use proper sentence structure with periods');
          }
        }

        return {
          valid: errors.length === 0,
          errors,
          warnings,
          info,
          suggestions
        };
      }
    });
  }

  private validateTypeValue(value: any, type: string): {
    valid: boolean;
    error?: string;
    suggestion?: string;
  } {
    switch (type) {
      case 'number':
        if (typeof value === 'number' && !isNaN(value)) {
          return { valid: true };
        }
        if (typeof value === 'string') {
          const num = parseFloat(value);
          if (!isNaN(num)) {
            return { valid: true };
          }
        }
        return {
          valid: false,
          error: `Value '${value}' is not a valid number`,
          suggestion: 'Provide a valid numeric value'
        };

      case 'string':
        if (typeof value === 'string') {
          return { valid: true };
        }
        return {
          valid: false,
          error: `Value is not a string`,
          suggestion: 'Provide a string value'
        };

      case 'boolean':
        if (typeof value === 'boolean') {
          return { valid: true };
        }
        if (typeof value === 'string') {
          if (['true', 'false', '1', '0'].includes(value.toLowerCase())) {
            return { valid: true };
          }
        }
        return {
          valid: false,
          error: `Value is not a valid boolean`,
          suggestion: 'Provide true, false, 1, or 0'
        };

      case 'object':
        if (typeof value === 'object' && value !== null) {
          return { valid: true };
        }
        return {
          valid: false,
          error: `Value is not a valid object`,
          suggestion: 'Provide a JSON object'
        };

      case 'any[]':
        if (Array.isArray(value)) {
          return { valid: true };
        }
        return {
          valid: false,
          error: `Value is not a valid array`,
          suggestion: 'Provide an array of values'
        };

      default:
        return { valid: true }; // Unknown types are considered valid
    }
  }

  private validateConstraint(constraint: ConstraintSpec, type: string): {
    valid: boolean;
    error?: string;
    suggestion?: string;
  } {
    switch (constraint.type) {
      case 'range':
        if (type !== 'number') {
          return {
            valid: false,
            error: 'Range constraint can only be applied to numeric parameters',
            suggestion: 'Use appropriate constraint type for parameter type'
          };
        }
        break;

      case 'length':
        if (type !== 'string') {
          return {
            valid: false,
            error: 'Length constraint can only be applied to string parameters',
            suggestion: 'Use appropriate constraint type for parameter type'
          };
        }
        break;

      case 'enum':
        if (!Array.isArray(constraint.value)) {
          return {
            valid: false,
            error: 'Enum constraint must have an array of values',
            suggestion: 'Provide an array of allowed values'
          };
        }
        break;

      case 'pattern':
        if (type !== 'string') {
          return {
            valid: false,
            error: 'Pattern constraint can only be applied to string parameters',
            suggestion: 'Use appropriate constraint type for parameter type'
          };
        }
        try {
          new RegExp(constraint.value as string);
        } catch {
          return {
            valid: false,
            error: 'Invalid regular expression pattern',
            suggestion: 'Provide a valid regular expression'
          };
        }
        break;
    }

    return { valid: true };
  }
}