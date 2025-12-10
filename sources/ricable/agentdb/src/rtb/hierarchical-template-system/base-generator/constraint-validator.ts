import { RTBParameter, RTBTemplate, ConstraintSpec, ConstraintValidator, ProcessingStats } from '../../types/rtb-types';
import { GeneratedTemplate } from './template-generator';

export interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
  score: number;
  processingTime: number;
}

export interface ValidationError {
  parameterId: string;
  constraintType: string;
  actualValue: any;
  expectedValue: any;
  message: string;
  severity: 'error' | 'critical';
  suggestion?: string;
}

export interface ValidationWarning {
  parameterId: string;
  constraintType: string;
  actualValue: any;
  message: string;
  suggestion?: string;
}

export interface ConstraintValidationRule {
  name: string;
  validator: (value: any, constraint: any) => ValidationResult;
  errorMessage: string;
  suggestion?: string;
}

export interface TemplateConstraintProfile {
  templateId: string;
  validationLevel: 'strict' | 'standard' | 'lenient';
  customValidators: Map<string, ConstraintValidator>;
  constraintOverrides: Map<string, ConstraintSpec[]>;
  validationResults: ValidationResult[];
}

export interface XMLConstraintSchema {
  parameterTypes: Map<string, ParameterTypeDefinition>;
  globalConstraints: GlobalConstraint[];
  dependencyRules: DependencyRule[];
  validationProfiles: Map<string, ValidationProfile>;
}

export interface ParameterTypeDefinition {
  typeName: string;
  baseType: string;
  constraints: ConstraintSpec[];
  validationRules: string[];
  defaultValue?: any;
  allowedOperations: string[];
}

export interface GlobalConstraint {
  name: string;
  scope: 'template' | 'moClass' | 'parameter';
  condition: string;
  constraint: ConstraintSpec;
  errorMessage: string;
}

export interface DependencyRule {
  sourceParameter: string;
  targetParameter: string;
  relationship: 'requires' | 'conflicts_with' | 'implies' | 'excludes';
  condition?: string;
  action: string;
}

export interface ValidationProfile {
  name: string;
  level: 'strict' | 'standard' | 'lenient';
  enabledConstraints: string[];
  disabledConstraints: string[];
  customRules: Map<string, ConstraintValidationRule>;
}

export class TemplateConstraintValidator {
  private validationRules: Map<string, ConstraintValidationRule> = new Map();
  private xmlSchema: XMLConstraintSchema;
  private validationProfiles: Map<string, ValidationProfile> = new Map();
  private startTime: number;

  constructor() {
    this.startTime = Date.now();
    this.xmlSchema = {
      parameterTypes: new Map(),
      globalConstraints: [],
      dependencyRules: [],
      validationProfiles: new Map()
    };
    this.initializeValidationRules();
    this.initializeValidationProfiles();
  }

  /**
   * Validate generated templates against XML constraints
   */
  async validateTemplates(
    templates: GeneratedTemplate[],
    xmlSchema?: XMLConstraintSchema
  ): Promise<Map<string, ValidationResult>> {
    console.log(`Validating ${templates.length} templates against constraints`);

    if (xmlSchema) {
      this.xmlSchema = xmlSchema;
    }

    const results = new Map<string, ValidationResult>();

    for (const template of templates) {
      const result = await this.validateTemplate(template);
      results.set(template.templateId, result);

      console.log(`Template ${template.templateId}: ${result.isValid ? 'VALID' : 'INVALID'} (Score: ${result.score.toFixed(2)})`);
    }

    return results;
  }

  /**
   * Validate individual template
   */
  private async validateTemplate(template: GeneratedTemplate): Promise<ValidationResult> {
    const startTime = Date.now();
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    try {
      // Validate template structure
      this.validateTemplateStructure(template, errors, warnings);

      // Validate parameters against constraints
      await this.validateParameters(template.parameters, errors, warnings);

      // Validate configuration values
      await this.validateConfiguration(template.template, errors, warnings);

      // Validate custom functions
      this.validateCustomFunctions(template.template, errors, warnings);

      // Apply global constraints
      await this.applyGlobalConstraints(template.template, errors, warnings);

      // Validate dependencies
      await this.validateDependencies(template.parameters, errors, warnings);

      // Apply cognitive constraint validation
      if (template.metadata.templateId.includes('cognitive')) {
        await this.validateCognitiveConstraints(template, errors, warnings);
      }

      const processingTime = (Date.now() - startTime) / 1000;
      const totalIssues = errors.length + warnings.length;
      const score = Math.max(0, 1 - (errors.length * 0.1 + warnings.length * 0.05));

      return {
        isValid: errors.length === 0,
        errors,
        warnings,
        score,
        processingTime
      };

    } catch (error) {
      return {
        isValid: false,
        errors: [{
          parameterId: 'template',
          constraintType: 'validation_error',
          actualValue: error,
          expectedValue: 'valid_template',
          message: `Template validation failed: ${error}`,
          severity: 'critical'
        }],
        warnings,
        score: 0,
        processingTime: (Date.now() - startTime) / 1000
      };
    }
  }

  /**
   * Validate template structure
   */
  private validateTemplateStructure(
    template: GeneratedTemplate,
    errors: ValidationError[],
    warnings: ValidationWarning[]
  ): void {
    // Check required fields
    if (!template.template.meta) {
      errors.push({
        parameterId: 'template',
        constraintType: 'required_field',
        actualValue: undefined,
        expectedValue: 'metadata',
        message: 'Template missing required metadata',
        severity: 'critical',
        suggestion: 'Add template.meta with version, description, and author'
      });
    }

    if (!template.template.configuration || Object.keys(template.template.configuration).length === 0) {
      warnings.push({
        parameterId: 'template',
        constraintType: 'empty_configuration',
        actualValue: template.template.configuration,
        message: 'Template has no configuration parameters',
        suggestion: 'Add configuration parameters to make the template useful'
      });
    }

    // Check template ID format
    if (!template.metadata.templateId.match(/^[a-zA-Z][a-zA-Z0-9_]*$/)) {
      errors.push({
        parameterId: 'template',
        constraintType: 'id_format',
        actualValue: template.metadata.templateId,
        expectedValue: 'valid_identifier',
        message: 'Template ID contains invalid characters',
        severity: 'error',
        suggestion: 'Use only alphanumeric characters and underscores, starting with a letter'
      });
    }
  }

  /**
   * Validate parameters against their constraints
   */
  private async validateParameters(
    parameters: RTBParameter[],
    errors: ValidationError[],
    warnings: ValidationWarning[]
  ): Promise<void> {
    for (const parameter of parameters) {
      await this.validateParameterConstraints(parameter, errors, warnings);
    }
  }

  /**
   * Validate individual parameter constraints
   */
  private async validateParameterConstraints(
    parameter: RTBParameter,
    errors: ValidationError[],
    warnings: ValidationWarning[]
  ): Promise<void> {
    if (!parameter.constraints) return;

    for (const constraint of parameter.constraints) {
      const validationResult = await this.validateConstraint(parameter, constraint);

      if (!validationResult.isValid) {
        for (const error of validationResult.errors) {
          errors.push({
            ...error,
            parameterId: parameter.id
          });
        }
      }

      for (const warning of validationResult.warnings) {
        warnings.push({
          ...warning,
          parameterId: parameter.id
        });
      }
    }
  }

  /**
   * Validate single constraint
   */
  private async validateConstraint(
    parameter: RTBParameter,
    constraint: ConstraintSpec
  ): Promise<ValidationResult> {
    const validationRule = this.validationRules.get(constraint.type);
    if (!validationRule) {
      return {
        isValid: false,
        errors: [{
          parameterId: parameter.id,
          constraintType: constraint.type,
          actualValue: constraint.value,
          expectedValue: 'valid_constraint_type',
          message: `Unknown constraint type: ${constraint.type}`,
          severity: 'error'
        }],
        warnings: [],
        score: 0,
        processingTime: 0
      };
    }

    const actualValue = parameter.defaultValue;
    return validationRule.validator(actualValue, constraint.value);
  }

  /**
   * Validate configuration values
   */
  private async validateConfiguration(
    template: RTBTemplate,
    errors: ValidationError[],
    warnings: ValidationWarning[]
  ): Promise<void> {
    if (!template.configuration) return;

    for (const [key, value] of Object.entries(template.configuration)) {
      // Validate configuration key format
      if (!key.match(/^[a-zA-Z][a-zA-Z0-9_]*$/)) {
        warnings.push({
          parameterId: `configuration.${key}`,
          constraintType: 'key_format',
          actualValue: key,
          message: 'Configuration key contains invalid characters',
          suggestion: 'Use only alphanumeric characters and underscores'
        });
      }

      // Validate configuration value type
      if (typeof value === 'object' && value !== null) {
        // Check if it's a constraint specification
        if (value.constraints) {
          await this.validateConfigurationConstraints(key, value, errors, warnings);
        }
      }
    }
  }

  /**
   * Validate configuration constraints
   */
  private async validateConfigurationConstraints(
    configKey: string,
    configValue: any,
    errors: ValidationError[],
    warnings: ValidationWarning[]
  ): Promise<void> {
    if (!configValue.constraints) return;

    for (const constraint of configValue.constraints) {
      const validationRule = this.validationRules.get(constraint.type);
      if (!validationRule) continue;

      const result = validationRule.validator(configValue.value, constraint.value);

      if (!result.isValid) {
        for (const error of result.errors) {
          errors.push({
            ...error,
            parameterId: `configuration.${configKey}`,
            actualValue: configValue.value
          });
        }
      }

      for (const warning of result.warnings) {
        warnings.push({
          ...warning,
          parameterId: `configuration.${configKey}`,
          actualValue: configValue.value
        });
      }
    }
  }

  /**
   * Validate custom functions
   */
  private validateCustomFunctions(
    template: RTBTemplate,
    errors: ValidationError[],
    warnings: ValidationWarning[]
  ): void {
    if (!template.custom) return;

    for (const func of template.custom) {
      // Validate function name
      if (!func.name || !func.name.match(/^[a-zA-Z][a-zA-Z0-9_]*$/)) {
        errors.push({
          parameterId: `function.${func.name}`,
          constraintType: 'function_name',
          actualValue: func.name,
          expectedValue: 'valid_function_name',
          message: 'Invalid function name',
          severity: 'error',
          suggestion: 'Use valid Python function name (alphanumeric + underscore, starting with letter)'
        });
      }

      // Validate function arguments
      if (!func.args || !Array.isArray(func.args)) {
        errors.push({
          parameterId: `function.${func.name}`,
          constraintType: 'function_args',
          actualValue: func.args,
          expectedValue: 'array_of_strings',
          message: 'Function arguments must be an array of strings',
          severity: 'error'
        });
      }

      // Validate function body
      if (!func.body || !Array.isArray(func.body) || func.body.length === 0) {
        errors.push({
          parameterId: `function.${func.name}`,
          constraintType: 'function_body',
          actualValue: func.body,
          expectedValue: 'array_of_strings',
          message: 'Function body must be a non-empty array of strings',
          severity: 'error'
        });
      }

      // Check for syntax issues in function body
      this.validateFunctionSyntax(func, errors, warnings);
    }
  }

  /**
   * Validate function syntax
   */
  private validateFunctionSyntax(
    func: any,
    errors: ValidationError[],
    warnings: ValidationWarning[]
  ): void {
    if (!func.body) return;

    const functionCode = func.body.join('\n');

    // Basic syntax checks
    if (functionCode.includes('import ')) {
      warnings.push({
        parameterId: `function.${func.name}`,
        constraintType: 'function_imports',
        actualValue: 'import_statement_found',
        message: 'Custom functions should not contain import statements',
        suggestion: 'Move imports to the main template or use built-in functions'
      });
    }

    if (functionCode.includes('exec(') || functionCode.includes('eval(')) {
      errors.push({
        parameterId: `function.${func.name}`,
        constraintType: 'function_security',
        actualValue: 'dynamic_execution',
        expectedValue: 'safe_code',
        message: 'Custom functions should not use exec() or eval()',
        severity: 'critical',
        suggestion: 'Use safer alternatives for dynamic execution'
      });
    }

    // Check for return statement
    if (!functionCode.includes('return ')) {
      warnings.push({
        parameterId: `function.${func.name}`,
        constraintType: 'function_return',
        actualValue: 'no_return',
        message: 'Function does not have a return statement',
        suggestion: 'Add appropriate return statement to function'
      });
    }
  }

  /**
   * Apply global constraints
   */
  private async applyGlobalConstraints(
    template: RTBTemplate,
    errors: ValidationError[],
    warnings: ValidationWarning[]
  ): Promise<void> {
    for (const globalConstraint of this.xmlSchema.globalConstraints) {
      if (globalConstraint.scope === 'template') {
        await this.validateGlobalConstraint(template, globalConstraint, errors, warnings);
      }
    }
  }

  /**
   * Validate single global constraint
   */
  private async validateGlobalConstraint(
    template: RTBTemplate,
    constraint: GlobalConstraint,
    errors: ValidationError[],
    warnings: ValidationWarning[]
  ): Promise<void> {
    // Check if constraint condition applies
    if (constraint.condition && !this.evaluateCondition(constraint.condition, template)) {
      return;
    }

    // Apply constraint validation
    const validationRule = this.validationRules.get(constraint.constraint.type);
    if (validationRule) {
      const result = validationRule.validator(template, constraint.constraint.value);

      if (!result.isValid) {
        for (const error of result.errors) {
          errors.push({
            ...error,
            parameterId: `global_constraint.${constraint.name}`,
            message: constraint.errorMessage || error.message,
            suggestion: constraint.constraint.errorMessage
          });
        }
      }
    }
  }

  /**
   * Validate parameter dependencies
   */
  private async validateDependencies(
    parameters: RTBParameter[],
    errors: ValidationError[],
    warnings: ValidationWarning[]
  ): Promise<void> {
    for (const dependency of this.xmlSchema.dependencyRules) {
      await this.validateDependency(dependency, parameters, errors, warnings);
    }
  }

  /**
   * Validate single dependency
   */
  private async validateDependency(
    dependency: DependencyRule,
    parameters: RTBParameter[],
    errors: ValidationError[],
    warnings: ValidationError[]
  ): Promise<void> {
    const sourceParam = parameters.find(p => p.id === dependency.sourceParameter);
    const targetParam = parameters.find(p => p.id === dependency.targetParameter);

    if (!sourceParam || !targetParam) {
      warnings.push({
        parameterId: dependency.sourceParameter,
        constraintType: 'dependency',
        actualValue: 'parameter_not_found',
        message: `Dependency validation skipped - parameters not found: ${dependency.sourceParameter}, ${dependency.targetParameter}`
      });
      return;
    }

    const isValid = await this.evaluateDependencyRule(dependency, sourceParam, targetParam);

    if (!isValid) {
      errors.push({
        parameterId: dependency.sourceParameter,
        constraintType: 'dependency_violation',
        actualValue: `${dependency.relationship} with ${dependency.targetParameter}`,
        expectedValue: 'valid_dependency',
        message: `Dependency violation: ${dependency.sourceParameter} ${dependency.relationship} ${dependency.targetParameter}`,
        severity: 'error',
        suggestion: dependency.action
      });
    }
  }

  /**
   * Validate cognitive constraints for cognitive templates
   */
  private async validateCognitiveConstraints(
    template: GeneratedTemplate,
    errors: ValidationError[],
    warnings: ValidationWarning[]
  ): Promise<void> {
    // Check for required cognitive functions
    const requiredCognitiveFunctions = [
      'strange_loop_self_optimize',
      'temporal_reasoning_analysis',
      'meta_cognitive_optimization'
    ];

    if (template.template.custom) {
      const functionNames = template.template.custom.map(f => f.name);

      for (const requiredFunc of requiredCognitiveFunctions) {
        if (!functionNames.includes(requiredFunc)) {
          warnings.push({
            parameterId: 'cognitive_template',
            constraintType: 'missing_cognitive_function',
            actualValue: 'function_not_found',
            message: `Cognitive template missing required function: ${requiredFunc}`,
            suggestion: `Add ${requiredFunc} function for full cognitive capabilities`
          });
        }
      }
    }

    // Validate cognitive-specific constraints
    if (template.template.configuration) {
      await this.validateCognitiveConfiguration(template.template.configuration, errors, warnings);
    }
  }

  /**
   * Validate cognitive configuration
   */
  private async validateCognitiveConfiguration(
    configuration: Record<string, any>,
    errors: ValidationError[],
    warnings: ValidationWarning[]
  ): Promise<void> {
    // Check for temporal reasoning parameters
    if (!configuration.temporal_expansion_factor) {
      warnings.push({
        parameterId: 'cognitive.temporal_expansion',
        constraintType: 'missing_cognitive_parameter',
        actualValue: undefined,
        message: 'Temporal expansion factor not specified',
        suggestion: 'Add temporal_expansion_factor for enhanced temporal reasoning'
      });
    } else if (configuration.temporal_expansion_factor < 100 || configuration.temporal_expansion_factor > 10000) {
      warnings.push({
        parameterId: 'cognitive.temporal_expansion',
        constraintType: 'temporal_expansion_range',
        actualValue: configuration.temporal_expansion_factor,
        message: 'Temporal expansion factor outside recommended range (100-10000)',
        suggestion: 'Use 1000x expansion for optimal cognitive performance'
      });
    }

    // Check for consciousness level
    if (!configuration.consciousness_level) {
      warnings.push({
        parameterId: 'cognitive.consciousness',
        constraintType: 'missing_cognitive_parameter',
        actualValue: undefined,
        message: 'Consciousness level not specified',
        suggestion: 'Add consciousness_level parameter for cognitive awareness'
      });
    }
  }

  /**
   * Evaluate condition expression
   */
  private evaluateCondition(condition: string, context: any): boolean {
    try {
      // Simple condition evaluation (in production, use a safe expression evaluator)
      // This is a simplified version for demonstration
      if (condition.includes('priority')) {
        return context.meta?.priority >= 5;
      }
      if (condition.includes('cognitive')) {
        return context.meta?.tags?.includes('cognitive') || false;
      }
      return true;
    } catch (error) {
      console.warn(`Failed to evaluate condition: ${condition}`, error);
      return false;
    }
  }

  /**
   * Evaluate dependency rule
   */
  private async evaluateDependencyRule(
    dependency: DependencyRule,
    sourceParam: RTBParameter,
    targetParam: RTBParameter
  ): Promise<boolean> {
    switch (dependency.relationship) {
      case 'requires':
        // Source parameter requires target parameter to be set
        return sourceParam.defaultValue !== undefined ||
               (targetParam.defaultValue !== undefined);

      case 'conflicts_with':
        // Source and target parameters should not both be set
        return !(sourceParam.defaultValue !== undefined && targetParam.defaultValue !== undefined);

      case 'implies':
        // If source is set, target must also be set
        return sourceParam.defaultValue === undefined ||
               targetParam.defaultValue !== undefined;

      case 'excludes':
        // If source is set, target must not be set
        return sourceParam.defaultValue === undefined ||
               targetParam.defaultValue === undefined;

      default:
        return true;
    }
  }

  /**
   * Initialize validation rules
   */
  private initializeValidationRules(): void {
    // Range constraint validator
    this.validationRules.set('range', {
      name: 'range',
      validator: (value: any, constraint: any) => {
        const errors: ValidationError[] = [];
        const warnings: ValidationWarning[] = [];

        if (value !== undefined && value !== null && typeof constraint === 'object') {
          const { min, max } = constraint;
          const numValue = Number(value);

          if (isNaN(numValue)) {
            errors.push({
              parameterId: 'unknown',
              constraintType: 'range',
              actualValue: value,
              expectedValue: 'number',
              message: `Range constraint requires numeric value, got ${typeof value}`,
              severity: 'error'
            });
          } else {
            if (min !== undefined && numValue < min) {
              errors.push({
                parameterId: 'unknown',
                constraintType: 'range',
                actualValue: numValue,
                expectedValue: `>= ${min}`,
                message: `Value ${numValue} is below minimum ${min}`,
                severity: 'error',
                suggestion: `Increase value to at least ${min}`
              });
            }

            if (max !== undefined && numValue > max) {
              errors.push({
                parameterId: 'unknown',
                constraintType: 'range',
                actualValue: numValue,
                expectedValue: `<= ${max}`,
                message: `Value ${numValue} is above maximum ${max}`,
                severity: 'error',
                suggestion: `Decrease value to at most ${max}`
              });
            }
          }
        }

        const score = errors.length === 0 ? 1 : Math.max(0, 1 - errors.length * 0.1);

        return {
          isValid: errors.length === 0,
          errors,
          warnings,
          score,
          processingTime: 0
        };
      },
      errorMessage: 'Value must be within specified range'
    });

    // Enum constraint validator
    this.validationRules.set('enum', {
      name: 'enum',
      validator: (value: any, constraint: any) => {
        const errors: ValidationError[] = [];
        const warnings: ValidationWarning[] = [];

        if (value !== undefined && Array.isArray(constraint)) {
          if (!constraint.includes(value)) {
            errors.push({
              parameterId: 'unknown',
              constraintType: 'enum',
              actualValue: value,
              expectedValue: constraint,
              message: `Value ${value} not in allowed values: [${constraint.join(', ')}]`,
              severity: 'error',
              suggestion: `Use one of: ${constraint.join(', ')}`
            });
          }
        }

        const score = errors.length === 0 ? 1 : Math.max(0, 1 - errors.length * 0.1);

        return {
          isValid: errors.length === 0,
          errors,
          warnings,
          score,
          processingTime: 0
        };
      },
      errorMessage: 'Value must be one of the allowed enumeration values'
    });

    // Pattern constraint validator
    this.validationRules.set('pattern', {
      name: 'pattern',
      validator: (value: any, constraint: any) => {
        const errors: ValidationError[] = [];
        const warnings: ValidationWarning[] = [];

        if (value !== undefined && typeof value === 'string' && typeof constraint === 'string') {
          try {
            const regex = new RegExp(constraint);
            if (!regex.test(value)) {
              errors.push({
                parameterId: 'unknown',
                constraintType: 'pattern',
                actualValue: value,
                expectedValue: constraint,
                message: `Value ${value} does not match pattern ${constraint}`,
                severity: 'error',
                suggestion: `Ensure value matches pattern: ${constraint}`
              });
            }
          } catch (error) {
            errors.push({
              parameterId: 'unknown',
              constraintType: 'pattern',
              actualValue: constraint,
              expectedValue: 'valid_regex',
              message: `Invalid regex pattern: ${constraint}`,
              severity: 'error'
            });
          }
        }

        const score = errors.length === 0 ? 1 : Math.max(0, 1 - errors.length * 0.1);

        return {
          isValid: errors.length === 0,
          errors,
          warnings,
          score,
          processingTime: 0
        };
      },
      errorMessage: 'Value must match the specified pattern'
    });

    // Length constraint validator
    this.validationRules.set('length', {
      name: 'length',
      validator: (value: any, constraint: any) => {
        const errors: ValidationError[] = [];
        const warnings: ValidationWarning[] = [];

        if (value !== undefined && typeof constraint === 'object') {
          const actualLength = typeof value === 'string' ? value.length :
                              Array.isArray(value) ? value.length : 0;
          const { min, max } = constraint;

          if (min !== undefined && actualLength < min) {
            errors.push({
              parameterId: 'unknown',
              constraintType: 'length',
              actualValue: actualLength,
              expectedValue: `>= ${min}`,
              message: `Length ${actualLength} is below minimum ${min}`,
              severity: 'error',
              suggestion: `Increase length to at least ${min}`
            });
          }

          if (max !== undefined && actualLength > max) {
            errors.push({
              parameterId: 'unknown',
              constraintType: 'length',
              actualValue: actualLength,
              expectedValue: `<= ${max}`,
              message: `Length ${actualLength} is above maximum ${max}`,
              severity: 'error',
              suggestion: `Decrease length to at most ${max}`
            });
          }
        }

        const score = errors.length === 0 ? 1 : Math.max(0, 1 - errors.length * 0.1);

        return {
          isValid: errors.length === 0,
          errors,
          warnings,
          score,
          processingTime: 0
        };
      },
      errorMessage: 'Value length must be within specified range'
    });
  }

  /**
   * Initialize validation profiles
   */
  private initializeValidationProfiles(): void {
    // Strict validation profile
    this.validationProfiles.set('strict', {
      name: 'strict',
      level: 'strict',
      enabledConstraints: ['range', 'enum', 'pattern', 'length', 'required'],
      disabledConstraints: [],
      customRules: new Map()
    });

    // Standard validation profile
    this.validationProfiles.set('standard', {
      name: 'standard',
      level: 'standard',
      enabledConstraints: ['range', 'enum', 'pattern'],
      disabledConstraints: ['length'],
      customRules: new Map()
    });

    // Lenient validation profile
    this.validationProfiles.set('lenient', {
      name: 'lenient',
      level: 'lenient',
      enabledConstraints: ['range'],
      disabledConstraints: ['enum', 'pattern', 'length', 'required'],
      customRules: new Map()
    });
  }

  /**
   * Generate validation report
   */
  generateValidationReport(results: Map<string, ValidationResult>): string {
    const report: string[] = [];
    report.push('# Template Validation Report');
    report.push(`Generated: ${new Date().toISOString()}`);
    report.push('');

    let totalErrors = 0;
    let totalWarnings = 0;
    let validTemplates = 0;

    for (const [templateId, result] of results) {
      report.push(`## Template: ${templateId}`);
      report.push(`Status: ${result.isValid ? 'VALID' : 'INVALID'}`);
      report.push(`Score: ${result.score.toFixed(2)}`);
      report.push(`Processing Time: ${result.processingTime.toFixed(3)}s`);
      report.push(`Errors: ${result.errors.length}`);
      report.push(`Warnings: ${result.warnings.length}`);
      report.push('');

      if (result.errors.length > 0) {
        report.push('### Errors:');
        for (const error of result.errors) {
          report.push(`- **${error.parameterId}**: ${error.message}`);
          if (error.suggestion) {
            report.push(`  - Suggestion: ${error.suggestion}`);
          }
        }
        report.push('');
      }

      if (result.warnings.length > 0) {
        report.push('### Warnings:');
        for (const warning of result.warnings) {
          report.push(`- **${warning.parameterId}**: ${warning.message}`);
          if (warning.suggestion) {
            report.push(`  - Suggestion: ${warning.suggestion}`);
          }
        }
        report.push('');
      }

      totalErrors += result.errors.length;
      totalWarnings += result.warnings.length;
      if (result.isValid) validTemplates++;
    }

    // Summary
    report.push('## Summary');
    report.push(`Total Templates: ${results.size}`);
    report.push(`Valid Templates: ${validTemplates}`);
    report.push(`Invalid Templates: ${results.size - validTemplates}`);
    report.push(`Total Errors: ${totalErrors}`);
    report.push(`Total Warnings: ${totalWarnings}`);
    report.push(`Average Score: ${(Array.from(results.values()).reduce((sum, r) => sum + r.score, 0) / results.size).toFixed(2)}`);

    return report.join('\n');
  }
}