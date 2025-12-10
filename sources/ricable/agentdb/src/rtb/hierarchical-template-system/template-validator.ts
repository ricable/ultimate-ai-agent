/**
 * Template Validator - Comprehensive Validation and Error Handling
 *
 * Provides extensive validation for RTB templates, priority rules,
 * inheritance chains, and configuration constraints with detailed error reporting.
 */

import {
  RTBTemplate,
  TemplateMeta,
  CustomFunction,
  ConditionOperator,
  EvaluationOperator,
  ConstraintSpec,
  RTBParameter
} from '../../types/rtb-types';
import {
  TemplatePriorityInfo,
  TemplateInheritanceChain,
  ParameterConflict,
  ValidationError,
  ValidationWarning,
  TemplateValidationResult
} from './priority-engine';
import { TemplateRegistry } from './template-registry';

/**
 * Validation severity levels
 */
export enum ValidationSeverity {
  ERROR = 'error',
  WARNING = 'warning',
  INFO = 'info'
}

/**
 * Validation rule definition
 */
export interface ValidationRule {
  name: string;
  description: string;
  severity: ValidationSeverity;
  enabled: boolean;
  validate: (context: ValidationContext) => ValidationResult[];
  category: 'structure' | 'inheritance' | 'priority' | 'content' | 'performance' | 'security';
  precedence: number; // Higher precedence rules run first
}

/**
 * Validation context
 */
export interface ValidationContext {
  template: RTBTemplate;
  templateName: string;
  priority: TemplatePriorityInfo;
  registry: TemplateRegistry;
  inheritanceChain?: TemplateInheritanceChain;
  metadata?: Record<string, any>;
  environment?: string;
  featureFlags?: Record<string, boolean>;
}

/**
 * Validation result
 */
export interface ValidationResult {
  valid: boolean;
  rule: string;
  severity: ValidationSeverity;
  message: string;
  code: string;
  path?: string;
  suggestion?: string;
  data?: any;
}

/**
 * Template validation summary
 */
export interface TemplateValidationSummary {
  templateName: string;
  isValid: boolean;
  totalErrors: number;
  totalWarnings: number;
  totalInfo: number;
  categories: Record<string, {
    errors: number;
    warnings: number;
    info: number;
  }>;
  rulesExecuted: number;
  validationTime: number;
  recommendations: string[];
}

/**
 * Validation configuration
 */
export interface ValidationConfig {
  enabledCategories?: string[];
  enabledRules?: string[];
  strictMode?: boolean;
  maxInheritanceDepth?: number;
  maxConfigurationSize?: number;
  maxCustomFunctions?: number;
  enablePerformanceValidation?: boolean;
  enableSecurityValidation?: boolean;
  customConstraints?: Record<string, ConstraintSpec[]>;
}

/**
 * Template Validator
 *
 * Comprehensive validation system for RTB templates with extensive
 * rule-based validation, detailed error reporting, and performance analysis.
 */
export class TemplateValidator {
  private registry: TemplateRegistry;
  private validationRules: ValidationRule[] = [];
  private validationCache = new Map<string, TemplateValidationSummary>();
  private config: Required<ValidationConfig>;

  constructor(registry: TemplateRegistry, config: ValidationConfig = {}) {
    this.registry = registry;
    this.config = {
      enabledCategories: config.enabledCategories || ['structure', 'inheritance', 'priority', 'content'],
      enabledRules: config.enabledRules || [],
      strictMode: config.strictMode ?? false,
      maxInheritanceDepth: config.maxInheritanceDepth ?? 10,
      maxConfigurationSize: config.maxConfigurationSize ?? 1024 * 1024, // 1MB
      maxCustomFunctions: config.maxCustomFunctions ?? 50,
      enablePerformanceValidation: config.enablePerformanceValidation ?? true,
      enableSecurityValidation: config.enableSecurityValidation ?? true,
      customConstraints: config.customConstraints || {}
    };

    this.initializeValidationRules();
  }

  /**
   * Validate a single template
   */
  async validateTemplate(
    templateName: string,
    template: RTBTemplate,
    priority: TemplatePriorityInfo,
    inheritanceChain?: TemplateInheritanceChain
  ): Promise<TemplateValidationSummary> {
    const startTime = Date.now();

    // Check cache first
    const cacheKey = this.generateValidationCacheKey(templateName, template, priority);
    if (this.validationCache.has(cacheKey)) {
      return this.validationCache.get(cacheKey)!;
    }

    const context: ValidationContext = {
      template,
      templateName,
      priority,
      registry: this.registry,
      inheritanceChain,
      environment: template.meta?.environment,
      featureFlags: {}
    };

    const results: ValidationResult[] = [];
    const categories: Record<string, { errors: number; warnings: number; info: number }> = {};

    // Get applicable rules
    const applicableRules = this.getApplicableRules();

    // Execute validation rules
    for (const rule of applicableRules) {
      try {
        const ruleResults = rule.validate(context);
        results.push(...ruleResults);

        // Categorize results
        if (!categories[rule.category]) {
          categories[rule.category] = { errors: 0, warnings: 0, info: 0 };
        }

        for (const result of ruleResults) {
          switch (result.severity) {
            case ValidationSeverity.ERROR:
              categories[rule.category].errors++;
              break;
            case ValidationSeverity.WARNING:
              categories[rule.category].warnings++;
              break;
            case ValidationSeverity.INFO:
              categories[rule.category].info++;
              break;
          }
        }
      } catch (error) {
        // Rule execution failed
        results.push({
          valid: false,
          rule: rule.name,
          severity: ValidationSeverity.ERROR,
          message: `Validation rule '${rule.name}' failed: ${error}`,
          code: 'RULE_EXECUTION_ERROR'
        });
      }
    }

    // Count results by severity
    const totalErrors = results.filter(r => r.severity === ValidationSeverity.ERROR).length;
    const totalWarnings = results.filter(r => r.severity === ValidationSeverity.WARNING).length;
    const totalInfo = results.filter(r => r.severity === ValidationSeverity.INFO).length;

    // Generate recommendations
    const recommendations = this.generateRecommendations(results);

    const validationTime = Date.now() - startTime;

    const summary: TemplateValidationSummary = {
      templateName,
      isValid: totalErrors === 0,
      totalErrors,
      totalWarnings,
      totalInfo,
      categories,
      rulesExecuted: applicableRules.length,
      validationTime,
      recommendations
    };

    // Cache result
    this.validationCache.set(cacheKey, summary);

    return summary;
  }

  /**
   * Validate inheritance chain
   */
  async validateInheritanceChain(chain: TemplateInheritanceChain): Promise<TemplateValidationResult> {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    // Validate chain structure
    if (chain.chain.length === 0) {
      errors.push({
        code: 'EMPTY_INHERITANCE_CHAIN',
        message: 'Inheritance chain cannot be empty',
        severity: 'error'
      });
    }

    // Validate priority ordering
    for (let i = 0; i < chain.chain.length - 1; i++) {
      const current = chain.chain[i];
      const next = chain.chain[i + 1];

      if (current.level > next.level) {
        warnings.push({
          code: 'INVALID_PRIORITY_ORDER',
          message: `Priority order violation: ${current.category} (${current.level}) should come after ${next.category} (${next.level})`,
          template: chain.templateName
        });
      }
    }

    // Validate inheritance depth
    if (chain.chain.length > this.config.maxInheritanceDepth) {
      errors.push({
        code: 'MAX_INHERITANCE_DEPTH_EXCEEDED',
        message: `Inheritance chain depth ${chain.chain.length} exceeds maximum allowed depth of ${this.config.maxInheritanceDepth}`,
        template: chain.templateName,
        severity: 'error'
      });
    }

    // Validate conflicts
    for (const conflict of chain.conflicts) {
      if (conflict.resolutionStrategy === 'custom' && !conflict.reason) {
        warnings.push({
          code: 'UNDOCUMENTED_CUSTOM_RESOLUTION',
          message: `Custom conflict resolution for parameter '${conflict.parameter}' lacks documentation`,
          template: chain.templateName,
          parameter: conflict.parameter,
          suggestion: 'Add a reason field to document custom resolution logic'
        });
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings,
      metadata: {
        validationTime: 0,
        parametersValidated: Object.keys(chain.resolvedTemplate.configuration || {}).length,
        constraintsChecked: chain.conflicts.length
      }
    };
  }

  /**
   * Get applicable validation rules
   */
  private getApplicableRules(): ValidationRule[] {
    let rules = [...this.validationRules];

    // Filter by enabled categories
    if (this.config.enabledCategories.length > 0) {
      rules = rules.filter(rule => this.config.enabledCategories.includes(rule.category));
    }

    // Filter by enabled rules
    if (this.config.enabledRules.length > 0) {
      rules = rules.filter(rule => this.config.enabledRules.includes(rule.name));
    }

    // Filter by enabled status
    rules = rules.filter(rule => rule.enabled);

    // Sort by precedence
    rules.sort((a, b) => b.precedence - a.precedence);

    return rules;
  }

  /**
   * Generate validation recommendations
   */
  private generateRecommendations(results: ValidationResult[]): string[] {
    const recommendations: string[] = [];

    // Analyze error patterns
    const errorCodes = results.filter(r => r.severity === ValidationSeverity.ERROR)
      .map(r => r.code);
    const warningCodes = results.filter(r => r.severity === ValidationSeverity.WARNING)
      .map(r => r.code);

    // Structure recommendations
    if (errorCodes.includes('MISSING_CONFIGURATION')) {
      recommendations.push('Add configuration object to template');
    }

    if (errorCodes.includes('INVALID_CUSTOM_FUNCTION')) {
      recommendations.push('Review custom function syntax and requirements');
    }

    // Inheritance recommendations
    if (errorCodes.includes('CIRCULAR_DEPENDENCY')) {
      recommendations.push('Resolve circular dependencies by restructuring template hierarchy');
    }

    if (warningCodes.includes('MAX_INHERITANCE_DEPTH_EXCEEDED')) {
      recommendations.push('Consider reducing inheritance depth for better performance');
    }

    // Performance recommendations
    if (warningCodes.includes('LARGE_CONFIGURATION_SIZE')) {
      recommendations.push('Consider splitting large templates into smaller, focused ones');
    }

    if (warningCodes.includes('MANY_CUSTOM_FUNCTIONS')) {
      recommendations.push('Consider moving complex logic to external libraries or services');
    }

    return recommendations;
  }

  /**
   * Generate validation cache key
   */
  private generateValidationCacheKey(
    templateName: string,
    template: RTBTemplate,
    priority: TemplatePriorityInfo
  ): string {
    const content = JSON.stringify({ template, priority: priority.level });
    const hash = Buffer.from(content).toString('base64').slice(0, 16);
    return `${templateName}:${hash}`;
  }

  /**
   * Initialize default validation rules
   */
  private initializeValidationRules(): void {
    this.validationRules = [
      // Structure validation rules
      {
        name: 'template_structure_validation',
        description: 'Validate basic template structure',
        severity: ValidationSeverity.ERROR,
        enabled: true,
        category: 'structure',
        precedence: 100,
        validate: (context) => this.validateTemplateStructure(context)
      },

      {
        name: 'custom_function_validation',
        description: 'Validate custom function syntax and content',
        severity: ValidationSeverity.ERROR,
        enabled: true,
        category: 'structure',
        precedence: 95,
        validate: (context) => this.validateCustomFunctions(context)
      },

      {
        name: 'condition_validation',
        description: 'Validate condition operators and expressions',
        severity: ValidationSeverity.ERROR,
        enabled: true,
        category: 'structure',
        precedence: 90,
        validate: (context) => this.validateConditions(context)
      },

      {
        name: 'evaluation_validation',
        description: 'Validate evaluation operators and expressions',
        severity: ValidationSeverity.ERROR,
        enabled: true,
        category: 'structure',
        precedence: 85,
        validate: (context) => this.validateEvaluations(context)
      },

      // Inheritance validation rules
      {
        name: 'inheritance_structure_validation',
        description: 'Validate inheritance structure and dependencies',
        severity: ValidationSeverity.WARNING,
        enabled: true,
        category: 'inheritance',
        precedence: 80,
        validate: (context) => this.validateInheritanceStructure(context)
      },

      {
        name: 'circular_dependency_detection',
        description: 'Detect circular dependencies in inheritance chains',
        severity: ValidationSeverity.ERROR,
        enabled: true,
        category: 'inheritance',
        precedence: 75,
        validate: (context) => this.detectCircularDependencies(context)
      },

      // Priority validation rules
      {
        name: 'priority_level_validation',
        description: 'Validate priority levels and categories',
        severity: ValidationSeverity.WARNING,
        enabled: true,
        category: 'priority',
        precedence: 70,
        validate: (context) => this.validatePriorityLevels(context)
      },

      {
        name: 'priority_consistency_validation',
        description: 'Validate priority consistency across inheritance chain',
        severity: ValidationSeverity.WARNING,
        enabled: true,
        category: 'priority',
        precedence: 65,
        validate: (context) => this.validatePriorityConsistency(context)
      },

      // Content validation rules
      {
        name: 'configuration_size_validation',
        description: 'Validate configuration size and complexity',
        severity: ValidationSeverity.WARNING,
        enabled: true,
        category: 'content',
        precedence: 60,
        validate: (context) => this.validateConfigurationSize(context)
      },

      {
        name: 'parameter_name_validation',
        description: 'Validate parameter naming conventions',
        severity: ValidationSeverity.INFO,
        enabled: true,
        category: 'content',
        precedence: 55,
        validate: (context) => this.validateParameterNames(context)
      },

      {
        name: 'metadata_validation',
        description: 'Validate template metadata completeness',
        severity: ValidationSeverity.WARNING,
        enabled: true,
        category: 'content',
        precedence: 50,
        validate: (context) => this.validateMetadata(context)
      },

      // Performance validation rules
      {
        name: 'performance_impact_validation',
        description: 'Validate potential performance issues',
        severity: ValidationSeverity.WARNING,
        enabled: this.config.enablePerformanceValidation,
        category: 'performance',
        precedence: 45,
        validate: (context) => this.validatePerformanceImpact(context)
      },

      // Security validation rules
      {
        name: 'security_validation',
        description: 'Validate security considerations',
        severity: ValidationSeverity.WARNING,
        enabled: this.config.enableSecurityValidation,
        category: 'security',
        precedence: 40,
        validate: (context) => this.validateSecurity(context)
      }
    ];
  }

  /**
   * Validate basic template structure
   */
  private validateTemplateStructure(context: ValidationContext): ValidationResult[] {
    const results: ValidationResult[] = [];
    const { template, templateName } = context;

    // Check if template is an object
    if (!template || typeof template !== 'object') {
      results.push({
        valid: false,
        rule: 'template_structure_validation',
        severity: ValidationSeverity.ERROR,
        message: 'Template must be a valid object',
        code: 'INVALID_TEMPLATE_TYPE'
      });
      return results;
    }

    // Check configuration exists
    if (!template.configuration) {
      results.push({
        valid: false,
        rule: 'template_structure_validation',
        severity: ValidationSeverity.ERROR,
        message: 'Template must have a configuration object',
        code: 'MISSING_CONFIGURATION'
      });
    } else if (typeof template.configuration !== 'object') {
      results.push({
        valid: false,
        rule: 'template_structure_validation',
        severity: ValidationSeverity.ERROR,
        message: 'Configuration must be an object',
        code: 'INVALID_CONFIGURATION_TYPE'
      });
    }

    // Check configuration is not empty
    if (template.configuration && Object.keys(template.configuration).length === 0) {
      results.push({
        valid: false,
        rule: 'template_structure_validation',
        severity: ValidationSeverity.WARNING,
        message: 'Configuration object is empty',
        code: 'EMPTY_CONFIGURATION',
        suggestion: 'Add at least one configuration parameter'
      });
    }

    return results;
  }

  /**
   * Validate custom functions
   */
  private validateCustomFunctions(context: ValidationContext): ValidationResult[] {
    const results: ValidationResult[] = [];
    const { template } = context;

    if (!template.custom) return results;

    if (template.custom.length > this.config.maxCustomFunctions) {
      results.push({
        valid: false,
        rule: 'custom_function_validation',
        severity: ValidationSeverity.WARNING,
        message: `Template has ${template.custom.length} custom functions, which exceeds the recommended maximum of ${this.config.maxCustomFunctions}`,
        code: 'TOO_MANY_CUSTOM_FUNCTIONS',
        suggestion: 'Consider moving complex logic to external libraries'
      });
    }

    for (const func of template.custom) {
      // Check function name
      if (!func.name || typeof func.name !== 'string') {
        results.push({
          valid: false,
          rule: 'custom_function_validation',
          severity: ValidationSeverity.ERROR,
          message: 'Custom function must have a valid name',
          code: 'INVALID_FUNCTION_NAME'
        });
        continue;
      }

      // Check function name format
      if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(func.name)) {
        results.push({
          valid: false,
          rule: 'custom_function_validation',
          severity: ValidationSeverity.WARNING,
          message: `Function name '${func.name}' should follow naming conventions`,
          code: 'INVALID_FUNCTION_NAMING',
          suggestion: 'Use camelCase with alphanumeric characters and underscores only'
        });
      }

      // Check args
      if (!Array.isArray(func.args)) {
        results.push({
          valid: false,
          rule: 'custom_function_validation',
          severity: ValidationSeverity.ERROR,
          message: `Function '${func.name}' args must be an array`,
          code: 'INVALID_FUNCTION_ARGS'
        });
      }

      // Check body
      if (!Array.isArray(func.body) || func.body.length === 0) {
        results.push({
          valid: false,
          rule: 'custom_function_validation',
          severity: ValidationSeverity.ERROR,
          message: `Function '${func.name}' must have a non-empty body array`,
          code: 'INVALID_FUNCTION_BODY'
        });
      }
    }

    return results;
  }

  /**
   * Validate condition operators
   */
  private validateConditions(context: ValidationContext): ValidationResult[] {
    const results: ValidationResult[] = [];
    const { template } = context;

    if (!template.conditions) return results;

    for (const [key, condition] of Object.entries(template.conditions)) {
      // Check if condition
      if (!condition.if || typeof condition.if !== 'string') {
        results.push({
          valid: false,
          rule: 'condition_validation',
          severity: ValidationSeverity.ERROR,
          message: `Condition '${key}' must have a valid 'if' expression`,
          code: 'INVALID_CONDITION_IF',
          path: `conditions.${key}.if`
        });
      }

      // Check then clause
      if (!condition.then) {
        results.push({
          valid: false,
          rule: 'condition_validation',
          severity: ValidationSeverity.ERROR,
          message: `Condition '${key}' must have a 'then' clause`,
          code: 'MISSING_CONDITION_THEN',
          path: `conditions.${key}.then`
        });
      }

      // Check else clause
      if (!condition.else) {
        results.push({
          valid: false,
          rule: 'condition_validation',
          severity: ValidationSeverity.WARNING,
          message: `Condition '${key}' should have an 'else' clause`,
          code: 'MISSING_CONDITION_ELSE',
          path: `conditions.${key}.else`,
          suggestion: 'Add an else clause to handle all cases'
        });
      }
    }

    return results;
  }

  /**
   * Validate evaluation operators
   */
  private validateEvaluations(context: ValidationContext): ValidationResult[] {
    const results: ValidationResult[] = [];
    const { template } = context;

    if (!template.evaluations) return results;

    for (const [key, evaluation] of Object.entries(template.evaluations)) {
      // Check eval expression
      if (!evaluation.eval || typeof evaluation.eval !== 'string') {
        results.push({
          valid: false,
          rule: 'evaluation_validation',
          severity: ValidationSeverity.ERROR,
          message: `Evaluation '${key}' must have a valid 'eval' expression`,
          code: 'INVALID_EVAL_EXPRESSION',
          path: `evaluations.${key}.eval`
        });
      }

      // Check args if present
      if (evaluation.args && !Array.isArray(evaluation.args)) {
        results.push({
          valid: false,
          rule: 'evaluation_validation',
          severity: ValidationSeverity.ERROR,
          message: `Evaluation '${key}' args must be an array`,
          code: 'INVALID_EVAL_ARGS',
          path: `evaluations.${key}.args`
        });
      }
    }

    return results;
  }

  /**
   * Validate inheritance structure
   */
  private validateInheritanceStructure(context: ValidationContext): ValidationResult[] {
    const results: ValidationResult[] = [];
    const { priority, templateName } = context;

    // Check inheritance references exist
    if (priority.inherits_from) {
      const parents = Array.isArray(priority.inherits_from)
        ? priority.inherits_from
        : [priority.inherits_from];

      for (const parent of parents) {
        if (!this.registry.hasTemplate(parent)) {
          results.push({
            valid: false,
            rule: 'inheritance_structure_validation',
            severity: ValidationSeverity.WARNING,
            message: `Parent template '${parent}' not found in registry`,
            code: 'MISSING_PARENT_TEMPLATE',
            suggestion: `Ensure template '${parent}' is registered or remove inheritance reference`
          });
        }
      }
    }

    return results;
  }

  /**
   * Detect circular dependencies
   */
  private detectCircularDependencies(context: ValidationContext): ValidationResult[] {
    const results: ValidationResult[] = [];
    const { templateName } = context;

    // This is a simplified circular dependency detection
    // In a full implementation, this would traverse the inheritance graph
    const visited = new Set<string>();
    const recursionStack = new Set<string>();

    const hasCycle = (name: string): boolean => {
      if (recursionStack.has(name)) return true;
      if (visited.has(name)) return false;

      visited.add(name);
      recursionStack.add(name);

      const template = this.registry.getTemplate(name);
      if (template?.meta?.inherits_from) {
        const parents = Array.isArray(template.meta.inherits_from)
          ? template.meta.inherits_from
          : [template.meta.inherits_from];

        for (const parent of parents) {
          if (hasCycle(parent)) return true;
        }
      }

      recursionStack.delete(name);
      return false;
    };

    if (hasCycle(templateName)) {
      results.push({
        valid: false,
        rule: 'circular_dependency_detection',
        severity: ValidationSeverity.ERROR,
        message: `Circular dependency detected in template inheritance chain`,
        code: 'CIRCULAR_DEPENDENCY',
        suggestion: 'Restructure template hierarchy to eliminate circular references'
      });
    }

    return results;
  }

  /**
   * Validate priority levels
   */
  private validatePriorityLevels(context: ValidationContext): ValidationResult[] {
    const results: ValidationResult[] = [];
    const { priority } = context;

    // Check priority level range
    if (priority.level < 0 || priority.level > 80) {
      results.push({
        valid: false,
        rule: 'priority_level_validation',
        severity: ValidationSeverity.WARNING,
        message: `Priority level ${priority.level} is outside recommended range (0-80)`,
        code: 'INVALID_PRIORITY_RANGE',
        suggestion: 'Use priority levels between 0 (highest) and 80 (lowest)'
      });
    }

    // Check category
    if (!priority.category) {
      results.push({
        valid: false,
        rule: 'priority_level_validation',
        severity: ValidationSeverity.WARNING,
        message: 'Template priority category is missing',
        code: 'MISSING_PRIORITY_CATEGORY',
        suggestion: 'Add a category to describe the template type'
      });
    }

    return results;
  }

  /**
   * Validate priority consistency
   */
  private validatePriorityConsistency(context: ValidationContext): ValidationResult[] {
    const results: ValidationResult[] = [];
    const { priority, inheritanceChain } = context;

    if (!inheritanceChain) return results;

    // Check if parent priorities are higher (lower numbers) than child
    for (let i = 0; i < inheritanceChain.chain.length - 1; i++) {
      const current = inheritanceChain.chain[i];
      const next = inheritanceChain.chain[i + 1];

      if (current.level <= next.level) {
        results.push({
          valid: false,
          rule: 'priority_consistency_validation',
          severity: ValidationSeverity.WARNING,
          message: `Priority inconsistency: ${current.category} (${current.level}) should have higher priority than ${next.category} (${next.level})`,
          code: 'INCONSISTENT_PRIORITY_ORDER',
          suggestion: 'Adjust priority levels to maintain consistent inheritance hierarchy'
        });
      }
    }

    return results;
  }

  /**
   * Validate configuration size
   */
  private validateConfigurationSize(context: ValidationContext): ValidationResult[] {
    const results: ValidationResult[] = [];
    const { template } = context;

    if (!template.configuration) return results;

    const configSize = JSON.stringify(template.configuration).length;

    if (configSize > this.config.maxConfigurationSize) {
      results.push({
        valid: false,
        rule: 'configuration_size_validation',
        severity: ValidationSeverity.WARNING,
        message: `Configuration size (${configSize} bytes) exceeds recommended maximum (${this.config.maxConfigurationSize} bytes)`,
        code: 'LARGE_CONFIGURATION_SIZE',
        suggestion: 'Consider splitting large configurations into smaller templates'
      });
    }

    // Check number of parameters
    const paramCount = Object.keys(template.configuration).length;
    if (paramCount > 100) {
      results.push({
        valid: false,
        rule: 'configuration_size_validation',
        severity: ValidationSeverity.INFO,
        message: `Template has ${paramCount} parameters, which may impact maintainability`,
        code: 'MANY_PARAMETERS',
        suggestion: 'Consider grouping related parameters or splitting into multiple templates'
      });
    }

    return results;
  }

  /**
   * Validate parameter names
   */
  private validateParameterNames(context: ValidationContext): ValidationResult[] {
    const results: ValidationResult[] = [];
    const { template } = context;

    if (!template.configuration) return results;

    for (const paramName of Object.keys(template.configuration)) {
      // Check naming convention
      if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(paramName)) {
        results.push({
          valid: false,
          rule: 'parameter_name_validation',
          severity: ValidationSeverity.INFO,
          message: `Parameter '${paramName}' should follow naming conventions`,
          code: 'INVALID_PARAMETER_NAMING',
          path: `configuration.${paramName}`,
          suggestion: 'Use camelCase with alphanumeric characters and underscores only'
        });
      }

      // Check length
      if (paramName.length > 50) {
        results.push({
          valid: false,
          rule: 'parameter_name_validation',
          severity: ValidationSeverity.WARNING,
          message: `Parameter name '${paramName}' is too long (${paramName.length} characters)`,
          code: 'LONG_PARAMETER_NAME',
          path: `configuration.${paramName}`,
          suggestion: 'Keep parameter names under 50 characters'
        });
      }
    }

    return results;
  }

  /**
   * Validate metadata
   */
  private validateMetadata(context: ValidationContext): ValidationResult[] {
    const results: ValidationResult[] = [];
    const { template, templateName } = context;

    if (!template.meta) {
      results.push({
        valid: false,
        rule: 'metadata_validation',
        severity: ValidationSeverity.INFO,
        message: 'Template lacks metadata',
        code: 'MISSING_METADATA',
        suggestion: 'Add metadata to improve template documentation and discoverability'
      });
      return results;
    }

    const meta = template.meta;

    // Check required fields
    const requiredFields = ['version', 'author', 'description'];
    for (const field of requiredFields) {
      if (!meta[field as keyof TemplateMeta]) {
        results.push({
          valid: false,
          rule: 'metadata_validation',
          severity: ValidationSeverity.WARNING,
          message: `Missing required metadata field: ${field}`,
          code: `MISSING_${field.toUpperCase()}`,
          suggestion: `Add ${field} to template metadata`
        });
      }
    }

    // Check version format
    if (meta.version && !/^\d+\.\d+\.\d+/.test(meta.version)) {
      results.push({
        valid: false,
        rule: 'metadata_validation',
        severity: ValidationSeverity.INFO,
        message: `Version '${meta.version}' should follow semantic versioning (x.y.z)`,
        code: 'INVALID_VERSION_FORMAT',
        suggestion: 'Use semantic versioning (e.g., 1.0.0)'
      });
    }

    return results;
  }

  /**
   * Validate performance impact
   */
  private validatePerformanceImpact(context: ValidationContext): ValidationResult[] {
    const results: ValidationResult[] = [];
    const { template } = context;

    // Check for complex conditions
    if (template.conditions) {
      for (const [key, condition] of Object.entries(template.conditions)) {
        if (condition.if && condition.if.length > 200) {
          results.push({
            valid: false,
            rule: 'performance_impact_validation',
            severity: ValidationSeverity.INFO,
            message: `Condition '${key}' has a complex expression that may impact performance`,
            code: 'COMPLEX_CONDITION',
            path: `conditions.${key}.if`,
            suggestion: 'Consider simplifying the condition or moving logic to a custom function'
          });
        }
      }
    }

    // Check for complex evaluations
    if (template.evaluations) {
      for (const [key, evaluation] of Object.entries(template.evaluations)) {
        if (evaluation.eval && evaluation.eval.length > 200) {
          results.push({
            valid: false,
            rule: 'performance_impact_validation',
            severity: ValidationSeverity.INFO,
            message: `Evaluation '${key}' has a complex expression that may impact performance`,
            code: 'COMPLEX_EVALUATION',
            path: `evaluations.${key}.eval`,
            suggestion: 'Consider simplifying the evaluation or moving logic to a custom function'
          });
        }
      }
    }

    return results;
  }

  /**
   * Validate security considerations
   */
  private validateSecurity(context: ValidationContext): ValidationResult[] {
    const results: ValidationResult[] = [];
    const { template } = context;

    // Check for potentially dangerous patterns in evaluations
    if (template.evaluations) {
      const dangerousPatterns = [
        /eval\s*\(/,
        /Function\s*\(/,
        /setTimeout\s*\(/,
        /setInterval\s*\(/,
        /require\s*\(/,
        /import\s+/
      ];

      for (const [key, evaluation] of Object.entries(template.evaluations)) {
        for (const pattern of dangerousPatterns) {
          if (pattern.test(evaluation.eval)) {
            results.push({
              valid: false,
              rule: 'security_validation',
              severity: ValidationSeverity.WARNING,
              message: `Evaluation '${key}' contains potentially dangerous pattern`,
              code: 'DANGEROUS_EVALUATION_PATTERN',
              path: `evaluations.${key}.eval`,
              suggestion: 'Review evaluation code for security implications'
            });
            break;
          }
        }
      }
    }

    // Check for sensitive parameter names
    if (template.configuration) {
      const sensitivePatterns = [
        /password/i,
        /secret/i,
        /token/i,
        /key/i,
        /credential/i
      ];

      for (const [paramName, paramValue] of Object.entries(template.configuration)) {
        for (const pattern of sensitivePatterns) {
          if (pattern.test(paramName)) {
            results.push({
              valid: false,
              rule: 'security_validation',
              severity: ValidationSeverity.INFO,
              message: `Parameter '${paramName}' appears to contain sensitive information`,
              code: 'SENSITIVE_PARAMETER',
              path: `configuration.${paramName}`,
              suggestion: 'Ensure sensitive parameters are properly encrypted and secured'
            });
            break;
          }
        }
      }
    }

    return results;
  }

  /**
   * Add custom validation rule
   */
  addValidationRule(rule: ValidationRule): void {
    this.validationRules.push(rule);
    this.clearCache();
  }

  /**
   * Remove validation rule
   */
  removeValidationRule(name: string): boolean {
    const index = this.validationRules.findIndex(rule => rule.name === name);
    if (index > -1) {
      this.validationRules.splice(index, 1);
      this.clearCache();
      return true;
    }
    return false;
  }

  /**
   * Enable/disable validation rule
   */
  toggleValidationRule(name: string, enabled: boolean): boolean {
    const rule = this.validationRules.find(rule => rule.name === name);
    if (rule) {
      rule.enabled = enabled;
      this.clearCache();
      return true;
    }
    return false;
  }

  /**
   * Get validation statistics
   */
  getValidationStats(): {
    totalRules: number;
    enabledRules: number;
    categories: Record<string, number>;
    cacheSize: number;
  } {
    const categories: Record<string, number> = {};
    let enabledRules = 0;

    for (const rule of this.validationRules) {
      if (rule.enabled) enabledRules++;
      categories[rule.category] = (categories[rule.category] || 0) + 1;
    }

    return {
      totalRules: this.validationRules.length,
      enabledRules,
      categories,
      cacheSize: this.validationCache.size
    };
  }

  /**
   * Clear validation cache
   */
  clearCache(): void {
    this.validationCache.clear();
  }

  /**
   * Get all validation rules
   */
  getValidationRules(): ValidationRule[] {
    return [...this.validationRules];
  }
}