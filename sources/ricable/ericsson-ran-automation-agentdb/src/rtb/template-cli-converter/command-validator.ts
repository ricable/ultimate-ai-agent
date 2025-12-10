/**
 * Command Validator
 *
 * Comprehensive validation of generated CLI commands including
 * syntax validation, semantic validation, and constraint checking.
 */

import {
  GeneratedCliCommand,
  CommandValidationResult,
  ValidationError,
  ValidationWarning,
  RecommendedFix,
  TemplateToCliContext,
  ValidationErrorType,
  ValidationWarningType
} from './types';

/**
 * Command Validator Configuration
 */
export interface CommandValidatorConfig {
  /** Validation strictness level */
  strictness: 'lenient' | 'normal' | 'strict';
  /** Enable syntax validation */
  enableSyntaxValidation: boolean;
  /** Enable semantic validation */
  enableSemanticValidation: boolean;
  /** Enable constraint validation */
  enableConstraintValidation: boolean;
  /** Enable performance validation */
  enablePerformanceValidation: boolean;
}

/**
 * Validation rule
 */
interface ValidationRule {
  name: string;
  type: 'syntax' | 'semantic' | 'constraint' | 'performance';
  severity: 'error' | 'warning' | 'info';
  applicable: (command: GeneratedCliCommand, context: TemplateToCliContext) => boolean;
  validate: (command: GeneratedCliCommand, context: TemplateToCliContext) => ValidationResult;
}

/**
 * Validation result
 */
interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
  suggestions?: string[];
}

/**
 * Command Validator Class
 */
export class CommandValidator {
  private config: CommandValidatorConfig;
  private validationRules: ValidationRule[] = [];
  private constraintValidators: Map<string, ConstraintValidator> = new Map();
  private syntaxPatterns: Map<string, RegExp> = new Map();

  constructor(config: CommandValidatorConfig) {
    this.config = {
      enableSyntaxValidation: true,
      enableSemanticValidation: true,
      enableConstraintValidation: true,
      enablePerformanceValidation: true,
      ...config
    };

    this.initializeValidationRules();
    this.initializeSyntaxPatterns();
    this.initializeConstraintValidators();
  }

  /**
   * Validate commands
   */
  public async validate(
    commands: GeneratedCliCommand[],
    context: TemplateToCliContext
  ): Promise<CommandValidationResult> {
    const startTime = Date.now();
    console.log(`Validating ${commands.length} commands...`);

    const allErrors: ValidationError[] = [];
    const allWarnings: ValidationWarning[] = [];
    const allFixes: RecommendedFix[] = [];

    for (const command of commands) {
      const commandResult = await this.validateCommand(command, context);
      allErrors.push(...commandResult.errors);
      allWarnings.push(...commandResult.warnings);
      allFixes.push(...commandResult.fixes);
    }

    // Cross-command validation
    const crossValidationResult = await this.validateCrossCommandDependencies(commands, context);
    allErrors.push(...crossValidationResult.errors);
    allWarnings.push(...crossValidationResult.warnings);
    allFixes.push(...crossValidationResult.fixes);

    const validationTime = Date.now() - startTime;
    const isValid = allErrors.length === 0;

    const stats = {
      totalCommands: commands.length,
      errorCount: allErrors.length,
      warningCount: allWarnings.length,
      validationTime,
      avgComplexity: this.calculateAverageComplexity(commands),
      riskAssessment: this.calculateRiskAssessment(commands)
    };

    console.log(`Validation completed in ${validationTime}ms: ${allErrors.length} errors, ${allWarnings.length} warnings`);

    return {
      isValid,
      errors: allErrors,
      warnings: allWarnings,
      stats,
      recommendedFixes: allFixes
    };
  }

  /**
   * Validate single command
   */
  private async validateCommand(
    command: GeneratedCliCommand,
    context: TemplateToCliContext
  ): Promise<{
    errors: ValidationError[];
    warnings: ValidationWarning[];
    fixes: RecommendedFix[];
  }> {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];
    const fixes: RecommendedFix[] = [];

    // Apply validation rules
    for (const rule of this.validationRules) {
      if (rule.applicable(command, context)) {
        const result = rule.validate(command, context);

        for (const error of result.errors) {
          errors.push({
            message: error,
            commandId: command.id,
            type: this.mapRuleTypeToErrorType(rule.type),
            severity: rule.severity as 'error' | 'critical',
            suggestion: result.suggestions?.[0]
          });
        }

        for (const warning of result.warnings) {
          warnings.push({
            message: warning,
            commandId: command.id,
            type: this.mapRuleTypeToWarningType(rule.type),
            severity: rule.severity as 'warning' | 'info',
            suggestion: result.suggestions?.[0]
          });
        }

        // Generate fixes if suggestions are available
        if (result.suggestions && result.suggestions.length > 0) {
          fixes.push({
            type: 'CORRECTION',
            commandId: command.id,
            description: `Apply validation suggestion: ${result.suggestions[0]}`,
            implementation: result.suggestions[0],
            priority: this.determineFixPriority(rule.severity),
            effort: 'easy',
            impact: { reliability: 0.8 }
          });
        }
      }
    }

    // Command-specific validations
    await this.performCommandSpecificValidations(command, context, errors, warnings, fixes);

    return { errors, warnings, fixes };
  }

  /**
   * Validate cross-command dependencies
   */
  private async validateCrossCommandDependencies(
    commands: GeneratedCliCommand[],
    context: TemplateToCliContext
  ): Promise<{
    errors: ValidationError[];
    warnings: ValidationWarning[];
    fixes: RecommendedFix[];
  }> {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];
    const fixes: RecommendedFix[] = [];

    // Check for duplicate operations
    const duplicates = this.findDuplicateOperations(commands);
    for (const duplicate of duplicates) {
      warnings.push({
        message: `Duplicate operation detected: ${duplicate.operation}`,
        commandId: duplicate.commandIds[0],
        type: 'REDUNDANCY',
        severity: 'warning',
        suggestion: 'Consider consolidating duplicate operations'
      });
    }

    // Check for conflicting operations
    const conflicts = this.findConflictingOperations(commands);
    for (const conflict of conflicts) {
      errors.push({
        message: `Conflicting operations: ${conflict.description}`,
        commandId: conflict.commandIds[0],
        type: 'CONSTRAINT',
        severity: 'error',
        suggestion: conflict.resolution
      });
    }

    // Check for execution order dependencies
    const orderIssues = this.findExecutionOrderIssues(commands);
    for (const issue of orderIssues) {
      warnings.push({
        message: `Potential execution order issue: ${issue.description}`,
        commandId: issue.commandId,
        type: 'DEPENDENCY',
        severity: 'warning',
        suggestion: issue.suggestion
      });
    }

    return { errors, warnings, fixes };
  }

  /**
   * Perform command-specific validations
   */
  private async performCommandSpecificValidations(
    command: GeneratedCliCommand,
    context: TemplateToCliContext,
    errors: ValidationError[],
    warnings: ValidationWarning[],
    fixes: RecommendedFix[]
  ): Promise<void> {
    switch (command.type) {
      case 'SET':
        await this.validateSetCommand(command, context, errors, warnings, fixes);
        break;
      case 'CREATE':
        await this.validateCreateCommand(command, context, errors, warnings, fixes);
        break;
      case 'DELETE':
        await this.validateDeleteCommand(command, context, errors, warnings, fixes);
        break;
      case 'GET':
        await this.validateGetCommand(command, context, errors, warnings, fixes);
        break;
      case 'SCRIPT':
        await this.validateScriptCommand(command, context, errors, warnings, fixes);
        break;
    }
  }

  /**
   * Validate SET command
   */
  private async validateSetCommand(
    command: GeneratedCliCommand,
    context: TemplateToCliContext,
    errors: ValidationError[],
    warnings: ValidationWarning[],
    fixes: RecommendedFix[]
  ): Promise<void> {
    // Check if command starts with 'cmedit set'
    if (!command.command.startsWith('cmedit set ')) {
      errors.push({
        message: 'SET command must start with "cmedit set"',
        commandId: command.id,
        type: 'SYNTAX',
        severity: 'error',
        suggestion: `Command should be: cmedit set ${command.command.substring(10)}`
      });
    }

    // Validate parameters
    if (command.parameters) {
      for (const [paramName, paramValue] of Object.entries(command.parameters)) {
        const validation = this.validateParameter(paramName, paramValue, command);
        if (!validation.isValid) {
          errors.push(...validation.errors.map(err => ({
            ...err,
            commandId: command.id,
            parameterPath: paramName
          })));
        }
        warnings.push(...validation.warnings.map(warn => ({
          ...warn,
          commandId: command.id,
          parameterPath: paramName
        })));
      }
    }

    // Check for dangerous operations
    if (this.isDangerousOperation(command)) {
      warnings.push({
        message: 'Potentially dangerous operation detected',
        commandId: command.id,
        type: 'RISK',
        severity: 'warning',
        suggestion: 'Consider using preview mode first'
      });
    }
  }

  /**
   * Validate CREATE command
   */
  private async validateCreateCommand(
    command: GeneratedCliCommand,
    context: TemplateToCliContext,
    errors: ValidationError[],
    warnings: ValidationWarning[],
    fixes: RecommendedFix[]
  ): Promise<void> {
    // Check if command starts with 'cmedit create'
    if (!command.command.startsWith('cmedit create ')) {
      errors.push({
        message: 'CREATE command must start with "cmedit create"',
        commandId: command.id,
        type: 'SYNTAX',
        severity: 'error'
      });
    }

    // Validate FDN path
    if (command.targetFdn) {
      const fdnValidation = this.validateFdnPath(command.targetFdn);
      if (!fdnValidation.isValid) {
        errors.push(...fdnValidation.errors.map(err => ({
          ...err,
          commandId: command.id
        })));
      }
    }

    // Check for prerequisite operations
    const prerequisites = this.identifyCreatePrerequisites(command);
    for (const prerequisite of prerequisites) {
      warnings.push({
        message: `CREATE operation may require prerequisite: ${prerequisite}`,
        commandId: command.id,
        type: 'DEPENDENCY',
        severity: 'warning',
        suggestion: `Ensure ${prerequisite} exists before creating this object`
      });
    }
  }

  /**
   * Validate DELETE command
   */
  private async validateDeleteCommand(
    command: GeneratedCliCommand,
    context: TemplateToCliContext,
    errors: ValidationError[],
    warnings: ValidationWarning[],
    fixes: RecommendedFix[]
  ): Promise<void> {
    // Check if command starts with 'cmedit delete'
    if (!command.command.startsWith('cmedit delete ')) {
      errors.push({
        message: 'DELETE command must start with "cmedit delete"',
        commandId: command.id,
        type: 'SYNTAX',
        severity: 'error'
      });
    }

    // Check for critical object deletion
    if (this.isCriticalObjectDeletion(command)) {
      errors.push({
        message: 'Attempt to delete critical object',
        commandId: command.id,
        type: 'CRITICAL',
        severity: 'critical',
        suggestion: 'Critical objects should not be deleted automatically'
      });
    }

    // Check for dependent objects
    const dependents = this.identifyDependentObjects(command);
    for (const dependent of dependents) {
      warnings.push({
        message: `Deleting this object may affect: ${dependent}`,
        commandId: command.id,
        type: 'RISK',
        severity: 'warning',
        suggestion: 'Consider the impact on dependent objects'
      });
    }
  }

  /**
   * Validate GET command
   */
  private async validateGetCommand(
    command: GeneratedCliCommand,
    context: TemplateToCliContext,
    errors: ValidationError[],
    warnings: ValidationWarning[],
    fixes: RecommendedFix[]
  ): Promise<void> {
    // Check if command starts with 'cmedit get'
    if (!command.command.startsWith('cmedit get ')) {
      errors.push({
        message: 'GET command must start with "cmedit get"',
        commandId: command.id,
        type: 'SYNTAX',
        severity: 'error'
      });
    }

    // Check for efficient query patterns
    if (this.isInefficientQuery(command)) {
      warnings.push({
        message: 'Inefficient query pattern detected',
        commandId: command.id,
        type: 'PERFORMANCE',
        severity: 'warning',
        suggestion: 'Consider using more specific query parameters'
      });
    }
  }

  /**
   * Validate SCRIPT command
   */
  private async validateScriptCommand(
    command: GeneratedCliCommand,
    context: TemplateToCliContext,
    errors: ValidationError[],
    warnings: ValidationWarning[],
    fixes: RecommendedFix[]
  ): Promise<void> {
    // Check script content
    if (!command.command || command.command.trim().length === 0) {
      errors.push({
        message: 'SCRIPT command cannot be empty',
        commandId: command.id,
        type: 'SYNTAX',
        severity: 'error'
      });
    }

    // Check for dangerous script content
    if (this.containsDangerousScriptContent(command.command)) {
      errors.push({
        message: 'Script contains potentially dangerous content',
        commandId: command.id,
        type: 'CRITICAL',
        severity: 'critical',
        suggestion: 'Review script content for safety'
      });
    }

    // Check for script performance
    if (this.isPerformanceIntensiveScript(command.command)) {
      warnings.push({
        message: 'Script may be performance intensive',
        commandId: command.id,
        type: 'PERFORMANCE',
        severity: 'warning',
        suggestion: 'Consider optimizing script or monitoring execution time'
      });
    }
  }

  /**
   * Validate parameter
   */
  private validateParameter(
    paramName: string,
    paramValue: any,
    command: GeneratedCliCommand
  ): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];
    const suggestions: string[] = [];

    // Get constraint validator for parameter
    const validator = this.constraintValidators.get(paramName);
    if (validator) {
      const result = validator.validate(paramValue);
      if (!result.isValid) {
        errors.push(...result.errors);
        warnings.push(...result.warnings);
        suggestions.push(...result.suggestions);
      }
    }

    // Type validation
    if (!this.isValidParameterType(paramValue)) {
      errors.push(`Invalid parameter type for ${paramName}: ${typeof paramValue}`);
    }

    // Range validation for numeric parameters
    if (typeof paramValue === 'number') {
      const rangeValidation = this.validateParameterRange(paramName, paramValue);
      if (!rangeValidation.isValid) {
        errors.push(...rangeValidation.errors);
        warnings.push(...rangeValidation.warnings);
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings,
      suggestions
    };
  }

  /**
   * Validate FDN path
   */
  private validateFdnPath(fdn: string): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Basic syntax validation
    if (!fdn || fdn.trim().length === 0) {
      errors.push('FDN path cannot be empty');
      return { isValid: false, errors, warnings };
    }

    // Check for invalid characters
    const invalidChars = /[<>"|;\\]/;
    if (invalidChars.test(fdn)) {
      errors.push('FDN contains invalid characters');
    }

    // Check structure
    const parts = fdn.split(',');
    if (parts.length === 0) {
      errors.push('Invalid FDN structure');
    }

    // Validate each part
    for (const part of parts) {
      if (!part.trim()) {
        errors.push('Empty component in FDN path');
        continue;
      }

      if (part.includes('=') && part.split('=').length !== 2) {
        errors.push(`Invalid parameter format in FDN: ${part}`);
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }

  /**
   * Helper methods
   */
  private mapRuleTypeToErrorType(ruleType: string): ValidationErrorType {
    const mapping: Record<string, ValidationErrorType> = {
      'syntax': 'SYNTAX',
      'semantic': 'TYPE_MISMATCH',
      'constraint': 'CONSTRAINT',
      'performance': 'PERFORMANCE'
    };
    return mapping[ruleType] || 'SYNTAX';
  }

  private mapRuleTypeToWarningType(ruleType: string): ValidationWarningType {
    const mapping: Record<string, ValidationWarningType> = {
      'syntax': 'BEST_PRACTICE',
      'semantic': 'INFO',
      'constraint': 'RISK',
      'performance': 'PERFORMANCE'
    };
    return mapping[ruleType] || 'INFO';
  }

  private determineFixPriority(severity: string): 'low' | 'medium' | 'high' | 'critical' {
    const mapping: Record<string, 'low' | 'medium' | 'high' | 'critical'> = {
      'info': 'low',
      'warning': 'medium',
      'error': 'high',
      'critical': 'critical'
    };
    return mapping[severity] || 'medium';
  }

  private isDangerousOperation(command: GeneratedCliCommand): boolean {
    const dangerousPatterns = [
      /adminState=LOCKED/i,
      /operState=DISABLED/i,
      /delete/i,
      /remove/i
    ];

    return dangerousPatterns.some(pattern => pattern.test(command.command));
  }

  private isCriticalObjectDeletion(command: GeneratedCliCommand): boolean {
    const criticalObjects = [
      'ManagedElement',
      'ENBFunction',
      'NRCellCU',
      'MeContext'
    ];

    return criticalObjects.some(obj =>
      command.command.toLowerCase().includes(obj.toLowerCase()) &&
      command.type === 'DELETE'
    );
  }

  private identifyDependentObjects(command: GeneratedCliCommand): string[] {
    // Simplified dependency identification
    const dependencies: string[] = [];

    if (command.command.includes('EUtranCellFDD')) {
      dependencies.push('EUtranRelation', 'NeighbourRelation');
    }

    if (command.command.includes('ENBFunction')) {
      dependencies.push('EUtranCellFDD', 'EUtranFreqRelation');
    }

    return dependencies;
  }

  private identifyCreatePrerequisites(command: GeneratedCliCommand): string[] {
    const prerequisites: string[] = [];

    if (command.command.includes('EUtranFreqRelation')) {
      prerequisites.push('ENBFunction');
    }

    if (command.command.includes('EUtranRelation')) {
      prerequisites.push('EUtranCellFDD');
    }

    return prerequisites;
  }

  private isInefficientQuery(command: GeneratedCliCommand): boolean {
    const inefficientPatterns = [
      /get.*\*[^]*-s/, // Get all attributes without filtering
      /get.*\.[^]*\./, // Deep nested queries
      /get.*where.*like/ // Vague LIKE queries
    ];

    return inefficientPatterns.some(pattern => pattern.test(command.command));
  }

  private containsDangerousScriptContent(script: string): boolean {
    const dangerousPatterns = [
      /rm\s+-rf/i,
      /shutdown/i,
      /reboot/i,
      /format/i,
      /dd\s+if=/i
    ];

    return dangerousPatterns.some(pattern => pattern.test(script));
  }

  private isPerformanceIntensiveScript(script: string): boolean {
    const intensivePatterns = [
      /for.*in.*\$\(/, // Command substitution in loops
      /while.*read/i, // Reading large files
      /find.*\/-type/i, // File system traversal
      /grep.*-r/i // Recursive grep
    ];

    return intensivePatterns.some(pattern => pattern.test(script));
  }

  private isValidParameterType(value: any): boolean {
    const validTypes = ['string', 'number', 'boolean'];
    return validTypes.includes(typeof value) || value === null || value === undefined;
  }

  private validateParameterRange(paramName: string, value: number): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Common parameter ranges
    const ranges: Record<string, { min: number; max: number }> = {
      'qRxLevMin': { min: -140, max: -44 },
      'qQualMin': { min: -34, max: -3 },
      'hysteresis': { min: 0, max: 30 },
      'timeToTrigger': { min: 0, max: 5000 },
      'cellIndividualOffset': { min: -24, max: 24 }
    };

    const range = ranges[paramName];
    if (range && (value < range.min || value > range.max)) {
      errors.push(`${paramName} value ${value} is outside valid range [${range.min}, ${range.max}]`);
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }

  private findDuplicateOperations(commands: GeneratedCliCommand[]): Array<{
    operation: string;
    commandIds: string[];
  }> {
    const operationMap = new Map<string, string[]>();
    const duplicates: Array<{ operation: string; commandIds: string[] }> = [];

    for (const command of commands) {
      const operationKey = this.normalizeOperation(command);
      if (!operationMap.has(operationKey)) {
        operationMap.set(operationKey, []);
      }
      operationMap.get(operationKey)!.push(command.id);
    }

    for (const [operation, commandIds] of operationMap) {
      if (commandIds.length > 1) {
        duplicates.push({ operation, commandIds });
      }
    }

    return duplicates;
  }

  private findConflictingOperations(commands: GeneratedCliCommand[]): Array<{
    commandIds: string[];
    description: string;
    resolution: string;
  }> {
    const conflicts: Array<{ commandIds: string[]; description: string; resolution: string }> = [];

    // Look for SET operations on the same parameter with different values
    const parameterSets = new Map<string, Array<{ command: GeneratedCliCommand; value: any }>>();

    for (const command of commands) {
      if (command.type === 'SET' && command.parameters) {
        for (const [paramName, value] of Object.entries(command.parameters)) {
          const key = `${command.targetFdn || 'unknown'}_${paramName}`;
          if (!parameterSets.has(key)) {
            parameterSets.set(key, []);
          }
          parameterSets.get(key)!.push({ command, value });
        }
      }
    }

    for (const [paramKey, settings] of parameterSets) {
      if (settings.length > 1) {
        const uniqueValues = new Set(settings.map(s => s.value));
        if (uniqueValues.size > 1) {
          conflicts.push({
            commandIds: settings.map(s => s.command.id),
            description: `Conflicting values for parameter ${paramKey}`,
            resolution: 'Use a single consistent value for this parameter'
          });
        }
      }
    }

    return conflicts;
  }

  private findExecutionOrderIssues(commands: GeneratedCliCommand[]): Array<{
    commandId: string;
    description: string;
    suggestion: string;
  }> {
    const issues: Array<{ commandId: string; description: string; suggestion: string }> = [];

    // Look for potential ordering issues
    const createCommands = commands.filter(cmd => cmd.type === 'CREATE');
    const setCommands = commands.filter(cmd => cmd.type === 'SET');

    for (const setCommand of setCommands) {
      const targetFdn = setCommand.targetFdn;
      if (targetFdn) {
        const relatedCreate = createCommands.find(create =>
          create.command.includes(targetFdn.split('=')[0])
        );

        if (relatedCreate) {
          // Check if SET comes before CREATE in the array
          const setIndex = commands.indexOf(setCommand);
          const createIndex = commands.indexOf(relatedCreate);

          if (setIndex < createIndex) {
            issues.push({
              commandId: setCommand.id,
              description: `SET command before CREATE for ${targetFdn}`,
              suggestion: 'Move CREATE command before SET operations'
            });
          }
        }
      }
    }

    return issues;
  }

  private normalizeOperation(command: GeneratedCliCommand): string {
    // Normalize command for duplicate detection
    return command.command
      .replace(/\d+/g, 'N') // Replace numbers with N
      .replace(/=\s*[^,\s]+/g, '=VALUE') // Replace parameter values
      .toLowerCase()
      .trim();
  }

  private calculateAverageComplexity(commands: GeneratedCliCommand[]): number {
    const complexityMap = { simple: 1, moderate: 2, complex: 3 };
    const total = commands.reduce((sum, cmd) =>
      sum + (complexityMap[cmd.metadata.complexity] || 1), 0);
    return total / commands.length;
  }

  private calculateRiskAssessment(commands: GeneratedCliCommand[]) {
    const assessment = { low: 0, medium: 0, high: 0, critical: 0 };

    for (const command of commands) {
      const risk = command.metadata.riskLevel;
      if (risk === 'critical') assessment.critical++;
      else if (risk === 'high') assessment.high++;
      else if (risk === 'medium') assessment.medium++;
      else assessment.low++;
    }

    return assessment;
  }

  /**
   * Initialize validation rules
   */
  private initializeValidationRules(): void {
    this.validationRules = [
      {
        name: 'Command syntax validation',
        type: 'syntax',
        severity: 'error',
        applicable: (cmd, ctx) => this.config.enableSyntaxValidation,
        validate: (cmd, ctx) => this.validateCommandSyntax(cmd)
      },
      {
        name: 'FDN path validation',
        type: 'semantic',
        severity: 'error',
        applicable: (cmd, ctx) => this.config.enableSemanticValidation && !!cmd.targetFdn,
        validate: (cmd, ctx) => this.validateFdnPath(cmd.targetFdn!)
      },
      {
        name: 'Parameter validation',
        type: 'constraint',
        severity: 'error',
        applicable: (cmd, ctx) => this.config.enableConstraintValidation && !!cmd.parameters,
        validate: (cmd, ctx) => this.validateCommandParameters(cmd)
      },
      {
        name: 'Performance validation',
        type: 'performance',
        severity: 'warning',
        applicable: (cmd, ctx) => this.config.enablePerformanceValidation,
        validate: (cmd, ctx) => this.validateCommandPerformance(cmd)
      }
    ];
  }

  /**
   * Initialize syntax patterns
   */
  private initializeSyntaxPatterns(): void {
    this.syntaxPatterns.set('cmedit_command', /^cmedit\s+(get|set|create|delete|monitor)\s+/i);
    this.syntaxPatterns.set('fdn_path', /^[A-Za-z][A-Za-z0-9_]*(=[^,]+)?(,[A-Za-z][A-Za-z0-9_]*(=[^,]+)?)*$/);
    this.syntaxPatterns.set('parameter_list', /^[a-zA-Z_][a-zA-Z0-9_]*=(.+)(,[a-zA-Z_][a-zA-Z0-9_]*=(.+))*$/);
  }

  /**
   * Initialize constraint validators
   */
  private initializeConstraintValidators(): void {
    // Add common parameter validators
    this.constraintValidators.set('qRxLevMin', new RangeValidator(-140, -44));
    this.constraintValidators.set('qQualMin', new RangeValidator(-34, -3));
    this.constraintValidators.set('hysteresis', new RangeValidator(0, 30));
    this.constraintValidators.set('timeToTrigger', new RangeValidator(0, 5000));
    this.constraintValidators.set('cellIndividualOffset', new RangeValidator(-24, 24));
  }

  /**
   * Validation rule implementations
   */
  private validateCommandSyntax(command: GeneratedCliCommand): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    const cmeditPattern = this.syntaxPatterns.get('cmedit_command');
    if (cmeditPattern && !cmeditPattern.test(command.command)) {
      errors.push('Invalid cmedit command syntax');
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }

  private validateCommandParameters(command: GeneratedCliCommand): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    if (command.parameters) {
      for (const [paramName, paramValue] of Object.entries(command.parameters)) {
        const validation = this.validateParameter(paramName, paramValue, command);
        if (!validation.isValid) {
          errors.push(...validation.errors);
        }
        warnings.push(...validation.warnings);
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }

  private validateCommandPerformance(command: GeneratedCliCommand): ValidationResult {
    const warnings: string[] = [];

    if (command.metadata.estimatedDuration > 10000) {
      warnings.push('Command has long estimated execution time');
    }

    if (command.command.includes('*') && command.type === 'GET') {
      warnings.push('Query may return large amount of data');
    }

    return {
      isValid: true,
      errors: [],
      warnings
    };
  }
}

/**
 * Constraint validator interface
 */
interface ConstraintValidator {
  validate(value: any): ValidationResult;
}

/**
 * Range validator
 */
class RangeValidator implements ConstraintValidator {
  constructor(private min: number, private max: number) {}

  validate(value: any): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    if (typeof value !== 'number') {
      errors.push('Value must be a number');
      return { isValid: false, errors, warnings };
    }

    if (value < this.min || value > this.max) {
      errors.push(`Value ${value} is outside range [${this.min}, ${this.max}]`);
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }
}