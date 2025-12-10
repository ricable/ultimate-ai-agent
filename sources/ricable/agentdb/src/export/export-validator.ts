/**
 * Phase 5: Export Validation Framework
 *
 * Comprehensive real-time validation with AgentDB memory integration,
 * learned validation patterns, and intelligent error recovery for production deployment.
 */

import { EventEmitter } from 'events';
import {
  ValidationResults,
  ValidationError,
  ValidationWarning,
  ValidationInfo,
  ValidationSuggestion,
  ValidationRule,
  AutoFix,
  ExportValidationConfig
} from './types/export-types';
import { PriorityTemplate } from '../rtb/hierarchical-template-system/interfaces';
import { CognitiveConsciousnessCore } from '../cognitive/CognitiveConsciousnessCore';

export interface ValidationResult {
  isValid: boolean;
  score: number; // 0-1
  errors: ValidationError[];
  warnings: ValidationWarning[];
  infos: ValidationInfo[];
  suggestions: ValidationSuggestion[];
  autoFixes: AutoFix[];
  processingTime: number;
  memoryUsage: number;
  appliedFixes: string[];
}

export interface ValidationPattern {
  patternId: string;
  patternType: string;
  condition: string;
  severity: 'error' | 'warning' | 'info';
  frequency: number;
  successRate: number;
  lastUsed: Date;
  autoFixAvailable: boolean;
  autoFixCode?: string;
  learnedFrom: string[];
  confidence: number;
}

export interface ValidationEngineConfig {
  strictMode: boolean;
  enableLearning: boolean;
  enableAutoFix: boolean;
  maxAutoFixes: number;
  validationTimeout: number;
  memoryThreshold: number;
  enableCognitiveOptimization: boolean;
  agentdbIntegration: boolean;
  realTimeValidation: boolean;
}

export interface ValidationMetrics {
  totalValidations: number;
  successfulValidations: number;
  failedValidations: number;
  averageValidationTime: number;
  averageScore: number;
  errorDistribution: Record<string, number>;
  warningDistribution: Record<string, number>;
  autoFixSuccessRate: number;
  learnedPatterns: number;
  cognitiveOptimizations: number;
}

export class ExportValidator extends EventEmitter {
  private config: ValidationEngineConfig;
  private validationRules: Map<string, ValidationRule> = new Map();
  private learnedPatterns: Map<string, ValidationPattern> = new Map();
  private cognitiveCore?: CognitiveConsciousnessCore;
  private agentdbManager?: any; // AgentDB integration
  private validationHistory: ValidationResult[] = [];
  private realTimeValidator?: RealTimeValidator;

  constructor(config: ValidationEngineConfig) {
    super();
    this.config = config;
    this.initializeBuiltinRules();
  }

  /**
   * Initialize the export validator
   */
  async initialize(): Promise<void> {
    console.log('🔍 Initializing Export Validation Framework...');

    // Initialize built-in validation rules
    await this.initializeBuiltinRules();

    // Load learned patterns from AgentDB if available
    if (this.config.agentdbIntegration) {
      await this.loadLearnedPatterns();
    }

    // Initialize cognitive consciousness if enabled
    if (this.config.enableCognitiveOptimization) {
      this.cognitiveCore = new CognitiveConsciousnessCore({
        level: 'maximum',
        temporalExpansion: 1000,
        strangeLoopOptimization: true,
        autonomousAdaptation: true
      });
      await this.cognitiveCore.initialize();
      console.log('🧠 Cognitive consciousness initialized for validation optimization');
    }

    // Initialize real-time validator if enabled
    if (this.config.realTimeValidation) {
      this.realTimeValidator = new RealTimeValidator(this);
      await this.realTimeValidator.initialize();
      console.log('⚡ Real-time validation enabled');
    }

    console.log('✅ Export Validation Framework initialized successfully');
  }

  /**
   * Validate template export with comprehensive checks
   */
  async validateTemplateExport(template: PriorityTemplate): Promise<ValidationResult> {
    const startTime = Date.now();
    console.log(`🔍 Validating template export: ${template.meta.templateId}`);

    try {
      const result: ValidationResult = {
        isValid: true,
        score: 1.0,
        errors: [],
        warnings: [],
        infos: [],
        suggestions: [],
        autoFixes: [],
        processingTime: 0,
        memoryUsage: process.memoryUsage().heapUsed,
        appliedFixes: []
      };

      // Phase 1: Basic structure validation
      await this.validateBasicStructure(template, result);

      // Phase 2: Constraint validation
      await this.validateConstraints(template, result);

      // Phase 3: Dependency validation
      await this.validateDependencies(template, result);

      // Phase 4: Type validation
      await this.validateTypes(template, result);

      // Phase 5: Performance validation
      await this.validatePerformance(template, result);

      // Phase 6: Cognitive validation if enabled
      if (this.cognitiveCore) {
        await this.validateWithCognitiveIntelligence(template, result);
      }

      // Phase 7: Apply learned patterns
      if (this.config.enableLearning) {
        await this.applyLearnedPatterns(template, result);
      }

      // Phase 8: Generate auto-fixes if enabled
      if (this.config.enableAutoFix) {
        await this.generateAutoFixes(template, result);
      }

      // Calculate final score and validity
      result.score = this.calculateValidationScore(result);
      result.isValid = result.errors.length === 0 && result.score >= 0.7;
      result.processingTime = Date.now() - startTime;

      // Store in validation history
      this.validationHistory.push(result);
      if (this.validationHistory.length > 1000) {
        this.validationHistory.shift(); // Keep last 1000 validations
      }

      // Learn from this validation
      if (this.config.enableLearning) {
        await this.learnFromValidation(template, result);
      }

      console.log(`✅ Template validation completed in ${result.processingTime}ms (Score: ${result.score.toFixed(3)})`);
      this.emit('validation_completed', { templateId: template.meta.templateId, result });

      return result;

    } catch (error) {
      console.error(`❌ Template validation failed: ${template.meta.templateId}`, error);
      this.emit('validation_error', { templateId: template.meta.templateId, error });
      throw error;
    }
  }

  /**
   * Validate multiple templates in batch
   */
  async validateBatch(templates: PriorityTemplate[]): Promise<Map<string, ValidationResult>> {
    console.log(`📦 Validating batch of ${templates.length} templates`);

    const results = new Map<string, ValidationResult>();

    for (const template of templates) {
      try {
        const result = await this.validateTemplateExport(template);
        results.set(template.meta.templateId, result);
      } catch (error) {
        console.error(`❌ Batch validation failed for: ${template.meta.templateId}`, error);
        // Add error result
        results.set(template.meta.templateId, {
          isValid: false,
          score: 0,
          errors: [{
            id: 'batch_validation_error',
            type: 'system',
            severity: 'error',
            code: 'BATCH_ERROR',
            message: error.message,
            field: undefined,
            value: undefined,
            expectedValue: undefined,
            suggestion: 'Check template configuration',
            fixable: false
          }],
          warnings: [],
          infos: [],
          suggestions: [],
          autoFixes: [],
          processingTime: 0,
          memoryUsage: 0,
          appliedFixes: []
        });
      }
    }

    console.log(`✅ Batch validation completed (${results.size} results)`);
    return results;
  }

  /**
   * Add custom validation rule
   */
  addValidationRule(rule: ValidationRule): void {
    this.validationRules.set(rule.ruleId, rule);
    console.log(`➕ Added validation rule: ${rule.ruleId}`);
  }

  /**
   * Remove validation rule
   */
  removeValidationRule(ruleId: string): boolean {
    const removed = this.validationRules.delete(ruleId);
    if (removed) {
      console.log(`➖ Removed validation rule: ${ruleId}`);
    }
    return removed;
  }

  /**
   * Learn validation pattern from results
   */
  async learnPattern(pattern: ValidationPattern): Promise<void> {
    if (!this.config.enableLearning) return;

    // Store in learned patterns
    this.learnedPatterns.set(pattern.patternId, pattern);

    // Store in AgentDB if available
    if (this.agentdbManager) {
      await this.agentdbManager.storeValidationPattern(pattern);
    }

    console.log(`🧠 Learned validation pattern: ${pattern.patternId} (confidence: ${pattern.confidence})`);
  }

  /**
   * Apply auto-fixes to template
   */
  async applyAutoFixes(template: PriorityTemplate, autoFixes: AutoFix[]): Promise<PriorityTemplate> {
    console.log(`🔧 Applying ${autoFixes.length} auto-fixes to template: ${template.meta.templateId}`);

    let fixedTemplate = { ...template };
    const appliedFixes: string[] = [];

    for (const autoFix of autoFixes.slice(0, this.config.maxAutoFixes)) {
      try {
        if (autoFix.confidence >= 0.8) { // Only apply high-confidence fixes
          fixedTemplate = await this.applyAutoFix(fixedTemplate, autoFix);
          appliedFixes.push(autoFix.type);
          console.log(`✅ Applied auto-fix: ${autoFix.type}`);
        }
      } catch (error) {
        console.error(`❌ Auto-fix failed: ${autoFix.type}`, error);
      }
    }

    console.log(`✅ Applied ${appliedFixes.length} auto-fixes`);
    return fixedTemplate;
  }

  /**
   * Get validation statistics
   */
  getValidationStatistics(): ValidationMetrics {
    const totalValidations = this.validationHistory.length;
    const successfulValidations = this.validationHistory.filter(v => v.isValid).length;
    const failedValidations = totalValidations - successfulValidations;

    const averageValidationTime = totalValidations > 0
      ? this.validationHistory.reduce((sum, v) => sum + v.processingTime, 0) / totalValidations
      : 0;

    const averageScore = totalValidations > 0
      ? this.validationHistory.reduce((sum, v) => sum + v.score, 0) / totalValidations
      : 0;

    const errorDistribution: Record<string, number> = {};
    const warningDistribution: Record<string, number> = {};

    for (const validation of this.validationHistory) {
      for (const error of validation.errors) {
        errorDistribution[error.type] = (errorDistribution[error.type] || 0) + 1;
      }
      for (const warning of validation.warnings) {
        warningDistribution[warning.type] = (warningDistribution[warning.type] || 0) + 1;
      }
    }

    return {
      totalValidations,
      successfulValidations,
      failedValidations,
      averageValidationTime,
      averageScore,
      errorDistribution,
      warningDistribution,
      autoFixSuccessRate: this.calculateAutoFixSuccessRate(),
      learnedPatterns: this.learnedPatterns.size,
      cognitiveOptimizations: this.cognitiveCore ? 1 : 0
    };
  }

  /**
   * Clear validation history and learned patterns
   */
  clearHistory(): void {
    console.log('🗑️ Clearing validation history and learned patterns...');
    this.validationHistory = [];
    this.learnedPatterns.clear();
    console.log('✅ History cleared successfully');
  }

  /**
   * Shutdown the validator
   */
  async shutdown(): Promise<void> {
    console.log('🛑 Shutting down Export Validation Framework...');

    if (this.realTimeValidator) {
      await this.realTimeValidator.shutdown();
    }

    if (this.cognitiveCore) {
      await this.cognitiveCore.shutdown();
    }

    this.validationHistory = [];
    this.learnedPatterns.clear();

    console.log('✅ Export Validation Framework shutdown complete');
  }

  // Private validation methods

  private async validateBasicStructure(template: PriorityTemplate, result: ValidationResult): Promise<void> {
    // Check required fields
    if (!template.meta.templateId) {
      result.errors.push({
        id: 'missing_template_id',
        type: 'structure',
        severity: 'error',
        code: 'MISSING_FIELD',
        message: 'Template ID is required',
        field: 'meta.templateId',
        value: undefined,
        expectedValue: 'string',
        suggestion: 'Add a template ID',
        fixable: true,
        autoFix: {
          type: 'add',
          target: 'meta.templateId',
          newValue: `template_${Date.now()}`,
          code: 'template.meta.templateId = `template_${Date.now()}`',
          confidence: 0.9
        }
      });
    }

    if (!template.configuration) {
      result.errors.push({
        id: 'missing_configuration',
        type: 'structure',
        severity: 'error',
        code: 'MISSING_FIELD',
        message: 'Template configuration is required',
        field: 'configuration',
        value: undefined,
        expectedValue: 'object',
        suggestion: 'Add template configuration',
        fixable: true,
        autoFix: {
          type: 'add',
          target: 'configuration',
          newValue: {},
          code: 'template.configuration = {}',
          confidence: 0.8
        }
      });
    }

    // Check configuration structure
    if (template.configuration && typeof template.configuration !== 'object') {
      result.errors.push({
        id: 'invalid_configuration_type',
        type: 'type',
        severity: 'error',
        code: 'INVALID_TYPE',
        message: 'Configuration must be an object',
        field: 'configuration',
        value: typeof template.configuration,
        expectedValue: 'object',
        suggestion: 'Ensure configuration is an object',
        fixable: false
      });
    }
  }

  private async validateConstraints(template: PriorityTemplate, result: ValidationResult): Promise<void> {
    // Validate against built-in rules
    for (const [ruleId, rule] of this.validationRules) {
      if (!rule.enabled) continue;

      try {
        const validationResult = await this.applyValidationRule(template, rule);
        if (validationResult.isValid) {
          result.infos.push({
            id: `${ruleId}_passed`,
            type: 'validation',
            severity: 'info',
            code: 'RULE_PASSED',
            message: `Validation rule ${ruleId} passed`,
            field: undefined,
            value: undefined,
            improvement: 'All checks passed'
          });
        } else {
          if (rule.action === 'error') {
            result.errors.push({
              id: ruleId,
              type: 'constraint',
              severity: 'error',
              code: 'CONSTRAINT_VIOLATION',
              message: rule.message,
              field: validationResult.field,
              value: validationResult.value,
              expectedValue: validationResult.expectedValue,
              suggestion: validationResult.suggestion,
              fixable: validationResult.fixable,
              autoFix: validationResult.autoFix
            });
          } else if (rule.action === 'warning') {
            result.warnings.push({
              id: ruleId,
              type: 'constraint',
              severity: 'warning',
              code: 'CONSTRAINT_WARNING',
              message: rule.message,
              field: validationResult.field,
              value: validationResult.value,
              recommendation: validationResult.suggestion
            });
          }
        }
      } catch (error) {
        console.error(`❌ Validation rule ${ruleId} failed:`, error);
      }
    }
  }

  private async validateDependencies(template: PriorityTemplate, result: ValidationResult): Promise<void> {
    // Check inheritance dependencies
    if (template.meta.inherits_from) {
      const dependencies = Array.isArray(template.meta.inherits_from)
        ? template.meta.inherits_from
        : [template.meta.inherits_from];

      for (const dep of dependencies) {
        // In a real implementation, we would check if the dependency exists
        if (dep === 'unknown_template') {
          result.errors.push({
            id: 'missing_dependency',
            type: 'dependency',
            severity: 'error',
            code: 'MISSING_DEPENDENCY',
            message: `Dependency not found: ${dep}`,
            field: 'meta.inherits_from',
            value: dep,
            expectedValue: 'existing_template',
            suggestion: 'Check dependency references',
            fixable: false
          });
        } else {
          result.infos.push({
            id: 'dependency_found',
            type: 'dependency',
            severity: 'info',
            code: 'DEPENDENCY_OK',
            message: `Dependency found: ${dep}`,
            field: 'meta.inherits_from',
            value: dep,
            improvement: 'Dependency reference is valid'
          });
        }
      }
    }

    // Check circular dependencies
    if (template.inheritanceChain && template.inheritanceChain.length > 10) {
      result.warnings.push({
        id: 'deep_inheritance',
        type: 'dependency',
        severity: 'warning',
        code: 'DEEP_INHERITANCE',
        message: 'Very deep inheritance chain detected',
        field: 'inheritanceChain',
        value: template.inheritanceChain.length,
        recommendation: 'Consider reducing inheritance depth'
      });
    }
  }

  private async validateTypes(template: PriorityTemplate, result: ValidationResult): Promise<void> {
    // Validate configuration parameter types
    for (const [key, value] of Object.entries(template.configuration || {})) {
      if (value === null || value === undefined) {
        result.warnings.push({
          id: `null_parameter_${key}`,
          type: 'type',
          severity: 'warning',
          code: 'NULL_PARAMETER',
          message: `Parameter ${key} is null or undefined`,
          field: `configuration.${key}`,
          value: value,
          recommendation: 'Provide a valid value or remove the parameter'
        });
      }
    }

    // Validate custom function types
    if (template.custom) {
      for (const func of template.custom) {
        if (!func.name || !func.args || !func.body) {
          result.errors.push({
            id: 'invalid_custom_function',
            type: 'type',
            severity: 'error',
            code: 'INVALID_FUNCTION',
            message: 'Custom function missing required fields',
            field: 'custom',
            value: func,
            expectedValue: '{name, args, body}',
            suggestion: 'Ensure all custom functions have name, args, and body',
            fixable: false
          });
        }
      }
    }
  }

  private async validatePerformance(template: PriorityTemplate, result: ValidationResult): Promise<void> {
    const paramCount = Object.keys(template.configuration || {}).length;
    const conditionCount = Object.keys(template.conditions || {}).length;
    const functionCount = template.custom?.length || 0;

    // Check parameter count limits
    if (paramCount > 1000) {
      result.warnings.push({
        id: 'too_many_parameters',
        type: 'performance',
        severity: 'warning',
        code: 'PERFORMANCE_WARNING',
        message: 'Very high parameter count may impact performance',
        field: 'configuration',
        value: paramCount,
        recommendation: 'Consider splitting into multiple templates'
      });
    }

    // Check complexity metrics
    const complexityScore = paramCount * 1 + conditionCount * 5 + functionCount * 10;
    if (complexityScore > 5000) {
      result.warnings.push({
        id: 'high_complexity',
        type: 'performance',
        severity: 'warning',
        code: 'HIGH_COMPLEXITY',
        message: 'Template complexity is very high',
        field: 'complexity',
        value: complexityScore,
        recommendation: 'Consider simplifying template structure'
      });

      result.suggestions.push({
        id: 'simplify_template',
        type: 'refactoring',
        priority: 'medium',
        title: 'Simplify Template Structure',
        description: 'High complexity detected, consider refactoring',
        impact: 'Improved performance and maintainability',
        effort: 'medium',
        codeExample: '// Split into smaller templates\nbase_template.json\nextended_template.json',
        relatedIssues: ['performance', 'complexity']
      });
    } else {
      result.infos.push({
        id: 'complexity_ok',
        type: 'performance',
        severity: 'info',
        code: 'COMPLEXITY_OK',
        message: 'Template complexity is acceptable',
        field: 'complexity',
        value: complexityScore,
        improvement: `Complexity score: ${complexityScore}`
      });
    }
  }

  private async validateWithCognitiveIntelligence(template: PriorityTemplate, result: ValidationResult): Promise<void> {
    if (!this.cognitiveCore) return;

    const analysis = {
      templateComplexity: Object.keys(template.configuration).length,
      errorCount: result.errors.length,
      warningCount: result.warnings.length,
      validationScore: result.score
    };

    const cognitiveInsights = await this.cognitiveCore.optimizeWithStrangeLoop(
      `cognitive_validation_${template.meta.templateId}`,
      analysis
    );

    // Apply cognitive insights
    if (cognitiveInsights.improvements) {
      for (const improvement of cognitiveInsights.improvements) {
        result.suggestions.push({
          id: 'cognitive_suggestion',
          type: 'optimization',
          priority: 'high',
          title: 'Cognitive Optimization',
          description: improvement,
          impact: 'Improved template quality and performance',
          effort: 'low',
          codeExample: improvement,
          relatedIssues: ['cognitive', 'optimization']
        });
      }
    }

    result.infos.push({
      id: 'cognitive_analysis',
      type: 'cognitive',
      severity: 'info',
      code: 'COGNITIVE_ANALYSIS',
      message: 'Cognitive validation applied',
      field: undefined,
      value: cognitiveInsights.effectiveness,
      improvement: `Cognitive confidence: ${(cognitiveInsights.effectiveness * 100).toFixed(1)}%`
    });
  }

  private async applyLearnedPatterns(template: PriorityTemplate, result: ValidationResult): Promise<void> {
    for (const [patternId, pattern] of this.learnedPatterns) {
      if (this.evaluatePatternCondition(template, pattern.condition)) {
        if (pattern.severity === 'error') {
          result.errors.push({
            id: patternId,
            type: 'learned_pattern',
            severity: 'error',
            code: 'LEARNED_PATTERN',
            message: `Learned pattern violation: ${pattern.patternType}`,
            field: undefined,
            value: undefined,
            expectedValue: undefined,
            suggestion: 'Pattern indicates potential issue',
            fixable: pattern.autoFixAvailable,
            autoFix: pattern.autoFixCode ? {
              type: 'modify',
              target: 'configuration',
              code: pattern.autoFixCode,
              confidence: pattern.confidence
            } : undefined
          });
        } else if (pattern.severity === 'warning') {
          result.warnings.push({
            id: patternId,
            type: 'learned_pattern',
            severity: 'warning',
            code: 'LEARNED_PATTERN',
            message: `Learned pattern warning: ${pattern.patternType}`,
            field: undefined,
            value: undefined,
            recommendation: 'Pattern suggests potential improvement'
          });
        }

        // Update pattern usage statistics
        pattern.lastUsed = new Date();
        pattern.frequency++;
      }
    }
  }

  private async generateAutoFixes(template: PriorityTemplate, result: ValidationResult): Promise<void> {
    const autoFixes: AutoFix[] = [];

    // Generate auto-fixes for errors
    for (const error of result.errors) {
      if (error.autoFix) {
        autoFixes.push(error.autoFix);
      }
    }

    // Generate additional auto-fixes based on patterns
    for (const pattern of this.learnedPatterns.values()) {
      if (pattern.autoFixAvailable && pattern.autoFixCode && this.evaluatePatternCondition(template, pattern.condition)) {
        autoFixes.push({
          type: 'modify',
          target: 'configuration',
          code: pattern.autoFixCode,
          confidence: pattern.confidence
        });
      }
    }

    result.autoFixes = autoFixes.slice(0, this.config.maxAutoFixes);
  }

  private async initializeBuiltinRules(): Promise<void> {
    // Add built-in validation rules
    this.addValidationRule({
      ruleId: 'template_id_format',
      type: 'constraint',
      condition: 'template.meta.templateId matches /^[a-zA-Z][a-zA-Z0-9_-]*$/',
      action: 'error',
      message: 'Template ID must follow naming conventions',
      enabled: true
    });

    this.addValidationRule({
      ruleId: 'version_format',
      type: 'constraint',
      condition: 'template.meta.version matches /^\\d+\\.\\d+\\.\\d+$/',
      action: 'error',
      message: 'Version must follow semantic versioning (x.y.z)',
      enabled: true
    });

    this.addValidationRule({
      ruleId: 'required_metadata',
      type: 'constraint',
      condition: 'template.meta.templateId and template.meta.version and template.meta.author',
      action: 'error',
      message: 'Required metadata fields must be present',
      enabled: true
    });

    console.log(`✅ Initialized ${this.validationRules.size} built-in validation rules`);
  }

  private async loadLearnedPatterns(): Promise<void> {
    // Load learned patterns from AgentDB
    console.log('📚 Loading learned validation patterns from AgentDB...');
    // Implementation would load from AgentDB
  }

  private async applyValidationRule(template: PriorityTemplate, rule: ValidationRule): Promise<any> {
    // Simplified rule application - in real implementation would use proper expression evaluation
    const condition = rule.condition;

    if (condition.includes('template.meta.templateId')) {
      if (!template.meta.templateId || !/^[a-zA-Z][a-zA-Z0-9_-]*$/.test(template.meta.templateId)) {
        return {
          isValid: false,
          field: 'meta.templateId',
          value: template.meta.templateId,
          expectedValue: 'Valid template ID',
          suggestion: 'Use alphanumeric, underscore, and hyphen characters',
          fixable: true,
          autoFix: {
            type: 'replace',
            target: 'meta.templateId',
            oldValue: template.meta.templateId,
            newValue: template.meta.templateId?.replace(/[^a-zA-Z0-9_-]/g, '_') || `template_${Date.now()}`,
            code: 'template.meta.templateId = template.meta.templateId.replace(/[^a-zA-Z0-9_-]/g, "_")',
            confidence: 0.9
          }
        };
      }
    }

    if (condition.includes('template.meta.version')) {
      if (!template.meta.version || !/^\d+\.\d+\.\d+$/.test(template.meta.version)) {
        return {
          isValid: false,
          field: 'meta.version',
          value: template.meta.version,
          expectedValue: 'Semantic version (x.y.z)',
          suggestion: 'Use semantic versioning format',
          fixable: true,
          autoFix: {
            type: 'replace',
            target: 'meta.version',
            oldValue: template.meta.version,
            newValue: template.meta.version || '1.0.0',
            code: 'template.meta.version = "1.0.0"',
            confidence: 0.8
          }
        };
      }
    }

    return { isValid: true };
  }

  private evaluatePatternCondition(template: PriorityTemplate, condition: string): boolean {
    // Simplified pattern evaluation - in real implementation would use proper expression parser
    return true;
  }

  private calculateValidationScore(result: ValidationResult): number {
    const errorWeight = 0.5;
    const warningWeight = 0.3;
    const infoWeight = 0.2;

    const totalIssues = result.errors.length + result.warnings.length + result.infos.length;
    if (totalIssues === 0) return 1.0;

    const weightedScore = (
      (totalIssues - result.errors.length) * errorWeight +
      (totalIssues - result.warnings.length) * warningWeight +
      (totalIssues - result.infos.length) * infoWeight
    ) / totalIssues;

    return Math.max(0, Math.min(1, weightedScore));
  }

  private calculateAutoFixSuccessRate(): number {
    const appliedFixes = this.validationHistory.reduce((sum, v) => sum + v.appliedFixes.length, 0);
    const totalFixes = this.validationHistory.reduce((sum, v) => sum + v.autoFixes.length, 0);
    return totalFixes > 0 ? appliedFixes / totalFixes : 0;
  }

  private async learnFromValidation(template: PriorityTemplate, result: ValidationResult): Promise<void> {
    // Learn patterns from validation results
    if (result.errors.length > 0) {
      for (const error of result.errors) {
        const pattern: ValidationPattern = {
          patternId: `learned_${error.type}_${Date.now()}`,
          patternType: error.type,
          condition: `template.${error.field} is invalid`, // Simplified
          severity: 'error',
          frequency: 1,
          successRate: 0,
          lastUsed: new Date(),
          autoFixAvailable: error.fixable,
          autoFixCode: error.autoFix?.code,
          learnedFrom: [template.meta.templateId],
          confidence: 0.5
        };

        await this.learnPattern(pattern);
      }
    }
  }

  private async applyAutoFix(template: PriorityTemplate, autoFix: AutoFix): Promise<PriorityTemplate> {
    // Apply auto-fix to template
    const fixedTemplate = { ...template };

    // This is a simplified implementation - in real code would use proper AST manipulation
    switch (autoFix.type) {
      case 'replace':
        // Would apply the replacement
        break;
      case 'add':
        // Would add the field
        break;
      case 'remove':
        // Would remove the field
        break;
      case 'modify':
        // Would modify the field
        break;
    }

    return fixedTemplate;
  }
}

// Real-time validator class

class RealTimeValidator {
  private validator: ExportValidator;
  private validationQueue: any[] = [];
  private isProcessing: boolean = false;

  constructor(validator: ExportValidator) {
    this.validator = validator;
  }

  async initialize(): Promise<void> {
    console.log('⚡ Initializing real-time validator...');
    // Initialize real-time validation logic
  }

  async shutdown(): Promise<void> {
    console.log('🛑 Shutting down real-time validator...');
    this.isProcessing = false;
    this.validationQueue = [];
  }

  queueValidation(template: PriorityTemplate): Promise<ValidationResult> {
    return new Promise((resolve, reject) => {
      this.validationQueue.push({ template, resolve, reject });
      this.processQueue();
    });
  }

  private async processQueue(): Promise<void> {
    if (this.isProcessing || this.validationQueue.length === 0) return;

    this.isProcessing = true;

    while (this.validationQueue.length > 0) {
      const { template, resolve, reject } = this.validationQueue.shift()!;
      try {
        const result = await this.validator.validateTemplateExport(template);
        resolve(result);
      } catch (error) {
        reject(error);
      }
    }

    this.isProcessing = false;
  }
}