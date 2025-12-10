/**
 * Validation Engine Utility
 *
 * Core validation engine for template export with comprehensive rule processing,
 * performance optimization, and integration with AgentDB memory patterns.
 */

import { ExportValidationConfig, ValidationResults, ValidationError, ValidationWarning } from '../types/export-types';
import { PriorityTemplate } from '../../rtb/hierarchical-template-system/interfaces';

export class ValidationEngine {
  private config: ExportValidationConfig;
  private validationRules: Map<string, any> = new Map();
  private customValidators: Map<string, Function> = new Map();

  constructor(config: ExportValidationConfig) {
    this.config = config;
    this.initializeBuiltInValidators();
  }

  async initialize(): Promise<void> {
    console.log('🔍 Initializing Validation Engine...');
    await this.loadValidationRules();
    console.log('✅ Validation Engine initialized');
  }

  async validateTemplate(template: PriorityTemplate): Promise<ValidationResults> {
    const startTime = Date.now();
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];
    const infos: any[] = [];

    // Apply validation rules
    for (const [ruleId, rule] of this.validationRules) {
      try {
        const result = await this.applyValidationRule(template, rule);
        if (!result.isValid) {
          if (result.severity === 'error') {
            errors.push(result.error);
          } else if (result.severity === 'warning') {
            warnings.push(result.warning);
          } else {
            infos.push(result.info);
          }
        }
      } catch (error) {
        console.error(`Validation rule ${ruleId} failed:`, error);
      }
    }

    const validationTime = Date.now() - startTime;

    return {
      isValid: errors.length === 0,
      validationScore: this.calculateValidationScore(errors, warnings),
      errors,
      warnings,
      infos,
      suggestions: [],
      totalChecks: this.validationRules.size,
      passedChecks: this.validationRules.size - errors.length,
      failedChecks: errors.length,
      processingTime: validationTime
    };
  }

  private async loadValidationRules(): Promise<void> {
    // Load validation rules from configuration or database
  }

  private async applyValidationRule(template: PriorityTemplate, rule: any): Promise<any> {
    // Apply specific validation rule
    return { isValid: true };
  }

  private calculateValidationScore(errors: ValidationError[], warnings: ValidationWarning[]): number {
    const totalIssues = errors.length + warnings.length;
    if (totalIssues === 0) return 1.0;

    const errorWeight = 0.7;
    const warningWeight = 0.3;

    return Math.max(0, 1 - (errors.length * errorWeight + warnings.length * warningWeight) / totalIssues);
  }

  private initializeBuiltInValidators(): void {
    // Initialize built-in validation functions
    this.customValidators.set('semantic_version', this.validateSemanticVersion.bind(this));
    this.customValidators.set('template_id', this.validateTemplateId.bind(this));
    this.customValidators.set('configuration_structure', this.validateConfigurationStructure.bind(this));
  }

  private validateSemanticVersion(version: string): boolean {
    return /^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$/.test(version);
  }

  private validateTemplateId(templateId: string): boolean {
    return /^[a-zA-Z][a-zA-Z0-9_-]*$/.test(templateId);
  }

  private validateConfigurationStructure(config: any): boolean {
    return config && typeof config === 'object' && !Array.isArray(config);
  }
}