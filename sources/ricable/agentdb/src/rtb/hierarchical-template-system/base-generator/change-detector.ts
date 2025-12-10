import { RTBParameter, RTBTemplate, TemplateMeta } from '../../types/rtb-types';
import { GeneratedTemplate } from './template-generator';

export interface ParameterChange {
  parameterId: string;
  changeType: 'added' | 'removed' | 'modified' | 'type_changed' | 'constraint_added' | 'constraint_removed' | 'constraint_modified';
  oldValue?: any;
  newValue?: any;
  oldType?: string;
  newType?: string;
  oldConstraints?: any;
  newConstraints?: any;
  impact: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  affectedTemplates: string[];
}

export interface TemplateChange {
  templateId: string;
  changeType: 'version_bump' | 'parameter_added' | 'parameter_removed' | 'parameter_modified' | 'function_added' | 'function_removed' | 'function_modified' | 'metadata_updated';
  changes: ParameterChange[];
  versionChange: VersionChange;
  impact: 'low' | 'medium' | 'high' | 'critical';
  requiresRegeneration: boolean;
  breakingChange: boolean;
  description: string;
}

export interface VersionChange {
  oldVersion: string;
  newVersion: string;
  changeType: 'patch' | 'minor' | 'major';
  reason: string;
  semverCompliant: boolean;
}

export interface ChangeSet {
  id: string;
  timestamp: Date;
  description: string;
  parameterChanges: ParameterChange[];
  templateChanges: TemplateChange[];
  summary: ChangeSetSummary;
  affectedTemplates: string[];
  recommendedActions: string[];
}

export interface ChangeSetSummary {
  totalChanges: number;
  changesByType: Record<string, number>;
  changesByImpact: Record<string, number>;
  breakingChanges: number;
  templatesAffected: number;
  parametersAffected: number;
}

export interface VersionHistory {
  templateId: string;
  versions: TemplateVersion[];
  currentVersion: string;
  latestVersion: TemplateVersion;
}

export interface TemplateVersion {
  version: string;
  timestamp: Date;
  parameters: RTBParameter[];
  template: RTBTemplate;
  metadata: TemplateMetadata;
  changeSetId?: string;
  isStable: boolean;
  downloadCount?: number;
  lastUsed?: Date;
}

export interface TemplateMetadata {
  generatedAt: Date;
  sourceFiles: string[];
  parameterCount: number;
  moClassCount: number;
  generationTime: number;
  optimizationApplied: string[];
  warnings: string[];
  checksum: string;
  gitCommit?: string;
  author?: string;
}

export interface ChangeDetectionConfig {
  strictComparison: boolean;
  detectConstraintChanges: boolean;
  detectTypeChanges: boolean;
  detectMetadataChanges: boolean;
  generateSemanticVersions: boolean;
  trackParameterHistory: boolean;
  maxHistoryVersions: number;
  impactAnalysisEnabled: boolean;
}

export class ParameterChangeDetector {
  private config: ChangeDetectionConfig;
  private versionHistory: Map<string, VersionHistory> = new Map();
  private parameterHistory: Map<string, ParameterHistory[]> = new Map();
  private changeSets: ChangeSet[] = [];
  private checksumCache: Map<string, string> = new Map();

  constructor(config: ChangeDetectionConfig = {
    strictComparison: true,
    detectConstraintChanges: true,
    detectTypeChanges: true,
    detectMetadataChanges: true,
    generateSemanticVersions: true,
    trackParameterHistory: true,
    maxHistoryVersions: 10,
    impactAnalysisEnabled: true
  }) {
    this.config = config;
  }

  /**
   * Detect changes between old and new templates
   */
  async detectChanges(
    oldTemplates: GeneratedTemplate[],
    newTemplates: GeneratedTemplate[],
    changeSetDescription: string = 'Template update'
  ): Promise<ChangeSet> {
    console.log('🔍 Detecting changes between template versions...');

    const changeSetId = this.generateChangeSetId();
    const timestamp = new Date();
    const parameterChanges: ParameterChange[] = [];
    const templateChanges: TemplateChange[] = [];
    const affectedTemplates = new Set<string>();

    // Build lookup maps
    const oldTemplateMap = new Map(oldTemplates.map(t => [t.templateId, t]));
    const newTemplateMap = new Map(newTemplates.map(t => [t.templateId, t]));

    // Find all template IDs
    const allTemplateIds = new Set([...oldTemplateMap.keys(), ...newTemplateMap.keys()]);

    for (const templateId of allTemplateIds) {
      const oldTemplate = oldTemplateMap.get(templateId);
      const newTemplate = newTemplateMap.get(templateId);

      if (oldTemplate && newTemplate) {
        // Template exists in both versions - detect modifications
        const templateChange = await this.detectTemplateModifications(oldTemplate, newTemplate);
        if (templateChange) {
          templateChanges.push(templateChange);
          parameterChanges.push(...templateChange.changes);
          affectedTemplates.add(templateId);
        }
      } else if (!oldTemplate && newTemplate) {
        // New template added
        const templateChange = this.createAddedTemplateChange(newTemplate);
        templateChanges.push(templateChange);
        parameterChanges.push(...templateChange.changes);
        affectedTemplates.add(templateId);
      } else if (oldTemplate && !newTemplate) {
        // Template removed
        const templateChange = this.createRemovedTemplateChange(oldTemplate);
        templateChanges.push(templateChange);
        parameterChanges.push(...templateChange.changes);
        affectedTemplates.add(templateId);
      }
    }

    // Analyze impact and generate recommendations
    const summary = this.generateChangeSummary(parameterChanges, templateChanges);
    const recommendedActions = this.generateRecommendedActions(templateChanges);

    const changeSet: ChangeSet = {
      id: changeSetId,
      timestamp,
      description: changeSetDescription,
      parameterChanges,
      templateChanges,
      summary,
      affectedTemplates: Array.from(affectedTemplates),
      recommendedActions
    };

    // Store change set
    this.changeSets.push(changeSet);

    console.log(`✅ Change detection complete: ${parameterChanges.length} parameter changes, ${templateChanges.length} template changes`);

    return changeSet;
  }

  /**
   * Detect modifications between two templates
   */
  private async detectTemplateModifications(
    oldTemplate: GeneratedTemplate,
    newTemplate: GeneratedTemplate
  ): Promise<TemplateChange | null> {
    const changes: ParameterChange[] = [];

    // Compare parameters
    const parameterChanges = await this.detectParameterChanges(
      oldTemplate.parameters,
      newTemplate.parameters,
      oldTemplate.templateId
    );
    changes.push(...parameterChanges);

    // Compare custom functions
    const functionChanges = this.detectFunctionChanges(oldTemplate, newTemplate);
    if (functionChanges.length > 0) {
      changes.push(...functionChanges);
    }

    // Compare metadata
    const metadataChanges = this.detectMetadataChanges(oldTemplate, newTemplate);
    if (metadataChanges.length > 0) {
      changes.push(...metadataChanges);
    }

    if (changes.length === 0) {
      return null; // No changes detected
    }

    // Determine version change
    const versionChange = this.calculateVersionChange(oldTemplate, newTemplate, changes);

    // Determine impact
    const impact = this.calculateChangeImpact(changes);
    const breakingChange = changes.some(c => c.impact === 'critical' || c.changeType.includes('removed'));

    return {
      templateId: oldTemplate.templateId,
      changeType: this.determineTemplateChangeType(changes),
      changes,
      versionChange,
      impact,
      requiresRegeneration: true,
      breakingChange,
      description: this.generateTemplateChangeDescription(oldTemplate.templateId, changes)
    };
  }

  /**
   * Detect parameter changes between old and new parameter sets
   */
  private async detectParameterChanges(
    oldParameters: RTBParameter[],
    newParameters: RTBParameter[],
    templateId: string
  ): Promise<ParameterChange[]> {
    const changes: ParameterChange[] = [];

    // Build parameter maps
    const oldParamMap = new Map(oldParameters.map(p => [p.id, p]));
    const newParamMap = new Map(newParameters.map(p => [p.id, p]));

    // Find all parameter IDs
    const allParamIds = new Set([...oldParamMap.keys(), ...newParamMap.keys()]);

    for (const paramId of allParamIds) {
      const oldParam = oldParamMap.get(paramId);
      const newParam = newParamMap.get(paramId);

      if (oldParam && newParam) {
        // Parameter exists in both versions - detect modifications
        const paramChanges = this.detectParameterModifications(oldParam, newParam);
        changes.push(...paramChanges);
      } else if (!oldParam && newParam) {
        // New parameter added
        changes.push(this.createAddedParameterChange(newParam, templateId));
      } else if (oldParam && !newParam) {
        // Parameter removed
        changes.push(this.createRemovedParameterChange(oldParam, templateId));
      }
    }

    return changes;
  }

  /**
   * Detect modifications between two parameters
   */
  private detectParameterModifications(
    oldParam: RTBParameter,
    newParam: RTBParameter
  ): ParameterChange[] {
    const changes: ParameterChange[] = [];

    // Check for type changes
    if (this.config.detectTypeChanges && oldParam.type !== newParam.type) {
      changes.push({
        parameterId: oldParam.id,
        changeType: 'type_changed',
        oldValue: oldParam.type,
        newValue: newParam.type,
        impact: this.calculateTypeChangeImpact(oldParam.type, newParam.type),
        description: `Parameter type changed from ${oldParam.type} to ${newParam.type}`,
        affectedTemplates: []
      });
    }

    // Check for default value changes
    if (!this.deepEqual(oldParam.defaultValue, newParam.defaultValue)) {
      changes.push({
        parameterId: oldParam.id,
        changeType: 'modified',
        oldValue: oldParam.defaultValue,
        newValue: newParam.defaultValue,
        impact: this.calculateValueChangeImpact(oldParam.defaultValue, newParam.defaultValue, oldParam.type),
        description: `Default value changed from ${oldParam.defaultValue} to ${newParam.defaultValue}`,
        affectedTemplates: []
      });
    }

    // Check for constraint changes
    if (this.config.detectConstraintChanges) {
      const constraintChanges = this.detectConstraintModifications(oldParam, newParam);
      changes.push(...constraintChanges);
    }

    // Check for description changes
    if (this.config.detectMetadataChanges && oldParam.description !== newParam.description) {
      changes.push({
        parameterId: oldParam.id,
        changeType: 'modified',
        oldValue: oldParam.description,
        newValue: newParam.description,
        impact: 'low',
        description: `Parameter description updated`,
        affectedTemplates: []
      });
    }

    return changes;
  }

  /**
   * Detect constraint modifications between parameters
   */
  private detectConstraintModifications(
    oldParam: RTBParameter,
    newParam: RTBParameter
  ): ParameterChange[] {
    const changes: ParameterChange[] = [];

    const oldConstraints = this.normalizeConstraints(oldParam.constraints);
    const newConstraints = this.normalizeConstraints(newParam.constraints);

    // Check for added constraints
    for (const [type, newConstraint] of newConstraints) {
      if (!oldConstraints.has(type)) {
        changes.push({
          parameterId: oldParam.id,
          changeType: 'constraint_added',
          newValue: newConstraint,
          impact: this.calculateConstraintImpact('added', type, newConstraint),
          description: `Constraint ${type} added`,
          affectedTemplates: []
        });
      } else {
        // Check for modified constraints
        const oldConstraint = oldConstraints.get(type)!;
        if (!this.deepEqual(oldConstraint, newConstraint)) {
          changes.push({
            parameterId: oldParam.id,
            changeType: 'constraint_modified',
            oldValue: oldConstraint,
            newValue: newConstraint,
            impact: this.calculateConstraintImpact('modified', type, newConstraint),
            description: `Constraint ${type} modified`,
            affectedTemplates: []
          });
        }
      }
    }

    // Check for removed constraints
    for (const [type, oldConstraint] of oldConstraints) {
      if (!newConstraints.has(type)) {
        changes.push({
          parameterId: oldParam.id,
          changeType: 'constraint_removed',
          oldValue: oldConstraint,
          impact: this.calculateConstraintImpact('removed', type, oldConstraint),
          description: `Constraint ${type} removed`,
          affectedTemplates: []
        });
      }
    }

    return changes;
  }

  /**
   * Detect function changes between templates
   */
  private detectFunctionChanges(
    oldTemplate: GeneratedTemplate,
    newTemplate: GeneratedTemplate
  ): ParameterChange[] {
    const changes: ParameterChange[] = [];

    const oldFunctions = new Map(
      (oldTemplate.template.custom || []).map(f => [f.name, f])
    );
    const newFunctions = new Map(
      (newTemplate.template.custom || []).map(f => [f.name, f])
    );

    // Find all function names
    const allFunctionNames = new Set([...oldFunctions.keys(), ...newFunctions.keys()]);

    for (const functionName of allFunctionNames) {
      const oldFunction = oldFunctions.get(functionName);
      const newFunction = newFunctions.get(functionName);

      if (oldFunction && newFunction) {
        // Compare function bodies
        const oldBody = oldFunction.body.join('\n');
        const newBody = newFunction.body.join('\n');

        if (oldBody !== newBody) {
          changes.push({
            parameterId: `${oldTemplate.templateId}.function.${functionName}`,
            changeType: 'modified',
            oldValue: oldBody,
            newValue: newBody,
            impact: 'medium',
            description: `Custom function ${functionName} modified`,
            affectedTemplates: [oldTemplate.templateId]
          });
        }
      } else if (!oldFunction && newFunction) {
        // Function added
        changes.push({
          parameterId: `${oldTemplate.templateId}.function.${functionName}`,
          changeType: 'added',
          newValue: newFunction.body.join('\n'),
          impact: 'medium',
          description: `Custom function ${functionName} added`,
          affectedTemplates: [oldTemplate.templateId]
        });
      } else if (oldFunction && !newFunction) {
        // Function removed
        changes.push({
          parameterId: `${oldTemplate.templateId}.function.${functionName}`,
          changeType: 'removed',
          oldValue: oldFunction.body.join('\n'),
          impact: 'high',
          description: `Custom function ${functionName} removed`,
          affectedTemplates: [oldTemplate.templateId]
        });
      }
    }

    return changes;
  }

  /**
   * Detect metadata changes between templates
   */
  private detectMetadataChanges(
    oldTemplate: GeneratedTemplate,
    newTemplate: GeneratedTemplate
  ): ParameterChange[] {
    const changes: ParameterChange[] = [];

    // Compare template metadata
    if (!this.deepEqual(oldTemplate.metadata, newTemplate.metadata)) {
      changes.push({
        parameterId: `${oldTemplate.templateId}.metadata`,
        changeType: 'modified',
        oldValue: oldTemplate.metadata,
        newValue: newTemplate.metadata,
        impact: 'low',
        description: 'Template metadata updated',
        affectedTemplates: [oldTemplate.templateId]
      });
    }

    return changes;
  }

  /**
   * Calculate version change based on detected changes
   */
  private calculateVersionChange(
    oldTemplate: GeneratedTemplate,
    newTemplate: GeneratedTemplate,
    changes: ParameterChange[]
  ): VersionChange {
    const oldVersion = oldTemplate.template.meta?.version || '1.0.0';
    let newVersion = oldVersion;
    let changeType: 'patch' | 'minor' | 'major' = 'patch';
    let reason = 'No breaking changes detected';

    if (this.config.generateSemanticVersions) {
      const criticalChanges = changes.filter(c => c.impact === 'critical');
      const removalChanges = changes.filter(c => c.changeType.includes('removed'));
      const typeChanges = changes.filter(c => c.changeType === 'type_changed');

      if (criticalChanges.length > 0 || removalChanges.length > 0 || typeChanges.length > 0) {
        // Major version bump for breaking changes
        changeType = 'major';
        newVersion = this.bumpVersion(oldVersion, 'major');
        reason = 'Breaking changes detected';
      } else if (changes.some(c => c.changeType === 'added')) {
        // Minor version bump for new features
        changeType = 'minor';
        newVersion = this.bumpVersion(oldVersion, 'minor');
        reason = 'New features added';
      } else {
        // Patch version bump for fixes
        changeType = 'patch';
        newVersion = this.bumpVersion(oldVersion, 'patch');
        reason = 'Bug fixes and improvements';
      }
    }

    return {
      oldVersion,
      newVersion,
      changeType,
      reason,
      semverCompliant: this.isSemverCompliant(newVersion)
    };
  }

  /**
   * Create added template change
   */
  private createAddedTemplateChange(template: GeneratedTemplate): TemplateChange {
    const changes: ParameterChange[] = template.parameters.map(param => ({
      parameterId: param.id,
      changeType: 'added' as const,
      newValue: param,
      impact: 'medium' as const,
      description: `New parameter ${param.name} added`,
      affectedTemplates: [template.templateId]
    }));

    return {
      templateId: template.templateId,
      changeType: 'parameter_added',
      changes,
      versionChange: {
        oldVersion: '0.0.0',
        newVersion: '1.0.0',
        changeType: 'minor',
        reason: 'New template added',
        semverCompliant: true
      },
      impact: 'medium',
      requiresRegeneration: true,
      breakingChange: false,
      description: `New template ${template.templateId} added with ${template.parameters.length} parameters`
    };
  }

  /**
   * Create removed template change
   */
  private createRemovedTemplateChange(template: GeneratedTemplate): TemplateChange {
    const changes: ParameterChange[] = template.parameters.map(param => ({
      parameterId: param.id,
      changeType: 'removed' as const,
      oldValue: param,
      impact: 'high' as const,
      description: `Parameter ${param.name} removed`,
      affectedTemplates: [template.templateId]
    }));

    return {
      templateId: template.templateId,
      changeType: 'parameter_removed',
      changes,
      versionChange: {
        oldVersion: template.template.meta?.version || '1.0.0',
        newVersion: '2.0.0',
        changeType: 'major',
        reason: 'Template removed',
        semverCompliant: true
      },
      impact: 'high',
      requiresRegeneration: false,
      breakingChange: true,
      description: `Template ${template.templateId} removed`
    };
  }

  /**
   * Create added parameter change
   */
  private createAddedParameterChange(parameter: RTBParameter, templateId: string): ParameterChange {
    return {
      parameterId: parameter.id,
      changeType: 'added',
      newValue: parameter,
      impact: this.calculateParameterAdditionImpact(parameter),
      description: `New parameter ${parameter.name} added`,
      affectedTemplates: [templateId]
    };
  }

  /**
   * Create removed parameter change
   */
  private createRemovedParameterChange(parameter: RTBParameter, templateId: string): ParameterChange {
    return {
      parameterId: parameter.id,
      changeType: 'removed',
      oldValue: parameter,
      impact: this.calculateParameterRemovalImpact(parameter),
      description: `Parameter ${parameter.name} removed`,
      affectedTemplates: [templateId]
    };
  }

  /**
   * Calculate change impact
   */
  private calculateChangeImpact(changes: ParameterChange[]): 'low' | 'medium' | 'high' | 'critical' {
    if (changes.some(c => c.impact === 'critical')) return 'critical';
    if (changes.some(c => c.impact === 'high')) return 'high';
    if (changes.some(c => c.impact === 'medium')) return 'medium';
    return 'low';
  }

  /**
   * Calculate type change impact
   */
  private calculateTypeChangeImpact(oldType: string, newType: string): 'low' | 'medium' | 'high' | 'critical' {
    const compatibilityMatrix: Record<string, Record<string, string>> = {
      'string': { 'number': 'critical', 'boolean': 'medium', 'object': 'critical' },
      'number': { 'string': 'high', 'boolean': 'medium', 'object': 'critical' },
      'boolean': { 'string': 'low', 'number': 'medium', 'object': 'high' },
      'object': { 'string': 'critical', 'number': 'critical', 'boolean': 'high' }
    };

    return (compatibilityMatrix[oldType]?.[newType] as any) || 'medium';
  }

  /**
   * Calculate value change impact
   */
  private calculateValueChangeImpact(oldValue: any, newValue: any, type: string): 'low' | 'medium' | 'high' | 'critical' {
    if (oldValue === undefined && newValue !== undefined) return 'low';
    if (oldValue !== undefined && newValue === undefined) return 'high';
    if (type === 'boolean') return 'medium';
    if (type === 'number' && Math.abs((newValue || 0) - (oldValue || 0)) > 100) return 'medium';
    return 'low';
  }

  /**
   * Calculate constraint impact
   */
  private calculateConstraintImpact(
    changeType: 'added' | 'removed' | 'modified',
    constraintType: string,
    constraint: any
  ): 'low' | 'medium' | 'high' | 'critical' {
    if (changeType === 'removed') return 'high';
    if (constraintType === 'enum') return 'medium';
    if (constraintType === 'range') return 'medium';
    return 'low';
  }

  /**
   * Calculate parameter addition impact
   */
  private calculateParameterAdditionImpact(parameter: RTBParameter): 'low' | 'medium' | 'high' | 'critical' {
    if (parameter.constraints && Array.isArray(parameter.constraints) && parameter.constraints.length > 0) {
      return 'medium';
    }
    return 'low';
  }

  /**
   * Calculate parameter removal impact
   */
  private calculateParameterRemovalImpact(parameter: RTBParameter): 'low' | 'medium' | 'high' | 'critical' {
    if (parameter.defaultValue !== undefined) {
      return 'medium';
    }
    return 'high';
  }

  /**
   * Generate change set ID
   */
  private generateChangeSetId(): string {
    return `changeset_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Generate change summary
   */
  private generateChangeSummary(parameterChanges: ParameterChange[], templateChanges: TemplateChange[]): ChangeSetSummary {
    const changesByType: Record<string, number> = {};
    const changesByImpact: Record<string, number> = {};

    for (const change of parameterChanges) {
      changesByType[change.changeType] = (changesByType[change.changeType] || 0) + 1;
      changesByImpact[change.impact] = (changesByImpact[change.impact] || 0) + 1;
    }

    const breakingChanges = templateChanges.filter(t => t.breakingChange).length;
    const templatesAffected = templateChanges.length;
    const parametersAffected = new Set(parameterChanges.map(c => c.parameterId)).size;

    return {
      totalChanges: parameterChanges.length,
      changesByType,
      changesByImpact,
      breakingChanges,
      templatesAffected,
      parametersAffected
    };
  }

  /**
   * Generate recommended actions
   */
  private generateRecommendedActions(templateChanges: TemplateChange[]): string[] {
    const actions: string[] = [];

    const breakingChanges = templateChanges.filter(t => t.breakingChange);
    if (breakingChanges.length > 0) {
      actions.push('Review breaking changes and update dependent templates');
    }

    const highImpactChanges = templateChanges.filter(t => t.impact === 'high' || t.impact === 'critical');
    if (highImpactChanges.length > 0) {
      actions.push('Test high-impact changes in development environment');
    }

    const versionChanges = templateChanges.filter(t => t.versionChange.changeType === 'major');
    if (versionChanges.length > 0) {
      actions.push('Update version compatibility documentation');
    }

    if (templateChanges.length > 10) {
      actions.push('Consider incremental rollout for large change sets');
    }

    return actions;
  }

  /**
   * Determine template change type
   */
  private determineTemplateChangeType(changes: ParameterChange[]): string {
    if (changes.some(c => c.changeType.includes('removed'))) return 'parameter_removed';
    if (changes.some(c => c.changeType === 'added')) return 'parameter_added';
    if (changes.some(c => c.changeType === 'modified')) return 'parameter_modified';
    return 'metadata_updated';
  }

  /**
   * Generate template change description
   */
  private generateTemplateChangeDescription(templateId: string, changes: ParameterChange[]): string {
    const changeCount = changes.length;
    const changeTypes = [...new Set(changes.map(c => c.changeType))];
    return `Template ${templateId} updated with ${changeCount} changes: ${changeTypes.join(', ')}`;
  }

  /**
   * Bump semantic version
   */
  private bumpVersion(version: string, type: 'patch' | 'minor' | 'major'): string {
    const parts = version.split('.').map(Number);

    if (type === 'patch') {
      parts[2] = (parts[2] || 0) + 1;
    } else if (type === 'minor') {
      parts[1] = (parts[1] || 0) + 1;
      parts[2] = 0;
    } else if (type === 'major') {
      parts[0] = (parts[0] || 1) + 1;
      parts[1] = 0;
      parts[2] = 0;
    }

    return parts.join('.');
  }

  /**
   * Check if version is semver compliant
   */
  private isSemverCompliant(version: string): boolean {
    return /^\d+\.\d+\.\d+$/.test(version);
  }

  /**
   * Normalize constraints to map format
   */
  private normalizeConstraints(constraints?: ConstraintSpec[] | Record<string, any>): Map<string, any> {
    const map = new Map<string, any>();

    if (!constraints) return map;

    if (Array.isArray(constraints)) {
      for (const constraint of constraints) {
        map.set(constraint.type, constraint.value);
      }
    } else if (typeof constraints === 'object') {
      for (const [key, value] of Object.entries(constraints)) {
        map.set(key, value);
      }
    }

    return map;
  }

  /**
   * Deep equality check
   */
  private deepEqual(a: any, b: any): boolean {
    if (a === b) return true;
    if (a == null || b == null) return false;
    if (typeof a !== typeof b) return false;

    if (typeof a === 'object') {
      const keysA = Object.keys(a);
      const keysB = Object.keys(b);
      if (keysA.length !== keysB.length) return false;

      for (const key of keysA) {
        if (!keysB.includes(key) || !this.deepEqual(a[key], b[key])) {
          return false;
        }
      }
      return true;
    }

    return false;
  }

  /**
   * Get change history for a template
   */
  getTemplateHistory(templateId: string): VersionHistory | null {
    return this.versionHistory.get(templateId) || null;
  }

  /**
   * Get all change sets
   */
  getChangeSets(): ChangeSet[] {
    return [...this.changeSets];
  }

  /**
   * Get change sets by date range
   */
  getChangeSetsByDateRange(startDate: Date, endDate: Date): ChangeSet[] {
    return this.changeSets.filter(cs => cs.timestamp >= startDate && cs.timestamp <= endDate);
  }
}

interface ParameterHistory {
  parameterId: string;
  versions: {
    timestamp: Date;
    value: any;
    type: string;
    constraints?: any;
  }[];
}

// Import constraint types for type checking
interface ConstraintSpec {
  type: string;
  value: any;
  errorMessage?: string;
  severity: string;
}