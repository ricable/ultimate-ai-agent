/**
 * Phase 5: RTB Integration Layer
 *
 * Integration with existing RTB hierarchical template system and AgentDB memory patterns
 * for seamless Phase 5 export system integration with production deployment.
 */

import { IPriorityTemplateEngine, PriorityTemplate, TemplatePriority } from '../../rtb/hierarchical-template-system/interfaces';
import { TemplateExporter, ExportValidator, VariantGenerator } from '../index';
import { ExportConfig, ExportResult, ValidationResults, VariantGenerationResult } from '../types/export-types';

export interface RTBIntegrationConfig {
  templateEngine: IPriorityTemplateEngine;
  exportConfig: ExportConfig;
  enableRealTimeExport: boolean;
  enableAutoVariantGeneration: boolean;
  syncWithAgentDB: boolean;
  cacheTemplates: boolean;
  performanceMonitoring: boolean;
}

export interface ExportWorkflow {
  workflowId: string;
  templateIds: string[];
  exportConfig: Partial<ExportConfig>;
  variantGeneration?: boolean;
  validationStrictness: 'lenient' | 'strict' | 'very_strict';
  autoApprove?: boolean;
  notificationTargets?: string[];
}

export interface IntegratedExportResult {
  workflowId: string;
  templateResults: Map<string, ExportResult>;
  variantResults?: Map<string, VariantGenerationResult>;
  workflowMetrics: WorkflowMetrics;
  recommendations: ExportRecommendation[];
  errors: IntegrationError[];
}

export interface WorkflowMetrics {
  startTime: Date;
  endTime: Date;
  totalProcessingTime: number;
  templatesProcessed: number;
  variantsGenerated: number;
  successRate: number;
  averageExportTime: number;
  cacheHitRate: number;
  cognitiveOptimizations: number;
}

export interface ExportRecommendation {
  type: 'performance' | 'validation' | 'optimization' | 'automation';
  priority: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  templateIds: string[];
  impact: string;
  effort: 'low' | 'medium' | 'high';
  actionable: boolean;
  estimatedBenefit: string;
}

export interface IntegrationError {
  errorId: string;
  type: 'template_error' | 'export_error' | 'validation_error' | 'integration_error';
  templateId?: string;
  workflowId?: string;
  message: string;
  details?: any;
  recoverable: boolean;
  recoveryAction?: string;
  timestamp: Date;
}

export class RTBIntegrationManager {
  private config: RTBIntegrationConfig;
  private templateEngine: IPriorityTemplateEngine;
  private templateExporter: TemplateExporter;
  private exportValidator: ExportValidator;
  private variantGenerator?: VariantGenerator;
  private activeWorkflows: Map<string, WorkflowMetrics> = new Map();
  private exportHistory: Map<string, IntegratedExportResult> = new Map();
  private templateCache: Map<string, PriorityTemplate> = new Map();

  constructor(config: RTBIntegrationConfig) {
    this.config = config;
    this.templateEngine = config.templateEngine;
  }

  async initialize(): Promise<void> {
    console.log('🔗 Initializing RTB Integration Manager...');

    // Initialize export system components
    this.templateExporter = new TemplateExporter({
      defaultExportConfig: this.config.exportConfig,
      validationConfig: {
        strictMode: false,
        validateConstraints: true,
        validateDependencies: true,
        validateTypes: true,
        validateInheritance: true,
        validatePerformance: true,
        maxProcessingTime: 5000,
        maxMemoryUsage: 512 * 1024 * 1024,
        allowedViolations: [],
        customValidators: []
      },
      cacheConfig: {
        enabled: true,
        maxSize: 1000,
        ttl: 30 * 60 * 1000,
        evictionPolicy: 'lru',
        compressionEnabled: false,
        compressionLevel: 6,
        keyPrefix: 'rtb_export_'
      },
      cognitiveConfig: {
        level: 'maximum',
        temporalExpansion: 1000,
        strangeLoopOptimization: true,
        autonomousAdaptation: true
      },
      performanceMonitoring: this.config.performanceMonitoring,
      parallelProcessing: true,
      maxConcurrency: 8
    });

    await this.templateExporter.initialize();

    this.exportValidator = new ExportValidator({
      strictMode: false,
      enableLearning: true,
      enableAutoFix: true,
      maxAutoFixes: 5,
      validationTimeout: 5000,
      memoryThreshold: 512 * 1024 * 1024,
      enableCognitiveOptimization: true,
      agentdbIntegration: this.config.syncWithAgentDB,
      realTimeValidation: this.config.enableRealTimeExport
    });

    await this.exportValidator.initialize();

    if (this.config.enableAutoVariantGeneration) {
      this.variantGenerator = new VariantGenerator({
        enableCognitiveOptimization: true,
        enableParallelGeneration: true,
        maxConcurrency: 4,
        validationStrictness: 'strict',
        performanceOptimization: true,
        cachingEnabled: true,
        cacheSize: 100,
        includeDocumentation: true,
        generateExamples: true
      });

      await this.variantGenerator.initialize();
    }

    console.log('✅ RTB Integration Manager initialized successfully');
  }

  /**
   * Export templates from RTB system with full integration
   */
  async exportTemplatesFromRTB(
    templateIds: string[],
    exportConfig?: Partial<ExportConfig>
  ): Promise<IntegratedExportResult> {
    const workflowId = this.generateWorkflowId();
    const startTime = new Date();

    console.log(`🔄 Starting RTB export workflow: ${workflowId} (${templateIds.length} templates)`);

    const workflowMetrics: WorkflowMetrics = {
      workflowId,
      startTime,
      endTime: new Date(),
      totalProcessingTime: 0,
      templatesProcessed: 0,
      variantsGenerated: 0,
      successRate: 0,
      averageExportTime: 0,
      cacheHitRate: 0,
      cognitiveOptimizations: 0
    };

    this.activeWorkflows.set(workflowId, workflowMetrics);

    try {
      const templateResults = new Map<string, ExportResult>();
      const variantResults = new Map<string, VariantGenerationResult>();
      const errors: IntegrationError[] = [];
      const recommendations: ExportRecommendation[] = [];

      // Phase 1: Load templates from RTB system
      const templates = await this.loadTemplatesFromRTB(templateIds);

      // Phase 2: Validate templates
      const validationResults = await this.validateTemplates(templates);

      // Phase 3: Export templates
      for (const [templateId, template] of templates.entries()) {
        try {
          const result = await this.templateExporter.exportTemplate(template, exportConfig);
          templateResults.set(templateId, result);
          workflowMetrics.templatesProcessed++;

          // Generate variants if enabled
          if (this.config.enableAutoVariantGeneration && this.variantGenerator) {
            const variantResult = await this.variantGenerator.generateAllVariants(template);
            variantResults.set(templateId, variantResult);
            workflowMetrics.variantsGenerated += variantResult.generatedVariants.length;
          }

        } catch (error) {
          errors.push({
            errorId: `export_${templateId}_${Date.now()}`,
            type: 'export_error',
            templateId,
            workflowId,
            message: error.message,
            details: error,
            recoverable: true,
            recoveryAction: 'Retry export with simplified configuration',
            timestamp: new Date()
          });
        }
      }

      // Phase 4: Generate recommendations
      recommendations.push(...this.generateRecommendations(templates, templateResults, validationResults));

      // Phase 5: Update metrics
      const endTime = new Date();
      workflowMetrics.endTime = endTime;
      workflowMetrics.totalProcessingTime = endTime.getTime() - startTime.getTime();
      workflowMetrics.successRate = templateResults.size / templates.size;
      workflowMetrics.averageExportTime = Array.from(templateResults.values())
        .reduce((sum, r) => sum + r.performanceMetrics.templateProcessingTime, 0) /
        Math.max(templateResults.size, 1);

      const result: IntegratedExportResult = {
        workflowId,
        templateResults,
        variantResults: variantResults.size > 0 ? variantResults : undefined,
        workflowMetrics,
        recommendations,
        errors
      };

      // Store in history
      this.exportHistory.set(workflowId, result);

      // Remove from active workflows
      this.activeWorkflows.delete(workflowId);

      console.log(`✅ RTB export workflow completed: ${workflowId} in ${workflowMetrics.totalProcessingTime}ms`);
      return result;

    } catch (error) {
      console.error(`❌ RTB export workflow failed: ${workflowId}`, error);
      this.activeWorkflows.delete(workflowId);
      throw error;
    }
  }

  /**
   * Execute predefined export workflow
   */
  async executeExportWorkflow(workflow: ExportWorkflow): Promise<IntegratedExportResult> {
    console.log(`🔧 Executing export workflow: ${workflow.workflowId}`);

    // Prepare export configuration
    const exportConfig = {
      ...this.config.exportConfig,
      ...workflow.exportConfig,
      includeValidation: true,
      includeDocumentation: true
    };

    // Execute the export
    const result = await this.exportTemplatesFromRTB(workflow.templateIds, exportConfig);

    // Apply auto-approval if configured
    if (workflow.autoApprove && result.errors.length === 0) {
      console.log(`✅ Auto-approved export workflow: ${workflow.workflowId}`);
      // Would trigger deployment or further processing
    }

    // Send notifications if targets are configured
    if (workflow.notificationTargets && workflow.notificationTargets.length > 0) {
      await this.sendNotifications(workflow, result);
    }

    return result;
  }

  /**
   * Get template from RTB system with caching
   */
  async getTemplateFromRTB(templateId: string): Promise<PriorityTemplate | null> {
    // Check cache first
    if (this.config.cacheTemplates) {
      const cached = this.templateCache.get(templateId);
      if (cached) {
        return cached;
      }
    }

    try {
      const template = await this.templateEngine.getTemplate(templateId);
      if (template) {
        // Cache the template
        if (this.config.cacheTemplates) {
          this.templateCache.set(templateId, template);
        }
        return template;
      }
      return null;
    } catch (error) {
      console.error(`❌ Failed to get template ${templateId} from RTB:`, error);
      return null;
    }
  }

  /**
   * Real-time template export (if enabled)
   */
  async enableRealTimeExport(): Promise<void> {
    if (!this.config.enableRealTimeExport) {
      console.log('Real-time export is disabled');
      return;
    }

    console.log('🔄 Enabling real-time template export...');

    // Listen to RTB template events
    // This would integrate with the RTB event system
    // For now, simulate real-time export

    setInterval(async () => {
      try {
        // Check for recently updated templates
        const recentTemplateIds = await this.getRecentlyUpdatedTemplates();

        if (recentTemplateIds.length > 0) {
          console.log(`🔄 Real-time export: ${recentTemplateIds.length} templates updated`);
          await this.exportTemplatesFromRTB(recentTemplateIds);
        }
      } catch (error) {
        console.error('❌ Real-time export failed:', error);
      }
    }, 30000); // Check every 30 seconds
  }

  /**
   * Get integration statistics
   */
  async getIntegrationStatistics(): Promise<any> {
    const activeWorkflows = this.activeWorkflows.size;
    const completedWorkflows = this.exportHistory.size;
    const cacheSize = this.templateCache.size;

    const exporterStats = await this.templateExporter.getExportStatus();
    const validatorStats = this.exportValidator.getValidationStatistics();
    const variantStats = this.variantGenerator?.getGenerationStatistics();

    return {
      activeWorkflows,
      completedWorkflows,
      cacheSize,
      exporter: exporterStats,
      validator: validatorStats,
      variantGenerator: variantStats,
      recommendations: this.generateSystemRecommendations()
    };
  }

  /**
   * Shutdown the integration manager
   */
  async shutdown(): Promise<void> {
    console.log('🛑 Shutting down RTB Integration Manager...');

    // Wait for active workflows to complete
    if (this.activeWorkflows.size > 0) {
      console.log(`⏳ Waiting for ${this.activeWorkflows.size} active workflows...`);
      // Would wait for workflows to complete
    }

    // Shutdown components
    await this.templateExporter.shutdown();
    await this.exportValidator.shutdown();
    if (this.variantGenerator) {
      await this.variantGenerator.shutdown();
    }

    // Clear caches
    this.templateCache.clear();
    this.exportHistory.clear();
    this.activeWorkflows.clear();

    console.log('✅ RTB Integration Manager shutdown complete');
  }

  // Private helper methods

  private async loadTemplatesFromRTB(templateIds: string[]): Promise<Map<string, PriorityTemplate>> {
    const templates = new Map<string, PriorityTemplate>();

    for (const templateId of templateIds) {
      const template = await this.getTemplateFromRTB(templateId);
      if (template) {
        templates.set(templateId, template);
      } else {
        console.warn(`⚠️ Template not found: ${templateId}`);
      }
    }

    return templates;
  }

  private async validateTemplates(templates: Map<string, PriorityTemplate>): Promise<Map<string, ValidationResults>> {
    const validationResults = new Map<string, ValidationResults>();

    for (const [templateId, template] of templates.entries()) {
      try {
        const result = await this.exportValidator.validateTemplateExport(template);
        validationResults.set(templateId, result);
      } catch (error) {
        console.error(`❌ Validation failed for template ${templateId}:`, error);
      }
    }

    return validationResults;
  }

  private generateRecommendations(
    templates: Map<string, PriorityTemplate>,
    exportResults: Map<string, ExportResult>,
    validationResults: Map<string, ValidationResults>
  ): ExportRecommendation[] {
    const recommendations: ExportRecommendation[] = [];

    // Performance recommendations
    const averageExportTime = Array.from(exportResults.values())
      .reduce((sum, r) => sum + r.performanceMetrics.templateProcessingTime, 0) /
      Math.max(exportResults.size, 1);

    if (averageExportTime > 1000) {
      recommendations.push({
        type: 'performance',
        priority: 'medium',
        title: 'Optimize Export Performance',
        description: `Average export time (${averageExportTime}ms) exceeds target of 1000ms`,
        templateIds: Array.from(exportResults.keys()),
        impact: 'Faster template processing and improved user experience',
        effort: 'medium',
        actionable: true,
        estimatedBenefit: '30-50% reduction in export time'
      });
    }

    // Validation recommendations
    let totalErrors = 0;
    let totalWarnings = 0;

    for (const validation of validationResults.values()) {
      totalErrors += validation.errors.length;
      totalWarnings += validation.warnings.length;
    }

    if (totalErrors > 0) {
      recommendations.push({
        type: 'validation',
        priority: 'high',
        title: 'Fix Template Validation Errors',
        description: `${totalErrors} validation errors found across templates`,
        templateIds: Array.from(validationResults.keys()),
        impact: 'Improved template quality and reliability',
        effort: 'medium',
        actionable: true,
        estimatedBenefit: '100% validation compliance'
      });
    }

    if (totalWarnings > 5) {
      recommendations.push({
        type: 'validation',
        priority: 'low',
        title: 'Address Template Validation Warnings',
        description: `${totalWarnings} validation warnings found across templates`,
        templateIds: Array.from(validationResults.keys()),
        impact: 'Better template compliance and best practices',
        effort: 'low',
        actionable: true,
        estimatedBenefit: 'Improved template quality'
      });
    }

    // Optimization recommendations
    const complexTemplates = Array.from(templates.entries())
      .filter(([_, template]) => Object.keys(template.configuration).length > 100)
      .map(([id, _]) => id);

    if (complexTemplates.length > 0) {
      recommendations.push({
        type: 'optimization',
        priority: 'medium',
        title: 'Simplify Complex Templates',
        description: `${complexTemplates.length} templates have >100 parameters`,
        templateIds: complexTemplates,
        impact: 'Improved maintainability and performance',
        effort: 'high',
        actionable: true,
        estimatedBenefit: '20-30% performance improvement'
      });
    }

    return recommendations;
  }

  private generateSystemRecommendations(): ExportRecommendation[] {
    const recommendations: ExportRecommendation[] = [];

    // Cache recommendations
    if (this.templateCache.size > 500) {
      recommendations.push({
        type: 'performance',
        priority: 'low',
        title: 'Optimize Template Cache',
        description: 'Template cache is getting large, consider cleaning up old entries',
        templateIds: [],
        impact: 'Reduced memory usage',
        effort: 'low',
        actionable: true,
        estimatedBenefit: '10-20% memory reduction'
      });
    }

    return recommendations;
  }

  private async getRecentlyUpdatedTemplates(): Promise<string[]> {
    // In a real implementation, would query RTB system for recently updated templates
    // For now, return empty array
    return [];
  }

  private async sendNotifications(workflow: ExportWorkflow, result: IntegratedExportResult): Promise<void> {
    console.log(`📧 Sending notifications for workflow: ${workflow.workflowId}`);

    for (const target of workflow.notificationTargets) {
      console.log(`   Notifying: ${target}`);
      // In a real implementation, would send email, Slack, etc.
    }
  }

  private generateWorkflowId(): string {
    return `workflow_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}