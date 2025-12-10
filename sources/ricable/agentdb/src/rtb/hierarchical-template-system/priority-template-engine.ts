/**
 * Priority Template Engine - Core Implementation
 *
 * Implements the main hierarchical template inheritance engine with priority-based
 * conflict resolution, template merging, and variant generation capabilities.
 *
 * Features:
 * - Priority-based template inheritance (0-80 priority levels)
 * - Intelligent conflict resolution with multiple strategies
 * - Template caching and performance optimization
 * - Comprehensive validation and error handling
 * - Event-driven processing with metrics
 */

import {
  IPriorityTemplateEngine,
  PriorityTemplate,
  TemplatePriority,
  TemplateInheritanceChain,
  TemplateConflict,
  TemplateWarning,
  ConflictResolutionStrategy,
  VariantGenerationConfig,
  TemplateValidationResult,
  TemplateFilter,
  TemplateProcessingEvent,
  ITemplateEventBus,
  TemplateProcessingMetrics,
  HierarchicalTemplateEngineConfig,
  TemplateSystemError,
  TemplateInheritanceError,
  TemplateValidationError,
  TemplateConflictError
} from './interfaces';

import {
  ITemplateVariantGenerator,
  ITemplateMerger,
  ITemplateConflictResolver,
  ITemplateValidator,
  TemplateEventBus,
  TemplateVariantGenerator,
  TemplateMerger,
  TemplateConflictResolver,
  TemplateValidator
} from './components';

import { LRUCache } from 'lru-cache';

/**
 * Main Priority Template Engine implementation
 */
export class PriorityTemplateEngine implements IPriorityTemplateEngine {
  private templates: Map<string, PriorityTemplate> = new Map();
  private inheritanceCache: LRUCache<string, TemplateInheritanceChain>;
  private metrics: Map<string, TemplateProcessingMetrics> = new Map();
  private eventBus: ITemplateEventBus;
  private config: HierarchicalTemplateEngineConfig;

  // Component dependencies
  private variantGenerator: ITemplateVariantGenerator;
  private templateMerger: ITemplateMerger;
  private conflictResolver: ITemplateConflictResolver;
  private validator: ITemplateValidator;

  constructor(config?: Partial<HierarchicalTemplateEngineConfig>) {
    this.config = {
      cachingEnabled: true,
      maxCacheSize: 1000,
      defaultConflictStrategy: ConflictResolutionStrategy.HIGHEST_PRIORITY_WINS,
      parallelProcessing: true,
      maxConcurrentOperations: 10,
      validationStrictness: 'strict',
      performanceMonitoring: true,
      detailedLogging: false,
      ...config
    };

    // Initialize LRU cache for inheritance chains
    this.inheritanceCache = new LRUCache<string, TemplateInheritanceChain>({
      max: this.config.maxCacheSize,
      ttl: 1000 * 60 * 15, // 15 minutes TTL
      updateAgeOnGet: true
    });

    // Initialize event bus
    this.eventBus = new TemplateEventBus();

    // Initialize components
    this.initializeComponents();
  }

  /**
   * Register a new template with the engine
   */
  async registerTemplate(template: PriorityTemplate): Promise<void> {
    const startTime = Date.now();

    try {
      // Validate template before registration
      const validationResult = await this.validateTemplate(template);
      if (!validationResult.isValid) {
        throw new TemplateValidationError(
          `Template validation failed: ${validationResult.errors.map(e => e.message).join(', ')}`,
          template.meta.version,
          validationResult.errors
        );
      }

      // Check for existing template
      if (this.templates.has(template.meta.version)) {
        console.warn(`[PriorityTemplateEngine] Overriding existing template: ${template.meta.version}`);
      }

      // Store template
      this.templates.set(template.meta.version, template);

      // Clear related cache entries
      this.clearCacheForTemplate(template.meta.version);

      // Record metrics
      const processingTime = Date.now() - startTime;
      this.recordMetrics(template.meta.version, processingTime);

      // Emit event
      await this.emitEvent({
        eventType: 'template_registered',
        templateId: template.meta.version,
        timestamp: new Date(),
        data: { template, validationResult },
        processingTime
      });

      console.log(`[PriorityTemplateEngine] Registered template: ${template.meta.version} (Priority: ${template.priority})`);

    } catch (error) {
      await this.emitEvent({
        eventType: 'template_registered',
        templateId: template.meta.version,
        timestamp: new Date(),
        data: { template, error },
        error: error as Error
      });

      throw error;
    }
  }

  /**
   * Resolve template inheritance chain
   */
  async resolveInheritance(templateId: string): Promise<TemplateInheritanceChain> {
    const startTime = Date.now();

    // Check cache first
    if (this.config.cachingEnabled && this.inheritanceCache.has(templateId)) {
      const cached = this.inheritanceCache.get(templateId)!;
      this.updateCacheMetrics(templateId, true);
      return cached;
    }

    try {
      const template = await this.getTemplate(templateId);
      if (!template) {
        throw new TemplateSystemError(`Template not found: ${templateId}`, 'TEMPLATE_NOT_FOUND', templateId);
      }

      // Build inheritance chain
      const chain = await this.buildInheritanceChain(template);

      // Resolve conflicts
      chain.conflicts = await this.conflictResolver.detectConflicts(
        chain.chain.map(link => this.templates.get(link.templateId)!).filter(Boolean)
      );

      // Resolve detected conflicts
      if (chain.conflicts.length > 0) {
        chain.conflicts = await Promise.all(
          chain.conflicts.map(conflict =>
            this.conflictResolver.resolveConflict(conflict, template.conflictResolution || this.config.defaultConflictStrategy)
          )
        );
      }

      // Cache the result
      if (this.config.cachingEnabled) {
        this.inheritanceCache.set(templateId, chain);
      }

      // Record metrics
      const processingTime = Date.now() - startTime;
      this.recordMetrics(templateId, processingTime);

      // Emit event
      await this.emitEvent({
        eventType: 'template_resolved',
        templateId,
        timestamp: new Date(),
        data: { chain, conflictCount: chain.conflicts.length },
        processingTime
      });

      this.updateCacheMetrics(templateId, false);
      return chain;

    } catch (error) {
      await this.emitEvent({
        eventType: 'template_resolved',
        templateId,
        timestamp: new Date(),
        data: { error },
        error: error as Error
      });

      throw error;
    }
  }

  /**
   * Merge multiple templates with conflict resolution
   */
  async mergeTemplates(
    templateIds: string[],
    strategy: ConflictResolutionStrategy = this.config.defaultConflictStrategy
  ): Promise<PriorityTemplate> {
    const startTime = Date.now();

    try {
      // Get all templates
      const templates = await Promise.all(
        templateIds.map(id => this.getTemplate(id))
      );

      const validTemplates = templates.filter(Boolean) as PriorityTemplate[];

      if (validTemplates.length === 0) {
        throw new TemplateSystemError('No valid templates found for merging', 'NO_TEMPLATES_FOUND');
      }

      // Validate all templates
      for (const template of validTemplates) {
        const validation = await this.validateTemplate(template);
        if (!validation.isValid) {
          throw new TemplateValidationError(
            `Template ${template.meta.version} validation failed`,
            template.meta.version,
            validation.errors
          );
        }
      }

      // Merge templates
      const mergedTemplate = await this.templateMerger.merge(validTemplates, strategy);

      // Cache the merged template
      const mergedId = `merged_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      await this.registerTemplate({
        ...mergedTemplate,
        meta: {
          ...mergedTemplate.meta,
          version: mergedId,
          description: `Merged template from ${templateIds.join(', ')}`
        }
      });

      // Record metrics
      const processingTime = Date.now() - startTime;
      this.recordMetrics(mergedId, processingTime);

      // Emit event
      await this.emitEvent({
        eventType: 'template_merged',
        templateId: mergedId,
        timestamp: new Date(),
        data: { sourceTemplates: templateIds, strategy, mergedTemplate },
        processingTime
      });

      console.log(`[PriorityTemplateEngine] Merged ${templateIds.length} templates into ${mergedId}`);
      return mergedTemplate;

    } catch (error) {
      await this.emitEvent({
        eventType: 'template_merged',
        templateId: 'merge_failed',
        timestamp: new Date(),
        data: { templateIds, strategy, error },
        error: error as Error
      });

      throw error;
    }
  }

  /**
   * Generate variant from base template
   */
  async generateVariant(config: VariantGenerationConfig): Promise<PriorityTemplate> {
    const startTime = Date.now();

    try {
      // Get base template
      const baseTemplate = await this.getTemplate(config.baseTemplateId);
      if (!baseTemplate) {
        throw new TemplateSystemError(
          `Base template not found: ${config.baseTemplateId}`,
          'BASE_TEMPLATE_NOT_FOUND',
          config.baseTemplateId
        );
      }

      // Generate variant
      const variantTemplate = await this.variantGenerator.generateCustomVariant(baseTemplate, config);

      // Register variant
      await this.registerTemplate(variantTemplate);

      // Record metrics
      const processingTime = Date.now() - startTime;
      this.recordMetrics(variantTemplate.meta.version, processingTime);

      // Emit event
      await this.emitEvent({
        eventType: 'template_registered',
        templateId: variantTemplate.meta.version,
        timestamp: new Date(),
        data: { config, baseTemplate: config.baseTemplateId, variantTemplate },
        processingTime
      });

      console.log(`[PriorityTemplateEngine] Generated variant: ${variantTemplate.meta.version} from ${config.baseTemplateId}`);
      return variantTemplate;

    } catch (error) {
      await this.emitEvent({
        eventType: 'template_registered',
        templateId: 'variant_generation_failed',
        timestamp: new Date(),
        data: { config, error },
        error: error as Error
      });

      throw error;
    }
  }

  /**
   * Validate template against constraints
   */
  async validateTemplate(template: PriorityTemplate): Promise<TemplateValidationResult> {
    const startTime = Date.now();

    try {
      const validationResult = await this.validator.validate(template);

      // Record metrics
      const processingTime = Date.now() - startTime;
      this.recordMetrics(template.meta.version, processingTime);

      // Emit event
      await this.emitEvent({
        eventType: 'template_validated',
        templateId: template.meta.version,
        timestamp: new Date(),
        data: { template, validationResult },
        processingTime
      });

      return validationResult;

    } catch (error) {
      await this.emitEvent({
        eventType: 'template_validated',
        templateId: template.meta.version,
        timestamp: new Date(),
        data: { template, error },
        error: error as Error
      });

      throw error;
    }
  }

  /**
   * Get template by ID
   */
  async getTemplate(templateId: string): Promise<PriorityTemplate | null> {
    return this.templates.get(templateId) || null;
  }

  /**
   * List templates with optional filtering
   */
  async listTemplates(filter?: TemplateFilter): Promise<PriorityTemplate[]> {
    let templates = Array.from(this.templates.values());

    if (filter) {
      templates = templates.filter(template => {
        if (filter.priority && template.priority !== filter.priority) return false;
        if (filter.variantType && template.meta.variantType !== filter.variantType) return false;
        if (filter.frequencyBand && template.meta.frequencyBand !== filter.frequencyBand) return false;
        if (filter.author && !template.meta.author.includes(filter.author)) return false;
        if (filter.tags && !filter.tags.some(tag => template.meta.tags?.includes(tag))) return false;
        if (filter.dateRange) {
          const templateDate = new Date(template.meta.description); // This should use a proper timestamp field
          if (templateDate < filter.dateRange.start || templateDate > filter.dateRange.end) return false;
        }
        return true;
      });
    }

    // Sort by priority (lowest number = highest priority)
    templates.sort((a, b) => a.priority - b.priority);

    return templates;
  }

  /**
   * Delete template
   */
  async deleteTemplate(templateId: string): Promise<boolean> {
    const deleted = this.templates.delete(templateId);

    if (deleted) {
      // Clear cache entries
      this.clearCacheForTemplate(templateId);

      // Remove metrics
      this.metrics.delete(templateId);

      console.log(`[PriorityTemplateEngine] Deleted template: ${templateId}`);
    }

    return deleted;
  }

  /**
   * Get processing metrics for a template
   */
  getMetrics(templateId: string): TemplateProcessingMetrics | undefined {
    return this.metrics.get(templateId);
  }

  /**
   * Get all processing metrics
   */
  getAllMetrics(): TemplateProcessingMetrics[] {
    return Array.from(this.metrics.values());
  }

  /**
   * Clear all caches
   */
  clearCache(): void {
    this.inheritanceCache.clear();
    console.log('[PriorityTemplateEngine] Cleared all caches');
  }

  /**
   * Get engine statistics
   */
  getStats(): {
    templateCount: number;
    cacheSize: number;
    cacheHitRate: number;
    totalProcessingTime: number;
    averageProcessingTime: number;
  } {
    const metrics = Array.from(this.metrics.values());
    const totalTime = metrics.reduce((sum, m) => sum + m.processingTime, 0);
    const avgTime = metrics.length > 0 ? totalTime / metrics.length : 0;

    return {
      templateCount: this.templates.size,
      cacheSize: this.inheritanceCache.size,
      cacheHitRate: this.calculateCacheHitRate(),
      totalProcessingTime: totalTime,
      averageProcessingTime: avgTime
    };
  }

  // ============================================================================
  // PRIVATE METHODS
  // ============================================================================

  /**
   * Initialize component dependencies
   */
  private initializeComponents(): void {
    this.variantGenerator = new TemplateVariantGenerator(this.config);
    this.templateMerger = new TemplateMerger(this.config);
    this.conflictResolver = new TemplateConflictResolver(this.config);
    this.validator = new TemplateValidator(this.config);
  }

  /**
   * Build inheritance chain for a template
   */
  private async buildInheritanceChain(template: PriorityTemplate): Promise<TemplateInheritanceChain> {
    const chain: TemplateInheritanceChain = {
      templateId: template.meta.version,
      chain: [],
      resolvedTemplate: template,
      conflicts: [],
      warnings: [],
      processingTime: 0
    };

    const visited = new Set<string>();
    const processingStack: string[] = [];

    await this.buildChainRecursive(template, chain, visited, processingStack);

    // Sort chain by priority (lowest number = highest priority)
    chain.chain.sort((a, b) => a.priority - b.priority);

    return chain;
  }

  /**
   * Recursively build inheritance chain
   */
  private async buildChainRecursive(
    template: PriorityTemplate,
    chain: TemplateInheritanceChain,
    visited: Set<string>,
    processingStack: string[]
  ): Promise<void> {
    const templateId = template.meta.version;

    // Check for circular dependencies
    if (processingStack.includes(templateId)) {
      throw new TemplateInheritanceError(
        `Circular dependency detected: ${processingStack.join(' -> ')} -> ${templateId}`,
        templateId,
        processingStack,
        processingStack
      );
    }

    if (visited.has(templateId)) {
      return;
    }

    visited.add(templateId);
    processingStack.push(templateId);

    try {
      // Process parent templates
      const parentIds = Array.isArray(template.meta.inherits_from)
        ? template.meta.inherits_from
        : template.meta.inherits_from
          ? [template.meta.inherits_from]
          : [];

      for (const parentId of parentIds) {
        const parentTemplate = await this.getTemplate(parentId);
        if (parentTemplate) {
          await this.buildChainRecursive(parentTemplate, chain, visited, processingStack);
        } else {
          chain.warnings.push({
            warningId: `missing_parent_${parentId}`,
            level: 'warning',
            message: `Parent template not found: ${parentId}`,
            templateId
          });
        }
      }

      // Add current template to chain
      chain.chain.push({
        templateId,
        priority: template.priority,
        appliedAt: new Date(),
        appliedParameters: Object.keys(template.configuration),
        overriddenParameters: [], // Will be calculated during resolution
        conflicts: []
      });

    } finally {
      processingStack.pop();
    }
  }

  /**
   * Clear cache entries for a specific template
   */
  private clearCacheForTemplate(templateId: string): void {
    // Remove from inheritance cache (both as key and in chain dependencies)
    for (const [key, chain] of this.inheritanceCache.entries()) {
      if (key === templateId || chain.chain.some(link => link.templateId === templateId)) {
        this.inheritanceCache.delete(key);
      }
    }
  }

  /**
   * Record processing metrics
   */
  private recordMetrics(templateId: string, processingTime: number): void {
    if (!this.config.performanceMonitoring) return;

    const existingMetrics = this.metrics.get(templateId);

    const metrics: TemplateProcessingMetrics = {
      templateId,
      processingTime,
      memoryUsage: process.memoryUsage().heapUsed,
      parameterCount: 0, // Would be calculated from template
      conflictCount: 0,  // Would be calculated from processing
      warningCount: 0,   // Would be calculated from processing
      cacheHits: existingMetrics?.cacheHits || 0,
      cacheMisses: existingMetrics?.cacheMisses || 0
    };

    this.metrics.set(templateId, metrics);
  }

  /**
   * Update cache metrics
   */
  private updateCacheMetrics(templateId: string, isHit: boolean): void {
    if (!this.config.performanceMonitoring) return;

    const metrics = this.metrics.get(templateId);
    if (metrics) {
      if (isHit) {
        metrics.cacheHits++;
      } else {
        metrics.cacheMisses++;
      }
      this.metrics.set(templateId, metrics);
    }
  }

  /**
   * Calculate cache hit rate
   */
  private calculateCacheHitRate(): number {
    const totalHits = Array.from(this.metrics.values()).reduce((sum, m) => sum + m.cacheHits, 0);
    const totalMisses = Array.from(this.metrics.values()).reduce((sum, m) => sum + m.cacheMisses, 0);
    const total = totalHits + totalMisses;

    return total > 0 ? totalHits / total : 0;
  }

  /**
   * Emit processing event
   */
  private async emitEvent(event: TemplateProcessingEvent): Promise<void> {
    try {
      await this.eventBus.publish(event);
    } catch (error) {
      console.error('[PriorityTemplateEngine] Error emitting event:', error);
    }
  }
}