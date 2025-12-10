/**
 * Template Registry - Centralized Template Storage and Management
 *
 * Provides efficient storage, retrieval, indexing, and metadata management
 * for RTB templates with priority-based organization.
 */

import {
  RTBTemplate,
  TemplateMeta,
  RTBParameter
} from '../../types/rtb-types';
import {
  TemplatePriorityInfo,
  TemplatePriority,
  TemplateInheritanceChain,
  TemplateValidationResult
} from './priority-engine';

/**
 * Template search filters
 */
export interface TemplateSearchFilter {
  category?: string | string[];
  priorityRange?: { min: number; max: number };
  environment?: string;
  tags?: string[];
  author?: string;
  source?: string;
  hasInheritance?: boolean;
  isActive?: boolean;
  createdAfter?: Date;
  createdBefore?: Date;
}

/**
 * Template search result
 */
export interface TemplateSearchResult {
  templates: Array<{
    name: string;
    template: RTBTemplate;
    priority: TemplatePriorityInfo;
    relevanceScore?: number;
  }>;
  totalCount: number;
  facets: {
    categories: Record<string, number>;
    priorities: Record<string, number>;
    environments: Record<string, number>;
    tags: Record<string, number>;
  };
  searchTime: number;
}

/**
 * Template metadata for registry
 */
export interface RegistryTemplateMeta {
  name: string;
  priority: TemplatePriorityInfo;
  hash: string;
  size: number;
  createdAt: Date;
  updatedAt: Date;
  accessCount: number;
  lastAccessed: Date;
  validationStatus: 'valid' | 'invalid' | 'pending';
  dependencies: string[];
  dependents: string[];
  tags: Set<string>;
  indexes: Record<string, any>;
}

/**
 * Template registry configuration
 */
export interface TemplateRegistryConfig {
  enableValidation?: boolean;
  enableCaching?: boolean;
  cacheSize?: number;
  enableIndexing?: boolean;
  indexParameters?: boolean;
  indexCustomFunctions?: boolean;
  enableVersioning?: boolean;
  maxVersions?: number;
  compressionThreshold?: number;
}

/**
 * Template statistics
 */
export interface TemplateRegistryStats {
  totalTemplates: number;
  totalSize: number;
  categories: Record<string, number>;
  priorities: Record<string, number>;
  averageInheritanceDepth: number;
  validationStatus: Record<string, number>;
  cacheHitRate: number;
  indexSize: number;
}

/**
 * Template Registry
 *
 * Centralized storage and management system for RTB templates
 * with efficient indexing, search, and metadata tracking.
 */
export class TemplateRegistry {
  private templates = new Map<string, RTBTemplate>();
  private metadata = new Map<string, RegistryTemplateMeta>();
  private indexes = {
    category: new Map<string, Set<string>>(),
    priority: new Map<number, Set<string>>(),
    environment: new Map<string, Set<string>>(),
    tag: new Map<string, Set<string>>(),
    author: new Map<string, Set<string>>(),
    source: new Map<string, Set<string>>(),
    parameter: new Map<string, Set<string>>(),
    customFunction: new Map<string, Set<string>>(),
    inheritance: new Map<string, Set<string>>()
  };

  private cache = new Map<string, any>();
  private validationResults = new Map<string, TemplateValidationResult>();
  private config: Required<TemplateRegistryConfig>;
  private accessStats = new Map<string, { count: number; lastAccess: Date }>();

  constructor(config: TemplateRegistryConfig = {}) {
    this.config = {
      enableValidation: config.enableValidation ?? true,
      enableCaching: config.enableCaching ?? true,
      cacheSize: config.cacheSize ?? 1000,
      enableIndexing: config.enableIndexing ?? true,
      indexParameters: config.indexParameters ?? true,
      indexCustomFunctions: config.indexCustomFunctions ?? true,
      enableVersioning: config.enableVersioning ?? false,
      maxVersions: config.maxVersions ?? 10,
      compressionThreshold: config.compressionThreshold ?? 10240
    };
  }

  /**
   * Register a template in the registry
   */
  async registerTemplate(
    name: string,
    template: RTBTemplate,
    priority: TemplatePriorityInfo
  ): Promise<void> {
    // Validate template if enabled
    if (this.config.enableValidation) {
      const validationResult = await this.validateTemplate(template);
      this.validationResults.set(name, validationResult);

      if (!validationResult.isValid) {
        throw new Error(
          `Template validation failed: ${validationResult.errors.map(e => e.message).join(', ')}`
        );
      }
    }

    // Generate template hash and metadata
    const hash = this.generateTemplateHash(template);
    const size = this.calculateTemplateSize(template);
    const now = new Date();

    // Check if template already exists
    const existingMeta = this.metadata.get(name);

    // Store template
    this.templates.set(name, template);

    // Create metadata
    const meta: RegistryTemplateMeta = {
      name,
      priority,
      hash,
      size,
      createdAt: existingMeta?.createdAt || now,
      updatedAt: now,
      accessCount: existingMeta?.accessCount || 0,
      lastAccessed: existingMeta?.lastAccessed || now,
      validationStatus: this.validationResults.get(name)?.isValid ? 'valid' : 'pending',
      dependencies: this.extractDependencies(template),
      dependents: [],
      tags: new Set(template.meta?.tags || []),
      indexes: {}
    };

    this.metadata.set(name, meta);

    // Update indexes if enabled
    if (this.config.enableIndexing) {
      this.updateIndexes(name, template, priority);
    }

    // Update dependency relationships
    this.updateDependencyRelationships(name, meta.dependencies);

    // Clear cache for this template
    this.clearCacheForTemplate(name);
  }

  /**
   * Get template from registry
   */
  getTemplate(name: string): RTBTemplate | undefined {
    const template = this.templates.get(name);
    if (template) {
      this.updateAccessStats(name);
    }
    return template;
  }

  /**
   * Get template metadata
   */
  getTemplateMetadata(name: string): RegistryTemplateMeta | undefined {
    return this.metadata.get(name);
  }

  /**
   * Search templates with filters
   */
  async searchTemplates(filter: TemplateSearchFilter): Promise<TemplateSearchResult> {
    const startTime = Date.now();

    // Start with all templates, then apply filters
    let candidates = Array.from(this.templates.keys());

    // Apply filters
    if (filter.category) {
      const categories = Array.isArray(filter.category) ? filter.category : [filter.category];
      candidates = candidates.filter(name =>
        categories.some(cat => this.indexes.category.get(cat)?.has(name))
      );
    }

    if (filter.priorityRange) {
      candidates = candidates.filter(name => {
        const meta = this.metadata.get(name);
        return meta &&
          meta.priority.level >= filter.priorityRange!.min &&
          meta.priority.level <= filter.priorityRange!.max;
      });
    }

    if (filter.environment) {
      candidates = candidates.filter(name =>
        this.indexes.environment.get(filter.environment!)?.has(name)
      );
    }

    if (filter.tags && filter.tags.length > 0) {
      candidates = candidates.filter(name =>
        filter.tags!.some(tag => this.indexes.tag.get(tag)?.has(name))
      );
    }

    if (filter.author) {
      candidates = candidates.filter(name =>
        this.indexes.author.get(filter.author!)?.has(name)
      );
    }

    if (filter.source) {
      candidates = candidates.filter(name =>
        this.indexes.source.get(filter.source!)?.has(name)
      );
    }

    if (filter.hasInheritance !== undefined) {
      candidates = candidates.filter(name => {
        const meta = this.metadata.get(name);
        return meta?.dependencies.length > 0 === filter.hasInheritance!;
      });
    }

    if (filter.isActive !== undefined) {
      candidates = candidates.filter(name => {
        const meta = this.metadata.get(name);
        return meta?.validationStatus === 'valid' === filter.isActive!;
      });
    }

    // Build results
    const results = candidates.map(name => {
      const template = this.templates.get(name)!;
      const priority = this.metadata.get(name)!.priority;

      return {
        name,
        template,
        priority,
        relevanceScore: this.calculateRelevanceScore(name, filter)
      };
    });

    // Sort by relevance score
    results.sort((a, b) => (b.relevanceScore || 0) - (a.relevanceScore || 0));

    // Build facets
    const facets = this.buildSearchFacets(candidates);

    const searchTime = Date.now() - startTime;

    return {
      templates: results,
      totalCount: results.length,
      facets,
      searchTime
    };
  }

  /**
   * Get templates by category
   */
  getTemplatesByCategory(category: string): Array<{ name: string; template: RTBTemplate }> {
    const names = this.indexes.category.get(category) || new Set();
    return Array.from(names).map(name => ({
      name,
      template: this.templates.get(name)!
    })).filter(item => item.template);
  }

  /**
   * Get templates by priority range
   */
  getTemplatesByPriority(minPriority: number, maxPriority: number): Array<{ name: string; template: RTBTemplate }> {
    const results: Array<{ name: string; template: RTBTemplate }> = [];

    for (let level = minPriority; level <= maxPriority; level++) {
      const names = this.indexes.priority.get(level) || new Set();
      for (const name of names) {
        const template = this.templates.get(name);
        if (template) {
          results.push({ name, template });
        }
      }
    }

    return results;
  }

  /**
   * Get templates that inherit from a specific template
   */
  getTemplateDependents(parentName: string): Array<{ name: string; template: RTBTemplate }> {
    const meta = this.metadata.get(parentName);
    if (!meta) return [];

    return meta.dependents.map(name => ({
      name,
      template: this.templates.get(name)!
    })).filter(item => item.template);
  }

  /**
   * Get template dependencies
   */
  getTemplateDependencies(name: string): Array<{ name: string; template: RTBTemplate }> {
    const meta = this.metadata.get(name);
    if (!meta) return [];

    return meta.dependencies.map(depName => ({
      name: depName,
      template: this.templates.get(depName)!
    })).filter(item => item.template);
  }

  /**
   * Remove template from registry
   */
  removeTemplate(name: string): boolean {
    const removed = this.templates.delete(name);
    if (removed) {
      // Remove metadata
      const meta = this.metadata.get(name);
      if (meta) {
        this.metadata.delete(name);

        // Remove from indexes
        this.removeFromIndexes(name, meta);

        // Update dependency relationships
        this.removeDependencyRelationships(name, meta.dependencies);
      }

      // Clear cache
      this.clearCacheForTemplate(name);

      // Remove validation results
      this.validationResults.delete(name);

      // Remove access stats
      this.accessStats.delete(name);
    }

    return removed;
  }

  /**
   * Update template in registry
   */
  async updateTemplate(
    name: string,
    template: Partial<RTBTemplate>,
    priority?: Partial<TemplatePriorityInfo>
  ): Promise<void> {
    const existingTemplate = this.templates.get(name);
    const existingMeta = this.metadata.get(name);

    if (!existingTemplate || !existingMeta) {
      throw new Error(`Template '${name}' not found`);
    }

    // Merge template data
    const updatedTemplate = { ...existingTemplate, ...template };

    // Merge priority info
    const updatedPriority = { ...existingMeta.priority, ...priority };

    // Re-register with updates
    await this.registerTemplate(name, updatedTemplate, updatedPriority);
  }

  /**
   * Validate template
   */
  private async validateTemplate(template: RTBTemplate): Promise<TemplateValidationResult> {
    // This is a placeholder - actual validation would be implemented
    // based on the RTB schema and business rules
    const errors: any[] = [];
    const warnings: any[] = [];

    // Basic structure validation
    if (!template.configuration) {
      errors.push({
        code: 'MISSING_CONFIGURATION',
        message: 'Template must have configuration object',
        severity: 'error'
      });
    }

    // Validate custom functions
    if (template.custom) {
      for (const func of template.custom) {
        if (!func.name || !func.body) {
          errors.push({
            code: 'INVALID_CUSTOM_FUNCTION',
            message: `Invalid custom function: ${func.name}`,
            severity: 'error'
          });
        }
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings,
      metadata: {
        validationTime: 0,
        parametersValidated: Object.keys(template.configuration || {}).length,
        constraintsChecked: 0
      }
    };
  }

  /**
   * Generate template hash
   */
  private generateTemplateHash(template: RTBTemplate): string {
    const content = JSON.stringify(template);
    return Buffer.from(content).toString('base64').slice(0, 16);
  }

  /**
   * Calculate template size
   */
  private calculateTemplateSize(template: RTBTemplate): number {
    return JSON.stringify(template).length;
  }

  /**
   * Extract template dependencies
   */
  private extractDependencies(template: RTBTemplate): string[] {
    const dependencies: string[] = [];

    // Extract from metadata
    if (template.meta?.inherits_from) {
      if (Array.isArray(template.meta.inherits_from)) {
        dependencies.push(...template.meta.inherits_from);
      } else {
        dependencies.push(template.meta.inherits_from);
      }
    }

    // Extract from evaluations (could contain template references)
    if (template.evaluations) {
      for (const [key, evalOp] of Object.entries(template.evaluations)) {
        // Simple regex to find template references
        const templateRefs = (evalOp.eval || '').match(/\b[A-Z][a-zA-Z0-9_]*Template\b/g) || [];
        dependencies.push(...templateRefs);
      }
    }

    return [...new Set(dependencies)];
  }

  /**
   * Update search indexes
   */
  private updateIndexes(name: string, template: RTBTemplate, priority: TemplatePriorityInfo): void {
    // Category index
    this.indexes.category.set(priority.category,
      (this.indexes.category.get(priority.category) || new Set()).add(name)
    );

    // Priority index
    this.indexes.priority.set(priority.level,
      (this.indexes.priority.get(priority.level) || new Set()).add(name)
    );

    // Environment index
    if (template.meta?.environment) {
      this.indexes.environment.set(template.meta.environment,
        (this.indexes.environment.get(template.meta.environment) || new Set()).add(name)
      );
    }

    // Tag index
    if (template.meta?.tags) {
      for (const tag of template.meta.tags) {
        this.indexes.tag.set(tag,
          (this.indexes.tag.get(tag) || new Set()).add(name)
        );
      }
    }

    // Author index
    if (template.meta?.author) {
      for (const author of template.meta.author) {
        this.indexes.author.set(author,
          (this.indexes.author.get(author) || new Set()).add(name)
        );
      }
    }

    // Source index
    if (priority.source) {
      this.indexes.source.set(priority.source,
        (this.indexes.source.get(priority.source) || new Set()).add(name)
      );
    }

    // Parameter index (if enabled)
    if (this.config.indexParameters && template.configuration) {
      for (const paramName of Object.keys(template.configuration)) {
        this.indexes.parameter.set(paramName,
          (this.indexes.parameter.get(paramName) || new Set()).add(name)
        );
      }
    }

    // Custom function index (if enabled)
    if (this.config.indexCustomFunctions && template.custom) {
      for (const func of template.custom) {
        this.indexes.customFunction.set(func.name,
          (this.indexes.customFunction.get(func.name) || new Set()).add(name)
        );
      }
    }
  }

  /**
   * Remove from indexes
   */
  private removeFromIndexes(name: string, meta: RegistryTemplateMeta): void {
    // Remove from all indexes
    for (const index of Object.values(this.indexes)) {
      for (const [key, names] of index) {
        names.delete(name);
        if (names.size === 0) {
          index.delete(key);
        }
      }
    }
  }

  /**
   * Update dependency relationships
   */
  private updateDependencyRelationships(name: string, dependencies: string[]): void {
    for (const dep of dependencies) {
      const depMeta = this.metadata.get(dep);
      if (depMeta && !depMeta.dependents.includes(name)) {
        depMeta.dependents.push(name);
      }
    }
  }

  /**
   * Remove dependency relationships
   */
  private removeDependencyRelationships(name: string, dependencies: string[]): void {
    for (const dep of dependencies) {
      const depMeta = this.metadata.get(dep);
      if (depMeta) {
        const index = depMeta.dependents.indexOf(name);
        if (index > -1) {
          depMeta.dependents.splice(index, 1);
        }
      }
    }
  }

  /**
   * Calculate relevance score for search
   */
  private calculateRelevanceScore(name: string, filter: TemplateSearchFilter): number {
    let score = 0;
    const meta = this.metadata.get(name);
    if (!meta) return 0;

    // Category match
    if (filter.category) {
      const categories = Array.isArray(filter.category) ? filter.category : [filter.category];
      if (categories.includes(meta.priority.category)) {
        score += 10;
      }
    }

    // Priority range match
    if (filter.priorityRange) {
      if (meta.priority.level >= filter.priorityRange.min &&
          meta.priority.level <= filter.priorityRange.max) {
        score += 5;
      }
    }

    // Tag matches
    if (filter.tags) {
      const matchingTags = filter.tags.filter(tag => meta.tags.has(tag));
      score += matchingTags.length * 3;
    }

    // Access frequency bonus
    score += Math.min(meta.accessCount / 10, 5);

    return score;
  }

  /**
   * Build search facets
   */
  private buildSearchFacets(candidateNames: string[]) {
    const facets = {
      categories: {} as Record<string, number>,
      priorities: {} as Record<string, number>,
      environments: {} as Record<string, number>,
      tags: {} as Record<string, number>
    };

    for (const name of candidateNames) {
      const meta = this.metadata.get(name);
      if (!meta) continue;

      // Category facet
      facets.categories[meta.priority.category] =
        (facets.categories[meta.priority.category] || 0) + 1;

      // Priority facet
      const priorityKey = `${meta.priority.level} (${meta.priority.category})`;
      facets.priorities[priorityKey] = (facets.priorities[priorityKey] || 0) + 1;

      // Environment facet
      const template = this.templates.get(name);
      if (template?.meta?.environment) {
        facets.environments[template.meta.environment] =
          (facets.environments[template.meta.environment] || 0) + 1;
      }

      // Tag facets
      for (const tag of meta.tags) {
        facets.tags[tag] = (facets.tags[tag] || 0) + 1;
      }
    }

    return facets;
  }

  /**
   * Update access statistics
   */
  private updateAccessStats(name: string): void {
    const stats = this.accessStats.get(name) || { count: 0, lastAccess: new Date() };
    stats.count++;
    stats.lastAccess = new Date();
    this.accessStats.set(name, stats);

    // Update metadata
    const meta = this.metadata.get(name);
    if (meta) {
      meta.accessCount = stats.count;
      meta.lastAccessed = stats.lastAccess;
    }
  }

  /**
   * Clear cache for template
   */
  private clearCacheForTemplate(name: string): void {
    for (const [key] of this.cache) {
      if (key.startsWith(`${name}:`)) {
        this.cache.delete(key);
      }
    }
  }

  /**
   * Get registry statistics
   */
  getRegistryStats(): TemplateRegistryStats {
    const stats: TemplateRegistryStats = {
      totalTemplates: this.templates.size,
      totalSize: Array.from(this.metadata.values()).reduce((sum, meta) => sum + meta.size, 0),
      categories: {},
      priorities: {},
      averageInheritanceDepth: 0,
      validationStatus: {},
      cacheHitRate: 0,
      indexSize: 0
    };

    let totalInheritanceDepth = 0;
    let inheritanceCount = 0;

    for (const meta of this.metadata.values()) {
      // Categories
      stats.categories[meta.priority.category] =
        (stats.categories[meta.priority.category] || 0) + 1;

      // Priorities
      const priorityKey = `${meta.priority.level}`;
      stats.priorities[priorityKey] = (stats.priorities[priorityKey] || 0) + 1;

      // Validation status
      stats.validationStatus[meta.validationStatus] =
        (stats.validationStatus[meta.validationStatus] || 0) + 1;

      // Inheritance depth
      if (meta.dependencies.length > 0) {
        totalInheritanceDepth += meta.dependencies.length;
        inheritanceCount++;
      }
    }

    stats.averageInheritanceDepth = inheritanceCount > 0 ? totalInheritanceDepth / inheritanceCount : 0;

    // Calculate index size
    for (const index of Object.values(this.indexes)) {
      for (const names of index.values()) {
        stats.indexSize += names.size;
      }
    }

    return stats;
  }

  /**
   * Clear all data
   */
  clear(): void {
    this.templates.clear();
    this.metadata.clear();
    this.cache.clear();
    this.validationResults.clear();
    this.accessStats.clear();

    for (const index of Object.values(this.indexes)) {
      index.clear();
    }
  }
}