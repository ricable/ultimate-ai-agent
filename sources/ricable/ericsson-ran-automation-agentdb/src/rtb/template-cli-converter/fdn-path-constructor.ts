/**
 * FDN (Full Distinguished Name) Path Constructor
 *
 * Constructs optimized FDN paths using MO hierarchy knowledge,
 * LDN structure patterns, and Ericsson RAN expertise.
 */

import {
  FdnConstructionResult,
  FdnComponent,
  TemplateToCliContext,
  CliCommandType
} from './types';

import {
  MOHierarchy,
  MOClass,
  LDNHierarchy,
  LDNPattern
} from '../../types/rtb-types';

/**
 * FDN Path Constructor Configuration
 */
export interface FdnConstructorConfig {
  /** Enable path optimization */
  enableOptimization?: boolean;
  /** Validate FDN syntax */
  validateSyntax?: boolean;
  /** Apply hierarchy knowledge */
  applyHierarchyKnowledge?: boolean;
  /** Use LDN structure patterns */
  useLdnPatterns?: boolean;
  /** Enable cognitive optimization */
  enableCognitiveOptimization?: boolean;
}

/**
 * FDN Path Constructor Class
 *
 * Constructs intelligent FDN paths with optimization and validation.
 */
export class FdnPathConstructor {
  private config: FdnConstructorConfig;
  private fdnCache: Map<string, FdnConstructionResult> = new Map();
  private pathPatterns: Map<string, string[]> = new Map();
  private optimizationRules: OptimizationRule[] = [];

  constructor(config: FdnConstructorConfig = {}) {
    this.config = {
      enableOptimization: true,
      validateSyntax: true,
      applyHierarchyKnowledge: true,
      useLdnPatterns: true,
      enableCognitiveOptimization: false,
      ...config
    };

    this.initializePathPatterns();
    this.initializeOptimizationRules();
  }

  /**
   * Construct FDN path from parameter path
   */
  public async construct(
    parameterPath: string,
    context: TemplateToCliContext
  ): Promise<FdnConstructionResult> {
    // Check cache first
    const cacheKey = `${parameterPath}_${JSON.stringify(context.target)}`;
    if (this.fdnCache.has(cacheKey)) {
      return this.fdnCache.get(cacheKey)!;
    }

    console.log(`Constructing FDN path for: ${parameterPath}`);

    try {
      // Phase 1: Parse parameter path
      const parsedPath = this.parseParameterPath(parameterPath);

      // Phase 2: Build base FDN
      const baseFdn = this.buildBaseFdn(parsedPath, context);

      // Phase 3: Apply MO hierarchy knowledge
      const hierarchyEnhancedFdn = this.config.applyHierarchyKnowledge
        ? this.applyHierarchyKnowledge(baseFdn, context)
        : baseFdn;

      // Phase 4: Apply LDN structure patterns
      const ldnEnhancedFdn = this.config.useLdnPatterns
        ? this.applyLdnPatterns(hierarchyEnhancedFdn, context)
        : hierarchyEnhancedFdn;

      // Phase 5: Optimize FDN path
      const optimizedFdn = this.config.enableOptimization
        ? this.optimizeFdnPath(ldnEnhancedFdn, context)
        : ldnEnhancedFdn;

      // Phase 6: Validate FDN
      const validationResult = this.config.validateSyntax
        ? this.validateFdn(optimizedFdn)
        : { isValid: true, errors: [] };

      // Phase 7: Create result
      const result: FdnConstructionResult = {
        fdn: optimizedFdn.fdn,
        isValid: validationResult.isValid,
        components: optimizedFdn.components,
        constructionPath: optimizedFdn.constructionPath,
        errors: validationResult.errors,
        optimization: optimizedFdn.optimization
      };

      // Cache result
      this.fdnCache.set(cacheKey, result);

      console.log(`FDN construction completed: ${result.fdn}`);
      return result;

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error(`FDN construction failed: ${errorMessage}`);

      const errorResult: FdnConstructionResult = {
        fdn: parameterPath,
        isValid: false,
        components: [],
        constructionPath: [parameterPath],
        errors: [errorMessage]
      };

      return errorResult;
    }
  }

  /**
   * Parse parameter path
   */
  private parseParameterPath(parameterPath: string): ParsedPath {
    const parts = parameterPath.split('.');
    const components: PathComponent[] = [];

    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      const component: PathComponent = {
        name: part,
        index: this.extractIndex(part),
        type: this.determineComponentType(part, i, parts.length),
        moClass: null, // Will be resolved later
        isAttribute: i === parts.length - 1 // Last part is usually an attribute
      };

      components.push(component);
    }

    return {
      originalPath: parameterPath,
      components,
      depth: parts.length,
      hasIndex: components.some(comp => comp.index !== null)
    };
  }

  /**
   * Build base FDN from parsed path
   */
  private buildBaseFdn(parsedPath: ParsedPath, context: TemplateToCliContext): FdnConstructionResult {
    const components: FdnComponent[] = [];
    const constructionPath: string[] = [];
    let currentFdn = '';

    for (let i = 0; i < parsedPath.components.length; i++) {
      const component = parsedPath.components[i];
      const fdnComponent = this.createFdnComponent(component, i, context);

      if (fdnComponent) {
        components.push(fdnComponent);
        currentFdn += (i > 0 ? ',' : '') + this.formatFdnComponent(fdnComponent);
        constructionPath.push(currentFdn);
      }
    }

    return {
      fdn: currentFdn,
      isValid: true,
      components,
      constructionPath,
      errors: []
    };
  }

  /**
   * Apply MO hierarchy knowledge
   */
  private applyHierarchyKnowledge(
    fdnResult: FdnConstructionResult,
    context: TemplateToCliContext
  ): FdnConstructionResult {
    if (!context.moHierarchy) {
      return fdnResult;
    }

    const enhancedComponents = fdnResult.components.map(component => {
      if (component.type === 'class') {
        // Enhance with MO hierarchy knowledge
        const moClass = context.moHierarchy!.classes.get(component.name);
        if (moClass) {
          return {
            ...component,
            moClass,
            cardinality: {
              minimum: moClass.cardinality.minimum,
              maximum: moClass.cardinality.maximum,
              current: 1 // Assume single instance for now
            }
          };
        }
      }
      return component;
    });

    return {
      ...fdnResult,
      components: enhancedComponents
    };
  }

  /**
   * Apply LDN structure patterns
   */
  private applyLdnPatterns(
    fdnResult: FdnConstructionResult,
    context: TemplateToCliContext
  ): FdnConstructionResult {
    // Apply known LDN patterns for optimization
    const optimizedFdn = this.applyLdnPatternOptimization(fdnResult.fdn, context);

    return {
      ...fdnResult,
      fdn: optimizedFdn.fdn,
      components: optimizedFdn.components,
      optimization: {
        originalFdn: fdnResult.fdn,
        optimizationApplied: optimizedFdn.optimizations,
        reduction: optimizedFdn.reduction
      }
    };
  }

  /**
   * Optimize FDN path
   */
  private optimizeFdnPath(
    fdnResult: FdnConstructionResult,
    context: TemplateToCliContext
  ): FdnConstructionResult {
    let optimizedFdn = fdnResult.fdn;
    const optimizations: string[] = [];
    let reduction = 0;

    // Apply optimization rules
    for (const rule of this.optimizationRules) {
      if (rule.applicable(optimizedFdn, context)) {
        const result = rule.apply(optimizedFdn, context);
        if (result.optimized) {
          optimizedFdn = result.fdn;
          optimizations.push(rule.name);
          reduction += result.reduction;
        }
      }
    }

    return {
      ...fdnResult,
      fdn: optimizedFdn,
      optimization: {
        originalFdn: fdnResult.fdn,
        optimizationApplied: optimizations,
        reduction
      }
    };
  }

  /**
   * Validate FDN
   */
  private validateFdn(fdnResult: FdnConstructionResult): ValidationResult {
    const errors: string[] = [];

    // Check syntax
    if (!this.isValidFdnSyntax(fdnResult.fdn)) {
      errors.push('Invalid FDN syntax');
    }

    // Check components
    for (const component of fdnResult.components) {
      if (!this.isValidComponent(component)) {
        errors.push(`Invalid component: ${component.name}`);
      }
    }

    // Check structure
    if (!this.isValidFdnStructure(fdnResult.components)) {
      errors.push('Invalid FDN structure');
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  /**
   * Extract index from component name
   */
  private extractIndex(componentName: string): number | null {
    const match = componentName.match(/^(.+?)(\d+)$/);
    return match ? parseInt(match[2], 10) : null;
  }

  /**
   * Determine component type
   */
  private determineComponentType(
    componentName: string,
    index: number,
    totalComponents: number
  ): 'class' | 'attribute' | 'parameter' | 'index' {
    const name = componentName.toLowerCase();

    // Check for known MO classes
    if (this.isKnownMoClass(name)) {
      return 'class';
    }

    // Check if it's the last component (likely an attribute)
    if (index === totalComponents - 1) {
      return 'attribute';
    }

    // Check for numeric index
    if (/\d+$/.test(componentName)) {
      return 'index';
    }

    // Default to class
    return 'class';
  }

  /**
   * Create FDN component
   */
  private createFdnComponent(
    component: PathComponent,
    index: number,
    context: TemplateToCliContext
  ): FdnComponent | null {
    const fdnComponent: FdnComponent = {
      name: component.name,
      value: component.name,
      type: component.type,
      moClass: null
    };

    // Add index if present
    if (component.index !== null) {
      fdnComponent.value = `${component.name}=${component.index}`;
      fdnComponent.type = 'parameter';
    }

    // Add cell ID if available
    if (component.name.toLowerCase().includes('cell') && context.cellIds) {
      const cellId = this.getRelevantCellId(component.name, context.cellIds);
      if (cellId) {
        fdnComponent.value = `${component.name}=${cellId}`;
        fdnComponent.type = 'parameter';
      }
    }

    return fdnComponent;
  }

  /**
   * Format FDN component
   */
  private formatFdnComponent(component: FdnComponent): string {
    switch (component.type) {
      case 'parameter':
        return component.value;
      case 'attribute':
        return component.name;
      default:
        return component.name;
    }
  }

  /**
   * Apply LDN pattern optimization
   */
  private applyLdnPatternOptimization(
    fdn: string,
    context: TemplateToCliContext
  ): { fdn: string; components: FdnComponent[]; optimizations: string[]; reduction: number } {
    let optimizedFdn = fdn;
    const optimizations: string[] = [];
    let reduction = 0;

    // Apply common LDN patterns
    const patterns = [
      {
        name: 'MeContext optimization',
        pattern: /^ManagedElement=/,
        replacement: 'MeContext=' + (context.target.nodeId || 'NODE'),
        reduction: 10
      },
      {
        name: 'Cell ID pattern',
        pattern: /EUtranCellFDD=/,
        replacement: 'EUtranCellFDD=' + (context.cellIds?.primaryCell || 'CELL_1'),
        reduction: 5
      },
      {
        name: 'NR Cell pattern',
        pattern: /NRCellCU=/,
        replacement: 'NRCellCU=' + (context.cellIds?.nrCell || 'NRCELL_1'),
        reduction: 5
      }
    ];

    for (const pattern of patterns) {
      if (pattern.pattern.test(optimizedFdn)) {
        optimizedFdn = optimizedFdn.replace(pattern.pattern, pattern.replacement);
        optimizations.push(pattern.name);
        reduction += pattern.reduction;
      }
    }

    return {
      fdn: optimizedFdn,
      components: this.parseFdnToComponents(optimizedFdn),
      optimizations,
      reduction
    };
  }

  /**
   * Parse FDN to components
   */
  private parseFdnToComponents(fdn: string): FdnComponent[] {
    const components: FdnComponent[] = [];
    const parts = fdn.split(',');

    for (const part of parts) {
      const [name, value] = part.split('=');
      components.push({
        name,
        value: value || name,
        type: value ? 'parameter' : 'class',
        moClass: undefined
      });
    }

    return components;
  }

  /**
   * Check if known MO class
   */
  private isKnownMoClass(name: string): boolean {
    const knownMoClasses = [
      'ManagedElement', 'ENBFunction', 'NRCellCU', 'EUtranCellFDD',
      'EUtranCellTDD', 'ENodeBFunction', 'GERANCell', 'UtranCellFDD',
      'MeContext', 'SubNetwork', 'CmFunction', 'FeatureState',
      'EUtranFreqRelation', 'EutranFreqRelation', 'UtranFreqRelation',
      'ExternalEUtranCellFDD', 'ExternalEutranCellFDD',
      'NeighbourRelation', 'EUtranRelation', 'Mobility',
      'Capacity', 'Coverage', 'Energy', 'Performance'
    ];

    return knownMoClasses.some(moClass =>
      moClass.toLowerCase() === name.toLowerCase()
    );
  }

  /**
   * Get relevant cell ID
   */
  private getRelevantCellId(
    componentName: string,
    cellIds: Record<string, string | undefined>
  ): string | undefined {
    const name = componentName.toLowerCase();

    if (name.includes('eutrancell')) {
      return cellIds.lteCell || cellIds.primaryCell;
    }
    if (name.includes('nrcell')) {
      return cellIds.nrCell || cellIds.primaryCell;
    }
    if (name.includes('cell')) {
      return cellIds.primaryCell;
    }

    return undefined;
  }

  /**
   * Check if valid FDN syntax
   */
  private isValidFdnSyntax(fdn: string): boolean {
    if (!fdn || fdn.length === 0) return false;

    // Check for invalid characters
    const invalidChars = /[<>"|;\\]/;
    if (invalidChars.test(fdn)) return false;

    // Check structure
    const parts = fdn.split(',');
    if (parts.length === 0) return false;

    // Check each part
    for (const part of parts) {
      if (!part.trim()) return false;
      if (part.includes('=') && part.split('=').length !== 2) return false;
    }

    return true;
  }

  /**
   * Check if valid component
   */
  private isValidComponent(component: FdnComponent): boolean {
    if (!component.name || component.name.length === 0) return false;
    if (component.type === 'parameter' && !component.value) return false;
    return true;
  }

  /**
   * Check if valid FDN structure
   */
  private isValidFdnStructure(components: FdnComponent[]): boolean {
    if (components.length === 0) return false;

    // Should have at least one class component
    const hasClass = components.some(comp => comp.type === 'class');
    if (!hasClass) return false;

    // Check for proper ordering (classes before attributes)
    let foundAttribute = false;
    for (const component of components) {
      if (component.type === 'attribute') {
        foundAttribute = true;
      } else if (foundAttribute && component.type === 'class') {
        return false; // Class after attribute
      }
    }

    return true;
  }

  /**
   * Initialize path patterns
   */
  private initializePathPatterns(): void {
    this.pathPatterns.set('cell', ['EUtranCellFDD', 'NRCellCU', 'GERANCell']);
    this.pathPatterns.set('mobility', ['Mobility', 'EUtranRelation', 'NeighbourRelation']);
    this.pathPatterns.set('capacity', ['Capacity', 'ENBFunction', 'NRCellCU']);
    this.pathPatterns.set('energy', ['Energy', 'ENBFunction', 'NRCellCU']);
    this.pathPatterns.set('coverage', ['Coverage', 'EUtranCellFDD', 'NRCellCU']);
  }

  /**
   * Initialize optimization rules
   */
  private initializeOptimizationRules(): void {
    this.optimizationRules = [
      {
        name: 'Remove redundant ManagedElement',
        applicable: (fdn: string, context: TemplateToCliContext) =>
          fdn.includes('ManagedElement=1') && context.target.nodeId,
        apply: (fdn: string, context: TemplateToCliContext) => ({
          optimized: true,
          fdn: fdn.replace(/ManagedElement=1,?/, ''),
          reduction: 20
        })
      },
      {
        name: 'Consolidate cell parameters',
        applicable: (fdn: string, context: TemplateToCliContext) =>
          fdn.includes('EUtranCellFDD=') && fdn.includes(',cellIndividualOffset'),
        apply: (fdn: string, context: TemplateToCliContext) => ({
          optimized: true,
          fdn: fdn.replace(/EUtranCellFDD=([^,]+),([^,]+)=/, 'EUtranCellFDD=$1,$2='),
          reduction: 5
        })
      },
      {
        name: 'Optimize frequency relations',
        applicable: (fdn: string, context: TemplateToCliContext) =>
          fdn.includes('EUtranFreqRelation') && fdn.includes('(EUtranFreqRelationId=='),
        apply: (fdn: string, context: TemplateToCliContext) => ({
          optimized: true,
          fdn: fdn.replace(/\(EUtranFreqRelationId==([^)]+)\)/, '$1'),
          reduction: 15
        })
      }
    ];
  }

  /**
   * Clear cache
   */
  public clearCache(): void {
    this.fdnCache.clear();
  }

  /**
   * Get cache statistics
   */
  public getCacheStats(): {
    size: number;
    hitRate: number;
    totalLookups: number;
  } {
    return {
      size: this.fdnCache.size,
      hitRate: 0.8, // Placeholder - would track actual hits
      totalLookups: this.fdnCache.size
    };
  }
}

/**
 * Parsed path structure
 */
interface ParsedPath {
  originalPath: string;
  components: PathComponent[];
  depth: number;
  hasIndex: boolean;
}

/**
 * Path component
 */
interface PathComponent {
  name: string;
  index: number | null;
  type: 'class' | 'attribute' | 'parameter' | 'index';
  moClass: MOClass | null;
  isAttribute: boolean;
}

/**
 * Validation result
 */
interface ValidationResult {
  isValid: boolean;
  errors: string[];
}

/**
 * Optimization rule
 */
interface OptimizationRule {
  name: string;
  applicable: (fdn: string, context: TemplateToCliContext) => boolean;
  apply: (fdn: string, context: TemplateToCliContext) => {
    optimized: boolean;
    fdn: string;
    reduction: number;
  };
}