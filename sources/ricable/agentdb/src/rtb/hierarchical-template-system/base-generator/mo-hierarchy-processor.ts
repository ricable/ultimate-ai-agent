import { MOClass, MOHierarchy, MOCardinality, MORelationship, LDNPattern, LDNHierarchy, ReservedByRelationship, ReservedByHierarchy, ProcessingStats } from '../../types/rtb-types';

export interface MOHierarchyProcessingResult {
  moHierarchy: MOHierarchy;
  ldnHierarchy: LDNHierarchy;
  reservedByHierarchy: ReservedByHierarchy;
  stats: ProcessingStats;
  errors: string[];
  warnings: string[];
}

export interface MOMetaData {
  moClass: string;
  className: string;
  displayName: string;
  description?: string;
  version?: string;
  flags: string[];
  attributes: string[];
  children: string[];
  parent: string;
  level: number;
  cardinality: MOCardinality;
  isAbstract: boolean;
  isLeaf: boolean;
  isRoot: boolean;
}

export interface LDNPathSegment {
  moClass: string;
  attributeName: string;
  cardinality: string;
  isOptional: boolean;
  isArray: boolean;
}

export interface HierarchyValidationRule {
  name: string;
  description: string;
  validator: (hierarchy: MOHierarchy) => ValidationResult;
  severity: 'error' | 'warning' | 'info';
}

export interface ValidationResult {
  isValid: boolean;
  message: string;
  details?: any;
}

export interface MOHierarchyConfig {
  strictValidation: boolean;
  validateCardinality: boolean;
  validateLDNPaths: boolean;
  validateReservedBy: boolean;
  includeDerivedClasses: boolean;
  maxHierarchyDepth: number;
}

export class MOClassHierarchyProcessor {
  private config: MOHierarchyConfig;
  private hierarchyRules: HierarchyValidationRule[];
  private startTime: number;

  constructor(config: MOHierarchyConfig = {
    strictValidation: true,
    validateCardinality: true,
    validateLDNPaths: true,
    validateReservedBy: true,
    includeDerivedClasses: true,
    maxHierarchyDepth: 10
  }) {
    this.config = config;
    this.startTime = Date.now();
    this.initializeHierarchyRules();
  }

  /**
   * Process MO class hierarchy from momt_tree.txt
   */
  async processMOTreeFile(filePath: string): Promise<MOHierarchyProcessingResult> {
    console.log(`Processing MO hierarchy from ${filePath}`);

    try {
      const fs = require('fs').promises;
      const fileContent = await fs.readFile(filePath, 'utf8');

      // Parse MO tree structure
      const moClasses = this.parseMOTreeContent(fileContent);
      console.log(`Parsed ${moClasses.size} MO classes from tree structure`);

      // Build MO hierarchy
      const moHierarchy = this.buildMOHierarchy(moClasses);

      // Validate hierarchy
      const validationErrors: string[] = [];
      const validationWarnings: string[] = [];
      await this.validateMOHierarchy(moHierarchy, validationErrors, validationWarnings);

      const endTime = Date.now();
      const processingTime = (endTime - this.startTime) / 1000;

      const stats: ProcessingStats = {
        xmlProcessingTime: 0,
        hierarchyProcessingTime: processingTime,
        validationTime: 0,
        totalParameters: 0,
        totalMOClasses: moClasses.size,
        totalRelationships: this.countRelationships(moHierarchy),
        memoryUsage: process.memoryUsage().heapUsed / 1024 / 1024,
        errorCount: validationErrors.length,
        warningCount: validationWarnings.length
      };

      console.log(`MO hierarchy processing complete: ${moClasses.size} classes, ${stats.totalRelationships} relationships`);

      return {
        moHierarchy,
        ldnHierarchy: {
          rootPatterns: [],
          patternsByLevel: new Map(),
          lldnStructure: new Map(),
          navigationPaths: new Map()
        },
        reservedByHierarchy: {
          totalRelationships: 0,
          relationships: new Map(),
          classDependencies: new Map(),
          constraintValidation: new Map(),
          circularDependencies: []
        },
        stats,
        errors: validationErrors,
        warnings: validationWarnings
      };

    } catch (error) {
      throw new Error(`Failed to process MO hierarchy: ${error}`);
    }
  }

  /**
   * Process LDN patterns from momtl_LDN.txt
   */
  async processLDNFile(filePath: string): Promise<LDNHierarchy> {
    console.log(`Processing LDN patterns from ${filePath}`);

    try {
      const fs = require('fs').promises;
      const fileContent = await fs.readFile(filePath, 'utf8');

      const ldnPatterns = this.parseLDNContent(fileContent);
      console.log(`Parsed ${ldnPatterns.size} LDN patterns`);

      return this.buildLDNHierarchy(ldnPatterns);

    } catch (error) {
      console.warn(`Failed to process LDN file: ${error}`);
      return {
        rootPatterns: [],
        patternsByLevel: new Map(),
        lldnStructure: new Map(),
        navigationPaths: new Map()
      };
    }
  }

  /**
   * Process reservedBy relationships from reservedby.txt
   */
  async processReservedByFile(filePath: string): Promise<ReservedByHierarchy> {
    console.log(`Processing reservedBy relationships from ${filePath}`);

    try {
      const fs = require('fs').promises;
      const fileContent = await fs.readFile(filePath, 'utf8');

      const reservedByRelationships = this.parseReservedByContent(fileContent);
      console.log(`Parsed ${reservedByRelationships.size} reservedBy relationships`);

      return this.buildReservedByHierarchy(reservedByRelationships);

    } catch (error) {
      console.warn(`Failed to process reservedBy file: ${error}`);
      return {
        totalRelationships: 0,
        relationships: new Map(),
        classDependencies: new Map(),
        constraintValidation: new Map(),
        circularDependencies: []
      };
    }
  }

  /**
   * Parse MO tree content from momt_tree.txt
   */
  private parseMOTreeContent(content: string): Map<string, MOMetaData> {
    const moClasses = new Map<string, MOMetaData>();
    const lines = content.split('\n');
    let currentLevel = 0;
    let parentStack: string[] = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line || line.startsWith('#')) continue; // Skip empty lines and comments

      // Parse MO class line
      const match = line.match(/^(\s*)([A-Za-z][A-Za-z0-9_]*)(?:\s*\(([^)]*)\))?$/);
      if (match) {
        const [, indent, moClass, attributes] = match;
        const level = Math.floor(indent.length / 2); // Assuming 2 spaces per level
        const className = this.extractClassName(moClass);
        const displayName = this.extractDisplayName(moClass);
        const flags = this.parseAttributes(attributes);

        // Adjust parent stack based on level
        while (parentStack.length > level) {
          parentStack.pop();
        }

        const parent = parentStack.length > 0 ? parentStack[parentStack.length - 1] : '';
        const cardinality = this.parseCardinality(attributes);
        const isAbstract = flags.includes('abstract');
        const isLeaf = !flags.includes('hasChildren');
        const isRoot = level === 0;

        const metaData: MOMetaData = {
          moClass,
          className,
          displayName,
          description: this.extractDescription(attributes),
          version: this.extractVersion(attributes),
          flags,
          attributes: [],
          children: [],
          parent,
          level,
          cardinality,
          isAbstract,
          isLeaf,
          isRoot
        };

        moClasses.set(moClass, metaData);

        // Update parent's children
        if (parent && moClasses.has(parent)) {
          moClasses.get(parent)!.children.push(moClass);
        }

        // Add to parent stack
        parentStack.push(moClass);
        currentLevel = level;
      }
    }

    return moClasses;
  }

  /**
   * Parse LDN content from momtl_LDN.txt
   */
  private parseLDNContent(content: string): Map<string, LDNPattern> {
    const ldnPatterns = new Map<string, LDNPattern>();
    const lines = content.split('\n');

    for (const line of lines) {
      const trimmedLine = line.trim();
      if (!trimmedLine || trimmedLine.startsWith('#')) continue;

      // Parse LDN pattern line
      const match = trimmedLine.match(/^([A-Za-z][A-Za-z0-9_]*)(?:\[(\d+)\])?(?:\.([A-Za-z][A-Za-z0-9_]*)(?:\[(\d+)\])?)*$/);
      if (match) {
        const path = trimmedLine.split('.');
        const rootMO = path[0];
        const leafMO = path[path.length - 1];
        const hierarchyLevel = path.length - 1;

        // Extract cardinality information
        const cardinality: Array<{ mo: string; cardinality: string }> = [];
        for (let i = 0; i < path.length; i++) {
          const segment = path[i];
          const matchCard = segment.match(/^([A-Za-z][A-Za-z0-9_]*)(?:\[(\d+)\])?$/);
          if (matchCard) {
            const moName = matchCard[1];
            const card = matchCard[2] || '1';
            cardinality.push({ mo: moName, cardinality: card });
          }
        }

        const ldnPattern: LDNPattern = {
          path,
          cardinality,
          hierarchyLevel,
          rootMO,
          leafMO,
          isValidLDN: this.validateLDNPath(path),
          description: `LDN pattern from ${rootMO} to ${leafMO}`
        };

        ldnPatterns.set(trimmedLine, ldnPattern);
      }
    }

    return ldnPatterns;
  }

  /**
   * Parse reservedBy content from reservedby.txt
   */
  private parseReservedByContent(content: string): Map<string, ReservedByRelationship> {
    const relationships = new Map<string, ReservedByRelationship>();
    const lines = content.split('\n');

    for (const line of lines) {
      const trimmedLine = line.trim();
      if (!trimmedLine || trimmedLine.startsWith('#')) continue;

      // Parse reservedBy relationship line
      // Format: sourceClass -> targetClass [relationshipType] [cardinality] [description]
      const match = trimmedLine.match(/^([A-Za-z][A-Za-z0-9_]*)\s*->\s*([A-Za-z][A-Za-z0-9_]*)\s*(?:\[(\w+)\])?\s*(?:\[(\d+)\-(\d+)\])?\s*(?:\{([^}]*)\})?$/);
      if (match) {
        const [, sourceClass, targetClass, relationshipType = 'reserves', minCard = '0', maxCard = '1', description = ''] = match;

        const cardinality: MOCardinality = {
          minimum: parseInt(minCard),
          maximum: parseInt(maxCard),
          type: maxCard === '1' ? 'single' : maxCard === '*' ? 'unbounded' : 'bounded'
        };

        const relationship: ReservedByRelationship = {
          sourceClass,
          targetClass,
          relationshipType: relationshipType as any,
          cardinality,
          constraints: {},
          description: description || `${sourceClass} ${relationshipType} ${targetClass}`
        };

        const key = `${sourceClass}->${targetClass}`;
        relationships.set(key, relationship);
      }
    }

    return relationships;
  }

  /**
   * Build MO hierarchy from parsed MO classes
   */
  private buildMOHierarchy(moClasses: Map<string, MOMetaData>): MOHierarchy {
    const classes = new Map<string, MOClass>();
    const relationships = new Map<string, MORelationship>();
    const cardinality = new Map<string, MOCardinality>();
    const inheritanceChain = new Map<string, string[]>();

    // Convert MOMetaData to MOClass
    for (const [moClass, metaData] of moClasses) {
      const moClassObj: MOClass = {
        id: moClass,
        name: metaData.className,
        parentClass: metaData.parent,
        cardinality: metaData.cardinality,
        flags: this.convertFlags(metaData.flags),
        children: metaData.children,
        attributes: metaData.attributes,
        derivedClasses: [] // Will be populated later
      };

      classes.set(moClass, moClassObj);
      cardinality.set(moClass, metaData.cardinality);
    }

    // Build inheritance chains
    for (const [moClass, metaData] of moClasses) {
      const chain = this.buildInheritanceChain(moClass, moClasses);
      inheritanceChain.set(moClass, chain);

      // Update derived classes for parents
      for (const ancestor of chain.slice(1)) { // Skip the class itself
        if (classes.has(ancestor)) {
          classes.get(ancestor)!.derivedClasses.push(moClass);
        }
      }
    }

    // Build relationships
    for (const [moClass, metaData] of moClasses) {
      for (const child of metaData.children) {
        const relationship: MORelationship = {
          parentId: moClass,
          relationType: 'parent-child',
          count: 1,
          description: `${moClass} contains ${child}`
        };

        relationships.set(`${moClass}->${child}`, relationship);
      }
    }

    // Find root class
    const rootClasses = Array.from(moClasses.values()).filter(m => m.isRoot);
    const rootClass = rootClasses.length > 0 ? rootClasses[0].moClass : 'ManagedElement';

    return {
      rootClass,
      classes,
      relationships,
      cardinality,
      inheritanceChain
    };
  }

  /**
   * Build LDN hierarchy from parsed patterns
   */
  private buildLDNHierarchy(ldnPatterns: Map<string, LDNPattern>): LDNHierarchy {
    const rootPatterns: LDNPattern[] = [];
    const patternsByLevel = new Map<number, LDNPattern[]>();
    const lldnStructure = new Map<string, LDNPattern>();
    const navigationPaths = new Map<string, LDNPattern[]>();

    // Categorize patterns by level and root
    for (const pattern of ldnPatterns.values()) {
      // Add to level-specific map
      if (!patternsByLevel.has(pattern.hierarchyLevel)) {
        patternsByLevel.set(pattern.hierarchyLevel, []);
      }
      patternsByLevel.get(pattern.hierarchyLevel)!.push(pattern);

      // Add to root patterns if it's a root
      if (pattern.hierarchyLevel === 0) {
        rootPatterns.push(pattern);
      }

      // Add to LLDN structure
      const lldnKey = pattern.path[0];
      if (!lldnStructure.has(lldnKey)) {
        lldnStructure.set(lldnKey, pattern);
      }

      // Build navigation paths
      const pathKey = pattern.path.join('.');
      if (!navigationPaths.has(pathKey)) {
        navigationPaths.set(pathKey, []);
      }
      navigationPaths.get(pathKey)!.push(pattern);
    }

    return {
      rootPatterns,
      patternsByLevel,
      lldnStructure,
      navigationPaths
    };
  }

  /**
   * Build reservedBy hierarchy from parsed relationships
   */
  private buildReservedByHierarchy(relationships: Map<string, ReservedByRelationship>): ReservedByHierarchy {
    const classDependencies = new Map<string, string[]>();
    const constraintValidation = new Map<string, any>();
    const circularDependencies: string[] = [];

    // Build dependency graph
    for (const [key, relationship] of relationships) {
      const { sourceClass, targetClass } = relationship;

      // Add to source class dependencies
      if (!classDependencies.has(sourceClass)) {
        classDependencies.set(sourceClass, []);
      }
      classDependencies.get(sourceClass)!.push(targetClass);

      // Add constraint validation
      const validator = {
        validatorType: 'dependency' as const,
        rules: [{
          type: 'dependency_check',
          value: { source: sourceClass, target: targetClass },
          errorMessage: `${sourceClass} requires ${targetClass}`
        }],
        errorMessage: `Dependency validation failed for ${sourceClass} -> ${targetClass}`
      };
      constraintValidation.set(key, validator);
    }

    // Detect circular dependencies
    const visited = new Set<string>();
    const recursionStack = new Set<string>();

    for (const sourceClass of classDependencies.keys()) {
      if (!visited.has(sourceClass)) {
        const cycle = this.detectCircularDependency(sourceClass, classDependencies, visited, recursionStack);
        if (cycle.length > 0) {
          circularDependencies.push(cycle.join(' -> '));
        }
      }
    }

    return {
      totalRelationships: relationships.size,
      relationships,
      classDependencies,
      constraintValidation,
      circularDependencies
    };
  }

  /**
   * Validate MO hierarchy
   */
  private async validateMOHierarchy(
    hierarchy: MOHierarchy,
    errors: string[],
    warnings: string[]
  ): Promise<void> {
    for (const rule of this.hierarchyRules) {
      try {
        const result = rule.validator(hierarchy);
        if (!result.isValid) {
          if (rule.severity === 'error') {
            errors.push(`Hierarchy validation [${rule.name}]: ${result.message}`);
          } else {
            warnings.push(`Hierarchy validation [${rule.name}]: ${result.message}`);
          }
        }
      } catch (error) {
        errors.push(`Hierarchy rule [${rule.name}] failed: ${error}`);
      }
    }
  }

  /**
   * Build inheritance chain for a MO class
   */
  private buildInheritanceChain(moClass: string, moClasses: Map<string, MOMetaData>): string[] {
    const chain: string[] = [];
    let current = moClass;

    while (current && moClasses.has(current)) {
      chain.push(current);
      const currentData = moClasses.get(current)!;
      current = currentData.parent;
    }

    return chain.reverse(); // Return from root to leaf
  }

  /**
   * Detect circular dependencies using DFS
   */
  private detectCircularDependency(
    node: string,
    dependencies: Map<string, string[]>,
    visited: Set<string>,
    recursionStack: Set<string>
  ): string[] {
    visited.add(node);
    recursionStack.add(node);

    const neighbors = dependencies.get(node) || [];
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor)) {
        const cycle = this.detectCircularDependency(neighbor, dependencies, visited, recursionStack);
        if (cycle.length > 0) {
          return [node, ...cycle];
        }
      } else if (recursionStack.has(neighbor)) {
        return [node, neighbor];
      }
    }

    recursionStack.delete(node);
    return [];
  }

  /**
   * Validate LDN path format
   */
  private validateLDNPath(path: string[]): boolean {
    if (path.length === 0) return false;
    if (path.length > this.config.maxHierarchyDepth) return false;

    for (const segment of path) {
      if (!segment.match(/^[A-Za-z][A-Za-z0-9_]*(?:\[\d+\])?$/)) {
        return false;
      }
    }

    return true;
  }

  /**
   * Parse attributes from MO class definition
   */
  private parseAttributes(attributes?: string): string[] {
    if (!attributes) return [];

    return attributes.split(',')
      .map(attr => attr.trim())
      .filter(attr => attr.length > 0);
  }

  /**
   * Parse cardinality from attributes
   */
  private parseCardinality(attributes?: string): MOCardinality {
    if (!attributes) return { minimum: 1, maximum: 1, type: 'single' };

    const cardinalityMatch = attributes.match(/cardinality:\s*(\d+)\s*(?:-\s*(\d+|\*))?/);
    if (cardinalityMatch) {
      const min = parseInt(cardinalityMatch[1]);
      const maxStr = cardinalityMatch[2];
      const max = maxStr === '*' ? Infinity : parseInt(maxStr || String(min));

      return {
        minimum: min,
        maximum: max,
        type: max === 1 ? 'single' : max === Infinity ? 'unbounded' : 'bounded'
      };
    }

    return { minimum: 1, maximum: 1, type: 'single' };
  }

  /**
   * Extract class name from MO class string
   */
  private extractClassName(moClass: string): string {
    // Remove any suffixes or prefixes
    return moClass.replace(/^(class|interface)\s*/, '').replace(/\s*(abstract|final)$/, '');
  }

  /**
   * Extract display name from MO class string
   */
  private extractDisplayName(moClass: string): string {
    // Convert camelCase to Title Case
    return moClass.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase()).trim();
  }

  /**
   * Extract description from attributes
   */
  private extractDescription(attributes?: string): string | undefined {
    if (!attributes) return undefined;

    const descMatch = attributes.match(/description:\s*([^,]+)/);
    return descMatch ? descMatch[1].trim() : undefined;
  }

  /**
   * Extract version from attributes
   */
  private extractVersion(attributes?: string): string | undefined {
    if (!attributes) return undefined;

    const versionMatch = attributes.match(/version:\s*([^,]+)/);
    return versionMatch ? versionMatch[1].trim() : undefined;
  }

  /**
   * Convert string flags to object format
   */
  private convertFlags(flags: string[]): Record<string, any> {
    const flagObj: Record<string, any> = {};
    for (const flag of flags) {
      flagObj[flag] = true;
    }
    return flagObj;
  }

  /**
   * Count relationships in hierarchy
   */
  private countRelationships(hierarchy: MOHierarchy): number {
    return hierarchy.relationships.size;
  }

  /**
   * Initialize hierarchy validation rules
   */
  private initializeHierarchyRules(): void {
    this.hierarchyRules = [
      {
        name: 'root_class_exists',
        description: 'Verify that a root class exists',
        validator: (hierarchy) => {
          const rootExists = hierarchy.classes.has(hierarchy.rootClass);
          return {
            isValid: rootExists,
            message: rootExists ? 'Root class exists' : `Root class '${hierarchy.rootClass}' not found`
          };
        },
        severity: 'error'
      },
      {
        name: 'no_orphan_classes',
        description: 'Verify that all classes have valid parent references',
        validator: (hierarchy) => {
          let invalidCount = 0;
          for (const [className, moClass] of hierarchy.classes) {
            if (moClass.parentClass && !hierarchy.classes.has(moClass.parentClass)) {
              invalidCount++;
            }
          }
          return {
            isValid: invalidCount === 0,
            message: invalidCount === 0 ? 'All classes have valid parent references' : `${invalidCount} classes have invalid parent references`
          };
        },
        severity: 'error'
      },
      {
        name: 'max_hierarchy_depth',
        description: 'Verify hierarchy depth does not exceed maximum',
        validator: (hierarchy) => {
          let maxDepth = 0;
          for (const chain of hierarchy.inheritanceChain.values()) {
            maxDepth = Math.max(maxDepth, chain.length);
          }
          return {
            isValid: maxDepth <= this.config.maxHierarchyDepth,
            message: maxDepth <= this.config.maxHierarchyDepth ?
              `Hierarchy depth (${maxDepth}) within limits` :
              `Hierarchy depth (${maxDepth}) exceeds maximum (${this.config.maxHierarchyDepth})`
          };
        },
        severity: 'warning'
      }
    ];
  }
}