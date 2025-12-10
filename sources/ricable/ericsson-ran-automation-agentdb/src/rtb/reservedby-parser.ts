import { promises as fs } from 'fs';
import { ReservedByRelationship, ReservedByHierarchy } from '../types/rtb-types';

export class ReservedByParser {
  private hierarchy: ReservedByHierarchy = {
    totalRelationships: 0,
    relationships: new Map(),
    classDependencies: new Map(),
    constraintValidation: new Map(),
    circularDependencies: []
  };

  async parseReservedBy(filePath: string): Promise<ReservedByHierarchy> {
    console.log(`[ReservedByParser] Parsing reservedBy relationships from ${filePath}`);

    try {
      const content = await fs.readFile(filePath, 'utf-8');
      const lines = content.split('\n').filter(line => line.trim());

      for (const line of lines) {
        this.parseReservedByLine(line);
      }

      console.log(`[ReservedByParser] Parsed ${this.hierarchy.totalRelationships} reservedBy relationships`);
      console.log(`[ReservedByParser] Found ${this.hierarchy.classDependencies.size} class dependencies`);

      return this.hierarchy;
    } catch (error) {
      console.error('[ReservedByParser] Error parsing reservedby.txt:', error);
      throw error;
    }
  }

  private parseReservedByLine(line: string): void {
    const trimmedLine = line.trim();
    if (!trimmedLine || trimmedLine.startsWith('#')) {
      return;
    }

    // Parse different relationship formats:
    // Format 1: "GNBCUCP.NRCellCU[0-] reserves 25+ profile classes"
    // Format 2: "GNBCUCP.NRFreqRelation[0-] reserved by cell relations"
    // Format 3: "GNBCUCP.ResourcePartition[0-] manages QoS tables"
    // Format 4: "TargetClass depends on SourceClass"
    // Format 5: "ClassA -> ClassB (relationship_type)"

    const relationship = this.parseRelationship(trimmedLine);
    if (relationship) {
      this.addRelationship(relationship);
    }
  }

  private parseRelationship(line: string): ReservedByRelationship | null {
    // Try different parsing patterns

    // Pattern 1: "Class reserves X classes"
    const reservesMatch = line.match(/^([^.]+)\.([^]+?)\s+reserves\s*(.+)$/);
    if (reservesMatch) {
      return {
        sourceClass: reservesMatch[1],
        targetClass: reservesMatch[2],
        relationshipType: 'reserves',
        cardinality: this.parseCardinality(reservesMatch[3]),
        description: `Reserves: ${reservesMatch[3]}`
      };
    }

    // Pattern 2: "Class reserved by other class"
    const reservedByMatch = line.match(/^([^.]+)\.([^]+?)\s+reserved\s+by\s+(.+)$/);
    if (reservedByMatch) {
      return {
        sourceClass: reservedByMatch[1],
        targetClass: reservedByMatch[2],
        relationshipType: 'depends_on',
        description: `Reserved by: ${reservedByMatch[3]}`
      };
    }

    // Pattern 3: "Class manages/manages_with something"
    const managesMatch = line.match(/^([^.]+)\.([^]+?)\s+(manages|manages_with|controls)\s+(.+)$/);
    if (managesMatch) {
      return {
        sourceClass: managesMatch[1],
        targetClass: managesMatch[2],
        relationshipType: 'modifies',
        description: `${managesMatch[3]}: ${managesMatch[4]}`
      };
    }

    // Pattern 4: "Class depends on Class"
    const dependsMatch = line.match(/^([^.]+)\.([^]+?)\s+depends\s+on\s+(.+)$/);
    if (dependsMatch) {
      return {
        sourceClass: dependsMatch[1],
        targetClass: dependsMatch[2],
        relationshipType: 'depends_on',
        description: `Depends on: ${dependsMatch[3]}`
      };
    }

    // Pattern 5: Arrow notation "Source -> Target (type)"
    const arrowMatch = line.match(/^([^.]+)\.([^]+?)\s+->\s+([^.]+)\.([^]+?)\s*\(([^)]+)\)/);
    if (arrowMatch) {
      return {
        sourceClass: arrowMatch[1],
        targetClass: arrowMatch[2],
        relationshipType: arrowMatch[5],
        description: `Arrow relationship: ${arrowMatch[3]}.${arrowMatch[4]} (${arrowMatch[5]})`
      };
    }

    // Pattern 6: Simple relationship "ClassA.ClassB relation_type"
    const simpleMatch = line.match(/^([^.]+)\.([^]+?)\s+(.+)$/);
    if (simpleMatch) {
      return {
        sourceClass: simpleMatch[1],
        targetClass: simpleMatch[2],
        relationshipType: 'depends_on',
        description: simpleMatch[3]
      };
    }

    return null;
  }

  private parseCardinality(description: string): { minimum: number; maximum: number; type: string } {
    // Parse cardinality from descriptions like "25+ profile classes", "0-256 entries", etc.
    const numberMatch = description.match(/(\d+)/);
    const plusMatch = description.match(/\+/);

    if (numberMatch && plusMatch) {
      return {
        minimum: parseInt(numberMatch[1]),
        maximum: Infinity,
        type: 'unbounded'
      };
    }

    const rangeMatch = description.match(/(\d+)-(\d+)/);
    if (rangeMatch) {
      return {
        minimum: parseInt(rangeMatch[1]),
        maximum: parseInt(rangeMatch[2]),
        type: 'bounded'
      };
    }

    // Default cardinality
    return {
      minimum: 1,
      maximum: 1,
      type: 'single'
    };
  }

  private addRelationship(relationship: ReservedByRelationship): void {
    const key = `${relationship.sourceClass}.${relationship.targetClass}`;

    // Add to relationships map
    this.hierarchy.relationships.set(key, relationship);
    this.hierarchy.totalRelationships++;

    // Update class dependencies
    if (!this.hierarchy.classDependencies.has(relationship.sourceClass)) {
      this.hierarchy.classDependencies.set(relationship.sourceClass, []);
    }
    this.hierarchy.classDependencies.get(relationship.sourceClass)!.push(relationship.targetClass);

    // Also add reverse dependency if it's a "reserves" relationship
    if (relationship.relationshipType === 'reserves') {
      if (!this.hierarchy.classDependencies.has(relationship.targetClass)) {
        this.hierarchy.classDependencies.set(relationship.targetClass, []);
      }
      const reverseDeps = this.hierarchy.classDependencies.get(relationship.targetClass)!;
      if (!reverseDeps.includes(relationship.sourceClass)) {
        reverseDeps.push(relationship.sourceClass);
      }
    }

    // Create constraint validators based on relationship type
    this.createConstraintValidator(relationship);
  }

  private createConstraintValidator(relationship: ReservedByRelationship): void {
    const validatorKey = `${relationship.sourceClass}.${relationship.targetClass}`;

    switch (relationship.relationshipType) {
      case 'reserves':
        this.hierarchy.constraintValidation.set(validatorKey, {
          validatorType: 'range',
          rules: [
            { type: 'min', value: relationship.cardinality.minimum },
            { type: 'max', value: relationship.cardinality.maximum }
          ],
          errorMessage: `Must reserve between ${relationship.cardinality.minimum} and ${relationship.cardinality.maximum} instances`
        });
        break;

      case 'depends_on':
        this.hierarchy.constraintValidation.set(validatorKey, {
          validatorType: 'custom',
          rules: [{ type: 'dependency_check' }],
          errorMessage: `Dependency validation for ${relationship.targetClass}`
        });
        break;

      case 'modifies':
        this.hierarchy.constraintValidation.set(validatorKey, {
          validatorType: 'pattern',
          rules: [{ type: 'modification_allowed' }],
          errorMessage: `Modification validation for ${relationship.targetClass}`
        });
        break;

      default:
        this.hierarchy.constraintValidation.set(validatorKey, {
          validatorType: 'custom',
          rules: [{ type: 'relationship_check' }],
          errorMessage: `Relationship validation for ${relationship.relationshipType}`
        });
        break;
    }
  }

  getHierarchy(): ReservedByHierarchy {
    return this.hierarchy;
  }

  findRelationshipsBySource(sourceClass: string): ReservedByRelationship[] {
    return Array.from(this.hierarchy.relationships.values())
      .filter(rel => rel.sourceClass === sourceClass);
  }

  findRelationshipsByTarget(targetClass: string): ReservedByRelationship[] {
    return Array.from(this.hierarchy.relationships.values())
      .filter(rel => rel.targetClass === targetClass);
  }

  getClassDependencies(className: string): string[] {
    return this.hierarchy.classDependencies.get(className) || [];
  }

  getDependencyChain(className: string, visited: Set<string> = new Set()): string[] {
    if (visited.has(className)) {
      return [className]; // Circular dependency
    }

    visited.add(className);
    const dependencies = this.getClassDependencies(className);
    const chain = [className];

    for (const dep of dependencies) {
      const depChain = this.getDependencyChain(dep, new Set(visited));
      chain.push(...depChain);
    }

    return chain;
  }

  findCircularDependencies(): Array<string[]> {
    const circularDeps: Array<string[]> = [];

    for (const className of this.hierarchy.classDependencies.keys()) {
      const chain = this.getDependencyChain(className);
      const firstIndex = chain.indexOf(className);

      if (firstIndex !== chain.length - 1) {
        const circularChain = chain.slice(firstIndex);
        if (!circularDeps.some(dep => this.arraysEqual(dep, circularChain))) {
          circularDeps.push(circularChain);
        }
      }
    }

    return circularDeps;
  }

  private arraysEqual(a: string[], b: string[]): boolean {
    return a.length === b.length && a.every((val, index) => val === b[index]);
  }

  getCardinalityConstraints(sourceClass: string, targetClass: string): { minimum: number; maximum: number; type: string } | null {
    const key = `${sourceClass}.${targetClass}`;
    const validator = this.hierarchy.constraintValidation.get(key);

    if (validator && validator.validatorType === 'range') {
      const minRule = validator.rules.find(r => r.type === 'min');
      const maxRule = validator.rules.find(r => r.type === 'max');

      return {
        minimum: minRule ? minRule.value : 1,
        maximum: maxRule ? maxRule.value : Infinity,
        type: 'bounded'
      };
    }

    return null;
  }

  validateRelationship(sourceClass: string, targetClass: string, instanceCount: number): { valid: boolean; errors: string[] } {
    const errors: string[] = [];
    const key = `${sourceClass}.${targetClass}`;
    const relationship = this.hierarchy.relationships.get(key);

    if (!relationship) {
      return { valid: false, errors: ['No relationship found'] };
    }

    // Check cardinality constraints
    const constraints = this.getCardinalityConstraints(sourceClass, targetClass);
    if (constraints) {
      if (instanceCount < constraints.minimum) {
        errors.push(`Instance count ${instanceCount} is less than minimum ${constraints.minimum}`);
      }
      if (instanceCount > constraints.maximum && constraints.maximum !== Infinity) {
        errors.push(`Instance count ${instanceCount} exceeds maximum ${constraints.maximum}`);
      }
    }

    // Check circular dependencies
    const circularDeps = this.findCircularDependencies();
    for (const cycle of circularDeps) {
      if (cycle.includes(sourceClass) && cycle.includes(targetClass)) {
        errors.push(`Circular dependency detected: ${cycle.join(' -> ')}`);
        break;
      }
    }

    return { valid: errors.length === 0, errors };
  }

  exportAsJSON(): any {
    return {
      totalRelationships: this.hierarchy.totalRelationships,
      relationships: Array.from(this.hierarchy.relationships.values()),
      classDependencies: Array.from(this.hierarchy.classDependencies.entries()),
      constraintValidation: Array.from(this.hierarchy.constraintValidation.entries()),
      circularDependencies: this.findCircularDependencies()
    };
  }

  // Helper methods for analysis
  getRelationshipStats(): {
    totalRelationships: number;
    relationshipTypes: Record<string, number>;
    sourceClasses: Record<string, number>;
    targetClasses: Record<string, number>;
    circularDependencies: number;
  } {
    const relationshipTypes: Record<string, number> = {};
    const sourceClasses: Record<string, number> = {};
    const targetClasses: Record<string, number> = {};

    for (const relationship of this.hierarchy.relationships.values()) {
      // Count relationship types
      relationshipTypes[relationship.relationshipType] = (relationshipTypes[relationship.relationshipType] || 0) + 1;

      // Count source classes
      sourceClasses[relationship.sourceClass] = (sourceClasses[relationship.sourceClass] || 0) + 1;

      // Count target classes
      targetClasses[relationship.targetClass] = (targetClasses[relationship.targetClass] || 0) + 1;
    }

    return {
      totalRelationships: this.hierarchy.totalRelationships,
      relationshipTypes,
      sourceClasses,
      targetClasses,
      circularDependencies: this.findCircularDependencies().length
    };
  }
}