import { promises as fs } from 'fs';
import { MOClass, MOHierarchy, MOCardinality } from '../types/rtb-types';

export class MOHierarchyParser {
  private hierarchy: MOHierarchy = {
    rootClass: 'ComTop.ManagedElement',
    classes: new Map(),
    relationships: new Map(),
    cardinality: new Map(),
    inheritanceChain: new Map()
  };

  async parseMomtTree(filePath: string): Promise<MOHierarchy> {
    console.log(`[MOHierarchyParser] Parsing MO hierarchy from ${filePath}`);

    try {
      const content = await fs.readFile(filePath, 'utf-8');
      const lines = content.split('\n').filter(line => line.trim());

      for (const line of lines) {
        this.parseMomtLine(line);
      }

      console.log(`[MOHierarchyParser] Parsed ${this.hierarchy.classes.size} MO classes`);
      console.log(`[MOHierarchyParser] Found ${this.hierarchy.relationships.size} relationships`);

      return this.hierarchy;
    } catch (error) {
      console.error('[MOHierarchyParser] Error parsing momt_tree.txt:', error);
      throw error;
    }
  }

  private parseMomtLine(line: string): void {
    // Format: ParentClass.Class[cardinality] (systemCreated/optional/systemMandatory)
    // Example: ComTop.ManagedElement (systemCreated)
    // Example: GNBCUCP.GNBCUCPFunction[0-] (systemCreated)
    // Example: GNBCUCP.NRCellCU[0-] reserves 25+ profile classes

    const trimmedLine = line.trim();
    if (!trimmedLine || trimmedLine.startsWith('#')) {
      return;
    }

    // Parse main structure
    const structureMatch = trimmedLine.match(/^([^.]+)\.([^(]+)(?:\s*\(([^)]+)\))?/);
    if (!structureMatch) return;

    const parentClass = structureMatch[1];
    const className = structureMatch[2].split(' ')[0]; // Remove cardinality for class name
    const cardinalityStr = structureMatch[2].match(/(\[.*?\])/);
    const cardinality = cardinalityStr ? cardinalityStr[1] : '[1-1]';
    const flags = structureMatch[3] || '';

    // Parse cardinality
    const cardinalityInfo = this.parseCardinality(cardinality, className);

    // Parse flags
    const flagsObj = this.parseFlags(flags);

    // Create MO class
    const moClass: MOClass = {
      id: `${parentClass}.${className}`,
      name: className,
      parentClass,
      cardinality: cardinalityInfo,
      flags: flagsObj,
      children: [],
      attributes: [],
      derivedClasses: []
    };

    // Add to hierarchy
    this.hierarchy.classes.set(moClass.id, moClass);

    // Update parent's children
    if (parentClass !== 'ComTop') {
      const parent = this.hierarchy.classes.get(`${parentClass}.${className.split('[')[0]}`);
      if (parent) {
        parent.children.push(className);
      }
    }

    // Handle reserved relationships
    if (trimmedLine.includes('reserves')) {
      const reservedMatch = trimmedLine.match(/reserves\s*(.+)/);
      if (reservedMatch) {
        this.parseReservedRelationship(moClass.id, reservedMatch[1]);
      }
    }

    // Add to inheritance chain
    this.updateInheritanceChain(moClass);
  }

  private parseCardinality(cardinality: string, className: string): MOCardinality {
    // Parse patterns like [0-], [1-1], [0-256], [0-1]
    const minMaxMatch = cardinality.match(/\[(\d+)-(\d+)\]/);
    if (minMaxMatch) {
      return {
        minimum: parseInt(minMaxMatch[1]),
        maximum: parseInt(minMaxMatch[2]),
        type: 'bounded'
      };
    }

    const unboundedMatch = cardinality.match(/\[(\d+)-\]/);
    if (unboundedMatch) {
      return {
        minimum: parseInt(unboundedMatch[1]),
        maximum: Infinity,
        type: 'unbounded'
      };
    }

    const singleMatch = cardinality.match(/\[(\d+)-(\d+)\]/);
    if (singleMatch && singleMatch[1] === singleMatch[2]) {
      return {
        minimum: parseInt(singleMatch[1]),
        maximum: parseInt(singleMatch[2]),
        type: 'single'
      };
    }

    // Default cardinality
    return {
      minimum: 1,
      maximum: 1,
      type: 'single'
    };
  }

  private parseFlags(flags: string): Record<string, any> {
    const result: Record<string, any> = {};

    if (flags) {
      const flagList = flags.split(',').map(f => f.trim());
      flagList.forEach(flag => {
        if (flag === 'systemCreated') {
          result.systemCreated = true;
        } else if (flag === 'optional') {
          result.optional = true;
        } else if (flag === 'systemMandatory') {
          result.systemMandatory = true;
        } else {
          result[flag] = true;
        }
      });
    }

    return result;
  }

  private parseReservedRelationship(parentId: string, reservedInfo: string): void {
    // Parse "reserves 25+ profile classes" or similar patterns
    const countMatch = reservedInfo.match(/(\d+)\+/);
    if (countMatch) {
      this.hierarchy.relationships.set(parentId, {
        parentId,
        relationType: 'reserves',
        count: parseInt(countMatch[1]),
        description: reservedInfo.trim()
      });
    } else {
      this.hierarchy.relationships.set(parentId, {
        parentId,
        relationType: 'reserves',
        description: reservedInfo.trim()
      });
    }
  }

  private updateInheritanceChain(moClass: MOClass): void {
    const chain = [moClass.id];
    let current = moClass.parentClass;

    while (current && current !== 'ComTop') {
      chain.push(current);
      const parent = this.hierarchy.classes.get(current);
      if (parent) {
        current = parent.parentClass;
      } else {
        break;
      }
    }

    this.hierarchy.inheritanceChain.set(moClass.id, chain.reverse());
  }

  getHierarchy(): MOHierarchy {
    return this.hierarchy;
  }

  findClass(className: string): MOClass | undefined {
    return this.hierarchy.classes.get(className);
  }

  getClassHierarchy(className: string): string[] {
    return this.hierarchy.inheritanceChain.get(className) || [];
  }

  getChildrenOfClass(className: string): MOClass[] {
    return Array.from(this.hierarchy.classes.values())
      .filter(cls => cls.parentClass === className);
  }

  getParentClass(className: string): MOClass | undefined {
    const classInfo = this.hierarchy.classes.get(className);
    if (classInfo && classInfo.parentClass) {
      return this.hierarchy.classes.get(classInfo.parentClass);
    }
    return undefined;
  }

  getCardinalityInfo(className: string): MOCardinality {
    const classInfo = this.hierarchy.classes.get(className);
    return classInfo ? classInfo.cardinality : {
      minimum: 1,
      maximum: 1,
      type: 'single'
    };
  }

  getClassByParent(parentClass: string): MOClass[] {
    return Array.from(this.hierarchy.classes.values())
      .filter(cls => cls.parentClass === parentClass);
  }

  getAllReservations(): Array<{ parentId: string; relationType: string; count?: number; description: string }> {
    return Array.from(this.hierarchy.relationships.values());
  }

  exportAsJSON(): any {
    return {
      rootClass: this.hierarchy.rootClass,
      totalClasses: this.hierarchy.classes.size,
      totalRelationships: this.hierarchy.relationships.size,
      classes: Array.from(this.hierarchy.classes.values()).map(cls => ({
        id: cls.id,
        name: cls.name,
        parentClass: cls.parentClass,
        cardinality: cls.cardinality,
        flags: cls.flags,
        children: cls.children,
        derivedClasses: cls.derivedClasses
      })),
      relationships: Array.from(this.hierarchy.relationships.values()),
      inheritanceChains: Array.from(this.hierarchy.inheritanceChain.entries())
    };
  }
}