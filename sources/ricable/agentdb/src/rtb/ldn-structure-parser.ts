import { promises as fs } from 'fs';
import { LDNPattern, LDNHierarchy } from '../types/rtb-types';

export class LDNStructureParser {
  private patterns: LDNPattern[] = [];
  private hierarchy: LDNHierarchy = {
    rootPatterns: [],
    patternsByLevel: new Map(),
    lldnStructure: new Map(),
    navigationPaths: new Map()
  };

  async parseMomtlLDN(filePath: string): Promise<LDNPattern[]> {
    console.log(`[LDNStructureParser] Parsing LDN structure from ${filePath}`);

    try {
      const content = await fs.readFile(filePath, 'utf-8');
      const lines = content.split('\n').filter(line => line.trim());

      for (const line of lines) {
        this.parseLDNLine(line);
      }

      console.log(`[LDNStructureParser] Parsed ${this.patterns.length} LDN patterns`);
      console.log(`[LDNStructureParser] Found ${this.hierarchy.lldnStructure.size} LLDN structures`);

      return this.patterns;
    } catch (error) {
      console.error('[LDNStructureParser] Error parsing momtl_LDN.txt:', error);
      throw error;
    }
  }

  private parseLDNLine(line: string): void {
    const trimmedLine = line.trim();
    if (!trimmedLine || trimmedLine.startsWith('#')) {
      return;
    }

    // Format: ManagedElement[1],Legacy[0-1]
    // Format: ManagedElement[1],SystemFunctions[1],BrM[1]
    // Format: ManagedElement[1],SystemFunctions[1],Fm[1],FmAlarmModel[0-]

    const components = trimmedLine.split(',').map(comp => comp.trim());
    if (components.length < 2) {
      return;
    }

    const pattern: LDNPattern = {
      path: components,
      cardinality: this.extractCardinalityFromPath(components),
      hierarchyLevel: components.length,
      rootMO: components[0],
      leafMO: components[components.length - 1],
      isValidLDN: this.validateLDNPath(components),
      description: this.generateDescription(components)
    };

    this.patterns.push(pattern);
    this.updateHierarchy(pattern);
  }

  private extractCardinalityFromPath(components: string[]): Array<{ mo: string; cardinality: string }> {
    return components.map(component => {
      const cardinalityMatch = component.match(/\[(.+)\]/);
      const moName = component.replace(/\[.+?\]/, '');
      const cardinality = cardinalityMatch ? cardinalityMatch[1] : '1';

      return {
        mo: moName,
        cardinality
      };
    });
  }

  private validateLDNPath(components: string[]): boolean {
    // LDN path validation rules:
    // 1. Must start with ManagedElement
    // 2. Must follow valid MO class hierarchy
    // 3. Cannot have duplicate MO types at same level
    // 4. Must follow proper inheritance patterns

    if (components[0] !== 'ManagedElement[1]') {
      return false;
    }

    const seenMOs = new Set<string>();
    for (let i = 0; i < components.length; i++) {
      const moName = components[i].split('[')[0];
      const key = `${moName}_${i}`;

      if (seenMOs.has(key)) {
        return false; // Duplicate MO type at same level
      }
      seenMOs.add(key);

      // Validate MO class hierarchy (basic validation)
      if (i > 0 && !this.isValidParentChild(components[i - 1], components[i])) {
        return false;
      }
    }

    return true;
  }

  private isValidParentChild(parent: string, child: string): boolean {
    const parentMO = parent.split('[')[0];
    const childMO = child.split('[')[0];

    // Basic validation - can be extended with actual hierarchy knowledge
    const validParents: Record<string, string[]> = {
      'ManagedElement': ['SystemFunctions', 'GNBCUCP', 'GNBCUUP', 'GNBDU', 'Lrat', 'Wrat'],
      'SystemFunctions': ['BrM', 'Fm', 'Hcm', 'Lm', 'LogM', 'PmEventM', 'Pm', 'SecM'],
      'GNBCUCP': ['GNBCUCPFunction'],
      'GNBCUUP': ['GNBCUUPFunction'],
      'GNBDU': ['GNBDUFunction'],
      'Lrat': ['ENodeBFunction'],
      'Wrat': ['NodeBFunction']
    };

    const validChildren = validParents[parentMO];
    return validChildren ? validChildren.includes(childMO) : false;
  }

  private generateDescription(components: string[]): string {
    const moNames = components.map(comp => comp.split('[')[0]);
    return `LDN path: ${moNames.join(' → ')} (${components.length} levels)`;
  }

  private updateHierarchy(pattern: LDNPattern): void {
    // Add to root patterns
    if (pattern.hierarchyLevel === 1) {
      this.hierarchy.rootPatterns.push(pattern);
    }

    // Add to patterns by level
    const level = pattern.hierarchyLevel;
    if (!this.hierarchy.patternsByLevel.has(level)) {
      this.hierarchy.patternsByLevel.set(level, []);
    }
    this.hierarchy.patternsByLevel.get(level)!.push(pattern);

    // Add to LLDN structure
    const lldnKey = pattern.path.join(',');
    this.hierarchy.lldnStructure.set(lldnKey, pattern);

    // Generate navigation paths
    this.generateNavigationPaths(pattern);
  }

  private generateNavigationPaths(pattern: LDNPattern): void {
    // Generate all possible navigation paths from root to leaf
    for (let i = 1; i <= pattern.path.length; i++) {
      const pathSegment = pattern.path.slice(0, i).join(',');

      if (!this.hierarchy.navigationPaths.has(pathSegment)) {
        this.hierarchy.navigationPaths.set(pathSegment, []);
      }

      this.hierarchy.navigationPaths.get(pathSegment)!.push(pattern);
    }
  }

  getPatterns(): LDNPattern[] {
    return this.patterns;
  }

  getHierarchy(): LDNHierarchy {
    return this.hierarchy;
  }

  findPatternsByLevel(level: number): LDNPattern[] {
    return this.hierarchy.patternsByLevel.get(level) || [];
  }

  findPathsByRoot(rootMO: string): LDNPattern[] {
    return this.patterns.filter(pattern => pattern.rootMO === rootMO);
  }

  findPathsByLeaf(leafMO: string): LDNPattern[] {
    return this.patterns.filter(pattern => pattern.leafMO === leafMO);
  }

  getFullPaths(): LDNPattern[] {
    return this.patterns.filter(pattern => pattern.path.length > 2);
  }

  getShortPaths(): LDNPattern[] {
    return this.patterns.filter(pattern => pattern.path.length <= 2);
  }

  generateFDNPath(lDNPattern: LDNPattern, specificValues: Record<string, string> = {}): string {
    const fdnComponents = lDNPattern.path.map((component, index) => {
      const moName = component.split('[')[0];
      const cardinality = component.match(/\[(.+)\]/)?.[1] || '1';

      // Use specific value if provided, otherwise use pattern
      if (specificValues[moName]) {
        return `${moName}=${specificValues[moName]}`;
      } else if (index === 0) {
        return `${moName}=${index + 1}`; // ManagedElement[1]
      } else if (cardinality.includes('-')) {
        return `${moName}[${cardinality}]`; // Unbounded
      } else {
        return `${moName}=${cardinality}`; // Bounded
      }
    });

    return fdnComponents.join(',');
  }

  findCompatiblePaths(targetMO: string): LDNPattern[] {
    // Find paths that include the target MO
    const compatiblePaths: LDNPattern[] = [];

    for (const pattern of this.patterns) {
      const patternMOs = pattern.path.map(comp => comp.split('[')[0]);
      const targetIndex = patternMOs.indexOf(targetMO);

      if (targetIndex !== -1) {
        // Create a sub-path from the target MO onwards
        const subPath = {
          ...pattern,
          path: pattern.path.slice(targetIndex),
          rootMO: pattern.path[targetIndex],
          description: `Sub-path for ${targetMO}: ${pattern.path.slice(targetIndex).join(' → ')}`
        };

        compatiblePaths.push(subPath);
      }
    }

    return compatiblePaths;
  }

  exportAsJSON(): any {
    return {
      totalPatterns: this.patterns.length,
      hierarchyLevels: Array.from(this.hierarchy.patternsByLevel.keys()),
      rootPatterns: this.hierarchy.rootPatterns,
      lldnStructures: Array.from(this.hierarchy.lldnStructure.entries()),
      navigationPaths: Array.from(this.hierarchy.navigationPaths.entries())
    };
  }

  // Helper methods for navigation
  getFullPathForPattern(pattern: LDNPattern): string {
    return pattern.path.join(',');
  }

  getParentPath(pattern: LDNPattern): string | null {
    if (pattern.path.length > 1) {
      return pattern.path.slice(0, -1).join(',');
    }
    return null;
  }

  getChildPaths(pattern: LDNPattern): LDNPattern[] {
    return this.patterns.filter(p => {
      if (p.path.length <= pattern.path.length) return false;
      const parentSegment = p.path.slice(0, pattern.path.length);
      return parentSegment.join(',') === pattern.path.join(',');
    });
  }
}