/**
 * 3GPP Managed Object Model (MOM) XML Parser
 *
 * Parses Ericsson 3GPP MOM XML files containing:
 * - MO class definitions
 * - Attribute specifications
 * - Parameter ranges and constraints
 * - Class hierarchies
 *
 * Supports both LTE (TS 36.xxx) and 5G NR (TS 38.xxx) specifications
 */

import { XMLParser, XMLBuilder } from 'fast-xml-parser';
import { promises as fs } from 'fs';
import * as path from 'path';
import { v4 as uuidv4 } from 'uuid';
import type {
  ThreeGPPMOM,
  MOMClass,
  MOMAttribute,
  MOMHierarchyNode,
  DocumentChunk,
} from '../core/types.js';
import { logger } from '../utils/logger.js';

export interface ThreeGPPParserOptions {
  /** Parse enum values */
  parseEnums: boolean;
  /** Include deprecated attributes */
  includeDeprecated: boolean;
  /** Extract constraint information */
  extractConstraints: boolean;
  /** Max chunk size for RAG */
  maxChunkSize: number;
}

const DEFAULT_OPTIONS: ThreeGPPParserOptions = {
  parseEnums: true,
  includeDeprecated: false,
  extractConstraints: true,
  maxChunkSize: 2000,
};

// Common MOM element names across Ericsson formats
const MOM_ELEMENTS = {
  classes: ['class', 'moClass', 'managedObject', 'mo', 'moc'],
  attributes: ['attribute', 'attr', 'parameter', 'param'],
  name: ['name', 'moName', 'className', 'id'],
  type: ['type', 'dataType', 'valueType'],
  description: ['description', 'desc', 'synopsis', 'documentation'],
  range: ['range', 'valueRange', 'constraint', 'domain'],
  parent: ['parent', 'containedBy', 'parentClass'],
  children: ['children', 'contains', 'childClasses'],
};

export class ThreeGPPParser {
  private options: ThreeGPPParserOptions;
  private xmlParser: XMLParser;

  constructor(options: Partial<ThreeGPPParserOptions> = {}) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
    this.xmlParser = new XMLParser({
      ignoreAttributes: false,
      attributeNamePrefix: '@_',
      textNodeName: '#text',
      parseAttributeValue: true,
      parseTagValue: true,
      trimValues: true,
      ignoreDeclaration: true,
      removeNSPrefix: true,
    });
  }

  /**
   * Parse a directory of MOM XML files
   */
  async parseDirectory(dirPath: string): Promise<ThreeGPPMOM[]> {
    logger.info('Parsing 3GPP MOM directory', { dirPath });

    const files = await fs.readdir(dirPath);
    const xmlFiles = files.filter(
      (f) => f.endsWith('.xml') || f.endsWith('.XML')
    );

    const moms: ThreeGPPMOM[] = [];

    for (const file of xmlFiles) {
      try {
        const filePath = path.join(dirPath, file);
        const mom = await this.parseFile(filePath);
        if (mom) {
          moms.push(mom);
          logger.info('Parsed 3GPP MOM file', {
            file,
            classCount: mom.classes.size,
          });
        }
      } catch (error) {
        logger.error('Failed to parse 3GPP MOM file', {
          file,
          error: (error as Error).message,
        });
      }
    }

    return moms;
  }

  /**
   * Parse a single MOM XML file
   */
  async parseFile(filePath: string): Promise<ThreeGPPMOM | null> {
    const content = await fs.readFile(filePath, 'utf-8');
    return this.parseXML(content, filePath);
  }

  /**
   * Parse MOM XML content
   */
  parseXML(xmlContent: string, sourceFile: string): ThreeGPPMOM | null {
    try {
      const parsed = this.xmlParser.parse(xmlContent);
      const momId = uuidv4();

      // Detect format and extract classes
      const classes = this.extractClasses(parsed);

      if (classes.size === 0) {
        logger.warn('No classes found in MOM file', { sourceFile });
        return null;
      }

      // Build hierarchy
      const hierarchy = this.buildHierarchy(classes);

      // Determine technology
      const technology = this.detectTechnology(parsed, sourceFile, classes);

      // Extract name and version
      const { name, version } = this.extractMomInfo(parsed, sourceFile);

      return {
        id: momId,
        name,
        version,
        technology,
        classes,
        hierarchy,
        sourceFile,
        processedAt: new Date(),
      };
    } catch (error) {
      logger.error('Failed to parse MOM XML', {
        error: (error as Error).message,
        sourceFile,
      });
      return null;
    }
  }

  /**
   * Extract MO classes from parsed XML
   */
  private extractClasses(parsed: any): Map<string, MOMClass> {
    const classes = new Map<string, MOMClass>();

    // Try different root element structures
    const rootElements = this.findRootElements(parsed);

    for (const root of rootElements) {
      this.extractClassesFromElement(root, classes);
    }

    return classes;
  }

  /**
   * Find potential root elements containing MO classes
   */
  private findRootElements(parsed: any): any[] {
    const roots: any[] = [];

    const searchKeys = [
      'models',
      'mom',
      'managedObjectModel',
      'mim',
      'schema',
      'definitions',
      'classes',
      'moClasses',
    ];

    const search = (obj: any, depth = 0) => {
      if (depth > 5 || !obj || typeof obj !== 'object') return;

      for (const key of Object.keys(obj)) {
        const lowerKey = key.toLowerCase();

        // Check if this key might contain classes
        if (searchKeys.some((sk) => lowerKey.includes(sk))) {
          roots.push(obj[key]);
        }

        // Check if this element directly contains class definitions
        if (MOM_ELEMENTS.classes.some((c) => lowerKey === c)) {
          roots.push({ [key]: obj[key] });
        }

        // Recurse
        if (typeof obj[key] === 'object') {
          search(obj[key], depth + 1);
        }
      }
    };

    search(parsed);

    // If no specific roots found, search the entire parsed object
    if (roots.length === 0) {
      roots.push(parsed);
    }

    return roots;
  }

  /**
   * Extract classes from a parsed element
   */
  private extractClassesFromElement(
    element: any,
    classes: Map<string, MOMClass>
  ): void {
    if (!element || typeof element !== 'object') return;

    // Find class elements
    for (const classKey of MOM_ELEMENTS.classes) {
      if (element[classKey]) {
        const classElements = Array.isArray(element[classKey])
          ? element[classKey]
          : [element[classKey]];

        for (const classEl of classElements) {
          const moClass = this.parseClass(classEl);
          if (moClass) {
            classes.set(moClass.name, moClass);
          }
        }
      }
    }

    // Recurse into child elements
    for (const key of Object.keys(element)) {
      if (typeof element[key] === 'object' && !MOM_ELEMENTS.classes.includes(key.toLowerCase())) {
        this.extractClassesFromElement(element[key], classes);
      }
    }
  }

  /**
   * Parse a single MO class
   */
  private parseClass(classEl: any): MOMClass | null {
    if (!classEl || typeof classEl !== 'object') return null;

    const name = this.extractValue(classEl, MOM_ELEMENTS.name);
    if (!name) return null;

    const description = this.extractValue(classEl, MOM_ELEMENTS.description) || '';
    const parent = this.extractValue(classEl, MOM_ELEMENTS.parent);

    // Extract attributes
    const attributes = this.extractAttributes(classEl);

    // Extract child class references
    const children = this.extractChildren(classEl);

    // Look for spec reference
    const specReference = this.extractSpecReference(classEl);

    return {
      name,
      fqn: this.buildFQN(name, parent),
      parent,
      description,
      attributes,
      children,
      specReference,
    };
  }

  /**
   * Extract attributes from a class element
   */
  private extractAttributes(classEl: any): MOMAttribute[] {
    const attributes: MOMAttribute[] = [];

    for (const attrKey of MOM_ELEMENTS.attributes) {
      if (classEl[attrKey]) {
        const attrElements = Array.isArray(classEl[attrKey])
          ? classEl[attrKey]
          : [classEl[attrKey]];

        for (const attrEl of attrElements) {
          const attr = this.parseAttribute(attrEl);
          if (attr) {
            attributes.push(attr);
          }
        }
      }
    }

    return attributes;
  }

  /**
   * Parse a single attribute
   */
  private parseAttribute(attrEl: any): MOMAttribute | null {
    if (!attrEl || typeof attrEl !== 'object') return null;

    const name = this.extractValue(attrEl, MOM_ELEMENTS.name);
    if (!name) return null;

    const type = this.extractValue(attrEl, MOM_ELEMENTS.type) || 'string';
    const description = this.extractValue(attrEl, MOM_ELEMENTS.description) || '';
    const defaultValue = this.extractDefaultValue(attrEl);
    const range = this.options.extractConstraints
      ? this.extractRange(attrEl)
      : undefined;
    const reference = this.extractSpecReference(attrEl);

    // Determine flags
    const readOnly = this.checkFlag(attrEl, ['readOnly', 'readonly', 'immutable']);
    const mandatory = this.checkFlag(attrEl, ['mandatory', 'required', 'notNull']);

    // Check for deprecated
    const deprecated = this.checkFlag(attrEl, ['deprecated', 'obsolete']);
    if (deprecated && !this.options.includeDeprecated) {
      return null;
    }

    return {
      name,
      type,
      defaultValue,
      range,
      description,
      reference,
      readOnly,
      mandatory,
    };
  }

  /**
   * Extract range/constraint information
   */
  private extractRange(attrEl: any): MOMAttribute['range'] | undefined {
    // Look for range elements
    const rangeEl = this.findElement(attrEl, MOM_ELEMENTS.range);
    if (!rangeEl) return undefined;

    const range: MOMAttribute['range'] = {};

    // Numeric range
    if (rangeEl.min !== undefined || rangeEl['@_min'] !== undefined) {
      range.min = rangeEl.min ?? rangeEl['@_min'];
    }
    if (rangeEl.max !== undefined || rangeEl['@_max'] !== undefined) {
      range.max = rangeEl.max ?? rangeEl['@_max'];
    }

    // Enum values
    if (this.options.parseEnums) {
      const enumValues = this.extractEnumValues(attrEl);
      if (enumValues.length > 0) {
        range.enum = enumValues;
      }
    }

    return Object.keys(range).length > 0 ? range : undefined;
  }

  /**
   * Extract enum values
   */
  private extractEnumValues(attrEl: any): string[] {
    const values: string[] = [];

    const enumKeys = ['enum', 'enumeration', 'allowedValues', 'values', 'member'];

    for (const key of enumKeys) {
      const enumEl = this.findElement(attrEl, [key]);
      if (enumEl) {
        if (Array.isArray(enumEl)) {
          for (const e of enumEl) {
            const val = typeof e === 'object' ? (e.name || e['@_name'] || e['#text']) : e;
            if (val) values.push(String(val));
          }
        } else if (typeof enumEl === 'object') {
          // Check for nested value elements
          const valueEl = enumEl.value || enumEl.member || enumEl.item;
          if (Array.isArray(valueEl)) {
            for (const v of valueEl) {
              const val = typeof v === 'object' ? (v.name || v['@_name'] || v['#text']) : v;
              if (val) values.push(String(val));
            }
          }
        }
      }
    }

    return values;
  }

  /**
   * Extract child class names
   */
  private extractChildren(classEl: any): string[] {
    const children: string[] = [];

    for (const childKey of MOM_ELEMENTS.children) {
      const childEl = classEl[childKey];
      if (childEl) {
        if (Array.isArray(childEl)) {
          for (const c of childEl) {
            const name = typeof c === 'object' ? (c.name || c['@_name']) : c;
            if (name) children.push(String(name));
          }
        } else if (typeof childEl === 'string') {
          children.push(childEl);
        }
      }
    }

    return children;
  }

  /**
   * Build class hierarchy from extracted classes
   */
  private buildHierarchy(classes: Map<string, MOMClass>): MOMHierarchyNode {
    // Find root classes (no parent)
    const roots: string[] = [];

    for (const [name, cls] of classes) {
      if (!cls.parent || !classes.has(cls.parent)) {
        roots.push(name);
      }
    }

    // Build tree recursively
    const buildNode = (className: string): MOMHierarchyNode => {
      const cls = classes.get(className);
      const children: MOMHierarchyNode[] = [];

      // Find direct children
      for (const [name, c] of classes) {
        if (c.parent === className) {
          children.push(buildNode(name));
        }
      }

      // Also include declared children
      if (cls?.children) {
        for (const childName of cls.children) {
          if (classes.has(childName) && !children.some((c) => c.className === childName)) {
            children.push(buildNode(childName));
          }
        }
      }

      return {
        className,
        children,
      };
    };

    // Create virtual root
    return {
      className: 'ROOT',
      children: roots.map(buildNode),
    };
  }

  /**
   * Detect technology (LTE/NR) from content
   */
  private detectTechnology(
    parsed: any,
    sourceFile: string,
    classes: Map<string, MOMClass>
  ): 'LTE' | 'NR' {
    const content = JSON.stringify(parsed).toLowerCase();
    const filename = sourceFile.toLowerCase();

    // Check file name
    if (filename.includes('nr') || filename.includes('5g') || filename.includes('gnb')) {
      return 'NR';
    }
    if (filename.includes('lte') || filename.includes('enb') || filename.includes('eutra')) {
      return 'LTE';
    }

    // Check content
    const nrIndicators = ['gnb', 'nrcell', 'nrdu', 'nrcu', 'nr-', '38.'];
    const lteIndicators = ['enb', 'eutran', 'eutra', 'lte-', '36.'];

    let nrScore = 0;
    let lteScore = 0;

    for (const ind of nrIndicators) {
      if (content.includes(ind)) nrScore++;
    }
    for (const ind of lteIndicators) {
      if (content.includes(ind)) lteScore++;
    }

    // Check class names
    for (const className of classes.keys()) {
      const lower = className.toLowerCase();
      if (lower.includes('nr') || lower.includes('gnb')) nrScore++;
      if (lower.includes('lte') || lower.includes('eutra') || lower.includes('enb')) lteScore++;
    }

    return nrScore > lteScore ? 'NR' : 'LTE';
  }

  /**
   * Extract MOM name and version
   */
  private extractMomInfo(
    parsed: any,
    sourceFile: string
  ): { name: string; version: string } {
    // Try to find in document metadata
    const name =
      parsed.mom?.name ||
      parsed.mom?.['@_name'] ||
      parsed.models?.name ||
      path.basename(sourceFile, path.extname(sourceFile));

    const version =
      parsed.mom?.version ||
      parsed.mom?.['@_version'] ||
      parsed.models?.version ||
      '1.0.0';

    return { name: String(name), version: String(version) };
  }

  /**
   * Extract spec reference (3GPP TS number)
   */
  private extractSpecReference(el: any): string | undefined {
    const refKeys = ['specRef', 'reference', 'specification', '3gppRef', 'tsRef'];

    for (const key of refKeys) {
      if (el[key]) {
        return String(el[key]);
      }
      if (el[`@_${key}`]) {
        return String(el[`@_${key}`]);
      }
    }

    // Look for TS pattern in description
    const desc = this.extractValue(el, MOM_ELEMENTS.description);
    if (desc) {
      const tsMatch = desc.match(/TS\s*(\d{2}\.\d{3})/i);
      if (tsMatch) {
        return `3GPP ${tsMatch[0]}`;
      }
    }

    return undefined;
  }

  /**
   * Helper to extract a value from an element
   */
  private extractValue(el: any, keys: string[]): string | undefined {
    if (!el || typeof el !== 'object') return undefined;

    for (const key of keys) {
      if (el[key] !== undefined) {
        const val = el[key];
        return typeof val === 'object' ? val['#text'] : String(val);
      }
      if (el[`@_${key}`] !== undefined) {
        return String(el[`@_${key}`]);
      }
    }

    return undefined;
  }

  /**
   * Helper to find an element
   */
  private findElement(el: any, keys: string[]): any {
    if (!el || typeof el !== 'object') return undefined;

    for (const key of keys) {
      if (el[key] !== undefined) return el[key];
    }

    return undefined;
  }

  /**
   * Helper to check a boolean flag
   */
  private checkFlag(el: any, keys: string[]): boolean {
    for (const key of keys) {
      if (el[key] === true || el[key] === 'true' || el[key] === '1') return true;
      if (el[`@_${key}`] === true || el[`@_${key}`] === 'true') return true;
    }
    return false;
  }

  /**
   * Extract default value
   */
  private extractDefaultValue(attrEl: any): string | undefined {
    const defaultKeys = ['default', 'defaultValue', 'initial', 'initialValue'];

    for (const key of defaultKeys) {
      if (attrEl[key] !== undefined) {
        const val = attrEl[key];
        return typeof val === 'object' ? val['#text'] : String(val);
      }
    }

    return undefined;
  }

  /**
   * Build fully qualified name
   */
  private buildFQN(name: string, parent?: string): string {
    return parent ? `${parent}.${name}` : name;
  }

  /**
   * Convert MOM to chunks for RAG
   */
  chunkMOM(mom: ThreeGPPMOM): DocumentChunk[] {
    const chunks: DocumentChunk[] = [];
    let chunkIndex = 0;

    for (const [className, cls] of mom.classes) {
      // Create chunk for class definition
      const classContent = this.formatClassForChunk(cls, mom.technology);
      chunks.push({
        id: `${mom.id}-chunk-${chunkIndex}`,
        documentId: mom.id,
        documentType: '3gpp',
        content: classContent,
        metadata: {
          title: `${mom.technology} MOM: ${className}`,
          section: 'Class Definition',
          chunkIndex,
          totalChunks: 0,
          sourceFile: mom.sourceFile,
          technology: mom.technology,
          momClass: className,
        },
        tokenCount: this.estimateTokens(classContent),
      });
      chunkIndex++;

      // Create chunks for attributes (grouped)
      const attrChunks = this.chunkAttributes(cls.attributes, className, mom);
      for (const attrContent of attrChunks) {
        chunks.push({
          id: `${mom.id}-chunk-${chunkIndex}`,
          documentId: mom.id,
          documentType: '3gpp',
          content: attrContent,
          metadata: {
            title: `${mom.technology} MOM: ${className} Attributes`,
            section: 'Attributes',
            chunkIndex,
            totalChunks: 0,
            sourceFile: mom.sourceFile,
            technology: mom.technology,
            momClass: className,
          },
          tokenCount: this.estimateTokens(attrContent),
        });
        chunkIndex++;
      }
    }

    // Update total chunks
    for (const chunk of chunks) {
      chunk.metadata.totalChunks = chunks.length;
    }

    return chunks;
  }

  /**
   * Format a class for chunking
   */
  private formatClassForChunk(cls: MOMClass, technology: string): string {
    const lines: string[] = [
      `# ${technology} Managed Object Class: ${cls.name}`,
      '',
    ];

    if (cls.fqn !== cls.name) {
      lines.push(`**Fully Qualified Name:** ${cls.fqn}`);
    }

    if (cls.parent) {
      lines.push(`**Parent Class:** ${cls.parent}`);
    }

    if (cls.specReference) {
      lines.push(`**3GPP Reference:** ${cls.specReference}`);
    }

    lines.push('');

    if (cls.description) {
      lines.push('## Description');
      lines.push(cls.description);
      lines.push('');
    }

    if (cls.children.length > 0) {
      lines.push('## Child Classes');
      for (const child of cls.children) {
        lines.push(`- ${child}`);
      }
      lines.push('');
    }

    lines.push(`**Total Attributes:** ${cls.attributes.length}`);

    return lines.join('\n');
  }

  /**
   * Chunk attributes with size limit
   */
  private chunkAttributes(
    attributes: MOMAttribute[],
    className: string,
    mom: ThreeGPPMOM
  ): string[] {
    const chunks: string[] = [];
    let currentChunk: string[] = [
      `# ${mom.technology} MOM Attributes: ${className}`,
      '',
    ];
    let currentSize = currentChunk.join('\n').length;

    for (const attr of attributes) {
      const attrText = this.formatAttribute(attr);
      const attrSize = attrText.length;

      if (currentSize + attrSize > this.options.maxChunkSize && currentChunk.length > 2) {
        chunks.push(currentChunk.join('\n'));
        currentChunk = [
          `# ${mom.technology} MOM Attributes: ${className} (continued)`,
          '',
        ];
        currentSize = currentChunk.join('\n').length;
      }

      currentChunk.push(attrText);
      currentSize += attrSize;
    }

    if (currentChunk.length > 2) {
      chunks.push(currentChunk.join('\n'));
    }

    return chunks;
  }

  /**
   * Format a single attribute
   */
  private formatAttribute(attr: MOMAttribute): string {
    const lines: string[] = [
      `## ${attr.name}`,
      '',
      `**Type:** ${attr.type}`,
    ];

    if (attr.mandatory) {
      lines.push('**Mandatory:** Yes');
    }

    if (attr.readOnly) {
      lines.push('**Read-Only:** Yes');
    }

    if (attr.defaultValue !== undefined) {
      lines.push(`**Default:** ${attr.defaultValue}`);
    }

    if (attr.range) {
      if (attr.range.min !== undefined || attr.range.max !== undefined) {
        const rangeStr = `[${attr.range.min ?? ''}..${attr.range.max ?? ''}]`;
        lines.push(`**Range:** ${rangeStr}`);
      }
      if (attr.range.enum && attr.range.enum.length > 0) {
        lines.push(`**Allowed Values:** ${attr.range.enum.join(', ')}`);
      }
    }

    if (attr.reference) {
      lines.push(`**Reference:** ${attr.reference}`);
    }

    if (attr.description) {
      lines.push('');
      lines.push(attr.description);
    }

    lines.push('');

    return lines.join('\n');
  }

  /**
   * Estimate token count
   */
  private estimateTokens(text: string): number {
    return Math.ceil(text.length / 4);
  }

  /**
   * Find parameter by name across all MOMs
   */
  findParameter(moms: ThreeGPPMOM[], paramName: string): MOMAttribute | null {
    const lowerName = paramName.toLowerCase();

    for (const mom of moms) {
      for (const cls of mom.classes.values()) {
        for (const attr of cls.attributes) {
          if (attr.name.toLowerCase() === lowerName) {
            return attr;
          }
        }
      }
    }

    return null;
  }

  /**
   * Get parameter hierarchy (class path)
   */
  getParameterPath(
    moms: ThreeGPPMOM[],
    paramName: string
  ): { mom: string; className: string; attribute: MOMAttribute } | null {
    const lowerName = paramName.toLowerCase();

    for (const mom of moms) {
      for (const [className, cls] of mom.classes) {
        for (const attr of cls.attributes) {
          if (attr.name.toLowerCase() === lowerName) {
            return { mom: mom.name, className, attribute: attr };
          }
        }
      }
    }

    return null;
  }
}

export default ThreeGPPParser;
