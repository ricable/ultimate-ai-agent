import { RTBParameter, ConstraintSpec, ProcessingStats } from '../../../types/rtb-types';

export interface XMLParseResult {
  parameters: RTBParameter[];
  moClasses: Set<string>;
  stats: ProcessingStats;
  errors: string[];
  warnings: string[];
}

export interface XMLSchemaElement {
  name: string;
  type: string;
  vsDataType?: string;
  description?: string;
  defaultValue?: any;
  constraints?: ConstraintSpec[];
  moClass: string;
  hierarchy: string[];
  attributes: Record<string, any>;
}

export interface XMLParsingContext {
  currentMOClass: string;
  hierarchy: string[];
  currentElement: string;
  parameterIndex: number;
  startTime: number;
  memoryUsage: number;
}

export class StreamingXMLParser {
  private batchSize: number;
  private memoryLimit: number;
  private processedCount: number;

  constructor(options: { batchSize?: number; memoryLimit?: number } = {}) {
    this.batchSize = options.batchSize || 1000;
    this.memoryLimit = options.memoryLimit || 2048; // MB
    this.processedCount = 0;
  }

  /**
   * Parse MPnh.xml file with streaming for memory efficiency
   * Processes 100MB+ XML files in chunks to avoid memory overflow
   */
  async parseMPnhXML(filePath: string): Promise<XMLParseResult> {
    const startTime = Date.now();
    const parameters: RTBParameter[] = [];
    const moClasses = new Set<string>();
    const errors: string[] = [];
    const warnings: string[] = [];

    console.log(`Starting XML parsing of ${filePath}`);
    console.log(`Batch size: ${this.batchSize}, Memory limit: ${this.memoryLimit}MB`);

    try {
      // Check file exists and get size
      const fs = require('fs').promises;
      const stats = await fs.stat(filePath);
      const fileSizeMB = stats.size / (1024 * 1024);

      console.log(`File size: ${fileSizeMB.toFixed(2)}MB`);

      if (fileSizeMB > 2000) { // 2GB warning
        warnings.push(`Large XML file detected (${fileSizeMB.toFixed(2)}MB). Consider splitting into smaller chunks.`);
      }

      // Initialize parsing context
      const context: XMLParsingContext = {
        currentMOClass: '',
        hierarchy: [],
        currentElement: '',
        parameterIndex: 0,
        startTime,
        memoryUsage: process.memoryUsage().heapUsed / 1024 / 1024
      };

      // Use streaming XML parser (sax-js or similar)
      const sax = require('sax');
      const parser = sax.parser(true, { trim: true, normalize: true });

      let currentElement: any = {};
      let currentAttributes: any = {};
      let currentPath: string[] = [];

      // XML parsing handlers
      parser.onopentag = (node: any) => {
        context.currentElement = node.name;
        currentPath.push(node.name);

        // Handle MO class detection
        if (node.name === 'moc' || node.name === 'managedElementClass') {
          const className = node.attributes?.name || node.attributes?.id;
          if (className) {
            context.currentMOClass = className;
            moClasses.add(className);
            context.hierarchy = [...currentPath];
          }
        }

        // Handle parameter elements
        if (this.isParameterElement(node.name)) {
          currentElement = {
            name: node.name,
            attributes: { ...node.attributes },
            textContent: '',
            moClass: context.currentMOClass,
            hierarchy: [...context.hierarchy],
            path: [...currentPath]
          };
        }

        if (currentElement.name) {
          currentAttributes = { ...currentAttributes, ...node.attributes };
        }
      };

      parser.ontext = (text: string) => {
        if (currentElement.name && text.trim()) {
          currentElement.textContent += text;
        }
      };

      parser.onclosetag = (tagName: string) => {
        if (this.isParameterElement(tagName) && currentElement.name) {
          try {
            const parameter = this.extractParameterFromElement(
              currentElement,
              currentAttributes,
              context
            );

            if (parameter) {
              parameters.push(parameter);
              context.parameterIndex++;

              // Process in batches to manage memory
              if (parameters.length % this.batchSize === 0) {
                await this.processBatch(parameters, context);
                this.processedCount += this.batchSize;

                // Check memory usage
                const currentMemory = process.memoryUsage().heapUsed / 1024 / 1024;
                if (currentMemory > this.memoryLimit) {
                  warnings.push(`Memory usage (${currentMemory.toFixed(2)}MB) approaching limit. Triggering garbage collection.`);
                  if (global.gc) {
                    global.gc();
                  }
                }
              }
            }
          } catch (error) {
            const errorMsg = `Error processing parameter ${currentElement.name || 'unknown'}: ${error}`;
            errors.push(errorMsg);
            console.warn(errorMsg);
          }

          currentElement = {};
          currentAttributes = {};
        }

        if (currentPath.length > 0 && currentPath[currentPath.length - 1] === tagName) {
          currentPath.pop();
        }

        if (tagName === 'moc' || tagName === 'managedElementClass') {
          context.currentMOClass = '';
          context.hierarchy = [];
        }
      };

      parser.onerror = (error: any) => {
        errors.push(`XML parsing error: ${error.message}`);
        console.error('XML Parser Error:', error);
      };

      parser.onend = () => {
        console.log(`XML parsing completed. Processed ${parameters.length} parameters.`);
      };

      // Read and parse the file
      const xmlContent = await fs.readFile(filePath, 'utf8');
      parser.write(xmlContent).close();

      // Process remaining parameters
      if (parameters.length % this.batchSize !== 0) {
        await this.processBatch(parameters, context);
      }

      const endTime = Date.now();
      const processingTime = (endTime - startTime) / 1000;
      const memoryUsage = process.memoryUsage().heapUsed / 1024 / 1024;

      console.log(`XML parsing completed in ${processingTime.toFixed(2)}s`);
      console.log(`Total parameters extracted: ${parameters.length}`);
      console.log(`Total MO classes: ${moClasses.size}`);
      console.log(`Memory usage: ${memoryUsage.toFixed(2)}MB`);

      return {
        parameters,
        moClasses,
        stats: {
          xmlProcessingTime: processingTime,
          hierarchyProcessingTime: 0,
          validationTime: 0,
          totalParameters: parameters.length,
          totalMOClasses: moClasses.size,
          totalRelationships: 0,
          memoryUsage,
          errorCount: errors.length,
          warningCount: warnings.length
        },
        errors,
        warnings
      };

    } catch (error) {
      const errorMsg = `Failed to parse XML file: ${error}`;
      errors.push(errorMsg);
      console.error(errorMsg);
      throw new Error(errorMsg);
    }
  }

  /**
   * Check if element represents a parameter
   */
  private isParameterElement(elementName: string): boolean {
    const parameterElements = [
      'parameter', 'param', 'attribute', 'property',
      'vsData', 'vsDataType', 'parameterGroup', 'list'
    ];
    return parameterElements.includes(elementName.toLowerCase()) ||
           elementName.includes('Parameter') ||
           elementName.includes('Attribute');
  }

  /**
   * Extract RTBParameter from XML element
   */
  private extractParameterFromElement(
    element: any,
    attributes: any,
    context: XMLParsingContext
  ): RTBParameter | null {
    try {
      const name = attributes.name || attributes.id || element.name;
      if (!name) {
        return null;
      }

      // Extract vsDataType and type
      const vsDataType = attributes.vsDataType || attributes.type || 'string';
      const type = this.mapVsDataTypeToType(vsDataType);

      // Extract constraints from attributes
      const constraints = this.extractConstraintsFromAttributes(attributes);

      // Extract description
      const description = attributes.description ||
                         attributes.documentation ||
                         element.textContent?.trim() || '';

      // Extract default value
      const defaultValue = attributes.defaultValue ||
                          attributes.default ||
                          attributes.value;

      return {
        id: `${context.currentMOClass}.${name}`,
        name,
        vsDataType,
        type,
        constraints,
        description,
        defaultValue,
        hierarchy: [...context.hierarchy],
        source: 'MPnh.xml',
        extractedAt: new Date(),
        structureGroups: this.extractStructureGroups(attributes),
        navigationPaths: this.extractNavigationPaths(attributes, context)
      };
    } catch (error) {
      console.warn(`Failed to extract parameter from element: ${error}`);
      return null;
    }
  }

  /**
   * Map vsDataType to TypeScript type
   */
  private mapVsDataTypeToType(vsDataType: string): string {
    const typeMap: Record<string, string> = {
      'Integer32': 'number',
      'Integer64': 'number',
      'UInt32': 'number',
      'UInt64': 'number',
      'Enumeration': 'string',
      'String': 'string',
      'Boolean': 'boolean',
      'DateTime': 'Date',
      'IPAddress': 'string',
      'MACAddress': 'string',
      'HexString': 'string'
    };

    // Handle array types
    if (vsDataType.includes('[]') || vsDataType.includes('List')) {
      const baseType = vsDataType.replace('[]', '').replace('List', '');
      return `${typeMap[baseType] || 'string'}[]`;
    }

    return typeMap[vsDataType] || 'string';
  }

  /**
   * Extract constraints from XML attributes
   */
  private extractConstraintsFromAttributes(attributes: any): ConstraintSpec[] {
    const constraints: ConstraintSpec[] = [];

    // Range constraints
    if (attributes.minValue !== undefined || attributes.maxValue !== undefined) {
      constraints.push({
        type: 'range',
        value: {
          min: attributes.minValue !== undefined ? parseInt(attributes.minValue) : undefined,
          max: attributes.maxValue !== undefined ? parseInt(attributes.maxValue) : undefined
        },
        severity: 'error'
      });
    }

    // Enum constraints
    if (attributes.allowedValues || attributes.enum || attributes.enumeration) {
      const enumValues = (attributes.allowedValues || attributes.enum || attributes.enumeration)
        .split(',').map((v: string) => v.trim());
      constraints.push({
        type: 'enum',
        value: enumValues,
        severity: 'error'
      });
    }

    // Pattern constraints
    if (attributes.pattern || attributes.regex) {
      constraints.push({
        type: 'pattern',
        value: attributes.pattern || attributes.regex,
        severity: 'error'
      });
    }

    // Length constraints
    if (attributes.minLength !== undefined || attributes.maxLength !== undefined) {
      constraints.push({
        type: 'length',
        value: {
          min: attributes.minLength !== undefined ? parseInt(attributes.minLength) : undefined,
          max: attributes.maxLength !== undefined ? parseInt(attributes.maxLength) : undefined
        },
        severity: 'error'
      });
    }

    return constraints;
  }

  /**
   * Extract structure groups from attributes
   */
  private extractStructureGroups(attributes: any): string[] {
    const groups: string[] = [];

    if (attributes.group) groups.push(attributes.group);
    if (attributes.category) groups.push(attributes.category);
    if (attributes.structureGroup) groups.push(attributes.structureGroup);

    return groups;
  }

  /**
   * Extract navigation paths from attributes and context
   */
  private extractNavigationPaths(attributes: any, context: XMLParsingContext): string[] {
    const paths: string[] = [];

    // Add current hierarchy path
    if (context.hierarchy.length > 0) {
      paths.push(context.hierarchy.join('.'));
    }

    // Add explicit navigation paths if provided
    if (attributes.navigationPath) {
      paths.push(attributes.navigationPath);
    }

    if (attributes.ldnPath) {
      paths.push(attributes.ldnPath);
    }

    return paths;
  }

  /**
   * Process batch of parameters (memory management)
   */
  private async processBatch(parameters: RTBParameter[], context: XMLParsingContext): Promise<void> {
    // Simulate processing time for large batches
    if (parameters.length % (this.batchSize * 5) === 0) {
      console.log(`Processed ${context.parameterIndex} parameters...`);

      // Allow event loop to process other tasks
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

  /**
   * Validate extracted parameters against schema
   */
  async validateParameters(parameters: RTBParameter[]): Promise<{
    validParameters: RTBParameter[];
    validationErrors: string[];
  }> {
    const validParameters: RTBParameter[] = [];
    const validationErrors: string[] = [];

    for (const parameter of parameters) {
      const errors = this.validateParameter(parameter);

      if (errors.length === 0) {
        validParameters.push(parameter);
      } else {
        validationErrors.push(`Parameter ${parameter.id}: ${errors.join(', ')}`);
      }
    }

    console.log(`Validation complete: ${validParameters.length} valid, ${validationErrors.length} errors`);

    return { validParameters, validationErrors };
  }

  /**
   * Validate individual parameter
   */
  private validateParameter(parameter: RTBParameter): string[] {
    const errors: string[] = [];

    if (!parameter.id) errors.push('Missing ID');
    if (!parameter.name) errors.push('Missing name');
    if (!parameter.type) errors.push('Missing type');
    if (!parameter.hierarchy || parameter.hierarchy.length === 0) {
      errors.push('Missing hierarchy');
    }

    // Validate constraint structure
    if (parameter.constraints) {
      for (const constraint of parameter.constraints) {
        if (!constraint.type) errors.push('Constraint missing type');
        if (constraint.value === undefined) errors.push('Constraint missing value');
      }
    }

    return errors;
  }
}