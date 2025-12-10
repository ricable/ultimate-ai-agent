import { createReadStream, promises as fs } from 'fs';
import { parseStringPromise } from 'xml2js';
import { RTBParameter } from '../types/rtb-types';
import { StreamingXMLParser } from './streaming-xml-parser';

export interface XMLParameter {
  name: string;
  vsData: string;
  type: string;
  constraints?: Record<string, any>;
  description?: string;
  hierarchy?: string[];
}

export class RTBXMLParser {
  private parameterCache = new Map<string, RTBParameter>();
  private vsDataTypes = new Set<string>();
  private processedCount = 0;
  private startTime = Date.now();

  async parseFile(xmlPath: string): Promise<RTBParameter[]> {
    console.log(`[RTBXMLParser] Starting parse of ${xmlPath}`);
    const startTime = Date.now();

    try {
      // Use streaming parser for memory efficiency
      const streamingParser = new StreamingXMLParser();
      const parameters = await streamingParser.parseFile(xmlPath);

      const processingTime = Date.now() - startTime;
      console.log(`[RTBXMLParser] Parsed ${parameters.length} parameters in ${processingTime}ms`);

      // Extract unique vsData types
      this.vsDataTypes = new Set(parameters.map(p => p.vsDataType));
      console.log(`[RTBXMLParser] Found ${this.vsDataTypes.size} unique vsData types`);

      return parameters;
    } catch (error) {
      console.error('[RTBXMLParser] Parse error:', error);
      throw error;
    }
  }

  async parseWithMemoryOptimization(xmlPath: string): Promise<RTBParameter[]> {
    console.log(`[RTBXMLParser] Memory-optimized parse of ${xmlPath}`);

    const fileHandle = await fs.open(xmlPath, 'r');
    const stats = await fileHandle.stat();
    const fileSize = stats.size;

    console.log(`[RTBXMLParser] File size: ${(fileSize / 1024 / 1024).toFixed(2)}MB`);

    const parameters: RTBParameter[] = [];
    const batchSize = 1000;
    let batch: RTBParameter[] = [];

    const stream = createReadStream(xmlPath, {
      highWaterMark: 64 * 1024 * 1024 // 64MB chunks
    });

    let currentChunk = '';
    let elementDepth = 0;

    return new Promise((resolve, reject) => {
      stream.on('data', (chunk: Buffer) => {
        currentChunk += chunk.toString();

        // Process complete elements
        const elements = this.extractCompleteElements(currentChunk);
        if (elements.length > 0) {
          const parsedElements = this.parseElements(elements);
          batch.push(...parsedElements);

          // Process in batches to manage memory
          if (batch.length >= batchSize) {
            parameters.push(...batch);
            batch = [];
            console.log(`[RTBXMLParser] Processed ${parameters.length} parameters...`);
          }

          // Keep remaining partial data
          currentChunk = elements[elements.length - 1].slice(-1000); // Keep end of last element
        }
      });

      stream.on('end', async () => {
        // Process remaining batch
        if (batch.length > 0) {
          parameters.push(...batch);
        }

        console.log(`[RTBXMLParser] Memory-optimized complete. Total parameters: ${parameters.length}`);
        await fileHandle.close();
        resolve(parameters);
      });

      stream.on('error', (error) => {
        reject(error);
      });
    });
  }

  private extractCompleteElements(chunk: string): string[] {
    const elements: string[] = [];
    let startIndex = 0;
    let depth = 0;

    for (let i = 0; i < chunk.length; i++) {
      if (chunk[i] === '<') {
        if (chunk.substring(i, i + 2) === '</') {
          depth--;
          if (depth === 0 && i > startIndex) {
            elements.push(chunk.substring(startIndex, i + 1));
            startIndex = i + 1;
          }
        } else if (chunk[i + 1] !== '?' && chunk[i + 1] !== '!') {
          depth++;
          if (depth === 1) {
            startIndex = i;
          }
        }
      }
    }

    return elements;
  }

  private parseElements(elementStrings: string[]): RTBParameter[] {
    return elementStrings
      .filter(element => element.includes('vsData'))
      .map(element => this.parseXMLElement(element))
      .filter(Boolean) as RTBParameter[];
  }

  private parseXMLElement(element: string): RTBParameter | null {
    try {
      // Extract parameter name
      const nameMatch = element.match(/name="([^"]+)"/);
      if (!nameMatch) return null;

      const name = nameMatch[1];

      // Extract vsData type
      const vsDataMatch = element.match(/vsData="([^"]+)"/);
      if (!vsDataMatch) return null;

      const vsDataType = vsDataMatch[1];

      // Extract type
      const typeMatch = element.match(/type="([^"]+)"/);
      const type = typeMatch ? typeMatch[1] : 'string';

      // Extract constraints if present
      const constraints: Record<string, any> = {};
      const constraintsMatches = element.matchAll(/constraint\.([^.]+)="([^"]+)"/g);
      for (const match of constraintsMatches) {
        constraints[match[1]] = match[2];
      }

      // Build hierarchy from element structure
      const hierarchy = this.extractHierarchy(element);

      return {
        id: `${vsDataType}_${name}`,
        name,
        vsDataType,
        type: this.mapXMLTypeToTypeScript(type),
        constraints,
        description: this.extractDescription(element),
        hierarchy,
        source: 'MPnh.xml',
        extractedAt: new Date()
      };
    } catch (error) {
      console.warn(`[RTBXMLParser] Failed to parse element: ${error}`);
      return null;
    }
  }

  private mapXMLTypeToTypeScript(xmlType: string): string {
    const typeMap: Record<string, string> = {
      'int': 'number',
      'integer': 'number',
      'string': 'string',
      'boolean': 'boolean',
      'float': 'number',
      'double': 'number',
      'dateTime': 'string',
      'date': 'string',
      'time': 'string',
      'list': 'any[]',
      'array': 'any[]',
      'object': 'object'
    };

    return typeMap[xmlType] || 'any';
  }

  private extractHierarchy(element: string): string[] {
    const hierarchy: string[] = [];

    // Extract namespace hierarchy
    const namespaceMatch = element.match(/SubNetwork[^>]*>/);
    if (namespaceMatch) {
      hierarchy.push('SubNetwork');
    }

    const meContextMatch = element.match(/MeContext[^>]*>/);
    if (meContextMatch) {
      hierarchy.push('MeContext');
    }

    const managedElementMatch = element.match(/ManagedElement[^>]*>/);
    if (managedElementMatch) {
      hierarchy.push('ManagedElement');
    }

    // Extract parent MO classes
    const parentMatches = element.matchAll(/([A-Z][a-zA-Z]*)[^>]*>/g);
    for (const match of parentMatches) {
      if (!['SubNetwork', 'MeContext', 'ManagedElement'].includes(match[1])) {
        hierarchy.push(match[1]);
      }
    }

    return hierarchy;
  }

  private extractDescription(element: string): string | undefined {
    const descriptionMatch = element.match(/description="([^"]+)"/);
    return descriptionMatch ? descriptionMatch[1] : undefined;
  }

  getVsDataTypes(): Set<string> {
    return this.vsDataTypes;
  }

  getProcessingStats(): { totalParameters: number; vsDataTypes: number; processingTime: number } {
    return {
      totalParameters: this.processedCount,
      vsDataTypes: this.vsDataTypes.size,
      processingTime: Date.now() - this.startTime
    };
  }
}