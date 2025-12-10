import { promises as fs } from 'fs';
import { ParameterSpec, ConstraintSpec } from '../types/rtb-types';

export class SpreadsheetParametersParser {
  private parameters: Map<string, ParameterSpec> = new Map();
  private vsDataTypeIndex: Map<string, ParameterSpec[]> = new Map();
  private parameterHierarchy: Map<string, string[]> = new Map();

  async parseSpreadsheetParameters(filePath: string): Promise<Map<string, ParameterSpec>> {
    console.log(`[SpreadsheetParametersParser] Parsing spreadsheet parameters from ${filePath}`);

    try {
      const content = await fs.readFile(filePath, 'utf-8');
      const lines = content.split('\n').filter(line => line.trim());

      // Skip header line
      const dataLines = lines.slice(1);

      for (const line of dataLines) {
        this.parseParameterLine(line);
      }

      console.log(`[SpreadsheetParametersParser] Parsed ${this.parameters.size} parameters`);

      // Build hierarchical index
      this.buildParameterHierarchy();

      console.log(`[SpreadsheetParametersParser] Built hierarchy for ${this.parameterHierarchy.size} parameter groups`);
      console.log(`[SpreadsheetParametersParser] Index ${this.vsDataTypeIndex.size} vsData types`);

      return this.parameters;
    } catch (error) {
      console.error('[SpreadsheetParametersParser] Error parsing Spreadsheets_Parameters.csv:', error);
      throw error;
    }
  }

  private parseParameterLine(line: string): void {
    if (!line.trim()) return;

    // CSV parsing with quoted fields
    const fields = this.parseCSVLine(line);
    if (fields.length < 8) return; // Minimum required fields

    const parameter: ParameterSpec = {
      name: fields[0]?.trim() || '',
      vsDataType: fields[1]?.trim() || '',
      type: this.mapSpreadsheetType(fields[2]?.trim() || 'string'),
      constraints: this.parseConstraints(fields[3], fields[4], fields[5]),
      description: fields[6]?.trim() || '',
      defaultValue: this.parseDefaultValue(fields[7]),
      deprecated: this.parseBoolean(fields[8]),
      introduced: fields[9]?.trim(),
      deprecatedSince: fields[10]?.trim()
    };

    if (parameter.name && parameter.vsDataType) {
      this.parameters.set(parameter.name, parameter);

      // Index by vsDataType
      if (!this.vsDataTypeIndex.has(parameter.vsDataType)) {
        this.vsDataTypeIndex.set(parameter.vsDataType, []);
      }
      this.vsDataTypeIndex.get(parameter.vsDataType)!.push(parameter);
    }
  }

  private parseCSVLine(line: string): string[] {
    const fields: string[] = [];
    let currentField = '';
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
      const char = line[i];

      if (char === '"') {
        if (inQuotes && i > 0 && line[i - 1] !== '\\') {
          inQuotes = false;
        } else {
          inQuotes = true;
        }
      } else if (char === ',' && !inQuotes) {
        fields.push(currentField.trim());
        currentField = '';
      } else {
        currentField += char;
      }
    }

    fields.push(currentField.trim());
    return fields;
  }

  private mapSpreadsheetType(spreadsheetType: string): string {
    const typeMap: Record<string, string> = {
      'int': 'number',
      'integer': 'number',
      'string': 'string',
      'bool': 'boolean',
      'boolean': 'boolean',
      'float': 'number',
      'double': 'number',
      'datetime': 'string',
      'date': 'string',
      'time': 'string',
      'array': 'any[]',
      'list': 'any[]',
      'object': 'object',
      'json': 'object'
    };

    return typeMap[spreadsheetType] || 'any';
  }

  private parseConstraints(constraintString: string, minValue: string, maxValue: string): ConstraintSpec[] {
    const constraints: ConstraintSpec[] = [];

    if (constraintString) {
      const constraintParts = constraintString.split(';').map(part => part.trim());

      for (const part of constraintParts) {
        if (part.includes('range')) {
          constraints.push({
            type: 'range',
            value: { min: minValue, max: maxValue },
            errorMessage: `Value must be between ${minValue} and ${maxValue}`,
            severity: 'error'
          });
        } else if (part.includes('enum')) {
          const enumValues = constraintString.match(/\[([^]]+)\]/);
          if (enumValues) {
            constraints.push({
              type: 'enum',
              value: enumValues[1].split(',').map(val => val.trim()),
              errorMessage: `Value must be one of: ${enumValues[1]}`,
              severity: 'error'
            });
          }
        } else if (part.includes('pattern')) {
          const patternMatch = constraintString.match(/\/(.+)\//);
          if (patternMatch) {
            constraints.push({
              type: 'pattern',
              value: patternMatch[1],
              errorMessage: 'Value must match the specified pattern',
              severity: 'error'
            });
          }
        } else if (part.includes('length')) {
          const lengthMatch = constraintString.match(/length\[(\d+),(\d+)\]/);
          if (lengthMatch) {
            constraints.push({
              type: 'length',
              value: { min: lengthMatch[1], max: lengthMatch[2] },
              errorMessage: `Length must be between ${lengthMatch[1]} and ${lengthMatch[2]} characters`,
              severity: 'error'
            });
          }
        }
      }
    }

    // Add required constraint if applicable
    if (constraintString && constraintString.toLowerCase().includes('required')) {
      constraints.push({
        type: 'required',
        value: true,
        errorMessage: 'This parameter is required',
        severity: 'error'
      });
    }

    return constraints;
  }

  private parseDefaultValue(defaultValue: string): any {
    if (!defaultValue || defaultValue === 'null') {
      return undefined;
    }

    // Try to parse as JSON first
    try {
      return JSON.parse(defaultValue);
    } catch {
      // If not JSON, return as string
      return defaultValue;
    }
  }

  private parseBoolean(value: string): boolean {
    return ['true', '1', 'yes', 'enabled'].includes(value.toLowerCase());
  }

  private buildParameterHierarchy(): void {
    // Build hierarchical relationships based on vsDataType names
    for (const [parameterName, parameter] of this.parameters) {
      const hierarchy = this.extractHierarchyFromParameter(parameter);
      this.parameterHierarchy.set(parameterName, hierarchy);
    }
  }

  private extractHierarchyFromParameter(parameter: ParameterSpec): string[] {
    const hierarchy: string[] = [];
    const parts = parameter.name.split('_');

    // Build hierarchy from underscore-separated names
    for (let i = 1; i <= parts.length; i++) {
      const levelName = parts.slice(0, i).join('_');
      hierarchy.push(levelName);
    }

    return hierarchy;
  }

  getParameters(): Map<string, ParameterSpec> {
    return this.parameters;
  }

  getParametersByVsDataType(vsDataType: string): ParameterSpec[] {
    return this.vsDataTypeIndex.get(vsDataType) || [];
  }

  findParametersByPattern(pattern: string): ParameterSpec[] {
    const regex = new RegExp(pattern, 'i');
    return Array.from(this.parameters.values())
      .filter(param => regex.test(param.name) || regex.test(param.description));
  }

  findParametersByType(type: string): ParameterSpec[] {
    return Array.from(this.parameters.values())
      .filter(param => param.type === type);
  }

  findDeprecatedParameters(): ParameterSpec[] {
    return Array.from(this.parameters.values())
      .filter(param => param.deprecated);
  }

  findRequiredParameters(): ParameterSpec[] {
    return Array.from(this.parameters.values())
      .filter(param =>
        param.constraints?.some(constraint => constraint.type === 'required' && constraint.value)
      );
  }

  getParameterHierarchy(): Map<string, string[]> {
    return this.parameterHierarchy;
  }

  getParentParameters(parameterName: string): ParameterSpec[] {
    const hierarchy = this.parameterHierarchy.get(parameterName) || [];
    const parentNames = hierarchy.slice(0, -1); // Remove the parameter itself

    return parentNames
      .map(name => this.parameters.get(name))
      .filter(Boolean) as ParameterSpec[];
  }

  getChildParameters(parameterName: string): ParameterSpec[] {
    const hierarchy = this.parameterHierarchy.get(parameterName) || [];
    if (hierarchy.length === 0) return [];

    const parentLevel = hierarchy.length - 1;
    const children: ParameterSpec[] = [];

    for (const [name, childHierarchy] of this.parameterHierarchy) {
      if (name !== parameterName && childHierarchy.length > parentLevel) {
        const childParentHierarchy = childHierarchy.slice(0, parentLevel + 1);
        if (childParentHierarchy[parentLevel] === parameterName) {
          const childParam = this.parameters.get(name);
          if (childParam) children.push(childParam);
        }
      }
    }

    return children;
  }

  getParameterStats(): {
    totalParameters: number;
    vsDataTypes: string[];
    typeDistribution: Record<string, number>;
    deprecatedCount: number;
    requiredCount: number;
    constraintCount: number;
    averageConstraints: number;
  } {
    const typeDistribution: Record<string, number> = {};
    let deprecatedCount = 0;
    let requiredCount = 0;
    let totalConstraints = 0;

    const vsDataTypes = Array.from(this.vsDataTypeIndex.keys());

    for (const parameter of this.parameters.values()) {
      // Type distribution
      const type = parameter.type;
      typeDistribution[type] = (typeDistribution[type] || 0) + 1;

      // Deprecated count
      if (parameter.deprecated) deprecatedCount++;

      // Required count
      if (parameter.constraints?.some(c => c.type === 'required' && c.value)) {
        requiredCount++;
      }

      // Total constraints
      if (parameter.constraints) {
        totalConstraints += parameter.constraints.length;
      }
    }

    const averageConstraints = this.parameters.size > 0
      ? totalConstraints / this.parameters.size
      : 0;

    return {
      totalParameters: this.parameters.size,
      vsDataTypes,
      typeDistribution,
      deprecatedCount,
      requiredCount,
      constraintCount: totalConstraints,
      averageConstraints
    };
  }

  exportAsJSON(): any {
    const stats = this.getParameterStats();
    return {
      ...stats,
      parameters: Array.from(this.parameters.values()),
      vsDataTypeIndex: Array.from(this.vsDataTypeIndex.entries()),
      parameterHierarchy: Array.from(this.parameterHierarchy.entries())
    };
  }

  // Validation methods
  validateParameter(parameterName: string, value: any): { valid: boolean; errors: string[] } {
    const errors: string[] = [];
    const parameter = this.parameters.get(parameterName);

    if (!parameter) {
      return { valid: false, errors: [`Parameter '${parameterName}' not found`] };
    }

    // Type validation
    if (!this.validateType(value, parameter.type)) {
      errors.push(`Value '${value}' is not of type '${parameter.type}'`);
    }

    // Constraint validation
    if (parameter.constraints) {
      for (const constraint of parameter.constraints) {
        if (!this.validateConstraint(value, constraint)) {
          errors.push(constraint.errorMessage || `Constraint validation failed`);
        }
      }
    }

    return { valid: errors.length === 0, errors };
  }

  private validateType(value: any, expectedType: string): boolean {
    switch (expectedType) {
      case 'number':
        return typeof value === 'number' || !isNaN(Number(value));
      case 'string':
        return typeof value === 'string';
      case 'boolean':
        return typeof value === 'boolean';
      case 'any[]':
      case 'object':
        return typeof value === 'object';
      default:
        return true; // Allow unknown types
    }
  }

  private validateConstraint(value: any, constraint: ConstraintSpec): boolean {
    switch (constraint.type) {
      case 'range':
        if (typeof value === 'number') {
          const range = constraint.value as { min: string; max: string };
          const min = parseFloat(range.min);
          const max = parseFloat(range.max);
          return value >= min && value <= max;
        }
        return false;

      case 'enum':
        const enumValues = constraint.value as string[];
        return enumValues.includes(String(value));

      case 'pattern':
        const pattern = new RegExp(constraint.value as string);
        return pattern.test(String(value));

      case 'length':
        if (typeof value === 'string') {
          const length = constraint.value as { min: string; max: string };
          const minLen = parseInt(length.min);
          const maxLen = parseInt(length.max);
          return value.length >= minLen && value.length <= maxLen;
        }
        return false;

      case 'required':
        return constraint.value ? value !== undefined && value !== null && value !== '' : true;

      default:
        return true;
    }
  }
}