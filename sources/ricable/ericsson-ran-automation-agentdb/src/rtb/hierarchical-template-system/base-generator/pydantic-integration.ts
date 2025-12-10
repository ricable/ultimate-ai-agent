/**
 * Pydantic Integration Module
 *
 * This module provides integration between the base template generator
 * and Pydantic models for automatic schema generation and validation.
 */

import { RTBParameter, RTBTemplate, ConstraintSpec } from '../../types/rtb-types';
import { GeneratedTemplate } from './template-generator';

export interface PydanticModelDefinition {
  className: string;
  fields: PydanticField[];
  imports: string[];
  dependencies: string[];
  validators: PydanticValidator[];
  classConfig: PydanticClassConfig;
}

export interface PydanticField {
  name: string;
  type: string;
  required: boolean;
  default?: any;
  description?: string;
  constraints?: PydanticFieldConstraints;
  examples?: any[];
}

export interface PydanticFieldConstraints {
  ge?: number;  // Greater than or equal
  gt?: number;  // Greater than
  le?: number;  // Less than or equal
  lt?: number;  // Less than
  min_length?: number;
  max_length?: number;
  regex?: string;
  enum?: string[];
}

export interface PydanticValidator {
  name: string;
  field: string;
  function: string;
  description?: string;
}

export interface PydanticClassConfig {
  title?: string;
  description?: string;
  extra?: 'forbid' | 'allow' | 'ignore';
  validate_assignment?: boolean;
  use_enum_values?: boolean;
  str_strip_whitespace?: boolean;
}

export interface PydanticSchemaConfig {
  outputFormat: 'single_file' | 'module' | 'package';
  includeValidators: boolean;
  includeExamples: boolean;
  includeDocstrings: boolean;
  useOptional: boolean;
  generateInit: boolean;
  strictTypes: boolean;
}

export class PydanticIntegration {
  private config: PydanticSchemaConfig;

  constructor(config: PydanticSchemaConfig = {
    outputFormat: 'module',
    includeValidators: true,
    includeExamples: true,
    includeDocstrings: true,
    useOptional: true,
    generateInit: true,
    strictTypes: true
  }) {
    this.config = config;
  }

  /**
   * Generate Pydantic models from RTB template
   */
  generatePydanticModels(template: GeneratedTemplate): string {
    console.log(`🐍 Generating Pydantic models for template: ${template.templateId}`);

    const modelDefinitions = this.extractModelDefinitions(template);
    const generatedCode = this.generatePydanticCode(modelDefinitions);

    console.log(`✅ Generated ${modelDefinitions.length} Pydantic models`);

    return generatedCode;
  }

  /**
   * Generate Pydantic models from multiple templates
   */
  generatePydanticModelsFromTemplates(templates: GeneratedTemplate[]): string {
    console.log(`🐍 Generating Pydantic models from ${templates.length} templates`);

    const allModelDefinitions: PydanticModelDefinition[] = [];
    const allImports = new Set<string>();

    for (const template of templates) {
      const modelDefinitions = this.extractModelDefinitions(template);
      allModelDefinitions.push(...modelDefinitions);

      // Collect imports
      modelDefinitions.forEach(def => {
        def.imports.forEach(imp => allImports.add(imp));
      });
    }

    const generatedCode = this.generatePydanticModule(allModelDefinitions, Array.from(allImports));

    console.log(`✅ Generated ${allModelDefinitions.length} Pydantic models from ${templates.length} templates`);

    return generatedCode;
  }

  /**
   * Extract model definitions from RTB template
   */
  private extractModelDefinitions(template: GeneratedTemplate): PydanticModelDefinition[] {
    const definitions: PydanticModelDefinition[] = [];

    // Extract main configuration model
    const mainModel = this.extractMainConfigModel(template);
    definitions.push(mainModel);

    // Extract parameter group models
    const groupModels = this.extractParameterGroupModels(template);
    definitions.push(...groupModels);

    // Extract custom function models
    if (template.template.custom) {
      const functionModels = this.extractFunctionModels(template);
      definitions.push(...functionModels);
    }

    return definitions;
  }

  /**
   * Extract main configuration model
   */
  private extractMainConfigModel(template: GeneratedTemplate): PydanticModelDefinition {
    const className = this.formatClassName(template.templateId);
    const fields: PydanticField[] = [];
    const validators: PydanticValidator[] = [];

    // Extract configuration fields
    for (const [key, value] of Object.entries(template.template.configuration || {})) {
      const field = this.extractFieldFromConfiguration(key, value, template.parameters);
      fields.push(field);
    }

    // Extract parameters as fields
    for (const parameter of template.parameters) {
      const field = this.extractFieldFromParameter(parameter);
      fields.push(field);

      // Generate validators for constrained parameters
      if (parameter.constraints && Array.isArray(parameter.constraints)) {
        const parameterValidators = this.generateParameterValidators(parameter);
        validators.push(...parameterValidators);
      }
    }

    // Extract condition fields
    if (template.template.conditions) {
      for (const [conditionName, condition] of Object.entries(template.template.conditions)) {
        const conditionField = this.extractFieldFromCondition(conditionName, condition);
        fields.push(conditionField);
      }
    }

    return {
      className,
      fields,
      imports: this.generateImports(fields),
      dependencies: [],
      validators,
      classConfig: {
        title: template.template.meta?.description || `${className} Configuration`,
        description: `Pydantic model for ${template.template.meta?.description || className}`,
        extra: 'forbid',
        validate_assignment: true,
        use_enum_values: true
      }
    };
  }

  /**
   * Extract parameter group models
   */
  private extractParameterGroupModels(template: GeneratedTemplate): PydanticModelDefinition[] {
    const groupModels: PydanticModelDefinition[] = [];
    const parameterGroups = this.groupParametersByStructure(template.parameters);

    for (const [groupName, parameters] of parameterGroups) {
      if (parameters.length < 2) continue; // Skip single-parameter groups

      const className = this.formatClassName(`${groupName}Config`);
      const fields: PydanticField[] = [];
      const validators: PydanticValidator[] = [];

      for (const parameter of parameters) {
        const field = this.extractFieldFromParameter(parameter);
        fields.push(field);

        if (parameter.constraints && Array.isArray(parameter.constraints)) {
          const parameterValidators = this.generateParameterValidators(parameter);
          validators.push(...parameterValidators);
        }
      }

      groupModels.push({
        className,
        fields,
        imports: this.generateImports(fields),
        dependencies: [],
        validators,
        classConfig: {
          title: `${groupName} Configuration`,
          description: `Configuration for ${groupName} parameters`,
          extra: 'forbid',
          validate_assignment: true
        }
      });
    }

    return groupModels;
  }

  /**
   * Extract function models
   */
  private extractFunctionModels(template: GeneratedTemplate): PydanticModelDefinition[] {
    const functionModels: PydanticModelDefinition[] = [];

    if (!template.template.custom) return functionModels;

    for (const func of template.template.custom) {
      const className = this.formatClassName(`${func.name}Config`);
      const fields: PydanticField[] = [];

      // Generate fields from function arguments
      for (const arg of func.args) {
        const field: PydanticField = {
          name: arg,
          type: 'Any',
          required: true,
          description: `Argument for ${func.name} function`
        };
        fields.push(field);
      }

      functionModels.push({
        className,
        fields,
        imports: ['from typing import Any'],
        dependencies: [],
        validators: [],
        classConfig: {
          title: `${func.name} Configuration`,
          description: `Configuration for ${func.name} custom function`,
          extra: 'forbid'
        }
      });
    }

    return functionModels;
  }

  /**
   * Extract field from configuration value
   */
  private extractFieldFromConfiguration(key: string, value: any, parameters: RTBParameter[]): PydanticField {
    // Find corresponding parameter if available
    const parameter = parameters.find(p => p.name === key);

    if (parameter) {
      return this.extractFieldFromParameter(parameter);
    }

    // Infer type from value
    const inferredType = this.inferTypeFromValue(value);
    const required = value !== undefined && value !== null && value !== '';

    return {
      name: this.formatFieldName(key),
      type: inferredType,
      required,
      default: required ? undefined : value,
      description: `Configuration parameter: ${key}`
    };
  }

  /**
   * Extract field from parameter
   */
  private extractFieldFromParameter(parameter: RTBParameter): PydanticField {
    const fieldType = this.mapTypeToPython(parameter.type);
    const required = parameter.defaultValue === undefined;
    const constraints = this.extractPydanticConstraints(parameter.constraints);

    return {
      name: this.formatFieldName(parameter.name),
      type: fieldType,
      required,
      default: required ? undefined : parameter.defaultValue,
      description: parameter.description,
      constraints,
      examples: parameter.defaultValue !== undefined ? [parameter.defaultValue] : undefined
    };
  }

  /**
   * Extract field from condition
   */
  private extractFieldFromCondition(conditionName: string, condition: any): PydanticField {
    return {
      name: this.formatFieldName(conditionName),
      type: 'Dict[str, Any]',
      required: false,
      default: condition,
      description: `Condition configuration for ${conditionName}`
    };
  }

  /**
   * Generate parameter validators
   */
  private generateParameterValidators(parameter: RTBParameter): PydanticValidator[] {
    const validators: PydanticValidator[] = [];

    if (!parameter.constraints || !Array.isArray(parameter.constraints)) {
      return validators;
    }

    for (const constraint of parameter.constraints) {
      const validator = this.generateValidatorFromConstraint(parameter, constraint);
      if (validator) {
        validators.push(validator);
      }
    }

    return validators;
  }

  /**
   * Generate validator from constraint
   */
  private generateValidatorFromConstraint(parameter: RTBParameter, constraint: ConstraintSpec): PydanticValidator | null {
    const fieldName = this.formatFieldName(parameter.name);

    switch (constraint.type) {
      case 'range':
        return this.generateRangeValidator(fieldName, constraint);
      case 'enum':
        return this.generateEnumValidator(fieldName, constraint);
      case 'pattern':
        return this.generatePatternValidator(fieldName, constraint);
      case 'length':
        return this.generateLengthValidator(fieldName, constraint);
      default:
        return null;
    }
  }

  /**
   * Generate range validator
   */
  private generateRangeValidator(fieldName: string, constraint: ConstraintSpec): PydanticValidator {
    const value = constraint.value as { min?: number; max?: number };
    let conditions: string[] = [];

    if (value.min !== undefined) {
      conditions.push(`v >= ${value.min}`);
    }
    if (value.max !== undefined) {
      conditions.push(`v <= ${value.max}`);
    }

    const condition = conditions.join(' and ');

    return {
      name: `validate_${fieldName}_range`,
      field: fieldName,
      function: `
@validator('${fieldName}')
@classmethod
def validate_${fieldName}_range(cls, v):
    if not (${condition}):
        raise ValueError(f'Value must be in range ${JSON.stringify(value)}')
    return v`,
      description: `Validate ${fieldName} is within range ${JSON.stringify(value)}`
    };
  }

  /**
   * Generate enum validator
   */
  private generateEnumValidator(fieldName: string, constraint: ConstraintSpec): PydanticValidator {
    const values = constraint.value as string[];
    const valuesStr = JSON.stringify(values);

    return {
      name: `validate_${fieldName}_enum`,
      field: fieldName,
      function: `
@validator('${fieldName}')
@classmethod
def validate_${fieldName}_enum(cls, v):
    if v not in ${valuesStr}:
        raise ValueError(f'Value must be one of ${valuesStr}')
    return v`,
      description: `Validate ${fieldName} is one of allowed values`
    };
  }

  /**
   * Generate pattern validator
   */
  private generatePatternValidator(fieldName: string, constraint: ConstraintSpec): PydanticValidator {
    const pattern = constraint.value as string;

    return {
      name: `validate_${fieldName}_pattern`,
      field: fieldName,
      function: `
@validator('${fieldName}')
@classmethod
def validate_${fieldName}_pattern(cls, v):
    if not re.match(r'${pattern}', str(v)):
        raise ValueError(f'Value must match pattern: ${pattern}')
    return v`,
      description: `Validate ${fieldName} matches pattern ${pattern}`
    };
  }

  /**
   * Generate length validator
   */
  private generateLengthValidator(fieldName: string, constraint: ConstraintSpec): PydanticValidator {
    const value = constraint.value as { min?: number; max?: number };
    let conditions: string[] = [];

    if (value.min !== undefined) {
      conditions.push(`len(v) >= ${value.min}`);
    }
    if (value.max !== undefined) {
      conditions.push(`len(v) <= ${value.max}`);
    }

    const condition = conditions.join(' and ');

    return {
      name: `validate_${fieldName}_length`,
      field: fieldName,
      function: `
@validator('${fieldName}')
@classmethod
def validate_${fieldName}_length(cls, v):
    if not (${condition}):
        raise ValueError(f'Length must be in range ${JSON.stringify(value)}')
    return v`,
      description: `Validate ${fieldName} length is within range`
    };
  }

  /**
   * Generate Pydantic code from model definitions
   */
  private generatePydanticCode(definitions: PydanticModelDefinition[]): string {
    if (this.config.outputFormat === 'single_file') {
      return this.generatePydanticModule(definitions);
    } else {
      return definitions.map(def => this.generatePydanticClass(def)).join('\n\n');
    }
  }

  /**
   * Generate complete Pydantic module
   */
  private generatePydanticModule(definitions: PydanticModelDefinition[], extraImports: string[] = []): string {
    const imports = new Set<string>([
      'from typing import Optional, Dict, Any, List, Union',
      'from pydantic import BaseModel, validator, Field',
      'import re',
      'import json'
    ]);

    // Add model-specific imports
    definitions.forEach(def => {
      def.imports.forEach(imp => imports.add(imp));
    });

    // Add extra imports
    extraImports.forEach(imp => imports.add(imp));

    let code = '';

    // Generate imports
    code += Array.from(imports).sort().join('\n');
    code += '\n\n';

    // Generate models
    code += this.generateModelClasses(definitions);

    // Generate example usage if enabled
    if (this.config.includeExamples) {
      code += '\n\n';
      code += this.generateExampleUsage(definitions);
    }

    return code;
  }

  /**
   * Generate model classes
   */
  private generateModelClasses(definitions: PydanticModelDefinition[]): string {
    return definitions.map(def => this.generatePydanticClass(def)).join('\n\n');
  }

  /**
   * Generate single Pydantic class
   */
  private generatePydanticClass(definition: PydanticModelDefinition): string {
    let code = '';

    // Class docstring
    if (this.config.includeDocstrings && definition.classConfig.description) {
      code += `"""\n${definition.classConfig.description}\n"""\n\n`;
    }

    // Class definition
    code += `class ${definition.className}(BaseModel):\n`;

    // Fields
    for (const field of definition.fields) {
      code += this.generateFieldDefinition(field);
    }

    // Class config
    if (definition.classConfig) {
      code += '\n';
      code += this.generateClassConfig(definition.classConfig);
    }

    // Validators
    if (this.config.includeValidators && definition.validators.length > 0) {
      code += '\n';
      for (const validator of definition.validators) {
        code += this.generateValidator(validator);
      }
    }

    return code;
  }

  /**
   * Generate field definition
   */
  private generateFieldDefinition(field: PydanticField): string {
    let fieldDef = '    ';

    // Field name
    const fieldName = field.required ? field.name : `${field.name}: Optional[${field.type}]`;
    fieldDef += `${fieldName}`;

    // Default value and Field definition
    if (field.default !== undefined) {
      const defaultStr = typeof field.default === 'string' ? `"${field.default}"` : String(field.default);
      fieldDef += ` = Field(default=${defaultStr}`;
    } else if (!field.required) {
      fieldDef += ` = Field(default=None`;
    } else {
      fieldDef += ` = Field(`;
    }

    // Field arguments
    const fieldArgs: string[] = [];

    if (field.description) {
      fieldArgs.push(`description="${field.description}"`);
    }

    if (field.constraints) {
      Object.entries(field.constraints).forEach(([key, value]) => {
        fieldArgs.push(`${key}=${value}`);
      });
    }

    if (field.examples && field.examples.length > 0) {
      fieldArgs.push(`example=${JSON.stringify(field.examples[0])}`);
    }

    if (fieldArgs.length > 0) {
      fieldDef += ', ' + fieldArgs.join(', ');
    }

    fieldDef += ')\n';

    return fieldDef;
  }

  /**
   * Generate class configuration
   */
  private generateClassConfig(config: PydanticClassConfig): string {
    let configDef = '    class Config:\n';

    const configItems: string[] = [];

    if (config.title) {
      configItems.push(`title = "${config.title}"`);
    }

    if (config.description) {
      configItems.push(`schema_extra = {"description": "${config.description}"}`);
    }

    if (config.extra) {
      configItems.push(`extra = "${config.extra}"`);
    }

    if (config.validate_assignment) {
      configItems.push('validate_assignment = True');
    }

    if (config.use_enum_values) {
      configItems.push('use_enum_values = True');
    }

    if (config.str_strip_whitespace) {
      configItems.push('str_strip_whitespace = True');
    }

    configItems.forEach(item => {
      configDef += `        ${item}\n`;
    });

    return configDef;
  }

  /**
   * Generate validator
   */
  private generateValidator(validator: PydanticValidator): string {
    return `    ${validator.function.trim()}\n`;
  }

  /**
   * Generate example usage
   */
  private generateExampleUsage(definitions: PydanticModelDefinition[]): string {
    let code = '# Example Usage\n\n';

    for (const definition of definitions.slice(0, 3)) { // Limit to first 3 models
      code += `# Create ${definition.className} instance\n`;
      code += `${definition.className.lower()}_instance = ${definition.className}(\n`;

      // Generate example values for first few required fields
      const requiredFields = definition.fields.filter(f => f.required).slice(0, 3);
      requiredFields.forEach(field => {
        const exampleValue = this.generateExampleValue(field);
        code += `    ${field.name}=${exampleValue},\n`;
      });

      code += ')\n\n';

      // Add usage example
      code += `# Validate the instance\n`;
      code += `try:\n`;
      code += `    ${definition.className.lower()}_instance.dict()\n`;
      code += `    print("${definition.className} validation passed")\n`;
      code += `except ValidationError as e:\n`;
      code += `    print(f"Validation error: {e}")\n\n`;
    }

    return code;
  }

  /**
   * Group parameters by structure
   */
  private groupParametersByStructure(parameters: RTBParameter[]): Map<string, RTBParameter[]> {
    const groups = new Map<string, RTBParameter[]>();

    for (const parameter of parameters) {
      const groupName = parameter.structureGroups?.[0] || 'default';
      if (!groups.has(groupName)) {
        groups.set(groupName, []);
      }
      groups.get(groupName)!.push(parameter);
    }

    return groups;
  }

  /**
   * Map TypeScript type to Python type
   */
  private mapTypeToPython(tsType: string): string {
    const typeMap: Record<string, string> = {
      'string': 'str',
      'number': 'float',
      'integer': 'int',
      'boolean': 'bool',
      'Date': 'datetime',
      'string[]': 'List[str]',
      'number[]': 'List[float]',
      'integer[]': 'List[int]',
      'boolean[]': 'List[bool]',
      'object': 'Dict[str, Any]',
      'any': 'Any'
    };

    // Handle array types
    if (tsType.endsWith('[]')) {
      const baseType = tsType.slice(0, -2);
      const mappedBaseType = typeMap[baseType] || 'Any';
      return `List[${mappedBaseType}]`;
    }

    return typeMap[tsType] || 'Any';
  }

  /**
   * Infer type from value
   */
  private inferTypeFromValue(value: any): string {
    if (value === null || value === undefined) {
      return 'Any';
    }

    if (typeof value === 'string') {
      return 'str';
    }

    if (typeof value === 'number') {
      return Number.isInteger(value) ? 'int' : 'float';
    }

    if (typeof value === 'boolean') {
      return 'bool';
    }

    if (Array.isArray(value)) {
      if (value.length === 0) {
        return 'List[Any]';
      }
      const elementType = this.inferTypeFromValue(value[0]);
      return `List[${elementType}]`;
    }

    if (typeof value === 'object') {
      return 'Dict[str, Any]';
    }

    return 'Any';
  }

  /**
   * Extract Pydantic constraints from RTB constraints
   */
  private extractPydanticConstraints(constraints?: ConstraintSpec[] | Record<string, any>): PydanticFieldConstraints | undefined {
    if (!constraints) return undefined;

    const pydanticConstraints: PydanticFieldConstraints = {};

    if (Array.isArray(constraints)) {
      for (const constraint of constraints) {
        switch (constraint.type) {
          case 'range':
            const rangeValue = constraint.value as { min?: number; max?: number };
            if (rangeValue.min !== undefined) pydanticConstraints.ge = rangeValue.min;
            if (rangeValue.max !== undefined) pydanticConstraints.le = rangeValue.max;
            break;
          case 'enum':
            pydanticConstraints.enum = constraint.value as string[];
            break;
          case 'pattern':
            pydanticConstraints.regex = constraint.value as string;
            break;
          case 'length':
            const lengthValue = constraint.value as { min?: number; max?: number };
            if (lengthValue.min !== undefined) pydanticConstraints.min_length = lengthValue.min;
            if (lengthValue.max !== undefined) pydanticConstraints.max_length = lengthValue.max;
            break;
        }
      }
    } else if (typeof constraints === 'object') {
      Object.assign(pydanticConstraints, constraints);
    }

    return Object.keys(pydanticConstraints).length > 0 ? pydanticConstraints : undefined;
  }

  /**
   * Generate imports for fields
   */
  private generateImports(fields: PydanticField[]): string[] {
    const imports = new Set<string>(['from typing import Optional, Any']);

    for (const field of fields) {
      if (field.type.includes('List')) {
        imports.add('from typing import List');
      }
      if (field.type.includes('Dict')) {
        imports.add('from typing import Dict');
      }
      if (field.type.includes('Union')) {
        imports.add('from typing import Union');
      }
      if (field.type.includes('datetime')) {
        imports.add('from datetime import datetime');
      }
    }

    return Array.from(imports);
  }

  /**
   * Format class name
   */
  private formatClassName(templateId: string): string {
    // Convert template ID to PascalCase
    return templateId
      .split(/[-_]/)
      .map(part => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
      .join('');
  }

  /**
   * Format field name
   */
  private formatFieldName(name: string): string {
    // Convert to snake_case
    return name
      .replace(/([A-Z])/g, '_$1')
      .toLowerCase()
      .replace(/^_/, '');
  }

  /**
   * Generate example value for field
   */
  private generateExampleValue(field: PydanticField): string {
    if (field.default !== undefined) {
      return typeof field.default === 'string' ? `"${field.default}"` : String(field.default);
    }

    switch (field.type) {
      case 'str':
        return '"example_value"';
      case 'int':
        return '1';
      case 'float':
        return '1.0';
      case 'bool':
        return 'True';
      case 'List[str]':
        return '["example1", "example2"]';
      case 'List[int]':
        return '[1, 2, 3]';
      case 'Dict[str, Any]':
        return '{"key": "value"}';
      default:
        return 'None';
    }
  }

  /**
   * Export Pydantic models to files
   */
  async exportToFiles(templates: GeneratedTemplate[], outputDir: string): Promise<string[]> {
    const fs = require('fs').promises;
    const path = require('path');

    // Ensure output directory exists
    await fs.mkdir(outputDir, { recursive: true });

    const generatedFiles: string[] = [];

    if (this.config.outputFormat === 'single_file') {
      // Generate single combined file
      const combinedCode = this.generatePydanticModelsFromTemplates(templates);
      const filePath = path.join(outputDir, 'rtb_models.py');
      await fs.writeFile(filePath, combinedCode);
      generatedFiles.push(filePath);
    } else {
      // Generate separate files for each template
      for (const template of templates) {
        const code = this.generatePydanticModels(template);
        const fileName = `${template.templateId.toLowerCase()}_models.py`;
        const filePath = path.join(outputDir, fileName);
        await fs.writeFile(filePath, code);
        generatedFiles.push(filePath);
      }
    }

    // Generate __init__.py file
    const initContent = this.generateInitFile(generatedFiles);
    const initPath = path.join(outputDir, '__init__.py');
    await fs.writeFile(initPath, initContent);
    generatedFiles.push(initPath);

    console.log(`🐍 Exported ${generatedFiles.length} Pydantic files to ${outputDir}`);

    return generatedFiles;
  }

  /**
   * Generate __init__.py file
   */
  private generateInitFile(files: string[]): string {
    const imports = files
      .filter(file => !file.endsWith('__init__.py'))
      .map(file => {
        const moduleName = path.basename(file, '.py');
        return `from .${moduleName} import *`;
      });

    return `"""RTB Pydantic Models

Generated by Base Template Auto-Generator
"""

${imports.join('\n')}

__all__ = [
    # Export all model classes
    "RTBConfiguration",
    "ENodeBFunctionConfig",
    # Add other model classes as needed
]
`;
  }
}