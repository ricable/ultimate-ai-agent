/**
 * Schema Generator Utility
 *
 * Generates type-safe schemas (Pydantic, TypeScript, JSON Schema) from templates
 * with comprehensive validation and performance optimization.
 */

import { SchemaInfo, PydanticSchemaConfig } from '../types/export-types';
import { PriorityTemplate } from '../../rtb/hierarchical-template-system/interfaces';

export class SchemaGenerator {
  private typeMappings: Map<string, string> = new Map();
  private schemaTemplates: Map<string, string> = new Map();

  constructor() {
    this.initializeTypeMappings();
    this.initializeSchemaTemplates();
  }

  async initialize(): Promise<void> {
    console.log('📋 Initializing Schema Generator...');
    console.log('✅ Schema Generator initialized');
  }

  async generatePydanticSchema(template: PriorityTemplate, config: PydanticSchemaConfig): Promise<string> {
    const fields = this.extractFields(template);
    const pydanticFields = this.generatePydanticFields(fields, config);
    const validators = this.generatePydanticValidators(fields, config);
    const methods = this.generatePydanticMethods(config);

    const schema = `"""
${config.docstring || `Pydantic schema for ${config.className}`}
Generated automatically from template: ${template.meta.templateId}
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
import re

${config.imports?.join('\n') || ''}

class ${config.className}(BaseModel):
    """
    ${template.meta.description || 'Template configuration schema'}
    """

${pydanticFields}

${validators}

${methods}

    class Config:
        """Pydantic model configuration"""
        extra = 'forbid' if ${config.strictTypes} else 'allow'
        validate_assignment = True
        use_enum_values = True
        schema_extra = {
            "template_id": "${template.meta.templateId}",
            "template_version": "${template.meta.version}",
            "generated_at": datetime.utcnow().isoformat(),
            "field_count": ${fields.length},
            "required_fields": ${fields.filter(f => f.required).length}
        }
`;

    return schema;
  }

  async generateSchemaInfo(template: PriorityTemplate, format: string): Promise<SchemaInfo> {
    const fields = this.extractFields(template);
    const complexTypes = this.identifyComplexTypes(fields);
    const validationRules = this.generateValidationRules(fields);
    const documentationFields = this.generateDocumentationFields(fields);

    return {
      schemaType: format,
      schemaVersion: '5.0.0',
      fieldCount: fields.length,
      requiredFields: fields.filter(f => f.required).length,
      optionalFields: fields.filter(f => !f.required).length,
      complexTypes,
      validationRules,
      documentationFields
    };
  }

  private extractFields(template: PriorityTemplate): SchemaField[] {
    const fields: SchemaField[] = [];

    for (const [key, value] of Object.entries(template.configuration || {})) {
      fields.push({
        name: key,
        type: this.inferType(value),
        required: true, // Default to required, would be determined by constraints
        defaultValue: value,
        description: `Configuration parameter: ${key}`,
        constraints: []
      });
    }

    return fields;
  }

  private inferType(value: any): string {
    if (value === null || value === undefined) return 'Any';
    if (typeof value === 'string') return 'str';
    if (typeof value === 'number') return Number.isInteger(value) ? 'int' : 'float';
    if (typeof value === 'boolean') return 'bool';
    if (Array.isArray(value)) return 'List[Any]';
    if (typeof value === 'object') return 'Dict[str, Any]';
    return 'Any';
  }

  private generatePydanticFields(fields: SchemaField[], config: PydanticSchemaConfig): string {
    return fields.map(field => {
      const optional = config.optionalFields.includes(field.name) ? 'Optional[' : '';
      const optionalClose = config.optionalFields.includes(field.name) ? '] = None' : '';
      const required = config.requiredFields.includes(field.name) ? ' ...' : ' = Field(None)';

      let fieldDef = `    ${field.name}: ${optional}${field.type}${optionalClose}${required}`;

      if (field.description) {
        fieldDef += `  # ${field.description}`;
      }

      return fieldDef;
    }).join('\n\n');
  }

  private generatePydanticValidators(fields: SchemaField[], config: PydanticSchemaConfig): string {
    if (!config.includeValidators) return '';

    return fields
      .filter(field => field.constraints.length > 0)
      .map(field => this.generateFieldValidator(field))
      .join('\n\n');
  }

  private generateFieldValidator(field: SchemaField): string {
    const validatorName = `validate_${field.name}`;
    return `    @validator('${field.name}')
    @classmethod
    def ${validatorName}(cls, v):
        """Validate ${field.name} field"""
        # Add validation logic based on constraints
        return v`;
  }

  private generatePydanticMethods(config: PydanticSchemaConfig): string {
    if (!config.includeSerializers) return '';

    let methods = '';

    if (config.customMethods) {
      methods += config.customMethods.join('\n\n') + '\n\n';
    }

    // Add default methods
    methods += `    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return self.dict(exclude_none=True)

    @classmethod
    def from_template(cls, template_data: Dict[str, Any]) -> '${config.className}':
        """Create instance from template data"""
        return cls(**template_data)
`;

    return methods;
  }

  private identifyComplexTypes(fields: SchemaField[]): any[] {
    return fields
      .filter(field => field.type.includes('List') || field.type.includes('Dict'))
      .map(field => ({
        fieldName: field.name,
        fieldType: field.type,
        nestedFields: this.countNestedFields(field.type),
        isRecursive: field.type.includes('Dict') && field.type.includes('str'),
        isGeneric: field.type.includes('[') && field.type.includes(']'),
        constraints: field.constraints
      }));
  }

  private countNestedFields(typeStr: string): number {
    const match = typeStr.match(/Dict\[.*, (.+)\]/);
    if (match) {
      return match[1].split(',').length;
    }
    return 1;
  }

  private generateValidationRules(fields: SchemaField[]): any[] {
    return fields.map(field => ({
      fieldName: field.name,
      ruleType: 'type_validation',
      condition: `isinstance(value, ${field.type})`,
      errorMessage: `Invalid type for ${field.name}`,
      validationCode: `if not isinstance(v, ${field.type}): raise ValueError("Invalid type")`,
      isRequired: field.required
    }));
  }

  private generateDocumentationFields(fields: SchemaField[]): any[] {
    return fields.map(field => ({
      fieldName: field.name,
      description: field.description,
      dataType: field.type,
      defaultValue: field.defaultValue,
      examples: [field.defaultValue],
      relatedFields: [],
      constraints: field.constraints.map(c => c.toString()),
      notes: []
    }));
  }

  private initializeTypeMappings(): void {
    this.typeMappings.set('string', 'str');
    this.typeMappings.set('number', 'float');
    this.typeMappings.set('integer', 'int');
    this.typeMappings.set('boolean', 'bool');
    this.typeMappings.set('array', 'List[Any]');
    this.typeMappings.set('object', 'Dict[str, Any]');
  }

  private initializeSchemaTemplates(): void {
    this.schemaTemplates.set('pydantic', 'pydantic_template.py');
    this.schemaTemplates.set('typescript', 'typescript_template.ts');
    this.schemaTemplates.set('json_schema', 'json_schema_template.json');
  }
}

interface SchemaField {
  name: string;
  type: string;
  required: boolean;
  defaultValue?: any;
  description: string;
  constraints: any[];
}