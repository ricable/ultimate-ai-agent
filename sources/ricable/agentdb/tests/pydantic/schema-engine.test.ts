/**
 * Phase 5 Implementation Tests - Schema Engine Unit Tests
 *
 * Comprehensive test suite for the Pydantic schema generation engine
 * with validation of model generation, performance optimization, and caching
 */

import { SchemaEngine, SchemaGenerationConfig, ModelGenerationOptions } from '../../src/pydantic/schema-engine';
import { MappingResult } from '../../src/pydantic/type-mapper';
import { MOClass } from '../../src/types/rtb-types';

describe('SchemaEngine', () => {
  let schemaEngine: SchemaEngine;
  let config: SchemaGenerationConfig;

  beforeEach(() => {
    config = {
      enableOptimizations: true,
      useCaching: true,
      strictMode: true,
      generateValidators: true,
      includeImports: true,
      cognitiveMode: false,
      performanceMode: false
    };
    schemaEngine = new SchemaEngine(config);
  });

  describe('Initialization', () => {
    it('should initialize with default configuration', async () => {
      await schemaEngine.initialize();
      expect(schemaEngine).toBeDefined();
    });

    it('should initialize with custom configuration', async () => {
      config.performanceMode = true;
      config.cognitiveMode = true;

      const customEngine = new SchemaEngine(config);
      await customEngine.initialize();

      expect(customEngine).toBeDefined();
    });
  });

  describe('Basic Model Generation', () => {
    beforeEach(async () => {
      await schemaEngine.initialize();
    });

    it('should generate basic Pydantic model', async () => {
      const moClass = createTestMOClass('TestModel');
      const parameterMappings = createTestParameterMappings([
        { name: 'name', xmlType: 'string', pythonType: 'str', typescriptType: 'string' },
        { name: 'age', xmlType: 'integer', pythonType: 'int', typescriptType: 'number' },
        { name: 'active', xmlType: 'boolean', pythonType: 'bool', typescriptType: 'boolean' }
      ]);

      const result = await schemaEngine.generateModel(moClass, parameterMappings);

      expect(result.className).toBe('TestModelModel');
      expect(result.pythonCode).toContain('class TestModelModel(BaseModel):');
      expect(result.typescriptCode).toContain('export interface TestModelModel {');
      expect(result.imports).toContain('from pydantic import BaseModel, Field, validator');
      expect(result.confidence).toBeGreaterThan(0);
      expect(result.generatedAt).toBeGreaterThan(0);
    });

    it('should generate model with proper field definitions', async () => {
      const moClass = createTestMOClass('PersonModel');
      const parameterMappings = createTestParameterMappings([
        { name: 'firstName', xmlType: 'string', pythonType: 'str', typescriptType: 'string', isOptional: false },
        { name: 'lastName', xmlType: 'string', pythonType: 'str', typescriptType: 'string', isOptional: false },
        { name: 'age', xmlType: 'integer', pythonType: 'int', typescriptType: 'number', isOptional: true },
        { name: 'email', xmlType: 'string', pythonType: 'str', typescriptType: 'string', isOptional: true }
      ]);

      const result = await schemaEngine.generateModel(moClass, parameterMappings);

      // Check required fields
      expect(result.pythonCode).toContain('firstName: str');
      expect(result.pythonCode).toContain('lastName: str');

      // Check optional fields
      expect(result.pythonCode).toContain('age: Optional[int]');
      expect(result.pythonCode).toContain('email: Optional[str]');

      // Check TypeScript interface
      expect(result.typescriptCode).toContain('firstName: string;');
      expect(result.typescriptCode).toContain('lastName: string;');
      expect(result.typescriptCode).toContain('age?: number;');
      expect(result.typescriptCode).toContain('email?: string;');
    });

    it('should generate model with default values', async () => {
      const moClass = createTestMOClass('DefaultModel');
      const parameterMappings = createTestParameterMappings([
        { name: 'status', xmlType: 'string', pythonType: 'str', typescriptType: 'string', defaultValue: 'active' },
        { name: 'count', xmlType: 'integer', pythonType: 'int', typescriptType: 'number', defaultValue: 0 },
        { name: 'enabled', xmlType: 'boolean', pythonType: 'bool', typescriptType: 'boolean', defaultValue: true }
      ]);

      const result = await schemaEngine.generateModel(moClass, parameterMappings);

      expect(result.pythonCode).toContain("status: str = Field(default='active')");
      expect(result.pythonCode).toContain('count: int = Field(default=0)');
      expect(result.pythonCode).toContain('enabled: bool = Field(default=True)');
    });
  });

  describe('Complex Type Generation', () => {
    beforeEach(async () => {
      await schemaEngine.initialize();
    });

    it('should generate model with datetime fields', async () => {
      const moClass = createTestMOClass('TimestampModel');
      const parameterMappings = createTestParameterMappings([
        { name: 'createdAt', xmlType: 'dateTime', pythonType: 'datetime', typescriptType: 'Date' },
        { name: 'updatedAt', xmlType: 'dateTime', pythonType: 'datetime', typescriptType: 'Date' }
      ]);

      const result = await schemaEngine.generateModel(moClass, parameterMappings);

      expect(result.pythonCode).toContain('createdAt: datetime');
      expect(result.pythonCode).toContain('updatedAt: datetime');
      expect(result.imports).toContain('from datetime import datetime, date');
      expect(result.typescriptCode).toContain('createdAt: Date;');
      expect(result.typescriptCode).toContain('updatedAt: Date;');
    });

    it('should generate model with decimal fields', async () => {
      const moClass = createTestMOClass('DecimalModel');
      const parameterMappings = createTestParameterMappings([
        { name: 'price', xmlType: 'decimal', pythonType: 'Decimal', typescriptType: 'number' },
        { name: 'quantity', xmlType: 'decimal', pythonType: 'Decimal', typescriptType: 'number' }
      ]);

      const result = await schemaEngine.generateModel(moClass, parameterMappings);

      expect(result.pythonCode).toContain('price: Decimal');
      expect(result.pythonCode).toContain('quantity: Decimal');
      expect(result.imports).toContain('from decimal import Decimal');
    });

    it('should generate model with list fields', async () => {
      const moClass = createTestMOClass('ListModel');
      const parameterMappings = createTestParameterMappings([
        { name: 'tags', xmlType: 'list', pythonType: 'List[str]', typescriptType: 'string[]' },
        { name: 'scores', xmlType: 'list', pythonType: 'List[int]', typescriptType: 'number[]' }
      ]);

      const result = await schemaEngine.generateModel(moClass, parameterMappings);

      expect(result.pythonCode).toContain('tags: List[str]');
      expect(result.pythonCode).toContain('scores: List[int]');
      expect(result.imports).toContain('from typing import Optional, List, Dict, Any, Union');
      expect(result.typescriptCode).toContain('tags: string[];');
      expect(result.typescriptCode).toContain('scores: number[];');
    });

    it('should generate model with dict fields', async () => {
      const moClass = createTestMOClass('DictModel');
      const parameterMappings = createTestParameterMappings([
        { name: 'metadata', xmlType: 'object', pythonType: 'Dict[str, Any]', typescriptType: 'Record<string, any>' },
        { name: 'properties', xmlType: 'object', pythonType: 'Dict[str, str]', typescriptType: 'Record<string, string>' }
      ]);

      const result = await schemaEngine.generateModel(moClass, parameterMappings);

      expect(result.pythonCode).toContain('metadata: Dict[str, Any]');
      expect(result.pythonCode).toContain('properties: Dict[str, str]');
      expect(result.typescriptCode).toContain('metadata: Record<string, any>;');
      expect(result.typescriptCode).toContain('properties: Record<string, string>;');
    });
  });

  describe('Validator Generation', () => {
    beforeEach(async () => {
      await schemaEngine.initialize();
    });

    it('should generate range validators', async () => {
      const moClass = createTestMOClass('RangeModel');
      const parameterMappings = createTestParameterMappings([
        {
          name: 'age',
          xmlType: 'integer',
          pythonType: 'int',
          typescriptType: 'number',
          constraints: [
            { type: 'range', value: { min: 0, max: 150 }, severity: 'error' }
          ]
        }
      ]);

      const result = await schemaEngine.generateModel(moClass, parameterMappings);

      expect(result.pythonCode).toContain('@validator(\'age\')');
      expect(result.pythonCode).toContain('def validate_age(cls, v):');
      expect(result.pythonCode).toContain('if not (0 <= v <= 150):');
      expect(result.pythonCode).toContain('raise ValueError(\'age must be between 0 and 150\')');
    });

    it('should generate enum validators', async () => {
      const moClass = createTestMOClass('EnumModel');
      const parameterMappings = createTestParameterMappings([
        {
          name: 'status',
          xmlType: 'string',
          pythonType: 'str',
          typescriptType: 'string',
          constraints: [
            { type: 'enum', value: ['ACTIVE', 'INACTIVE', 'PENDING'], severity: 'error' }
          ]
        }
      ]);

      const result = await schemaEngine.generateModel(moClass, parameterMappings);

      expect(result.pythonCode).toContain('@validator(\'status\')');
      expect(result.pythonCode).toContain('def validate_status(cls, v):');
      expect(result.pythonCode).toContain('if v not in [\'ACTIVE\', \'INACTIVE\', \'PENDING\']:');
      expect(result.pythonCode).toContain('raise ValueError(\'status must be one of [ACTIVE, INACTIVE, PENDING]\')');
    });

    it('should generate pattern validators', async () => {
      const moClass = createTestMOClass('PatternModel');
      const parameterMappings = createTestParameterMappings([
        {
          name: 'code',
          xmlType: 'string',
          pythonType: 'str',
          typescriptType: 'string',
          constraints: [
            { type: 'pattern', value: '^[A-Z]{3}-\\d{3}$', severity: 'error' }
          ]
        }
      ]);

      const result = await schemaEngine.generateModel(moClass, parameterMappings);

      expect(result.pythonCode).toContain('import re');
      expect(result.pythonCode).toContain('@validator(\'code\')');
      expect(result.pythonCode).toContain('def validate_code(cls, v):');
      expect(result.pythonCode).toContain('if not re.match(r\'^[A-Z]{3}-\\d{3}$\', str(v)):');
      expect(result.pythonCode).toContain('raise ValueError(\'code does not match required pattern\')');
    });

    it('should generate length validators', async () => {
      const moClass = createTestMOClass('LengthModel');
      const parameterMappings = createTestParameterMappings([
        {
          name: 'username',
          xmlType: 'string',
          pythonType: 'str',
          typescriptType: 'string',
          constraints: [
            { type: 'length', value: { minLength: 3, maxLength: 20 }, severity: 'error' }
          ]
        }
      ]);

      const result = await schemaEngine.generateModel(moClass, parameterMappings);

      expect(result.pythonCode).toContain('@validator(\'username\')');
      expect(result.pythonCode).toContain('def validate_username(cls, v):');
      expect(result.pythonCode).toContain('if not (3 <= len(str(v)) <= 20):');
      expect(result.pythonCode).toContain('raise ValueError(\'username length must be between 3 and 20 characters long\')');
    });

    it('should generate multiple validators for a field', async () => {
      const moClass = createTestMOClass('MultiValidatorModel');
      const parameterMappings = createTestParameterMappings([
        {
          name: 'code',
          xmlType: 'string',
          pythonType: 'str',
          typescriptType: 'string',
          constraints: [
            { type: 'length', value: { minLength: 3, maxLength: 10 }, severity: 'error' },
            { type: 'pattern', value: '^[A-Z]+$', severity: 'error' }
          ]
        }
      ]);

      const result = await schemaEngine.generateModel(moClass, parameterMappings);

      expect(result.pythonCode).toContain('@validator(\'code\')');
      // Should have both length and pattern validation
      expect(result.pythonCode).toMatch(/validate_code.*length.*pattern/);
    });
  });

  describe('Method Generation', () => {
    beforeEach(async () => {
      await schemaEngine.initialize();
    });

    it('should generate to_dict method', async () => {
      const moClass = createTestMOClass('MethodModel');
      const parameterMappings = createTestParameterMappings([
        { name: 'name', xmlType: 'string', pythonType: 'str', typescriptType: 'string' }
      ]);

      const result = await schemaEngine.generateModel(moClass, parameterMappings);

      expect(result.pythonCode).toContain('def to_dict(self) -> Dict[str, Any]:');
      expect(result.pythonCode).toContain('return self.dict()');
      expect(result.typescriptCode).toContain('to_dict(): Dict[str, Any];');
    });

    it('should generate from_dict method', async () => {
      const moClass = createTestMOClass('MethodModel');
      const parameterMappings = createTestParameterMappings([
        { name: 'name', xmlType: 'string', pythonType: 'str', typescriptType: 'string' },
        { name: 'age', xmlType: 'integer', pythonType: 'int', typescriptType: 'number' }
      ]);

      const result = await schemaEngine.generateModel(moClass, parameterMappings);

      expect(result.pythonCode).toContain('@staticmethod');
      expect(result.pythonCode).toContain('def from_dict(data: Dict[str, Any]) -> cls:');
      expect(result.pythonCode).toContain('return cls(**{k: v for k, v in data.items() if k in [\'name\', \'age\']})');
    });

    it('should generate validate method when enabled', async () => {
      const options: ModelGenerationOptions = {
        includeValidators: true
      };

      const moClass = createTestMOClass('MethodModel');
      const parameterMappings = createTestParameterMappings([
        { name: 'name', xmlType: 'string', pythonType: 'str', typescriptType: 'string' }
      ]);

      const result = await schemaEngine.generateModel(moClass, parameterMappings, options);

      expect(result.pythonCode).toContain('def validate(self, strict: bool = True) -> bool:');
      expect(result.pythonCode).toContain('try:');
      expect(result.pythonCode).toContain('self.dict(strict=strict)');
      expect(result.pythonCode).toContain('return True');
      expect(result.pythonCode).toContain('except Exception:');
      expect(result.pythonCode).toContain('return False');
    });

    it('should generate getter methods for high-confidence fields', async () => {
      const moClass = createTestMOClass('GetterModel');
      const parameterMappings = createTestParameterMappings([
        { name: 'importantField', xmlType: 'string', pythonType: 'str', typescriptType: 'string', mappingConfidence: 0.9 },
        { name: 'lowConfidenceField', xmlType: 'string', pythonType: 'str', typescriptType: 'string', mappingConfidence: 0.6 }
      ]);

      const result = await schemaEngine.generateModel(moClass, parameterMappings);

      expect(result.pythonCode).toContain('def get_importantField(self) -> str:');
      expect(result.pythonCode).not.toContain('def get_lowConfidenceField');
      expect(result.typescriptCode).toContain('get_importantField(): str;');
    });
  });

  describe('Caching and Performance', () => {
    beforeEach(async () => {
      await schemaEngine.initialize();
    });

    it('should cache generated models', async () => {
      const moClass = createTestMOClass('CacheModel');
      const parameterMappings = createTestParameterMappings([
        { name: 'name', xmlType: 'string', pythonType: 'str', typescriptType: 'string' }
      ]);

      const startTime1 = Date.now();
      const result1 = await schemaEngine.generateModel(moClass, parameterMappings);
      const endTime1 = Date.now();

      const startTime2 = Date.now();
      const result2 = await schemaEngine.generateModel(moClass, parameterMappings);
      const endTime2 = Date.now();

      expect(result1.className).toBe(result2.className);
      expect(result1.pythonCode).toBe(result2.pythonCode);

      // Second call should be faster due to caching
      const firstCallTime = endTime1 - startTime1;
      const secondCallTime = endTime2 - startTime2;
      expect(secondCallTime).toBeLessThanOrEqual(firstCallTime);
    });

    it('should clear cache', async () => {
      const moClass = createTestMOClass('ClearCacheModel');
      const parameterMappings = createTestParameterMappings([
        { name: 'name', xmlType: 'string', pythonType: 'str', typescriptType: 'string' }
      ]);

      await schemaEngine.generateModel(moClass, parameterMappings);
      schemaEngine.clearCache();

      const metrics = schemaEngine.getMetrics();
      // Cache is cleared but metrics remain
      expect(metrics.totalClasses).toBeGreaterThan(0);
    });

    it('should provide accurate metrics', async () => {
      const moClasses = [
        createTestMOClass('MetricsModel1'),
        createTestMOClass('MetricsModel2'),
        createTestMOClass('MetricsModel3')
      ];

      for (const moClass of moClasses) {
        const parameterMappings = createTestParameterMappings([
          { name: 'field1', xmlType: 'string', pythonType: 'str', typescriptType: 'string' },
          { name: 'field2', xmlType: 'integer', pythonType: 'int', typescriptType: 'number' }
        ]);
        await schemaEngine.generateModel(moClass, parameterMappings);
      }

      const metrics = schemaEngine.getMetrics();

      expect(metrics.totalClasses).toBe(3);
      expect(metrics.totalFields).toBe(6); // 2 fields per model
      expect(metrics.averageConfidence).toBeGreaterThan(0);
      expect(metrics.generationTime).toBeGreaterThan(0);
    });
  });

  describe('Import Generation', () => {
    beforeEach(async () => {
      await schemaEngine.initialize();
    });

    it('should generate basic imports', async () => {
      const moClass = createTestMOClass('ImportModel');
      const parameterMappings = createTestParameterMappings([
        { name: 'name', xmlType: 'string', pythonType: 'str', typescriptType: 'string' }
      ]);

      const result = await schemaEngine.generateModel(moClass, parameterMappings);

      expect(result.imports).toContain('from pydantic import BaseModel, Field, validator');
      expect(result.imports).toContain('from typing import Optional, List, Dict, Any, Union');
    });

    it('should include datetime imports when needed', async () => {
      const moClass = createTestMOClass('DatetimeImportModel');
      const parameterMappings = createTestParameterMappings([
        { name: 'timestamp', xmlType: 'dateTime', pythonType: 'datetime', typescriptType: 'Date' }
      ]);

      const result = await schemaEngine.generateModel(moClass, parameterMappings);

      expect(result.imports).toContain('from datetime import datetime, date');
    });

    it('should include decimal imports when needed', async () => {
      const moClass = createTestMOClass('DecimalImportModel');
      const parameterMappings = createTestParameterMappings([
        { name: 'price', xmlType: 'decimal', pythonType: 'Decimal', typescriptType: 'number' }
      ]);

      const result = await schemaEngine.generateModel(moClass, parameterMappings);

      expect(result.imports).toContain('from decimal import Decimal');
    });

    it('should include regex imports when pattern validators exist', async () => {
      const moClass = createTestMOClass('RegexImportModel');
      const parameterMappings = createTestParameterMappings([
        {
          name: 'code',
          xmlType: 'string',
          pythonType: 'str',
          typescriptType: 'string',
          constraints: [
            { type: 'pattern', value: '^[A-Z]+$', severity: 'error' }
          ]
        }
      ]);

      const result = await schemaEngine.generateModel(moClass, parameterMappings);

      expect(result.imports).toContain('import re');
    });
  });

  describe('Error Handling', () => {
    beforeEach(async () => {
      await schemaEngine.initialize();
    });

    it('should handle empty parameter mappings', async () => {
      const moClass = createTestMOClass('EmptyModel');
      const parameterMappings: MappingResult[] = [];

      const result = await schemaEngine.generateModel(moClass, parameterMappings);

      expect(result.className).toBe('EmptyModelModel');
      expect(result.pythonCode).toContain('class EmptyModelModel(BaseModel):');
      expect(result.pythonCode).not.toContain('def __init__');
    });

    it('should handle malformed field names', async () => {
      const moClass = createTestMOClass('MalformedModel');
      const parameterMappings = createTestParameterMappings([
        { name: '123invalid', xmlType: 'string', pythonType: 'str', typescriptType: 'string' },
        { name: 'field-with-dashes', xmlType: 'integer', pythonType: 'int', typescriptType: 'number' }
      ]);

      const result = await schemaEngine.generateModel(moClass, parameterMappings);

      expect(result.pythonCode).toContain('_123invalid: str');
      expect(result.pythonCode).toContain('field_with_dashes: int');
    });

    it('should handle invalid constraint data gracefully', async () => {
      const moClass = createTestMOClass('InvalidConstraintModel');
      const parameterMappings = createTestParameterMappings([
        {
          name: 'field',
          xmlType: 'string',
          pythonType: 'str',
          typescriptType: 'string',
          constraints: [
            { type: 'range', value: 'invalid_range_data', severity: 'error' }
          ]
        }
      ]);

      // Should not throw error but handle gracefully
      await expect(schemaEngine.generateModel(moClass, parameterMappings)).resolves.toBeDefined();
    });
  });

  describe('Performance Requirements', () => {
    beforeEach(async () => {
      await schemaEngine.initialize();
    });

    it('should generate 100 models within performance target', async () => {
      const startTime = Date.now();

      for (let i = 0; i < 100; i++) {
        const moClass = createTestMOClass(`PerfModel${i}`);
        const parameterMappings = createTestParameterMappings([
          { name: 'field1', xmlType: 'string', pythonType: 'str', typescriptType: 'string' },
          { name: 'field2', xmlType: 'integer', pythonType: 'int', typescriptType: 'number' }
        ]);

        await schemaEngine.generateModel(moClass, parameterMappings);
      }

      const endTime = Date.now();
      const totalTime = endTime - startTime;

      expect(totalTime).toBeLessThan(2000); // Less than 2 seconds
    });

    it('should handle complex models efficiently', async () => {
      const complexParameterMappings = createTestParameterMappings([]);

      // Create 50 fields with various types and constraints
      for (let i = 0; i < 50; i++) {
        complexParameterMappings.push({
          propertyName: `field${i}`,
          xmlType: i % 3 === 0 ? 'string' : i % 3 === 1 ? 'integer' : 'decimal',
          pythonType: i % 3 === 0 ? 'str' : i % 3 === 1 ? 'int' : 'Decimal',
          typescriptType: 'string' || 'number',
          constraints: i % 5 === 0 ? [
            { type: 'range', value: { min: 0, max: 100 }, severity: 'error' }
          ] : []
        });
      }

      const moClass = createTestMOClass('ComplexModel');

      const startTime = Date.now();
      const result = await schemaEngine.generateModel(moClass, complexParameterMappings);
      const endTime = Date.now();

      expect(result.fields).toHaveLength(50);
      expect(endTime - startTime).toBeLessThan(500); // Less than 0.5 seconds
    });
  });
});

// Helper functions
function createTestMOClass(name: string): MOClass {
  return {
    id: name,
    name,
    parentClass: 'ManagedElement',
    cardinality: { minimum: 0, maximum: 1, type: 'single' },
    flags: {},
    children: [],
    attributes: [],
    derivedClasses: []
  };
}

function createTestParameterMappings(fields: Array<{
  name: string;
  xmlType: string;
  pythonType: string;
  typescriptType: string;
  isOptional?: boolean;
  defaultValue?: any;
  constraints?: any[];
  mappingConfidence?: number;
}>): MappingResult[] {
  return fields.map(field => ({
    propertyName: field.name,
    xmlType: field.xmlType,
    pythonType: field.pythonType,
    typescriptType: field.typescriptType,
    constraints: field.constraints || [],
    defaultValue: field.defaultValue,
    isOptional: field.isOptional || false,
    mappingConfidence: field.mappingConfidence || 0.8
  }));
}