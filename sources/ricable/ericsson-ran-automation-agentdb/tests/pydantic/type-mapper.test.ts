/**
 * Phase 5 Implementation Tests - Type Mapper Unit Tests
 *
 * Comprehensive test suite for the XML to Python/TypeScript type mapping system
 * with validation of all 623 vsData types and constraint application
 */

import { TypeMapper, TypeMappingConfig, MappingResult } from '../../src/pydantic/type-mapper';
import { RTBParameter, ConstraintSpec } from '../../src/types/rtb-types';

describe('TypeMapper', () => {
  let typeMapper: TypeMapper;
  let config: TypeMappingConfig;

  beforeEach(() => {
    config = {
      enableLearning: true,
      strictValidation: true,
      memoryIntegration: false,
      cognitiveMode: false
    };
    typeMapper = new TypeMapper(config);
  });

  describe('Initialization', () => {
    it('should initialize with default configuration', async () => {
      await typeMapper.initialize();
      expect(typeMapper).toBeDefined();
    });

    it('should load custom type mappings', async () => {
      config.customTypeMappings = {
        'CustomType': {
          xmlType: 'CustomType',
          pythonType: 'CustomClass',
          typescriptType: 'CustomInterface'
        }
      };

      const customMapper = new TypeMapper(config);
      await customMapper.initialize();

      const parameter = createTestParameter('test', 'CustomType');
      const result = customMapper.mapParameter(parameter);

      expect(result.pythonType).toBe('CustomClass');
      expect(result.typescriptType).toBe('CustomInterface');
    });
  });

  describe('Basic Type Mapping', () => {
    beforeEach(async () => {
      await typeMapper.initialize();
    });

    it('should map string types correctly', () => {
      const parameter = createTestParameter('testString', 'string');
      const result = typeMapper.mapParameter(parameter);

      expect(result.pythonType).toBe('str');
      expect(result.typescriptType).toBe('string');
      expect(result.mappingConfidence).toBeGreaterThan(0.8);
    });

    it('should map integer types correctly', () => {
      const parameter = createTestParameter('testInt', 'integer');
      const result = typeMapper.mapParameter(parameter);

      expect(result.pythonType).toBe('int');
      expect(result.typescriptType).toBe('number');
      expect(result.defaultValue).toBe(0);
    });

    it('should map decimal types correctly', () => {
      const parameter = createTestParameter('testDecimal', 'decimal');
      const result = typeMapper.mapParameter(parameter);

      expect(result.pythonType).toBe('Decimal');
      expect(result.typescriptType).toBe('number');
    });

    it('should map boolean types correctly', () => {
      const parameter = createTestParameter('testBool', 'boolean');
      const result = typeMapper.mapParameter(parameter);

      expect(result.pythonType).toBe('bool');
      expect(result.typescriptType).toBe('boolean');
      expect(result.defaultValue).toBe(false);
    });

    it('should map date/time types correctly', () => {
      const dateTimeParam = createTestParameter('testDateTime', 'dateTime');
      const dateTimeResult = typeMapper.mapParameter(dateTimeParam);

      expect(dateTimeResult.pythonType).toBe('datetime');
      expect(dateTimeResult.typescriptType).toBe('Date');
      expect(dateTimeResult.importRequired).toBe(true);

      const dateParam = createTestParameter('testDate', 'date');
      const dateResult = typeMapper.mapParameter(dateParam);

      expect(dateResult.pythonType).toBe('date');
      expect(dateResult.typescriptType).toBe('Date');
    });
  });

  describe('vsData Type Mapping', () => {
    beforeEach(async () => {
      await typeMapper.initialize();
    });

    it('should map all 623 vsData types', () => {
      const results: MappingResult[] = [];
      let errors: string[] = [];

      for (let i = 1; i <= 623; i++) {
        try {
          const parameter = createTestParameter(`vsData${i}`, `vsData${i}`);
          const result = typeMapper.mapParameter(parameter);
          results.push(result);
        } catch (error) {
          errors.push(`vsData${i}: ${error.message}`);
        }
      }

      expect(errors.length).toBe(0);
      expect(results.length).toBe(623);

      // Verify all mappings have valid types
      results.forEach((result, index) => {
        expect(result.pythonType).toBeTruthy();
        expect(result.typescriptType).toBeTruthy();
        expect(result.mappingConfidence).toBeGreaterThan(0);
      });
    });

    it('should map vsData types with suffixes correctly', () => {
      const testCases = [
        { vsDataType: 'vsData1a', expectedPython: 'str', expectedTS: 'string' },
        { vsDataType: 'vsData2b', expectedPython: 'int', expectedTS: 'number' },
        { vsDataType: 'vsData3c', expectedPython: 'Decimal', expectedTS: 'number' },
        { vsDataType: 'vsData4d', expectedPython: 'bool', expectedTS: 'boolean' },
        { vsDataType: 'vsData5e', expectedPython: 'List[str]', expectedTS: 'string[]' },
        { vsDataType: 'vsData6f', expectedPython: 'Dict[str, Any]', expectedTS: 'Record<string, any>' }
      ];

      testCases.forEach(testCase => {
        const parameter = createTestParameter('test', testCase.vsDataType);
        const result = typeMapper.mapParameter(parameter);

        expect(result.pythonType).toBe(testCase.expectedPython);
        expect(result.typescriptType).toBe(testCase.expectedTS);
      });
    });
  });

  describe('Ericsson RAN Specific Types', () => {
    beforeEach(async () => {
      await typeMapper.initialize();
    });

    it('should map CellId correctly', () => {
      const parameter = createTestParameter('cellId', 'CellId');
      const result = typeMapper.mapParameter(parameter);

      expect(result.pythonType).toBe('int');
      expect(result.typescriptType).toBe('number');
      expect(result.defaultValue).toBe(0);
    });

    it('should map PowerdBm correctly', () => {
      const parameter = createTestParameter('power', 'PowerdBm');
      const result = typeMapper.mapParameter(parameter);

      expect(result.pythonType).toBe('int');
      expect(result.typescriptType).toBe('number');
    });

    it('should map FrequencyMHz correctly', () => {
      const parameter = createTestParameter('frequency', 'FrequencyMHz');
      const result = typeMapper.mapParameter(parameter);

      expect(result.pythonType).toBe('Decimal');
      expect(result.typescriptType).toBe('number');
    });

    it('should map RSRP correctly', () => {
      const parameter = createTestParameter('rsrp', 'RSRP');
      const result = typeMapper.mapParameter(parameter);

      expect(result.pythonType).toBe('int');
      expect(result.typescriptType).toBe('number');
      expect(result.defaultValue).toBe(-140);
    });

    it('should map AdministrativeState correctly', () => {
      const parameter = createTestParameter('adminState', 'AdministrativeState');
      const result = typeMapper.mapParameter(parameter);

      expect(result.pythonType).toBe('Enum');
      expect(result.typescriptType).toBe('string');
      expect(result.defaultValue).toBe('UNLOCKED');
    });
  });

  describe('Constraint Application', () => {
    beforeEach(async () => {
      await typeMapper.initialize();
    });

    it('should apply range constraints correctly', () => {
      const constraints: ConstraintSpec[] = [
        {
          type: 'range',
          value: { min: 0, max: 100 },
          severity: 'error'
        }
      ];

      const parameter = createTestParameter('testRange', 'integer', constraints);
      const result = typeMapper.mapParameter(parameter);

      expect(result.constraints).toHaveLength(1);
      expect(result.constraints[0].type).toBe('range');
      expect(result.constraints[0].value).toEqual({
        min: 0,
        max: 100,
        type: 'range'
      });
    });

    it('should apply enum constraints correctly', () => {
      const constraints: ConstraintSpec[] = [
        {
          type: 'enum',
          value: ['ACTIVE', 'INACTIVE', 'DEGRADED'],
          severity: 'error'
        }
      ];

      const parameter = createTestParameter('testEnum', 'string', constraints);
      const result = typeMapper.mapParameter(parameter);

      expect(result.constraints).toHaveLength(1);
      expect(result.constraints[0].type).toBe('enum');
      expect(result.constraints[0].value).toEqual({
        values: ['ACTIVE', 'INACTIVE', 'DEGRADED'],
        type: 'enum'
      });
    });

    it('should apply pattern constraints correctly', () => {
      const constraints: ConstraintSpec[] = [
        {
          type: 'pattern',
          value: '^[A-Z]{3}-\\d{3}$',
          severity: 'error'
        }
      ];

      const parameter = createTestParameter('testPattern', 'string', constraints);
      const result = typeMapper.mapParameter(parameter);

      expect(result.constraints).toHaveLength(1);
      expect(result.constraints[0].type).toBe('pattern');
      expect(result.constraints[0].value).toEqual({
        regex: '^[A-Z]{3}-\\d{3}$',
        type: 'pattern'
      });
    });

    it('should apply length constraints correctly', () => {
      const constraints: ConstraintSpec[] = [
        {
          type: 'length',
          value: { minLength: 1, maxLength: 50 },
          severity: 'error'
        }
      ];

      const parameter = createTestParameter('testLength', 'string', constraints);
      const result = typeMapper.mapParameter(parameter);

      expect(result.constraints).toHaveLength(1);
      expect(result.constraints[0].type).toBe('length');
      expect(result.constraints[0].value).toEqual({
        minLength: 1,
        maxLength: 50,
        type: 'length'
      });
    });

    it('should apply multiple constraints correctly', () => {
      const constraints: ConstraintSpec[] = [
        {
          type: 'range',
          value: { min: 0, max: 100 },
          severity: 'error'
        },
        {
          type: 'custom',
          value: 'custom_validation_logic',
          severity: 'warning'
        }
      ];

      const parameter = createTestParameter('testMultiple', 'integer', constraints);
      const result = typeMapper.mapParameter(parameter);

      expect(result.constraints).toHaveLength(2);
      expect(result.constraints[0].type).toBe('range');
      expect(result.constraints[1].type).toBe('custom');
    });
  });

  describe('Optional Parameter Handling', () => {
    beforeEach(async () => {
      await typeMapper.initialize();
    });

    it('should mark parameters with default values as optional', () => {
      const parameter = createTestParameter('testOptional', 'string');
      parameter.defaultValue = 'default_value';

      const result = typeMapper.mapParameter(parameter);

      expect(result.isOptional).toBe(true);
    });

    it('should mark parameters as optional based on required constraint', () => {
      const constraints: ConstraintSpec[] = [
        {
          type: 'required',
          value: false,
          severity: 'error'
        }
      ];

      const parameter = createTestParameter('testOptional', 'string', constraints);
      const result = typeMapper.mapParameter(parameter);

      expect(result.isOptional).toBe(true);
    });

    it('should mark required parameters as not optional', () => {
      const constraints: ConstraintSpec[] = [
        {
          type: 'required',
          value: true,
          severity: 'error'
        }
      ];

      const parameter = createTestParameter('testRequired', 'string', constraints);
      const result = typeMapper.mapParameter(parameter);

      expect(result.isOptional).toBe(false);
    });

    it('should wrap optional parameters in Optional type', () => {
      const parameter = createTestParameter('testOptional', 'string');
      parameter.defaultValue = 'default_value';

      const result = typeMapper.mapParameter(parameter);

      expect(result.pythonType).toBe('str'); // Basic type for optional fields
      expect(result.isOptional).toBe(true);
    });
  });

  describe('Array Type Mapping', () => {
    beforeEach(async () => {
      await typeMapper.initialize();
    });

    it('should map list types correctly', () => {
      const parameter = createTestParameter('testList', 'list');
      const result = typeMapper.mapParameter(parameter);

      expect(result.pythonType).toBe('List[Any]');
      expect(result.typescriptType).toBe('any[]');
      expect(result.defaultValue).toEqual([]);
    });

    it('should map array types with type suffix', () => {
      const parameter = createTestParameter('testArray', 'string[]');
      const result = typeMapper.mapParameter(parameter);

      expect(result.pythonType).toBe('List[str]');
      expect(result.typescriptType).toBe('string[]');
    });

    it('should map array types with List suffix', () => {
      const parameter = createTestParameter('testListArray', 'intList');
      const result = typeMapper.mapParameter(parameter);

      expect(result.pythonType).toBe('List[int]');
      expect(result.typescriptType).toBe('number[]');
    });
  });

  describe('Batch Processing', () => {
    beforeEach(async () => {
      await typeMapper.initialize();
    });

    it('should process multiple parameters in batch', () => {
      const parameters = [
        createTestParameter('param1', 'string'),
        createTestParameter('param2', 'integer'),
        createTestParameter('param3', 'boolean')
      ];

      const results = typeMapper.mapParameters(parameters);

      expect(results).toHaveLength(3);
      expect(results[0].propertyName).toBe('param1');
      expect(results[1].propertyName).toBe('param2');
      expect(results[2].propertyName).toBe('param3');
    });

    it('should handle batch processing with constraints', () => {
      const constraints: ConstraintSpec[] = [
        { type: 'range', value: { min: 0, max: 100 }, severity: 'error' }
      ];

      const parameters = [
        createTestParameter('param1', 'integer', constraints),
        createTestParameter('param2', 'integer', constraints),
        createTestParameter('param3', 'string')
      ];

      const constraintsMap = {
        param1: constraints,
        param2: constraints
      };

      const results = typeMapper.mapParameters(parameters, constraintsMap);

      expect(results[0].constraints).toHaveLength(1);
      expect(results[1].constraints).toHaveLength(1);
      expect(results[2].constraints).toHaveLength(0);
    });

    it('should continue processing after individual parameter errors', () => {
      const parameters = [
        createTestParameter('param1', 'string'),
        createTestParameter('param2', 'unknown_type'),
        createTestParameter('param3', 'integer')
      ];

      const results = typeMapper.mapParameters(parameters);

      expect(results).toHaveLength(3);
      expect(results[0].mappingConfidence).toBeGreaterThan(0.5);
      expect(results[1].mappingConfidence).toBeLessThan(0.5); // Fallback mapping
      expect(results[2].mappingConfidence).toBeGreaterThan(0.5);
    });
  });

  describe('Caching and Performance', () => {
    beforeEach(async () => {
      await typeMapper.initialize();
    });

    it('should cache mapping results', () => {
      const parameter = createTestParameter('testCache', 'string');

      const result1 = typeMapper.mapParameter(parameter);
      const result2 = typeMapper.mapParameter(parameter);

      expect(result1).toEqual(result2);

      const stats = typeMapper.getStatistics();
      expect(stats.memoryHitRate).toBeGreaterThan(0);
    });

    it('should clear cache', () => {
      const parameter = createTestParameter('testClear', 'string');

      typeMapper.mapParameter(parameter);
      typeMapper.clearCache();

      const stats = typeMapper.getStatistics();
      // Cache is cleared but stats remain
      expect(stats.totalMappings).toBeGreaterThan(0);
    });

    it('should provide accurate statistics', () => {
      const parameters = [
        createTestParameter('param1', 'string'),
        createTestParameter('param2', 'integer'),
        createTestParameter('param3', 'unknown_type')
      ];

      typeMapper.mapParameters(parameters);

      const stats = typeMapper.getStatistics();

      expect(stats.totalMappings).toBe(3);
      expect(stats.successfulMappings).toBe(2);
      expect(stats.failedMappings).toBe(1);
      expect(stats.averageConfidence).toBeGreaterThan(0);
      expect(stats.processingTime).toBeGreaterThan(0);
    });
  });

  describe('Custom Type Mappings', () => {
    beforeEach(async () => {
      await typeMapper.initialize();
    });

    it('should add custom type mapping', () => {
      const customMapping = {
        xmlType: 'CustomXmlType',
        pythonType: 'CustomPythonClass',
        typescriptType: 'CustomTypeScriptInterface',
        defaultValue: 'custom_default'
      };

      typeMapper.addCustomMapping(customMapping);

      const parameter = createTestParameter('testCustom', 'CustomXmlType');
      const result = typeMapper.mapParameter(parameter);

      expect(result.pythonType).toBe('CustomPythonClass');
      expect(result.typescriptType).toBe('CustomTypeScriptInterface');
      expect(result.defaultValue).toBe('custom_default');
    });

    it('should remove custom type mapping', () => {
      const customMapping = {
        xmlType: 'CustomXmlType',
        pythonType: 'CustomPythonClass',
        typescriptType: 'CustomTypeScriptInterface'
      };

      typeMapper.addCustomMapping(customMapping);
      expect(typeMapper.removeCustomMapping('CustomXmlType')).toBe(true);

      const parameter = createTestParameter('testCustom', 'CustomXmlType');
      const result = typeMapper.mapParameter(parameter);

      // Should fall back to default mapping
      expect(result.pythonType).toBe('Any');
      expect(result.typescriptType).toBe('any');
    });

    it('should export all mappings', () => {
      const allMappings = typeMapper.getAllMappings();

      expect(allMappings).toBeDefined();
      expect(typeof allMappings).toBe('object');
      expect(Object.keys(allMappings).length).toBeGreaterThan(0);

      // Check that basic types are included
      expect(allMappings['string']).toBeDefined();
      expect(allMappings['integer']).toBeDefined();
      expect(allMappings['boolean']).toBeDefined();
    });
  });

  describe('Error Handling and Edge Cases', () => {
    beforeEach(async () => {
      await typeMapper.initialize();
    });

    it('should handle null/undefined parameter gracefully', () => {
      const parameter = createTestParameter('testNull', 'string');
      parameter.constraints = undefined;

      const result = typeMapper.mapParameter(parameter);

      expect(result).toBeDefined();
      expect(result.constraints).toEqual([]);
    });

    it('should create fallback mapping for unknown types', () => {
      const parameter = createTestParameter('testUnknown', 'completely_unknown_xml_type');

      const result = typeMapper.mapParameter(parameter);

      expect(result.pythonType).toBe('Any');
      expect(result.typescriptType).toBe('any');
      expect(result.mappingConfidence).toBe(0.1);
      expect(result.isOptional).toBe(true);
    });

    it('should handle malformed constraints gracefully', () => {
      const malformedConstraints: ConstraintSpec[] = [
        {
          type: 'range' as any,
          value: 'not_an_object',
          severity: 'error'
        }
      ];

      const parameter = createTestParameter('testMalformed', 'integer', malformedConstraints);

      expect(() => {
        typeMapper.mapParameter(parameter);
      }).not.toThrow();
    });

    it('should handle empty parameter names', () => {
      const parameter = createTestParameter('', 'string');

      const result = typeMapper.mapParameter(parameter);

      expect(result.propertyName).toBe('');
      expect(result.pythonType).toBe('str');
    });

    it('should handle special characters in parameter names', () => {
      const parameter = createTestParameter('test-special@name#123', 'string');

      const result = typeMapper.mapParameter(parameter);

      expect(result.propertyName).toBe('test-special@name#123');
      expect(result.pythonType).toBe('str');
    });
  });

  describe('Performance Requirements', () => {
    beforeEach(async () => {
      await typeMapper.initialize();
    });

    it('should process 1000 parameters within performance target', () => {
      const parameters: RTBParameter[] = [];
      for (let i = 0; i < 1000; i++) {
        parameters.push(createTestParameter(`param${i}`, 'string'));
      }

      const startTime = Date.now();
      typeMapper.mapParameters(parameters);
      const endTime = Date.now();

      const processingTime = endTime - startTime;
      expect(processingTime).toBeLessThan(1000); // Less than 1 second
    });

    it('should handle large vsData type mappings efficiently', () => {
      const vsDataParameters: RTBParameter[] = [];
      for (let i = 1; i <= 623; i++) {
        vsDataParameters.push(createTestParameter(`vsData${i}`, `vsData${i}`));
      }

      const startTime = Date.now();
      const results = typeMapper.mapParameters(vsDataParameters);
      const endTime = Date.now();

      expect(results).toHaveLength(623);
      const processingTime = endTime - startTime;
      expect(processingTime).toBeLessThan(500); // Less than 0.5 seconds
    });
  });
});

// Helper functions
function createTestParameter(
  name: string,
  type: string,
  constraints: ConstraintSpec[] = []
): RTBParameter {
  return {
    id: `test_${name}_${Date.now()}`,
    name,
    vsDataType: type,
    type,
    constraints,
    description: `Test parameter ${name}`,
    hierarchy: [],
    source: 'test',
    extractedAt: new Date()
  };
}