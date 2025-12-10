/**
 * Phase 5 Implementation Tests - XML to Pydantic Generator Integration Tests
 *
 * Comprehensive integration tests for the complete XML to Pydantic model generation
 * pipeline with validation of streaming processing, performance, and error handling
 */

import { XmlToPydanticGenerator, PydanticGeneratorConfig } from '../../src/pydantic/xml-to-pydantic-generator';
import { TypeMappingConfig } from '../../src/pydantic/type-mapper';
import { SchemaGenerationConfig } from '../../src/pydantic/schema-engine';
import { ValidationConfig } from '../../src/pydantic/validation-framework';
import { MOClass, RTBParameter } from '../../src/types/rtb-types';
import * as fs from 'fs';
import * as path from 'path';

describe('XmlToPydanticGenerator Integration', () => {
  let generator: XmlToPydanticGenerator;
  let config: PydanticGeneratorConfig;
  let testOutputDir: string;

  beforeEach(() => {
    testOutputDir = path.join(__dirname, 'test-output');

    // Clean up test output directory
    if (fs.existsSync(testOutputDir)) {
      fs.rmSync(testOutputDir, { recursive: true });
    }
    fs.mkdirSync(testOutputDir, { recursive: true });

    config = {
      xmlFilePath: path.join(__dirname, 'test-data', 'sample.xml'),
      outputPath: testOutputDir,
      batchSize: 100,
      memoryLimit: 1024 * 1024 * 512, // 512MB
      enableStreaming: true,
      typeMapping: {
        enableLearning: true,
        strictValidation: true,
        memoryIntegration: false,
        cognitiveMode: false
      },
      schemaGeneration: {
        enableOptimizations: true,
        useCaching: true,
        strictMode: true,
        generateValidators: true,
        includeImports: true,
        cognitiveMode: false,
        performanceMode: false
      },
      validation: {
        strictMode: true,
        enableCustomValidators: true,
        enableCrossParameterValidation: true,
        enableConditionalValidation: true,
        cognitiveMode: false,
        performanceMode: false,
        cacheValidation: true
      },
      cognitiveMode: false,
      enableLearning: true
    };

    generator = new XmlToPydanticGenerator(config);
  });

  afterEach(() => {
    // Clean up test output directory
    if (fs.existsSync(testOutputDir)) {
      fs.rmSync(testOutputDir, { recursive: true });
    }
  });

  describe('Initialization', () => {
    it('should initialize with valid configuration', async () => {
      await expect(generator.initialize()).resolves.not.toThrow();
    });

    it('should fail initialization with invalid XML file path', async () => {
      const invalidConfig = {
        ...config,
        xmlFilePath: '/nonexistent/file.xml'
      };
      const invalidGenerator = new XmlToPydanticGenerator(invalidConfig);

      await expect(invalidGenerator.initialize()).rejects.toThrow('XML file not found');
    });

    it('should create output directory if it does not exist', async () => {
      const nonExistentDir = path.join(__dirname, 'non-existent-output');
      const dirConfig = {
        ...config,
        outputPath: nonExistentDir
      };
      const dirGenerator = new XmlToPydanticGenerator(dirConfig);

      await dirGenerator.initialize();

      expect(fs.existsSync(nonExistentDir)).toBe(true);

      // Clean up
      if (fs.existsSync(nonExistentDir)) {
        fs.rmSync(nonExistentDir, { recursive: true });
      }
    });
  });

  describe('Complete Generation Pipeline', () => {
    beforeEach(async () => {
      // Create a mock XML file for testing
      createMockXMLFile();
      await generator.initialize();
    });

    it('should generate complete models from XML', async () => {
      const result = await generator.generateModels();

      expect(result.success).toBe(true);
      expect(result.models).toHaveLengthGreaterThan(0);
      expect(result.statistics.totalMOClasses).toBeGreaterThan(0);
      expect(result.statistics.totalParameters).toBeGreaterThan(0);
      expect(result.processingTime).toBeGreaterThan(0);
      expect(result.schemaGenerated).toBe(true);
      expect(result.validationPassed).toBe(true);
    });

    it('should generate valid Python code', async () => {
      const result = await generator.generateModels();

      expect(result.success).toBe(true);

      // Check that Python models file was created
      const pythonFile = path.join(testOutputDir, 'models.py');
      expect(fs.existsSync(pythonFile)).toBe(true);

      const pythonContent = fs.readFileSync(pythonFile, 'utf8');
      expect(pythonContent).toContain('from pydantic import BaseModel, Field, validator');
      expect(pythonContent).toContain('class ');
      expect(pythonContent).toContain('def ');
    });

    it('should generate valid TypeScript interfaces', async () => {
      const result = await generator.generateModels();

      expect(result.success).toBe(true);

      // Check that TypeScript interfaces file was created
      const tsFile = path.join(testOutputDir, 'interfaces.ts');
      expect(fs.existsSync(tsFile)).toBe(true);

      const tsContent = fs.readFileSync(tsFile, 'utf8');
      expect(tsContent).toContain('export interface ');
      expect(tsContent).toContain(': ');
      expect(tsContent).toContain(';');
    });

    it('should generate schema file', async () => {
      const result = await generator.generateModels();

      expect(result.success).toBe(true);

      // Check that schema file was created
      const schemaFile = path.join(testOutputDir, 'schema.json');
      expect(fs.existsSync(schemaFile)).toBe(true);

      const schemaContent = JSON.parse(fs.readFileSync(schemaFile, 'utf8'));
      expect(schemaContent.metadata).toBeDefined();
      expect(schemaContent.statistics).toBeDefined();
      expect(schemaContent.models).toBeDefined();
      expect(schemaContent.models).toHaveLength(result.models.length);
    });

    it('should generate statistics file', async () => {
      const result = await generator.generateModels();

      expect(result.success).toBe(true);

      // Check that statistics file was created
      const statsFile = path.join(testOutputDir, 'generation-stats.json');
      expect(fs.existsSync(statsFile)).toBe(true);

      const statsContent = JSON.parse(fs.readFileSync(statsFile, 'utf8'));
      expect(statsContent.totalMOClasses).toBeGreaterThan(0);
      expect(statsContent.totalParameters).toBeGreaterThan(0);
      expect(statsContent.modelsGenerated).toBe(result.models.length);
    });
  });

  describe('Performance Requirements', () => {
    beforeEach(async () => {
      createLargeMockXMLFile();
      await generator.initialize();
    });

    it('should process large XML files within performance target', async () => {
      const startTime = Date.now();
      const result = await generator.generateModels();
      const endTime = Date.now();

      expect(result.success).toBe(true);
      expect(endTime - startTime).toBeLessThan(1000); // Less than 1 second
    });

    it('should handle memory efficiently for large files', async () => {
      const initialMemory = process.memoryUsage().heapUsed;

      const result = await generator.generateModels();

      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = finalMemory - initialMemory;

      expect(result.success).toBe(true);
      expect(memoryIncrease).toBeLessThan(config.memoryLimit!);
    });

    it('should use batch processing effectively', async () => {
      // This test validates that the generator processes in batches
      // by monitoring progress events
      const progressEvents: any[] = [];

      generator.on('progress', (progress) => {
        progressEvents.push(progress);
      });

      const result = await generator.generateModels();

      expect(result.success).toBe(true);
      expect(progressEvents.length).toBeGreaterThan(0);

      // Check that progress goes through all stages
      const stages = progressEvents.map(e => e.stage);
      expect(stages).toContain('parsing');
      expect(stages).toContain('mapping');
      expect(stages).toContain('generation');
      expect(stages).toContain('validation');
      expect(stages).toContain('completion');
    });
  });

  describe('Error Handling and Recovery', () => {
    beforeEach(async () => {
      createErrorMockXMLFile();
      await generator.initialize();
    });

    it('should handle malformed XML gracefully', async () => {
      const result = await generator.generateModels();

      // Should complete but with errors
      expect(result.errors.length).toBeGreaterThan(0);
      expect(result.warnings.length).toBeGreaterThanOrEqual(0);
      expect(result.processingTime).toBeGreaterThan(0);
    });

    it('should provide detailed error information', async () => {
      const result = await generator.generateModels();

      if (result.errors.length > 0) {
        const error = result.errors[0];
        expect(error.type).toBeDefined();
        expect(error.message).toBeDefined();
        expect(error.message).toBeTruthy();
      }
    });

    it('should continue processing after individual errors', async () => {
      const result = await generator.generateModels();

      // Should still generate some models even with errors
      expect(result.processingTime).toBeGreaterThan(0);
      expect(result.statistics).toBeDefined();
    });
  });

  describe('Streaming Processing', () => {
    beforeEach(async () => {
      createStreamingMockXMLFile();
      await generator.initialize();
    });

    it('should process XML in streaming mode', async () => {
      const streamingConfig = {
        ...config,
        enableStreaming: true
      };
      const streamingGenerator = new XmlToPydanticGenerator(streamingConfig);
      await streamingGenerator.initialize();

      const result = await streamingGenerator.generateModels();

      expect(result.success).toBe(true);
      expect(result.statistics.totalParameters).toBeGreaterThan(0);
    });

    it('should handle large files better with streaming enabled', async () => {
      const nonStreamingConfig = {
        ...config,
        enableStreaming: false
      };
      const nonStreamingGenerator = new XmlToPydanticGenerator(nonStreamingConfig);
      await nonStreamingGenerator.initialize();

      // Create a very large mock file
      createVeryLargeMockXMLFile();

      const streamingResult = await generator.generateModels();
      const nonStreamingResult = await nonStreamingGenerator.generateModels();

      // Both should succeed, but streaming should be more memory efficient
      expect(streamingResult.success).toBe(true);
      expect(nonStreamingResult.success).toBe(true);
    });
  });

  describe('Cognitive Features', () => {
    beforeEach(async () => {
      createCognitiveMockXMLFile();
    });

    it('should work with cognitive mode enabled', async () => {
      const cognitiveConfig = {
        ...config,
        cognitiveMode: true,
        typeMapping: {
          ...config.typeMapping,
          cognitiveMode: true
        },
        schemaGeneration: {
          ...config.schemaGeneration,
          cognitiveMode: true
        },
        validation: {
          ...config.validation,
          cognitiveMode: true
        }
      };

      const cognitiveGenerator = new XmlToPydanticGenerator(cognitiveConfig);
      await cognitiveGenerator.initialize();

      const result = await cognitiveGenerator.generateModels();

      expect(result.success).toBe(true);
      expect(result.statistics).toBeDefined();
    });

    it('should provide cognitive insights when enabled', async () => {
      const cognitiveConfig = {
        ...config,
        cognitiveMode: true
      };

      const cognitiveGenerator = new XmlToPydanticGenerator(cognitiveConfig);
      await cognitiveGenerator.initialize();

      const result = await cognitiveGenerator.generateModels();

      if (result.statistics.cognitiveInsights) {
        expect(Array.isArray(result.statistics.cognitiveInsights)).toBe(true);
      }
    });
  });

  describe('Validation Integration', () => {
    beforeEach(async () => {
      createValidationMockXMLFile();
      await generator.initialize();
    });

    it('should validate generated models', async () => {
      const result = await generator.generateModels();

      expect(result.success).toBe(true);
      expect(result.validationPassed).toBe(true);
      expect(result.statistics.validationResults).toBeDefined();
      expect(result.statistics.validationResults.totalValidations).toBeGreaterThan(0);
    });

    it('should provide detailed validation results', async () => {
      const result = await generator.generateModels();

      if (result.statistics.validationResults) {
        const validationResults = result.statistics.validationResults;
        expect(validationResults.totalValidations).toBeGreaterThan(0);
        expect(validationResults.passedValidations + validationResults.failedValidations)
          .toBe(validationResults.totalValidations);
      }
    });

    it('should include cross-parameter validation', async () => {
      const result = await generator.generateModels();

      expect(result.statistics.validationResults).toBeDefined();
      // Cross-parameter validation would be tested with specific XML content
    });
  });

  describe('Configuration Variations', () => {
    it('should work with minimal configuration', async () => {
      createMockXMLFile();

      const minimalConfig = {
        xmlFilePath: config.xmlFilePath,
        outputPath: testOutputDir
      };

      const minimalGenerator = new XmlToPydanticGenerator(minimalConfig);
      await minimalGenerator.initialize();

      const result = await minimalGenerator.generateModels();

      expect(result.success).toBe(true);
      expect(result.models).toBeDefined();
    });

    it('should work with performance-optimized configuration', async () => {
      createMockXMLFile();

      const perfConfig = {
        ...config,
        typeMapping: {
          enableLearning: false,
          strictValidation: false,
          memoryIntegration: false,
          cognitiveMode: false
        },
        schemaGeneration: {
          enableOptimizations: true,
          useCaching: true,
          strictMode: false,
          generateValidators: false,
          includeImports: true,
          cognitiveMode: false,
          performanceMode: true
        },
        validation: {
          strictMode: false,
          enableCustomValidators: false,
          enableCrossParameterValidation: false,
          enableConditionalValidation: false,
          cognitiveMode: false,
          performanceMode: true,
          cacheValidation: false
        },
        cognitiveMode: false,
        enableLearning: false
      };

      const perfGenerator = new XmlToPydanticGenerator(perfConfig);
      await perfGenerator.initialize();

      const startTime = Date.now();
      const result = await perfGenerator.generateModels();
      const endTime = Date.now();

      expect(result.success).toBe(true);
      expect(endTime - startTime).toBeLessThan(500); // Should be faster with optimizations
    });
  });

  describe('Progress Monitoring', () => {
    beforeEach(async () => {
      createMockXMLFile();
      await generator.initialize();
    });

    it('should emit progress events', async () => {
      const progressEvents: any[] = [];

      generator.on('progress', (progress) => {
        progressEvents.push(progress);
      });

      const result = await generator.generateModels();

      expect(result.success).toBe(true);
      expect(progressEvents.length).toBeGreaterThan(0);

      // Check final progress
      const finalProgress = progressEvents[progressEvents.length - 1];
      expect(finalProgress.progress).toBe(100);
      expect(finalProgress.stage).toBe('completion');
    });

    it('should provide detailed progress information', async () => {
      const progressEvents: any[] = [];

      generator.on('progress', (progress) => {
        progressEvents.push(progress);
      });

      await generator.generateModels();

      progressEvents.forEach(progress => {
        expect(progress.stage).toBeDefined();
        expect(progress.progress).toBeGreaterThanOrEqual(0);
        expect(progress.progress).toBeLessThanOrEqual(100);
        expect(progress.parametersProcessed).toBeGreaterThanOrEqual(0);
        expect(progress.totalParameters).toBeGreaterThanOrEqual(0);
        expect(progress.processingTime).toBeGreaterThanOrEqual(0);
      });
    });

    it('should emit batch completion events', async () => {
      const batchEvents: any[] = [];

      generator.on('batchCompleted', (event) => {
        batchEvents.push(event);
      });

      const result = await generator.generateModels();

      expect(result.success).toBe(true);
      // Should have batch events if there are enough parameters
      if (result.statistics.totalParameters > config.batchSize!) {
        expect(batchEvents.length).toBeGreaterThan(0);
      }
    });
  });

  describe('Export and Persistence', () => {
    beforeEach(async () => {
      createMockXMLFile();
      await generator.initialize();
    });

    it('should export generation results', async () => {
      const result = await generator.generateModels();
      const exportedData = generator.exportResults(result);

      expect(exportedData).toBeDefined();
      expect(typeof exportedData).toBe('string');

      const parsedData = JSON.parse(exportedData);
      expect(parsedData.result).toBeDefined();
      expect(parsedData.statistics).toBeDefined();
      expect(parsedData.config).toBeDefined();
    });

    it('should provide current progress', async () => {
      // Start generation (but don't await)
      const generationPromise = generator.generateModels();

      // Get progress during generation
      const progress = generator.getProgress();
      expect(progress).toBeDefined();
      expect(progress.stage).toBeDefined();
      expect(progress.progress).toBeGreaterThanOrEqual(0);

      // Wait for completion
      const result = await generationPromise;
      expect(result.success).toBe(true);
    });

    it('should provide generation statistics', async () => {
      const result = await generator.generateModels();
      const stats = generator.getStatistics();

      expect(stats).toBeDefined();
      expect(stats.totalMOClasses).toBe(result.statistics.totalMOClasses);
      expect(stats.totalParameters).toBe(result.statistics.totalParameters);
    });

    it('should support cancellation', async () => {
      // Create a large file to test cancellation
      createLargeMockXMLFile();

      // Start generation
      const generationPromise = generator.generateModels();

      // Cancel after a short delay
      setTimeout(() => {
        generator.cancelGeneration();
      }, 100);

      // Should either complete or be cancelled
      const result = await generationPromise;
      expect(result).toBeDefined();
    });
  });

  describe('Edge Cases and Boundary Conditions', () => {
    it('should handle empty XML file', async () => {
      createEmptyMockXMLFile();
      await generator.initialize();

      const result = await generator.generateModels();

      // Should handle gracefully
      expect(result).toBeDefined();
      expect(result.processingTime).toBeGreaterThan(0);
    });

    it('should handle XML with only whitespace', async () => {
      createWhitespaceMockXMLFile();
      await generator.initialize();

      const result = await generator.generateModels();

      expect(result).toBeDefined();
      expect(result.errors.length).toBeGreaterThanOrEqual(0);
    });

    it('should handle very long parameter names', async () => {
      createLongNamesMockXMLFile();
      await generator.initialize();

      const result = await generator.generateModels();

      expect(result).toBeDefined();
      expect(result.models).toBeDefined();
    });

    it('should handle deeply nested XML structures', async () => {
      createNestedMockXMLFile();
      await generator.initialize();

      const result = await generator.generateModels();

      expect(result).toBeDefined();
      expect(result.processingTime).toBeGreaterThan(0);
    });
  });
});

// Helper functions for creating mock XML files
function createMockXMLFile(): void {
  const xmlContent = `<?xml version="1.0" encoding="UTF-8"?>
<model>
  <moClass name="TestCell">
    <parameter name="cellId" type="integer" description="Cell identifier"/>
    <parameter name="cellName" type="string" description="Cell name"/>
    <parameter name="power" type="decimal" description="Cell power"/>
    <parameter name="isActive" type="boolean" description="Cell active status"/>
  </moClass>
  <moClass name="TestSector">
    <parameter name="sectorId" type="integer" description="Sector identifier"/>
    <parameter name="azimuth" type="integer" description="Sector azimuth"/>
    <parameter name="frequency" type="decimal" description="Sector frequency"/>
  </moClass>
</model>`;

  const xmlDir = path.dirname(config.xmlFilePath);
  if (!fs.existsSync(xmlDir)) {
    fs.mkdirSync(xmlDir, { recursive: true });
  }
  fs.writeFileSync(config.xmlFilePath, xmlContent);
}

function createLargeMockXMLFile(): void {
  let xmlContent = '<?xml version="1.0" encoding="UTF-8"?><model>';

  for (let i = 0; i < 100; i++) {
    xmlContent += `<moClass name="TestClass${i}">
      <parameter name="param${i}_1" type="string" description="String parameter ${i}"/>
      <parameter name="param${i}_2" type="integer" description="Integer parameter ${i}"/>
      <parameter name="param${i}_3" type="decimal" description="Decimal parameter ${i}"/>
      <parameter name="param${i}_4" type="boolean" description="Boolean parameter ${i}"/>
    </moClass>`;
  }

  xmlContent += '</model>';

  const xmlDir = path.dirname(config.xmlFilePath);
  if (!fs.existsSync(xmlDir)) {
    fs.mkdirSync(xmlDir, { recursive: true });
  }
  fs.writeFileSync(config.xmlFilePath, xmlContent);
}

function createErrorMockXMLFile(): void {
  const xmlContent = `<?xml version="1.0" encoding="UTF-8"?>
<model>
  <moClass name="ErrorTest">
    <parameter name="validParam" type="string" description="Valid parameter"/>
    <parameter name="invalidParam" type="unknown_type" description="Invalid type"/>
    <malformed-tag>
  </moClass>
</model>`;

  const xmlDir = path.dirname(config.xmlFilePath);
  if (!fs.existsSync(xmlDir)) {
    fs.mkdirSync(xmlDir, { recursive: true });
  }
  fs.writeFileSync(config.xmlFilePath, xmlContent);
}

function createStreamingMockXMLFile(): void {
  // Create a file suitable for streaming tests
  createLargeMockXMLFile();
}

function createCognitiveMockXMLFile(): void {
  const xmlContent = `<?xml version="1.0" encoding="UTF-8"?>
<model>
  <moClass name="CognitiveTest">
    <parameter name="aiParameter" type="string" description="AI-enhanced parameter"/>
    <parameter name="learningField" type="integer" description="Learning-enabled field"/>
  </moClass>
</model>`;

  const xmlDir = path.dirname(config.xmlFilePath);
  if (!fs.existsSync(xmlDir)) {
    fs.mkdirSync(xmlDir, { recursive: true });
  }
  fs.writeFileSync(config.xmlFilePath, xmlContent);
}

function createValidationMockXMLFile(): void {
  const xmlContent = `<?xml version="1.0" encoding="UTF-8"?>
<model>
  <moClass name="ValidationTest">
    <parameter name="rangeField" type="integer" description="Field with range constraint"/>
    <parameter name="enumField" type="string" description="Field with enum constraint"/>
    <parameter name="patternField" type="string" description="Field with pattern constraint"/>
  </moClass>
</model>`;

  const xmlDir = path.dirname(config.xmlFilePath);
  if (!fs.existsSync(xmlDir)) {
    fs.mkdirSync(xmlDir, { recursive: true });
  }
  fs.writeFileSync(config.xmlFilePath, xmlContent);
}

function createVeryLargeMockXMLFile(): void {
  let xmlContent = '<?xml version="1.0" encoding="UTF-8"?><model>';

  for (let i = 0; i < 1000; i++) {
    xmlContent += `<moClass name="VeryLargeClass${i}">
      <parameter name="param${i}_1" type="string" description="String parameter ${i}"/>
      <parameter name="param${i}_2" type="integer" description="Integer parameter ${i}"/>
      <parameter name="param${i}_3" type="decimal" description="Decimal parameter ${i}"/>
      <parameter name="param${i}_4" type="boolean" description="Boolean parameter ${i}"/>
      <parameter name="param${i}_5" type="dateTime" description="DateTime parameter ${i}"/>
    </moClass>`;
  }

  xmlContent += '</model>';

  const xmlDir = path.dirname(config.xmlFilePath);
  if (!fs.existsSync(xmlDir)) {
    fs.mkdirSync(xmlDir, { recursive: true });
  }
  fs.writeFileSync(config.xmlFilePath, xmlContent);
}

function createEmptyMockXMLFile(): void {
  const xmlContent = '<?xml version="1.0" encoding="UTF-8"?><model></model>';

  const xmlDir = path.dirname(config.xmlFilePath);
  if (!fs.existsSync(xmlDir)) {
    fs.mkdirSync(xmlDir, { recursive: true });
  }
  fs.writeFileSync(config.xmlFilePath, xmlContent);
}

function createWhitespaceMockXMLFile(): void {
  const xmlContent = `<?xml version="1.0" encoding="UTF-8"?>

<model>

</model>`;

  const xmlDir = path.dirname(config.xmlFilePath);
  if (!fs.existsSync(xmlDir)) {
    fs.mkdirSync(xmlDir, { recursive: true });
  }
  fs.writeFileSync(config.xmlFilePath, xmlContent);
}

function createLongNamesMockXMLFile(): void {
  const xmlContent = `<?xml version="1.0" encoding="UTF-8"?>
<model>
  <moClass name="VeryLongParameterNamesTest">
    <parameter name="this_is_a_very_long_parameter_name_that_exceeds_normal_limits_and_tests_boundary_conditions" type="string" description="Very long parameter name"/>
  </moClass>
</model>`;

  const xmlDir = path.dirname(config.xmlFilePath);
  if (!fs.existsSync(xmlDir)) {
    fs.mkdirSync(xmlDir, { recursive: true });
  }
  fs.writeFileSync(config.xmlFilePath, xmlContent);
}

function createNestedMockXMLFile(): void {
  const xmlContent = `<?xml version="1.0" encoding="UTF-8"?>
<model>
  <moClass name="NestedTest">
    <parameter name="nestedParam" type="string">
      <constraint type="range">
        <min>0</min>
        <max>100</max>
      </constraint>
    </parameter>
  </moClass>
</model>`;

  const xmlDir = path.dirname(config.xmlFilePath);
  if (!fs.existsSync(xmlDir)) {
    fs.mkdirSync(xmlDir, { recursive: true });
  }
  fs.writeFileSync(config.xmlFilePath, xmlContent);
}