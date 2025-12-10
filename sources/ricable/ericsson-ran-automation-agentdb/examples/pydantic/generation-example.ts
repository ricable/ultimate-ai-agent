/**
 * Phase 5 Implementation Example - Pydantic Model Generation
 *
 * Example demonstrating how to use the XML to Pydantic model generator
 * with full system integration including AgentDB memory, temporal reasoning,
 * and cognitive consciousness
 */

import * as path from 'path';
import { createXmlToPydanticGenerator, PydanticGeneratorConfig } from '../../src/pydantic';
import { PydanticIntegration, IntegrationConfig } from '../../src/pydantic/integration';
import { AgentDBIntegration } from '../../src/closed-loop/agentdb-integration';
import { TemporalReasoningCore } from '../../src/closed-loop/temporal-reasoning';
import { CognitiveConsciousnessCore } from '../../src/cognitive/CognitiveConsciousnessCore';

/**
 * Basic example: Simple XML to Pydantic generation
 */
async function basicGenerationExample() {
  console.log('=== Basic Generation Example ===');

  const config: PydanticGeneratorConfig = {
    xmlFilePath: path.join(__dirname, 'sample-data', 'basic-example.xml'),
    outputPath: path.join(__dirname, 'output', 'basic-example'),
    batchSize: 100,
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
      includeImports: true
    },
    validation: {
      strictMode: true,
      enableCustomValidators: true,
      enableCrossParameterValidation: true,
      enableConditionalValidation: true
    }
  };

  const generator = createXmlToPydanticGenerator(config);
  await generator.initialize();

  console.log('Starting model generation...');

  // Listen to progress events
  generator.on('progress', (progress) => {
    console.log(`Progress: ${progress.stage} - ${progress.progress.toFixed(1)}%`);
    console.log(`  Parameters processed: ${progress.parametersProcessed}/${progress.totalParameters}`);
    console.log(`  Memory usage: ${(progress.memoryUsage / 1024 / 1024).toFixed(1)}MB`);
  });

  generator.on('modelGenerated', (event) => {
    console.log(`Generated model: ${event.className} (confidence: ${event.confidence.toFixed(2)})`);
  });

  const result = await generator.generateModels();

  console.log('\nGeneration Results:');
  console.log(`✅ Success: ${result.success}`);
  console.log(`📊 Models generated: ${result.models.length}`);
  console.log(`⏱️  Processing time: ${result.processingTime}ms`);
  console.log(`📁 Output directory: ${config.outputPath}`);
  console.log(`🔍 Validation passed: ${result.validationPassed}`);

  if (result.errors.length > 0) {
    console.log(`❌ Errors: ${result.errors.length}`);
    result.errors.forEach(error => {
      console.log(`   - ${error.type}: ${error.message}`);
    });
  }

  if (result.warnings.length > 0) {
    console.log(`⚠️  Warnings: ${result.warnings.length}`);
    result.warnings.forEach(warning => {
      console.log(`   - ${warning.type}: ${warning.message}`);
    });
  }

  console.log('\nGenerated Files:');
  console.log('  📄 models.py - Pydantic model classes');
  console.log('  📄 interfaces.ts - TypeScript interfaces');
  console.log('  📄 schema.json - Complete schema definition');
  console.log('  📄 generation-stats.json - Generation statistics');
}

/**
 * Advanced example: Full system integration with cognitive features
 */
async function advancedIntegrationExample() {
  console.log('\n=== Advanced Integration Example ===');

  // Initialize system components
  const agentDB = new AgentDBIntegration({
    connectionString: 'memory://test-agentdb',
    enableLearning: true,
    syncInterval: 1000
  });

  const temporalReasoning = new TemporalReasoningCore({
    expansionFactor: 1000,
    reasoningDepth: 'deep',
    enablePrediction: true
  });

  const cognitiveConsciousness = new CognitiveConsciousnessCore({
    consciousnessLevel: 'maximum',
    enableStrangeLoopOptimization: true,
    enableMetaCognition: true
  });

  await Promise.all([
    agentDB.initialize(),
    temporalReasoning.initialize(),
    cognitiveConsciousness.initialize()
  ]);

  // Configure integration
  const integrationConfig: IntegrationConfig = {
    agentDBIntegration: agentDB,
    temporalReasoning: temporalReasoning,
    cognitiveConsciousness: cognitiveConsciousness,
    enableMemoryLearning: true,
    enableTemporalAnalysis: true,
    enableCognitiveOptimization: true,
    performanceOptimization: true
  };

  const integration = new PydanticIntegration(integrationConfig);
  await integration.initialize();

  // Configure generator with advanced features
  const config: PydanticGeneratorConfig = {
    xmlFilePath: path.join(__dirname, 'sample-data', 'advanced-example.xml'),
    outputPath: path.join(__dirname, 'output', 'advanced-example'),
    batchSize: 1000,
    memoryLimit: 1024 * 1024 * 1024, // 1GB
    enableStreaming: true,
    typeMapping: {
      enableLearning: true,
      strictValidation: true,
      memoryIntegration: true,
      cognitiveMode: true
    },
    schemaGeneration: {
      enableOptimizations: true,
      useCaching: true,
      strictMode: true,
      generateValidators: true,
      includeImports: true,
      cognitiveMode: true,
      performanceMode: true
    },
    validation: {
      strictMode: true,
      enableCustomValidators: true,
      enableCrossParameterValidation: true,
      enableConditionalValidation: true,
      cognitiveMode: true,
      performanceMode: true
    },
    cognitiveMode: true,
    enableLearning: true
  };

  console.log('Starting advanced generation with full system integration...');

  // Listen to integration events
  integration.on('integrationStarted', () => {
    console.log('🚀 Integration started');
  });

  integration.on('memoryLearningCompleted', (event) => {
    console.log(`🧠 Memory learning: ${event.patternsLearned} patterns learned`);
  });

  integration.on('temporalAnalysisCompleted', (event) => {
    console.log(`⏰ Temporal analysis: ${event.patternsFound} patterns with ${event.expansionFactor}x expansion`);
  });

  integration.on('cognitiveOptimizationCompleted', (event) => {
    console.log(`🤖 Cognitive optimization: ${event.optimizationsGenerated} optimizations (level: ${event.consciousnessLevel})`);
  });

  const result = await integration.generateWithIntegration(config);

  console.log('\nAdvanced Integration Results:');
  console.log(`✅ Success: ${result.success}`);
  console.log(`📊 Cognitive insights: ${result.cognitiveInsights.length}`);
  console.log(`🧠 Learned patterns: ${result.learnedPatterns.length}`);
  console.log(`⏰ Temporal patterns: ${result.temporalAnalysis.patterns.length}`);
  console.log(`🤖 Cognitive optimizations: ${result.cognitiveOptimizations.length}`);

  console.log('\nPerformance Metrics:');
  console.log(`⏱️  Total processing time: ${result.performanceMetrics.totalProcessingTime}ms`);
  console.log(`🧠 Memory learning time: ${result.performanceMetrics.memoryLearningTime}ms`);
  console.log(`⏰ Temporal analysis time: ${result.performanceMetrics.temporalAnalysisTime}ms`);
  console.log(`🤖 Cognitive optimization time: ${result.performanceMetrics.cognitiveOptimizationTime}ms`);
  console.log(`💾 Memory usage: ${(result.performanceMetrics.memoryUsage / 1024 / 1024).toFixed(1)}MB`);
  console.log(`🎯 Cache hit rate: ${(result.performanceMetrics.cacheHitRate * 100).toFixed(1)}%`);
  console.log(`⚡ Optimization score: ${(result.performanceMetrics.optimizationScore * 100).toFixed(1)}%`);

  // Display cognitive insights
  if (result.cognitiveInsights.length > 0) {
    console.log('\n💡 Cognitive Insights:');
    result.cognitiveInsights.forEach(insight => {
      console.log(`   ${insight.type}: ${insight.description}`);
      console.log(`     Confidence: ${(insight.confidence * 100).toFixed(1)}% | Impact: ${insight.impact}`);
    });
  }

  // Shutdown components
  await Promise.all([
    agentDB.shutdown(),
    temporalReasoning.shutdown(),
    cognitiveConsciousness.shutdown()
  ]);
}

/**
 * Performance example: Large-scale XML processing
 */
async function performanceExample() {
  console.log('\n=== Performance Example ===');

  const config: PydanticGeneratorConfig = {
    xmlFilePath: path.join(__dirname, 'sample-data', 'large-example.xml'),
    outputPath: path.join(__dirname, 'output', 'performance-example'),
    batchSize: 2000, // Large batch size for performance
    memoryLimit: 1024 * 1024 * 2048, // 2GB
    enableStreaming: true,
    typeMapping: {
      enableLearning: false, // Disable for pure performance
      strictValidation: false,
      memoryIntegration: false,
      cognitiveMode: false
    },
    schemaGeneration: {
      enableOptimizations: true,
      useCaching: true,
      strictMode: false,
      generateValidators: false, // Skip validators for performance
      includeImports: true,
      performanceMode: true
    },
    validation: {
      strictMode: false,
      enableCustomValidators: false,
      enableCrossParameterValidation: false,
      enableConditionalValidation: false,
      performanceMode: true,
      cacheValidation: false
    }
  };

  const generator = createXmlToPydanticGenerator(config);
  await generator.initialize();

  console.log('Starting large-scale performance test...');

  const startTime = Date.now();

  // Monitor performance
  generator.on('batchCompleted', (event) => {
    console.log(`Batch ${event.batchNumber}/${event.totalBatches} completed (${event.processedCount} parameters)`);
  });

  const result = await generator.generateModels();
  const endTime = Date.now();

  console.log('\nPerformance Results:');
  console.log(`✅ Success: ${result.success}`);
  console.log(`⏱️  Total time: ${endTime - startTime}ms`);
  console.log(`📊 Models generated: ${result.models.length}`);
  console.log(`📊 Parameters processed: ${result.statistics.totalParameters}`);
  console.log(`⚡ Throughput: ${(result.statistics.totalParameters / ((endTime - startTime) / 1000)).toFixed(0)} params/sec`);
  console.log(`💾 Peak memory: ${(result.memoryPeak / 1024 / 1024).toFixed(1)}MB`);

  if (result.processingTime < 1000) {
    console.log('🎯 Performance target achieved: < 1 second processing time');
  } else {
    console.log(`⚠️  Performance target not met: ${result.processingTime}ms (target: < 1000ms)`);
  }
}

/**
 * Custom validation example
 */
async function customValidationExample() {
  console.log('\n=== Custom Validation Example ===');

  const config: PydanticGeneratorConfig = {
    xmlFilePath: path.join(__dirname, 'sample-data', 'validation-example.xml'),
    outputPath: path.join(__dirname, 'output', 'validation-example'),
    batchSize: 500,
    enableStreaming: true,
    validation: {
      strictMode: true,
      enableCustomValidators: true,
      enableCrossParameterValidation: true,
      enableConditionalValidation: true,
      cognitiveMode: false,
      performanceMode: false,
      cacheValidation: true
    }
  };

  const generator = createXmlToPydanticGenerator(config);
  await generator.initialize();

  // Add custom validators
  const validationFramework = generator['validationFramework'];

  // Add custom range validator for frequency bands
  validationFramework.addCustomValidator({
    name: 'frequency_band_validator',
    implementation: (modelData: any) => {
      const frequency = modelData.parameters.find((p: any) => p.name === 'frequency');
      if (frequency && frequency.pythonType === 'int') {
        const freq = parseInt(frequency.defaultValue || 0);
        return freq >= 700 && freq <= 6000; // Typical frequency range
      }
      return true;
    },
    returnType: 'boolean',
    description: 'Validates that frequency is within typical cellular range (700-6000 MHz)'
  });

  // Add custom dependency validator
  validationFramework.addCustomValidator({
    name: 'power_dependency_validator',
    implementation: (modelData: any) => {
      const hasPower = modelData.parameters.some((p: any) => p.name === 'power');
      const hasEnabled = modelData.parameters.some((p: any) => p.name === 'enabled');

      // If power field exists, enabled field should also exist
      if (hasPower && !hasEnabled) {
        return false;
      }
      return true;
    },
    returnType: 'boolean',
    description: 'Validates that power fields have corresponding enabled fields'
  });

  console.log('Starting generation with custom validators...');

  const result = await generator.generateModels();

  console.log('\nCustom Validation Results:');
  console.log(`✅ Success: ${result.success}`);
  console.log(`📊 Models generated: ${result.models.length}`);
  console.log(`🔍 Validation passed: ${result.validationPassed}`);

  if (result.statistics.validationResults) {
    const validationResults = result.statistics.validationResults;
    console.log(`📊 Validations performed: ${validationResults.totalValidations}`);
    console.log(`✅ Passed: ${validationResults.passedValidations}`);
    console.log(`❌ Failed: ${validationResults.failedValidations}`);

    if (validationResults.validationErrors.length > 0) {
      console.log('\nValidation Errors:');
      validationResults.validationErrors.forEach(error => {
        console.log(`   - ${error}`);
      });
    }

    if (validationResults.validationWarnings.length > 0) {
      console.log('\nValidation Warnings:');
      validationResults.validationWarnings.forEach(warning => {
        console.log(`   - ${warning}`);
      });
    }
  }
}

/**
 * Main function to run all examples
 */
async function runAllExamples() {
  try {
    console.log('🚀 Starting Pydantic Model Generation Examples\n');

    // Create output directories
    const outputDir = path.join(__dirname, 'output');
    if (!require('fs').existsSync(outputDir)) {
      require('fs').mkdirSync(outputDir, { recursive: true });
    }

    // Create sample data directories and files if they don't exist
    const sampleDataDir = path.join(__dirname, 'sample-data');
    if (!require('fs').existsSync(sampleDataDir)) {
      require('fs').mkdirSync(sampleDataDir, { recursive: true });
      createSampleXMLFiles(sampleDataDir);
    }

    // Run examples
    await basicGenerationExample();
    await advancedIntegrationExample();
    await performanceExample();
    await customValidationExample();

    console.log('\n✅ All examples completed successfully!');
    console.log('\n📁 Check the output directories for generated files:');
    console.log('   - basic-example/');
    console.log('   - advanced-example/');
    console.log('   - performance-example/');
    console.log('   - validation-example/');

  } catch (error) {
    console.error('❌ Example execution failed:', error);
    process.exit(1);
  }
}

/**
 * Create sample XML files for demonstration
 */
function createSampleXMLFiles(outputDir: string) {
  const fs = require('fs');

  // Basic example XML
  const basicXML = `<?xml version="1.0" encoding="UTF-8"?>
<model>
  <moClass name="CellConfiguration">
    <parameter name="cellId" type="integer" description="Cell identifier">
      <constraint type="range" min="1" max="65535"/>
    </parameter>
    <parameter name="cellName" type="string" description="Cell name">
      <constraint type="length" minLength="1" maxLength="50"/>
    </parameter>
    <parameter name="power" type="decimal" description="Cell power in dBm">
      <constraint type="range" min="-30" max="60"/>
    </parameter>
    <parameter name="isActive" type="boolean" description="Cell active status"/>
  </moClass>
  <moClass name="SectorConfiguration">
    <parameter name="sectorId" type="integer" description="Sector identifier"/>
    <parameter name="azimuth" type="integer" description="Sector azimuth in degrees">
      <constraint type="range" min="0" max="359"/>
    </parameter>
    <parameter name="frequency" type="decimal" description="Sector frequency in MHz"/>
  </moClass>
</model>`;

  // Advanced example XML with complex constraints
  const advancedXML = `<?xml version="1.0" encoding="UTF-8"?>
<model>
  <moClass name="AdvancedCell">
    <parameter name="vsData1" type="vsData1" description="Cell identifier"/>
    <parameter name="vsData2a" type="vsData2a" description="Cell name"/>
    <parameter name="vsData3c" type="vsData3c" description="Cell power"/>
    <parameter name="vsData4d" type="vsData4d" description="Cell status"/>
    <parameter name="createdAt" type="dateTime" description="Creation timestamp"/>
    <parameter name="features" type="list" description="Feature list"/>
    <parameter name="metadata" type="object" description="Additional metadata"/>
  </moClass>
  <moClass name="NetworkElement">
    <parameter name="elementId" type="string" description="Element identifier">
      <constraint type="pattern" value="^[A-Z]{3}-\\d{3}$"/>
    </parameter>
    <parameter name="administrativeState" type="string" description="Admin state">
      <constraint type="enum" value="UNLOCKED,LOCKED,SHUTTING"/>
    </parameter>
    <parameter name="operationalState" type="string" description="Operational state">
      <constraint type="enum" value="ENABLED,DISABLED,DEGRADED"/>
    </parameter>
  </moClass>
</model>`;

  // Large example XML (simplified for demo)
  let largeXML = '<?xml version="1.0" encoding="UTF-8"?><model>';
  for (let i = 0; i < 100; i++) {
    largeXML += `
  <moClass name="TestClass${i}">
    <parameter name="param${i}_id" type="integer" description="Parameter ID"/>
    <parameter name="param${i}_name" type="string" description="Parameter name"/>
    <parameter name="param${i}_value" type="decimal" description="Parameter value"/>
    <parameter name="param${i}_active" type="boolean" description="Parameter active flag"/>
  </moClass>`;
  }
  largeXML += '</model>';

  // Validation example XML
  const validationXML = `<?xml version="1.0" encoding="UTF-8"?>
<model>
  <moClass name="ValidationTest">
    <parameter name="frequency" type="integer" description="Frequency in MHz"/>
    <parameter name="power" type="decimal" description="Power in dBm"/>
    <parameter name="enabled" type="boolean" description="Enable flag"/>
    <parameter name="band" type="string" description="Frequency band">
      <constraint type="enum" value="700,850,900,1800,1900,2100,2600,3500"/>
    </parameter>
  </moClass>
</model>`;

  fs.writeFileSync(path.join(outputDir, 'basic-example.xml'), basicXML);
  fs.writeFileSync(path.join(outputDir, 'advanced-example.xml'), advancedXML);
  fs.writeFileSync(path.join(outputDir, 'large-example.xml'), largeXML);
  fs.writeFileSync(path.join(outputDir, 'validation-example.xml'), validationXML);

  console.log('📄 Sample XML files created in sample-data/ directory');
}

// Run examples if this file is executed directly
if (require.main === module) {
  runAllExamples().catch(console.error);
}

export {
  basicGenerationExample,
  advancedIntegrationExample,
  performanceExample,
  customValidationExample,
  runAllExamples
};