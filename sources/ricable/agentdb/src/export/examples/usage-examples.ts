/**
 * Phase 5: Type-Safe Template Export System - Usage Examples
 *
 * Comprehensive examples demonstrating all features of the export system
 * including Pydantic schema generation, validation, variant generation, and cognitive optimization.
 */

import {
  createExportSystem,
  quickExport,
  batchExport,
  validateTemplate,
  generateVariants,
  TemplateExporter,
  VariantGenerator,
  ExportValidator,
  MetadataGenerator,
  PriorityTemplate,
  TemplatePriority,
  TemplateVariantType
} from '../index';

// Example template data
const exampleTemplate: PriorityTemplate = {
  meta: {
    templateId: 'lte_cell_optimization',
    version: '1.0.0',
    author: ['RAN Automation Team'],
    description: 'LTE cell optimization template for capacity and coverage enhancement',
    tags: ['lte', 'optimization', 'capacity', 'coverage'],
    environment: 'production',
    priority: TemplatePriority.BASE,
    templateId: 'lte_cell_optimization',
    version: '1.0.0',
    author: ['RAN Automation Team'],
    description: 'LTE cell optimization template for capacity and coverage enhancement',
    tags: ['lte', 'optimization', 'capacity', 'coverage']
  },
  priority: TemplatePriority.BASE,
  configuration: {
    cellId: 'LTE_001',
    pci: 1,
    earfcn: 1800,
    bandwidth: 20,
    qRxLevMin: -140,
    qQualMin: -34,
    cellIndividualOffset: 0,
    p0NominalPusch: -80,
    alpha: 0.8,
    maxTxPower: 43,
    antennaPorts: 2,
    prachConfigIndex: 103,
    rootSequenceIndex: 0,
    zeroCorrelationZoneConfig: 10,
    ncs: false,
    rachConfigCommon: {
      preambleInfo: {
        numberOfRA_Preambles: 64,
        preamblesGroupAConfig: {
          sizeOfRA_PreamblesGroupA: 48,
            numberOfRA_PreamblesGroupA: 48
        }
      }
    }
  },
  custom: [
    {
      name: 'calculateCapacity',
      args: ['bandwidth', 'users', 'traffic'],
      body: [
        'return bandwidth * users * traffic_factor',
        'where traffic_factor = 0.7 for urban, 0.5 for rural'
      ]
    }
  ],
  conditions: {
    high_traffic: {
      if: 'users > 1000',
      then: { capacityMode: 'high', loadBalancing: true },
      else: { capacityMode: 'normal' }
    }
  },
  evaluations: {
    dynamic_power: {
      eval: 'calculateOptimalPower(traffic, interference)',
      args: ['current_traffic', 'measured_interference']
    }
  }
};

/**
 * Example 1: Quick Export - Simple template export with default settings
 */
export async function example1_QuickExport() {
  console.log('🚀 Example 1: Quick Export');

  try {
    const result = await quickExport(
      exampleTemplate,
      './exports/quick',
      'json'
    );

    console.log('✅ Quick export completed:');
    console.log(`   Output: ${result.outputPath}`);
    console.log(`   File size: ${result.fileSize} bytes`);
    console.log(`   Checksum: ${result.checksum}`);
    console.log(`   Validation: ${result.validationResults.isValid ? 'PASSED' : 'FAILED'}`);
    console.log(`   Processing time: ${result.performanceMetrics.templateProcessingTime}ms`);

  } catch (error) {
    console.error('❌ Quick export failed:', error);
  }
}

/**
 * Example 2: Advanced Export - Custom configuration with all features
 */
export async function example2_AdvancedExport() {
  console.log('🚀 Example 2: Advanced Export with Custom Configuration');

  try {
    // Create exporter with custom configuration
    const exporter = createExportSystem({
      defaultExportConfig: {
        outputFormat: 'pydantic',
        includeMetadata: true,
        includeValidation: true,
        includeDocumentation: true,
        outputDirectory: './exports/advanced',
        compressionLevel: 'gzip',
        batchProcessing: true,
        parallelExecution: true,
        maxConcurrency: 4
      },
      cognitiveConfig: {
        level: 'maximum',
        temporalExpansion: 1000,
        strangeLoopOptimization: true,
        autonomousAdaptation: true
      },
      performanceMonitoring: true
    });

    await exporter.initialize();

    // Export with custom configuration
    const exportConfig = {
      outputFormat: 'pydantic' as const,
      includeMetadata: true,
      includeValidation: true,
      includeDocumentation: true,
      outputDirectory: './exports/advanced',
      filenameTemplate: '{templateId}_{variant}_{timestamp}.{format}'
    };

    const result = await exporter.exportTemplate(exampleTemplate, exportConfig);

    console.log('✅ Advanced export completed:');
    console.log(`   Output: ${result.outputPath}`);
    console.log(`   Format: ${result.outputFormat}`);
    console.log(`   Validation errors: ${result.validationResults.errors.length}`);
    console.log(`   Validation warnings: ${result.validationResults.warnings.length}`);
    console.log(`   Processing time: ${result.performanceMetrics.templateProcessingTime}ms`);

    // Get export status
    const status = await exporter.getExportStatus();
    console.log(`   System status: ${status.isActive ? 'ACTIVE' : 'INACTIVE'}`);
    console.log(`   Active jobs: ${status.activeJobs}`);

    await exporter.shutdown();

  } catch (error) {
    console.error('❌ Advanced export failed:', error);
  }
}

/**
 * Example 3: Batch Export - Multiple templates in parallel
 */
export async function example3_BatchExport() {
  console.log('🚀 Example 3: Batch Export of Multiple Templates');

  try {
    // Create multiple template variants
    const templates = [
      exampleTemplate,
      {
        ...exampleTemplate,
        meta: { ...exampleTemplate.meta, templateId: 'lte_cell_optimization_urban' },
        configuration: { ...exampleTemplate.configuration, capacityMode: 'high' }
      },
      {
        ...exampleTemplate,
        meta: { ...exampleTemplate.meta, templateId: 'lte_cell_optimization_rural' },
        configuration: { ...exampleTemplate.configuration, capacityMode: 'low' }
      }
    ];

    const results = await batchExport(templates, './exports/batch', 'json');

    console.log('✅ Batch export completed:');
    results.forEach((result, index) => {
      console.log(`   Template ${index + 1}: ${result.templateId}`);
      console.log(`     Output: ${result.outputPath}`);
      console.log(`     Valid: ${result.validationResults.isValid ? 'YES' : 'NO'}`);
      console.log(`     Time: ${result.performanceMetrics.templateProcessingTime}ms`);
    });

    const totalTime = results.reduce((sum, r) => sum + r.performanceMetrics.templateProcessingTime, 0);
    console.log(`   Total processing time: ${totalTime}ms`);
    console.log(`   Average time per template: ${Math.round(totalTime / results.length)}ms`);

  } catch (error) {
    console.error('❌ Batch export failed:', error);
  }
}

/**
 * Example 4: Template Validation - Standalone validation with detailed reporting
 */
export async function example4_TemplateValidation() {
  console.log('🚀 Example 4: Comprehensive Template Validation');

  try {
    // Validate with strict mode
    const strictValidation = await validateTemplate(exampleTemplate, true);
    console.log('✅ Strict validation results:');
    console.log(`   Valid: ${strictValidation.isValid}`);
    console.log(`   Score: ${strictValidation.validationScore.toFixed(3)}`);
    console.log(`   Errors: ${strictValidation.errors.length}`);
    console.log(`   Warnings: ${strictValidation.warnings.length}`);
    console.log(`   Processing time: ${strictValidation.processingTime}ms`);

    if (strictValidation.errors.length > 0) {
      console.log('   Errors:');
      strictValidation.errors.forEach(error => {
        console.log(`     - ${error.code}: ${error.message}`);
        if (error.suggestion) {
          console.log(`       Suggestion: ${error.suggestion}`);
        }
      });
    }

    // Validate with lenient mode
    const lenientValidation = await validateTemplate(exampleTemplate, false);
    console.log('✅ Lenient validation results:');
    console.log(`   Valid: ${lenientValidation.isValid}`);
    console.log(`   Score: ${lenientValidation.validationScore.toFixed(3)}`);
    console.log(`   Auto-fixes available: ${lenientValidation.autoFixes.length}`);

    if (lenientValidation.autoFixes.length > 0) {
      console.log('   Available auto-fixes:');
      lenientValidation.autoFixes.forEach(fix => {
        console.log(`     - ${fix.type}: ${fix.code} (confidence: ${(fix.confidence * 100).toFixed(1)}%)`);
      });
    }

  } catch (error) {
    console.error('❌ Template validation failed:', error);
  }
}

/**
 * Example 5: Variant Generation - Generate all template variants
 */
export async function example5_VariantGeneration() {
  console.log('🚀 Example 5: Template Variant Generation');

  try {
    // Generate all variants
    const allVariantsResult = await generateVariants(exampleTemplate);
    console.log('✅ All variants generated:');
    console.log(`   Total variants: ${allVariantsResult.generatedVariants.length}`);
    console.log(`   Success rate: ${(allVariantsResult.successRate * 100).toFixed(1)}%`);
    console.log(`   Generation time: ${allVariantsResult.generationTime}ms`);
    console.log(`   Errors: ${allVariantsResult.errors.length}`);

    allVariantsResult.generatedVariants.forEach(variant => {
      console.log(`   - ${variant.meta.templateId} (${variant.meta.variantType})`);
      console.log(`     Priority: ${variant.priority}`);
      console.log(`     Parameters: ${Object.keys(variant.configuration).length}`);
    });

    // Generate specific variants
    const specificVariantsResult = await generateVariants(exampleTemplate, [
      TemplateVariantType.URBAN,
      TemplateVariantType.HIGH_MOBILITY,
      TemplateVariantType.SLEEP_MODE
    ]);

    console.log('✅ Specific variants generated:');
    specificVariantsResult.generatedVariants.forEach(variant => {
      console.log(`   - ${variant.meta.templateId}`);
    });

    if (allVariantsResult.cognitiveInsights) {
      console.log('🧠 Cognitive insights:');
      console.log(`   Consciousness level: ${(allVariantsResult.cognitiveInsights.consciousnessLevel * 100).toFixed(1)}%`);
      console.log(`   Strange-loop optimizations: ${allVariantsResult.cognitiveInsights.strangeLoopOptimizations.length}`);
    }

  } catch (error) {
    console.error('❌ Variant generation failed:', error);
  }
}

/**
 * Example 6: Cognitive Optimization - Template export with cognitive consciousness
 */
export async function example6_CognitiveOptimization() {
  console.log('🚀 Example 6: Cognitive Optimization');

  try {
    // Create exporter with maximum cognitive optimization
    const exporter = createExportSystem({
      cognitiveConfig: {
        level: 'maximum',
        temporalExpansion: 1000,
        strangeLoopOptimization: true,
        autonomousAdaptation: true
      },
      performanceMonitoring: true,
      parallelProcessing: true,
      maxConcurrency: 4
    });

    await exporter.initialize();

    // Export template with cognitive optimization
    const result = await exporter.exportTemplate(exampleTemplate, {
      outputFormat: 'json',
      includeMetadata: true,
      includeValidation: true,
      includeDocumentation: true,
      outputDirectory: './exports/cognitive'
    });

    console.log('✅ Cognitive optimization completed:');
    console.log(`   Output: ${result.outputPath}`);
    console.log(`   Processing time: ${result.performanceMetrics.templateProcessingTime}ms`);

    // Get detailed performance metrics
    const status = await exporter.getExportStatus();
    if (status.cognitive) {
      console.log('🧠 Cognitive status:');
      console.log(`   Consciousness level: ${(status.cognitive.consciousnessLevel * 100).toFixed(1)}%`);
      console.log(`   Evolution score: ${status.cognitive.evolutionScore.toFixed(3)}`);
      console.log(`   Strange-loop iterations: ${status.cognitive.strangeLoopIteration}`);
      console.log(`   Active patterns: ${status.cognitive.activeStrangeLoops.join(', ')}`);
    }

    await exporter.shutdown();

  } catch (error) {
    console.error('❌ Cognitive optimization failed:', error);
  }
}

/**
 * Example 7: Performance Monitoring - Real-time performance analysis
 */
export async function example7_PerformanceMonitoring() {
  console.log('🚀 Example 7: Performance Monitoring');

  try {
    const exporter = createExportSystem({
      performanceMonitoring: true,
      cacheConfig: {
        enabled: true,
        maxSize: 100,
        ttl: 60000, // 1 minute
        evictionPolicy: 'lru',
        compressionEnabled: true,
        compressionLevel: 6,
        keyPrefix: 'perf_test_'
      }
    });

    await exporter.initialize();

    // Export multiple times to test performance
    const exportPromises = Array(10).fill(null).map((_, index) =>
      exporter.exportTemplate({
        ...exampleTemplate,
        meta: { ...exampleTemplate.meta, templateId: `perf_test_${index}` }
      })
    );

    const results = await Promise.all(exportPromises);

    console.log('✅ Performance test completed:');
    const totalTime = results.reduce((sum, r) => sum + r.performanceMetrics.templateProcessingTime, 0);
    console.log(`   Total exports: ${results.length}`);
    console.log(`   Total time: ${totalTime}ms`);
    console.log(`   Average time: ${Math.round(totalTime / results.length)}ms`);
    console.log(`   Fastest: ${Math.min(...results.map(r => r.performanceMetrics.templateProcessingTime))}ms`);
    console.log(`   Slowest: ${Math.max(...results.map(r => r.performanceMetrics.templateProcessingTime))}ms`);

    // Get system statistics
    const stats = await exporter.getExportStatus();
    console.log('📊 System statistics:');
    console.log(`   Cache hit rate: ${(stats.cache.hitRate * 100).toFixed(1)}%`);
    console.log(`   Memory usage: ${(stats.performance.memoryUsage.averageMemoryUsage / 1024 / 1024).toFixed(1)}MB`);
    console.log(`   System load: ${stats.systemLoad.toFixed(2)}`);

    await exporter.shutdown();

  } catch (error) {
    console.error('❌ Performance monitoring failed:', error);
  }
}

/**
 * Example 8: Error Recovery - Auto-fix and error handling
 */
export async function example8_ErrorRecovery() {
  console.log('🚀 Example 8: Error Recovery and Auto-fix');

  try {
    // Create template with intentional errors
    const invalidTemplate = {
      ...exampleTemplate,
      meta: {
        ...exampleTemplate.meta,
        templateId: 'invalid-template-name!', // Invalid characters
        version: '1.0' // Invalid version format
      },
      configuration: {
        ...exampleTemplate.configuration,
        invalidField: null // Null value
      }
    };

    const validator = new ExportValidator({
      strictMode: true,
      enableLearning: true,
      enableAutoFix: true,
      maxAutoFixes: 5,
      validationTimeout: 5000,
      memoryThreshold: 512 * 1024 * 1024,
      enableCognitiveOptimization: true,
      agentdbIntegration: false,
      realTimeValidation: false
    });

    await validator.initialize();

    // Validate invalid template
    const result = await validator.validateTemplateExport(invalidTemplate);

    console.log('✅ Error recovery analysis:');
    console.log(`   Valid: ${result.isValid}`);
    console.log(`   Score: ${result.validationScore.toFixed(3)}`);
    console.log(`   Errors: ${result.errors.length}`);
    console.log(`   Auto-fixes available: ${result.autoFixes.length}`);

    if (result.errors.length > 0) {
      console.log('   Errors detected:');
      result.errors.forEach(error => {
        console.log(`     - ${error.code}: ${error.message}`);
        console.log(`       Field: ${error.field}`);
        console.log(`       Fixable: ${error.fixable ? 'YES' : 'NO'}`);
        if (error.autoFix) {
          console.log(`       Auto-fix: ${error.autoFix.type} (${(error.autoFix.confidence * 100).toFixed(1)}% confidence)`);
        }
      });
    }

    // Apply auto-fixes
    if (result.autoFixes.length > 0) {
      console.log('🔧 Applying auto-fixes...');
      const fixedTemplate = await validator.applyAutoFixes(invalidTemplate, result.autoFixes);
      console.log(`   Applied ${result.appliedFixes.length} fixes`);

      // Re-validate fixed template
      const fixedResult = await validator.validateTemplateExport(fixedTemplate);
      console.log(`   Fixed template valid: ${fixedResult.isValid}`);
      console.log(`   Fixed template score: ${fixedResult.validationScore.toFixed(3)}`);
    }

    await validator.shutdown();

  } catch (error) {
    console.error('❌ Error recovery failed:', error);
  }
}

/**
 * Run all examples
 */
export async function runAllExamples() {
  console.log('🚀 Running Phase 5 Export System Examples\n');

  const examples = [
    example1_QuickExport,
    example2_AdvancedExport,
    example3_BatchExport,
    example4_TemplateValidation,
    example5_VariantGeneration,
    example6_CognitiveOptimization,
    example7_PerformanceMonitoring,
    example8_ErrorRecovery
  ];

  for (const example of examples) {
    try {
      await example();
      console.log('\n' + '='.repeat(60) + '\n');
    } catch (error) {
      console.error(`Example failed:`, error);
      console.log('\n' + '='.repeat(60) + '\n');
    }
  }
}

// Export examples for individual testing
export {
  example1_QuickExport,
  example2_AdvancedExport,
  example3_BatchExport,
  example4_TemplateValidation,
  example5_VariantGeneration,
  example6_CognitiveOptimization,
  example7_PerformanceMonitoring,
  example8_ErrorRecovery
};