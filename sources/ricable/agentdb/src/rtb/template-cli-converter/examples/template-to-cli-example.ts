/**
 * Template-to-CLI Converter Example
 *
 * Comprehensive example demonstrating the template-to-CLI conversion system
 * with RTB template inheritance, cognitive optimization, and Ericsson RAN expertise.
 */

import {
  TemplateToCliConverter,
  createRanOptimizedConverter,
  DEFAULT_CONFIG,
  SAFE_CONFIG
} from '../index';

import { RtbTemplateIntegration } from '../integration';

import {
  IntegratedTemplateSystem
} from '../../hierarchical-template-system';

import {
  RTBTemplate,
  TemplateMeta,
  MOHierarchy,
  ReservedByHierarchy
} from '../../../types/rtb-types';

import {
  TemplateToCliContext,
  CliCommandSet
} from '../types';

/**
 * Example RTB template for LTE cell configuration
 */
const EXAMPLE_LTE_TEMPLATE: RTBTemplate = {
  meta: {
    version: '1.0.0',
    author: ['RAN Automation Team'],
    description: 'LTE cell optimization template',
    tags: ['lte', 'cell', 'optimization'],
    environment: 'production',
    priority: 50,
    inherits_from: 'base_lte_template'
  },
  configuration: {
    'EUtranCellFDD.qRxLevMin': -130,
    'EUtranCellFDD.qQualMin': -32,
    'EUtranCellFDD.cellIndividualOffset': 0,
    'EUtranCellFDD.referenceSignalPower': 15,
    'EUtranCellFDD.antennaPortsCount': 2,
    'EUtranCellFDD.prachRootSequenceIndex': 0,
    'EUtranCellFDD.prachConfigIndex': 0,
    'ENBFunction.endcEnabled': true,
    'ENBFunction.splitBearerSupport': true
  },
  conditions: {
    'if_enable_endc': {
      if: 'ENBFunction.endcEnabled == true',
      then: {
        'ENBFunction.nrEventB1Threshold': -110,
        'ENBFunction.nrEventB1Hysteresis': 2,
        'ENBFunction.nrEventB1TimeToTrigger': 640
      },
      else: 'skip_nr_configuration'
    },
    'if_high_traffic': {
      if: 'traffic_load > 0.8',
      then: {
        'EUtranCellFDD.ul256qamEnabled': true,
        'EUtranCellFDD.dl256qamEnabled': true
      },
      else: {
        'EUtranCellFDD.ul256qamEnabled': false,
        'EUtranCellFDD.dl256qamEnabled': false
      }
    }
  },
  evaluations: {
    'calculate_optimal_power': {
      eval: 'optimize_reference_signal_power',
      args: ['coverage_target', 'interference_constraint']
    },
    'validate_configuration': {
      eval: 'validate_cell_parameters',
      args: ['EUtranCellFDD']
    }
  }
};

/**
 * Example MO hierarchy (simplified)
 */
const EXAMPLE_MO_HIERARCHY: MOHierarchy = {
  rootClass: 'ManagedElement',
  classes: new Map([
    ['ManagedElement', {
      id: 'ManagedElement',
      name: 'ManagedElement',
      parentClass: '',
      cardinality: { minimum: 1, maximum: 1, type: 'single' },
      flags: {},
      children: ['ENBFunction', 'EUtranCellFDD'],
      attributes: ['userLabel', 'neType'],
      derivedClasses: []
    }],
    ['ENBFunction', {
      id: 'ENBFunction',
      name: 'ENBFunction',
      parentClass: 'ManagedElement',
      cardinality: { minimum: 1, maximum: 1, type: 'single' },
      flags: {},
      children: ['EUtranCellFDD'],
      attributes: ['endcEnabled', 'splitBearerSupport'],
      derivedClasses: []
    }],
    ['EUtranCellFDD', {
      id: 'EUtranCellFDD',
      name: 'EUtranCellFDD',
      parentClass: 'ENBFunction',
      cardinality: { minimum: 1, maximum: 256, type: 'bounded' },
      flags: {},
      children: [],
      attributes: ['qRxLevMin', 'qQualMin', 'referenceSignalPower'],
      derivedClasses: []
    }]
  ]),
  relationships: new Map(),
  cardinality: new Map(),
  inheritanceChain: new Map()
};

/**
 * Example context for conversion
 */
const EXAMPLE_CONTEXT: TemplateToCliContext = {
  target: {
    nodeId: 'LTE_NODE_001'
  },
  cellIds: {
    primaryCell: 'CELL_001',
    lteCell: 'CELL_001',
    nrCell: 'NRCELL_001'
  },
  options: {
    preview: false,
    force: false,
    verbose: true,
    timeout: 60,
    batchMode: true,
    dependencyAnalysis: true,
    cognitiveOptimization: true,
    generateRollback: true,
    generateValidation: true
  },
  parameters: {
    traffic_load: 0.9,
    coverage_target: 'excellent',
    interference_constraint: 'low'
  },
  moHierarchy: EXAMPLE_MO_HIERARCHY,
  reservedBy: {} as ReservedByHierarchy
};

/**
 * Example 1: Basic template-to-CLI conversion
 */
export async function example1_BasicConversion(): Promise<void> {
  console.log('\n=== Example 1: Basic Template-to-CLI Conversion ===');

  try {
    // Create converter with default configuration
    const converter = createTemplateToCliConverter();

    console.log('Converting LTE template to CLI commands...');
    const startTime = Date.now();

    // Convert template
    const commandSet = await converter.convertTemplate(EXAMPLE_LTE_TEMPLATE, EXAMPLE_CONTEXT);

    const duration = Date.now() - startTime;

    // Display results
    console.log(`\n✅ Conversion completed in ${duration}ms`);
    console.log(`📊 Generated ${commandSet.commands.length} CLI commands`);
    console.log(`🎯 Command set ID: ${commandSet.id}`);
    console.log(`⚡ Complexity: ${commandSet.metadata.complexity}`);
    console.log(`⚠️  Risk level: ${commandSet.metadata.riskLevel}`);

    console.log('\n📋 Generated commands:');
    for (const command of commandSet.commands.slice(0, 5)) { // Show first 5 commands
      console.log(`  ${command.type}: ${command.command}`);
      console.log(`    Description: ${command.description}`);
      if (command.cognitive) {
        console.log(`    Cognitive level: ${command.cognitive.optimizationLevel}`);
        console.log(`    Confidence: ${command.cognitive.confidence}`);
      }
      console.log('');
    }

    if (commandSet.commands.length > 5) {
      console.log(`  ... and ${commandSet.commands.length - 5} more commands`);
    }

  } catch (error) {
    console.error('❌ Basic conversion failed:', error);
  }
}

/**
 * Example 2: Integrated conversion with inheritance
 */
export async function example2_IntegratedConversion(): Promise<void> {
  console.log('\n=== Example 2: Integrated Conversion with Template Inheritance ===');

  try {
    // Create template system
    const templateSystem = new IntegratedTemplateSystem();

    // Create integration
    const integration = new RtbTemplateIntegration(templateSystem, {
      enableInheritanceProcessing: true,
      enablePriorityOptimization: true,
      enableTemplateValidation: true,
      enableMergeResultAnalysis: true
    });

    console.log('Processing template with inheritance...');
    const startTime = Date.now();

    // Convert with inheritance processing
    const result = await integration.convertTemplateWithInheritance(
      EXAMPLE_LTE_TEMPLATE,
      EXAMPLE_CONTEXT,
      {
        optimizeForPriority: true,
        skipValidation: false
      }
    );

    const duration = Date.now() - startTime;

    // Display results
    console.log(`\n✅ Integrated conversion completed in ${duration}ms`);
    console.log(`📊 Template: ${result.originalTemplate.id}`);
    console.log(`🎯 Priority: ${result.originalTemplate.priority}`);
    console.log(`📈 Inheritance depth: ${result.integrationStats.templateComplexity.inheritanceDepth}`);

    if (result.mergeResult) {
      console.log(`🔗 Merge conflicts resolved: ${result.mergeResult.resolvedConflicts.length}`);
    }

    console.log(`⚡ Generated ${result.commandSet.commands.length} CLI commands`);

    console.log('\n💡 Integration insights:');
    for (const insight of result.insights.slice(0, 3)) {
      console.log(`  [${insight.type.toUpperCase()}] ${insight.description}`);
      console.log(`    Recommendation: ${insight.recommendation}`);
      console.log('');
    }

  } catch (error) {
    console.error('❌ Integrated conversion failed:', error);
  }
}

/**
 * Example 3: Batch conversion of multiple templates
 */
export async function example3_BatchConversion(): Promise<void> {
  console.log('\n=== Example 3: Batch Conversion of Multiple Templates ===');

  try {
    // Create multiple templates
    const templates = [
      EXAMPLE_LTE_TEMPLATE,
      {
        ...EXAMPLE_LTE_TEMPLATE,
        meta: {
          ...EXAMPLE_LTE_TEMPLATE.meta,
          description: 'LTE cell template - Urban Dense',
          priority: 60
        },
        configuration: {
          ...EXAMPLE_LTE_TEMPLATE.configuration,
          'EUtranCellFDD.referenceSignalPower': 18, // Higher power for dense urban
          'EUtranCellFDD.antennaPortsCount': 4 // 4T4R MIMO
        }
      },
      {
        ...EXAMPLE_LTE_TEMPLATE,
        meta: {
          ...EXAMPLE_LTE_TEMPLATE.meta,
          description: 'LTE cell template - Rural',
          priority: 30
        },
        configuration: {
          ...EXAMPLE_LTE_TEMPLATE.configuration,
          'EUtranCellFDD.referenceSignalPower': 12, // Lower power for rural
          'EUtranCellFDD.antennaPortsCount': 2 // 2T2R MIMO
        }
      }
    ];

    // Create integration
    const templateSystem = new IntegratedTemplateSystem();
    const integration = new RtbTemplateIntegration(templateSystem);

    console.log(`Converting ${templates.length} templates in batch...`);
    const startTime = Date.now();

    // Convert in batch with parallel processing
    const results = await integration.convertTemplatesBatch(
      templates,
      EXAMPLE_CONTEXT,
      {
        processInParallel: true,
        maxConcurrency: 3,
        continueOnError: true
      }
    );

    const duration = Date.now() - startTime;

    // Display results
    console.log(`\n✅ Batch conversion completed in ${duration}ms`);
    console.log(`📊 Successfully processed: ${results.length}/${templates.length} templates`);

    for (const result of results) {
      console.log(`\n📋 Template: ${result.originalTemplate.description}`);
      console.log(`   Commands: ${result.commandSet.commands.length}`);
      console.log(`   Complexity: ${result.commandSet.metadata.complexity}`);
      console.log(`   Processing time: ${result.integrationStats.totalIntegrationTime}ms`);
    }

    // Get conversion statistics
    const stats = integration.getConversionStatistics();
    console.log(`\n📈 Conversion Statistics:`);
    console.log(`   Total conversions: ${stats.totalConversions}`);
    console.log(`   Average processing time: ${stats.averageProcessingTime.toFixed(2)}ms`);
    console.log(`   Average command count: ${stats.averageCommandCount.toFixed(1)}`);
    console.log(`   Complexity distribution:`, stats.complexityDistribution);

  } catch (error) {
    console.error('❌ Batch conversion failed:', error);
  }
}

/**
 * Example 4: Safe conversion with preview and rollback
 */
export async function example4_SafeConversion(): Promise<void> {
  console.log('\n=== Example 4: Safe Conversion with Preview and Rollback ===');

  try {
    // Create converter with safe configuration
    const converter = new TemplateToCliConverter(SAFE_CONFIG);

    // Create context with preview mode
    const safeContext: TemplateToCliContext = {
      ...EXAMPLE_CONTEXT,
      options: {
        ...EXAMPLE_CONTEXT.options,
        preview: true, // Enable preview mode
        generateRollback: true,
        generateValidation: true
      }
    };

    console.log('Converting template with safety measures...');
    const startTime = Date.now();

    // Convert with safety
    const commandSet = await converter.convertTemplate(EXAMPLE_LTE_TEMPLATE, safeContext);

    const duration = Date.now() - startTime;

    // Display results
    console.log(`\n✅ Safe conversion completed in ${duration}ms`);
    console.log(`📊 Generated ${commandSet.commands.length} commands`);
    console.log(`🛡️  Rollback commands: ${commandSet.rollbackCommands.length}`);
    console.log(`✅ Validation commands: ${commandSet.validationCommands.length}`);

    console.log('\n🔍 Preview commands (first 3):');
    for (const command of commandSet.commands.slice(0, 3)) {
      console.log(`  ${command.command} --preview`);
    }

    console.log('\n🔄 Rollback commands (first 3):');
    for (const rollbackCommand of commandSet.rollbackCommands.slice(0, 3)) {
      console.log(`  ${rollbackCommand.command}`);
    }

    console.log('\n✅ Validation commands (first 3):');
    for (const validationCommand of commandSet.validationCommands.slice(0, 3)) {
      console.log(`  ${validationCommand.command}`);
    }

  } catch (error) {
    console.error('❌ Safe conversion failed:', error);
  }
}

/**
 * Example 5: Performance-optimized conversion for large-scale deployment
 */
export async function example5_PerformanceOptimized(): Promise<void> {
  console.log('\n=== Example 5: Performance-Optimized Large-Scale Conversion ===');

  try {
    // Create converter with high-performance configuration
    const converter = new TemplateToCliConverter({
      maxCommandsPerBatch: 100,
      enableCognitiveOptimization: false, // Disable for performance
      enableDependencyAnalysis: true,
      performanceOptimization: {
        enableParallelExecution: true,
        maxParallelCommands: 20,
        enableBatching: true,
        batchSize: 50
      }
    });

    // Create a large template (simulating many cells)
    const largeTemplate = createLargeCellTemplate(50); // 50 cells

    console.log(`Converting large template with ${Object.keys(largeTemplate.configuration).length} parameters...`);
    const startTime = Date.now();

    // Convert with performance optimization
    const commandSet = await converter.convertTemplate(largeTemplate, EXAMPLE_CONTEXT);

    const duration = Date.now() - startTime;

    // Display results
    console.log(`\n✅ Performance-optimized conversion completed in ${duration}ms`);
    console.log(`📊 Generated ${commandSet.commands.length} CLI commands`);
    console.log(`⚡ Processing rate: ${(Object.keys(largeTemplate.configuration).length / (duration / 1000)).toFixed(1)} params/sec`);

    console.log('\n📈 Performance statistics:');
    console.log(`   Command generation time: ${commandSet.stats.commandGenerationTime}ms`);
    console.log(`   Dependency analysis time: ${commandSet.stats.dependencyAnalysisTime}ms`);
    console.log(`   Validation time: ${commandSet.stats.validationTime}ms`);
    console.log(`   Total conversion time: ${commandSet.stats.totalConversionTime}ms`);

    // Analyze command distribution
    const commandTypeDistribution: Record<string, number> = {};
    for (const command of commandSet.commands) {
      commandTypeDistribution[command.type] = (commandTypeDistribution[command.type] || 0) + 1;
    }

    console.log('\n📊 Command type distribution:');
    for (const [type, count] of Object.entries(commandTypeDistribution)) {
      console.log(`   ${type}: ${count} commands`);
    }

  } catch (error) {
    console.error('❌ Performance-optimized conversion failed:', error);
  }
}

/**
 * Create a large template with many cell configurations
 */
function createLargeCellTemplate(cellCount: number): RTBTemplate {
  const configuration: Record<string, any> = {};

  for (let i = 1; i <= cellCount; i++) {
    configuration[`EUtranCellFDD=CELL_${i.toString().padStart(3, '0')}.qRxLevMin`] = -130 + (i % 5);
    configuration[`EUtranCellFDD=CELL_${i.toString().padStart(3, '0')}.qQualMin`] = -32 + (i % 3);
    configuration[`EUtranCellFDD=CELL_${i.toString().padStart(3, '0')}.referenceSignalPower`] = 15 + (i % 10);
    configuration[`EUtranCellFDD=CELL_${i.toString().padStart(3, '0')}.cellIndividualOffset`] = (i % 5) - 2;
  }

  return {
    meta: {
      version: '1.0.0',
      author: ['RAN Automation Team'],
      description: `Large scale LTE template with ${cellCount} cells`,
      tags: ['lte', 'large-scale', 'optimization'],
      environment: 'production',
      priority: 40
    },
    configuration,
    conditions: {},
    evaluations: {}
  };
}

/**
 * Run all examples
 */
export async function runAllExamples(): Promise<void> {
  console.log('🚀 Starting Template-to-CLI Converter Examples...\n');

  await example1_BasicConversion();
  await example2_IntegratedConversion();
  await example3_BatchConversion();
  await example4_SafeConversion();
  await example5_PerformanceOptimized();

  console.log('\n✅ All examples completed successfully!');
}

// Export examples for individual execution
export {
  example1_BasicConversion,
  example2_IntegratedConversion,
  example3_BatchConversion,
  example4_SafeConversion,
  example5_PerformanceOptimized,
  runAllExamples
};