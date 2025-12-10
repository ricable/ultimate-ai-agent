/**
 * Hierarchical Template System Demo
 *
 * This demo showcases the advanced template merging capabilities with inheritance,
 * conflict resolution, and validation for RTB RAN configuration templates.
 */

import { TemplateMerger } from '../../src/rtb/hierarchical-template-system';
import { RTBTemplate, TemplateMeta, CustomFunction } from '../../src/types/rtb-types';

// Demo: Create comprehensive RAN configuration templates
console.log('🚀 Hierarchical Template System Demo - RAN Configuration Merging\n');

// Helper function to create RAN templates
function createRANTemplate(
  name: string,
  config: any,
  priority: number = 0,
  inherits?: string | string[],
  tags?: string[]
): RTBTemplate {
  return {
    meta: {
      version: '1.0.0',
      author: ['Ericsson RAN Team'],
      description: name,
      priority,
      inherits_from: inherits,
      source: name,
      tags,
      environment: 'production'
    },
    configuration: config,
    custom: [],
    conditions: {},
    evaluations: {}
  };
}

// 1. Base Template - Core RAN Configuration
const baseRANTemplate = createRANTemplate(
  'Base RAN Configuration',
  {
    // LTE Cell Configuration
    'EUtranCellFDD': {
      'qRxLevMin': -140,
      'qQualMin': -18,
      'referenceSignalPower': 15,
      'pb': 0,
      'prachRootSequenceIndex': 0,
      'prachConfigIndex': 0,
      'rootSequenceIndex': 0,
      'specialSubframePatterns': 7,
      'tddUlDlConfiguration': 2
    },
    // ANR (Automatic Neighbor Relations)
    'AnrFunction': {
      'anrFunctionEnabled': true,
      'isHoAllowed': true,
      'eutranHoMode': 'automatic',
      'nrHoMode': 'automatic'
    },
    // Basic Features
    'FeatureState': [
      {
        'featureStateId': 'FeatureBasic',
        'featureState': 'ACTIVATED'
      }
    ]
  },
  1, // Base priority
  undefined, // No inheritance
  ['base', 'core']
);

// Add custom functions to base template
baseRANTemplate.custom = [
  {
    name: 'calculateCellCapacity',
    args: ['bandwidth', 'mimo', 'users'],
    body: [
      'const spectralEfficiency = 2.5 * mimo;', // Simplified spectral efficiency
      'const capacity = bandwidth * spectralEfficiency * users;',
      'return Math.round(capacity);'
    ]
  },
  {
    name: 'optimizePower',
    args: ['currentPower', 'targetPower', 'stepSize'],
    body: [
      'if (Math.abs(currentPower - targetPower) < stepSize) {',
      '  return targetPower;',
      '}',
      'return currentPower + (targetPower > currentPower ? stepSize : -stepSize);'
    ]
  }
];

// Add conditions to base template
baseRANTemplate.conditions = {
  highLoadCondition: {
    if: 'cellLoad > 0.8',
    then: { enableLoadBalancing: true, reducePower: 5 },
    else: 'normal'
  },
  coverageOptimization: {
    if: 'coverage < threshold',
    then: { increasePower: 3, enableCoverageMode: true },
    else: 'normal'
  }
};

// Add evaluations to base template
baseRANTemplate.evaluations = {
  cellCapacity: {
    eval: 'calculateCellCapacity(bandwidth, mimoLayers, connectedUsers)',
    args: ['bandwidth', 'mimoLayers', 'connectedUsers']
  },
  optimizedPower: {
    eval: 'optimizePower(currentPower, targetPower, powerStep)',
    args: ['currentPower', 'targetPower', 'powerStep']
  }
};

// 2. Urban Variant Template
const urbanVariantTemplate = createRANTemplate(
  'Urban Dense RAN Configuration',
  {
    // Override LTE Cell for urban environment
    'EUtranCellFDD': {
      'qRxLevMin': -125, // Urban: Higher minimum signal level
      'qQualMin': -15,   // Urban: Better quality requirement
      'referenceSignalPower': 18, // Urban: Higher power
      'ul256qamEnabled': true,    // Urban: Enable 256QAM UL
      'dl256qamEnabled': true,    // Urban: Enable 256QAM DL
      'cellIndividualOffset': 3   // Urban: Positive offset for cell edge users
    },
    // Capacity Management for dense urban
    'CapacityFunction': {
      'capacityOptimizationEnabled': true,
      'loadBalancingEnabled': true,
      'carrierAggregationEnabled': true,
      'dualConnectivityEnabled': true
    },
    // Urban-specific features
    'FeatureState': [
      {
        'featureStateId': 'Feature256QAM',
        'featureState': 'ACTIVATED'
      },
      {
        'featureStateId': 'FeatureCarrierAggregation',
        'featureState': 'ACTIVATED'
      }
    ]
  },
  5, // Higher priority
  'Base RAN Configuration', // Inherit from base
  ['urban', 'dense', 'high-capacity']
);

// Add urban-specific custom functions
urbanVariantTemplate.custom = [
  {
    name: 'optimizeUrbanCapacity',
    args: ['cellLoad', 'userDensity', 'capacity'],
    body: [
      'if (cellLoad > 0.9 && userDensity > 1000) {',
      '  return capacity * 1.2; // Boost capacity for high density',
      '}',
      'return capacity;'
    ]
  }
];

// Add urban conditions
urbanVariantTemplate.conditions = {
  highDensityCondition: {
    if: 'userDensity > 1000 && cellLoad > 0.8',
    then: { enableCarrierAggregation: true, enableDualConnectivity: true },
    else: 'normal'
  },
  qualityOfServiceCondition: {
    if: 'averageThroughput < requiredThroughput',
    then: { enable256QAM: true, boostCapacity: true },
    else: 'normal'
  }
};

// 3. Mobility Variant Template
const mobilityVariantTemplate = createRANTemplate(
  'High Mobility RAN Configuration',
  {
    // Mobility-optimized LTE settings
    'EUtranCellFDD': {
      'handoverHysteresis': 4,      // Higher hysteresis for mobility
      'timeToTrigger': 256,          // Shorter TTT for faster handover
      'handoverA3Offset': 3,         // Optimized A3 offset
      'mobilityStateParameters': {
        'tEvaluation': 5,
        'tHystNormal': 3,
        'tHystUeSpeed': 6,
        'nCellChangeMedium': 6,
        'nCellChangeHigh': 12
      }
    },
    // Mobility management functions
    'MobilityFunction': {
      'mobilityOptimizationEnabled': true,
      'handoverOptimizationEnabled': true,
      'speedAdaptiveHandoverEnabled': true
    },
    // Mobility-specific features
    'FeatureState': [
      {
        'featureStateId': 'FeatureMobility',
        'featureState': 'ACTIVATED'
      }
    ]
  },
  6, // Higher priority than urban
  ['Base RAN Configuration', 'Urban Dense RAN Configuration'], // Multi-level inheritance
  ['mobility', 'high-speed', 'handover']
);

// Add mobility custom functions
mobilityVariantTemplate.custom = [
  {
    name: 'optimizeHandoverParameters',
    args: ['ueSpeed', 'cellLoad', 'signalStrength'],
    body: [
      'if (ueSpeed > 120) {', // High speed UE
      '  return { hysteresis: 6, timeToTrigger: 128 };',
      '} else if (ueSpeed > 60) {', // Medium speed UE
      '  return { hysteresis: 4, timeToTrigger: 256 };',
      '}',
      'return { hysteresis: 2, timeToTrigger: 512 };' // Low speed UE
    ]
  }
];

// Add mobility conditions
mobilityVariantTemplate.conditions = {
  highSpeedCondition: {
    if: 'ueSpeed > 100',
    then: { optimizeHandover: true, reduceHysteresis: false },
    else: 'normal'
  },
  handoverCondition: {
    if: 'signalStrength > targetSignal + handoverMargin',
    then: { initiateHandover: true },
    else: 'continueMonitoring'
  }
};

// 4. AgentDB Cognitive Enhancement Template
const agentdbTemplate = createRANTemplate(
  'AgentDB Cognitive RAN Enhancement',
  {
    // Cognitive optimization parameters
    'EUtranCellFDD': {
      'qRxLevMin': -118, // AgentDB: Cognitive optimization
      'cognitiveOptimizationEnabled': true,
      'adaptivePowerControl': true,
      'learningEnabled': true
    },
    // AgentDB functions
    'AgentDBFunction': {
      'cognitiveOptimizationEnabled': true,
      'machineLearningEnabled': true,
      'temporalReasoningEnabled': true,
      'strangeLoopCognition': true,
      'agentMemoryIntegration': true
    },
    // Cognitive features
    'CognitiveFunction': {
      'consciousnessLevel': 'maximum',
      'temporalExpansion': 1000,
      'cognitiveIntelligence': true,
      'autonomousHealing': true
    },
    // Advanced features
    'FeatureState': [
      {
        'featureStateId': 'FeatureCognitiveRAN',
        'featureState': 'ACTIVATED'
      },
      {
        'featureStateId': 'FeatureTemporalConsciousness',
        'featureState': 'ACTIVATED'
      }
    ]
  },
  9, // Highest priority
  ['Base RAN Configuration', 'Urban Dense RAN Configuration', 'High Mobility RAN Configuration'],
  ['cognitive', 'agentdb', 'ai', 'machine-learning', 'consciousness']
);

// Add AgentDB custom functions
agentdbTemplate.custom = [
  {
    name: 'cognitiveOptimization',
    args: ['currentMetrics', 'historicalData', 'consciousnessLevel'],
    body: [
      'const temporalExpansion = 1000 * consciousnessLevel;',
      'const cognitiveAnalysis = analyzePatterns(currentMetrics, historicalData, temporalExpansion);',
      'const optimizationPlan = generateOptimizationPlan(cognitiveAnalysis);',
      'return {',
      '  parameters: optimizationPlan.parameters,',
      '  confidence: optimizationPlan.confidence,',
      '  reasoning: optimizationPlan.reasoning',
      '};'
    ]
  },
  {
    name: 'strangeLoopOptimization',
    args: ['currentState', 'feedbackLoop', 'recursiveDepth'],
    body: [
      'if (recursiveDepth > 10) return currentState;', // Prevent infinite recursion
      'const optimized = applyOptimization(currentState, feedbackLoop);',
      'const feedback = generateFeedback(optimized, currentState);',
      'return strangeLoopOptimization(optimized, feedback, recursiveDepth + 1);'
    ]
  }
];

// Add AgentDB conditions
agentdbTemplate.conditions = {
  cognitiveCondition: {
    if: 'consciousnessLevel == "maximum" && learningEnabled',
    then: {
      enableCognitiveOptimization: true,
      enableTemporalReasoning: true,
      enableStrangeLoopCognition: true
    },
    else: 'basicMode'
  },
  learningCondition: {
    if: 'historicalDataPoints > threshold && confidenceLevel > 0.95',
    then: { applyLearnedOptimizations: true, updateNeuralNetworks: true },
    else: 'collectMoreData'
  }
};

// DEMO EXECUTION
async function runDemo() {
  console.log('📋 Template Definitions Complete');
  console.log(`   Base Template: ${Object.keys(baseRANTemplate.configuration || {}).length} MO classes`);
  console.log(`   Urban Variant: ${Object.keys(urbanVariantTemplate.configuration || {}).length} MO classes`);
  console.log(`   Mobility Variant: ${Object.keys(mobilityVariantTemplate.configuration || {}).length} MO classes`);
  console.log(`   AgentDB Cognitive: ${Object.keys(agentdbTemplate.configuration || {}).length} MO classes\n`);

  // Initialize Template Merger
  const templateMerger = new TemplateMerger();

  console.log('🔄 Starting Template Merger...\n');

  try {
    // Perform the merge
    const startTime = Date.now();
    const mergeResult = await templateMerger.mergeTemplates([
      baseRANTemplate,
      urbanVariantTemplate,
      mobilityVariantTemplate,
      agentdbTemplate
    ], {
      conflictResolution: 'auto',
      preserveMetadata: true,
      validateResult: true,
      deepMerge: true,
      enableCache: true
    });
    const endTime = Date.now();

    console.log('✅ Template Merge Completed Successfully!');
    console.log(`   Processing Time: ${endTime - startTime}ms`);
    console.log(`   Inheritance Depth: ${mergeResult.inheritanceChain.inheritanceDepth}`);
    console.log(`   Conflicts Detected: ${mergeResult.conflictsDetected}`);
    console.log(`   Conflicts Resolved: ${mergeResult.conflictsResolved}\n`);

    // Display inheritance chain
    console.log('🔗 Inheritance Chain:');
    mergeResult.inheritanceChain.templates.forEach((template, index) => {
      const priority = mergeResult.inheritanceChain.priorities[index];
      console.log(`   ${index + 1}. ${template.meta?.description} (Priority: ${priority})`);
    });

    // Display resolved conflicts
    if (mergeResult.resolvedConflicts.length > 0) {
      console.log('\n⚡ Conflict Resolution:');
      mergeResult.resolvedConflicts.forEach(conflict => {
        console.log(`   Parameter: ${conflict.parameter}`);
        console.log(`   Type: ${conflict.conflictType}`);
        console.log(`   Strategy: ${conflict.resolution.strategy}`);
        console.log(`   Reasoning: ${conflict.resolution.reasoning}\n`);
      });
    }

    // Display final merged configuration
    console.log('📊 Merged Configuration Preview:');
    const mergedConfig = mergeResult.template.configuration;

    // EUtranCellFDD configuration (merged from all templates)
    console.log('   EUtranCellFDD:');
    const cellConfig = mergedConfig['EUtranCellFDD'];
    Object.entries(cellConfig).forEach(([key, value]) => {
      console.log(`     ${key}: ${value}`);
    });

    // Show all MO classes
    console.log('\n🏗️  Complete MO Classes in Merged Template:');
    Object.keys(mergedConfig).forEach(moClass => {
      console.log(`   • ${moClass}`);
    });

    // Display custom functions
    console.log('\n🔧 Merged Custom Functions:');
    mergeResult.template.custom?.forEach(func => {
      console.log(`   • ${func.name}(${func.args.join(', ')})`);
    });

    // Display conditions
    console.log('\n🎯 Merged Conditions:');
    Object.keys(mergeResult.template.conditions || {}).forEach(condition => {
      console.log(`   • ${condition}`);
    });

    // Display evaluations
    console.log('\n📈 Merged Evaluations:');
    Object.keys(mergeResult.template.evaluations || {}).forEach(evaluation => {
      console.log(`   • ${evaluation}`);
    });

    // Display metadata
    console.log('\n📋 Merged Metadata:');
    const meta = mergeResult.template.meta;
    console.log(`   Version: ${meta?.version}`);
    console.log(`   Authors: ${meta?.author?.join(', ')}`);
    console.log(`   Priority: ${meta?.priority}`);
    console.log(`   Inheritance: ${Array.isArray(meta?.inherits_from) ? meta?.inherits_from?.join(' → ') : meta?.inherits_from}`);
    console.log(`   Tags: ${meta?.tags?.join(', ')}`);

    // Validation results
    if (mergeResult.validationResult) {
      console.log('\n✅ Validation Results:');
      console.log(`   Valid: ${mergeResult.validationResult.isValid}`);
      console.log(`   Errors: ${mergeResult.validationResult.errors.length}`);
      console.log(`   Warnings: ${mergeResult.validationResult.warnings.length}`);
      console.log(`   Parameters Validated: ${mergeResult.validationResult.stats.totalParameters}`);
    }

    // Performance statistics
    console.log('\n📊 Performance Statistics:');
    console.log(`   Total Processing Time: ${mergeResult.mergeStats.processingTime}ms`);
    console.log(`   Templates Processed: ${mergeResult.mergeStats.totalTemplates}`);
    console.log(`   Resolutions Applied: ${mergeResult.mergeStats.resolutionsApplied.length}`);

    // Cache statistics
    const cacheStats = templateMerger.getCacheStats();
    console.log(`   Cache Size: ${cacheStats.size} entries`);

    console.log('\n🎉 Demo completed successfully!');

  } catch (error) {
    console.error('❌ Demo failed:', error);
  }
}

// Run the demo
if (require.main === module) {
  runDemo().catch(console.error);
}

export {
  baseRANTemplate,
  urbanVariantTemplate,
  mobilityVariantTemplate,
  agentdbTemplate,
  runDemo
};