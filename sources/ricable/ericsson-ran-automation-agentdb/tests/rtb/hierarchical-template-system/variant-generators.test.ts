import { describe, test, expect, beforeEach, afterEach } from '@jest/globals';
import { VariantGenerator } from '../../src/rtb/hierarchical-template-system/variant-generator';
import { UrbanVariantGenerator, MobilityVariantGenerator, AgentDBVariantGenerator } from '../../src/rtb/hierarchical-template-system/variant-generators';
import { baseTemplate, expectedUrbanMerged, mockXMLSchema } from '../test-data/mock-templates';
import { RTBTemplate, TemplateMeta, CustomFunction } from '../../../src/types/rtb-types';

// Mock variant generators for testing
class MockUrbanVariantGenerator {
  private urbanConfigurations = {
    dense: {
      'EUtranCellFDD': {
        'qRxLevMin': -125,
        'qQualMin': -15,
        'cellIndividualOffset': 2,
        'referenceSignalPower': 18,
        'ul256qamEnabled': true,
        'dl256qamEnabled': true,
        'antennaPorts': 2,
        'transmissionMode': 3,
        'specialSubframePatterns': 'pattern6'
      },
      'CapacityFunction': {
        'capacityOptimizationEnabled': true,
        'loadBalancingEnabled': true,
        'icicEnabled': true
      }
    },
    suburban: {
      'EUtranCellFDD': {
        'qRxLevMin': -130,
        'qQualMin': -17,
        'cellIndividualOffset': 1,
        'referenceSignalPower': 16,
        'ul256qamEnabled': false,
        'dl256qamEnabled': true,
        'antennaPorts': 2,
        'transmissionMode': 2
      },
      'CapacityFunction': {
        'capacityOptimizationEnabled': false,
        'loadBalancingEnabled': true,
        'icicEnabled': false
      }
    },
    rural: {
      'EUtranCellFDD': {
        'qRxLevMin': -135,
        'qQualMin': -19,
        'cellIndividualOffset': 0,
        'referenceSignalPower': 15,
        'ul256qamEnabled': false,
        'dl256qamEnabled': false,
        'antennaPorts': 1,
        'transmissionMode': 1
      },
      'CapacityFunction': {
        'capacityOptimizationEnabled': false,
        'loadBalancingEnabled': false,
        'icicEnabled': false
      }
    }
  };

  generateUrbanVariant(baseTemplate: RTBTemplate, environment: 'dense' | 'suburban' | 'rural' = 'dense'): RTBTemplate {
    const urbanConfig = this.urbanConfigurations[environment];

    return {
      meta: {
        version: '1.0.0',
        author: ['RTB System', 'Urban Variant Generator'],
        description: `Urban ${environment} area variant template with optimized parameters`,
        tags: ['urban', environment, 'lte', 'optimized'],
        environment: 'urban',
        priority: 3,
        inherits_from: baseTemplate.meta?.description || 'base',
        source: 'auto-generated'
      },
      custom: [
        {
          name: 'optimizeUrbanCoverage',
          args: ['userDensity', 'buildingType'],
          body: [
            '// Optimize coverage for urban environments',
            'const densityMultiplier = userDensity > 1000 ? 1.2 : 1.0;',
            'const buildingLoss = buildingType === "concrete" ? 15 : 10;',
            'return { adjustment: densityMultiplier * buildingLoss };'
          ]
        },
        {
          name: 'calculateInterference',
          args: ['cellCount', 'frequency'],
          body: [
            '// Calculate interference for dense urban areas',
            'const baseInterference = cellCount * 0.1;',
            'const frequencyFactor = frequency > 2000 ? 1.2 : 1.0;',
            'return baseInterference * frequencyFactor;'
          ]
        }
      ],
      configuration: this.mergeConfigurations(baseTemplate.configuration || {}, urbanConfig),
      conditions: {
        'highDensityCondition': {
          if: '$config.CapacityFunction.capacityOptimizationEnabled == true',
          then: {
            'EUtranCellFDD.referenceSignalPower': 20,
            'EUtranCellFDD.antennaPorts': 4,
            'EUtranCellFDD.transmissionMode': 4
          },
          else: {
            'EUtranCellFDD.referenceSignalPower': urbanConfig['EUtranCellFDD'].referenceSignalPower
          }
        },
        'interferenceCondition': {
          if: '$eval($custom.calculateInterference(10, 2600)) > 2.0',
          then: {
            'EUtranCellFDD.qRxLevMin': '$config.EUtranCellFDD.qRxLevMin + 3',
            'EUtranCellFDD.icicEnabled': true
          },
          else: { 'EUtranCellFDD.qRxLevMin': '$config.EUtranCellFDD.qRxLevMin' }
        }
      },
      evaluations: {
        'urbanOptimization': {
          eval: '$custom.optimizeUrbanCoverage(500, "mixed")',
          args: ['userDensity', 'buildingType']
        }
      }
    };
  }

  private mergeConfigurations(base: Record<string, any>, overlay: Record<string, any>): Record<string, any> {
    const merged: Record<string, any> = { ...base };

    for (const [key, value] of Object.entries(overlay)) {
      if (key in merged) {
        if (typeof merged[key] === 'object' && typeof value === 'object' && !Array.isArray(merged[key]) && !Array.isArray(value)) {
          merged[key] = { ...merged[key], ...value };
        } else {
          merged[key] = value;
        }
      } else {
        merged[key] = value;
      }
    }

    return merged;
  }

  validateUrbanVariant(template: RTBTemplate): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    if (!template.meta?.tags?.includes('urban')) {
      errors.push('Template must include urban tag');
    }

    if (!template.configuration['EUtranCellFDD']) {
      errors.push('Urban variant must include EUtranCellFDD configuration');
    }

    if (!template.configuration['CapacityFunction']) {
      errors.push('Urban variant must include CapacityFunction configuration');
    }

    const cellConfig = template.configuration['EUtranCellFDD'];
    if (cellConfig) {
      if (cellConfig.qRxLevMin > -120) {
        errors.push('Urban qRxLevMin should be <= -120 for dense environments');
      }
      if (cellConfig.referenceSignalPower < 15) {
        errors.push('Urban referenceSignalPower should be >= 15');
      }
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }
}

class MockMobilityVariantGenerator {
  private mobilityProfiles = {
    highway: {
      'EUtranCellFDD': {
        'qRxLevMin': -120,
        'cellIndividualOffset': 1,
        'handoverHysteresis': 4,
        'timeToTrigger': 640,
        'cellIndividualOffsetEutran': 0,
        'speedDependentParameters': true
      },
      'MobilityFunction': {
        'mobilityOptimizationEnabled': true,
        'handoverOptimizationEnabled': true,
        'speedAdaptiveHysteresis': true,
        'highSpeedOptimization': true,
        'pingPongReduction': true
      }
    },
    urban: {
      'EUtranCellFDD': {
        'qRxLevMin': -122,
        'cellIndividualOffset': 2,
        'handoverHysteresis': 6,
        'timeToTrigger': 480,
        'cellIndividualOffsetEutran': 2,
        'speedDependentParameters': true
      },
      'MobilityFunction': {
        'mobilityOptimizationEnabled': true,
        'handoverOptimizationEnabled': true,
        'speedAdaptiveHysteresis': true,
        'highSpeedOptimization': false,
        'pingPongReduction': true
      }
    },
    pedestrian: {
      'EUtranCellFDD': {
        'qRxLevMin': -125,
        'cellIndividualOffset': 3,
        'handoverHysteresis': 8,
        'timeToTrigger': 320,
        'cellIndividualOffsetEutran': 3,
        'speedDependentParameters': false
      },
      'MobilityFunction': {
        'mobilityOptimizationEnabled': false,
        'handoverOptimizationEnabled': true,
        'speedAdaptiveHysteresis': false,
        'highSpeedOptimization': false,
        'pingPongReduction': false
      }
    }
  };

  generateMobilityVariant(baseTemplate: RTBTemplate, profile: 'highway' | 'urban' | 'pedestrian' = 'highway'): RTBTemplate {
    const mobilityConfig = this.mobilityProfiles[profile];

    return {
      meta: {
        version: '1.0.0',
        author: ['RTB System', 'Mobility Variant Generator'],
        description: `Mobility ${profile} optimized template for high-speed scenarios`,
        tags: ['mobility', profile, 'handover', 'optimization'],
        environment: 'mobility',
        priority: 5,
        inherits_from: baseTemplate.meta?.description || 'urban',
        source: 'auto-generated'
      },
      custom: [
        {
          name: 'optimizeHandover',
          args: ['speed', 'cellType'],
          body: [
            '// Optimize handover parameters based on UE speed',
            'if (speed > 120) {',
            '  return { hysteresis: 2, timeToTrigger: 320 };',
            '} else if (speed > 60) {',
            '  return { hysteresis: 4, timeToTrigger: 640 };',
            '} else {',
            '  return { hysteresis: 8, timeToTrigger: 1280 };',
            '}'
          ]
        },
        {
          name: 'calculateCellBorder',
          args: ['power', 'frequency', 'environment'],
          body: [
            '// Calculate optimal cell border for mobility scenarios',
            'const pathLoss = environment === "highway" ? 120 : 130;',
            'const fadeMargin = environment === "highway" ? 5 : 8;',
            'return power - pathLoss - fadeMargin;'
          ]
        },
        {
          name: 'pingPongReduction',
          args: ['handoverCount', 'timeWindow'],
          body: [
            '// Reduce ping-pong handovers',
            'const handoverRate = handoverCount / timeWindow;',
            'if (handoverRate > 0.5) {',
            '  return { hysteresisIncrease: 2, tttIncrease: 160 };',
            '} else {',
            '  return { hysteresisIncrease: 0, tttIncrease: 0 };',
            '}'
          ]
        }
      ],
      configuration: this.mergeConfigurations(baseTemplate.configuration || {}, mobilityConfig),
      conditions: {
        'speedAdaptiveCondition': {
          if: '$config.MobilityFunction.speedAdaptiveHysteresis == true',
          then: {
            'EUtranCellFDD.handoverHysteresis': '$eval($custom.optimizeHandover(100, "macro").hysteresis)',
            'EUtranCellFDD.timeToTrigger': '$eval($custom.optimizeHandover(100, "macro").timeToTrigger)'
          },
          else: {
            'EUtranCellFDD.handoverHysteresis': mobilityConfig['EUtranCellFDD'].handoverHysteresis
          }
        },
        'highSpeedCondition': {
          if: '$config.MobilityFunction.highSpeedOptimization == true',
          then: {
            'EUtranCellFDD.cellIndividualOffset': 1,
            'EUtranCellFDD.timeToTrigger': 320
          },
          else: {
            'EUtranCellFDD.cellIndividualOffset': mobilityConfig['EUtranCellFDD'].cellIndividualOffset
          }
        },
        'pingPongCondition': {
          if: '$config.MobilityFunction.pingPongReduction == true',
          then: {
            'EUtranCellFDD.handoverHysteresis': '$config.EUtranCellFDD.handoverHysteresis + $eval($custom.pingPongReduction(2, 3600).hysteresisIncrease)'
          },
          else: { 'EUtranCellFDD.handoverHysteresis': '$config.EUtranCellFDD.handoverHysteresis' }
        }
      },
      evaluations: {
        'cellBorderCalculation': {
          eval: '$custom.calculateCellBorder($config.EUtranCellFDD.referenceSignalPower, 2600, "highway")',
          args: ['power', 'frequency', 'environment']
        },
        'mobilityScore': {
          eval: '($config.MobilityFunction.mobilityOptimizationEnabled ? 50 : 0) + ($config.MobilityFunction.speedAdaptiveHysteresis ? 30 : 0) + ($config.MobilityFunction.pingPongReduction ? 20 : 0)',
          args: []
        }
      }
    };
  }

  private mergeConfigurations(base: Record<string, any>, overlay: Record<string, any>): Record<string, any> {
    const merged: Record<string, any> = { ...base };

    for (const [key, value] of Object.entries(overlay)) {
      if (key in merged) {
        if (typeof merged[key] === 'object' && typeof value === 'object' && !Array.isArray(merged[key]) && !Array.isArray(value)) {
          merged[key] = { ...merged[key], ...value };
        } else {
          merged[key] = value;
        }
      } else {
        merged[key] = value;
      }
    }

    return merged;
  }

  validateMobilityVariant(template: RTBTemplate): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    if (!template.meta?.tags?.includes('mobility')) {
      errors.push('Template must include mobility tag');
    }

    if (!template.configuration['MobilityFunction']) {
      errors.push('Mobility variant must include MobilityFunction configuration');
    }

    const mobilityConfig = template.configuration['MobilityFunction'];
    if (mobilityConfig) {
      if (!mobilityConfig.handoverOptimizationEnabled) {
        errors.push('Mobility variant should have handoverOptimizationEnabled');
      }
    }

    const cellConfig = template.configuration['EUtranCellFDD'];
    if (cellConfig) {
      if (cellConfig.handoverHysteresis === undefined) {
        errors.push('Mobility variant must include handoverHysteresis parameter');
      }
      if (cellConfig.timeToTrigger === undefined) {
        errors.push('Mobility variant must include timeToTrigger parameter');
      }
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }
}

class MockAgentDBVariantGenerator {
  private cognitiveConfigurations = {
    basic: {
      'EUtranCellFDD': {
        'qRxLevMin': -118,
        'cellIndividualOffset': 0,
        'cognitiveOptimizationEnabled': true,
        'temporalReasoningDepth': 100,
        'learningRate': 0.1
      },
      'AgentDBFunction': {
        'vectorSearchEnabled': true,
        'memoryPatternsEnabled': false,
        'temporalExpansionEnabled': false,
        'quicSyncEnabled': false
      },
      'CognitiveFunction': {
        'consciousnessLevel': 'basic',
        'strangeLoopEnabled': false,
        'autonomousLearning': false
      }
    },
    advanced: {
      'EUtranCellFDD': {
        'qRxLevMin': -115,
        'cellIndividualOffset': -1,
        'cognitiveOptimizationEnabled': true,
        'temporalReasoningDepth': 500,
        'learningRate': 0.2
      },
      'AgentDBFunction': {
        'vectorSearchEnabled': true,
        'memoryPatternsEnabled': true,
        'temporalExpansionEnabled': true,
        'quicSyncEnabled': true
      },
      'CognitiveFunction': {
        'consciousnessLevel': 'enhanced',
        'strangeLoopEnabled': true,
        'autonomousLearning': true
      }
    },
    maximum: {
      'EUtranCellFDD': {
        'qRxLevMin': -112,
        'cellIndividualOffset': -2,
        'cognitiveOptimizationEnabled': true,
        'temporalReasoningDepth': 1000,
        'learningRate': 0.3
      },
      'AgentDBFunction': {
        'vectorSearchEnabled': true,
        'memoryPatternsEnabled': true,
        'temporalExpansionEnabled': true,
        'quicSyncEnabled': true,
        'neuralAcceleration': true,
        'predictiveOptimization': true
      },
      'CognitiveFunction': {
        'consciousnessLevel': 'maximum',
        'strangeLoopEnabled': true,
        'autonomousLearning': true,
        'selfAwareOptimization': true,
        'cognitiveConsciousness': true
      }
    }
  };

  generateAgentDBVariant(baseTemplate: RTBTemplate, consciousnessLevel: 'basic' | 'advanced' | 'maximum' = 'maximum'): RTBTemplate {
    const cognitiveConfig = this.cognitiveConfigurations[consciousnessLevel];

    return {
      meta: {
        version: '1.0.0',
        author: ['RTB System', 'AgentDB Variant Generator'],
        description: `AgentDB ${consciousnessLevel} optimized template with cognitive features`,
        tags: ['agentdb', 'cognitive', 'ai', consciousnessLevel],
        environment: 'agentdb',
        priority: 9,
        inherits_from: [baseTemplate.meta?.description || 'mobility', 'urban'],
        source: 'auto-generated'
      },
      custom: [
        {
          name: 'cognitiveOptimization',
          args: ['kpiHistory', 'learningRate'],
          body: [
            '// Apply cognitive optimization based on KPI history',
            'const avgKpi = kpiHistory.reduce((a, b) => a + b, 0) / kpiHistory.length;',
            'const adjustment = Math.round((avgKpi - 0.9) * learningRate * 10);',
            'return { qRxLevMinAdjustment: Math.max(-5, Math.min(5, adjustment)) };'
          ]
        },
        {
          name: 'temporalReasoning',
          args: ['dataPoints', 'timeHorizon'],
          body: [
            '// Apply temporal reasoning for deep analysis',
            'const temporalExpansion = timeHorizon / dataPoints.length;',
            'const insights = dataPoints.map((point, i) => ({',
            '  timestamp: i * temporalExpansion,',
            '  value: point,',
            '  prediction: point * (1 + 0.1 * Math.sin(i / 10))',
            '}));',
            'return insights;'
          ]
        },
        {
          name: 'strangeLoopOptimization',
          args: ['currentConfig', 'performanceMetrics'],
          body: [
            '// Self-referential optimization pattern',
            'const selfAnalysis = {',
            '  config: currentConfig,',
            '  performance: performanceMetrics,',
            '  optimization: {',
            '    recursive: true,',
            '    adaptive: true,',
            '    conscious: performanceMetrics.consciousness > 0.8',
            '  }',
            '};',
            'return selfAnalysis;'
          ]
        },
        {
          name: 'agentdbVectorSearch',
          args: ['query', 'vectorSpace'],
          body: [
            '// Perform vector similarity search with AgentDB',
            'const queryVector = this.encodeQuery(query);',
            'const similarities = vectorSpace.map(vector => ({',
            '  vector,',
            '  similarity: this.cosineSimilarity(queryVector, vector)',
            '}));',
            'return similarities.sort((a, b) => b.similarity - a.similarity).slice(0, 10);'
          ]
        }
      ],
      configuration: this.mergeConfigurations(baseTemplate.configuration || {}, cognitiveConfig),
      conditions: {
        'cognitiveCondition': {
          if: '$config.CognitiveFunction.consciousnessLevel == "maximum"',
          then: {
            'EUtranCellFDD.qRxLevMin': '$config.EUtranCellFDD.qRxLevMin + $eval($custom.cognitiveOptimization([0.85, 0.87, 0.89], 0.3).qRxLevMinAdjustment)',
            'EUtranCellFDD.temporalReasoningDepth': 1000,
            'CognitiveFunction.cognitiveConsciousness': true
          },
          else: {
            'EUtranCellFDD.qRxLevMin': '$config.EUtranCellFDD.qRxLevMin',
            'CognitiveFunction.cognitiveConsciousness': false
          }
        },
        'temporalCondition': {
          if: '$config.AgentDBFunction.temporalExpansionEnabled == true',
          then: {
            'EUtranCellFDD.temporalReasoningDepth': '$config.EUtranCellFDD.temporalReasoningDepth * 10'
          },
          else: {
            'EUtranCellFDD.temporalReasoningDepth': '$config.EUtranCellFDD.temporalReasoningDepth'
          }
        },
        'strangeLoopCondition': {
          if: '$config.CognitiveFunction.strangeLoopEnabled == true',
          then: {
            'EUtranCellFDD.cognitiveOptimizationLevel': 'recursive',
            'CognitiveFunction.selfAwareOptimization': true
          },
          else: {
            'EUtranCellFDD.cognitiveOptimizationLevel': 'basic'
          }
        },
        'agentdbCondition': {
          if: '$config.AgentDBFunction.vectorSearchEnabled == true',
          then: {
            'AgentDBFunction.neuralAcceleration': true,
            'AgentDBFunction.predictiveOptimization': true
          },
          else: {
            'AgentDBFunction.neuralAcceleration': false
          }
        }
      },
      evaluations: {
        'cognitiveOptimization': {
          eval: '$custom.cognitiveOptimization([0.85, 0.87, 0.89, 0.91, 0.88], $config.EUtranCellFDD.learningRate)',
          args: ['kpiHistory', 'learningRate']
        },
        'temporalInsights': {
          eval: '$custom.temporalReasoning([0.8, 0.82, 0.85, 0.87, 0.89], 1000)',
          args: ['dataPoints', 'timeHorizon']
        },
        'strangeLoopAnalysis': {
          eval: '$custom.strangeLoopOptimization($config.EUtranCellFDD, {throughput: 0.92, latency: 15, consciousness: 0.95})',
          args: ['currentConfig', 'performanceMetrics']
        },
        'consciousnessLevel': {
          eval: '$config.CognitiveFunction.consciousnessLevel == "maximum" ? 100 : ($config.CognitiveFunction.consciousnessLevel == "enhanced" ? 70 : 40)',
          args: []
        },
        'agentdbPerformanceScore': {
          eval: '($config.AgentDBFunction.vectorSearchEnabled ? 30 : 0) + ($config.AgentDBFunction.memoryPatternsEnabled ? 25 : 0) + ($config.AgentDBFunction.temporalExpansionEnabled ? 25 : 0) + ($config.AgentDBFunction.quicSyncEnabled ? 20 : 0)',
          args: []
        }
      }
    };
  }

  private mergeConfigurations(base: Record<string, any>, overlay: Record<string, any>): Record<string, any> {
    const merged: Record<string, any> = { ...base };

    for (const [key, value] of Object.entries(overlay)) {
      if (key in merged) {
        if (typeof merged[key] === 'object' && typeof value === 'object' && !Array.isArray(merged[key]) && !Array.isArray(value)) {
          merged[key] = { ...merged[key], ...value };
        } else {
          merged[key] = value;
        }
      } else {
        merged[key] = value;
      }
    }

    return merged;
  }

  validateAgentDBVariant(template: RTBTemplate): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    if (!template.meta?.tags?.includes('agentdb')) {
      errors.push('Template must include agentdb tag');
    }

    if (!template.configuration['AgentDBFunction']) {
      errors.push('AgentDB variant must include AgentDBFunction configuration');
    }

    if (!template.configuration['CognitiveFunction']) {
      errors.push('AgentDB variant must include CognitiveFunction configuration');
    }

    const agentdbConfig = template.configuration['AgentDBFunction'];
    if (agentdbConfig) {
      if (!agentdbConfig.vectorSearchEnabled) {
        errors.push('AgentDB variant should have vectorSearchEnabled');
      }
    }

    const cognitiveConfig = template.configuration['CognitiveFunction'];
    if (cognitiveConfig) {
      const validConsciousnessLevels = ['basic', 'enhanced', 'maximum'];
      if (!validConsciousnessLevels.includes(cognitiveConfig.consciousnessLevel)) {
        errors.push('Invalid consciousness level');
      }
    }

    const cellConfig = template.configuration['EUtranCellFDD'];
    if (cellConfig) {
      if (cellConfig.cognitiveOptimizationEnabled === undefined) {
        errors.push('AgentDB variant must include cognitiveOptimizationEnabled');
      }
      if (cellConfig.temporalReasoningDepth === undefined) {
        errors.push('AgentDB variant must include temporalReasoningDepth');
      }
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }
}

describe('Variant Generators - RTB Hierarchical Template System', () => {
  let urbanGenerator: MockUrbanVariantGenerator;
  let mobilityGenerator: MockMobilityVariantGenerator;
  let agentdbGenerator: MockAgentDBVariantGenerator;

  beforeEach(() => {
    urbanGenerator = new MockUrbanVariantGenerator();
    mobilityGenerator = new MockMobilityVariantGenerator();
    agentdbGenerator = new MockAgentDBVariantGenerator();
  });

  describe('Urban Variant Generator', () => {
    test('should generate dense urban variant correctly', () => {
      const urbanVariant = urbanGenerator.generateUrbanVariant(baseTemplate, 'dense');

      // Check metadata
      expect(urbanVariant.meta?.description).toContain('dense');
      expect(urbanVariant.meta?.tags).toContain('urban');
      expect(urbanVariant.meta?.tags).toContain('dense');
      expect(urbanVariant.meta?.priority).toBe(3);
      expect(urbanVariant.meta?.inherits_from).toBe(baseTemplate.meta?.description);

      // Check configuration
      expect(urbanVariant.configuration['EUtranCellFDD'].qRxLevMin).toBe(-125);
      expect(urbanVariant.configuration['EUtranCellFDD'].qQualMin).toBe(-15);
      expect(urbanVariant.configuration['EUtranCellFDD'].ul256qamEnabled).toBe(true);
      expect(urbanVariant.configuration['EUtranCellFDD'].dl256qamEnabled).toBe(true);

      // Check capacity functions
      expect(urbanVariant.configuration['CapacityFunction']).toBeDefined();
      expect(urbanVariant.configuration['CapacityFunction'].capacityOptimizationEnabled).toBe(true);
      expect(urbanVariant.configuration['CapacityFunction'].icicEnabled).toBe(true);

      // Check custom functions
      expect(urbanVariant.custom).toHaveLength(2);
      expect(urbanVariant.custom?.map(f => f.name)).toContain('optimizeUrbanCoverage');
      expect(urbanVariant.custom?.map(f => f.name)).toContain('calculateInterference');

      // Check conditions
      expect(urbanVariant.conditions).toHaveProperty('highDensityCondition');
      expect(urbanVariant.conditions).toHaveProperty('interferenceCondition');

      // Check evaluations
      expect(urbanVariant.evaluations).toHaveProperty('urbanOptimization');
    });

    test('should generate suburban urban variant correctly', () => {
      const urbanVariant = urbanGenerator.generateUrbanVariant(baseTemplate, 'suburban');

      expect(urbanVariant.configuration['EUtranCellFDD'].qRxLevMin).toBe(-130);
      expect(urbanVariant.configuration['EUtranCellFDD'].qQualMin).toBe(-17);
      expect(urbanVariant.configuration['EUtranCellFDD'].ul256qamEnabled).toBe(false);
      expect(urbanVariant.configuration['EUtranCellFDD'].dl256qamEnabled).toBe(true);

      expect(urbanVariant.configuration['CapacityFunction'].capacityOptimizationEnabled).toBe(false);
      expect(urbanVariant.configuration['CapacityFunction'].loadBalancingEnabled).toBe(true);
      expect(urbanVariant.configuration['CapacityFunction'].icicEnabled).toBe(false);
    });

    test('should generate rural urban variant correctly', () => {
      const urbanVariant = urbanGenerator.generateUrbanVariant(baseTemplate, 'rural');

      expect(urbanVariant.configuration['EUtranCellFDD'].qRxLevMin).toBe(-135);
      expect(urbanVariant.configuration['EUtranCellFDD'].qQualMin).toBe(-19);
      expect(urbanVariant.configuration['EUtranCellFDD'].ul256qamEnabled).toBe(false);
      expect(urbanVariant.configuration['EUtranCellFDD'].dl256qamEnabled).toBe(false);
      expect(urbanVariant.configuration['EUtranCellFDD'].antennaPorts).toBe(1);
      expect(urbanVariant.configuration['EUtranCellFDD'].transmissionMode).toBe(1);

      expect(urbanVariant.configuration['CapacityFunction'].capacityOptimizationEnabled).toBe(false);
      expect(urbanVariant.configuration['CapacityFunction'].loadBalancingEnabled).toBe(false);
      expect(urbanVariant.configuration['CapacityFunction'].icicEnabled).toBe(false);
    });

    test('should validate urban variant correctly', () => {
      const validVariant = urbanGenerator.generateUrbanVariant(baseTemplate, 'dense');
      const validation = urbanGenerator.validateUrbanVariant(validVariant);

      expect(validation.isValid).toBe(true);
      expect(validation.errors).toHaveLength(0);
    });

    test('should detect invalid urban variants', () => {
      const invalidVariant: RTBTemplate = {
        meta: {
          version: '1.0.0',
          description: 'Invalid urban variant',
          tags: ['invalid'], // Missing urban tag
          priority: 3
        },
        configuration: {
          'EUtranCellFDD': {
            'qRxLevMin': -100, // Too high for urban
            'referenceSignalPower': 10 // Too low for urban
          }
          // Missing CapacityFunction
        }
      };

      const validation = urbanGenerator.validateUrbanVariant(invalidVariant);

      expect(validation.isValid).toBe(false);
      expect(validation.errors.length).toBeGreaterThan(0);
      expect(validation.errors).toContain('Template must include urban tag');
      expect(validation.errors).toContain('Urban variant must include CapacityFunction configuration');
      expect(validation.errors).toContain('Urban qRxLevMin should be <= -120 for dense environments');
      expect(validation.errors).toContain('Urban referenceSignalPower should be >= 15');
    });
  });

  describe('Mobility Variant Generator', () => {
    test('should generate highway mobility variant correctly', () => {
      const mobilityVariant = mobilityGenerator.generateMobilityVariant(baseTemplate, 'highway');

      // Check metadata
      expect(mobilityVariant.meta?.description).toContain('highway');
      expect(mobilityVariant.meta?.tags).toContain('mobility');
      expect(mobilityVariant.meta?.tags).toContain('highway');
      expect(mobilityVariant.meta?.priority).toBe(5);

      // Check configuration
      expect(mobilityVariant.configuration['EUtranCellFDD'].qRxLevMin).toBe(-120);
      expect(mobilityVariant.configuration['EUtranCellFDD'].handoverHysteresis).toBe(4);
      expect(mobilityVariant.configuration['EUtranCellFDD'].timeToTrigger).toBe(640);
      expect(mobilityVariant.configuration['EUtranCellFDD'].cellIndividualOffset).toBe(1);

      // Check mobility functions
      expect(mobilityVariant.configuration['MobilityFunction']).toBeDefined();
      expect(mobilityVariant.configuration['MobilityFunction'].highSpeedOptimization).toBe(true);
      expect(mobilityVariant.configuration['MobilityFunction'].pingPongReduction).toBe(true);
      expect(mobilityVariant.configuration['MobilityFunction'].speedAdaptiveHysteresis).toBe(true);

      // Check custom functions
      expect(mobilityVariant.custom).toHaveLength(3);
      expect(mobilityVariant.custom?.map(f => f.name)).toEqual([
        'optimizeHandover',
        'calculateCellBorder',
        'pingPongReduction'
      ]);

      // Check conditions
      expect(mobilityVariant.conditions).toHaveProperty('speedAdaptiveCondition');
      expect(mobilityVariant.conditions).toHaveProperty('highSpeedCondition');
      expect(mobilityVariant.conditions).toHaveProperty('pingPongCondition');

      // Check evaluations
      expect(mobilityVariant.evaluations).toHaveProperty('cellBorderCalculation');
      expect(mobilityVariant.evaluations).toHaveProperty('mobilityScore');
    });

    test('should generate urban mobility variant correctly', () => {
      const mobilityVariant = mobilityGenerator.generateMobilityVariant(baseTemplate, 'urban');

      expect(mobilityVariant.configuration['EUtranCellFDD'].qRxLevMin).toBe(-122);
      expect(mobilityVariant.configuration['EUtranCellFDD'].handoverHysteresis).toBe(6);
      expect(mobilityVariant.configuration['EUtranCellFDD'].timeToTrigger).toBe(480);
      expect(mobilityVariant.configuration['EUtranCellFDD'].cellIndividualOffset).toBe(2);

      expect(mobilityVariant.configuration['MobilityFunction'].highSpeedOptimization).toBe(false);
      expect(mobilityVariant.configuration['MobilityFunction'].pingPongReduction).toBe(true);
    });

    test('should generate pedestrian mobility variant correctly', () => {
      const mobilityVariant = mobilityGenerator.generateMobilityVariant(baseTemplate, 'pedestrian');

      expect(mobilityVariant.configuration['EUtranCellFDD'].qRxLevMin).toBe(-125);
      expect(mobilityVariant.configuration['EUtranCellFDD'].handoverHysteresis).toBe(8);
      expect(mobilityVariant.configuration['EUtranCellFDD'].timeToTrigger).toBe(320);
      expect(mobilityVariant.configuration['EUtranCellFDD'].cellIndividualOffset).toBe(3);

      expect(mobilityVariant.configuration['MobilityFunction'].mobilityOptimizationEnabled).toBe(false);
      expect(mobilityVariant.configuration['MobilityFunction'].speedAdaptiveHysteresis).toBe(false);
      expect(mobilityVariant.configuration['MobilityFunction'].pingPongReduction).toBe(false);
    });

    test('should validate mobility variant correctly', () => {
      const validVariant = mobilityGenerator.generateMobilityVariant(baseTemplate, 'highway');
      const validation = mobilityGenerator.validateMobilityVariant(validVariant);

      expect(validation.isValid).toBe(true);
      expect(validation.errors).toHaveLength(0);
    });

    test('should detect invalid mobility variants', () => {
      const invalidVariant: RTBTemplate = {
        meta: {
          version: '1.0.0',
          description: 'Invalid mobility variant',
          tags: ['invalid'], // Missing mobility tag
          priority: 5
        },
        configuration: {
          'EUtranCellFDD': {
            // Missing handoverHysteresis and timeToTrigger
          }
          // Missing MobilityFunction
        }
      };

      const validation = mobilityGenerator.validateMobilityVariant(invalidVariant);

      expect(validation.isValid).toBe(false);
      expect(validation.errors.length).toBeGreaterThan(0);
      expect(validation.errors).toContain('Template must include mobility tag');
      expect(validation.errors).toContain('Mobility variant must include MobilityFunction configuration');
      expect(validation.errors).toContain('Mobility variant must include handoverHysteresis parameter');
      expect(validation.errors).toContain('Mobility variant must include timeToTrigger parameter');
    });
  });

  describe('AgentDB Variant Generator', () => {
    test('should generate maximum consciousness variant correctly', () => {
      const agentdbVariant = agentdbGenerator.generateAgentDBVariant(baseTemplate, 'maximum');

      // Check metadata
      expect(agentdbVariant.meta?.description).toContain('maximum');
      expect(agentdbVariant.meta?.tags).toContain('agentdb');
      expect(agentdbVariant.meta?.tags).toContain('cognitive');
      expect(agentdbVariant.meta?.tags).toContain('maximum');
      expect(agentdbVariant.meta?.priority).toBe(9);
      expect(Array.isArray(agentdbVariant.meta?.inherits_from)).toBe(true);

      // Check configuration
      expect(agentdbVariant.configuration['EUtranCellFDD'].qRxLevMin).toBe(-112);
      expect(agentdbVariant.configuration['EUtranCellFDD'].temporalReasoningDepth).toBe(1000);
      expect(agentdbVariant.configuration['EUtranCellFDD'].cognitiveOptimizationEnabled).toBe(true);

      // Check AgentDB functions
      expect(agentdbVariant.configuration['AgentDBFunction']).toBeDefined();
      expect(agentdbVariant.configuration['AgentDBFunction'].vectorSearchEnabled).toBe(true);
      expect(agentdbVariant.configuration['AgentDBFunction'].memoryPatternsEnabled).toBe(true);
      expect(agentdbVariant.configuration['AgentDBFunction'].temporalExpansionEnabled).toBe(true);
      expect(agentdbVariant.configuration['AgentDBFunction'].quicSyncEnabled).toBe(true);
      expect(agentdbVariant.configuration['AgentDBFunction'].neuralAcceleration).toBe(true);

      // Check Cognitive functions
      expect(agentdbVariant.configuration['CognitiveFunction']).toBeDefined();
      expect(agentdbVariant.configuration['CognitiveFunction'].consciousnessLevel).toBe('maximum');
      expect(agentdbVariant.configuration['CognitiveFunction'].strangeLoopEnabled).toBe(true);
      expect(agentdbVariant.configuration['CognitiveFunction'].autonomousLearning).toBe(true);
      expect(agentdbVariant.configuration['CognitiveFunction'].cognitiveConsciousness).toBe(true);

      // Check custom functions
      expect(agentdbVariant.custom).toHaveLength(4);
      expect(agentdbVariant.custom?.map(f => f.name)).toEqual([
        'cognitiveOptimization',
        'temporalReasoning',
        'strangeLoopOptimization',
        'agentdbVectorSearch'
      ]);

      // Check conditions
      expect(agentdbVariant.conditions).toHaveProperty('cognitiveCondition');
      expect(agentdbVariant.conditions).toHaveProperty('temporalCondition');
      expect(agentdbVariant.conditions).toHaveProperty('strangeLoopCondition');
      expect(agentdbVariant.conditions).toHaveProperty('agentdbCondition');

      // Check evaluations
      expect(agentdbVariant.evaluations).toHaveProperty('cognitiveOptimization');
      expect(agentdbVariant.evaluations).toHaveProperty('temporalInsights');
      expect(agentdbVariant.evaluations).toHaveProperty('strangeLoopAnalysis');
      expect(agentdbVariant.evaluations).toHaveProperty('consciousnessLevel');
      expect(agentdbVariant.evaluations).toHaveProperty('agentdbPerformanceScore');
    });

    test('should generate advanced consciousness variant correctly', () => {
      const agentdbVariant = agentdbGenerator.generateAgentDBVariant(baseTemplate, 'advanced');

      expect(agentdbVariant.configuration['EUtranCellFDD'].qRxLevMin).toBe(-115);
      expect(agentdbVariant.configuration['EUtranCellFDD'].temporalReasoningDepth).toBe(500);
      expect(agentdbVariant.configuration['EUtranCellFDD'].learningRate).toBe(0.2);

      expect(agentdbVariant.configuration['CognitiveFunction'].consciousnessLevel).toBe('enhanced');
      expect(agentdbVariant.configuration['CognitiveFunction'].strangeLoopEnabled).toBe(true);
      expect(agentdbVariant.configuration['CognitiveFunction'].cognitiveConsciousness).toBeUndefined();

      expect(agentdbVariant.configuration['AgentDBFunction'].neuralAcceleration).toBeUndefined();
      expect(agentdbVariant.configuration['AgentDBFunction'].memoryPatternsEnabled).toBe(true);
    });

    test('should generate basic consciousness variant correctly', () => {
      const agentdbVariant = agentdbGenerator.generateAgentDBVariant(baseTemplate, 'basic');

      expect(agentdbVariant.configuration['EUtranCellFDD'].qRxLevMin).toBe(-118);
      expect(agentdbVariant.configuration['EUtranCellFDD'].temporalReasoningDepth).toBe(100);
      expect(agentdbVariant.configuration['EUtranCellFDD'].learningRate).toBe(0.1);

      expect(agentdbVariant.configuration['CognitiveFunction'].consciousnessLevel).toBe('basic');
      expect(agentdbVariant.configuration['CognitiveFunction'].strangeLoopEnabled).toBe(false);
      expect(agentdbVariant.configuration['CognitiveFunction'].autonomousLearning).toBe(false);

      expect(agentdbVariant.configuration['AgentDBFunction'].vectorSearchEnabled).toBe(true);
      expect(agentdbVariant.configuration['AgentDBFunction'].memoryPatternsEnabled).toBe(false);
      expect(agentdbVariant.configuration['AgentDBFunction'].temporalExpansionEnabled).toBe(false);
    });

    test('should validate AgentDB variant correctly', () => {
      const validVariant = agentdbGenerator.generateAgentDBVariant(baseTemplate, 'maximum');
      const validation = agentdbGenerator.validateAgentDBVariant(validVariant);

      expect(validation.isValid).toBe(true);
      expect(validation.errors).toHaveLength(0);
    });

    test('should detect invalid AgentDB variants', () => {
      const invalidVariant: RTBTemplate = {
        meta: {
          version: '1.0.0',
          description: 'Invalid AgentDB variant',
          tags: ['invalid'], // Missing agentdb tag
          priority: 9
        },
        configuration: {
          'EUtranCellFDD': {
            // Missing cognitiveOptimizationEnabled and temporalReasoningDepth
          }
          // Missing AgentDBFunction and CognitiveFunction
        }
      };

      const validation = agentdbGenerator.validateAgentDBVariant(invalidVariant);

      expect(validation.isValid).toBe(false);
      expect(validation.errors.length).toBeGreaterThan(0);
      expect(validation.errors).toContain('Template must include agentdb tag');
      expect(validation.errors).toContain('AgentDB variant must include AgentDBFunction configuration');
      expect(validation.errors).toContain('AgentDB variant must include CognitiveFunction configuration');
      expect(validation.errors).toContain('AgentDB variant must include cognitiveOptimizationEnabled');
      expect(validation.errors).toContain('AgentDB variant must include temporalReasoningDepth');
    });
  });

  describe('Variant Generation Integration', () => {
    test('should create complete inheritance chain correctly', () => {
      // Base -> Urban -> Mobility -> AgentDB
      const urbanVariant = urbanGenerator.generateUrbanVariant(baseTemplate, 'dense');
      const mobilityVariant = mobilityGenerator.generateMobilityVariant(urbanVariant, 'highway');
      const agentdbVariant = agentdbGenerator.generateAgentDBVariant(mobilityVariant, 'maximum');

      // Check inheritance chain
      expect(agentdbVariant.meta?.inherits_from).toContain(mobilityVariant.meta?.description);
      expect(agentdbVariant.meta?.inherits_from).toContain('urban');

      // Check that all configurations are present
      expect(agentdbVariant.configuration['EUtranCellFDD']).toBeDefined();
      expect(agentdbVariant.configuration['CapacityFunction']).toBeDefined(); // From urban
      expect(agentdbVariant.configuration['MobilityFunction']).toBeDefined(); // From mobility
      expect(agentdbVariant.configuration['AgentDBFunction']).toBeDefined(); // From agentdb
      expect(agentdbVariant.configuration['CognitiveFunction']).toBeDefined(); // From agentdb

      // Check that highest priority values are used
      expect(agentdbVariant.configuration['EUtranCellFDD'].qRxLevMin).toBe(-112); // AgentDB override
      expect(agentdbVariant.configuration['EUtranCellFDD'].handoverHysteresis).toBe(4); // From mobility
      expect(agentdbVariant.configuration['EUtranCellFDD'].ul256qamEnabled).toBe(true); // From urban

      // Check that all custom functions are present
      expect(agentdbVariant.custom?.length).toBeGreaterThan(0);
      const functionNames = agentdbVariant.custom?.map(f => f.name);
      expect(functionNames).toContain('optimizeUrbanCoverage'); // From urban
      expect(functionNames).toContain('optimizeHandover'); // From mobility
      expect(functionNames).toContain('cognitiveOptimization'); // From agentdb
    });

    test('should handle different variant combinations', () => {
      const combinations = [
        { urban: 'dense', mobility: 'highway', agentdb: 'maximum' },
        { urban: 'suburban', mobility: 'urban', agentdb: 'advanced' },
        { urban: 'rural', mobility: 'pedestrian', agentdb: 'basic' }
      ];

      for (const combo of combinations) {
        const urbanVariant = urbanGenerator.generateUrbanVariant(baseTemplate, combo.urban as any);
        const mobilityVariant = mobilityGenerator.generateMobilityVariant(urbanVariant, combo.mobility as any);
        const agentdbVariant = agentdbGenerator.generateAgentDBVariant(mobilityVariant, combo.agentdb as any);

        // Validate each variant
        const urbanValidation = urbanGenerator.validateUrbanVariant(urbanVariant);
        const mobilityValidation = mobilityGenerator.validateMobilityVariant(mobilityVariant);
        const agentdbValidation = agentdbGenerator.validateAgentDBVariant(agentdbVariant);

        expect(urbanValidation.isValid).toBe(true);
        expect(mobilityValidation.isValid).toBe(true);
        expect(agentdbValidation.isValid).toBe(true);

        // Check that configuration values make sense for the combination
        expect(agentdbVariant.configuration['EUtranCellFDD'].qRxLevMin).toBeDefined();
        expect(agentdbVariant.configuration['EUtranCellFDD'].cognitiveOptimizationEnabled).toBe(true);
      }
    });

    test('should maintain consistency across variant parameters', () => {
      // Generate all urban variants
      const denseUrban = urbanGenerator.generateUrbanVariant(baseTemplate, 'dense');
      const suburbanUrban = urbanGenerator.generateUrbanVariant(baseTemplate, 'suburban');
      const ruralUrban = urbanGenerator.generateUrbanVariant(baseTemplate, 'rural');

      // Check parameter consistency
      const urbanVariants = [denseUrban, suburbanUrban, ruralUrban];

      for (const variant of urbanVariants) {
        const validation = urbanGenerator.validateUrbanVariant(variant);
        expect(validation.isValid).toBe(true);

        // Check that required parameters exist
        expect(variant.configuration['EUtranCellFDD']).toBeDefined();
        expect(variant.configuration['CapacityFunction']).toBeDefined();

        // Check parameter ranges are logical
        const cellConfig = variant.configuration['EUtranCellFDD'];
        expect(cellConfig.qRxLevMin).toBeGreaterThanOrEqual(-140);
        expect(cellConfig.qRxLevMin).toBeLessThanOrEqual(-120);
        expect(cellConfig.referenceSignalPower).toBeGreaterThanOrEqual(15);
        expect(cellConfig.referenceSignalPower).toBeLessThanOrEqual(20);
      }

      // Verify dense is more aggressive than suburban, which is more aggressive than rural
      expect(denseUrban.configuration['EUtranCellFDD'].qRxLevMin).toBeGreaterThan(suburbanUrban.configuration['EUtranCellFDD'].qRxLevMin);
      expect(suburbanUrban.configuration['EUtranCellFDD'].qRxLevMin).toBeGreaterThan(ruralUrban.configuration['EUtranCellFDD'].qRxLevMin);
    });
  });

  describe('Performance and Scalability', () => {
    test('should generate variants efficiently', async () => {
      const startTime = performance.now();

      // Generate multiple variants
      const variants: RTBTemplate[] = [];

      for (let i = 0; i < 10; i++) {
        const urbanVariant = urbanGenerator.generateUrbanVariant(baseTemplate, 'dense');
        const mobilityVariant = mobilityGenerator.generateMobilityVariant(urbanVariant, 'highway');
        const agentdbVariant = agentdbGenerator.generateAgentDBVariant(mobilityVariant, 'maximum');

        variants.push(agentdbVariant);
      }

      const endTime = performance.now();
      const generationTime = endTime - startTime;

      // Should generate 30 variants (10 * 3) quickly
      expect(generationTime).toBeLessThan(1000); // < 1 second
      expect(variants).toHaveLength(10);

      // Validate all variants
      for (const variant of variants) {
        const validation = agentdbGenerator.validateAgentDBVariant(variant);
        expect(validation.isValid).toBe(true);
      }
    });

    test('should handle complex variant scenarios', () => {
      // Generate a complete complex variant
      const urbanVariant = urbanGenerator.generateUrbanVariant(baseTemplate, 'dense');
      const mobilityVariant = mobilityGenerator.generateMobilityVariant(urbanVariant, 'highway');
      const agentdbVariant = agentdbGenerator.generateAgentDBVariant(mobilityVariant, 'maximum');

      // Check that complex conditions and evaluations are preserved
      expect(Object.keys(agentdbVariant.conditions || {})).toContain('highDensityCondition'); // From urban
      expect(Object.keys(agentdbVariant.conditions || {})).toContain('speedAdaptiveCondition'); // From mobility
      expect(Object.keys(agentdbVariant.conditions || {})).toContain('cognitiveCondition'); // From agentdb

      expect(Object.keys(agentdbVariant.evaluations || {})).toContain('urbanOptimization'); // From urban
      expect(Object.keys(agentdbVariant.evaluations || {})).toContain('mobilityScore'); // From mobility
      expect(Object.keys(agentdbVariant.evaluations || {})).toContain('consciousnessLevel'); // From agentdb

      // Check that the variant is still valid
      const validation = agentdbGenerator.validateAgentDBVariant(agentdbVariant);
      expect(validation.isValid).toBe(true);
    });
  });
});