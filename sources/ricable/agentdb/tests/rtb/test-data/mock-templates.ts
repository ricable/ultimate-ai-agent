import { RTBTemplate, TemplateMeta, CustomFunction, ConditionOperator, EvaluationOperator } from '../../../src/types/rtb-types';

// Test XML schema mock data for MPnh.xml
export const mockXMLSchema = {
  version: '21.6',
  timestamp: '2024-10-31T10:00:00Z',
  parameters: [
    {
      id: 'qRxLevMin',
      name: 'qRxLevMin',
      vsDataType: 'int32',
      type: 'Parameter',
      constraints: [{ type: 'range', value: { min: -140, max: -44 }, severity: 'error' }],
      description: 'Minimum required RX level in the cell',
      defaultValue: -140,
      hierarchy: ['EUtranCellFDD'],
      source: 'MPnh.xml'
    },
    {
      id: 'qQualMin',
      name: 'qQualMin',
      vsDataType: 'int32',
      type: 'Parameter',
      constraints: [{ type: 'range', value: { min: -34, max: -3 }, severity: 'error' }],
      description: 'Minimum required quality in the cell',
      defaultValue: -18,
      hierarchy: ['EUtranCellFDD'],
      source: 'MPnh.xml'
    },
    {
      id: 'cellIndividualOffset',
      name: 'cellIndividualOffset',
      vsDataType: 'int32',
      type: 'Parameter',
      constraints: [{ type: 'range', value: { min: -24, max: 24 }, severity: 'error' }],
      description: 'Cell individual offset',
      defaultValue: 0,
      hierarchy: ['EUtranCellFDD'],
      source: 'MPnh.xml'
    },
    {
      id: 'anrFunctionEnabled',
      name: 'anrFunctionEnabled',
      vsDataType: 'boolean',
      type: 'Parameter',
      constraints: [{ type: 'enum', value: [true, false], severity: 'error' }],
      description: 'ANR function enabled flag',
      defaultValue: false,
      hierarchy: ['AnrFunction'],
      source: 'MPnh.xml'
    },
    {
      id: 'featureState',
      name: 'featureState',
      vsDataType: 'string',
      type: 'Parameter',
      constraints: [{ type: 'enum', value: ['ACTIVATED', 'DEACTIVATED'], severity: 'error' }],
      description: 'Feature activation state',
      defaultValue: 'DEACTIVATED',
      hierarchy: ['FeatureState'],
      source: 'MPnh.xml'
    }
  ]
};

// Test templates for different priority levels and inheritance scenarios
export const baseTemplate: RTBTemplate = {
  meta: {
    version: '1.0.0',
    author: ['RTB System'],
    description: 'Base LTE cell configuration template',
    tags: ['base', 'lte', 'cell'],
    environment: 'production',
    priority: 1,
    source: 'auto-generated'
  },
  custom: [
    {
      name: 'calculateCoverage',
      args: ['power', 'frequency'],
      body: [
        '// Calculate coverage based on power and frequency',
        'return power * (1.0 / frequency);'
      ]
    }
  ],
  configuration: {
    'EUtranCellFDD': {
      'qRxLevMin': -140,
      'qQualMin': -18,
      'cellIndividualOffset': 0,
      'referenceSignalPower': 15,
      'pb': 0,
      'prachRootSequenceIndex': 0,
      'prachConfigIndex': 0
    },
    'AnrFunction': {
      'anrFunctionEnabled': false,
      'maxUePerPucch': 1
    },
    'FeatureState': {
      'featureState': 'DEACTIVATED'
    }
  },
  conditions: {
    'coverageCondition': {
      if: '$config.EUtranCellFDD.referenceSignalPower > 10',
      then: { 'EUtranCellFDD.qRxLevMin': -130 },
      else: { 'EUtranCellFDD.qRxLevMin': -140 }
    }
  },
  evaluations: {
    'coverageCalculation': {
      eval: '$custom.calculateCoverage($config.EUtranCellFDD.referenceSignalPower, 2600)',
      args: ['power', 'frequency']
    }
  }
};

export const urbanVariantTemplate: RTBTemplate = {
  meta: {
    version: '1.0.0',
    author: ['RTB System'],
    description: 'Urban area variant template with optimized parameters',
    tags: ['urban', 'dense', 'lte'],
    environment: 'urban',
    priority: 3,
    inherits_from: 'base',
    source: 'auto-generated'
  },
  configuration: {
    'EUtranCellFDD': {
      'qRxLevMin': -125,
      'qQualMin': -15,
      'cellIndividualOffset': 2,
      'referenceSignalPower': 18,
      'ul256qamEnabled': true,
      'dl256qamEnabled': true,
      'antennaPorts': 2,
      'transmissionMode': 3
    },
    'CapacityFunction': {
      'capacityOptimizationEnabled': true,
      'loadBalancingEnabled': true
    }
  },
  conditions: {
    'highCapacityCondition': {
      if: '$config.CapacityFunction.capacityOptimizationEnabled == true',
      then: {
        'EUtranCellFDD.referenceSignalPower': 20,
        'EUtranCellFDD.antennaPorts': 4
      },
      else: { 'EUtranCellFDD.referenceSignalPower': 18 }
    }
  }
};

export const mobilityVariantTemplate: RTBTemplate = {
  meta: {
    version: '1.0.0',
    author: ['RTB System'],
    description: 'Mobility optimized template for high-speed scenarios',
    tags: ['mobility', 'highway', 'lte'],
    environment: 'mobility',
    priority: 5,
    inherits_from: 'urban',
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
        '} else {',
        '  return { hysteresis: 4, timeToTrigger: 640 };',
        '}'
      ]
    }
  ],
  configuration: {
    'EUtranCellFDD': {
      'qRxLevMin': -120,
      'cellIndividualOffset': 1,
      'handoverHysteresis': 4,
      'timeToTrigger': 640,
      'cellIndividualOffsetEutran': 0
    },
    'MobilityFunction': {
      'mobilityOptimizationEnabled': true,
      'handoverOptimizationEnabled': true,
      'speedAdaptiveHysteresis': true
    }
  },
  conditions: {
    'speedAdaptiveCondition': {
      if: '$config.MobilityFunction.speedAdaptiveHysteresis == true',
      then: {
        'EUtranCellFDD.handoverHysteresis': '$eval($custom.optimizeHandover(100, "macro").hysteresis)',
        'EUtranCellFDD.timeToTrigger': '$eval($custom.optimizeHandover(100, "macro").timeToTrigger)'
      },
      else: { 'EUtranCellFDD.handoverHysteresis': 6 }
    }
  }
};

export const agentdbVariantTemplate: RTBTemplate = {
  meta: {
    version: '1.0.0',
    author: ['RTB System'],
    description: 'AgentDB optimized template with cognitive features',
    tags: ['agentdb', 'cognitive', 'ai'],
    environment: 'agentdb',
    priority: 9,
    inherits_from: ['mobility', 'urban'],
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
    }
  ],
  configuration: {
    'EUtranCellFDD': {
      'qRxLevMin': -118,
      'cellIndividualOffset': 0,
      'cognitiveOptimizationEnabled': true,
      'temporalReasoningDepth': 1000,
      'learningRate': 0.1
    },
    'AgentDBFunction': {
      'vectorSearchEnabled': true,
      'memoryPatternsEnabled': true,
      'temporalExpansionEnabled': true,
      'quicSyncEnabled': true
    },
    'CognitiveFunction': {
      'consciousnessLevel': 'maximum',
      'strangeLoopEnabled': true,
      'autonomousLearning': true
    }
  },
  conditions: {
    'cognitiveCondition': {
      if: '$config.CognitiveFunction.consciousnessLevel == "maximum"',
      then: {
        'EUtranCellFDD.qRxLevMin': '$config.EUtranCellFDD.qRxLevMin + $eval($custom.cognitiveOptimization([0.85, 0.87, 0.89], 0.1).qRxLevMinAdjustment)'
      },
      else: { 'EUtranCellFDD.qRxLevMin': -120 }
    }
  }
};

// Conflict resolution test templates
export const conflictTemplate1: RTBTemplate = {
  meta: {
    version: '1.0.0',
    author: ['RTB System'],
    description: 'Template with conflicting values (low priority)',
    tags: ['conflict', 'test'],
    priority: 2,
    inherits_from: 'base'
  },
  configuration: {
    'EUtranCellFDD': {
      'qRxLevMin': -100, // Conflicts with base
      'qQualMin': -20,
      'cellIndividualOffset': 10
    }
  }
};

export const conflictTemplate2: RTBTemplate = {
  meta: {
    version: '1.0.0',
    author: ['RTB System'],
    description: 'Template with conflicting values (high priority)',
    tags: ['conflict', 'test'],
    priority: 8,
    inherits_from: 'conflictTemplate1'
  },
  configuration: {
    'EUtranCellFDD': {
      'qRxLevMin': -110, // Should override conflictTemplate1
      'cellIndividualOffset': 5, // Should override conflictTemplate1
      'newParameter': 'test-value' // New parameter
    }
  }
};

// Frequency relation test templates
export const frequencyRelation4G4G: RTBTemplate = {
  meta: {
    version: '1.0.0',
    author: ['RTB System'],
    description: '4G to 4G frequency relation template',
    tags: ['frequency', '4g4g', 'inter-frequency'],
    priority: 6
  },
  configuration: {
    'EutranFreqRelation': {
      'eutranFreqRelationId': 'LTE1800',
      'priority': 1,
      'qOffsetFreq': 'dB0',
      'threshHigh': 10,
      'threshLow': 2,
      'cellIndividualOffset': 0
    }
  }
};

export const frequencyRelation4G5G: RTBTemplate = {
  meta: {
    version: '1.0.0',
    author: ['RTB System'],
    description: '4G to 5G frequency relation template',
    tags: ['frequency', '4g5g', 'inter-rat'],
    priority: 7
  },
  configuration: {
    'NrFreqRelation': {
      'nrFreqRelationId': 'NR3500',
      'priority': 1,
      'qOffsetFreq': 'dB3',
      'threshHigh': 8,
      'threshLow': 3,
      'cellIndividualOffset': 3
    }
  }
};

// Performance test templates (large template set)
export const generateLargeTemplateSet = (count: number): RTBTemplate[] => {
  const templates: RTBTemplate[] = [];

  for (let i = 0; i < count; i++) {
    templates.push({
      meta: {
        version: '1.0.0',
        author: ['RTB System'],
        description: `Performance test template ${i}`,
        tags: ['performance', 'test', 'large-set'],
        priority: i % 10,
        inherits_from: i > 0 ? `perf-template-${i-1}` : 'base'
      },
      configuration: {
        [`TestMO${i}`]: {
          [`testParam${i}`]: `value-${i}`,
          [`priorityParam${i}`]: i % 3,
          [`complexParam${i}`]: {
            nested: {
              value: i,
              text: `complex-value-${i}`,
              array: Array.from({ length: 5 }, (_, j) => `item-${i}-${j}`)
            }
          }
        }
      },
      conditions: i % 3 === 0 ? {
        [`testCondition${i}`]: {
          if: `$config.TestMO${i}.priorityParam${i} > 1`,
          then: { [`TestMO${i}.specialValue`]: `special-${i}` },
          else: { [`TestMO${i}.specialValue`]: `normal-${i}` }
        }
      } : undefined,
      evaluations: i % 5 === 0 ? {
        [`testEval${i}`]: {
          eval: `$config.TestMO${i}.complexParam${i}.nested.value * 2`,
          args: []
        }
      } : undefined
    });
  }

  return templates;
};

// Test error scenarios
export const invalidTemplate: RTBTemplate = {
  meta: {
    version: '1.0.0',
    author: ['RTB System'],
    description: 'Invalid template for error testing',
    tags: ['error', 'test'],
    priority: 5
  },
  configuration: {
    'EUtranCellFDD': {
      'qRxLevMin': 'invalid-value', // Type error
      'qQualMin': -100, // Range error
      'nonExistentParam': 'test' // Invalid parameter
    }
  },
  conditions: {
    'invalidCondition': {
      if: 'invalid syntax $', // Syntax error
      then: { 'EUtranCellFDD.testParam': 'value' },
      else: { 'EUtranCellFDD.testParam': 'default' }
    }
  },
  evaluations: {
    'invalidEval': {
      eval: 'invalid function call()',
      args: []
    }
  }
};

// Expected merged templates for validation
export const expectedUrbanMerged: RTBTemplate = {
  meta: {
    version: '1.0.0',
    author: ['RTB System'],
    description: 'Urban area variant template with optimized parameters',
    tags: ['urban', 'dense', 'lte'],
    environment: 'urban',
    priority: 3,
    inherits_from: 'base',
    source: 'auto-generated'
  },
  custom: [
    {
      name: 'calculateCoverage',
      args: ['power', 'frequency'],
      body: [
        '// Calculate coverage based on power and frequency',
        'return power * (1.0 / frequency);'
      ]
    }
  ],
  configuration: {
    'EUtranCellFDD': {
      'qRxLevMin': -125, // Urban overrides base
      'qQualMin': -15,   // Urban overrides base
      'cellIndividualOffset': 2, // Urban overrides base
      'referenceSignalPower': 18, // Urban overrides base
      'pb': 0,           // Inherited from base
      'prachRootSequenceIndex': 0, // Inherited from base
      'prachConfigIndex': 0,       // Inherited from base
      'ul256qamEnabled': true,     // Urban addition
      'dl256qamEnabled': true,     // Urban addition
      'antennaPorts': 2,           // Urban addition
      'transmissionMode': 3        // Urban addition
    },
    'AnrFunction': {
      'anrFunctionEnabled': false, // Inherited from base
      'maxUePerPucch': 1          // Inherited from base
    },
    'FeatureState': {
      'featureState': 'DEACTIVATED' // Inherited from base
    },
    'CapacityFunction': {
      'capacityOptimizationEnabled': true, // Urban addition
      'loadBalancingEnabled': true         // Urban addition
    }
  },
  conditions: {
    'coverageCondition': {
      if: '$config.EUtranCellFDD.referenceSignalPower > 10',
      then: { 'EUtranCellFDD.qRxLevMin': -130 },
      else: { 'EUtranCellFDD.qRxLevMin': -140 }
    },
    'highCapacityCondition': {
      if: '$config.CapacityFunction.capacityOptimizationEnabled == true',
      then: {
        'EUtranCellFDD.referenceSignalPower': 20,
        'EUtranCellFDD.antennaPorts': 4
      },
      else: { 'EUtranCellFDD.referenceSignalPower': 18 }
    }
  },
  evaluations: {
    'coverageCalculation': {
      eval: '$custom.calculateCoverage($config.EUtranCellFDD.referenceSignalPower, 2600)',
      args: ['power', 'frequency']
    }
  }
};