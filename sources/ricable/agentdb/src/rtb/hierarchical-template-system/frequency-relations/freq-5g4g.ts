/**
 * 5G4G Frequency Relations Templates (Priority 80)
 *
 * Comprehensive 5G to 4G fallback configuration templates for Ericsson RAN
 * including service continuity, seamless handover, and intelligent fallback triggers
 */

import type {
  Freq5G4GRelation,
  FrequencyRelationTemplate,
  TemplateParameter,
  ValidationRule,
  CmeditCommandTemplate,
  HandoverConfiguration,
  CapacitySharingParams,
  InterferenceSettings,
  FrequencyBand,
  FrequencyRelationMetrics
} from './freq-types';

/**
 * LTE frequency bands for fallback scenarios
 */
export const FALLBACK_LTE_BANDS: Record<number, FrequencyBand> = {
  1: {
    bandNumber: 1,
    frequencyRange: {
      uplink: { start: 1920, end: 1980 },
      downlink: { start: 2110, end: 2170 }
    },
    bandCategory: 'LTE',
    primaryUse: 'CAPACITY'
  },
  3: {
    bandNumber: 3,
    frequencyRange: {
      uplink: { start: 1710, end: 1785 },
      downlink: { start: 1805, end: 1880 }
    },
    bandCategory: 'LTE',
    primaryUse: 'CAPACITY'
  },
  7: {
    bandNumber: 7,
    frequencyRange: {
      uplink: { start: 2500, end: 2570 },
      downlink: { start: 2620, end: 2690 }
    },
    bandCategory: 'LTE',
    primaryUse: 'CAPACITY'
  },
  20: {
    bandNumber: 20,
    frequencyRange: {
      uplink: { start: 832, end: 862 },
      downlink: { start: 791, end: 821 }
    },
    bandCategory: 'LTE',
    primaryUse: 'COVERAGE'
  },
  28: {
    bandNumber: 28,
    frequencyRange: {
      uplink: { start: 703, end: 748 },
      downlink: { start: 758, end: 803 }
    },
    bandCategory: 'LTE',
    primaryUse: 'COVERAGE'
  }
};

/**
 * NR frequency bands for fallback scenarios
 */
export const FALLBACK_NR_BANDS: Record<number, FrequencyBand> = {
  78: {
    bandNumber: 78,
    frequencyRange: {
      uplink: { start: 3300, end: 3800 },
      downlink: { start: 3300, end: 3800 }
    },
    bandCategory: 'NR',
    primaryUse: 'CAPACITY'
  },
  41: {
    bandNumber: 41,
    frequencyRange: {
      uplink: { start: 2496, end: 2690 },
      downlink: { start: 2496, end: 2690 }
    },
    bandCategory: 'NR',
    primaryUse: 'CAPACITY'
  },
  77: {
    bandNumber: 77,
    frequencyRange: {
      uplink: { start: 3300, end: 4200 },
      downlink: { start: 3300, end: 4200 }
    },
    bandCategory: 'NR',
    primaryUse: 'CAPACITY'
  },
  28: {
    bandNumber: 28,
    frequencyRange: {
      uplink: { start: 703, end: 748 },
      downlink: { start: 758, end: 803 }
    },
    bandCategory: 'NR',
    primaryUse: 'COVERAGE'
  },
  71: {
    bandNumber: 71,
    frequencyRange: {
      uplink: { start: 663, end: 698 },
      downlink: { start: 617, end: 652 }
    },
    bandCategory: 'NR',
    primaryUse: 'COVERAGE'
  }
};

/**
 * Standard 5G4G fallback handover configuration
 */
export const STANDARD_5G4G_HANDOVER: HandoverConfiguration = {
  triggerType: 'A2',
  hysteresis: 2,
  timeToTrigger: 320,
  cellIndividualOffset: 0,
  freqSpecificOffset: 2,
  eventBasedConfig: {
    threshold1: -115
  },
  measurementConfig: {
    reportInterval: 240,
    maxReportCells: 8,
    reportAmount: '8'
  }
};

/**
 * Aggressive 5G4G fallback for service continuity
 */
export const AGGRESSIVE_5G4G_HANDOVER: HandoverConfiguration = {
  triggerType: 'A2',
  hysteresis: 1,
  timeToTrigger: 160,
  cellIndividualOffset: 2,
  freqSpecificOffset: 3,
  eventBasedConfig: {
    threshold1: -120
  },
  measurementConfig: {
    reportInterval: 120,
    maxReportCells: 16,
    reportAmount: '16'
  }
};

/**
 * Conservative 5G4G fallback for stability
 */
export const CONSERVATIVE_5G4G_HANDOVER: HandoverConfiguration = {
  triggerType: 'A2',
  hysteresis: 4,
  timeToTrigger: 640,
  cellIndividualOffset: 0,
  freqSpecificOffset: 1,
  eventBasedConfig: {
    threshold1: -110
  },
  measurementConfig: {
    reportInterval: 480,
    maxReportCells: 4,
    reportAmount: '4'
  }
};

/**
 * Service continuity capacity sharing
 */
export const SERVICE_CONTINUITY_CAPACITY_SHARING: CapacitySharingParams = {
  enabled: true,
  strategy: 'PRIORITY_BASED',
  loadBalancingThreshold: 90,
  maxCapacityRatio: 1.0,
  minGuaranteedCapacity: 0.0,
  dynamicRebalancing: false,
  rebalancingInterval: 0
};

/**
 * Fallback interference coordination settings
 */
export const FALLBACK_INTERFERENCE_SETTINGS: InterferenceSettings = {
  enabled: false,
  coordinationType: 'ICIC',
  interBandManagement: {
    almostBlankSubframes: false,
    crsPowerBoost: 0,
    powerControlCoordination: false
  },
  dynamicCoordination: false,
  coordinationInterval: 0
};

/**
 * Create base 5G4G frequency relation
 */
function createBase5G4GRelation(
  relationId: string,
  referenceBand: FrequencyBand,
  relatedBand: FrequencyBand,
  priority: number = 80
): Freq5G4GRelation {
  return {
    relationId,
    referenceFreq: referenceBand,
    relatedFreq: relatedBand,
    relationType: '5G4G',
    priority,
    adminState: 'UNLOCKED',
    operState: 'ENABLED',
    createdAt: new Date(),
    modifiedAt: new Date(),
    fallbackConfig: {
      fallbackTriggers: {
        nrCoverageThreshold: -120,
        serviceInterruptionTime: 5000,
        ueCapabilityFallback: true,
        networkCongestionFallback: false
      },
      fallbackHandover: {
        prepareFallbackTimeout: 2000,
        executeFallbackTimeout: 3000,
        fallbackPreparationRetryCount: 3,
        immediateFallbackAllowed: true
      },
      serviceContinuity: {
        sessionContinuity: true,
        ipAddressPreservation: true,
        qosPreservation: true
      },
      returnTo5G: {
        enabled: true,
        returnTriggers: {
          nrCoverageImprovement: -105,
          nrServiceQuality: 80,
          networkLoadImprovement: 60
        },
        returnEvaluationInterval: 30000,
        min5GStayTime: 30000
      }
    }
  };
}

/**
 * Template 1: Standard 5G4G Fallback (Priority 80)
 * Basic 5G to 4G fallback configuration with standard service continuity
 */
export const STANDARD_5G4G_TEMPLATE: FrequencyRelationTemplate = {
  templateId: 'FREQ_5G4G_STANDARD_001',
  templateName: 'Standard 5G4G Fallback Configuration',
  templateDescription: 'Basic 5G to 4G fallback configuration with standard service continuity and return to 5G',
  version: '1.0.0',
  templateType: '5G4G',
  priority: 80,
  baseConfig: createBase5G4GRelation(
    '5G4G_STANDARD',
    FALLBACK_NR_BANDS[78],
    FALLBACK_LTE_BANDS[3]
  ),
  parameters: [
    {
      name: 'nrBand',
      type: 'INTEGER',
      description: '5G NR band that will fallback to LTE',
      defaultValue: 78,
      allowedValues: [41, 77, 78, 28, 71],
      category: 'BASIC'
    },
    {
      name: 'lteFallbackBand',
      type: 'INTEGER',
      description: 'LTE band for fallback',
      defaultValue: 3,
      allowedValues: [1, 3, 7, 20, 28],
      category: 'BASIC'
    },
    {
      name: 'fallbackThreshold',
      type: 'INTEGER',
      description: 'NR coverage threshold for fallback in dBm',
      defaultValue: -120,
      constraints: { min: -130, max: -105 },
      category: 'BASIC'
    },
    {
      name: 'serviceContinuity',
      type: 'BOOLEAN',
      description: 'Enable service continuity during fallback',
      defaultValue: true,
      category: 'BASIC'
    },
    {
      name: 'returnTo5GEnabled',
      type: 'BOOLEAN',
      description: 'Enable automatic return to 5G',
      defaultValue: true,
      category: 'BASIC'
    },
    {
      name: 'returnTo5GThreshold',
      type: 'INTEGER',
      description: 'NR coverage threshold for return in dBm',
      defaultValue: -105,
      constraints: { min: -115, max: -95 },
      category: 'ADVANCED'
    }
  ],
  validationRules: [
    {
      name: 'valid_fallback_combination',
      description: 'Valid NR-LTE fallback combination required',
      type: 'CONSISTENCY',
      condition: 'isValid5G4GCombination(nrBand, lteFallbackBand)',
      action: 'ERROR'
    },
    {
      name: 'threshold_range',
      description: 'Fallback threshold must be within valid range',
      type: 'RANGE',
      condition: 'fallbackThreshold >= -130 && fallbackThreshold <= -105',
      action: 'ERROR'
    },
    {
      name: 'return_threshold_consistency',
      description: 'Return threshold should be better than fallback threshold',
      type: 'CONSISTENCY',
      condition: 'returnTo5GThreshold > fallbackThreshold',
      action: 'WARNING'
    }
  ],
  cmeditTemplates: [
    {
      commandName: 'configure_fallback',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${nrCellId} fallbackEnabled=true,fallbackBand=${lteFallbackBand},fallbackThreshold=${fallbackThreshold}',
      parameterMapping: {
        nodeId: 'nodeId',
        nrCellId: 'nrCellId',
        lteFallbackBand: 'lteFallbackBand',
        fallbackThreshold: 'fallbackThreshold'
      },
      description: 'Configure 5G to 4G fallback parameters'
    },
    {
      commandName: 'setup_service_continuity',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${nrCellId} serviceContinuity=${serviceContinuity},ipAddressPreservation=true,qosPreservation=true',
      parameterMapping: {
        nodeId: 'nodeId',
        nrCellId: 'nrCellId',
        serviceContinuity: 'serviceContinuity'
      },
      description: 'Configure service continuity during fallback'
    },
    {
      commandName: 'configure_return_to_5g',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${nrCellId} returnTo5GEnabled=${returnTo5GEnabled},returnTo5GThreshold=${returnTo5GThreshold}',
      parameterMapping: {
        nodeId: 'nodeId',
        nrCellId: 'nrCellId',
        returnTo5GEnabled: 'returnTo5GEnabled',
        returnTo5GThreshold: 'returnTo5GThreshold'
      },
      description: 'Configure automatic return to 5G parameters'
    }
  ]
};

/**
 * Template 2: Emergency 5G4G Fallback (Priority 80)
 * Emergency fallback configuration for critical service preservation
 */
export const EMERGENCY_5G4G_TEMPLATE: FrequencyRelationTemplate = {
  templateId: 'FREQ_5G4G_EMERGENCY_002',
  templateName: 'Emergency 5G4G Fallback Configuration',
  templateDescription: 'Emergency 5G to 4G fallback configuration for critical service preservation with immediate fallback',
  version: '1.0.0',
  templateType: '5G4G',
  priority: 80,
  baseConfig: Object.assign(
    createBase5G4GRelation('5G4G_EMERGENCY',
      FALLBACK_NR_BANDS[78],
      FALLBACK_LTE_BANDS[20]
    ),
    {
      handoverConfig: AGGRESSIVE_5G4G_HANDOVER,
      fallbackConfig: {
        fallbackTriggers: {
          nrCoverageThreshold: -125,
          serviceInterruptionTime: 2000,
          ueCapabilityFallback: true,
          networkCongestionFallback: true
        },
        fallbackHandover: {
          prepareFallbackTimeout: 1000,
          executeFallbackTimeout: 1500,
          fallbackPreparationRetryCount: 5,
          immediateFallbackAllowed: true
        },
        serviceContinuity: {
          sessionContinuity: true,
          ipAddressPreservation: true,
          qosPreservation: true
        },
        returnTo5G: {
          enabled: false,
          returnTriggers: {
            nrCoverageImprovement: -100,
            nrServiceQuality: 90,
            networkLoadImprovement: 70
          },
          returnEvaluationInterval: 60000,
          min5GStayTime: 120000
        }
      }
    }
  ),
  parameters: [
    {
      name: 'emergencyNrBand',
      type: 'INTEGER',
      description: '5G NR band requiring emergency fallback',
      defaultValue: 78,
      allowedValues: [41, 77, 78, 28, 71],
      category: 'BASIC'
    },
    {
      name: 'emergencyLteBand',
      type: 'INTEGER',
      description: 'LTE band for emergency fallback (coverage preferred)',
      defaultValue: 20,
      allowedValues: [20, 28],
      category: 'BASIC'
    },
    {
      name: 'emergencyFallbackThreshold',
      type: 'INTEGER',
      description: 'Emergency fallback threshold in dBm',
      defaultValue: -125,
      constraints: { min: -130, max: -110 },
      category: 'BASIC'
    },
    {
      name: 'immediateFallback',
      type: 'BOOLEAN',
      description: 'Enable immediate emergency fallback',
      defaultValue: true,
      category: 'BASIC'
    },
    {
      name: 'criticalServicePriority',
      type: 'BOOLEAN',
      description: 'Prioritize critical services during fallback',
      defaultValue: true,
      category: 'ADVANCED'
    },
    {
      name: 'maxFallbackTime',
      type: 'INTEGER',
      description: 'Maximum fallback execution time in milliseconds',
      defaultValue: 1500,
      constraints: { min: 500, max: 5000 },
      category: 'EXPERT'
    }
  ],
  validationRules: [
    {
      name: 'emergency_band_selection',
      description: 'Emergency fallback should use coverage LTE band',
      type: 'CONSISTENCY',
      condition: 'isCoverageBand(emergencyLteBand)',
      action: 'WARNING'
    },
    {
      name: 'emergency_threshold',
      description: 'Emergency threshold should be very conservative',
      type: 'CONSISTENCY',
      condition: 'emergencyFallbackThreshold <= -120',
      action: 'WARNING'
    },
    {
      name: 'fallback_time_limit',
      description: 'Emergency fallback should be fast',
      type: 'CONSISTENCY',
      condition: 'maxFallbackTime <= 3000',
      action: 'WARNING'
    }
  ],
  cmeditTemplates: [
    {
      commandName: 'configure_emergency_fallback',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${nrCellId} emergencyFallback=true,fallbackThreshold=${emergencyFallbackThreshold},immediateFallback=${immediateFallback}',
      parameterMapping: {
        nodeId: 'nodeId',
        nrCellId: 'nrCellId',
        emergencyFallbackThreshold: 'emergencyFallbackThreshold',
        immediateFallback: 'immediateFallback'
      },
      description: 'Configure emergency fallback parameters'
    },
    {
      commandName: 'setup_critical_service_priority',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${nrCellId} criticalServicePriority=${criticalServicePriority},maxFallbackTime=${maxFallbackTime}',
      parameterMapping: {
        nodeId: 'nodeId',
        nrCellId: 'nrCellId',
        criticalServicePriority: 'criticalServicePriority',
        maxFallbackTime: 'maxFallbackTime'
      },
      description: 'Setup critical service priority during emergency fallback'
    }
  ]
};

/**
 * Template 3: Coverage-Centric 5G4G Fallback (Priority 80)
 * Fallback configuration optimized for coverage scenarios
 */
export const COVERAGE_5G4G_TEMPLATE: FrequencyRelationTemplate = {
  templateId: 'FREQ_5G4G_COVERAGE_003',
  templateName: 'Coverage-Centric 5G4G Fallback Configuration',
  templateDescription: '5G to 4G fallback configuration optimized for coverage expansion scenarios',
  version: '1.0.0',
  templateType: '5G4G',
  priority: 80,
  baseConfig: Object.assign(
    createBase5G4GRelation('5G4G_COVERAGE',
      FALLBACK_NR_BANDS[28],
      FALLBACK_LTE_BANDS[20]
    ),
    {
      handoverConfig: CONSERVATIVE_5G4G_HANDOVER,
      fallbackConfig: {
        fallbackTriggers: {
          nrCoverageThreshold: -115,
          serviceInterruptionTime: 8000,
          ueCapabilityFallback: true,
          networkCongestionFallback: false
        },
        fallbackHandover: {
          prepareFallbackTimeout: 3000,
          executeFallbackTimeout: 4000,
          fallbackPreparationRetryCount: 2,
          immediateFallbackAllowed: false
        },
        serviceContinuity: {
          sessionContinuity: true,
          ipAddressPreservation: true,
          qosPreservation: true
        },
        returnTo5G: {
          enabled: true,
          returnTriggers: {
            nrCoverageImprovement: -108,
            nrServiceQuality: 75,
            networkLoadImprovement: 50
          },
          returnEvaluationInterval: 60000,
          min5GStayTime: 120000
        }
      }
    }
  ),
  parameters: [
    {
      name: 'coverageNrBand',
      type: 'INTEGER',
      description: '5G NR band in coverage scenario',
      defaultValue: 28,
      allowedValues: [28, 71],
      category: 'BASIC'
    },
    {
      name: 'coverageLteBand',
      type: 'INTEGER',
      description: 'LTE band for coverage fallback',
      defaultValue: 20,
      allowedValues: [20, 28],
      category: 'BASIC'
    },
    {
      name: 'coverageHysteresis',
      type: 'INTEGER',
      description: 'Coverage handover hysteresis in dB',
      defaultValue: 4,
      constraints: { min: 2, max: 8 },
      category: 'ADVANCED'
    },
    {
      name: 'stableFallback',
      type: 'BOOLEAN',
      description: 'Enable stable fallback (avoid ping-pong)',
      defaultValue: true,
      category: 'ADVANCED'
    },
    {
      name: 'extendedEvaluation',
      type: 'BOOLEAN',
      description: 'Enable extended evaluation before fallback',
      defaultValue: true,
      category: 'EXPERT'
    }
  ],
  validationRules: [
    {
      name: 'coverage_band_selection',
      description: 'Coverage scenarios should use low-frequency bands',
      type: 'CONSISTENCY',
      condition: 'isLowFrequencyBand(coverageNrBand) && isLowFrequencyBand(coverageLteBand)',
      action: 'WARNING'
    },
    {
      name: 'coverage_hysteresis',
      description: 'Higher hysteresis recommended for coverage stability',
      type: 'CONSISTENCY',
      condition: 'coverageHysteresis >= 3',
      action: 'WARNING'
    },
    {
      name: 'stable_fallback_consistency',
      description: 'Stable fallback should disable immediate fallback',
      type: 'CONSISTENCY',
      condition: '!stableFallback || !immediateFallbackAllowed',
      action: 'WARNING'
    }
  ],
  cmeditTemplates: [
    {
      commandName: 'configure_coverage_fallback',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${nrCellId} coverageMode=true,fallbackHysteresis=${coverageHysteresis},stableFallback=${stableFallback}',
      parameterMapping: {
        nodeId: 'nodeId',
        nrCellId: 'nrCellId',
        coverageHysteresis: 'coverageHysteresis',
        stableFallback: 'stableFallback'
      },
      description: 'Configure coverage-optimized fallback parameters'
    },
    {
      commandName: 'setup_stable_evaluation',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${nrCellId} extendedEvaluation=${extendedEvaluation},minEvaluationTime=10000',
      parameterMapping: {
        nodeId: 'nodeId',
        nrCellId: 'nrCellId',
        extendedEvaluation: 'extendedEvaluation'
      },
      description: 'Setup stable evaluation for coverage fallback'
    }
  ]
};

/**
 * Template 4: Mobility-Aware 5G4G Fallback (Priority 80)
 * Fallback configuration optimized for high mobility scenarios
 */
export const MOBILITY_5G4G_TEMPLATE: FrequencyRelationTemplate = {
  templateId: 'FREQ_5G4G_MOBILITY_004',
  templateName: 'Mobility-Aware 5G4G Fallback Configuration',
  templateDescription: '5G to 4G fallback configuration optimized for high mobility scenarios with fast handover',
  version: '1.0.0',
  templateType: '5G4G',
  priority: 80,
  baseConfig: Object.assign(
    createBase5G4GRelation('5G4G_MOBILITY',
      FALLBACK_NR_BANDS[78],
      FALLBACK_LTE_BANDS[1]
    ),
    {
      handoverConfig: AGGRESSIVE_5G4G_HANDOVER,
      fallbackConfig: {
        fallbackTriggers: {
          nrCoverageThreshold: -118,
          serviceInterruptionTime: 3000,
          ueCapabilityFallback: true,
          networkCongestionFallback: false
        },
        fallbackHandover: {
          prepareFallbackTimeout: 800,
          executeFallbackTimeout: 1200,
          fallbackPreparationRetryCount: 3,
          immediateFallbackAllowed: true
        },
        serviceContinuity: {
          sessionContinuity: true,
          ipAddressPreservation: true,
          qosPreservation: true
        },
        returnTo5G: {
          enabled: true,
          returnTriggers: {
            nrCoverageImprovement: -110,
            nrServiceQuality: 85,
            networkLoadImprovement: 65
          },
          returnEvaluationInterval: 15000,
          min5GStayTime: 30000
        }
      }
    }
  ),
  parameters: [
    {
      name: 'mobilityNrBand',
      type: 'INTEGER',
      description: '5G NR band in mobility scenario',
      defaultValue: 78,
      allowedValues: [41, 77, 78],
      category: 'BASIC'
    },
    {
      name: 'mobilityLteBand',
      type: 'INTEGER',
      description: 'LTE band for mobility fallback',
      defaultValue: 1,
      allowedValues: [1, 3, 7],
      category: 'BASIC'
    },
    {
      name: 'fastFallback',
      type: 'BOOLEAN',
      description: 'Enable fast fallback for mobility',
      defaultValue: true,
      category: 'BASIC'
    },
    {
      name: 'mobilityPrediction',
      type: 'BOOLEAN',
      description: 'Enable mobility-aware prediction',
      defaultValue: true,
      category: 'ADVANCED'
    },
    {
      name: 'handoverOptimization',
      type: 'ENUM',
      description: 'Handover optimization level',
      defaultValue: 'HIGH',
      allowedValues: ['LOW', 'MEDIUM', 'HIGH'],
      category: 'EXPERT'
    }
  ],
  validationRules: [
    {
      name: 'mobility_band_selection',
      description: 'Mobility scenarios should use suitable bands',
      type: 'CONSISTENCY',
      condition: 'supportsHighMobility(mobilityNrBand) && supportsHighMobility(mobilityLteBand)',
      action: 'WARNING'
    },
    {
      name: 'fast_fallback_consistency',
      description: 'Fast fallback requires optimized handover',
      type: 'CONSISTENCY',
      condition: '!fastFallback || handoverOptimization == "HIGH"',
      action: 'WARNING'
    },
    {
      name: 'mobility_prediction_requirements',
      description: 'Mobility prediction requires advanced features',
      type: 'CONSISTENCY',
      condition: '!mobilityPrediction || handoverOptimization == "HIGH"',
      action: 'WARNING'
    }
  ],
  cmeditTemplates: [
    {
      commandName: 'configure_mobility_fallback',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${nrCellId} mobilityMode=true,fastFallback=${fastFallback}',
      parameterMapping: {
        nodeId: 'nodeId',
        nrCellId: 'nrCellId',
        fastFallback: 'fastFallback'
      },
      description: 'Configure mobility-optimized fallback parameters'
    },
    {
      commandName: 'setup_mobility_prediction',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${nrCellId} mobilityPrediction=${mobilityPrediction},handoverOptimization=${handoverOptimization}',
      parameterMapping: {
        nodeId: 'nodeId',
        nrCellId: 'nrCellId',
        mobilityPrediction: 'mobilityPrediction',
        handoverOptimization: 'handoverOptimization'
      },
      description: 'Setup mobility prediction and optimization'
    }
  ]
};

/**
 * Template 5: QoS-Aware 5G4G Fallback (Priority 80)
 * Fallback configuration with QoS preservation and service differentiation
 */
export const QOS_AWARE_5G4G_TEMPLATE: FrequencyRelationTemplate = {
  templateId: 'FREQ_5G4G_QOS_005',
  templateName: 'QoS-Aware 5G4G Fallback Configuration',
  templateDescription: '5G to 4G fallback configuration with QoS preservation and service differentiation',
  version: '1.0.0',
  templateType: '5G4G',
  priority: 80,
  baseConfig: Object.assign(
    createBase5G4GRelation('5G4G_QOS',
      FALLBACK_NR_BANDS[78],
      FALLBACK_LTE_BANDS[3]
    ),
    {
      handoverConfig: STANDARD_5G4G_HANDOVER,
      fallbackConfig: {
        fallbackTriggers: {
          nrCoverageThreshold: -118,
          serviceInterruptionTime: 4000,
          ueCapabilityFallback: true,
          networkCongestionFallback: true
        },
        fallbackHandover: {
          prepareFallbackTimeout: 2000,
          executeFallbackTimeout: 3000,
          fallbackPreparationRetryCount: 3,
          immediateFallbackAllowed: false
        },
        serviceContinuity: {
          sessionContinuity: true,
          ipAddressPreservation: true,
          qosPreservation: true
        },
        returnTo5G: {
          enabled: true,
          returnTriggers: {
            nrCoverageImprovement: -108,
            nrServiceQuality: 80,
            networkLoadImprovement: 60
          },
          returnEvaluationInterval: 30000,
          min5GStayTime: 60000
        }
      }
    }
  ),
  parameters: [
    {
      name: 'qosNrBand',
      type: 'INTEGER',
      description: '5G NR band for QoS-aware fallback',
      defaultValue: 78,
      allowedValues: [41, 77, 78],
      category: 'BASIC'
    },
    {
      name: 'qosLteBand',
      type: 'INTEGER',
      description: 'LTE band for QoS preservation',
      defaultValue: 3,
      allowedValues: [1, 3, 7],
      category: 'BASIC'
    },
    {
      name: 'qosPreservation',
      type: 'BOOLEAN',
      description: 'Enable QoS preservation during fallback',
      defaultValue: true,
      category: 'BASIC'
    },
    {
      name: 'serviceDifferentiation',
      type: 'BOOLEAN',
      description: 'Enable service differentiation during fallback',
      defaultValue: true,
      category: 'ADVANCED'
    },
    {
      name: 'priorityFallback',
      type: 'BOOLEAN',
      description: 'Enable priority-based fallback for high QoS users',
      defaultValue: true,
      category: 'ADVANCED'
    },
    {
      name: 'qosMapping',
      type: 'ENUM',
      description: 'QoS mapping strategy during fallback',
      defaultValue: 'PRESERVE',
      allowedValues: ['PRESERVE', 'ADAPT', 'DEGRADE'],
      category: 'EXPERT'
    }
  ],
  validationRules: [
    {
      name: 'qos_band_compatibility',
      description: 'Bands should support QoS features',
      type: 'CONSISTENCY',
      condition: 'supportsQoS(qosNrBand) && supportsQoS(qosLteBand)',
      action: 'WARNING'
    },
    {
      name: 'qos_preservation_consistency',
      description: 'QoS preservation requires service differentiation',
      type: 'CONSISTENCY',
      condition: '!qosPreservation || serviceDifferentiation',
      action: 'WARNING'
    },
    {
      name: 'priority_fallback_requirements',
      description: 'Priority fallback requires QoS preservation',
      type: 'CONSISTENCY',
      condition: '!priorityFallback || qosPreservation',
      action: 'WARNING'
    }
  ],
  cmeditTemplates: [
    {
      commandName: 'configure_qos_fallback',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${nrCellId} qosPreservation=${qosPreservation},serviceDifferentiation=${serviceDifferentiation}',
      parameterMapping: {
        nodeId: 'nodeId',
        nrCellId: 'nrCellId',
        qosPreservation: 'qosPreservation',
        serviceDifferentiation: 'serviceDifferentiation'
      },
      description: 'Configure QoS-aware fallback parameters'
    },
    {
      commandName: 'setup_priority_fallback',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${nrCellId} priorityFallback=${priorityFallback},qosMapping=${qosMapping}',
      parameterMapping: {
        nodeId: 'nodeId',
        nrCellId: 'nrCellId',
        priorityFallback: 'priorityFallback',
        qosMapping: 'qosMapping'
      },
      description: 'Setup priority-based QoS fallback'
    }
  ]
};

/**
 * Collection of all 5G4G frequency relation templates
 */
export const FREQ_5G4G_TEMPLATES: FrequencyRelationTemplate[] = [
  STANDARD_5G4G_TEMPLATE,
  EMERGENCY_5G4G_TEMPLATE,
  COVERAGE_5G4G_TEMPLATE,
  MOBILITY_5G4G_TEMPLATE,
  QOS_AWARE_5G4G_TEMPLATE
];

/**
 * Helper functions for 5G4G template validation and configuration
 */

/**
 * Check if NR-LTE combination is valid for fallback
 */
export function isValid5G4GCombination(nrBand: number, lteBand: number): boolean {
  // Valid fallback combinations based on typical deployments
  const validFallbackCombinations = [
    { nr: 78, lte: [1, 3, 7, 20, 28] },
    { nr: 41, lte: [1, 3, 7, 28] },
    { nr: 77, lte: [1, 3, 7, 28] },
    { nr: 28, lte: [20, 28] },
    { nr: 71, lte: [20, 28] }
  ];

  return validFallbackCombinations.some(combination =>
    combination.nr === nrBand && combination.lte.includes(lteBand)
  );
}

/**
 * Check if band is coverage band
 */
export function isCoverageBand(band: number): boolean {
  const coverageBands = [20, 28, 71];
  return coverageBands.includes(band);
}

/**
 * Check if band is low frequency band
 */
export function isLowFrequencyBand(band: number): boolean {
  const lowFrequencyBands = [20, 28, 71];
  return lowFrequencyBands.includes(band);
}

/**
 * Check if band supports high mobility
 */
export function supportsHighMobility(band: number): boolean {
  const highMobilityBands = [1, 3, 7, 41, 77, 78];
  return highMobilityBands.includes(band);
}

/**
 * Check if band supports QoS features
 */
export function supportsQoS(band: number): boolean {
  const qosBands = [1, 3, 7, 41, 77, 78];
  return qosBands.includes(band);
}

/**
 * Calculate 5G4G frequency relation performance metrics
 */
export function calculate5G4GMetrics(config: Freq5G4GRelation): FrequencyRelationMetrics {
  // Base metrics for 5G4G fallback configuration
  const baseMetrics = {
    handoverSuccessRate: 0.92,
    averageHandoverLatency: 120,
    interferenceLevel: 0.15,
    capacityUtilization: 0.5,
    userThroughput: { average: 40, peak: 150, cellEdge: 8 },
    callDropRate: 0.02,
    setupSuccessRate: 0.94
  };

  // Adjust metrics based on fallback configuration
  if (config.fallbackConfig.serviceContinuity.sessionContinuity) {
    baseMetrics.handoverSuccessRate *= 1.05;
    baseMetrics.callDropRate *= 0.7;
  }

  if (config.fallbackConfig.fallbackTriggers.ueCapabilityFallback) {
    baseMetrics.setupSuccessRate *= 1.03;
  }

  if (config.fallbackConfig.returnTo5G.enabled) {
    baseMetrics.userThroughput.average *= 1.3;
    baseMetrics.userThroughput.peak *= 1.5;
  }

  // Adjust for handover configuration
  if (config.handoverConfig.hysteresis < 2) {
    baseMetrics.handoverSuccessRate *= 0.95;
    baseMetrics.averageHandoverLatency *= 0.8;
  }

  // Adjust for coverage vs capacity bands
  if (isCoverageBand(config.relatedFreq.bandNumber)) {
    baseMetrics.userThroughput.cellEdge *= 1.2;
    baseMetrics.interferenceLevel *= 0.8;
  } else {
    baseMetrics.userThroughput.average *= 1.3;
    baseMetrics.userThroughput.peak *= 1.4;
  }

  return baseMetrics;
}