/**
 * 4G5G Frequency Relations Templates (Priority 60)
 *
 * Comprehensive EN-DC (E-UTRAN-NR Dual Connectivity) configuration templates
 * for Ericsson RAN including split bearer, multi-connectivity, and cross-technology mobility
 */

import type {
  Freq4G5GRelation,
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
 * Common 5G NR frequency bands
 */
export const NR_BANDS: Record<number, FrequencyBand> = {
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
 * Standard EN-DC handover configuration
 */
export const STANDARD_ENDC_HANDOVER: HandoverConfiguration = {
  triggerType: 'B1',
  hysteresis: 2,
  timeToTrigger: 320,
  cellIndividualOffset: 0,
  freqSpecificOffset: 0,
  eventBasedConfig: {
    threshold1: -110
  },
  measurementConfig: {
    reportInterval: 240,
    maxReportCells: 8,
    reportAmount: '8'
  }
};

/**
 * Aggressive EN-DC handover for high mobility
 */
export const AGGRESSIVE_ENDC_HANDOVER: HandoverConfiguration = {
  triggerType: 'B1',
  hysteresis: 1,
  timeToTrigger: 160,
  cellIndividualOffset: 2,
  freqSpecificOffset: 1,
  eventBasedConfig: {
    threshold1: -115
  },
  measurementConfig: {
    reportInterval: 120,
    maxReportCells: 16,
    reportAmount: '16'
  }
};

/**
 * Conservative EN-DC handover for stability
 */
export const CONSERVATIVE_ENDC_HANDOVER: HandoverConfiguration = {
  triggerType: 'B1',
  hysteresis: 4,
  timeToTrigger: 640,
  cellIndividualOffset: 0,
  freqSpecificOffset: 0,
  eventBasedConfig: {
    threshold1: -105
  },
  measurementConfig: {
    reportInterval: 480,
    maxReportCells: 4,
    reportAmount: '4'
  }
};

/**
 * EN-DC capacity sharing configuration
 */
export const ENDC_CAPACITY_SHARING: CapacitySharingParams = {
  enabled: true,
  strategy: 'PRIORITY_BASED',
  loadBalancingThreshold: 80,
  maxCapacityRatio: 0.8,
  minGuaranteedCapacity: 0.2,
  dynamicRebalancing: true,
  rebalancingInterval: 180
};

/**
 * Cross-technology interference coordination
 */
export const CROSS_TECH_INTERFERENCE_SETTINGS: InterferenceSettings = {
  enabled: true,
  coordinationType: 'eICIC',
  interBandManagement: {
    almostBlankSubframes: false,
    crsPowerBoost: 0,
    powerControlCoordination: true
  },
  dynamicCoordination: true,
  coordinationInterval: 500
};

/**
 * Create base 4G5G frequency relation
 */
function createBase4G5GRelation(
  relationId: string,
  referenceBand: FrequencyBand,
  relatedBand: FrequencyBand,
  priority: number = 60
): Freq4G5GRelation {
  return {
    relationId,
    referenceFreq: referenceBand,
    relatedFreq: relatedBand,
    relationType: '4G5G',
    priority,
    adminState: 'UNLOCKED',
    operState: 'ENABLED',
    createdAt: new Date(),
    modifiedAt: new Date(),
    endcConfig: {
      meNbConfig: {
        splitBearerSupport: true,
        dualConnectivitySupport: true,
        releaseVersion: 'REL16'
      },
      sgNbConfig: {
        sgNbAdditionAllowed: true,
        sgNbModificationAllowed: true,
        sgNbReleaseAllowed: true,
        maxSgNbPerUe: 4
      },
      pdcpDuplication: {
        enabled: false,
        duplicationActivation: 'RLC',
        duplicationDeactivation: 'RLC'
      },
      endcMeasurements: {
        nrEventB1: {
          threshold: -110,
          hysteresis: 2,
          timeToTrigger: 320
        }
      }
    }
  };
}

/**
 * Template 1: Standard EN-DC Configuration (Priority 60)
 * Basic 4G5G dual connectivity configuration with standard parameters
 */
export const STANDARD_ENDC_TEMPLATE: FrequencyRelationTemplate = {
  templateId: 'FREQ_4G5G_STANDARD_001',
  templateName: 'Standard EN-DC Configuration',
  templateDescription: 'Basic E-UTRAN-NR Dual Connectivity configuration with standard split bearer and mobility parameters',
  version: '1.0.0',
  templateType: '4G5G',
  priority: 60,
  baseConfig: createBase4G5GRelation(
    '4G5G_STANDARD',
    { bandNumber: 3, frequencyRange: { uplink: { start: 1710, end: 1785 }, downlink: { start: 1805, end: 1880 } }, bandCategory: 'LTE', primaryUse: 'CAPACITY' },
    NR_BANDS[78]
  ),
  parameters: [
    {
      name: 'lteBand',
      type: 'INTEGER',
      description: 'LTE band number for Master eNodeB',
      defaultValue: 3,
      allowedValues: [1, 3, 7, 20, 28],
      category: 'BASIC'
    },
    {
      name: 'nrBand',
      type: 'INTEGER',
      description: 'NR band number for Secondary gNodeB',
      defaultValue: 78,
      allowedValues: [41, 77, 78, 28, 71],
      category: 'BASIC'
    },
    {
      name: 'endcRelease',
      type: 'ENUM',
      description: 'EN-DC 3GPP release version',
      defaultValue: 'REL16',
      allowedValues: ['REL15', 'REL16', 'REL17'],
      category: 'BASIC'
    },
    {
      name: 'splitBearerSupport',
      type: 'BOOLEAN',
      description: 'Enable split bearer support',
      defaultValue: true,
      category: 'BASIC'
    },
    {
      name: 'nrEventB1Threshold',
      type: 'INTEGER',
      description: 'NR B1 event threshold in dBm',
      defaultValue: -110,
      constraints: { min: -125, max: -85 },
      category: 'ADVANCED'
    },
    {
      name: 'maxSgNbPerUe',
      type: 'INTEGER',
      description: 'Maximum number of SgNBs per UE',
      defaultValue: 4,
      constraints: { min: 1, max: 8 },
      category: 'ADVANCED'
    }
  ],
  validationRules: [
    {
      name: 'valid_endc_combination',
      description: 'Valid LTE-NR band combination required',
      type: 'CONSISTENCY',
      condition: 'isValidENDCCombination(lteBand, nrBand)',
      action: 'ERROR'
    },
    {
      name: 'b1_threshold_range',
      description: 'B1 threshold must be within valid range',
      type: 'RANGE',
      condition: 'nrEventB1Threshold >= -125 && nrEventB1Threshold <= -85',
      action: 'ERROR'
    },
    {
      name: 'sgnb_limit',
      description: 'Maximum SgNB per UE must be reasonable',
      type: 'RANGE',
      condition: 'maxSgNbPerUe >= 1 && maxSgNbPerUe <= 8',
      action: 'ERROR'
    }
  ],
  cmeditTemplates: [
    {
      commandName: 'enable_endc',
      commandTemplate: 'cmedit set ${nodeId} ENBFunction endcEnabled=true,splitBearerSupport=${splitBearerSupport}',
      parameterMapping: {
        nodeId: 'nodeId',
        splitBearerSupport: 'splitBearerSupport'
      },
      description: 'Enable EN-DC functionality on eNodeB'
    },
    {
      commandName: 'configure_nr_measurements',
      commandTemplate: 'cmedit set ${nodeId} ENBFunction nrEventB1Threshold=${nrEventB1Threshold},nrEventB1Hysteresis=${nrEventB1Hysteresis}',
      parameterMapping: {
        nodeId: 'nodeId',
        nrEventB1Threshold: 'nrEventB1Threshold',
        nrEventB1Hysteresis: 'nrEventB1Hysteresis'
      },
      description: 'Configure NR measurement parameters for EN-DC'
    },
    {
      commandName: 'setup_sgNB_config',
      commandTemplate: 'cmedit set ${nodeId} ENBFunction sgNbAdditionAllowed=true,sgNbModificationAllowed=true,sgNbReleaseAllowed=true,maxSgNbPerUe=${maxSgNbPerUe}',
      parameterMapping: {
        nodeId: 'nodeId',
        maxSgNbPerUe: 'maxSgNbPerUe'
      },
      description: 'Configure SgNB management parameters'
    }
  ]
};

/**
 * Template 2: High Performance EN-DC (Priority 60)
 * EN-DC configuration optimized for maximum throughput and performance
 */
export const HIGH_PERFORMANCE_ENDC_TEMPLATE: FrequencyRelationTemplate = {
  templateId: 'FREQ_4G5G_PERFORMANCE_002',
  templateName: 'High Performance EN-DC Configuration',
  templateDescription: 'EN-DC configuration optimized for maximum throughput with PDCP duplication and advanced features',
  version: '1.0.0',
  templateType: '4G5G',
  priority: 60,
  baseConfig: Object.assign(
    createBase4G5GRelation('4G5G_PERF',
      { bandNumber: 7, frequencyRange: { uplink: { start: 2500, end: 2570 }, downlink: { start: 2620, end: 2690 } }, bandCategory: 'LTE', primaryUse: 'CAPACITY' },
      NR_BANDS[78]
    ),
    {
      handoverConfig: AGGRESSIVE_ENDC_HANDOVER,
      capacitySharing: ENDC_CAPACITY_SHARING,
      endcConfig: {
        meNbConfig: {
          splitBearerSupport: true,
          dualConnectivitySupport: true,
          releaseVersion: 'REL17'
        },
        sgNbConfig: {
          sgNbAdditionAllowed: true,
          sgNbModificationAllowed: true,
          sgNbReleaseAllowed: true,
          maxSgNbPerUe: 8
        },
        pdcpDuplication: {
          enabled: true,
          duplicationActivation: 'RLC',
          duplicationDeactivation: 'RLC'
        },
        endcMeasurements: {
          nrEventB1: {
            threshold: -115,
            hysteresis: 1,
            timeToTrigger: 160
          }
        }
      }
    }
  ),
  parameters: [
    {
      name: 'lteBand',
      type: 'INTEGER',
      description: 'LTE band number (high capacity preferred)',
      defaultValue: 7,
      allowedValues: [1, 3, 7],
      category: 'BASIC'
    },
    {
      name: 'nrBand',
      type: 'INTEGER',
      description: 'NR band number (high capacity preferred)',
      defaultValue: 78,
      allowedValues: [41, 77, 78],
      category: 'BASIC'
    },
    {
      name: 'pdcpDuplication',
      type: 'BOOLEAN',
      description: 'Enable PDCP duplication for reliability',
      defaultValue: true,
      category: 'ADVANCED'
    },
    {
      name: 'duplicationActivation',
      type: 'ENUM',
      description: 'PDCP duplication activation method',
      defaultValue: 'RLC',
      allowedValues: ['RLC', 'MAC'],
      category: 'ADVANCED'
    },
    {
      name: 'maxSgNbPerUe',
      type: 'INTEGER',
      description: 'Maximum number of SgNBs per UE (high performance)',
      defaultValue: 8,
      constraints: { min: 4, max: 16 },
      category: 'ADVANCED'
    },
    {
      name: 'throughputOptimization',
      type: 'BOOLEAN',
      description: 'Enable throughput optimization features',
      defaultValue: true,
      category: 'EXPERT'
    }
  ],
  validationRules: [
    {
      name: 'high_capacity_bands',
      description: 'High performance configuration should use high capacity bands',
      type: 'CONSISTENCY',
      condition: 'isHighCapacityBand(lteBand) && isHighCapacityBand(nrBand)',
      action: 'WARNING'
    },
    {
      name: 'pdcp_duplication_consistency',
      description: 'PDCP duplication requires REL16 or higher',
      type: 'CONSISTENCY',
      condition: '!pdcpDuplication || endcRelease != "REL15"',
      action: 'ERROR'
    },
    {
      name: 'sgnb_capacity',
      description: 'High performance should support more SgNBs',
      type: 'CONSISTENCY',
      condition: 'maxSgNbPerUe >= 6',
      action: 'WARNING'
    }
  ],
  cmeditTemplates: [
    {
      commandName: 'configure_pdcp_duplication',
      commandTemplate: 'cmedit set ${nodeId} ENBFunction pdcpDuplicationEnabled=true,pduplicationActivation=${duplicationActivation}',
      parameterMapping: {
        nodeId: 'nodeId',
        duplicationActivation: 'duplicationActivation'
      },
      description: 'Configure PDCP duplication for enhanced reliability'
    },
    {
      commandName: 'setup_high_performance',
      commandTemplate: 'cmedit set ${nodeId} ENBFunction maxSgNbPerUe=${maxSgNbPerUe},endcOptimization=HIGH_PERFORMANCE',
      parameterMapping: {
        nodeId: 'nodeId',
        maxSgNbPerUe: 'maxSgNbPerUe'
      },
      description: 'Configure high performance EN-DC parameters'
    },
    {
      commandName: 'optimize_throughput',
      commandTemplate: 'cmedit set ${nodeId} ENBFunction throughputOptimization=true,advancedScheduling=true',
      parameterMapping: {
        nodeId: 'nodeId'
      },
      description: 'Enable throughput optimization features'
    }
  ]
};

/**
 * Template 3: Coverage EN-DC (Priority 60)
 * EN-DC configuration optimized for coverage expansion scenarios
 */
export const COVERAGE_ENDC_TEMPLATE: FrequencyRelationTemplate = {
  templateId: 'FREQ_4G5G_COVERAGE_003',
  templateName: 'Coverage EN-DC Configuration',
  templateDescription: 'EN-DC configuration optimized for coverage expansion with low frequency LTE and high frequency NR',
  version: '1.0.0',
  templateType: '4G5G',
  priority: 60,
  baseConfig: Object.assign(
    createBase4G5GRelation('4G5G_COV',
      { bandNumber: 20, frequencyRange: { uplink: { start: 832, end: 862 }, downlink: { start: 791, end: 821 } }, bandCategory: 'LTE', primaryUse: 'COVERAGE' },
      NR_BANDS[78]
    ),
    {
      handoverConfig: CONSERVATIVE_ENDC_HANDOVER
    }
  ),
  parameters: [
    {
      name: 'coverageLteBand',
      type: 'INTEGER',
      description: 'Low frequency LTE band for coverage',
      defaultValue: 20,
      allowedValues: [20, 28],
      category: 'BASIC'
    },
    {
      name: 'capacityNrBand',
      type: 'INTEGER',
      description: 'High frequency NR band for capacity boost',
      defaultValue: 78,
      allowedValues: [41, 77, 78],
      category: 'BASIC'
    },
    {
      name: 'coverageThreshold',
      type: 'INTEGER',
      description: 'Coverage threshold for EN-DC activation in dBm',
      defaultValue: -105,
      constraints: { min: -115, max: -95 },
      category: 'ADVANCED'
    },
    {
      name: 'earlyRelease',
      type: 'BOOLEAN',
      description: 'Enable early SgNB release for coverage preservation',
      defaultValue: true,
      category: 'ADVANCED'
    },
    {
      name: 'fallbackHysteresis',
      type: 'INTEGER',
      description: 'Fallback hysteresis in dB',
      defaultValue: 4,
      constraints: { min: 2, max: 8 },
      category: 'EXPERT'
    }
  ],
  validationRules: [
    {
      name: 'coverage_band_selection',
      description: 'Coverage LTE band should be low frequency',
      type: 'CONSISTENCY',
      condition: 'isLowFrequencyBand(coverageLteBand)',
      action: 'WARNING'
    },
    {
      name: 'capacity_band_selection',
      description: 'Capacity NR band should be mid/high frequency',
      type: 'CONSISTENCY',
      condition: '!isLowFrequencyBand(capacityNrBand)',
      action: 'WARNING'
    },
    {
      name: 'coverage_threshold',
      description: 'Coverage threshold should be conservative',
      type: 'CONSISTENCY',
      condition: 'coverageThreshold <= -100',
      action: 'WARNING'
    }
  ],
  cmeditTemplates: [
    {
      commandName: 'setup_coverage_endc',
      commandTemplate: 'cmedit set ${nodeId} ENBFunction endcCoverageMode=true,coverageThreshold=${coverageThreshold}',
      parameterMapping: {
        nodeId: 'nodeId',
        coverageThreshold: 'coverageThreshold'
      },
      description: 'Configure coverage-optimized EN-DC parameters'
    },
    {
      commandName: 'configure_early_release',
      commandTemplate: 'cmedit set ${nodeId} ENBFunction earlySgNbRelease=${earlyRelease},fallbackHysteresis=${fallbackHysteresis}',
      parameterMapping: {
        nodeId: 'nodeId',
        earlyRelease: 'earlyRelease',
        fallbackHysteresis: 'fallbackHysteresis'
      },
      description: 'Configure early SgNB release for coverage scenarios'
    }
  ]
};

/**
 * Template 4: Mobility EN-DC (Priority 60)
 * EN-DC configuration optimized for high mobility scenarios
 */
export const MOBILITY_ENDC_TEMPLATE: FrequencyRelationTemplate = {
  templateId: 'FREQ_4G5G_MOBILITY_004',
  templateName: 'Mobility EN-DC Configuration',
  templateDescription: 'EN-DC configuration optimized for high mobility scenarios with fast handover and robust connectivity',
  version: '1.0.0',
  templateType: '4G5G',
  priority: 60,
  baseConfig: Object.assign(
    createBase4G5GRelation('4G5G_MOB',
      { bandNumber: 3, frequencyRange: { uplink: { start: 1710, end: 1785 }, downlink: { start: 1805, end: 1880 } }, bandCategory: 'LTE', primaryUse: 'CAPACITY' },
      NR_BANDS[41]
    ),
    {
      handoverConfig: AGGRESSIVE_ENDC_HANDOVER,
      interferenceConfig: CROSS_TECH_INTERFERENCE_SETTINGS
    }
  ),
  parameters: [
    {
      name: 'mobilityLteBand',
      type: 'INTEGER',
      description: 'LTE band optimized for mobility',
      defaultValue: 3,
      allowedValues: [1, 3, 7],
      category: 'BASIC'
    },
    {
      name: 'mobilityNrBand',
      type: 'INTEGER',
      description: 'NR band optimized for mobility',
      defaultValue: 41,
      allowedValues: [41, 77, 78],
      category: 'BASIC'
    },
    {
      name: 'handoverType',
      type: 'ENUM',
      description: 'Handover type for mobility optimization',
      defaultValue: 'MAKE_BEFORE_BREAK',
      allowedValues: ['MAKE_BEFORE_BREAK', 'BREAK_BEFORE_MAKE', 'CONDITIONAL_HANDOVER'],
      category: 'ADVANCED'
    },
    {
      name: 'fastHandover',
      type: 'BOOLEAN',
      description: 'Enable fast handover features',
      defaultValue: true,
      category: 'ADVANCED'
    },
    {
      name: 'mobilityRobustness',
      type: 'ENUM',
      description: 'Mobility robustness level',
      defaultValue: 'HIGH',
      allowedValues: ['LOW', 'MEDIUM', 'HIGH'],
      category: 'EXPERT'
    }
  ],
  validationRules: [
    {
      name: 'mobility_bands',
      description: 'Mobility bands should support high speed scenarios',
      type: 'CONSISTENCY',
      condition: 'supportsHighMobility(mobilityLteBand) && supportsHighMobility(mobilityNrBand)',
      action: 'WARNING'
    },
    {
      name: 'fast_handover_consistency',
      description: 'Fast handover requires compatible release',
      type: 'CONSISTENCY',
      condition: '!fastHandover || endcRelease != "REL15"',
      action: 'ERROR'
    }
  ],
  cmeditTemplates: [
    {
      commandName: 'configure_mobility',
      commandTemplate: 'cmedit set ${nodeId} ENBFunction handoverType=${handoverType},fastHandover=${fastHandover}',
      parameterMapping: {
        nodeId: 'nodeId',
        handoverType: 'handoverType',
        fastHandover: 'fastHandover'
      },
      description: 'Configure mobility-optimized EN-DC parameters'
    },
    {
      commandName: 'setup_robustness',
      commandTemplate: 'cmedit set ${nodeId} ENBFunction mobilityRobustness=${mobilityRobustness},conditionalHandover=true',
      parameterMapping: {
        nodeId: 'nodeId',
        mobilityRobustness: 'mobilityRobustness'
      },
      description: 'Configure mobility robustness features'
    }
  ]
};

/**
 * Template 5: Enterprise EN-DC (Priority 60)
 * EN-DC configuration optimized for enterprise/campus scenarios
 */
export const ENTERPRISE_ENDC_TEMPLATE: FrequencyRelationTemplate = {
  templateId: 'FREQ_4G5G_ENTERPRISE_005',
  templateName: 'Enterprise EN-DC Configuration',
  templateDescription: 'EN-DC configuration optimized for enterprise/campus deployments with URLLC and QoS optimization',
  version: '1.0.0',
  templateType: '4G5G',
  priority: 60,
  baseConfig: Object.assign(
    createBase4G5GRelation('4G5G_ENT',
      { bandNumber: 78, frequencyRange: { uplink: { start: 3300, end: 3800 }, downlink: { start: 3300, end: 3800 } }, bandCategory: 'NR', primaryUse: 'HOTSPOT' },
      { bandNumber: 1, frequencyRange: { uplink: { start: 1920, end: 1980 }, downlink: { start: 2110, end: 2170 } }, bandCategory: 'LTE', primaryUse: 'HOTSPOT' }
    ),
    {
      handoverConfig: CONSERVATIVE_ENDC_HANDOVER,
      capacitySharing: {
        enabled: true,
        strategy: 'PRIORITY_BASED',
        loadBalancingThreshold: 70,
        maxCapacityRatio: 0.9,
        minGuaranteedCapacity: 0.1,
        dynamicRebalancing: true,
        rebalancingInterval: 120
      }
    }
  ),
  parameters: [
    {
      name: 'enterpriseLteBand',
      type: 'INTEGER',
      description: 'LTE band for enterprise deployment',
      defaultValue: 1,
      allowedValues: [1, 3, 7, 41],
      category: 'BASIC'
    },
    {
      name: 'enterpriseNrBand',
      type: 'INTEGER',
      description: 'NR band for enterprise deployment',
      defaultValue: 78,
      allowedValues: [41, 77, 78],
      category: 'BASIC'
    },
    {
      name: 'urlcSupport',
      type: 'BOOLEAN',
      description: 'Enable URLLC support for enterprise applications',
      defaultValue: true,
      category: 'BASIC'
    },
    {
      name: 'qosOptimization',
      type: 'BOOLEAN',
      description: 'Enable QoS optimization for enterprise traffic',
      defaultValue: true,
      category: 'ADVANCED'
    },
    {
      name: 'privateNetwork',
      type: 'BOOLEAN',
      description: 'Configure for private network deployment',
      defaultValue: false,
      category: 'ADVANCED'
    },
    {
      name: 'capacityPriority',
      type: 'ENUM',
      description: 'Capacity priority strategy',
      defaultValue: 'NR_PRIORITY',
      allowedValues: ['LTE_PRIORITY', 'NR_PRIORITY', 'BALANCED'],
      category: 'EXPERT'
    }
  ],
  validationRules: [
    {
      name: 'enterprise_bands',
      description: 'Enterprise deployment should use high capacity bands',
      type: 'CONSISTENCY',
      condition: 'isHighCapacityBand(enterpriseLteBand) && isHighCapacityBand(enterpriseNrBand)',
      action: 'WARNING'
    },
    {
      name: 'urlc_release',
      description: 'URLLC requires REL16 or higher',
      type: 'CONSISTENCY',
      condition: '!urlcSupport || endcRelease != "REL15"',
      action: 'ERROR'
    }
  ],
  cmeditTemplates: [
    {
      commandName: 'configure_enterprise',
      commandTemplate: 'cmedit set ${nodeId} ENBFunction enterpriseMode=true,urlcSupport=${urlcSupport},qosOptimization=${qosOptimization}',
      parameterMapping: {
        nodeId: 'nodeId',
        urlcSupport: 'urlcSupport',
        qosOptimization: 'qosOptimization'
      },
      description: 'Configure enterprise-optimized EN-DC parameters'
    },
    {
      commandName: 'setup_capacity_strategy',
      commandTemplate: 'cmedit set ${nodeId} ENBFunction capacityPriority=${capacityPriority},maxCapacityRatio=${maxCapacityRatio}',
      parameterMapping: {
        nodeId: 'nodeId',
        capacityPriority: 'capacityPriority',
        maxCapacityRatio: 'maxCapacityRatio'
      },
      description: 'Setup capacity priority strategy for enterprise'
    }
  ]
};

/**
 * Collection of all 4G5G frequency relation templates
 */
export const FREQ_4G5G_TEMPLATES: FrequencyRelationTemplate[] = [
  STANDARD_ENDC_TEMPLATE,
  HIGH_PERFORMANCE_ENDC_TEMPLATE,
  COVERAGE_ENDC_TEMPLATE,
  MOBILITY_ENDC_TEMPLATE,
  ENTERPRISE_ENDC_TEMPLATE
];

/**
 * Helper functions for 4G5G template validation and configuration
 */

/**
 * Check if LTE-NR band combination is valid for EN-DC
 */
export function isValidENDCCombination(lteBand: number, nrBand: number): boolean {
  const validENDCCombinations = [
    // LTE Band 1 combinations
    { lte: 1, nr: [78, 77, 41] },
    // LTE Band 3 combinations
    { lte: 3, nr: [78, 77, 41, 28] },
    // LTE Band 7 combinations
    { lte: 7, nr: [78, 77, 41] },
    // LTE Band 20 combinations
    { lte: 20, nr: [78, 77, 41, 28] },
    // LTE Band 28 combinations
    { lte: 28, nr: [78, 77, 41] }
  ];

  return validENDCCombinations.some(combination =>
    combination.lte === lteBand && combination.nr.includes(nrBand)
  );
}

/**
 * Check if band is high capacity
 */
export function isHighCapacityBand(band: number): boolean {
  const highCapacityBands = [1, 3, 7, 41, 77, 78];
  return highCapacityBands.includes(band);
}

/**
 * Check if band is low frequency
 */
export function isLowFrequencyBand(band: number): boolean {
  const lowFrequencyBands = [20, 28, 71];
  return lowFrequencyBands.includes(band);
}

/**
 * Check if band supports high mobility scenarios
 */
export function supportsHighMobility(band: number): boolean {
  const highMobilityBands = [1, 3, 7, 41, 77, 78];
  return highMobilityBands.includes(band);
}

/**
 * Calculate 4G5G frequency relation performance metrics
 */
export function calculate4G5GMetrics(config: Freq4G5GRelation): FrequencyRelationMetrics {
  // Base metrics for EN-DC configuration
  const baseMetrics = {
    handoverSuccessRate: 0.93,
    averageHandoverLatency: 80,
    interferenceLevel: 0.25,
    capacityUtilization: 0.7,
    userThroughput: { average: 150, peak: 800, cellEdge: 20 },
    callDropRate: 0.008,
    setupSuccessRate: 0.96
  };

  // Adjust metrics based on configuration
  if (config.endcConfig.meNbConfig.splitBearerSupport) {
    baseMetrics.userThroughput.average *= 1.5;
    baseMetrics.userThroughput.peak *= 1.8;
  }

  if (config.endcConfig.pdcpDuplication.enabled) {
    baseMetrics.callDropRate *= 0.5;
    baseMetrics.handoverSuccessRate *= 1.03;
  }

  if (config.endcConfig.releaseVersion === 'REL17') {
    baseMetrics.userThroughput.average *= 1.2;
    baseMetrics.setupSuccessRate *= 1.02;
  }

  // Adjust for handover configuration
  if (config.handoverConfig.hysteresis < 2) {
    baseMetrics.handoverSuccessRate *= 0.96;
    baseMetrics.averageHandoverLatency *= 0.7;
  }

  return baseMetrics;
}