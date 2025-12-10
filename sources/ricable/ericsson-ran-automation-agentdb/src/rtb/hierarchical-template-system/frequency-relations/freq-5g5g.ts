/**
 * 5G5G Frequency Relations Templates (Priority 70)
 *
 * Comprehensive NR-NR Dual Connectivity configuration templates for Ericsson RAN
 * including multi-band carrier aggregation, advanced beam management, and dynamic spectrum sharing
 */

import type {
  Freq5G5GRelation,
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
 * Advanced 5G NR frequency bands including mmWave
 */
export const ADVANCED_NR_BANDS: Record<number, FrequencyBand> = {
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
  257: {
    bandNumber: 257,
    frequencyRange: {
      uplink: { start: 26500, end: 29500 },
      downlink: { start: 26500, end: 29500 }
    },
    bandCategory: 'MMWAVE',
    primaryUse: 'HOTSPOT'
  },
  260: {
    bandNumber: 260,
    frequencyRange: {
      uplink: { start: 37000, end: 40000 },
      downlink: { start: 37000, end: 40000 }
    },
    bandCategory: 'MMWAVE',
    primaryUse: 'HOTSPOT'
  },
  261: {
    bandNumber: 261,
    frequencyRange: {
      uplink: { start: 27500, end: 28350 },
      downlink: { start: 27500, end: 28350 }
    },
    bandCategory: 'MMWAVE',
    primaryUse: 'HOTSPOT'
  }
};

/**
 * Standard NR-NR handover configuration
 */
export const STANDARD_NRNR_HANDOVER: HandoverConfiguration = {
  triggerType: 'A3',
  hysteresis: 2,
  timeToTrigger: 160,
  cellIndividualOffset: 0,
  freqSpecificOffset: 0,
  eventBasedConfig: {
    a3Offset: 2
  },
  measurementConfig: {
    reportInterval: 120,
    maxReportCells: 16,
    reportAmount: '16'
  }
};

/**
 * Ultra-low latency NR-NR handover for URLLC
 */
export const URLLC_NRNR_HANDOVER: HandoverConfiguration = {
  triggerType: 'A3',
  hysteresis: 1,
  timeToTrigger: 80,
  cellIndividualOffset: 1,
  freqSpecificOffset: 1,
  eventBasedConfig: {
    a3Offset: 1
  },
  measurementConfig: {
    reportInterval: 40,
    maxReportCells: 32,
    reportAmount: 'INFINITY'
  }
};

/**
 * mmWave-aware NR-NR handover configuration
 */
export const MMWAVE_NRNR_HANDOVER: HandoverConfiguration = {
  triggerType: 'A5',
  hysteresis: 3,
  timeToTrigger: 240,
  cellIndividualOffset: 2,
  freqSpecificOffset: 2,
  eventBasedConfig: {
    threshold1: -100,
    threshold2: -90,
    a3Offset: 3
  },
  measurementConfig: {
    reportInterval: 80,
    maxReportCells: 8,
    reportAmount: '8'
  }
};

/**
 * NR-NR capacity sharing with multi-band optimization
 */
export const NRNR_CAPACITY_SHARING: CapacitySharingParams = {
  enabled: true,
  strategy: 'PRIORITY_BASED',
  loadBalancingThreshold: 85,
  maxCapacityRatio: 0.9,
  minGuaranteedCapacity: 0.1,
  dynamicRebalancing: true,
  rebalancingInterval: 60
};

/**
 * Advanced beam management interference coordination
 */
export const BEAM_MANAGEMENT_INTERFERENCE_SETTINGS: InterferenceSettings = {
  enabled: true,
  coordinationType: 'FeICIC',
  interBandManagement: {
    almostBlankSubframes: false,
    crsPowerBoost: 0,
    powerControlCoordination: true
  },
  dynamicCoordination: true,
  coordinationInterval: 100
};

/**
 * Create base 5G5G frequency relation
 */
function createBase5G5GRelation(
  relationId: string,
  referenceBand: FrequencyBand,
  relatedBand: FrequencyBand,
  priority: number = 70
): Freq5G5GRelation {
  return {
    relationId,
    referenceFreq: referenceBand,
    relatedFreq: relatedBand,
    relationType: '5G5G',
    priority,
    adminState: 'UNLOCKED',
    operState: 'ENABLED',
    createdAt: new Date(),
    modifiedAt: new Date(),
    nrdcConfig: {
      pCellConfig: {
        cellType: 'PCELL',
        servingCellPriority: 7,
        cellReselectionPriority: 7
      },
      scgConfig: {
        scgAdditionSupported: true,
        scgChangeSupported: true,
        scgReleaseSupported: true,
        maxScgPerUe: 4
      },
      mbcaConfig: {
        enabled: false,
        aggregatedBands: [],
        maxAggregatedBandwidth: 0,
        crossScheduling: false,
        dynamicSlotAllocation: false
      },
      beamManagement: {
        beamFailureRecovery: true,
        beamManagementConfig: {
          maxBeamCandidates: 8,
          beamReportInterval: 40,
          beamSwitchingTime: 20
        }
      },
      dssConfig: {
        enabled: false,
        sharingMode: 'STATIC',
        spectrumAllocation: {
          nrShare: 1.0,
          lteShare: 0.0
        }
      }
    }
  };
}

/**
 * Template 1: Standard NR-DC Configuration (Priority 70)
 * Basic 5G5G dual connectivity configuration with standard parameters
 */
export const STANDARD_NRDC_TEMPLATE: FrequencyRelationTemplate = {
  templateId: 'FREQ_5G5G_STANDARD_001',
  templateName: 'Standard NR-DC Configuration',
  templateDescription: 'Basic NR-NR Dual Connectivity configuration with standard multi-band carrier aggregation and beam management',
  version: '1.0.0',
  templateType: '5G5G',
  priority: 70,
  baseConfig: createBase5G5GRelation(
    '5G5G_STANDARD',
    ADVANCED_NR_BANDS[78],
    ADVANCED_NR_BANDS[41]
  ),
  parameters: [
    {
      name: 'primaryNrBand',
      type: 'INTEGER',
      description: 'Primary NR band number for PCell',
      defaultValue: 78,
      allowedValues: [41, 77, 78, 28, 71],
      category: 'BASIC'
    },
    {
      name: 'secondaryNrBand',
      type: 'INTEGER',
      description: 'Secondary NR band number for SCell',
      defaultValue: 41,
      allowedValues: [41, 77, 78, 28, 71],
      category: 'BASIC'
    },
    {
      name: 'mbcaEnabled',
      type: 'BOOLEAN',
      description: 'Enable Multi-band Carrier Aggregation',
      defaultValue: true,
      category: 'BASIC'
    },
    {
      name: 'maxAggregatedBandwidth',
      type: 'INTEGER',
      description: 'Maximum aggregated bandwidth in MHz',
      defaultValue: 400,
      constraints: { min: 100, max: 1000 },
      category: 'ADVANCED'
    },
    {
      name: 'crossScheduling',
      type: 'BOOLEAN',
      description: 'Enable cross-carrier scheduling',
      defaultValue: true,
      category: 'ADVANCED'
    },
    {
      name: 'maxScgPerUe',
      type: 'INTEGER',
      description: 'Maximum number of SCGs per UE',
      defaultValue: 4,
      constraints: { min: 1, max: 8 },
      category: 'ADVANCED'
    }
  ],
  validationRules: [
    {
      name: 'valid_nrnr_combination',
      description: 'Valid NR-NR band combination required',
      type: 'CONSISTENCY',
      condition: 'isValidNRNRCombination(primaryNrBand, secondaryNrBand)',
      action: 'ERROR'
    },
    {
      name: 'different_bands',
      description: 'Primary and secondary bands must be different',
      type: 'CONSISTENCY',
      condition: 'primaryNrBand != secondaryNrBand',
      action: 'ERROR'
    },
    {
      name: 'bandwidth_limits',
      description: 'Aggregated bandwidth within supported limits',
      type: 'RANGE',
      condition: 'maxAggregatedBandwidth <= getMaxSupportedBandwidth(primaryNrBand, secondaryNrBand)',
      action: 'ERROR'
    }
  ],
  cmeditTemplates: [
    {
      commandName: 'enable_nrdc',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${primaryCellId} nrdcEnabled=true,mbcaEnabled=${mbcaEnabled}',
      parameterMapping: {
        nodeId: 'nodeId',
        primaryCellId: 'primaryCellId',
        mbcaEnabled: 'mbcaEnabled'
      },
      description: 'Enable NR-DC functionality on gNodeB'
    },
    {
      commandName: 'configure_mbca',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${primaryCellId} maxAggregatedBandwidth=${maxAggregatedBandwidth},crossScheduling=${crossScheduling}',
      parameterMapping: {
        nodeId: 'nodeId',
        primaryCellId: 'primaryCellId',
        maxAggregatedBandwidth: 'maxAggregatedBandwidth',
        crossScheduling: 'crossScheduling'
      },
      description: 'Configure Multi-band Carrier Aggregation parameters'
    },
    {
      commandName: 'setup_scg_management',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${primaryCellId} maxScgPerUe=${maxScgPerUe},scgAdditionSupported=true,scgChangeSupported=true',
      parameterMapping: {
        nodeId: 'nodeId',
        primaryCellId: 'primaryCellId',
        maxScgPerUe: 'maxScgPerUe'
      },
      description: 'Configure SCG management parameters'
    }
  ]
};

/**
 * Template 2: mmWave Integration (Priority 70)
 * NR-DC configuration integrating mmWave bands with sub-6 GHz anchors
 */
export const MMWAVE_INTEGRATION_TEMPLATE: FrequencyRelationTemplate = {
  templateId: 'FREQ_5G5G_MMWAVE_002',
  templateName: 'mmWave Integration NR-DC Configuration',
  templateDescription: 'NR-DC configuration integrating mmWave bands with sub-6 GHz anchor bands for ultra-high capacity',
  version: '1.0.0',
  templateType: '5G5G',
  priority: 70,
  baseConfig: Object.assign(
    createBase5G5GRelation('5G5G_MMWAVE',
      ADVANCED_NR_BANDS[78],
      ADVANCED_NR_BANDS[257]
    ),
    {
      handoverConfig: MMWAVE_NRNR_HANDOVER,
      nrdcConfig: {
        pCellConfig: {
          cellType: 'PCELL',
          servingCellPriority: 7,
          cellReselectionPriority: 7
        },
        scgConfig: {
          scgAdditionSupported: true,
          scgChangeSupported: true,
          scgReleaseSupported: true,
          maxScgPerUe: 8
        },
        mbcaConfig: {
          enabled: true,
          aggregatedBands: [78, 257],
          maxAggregatedBandwidth: 800,
          crossScheduling: true,
          dynamicSlotAllocation: true
        },
        beamManagement: {
          beamFailureRecovery: true,
          beamManagementConfig: {
            maxBeamCandidates: 16,
            beamReportInterval: 20,
            beamSwitchingTime: 10
          }
        },
        dssConfig: {
          enabled: false,
          sharingMode: 'STATIC',
          spectrumAllocation: {
            nrShare: 1.0,
            lteShare: 0.0
          }
        }
      }
    }
  ),
  parameters: [
    {
      name: 'anchorBand',
      type: 'INTEGER',
      description: 'Sub-6 GHz anchor band',
      defaultValue: 78,
      allowedValues: [41, 77, 78],
      category: 'BASIC'
    },
    {
      name: 'mmwaveBand',
      type: 'INTEGER',
      description: 'mmWave capacity band',
      defaultValue: 257,
      allowedValues: [257, 260, 261],
      category: 'BASIC'
    },
    {
      name: 'beamManagement',
      type: 'BOOLEAN',
      description: 'Enable advanced beam management',
      defaultValue: true,
      category: 'BASIC'
    },
    {
      name: 'beamFailureRecovery',
      type: 'BOOLEAN',
      description: 'Enable beam failure recovery',
      defaultValue: true,
      category: 'BASIC'
    },
    {
      name: 'maxBeamCandidates',
      type: 'INTEGER',
      description: 'Maximum beam candidates for tracking',
      defaultValue: 16,
      constraints: { min: 4, max: 32 },
      category: 'ADVANCED'
    },
    {
      name: 'mmwaveHysteresis',
      type: 'INTEGER',
      description: 'mmWave handover hysteresis in dB',
      defaultValue: 3,
      constraints: { min: 1, max: 8 },
      category: 'EXPERT'
    }
  ],
  validationRules: [
    {
      name: 'mmwave_anchor_validation',
      description: 'mmWave requires sub-6 GHz anchor',
      type: 'CONSISTENCY',
      condition: 'isSub6Band(anchorBand) && isMmwaveBand(mmwaveBand)',
      action: 'ERROR'
    },
    {
      name: 'beam_management_required',
      description: 'mmWave integration requires beam management',
      type: 'CONSISTENCY',
      condition: 'beamManagement && beamFailureRecovery',
      action: 'WARNING'
    },
    {
      name: 'beam_candidate_limits',
      description: 'Beam candidates within reasonable limits',
      type: 'RANGE',
      condition: 'maxBeamCandidates >= 4 && maxBeamCandidates <= 32',
      action: 'ERROR'
    }
  ],
  cmeditTemplates: [
    {
      commandName: 'configure_mmwave',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${anchorCellId} mmwaveIntegration=true,mmwaveBand=${mmwaveBand}',
      parameterMapping: {
        nodeId: 'nodeId',
        anchorCellId: 'anchorCellId',
        mmwaveBand: 'mmwaveBand'
      },
      description: 'Configure mmWave integration parameters'
    },
    {
      commandName: 'setup_beam_management',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${anchorCellId} beamManagement=true,beamFailureRecovery=true,maxBeamCandidates=${maxBeamCandidates}',
      parameterMapping: {
        nodeId: 'nodeId',
        anchorCellId: 'anchorCellId',
        maxBeamCandidates: 'maxBeamCandidates'
      },
      description: 'Configure advanced beam management for mmWave'
    },
    {
      commandName: 'configure_mmwave_handover',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${anchorCellId} handoverType=A5,hysteresis=${mmwaveHysteresis},threshold1=-100,threshold2=-90',
      parameterMapping: {
        nodeId: 'nodeId',
        anchorCellId: 'anchorCellId',
        mmwaveHysteresis: 'mmwaveHysteresis'
      },
      description: 'Configure mmWave-optimized handover parameters'
    }
  ]
};

/**
 * Template 3: URLLC NR-DC (Priority 70)
 * NR-DC configuration optimized for Ultra-Reliable Low Latency Communication
 */
export const URLLC_NRDC_TEMPLATE: FrequencyRelationTemplate = {
  templateId: 'FREQ_5G5G_URLLC_003',
  templateName: 'URLLC NR-DC Configuration',
  templateDescription: 'NR-DC configuration optimized for URLLC applications with ultra-low latency and high reliability',
  version: '1.0.0',
  templateType: '5G5G',
  priority: 70,
  baseConfig: Object.assign(
    createBase5G5GRelation('5G5G_URLLC',
      ADVANCED_NR_BANDS[78],
      ADVANCED_NR_BANDS[41]
    ),
    {
      handoverConfig: URLLC_NRNR_HANDOVER,
      capacitySharing: {
        enabled: true,
        strategy: 'PRIORITY_BASED',
        loadBalancingThreshold: 60,
        maxCapacityRatio: 0.95,
        minGuaranteedCapacity: 0.05,
        dynamicRebalancing: true,
        rebalancingInterval: 30
      },
      nrdcConfig: {
        pCellConfig: {
          cellType: 'PCELL',
          servingCellPriority: 7,
          cellReselectionPriority: 7
        },
        scgConfig: {
          scgAdditionSupported: true,
          scgChangeSupported: true,
          scgReleaseSupported: true,
          maxScgPerUe: 2
        },
        mbcaConfig: {
          enabled: true,
          aggregatedBands: [78, 41],
          maxAggregatedBandwidth: 200,
          crossScheduling: true,
          dynamicSlotAllocation: true
        },
        beamManagement: {
          beamFailureRecovery: true,
          beamManagementConfig: {
            maxBeamCandidates: 4,
            beamReportInterval: 10,
            beamSwitchingTime: 5
          }
        },
        dssConfig: {
          enabled: false,
          sharingMode: 'STATIC',
          spectrumAllocation: {
            nrShare: 1.0,
            lteShare: 0.0
          }
        }
      }
    }
  ),
  parameters: [
    {
      name: 'primaryUrlccBand',
      type: 'INTEGER',
      description: 'Primary band for URLLC (low latency)',
      defaultValue: 78,
      allowedValues: [41, 77, 78],
      category: 'BASIC'
    },
    {
      name: 'secondaryUrlccBand',
      type: 'INTEGER',
      description: 'Secondary band for URLLC redundancy',
      defaultValue: 41,
      allowedValues: [41, 77, 78],
      category: 'BASIC'
    },
    {
      name: 'urlccReliability',
      type: 'ENUM',
      description: 'URLLC reliability level',
      defaultValue: 'HIGH',
      allowedValues: ['MEDIUM', 'HIGH', 'ULTRA_HIGH'],
      category: 'BASIC'
    },
    {
      name: 'latencyTarget',
      type: 'INTEGER',
      description: 'Target latency in milliseconds',
      defaultValue: 1,
      constraints: { min: 0.5, max: 5 },
      category: 'ADVANCED'
    },
    {
      name: 'duplicationMode',
      type: 'ENUM',
      description: 'PDCP duplication mode for reliability',
      defaultValue: 'BOTH_BANDS',
      allowedValues: ['PRIMARY_ONLY', 'SECONDARY_ONLY', 'BOTH_BANDS'],
      category: 'ADVANCED'
    },
    {
      name: 'grantFreeUplink',
      type: 'BOOLEAN',
      description: 'Enable grant-free uplink for ultra-low latency',
      defaultValue: true,
      category: 'EXPERT'
    }
  ],
  validationRules: [
    {
      name: 'urlcc_band_selection',
      description: 'URLLC bands should support low latency features',
      type: 'CONSISTENCY',
      condition: 'supportsLowLatency(primaryUrlccBand) && supportsLowLatency(secondaryUrlccBand)',
      action: 'WARNING'
    },
    {
      name: 'latency_consistency',
      description: 'Latency target should be realistic for configuration',
      type: 'CONSISTENCY',
      condition: 'latencyTarget >= 0.5 && latencyTarget <= 5',
      action: 'WARNING'
    },
    {
      name: 'reliability_duplication',
      description: 'High reliability requires duplication',
      type: 'CONSISTENCY',
      condition: 'urlccReliability != "ULTRA_HIGH" || duplicationMode == "BOTH_BANDS"',
      action: 'WARNING'
    }
  ],
  cmeditTemplates: [
    {
      commandName: 'configure_urlcc',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${primaryCellId} urlccEnabled=true,reliabilityLevel=${urlccReliability},latencyTarget=${latencyTarget}',
      parameterMapping: {
        nodeId: 'nodeId',
        primaryCellId: 'primaryCellId',
        urlccReliability: 'urlccReliability',
        latencyTarget: 'latencyTarget'
      },
      description: 'Configure URLLC-specific parameters'
    },
    {
      commandName: 'setup_duplication',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${primaryCellId} pdcpDuplication=true,duplicationMode=${duplicationMode}',
      parameterMapping: {
        nodeId: 'nodeId',
        primaryCellId: 'primaryCellId',
        duplicationMode: 'duplicationMode'
      },
      description: 'Configure PDCP duplication for URLLC reliability'
    },
    {
      commandName: 'enable_low_latency',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${primaryCellId} grantFreeUplink=${grantFreeUplink},miniSlot=true',
      parameterMapping: {
        nodeId: 'nodeId',
        primaryCellId: 'primaryCellId',
        grantFreeUplink: 'grantFreeUplink'
      },
      description: 'Enable ultra-low latency features'
    }
  ]
};

/**
 * Template 4: Dynamic Spectrum Sharing NR-DC (Priority 70)
 * NR-DC configuration with Dynamic Spectrum Sharing between NR and LTE
 */
export const DSS_NRDC_TEMPLATE: FrequencyRelationTemplate = {
  templateId: 'FREQ_5G5G_DSS_004',
  templateName: 'Dynamic Spectrum Sharing NR-DC Configuration',
  templateDescription: 'NR-DC configuration with Dynamic Spectrum Sharing for optimal spectrum utilization',
  version: '1.0.0',
  templateType: '5G5G',
  priority: 70,
  baseConfig: Object.assign(
    createBase5G5GRelation('5G5G_DSS',
      ADVANCED_NR_BANDS[78],
      ADVANCED_NR_BANDS[41]
    ),
    {
      handoverConfig: STANDARD_NRNR_HANDOVER,
      nrdcConfig: {
        pCellConfig: {
          cellType: 'PCELL',
          servingCellPriority: 7,
          cellReselectionPriority: 7
        },
        scgConfig: {
          scgAdditionSupported: true,
          scgChangeSupported: true,
          scgReleaseSupported: true,
          maxScgPerUe: 4
        },
        mbcaConfig: {
          enabled: true,
          aggregatedBands: [78, 41],
          maxAggregatedBandwidth: 300,
          crossScheduling: true,
          dynamicSlotAllocation: true
        },
        beamManagement: {
          beamFailureRecovery: true,
          beamManagementConfig: {
            maxBeamCandidates: 8,
            beamReportInterval: 40,
            beamSwitchingTime: 20
          }
        },
        dssConfig: {
          enabled: true,
          sharingMode: 'DYNAMIC',
          spectrumAllocation: {
            nrShare: 0.7,
            lteShare: 0.3
          }
        }
      }
    }
  ),
  parameters: [
    {
      name: 'dssPrimaryBand',
      type: 'INTEGER',
      description: 'Primary band with DSS support',
      defaultValue: 78,
      allowedValues: [41, 77, 78],
      category: 'BASIC'
    },
    {
      name: 'dssSecondaryBand',
      type: 'INTEGER',
      description: 'Secondary band for DSS',
      defaultValue: 41,
      allowedValues: [41, 77, 78],
      category: 'BASIC'
    },
    {
      name: 'sharingMode',
      type: 'ENUM',
      description: 'DSS sharing mode',
      defaultValue: 'DYNAMIC',
      allowedValues: ['STATIC', 'DYNAMIC', 'ADAPTIVE'],
      category: 'BASIC'
    },
    {
      name: 'nrShareRatio',
      type: 'FLOAT',
      description: 'NR spectrum share ratio',
      defaultValue: 0.7,
      constraints: { min: 0.3, max: 0.9 },
      category: 'ADVANCED'
    },
    {
      name: 'lteShareRatio',
      type: 'FLOAT',
      description: 'LTE spectrum share ratio',
      defaultValue: 0.3,
      constraints: { min: 0.1, max: 0.7 },
      category: 'ADVANCED'
    },
    {
      name: 'dynamicAllocation',
      type: 'BOOLEAN',
      description: 'Enable dynamic spectrum allocation',
      defaultValue: true,
      category: 'EXPERT'
    }
  ],
  validationRules: [
    {
      name: 'dss_band_support',
      description: 'Selected bands must support DSS',
      type: 'CONSISTENCY',
      condition: 'supportsDSS(dssPrimaryBand) && supportsDSS(dssSecondaryBand)',
      action: 'ERROR'
    },
    {
      name: 'share_ratio_sum',
      description: 'NR and LTE share ratios must sum to 1.0',
      type: 'CONSISTENCY',
      condition: 'Math.abs((nrShareRatio + lteShareRatio) - 1.0) < 0.01',
      action: 'ERROR'
    },
    {
      name: 'dss_dynamic_mode',
      description: 'Dynamic allocation requires DYNAMIC or ADAPTIVE mode',
      type: 'CONSISTENCY',
      condition: '!dynamicAllocation || sharingMode != "STATIC"',
      action: 'ERROR'
    }
  ],
  cmeditTemplates: [
    {
      commandName: 'enable_dss',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${primaryCellId} dssEnabled=true,sharingMode=${sharingMode}',
      parameterMapping: {
        nodeId: 'nodeId',
        primaryCellId: 'primaryCellId',
        sharingMode: 'sharingMode'
      },
      description: 'Enable Dynamic Spectrum Sharing'
    },
    {
      commandName: 'configure_spectrum_sharing',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${primaryCellId} nrShareRatio=${nrShareRatio},lteShareRatio=${lteShareRatio},dynamicAllocation=${dynamicAllocation}',
      parameterMapping: {
        nodeId: 'nodeId',
        primaryCellId: 'primaryCellId',
        nrShareRatio: 'nrShareRatio',
        lteShareRatio: 'lteShareRatio',
        dynamicAllocation: 'dynamicAllocation'
      },
      description: 'Configure spectrum sharing parameters'
    }
  ]
};

/**
 * Template 5: High-Capacity NR-DC (Priority 70)
 * NR-DC configuration optimized for maximum capacity and throughput
 */
export const HIGH_CAPACITY_NRDC_TEMPLATE: FrequencyRelationTemplate = {
  templateId: 'FREQ_5G5G_CAPACITY_005',
  templateName: 'High Capacity NR-DC Configuration',
  templateDescription: 'NR-DC configuration optimized for maximum capacity with advanced carrier aggregation and massive MIMO',
  version: '1.0.0',
  templateType: '5G5G',
  priority: 70,
  baseConfig: Object.assign(
    createBase5G5GRelation('5G5G_CAPACITY',
      ADVANCED_NR_BANDS[78],
      ADVANCED_NR_BANDS[77]
    ),
    {
      handoverConfig: STANDARD_NRNR_HANDOVER,
      capacitySharing: NRNR_CAPACITY_SHARING,
      nrdcConfig: {
        pCellConfig: {
          cellType: 'PCELL',
          servingCellPriority: 7,
          cellReselectionPriority: 7
        },
        scgConfig: {
          scgAdditionSupported: true,
          scgChangeSupported: true,
          scgReleaseSupported: true,
          maxScgPerUe: 8
        },
        mbcaConfig: {
          enabled: true,
          aggregatedBands: [78, 77, 41],
          maxAggregatedBandwidth: 800,
          crossScheduling: true,
          dynamicSlotAllocation: true
        },
        beamManagement: {
          beamFailureRecovery: true,
          beamManagementConfig: {
            maxBeamCandidates: 32,
            beamReportInterval: 20,
            beamSwitchingTime: 10
          }
        },
        dssConfig: {
          enabled: false,
          sharingMode: 'STATIC',
          spectrumAllocation: {
            nrShare: 1.0,
            lteShare: 0.0
          }
        }
      }
    }
  ),
  parameters: [
    {
      name: 'capacityPrimaryBand',
      type: 'INTEGER',
      description: 'Primary high-capacity band',
      defaultValue: 78,
      allowedValues: [41, 77, 78],
      category: 'BASIC'
    },
    {
      name: 'capacitySecondaryBands',
      type: 'STRING',
      description: 'Comma-separated secondary capacity bands',
      defaultValue: '77,41',
      category: 'BASIC'
    },
    {
      name: 'massiveMimo',
      type: 'BOOLEAN',
      description: 'Enable massive MIMO configuration',
      defaultValue: true,
      category: 'BASIC'
    },
    {
      name: 'maxAggregatedBandwidth',
      type: 'INTEGER',
      description: 'Maximum aggregated bandwidth in MHz',
      defaultValue: 800,
      constraints: { min: 200, max: 1000 },
      category: 'ADVANCED'
    },
    {
      name: 'multiUserMimo',
      type: 'BOOLEAN',
      description: 'Enable multi-user MIMO',
      defaultValue: true,
      category: 'ADVANCED'
    },
    {
      name: 'advancedScheduling',
      type: 'BOOLEAN',
      description: 'Enable advanced scheduling features',
      defaultValue: true,
      category: 'EXPERT'
    }
  ],
  validationRules: [
    {
      name: 'capacity_band_selection',
      description: 'All bands should be high-capacity',
      type: 'CONSISTENCY',
      condition: 'isHighCapacityBand(capacityPrimaryBand) && areHighCapacityBands(capacitySecondaryBands)',
      action: 'WARNING'
    },
    {
      name: 'massive_mimo_consistency',
      description: 'Massive MIMO requires compatible bands',
      type: 'CONSISTENCY',
      condition: '!massiveMimo || supportsMassiveMimo(capacityPrimaryBand)',
      action: 'WARNING'
    },
    {
      name: 'bandwidth_capacity',
      description: 'High capacity requires sufficient bandwidth',
      type: 'CONSISTENCY',
      condition: 'maxAggregatedBandwidth >= 400',
      action: 'WARNING'
    }
  ],
  cmeditTemplates: [
    {
      commandName: 'configure_high_capacity',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${primaryCellId} highCapacityMode=true,massiveMimo=${massiveMimo}',
      parameterMapping: {
        nodeId: 'nodeId',
        primaryCellId: 'primaryCellId',
        massiveMimo: 'massiveMimo'
      },
      description: 'Configure high capacity NR-DC parameters'
    },
    {
      commandName: 'setup_multi_band_ca',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${primaryCellId} mbcaEnabled=true,aggregatedBands=${capacitySecondaryBands},maxAggregatedBandwidth=${maxAggregatedBandwidth}',
      parameterMapping: {
        nodeId: 'nodeId',
        primaryCellId: 'primaryCellId',
        capacitySecondaryBands: 'capacitySecondaryBands',
        maxAggregatedBandwidth: 'maxAggregatedBandwidth'
      },
      description: 'Setup multi-band carrier aggregation'
    },
    {
      commandName: 'enable_advanced_features',
      commandTemplate: 'cmedit set ${nodeId} NRCellCU=${primaryCellId} multiUserMimo=${multiUserMimo},advancedScheduling=${advancedScheduling}',
      parameterMapping: {
        nodeId: 'nodeId',
        primaryCellId: 'primaryCellId',
        multiUserMimo: 'multiUserMimo',
        advancedScheduling: 'advancedScheduling'
      },
      description: 'Enable advanced capacity features'
    }
  ]
};

/**
 * Collection of all 5G5G frequency relation templates
 */
export const FREQ_5G5G_TEMPLATES: FrequencyRelationTemplate[] = [
  STANDARD_NRDC_TEMPLATE,
  MMWAVE_INTEGRATION_TEMPLATE,
  URLLC_NRDC_TEMPLATE,
  DSS_NRDC_TEMPLATE,
  HIGH_CAPACITY_NRDC_TEMPLATE
];

/**
 * Helper functions for 5G5G template validation and configuration
 */

/**
 * Check if NR-NR band combination is valid
 */
export function isValidNRNRCombination(primaryBand: number, secondaryBand: number): boolean {
  // Avoid same band combinations
  if (primaryBand === secondaryBand) return false;

  // Valid NR-NR combinations based on 3GPP specifications
  const validNRNRCombinations = [
    [41, 78], [41, 77], [78, 77],    // Sub-6 GHz combinations
    [78, 257], [41, 257], [77, 257], // Sub-6 + mmWave
    [257, 260], [257, 261],          // mmWave combinations
    [41, 78, 77], [78, 41, 257]      // Multi-band combinations
  ];

  const combination = [primaryBand, secondaryBand].sort((a, b) => a - b);

  return validNRNRCombinations.some(validComb =>
    validComb.length === 2 &&
    validComb.every((band, index) => band === combination[index])
  );
}

/**
 * Check if band is sub-6 GHz
 */
export function isSub6Band(band: number): boolean {
  const sub6Bands = [41, 77, 78, 28, 71];
  return sub6Bands.includes(band);
}

/**
 * Check if band is mmWave
 */
export function isMmwaveBand(band: number): boolean {
  const mmwaveBands = [257, 260, 261];
  return mmwaveBands.includes(band);
}

/**
 * Check if band supports low latency features
 */
export function supportsLowLatency(band: number): boolean {
  const lowLatencyBands = [41, 77, 78];
  return lowLatencyBands.includes(band);
}

/**
 * Check if band supports DSS
 */
export function supportsDSS(band: number): boolean {
  const dssBands = [41, 77, 78];
  return dssBands.includes(band);
}

/**
 * Check if band supports massive MIMO
 */
export function supportsMassiveMimo(band: number): boolean {
  const massiveMimoBands = [41, 77, 78];
  return massiveMimoBands.includes(band);
}

/**
 * Check if bands are high capacity
 */
export function areHighCapacityBands(bandsString: string): boolean {
  const bands = bandsString.split(',').map(b => parseInt(b.trim()));
  const highCapacityBands = [41, 77, 78, 257, 260, 261];
  return bands.every(band => highCapacityBands.includes(band));
}

/**
 * Get maximum supported bandwidth for band combination
 */
export function getMaxSupportedBandwidth(primaryBand: number, secondaryBand: number): number {
  const bandWidths: Record<number, number> = {
    41: 100, 77: 400, 78: 400, 28: 100, 71: 100,
    257: 400, 260: 800, 261: 400
  };

  return Math.min(bandWidths[primaryBand] || 100, bandWidths[secondaryBand] || 100) * 2;
}

/**
 * Calculate 5G5G frequency relation performance metrics
 */
export function calculate5G5GMetrics(config: Freq5G5GRelation): FrequencyRelationMetrics {
  // Base metrics for NR-DC configuration
  const baseMetrics = {
    handoverSuccessRate: 0.94,
    averageHandoverLatency: 60,
    interferenceLevel: 0.2,
    capacityUtilization: 0.8,
    userThroughput: { average: 300, peak: 2000, cellEdge: 50 },
    callDropRate: 0.005,
    setupSuccessRate: 0.97
  };

  // Adjust metrics based on configuration
  if (config.nrdcConfig.mbcaConfig.enabled) {
    baseMetrics.userThroughput.average *= 2.0;
    baseMetrics.userThroughput.peak *= 3.0;
    baseMetrics.capacityUtilization *= 1.4;
  }

  if (isMmwaveBand(config.relatedFreq.bandNumber) || isMmwaveBand(config.referenceFreq.bandNumber)) {
    baseMetrics.userThroughput.peak *= 2.5;
    baseMetrics.userThroughput.cellEdge *= 0.7; // mmWave has worse cell edge
  }

  if (config.nrdcConfig.dssConfig.enabled) {
    baseMetrics.capacityUtilization *= 1.2;
    baseMetrics.interferenceLevel *= 1.3; // DSS increases interference
  }

  if (config.nrdcConfig.beamManagement.beamFailureRecovery) {
    baseMetrics.callDropRate *= 0.4;
    baseMetrics.handoverSuccessRate *= 1.04;
  }

  return baseMetrics;
}