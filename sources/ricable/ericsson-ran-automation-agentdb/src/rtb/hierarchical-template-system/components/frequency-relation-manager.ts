/**
 * Frequency Relation Manager - Inter-frequency Configuration Implementation
 *
 * Manages frequency relation templates for different RAN scenarios:
 * - 4G4G frequency relations (LTE-LTE)
 * - 4G5G frequency relations (EN-DC - E-UTRA-NR Dual Connectivity)
 * - 5G5G frequency relations (NR-NR Dual Connectivity)
 * - 5G4G frequency relations (fallback scenarios)
 *
 * Each frequency relation template includes:
 * - Neighbor relation configurations
 * - Handover parameter optimization
 * - Capacity and load balancing parameters
 * - Frequency-specific constraints and validation
 */

import {
  PriorityTemplate,
  TemplatePriority,
  FrequencyRelationConfig,
  FrequencyBand,
  NeighborRelationConfig,
  HandoverParameterConfig,
  CapacityParameterConfig,
  FrequencyCompatibilityMatrix,
  FrequencyRecommendation,
  FrequencyConstraint,
  EnhancedTemplateMeta
} from '../interfaces';

import { HierarchicalTemplateEngineConfig } from '../interfaces';

/**
 * Frequency Relation Manager implementation
 */
export class FrequencyRelationManager {
  private config: HierarchicalTemplateEngineConfig;
  private compatibilityMatrix: FrequencyCompatibilityMatrix;

  constructor(config: HierarchicalTemplateEngineConfig) {
    this.config = config;
    this.compatibilityMatrix = this.initializeCompatibilityMatrix();
  }

  /**
   * Generate 4G4G frequency relation template
   * Optimized for LTE-LTE inter-frequency handovers and load balancing
   */
  async generate4G4GRelation(config: FrequencyRelationConfig): Promise<PriorityTemplate> {
    const template: PriorityTemplate = {
      meta: {
        version: `4G4G_${config.sourceBand}_${config.targetBand}_${Date.now()}`,
        author: ['Frequency Relation Manager'],
        description: `4G4G frequency relation template: ${config.sourceBand} -> ${config.targetBand}`,
        priority: TemplatePriority.FREQUENCY_4G4G,
        frequencyBand: config.sourceBand,
        tags: ['4G4G', 'inter-frequency', 'LTE', 'handover'],
        validationRules: this.generate4G4GValidationRules(config)
      },
      priority: TemplatePriority.FREQUENCY_4G4G,
      configuration: this.generate4G4GConfiguration(config),
      conditions: this.generate4G4GConditions(config),
      evaluations: this.generate4G4GEvaluations(config),
      custom: this.generate4G4GFunctions(config)
    };

    return template;
  }

  /**
   * Generate 4G5G EN-DC frequency relation template
   * Optimized for E-UTRA-NR Dual Connectivity scenarios
   */
  async generate4G5GRelation(config: FrequencyRelationConfig): Promise<PriorityTemplate> {
    const template: PriorityTemplate = {
      meta: {
        version: `4G5G_${config.sourceBand}_${config.targetBand}_${Date.now()}`,
        author: ['Frequency Relation Manager'],
        description: `4G5G EN-DC frequency relation template: ${config.sourceBand} -> ${config.targetBand}`,
        priority: TemplatePriority.FREQUENCY_4G5G,
        frequencyBand: config.sourceBand,
        tags: ['4G5G', 'EN-DC', 'dual-connectivity', 'LTE-NR'],
        validationRules: this.generate4G5GValidationRules(config)
      },
      priority: TemplatePriority.FREQUENCY_4G5G,
      configuration: this.generate4G5GConfiguration(config),
      conditions: this.generate4G5GConditions(config),
      evaluations: this.generate4G5GEvaluations(config),
      custom: this.generate4G5GFunctions(config)
    };

    return template;
  }

  /**
   * Generate 5G5G NR-NR DC frequency relation template
   * Optimized for NR-NR Dual Connectivity scenarios
   */
  async generate5G5GRelation(config: FrequencyRelationConfig): Promise<PriorityTemplate> {
    const template: PriorityTemplate = {
      meta: {
        version: `5G5G_${config.sourceBand}_${config.targetBand}_${Date.now()}`,
        author: ['Frequency Relation Manager'],
        description: `5G5G NR-NR DC frequency relation template: ${config.sourceBand} -> ${config.targetBand}`,
        priority: TemplatePriority.FREQUENCY_5G5G,
        frequencyBand: config.sourceBand,
        tags: ['5G5G', 'NR-NR-DC', 'dual-connectivity', 'NR'],
        validationRules: this.generate5G5GValidationRules(config)
      },
      priority: TemplatePriority.FREQUENCY_5G5G,
      configuration: this.generate5G5GConfiguration(config),
      conditions: this.generate5G5GConditions(config),
      evaluations: this.generate5G5GEvaluations(config),
      custom: this.generate5G5GFunctions(config)
    };

    return template;
  }

  /**
   * Generate 5G4G fallback frequency relation template
   * Optimized for 5G to 4G fallback scenarios
   */
  async generate5G4GRelation(config: FrequencyRelationConfig): Promise<PriorityTemplate> {
    const template: PriorityTemplate = {
      meta: {
        version: `5G4G_${config.sourceBand}_${config.targetBand}_${Date.now()}`,
        author: ['Frequency Relation Manager'],
        description: `5G4G fallback frequency relation template: ${config.sourceBand} -> ${config.targetBand}`,
        priority: TemplatePriority.FREQUENCY_5G4G,
        frequencyBand: config.sourceBand,
        tags: ['5G4G', 'fallback', 'inter-RAT', 'NR-LTE'],
        validationRules: this.generate5G4GValidationRules(config)
      },
      priority: TemplatePriority.FREQUENCY_5G4G,
      configuration: this.generate5G4GConfiguration(config),
      conditions: this.generate5G4GConditions(config),
      evaluations: this.generate5G4GEvaluations(config),
      custom: this.generate5G4GFunctions(config)
    };

    return template;
  }

  /**
   * Get frequency compatibility matrix
   */
  async getCompatibilityMatrix(): Promise<FrequencyCompatibilityMatrix> {
    return this.compatibilityMatrix;
  }

  // ============================================================================
  // 4G4G FREQUENCY RELATION GENERATION METHODS
  // ============================================================================

  /**
   * Generate 4G4G specific configuration
   */
  private generate4G4GConfiguration(config: FrequencyRelationConfig): Record<string, any> {
    const baseConfig = {
      ...config.parameters,
      'EUtranFreqRelation.isHoAllowed': true,
      'EUtranFreqRelation.isRemovable': true,
      'EUtranFreqRelation.qRxLevMinOffset': 0,
      'EUtranFreqRelation.qQualMinOffset': 0
    };

    // Add band-specific parameters
    const bandConfig = this.get4G4GBandParameters(config.sourceBand, config.targetBand);

    // Add neighbor relations
    for (const neighbor of config.neighborRelations) {
      baseConfig[`neighbor_${neighbor.relationId}`] = neighbor.parameters;
    }

    // Add handover parameters
    if (config.handoverParameters) {
      Object.assign(baseConfig, this.map4G4GHandoverParameters(config.handoverParameters));
    }

    // Add capacity parameters
    if (config.capacityParameters) {
      Object.assign(baseConfig, this.map4G4GCapacityParameters(config.capacityParameters));
    }

    return { ...baseConfig, ...bandConfig };
  }

  /**
   * Generate 4G4G specific conditions
   */
  private generate4G4GConditions(config: FrequencyRelationConfig): Record<string, any> {
    return {
      'load_balancing_active': {
        if: '${sourceCellLoad} > 80 && ${targetCellLoad} < 60',
        then: {
          'EUtranFreqRelation.cellIndividualOffset': 3,
          'EUtranFreqRelation.qOffsetList.dCell.qOffset1': 2
        },
        else: 'reset_to_default'
      },
      'interference_scenario': {
        if: '${interferenceLevel} > -85',
        then: {
          'EUtranFreqRelation.qRxLevMinOffset': 2,
          'EUtranFreqRelation.qQualMinOffset': 2
        },
        else: 'reset_to_default'
      },
      'capacity_optimization': {
        if: '${targetCellCapacity} > ${sourceCellCapacity}',
        then: {
          'EUtranFreqRelation.cellIndividualOffset': 4,
          'EUtranFreqRelation.isHoAllowed': true
        },
        else: {
          'EUtranFreqRelation.cellIndividualOffset': 1,
          'EUtranFreqRelation.isHoAllowed': true
        }
      }
    };
  }

  /**
   * Generate 4G4G specific evaluation operators
   */
  private generate4G4GEvaluations(config: FrequencyRelationConfig): Record<string, any> {
    return {
      'optimal_offset': {
        eval: 'calculateOptimal4G4GOffset',
        args: ['${sourceCellLoad}', '${targetCellLoad}', '${interferenceLevel}', '${distance}']
      },
      'handover_probability': {
        eval: 'calculateHandoverProbability',
        args: ['${signalQuality}', '${cellIndividualOffset}', '${hysteresis}']
      }
    };
  }

  /**
   * Generate 4G4G specific custom functions
   */
  private generate4G4GFunctions(config: FrequencyRelationConfig): any[] {
    return [
      {
        name: 'calculateOptimal4G4GOffset',
        args: ['sourceLoad', 'targetLoad', 'interference', 'distance'],
        body: [
          'const loadFactor = Math.max(0, (sourceLoad - targetLoad) / 20);',
          'const interferenceFactor = Math.max(0, (interference + 100) / 15);',
          'const distanceFactor = Math.max(0, Math.min(1, distance / 2000)); // 2km normalization',
          'const optimalOffset = Math.floor(loadFactor * 3 + interferenceFactor * 2 + distanceFactor * 1);',
          'return Math.min(6, Math.max(0, optimalOffset));'
        ]
      },
      {
        name: 'calculateHandoverProbability',
        args: ['signalQuality', 'offset', 'hysteresis'],
        body: [
          'const adjustedSignalQuality = signalQuality + offset;',
          'const hysteresisMargin = hysteresis || 4;',
          'const probability = Math.max(0, Math.min(1, (adjustedSignalQuality + hysteresisMargin) / 20));',
          'return probability;'
        ]
      }
    ];
  }

  /**
   * Generate 4G4G validation rules
   */
  private generate4G4GValidationRules(config: FrequencyRelationConfig): any[] {
    return [
      {
        ruleId: '4g4g_frequency_separation',
        type: 'parameter_constraint',
        condition: 'Math.abs(sourceFreq - targetFreq) < 5',
        action: 'warning',
        message: 'Frequency separation less than 5MHz may cause interference',
        enabled: true
      },
      {
        ruleId: '4g4g_handover_enabled',
        type: 'consistency_check',
        condition: 'isHoAllowed != true',
        action: 'error',
        message: 'Handover must be allowed for 4G4G frequency relations',
        enabled: true
      }
    ];
  }

  // ============================================================================
  // 4G5G EN-DC FREQUENCY RELATION GENERATION METHODS
  // ============================================================================

  /**
   * Generate 4G5G EN-DC specific configuration
   */
  private generate4G5GConfiguration(config: FrequencyRelationConfig): Record<string, any> {
    const baseConfig = {
      ...config.parameters,
      'NRCellRelation.isNrHoAllowed': true,
      'NRCellRelation.isEndcAvailable': true,
      'NRCellRelation.scgActivationStatus': 'ACTIVE',
      'NRCellRelation.scgDeactivationStatus': 'ACTIVE'
    };

    // Add EN-DC specific parameters
    const endcConfig = {
      'NRCellRelation.nrArfcn': this.getNRARFCN(config.targetBand),
      'NRCellRelation.ssbFrequency': this.getSSBFrequency(config.targetBand),
      'NRCellRelation.ssbSubcarrierSpacing': 30, // kHz
      'NRCellRelation.ssbPatternType': 'A',
      'NRCellRelation.nrScsSpecificCarrierList': {
        'nrScsSpecificCarrier': {
          'offsetToCarrier': 0,
          'subcarrierSpacing': 30,
          'carrierBandwidth': this.getCarrierBandwidth(config.targetBand)
        }
      }
    };

    // Add dual connectivity parameters
    const dualConnectivityConfig = {
      'NRCellRelation.isScgFailureRecoveryEnabled': true,
      'NRCellRelation.scgFailureRecoveryTimer': 100,
      'NRCellRelation.maxScgFailures': 3
    };

    return { ...baseConfig, ...endcConfig, ...dualConnectivityConfig };
  }

  /**
   * Generate 4G5G EN-DC specific conditions
   */
  private generate4G5GConditions(config: FrequencyRelationConfig): Record<string, any> {
    return {
      'endc_activation': {
        if: '${ueCapabilityNR} == true && ${signalQualityNR} > -110',
        then: {
          'NRCellRelation.isEndcAvailable': true,
          'NRCellRelation.scgActivationStatus': 'ACTIVE'
        },
        else: {
          'NRCellRelation.isEndcAvailable': false,
          'NRCellRelation.scgActivationStatus': 'INACTIVE'
        }
      },
      'load_balancing_endc': {
        if: '${lteCellLoad} > 85 && ${nrCellLoad} < 50',
        then: {
          'NRCellRelation.nrPrio': 2,
          'NRCellRelation.qOffsetNR': 3
        },
        else: {
          'NRCellRelation.nrPrio': 1,
          'NRCellRelation.qOffsetNR': 0
        }
      },
      'coverage_enhancement': {
        if: '${distanceFromgNB} > 2000',
        then: {
          'NRCellRelation.isEndcAvailable': false,
          'fallbackToLTE': true
        },
        else: 'maintain_endc'
      }
    };
  }

  /**
   * Generate 4G5G EN-DC specific evaluation operators
   */
  private generate4G5GEvaluations(config: FrequencyRelationConfig): Record<string, any> {
    return {
      'endc_throughput_gain': {
        eval: 'calculateENDCThroughputGain',
        args: ['${nrSignalQuality}', '${nrCellLoad}', '${ueCapability}']
      },
      'optimal_split_ratio': {
        eval: 'calculateOptimalSplitRatio',
        args: ['${lteCapacity}', '${nrCapacity}', '${totalDemand}']
      }
    };
  }

  /**
   * Generate 4G5G EN-DC specific custom functions
   */
  private generate4G5GFunctions(config: FrequencyRelationConfig): any[] {
    return [
      {
        name: 'calculateENDCThroughputGain',
        args: ['nrSignalQuality', 'nrCellLoad', 'ueCapability'],
        body: [
          'const signalFactor = Math.max(0, (nrSignalQuality + 130) / 30);',
          'const loadFactor = Math.max(0.3, 1 - nrCellLoad / 100);',
          'const capabilityFactor = ueCapability ? 1.0 : 0.5;',
          'const throughputGain = Math.floor(signalFactor * loadFactor * capabilityFactor * 500); // Mbps',
          'return Math.max(0, Math.min(1000, throughputGain));'
        ]
      },
      {
        name: 'calculateOptimalSplitRatio',
        args: ['lteCapacity', 'nrCapacity', 'totalDemand'],
        body: [
          'const totalCapacity = lteCapacity + nrCapacity;',
          'const lteRatio = Math.max(0.2, Math.min(0.8, lteCapacity / totalCapacity));',
          'const nrRatio = 1 - lteRatio;',
          'return { lte: lteRatio, nr: nrRatio };'
        ]
      }
    ];
  }

  /**
   * Generate 4G5G validation rules
   */
  private generate4G5GValidationRules(config: FrequencyRelationConfig): any[] {
    return [
      {
        ruleId: 'endc_capability_check',
        type: 'consistency_check',
        condition: 'isEndcAvailable == true && nrArfcn == null',
        action: 'error',
        message: 'EN-DC availability requires valid NR ARFCN configuration',
        enabled: true
      },
      {
        ruleId: 'nr_scg_config',
        type: 'parameter_constraint',
        condition: 'scgActivationStatus != "ACTIVE" && isEndcAvailable == true',
        action: 'warning',
        message: 'EN-DC available but SCG not activated - performance may be degraded',
        enabled: true
      }
    ];
  }

  // ============================================================================
  // 5G5G NR-NR DC FREQUENCY RELATION GENERATION METHODS
  // ============================================================================

  /**
   * Generate 5G5G NR-NR DC specific configuration
   */
  private generate5G5GConfiguration(config: FrequencyRelationConfig): Record<string, any> {
    const baseConfig = {
      ...config.parameters,
      'NRCellRelation.isNrNrDcEnabled': true,
      'NRCellRelation.dcCarrierType': 'SCG', // Secondary Cell Group
      'NRCellRelation.isMrdcAvailable': true
    };

    // Add NR-NR DC specific parameters
    const nrdcConfig = {
      'NRCellRelation.scgCellGroupId': 1,
      'NRCellRelation.pdcchBlindDetectionNr': 2,
      'NRCellRelation.pdschTimeDomainAllocationList': {
        'pdschTimeDomainAllocation': [
          { 'k2': 2, 'mappingType': 'typeA', 'startSymbolAndLength': 2 },
          { 'k2': 3, 'mappingType': 'typeA', 'startSymbolAndLength': 2 }
        ]
      },
      'NRCellRelation.puschTimeDomainAllocationList': {
        'puschTimeDomainAllocation': [
          { 'k2': 2, 'mappingType': 'typeA', 'startSymbolAndLength': 2 },
          { 'k2': 3, 'mappingType': 'typeA', 'startSymbolAndLength': 2 }
        ]
      }
    };

    return { ...baseConfig, ...nrdcConfig };
  }

  /**
   * Generate 5G5G NR-NR DC specific conditions
   */
  private generate5G5GConditions(config: FrequencyRelationConfig): Record<string, any> {
    return {
      'nrdc_activation': {
        if: '${ueCapabilityNRDC} == true && ${signalQualityTarget} > -105',
        then: {
          'NRCellRelation.isNrNrDcEnabled': true,
          'NRCellRelation.isMrdcAvailable': true
        },
        else: {
          'NRCellRelation.isNrNrDcEnabled': false,
          'NRCellRelation.isMrdcAvailable': false
        }
      },
      'carrier_aggregation': {
        if: '${totalBandwidth} > 200', // MHz
        then: {
          'NRCellRelation.caEnabled': true,
          'NRCellRelation.caBandwidthClass': 'nCA'
        },
        else: {
          'NRCellRelation.caEnabled': false
        }
      }
    };
  }

  /**
   * Generate 5G5G NR-NR DC specific evaluation operators
   */
  private generate5G5GEvaluations(config: FrequencyRelationConfig): Record<string, any> {
    return {
      'nrdc_capacity_gain': {
        eval: 'calculateNRDCCapacityGain',
        args: ['${primaryCellCapacity}', '${secondaryCellCapacity}', '${trafficType}']
      },
      'optimal_carrier_selection': {
        eval: 'selectOptimalCarrier',
        args: ['${signalQualityPCell}', '${signalQualitySCell}', '${trafficDemand}']
      }
    };
  }

  /**
   * Generate 5G5G NR-NR DC specific custom functions
   */
  private generate5G5GFunctions(config: FrequencyRelationConfig): any[] {
    return [
      {
        name: 'calculateNRDCCapacityGain',
        args: ['primaryCapacity', 'secondaryCapacity', 'trafficType'],
        body: [
          'const baseCapacity = primaryCapacity + secondaryCapacity;',
          'const trafficMultiplier = trafficType === "high_speed" ? 1.5 : 1.2;',
          'const capacityGain = Math.floor(baseCapacity * trafficMultiplier);',
          'return Math.max(primaryCapacity, capacityGain);'
        ]
      },
      {
        name: 'selectOptimalCarrier',
        args: ['pCellQuality', 'sCellQuality', 'trafficDemand'],
        body: [
          'const qualityDifference = sCellQuality - pCellQuality;',
          'const loadThreshold = trafficDemand > 0.8 ? 3 : 1; // dB',
          'if (qualityDifference > loadThreshold) {',
          '  return "secondary";',
          '} else {',
          '  return "primary";',
          '}'
        ]
      }
    ];
  }

  /**
   * Generate 5G5G validation rules
   */
  private generate5G5GValidationRules(config: FrequencyRelationConfig): any[] {
    return [
      {
        ruleId: 'nrdc_capability_check',
        type: 'consistency_check',
        condition: 'isNrNrDcEnabled == true && ueCapabilityNRDC != true',
        action: 'error',
        message: 'NR-NR DC enabled but UE capability not supported',
        enabled: true
      }
    ];
  }

  // ============================================================================
  // 5G4G FALLBACK FREQUENCY RELATION GENERATION METHODS
  // ============================================================================

  /**
   * Generate 5G4G fallback specific configuration
   */
  private generate5G4GConfiguration(config: FrequencyRelationConfig): Record<string, any> {
    const baseConfig = {
      ...config.parameters,
      'NRCellRelation.isFallbackToLTE': true,
      'NRCellRelation.fallbackTriggerType': 'SIGNAL_QUALITY',
      'NRCellRelation.fallbackThreshold': -110, // dBm
      'NRCellRelation.fallbackHysteresis': 2, // dB
      'NRCellRelation.fallbackTimeToTrigger': 640 // ms
    };

    return baseConfig;
  }

  /**
   * Generate 5G4G fallback specific conditions
   */
  private generate5G4GConditions(config: FrequencyRelationConfig): Record<string, any> {
    return {
      'nr_coverage_loss': {
        if: '${nrSignalQuality} < ${fallbackThreshold}',
        then: {
          'NRCellRelation.isFallbackToLTE': true,
          'initiateHandoverToLTE': true
        },
        else: {
          'NRCellRelation.isFallbackToLTE': false,
          'maintainNRConnection': true
        }
      },
      'interference_scenario': {
        if: '${nrInterferenceLevel} > -95',
        then: {
          'NRCellRelation.fallbackThreshold': -105, // More aggressive fallback
          'NRCellRelation.fallbackTimeToTrigger': 320
        },
        else: 'use_defaults'
      }
    };
  }

  /**
   * Generate 5G4G fallback specific evaluation operators
   */
  private generate5G4GEvaluations(config: FrequencyRelationConfig): Record<string, any> {
    return {
      'fallback_probability': {
        eval: 'calculateFallbackProbability',
        args: ['${nrSignalQuality}', '${lteSignalQuality}', '${interferenceLevel}']
      }
    };
  }

  /**
   * Generate 5G4G fallback specific custom functions
   */
  private generate5G4GFunctions(config: FrequencyRelationConfig): any[] {
    return [
      {
        name: 'calculateFallbackProbability',
        args: ['nrQuality', 'lteQuality', 'interference'],
        body: [
          'const qualityDifference = lteQuality - nrQuality;',
          'const interferenceFactor = Math.max(0, (interference + 100) / 20);',
          'const fallbackProbability = Math.max(0, Math.min(1, (qualityDifference + interferenceFactor * 3) / 15));',
          'return fallbackProbability;'
        ]
      }
    ];
  }

  /**
   * Generate 5G4G validation rules
   */
  private generate5G4GValidationRules(config: FrequencyRelationConfig): any[] {
    return [
      {
        ruleId: 'fallback_threshold_validation',
        type: 'parameter_constraint',
        condition: 'fallbackThreshold > -100',
        action: 'warning',
        message: 'Fallback threshold too high may cause unnecessary handovers',
        enabled: true
      }
    ];
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  /**
   * Initialize frequency compatibility matrix
   */
  private initializeCompatibilityMatrix(): FrequencyCompatibilityMatrix {
    const matrix: Record<string, Record<string, boolean>> = {};

    // Define compatibility between frequency bands
    const compatiblePairs = [
      ['lte_800', 'lte_1800'],
      ['lte_800', 'lte_2100'],
      ['lte_1800', 'lte_2100'],
      ['lte_1800', 'lte_2600'],
      ['lte_2100', 'lte_2600'],
      ['lte_1800', 'nr_3500'], // 4G5G EN-DC
      ['lte_2100', 'nr_3500'], // 4G5G EN-DC
      ['lte_2600', 'nr_3500'], // 4G5G EN-DC
      ['nr_3500', 'nr_26000'], // 5G5G NR-NR DC
      ['nr_3500', 'nr_28000'], // 5G5G NR-NR DC
      ['nr_700', 'lte_800'], // 5G4G fallback
      ['nr_3500', 'lte_1800'] // 5G4G fallback
    ];

    for (const [band1, band2] of compatiblePairs) {
      if (!matrix[band1]) matrix[band1] = {};
      if (!matrix[band2]) matrix[band2] = {};
      matrix[band1][band2] = true;
      matrix[band2][band1] = true;
    }

    const recommendations: FrequencyRecommendation[] = [
      {
        sourceBand: FrequencyBand.LTE_1800,
        targetBand: FrequencyBand.NR_3500,
        recommendation: 'recommended',
        reason: 'Optimal EN-DC pairing with good coverage and capacity balance',
        useCase: ['urban', 'suburban', 'high_capacity']
      },
      {
        sourceBand: FrequencyBand.LTE_2100,
        targetBand: FrequencyBand.NR_3500,
        recommendation: 'recommended',
        reason: 'Wide coverage area with good capacity potential',
        useCase: ['urban', 'business_district', 'stadium']
      }
    ];

    const constraints: FrequencyConstraint[] = [
      {
        bands: [FrequencyBand.LTE_800, FrequencyBand.NR_3500],
        constraint: 'guard_band_required',
        parameters: { 'guardBandSize': 10 }, // MHz
        description: 'Guard band required for 800MHz LTE and 3.5GHz NR coexistence'
      }
    ];

    return {
      matrix,
      recommendations,
      constraints
    };
  }

  /**
   * Get 4G4G band-specific parameters
   */
  private get4G4GBandParameters(sourceBand: FrequencyBand, targetBand: FrequencyBand): Record<string, any> {
    const bandParams: Record<string, Record<string, any>> = {
      'lte_800-lte_1800': {
        'EUtranFreqRelation.threshXHigh': 12,
        'EUtranFreqRelation.threshXLow': 0
      },
      'lte_1800-lte_2100': {
        'EUtranFreqRelation.threshXHigh': 8,
        'EUtranFreqRelation.threshXLow': -2
      },
      'lte_2100-lte_2600': {
        'EUtranFreqRelation.threshXHigh': 10,
        'EUtranFreqRelation.threshXLow': -4
      }
    };

    const key = `${sourceBand}-${targetBand}`;
    return bandParams[key] || bandParams['lte_1800-lte_2100'];
  }

  /**
   * Map 4G4G handover parameters
   */
  private map4G4GHandoverParameters(params: HandoverParameterConfig): Record<string, any> {
    return {
      'EUtranFreqRelation.hysteresis': params.hysteresis,
      'EUtranFreqRelation.a3Offset': params.a3Offset,
      'EUtranFreqRelation.a5Offset1': params.a5Offset1,
      'EUtranFreqRelation.a5Offset2': params.a5Offset2,
      'EUtranFreqRelation.timeToTrigger': params.timeToTrigger,
      ...params.parameters
    };
  }

  /**
   * Map 4G4G capacity parameters
   */
  private map4G4GCapacityParameters(params: CapacityParameterConfig): Record<string, any> {
    return {
      'EUtranFreqRelation.loadBalancingEnabled': params.loadBalancing,
      'EUtranFreqRelation.cellIndividualOffset': params.cellIndividualOffset,
      ...params.parameters
    };
  }

  /**
   * Get NR ARFCN for frequency band
   */
  private getNRARFCN(band: FrequencyBand): number {
    const arfcnMap: Record<string, number> = {
      'nr_700': 620000,
      'nr_3500': 520000,
      'nr_26000': 2220000,
      'nr_28000': 2400000
    };
    return arfcnMap[band] || 520000;
  }

  /**
   * Get SSB frequency for NR band
   */
  private getSSBFrequency(band: FrequencyBand): number {
    const ssbMap: Record<string, number> = {
      'nr_700': 751,
      'nr_3500': 3610,
      'nr_26000': 26270,
      'nr_28000': 28270
    };
    return ssbMap[band] || 3610;
  }

  /**
   * Get carrier bandwidth for NR band
   */
  private getCarrierBandwidth(band: FrequencyBand): number {
    const bandwidthMap: Record<string, number> = {
      'nr_700': 20, // MHz
      'nr_3500': 100,
      'nr_26000': 400,
      'nr_28000': 400
    };
    return bandwidthMap[band] || 100;
  }
}