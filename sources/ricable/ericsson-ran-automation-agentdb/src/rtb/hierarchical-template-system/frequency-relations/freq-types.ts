/**
 * Frequency Relations Type Definitions
 *
 * Comprehensive type system for inter-frequency relationships in Ericsson RAN
 * covering 4G4G, 4G5G, 5G5G, and 5G4G scenarios
 */

import type {
  EUtranCellFDD,
  EUtranFreqRelation,
  NRCellCU,
  NRFreqRelation,
  EutranExternalNrcellFDD,
  NrcellDu,
  CarrierAggregation,
  ENDCConfiguration,
  NRDCConfiguration
} from '../../types/rtb-types';

/**
 * Base frequency relation configuration
 */
export interface BaseFrequencyRelation {
  /** Unique identifier for the frequency relation */
  relationId: string;
  /** Reference frequency band (anchor) */
  referenceFreq: FrequencyBand;
  /** Related frequency band (target) */
  relatedFreq: FrequencyBand;
  /** Relation type classification */
  relationType: FrequencyRelationType;
  /** Priority level for template inheritance */
  priority: number;
  /** Administrative state */
  adminState: 'UNLOCKED' | 'LOCKED';
  /** Operational state */
  operState: 'ENABLED' | 'DISABLED';
  /** Creation timestamp */
  createdAt: Date;
  /** Last modification timestamp */
  modifiedAt: Date;
}

/**
 * Frequency band definition
 */
export interface FrequencyBand {
  /** Band number (e.g., 1, 3, 7, 20, 78) */
  bandNumber: number;
  /** Frequency range in MHz */
  frequencyRange: {
    uplink?: { start: number; end: number };
    downlink: { start: number; end: number };
  };
  /** Band category */
  bandCategory: 'LTE' | 'NR' | 'MMWAVE';
  /** Primary use case */
  primaryUse: 'COVERAGE' | 'CAPACITY' | 'HOTSPOT' | 'INDOOR';
}

/**
 * Frequency relation type enumeration
 */
export type FrequencyRelationType =
  | '4G4G'    // LTE inter-frequency
  | '4G5G'    // EN-DC (E-UTRAN-NR Dual Connectivity)
  | '5G5G'    // NR-NR Dual Connectivity
  | '5G4G';   // 5G to 4G fallback

/**
 * Handover configuration parameters
 */
export interface HandoverConfiguration {
  /** Handover trigger type */
  triggerType: 'A3' | 'A4' | 'A5' | 'B1' | 'B2';
  /** Hysteresis in dB */
  hysteresis: number;
  /** Time-to-trigger in milliseconds */
  timeToTrigger: number;
  /** Cell individual offset in dB */
  cellIndividualOffset: number;
  /** Frequency specific offset in dB */
  freqSpecificOffset: number;
  /** Event-based configuration */
  eventBasedConfig?: {
    threshold1?: number;
    threshold2?: number;
    a3Offset?: number;
  };
  /** Measurement configuration */
  measurementConfig: {
    reportInterval: number;
    maxReportCells: number;
    reportAmount: '1' | '2' | '4' | '8' | '16' | '32' | '64' | 'INFINITY';
  };
}

/**
 * Capacity sharing parameters
 */
export interface CapacitySharingParams {
  /** Capacity sharing enabled */
  enabled: boolean;
  /** Sharing strategy */
  strategy: 'LOAD_BALANCING' | 'FAIR_SHARE' | 'PRIORITY_BASED';
  /** Load balancing threshold in percentage */
  loadBalancingThreshold: number;
  /** Maximum capacity ratio */
  maxCapacityRatio: number;
  /** Minimum guaranteed capacity */
  minGuaranteedCapacity: number;
  /** Dynamic load rebalancing */
  dynamicRebalancing: boolean;
  /** Rebalancing interval in seconds */
  rebalancingInterval: number;
}

/**
 * Interference mitigation settings
 */
export interface InterferenceSettings {
  /** Interference coordination enabled */
  enabled: boolean;
  /** Coordination type */
  coordinationType: 'ICIC' | 'eICIC' | 'FeICIC';
  /** Inter-band interference management */
  interBandManagement: {
    /** Almost blank subframes */
    almostBlankSubframes: boolean;
    /** Cell-specific reference signal power boost */
    crsPowerBoost: number;
    /** Power control coordination */
    powerControlCoordination: boolean;
  };
  /** Dynamic interference coordination */
  dynamicCoordination: boolean;
  /** Coordination update interval in milliseconds */
  coordinationInterval: number;
}

/**
 * 4G4G Frequency Relation Configuration
 */
export interface Freq4G4GRelation extends BaseFrequencyRelation {
  relationType: '4G4G';
  priority: 50;

  /** LTE specific parameters */
  lteConfig: {
    /** Carrier aggregation enabled */
    carrierAggregation: boolean;
    /** CA configuration if enabled */
    caConfig?: {
      primaryCell: string;
      secondaryCells: string[];
      maxAggregatedBandwidth: number;
      crossCarrierScheduling: boolean;
    };
    /** Inter-frequency mobility parameters */
    mobilityParams: {
      /** Handover preparation timeout */
      handoverPreparationTimeout: number;
      /** Handover execution timeout */
      handoverExecutionTimeout: number;
      /** Reestablishment allowed */
      reestablishmentAllowed: boolean;
    };
    /** Measurement gap configuration */
    measurementGapConfig: {
      gapPattern: 'GP0' | 'GP1' | 'GP2';
      gapOffset: number;
      gapLength: number;
      gapRepetitionPeriod: number;
    };
  };
}

/**
 * 4G5G Frequency Relation Configuration (EN-DC)
 */
export interface Freq4G5GRelation extends BaseFrequencyRelation {
  relationType: '4G5G';
  priority: 60;

  /** EN-DC specific parameters */
  endcConfig: {
    /** Master eNodeB (MeNB) configuration */
    meNbConfig: {
      /** Split bearer support */
      splitBearerSupport: boolean;
      /** Dual connectivity MRDC support */
      dualConnectivitySupport: boolean;
      /** EN-DC release version */
      releaseVersion: 'REL15' | 'REL16' | 'REL17';
    };
    /** Secondary gNodeB (SgNB) configuration */
    sgNbConfig: {
      /** SgNB addition allowed */
      sgNbAdditionAllowed: boolean;
      /** SgNB modification allowed */
      sgNbModificationAllowed: boolean;
      /** SgNB release allowed */
      sgNbReleaseAllowed: boolean;
      /** Maximum number of SgNBs */
      maxSgNbPerUe: number;
    };
    /** PDCP duplication */
    pdcpDuplication: {
      enabled: boolean;
      duplicationActivation: 'RLC' | 'MAC';
      duplicationDeactivation: 'RLC' | 'MAC';
    };
    /** EN-DC specific measurements */
    endcMeasurements: {
      nrEventB1: {
        threshold: number;
        hysteresis: number;
        timeToTrigger: number;
      };
      nrEventB2?: {
        threshold1: number;
        threshold2: number;
        hysteresis: number;
        timeToTrigger: number;
      };
    };
  };
}

/**
 * 5G5G Frequency Relation Configuration (NR-DC)
 */
export interface Freq5G5GRelation extends BaseFrequencyRelation {
  relationType: '5G5G';
  priority: 70;

  /** NR-DC specific parameters */
  nrdcConfig: {
    /** Primary Cell Group (PCell) configuration */
    pCellConfig: {
      /** Cell type */
      cellType: 'PCELL' | 'PSCell' | 'SCELL';
      /** Serving cell priority */
      servingCellPriority: number;
      /** Cell reselection priority */
      cellReselectionPriority: number;
    };
    /** Secondary Cell Group (SCG) configuration */
    scgConfig: {
      /** SCG addition supported */
      scgAdditionSupported: boolean;
      /** SCG change supported */
      scgChangeSupported: boolean;
      /** SCG release supported */
      scgReleaseSupported: boolean;
      /** Maximum number of SCGs */
      maxScgPerUe: number;
    };
    /** Multi-band carrier aggregation */
    mbcaConfig: {
      enabled: boolean;
      aggregatedBands: number[];
      maxAggregatedBandwidth: number;
      crossScheduling: boolean;
      dynamicSlotAllocation: boolean;
    };
    /** Advanced beam management */
    beamManagement: {
      /** Beam failure recovery */
      beamFailureRecovery: boolean;
      /** Beam management configuration */
      beamManagementConfig: {
        maxBeamCandidates: number;
        beamReportInterval: number;
        beamSwitchingTime: number;
      };
    };
    /** Dynamic spectrum sharing */
    dssConfig: {
      enabled: boolean;
      sharingMode: 'STATIC' | 'DYNAMIC';
      spectrumAllocation: {
        nrShare: number;
        lteShare: number;
      };
    };
  };
}

/**
 * 5G4G Frequency Relation Configuration (Fallback)
 */
export interface Freq5G4GRelation extends BaseFrequencyRelation {
  relationType: '5G4G';
  priority: 80;

  /** Fallback specific parameters */
  fallbackConfig: {
    /** Fallback trigger conditions */
    fallbackTriggers: {
      /** NR coverage threshold */
      nrCoverageThreshold: number;
      /** 5G service interruption time */
      serviceInterruptionTime: number;
      /** UE capability fallback */
      ueCapabilityFallback: boolean;
      /** Network congestion fallback */
      networkCongestionFallback: boolean;
    };
    /** Fallback handover configuration */
    fallbackHandover: {
      /** Prepare fallback timeout */
      prepareFallbackTimeout: number;
      /** Execute fallback timeout */
      executeFallbackTimeout: number;
      /** Fallback preparation retry count */
      fallbackPreparationRetryCount: number;
      /** Immediate fallback allowed */
      immediateFallbackAllowed: boolean;
    };
    /** Service continuity */
    serviceContinuity: {
      /** Session continuity */
      sessionContinuity: boolean;
      /** IP address preservation */
      ipAddressPreservation: boolean;
      /** QoS preservation */
      qosPreservation: boolean;
    };
    /** Return to 5G configuration */
    returnTo5G: {
      enabled: boolean;
      /** Return trigger conditions */
      returnTriggers: {
        nrCoverageImprovement: number;
        nrServiceQuality: number;
        networkLoadImprovement: number;
      };
      /** Return evaluation interval */
      returnEvaluationInterval: number;
      /** Minimum 5G stay time */
      min5GStayTime: number;
    };
  };
}

/**
 * Union type for all frequency relation configurations
 */
export type FrequencyRelation =
  | Freq4G4GRelation
  | Freq4G5GRelation
  | Freq5G5GRelation
  | Freq5G4GRelation;

/**
 * Frequency relation template configuration
 */
export interface FrequencyRelationTemplate {
  /** Template unique identifier */
  templateId: string;
  /** Template name */
  templateName: string;
  /** Template description */
  templateDescription: string;
  /** Template version */
  version: string;
  /** Template type */
  templateType: FrequencyRelationType;
  /** Priority level */
  priority: number;
  /** Base frequency relation configuration */
  baseConfig: FrequencyRelation;
  /** Template parameters for customization */
  parameters: TemplateParameter[];
  /** Validation rules */
  validationRules: ValidationRule[];
  /** cmedit command templates */
  cmeditTemplates: CmeditCommandTemplate[];
}

/**
 * Template parameter definition
 */
export interface TemplateParameter {
  /** Parameter name */
  name: string;
  /** Parameter type */
  type: 'STRING' | 'INTEGER' | 'FLOAT' | 'BOOLEAN' | 'ENUM';
  /** Parameter description */
  description: string;
  /** Default value */
  defaultValue: any;
  /** Allowed values (for ENUM type) */
  allowedValues?: any[];
  /** Validation constraints */
  constraints?: {
    min?: number;
    max?: number;
    pattern?: string;
    required?: boolean;
  };
  /** Parameter category */
  category: 'BASIC' | 'ADVANCED' | 'EXPERT';
}

/**
 * Validation rule definition
 */
export interface ValidationRule {
  /** Rule name */
  name: string;
  /** Rule description */
  description: string;
  /** Rule type */
  type: 'RANGE' | 'DEPENDENCY' | 'CONSISTENCY' | 'PERFORMANCE';
  /** Rule condition */
  condition: string;
  /** Rule action */
  action: 'WARNING' | 'ERROR' | 'AUTO_CORRECT';
}

/**
 * cmedit command template
 */
export interface CmeditCommandTemplate {
  /** Command name */
  commandName: string;
  /** Command template */
  commandTemplate: string;
  /** Parameter mapping */
  parameterMapping: Record<string, string>;
  /** Command description */
  description: string;
  /** Execution conditions */
  executionConditions?: string[];
}

/**
 * Frequency relation optimization metrics
 */
export interface FrequencyRelationMetrics {
  /** Handover success rate */
  handoverSuccessRate: number;
  /** Average handover latency */
  averageHandoverLatency: number;
  /** Interference level */
  interferenceLevel: number;
  /** Capacity utilization */
  capacityUtilization: number;
  /** User throughput */
  userThroughput: {
    average: number;
    peak: number;
    cellEdge: number;
  };
  /** Call drop rate */
  callDropRate: number;
  /** Setup success rate */
  setupSuccessRate: number;
}

/**
 * Frequency relation optimization recommendation
 */
export interface FrequencyRelationRecommendation {
  /** Recommendation ID */
  id: string;
  /** Target relation ID */
  relationId: string;
  /** Recommendation type */
  type: 'PARAMETER_TUNING' | 'TOPOLOGY_CHANGE' | 'FEATURE_ACTIVATION';
  /** Priority */
  priority: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  /** Expected impact */
  expectedImpact: {
    performanceImprovement: number;
    capacityGain: number;
    interferenceReduction: number;
  };
  /** Recommended changes */
  recommendedChanges: ParameterChange[];
  /** Implementation complexity */
  implementationComplexity: 'LOW' | 'MEDIUM' | 'HIGH';
  /** Risk assessment */
  riskAssessment: 'LOW' | 'MEDIUM' | 'HIGH';
}

/**
 * Parameter change definition
 */
export interface ParameterChange {
  /** Parameter path */
  path: string;
  /** Current value */
  currentValue: any;
  /** Recommended value */
  recommendedValue: any;
  /** Change reason */
  reason: string;
  /** Validation status */
  validationStatus: 'VALID' | 'WARNING' | 'INVALID';
}