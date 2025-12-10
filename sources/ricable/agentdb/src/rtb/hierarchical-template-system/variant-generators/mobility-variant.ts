/**
 * High Mobility Variant Generator
 *
 * Priority 30 templates optimized for fast train, motorway, and high-speed mobility scenarios.
 * Focuses on handover optimization, mobility robustness, and speed-based adaptation.
 */

import { VariantGeneratorCore, VariantConfig, VariantOptimization } from '../variant-generator-core';

export interface MobilityDeploymentContext {
  mobilityType: 'high_speed_train' | 'motorway' | 'airport' | 'mixed_transport';
  speedRange: { min: number; max: number }; // km/h
  handoverFrequency: 'low' | 'medium' | 'high' | 'very_high';
  cellSize: 'small' | 'medium' | 'large';
  trafficPattern: 'continuous' | 'bursty' | 'scheduled';
  servicePriority: 'voice' | 'data' | 'mixed';
  redundancyLevel: 'standard' | 'high' | 'maximum';
}

export class MobilityVariantGenerator extends VariantGeneratorCore {
  private readonly mobilityOptimizations: VariantOptimization[] = [
    // Handover Parameter Optimization
    {
      parameter: 'EUtranCellFDD.handoverParameters',
      value: {
        a3Offset: 2,
        a3Hysteresis: 1,
        timeToTrigger: { ttT312: 128, ttT313: 256 },
        cellIndividualOffset: 0,
        handoverMode: 'predictive',
        makeBeforeBreak: true
      },
      context: 'handover_optimization',
      description: 'Optimized handover parameters for high-speed mobility',
      priority: 1
    },
    {
      parameter: 'EUtranCellFDD.mobilityRobustness',
      value: {
        enabled: true,
        pingPongPrevention: true,
        handoverFailureRecovery: true,
        adaptiveHysteresis: true,
        speedBasedOptimization: true
      },
      context: 'mobility_robustness',
      description: 'Mobility robustness features to prevent ping-pong and failures',
      priority: 1
    },

    // High-Speed Train Specific
    {
      parameter: 'EUtranCellFDD.trainMode',
      value: {
        enabled: true,
        directionalAntenna: true,
        beamSteering: true,
        predictiveHandover: true,
        dopplerCompensation: true,
        cellBlasting: true
      },
      context: 'train_optimization',
      description: 'High-speed train mode with directional coverage',
      priority: 1,
      conditions: ['mobilityType_high_speed_train']
    },

    // Motorway Specific
    {
      parameter: 'EUtranCellFDD.motorwayMode',
      value: {
        enabled: true,
        longCellCoverage: true,
        highGainAntenna: true,
        overlappingCells: true,
        fastHandover: true,
        speedDetection: true
      },
      context: 'motorway_optimization',
      description: 'Motorway mode with extended coverage and fast handover',
      priority: 1,
      conditions: ['mobilityType_motorway']
    },

    // Cell Reselection Optimization
    {
      parameter: 'EUtranCellFDD.cellReselection',
      value: {
        priorityReselection: true,
        fastReselection: true,
        speedBasedReselection: true,
        hysteresis: 2,
        reselectionTimer: 1,
        minimumRxLevel: -104
      },
      context: 'cell_reselection',
      description: 'Fast cell reselection optimized for mobile users',
      priority: 2
    },

    // Speed-Based Parameter Adaptation
    {
      parameter: 'EUtranCellFDD.speedAdaptation',
      value: {
        enabled: true,
        speedThresholds: [
          { speed: 0, hysteresis: 4, timeToTrigger: 512 },
          { speed: 30, hysteresis: 3, timeToTrigger: 256 },
          { speed: 60, hysteresis: 2, timeToTrigger: 128 },
          { speed: 120, hysteresis: 1, timeToTrigger: 64 }
        ],
        adaptiveParameters: ['hysteresis', 'timeToTrigger', 'cellIndividualOffset']
      },
      context: 'speed_adaptation',
      description: 'Dynamic parameter adaptation based on user speed',
      priority: 2
    },

    // Enhanced Mobility Management
    {
      parameter: 'MobilityManagement.configuration',
      value: {
        trackingAreaUpdate: 'frequent',
        locationUpdateTimer: 60,
        periodicTAU: true,
        pagingOptimization: true,
        mobilityStateEstimation: true
      },
      context: 'mobility_management',
      description: 'Enhanced mobility management for frequent handovers',
      priority: 2
    },

    // Doppler Compensation
    {
      parameter: 'PhysicalLayer.dopplerCompensation',
      value: {
        enabled: true,
        compensationRange: { min: -1500, max: 1500 },
        adaptiveCompensation: true,
        frequencyCorrection: true,
        phaseTracking: true
      },
      context: 'doppler_compensation',
      description: 'Advanced Doppler compensation for high-speed scenarios',
      priority: 3
    },

    // Predictive Scheduling
    {
      parameter: 'SchedulerConfiguration.predictiveMode',
      value: {
        enabled: true,
        trajectoryPrediction: true,
        bufferPrediction: true,
        handoverPrediction: true,
        predictionHorizon: 2000, // ms
        confidenceThreshold: 0.8
      },
      context: 'predictive_scheduling',
      description: 'Predictive resource scheduling based on mobility patterns',
      priority: 3
    },

    // Redundancy and Reliability
    {
      parameter: 'ReliabilityConfiguration.mobility',
      value: {
        dualConnectivity: true,
        carrierAggregation: 'inter_band',
        handoverRedundancy: true,
        fallbackConnections: true,
        qualityMonitoring: true
      },
      context: 'mobility_redundancy',
      description: 'Redundancy mechanisms for high mobility reliability',
      priority: 3
    },

    // QoS for Mobility
    {
      parameter: 'QoSConfiguration.mobilityProfile',
      value: {
        voice: {
          priority: 1,
          bitrate: '32kbps',
          delay: '150ms',
          jitter: '50ms',
          packetLoss: '0.01%'
        },
        video: {
          priority: 2,
          bitrate: '2Mbps',
          delay: '200ms',
          jitter: '100ms'
        },
        data: {
          priority: 3,
          bitrate: 'best_effort',
          delay: '500ms',
          buffering: 'adaptive'
        },
        emergency: {
          priority: 0,
          bitrate: 'guaranteed',
          delay: '100ms',
          reliability: '99.999%'
        }
      },
      context: 'mobility_qos',
      description: 'QoS profile optimized for mobile users',
      priority: 2
    }
  ];

  constructor() {
    super();
    this.registerMobilityVariantConfig();
  }

  /**
   * Register mobility-specific variant configuration
   */
  private registerMobilityVariantConfig(): void {
    const mobilityConfig: VariantConfig = {
      variantType: 'mobility',
      priority: 30,
      baseTemplates: ['base_4g', 'base_5g'],
      optimizations: this.mobilityOptimizations,
      customLogic: this.getMobilityCustomFunctions(),
      conditions: this.getMobilityConditions(),
      evaluations: this.getMobilityEvaluations()
    };

    this.registerVariantConfig('mobility', mobilityConfig);
  }

  /**
   * Get mobility-specific custom functions
   */
  private getMobilityCustomFunctions() {
    return [
      {
        name: 'calculateHandoverParameters',
        args: ['userSpeed', 'cellSize', 'interferenceLevel'],
        body: [
          'const baseHysteresis = userSpeed > 120 ? 1 : userSpeed > 60 ? 2 : 3;',
          'const baseTTT = userSpeed > 120 ? 64 : userSpeed > 60 ? 128 : 256;',
          'const interferenceCompensation = interferenceLevel > -100 ? 1 : 0;',
          'return {',
          '  hysteresis: Math.max(1, baseHysteresis - interferenceCompensation),',
          '  timeToTrigger: Math.max(32, baseTTT - (interferenceCompensation * 32)),',
          '  a3Offset: cellSize === "small" ? 3 : cellSize === "medium" ? 2 : 1,',
          '};'
        ]
      },
      {
        name: 'predictiveHandoverDecision',
        args: ['trajectory', 'cellCoverage', 'currentSignal'],
        body: [
          'const nextCellInPath = findNextCellOnTrajectory(trajectory, cellCoverage);',
          'const predictedSignal = calculatePredictedSignal(nextCellInPath, trajectory.position);',
          'const currentMargin = currentSignal - nextCellInPath.threshold;',
          'const predictedMargin = predictedSignal - nextCellInPath.threshold;',
          'if (predictedMargin > currentMargin + 2 && trajectory.timeToCell < 3000) {',
          '  return { action: "prepare_handover", target: nextCellInPath.id, urgency: "high" };',
          '}',
          'return { action: "monitor", nextCheck: 500 };'
        ]
      },
      {
        name: 'optimizeForTransportType',
        args: ['transportType', 'speedProfile'],
        body: [
          'switch(transportType) {',
          '  case "high_speed_train":',
          '    return {',
          '      antennaPattern: "directional",',
          '      handoverMode: "predictive",',
          '      beamSteering: true,',
          '      cellAlignment: "linear",',
          '    };',
          '  case "motorway":',
          '    return {',
          '      antennaPattern: "sector",',
          '      handoverMode: "fast",',
          '      overlappingCoverage: true,',
          '      cellAlignment: "curved",',
          '    };',
          '  case "airport":',
          '    return {',
          '      antennaPattern: "omnidirectional",',
          '      handoverMode: "standard",',
          '      capacityFocus: true,',
          '      cellAlignment: "hexagonal",',
          '    };',
          '  default:',
          '    return { antennaPattern: "sector", handoverMode: "adaptive" };',
          '}'
        ]
      },
      {
        name: 'calculateDopplerShift',
        args: ['carrierFrequency', 'userSpeed', 'angleOfArrival'],
        body: [
          'const c = 3e8; // Speed of light in m/s',
          'const speedMs = userSpeed / 3.6; // Convert km/h to m/s',
          'const dopplerShift = (speedMs * carrierFrequency * 1e9 * Math.cos(angleOfArrival)) / c;',
          'return {',
          '  frequencyShift: dopplerShift,',
          '  compensationNeeded: Math.abs(dopplerShift) > 100,',
          '  compensationFactor: dopplerShift / carrierFrequency,',
          '};'
        ]
      },
      {
        name: 'mobilityStateEstimation',
        args: ['handoverHistory', 'speedHistory', 'locationHistory'],
        body: [
          'const avgSpeed = speedHistory.reduce((a, b) => a + b, 0) / speedHistory.length;',
          'const handoverRate = handoverHistory.length / locationHistory.length * 1000;',
          'const directionConsistency = calculateDirectionConsistency(locationHistory);',
          'let mobilityState = "stationary";',
          'if (avgSpeed > 3 && handoverRate > 0.1) {',
          '  mobilityState = avgSpeed > 30 ? "high_speed" : "medium_speed";',
          '}',
          'return {',
          '  state: mobilityState,',
          '  avgSpeed,',
          '  handoverRate,',
          '  directionConsistency,',
          '  confidence: Math.min(1.0, speedHistory.length / 10),',
          '};'
        ]
      },
      {
        name: 'calculateRedundancyLevel',
        args: ['serviceType', 'mobilityLevel', 'reliabilityRequirement'],
        body: [
          'let redundancyLevel = "standard";',
          'if (serviceType === "emergency" || reliabilityRequirement > 0.999) {',
          '  redundancyLevel = "maximum";',
          '} else if (mobilityLevel === "high_speed" || serviceType === "voice") {',
          '  redundancyLevel = "high";',
          '}',
          'return {',
          '  level: redundancyLevel,',
          '  dualConnectivity: redundancyLevel !== "standard",',
          '  additionalCarriers: redundancyLevel === "maximum" ? 2 : 1,',
          '  fallbackMechanism: true,',
          '}'
        ]
      }
    ];
  }

  /**
   * Get mobility-specific conditions
   */
  private getMobilityConditions() {
    return {
      highSpeedMode: {
        if: 'userSpeed > 120',
        then: {
          handoverMode: 'predictive',
          hysteresis: 1,
          timeToTrigger: 64,
          dopplerCompensation: true,
          dualConnectivity: true
        },
        else: {
          handoverMode: 'standard',
          hysteresis: 3,
          timeToTrigger: 256,
          dopplerCompensation: false
        }
      },
      trainMode: {
        if: 'mobilityType === "high_speed_train"',
        then: {
          antennaConfiguration: 'directional',
          cellAlignment: 'linear',
          beamSteering: true,
          predictiveScheduling: true,
          trainSpecificOptimizations: true
        },
        else: {
          antennaConfiguration: 'sector',
          cellAlignment: 'hexagonal',
          beamSteering: false
        }
      },
      motorwayMode: {
        if: 'mobilityType === "motorway"',
        then: {
          longCellCoverage: true,
          overlappingCells: true,
          fastHandover: true,
          highGainAntenna: true,
          speedBasedOptimization: true
        },
        else: {
          longCellCoverage: false,
          overlappingCells: false,
          fastHandover: false
        }
      },
      emergencyMode: {
        if: 'servicePriority === "emergency"',
        then: {
          handoverPriority: 'highest',
          resourceReservation: true,
          redundancyLevel: 'maximum',
          reliabilityMode: 'ultra_reliable',
          latencyOptimization: true
        },
        else: {
          handoverPriority: 'normal',
          resourceReservation: false
        }
      },
      highMobilityPeriod: {
        if: 'timeOfDay >= 7 && timeOfDay <= 9 || timeOfDay >= 17 && timeOfDay <= 19',
        then: {
          mobilityOptimization: 'enhanced',
          resourceAllocation: 'mobility_focused',
          handoverPreparation: 'aggressive'
        },
        else: {
          mobilityOptimization: 'standard',
          resourceAllocation: 'balanced'
        }
      }
    };
  }

  /**
   * Get mobility-specific evaluations
   */
  private getMobilityEvaluations() {
    return {
      handoverParameterCalculation: {
        eval: 'calculateHandoverParameters(currentSpeed, cellSize, interferenceLevel)',
        args: []
      },
      predictiveHandover: {
        eval: 'predictiveHandoverDecision(userTrajectory, cellCoverageMap, currentSignalStrength)',
        args: []
      },
      transportOptimization: {
        eval: 'optimizeForTransportType(transportType, speedCharacteristics)',
        args: []
      },
      dopplerCompensation: {
        eval: 'calculateDopplerShift(carrierFrequency, userSpeed, angleOfArrival)',
        args: []
      },
      mobilityStateDetection: {
        eval: 'mobilityStateEstimation(handoverLog, speedLog, locationLog)',
        args: []
      },
      redundancyCalculation: {
        eval: 'calculateRedundancyLevel(serviceType, mobilityLevel, reliabilityTarget)',
        args: []
      }
    };
  }

  /**
   * Generate mobility-specific variant with context awareness
   */
  generateMobilityVariant(
    baseTemplateName: string,
    context: MobilityDeploymentContext,
    options: any = {}
  ) {
    const mobilityOptions = {
      ...options,
      targetEnvironment: 'high_mobility',
      customOverrides: {
        'EUtranCellFDD.mobilityContext': context,
        'ENodeBFunction.mobilityOptimization': true,
        'SystemConfiguration.mobilityMode': 'high_speed',
        ...options.customOverrides
      }
    };

    return this.generateVariant('mobility', baseTemplateName, mobilityOptions);
  }

  /**
   * Generate variants for different mobility scenarios
   */
  generateMobilityScenarioVariants(baseTemplateName: string): Record<string, any> {
    const scenarios: Record<string, MobilityDeploymentContext> = {
      high_speed_train: {
        mobilityType: 'high_speed_train' as const,
        speedRange: { min: 200, max: 350 },
        handoverFrequency: 'very_high' as const,
        cellSize: 'large' as const,
        trafficPattern: 'continuous' as const,
        servicePriority: 'data' as const,
        redundancyLevel: 'maximum' as const
      },
      motorway: {
        mobilityType: 'motorway' as const,
        speedRange: { min: 80, max: 150 },
        handoverFrequency: 'high' as const,
        cellSize: 'large' as const,
        trafficPattern: 'bursty' as const,
        servicePriority: 'mixed' as const,
        redundancyLevel: 'high' as const
      },
      airport: {
        mobilityType: 'airport' as const,
        speedRange: { min: 0, max: 50 },
        handoverFrequency: 'medium' as const,
        cellSize: 'small' as const,
        trafficPattern: 'scheduled' as const,
        servicePriority: 'data' as const,
        redundancyLevel: 'high' as const
      },
      mixed_transport: {
        mobilityType: 'mixed_transport' as const,
        speedRange: { min: 0, max: 200 },
        handoverFrequency: 'medium' as const,
        cellSize: 'medium' as const,
        trafficPattern: 'continuous' as const,
        servicePriority: 'mixed' as const,
        redundancyLevel: 'standard' as const
      }
    };

    const variants: Record<string, any> = {};
    Object.entries(scenarios).forEach(([scenarioName, context]) => {
      variants[scenarioName] = this.generateMobilityVariant(baseTemplateName, context);
    });

    return variants;
  }

  /**
   * Get mobility deployment recommendations
   */
  getMobilityRecommendations(context: MobilityDeploymentContext): string[] {
    const recommendations: string[] = [];

    // Speed-based recommendations
    if (context.speedRange.max > 300) {
      recommendations.push('Enable predictive handover with 64ms TTT and 1dB hysteresis');
      recommendations.push('Implement directional antenna configuration for train alignment');
      recommendations.push('Enable advanced Doppler compensation up to ±1500 Hz');
    } else if (context.speedRange.max > 120) {
      recommendations.push('Configure fast handover with adaptive hysteresis (1-3dB)');
      recommendations.push('Enable mobility state estimation and speed-based optimization');
    }

    // Mobility type specific recommendations
    switch (context.mobilityType) {
      case 'high_speed_train':
        recommendations.push('Deploy linear cell alignment along railway tracks');
        recommendations.push('Implement cell blasting for extended coverage');
        recommendations.push('Enable dual connectivity with inter-band carrier aggregation');
        break;
      case 'motorway':
        recommendations.push('Configure long cell coverage with overlapping sectors');
        recommendations.push('Enable high-gain antennas with mechanical tilt');
        recommendations.push('Implement ping-pong prevention algorithms');
        break;
      case 'airport':
        recommendations.push('Focus on capacity optimization with small cells');
        recommendations.push('Enable scheduled resource allocation for flight patterns');
        recommendations.push('Implement enhanced QoS for premium airline services');
        break;
    }

    // Redundancy recommendations
    if (context.redundancyLevel === 'maximum') {
      recommendations.push('Enable maximum redundancy with dual connectivity');
      recommendations.push('Configure automatic fallback mechanisms');
      recommendations.push('Implement ultra-reliable low latency communication');
    }

    // General mobility recommendations
    recommendations.push('Enable mobility robustness optimization');
    recommendations.push('Implement predictive scheduling based on trajectory');
    recommendations.push('Configure adaptive QoS profiles for mobile users');
    recommendations.push('Enable enhanced cell reselection for fast camp-on');

    return recommendations;
  }
}