/**
 * RTB Hierarchical Template System - Usage Example
 *
 * Demonstrates how to use the complete priority template system
 * for RTB configuration management with inheritance, validation,
 * and performance optimization.
 */

import {
  IntegratedTemplateSystem,
  MOTemplateContext
} from '../../src/rtb/hierarchical-template-system/integrated-template-system';

import {
  TemplatePriority,
  TemplatePriorityInfo
} from '../../src/rtb/hierarchical-template-system/priority-engine';

import {
  RTBTemplate,
  TemplateMeta
} from '../../src/types/rtb-types';

/**
 * Example: RTB Template Hierarchy for Mobile Network Optimization
 *
 * This example demonstrates a typical RTB template hierarchy used for
 * RAN (Radio Access Network) configuration with priority-based inheritance.
 */
async function rttTemplateExample() {
  console.log('🚀 RTB Hierarchical Template System Example');
  console.log('==========================================');

  // Initialize the integrated template system
  const templateSystem = new IntegratedTemplateSystem({
    enablePrioritySystem: true,
    enableValidation: true,
    enablePerformanceOptimization: true,
    enableSchemaValidation: true,
    defaultPriority: TemplatePriority.BASE,
    maxInheritanceDepth: 10
  });

  // Define base template with common RAN parameters
  const baseTemplate: RTBTemplate = {
    meta: {
      version: '2.1.0',
      author: ['RAN Team', 'Ericsson'],
      description: 'Base RAN configuration template with common parameters',
      tags: ['base', 'ran', '5G', '4G'],
      environment: 'production',
      priority: TemplatePriority.BASE
    },
    custom: [
      {
        name: 'calculateCellCapacity',
        args: ['bandwidth', 'users', 'technology'],
        body: [
          'const capacityFactors = {',
          '  "5G": { "base": 1000, "multiplier": 2.5 },',
          '  "4G": { "base": 500, "multiplier": 1.8 }',
          '};',
          'const factor = capacityFactors[technology] || capacityFactors["4G"];',
          'return Math.floor(factor.base * (bandwidth / 20) * factor.multiplier * (1 + users / 100));'
        ]
      },
      {
        name: 'optimizePowerConsumption',
        args: ['trafficLoad', 'timeOfDay'],
        body: [
          'const basePower = 100;',
          'const trafficFactor = Math.max(0.3, 1 - trafficLoad / 100);',
          'const timeFactor = timeOfDay >= 22 || timeOfDay <= 6 ? 0.7 : 1.0;',
          'return Math.floor(basePower * trafficFactor * timeFactor);'
        ]
      }
    ],
    configuration: {
      // Network basic configuration
      networkOperator: 'OperatorName',
      networkName: 'RAN-Network-5G',
      plmn: {
        mcc: '208',
        mnc: '01'
      },

      // Cell configuration
      cellConfiguration: {
        earfcnDl: 3650,
        earfcnUl: 19650,
        pci: 1,
        tac: 1,
        bandwidthDl: 20,
        bandwidthUl: 20
      },

      // Power configuration
      powerConfiguration: {
        maxTxPower: 46,
        minTxPower: 30,
        referenceSignalPower: 15
      },

      // Advanced features
      features: {
        carrierAggregation: true,
        massiveMIMO: true,
        beamforming: true,
        tdd: false
      },

      // Quality of Service
      qos: {
        defaultQci: 9,
        priorityLevel: 1,
        preemptionCapability: true,
        preemptionVulnerability: false
      }
    },
    conditions: {
      enableAdvancedFeatures: {
        if: 'features.massiveMIMO == true && features.beamforming == true',
        then: {
          advancedParameters: {
            antennaPorts: 64,
            beamManagement: true,
            beamSweeping: true
          }
        },
        else: {
          advancedParameters: {
            antennaPorts: 4,
            beamManagement: false,
            beamSweeping: false
          }
        }
      }
    },
    evaluations: {
      cellCapacity: {
        eval: 'calculateCellCapacity(cellConfiguration.bandwidthDl, 1000, "5G")',
        args: []
      },
      optimizedPower: {
        eval: 'optimizePowerConsumption(75, 14)',
        args: []
      }
    }
  };

  // Define urban area template (inherits from base)
  const urbanTemplate: RTBTemplate = {
    meta: {
      version: '2.1.0',
      author: ['RAN Team'],
      description: 'Urban area RAN configuration with high capacity optimization',
      tags: ['urban', 'high-density', 'capacity'],
      environment: 'production',
      priority: TemplatePriority.SCENARIO,
      inherits_from: 'base'
    },
    custom: [
      {
        name: 'calculateInterference',
        args: ['cellDensity', 'buildingDensity'],
        body: [
          'const baseInterference = 10;',
          'const densityFactor = (cellDensity + buildingDensity) / 200;',
          'return Math.floor(baseInterference * (1 + densityFactor));'
        }
      }
    ],
    configuration: {
      // Override base configuration for urban areas
      cellConfiguration: {
        earfcnDl: 3800,
        pci: 15,
        tac: 101,
        bandwidthDl: 20,
        bandwidthUl: 20,
        // Urban-specific additions
        cellRadius: 500,
        antennaHeight: 30,
        antennaTilt: 8
      },

      // Enhanced power for urban coverage
      powerConfiguration: {
        maxTxPower: 46,
        minTxPower: 35,
        referenceSignalPower: 18,
        // Urban-specific
        downtilt: 8,
        azimith: 0
      },

      // Urban-specific features
      urbanFeatures: {
        smallCellDeployment: true,
        densification: 'high',
        hotspotOptimization: true,
        indoorCoverage: true
      },

      // Enhanced QoS for urban areas
      qos: {
        defaultQci: 8,
        priorityLevel: 1,
        guaranteedBitrate: {
          downlink: 50000,
          uplink: 10000
        }
      }
    },
    conditions: {
      enableDensification: {
        if: 'urbanFeatures.smallCellDeployment == true && urbanFeatures.densification == "high"',
        then: {
          densificationConfig: {
            smallCellCount: 50,
            interSiteDistance: 200,
          }
        }
      }
    },
    evaluations: {
      urbanInterference: {
        eval: 'calculateInterference(85, 90)',
        args: []
      }
    }
  };

  // Define mobility template (inherits from base)
  const mobilityTemplate: RTBTemplate = {
    meta: {
      version: '2.1.0',
      author: ['RAN Team'],
      description: 'High mobility area RAN configuration with handover optimization',
      tags: ['mobility', 'handover', 'highway'],
      environment: 'production',
      priority: TemplatePriority.SCENARIO,
      inherits_from: 'base'
    },
    configuration: {
      // Mobility-optimized configuration
      cellConfiguration: {
        earfcnDl: 3650,
        pci: 25,
        tac: 201,
        bandwidthDl: 15,
        bandwidthUl: 15,
        // Mobility-specific
        cellRadius: 2000,
        antennaHeight: 45,
        antennaTilt: 3
      },

      // Handover optimization
      handoverConfiguration: {
        a3Offset: 3,
        hysteresis: 3,
        timeToTrigger: 320,
        // Mobility-specific
        fastHandover: true,
        predictiveHandover: true,
        makeBeforeBreak: true
      },

      // Mobility features
      mobilityFeatures: {
        highSpeedHandover: true,
        predictiveAlgorithms: true,
        loadBalancing: true,
        carrierAggregation: true
      }
    },
    conditions: {
      enableHighSpeedOptimization: {
        if: 'mobilityFeatures.fastHandover == true && mobilityFeatures.predictiveHandover == true',
        then: {
          highSpeedConfig: {
            maxSupportedSpeed: 350,
            DopplerCompensation: true,
            channelEstimationEnhancement: true
          }
        }
      }
    }
  };

  // Define specific site template (inherits from urban and mobility)
  const highwaySiteTemplate: RTBTemplate = {
    meta: {
      version: '2.1.0',
      author: ['RAN Team'],
      description: 'Highway site configuration combining urban and mobility optimizations',
      tags: ['highway', 'mobility', 'urban-mobility'],
      environment: 'production',
      priority: TemplatePriority.CONTEXT_SPECIFIC,
      inherits_from: ['urban', 'mobility']
    },
    configuration: {
      // Site-specific overrides
      cellConfiguration: {
        earfcnDl: 3700,
        pci: 127,
        tac: 301,
        // Highway-specific
        cellRadius: 1500,
        antennaHeight: 40,
        antennaTilt: 5,
        highwayOrientation: true
      },

      // Highway-specific features
      highwayFeatures: {
        vehicleDetection: true,
        trafficPatternOptimization: true,
        emergencyServicePriority: true,
        weatherAdaptation: true
      },

      // Enhanced monitoring
      monitoringConfiguration: {
        realtimePerformance: true,
        anomalyDetection: true,
        automaticOptimization: true,
        alertThresholds: {
          callDropRate: 0.01,
          handoverFailureRate: 0.02,
          throughputThreshold: 100
        }
      }
    },
    conditions: {
      enableWeatherAdaptation: {
        if: 'highwayFeatures.weatherAdaptation == true',
        then: {
          weatherConfig: {
            rainMitigation: true,
            snowMitigation: true,
            fogCompensation: true,
            windAdjustment: true
          }
        }
      }
    }
  };

  // AgentDB learned template (highest priority)
  const agentdbLearnedTemplate: RTBTemplate = {
    meta: {
      version: '2.1.1',
      author: ['AgentDB ML System'],
      description: 'AgentDB learned optimizations based on network performance data',
      tags: ['agentdb', 'ml', 'learned', 'optimized'],
      environment: 'production',
      priority: TemplatePriority.AGENTDB_LEARNED,
      inherits_from: 'highway_site'
    },
    configuration: {
      // AgentDB-learned parameter optimizations
      learnedOptimizations: {
        powerEfficiency: 0.85,
        capacityOptimization: 1.15,
        interferenceMitigation: 0.92,
        handoverSuccessRate: 0.98
      },

      // Fine-tuned parameters based on ML analysis
      cellConfiguration: {
        referenceSignalPower: 17.5, // ML-optimized value
        antennaTilt: 4.8,           // ML-optimized value
        pdcchConfig: {
          agLevel: 3,               // ML-optimized
          cceAggregationLevel: 4    // ML-optimized
        }
      },

      // Predictive configurations
      predictiveConfig: {
        trafficPrediction: true,
        loadBalancingPrediction: true,
        preventiveOptimization: true
      }
    },
    evaluations: {
      mlOptimizedCapacity: {
        eval: 'cellCapacity * learnedOptimizations.capacityOptimization',
        args: []
      },
      mlOptimizedPower: {
        eval: 'optimizedPower * learnedOptimizations.powerEfficiency',
        args: []
      }
    }
  };

  try {
    console.log('\n📝 Registering templates in hierarchy...');

    // Register templates with their priorities
    await templateSystem.registerTemplate(
      'base',
      baseTemplate,
      createTestPriority('base', TemplatePriority.BASE)
    );

    await templateSystem.registerTemplate(
      'urban',
      urbanTemplate,
      createTestPriority('urban', TemplatePriority.SCENARIO, 'base')
    );

    await templateSystem.registerTemplate(
      'mobility',
      mobilityTemplate,
      createTestPriority('mobility', TemplatePriority.SCENARIO, 'base')
    );

    await templateSystem.registerTemplate(
      'highway_site',
      highwaySiteTemplate,
      createTestPriority('highway_site', TemplatePriority.CONTEXT_SPECIFIC, ['urban', 'mobility'])
    );

    await templateSystem.registerTemplate(
      'agentdb_learned',
      agentdbLearnedTemplate,
      createTestPriority('agentdb_learned', TemplatePriority.AGENTDB_LEARNED, 'highway_site')
    );

    console.log('✅ Templates registered successfully');

    // Process the highest priority template (AgentDB learned)
    console.log('\n🔄 Processing template with inheritance resolution...');

    const context: MOTemplateContext = {
      environment: 'production',
      featureFlags: {
        advanced_ml: true,
        real_time_optimization: true,
        predictive_maintenance: true
      }
    };

    const result = await templateSystem.processTemplate('agentdb_learned', context);

    console.log('\n📊 Processing Results:');
    console.log('=====================');
    console.log(`✨ Template processed in ${result.processingStats.validationTime}ms`);
    console.log(`📋 Total parameters: ${result.processingStats.totalParameters}`);
    console.log(`🔗 Inheritance depth: ${result.inheritanceChain.chain.length}`);
    console.log(`⚡ Applied optimizations: ${result.appliedOptimizations.join(', ')}`);

    if (result.warnings.length > 0) {
      console.log(`⚠️  Warnings: ${result.warnings.length}`);
      result.warnings.forEach(warning => console.log(`   - ${warning}`));
    }

    if (result.errors.length > 0) {
      console.log(`❌ Errors: ${result.errors.length}`);
      result.errors.forEach(error => console.log(`   - ${error}`));
    }

    // Display resolved configuration
    console.log('\n🎯 Resolved Configuration:');
    console.log('==========================');
    const config = result.template.configuration;

    console.log('📡 Network Configuration:');
    console.log(`   Operator: ${config.networkOperator}`);
    console.log(`   PLMN: ${config.plmn.mcc}-${config.plmn.mnc}`);
    console.log(`   Cell ID: ${config.cellConfiguration?.pci}`);
    console.log(`   Bandwidth: ${config.cellConfiguration?.bandwidthDl} MHz DL / ${config.cellConfiguration?.bandwidthUl} MHz UL`);

    if (config.learnedOptimizations) {
      console.log('\n🧠 AgentDB Learned Optimizations:');
      console.log(`   Power Efficiency: ${(config.learnedOptimizations.powerEfficiency * 100).toFixed(1)}%`);
      console.log(`   Capacity Boost: ${((config.learnedOptimizations.capacityOptimization - 1) * 100).toFixed(1)}%`);
      console.log(`   Handover Success Rate: ${(config.learnedOptimizations.handoverSuccessRate * 100).toFixed(1)}%`);
    }

    if (result.template.evaluations) {
      console.log('\n📈 Computed Values:');
      console.log('==================');
      if (result.template.evaluations.mlOptimizedCapacity) {
        console.log(`   ML Optimized Capacity: ${result.template.evaluations.mlOptimizedCapacity.eval} (computed)`);
      }
      if (result.template.evaluations.mlOptimizedPower) {
        console.log(`   ML Optimized Power: ${result.template.evaluations.mlOptimizedPower.eval} (computed)`);
      }
    }

    // Display inheritance chain
    console.log('\n🔗 Inheritance Chain:');
    console.log('=====================');
    result.inheritanceChain.chain.forEach((priority, index) => {
      const indent = '  '.repeat(index);
      console.log(`${indent}📁 ${priority.category} (Priority: ${priority.level}) - ${priority.source}`);
    });

    // Display parameter conflicts and resolutions
    if (result.inheritanceChain.conflicts.length > 0) {
      console.log('\n⚔️  Parameter Conflicts & Resolutions:');
      console.log('===================================');
      result.inheritanceChain.conflicts.forEach((conflict, index) => {
        console.log(`${index + 1}. Parameter: ${conflict.parameter}`);
        console.log(`   Sources: ${conflict.templates.join(' → ')}`);
        console.log(`   Values: ${conflict.values.map(v => JSON.stringify(v)).join(' vs ')}`);
        console.log(`   Resolved: ${JSON.stringify(conflict.resolvedValue)} (${conflict.resolutionStrategy})`);
        console.log(`   Reason: ${conflict.reason}\n`);
      });
    }

    // Batch processing example
    console.log('\n🔄 Batch Processing Example:');
    console.log('============================');

    const batchResults = await templateSystem.batchProcessTemplates(
      ['base', 'urban', 'highway_site', 'agentdb_learned'],
      context
    );

    batchResults.forEach((result, index) => {
      console.log(`${index + 1}. ${result.inheritanceChain.templateName}:`);
      console.log(`   Parameters: ${result.processingStats.totalParameters}`);
      console.log(`   Processing time: ${result.processingStats.validationTime}ms`);
      console.log(`   Optimizations: ${result.appliedOptimizations.join(', ')}`);
    });

    // Search templates example
    console.log('\n🔍 Template Search Example:');
    console.log('==========================');

    const searchResults = await templateSystem.searchTemplates({
      tags: ['mobility', 'learned'],
      priorityRange: { min: 0, max: 50 }
    });

    console.log(`Found ${searchResults.totalCount} templates matching criteria:`);
    searchResults.templates.forEach((template: any) => {
      console.log(`   - ${template.name} (${template.priority.category})`);
    });

    // System statistics
    console.log('\n📊 System Statistics:');
    console.log('====================');

    const stats = templateSystem.getSystemStats();
    console.log(`📋 Registry: ${stats.registry.totalTemplates} templates`);
    console.log(`🎯 Priority: ${stats.priority.templates} templates, cache hit rate: ${(stats.priority.cacheStats.hitRate * 100).toFixed(1)}%`);
    console.log(`⚡ Performance: ${stats.performance.summary.totalOperations} operations, avg duration: ${stats.performance.summary.averageDuration.toFixed(2)}ms`);
    console.log(`✅ Validation: ${stats.validation.totalRules} rules, ${stats.validation.enabledRules} enabled`);

    console.log('\n🎉 RTB Hierarchical Template System Demo Complete!');
    console.log('==================================================');
    console.log('The system successfully demonstrated:');
    console.log('✅ Priority-based template inheritance');
    console.log('✅ Parameter conflict resolution');
    console.log('✅ Custom function evaluation');
    console.log('✅ Conditional logic processing');
    console.log('✅ Comprehensive validation');
    console.log('✅ Performance optimization');
    console.log('✅ Batch processing capabilities');
    console.log('✅ Template search and filtering');
    console.log('✅ Integration with AgentDB learned patterns');

  } catch (error) {
    console.error('❌ Error in template processing:', error);
  }
}

/**
 * Helper function to create test priority info
 */
function createTestPriority(
  category: string,
  level: number,
  inheritsFrom?: string | string[]
): TemplatePriorityInfo {
  return {
    level,
    category,
    source: 'example',
    inherits_from: inheritsFrom
  };
}

// Export for use in other examples or tests
export {
  rttTemplateExample,
  createTestTemplate
};

/**
 * Create a test template with basic structure
 */
function createTestTemplate(name: string, config: Record<string, any> = {}): RTBTemplate {
  return {
    meta: {
      version: '1.0.0',
      author: ['Example'],
      description: `Test template ${name}`,
      tags: ['test'],
      priority: TemplatePriority.BASE
    },
    custom: [],
    configuration: config,
    conditions: {},
    evaluations: {}
  };
}

// Run the example if this file is executed directly
if (require.main === module) {
  rttTemplateExample().catch(console.error);
}