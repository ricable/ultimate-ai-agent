/**
 * Template Variant Generator Tests
 *
 * Comprehensive test suite for the RTB template variant generation system.
 * Tests all variant generators, edge cases, and integration scenarios.
 */

import { VariantGeneratorCore } from '../../../src/rtb/hierarchical-template-system/variant-generator-core';
import { UrbanVariantGenerator, UrbanDeploymentContext } from '../../../src/rtb/hierarchical-template-system/variant-generators/urban-variant';
import { MobilityVariantGenerator, MobilityDeploymentContext } from '../../../src/rtb/hierarchical-template-system/variant-generators/mobility-variant';
import { SleepVariantGenerator, SleepModeContext } from '../../../src/rtb/hierarchical-template-system/variant-generators/sleep-variant';
import { VariantGeneratorManager, CombinedDeploymentContext } from '../../../src/rtb/hierarchical-template-system/variant-generator-manager';
import { RTBTemplate } from '../../../src/types/rtb-types';

describe('Template Variant Generator System', () => {
  let coreGenerator: VariantGeneratorCore;
  let urbanGenerator: UrbanVariantGenerator;
  let mobilityGenerator: MobilityVariantGenerator;
  let sleepGenerator: SleepVariantGenerator;
  let manager: VariantGeneratorManager;

  beforeEach(() => {
    coreGenerator = new VariantGeneratorCore();
    urbanGenerator = new UrbanVariantGenerator();
    mobilityGenerator = new MobilityVariantGenerator();
    sleepGenerator = new SleepVariantGenerator();
    manager = new VariantGeneratorManager();
  });

  describe('VariantGeneratorCore', () => {
    const baseTemplate: RTBTemplate = {
      meta: {
        version: '1.0.0',
        author: ['Test'],
        description: 'Base template for testing',
        priority: 9
      },
      configuration: {
        'EUtranCellFDD.basicParam': 'defaultValue',
        'ENodeBFunction.setting': true
      }
    };

    beforeEach(() => {
      coreGenerator.registerBaseTemplate('test_base', baseTemplate);
    });

    describe('Base Template Registration', () => {
      it('should register base template successfully', () => {
        const retrieved = coreGenerator['baseTemplates'].get('test_base');
        expect(retrieved).toBeDefined();
        expect(retrieved?.meta?.description).toBe('Base template for testing');
      });

      it('should throw error when base template not found', async () => {
        await expect(
          coreGenerator.generateVariant('urban', 'non_existent_template')
        ).rejects.toThrow('Base template \'non_existent_template\' not found');
      });
    });

    describe('Variant Generation', () => {
      it('should generate variant with proper metadata', () => {
        const variant = coreGenerator.generateVariant('urban', 'test_base');

        expect(variant.meta).toBeDefined();
        expect(variant.meta?.priority).toBe(20);
        expect(variant.meta?.tags).toContain('variant-urban');
        expect(variant.meta?.tags).toContain('priority-20');
      });

      it('should apply custom overrides', () => {
        const options = {
          customOverrides: {
            'EUtranCellFDD.customParam': 'customValue',
            'ENodeBFunction.overrideSetting': false
          }
        };

        const variant = coreGenerator.generateVariant('urban', 'test_base', options);

        expect(variant.configuration['EUtranCellFDD.customParam']).toBe('customValue');
        expect(variant.configuration['ENodeBFunction.overrideSetting']).toBe(false);
      });

      it('should merge custom functions correctly', () => {
        const variant = coreGenerator.generateVariant('urban', 'test_base');

        expect(variant.custom).toBeDefined();
        expect(variant.custom!.length).toBeGreaterThan(0);
        expect(variant.custom!.some(func => func.name === 'calculateUrbanCapacity')).toBe(true);
      });
    });

    describe('Template Validation', () => {
      it('should validate correct template', () => {
        const validTemplate: RTBTemplate = {
          meta: {
            version: '1.0.0',
            author: ['Test'],
            description: 'Valid template',
            priority: 10
          },
          configuration: { param1: 'value1' },
          custom: [{ name: 'testFunc', args: [], body: ['return true;'] }],
          conditions: { testCondition: { if: 'true', then: { result: true }, else: false } },
          evaluations: { testEval: { eval: 'testFunction()' } }
        };

        const validation = coreGenerator.validateTemplate(validTemplate);
        expect(validation.valid).toBe(true);
        expect(validation.errors).toHaveLength(0);
      });

      it('should identify template errors', () => {
        const invalidTemplate: RTBTemplate = {
          configuration: {},
          custom: [{ name: '', args: [], body: [] }] // Invalid function
        };

        const validation = coreGenerator.validateTemplate(invalidTemplate);
        expect(validation.valid).toBe(false);
        expect(validation.errors.length).toBeGreaterThan(0);
      });
    });

    describe('Optimization History', () => {
      it('should track optimization statistics', () => {
        coreGenerator.generateVariant('urban', 'test_base');
        coreGenerator.generateVariant('mobility', 'test_base');

        const stats = coreGenerator.getOptimizationStats();
        expect(stats).toHaveProperty('urban');
        expect(stats).toHaveProperty('mobility');
      });
    });
  });

  describe('UrbanVariantGenerator', () => {
    const urbanContext: UrbanDeploymentContext = {
      populationDensity: 'high',
      buildingType: 'high_rise',
      trafficPattern: 'business_district',
      stadiumEvents: false,
      transportHubs: true,
      capacityBoost: 'high'
    };

    beforeEach(() => {
      const baseTemplate: RTBTemplate = {
        meta: { version: '1.0.0', author: ['Test'], description: 'Base' },
        configuration: {}
      };
      urbanGenerator.registerBaseTemplate('urban_base', baseTemplate);
    });

    it('should generate urban variant with massive MIMO', () => {
      const variant = urbanGenerator.generateUrbanVariant('urban_base', urbanContext);

      expect(variant.configuration['EUtranCellFDD.mimoConfiguration']).toBeDefined();
      expect(variant.configuration['EUtranCellFDD.antennaConfiguration']).toBeDefined();
    });

    it('should include carrier aggregation configuration', () => {
      const variant = urbanGenerator.generateUrbanVariant('urban_base', urbanContext);

      const caConfig = variant.configuration['EUtranCellFDD.carrierAggregation'];
      expect(caConfig).toBeDefined();
      expect(caConfig.enabled).toBe(true);
      expect(caConfig.maxCCs).toBe(4);
    });

    it('should enable stadium mode when stadiumEvents is true', () => {
      const stadiumContext = { ...urbanContext, stadiumEvents: true };
      const variant = urbanGenerator.generateUrbanVariant('urban_base', stadiumContext);

      expect(variant.configuration['EUtranCellFDD.stadiumMode'].enabled).toBe(true);
    });

    it('should generate urban-specific custom functions', () => {
      const variant = urbanGenerator.generateUrbanVariant('urban_base', urbanContext);

      expect(variant.custom).toBeDefined();
      const customFuncNames = variant.custom!.map(func => func.name);
      expect(customFuncNames).toContain('calculateUrbanCapacity');
      expect(customFuncNames).toContain('optimizeBeamConfiguration');
      expect(customFuncNames).toContain('calculateOptimalCA');
    });

    it('should provide deployment recommendations', () => {
      const recommendations = urbanGenerator.getUrbanRecommendations(urbanContext);

      expect(recommendations).toBeInstanceOf(Array);
      expect(recommendations.length).toBeGreaterThan(0);
      expect(recommendations.some(rec => rec.includes('massive MIMO'))).toBe(true);
    });

    it('should generate scenario variants', () => {
      const scenarios = urbanGenerator.generateUrbanScenarioVariants('urban_base');

      expect(scenarios).toHaveProperty('financial_district');
      expect(scenarios).toHaveProperty('residential_high_rise');
      expect(scenarios).toHaveProperty('mixed_use_center');
      expect(scenarios).toHaveProperty('stadium_complex');

      const financialDistrict = scenarios.financial_district;
      expect(financialDistrict.configuration['EUtranCellFDD.urbanContext'].populationDensity).toBe('ultra_high');
    });
  });

  describe('MobilityVariantGenerator', () => {
    const mobilityContext: MobilityDeploymentContext = {
      mobilityType: 'high_speed_train',
      speedRange: { min: 200, max: 350 },
      handoverFrequency: 'very_high',
      cellSize: 'large',
      trafficPattern: 'continuous',
      servicePriority: 'data',
      redundancyLevel: 'maximum'
    };

    beforeEach(() => {
      const baseTemplate: RTBTemplate = {
        meta: { version: '1.0.0', author: ['Test'], description: 'Base' },
        configuration: {}
      };
      mobilityGenerator.registerBaseTemplate('mobility_base', baseTemplate);
    });

    it('should generate mobility variant with handover optimization', () => {
      const variant = mobilityGenerator.generateMobilityVariant('mobility_base', mobilityContext);

      const handoverConfig = variant.configuration['EUtranCellFDD.handoverParameters'];
      expect(handoverConfig).toBeDefined();
      expect(handoverConfig.makeBeforeBreak).toBe(true);
      expect(handoverConfig.handoverMode).toBe('predictive');
    });

    it('should include train mode for high-speed trains', () => {
      const variant = mobilityGenerator.generateMobilityVariant('mobility_base', mobilityContext);

      expect(variant.configuration['EUtranCellFDD.trainMode'].enabled).toBe(true);
      expect(variant.configuration['EUtranCellFDD.trainMode'].directionalAntenna).toBe(true);
    });

    it('should include speed-based adaptation', () => {
      const variant = mobilityGenerator.generateMobilityVariant('mobility_base', mobilityContext);

      const speedAdapt = variant.configuration['EUtranCellFDD.speedAdaptation'];
      expect(speedAdapt).toBeDefined();
      expect(speedAdapt.enabled).toBe(true);
      expect(speedAdapt.speedThresholds).toBeInstanceOf(Array);
    });

    it('should generate mobility-specific custom functions', () => {
      const variant = mobilityGenerator.generateMobilityVariant('mobility_base', mobilityContext);

      expect(variant.custom).toBeDefined();
      const customFuncNames = variant.custom!.map(func => func.name);
      expect(customFuncNames).toContain('calculateHandoverParameters');
      expect(customFuncNames).toContain('predictiveHandoverDecision');
      expect(customFuncNames).toContain('calculateDopplerShift');
    });

    it('should provide mobility recommendations', () => {
      const recommendations = mobilityGenerator.getMobilityRecommendations(mobilityContext);

      expect(recommendations).toBeInstanceOf(Array);
      expect(recommendations.length).toBeGreaterThan(0);
      expect(recommendations.some(rec => rec.includes('predictive handover'))).toBe(true);
    });

    it('should generate different motorway configuration', () => {
      const motorwayContext = { ...mobilityContext, mobilityType: 'motorway' as const };
      const variant = mobilityGenerator.generateMobilityVariant('mobility_base', motorwayContext);

      expect(variant.configuration['EUtranCellFDD.motorwayMode'].enabled).toBe(true);
      expect(variant.configuration['EUtranCellFDD.motorwayMode'].longCellCoverage).toBe(true);
    });
  });

  describe('SleepVariantGenerator', () => {
    const sleepContext: SleepModeContext = {
      sleepModeType: 'night_time',
      energySavingLevel: 'maximum',
      wakeUpTriggers: ['emergency_call', 'high_traffic', 'time_based'],
      minimumCapacityGuarantee: 20,
      trafficProfile: 'residential',
      environmentalConsiderations: {
        temperatureRange: { min: 15, max: 25 },
        powerSource: 'grid',
        backupPower: true
      },
      serviceObligations: {
        emergencyServices: true,
        criticalInfrastructure: false,
        premiumUsers: false
      }
    };

    beforeEach(() => {
      const baseTemplate: RTBTemplate = {
        meta: { version: '1.0.0', author: ['Test'], description: 'Base' },
        configuration: {}
      };
      sleepGenerator.registerBaseTemplate('sleep_base', baseTemplate);
    });

    it('should generate sleep variant with energy saving configuration', () => {
      const variant = sleepGenerator.generateSleepVariant('sleep_base', sleepContext);

      const energyConfig = variant.configuration['EUtranCellFDD.energySaving'];
      expect(energyConfig).toBeDefined();
      expect(energyConfig.enabled).toBe(true);
      expect(energyConfig.mode).toBe('adaptive');
      expect(energyConfig.sleepMode).toBe('deep');
    });

    it('should include MIMO sleep mode configuration', () => {
      const variant = sleepGenerator.generateSleepVariant('sleep_base', sleepContext);

      const mimoSleep = variant.configuration['EUtranCellFDD.mimoSleepMode'];
      expect(mimoSleep).toBeDefined();
      expect(mimoSleep.enabled).toBe(true);
      expect(mimoSleep.sleepStrategy).toBe('layer_based');
      expect(mimoSleep.activeLayers).toBe(1);
    });

    it('should configure wake-up triggers', () => {
      const variant = sleepGenerator.generateSleepVariant('sleep_base', sleepContext);

      const wakeUpTriggers = variant.configuration['EUtranCellFDD.wakeUpTriggers'];
      expect(wakeUpTriggers).toBeDefined();
      expect(wakeUpTriggers.triggers).toBeInstanceOf(Array);
      expect(wakeUpTriggers.triggers.length).toBeGreaterThan(0);
      expect(wakeUpTriggers.triggers.some((t: any) => t.type === 'emergency_call')).toBe(true);
    });

    it('should generate sleep-specific custom functions', () => {
      const variant = sleepGenerator.generateSleepVariant('sleep_base', sleepContext);

      expect(variant.custom).toBeDefined();
      const customFuncNames = variant.custom!.map(func => func.name);
      expect(customFuncNames).toContain('calculateOptimalSleepLevel');
      expect(customFuncNames).toContain('predictWakeUpTime');
      expect(customFuncNames).toContain('calculateEnergySavings');
    });

    it('should provide sleep mode recommendations', () => {
      const recommendations = sleepGenerator.getSleepRecommendations(sleepContext);

      expect(recommendations).toBeInstanceOf(Array);
      expect(recommendations.length).toBeGreaterThan(0);
      expect(recommendations.some(rec => rec.includes('deep sleep'))).toBe(true);
    });

    it('should configure different sleep levels', () => {
      const moderateContext = { ...sleepContext, energySavingLevel: 'moderate' as const };
      const variant = sleepGenerator.generateSleepVariant('sleep_base', moderateContext);

      expect(variant.configuration['EUtranCellFDD.energySaving'].powerScaling).toBe(30); // Moderate level
    });
  });

  describe('VariantGeneratorManager', () => {
    const baseTemplate: RTBTemplate = {
      meta: { version: '1.0.0', author: ['Test'], description: 'Base' },
      configuration: {}
    };

    beforeEach(() => {
      urbanGenerator.registerBaseTemplate('urban_base', baseTemplate);
      mobilityGenerator.registerBaseTemplate('mobility_base', baseTemplate);
      sleepGenerator.registerBaseTemplate('sleep_base', baseTemplate);
    });

    it('should generate urban variant through manager', async () => {
      const urbanContext: UrbanDeploymentContext = {
        populationDensity: 'high',
        buildingType: 'mixed',
        trafficPattern: 'business_district',
        stadiumEvents: false,
        transportHubs: true,
        capacityBoost: 'high'
      };

      const combinedContext: CombinedDeploymentContext = {
        primaryScenario: 'urban',
        urbanContext,
        globalSettings: {
          cellCount: 100,
          trafficProfile: 'high',
          energyMode: 'performance',
          targetEnvironment: 'test',
          customOverrides: {}
        }
      };

      const result = await manager.generateVariant('urban_base', combinedContext);

      expect(result.template).toBeDefined();
      expect(result.metadata.variantType).toBe('urban');
      expect(result.validation.valid).toBe(true);
      expect(result.performance.estimatedCapacityImprovement).toBeGreaterThan(0);
    });

    it('should generate mobility variant through manager', async () => {
      const mobilityContext: MobilityDeploymentContext = {
        mobilityType: 'motorway',
        speedRange: { min: 80, max: 150 },
        handoverFrequency: 'high',
        cellSize: 'large',
        trafficPattern: 'bursty',
        servicePriority: 'mixed',
        redundancyLevel: 'high'
      };

      const combinedContext: CombinedDeploymentContext = {
        primaryScenario: 'mobility',
        mobilityContext,
        globalSettings: {
          cellCount: 50,
          trafficProfile: 'medium',
          energyMode: 'balanced',
          targetEnvironment: 'test',
          customOverrides: {}
        }
      };

      const result = await manager.generateVariant('mobility_base', combinedContext);

      expect(result.template).toBeDefined();
      expect(result.metadata.variantType).toBe('mobility');
      expect(result.validation.valid).toBe(true);
      expect(result.performance.estimatedLatencyImprovement).toBeGreaterThan(0);
    });

    it('should generate sleep variant through manager', async () => {
      const sleepContext: SleepModeContext = {
        sleepModeType: 'night_time',
        energySavingLevel: 'aggressive',
        wakeUpTriggers: ['emergency_call'],
        minimumCapacityGuarantee: 30,
        trafficProfile: 'residential',
        environmentalConsiderations: {
          temperatureRange: { min: 10, max: 30 },
          powerSource: 'grid',
          backupPower: true
        },
        serviceObligations: {
          emergencyServices: true,
          criticalInfrastructure: false,
          premiumUsers: false
        }
      };

      const combinedContext: CombinedDeploymentContext = {
        primaryScenario: 'sleep',
        sleepContext,
        globalSettings: {
          cellCount: 75,
          trafficProfile: 'low',
          energyMode: 'energy_saving',
          targetEnvironment: 'test',
          customOverrides: {}
        }
      };

      const result = await manager.generateVariant('sleep_base', combinedContext);

      expect(result.template).toBeDefined();
      expect(result.metadata.variantType).toBe('sleep');
      expect(result.validation.valid).toBe(true);
      expect(result.performance.estimatedEnergySavings).toBeGreaterThan(0);
    });

    it('should generate hybrid variant', async () => {
      const urbanContext: UrbanDeploymentContext = {
        populationDensity: 'high',
        buildingType: 'mixed',
        trafficPattern: 'mixed_use',
        stadiumEvents: false,
        transportHubs: true,
        capacityBoost: 'high'
      };

      const mobilityContext: MobilityDeploymentContext = {
        mobilityType: 'mixed_transport',
        speedRange: { min: 0, max: 100 },
        handoverFrequency: 'medium',
        cellSize: 'medium',
        trafficPattern: 'continuous',
        servicePriority: 'mixed',
        redundancyLevel: 'standard'
      };

      const combinedContext: CombinedDeploymentContext = {
        primaryScenario: 'hybrid',
        urbanContext,
        mobilityContext,
        hybridConfig: {
          scenarios: ['urban', 'mobility'],
          weights: { urban: 0.6, mobility: 0.4 },
          transitionRules: {
            'high_traffic': 'urban_priority',
            'peak_hours': 'mobility_priority'
          }
        },
        globalSettings: {
          cellCount: 150,
          trafficProfile: 'high',
          energyMode: 'balanced',
          targetEnvironment: 'test',
          customOverrides: {}
        }
      };

      const result = await manager.generateVariant('urban_base', combinedContext);

      expect(result.template).toBeDefined();
      expect(result.metadata.variantType).toBe('hybrid');
      expect(result.validation.valid).toBe(true);
      expect(result.template.meta?.tags).toContain('hybrid');
      expect(result.template.meta?.tags).toContain('urban');
      expect(result.template.meta?.tags).toContain('mobility');
    });

    it('should handle batch generation', async () => {
      const contexts: CombinedDeploymentContext[] = [
        {
          primaryScenario: 'urban',
          urbanContext: {
            populationDensity: 'medium',
            buildingType: 'mixed',
            trafficPattern: 'residential',
            stadiumEvents: false,
            transportHubs: false,
            capacityBoost: 'standard'
          },
          globalSettings: {
            cellCount: 50,
            trafficProfile: 'medium',
            energyMode: 'balanced',
            targetEnvironment: 'test',
            customOverrides: {}
          }
        },
        {
          primaryScenario: 'sleep',
          sleepContext: {
            sleepModeType: 'weekend',
            energySavingLevel: 'moderate',
            wakeUpTriggers: ['emergency_call'],
            minimumCapacityGuarantee: 40,
            trafficProfile: 'mixed',
            environmentalConsiderations: {
              temperatureRange: { min: 15, max: 25 },
              powerSource: 'grid',
              backupPower: true
            },
            serviceObligations: {
              emergencyServices: true,
              criticalInfrastructure: false,
              premiumUsers: false
            }
          },
          globalSettings: {
            cellCount: 60,
            trafficProfile: 'low',
            energyMode: 'energy_saving',
            targetEnvironment: 'test',
            customOverrides: {}
          }
        }
      ];

      const results = await manager.generateBatchVariants('urban_base', contexts);

      expect(Object.keys(results)).toHaveLength(2);
      expect(results['urban_0']).toBeDefined();
      expect(results['sleep_1']).toBeDefined();
      expect(results['urban_0'].validation.valid).toBe(true);
      expect(results['sleep_1'].validation.valid).toBe(true);
    });

    it('should track generation statistics', async () => {
      // Generate a few variants to populate statistics
      await manager.generateVariant('urban_base', {
        primaryScenario: 'urban',
        urbanContext: {
          populationDensity: 'high',
          buildingType: 'mixed',
          trafficPattern: 'business',
          stadiumEvents: false,
          transportHubs: true,
          capacityBoost: 'high'
        },
        globalSettings: {
          cellCount: 100,
          trafficProfile: 'high',
          energyMode: 'performance',
          targetEnvironment: 'test',
          customOverrides: {}
        }
      });

      await manager.generateVariant('sleep_base', {
        primaryScenario: 'sleep',
        sleepContext: {
          sleepModeType: 'night_time',
          energySavingLevel: 'maximum',
          wakeUpTriggers: ['emergency_call'],
          minimumCapacityGuarantee: 20,
          trafficProfile: 'residential',
          environmentalConsiderations: {
            temperatureRange: { min: 15, max: 25 },
            powerSource: 'grid',
            backupPower: true
          },
          serviceObligations: {
            emergencyServices: true,
            criticalInfrastructure: false,
            premiumUsers: false
          }
        },
        globalSettings: {
          cellCount: 80,
          trafficProfile: 'low',
          energyMode: 'energy_saving',
          targetEnvironment: 'test',
          customOverrides: {}
        }
      });

      const stats = manager.getGenerationStatistics();

      expect(stats.totalGenerations).toBe(2);
      expect(stats.successRate).toBe(100);
      expect(stats.variantTypeDistribution).toHaveProperty('urban', 1);
      expect(stats.variantTypeDistribution).toHaveProperty('sleep', 1);
      expect(stats.averageGenerationTime).toBeGreaterThan(0);
    });

    it('should provide recommendations based on context', () => {
      const recommendations = manager.getRecommendations({
        globalSettings: {
          cellCount: 200,
          trafficProfile: 'high',
          energyMode: 'performance'
        }
      });

      expect(recommendations).toBeInstanceOf(Array);
      expect(recommendations.length).toBeGreaterThan(0);
      expect(recommendations.some(rec => rec.includes('Urban'))).toBe(true);
    });

    it('should handle generation errors gracefully', async () => {
      const invalidContext: CombinedDeploymentContext = {
        primaryScenario: 'urban',
        // Missing urbanContext - should cause error
        globalSettings: {
          cellCount: 50,
          trafficProfile: 'medium',
          energyMode: 'balanced',
          targetEnvironment: 'test',
          customOverrides: {}
        }
      };

      await expect(
        manager.generateVariant('urban_base', invalidContext)
      ).rejects.toThrow('Urban context required for urban variant generation');
    });
  });

  describe('Integration Tests', () => {
    it('should handle complex end-to-end scenario', async () => {
      // Create a complex multi-scenario deployment
      const manager = new VariantGeneratorManager();
      const baseTemplate: RTBTemplate = {
        meta: { version: '1.0.0', author: ['Integration Test'], description: 'Complex scenario base' },
        configuration: {
          'SystemConfiguration.baseParam': 'baseValue'
        }
      };

      // Register base templates
      manager['urbanGenerator'].registerBaseTemplate('complex_base', baseTemplate);
      manager['mobilityGenerator'].registerBaseTemplate('complex_base', baseTemplate);
      manager['sleepGenerator'].registerBaseTemplate('complex_base', baseTemplate);

      // Generate multiple variants for different times of day
      const dayVariant = await manager.generateVariant('complex_base', {
        primaryScenario: 'urban',
        urbanContext: {
          populationDensity: 'ultra_high',
          buildingType: 'sky_scraper',
          trafficPattern: 'business_district',
          stadiumEvents: false,
          transportHubs: true,
          capacityBoost: 'maximum'
        },
        globalSettings: {
          cellCount: 300,
          trafficProfile: 'high',
          energyMode: 'performance',
          targetEnvironment: 'day_time_business',
          customOverrides: {}
        }
      });

      const nightVariant = await manager.generateVariant('complex_base', {
        primaryScenario: 'sleep',
        sleepContext: {
          sleepModeType: 'night_time',
          energySavingLevel: 'maximum',
          wakeUpTriggers: ['emergency_call'],
          minimumCapacityGuarantee: 15,
          trafficProfile: 'residential',
          environmentalConsiderations: {
            temperatureRange: { min: 12, max: 22 },
            powerSource: 'grid',
            backupPower: true
          },
          serviceObligations: {
            emergencyServices: true,
            criticalInfrastructure: false,
            premiumUsers: false
          }
        },
        globalSettings: {
          cellCount: 300,
          trafficProfile: 'low',
          energyMode: 'energy_saving',
          targetEnvironment: 'night_time_residential',
          customOverrides: {}
        }
      });

      const hybridVariant = await manager.generateVariant('complex_base', {
        primaryScenario: 'hybrid',
        urbanContext: {
          populationDensity: 'high',
          buildingType: 'mixed',
          trafficPattern: 'mixed_use',
          stadiumEvents: false,
          transportHubs: true,
          capacityBoost: 'high'
        },
        mobilityContext: {
          mobilityType: 'mixed_transport',
          speedRange: { min: 0, max: 80 },
          handoverFrequency: 'medium',
          cellSize: 'medium',
          trafficPattern: 'continuous',
          servicePriority: 'mixed',
          redundancyLevel: 'standard'
        },
        hybridConfig: {
          scenarios: ['urban', 'mobility'],
          weights: { urban: 0.7, mobility: 0.3 },
          transitionRules: {
            'business_hours': 'urban_priority',
            'peak_traffic': 'mobility_priority'
          }
        },
        globalSettings: {
          cellCount: 300,
          trafficProfile: 'medium',
          energyMode: 'balanced',
          targetEnvironment: 'transit_hub_area',
          customOverrides: {}
        }
      });

      // Verify all variants are valid and have expected characteristics
      expect(dayVariant.validation.valid).toBe(true);
      expect(nightVariant.validation.valid).toBe(true);
      expect(hybridVariant.validation.valid).toBe(true);

      expect(dayVariant.performance.estimatedCapacityImprovement).toBeGreaterThan(100);
      expect(nightVariant.performance.estimatedEnergySavings).toBeGreaterThan(60);
      expect(hybridVariant.metadata.variantType).toBe('hybrid');

      // Check generation statistics
      const stats = manager.getGenerationStatistics();
      expect(stats.totalGenerations).toBe(3);
      expect(stats.successRate).toBe(100);

      console.log('✅ Complex integration test passed successfully');
    });
  });

  describe('Performance Tests', () => {
    it('should handle large batch generation efficiently', async () => {
      const startTime = Date.now();

      const contexts: CombinedDeploymentContext[] = [];
      for (let i = 0; i < 10; i++) {
        contexts.push({
          primaryScenario: i % 2 === 0 ? 'urban' : 'sleep',
          urbanContext: i % 2 === 0 ? {
            populationDensity: 'high',
            buildingType: 'mixed',
            trafficPattern: 'business',
            stadiumEvents: false,
            transportHubs: true,
            capacityBoost: 'high'
          } : undefined,
          sleepContext: i % 2 === 1 ? {
            sleepModeType: 'night_time',
            energySavingLevel: 'maximum',
            wakeUpTriggers: ['emergency_call'],
            minimumCapacityGuarantee: 20,
            trafficProfile: 'residential',
            environmentalConsiderations: {
              temperatureRange: { min: 15, max: 25 },
              powerSource: 'grid',
              backupPower: true
            },
            serviceObligations: {
              emergencyServices: true,
              criticalInfrastructure: false,
              premiumUsers: false
            }
          } : undefined,
          globalSettings: {
            cellCount: 50 + i * 10,
            trafficProfile: 'medium',
            energyMode: 'balanced',
            targetEnvironment: `test_${i}`,
            customOverrides: { [`testParam_${i}`]: `testValue_${i}` }
          }
        });
      }

      const baseTemplate: RTBTemplate = {
        meta: { version: '1.0.0', author: ['Performance Test'], description: 'Performance test base' },
        configuration: {}
      };

      manager['urbanGenerator'].registerBaseTemplate('perf_base', baseTemplate);
      manager['sleepGenerator'].registerBaseTemplate('perf_base', baseTemplate);

      const results = await manager.generateBatchVariants('perf_base', contexts);
      const endTime = Date.now();
      const duration = endTime - startTime;

      expect(Object.keys(results)).toHaveLength(10);
      expect(duration).toBeLessThan(5000); // Should complete within 5 seconds

      const successCount = Object.values(results).filter(r => r.validation.valid).length;
      expect(successCount).toBe(10);

      console.log(`✅ Performance test: Generated 10 variants in ${duration}ms (${duration/10}ms per variant)`);
    });
  });
});