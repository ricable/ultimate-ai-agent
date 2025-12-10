/**
 * RTB Hierarchical Template System - Variant Generators
 *
 * Comprehensive template variant generation system for RAN deployment scenarios.
 * Supports urban high-capacity, high mobility, and sleep mode variants with
 * intelligent optimization and context-aware generation.
 *
 * Main Components:
 * - VariantGeneratorCore: Core generation engine with template inheritance
 * - UrbanVariantGenerator: High-capacity urban deployment variants (Priority 20)
 * - MobilityVariantGenerator: High-speed mobility variants (Priority 30)
 * - SleepVariantGenerator: Energy-saving sleep mode variants (Priority 40)
 * - VariantGeneratorManager: Unified management and orchestration
 */

// Core exports
export { VariantGeneratorCore } from '../variant-generator-core';
export type {
  VariantConfig,
  VariantOptimization,
  VariantGenerationOptions
} from '../variant-generator-core';

// Variant generators
export { UrbanVariantGenerator } from './urban-variant';
export type { UrbanDeploymentContext } from './urban-variant';

export { MobilityVariantGenerator } from './mobility-variant';
export type { MobilityDeploymentContext } from './mobility-variant';

export { SleepVariantGenerator } from './sleep-variant';
export type { SleepModeContext } from './sleep-variant';

// Management and orchestration
export { VariantGeneratorManager } from '../variant-generator-manager';
export type {
  CombinedDeploymentContext,
  VariantGenerationResult
} from '../variant-generator-manager';

// Examples and usage patterns
export {
  exampleUrbanDeployment,
  exampleHighSpeedTrainDeployment,
  exampleNightTimeSleepMode,
  exampleHybridDeployment,
  exampleBatchGeneration,
  examplePerformanceAnalysis,
  exampleAdvancedScenarios,
  runAllExamples
} from '../examples/variant-generator-examples';

/**
 * Factory function to create a pre-configured variant generator manager
 * with all generators initialized and ready for use.
 */
export function createVariantGeneratorManager(): VariantGeneratorManager {
  return new VariantGeneratorManager();
}

/**
 * Quick generation function for common scenarios
 */
export async function generateVariantForScenario(
  scenario: 'urban_dense' | 'high_speed_train' | 'night_time' | 'weekend',
  baseTemplateName: string,
  customizations?: any
) {
  const manager = createVariantGeneratorManager();

  switch (scenario) {
    case 'urban_dense':
      return await manager.generateVariant(baseTemplateName, {
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
          cellCount: 200,
          trafficProfile: 'high',
          energyMode: 'performance',
          targetEnvironment: 'urban_dense',
          customOverrides: customizations || {}
        }
      });

    case 'high_speed_train':
      return await manager.generateVariant(baseTemplateName, {
        primaryScenario: 'mobility',
        mobilityContext: {
          mobilityType: 'high_speed_train',
          speedRange: { min: 200, max: 350 },
          handoverFrequency: 'very_high',
          cellSize: 'large',
          trafficPattern: 'continuous',
          servicePriority: 'data',
          redundancyLevel: 'maximum'
        },
        globalSettings: {
          cellCount: 50,
          trafficProfile: 'medium',
          energyMode: 'balanced',
          targetEnvironment: 'high_speed_railway',
          customOverrides: customizations || {}
        }
      });

    case 'night_time':
      return await manager.generateVariant(baseTemplateName, {
        primaryScenario: 'sleep',
        sleepContext: {
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
        },
        globalSettings: {
          cellCount: 100,
          trafficProfile: 'low',
          energyMode: 'energy_saving',
          targetEnvironment: 'night_time',
          customOverrides: customizations || {}
        }
      });

    case 'weekend':
      return await manager.generateVariant(baseTemplateName, {
        primaryScenario: 'sleep',
        sleepContext: {
          sleepModeType: 'weekend',
          energySavingLevel: 'aggressive',
          wakeUpTriggers: ['emergency_call', 'scheduled_events'],
          minimumCapacityGuarantee: 30,
          trafficProfile: 'mixed',
          environmentalConsiderations: {
            temperatureRange: { min: 18, max: 30 },
            powerSource: 'hybrid',
            backupPower: true
          },
          serviceObligations: {
            emergencyServices: true,
            criticalInfrastructure: true,
            premiumUsers: true
          }
        },
        globalSettings: {
          cellCount: 80,
          trafficProfile: 'low',
          energyMode: 'energy_saving',
          targetEnvironment: 'weekend_mode',
          customOverrides: customizations || {}
        }
      });

    default:
      throw new Error(`Unknown scenario: ${scenario}`);
  }
}

/**
 * Get variant generator system information
 */
export function getSystemInfo() {
  return {
    version: '1.0.0',
    name: 'RTB Hierarchical Template System - Variant Generators',
    description: 'Advanced template variant generation for RAN deployment scenarios',
    supportedScenarios: ['urban', 'mobility', 'sleep', 'hybrid'],
    features: [
      'Priority-based template inheritance',
      'Context-aware optimization',
      'Custom function generation',
      'Conditional logic support',
      'Performance estimation',
      'Batch generation',
      'Validation framework',
      'Statistics tracking'
    ],
    performanceTargets: {
      generationTime: '< 100ms per variant',
      batchProcessing: '< 5 seconds for 10 variants',
      memoryUsage: '< 50MB for typical deployments',
      validationTime: '< 10ms per template'
    }
  };
}

/**
 * Recommended usage patterns and best practices
 */
export const usageGuidelines = {
  recommendedPriorities: {
    urban: 20,
    mobility: 30,
    sleep: 40,
    hybrid: 25
  },
  bestPractices: [
    'Always validate generated templates before deployment',
    'Use custom overrides for deployment-specific parameters',
    'Monitor generation statistics for performance optimization',
    'Test variants in staging environments before production',
    'Use batch generation for multiple scenarios',
    'Leverage recommendations for optimal variant selection',
    'Track optimization history for learning and improvement'
  ],
  integrationTips: [
    'Register base templates before variant generation',
    'Use context objects for complex deployment scenarios',
    'Implement proper error handling for generation failures',
    'Cache generated variants for repeated use',
    'Use performance statistics to optimize generation pipeline'
  ]
};

// Default export for convenience
export default {
  VariantGeneratorManager,
  createVariantGeneratorManager,
  generateVariantForScenario,
  getSystemInfo,
  usageGuidelines
};