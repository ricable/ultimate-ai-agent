/**
 * cmedit Command Generation Engine - Main Export
 *
 * Core module providing intelligent command generation for Ericsson RAN ENM CLI
 * integration with cognitive optimization and RAN expertise patterns.
 */

// Export core types
export * from './types';

// Export core engine classes
export { CmeditCommandParser } from './command-parser';
export { FDNPathGenerator } from './fdn-generator';
export { ConstraintsValidator } from './constraints-validator';
export { EricssonRANExpertiseEngine } from './ericsson-expertise';
export { CmeditEngine } from './cmedit-engine';

// Re-export commonly used types for convenience
export type {
  CmeditCommand,
  CmeditCommandType,
  CommandContext,
  CommandGenerationResult,
  FDNPath,
  EricssonExpertisePattern,
  CognitiveLevel,
  NetworkContext,
  OperationPurpose
} from './types';

// Export factory functions for easy instantiation
export function createCmeditEngine(
  moHierarchy: any,
  reservedByRelationships: any[],
  options?: any
): CmeditEngine {
  return new CmeditEngine(moHierarchy, reservedByRelationships, options);
}

// Export utility functions
export function buildDefaultCommandContext(
  overrides?: Partial<CommandContext>
): CommandContext {
  return {
    moClasses: [],
    purpose: 'configuration_management',
    networkContext: {
      technology: '4G',
      environment: 'urban_medium',
      vendor: {
        primary: 'ericsson',
        multiVendor: false,
        compatibilityMode: false
      },
      topology: {
        cellCount: 1,
        siteCount: 1,
        frequencyBands: [],
        carrierAggregation: false,
        networkSharing: false
      }
    },
    cognitiveLevel: 'enhanced',
    expertisePatterns: [],
    generatedAt: new Date(),
    priority: 'medium',
    ...overrides
  };
}

// Version information
export const CMEDIT_ENGINE_VERSION = '1.0.0';
export const SUPPORTED_COMMAND_TYPES = ['get', 'set', 'create', 'delete', 'mon', 'unmon'];
export const DEFAULT_COGNITIVE_LEVELS = ['basic', 'enhanced', 'cognitive', 'autonomous', 'conscious'];