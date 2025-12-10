import { describe, test, expect, beforeEach, afterEach } from '@jest/globals';
import { BaseTemplateGenerator } from '../../src/rtb/hierarchical-template-system/base-generator';
import { mockXMLSchema } from '../test-data/mock-templates';
import { RTBTemplate, RTBParameter } from '../../../src/types/rtb-types';

// Mock base template generator for testing
class MockBaseTemplateGenerator {
  private xmlSchema: any;
  private parameterMapping: Map<string, RTBParameter> = new Map();

  constructor(xmlSchema?: any) {
    this.xmlSchema = xmlSchema || mockXMLSchema;
    this.initializeParameterMapping();
  }

  private initializeParameterMapping(): void {
    if (this.xmlSchema.parameters) {
      for (const param of this.xmlSchema.parameters) {
        this.parameterMapping.set(param.name, param);
      }
    }
  }

  generateBaseTemplate(moClasses: string[], options?: {
    includeConditions?: boolean;
    includeEvaluations?: boolean;
    optimizationLevel?: 'basic' | 'standard' | 'comprehensive';
  }): RTBTemplate {
    const opts = {
      includeConditions: true,
      includeEvaluations: true,
      optimizationLevel: 'standard' as const,
      ...options
    };

    const configuration = this.generateBaseConfiguration(moClasses, opts);
    const customFunctions = this.generateBaseCustomFunctions(moClasses, opts);
    const conditions = opts.includeConditions ? this.generateBaseConditions(moClasses, opts) : undefined;
    const evaluations = opts.includeEvaluations ? this.generateBaseEvaluations(moClasses, opts) : undefined;

    return {
      meta: {
        version: '1.0.0',
        author: ['RTB System', 'Base Template Generator'],
        description: `Auto-generated base template for ${moClasses.join(', ')}`,
        tags: ['base', 'auto-generated', 'lte', ...moClasses],
        environment: 'base',
        priority: 1,
        source: 'xml-derived'
      },
      custom: customFunctions,
      configuration,
      conditions,
      evaluations
    };
  }

  private generateBaseConfiguration(moClasses: string[], options: any): Record<string, any> {
    const configuration: Record<string, any> = {};

    for (const moClass of moClasses) {
      const moConfig = this.generateMOConfiguration(moClass, options);
      configuration[moClass] = moConfig;
    }

    return configuration;
  }

  private generateMOConfiguration(moClass: string, options: any): Record<string, any> {
    const moConfig: Record<string, any> = {};

    // Get parameters for this MO class
    const relevantParams = Array.from(this.parameterMapping.values())
      .filter(param => param.hierarchy.includes(moClass));

    for (const param of relevantParams) {
      moConfig[param.name] = this.getParameterValue(param, options);
    }

    // Add MO-specific default configurations
    this.addMOSpecificDefaults(moClass, moConfig, options);

    return moConfig;
  }

  private getParameterValue(param: RTBParameter, options: any): any {
    // Use default value if available
    if (param.defaultValue !== undefined) {
      return param.defaultValue;
    }

    // Generate appropriate default based on parameter type and constraints
    if (param.constraints) {
      for (const constraint of param.constraints) {
        if (constraint.type === 'range') {
          const range = constraint.value;
          if (param.vsDataType === 'int32' || param.vsDataType === 'int64') {
            return Math.floor((range.min + range.max) / 2);
          } else if (param.vsDataType === 'float' || param.vsDataType === 'double') {
            return (range.min + range.max) / 2;
          }
        } else if (constraint.type === 'enum') {
          return constraint.value[0]; // Use first enum value
        }
      }
    }

    // Fallback defaults based on data type
    switch (param.vsDataType) {
      case 'boolean':
        return false;
      case 'int32':
      case 'int64':
        return 0;
      case 'float':
      case 'double':
        return 0.0;
      case 'string':
        return '';
      default:
        return null;
    }
  }

  private addMOSpecificDefaults(moClass: string, moConfig: Record<string, any>, options: any): void {
    switch (moClass) {
      case 'EUtranCellFDD':
        this.addEUtranCellFDDDefaults(moConfig, options);
        break;
      case 'AnrFunction':
        this.addAnrFunctionDefaults(moConfig, options);
        break;
      case 'FeatureState':
        this.addFeatureStateDefaults(moConfig, options);
        break;
      default:
        // Add generic defaults for unknown MO classes
        if (Object.keys(moConfig).length === 0) {
          moConfig['enabled'] = false;
        }
    }
  }

  private addEUtranCellFDDDefaults(moConfig: Record<string, any>, options: any): void {
    const defaults = {
      referenceSignalPower: 15,
      pb: 0,
      prachRootSequenceIndex: 0,
      prachConfigIndex: 0,
      phichDuration: 'normal',
      phichResource: 'oneSixth',
      siConfiguration: {
        schedulingInterval: 80,
        sfnOffset: 0
      },
      antennaPortsCount: 1,
      transmissionMode: 1,
      uplinkPowerControl: {
        p0NominalPUSCH: -80,
        alpha: 0.7,
        p0NominalPUCCH: -80
      },
      schedulingInfo: {
        siPeriodicity: 'rf8'
      }
    };

    // Apply defaults only if not already set
    for (const [key, value] of Object.entries(defaults)) {
      if (!(key in moConfig)) {
        moConfig[key] = value;
      }
    }

    // Optimization level specific adjustments
    if (options.optimizationLevel === 'comprehensive') {
      moConfig.ul256qamEnabled = true;
      moConfig.dl256qamEnabled = true;
      moConfig.twoAntennaPortActivated = true;
    }
  }

  private addAnrFunctionDefaults(moConfig: Record<string, any>, options: any): void {
    const defaults = {
      anrFunctionEnabled: false,
      maxUePerPucch: 1,
      anrOptimizationEnabled: false,
      hoPreparationAllowed: true
    };

    for (const [key, value] of Object.entries(defaults)) {
      if (!(key in moConfig)) {
        moConfig[key] = value;
      }
    }
  }

  private addFeatureStateDefaults(moConfig: Record<string, any>, options: any): void {
    const defaults = {
      featureState: 'DEACTIVATED',
      featureMode: 'normal'
    };

    for (const [key, value] of Object.entries(defaults)) {
      if (!(key in moConfig)) {
        moConfig[key] = value;
      }
    }
  }

  private generateBaseCustomFunctions(moClasses: string[], options: any): any[] {
    const functions: any[] = [];

    // Basic calculation functions
    functions.push({
      name: 'calculateCoverage',
      args: ['power', 'frequency'],
      body: [
        '// Calculate cell coverage based on power and frequency',
        'const pathLoss = 128.1 + 37.6 * Math.log10(frequency / 1000);',
        'return power - pathLoss + 10; // Add 10dB margin'
      ]
    });

    functions.push({
      name: 'calculateCapacity',
      args: ['bandwidth', 'mimoLayers', 'modulationIndex'],
      body: [
        '// Calculate theoretical cell capacity',
        'const spectralEfficiency = mimoLayers * modulationIndex;',
        'const capacity = bandwidth * spectralEfficiency * Math.log2(1 + 10); // SNR assumption',
        'return Math.round(capacity);'
      ]
    });

    functions.push({
      name: 'validateParameter',
      args: ['paramName', 'value', 'constraints'],
      body: [
        '// Validate parameter against constraints',
        'if (!constraints) return true;',
        'for (const constraint of constraints) {',
        '  if (constraint.type === "range") {',
        '    if (value < constraint.value.min || value > constraint.value.max) return false;',
        '  } else if (constraint.type === "enum") {',
        '    if (!constraint.value.includes(value)) return false;',
        '  }',
        '}',
        'return true;'
      ]
    });

    // Add MO-specific functions if present
    if (moClasses.includes('EUtranCellFDD')) {
      functions.push({
        name: 'optimizeCellParameters',
        args: ['load', 'interference'],
        body: [
          '// Optimize cell parameters based on load and interference',
          'const adjustments = {};',
          'if (load > 0.8) {',
          '  adjustments.referenceSignalPower = Math.min(20, 15 + Math.ceil(load * 3));',
          '}',
          'if (interference > 10) {',
          '  adjustments.qRxLevMin = Math.max(-120, -140 + Math.ceil(interference / 2));',
          '}',
          'return adjustments;'
        ]
      });
    }

    if (options.optimizationLevel === 'comprehensive') {
      functions.push({
        name: 'advancedOptimization',
        args: ['kpiHistory', 'trends'],
        body: [
          '// Advanced optimization using KPI history and trends',
          'const avgKpi = kpiHistory.reduce((a, b) => a + b, 0) / kpiHistory.length;',
          'const trendSlope = trends[trends.length - 1] - trends[0];',
          'const adjustments = {',
          '  performanceBoost: avgKpi < 0.8 ? 2 : 0,',
          '  trendCompensation: Math.abs(trendSlope) > 0.1 ? Math.sign(trendSlope) : 0',
          '};',
          'return adjustments;'
        ]
      });
    }

    return functions;
  }

  private generateBaseConditions(moClasses: string[], options: any): Record<string, any> {
    const conditions: Record<string, any> = {};

    // Load-based conditions
    if (moClasses.includes('EUtranCellFDD')) {
      conditions['loadBasedOptimization'] = {
        if: '$config.EUtranCellFDD.referenceSignalPower > 18',
        then: { 'EUtranCellFDD.qRxLevMin': -130 },
        else: { 'EUtranCellFDD.qRxLevMin': -140 }
      };

      conditions['coverageCondition'] = {
        if: '$eval($custom.calculateCoverage($config.EUtranCellFDD.referenceSignalPower, 2600)) > -110',
        then: { 'EUtranCellFDD.cellEdgeUserThroughput': 'high' },
        else: { 'EUtranCellFDD.cellEdgeUserThroughput': 'low' }
      };
    }

    // Feature-based conditions
    if (moClasses.includes('FeatureState')) {
      conditions['featureActivationCondition'] = {
        if: '$config.FeatureState.featureState == "ACTIVATED"',
        then: {
          'AnrFunction.anrFunctionEnabled': true,
          'EUtranCellFDD.dl256qamEnabled': true
        },
        else: {
          'AnrFunction.anrFunctionEnabled': false,
          'EUtranCellFDD.dl256qamEnabled': false
        }
      };
    }

    // Optimization level specific conditions
    if (options.optimizationLevel === 'comprehensive') {
      conditions['performanceBasedOptimization'] = {
        if: '$eval($custom.advancedOptimization([0.85, 0.87, 0.89], [0.02, 0.01, 0.03]).performanceBoost) > 0',
        then: {
          'EUtranCellFDD.transmissionMode': 4,
          'EUtranCellFDD.twoAntennaPortActivated': true
        },
        else: {
          'EUtranCellFDD.transmissionMode': 2,
          'EUtranCellFDD.twoAntennaPortActivated': false
        }
      };
    }

    return conditions;
  }

  private generateBaseEvaluations(moClasses: string[], options: any): Record<string, any> {
    const evaluations: Record<string, any> = {};

    // Basic evaluations
    if (moClasses.includes('EUtranCellFDD')) {
      evaluations['coverageCalculation'] = {
        eval: '$custom.calculateCoverage($config.EUtranCellFDD.referenceSignalPower, 2600)',
        args: ['power', 'frequency']
      };

      evaluations['capacityEstimate'] = {
        eval: '$custom.calculateCapacity(20, $config.EUtranCellFDD.antennaPortsCount || 1, 4)',
        args: ['bandwidth', 'mimoLayers', 'modulationIndex']
      };
    }

    // Parameter validation evaluations
    evaluations['parameterValidation'] = {
      eval: '$custom.validateParameter("qRxLevMin", $config.EUtranCellFDD.qRxLevMin, [{type: "range", value: {min: -140, max: -44}}])',
      args: ['paramName', 'value', 'constraints']
    };

    // Optimization level specific evaluations
    if (options.optimizationLevel === 'comprehensive') {
      evaluations['optimizationScore'] = {
        eval: '($eval($custom.advancedOptimization([0.85, 0.87, 0.89], [0.02, 0.01, 0.03]).performanceBoost || 0) * 50 + ($config.EUtranCellFDD.referenceSignalPower / 20) * 30 + ($config.EUtranCellFDD.transmissionMode / 4) * 20',
        args: []
      };

      evaluations['systemHealth'] = {
        eval: '($eval($custom.calculateCoverage($config.EUtranCellFDD.referenceSignalPower, 2600)) > -110 ? 50 : 0) + ($config.FeatureState.featureState == "ACTIVATED" ? 30 : 0) + ($config.AnrFunction.anrFunctionEnabled ? 20 : 0)',
        args: []
      };
    }

    return evaluations;
  }

  validateBaseTemplate(template: RTBTemplate): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Check basic structure
    if (!template.meta) {
      errors.push('Template metadata is required');
    } else {
      if (!template.meta.version) {
        errors.push('Template version is required');
      }
      if (!template.meta.description) {
        errors.push('Template description is required');
      }
      if (!template.meta.tags?.includes('base')) {
        errors.push('Base template must include "base" tag');
      }
      if (template.meta.priority !== 1) {
        errors.push('Base template should have priority 1');
      }
    }

    // Check configuration
    if (!template.configuration) {
      errors.push('Template configuration is required');
    } else {
      // Validate MO configurations
      for (const [moName, moConfig] of Object.entries(template.configuration)) {
        if (!moConfig || typeof moConfig !== 'object') {
          errors.push(`Invalid MO configuration for ${moName}`);
          continue;
        }

        // Check for required parameters based on MO type
        this.validateMOConfiguration(moName, moConfig, errors);
      }
    }

    // Check custom functions
    if (!template.custom || template.custom.length === 0) {
      errors.push('Base template should include custom functions');
    } else {
      for (const func of template.custom) {
        if (!func.name) {
          errors.push('Custom function name is required');
        }
        if (!func.args || !Array.isArray(func.args)) {
          errors.push(`Custom function ${func.name} args must be an array`);
        }
        if (!func.body || !Array.isArray(func.body)) {
          errors.push(`Custom function ${func.name} body must be an array`);
        }
      }
    }

    // Check conditions and evaluations if included
    if (template.conditions) {
      for (const [conditionName, condition] of Object.entries(template.conditions)) {
        if (!condition.if || !condition.then || !condition.else) {
          errors.push(`Condition ${conditionName} must have if, then, and else properties`);
        }
      }
    }

    if (template.evaluations) {
      for (const [evalName, evaluation] of Object.entries(template.evaluations)) {
        if (!evaluation.eval) {
          errors.push(`Evaluation ${evalName} must have eval property`);
        }
      }
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  private validateMOConfiguration(moName: string, moConfig: any, errors: string[]): void {
    switch (moName) {
      case 'EUtranCellFDD':
        this.validateEUtranCellFDD(moConfig, errors);
        break;
      case 'AnrFunction':
        this.validateAnrFunction(moConfig, errors);
        break;
      case 'FeatureState':
        this.validateFeatureState(moConfig, errors);
        break;
    }
  }

  private validateEUtranCellFDD(config: any, errors: string[]): void {
    const requiredParams = ['qRxLevMin', 'qQualMin', 'referenceSignalPower'];
    for (const param of requiredParams) {
      if (!(param in config)) {
        errors.push(`EUtranCellFDD must include ${param} parameter`);
      }
    }

    // Validate parameter ranges
    if (config.qRxLevMin !== undefined) {
      if (config.qRxLevMin < -140 || config.qRxLevMin > -44) {
        errors.push('EUtranCellFDD qRxLevMin must be between -140 and -44');
      }
    }

    if (config.qQualMin !== undefined) {
      if (config.qQualMin < -34 || config.qQualMin > -3) {
        errors.push('EUtranCellFDD qQualMin must be between -34 and -3');
      }
    }

    if (config.referenceSignalPower !== undefined) {
      if (config.referenceSignalPower < -60 || config.referenceSignalPower > 50) {
        errors.push('EUtranCellFDD referenceSignalPower must be between -60 and 50');
      }
    }
  }

  private validateAnrFunction(config: any, errors: string[]): void {
    const requiredParams = ['anrFunctionEnabled'];
    for (const param of requiredParams) {
      if (!(param in config)) {
        errors.push(`AnrFunction must include ${param} parameter`);
      }
    }
  }

  private validateFeatureState(config: any, errors: string[]): void {
    const requiredParams = ['featureState'];
    for (const param of requiredParams) {
      if (!(param in config)) {
        errors.push(`FeatureState must include ${param} parameter`);
      }
    }

    if (config.featureState !== undefined) {
      const validStates = ['ACTIVATED', 'DEACTIVATED'];
      if (!validStates.includes(config.featureState)) {
        errors.push(`FeatureState.featureState must be one of: ${validStates.join(', ')}`);
      }
    }
  }

  optimizeBaseTemplate(template: RTBTemplate, kpiData: any): RTBTemplate {
    const optimized = JSON.parse(JSON.stringify(template)); // Deep clone

    // Simple optimization based on KPI data
    if (kpiData.coverageIssues && optimized.configuration['EUtranCellFDD']) {
      const cellConfig = optimized.configuration['EUtranCellFDD'];
      if (cellConfig.referenceSignalPower !== undefined) {
        cellConfig.referenceSignalPower = Math.min(20, cellConfig.referenceSignalPower + 3);
      }
      if (cellConfig.qRxLevMin !== undefined) {
        cellConfig.qRxLevMin = Math.max(-130, cellConfig.qRxLevMin + 5);
      }
    }

    if (kpiData.capacityIssues && optimized.configuration['EUtranCellFDD']) {
      const cellConfig = optimized.configuration['EUtranCellFDD'];
      if (cellConfig.transmissionMode !== undefined) {
        cellConfig.transmissionMode = Math.min(4, cellConfig.transmissionMode + 1);
      }
    }

    return optimized;
  }
}

describe('Base Template Generator', () => {
  let generator: MockBaseTemplateGenerator;

  beforeEach(() => {
    generator = new MockBaseTemplateGenerator();
  });

  describe('Basic Template Generation', () => {
    test('should generate base template for single MO class', () => {
      const template = generator.generateBaseTemplate(['EUtranCellFDD']);

      // Check metadata
      expect(template.meta?.version).toBe('1.0.0');
      expect(template.meta?.description).toContain('EUtranCellFDD');
      expect(template.meta?.tags).toContain('base');
      expect(template.meta?.tags).toContain('auto-generated');
      expect(template.meta?.tags).toContain('lte');
      expect(template.meta?.tags).toContain('EUtranCellFDD');
      expect(template.meta?.priority).toBe(1);
      expect(template.meta?.source).toBe('xml-derived');

      // Check configuration
      expect(template.configuration).toHaveProperty('EUtranCellFDD');
      const cellConfig = template.configuration['EUtranCellFDD'];
      expect(cellConfig).toHaveProperty('qRxLevMin');
      expect(cellConfig).toHaveProperty('qQualMin');
      expect(cellConfig).toHaveProperty('referenceSignalPower');
      expect(cellConfig).toHaveProperty('pb');
      expect(cellConfig).toHaveProperty('prachRootSequenceIndex');

      // Check custom functions
      expect(template.custom).toHaveLength(3);
      expect(template.custom?.map(f => f.name)).toContain('calculateCoverage');
      expect(template.custom?.map(f => f.name)).toContain('calculateCapacity');
      expect(template.custom?.map(f => f.name)).toContain('validateParameter');

      // Check conditions and evaluations
      expect(template.conditions).toBeDefined();
      expect(template.evaluations).toBeDefined();
    });

    test('should generate base template for multiple MO classes', () => {
      const moClasses = ['EUtranCellFDD', 'AnrFunction', 'FeatureState'];
      const template = generator.generateBaseTemplate(moClasses);

      // Check configuration includes all MO classes
      for (const moClass of moClasses) {
        expect(template.configuration).toHaveProperty(moClass);
      }

      // Check each MO has required parameters
      expect(template.configuration['EUtranCellFDD']).toHaveProperty('qRxLevMin');
      expect(template.configuration['AnrFunction']).toHaveProperty('anrFunctionEnabled');
      expect(template.configuration['FeatureState']).toHaveProperty('featureState');

      // Check MO-specific custom functions
      expect(template.custom?.map(f => f.name)).toContain('optimizeCellParameters');
    });

    test('should handle MO class without XML parameters', () => {
      const template = generator.generateBaseTemplate(['UnknownMO']);

      expect(template.configuration).toHaveProperty('UnknownMO');
      expect(template.configuration['UnknownMO']).toHaveProperty('enabled');
      expect(template.configuration['UnknownMO'].enabled).toBe(false);
    });

    test('should use XML parameter values when available', () => {
      const template = generator.generateBaseTemplate(['EUtranCellFDD']);

      const cellConfig = template.configuration['EUtranCellFDD'];

      // Should use default values from XML schema
      expect(cellConfig.qRxLevMin).toBe(-140); // From mockXMLSchema
      expect(cellConfig.qQualMin).toBe(-18);   // From mockXMLSchema
      expect(cellConfig.cellIndividualOffset).toBe(0); // From mockXMLSchema
    });
  });

  describe('Optimization Levels', () => {
    test('should generate basic optimization level template', () => {
      const template = generator.generateBaseTemplate(['EUtranCellFDD'], {
        optimizationLevel: 'basic'
      });

      const cellConfig = template.configuration['EUtranCellFDD'];

      // Basic level should not include advanced features
      expect(cellConfig.ul256qamEnabled).toBeUndefined();
      expect(cellConfig.dl256qamEnabled).toBeUndefined();
      expect(cellConfig.twoAntennaPortActivated).toBeUndefined();

      // Should not include advanced custom functions
      expect(template.custom?.map(f => f.name)).not.toContain('advancedOptimization');
    });

    test('should generate standard optimization level template', () => {
      const template = generator.generateBaseTemplate(['EUtranCellFDD'], {
        optimizationLevel: 'standard'
      });

      // Standard is default behavior
      expect(template.custom).toHaveLength(4); // 3 basic + 1 MO-specific
      expect(template.custom?.map(f => f.name)).toContain('optimizeCellParameters');
    });

    test('should generate comprehensive optimization level template', () => {
      const template = generator.generateBaseTemplate(['EUtranCellFDD'], {
        optimizationLevel: 'comprehensive'
      });

      const cellConfig = template.configuration['EUtranCellFDD'];

      // Comprehensive level should include advanced features
      expect(cellConfig.ul256qamEnabled).toBe(true);
      expect(cellConfig.dl256qamEnabled).toBe(true);
      expect(cellConfig.twoAntennaPortActivated).toBe(true);

      // Should include advanced custom functions
      expect(template.custom?.map(f => f.name)).toContain('advancedOptimization');

      // Should include advanced conditions and evaluations
      expect(template.conditions).toHaveProperty('performanceBasedOptimization');
      expect(template.evaluations).toHaveProperty('optimizationScore');
      expect(template.evaluations).toHaveProperty('systemHealth');
    });
  });

  describe('Conditional Generation Options', () => {
    test('should generate template without conditions', () => {
      const template = generator.generateBaseTemplate(['EUtranCellFDD'], {
        includeConditions: false
      });

      expect(template.conditions).toBeUndefined();
      expect(template.evaluations).toBeDefined(); // Evaluations should still be included
    });

    test('should generate template without evaluations', () => {
      const template = generator.generateBaseTemplate(['EUtranCellFDD'], {
        includeEvaluations: false
      });

      expect(template.evaluations).toBeUndefined();
      expect(template.conditions).toBeDefined(); // Conditions should still be included
    });

    test('should generate template with minimal options', () => {
      const template = generator.generateBaseTemplate(['EUtranCellFDD'], {
        includeConditions: false,
        includeEvaluations: false,
        optimizationLevel: 'basic'
      });

      expect(template.conditions).toBeUndefined();
      expect(template.evaluations).toBeUndefined();
      expect(template.custom).toHaveLength(3); // Only basic functions
    });
  });

  describe('Template Validation', () => {
    test('should validate correctly generated base template', () => {
      const template = generator.generateBaseTemplate(['EUtranCellFDD', 'AnrFunction']);
      const validation = generator.validateBaseTemplate(template);

      expect(validation.isValid).toBe(true);
      expect(validation.errors).toHaveLength(0);
    });

    test('should detect invalid template structure', () => {
      const invalidTemplate: RTBTemplate = {
        meta: {
          version: '1.0.0',
          description: 'Invalid template'
          // Missing base tag and correct priority
        },
        configuration: {
          'EUtranCellFDD': {
            // Missing required parameters
          }
        },
        custom: [] // Missing custom functions
      };

      const validation = generator.validateBaseTemplate(invalidTemplate);

      expect(validation.isValid).toBe(false);
      expect(validation.errors.length).toBeGreaterThan(0);
      expect(validation.errors).toContain('Base template must include "base" tag');
      expect(validation.errors).toContain('Base template should have priority 1');
      expect(validation.errors).toContain('EUtranCellFDD must include qRxLevMin parameter');
      expect(validation.errors).toContain('Base template should include custom functions');
    });

    test('should validate parameter ranges', () => {
      const template = generator.generateBaseTemplate(['EUtranCellFDD']);

      // Manually create invalid configuration for testing
      template.configuration['EUtranCellFDD'] = {
        qRxLevMin: -200, // Invalid range
        qQualMin: 100,   // Invalid range
        referenceSignalPower: 100 // Invalid range
      };

      const validation = generator.validateBaseTemplate(template);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContain('EUtranCellFDD qRxLevMin must be between -140 and -44');
      expect(validation.errors).toContain('EUtranCellFDD qQualMin must be between -34 and -3');
      expect(validation.errors).toContain('EUtranCellFDD referenceSignalPower must be between -60 and 50');
    });

    test('should validate custom function structure', () => {
      const template = generator.generateBaseTemplate(['EUtranCellFDD']);

      // Add invalid custom function
      template.custom!.push({
        name: '', // Empty name
        args: 'not-array', // Invalid args
        body: 'not-array'  // Invalid body
      } as any);

      const validation = generator.validateBaseTemplate(template);

      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContain('Custom function name is required');
      expect(validation.errors).toContain('Custom function  args must be an array');
      expect(validation.errors).toContain('Custom function  body must be an array');
    });
  });

  describe('Parameter Value Generation', () => {
    test('should generate appropriate default values for different data types', () => {
      // Create mock parameters with different types
      const mockParams = [
        { name: 'boolParam', vsDataType: 'boolean', defaultValue: undefined },
        { name: 'intParam', vsDataType: 'int32', defaultValue: undefined },
        { name: 'floatParam', vsDataType: 'float', defaultValue: undefined },
        { name: 'stringParam', vsDataType: 'string', defaultValue: undefined }
      ];

      // Manually set up parameter mapping
      for (const param of mockParams) {
        generator['parameterMapping'].set(param.name, param as any);
      }

      const template = generator.generateBaseTemplate(['TestMO']);

      const testMOConfig = template.configuration['TestMO'];

      // Check generated defaults
      expect(testMOConfig.boolParam).toBe(false);
      expect(testMOConfig.intParam).toBe(0);
      expect(testMOConfig.floatParam).toBe(0.0);
      expect(testMOConfig.stringParam).toBe('');
    });

    test('should use constraint-based defaults when available', () => {
      const mockParams = [
        {
          name: 'rangeParam',
          vsDataType: 'int32',
          defaultValue: undefined,
          constraints: [{ type: 'range', value: { min: 10, max: 20 } }]
        },
        {
          name: 'enumParam',
          vsDataType: 'string',
          defaultValue: undefined,
          constraints: [{ type: 'enum', value: ['option1', 'option2', 'option3'] }]
        }
      ];

      for (const param of mockParams) {
        generator['parameterMapping'].set(param.name, param as any);
      }

      const template = generator.generateBaseTemplate(['TestMO']);
      const testMOConfig = template.configuration['TestMO'];

      // Should use constraint-based defaults
      expect(testMOConfig.rangeParam).toBe(15); // (10 + 20) / 2
      expect(testMOConfig.enumParam).toBe('option1'); // First enum value
    });
  });

  describe('MO-Specific Configuration', () => {
    test('should generate EUtranCellFDD configuration with all defaults', () => {
      const template = generator.generateBaseTemplate(['EUtranCellFDD']);
      const cellConfig = template.configuration['EUtranCellFDD'];

      // Check that all default parameters are present
      expect(cellConfig).toHaveProperty('referenceSignalPower');
      expect(cellConfig).toHaveProperty('pb');
      expect(cellConfig).toHaveProperty('prachRootSequenceIndex');
      expect(cellConfig).toHaveProperty('prachConfigIndex');
      expect(cellConfig).toHaveProperty('phichDuration');
      expect(cellConfig).toHaveProperty('phichResource');
      expect(cellConfig).toHaveProperty('antennaPortsCount');
      expect(cellConfig).toHaveProperty('transmissionMode');

      // Check nested configurations
      expect(cellConfig).toHaveProperty('siConfiguration');
      expect(cellConfig).toHaveProperty('uplinkPowerControl');
      expect(cellConfig).toHaveProperty('schedulingInfo');

      // Check default values
      expect(cellConfig.referenceSignalPower).toBe(15);
      expect(cellConfig.pb).toBe(0);
      expect(cellConfig.transmissionMode).toBe(1);
    });

    test('should generate AnrFunction configuration with defaults', () => {
      const template = generator.generateBaseTemplate(['AnrFunction']);
      const anrConfig = template.configuration['AnrFunction'];

      expect(anrConfig).toHaveProperty('anrFunctionEnabled');
      expect(anrConfig).toHaveProperty('maxUePerPucch');
      expect(anrConfig).toHaveProperty('anrOptimizationEnabled');
      expect(anrConfig).toHaveProperty('hoPreparationAllowed');

      expect(anrConfig.anrFunctionEnabled).toBe(false);
      expect(anrConfig.maxUePerPucch).toBe(1);
    });

    test('should generate FeatureState configuration with defaults', () => {
      const template = generator.generateBaseTemplate(['FeatureState']);
      const featureConfig = template.configuration['FeatureState'];

      expect(featureConfig).toHaveProperty('featureState');
      expect(featureConfig).toHaveProperty('featureMode');

      expect(featureConfig.featureState).toBe('DEACTIVATED');
      expect(featureConfig.featureMode).toBe('normal');
    });
  });

  describe('Custom Function Generation', () => {
    test('should generate basic custom functions for all templates', () => {
      const template = generator.generateBaseTemplate(['EUtranCellFDD']);

      const functionNames = template.custom?.map(f => f.name) || [];
      expect(functionNames).toContain('calculateCoverage');
      expect(functionNames).toContain('calculateCapacity');
      expect(functionNames).toContain('validateParameter');

      // Check function structure
      for (const func of template.custom || []) {
        expect(func.name).toBeDefined();
        expect(Array.isArray(func.args)).toBe(true);
        expect(Array.isArray(func.body)).toBe(true);
        expect(func.body.length).toBeGreaterThan(0);
      }
    });

    test('should add MO-specific custom functions', () => {
      const template = generator.generateBaseTemplate(['EUtranCellFDD']);
      const functionNames = template.custom?.map(f => f.name) || [];

      expect(functionNames).toContain('optimizeCellParameters');
    });

    test('should add advanced custom functions for comprehensive optimization', () => {
      const template = generator.generateBaseTemplate(['EUtranCellFDD'], {
        optimizationLevel: 'comprehensive'
      });

      const functionNames = template.custom?.map(f => f.name) || [];
      expect(functionNames).toContain('advancedOptimization');
    });
  });

  describe('Conditions and Evaluations Generation', () => {
    test('should generate appropriate conditions for MO classes', () => {
      const template = generator.generateBaseTemplate(['EUtranCellFDD', 'AnrFunction', 'FeatureState']);

      expect(template.conditions).toHaveProperty('loadBasedOptimization');
      expect(template.conditions).toHaveProperty('coverageCondition');
      expect(template.conditions).toHaveProperty('featureActivationCondition');

      // Check condition structure
      for (const [name, condition] of Object.entries(template.conditions || {})) {
        expect(condition).toHaveProperty('if');
        expect(condition).toHaveProperty('then');
        expect(condition).toHaveProperty('else');
      }
    });

    test('should generate appropriate evaluations for MO classes', () => {
      const template = generator.generateBaseTemplate(['EUtranCellFDD']);

      expect(template.evaluations).toHaveProperty('coverageCalculation');
      expect(template.evaluations).toHaveProperty('capacityEstimate');
      expect(template.evaluations).toHaveProperty('parameterValidation');

      // Check evaluation structure
      for (const [name, evaluation] of Object.entries(template.evaluations || {})) {
        expect(evaluation).toHaveProperty('eval');
        expect(Array.isArray(evaluation.args)).toBe(true);
      }
    });

    test('should reference custom functions correctly in evaluations', () => {
      const template = generator.generateBaseTemplate(['EUtranCellFDD']);

      const coverageEval = template.evaluations?.coverageCalculation?.eval;
      expect(coverageEval).toContain('$custom.calculateCoverage');
      expect(coverageEval).toContain('$config.EUtranCellFDD.referenceSignalPower');

      const capacityEval = template.evaluations?.capacityEstimate?.eval;
      expect(capacityEval).toContain('$custom.calculateCapacity');
    });
  });

  describe('Template Optimization', () => {
    test('should optimize template based on KPI data', () => {
      const template = generator.generateBaseTemplate(['EUtranCellFDD']);
      const originalPower = template.configuration['EUtranCellFDD'].referenceSignalPower;
      const originalQrxLevMin = template.configuration['EUtranCellFDD'].qRxLevMin;

      const kpiData = {
        coverageIssues: true,
        capacityIssues: true
      };

      const optimizedTemplate = generator.optimizeBaseTemplate(template, kpiData);
      const optimizedConfig = optimizedTemplate.configuration['EUtranCellFDD'];

      // Should increase power for coverage issues
      expect(optimizedConfig.referenceSignalPower).toBeGreaterThan(originalPower);

      // Should improve qRxLevMin for coverage issues
      expect(optimizedConfig.qRxLevMin).toBeGreaterThan(originalQrxLevMin);

      // Should increase transmission mode for capacity issues
      expect(optimizedConfig.transmissionMode).toBeGreaterThan(template.configuration['EUtranCellFDD'].transmissionMode);
    });

    test('should not change template when no optimization needed', () => {
      const template = generator.generateBaseTemplate(['EUtranCellFDD']);
      const originalPower = template.configuration['EUtranCellFDD'].referenceSignalPower;

      const kpiData = {
        coverageIssues: false,
        capacityIssues: false
      };

      const optimizedTemplate = generator.optimizeBaseTemplate(template, kpiData);
      const optimizedConfig = optimizedTemplate.configuration['EUtranCellFDD'];

      // Should not change parameters when no issues
      expect(optimizedConfig.referenceSignalPower).toBe(originalPower);
    });
  });

  describe('Performance and Scalability', () => {
    test('should generate templates efficiently', () => {
      const startTime = performance.now();

      const templates: RTBTemplate[] = [];
      const moClassesList = [
        ['EUtranCellFDD'],
        ['EUtranCellFDD', 'AnrFunction'],
        ['EUtranCellFDD', 'AnrFunction', 'FeatureState'],
        ['EUtranCellFDD', 'AnrFunction', 'FeatureState', 'UnknownMO1'],
        ['EUtranCellFDD', 'AnrFunction', 'FeatureState', 'UnknownMO1', 'UnknownMO2']
      ];

      for (const moClasses of moClassesList) {
        const template = generator.generateBaseTemplate(moClasses, {
          optimizationLevel: 'comprehensive'
        });
        templates.push(template);
      }

      const endTime = performance.now();
      const generationTime = endTime - startTime;

      expect(generationTime).toBeLessThan(200); // < 200ms for 5 templates
      expect(templates).toHaveLength(5);

      // Validate all templates
      for (const template of templates) {
        const validation = generator.validateBaseTemplate(template);
        expect(validation.isValid).toBe(true);
      }
    });

    test('should handle large MO class sets efficiently', () => {
      const largeMOSet = Array.from({ length: 20 }, (_, i) => `UnknownMO${i}`);

      const startTime = performance.now();
      const template = generator.generateBaseTemplate(largeMOSet, {
        optimizationLevel: 'comprehensive'
      });
      const endTime = performance.now();

      expect(endTime - startTime).toBeLessThan(500); // < 500ms for large MO set

      // Validate template structure
      const validation = generator.validateBaseTemplate(template);
      expect(validation.isValid).toBe(true);

      // Check that all MO classes are included
      for (const moClass of largeMOSet) {
        expect(template.configuration).toHaveProperty(moClass);
      }
    });
  });
});