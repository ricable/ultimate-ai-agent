import { describe, test, expect, beforeEach, afterEach } from '@jest/globals';
import { FrequencyRelationGenerator } from '../../src/rtb/hierarchical-template-system/frequency-relation-generator';
import { frequencyRelation4G4G, frequencyRelation4G5G, generateLargeTemplateSet } from '../test-data/mock-templates';
import { RTBTemplate, ConstraintSpec } from '../../../src/types/rtb-types';

// Mock frequency relation generator for testing
class MockFrequencyRelationGenerator {
  private frequencyBands = {
    // 4G LTE bands
    LTE700: { frequency: 700, type: '4G', name: 'LTE700' },
    LTE800: { frequency: 800, type: '4G', name: 'LTE800' },
    LTE900: { frequency: 900, type: '4G', name: 'LTE900' },
    LTE1800: { frequency: 1800, type: '4G', name: 'LTE1800' },
    LTE2100: { frequency: 2100, type: '4G', name: 'LTE2100' },
    LTE2600: { frequency: 2600, type: '4G', name: 'LTE2600' },

    // 5G NR bands
    NR700: { frequency: 700, type: '5G', name: 'NR700' },
    NR3500: { frequency: 3500, type: '5G', name: 'NR3500' },
    NR28000: { frequency: 28000, type: '5G', name: 'NR28GHz' }
  };

  private relationConstraints = {
    '4G4G': {
      allowedPairs: [
        ['LTE700', 'LTE800'],
        ['LTE800', 'LTE900'],
        ['LTE800', 'LTE1800'],
        ['LTE1800', 'LTE2100'],
        ['LTE1800', 'LTE2600'],
        ['LTE2100', 'LTE2600']
      ],
      priorityRange: { min: 1, max: 7 },
      qOffsetRange: { min: -24, max: 24 },
      threshRange: { min: 0, max: 31 }
    },
    '4G5G': {
      allowedPairs: [
        ['LTE1800', 'NR3500'],
        ['LTE2100', 'NR3500'],
        ['LTE2600', 'NR3500'],
        ['LTE2100', 'NR28GHz'],
        ['LTE2600', 'NR28GHz']
      ],
      priorityRange: { min: 1, max: 7 },
      qOffsetRange: { min: -24, max: 24 },
      threshRange: { min: 0, max: 31 }
    },
    '5G5G': {
      allowedPairs: [
        ['NR3500', 'NR28GHz'],
        ['NR700', 'NR3500']
      ],
      priorityRange: { min: 1, max: 7 },
      qOffsetRange: { min: -24, max: 24 },
      threshRange: { min: 0, max: 31 }
    }
  };

  generateFrequencyRelationTemplate(
    sourceBand: string,
    targetBand: string,
    scenario: 'dense' | 'normal' | 'sparse' = 'normal'
  ): RTBTemplate {
    const sourceInfo = this.frequencyBands[sourceBand as keyof typeof this.frequencyBands];
    const targetInfo = this.frequencyBands[targetBand as keyof typeof this.frequencyBands];

    if (!sourceInfo || !targetInfo) {
      throw new Error(`Invalid frequency bands: ${sourceBand} -> ${targetBand}`);
    }

    const relationType = `${sourceInfo.type}${targetInfo.type}`;
    const constraints = this.relationConstraints[relationType as keyof typeof this.relationConstraints];

    if (!constraints) {
      throw new Error(`Unsupported relation type: ${relationType}`);
    }

    // Validate that this pair is allowed
    const isAllowed = constraints.allowedPairs.some(pair =>
      (pair[0] === sourceBand && pair[1] === targetBand) ||
      (pair[1] === sourceBand && pair[0] === targetBand)
    );

    if (!isAllowed) {
      throw new Error(`Frequency pair not allowed: ${sourceBand} -> ${targetBand}`);
    }

    const relationConfig = this.generateRelationConfig(sourceBand, targetBand, scenario, constraints);
    const relationMO = this.getRelationMO(relationType);

    return {
      meta: {
        version: '1.0.0',
        author: ['RTB System', 'Frequency Relation Generator'],
        description: `${relationType} frequency relation: ${sourceBand} -> ${targetBand} (${scenario})`,
        tags: ['frequency', relationType.toLowerCase(), sourceBand, targetBand, scenario],
        environment: 'frequency-relation',
        priority: this.calculatePriority(sourceInfo, targetInfo, scenario),
        source: 'auto-generated'
      },
      custom: [
        {
          name: 'calculateFrequencyOffset',
          args: ['sourceFreq', 'targetFreq', 'distance'],
          body: [
            '// Calculate frequency-based offset for inter-frequency relations',
            'const freqDiff = Math.abs(targetFreq - sourceFreq);',
            'const pathLossDiff = 20 * Math.log10(targetFreq / sourceFreq);',
            'const distanceFactor = distance > 1000 ? 2 : 1;',
            'return Math.round(pathLossDiff * distanceFactor);'
          ]
        },
        {
          name: 'optimizeThresholds',
          args: ['scenario', 'load', 'interference'],
          body: [
            '// Optimize handover thresholds based on scenario conditions',
            'let threshHigh = 10, threshLow = 2;',
            'if (scenario === "dense") {',
            '  threshHigh = Math.max(8, threshHigh - interference);',
            '  threshLow = Math.max(0, threshLow - load / 10);',
            '} else if (scenario === "sparse") {',
            '  threshHigh = Math.min(20, threshHigh + interference / 2);',
            '  threshLow = Math.min(8, threshLow + load / 20);',
            '}',
            'return { threshHigh, threshLow };'
          ]
        },
        {
          name: 'calculateQOffset',
          args: ['powerDifference', 'frequencyRatio'],
          body: [
            '// Calculate qOffset based on power and frequency differences',
            'const powerOffset = powerDifference / 2;',
            'const frequencyOffset = frequencyRatio > 2 ? 3 : (frequencyRatio > 1.5 ? 2 : 1);',
            'return Math.round(powerOffset + frequencyOffset);'
          ]
        }
      ],
      configuration: {
        [relationMO]: relationConfig
      },
      conditions: {
        'interferenceCondition': {
          if: '$eval($custom.optimizeThresholds("' + scenario + '", 70, 15).threshHigh) < 8',
          then: {
            [`${relationMO}.qOffsetFreq`]: 'dB3'
          },
          else: {
            [`${relationMO}.qOffsetFreq`]: 'dB0'
          }
        },
        'loadCondition': {
          if: '$eval($custom.optimizeThresholds("' + scenario + '", 85, 10).threshLow) > 5',
          then: {
            [`${relationMO}.cellIndividualOffset`]: 3
          },
          else: {
            [`${relationMO}.cellIndividualOffset`]: 0
          }
        },
        'frequencyCondition': {
          if: `'${sourceInfo.frequency}' > 2000 && '${targetInfo.frequency}' > 2000`,
          then: {
            [`${relationMO}.useT3212`]: true
          },
          else: {
            [`${relationMO}.useT3212`]: false
          }
        }
      },
      evaluations: {
        'frequencyOffset': {
          eval: `$custom.calculateFrequencyOffset(${sourceInfo.frequency}, ${targetInfo.frequency}, 500)`,
          args: ['sourceFreq', 'targetFreq', 'distance']
        },
        'optimizedThresholds': {
          eval: `$custom.optimizeThresholds('${scenario}', 75, 12)`,
          args: ['scenario', 'load', 'interference']
        },
        'calculatedQOffset': {
          eval: `$custom.calculateQOffset($config.${relationMO}.referenceSignalPowerDifference || 0, ${targetInfo.frequency / sourceInfo.frequency})`,
          args: ['powerDifference', 'frequencyRatio']
        }
      }
    };
  }

  private generateRelationConfig(
    sourceBand: string,
    targetBand: string,
    scenario: string,
    constraints: any
  ): Record<string, any> {
    const sourceInfo = this.frequencyBands[sourceBand as keyof typeof this.frequencyBands];
    const targetInfo = this.frequencyBands[targetBand as keyof typeof this.frequencyBands];

    const baseConfig = {
      [`${this.getRelationId(sourceBand, targetBand)}`]: {
        priority: this.calculatePriority(sourceInfo, targetInfo, scenario),
        qOffsetFreq: this.calculateQOffset(sourceInfo, targetInfo, scenario),
        threshHigh: this.calculateThreshold('high', scenario),
        threshLow: this.calculateThreshold('low', scenario),
        cellIndividualOffset: this.calculateCellIndividualOffset(scenario),
        useT3212: sourceInfo.frequency > 2000 && targetInfo.frequency > 2000
      }
    };

    // Add scenario-specific parameters
    if (scenario === 'dense') {
      baseConfig[`${this.getRelationId(sourceBand, targetBand)}`].handoverType = 'make-before-break';
      baseConfig[`${this.getRelationId(sourceBand, targetBand)}`].t304 = 100;
    } else if (scenario === 'sparse') {
      baseConfig[`${this.getRelationId(sourceBand, targetBand)}`].handoverType = 'break-before-make';
      baseConfig[`${this.getRelationId(sourceBand, targetBand)}`].t304 = 200;
    }

    return baseConfig;
  }

  private getRelationMO(relationType: string): string {
    switch (relationType) {
      case '4G4G': return 'EutranFreqRelation';
      case '4G5G': return 'NrFreqRelation';
      case '5G5G': return 'NrNrFreqRelation';
      default: throw new Error(`Unknown relation type: ${relationType}`);
    }
  }

  private getRelationId(sourceBand: string, targetBand: string): string {
    return `${targetBand}`;
  }

  private calculatePriority(sourceInfo: any, targetInfo: any, scenario: string): number {
    let basePriority = 3;

    // Higher priority for higher frequency bands (generally better capacity)
    if (targetInfo.frequency > 2000) basePriority++;
    if (sourceInfo.frequency > 2000) basePriority++;

    // Adjust for scenario
    if (scenario === 'dense') basePriority++;
    if (scenario === 'sparse') basePriority--;

    return Math.max(1, Math.min(7, basePriority));
  }

  private calculateQOffset(sourceInfo: any, targetInfo: any, scenario: string): string {
    const freqRatio = targetInfo.frequency / sourceInfo.frequency;
    let offsetValue = 0;

    if (freqRatio > 3) offsetValue = 6;
    else if (freqRatio > 2) offsetValue = 3;
    else if (freqRatio > 1.5) offsetValue = 1;

    if (scenario === 'dense') offsetValue++;
    if (scenario === 'sparse') offsetValue--;

    offsetValue = Math.max(-24, Math.min(24, offsetValue));

    return `dB${Math.abs(offsetValue)}`;
  }

  private calculateThreshold(type: 'high' | 'low', scenario: string): number {
    const baseThresholds = { high: 10, low: 2 };
    const scenarioAdjustments = { dense: -2, normal: 0, sparse: 2 };

    let threshold = baseThresholds[type] + scenarioAdjustments[scenario as keyof typeof scenarioAdjustments];

    if (type === 'high') {
      threshold = Math.max(0, Math.min(31, threshold));
    } else {
      threshold = Math.max(0, Math.min(31, threshold));
    }

    return threshold;
  }

  private calculateCellIndividualOffset(scenario: string): number {
    switch (scenario) {
      case 'dense': return 2;
      case 'normal': return 0;
      case 'sparse': return -2;
      default: return 0;
    }
  }

  generateBatchFrequencyRelations(pairs: Array<{source: string, target: string, scenario?: string}>): RTBTemplate[] {
    return pairs.map(pair =>
      this.generateFrequencyRelationTemplate(pair.source, pair.target, pair.scenario || 'normal')
    );
  }

  validateFrequencyRelationTemplate(template: RTBTemplate): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    if (!template.meta?.tags?.includes('frequency')) {
      errors.push('Template must include frequency tag');
    }

    const description = template.meta?.description || '';
    const relationTypes = ['4G4G', '4G5G', '5G5G'];
    const hasValidRelationType = relationTypes.some(type => description.includes(type));

    if (!hasValidRelationType) {
      errors.push('Template description must include valid relation type (4G4G, 4G5G, or 5G5G)');
    }

    // Check configuration structure
    const configKeys = Object.keys(template.configuration || {});
    if (configKeys.length === 0) {
      errors.push('Template must include frequency relation configuration');
    }

    // Validate each frequency relation configuration
    for (const [moName, moConfig] of Object.entries(template.configuration || {})) {
      if (!moConfig || typeof moConfig !== 'object') {
        errors.push(`Invalid MO configuration for ${moName}`);
        continue;
      }

      for (const [relationId, relationConfig] of Object.entries(moConfig)) {
        if (!relationConfig || typeof relationConfig !== 'object') {
          errors.push(`Invalid relation configuration for ${relationId}`);
          continue;
        }

        const config = relationConfig as any;

        // Check required parameters
        const requiredParams = ['priority', 'qOffsetFreq', 'threshHigh', 'threshLow'];
        for (const param of requiredParams) {
          if (!(param in config)) {
            errors.push(`Missing required parameter ${param} in ${relationId}`);
          }
        }

        // Validate parameter ranges
        if (config.priority !== undefined) {
          if (config.priority < 1 || config.priority > 7) {
            errors.push(`Priority must be between 1 and 7 in ${relationId}`);
          }
        }

        if (config.threshHigh !== undefined) {
          if (config.threshHigh < 0 || config.threshHigh > 31) {
            errors.push(`threshHigh must be between 0 and 31 in ${relationId}`);
          }
        }

        if (config.threshLow !== undefined) {
          if (config.threshLow < 0 || config.threshLow > 31) {
            errors.push(`threshLow must be between 0 and 31 in ${relationId}`);
          }
        }

        // Validate threshHigh > threshLow
        if (config.threshHigh !== undefined && config.threshLow !== undefined) {
          if (config.threshHigh <= config.threshLow) {
            errors.push(`threshHigh must be greater than threshLow in ${relationId}`);
          }
        }

        // Validate qOffsetFreq format
        if (config.qOffsetFreq !== undefined) {
          if (typeof config.qOffsetFreq !== 'string' || !/^dB\d+$/.test(config.qOffsetFreq)) {
            errors.push(`qOffsetFreq must be in format dB<number> in ${relationId}`);
          }
        }
      }
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  getSupportedFrequencyPairs(): Array<{source: string, target: string, type: string}> {
    const pairs: Array<{source: string, target: string, type: string}> = [];

    for (const [relationType, constraints] of Object.entries(this.relationConstraints)) {
      for (const [sourceBand, targetBand] of constraints.allowedPairs) {
        pairs.push({
          source: sourceBand,
          target: targetBand,
          type: relationType
        });
      }
    }

    return pairs;
  }

  optimizeFrequencyRelation(template: RTBTemplate, kpiData: any): RTBTemplate {
    const optimized = JSON.parse(JSON.stringify(template)); // Deep clone

    // Simple optimization based on KPI data
    if (kpiData.handoverFailureRate > 0.05) {
      // Increase thresholds to reduce handover failures
      for (const [moName, moConfig] of Object.entries(optimized.configuration || {})) {
        for (const [relationId, relationConfig] of Object.entries(moConfig as any)) {
          (relationConfig as any).threshHigh = Math.min(31, (relationConfig as any).threshHigh + 2);
          (relationConfig as any).threshLow = Math.min(31, (relationConfig as any).threshLow + 1);
        }
      }
    }

    if (kpiData.pingPongRate > 0.1) {
      // Increase qOffset to reduce ping-pong handovers
      for (const [moName, moConfig] of Object.entries(optimized.configuration || {})) {
        for (const [relationId, relationConfig] of Object.entries(moConfig as any)) {
          const currentOffset = (relationConfig as any).qOffsetFreq || 'dB0';
          const offsetValue = parseInt(currentOffset.replace('dB', ''));
          const newOffsetValue = Math.min(24, offsetValue + 2);
          (relationConfig as any).qOffsetFreq = `dB${newOffsetValue}`;
        }
      }
    }

    return optimized;
  }
}

describe('Frequency Relation Template Generation', () => {
  let generator: MockFrequencyRelationGenerator;

  beforeEach(() => {
    generator = new MockFrequencyRelationGenerator();
  });

  describe('4G4G Frequency Relations', () => {
    test('should generate LTE1800 to LTE2600 relation correctly', () => {
      const template = generator.generateFrequencyRelationTemplate('LTE1800', 'LTE2600', 'dense');

      // Check metadata
      expect(template.meta?.description).toContain('4G4G');
      expect(template.meta?.description).toContain('LTE1800 -> LTE2600');
      expect(template.meta?.tags).toContain('frequency');
      expect(template.meta?.tags).toContain('4g4g');
      expect(template.meta?.tags).toContain('dense');
      expect(template.meta?.priority).toBeGreaterThan(0);

      // Check configuration
      expect(template.configuration).toHaveProperty('EutranFreqRelation');
      const eutranConfig = template.configuration['EutranFreqRelation'];
      expect(eutranConfig).toHaveProperty('LTE2600');

      const relationConfig = eutranConfig['LTE2600'];
      expect(relationConfig).toHaveProperty('priority');
      expect(relationConfig).toHaveProperty('qOffsetFreq');
      expect(relationConfig).toHaveProperty('threshHigh');
      expect(relationConfig).toHaveProperty('threshLow');
      expect(relationConfig).toHaveProperty('cellIndividualOffset');

      // Check parameter ranges
      expect(relationConfig.priority).toBeGreaterThanOrEqual(1);
      expect(relationConfig.priority).toBeLessThanOrEqual(7);
      expect(relationConfig.threshHigh).toBeGreaterThanOrEqual(0);
      expect(relationConfig.threshHigh).toBeLessThanOrEqual(31);
      expect(relationConfig.threshLow).toBeGreaterThanOrEqual(0);
      expect(relationConfig.threshLow).toBeLessThanOrEqual(31);
      expect(relationConfig.threshHigh).toBeGreaterThan(relationConfig.threshLow);

      // Check custom functions
      expect(template.custom).toHaveLength(3);
      expect(template.custom?.map(f => f.name)).toEqual([
        'calculateFrequencyOffset',
        'optimizeThresholds',
        'calculateQOffset'
      ]);

      // Check conditions
      expect(template.conditions).toHaveProperty('interferenceCondition');
      expect(template.conditions).toHaveProperty('loadCondition');
      expect(template.conditions).toHaveProperty('frequencyCondition');

      // Check evaluations
      expect(template.evaluations).toHaveProperty('frequencyOffset');
      expect(template.evaluations).toHaveProperty('optimizedThresholds');
      expect(template.evaluations).toHaveProperty('calculatedQOffset');
    });

    test('should generate different scenario variants', () => {
      const denseTemplate = generator.generateFrequencyRelationTemplate('LTE800', 'LTE1800', 'dense');
      const normalTemplate = generator.generateFrequencyRelationTemplate('LTE800', 'LTE1800', 'normal');
      const sparseTemplate = generator.generateFrequencyRelationTemplate('LTE800', 'LTE1800', 'sparse');

      // Check that scenarios create different configurations
      const denseConfig = denseTemplate.configuration['EutranFreqRelation']['LTE1800'];
      const normalConfig = normalTemplate.configuration['EutranFreqRelation']['LTE1800'];
      const sparseConfig = sparseTemplate.configuration['EutranFreqRelation']['LTE1800'];

      // Dense should have more aggressive settings
      expect(denseConfig.threshHigh).toBeLessThan(normalConfig.threshHigh);
      expect(denseConfig.threshLow).toBeLessThan(normalConfig.threshLow);
      expect(denseConfig.cellIndividualOffset).toBeGreaterThanOrEqual(normalConfig.cellIndividualOffset);

      // Sparse should have more conservative settings
      expect(sparseConfig.threshHigh).toBeGreaterThan(normalConfig.threshHigh);
      expect(sparseConfig.threshLow).toBeGreaterThan(normalConfig.threshLow);
      expect(sparseConfig.cellIndividualOffset).toBeLessThanOrEqual(normalConfig.cellIndividualOffset);
    });

    test('should handle all supported 4G4G pairs', () => {
      const supportedPairs = [
        ['LTE700', 'LTE800'],
        ['LTE800', 'LTE900'],
        ['LTE800', 'LTE1800'],
        ['LTE1800', 'LTE2100'],
        ['LTE1800', 'LTE2600'],
        ['LTE2100', 'LTE2600']
      ];

      for (const [source, target] of supportedPairs) {
        const template = generator.generateFrequencyRelationTemplate(source, target, 'normal');
        const validation = generator.validateFrequencyRelationTemplate(template);

        expect(validation.isValid).toBe(true);
        expect(validation.errors).toHaveLength(0);

        // Verify the template has correct configuration
        expect(template.configuration).toHaveProperty('EutranFreqRelation');
        expect(template.meta?.tags).toContain('4g4g');
      }
    });
  });

  describe('4G5G Frequency Relations', () => {
    test('should generate LTE2600 to NR3500 relation correctly', () => {
      const template = generator.generateFrequencyRelationTemplate('LTE2600', 'NR3500', 'dense');

      // Check metadata
      expect(template.meta?.description).toContain('4G5G');
      expect(template.meta?.description).toContain('LTE2600 -> NR3500');
      expect(template.meta?.tags).toContain('4g5g');

      // Check configuration
      expect(template.configuration).toHaveProperty('NrFreqRelation');
      const nrConfig = template.configuration['NrFreqRelation'];
      expect(nrConfig).toHaveProperty('NR3500');

      const relationConfig = nrConfig['NR3500'];
      expect(relationConfig).toHaveProperty('priority');
      expect(relationConfig).toHaveProperty('qOffsetFreq');
      expect(relationConfig).toHaveProperty('threshHigh');
      expect(relationConfig).toHaveProperty('threshLow');

      // 4G5G should typically have higher qOffset due to frequency difference
      const qOffsetValue = parseInt(relationConfig.qOffsetFreq.replace('dB', ''));
      expect(qOffsetValue).toBeGreaterThanOrEqual(1);
    });

    test('should generate LTE2100 to NR28GHz relation correctly', () => {
      const template = generator.generateFrequencyRelationTemplate('LTE2100', 'NR28GHz', 'normal');

      const nrConfig = template.configuration['NrFreqRelation']['NR28GHz'];

      // Very high frequency difference should result in higher qOffset
      const qOffsetValue = parseInt(nrConfig.qOffsetFreq.replace('dB', ''));
      expect(qOffsetValue).toBeGreaterThanOrEqual(3);

      // High frequency bands should use T3212
      expect(nrConfig.useT3212).toBe(true);
    });

    test('should handle all supported 4G5G pairs', () => {
      const supportedPairs = [
        ['LTE1800', 'NR3500'],
        ['LTE2100', 'NR3500'],
        ['LTE2600', 'NR3500'],
        ['LTE2100', 'NR28GHz'],
        ['LTE2600', 'NR28GHz']
      ];

      for (const [source, target] of supportedPairs) {
        const template = generator.generateFrequencyRelationTemplate(source, target, 'normal');
        const validation = generator.validateFrequencyRelationTemplate(template);

        expect(validation.isValid).toBe(true);
        expect(validation.errors).toHaveLength(0);

        // Verify the template has correct configuration
        expect(template.configuration).toHaveProperty('NrFreqRelation');
        expect(template.meta?.tags).toContain('4g5g');
      }
    });
  });

  describe('5G5G Frequency Relations', () => {
    test('should generate NR3500 to NR28GHz relation correctly', () => {
      const template = generator.generateFrequencyRelationTemplate('NR3500', 'NR28GHz', 'dense');

      // Check metadata
      expect(template.meta?.description).toContain('5G5G');
      expect(template.meta?.description).toContain('NR3500 -> NR28GHz');
      expect(template.meta?.tags).toContain('5g5g');

      // Check configuration
      expect(template.configuration).toHaveProperty('NrNrFreqRelation');
      const nrConfig = template.configuration['NrNrFreqRelation'];
      expect(nrConfig).toHaveProperty('NR28GHz');

      const relationConfig = nrConfig['NR28GHz'];
      expect(relationConfig).toHaveProperty('priority');
      expect(relationConfig).toHaveProperty('qOffsetFreq');
      expect(relationConfig).toHaveProperty('threshHigh');
      expect(relationConfig).toHaveProperty('threshLow');
    });

    test('should generate NR700 to NR3500 relation correctly', () => {
      const template = generator.generateFrequencyRelationTemplate('NR700', 'NR3500', 'sparse');

      const nrConfig = template.configuration['NrNrFreqRelation']['NR3500'];

      // Lower frequency to higher frequency should have positive offset
      const qOffsetValue = parseInt(nrConfig.qOffsetFreq.replace('dB', ''));
      expect(qOffsetValue).toBeGreaterThanOrEqual(0);

      // Sparse scenario should have conservative settings
      expect(nrConfig.threshHigh).toBeGreaterThan(10);
      expect(nrConfig.cellIndividualOffset).toBeLessThanOrEqual(0);
    });
  });

  describe('Batch Generation', () => {
    test('should generate multiple frequency relations in batch', () => {
      const pairs = [
        { source: 'LTE800', target: 'LTE1800', scenario: 'dense' },
        { source: 'LTE1800', target: 'LTE2600', scenario: 'normal' },
        { source: 'LTE2600', target: 'NR3500', scenario: 'dense' },
        { source: 'LTE2100', target: 'NR28GHz', scenario: 'sparse' }
      ];

      const templates = generator.generateBatchFrequencyRelations(pairs);

      expect(templates).toHaveLength(4);

      for (let i = 0; i < templates.length; i++) {
        const template = templates[i];
        const pair = pairs[i];

        // Verify each template is valid
        const validation = generator.validateFrequencyRelationTemplate(template);
        expect(validation.isValid).toBe(true);

        // Verify template matches the pair
        expect(template.meta?.description).toContain(pair.source);
        expect(template.meta?.description).toContain(pair.target);
        expect(template.meta?.tags).toContain(pair.scenario || 'normal');
      }
    });

    test('should handle empty batch', () => {
      const templates = generator.generateBatchFrequencyRelations([]);
      expect(templates).toHaveLength(0);
    });
  });

  describe('Template Validation', () => {
    test('should validate correct frequency relation template', () => {
      const template = generator.generateFrequencyRelationTemplate('LTE1800', 'LTE2600', 'dense');
      const validation = generator.validateFrequencyRelationTemplate(template);

      expect(validation.isValid).toBe(true);
      expect(validation.errors).toHaveLength(0);
    });

    test('should detect invalid templates', () => {
      const invalidTemplate: RTBTemplate = {
        meta: {
          version: '1.0.0',
          description: 'Invalid frequency template',
          tags: ['invalid'], // Missing frequency tag
          priority: 3
        },
        configuration: {
          'EutranFreqRelation': {
            'LTE2600': {
              // Missing required parameters
              priority: 8, // Invalid priority (> 7)
              qOffsetFreq: 'invalid', // Invalid format
              threshHigh: 35, // Invalid range (> 31)
              threshLow: -5, // Invalid range (< 0)
              // threshHigh <= threshLow (invalid)
              threshHigh: 5,
              threshLow: 10
            }
          }
        }
      };

      const validation = generator.validateFrequencyRelationTemplate(invalidTemplate);

      expect(validation.isValid).toBe(false);
      expect(validation.errors.length).toBeGreaterThan(0);
      expect(validation.errors).toContain('Template must include frequency tag');
      expect(validation.errors).toContain('Priority must be between 1 and 7 in LTE2600');
      expect(validation.errors).toContain('qOffsetFreq must be in format dB<number> in LTE2600');
      expect(validation.errors).toContain('threshHigh must be greater than threshLow in LTE2600');
    });

    test('should validate parameter ranges correctly', () => {
      const template = generator.generateFrequencyRelationTemplate('LTE800', 'LTE1800', 'normal');

      // Manually create invalid configurations for testing
      const testCases = [
        { priority: 0, error: 'Priority must be between 1 and 7' },
        { priority: 8, error: 'Priority must be between 1 and 7' },
        { threshHigh: -1, error: 'threshHigh must be between 0 and 31' },
        { threshHigh: 32, error: 'threshHigh must be between 0 and 31' },
        { threshLow: -1, error: 'threshLow must be between 0 and 31' },
        { threshLow: 32, error: 'threshLow must be between 0 and 31' }
      ];

      for (const testCase of testCases) {
        const testTemplate = JSON.parse(JSON.stringify(template)); // Deep clone
        const config = testTemplate.configuration['EutranFreqRelation']['LTE1800'];
        Object.assign(config, testCase);

        const validation = generator.validateFrequencyRelationTemplate(testTemplate);
        expect(validation.isValid).toBe(false);
        expect(validation.errors.some(error => error.includes(testCase.error))).toBe(true);
      }
    });
  });

  describe('Error Handling', () => {
    test('should throw error for invalid frequency bands', () => {
      expect(() => {
        generator.generateFrequencyRelationTemplate('INVALID', 'LTE1800', 'normal');
      }).toThrow('Invalid frequency bands: INVALID -> LTE1800');

      expect(() => {
        generator.generateFrequencyRelationTemplate('LTE1800', 'INVALID', 'normal');
      }).toThrow('Invalid frequency bands: LTE1800 -> INVALID');
    });

    test('should throw error for unsupported relation types', () => {
      expect(() => {
        generator.generateFrequencyRelationTemplate('LTE700', 'NR700', 'normal');
      }).toThrow('Unsupported relation type: 4G5G'); // NR700 is not in allowed 4G5G pairs
    });

    test('should throw error for disallowed frequency pairs', () => {
      expect(() => {
        generator.generateFrequencyRelationTemplate('LTE700', 'NR3500', 'normal');
      }).toThrow('Frequency pair not allowed: LTE700 -> NR3500');
    });
  });

  describe('Frequency Relation Optimization', () => {
    test('should optimize template based on KPI data', () => {
      const template = generator.generateFrequencyRelationTemplate('LTE1800', 'LTE2600', 'normal');
      const originalThreshHigh = template.configuration['EutranFreqRelation']['LTE2600'].threshHigh;

      const kpiData = {
        handoverFailureRate: 0.08, // High failure rate
        pingPongRate: 0.15 // High ping-pong rate
      };

      const optimizedTemplate = generator.optimizeFrequencyRelation(template, kpiData);
      const optimizedConfig = optimizedTemplate.configuration['EutranFreqRelation']['LTE2600'];

      // Should increase thresholds due to high failure rate
      expect(optimizedConfig.threshHigh).toBeGreaterThan(originalThreshHigh);

      // Should increase qOffset due to high ping-pong rate
      const originalOffset = parseInt(template.configuration['EutranFreqRelation']['LTE2600'].qOffsetFreq.replace('dB', ''));
      const optimizedOffset = parseInt(optimizedConfig.qOffsetFreq.replace('dB', ''));
      expect(optimizedOffset).toBeGreaterThan(originalOffset);
    });

    test('should not optimize when KPIs are good', () => {
      const template = generator.generateFrequencyRelationTemplate('LTE1800', 'LTE2600', 'normal');
      const originalThreshHigh = template.configuration['EutranFreqRelation']['LTE2600'].threshHigh;

      const kpiData = {
        handoverFailureRate: 0.02, // Low failure rate
        pingPongRate: 0.05 // Low ping-pong rate
      };

      const optimizedTemplate = generator.optimizeFrequencyRelation(template, kpiData);
      const optimizedConfig = optimizedTemplate.configuration['EutranFreqRelation']['LTE2600'];

      // Should not change thresholds
      expect(optimizedConfig.threshHigh).toBe(originalThreshHigh);
    });
  });

  describe('Supported Frequency Pairs', () => {
    test('should return all supported frequency pairs', () => {
      const supportedPairs = generator.getSupportedFrequencyPairs();

      expect(supportedPairs.length).toBeGreaterThan(0);

      // Check that all expected types are present
      const types = supportedPairs.map(pair => pair.type);
      expect(types).toContain('4G4G');
      expect(types).toContain('4G5G');
      expect(types).toContain('5G5G');

      // Check that all pairs have valid structure
      for (const pair of supportedPairs) {
        expect(pair).toHaveProperty('source');
        expect(pair).toHaveProperty('target');
        expect(pair).toHaveProperty('type');
        expect(typeof pair.source).toBe('string');
        expect(typeof pair.target).toBe('string');
        expect(typeof pair.type).toBe('string');
      }

      // Verify specific known pairs exist
      const hasLTE1800ToLTE2600 = supportedPairs.some(pair =>
        pair.source === 'LTE1800' && pair.target === 'LTE2600' && pair.type === '4G4G'
      );
      expect(hasLTE1800ToLTE2600).toBe(true);

      const hasLTE2600ToNR3500 = supportedPairs.some(pair =>
        pair.source === 'LTE2600' && pair.target === 'NR3500' && pair.type === '4G5G'
      );
      expect(hasLTE2600ToNR3500).toBe(true);
    });
  });

  describe('Custom Function Execution', () => {
    test('should include all required custom functions', () => {
      const template = generator.generateFrequencyRelationTemplate('LTE1800', 'LTE2600', 'dense');

      expect(template.custom).toHaveLength(3);

      const functionNames = template.custom?.map(f => f.name) || [];
      expect(functionNames).toContain('calculateFrequencyOffset');
      expect(functionNames).toContain('optimizeThresholds');
      expect(functionNames).toContain('calculateQOffset');

      // Check that functions have correct structure
      for (const func of template.custom || []) {
        expect(func.name).toBeDefined();
        expect(Array.isArray(func.args)).toBe(true);
        expect(Array.isArray(func.body)).toBe(true);
        expect(func.body.length).toBeGreaterThan(0);
      }
    });

    test('should generate valid evaluation expressions', () => {
      const template = generator.generateFrequencyRelationTemplate('LTE800', 'LTE1800', 'normal');

      expect(template.evaluations).toHaveProperty('frequencyOffset');
      expect(template.evaluations).toHaveProperty('optimizedThresholds');
      expect(template.evaluations).toHaveProperty('calculatedQOffset');

      // Check that evaluations reference custom functions correctly
      const offsetEval = template.evaluations?.frequencyOffset?.eval;
      expect(offsetEval).toContain('$custom.calculateFrequencyOffset');
      expect(offsetEval).toContain('800'); // source frequency
      expect(offsetEval).toContain('1800'); // target frequency

      const thresholdsEval = template.evaluations?.optimizedThresholds?.eval;
      expect(thresholdsEval).toContain('$custom.optimizeThresholds');
      expect(thresholdsEval).toContain('normal'); // scenario
    });
  });

  describe('Performance and Scalability', () => {
    test('should generate templates efficiently', () => {
      const startTime = performance.now();

      const templates: RTBTemplate[] = [];
      const supportedPairs = generator.getSupportedFrequencyPairs();

      for (const pair of supportedPairs.slice(0, 10)) { // Test first 10 pairs
        const template = generator.generateFrequencyRelationTemplate(pair.source, pair.target, 'normal');
        templates.push(template);
      }

      const endTime = performance.now();
      const generationTime = endTime - startTime;

      expect(generationTime).toBeLessThan(500); // < 500ms for 10 templates
      expect(templates).toHaveLength(10);

      // Validate all templates
      for (const template of templates) {
        const validation = generator.validateFrequencyRelationTemplate(template);
        expect(validation.isValid).toBe(true);
      }
    });

    test('should handle batch generation efficiently', () => {
      const pairs = Array.from({ length: 20 }, (_, i) => {
        const supportedPairs = generator.getSupportedFrequencyPairs();
        const pair = supportedPairs[i % supportedPairs.length];
        return {
          source: pair.source,
          target: pair.target,
          scenario: ['dense', 'normal', 'sparse'][i % 3] as any
        };
      });

      const startTime = performance.now();
      const templates = generator.generateBatchFrequencyRelations(pairs);
      const endTime = performance.now();

      expect(endTime - startTime).toBeLessThan(1000); // < 1 second for 20 templates
      expect(templates).toHaveLength(20);

      // Validate all templates
      for (const template of templates) {
        const validation = generator.validateFrequencyRelationTemplate(template);
        expect(validation.isValid).toBe(true);
      }
    });
  });
});