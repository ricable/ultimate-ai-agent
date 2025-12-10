/**
 * Phase 5 Test Utilities and Helpers
 * Provides common testing utilities for Pydantic Schema Generation, Validation Engine, and Production Integration tests
 */

import { jest } from '@jest/globals';
import { PerformanceObserver } from 'perf_hooks';
import type {
  XMLSchemaElement,
  ValidationRule,
  TemplateConfiguration,
  PerformanceMetrics
} from '../../src/types';

// Mock XML data for testing
export const mockXMLSchema: XMLSchemaElement = {
  name: 'vsData',
  type: 'complexType',
  attributes: [
    { name: 'class', type: 'string', required: true },
    { name: 'version', type: 'string', required: false }
  ],
  elements: [
    {
      name: 'EUtranCellFDD',
      type: 'complexType',
      attributes: [
        { name: 'id', type: 'string', required: true },
        { name: 'qRxLevMin', type: 'integer', min: -140, max: -44, default: -140 },
        { name: 'qQualMin', type: 'integer', min: -34, max: -3, default: -20 },
        { name: 'cellIndividualOffset', type: 'integer', min: 0, max: 30, default: 0 }
      ]
    }
  ]
};

// Mock validation rules
export const mockValidationRules: ValidationRule[] = [
  {
    name: 'qRxLevMinRange',
    type: 'range',
    field: 'qRxLevMin',
    constraints: { min: -140, max: -44 },
    message: 'qRxLevMin must be between -140 and -44'
  },
  {
    name: 'qQualMinRange',
    type: 'range',
    field: 'qQualMin',
    constraints: { min: -34, max: -3 },
    message: 'qQualMin must be between -34 and -3'
  },
  {
    name: 'crossParameterValidation',
    type: 'conditional',
    field: 'cellIndividualOffset',
    constraints: {
      condition: 'qRxLevMin < -130',
      action: 'cellIndividualOffset >= 3'
    },
    message: 'When qRxLevMin < -130, cellIndividualOffset must be >= 3'
  }
];

// Mock template configuration
export const mockTemplateConfig: TemplateConfiguration = {
  name: 'UrbanDenseTemplate',
  version: '1.0.0',
  priority: 50,
  variants: ['urban', 'dense', 'mobility'],
  parameters: {
    'EUtranCellFDD.qRxLevMin': -128,
    'EUtranCellFDD.qQualMin': -25,
    'EUtranCellFDD.cellIndividualOffset': 5
  },
  metadata: {
    description: 'Optimized template for urban dense environments',
    author: 'Ericsson RAN Optimization Team',
    createdAt: new Date().toISOString(),
    validated: true,
    performance: {
      expectedImprovement: '15%',
      validationScore: 0.95
    }
  }
};

// Performance measurement utilities
export class PerformanceMeasurement {
  private measurements: Map<string, number[]> = new Map();
  private observer: PerformanceObserver;

  constructor() {
    this.observer = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      entries.forEach(entry => {
        const name = entry.name;
        const duration = entry.duration;

        if (!this.measurements.has(name)) {
          this.measurements.set(name, []);
        }
        this.measurements.get(name)!.push(duration);
      });
    });
    this.observer.observe({ entryTypes: ['measure'] });
  }

  startMeasurement(name: string): void {
    performance.mark(`${name}-start`);
  }

  endMeasurement(name: string): number {
    performance.mark(`${name}-end`);
    performance.measure(name, `${name}-start`, `${name}-end`);

    const measures = this.measurements.get(name) || [];
    return measures[measures.length - 1] || 0;
  }

  getAverageTime(name: string): number {
    const times = this.measurements.get(name) || [];
    return times.length > 0 ? times.reduce((a, b) => a + b, 0) / times.length : 0;
  }

  getMetrics(): PerformanceMetrics {
    const metrics: PerformanceMetrics = {
      totalMeasurements: 0,
      averageTime: 0,
      minTime: Infinity,
      maxTime: 0,
      measurements: {}
    };

    this.measurements.forEach((times, name) => {
      const avg = times.reduce((a, b) => a + b, 0) / times.length;
      const min = Math.min(...times);
      const max = Math.max(...times);

      metrics.measurements[name] = { average: avg, min, max, count: times.length };
      metrics.totalMeasurements += times.length;
      metrics.minTime = Math.min(metrics.minTime, min);
      metrics.maxTime = Math.max(metrics.maxTime, max);
    });

    if (metrics.totalMeasurements > 0) {
      const allTimes = Array.from(this.measurements.values()).flat();
      metrics.averageTime = allTimes.reduce((a, b) => a + b, 0) / allTimes.length;
    }

    return metrics;
  }

  reset(): void {
    this.measurements.clear();
    this.observer.disconnect();
  }
}

// Test data generators
export class TestDataGenerator {
  static generateXMLVsDataTypes(count: number = 623): XMLSchemaElement[] {
    const types: XMLSchemaElement[] = [];

    for (let i = 0; i < count; i++) {
      types.push({
        name: `vsData${i}`,
        type: 'complexType',
        attributes: [
          { name: `attr${i}_1`, type: 'string', required: i % 2 === 0 },
          { name: `attr${i}_2`, type: 'integer', required: i % 3 === 0, min: -100, max: 100 }
        ],
        elements: i % 4 === 0 ? [{
          name: `nested${i}`,
          type: 'complexType',
          attributes: [
            { name: `nested_attr${i}`, type: 'boolean', required: false }
          ]
        }] : []
      });
    }

    return types;
  }

  static generateValidationRules(count: number = 100): ValidationRule[] {
    const rules: ValidationRule[] = [];
    const ruleTypes = ['range', 'conditional', 'pattern', 'required'];

    for (let i = 0; i < count; i++) {
      const type = ruleTypes[i % ruleTypes.length] as 'range' | 'conditional' | 'pattern' | 'required';

      const rule: ValidationRule = {
        name: `rule${i}`,
        type,
        field: `field${i}`,
        message: `Validation rule ${i} failed`
      };

      switch (type) {
        case 'range':
          rule.constraints = { min: -100, max: 100 };
          break;
        case 'conditional':
          rule.constraints = {
            condition: `field${i - 1} > 0`,
            action: `field${i} >= 10`
          };
          break;
        case 'pattern':
          rule.constraints = { pattern: '^[A-Z]{3}-\\d{4}$' };
          break;
        case 'required':
          rule.constraints = { required: true };
          break;
      }

      rules.push(rule);
    }

    return rules;
  }

  static generateTemplateConfigurations(count: number = 50): TemplateConfiguration[] {
    const templates: TemplateConfiguration[] = [];
    const priorities = [9, 20, 30, 40, 50, 60, 70, 80];
    const variants = ['urban', 'mobility', 'sleep', 'dense', 'rural'];

    for (let i = 0; i < count; i++) {
      templates.push({
        name: `Template${i}`,
        version: `${Math.floor(Math.random() * 5) + 1}.0.0`,
        priority: priorities[i % priorities.length],
        variants: variants.slice(0, Math.floor(Math.random() * variants.length) + 1),
        parameters: {
          [`param${i}_1`]: Math.random() * 100,
          [`param${i}_2`]: Math.random() * 200 - 100
        },
        metadata: {
          description: `Test template ${i}`,
          author: 'Test Generator',
          createdAt: new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString(),
          validated: Math.random() > 0.2,
          performance: {
            expectedImprovement: `${Math.floor(Math.random() * 30)}%`,
            validationScore: Math.random() * 0.3 + 0.7
          }
        }
      });
    }

    return templates;
  }
}

// Mock implementations
export class MockXMLParser {
  async parseSchema(content: string): Promise<XMLSchemaElement[]> {
    // Simulate parsing delay
    await new Promise(resolve => setTimeout(resolve, 10 + Math.random() * 40));
    return TestDataGenerator.generateXMLVsDataTypes(10);
  }

  async validateSchema(schema: XMLSchemaElement[]): Promise<boolean> {
    await new Promise(resolve => setTimeout(resolve, 5 + Math.random() * 15));
    return schema.length > 0;
  }
}

export class MockPydanticGenerator {
  async generateModels(schema: XMLSchemaElement[]): Promise<string[]> {
    // Simulate generation delay
    await new Promise(resolve => setTimeout(resolve, 20 + Math.random() * 80));
    return schema.map(element => this.generateModel(element));
  }

  private generateModel(element: XMLSchemaElement): string {
    return `
class ${element.name}(BaseModel):
    ${element.attributes?.map(attr =>
        `${attr.name}: ${this.mapType(attr.type)}${attr.required ? '' : ' = None'}`
    ).join('\n    ') || ''}

    class Config:
        extra = "forbid"
        schema_extra = {
            "example": {
                ${element.attributes?.map(attr =>
                    `"${attr.name}": ${this.generateExample(attr)}`
                ).join(',\n                ') || ''}
            }
        }
`;
  }

  private mapType(type: string): string {
    const typeMap: { [key: string]: string } = {
      'string': 'str',
      'integer': 'int',
      'boolean': 'bool',
      'float': 'float'
    };
    return typeMap[type] || 'Any';
  }

  private generateExample(attr: any): string {
    if (attr.type === 'string') return '"example"';
    if (attr.type === 'integer') return attr.default || Math.floor((attr.min + attr.max) / 2);
    if (attr.type === 'boolean') return 'true';
    return 'null';
  }
}

export class MockValidationEngine {
  async validateRules(data: any, rules: ValidationRule[]): Promise<{ valid: boolean; errors: string[] }> {
    await new Promise(resolve => setTimeout(resolve, 5 + Math.random() * 25));

    const errors: string[] = [];

    for (const rule of rules) {
      if (!this.validateRule(data, rule)) {
        errors.push(rule.message);
      }
    }

    return { valid: errors.length === 0, errors };
  }

  private validateRule(data: any, rule: ValidationRule): boolean {
    const value = data[rule.field];

    switch (rule.type) {
      case 'range':
        const { min, max } = rule.constraints as { min?: number; max?: number };
        return value >= (min ?? -Infinity) && value <= (max ?? Infinity);

      case 'required':
        return value !== undefined && value !== null && value !== '';

      case 'pattern':
        const pattern = new RegExp((rule.constraints as { pattern: string }).pattern);
        return pattern.test(value);

      case 'conditional':
        // Simplified conditional validation
        return Math.random() > 0.1; // 90% success rate

      default:
        return true;
    }
  }
}

// Test environment setup
export class TestEnvironment {
  private static cleanupTasks: (() => void)[] = [];

  static setup(): void {
    // Set test environment variables
    process.env.NODE_ENV = 'test';
    process.env.LOG_LEVEL = 'error';
    process.env.TEST_MODE = 'true';
  }

  static addCleanupTask(task: () => void): void {
    this.cleanupTasks.push(task);
  }

  static cleanup(): void {
    this.cleanupTasks.forEach(task => {
      try {
        task();
      } catch (error) {
        console.error('Cleanup task failed:', error);
      }
    });
    this.cleanupTasks = [];
  }

  static createTempDirectory(): string {
    const tempDir = `/tmp/test-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    this.addCleanupTask(() => {
      // In a real implementation, this would delete the directory
      console.log(`Cleaning up temp directory: ${tempDir}`);
    });
    return tempDir;
  }
}

// Integration test helpers
export class IntegrationTestHelper {
  static async setupDockerEnvironment(): Promise<string> {
    // Mock Docker setup for testing
    return 'test-container-' + Math.random().toString(36).substr(2, 9);
  }

  static async setupKubernetesEnvironment(): Promise<string> {
    // Mock Kubernetes setup for testing
    return 'test-namespace-' + Math.random().toString(36).substr(2, 9);
  }

  static async mockCICDPipeline(): Promise<{ success: boolean; artifacts: string[] }> {
    // Mock CI/CD pipeline execution
    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
    return {
      success: Math.random() > 0.1, // 90% success rate
      artifacts: ['build.log', 'test-results.xml', 'coverage-report.html']
    };
  }
}

// Performance assertion helpers
export const expectPerformance = {
  async completesWithin(
    operation: () => Promise<any> | void,
    maxTimeMs: number,
    operationName: string = 'operation'
  ): Promise<void> {
    const measurement = new PerformanceMeasurement();
    measurement.startMeasurement(operationName);

    await operation();

    const duration = measurement.endMeasurement(operationName);
    expect(duration).toBeLessThanOrEqual(maxTimeMs);
  },

  async maintainsPerformance(
    operation: () => Promise<any> | void,
    maxAverageTime: number,
    iterations: number = 10,
    operationName: string = 'operation'
  ): Promise<void> {
    const measurement = new PerformanceMeasurement();
    const times: number[] = [];

    for (let i = 0; i < iterations; i++) {
      const name = `${operationName}-${i}`;
      measurement.startMeasurement(name);
      await operation();
      times.push(measurement.endMeasurement(name));
    }

    const averageTime = times.reduce((a, b) => a + b, 0) / times.length;
    expect(averageTime).toBeLessThanOrEqual(maxAverageTime);
  }
};

// Export all utilities
export * as Helpers from './test-helpers';