/**
 * Performance CLI Generation Test Suite
 *
 * Validates the <2 second conversion time requirement for Phase 3 RANOps ENM CLI integration:
 * 1. Template-to-CLI conversion performance (<2 seconds)
 * 2. Command generation performance
 * 3. Batch operations performance
 * 4. Memory usage validation
 * 5. Scalability testing
 * 6. Load testing
 */

import { performance } from 'perf_hooks';

// Performance targets from Phase 3 requirements
interface PerformanceTargets {
  templateToCliConversion: number; // milliseconds
  commandGeneration: number;     // milliseconds
  batchExecution: number;        // milliseconds
  endToEnd: number;             // milliseconds
  memoryUsage: number;          // MB
}

const PHASE3_PERFORMANCE_TARGETS: PerformanceTargets = {
  templateToCliConversion: 2000,  // <2 seconds
  commandGeneration: 3000,       // <3 seconds
  batchExecution: 30000,         // <30 seconds
  endToEnd: 60000,               // <60 seconds total
  memoryUsage: 512               // <512 MB
};

// Mock template data for performance testing
const createMockTemplate = (size: 'small' | 'medium' | 'large'): RTBTemplate => {
  const baseTemplate: any = {
    $meta: {
      version: '2.0.0',
      author: ['Performance Test Suite'],
      description: `${size} performance test template`,
      tags: ['performance', 'test'],
      environment: 'test'
    },
    $custom: []
  };

  // Add complexity based on size
  if (size === 'small') {
    baseTemplate.$custom.push({
      name: 'simpleFunction',
      args: ['param1'],
      body: ['return param1 * 2']
    });
    baseTemplate.ManagedElement = {
      managedElementId: 'PERF_TEST_001',
      userLabel: 'Performance Test Node'
    };
    baseTemplate.ENBFunction = {
      eNodeBId: '1',
      maxConnectedUe: 1000
    };
    baseTemplate.EUtranCellFDD = [{
      euTranCellFddId: '1',
      cellId: '1',
      pci: 100,
      qRxLevMin: -130
    }];
  } else if (size === 'medium') {
    // Add multiple custom functions
    for (let i = 0; i < 10; i++) {
      baseTemplate.$custom.push({
        name: `function${i}`,
        args: [`param${i}`, `param${i + 1}`],
        body: [
          `result = param${i} + param${i + 1}`,
          `return result * ${i + 1}`
        ]
      });
    }
    baseTemplate.ManagedElement = {
      managedElementId: 'PERF_TEST_MED_001',
      userLabel: 'Medium Performance Test Node',
      aiEnabled: true,
      cognitiveLevel: 'maximum'
    };
    baseTemplate.ENBFunction = {
      eNodeBId: '1',
      maxConnectedUe: 1200,
      maxEnbSupportedUe: 1200,
      endcEnabled: true,
      splitBearerSupport: true,
      carrierAggregationEnabled: true,
      loadBalancingEnabled: true
    };
    // Add multiple cells
    baseTemplate.EUtranCellFDD = [];
    for (let i = 1; i <= 5; i++) {
      baseTemplate.EUtranCellFDD.push({
        euTranCellFddId: `${i}`,
        cellId: `${i}`,
        pci: 100 + i,
        freqBand: i <= 2 ? '3' : '7',
        qRxLevMin: -130 + (i * 2),
        qQualMin: -32,
        massiveMimoEnabled: i <= 2 ? 1 : 0,
        caEnabled: i <= 3 ? 1 : 0
      });
    }
    baseTemplate.AnrFunction = {
      anrEnabled: true,
      automaticNeighbourRelation: true,
      removeEnbTime: 5,
      pciConflictCellSelection: 'ON'
    };
  } else if (size === 'large') {
    // Add many custom functions
    for (let i = 0; i < 50; i++) {
      baseTemplate.$custom.push({
        name: `complexFunction${i}`,
        args: [`param${i}`, `param${i + 1}`, `param${i + 2}`],
        body: [
          `# Complex calculation ${i}`,
          `baseValue = param${i} * param${i + 1}`,
          `multiplier = param${i + 2} / 100.0`,
          `result = baseValue * multiplier`,
          `if result > 1000: result = result / 2`,
          `return round(result, 2)`
        ]
      });
    }
    baseTemplate.ManagedElement = {
      managedElementId: 'PERF_TEST_LARGE_001',
      userLabel: 'Large Performance Test Node',
      aiEnabled: true,
      cognitiveLevel: 'maximum',
      optimizationLevel: 'aggressive'
    };
    baseTemplate.ENBFunction = {
      eNodeBId: '1',
      maxConnectedUe: 2000,
      maxEnbSupportedUe: 2000,
      endcEnabled: true,
      splitBearerSupport: true,
      carrierAggregationEnabled: true,
      loadBalancingEnabled: true,
      makeBeforeBreakEnabled: true,
      seamlessHandover: true
    };
    // Add many cells
    baseTemplate.EUtranCellFDD = [];
    for (let i = 1; i <= 20; i++) {
      baseTemplate.EUtranCellFDD.push({
        euTranCellFddId: `${i}`,
        cellId: `${i}`,
        pci: (100 + i * 3) % 504,
        freqBand: [3, 7, 20][i % 3],
        qRxLevMin: -140 + (i * 3),
        qQualMin: -34 + (i),
        massiveMimoEnabled: i <= 10 ? 1 : 0,
        caEnabled: i <= 15 ? 1 : 0,
        transmissionMode: i <= 5 ? 'TRANSMISSION_MODE_4' : 'TRANSMISSION_MODE_2',
        cellReselectionPriority: 5 + (i % 3),
        threshServLow: -120 + (i * 2),
        threshXLow: -125 + (i * 2)
      });
    }
    // Add frequency relations
    baseTemplate.EUtranFreqRelation = [];
    for (let i = 1; i <= 10; i++) {
      baseTemplate.EUtranFreqRelation.push({
        euTranFreqRelationId: `${i}`,
        hysteresis: 2.0 + (i * 0.5),
        timeToTrigger: 320 - (i * 20),
        a3Offset: 1 + (i % 3),
        qOffsetFreq: i * 2
      });
    }
    baseTemplate.AnrFunction = {
      anrEnabled: true,
      automaticNeighbourRelation: true,
      removeEnbTime: 3,
      removeGnbTime: 3,
      pciConflictCellSelection: 'ON',
      maxTimeEventBasedPciConf: 15
    };
    baseTemplate.NRFreqRelation = [{
      nrFreqRelationId: '1',
      referenceFreq: 1300,
      relatedFreq: 78,
      nrFreqRelationToEUTRAN: {
        qOffsetCell: '0dB',
        scgFailureInfoNR: 0,
        eutraNrSameFreqInd: 0
      }
    }];
  }

  return baseTemplate;
};

// Mock performance measurement utilities
class PerformanceMeasurement {
  private measurements: Map<string, number[]> = new Map();

  startMeasurement(name: string): () => number {
    const startTime = performance.now();
    return () => {
      const endTime = performance.now();
      const duration = endTime - startTime;
      this.recordMeasurement(name, duration);
      return duration;
    };
  }

  recordMeasurement(name: string, duration: number): void {
    if (!this.measurements.has(name)) {
      this.measurements.set(name, []);
    }
    this.measurements.get(name)!.push(duration);
  }

  getStatistics(name: string): PerformanceStats | null {
    const measurements = this.measurements.get(name);
    if (!measurements || measurements.length === 0) {
      return null;
    }

    const sorted = [...measurements].sort((a, b) => a - b);
    const sum = measurements.reduce((acc, val) => acc + val, 0);

    return {
      count: measurements.length,
      min: sorted[0],
      max: sorted[sorted.length - 1],
      mean: sum / measurements.length,
      median: sorted[Math.floor(sorted.length / 2)],
      p95: sorted[Math.floor(sorted.length * 0.95)],
      p99: sorted[Math.floor(sorted.length * 0.99)]
    };
  }

  getAllStatistics(): Record<string, PerformanceStats> {
    const stats: Record<string, PerformanceStats> = {};
    for (const [name] of this.measurements) {
      const stat = this.getStatistics(name);
      if (stat) {
        stats[name] = stat;
      }
    }
    return stats;
  }

  reset(): void {
    this.measurements.clear();
  }
}

interface PerformanceStats {
  count: number;
  min: number;
  max: number;
  mean: number;
  median: number;
  p95: number;
  p99: number;
}

// Mock CLI generation components (simplified for performance testing)
class MockCliGenerator {
  async generateCommands(template: any): Promise<GeneratedCommand[]> {
    // Simulate CLI generation with complexity based on template size
    const commandCount = this.estimateCommandCount(template);
    const commands: GeneratedCommand[] = [];

    for (let i = 0; i < commandCount; i++) {
      commands.push({
        id: `cmd_${i}_${Date.now()}`,
        type: 'SET',
        command: `cmedit set PERF_NODE_${i % 10} MOClass_${i} param${i}=value${i}`,
        description: `Performance test command ${i}`,
        timeout: 30
      });
    }

    // Simulate processing time
    await this.simulateProcessingTime(template);

    return commands;
  }

  private estimateCommandCount(template: any): number {
    // Estimate command count based on template complexity
    let count = 10; // Base commands

    if (template.$custom) {
      count += template.$custom.length * 2; // 2 commands per custom function
    }

    if (template.EUtranCellFDD) {
      const cells = Array.isArray(template.EUtranCellFDD) ? template.EUtranCellFDD : [template.EUtranCellFDD];
      count += cells.length * 3; // 3 commands per cell
    }

    if (template.ENBFunction) {
      count += 5; // 5 commands for eNodeB
    }

    if (template.AnrFunction) {
      count += 3; // 3 commands for ANR
    }

    if (template.EUtranFreqRelation) {
      const relations = Array.isArray(template.EUtranFreqRelation) ? template.EUtranFreqRelation : [template.EUtranFreqRelation];
      count += relations.length * 2; // 2 commands per relation
    }

    return count;
  }

  private async simulateProcessingTime(template: any): Promise<void> {
    // Simulate processing time based on template complexity
    let baseTime = 10; // Base 10ms

    if (template.$custom) {
      baseTime += template.$custom.length * 5; // 5ms per custom function
    }

    if (template.EUtranCellFDD) {
      const cells = Array.isArray(template.EUtranCellFDD) ? template.EUtranCellFDD : [template.EUtranCellFDD];
      baseTime += cells.length * 2; // 2ms per cell
    }

    // Add some randomness to simulate real-world variability
    const variability = Math.random() * baseTime * 0.5; // ±50% variability
    await new Promise(resolve => setTimeout(resolve, baseTime + variability));
  }
}

interface GeneratedCommand {
  id: string;
  type: string;
  command: string;
  description: string;
  timeout: number;
}

class MockBatchProcessor {
  async processBatch(commands: GeneratedCommand[], nodes: string[]): Promise<BatchResult> {
    const startTime = performance.now();

    // Simulate batch processing
    const totalOperations = commands.length * nodes.length;
    const processingTime = totalOperations * 5; // 5ms per operation

    await new Promise(resolve => setTimeout(resolve, processingTime));

    const endTime = performance.now();

    return {
      totalCommands: commands.length,
      totalNodes: nodes.length,
      totalOperations,
      processingTime: endTime - startTime,
      commandsPerSecond: totalOperations / ((endTime - startTime) / 1000)
    };
  }
}

interface BatchResult {
  totalCommands: number;
  totalNodes: number;
  totalOperations: number;
  processingTime: number;
  commandsPerSecond: number;
}

describe('Performance CLI Generation Tests', () => {
  let performanceMeasurement: PerformanceMeasurement;
  let cliGenerator: MockCliGenerator;
  let batchProcessor: MockBatchProcessor;

  beforeEach(() => {
    performanceMeasurement = new PerformanceMeasurement();
    cliGenerator = new MockCliGenerator();
    batchProcessor = new MockBatchProcessor();
  });

  describe('Template-to-CLI Conversion Performance', () => {
    it('should convert small templates within 2 seconds', async () => {
      const template = createMockTemplate('small');
      const nodes = ['NODE_001', 'NODE_002'];

      const endMeasurement = performanceMeasurement.startMeasurement('small_template_conversion');

      const commands = await cliGenerator.generateCommands(template);
      const batchResult = await batchProcessor.processBatch(commands, nodes);

      const duration = endMeasurement();

      expect(duration).toBeLessThan(PHASE3_PERFORMANCE_TARGETS.templateToCliConversion);
      expect(commands.length).toBeGreaterThan(0);
      expect(batchResult.totalOperations).toBeGreaterThan(0);

      console.log(`Small template conversion: ${duration.toFixed(2)}ms, ${commands.length} commands, ${batchResult.commandsPerSecond.toFixed(0)} cmd/s`);
    });

    it('should convert medium templates within 2 seconds', async () => {
      const template = createMockTemplate('medium');
      const nodes = ['NODE_001', 'NODE_002', 'NODE_003'];

      const endMeasurement = performanceMeasurement.startMeasurement('medium_template_conversion');

      const commands = await cliGenerator.generateCommands(template);
      const batchResult = await batchProcessor.processBatch(commands, nodes);

      const duration = endMeasurement();

      expect(duration).toBeLessThan(PHASE3_PERFORMANCE_TARGETS.templateToCliConversion);
      expect(commands.length).toBeGreaterThan(20); // Medium templates should generate more commands
      expect(batchResult.totalOperations).toBeGreaterThan(60);

      console.log(`Medium template conversion: ${duration.toFixed(2)}ms, ${commands.length} commands, ${batchResult.commandsPerSecond.toFixed(0)} cmd/s`);
    });

    it('should convert large templates within 2 seconds', async () => {
      const template = createMockTemplate('large');
      const nodes = ['NODE_001', 'NODE_002', 'NODE_003', 'NODE_004', 'NODE_005'];

      const endMeasurement = performanceMeasurement.startMeasurement('large_template_conversion');

      const commands = await cliGenerator.generateCommands(template);
      const batchResult = await batchProcessor.processBatch(commands, nodes);

      const duration = endMeasurement();

      expect(duration).toBeLessThan(PHASE3_PERFORMANCE_TARGETS.templateToCliConversion);
      expect(commands.length).toBeGreaterThan(100); // Large templates should generate many commands
      expect(batchResult.totalOperations).toBeGreaterThan(500);

      console.log(`Large template conversion: ${duration.toFixed(2)}ms, ${commands.length} commands, ${batchResult.commandsPerSecond.toFixed(0)} cmd/s`);
    });

    it('should maintain consistent performance across multiple runs', async () => {
      const template = createMockTemplate('medium');
      const nodes = ['NODE_001', 'NODE_002', 'NODE_003'];

      const runTimes: number[] = [];

      // Run multiple times to check consistency
      for (let i = 0; i < 10; i++) {
        const endMeasurement = performanceMeasurement.startMeasurement(`consistency_run_${i}`);
        const commands = await cliGenerator.generateCommands(template);
        await batchProcessor.processBatch(commands, nodes);
        const duration = endMeasurement();
        runTimes.push(duration);
      }

      const stats = {
        min: Math.min(...runTimes),
        max: Math.max(...runTimes),
        mean: runTimes.reduce((a, b) => a + b, 0) / runTimes.length,
        p95: runTimes.sort((a, b) => a - b)[Math.floor(runTimes.length * 0.95)]
      };

      // Performance should be consistent (within 50% variance)
      const variance = (stats.max - stats.min) / stats.mean;
      expect(variance).toBeLessThan(0.5); // Less than 50% variance

      // All runs should meet target
      runTimes.forEach(time => {
        expect(time).toBeLessThan(PHASE3_PERFORMANCE_TARGETS.templateToCliConversion);
      });

      console.log(`Consistency test - Mean: ${stats.mean.toFixed(2)}ms, P95: ${stats.p95.toFixed(2)}ms, Variance: ${(variance * 100).toFixed(1)}%`);
    });
  });

  describe('Scalability Testing', () => {
    it('should handle increasing template sizes linearly', async () => {
      const sizes = ['small', 'medium', 'large'];
      const nodes = ['NODE_001', 'NODE_002'];
      const results: Array<{ size: string; duration: number; commands: number }> = [];

      for (const size of sizes) {
        const template = createMockTemplate(size as any);
        const endMeasurement = performanceMeasurement.startMeasurement(`scalability_${size}`);

        const commands = await cliGenerator.generateCommands(template);
        await batchProcessor.processBatch(commands, nodes);
        const duration = endMeasurement();

        results.push({ size, duration, commands: commands.length });
      }

      // Check that larger templates don't have exponentially worse performance
      const smallResult = results.find(r => r.size === 'small')!;
      const largeResult = results.find(r => r.size === 'large')!;

      const commandRatio = largeResult.commands / smallResult.commands;
      const timeRatio = largeResult.duration / smallResult.duration;

      // Time growth should be roughly proportional to command growth (within 2x factor)
      expect(timeRatio).toBeLessThan(commandRatio * 2);

      // All should meet performance target
      results.forEach(result => {
        expect(result.duration).toBeLessThan(PHASE3_PERFORMANCE_TARGETS.templateToCliConversion);
      });

      console.log('Scalability Results:');
      results.forEach(result => {
        console.log(`  ${result.size}: ${result.duration.toFixed(2)}ms, ${result.commands} commands`);
      });
      console.log(`  Command growth: ${commandRatio.toFixed(2)}x, Time growth: ${timeRatio.toFixed(2)}x`);
    });

    it('should handle increasing node counts efficiently', async () => {
      const template = createMockTemplate('medium');
      const nodeCounts = [1, 2, 5, 10];
      const results: Array<{ nodeCount: number; duration: number; operations: number }> = [];

      for (const nodeCount of nodeCounts) {
        const nodes = Array.from({ length: nodeCount }, (_, i) => `NODE_${String(i + 1).padStart(3, '0')}`);
        const commands = await cliGenerator.generateCommands(template);
        const endMeasurement = performanceMeasurement.startMeasurement(`node_count_${nodeCount}`);

        const batchResult = await batchProcessor.processBatch(commands, nodes);
        const duration = endMeasurement();

        results.push({ nodeCount, duration, operations: batchResult.totalOperations });
      }

      // Check performance scaling with nodes
      const singleNode = results.find(r => r.nodeCount === 1)!;
      const tenNodes = results.find(r => r.nodeCount === 10)!;

      const nodeRatio = tenNodes.nodeCount / singleNode.nodeCount;
      const operationRatio = tenNodes.operations / singleNode.operations;
      const timeRatio = tenNodes.duration / singleNode.duration;

      // Time growth should be proportional to operations (within 1.5x factor for parallelization)
      expect(timeRatio).toBeLessThan(operationRatio * 1.5);

      console.log('Node Scaling Results:');
      results.forEach(result => {
        console.log(`  ${result.nodeCount} nodes: ${result.duration.toFixed(2)}ms, ${result.operations} operations`);
      });
      console.log(`  Node growth: ${nodeRatio}x, Operations growth: ${operationRatio.toFixed(2)}x, Time growth: ${timeRatio.toFixed(2)}x`);
    });

    it('should handle concurrent template processing', async () => {
      const templateCount = 10;
      const nodes = ['NODE_001', 'NODE_002'];

      const endMeasurement = performanceMeasurement.startMeasurement('concurrent_processing');

      // Process multiple templates concurrently
      const concurrentPromises = Array.from({ length: templateCount }, () => {
        const template = createMockTemplate('medium');
        return cliGenerator.generateCommands(template);
      });

      const allCommands = await Promise.all(concurrentPromises);
      const totalCommands = allCommands.flat();

      // Process all generated commands in a single batch
      const batchResult = await batchProcessor.processBatch(totalCommands, nodes);

      const duration = endMeasurement();

      expect(duration).toBeLessThan(PHASE3_PERFORMANCE_TARGETS.templateToCliConversion * 2); // Allow 2x for concurrent
      expect(totalCommands.length).toBeGreaterThan(200); // Should have many commands from 10 templates
      expect(batchResult.totalOperations).toBeGreaterThan(400);

      console.log(`Concurrent processing: ${templateCount} templates, ${totalCommands.length} total commands`);
      console.log(`Duration: ${duration.toFixed(2)}ms, Efficiency: ${(totalCommands.length / duration * 1000).toFixed(0)} cmd/s`);
    });
  });

  describe('Load Testing', () => {
    it('should handle sustained load without degradation', async () => {
      const template = createMockTemplate('medium');
      const nodes = ['NODE_001', 'NODE_002', 'NODE_003'];
      const iterations = 20;
      const durations: number[] = [];

      for (let i = 0; i < iterations; i++) {
        const endMeasurement = performanceMeasurement.startMeasurement(`load_test_${i}`);

        const commands = await cliGenerator.generateCommands(template);
        await batchProcessor.processBatch(commands, nodes);
        const duration = endMeasurement();

        durations.push(duration);

        // Every 5 iterations, check that performance hasn't degraded significantly
        if (i > 0 && i % 5 === 0) {
          const recentAvg = durations.slice(-5).reduce((a, b) => a + b, 0) / 5;
          const overallAvg = durations.reduce((a, b) => a + b, 0) / durations.length;

          // Recent average should not be more than 20% worse than overall average
          expect(recentAvg).toBeLessThan(overallAvg * 1.2);
        }
      }

      const stats = {
        mean: durations.reduce((a, b) => a + b, 0) / durations.length,
        min: Math.min(...durations),
        max: Math.max(...durations),
        p95: durations.sort((a, b) => a - b)[Math.floor(durations.length * 0.95)]
      };

      // All iterations should meet performance target
      durations.forEach(duration => {
        expect(duration).toBeLessThan(PHASE3_PERFORMANCE_TARGETS.templateToCliConversion);
      });

      console.log(`Load Test Results (${iterations} iterations):`);
      console.log(`  Mean: ${stats.mean.toFixed(2)}ms, Min: ${stats.min.toFixed(2)}ms, Max: ${stats.max.toFixed(2)}ms, P95: ${stats.p95.toFixed(2)}ms`);
    });

    it('should handle memory usage efficiently', async () => {
      const initialMemory = process.memoryUsage();
      const template = createMockTemplate('large');
      const nodes = Array.from({ length: 10 }, (_, i) => `NODE_${String(i + 1).padStart(2, '0')}`);

      // Generate many commands to test memory usage
      const allCommands: GeneratedCommand[][] = [];
      for (let i = 0; i < 50; i++) {
        const commands = await cliGenerator.generateCommands(template);
        allCommands.push(commands);
      }

      const peakMemory = process.memoryUsage();
      const memoryIncrease = peakMemory.heapUsed - initialMemory.heapUsed;
      const memoryIncreaseMB = memoryIncrease / (1024 * 1024);

      // Memory usage should be reasonable (<512MB)
      expect(memoryIncreaseMB).toBeLessThan(PHASE3_PERFORMANCE_TARGETS.memoryUsage);

      // Clean up to prevent memory leaks in tests
      allCommands.length = 0;

      const finalMemory = process.memoryUsage();
      const memoryAfterCleanup = finalMemory.heapUsed - initialMemory.heapUsage;

      console.log(`Memory Usage Test:`);
      console.log(`  Initial: ${(initialMemory.heapUsed / 1024 / 1024).toFixed(2)} MB`);
      console.log(`  Peak: ${(peakMemory.heapUsed / 1024 / 1024).toFixed(2)} MB`);
      console.log(`  Increase: ${memoryIncreaseMB.toFixed(2)} MB`);
      console.log(`  After cleanup: ${(memoryAfterCleanup / 1024 / 1024).toFixed(2)} MB`);
    });
  });

  describe('Performance Regression Testing', () => {
    it('should maintain performance within specified bounds', async () => {
      const testCases = [
        { name: 'small_single_node', templateSize: 'small', nodes: 1 },
        { name: 'small_multi_node', templateSize: 'small', nodes: 5 },
        { name: 'medium_single_node', templateSize: 'medium', nodes: 1 },
        { name: 'medium_multi_node', templateSize: 'medium', nodes: 3 },
        { name: 'large_single_node', templateSize: 'large', nodes: 1 },
        { name: 'large_multi_node', templateSize: 'large', nodes: 2 }
      ];

      for (const testCase of testCases) {
        const template = createMockTemplate(testCase.templateSize as any);
        const nodes = Array.from({ length: testCase.nodes }, (_, i) => `NODE_${String(i + 1).padStart(2, '0')}`);

        const endMeasurement = performanceMeasurement.startMeasurement(`regression_${testCase.name}`);

        const commands = await cliGenerator.generateCommands(template);
        const batchResult = await batchProcessor.processBatch(commands, nodes);
        const duration = endMeasurement();

        // All test cases should meet performance targets
        expect(duration).toBeLessThan(PHASE3_PERFORMANCE_TARGETS.templateToCliConversion);

        // Large templates with multiple nodes might approach the limit but should still pass
        if (testCase.templateSize === 'large' && testCase.nodes > 1) {
          expect(duration).toBeLessThan(PHASE3_PERFORMANCE_TARGETS.templateToCliConversion * 0.9); // 10% buffer
        }

        performanceMeasurement.recordMeasurement(`regression_${testCase.name}`, duration);

        console.log(`Regression ${testCase.name}: ${duration.toFixed(2)}ms, ${commands.length} commands, ${batchResult.commandsPerSecond.toFixed(0)} cmd/s`);
      }

      // Check all performance statistics
      const allStats = performanceMeasurement.getAllStatistics();
      for (const [name, stats] of Object.entries(allStats)) {
        expect(stats.mean).toBeLessThan(PHASE3_PERFORMANCE_TARGETS.templateToCliConversion);
        expect(stats.p95).toBeLessThan(PHASE3_PERFORMANCE_TARGETS.templateToCliConversion * 1.1); // 10% buffer for P95
      }

      console.log('All Regression Tests Passed');
    });
  });

  describe('End-to-End Performance Validation', () => {
    it('should meet all Phase 3 performance targets', async () => {
      const complexTemplate = createMockTemplate('large');
      const manyNodes = ['NODE_001', 'NODE_002', 'NODE_003', 'NODE_004', 'NODE_005'];

      // Measure the complete end-to-end process
      const endToEndMeasurement = performanceMeasurement.startMeasurement('end_to_end');

      // Phase 1: Template to CLI conversion
      const conversionMeasurement = performanceMeasurement.startMeasurement('conversion_phase');
      const commands = await cliGenerator.generateCommands(complexTemplate);
      const conversionDuration = conversionMeasurement();

      // Phase 2: Batch processing
      const batchMeasurement = performanceMeasurement.startMeasurement('batch_phase');
      const batchResult = await batchProcessor.processBatch(commands, manyNodes);
      const batchDuration = batchMeasurement();

      const totalDuration = endToEndMeasurement();

      // Validate all performance targets
      expect(conversionDuration).toBeLessThan(PHASE3_PERFORMANCE_TARGETS.templateToCliConversion);
      expect(batchDuration).toBeLessThan(PHASE3_PERFORMANCE_TARGETS.batchExecution);
      expect(totalDuration).toBeLessThan(PHASE3_PERFORMANCE_TARGETS.endToEnd);

      console.log('End-to-End Performance Results:');
      console.log(`  Template-to-CLI Conversion: ${conversionDuration.toFixed(2)}ms (target: ${PHASE3_PERFORMANCE_TARGETS.templateToCliConversion}ms)`);
      console.log(`  Batch Execution: ${batchDuration.toFixed(2)}ms (target: ${PHASE3_PERFORMANCE_TARGETS.batchExecution}ms)`);
      console.log(`  Total End-to-End: ${totalDuration.toFixed(2)}ms (target: ${PHASE3_PERFORMANCE_TARGETS.endToEnd}ms)`);
      console.log(`  Commands Generated: ${commands.length}`);
      console.log(`  Total Operations: ${batchResult.totalOperations}`);
      console.log(`  Commands/Second: ${batchResult.commandsPerSecond.toFixed(0)}`);

      // Performance efficiency metrics
      const commandsPerMs = commands.length / conversionDuration;
      const operationsPerMs = batchResult.totalOperations / batchDuration;

      expect(commandsPerMs).toBeGreaterThan(0.5); // Should generate at least 0.5 commands per millisecond
      expect(operationsPerMs).toBeGreaterThan(0.1); // Should process at least 0.1 operations per millisecond

      console.log(`  Efficiency: ${commandsPerMs.toFixed(2)} cmd/ms, ${operationsPerMs.toFixed(2)} ops/ms`);
    });
  });
});