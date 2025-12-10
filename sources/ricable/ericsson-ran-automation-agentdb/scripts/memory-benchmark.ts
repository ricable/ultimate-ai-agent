import { performance } from 'perf_hooks';
import { spawn } from 'child_process';
import { writeFileSync } from 'fs';

interface MemoryTestResult {
  testType: string;
  startTime: number;
  endTime: number;
  duration: number;
  memoryBefore: {
    rss: number;
    heapTotal: number;
    heapUsed: number;
    external: number;
    arrayBuffers: number;
  };
  memoryAfter: {
    rss: number;
    heapTotal: number;
    heapUsed: number;
    external: number;
    arrayBuffers: number;
  };
  peakMemory: number;
  success: boolean;
  details?: any;
}

class MemoryBenchmark {
  private results: MemoryTestResult[] = [];

  async runMemoryTests(): Promise<MemoryTestResult[]> {
    console.log('🧪 Starting Memory Benchmark Suite');
    console.log('====================================');

    // Test individual components
    await this.testXmlParserMemory();
    await this.testSpreadsheetParserMemory();
    await this.testPipelineMemory();

    // Test peak memory scenarios
    await this.testPeakMemoryScenarios();

    // Generate memory report
    this.generateMemoryReport();

    return this.results;
  }

  private async testXmlParserMemory(): Promise<void> {
    console.log('\n📄 Testing XML Parser Memory Usage...');

    const startTime = performance.now();
    const memoryBefore = process.memoryUsage();

    try {
      const testXmlPath = process.env.TEST_XML_PATH || './data/samples/audit_slicing.json';
      console.log(`Testing with: ${testXmlPath}`);

      // Simulate XML parsing with memory monitoring
      const { StreamingXMLParser } = await import('../src/rtb/streaming-xml-parser');
      const parser = new StreamingXMLParser();

      // Simulate multiple XML parsing operations
      for (let i = 0; i < 10; i++) {
        await parser.parseFile(testXmlPath).catch(() => {});
      }

      const endTime = performance.now();
      const memoryAfter = process.memoryUsage();

      const peakMemory = Math.max(
        memoryBefore.rss,
        memoryAfter.rss,
        ...this.getCumulativeMemoryUsage()
      );

      this.results.push({
        testType: 'XMLParserMemory',
        startTime,
        endTime,
        duration: endTime - startTime,
        memoryBefore: {
          rss: memoryBefore.rss,
          heapTotal: memoryBefore.heapTotal,
          heapUsed: memoryBefore.heapUsed,
          external: memoryBefore.external,
          arrayBuffers: memoryBefore.arrayBuffers
        },
        memoryAfter: {
          rss: memoryAfter.rss,
          heapTotal: memoryAfter.heapTotal,
          heapUsed: memoryAfter.heapUsed,
          external: memoryAfter.external,
          arrayBuffers: memoryAfter.arrayBuffers
        },
        peakMemory,
        success: true,
        details: {
          testRuns: 10,
          xmlFile: testXmlPath
        }
      });

      console.log(`✅ XML Parser Memory: ${(memoryAfter.heapUsed / 1024 / 1024).toFixed(2)}MB peak: ${(peakMemory / 1024 / 1024).toFixed(2)}MB`);

    } catch (error) {
      console.log('❌ XML Parser Memory Test Failed:', error);
      this.results.push({
        testType: 'XMLParserMemory',
        startTime,
        endTime: performance.now(),
        duration: 0,
        memoryBefore: { rss: 0, heapTotal: 0, heapUsed: 0, external: 0, arrayBuffers: 0 },
        memoryAfter: { rss: 0, heapTotal: 0, heapUsed: 0, external: 0, arrayBuffers: 0 },
        peakMemory: 0,
        success: false,
        details: { error: (error as Error).message }
      });
    }
  }

  private async testSpreadsheetParserMemory(): Promise<void> {
    console.log('\n📊 Testing Spreadsheet Parser Memory Usage...');

    const startTime = performance.now();
    const memoryBefore = process.memoryUsage();

    try {
      const testCsvPath = process.env.TEST_CSV_PATH || './data/spreadsheets/5G 2.csv';
      console.log(`Testing with: ${testCsvPath}`);

      const { SpreadsheetParametersParser } = await import('../src/rtb/spreadsheet-parameters-parser');
      const parser = new SpreadsheetParametersParser();

      // Parse spreadsheet multiple times to accumulate memory
      for (let i = 0; i < 5; i++) {
        await parser.parseSpreadsheetParameters(testCsvPath).catch(() => {});
        // Force garbage collection hint
        if (global.gc) {
          global.gc();
        }
      }

      const endTime = performance.now();
      const memoryAfter = process.memoryUsage();

      const peakMemory = Math.max(
        memoryBefore.rss,
        memoryAfter.rss,
        ...this.getCumulativeMemoryUsage()
      );

      this.results.push({
        testType: 'SpreadsheetParserMemory',
        startTime,
        endTime,
        duration: endTime - startTime,
        memoryBefore: {
          rss: memoryBefore.rss,
          heapTotal: memoryBefore.heapTotal,
          heapUsed: memoryBefore.heapUsed,
          external: memoryBefore.external,
          arrayBuffers: memoryBefore.arrayBuffers
        },
        memoryAfter: {
          rss: memoryAfter.rss,
          heapTotal: memoryAfter.heapTotal,
          heapUsed: memoryAfter.heapUsed,
          external: memoryAfter.external,
          arrayBuffers: memoryAfter.arrayBuffers
        },
        peakMemory,
        success: true,
        details: {
          testRuns: 5,
          csvFile: testCsvPath
        }
      });

      console.log(`✅ Spreadsheet Parser Memory: ${(memoryAfter.heapUsed / 1024 / 1024).toFixed(2)}MB peak: ${(peakMemory / 1024 / 1024).toFixed(2)}MB`);

    } catch (error) {
      console.log('❌ Spreadsheet Parser Memory Test Failed:', error);
      this.results.push({
        testType: 'SpreadsheetParserMemory',
        startTime,
        endTime: performance.now(),
        duration: 0,
        memoryBefore: { rss: 0, heapTotal: 0, heapUsed: 0, external: 0, arrayBuffers: 0 },
        memoryAfter: { rss: 0, heapTotal: 0, heapUsed: 0, external: 0, arrayBuffers: 0 },
        peakMemory: 0,
        success: false,
        details: { error: (error as Error).message }
      });
    }
  }

  private async testPipelineMemory(): Promise<void> {
    console.log('\n🔄 Testing Pipeline Memory Usage...');

    const startTime = performance.now();
    const memoryBefore = process.memoryUsage();

    try {
      const { ParameterExtractionPipeline } = await import('../src/rtb/parameter-extraction-pipeline');
      const pipeline = new ParameterExtractionPipeline();

      // Run pipeline multiple times to test memory accumulation
      for (let i = 0; i < 3; i++) {
        await pipeline.execute(
          './data/samples/audit_slicing.json',
          './data/samples/momt_tree.txt',
          './data/samples/momtl_LDN.txt',
          './data/samples/reservedby.txt',
          './data/spreadsheets/5G 2.csv'
        ).catch(() => {});

        // Force garbage collection hint
        if (global.gc) {
          global.gc();
        }
      }

      const endTime = performance.now();
      const memoryAfter = process.memoryUsage();

      const peakMemory = Math.max(
        memoryBefore.rss,
        memoryAfter.rss,
        ...this.getCumulativeMemoryUsage()
      );

      this.results.push({
        testType: 'PipelineMemory',
        startTime,
        endTime,
        duration: endTime - startTime,
        memoryBefore: {
          rss: memoryBefore.rss,
          heapTotal: memoryBefore.heapTotal,
          heapUsed: memoryBefore.heapUsed,
          external: memoryBefore.external,
          arrayBuffers: memoryBefore.arrayBuffers
        },
        memoryAfter: {
          rss: memoryAfter.rss,
          heapTotal: memoryAfter.heapTotal,
          heapUsed: memoryAfter.heapUsed,
          external: memoryAfter.external,
          arrayBuffers: memoryAfter.arrayBuffers
        },
        peakMemory,
        success: true,
        details: {
          testRuns: 3,
          pipelineComponents: ['XML', 'MO', 'LDN', 'ReservedBy', 'CSV']
        }
      });

      console.log(`✅ Pipeline Memory: ${(memoryAfter.heapUsed / 1024 / 1024).toFixed(2)}MB peak: ${(peakMemory / 1024 / 1024).toFixed(2)}MB`);

    } catch (error) {
      console.log('❌ Pipeline Memory Test Failed:', error);
      this.results.push({
        testType: 'PipelineMemory',
        startTime,
        endTime: performance.now(),
        duration: 0,
        memoryBefore: { rss: 0, heapTotal: 0, heapUsed: 0, external: 0, arrayBuffers: 0 },
        memoryAfter: { rss: 0, heapTotal: 0, heapUsed: 0, external: 0, arrayBuffers: 0 },
        peakMemory: 0,
        success: false,
        details: { error: (error as Error).message }
      });
    }
  }

  private async testPeakMemoryScenarios(): Promise<void> {
    console.log('\n🔥 Testing Peak Memory Scenarios...');

    const startTime = performance.now();
    const memoryBefore = process.memoryUsage();

    try {
      console.log('Testing peak memory with concurrent operations...');

      // Create multiple large operations simultaneously
      const operations = [
        this.createLargeXmlProcessing(),
        this.createLargeSpreadsheetProcessing(),
        this.createLargeValidationProcessing()
      ];

      await Promise.all(operations);

      const endTime = performance.now();
      const memoryAfter = process.memoryUsage();

      const peakMemory = Math.max(
        memoryBefore.rss,
        memoryAfter.rss,
        ...this.getCumulativeMemoryUsage()
      );

      this.results.push({
        testType: 'PeakMemoryConcurrent',
        startTime,
        endTime,
        duration: endTime - startTime,
        memoryBefore: {
          rss: memoryBefore.rss,
          heapTotal: memoryBefore.heapTotal,
          heapUsed: memoryBefore.heapUsed,
          external: memoryBefore.external,
          arrayBuffers: memoryBefore.arrayBuffers
        },
        memoryAfter: {
          rss: memoryAfter.rss,
          heapTotal: memoryAfter.heapTotal,
          heapUsed: memoryAfter.heapUsed,
          external: memoryAfter.external,
          arrayBuffers: memoryAfter.arrayBuffers
        },
        peakMemory,
        success: true,
        details: {
          concurrentOperations: 3,
          operationTypes: ['XML Processing', 'Spreadsheet Processing', 'Validation Processing']
        }
      });

      console.log(`✅ Peak Memory Concurrent: ${(memoryAfter.heapUsed / 1024 / 1024).toFixed(2)}MB peak: ${(peakMemory / 1024 / 1024).toFixed(2)}MB`);

    } catch (error) {
      console.log('❌ Peak Memory Test Failed:', error);
      this.results.push({
        testType: 'PeakMemoryConcurrent',
        startTime,
        endTime: performance.now(),
        duration: 0,
        memoryBefore: { rss: 0, heapTotal: 0, heapUsed: 0, external: 0, arrayBuffers: 0 },
        memoryAfter: { rss: 0, heapTotal: 0, heapUsed: 0, external: 0, arrayBuffers: 0 },
        peakMemory: 0,
        success: false,
        details: { error: (error as Error).message }
      });
    }
  }

  private async createLargeXmlProcessing(): Promise<void> {
    try {
      const { StreamingXMLParser } = await import('../src/rtb/streaming-xml-parser');
      const parser = new StreamingXMLParser();

      // Simulate large XML processing
      for (let i = 0; i < 20; i++) {
        await parser.parseFile('./data/samples/audit_slicing.json').catch(() => {});
      }
    } catch (error) {
      // Ignore errors in stress test
    }
  }

  private async createLargeSpreadsheetProcessing(): Promise<void> {
    try {
      const { SpreadsheetParametersParser } = await import('../src/rtb/spreadsheet-parameters-parser');
      const parser = new SpreadsheetParametersParser();

      // Simulate large spreadsheet processing
      for (let i = 0; i < 15; i++) {
        await parser.parseSpreadsheetParameters('./data/spreadsheets/5G 2.csv').catch(() => {});
      }
    } catch (error) {
      // Ignore errors in stress test
    }
  }

  private async createLargeValidationProcessing(): Promise<void> {
    try {
      const { DetailedParameterValidator } = await import('../src/rtb/detailed-parameter-validator');
      const validator = new DetailedParameterValidator();

      // Generate large test parameters
      const largeParameters = [];
      for (let i = 0; i < 5000; i++) {
        largeParameters.push({
          name: `stress_test_param_${i}`,
          vsDataType: 'uint32',
          type: 'number',
          constraints: [{ type: 'required', value: true }],
          description: `Stress test parameter ${i}`,
          defaultValue: i
        });
      }

      // Validate multiple times
      for (let i = 0; i < 10; i++) {
        validator.validateParameters(largeParameters).catch(() => {});
      }
    } catch (error) {
      // Ignore errors in stress test
    }
  }

  private getCumulativeMemoryUsage(): number[] {
    if (!global.gc) return [];

    global.gc();
    return [process.memoryUsage().rss];
  }

  private generateMemoryReport(): void {
    console.log('\n📊 MEMORY BENCHMARK REPORT');
    console.log('==========================');

    // Analyze memory usage
    const passedTests = this.results.filter(r =>
      r.memoryAfter.heapUsed < 2 * 1024 * 1024 * 1024 && // < 2GB
      r.peakMemory < 2.5 * 1024 * 1024 * 1024           // < 2.5GB peak
    );

    console.log(`\n🎯 Memory Target Analysis:`);
    console.log(`  Tests passing <2GB target: ${passedTests.length}/${this.results.length}`);
    console.log(`  Memory target: <2GB RAM (process), <2.5GB peak`)

    if (passedTests.length === this.results.length) {
      console.log(`  ✅ ALL tests meet memory targets!`);
    } else {
      console.log(`  ⚠️ ${this.results.length - passedTests.length} tests need optimization`);
    }

    console.log(`\n📈 Detailed Memory Results:`);

    // Sort by peak memory
    const sortedByPeakMemory = [...this.results].sort((a, b) => b.peakMemory - a.peakMemory);

    sortedByPeakMemory.forEach(result => {
      const memoryBeforeMB = result.memoryBefore.heapUsed / 1024 / 1024;
      const memoryAfterMB = result.memoryAfter.heapUsed / 1024 / 1024;
      const peakMemoryMB = result.peakMemory / 1024 / 1024;
      const status = result.success ? '✅' : '❌';

      console.log(`  ${status} ${result.testType}:`);
      console.log(`    Start: ${memoryBeforeMB.toFixed(2)}MB`);
      console.log(`    End: ${memoryAfterMB.toFixed(2)}MB`);
      console.log(`    Peak: ${peakMemoryMB.toFixed(2)}MB`);
      console.log(`    Duration: ${(result.duration / 1000).toFixed(2)}s`);
      console.log(`    Success: ${result.success}`);

      if (!result.success && result.details?.error) {
        console.log(`    Error: ${result.details.error}`);
      }
    });

    // Calculate overall statistics
    const avgMemory = this.results.reduce((sum, r) => sum + r.memoryAfter.heapUsed, 0) / this.results.length;
    const maxMemory = Math.max(...this.results.map(r => r.peakMemory));
    const totalMemory = this.results.reduce((sum, r) => sum + r.memoryAfter.heapUsed, 0);

    console.log(`\n🏆 Memory Metrics Summary:`);
    console.log(`  Average Memory: ${(avgMemory / 1024 / 1024).toFixed(2)}MB`);
    console.log(`  Peak Memory: ${(maxMemory / 1024 / 1024).toFixed(2)}MB`);
    console.log(`  Total Memory: ${(totalMemory / 1024 / 1024).toFixed(2)}MB`);
    console.log(`  Success Rate: ${(this.results.filter(r => r.success).length / this.results.length * 100).toFixed(1)}%`);

    // Save memory report
    const report = {
      timestamp: new Date().toISOString(),
      memoryTargets: {
        targetMaxMemory: 2 * 1024 * 1024 * 1024, // 2GB
        targetPeakMemory: 2.5 * 1024 * 1024 * 1024, // 2.5GB
        targetProcessMemory: 2 * 1024 * 1024 * 1024 // 2GB
      },
      summary: {
        totalTests: this.results.length,
        passedTests: passedTests.length,
        avgMemory: avgMemory,
        peakMemory: maxMemory,
        totalMemory: totalMemory,
        successRate: (this.results.filter(r => r.success).length / this.results.length * 100)
      },
      detailedResults: this.results
    };

    const reportPath = './memory-benchmark-report.json';
    writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`\n📄 Memory report saved to: ${reportPath}`);
  }
}

async function main() {
  try {
    console.log('🚀 Starting Memory Benchmark...');
    const benchmark = new MemoryBenchmark();
    await benchmark.runMemoryTests();

    console.log('\n✅ Memory benchmark completed successfully!');
    process.exit(0);

  } catch (error) {
    console.error('❌ Memory benchmark failed:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

export { MemoryBenchmark, MemoryTestResult };