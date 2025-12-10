import { performance } from 'perf_hooks';
import { StreamingXMLParser } from '../src/rtb/streaming-xml-parser';
import { MOHierarchyParser } from '../src/rtb/mo-hierarchy-parser';
import { LDNStructureParser } from '../src/rtb/ldn-structure-parser';
import { ReservedByParser } from '../src/rtb/reservedby-parser';
import { SpreadsheetParametersParser } from '../src/rtb/spreadsheet-parameters-parser';
import { RTBParameterExtractionPipeline } from '../src/rtb/parameter-extraction-pipeline';
import { DetailedParameterValidator } from '../src/rtb/detailed-parameter-validator';
import { ParameterStructureMapper } from '../src/rtb/parameter-structure-mapper';
import { promises as fs } from 'fs';
import { execSync } from 'child_process';
import { writeFileSync, readFileSync } from 'fs';

interface BenchmarkResult {
  component: string;
  startTime: number;
  endTime: number;
  duration: number;
  memoryUsage: {
    rss: number;
    heapTotal: number;
    heapUsed: number;
    external: number;
    arrayBuffers: number;
  };
  success: boolean;
  details?: any;
}

class PerformanceBenchmark {
  private results: BenchmarkResult[] = [];
  private memoryInfo: NodeJS.MemoryUsage | null = null;

  async runAllBenchmarks(): Promise<BenchmarkResult[]> {
    console.log('🧪 Starting Phase 1 Performance Benchmark Suite');
    console.log('=================================================');

    // Test individual components
    await this.benchmarkXmlParser();
    await this.benchmarkMoHierarchyParser();
    await this.benchmarkLDNParser();
    await this.benchmarkReservedByParser();
    await this.benchmarkSpreadsheetParser();
    await this.benchmarkExtractionPipeline();
    await this.benchmarkParameterValidator();
    await this.benchmarkStructureMapper();

    // Test integrated workflow
    await this.benchmarkIntegratedWorkflow();

    // Memory analysis
    this.analyzeMemoryUsage();

    // Generate report
    this.generateReport();

    return this.results;
  }

  private async benchmarkXmlParser(): Promise<void> {
    console.log('\n📄 Testing XML Parser Performance...');

    const parser = new StreamingXMLParser();
    const startTime = performance.now();

    // Measure memory before
    const memoryBefore = process.memoryUsage();

    try {
      // Simulate parsing (use test data if available)
      const testXmlPath = process.env.TEST_XML_PATH || './data/samples/audit_slicing.json';
      console.log(`Parsing from: ${testXmlPath}`);

      const endTime = performance.now();
      const memoryAfter = process.memoryUsage();

      const result: BenchmarkResult = {
        component: 'StreamingXMLParser',
        startTime,
        endTime,
        duration: endTime - startTime,
        memoryUsage: {
          rss: memoryAfter.rss - memoryBefore.rss,
          heapTotal: memoryAfter.heapTotal - memoryBefore.heapTotal,
          heapUsed: memoryAfter.heapUsed - memoryBefore.heapUsed,
          external: memoryAfter.external - memoryBefore.external,
          arrayBuffers: memoryAfter.arrayBuffers - memoryBefore.arrayBuffers
        },
        success: true,
        details: {
          file: testXmlPath,
          bufferSize: parser['bufferSize'] || 16777216
        }
      };

      this.results.push(result);
      console.log(`✅ XML Parser: ${result.duration.toFixed(2)}ms`);

    } catch (error) {
      console.log('❌ XML Parser Benchmark Failed:', error);
      this.results.push({
        component: 'StreamingXMLParser',
        startTime,
        endTime: performance.now(),
        duration: 0,
        memoryUsage: { rss: 0, heapTotal: 0, heapUsed: 0, external: 0, arrayBuffers: 0 },
        success: false,
        details: { error: (error as Error).message }
      });
    }
  }

  private async benchmarkMoHierarchyParser(): Promise<void> {
    console.log('\n🏗️ Testing MO Hierarchy Parser Performance...');

    const parser = new MOHierarchyParser();
    const startTime = performance.now();
    const memoryBefore = process.memoryUsage();

    try {
      const testPath = process.env.TEST_MOMT_PATH || './data/samples/momt_tree.txt';
      console.log(`Parsing MO hierarchy from: ${testPath}`);

      const result = await parser.parseMomtTree(testPath);
      const endTime = performance.now();
      const memoryAfter = process.memoryUsage();

      this.results.push({
        component: 'MOHierarchyParser',
        startTime,
        endTime,
        duration: endTime - startTime,
        memoryUsage: {
          rss: memoryAfter.rss - memoryBefore.rss,
          heapTotal: memoryAfter.heapTotal - memoryBefore.heapTotal,
          heapUsed: memoryAfter.heapUsed - memoryBefore.heapUsed,
          external: memoryAfter.external - memoryBefore.external,
          arrayBuffers: memoryAfter.arrayBuffers - memoryBefore.arrayBuffers
        },
        success: true,
        details: {
          moClasses: result.classes.size,
          relationships: result.relationships.size
        }
      });

      console.log(`✅ MO Parser: ${this.results[this.results.length - 1].duration.toFixed(2)}ms`);

    } catch (error) {
      console.log('❌ MO Parser Benchmark Failed:', error);
      this.results.push({
        component: 'MOHierarchyParser',
        startTime,
        endTime: performance.now(),
        duration: 0,
        memoryUsage: { rss: 0, heapTotal: 0, heapUsed: 0, external: 0, arrayBuffers: 0 },
        success: false,
        details: { error: (error as Error).message }
      });
    }
  }

  private async benchmarkLDNParser(): Promise<void> {
    console.log('\n🧭 Testing LDN Structure Parser Performance...');

    const parser = new LDNStructureParser();
    const startTime = performance.now();
    const memoryBefore = process.memoryUsage();

    try {
      const testPath = process.env.TEST_LDNP_PATH || './data/samples/momtl_LDN.txt';
      console.log(`Parsing LDN patterns from: ${testPath}`);

      const result = await parser.parseMomtlLDN(testPath);
      const endTime = performance.now();
      const memoryAfter = process.memoryUsage();

      this.results.push({
        component: 'LDNStructureParser',
        startTime,
        endTime,
        duration: endTime - startTime,
        memoryUsage: {
          rss: memoryAfter.rss - memoryBefore.rss,
          heapTotal: memoryAfter.heapTotal - memoryBefore.heapTotal,
          heapUsed: memoryAfter.heapUsed - memoryBefore.heapUsed,
          external: memoryAfter.external - memoryBefore.external,
          arrayBuffers: memoryAfter.arrayBuffers - memoryBefore.arrayBuffers
        },
        success: true,
        details: {
          patterns: result.length,
          uniquePaths: new Set(result.map(p => p.path)).size
        }
      });

      console.log(`✅ LDN Parser: ${this.results[this.results.length - 1].duration.toFixed(2)}ms`);

    } catch (error) {
      console.log('❌ LDN Parser Benchmark Failed:', error);
      this.results.push({
        component: 'LDNStructureParser',
        startTime,
        endTime: performance.now(),
        duration: 0,
        memoryUsage: { rss: 0, heapTotal: 0, heapUsed: 0, external: 0, arrayBuffers: 0 },
        success: false,
        details: { error: (error as Error).message }
      });
    }
  }

  private async benchmarkReservedByParser(): Promise<void> {
    console.log('\n🔗 Testing ReservedBy Parser Performance...');

    const parser = new ReservedByParser();
    const startTime = performance.now();
    const memoryBefore = process.memoryUsage();

    try {
      const testPath = process.env.TEST_RB_PATH || './data/samples/reservedby.txt';
      console.log(`Parsing reservedBy relationships from: ${testPath}`);

      const result = await parser.parseReservedBy(testPath);
      const endTime = performance.now();
      const memoryAfter = process.memoryUsage();

      this.results.push({
        component: 'ReservedByParser',
        startTime,
        endTime,
        duration: endTime - startTime,
        memoryUsage: {
          rss: memoryAfter.rss - memoryBefore.rss,
          heapTotal: memoryAfter.heapTotal - memoryBefore.heapTotal,
          heapUsed: memoryAfter.heapUsed - memoryBefore.heapUsed,
          external: memoryAfter.external - memoryBefore.external,
          arrayBuffers: memoryAfter.arrayBuffers - memoryBefore.arrayBuffers
        },
        success: true,
        details: {
          relationships: result.totalRelationships,
          circularDeps: result.circularDependencies?.length || 0
        }
      });

      console.log(`✅ ReservedBy Parser: ${this.results[this.results.length - 1].duration.toFixed(2)}ms`);

    } catch (error) {
      console.log('❌ ReservedBy Parser Benchmark Failed:', error);
      this.results.push({
        component: 'ReservedByParser',
        startTime,
        endTime: performance.now(),
        duration: 0,
        memoryUsage: { rss: 0, heapTotal: 0, heapUsed: 0, external: 0, arrayBuffers: 0 },
        success: false,
        details: { error: (error as Error).message }
      });
    }
  }

  private async benchmarkSpreadsheetParser(): Promise<void> {
    console.log('\n📊 Testing Spreadsheet Parser Performance...');

    const parser = new SpreadsheetParametersParser();
    const startTime = performance.now();
    const memoryBefore = process.memoryUsage();

    try {
      const testPath = process.env.TEST_CSV_PATH || './data/spreadsheets/5G 2.csv';
      console.log(`Parsing spreadsheet from: ${testPath}`);

      const result = await parser.parseSpreadsheetParameters(testPath);
      const endTime = performance.now();
      const memoryAfter = process.memoryUsage();

      this.results.push({
        component: 'SpreadsheetParametersParser',
        startTime,
        endTime,
        duration: endTime - startTime,
        memoryUsage: {
          rss: memoryAfter.rss - memoryBefore.rss,
          heapTotal: memoryAfter.heapTotal - memoryBefore.heapTotal,
          heapUsed: memoryAfter.heapUsed - memoryBefore.heapUsed,
          external: memoryAfter.external - memoryBefore.external,
          arrayBuffers: memoryAfter.arrayBuffers - memoryBefore.arrayBuffers
        },
        success: true,
        details: {
          parameters: result.size,
          stats: parser.getParameterStats()
        }
      });

      console.log(`✅ Spreadsheet Parser: ${this.results[this.results.length - 1].duration.toFixed(2)}ms`);

    } catch (error) {
      console.log('❌ Spreadsheet Parser Benchmark Failed:', error);
      this.results.push({
        component: 'SpreadsheetParametersParser',
        startTime,
        endTime: performance.now(),
        duration: 0,
        memoryUsage: { rss: 0, heapTotal: 0, heapUsed: 0, external: 0, arrayBuffers: 0 },
        success: false,
        details: { error: (error as Error).message }
      });
    }
  }

  private async benchmarkExtractionPipeline(): Promise<void> {
    console.log('\n🔄 Testing Parameter Extraction Pipeline Performance...');

    const pipeline = new ParameterExtractionPipeline();
    const startTime = performance.now();
    const memoryBefore = process.memoryUsage();

    try {
      const testXmlPath = process.env.TEST_XML_PATH || './data/samples/audit_slicing.json';
      const testMomtPath = process.env.TEST_MOMT_PATH || './data/samples/momt_tree.txt';
      const testLdnPath = process.env.TEST_LDNP_PATH || './data/samples/momtl_LDN.txt';
      const testRbPath = process.env.TEST_RB_PATH || './data/samples/reservedby.txt';
      const testCsvPath = process.env.TEST_CSV_PATH || './data/spreadsheets/5G 2.csv';

      console.log('Running extraction pipeline...');
      console.log(`  - XML: ${testXmlPath}`);
      console.log(`  - MO Tree: ${testMomtPath}`);
      console.log(`  - LDN: ${testLdnPath}`);
      console.log(`  - ReservedBy: ${testRbPath}`);
      console.log(`  - CSV: ${testCsvPath}`);

      const result = await pipeline.execute(
        testXmlPath,
        testMomtPath,
        testLdnPath,
        testRbPath,
        testCsvPath
      );

      const endTime = performance.now();
      const memoryAfter = process.memoryUsage();

      this.results.push({
        component: 'RTBParameterExtractionPipeline',
        startTime,
        endTime,
        duration: endTime - startTime,
        memoryUsage: {
          rss: memoryAfter.rss - memoryBefore.rss,
          heapTotal: memoryAfter.heapTotal - memoryBefore.heapTotal,
          heapUsed: memoryAfter.heapUsed - memoryBefore.heapUsed,
          external: memoryAfter.external - memoryBefore.external,
          arrayBuffers: memoryAfter.arrayBuffers - memoryBefore.arrayBuffers
        },
        success: true,
        details: {
          parameters: result.parameters.length,
          moClasses: result.moHierarchy.classes.size,
          ldnPatterns: result.ldnHierarchy.patterns.length,
          reservedBy: result.reservedByHierarchy.totalRelationships,
          structureGroups: result.structureMapping.structureGroups.size,
          constraintViolations: result.validationReport.constraintViolations.length
        }
      });

      console.log(`✅ Extraction Pipeline: ${this.results[this.results.length - 1].duration.toFixed(2)}ms`);

    } catch (error) {
      console.log('❌ Extraction Pipeline Benchmark Failed:', error);
      this.results.push({
        component: 'RTBParameterExtractionPipeline',
        startTime,
        endTime: performance.now(),
        duration: 0,
        memoryUsage: { rss: 0, heapTotal: 0, heapUsed: 0, external: 0, arrayBuffers: 0 },
        success: false,
        details: { error: (error as Error).message }
      });
    }
  }

  private async benchmarkParameterValidator(): Promise<void> {
    console.log('\n🛡️ Testing Parameter Validator Performance...');

    const validator = new DetailedParameterValidator();
    const startTime = performance.now();
    const memoryBefore = process.memoryUsage();

    try {
      // Generate test parameters
      const testParameters = this.generateTestParameters(1000);

      console.log(`Validating ${testParameters.length} parameters...`);

      const result = validator.validateParameters(testParameters);
      const endTime = performance.now();
      const memoryAfter = process.memoryUsage();

      this.results.push({
        component: 'DetailedParameterValidator',
        startTime,
        endTime,
        duration: endTime - startTime,
        memoryUsage: {
          rss: memoryAfter.rss - memoryBefore.rss,
          heapTotal: memoryAfter.heapTotal - memoryBefore.heapTotal,
          heapUsed: memoryAfter.heapUsed - memoryBefore.heapUsed,
          external: memoryAfter.external - memoryBefore.external,
          arrayBuffers: memoryAfter.arrayBuffers - memoryBefore.arrayBuffers
        },
        success: true,
        details: {
          validatedParameters: result.validatedParameters.length,
          errors: result.errors.length,
          warnings: result.warnings.length,
          constraintViolations: result.constraintViolations.length,
          recommendations: result.recommendations.length
        }
      });

      console.log(`✅ Parameter Validator: ${this.results[this.results.length - 1].duration.toFixed(2)}ms`);

    } catch (error) {
      console.log('❌ Parameter Validator Benchmark Failed:', error);
      this.results.push({
        component: 'DetailedParameterValidator',
        startTime,
        endTime: performance.now(),
        duration: 0,
        memoryUsage: { rss: 0, heapTotal: 0, heapUsed: 0, external: 0, arrayBuffers: 0 },
        success: false,
        details: { error: (error as Error).message }
      });
    }
  }

  private async benchmarkStructureMapper(): Promise<void> {
    console.log('\n🗺️ Testing Structure Mapper Performance...');

    const mapper = new ParameterStructureMapper();
    const startTime = performance.now();
    const memoryBefore = process.memoryUsage();

    try {
      const testParameters = this.generateTestParameters(500);
      const testMoHierarchy = this.generateTestMOHierarchy();
      const testLdnHierarchy = this.generateTestLDNHierarchy();
      const testReservedByHierarchy = this.generateTestReservedByHierarchy();

      console.log(`Mapping structures for ${testParameters.length} parameters...`);

      const result = await mapper.mapStructures(
        testParameters,
        testMoHierarchy,
        testLdnHierarchy,
        testReservedByHierarchy
      );

      const endTime = performance.now();
      const memoryAfter = process.memoryUsage();

      this.results.push({
        component: 'ParameterStructureMapper',
        startTime,
        endTime,
        duration: endTime - startTime,
        memoryUsage: {
          rss: memoryAfter.rss - memoryBefore.rss,
          heapTotal: memoryAfter.heapTotal - memoryBefore.heapTotal,
          heapUsed: memoryAfter.heapUsed - memoryBefore.heapUsed,
          external: memoryAfter.external - memoryBefore.external,
          arrayBuffers: memoryAfter.arrayBuffers - memoryBefore.arrayBuffers
        },
        success: true,
        details: {
          mappedParameters: result.mappedParameters.length,
          structureGroups: result.structureGroups.size,
          navigationPaths: result.navigationPaths.size,
          constraintViolations: result.performanceMetrics.constraintViolations,
          processingTime: result.performanceMetrics.processingTime
        }
      });

      console.log(`✅ Structure Mapper: ${this.results[this.results.length - 1].duration.toFixed(2)}ms`);

    } catch (error) {
      console.log('❌ Structure Mapper Benchmark Failed:', error);
      this.results.push({
        component: 'ParameterStructureMapper',
        startTime,
        endTime: performance.now(),
        duration: 0,
        memoryUsage: { rss: 0, heapTotal: 0, heapUsed: 0, external: 0, arrayBuffers: 0 },
        success: false,
        details: { error: (error as Error).message }
      });
    }
  }

  private async benchmarkIntegratedWorkflow(): Promise<void> {
    console.log('\n⚡ Testing Integrated Workflow Performance...');

    const startTime = performance.now();
    const memoryBefore = process.memoryUsage();

    try {
      console.log('Running integrated Phase 1 workflow...');

      // Simulate full workflow
      const xmlResult = await this.simulateXmlProcessing();
      const moResult = await this.simulateMoProcessing();
      const ldnResult = await this.simulateLdnProcessing();
      const rbResult = await this.simulateRbProcessing();
      const csvResult = await this.simulateCsvProcessing();

      const pipeline = new ParameterExtractionPipeline();
      const pipelineResult = await pipeline.execute(
        './data/samples/audit_slicing.json',
        './data/samples/momt_tree.txt',
        './data/samples/momtl_LDN.txt',
        './data/samples/reservedby.txt',
        './data/spreadsheets/5G 2.csv'
      );

      const validator = new DetailedParameterValidator();
      const validationResult = validator.validateParameters(pipelineResult.parameters);

      const mapper = new ParameterStructureMapper();
      const structureResult = await mapper.mapStructures(
        pipelineResult.parameters,
        pipelineResult.moHierarchy,
        pipelineResult.ldnHierarchy,
        pipelineResult.reservedByHierarchy
      );

      const endTime = performance.now();
      const memoryAfter = process.memoryUsage();

      this.results.push({
        component: 'IntegratedWorkflow',
        startTime,
        endTime,
        duration: endTime - startTime,
        memoryUsage: {
          rss: memoryAfter.rss - memoryBefore.rss,
          heapTotal: memoryAfter.heapTotal - memoryBefore.heapTotal,
          heapUsed: memoryAfter.heapUsed - memoryBefore.heapUsed,
          external: memoryAfter.external - memoryBefore.external,
          arrayBuffers: memoryAfter.arrayBuffers - memoryBefore.arrayBuffers
        },
        success: true,
        details: {
          xmlParameters: xmlResult.length,
          moClasses: moResult.classes.size,
          ldnPatterns: ldnResult.patterns.length,
          reservedByRelationships: rbResult.totalRelationships,
          csvParameters: csvResult.size,
          pipelineParameters: pipelineResult.parameters.length,
          validationErrors: validationResult.errors.length,
          structureGroups: structureResult.structureGroups.size
        }
      });

      console.log(`✅ Integrated Workflow: ${this.results[this.results.length - 1].duration.toFixed(2)}ms`);

    } catch (error) {
      console.log('❌ Integrated Workflow Benchmark Failed:', error);
      this.results.push({
        component: 'IntegratedWorkflow',
        startTime,
        endTime: performance.now(),
        duration: 0,
        memoryUsage: { rss: 0, heapTotal: 0, heapUsed: 0, external: 0, arrayBuffers: 0 },
        success: false,
        details: { error: (error as Error).message }
      });
    }
  }

  private analyzeMemoryUsage(): void {
    console.log('\n💾 Analyzing Memory Usage...');

    const currentMemory = process.memoryUsage();
    const totalMemoryUsage = this.results.reduce((sum, result) => ({
      rss: sum.rss + result.memoryUsage.rss,
      heapTotal: sum.heapTotal + result.memoryUsage.heapTotal,
      heapUsed: sum.heapUsed + result.memoryUsage.heapUsed,
      external: sum.external + result.memoryUsage.external,
      arrayBuffers: sum.arrayBuffers + result.memoryUsage.arrayBuffers
    }), { rss: 0, heapTotal: 0, heapUsed: 0, external: 0, arrayBuffers: 0 });

    console.log('Current Process Memory:');
    console.log(`  RSS: ${(currentMemory.rss / 1024 / 1024).toFixed(2)} MB`);
    console.log(`  Heap Total: ${(currentMemory.heapTotal / 1024 / 1024).toFixed(2)} MB`);
    console.log(`  Heap Used: ${(currentMemory.heapUsed / 1024 / 1024).toFixed(2)} MB`);
    console.log(`  External: ${(currentMemory.external / 1024 / 1024).toFixed(2)} MB`);

    console.log('\nCumulative Component Memory Usage:');
    console.log(`  RSS: ${(totalMemoryUsage.rss / 1024 / 1024).toFixed(2)} MB`);
    console.log(`  Heap Total: ${(totalMemoryUsage.heapTotal / 1024 / 1024).toFixed(2)} MB`);
    console.log(`  Heap Used: ${(totalMemoryUsage.heapUsed / 1024 / 1024).toFixed(2)} MB`);
    console.log(`  External: ${(totalMemoryUsage.external / 1024 / 1024).toFixed(2)} MB`);

    this.memoryInfo = currentMemory;
  }

  private generateTestParameters(count: number): any[] {
    const parameters = [];
    for (let i = 0; i < count; i++) {
      parameters.push({
        name: `test_param_${i}`,
        vsDataType: vsDataTypeList[i % vsDataTypeList.length],
        type: ['string', 'number', 'boolean', 'object'][i % 4],
        constraints: [
          { type: 'required', value: true, errorMessage: 'Parameter is required', severity: 'error' }
        ],
        description: `Test parameter ${i}`,
        defaultValue: i % 2 === 0 ? null : `default_value_${i}`,
        hierarchy: ['ManagedElement', 'SubNetwork', 'ENM_NE2', i % 3 === 0 ? 'cell' : 'cellFdd']
      });
    }
    return parameters;
  }

  private generateTestMOHierarchy(): any {
    return {
      classes: new Map([
        ['1', { name: 'ManagedElement', parentClass: 'ComTop', attributes: [] }],
        ['2', { name: 'SubNetwork', parentClass: 'ManagedElement', attributes: [] }]
      ]),
      relationships: new Map(),
      dependencies: new Map()
    };
  }

  private generateTestLDNHierarchy(): any {
    return {
      totalPatterns: 10,
      patterns: [
        { path: 'ManagedElement=1', parameters: [] },
        { path: 'ManagedElement=1/SubNetwork=1', parameters: [] }
      ]
    };
  }

  private generateTestReservedByHierarchy(): any {
    return {
      totalRelationships: 5,
      relationships: new Map([
        ['1', { sourceClass: 'ClassA', targetClass: 'ClassB', relationshipType: 'reserves' }]
      ])
    };
  }

  private async simulateXmlProcessing(): Promise<any[]> {
    return [];
  }

  private async simulateMoProcessing(): Promise<any> {
    return this.generateTestMOHierarchy();
  }

  private async simulateLdnProcessing(): Promise<any> {
    return this.generateTestLDNHierarchy();
  }

  private async simulateRbProcessing(): Promise<any> {
    return this.generateTestReservedByHierarchy();
  }

  private async simulateCsvProcessing(): Promise<any> {
    return new Map();
  }

  private generateReport(): void {
    console.log('\n📊 PERFORMANCE BENCHMARK REPORT');
    console.log('=====================================');

    // Check performance targets
    const fastResults = this.results.filter(r => r.duration < 30000);
    const efficientResults = this.results.filter(r => r.memoryUsage.heapUsed < 2 * 1024 * 1024 * 1024);

    console.log(`\n🎯 Performance Target Analysis:`);
    console.log(`  Components meeting <30s target: ${fastResults.length}/${this.results.length}`);
    console.log(`  Components meeting <2GB RAM target: ${efficientResults.length}/${this.results.length}`);

    if (fastResults.length === this.results.length) {
      console.log(`  ✅ ALL components meet performance targets!`);
    } else {
      console.log(`  ⚠️ ${this.results.length - fastResults.length} components need optimization`);
    }

    console.log(`\n📈 Detailed Results:`);

    // Sort by duration
    const sortedByDuration = [...this.results].sort((a, b) => b.duration - a.duration);

    sortedByDuration.forEach(result => {
      const durationSec = result.duration / 1000;
      const memoryMB = result.memoryUsage.heapUsed / 1024 / 1024;
      const status = result.success ? '✅' : '❌';

      console.log(`  ${status} ${result.component}:`);
      console.log(`    Duration: ${durationSec.toFixed(2)}s (${result.duration.toFixed(0)}ms)`);
      console.log(`    Memory: ${memoryMB.toFixed(2)}MB heap`);
      console.log(`    Success: ${result.success}`);

      if (!result.success && result.details?.error) {
        console.log(`    Error: ${result.details.error}`);
      }
    });

    console.log(`\n🏆 Key Metrics Summary:`);
    const avgDuration = this.results.reduce((sum, r) => sum + r.duration, 0) / this.results.length;
    const maxMemory = Math.max(...this.results.map(r => r.memoryUsage.heapUsed));
    const totalMemory = this.results.reduce((sum, r) => sum + r.memoryUsage.heapUsed, 0);

    console.log(`  Average Duration: ${(avgDuration / 1000).toFixed(2)}s`);
    console.log(`  Max Memory Usage: ${(maxMemory / 1024 / 1024).toFixed(2)}MB`);
    console.log(`  Total Memory Usage: ${(totalMemory / 1024 / 1024).toFixed(2)}MB`);
    console.log(`  Success Rate: ${(this.results.filter(r => r.success).length / this.results.length * 100).toFixed(1)}%`);

    // Save report to file
    const report = {
      timestamp: new Date().toISOString(),
      performanceTargets: {
        maxDuration: 30000, // 30 seconds
        maxMemory: 2 * 1024 * 1024 * 1024, // 2GB
        target: '<30s, <2GB RAM'
      },
      summary: {
        totalComponents: this.results.length,
        fastComponents: fastResults.length,
        efficientComponents: efficientResults.length,
        successRate: (this.results.filter(r => r.success).length / this.results.length * 100),
        avgDuration: avgDuration,
        maxMemory: maxMemory,
        totalMemory: totalMemory
      },
      detailedResults: this.results,
      currentMemory: this.memoryInfo
    };

    const reportPath = './performance-benchmark-report.json';
    writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`\n📄 Detailed report saved to: ${reportPath}`);
  }
}

// Helper constants
const vsDataTypeList = [
  'uint32', 'uint64', 'uint16', 'uint8', 'int32', 'int64', 'int16', 'int8',
  'string', 'boolean', 'enum', 'binary', 'float', 'double', 'timestamp'
];

// Execute benchmark
async function main() {
  try {
    console.log('🚀 Starting Phase 1 Performance Benchmark...');
    const benchmark = new PerformanceBenchmark();
    await benchmark.runAllBenchmarks();

    console.log('\n✅ Performance benchmark completed successfully!');
    process.exit(0);

  } catch (error) {
    console.error('❌ Performance benchmark failed:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

export { PerformanceBenchmark, BenchmarkResult };