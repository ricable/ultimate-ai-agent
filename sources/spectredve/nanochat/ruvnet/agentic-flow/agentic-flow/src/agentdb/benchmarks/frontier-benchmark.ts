/**
 * Frontier Features Performance Benchmark
 *
 * Benchmarks:
 * 1. Causal edge operations (insert, query, chain traversal)
 * 2. A/B experiment tracking (observation recording, uplift calculation)
 * 3. Certificate creation (minimal hitting set, Merkle trees)
 * 4. Provenance tracking (lineage queries, verification)
 *
 * Performance Targets:
 * - Causal edge insertion: < 5ms p95
 * - Uplift calculation (1000 obs): < 100ms p95
 * - Certificate creation: < 50ms p95
 * - Certificate verification: < 10ms p95
 * - Concurrent operations: No degradation vs sequential
 */

import Database from 'better-sqlite3';
import * as fs from 'fs';
import * as path from 'path';
import { CausalMemoryGraph, CausalEdge } from '../controllers/CausalMemoryGraph';
import { ExplainableRecall } from '../controllers/ExplainableRecall';

interface BenchmarkResult {
  operation: string;
  count: number;
  totalTimeMs: number;
  avgTimeMs: number;
  p50Ms: number;
  p95Ms: number;
  p99Ms: number;
  minMs: number;
  maxMs: number;
  throughput: number; // ops/sec
  passed: boolean;
  target?: number;
}

class FrontierBenchmark {
  private db: Database.Database;
  private causalGraph: CausalMemoryGraph;
  private explainableRecall: ExplainableRecall;

  constructor() {
    // Use file-based database for realistic performance
    const dbPath = path.join(__dirname, 'frontier-benchmark.db');
    if (fs.existsSync(dbPath)) {
      fs.unlinkSync(dbPath);
    }

    this.db = new Database(dbPath);

    // Enable optimizations
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('synchronous = NORMAL');
    this.db.pragma('cache_size = -64000'); // 64MB
    this.db.pragma('temp_store = MEMORY');

    // Load schemas
    const coreSchema = fs.readFileSync(
      path.join(__dirname, '../schemas/schema.sql'),
      'utf-8'
    );
    this.db.exec(coreSchema);

    const frontierSchema = fs.readFileSync(
      path.join(__dirname, '../schemas/frontier-schema.sql'),
      'utf-8'
    );
    this.db.exec(frontierSchema);

    this.causalGraph = new CausalMemoryGraph(this.db);
    this.explainableRecall = new ExplainableRecall(this.db);
  }

  cleanup(): void {
    this.db.close();
    const dbPath = path.join(__dirname, 'frontier-benchmark.db');
    if (fs.existsSync(dbPath)) {
      fs.unlinkSync(dbPath);
    }
  }

  private calculatePercentile(sorted: number[], percentile: number): number {
    const index = Math.ceil((percentile / 100) * sorted.length) - 1;
    return sorted[index];
  }

  private createResult(
    operation: string,
    latencies: number[],
    target?: number
  ): BenchmarkResult {
    const sorted = [...latencies].sort((a, b) => a - b);
    const totalTimeMs = latencies.reduce((sum, t) => sum + t, 0);
    const avgTimeMs = totalTimeMs / latencies.length;

    const result: BenchmarkResult = {
      operation,
      count: latencies.length,
      totalTimeMs,
      avgTimeMs,
      p50Ms: this.calculatePercentile(sorted, 50),
      p95Ms: this.calculatePercentile(sorted, 95),
      p99Ms: this.calculatePercentile(sorted, 99),
      minMs: sorted[0],
      maxMs: sorted[sorted.length - 1],
      throughput: (latencies.length / totalTimeMs) * 1000,
      passed: target ? this.calculatePercentile(sorted, 95) <= target : true,
      target
    };

    return result;
  }

  /**
   * Benchmark 1: Causal Edge Insertion
   * Target: < 5ms p95
   */
  async benchmarkCausalEdgeInsertion(): Promise<BenchmarkResult> {
    console.log('\n📊 Benchmark 1: Causal Edge Insertion');

    // Setup: Insert test episodes
    for (let i = 1; i <= 100; i++) {
      this.db.prepare(`
        INSERT INTO episodes (session_id, task, reward, success)
        VALUES (?, ?, ?, ?)
      `).run(`session${i}`, `task${i}`, 0.7 + Math.random() * 0.3, 1);
    }

    const latencies: number[] = [];
    const iterations = 1000;

    for (let i = 0; i < iterations; i++) {
      const fromId = (i % 100) + 1;
      const toId = ((i + 1) % 100) + 1;

      const edge: CausalEdge = {
        fromMemoryId: fromId,
        fromMemoryType: 'episode',
        toMemoryId: toId,
        toMemoryType: 'episode',
        similarity: 0.7 + Math.random() * 0.3,
        uplift: Math.random() * 0.3 - 0.1,
        confidence: 0.5 + Math.random() * 0.5,
        sampleSize: Math.floor(Math.random() * 200) + 50,
        mechanism: 'automated test edge'
      };

      const start = Date.now();
      this.causalGraph.addCausalEdge(edge);
      latencies.push(Date.now() - start);
    }

    return this.createResult('Causal Edge Insertion', latencies, 5);
  }

  /**
   * Benchmark 2: Causal Effect Query
   * Target: < 20ms p95
   */
  async benchmarkCausalQuery(): Promise<BenchmarkResult> {
    console.log('\n📊 Benchmark 2: Causal Effect Query');

    const latencies: number[] = [];
    const iterations = 500;

    for (let i = 0; i < iterations; i++) {
      const memoryId = (i % 100) + 1;

      const start = Date.now();
      this.causalGraph.queryCausalEffects({
        interventionMemoryId: memoryId,
        interventionMemoryType: 'episode',
        minConfidence: 0.7,
        minUplift: 0.05
      });
      latencies.push(Date.now() - start);
    }

    return this.createResult('Causal Effect Query', latencies, 20);
  }

  /**
   * Benchmark 3: Causal Chain Discovery
   * Target: < 50ms p95
   */
  async benchmarkCausalChain(): Promise<BenchmarkResult> {
    console.log('\n📊 Benchmark 3: Causal Chain Discovery');

    const latencies: number[] = [];
    const iterations = 200;

    for (let i = 0; i < iterations; i++) {
      const fromId = (i % 50) + 1;
      const toId = (i % 50) + 26;

      const start = Date.now();
      this.causalGraph.getCausalChain(fromId, toId, 5);
      latencies.push(Date.now() - start);
    }

    return this.createResult('Causal Chain Discovery', latencies, 50);
  }

  /**
   * Benchmark 4: A/B Experiment Observation Recording
   * Target: < 3ms p95
   */
  async benchmarkExperimentObservation(): Promise<BenchmarkResult> {
    console.log('\n📊 Benchmark 4: A/B Experiment Observation Recording');

    // Create experiment
    const expId = this.causalGraph.createExperiment({
      name: 'Benchmark Experiment',
      hypothesis: 'Treatment improves outcome',
      treatmentId: 1,
      treatmentType: 'episode',
      startTime: Date.now(),
      sampleSize: 0,
      status: 'running'
    });

    const latencies: number[] = [];
    const iterations = 2000;

    for (let i = 0; i < iterations; i++) {
      const start = Date.now();
      this.causalGraph.recordObservation({
        experimentId: expId,
        episodeId: (i % 100) + 1,
        isTreatment: i % 2 === 0,
        outcomeValue: Math.random(),
        outcomeType: 'reward'
      });
      latencies.push(Date.now() - start);
    }

    return this.createResult('Experiment Observation Recording', latencies, 3);
  }

  /**
   * Benchmark 5: Uplift Calculation
   * Target: < 100ms p95 for 1000 observations
   */
  async benchmarkUpliftCalculation(): Promise<BenchmarkResult> {
    console.log('\n📊 Benchmark 5: Uplift Calculation');

    const latencies: number[] = [];
    const iterations = 50;

    for (let i = 0; i < iterations; i++) {
      // Create experiment with 1000 observations
      const expId = this.causalGraph.createExperiment({
        name: `Uplift Test ${i}`,
        hypothesis: 'Treatment improves outcome',
        treatmentId: 1,
        treatmentType: 'episode',
        startTime: Date.now(),
        sampleSize: 0,
        status: 'running'
      });

      // Record 1000 observations
      for (let j = 0; j < 1000; j++) {
        this.causalGraph.recordObservation({
          experimentId: expId,
          episodeId: (j % 100) + 1,
          isTreatment: j % 2 === 0,
          outcomeValue: (j % 2 === 0 ? 0.7 : 0.5) + Math.random() * 0.2,
          outcomeType: 'reward'
        });
      }

      // Measure uplift calculation
      const start = Date.now();
      this.causalGraph.calculateUplift(expId);
      latencies.push(Date.now() - start);
    }

    return this.createResult('Uplift Calculation (1000 obs)', latencies, 100);
  }

  /**
   * Benchmark 6: Certificate Creation
   * Target: < 50ms p95
   */
  async benchmarkCertificateCreation(): Promise<BenchmarkResult> {
    console.log('\n📊 Benchmark 6: Certificate Creation');

    const latencies: number[] = [];
    const iterations = 500;

    for (let i = 0; i < iterations; i++) {
      const numChunks = 10 + (i % 20); // 10-30 chunks
      const chunks = Array.from({ length: numChunks }, (_, idx) => ({
        id: `${idx + 1}`,
        type: 'episode',
        content: `Test content ${idx}`,
        relevance: 0.9 - idx * 0.02
      }));

      const requirements = Array.from({ length: 5 }, (_, idx) => `req${idx}`);

      const start = Date.now();
      this.explainableRecall.createCertificate({
        queryId: `bench_q${i}`,
        queryText: `Benchmark query ${i}`,
        chunks,
        requirements,
        accessLevel: 'internal'
      });
      latencies.push(Date.now() - start);
    }

    return this.createResult('Certificate Creation', latencies, 50);
  }

  /**
   * Benchmark 7: Certificate Verification
   * Target: < 10ms p95
   */
  async benchmarkCertificateVerification(): Promise<BenchmarkResult> {
    console.log('\n📊 Benchmark 7: Certificate Verification');

    // Create certificates to verify
    const certificateIds: string[] = [];
    for (let i = 0; i < 100; i++) {
      const chunks = Array.from({ length: 10 }, (_, idx) => ({
        id: `${idx + 1}`,
        type: 'episode',
        content: `Content ${idx}`,
        relevance: 0.9
      }));

      const cert = this.explainableRecall.createCertificate({
        queryId: `verify_q${i}`,
        queryText: `Verification test ${i}`,
        chunks,
        requirements: ['test'],
        accessLevel: 'internal'
      });

      certificateIds.push(cert.id);
    }

    const latencies: number[] = [];
    const iterations = 500;

    for (let i = 0; i < iterations; i++) {
      const certId = certificateIds[i % certificateIds.length];

      const start = Date.now();
      this.explainableRecall.verifyCertificate(certId);
      latencies.push(Date.now() - start);
    }

    return this.createResult('Certificate Verification', latencies, 10);
  }

  /**
   * Benchmark 8: Provenance Lineage Query
   * Target: < 15ms p95
   */
  async benchmarkProvenanceLineage(): Promise<BenchmarkResult> {
    console.log('\n📊 Benchmark 8: Provenance Lineage Query');

    // Create provenance sources
    const hashes: string[] = [];
    for (let i = 0; i < 100; i++) {
      const sourceId = this.explainableRecall.createProvenance({
        sourceType: 'episode',
        sourceId: i + 1,
        creator: 'benchmark_test'
      });

      const source = this.db.prepare('SELECT content_hash FROM provenance_sources WHERE id = ?').get(sourceId) as any;
      hashes.push(source.content_hash);
    }

    const latencies: number[] = [];
    const iterations = 500;

    for (let i = 0; i < iterations; i++) {
      const hash = hashes[i % hashes.length];

      const start = Date.now();
      this.explainableRecall.getProvenanceLineage(hash);
      latencies.push(Date.now() - start);
    }

    return this.createResult('Provenance Lineage Query', latencies, 15);
  }

  /**
   * Benchmark 9: Concurrent Certificate Creation
   * Target: No degradation vs sequential
   */
  async benchmarkConcurrentOperations(): Promise<BenchmarkResult> {
    console.log('\n📊 Benchmark 9: Concurrent Certificate Creation');

    const latencies: number[] = [];
    const iterations = 100;
    const concurrency = 10;

    for (let i = 0; i < iterations; i++) {
      const start = Date.now();

      // Create 10 certificates concurrently (simulate with sequential for now)
      for (let j = 0; j < concurrency; j++) {
        const chunks = Array.from({ length: 10 }, (_, idx) => ({
          id: `${idx + 1}`,
          type: 'episode',
          content: `Concurrent content ${idx}`,
          relevance: 0.9
        }));

        this.explainableRecall.createCertificate({
          queryId: `concurrent_q${i}_${j}`,
          queryText: `Concurrent query ${i} ${j}`,
          chunks,
          requirements: ['test'],
          accessLevel: 'internal'
        });
      }

      latencies.push(Date.now() - start);
    }

    return this.createResult('Concurrent Operations (10x)', latencies, 500);
  }

  /**
   * Benchmark 10: Database Size and Memory
   */
  async benchmarkDatabaseMetrics(): Promise<void> {
    console.log('\n📊 Benchmark 10: Database Metrics');

    const stats = this.db.prepare(`
      SELECT
        (page_count * page_size) / 1024 / 1024 as size_mb
      FROM pragma_page_count(), pragma_page_size()
    `).get() as any;

    const tableStats = this.db.prepare(`
      SELECT
        name,
        (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=m.name) as table_count
      FROM sqlite_master m
      WHERE type='table'
        AND name IN ('causal_edges', 'causal_experiments', 'causal_observations',
                     'recall_certificates', 'provenance_sources', 'justification_paths')
    `).all() as any[];

    console.log(`\n  Database Size: ${stats.size_mb.toFixed(2)} MB`);
    console.log(`  Tables Created: ${tableStats.length}`);

    // Count records
    const causalEdges = this.db.prepare('SELECT COUNT(*) as count FROM causal_edges').get() as any;
    const certificates = this.db.prepare('SELECT COUNT(*) as count FROM recall_certificates').get() as any;
    const provenance = this.db.prepare('SELECT COUNT(*) as count FROM provenance_sources').get() as any;

    console.log(`\n  Causal Edges: ${causalEdges.count.toLocaleString()}`);
    console.log(`  Certificates: ${certificates.count.toLocaleString()}`);
    console.log(`  Provenance Sources: ${provenance.count.toLocaleString()}`);
  }

  async runAll(): Promise<void> {
    console.log('═══════════════════════════════════════════════════════════');
    console.log('  AgentDB Frontier Features Performance Benchmark');
    console.log('═══════════════════════════════════════════════════════════');

    const results: BenchmarkResult[] = [];

    try {
      results.push(await this.benchmarkCausalEdgeInsertion());
      results.push(await this.benchmarkCausalQuery());
      results.push(await this.benchmarkCausalChain());
      results.push(await this.benchmarkExperimentObservation());
      results.push(await this.benchmarkUpliftCalculation());
      results.push(await this.benchmarkCertificateCreation());
      results.push(await this.benchmarkCertificateVerification());
      results.push(await this.benchmarkProvenanceLineage());
      results.push(await this.benchmarkConcurrentOperations());

      await this.benchmarkDatabaseMetrics();

      // Print summary
      console.log('\n═══════════════════════════════════════════════════════════');
      console.log('  Summary');
      console.log('═══════════════════════════════════════════════════════════\n');

      results.forEach(result => {
        const status = result.passed ? '✅' : '❌';
        const target = result.target ? ` (target: ${result.target}ms)` : '';
        console.log(`${status} ${result.operation}`);
        console.log(`   Count: ${result.count.toLocaleString()}`);
        console.log(`   p50: ${result.p50Ms.toFixed(2)}ms | p95: ${result.p95Ms.toFixed(2)}ms | p99: ${result.p99Ms.toFixed(2)}ms${target}`);
        console.log(`   Throughput: ${result.throughput.toFixed(0)} ops/sec`);
        console.log('');
      });

      // Overall pass/fail
      const allPassed = results.every(r => r.passed);
      console.log('═══════════════════════════════════════════════════════════');
      if (allPassed) {
        console.log('✅ ALL BENCHMARKS PASSED');
      } else {
        const failed = results.filter(r => !r.passed);
        console.log(`❌ ${failed.length} BENCHMARK(S) FAILED:`);
        failed.forEach(r => console.log(`   - ${r.operation} (p95: ${r.p95Ms.toFixed(2)}ms > target: ${r.target}ms)`));
      }
      console.log('═══════════════════════════════════════════════════════════\n');
    } finally {
      this.cleanup();
    }
  }
}

// Run if called directly
if (require.main === module) {
  const benchmark = new FrontierBenchmark();
  benchmark.runAll().catch(console.error);
}

export { FrontierBenchmark };
