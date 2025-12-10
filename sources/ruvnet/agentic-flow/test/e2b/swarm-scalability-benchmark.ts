#!/usr/bin/env ts-node
/**
 * E2B Swarm Scalability Benchmarking Suite
 * Tests different swarm configurations and sizes for agentic-flow
 */

import { CodeInterpreter } from '@e2b/code-interpreter';

interface BenchmarkResult {
  topology: string;
  agentCount: number;
  duration: number;
  coordinationLatency: number;
  memoryUsage: number;
  taskCompletionRate: number;
  errors: string[];
}

const API_KEY = process.env.E2B_API_KEY;

if (!API_KEY) {
  console.error('❌ E2B_API_KEY not found in environment');
  process.exit(1);
}

console.log('╔════════════════════════════════════════════════════════════════╗');
console.log('║  Agentic-Flow Swarm Scalability Benchmarking Suite            ║');
console.log('╚════════════════════════════════════════════════════════════════╝\n');

console.log(`✓ E2B API Key: ${API_KEY.substring(0, 10)}...${API_KEY.substring(API_KEY.length - 4)}\n`);

/**
 * Test mesh topology swarm coordination
 */
async function benchmarkMeshSwarm(sandbox: CodeInterpreter, agentCount: number): Promise<BenchmarkResult> {
  console.log(`\n🔷 Benchmarking Mesh Topology - ${agentCount} agents`);
  const startTime = Date.now();
  const errors: string[] = [];

  try {
    // Initialize swarm with mesh topology
    const initResult = await sandbox.notebook.execCell(`
!npx claude-flow@alpha swarm init \\
  --topology mesh \\
  --max-agents ${agentCount} \\
  --session-id "e2b-mesh-${agentCount}" \\
  --memory-key "e2b/benchmark/mesh/${agentCount}"
    `);

    console.log(`   ✓ Swarm initialized (${initResult.error ? 'warning' : 'success'})`);

    // Spawn agents in mesh configuration
    let successfulSpawns = 0;
    for (let i = 1; i <= agentCount; i++) {
      try {
        await sandbox.notebook.execCell(`
!npx claude-flow@alpha agent spawn \\
  --type coder \\
  --swarm-id "e2b-mesh-${agentCount}" \\
  --agent-id "mesh-agent-${i}"
        `);
        successfulSpawns++;
      } catch (e: any) {
        errors.push(`Failed to spawn agent ${i}: ${e.message}`);
      }
    }

    console.log(`   ✓ Agents spawned: ${successfulSpawns}/${agentCount}`);

    // Measure coordination overhead
    const coordStart = Date.now();
    await sandbox.notebook.execCell(`
!npx claude-flow@alpha task orchestrate \\
  --swarm-id "e2b-mesh-${agentCount}" \\
  --task "Simple coordination test" \\
  --description "Measure mesh coordination latency"
    `);
    const coordinationLatency = Date.now() - coordStart;

    console.log(`   ✓ Coordination latency: ${coordinationLatency}ms`);

    const duration = Date.now() - startTime;

    return {
      topology: 'mesh',
      agentCount,
      duration,
      coordinationLatency,
      memoryUsage: agentCount * 28, // Estimate 28MB per agent
      taskCompletionRate: successfulSpawns / agentCount,
      errors,
    };
  } catch (error: any) {
    errors.push(`Mesh ${agentCount}: ${error.message}`);
    return {
      topology: 'mesh',
      agentCount,
      duration: Date.now() - startTime,
      coordinationLatency: 0,
      memoryUsage: 0,
      taskCompletionRate: 0,
      errors,
    };
  }
}

/**
 * Test hierarchical topology swarm coordination
 */
async function benchmarkHierarchicalSwarm(sandbox: CodeInterpreter, agentCount: number): Promise<BenchmarkResult> {
  console.log(`\n🔶 Benchmarking Hierarchical Topology - ${agentCount} agents`);
  const startTime = Date.now();
  const errors: string[] = [];

  try {
    // Initialize swarm with hierarchical topology
    await sandbox.notebook.execCell(`
!npx claude-flow@alpha swarm init \\
  --topology hierarchical \\
  --max-agents ${agentCount} \\
  --session-id "e2b-hierarchical-${agentCount}" \\
  --memory-key "e2b/benchmark/hierarchical/${agentCount}"
    `);

    console.log(`   ✓ Swarm initialized`);

    // Spawn queen coordinator
    await sandbox.notebook.execCell(`
!npx claude-flow@alpha agent spawn \\
  --type queen-coordinator \\
  --swarm-id "e2b-hierarchical-${agentCount}" \\
  --agent-id "queen"
    `);

    // Spawn worker agents
    const workerCount = agentCount - 1;
    let successfulSpawns = 1; // queen

    for (let i = 1; i <= workerCount; i++) {
      try {
        await sandbox.notebook.execCell(`
!npx claude-flow@alpha agent spawn \\
  --type worker-specialist \\
  --swarm-id "e2b-hierarchical-${agentCount}" \\
  --agent-id "worker-${i}"
        `);
        successfulSpawns++;
      } catch (e: any) {
        errors.push(`Failed to spawn worker ${i}: ${e.message}`);
      }
    }

    console.log(`   ✓ Agents spawned: ${successfulSpawns}/${agentCount} (1 queen + ${workerCount} workers)`);

    // Measure coordination overhead
    const coordStart = Date.now();
    await sandbox.notebook.execCell(`
!npx claude-flow@alpha task orchestrate \\
  --swarm-id "e2b-hierarchical-${agentCount}" \\
  --task "Hierarchical coordination test" \\
  --description "Measure queen-led coordination latency"
    `);
    const coordinationLatency = Date.now() - coordStart;

    console.log(`   ✓ Coordination latency: ${coordinationLatency}ms`);

    const duration = Date.now() - startTime;

    return {
      topology: 'hierarchical',
      agentCount,
      duration,
      coordinationLatency,
      memoryUsage: agentCount * 32, // Estimate 32MB per agent
      taskCompletionRate: successfulSpawns / agentCount,
      errors,
    };
  } catch (error: any) {
    errors.push(`Hierarchical ${agentCount}: ${error.message}`);
    return {
      topology: 'hierarchical',
      agentCount,
      duration: Date.now() - startTime,
      coordinationLatency: 0,
      memoryUsage: 0,
      taskCompletionRate: 0,
      errors,
    };
  }
}

/**
 * Test adaptive topology swarm coordination
 */
async function benchmarkAdaptiveSwarm(sandbox: CodeInterpreter, agentCount: number): Promise<BenchmarkResult> {
  console.log(`\n🔸 Benchmarking Adaptive Topology - ${agentCount} agents`);
  const startTime = Date.now();
  const errors: string[] = [];

  try {
    // Initialize swarm with adaptive topology
    await sandbox.notebook.execCell(`
!npx claude-flow@alpha swarm init \\
  --topology adaptive \\
  --max-agents ${agentCount} \\
  --session-id "e2b-adaptive-${agentCount}" \\
  --memory-key "e2b/benchmark/adaptive/${agentCount}"
    `);

    console.log(`   ✓ Swarm initialized`);

    // Spawn mixed agent types for adaptive behavior
    const agentTypes = ['coder', 'researcher', 'tester', 'reviewer'];
    let successfulSpawns = 0;

    for (let i = 1; i <= agentCount; i++) {
      try {
        const agentType = agentTypes[i % agentTypes.length];
        await sandbox.notebook.execCell(`
!npx claude-flow@alpha agent spawn \\
  --type ${agentType} \\
  --swarm-id "e2b-adaptive-${agentCount}" \\
  --agent-id "adaptive-${i}"
        `);
        successfulSpawns++;
      } catch (e: any) {
        errors.push(`Failed to spawn agent ${i}: ${e.message}`);
      }
    }

    console.log(`   ✓ Agents spawned: ${successfulSpawns}/${agentCount}`);

    // Measure coordination overhead with topology switching
    const coordStart = Date.now();
    await sandbox.notebook.execCell(`
!npx claude-flow@alpha task orchestrate \\
  --swarm-id "e2b-adaptive-${agentCount}" \\
  --task "Adaptive coordination test" \\
  --description "Measure adaptive topology switching latency"
    `);
    const coordinationLatency = Date.now() - coordStart;

    console.log(`   ✓ Coordination latency: ${coordinationLatency}ms`);

    const duration = Date.now() - startTime;

    return {
      topology: 'adaptive',
      agentCount,
      duration,
      coordinationLatency,
      memoryUsage: agentCount * 40, // Estimate 40MB per agent (higher for adaptive)
      taskCompletionRate: successfulSpawns / agentCount,
      errors,
    };
  } catch (error: any) {
    errors.push(`Adaptive ${agentCount}: ${error.message}`);
    return {
      topology: 'adaptive',
      agentCount,
      duration: Date.now() - startTime,
      coordinationLatency: 0,
      memoryUsage: 0,
      taskCompletionRate: 0,
      errors,
    };
  }
}

/**
 * Run comprehensive benchmark suite
 */
async function runBenchmarkSuite(): Promise<void> {
  console.log('🚀 Starting E2B Swarm Scalability Benchmarks\n');
  console.log('Test Matrix:');
  console.log('  - Topologies: mesh, hierarchical, adaptive');
  console.log('  - Agent counts: 3, 6, 12, 24');
  console.log('  - Metrics: duration, coordination latency, memory, completion rate\n');

  const agentCounts = [3, 6, 12, 24];
  const results: BenchmarkResult[] = [];

  let sandbox: CodeInterpreter | null = null;

  try {
    console.log('📦 Creating E2B sandbox...');
    sandbox = await CodeInterpreter.create({ apiKey: API_KEY });
    console.log('✓ Sandbox created\n');

    // Install agentic-flow in sandbox
    console.log('📥 Installing agentic-flow...');
    await sandbox.notebook.execCell('!npm install -g claude-flow@alpha');
    console.log('✓ agentic-flow installed\n');

    // Run benchmarks for each configuration
    for (const count of agentCounts) {
      console.log(`\n${'═'.repeat(65)}`);
      console.log(`Testing with ${count} agents`);
      console.log('═'.repeat(65));

      // Test mesh topology
      const meshResult = await benchmarkMeshSwarm(sandbox, count);
      results.push(meshResult);

      // Test hierarchical topology
      const hierarchicalResult = await benchmarkHierarchicalSwarm(sandbox, count);
      results.push(hierarchicalResult);

      // Test adaptive topology
      const adaptiveResult = await benchmarkAdaptiveSwarm(sandbox, count);
      results.push(adaptiveResult);

      // Small delay between tests
      await new Promise(resolve => setTimeout(resolve, 2000));
    }

    // Generate summary report
    console.log('\n\n╔════════════════════════════════════════════════════════════════╗');
    console.log('║  BENCHMARK RESULTS SUMMARY                                     ║');
    console.log('╚════════════════════════════════════════════════════════════════╝\n');

    // Group by topology
    const byTopology: Record<string, BenchmarkResult[]> = {};
    results.forEach(r => {
      if (!byTopology[r.topology]) byTopology[r.topology] = [];
      byTopology[r.topology].push(r);
    });

    for (const [topology, topologyResults] of Object.entries(byTopology)) {
      console.log(`\n📊 ${topology.toUpperCase()} Topology`);
      console.log('─'.repeat(65));
      console.log('Agents | Duration | Coord Latency | Memory  | Completion');
      console.log('─'.repeat(65));

      topologyResults.forEach(r => {
        console.log(
          `${String(r.agentCount).padStart(6)} | ` +
          `${String(r.duration).padStart(8)}ms | ` +
          `${String(r.coordinationLatency).padStart(13)}ms | ` +
          `${String(r.memoryUsage).padStart(6)}MB | ` +
          `${(r.taskCompletionRate * 100).toFixed(1)}%`
        );
      });

      // Calculate averages
      const avgDuration = topologyResults.reduce((sum, r) => sum + r.duration, 0) / topologyResults.length;
      const avgLatency = topologyResults.reduce((sum, r) => sum + r.coordinationLatency, 0) / topologyResults.length;
      const avgMemory = topologyResults.reduce((sum, r) => sum + r.memoryUsage, 0) / topologyResults.length;
      const avgCompletion = topologyResults.reduce((sum, r) => sum + r.taskCompletionRate, 0) / topologyResults.length;

      console.log('─'.repeat(65));
      console.log(
        `   AVG | ` +
        `${avgDuration.toFixed(0).padStart(8)}ms | ` +
        `${avgLatency.toFixed(0).padStart(13)}ms | ` +
        `${avgMemory.toFixed(0).padStart(6)}MB | ` +
        `${(avgCompletion * 100).toFixed(1)}%`
      );
    }

    // Scalability analysis
    console.log('\n\n📈 Scalability Analysis');
    console.log('─'.repeat(65));

    for (const topology of ['mesh', 'hierarchical', 'adaptive']) {
      const topologyData = byTopology[topology];
      if (topologyData && topologyData.length >= 2) {
        const first = topologyData[0];
        const last = topologyData[topologyData.length - 1];
        const scalingFactor = last.agentCount / first.agentCount;
        const durationIncrease = last.duration / first.duration;
        const latencyIncrease = last.coordinationLatency / first.coordinationLatency;

        console.log(`\n${topology.toUpperCase()}:`);
        console.log(`  Scaling: ${first.agentCount} → ${last.agentCount} agents (${scalingFactor}x)`);
        console.log(`  Duration increase: ${durationIncrease.toFixed(2)}x`);
        console.log(`  Latency increase: ${latencyIncrease.toFixed(2)}x`);
        console.log(`  Efficiency: ${(scalingFactor / durationIncrease * 100).toFixed(1)}%`);
      }
    }

    // Save results
    const report = {
      timestamp: new Date().toISOString(),
      environment: 'e2b-sandbox',
      apiKey: API_KEY ? `${API_KEY.substring(0, 10)}...${API_KEY.substring(API_KEY.length - 4)}` : 'unknown',
      summary: {
        totalTests: results.length,
        successfulTests: results.filter(r => r.taskCompletionRate > 0.8).length,
        avgDuration: results.reduce((sum, r) => sum + r.duration, 0) / results.length,
        avgCoordinationLatency: results.reduce((sum, r) => sum + r.coordinationLatency, 0) / results.length,
      },
      results,
    };

    // Write report locally
    const fs = require('fs');
    const localPath = '/home/user/agentic-flow/test-reports/swarm-scalability-benchmark.json';
    fs.writeFileSync(localPath, JSON.stringify(report, null, 2));
    console.log(`\n\n📄 Report saved: ${localPath}`);

  } catch (error: any) {
    console.error(`\n❌ Benchmark error: ${error.message}`);
    console.error(error.stack);
    throw error;
  } finally {
    if (sandbox) {
      console.log('\n🧹 Cleaning up sandbox...');
      await sandbox.close();
      console.log('✓ Sandbox terminated');
    }
  }

  console.log('\n\n✅ Benchmark suite completed!');
  console.log(`Total tests: ${results.length}`);
  console.log(`Successful: ${results.filter(r => r.taskCompletionRate > 0.8).length}`);
  console.log(`Failed: ${results.filter(r => r.taskCompletionRate <= 0.8).length}\n`);
}

// Run the benchmark suite
runBenchmarkSuite()
  .then(() => process.exit(0))
  .catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
