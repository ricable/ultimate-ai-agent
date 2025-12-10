/**
 * Kimi-K2 Performance Benchmarks
 * Comprehensive performance testing for Kimi-K2 integration
 */

const { describe, test, expect, beforeAll, afterAll, jest } = require('@jest/globals');
const { performance } = require('perf_hooks');
const { KimiK2Client } = require('../../../src/js/synaptic-cli/lib/kimi-k2-client');
const { SynapticMesh } = require('../../../src/js/synaptic-cli/lib/synaptic-mesh');
const os = require('os');
const fs = require('fs-extra');

describe('Kimi-K2 Performance Benchmarks', () => {
  let kimiClient;
  let mesh;
  let benchmarkResults;
  
  beforeAll(async () => {
    kimiClient = new KimiK2Client({
      provider: 'mocktest',
      model: 'kimi-k2-instruct',
      contextWindow: 128000
    });
    
    mesh = new SynapticMesh({
      nodeId: 'benchmark-node',
      port: 19080
    });
    
    benchmarkResults = {
      timestamp: Date.now(),
      systemInfo: {
        platform: os.platform(),
        cpus: os.cpus().length,
        totalMemory: Math.round(os.totalmem() / 1024 / 1024 / 1024), // GB
        nodeVersion: process.version
      },
      tests: []
    };
    
    await kimiClient.initialize();
    await mesh.initialize();
  });
  
  afterAll(async () => {
    await kimiClient.shutdown();
    await mesh.shutdown();
    
    // Save benchmark results
    await fs.writeJSON(`/tmp/kimi-k2-benchmark-${Date.now()}.json`, benchmarkResults);
  });

  describe('Response Latency Benchmarks', () => {
    test('should measure basic query response time', async () => {
      const query = "What are the key principles of distributed systems?";
      const iterations = 10;
      const latencies = [];
      
      for (let i = 0; i < iterations; i++) {
        const startTime = performance.now();
        const result = await kimiClient.query(query);
        const endTime = performance.now();
        
        const latency = endTime - startTime;
        latencies.push(latency);
        
        expect(result).toBeTruthy();
        expect(latency).toBeLessThan(5000); // 5 second max
      }
      
      const avgLatency = latencies.reduce((a, b) => a + b) / latencies.length;
      const p95Latency = latencies.sort((a, b) => a - b)[Math.floor(iterations * 0.95)];
      
      benchmarkResults.tests.push({
        test: 'basic_query_latency',
        avgLatency: Math.round(avgLatency),
        p95Latency: Math.round(p95Latency),
        iterations,
        passed: avgLatency < 3000 // 3 second average target
      });
      
      expect(avgLatency).toBeLessThan(3000);
      expect(p95Latency).toBeLessThan(5000);
    });

    test('should measure large context processing time', async () => {
      const largeContext = `
        Please analyze this comprehensive system specification:
        ${'System component details: '.repeat(2000)}
        
        Provide a detailed architectural analysis.
      `;
      
      const startTime = performance.now();
      const result = await kimiClient.query(largeContext);
      const endTime = performance.now();
      
      const processingTime = endTime - startTime;
      
      benchmarkResults.tests.push({
        test: 'large_context_processing',
        processingTime: Math.round(processingTime),
        contextSize: largeContext.length,
        passed: processingTime < 15000 // 15 second target
      });
      
      expect(result).toBeTruthy();
      expect(processingTime).toBeLessThan(15000);
    });

    test('should measure tool execution latency', async () => {
      const toolExecutions = [
        { tool: 'file_operations', operation: 'list', path: '/tmp' },
        { tool: 'shell_commands', command: 'echo "benchmark test"' },
        { tool: 'memory_operations', operation: 'store', key: 'test', value: 'data' }
      ];
      
      const executionTimes = [];
      
      for (const toolExec of toolExecutions) {
        const startTime = performance.now();
        const result = await kimiClient.executeTool(toolExec.tool, toolExec);
        const endTime = performance.now();
        
        const execTime = endTime - startTime;
        executionTimes.push({
          tool: toolExec.tool,
          time: execTime
        });
        
        expect(result.success).toBe(true);
        expect(execTime).toBeLessThan(2000); // 2 second max per tool
      }
      
      const avgToolTime = executionTimes.reduce((sum, t) => sum + t.time, 0) / executionTimes.length;
      
      benchmarkResults.tests.push({
        test: 'tool_execution_latency',
        avgToolTime: Math.round(avgToolTime),
        toolTimes: executionTimes,
        passed: avgToolTime < 1000 // 1 second average
      });
      
      expect(avgToolTime).toBeLessThan(1000);
    });
  });

  describe('Throughput Benchmarks', () => {
    test('should measure concurrent query throughput', async () => {
      const concurrencyLevels = [1, 5, 10, 20];
      const queryDuration = 30000; // 30 seconds
      const throughputResults = [];
      
      for (const concurrency of concurrencyLevels) {
        const startTime = Date.now();
        let completedQueries = 0;
        const workers = [];
        
        for (let i = 0; i < concurrency; i++) {
          workers.push((async () => {
            while (Date.now() - startTime < queryDuration) {
              try {
                await kimiClient.query(`Benchmark query ${completedQueries++}`);
              } catch (error) {
                // Count failures but continue
              }
            }
          })());
        }
        
        await Promise.all(workers);
        
        const actualDuration = Date.now() - startTime;
        const throughput = (completedQueries / actualDuration) * 1000; // queries per second
        
        throughputResults.push({
          concurrency,
          throughput: Math.round(throughput * 100) / 100,
          totalQueries: completedQueries
        });
        
        expect(throughput).toBeGreaterThan(0);
      }
      
      benchmarkResults.tests.push({
        test: 'concurrent_throughput',
        results: throughputResults,
        passed: throughputResults[0].throughput > 0.5 // At least 0.5 QPS
      });
      
      expect(throughputResults[0].throughput).toBeGreaterThan(0.5);
    });

    test('should measure mesh coordination throughput', async () => {
      const agents = Array(5).fill().map((_, i) => ({
        id: `kimi-agent-${i}`,
        type: 'kimi-k2'
      }));
      
      // Register agents
      await Promise.all(agents.map(agent => mesh.registerAgent(agent)));
      
      const coordinationTasks = Array(50).fill().map((_, i) => 
        `Coordination task ${i}: analyze system component interactions`
      );
      
      const startTime = performance.now();
      
      const results = await Promise.all(
        coordinationTasks.map(task => 
          mesh.coordinateTask(task, { agentType: 'kimi-k2' })
        )
      );
      
      const endTime = performance.now();
      const totalTime = endTime - startTime;
      const coordinationThroughput = (coordinationTasks.length / totalTime) * 1000;
      
      benchmarkResults.tests.push({
        test: 'mesh_coordination_throughput',
        throughput: Math.round(coordinationThroughput * 100) / 100,
        totalTasks: coordinationTasks.length,
        totalTime: Math.round(totalTime),
        passed: coordinationThroughput > 2 // 2 tasks per second
      });
      
      expect(results).toHaveLength(50);
      expect(coordinationThroughput).toBeGreaterThan(2);
    });
  });

  describe('Memory Usage Benchmarks', () => {
    test('should measure memory efficiency under load', async () => {
      const initialMemory = process.memoryUsage();
      
      // Simulate heavy usage
      const heavyTasks = Array(20).fill().map((_, i) => 
        kimiClient.query(`Memory test ${i}: ${'large context data '.repeat(500)}`)
      );
      
      await Promise.all(heavyTasks);
      
      const peakMemory = process.memoryUsage();
      const memoryIncrease = (peakMemory.heapUsed - initialMemory.heapUsed) / 1024 / 1024; // MB
      
      // Force garbage collection and measure final memory
      if (global.gc) {
        global.gc();
      }
      
      const finalMemory = process.memoryUsage();
      const retainedMemory = (finalMemory.heapUsed - initialMemory.heapUsed) / 1024 / 1024; // MB
      
      benchmarkResults.tests.push({
        test: 'memory_efficiency',
        memoryIncrease: Math.round(memoryIncrease),
        retainedMemory: Math.round(retainedMemory),
        passed: retainedMemory < 100 // Less than 100MB retained
      });
      
      expect(memoryIncrease).toBeLessThan(500); // Peak increase < 500MB
      expect(retainedMemory).toBeLessThan(100); // Retained < 100MB
    });

    test('should measure context window memory usage', async () => {
      const contextSizes = [1000, 10000, 50000, 100000]; // tokens
      const memoryUsages = [];
      
      for (const size of contextSizes) {
        const largeContext = 'token '.repeat(size);
        
        const beforeMemory = process.memoryUsage().heapUsed;
        await kimiClient.query(`Analyze: ${largeContext}`);
        const afterMemory = process.memoryUsage().heapUsed;
        
        const memoryUsed = (afterMemory - beforeMemory) / 1024 / 1024; // MB
        memoryUsages.push({
          contextSize: size,
          memoryUsed: Math.round(memoryUsed)
        });
      }
      
      benchmarkResults.tests.push({
        test: 'context_window_memory',
        measurements: memoryUsages,
        passed: memoryUsages[memoryUsages.length - 1].memoryUsed < 1000 // < 1GB for max context
      });
      
      // Memory should scale reasonably with context size
      expect(memoryUsages[memoryUsages.length - 1].memoryUsed).toBeLessThan(1000);
    });
  });

  describe('Scalability Benchmarks', () => {
    test('should measure agent scaling performance', async () => {
      const agentCounts = [1, 5, 10, 20];
      const scalingResults = [];
      
      for (const count of agentCounts) {
        const agents = Array(count).fill().map((_, i) => ({
          id: `scale-agent-${i}`,
          type: 'kimi-k2'
        }));
        
        const startTime = performance.now();
        
        // Register all agents
        await Promise.all(agents.map(agent => mesh.registerAgent(agent)));
        
        // Test coordination with all agents
        const task = "Collaborative analysis task for scalability testing";
        const result = await mesh.coordinateTask(task, {
          agentType: 'kimi-k2',
          participantCount: count
        });
        
        const endTime = performance.now();
        const totalTime = endTime - startTime;
        
        scalingResults.push({
          agentCount: count,
          totalTime: Math.round(totalTime),
          timePerAgent: Math.round(totalTime / count)
        });
        
        expect(result.success).toBe(true);
        
        // Cleanup
        await Promise.all(agents.map(agent => mesh.unregisterAgent(agent.id)));
      }
      
      benchmarkResults.tests.push({
        test: 'agent_scaling',
        results: scalingResults,
        passed: scalingResults[scalingResults.length - 1].timePerAgent < 1000 // < 1s per agent
      });
      
      // Time per agent should not increase dramatically
      const firstTimePerAgent = scalingResults[0].timePerAgent;
      const lastTimePerAgent = scalingResults[scalingResults.length - 1].timePerAgent;
      
      expect(lastTimePerAgent).toBeLessThan(firstTimePerAgent * 3); // No more than 3x degradation
    });

    test('should measure network partition recovery time', async () => {
      await mesh.registerAgent({ id: 'recovery-test-agent', type: 'kimi-k2' });
      
      // Simulate network partition
      const partitionStart = performance.now();
      await mesh.simulateNetworkPartition(['node1'], ['node2', 'node3']);
      
      // Test recovery
      const recoveryStart = performance.now();
      await mesh.healNetworkPartition();
      const recoveryEnd = performance.now();
      
      const recoveryTime = recoveryEnd - recoveryStart;
      
      // Verify mesh consistency
      const consistencyCheck = await mesh.verifyConsistency();
      
      benchmarkResults.tests.push({
        test: 'network_recovery',
        recoveryTime: Math.round(recoveryTime),
        consistencyRestored: consistencyCheck.consistent,
        passed: recoveryTime < 5000 && consistencyCheck.consistent
      });
      
      expect(recoveryTime).toBeLessThan(5000); // 5 second recovery
      expect(consistencyCheck.consistent).toBe(true);
    });
  });

  describe('Quality Benchmarks', () => {
    test('should measure reasoning quality consistency', async () => {
      const complexQuery = `
        Design a distributed neural network architecture that can:
        1. Handle 1 million concurrent connections
        2. Provide sub-100ms response times
        3. Maintain 99.9% uptime
        4. Scale horizontally across multiple data centers
        
        Provide detailed architectural diagrams and implementation strategies.
      `;
      
      const responses = [];
      for (let i = 0; i < 5; i++) {
        const response = await kimiClient.query(complexQuery);
        responses.push(response);
      }
      
      // Measure consistency (simplified semantic similarity)
      const consistencyScores = [];
      for (let i = 1; i < responses.length; i++) {
        const similarity = calculateSimilarity(responses[0], responses[i]);
        consistencyScores.push(similarity);
      }
      
      const avgConsistency = consistencyScores.reduce((a, b) => a + b) / consistencyScores.length;
      
      benchmarkResults.tests.push({
        test: 'reasoning_quality_consistency',
        avgConsistency: Math.round(avgConsistency * 100) / 100,
        responses: responses.length,
        passed: avgConsistency > 0.7 // 70% consistency threshold
      });
      
      expect(avgConsistency).toBeGreaterThan(0.7);
    });

    test('should measure tool execution accuracy', async () => {
      const toolTests = [
        {
          tool: 'file_operations',
          operation: 'write',
          path: '/tmp/accuracy-test.txt',
          content: 'Test content for accuracy validation',
          validation: async () => {
            const content = await fs.readFile('/tmp/accuracy-test.txt', 'utf8');
            return content === 'Test content for accuracy validation';
          }
        },
        {
          tool: 'shell_commands',
          command: 'echo "accuracy test output"',
          validation: (result) => {
            return result.stdout.includes('accuracy test output');
          }
        }
      ];
      
      let accurateExecutions = 0;
      
      for (const test of toolTests) {
        const result = await kimiClient.executeTool(test.tool, test);
        const accurate = await test.validation(result);
        
        if (accurate) {
          accurateExecutions++;
        }
      }
      
      const accuracy = accurateExecutions / toolTests.length;
      
      benchmarkResults.tests.push({
        test: 'tool_execution_accuracy',
        accuracy: Math.round(accuracy * 100) / 100,
        totalTests: toolTests.length,
        passed: accuracy === 1.0 // 100% accuracy required
      });
      
      expect(accuracy).toBe(1.0);
    });
  });
});

describe('Kimi-K2 Stress Tests', () => {
  let kimiClient;
  
  beforeAll(async () => {
    kimiClient = new KimiK2Client({
      provider: 'mocktest',
      model: 'kimi-k2-instruct'
    });
    await kimiClient.initialize();
  });
  
  afterAll(async () => {
    await kimiClient.shutdown();
  });

  test('should handle extreme concurrent load', async () => {
    const concurrentRequests = 100;
    const requests = Array(concurrentRequests).fill().map((_, i) => 
      kimiClient.query(`Stress test query ${i}`)
    );
    
    const startTime = performance.now();
    const results = await Promise.allSettled(requests);
    const endTime = performance.now();
    
    const successfulRequests = results.filter(r => r.status === 'fulfilled').length;
    const successRate = successfulRequests / concurrentRequests;
    const totalTime = endTime - startTime;
    
    expect(successRate).toBeGreaterThan(0.95); // 95% success rate
    expect(totalTime).toBeLessThan(60000); // Complete within 60 seconds
  });

  test('should handle memory pressure gracefully', async () => {
    const largeQueries = Array(50).fill().map((_, i) => 
      `Memory pressure test ${i}: ${'large data chunk '.repeat(1000)}`
    );
    
    const results = [];
    for (const query of largeQueries) {
      try {
        const result = await kimiClient.query(query);
        results.push(result);
        
        // Check memory usage periodically
        const memUsage = process.memoryUsage();
        expect(memUsage.heapUsed).toBeLessThan(2 * 1024 * 1024 * 1024); // 2GB limit
      } catch (error) {
        // Some failures acceptable under extreme pressure
      }
    }
    
    expect(results.length).toBeGreaterThan(largeQueries.length * 0.8); // 80% success
  });

  test('should maintain performance during extended operation', async () => {
    const operationDuration = 60000; // 60 seconds
    const startTime = Date.now();
    let queryCount = 0;
    const latencies = [];
    
    while (Date.now() - startTime < operationDuration) {
      const queryStart = performance.now();
      
      try {
        await kimiClient.query(`Extended operation query ${queryCount++}`);
        const queryEnd = performance.now();
        latencies.push(queryEnd - queryStart);
      } catch (error) {
        // Track but continue
      }
      
      // Brief pause to avoid overwhelming
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    const avgLatency = latencies.reduce((a, b) => a + b) / latencies.length;
    const latencyTrend = calculateLatencyTrend(latencies);
    
    expect(queryCount).toBeGreaterThan(100); // Reasonable throughput
    expect(avgLatency).toBeLessThan(5000); // Maintain reasonable latency
    expect(latencyTrend).toBeLessThan(0.1); // No significant degradation trend
  });
});

// Helper functions
function calculateSimilarity(text1, text2) {
  // Simplified similarity calculation (in practice, use proper semantic similarity)
  const words1 = text1.toLowerCase().split(/\W+/);
  const words2 = text2.toLowerCase().split(/\W+/);
  
  const commonWords = words1.filter(word => words2.includes(word));
  const totalUniqueWords = new Set([...words1, ...words2]).size;
  
  return commonWords.length / totalUniqueWords;
}

function calculateLatencyTrend(latencies) {
  // Calculate slope of latency over time (simplified linear regression)
  const n = latencies.length;
  const sumX = (n * (n - 1)) / 2;
  const sumY = latencies.reduce((a, b) => a + b);
  const sumXY = latencies.reduce((sum, y, x) => sum + x * y, 0);
  const sumX2 = (n * (n - 1) * (2 * n - 1)) / 6;
  
  const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  return slope / (sumY / n); // Normalized slope
}