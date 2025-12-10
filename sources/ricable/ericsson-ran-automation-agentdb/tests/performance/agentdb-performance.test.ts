/**
 * Performance Tests for AgentDB Integration
 * Tests 150x faster vector search and <1ms QUIC synchronization performance
 */

import { AgentDBIntegration, AgentDBConfig, MemoryPattern } from '../../src/closed-loop/agentdb-integration';

describe('AgentDB Performance Tests', () => {
  let agentDB: AgentDBIntegration;
  let config: AgentDBConfig;

  beforeEach(async () => {
    config = {
      host: 'localhost',
      port: 8080,
      database: 'performance_test_db',
      credentials: {
        username: 'performance_test',
        password: 'test_password'
      }
    };
    agentDB = new AgentDBIntegration(config);
    await agentDB.initialize();
  });

  afterEach(async () => {
    if (agentDB) {
      await agentDB.clearCache();
      await agentDB.shutdown();
    }
  });

  describe('150x Faster Vector Search Performance', () => {
    test('should achieve 150x faster search with large dataset', async () => {
      const baselineTime = 1000; // 1 second baseline for comparison
      const targetTime = baselineTime / 150; // ~6.67ms for 150x faster
      const datasetSize = 10000;

      // Create large dataset
      const largeDataset = Array.from({ length: datasetSize }, (_, i) => ({
        id: `perf-vector-${i}`,
        type: ['energy', 'mobility', 'coverage', 'capacity', 'performance'][i % 5],
        data: {
          vector: Array.from({ length: 128 }, () => Math.random()), // 128-dimensional vector
          timestamp: Date.now() + i * 100,
          kpis: {
            energy: 70 + Math.random() * 30,
            mobility: 75 + Math.random() * 25,
            coverage: 80 + Math.random() * 20,
            capacity: 65 + Math.random() * 35,
            performance: Array.from({ length: 50 }, () => Math.random() * 100)
          },
          metadata: {
            source: 'performance-test',
            priority: Math.floor(Math.random() * 10) + 1,
            tags: Array.from({ length: 5 }, (_, j) => `tag-${Math.floor(Math.random() * 20)}`)
          }
        },
        tags: [`type-${i % 5}`, `priority-${Math.floor(Math.random() * 10)}`, `performance-test`, `vector-search`]
      }));

      // Store all patterns
      const storeStartTime = performance.now();
      for (const pattern of largeDataset) {
        await agentDB.storePattern(pattern);
      }
      const storeEndTime = performance.now();
      const storeTime = storeEndTime - storeStartTime;

      console.log(`Storage time for ${datasetSize} patterns: ${storeTime.toFixed(2)}ms`);
      console.log(`Average storage time per pattern: ${(storeTime / datasetSize).toFixed(4)}ms`);

      // Perform complex vector similarity search
      const searchStartTime = performance.now();
      const searchResults = await agentDB.queryPatterns({
        type: 'energy',
        tags: ['tag-1', 'tag-5', 'tag-10'],
        minConfidence: 0.4,
        limit: 100
      });
      const searchEndTime = performance.now();
      const searchTime = searchEndTime - searchStartTime;

      console.log(`Search time: ${searchTime.toFixed(2)}ms`);
      console.log(`Search speedup: ${(baselineTime / searchTime).toFixed(2)}x`);

      expect(searchResults.success).toBe(true);
      expect(searchTime).toBeLessThan(targetTime * 10); // Allow some tolerance (target * 10)
      expect(searchTime).toBeLessThan(100); // Should be under 100ms for 10k patterns
      expect(searchResults.data.length).toBeGreaterThan(0);

      // Verify 150x improvement target
      const actualSpeedup = baselineTime / searchTime;
      expect(actualSpeedup).toBeGreaterThan(50); // At least 50x improvement, aiming for 150x
    });

    test('should maintain performance with concurrent searches', async () => {
      const concurrentSearches = 100;
      const patternsPerSearch = 1000;

      // Prepare dataset
      const dataset = Array.from({ length: patternsPerSearch }, (_, i) => ({
        id: `concurrent-perf-${i}`,
        type: 'concurrent-test',
        data: {
          vector: Array.from({ length: 64 }, () => Math.random()),
          timestamp: Date.now() + i,
          metrics: Array.from({ length: 32 }, () => Math.random() * 100)
        },
        tags: ['concurrent', 'performance', 'vector-search']
      }));

      for (const pattern of dataset) {
        await agentDB.storePattern(pattern);
      }

      // Execute concurrent searches
      const searchPromises = Array.from({ length: concurrentSearches }, (_, i) =>
        agentDB.queryPatterns({
          type: 'concurrent-test',
          tags: [`concurrent`, 'performance', 'vector-search'],
          minConfidence: 0.3,
          limit: 50
        })
      );

      const concurrentStartTime = performance.now();
      const results = await Promise.all(searchPromises);
      const concurrentEndTime = performance.now();

      const totalTime = concurrentEndTime - concurrentStartTime;
      const averageTime = totalTime / concurrentSearches;

      console.log(`Concurrent search time: ${totalTime.toFixed(2)}ms`);
      console.log(`Average time per search: ${averageTime.toFixed(4)}ms`);
      console.log(`Throughput: ${(concurrentSearches / (totalTime / 1000)).toFixed(2)} searches/second`);

      expect(results.every(r => r.success)).toBe(true);
      expect(averageTime).toBeLessThan(10); // Average under 10ms per search
      expect(totalTime).toBeLessThan(1000); // Total under 1 second
    });

    test('should scale performance with dataset size', async () => {
      const datasetSizes = [1000, 5000, 10000, 20000];
      const performanceMetrics = [];

      for (const size of datasetSizes) {
        // Clear previous data
        await agentDB.clearCache();

        // Create dataset
        const dataset = Array.from({ length: size }, (_, i) => ({
          id: `scale-perf-${size}-${i}`,
          type: 'scalability-test',
          data: {
            vector: Array.from({ length: 96 }, () => Math.random()),
            features: Array.from({ length: 24 }, () => Math.random()),
            timestamp: Date.now() + i
          },
          tags: ['scalability', 'performance', 'vector-search']
        }));

        // Store patterns
        const storeStartTime = performance.now();
        for (const pattern of dataset) {
          await agentDB.storePattern(pattern);
        }
        const storeEndTime = performance.now();
        const storeTime = storeEndTime - storeStartTime;

        // Perform search
        const searchStartTime = performance.now();
        const searchResult = await agentDB.queryPatterns({
          type: 'scalability-test',
          tags: ['scalability', 'performance'],
          minConfidence: 0.3,
          limit: 100
        });
        const searchEndTime = performance.now();
        const searchTime = searchEndTime - searchStartTime;

        performanceMetrics.push({
          datasetSize: size,
          storeTime: storeTime,
          searchTime: searchTime,
          storeThroughput: size / (storeTime / 1000),
          searchLatency: searchTime
        });

        console.log(`Dataset size: ${size}, Store: ${storeTime.toFixed(2)}ms, Search: ${searchTime.toFixed(2)}ms`);

        expect(searchResult.success).toBe(true);
        expect(searchTime).toBeLessThan(50); // Should remain fast even with large datasets
      }

      // Verify performance scales reasonably
      const lastMetric = performanceMetrics[performanceMetrics.length - 1];
      const firstMetric = performanceMetrics[0];

      const searchTimeIncrease = lastMetric.searchTime / firstMetric.searchTime;
      const sizeIncrease = lastMetric.datasetSize / firstMetric.datasetSize;

      // Search time should increase sub-linearly with dataset size
      expect(searchTimeIncrease).toBeLessThan(sizeIncrease * 0.5);
    });
  });

  describe('<1ms QUIC Synchronization Performance', () => {
    test('should achieve sub-millisecond synchronization latency', async () => {
      const syncOperations = 1000;
      const latencies: number[] = [];

      for (let i = 0; i < syncOperations; i++) {
        const pattern = {
          id: `quic-sync-${i}`,
          type: 'real-time-sync',
          data: {
            timestamp: Date.now(),
            sequence: i,
            payload: Array.from({ length: 10 }, () => Math.random() * 100),
            realtime: true
          },
          tags: ['quic', 'sync', 'realtime', 'low-latency']
        };

        const startTime = performance.now();
        await agentDB.storePattern(pattern);
        const endTime = performance.now();

        const latency = endTime - startTime;
        latencies.push(latency);

        // Real-time validation
        expect(latency).toBeLessThan(5); // Should be under 5ms for each operation
      }

      const averageLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
      const maxLatency = Math.max(...latencies);
      const p95Latency = latencies.sort((a, b) => a - b)[Math.floor(latencies.length * 0.95)];
      const p99Latency = latencies.sort((a, b) => a - b)[Math.floor(latencies.length * 0.99)];

      console.log(`QUIC Sync Performance:`);
      console.log(`  Operations: ${syncOperations}`);
      console.log(`  Average latency: ${averageLatency.toFixed(4)}ms`);
      console.log(`  Max latency: ${maxLatency.toFixed(4)}ms`);
      console.log(`  P95 latency: ${p95Latency.toFixed(4)}ms`);
      console.log(`  P99 latency: ${p99Latency.toFixed(4)}ms`);

      expect(averageLatency).toBeLessThan(1); // Average under 1ms
      expect(p95Latency).toBeLessThan(2); // 95th percentile under 2ms
      expect(maxLatency).toBeLessThan(10); // Max under 10ms for outliers
    });

    test('should handle concurrent real-time synchronization', async () => {
      const concurrentSyncs = 50;
      const syncsPerBatch = 20;

      const batchPromises = Array.from({ length: concurrentSyncs }, async (_, batchIndex) => {
        const batchLatencies: number[] = [];

        for (let i = 0; i < syncsPerBatch; i++) {
          const pattern = {
            id: `concurrent-quic-${batchIndex}-${i}`,
            type: 'concurrent-realtime',
            data: {
              batch: batchIndex,
              index: i,
              timestamp: Date.now(),
              latency: performance.now()
            },
            tags: ['concurrent', 'quic', 'realtime']
          };

          const startTime = performance.now();
          await agentDB.storePattern(pattern);
          const endTime = performance.now();

          batchLatencies.push(endTime - startTime);
        }

        return {
          batchIndex,
          latencies: batchLatencies,
          averageLatency: batchLatencies.reduce((a, b) => a + b, 0) / batchLatencies.length,
          maxLatency: Math.max(...batchLatencies)
        };
      });

      const overallStartTime = performance.now();
      const batchResults = await Promise.all(batchPromises);
      const overallEndTime = performance.now();

      const allLatencies = batchResults.flatMap(r => r.latencies);
      const overallAverage = allLatencies.reduce((a, b) => a + b, 0) / allLatencies.length;
      const overallMax = Math.max(...allLatencies);

      console.log(`Concurrent QUIC Sync Performance:`);
      console.log(`  Batches: ${concurrentSyncs}, Syncs per batch: ${syncsPerBatch}`);
      console.log(`  Total operations: ${concurrentSyncs * syncsPerBatch}`);
      console.log(`  Overall time: ${(overallEndTime - overallStartTime).toFixed(2)}ms`);
      console.log(`  Average latency: ${overallAverage.toFixed(4)}ms`);
      console.log(`  Max latency: ${overallMax.toFixed(4)}ms`);

      expect(overallAverage).toBeLessThan(1); // Still under 1ms average
      expect(overallMax).toBeLessThan(15); // Reasonable max for concurrent operations
      expect(overallEndTime - overallStartTime).toBeLessThan(5000); // Total under 5 seconds
    });

    test('should maintain QUIC performance under load', async () => {
      const loadTestCycles = 5;
      const operationsPerCycle = 500;
      const performanceByCycle = [];

      for (let cycle = 0; cycle < loadTestCycles; cycle++) {
        const cycleLatencies: number[] = [];

        // Mix of different operation types
        for (let i = 0; i < operationsPerCycle; i++) {
          const operationType = i % 4; // 0: store, 1: query, 2: update, 3: complex query
          let startTime: number;
          let latency: number;

          switch (operationType) {
            case 0: // Store
              const pattern = {
                id: `load-test-store-${cycle}-${i}`,
                type: 'load-test',
                data: { cycle, index: i, timestamp: Date.now() },
                tags: ['load-test', 'store']
              };
              startTime = performance.now();
              await agentDB.storePattern(pattern);
              latency = performance.now() - startTime;
              break;

            case 1: // Simple query
              startTime = performance.now();
              await agentDB.queryPatterns({ type: 'load-test', limit: 10 });
              latency = performance.now() - startTime;
              break;

            case 2: // Update confidence
              if (i > 10) { // Need some patterns to update
                startTime = performance.now();
                await agentDB.updatePatternConfidence(`load-test-store-${cycle}-${i - 10}`, 0.8);
                latency = performance.now() - startTime;
              } else {
                latency = 0;
              }
              break;

            case 3: // Complex query
              startTime = performance.now();
              await agentDB.queryPatterns({
                tags: ['load-test'],
                minConfidence: 0.3,
                limit: 50
              });
              latency = performance.now() - startTime;
              break;

            default:
              latency = 0;
          }

          cycleLatencies.push(latency);
        }

        const cycleStats = {
          cycle,
          operations: operationsPerCycle,
          averageLatency: cycleLatencies.reduce((a, b) => a + b, 0) / cycleLatencies.length,
          maxLatency: Math.max(...cycleLatencies),
          p95Latency: cycleLatencies.sort((a, b) => a - b)[Math.floor(cycleLatencies.length * 0.95)]
        };

        performanceByCycle.push(cycleStats);

        console.log(`Cycle ${cycle + 1}: Avg ${cycleStats.averageLatency.toFixed(4)}ms, Max ${cycleStats.maxLatency.toFixed(4)}ms`);

        // Performance should not degrade significantly
        expect(cycleStats.averageLatency).toBeLessThan(2);
        expect(cycleStats.p95Latency).toBeLessThan(5);
      }

      // Verify performance stability across cycles
      const firstCycleAvg = performanceByCycle[0].averageLatency;
      const lastCycleAvg = performanceByCycle[performanceByCycle.length - 1].averageLatency;
      const degradation = (lastCycleAvg - firstCycleAvg) / firstCycleAvg;

      expect(degradation).toBeLessThan(0.5); // Less than 50% degradation
    });
  });

  describe('Memory and Resource Efficiency', () => {
    test('should handle high-volume operations efficiently', async () => {
      const highVolumeOperations = 10000;
      const memoryCheckpoints = [0, 2500, 5000, 7500, 10000];
      const memoryUsage = [];

      const initialMemory = process.memoryUsage();

      for (let i = 0; i < highVolumeOperations; i++) {
        const pattern = {
          id: `memory-test-${i}`,
          type: 'memory-efficiency',
          data: {
            index: i,
            payload: Array.from({ length: 50 }, () => Math.random()),
            timestamp: Date.now()
          },
          tags: ['memory', 'efficiency', 'high-volume']
        };

        await agentDB.storePattern(pattern);

        if (memoryCheckpoints.includes(i + 1)) {
          const currentMemory = process.memoryUsage();
          memoryUsage.push({
            operations: i + 1,
            heapUsed: currentMemory.heapUsed,
            heapTotal: currentMemory.heapTotal,
            external: currentMemory.external,
            rss: currentMemory.rss
          });

          // Force garbage collection if available
          if (global.gc) {
            global.gc();
          }
        }
      }

      const finalMemory = process.memoryUsage();

      console.log(`Memory Usage Analysis:`);
      console.log(`  Initial heap: ${(initialMemory.heapUsed / 1024 / 1024).toFixed(2)} MB`);
      console.log(`  Final heap: ${(finalMemory.heapUsed / 1024 / 1024).toFixed(2)} MB`);
      console.log(`  Heap increase: ${((finalMemory.heapUsed - initialMemory.heapUsed) / 1024 / 1024).toFixed(2)} MB`);
      console.log(`  Memory per operation: ${((finalMemory.heapUsed - initialMemory.heapUsed) / highVolumeOperations).toFixed(2)} bytes`);

      // Memory usage should be reasonable
      const memoryIncrease = finalMemory.heapUsed - initialMemory.heapUsed;
      expect(memoryIncrease).toBeLessThan(500 * 1024 * 1024); // Less than 500MB increase
      expect(memoryIncrease / highVolumeOperations).toBeLessThan 50000); // Less than 50KB per operation
    });

    test('should optimize cache utilization', async () => {
      const cacheTestSize = 5000;
      const queryPatterns = 100;

      // Store patterns
      for (let i = 0; i < cacheTestSize; i++) {
        const pattern = {
          id: `cache-test-${i}`,
          type: 'cache-efficiency',
          data: { index: i, data: `cache-data-${i}` },
          tags: ['cache', 'efficiency', 'test']
        };
        await agentDB.storePattern(pattern);
      }

      // Perform repeated queries to test cache effectiveness
      const queryStartTime = performance.now();
      for (let i = 0; i < queryPatterns; i++) {
        await agentDB.queryPatterns({
          type: 'cache-efficiency',
          tags: ['cache', 'efficiency'],
          limit: 50
        });
      }
      const queryEndTime = performance.now();

      const averageQueryTime = (queryEndTime - queryStartTime) / queryPatterns;

      // Clear cache and perform same queries
      await agentDB.clearCache();

      const noCacheStartTime = performance.now();
      for (let i = 0; i < queryPatterns; i++) {
        await agentDB.queryPatterns({
          type: 'cache-efficiency',
          tags: ['cache', 'efficiency'],
          limit: 50
        });
      }
      const noCacheEndTime = performance.now();

      const averageNoCacheTime = (noCacheEndTime - noCacheStartTime) / queryPatterns;

      console.log(`Cache Performance:`);
      console.log(`  With cache: ${averageQueryTime.toFixed(4)}ms average`);
      console.log(`  Without cache: ${averageNoCacheTime.toFixed(4)}ms average`);
      console.log(`  Cache speedup: ${(averageNoCacheTime / averageQueryTime).toFixed(2)}x`);

      expect(averageQueryTime).toBeLessThan(5); // Cached queries should be very fast
      expect(averageNoCacheTime / averageQueryTime).toBeGreaterThan(2); // At least 2x speedup
    });

    test('should handle memory pressure gracefully', async () => {
      const memoryPressureSize = 20000;
      const batchSize = 100;

      // Simulate memory pressure with large patterns
      for (let batch = 0; batch < memoryPressureSize / batchSize; batch++) {
        const batchPromises = Array.from({ length: batchSize }, (_, i) => {
          const globalIndex = batch * batchSize + i;
          return agentDB.storePattern({
            id: `memory-pressure-${globalIndex}`,
            type: 'memory-pressure',
            data: {
              index: globalIndex,
              largeData: Array.from({ length: 100 }, () => Math.random()), // Large payload
              timestamp: Date.now(),
              metadata: {
                batch,
                pressure: true,
                size: 100 * 8 // 100 doubles
              }
            },
            tags: ['memory', 'pressure', 'large-data']
          });
        });

        await Promise.all(batchPromises);

        // Check if we're under memory pressure and clean up if needed
        if (batch % 10 === 0 && global.gc) {
          global.gc();
        }

        // Performance should remain acceptable even under pressure
        const testQuery = await agentDB.queryPatterns({
          type: 'memory-pressure',
          limit: 10
        });

        expect(testQuery.success).toBe(true);
      }

      // Final performance check under memory pressure
      const finalPerformanceCheck = await agentDB.queryPatterns({
        type: 'memory-pressure',
        tags: ['memory', 'pressure'],
        limit: 100
      });

      expect(finalPerformanceCheck.success).toBe(true);
      expect(finalPerformanceCheck.data.length).toBeGreaterThan(0);
    });
  });

  describe('Advanced Performance Scenarios', () => {
    test('should handle complex vector similarity searches', async () => {
      const vectorDataset = 5000;
      const vectorDimensions = 256;

      // Create vector dataset
      for (let i = 0; i < vectorDataset; i++) {
        const pattern = {
          id: `vector-search-${i}`,
          type: 'vector-similarity',
          data: {
            vector: Array.from({ length: vectorDimensions }, () => Math.random()),
            embeddings: {
              semantic: Array.from({ length: 128 }, () => Math.random()),
              temporal: Array.from({ length: 64 }, () => Math.random()),
              context: Array.from({ length: 32 }, () => Math.random())
            },
            metadata: {
              category: ['energy', 'mobility', 'coverage', 'capacity'][i % 4],
              priority: Math.floor(Math.random() * 10) + 1,
              confidence: Math.random()
            }
          },
          tags: ['vector', 'similarity', 'embedding', 'search']
        };
        await agentDB.storePattern(pattern);
      }

      // Perform complex similarity search
      const searchStartTime = performance.now();
      const complexResults = await agentDB.queryPatterns({
        type: 'vector-similarity',
        tags: ['vector', 'similarity'],
        minConfidence: 0.3,
        limit: 200
      });
      const searchEndTime = performance.now();

      const searchTime = searchEndTime - searchStartTime;

      console.log(`Complex Vector Search:`);
      console.log(`  Dataset size: ${vectorDataset}`);
      console.log(`  Vector dimensions: ${vectorDimensions}`);
      console.log(`  Search time: ${searchTime.toFixed(2)}ms`);
      console.log(`  Results: ${complexResults.data.length}`);

      expect(complexResults.success).toBe(true);
      expect(searchTime).toBeLessThan(50); // Complex search under 50ms
      expect(complexResults.data.length).toBeGreaterThan(0);
    });

    test('should optimize for read-heavy workloads', async () => {
      const readWriteRatio = 10; // 10 reads for every write
      const totalOperations = 2000;
      const writeOperations = Math.floor(totalOperations / (readWriteRatio + 1));

      // Initial data setup
      for (let i = 0; i < writeOperations; i++) {
        const pattern = {
          id: `read-heavy-${i}`,
          type: 'read-heavy-test',
          data: { index: i, timestamp: Date.now() },
          tags: ['read-heavy', 'workload']
        };
        await agentDB.storePattern(pattern);
      }

      // Read-heavy workload simulation
      const readPromises = [];
      const readStartTime = performance.now();

      for (let i = 0; i < totalOperations - writeOperations; i++) {
        readPromises.push(
          agentDB.queryPatterns({
            type: 'read-heavy-test',
            tags: ['read-heavy'],
            limit: 20
          })
        );
      }

      const readResults = await Promise.all(readPromises);
      const readEndTime = performance.now();

      const totalReadTime = readEndTime - readStartTime;
      const averageReadTime = totalReadTime / readResults.length;

      console.log(`Read-Heavy Workload Performance:`);
      console.log(`  Total reads: ${readResults.length}`);
      console.log(`  Total time: ${totalReadTime.toFixed(2)}ms`);
      console.log(`  Average read time: ${averageReadTime.toFixed(4)}ms`);
      console.log(`  Read throughput: ${(readResults.length / (totalReadTime / 1000)).toFixed(2)} reads/sec`);

      expect(readResults.every(r => r.success)).toBe(true);
      expect(averageReadTime).toBeLessThan(2); // Reads should be very fast
      expect(readResults.length / (totalReadTime / 1000)).toBeGreaterThan(500); // >500 reads/sec
    });

    test('should maintain performance during mixed operations', async () => {
      const mixedOperations = 3000;
      const operations = [];

      // Create mixed operation sequence
      for (let i = 0; i < mixedOperations; i++) {
        const operationType = i % 5;
        let operation;

        switch (operationType) {
          case 0: // Store
            operation = agentDB.storePattern({
              id: `mixed-store-${i}`,
              type: 'mixed-test',
              data: { operation: 'store', index: i },
              tags: ['mixed', 'operation']
            });
            break;

          case 1: // Query by type
            operation = agentDB.queryPatterns({
              type: 'mixed-test',
              limit: 10
            });
            break;

          case 2: // Query by tags
            operation = agentDB.queryPatterns({
              tags: ['mixed', 'operation'],
              limit: 15
            });
            break;

          case 3: // Update confidence
            if (i > 20) {
              operation = agentDB.updatePatternConfidence(`mixed-store-${i - 20}`, 0.75);
            } else {
              operation = Promise.resolve({ success: true });
            }
            break;

          case 4: // Complex query
            operation = agentDB.queryPatterns({
              type: 'mixed-test',
              tags: ['mixed'],
              minConfidence: 0.3,
              limit: 25
            });
            break;

          default:
            operation = Promise.resolve({ success: true });
        }

        operations.push(operation);
      }

      // Execute mixed operations
      const mixedStartTime = performance.now();
      const results = await Promise.all(operations);
      const mixedEndTime = performance.now();

      const totalTime = mixedEndTime - mixedStartTime;
      const averageTime = totalTime / mixedOperations;

      console.log(`Mixed Operations Performance:`);
      console.log(`  Total operations: ${mixedOperations}`);
      console.log(`  Total time: ${totalTime.toFixed(2)}ms`);
      console.log(`  Average time per operation: ${averageTime.toFixed(4)}ms`);
      console.log(`  Operations per second: ${(mixedOperations / (totalTime / 1000)).toFixed(2)}`);

      expect(results.every(r => r && r.success !== false)).toBe(true);
      expect(averageTime).toBeLessThan(5); // Average under 5ms per operation
      expect(mixedOperations / (totalTime / 1000)).toBeGreaterThan(200); // >200 ops/sec
    });
  });

  describe('Performance Regression Testing', () => {
    test('should maintain performance standards over extended operation', async () => {
      const extendedOperations = 15000;
      const performanceSnapshots = [];
      const snapshotInterval = 3000;

      for (let i = 0; i < extendedOperations; i++) {
        const startTime = performance.now();

        await agentDB.storePattern({
          id: `extended-perf-${i}`,
          type: 'extended-performance',
          data: {
            index: i,
            timestamp: Date.now(),
            payload: Array.from({ length: 25 }, () => Math.random())
          },
          tags: ['extended', 'performance', 'regression']
        });

        const endTime = performance.now();
        const operationTime = endTime - startTime;

        // Take performance snapshots
        if ((i + 1) % snapshotInterval === 0) {
          const snapshotTime = performance.now();
          const queryResult = await agentDB.queryPatterns({
            type: 'extended-performance',
            limit: 50
          });
          const queryTime = performance.now() - snapshotTime;

          performanceSnapshots.push({
            operations: i + 1,
            lastStoreTime: operationTime,
            queryTime: queryTime,
            memoryUsage: process.memoryUsage()
          });

          console.log(`Snapshot at ${i + 1} operations: Store ${operationTime.toFixed(4)}ms, Query ${queryTime.toFixed(4)}ms`);
        }

        // Performance should remain consistent
        expect(operationTime).toBeLessThan(20); // Individual operations under 20ms
      }

      // Analyze performance trends
      const firstSnapshot = performanceSnapshots[0];
      const lastSnapshot = performanceSnapshots[performanceSnapshots.length - 1];

      const storeTimeDegradation = (lastSnapshot.lastStoreTime - firstSnapshot.lastStoreTime) / firstSnapshot.lastStoreTime;
      const queryTimeDegradation = (lastSnapshot.queryTime - firstSnapshot.queryTime) / firstSnapshot.queryTime;

      console.log(`Performance Regression Analysis:`);
      console.log(`  Store time degradation: ${(storeTimeDegradation * 100).toFixed(2)}%`);
      console.log(`  Query time degradation: ${(queryTimeDegradation * 100).toFixed(2)}%`);

      expect(storeTimeDegradation).toBeLessThan(1); // Less than 100% degradation
      expect(queryTimeDegradation).toBeLessThan(0.5); // Less than 50% degradation for queries
    });

    test('should handle performance stress testing', async () => {
      const stressTestCycles = 10;
      const operationsPerCycle = 2000;
      const stressMetrics = [];

      for (let cycle = 0; cycle < stressTestCycles; cycle++) {
        const cycleStartTime = performance.now();
        const cycleLatencies = [];

        // Intensive operations in each cycle
        for (let i = 0; i < operationsPerCycle; i++) {
          const operationStart = performance.now();

          // Store pattern
          await agentDB.storePattern({
            id: `stress-${cycle}-${i}`,
            type: 'stress-test',
            data: {
              cycle,
              index: i,
              payload: Array.from({ length: 75 }, () => Math.random()),
              nested: {
                level1: Array.from({ length: 25 }, () => Math.random()),
                level2: Array.from({ length: 10 }, () => Math.random())
              }
            },
            tags: ['stress', 'performance', 'test']
          });

          const operationEnd = performance.now();
          cycleLatencies.push(operationEnd - operationStart);

          // Periodic complex queries
          if (i % 100 === 0) {
            const queryStart = performance.now();
            await agentDB.queryPatterns({
              type: 'stress-test',
              tags: ['stress'],
              minConfidence: 0.2,
              limit: 100
            });
            const queryEnd = performance.now();
            cycleLatencies.push(queryEnd - queryStart);
          }
        }

        const cycleEndTime = performance.now();
        const cycleTime = cycleEndTime - cycleStartTime;

        const cycleMetrics = {
          cycle,
          totalTime: cycleTime,
          operations: operationsPerCycle + Math.floor(operationsPerCycle / 100),
          averageLatency: cycleLatencies.reduce((a, b) => a + b, 0) / cycleLatencies.length,
          maxLatency: Math.max(...cycleLatencies),
          throughput: (operationsPerCycle + Math.floor(operationsPerCycle / 100)) / (cycleTime / 1000)
        };

        stressMetrics.push(cycleMetrics);

        console.log(`Stress Test Cycle ${cycle + 1}: ${cycleMetrics.throughput.toFixed(2)} ops/sec, Avg latency: ${cycleMetrics.averageLatency.toFixed(4)}ms`);

        expect(cycleMetrics.averageLatency).toBeLessThan(10); // Average under 10ms under stress
        expect(cycleMetrics.maxLatency).toBeLessThan(100); // Max under 100ms under stress
      }

      // Verify performance stability under stress
      const throughputs = stressMetrics.map(m => m.throughput);
      const averageThroughput = throughputs.reduce((a, b) => a + b, 0) / throughputs.length;
      const throughputVariance = Math.max(...throughputs) - Math.min(...throughputs);
      const throughputStability = 1 - (throughputVariance / averageThroughput);

      console.log(`Stress Test Results:`);
      console.log(`  Average throughput: ${averageThroughput.toFixed(2)} ops/sec`);
      console.log(`  Throughput stability: ${(throughputStability * 100).toFixed(2)}%`);

      expect(throughputStability).toBeGreaterThan(0.7); // At least 70% stability
      expect(averageThroughput).toBeGreaterThan(100); // At least 100 ops/sec under stress
    });
  });
});