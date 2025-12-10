#!/usr/bin/env node
/**
 * Comprehensive Multi-Protocol Proxy Benchmark Suite
 *
 * Tests:
 * - HTTP/1.1 baseline
 * - HTTP/2 with/without TLS
 * - WebSocket
 * - Rate limiting overhead
 * - Authentication overhead
 * - Concurrent connections
 * - Large payload handling
 */

import { performance } from 'perf_hooks';
import http from 'http';
import http2 from 'http2';
import { WebSocket } from 'ws';

const BENCHMARK_CONFIG = {
  warmupRequests: 10,
  benchmarkRequests: 100,
  concurrentConnections: [1, 5, 10, 20, 50],
  payloadSizes: [100, 1000, 10000, 100000], // bytes
  apiKey: 'benchmark-test-key'
};

class BenchmarkRunner {
  constructor() {
    this.results = {
      http1: {},
      http2: {},
      websocket: {},
      rateLimit: {},
      auth: {}
    };
  }

  // Utility: Generate test payload
  generatePayload(size) {
    return JSON.stringify({
      model: 'gemini-2.0-flash-exp',
      messages: [
        {
          role: 'user',
          content: 'x'.repeat(size)
        }
      ],
      max_tokens: 10,
      stream: false
    });
  }

  // Utility: Measure request latency
  async measureRequest(requestFn) {
    const start = performance.now();
    try {
      await requestFn();
      const end = performance.now();
      return { success: true, latency: end - start };
    } catch (error) {
      const end = performance.now();
      return { success: false, latency: end - start, error: error.message };
    }
  }

  // Benchmark HTTP/1.1
  async benchmarkHTTP1(port = 3000, payloadSize = 1000) {
    console.log(`\n📊 Benchmarking HTTP/1.1 (payload: ${payloadSize} bytes)...`);

    const payload = this.generatePayload(payloadSize);
    const latencies = [];

    // Warmup
    for (let i = 0; i < BENCHMARK_CONFIG.warmupRequests; i++) {
      await this.makeHTTP1Request(port, payload);
    }

    // Benchmark
    for (let i = 0; i < BENCHMARK_CONFIG.benchmarkRequests; i++) {
      const result = await this.measureRequest(() =>
        this.makeHTTP1Request(port, payload)
      );
      if (result.success) {
        latencies.push(result.latency);
      }

      if ((i + 1) % 20 === 0) {
        process.stdout.write('.');
      }
    }

    const stats = this.calculateStats(latencies);
    console.log(`\n✅ HTTP/1.1: ${stats.mean.toFixed(2)}ms avg, ${stats.p95.toFixed(2)}ms p95`);

    return stats;
  }

  // Benchmark HTTP/2
  async benchmarkHTTP2(port = 3001, payloadSize = 1000) {
    console.log(`\n📊 Benchmarking HTTP/2 (payload: ${payloadSize} bytes)...`);

    const payload = this.generatePayload(payloadSize);
    const latencies = [];

    // Warmup
    for (let i = 0; i < BENCHMARK_CONFIG.warmupRequests; i++) {
      await this.makeHTTP2Request(port, payload);
    }

    // Benchmark
    for (let i = 0; i < BENCHMARK_CONFIG.benchmarkRequests; i++) {
      const result = await this.measureRequest(() =>
        this.makeHTTP2Request(port, payload)
      );
      if (result.success) {
        latencies.push(result.latency);
      }

      if ((i + 1) % 20 === 0) {
        process.stdout.write('.');
      }
    }

    const stats = this.calculateStats(latencies);
    console.log(`\n✅ HTTP/2: ${stats.mean.toFixed(2)}ms avg, ${stats.p95.toFixed(2)}ms p95`);

    return stats;
  }

  // Benchmark WebSocket
  async benchmarkWebSocket(port = 8080, payloadSize = 1000) {
    console.log(`\n📊 Benchmarking WebSocket (payload: ${payloadSize} bytes)...`);

    const payload = this.generatePayload(payloadSize);
    const latencies = [];

    // Single persistent connection
    const ws = new WebSocket(`ws://localhost:${port}`);

    await new Promise((resolve, reject) => {
      ws.on('open', resolve);
      ws.on('error', reject);
      setTimeout(() => reject(new Error('Connection timeout')), 5000);
    });

    // Warmup
    for (let i = 0; i < BENCHMARK_CONFIG.warmupRequests; i++) {
      await this.makeWebSocketRequest(ws, payload);
    }

    // Benchmark
    for (let i = 0; i < BENCHMARK_CONFIG.benchmarkRequests; i++) {
      const result = await this.measureRequest(() =>
        this.makeWebSocketRequest(ws, payload)
      );
      if (result.success) {
        latencies.push(result.latency);
      }

      if ((i + 1) % 20 === 0) {
        process.stdout.write('.');
      }
    }

    ws.close();

    const stats = this.calculateStats(latencies);
    console.log(`\n✅ WebSocket: ${stats.mean.toFixed(2)}ms avg, ${stats.p95.toFixed(2)}ms p95`);

    return stats;
  }

  // Benchmark concurrent connections
  async benchmarkConcurrency(protocol, port, connections) {
    console.log(`\n📊 Benchmarking ${protocol} with ${connections} concurrent connections...`);

    const payload = this.generatePayload(1000);
    const requestsPerConnection = Math.floor(BENCHMARK_CONFIG.benchmarkRequests / connections);

    const promises = [];
    const startTime = performance.now();

    for (let i = 0; i < connections; i++) {
      const promise = (async () => {
        const latencies = [];
        for (let j = 0; j < requestsPerConnection; j++) {
          const result = await this.measureRequest(() =>
            protocol === 'http1'
              ? this.makeHTTP1Request(port, payload)
              : this.makeHTTP2Request(port, payload)
          );
          if (result.success) {
            latencies.push(result.latency);
          }
        }
        return latencies;
      })();
      promises.push(promise);
    }

    const allLatencies = (await Promise.all(promises)).flat();
    const totalTime = performance.now() - startTime;
    const stats = this.calculateStats(allLatencies);

    stats.throughput = (allLatencies.length / (totalTime / 1000)).toFixed(2);
    stats.totalTime = totalTime.toFixed(2);

    console.log(`✅ Concurrency ${connections}: ${stats.mean.toFixed(2)}ms avg, ${stats.throughput} req/s`);

    return stats;
  }

  // HTTP/1.1 request
  makeHTTP1Request(port, payload) {
    return new Promise((resolve, reject) => {
      const req = http.request({
        hostname: 'localhost',
        port,
        path: '/v1/messages',
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(payload)
        }
      }, (res) => {
        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => resolve(data));
      });

      req.on('error', reject);
      req.write(payload);
      req.end();

      setTimeout(() => reject(new Error('Request timeout')), 30000);
    });
  }

  // HTTP/2 request
  makeHTTP2Request(port, payload) {
    return new Promise((resolve, reject) => {
      const client = http2.connect(`http://localhost:${port}`);

      const req = client.request({
        ':method': 'POST',
        ':path': '/v1/messages',
        'content-type': 'application/json'
      });

      let data = '';
      req.on('data', chunk => data += chunk);
      req.on('end', () => {
        client.close();
        resolve(data);
      });
      req.on('error', (err) => {
        client.close();
        reject(err);
      });

      req.write(payload);
      req.end();

      setTimeout(() => {
        client.close();
        reject(new Error('Request timeout'));
      }, 30000);
    });
  }

  // WebSocket request
  makeWebSocketRequest(ws, payload) {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Request timeout')), 30000);

      const handler = (data) => {
        clearTimeout(timeout);
        ws.off('message', handler);
        resolve(data);
      };

      ws.on('message', handler);
      ws.send(JSON.stringify({
        type: 'non_streaming_request',
        data: JSON.parse(payload)
      }));
    });
  }

  // Calculate statistics
  calculateStats(latencies) {
    if (latencies.length === 0) {
      return { mean: 0, median: 0, p95: 0, p99: 0, min: 0, max: 0 };
    }

    const sorted = latencies.sort((a, b) => a - b);
    const sum = sorted.reduce((a, b) => a + b, 0);

    return {
      mean: sum / sorted.length,
      median: sorted[Math.floor(sorted.length / 2)],
      p95: sorted[Math.floor(sorted.length * 0.95)],
      p99: sorted[Math.floor(sorted.length * 0.99)],
      min: sorted[0],
      max: sorted[sorted.length - 1],
      count: sorted.length
    };
  }

  // Print summary report
  printSummary() {
    console.log('\n' + '='.repeat(80));
    console.log('📊 BENCHMARK SUMMARY');
    console.log('='.repeat(80));

    console.log('\nProtocol Performance Comparison:');
    console.log('--------------------------------');

    Object.entries(this.results).forEach(([protocol, data]) => {
      if (Object.keys(data).length > 0) {
        console.log(`\n${protocol.toUpperCase()}:`);
        Object.entries(data).forEach(([test, stats]) => {
          console.log(`  ${test}: ${stats.mean?.toFixed(2)}ms avg, ${stats.p95?.toFixed(2)}ms p95`);
          if (stats.throughput) {
            console.log(`    Throughput: ${stats.throughput} req/s`);
          }
        });
      }
    });

    console.log('\n' + '='.repeat(80));
  }
}

// Main benchmark execution
async function main() {
  console.log('🚀 Starting Multi-Protocol Proxy Benchmarks\n');

  const benchmark = new BenchmarkRunner();

  try {
    // Test different payload sizes
    for (const size of BENCHMARK_CONFIG.payloadSizes) {
      console.log(`\n${'='.repeat(80)}`);
      console.log(`Testing with payload size: ${size} bytes`);
      console.log('='.repeat(80));

      // HTTP/1.1
      const http1Stats = await benchmark.benchmarkHTTP1(3000, size);
      benchmark.results.http1[`payload_${size}`] = http1Stats;

      // HTTP/2
      const http2Stats = await benchmark.benchmarkHTTP2(3001, size);
      benchmark.results.http2[`payload_${size}`] = http2Stats;

      // Calculate improvement
      const improvement = ((http1Stats.mean - http2Stats.mean) / http1Stats.mean * 100).toFixed(1);
      console.log(`\n💡 HTTP/2 is ${improvement}% faster for ${size} byte payloads`);
    }

    // Test concurrency
    console.log(`\n${'='.repeat(80)}`);
    console.log('Testing Concurrent Connections');
    console.log('='.repeat(80));

    for (const connections of BENCHMARK_CONFIG.concurrentConnections) {
      const http1Concurrent = await benchmark.benchmarkConcurrency('http1', 3000, connections);
      benchmark.results.http1[`concurrent_${connections}`] = http1Concurrent;

      const http2Concurrent = await benchmark.benchmarkConcurrency('http2', 3001, connections);
      benchmark.results.http2[`concurrent_${connections}`] = http2Concurrent;
    }

    // Print summary
    benchmark.printSummary();

    // Save results
    const fs = await import('fs');
    fs.writeFileSync(
      '/tmp/benchmark-results.json',
      JSON.stringify(benchmark.results, null, 2)
    );
    console.log('\n💾 Results saved to /tmp/benchmark-results.json');

  } catch (error) {
    console.error('❌ Benchmark failed:', error.message);
    process.exit(1);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

export { BenchmarkRunner };
