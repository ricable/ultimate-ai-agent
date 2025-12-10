/**
 * QUIC Transport Benchmark Suite
 * Comprehensive performance testing for QUIC vs HTTP/2 vs WebSocket
 */

import { performance } from 'perf_hooks';
import * as quic from '@fails-components/webtransport';
import http2 from 'http2';
import WebSocket from 'ws';
import { createServer as createHttpServer } from 'http';
import { cpuUsage, memoryUsage } from 'process';

// ============================================================================
// Benchmark Configuration
// ============================================================================

interface BenchmarkConfig {
  name: string;
  agentCounts: number[];
  messageSizes: number[];
  latencies: number[];
  duration: number; // seconds
  warmupRounds: number;
}

const DEFAULT_CONFIG: BenchmarkConfig = {
  name: 'QUIC Performance Benchmark',
  agentCounts: [10, 100, 1000],
  messageSizes: [1024, 10240, 102400], // 1KB, 10KB, 100KB
  latencies: [0, 50, 100], // ms
  duration: 60,
  warmupRounds: 3,
};

// ============================================================================
// Performance Metrics
// ============================================================================

interface Metrics {
  connectionTime: number;
  throughput: number; // messages/sec
  latency: number; // ms
  memoryUsage: number; // MB
  cpuUsage: number; // percentage
  streamEfficiency: number;
  errorRate: number;
  bandwidthUtilization: number; // Mbps
}

interface BenchmarkResult {
  protocol: 'QUIC' | 'HTTP/2' | 'WebSocket';
  agentCount: number;
  messageSize: number;
  networkLatency: number;
  metrics: Metrics;
  timestamp: number;
}

// ============================================================================
// QUIC Benchmark Implementation
// ============================================================================

class QUICBenchmark {
  private server: any;
  private clients: any[] = [];
  private metrics: Partial<Metrics> = {};

  async setup(port: number = 4433): Promise<void> {
    // Create QUIC server with optimized settings
    this.server = quic.createServer({
      port,
      host: '0.0.0.0',
      secret: Buffer.from('supersecretkey12345678901234567890'),
      cert: await this.generateSelfSignedCert(),
      key: await this.generatePrivateKey(),

      // Optimized QUIC parameters
      initialMaxStreamsBidi: 100,
      initialMaxStreamsUni: 100,
      initialMaxData: 10 * 1024 * 1024, // 10MB
      initialMaxStreamDataBidiLocal: 1024 * 1024, // 1MB
      initialMaxStreamDataBidiRemote: 1024 * 1024,
      initialMaxStreamDataUni: 1024 * 1024,

      // BBR congestion control
      congestionControl: 'bbr',

      // Connection migration
      disableMigration: false,

      // Performance tuning
      maxIdleTimeout: 30000,
      maxAckDelay: 25,
    });

    await this.server.listen();
  }

  async benchmarkConnectionEstablishment(agentCount: number): Promise<number> {
    const start = performance.now();
    const connectionPromises = [];

    for (let i = 0; i < agentCount; i++) {
      connectionPromises.push(this.createClient(i));
    }

    await Promise.all(connectionPromises);
    const duration = performance.now() - start;

    return duration / agentCount; // Average connection time per agent
  }

  async benchmarkMultiAgentSpawn(agentCount: number): Promise<number> {
    const start = performance.now();

    // Spawn agents in parallel using QUIC streams
    const spawnPromises = this.clients.map(async (client, idx) => {
      const stream = await client.createBidirectionalStream();
      const message = Buffer.from(JSON.stringify({
        type: 'spawn',
        agentId: `agent-${idx}`,
        capabilities: ['researcher', 'coder', 'tester'],
      }));

      await stream.writable.getWriter().write(message);
      await stream.readable.getReader().read();
    });

    await Promise.all(spawnPromises);
    const duration = performance.now() - start;

    return duration;
  }

  async benchmarkMessageThroughput(
    agentCount: number,
    messageSize: number,
    duration: number
  ): Promise<number> {
    let messagesSent = 0;
    const startTime = performance.now();
    const endTime = startTime + duration * 1000;

    const message = Buffer.alloc(messageSize);

    while (performance.now() < endTime) {
      const promises = this.clients.map(async (client) => {
        const stream = await client.createBidirectionalStream();
        const writer = stream.writable.getWriter();
        await writer.write(message);
        await writer.close();
        messagesSent++;
      });

      await Promise.all(promises);
    }

    const actualDuration = (performance.now() - startTime) / 1000;
    return messagesSent / actualDuration; // messages per second
  }

  async benchmarkStreamMultiplexing(agentCount: number): Promise<number> {
    const streamsPerAgent = 10;
    const start = performance.now();

    const allStreams = [];

    for (const client of this.clients) {
      for (let i = 0; i < streamsPerAgent; i++) {
        allStreams.push(
          client.createBidirectionalStream().then(async (stream: any) => {
            const writer = stream.writable.getWriter();
            await writer.write(Buffer.from(`Stream ${i} data`));
            await writer.close();
          })
        );
      }
    }

    await Promise.all(allStreams);
    const duration = performance.now() - start;

    const totalStreams = agentCount * streamsPerAgent;
    return totalStreams / (duration / 1000); // streams per second
  }

  async measureResourceUsage(): Promise<{ memory: number; cpu: number }> {
    const mem = memoryUsage();
    const cpu = cpuUsage();

    return {
      memory: mem.heapUsed / 1024 / 1024, // MB
      cpu: (cpu.user + cpu.system) / 1000000, // percentage estimate
    };
  }

  private async createClient(id: number): Promise<any> {
    const client = quic.connect({
      url: 'https://localhost:4433',
      serverCertificateHashes: [
        {
          algorithm: 'sha-256',
          value: await this.getCertHash(),
        },
      ],
    });

    await client.ready;
    this.clients.push(client);
    return client;
  }

  private async generateSelfSignedCert(): Promise<Buffer> {
    // In production, use proper certificate generation
    return Buffer.from('MOCK_CERT');
  }

  private async generatePrivateKey(): Promise<Buffer> {
    return Buffer.from('MOCK_KEY');
  }

  private async getCertHash(): Promise<ArrayBuffer> {
    // Mock implementation
    return new ArrayBuffer(32);
  }

  async cleanup(): Promise<void> {
    await Promise.all(this.clients.map(c => c.close()));
    await this.server.close();
    this.clients = [];
  }
}

// ============================================================================
// HTTP/2 Benchmark Implementation
// ============================================================================

class HTTP2Benchmark {
  private server: http2.Http2SecureServer | null = null;
  private clients: http2.ClientHttp2Session[] = [];

  async setup(port: number = 8443): Promise<void> {
    this.server = http2.createSecureServer({
      allowHTTP1: false,
      maxConcurrentStreams: 100,
    });

    this.server.on('stream', (stream, headers) => {
      stream.respond({ ':status': 200 });
      stream.on('data', (chunk) => {
        stream.write(chunk);
      });
      stream.on('end', () => {
        stream.end();
      });
    });

    await new Promise<void>((resolve) => {
      this.server!.listen(port, resolve);
    });
  }

  async benchmarkConnectionEstablishment(agentCount: number): Promise<number> {
    const start = performance.now();

    for (let i = 0; i < agentCount; i++) {
      const client = http2.connect('https://localhost:8443', {
        rejectUnauthorized: false,
      });
      this.clients.push(client);
      await new Promise<void>((resolve) => {
        client.on('connect', resolve);
      });
    }

    return (performance.now() - start) / agentCount;
  }

  async benchmarkMessageThroughput(
    agentCount: number,
    messageSize: number,
    duration: number
  ): Promise<number> {
    let messagesSent = 0;
    const startTime = performance.now();
    const endTime = startTime + duration * 1000;
    const message = Buffer.alloc(messageSize);

    while (performance.now() < endTime) {
      const promises = this.clients.map((client) => {
        return new Promise<void>((resolve) => {
          const req = client.request({ ':path': '/' });
          req.write(message);
          req.on('response', () => {
            messagesSent++;
            resolve();
          });
          req.end();
        });
      });

      await Promise.all(promises);
    }

    return messagesSent / (duration);
  }

  async cleanup(): Promise<void> {
    this.clients.forEach(c => c.close());
    this.server?.close();
    this.clients = [];
  }
}

// ============================================================================
// WebSocket Benchmark Implementation
// ============================================================================

class WebSocketBenchmark {
  private server: WebSocket.Server | null = null;
  private clients: WebSocket[] = [];

  async setup(port: number = 8080): Promise<void> {
    const httpServer = createHttpServer();
    this.server = new WebSocket.Server({ server: httpServer });

    this.server.on('connection', (ws) => {
      ws.on('message', (data) => {
        ws.send(data);
      });
    });

    await new Promise<void>((resolve) => {
      httpServer.listen(port, resolve);
    });
  }

  async benchmarkConnectionEstablishment(agentCount: number): Promise<number> {
    const start = performance.now();

    for (let i = 0; i < agentCount; i++) {
      const ws = new WebSocket('ws://localhost:8080');
      this.clients.push(ws);
      await new Promise<void>((resolve) => {
        ws.on('open', resolve);
      });
    }

    return (performance.now() - start) / agentCount;
  }

  async benchmarkMessageThroughput(
    agentCount: number,
    messageSize: number,
    duration: number
  ): Promise<number> {
    let messagesSent = 0;
    const startTime = performance.now();
    const endTime = startTime + duration * 1000;
    const message = Buffer.alloc(messageSize);

    while (performance.now() < endTime) {
      const promises = this.clients.map((client) => {
        return new Promise<void>((resolve) => {
          client.once('message', () => {
            messagesSent++;
            resolve();
          });
          client.send(message);
        });
      });

      await Promise.all(promises);
    }

    return messagesSent / duration;
  }

  async cleanup(): Promise<void> {
    this.clients.forEach(c => c.close());
    this.server?.close();
    this.clients = [];
  }
}

// ============================================================================
// Network Latency Simulator
// ============================================================================

class LatencySimulator {
  static async simulate(latencyMs: number, fn: () => Promise<any>): Promise<any> {
    const start = performance.now();
    const result = await fn();
    const elapsed = performance.now() - start;

    if (elapsed < latencyMs) {
      await new Promise(resolve => setTimeout(resolve, latencyMs - elapsed));
    }

    return result;
  }
}

// ============================================================================
// Benchmark Runner
// ============================================================================

class BenchmarkRunner {
  private results: BenchmarkResult[] = [];

  async runFullSuite(config: BenchmarkConfig = DEFAULT_CONFIG): Promise<void> {
    console.log(`\n${'='.repeat(80)}`);
    console.log(`  ${config.name}`);
    console.log(`${'='.repeat(80)}\n`);

    for (const agentCount of config.agentCounts) {
      for (const messageSize of config.messageSizes) {
        for (const latency of config.latencies) {
          console.log(`\nBenchmarking: ${agentCount} agents, ${messageSize}B messages, ${latency}ms latency`);

          // Run QUIC benchmark
          await this.runProtocolBenchmark('QUIC', agentCount, messageSize, latency, config.duration);

          // Run HTTP/2 benchmark
          await this.runProtocolBenchmark('HTTP/2', agentCount, messageSize, latency, config.duration);

          // Run WebSocket benchmark
          await this.runProtocolBenchmark('WebSocket', agentCount, messageSize, latency, config.duration);
        }
      }
    }

    this.printResults();
    await this.exportResults();
  }

  private async runProtocolBenchmark(
    protocol: 'QUIC' | 'HTTP/2' | 'WebSocket',
    agentCount: number,
    messageSize: number,
    latency: number,
    duration: number
  ): Promise<void> {
    let benchmark: any;
    let port: number;

    if (protocol === 'QUIC') {
      benchmark = new QUICBenchmark();
      port = 4433;
    } else if (protocol === 'HTTP/2') {
      benchmark = new HTTP2Benchmark();
      port = 8443;
    } else {
      benchmark = new WebSocketBenchmark();
      port = 8080;
    }

    try {
      await benchmark.setup(port);

      // Warmup
      console.log(`  [${protocol}] Warming up...`);
      await benchmark.benchmarkConnectionEstablishment(5);

      // Connection establishment
      const connTime = await LatencySimulator.simulate(
        latency,
        () => benchmark.benchmarkConnectionEstablishment(agentCount)
      );

      // Message throughput
      const throughput = await LatencySimulator.simulate(
        latency,
        () => benchmark.benchmarkMessageThroughput(agentCount, messageSize, duration)
      );

      // Resource usage
      const resources = await benchmark.measureResourceUsage?.() || { memory: 0, cpu: 0 };

      // Stream efficiency (QUIC only)
      let streamEfficiency = 0;
      if (protocol === 'QUIC' && benchmark.benchmarkStreamMultiplexing) {
        streamEfficiency = await benchmark.benchmarkStreamMultiplexing(agentCount);
      }

      const result: BenchmarkResult = {
        protocol,
        agentCount,
        messageSize,
        networkLatency: latency,
        metrics: {
          connectionTime: connTime,
          throughput,
          latency: latency,
          memoryUsage: resources.memory,
          cpuUsage: resources.cpu,
          streamEfficiency,
          errorRate: 0,
          bandwidthUtilization: (throughput * messageSize * 8) / 1000000, // Mbps
        },
        timestamp: Date.now(),
      };

      this.results.push(result);

      console.log(`  [${protocol}] ✓ Complete`);
      console.log(`    - Connection Time: ${connTime.toFixed(2)}ms`);
      console.log(`    - Throughput: ${throughput.toFixed(0)} msg/s`);
      console.log(`    - Bandwidth: ${result.metrics.bandwidthUtilization.toFixed(2)} Mbps`);
      console.log(`    - Memory: ${resources.memory.toFixed(2)} MB`);

    } finally {
      await benchmark.cleanup();
    }
  }

  private printResults(): void {
    console.log(`\n${'='.repeat(80)}`);
    console.log(`  BENCHMARK RESULTS SUMMARY`);
    console.log(`${'='.repeat(80)}\n`);

    const grouped = this.groupResultsByScenario();

    for (const [scenario, results] of Object.entries(grouped)) {
      console.log(`\n${scenario}:`);

      const quic = results.find(r => r.protocol === 'QUIC');
      const http2 = results.find(r => r.protocol === 'HTTP/2');
      const ws = results.find(r => r.protocol === 'WebSocket');

      if (quic && http2) {
        const improvement = ((http2.metrics.connectionTime - quic.metrics.connectionTime) / http2.metrics.connectionTime * 100);
        console.log(`  QUIC vs HTTP/2 connection time improvement: ${improvement.toFixed(1)}%`);
      }

      if (quic && ws) {
        const improvement = ((ws.metrics.connectionTime - quic.metrics.connectionTime) / ws.metrics.connectionTime * 100);
        console.log(`  QUIC vs WebSocket connection time improvement: ${improvement.toFixed(1)}%`);
      }
    }
  }

  private groupResultsByScenario(): Record<string, BenchmarkResult[]> {
    const grouped: Record<string, BenchmarkResult[]> = {};

    for (const result of this.results) {
      const key = `${result.agentCount} agents, ${result.messageSize}B, ${result.networkLatency}ms`;
      if (!grouped[key]) {
        grouped[key] = [];
      }
      grouped[key].push(result);
    }

    return grouped;
  }

  private async exportResults(): Promise<void> {
    const fs = await import('fs/promises');
    await fs.writeFile(
      '/workspaces/agentic-flow/docs/benchmarks/results.json',
      JSON.stringify(this.results, null, 2)
    );
    console.log('\n✓ Results exported to docs/benchmarks/results.json');
  }
}

// ============================================================================
// Main Execution
// ============================================================================

if (require.main === module) {
  const runner = new BenchmarkRunner();

  runner.runFullSuite({
    name: 'QUIC vs HTTP/2 vs WebSocket Performance Comparison',
    agentCounts: [10, 100, 1000],
    messageSizes: [1024, 10240, 102400],
    latencies: [0, 50, 100],
    duration: 10,
    warmupRounds: 3,
  }).catch(console.error);
}

export { BenchmarkRunner, QUICBenchmark, HTTP2Benchmark, WebSocketBenchmark };
