/**
 * Distributed Cluster Initialization
 * Sets up multi-node agent cluster with hardware auto-detection
 */

import 'dotenv/config';
import { readFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { machineIdSync } from 'node-machine-id';
import os from 'os';
import Redis from 'ioredis';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

class ClusterInitializer {
  constructor() {
    this.nodeId = machineIdSync();
    this.redis = null;
  }

  /**
   * Detect hardware configuration
   */
  async detectHardware() {
    const platform = os.platform();
    const arch = os.arch();
    const cpus = os.cpus();
    const totalRAM = os.totalmem() / (1024 ** 3); // GB

    console.log('\n🔍 Detecting Hardware...');
    console.log('Platform:', platform);
    console.log('Architecture:', arch);
    console.log('CPU:', cpus[0]?.model || 'Unknown');
    console.log('Cores:', cpus.length);
    console.log('RAM:', totalRAM.toFixed(1), 'GB');

    // Determine hardware type
    let hardwareType = null;

    if (platform === 'darwin' && arch === 'arm64') {
      // Mac Silicon detection
      if (totalRAM >= 100) {
        hardwareType = 'macbook-m3-max';
        console.log('✅ Detected: MacBook M3 Max (or similar high-end)');
      } else if (totalRAM >= 50) {
        hardwareType = 'mac-studio-m1';
        console.log('✅ Detected: Mac Studio M1 Ultra');
      } else {
        hardwareType = 'mac-studio-m1'; // Default for Mac
        console.log('✅ Detected: Mac (assuming Studio M1)');
      }
    } else if (platform === 'linux') {
      if (arch === 'arm' || arch === 'arm64') {
        if (totalRAM < 8) {
          hardwareType = 'raspberry-pi';
          console.log('✅ Detected: Raspberry Pi');
        } else {
          hardwareType = 'intel-nuc'; // Or other ARM server
          console.log('✅ Detected: ARM Linux (treating as NUC)');
        }
      } else {
        // x86_64 Linux
        if (cpus[0]?.model?.includes('Intel')) {
          hardwareType = 'intel-nuc';
          console.log('✅ Detected: Intel NUC (or compatible)');
        } else {
          hardwareType = 'intel-nuc'; // Default
          console.log('✅ Detected: x86_64 Linux (treating as NUC)');
        }
      }
    }

    return {
      type: hardwareType,
      platform,
      arch,
      cpuModel: cpus[0]?.model || 'Unknown',
      cores: cpus.length,
      ramGB: totalRAM
    };
  }

  /**
   * Load hardware-specific configuration
   */
  loadHardwareConfig(hardwareType) {
    const configPath = join(__dirname, '../../config/hardware', `${hardwareType}.json`);

    if (!existsSync(configPath)) {
      throw new Error(`Hardware config not found: ${configPath}`);
    }

    const config = JSON.parse(readFileSync(configPath, 'utf8'));
    console.log(`\n📋 Loaded config: ${hardwareType}`);
    console.log('Role:', config.orchestration.role);
    console.log('Capabilities:', config.orchestration.capabilities.join(', '));

    return config;
  }

  /**
   * Initialize Redis connection for cluster coordination
   */
  async initRedis() {
    const redisHost = process.env.REDIS_HOST || 'localhost';
    const redisPort = parseInt(process.env.REDIS_PORT) || 6379;

    console.log(`\n🔗 Connecting to Redis: ${redisHost}:${redisPort}`);

    this.redis = new Redis({
      host: redisHost,
      port: redisPort,
      password: process.env.REDIS_PASSWORD || undefined,
      retryStrategy: (times) => {
        if (times > 3) {
          console.error('❌ Redis connection failed after 3 retries');
          return null;
        }
        return Math.min(times * 1000, 3000);
      }
    });

    return new Promise((resolve, reject) => {
      this.redis.on('connect', () => {
        console.log('✅ Connected to Redis');
        resolve();
      });

      this.redis.on('error', (err) => {
        console.error('❌ Redis error:', err.message);
        reject(err);
      });
    });
  }

  /**
   * Register node in cluster
   */
  async registerNode(hardwareConfig, detectedHW) {
    const nodeInfo = {
      id: this.nodeId,
      name: process.env.NODE_NAME || `node-${this.nodeId.slice(0, 8)}`,
      type: hardwareConfig.hardware.type,
      role: hardwareConfig.orchestration.role,
      capabilities: hardwareConfig.orchestration.capabilities,
      hardware: {
        platform: detectedHW.platform,
        arch: detectedHW.arch,
        cpuModel: detectedHW.cpuModel,
        cores: detectedHW.cores,
        ramGB: detectedHW.ramGB
      },
      status: 'online',
      registeredAt: Date.now(),
      lastHeartbeat: Date.now()
    };

    await this.redis.set(
      `cluster:node:${this.nodeId}`,
      JSON.stringify(nodeInfo),
      'EX',
      300 // Expire after 5 minutes (heartbeat will refresh)
    );

    await this.redis.sadd('cluster:nodes', this.nodeId);

    console.log('\n✅ Node registered in cluster:');
    console.log('Node ID:', nodeInfo.id);
    console.log('Node Name:', nodeInfo.name);
    console.log('Role:', nodeInfo.role);

    return nodeInfo;
  }

  /**
   * Start heartbeat
   */
  startHeartbeat() {
    console.log('\n💓 Starting heartbeat (every 60s)');

    setInterval(async () => {
      try {
        await this.redis.set(
          `cluster:heartbeat:${this.nodeId}`,
          Date.now().toString(),
          'EX',
          120
        );

        // Update node info
        const nodeKey = `cluster:node:${this.nodeId}`;
        const nodeInfoStr = await this.redis.get(nodeKey);

        if (nodeInfoStr) {
          const nodeInfo = JSON.parse(nodeInfoStr);
          nodeInfo.lastHeartbeat = Date.now();
          nodeInfo.status = 'online';

          await this.redis.set(
            nodeKey,
            JSON.stringify(nodeInfo),
            'EX',
            300
          );
        }
      } catch (error) {
        console.error('❌ Heartbeat failed:', error.message);
      }
    }, 60000);
  }

  /**
   * Display cluster topology
   */
  async displayCluster() {
    const nodeIds = await this.redis.smembers('cluster:nodes');

    console.log('\n' + '='.repeat(60));
    console.log('🌐 CLUSTER TOPOLOGY');
    console.log('='.repeat(60));

    for (const nodeId of nodeIds) {
      const nodeInfoStr = await this.redis.get(`cluster:node:${nodeId}`);

      if (nodeInfoStr) {
        const node = JSON.parse(nodeInfoStr);
        const isCurrentNode = node.id === this.nodeId;

        console.log(`\n${isCurrentNode ? '🟢' : '🔵'} ${node.name} ${isCurrentNode ? '(THIS NODE)' : ''}`);
        console.log(`   ID: ${node.id.slice(0, 16)}...`);
        console.log(`   Type: ${node.type}`);
        console.log(`   Role: ${node.role}`);
        console.log(`   Status: ${node.status}`);
        console.log(`   Hardware: ${node.hardware.cores} cores, ${node.hardware.ramGB.toFixed(1)}GB RAM`);
        console.log(`   Capabilities: ${node.capabilities.slice(0, 3).join(', ')}...`);
      }
    }

    console.log('\n' + '='.repeat(60));
    console.log(`Total Nodes: ${nodeIds.length}`);
    console.log('='.repeat(60));
  }

  /**
   * Main initialization
   */
  async initialize() {
    console.log('🚀 Initializing Distributed Cluster');
    console.log('====================================\n');

    try {
      // Step 1: Detect hardware
      const detectedHW = await this.detectHardware();

      if (!detectedHW.type) {
        throw new Error('Unable to detect hardware type');
      }

      // Step 2: Load configuration
      const config = this.loadHardwareConfig(detectedHW.type);

      // Step 3: Connect to Redis
      await this.initRedis();

      // Step 4: Register node
      const nodeInfo = await this.registerNode(config, detectedHW);

      // Step 5: Start heartbeat
      this.startHeartbeat();

      // Step 6: Display cluster
      await this.displayCluster();

      console.log('\n✅ Cluster initialization complete!');
      console.log('\nNext steps:');
      console.log('1. Start agent orchestrator: npm run start:quad');
      console.log('2. Check cluster status: npm run cluster:status');
      console.log('3. Start inference: gaianet start (or llamaedge)');

      // Keep process alive
      console.log('\n💓 Heartbeat active. Press Ctrl+C to exit.\n');

    } catch (error) {
      console.error('\n❌ Initialization failed:', error.message);
      process.exit(1);
    }
  }
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const initializer = new ClusterInitializer();
  initializer.initialize().catch(console.error);
}

export { ClusterInitializer };
