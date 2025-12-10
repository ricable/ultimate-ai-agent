/**
 * Cluster Status Monitor
 * Displays real-time status of all nodes in the cluster
 */

import 'dotenv/config';
import Redis from 'ioredis';

const redis = new Redis({
  host: process.env.REDIS_HOST || 'localhost',
  port: parseInt(process.env.REDIS_PORT) || 6379,
  password: process.env.REDIS_PASSWORD || undefined
});

/**
 * Get all cluster nodes
 */
async function getClusterNodes() {
  const nodeIds = await redis.smembers('cluster:nodes');
  const nodes = [];

  for (const nodeId of nodeIds) {
    const nodeInfoStr = await redis.get(`cluster:node:${nodeId}`);
    const heartbeat = await redis.get(`cluster:heartbeat:${nodeId}`);

    if (nodeInfoStr) {
      const node = JSON.parse(nodeInfoStr);
      node.heartbeatTimestamp = heartbeat ? parseInt(heartbeat) : null;
      node.timeSinceHeartbeat = heartbeat
        ? Date.now() - parseInt(heartbeat)
        : null;

      nodes.push(node);
    }
  }

  return nodes;
}

/**
 * Get cluster statistics
 */
async function getClusterStats(nodes) {
  const stats = {
    totalNodes: nodes.length,
    onlineNodes: 0,
    offlineNodes: 0,
    totalCores: 0,
    totalRAM: 0,
    byRole: {},
    byType: {},
    capabilities: new Set()
  };

  for (const node of nodes) {
    // Status
    const isOnline = node.timeSinceHeartbeat && node.timeSinceHeartbeat < 120000; // 2 minutes
    if (isOnline) {
      stats.onlineNodes++;
    } else {
      stats.offlineNodes++;
    }

    // Resources
    stats.totalCores += node.hardware.cores || 0;
    stats.totalRAM += node.hardware.ramGB || 0;

    // Role distribution
    stats.byRole[node.role] = (stats.byRole[node.role] || 0) + 1;

    // Type distribution
    stats.byType[node.type] = (stats.byType[node.type] || 0) + 1;

    // Capabilities
    if (node.capabilities) {
      node.capabilities.forEach(cap => stats.capabilities.add(cap));
    }
  }

  return stats;
}

/**
 * Display cluster status
 */
async function displayStatus() {
  console.clear();
  console.log('='.repeat(80));
  console.log('🌐  DISTRIBUTED CLUSTER STATUS'.padStart(50));
  console.log('='.repeat(80));
  console.log();

  try {
    const nodes = await getClusterNodes();

    if (nodes.length === 0) {
      console.log('⚠️  No nodes registered in cluster');
      console.log('\nRun: npm run cluster:init on each node to register');
      return;
    }

    const stats = await getClusterStats(nodes);

    // Overall stats
    console.log('📊 CLUSTER OVERVIEW');
    console.log('-'.repeat(80));
    console.log(`Total Nodes:     ${stats.totalNodes}`);
    console.log(`Online:          🟢 ${stats.onlineNodes}`);
    console.log(`Offline:         🔴 ${stats.offlineNodes}`);
    console.log(`Total Cores:     ${stats.totalCores}`);
    console.log(`Total RAM:       ${stats.totalRAM.toFixed(1)} GB`);
    console.log();

    // Distribution
    console.log('📋 DISTRIBUTION');
    console.log('-'.repeat(80));
    console.log('By Role:');
    Object.entries(stats.byRole).forEach(([role, count]) => {
      console.log(`  ${role.padEnd(30)} ${count}`);
    });
    console.log();
    console.log('By Hardware Type:');
    Object.entries(stats.byType).forEach(([type, count]) => {
      console.log(`  ${type.padEnd(30)} ${count}`);
    });
    console.log();

    // Capabilities
    console.log('🔧 CLUSTER CAPABILITIES');
    console.log('-'.repeat(80));
    const capArray = Array.from(stats.capabilities);
    for (let i = 0; i < capArray.length; i += 3) {
      console.log(capArray.slice(i, i + 3).map(c => c.padEnd(26)).join(''));
    }
    console.log();

    // Node details
    console.log('🖥️  NODES');
    console.log('-'.repeat(80));
    console.log(
      'Status'.padEnd(8) +
      'Name'.padEnd(24) +
      'Type'.padEnd(20) +
      'Role'.padEnd(20) +
      'Resources'
    );
    console.log('-'.repeat(80));

    nodes
      .sort((a, b) => {
        // Sort by status (online first), then by role priority
        const aOnline = a.timeSinceHeartbeat && a.timeSinceHeartbeat < 120000;
        const bOnline = b.timeSinceHeartbeat && b.timeSinceHeartbeat < 120000;

        if (aOnline !== bOnline) return bOnline ? 1 : -1;

        const rolePriority = {
          'super-coordinator': 1,
          'coordinator': 2,
          'general-worker': 3,
          'worker': 4,
          'edge-worker': 5
        };

        return (rolePriority[a.role] || 99) - (rolePriority[b.role] || 99);
      })
      .forEach(node => {
        const isOnline = node.timeSinceHeartbeat && node.timeSinceHeartbeat < 120000;
        const status = isOnline ? '🟢' : '🔴';
        const name = node.name.slice(0, 22);
        const type = node.type.slice(0, 18);
        const role = node.role.slice(0, 18);
        const resources = `${node.hardware.cores}c/${node.hardware.ramGB.toFixed(0)}GB`;

        console.log(
          `${status}      ${name.padEnd(24)}${type.padEnd(20)}${role.padEnd(20)}${resources}`
        );

        if (!isOnline && node.lastHeartbeat) {
          const offline = Math.floor((Date.now() - node.lastHeartbeat) / 1000);
          console.log(`        ⚠️  Offline for ${offline}s`);
        }
      });

    console.log();
    console.log('='.repeat(80));
    console.log(`Last updated: ${new Date().toLocaleTimeString()}`);
    console.log('Press Ctrl+C to exit');

  } catch (error) {
    console.error('❌ Error:', error.message);
    console.log('\nMake sure Redis is running:');
    console.log('  docker run -d -p 6379:6379 redis:alpine');
  }
}

/**
 * Watch mode - continuous updates
 */
async function watchStatus() {
  console.log('Starting cluster status monitor (updating every 5s)...\n');

  // Initial display
  await displayStatus();

  // Update every 5 seconds
  setInterval(async () => {
    await displayStatus();
  }, 5000);
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const watchMode = process.argv.includes('--watch') || process.argv.includes('-w');

  if (watchMode) {
    watchStatus().catch(console.error);
  } else {
    displayStatus()
      .then(() => process.exit(0))
      .catch(err => {
        console.error(err);
        process.exit(1);
      });
  }
}

export { getClusterNodes, getClusterStats, displayStatus };
