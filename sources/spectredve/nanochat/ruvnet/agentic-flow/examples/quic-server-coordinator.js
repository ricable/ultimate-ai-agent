#!/usr/bin/env node
/**
 * QUIC Server - Central Coordinator for Agent Swarms
 *
 * This server acts as the central coordinator for distributed agent swarms,
 * handling high-frequency task distribution and real-time state synchronization.
 *
 * Performance benefits:
 * - Handles 100+ concurrent agent connections on a single QUIC socket
 * - 0-RTT reconnection for returning agents (instant task assignment)
 * - Stream multiplexing eliminates head-of-line blocking
 * - Built-in TLS 1.3 security (no unencrypted mode)
 *
 * Usage:
 *   node examples/quic-server-coordinator.js
 *
 * Or via CLI:
 *   npx agentic-flow quic --port 4433
 */

import { QuicServer } from 'agentic-flow/transport/quic';
import { getQuicConfig } from 'agentic-flow/dist/config/quic.js';

class AgentCoordinator {
  constructor() {
    this.server = null;
    this.agents = new Map(); // Active agent registry
    this.tasks = []; // Task queue
    this.metrics = {
      totalTasks: 0,
      completedTasks: 0,
      activeAgents: 0,
      startTime: Date.now()
    };
  }

  async start(port = 4433) {
    console.log('🚀 Starting QUIC Agent Coordinator...\n');

    // Initialize QUIC server
    this.server = new QuicServer({
      host: '0.0.0.0',
      port: port,
      maxConnections: 1000,        // Support up to 1000 agents
      maxConcurrentStreams: 100,   // 100 streams per agent
      certPath: './certs/cert.pem',
      keyPath: './certs/key.pem'
    });

    await this.server.initialize();
    await this.server.listen();

    console.log(`✅ QUIC server listening on port ${port}`);
    console.log(`📊 Max agents: ${this.server.config.maxConnections}`);
    console.log(`⚡ Streams per agent: ${this.server.config.maxConcurrentStreams}`);
    console.log(`🔒 TLS 1.3 enabled (secure by default)\n`);

    // Start metrics reporting
    this.startMetricsReporting();

    // Handle graceful shutdown
    process.on('SIGINT', () => this.shutdown());
    process.on('SIGTERM', () => this.shutdown());

    console.log('💡 Agent coordinator ready for connections!\n');
    console.log('Example client connection:');
    console.log('  node examples/quic-swarm-coordination.js\n');
  }

  // Register new agent
  registerAgent(agentId, connectionId) {
    this.agents.set(agentId, {
      id: agentId,
      connectionId,
      status: 'idle',
      tasksCompleted: 0,
      connectedAt: Date.now(),
      lastSeen: Date.now()
    });

    this.metrics.activeAgents = this.agents.size;

    console.log(`🤖 Agent registered: ${agentId}`);
    console.log(`   Total agents: ${this.agents.size}`);
    console.log(`   Connection: ${connectionId} (QUIC 0-RTT)\n`);
  }

  // Assign task to agent
  async assignTask(agentId, task) {
    const agent = this.agents.get(agentId);

    if (!agent) {
      console.warn(`⚠️  Unknown agent: ${agentId}`);
      return;
    }

    agent.status = 'working';
    agent.lastSeen = Date.now();

    this.metrics.totalTasks++;

    console.log(`📋 Task assigned to ${agentId}:`);
    console.log(`   Type: ${task.type}`);
    console.log(`   Data: ${JSON.stringify(task.data).slice(0, 50)}...`);
    console.log(`   Stream: QUIC multiplexed (no blocking)\n`);

    // Simulate task processing via QUIC stream
    // In real implementation, this would send via QUIC transport
    return {
      taskId: `task-${this.metrics.totalTasks}`,
      agentId,
      status: 'assigned',
      timestamp: Date.now()
    };
  }

  // Mark task as complete
  completeTask(agentId, taskId, result) {
    const agent = this.agents.get(agentId);

    if (agent) {
      agent.status = 'idle';
      agent.tasksCompleted++;
      agent.lastSeen = Date.now();

      this.metrics.completedTasks++;

      console.log(`✅ Task completed: ${taskId}`);
      console.log(`   Agent: ${agentId}`);
      console.log(`   Result: ${result ? 'success' : 'failed'}`);
      console.log(`   Agent total: ${agent.tasksCompleted} tasks\n`);
    }
  }

  // Handle agent disconnection
  disconnectAgent(agentId) {
    const agent = this.agents.get(agentId);

    if (agent) {
      const sessionDuration = ((Date.now() - agent.connectedAt) / 1000).toFixed(1);

      console.log(`👋 Agent disconnected: ${agentId}`);
      console.log(`   Session duration: ${sessionDuration}s`);
      console.log(`   Tasks completed: ${agent.tasksCompleted}\n`);

      this.agents.delete(agentId);
      this.metrics.activeAgents = this.agents.size;
    }
  }

  // Periodic metrics reporting
  startMetricsReporting() {
    setInterval(() => {
      const uptime = ((Date.now() - this.metrics.startTime) / 1000).toFixed(0);
      const avgTaskRate = this.metrics.totalTasks / (uptime || 1);

      console.log('═══════════════════════════════════════════════════════════');
      console.log(`📊 Coordinator Metrics (uptime: ${uptime}s)`);
      console.log('═══════════════════════════════════════════════════════════');
      console.log(`   Active agents: ${this.metrics.activeAgents}`);
      console.log(`   Total tasks: ${this.metrics.totalTasks}`);
      console.log(`   Completed: ${this.metrics.completedTasks}`);
      console.log(`   Task rate: ${avgTaskRate.toFixed(2)} tasks/sec`);

      // QUIC server stats
      const stats = this.server.getStats();
      console.log(`\n   QUIC connections: ${stats.activeConnections}`);
      console.log(`   QUIC streams: ${stats.activeStreams}`);
      console.log(`   RTT latency: ${stats.rttMs}ms`);
      console.log('═══════════════════════════════════════════════════════════\n');

    }, 10000); // Report every 10 seconds
  }

  // Graceful shutdown
  async shutdown() {
    console.log('\n🛑 Shutting down coordinator...\n');

    // Final metrics
    const uptime = ((Date.now() - this.metrics.startTime) / 1000).toFixed(1);

    console.log('📊 Final Statistics:');
    console.log(`   Uptime: ${uptime}s`);
    console.log(`   Total agents: ${this.agents.size}`);
    console.log(`   Total tasks: ${this.metrics.totalTasks}`);
    console.log(`   Completed: ${this.metrics.completedTasks}`);
    console.log(`   Success rate: ${((this.metrics.completedTasks / this.metrics.totalTasks) * 100).toFixed(1)}%\n`);

    // Close QUIC server
    if (this.server) {
      await this.server.stop();
      console.log('✅ QUIC server stopped\n');
    }

    process.exit(0);
  }
}

// Start coordinator
async function main() {
  const port = process.env.QUIC_PORT || 4433;

  console.log('═══════════════════════════════════════════════════════════');
  console.log('  QUIC Agent Coordinator - Agentic Flow');
  console.log('═══════════════════════════════════════════════════════════\n');

  console.log('💡 Use Case: Central hub for distributed agent swarms');
  console.log('   - High-frequency task distribution');
  console.log('   - Real-time agent state synchronization');
  console.log('   - Connection migration support (mobile agents)');
  console.log('   - Stream multiplexing (100+ concurrent per agent)\n');

  const coordinator = new AgentCoordinator();
  await coordinator.start(port);

  // Simulate some agent activity (for demo purposes)
  setTimeout(() => {
    // Simulate agent connections
    coordinator.registerAgent('agent-1', 'conn-1');
    coordinator.registerAgent('agent-2', 'conn-2');
    coordinator.registerAgent('agent-3', 'conn-3');

    // Assign tasks
    setTimeout(async () => {
      await coordinator.assignTask('agent-1', {
        type: 'code_review',
        data: { file: 'src/App.tsx', lines: 250 }
      });

      await coordinator.assignTask('agent-2', {
        type: 'test_generation',
        data: { file: 'src/utils.ts', coverage: 90 }
      });

      await coordinator.assignTask('agent-3', {
        type: 'refactor',
        data: { file: 'src/api.ts', pattern: 'async-await' }
      });

      // Complete tasks after 2 seconds
      setTimeout(() => {
        coordinator.completeTask('agent-1', 'task-1', true);
        coordinator.completeTask('agent-2', 'task-2', true);
        coordinator.completeTask('agent-3', 'task-3', true);
      }, 2000);

    }, 1000);

  }, 2000);
}

main().catch(error => {
  console.error('❌ Fatal error:', error);
  process.exit(1);
});
