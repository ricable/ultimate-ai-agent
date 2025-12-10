/**
 * Multi-Agent Collaboration Demo (In-Process Simulation)
 *
 * Simulates 5 agents collaborating through a shared federation hub
 * without requiring Docker or full AgentDB integration.
 */

import { FederationHubServer } from '../../src/federation/FederationHubServer.js';
import { FederationHubClient } from '../../src/federation/FederationHubClient.js';
import { SecurityManager } from '../../src/federation/SecurityManager.js';

interface SimulatedAgentConfig {
  agentId: string;
  agentType: string;
  tenantId: string;
}

class SimulatedAgent {
  private client!: FederationHubClient;
  private security: SecurityManager;
  private iteration: number = 0;
  private episodes: any[] = [];

  constructor(private config: SimulatedAgentConfig) {
    this.security = new SecurityManager();
  }

  async initialize(hubEndpoint: string): Promise<void> {
    console.log(`🤖 [${this.config.agentType}] Initializing ${this.config.agentId}...`);

    // Create token
    const token = await this.security.createAgentToken({
      agentId: this.config.agentId,
      tenantId: this.config.tenantId,
      expiresAt: Date.now() + 300000
    });

    // Create client
    this.client = new FederationHubClient({
      endpoint: hubEndpoint,
      agentId: this.config.agentId,
      tenantId: this.config.tenantId,
      token
    });

    // Connect
    await this.client.connect();
    console.log(`✅ [${this.config.agentType}] Connected to hub`);
  }

  async collaborate(iterations: number): Promise<void> {
    for (let i = 0; i < iterations; i++) {
      this.iteration++;

      // Execute task
      const episode = this.executeTask();
      this.episodes.push(episode);

      // Sync with hub (simulated)
      await this.syncWithHub(episode);

      console.log(`[${this.config.agentType}] Iteration ${this.iteration}: ${episode.task} (reward: ${episode.reward.toFixed(2)})`);

      // Wait between iterations
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }

  private executeTask(): any {
    const taskTypes: Record<string, any> = {
      researcher: {
        task: 'research-patterns',
        output: 'Found optimal design patterns',
        baseReward: 0.75
      },
      coder: {
        task: 'implement-solution',
        output: 'Implemented feature with tests',
        baseReward: 0.80
      },
      tester: {
        task: 'validate-work',
        output: 'All tests passed',
        baseReward: 0.85
      },
      reviewer: {
        task: 'quality-check',
        output: 'Code quality: A+',
        baseReward: 0.90
      }
    };

    const taskType = taskTypes[this.config.agentType] || taskTypes.researcher;

    return {
      sessionId: this.config.agentId,
      task: `${taskType.task}-${this.iteration}`,
      input: `Input for iteration ${this.iteration}`,
      output: taskType.output,
      reward: taskType.baseReward + Math.random() * 0.15,
      success: true,
      tokensUsed: Math.floor(Math.random() * 1000) + 500,
      latencyMs: Math.floor(Math.random() * 100) + 50
    };
  }

  private async syncWithHub(episode: any): Promise<void> {
    // Simulate sync (in production, would use AgentDB)
    try {
      await this.client.sync({} as any);
    } catch (error) {
      // Ignore sync errors in demo
    }
  }

  async cleanup(): Promise<void> {
    await this.client.disconnect();

    const avgReward = this.episodes.reduce((sum, e) => sum + e.reward, 0) / this.episodes.length;
    console.log(`📊 [${this.config.agentType}] Summary: ${this.iteration} iterations, avg reward: ${avgReward.toFixed(3)}`);
  }
}

async function runMultiAgentDemo() {
  console.log('\n🌐 Federation Multi-Agent Collaboration Demo');
  console.log('='.repeat(50));
  console.log('');

  // Start federation hub
  console.log('🚀 Starting federation hub...');
  const hub = new FederationHubServer({
    port: 8443,
    dbPath: ':memory:',
    maxAgents: 100
  });

  await hub.start();
  console.log('✅ Hub started on port 8443\n');

  // Wait for hub to be ready
  await new Promise(resolve => setTimeout(resolve, 1000));

  // Create agents
  const agents = [
    new SimulatedAgent({
      agentId: 'researcher-001',
      agentType: 'researcher',
      tenantId: 'test-collaboration'
    }),
    new SimulatedAgent({
      agentId: 'coder-001',
      agentType: 'coder',
      tenantId: 'test-collaboration'
    }),
    new SimulatedAgent({
      agentId: 'tester-001',
      agentType: 'tester',
      tenantId: 'test-collaboration'
    }),
    new SimulatedAgent({
      agentId: 'reviewer-001',
      agentType: 'reviewer',
      tenantId: 'test-collaboration'
    }),
    new SimulatedAgent({
      agentId: 'isolated-001',
      agentType: 'researcher',
      tenantId: 'different-tenant'
    })
  ];

  // Initialize all agents
  console.log('🔌 Connecting agents to hub...');
  await Promise.all(
    agents.map(agent => agent.initialize('ws://localhost:8443'))
  );
  console.log('✅ All agents connected\n');

  // Wait for connections to stabilize
  await new Promise(resolve => setTimeout(resolve, 1000));

  // Run collaboration (5 iterations per agent)
  console.log('🤝 Starting collaboration...\n');

  await Promise.all(
    agents.map((agent, index) =>
      new Promise(resolve => {
        setTimeout(async () => {
          await agent.collaborate(5);
          resolve(null);
        }, index * 200); // Stagger starts
      })
    )
  );

  // Wait for final syncs
  await new Promise(resolve => setTimeout(resolve, 2000));

  // Cleanup agents
  console.log('\n🧹 Cleaning up agents...');
  await Promise.all(agents.map(agent => agent.cleanup()));

  // Get hub stats
  console.log('\n📊 Final Hub Statistics:');
  const stats = hub.getStats();
  console.log(`   Connected agents: ${stats.connectedAgents}`);
  console.log(`   Total episodes: ${stats.totalEpisodes}`);
  console.log(`   Active tenants: ${stats.tenants}`);
  console.log(`   Uptime: ${Math.floor(stats.uptime)}s`);

  // Validate results
  console.log('\n✅ Validation:');
  console.log(`   ${stats.totalEpisodes >= 20 ? '✓' : '✗'} Episodes stored: ${stats.totalEpisodes} (expected: ≥20)`);
  console.log(`   ${stats.tenants === 2 ? '✓' : '✗'} Tenants isolated: ${stats.tenants} (expected: 2)`);
  console.log(`   ${stats.connectedAgents === 0 ? '✓' : '✗'} Agents disconnected: ${stats.connectedAgents} (expected: 0)`);

  // Stop hub
  await hub.stop();

  console.log('\n🎉 Demo complete!\n');

  // Success criteria
  const success = stats.totalEpisodes >= 20 && stats.tenants === 2 && stats.connectedAgents === 0;

  if (success) {
    console.log('✅ All validation checks passed!');
  } else {
    console.log('❌ Some validation checks failed');
    process.exit(1);
  }
}

// Run demo
runMultiAgentDemo().catch((error) => {
  console.error('\n❌ Demo failed:', error);
  console.error(error.stack);
  process.exit(1);
});
