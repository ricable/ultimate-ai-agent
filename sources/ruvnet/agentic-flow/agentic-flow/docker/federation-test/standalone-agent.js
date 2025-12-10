/**
 * Standalone Agent for Docker
 * Uses built npm package
 */

import { FederationHubClient } from '../../dist/federation/FederationHubClient.js';
import { SecurityManager } from '../../dist/federation/SecurityManager.js';

const AGENT_TYPE = process.env.AGENT_TYPE || 'researcher';
const AGENT_ID = process.env.AGENT_ID || `${AGENT_TYPE}-${Date.now()}`;
const TENANT_ID = process.env.TENANT_ID || 'test-collaboration';
const HUB_ENDPOINT = process.env.HUB_ENDPOINT || 'ws://federation-hub:8443';

class CollaborativeAgent {
  constructor(agentId, agentType, tenantId, hubEndpoint) {
    this.agentId = agentId;
    this.agentType = agentType;
    this.tenantId = tenantId;
    this.hubEndpoint = hubEndpoint;
    this.security = new SecurityManager();
    this.episodes = [];
    this.iteration = 0;
  }

  async initialize() {
    console.log(`\n🤖 Initializing ${this.agentType} agent: ${this.agentId}`);
    console.log(`   Tenant: ${this.tenantId}`);
    console.log(`   Hub: ${this.hubEndpoint}`);

    // Create authentication token
    const token = await this.security.createAgentToken({
      agentId: this.agentId,
      tenantId: this.tenantId,
      expiresAt: Date.now() + 300000 // 5 minutes
    });

    // Create hub client
    this.client = new FederationHubClient({
      endpoint: this.hubEndpoint,
      agentId: this.agentId,
      tenantId: this.tenantId,
      token
    });

    // Connect to hub
    await this.client.connect();
    console.log(`✅ Connected to federation hub\n`);
  }

  async collaborate() {
    console.log(`🚀 Starting collaboration loop...\n`);

    // Run collaboration for 60 seconds
    const endTime = Date.now() + 60000;

    while (Date.now() < endTime) {
      this.iteration++;

      try {
        // Execute agent-specific task
        const episode = await this.executeTask();
        this.episodes.push(episode);

        // Sync with hub
        await this.syncWithHub();

        console.log(`[${this.agentType}] Iteration ${this.iteration} complete (reward: ${episode.reward.toFixed(2)})`);

        // Wait before next iteration
        await new Promise(resolve => setTimeout(resolve, 5000));

      } catch (error) {
        console.error(`[${this.agentType}] Error:`, error.message);
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }

    console.log(`\n✅ Collaboration complete!`);
    this.printSummary();
  }

  async executeTask() {
    const startTime = Date.now();

    let task, output, reward, critique;

    switch (this.agentType) {
      case 'researcher':
        task = 'research-patterns';
        output = ['Optimal async patterns', 'Performance bottlenecks', 'Reusable structures'][Math.floor(Math.random() * 3)];
        reward = 0.75 + Math.random() * 0.2;
        critique = 'Found collaboration patterns';
        break;

      case 'coder':
        task = 'implement-solution';
        output = ['Async fetcher with retry', 'Validation middleware', 'Caching layer'][Math.floor(Math.random() * 3)];
        reward = 0.80 + Math.random() * 0.15;
        critique = 'Code follows best practices';
        break;

      case 'tester':
        task = 'validate-work';
        output = ['Unit tests: 45 passed', 'Integration: 12 passed', 'E2E: 8 passed'][Math.floor(Math.random() * 3)];
        reward = 0.85 + Math.random() * 0.1;
        critique = 'Tests passed, learned from failures';
        break;

      case 'reviewer':
        task = 'quality-check';
        output = ['Quality: A+', 'Security: No issues', 'Performance: Good'][Math.floor(Math.random() * 3)];
        reward = 0.90 + Math.random() * 0.05;
        critique = 'Quality improved from team learning';
        break;

      default:
        task = 'generic-task';
        output = 'Task completed';
        reward = 0.7 + Math.random() * 0.25;
        critique = 'Completed successfully';
    }

    const latencyMs = Date.now() - startTime;

    return {
      sessionId: this.agentId,
      task,
      input: `Execute task ${this.iteration}`,
      output,
      reward,
      critique,
      success: reward >= 0.7,
      tokensUsed: Math.floor(Math.random() * 1000) + 500,
      latencyMs
    };
  }

  async syncWithHub() {
    await this.client.sync({});
  }

  printSummary() {
    const avgReward = this.episodes.reduce((sum, e) => sum + e.reward, 0) / this.episodes.length;
    const successRate = this.episodes.filter(e => e.success).length / this.episodes.length;

    console.log(`\n📊 Agent Summary:`);
    console.log(`   Type: ${this.agentType}`);
    console.log(`   ID: ${this.agentId}`);
    console.log(`   Iterations: ${this.iteration}`);
    console.log(`   Average Reward: ${avgReward.toFixed(3)}`);
    console.log(`   Success Rate: ${(successRate * 100).toFixed(1)}%`);
  }

  async cleanup() {
    console.log(`\n🧹 Cleaning up agent...`);
    await this.client.disconnect();
    console.log(`✅ Agent disconnected`);
  }
}

async function main() {
  const agent = new CollaborativeAgent(
    AGENT_ID,
    AGENT_TYPE,
    TENANT_ID,
    HUB_ENDPOINT
  );

  try {
    await agent.initialize();
    await agent.collaborate();
  } catch (error) {
    console.error('Agent failed:', error);
    process.exit(1);
  } finally {
    await agent.cleanup();
  }
}

main();
