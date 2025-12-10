/**
 * Run Federation Test Agent
 *
 * Each agent performs a specific role in the collaboration:
 * - Researcher: Finds patterns and insights
 * - Coder: Implements solutions
 * - Tester: Validates work
 * - Reviewer: Quality checks
 */

import { FederationHubClient } from '../../src/federation/FederationHubClient.js';
import { SecurityManager } from '../../src/federation/SecurityManager.js';

const AGENT_TYPE = process.env.AGENT_TYPE || 'researcher';
const AGENT_ID = process.env.AGENT_ID || `${AGENT_TYPE}-${Date.now()}`;
const TENANT_ID = process.env.TENANT_ID || 'test-collaboration';
const HUB_ENDPOINT = process.env.HUB_ENDPOINT || 'ws://federation-hub:8443';
const TASK = process.env.TASK || 'collaborate';

interface Episode {
  sessionId: string;
  task: string;
  input: string;
  output: string;
  reward: number;
  critique?: string;
  success: boolean;
  tokensUsed: number;
  latencyMs: number;
}

class CollaborativeAgent {
  private client: FederationHubClient;
  private security: SecurityManager;
  private episodes: Episode[] = [];
  private iteration: number = 0;

  constructor(
    private agentId: string,
    private agentType: string,
    private tenantId: string,
    private hubEndpoint: string
  ) {
    this.security = new SecurityManager();
  }

  async initialize(): Promise<void> {
    console.log(`🤖 Initializing ${this.agentType} agent: ${this.agentId}`);
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
    console.log(`✅ Connected to federation hub`);
  }

  async collaborate(): Promise<void> {
    console.log(`\n🚀 Starting collaboration loop...`);

    // Run collaboration for 60 seconds
    const endTime = Date.now() + 60000;

    while (Date.now() < endTime) {
      this.iteration++;

      try {
        // Execute agent-specific task
        const episode = await this.executeTask();

        // Store episode
        this.episodes.push(episode);

        // Sync with hub (share learning with other agents)
        await this.syncWithHub();

        console.log(`[${this.agentType}] Iteration ${this.iteration} complete (reward: ${episode.reward.toFixed(2)})`);

        // Wait before next iteration
        await new Promise(resolve => setTimeout(resolve, 5000));

      } catch (error: any) {
        console.error(`[${this.agentType}] Error:`, error.message);
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }

    console.log(`\n✅ Collaboration complete!`);
    this.printSummary();
  }

  private async executeTask(): Promise<Episode> {
    const startTime = Date.now();

    // Different behavior based on agent type
    let task: string;
    let input: string;
    let output: string;
    let reward: number;
    let critique: string;

    switch (this.agentType) {
      case 'researcher':
        task = 'research-patterns';
        input = `Analyze codebase for iteration ${this.iteration}`;
        output = this.simulateResearch();
        reward = 0.75 + Math.random() * 0.2;
        critique = 'Found interesting patterns in agent collaboration';
        break;

      case 'coder':
        task = 'implement-solution';
        input = `Implement feature based on research ${this.iteration}`;
        output = this.simulateCoding();
        reward = 0.80 + Math.random() * 0.15;
        critique = 'Code follows best practices from past iterations';
        break;

      case 'tester':
        task = 'validate-work';
        input = `Test implementation ${this.iteration}`;
        output = this.simulateTesting();
        reward = 0.85 + Math.random() * 0.1;
        critique = 'All tests passed, learned from previous failures';
        break;

      case 'reviewer':
        task = 'quality-check';
        input = `Review code quality ${this.iteration}`;
        output = this.simulateReview();
        reward = 0.90 + Math.random() * 0.05;
        critique = 'Code quality improved based on team learnings';
        break;

      default:
        task = 'generic-task';
        input = `Execute task ${this.iteration}`;
        output = 'Task completed';
        reward = 0.7 + Math.random() * 0.25;
        critique = 'Task completed successfully';
    }

    const latencyMs = Date.now() - startTime;

    return {
      sessionId: this.agentId,
      task,
      input,
      output,
      reward,
      critique,
      success: reward >= 0.7,
      tokensUsed: Math.floor(Math.random() * 1000) + 500,
      latencyMs
    };
  }

  private simulateResearch(): string {
    const patterns = [
      'Identified optimal async/await patterns',
      'Found performance bottleneck in data processing',
      'Discovered reusable component structure',
      'Analyzed error handling best practices'
    ];
    return patterns[Math.floor(Math.random() * patterns.length)];
  }

  private simulateCoding(): string {
    const implementations = [
      'Implemented async data fetcher with retry logic',
      'Created reusable validation middleware',
      'Built caching layer for API responses',
      'Refactored error handling with custom exceptions'
    ];
    return implementations[Math.floor(Math.random() * implementations.length)];
  }

  private simulateTesting(): string {
    const tests = [
      'Unit tests: 45 passed, 0 failed',
      'Integration tests: 12 passed, 0 failed',
      'E2E tests: 8 passed, 0 failed',
      'Performance tests: All benchmarks met'
    ];
    return tests[Math.floor(Math.random() * tests.length)];
  }

  private simulateReview(): string {
    const reviews = [
      'Code quality: A+, No critical issues',
      'Security scan: No vulnerabilities found',
      'Performance: Within acceptable limits',
      'Documentation: Complete and accurate'
    ];
    return reviews[Math.floor(Math.random() * reviews.length)];
  }

  private async syncWithHub(): Promise<void> {
    // In a real implementation, this would use AgentDB
    // For now, just sync the connection
    await this.client.sync({} as any);
  }

  private printSummary(): void {
    const avgReward = this.episodes.reduce((sum, e) => sum + e.reward, 0) / this.episodes.length;
    const successRate = this.episodes.filter(e => e.success).length / this.episodes.length;

    console.log(`\n📊 Agent Summary:`);
    console.log(`   Type: ${this.agentType}`);
    console.log(`   ID: ${this.agentId}`);
    console.log(`   Iterations: ${this.iteration}`);
    console.log(`   Average Reward: ${avgReward.toFixed(3)}`);
    console.log(`   Success Rate: ${(successRate * 100).toFixed(1)}%`);
  }

  async cleanup(): Promise<void> {
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
  } catch (error: any) {
    console.error('Agent failed:', error);
    process.exit(1);
  } finally {
    await agent.cleanup();
  }
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
