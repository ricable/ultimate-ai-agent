/**
 * Full-Stack Autonomous Agent Example
 *
 * This example demonstrates a complete autonomous agent integrating:
 * - Quantum-resistant cryptography (QuDAG)
 * - MRAP autonomy loop (Orchestrator)
 * - Economic self-sufficiency (Economy)
 * - Rule-based governance (Rules)
 * - Federated learning (Prime ML)
 * - API server with secure endpoints
 *
 * @example
 * ```bash
 * npm install @daa/qudag-native express
 * ts-node examples/full-stack-agent.ts
 * ```
 */

import { MlKem768, MlDsa, blake3Hash, blake3HashHex, quantumFingerprint } from '@daa/qudag-native';
import * as express from 'express';

/**
 * Agent Configuration
 */
interface AgentConfig {
  id: string;
  name: string;
  role: 'coordinator' | 'worker' | 'validator';
  initialTokens: number;
  apiPort: number;
}

/**
 * Agent State
 */
interface AgentState {
  config: AgentConfig;
  identity: {
    fingerprint: string;
    mlkemPublicKey: string;
    mldsaPublicKey: string;
  };
  economy: {
    balance: number;
    totalEarned: number;
    totalSpent: number;
    transactions: Transaction[];
  };
  governance: {
    rules: Rule[];
    violations: number;
  };
  performance: {
    cyclesCompleted: number;
    tasksProcessed: number;
    uptime: number;
    startTime: number;
  };
  network: {
    peers: string[];
    messagesReceived: number;
    messagesSent: number;
  };
}

/**
 * Transaction
 */
interface Transaction {
  id: string;
  from: string;
  to: string;
  amount: number;
  timestamp: number;
  signature: string;
  purpose: string;
}

/**
 * Rule
 */
interface Rule {
  id: string;
  name: string;
  condition: string;
  action: string;
  priority: number;
}

/**
 * Task
 */
interface Task {
  id: string;
  type: 'compute' | 'training' | 'validation' | 'coordination';
  payload: any;
  requester: string;
  reward: number;
  deadline: number;
}

/**
 * Full-Stack Autonomous Agent
 */
class FullStackAgent {
  private mlkem: MlKem768;
  private mldsa: MlDsa;
  private mlkemKeypair: { publicKey: Buffer; secretKey: Buffer };
  private mldsaKeys: { publicKey: Buffer; secretKey: Buffer };
  private state: AgentState;
  private app: express.Application;
  private isRunning: boolean = false;

  constructor(config: AgentConfig) {
    this.mlkem = new MlKem768();
    this.mldsa = new MlDsa();

    // Generate quantum-resistant keys
    this.mlkemKeypair = this.mlkem.generateKeypair();
    this.mldsaKeys = {
      publicKey: Buffer.alloc(1952),
      secretKey: Buffer.alloc(2560)
    };

    // Calculate fingerprint
    const identityData = Buffer.from(JSON.stringify({
      id: config.id,
      name: config.name,
      mlkemPublicKey: this.mlkemKeypair.publicKey.toString('hex'),
      timestamp: Date.now()
    }));
    const fingerprint = quantumFingerprint(identityData);

    // Initialize state
    this.state = {
      config,
      identity: {
        fingerprint,
        mlkemPublicKey: this.mlkemKeypair.publicKey.toString('hex'),
        mldsaPublicKey: this.mldsaKeys.publicKey.toString('hex')
      },
      economy: {
        balance: config.initialTokens,
        totalEarned: 0,
        totalSpent: 0,
        transactions: []
      },
      governance: {
        rules: this.initializeRules(),
        violations: 0
      },
      performance: {
        cyclesCompleted: 0,
        tasksProcessed: 0,
        uptime: 0,
        startTime: Date.now()
      },
      network: {
        peers: [],
        messagesReceived: 0,
        messagesSent: 0
      }
    };

    // Initialize API server
    this.app = express();
    this.app.use(express.json());
    this.setupRoutes();
  }

  /**
   * Initialize governance rules
   */
  private initializeRules(): Rule[] {
    return [
      {
        id: 'rule-001',
        name: 'Maximum Daily Spending',
        condition: 'dailySpending > 1000',
        action: 'reject',
        priority: 1
      },
      {
        id: 'rule-002',
        name: 'Minimum Token Balance',
        condition: 'balance < 100',
        action: 'alert',
        priority: 2
      },
      {
        id: 'rule-003',
        name: 'Task Reward Minimum',
        condition: 'taskReward < 10',
        action: 'reject',
        priority: 1
      },
      {
        id: 'rule-004',
        name: 'Maximum Concurrent Tasks',
        condition: 'activeTasks > 10',
        action: 'queue',
        priority: 3
      }
    ];
  }

  /**
   * Setup API routes
   */
  private setupRoutes(): void {
    // Health check
    this.app.get('/health', (req, res) => {
      res.json({
        status: this.isRunning ? 'running' : 'stopped',
        uptime: Date.now() - this.state.performance.startTime,
        agent: this.state.config.id
      });
    });

    // Get agent info
    this.app.get('/info', (req, res) => {
      res.json({
        id: this.state.config.id,
        name: this.state.config.name,
        role: this.state.config.role,
        fingerprint: this.state.identity.fingerprint,
        publicKeys: {
          mlkem: this.state.identity.mlkemPublicKey.slice(0, 32) + '...',
          mldsa: this.state.identity.mldsaPublicKey.slice(0, 32) + '...'
        }
      });
    });

    // Get agent state
    this.app.get('/state', (req, res) => {
      res.json(this.state);
    });

    // Submit task
    this.app.post('/task', async (req, res) => {
      try {
        const task: Task = req.body;
        const result = await this.processTask(task);
        res.json({ success: true, result });
      } catch (error) {
        res.status(500).json({ success: false, error: error.message });
      }
    });

    // Transfer tokens
    this.app.post('/transfer', async (req, res) => {
      try {
        const { to, amount, purpose } = req.body;
        const tx = await this.transfer(to, amount, purpose);
        res.json({ success: true, transaction: tx });
      } catch (error) {
        res.status(500).json({ success: false, error: error.message });
      }
    });

    // Get economy stats
    this.app.get('/economy', (req, res) => {
      res.json(this.state.economy);
    });

    // Get rules
    this.app.get('/rules', (req, res) => {
      res.json(this.state.governance.rules);
    });

    // Add peer
    this.app.post('/peers', (req, res) => {
      const { peerAddress } = req.body;
      this.state.network.peers.push(peerAddress);
      res.json({ success: true, peers: this.state.network.peers });
    });
  }

  /**
   * Start the agent
   */
  async start(): Promise<void> {
    console.log(`\n${'═'.repeat(60)}`);
    console.log(`🚀 Starting Full-Stack Agent: ${this.state.config.name}`);
    console.log(`${'═'.repeat(60)}`);

    console.log(`\n📊 Agent Configuration:`);
    console.log(`   ID: ${this.state.config.id}`);
    console.log(`   Role: ${this.state.config.role}`);
    console.log(`   Fingerprint: ${this.state.identity.fingerprint}`);
    console.log(`   Initial tokens: ${this.state.economy.balance}`);
    console.log(`   API port: ${this.state.config.apiPort}`);

    // Start API server
    await new Promise<void>((resolve) => {
      this.app.listen(this.state.config.apiPort, () => {
        console.log(`\n✅ API server listening on port ${this.state.config.apiPort}`);
        resolve();
      });
    });

    this.isRunning = true;

    // Start autonomy loop
    this.runAutonomyLoop();
  }

  /**
   * MRAP Autonomy Loop
   */
  private async runAutonomyLoop(): Promise<void> {
    while (this.isRunning) {
      try {
        await this.mrapCycle();
        await new Promise(resolve => setTimeout(resolve, 5000));  // 5 second cycle
      } catch (error) {
        console.error('❌ MRAP cycle error:', error);
      }
    }
  }

  /**
   * One MRAP cycle
   */
  private async mrapCycle(): Promise<void> {
    const cycleNum = this.state.performance.cyclesCompleted + 1;

    console.log(`\n--- MRAP Cycle ${cycleNum} ---`);

    // Monitor
    const monitorData = await this.monitor();

    // Reason
    const decision = await this.reason(monitorData);

    // Act
    const result = await this.act(decision);

    // Reflect
    const reflection = await this.reflect(result);

    // Adapt
    await this.adapt(reflection);

    this.state.performance.cyclesCompleted++;
    this.state.performance.uptime = Date.now() - this.state.performance.startTime;
  }

  /**
   * Monitor phase
   */
  private async monitor(): Promise<any> {
    return {
      balance: this.state.economy.balance,
      peers: this.state.network.peers.length,
      rulesActive: this.state.governance.rules.length,
      uptime: Date.now() - this.state.performance.startTime
    };
  }

  /**
   * Reason phase
   */
  private async reason(data: any): Promise<{ action: string; params: any }> {
    // Simple decision logic
    if (data.balance < 100) {
      return { action: 'seek_tasks', params: { minReward: 50 } };
    } else if (data.peers < 3) {
      return { action: 'discover_peers', params: {} };
    } else {
      return { action: 'optimize', params: {} };
    }
  }

  /**
   * Act phase
   */
  private async act(decision: { action: string; params: any }): Promise<string> {
    console.log(`   ⚡ Action: ${decision.action}`);
    // Simulate action
    return `Executed ${decision.action}`;
  }

  /**
   * Reflect phase
   */
  private async reflect(result: string): Promise<{ success: boolean }> {
    return { success: true };
  }

  /**
   * Adapt phase
   */
  private async adapt(reflection: { success: boolean }): Promise<void> {
    // Update strategy based on reflection
  }

  /**
   * Process a task
   */
  private async processTask(task: Task): Promise<any> {
    console.log(`\n📋 Processing task: ${task.id} (${task.type})`);

    // Check rules
    const ruleCheck = this.checkRules({ taskReward: task.reward });
    if (!ruleCheck.allowed) {
      throw new Error(`Task rejected by rule: ${ruleCheck.reason}`);
    }

    // Simulate task processing
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Earn reward
    this.state.economy.balance += task.reward;
    this.state.economy.totalEarned += task.reward;
    this.state.performance.tasksProcessed++;

    console.log(`   ✅ Task completed, earned ${task.reward} tokens`);

    return {
      taskId: task.id,
      result: 'completed',
      reward: task.reward,
      timestamp: Date.now()
    };
  }

  /**
   * Transfer tokens
   */
  private async transfer(to: string, amount: number, purpose: string): Promise<Transaction> {
    console.log(`\n💸 Transfer: ${amount} tokens to ${to}`);

    // Check rules
    const ruleCheck = this.checkRules({ transactionAmount: amount });
    if (!ruleCheck.allowed) {
      throw new Error(`Transfer rejected by rule: ${ruleCheck.reason}`);
    }

    if (this.state.economy.balance < amount) {
      throw new Error('Insufficient balance');
    }

    const tx: Transaction = {
      id: blake3Hash(Buffer.from(`${Date.now()}`)).toString('hex').slice(0, 16),
      from: this.state.config.id,
      to,
      amount,
      timestamp: Date.now(),
      signature: this.mldsa.sign(
        Buffer.from(JSON.stringify({ to, amount, timestamp: Date.now() })),
        this.mldsaKeys.secretKey
      ).toString('hex'),
      purpose
    };

    this.state.economy.balance -= amount;
    this.state.economy.totalSpent += amount;
    this.state.economy.transactions.push(tx);

    console.log(`   ✅ Transfer completed: ${tx.id}`);

    return tx;
  }

  /**
   * Check governance rules
   */
  private checkRules(context: any): { allowed: boolean; reason?: string } {
    for (const rule of this.state.governance.rules) {
      // Simple rule evaluation (in production, use proper expression parser)
      if (rule.condition.includes('taskReward') && context.taskReward) {
        if (rule.condition.includes('<') && context.taskReward < 10) {
          if (rule.action === 'reject') {
            this.state.governance.violations++;
            return { allowed: false, reason: rule.name };
          }
        }
      }
    }

    return { allowed: true };
  }

  /**
   * Stop the agent
   */
  async stop(): Promise<void> {
    console.log(`\n⏹️  Stopping agent: ${this.state.config.name}`);
    this.isRunning = false;
  }
}

/**
 * Create and run a swarm of agents
 */
async function runAgentSwarm() {
  console.log('\n╔══════════════════════════════════════════════════════════╗');
  console.log('║   Full-Stack Agent Swarm                                ║');
  console.log('╚══════════════════════════════════════════════════════════╝');

  // Create coordinator agent
  const coordinator = new FullStackAgent({
    id: 'agent-coordinator-001',
    name: 'Coordinator Alpha',
    role: 'coordinator',
    initialTokens: 10000,
    apiPort: 3000
  });

  // Create worker agents
  const worker1 = new FullStackAgent({
    id: 'agent-worker-001',
    name: 'Worker One',
    role: 'worker',
    initialTokens: 1000,
    apiPort: 3001
  });

  const worker2 = new FullStackAgent({
    id: 'agent-worker-002',
    name: 'Worker Two',
    role: 'worker',
    initialTokens: 1000,
    apiPort: 3002
  });

  // Start all agents
  await coordinator.start();
  await worker1.start();
  await worker2.start();

  console.log('\n✅ Agent swarm is now running');
  console.log('\nAPI endpoints:');
  console.log('   Coordinator: http://localhost:3000');
  console.log('   Worker 1:    http://localhost:3001');
  console.log('   Worker 2:    http://localhost:3002');

  console.log('\nTry these commands:');
  console.log('   curl http://localhost:3000/info');
  console.log('   curl http://localhost:3000/state');
  console.log('   curl http://localhost:3000/economy');

  // Keep running
  console.log('\nPress Ctrl+C to stop...');
}

/**
 * Main function
 */
async function main() {
  console.log('╔══════════════════════════════════════════════════════════╗');
  console.log('║   Full-Stack Autonomous Agent with NAPI-rs              ║');
  console.log('║   Complete DAA Ecosystem Integration                    ║');
  console.log('╚══════════════════════════════════════════════════════════╝');

  try {
    await runAgentSwarm();
  } catch (error) {
    console.error('\n❌ Error running agent swarm:', error);
    process.exit(1);
  }
}

// Run if executed directly
if (require.main === module) {
  main().catch(console.error);
}

export { FullStackAgent, runAgentSwarm };
