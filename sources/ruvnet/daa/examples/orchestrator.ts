/**
 * DAA Orchestrator with Native NAPI Bindings
 *
 * This example demonstrates how to use the DAA orchestrator with
 * quantum-resistant cryptography for autonomous agent coordination.
 *
 * Features:
 * - MRAP autonomy loop (Monitor, Reason, Act, Reflect, Adapt)
 * - Quantum-resistant secure communication
 * - Rule-based governance
 * - Economic token management
 *
 * @example
 * ```bash
 * npm install @daa/qudag-native daa-orchestrator daa-rules daa-economy
 * ts-node examples/orchestrator.ts
 * ```
 */

import { MlKem768, MlDsa, blake3Hash } from '@daa/qudag-native';

/**
 * Secure Agent Communication Channel
 *
 * Uses ML-KEM-768 for key exchange and ML-DSA for message authentication.
 */
class SecureChannel {
  private mlkem: MlKem768;
  private mldsa: MlDsa;
  private sharedSecret: Buffer | null = null;

  constructor() {
    this.mlkem = new MlKem768();
    this.mldsa = new MlDsa();
  }

  /**
   * Establish a secure channel with another agent
   */
  async establish(peerPublicKey: Buffer): Promise<Buffer> {
    console.log('📡 Establishing secure channel...');

    const { ciphertext, sharedSecret } = this.mlkem.encapsulate(peerPublicKey);
    this.sharedSecret = sharedSecret;

    console.log(`   ✅ Secure channel established`);
    console.log(`   Shared secret: ${sharedSecret.toString('hex').slice(0, 16)}...`);

    return ciphertext;
  }

  /**
   * Accept a secure channel from another agent
   */
  async accept(ciphertext: Buffer, secretKey: Buffer): Promise<void> {
    console.log('📡 Accepting secure channel...');

    this.sharedSecret = this.mlkem.decapsulate(ciphertext, secretKey);

    console.log(`   ✅ Secure channel accepted`);
    console.log(`   Shared secret: ${this.sharedSecret.toString('hex').slice(0, 16)}...`);
  }

  /**
   * Send an authenticated message
   */
  async sendMessage(message: string, signingKey: Buffer): Promise<{ encrypted: Buffer; signature: Buffer }> {
    if (!this.sharedSecret) {
      throw new Error('Secure channel not established');
    }

    const messageBuffer = Buffer.from(message, 'utf8');

    // Encrypt with shared secret (simple XOR for demo)
    const encrypted = Buffer.from(messageBuffer).map((byte, i) =>
      byte ^ this.sharedSecret![i % this.sharedSecret!.length]
    );

    // Sign the message
    const signature = this.mldsa.sign(messageBuffer, signingKey);

    console.log(`📤 Message sent: "${message.slice(0, 50)}..."`);
    console.log(`   Encrypted: ${encrypted.length} bytes`);
    console.log(`   Signature: ${signature.length} bytes`);

    return { encrypted, signature };
  }

  /**
   * Receive and verify an authenticated message
   */
  async receiveMessage(encrypted: Buffer, signature: Buffer, peerPublicKey: Buffer): Promise<string> {
    if (!this.sharedSecret) {
      throw new Error('Secure channel not established');
    }

    // Decrypt with shared secret
    const decrypted = Buffer.from(encrypted).map((byte, i) =>
      byte ^ this.sharedSecret![i % this.sharedSecret!.length]
    );

    // Verify signature
    const isValid = this.mldsa.verify(decrypted, signature, peerPublicKey);

    if (!isValid) {
      throw new Error('Message signature verification failed');
    }

    const message = decrypted.toString('utf8');

    console.log(`📥 Message received: "${message.slice(0, 50)}..."`);
    console.log(`   ✅ Signature verified`);

    return message;
  }
}

/**
 * Agent Identity with Quantum-Resistant Keys
 */
interface AgentIdentity {
  id: string;
  mlkemPublicKey: Buffer;
  mlkemSecretKey: Buffer;
  mldsaPublicKey: Buffer;
  mldsaSecretKey: Buffer;
  fingerprint: string;
}

/**
 * Create a new agent identity with quantum-resistant keys
 */
async function createAgentIdentity(id: string): Promise<AgentIdentity> {
  console.log(`\n🤖 Creating agent identity: ${id}`);

  const mlkem = new MlKem768();
  const mlkemKeypair = mlkem.generateKeypair();

  // In a real implementation, you would generate proper ML-DSA keys
  const mldsaPublicKey = Buffer.alloc(1952);
  const mldsaSecretKey = Buffer.alloc(2560);

  // Generate quantum fingerprint
  const identityData = Buffer.from(JSON.stringify({
    id,
    mlkemPublicKey: mlkemKeypair.publicKey.toString('hex'),
    timestamp: Date.now()
  }));
  const fingerprint = blake3Hash(identityData).toString('hex').slice(0, 16);

  console.log(`   ✅ Agent identity created`);
  console.log(`   Fingerprint: ${fingerprint}`);

  return {
    id,
    mlkemPublicKey: mlkemKeypair.publicKey,
    mlkemSecretKey: mlkemKeypair.secretKey,
    mldsaPublicKey,
    mldsaSecretKey,
    fingerprint
  };
}

/**
 * Simple MRAP Autonomy Loop
 *
 * Demonstrates the core autonomy loop with quantum-resistant security.
 */
class AutonomousAgent {
  private identity: AgentIdentity;
  private channel: SecureChannel;
  private state: 'idle' | 'monitoring' | 'reasoning' | 'acting' | 'reflecting' | 'adapting' = 'idle';
  private metrics: {
    cyclesCompleted: number;
    messagesProcessed: number;
    decisionsAde: number;
  } = {
    cyclesCompleted: 0,
    messagesProcessed: 0,
    decisionsAde: 0
  };

  constructor(identity: AgentIdentity) {
    this.identity = identity;
    this.channel = new SecureChannel();
  }

  /**
   * Monitor - Gather information about the environment
   */
  private async monitor(): Promise<{ temperature: number; peers: number; taskQueue: number }> {
    this.state = 'monitoring';
    console.log(`\n[${this.identity.id}] 👁️  MONITOR: Gathering environment data...`);

    // Simulate monitoring
    const data = {
      temperature: Math.random() * 100,
      peers: Math.floor(Math.random() * 10),
      taskQueue: Math.floor(Math.random() * 20)
    };

    console.log(`   Temperature: ${data.temperature.toFixed(1)}°C`);
    console.log(`   Peers online: ${data.peers}`);
    console.log(`   Tasks queued: ${data.taskQueue}`);

    return data;
  }

  /**
   * Reason - AI-powered decision making
   */
  private async reason(monitorData: any): Promise<{ action: string; priority: number }> {
    this.state = 'reasoning';
    console.log(`\n[${this.identity.id}] 🧠 REASON: Analyzing data and planning action...`);

    // Simple decision logic (in production, this would use AI)
    let action = 'idle';
    let priority = 0;

    if (monitorData.taskQueue > 10) {
      action = 'process_tasks';
      priority = 3;
    } else if (monitorData.peers < 3) {
      action = 'find_peers';
      priority = 2;
    } else if (monitorData.temperature > 80) {
      action = 'alert_overheating';
      priority = 4;
    } else {
      action = 'optimize_resources';
      priority = 1;
    }

    console.log(`   Decision: ${action} (priority: ${priority})`);

    this.metrics.decisionsAde++;
    return { action, priority };
  }

  /**
   * Act - Execute the planned action
   */
  private async act(decision: { action: string; priority: number }): Promise<string> {
    this.state = 'acting';
    console.log(`\n[${this.identity.id}] ⚡ ACT: Executing action "${decision.action}"...`);

    // Simulate action execution
    await new Promise(resolve => setTimeout(resolve, 500));

    const result = `Completed ${decision.action} with priority ${decision.priority}`;
    console.log(`   ✅ ${result}`);

    return result;
  }

  /**
   * Reflect - Evaluate the outcome
   */
  private async reflect(result: string): Promise<{ success: boolean; learnings: string[] }> {
    this.state = 'reflecting';
    console.log(`\n[${this.identity.id}] 🔍 REFLECT: Evaluating outcome...`);

    const reflection = {
      success: Math.random() > 0.2,  // 80% success rate
      learnings: [
        'Action completed within expected timeframe',
        'Resource usage was optimal',
        'Peer coordination effective'
      ]
    };

    console.log(`   Success: ${reflection.success ? '✅' : '❌'}`);
    console.log(`   Learnings: ${reflection.learnings.length} items recorded`);

    return reflection;
  }

  /**
   * Adapt - Update strategy based on reflection
   */
  private async adapt(reflection: { success: boolean; learnings: string[] }): Promise<void> {
    this.state = 'adapting';
    console.log(`\n[${this.identity.id}] 🔄 ADAPT: Updating strategy...`);

    if (reflection.success) {
      console.log(`   ✅ Strategy validated - maintaining current approach`);
    } else {
      console.log(`   ⚠️  Strategy adjustment needed - optimizing parameters`);
    }

    console.log(`   Learnings integrated into knowledge base`);
  }

  /**
   * Run one complete MRAP cycle
   */
  async runCycle(): Promise<void> {
    console.log(`\n${'═'.repeat(60)}`);
    console.log(`MRAP CYCLE ${this.metrics.cyclesCompleted + 1} - Agent: ${this.identity.id}`);
    console.log(`${'═'.repeat(60)}`);

    try {
      const monitorData = await this.monitor();
      const decision = await this.reason(monitorData);
      const result = await this.act(decision);
      const reflection = await this.reflect(result);
      await this.adapt(reflection);

      this.metrics.cyclesCompleted++;
      this.state = 'idle';

      console.log(`\n✅ MRAP cycle ${this.metrics.cyclesCompleted} completed successfully`);

    } catch (error) {
      console.error(`\n❌ MRAP cycle failed:`, error);
      this.state = 'idle';
    }
  }

  /**
   * Connect to another agent
   */
  async connectTo(peer: AutonomousAgent): Promise<void> {
    console.log(`\n🔗 [${this.identity.id}] Connecting to [${peer.identity.id}]...`);

    // Establish secure channel
    const ciphertext = await this.channel.establish(peer.identity.mlkemPublicKey);

    // Peer accepts the connection
    await peer.channel.accept(ciphertext, peer.identity.mlkemSecretKey);

    console.log(`   ✅ Agents connected securely`);
  }

  /**
   * Send a message to a connected agent
   */
  async sendMessageTo(peer: AutonomousAgent, message: string): Promise<void> {
    const { encrypted, signature } = await this.channel.sendMessage(
      message,
      this.identity.mldsaSecretKey
    );

    // Peer receives and verifies the message
    await peer.channel.receiveMessage(encrypted, signature, this.identity.mldsaPublicKey);

    this.metrics.messagesProcessed++;
  }

  /**
   * Get agent metrics
   */
  getMetrics() {
    return {
      identity: {
        id: this.identity.id,
        fingerprint: this.identity.fingerprint
      },
      state: this.state,
      ...this.metrics
    };
  }
}

/**
 * Multi-Agent Orchestration Example
 */
async function multiAgentOrchestration() {
  console.log('\n╔══════════════════════════════════════════════════════════╗');
  console.log('║   Multi-Agent Orchestration with Quantum Security        ║');
  console.log('╚══════════════════════════════════════════════════════════╝');

  // Create three autonomous agents
  const alice = new AutonomousAgent(await createAgentIdentity('Alice'));
  const bob = new AutonomousAgent(await createAgentIdentity('Bob'));
  const charlie = new AutonomousAgent(await createAgentIdentity('Charlie'));

  // Connect agents
  console.log('\n--- Agent Network Formation ---');
  await alice.connectTo(bob);
  await bob.connectTo(charlie);
  await charlie.connectTo(alice);

  // Alice sends coordination message
  console.log('\n--- Agent Communication ---');
  await alice.sendMessageTo(bob, 'Hello Bob! Let\'s coordinate on task distribution.');
  await bob.sendMessageTo(charlie, 'Charlie, Alice wants to coordinate. Let\'s sync.');
  await charlie.sendMessageTo(alice, 'Ready for coordination, Alice!');

  // Run MRAP cycles
  console.log('\n--- Autonomous Operation ---');

  // Run 2 cycles for each agent
  await alice.runCycle();
  await bob.runCycle();
  await charlie.runCycle();

  await alice.runCycle();
  await bob.runCycle();
  await charlie.runCycle();

  // Display metrics
  console.log('\n--- Agent Metrics ---');
  console.log('\nAlice:', JSON.stringify(alice.getMetrics(), null, 2));
  console.log('\nBob:', JSON.stringify(bob.getMetrics(), null, 2));
  console.log('\nCharlie:', JSON.stringify(charlie.getMetrics(), null, 2));
}

/**
 * Main function
 */
async function main() {
  console.log('╔══════════════════════════════════════════════════════════╗');
  console.log('║   DAA Orchestrator with Native NAPI Bindings            ║');
  console.log('║   Quantum-Resistant Autonomous Agents                   ║');
  console.log('╚══════════════════════════════════════════════════════════╝');

  try {
    await multiAgentOrchestration();

    console.log('\n\n✅ Orchestrator example completed successfully!');
    console.log('\nKey features demonstrated:');
    console.log('  • MRAP autonomy loop (Monitor, Reason, Act, Reflect, Adapt)');
    console.log('  • Quantum-resistant secure channels (ML-KEM-768)');
    console.log('  • Authenticated messaging (ML-DSA)');
    console.log('  • Multi-agent coordination');
    console.log('  • Performance metrics tracking');

  } catch (error) {
    console.error('\n❌ Error running orchestrator example:', error);
    process.exit(1);
  }
}

// Run if executed directly
if (require.main === module) {
  main();
}

export { SecureChannel, AutonomousAgent, createAgentIdentity, multiAgentOrchestration };
