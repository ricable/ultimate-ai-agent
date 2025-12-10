/**
 * DAA SDK - Unified TypeScript SDK for Distributed Agentic Architecture
 *
 * Provides automatic platform detection and seamless integration with:
 * - Native NAPI-rs bindings (Node.js) for maximum performance
 * - WebAssembly bindings (browsers) for cross-platform compatibility
 *
 * @module daa-sdk
 * @version 0.1.0
 */

export * from './platform';
export * from './qudag';
// export * from './orchestrator'; // TODO: Implement orchestrator bindings
// export * from './prime'; // TODO: Implement prime bindings

import { detectPlatform, loadQuDAG, loadOrchestrator, loadPrime } from './platform';

/**
 * Configuration options for DAA SDK initialization
 */
export interface DAAConfig {
  /** Orchestrator configuration */
  orchestrator?: {
    enableMRAP?: boolean;
    workflowEngine?: boolean;
    eventBusSize?: number;
  };

  /** QuDAG network configuration */
  qudag?: {
    enableCrypto?: boolean;
    enableVault?: boolean;
    networkMode?: 'p2p' | 'client' | 'server';
  };

  /** Prime ML configuration */
  prime?: {
    enableTraining?: boolean;
    enableCoordination?: boolean;
    gpuAcceleration?: boolean;
  };

  /** Force platform selection */
  forcePlatform?: 'native' | 'wasm';
}

/**
 * Main DAA SDK class providing unified access to all DAA components
 *
 * @example
 * ```typescript
 * import { DAA } from 'daa-sdk';
 *
 * const daa = new DAA();
 * await daa.init();
 *
 * // Use quantum-resistant crypto
 * const mlkem = daa.crypto.mlkem();
 * const keypair = mlkem.generateKeypair();
 *
 * // Start orchestrator
 * await daa.orchestrator.start();
 * ```
 */
export class DAA {
  private platform: 'native' | 'wasm';
  private qudag: any;
  private orchestratorLib: any;
  private primeLib: any;
  private config: DAAConfig;
  private initialized: boolean = false;

  constructor(config: DAAConfig = {}) {
    this.config = config;
    this.platform = config.forcePlatform || detectPlatform();
    console.log(`🚀 DAA SDK initialized with ${this.platform} runtime`);
  }

  /**
   * Initialize the SDK and load all required bindings
   */
  async init(): Promise<void> {
    if (this.initialized) {
      console.warn('DAA SDK already initialized');
      return;
    }

    console.log('📦 Loading DAA components...');

    try {
      // Load QuDAG (crypto & networking)
      if (this.config.qudag !== undefined && typeof this.config.qudag === 'object' && this.config.qudag !== null) {
        this.qudag = await loadQuDAG(this.platform);
        console.log('✅ QuDAG loaded');
      } else if (this.config.qudag === undefined) {
        // Load by default if not specified
        this.qudag = await loadQuDAG(this.platform);
        console.log('✅ QuDAG loaded');
      }

      // Load Orchestrator (MRAP & workflows) - TODO: Implement
      if (this.config.orchestrator !== undefined && typeof this.config.orchestrator === 'object' && this.config.orchestrator !== null) {
        // this.orchestratorLib = await loadOrchestrator(this.platform);
        console.log('⚠️  Orchestrator not yet implemented');
      }

      // Load Prime (ML training) - TODO: Implement
      if (this.config.prime !== undefined && typeof this.config.prime === 'object' && this.config.prime !== null) {
        // this.primeLib = await loadPrime(this.platform);
        console.log('⚠️  Prime ML not yet implemented');
      }

      this.initialized = true;
      console.log('🎉 DAA SDK ready!');
    } catch (error) {
      console.error('❌ Failed to initialize DAA SDK:', error);
      throw error;
    }
  }

  /**
   * Get current platform runtime
   */
  getPlatform(): 'native' | 'wasm' {
    return this.platform;
  }

  /**
   * Check if SDK is initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Quantum-resistant cryptography operations
   */
  crypto = {
    /**
     * ML-KEM-768 key encapsulation mechanism
     */
    mlkem: () => {
      this.ensureInitialized();
      return new this.qudag.MlKem768();
    },

    /**
     * ML-DSA digital signature algorithm
     */
    mldsa: () => {
      this.ensureInitialized();
      return new this.qudag.MlDsa();
    },

    /**
     * BLAKE3 cryptographic hash function
     */
    blake3: (data: Buffer): Buffer => {
      this.ensureInitialized();
      return this.qudag.Blake3.hash(data);
    },

    /**
     * BLAKE3 hash as hex string
     */
    blake3Hex: (data: Buffer): string => {
      this.ensureInitialized();
      return this.qudag.Blake3.hashHex(data);
    },

    /**
     * Quantum fingerprinting for data integrity
     */
    quantumFingerprint: (data: Buffer): string => {
      this.ensureInitialized();
      return this.qudag.Blake3.quantumFingerprint(data);
    },
  };

  /**
   * Password vault operations
   */
  vault = {
    /**
     * Create a new password vault
     */
    create: (masterPassword: string) => {
      this.ensureInitialized();
      return new this.qudag.Vault.PasswordVault(masterPassword);
    },
  };

  /**
   * DAA orchestrator operations (MRAP loop)
   */
  orchestrator = {
    /**
     * Start the MRAP autonomy loop
     */
    start: async () => {
      this.ensureInitialized();
      const orchestrator = new this.orchestratorLib.Orchestrator(
        this.config.orchestrator || {}
      );
      return orchestrator.start();
    },

    /**
     * Monitor system state
     */
    monitor: async () => {
      this.ensureInitialized();
      const orchestrator = new this.orchestratorLib.Orchestrator();
      return orchestrator.monitor();
    },

    /**
     * Create a new workflow
     */
    createWorkflow: async (definition: any) => {
      this.ensureInitialized();
      const engine = new this.orchestratorLib.WorkflowEngine();
      return engine.createWorkflow(definition);
    },

    /**
     * Execute a workflow
     */
    executeWorkflow: async (workflowId: string, input: any) => {
      this.ensureInitialized();
      const engine = new this.orchestratorLib.WorkflowEngine();
      return engine.executeWorkflow(workflowId, input);
    },
  };

  /**
   * Rules engine operations
   */
  rules = {
    /**
     * Evaluate rules against context
     */
    evaluate: async (context: any) => {
      this.ensureInitialized();
      const engine = new this.orchestratorLib.RulesEngine();
      return engine.evaluate(context);
    },

    /**
     * Add a new rule
     */
    addRule: async (rule: any) => {
      this.ensureInitialized();
      const engine = new this.orchestratorLib.RulesEngine();
      return engine.addRule(rule);
    },
  };

  /**
   * Economy management operations (rUv tokens)
   */
  economy = {
    /**
     * Get token balance for an agent
     */
    getBalance: async (agentId: string) => {
      this.ensureInitialized();
      const manager = new this.orchestratorLib.EconomyManager();
      return manager.getBalance(agentId);
    },

    /**
     * Transfer tokens between agents
     */
    transfer: async (from: string, to: string, amount: number) => {
      this.ensureInitialized();
      const manager = new this.orchestratorLib.EconomyManager();
      return manager.transfer(from, to, amount);
    },

    /**
     * Calculate dynamic fee for an operation
     */
    calculateFee: async (operation: any) => {
      this.ensureInitialized();
      const manager = new this.orchestratorLib.EconomyManager();
      return manager.calculateFee(operation);
    },
  };

  /**
   * Prime ML training operations
   */
  prime = {
    /**
     * Create a new training node
     */
    createNode: (config: any) => {
      this.ensureInitialized();
      return new this.primeLib.TrainingNode(config);
    },

    /**
     * Start federated training
     */
    startTraining: async (config: any) => {
      this.ensureInitialized();
      const coordinator = new this.primeLib.Coordinator();
      return coordinator.startTraining(config);
    },

    /**
     * Get training progress
     */
    getProgress: async (sessionId: string) => {
      this.ensureInitialized();
      const coordinator = new this.primeLib.Coordinator();
      return coordinator.getProgress(sessionId);
    },
  };

  /**
   * QuDAG exchange operations (rUv token trading)
   */
  exchange = {
    /**
     * Create a new token transaction
     */
    createTransaction: async (from: string, to: string, amount: number) => {
      this.ensureInitialized();
      const ruvToken = new this.qudag.Exchange.RuvToken();
      return ruvToken.createTransaction(from, to, amount);
    },

    /**
     * Sign a transaction with quantum-resistant signature
     */
    signTransaction: async (tx: any, privateKey: Uint8Array) => {
      this.ensureInitialized();
      const ruvToken = new this.qudag.Exchange.RuvToken();
      return ruvToken.signTransaction(tx, privateKey);
    },

    /**
     * Verify a signed transaction
     */
    verifyTransaction: (signedTx: any): boolean => {
      this.ensureInitialized();
      const ruvToken = new this.qudag.Exchange.RuvToken();
      return ruvToken.verifyTransaction(signedTx);
    },

    /**
     * Submit transaction to the network
     */
    submitTransaction: async (signedTx: any) => {
      this.ensureInitialized();
      const ruvToken = new this.qudag.Exchange.RuvToken();
      return ruvToken.submitTransaction(signedTx);
    },
  };

  /**
   * Ensure SDK is initialized before operations
   */
  private ensureInitialized(): void {
    if (!this.initialized) {
      throw new Error('DAA SDK not initialized. Call await daa.init() first.');
    }
  }
}

/**
 * Default export for convenience
 */
export default DAA;
