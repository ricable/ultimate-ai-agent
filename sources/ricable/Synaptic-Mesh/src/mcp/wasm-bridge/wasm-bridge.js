/**
 * WASM Bridge for Synaptic Neural Mesh
 * Connects MCP tools to Rust WASM modules and native implementations
 */

import { EventEmitter } from 'events';
import { promises as fs } from 'fs';
import path from 'path';

export class WasmBridge extends EventEmitter {
  constructor(config) {
    super();
    this.config = config;
    this.wasmModules = new Map();
    this.loadedModules = new Map();
    this.performanceMetrics = new Map();
    
    this.isInitialized = false;
    this.supportsSIMD = false;
    this.supportsThreads = false;
    
    this.initializeBridge();
  }

  async initializeBridge() {
    try {
      // Detect WASM capabilities
      await this.detectCapabilities();
      
      // Load core WASM modules
      await this.loadCoreModules();
      
      this.isInitialized = true;
      console.log('🦀 WASM Bridge initialized successfully');
      
    } catch (error) {
      console.error('❌ Failed to initialize WASM Bridge:', error);
      throw error;
    }
  }

  async detectCapabilities() {
    try {
      // Check for SIMD support
      this.supportsSIMD = await this.checkSIMDSupport();
      
      // Check for SharedArrayBuffer (threads) support
      this.supportsThreads = typeof SharedArrayBuffer !== 'undefined';
      
      console.log(`🔍 WASM Capabilities detected:
        SIMD: ${this.supportsSIMD ? '✅' : '❌'}
        Threads: ${this.supportsThreads ? '✅' : '❌'}`);
        
    } catch (error) {
      console.warn('⚠️ Could not detect all WASM capabilities:', error.message);
    }
  }

  async checkSIMDSupport() {
    try {
      // Try to compile a simple SIMD instruction
      const simdTestBytes = new Uint8Array([
        0x00, 0x61, 0x73, 0x6d, // WASM magic
        0x01, 0x00, 0x00, 0x00, // version
        0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b, // type section
        0x03, 0x02, 0x01, 0x00, // function section
        0x0a, 0x09, 0x01, 0x07, 0x00, 0xfd, 0x0c, 0x00, 0x0b // code section with v128.const
      ]);
      
      const module = await WebAssembly.compile(simdTestBytes);
      return true;
    } catch {
      return false;
    }
  }

  async loadCoreModules() {
    const moduleConfigs = [
      {
        name: 'qudag',
        path: '../rs/QuDAG/QuDAG-main/qudag-wasm/pkg',
        required: true
      },
      {
        name: 'ruv_swarm',
        path: '../rs/ruv-FANN/ruv-swarm/wasm-unified',
        required: true
      },
      {
        name: 'daa',
        path: '../rs/daa/daa-main/wasm',
        required: false
      },
      {
        name: 'cuda_wasm',
        path: '../rs/cuda-wasm/pkg',
        required: false
      }
    ];

    for (const config of moduleConfigs) {
      try {
        await this.loadWasmModule(config.name, config.path);
        console.log(`📦 Loaded WASM module: ${config.name}`);
      } catch (error) {
        if (config.required) {
          throw new Error(`Required WASM module '${config.name}' failed to load: ${error.message}`);
        } else {
          console.warn(`⚠️ Optional WASM module '${config.name}' not available:`, error.message);
        }
      }
    }
  }

  async loadWasmModule(name, modulePath) {
    try {
      // Try different possible locations and file extensions
      const possiblePaths = [
        path.join(modulePath, `${name}.js`),
        path.join(modulePath, `${name}_bg.wasm`),
        path.join(modulePath, 'index.js'),
        path.join(modulePath, `${name}.wasm`)
      ];

      let moduleExports = null;
      let wasmPath = null;

      // Try to find and load the module
      for (const filePath of possiblePaths) {
        try {
          const fullPath = path.resolve(filePath);
          await fs.access(fullPath);
          
          if (filePath.endsWith('.js')) {
            // Load JS wrapper
            const module = await import(fullPath);
            moduleExports = module;
            break;
          } else if (filePath.endsWith('.wasm')) {
            // Load raw WASM
            wasmPath = fullPath;
          }
        } catch {
          continue;
        }
      }

      if (!moduleExports && wasmPath) {
        // Load raw WASM file
        const wasmBytes = await fs.readFile(wasmPath);
        const wasmModule = await WebAssembly.compile(wasmBytes);
        const wasmInstance = await WebAssembly.instantiate(wasmModule);
        moduleExports = wasmInstance.exports;
      }

      if (!moduleExports) {
        throw new Error(`Could not find module at any of: ${possiblePaths.join(', ')}`);
      }

      // Store module with metadata
      this.loadedModules.set(name, {
        exports: moduleExports,
        path: modulePath,
        loadedAt: Date.now(),
        callCount: 0,
        totalExecutionTime: 0
      });

      // Initialize performance tracking
      this.performanceMetrics.set(name, {
        callCount: 0,
        totalTime: 0,
        averageTime: 0,
        errors: 0
      });

      return moduleExports;
      
    } catch (error) {
      throw new Error(`Failed to load WASM module '${name}': ${error.message}`);
    }
  }

  async initializeMesh(meshConfig) {
    if (!this.isInitialized) {
      await this.initializeBridge();
    }

    try {
      const results = {};

      // Initialize QuDAG for consensus
      if (this.loadedModules.has('qudag')) {
        results.qudag = await this.callWasmFunction('qudag', 'init_dag', {
          topology: meshConfig.topology,
          max_nodes: meshConfig.maxAgents,
          crypto_level: meshConfig.cryptoLevel
        });
      }

      // Initialize ruv-swarm for agent coordination
      if (this.loadedModules.has('ruv_swarm')) {
        results.ruv_swarm = await this.callWasmFunction('ruv_swarm', 'init_swarm', {
          strategy: meshConfig.strategy,
          max_agents: meshConfig.maxAgents,
          simd_enabled: this.supportsSIMD
        });
      }

      // Initialize DAA for distributed algorithms
      if (this.loadedModules.has('daa')) {
        results.daa = await this.callWasmFunction('daa', 'init_daa', {
          consensus_type: 'raft',
          node_count: meshConfig.maxAgents
        });
      }

      this.emit('meshInitialized', { meshConfig, results });
      return results;

    } catch (error) {
      this.emit('error', { operation: 'initializeMesh', error });
      throw error;
    }
  }

  async initializeAgent(agent) {
    try {
      const results = {};

      // Initialize neural agent in ruv-swarm
      if (this.loadedModules.has('ruv_swarm')) {
        results.neural_agent = await this.callWasmFunction('ruv_swarm', 'create_agent', {
          agent_id: agent.id,
          agent_type: agent.type,
          capabilities: agent.capabilities,
          neural_model: agent.neuralModel || 'default'
        });
      }

      // Set up consensus participation in QuDAG
      if (this.loadedModules.has('qudag')) {
        results.consensus_node = await this.callWasmFunction('qudag', 'add_node', {
          node_id: agent.id,
          node_type: 'agent',
          voting_weight: 1.0
        });
      }

      this.emit('agentInitialized', { agent, results });
      return results;

    } catch (error) {
      this.emit('error', { operation: 'initializeAgent', error });
      throw error;
    }
  }

  async callWasmFunction(moduleName, functionName, args = {}) {
    const startTime = performance.now();
    
    try {
      const module = this.loadedModules.get(moduleName);
      if (!module) {
        throw new Error(`WASM module '${moduleName}' not loaded`);
      }

      // Update call count
      module.callCount++;
      
      // Get function from exports
      const func = module.exports[functionName];
      if (!func || typeof func !== 'function') {
        throw new Error(`Function '${functionName}' not found in module '${moduleName}'`);
      }

      // Call the function with appropriate argument handling
      let result;
      if (Object.keys(args).length === 0) {
        result = await func();
      } else {
        // Convert args to format expected by WASM
        const wasmArgs = this.prepareWasmArgs(args);
        result = await func(...wasmArgs);
      }

      // Update performance metrics
      const executionTime = performance.now() - startTime;
      this.updatePerformanceMetrics(moduleName, executionTime, true);
      
      this.emit('wasmFunctionCalled', {
        module: moduleName,
        function: functionName,
        args,
        result,
        executionTime
      });

      return result;

    } catch (error) {
      const executionTime = performance.now() - startTime;
      this.updatePerformanceMetrics(moduleName, executionTime, false);
      
      this.emit('wasmFunctionError', {
        module: moduleName,
        function: functionName,
        args,
        error,
        executionTime
      });
      
      throw error;
    }
  }

  prepareWasmArgs(args) {
    // Convert JavaScript objects to WASM-compatible format
    if (typeof args === 'object' && args !== null) {
      // For complex objects, serialize to JSON string
      return [JSON.stringify(args)];
    } else if (Array.isArray(args)) {
      // For arrays, pass individual elements
      return args;
    } else {
      // For primitives, pass as-is
      return [args];
    }
  }

  updatePerformanceMetrics(moduleName, executionTime, success) {
    const metrics = this.performanceMetrics.get(moduleName);
    if (metrics) {
      metrics.callCount++;
      metrics.totalTime += executionTime;
      metrics.averageTime = metrics.totalTime / metrics.callCount;
      
      if (!success) {
        metrics.errors++;
      }
    }
  }

  getPerformanceMetrics() {
    const metrics = {};
    
    for (const [moduleName, moduleMetrics] of this.performanceMetrics) {
      metrics[moduleName] = {
        ...moduleMetrics,
        errorRate: moduleMetrics.callCount > 0 ? 
          (moduleMetrics.errors / moduleMetrics.callCount) * 100 : 0
      };
    }

    return {
      modules: metrics,
      capabilities: {
        simd: this.supportsSIMD,
        threads: this.supportsThreads
      },
      totalModules: this.loadedModules.size
    };
  }

  async trainNeuralNetwork(trainingData, config = {}) {
    if (!this.loadedModules.has('ruv_swarm')) {
      throw new Error('ruv_swarm module required for neural training');
    }

    return await this.callWasmFunction('ruv_swarm', 'train_network', {
      training_data: trainingData,
      epochs: config.epochs || 100,
      learning_rate: config.learningRate || 0.01,
      batch_size: config.batchSize || 32,
      use_simd: this.supportsSIMD && config.useSIMD !== false
    });
  }

  async processConsensus(proposalData, participants) {
    if (!this.loadedModules.has('qudag')) {
      throw new Error('qudag module required for consensus processing');
    }

    return await this.callWasmFunction('qudag', 'process_consensus', {
      proposal: proposalData,
      participants: participants,
      consensus_type: 'dag'
    });
  }

  async optimizeTopology(currentTopology, constraints) {
    if (!this.loadedModules.has('daa')) {
      throw new Error('daa module required for topology optimization');
    }

    return await this.callWasmFunction('daa', 'optimize_topology', {
      current_topology: currentTopology,
      constraints: constraints,
      optimization_target: 'balanced'
    });
  }

  getLoadedModules() {
    return Array.from(this.loadedModules.keys());
  }

  getModuleInfo(moduleName) {
    const module = this.loadedModules.get(moduleName);
    const metrics = this.performanceMetrics.get(moduleName);
    
    if (!module) return null;
    
    return {
      name: moduleName,
      path: module.path,
      loadedAt: module.loadedAt,
      callCount: module.callCount,
      totalExecutionTime: module.totalExecutionTime,
      metrics: metrics || null,
      functions: Object.keys(module.exports).filter(key => 
        typeof module.exports[key] === 'function'
      )
    };
  }
}

export default WasmBridge;