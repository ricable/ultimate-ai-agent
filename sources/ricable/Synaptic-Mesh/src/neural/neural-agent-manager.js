/**
 * Neural Agent Manager - Manages ephemeral neural agents with lifecycle support
 * 
 * This module provides:
 * - Neural agent spawning and termination
 * - Memory management (<50MB per agent)
 * - Performance monitoring (<100ms inference)
 * - Cross-agent learning protocols
 * - WASM integration with SIMD optimization
 */

const path = require('path');
const EventEmitter = require('events');

class NeuralAgentManager extends EventEmitter {
    constructor(options = {}) {
        super();
        
        this.config = {
            maxAgents: options.maxAgents || 10,
            memoryLimitPerAgent: options.memoryLimitPerAgent || 50 * 1024 * 1024, // 50MB
            inferenceTimeout: options.inferenceTimeout || 100, // 100ms
            wasmModulePath: options.wasmModulePath || path.join(__dirname, '../js/synaptic-cli/wasm'),
            simdEnabled: options.simdEnabled !== false,
            crossLearningEnabled: options.crossLearningEnabled !== false,
            ...options
        };
        
        this.agents = new Map(); // agentId -> AgentInstance
        this.wasmModule = null;
        this.performanceMetrics = {
            totalSpawned: 0,
            totalTerminated: 0,
            averageInferenceTime: 0,
            memoryUsage: 0,
            crossLearningEvents: 0
        };
        
        this.memoryManager = new MemoryManager(this.config.maxAgents);
        this.learningProtocol = new CrossAgentLearningProtocol();
        
        this.init();
    }
    
    async init() {
        try {
            // Load WASM modules
            await this.loadWasmModules();
            this.emit('initialized');
        } catch (error) {
            this.emit('error', error);
        }
    }
    
    async loadWasmModules() {
        const wasmFiles = [
            'ruv-fann.wasm',
            'ruv_swarm_simd.wasm',
            'neuro-divergent.wasm'
        ];
        
        this.wasmModule = {};
        
        for (const file of wasmFiles) {
            try {
                const wasmPath = path.join(this.config.wasmModulePath, file);
                const wasmBuffer = require('fs').readFileSync(wasmPath);
                const wasmModule = await WebAssembly.instantiate(wasmBuffer);
                this.wasmModule[file.replace('.wasm', '')] = wasmModule;
            } catch (error) {
                console.warn(`Failed to load WASM module ${file}:`, error.message);
            }
        }
    }
    
    /**
     * Spawn a new neural agent
     * @param {Object} config - Agent configuration
     * @param {string} config.type - Agent type (mlp, lstm, cnn)
     * @param {Array} config.architecture - Network architecture [input, hidden..., output]
     * @param {string} config.activationFunction - Activation function
     * @param {Object} config.trainingData - Initial training data
     * @returns {Promise<string>} Agent ID
     */
    async spawnAgent(config) {
        const startTime = Date.now();
        
        if (this.agents.size >= this.config.maxAgents) {
            throw new Error(`Maximum agent limit (${this.config.maxAgents}) reached`);
        }
        
        const agentId = this.generateAgentId();
        
        try {
            // Allocate memory for agent
            const memory = await this.memoryManager.allocateAgentMemory(agentId, config);
            
            // Create neural network instance
            const network = await this.createNeuralNetwork(config, memory);
            
            // Initialize agent
            const agent = new NeuralAgent(agentId, network, memory, this.config);
            
            // Register agent
            this.agents.set(agentId, agent);
            
            // Start agent lifecycle
            await agent.initialize();
            
            // Update metrics
            this.performanceMetrics.totalSpawned++;
            const spawnTime = Date.now() - startTime;
            
            this.emit('agentSpawned', {
                agentId,
                spawnTime,
                config,
                memoryUsage: memory.getUsage()
            });
            
            return agentId;
            
        } catch (error) {
            // Cleanup on failure
            if (this.agents.has(agentId)) {
                await this.terminateAgent(agentId);
            }
            throw error;
        }
    }
    
    /**
     * Terminate a neural agent
     * @param {string} agentId - Agent ID
     */
    async terminateAgent(agentId) {
        const agent = this.agents.get(agentId);
        if (!agent) {
            throw new Error(`Agent ${agentId} not found`);
        }
        
        try {
            // Save learning state for cross-agent learning
            if (this.config.crossLearningEnabled) {
                await this.learningProtocol.saveAgentState(agent);
            }
            
            // Cleanup agent resources
            await agent.terminate();
            
            // Deallocate memory
            await this.memoryManager.deallocateAgentMemory(agentId);
            
            // Remove from registry
            this.agents.delete(agentId);
            
            // Update metrics
            this.performanceMetrics.totalTerminated++;
            
            this.emit('agentTerminated', { agentId });
            
        } catch (error) {
            this.emit('error', error);
            throw error;
        }
    }
    
    /**
     * Run inference on an agent
     * @param {string} agentId - Agent ID
     * @param {Array} inputs - Input data
     * @returns {Promise<Array>} Output data
     */
    async runInference(agentId, inputs) {
        const agent = this.agents.get(agentId);
        if (!agent) {
            throw new Error(`Agent ${agentId} not found`);
        }
        
        const startTime = Date.now();
        
        try {
            const outputs = await agent.inference(inputs, this.config.inferenceTimeout);
            
            const inferenceTime = Date.now() - startTime;
            
            // Update performance metrics
            this.updateInferenceMetrics(inferenceTime);
            
            this.emit('inferenceComplete', {
                agentId,
                inferenceTime,
                inputSize: inputs.length,
                outputSize: outputs.length
            });
            
            return outputs;
            
        } catch (error) {
            this.emit('error', error);
            throw error;
        }
    }
    
    /**
     * Train an agent with new data
     * @param {string} agentId - Agent ID
     * @param {Object} trainingData - Training data
     */
    async trainAgent(agentId, trainingData) {
        const agent = this.agents.get(agentId);
        if (!agent) {
            throw new Error(`Agent ${agentId} not found`);
        }
        
        try {
            await agent.train(trainingData);
            
            // Share learning with other agents if enabled
            if (this.config.crossLearningEnabled) {
                await this.learningProtocol.shareKnowledge(agent, this.agents);
                this.performanceMetrics.crossLearningEvents++;
            }
            
            this.emit('agentTrained', { agentId, trainingData });
            
        } catch (error) {
            this.emit('error', error);
            throw error;
        }
    }
    
    /**
     * Create neural network instance using WASM
     */
    async createNeuralNetwork(config, memory) {
        const { type, architecture, activationFunction } = config;
        
        // Select appropriate WASM module
        let wasmModule;
        switch (type) {
            case 'mlp':
                wasmModule = this.wasmModule['ruv-fann'];
                break;
            case 'lstm':
            case 'cnn':
                wasmModule = this.wasmModule['neuro-divergent'];
                break;
            default:
                wasmModule = this.wasmModule['ruv-fann'];
        }
        
        if (!wasmModule) {
            throw new Error(`WASM module not available for network type: ${type}`);
        }
        
        // Create network using WASM
        const network = new WasmNeuralNetwork(wasmModule, {
            type,
            architecture,
            activationFunction,
            memory,
            simdEnabled: this.config.simdEnabled
        });
        
        await network.initialize();
        return network;
    }
    
    generateAgentId() {
        return `agent_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    
    updateInferenceMetrics(inferenceTime) {
        const alpha = 0.1; // Exponential moving average factor
        this.performanceMetrics.averageInferenceTime = 
            alpha * inferenceTime + (1 - alpha) * this.performanceMetrics.averageInferenceTime;
    }
    
    /**
     * Get performance metrics
     */
    getMetrics() {
        return {
            ...this.performanceMetrics,
            activeAgents: this.agents.size,
            memoryUsage: this.memoryManager.getTotalUsage(),
            agents: Array.from(this.agents.keys())
        };
    }
    
    /**
     * Cleanup all agents
     */
    async cleanup() {
        const agentIds = Array.from(this.agents.keys());
        await Promise.all(agentIds.map(id => this.terminateAgent(id)));
        await this.memoryManager.cleanup();
    }
}

/**
 * Individual Neural Agent
 */
class NeuralAgent {
    constructor(id, network, memory, config) {
        this.id = id;
        this.network = network;
        this.memory = memory;
        this.config = config;
        this.state = 'initializing';
        this.createdAt = Date.now();
        this.lastInference = null;
    }
    
    async initialize() {
        await this.network.initialize();
        this.state = 'ready';
    }
    
    async inference(inputs, timeout) {
        if (this.state !== 'ready') {
            throw new Error(`Agent ${this.id} not ready for inference`);
        }
        
        const promise = this.network.run(inputs);
        const timeoutPromise = new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Inference timeout')), timeout)
        );
        
        const outputs = await Promise.race([promise, timeoutPromise]);
        this.lastInference = Date.now();
        return outputs;
    }
    
    async train(trainingData) {
        if (this.state !== 'ready') {
            throw new Error(`Agent ${this.id} not ready for training`);
        }
        
        await this.network.train(trainingData);
    }
    
    async terminate() {
        this.state = 'terminating';
        await this.network.cleanup();
        this.state = 'terminated';
    }
    
    getState() {
        return {
            id: this.id,
            state: this.state,
            createdAt: this.createdAt,
            lastInference: this.lastInference,
            memoryUsage: this.memory.getUsage()
        };
    }
}

/**
 * Memory Manager for Neural Agents
 */
class MemoryManager {
    constructor(maxAgents) {
        this.maxAgents = maxAgents;
        this.agentMemory = new Map();
        this.totalUsage = 0;
    }
    
    async allocateAgentMemory(agentId, config) {
        const estimatedSize = this.estimateMemoryUsage(config);
        
        if (this.totalUsage + estimatedSize > this.maxAgents * 50 * 1024 * 1024) {
            throw new Error('Memory limit exceeded');
        }
        
        const memory = new AgentMemory(agentId, estimatedSize);
        this.agentMemory.set(agentId, memory);
        this.totalUsage += estimatedSize;
        
        return memory;
    }
    
    async deallocateAgentMemory(agentId) {
        const memory = this.agentMemory.get(agentId);
        if (memory) {
            this.totalUsage -= memory.getSize();
            this.agentMemory.delete(agentId);
            await memory.cleanup();
        }
    }
    
    estimateMemoryUsage(config) {
        // Estimate based on network architecture
        const { architecture } = config;
        let weights = 0;
        
        for (let i = 0; i < architecture.length - 1; i++) {
            weights += architecture[i] * architecture[i + 1];
        }
        
        // 4 bytes per float32 weight + overhead
        return weights * 4 * 2; // weights + gradients
    }
    
    getTotalUsage() {
        return this.totalUsage;
    }
    
    async cleanup() {
        for (const memory of this.agentMemory.values()) {
            await memory.cleanup();
        }
        this.agentMemory.clear();
        this.totalUsage = 0;
    }
}

/**
 * Agent Memory Management
 */
class AgentMemory {
    constructor(agentId, size) {
        this.agentId = agentId;
        this.size = size;
        this.buffers = new Map();
        this.usage = 0;
    }
    
    allocate(name, size) {
        if (this.usage + size > this.size) {
            throw new Error(`Memory limit exceeded for agent ${this.agentId}`);
        }
        
        const buffer = new Float32Array(size / 4);
        this.buffers.set(name, buffer);
        this.usage += size;
        return buffer;
    }
    
    deallocate(name) {
        const buffer = this.buffers.get(name);
        if (buffer) {
            this.usage -= buffer.byteLength;
            this.buffers.delete(name);
        }
    }
    
    getUsage() {
        return this.usage;
    }
    
    getSize() {
        return this.size;
    }
    
    async cleanup() {
        this.buffers.clear();
        this.usage = 0;
    }
}

/**
 * WASM Neural Network Wrapper
 */
class WasmNeuralNetwork {
    constructor(wasmModule, config) {
        this.wasmModule = wasmModule;
        this.config = config;
        this.networkId = null;
        this.initialized = false;
    }
    
    async initialize() {
        const { instance } = this.wasmModule;
        const { type, architecture, activationFunction } = this.config;
        
        // Create network in WASM
        if (instance.exports.create_network) {
            this.networkId = instance.exports.create_network(
                new Uint32Array(architecture),
                this.getActivationCode(activationFunction)
            );
        }
        
        this.initialized = true;
    }
    
    async run(inputs) {
        if (!this.initialized) {
            throw new Error('Network not initialized');
        }
        
        const { instance } = this.wasmModule;
        
        // Allocate input buffer in WASM memory
        const inputPtr = instance.exports.allocate(inputs.length * 4);
        const inputArray = new Float32Array(instance.exports.memory.buffer, inputPtr, inputs.length);
        inputArray.set(inputs);
        
        // Run inference
        const outputPtr = instance.exports.run_network(this.networkId, inputPtr, inputs.length);
        const outputSize = this.config.architecture[this.config.architecture.length - 1];
        const outputs = new Float32Array(instance.exports.memory.buffer, outputPtr, outputSize);
        
        // Copy results
        const result = Array.from(outputs);
        
        // Cleanup WASM memory
        instance.exports.deallocate(inputPtr);
        instance.exports.deallocate(outputPtr);
        
        return result;
    }
    
    async train(trainingData) {
        if (!this.initialized) {
            throw new Error('Network not initialized');
        }
        
        const { instance } = this.wasmModule;
        const { inputs, outputs, epochs = 100, learningRate = 0.1 } = trainingData;
        
        // Train network in WASM
        if (instance.exports.train_network) {
            instance.exports.train_network(
                this.networkId,
                inputs,
                outputs,
                epochs,
                learningRate
            );
        }
    }
    
    getActivationCode(activationFunction) {
        const codes = {
            'linear': 0,
            'sigmoid': 1,
            'relu': 2,
            'tanh': 3,
            'gaussian': 4
        };
        return codes[activationFunction] || 1; // Default to sigmoid
    }
    
    async cleanup() {
        if (this.initialized && this.networkId !== null) {
            const { instance } = this.wasmModule;
            if (instance.exports.destroy_network) {
                instance.exports.destroy_network(this.networkId);
            }
        }
    }
}

/**
 * Cross-Agent Learning Protocol
 */
class CrossAgentLearningProtocol {
    constructor() {
        this.knowledgeBase = new Map();
    }
    
    async saveAgentState(agent) {
        // Save agent's learned patterns for sharing
        const state = {
            weights: await agent.network.getWeights(),
            performance: agent.getPerformance(),
            timestamp: Date.now()
        };
        
        this.knowledgeBase.set(agent.id, state);
    }
    
    async shareKnowledge(sourceAgent, allAgents) {
        // Transfer learning between agents
        const sourceState = this.knowledgeBase.get(sourceAgent.id);
        if (!sourceState) return;
        
        for (const [agentId, agent] of allAgents) {
            if (agentId !== sourceAgent.id && agent.state === 'ready') {
                try {
                    await this.transferKnowledge(sourceState, agent);
                } catch (error) {
                    console.warn(`Failed to transfer knowledge to agent ${agentId}:`, error.message);
                }
            }
        }
    }
    
    async transferKnowledge(sourceState, targetAgent) {
        // Simple weight averaging for now - can be enhanced with more sophisticated methods
        const targetWeights = await targetAgent.network.getWeights();
        const sourceWeights = sourceState.weights;
        
        if (targetWeights.length === sourceWeights.length) {
            const blendedWeights = targetWeights.map((w, i) => 
                0.9 * w + 0.1 * sourceWeights[i]
            );
            
            await targetAgent.network.setWeights(blendedWeights);
        }
    }
}

module.exports = {
    NeuralAgentManager,
    NeuralAgent,
    MemoryManager,
    WasmNeuralNetwork,
    CrossAgentLearningProtocol
};