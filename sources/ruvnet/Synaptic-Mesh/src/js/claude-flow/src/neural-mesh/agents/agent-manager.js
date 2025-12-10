/**
 * Neural Agent Manager
 * Manages lifecycle and coordination of neural agents
 */

import { EventEmitter } from 'events';
import { nanoid } from 'nanoid';

export class NeuralAgent extends EventEmitter {
  constructor(config) {
    super();
    this.id = config.id;
    this.type = config.type;
    this.capabilities = config.capabilities || [];
    this.status = 'created';
    this.connections = new Map();
    this.activeTasks = [];
    this.maxTasks = config.maxTasks || 5;
    this.memoryUsage = 0;
    this.availableMemory = config.memory || 256;
    this.currentLoad = 0;
    this.wasmBridge = config.wasmBridge;
    this.coordinator = config.coordinator;
    this.createdAt = Date.now();
    this.lastActivity = Date.now();
  }

  async start() {
    this.status = 'starting';
    
    // Initialize neural capabilities
    await this.initializeCapabilities();
    
    // Connect to mesh
    if (this.coordinator) {
      this.coordinator.emit('agentConnected', this);
    }
    
    this.status = 'ready';
    this.emit('started');
  }

  async stop(force = false) {
    this.status = force ? 'force_stopping' : 'stopping';
    
    // Complete or cancel active tasks
    if (!force && this.activeTasks.length > 0) {
      await this.finishActiveTasks();
    } else if (force) {
      this.cancelAllTasks();
    }
    
    // Cleanup connections
    this.connections.clear();
    
    this.status = 'stopped';
    this.emit('stopped');
    
    if (this.coordinator) {
      this.coordinator.emit('agentDisconnected', this);
    }
  }

  async assignTask(task) {
    if (this.activeTasks.length >= this.maxTasks) {
      throw new Error(`Agent ${this.id} at maximum task capacity`);
    }
    
    this.activeTasks.push(task);
    this.updateLoad();
    this.lastActivity = Date.now();
    
    this.emit('taskAssigned', { task });
    
    // Execute task asynchronously
    this.executeTask(task).catch(error => {
      this.emit('taskError', { task, error });
    });
  }

  async executeTask(task) {
    const startTime = Date.now();
    
    try {
      this.emit('taskStarted', { task });
      
      // Execute based on agent type and capabilities
      const result = await this.processTask(task);
      
      task.status = 'completed';
      task.result = result;
      task.completedAt = Date.now();
      task.duration = task.completedAt - startTime;
      
      this.removeActiveTask(task.id);
      this.emit('taskCompleted', { task, result });
      
      return result;
      
    } catch (error) {
      task.status = 'failed';
      task.error = error.message;
      task.failedAt = Date.now();
      
      this.removeActiveTask(task.id);
      this.emit('taskFailed', { task, error });
      
      throw error;
    }
  }

  async processTask(task) {
    // Process task based on agent type
    switch (this.type) {
      case 'neural':
        return await this.processNeuralTask(task);
      case 'cognitive':
        return await this.processCognitiveTask(task);
      case 'adaptive':
        return await this.processAdaptiveTask(task);
      default:
        return await this.processGenericTask(task);
    }
  }

  async processNeuralTask(task) {
    // Neural processing with WASM integration
    if (this.wasmBridge && task.useWasm) {
      return await this.executeWasmTask(task);
    }
    
    // Simulate neural computation
    await this.simulateComputation(task.complexity || 'medium');
    
    return {
      type: 'neural',
      processed: true,
      agentId: this.id,
      timestamp: Date.now()
    };
  }

  async processCognitiveTask(task) {
    // Cognitive reasoning and decision making
    await this.simulateComputation('high');
    
    return {
      type: 'cognitive',
      reasoning: 'Task analyzed and processed',
      decision: 'Optimal solution found',
      agentId: this.id,
      timestamp: Date.now()
    };
  }

  async processAdaptiveTask(task) {
    // Adaptive learning and optimization
    await this.simulateComputation('variable');
    
    return {
      type: 'adaptive',
      adaptation: 'Strategy optimized',
      learning: 'Patterns recognized',
      agentId: this.id,
      timestamp: Date.now()
    };
  }

  async processGenericTask(task) {
    // Generic task processing
    await this.simulateComputation('medium');
    
    return {
      type: 'generic',
      result: 'Task completed',
      agentId: this.id,
      timestamp: Date.now()
    };
  }

  async executeWasmTask(task) {
    if (!this.wasmBridge) {
      throw new Error('WASM bridge not available');
    }
    
    try {
      const result = await this.wasmBridge.executeWasmFunction(
        task.wasmModule,
        task.wasmFunction,
        task.wasmArgs || []
      );
      
      return {
        type: 'wasm',
        result,
        moduleId: task.wasmModule,
        function: task.wasmFunction,
        agentId: this.id,
        timestamp: Date.now()
      };
      
    } catch (error) {
      throw new Error(`WASM execution failed: ${error.message}`);
    }
  }

  async simulateComputation(complexity) {
    const delays = {
      low: 100,
      medium: 500,
      high: 1000,
      variable: Math.random() * 1000 + 100
    };
    
    const delay = delays[complexity] || delays.medium;
    await new Promise(resolve => setTimeout(resolve, delay));
  }

  async initializeCapabilities() {
    // Initialize capabilities based on agent type
    for (const capability of this.capabilities) {
      await this.initializeCapability(capability);
    }
  }

  async initializeCapability(capability) {
    // Initialize specific capability
    switch (capability) {
      case 'neural-processing':
        // Setup neural processing
        break;
      case 'machine-learning':
        // Setup ML capabilities
        break;
      case 'pattern-recognition':
        // Setup pattern recognition
        break;
      case 'optimization':
        // Setup optimization algorithms
        break;
      default:
        // Generic capability setup
        break;
    }
  }

  getTaskStatus(taskId) {
    const task = this.activeTasks.find(t => t.id === taskId);
    return task ? task.status : 'not_found';
  }

  removeActiveTask(taskId) {
    this.activeTasks = this.activeTasks.filter(t => t.id !== taskId);
    this.updateLoad();
  }

  updateLoad() {
    this.currentLoad = this.activeTasks.length / this.maxTasks;
    this.emit('loadChanged', { load: this.currentLoad });
  }

  async finishActiveTasks() {
    // Wait for all active tasks to complete
    while (this.activeTasks.length > 0) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }

  cancelAllTasks() {
    for (const task of this.activeTasks) {
      task.status = 'cancelled';
      this.emit('taskCancelled', { task });
    }
    this.activeTasks = [];
    this.updateLoad();
  }

  getMetrics() {
    return {
      id: this.id,
      type: this.type,
      status: this.status,
      activeTasks: this.activeTasks.length,
      maxTasks: this.maxTasks,
      currentLoad: this.currentLoad,
      memoryUsage: this.memoryUsage,
      availableMemory: this.availableMemory,
      connections: this.connections.size,
      uptime: Date.now() - this.createdAt,
      lastActivity: this.lastActivity
    };
  }
}

export class AgentManager extends EventEmitter {
  constructor() {
    super();
    this.agents = new Map();
    this.agentTypes = new Map();
    this.maxAgents = 8;
    this.wasmBridge = null;
  }

  async initialize(config = {}) {
    this.maxAgents = config.maxAgents || 8;
    this.wasmBridge = config.wasmBridge;
    
    // Register default agent types
    this.registerAgentTypes();
    
    this.emit('initialized');
  }

  registerAgentTypes() {
    this.agentTypes.set('neural', {
      defaultCapabilities: ['neural-processing', 'pattern-recognition'],
      memoryRequirement: 128,
      maxTasks: 3
    });
    
    this.agentTypes.set('cognitive', {
      defaultCapabilities: ['reasoning', 'decision-making'],
      memoryRequirement: 256,
      maxTasks: 5
    });
    
    this.agentTypes.set('adaptive', {
      defaultCapabilities: ['learning', 'optimization'],
      memoryRequirement: 512,
      maxTasks: 2
    });
  }

  async createAgent(config) {
    if (this.agents.size >= this.maxAgents) {
      throw new Error(`Maximum agent limit reached (${this.maxAgents})`);
    }
    
    const agentId = config.id || `agent-${nanoid(8)}`;
    
    // Get type configuration
    const typeConfig = this.agentTypes.get(config.type) || {};
    
    // Merge configurations
    const agentConfig = {
      ...typeConfig,
      ...config,
      id: agentId,
      wasmBridge: this.wasmBridge,
      capabilities: config.capabilities || typeConfig.defaultCapabilities || []
    };
    
    const agent = new NeuralAgent(agentConfig);
    
    // Setup event forwarding
    this.setupAgentEvents(agent);
    
    this.agents.set(agentId, agent);
    this.emit('agentCreated', { agent });
    
    // Start the agent
    await agent.start();
    
    return agent;
  }

  setupAgentEvents(agent) {
    agent.on('taskCompleted', (data) => {
      this.emit('agentTaskCompleted', { agent, ...data });
    });
    
    agent.on('taskFailed', (data) => {
      this.emit('agentTaskFailed', { agent, ...data });
    });
    
    agent.on('loadChanged', (data) => {
      this.emit('agentLoadChanged', { agent, ...data });
    });
  }

  getAgent(agentId) {
    return this.agents.get(agentId);
  }

  getAllAgents() {
    return Array.from(this.agents.values());
  }

  getAgentsByType(type) {
    return Array.from(this.agents.values()).filter(agent => agent.type === type);
  }

  getAvailableAgents() {
    return Array.from(this.agents.values()).filter(
      agent => agent.status === 'ready' && agent.currentLoad < 1.0
    );
  }

  async stopAgent(agentId, force = false) {
    const agent = this.agents.get(agentId);
    if (!agent) {
      throw new Error(`Agent ${agentId} not found`);
    }
    
    await agent.stop(force);
    this.agents.delete(agentId);
    this.emit('agentStopped', { agentId });
  }

  async stopAllAgents(force = false) {
    const stopPromises = Array.from(this.agents.keys()).map(
      agentId => this.stopAgent(agentId, force)
    );
    
    await Promise.all(stopPromises);
  }

  getManagerMetrics() {
    const agents = Array.from(this.agents.values());
    
    return {
      totalAgents: agents.length,
      maxAgents: this.maxAgents,
      agentsByType: this.getAgentCountByType(agents),
      agentsByStatus: this.getAgentCountByStatus(agents),
      totalLoad: agents.reduce((sum, agent) => sum + agent.currentLoad, 0),
      averageLoad: agents.length > 0 ? 
        agents.reduce((sum, agent) => sum + agent.currentLoad, 0) / agents.length : 0,
      totalActiveTasks: agents.reduce((sum, agent) => sum + agent.activeTasks.length, 0)
    };
  }

  getAgentCountByType(agents) {
    return agents.reduce((acc, agent) => {
      acc[agent.type] = (acc[agent.type] || 0) + 1;
      return acc;
    }, {});
  }

  getAgentCountByStatus(agents) {
    return agents.reduce((acc, agent) => {
      acc[agent.status] = (acc[agent.status] || 0) + 1;
      return acc;
    }, {});
  }
}

export default AgentManager;