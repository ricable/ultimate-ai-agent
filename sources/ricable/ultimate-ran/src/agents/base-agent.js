/**
 * Base Agent Class
 * Foundation for all agents in the Titan Cognitive Mesh
 */

import { EventEmitter } from 'events';

export class BaseAgent extends EventEmitter {
  constructor(config) {
    super();
    this.id = config.id || `agent-${Date.now()}`;
    this.type = config.type;
    this.role = config.role;
    this.capabilities = config.capabilities || [];
    this.tools = config.tools || [];
    this.status = 'initialized';
    this.createdAt = new Date().toISOString();
  }

  /**
   * Initialize the agent
   */
  async initialize() {
    console.log(`[${this.type.toUpperCase()}] Initializing agent ${this.id}...`);
    this.status = 'ready';
  }

  /**
   * Execute a task
   */
  async execute(task) {
    console.log(`[${this.type.toUpperCase()}] Executing task: ${task.id}`);
    this.status = 'executing';

    try {
      const result = await this.processTask(task);
      this.status = 'completed';
      return result;
    } catch (error) {
      this.status = 'failed';
      throw error;
    }
  }

  /**
   * Process task - to be overridden by subclasses
   */
  async processTask(task) {
    throw new Error('processTask must be implemented by subclass');
  }

  /**
   * Emit an AG-UI event
   */
  emitAGUI(eventType, payload) {
    this.emit('agui', { type: eventType, payload, agentId: this.id });
  }

  /**
   * Log a reflexion (self-critique) to AgentDB
   */
  async logReflexion(action, result, critique) {
    console.log(`[${this.type.toUpperCase()}] Reflexion: ${critique}`);
    this.emit('reflexion', { action, result, critique });
  }
}
