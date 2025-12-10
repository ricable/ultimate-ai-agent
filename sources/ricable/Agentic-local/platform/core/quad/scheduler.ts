/**
 * QUAD - Quantum-Inspired Distributed Agent Graph Scheduler
 *
 * Advanced DAG-based task scheduling for distributed AI agents
 * with fault tolerance, dynamic rebalancing, and priority queuing.
 */

import { EventEmitter } from 'events';

// ============================================================================
// TYPES & INTERFACES
// ============================================================================

interface QDAGNode {
  id: string;
  type: 'task' | 'agent' | 'checkpoint' | 'merge';
  dependencies: string[];
  priority: number;
  state: 'pending' | 'running' | 'completed' | 'failed' | 'retrying';
  retries: number;
  maxRetries: number;
  payload: TaskPayload;
  assignedAgent?: string;
  startTime?: number;
  endTime?: number;
  result?: any;
  error?: Error;
}

interface TaskPayload {
  action: string;
  input: any;
  context?: Record<string, any>;
  timeout?: number;
  resourceRequirements?: ResourceRequirements;
}

interface ResourceRequirements {
  cpu?: number;      // millicores
  memory?: number;   // MB
  gpu?: boolean;
  gpuMemory?: number; // MB
}

interface AgentCapabilities {
  id: string;
  type: string;
  status: 'idle' | 'busy' | 'offline';
  capabilities: string[];
  resources: {
    cpu: number;
    memory: number;
    hasGpu: boolean;
    gpuMemory?: number;
  };
  currentLoad: number;
  completedTasks: number;
  failedTasks: number;
  averageLatency: number;
  cluster: string;
  location: 'local' | 'edge' | 'cloud';
}

interface SchedulerConfig {
  maxConcurrentTasks: number;
  defaultTimeout: number;
  maxRetries: number;
  loadBalanceStrategy: 'round-robin' | 'least-loaded' | 'capability-match' | 'locality-aware';
  priorityLevels: number;
  enablePreemption: boolean;
  checkpointInterval: number;
  enableSpeculativeExecution: boolean;
}

interface DAGExecution {
  id: string;
  dag: Map<string, QDAGNode>;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'paused';
  startTime: number;
  progress: number;
  completedNodes: Set<string>;
  runningNodes: Set<string>;
}

// ============================================================================
// QUAD SCHEDULER IMPLEMENTATION
// ============================================================================

export class QUADScheduler extends EventEmitter {
  private config: SchedulerConfig;
  private agents: Map<string, AgentCapabilities> = new Map();
  private executions: Map<string, DAGExecution> = new Map();
  private taskQueue: QDAGNode[] = [];
  private priorityQueues: Map<number, QDAGNode[]> = new Map();
  private checkpoints: Map<string, any> = new Map();

  constructor(config: Partial<SchedulerConfig> = {}) {
    super();
    this.config = {
      maxConcurrentTasks: 100,
      defaultTimeout: 30000,
      maxRetries: 3,
      loadBalanceStrategy: 'locality-aware',
      priorityLevels: 5,
      enablePreemption: true,
      checkpointInterval: 5000,
      enableSpeculativeExecution: true,
      ...config,
    };

    // Initialize priority queues
    for (let i = 0; i < this.config.priorityLevels; i++) {
      this.priorityQueues.set(i, []);
    }

    this.startSchedulerLoop();
    this.startCheckpointLoop();
  }

  // ==========================================================================
  // DAG MANAGEMENT
  // ==========================================================================

  /**
   * Create a new DAG from task definitions
   */
  createDAG(tasks: Array<{
    id: string;
    action: string;
    input: any;
    dependencies?: string[];
    priority?: number;
    requirements?: ResourceRequirements;
  }>): string {
    const dagId = `dag_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const dag = new Map<string, QDAGNode>();

    for (const task of tasks) {
      const node: QDAGNode = {
        id: task.id,
        type: 'task',
        dependencies: task.dependencies || [],
        priority: task.priority ?? 2,
        state: 'pending',
        retries: 0,
        maxRetries: this.config.maxRetries,
        payload: {
          action: task.action,
          input: task.input,
          resourceRequirements: task.requirements,
        },
      };
      dag.set(task.id, node);
    }

    // Validate DAG (no cycles, valid dependencies)
    this.validateDAG(dag);

    const execution: DAGExecution = {
      id: dagId,
      dag,
      status: 'pending',
      startTime: Date.now(),
      progress: 0,
      completedNodes: new Set(),
      runningNodes: new Set(),
    };

    this.executions.set(dagId, execution);
    this.emit('dag:created', { dagId, nodeCount: dag.size });

    return dagId;
  }

  /**
   * Validate DAG structure
   */
  private validateDAG(dag: Map<string, QDAGNode>): void {
    const visited = new Set<string>();
    const recursionStack = new Set<string>();

    const hasCycle = (nodeId: string): boolean => {
      visited.add(nodeId);
      recursionStack.add(nodeId);

      const node = dag.get(nodeId);
      if (!node) throw new Error(`Node ${nodeId} not found in DAG`);

      for (const dep of node.dependencies) {
        if (!dag.has(dep)) {
          throw new Error(`Dependency ${dep} not found for node ${nodeId}`);
        }
        if (!visited.has(dep)) {
          if (hasCycle(dep)) return true;
        } else if (recursionStack.has(dep)) {
          return true;
        }
      }

      recursionStack.delete(nodeId);
      return false;
    };

    for (const nodeId of dag.keys()) {
      if (!visited.has(nodeId)) {
        if (hasCycle(nodeId)) {
          throw new Error('DAG contains a cycle');
        }
      }
    }
  }

  /**
   * Execute a DAG
   */
  async executeDAG(dagId: string): Promise<Map<string, any>> {
    const execution = this.executions.get(dagId);
    if (!execution) throw new Error(`DAG ${dagId} not found`);

    execution.status = 'running';
    this.emit('dag:started', { dagId });

    return new Promise((resolve, reject) => {
      const checkCompletion = () => {
        if (execution.status === 'completed') {
          const results = new Map<string, any>();
          for (const [nodeId, node] of execution.dag) {
            results.set(nodeId, node.result);
          }
          resolve(results);
        } else if (execution.status === 'failed') {
          const failedNodes = [...execution.dag.values()].filter(n => n.state === 'failed');
          reject(new Error(`DAG failed: ${failedNodes.map(n => n.id).join(', ')}`));
        }
      };

      this.on(`dag:${dagId}:completed`, checkCompletion);
      this.on(`dag:${dagId}:failed`, checkCompletion);

      // Queue initial tasks (no dependencies)
      this.queueReadyTasks(dagId);
    });
  }

  /**
   * Queue tasks that are ready to run
   */
  private queueReadyTasks(dagId: string): void {
    const execution = this.executions.get(dagId);
    if (!execution || execution.status !== 'running') return;

    for (const [nodeId, node] of execution.dag) {
      if (node.state !== 'pending') continue;

      // Check if all dependencies are completed
      const depsCompleted = node.dependencies.every(depId => {
        const dep = execution.dag.get(depId);
        return dep && dep.state === 'completed';
      });

      if (depsCompleted) {
        this.enqueueTask(dagId, nodeId);
      }
    }
  }

  /**
   * Add task to priority queue
   */
  private enqueueTask(dagId: string, nodeId: string): void {
    const execution = this.executions.get(dagId);
    if (!execution) return;

    const node = execution.dag.get(nodeId);
    if (!node) return;

    const priority = Math.min(node.priority, this.config.priorityLevels - 1);
    const queue = this.priorityQueues.get(priority);

    if (queue) {
      queue.push({ ...node, payload: { ...node.payload, context: { dagId, nodeId } } });
      this.emit('task:queued', { dagId, nodeId, priority });
    }
  }

  // ==========================================================================
  // AGENT MANAGEMENT
  // ==========================================================================

  /**
   * Register an agent with the scheduler
   */
  registerAgent(agent: AgentCapabilities): void {
    this.agents.set(agent.id, agent);
    this.emit('agent:registered', { agentId: agent.id });
  }

  /**
   * Update agent status
   */
  updateAgentStatus(agentId: string, status: Partial<AgentCapabilities>): void {
    const agent = this.agents.get(agentId);
    if (agent) {
      Object.assign(agent, status);
      this.emit('agent:updated', { agentId, status });
    }
  }

  /**
   * Remove agent from scheduler
   */
  deregisterAgent(agentId: string): void {
    this.agents.delete(agentId);
    this.emit('agent:deregistered', { agentId });

    // Reassign any tasks from this agent
    this.reassignAgentTasks(agentId);
  }

  /**
   * Reassign tasks from a failed/removed agent
   */
  private reassignAgentTasks(agentId: string): void {
    for (const [dagId, execution] of this.executions) {
      for (const [nodeId, node] of execution.dag) {
        if (node.assignedAgent === agentId && node.state === 'running') {
          node.state = 'pending';
          node.assignedAgent = undefined;
          node.retries++;
          execution.runningNodes.delete(nodeId);

          if (node.retries <= node.maxRetries) {
            this.enqueueTask(dagId, nodeId);
            this.emit('task:reassigned', { dagId, nodeId, reason: 'agent_removed' });
          } else {
            node.state = 'failed';
            this.checkDAGCompletion(dagId);
          }
        }
      }
    }
  }

  // ==========================================================================
  // SCHEDULING LOGIC
  // ==========================================================================

  /**
   * Main scheduler loop
   */
  private startSchedulerLoop(): void {
    setInterval(() => {
      this.scheduleNextBatch();
    }, 100); // 100ms scheduling interval
  }

  /**
   * Schedule next batch of tasks
   */
  private scheduleNextBatch(): void {
    const idleAgents = [...this.agents.values()].filter(a => a.status === 'idle');
    if (idleAgents.length === 0) return;

    // Process priority queues from highest to lowest
    for (let priority = 0; priority < this.config.priorityLevels; priority++) {
      const queue = this.priorityQueues.get(priority);
      if (!queue || queue.length === 0) continue;

      while (queue.length > 0 && idleAgents.length > 0) {
        const task = queue.shift()!;
        const agent = this.selectAgent(task, idleAgents);

        if (agent) {
          this.assignTask(task, agent);
          idleAgents.splice(idleAgents.indexOf(agent), 1);
        } else {
          // No suitable agent, put back in queue
          queue.unshift(task);
          break;
        }
      }
    }
  }

  /**
   * Select best agent for a task based on strategy
   */
  private selectAgent(task: QDAGNode, availableAgents: AgentCapabilities[]): AgentCapabilities | null {
    const requirements = task.payload.resourceRequirements;

    // Filter agents that meet resource requirements
    let candidates = availableAgents.filter(agent => {
      if (requirements?.gpu && !agent.resources.hasGpu) return false;
      if (requirements?.memory && agent.resources.memory < requirements.memory) return false;
      return true;
    });

    if (candidates.length === 0) return null;

    switch (this.config.loadBalanceStrategy) {
      case 'round-robin':
        return candidates[0];

      case 'least-loaded':
        return candidates.sort((a, b) => a.currentLoad - b.currentLoad)[0];

      case 'capability-match':
        // Score agents based on capability match
        return candidates.sort((a, b) => {
          const scoreA = a.capabilities.filter(c => task.payload.action.includes(c)).length;
          const scoreB = b.capabilities.filter(c => task.payload.action.includes(c)).length;
          return scoreB - scoreA;
        })[0];

      case 'locality-aware':
        // Prefer local agents, then edge, then cloud
        const localityOrder = { local: 0, edge: 1, cloud: 2 };
        return candidates.sort((a, b) => {
          const locDiff = localityOrder[a.location] - localityOrder[b.location];
          if (locDiff !== 0) return locDiff;
          return a.averageLatency - b.averageLatency;
        })[0];

      default:
        return candidates[0];
    }
  }

  /**
   * Assign task to agent
   */
  private async assignTask(task: QDAGNode, agent: AgentCapabilities): Promise<void> {
    const context = task.payload.context as { dagId: string; nodeId: string };
    const execution = this.executions.get(context.dagId);
    if (!execution) return;

    const node = execution.dag.get(context.nodeId);
    if (!node) return;

    node.state = 'running';
    node.assignedAgent = agent.id;
    node.startTime = Date.now();
    execution.runningNodes.add(context.nodeId);

    agent.status = 'busy';
    agent.currentLoad++;

    this.emit('task:assigned', {
      dagId: context.dagId,
      nodeId: context.nodeId,
      agentId: agent.id,
    });

    // Execute task (simulated - in real impl, send to agent)
    try {
      const result = await this.executeTask(task, agent);
      this.completeTask(context.dagId, context.nodeId, result);
    } catch (error) {
      this.failTask(context.dagId, context.nodeId, error as Error);
    }
  }

  /**
   * Execute task on agent (abstracted)
   */
  private async executeTask(task: QDAGNode, agent: AgentCapabilities): Promise<any> {
    // In real implementation, this would send to the agent via A2A protocol
    const timeout = task.payload.timeout || this.config.defaultTimeout;

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error('Task timeout'));
      }, timeout);

      // Simulated execution
      setTimeout(() => {
        clearTimeout(timer);
        resolve({ success: true, action: task.payload.action });
      }, Math.random() * 1000);
    });
  }

  /**
   * Mark task as completed
   */
  private completeTask(dagId: string, nodeId: string, result: any): void {
    const execution = this.executions.get(dagId);
    if (!execution) return;

    const node = execution.dag.get(nodeId);
    if (!node) return;

    node.state = 'completed';
    node.result = result;
    node.endTime = Date.now();
    execution.completedNodes.add(nodeId);
    execution.runningNodes.delete(nodeId);

    // Update agent stats
    if (node.assignedAgent) {
      const agent = this.agents.get(node.assignedAgent);
      if (agent) {
        agent.status = 'idle';
        agent.currentLoad--;
        agent.completedTasks++;
        const latency = node.endTime - (node.startTime || 0);
        agent.averageLatency = (agent.averageLatency * (agent.completedTasks - 1) + latency) / agent.completedTasks;
      }
    }

    // Update progress
    execution.progress = (execution.completedNodes.size / execution.dag.size) * 100;

    this.emit('task:completed', { dagId, nodeId, result });

    // Queue dependent tasks
    this.queueReadyTasks(dagId);

    // Check if DAG is complete
    this.checkDAGCompletion(dagId);
  }

  /**
   * Mark task as failed
   */
  private failTask(dagId: string, nodeId: string, error: Error): void {
    const execution = this.executions.get(dagId);
    if (!execution) return;

    const node = execution.dag.get(nodeId);
    if (!node) return;

    node.retries++;
    execution.runningNodes.delete(nodeId);

    // Update agent stats
    if (node.assignedAgent) {
      const agent = this.agents.get(node.assignedAgent);
      if (agent) {
        agent.status = 'idle';
        agent.currentLoad--;
        agent.failedTasks++;
      }
    }

    if (node.retries <= node.maxRetries) {
      node.state = 'retrying';
      node.assignedAgent = undefined;
      this.emit('task:retrying', { dagId, nodeId, attempt: node.retries });

      // Re-queue with delay
      setTimeout(() => {
        node.state = 'pending';
        this.enqueueTask(dagId, nodeId);
      }, Math.pow(2, node.retries) * 1000); // Exponential backoff
    } else {
      node.state = 'failed';
      node.error = error;
      this.emit('task:failed', { dagId, nodeId, error: error.message });
      this.checkDAGCompletion(dagId);
    }
  }

  /**
   * Check if DAG execution is complete
   */
  private checkDAGCompletion(dagId: string): void {
    const execution = this.executions.get(dagId);
    if (!execution) return;

    const allCompleted = [...execution.dag.values()].every(n => n.state === 'completed');
    const anyFailed = [...execution.dag.values()].some(n => n.state === 'failed');

    if (allCompleted) {
      execution.status = 'completed';
      this.emit('dag:completed', { dagId, progress: 100 });
      this.emit(`dag:${dagId}:completed`);
    } else if (anyFailed && execution.runningNodes.size === 0) {
      // Check if any remaining tasks can run
      const canContinue = [...execution.dag.values()].some(n => {
        if (n.state !== 'pending') return false;
        return n.dependencies.every(depId => {
          const dep = execution.dag.get(depId);
          return dep && dep.state === 'completed';
        });
      });

      if (!canContinue) {
        execution.status = 'failed';
        this.emit('dag:failed', { dagId });
        this.emit(`dag:${dagId}:failed`);
      }
    }
  }

  // ==========================================================================
  // CHECKPOINTING
  // ==========================================================================

  private startCheckpointLoop(): void {
    setInterval(() => {
      this.createCheckpoints();
    }, this.config.checkpointInterval);
  }

  private createCheckpoints(): void {
    for (const [dagId, execution] of this.executions) {
      if (execution.status === 'running') {
        const checkpoint = {
          timestamp: Date.now(),
          progress: execution.progress,
          completedNodes: [...execution.completedNodes],
          nodeStates: [...execution.dag.entries()].map(([id, node]) => ({
            id,
            state: node.state,
            result: node.result,
          })),
        };
        this.checkpoints.set(dagId, checkpoint);
        this.emit('checkpoint:created', { dagId, progress: execution.progress });
      }
    }
  }

  /**
   * Restore DAG from checkpoint
   */
  restoreFromCheckpoint(dagId: string): void {
    const checkpoint = this.checkpoints.get(dagId);
    const execution = this.executions.get(dagId);

    if (!checkpoint || !execution) return;

    for (const nodeState of checkpoint.nodeStates) {
      const node = execution.dag.get(nodeState.id);
      if (node && nodeState.state === 'completed') {
        node.state = 'completed';
        node.result = nodeState.result;
        execution.completedNodes.add(nodeState.id);
      }
    }

    execution.status = 'running';
    this.queueReadyTasks(dagId);
    this.emit('dag:restored', { dagId, fromProgress: checkpoint.progress });
  }

  // ==========================================================================
  // PUBLIC API
  // ==========================================================================

  getDAGStatus(dagId: string): DAGExecution | undefined {
    return this.executions.get(dagId);
  }

  getAllAgents(): AgentCapabilities[] {
    return [...this.agents.values()];
  }

  getSchedulerStats(): {
    totalAgents: number;
    idleAgents: number;
    activeDAGs: number;
    queuedTasks: number;
  } {
    return {
      totalAgents: this.agents.size,
      idleAgents: [...this.agents.values()].filter(a => a.status === 'idle').length,
      activeDAGs: [...this.executions.values()].filter(e => e.status === 'running').length,
      queuedTasks: [...this.priorityQueues.values()].reduce((sum, q) => sum + q.length, 0),
    };
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export default QUADScheduler;
