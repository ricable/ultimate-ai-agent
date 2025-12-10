/**
 * Swarm Performance Monitoring System
 *
 * Real-time monitoring of agent coordination, task distribution,
 * communication patterns, and swarm health metrics
 */

import { SwarmPerformanceMetrics, PerformanceSnapshot } from '../metrics/MLPerformanceMetrics';
import { EventEmitter } from 'events';

export interface AgentStatus {
  id: string;
  type: string;
  status: 'active' | 'idle' | 'busy' | 'failed' | 'recovering';
  currentTask?: string;
  lastHeartbeat: Date;
  resourceUsage: {
    cpu: number;
    memory: number;
    network: number;
  };
  performance: {
    tasksCompleted: number;
    averageTaskDuration: number;
    successRate: number;
    errorCount: number;
  };
  capabilities: string[];
  location?: string;
}

export interface TaskMetrics {
  id: string;
  type: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  assignedAgent?: string;
  createdAt: Date;
  startedAt?: Date;
  completedAt?: Date;
  duration?: number;
  priority: 'low' | 'medium' | 'high' | 'critical';
  complexity: 'simple' | 'moderate' | 'complex' | 'expert';
  dependencies: string[];
  resourceRequirements: {
    cpu: number;
    memory: number;
    specializedCapability?: string;
  };
  result?: any;
  error?: string;
}

export interface CommunicationMetrics {
  sourceAgent: string;
  targetAgent: string;
  messageType: 'task_assignment' | 'result_sharing' | 'coordination' | 'heartbeat' | 'data_sync';
  timestamp: Date;
  latency: number;
  dataSize: number;
  success: boolean;
  retryCount?: number;
}

export interface SwarmTopology {
  type: 'hierarchical' | 'mesh' | 'ring' | 'star' | 'adaptive';
  agents: AgentNode[];
  connections: AgentConnection[];
  efficiency: number;
  bottleneckNodes: string[];
  optimizationSuggestions: string[];
}

export interface AgentNode {
  id: string;
  type: string;
  status: AgentStatus['status'];
  capabilities: string[];
  connections: string[];
  load: number;
  performance: number;
}

export interface AgentConnection {
  source: string;
  target: string;
  latency: number;
  bandwidth: number;
  reliability: number;
  messageCount: number;
}

export interface SwarmHealthIndicator {
  overallHealth: number; // 0-100
  agentHealth: {
    healthyAgents: number;
    totalAgents: number;
    failedAgents: number;
    recoveringAgents: number;
  };
  taskHealth: {
    tasksInQueue: number;
    runningTasks: number;
    completedTasks: number;
    failedTasks: number;
    averageWaitTime: number;
  };
  communicationHealth: {
    averageLatency: number;
    messageSuccessRate: number;
    activeConnections: number;
    networkUtilization: number;
  };
  resourceHealth: {
    cpuUtilization: number;
    memoryUtilization: number;
    networkUtilization: number;
    resourceContention: boolean;
  };
}

export class SwarmMonitor extends EventEmitter {
  private agents: Map<string, AgentStatus> = new Map();
  private tasks: Map<string, TaskMetrics> = new Map();
  private communications: CommunicationMetrics[] = [];
  private topology: SwarmTopology | null = null;
  private healthHistory: SwarmHealthIndicator[] = [];
  private monitoringInterval: NodeJS.Timeout | null = null;
  private maxCommunicationHistory: number = 10000;

  constructor() {
    super();
    this.startMonitoring();
  }

  public registerAgent(agent: AgentStatus): void {
    this.agents.set(agent.id, agent);
    this.emit('agent_registered', agent);
    this.updateTopology();
  }

  public updateAgentStatus(agentId: string, updates: Partial<AgentStatus>): void {
    const agent = this.agents.get(agentId);
    if (agent) {
      Object.assign(agent, updates);
      agent.lastHeartbeat = new Date();
      this.emit('agent_updated', agent);
      this.updateTopology();
    }
  }

  public unregisterAgent(agentId: string): void {
    const agent = this.agents.get(agentId);
    if (agent) {
      this.agents.delete(agentId);
      this.emit('agent_unregistered', agent);
      this.updateTopology();
    }
  }

  public createTask(task: Omit<TaskMetrics, 'id'>): string {
    const taskId = `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const fullTask: TaskMetrics = {
      ...task,
      id: taskId
    };

    this.tasks.set(taskId, fullTask);
    this.emit('task_created', fullTask);
    return taskId;
  }

  public updateTask(taskId: string, updates: Partial<TaskMetrics>): void {
    const task = this.tasks.get(taskId);
    if (task) {
      Object.assign(task, updates);
      this.emit('task_updated', task);
    }
  }

  public completeTask(taskId: string, result?: any, error?: string): void {
    const task = this.tasks.get(taskId);
    if (task) {
      task.status = error ? 'failed' : 'completed';
      task.completedAt = new Date();
      task.duration = task.completedAt.getTime() - (task.startedAt?.getTime() || task.createdAt.getTime());
      task.result = result;
      task.error = error;

      // Update agent performance metrics
      if (task.assignedAgent) {
        const agent = this.agents.get(task.assignedAgent);
        if (agent) {
          agent.performance.tasksCompleted++;
          if (task.duration) {
            agent.performance.averageTaskDuration =
              (agent.performance.averageTaskDuration * (agent.performance.tasksCompleted - 1) + task.duration) /
              agent.performance.tasksCompleted;
          }
          if (error) {
            agent.performance.errorCount++;
          }
          agent.performance.successRate =
            (agent.performance.tasksCompleted - agent.performance.errorCount) / agent.performance.tasksCompleted;
        }
      }

      this.emit('task_completed', task);
    }
  }

  public recordCommunication(communication: Omit<CommunicationMetrics, 'timestamp'>): void {
    const fullCommunication: CommunicationMetrics = {
      ...communication,
      timestamp: new Date()
    };

    this.communications.push(fullCommunication);

    // Maintain communication history size
    if (this.communications.length > this.maxCommunicationHistory) {
      this.communications.shift();
    }

    this.emit('communication_recorded', fullCommunication);
  }

  public getSwarmMetrics(): SwarmPerformanceMetrics {
    const now = new Date();
    const fiveMinutesAgo = new Date(now.getTime() - 5 * 60 * 1000);

    // Filter active agents (heartbeat within last 5 minutes)
    const activeAgents = Array.from(this.agents.values())
      .filter(agent => agent.lastHeartbeat > fiveMinutesAgo);

    // Calculate agent state metrics
    const agentStates = {
      activeAgents: activeAgents.filter(a => a.status === 'active').length,
      idleAgents: activeAgents.filter(a => a.status === 'idle').length,
      busyAgents: activeAgents.filter(a => a.status === 'busy').length,
      failedAgents: activeAgents.filter(a => a.status === 'failed').length,
      agentUtilizationRate: activeAgents.length > 0
        ? activeAgents.filter(a => a.status === 'busy').length / activeAgents.length
        : 0
    };

    // Calculate task performance metrics
    const recentTasks = Array.from(this.tasks.values())
      .filter(task => task.createdAt > fiveMinutesAgo);

    const completedTasks = recentTasks.filter(task => task.status === 'completed');
    const runningTasks = recentTasks.filter(task => task.status === 'running');

    const taskPerformance = {
      taskCompletionRate: recentTasks.length > 0
        ? completedTasks.length / recentTasks.length
        : 0,
      averageTaskDuration: completedTasks.length > 0
        ? completedTasks.reduce((sum, task) => sum + (task.duration || 0), 0) / completedTasks.length
        : 0,
      taskQueueLength: recentTasks.filter(task => task.status === 'pending').length,
      throughput: completedTasks.length / 5, // tasks per minute
      errorRate: recentTasks.length > 0
        ? recentTasks.filter(task => task.status === 'failed').length / recentTasks.length
        : 0
    };

    // Calculate agent coordination metrics
    const recentCommunications = this.communications
      .filter(comm => comm.timestamp > fiveMinutesAgo);

    const coordinationMetrics = {
      topologyEfficiency: this.topology?.efficiency || 0.8,
      communicationLatency: recentCommunications.length > 0
        ? recentCommunications.reduce((sum, comm) => sum + comm.latency, 0) / recentCommunications.length
        : 0,
      taskDistributionBalance: this.calculateTaskDistributionBalance(activeAgents),
      consensusSpeed: this.estimateConsensusSpeed(recentCommunications),
      synchronizationAccuracy: this.calculateSynchronizationAccuracy(recentCommunications)
    };

    // Calculate resource utilization
    const resourceUtilization = {
      cpuUsage: activeAgents.length > 0
        ? activeAgents.reduce((sum, agent) => sum + agent.resourceUsage.cpu, 0) / activeAgents.length
        : 0,
      memoryUsage: activeAgents.length > 0
        ? activeAgents.reduce((sum, agent) => sum + agent.resourceUsage.memory, 0) / activeAgents.length
        : 0,
      networkBandwidth: this.calculateNetworkUtilization(recentCommunications),
      diskIOPS: this.estimateDiskIOPS(recentTasks),
      gpuUtilization: activeAgents.length > 0
        ? activeAgents.reduce((sum, agent) => sum + agent.resourceUsage.network, 0) / activeAgents.length
        : 0
    };

    return {
      agentCoordination: coordinationMetrics,
      agentStates,
      taskPerformance,
      resourceUtilization
    };
  }

  private calculateTaskDistributionBalance(agents: AgentStatus[]): number {
    if (agents.length === 0) return 0;

    const taskCounts = agents.map(agent =>
      agent.status === 'busy' ? 1 : 0
    );

    const averageTasks = taskCounts.reduce((sum, count) => sum + count, 0) / agents.length;
    const variance = taskCounts.reduce((sum, count) => sum + Math.pow(count - averageTasks, 2), 0) / agents.length;
    const standardDeviation = Math.sqrt(variance);

    // Perfect balance = 1, completely imbalanced = 0
    return Math.max(0, 1 - (standardDeviation / Math.max(1, averageTasks)));
  }

  private estimateConsensusSpeed(communications: CommunicationMetrics[]): number {
    const coordinationMessages = communications.filter(comm =>
      comm.messageType === 'coordination'
    );

    if (coordinationMessages.length === 0) return 100; // Default 100ms

    // Group messages by timestamp windows to estimate consensus rounds
    const timeWindows = new Map<number, CommunicationMetrics[]>();

    coordinationMessages.forEach(comm => {
      const windowKey = Math.floor(comm.timestamp.getTime() / 5000); // 5-second windows
      if (!timeWindows.has(windowKey)) {
        timeWindows.set(windowKey, []);
      }
      timeWindows.get(windowKey)!.push(comm);
    });

    // Calculate average consensus time
    let totalConsensusTime = 0;
    let consensusRounds = 0;

    timeWindows.forEach(messages => {
      if (messages.length >= 3) { // Minimum messages for consensus
        const startTime = Math.min(...messages.map(m => m.timestamp.getTime()));
        const endTime = Math.max(...messages.map(m => m.timestamp.getTime()));
        totalConsensusTime += endTime - startTime;
        consensusRounds++;
      }
    });

    return consensusRounds > 0 ? totalConsensusTime / consensusRounds : 100;
  }

  private calculateSynchronizationAccuracy(communications: CommunicationMetrics[]): number {
    const syncMessages = communications.filter(comm =>
      comm.messageType === 'data_sync' || comm.messageType === 'coordination'
    );

    if (syncMessages.length === 0) return 0.95; // Default 95%

    const successfulSyncs = syncMessages.filter(comm => comm.success).length;
    return successfulSyncs / syncMessages.length;
  }

  private calculateNetworkUtilization(communications: CommunicationMetrics[]): number {
    if (communications.length === 0) return 0;

    const totalDataTransferred = communications.reduce((sum, comm) => sum + comm.dataSize, 0);
    const timeWindow = 5 * 60 * 1000; // 5 minutes in milliseconds

    // Convert to Mbps (assuming data size is in bytes)
    const throughputBps = (totalDataTransferred * 8) / timeWindow;
    const throughputMbps = throughputBps / (1024 * 1024);

    // Normalize to 0-1 scale (assuming 1 Gbps network capacity)
    return Math.min(1, throughputMbps / 1000);
  }

  private estimateDiskIOPS(tasks: TaskMetrics[]): number {
    // This is a simplified estimation based on task complexity
    const ioIntensiveTasks = tasks.filter(task =>
      task.type.includes('database') ||
      task.type.includes('file') ||
      task.complexity === 'expert'
    );

    return ioIntensiveTasks.length * 100; // Estimate 100 IOPS per IO-intensive task
  }

  public getSwarmHealth(): SwarmHealthIndicator {
    const swarmMetrics = this.getSwarmMetrics();
    const now = new Date();
    const fiveMinutesAgo = new Date(now.getTime() - 5 * 60 * 1000);

    // Agent health
    const totalAgents = this.agents.size;
    const activeAgents = Array.from(this.agents.values())
      .filter(agent => agent.lastHeartbeat > fiveMinutesAgo);
    const healthyAgents = activeAgents.filter(agent =>
      agent.status !== 'failed' && agent.status !== 'recovering'
    ).length;
    const failedAgents = activeAgents.filter(agent => agent.status === 'failed').length;
    const recoveringAgents = activeAgents.filter(agent => agent.status === 'recovering').length;

    // Task health
    const recentTasks = Array.from(this.tasks.values())
      .filter(task => task.createdAt > fiveMinutesAgo);
    const tasksInQueue = recentTasks.filter(task => task.status === 'pending').length;
    const runningTasks = recentTasks.filter(task => task.status === 'running').length;
    const completedTasks = recentTasks.filter(task => task.status === 'completed').length;
    const failedTasks = recentTasks.filter(task => task.status === 'failed').length;

    // Calculate average wait time for pending tasks
    const pendingTasks = recentTasks.filter(task => task.status === 'pending');
    const averageWaitTime = pendingTasks.length > 0
      ? pendingTasks.reduce((sum, task) => sum + (now.getTime() - task.createdAt.getTime()), 0) / pendingTasks.length
      : 0;

    // Communication health
    const recentCommunications = this.communications
      .filter(comm => comm.timestamp > fiveMinutesAgo);
    const averageLatency = recentCommunications.length > 0
      ? recentCommunications.reduce((sum, comm) => sum + comm.latency, 0) / recentCommunications.length
      : 0;
    const messageSuccessRate = recentCommunications.length > 0
      ? recentCommunications.filter(comm => comm.success).length / recentCommunications.length
      : 1;
    const activeConnections = this.countActiveConnections(recentCommunications);
    const networkUtilization = this.calculateNetworkUtilization(recentCommunications);

    // Resource health
    const cpuUtilization = swarmMetrics.resourceUtilization.cpuUsage;
    const memoryUtilization = swarmMetrics.resourceUtilization.memoryUsage;
    const networkUtilizationResource = swarmMetrics.resourceUtilization.networkBandwidth;
    const resourceContention = cpuUtilization > 0.9 || memoryUtilization > 0.9 || networkUtilizationResource > 0.9;

    // Calculate overall health score (0-100)
    const agentHealthScore = totalAgents > 0 ? (healthyAgents / totalAgents) * 100 : 100;
    const taskHealthScore = recentTasks.length > 0 ? (completedTasks / recentTasks.length) * 100 : 100;
    const communicationHealthScore = (messageSuccessRate * 100) * (1 - Math.min(1, averageLatency / 1000));
    const resourceHealthScore = Math.max(0, 100 - (Math.max(0, cpuUtilization - 0.8) + Math.max(0, memoryUtilization - 0.8)) * 100);

    const overallHealth = (agentHealthScore * 0.3 + taskHealthScore * 0.3 +
                          communicationHealthScore * 0.2 + resourceHealthScore * 0.2);

    const healthIndicator: SwarmHealthIndicator = {
      overallHealth: Math.round(overallHealth),
      agentHealth: {
        healthyAgents,
        totalAgents,
        failedAgents,
        recoveringAgents
      },
      taskHealth: {
        tasksInQueue,
        runningTasks,
        completedTasks,
        failedTasks,
        averageWaitTime
      },
      communicationHealth: {
        averageLatency,
        messageSuccessRate,
        activeConnections,
        networkUtilization
      },
      resourceHealth: {
        cpuUtilization,
        memoryUtilization,
        networkUtilization: networkUtilizationResource,
        resourceContention
      }
    };

    // Store health history
    this.healthHistory.push(healthIndicator);
    if (this.healthHistory.length > 1000) {
      this.healthHistory.shift();
    }

    return healthIndicator;
  }

  private countActiveConnections(communications: CommunicationMetrics[]): number {
    const connections = new Set<string>();

    communications.forEach(comm => {
      connections.add(`${comm.sourceAgent}-${comm.targetAgent}`);
      connections.add(`${comm.targetAgent}-${comm.sourceAgent}`);
    });

    return connections.size;
  }

  private updateTopology(): void {
    const agents = Array.from(this.agents.values()).map(agent => ({
      id: agent.id,
      type: agent.type,
      status: agent.status,
      capabilities: agent.capabilities,
      connections: [], // Will be populated based on communications
      load: agent.status === 'busy' ? 1 : 0,
      performance: agent.performance.successRate
    }));

    // Analyze recent communications to determine connections
    const recentCommunications = this.communications
      .filter(comm => comm.timestamp > new Date(Date.now() - 10 * 60 * 1000)); // Last 10 minutes

    const connections = new Map<string, AgentConnection>();

    recentCommunications.forEach(comm => {
      const connectionKey = `${comm.sourceAgent}-${comm.targetAgent}`;

      if (!connections.has(connectionKey)) {
        connections.set(connectionKey, {
          source: comm.sourceAgent,
          target: comm.targetAgent,
          latency: comm.latency,
          bandwidth: comm.dataSize * 8 / comm.latency, // bps
          reliability: comm.success ? 1 : 0.5,
          messageCount: 1
        });
      } else {
        const conn = connections.get(connectionKey)!;
        conn.latency = (conn.latency + comm.latency) / 2;
        conn.bandwidth = (conn.bandwidth + comm.dataSize * 8 / comm.latency) / 2;
        conn.reliability = comm.success ? (conn.reliability + 1) / 2 : conn.reliability * 0.9;
        conn.messageCount++;
      }
    });

    // Update agent connections
    connections.forEach(conn => {
      const sourceAgent = agents.find(a => a.id === conn.source);
      const targetAgent = agents.find(a => a.id === conn.target);

      if (sourceAgent && !sourceAgent.connections.includes(conn.target)) {
        sourceAgent.connections.push(conn.target);
      }
      if (targetAgent && !targetAgent.connections.includes(conn.source)) {
        targetAgent.connections.push(conn.source);
      }
    });

    // Determine topology type (simplified)
    let topologyType: SwarmTopology['type'] = 'mesh';
    if (agents.length > 10) {
      topologyType = 'hierarchical';
    } else if (connections.size === agents.length - 1) {
      topologyType = 'star';
    }

    // Calculate efficiency and identify bottlenecks
    const efficiency = this.calculateTopologyEfficiency(agents, Array.from(connections.values()));
    const bottleneckNodes = this.identifyBottleneckNodes(agents, Array.from(connections.values()));
    const optimizationSuggestions = this.generateOptimizationSuggestions(agents, Array.from(connections.values()));

    this.topology = {
      type: topologyType,
      agents,
      connections: Array.from(connections.values()),
      efficiency,
      bottleneckNodes,
      optimizationSuggestions
    };

    this.emit('topology_updated', this.topology);
  }

  private calculateTopologyEfficiency(agents: AgentNode[], connections: AgentConnection[]): number {
    if (agents.length === 0) return 1;
    if (connections.length === 0) return 0;

    // Calculate average connection quality
    const avgLatency = connections.reduce((sum, conn) => sum + conn.latency, 0) / connections.length;
    const avgReliability = connections.reduce((sum, conn) => sum + conn.reliability, 0) / connections.length;
    const connectivityRatio = connections.length / (agents.length * (agents.length - 1) / 2);

    // Normalize metrics
    const latencyScore = Math.max(0, 1 - avgLatency / 1000); // 1000ms as baseline
    const reliabilityScore = avgReliability;
    const connectivityScore = Math.min(1, connectivityRatio * 2); // Prefer full connectivity but not essential

    return (latencyScore * 0.4 + reliabilityScore * 0.4 + connectivityScore * 0.2);
  }

  private identifyBottleneckNodes(agents: AgentNode[], connections: AgentConnection[]): string[] {
    const bottlenecks: string[] = [];

    agents.forEach(agent => {
      let isBottleneck = false;

      // High load
      if (agent.load > 0.9) {
        isBottleneck = true;
      }

      // Low performance
      if (agent.performance < 0.7) {
        isBottleneck = true;
      }

      // High degree (many connections) - potential communication bottleneck
      const degree = connections.filter(conn =>
        conn.source === agent.id || conn.target === agent.id
      ).length;

      if (degree > agents.length * 0.7) {
        isBottleneck = true;
      }

      if (isBottleneck) {
        bottlenecks.push(agent.id);
      }
    });

    return bottlenecks;
  }

  private generateOptimizationSuggestions(agents: AgentNode[], connections: AgentConnection[]): string[] {
    const suggestions: string[] = [];

    // Check for connectivity issues
    const avgDegree = agents.reduce((sum, agent) => sum + agent.connections.length, 0) / agents.length;
    if (avgDegree < 2) {
      suggestions.push('Increase agent connectivity for better coordination');
    }

    // Check for high-latency connections
    const highLatencyConnections = connections.filter(conn => conn.latency > 500);
    if (highLatencyConnections.length > 0) {
      suggestions.push('Optimize network configuration for high-latency connections');
    }

    // Check for load imbalance
    const busyAgents = agents.filter(agent => agent.load > 0.8).length;
    const totalAgents = agents.length;
    if (busyAgents / totalAgents > 0.7) {
      suggestions.push('Scale agent pool or improve task distribution');
    }

    // Check for performance issues
    const lowPerformanceAgents = agents.filter(agent => agent.performance < 0.8).length;
    if (lowPerformanceAgents / totalAgents > 0.3) {
      suggestions.push('Investigate agent performance issues and consider agent restart or reconfiguration');
    }

    return suggestions;
  }

  public getTopology(): SwarmTopology | null {
    return this.topology;
  }

  public getAgents(): AgentStatus[] {
    return Array.from(this.agents.values());
  }

  public getActiveAgents(): AgentStatus[] {
    const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000);
    return Array.from(this.agents.values())
      .filter(agent => agent.lastHeartbeat > fiveMinutesAgo);
  }

  public getTasks(filter?: {
    status?: TaskMetrics['status'];
    agentId?: string;
    timeRange?: { start: Date; end: Date };
  }): TaskMetrics[] {
    let tasks = Array.from(this.tasks.values());

    if (filter) {
      if (filter.status) {
        tasks = tasks.filter(task => task.status === filter.status);
      }
      if (filter.agentId) {
        tasks = tasks.filter(task => task.assignedAgent === filter.agentId);
      }
      if (filter.timeRange) {
        tasks = tasks.filter(task =>
          task.createdAt >= filter.timeRange!.start &&
          task.createdAt <= filter.timeRange!.end
        );
      }
    }

    return tasks;
  }

  public getCommunications(filter?: {
    sourceAgent?: string;
    targetAgent?: string;
    messageType?: CommunicationMetrics['messageType'];
    timeRange?: { start: Date; end: Date };
  }): CommunicationMetrics[] {
    let communications = [...this.communications];

    if (filter) {
      if (filter.sourceAgent) {
        communications = communications.filter(comm => comm.sourceAgent === filter.sourceAgent);
      }
      if (filter.targetAgent) {
        communications = communications.filter(comm => comm.targetAgent === filter.targetAgent);
      }
      if (filter.messageType) {
        communications = communications.filter(comm => comm.messageType === filter.messageType);
      }
      if (filter.timeRange) {
        communications = communications.filter(comm =>
          comm.timestamp >= filter.timeRange!.start &&
          comm.timestamp <= filter.timeRange!.end
        );
      }
    }

    return communications;
  }

  public getHealthHistory(limit?: number): SwarmHealthIndicator[] {
    return limit ? this.healthHistory.slice(-limit) : this.healthHistory;
  }

  private startMonitoring(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
    }

    this.monitoringInterval = setInterval(() => {
      try {
        const health = this.getSwarmHealth();
        this.emit('health_update', health);

        // Check for agent health issues
        this.checkAgentHealth();

        // Update topology periodically
        if (Date.now() % 30000 < 1000) { // Every 30 seconds
          this.updateTopology();
        }
      } catch (error) {
        console.error('Error in swarm monitoring:', error);
      }
    }, 5000); // Monitor every 5 seconds
  }

  private checkAgentHealth(): void {
    const now = new Date();
    const fiveMinutesAgo = new Date(now.getTime() - 5 * 60 * 1000);

    this.agents.forEach((agent, agentId) => {
      if (agent.lastHeartbeat < fiveMinutesAgo && agent.status !== 'failed') {
        // Agent appears to be unresponsive
        this.updateAgentStatus(agentId, {
          status: 'failed'
        });
        this.emit('agent_unresponsive', agent);
      }
    });
  }

  public stopMonitoring(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
  }

  public clearHistory(): void {
    this.communications = [];
    this.healthHistory = [];
    this.tasks.clear();
  }

  public exportMetrics(): any {
    return {
      timestamp: new Date(),
      agents: Array.from(this.agents.values()),
      tasks: Array.from(this.tasks.values()),
      communications: this.communications.slice(-1000), // Last 1000 communications
      topology: this.topology,
      health: this.getSwarmHealth(),
      healthHistory: this.healthHistory.slice(-100) // Last 100 health records
    };
  }
}