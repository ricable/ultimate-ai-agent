/**
 * Real-Time Federation System using Supabase
 *
 * Leverages Supabase Real-Time for:
 * - Live agent coordination
 * - Instant memory synchronization
 * - Real-time presence tracking
 * - Collaborative multi-agent workflows
 * - Event-driven agent communication
 */

import { createClient, SupabaseClient, RealtimeChannel } from '@supabase/supabase-js';

export interface RealtimeConfig {
  url: string;
  anonKey: string;
  serviceRoleKey?: string;
  presenceHeartbeat?: number; // ms, default 30000
  memorySync?: boolean; // Auto-sync memories
  broadcastLatency?: 'low' | 'high'; // Trade-off: low = less batching
}

export interface AgentPresence {
  agent_id: string;
  tenant_id: string;
  status: 'online' | 'busy' | 'offline';
  task?: string;
  started_at: string;
  last_heartbeat: string;
  metadata?: Record<string, any>;
}

export interface MemoryEvent {
  type: 'memory_added' | 'memory_updated' | 'memory_retrieved';
  tenant_id: string;
  agent_id: string;
  session_id: string;
  content: string;
  embedding?: number[];
  metadata?: Record<string, any>;
  timestamp: string;
}

export interface CoordinationMessage {
  from_agent: string;
  to_agent?: string; // undefined = broadcast
  type: 'task_assignment' | 'task_complete' | 'request_help' | 'share_knowledge' | 'status_update';
  payload: any;
  timestamp: string;
}

export interface TaskAssignment {
  task_id: string;
  assigned_to: string;
  description: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  deadline?: string;
  dependencies?: string[];
}

export class RealtimeFederationHub {
  private client: SupabaseClient;
  private config: RealtimeConfig;
  private channels: Map<string, RealtimeChannel> = new Map();
  private presenceChannel?: RealtimeChannel;
  private agentId: string;
  private tenantId: string;
  private heartbeatInterval?: NodeJS.Timeout;
  private eventHandlers: Map<string, Set<Function>> = new Map();

  constructor(config: RealtimeConfig, agentId: string, tenantId: string = 'default') {
    this.config = config;
    this.agentId = agentId;
    this.tenantId = tenantId;

    const key = config.serviceRoleKey || config.anonKey;
    this.client = createClient(config.url, key, {
      realtime: {
        params: {
          eventsPerSecond: config.broadcastLatency === 'low' ? 10 : 2,
        },
      },
    });
  }

  /**
   * Initialize real-time federation
   */
  async initialize(): Promise<void> {
    console.log('🌐 Initializing Real-Time Federation Hub...');
    console.log(`   Agent: ${this.agentId}`);
    console.log(`   Tenant: ${this.tenantId}`);

    // Set up presence tracking
    await this.setupPresence();

    // Set up memory sync channel
    if (this.config.memorySync !== false) {
      await this.setupMemorySync();
    }

    // Set up coordination channel
    await this.setupCoordination();

    // Start heartbeat
    this.startHeartbeat();

    console.log('✅ Real-Time Federation Hub Active');
  }

  /**
   * Set up presence tracking for all agents in tenant
   */
  private async setupPresence(): Promise<void> {
    const channelName = `presence:${this.tenantId}`;
    this.presenceChannel = this.client.channel(channelName, {
      config: {
        presence: {
          key: this.agentId,
        },
      },
    });

    // Track agent join
    this.presenceChannel.on('presence', { event: 'join' }, ({ key, newPresences }) => {
      console.log(`🟢 Agent joined: ${key}`);
      this.emit('agent:join', { agent_id: key, presences: newPresences });
    });

    // Track agent leave
    this.presenceChannel.on('presence', { event: 'leave' }, ({ key, leftPresences }) => {
      console.log(`🔴 Agent left: ${key}`);
      this.emit('agent:leave', { agent_id: key, presences: leftPresences });
    });

    // Track agent sync (periodic updates)
    this.presenceChannel.on('presence', { event: 'sync' }, () => {
      const state = this.presenceChannel!.presenceState();
      this.emit('agents:sync', { agents: this.getActiveAgents(state) });
    });

    // Subscribe and track presence
    await this.presenceChannel.subscribe(async (status) => {
      if (status === 'SUBSCRIBED') {
        await this.presenceChannel!.track({
          agent_id: this.agentId,
          tenant_id: this.tenantId,
          status: 'online',
          started_at: new Date().toISOString(),
          last_heartbeat: new Date().toISOString(),
        });
      }
    });
  }

  /**
   * Set up real-time memory synchronization
   */
  private async setupMemorySync(): Promise<void> {
    const channelName = `memories:${this.tenantId}`;
    const channel = this.client.channel(channelName);

    // Listen for new memories from database
    channel
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'agent_memories',
          filter: `tenant_id=eq.${this.tenantId}`,
        },
        (payload) => {
          const memory: MemoryEvent = {
            type: 'memory_added',
            tenant_id: payload.new.tenant_id,
            agent_id: payload.new.agent_id,
            session_id: payload.new.session_id,
            content: payload.new.content,
            embedding: payload.new.embedding,
            metadata: payload.new.metadata,
            timestamp: payload.new.created_at,
          };
          this.emit('memory:added', memory);
        }
      )
      .on(
        'postgres_changes',
        {
          event: 'UPDATE',
          schema: 'public',
          table: 'agent_memories',
          filter: `tenant_id=eq.${this.tenantId}`,
        },
        (payload) => {
          const memory: MemoryEvent = {
            type: 'memory_updated',
            tenant_id: payload.new.tenant_id,
            agent_id: payload.new.agent_id,
            session_id: payload.new.session_id,
            content: payload.new.content,
            embedding: payload.new.embedding,
            metadata: payload.new.metadata,
            timestamp: new Date().toISOString(),
          };
          this.emit('memory:updated', memory);
        }
      );

    await channel.subscribe();
    this.channels.set('memory-sync', channel);

    console.log('💾 Real-time memory sync enabled');
  }

  /**
   * Set up coordination channel for agent-to-agent communication
   */
  private async setupCoordination(): Promise<void> {
    const channelName = `coordination:${this.tenantId}`;
    const channel = this.client.channel(channelName);

    // Listen for broadcast messages
    channel.on('broadcast', { event: 'coordination' }, ({ payload }) => {
      const message: CoordinationMessage = payload;

      // Only process if message is for us or broadcast
      if (!message.to_agent || message.to_agent === this.agentId) {
        this.emit('message:received', message);

        // Emit specific event types
        this.emit(`message:${message.type}`, message);
      }
    });

    await channel.subscribe();
    this.channels.set('coordination', channel);

    console.log('📡 Agent coordination channel active');
  }

  /**
   * Broadcast message to all agents in tenant
   */
  async broadcast(
    type: CoordinationMessage['type'],
    payload: any
  ): Promise<void> {
    const channel = this.channels.get('coordination');
    if (!channel) {
      throw new Error('Coordination channel not initialized');
    }

    const message: CoordinationMessage = {
      from_agent: this.agentId,
      type,
      payload,
      timestamp: new Date().toISOString(),
    };

    await channel.send({
      type: 'broadcast',
      event: 'coordination',
      payload: message,
    });
  }

  /**
   * Send direct message to specific agent
   */
  async sendMessage(
    toAgent: string,
    type: CoordinationMessage['type'],
    payload: any
  ): Promise<void> {
    const channel = this.channels.get('coordination');
    if (!channel) {
      throw new Error('Coordination channel not initialized');
    }

    const message: CoordinationMessage = {
      from_agent: this.agentId,
      to_agent: toAgent,
      type,
      payload,
      timestamp: new Date().toISOString(),
    };

    await channel.send({
      type: 'broadcast',
      event: 'coordination',
      payload: message,
    });
  }

  /**
   * Assign task to another agent
   */
  async assignTask(task: TaskAssignment): Promise<void> {
    await this.sendMessage(task.assigned_to, 'task_assignment', task);
    console.log(`📋 Task assigned: ${task.task_id} → ${task.assigned_to}`);
  }

  /**
   * Report task completion
   */
  async reportTaskComplete(taskId: string, result: any): Promise<void> {
    await this.broadcast('task_complete', {
      task_id: taskId,
      result,
      completed_by: this.agentId,
    });
    console.log(`✅ Task completed: ${taskId}`);
  }

  /**
   * Request help from other agents
   */
  async requestHelp(problem: string, context?: any): Promise<void> {
    await this.broadcast('request_help', {
      problem,
      context,
      from: this.agentId,
    });
    console.log(`🆘 Help requested: ${problem}`);
  }

  /**
   * Share knowledge with other agents
   */
  async shareKnowledge(knowledge: string, metadata?: any): Promise<void> {
    await this.broadcast('share_knowledge', {
      knowledge,
      metadata,
      from: this.agentId,
    });
    console.log(`💡 Knowledge shared: ${knowledge.substring(0, 50)}...`);
  }

  /**
   * Update agent status
   */
  async updateStatus(
    status: 'online' | 'busy' | 'offline',
    task?: string
  ): Promise<void> {
    if (!this.presenceChannel) return;

    await this.presenceChannel.track({
      agent_id: this.agentId,
      tenant_id: this.tenantId,
      status,
      task,
      last_heartbeat: new Date().toISOString(),
    });

    await this.broadcast('status_update', {
      agent_id: this.agentId,
      status,
      task,
    });
  }

  /**
   * Get list of active agents
   */
  getActiveAgents(presenceState?: any): AgentPresence[] {
    if (!this.presenceChannel) return [];

    const state = presenceState || this.presenceChannel.presenceState();
    const agents: AgentPresence[] = [];

    for (const [agentId, presences] of Object.entries(state)) {
      const presence = (presences as any[])[0];
      agents.push({
        agent_id: agentId,
        tenant_id: presence.tenant_id,
        status: presence.status,
        task: presence.task,
        started_at: presence.started_at,
        last_heartbeat: presence.last_heartbeat,
        metadata: presence.metadata,
      });
    }

    return agents;
  }

  /**
   * Start heartbeat to maintain presence
   */
  private startHeartbeat(): void {
    const interval = this.config.presenceHeartbeat || 30000;

    this.heartbeatInterval = setInterval(async () => {
      if (this.presenceChannel) {
        await this.presenceChannel.track({
          agent_id: this.agentId,
          tenant_id: this.tenantId,
          last_heartbeat: new Date().toISOString(),
        });
      }
    }, interval);
  }

  /**
   * Stop heartbeat
   */
  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = undefined;
    }
  }

  /**
   * Subscribe to events
   */
  on(event: string, handler: Function): void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, new Set());
    }
    this.eventHandlers.get(event)!.add(handler);
  }

  /**
   * Unsubscribe from events
   */
  off(event: string, handler: Function): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.delete(handler);
    }
  }

  /**
   * Emit event to handlers
   */
  private emit(event: string, data: any): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.forEach((handler) => {
        try {
          handler(data);
        } catch (error) {
          console.error(`Error in event handler for ${event}:`, error);
        }
      });
    }
  }

  /**
   * Get real-time statistics
   */
  async getStats(): Promise<any> {
    const activeAgents = this.getActiveAgents();

    return {
      tenant_id: this.tenantId,
      agent_id: this.agentId,
      active_agents: activeAgents.length,
      agents: activeAgents,
      channels: Array.from(this.channels.keys()),
      heartbeat_interval: this.config.presenceHeartbeat || 30000,
      memory_sync: this.config.memorySync !== false,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Shutdown and cleanup
   */
  async shutdown(): Promise<void> {
    console.log('🛑 Shutting down Real-Time Federation Hub...');

    // Stop heartbeat
    this.stopHeartbeat();

    // Update status to offline
    if (this.presenceChannel) {
      await this.presenceChannel.track({
        agent_id: this.agentId,
        tenant_id: this.tenantId,
        status: 'offline',
      });
      await this.presenceChannel.untrack();
    }

    // Unsubscribe from all channels
    for (const [name, channel] of this.channels) {
      await channel.unsubscribe();
      console.log(`   Unsubscribed from ${name}`);
    }

    if (this.presenceChannel) {
      await this.presenceChannel.unsubscribe();
    }

    this.channels.clear();
    this.eventHandlers.clear();

    console.log('✅ Real-Time Federation Hub shutdown complete');
  }
}

/**
 * Create real-time federation hub from environment
 */
export function createRealtimeHub(
  agentId: string,
  tenantId: string = 'default'
): RealtimeFederationHub {
  const url = process.env.SUPABASE_URL;
  const anonKey = process.env.SUPABASE_ANON_KEY;
  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!url || !anonKey) {
    throw new Error(
      'Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_ANON_KEY'
    );
  }

  return new RealtimeFederationHub(
    {
      url,
      anonKey,
      serviceRoleKey,
      presenceHeartbeat: parseInt(
        process.env.FEDERATION_HEARTBEAT_INTERVAL || '30000'
      ),
      memorySync: process.env.FEDERATION_MEMORY_SYNC !== 'false',
      broadcastLatency:
        (process.env.FEDERATION_BROADCAST_LATENCY as 'low' | 'high') || 'low',
    },
    agentId,
    tenantId
  );
}
