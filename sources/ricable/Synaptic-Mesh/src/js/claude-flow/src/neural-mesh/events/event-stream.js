/**
 * Neural Mesh Event Stream
 * Real-time event streaming for mesh coordination
 */

import { EventEmitter } from 'events';
import { nanoid } from 'nanoid';

export class EventChannel extends EventEmitter {
  constructor(id, participants = []) {
    super();
    this.id = id;
    this.participants = new Set(participants);
    this.messageQueue = [];
    this.maxQueueSize = 1000;
    this.createdAt = Date.now();
    this.lastActivity = Date.now();
  }

  addParticipant(participantId) {
    this.participants.add(participantId);
    this.emit('participantAdded', { participantId });
  }

  removeParticipant(participantId) {
    this.participants.delete(participantId);
    this.emit('participantRemoved', { participantId });
  }

  publishMessage(message) {
    const enrichedMessage = {
      ...message,
      id: nanoid(12),
      timestamp: Date.now(),
      channel: this.id
    };

    // Add to queue
    this.messageQueue.push(enrichedMessage);
    
    // Trim queue if too large
    if (this.messageQueue.length > this.maxQueueSize) {
      this.messageQueue.shift();
    }

    this.lastActivity = Date.now();
    this.emit('message', enrichedMessage);
    
    return enrichedMessage;
  }

  getRecentMessages(count = 10) {
    return this.messageQueue.slice(-count);
  }

  getMessageHistory(since) {
    return this.messageQueue.filter(msg => msg.timestamp >= since);
  }

  clear() {
    this.messageQueue = [];
    this.emit('cleared');
  }
}

export class EventStream extends EventEmitter {
  constructor() {
    super();
    this.channels = new Map();
    this.subscribers = new Map();
    this.eventFilters = new Map();
    this.metrics = {
      totalEvents: 0,
      eventsPerSecond: 0,
      channelsCreated: 0,
      subscribersActive: 0
    };
    this.metricsInterval = null;
  }

  async initialize() {
    console.log('🌊 Initializing Event Stream...');
    
    // Start metrics collection
    this.startMetricsCollection();
    
    // Setup error handling
    this.setupErrorHandling();
    
    this.emit('initialized');
    console.log('✅ Event Stream initialized');
  }

  /**
   * Create a new event channel
   */
  createChannel(channelId, participants = []) {
    if (this.channels.has(channelId)) {
      throw new Error(`Channel ${channelId} already exists`);
    }

    const channel = new EventChannel(channelId, participants);
    
    // Setup channel event forwarding
    channel.on('message', (message) => {
      this.metrics.totalEvents++;
      this.emit('channelMessage', { channelId, message });
      this.routeMessage(channelId, message);
    });

    channel.on('participantAdded', (data) => {
      this.emit('channelParticipantAdded', { channelId, ...data });
    });

    channel.on('participantRemoved', (data) => {
      this.emit('channelParticipantRemoved', { channelId, ...data });
    });

    this.channels.set(channelId, channel);
    this.metrics.channelsCreated++;
    
    this.emit('channelCreated', { channelId, channel });
    
    console.log(`📺 Channel ${channelId} created with ${participants.length} participants`);
    
    return channel;
  }

  /**
   * Get or create a channel
   */
  getChannel(channelId) {
    return this.channels.get(channelId);
  }

  /**
   * Delete a channel
   */
  deleteChannel(channelId) {
    const channel = this.channels.get(channelId);
    if (!channel) {
      return false;
    }

    channel.removeAllListeners();
    this.channels.delete(channelId);
    
    this.emit('channelDeleted', { channelId });
    
    console.log(`🗑️ Channel ${channelId} deleted`);
    
    return true;
  }

  /**
   * Subscribe to events
   */
  subscribe(subscriberId, options = {}) {
    const subscription = {
      id: subscriberId,
      channels: new Set(options.channels || []),
      eventTypes: new Set(options.eventTypes || []),
      filters: options.filters || {},
      callback: options.callback,
      createdAt: Date.now()
    };

    this.subscribers.set(subscriberId, subscription);
    this.metrics.subscribersActive = this.subscribers.size;
    
    this.emit('subscribed', { subscriberId, subscription });
    
    console.log(`👂 Subscriber ${subscriberId} subscribed to ${subscription.channels.size} channels`);
    
    return subscription;
  }

  /**
   * Unsubscribe from events
   */
  unsubscribe(subscriberId) {
    const subscription = this.subscribers.get(subscriberId);
    if (!subscription) {
      return false;
    }

    this.subscribers.delete(subscriberId);
    this.metrics.subscribersActive = this.subscribers.size;
    
    this.emit('unsubscribed', { subscriberId });
    
    console.log(`👋 Subscriber ${subscriberId} unsubscribed`);
    
    return true;
  }

  /**
   * Publish event to a channel
   */
  publish(channelId, event) {
    const channel = this.channels.get(channelId);
    if (!channel) {
      throw new Error(`Channel ${channelId} not found`);
    }

    return channel.publishMessage(event);
  }

  /**
   * Broadcast event to multiple channels
   */
  broadcast(channelIds, event) {
    const results = [];
    
    for (const channelId of channelIds) {
      try {
        const message = this.publish(channelId, event);
        results.push({ channelId, success: true, message });
      } catch (error) {
        results.push({ channelId, success: false, error: error.message });
      }
    }
    
    this.emit('broadcast', { channelIds, event, results });
    
    return results;
  }

  /**
   * Route message to subscribers
   */
  routeMessage(channelId, message) {
    for (const [subscriberId, subscription] of this.subscribers) {
      // Check if subscriber is interested in this channel
      if (subscription.channels.size === 0 || subscription.channels.has(channelId)) {
        // Check event type filter
        if (subscription.eventTypes.size === 0 || subscription.eventTypes.has(message.type)) {
          // Apply custom filters
          if (this.passesFilters(message, subscription.filters)) {
            this.deliverMessage(subscriberId, subscription, message);
          }
        }
      }
    }
  }

  /**
   * Check if message passes filters
   */
  passesFilters(message, filters) {
    for (const [key, value] of Object.entries(filters)) {
      if (message[key] !== value) {
        return false;
      }
    }
    return true;
  }

  /**
   * Deliver message to subscriber
   */
  deliverMessage(subscriberId, subscription, message) {
    try {
      if (subscription.callback) {
        subscription.callback(message, subscriberId);
      }
      
      this.emit('messageDelivered', { subscriberId, message });
      
    } catch (error) {
      console.error(`Failed to deliver message to ${subscriberId}:`, error);
      this.emit('deliveryError', { subscriberId, message, error });
    }
  }

  /**
   * Create event filter
   */
  createFilter(filterId, filterFunction) {
    this.eventFilters.set(filterId, filterFunction);
  }

  /**
   * Remove event filter
   */
  removeFilter(filterId) {
    return this.eventFilters.delete(filterId);
  }

  /**
   * Get stream statistics
   */
  getStats() {
    const channels = Array.from(this.channels.values());
    const subscribers = Array.from(this.subscribers.values());
    
    return {
      ...this.metrics,
      channels: {
        total: channels.length,
        active: channels.filter(c => c.lastActivity > Date.now() - 60000).length,
        totalParticipants: channels.reduce((sum, c) => sum + c.participants.size, 0)
      },
      subscribers: {
        total: subscribers.length,
        activeChannels: new Set(
          subscribers.flatMap(s => Array.from(s.channels))
        ).size
      },
      uptime: Date.now() - (this.startTime || Date.now())
    };
  }

  /**
   * Get channel information
   */
  getChannelInfo(channelId) {
    const channel = this.channels.get(channelId);
    if (!channel) {
      return null;
    }

    return {
      id: channel.id,
      participants: Array.from(channel.participants),
      messageCount: channel.messageQueue.length,
      lastActivity: channel.lastActivity,
      createdAt: channel.createdAt,
      recentMessages: channel.getRecentMessages(5)
    };
  }

  /**
   * Start metrics collection
   */
  startMetricsCollection() {
    this.startTime = Date.now();
    let lastEventCount = 0;
    
    this.metricsInterval = setInterval(() => {
      const currentEventCount = this.metrics.totalEvents;
      this.metrics.eventsPerSecond = currentEventCount - lastEventCount;
      lastEventCount = currentEventCount;
      
      this.emit('metricsUpdated', this.metrics);
    }, 1000);
  }

  /**
   * Setup error handling
   */
  setupErrorHandling() {
    this.on('error', (error) => {
      console.error('Event Stream Error:', error);
    });

    // Handle uncaught errors in channels
    process.on('uncaughtException', (error) => {
      console.error('Uncaught exception in Event Stream:', error);
      this.emit('error', error);
    });
  }

  /**
   * Cleanup resources
   */
  async cleanup() {
    console.log('🧹 Cleaning up Event Stream...');
    
    // Stop metrics collection
    if (this.metricsInterval) {
      clearInterval(this.metricsInterval);
    }
    
    // Clear all channels
    for (const [channelId] of this.channels) {
      this.deleteChannel(channelId);
    }
    
    // Clear all subscribers
    this.subscribers.clear();
    
    // Clear filters
    this.eventFilters.clear();
    
    this.emit('cleanup');
    
    console.log('✅ Event Stream cleanup completed');
  }

  /**
   * Create a neural coordination channel
   */
  createNeuralChannel(agentIds, options = {}) {
    const channelId = `neural-${nanoid(8)}`;
    const channel = this.createChannel(channelId, agentIds);
    
    // Add neural-specific event handling
    channel.on('message', (message) => {
      if (message.type === 'neural-sync') {
        this.handleNeuralSync(channelId, message);
      } else if (message.type === 'task-coordination') {
        this.handleTaskCoordination(channelId, message);
      }
    });
    
    return channel;
  }

  /**
   * Handle neural synchronization events
   */
  handleNeuralSync(channelId, message) {
    this.emit('neuralSync', { channelId, message });
    
    // Broadcast sync to all participants
    const channel = this.channels.get(channelId);
    if (channel) {
      for (const participantId of channel.participants) {
        this.emit('syncToAgent', { agentId: participantId, message });
      }
    }
  }

  /**
   * Handle task coordination events
   */
  handleTaskCoordination(channelId, message) {
    this.emit('taskCoordination', { channelId, message });
    
    // Apply coordination logic based on message content
    if (message.action === 'distribute') {
      this.distributeTask(channelId, message.task);
    } else if (message.action === 'collect') {
      this.collectResults(channelId, message.taskId);
    }
  }

  /**
   * Distribute task across channel participants
   */
  distributeTask(channelId, task) {
    const channel = this.channels.get(channelId);
    if (!channel) return;
    
    const participants = Array.from(channel.participants);
    const taskParts = this.splitTask(task, participants.length);
    
    participants.forEach((agentId, index) => {
      const taskPart = taskParts[index];
      const message = {
        type: 'task-assignment',
        taskId: task.id,
        taskPart,
        agentId,
        totalParts: taskParts.length,
        partIndex: index
      };
      
      channel.publishMessage(message);
    });
  }

  /**
   * Split task into parts for parallel execution
   */
  splitTask(task, partCount) {
    // Simple task splitting - can be enhanced based on task type
    const parts = [];
    for (let i = 0; i < partCount; i++) {
      parts.push({
        ...task,
        id: `${task.id}-part-${i}`,
        partition: i,
        totalPartitions: partCount
      });
    }
    return parts;
  }

  /**
   * Collect results from distributed task execution
   */
  collectResults(channelId, taskId) {
    const channel = this.channels.get(channelId);
    if (!channel) return;
    
    const results = channel.messageQueue.filter(
      msg => msg.type === 'task-result' && msg.taskId === taskId
    );
    
    if (results.length === channel.participants.size) {
      // All results collected
      const aggregatedResult = this.aggregateResults(results);
      
      channel.publishMessage({
        type: 'task-completed',
        taskId,
        result: aggregatedResult,
        timestamp: Date.now()
      });
    }
  }

  /**
   * Aggregate results from multiple agents
   */
  aggregateResults(results) {
    return {
      combined: true,
      results: results.map(r => r.result),
      totalParts: results.length,
      aggregatedAt: Date.now()
    };
  }
}

export default EventStream;