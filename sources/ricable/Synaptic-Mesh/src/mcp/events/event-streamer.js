/**
 * Event Streaming System for Real-time MCP Communication
 * Handles real-time events, notifications, and streaming data
 */

import { EventEmitter } from 'events';
import { nanoid } from 'nanoid';

export class EventStreamer extends EventEmitter {
  constructor(config) {
    super();
    this.config = config;
    this.streams = new Map();
    this.subscribers = new Map();
    this.eventBuffer = new Map();
    this.metrics = {
      totalEvents: 0,
      totalStreams: 0,
      totalSubscribers: 0,
      eventsPerSecond: 0
    };
    
    this.bufferSize = config.bufferSize || 1000;
    this.compressionEnabled = config.compression || false;
    this.retentionTime = config.retentionTime || 3600000; // 1 hour
    
    this.setupEventHandlers();
    this.startMetricsCollection();
  }

  setupEventHandlers() {
    // Handle neural mesh events
    this.on('neuralMeshEvent', (event) => {
      this.processEvent('neural-mesh', event);
    });

    // Handle agent events
    this.on('agentEvent', (event) => {
      this.processEvent('agent', event);
    });

    // Handle consensus events
    this.on('consensusEvent', (event) => {
      this.processEvent('consensus', event);
    });

    // Handle performance events
    this.on('performanceEvent', (event) => {
      this.processEvent('performance', event);
    });

    // Handle error events
    this.on('errorEvent', (event) => {
      this.processEvent('error', event);
    });
  }

  createStream(type, filter = {}, options = {}) {
    const streamId = nanoid();
    const stream = {
      id: streamId,
      type,
      filter,
      options: {
        realtime: true,
        includeHistory: false,
        maxEvents: 100,
        ...options
      },
      subscribers: new Set(),
      buffer: [],
      createdAt: Date.now(),
      lastActivity: Date.now(),
      stats: {
        eventsStreamed: 0,
        subscriberCount: 0
      }
    };

    this.streams.set(streamId, stream);
    this.metrics.totalStreams++;

    // Include historical events if requested
    if (stream.options.includeHistory) {
      const historicalEvents = this.getHistoricalEvents(type, filter);
      stream.buffer.push(...historicalEvents.slice(-stream.options.maxEvents));
    }

    console.log(`📡 Event stream created: ${streamId} (type: ${type})`);
    return streamId;
  }

  subscribe(streamId, callback, clientId = null) {
    const stream = this.streams.get(streamId);
    if (!stream) {
      throw new Error(`Stream '${streamId}' not found`);
    }

    const subscriberId = clientId || nanoid();
    const subscriber = {
      id: subscriberId,
      streamId,
      callback,
      subscribedAt: Date.now(),
      lastEventAt: null,
      eventsReceived: 0
    };

    stream.subscribers.add(subscriberId);
    this.subscribers.set(subscriberId, subscriber);
    stream.stats.subscriberCount = stream.subscribers.size;
    this.metrics.totalSubscribers++;

    // Send buffered events to new subscriber
    if (stream.buffer.length > 0) {
      stream.buffer.forEach(event => {
        this.deliverEvent(subscriber, event);
      });
    }

    console.log(`📡 Client subscribed to stream ${streamId}: ${subscriberId}`);
    return subscriberId;
  }

  unsubscribe(subscriberId) {
    const subscriber = this.subscribers.get(subscriberId);
    if (!subscriber) {
      return false;
    }

    const stream = this.streams.get(subscriber.streamId);
    if (stream) {
      stream.subscribers.delete(subscriberId);
      stream.stats.subscriberCount = stream.subscribers.size;
    }

    this.subscribers.delete(subscriberId);
    this.metrics.totalSubscribers--;

    console.log(`📡 Client unsubscribed: ${subscriberId}`);
    return true;
  }

  processEvent(type, eventData) {
    const event = {
      id: nanoid(),
      type,
      data: eventData,
      timestamp: Date.now(),
      source: 'neural-mesh'
    };

    // Add to global event buffer
    this.addToEventBuffer(type, event);

    // Find matching streams
    const matchingStreams = this.findMatchingStreams(type, event);

    // Deliver to subscribers
    matchingStreams.forEach(stream => {
      this.deliverToStream(stream, event);
    });

    this.metrics.totalEvents++;
  }

  findMatchingStreams(type, event) {
    const matching = [];

    for (const stream of this.streams.values()) {
      if (this.eventMatchesStream(stream, type, event)) {
        matching.push(stream);
      }
    }

    return matching;
  }

  eventMatchesStream(stream, type, event) {
    // Check type match
    if (stream.type !== '*' && stream.type !== type) {
      return false;
    }

    // Check filter criteria
    const filter = stream.filter;
    if (Object.keys(filter).length === 0) {
      return true; // No filter means accept all
    }

    // Apply filters
    for (const [key, value] of Object.entries(filter)) {
      if (this.applyFilter(event, key, value)) {
        continue;
      } else {
        return false;
      }
    }

    return true;
  }

  applyFilter(event, key, value) {
    // Support nested property access (e.g., "data.meshId")
    const getValue = (obj, path) => {
      return path.split('.').reduce((current, prop) => current?.[prop], obj);
    };

    const eventValue = getValue(event, key);

    if (typeof value === 'string' && value.includes('*')) {
      // Wildcard matching
      const regex = new RegExp(value.replace(/\*/g, '.*'));
      return regex.test(String(eventValue));
    } else if (Array.isArray(value)) {
      // Array contains check
      return value.includes(eventValue);
    } else {
      // Exact match
      return eventValue === value;
    }
  }

  deliverToStream(stream, event) {
    // Add to stream buffer
    stream.buffer.push(event);
    if (stream.buffer.length > stream.options.maxEvents) {
      stream.buffer.shift(); // Remove oldest event
    }

    stream.lastActivity = Date.now();
    stream.stats.eventsStreamed++;

    // Deliver to all subscribers
    for (const subscriberId of stream.subscribers) {
      const subscriber = this.subscribers.get(subscriberId);
      if (subscriber) {
        this.deliverEvent(subscriber, event);
      }
    }
  }

  deliverEvent(subscriber, event) {
    try {
      subscriber.callback(event);
      subscriber.lastEventAt = Date.now();
      subscriber.eventsReceived++;
    } catch (error) {
      console.error(`Error delivering event to subscriber ${subscriber.id}:`, error);
      // Optionally remove problematic subscriber
      this.unsubscribe(subscriber.id);
    }
  }

  addToEventBuffer(type, event) {
    if (!this.eventBuffer.has(type)) {
      this.eventBuffer.set(type, []);
    }

    const buffer = this.eventBuffer.get(type);
    buffer.push(event);

    // Maintain buffer size
    if (buffer.length > this.bufferSize) {
      buffer.shift();
    }

    // Clean old events
    this.cleanOldEvents(buffer);
  }

  cleanOldEvents(buffer) {
    const cutoff = Date.now() - this.retentionTime;
    const index = buffer.findIndex(event => event.timestamp > cutoff);
    if (index > 0) {
      buffer.splice(0, index);
    }
  }

  getHistoricalEvents(type, filter = {}) {
    const buffer = this.eventBuffer.get(type) || [];
    
    if (Object.keys(filter).length === 0) {
      return buffer;
    }

    return buffer.filter(event => {
      for (const [key, value] of Object.entries(filter)) {
        if (!this.applyFilter(event, key, value)) {
          return false;
        }
      }
      return true;
    });
  }

  // Real-time streaming methods for specific event types

  streamNeuralMeshEvents(filter = {}) {
    return this.createStream('neural-mesh', filter, {
      realtime: true,
      includeHistory: true,
      maxEvents: 50
    });
  }

  streamAgentEvents(meshId = null, agentId = null) {
    const filter = {};
    if (meshId) filter['data.meshId'] = meshId;
    if (agentId) filter['data.agentId'] = agentId;

    return this.createStream('agent', filter, {
      realtime: true,
      includeHistory: false,
      maxEvents: 100
    });
  }

  streamConsensusEvents(meshId) {
    return this.createStream('consensus', {
      'data.meshId': meshId
    }, {
      realtime: true,
      includeHistory: true,
      maxEvents: 20
    });
  }

  streamPerformanceMetrics(interval = 5000) {
    const streamId = this.createStream('performance', {}, {
      realtime: true,
      includeHistory: false,
      maxEvents: 200
    });

    // Set up periodic performance event generation
    const performanceInterval = setInterval(() => {
      this.emit('performanceEvent', {
        timestamp: Date.now(),
        metrics: this.getStreamingMetrics(),
        interval
      });
    }, interval);

    // Clean up interval when stream is destroyed
    const stream = this.streams.get(streamId);
    stream.cleanupInterval = performanceInterval;

    return streamId;
  }

  startMetricsCollection() {
    setInterval(() => {
      // Calculate events per second
      const currentTime = Date.now();
      if (!this.lastMetricsTime) {
        this.lastMetricsTime = currentTime;
        this.lastEventCount = this.metrics.totalEvents;
        return;
      }

      const timeDiff = (currentTime - this.lastMetricsTime) / 1000;
      const eventDiff = this.metrics.totalEvents - this.lastEventCount;
      
      this.metrics.eventsPerSecond = eventDiff / timeDiff;
      
      this.lastMetricsTime = currentTime;
      this.lastEventCount = this.metrics.totalEvents;

    }, 5000); // Update every 5 seconds
  }

  destroyStream(streamId) {
    const stream = this.streams.get(streamId);
    if (!stream) {
      return false;
    }

    // Unsubscribe all subscribers
    for (const subscriberId of stream.subscribers) {
      this.unsubscribe(subscriberId);
    }

    // Clean up any intervals
    if (stream.cleanupInterval) {
      clearInterval(stream.cleanupInterval);
    }

    this.streams.delete(streamId);
    this.metrics.totalStreams--;

    console.log(`📡 Event stream destroyed: ${streamId}`);
    return true;
  }

  getStreamingMetrics() {
    return {
      ...this.metrics,
      activeStreams: this.streams.size,
      activeSubscribers: this.subscribers.size,
      bufferSizes: Array.from(this.eventBuffer.entries()).map(([type, buffer]) => ({
        type,
        size: buffer.length
      }))
    };
  }

  getStreamInfo(streamId) {
    const stream = this.streams.get(streamId);
    if (!stream) {
      return null;
    }

    return {
      id: stream.id,
      type: stream.type,
      filter: stream.filter,
      options: stream.options,
      subscriberCount: stream.subscribers.size,
      stats: stream.stats,
      createdAt: stream.createdAt,
      lastActivity: stream.lastActivity,
      bufferSize: stream.buffer.length
    };
  }

  listActiveStreams() {
    return Array.from(this.streams.values()).map(stream => ({
      id: stream.id,
      type: stream.type,
      subscriberCount: stream.subscribers.size,
      eventsStreamed: stream.stats.eventsStreamed,
      createdAt: stream.createdAt,
      lastActivity: stream.lastActivity
    }));
  }

  // Event emission helpers for different components

  emitNeuralMeshEvent(eventType, data) {
    this.emit('neuralMeshEvent', {
      eventType,
      ...data
    });
  }

  emitAgentEvent(eventType, agentId, meshId, data) {
    this.emit('agentEvent', {
      eventType,
      agentId,
      meshId,
      ...data
    });
  }

  emitConsensusEvent(eventType, meshId, consensusData) {
    this.emit('consensusEvent', {
      eventType,
      meshId,
      consensus: consensusData
    });
  }

  emitPerformanceEvent(metrics) {
    this.emit('performanceEvent', metrics);
  }

  emitErrorEvent(error, context = {}) {
    this.emit('errorEvent', {
      error: error.message,
      stack: error.stack,
      context
    });
  }
}

export default EventStreamer;