/**
 * Communication Protocol Tests
 * Tests inter-agent messaging, broadcasts, and message handling
 */

import { describe, it, expect, beforeEach, jest } from '@jest/globals';
import EventEmitter from 'events';

// Message types
const MESSAGE_TYPES = {
  INFO: 'info',
  REQUEST: 'request',
  RESPONSE: 'response',
  BROADCAST: 'broadcast',
  CONSENSUS: 'consensus',
  TASK: 'task',
  STATUS: 'status',
  ERROR: 'error'
};

// Message priorities
const PRIORITIES = {
  LOW: 0,
  MEDIUM: 1,
  HIGH: 2,
  CRITICAL: 3
};

// Communication Channel
class CommunicationChannel extends EventEmitter {
  constructor() {
    super();
    this.messages = [];
    this.messageQueue = new Map(); // agent -> message queue
    this.subscriptions = new Map(); // topic -> Set of agents
    this.messageHandlers = new Map(); // agent -> handler function
    this.metrics = {
      sent: 0,
      received: 0,
      broadcast: 0,
      failed: 0,
      queuedMessages: 0
    };
  }

  registerAgent(agentId, handler) {
    this.messageQueue.set(agentId, []);
    this.messageHandlers.set(agentId, handler);
    this.emit('agent:registered', agentId);
  }

  unregisterAgent(agentId) {
    this.messageQueue.delete(agentId);
    this.messageHandlers.delete(agentId);
    
    // Remove from all subscriptions
    for (const [topic, agents] of this.subscriptions) {
      agents.delete(agentId);
    }
    
    this.emit('agent:unregistered', agentId);
  }

  sendMessage(from, to, content, type = MESSAGE_TYPES.INFO, priority = PRIORITIES.MEDIUM) {
    const message = {
      id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      from,
      to,
      type,
      content,
      priority,
      timestamp: Date.now(),
      status: 'pending'
    };

    // Validate sender and recipient
    if (!this.messageHandlers.has(from)) {
      throw new Error(`Sender agent not registered: ${from}`);
    }

    if (to && !this.messageHandlers.has(to)) {
      throw new Error(`Recipient agent not registered: ${to}`);
    }

    // Store message
    this.messages.push(message);
    this.metrics.sent++;

    // Route message
    if (to) {
      this._deliverMessage(message);
    } else {
      this._broadcastMessage(message);
    }

    return message;
  }

  _deliverMessage(message) {
    const queue = this.messageQueue.get(message.to);
    if (!queue) {
      message.status = 'failed';
      this.metrics.failed++;
      return;
    }

    // Add to queue based on priority
    queue.push(message);
    queue.sort((a, b) => b.priority - a.priority);
    this.metrics.queuedMessages++;

    // Process immediately in tests
    this._processMessageQueue(message.to);
  }

  _broadcastMessage(message) {
    message.type = MESSAGE_TYPES.BROADCAST;
    this.metrics.broadcast++;

    for (const [agentId] of this.messageHandlers) {
      if (agentId !== message.from) {
        const broadcastMsg = { ...message, to: agentId };
        this._deliverMessage(broadcastMsg);
      }
    }
  }

  _processMessageQueue(agentId) {
    const queue = this.messageQueue.get(agentId);
    const handler = this.messageHandlers.get(agentId);

    if (!queue || !handler || queue.length === 0) {
      return;
    }

    // Process messages in priority order
    while (queue.length > 0) {
      const message = queue.shift();
      this.metrics.queuedMessages--;

      try {
        message.status = 'delivered';
        this.metrics.received++;
        handler(message);
        this.emit('message:delivered', message);
      } catch (error) {
        message.status = 'failed';
        message.error = error.message;
        this.metrics.failed++;
        this.emit('message:failed', message);
      }
    }
  }

  subscribe(agentId, topic) {
    if (!this.subscriptions.has(topic)) {
      this.subscriptions.set(topic, new Set());
    }
    this.subscriptions.get(topic).add(agentId);
    this.emit('subscription:added', { agentId, topic });
  }

  unsubscribe(agentId, topic) {
    const subscribers = this.subscriptions.get(topic);
    if (subscribers) {
      subscribers.delete(agentId);
      if (subscribers.size === 0) {
        this.subscriptions.delete(topic);
      }
    }
    this.emit('subscription:removed', { agentId, topic });
  }

  publishToTopic(topic, content, from) {
    const subscribers = this.subscriptions.get(topic);
    if (!subscribers || subscribers.size === 0) {
      return 0;
    }

    let sent = 0;
    for (const agentId of subscribers) {
      if (agentId !== from) {
        this.sendMessage(from, agentId, content, MESSAGE_TYPES.INFO);
        sent++;
      }
    }

    return sent;
  }

  getMessageHistory(filter = {}) {
    let messages = [...this.messages];

    if (filter.from) {
      messages = messages.filter(m => m.from === filter.from);
    }

    if (filter.to) {
      messages = messages.filter(m => m.to === filter.to);
    }

    if (filter.type) {
      messages = messages.filter(m => m.type === filter.type);
    }

    if (filter.since) {
      messages = messages.filter(m => m.timestamp >= filter.since);
    }

    return messages;
  }

  getConversation(agent1, agent2) {
    return this.messages.filter(m => 
      (m.from === agent1 && m.to === agent2) ||
      (m.from === agent2 && m.to === agent1)
    ).sort((a, b) => a.timestamp - b.timestamp);
  }

  getMetrics() {
    return {
      ...this.metrics,
      totalMessages: this.messages.length,
      registeredAgents: this.messageHandlers.size,
      activeTopics: this.subscriptions.size
    };
  }

  clearHistory() {
    this.messages = [];
    this.metrics.sent = 0;
    this.metrics.received = 0;
    this.metrics.broadcast = 0;
    this.metrics.failed = 0;
  }
}

// Protocol handler for structured communication
class ProtocolHandler {
  constructor(channel) {
    this.channel = channel;
    this.pendingRequests = new Map(); // requestId -> { resolve, reject, timeout }
  }

  async request(from, to, action, params, timeout = 5000) {
    const requestId = `req-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    const requestContent = {
      requestId,
      action,
      params
    };

    return new Promise((resolve, reject) => {
      // Set up timeout
      const timeoutId = setTimeout(() => {
        this.pendingRequests.delete(requestId);
        reject(new Error(`Request timeout: ${action}`));
      }, timeout);

      // Store pending request
      this.pendingRequests.set(requestId, {
        resolve,
        reject,
        timeoutId
      });

      // Send request
      this.channel.sendMessage(from, to, requestContent, MESSAGE_TYPES.REQUEST, PRIORITIES.HIGH);
    });
  }

  response(from, to, requestId, result, error = null) {
    const responseContent = {
      requestId,
      result,
      error
    };

    this.channel.sendMessage(from, to, responseContent, MESSAGE_TYPES.RESPONSE, PRIORITIES.HIGH);
  }

  handleResponse(message) {
    if (message.type !== MESSAGE_TYPES.RESPONSE) {
      return false;
    }

    const { requestId, result, error } = message.content;
    const pending = this.pendingRequests.get(requestId);

    if (!pending) {
      return false;
    }

    clearTimeout(pending.timeoutId);
    this.pendingRequests.delete(requestId);

    if (error) {
      pending.reject(new Error(error));
    } else {
      pending.resolve(result);
    }

    return true;
  }
}

describe('Communication Protocol', () => {
  let channel;
  let protocol;

  beforeEach(() => {
    channel = new CommunicationChannel();
    protocol = new ProtocolHandler(channel);
  });

  describe('Agent Registration', () => {
    it('should register agents with handlers', () => {
      const handler = jest.fn();
      channel.registerAgent('agent-1', handler);

      expect(channel.messageHandlers.has('agent-1')).toBe(true);
      expect(channel.messageQueue.has('agent-1')).toBe(true);
    });

    it('should emit registration event', (done) => {
      channel.on('agent:registered', (agentId) => {
        expect(agentId).toBe('agent-1');
        done();
      });

      channel.registerAgent('agent-1', jest.fn());
    });

    it('should unregister agents', () => {
      channel.registerAgent('agent-1', jest.fn());
      channel.unregisterAgent('agent-1');

      expect(channel.messageHandlers.has('agent-1')).toBe(false);
      expect(channel.messageQueue.has('agent-1')).toBe(false);
    });

    it('should clean up subscriptions on unregister', () => {
      channel.registerAgent('agent-1', jest.fn());
      channel.subscribe('agent-1', 'updates');
      
      expect(channel.subscriptions.get('updates').has('agent-1')).toBe(true);
      
      channel.unregisterAgent('agent-1');
      
      expect(channel.subscriptions.get('updates').has('agent-1')).toBe(false);
    });
  });

  describe('Direct Messaging', () => {
    it('should send message between agents', () => {
      const handler1 = jest.fn();
      const handler2 = jest.fn();
      
      channel.registerAgent('agent-1', handler1);
      channel.registerAgent('agent-2', handler2);

      const message = channel.sendMessage('agent-1', 'agent-2', 'Hello');

      expect(message.from).toBe('agent-1');
      expect(message.to).toBe('agent-2');
      expect(message.content).toBe('Hello');
      expect(handler2).toHaveBeenCalledWith(expect.objectContaining({
        content: 'Hello',
        status: 'delivered'
      }));
    });

    it('should validate sender exists', () => {
      channel.registerAgent('agent-1', jest.fn());

      expect(() => {
        channel.sendMessage('non-existent', 'agent-1', 'Hello');
      }).toThrow(/Sender agent not registered/);
    });

    it('should validate recipient exists', () => {
      channel.registerAgent('agent-1', jest.fn());

      expect(() => {
        channel.sendMessage('agent-1', 'non-existent', 'Hello');
      }).toThrow(/Recipient agent not registered/);
    });

    it('should handle message priorities', () => {
      const received = [];
      const handler = jest.fn(msg => received.push(msg.content));
      
      channel.registerAgent('agent-1', jest.fn());
      channel.registerAgent('agent-2', handler);

      // Prevent immediate processing
      channel._processMessageQueue = jest.fn();

      // Send messages with different priorities
      channel.sendMessage('agent-1', 'agent-2', 'Low', MESSAGE_TYPES.INFO, PRIORITIES.LOW);
      channel.sendMessage('agent-1', 'agent-2', 'Critical', MESSAGE_TYPES.INFO, PRIORITIES.CRITICAL);
      channel.sendMessage('agent-1', 'agent-2', 'Medium', MESSAGE_TYPES.INFO, PRIORITIES.MEDIUM);

      // Process queue manually
      const queue = channel.messageQueue.get('agent-2');
      expect(queue[0].priority).toBe(PRIORITIES.CRITICAL);
      expect(queue[1].priority).toBe(PRIORITIES.MEDIUM);
      expect(queue[2].priority).toBe(PRIORITIES.LOW);
    });
  });

  describe('Broadcasting', () => {
    it('should broadcast to all agents except sender', () => {
      const handlers = {
        agent1: jest.fn(),
        agent2: jest.fn(),
        agent3: jest.fn()
      };

      Object.entries(handlers).forEach(([id, handler]) => {
        channel.registerAgent(id, handler);
      });

      channel.sendMessage('agent1', null, 'Broadcast message');

      expect(handlers.agent1).not.toHaveBeenCalled();
      expect(handlers.agent2).toHaveBeenCalledWith(expect.objectContaining({
        content: 'Broadcast message',
        type: MESSAGE_TYPES.BROADCAST
      }));
      expect(handlers.agent3).toHaveBeenCalledWith(expect.objectContaining({
        content: 'Broadcast message',
        type: MESSAGE_TYPES.BROADCAST
      }));
    });

    it('should track broadcast metrics', () => {
      channel.registerAgent('agent-1', jest.fn());
      channel.registerAgent('agent-2', jest.fn());
      channel.registerAgent('agent-3', jest.fn());

      channel.sendMessage('agent-1', null, 'Broadcast');

      const metrics = channel.getMetrics();
      expect(metrics.broadcast).toBe(1);
      expect(metrics.sent).toBe(1);
      expect(metrics.received).toBe(2); // 2 agents received
    });
  });

  describe('Topic Subscriptions', () => {
    it('should subscribe agents to topics', () => {
      channel.registerAgent('agent-1', jest.fn());
      channel.registerAgent('agent-2', jest.fn());

      channel.subscribe('agent-1', 'updates');
      channel.subscribe('agent-2', 'updates');

      const subscribers = channel.subscriptions.get('updates');
      expect(subscribers.size).toBe(2);
      expect(subscribers.has('agent-1')).toBe(true);
      expect(subscribers.has('agent-2')).toBe(true);
    });

    it('should publish to topic subscribers', () => {
      const handler1 = jest.fn();
      const handler2 = jest.fn();
      const handler3 = jest.fn();

      channel.registerAgent('agent-1', handler1);
      channel.registerAgent('agent-2', handler2);
      channel.registerAgent('agent-3', handler3);

      channel.subscribe('agent-1', 'news');
      channel.subscribe('agent-2', 'news');
      // agent-3 not subscribed

      const sent = channel.publishToTopic('news', 'Breaking news!', 'agent-1');

      expect(sent).toBe(1); // Only agent-2 receives (not sender)
      expect(handler1).not.toHaveBeenCalled();
      expect(handler2).toHaveBeenCalled();
      expect(handler3).not.toHaveBeenCalled();
    });

    it('should unsubscribe from topics', () => {
      channel.registerAgent('agent-1', jest.fn());
      channel.subscribe('agent-1', 'updates');
      channel.unsubscribe('agent-1', 'updates');

      const subscribers = channel.subscriptions.get('updates');
      expect(subscribers).toBeUndefined();
    });
  });

  describe('Message History', () => {
    beforeEach(() => {
      channel.registerAgent('agent-1', jest.fn());
      channel.registerAgent('agent-2', jest.fn());
      channel.registerAgent('agent-3', jest.fn());
    });

    it('should store message history', () => {
      channel.sendMessage('agent-1', 'agent-2', 'Message 1');
      channel.sendMessage('agent-2', 'agent-1', 'Message 2');
      channel.sendMessage('agent-1', 'agent-3', 'Message 3');

      const history = channel.getMessageHistory();
      expect(history).toHaveLength(3);
    });

    it('should filter history by sender', () => {
      channel.sendMessage('agent-1', 'agent-2', 'From 1');
      channel.sendMessage('agent-2', 'agent-1', 'From 2');
      channel.sendMessage('agent-1', 'agent-3', 'From 1 again');

      const fromAgent1 = channel.getMessageHistory({ from: 'agent-1' });
      expect(fromAgent1).toHaveLength(2);
      expect(fromAgent1.every(m => m.from === 'agent-1')).toBe(true);
    });

    it('should filter history by type', () => {
      channel.sendMessage('agent-1', 'agent-2', 'Info', MESSAGE_TYPES.INFO);
      channel.sendMessage('agent-1', 'agent-2', 'Request', MESSAGE_TYPES.REQUEST);
      channel.sendMessage('agent-1', 'agent-2', 'Error', MESSAGE_TYPES.ERROR);

      const requests = channel.getMessageHistory({ type: MESSAGE_TYPES.REQUEST });
      expect(requests).toHaveLength(1);
      expect(requests[0].content).toBe('Request');
    });

    it('should get conversation between two agents', () => {
      channel.sendMessage('agent-1', 'agent-2', 'Hi');
      channel.sendMessage('agent-2', 'agent-1', 'Hello');
      channel.sendMessage('agent-1', 'agent-3', 'Different convo');
      channel.sendMessage('agent-1', 'agent-2', 'How are you?');

      const conversation = channel.getConversation('agent-1', 'agent-2');
      expect(conversation).toHaveLength(3);
      expect(conversation[0].content).toBe('Hi');
      expect(conversation[1].content).toBe('Hello');
      expect(conversation[2].content).toBe('How are you?');
    });
  });

  describe('Request/Response Protocol', () => {
    it('should handle request-response pattern', async () => {
      const responder = jest.fn((message) => {
        if (message.type === MESSAGE_TYPES.REQUEST) {
          const { requestId, action, params } = message.content;
          
          if (action === 'calculate') {
            const result = params.a + params.b;
            protocol.response('agent-2', 'agent-1', requestId, result);
          }
        }
      });

      const requester = jest.fn((message) => {
        protocol.handleResponse(message);
      });

      channel.registerAgent('agent-1', requester);
      channel.registerAgent('agent-2', responder);

      const result = await protocol.request(
        'agent-1',
        'agent-2',
        'calculate',
        { a: 5, b: 3 }
      );

      expect(result).toBe(8);
    });

    it('should handle request timeout', async () => {
      channel.registerAgent('agent-1', jest.fn());
      channel.registerAgent('agent-2', jest.fn()); // Doesn't respond

      await expect(
        protocol.request('agent-1', 'agent-2', 'slow-action', {}, 100)
      ).rejects.toThrow(/Request timeout/);
    });

    it('should handle error responses', async () => {
      const responder = jest.fn((message) => {
        if (message.type === MESSAGE_TYPES.REQUEST) {
          const { requestId } = message.content;
          protocol.response('agent-2', 'agent-1', requestId, null, 'Division by zero');
        }
      });

      const requester = jest.fn((message) => {
        protocol.handleResponse(message);
      });

      channel.registerAgent('agent-1', requester);
      channel.registerAgent('agent-2', responder);

      await expect(
        protocol.request('agent-1', 'agent-2', 'divide', { a: 10, b: 0 })
      ).rejects.toThrow(/Division by zero/);
    });
  });

  describe('Metrics and Monitoring', () => {
    it('should track message metrics', () => {
      channel.registerAgent('agent-1', jest.fn());
      channel.registerAgent('agent-2', jest.fn());
      channel.registerAgent('agent-3', jest.fn());

      channel.sendMessage('agent-1', 'agent-2', 'Direct');
      channel.sendMessage('agent-1', null, 'Broadcast');

      const metrics = channel.getMetrics();
      expect(metrics.sent).toBe(2);
      expect(metrics.received).toBe(3); // 1 direct + 2 broadcast
      expect(metrics.broadcast).toBe(1);
      expect(metrics.failed).toBe(0);
      expect(metrics.totalMessages).toBe(4); // 1 original broadcast + 2 delivered + 1 direct
    });

    it('should track failed messages', () => {
      channel.registerAgent('agent-1', jest.fn());
      channel.registerAgent('agent-2', jest.fn(() => {
        throw new Error('Handler error');
      }));

      channel.sendMessage('agent-1', 'agent-2', 'Will fail');

      const metrics = channel.getMetrics();
      expect(metrics.failed).toBe(1);
    });

    it('should clear history', () => {
      channel.registerAgent('agent-1', jest.fn());
      channel.registerAgent('agent-2', jest.fn());

      channel.sendMessage('agent-1', 'agent-2', 'Message');
      expect(channel.getMessageHistory()).toHaveLength(1);

      channel.clearHistory();
      expect(channel.getMessageHistory()).toHaveLength(0);
      expect(channel.getMetrics().sent).toBe(0);
    });
  });
});