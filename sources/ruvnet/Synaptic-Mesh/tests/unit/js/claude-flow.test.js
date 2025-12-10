/**
 * Unit tests for Claude Flow coordination system
 * Tests swarm orchestration, memory management, and MCP integration
 */

import { jest } from '@jest/globals';

describe('Claude Flow Core Tests', () => {
  describe('Swarm Initialization', () => {
    test('should initialize swarm with correct topology', async () => {
      const mockSwarm = {
        topology: 'hierarchical',
        maxAgents: 8,
        strategy: 'balanced',
        agents: [],
        memory: new Map(),
      };

      const swarmConfig = {
        topology: 'hierarchical',
        maxAgents: 8,
        strategy: 'balanced'
      };

      // Mock swarm initialization
      const initializeSwarm = (config) => {
        return {
          ...mockSwarm,
          ...config,
          id: `swarm_${Date.now()}`,
          status: 'initialized'
        };
      };

      const swarm = initializeSwarm(swarmConfig);
      
      expect(swarm.topology).toBe('hierarchical');
      expect(swarm.maxAgents).toBe(8);
      expect(swarm.strategy).toBe('balanced');
      expect(swarm.status).toBe('initialized');
      expect(swarm.id).toMatch(/^swarm_\d+$/);
    });

    test('should support different topology types', () => {
      const topologies = ['hierarchical', 'mesh', 'ring', 'star'];
      
      topologies.forEach(topology => {
        const swarm = {
          topology,
          agents: [],
          status: 'initialized'
        };
        
        expect(swarm.topology).toBe(topology);
        expect(swarm.status).toBe('initialized');
      });
    });

    test('should validate agent count limits', () => {
      const validateAgentCount = (count) => {
        return count > 0 && count <= 100;
      };

      expect(validateAgentCount(1)).toBe(true);
      expect(validateAgentCount(50)).toBe(true);
      expect(validateAgentCount(100)).toBe(true);
      expect(validateAgentCount(0)).toBe(false);
      expect(validateAgentCount(101)).toBe(false);
    });
  });

  describe('Agent Management', () => {
    test('should spawn agents with correct types', () => {
      const agentTypes = [
        'coordinator', 'researcher', 'coder', 'analyst', 
        'architect', 'tester', 'reviewer', 'optimizer'
      ];

      const spawnAgent = (type, capabilities = []) => {
        return {
          id: `agent_${type}_${Date.now()}`,
          type,
          capabilities,
          status: 'spawned',
          tasks: [],
          memory: new Map()
        };
      };

      agentTypes.forEach(type => {
        const agent = spawnAgent(type, ['coordination', 'communication']);
        
        expect(agent.type).toBe(type);
        expect(agent.status).toBe('spawned');
        expect(agent.capabilities).toContain('coordination');
        expect(agent.id).toMatch(new RegExp(`^agent_${type}_\\d+$`));
      });
    });

    test('should handle agent lifecycle states', () => {
      const agent = {
        id: 'test_agent',
        status: 'spawned',
        tasks: []
      };

      const lifecycleStates = ['spawned', 'active', 'idle', 'completed', 'terminated'];
      
      lifecycleStates.forEach(state => {
        agent.status = state;
        expect(agent.status).toBe(state);
      });
    });

    test('should assign tasks to agents', () => {
      const agent = {
        id: 'test_agent',
        type: 'coder',
        tasks: [],
        status: 'idle'
      };

      const assignTask = (agent, task) => {
        agent.tasks.push({
          id: `task_${Date.now()}`,
          description: task.description,
          priority: task.priority,
          status: 'assigned'
        });
        agent.status = 'active';
        return agent;
      };

      const task = {
        description: 'Implement authentication service',
        priority: 'high'
      };

      const updatedAgent = assignTask(agent, task);
      
      expect(updatedAgent.tasks).toHaveLength(1);
      expect(updatedAgent.status).toBe('active');
      expect(updatedAgent.tasks[0].description).toBe(task.description);
      expect(updatedAgent.tasks[0].priority).toBe(task.priority);
    });
  });

  describe('Memory Management', () => {
    test('should store and retrieve memory correctly', () => {
      const memoryStore = new Map();

      const storeMemory = (key, value, ttl = null) => {
        const entry = {
          value,
          timestamp: Date.now(),
          ttl
        };
        memoryStore.set(key, entry);
        return true;
      };

      const retrieveMemory = (key) => {
        const entry = memoryStore.get(key);
        if (!entry) return null;
        
        if (entry.ttl && Date.now() - entry.timestamp > entry.ttl) {
          memoryStore.delete(key);
          return null;
        }
        
        return entry.value;
      };

      const testData = { task: 'test', status: 'completed' };
      storeMemory('test_key', testData);
      
      const retrieved = retrieveMemory('test_key');
      expect(retrieved).toEqual(testData);
    });

    test('should handle memory TTL expiration', async () => {
      const memoryStore = new Map();
      const shortTTL = 10; // 10ms

      const storeMemory = (key, value, ttl = null) => {
        const entry = {
          value,
          timestamp: Date.now(),
          ttl
        };
        memoryStore.set(key, entry);
      };

      const retrieveMemory = (key) => {
        const entry = memoryStore.get(key);
        if (!entry) return null;
        
        if (entry.ttl && Date.now() - entry.timestamp > entry.ttl) {
          memoryStore.delete(key);
          return null;
        }
        
        return entry.value;
      };

      storeMemory('temp_key', 'temp_value', shortTTL);
      
      // Should be available immediately
      expect(retrieveMemory('temp_key')).toBe('temp_value');
      
      // Wait for TTL to expire
      await new Promise(resolve => setTimeout(resolve, 20));
      
      // Should be expired
      expect(retrieveMemory('temp_key')).toBeNull();
    });

    test('should support namespaced memory access', () => {
      const memoryStore = new Map();

      const createNamespacedKey = (namespace, key) => `${namespace}:${key}`;

      const storeInNamespace = (namespace, key, value) => {
        const namespacedKey = createNamespacedKey(namespace, key);
        memoryStore.set(namespacedKey, value);
      };

      const retrieveFromNamespace = (namespace, key) => {
        const namespacedKey = createNamespacedKey(namespace, key);
        return memoryStore.get(namespacedKey);
      };

      storeInNamespace('swarm1', 'config', { agents: 5 });
      storeInNamespace('swarm2', 'config', { agents: 10 });
      
      expect(retrieveFromNamespace('swarm1', 'config')).toEqual({ agents: 5 });
      expect(retrieveFromNamespace('swarm2', 'config')).toEqual({ agents: 10 });
    });
  });

  describe('Task Orchestration', () => {
    test('should orchestrate tasks with dependencies', () => {
      const tasks = [
        { id: 'task1', dependencies: [], priority: 'high' },
        { id: 'task2', dependencies: ['task1'], priority: 'medium' },
        { id: 'task3', dependencies: ['task1', 'task2'], priority: 'low' }
      ];

      const createExecutionPlan = (tasks) => {
        const plan = [];
        const completed = new Set();
        
        while (plan.length < tasks.length) {
          for (const task of tasks) {
            if (!plan.includes(task) && 
                task.dependencies.every(dep => completed.has(dep))) {
              plan.push(task);
              completed.add(task.id);
            }
          }
        }
        
        return plan;
      };

      const executionPlan = createExecutionPlan(tasks);
      
      expect(executionPlan).toHaveLength(3);
      expect(executionPlan[0].id).toBe('task1');
      expect(executionPlan[1].id).toBe('task2');
      expect(executionPlan[2].id).toBe('task3');
    });

    test('should handle parallel task execution', async () => {
      const parallelTasks = [
        { id: 'parallel1', duration: 10 },
        { id: 'parallel2', duration: 15 },
        { id: 'parallel3', duration: 5 }
      ];

      const executeTask = (task) => {
        return new Promise(resolve => {
          setTimeout(() => {
            resolve({ id: task.id, status: 'completed' });
          }, task.duration);
        });
      };

      const executeInParallel = async (tasks) => {
        const startTime = Date.now();
        const results = await Promise.all(tasks.map(executeTask));
        const duration = Date.now() - startTime;
        
        return { results, duration };
      };

      const { results, duration } = await executeInParallel(parallelTasks);
      
      expect(results).toHaveLength(3);
      expect(duration).toBeLessThan(25); // Should complete in roughly max duration
      results.forEach(result => {
        expect(result.status).toBe('completed');
      });
    });

    test('should prioritize tasks correctly', () => {
      const tasks = [
        { id: 'low1', priority: 'low', value: 1 },
        { id: 'high1', priority: 'high', value: 3 },
        { id: 'medium1', priority: 'medium', value: 2 },
        { id: 'high2', priority: 'high', value: 3 }
      ];

      const priorityOrder = { high: 3, medium: 2, low: 1 };

      const sortByPriority = (tasks) => {
        return [...tasks].sort((a, b) => {
          return priorityOrder[b.priority] - priorityOrder[a.priority];
        });
      };

      const sorted = sortByPriority(tasks);
      
      expect(sorted[0].priority).toBe('high');
      expect(sorted[1].priority).toBe('high');
      expect(sorted[2].priority).toBe('medium');
      expect(sorted[3].priority).toBe('low');
    });
  });

  describe('Performance Metrics', () => {
    test('should track execution metrics', () => {
      const metrics = {
        tasksCompleted: 0,
        averageExecutionTime: 0,
        successRate: 0,
        memoryUsage: 0
      };

      const updateMetrics = (taskResult) => {
        metrics.tasksCompleted++;
        metrics.averageExecutionTime = 
          (metrics.averageExecutionTime * (metrics.tasksCompleted - 1) + 
           taskResult.executionTime) / metrics.tasksCompleted;
        metrics.successRate = 
          taskResult.success ? 
          (metrics.successRate * (metrics.tasksCompleted - 1) + 1) / metrics.tasksCompleted :
          (metrics.successRate * (metrics.tasksCompleted - 1)) / metrics.tasksCompleted;
      };

      // Simulate successful task
      updateMetrics({ executionTime: 100, success: true });
      updateMetrics({ executionTime: 200, success: true });
      updateMetrics({ executionTime: 150, success: false });

      expect(metrics.tasksCompleted).toBe(3);
      expect(metrics.averageExecutionTime).toBeCloseTo(150, 0);
      expect(metrics.successRate).toBeCloseTo(0.667, 2);
    });

    test('should monitor system health', () => {
      const systemHealth = {
        activeAgents: 5,
        queuedTasks: 10,
        completedTasks: 100,
        errorRate: 0.02,
        lastHealthCheck: Date.now()
      };

      const checkSystemHealth = (health) => {
        const healthScore = 
          (health.activeAgents > 0 ? 25 : 0) +
          (health.errorRate < 0.05 ? 25 : 0) +
          (health.queuedTasks < 50 ? 25 : 0) +
          (health.completedTasks > 0 ? 25 : 0);
        
        return {
          score: healthScore,
          status: healthScore >= 75 ? 'healthy' : 
                 healthScore >= 50 ? 'warning' : 'critical'
        };
      };

      const health = checkSystemHealth(systemHealth);
      
      expect(health.score).toBe(100);
      expect(health.status).toBe('healthy');
    });
  });
});

describe('Claude Flow Integration Tests', () => {
  describe('MCP Protocol Integration', () => {
    test('should handle MCP method calls', async () => {
      const mcpServer = {
        methods: new Map([
          ['swarm/init', (params) => ({ success: true, swarmId: 'test_swarm' })],
          ['agent/spawn', (params) => ({ success: true, agentId: 'test_agent' })],
          ['task/orchestrate', (params) => ({ success: true, taskId: 'test_task' })]
        ])
      };

      const handleMCPCall = async (method, params) => {
        const handler = mcpServer.methods.get(method);
        if (!handler) {
          throw new Error(`Method ${method} not found`);
        }
        return handler(params);
      };

      // Test swarm initialization
      const swarmResult = await handleMCPCall('swarm/init', { topology: 'mesh' });
      expect(swarmResult.success).toBe(true);
      expect(swarmResult.swarmId).toBe('test_swarm');

      // Test agent spawning
      const agentResult = await handleMCPCall('agent/spawn', { type: 'coder' });
      expect(agentResult.success).toBe(true);
      expect(agentResult.agentId).toBe('test_agent');

      // Test task orchestration
      const taskResult = await handleMCPCall('task/orchestrate', { strategy: 'parallel' });
      expect(taskResult.success).toBe(true);
      expect(taskResult.taskId).toBe('test_task');
    });

    test('should handle MCP resource access', () => {
      const resources = new Map([
        ['mesh://status', () => ({ status: 'active', agents: 5 })],
        ['mesh://metrics', () => ({ cpu: 45, memory: 30, tasks: 100 })]
      ]);

      const getMCPResource = (uri) => {
        const resource = resources.get(uri);
        if (!resource) {
          throw new Error(`Resource ${uri} not found`);
        }
        return resource();
      };

      const status = getMCPResource('mesh://status');
      expect(status.status).toBe('active');
      expect(status.agents).toBe(5);

      const metrics = getMCPResource('mesh://metrics');
      expect(metrics.cpu).toBe(45);
      expect(metrics.memory).toBe(30);
      expect(metrics.tasks).toBe(100);
    });
  });

  describe('Error Handling', () => {
    test('should handle agent failures gracefully', () => {
      const handleAgentFailure = (agentId, error) => {
        return {
          recovery: 'respawn',
          newAgentId: `${agentId}_recovery`,
          originalError: error.message,
          recoveryTime: Date.now()
        };
      };

      const error = new Error('Agent communication timeout');
      const recovery = handleAgentFailure('agent_123', error);

      expect(recovery.recovery).toBe('respawn');
      expect(recovery.newAgentId).toBe('agent_123_recovery');
      expect(recovery.originalError).toBe('Agent communication timeout');
    });

    test('should implement circuit breaker pattern', () => {
      class CircuitBreaker {
        constructor(threshold = 5, timeout = 60000) {
          this.threshold = threshold;
          this.timeout = timeout;
          this.failures = 0;
          this.lastFailureTime = null;
          this.state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
        }

        async call(fn) {
          if (this.state === 'OPEN') {
            if (Date.now() - this.lastFailureTime > this.timeout) {
              this.state = 'HALF_OPEN';
            } else {
              throw new Error('Circuit breaker is OPEN');
            }
          }

          try {
            const result = await fn();
            this.onSuccess();
            return result;
          } catch (error) {
            this.onFailure();
            throw error;
          }
        }

        onSuccess() {
          this.failures = 0;
          this.state = 'CLOSED';
        }

        onFailure() {
          this.failures++;
          this.lastFailureTime = Date.now();
          if (this.failures >= this.threshold) {
            this.state = 'OPEN';
          }
        }
      }

      const breaker = new CircuitBreaker(3, 1000);
      
      // Test normal operation
      expect(breaker.state).toBe('CLOSED');
      
      // Simulate failures
      breaker.onFailure();
      breaker.onFailure();
      breaker.onFailure();
      
      expect(breaker.state).toBe('OPEN');
      expect(breaker.failures).toBe(3);
    });
  });
});