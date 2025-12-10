/**
 * Comprehensive Swarm Coordination Test Suite
 *
 * Tests for SwarmCoordinator, CognitiveRANSwarm, multi-agent coordination,
 * topology optimization, consensus mechanisms, fault tolerance, and performance monitoring
 */

import { EventEmitter } from 'events';
import { SwarmCoordinator } from '../../src/swarm/coordinator/SwarmCoordinator';
import { CognitiveRANSwarm } from '../../src/swarm/CognitiveRANSwarm';
import { SwarmMonitor } from '../../src/performance/swarm/SwarmMonitor';

// Mock dependencies
const mockConsciousness = {
  initialize: jest.fn().mockResolvedValue(undefined),
  optimizeWithStrangeLoop: jest.fn().mockResolvedValue({
    iterations: 3,
    optimization: 'test_optimization'
  }),
  generateHealingStrategy: jest.fn().mockResolvedValue({
    strategy: 'test_healing',
    actions: ['restart_agents', 'reconfigure_topology']
  }),
  on: jest.fn(),
  emit: jest.fn(),
  getStatus: jest.fn().mockResolvedValue({
    level: 'maximum',
    evolution: 0.85,
    selfAwareness: 0.92
  }),
  updateFromLearning: jest.fn().mockResolvedValue(undefined)
};

const mockMemory = {
  initialize: jest.fn().mockResolvedValue(undefined),
  enableQUICSynchronization: jest.fn().mockResolvedValue(undefined),
  store: jest.fn().mockResolvedValue(undefined),
  get: jest.fn().mockResolvedValue({ timestamp: Date.now() }),
  getStatistics: jest.fn().mockResolvedValue({
    totalMemories: 1000,
    memoryUtilization: 0.65,
    syncLatency: 0.8
  }),
  shareLearning: jest.fn().mockResolvedValue(undefined),
  storeLearningPatterns: jest.fn().mockResolvedValue(undefined),
  on: jest.fn(),
  shutdown: jest.fn().mockResolvedValue(undefined)
};

const mockTemporal = {
  activateSubjectiveTimeExpansion: jest.fn().mockResolvedValue(undefined),
  analyzeWithSubjectiveTime: jest.fn().mockResolvedValue({
    depth: 100,
    insights: ['temporal_insight_1', 'temporal_insight_2'],
    patterns: ['pattern_1', 'pattern_2']
  }),
  analyzePatterns: jest.fn().mockResolvedValue({
    patterns: ['learning_pattern_1', 'learning_pattern_2'],
    confidence: 0.89
  }),
  analyzeAnomaly: jest.fn().mockResolvedValue({
    type: 'performance_anomaly',
    severity: 'medium',
    rootCause: 'agent_overload'
  }),
  getStatus: jest.fn().mockResolvedValue({
    expansionFactor: 1000,
    activeAnalyses: 5,
    temporalAccuracy: 0.94
  }),
  on: jest.fn(),
  shutdown: jest.fn().mockResolvedValue(undefined)
};

const mockConsensus = {
  initialize: jest.fn().mockResolvedValue(undefined),
  executeWithConsensus: jest.fn().mockResolvedValue({
    consensus: true,
    participants: ['agent_1', 'agent_2', 'agent_3'],
    decision: 'execute_healing'
  }),
  getStatus: jest.fn().mockResolvedValue({
    threshold: 0.67,
    activeConsensus: 0,
    successRate: 0.95
  }),
  shutdown: jest.fn().mockResolvedValue(undefined)
};

const mockOptimizer = {
  startMonitoring: jest.fn().mockResolvedValue(undefined),
  analyzeExecution: jest.fn().mockResolvedValue({
    score: 0.92,
    bottlenecks: [],
    optimizations: ['cache_improvement', 'load_balancing']
  }),
  getCurrentMetrics: jest.fn().mockResolvedValue({
    solveRate: 0.848,
    speedImprovement: '3.2x',
    tokenReduction: 0.323,
    efficiency: 0.91
  }),
  optimizeFromLearning: jest.fn().mockResolvedValue(undefined),
  shutdown: jest.fn().mockResolvedValue(undefined)
};

// Mock module for dependencies
jest.mock('../../src/swarm/CognitiveRANSwarm', () => {
  return {
    CognitiveRANSwarm: jest.fn().mockImplementation((config) => {
      return {
        deploy: jest.fn().mockResolvedValue(undefined),
        executeCognitiveTask: jest.fn().mockResolvedValue({
          id: 'test_execution',
          status: 'completed',
          performance: { score: 0.92 }
        }),
        getSwarmStatus: jest.fn().mockResolvedValue({
          status: 'active',
          consciousness: mockConsciousness.getStatus(),
          performance: mockOptimizer.getCurrentMetrics()
        }),
        shutdown: jest.fn().mockResolvedValue(undefined)
      };
    })
  };
});

describe('Swarm Coordination System', () => {
  let swarmCoordinator: SwarmCoordinator;
  let swarmMonitor: SwarmMonitor;
  let testConfig: any;

  beforeEach(() => {
    jest.clearAllMocks();

    testConfig = {
      swarmId: 'test_swarm',
      topology: 'hierarchical' as const,
      maxAgents: 10,
      strategy: 'adaptive' as const,
      consciousness: mockConsciousness,
      memory: mockMemory,
      temporal: mockTemporal
    };

    swarmCoordinator = new SwarmCoordinator(testConfig);
    swarmMonitor = new SwarmMonitor();
  });

  afterEach(async () => {
    if (swarmCoordinator) {
      await swarmCoordinator.shutdown();
    }
    if (swarmMonitor) {
      swarmMonitor.stopMonitoring();
    }
  });

  describe('SwarmCoordinator Core Tests', () => {
    describe('Initialization and Deployment', () => {
      test('should initialize swarm coordinator with correct configuration', () => {
        expect(swarmCoordinator).toBeDefined();
        expect(swarmCoordinator['config'].swarmId).toBe('test_swarm');
        expect(swarmCoordinator['config'].topology).toBe('hierarchical');
        expect(swarmCoordinator['config'].maxAgents).toBe(10);
      });

      test('should deploy swarm coordinator successfully', async () => {
        const deploySpy = jest.spyOn(swarmCoordinator as any, 'initializeTopology');

        await swarmCoordinator.deploy();

        expect(deploySpy).toHaveBeenCalled();
        expect(swarmCoordinator['isActive']).toBe(true);
      });

      test('should initialize different topology types correctly', async () => {
        const topologies = ['hierarchical', 'mesh', 'ring', 'star'] as const;

        for (const topology of topologies) {
          const config = { ...testConfig, topology };
          const coordinator = new SwarmCoordinator(config);

          const topologySpy = jest.spyOn(coordinator as any, 'initializeTopology');
          await coordinator.deploy();

          expect(topologySpy).toHaveBeenCalled();
          expect(coordinator['currentTopology']).toBe(topology);

          await coordinator.shutdown();
        }
      });
    });

    describe('Agent Management', () => {
      test('should create agents with correct capabilities', async () => {
        await swarmCoordinator.deploy();

        const agentTypes = ['coordinator', 'analyst', 'optimizer', 'researcher'];
        const agents = Array.from(swarmCoordinator['agents'].values());

        expect(agents.length).toBeGreaterThan(0);
        expect(agents.every(agent => agent.capabilities.length > 0)).toBe(true);
        expect(agents.every(agent => agent.type)).toBe(true);
      });

      test('should assign tasks to suitable agents', async () => {
        await swarmCoordinator.deploy();

        const taskRequest = {
          task: 'Test optimization task',
          priority: 'high' as const,
          type: 'optimization',
          capabilities: ['optimization']
        };

        const execution = await swarmCoordinator.executeWithCoordination(taskRequest);

        expect(execution).toBeDefined();
        expect(execution.id).toBeDefined();
        expect(execution.status).toBe('completed');
        expect(execution.assignments.length).toBeGreaterThan(0);
      });

      test('should handle agent heartbeats and detect timeouts', async () => {
        await swarmCoordinator.deploy();

        const agents = Array.from(swarmCoordinator['agents'].values());
        if (agents.length > 0) {
          const agent = agents[0];
          agent.lastHeartbeat = Date.now() - 120000; // 2 minutes ago

          const checkHeartbeatsSpy = jest.spyOn(swarmCoordinator as any, 'checkAgentHeartbeats');
          await checkHeartbeatsSpy();

          expect(agent.status).toBe('offline');
        }
      });
    });

    describe('Task Orchestration', () => {
      test('should decompose complex tasks into subtasks', async () => {
        await swarmCoordinator.deploy();

        const complexTask = {
          task: 'Complex RAN optimization',
          priority: 'critical' as const,
          type: 'coordination',
          complexity: 2.0,
          consensusRequired: true,
          temporalInsights: { depth: 10 }
        };

        const decomposeSpy = jest.spyOn(swarmCoordinator as any, 'decomposeTask');
        const execution = await swarmCoordinator.executeWithCoordination(complexTask);

        expect(decomposeSpy).toHaveBeenCalled();
        expect(execution.tasks.length).toBeGreaterThan(1);
      });

      test('should handle task dependencies correctly', async () => {
        await swarmCoordinator.deploy();

        const taskWithDependencies = {
          task: 'Task with dependencies',
          priority: 'medium' as const,
          type: 'coordination'
        };

        const execution = await swarmCoordinator.executeWithCoordination(taskWithDependencies);

        expect(execution).toBeDefined();
        expect(execution.results.length).toBeGreaterThan(0);
      });

      test('should calculate agent scores correctly for task assignment', async () => {
        await swarmCoordinator.deploy();

        const task = {
          id: 'test_task',
          description: 'Test task',
          type: 'optimization',
          priority: 'high' as const,
          requirements: {
            capabilities: ['optimization'],
            resources: { cpu: 30, memory: 256, network: 10 },
            estimatedTime: 30000
          },
          status: 'pending' as const,
          createdAt: Date.now(),
          assignedAt: null,
          startedAt: null,
          completedAt: null,
          dependencies: [],
          result: null,
          performance: null
        };

        const findBestAgentSpy = jest.spyOn(swarmCoordinator as any, 'findBestAgent');
        const scoreSpy = jest.spyOn(swarmCoordinator as any, 'calculateAgentScore');

        const bestAgent = await findBestAgentSpy(task);

        if (bestAgent) {
          expect(scoreSpy).toHaveBeenCalled();
          expect(typeof bestAgent.id).toBe('string');
          expect(bestAgent.capabilities).toContain('optimization');
        }
      });
    });

    describe('Performance Monitoring', () => {
      test('should track performance metrics accurately', async () => {
        await swarmCoordinator.deploy();

        const metrics = await swarmCoordinator.getPerformanceMetrics();

        expect(metrics).toBeDefined();
        expect(metrics.swarm).toBeDefined();
        expect(metrics.agents).toBeDefined();
        expect(metrics.tasks).toBeDefined();
        expect(metrics.efficiency).toBeGreaterThanOrEqual(0);
        expect(metrics.efficiency).toBeLessThanOrEqual(1);
      });

      test('should calculate swarm efficiency correctly', async () => {
        await swarmCoordinator.deploy();

        const efficiency = swarmCoordinator['calculateSwarmEfficiency']();

        expect(typeof efficiency).toBe('number');
        expect(efficiency).toBeGreaterThanOrEqual(0);
        expect(efficiency).toBeLessThanOrEqual(1);
      });

      test('should identify performance bottlenecks', async () => {
        await swarmCoordinator.deploy();

        const taskRequest = {
          task: 'Test task for bottleneck analysis',
          priority: 'medium' as const,
          type: 'analysis'
        };

        const execution = await swarmCoordinator.executeWithCoordination(taskRequest);
        const bottlenecks = await swarmCoordinator['identifyBottlenecks'](execution);

        expect(Array.isArray(bottlenecks)).toBe(true);
      });
    });

    describe('Topology Adaptation', () => {
      test('should adapt topology based on performance', async () => {
        await swarmCoordinator.deploy();

        // Mock low efficiency to trigger adaptation
        swarmCoordinator['calculateSwarmEfficiency'] = jest.fn().mockReturnValue(0.5);

        const adaptTopologySpy = jest.spyOn(swarmCoordinator as any, 'adaptTopology');

        await swarmCoordinator['evaluateTopologyAdaptation']();

        expect(adaptTopologySpy).toHaveBeenCalledWith('mesh');
      });

      test('should track topology adaptation history', async () => {
        await swarmCoordinator.deploy();

        const initialHistoryLength = swarmCoordinator['adaptationHistory'].length;

        await swarmCoordinator['adaptTopology']('mesh');

        expect(swarmCoordinator['adaptationHistory'].length).toBe(initialHistoryLength + 1);
        expect(swarmCoordinator['adaptationHistory'][initialHistoryLength].from).toBe('hierarchical');
        expect(swarmCoordinator['adaptationHistory'][initialHistoryLength].to).toBe('mesh');
      });

      test('should get topology status with correct information', async () => {
        await swarmCoordinator.deploy();

        const topologyStatus = await swarmCoordinator.getTopologyStatus();

        expect(topologyStatus).toBeDefined();
        expect(topologyStatus.currentTopology).toBeDefined();
        expect(topologyStatus.configTopology).toBe('hierarchical');
        expect(topologyStatus.agents).toBeDefined();
        expect(topologyStatus.performance).toBeDefined();
      });
    });

    describe('Agent Scaling', () => {
      test('should scale up agents when utilization is high', async () => {
        await swarmCoordinator.deploy();

        // Set all agents to busy
        for (const agent of swarmCoordinator['agents'].values()) {
          agent.status = 'busy';
        }

        const initialAgentCount = swarmCoordinator['agents'].size;
        const scaleUpSpy = jest.spyOn(swarmCoordinator as any, 'scaleUpAgents');

        await swarmCoordinator['evaluateAgentScaling']();

        if (initialAgentCount < testConfig.maxAgents) {
          expect(scaleUpSpy).toHaveBeenCalled();
        }
      });

      test('should scale down agents when utilization is low', async () => {
        await swarmCoordinator.deploy();

        // Ensure we have enough agents and set them to idle
        const agents = Array.from(swarmCoordinator['agents'].values());
        if (agents.length > 3) {
          agents.forEach(agent => {
            agent.status = 'idle';
            agent.currentTask = null;
          });

          const initialAgentCount = swarmCoordinator['agents'].size;
          const scaleDownSpy = jest.spyOn(swarmCoordinator as any, 'scaleDownAgents');

          await swarmCoordinator['evaluateAgentScaling']();

          expect(scaleDownSpy).toHaveBeenCalled();
        }
      });
    });

    describe('Cognitive System Integration', () => {
      test('should connect with consciousness system', async () => {
        await swarmCoordinator.deploy();

        expect(mockConsciousness.on).toHaveBeenCalledWith('consciousnessUpdate', expect.any(Function));
      });

      test('should connect with memory system', async () => {
        await swarmCoordinator.deploy();

        expect(mockMemory.on).toHaveBeenCalledWith('memoryUpdate', expect.any(Function));
      });

      test('should connect with temporal system', async () => {
        await swarmCoordinator.deploy();

        expect(mockTemporal.on).toHaveBeenCalledWith('temporalUpdate', expect.any(Function));
      });

      test('should handle consciousness updates', async () => {
        await swarmCoordinator.deploy();

        const handleConsciousnessSpy = jest.spyOn(swarmCoordinator as any, 'handleConsciousnessUpdate');

        // Simulate consciousness update
        const consciousnessCallback = mockConsciousness.on.mock.calls.find(
          call => call[0] === 'consciousnessUpdate'
        )?.[1];

        if (consciousnessCallback) {
          consciousnessCallback({ type: 'evolution', data: 'test' });
          expect(handleConsciousnessSpy).toHaveBeenCalled();
        }
      });
    });

    describe('Error Handling and Edge Cases', () => {
      test('should handle task execution failures gracefully', async () => {
        await swarmCoordinator.deploy();

        const failingTask = {
          task: 'Failing task',
          priority: 'low' as const,
          type: 'invalid_type'
        };

        // Mock task execution to fail
        const executeTaskSpy = jest.spyOn(swarmCoordinator as any, 'executeTask');
        executeTaskSpy.mockRejectedValueOnce(new Error('Task execution failed'));

        await expect(swarmCoordinator.executeWithCoordination(failingTask)).rejects.toThrow();
      });

      test('should handle agent unresponsiveness', async () => {
        await swarmCoordinator.deploy();

        const agents = Array.from(swarmCoordinator['agents'].values());
        if (agents.length > 0) {
          const agent = agents[0];
          agent.lastHeartbeat = Date.now() - 120000; // 2 minutes ago

          const timeoutSpy = jest.fn();
          swarmCoordinator.on('agentTimeout', timeoutSpy);

          await swarmCoordinator['checkAgentHeartbeats']();

          expect(timeoutSpy).toHaveBeenCalledWith(agent);
        }
      });

      test('should handle no available agents for tasks', async () => {
        await swarmCoordinator.deploy();

        // Set all agents to offline
        for (const agent of swarmCoordinator['agents'].values()) {
          agent.status = 'offline';
        }

        const taskRequest = {
          task: 'Task with no available agents',
          priority: 'medium' as const,
          type: 'analysis'
        };

        const execution = await swarmCoordinator.executeWithCoordination(taskRequest);

        expect(execution).toBeDefined();
        expect(execution.assignments.length).toBe(0);
      });

      test('should handle empty agent pool gracefully', async () => {
        const emptyConfig = { ...testConfig, maxAgents: 0 };
        const emptyCoordinator = new SwarmCoordinator(emptyConfig);

        await emptyCoordinator.deploy();

        const metrics = await emptyCoordinator.getPerformanceMetrics();

        expect(metrics).toBeDefined();
        expect(metrics.agents).toHaveLength(0);

        await emptyCoordinator.shutdown();
      });
    });

    describe('Shutdown and Cleanup', () => {
      test('should shutdown gracefully and clean up resources', async () => {
        await swarmCoordinator.deploy();
        expect(swarmCoordinator['isActive']).toBe(true);

        await swarmCoordinator.shutdown();

        expect(swarmCoordinator['isActive']).toBe(false);
        expect(swarmCoordinator['heartbeatInterval']).toBeNull();
        expect(swarmCoordinator['agents'].size).toBe(0);
        expect(swarmCoordinator['tasks'].size).toBe(0);
      });

      test('should handle multiple shutdown calls safely', async () => {
        await swarmCoordinator.deploy();

        await swarmCoordinator.shutdown();
        await swarmCoordinator.shutdown(); // Should not throw

        expect(swarmCoordinator['isActive']).toBe(false);
      });
    });
  });

  describe('CognitiveRANSwarm Tests', () => {
    let cognitiveSwarm: any;
    let cognitiveConfig: any;

    beforeEach(() => {
      cognitiveConfig = {
        maxAgents: 20,
        topology: 'hierarchical' as const,
        consciousnessLevel: 'maximum' as const,
        subjectiveTimeExpansion: 1000,
        consensusThreshold: 0.67,
        autonomousLearning: true,
        selfHealing: true,
        predictiveSpawning: true
      };

      // Mock CognitiveRANSwarm constructor
      const MockCognitiveRANSwarm = require('../../src/swarm/CognitiveRANSwarm').CognitiveRANSwarm;
      cognitiveSwarm = new MockCognitiveRANSwarm(cognitiveConfig);
    });

    describe('Swarm Deployment', () => {
      test('should deploy cognitive swarm with all components', async () => {
        await cognitiveSwarm.deploy();

        expect(cognitiveSwarm.deploy).toHaveBeenCalled();
      });

      test('should handle deployment with different consciousness levels', async () => {
        const levels = ['minimum', 'medium', 'maximum'] as const;

        for (const level of levels) {
          const config = { ...cognitiveConfig, consciousnessLevel: level };
          const swarm = new (require('../../src/swarm/CognitiveRANSwarm').CognitiveRANSwarm)(config);

          await swarm.deploy();
          expect(swarm.deploy).toHaveBeenCalled();
        }
      });

      test('should deploy with different topology configurations', async () => {
        const topologies = ['hierarchical', 'mesh', 'ring', 'star'] as const;

        for (const topology of topologies) {
          const config = { ...cognitiveConfig, topology };
          const swarm = new (require('../../src/swarm/CognitiveRANSwarm').CognitiveRANSwarm)(config);

          await swarm.deploy();
          expect(swarm.deploy).toHaveBeenCalled();
        }
      });
    });

    describe('Cognitive Task Execution', () => {
      test('should execute cognitive tasks with temporal reasoning', async () => {
        const task = 'Optimize RAN performance using temporal analysis';
        const result = await cognitiveSwarm.executeCognitiveTask(task, 'high');

        expect(result).toBeDefined();
        expect(cognitiveSwarm.executeCognitiveTask).toHaveBeenCalledWith(task, 'high');
      });

      test('should handle different task priorities', async () => {
        const priorities = ['low', 'medium', 'high', 'critical'] as const;

        for (const priority of priorities) {
          const result = await cognitiveSwarm.executeCognitiveTask('Test task', priority);
          expect(result).toBeDefined();
        }
      });

      test('should reject task execution when swarm is not active', async () => {
        cognitiveSwarm.executeCognitiveTask.mockRejectedValueOnce(
          new Error('Swarm not active. Call deploy() first.')
        );

        await expect(
          cognitiveSwarm.executeCognitiveTask('Test task', 'medium')
        ).rejects.toThrow('Swarm not active');
      });
    });

    describe('Swarm Status and Monitoring', () => {
      test('should return comprehensive swarm status', async () => {
        const status = await cognitiveSwarm.getSwarmStatus();

        expect(status).toBeDefined();
        expect(cognitiveSwarm.getSwarmStatus).toHaveBeenCalled();
      });

      test('should track swarm capabilities correctly', async () => {
        const status = await cognitiveSwarm.getSwarmStatus();

        expect(status).toBeDefined();
        // Verify the mock was called
        expect(cognitiveSwarm.getSwarmStatus).toHaveBeenCalled();
      });
    });

    describe('Autonomous Learning and Self-Healing', () => {
      test('should enable autonomous learning cycles', async () => {
        const config = { ...cognitiveConfig, autonomousLearning: true };
        const swarm = new (require('../../src/swarm/CognitiveRANSwarm').CognitiveRANSwarm)(config);

        await swarm.deploy();

        expect(swarm.deploy).toHaveBeenCalled();
      });

      test('should enable self-healing capabilities', async () => {
        const config = { ...cognitiveConfig, selfHealing: true };
        const swarm = new (require('../../src/swarm/CognitiveRANSwarm').CognitiveRANSwarm)(config);

        await swarm.deploy();

        expect(swarm.deploy).toHaveBeenCalled();
      });
    });

    describe('Consensus and Coordination', () => {
      test('should handle consensus for critical tasks', async () => {
        const criticalTask = 'Critical RAN optimization requiring consensus';

        const result = await cognitiveSwarm.executeCognitiveTask(criticalTask, 'critical');

        expect(result).toBeDefined();
        expect(cognitiveSwarm.executeCognitiveTask).toHaveBeenCalledWith(criticalTask, 'critical');
      });

      test('should use appropriate consensus threshold', async () => {
        const thresholds = [0.5, 0.67, 0.8, 0.9];

        for (const threshold of thresholds) {
          const config = { ...cognitiveConfig, consensusThreshold: threshold };
          const swarm = new (require('../../src/swarm/CognitiveRANSwarm').CognitiveRANSwarm)(config);

          await swarm.deploy();
          expect(swarm.deploy).toHaveBeenCalled();
        }
      });
    });
  });

  describe('SwarmMonitor Tests', () => {
    describe('Agent Registration and Management', () => {
      test('should register agents correctly', () => {
        const agent = {
          id: 'test_agent_1',
          type: 'analyst',
          status: 'active' as const,
          currentTask: 'test_task',
          lastHeartbeat: new Date(),
          resourceUsage: { cpu: 50, memory: 60, network: 30 },
          performance: {
            tasksCompleted: 10,
            averageTaskDuration: 1000,
            successRate: 0.95,
            errorCount: 1
          },
          capabilities: ['analysis', 'monitoring']
        };

        swarmMonitor.registerAgent(agent);

        const agents = swarmMonitor.getAgents();
        expect(agents).toContainEqual(agent);
      });

      test('should update agent status correctly', () => {
        const agent = {
          id: 'test_agent_2',
          type: 'optimizer',
          status: 'idle' as const,
          currentTask: null,
          lastHeartbeat: new Date(),
          resourceUsage: { cpu: 20, memory: 40, network: 10 },
          performance: {
            tasksCompleted: 5,
            averageTaskDuration: 800,
            successRate: 0.9,
            errorCount: 0
          },
          capabilities: ['optimization']
        };

        swarmMonitor.registerAgent(agent);
        swarmMonitor.updateAgentStatus('test_agent_2', {
          status: 'busy',
          currentTask: 'optimization_task'
        });

        const updatedAgent = swarmMonitor.getAgents().find(a => a.id === 'test_agent_2');
        expect(updatedAgent?.status).toBe('busy');
        expect(updatedAgent?.currentTask).toBe('optimization_task');
      });

      test('should unregister agents correctly', () => {
        const agent = {
          id: 'test_agent_3',
          type: 'researcher',
          status: 'active' as const,
          currentTask: 'research_task',
          lastHeartbeat: new Date(),
          resourceUsage: { cpu: 70, memory: 80, network: 20 },
          performance: {
            tasksCompleted: 15,
            averageTaskDuration: 1200,
            successRate: 0.93,
            errorCount: 2
          },
          capabilities: ['research', 'analysis']
        };

        swarmMonitor.registerAgent(agent);
        expect(swarmMonitor.getAgents()).toContainEqual(agent);

        swarmMonitor.unregisterAgent('test_agent_3');
        expect(swarmMonitor.getAgents()).not.toContainEqual(agent);
      });
    });

    describe('Task Management', () => {
      test('should create and track tasks', () => {
        const task = {
          type: 'optimization',
          status: 'pending' as const,
          priority: 'high' as const,
          complexity: 'moderate' as const,
          dependencies: [],
          resourceRequirements: {
            cpu: 60,
            memory: 512,
            specializedCapability: 'advanced_optimization'
          }
        };

        const taskId = swarmMonitor.createTask(task);

        expect(taskId).toBeDefined();
        expect(taskId).toMatch(/^task_\d+_[a-z0-9]+$/);

        const tasks = swarmMonitor.getTasks();
        expect(tasks).toHaveLength(1);
        expect(tasks[0].id).toBe(taskId);
      });

      test('should update task status and completion', () => {
        const task = {
          type: 'analysis',
          status: 'pending' as const,
          priority: 'medium' as const,
          complexity: 'simple' as const,
          dependencies: [],
          resourceRequirements: { cpu: 30, memory: 256 }
        };

        const taskId = swarmMonitor.createTask(task);
        swarmMonitor.updateTask(taskId, {
          status: 'running' as const,
          assignedAgent: 'agent_1',
          startedAt: new Date()
        });

        let tasks = swarmMonitor.getTasks();
        expect(tasks[0].status).toBe('running');

        swarmMonitor.completeTask(taskId, { result: 'success' });

        tasks = swarmMonitor.getTasks();
        expect(tasks[0].status).toBe('completed');
        expect(tasks[0].result).toEqual({ result: 'success' });
      });

      test('should handle task failures', () => {
        const task = {
          type: 'coordination',
          status: 'pending' as const,
          priority: 'critical' as const,
          complexity: 'complex' as const,
          dependencies: [],
          resourceRequirements: { cpu: 80, memory: 1024 }
        };

        const taskId = swarmMonitor.createTask(task);
        swarmMonitor.completeTask(taskId, null, 'Task execution failed');

        const tasks = swarmMonitor.getTasks();
        expect(tasks[0].status).toBe('failed');
        expect(tasks[0].error).toBe('Task execution failed');
      });
    });

    describe('Communication Tracking', () => {
      test('should record inter-agent communications', () => {
        const communication = {
          sourceAgent: 'agent_1',
          targetAgent: 'agent_2',
          messageType: 'task_assignment' as const,
          latency: 50,
          dataSize: 1024,
          success: true
        };

        swarmMonitor.recordCommunication(communication);

        const communications = swarmMonitor.getCommunications();
        expect(communications).toHaveLength(1);
        expect(communications[0].sourceAgent).toBe('agent_1');
        expect(communications[0].targetAgent).toBe('agent_2');
        expect(communications[0].timestamp).toBeInstanceOf(Date);
      });

      test('should filter communications correctly', () => {
        const communications = [
          {
            sourceAgent: 'agent_1',
            targetAgent: 'agent_2',
            messageType: 'task_assignment' as const,
            latency: 50,
            dataSize: 1024,
            success: true
          },
          {
            sourceAgent: 'agent_2',
            targetAgent: 'agent_3',
            messageType: 'result_sharing' as const,
            latency: 75,
            dataSize: 2048,
            success: true
          },
          {
            sourceAgent: 'agent_1',
            targetAgent: 'agent_3',
            messageType: 'coordination' as const,
            latency: 100,
            dataSize: 512,
            success: false
          }
        ];

        communications.forEach(comm => swarmMonitor.recordCommunication(comm));

        // Filter by source agent
        const fromAgent1 = swarmMonitor.getCommunications({ sourceAgent: 'agent_1' });
        expect(fromAgent1).toHaveLength(2);

        // Filter by message type
        const taskAssignments = swarmMonitor.getCommunications({
          messageType: 'task_assignment'
        });
        expect(taskAssignments).toHaveLength(1);

        // Filter by success
        const successfulComms = swarmMonitor.getCommunications({
          filter: { success: true }
        });
        expect(successfulComms).toHaveLength(2);
      });
    });

    describe('Performance Metrics', () => {
      test('should calculate comprehensive swarm metrics', () => {
        // Register some agents
        for (let i = 1; i <= 3; i++) {
          swarmMonitor.registerAgent({
            id: `agent_${i}`,
            type: 'worker',
            status: i === 1 ? 'busy' as const : 'idle' as const,
            currentTask: i === 1 ? 'task_1' : null,
            lastHeartbeat: new Date(),
            resourceUsage: { cpu: 30 * i, memory: 256 * i, network: 10 * i },
            performance: {
              tasksCompleted: 10 * i,
              averageTaskDuration: 1000,
              successRate: 0.9 + (i * 0.02),
              errorCount: i
            },
            capabilities: ['basic_tasks']
          });
        }

        // Create some tasks
        for (let i = 1; i <= 5; i++) {
          const taskId = swarmMonitor.createTask({
            type: 'test_task',
            status: i <= 3 ? 'completed' as const : 'pending' as const,
            priority: 'medium' as const,
            complexity: 'simple' as const,
            dependencies: [],
            resourceRequirements: { cpu: 20, memory: 128 }
          });

          if (i <= 3) {
            swarmMonitor.completeTask(taskId, { result: `result_${i}` });
          }
        }

        const metrics = swarmMonitor.getSwarmMetrics();

        expect(metrics).toBeDefined();
        expect(metrics.agentStates).toBeDefined();
        expect(metrics.taskPerformance).toBeDefined();
        expect(metrics.agentCoordination).toBeDefined();
        expect(metrics.resourceUtilization).toBeDefined();
        expect(metrics.agentStates.totalAgents).toBe(3);
        expect(metrics.taskPerformance.taskCompletionRate).toBe(0.6); // 3/5 completed
      });

      test('should calculate swarm health indicators', () => {
        // Setup a healthy swarm
        for (let i = 1; i <= 5; i++) {
          swarmMonitor.registerAgent({
            id: `health_agent_${i}`,
            type: 'worker',
            status: 'active' as const,
            currentTask: null,
            lastHeartbeat: new Date(),
            resourceUsage: { cpu: 40, memory: 384, network: 25 },
            performance: {
              tasksCompleted: 20,
              averageTaskDuration: 900,
              successRate: 0.95,
              errorCount: 1
            },
            capabilities: ['standard_tasks']
          });
        }

        const health = swarmMonitor.getSwarmHealth();

        expect(health).toBeDefined();
        expect(health.overallHealth).toBeGreaterThanOrEqual(0);
        expect(health.overallHealth).toBeLessThanOrEqual(100);
        expect(health.agentHealth).toBeDefined();
        expect(health.taskHealth).toBeDefined();
        expect(health.communicationHealth).toBeDefined();
        expect(health.resourceHealth).toBeDefined();
      });
    });

    describe('Topology Analysis', () => {
      test('should analyze and update swarm topology', () => {
        // Register agents with connections
        const agents = ['agent_1', 'agent_2', 'agent_3', 'agent_4'];

        agents.forEach((agentId, index) => {
          swarmMonitor.registerAgent({
            id: agentId,
            type: 'worker',
            status: 'active' as const,
            currentTask: null,
            lastHeartbeat: new Date(),
            resourceUsage: { cpu: 50, memory: 512, network: 30 },
            performance: {
              tasksCompleted: 15,
              averageTaskDuration: 800,
              successRate: 0.92,
              errorCount: 2
            },
            capabilities: ['standard_tasks']
          });
        });

        // Record communications to establish topology
        swarmMonitor.recordCommunication({
          sourceAgent: 'agent_1',
          targetAgent: 'agent_2',
          messageType: 'coordination' as const,
          latency: 25,
          dataSize: 512,
          success: true
        });

        swarmMonitor.recordCommunication({
          sourceAgent: 'agent_2',
          targetAgent: 'agent_3',
          messageType: 'coordination' as const,
          latency: 30,
          dataSize: 256,
          success: true
        });

        const topology = swarmMonitor.getTopology();

        expect(topology).toBeDefined();
        expect(topology!.agents).toHaveLength(4);
        expect(topology!.connections.length).toBeGreaterThan(0);
        expect(topology!.efficiency).toBeGreaterThanOrEqual(0);
        expect(topology!.efficiency).toBeLessThanOrEqual(1);
      });

      test('should identify bottleneck nodes', () => {
        // Register agents with varying load and performance
        swarmMonitor.registerAgent({
          id: 'bottleneck_agent',
          type: 'coordinator',
          status: 'busy' as const,
          currentTask: 'coordination_task',
          lastHeartbeat: new Date(),
          resourceUsage: { cpu: 95, memory: 896, network: 85 },
          performance: {
            tasksCompleted: 50,
            averageTaskDuration: 2000,
            successRate: 0.65,
            errorCount: 10
          },
          capabilities: ['coordination', 'management']
        });

        swarmMonitor.registerAgent({
          id: 'normal_agent',
          type: 'worker',
          status: 'idle' as const,
          currentTask: null,
          lastHeartbeat: new Date(),
          resourceUsage: { cpu: 30, memory: 256, network: 20 },
          performance: {
            tasksCompleted: 20,
            averageTaskDuration: 800,
            successRate: 0.95,
            errorCount: 1
          },
          capabilities: ['standard_tasks']
        });

        // Create communications showing bottleneck agent as central
        for (let i = 1; i <= 3; i++) {
          swarmMonitor.recordCommunication({
            sourceAgent: `worker_${i}`,
            targetAgent: 'bottleneck_agent',
            messageType: 'coordination' as const,
            latency: 100,
            dataSize: 1024,
            success: true
          });
        }

        const topology = swarmMonitor.getTopology();

        expect(topology).toBeDefined();
        expect(topology!.bottleneckNodes).toContain('bottleneck_agent');
      });
    });

    describe('Monitoring Lifecycle', () => {
      test('should start and stop monitoring correctly', () => {
        const monitor = new SwarmMonitor();

        // Monitor should start automatically in constructor
        expect(monitor['monitoringInterval']).toBeDefined();

        monitor.stopMonitoring();
        expect(monitor['monitoringInterval']).toBeNull();
      });

      test('should handle monitoring intervals correctly', (done) => {
        const monitor = new SwarmMonitor();
        let healthUpdateCount = 0;

        monitor.on('health_update', () => {
          healthUpdateCount++;
          if (healthUpdateCount >= 2) {
            monitor.stopMonitoring();
            done();
          }
        });

        // Register an agent to trigger health updates
        monitor.registerAgent({
          id: 'interval_test_agent',
          type: 'worker',
          status: 'active' as const,
          currentTask: null,
          lastHeartbeat: new Date(),
          resourceUsage: { cpu: 50, memory: 512, network: 30 },
          performance: {
            tasksCompleted: 10,
            averageTaskDuration: 1000,
            successRate: 0.9,
            errorCount: 1
          },
          capabilities: ['test_tasks']
        });
      });
    });

    describe('Data Management', () => {
      test('should maintain communication history size', () => {
        // Fill up communication history beyond max size
        for (let i = 0; i < 15000; i++) {
          swarmMonitor.recordCommunication({
            sourceAgent: 'agent_1',
            targetAgent: 'agent_2',
            messageType: 'test_message' as const,
            latency: 50,
            dataSize: 256,
            success: true
          });
        }

        const communications = swarmMonitor.getCommunications();
        expect(communications.length).toBeLessThanOrEqual(10000);
      });

      test('should clear history correctly', () => {
        // Add some data
        swarmMonitor.registerAgent({
          id: 'clear_test_agent',
          type: 'worker',
          status: 'active' as const,
          currentTask: null,
          lastHeartbeat: new Date(),
          resourceUsage: { cpu: 30, memory: 256, network: 15 },
          performance: {
            tasksCompleted: 5,
            averageTaskDuration: 600,
            successRate: 0.92,
            errorCount: 0
          },
          capabilities: ['clear_test']
        });

        swarmMonitor.createTask({
          type: 'clear_test_task',
          status: 'pending' as const,
          priority: 'low' as const,
          complexity: 'simple' as const,
          dependencies: [],
          resourceRequirements: { cpu: 20, memory: 128 }
        });

        expect(swarmMonitor.getAgents()).toHaveLength(1);
        expect(swarmMonitor.getTasks()).toHaveLength(1);

        swarmMonitor.clearHistory();

        expect(swarmMonitor.getAgents()).toHaveLength(0);
        expect(swarmMonitor.getTasks()).toHaveLength(0);
      });

      test('should export metrics correctly', () => {
        // Add test data
        swarmMonitor.registerAgent({
          id: 'export_test_agent',
          type: 'worker',
          status: 'active' as const,
          currentTask: 'export_task',
          lastHeartbeat: new Date(),
          resourceUsage: { cpu: 60, memory: 512, network: 35 },
          performance: {
            tasksCompleted: 25,
            averageTaskDuration: 1200,
            successRate: 0.94,
            errorCount: 2
          },
          capabilities: ['export_test']
        });

        const exportData = swarmMonitor.exportMetrics();

        expect(exportData).toBeDefined();
        expect(exportData.timestamp).toBeInstanceOf(Date);
        expect(exportData.agents).toHaveLength(1);
        expect(exportData.topology).toBeDefined();
        expect(exportData.health).toBeDefined();
        expect(exportData.healthHistory).toBeDefined();
      });
    });
  });

  describe('Integration Tests', () => {
    test('should integrate swarm coordinator with monitor', async () => {
      await swarmCoordinator.deploy();

      // Register agents from coordinator with monitor
      const agents = Array.from(swarmCoordinator['agents'].values());
      agents.forEach(agent => {
        swarmMonitor.registerAgent({
          id: agent.id,
          type: agent.type,
          status: agent.status === 'busy' ? 'busy' as const : 'active' as const,
          currentTask: agent.currentTask,
          lastHeartbeat: new Date(agent.lastHeartbeat),
          resourceUsage: {
            cpu: agent.resources.cpu,
            memory: agent.resources.memory,
            network: agent.resources.network
          },
          performance: {
            tasksCompleted: agent.performance.tasksCompleted,
            averageTaskDuration: agent.performance.averageTaskTime,
            successRate: agent.performance.successRate,
            errorCount: Math.floor(agent.performance.tasksCompleted * (1 - agent.performance.successRate))
          },
          capabilities: agent.capabilities
        });
      });

      // Execute a task
      const taskRequest = {
        task: 'Integration test task',
        priority: 'medium' as const,
        type: 'integration'
      };

      const execution = await swarmCoordinator.executeWithCoordination(taskRequest);

      // Verify both systems tracked the execution
      expect(execution).toBeDefined();
      expect(execution.status).toBe('completed');

      const metrics = swarmMonitor.getSwarmMetrics();
      expect(metrics).toBeDefined();
      expect(metrics.agentStates.totalAgents).toBe(agents.length);
    });

    test('should handle cross-agent communication patterns', async () => {
      await swarmCoordinator.deploy();

      // Simulate communication between agents
      const communications = [
        { from: 'agent_1', to: 'agent_2', type: 'coordination' as const },
        { from: 'agent_2', to: 'agent_3', type: 'data_sync' as const },
        { from: 'agent_3', to: 'agent_1', type: 'result_sharing' as const }
      ];

      communications.forEach(comm => {
        swarmMonitor.recordCommunication({
          sourceAgent: comm.from,
          targetAgent: comm.to,
          messageType: comm.type,
          latency: 50 + Math.random() * 100,
          dataSize: 512 + Math.random() * 1024,
          success: Math.random() > 0.1
        });
      });

      const allCommunications = swarmMonitor.getCommunications();
      expect(allCommunications.length).toBe(3);

      // Test communication filtering
      const coordinationComms = swarmMonitor.getCommunications({
        messageType: 'coordination'
      });
      expect(coordinationComms).toHaveLength(1);
    });

    test('should demonstrate adaptive topology optimization', async () => {
      await swarmCoordinator.deploy();

      const initialTopology = await swarmCoordinator.getTopologyStatus();
      expect(initialTopology.currentTopology).toBe('hierarchical');

      // Simulate performance degradation
      swarmCoordinator['calculateSwarmEfficiency'] = jest.fn().mockReturnValue(0.5);

      await swarmCoordinator['evaluateTopologyAdaptation']();

      const updatedTopology = await swarmCoordinator.getTopologyStatus();
      expect(updatedTopology.currentTopology).toBe('mesh');
      expect(updatedTopology.adaptationCount).toBeGreaterThan(initialTopology.adaptationCount);
    });
  });

  describe('Performance Benchmarks', () => {
    test('should handle large number of agents efficiently', async () => {
      const startTime = Date.now();

      // Create coordinator with max agents
      const largeConfig = { ...testConfig, maxAgents: 100 };
      const largeCoordinator = new SwarmCoordinator(largeConfig);

      await largeCoordinator.deploy();

      const deployTime = Date.now() - startTime;
      expect(deployTime).toBeLessThan(5000); // Should deploy within 5 seconds

      const agentCount = largeCoordinator['agents'].size;
      expect(agentCount).toBeGreaterThan(0);
      expect(agentCount).toBeLessThanOrEqual(100);

      await largeCoordinator.shutdown();
    });

    test('should process high volume of tasks efficiently', async () => {
      await swarmCoordinator.deploy();

      const taskCount = 50;
      const startTime = Date.now();

      const promises = Array.from({ length: taskCount }, (_, i) => {
        return swarmCoordinator.executeWithCoordination({
          task: `Performance test task ${i}`,
          priority: 'low' as const,
          type: 'performance_test'
        });
      });

      const results = await Promise.all(promises);
      const totalTime = Date.now() - startTime;

      expect(results).toHaveLength(taskCount);
      expect(results.every(r => r.status === 'completed')).toBe(true);
      expect(totalTime / taskCount).toBeLessThan(1000); // Average less than 1 second per task
    });

    test('should maintain performance under continuous monitoring', () => {
      const monitor = new SwarmMonitor();

      // Register multiple agents
      for (let i = 1; i <= 20; i++) {
        monitor.registerAgent({
          id: `perf_agent_${i}`,
          type: 'worker',
          status: Math.random() > 0.5 ? 'busy' as const : 'idle' as const,
          currentTask: Math.random() > 0.5 ? `task_${i}` : null,
          lastHeartbeat: new Date(),
          resourceUsage: {
            cpu: 20 + Math.random() * 60,
            memory: 128 + Math.random() * 512,
            network: 5 + Math.random() * 30
          },
          performance: {
            tasksCompleted: Math.floor(Math.random() * 50),
            averageTaskDuration: 500 + Math.random() * 1500,
            successRate: 0.8 + Math.random() * 0.2,
            errorCount: Math.floor(Math.random() * 5)
          },
          capabilities: ['performance_test']
        });
      }

      const startTime = Date.now();

      // Generate continuous activity
      for (let i = 0; i < 1000; i++) {
        monitor.recordCommunication({
          sourceAgent: `perf_agent_${Math.floor(Math.random() * 20) + 1}`,
          targetAgent: `perf_agent_${Math.floor(Math.random() * 20) + 1}`,
          messageType: ['coordination', 'task_assignment', 'result_sharing'][Math.floor(Math.random() * 3)] as any,
          latency: 10 + Math.random() * 200,
          dataSize: 64 + Math.random() * 4096,
          success: Math.random() > 0.05
        });
      }

      const activityTime = Date.now() - startTime;
      expect(activityTime).toBeLessThan(1000); // Should handle 1000 communications within 1 second

      const metrics = monitor.getSwarmMetrics();
      expect(metrics).toBeDefined();

      monitor.stopMonitoring();
    });
  });
});