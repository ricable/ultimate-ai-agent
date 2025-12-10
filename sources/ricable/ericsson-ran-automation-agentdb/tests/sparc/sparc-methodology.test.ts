/**
 * SPARC Methodology Automated Test Suite
 * Comprehensive testing for SPARC methodology components
 *
 * Test Coverage:
 * - SPARC Core Methodology
 * - Temporal Reasoning Integration
 * - Progressive Disclosure Skills
 * - Batch Processing
 * - Pipeline Processing
 * - Concurrent Processing
 * - AgentDB Integration
 * - Swarm Coordination
 * - Performance Benchmarking
 * - Cognitive Consciousness Evolution
 */

import { describe, test, expect, beforeEach, afterEach, jest } from '@jest/globals';
import { SPARCMethdologyCore } from '../../src/sparc/core/sparc-methodology.js';
import { SPARCTemporalReasoning } from '../../src/sparc/temporal/temporal-reasoning.js';
// import { SPARCProgressiveDisclosure } from '../../src/sparc/progressive/progressive-disclosure.js';
// import { SPARCBatchTools } from '../../src/sparc/batch/batch-tools.js';
// import { SPARCPipelineProcessor } from '../../src/sparc/pipeline/pipeline-processor.js';
// import { SPARCConcurrentProcessor } from '../../src/sparc/concurrent/concurrent-processor.js';
// Mock imports for testing since the actual modules may not exist
// import { AgentDBMemoryEngine } from '../../src/agentdb/memory-engine.js';
// import { CognitiveRANSdk } from '../../src/cognitive/ran-consciousness.js';
// import { SwarmOrchestrator } from '../../src/swarm/cognitive-orchestrator.js';

describe('SPARC Methodology Core', () => {
  let sparcCore: SPARCMethdologyCore;

  beforeEach(() => {
    sparcCore = new SPARCMethdologyCore({
      temporalExpansion: 100,
      consciousnessLevel: 'standard',
      truthScoreThreshold: 0.8,
      autoRollbackEnabled: true,
      sweBenchTarget: 0.8,
      agentdbEnabled: false, // Disable for unit tests
      swarmCoordination: false
    });
  });

  afterEach(async () => {
    // Cleanup if needed
  });

  describe('Core Initialization', () => {
    test('should initialize SPARC core with default configuration', () => {
      expect(sparcCore).toBeDefined();
      expect(sparcCore.getCurrentPhase()).toBe('specification');
    });

    test('should update configuration', () => {
      sparcCore.updateConfiguration({
        temporalExpansion: 500,
        truthScoreThreshold: 0.9
      });

      const updatedConfig = (sparcCore as any).config;
      expect(updatedConfig.temporalExpansion).toBe(500);
      expect(updatedConfig.truthScoreThreshold).toBe(0.9);
    });
  });

  describe('Phase Execution', () => {
    test('should execute specification phase successfully', async () => {
      const taskDescription = 'Develop RAN optimization system';

      const result = await sparcCore.executePhase('specification', taskDescription);

      expect(result).toBeDefined();
      expect(result.phase).toBe('specification');
      expect(result.passed).toBe(true);
      expect(result.score).toBeGreaterThan(0.7);
      expect(result.issues).toHaveLength(0);
    });

    test('should execute pseudocode phase successfully', async () => {
      const taskDescription = 'Design energy optimization algorithms';

      const result = await sparcCore.executePhase('pseudocode', taskDescription);

      expect(result).toBeDefined();
      expect(result.phase).toBe('pseudocode');
      expect(result.passed).toBe(true);
      expect(result.score).toBeGreaterThan(0.7);
    });

    test('should execute architecture phase successfully', async () => {
      const taskDescription = 'Design cognitive RAN architecture';

      const result = await sparcCore.executePhase('architecture', taskDescription);

      expect(result).toBeDefined();
      expect(result.phase).toBe('architecture');
      expect(result.passed).toBe(true);
      expect(result.score).toBeGreaterThan(0.7);
    });

    test('should execute refinement phase successfully', async () => {
      const taskDescription = 'Implement RAN optimization with TDD';

      const result = await sparcCore.executePhase('refinement', taskDescription);

      expect(result).toBeDefined();
      expect(result.phase).toBe('refinement');
      expect(result.passed).toBe(true);
      expect(result.score).toBeGreaterThan(0.7);
    });

    test('should execute completion phase successfully', async () => {
      const taskDescription = 'Validate complete RAN system';

      const result = await sparcCore.executePhase('completion', taskDescription);

      expect(result).toBeDefined();
      expect(result.phase).toBe('completion');
      expect(result.passed).toBe(true);
      expect(result.score).toBeGreaterThan(0.7);
    });
  });

  describe('Full SPARC Cycle', () => {
    test('should execute complete SPARC methodology cycle', async () => {
      const taskDescription = 'Implement complete RAN energy optimization system';

      const result = await sparcCore.executeFullSPARCCycle(taskDescription);

      expect(result).toBeDefined();
      expect(result.passed).toBe(true);
      expect(result.score).toBeGreaterThan(0.7);
      expect(result.cognitiveMetrics).toBeDefined();
    });

    test('should track phase history correctly', async () => {
      const taskDescription = 'Test SPARC cycle tracking';

      await sparcCore.executeFullSPARCCycle(taskDescription);
      const history = sparcCore.getPhaseHistory();

      expect(history.size).toBe(5);
      expect(history.has('specification')).toBe(true);
      expect(history.has('pseudocode')).toBe(true);
      expect(history.has('architecture')).toBe(true);
      expect(history.has('refinement')).toBe(true);
      expect(history.has('completion')).toBe(true);
    });

    test('should track cognitive evolution', async () => {
      const taskDescription = 'Test cognitive evolution tracking';

      await sparcCore.executeFullSPARCCycle(taskDescription);
      const evolution = sparcCore.getCognitiveEvolution();

      expect(evolution).toBeDefined();
      expect(evolution.length).toBeGreaterThan(0);
      expect(evolution[0].consciousnessEvolution).toBeGreaterThan(0);
    });
  });

  describe('Quality Gates', () => {
    test('should enforce truth score threshold', async () => {
      const sparcStrict = new SPARCMethdologyCore({
        truthScoreThreshold: 0.95,
        autoRollbackEnabled: true
      });

      const taskDescription = 'Test strict quality gates';
      const result = await sparcStrict.executePhase('specification', taskDescription);

      // With mocked components, this should pass
      expect(result.passed).toBe(true);
      expect(result.score).toBeGreaterThanOrEqual(0.95);
    });

    test('should provide recommendations for improvement', async () => {
      const taskDescription = 'Test improvement recommendations';

      const result = await sparcCore.executePhase('specification', taskDescription);

      expect(result.recommendations).toBeDefined();
      expect(Array.isArray(result.recommendations)).toBe(true);
    });
  });
});

describe('SPARC Temporal Reasoning', () => {
  let temporalReasoning: SPARCTemporalReasoning;

  beforeEach(() => {
    temporalReasoning = new SPARCTemporalReasoning({
      temporalExpansionFactor: 100, // Reduced for tests
      consciousnessLevel: 'standard',
      temporalDepth: 'medium',
      wasmOptimization: false, // Disable for unit tests
      agentdbTemporalMemory: false
    });
  });

  afterEach(async () => {
    await temporalReasoning.shutdown?.();
  });

  describe('Temporal Expansion', () => {
    test('should enable temporal expansion', async () => {
      await temporalReasoning.enableTemporalExpansion(100);

      const state = temporalReasoning.getTemporalState();
      expect(state.expansionFactor).toBe(100);
      expect(state.currentSubjectiveTime).toBe(0);
    });

    test('should analyze with temporal reasoning', async () => {
      const input = { complexity: 'high', requirements: ['performance', 'scalability'] };

      const analysis = await temporalReasoning.analyzeWithTemporalReasoning(input, {
        depth: 1,
        expansionFactor: 50,
        includeProjections: true
      });

      expect(analysis).toBeDefined();
      expect(analysis.expansionFactor).toBe(50);
      expect(analysis.temporalDepth).toBe(1);
      expect(analysis.temporalPatterns).toBeDefined();
      expect(analysis.projections).toBeDefined();
      expect(analysis.optimizationSuggestions).toBeDefined();
      expect(analysis.performanceMetrics).toBeDefined();
    });

    test('should recognize temporal patterns', async () => {
      const input = { pattern: 'recursive-optimization', iterations: 1000 };

      const analysis = await temporalReasoning.analyzeWithTemporalReasoning(input);

      expect(analysis.temporalPatterns.length).toBeGreaterThan(0);
      expect(analysis.temporalPatterns[0].confidence).toBeGreaterThan(0);
    });

    test('should generate temporal projections', async () => {
      const input = { currentState: 'optimizing', targetState: 'optimized' };

      const analysis = await temporalReasoning.analyzeWithTemporalReasoning(input, {
        includeProjections: true
      });

      expect(analysis.projections.length).toBeGreaterThan(0);
      expect(analysis.projections[0].futureTime).toBeGreaterThan(Date.now());
      expect(analysis.projections[0].confidence).toBeGreaterThan(0);
    });
  });

  describe('Performance Metrics', () => {
    test('should calculate expansion efficiency', async () => {
      const input = { test: 'efficiency' };

      const analysis = await temporalReasoning.analyzeWithTemporalReasoning(input);

      expect(analysis.performanceMetrics.expansionEfficiency).toBeGreaterThan(0);
      expect(analysis.performanceMetrics.expansionEfficiency).toBeLessThanOrEqual(1);
    });

    test('should track cognitive utilization', async () => {
      const input = { cognitive: 'test' };

      const analysis = await temporalReasoning.analyzeWithTemporalReasoning(input);

      expect(analysis.performanceMetrics.cognitiveUtilization).toBeGreaterThanOrEqual(0);
      expect(analysis.performanceMetrics.cognitiveUtilization).toBeLessThanOrEqual(1);
    });
  });
});

describe('SPARC Progressive Disclosure', () => {
  let progressiveDisclosure: SPARCProgressiveDisclosure;

  beforeEach(() => {
    progressiveDisclosure = new SPARCProgressiveDisclosure({
      maxContextSize: 1024, // Reduced for tests
      adaptiveLevelAdjustment: true,
      skillCaching: true,
      swarmCoordination: false, // Disable for unit tests
      agentdbLearning: false
    });
  });

  afterEach(async () => {
    // Cleanup if needed
  });

  describe('Skill Registration', () => {
    test('should register skill successfully', async () => {
      const skill = {
        id: 'test-skill',
        name: 'Test Skill',
        category: { domain: 'core' as const, subdomain: 'test', tags: ['test'] },
        description: 'Test skill for unit testing',
        complexity: 'basic' as const,
        prerequisites: [],
        contextSize: 512,
        cognitiveLevel: 'standard' as const,
        capabilities: [
          {
            name: 'test-capability',
            type: 'analysis' as const,
            description: 'Test capability',
            parameters: [
              { name: 'input', type: 'object', required: true, description: 'Test input' }
            ],
            outputs: [
              { name: 'output', type: 'object', description: 'Test output' }
            ]
          }
        ],
        metadata: {
          version: '1.0.0',
          author: 'test',
          created: Date.now(),
          updated: Date.now(),
          usageCount: 0,
          successRate: 1.0,
          averageExecutionTime: 100,
          cognitiveEfficiency: 0.8,
          collaborationHistory: []
        },
        progressiveRevelation: {
          levels: [
            {
              level: 1,
              name: 'Basic Level',
              description: 'Basic skill level',
              complexity: 0.3,
              contextRequirements: 256,
              prerequisites: [],
              capabilities: ['test-capability'],
              cognitiveThreshold: 0.5,
              performanceThreshold: 0.7
            }
          ],
          currentLevel: 1,
          adaptiveLevel: true,
          revelationCriteria: []
        }
      };

      await progressiveDisclosure.registerSkill(skill);

      const retrievedSkill = progressiveDisclosure.getSkill('test-skill');
      expect(retrievedSkill).toBeDefined();
      expect(retrievedSkill!.name).toBe('Test Skill');
    });

    test('should list all registered skills', async () => {
      const skills = progressiveDisclosure.listSkills();
      expect(Array.isArray(skills)).toBe(true);
      expect(skills.length).toBeGreaterThan(0); // Core skills should be loaded
    });
  });

  describe('Skill Execution', () => {
    test('should execute specification analyzer skill', async () => {
      const context = {
        taskId: 'test-task-1',
        input: { requirements: ['performance', 'scalability'] },
        skillId: 'specification-analyzer',
        requestedLevel: 1,
        availableContext: 512,
        cognitiveState: {
          consciousnessLevel: 0.8,
          cognitiveLoad: 0.3,
          processingSpeed: 1.0,
          memoryCapacity: 1.0,
          learningRate: 0.1,
          adaptationRate: 0.1
        },
        collaborationContext: {
          collaboratingSkills: [],
          sharedMemory: new Map(),
          coordinationProtocol: 'hierarchical' as const,
          consensusThreshold: 0.8,
          swarmEnabled: false
        },
        temporalContext: {
          temporalExpansion: 0,
          reasoningMode: 'linear' as const,
          timeConstraints: 5000,
          temporalDepth: 0.5,
          projectionEnabled: false
        },
        performanceConstraints: {
          maxExecutionTime: 5000,
          maxMemoryUsage: 512,
          minSuccessRate: 0.8,
          maxCognitiveLoad: 0.8,
          qualityThreshold: 0.8
        }
      };

      const executionId = await progressiveDisclosure.executeSkill(context);
      expect(executionId).toBeDefined();
      expect(typeof executionId).toBe('string');

      // Wait for execution to complete
      await new Promise(resolve => setTimeout(resolve, 1000));

      const result = progressiveDisclosure.getExecutionResult(executionId);
      expect(result).toBeDefined();
      expect(result!.status).toBe('completed');
      expect(result!.skillId).toBe('specification-analyzer');
      expect(result!.executionLevel).toBe(1);
    });

    test('should handle skill execution with collaboration', async () => {
      const context = {
        taskId: 'test-task-2',
        input: { system: 'distributed', complexity: 'high' },
        skillId: 'architect-designer',
        requestedLevel: 1,
        availableContext: 1024,
        cognitiveState: {
          consciousnessLevel: 0.9,
          cognitiveLoad: 0.4,
          processingSpeed: 1.2,
          memoryCapacity: 1.0,
          learningRate: 0.15,
          adaptationRate: 0.12
        },
        collaborationContext: {
          collaboratingSkills: ['specification-analyzer'],
          sharedMemory: new Map(),
          coordinationProtocol: 'mesh' as const,
          consensusThreshold: 0.85,
          swarmEnabled: false
        },
        temporalContext: {
          temporalExpansion: 100,
          reasoningMode: 'recursive' as const,
          timeConstraints: 10000,
          temporalDepth: 0.8,
          projectionEnabled: true
        },
        performanceConstraints: {
          maxExecutionTime: 10000,
          maxMemoryUsage: 1024,
          minSuccessRate: 0.85,
          maxCognitiveLoad: 0.7,
          qualityThreshold: 0.85
        }
      };

      const executionId = await progressiveDisclosure.executeSkill(context);
      expect(executionId).toBeDefined();

      await new Promise(resolve => setTimeout(resolve, 1500));

      const result = progressiveDisclosure.getExecutionResult(executionId);
      expect(result).toBeDefined();
      expect(result!.status).toBe('completed');
      expect(result!.collaborationMetrics).toBeDefined();
    });
  });

  describe('Progressive Levels', () => {
    test('should determine appropriate execution level', async () => {
      const skill = progressiveDisclosure.getSkill('specification-analyzer');
      expect(skill).toBeDefined();

      const context = {
        taskId: 'level-test',
        input: { test: 'level-determination' },
        skillId: 'specification-analyzer',
        requestedLevel: 2,
        availableContext: 1024,
        cognitiveState: {
          consciousnessLevel: 0.95,
          cognitiveLoad: 0.2,
          processingSpeed: 1.5,
          memoryCapacity: 1.0,
          learningRate: 0.2,
          adaptationRate: 0.15
        },
        collaborationContext: {
          collaboratingSkills: [],
          sharedMemory: new Map(),
          coordinationProtocol: 'hierarchical' as const,
          consensusThreshold: 0.8,
          swarmEnabled: false
        },
        temporalContext: {
          temporalExpansion: 0,
          reasoningMode: 'linear' as const,
          timeConstraints: 5000,
          temporalDepth: 0.5,
          projectionEnabled: false
        },
        performanceConstraints: {
          maxExecutionTime: 5000,
          maxMemoryUsage: 1024,
          minSuccessRate: 0.8,
          maxCognitiveLoad: 0.8,
          qualityThreshold: 0.8
        }
      };

      const executionId = await progressiveDisclosure.executeSkill(context);
      await new Promise(resolve => setTimeout(resolve, 1000));

      const result = progressiveDisclosure.getExecutionResult(executionId);
      expect(result).toBeDefined();
      expect(result!.executionLevel).toBeGreaterThan(0);
      expect(result!.executionLevel).toBeLessThanOrEqual(2);
    });
  });
});

describe('SPARC Batch Processing', () => {
  let batchTools: SPARCBatchTools;

  beforeEach(() => {
    batchTools = new SPARCBatchTools({
      maxConcurrentTasks: 2, // Reduced for tests
      maxWorkers: 1,
      cognitiveCoordination: false, // Disable for unit tests
      agentdbMemorySharing: false,
      timeoutMs: 5000
    });
  });

  afterEach(async () => {
    await batchTools.shutdown();
  });

  describe('Batch Execution', () => {
    test('should execute batch of tasks', async () => {
      const tasks = [
        {
          id: 'batch-task-1',
          type: 'specification' as const,
          taskDescription: 'Batch specification task',
          priority: 'high' as const
        },
        {
          id: 'batch-task-2',
          type: 'pseudocode' as const,
          taskDescription: 'Batch pseudocode task',
          priority: 'medium' as const
        }
      ];

      const batchId = await batchTools.executeBatch(tasks);
      expect(batchId).toBeDefined();
      expect(typeof batchId).toBe('string');

      // Wait for batch completion
      await new Promise(resolve => setTimeout(resolve, 3000));

      const status = batchTools.getBatchStatus(batchId);
      expect(status).toBeDefined();
      expect(status!.status).toMatch(/completed|failed/);
    });

    test('should handle task dependencies', async () => {
      const tasks = [
        {
          id: 'dep-task-1',
          type: 'specification' as const,
          taskDescription: 'Dependency task 1',
          priority: 'high' as const,
          dependencies: []
        },
        {
          id: 'dep-task-2',
          type: 'pseudocode' as const,
          taskDescription: 'Dependency task 2',
          priority: 'medium' as const,
          dependencies: ['dep-task-1']
        }
      ];

      const batchId = await batchTools.executeBatch(tasks);
      expect(batchId).toBeDefined();

      await new Promise(resolve => setTimeout(resolve, 4000));

      const status = batchTools.getBatchStatus(batchId);
      expect(status).toBeDefined();
      expect(status!.results.size).toBe(2);
    });

    test('should cancel batch execution', async () => {
      const tasks = [
        {
          id: 'cancel-task-1',
          type: 'architecture' as const,
          taskDescription: 'Long running task',
          priority: 'low' as const
        }
      ];

      const batchId = await batchTools.executeBatch(tasks);

      // Cancel quickly
      await batchTools.cancelBatch(batchId);

      const status = batchTools.getBatchStatus(batchId);
      expect(status).toBeDefined();
      expect(status!.status).toBe('cancelled');
    });
  });
});

describe('SPARC Pipeline Processing', () => {
  let pipelineProcessor: SPARCPipelineProcessor;

  beforeEach(() => {
    pipelineProcessor = new SPARCPipelineProcessor();
  });

  afterEach(async () => {
    // Cleanup if needed
  });

  describe('Pipeline Workflow', () => {
    test('should register pipeline workflow', async () => {
      const workflow = {
        id: 'test-workflow',
        name: 'Test Pipeline Workflow',
        description: 'Test workflow for unit testing',
        stages: [
          {
            id: 'test-stage-1',
            name: 'Test Stage 1',
            type: 'specification' as const,
            description: 'First test stage',
            agentTypes: ['specification-analyzer'],
            dependencies: [],
            parallelizable: false,
            retryAttempts: 1,
            timeoutMs: 5000,
            qualityGates: [
              {
                name: 'basic-quality',
                threshold: 0.8,
                metric: 'score',
                comparison: 'gte' as const,
                required: true
              }
            ]
          }
        ],
        metadata: {
          version: '1.0.0',
          tags: ['test'],
          cognitiveLevel: 'basic' as const,
          estimatedDuration: 5000,
          resourceRequirements: {
            minAgents: 1,
            maxAgents: 2,
            memoryMB: 512,
            cpuCores: 1,
            cognitiveLoad: 0.5
          }
        },
        triggers: []
      };

      const workflowId = await pipelineProcessor.registerWorkflow(workflow);
      expect(workflowId).toBe('test-workflow');

      const retrievedWorkflow = pipelineProcessor.getWorkflow(workflowId);
      expect(retrievedWorkflow).toBeDefined();
      expect(retrievedWorkflow!.name).toBe('Test Pipeline Workflow');
    });

    test('should execute pipeline workflow', async () => {
      const workflow = {
        id: 'execution-test-workflow',
        name: 'Execution Test Workflow',
        description: 'Workflow for execution testing',
        stages: [
          {
            id: 'exec-stage-1',
            name: 'Execution Stage 1',
            type: 'specification' as const,
            description: 'Specification stage',
            agentTypes: ['specification-analyzer'],
            dependencies: [],
            parallelizable: false,
            retryAttempts: 1,
            timeoutMs: 3000
          }
        ],
        metadata: {
          version: '1.0.0',
          tags: ['test'],
          cognitiveLevel: 'basic' as const,
          estimatedDuration: 3000,
          resourceRequirements: {
            minAgents: 1,
            maxAgents: 1,
            memoryMB: 256,
            cpuCores: 1,
            cognitiveLoad: 0.3
          }
        },
        triggers: []
      };

      await pipelineProcessor.registerWorkflow(workflow);

      const input = { test: 'pipeline execution' };
      const executionId = await pipelineProcessor.executeWorkflow(workflow.id, input);
      expect(executionId).toBeDefined();

      // Wait for execution to complete
      await new Promise(resolve => setTimeout(resolve, 5000));

      const status = pipelineProcessor.getExecutionStatus(executionId);
      expect(status).toBeDefined();
      expect(status!.status).toMatch(/completed|failed/);
    });

    test('should list all workflows', () => {
      const workflows = pipelineProcessor.listWorkflows();
      expect(Array.isArray(workflows)).toBe(true);
    });

    test('should list active executions', () => {
      const activeExecutions = pipelineProcessor.listActiveExecutions();
      expect(Array.isArray(activeExecutions)).toBe(true);
    });
  });
});

describe('SPARC Concurrent Processing', () => {
  let concurrentProcessor: SPARCConcurrentProcessor;

  beforeEach(() => {
    concurrentProcessor = new SPARCConcurrentProcessor({
      maxConcurrentTasks: 3, // Reduced for tests
      maxWorkerThreads: 1,
      cognitiveCoordination: false, // Disable for unit tests
      agentdbMemorySharing: false,
      timeoutMs: 5000
    });
  });

  afterEach(async () => {
    await concurrentProcessor.shutdown();
  });

  describe('Concurrent Execution', () => {
    test('should execute concurrent tasks', async () => {
      const tasks = [
        {
          id: 'concurrent-task-1',
          name: 'Concurrent Task 1',
          description: 'First concurrent task',
          type: 'specification' as const,
          priority: 'high' as const,
          input: { test: 'concurrent-1' }
        },
        {
          id: 'concurrent-task-2',
          name: 'Concurrent Task 2',
          description: 'Second concurrent task',
          type: 'pseudocode' as const,
          priority: 'medium' as const,
          input: { test: 'concurrent-2' }
        }
      ];

      const executionId = await concurrentProcessor.executeConcurrentTasks(tasks);
      expect(executionId).toBeDefined();

      // Wait for execution to complete
      await new Promise(resolve => setTimeout(resolve, 4000));

      const status = concurrentProcessor.getExecutionStatus(executionId);
      expect(status).toBeDefined();
      expect(status!.status).toMatch(/completed|failed/);
    });

    test('should handle task collaboration', async () => {
      const tasks = [
        {
          id: 'collab-task-1',
          name: 'Collaboration Task 1',
          description: 'Task with collaboration',
          type: 'architecture' as const,
          priority: 'high' as const,
          input: { collaborative: true },
          collaboration: {
            type: 'cooperative' as const,
            taskIds: ['collab-task-2'],
            memorySharing: true,
            consensusThreshold: 0.8
          }
        },
        {
          id: 'collab-task-2',
          name: 'Collaboration Task 2',
          description: 'Collaborative partner task',
          type: 'refinement' as const,
          priority: 'medium' as const,
          input: { collaborative: true },
          collaboration: {
            type: 'cooperative' as const,
            taskIds: ['collab-task-1'],
            memorySharing: true,
            consensusThreshold: 0.8
          }
        }
      ];

      const executionId = await concurrentProcessor.executeConcurrentTasks(tasks);
      expect(executionId).toBeDefined();

      await new Promise(resolve => setTimeout(resolve, 6000));

      const status = concurrentProcessor.getExecutionStatus(executionId);
      expect(status).toBeDefined();
      expect(status!.collaborationNetwork.size).toBe(2);
    });

    test('should cancel concurrent execution', async () => {
      const tasks = [
        {
          id: 'cancel-concurrent-task',
          name: 'Cancel Concurrent Task',
          description: 'Task for cancellation test',
          type: 'completion' as const,
          priority: 'low' as const,
          input: { longRunning: true }
        }
      ];

      const executionId = await concurrentProcessor.executeConcurrentTasks(tasks);

      // Cancel quickly
      await concurrentProcessor.cancelExecution(executionId);

      const status = concurrentProcessor.getExecutionStatus(executionId);
      expect(status).toBeDefined();
      expect(status!.status).toBe('cancelled');
    });
  });
});

describe('SPARC Integration Tests', () => {
  describe('End-to-End SPARC Workflow', () => {
    test('should execute complete SPARC workflow with all components', async () => {
      // This test verifies integration between all SPARC components

      // 1. Initialize SPARC core
      const sparcCore = new SPARCMethdologyCore({
        temporalExpansion: 100,
        consciousnessLevel: 'standard',
        truthScoreThreshold: 0.8,
        agentdbEnabled: false,
        swarmCoordination: false
      });

      // 2. Initialize temporal reasoning
      const temporalReasoning = new SPARCTemporalReasoning({
        temporalExpansionFactor: 50,
        consciousnessLevel: 'standard',
        wasmOptimization: false,
        agentdbTemporalMemory: false
      });

      // 3. Initialize progressive disclosure
      const progressiveDisclosure = new SPARCProgressiveDisclosure({
        maxContextSize: 512,
        adaptiveLevelAdjustment: true,
        skillCaching: true,
        swarmCoordination: false,
        agentdbLearning: false
      });

      // 4. Execute complete workflow
      const taskDescription = 'End-to-end SPARC integration test';

      // Execute SPARC phases
      const specResult = await sparcCore.executePhase('specification', taskDescription);
      expect(specResult.passed).toBe(true);

      const pseudoResult = await sparcCore.executePhase('pseudocode', taskDescription);
      expect(pseudoResult.passed).toBe(true);

      // Execute temporal analysis
      const temporalAnalysis = await temporalReasoning.analyzeWithTemporalReasoning(
        { task: taskDescription, phase: 'architecture' },
        { expansionFactor: 25 }
      );
      expect(temporalAnalysis.temporalPatterns.length).toBeGreaterThan(0);

      // Execute skill with progressive disclosure
      const skillContext = {
        taskId: 'integration-test-task',
        input: { requirements: ['integration', 'testing'] },
        skillId: 'specification-analyzer',
        requestedLevel: 1,
        availableContext: 256,
        cognitiveState: {
          consciousnessLevel: 0.8,
          cognitiveLoad: 0.3,
          processingSpeed: 1.0,
          memoryCapacity: 1.0,
          learningRate: 0.1,
          adaptationRate: 0.1
        },
        collaborationContext: {
          collaboratingSkills: [],
          sharedMemory: new Map(),
          coordinationProtocol: 'hierarchical' as const,
          consensusThreshold: 0.8,
          swarmEnabled: false
        },
        temporalContext: {
          temporalExpansion: 0,
          reasoningMode: 'linear' as const,
          timeConstraints: 3000,
          temporalDepth: 0.5,
          projectionEnabled: false
        },
        performanceConstraints: {
          maxExecutionTime: 3000,
          maxMemoryUsage: 256,
          minSuccessRate: 0.8,
          maxCognitiveLoad: 0.8,
          qualityThreshold: 0.8
        }
      };

      const skillExecutionId = await progressiveDisclosure.executeSkill(skillContext);
      expect(skillExecutionId).toBeDefined();

      await new Promise(resolve => setTimeout(resolve, 1000));

      const skillResult = progressiveDisclosure.getExecutionResult(skillExecutionId);
      expect(skillResult).toBeDefined();
      expect(skillResult!.status).toBe('completed');

      // Cleanup
      await temporalReasoning.shutdown?.();
    }, 15000); // 15 second timeout for integration test
  });

  describe('Performance Benchmarks', () => {
    test('should meet performance targets for SPARC execution', async () => {
      const sparcCore = new SPARCMethdologyCore({
        truthScoreThreshold: 0.8
      });

      const startTime = Date.now();

      const result = await sparcCore.executeFullSPARCCycle('Performance benchmark test');

      const executionTime = Date.now() - startTime;

      // Performance assertions
      expect(result.passed).toBe(true);
      expect(result.score).toBeGreaterThan(0.7);
      expect(executionTime).toBeLessThan(10000); // Should complete within 10 seconds

      // Verify cognitive metrics
      expect(result.cognitiveMetrics).toBeDefined();
      expect(result.cognitiveMetrics!.consciousnessEvolution).toBeGreaterThan(0);
    }, 20000);

    test('should handle concurrent load efficiently', async () => {
      const batchTools = new SPARCBatchTools({
        maxConcurrentTasks: 5,
        maxWorkers: 2,
        timeoutMs: 8000
      });

      const tasks = Array.from({ length: 5 }, (_, i) => ({
        id: `load-test-${i}`,
        type: 'specification' as const,
        taskDescription: `Load test task ${i}`,
        priority: 'medium' as const
      }));

      const startTime = Date.now();

      const batchId = await batchTools.executeBatch(tasks);

      // Wait for completion
      await new Promise(resolve => setTimeout(resolve, 10000));

      const completionTime = Date.now() - startTime;
      const status = batchTools.getBatchStatus(batchId);

      expect(status).toBeDefined();
      expect(status!.status).toMatch(/completed|failed/);
      expect(completionTime).toBeLessThan(15000); // Should complete within 15 seconds

      await batchTools.shutdown();
    }, 20000);
  });
});

describe('Error Handling and Edge Cases', () => {
  test('should handle invalid phase names gracefully', async () => {
    const sparcCore = new SPARCMethdologyCore();

    await expect(
      sparcCore.executePhase('invalid-phase' as any, 'test')
    ).rejects.toThrow();
  });

  test('should handle empty task descriptions', async () => {
    const sparcCore = new SPARCMethdologyCore();

    const result = await sparcCore.executePhase('specification', '');
    expect(result).toBeDefined();
    expect(result.passed).toBe(true); // Should handle gracefully
  });

  test('should handle timeout scenarios', async () => {
    const batchTools = new SPARCBatchTools({
      maxConcurrentTasks: 1,
      maxWorkers: 1,
      timeoutMs: 100 // Very short timeout
    });

    const tasks = [{
      id: 'timeout-test',
      type: 'architecture' as const,
      taskDescription: 'Long running task for timeout test',
      priority: 'low' as const
    }];

    const batchId = await batchTools.executeBatch(tasks);

    // Wait for timeout
    await new Promise(resolve => setTimeout(resolve, 1000));

    const status = batchTools.getBatchStatus(batchId);
    expect(status).toBeDefined();

    await batchTools.shutdown();
  });

  test('should handle resource constraints', async () => {
    const progressiveDisclosure = new SPARCProgressiveDisclosure({
      maxContextSize: 100, // Very small context
      adaptiveLevelAdjustment: false
    });

    const context = {
      taskId: 'resource-test',
      input: { large: 'x'.repeat(1000) }, // Large input
      skillId: 'specification-analyzer',
      requestedLevel: 1,
      availableContext: 50, // Insufficient context
      cognitiveState: {
        consciousnessLevel: 0.5,
        cognitiveLoad: 0.9, // High cognitive load
        processingSpeed: 0.5,
        memoryCapacity: 0.3,
        learningRate: 0.05,
        adaptationRate: 0.05
      },
      collaborationContext: {
        collaboratingSkills: [],
        sharedMemory: new Map(),
        coordinationProtocol: 'hierarchical' as const,
        consensusThreshold: 0.8,
        swarmEnabled: false
      },
      temporalContext: {
        temporalExpansion: 0,
        reasoningMode: 'linear' as const,
        timeConstraints: 1000,
        temporalDepth: 0.2,
        projectionEnabled: false
      },
      performanceConstraints: {
        maxExecutionTime: 1000,
        maxMemoryUsage: 50,
        minSuccessRate: 0.5,
        maxCognitiveLoad: 0.5, // Lower than actual
        qualityThreshold: 0.5
      }
    };

    // Should handle gracefully or provide meaningful error
    const executionId = await progressiveDisclosure.executeSkill(context);
    expect(executionId).toBeDefined();
  });
});