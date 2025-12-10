/**
 * Agent Spawning Tests
 * Tests agent creation, validation, and lifecycle management
 */

import { describe, it, expect, beforeEach, jest } from '@jest/globals';
import { v4 as uuidv4 } from 'uuid';

// Mock agent types and their capabilities
const AGENT_TYPES = {
  researcher: {
    capabilities: ['search', 'analyze', 'summarize', 'cite'],
    defaultPrompt: 'You are a research specialist focused on gathering and analyzing information.',
    resourceLimits: { memory: '512MB', cpu: 0.5 }
  },
  coder: {
    capabilities: ['implement', 'debug', 'refactor', 'test'],
    defaultPrompt: 'You are a software developer focused on writing clean, efficient code.',
    resourceLimits: { memory: '1GB', cpu: 1.0 }
  },
  analyst: {
    capabilities: ['evaluate', 'optimize', 'benchmark', 'report'],
    defaultPrompt: 'You are a data analyst focused on performance and metrics.',
    resourceLimits: { memory: '768MB', cpu: 0.75 }
  },
  architect: {
    capabilities: ['design', 'plan', 'structure', 'document'],
    defaultPrompt: 'You are a system architect focused on design and structure.',
    resourceLimits: { memory: '512MB', cpu: 0.5 }
  },
  tester: {
    capabilities: ['test', 'validate', 'verify', 'qa'],
    defaultPrompt: 'You are a QA specialist focused on testing and validation.',
    resourceLimits: { memory: '512MB', cpu: 0.5 }
  },
  coordinator: {
    capabilities: ['orchestrate', 'delegate', 'monitor', 'report'],
    defaultPrompt: 'You are a project coordinator managing tasks and agents.',
    resourceLimits: { memory: '256MB', cpu: 0.25 }
  }
};

// Mock agent factory
class AgentFactory {
  constructor() {
    this.agents = new Map();
    this.resourceUsage = {
      totalMemory: '8GB',
      totalCpu: 8,
      usedMemory: 0,
      usedCpu: 0
    };
  }

  validateAgentType(type) {
    if (!AGENT_TYPES[type]) {
      throw new Error(`Invalid agent type: ${type}. Valid types: ${Object.keys(AGENT_TYPES).join(', ')}`);
    }
  }

  validateResources(type, count = 1) {
    const agentConfig = AGENT_TYPES[type];
    const requiredMemory = this.parseMemory(agentConfig.resourceLimits.memory) * count;
    const requiredCpu = agentConfig.resourceLimits.cpu * count;

    const availableMemory = this.parseMemory(this.resourceUsage.totalMemory) - this.resourceUsage.usedMemory;
    const availableCpu = this.resourceUsage.totalCpu - this.resourceUsage.usedCpu;

    if (requiredMemory > availableMemory) {
      throw new Error(`Insufficient memory: required ${requiredMemory}MB, available ${availableMemory}MB`);
    }

    if (requiredCpu > availableCpu) {
      throw new Error(`Insufficient CPU: required ${requiredCpu}, available ${availableCpu}`);
    }

    return true;
  }

  parseMemory(memStr) {
    const match = memStr.match(/(\d+)(MB|GB)/);
    if (!match) return 0;
    const [, value, unit] = match;
    return unit === 'GB' ? parseInt(value) * 1024 : parseInt(value);
  }

  createAgent(type, name, options = {}) {
    this.validateAgentType(type);
    this.validateResources(type);

    const agentConfig = AGENT_TYPES[type];
    const agent = {
      id: options.id || `agent-${uuidv4().substring(0, 8)}`,
      name: name || `${type}-${Date.now()}`,
      type,
      status: 'initializing',
      capabilities: [...agentConfig.capabilities],
      prompt: options.prompt || agentConfig.defaultPrompt,
      resourceLimits: { ...agentConfig.resourceLimits },
      metadata: {
        createdAt: Date.now(),
        lastActive: Date.now(),
        taskCount: 0,
        successRate: 0,
        ...options.metadata
      }
    };

    // Update resource usage
    this.resourceUsage.usedMemory += this.parseMemory(agentConfig.resourceLimits.memory);
    this.resourceUsage.usedCpu += agentConfig.resourceLimits.cpu;

    // Store agent
    this.agents.set(agent.id, agent);

    // Simulate initialization
    setTimeout(() => {
      agent.status = 'active';
    }, 100);

    return agent;
  }

  spawnBatch(batchConfig) {
    const spawned = [];
    const errors = [];

    for (const [type, count] of Object.entries(batchConfig)) {
      try {
        this.validateAgentType(type);
        this.validateResources(type, count);

        for (let i = 0; i < count; i++) {
          const agent = this.createAgent(type, `${type}-${i + 1}`);
          spawned.push(agent);
        }
      } catch (error) {
        errors.push({ type, count, error: error.message });
      }
    }

    return { spawned, errors };
  }

  getAgent(id) {
    return this.agents.get(id);
  }

  listAgents(filter = {}) {
    let agents = Array.from(this.agents.values());

    if (filter.type) {
      agents = agents.filter(a => a.type === filter.type);
    }

    if (filter.status) {
      agents = agents.filter(a => a.status === filter.status);
    }

    return agents;
  }

  updateAgent(id, updates) {
    const agent = this.agents.get(id);
    if (!agent) {
      throw new Error(`Agent not found: ${id}`);
    }

    Object.assign(agent, updates);
    agent.metadata.lastActive = Date.now();

    return agent;
  }

  removeAgent(id) {
    const agent = this.agents.get(id);
    if (!agent) {
      throw new Error(`Agent not found: ${id}`);
    }

    // Free up resources
    this.resourceUsage.usedMemory -= this.parseMemory(agent.resourceLimits.memory);
    this.resourceUsage.usedCpu -= agent.resourceLimits.cpu;

    this.agents.delete(id);
    return true;
  }
}

describe('Agent Spawning', () => {
  let factory;

  beforeEach(() => {
    factory = new AgentFactory();
  });

  describe('Agent Type Validation', () => {
    it('should validate known agent types', () => {
      expect(() => factory.validateAgentType('researcher')).not.toThrow();
      expect(() => factory.validateAgentType('coder')).not.toThrow();
      expect(() => factory.validateAgentType('analyst')).not.toThrow();
      expect(() => factory.validateAgentType('architect')).not.toThrow();
      expect(() => factory.validateAgentType('tester')).not.toThrow();
      expect(() => factory.validateAgentType('coordinator')).not.toThrow();
    });

    it('should reject unknown agent types', () => {
      expect(() => factory.validateAgentType('invalid')).toThrow(/Invalid agent type/);
      expect(() => factory.validateAgentType('')).toThrow(/Invalid agent type/);
      expect(() => factory.validateAgentType(null)).toThrow(/Invalid agent type/);
    });

    it('should provide list of valid types in error', () => {
      try {
        factory.validateAgentType('unknown');
      } catch (error) {
        expect(error.message).toContain('researcher');
        expect(error.message).toContain('coder');
        expect(error.message).toContain('analyst');
      }
    });
  });

  describe('Resource Management', () => {
    it('should validate available resources', () => {
      expect(() => factory.validateResources('coder', 1)).not.toThrow();
      expect(() => factory.validateResources('researcher', 5)).not.toThrow();
    });

    it('should prevent over-allocation of memory', () => {
      // Spawn agents until memory is nearly full
      for (let i = 0; i < 7; i++) {
        factory.createAgent('coder', `coder-${i}`); // 7GB used
      }

      // Try to spawn another large agent
      expect(() => factory.validateResources('coder', 2)).toThrow(/Insufficient memory/);
    });

    it('should prevent over-allocation of CPU', () => {
      // Spawn agents until CPU is nearly full
      for (let i = 0; i < 8; i++) {
        factory.createAgent('coder', `coder-${i}`); // 8 CPU cores used
      }

      // Try to spawn another agent
      expect(() => factory.validateResources('researcher', 1)).toThrow(/Insufficient CPU/);
    });

    it('should track resource usage accurately', () => {
      factory.createAgent('researcher', 'r1'); // 512MB, 0.5 CPU
      factory.createAgent('coder', 'c1'); // 1GB, 1.0 CPU
      factory.createAgent('analyst', 'a1'); // 768MB, 0.75 CPU

      expect(factory.resourceUsage.usedMemory).toBe(512 + 1024 + 768);
      expect(factory.resourceUsage.usedCpu).toBe(0.5 + 1.0 + 0.75);
    });

    it('should free resources on agent removal', () => {
      const agent = factory.createAgent('coder', 'test');
      const initialMemory = factory.resourceUsage.usedMemory;
      const initialCpu = factory.resourceUsage.usedCpu;

      factory.removeAgent(agent.id);

      expect(factory.resourceUsage.usedMemory).toBe(initialMemory - 1024);
      expect(factory.resourceUsage.usedCpu).toBe(initialCpu - 1.0);
    });
  });

  describe('Single Agent Creation', () => {
    it('should create agent with auto-generated ID', () => {
      const agent = factory.createAgent('researcher', 'TestBot');
      
      expect(agent.id).toMatch(/^agent-[a-z0-9]{8}$/);
      expect(agent.name).toBe('TestBot');
      expect(agent.type).toBe('researcher');
    });

    it('should create agent with custom ID', () => {
      const agent = factory.createAgent('coder', 'CustomBot', { id: 'custom-001' });
      
      expect(agent.id).toBe('custom-001');
    });

    it('should assign default capabilities', () => {
      const researcher = factory.createAgent('researcher', 'R1');
      expect(researcher.capabilities).toContain('search');
      expect(researcher.capabilities).toContain('analyze');

      const coder = factory.createAgent('coder', 'C1');
      expect(coder.capabilities).toContain('implement');
      expect(coder.capabilities).toContain('debug');
    });

    it('should set initial status to initializing', () => {
      const agent = factory.createAgent('analyst', 'A1');
      expect(agent.status).toBe('initializing');
    });

    it('should transition to active status', (done) => {
      const agent = factory.createAgent('tester', 'T1');
      
      setTimeout(() => {
        expect(agent.status).toBe('active');
        done();
      }, 150);
    });

    it('should include metadata', () => {
      const agent = factory.createAgent('coordinator', 'C1', {
        metadata: { project: 'test-project' }
      });

      expect(agent.metadata.createdAt).toBeDefined();
      expect(agent.metadata.lastActive).toBeDefined();
      expect(agent.metadata.taskCount).toBe(0);
      expect(agent.metadata.successRate).toBe(0);
      expect(agent.metadata.project).toBe('test-project');
    });
  });

  describe('Batch Spawning', () => {
    it('should spawn multiple agents of same type', () => {
      const result = factory.spawnBatch({ researcher: 3 });
      
      expect(result.spawned).toHaveLength(3);
      expect(result.errors).toHaveLength(0);
      expect(result.spawned.every(a => a.type === 'researcher')).toBe(true);
    });

    it('should spawn mixed agent types', () => {
      const result = factory.spawnBatch({
        researcher: 2,
        coder: 3,
        analyst: 1
      });

      expect(result.spawned).toHaveLength(6);
      expect(result.spawned.filter(a => a.type === 'researcher')).toHaveLength(2);
      expect(result.spawned.filter(a => a.type === 'coder')).toHaveLength(3);
      expect(result.spawned.filter(a => a.type === 'analyst')).toHaveLength(1);
    });

    it('should handle partial failures', () => {
      // Fill up most resources
      for (let i = 0; i < 7; i++) {
        factory.createAgent('coder', `existing-${i}`);
      }

      const result = factory.spawnBatch({
        researcher: 2, // Should succeed
        coder: 3 // Should fail due to resources
      });

      expect(result.spawned.length).toBeGreaterThan(0);
      expect(result.errors.length).toBeGreaterThan(0);
      expect(result.errors[0].type).toBe('coder');
      expect(result.errors[0].error).toContain('Insufficient');
    });

    it('should validate all types before spawning', () => {
      const result = factory.spawnBatch({
        researcher: 1,
        invalidType: 2,
        coder: 1
      });

      expect(result.errors).toHaveLength(1);
      expect(result.errors[0].type).toBe('invalidType');
      expect(result.errors[0].error).toContain('Invalid agent type');
    });
  });

  describe('Agent Lifecycle', () => {
    it('should update agent properties', () => {
      const agent = factory.createAgent('researcher', 'R1');
      const updated = factory.updateAgent(agent.id, {
        status: 'busy',
        metadata: { currentTask: 'research-001' }
      });

      expect(updated.status).toBe('busy');
      expect(updated.metadata.currentTask).toBe('research-001');
      expect(updated.metadata.lastActive).toBeGreaterThan(agent.metadata.createdAt);
    });

    it('should list agents with filters', () => {
      factory.createAgent('researcher', 'R1');
      factory.createAgent('researcher', 'R2');
      factory.createAgent('coder', 'C1');
      factory.createAgent('coder', 'C2');

      const allAgents = factory.listAgents();
      expect(allAgents).toHaveLength(4);

      const researchers = factory.listAgents({ type: 'researcher' });
      expect(researchers).toHaveLength(2);

      const coders = factory.listAgents({ type: 'coder' });
      expect(coders).toHaveLength(2);
    });

    it('should remove agents', () => {
      const agent = factory.createAgent('analyst', 'A1');
      expect(factory.getAgent(agent.id)).toBeDefined();

      factory.removeAgent(agent.id);
      expect(factory.getAgent(agent.id)).toBeUndefined();
    });

    it('should handle errors gracefully', () => {
      expect(() => factory.updateAgent('non-existent', {})).toThrow(/Agent not found/);
      expect(() => factory.removeAgent('non-existent')).toThrow(/Agent not found/);
    });
  });

  describe('Agent Capabilities', () => {
    it('should assign type-specific capabilities', () => {
      const agents = [
        factory.createAgent('researcher', 'R1'),
        factory.createAgent('coder', 'C1'),
        factory.createAgent('analyst', 'A1'),
        factory.createAgent('architect', 'AR1'),
        factory.createAgent('tester', 'T1'),
        factory.createAgent('coordinator', 'CO1')
      ];

      expect(agents[0].capabilities).toEqual(['search', 'analyze', 'summarize', 'cite']);
      expect(agents[1].capabilities).toEqual(['implement', 'debug', 'refactor', 'test']);
      expect(agents[2].capabilities).toEqual(['evaluate', 'optimize', 'benchmark', 'report']);
      expect(agents[3].capabilities).toEqual(['design', 'plan', 'structure', 'document']);
      expect(agents[4].capabilities).toEqual(['test', 'validate', 'verify', 'qa']);
      expect(agents[5].capabilities).toEqual(['orchestrate', 'delegate', 'monitor', 'report']);
    });

    it('should allow custom prompts', () => {
      const agent = factory.createAgent('researcher', 'R1', {
        prompt: 'You are a specialized ML researcher focusing on neural networks.'
      });

      expect(agent.prompt).toContain('ML researcher');
      expect(agent.prompt).toContain('neural networks');
    });

    it('should preserve default prompts', () => {
      const agent = factory.createAgent('coder', 'C1');
      expect(agent.prompt).toBe(AGENT_TYPES.coder.defaultPrompt);
    });
  });

  describe('Agent Naming', () => {
    it('should auto-generate names if not provided', () => {
      const agent = factory.createAgent('researcher');
      expect(agent.name).toMatch(/^researcher-\d+$/);
    });

    it('should use provided names', () => {
      const agent = factory.createAgent('coder', 'SuperCoder');
      expect(agent.name).toBe('SuperCoder');
    });

    it('should generate unique names in batch spawn', () => {
      const result = factory.spawnBatch({ analyst: 5 });
      const names = result.spawned.map(a => a.name);
      const uniqueNames = new Set(names);
      
      expect(uniqueNames.size).toBe(5);
      expect(names.every(n => n.match(/^analyst-\d+$/))).toBe(true);
    });
  });
});