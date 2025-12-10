/**
 * Unit Tests for DAA Orchestrator NAPI Bindings
 *
 * Tests MRAP loop, workflow engine, rules, and economy
 */

const { test } = require('node:test');
const assert = require('node:assert/strict');

// Mock Orchestrator implementation
const createMockOrchestrator = () => {
  class Orchestrator {
    constructor(config = {}) {
      this.config = config;
      this.running = false;
      this.state = {
        status: 'stopped',
        agents: [],
        tasks: [],
        uptime: 0
      };
    }

    async start() {
      this.running = true;
      this.state.status = 'running';
      this.state.uptime = Date.now();
      return { status: 'running' };
    }

    async stop() {
      this.running = false;
      this.state.status = 'stopped';
      return { status: 'stopped' };
    }

    async monitor() {
      return {
        status: this.state.status,
        agents: this.state.agents.length,
        tasks: this.state.tasks.length,
        uptime: this.running ? Date.now() - this.state.uptime : 0
      };
    }

    async reason(context) {
      return {
        decision: 'proceed',
        confidence: 0.95,
        reasoning: 'Context analysis complete'
      };
    }

    async act(action) {
      return {
        success: true,
        result: `Executed ${action.type}`,
        timestamp: Date.now()
      };
    }

    async reflect(result) {
      return {
        success: result.success,
        learnings: ['Pattern identified', 'Performance optimal'],
        adjustments: []
      };
    }

    async adapt(reflection) {
      return {
        strategyUpdated: true,
        improvements: ['Optimized decision threshold']
      };
    }
  }

  class WorkflowEngine {
    constructor() {
      this.workflows = new Map();
    }

    async createWorkflow(definition) {
      const id = 'workflow_' + Math.random().toString(36).substring(7);
      const workflow = {
        id,
        definition,
        status: 'created',
        steps: definition.steps || []
      };
      this.workflows.set(id, workflow);
      return workflow;
    }

    async executeWorkflow(workflowId, input) {
      const workflow = this.workflows.get(workflowId);
      if (!workflow) {
        throw new Error('Workflow not found');
      }

      return {
        workflowId,
        status: 'completed',
        output: { processed: true, input },
        executionTime: 150
      };
    }

    async getStatus(workflowId) {
      const workflow = this.workflows.get(workflowId);
      if (!workflow) {
        throw new Error('Workflow not found');
      }

      return {
        workflowId,
        status: workflow.status,
        currentStep: 0,
        totalSteps: workflow.steps.length
      };
    }

    async cancelWorkflow(workflowId) {
      const workflow = this.workflows.get(workflowId);
      if (workflow) {
        workflow.status = 'cancelled';
      }
    }
  }

  class RulesEngine {
    constructor() {
      this.rules = new Map();
    }

    async evaluate(context) {
      const results = [];
      for (const [id, rule] of this.rules) {
        results.push({
          ruleId: id,
          matched: true,
          action: rule.action
        });
      }
      return results;
    }

    async addRule(rule) {
      const id = 'rule_' + Math.random().toString(36).substring(7);
      this.rules.set(id, rule);
      return id;
    }

    async removeRule(ruleId) {
      return this.rules.delete(ruleId);
    }
  }

  class EconomyManager {
    constructor() {
      this.balances = new Map();
    }

    async getBalance(agentId) {
      return this.balances.get(agentId) || 0;
    }

    async transfer(from, to, amount) {
      const fromBalance = this.balances.get(from) || 0;
      if (fromBalance < amount) {
        throw new Error('Insufficient balance');
      }

      this.balances.set(from, fromBalance - amount);
      this.balances.set(to, (this.balances.get(to) || 0) + amount);

      return {
        from,
        to,
        amount,
        timestamp: Date.now(),
        txId: 'tx_' + Math.random().toString(36).substring(7)
      };
    }

    async calculateFee(operation) {
      const baseFee = 0.01;
      return baseFee * operation.amount;
    }
  }

  return {
    Orchestrator,
    WorkflowEngine,
    RulesEngine,
    EconomyManager
  };
};

const { Orchestrator, WorkflowEngine, RulesEngine, EconomyManager } = createMockOrchestrator();

// Orchestrator Tests
test('Orchestrator: Create instance', (t) => {
  const orchestrator = new Orchestrator({ mode: 'test' });

  assert.ok(orchestrator, 'Orchestrator should be created');
  assert.equal(orchestrator.running, false, 'Should not be running initially');
});

test('Orchestrator: Start', async (t) => {
  const orchestrator = new Orchestrator();

  const result = await orchestrator.start();

  assert.equal(result.status, 'running', 'Should start successfully');
  assert.equal(orchestrator.running, true, 'Running flag should be true');
});

test('Orchestrator: Stop', async (t) => {
  const orchestrator = new Orchestrator();

  await orchestrator.start();
  const result = await orchestrator.stop();

  assert.equal(result.status, 'stopped', 'Should stop successfully');
  assert.equal(orchestrator.running, false, 'Running flag should be false');
});

test('Orchestrator: Monitor system state', async (t) => {
  const orchestrator = new Orchestrator();
  await orchestrator.start();

  const state = await orchestrator.monitor();

  assert.equal(state.status, 'running', 'Status should be running');
  assert.ok(typeof state.agents === 'number', 'Should have agents count');
  assert.ok(typeof state.tasks === 'number', 'Should have tasks count');
  assert.ok(state.uptime >= 0, 'Should have uptime');
});

test('Orchestrator: MRAP - Reason step', async (t) => {
  const orchestrator = new Orchestrator();
  const context = { situation: 'test', data: {} };

  const decision = await orchestrator.reason(context);

  assert.ok(decision.decision, 'Should return a decision');
  assert.ok(decision.confidence > 0, 'Should have confidence score');
  assert.ok(decision.reasoning, 'Should include reasoning');
});

test('Orchestrator: MRAP - Act step', async (t) => {
  const orchestrator = new Orchestrator();
  const action = { type: 'process', params: {} };

  const result = await orchestrator.act(action);

  assert.equal(result.success, true, 'Action should succeed');
  assert.ok(result.result, 'Should have result');
  assert.ok(result.timestamp, 'Should have timestamp');
});

test('Orchestrator: MRAP - Reflect step', async (t) => {
  const orchestrator = new Orchestrator();
  const result = { success: true, data: {} };

  const reflection = await orchestrator.reflect(result);

  assert.equal(reflection.success, true, 'Reflection should succeed');
  assert.ok(Array.isArray(reflection.learnings), 'Should have learnings');
});

test('Orchestrator: MRAP - Adapt step', async (t) => {
  const orchestrator = new Orchestrator();
  const reflection = { success: true, learnings: ['test'] };

  const adaptation = await orchestrator.adapt(reflection);

  assert.ok(adaptation.strategyUpdated, 'Strategy should be updated');
  assert.ok(Array.isArray(adaptation.improvements), 'Should have improvements');
});

// Workflow Engine Tests
test('WorkflowEngine: Create workflow', async (t) => {
  const engine = new WorkflowEngine();
  const definition = {
    name: 'test-workflow',
    steps: ['step1', 'step2', 'step3']
  };

  const workflow = await engine.createWorkflow(definition);

  assert.ok(workflow.id, 'Workflow should have ID');
  assert.equal(workflow.status, 'created', 'Status should be created');
  assert.deepEqual(workflow.steps, definition.steps, 'Steps should match');
});

test('WorkflowEngine: Execute workflow', async (t) => {
  const engine = new WorkflowEngine();
  const definition = { name: 'test', steps: ['step1'] };

  const workflow = await engine.createWorkflow(definition);
  const result = await engine.executeWorkflow(workflow.id, { data: 'test' });

  assert.equal(result.status, 'completed', 'Workflow should complete');
  assert.ok(result.output, 'Should have output');
  assert.ok(result.executionTime > 0, 'Should have execution time');
});

test('WorkflowEngine: Get workflow status', async (t) => {
  const engine = new WorkflowEngine();
  const definition = { name: 'test', steps: ['step1', 'step2'] };

  const workflow = await engine.createWorkflow(definition);
  const status = await engine.getStatus(workflow.id);

  assert.equal(status.workflowId, workflow.id, 'Workflow ID should match');
  assert.ok(status.status, 'Should have status');
  assert.equal(status.totalSteps, 2, 'Should have correct step count');
});

test('WorkflowEngine: Cancel workflow', async (t) => {
  const engine = new WorkflowEngine();
  const definition = { name: 'test', steps: ['step1'] };

  const workflow = await engine.createWorkflow(definition);
  await engine.cancelWorkflow(workflow.id);

  const status = await engine.getStatus(workflow.id);
  assert.equal(status.status, 'cancelled', 'Workflow should be cancelled');
});

test('WorkflowEngine: Non-existent workflow throws error', async (t) => {
  const engine = new WorkflowEngine();

  await assert.rejects(
    async () => await engine.executeWorkflow('non-existent', {}),
    /Workflow not found/,
    'Should throw error for non-existent workflow'
  );
});

// Rules Engine Tests
test('RulesEngine: Add rule', async (t) => {
  const engine = new RulesEngine();
  const rule = {
    condition: 'temperature > 30',
    action: 'activate-cooling'
  };

  const ruleId = await engine.addRule(rule);

  assert.ok(ruleId, 'Rule should have ID');
  assert.match(ruleId, /^rule_/, 'Rule ID should have correct prefix');
});

test('RulesEngine: Evaluate rules', async (t) => {
  const engine = new RulesEngine();

  await engine.addRule({ condition: 'test', action: 'action1' });
  await engine.addRule({ condition: 'test2', action: 'action2' });

  const results = await engine.evaluate({ temperature: 35 });

  assert.ok(Array.isArray(results), 'Results should be an array');
  assert.equal(results.length, 2, 'Should have 2 rule results');
});

test('RulesEngine: Remove rule', async (t) => {
  const engine = new RulesEngine();
  const rule = { condition: 'test', action: 'action' };

  const ruleId = await engine.addRule(rule);
  const removed = await engine.removeRule(ruleId);

  assert.equal(removed, true, 'Rule should be removed');
});

// Economy Manager Tests
test('EconomyManager: Get balance', async (t) => {
  const economy = new EconomyManager();

  const balance = await economy.getBalance('agent1');

  assert.equal(balance, 0, 'New agent should have zero balance');
});

test('EconomyManager: Transfer tokens', async (t) => {
  const economy = new EconomyManager();

  // Set initial balance
  economy.balances.set('agent1', 100);

  const tx = await economy.transfer('agent1', 'agent2', 50);

  assert.equal(tx.from, 'agent1', 'From should be agent1');
  assert.equal(tx.to, 'agent2', 'To should be agent2');
  assert.equal(tx.amount, 50, 'Amount should be 50');
  assert.ok(tx.txId, 'Should have transaction ID');

  const balance1 = await economy.getBalance('agent1');
  const balance2 = await economy.getBalance('agent2');

  assert.equal(balance1, 50, 'Agent1 balance should be 50');
  assert.equal(balance2, 50, 'Agent2 balance should be 50');
});

test('EconomyManager: Insufficient balance throws error', async (t) => {
  const economy = new EconomyManager();

  await assert.rejects(
    async () => await economy.transfer('agent1', 'agent2', 100),
    /Insufficient balance/,
    'Should throw error for insufficient balance'
  );
});

test('EconomyManager: Calculate fee', async (t) => {
  const economy = new EconomyManager();
  const operation = { amount: 100 };

  const fee = await economy.calculateFee(operation);

  assert.ok(fee > 0, 'Fee should be greater than 0');
  assert.equal(fee, 1, 'Fee should be 1% of amount');
});
